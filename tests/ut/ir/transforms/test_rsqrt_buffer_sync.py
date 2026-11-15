# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Default FP32 reciprocal square root uses the existing unary Buffer contract."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes, testing
from pypto.debug import torch_codegen
from pypto.pypto_core import ir as core_ir

from .sync_closed_command_oracle import check_closed_commands
from .sync_matmul_oracle import validate
from .test_physical_sync_realization import check
from .test_select_buffer_ir_functions import (
    _PLANNERS,
    _compile_dynamic_state_native,
    _kernel,
    _lower,
)
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("explicit_default", [False, True])
def test_default_rsqrt_roundtrip_torch_and_native_cpp(planner, explicit_default, tmp_path):
    class Default(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("tile.rsqrt").name and explicit_default:
                return ir.Call(op.op, op.args, {"high_precision": False}, op.type, op.span)
            return super().visit_call(op)

    program = pl.parse_program("""
@pl.program
class DefaultRsqrt:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32])
        result = pl.tile.rsqrt(value)
        return pl.store(result, [0, 0], out)
""")
    lowered = _lower(Default().visit_program(program), planner)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    positive = torch.linspace(0.25, 4, 512).reshape(16, 32)
    output = torch.empty_like(positive)
    namespace["kernel"](positive, output)
    torch.testing.assert_close(output, torch.rsqrt(positive))
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    if planner != passes.MemoryPlanner.PTOAS:
        validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("planner", _PLANNERS)
def test_explicit_workspace_rsqrt_is_not_silently_reduced_to_default(planner):
    program = pl.parse_program("""
@pl.program
class Workspace:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32])
        scratch = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.tile.rsqrt(value, scratch, high_precision=True)
        return pl.store(result, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_rsqrt_uses_existing_runtime_valid_guards(planner, tmp_path):
    program = pl.parse_program("""
@pl.program
class RuntimeValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], rows: pl.Scalar[pl.INDEX],
               cols: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32], valid_shape=[rows, cols])
        result = pl.tile.rsqrt(value)
        return pl.store(result, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.linspace(0.25, 4, 512).reshape(16, 32)
    for rows, cols in ((0, 0), (0, 32), (16, 0), (1, 1), (16, 32), (7, 13), (0, 0)):
        out = torch.full_like(x, -777)
        namespace["kernel"](x, rows, cols, out)
        expected = torch.full_like(x, -777)
        expected[:rows, :cols] = torch.rsqrt(x[:rows, :cols])
        torch.testing.assert_close(out, expected)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


def explicit_rsqrt(*, dtype=ir.DataType.FP32, column=False, same=False, extra=False, kwargs=None):
    ty = ir.BufferType(
        [8, 1] if column else [2, 8],
        dtype,
        ir.MemorySpace.Vec,
        blayout=ir.TileLayout.col_major if column else ir.TileLayout.row_major,
    )
    source, destination = ir.Var("source", ty, SPAN), ir.Var("destination", ty, SPAN)
    args = [source, source if same else destination]
    if extra:
        args.insert(1, ir.Var("scratch", ty, SPAN))
    return core_ir._create_internal_op_call("buffer.rsqrt", args, kwargs or {}, SPAN)


def test_default_rsqrt_native_effects_are_explicit_without_scratch_or_flags():
    call = explicit_rsqrt(kwargs={"high_precision": False})
    result = testing.describe_buffer_sync_operation(call, ir.FunctionType.AIV)
    assert result["pipe"] == ir.PipeType.V
    assert [item["data"] for item in result["arguments"]] == ["read", "write"]
    assert result["implicit_effects"] == []
    with pytest.raises(ValueError, match="qualified native implementation"):
        testing.describe_buffer_sync_operation(call, ir.FunctionType.AIV, "unknown")
    # Execution resource qualification exercises the selected factory-owned
    # signal-free fact; the empty catalog lookup is not its justification.
    physical, _, _, _ = check(vec_fixture(operation="buffer.rsqrt"), list(range(6)))
    assert physical["native_reserved_ids"] == []


@pytest.mark.parametrize("invalid", ["dtype", "column", "same", "extra", "precision", "unknown"])
def test_unqualified_rsqrt_forms_remain_refusals(invalid):
    with pytest.raises(ValueError):
        explicit_rsqrt(
            dtype=ir.DataType.FP16 if invalid == "dtype" else ir.DataType.FP32,
            column=invalid == "column",
            same=invalid == "same",
            extra=invalid == "extra",
            kwargs={"high_precision": True}
            if invalid == "precision"
            else {"unqualified": True}
            if invalid == "unknown"
            else {},
        )


def test_distinct_rsqrt_handles_do_not_prove_physical_disjointness():
    class Overlap(ir.IRMutator):
        def visit_call(self, op):
            if (
                op.op.name == ir.get_op("buffer.alloc").name
                and isinstance(op.args[1], ir.ConstInt)
                and op.args[1].value == 2048
            ):
                return ir.Call(
                    op.op, [op.args[0], ir.ConstInt(32, ir.DataType.INDEX, op.span)], op.type, op.span
                )
            return op

    original = Overlap().visit_function(vec_fixture(operation="buffer.rsqrt"))
    with pytest.raises(ValueError, match="overlap|disjoint"):
        testing.analyze_placed_sync(original, True)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_rsqrt_exact_covers_and_conservative_actual_commands(count):
    original = vec_fixture(count, visits=2, operation="buffer.rsqrt")
    validate(original, testing.analyze_placed_sync(original, True))
    if count:
        # This binary-derived fixture retains an unused second load. Its
        # inter-visit MTE2 WAW requires a same-DMA completion mechanism that
        # direct physical lowering does not provide. Keep that precise refusal
        # separate from the public conservative protocol checked below.
        with pytest.raises(ValueError, match="Unavailable same-pipe barrier mechanism"):
            check(original, list(range(6)))
    else:
        check(original, list(range(6)))
    program = ir.Program([original], "DefaultRsqrt", original.span)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = next(iter(passes.insert_sync()(program).functions.values()))
    with passes.PassContext([]):
        physical = next(
            iter(passes.lower_sync()(ir.Program([logical], "Logical", original.span)).functions.values())
        )
    check_closed_commands(logical, physical, {}, list(range(6)), compare_commands=False)


def test_rsqrt_repeated_representation_does_not_expand_trip_counts():
    sizes = []
    for count in (3, 31, 1023):
        description = testing.analyze_placed_sync(
            vec_fixture(count, operation="buffer.rsqrt"), True
        ).describe()
        sizes.append((description["cells"], tuple(len(r["demands"]) for r in description["regions"])))
    assert len(set(sizes)) == 1
