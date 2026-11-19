# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call-qualified atomic effects share ordinary Buffer transfer geometry."""

import pypto.language as pl
import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto.debug import torch_codegen
from pypto.pypto_core import ir as core_ir

from .sync_closed_command_oracle import check_closed_commands
from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import validate
from .test_select_buffer_ir_functions import _PLANNERS, _compile_dynamic_state_native, _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def atomic_fixture(count=2, visits=2):
    class Atomic(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.store").name:
                return ir.Call(op.op, op.args, {"atomic": 1}, op.type, op.span)
            return super().visit_call(op)

    return Atomic().visit_function(staged_fixture(count, stages=1, visits=visits))


def atomic_store_call(
    *,
    space=ir.MemorySpace.Acc,
    source_dtype=ir.DataType.FP32,
    destination_dtype=ir.DataType.FP32,
    atomic=1,
    extra=None,
):
    ty = ir.BufferType(
        [16, 16],
        source_dtype,
        space,
        blayout=ir.TileLayout.col_major if space == ir.MemorySpace.Acc else ir.TileLayout.row_major,
        slayout=ir.TileLayout.row_major if space == ir.MemorySpace.Acc else ir.TileLayout.none_box,
        fractal=1024 if space == ir.MemorySpace.Acc else 512,
    )
    source = ir.Var("source", ty, SPAN)
    output = ir.Var("output", ir.TensorType([16, 32], destination_dtype), SPAN)

    def tup(values):
        return ir.MakeTuple([ir.ConstInt(v, ir.DataType.INDEX, SPAN) for v in values], SPAN)

    kwargs = {"atomic": atomic}
    if extra is not None:
        kwargs[extra] = 1
    return core_ir._create_internal_op_call(
        "buffer.store", [source, tup([0, 0]), tup([16, 16]), output], kwargs, SPAN
    )


def test_conditional_buffer_effect_needs_the_actual_call():
    with pytest.raises(ValueError, match="actual call kwargs"):
        ir.get_op_buffer_arg_effect("buffer.store", 3)
    for kwargs, mode in [
        ({}, ir.BufferAccess.Write),
        ({"atomic": 0}, ir.BufferAccess.Write),
        ({"atomic": 1}, ir.BufferAccess.ReadWrite),
    ]:
        effect = ir.get_op_buffer_arg_effect("buffer.store", 3, kwargs)
        assert (effect.data, effect.metadata, effect.non_memory) == (mode, ir.BufferAccess.Read, False)
    for kwargs, error, message in (
        ({"atomic": 2}, ValueError, "None or Add"),
        ({"atomic": True}, TypeError, "incompatible type"),
        ({"unrecognized": 1}, ValueError, "Unknown kwarg"),
    ):
        with pytest.raises(error, match=message):
            ir.get_op_buffer_arg_effect("buffer.store", 3, kwargs)
    # Other operand classifications and call-independent APIs remain useful.
    assert ir.get_op_buffer_arg_effect("buffer.store", 0).data == ir.BufferAccess.Read
    assert ir.get_op_buffer_arg_effect("buffer.store", 1).non_memory
    assert ir.get_op_buffer_arg_effect("buffer.store", 2).non_memory


def test_atomic_native_contract_retains_gm_read_modify_write():
    call = atomic_store_call()
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)
    operation = testing.describe_buffer_sync_operation(call, ir.FunctionType.AIC)
    assert operation["pipe"] == ir.PipeType.FIX
    assert [e["data"] for e in operation["arguments"]] == ["read", "none", "none", "read_write"]
    assert operation["implicit_effects"] == []
    with pytest.raises(ValueError, match="qualified A2/A3"):
        testing.describe_buffer_sync_operation(call, ir.FunctionType.AIC, "unknown")


@pytest.mark.parametrize("atomic", [False, True])
def test_outlining_uses_call_qualified_buffer_data_effects(atomic):
    # Exercise the existing generic scope interface with typed Buffer commands.
    # This isolated pass test does not move outlining into the late production
    # pipeline or claim that outlining supplies physical placement evidence.
    child = atomic_fixture(1, 1) if atomic else staged_fixture(1, stages=1, visits=1)
    assert isinstance(child.body, ir.SeqStmts)
    scope = ir.InCoreScopeStmt(
        name_hint="writeback",
        body=ir.SeqStmts(child.body.stmts[:-1], SPAN),
        span=SPAN,
    )
    root = ir.Function(
        "main",
        list(zip(child.params, child.param_directions)),
        child.return_types,
        ir.SeqStmts([scope, ir.ReturnStmt([child.params[-1]], SPAN)], SPAN),
        SPAN,
        ir.FunctionType.Orchestration,
    )
    with passes.PassContext([], passes.VerificationLevel.NONE):
        result = passes.outline_incore_scopes()(ir.Program([root], "OutlineWriteback", SPAN))
    outlined = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.InCore)
    directions = {
        param.name_hint: direction for param, direction in zip(outlined.params, outlined.param_directions)
    }
    assert directions[child.params[-1].name_hint] == (
        ir.ParamDirection.InOut if atomic else ir.ParamDirection.Out
    )


@pytest.mark.parametrize("invalid", ["vec", "dtype", "mode", "phase", "cache", "pre_quant", "pre_relu"])
def test_atomic_store_does_not_broaden_unqualified_native_variants(invalid):
    with pytest.raises(ValueError):
        atomic_store_call(
            space=ir.MemorySpace.Vec if invalid == "vec" else ir.MemorySpace.Acc,
            destination_dtype=ir.DataType.BF16 if invalid == "dtype" else ir.DataType.FP32,
            atomic=2 if invalid == "mode" else 1,
            extra="st_phase"
            if invalid == "phase"
            else invalid
            if invalid in ("cache", "pre_quant", "pre_relu")
            else None,
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("initialize", [False, True])
def test_atomic_store_default_pipeline_torch_and_native_cpp(planner, initialize, tmp_path):
    initial = "out = pl.store(product, [0, 0], out)" if initialize else ""
    program = pl.parse_program(f"""
@pl.program
class AtomicWriteback:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.BF16], b: pl.Tensor[[32, 16], pl.BF16],
               out: pl.InOut[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        left = pl.load(a, [0, 0], [16, 32])
        right = pl.load(b, [0, 0], [32, 16])
        product = pl.matmul(left, right)
        {initial}
        out = pl.store(product, [0, 0], out, atomic=pl.AtomicType.Add)
        return pl.store(product, [0, 16], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    left = torch.linspace(-1, 1, 512).reshape(16, 32).to(torch.bfloat16)
    right = torch.linspace(0.25, 2, 512).reshape(32, 16).to(torch.bfloat16)
    product = left.float() @ right.float()
    output = torch.full((16, 32), 7.0)
    namespace["kernel"](left, right, output)
    expected = torch.cat((2 * product if initialize else product + 7, product), dim=1)
    torch.testing.assert_close(output, expected)
    pto = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    stores = [line for line in pto.splitlines() if "pto.tstore " in line]
    assert len(stores) == (3 if initialize else 2)
    assert sum("atomic_add" in line for line in stores) == 1
    assert "atomic_add" not in stores[-1], "atomic mode leaked into the following ordinary transfer"
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    if planner != passes.MemoryPlanner.PTOAS:
        validate(_kernel(lowered), testing.analyze_placed_sync(_kernel(lowered), True))


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("visits", [0, 1, 2])
def test_atomic_retained_accesses_covers_queries_and_selectors(count, visits):
    function = atomic_fixture(count, visits)
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("overlap", [False, True])
def test_atomic_public_conservative_commands_cover_reuse_and_aliases(count, overlap):
    original = atomic_fixture(count, visits=2)
    before = ir.serialize(original)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(ir.Program([original], "Atomic", SPAN))
    logical = next(iter(logical_program.functions.values()))
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    physical = next(iter(physical_program.functions.values()))
    assert ir.serialize(original) == before
    check_closed_commands(
        logical,
        physical,
        {},
        list(range(6)),
        tensor_bases={0: 0, 1: 1 << 20, 2: 0 if overlap else 2 << 20},
        compare_commands=False,
        hardware_source=original,
    )


def test_atomic_compact_state_does_not_expand_trip_counts():
    sizes = []
    for count in (3, 31, 1023):
        result = testing.analyze_placed_sync(atomic_fixture(count), True).describe()
        sizes.append((result["cells"], tuple(len(r["demands"]) for r in result["regions"])))
    assert len(set(sizes)) == 1
