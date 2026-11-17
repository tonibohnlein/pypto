# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original valid-axis identities survive placed contiguous reinterpretations."""

import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto import language as pl
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import PassManager

from .test_runtime_valid_broadcast import check_runtime_commands, lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


def axis_program(reduction="row_sum", visits=3, replay=2, mixed=False, varying=False):
    operation = "pl.add(pl.tile.full([1,8], value=0.0, dtype=pl.FP32), row)" if mixed else "pl.add(row, 1.0)"
    active = "pl.min(rows, visit % 8)" if varying else "rows"
    return pl.parse_program(f"""
@pl.program
class RuntimeAxis:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,16], pl.FP32],
               rows: pl.Scalar[pl.INDEX], choice: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[8,16], pl.FP32]]) -> pl.Tensor[[8,16], pl.FP32]:
        for outer in pl.range({replay}):
            for visit in pl.range({visits}):
                if visit % 2 == choice:
                    active = {active}
                    source = pl.load(x, [0,0], [8,16], valid_shape=[active,16])
                    workspace = pl.create_tile([8,16], dtype=pl.FP32)
                    column = pl.{reduction}(pl.mul(source, source), workspace)
                    row = pl.reshape(column, [1,8])
                    shifted = {operation}
                    carrier = pl.reshape(shifted, [8,1])
                    result = pl.row_expand_mul(source, carrier)
                    out = pl.store(result, [0,0], out)
        return out
""")


@pytest.mark.parametrize("reduction", ["row_sum", "row_max"])
@pytest.mark.parametrize("visits,replay", [(1, 1), (3, 2)])
def test_runtime_axis_metadata_bytes_matching_and_reuse(reduction, visits, replay, tmp_path):
    program, original = lower(axis_program(reduction, visits, replay))
    summary = testing.analyze_conservative_sync(original)
    assert not summary["exact_storage_selectors"]
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 8
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [p.get_function("kernel") for p in [logical_program, physical_program]]
    assert logical is not None and physical is not None
    for rows in [0, 1, 7, 8]:
        for choice in [0, 1]:
            output = torch.full_like(x, -777)
            namespace["kernel"](x, rows, choice, output)
            expected = torch.full_like(x, -777)
            if visits > choice:
                squares = x[:rows] * x[:rows]
                reduced = (
                    squares.sum(dim=1, keepdim=True)
                    if reduction == "row_sum"
                    else squares.amax(dim=1, keepdim=True)
                )
                expected[:rows] = x[:rows] * (reduced + 1)
            torch.testing.assert_close(output, expected)
            check_runtime_commands(logical, physical, {1: rows, 2: choice})
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "axis.pto", tmp_path / "axis.cpp"
        pto.write_text(codegen.PTOCodegen().generate(physical_program))
        _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2"])
        assert cpp.is_file()


def test_mixed_full_and_partial_add_keeps_original_admission_refusal():
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        program = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
            axis_program(mixed=True)
        )
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert "buffer_admission_property" in function.attrs


def test_runtime_axis_retained_size_does_not_unfold_trips():
    summaries = []
    for count in [3, 1023, 1000000]:
        _, function = lower(axis_program(visits=count))
        summaries.append(testing.analyze_conservative_sync(function))
    assert all(summary["cost"] == summaries[0]["cost"] for summary in summaries)
    assert all(len(summary["sites"]) == len(summaries[0]["sites"]) for summary in summaries)


def test_runtime_axis_requires_placed_native_metadata_objects():
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PTOAS):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        program = manager.run_passes(axis_program())
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert function.attrs["buffer_ir_selected"] is False


def test_reshape_requested_shape_must_match_the_declared_axis_map():
    class ForgeShape(ir.IRMutator):
        changed = False

        def visit_call(self, op):
            result = super().visit_call(op)
            assert isinstance(result, ir.Call)
            if self.changed or result.op.name != ir.get_op("tile.reshape").name:
                return result
            self.changed = True
            shape = ir.MakeTuple(
                [ir.ConstInt(n, ir.DataType.INDEX, result.span) for n in [2, 4]], result.span
            )
            return ir.Call(
                result.op,
                [result.args[0], shape],
                dict(result.kwargs),
                dict(result.attrs),
                result.type,
                result.span,
            )

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = axis_program()
        for transform in manager.passes[:boundary]:
            program = transform(program)
        mutation = ForgeShape()
        forged = mutation.visit_program(program)
        assert mutation.changed
        original = ir.serialize(forged)
        selected = passes.select_buffer_ir_functions()(forged)
    assert ir.serialize(forged) == original
    function = selected.get_function("kernel")
    assert function is not None and function.attrs["buffer_ir_selected"] is False


def test_runtime_axis_actual_metadata_changes_across_reset_visits():
    program, _ = lower(axis_program(varying=True))
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [p.get_function("kernel") for p in [logical_program, physical_program]]
    assert logical is not None and physical is not None
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 8
    for rows in [0, 1, 7, 8]:
        for choice in [0, 1]:
            output = torch.full_like(x, -777)
            namespace["kernel"](x, rows, choice, output)
            expected = torch.full_like(x, -777)
            for _ in range(2):
                for visit in range(3):
                    if visit % 2 == choice:
                        active = min(rows, visit % 8)
                        expected[:active] = x[:active] * ((x[:active] ** 2).sum(dim=1, keepdim=True) + 1)
            torch.testing.assert_close(output, expected)
            check_runtime_commands(logical, physical, {1: rows, 2: choice})
