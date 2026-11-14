# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer loop admission must prove positivity in the original scalar width."""

import pytest
from pypto import DataType, ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

from tests.ut.ir.transforms.test_select_buffer_ir_functions import _control_flow_program, _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _step_case(kind, span, buffer):
    def literal(value, dtype=DataType.INDEX):
        return ir.ConstInt(value, dtype, span)

    name = "buffer.get_block_num" if buffer else "tile.get_block_num"
    query = ir.Call(ir.get_op(name), [], ir.ScalarType(DataType.INDEX), span)
    count = ir.Var("native_launch_count", ir.ScalarType(DataType.INDEX), span)
    definitions = [ir.AssignStmt(count, query, span)]
    if kind == "launch":
        return definitions, count
    if kind == "aliases":
        first = ir.Var("launch_count", ir.ScalarType(DataType.INDEX), span)
        second = ir.Var("launch_stride", first.type, span)
        return [*definitions, ir.AssignStmt(first, count, span), ir.AssignStmt(second, first, span)], second
    if kind == "constant":
        return definitions, ir.Neg(literal(-2), DataType.INDEX, span)
    if kind == "constant_narrow_wrap":
        return definitions, ir.Cast(literal(128), DataType.INT8, span)
    if kind == "unproved_dynamic_add":
        return definitions, ir.Add(count, literal(1), DataType.INDEX, span)
    parity = ir.FloorMod(count, literal(2), DataType.INDEX, span)
    narrowed = ir.Cast(parity, DataType.INT32, span)
    # Mathematical bounds are [INT32_MAX, INT32_MAX+1], but the original
    # i32 addition is negative on an odd launch count. A later cast to INDEX
    # does not undo that wrap, nor does a positive final expression prove that
    # all preceding operations satisfied the admitted arithmetic contract.
    wrapped = ir.Add(narrowed, literal(2**31 - 1, DataType.INT32), DataType.INT32, span)
    if kind == "narrow_add_wrap":
        return definitions, wrapped
    if kind == "wrap_then_widen":
        return definitions, ir.Cast(wrapped, DataType.INDEX, span)
    if kind == "intermediate_wrap":
        positive = ir.Add(
            ir.Mul(wrapped, literal(0, DataType.INT32), DataType.INT32, span),
            literal(1, DataType.INT32),
            DataType.INT32,
            span,
        )
        return definitions, positive
    raise AssertionError(kind)


def _replace_step(program, kind, buffer):
    class ReplaceStep(ir.IRMutator):
        def visit_for_stmt(self, op):
            definitions, step = _step_case(kind, op.span, buffer)
            loop = ir.ForStmt(
                op.loop_var, op.start, op.stop, step, op.iter_args, op.body, op.return_vars, op.span
            )
            return ir.SeqStmts([*definitions, loop], op.span) if definitions else loop

    return ReplaceStep().visit_program(program)


@pytest.mark.parametrize(
    "kind,accepted",
    [
        ("launch", True),
        ("aliases", True),
        ("constant", True),
        ("constant_narrow_wrap", False),
        ("unproved_dynamic_add", False),
        ("narrow_add_wrap", False),
        ("wrap_then_widen", False),
        ("intermediate_wrap", False),
    ],
)
def test_selection_and_emission_share_machine_step_qualification(kind, accepted):
    # Inject at the actual selection boundary: earlier constant folding must
    # not hide an unsafe intermediate or turn this into a Simplify-only test.
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        manager = PassManager(OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = _control_flow_program("for")
        for transform in manager.passes[:boundary]:
            program = transform(program)
        program = _replace_step(program, kind, False)
        selected = passes.select_buffer_ir_functions()(program)
        assert _kernel(selected).attrs["buffer_ir_selected"] is accepted

        # Also exercise the independent Buffer emitter entry point; selection
        # alone cannot protect callers supplying Buffer IR directly.
        placed = _lower(_control_flow_program("for"), passes.MemoryPlanner.PYPTO)
        placed = _replace_step(placed, kind, True)
        if accepted:
            assert "scf.for" in codegen.PTOCodegen().generate(placed, emit_tile_addr=True)
        else:
            with pytest.raises(ValueError, match="requires a provably positive for-loop step"):
                codegen.PTOCodegen().generate(placed, emit_tile_addr=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
