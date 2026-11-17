# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static row uses retain their descriptor after a different alias changes metadata."""

import pytest
import torch
from pypto import codegen, ir, passes
from pypto import language as pl
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import PassManager

from .buffer_test_utils import statements
from .test_runtime_valid_broadcast import diagnostics, metadata_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


def row_alias_program(columns=16, reduction="row_sum", broadcast="row_expand_sub"):
    return pl.parse_program(f"""
@pl.program
class StaticRowAlias:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,32], pl.FP32], width: pl.Scalar[pl.INDEX],
               seen: pl.InOut[pl.Tensor[[8,32], pl.FP32]],
               again: pl.InOut[pl.Tensor[[8,32], pl.FP32]],
               out: pl.InOut[pl.Tensor[[8,32], pl.FP32]]) -> pl.Tensor[[8,32], pl.FP32]:
        source = pl.load(x, [0,0], [8,32], valid_shape=[8,{columns}])
        narrow = pl.tile.set_validshape(source, 8, width)
        seen = pl.store(narrow, [0,0], seen)
        workspace = pl.create_tile([8,32], dtype=pl.FP32)
        column = pl.{reduction}(source, workspace)
        result = pl.{broadcast}(source, column)
        out = pl.store(result, [0,0], out)
        again = pl.store(narrow, [0,0], again)
        return out
""")


def lower(program, planner, enabled=True):
    with passes.PassContext([], enable_buffer_ir=enabled, memory_planner=planner):
        return PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("columns", [16, 32])
@pytest.mark.parametrize(
    "reduction,broadcast", [("row_sum", "row_expand_sub"), ("row_max", "row_expand_mul")]
)
def test_static_row_metadata_is_independent_of_dynamic_alias(planner, columns, reduction, broadcast):
    original = row_alias_program(columns, reduction, broadcast)
    before = ir.serialize(original)
    lowered = lower(original, planner)
    assert ir.serialize(original) == before
    function = lowered.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    assert not diagnostics(function)
    row_calls = [
        stmt.expr
        for stmt in statements(lowered)
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name
        in {ir.get_op(f"buffer.{reduction}").name, ir.get_op(f"buffer.{broadcast}").name}
    ]
    assert len(row_calls) == 2
    # Check actual consumer operands, rather than merely finding some static
    # descriptor elsewhere in the function. Native full-column dynamic checks
    # must not reinterpret the original partial-column access as width 32.
    for call in row_calls:
        assert isinstance(call.args[0].type, ir.BufferType)
        assert list(call.args[0].type.shape) == [8, 32]
        assert list(call.args[0].type.valid_shape) == [8, columns]
    assert isinstance(row_calls[0].args[1].type, ir.BufferType)
    assert isinstance(row_calls[0].args[2].type, ir.BufferType)
    assert isinstance(row_calls[1].args[2].type, ir.BufferType)
    assert list(row_calls[0].args[1].type.valid_shape) == [8, 32]
    assert list(row_calls[0].args[2].type.valid_shape) == [8, 1]
    assert list(row_calls[1].args[2].type.valid_shape) == [8, columns]
    generated = codegen.PTOCodegen().generate(lowered)
    assert "pto.trowsum" in generated if reduction == "row_sum" else "pto.trowmax" in generated
    # The independent executable reference uses original physical values and
    # logical valid widths, not compiler storage classes or demand generators.
    namespace = {}
    exec(torch_codegen(lowered), namespace)
    x = torch.arange(256, dtype=torch.float32).reshape(8, 32) + 1
    for width in [0, 8, columns]:
        outputs = [torch.full_like(x, -777) for _ in range(3)]
        namespace["kernel"](x, width, *outputs)
        expected_seen = torch.full_like(x, -777)
        expected_seen[:, :width] = x[:, :width]
        torch.testing.assert_close(outputs[0], expected_seen)
        torch.testing.assert_close(outputs[1], expected_seen)
        active = x[:, :columns]
        column = (
            active.sum(dim=1, keepdim=True) if reduction == "row_sum" else active.amax(dim=1, keepdim=True)
        )
        expected = torch.full_like(x, -777)
        expected[:, :columns] = active - column if broadcast == "row_expand_sub" else active * column
        torch.testing.assert_close(outputs[2], expected)


@pytest.mark.parametrize("planner", PLANNERS)
def test_buffer_disabled_preserves_functional_row_path(planner):
    program = lower(row_alias_program(), planner, enabled=False)
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert "buffer.row_sum" not in str(program)


def test_static_bridge_does_not_qualify_unrelated_runtime_rows():
    # A static descriptor at a different use does not prove dynamic descriptor
    # equality at this call. Keep the existing function/use-site negative.
    errors = diagnostics(metadata_fixture("mismatch"))
    assert errors and any("matching runtime rows" in error.message for error in errors)
