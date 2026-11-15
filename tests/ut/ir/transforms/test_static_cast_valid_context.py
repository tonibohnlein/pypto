# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static cast descriptors remain local to each use in dynamic-valid kernels."""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.debug import torch_codegen

from .test_select_buffer_ir_functions import _PLANNERS, _compile_dynamic_state_native, _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


class _Calls(ir.IRVisitor):
    def __init__(self):
        super().__init__()
        self.calls = []

    def visit_call(self, op):
        self.calls.append(op)
        super().visit_call(op)


def _checked_lower(program, planner):
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    visitor = _Calls()
    visitor.visit_function(_kernel(lowered))
    casts = [call for call in visitor.calls if call.op.name == ir.get_op("buffer.cast").name]
    assert casts
    # Static native qualification belongs to each actual operand, including
    # aliases whose allocation descriptor was promoted to dynamic valid state.
    for call in casts:
        for operand in call.args:
            assert isinstance(operand.type, ir.BufferType)
            assert all(extent > 0 for extent in operand.type.valid_shape)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    return lowered, visitor.calls, namespace["kernel"]


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("valid", [(16, 32), (15, 31)])
def test_static_cast_with_dynamic_tail(tmp_path, planner, valid):
    rows, cols = valid
    program = pl.parse_program(f"""
@pl.program
class CastTail:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], width: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[32, 32], pl.FP16]]) -> pl.Tensor[[32, 32], pl.FP16]:
        loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[{rows}, {cols}])
        converted = pl.cast(loaded, pl.FP16, mode="rint")
        tail = pl.tile.slice(converted, [16, 32], [0, 0], valid_shape=[{rows}, width])
        out = pl.store(tail, [0, 0], out)
        return pl.store(converted, [16, 0], out)
""")
    lowered, _, execute = _checked_lower(program, planner)
    x = (torch.arange(512).reshape(16, 32).float() - 257) / 16
    for width in (0, 1, cols, 0):
        out = torch.full((32, 32), -777.0, dtype=torch.float16)
        execute(x, width, out)
        expected = torch.full_like(out, -777)
        expected[:rows, :width] = x[:rows, :width].half()
        expected[16 : 16 + rows, :cols] = x[:rows, :cols].half()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_static_cast_restores_promoted_source_and_destination(tmp_path, planner):
    program = pl.parse_program("""
@pl.program
class CastAliases:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], width: pl.Scalar[pl.INDEX],
               count: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[32, 32], pl.FP16]],
               seen: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        loaded = pl.load(x, [0, 0], [16, 32])
        for visit in pl.range(count):
            for inner in pl.range(2):
                if inner == 0:
                    narrow = pl.tile.set_validshape(loaded, 16, width)
                    seen = pl.store(narrow, [0, 0], seen)
                else:
                    narrow_other = pl.tile.set_validshape(loaded, 16, 32 - width)
                    seen = pl.store(narrow_other, [0, 0], seen)
                converted = pl.cast(loaded, pl.FP16, mode="rint")
                tail = pl.tile.set_validshape(converted, 16, width)
                out = pl.store(tail, [0, 0], out)
                restored = pl.cast(converted, pl.FP32, mode="rint")
                seen = pl.store(restored, [16, 0], seen)
                out = pl.store(converted, [16, 0], out)
        return seen
""")
    lowered, calls, execute = _checked_lower(program, planner)
    # Both sides of the first cast must exercise the promoted-handle bridge.
    windows = [call for call in calls if call.op.name == ir.get_op("buffer.subview").name]
    promoted = [call for call in windows if any(extent < 0 for extent in call.args[0].type.valid_shape)]
    assert len(promoted) >= 2
    x = (torch.arange(512).reshape(16, 32).float() - 257) / 16
    for width, count in ((0, 0), (1, 1), (32, 3), (0, 3), (7, 2)):
        out = torch.full((32, 32), -777.0, dtype=torch.float16)
        seen = torch.full((32, 32), -777.0)
        execute(x, width, count, out, seen)
        expected_out, expected_seen = torch.full_like(out, -777), torch.full_like(seen, -777)
        if count:
            expected_out[:16, :width] = x[:, :width].half()
            expected_out[16:] = x.half()
            seen_width = max(width, 32 - width)
            expected_seen[:16, :seen_width] = x[:, :seen_width]
            expected_seen[16:] = x.half().float()
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(seen, expected_seen, rtol=0, atol=0)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["dynamic", "empty", "window", "rounding"])
def test_unqualified_cast_context_remains_tile(planner, kind):
    source = {
        "dynamic": "source = pl.tile.set_validshape(loaded, 16, width)",
        "empty": "source = pl.tile.set_validshape(loaded, 0, 32)",
        "window": "source = pl.tile.slice(loaded, [16, 32], [0, 0])",
        "rounding": "source = loaded",
    }[kind]
    mode = "ceil" if kind == "rounding" else "rint"
    program = pl.parse_program(f"""
@pl.program
class UnqualifiedCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], width: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP16]]) -> pl.Tensor[[16, 32], pl.FP16]:
        loaded = pl.load(x, [0, 0], [16, 32])
        {source}
        converted = pl.cast(source, pl.FP16, mode="{mode}")
        return pl.store(converted, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional
