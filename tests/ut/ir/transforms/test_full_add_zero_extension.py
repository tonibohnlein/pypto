# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Full-result adds materialize unspecified tails without shortening stores."""

import pytest
import torch
from pypto import ir, passes
from pypto import language as pl
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import PassManager

from .test_runtime_valid_broadcast import lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pl.program
class ResidualAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        hidden: pl.Tensor[[8, 16], pl.BF16],
        residual: pl.Tensor[[8, 16], pl.FP32],
        rows: pl.Scalar[pl.INDEX],
        output: pl.InOut[pl.Tensor[[24, 16], pl.FP32]],
    ) -> pl.Tensor[[24, 16], pl.FP32]:
        for visit in pl.range(3):
            active = pl.min(8, pl.max(0, 8 - visit * (8 - rows)))
            prefix = pl.cast(pl.load(hidden, [0, 0], [8, 16], valid_shape=[active, 16]), pl.FP32)
            base = pl.load(residual, [0, 0], [8, 16])
            output = pl.store(pl.add(base, prefix), [visit * 8, 0], output)
        return output


@pl.program
class ReductionAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pl.Tensor[[8, 16], pl.FP32],
        rows: pl.Scalar[pl.INDEX],
        output: pl.InOut[pl.Tensor[[24, 1], pl.FP32]],
    ) -> pl.Tensor[[24, 1], pl.FP32]:
        for visit in pl.range(3):
            active = pl.min(8, pl.max(0, 8 - visit * (8 - rows)))
            total = pl.tile.full([1, 8], value=2.0, dtype=pl.FP32)
            for _ in pl.range(2):
                prefix = pl.load(data, [0, 0], [8, 16], valid_shape=[active, 16])
                scratch = pl.create_tile([8, 16], dtype=pl.FP32)
                col = pl.row_sum(pl.mul(prefix, prefix), scratch)
                rhs = pl.reshape(col, [1, 8])
                total = pl.add(total, rhs)
            result = pl.reshape(pl.recip(pl.sqrt(total)), [8, 1])
            output = pl.store(result, [visit * 8, 0], output)
        return output


def _with_dirty_storage(program: ir.Program, byte: int):
    """Use deterministic nonzero or NaN bytes for every native allocation pool."""
    source = torch_codegen(program)
    allocation = "torch.empty(size, dtype=torch.uint8)"
    assert allocation in source
    namespace = {}
    exec(source.replace(allocation, f"torch.full((size,), {byte}, dtype=torch.uint8)"), namespace)
    return namespace["kernel"]


@pytest.mark.parametrize("rows", [0, 1, 5, 7, 8])
@pytest.mark.parametrize("byte", [0x3F, 0xFF])
def test_full_residual_store_preserves_every_lane(rows, byte):
    program, _ = lower(ResidualAdd)
    assert str(program).count("buffer.fillpad") == 1
    kernel = _with_dirty_storage(program, byte)
    residual = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 16
    hidden = (residual + 4).to(torch.bfloat16)
    output = torch.full((24, 16), -777.0)
    kernel(hidden, residual, rows, output)
    expected = residual.repeat(3, 1)
    for visit in range(3):
        active = max(0, 8 - visit * (8 - rows))
        expected[visit * 8 : visit * 8 + active] += hidden[:active].float()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 5, 7, 8])
@pytest.mark.parametrize("byte", [0x3F, 0xFF])
def test_full_reduction_store_keeps_zero_extension_across_reuse(rows, byte):
    program, _ = lower(ReductionAdd)
    assert str(program).count("buffer.fillpad") == 1
    kernel = _with_dirty_storage(program, byte)
    data = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 16
    output = torch.full((24, 1), -777.0)
    kernel(data, rows, output)
    expected = torch.full_like(output, 2.0)
    for visit in range(3):
        active = max(0, 8 - visit * (8 - rows))
        expected[visit * 8 : visit * 8 + active] += 2 * data[:active].square().sum(dim=1, keepdim=True)
    torch.testing.assert_close(output, expected.sqrt().reciprocal())


def test_narrowing_a_full_value_does_not_make_its_old_tail_unspecified():
    @pl.program
    class DefinedTail:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pl.Tensor[[8, 16], pl.FP32],
            rows: pl.Scalar[pl.INDEX],
            output: pl.InOut[pl.Tensor[[8, 16], pl.FP32]],
        ) -> pl.Tensor[[8, 16], pl.FP32]:
            full = pl.load(data, [0, 0], [8, 16])
            prefix = pl.tile.set_validshape(full, rows, 16)
            base = pl.tile.full([8, 16], value=2.0, dtype=pl.FP32)
            return pl.store(pl.add(base, prefix), [0, 0], output)

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        passes_before_selection = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = passes_before_selection.get_pass_names().index("SelectBufferIRFunctions")
        original = DefinedTail
        for transform in passes_before_selection.passes[:boundary]:
            original = transform(original)
        selected = passes.select_buffer_ir_functions()(original).get_function("kernel")
    assert selected is not None and selected.attrs["buffer_ir_selected"] is False
    kernel = original.get_function("kernel")
    assert kernel is not None
    ir.assert_structural_equal(kernel.body, selected.body)
