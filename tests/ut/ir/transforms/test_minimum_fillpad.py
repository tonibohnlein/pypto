# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single-leaf lowering of a minimum fill with an unknown signed column prefix."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pl.program
class MinimumPrefix:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        source: pl.Tensor[[16, 128], pl.FP32],
        width: pl.Scalar[pl.INDEX],
        padded_output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        original_output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    ) -> tuple[pl.Tensor[[16, 128], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
        loaded = pl.load(source, [0, 0], [16, 128])
        scaled = pl.tile.muls(loaded, 2.0)
        prefix = pl.set_validshape(scaled, 5, width)
        padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
        padded_output = pl.store(padded, [0, 0], padded_output)
        original_output = pl.store(scaled, [0, 0], original_output)
        return padded_output, original_output


def _lower(program):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        return PassManager(OptimizationStrategy.Default).run_passes(program)


def test_minimum_fillpad_writes_every_lane_at_all_signed_widths():
    program = _lower(MinimumPrefix)
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    pto = codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False)
    assert pto.count("pto.tfillpad") == 1
    assert pto.count("pto.texpands") == 1
    assert "0xff800000" in pto.lower()

    lowered = {}
    exec(torch_codegen(program), lowered)  # noqa: S102
    data = torch.arange(1, 16 * 128 + 1, dtype=torch.float32).reshape(16, 128)
    for width in (-(2**63), -1, 0, 1, 7, 8, 9, 127, 128, 129, 2**63 - 1):
        result = torch.empty_like(data)
        old_source = torch.empty_like(data)
        lowered["kernel"](data, width, result, old_source)
        expected = torch.full_like(data, -torch.inf)
        columns = min(max(width, 0), 128)
        expected[:5, :columns] = 2 * data[:5, :columns]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(old_source, 2 * data, rtol=0, atol=0)


@pytest.mark.ptoas_compile
def test_empty_prefix_emits_representable_native_splat(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    program = _lower(MinimumPrefix)
    source = tmp_path / "minimum.pto"
    cpp = source.with_suffix(".cpp")
    source.write_text(codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False))
    _run_ptoas(str(source), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=false"])
    native = cpp.read_text()
    assert "TEXPANDS(" in native and "TFILLPAD(" in native
    assert "-8388608" in native  # FP32 -infinity as signed 32-bit storage.
    assert "Inf.0" not in native and "NaN.0" not in native


def test_unproved_minimum_fillpad_cannot_defer_a_shared_prefix_setter():
    @pl.program
    class SharedPrefix:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            source: pl.Tensor[[16, 128], pl.FP32],
            width: pl.Scalar[pl.INT64],
            padded_output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            prefix_output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 128], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
            loaded = pl.load(source, [0, 0], [16, 128])
            prefix = pl.set_validshape(loaded, 5, width)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            padded_output = pl.store(padded, [0, 0], padded_output)
            prefix_output = pl.store(prefix, [0, 0], prefix_output)
            return padded_output, prefix_output

    function = _lower(SharedPrefix).get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert function.attrs["buffer_admission_property"] == "dynamic_valid_recipe"
    assert function.attrs["buffer_admission_operation"] == ir.get_op("tile.fillpad").name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
