# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Target legalization for GM-to-Mat loads with a wide parent row stride."""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _program(n: int):
    @pl.program
    class WideRhs:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[16, 16], pl.BF16],
            rhs: pl.Tensor[[16, n], pl.BF16],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            lhs_tile = pl.load(lhs, [0, 0], [16, 16])
            rhs_tile = pl.load(rhs, [0, 0], [16, 16])
            acc = pl.matmul(lhs_tile, rhs_tile, out_dtype=pl.FP32)
            return pl.store(acc, [0, 0], output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            lhs: pl.Tensor[[16, 16], pl.BF16],
            rhs: pl.Tensor[[16, n], pl.BF16],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return self.kernel(lhs, rhs, output)

    return WideRhs


def _optimized(n: int, backend_type: BackendType = BackendType.Ascend910B):
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_program(n))


class _OpCollector(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def visit_call(self, op: ir.Call) -> None:
        self.names.append(op.op.name)
        super().visit_call(op)


def _kernel_pto(n: int) -> tuple[_OpCollector, str]:
    optimized = _optimized(n)
    collector = _OpCollector()
    collector.visit_program(optimized)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    pto = codegen.PTOCodegen().generate(ir.Program([func], "kernel", optimized.span))
    return collector, pto


@pytest.mark.parametrize("n", [65024, 65535])
def test_safe_rhs_stride_keeps_direct_mat_load(n: int):
    collector, pto = _kernel_pto(n)
    assert "tile.load_rebased_row" not in collector.names
    assert "rebased_ptr" not in pto
    assert f"%c{n}_index" in pto


@pytest.mark.parametrize("n", [65536, 65552, 152064])
def test_wide_rhs_stride_uses_compact_rebased_rows(n: int):
    collector, pto = _kernel_pto(n)
    assert collector.names.count("tile.load_rebased_row") == 1
    assert "pto.addptr" in pto
    assert "compact_row_view" in pto
    assert "shape = [%c1_index, %c16_index], strides = [%c16_index, %c1_index]" in pto
    assert "scf.for" in pto
    # The parent TensorView may still be declared for the ABI parameter, but no
    # load-bearing partition of it may feed the RHS Mat tile.
    rebased_view = next(
        line for line in pto.splitlines() if "compact_row_view" in line and "make_tensor_view" in line
    )
    assert f"%c{n}_index" not in rebased_view


def test_backend_without_stride_limit_keeps_direct_mat_load():
    optimized = _optimized(65536, BackendType.Ascend950)
    collector = _OpCollector()
    collector.visit_program(optimized)
    assert "tile.load_rebased_row" not in collector.names


@pytest.mark.parametrize("n", [65024, 65535, 65536, 65552, 152064])
def test_wide_rhs_boundary_controls_survive_real_ptoas(tmp_path: Path, n: int):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS not installed")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        ir.compile(_program(n), output_dir=str(tmp_path), dump_passes=False, skip_ptoas=False)
    assert list((tmp_path / "kernels").rglob("*.cpp"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
