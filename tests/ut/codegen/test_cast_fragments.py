# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cast row-pitch and masked-tail regressions, including real PTOAS coverage."""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def cast_program(cols: int, valid_cols: int, valid_rows: int = 16, dynamic: bool = False):
    if dynamic:
        return _dynamic_cast_program(cols, valid_rows)

    @pl.program
    class CastProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            value: pl.Tensor[[32, cols], pl.INT32],
            output: pl.Out[pl.Tensor[[32, cols], pl.FP16]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], pl.FP16]:
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, valid_cols])
            cast = pl.cast(tile, pl.FP16, mode="round")
            return pl.store(cast, [0, 0], output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            value: pl.Tensor[[32, cols], pl.INT32],
            output: pl.Out[pl.Tensor[[32, cols], pl.FP16]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], pl.FP16]:
            return self.kernel(value, output, logical_cols)

    return CastProgram


def _dynamic_cast_program(cols: int, valid_rows: int):
    @pl.program
    class DynamicCastProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            value: pl.Tensor[[32, cols], pl.INT32],
            output: pl.Out[pl.Tensor[[32, cols], pl.FP16]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], pl.FP16]:
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, logical_cols])
            cast = pl.cast(tile, pl.FP16, mode="round")
            return pl.store(cast, [0, 0], output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            value: pl.Tensor[[32, cols], pl.INT32],
            output: pl.Out[pl.Tensor[[32, cols], pl.FP16]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], pl.FP16]:
            return self.kernel(value, output, logical_cols)

    return DynamicCastProgram


def _pto(program) -> str:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(ir.Program([func], "kernel", optimized.span))


@pytest.mark.parametrize("cols,valid_cols", [(224, 224), (448, 448), (128, 112), (128, 104)])
def test_native_cast_fragments_write_destination_views(cols: int, valid_cols: int):
    pto = _pto(cast_program(cols, valid_cols))
    assert "scf.for %tcvt_row" in pto
    assert "tcvt_src_fragment" in pto and "tcvt_dst_fragment" in pto
    assert "sizes [1, " in pto
    assert "pto.tmov" not in pto and "assemble" not in pto
    assert pto.count("pto.alloc_tile") == 2
    assert pto.count("pto.tload ") == pto.count("pto.tstore ") == 1


@pytest.mark.parametrize("cols", [32, 64, 128, 256, 896])
def test_aligned_cast_keeps_native_fast_path(cols: int):
    pto = _pto(cast_program(cols, cols))
    assert "tcvt_row" not in pto
    assert pto.count("pto.tcvt ") == 1


@pytest.mark.parametrize("valid_rows", [0, 1, 16, 32])
def test_dynamic_cast_clips_fragments_and_guards_empty_tail(valid_rows: int):
    pto = _pto(cast_program(224, 224, valid_rows, dynamic=True))
    assert "arith.maxsi" in pto and "arith.minsi" in pto
    assert "scf.if %tcvt_nonempty" in pto
    assert "pto.tmov" not in pto


@pytest.mark.parametrize(
    "cols,valid_cols,dynamic",
    [
        (224, 224, False),
        (448, 448, False),
        (128, 112, False),
        (128, 104, False),
        (224, 224, True),
    ],
)
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_cast_fragments_survive_real_ptoas(
    tmp_path: Path, cols: int, valid_cols: int, dynamic: bool, planner
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS not installed")
    with passes.PassContext([], memory_planner=planner):
        ir.compile(
            cast_program(cols, valid_cols, dynamic=dynamic),
            output_dir=str(tmp_path),
            dump_passes=False,
            skip_ptoas=False,
        )
    kernels = list((tmp_path / "kernels").rglob("*.cpp"))
    assert kernels, "PTOAS must produce kernel C++, not only a .pto file"
    cpp = "\n".join(path.read_text() for path in kernels)
    assert "TCVT(" in cpp
    assert "TMOV(" not in cpp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
