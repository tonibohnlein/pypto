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


def cast_program(
    cols: int,
    valid_cols: int,
    valid_rows: int = 16,
    dynamic: bool = False,
    source_dtype=pl.INT32,
    target_dtype=pl.FP16,
    mode: str = "round",
    saturation_mode: str | None = None,
):
    if dynamic:
        return _dynamic_cast_program(cols, valid_rows, source_dtype, target_dtype, mode)

    @pl.program
    class CastProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            value: pl.Tensor[[32, cols], source_dtype],
            output: pl.Out[pl.Tensor[[32, cols], target_dtype]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], target_dtype]:
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, valid_cols])
            cast = pl.cast(tile, target_dtype, mode=mode, saturation_mode=saturation_mode)
            return pl.store(cast, [0, 0], output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            value: pl.Tensor[[32, cols], source_dtype],
            output: pl.Out[pl.Tensor[[32, cols], target_dtype]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], target_dtype]:
            return self.kernel(value, output, logical_cols)

    return CastProgram


def _dynamic_cast_program(cols: int, valid_rows: int, source_dtype, target_dtype, mode: str):
    @pl.program
    class DynamicCastProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            value: pl.Tensor[[32, cols], source_dtype],
            output: pl.Out[pl.Tensor[[32, cols], target_dtype]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], target_dtype]:
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, logical_cols])
            cast = pl.cast(tile, target_dtype, mode=mode)
            return pl.store(cast, [0, 0], output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            value: pl.Tensor[[32, cols], source_dtype],
            output: pl.Out[pl.Tensor[[32, cols], target_dtype]],
            logical_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32, cols], target_dtype]:
            return self.kernel(value, output, logical_cols)

    return DynamicCastProgram


def _pto(program) -> str:
    optimized = _optimized(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(ir.Program([func], "kernel", optimized.span))


def _optimized(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


class _OpCollector(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ir.Call] = []

    def visit_call(self, op: ir.Call) -> None:
        self.calls.append(op)
        super().visit_call(op)


def _calls(program, name: str) -> list[ir.Call]:
    collector = _OpCollector()
    collector.visit_program(program)
    return [call for call in collector.calls if call.op.name == name]


@pytest.mark.parametrize(
    "source_dtype,target_dtype,mode",
    [(pl.INT32, pl.FP16, "round"), (pl.FP16, pl.INT8, "trunc")],
)
@pytest.mark.parametrize(
    "cols,valid_cols,fragments",
    [
        (224, 224, 2),
        (448, 448, 4),
        (128, 112, 1),
        (128, 104, 1),
        # Residual logical widths in aligned physical frames: 128+1,
        # 128+127, and 128+128+1.
        (256, 129, 2),
        (256, 255, 2),
        (384, 257, 3),
    ],
)
def test_native_cast_fragments_write_destination_views(
    cols: int, valid_cols: int, fragments: int, source_dtype, target_dtype, mode: str
):
    program = cast_program(cols, valid_cols, source_dtype=source_dtype, target_dtype=target_dtype, mode=mode)
    optimized = _optimized(program)
    fragment_calls = _calls(optimized, "tile.cast_fragment")
    assert len(fragment_calls) == fragments
    assert not _calls(optimized, "tile.cast")
    for call in fragment_calls:
        assert len(call.args) in (2, 3)
        src_type, dst_type = call.args[0].type, call.args[1].type
        assert isinstance(src_type, ir.TileType) and isinstance(dst_type, ir.TileType)
        assert src_type.shape == dst_type.shape
        assert src_type.memref is None and dst_type.memref is None

    pto = _pto(program)
    assert "scf.for" in pto and "row_fragment" in pto
    assert "sizes [1, " in pto
    assert "pto.tmov" not in pto and "assemble" not in pto
    assert pto.count("pto.tcvt ") == fragments
    assert pto.count("pto.subview ") == fragments * 2
    assert pto.count("pto.alloc_tile") == 2
    assert pto.count("pto.tload ") == pto.count("pto.tstore ") == 1


@pytest.mark.parametrize(
    "source_dtype,target_dtype,mode",
    [(pl.INT32, pl.FP16, "round"), (pl.FP16, pl.INT8, "trunc")],
)
@pytest.mark.parametrize("cols", [32, 64, 128, 256, 896])
def test_aligned_cast_keeps_native_fast_path(cols: int, source_dtype, target_dtype, mode: str):
    pto = _pto(cast_program(cols, cols, source_dtype=source_dtype, target_dtype=target_dtype, mode=mode))
    assert "tcvt_row" not in pto
    assert pto.count("pto.tcvt ") == 1


@pytest.mark.parametrize("saturation_mode", ["on", "off"])
def test_fragmented_cast_preserves_saturation_mode(saturation_mode: str):
    """Every fragmented tcvt keeps the authored saturation mode."""

    pto = _pto(
        cast_program(
            224,
            224,
            source_dtype=pl.FP16,
            target_dtype=pl.INT8,
            mode="trunc",
            saturation_mode=saturation_mode,
        )
    )
    tcvt_lines = [line for line in pto.splitlines() if "pto.tcvt " in line]
    assert tcvt_lines
    expected = f"satmode = #pto<saturation_mode {saturation_mode.upper()}>"
    assert all(expected in line for line in tcvt_lines)
    assert all("rmode = #pto<round_mode TRUNC>" in line for line in tcvt_lines)


def test_fragmented_nonsaturating_cast_threads_the_level3_scratch():
    """Every fragment consumes the allocated level3 tmp, never PTOAS's fixed UB region.

    A2/A3 materializes a tcvt scratch tile for non-saturating FP16->INT8. PTOAS's
    no-tmp tcvt overload reaches for a fixed TMP_UB_OFFSET region that PyPTO does
    not reserve, so a fragment that omits the operand can overwrite a live tile
    while the reserved scratch goes unused.
    """

    pto = _pto(
        cast_program(
            224,
            224,
            source_dtype=pl.FP16,
            target_dtype=pl.INT8,
            mode="trunc",
            saturation_mode="off",
        )
    )
    tcvt_lines = [line for line in pto.splitlines() if "pto.tcvt " in line]
    assert len(tcvt_lines) == 2, tcvt_lines
    for line in tcvt_lines:
        ins = line.split("ins(", 1)[1].split(")", 1)[0]
        operands = ins.split("{", 1)[0]
        assert operands.count(",") == 1, line


def test_saturating_cast_fragments_take_no_scratch():
    """A saturating narrowing cast is native, so no tmp is allocated or consumed."""

    pto = _pto(
        cast_program(
            224,
            224,
            source_dtype=pl.FP16,
            target_dtype=pl.INT8,
            mode="trunc",
            saturation_mode="on",
        )
    )
    assert "tcvt_tmp" not in pto


def test_internal_cast_fragments_survive_python_roundtrip():
    optimized = _optimized(cast_program(224, 224))
    printed = ir.python_print(optimized)
    assert "cast_fragment" in printed
    ir.assert_structural_equal(optimized, pl.parse_program(printed))


@pytest.mark.parametrize("valid_rows", [0, 1, 16, 32])
def test_dynamic_cast_clips_fragments_and_guards_empty_tail(valid_rows: int):
    pto = _pto(cast_program(224, 224, valid_rows, dynamic=True))
    lines = pto.splitlines()
    row_loop = next(i for i, line in enumerate(lines) if "scf.for" in line and "row_fragment" in line)
    clip_lines = [i for i, line in enumerate(lines) if "arith.maxsi" in line or "arith.minsi" in line]
    assert len(clip_lines) == 4
    assert all(i < row_loop for i in clip_lines), "runtime fragment widths must be loop-invariant"
    assert pto.count("scf.if ") == 2
    assert pto.count("pto.tcvt ") == 2
    assert "pto.tmov" not in pto


@pytest.mark.parametrize(
    "cols,valid_cols,dynamic,source_dtype,target_dtype,mode",
    [
        (224, 224, False, pl.INT32, pl.FP16, "round"),
        (448, 448, False, pl.INT32, pl.FP16, "round"),
        (128, 112, False, pl.INT32, pl.FP16, "round"),
        (128, 104, False, pl.INT32, pl.FP16, "round"),
        (256, 129, False, pl.INT32, pl.FP16, "round"),
        (256, 255, False, pl.INT32, pl.FP16, "round"),
        (384, 257, False, pl.INT32, pl.FP16, "round"),
        (224, 224, True, pl.INT32, pl.FP16, "round"),
        (224, 224, False, pl.FP16, pl.INT8, "trunc"),
        (448, 448, False, pl.FP16, pl.INT8, "trunc"),
        (128, 112, False, pl.FP16, pl.INT8, "trunc"),
        (128, 104, False, pl.FP16, pl.INT8, "trunc"),
        (256, 129, False, pl.FP16, pl.INT8, "trunc"),
        (256, 255, False, pl.FP16, pl.INT8, "trunc"),
        (384, 257, False, pl.FP16, pl.INT8, "trunc"),
        (224, 224, True, pl.FP16, pl.INT8, "trunc"),
    ],
)
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_cast_fragments_survive_real_ptoas(
    tmp_path: Path,
    cols: int,
    valid_cols: int,
    dynamic: bool,
    source_dtype,
    target_dtype,
    mode: str,
    planner,
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS not installed")
    with passes.PassContext([], memory_planner=planner):
        ir.compile(
            cast_program(
                cols,
                valid_cols,
                dynamic=dynamic,
                source_dtype=source_dtype,
                target_dtype=target_dtype,
                mode=mode,
            ),
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
