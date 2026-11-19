# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A2/A3 native integer fields in the ordinary Functional and Buffer pipelines."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags, _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

pytestmark = [
    pytest.mark.usefixtures("ascend_backend"),
    pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["functional", "buffer"]),
]


def _lower(program: ir.Program, enable_buffer_ir: bool, *, buffer_eligible: bool = True) -> ir.Program:
    with passes.PassContext(
        [],
        enable_buffer_ir=enable_buffer_ir,
        enable_pypto_autosync=enable_buffer_ir,
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    kernel = lowered.get_function("kernel")
    assert kernel is not None
    expected = (
        ir.FunctionIRStage.Buffer if enable_buffer_ir and buffer_eligible else ir.FunctionIRStage.Functional
    )
    assert kernel.ir_stage == expected
    return lowered


def _generate(program: ir.Program) -> str:
    return codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False)


def _compile_native(program: ir.Program, tmp_path: Path) -> str:
    pto = _generate(program)
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS compiler unavailable")
    source = tmp_path / "kernel.pto"
    source.write_text(pto, encoding="utf-8")
    output = source.with_suffix(".cpp")
    kernel = program.get_function("kernel")
    assert kernel is not None
    flags = _get_ptoas_flags(sync_owned=bool(kernel.attrs.get("pypto_autosync")))
    _run_ptoas(str(source), str(output), flags)
    return output.read_text(encoding="utf-8")


def _broadcast_valid_shape(cpp: str, rows: int, cols: int) -> tuple[int, int]:
    # The native template selects normal mode when ValidCol equals Cols or Rows
    # equals 1. Inspect the tile passed as the destination of the native call.
    call = re.search(r"\bTCOLEXPANDMUL\((\w+),", cpp)
    assert call is not None
    descriptor = re.search(
        rf"Tile<TileType::Vec, float, {rows}, {cols}, BLayout::RowMajor, (-?\d+), "
        rf"(-?\d+), [^>\n]+> {call[1]}\b",
        cpp,
    )
    assert descriptor is not None
    return int(descriptor[1]), int(descriptor[2])


def _broadcast_program(rows: int, cols: int, valid_rows: int | str, valid_cols: int) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class Broadcast:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[{rows}, {cols}], pl.FP32], n: pl.Scalar[pl.INDEX],
               carrier: pl.Tensor[[1, {cols}], pl.FP32],
               out: pl.Out[pl.Tensor[[{rows}, {cols}], pl.FP32]]) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:
        source = pl.load(x, [0, 0], [{rows}, {cols}], valid_shape=[{valid_rows}, {valid_cols}])
        row = pl.load(carrier, [0, 0], [1, {cols}], valid_shape=[1, {valid_cols}])
        result = pl.tile.col_expand_mul(source, row)
        return pl.store(result, [0, 0], out)
""")


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "rows,cols",
    [
        pytest.param(256, 8, id="repeat-256"),
        pytest.param(257, 8, id="repeat-257"),
        pytest.param(2, 2048, id="stride-256-blocks"),
    ],
)
def test_broadcast_uses_count_mode_when_normal_mode_overflows(
    enable_buffer_ir: bool, rows: int, cols: int, tmp_path: Path
) -> None:
    """Keep the Functional count-mode path when static Buffer fields would wrap."""
    lowered = _lower(_broadcast_program(rows, cols, rows, cols), enable_buffer_ir, buffer_eligible=False)
    cpp = _compile_native(lowered, tmp_path)
    assert _broadcast_valid_shape(cpp, rows, cols) == (-1, -1)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "rows,cols,valid_rows,valid_cols",
    [
        pytest.param(255, 8, 255, 8, id="repeat-255"),
        pytest.param(2, 2040, 2, 2040, id="stride-255-blocks"),
        pytest.param(257, 8, 255, 8, id="active-repeat-255"),
        pytest.param(2, 2048, 1, 2048, id="single-active-row"),
        pytest.param(1, 2048, 1, 2048, id="single-physical-row"),
        pytest.param(257, 16, 257, 8, id="partial-cols-count-mode"),
        pytest.param(2, 2048, 2, 2040, id="partial-cols-wide-stride"),
        pytest.param(257, 8, "n", 8, id="dynamic-rows-large-repeat"),
        pytest.param(2, 2048, "n", 2048, id="dynamic-rows-wide-stride"),
    ],
)
def test_broadcast_safe_native(
    enable_buffer_ir: bool, rows: int, cols: int, valid_rows: int | str, valid_cols: int, tmp_path: Path
) -> None:
    """Keep in-range normal mode and count mode available."""
    lowered = _lower(_broadcast_program(rows, cols, valid_rows, valid_cols), enable_buffer_ir)
    cpp = _compile_native(lowered, tmp_path)
    expected = (valid_rows, valid_cols) if enable_buffer_ir and isinstance(valid_rows, int) else (-1, -1)
    assert _broadcast_valid_shape(cpp, rows, cols) == expected


def _extract_program(
    rows: int,
    cols: int,
    row: int | str,
    col: int | str,
    width: int = 16,
    *,
    dtype: str = "FP16",
    source_valid_width: int | None = None,
) -> ir.Program:
    validity = "" if source_valid_width is None else f", valid_shape=[{rows}, {source_valid_width}]"
    return pl.parse_program(f"""
@pl.program
class Extract:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[{rows}, {cols}], pl.{dtype}],
               r: pl.Scalar[pl.INDEX], c: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[1, {width}], pl.{dtype}]]) -> pl.Tensor[[1, {width}], pl.{dtype}]:
        source = pl.load(x, [0, 0], [{rows}, {cols}]{validity})
        result = pl.tile.extract(source, {row}, {col}, [1, {width}], target_memory=pl.Mem.Vec)
        return pl.store(result, [0, 0], out)
""")


@pytest.mark.parametrize(
    "col",
    [pytest.param(65536, id="static"), pytest.param("c * 16", id="dynamic")],
)
def test_extract_rejects_unrepresentable_column(enable_buffer_ir: bool, col: int | str) -> None:
    """The wide tile permits a 32-byte aligned start at 65536 that would wrap to zero."""
    lowered = _lower(_extract_program(1, 65552, 0, col), enable_buffer_ir)
    with pytest.raises(ValueError, match=r"A2/A3 TEXTRACT.*\[0, 65535\]"):
        _generate(lowered)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "rows,cols,row,col,width",
    [
        pytest.param(1, 65552, 0, 65520, 16, id="last-aligned-column"),
        pytest.param(4, 64, "r", "c * 16", 16, id="dynamic-both-coordinates"),
        pytest.param(1, 65552, "r", 65520, 16, id="dynamic-row-wide-source"),
        pytest.param(1, 65552, 0, "c * 16", 32, id="dynamic-column-bounded-by-window"),
    ],
)
def test_extract_safe_native(
    enable_buffer_ir: bool, rows: int, cols: int, row: int | str, col: int | str, width: int, tmp_path: Path
) -> None:
    """Accept each coordinate when the static start or in-bounds geometry fits uint16."""
    lowered = _lower(_extract_program(rows, cols, row, col, width), enable_buffer_ir)
    assert "TEXTRACT(" in _compile_native(lowered, tmp_path)


@pytest.mark.parametrize(
    "modulus,representable",
    [
        pytest.param(65536, True, marks=pytest.mark.ptoas_compile, id="bounded-offset"),
        pytest.param(65552, False, id="offset-can-reach-65536"),
    ],
)
def test_extract_dynamic_column_floor_mod(
    enable_buffer_ir: bool, modulus: int, representable: bool, tmp_path: Path
) -> None:
    # Both divisors preserve block alignment, but only the smaller divisor
    # excludes 65536. Buffer selection cannot infer alignment through FloorMod.
    program = _extract_program(1, 65552, 0, f"(c * 16) % {modulus}")
    lowered = _lower(program, enable_buffer_ir, buffer_eligible=False)
    if representable:
        assert "TEXTRACT(" in _compile_native(lowered, tmp_path)
    else:
        with pytest.raises(ValueError, match=r"A2/A3 TEXTRACT.*\[0, 65535\]"):
            _generate(lowered)


@pytest.mark.parametrize(
    "source_valid_width",
    [
        pytest.param(None, id="full-window"),
        pytest.param(65536, marks=pytest.mark.ptoas_compile, id="partial-window"),
    ],
)
def test_extract_checks_active_destination_extent(
    enable_buffer_ir: bool, source_valid_width: int | None, tmp_path: Path
) -> None:
    # INT8 fits both physical windows in Vec memory and selects Functional.
    # Starting at column 32 leaves 65504 valid columns when the source is partial.
    program = _extract_program(1, 65568, 0, 32, 65536, dtype="INT8", source_valid_width=source_valid_width)
    lowered = _lower(program, enable_buffer_ir, buffer_eligible=False)
    if source_valid_width is None:
        with pytest.raises(ValueError, match=r"A2/A3 TEXTRACT destination valid column extent 65536"):
            _generate(lowered)
    else:
        cpp = _compile_native(lowered, tmp_path)
        extract = re.search(r"\bTEXTRACT\((\w+),", cpp)
        assert extract is not None
        destination = re.search(
            rf"Tile<TileType::Vec, int8_t, 1, 65536, [^\n]+> {extract[1]} = [^\n]+\(\w+, (\w+)\);", cpp
        )
        assert destination is not None
        assert re.search(rf"\bconst int64_t {destination[1]} = 65504;", cpp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
