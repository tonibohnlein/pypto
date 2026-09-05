# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Executable cast-stage/quantization regressions with explicit output directions."""

import pypto.language as pl
import pytest
import torch
from harness import st


def cast_stage_kernel(cols: int, valid_rows: int, valid_cols: int, stage: str):
    @pl.jit
    def int32_half(value: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, valid_cols])
            converted = pl.cast(tile, target_type=pl.FP16, mode="round")
            pl.store(converted, [0, 0], output)
        return output

    @pl.jit
    def float32_int32(value: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, valid_cols])
            converted = pl.cast(tile, target_type=pl.INT32, mode="rint")
            pl.store(converted, [0, 0], output)
        return output

    @pl.jit
    def half_int8(value: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, valid_cols])
            converted = pl.cast(tile, target_type=pl.INT8, mode="trunc")
            pl.store(converted, [0, 0], output)
        return output

    return {"i32_f16": int32_half, "f32_i32": float32_int32, "f16_i8": half_int8}[stage]


def quantize_kernel(cols: int, valid_rows: int, valid_cols: int):
    @pl.jit
    def quantize(value: pl.Tensor, scale: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(value, [0, 0], [32, cols], valid_shape=[valid_rows, valid_cols])
            row_scale = pl.load(scale, [0, 0], [32, 1], valid_shape=[valid_rows, 1])
            scaled = pl.row_expand_mul(tile, row_scale)
            integer = pl.cast(scaled, target_type=pl.INT32, mode="rint")
            half = pl.cast(integer, target_type=pl.FP16, mode="round")
            quantized = pl.cast(half, target_type=pl.INT8, mode="trunc")
            pl.store(quantized, [0, 0], output)
        return output

    return quantize


# Same physical frame, differing valid width; no spatial dispatch or strip-loop
# changes confound the padding discriminator. Wide widths exercise row pitches.
GEOMETRIES = [
    (128, 16, 128),
    (128, 16, 112),
    (128, 16, 104),
    (224, 16, 224),
    (448, 16, 448),
    (256, 16, 256),
    (128, 1, 112),
    (128, 32, 112),
]


def valid_rectangle(rows: int, cols: int):
    """Compare only the logical result; storage outside valid_shape is unspecified."""

    def compare(actual, expected):
        assert actual.keys() == expected.keys()
        for name, got in actual.items():
            torch.testing.assert_close(got[:rows, :cols], expected[name][:rows, :cols], rtol=0, atol=0)

    return compare


def stage_cases():
    stages = [
        ("f32_i32", torch.float32, torch.int32),
        ("i32_f16", torch.int32, torch.float16),
        ("f16_i8", torch.float16, torch.int8),
    ]
    for cols, rows, valid in GEOMETRIES:
        for seed in range(3):
            generator = torch.Generator().manual_seed(seed)
            integers = torch.randint(-127, 128, (32, cols), generator=generator, dtype=torch.int32)
            for name, src_dtype, dst_dtype in stages:
                value = integers.to(src_dtype)
                output = torch.full((32, cols), -128, dtype=dst_dtype)
                expected = output.clone()
                expected[:rows, :valid] = value[:rows, :valid].to(dst_dtype)
                yield st.case(
                    cast_stage_kernel(cols, rows, valid, name),
                    value,
                    output,
                    name=f"cast_{name}_{cols}_{rows}_{valid}_seed{seed}",
                    golden=lambda _, expected=expected: expected,
                    compare=valid_rectangle(rows, valid),
                )


def quantize_cases():
    for cols, rows, valid in GEOMETRIES:
        for seed in range(3):
            generator = torch.Generator().manual_seed(seed)
            value = torch.rand((32, cols), generator=generator) * 254 - 127
            scale = torch.ones((32, 1), dtype=torch.float32)
            output = torch.full((32, cols), -128, dtype=torch.int8)
            expected = output.clone()
            expected[:rows, :valid] = torch.round(value[:rows, :valid]).to(torch.float16).to(torch.int8)
            yield st.case(
                quantize_kernel(cols, rows, valid),
                value,
                scale,
                output,
                name=f"quantize_{cols}_{rows}_{valid}_seed{seed}",
                golden=lambda _, expected=expected: expected,
                compare=valid_rectangle(rows, valid),
            )


@st.cases(*stage_cases())
def test_cast_stage_tail(case_run):
    case_run.assert_passed()


@st.cases(*quantize_cases())
def test_quantize_padded_tail(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
