# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The pinned matrix-storage-reuse kernel, copied unchanged, plus a dispatch wrapper.

The kernel text between the markers is a verbatim copy of the
``pl.parse_program(...)`` expression in
``tests/ut/ir/transforms/test_conservative_sync_storage.py::
test_affine_matrix_slices_depend_on_the_native_layout`` at the pinned revision,
parameterized by ``rows`` exactly as the test parameterizes it. Only an
Orchestration entry is appended, because an InCore function has no dispatch ABI
of its own; nothing inside ``kernel`` is altered.
"""

import pypto.language as pl

# --- begin verbatim kernel (pinned test) ---------------------------------
KERNEL = """
@pl.program
class MatrixFiberReuse:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[{rows}, 32], pl.FP16],
               seed_a: pl.Tensor[[16, 32], pl.FP16],
               seed_b: pl.Tensor[[32, 32], pl.FP16],
               out: pl.Out[pl.Tensor[[32, 16], pl.FP32]]) -> pl.Tensor[[32, 16], pl.FP32]:
        seed_l1 = pl.tile.load(seed_a, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        seed_r1 = pl.tile.load(seed_b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        seed_l0 = pl.tile.move(seed_l1, target_memory=pl.Mem.Left)
        seed_r0 = pl.tile.move(seed_r1, target_memory=pl.Mem.Right)
        seed = pl.tile.matmul(seed_l0, seed_r0)
        for i in pl.range(2):
            source = pl.tile.load(a, [0, 0], [{rows}, 32], target_memory=pl.Mem.Mat)
            left = pl.tile.extract(source, {shift} + 16 * i, 0, [16, 32], target_memory=pl.Mem.Left)
            replacement = pl.tile.create([32, 32], pl.FP16, target_memory=pl.Mem.Mat)
            replacement = pl.tile.assemble(replacement, seed, [16 * i, 0])
            transposed = pl.tile.transpose_view(replacement)
            right = pl.tile.extract(transposed, 0, 16 * i, [32, 16], target_memory=pl.Mem.Right)
            result = pl.tile.matmul(left, right)
            out = pl.tile.store(result, [16 * i, 0], out)
        return out
"""
# --- end verbatim kernel -------------------------------------------------

# Appended only so the InCore kernel can be dispatched; the kernel is untouched.
WRAPPER = """
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[{rows}, 32], pl.FP16],
             seed_a: pl.Tensor[[16, 32], pl.FP16],
             seed_b: pl.Tensor[[32, 32], pl.FP16],
             out: pl.Out[pl.Tensor[[32, 16], pl.FP32]]):
        self.kernel(a, seed_a, seed_b, out)
"""


def source(rows: int, *, with_wrapper: bool = True) -> str:
    """The program text for one row count."""
    text = KERNEL.format(rows=rows, shift=rows - 32)
    if with_wrapper:
        text += WRAPPER.format(rows=rows)
    return text


def program(rows: int, *, with_wrapper: bool = True):
    """Parse the program for one row count."""
    return pl.parse_program(source(rows, with_wrapper=with_wrapper))
