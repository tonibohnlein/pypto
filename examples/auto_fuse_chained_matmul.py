# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Auto-fused chained matmul — two back-to-back matmuls fused into one kernel.

``C = (A @ B) @ D`` marked ``attrs={"auto_fuse": True}``. The MLSys solver decides
to *fuse* the two matmuls into a single group (the intermediate ``T = A @ B`` is
ephemeral), and picks an output tile. AutoFuse realizes the fusion as an inner
serial matmul chain inside a parallel-outer tiling:

    for (mt, nt) in pl.parallel(...):        # outer: C's [w,h] output tiles, across cores
        T_band = matmul(A[mt-rows, :], B)    # MM1 -> [h, K2], k-pipelined, stays on-chip
        C_tile = matmul(T_band, D[:, nt-cols])  # MM2 -> [h, w]
        C = assemble(C, C_tile, [mt, nt])

The intermediate ``T_band`` never touches DDR — that is the fusion. The whole
chain lowers to ONE cube (AIC) kernel distributed across the cores.

Constraint: the per-tile ``T_band`` (MM1's output) must fit L0c; these shapes keep
it well within budget (a larger intermediate needs AutoTileMatmulL0 M/N-tiling).

Run (repo root, extension built):
    PYTHONPATH=python PYPTO_LOG_LEVEL=info python examples/auto_fuse_chained_matmul.py
"""

import os

import pypto.language as pl
from pypto import ir


@pl.function(attrs={"auto_fuse": True})
def chained_matmul_raw(
    A: pl.Tensor[[128, 256], pl.FP32],
    B: pl.Tensor[[256, 128], pl.FP32],
    D: pl.Tensor[[128, 256], pl.FP32],
) -> pl.Tensor[[128, 256], pl.FP32]:
    """C = (A @ B) @ D, as a flat tensor-op graph (no manual grouping or tiling)."""
    T: pl.Tensor[[128, 128], pl.FP32] = pl.matmul(A, B)
    C: pl.Tensor[[128, 256], pl.FP32] = pl.matmul(T, D)
    return C


def main() -> None:
    print("=== raw tensor-op chained matmul (marked auto_fuse) ===")
    print(chained_matmul_raw.as_python())

    prog = ir.Program([chained_matmul_raw], "chained_matmul_raw", ir.Span.unknown())

    # Compile end-to-end. AutoFuse fuses the two matmuls (T stays on-chip), tiles
    # C's output across cores, and the rest of the pipeline lowers the inner serial
    # chain to a single cube (AIC) kernel.
    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "build_output", "auto_fuse_chained_matmul")
    print("\n=== compile end-to-end (PYPTO_LOG_LEVEL=info shows the solver fusing both matmuls) ===")
    compiled = ir.compile(prog, output_dir=out_dir, skip_ptoas=True)
    print(f"compiled to: {compiled}")

    pto_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(str(compiled))
        for f in files
        if f.endswith(".pto")
    ]
    for path in sorted(pto_files):
        print(f"\n=== generated kernel: {os.path.basename(path)} ===")
        with open(path) as f:
            print(f.read())


if __name__ == "__main__":
    main()
