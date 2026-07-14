# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Auto-fused matmul — the matmul twin of ``auto_fuse_vector_dag.py``, compiled end-to-end.

``examples/kernels/03_matmul.py`` writes a matmul kernel BY HAND: it picks the InCore
grouping (``with pl.incore():``) and hard-codes the tile (``pl.load(a, [0, 0], [64, 64])``,
the Mat -> Left/Right moves, the ``pl.matmul``). Those two decisions — *which ops form a
kernel* and *what tile* — are exactly what AutoFuse automates.

This file expresses the matmul as a pure tensor-op graph (``c = pl.matmul(a, b)`` — no
``pl.incore``, no tile shapes) and marks it ``attrs={"auto_fuse": True}``. Compiling it runs
the AutoFuse pass: it extracts the op+tensor DAG, runs PTO Fusebox to choose the fusion
partition + tile, and rewrites the body to realize that decision for the rest of the pipeline
(Outline -> ConvertTensorToTileOps -> AutoTileMatmulL0 -> ... -> codegen) to lower into a cube
kernel.

AutoFuse applies the solver's spatial ``[w, h]`` tile: it emits the output tiling as
``AutoInCore`` chunked-parallel loops distributed across the cube cores, each tile's body
streaming the contraction in k-strips through a ``stage=2`` DDR<->L1 software pipeline
(``matmul``/``matmul_acc`` accumulator). ``AutoTileMatmulL0`` then picks the L0 sub-tile
downstream. So even this 64x64 matmul lowers to the solver's sub-tile regions fanned across
the cores, each a cube kernel; a larger output is simply more regions.

Run (repo root, extension built):
    PYTHONPATH=python PYPTO_LOG_LEVEL=info python examples/auto_fuse_matmul.py
"""

import os

import pypto.language as pl
from pypto import ir


@pl.function(attrs={"auto_fuse": True})
def matmul_raw(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    """c = a @ b, as a flat tensor-op graph (no manual grouping or tiling)."""
    c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
    return c


def main() -> None:
    print("=== raw tensor-op matmul (marked auto_fuse) ===")
    print(matmul_raw.as_python())

    # @pl.function returns an ir.Function; the compile pipeline needs a Program.
    prog = ir.Program([matmul_raw], "matmul_raw", ir.Span.unknown())

    # Compile end-to-end. AutoFuse runs inside the Default strategy: it solves the
    # fusion partition, emits the InCore scope, and the rest of the pipeline lowers
    # it to a cube kernel. skip_ptoas=True emits raw MLIR (.pto) — no device toolchain
    # needed. Run with PYPTO_LOG_LEVEL=info to see the solver's grouping + tile.
    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "build_output", "auto_fuse_matmul")
    print("\n=== compile end-to-end (AutoFuse solves + emits, then the pipeline lowers) ===")
    compiled = ir.compile(prog, output_dir=out_dir, skip_ptoas=True)
    print(f"compiled to: {compiled}")

    # Show the generated cube kernel.
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
