# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Auto-fused pointwise chain — two elementwise ops fused into one tiled vector kernel.

``c = (a + 1.0) * 2.0`` over a large ``[4096, 384]`` tensor marked
``attrs={"auto_fuse": True}``. The MLSys solver fuses both ops into one group (the
intermediate ``t = a + 1.0`` is ephemeral), and AutoFuse realizes it as the
solver's ``[w, h]`` output tiling distributed across the vector cores — for this
shape, 48 tiles, one per AIV core. Each tile's body replays the *whole chain* on a
``[h, w]`` slice:

    for (mt, nt) in pl.parallel(...):            # outer: output tiles, across cores
        a_tile = a[mt-rows, nt-cols]             # slice the input
        t_tile = a_tile + 1.0                    # MM-free: stays in UB (the fusion)
        c_tile = t_tile * 2.0
        c = assemble(c, c_tile, [mt, nt])

The intermediate ``t_tile`` never touches DDR — it lives in the vector unit (UB)
for the duration of the tile. That is the fusion: two ops, one kernel, no
round-trip of the intermediate through memory. Contrast the single-op
``auto_fuse_pointwise.py`` (one op per tile) and ``auto_fuse_vector_dag.py`` (a
5-op diamond) — this is the minimal two-op chain.

Run (repo root, extension built):
    PYTHONPATH=python PYPTO_LOG_LEVEL=info python examples/auto_fuse_pointwise_chain.py
"""

import os

import pypto.language as pl
from pypto import ir


@pl.function(attrs={"auto_fuse": True})
def pointwise_chain_raw(a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
    """c = (a + 1.0) * 2.0, as a flat tensor-op graph (no manual grouping or tiling)."""
    t: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
    c: pl.Tensor[[4096, 384], pl.FP32] = pl.mul(t, 2.0)
    return c


def main() -> None:
    print("=== raw tensor-op pointwise chain (marked auto_fuse) ===")
    print(pointwise_chain_raw.as_python())

    prog = ir.Program([pointwise_chain_raw], "pointwise_chain_raw", ir.Span.unknown())

    # Compile end-to-end. AutoFuse fuses the two ops (intermediate stays on-chip),
    # tiles the output across the vector cores, and the rest of the pipeline lowers
    # the per-tile chain to a single vector (AIV) kernel.
    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "build_output", "auto_fuse_pointwise_chain")
    print("\n=== compile end-to-end (PYPTO_LOG_LEVEL=info shows the solver fusing both ops) ===")
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
