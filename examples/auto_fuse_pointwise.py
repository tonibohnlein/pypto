# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Auto-fused pointwise op — a single elementwise op tiled across the vector cores.

A lone ``c = a + 1.0`` over a large ``[4096, 384]`` tensor marked
``attrs={"auto_fuse": True}``. PTO Fusebox tiles the output into ``[w, h]``
regions sized for the vector unit; for this shape it picks 48 tiles — one per AIV
(vector) core. AutoFuse emits the chunked-parallel ``AutoInCore`` form (the twin
of the matmul ``examples/auto_fuse_matmul.py``, but the per-tile body is the
pointwise op on a slice, with no contraction k-loop). The existing
Split/Interchange/Outline passes then distribute the 48 tiles across the cores and
outline the per-tile (vector) kernel; the full output stays a DDR tensor.

Run (repo root, extension built):
    PYTHONPATH=python PYPTO_LOG_LEVEL=info python examples/auto_fuse_pointwise.py
"""

import os

import pypto.language as pl
from pypto import ir


@pl.function(attrs={"auto_fuse": True})
def add_scalar_raw(a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
    """c = a + 1.0, as a flat tensor-op graph (no manual grouping or tiling)."""
    c: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
    return c


def main() -> None:
    print("=== raw tensor-op pointwise (marked auto_fuse) ===")
    print(add_scalar_raw.as_python())

    prog = ir.Program([add_scalar_raw], "add_scalar_raw", ir.Span.unknown())

    # Compile end-to-end. AutoFuse solves the [w,h] tiling (48 regions here),
    # emits the chunked-parallel AutoInCore form, and the rest of the pipeline
    # lowers it to a vector (AIV) kernel distributed across the cores.
    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "build_output", "auto_fuse_pointwise")
    print("\n=== compile end-to-end (PYPTO_LOG_LEVEL=info shows the solver's tiling) ===")
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
