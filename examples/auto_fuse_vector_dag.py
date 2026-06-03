# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Raw (un-grouped, un-tiled) twin of models/02_vector_dag.py, for the AutoFuse pass.

``models/02_vector_dag.py`` writes the formula ``f = (a+b+1)(a+b+2) + (a+b)`` by hand:
every op is its own ``@pl.jit.incore`` kernel (the *grouping*) and each kernel hard-codes
a ``[128,128]`` tile (the *tiling*). Those are exactly the two decisions AutoFuse automates.

This file expresses the *same* diamond as a pure tensor-op graph — no ``pl.incore``, no
``pl.load``/tile shapes — and marks it ``attrs={"auto_fuse": True}``. The AutoFuse pass
(now wired into the pass manager just after the normalize passes, before the Outline
passes) extracts that raw op+tensor DAG, runs the MLSys solver, and prints the grouping +
tile sizes it picks — i.e. the auto equivalent of the hand-written kernel split.

Run (repo root, extension built):
    PYTHONPATH=python PYPTO_LOG_LEVEL=info python examples/auto_fuse_vector_dag.py
"""

import pypto.language as pl
from pypto.pypto_core import ir, passes


@pl.function(attrs={"auto_fuse": True})
def vector_dag_raw(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
) -> pl.Tensor[[128, 128], pl.FP32]:
    """f = (a+b+1)(a+b+2) + (a+b), as a flat tensor-op DAG (no manual grouping/tiling)."""
    c: pl.Tensor[[128, 128], pl.FP32] = pl.add(a, b)
    d: pl.Tensor[[128, 128], pl.FP32] = pl.add(c, 1.0)
    e: pl.Tensor[[128, 128], pl.FP32] = pl.add(c, 2.0)
    g: pl.Tensor[[128, 128], pl.FP32] = pl.mul(d, e)
    f: pl.Tensor[[128, 128], pl.FP32] = pl.add(g, c)
    return f


def main() -> None:
    print("=== raw tensor-op diamond (marked auto_fuse) ===")
    print(vector_dag_raw.as_python())

    # @pl.function returns an ir.Function; a Program-level pass needs a Program.
    prog = ir.Program([vector_dag_raw], "vector_dag_raw", ir.Span.unknown())

    print("\n=== AutoFuse intercepts the tensor graph (PYPTO_LOG_LEVEL=info) ===")
    passes.auto_fuse()(prog)


if __name__ == "__main__":
    main()
