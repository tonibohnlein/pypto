# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FFN compute graph with the AutoFuse pass active (adapted from models/01_ffn.py).

``models/01_ffn.py`` writes the FFN as hand-tiled InCore kernels (``matmul_kernel``,
``gelu_kernel``) chained through an orchestration entry — fusion decided by hand.
AutoFuse instead runs on the *tensor-level* graph, before kernels are formed, and
decides the fusion itself. So this expresses the same matmul -> activation ->
matmul shape as a single tensor-level function marked ``auto_fuse=True`` (via the
``attrs`` decorator argument), then runs the pass and renders the DAG it extracts.

Run (repo root, extension built):
    PYTHONPATH=python PYPTO_LOG_LEVEL=info python examples/auto_fuse_ffn.py
"""

import os
import subprocess

import pypto.language as pl
from pypto.pypto_core import ir, passes

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pl.function(attrs={"auto_fuse": True})
def ffn(
    hidden: pl.Tensor[[128, 128], pl.FP32],
    gate_w: pl.Tensor[[128, 128], pl.FP32],
    down_w: pl.Tensor[[128, 128], pl.FP32],
) -> pl.Tensor[[128, 128], pl.FP32]:
    """matmul -> (x*x + x) activation -> matmul, kept fully tensor-level."""
    gate: pl.Tensor[[128, 128], pl.FP32] = pl.matmul(hidden, gate_w)  # cube
    sq: pl.Tensor[[128, 128], pl.FP32] = pl.mul(gate, gate)  # pointwise
    act: pl.Tensor[[128, 128], pl.FP32] = pl.add(sq, gate)  # pointwise
    out: pl.Tensor[[128, 128], pl.FP32] = pl.matmul(act, down_w)  # cube
    return out


def main() -> None:
    print("=== tensor-level FFN (marked auto_fuse) ===")
    print(ffn.as_python())

    # @pl.function returns an ir.Function; a Program-level pass needs a Program.
    prog = ir.Program([ffn], "ffn_autofuse", ir.Span.unknown())

    out_dir = os.path.join(_REPO, "fusion_dag")
    os.makedirs(out_dir, exist_ok=True)
    os.environ["PYPTO_AUTOFUSE_DUMP"] = out_dir  # tells the pass to dump the extracted DAG

    print("\n=== running passes.auto_fuse()  (PYPTO_LOG_LEVEL=info shows the result) ===")
    passes.auto_fuse()(prog)

    # Render the extracted DAG via the mlsys visualize.py + graphviz, if available.
    viz = os.path.join(_REPO, "3rdparty", "mlsys26", "scripts", "visualize.py")
    dag = os.path.join(out_dir, "ffn.dag.json")
    if os.path.exists(dag) and os.path.exists(viz):
        dot = os.path.join(out_dir, "ffn.dag.dot")
        png = os.path.join(out_dir, "ffn.dag.png")
        subprocess.run(["python", viz, "instance", dag, dot], check=True, capture_output=True)
        subprocess.run(["dot", "-Tpng", dot, "-o", png], check=True, capture_output=True)
        print(f"\nextracted DAG : {dag}\nrendered graph: {png}")


if __name__ == "__main__":
    main()
