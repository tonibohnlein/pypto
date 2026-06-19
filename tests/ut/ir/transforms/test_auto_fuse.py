# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AutoFuse pass (MLSys-solver-driven fusion + IR emit).

AutoFuse intercepts the raw tensor-op DAG of a function marked
``attrs={"auto_fuse": True}``, runs the MLSys solver to choose a fusion
partition + tile, and rewrites the body so each fused group is wrapped in an
``InCoreScopeStmt`` for the Outline/Convert/Tile pipeline to lower into a kernel.

v0 emits one InCore scope per group and ignores the chosen spatial tile
(``AutoTileMatmulL0`` picks the L0 tile downstream), so a matmul whose output
fits L0 lowers end-to-end; applying the solver's ``[w,h]`` as cross-core chunk
loops (for larger outputs) is a later increment.
"""

import re

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


class TestAutoFuse:
    """AutoFuse solver-driven fusion + emit."""

    def test_single_matmul_emits_tiled_pipeline(self):
        """A lone matmul becomes the solver's output ``[w,h]`` tiling, each tile a
        per-tile InCore kernel with a stage=2 k-pipeline inside.

        The output is tiled into ``[w,h]`` regions (assembled into the DDR output),
        and within each tile the contraction is streamed in k-strips with a
        ``matmul``/``matmul_acc`` accumulator — the DDR<->L1 double-buffer that
        ``LowerPipelineLoops`` lowers to ping-pong GM->Mat loads.
        """

        @pl.program
        class Before:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        After = passes.auto_fuse()(Before)
        body = next(f for _, f in After.functions.items() if f.name == "mm").as_python()
        assert body.count("pl.at(level=pl.Level.CORE_GROUP") == 1  # the per-tile InCore kernel
        assert "pl.pipeline(" in body and "stage=2" in body  # the k-pipeline
        assert "pl.tensor.matmul_acc(" in body  # the per-strip accumulation
        assert "pl.tensor.slice(" in body  # the k-strip operand slices
        assert "pl.tensor.assemble(" in body  # the output-tile assembly

    def test_single_matmul_lowers_to_cube_kernel(self):
        """The emitted scope lowers through the full pipeline to a cube PTO kernel."""
        set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        out = pm.run_passes(Prog)

        # The matmul group was outlined into exactly one kernel (cube/AIC) and the
        # host became an orchestration function calling it.
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        kernel = incores[0]

        mlir = codegen.PTOCodegen().generate(ir.Program([kernel], kernel.name, kernel.span))
        assert "pto.kernel_kind" in mlir
        assert "cube" in mlir  # a pure matmul lowers to a cube kernel
        assert "pto.tload" in mlir

        # The k-pipeline lowered to a ping-pong DDR<->L1 double-buffer: the k-strip
        # GM->Mat loads land in distinct L1 buffers and accumulate via tmatmul.acc.
        mat_addrs = set()
        for line in mlir.splitlines():
            if "alloc_tile" in line and "loc=mat" in line:
                m = re.search(r"addr = (%c\d+_i64)", line)
                if m:
                    mat_addrs.add(m.group(1))
        assert len(mat_addrs) >= 2, sorted(mat_addrs)  # distinct buffers = ping-pong
        assert "pto.tmatmul.acc" in mlir  # the k-strip accumulation

    def test_large_matmul_tiles_to_fit_l0c(self):
        """A matmul whose full output exceeds L0c lowers via the output `[w,h]`
        tiling — each per-tile kernel's accumulator fits the L0c (Acc) budget.

        256x256 FP32 output = 256 KB > 128 KB L0c, so without output tiling the
        Acc buffer overflows; the solver's `[64,128]` tile keeps each kernel's
        output within L0c.
        """
        set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[256, 256], pl.FP32],
                b: pl.Tensor[[256, 256], pl.FP32],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b)
                return c

        # Lowers end-to-end without an L0c-overflow (raises on failure).
        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1
        mlir = codegen.PTOCodegen().generate(
            ir.Program([incores[0]], incores[0].name, incores[0].span)
        )
        assert "pto.tmatmul.acc" in mlir  # k-pipelined per tile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
