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

    def test_single_matmul_emits_chunked_tiled_kernel(self):
        """A lone matmul becomes the solver's output ``[w,h]`` tiling distributed
        across cores.

        An AutoInCore (``chunked_loop_optimizer``) scope wraps chunked ``parallel``
        loops over the output tiles — the existing Split/Interchange/Outline passes
        distribute those tiles across cores. Each tile's body streams the
        contraction in k-strips with a ``matmul``/``matmul_acc`` accumulator (the
        DDR<->L1 double-buffer) and assembles the tile into the DDR output.
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
        assert "chunked_loop_optimizer" in body  # AutoInCore chunked scope
        assert "pl.parallel(" in body and "chunk=" in body  # cross-core tile distribution
        assert "pl.pipeline(" in body and "stage=2" in body  # the per-tile k-pipeline
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

    def test_single_pointwise_tiles_across_vector_cores(self):
        """A large pointwise op is tiled into the solver's `[w,h]` regions and
        distributed across the vector cores, lowering to a vector (AIV) kernel.

        For `[4096,384]` the solver picks 48 output tiles — one per AIV core.
        """
        set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw").as_python()
        assert "chunked_loop_optimizer" in body  # AutoInCore chunked scope
        assert "pl.parallel(48" in body  # 48 output tiles — one per vector core
        assert "pl.tensor.adds(" in body and "pl.tensor.assemble(" in body  # per-tile op + output assembly

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1
        assert str(incores[0].func_type) == "FunctionType.AIV"  # pointwise -> vector kernel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
