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
partition + tile, and rewrites the body to realize that decision: a matmul or a
run of fused pointwise ops becomes the solver's ``[w,h]`` output tiling distributed
across cores (chunked-parallel ``AutoInCore`` scopes — k-pipelined per tile for
matmul, the whole op chain replayed per tile with intermediates on-chip for
pointwise), and two chained matmuls the solver groups together likewise fuse into
one kernel. The Outline/Convert/Tile pipeline then lowers each scope to a cube
(AIC) or vector (AIV) kernel.
"""

import json
import re

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
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
        assert "pl.auto_chunk" in body  # AutoInCore chunked scope
        # A SINGLE flat tile loop with chunk=1 — so each tile lowers to one cross-core
        # task submission of one kernel (chunk=tile-count would serialize them on one
        # core; a nested 2D loop would collide in the orchestration codegen's naming).
        assert body.count("pl.parallel(") == 1 and "chunk=1" in body
        assert "pl.pipeline(" in body and "stage=2" in body  # the per-tile k-pipeline
        assert "pl.tensor.matmul_acc(" in body  # the per-strip accumulation
        assert "pl.tensor.slice(" in body  # the k-strip operand slices
        assert "pl.tensor.assemble(" in body  # the output-tile assembly

    def test_single_matmul_lowers_to_cube_kernel(self, ascend_backend):
        """The emitted scope lowers through the full pipeline to a cube PTO kernel."""

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

    def test_large_matmul_tiles_to_fit_l0c(self, ascend_backend):
        """A matmul whose full output exceeds L0c lowers via the output `[w,h]`
        tiling — each per-tile kernel's accumulator fits the L0c (Acc) budget.

        256x256 FP32 output = 256 KB > 128 KB L0c, so without output tiling the
        Acc buffer overflows; the solver's `[64,128]` tile keeps each kernel's
        output within L0c.
        """

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

    def test_single_pointwise_tiles_across_vector_cores(self, ascend_backend):
        """A large pointwise op is tiled into the solver's `[w,h]` regions and
        distributed across the vector cores, lowering to a vector (AIV) kernel.

        For `[4096,384]` the solver picks 48 output tiles — one per AIV core.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw").as_python()
        assert "pl.auto_chunk" in body  # AutoInCore chunked scope
        # one flat 48-tile loop with chunk=1 -> 48 cross-core task submissions of one kernel
        assert "pl.parallel(48" in body and body.count("pl.parallel(") == 1 and "chunk=1" in body
        assert "pl.tensor.adds(" in body and "pl.tensor.assemble(" in body  # per-tile op + output assembly

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1
        assert str(incores[0].func_type) == "FunctionType.AIV"  # pointwise -> vector kernel

    def test_chained_matmul_fuses_to_one_cube_kernel(self, ascend_backend):
        """Two back-to-back matmuls the solver groups together fuse into ONE kernel,
        with the intermediate staying on-chip.

        For ``C = (A@B)@D`` the solver fuses both matmuls; AutoFuse emits a single
        AutoInCore scope tiling C's output across cores, and each tile's body is the
        inner serial chain ``T_band = A_slice@B`` (on-chip) then ``C_tile =
        T_band@D_slice`` — so both matmuls land in one cube kernel rather than two
        kernels with the intermediate round-tripping DDR.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.FP32],
                b: pl.Tensor[[256, 128], pl.FP32],
                d: pl.Tensor[[128, 256], pl.FP32],
            ) -> pl.Tensor[[128, 256], pl.FP32]:
                t: pl.Tensor[[128, 128], pl.FP32] = pl.matmul(a, b)
                c: pl.Tensor[[128, 256], pl.FP32] = pl.matmul(t, d)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "chain").as_python()
        assert "pl.auto_chunk" in body  # one fused AutoInCore scope
        assert body.count("pl.tensor.matmul(") == 2  # both matmuls in the same per-tile body
        assert "_tband" in body  # the on-chip intermediate (T never touches DDR)
        assert body.count("pl.parallel(") == 1 and "chunk=1" in body  # one flat tile loop -> N cross-core submissions

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        # The fused chain is ONE cube kernel, not two separate matmul kernels.
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        assert str(incores[0].func_type) == "FunctionType.AIC"
        mlir = codegen.PTOCodegen().generate(ir.Program([incores[0]], incores[0].name, incores[0].span))
        assert "cube" in mlir and mlir.count("pto.tmatmul") >= 2  # both matmuls fused in

    def test_chained_matmul_preserves_operand_input_order(self, tmp_path, monkeypatch):
        """Regression: the solver Problem must list each matmul's inputs in OPERAND
        order — inputs[0]=LHS, inputs[1]=RHS — because the cost model derives
        M/N/K positionally (K = inputs[0].width, N = inputs[1].width).

        The builder collected inputs into a ``std::set<size_t>``, which re-sorts by
        tensor index. In-params are registered before op outputs, so for a chained
        ``(A@B)@D`` the sink ``matmul(t, d)`` came out as ``[d, t]`` (in-param d has
        the lower index) instead of ``[t, d]`` — silently swapping LHS/RHS and
        scrambling the sink's M/N/K. We assert via the env-gated Problem dump that
        the on-chip intermediate is the sink's FIRST input.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.FP32],
                b: pl.Tensor[[256, 128], pl.FP32],
                d: pl.Tensor[[128, 256], pl.FP32],
            ) -> pl.Tensor[[128, 256], pl.FP32]:
                t: pl.Tensor[[128, 128], pl.FP32] = pl.matmul(a, b)
                c: pl.Tensor[[128, 256], pl.FP32] = pl.matmul(t, d)
                return c

        passes.auto_fuse()(Prog)

        dag = json.loads((tmp_path / "chain.dag.json").read_text())
        inputs, outputs = dag["inputs"], dag["outputs"]
        # The sink is the op that consumes another op's output (the intermediate t).
        sink_idx = intermediate = None
        for i, ins in enumerate(inputs):
            for j, outs in enumerate(outputs):
                if j != i and outs[0] in ins:
                    sink_idx, intermediate = i, outs[0]
        assert sink_idx is not None, dag
        sink_inputs = inputs[sink_idx]
        # pl.matmul(t, d): the intermediate t is the LHS, so it MUST be inputs[0].
        assert sink_inputs[0] == intermediate, (sink_inputs, intermediate)
        assert len(sink_inputs) == 2 and sink_inputs[1] != intermediate

    def test_chained_pointwise_fuses_into_one_tiled_kernel(self, ascend_backend):
        """Two chained pointwise ops the solver groups fuse into one tiled vector
        kernel, with the intermediate staying on-chip.

        For ``c = (a+1.0)*2.0`` over ``[4096,384]`` the solver fuses both ops and
        tiles the output across the vector cores (48 tiles). Each tile's body
        replays the whole chain on a ``[h,w]`` slice — so both ops land in one AIV
        kernel and the intermediate ``t`` is never materialized to DDR (a single
        output assemble), rather than two kernels round-tripping ``t`` through memory.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw2(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                t: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.mul(t, 2.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw2").as_python()
        assert "pl.auto_chunk" in body  # one fused AutoInCore scope
        # one flat 48-tile loop with chunk=1 -> 48 cross-core task submissions of one kernel
        assert "pl.parallel(48" in body and body.count("pl.parallel(") == 1 and "chunk=1" in body
        assert "pl.tensor.adds(" in body and "pl.tensor.muls(" in body  # both ops in the per-tile body
        assert body.count("pl.tensor.assemble(") == 1  # only the output is assembled; the intermediate stays on-chip

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        # The fused chain is ONE vector kernel, not two.
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        assert str(incores[0].func_type) == "FunctionType.AIV"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
