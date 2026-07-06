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

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

# These tests were rewritten against the grounded SPMD emit: the pre-grounding cost model
# plus the #1895 auto_chunk removal changed the solver's decisions (small matmuls now
# split-K) and migrated the emit onto SPMD (pl.spmd + a tiled zero-seed + atomic-add
# merge), so the old pl.auto_chunk / pl.pipeline(stage=2) / matmul_acc assertions were
# stale. All cases are now re-derived and live. The one exception is the chained-matmul
# LOWERING, which is xfail on hw-native-sys/pypto#1908 (an AllocateMemoryAddr bump-allocator
# limitation that can't pack the chain's L0A buffers — NOT an AutoFuse emit bug).


class TestAutoFuse:
    """AutoFuse solver-driven fusion + emit."""

    def test_single_matmul_emits_spmd_tiled_split_k_kernel(self, ascend_backend):
        """A lone 64x64 matmul becomes the grounded solver's SPMD output tiling with split-K.

        Pinned to Ascend910B (``ascend_backend``): the solver's tile/split decision is
        backend-specific — the grounded 910B model (24 cube cores) tiles the 64x64 output
        into 32x32 regions (tile=32x32x32) and splits the K axis in 2 to fill more cores —
        4 output tiles x 2 K-partials = 8 SPMD blocks. (Under this directory's default
        Ascend950 the solver instead picks tile=32x32x64, split=1 — a different emit.)
        AutoFuse emits TWO flat SPMD loops:

          - a 4-block ``pl.spmd(4)`` zero-seed — one ``[32,32]`` ``pl.tensor.full`` tile per
            output tile — so the split-K partials accumulate onto a zeroed output WITHOUT
            any core materializing the full ``[64,64]`` (which would overflow UB at scale);
          - an 8-block ``pl.spmd(8)`` matmul (one block = one cross-core task submission)
            whose body slices the K-strip and output tile (``pl.tensor.slice``), runs the
            32x32x32 tile matmul, and scatters the partial back with
            ``atomic=pl.AtomicType.Add`` — the split-K merge.
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
        # Two flat SPMD loops: the tiled zero-seed (4 output tiles) and the matmul (4 tiles
        # x 2 K-split = 8 blocks). Both flat (not nested 2D, which would collide in the
        # orchestration codegen's variable naming).
        assert "pl.spmd(4" in body  # tiled zero-seed, one [32,32] tile per output tile
        assert "pl.spmd(8" in body  # split-K matmul
        assert body.count("pl.spmd(") == 2
        # The seed zeroes per-tile (pl.tensor.full on a [32,32] tile, never the full output).
        assert "pl.tensor.full(" in body
        # Per-block matmul body: K-strip + output-tile slices, the 32x32x32 tile matmul, atomic merge.
        assert "pl.tensor.slice(" in body
        assert "pl.tensor.matmul(" in body
        assert "atomic=pl.AtomicType.Add" in body

    def test_single_matmul_emit_is_numerically_correct(self, ascend_backend):
        """The SPMD-tiled + split-K emit computes the same result as a plain matmul.

        ``torch_codegen(..., run_all_spmd_blocks=True)`` runs all 8 SPMD blocks serially
        into the shared atomic-seeded output, so the generated function reproduces the
        FULL 64x64 result (not just block 0). It must match ``torch.matmul`` to fp32
        tolerance — verifying the tiling, K-strip slicing, and split-K atomic merge are
        numerically faithful, not only structurally present.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

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
        code = torch_codegen(After, run_all_spmd_blocks=True)
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102

        torch.manual_seed(0)
        a = torch.randn(64, 64, dtype=torch.float32)
        b = torch.randn(64, 64, dtype=torch.float32)
        out = namespace["mm"](a, b)
        assert torch.allclose(out, a @ b, rtol=1e-4, atol=1e-4), (
            f"max abs diff {(out - a @ b).abs().max().item():.3e}"
        )

    def test_single_matmul_lowers_to_cube_kernel(self, ascend_backend):
        """The emitted scope lowers through the full pipeline to a cube PTO kernel.

        Under the grounded 910B split-K decision (split=2), the 64x64 matmul lowers to
        TWO in-core kernels: a vector (AIV) kernel that zero-seeds the output, and the
        cube (AIC) matmul kernel whose SPMD blocks accumulate their 32x32 tile into that
        seed via atomic-add — the K partials merge across blocks in DDR. (The pre-grounding
        emit was a single non-split kernel with a k-pipeline ping-pong / ``tmatmul.acc``;
        that shape no longer applies.)
        """

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

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)

        # split-K outlines into a cube (AIC) matmul kernel + a vector (AIV) zero-seed kernel;
        # the host `mm` becomes an Orchestration function driving the SPMD dispatch.
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        by_type: dict[str, list] = {}
        for f in incores:
            by_type.setdefault(str(f.func_type), []).append(f)
        assert len(by_type.get("FunctionType.AIC", [])) == 1, [f.name for f in incores]
        assert len(by_type.get("FunctionType.AIV", [])) == 1, [f.name for f in incores]

        cube = by_type["FunctionType.AIC"][0]
        mlir = codegen.PTOCodegen().generate(ir.Program([cube], cube.name, cube.span))
        assert "pto.kernel_kind" in mlir
        assert "cube" in mlir  # a pure matmul lowers to a cube kernel
        assert "pto.tload" in mlir and "pto.tmatmul" in mlir
        # SPMD-distributed split-K: each block indexes off spmd_block_idx and merges its
        # partial into the seeded output via atomic-add (not a per-tile tmatmul.acc).
        assert "spmd_block_idx" in mlir
        assert "atomic_add" in mlir

    def test_large_matmul_tiled_seed_avoids_ub_overflow(self, ascend_backend):
        """A large matmul's split-K zero-seed is TILED so it never overflows UB.

        The grounded 910B solver splits-K on a 256x256 matmul (tile=128x128/split=4). The
        zero-seed must NOT materialize the full 256x256 output (262144 B > the 188416 B UB
        budget) on one core — AutoFuse tiles it across SPMD blocks (one [128,128] zero tile
        each), so the whole thing lowers end-to-end. Regression for the seed-overflow bug.
        (The old L0c-overflow framing is obsolete: fitting Acc is AutoTileMatmulL0's job.)
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

        # Must lower end-to-end without a UB-overflow VerificationError on the seed kernel.
        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        by_type: dict[str, list] = {}
        for f in incores:
            by_type.setdefault(str(f.func_type), []).append(f)
        assert len(by_type.get("FunctionType.AIC", [])) == 1, [f.name for f in incores]
        assert len(by_type.get("FunctionType.AIV", [])) == 1, [f.name for f in incores]

        # The seed kernel is SPMD-tiled (block-indexed), not a single full-output alloc.
        seed = by_type["FunctionType.AIV"][0]
        seed_mlir = codegen.PTOCodegen().generate(ir.Program([seed], seed.name, seed.span))
        assert "spmd_block_idx" in seed_mlir
        cube = by_type["FunctionType.AIC"][0]
        cube_mlir = codegen.PTOCodegen().generate(ir.Program([cube], cube.name, cube.span))
        assert "cube" in cube_mlir and "atomic_add" in cube_mlir  # split-K merge

    def test_single_pointwise_tiles_across_vector_cores(self, ascend_backend):
        """A large pointwise op is tiled into the solver's `[w,h]` regions and
        distributed across the vector cores, lowering to a vector (AIV) kernel.

        Pinned to Ascend910B (48 vector cores): for `[4096,384]` the grounded solver picks
        a `[512,64]` tile, so the output tiles into 8x6 = 48 disjoint regions — one per AIV
        core — emitted as a single flat `pl.spmd(48)` loop. No split-K (pointwise has no
        contraction), so no zero-seed and a plain (non-atomic) per-tile `assemble`.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw").as_python()
        # A single flat 48-block SPMD loop -> 48 cross-core task submissions of one kernel.
        assert "pl.spmd(48" in body and body.count("pl.spmd(") == 1
        # Per-block: slice the tile, apply the op, assemble into the output. No split-K, so
        # no CORE_GROUP zero-seed and no atomic merge.
        assert "pl.tensor.slice(" in body
        assert "pl.tensor.adds(" in body and "pl.tensor.assemble(" in body
        assert "pl.at(level=pl.Level.CORE_GROUP" not in body and "AtomicType" not in body

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1  # no split-K seed kernel -> just the vector kernel
        assert str(incores[0].func_type) == "FunctionType.AIV"  # pointwise -> vector kernel

    def test_chained_matmul_fuses_to_one_kernel(self, ascend_backend):
        """Two back-to-back matmuls the solver groups together fuse into ONE kernel,
        with the intermediate staying on-chip.

        For ``C = (A@B)@D`` the solver fuses both matmuls into one group; AutoFuse emits a
        single CORE_GROUP scope holding both ``tensor.matmul``s, with the intermediate ``t``
        as a scope-local consumed by the second matmul — never assembled out to DDR. That is
        the fusion this checks (one kernel, not two round-tripping ``t``). NB: the solver's
        96x64 tile does not divide the 128x256 output, so the emit uses the whole-output
        fused scope rather than a spatial SPMD tiling; the lowering of this scope is exercised
        separately (test_chained_matmul_lowers_to_cube_kernel, xfail on #1908).
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
        # Both matmuls fused into ONE CORE_GROUP scope; the intermediate t stays on-chip
        # (consumed by the second matmul), never assembled to DDR -> one kernel, not two.
        assert "pl.at(level=pl.Level.CORE_GROUP" in body
        assert body.count("pl.tensor.matmul(") == 2
        assert "pl.tensor.assemble(" not in body  # t never round-trips DDR

    @pytest.mark.xfail(
        reason="chained-matmul lowering blocked on hw-native-sys/pypto#1908: AllocateMemoryAddr "
        "(bump allocator) can't pack the chain's A-stationary producer (64KB L0A) + "
        "double-buffered consumer (2x32KB) -> 'Left buffer usage 98304 > 65536'. Not an "
        "AutoFuse emit bug. Remove this marker when #1908's offset-packing lands.",
        strict=True,
    )
    def test_chained_matmul_lowers_to_cube_kernel(self, ascend_backend):
        """The fused chain lowers through the Default pipeline to a single cube kernel.

        XFAIL(#1908): the Default pipeline currently raises an AllocateMemoryAddr L0A
        overflow on the chained kernel. When #1908 lands this XPASSes (strict) -> drop the
        marker.
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

        For ``c = (a+1.0)*2.0`` over ``[4096,384]`` (Ascend910B) the solver fuses both ops
        and tiles the output into 48 `[512,64]` regions across the vector cores, emitted as
        one flat ``pl.spmd(48)`` loop. Each block replays the whole chain on its slice — so
        both ops land in one AIV kernel and the intermediate ``t`` is never materialized to
        DDR (a single output ``assemble``), rather than two kernels round-tripping ``t``.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw2(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                t: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.mul(t, 2.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw2").as_python()
        # One fused flat 48-block SPMD loop -> 48 cross-core task submissions of one kernel.
        assert "pl.spmd(48" in body and body.count("pl.spmd(") == 1
        assert "pl.tensor.adds(" in body and "pl.tensor.muls(" in body  # both ops in the per-block body
        assert body.count("pl.tensor.assemble(") == 1  # only the output is assembled; the intermediate stays on-chip

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        # The fused chain is ONE vector kernel, not two.
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        assert str(incores[0].func_type) == "FunctionType.AIV"

    def test_inout_param_is_a_solver_input(self, ascend_backend):
        """An InOut param is READ by a fused op, so the solver must register it as a graph
        input (like an In param). Before, only In params were registered, so ``add(T, x)`` was
        seen with an incomplete input set (the InOut ``T`` dropped — undercounting its DDR read
        or making the op look input-less). This asserts an InOut auto_fuse function fuses (the
        add scoped, reading BOTH ``T`` and ``x``) and lowers end-to-end through the pipeline.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def acc(
                self,
                T: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                x: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                r: pl.Tensor[[64, 64], pl.FP32] = pl.add(T, x)
                return r

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "acc").as_python()
        # The add is fused + tiled across cores; its per-tile body slices BOTH the InOut T (now a
        # registered graph input) and x, then adds the slices — so T is read as a tracked input.
        assert "fused_0" in body
        assert "pl.tensor.slice(T," in body  # the InOut param is read (sliced) as an input
        assert "pl.tensor.add(" in body
        # Lowers end-to-end through the full pipeline (InOut discipline write-back included).
        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]

    def test_wide_reduction_avoids_subgranule_strip_overflow(self, ascend_backend, monkeypatch):
        """A wide fused row reduction (rmsnorm over W=1024) must lower without a UB overflow.

        A reduction tile is col-major, so its row axis is padded to the DMA granule g. The
        pipeline chunks the free row axis h; when h/num_strips < g (or not a multiple of it),
        each strip is padded up to g and the stage=2 ping-pong double-buffers those padded
        strips, blowing past UB on a wide tile. The generic emit now requires granule-multiple
        reduction strips and otherwise stays serial (the un-chunked tile fits). This asserts the
        wide case lowers through AllocateMemoryAddr end-to-end — with sub-granule pipelining it
        overflowed. (Behind the generic-emit flag, where the fix lives.)
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def rms(self, a: pl.Tensor[[256, 1024], pl.FP32]) -> pl.Tensor[[256, 1024], pl.FP32]:
                sq: pl.Tensor[[256, 1024], pl.FP32] = pl.mul(a, a)
                ms: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(sq)
                r: pl.Tensor[[256, 1024], pl.FP32] = pl.row_expand_div(a, ms)
                return r

        # Reaches AllocateMemoryAddr without raising a Vec-buffer-overflow (the serial fallback fits).
        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
