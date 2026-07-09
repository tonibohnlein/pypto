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

    def test_nonuniform_matmul_tiles_via_ceil_clamp_grid(self, ascend_backend):
        """A matmul whose solver tile does NOT divide the output tiles via a ceil+clamp
        SPMD grid (the G-A fix for the "matmul-tiling gap").

        Pinned to Ascend910B: for ``[272,272]`` the grounded solver picks a non-uniform
        ``80x144`` spatial grid (``272 % 80 != 0``, ``272 % 144 != 0``, parts 2x4). Instead
        of declining to one untiled InCore scope (the old fallback), the emitter realizes a
        ``ceil(272/144) x ceil(272/80) = 2x4 = 8``-block ``pl.spmd(8)`` grid whose per-block
        offsets are CLAMPED in-bounds (``pl.min(mt*144, 128)`` / ``pl.min(nt*80, 192)``). Every
        block reads a full ``[144,80]`` tile; the ragged blocks OVERLAP the previous, and the
        NON-atomic spatial assemble recomputes the overlap identically (idempotent), so the
        result is exact. This mirrors the vector ``emit_strip`` ceil+clamp. split=1 here, so no
        zero-seed and no ``atomic`` merge (split-K stays divisor-only).
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[272, 272], pl.FP32],
                b: pl.Tensor[[272, 272], pl.FP32],
            ) -> pl.Tensor[[272, 272], pl.FP32]:
                c: pl.Tensor[[272, 272], pl.FP32] = pl.matmul(a, b)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "mm").as_python()
        # Ceil+clamp grid: a single flat 8-block SPMD loop (2x4), NOT an untiled CORE_GROUP scope.
        assert "pl.spmd(8" in body and body.count("pl.spmd(") == 1
        assert "pl.at(level=pl.Level.CORE_GROUP" not in body
        # Per-block: sliced [144,80] operands, one tile matmul, clamped offsets (pl.min),
        # non-atomic assemble (split=1 -> no split-K seed, no atomic merge).
        assert "pl.tensor.slice(" in body and body.count("pl.tensor.matmul(") == 1
        assert "pl.min(" in body  # in-bounds offset clamp for the ragged (ceil) blocks
        assert "AtomicType" not in body and "pl.tensor.full(" not in body

        # Lowers end-to-end to a single cube (AIC) kernel — the clamped offsets survive the
        # full pipeline (the ragged-grid emit is not just structurally present but lowerable).
        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        assert str(incores[0].func_type) == "FunctionType.AIC"

    def test_nonuniform_matmul_ceil_clamp_is_numerically_correct(self, ascend_backend):
        """The ceil+clamp non-uniform grid computes the same result as a plain matmul.

        ``torch_codegen(..., run_all_spmd_blocks=True)`` runs all blocks of the ragged grid
        serially into the shared (non-atomic) output; the overlapping ceil blocks recompute
        their region identically, so the whole ``[272,272]`` result must match ``torch.matmul``
        to fp32 tolerance. Verifies the tiling, clamped slicing, and idempotent overlap are
        numerically faithful, not just structurally present. A second non-square shape
        (``[272,272]@[272,240]``) exercises an independent M/N clamp.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        def _check(M: int, K: int, N: int) -> None:
            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[M, K], pl.FP32],
                    b: pl.Tensor[[K, N], pl.FP32],
                ) -> pl.Tensor[[M, N], pl.FP32]:
                    c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b)
                    return c

            namespace: dict = {}
            exec(torch_codegen(passes.auto_fuse()(Prog), run_all_spmd_blocks=True), namespace)  # noqa: S102
            torch.manual_seed(0)
            a = torch.randn(M, K, dtype=torch.float32)
            b = torch.randn(K, N, dtype=torch.float32)
            out = namespace["mm"](a, b)
            assert torch.allclose(out, a @ b, rtol=1e-4, atol=1e-4), (
                f"[{M},{K}]@[{K},{N}]: max abs diff {(out - a @ b).abs().max().item():.3e}"
            )

        _check(272, 272, 272)  # square non-divisor: 2x4 ceil grid, clamp on both axes
        _check(272, 272, 240)  # non-square non-divisor: independent M/N clamp

    def test_ragged_k_pipeline_peel_is_numerically_correct(self, ascend_backend):
        """A matmul whose per-core contraction slice does NOT divide evenly by the solver's
        k-tile pipelines the full k-strips + one matmul_acc tail (ragged-K peel).

        ``BuildTileMatmul`` streams the contraction K in ``k``-strips with a stage-2 pipeline.
        When ``k`` does not divide the (per-split) contraction, it runs ``floor(K/k)`` full
        strips and folds the ragged remainder in as ONE extra ``matmul_acc`` tail (width
        ``K - floor(K/k)*k``, which is 16-aligned since the solver's ``k`` is). Pinned to
        Ascend910B: ``[64,5040]@[5040,256]`` splits K into 7 slices of 720; the solver's
        ``k=336`` peels each into ``2*336 + 48``. ``[64,4096]@[4096,256]`` is the exact-division
        control (slice 512, ``k=256`` -> ``2*256``, no tail). Both must match ``torch.matmul``;
        the ragged path must be numerically exact, not silently dropping the tail contribution.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        def _fuse_body(M: int, K: int, N: int):
            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[M, K], pl.FP32],
                    b: pl.Tensor[[K, N], pl.FP32],
                ) -> pl.Tensor[[M, N], pl.FP32]:
                    c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b)
                    return c

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "mm").as_python()
            namespace: dict = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)  # noqa: S102
            torch.manual_seed(0)
            a = torch.randn(M, K, dtype=torch.float32) * 0.05
            b = torch.randn(K, N, dtype=torch.float32) * 0.05
            out = namespace["mm"](a, b)
            assert torch.allclose(out, a @ b, rtol=1e-3, atol=1e-3), (
                f"[{M},{K}]@[{K},{N}]: max abs diff {(out - a @ b).abs().max().item():.3e}"
            )
            return body

        # Ragged peel: per-split contraction 720, k=336 -> 2 full strips (pipeline bound 672)
        # + a 48-wide matmul_acc tail (720 - 2*336, 16-aligned). Pins the current solver plan.
        peel_body = _fuse_body(64, 5040, 256)
        assert "pl.pipeline(0, 672, 336" in peel_body  # 2 full k-strips (bound < slice 720)
        assert "pl.tensor.matmul_acc(" in peel_body  # loop accumulate + the peel tail fold
        assert "[64, 48]" in peel_body  # the ragged K-tail slice (720 - 2*336 = 48)

        # Exact-division control: k-pipeline with no tail (slice 512, k=256 -> bound 512).
        div_body = _fuse_body(64, 4096, 256)
        assert "pl.pipeline(0, 512, 256" in div_body  # 2 full strips, no ragged tail
        assert "[64, 48]" not in div_body  # no peel tail for the dividing case

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
        # The intermediate t is a scope-local consumed inside the CORE_GROUP scope (never
        # assembled). The ONLY assemble is the output-wiring copy of the returned result `c` into
        # the appended `c_out` param — a matmul-produced return has no create to lift, so it is
        # copied into an Out param (device/harness binds outputs by position; without it the
        # output is unwritten). So exactly one assemble, and the return is wired to an Out param.
        assert body.count("pl.tensor.assemble(") == 1
        assert "pl.Out[" in body

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

    def test_chained_matmul_deep_t_tiles_shared_dim(self, ascend_backend):
        """When a fused chain's per-tile intermediate T_band [h,K2] exceeds L0c, the shared
        dimension K2 is tiled so MM2 becomes a matmul_acc over panels (deep-T / G-B).

        Pinned to Ascend910B (L0c/Acc = 128 KB): for ``(A@B)@D`` with M=N=256, K1=256,
        K2=512 the solver tiles C's output into ``[128,256]`` regions, so the naive per-tile
        T_band would be ``[128,512]`` fp32 = 256 KB > 128 KB L0c. Deep-T tiles K2 into
        ``512/256 = 2`` panels: each MM1 produces a ``[128,256]`` T_panel (128 KB, fits L0c)
        and MM2 folds ``T_panel @ D[panel]`` into the output tile via ``tensor.matmul_acc`` —
        so the full ``[128,512]`` intermediate never materializes. Verified numerically via
        ``torch_codegen`` (the tensor-level interpreter); the full cube LOWERING of a fused
        chain remains xfail on #1908, so this does NOT run the Default pipeline.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def ch(
                self,
                a: pl.Tensor[[256, 256], pl.FP32],
                b: pl.Tensor[[256, 512], pl.FP32],
                d: pl.Tensor[[512, 256], pl.FP32],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                t: pl.Tensor[[256, 512], pl.FP32] = pl.matmul(a, b)
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(t, d)
                return c

        after = passes.auto_fuse()(Prog)
        body = next(f for _, f in after.functions.items() if f.name == "ch").as_python()
        # Deep-T fired: spatially tiled (not the whole-output CORE_GROUP fallback), MM2 folds
        # panels via matmul_acc, and the full [128,512] T_band never materializes (K2 is tiled
        # into [128,256] panels that each fit L0c).
        assert "pl.at(level=pl.Level.CORE_GROUP" not in body and "pl.spmd(" in body
        assert "pl.tensor.matmul_acc(" in body
        assert "[128, 512]" not in body  # T is panelled; the full [h,K2] intermediate is gone

        # Numeric faithfulness of the panelled matmul_acc chain (tensor-level interpreter).
        namespace: dict = {}
        exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)  # noqa: S102
        torch.manual_seed(0)
        a = torch.randn(256, 256, dtype=torch.float32) * 0.03
        b = torch.randn(256, 512, dtype=torch.float32) * 0.03
        d = torch.randn(512, 256, dtype=torch.float32) * 0.03
        out = namespace["ch"](a, b, d)
        ref = (a @ b) @ d
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), f"max abs diff {(out - ref).abs().max().item():.3e}"

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

    def test_streamed_reduction_lowers_and_is_correct(self, ascend_backend, monkeypatch):
        """A reduction whose reduced axis is too large to fit one UB tile must STREAM (P1): SPMD
        over the FREE axis + an inner chunk-accumulation loop over the pinned axis, persisting only
        the small [.,1]/[1,.] accumulator (the big [.,chunk] slices are transient). Without streaming
        the [IM,w] / [h,IN] pinned tile overflows UB. Asserts (a) a huge-axis reduction lowers
        through AllocateMemoryAddr without a Vec overflow (streaming fired), and (b) a smaller
        streamed reduction is NUMERICALLY exact for BOTH merges — add (col_sum) and max on SIGNED
        data (row_max) — via torch_codegen. On-core accumulation, so exact (no reassociation).
        (Behind the generic-emit flag, where streaming lives.)
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        # (a) huge reduced axis lowers via streaming — col_sum[16384,128] = 8 MB >> 184 KB UB.
        @pl.program
        class Big:
            @pl.function(attrs={"auto_fuse": True})
            def cs(self, a: pl.Tensor[[16384, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
                c: pl.Tensor[[1, 128], pl.FP32] = pl.col_sum(a)
                return c

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Big)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]

        # (b) numerical exactness of the streamed emit (executes the actual streamed IR on CPU).
        def _numeric(program, entry, x, ref):
            ns: dict = {}
            exec(torch_codegen(passes.auto_fuse()(program), run_all_spmd_blocks=True), ns)  # noqa: S102
            got = ns[entry](x)
            diff = (got - ref).abs().max().item()
            assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), f"{entry}: max abs diff {diff:.3e}"

        @pl.program
        class ColSum:  # reduce M, add-merge; [256,128] streams (2*256*128*4 = 256 KB > 184 KB)
            @pl.function(attrs={"auto_fuse": True})
            def cs(self, a: pl.Tensor[[256, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
                c: pl.Tensor[[1, 128], pl.FP32] = pl.col_sum(a)
                return c

        @pl.program
        class RowMax:  # reduce N, max-merge on SIGNED data (proves mask != zero-fill)
            @pl.function(attrs={"auto_fuse": True})
            def rm(self, a: pl.Tensor[[128, 256], pl.FP32]) -> pl.Tensor[[128, 1], pl.FP32]:
                c: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(a)
                return c

        @pl.program
        class ColMax:  # reduce M, MAX-merge on SIGNED data; [256,128] streams — guards merge-op
            @pl.function(attrs={"auto_fuse": True})
            def cm(self, a: pl.Tensor[[256, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
                c: pl.Tensor[[1, 128], pl.FP32] = pl.col_max(a)
                return c

        @pl.program
        class RowSum:  # reduce N, add-merge; bare row reduction — guards the reduced AXIS
            @pl.function(attrs={"auto_fuse": True})
            def rs(self, a: pl.Tensor[[128, 256], pl.FP32]) -> pl.Tensor[[128, 1], pl.FP32]:
                c: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(a)
                return c

        x_cs = torch.arange(256 * 128, dtype=torch.float32).reshape(256, 128) * 0.01
        _numeric(ColSum, "cs", x_cs, x_cs.sum(dim=0, keepdim=True))
        x_rm = (torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256) % 97) - 48.0
        _numeric(RowMax, "rm", x_rm, x_rm.max(dim=1, keepdim=True).values)
        # col_max: MAX-merge (not add) on signed data — a sum would flip the sign of the answer.
        x_cm = (torch.arange(256 * 128, dtype=torch.float32).reshape(256, 128) % 97) - 48.0
        _numeric(ColMax, "cm", x_cm, x_cm.max(dim=0, keepdim=True).values)
        # bare row_sum: reduces N (width), not M — a wrong-axis reduction gives [128,1] of Σ over M.
        x_rs = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256) * 0.01
        _numeric(RowSum, "rs", x_rs, x_rs.sum(dim=1, keepdim=True))

    def test_streamed_reduction_apply_p2(self, ascend_backend, monkeypatch):
        """P2: a POINTWISE sink consuming a single reduction (x - row_max(x), rmsnorm) whose output
        SPANS the reduced axis. Two-pass stream: pass 0 accumulates the reduction; pass 1 re-streams
        the reduced axis, recomputes the pointwise cone with the finalized reduction substituted, and
        assembles each output chunk — the final apply CHUNKS the reduced axis (else the full-shape
        output re-overflows UB, review R3 #2). Asserts a huge case lowers and two shapes are exact
        (incl. a pre-reduction pointwise x*x recomputed per chunk).
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        # (a) huge full-shape output over the reduced axis lowers via the chunked final apply.
        @pl.program
        class Big:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP32]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                return pl.sub(x, m)

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Big)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]

        def _numeric(program, entry, x, ref):
            ns: dict = {}
            exec(torch_codegen(passes.auto_fuse()(program), run_all_spmd_blocks=True), ns)  # noqa: S102
            got = ns[entry](x)
            diff = (got - ref).abs().max().item()
            assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), f"{entry}: max abs diff {diff:.3e}"

        # The numerical cases use [128, 16384] — large enough that the solver FUSES the
        # reduction+pointwise into one group AND the reduced axis overflows UB, so the STREAMED
        # 2-pass P2 apply (not the non-streaming path) is what is exercised.
        @pl.program
        class SubMax:  # reduction -> pointwise (max-merge, signed data)
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP32]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                return pl.sub(x, m)

        @pl.program
        class RmsLike:  # pre-reduction pointwise (x*x) recomputed per chunk + reduction -> pointwise
            @pl.function(attrs={"auto_fuse": True})
            def rms(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP32]:
                sq: pl.Tensor[[128, 16384], pl.FP32] = pl.mul(x, x)
                ms: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(sq)
                return pl.mul(x, ms)

        x_sm = (torch.arange(128 * 16384, dtype=torch.float32).reshape(128, 16384) % 91) - 45.0
        _numeric(SubMax, "sm", x_sm, x_sm - x_sm.max(dim=1, keepdim=True).values)
        x_rms = (torch.arange(128 * 16384, dtype=torch.float32).reshape(128, 16384) % 13) * 0.1 + 0.05
        _numeric(RmsLike, "rms", x_rms, x_rms * (x_rms * x_rms).sum(dim=1, keepdim=True))

    def test_inline_return_multi_reduction_lowers_and_is_correct(self, ascend_backend, monkeypatch):
        """A multi-reduction group (softmax = row_max + row_sum; layernorm = two row_sums) written
        with a DIRECT ``return pl.op(...)`` — the idiomatic form. Two guards on one path:

        1. Inline-return hoisting: the returned op has no SSA name, so the solver-graph builder used
           to MISS it (it registers only ``var = <call>`` ops). Its operands then looked group-
           internal and the emit dropped them, leaving the raw return referencing an unexposed
           intermediate (``return pl.mul(xc, iv)`` where ``xc`` is a fused-group intermediate ->
           dangling ``xc``). AutoFuse now hoists ``return <call>`` to ``_ret = <call>; return _ret``
           so every op is visible to the partitioner (BUG-LN-2 regression).
        2. G1 cut of the un-streamable multi-reduction group into >=2 buildable kernels, and the
           cross-group intermediate threaded correctly end-to-end (numerically exact).
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        eps = 1e-5

        @pl.program
        class Softmax:  # row_max + row_sum, direct return of the final div
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP32]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                s: pl.Tensor[[128, 16384], pl.FP32] = pl.sub(x, m)
                e: pl.Tensor[[128, 16384], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                return pl.div(e, d)

        @pl.program
        class LayerNorm:  # two row_sums; final `mul(xc, iv)` returned inline consumes intermediate xc
            @pl.function(attrs={"auto_fuse": True})
            def ln(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP32]:
                sx: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(x)
                mu: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sx, 1.0 / 16384)
                xc: pl.Tensor[[128, 16384], pl.FP32] = pl.sub(x, mu)
                sq: pl.Tensor[[128, 16384], pl.FP32] = pl.mul(xc, xc)
                sv: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(sq)
                var: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sv, 1.0 / 16384)
                ve: pl.Tensor[[128, 1], pl.FP32] = pl.add(var, eps)
                iv: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(ve)
                return pl.mul(xc, iv)

        # (1) Structural: G1 cuts each un-streamable multi-reduction group into >=2 buildable
        # kernels (was a hard AllocateMemoryAddr overflow before G1; a dangling xc before hoisting).
        for prog, name in ((Softmax, "sm"), (LayerNorm, "ln")):
            out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(prog)
            incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
            assert len(incores) >= 2, f"{name}: expected G1 to cut into >=2 kernels, got {incores}"

        # (2) Numerical: the inline-returned op is emitted (not dangling) and the cross-group
        # intermediate is threaded correctly. Bounded data keeps exp() in range.
        def _numeric(program, entry, x, ref):
            ns: dict = {}
            exec(torch_codegen(passes.auto_fuse()(program), run_all_spmd_blocks=True), ns)  # noqa: S102
            got = ns[entry](x)
            diff = (got - ref).abs().max().item()
            assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), f"{entry}: max abs diff {diff:.3e}"

        x = (torch.arange(128 * 16384, dtype=torch.float32).reshape(128, 16384) % 13) * 0.1 - 0.6
        _numeric(Softmax, "sm", x, torch.softmax(x, dim=1))
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        _numeric(LayerNorm, "ln", x, xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps))

        # Control: a small multi-reduction (reduced axis fits UB) stays FUSED into ONE kernel.
        @pl.program
        class SmallSoftmax:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[256, 128], pl.FP32]) -> pl.Tensor[[256, 128], pl.FP32]:
                m: pl.Tensor[[256, 1], pl.FP32] = pl.row_max(x)
                s: pl.Tensor[[256, 128], pl.FP32] = pl.sub(x, m)
                e: pl.Tensor[[256, 128], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(e)
                return pl.div(e, d)

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(SmallSoftmax)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, f"small softmax should stay fused into one kernel, got {incores}"
        xs = (torch.arange(256 * 128, dtype=torch.float32).reshape(256, 128) % 13) * 0.1 - 0.6
        _numeric(SmallSoftmax, "sm", xs, torch.softmax(xs, dim=1))

    def test_multi_reduction_g1_threshold_no_overflow(self, ascend_backend, monkeypatch):
        """BUG-G1THRESH regression. Feasibility (vector_peak_ub) and the emit's materialize-vs-stream
        trigger both used UNPADDED tile bytes, while the emit allocates DMA-block-padded tiles. A thin
        free axis (softmax/layernorm M-tile of 3 -> 8 for fp32, ~2.7x) was under-counted, so the mid
        sizes N=4096/8192 looked UB-materializable, fused into one multi-reduction group, and
        overflowed AllocateMemoryAddr. Both sites now count the padded footprint (Problem.
        vec_dma_align_bytes), so the group is correctly detected as over-UB -> G1 cuts it. Guards the
        two sizes that slipped between the fused-small (N<=2048) and streamed-large (N=16384) cases.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        def softmax(n):
            @pl.program
            class P:
                @pl.function(attrs={"auto_fuse": True})
                def sm(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                    m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                    s: pl.Tensor[[128, n], pl.FP32] = pl.sub(x, m)
                    e: pl.Tensor[[128, n], pl.FP32] = pl.exp(s)
                    d: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                    return pl.div(e, d)

            return P

        def layernorm(n):
            @pl.program
            class P:
                @pl.function(attrs={"auto_fuse": True})
                def ln(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                    sx: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(x)
                    mu: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sx, 1.0 / n)
                    xc: pl.Tensor[[128, n], pl.FP32] = pl.sub(x, mu)
                    sq: pl.Tensor[[128, n], pl.FP32] = pl.mul(xc, xc)
                    sv: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(sq)
                    var: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sv, 1.0 / n)
                    ve: pl.Tensor[[128, 1], pl.FP32] = pl.add(var, 1e-5)
                    iv: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(ve)
                    return pl.mul(xc, iv)

            return P

        # Compile each threshold size: G1 must cut into >=2 kernels (was AllocateMemoryAddr overflow).
        for n in (4096, 8192):
            for mk, name in ((softmax, f"softmax[128,{n}]"), (layernorm, f"layernorm[128,{n}]")):
                out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(mk(n))
                incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
                assert len(incores) >= 2, f"{name}: expected G1 cut into >=2 kernels, got {incores}"

        # And the streamed result at the threshold is numerically exact.
        x = (torch.arange(128 * 4096, dtype=torch.float32).reshape(128, 4096) % 13) * 0.1 - 0.6
        ns: dict = {}
        exec(torch_codegen(passes.auto_fuse()(softmax(4096)), run_all_spmd_blocks=True), ns)  # noqa: S102
        got = ns["sm"](x)
        ref = torch.softmax(x, dim=1)
        assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), f"diff {(got - ref).abs().max().item():.3e}"

    def test_broadcast_operand_fuses_and_is_correct(self, ascend_backend, monkeypatch):
        """G4: a BROADCAST external operand — one axis is the full extent, the other is 1 (the FIXED_1
        read-in-full role, contract §3/A3) — now fuses instead of declining to the legacy tiler. Covers
        the M-broadcast `[1,N]` (bias-add, ubiquitous in FFN/attention), the N-broadcast `[M,1]`
        (per-row scale), a fused chain mixing both, and a P2 reduction group that takes an external
        `[M,1]` stat (the shape a G1/G3 softmax cut produces — this is what unblocks G3's buildable path).
        emit_strip slices a broadcast operand `[aM==1?1:sh, aN==1?1:sw]` at `[aM==1?0:smi, aN==1?0:sni]`
        and the op replay re-infers the broadcast.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        m_, n_ = 128, 512

        def _check(program, entry, args, ref, want_incores=1):
            out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
            incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
            assert len(incores) == want_incores, f"{entry}: expected {want_incores} incore(s), got {incores}"
            ns: dict = {}
            exec(torch_codegen(passes.auto_fuse()(program), run_all_spmd_blocks=True), ns)  # noqa: S102
            got = ns[entry](*args)
            diff = (got - ref).abs().max().item()
            assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), f"{entry}: max abs diff {diff:.3e}"

        @pl.program
        class BiasAdd:  # M-broadcast [1,N]
            @pl.function(attrs={"auto_fuse": True})
            def ba(self, x: pl.Tensor[[m_, n_], pl.FP32], b: pl.Tensor[[1, n_], pl.FP32]) -> pl.Tensor[[m_, n_], pl.FP32]:
                return pl.add(x, b)

        @pl.program
        class RowScale:  # N-broadcast [M,1]
            @pl.function(attrs={"auto_fuse": True})
            def rs(self, x: pl.Tensor[[m_, n_], pl.FP32], s: pl.Tensor[[m_, 1], pl.FP32]) -> pl.Tensor[[m_, n_], pl.FP32]:
                return pl.mul(x, s)

        @pl.program
        class Chain:  # both broadcasts fused + a unary
            @pl.function(attrs={"auto_fuse": True})
            def ch(self, x: pl.Tensor[[m_, n_], pl.FP32], b: pl.Tensor[[1, n_], pl.FP32], s: pl.Tensor[[m_, 1], pl.FP32]) -> pl.Tensor[[m_, n_], pl.FP32]:
                t: pl.Tensor[[m_, n_], pl.FP32] = pl.add(x, b)
                u: pl.Tensor[[m_, n_], pl.FP32] = pl.mul(t, s)
                return pl.exp(u)

        @pl.program
        class P2Bcast:  # reduction group taking an external [M,1] stat (a G1/G3 softmax cut piece)
            @pl.function(attrs={"auto_fuse": True})
            def sm2(self, x: pl.Tensor[[m_, n_], pl.FP32], mstat: pl.Tensor[[m_, 1], pl.FP32]) -> pl.Tensor[[m_, n_], pl.FP32]:
                s: pl.Tensor[[m_, n_], pl.FP32] = pl.sub(x, mstat)
                e: pl.Tensor[[m_, n_], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[m_, 1], pl.FP32] = pl.row_sum(e)
                return pl.div(e, d)

        x = torch.randn(m_, n_)
        b = torch.randn(1, n_)
        s = torch.randn(m_, 1)
        _check(BiasAdd, "ba", (x, b), x + b)
        _check(RowScale, "rs", (x, s), x * s)
        _check(Chain, "ch", (x, b, s), torch.exp((x + b) * s))
        mstat = torch.randn(m_, 1)
        e = torch.exp(x - mstat)
        _check(P2Bcast, "sm2", (x, mstat), e / e.sum(dim=1, keepdim=True))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
