# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AutoFuse pass (PTO-Fusebox-driven fusion + IR emit).

AutoFuse intercepts the raw tensor-op DAG of a function marked
``attrs={"auto_fuse": True}``, runs PTO Fusebox to choose a fusion
partition + tile, and rewrites the body to realize that decision: a matmul or a
run of fused pointwise ops becomes the solver's ``[w,h]`` output tiling distributed
across cores (chunked-parallel ``AutoInCore`` scopes — k-pipelined per tile for
matmul, the whole op chain replayed per tile with intermediates on-chip for
pointwise), and two chained matmuls the solver groups together likewise fuse into
one kernel. The Outline/Convert/Tile pipeline then lowers each scope to a cube
(AIC) or vector (AIV) kernel.
"""

import json
import os
import re
import subprocess
import sys
import textwrap

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

# These tests were rewritten against the grounded SPMD emit: the pre-grounding cost model
# plus the #1895 auto_chunk removal changed the solver's decisions and migrated the emit
# onto SPMD. Plans with split-K use a tiled zero-seed + atomic-add merge; plans without it
# write their spatial tiles directly. The old pl.auto_chunk / matmul_acc assertions were
# stale. All cases are now re-derived and live. The shared L0 planner and
# output-tile-outer replay also remove the former chained-matmul packing xfail:
# producer and consumer child plans are lowered sequentially instead of keeping
# both sets of L0 ping-pong buffers live at once.


class TestAutoFuse:
    """AutoFuse solver-driven fusion + emit."""

    def test_single_matmul_emits_spmd_tiled_kernel(self, ascend_backend):
        """A lone 64x64 matmul becomes the grounded solver's SPMD output tiling.

        Pinned to Ascend910B (``ascend_backend``): the solver's tile/split decision is
        backend-specific. The overlap-fidelity gate rejects a one-iteration K pipeline, so
        the grounded 910B model now selects tile=32x32x64, split=1. Its balanced 2x3 spatial
        grid is emitted as one six-block SPMD loop. With no cross-core K merge there is no
        zero-seed kernel and each block stores its complete output tile directly.
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
        # One flat SPMD loop: six spatial tasks, each with the full K contraction.
        assert "pl.spmd(6" in body
        assert body.count("pl.spmd(") == 1
        assert "pl.tensor.full(" not in body
        # Per-block body: operand slices, a tiled matmul, and a direct non-atomic store.
        assert "pl.tensor.slice(" in body
        assert "pl.tensor.matmul(" in body
        assert "atomic=pl.AtomicType.Add" not in body

    def test_single_matmul_emit_is_numerically_correct(self, ascend_backend):
        """The SPMD-tiled emit computes the same result as a plain matmul.

        ``torch_codegen(..., run_all_spmd_blocks=True)`` runs all spatial blocks serially,
        so the generated function reproduces the FULL 64x64 result (not just block 0). It
        must match ``torch.matmul`` to fp32 tolerance, verifying that slicing and stores
        are numerically faithful, not only structurally present.
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

        The overlap-fidelity gate selects split=1 for this small contraction, so the
        function lowers to one cube (AIC) kernel. No vector zero-seed kernel or cross-core
        atomic merge is needed.
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

        # The host `mm` becomes an Orchestration function driving one AIC SPMD kernel.
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        by_type: dict[str, list] = {}
        for f in incores:
            by_type.setdefault(str(f.func_type), []).append(f)
        assert len(by_type.get("FunctionType.AIC", [])) == 1, [f.name for f in incores]
        assert len(by_type.get("FunctionType.AIV", [])) == 0, [f.name for f in incores]

        cube = by_type["FunctionType.AIC"][0]
        mlir = codegen.PTOCodegen().generate(ir.Program([cube], cube.name, cube.span))
        assert "pto.kernel_kind" in mlir
        assert "cube" in mlir  # a pure matmul lowers to a cube kernel
        assert "pto.tload" in mlir and "pto.tmatmul" in mlir
        # SPMD-distributed spatial tiling: each block indexes off spmd_block_idx and writes
        # a complete tile. A split-K atomic merge would violate the selected split=1 plan.
        assert "spmd_block_idx" in mlir
        assert "atomic_add" not in mlir

    def test_large_matmul_ragged_grid_avoids_fictional_split_seed(self, ascend_backend, monkeypatch):
        """A ragged cube grid cannot be combined with split-K atomic ownership.

        Ceil-and-clamp spatial tiles overlap at the output edge.  They are valid for a
        split=1 lone matmul because the overlapping writes are idempotent, but atomic
        split-K would assign those edge elements to multiple spatial regions.  The
        buildable model must therefore choose a split=1 plan and emit no vector seed.
        The separate 512x512 regression below covers a legal uniform split-K seed.
        """

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")

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

        planned = passes.auto_fuse()(Prog)
        planned_body = next(f for _, f in planned.functions.items() if f.name == "mm").as_python()
        assert "pl.spmd(" in planned_body
        assert "pl.min(" in planned_body
        assert "__autofuse_l0_matmul_plan" in planned_body
        assert "AtomicType" not in planned_body and "pl.tensor.full(" not in planned_body

        # Must lower end-to-end as one cube kernel without inventing a split seed.
        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        by_type: dict[str, list] = {}
        for f in incores:
            by_type.setdefault(str(f.func_type), []).append(f)
        assert len(by_type.get("FunctionType.AIC", [])) == 1, [f.name for f in incores]
        assert len(by_type.get("FunctionType.AIV", [])) == 0, [f.name for f in incores]

        cube = by_type["FunctionType.AIC"][0]
        cube_mlir = codegen.PTOCodegen().generate(ir.Program([cube], cube.name, cube.span))
        assert "cube" in cube_mlir and "spmd_block_idx" in cube_mlir
        assert "atomic_add" not in cube_mlir

    def test_cube_l0_cost_modes_separate_residency_intent_from_exact_plan(self):
        """Analytic costing delegates L0 geometry; exact costing pins it.

        Both modes need the semantic Acc/L1/GM handoff because AutoFuse owns
        the outer GM-to-L1 window loop. Only exact/co-optimized costing may
        serialize the detailed L0 geometry that AutoTileMatmulL0 must replay.
        """
        script = textwrap.dedent(
            """
            import os

            import pypto.language as pl
            from pypto import backend, ir, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)
            os.environ["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
            os.environ["PYPTO_AUTOFUSE_FORCE_PLAN"] = "64,64,1,1,1"
            os.environ.pop("PYPTO_AUTOFUSE_EXACT_L0_COST", None)

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

            analytic = passes.auto_fuse()(Prog)
            analytic_body = next(
                f for _, f in analytic.functions.items() if f.name == "mm"
            ).as_python()
            assert "__autofuse_l0_output_target" in analytic_body
            assert "__autofuse_l0_matmul_plan" not in analytic_body

            lowered = PassManager.get_strategy(
                OptimizationStrategy.Default
            ).run_passes(Prog)
            aic = [
                f for _, f in lowered.functions.items()
                if str(f.func_type) == "FunctionType.AIC"
            ]
            assert len(aic) == 1
            lowered_body = aic[0].as_python()
            assert "__autofuse_l0_output_target" not in lowered_body
            assert "__autofuse_l0_matmul_plan" not in lowered_body

            os.environ["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
            exact = passes.auto_fuse()(Prog)
            exact_body = next(
                f for _, f in exact.functions.items() if f.name == "mm"
            ).as_python()
            assert "__autofuse_l0_output_target" in exact_body
            assert "__autofuse_l0_matmul_plan" in exact_body
            """
        )
        env = os.environ.copy()
        env.pop("PYPTO_AUTOFUSE_EXACT_L0_COST", None)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_ragged_cube_region_subdivides_l0c_exactly(self):
        """A ragged solver work unit larger than L0c uses exact, non-overlapping subtiles.

        ``PYPTO_AUTOFUSE_FORCE_PLAN`` is cached on first use, so this experiment must run
        in a fresh process. The forced 144x272 solver tile clamps to the full 130x260
        output. That region is just over L0c capacity, so it must split into one 128x256
        base accumulator plus exact ragged edge tiles (down to 2x4), instead of crashing,
        overlapping atomic regions, or degenerating to one-element tiles. Numeric execution
        verifies complete coverage.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager
            from pypto.debug import torch_codegen

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[130, 64], pl.FP32],
                    b: pl.Tensor[[64, 260], pl.FP32],
                ) -> pl.Tensor[[130, 260], pl.FP32]:
                    c: pl.Tensor[[130, 260], pl.FP32] = pl.matmul(a, b)
                    return c

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "mm").as_python()
            assert body.count("pl.tensor.matmul(") == 4
            assert "[128, 256]" in body
            assert "[2, 4]" in body

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(130, 64, dtype=torch.float32)
            b = torch.randn(64, 260, dtype=torch.float32)
            out = namespace["mm"](a, b)
            assert torch.allclose(out, a @ b, rtol=1e-4, atol=1e-4)
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "272,144,1,1,1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_generic_cube_rejects_subfractal_oversized_region(self):
        """Analytic and exact modes reject the same unbuildable L0 edge.

        The forced outer region is logically ``130x260`` but its aligned
        ``144x272`` request would leave ``2x4`` L0 edge variants. The shared L0
        plan cannot yet distinguish their physical 16x16 allocation from valid
        extents, so compiler-mode costing must not expose the forced candidate.
        Both modes fall back to their buildable natural clamped-overlap plan.
        """

        script = textwrap.dedent(
            """
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[130, 64], pl.FP32],
                    b: pl.Tensor[[64, 260], pl.FP32],
                ) -> pl.Tensor[[130, 260], pl.FP32]:
                    c: pl.Tensor[[130, 260], pl.FP32] = pl.matmul(a, b)
                    return c

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "mm").as_python()
            assert "pl.spmd(" in body
            """
        )
        for exact in (False, True):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(os.path.abspath("python"))
            env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
            env["PYPTO_AUTOFUSE_MIXED"] = "0"
            env["PYPTO_AUTOFUSE_STRICT"] = "1"
            env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "272,144,1,1,1"
            env["PYPTO_LOG_LEVEL"] = "info"
            if exact:
                env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
            else:
                env.pop("PYPTO_AUTOFUSE_EXACT_L0_COST", None)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + "\n" + proc.stderr
            assert proc.returncode == 0, output
            assert "matched NO feasible candidate" in output
            assert "spatial_policy=clamped_overlap" in output

    def test_nonuniform_matmul_tiles_via_ceil_clamp_grid(self, ascend_backend):
        """A matmul whose solver tile does NOT divide the output tiles via a ceil+clamp
        SPMD grid (the G-A fix for the "matmul-tiling gap").

        Pinned to Ascend910B: for ``[272,272]`` the grounded solver picks a non-uniform
        ``80x144`` spatial grid (``272 % 80 != 0``, ``272 % 144 != 0``, parts 2x4). Instead
        of declining to one untiled InCore scope (the old fallback), the emitter realizes a
        ``ceil(272/144) x ceil(272/80) = 2x4 = 8``-block ``pl.spmd(8)`` grid whose per-block
        offsets are CLAMPED in-bounds (``pl.min(mt*144, 128)`` / ``pl.min(nt*80, 192)``). Every
        block owns a ``[144,80]`` L1 region, explicitly subdivided at the grounded L0-M
        bound into ``[128,80]`` and ``[16,80]`` accumulators. Ragged spatial blocks OVERLAP
        the previous block, and the NON-atomic stores recompute the overlap identically
        (idempotent), so the result is exact. split=1 here, so there is no zero-seed or
        ``atomic`` merge (split-K stays divisor-only).
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
        # Per block: the [144,80] region is two grounded L0-M subtiles, with clamped
        # spatial offsets (pl.min) and non-atomic stores.
        assert "pl.tensor.slice(" in body and body.count("pl.tensor.matmul(") == 2
        assert body.count("pl.tensor.assemble(") == 2
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

        ``CubeSchedulePlan`` separates the L1-resident window from the actual per-load chunk:
        two chunks must fit together, and all full chunks share one stage-2 ring. The K=0
        iteration initializes the carry with ``matmul``; later iterations use ``matmul_acc``.
        When the load chunk does not divide the per-split contraction, one serial
        ``matmul_acc`` tail folds in the remainder. Pinned to Ascend910B:
        ``[64,5040]@[5040,256]`` splits K into 7 slices of 720; its 336-wide L1 window selects a
        160-wide load, emitted as pipeline ``[0,640)`` + tail 80.
        ``[64,4096]@[4096,256]`` is the exact-division control (slice 512, load 128). Both must
        match ``torch.matmul``; the ragged path must not silently drop the tail contribution.
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

        # Ragged peel: window 336, actual load 160 -> four full strips in one stage ring through
        # 640 + an 80-wide tail. The loop-carried create is only an accumulator identity; K=0
        # initializes it with a real matmul before any matmul_acc can observe it.
        peel_body = _fuse_body(64, 5040, 256)
        assert "pl.pipeline(0, 640, 160" in peel_body
        assert "pl.tensor.matmul_acc(" in peel_body  # loop accumulate + the peel tail fold
        assert "[64, 80]" in peel_body  # the ragged K-tail slice (720 - 4*160 = 80)
        assert "_acc_init" in peel_body

        # Exact-division control: all four 128-wide strips share the ring, no tail.
        div_body = _fuse_body(64, 4096, 256)
        assert "pl.pipeline(0, 512, 128" in div_body
        assert "_a_tl" not in div_body and "_b_tl" not in div_body

    def test_single_pointwise_tiles_across_vector_cores(self, ascend_backend):
        """A large pointwise op is tiled into the solver's `[w,h]` regions and
        distributed across the vector cores, lowering to a vector (AIV) kernel.

        Pinned to Ascend910B (48 vector cores): for `[4096,384]` the grounded solver picks
        a `[512,64]` tile, so the output tiles into 8x6 = 48 disjoint regions — one per AIV
        core — emitted as a single flat `pl.spmd(48)` loop. No split-K (pointwise has no
        contraction), so no zero-seed and a plain (non-atomic) per-tile `assemble`. (This is
        the legacy tiler; the C3 per-task-overhead term that prefers fewer tiles is gated on
        the generic emit — see test_c3_per_task_overhead_prefers_fewer_tiles.)
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

    def test_chained_matmul_fuses_to_one_kernel(self, ascend_backend, monkeypatch):
        """The natural buildable cube plan fuses a two-matmul chain across AIC cores.

        One AIC SPMD kernel recursively produces the requested ``A@B`` regions into an
        L1 scratch and immediately consumes them in the sink.  Each output subtile stays
        in L0C across its complete K-window sequence and drains once.  No intermediate
        reaches GM and a split=1 winner needs neither a vector seed nor atomic stores.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.BF16],
                b: pl.Tensor[[256, 128], pl.BF16],
                d: pl.Tensor[[128, 256], pl.BF16],
            ) -> pl.Tensor[[128, 256], pl.BF16]:
                t: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(a, b)
                c: pl.Tensor[[128, 256], pl.BF16] = pl.matmul(t, d)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "chain").as_python()
        assert "pl.at(level=pl.Level.CORE_GROUP" not in body
        assert body.count("pl.spmd(") == 1
        assert body.count("pl.tensor.matmul(") >= 2
        assert "pl.tensor.create_l1(" in body
        assert body.count("pl.tensor.assemble(") >= 2  # internal L1 plus root GM drains
        assert "AtomicType.Add" not in body
        assert "pl.Out[" in body

    def test_produced_rhs_cube_plan_emits_recursive_tree(self):
        """A non-square both-input-produced root replays one role-aware cube plan.

        The fixed plan uses four spatial regions and five root-K shares. Its left
        producer materializes ``SpatialM x ParallelK`` regions while its right
        producer materializes ``ParallelK x SpatialN`` regions. Strict mode proves
        the plan does not fall through to dependency-ordered standalone matmuls.

        ``PYPTO_AUTOFUSE_FORCE_PLAN`` is process-cached, so run in a fresh process.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def tree(
                    self,
                    a: pl.Tensor[[32, 48], pl.BF16],
                    b: pl.Tensor[[48, 80], pl.BF16],
                    c: pl.Tensor[[80, 64], pl.BF16],
                    d: pl.Tensor[[64, 96], pl.BF16],
                ) -> pl.Tensor[[32, 96], pl.BF16]:
                    lhs: pl.Tensor[[32, 80], pl.BF16] = pl.matmul(a, b)
                    rhs: pl.Tensor[[80, 96], pl.BF16] = pl.matmul(c, d)
                    out: pl.Tensor[[32, 96], pl.BF16] = pl.matmul(lhs, rhs)
                    return out

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "tree").as_python()
            assert "pl.spmd(20" in body  # 2 M regions * 2 N regions * 5 K shares
            assert "pl.spmd(4" in body   # one disjoint zero seed per spatial region
            assert body.count("pl.tensor.matmul(") == 3
            assert body.count("pl.tensor.matmul_acc(") >= 2
            assert "pl.at(level=pl.Level.CORE_GROUP" not in body

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(32, 48, dtype=torch.bfloat16) * 0.05
            b = torch.randn(48, 80, dtype=torch.bfloat16) * 0.05
            c = torch.randn(80, 64, dtype=torch.bfloat16) * 0.05
            d = torch.randn(64, 96, dtype=torch.bfloat16) * 0.05
            expected = (a @ b) @ (c @ d)
            actual = namespace["tree"](a, b, c, d)
            assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3), (
                actual - expected
            ).abs().max().item()
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "48,16,5,2,2"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_cube_plan_recomputes_fanout_for_distinct_roles(self):
        """A fan-out producer is shared by identity and recomputed by requested role.

        ``T`` feeds one root as its LHS and another as its RHS. For a 2x2 spatial
        grid, those consumers need different regions: ``[32,64]`` row bands and
        ``[64,32]`` column bands. The plan therefore contains two instances of
        ``T = A@B`` and two boundary roots in one four-work-unit SPMD kernel.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def fanout(
                    self,
                    a: pl.Tensor[[64, 64], pl.BF16],
                    b: pl.Tensor[[64, 64], pl.BF16],
                    c: pl.Tensor[[64, 64], pl.BF16],
                    d: pl.Tensor[[64, 64], pl.BF16],
                ) -> tuple[
                    pl.Tensor[[64, 64], pl.BF16],
                    pl.Tensor[[64, 64], pl.BF16],
                ]:
                    shared: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(a, b)
                    left: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(shared, c)
                    right: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(d, shared)
                    return left, right

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "fanout").as_python()
            assert body.count("pl.spmd(") == 1 and "pl.spmd(4" in body
            assert body.count("pl.tensor.matmul(") == 4
            assert body.count("pl.Out[") == 2

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(1)
            a, b, c, d = (
                torch.randn(64, 64, dtype=torch.bfloat16) * 0.05 for _ in range(4)
            )
            shared = a @ b
            actual_left, actual_right = namespace["fanout"](a, b, c, d)
            assert torch.allclose(actual_left, shared @ c, rtol=1e-3, atol=1e-3)
            assert torch.allclose(actual_right, d @ shared, rtol=1e-3, atol=1e-3)

            @pl.program
            class DeepProg:
                @pl.function(attrs={"auto_fuse": True})
                def deep(
                    self,
                    a: pl.Tensor[[64, 64], pl.BF16],
                    b: pl.Tensor[[64, 64], pl.BF16],
                    c: pl.Tensor[[64, 64], pl.BF16],
                    d: pl.Tensor[[64, 64], pl.BF16],
                    e: pl.Tensor[[64, 64], pl.BF16],
                ) -> pl.Tensor[[64, 64], pl.BF16]:
                    t0: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(a, b)
                    t1: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(t0, c)
                    t2: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(t1, d)
                    out: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(t2, e)
                    return out

            deep_after = passes.auto_fuse()(DeepProg)
            deep_body = next(
                f for _, f in deep_after.functions.items() if f.name == "deep"
            ).as_python()
            assert deep_body.count("pl.spmd(") == 1 and "pl.spmd(4" in deep_body
            assert deep_body.count("pl.tensor.matmul(") == 4

            deep_namespace = {}
            exec(torch_codegen(deep_after, run_all_spmd_blocks=True), deep_namespace)
            torch.manual_seed(2)
            values = [torch.randn(64, 64, dtype=torch.bfloat16) * 0.05 for _ in range(5)]
            expected = (((values[0] @ values[1]) @ values[2]) @ values[3]) @ values[4]
            actual = deep_namespace["deep"](*values)
            assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "32,32,1,2,2"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_chained_matmul_lowers_to_cube_kernel(self, ascend_backend, monkeypatch):
        """The fused chain lowers through the Default pipeline to a single cube kernel.

        Output-tile-outer replay keeps only one shared-planner child live in L0 at a
        time.  This removes the former #1908 false overflow from simultaneously packing
        producer and consumer ping-pong buffers while retaining one fused AIC kernel.
        """

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.BF16],
                b: pl.Tensor[[256, 128], pl.BF16],
                d: pl.Tensor[[128, 256], pl.BF16],
            ) -> pl.Tensor[[128, 256], pl.BF16]:
                t: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(a, b)
                c: pl.Tensor[[128, 256], pl.BF16] = pl.matmul(t, d)
                return c

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        # The fused chain is ONE cube kernel, not two separate matmul kernels.
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        assert str(incores[0].func_type) == "FunctionType.AIC"
        mlir = codegen.PTOCodegen().generate(ir.Program([incores[0]], incores[0].name, incores[0].span))
        assert "cube" in mlir and mlir.count("pto.tmatmul") >= 2  # both matmuls fused in

    def test_fp32_chained_matmul_declines_unsupported_l1_handoff(self, ascend_backend, monkeypatch):
        """A2/A3 cannot keep a same-type FP32 matmul result in L1 for a consumer.

        The hardware fused-chain path is FP32 Acc -> BF16/FP16 Mat. There is no
        FP32 Acc -> FP32 Mat handoff and no Mat -> Acc reload, so preserving an
        explicitly FP32 intermediate requires two standalone cube kernels and a
        GM boundary. The solver must split instead of pricing fictional fusion.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")

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

        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        aic = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIC"]
        assert len(aic) == 2, [f.name for _, f in lowered.functions.items()]

    def test_cube_plan_reuses_shared_boundary_across_requests(self, ascend_backend, monkeypatch):
        """A shared boundary region is loaded once and lives through its last use.

        Both roots consume the same produced ``t`` and the same boundary RHS
        ``d``.  The role-expanded pebbling plan keeps one canonical RHS Mat
        value across the two consumer requests; the emitter replays all three
        matmuls in one AIC kernel instead of silently reloading ``d`` or cutting
        the group.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[64, 64], pl.BF16],
                b: pl.Tensor[[64, 64], pl.BF16],
                d: pl.Tensor[[64, 64], pl.BF16],
            ) -> tuple[pl.Tensor[[64, 64], pl.BF16], pl.Tensor[[64, 64], pl.BF16]]:
                t: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(a, b)
                c: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(t, d)
                e: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(t, d)
                return c, e

        planned = passes.auto_fuse()(Prog)
        body = next(f for _, f in planned.functions.items() if f.name == "chain").as_python()
        assert body.count("pl.spmd(") == 1, body
        assert body.count("pl.tensor.matmul(") == 3, body
        resident_defs = [
            line
            for line in body.splitlines()
            if "_resident_" in line.partition(":")[0] and "pl.tensor.slice" in line
        ]
        assert len(resident_defs) == 1 and "_rhs_l1" in resident_defs[0], body

        namespace = {}
        exec(torch_codegen(planned, run_all_spmd_blocks=True), namespace)  # noqa: S102
        torch.manual_seed(0)
        a = torch.randn(64, 64, dtype=torch.bfloat16) * 0.05
        b = torch.randn(64, 64, dtype=torch.bfloat16) * 0.05
        d = torch.randn(64, 64, dtype=torch.bfloat16) * 0.05
        t = a @ b
        actual_c, actual_e = namespace["chain"](a, b, d)
        expected = t @ d
        assert torch.allclose(actual_c, expected, rtol=1e-3, atol=1e-3)
        assert torch.allclose(actual_e, expected, rtol=1e-3, atol=1e-3)

        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        aic = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIC"]
        assert len(aic) == 1, [f.name for _, f in lowered.functions.items()]

    def test_cube_plan_keeps_lhs_rhs_boundary_roles_distinct(self, ascend_backend, monkeypatch, tmp_path):
        """The two representations of a boundary used by ``A @ A`` do not alias."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def square(
                self, a: pl.Tensor[[64, 64], pl.BF16]
            ) -> tuple[pl.Tensor[[64, 64], pl.BF16], pl.Tensor[[64, 64], pl.BF16]]:
                c: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(a, a)
                d: pl.Tensor[[64, 64], pl.BF16] = pl.matmul(a, a)
                return c, d

        planned = passes.auto_fuse()(Prog)
        dag = json.loads((tmp_path / "square.dag.json").read_text())
        assert all(len(inputs) == 2 and inputs[0] == inputs[1] for inputs in dag["inputs"]), dag
        body = next(f for _, f in planned.functions.items() if f.name == "square").as_python()
        resident_defs = [
            line
            for line in body.splitlines()
            if "_resident_" in line.partition(":")[0] and "pl.tensor.slice" in line
        ]
        assert len(resident_defs) == 2, body
        assert any("_lhs_l1" in line for line in resident_defs), body
        assert any("_rhs_l1" in line for line in resident_defs), body

        namespace = {}
        exec(torch_codegen(planned, run_all_spmd_blocks=True), namespace)  # noqa: S102
        torch.manual_seed(0)
        a = torch.randn(64, 64, dtype=torch.bfloat16) * 0.05
        actual_c, actual_d = namespace["square"](a)
        expected = a @ a
        assert torch.allclose(actual_c, expected, rtol=1e-3, atol=1e-3)
        assert torch.allclose(actual_d, expected, rtol=1e-3, atol=1e-3)

        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        aic = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIC"]
        assert len(aic) == 1, [f.name for _, f in lowered.functions.items()]

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

        solution = json.loads((tmp_path / "chain.sol.json").read_text())
        assert len(solution["cube_schedule"]) == len(solution["subgraphs"])
        cube_plans = [plan for plan in solution["cube_schedule"] if plan is not None]
        assert cube_plans
        assert all(plan["work_units"] >= 1 and plan["matmuls"] for plan in cube_plans)
        assert all(plan["matmuls"][0]["output_variants"] for plan in cube_plans)

    def test_vector_problem_dump_records_emitted_primitive_geometry(self, tmp_path, monkeypatch):
        """The adapter describes the tile ops replayed by VectorStreamPlan.

        A generic tensor subtraction with an ``[M,1]`` operand lowers to
        ``tile.row_expand_sub``; the following exp is flat; multiplying by a
        scalar lowers to ``tile.muls``; and the final row sum marks a grounded
        reduction. Costing must see those exact primitive families and
        geometries without inspecting PyPTO names in the solver.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def vector_semantics(
                self,
                x: pl.Tensor[[8, 512], pl.FP32],
                stat: pl.Tensor[[8, 1], pl.FP32],
            ) -> pl.Tensor[[8, 1], pl.FP32]:
                shifted: pl.Tensor[[8, 512], pl.FP32] = pl.sub(x, stat)
                exponent: pl.Tensor[[8, 512], pl.FP32] = pl.exp(shifted)
                scaled: pl.Tensor[[8, 512], pl.FP32] = pl.mul(exponent, 0.5)
                total: pl.Tensor[[8, 1], pl.FP32] = pl.row_sum(scaled)
                return total

        passes.auto_fuse()(Prog)

        dag = json.loads((tmp_path / "vector_semantics.dag.json").read_text())
        assert dag["vector_primitive_families"] == ["add", "exp", "scalar_mul", "row_sum"]
        assert dag["vector_op_geometries"] == ["row_expand", "flat", "flat", "flat"]
        assert dag["vector_op_capabilities"] == [
            "elementwise",
            "elementwise",
            "elementwise",
            "reduction_sum",
        ]
        assert dag["per_task_overhead_cycles"] == 64

        solution = json.loads((tmp_path / "vector_semantics.sol.json").read_text())
        vector_plan = next(plan for plan in solution["vector_stream"] if plan is not None)
        assert vector_plan["work_units"] >= 1
        assert vector_plan["m_partition"]["parts"] >= 1
        assert vector_plan["n_partition"]["parts"] >= 1
        assert vector_plan["free_tile_alloc"] >= vector_plan["free_tile"]
        assert set(vector_plan["serial_phases"]) == {
            "stats_init",
            "stats_tail",
            "apply_tail",
            "finalize",
        }

    def test_vector_capabilities_decline_unimplemented_reduction_algorithms(self, tmp_path, monkeypatch):
        """Prod/arg/min never enter the generic strip/stream emitter.

        These operations need algorithms that are not equivalent to the
        implemented sum/max accumulator. Shape-producing operations such as
        ``full`` use the same unsupported capability in the adapter.
        Their explicit capability is therefore ``unsupported`` and their op
        category is an opaque partition barrier.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Reductions:
            @pl.function(attrs={"auto_fuse": True})
            def unsupported(
                self, x: pl.Tensor[[8, 8192], pl.FP32]
            ) -> tuple[
                pl.Tensor[[8, 1], pl.FP32],
                pl.Tensor[[8, 1], pl.FP32],
                pl.Tensor[[8, 1], pl.INT32],
            ]:
                minimum: pl.Tensor[[8, 1], pl.FP32] = pl.row_min(x)
                product: pl.Tensor[[8, 1], pl.FP32] = pl.row_prod(x)
                argmax: pl.Tensor[[8, 1], pl.INT32] = pl.row_argmax(x)
                return minimum, product, argmax

        reduced = passes.auto_fuse()(Reductions)
        body = next(f for _, f in reduced.functions.items() if f.name == "unsupported").as_python()
        assert "pl.pipeline(" not in body
        dag = json.loads((tmp_path / "unsupported.dag.json").read_text())
        assert dag["op_types"] == ["Opaque"] * 3
        assert dag["vector_op_capabilities"] == ["unsupported"] * 3

    def test_bare_terminal_unsupported_reduction_declines_whole_function(self, tmp_path, monkeypatch):
        """A singleton opaque sink never reaches solver tile enumeration."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class BareMinimum:
            @pl.function(attrs={"auto_fuse": True})
            def minimum(self, x: pl.Tensor[[8, 8192], pl.FP32]) -> pl.Tensor[[8, 1], pl.FP32]:
                result: pl.Tensor[[8, 1], pl.FP32] = pl.row_min(x)
                return result

        after = passes.auto_fuse()(BareMinimum)

        ir.assert_structural_equal(after, BareMinimum)
        dag = json.loads((tmp_path / "minimum.dag.json").read_text())
        assert dag["op_types"] == ["Opaque"]
        assert dag["vector_op_capabilities"] == ["unsupported"]

    def test_supported_cone_around_unsupported_reduction_declines_whole_function(self, tmp_path, monkeypatch):
        """Supported consumers do not pull an opaque barrier into the solver."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class MinimumConsumer:
            @pl.function(attrs={"auto_fuse": True})
            def normalize(self, x: pl.Tensor[[8, 8192], pl.FP32]) -> pl.Tensor[[8, 8192], pl.FP32]:
                minimum: pl.Tensor[[8, 1], pl.FP32] = pl.row_min(x)
                shifted: pl.Tensor[[8, 8192], pl.FP32] = pl.row_expand_sub(x, minimum)
                result: pl.Tensor[[8, 8192], pl.FP32] = pl.exp(shifted)
                return result

        after = passes.auto_fuse()(MinimumConsumer)

        ir.assert_structural_equal(after, MinimumConsumer)
        dag = json.loads((tmp_path / "normalize.dag.json").read_text())
        assert dag["op_types"] == ["Opaque", "Pointwise", "Pointwise"]
        assert dag["vector_op_capabilities"] == [
            "unsupported",
            "elementwise",
            "elementwise",
        ]

    def test_vector_problem_dump_covers_grounded_unary_scalar_and_column_ops(self, tmp_path, monkeypatch):
        """Known one-instruction PTO lowerings have exact source descriptors.

        Composite high-precision rsqrt deliberately stays generic: it lowers to
        scratch fill + sqrt + barrier + divide, not the basic TRSQRT primitive.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_DUMP", str(tmp_path))
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class UnaryScalar:
            @pl.function(attrs={"auto_fuse": True})
            def grounded(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                y: pl.Tensor[[16, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a: pl.Tensor[[16, 64], pl.FP32] = pl.abs(x)
                s: pl.Tensor[[16, 64], pl.FP32] = pl.sqrt(a)
                n: pl.Tensor[[16, 64], pl.FP32] = pl.neg(s)
                hi: pl.Tensor[[16, 64], pl.FP32] = pl.maximum(n, 0.0)
                lo: pl.Tensor[[16, 64], pl.FP32] = pl.minimum(hi, 1.0)
                pa: pl.Tensor[[16, 64], pl.FP32] = pl.part_add(lo, y)
                pm: pl.Tensor[[16, 64], pl.FP32] = pl.part_mul(pa, y)
                px: pl.Tensor[[16, 64], pl.FP32] = pl.part_max(pm, y)
                out: pl.Tensor[[16, 64], pl.FP32] = pl.part_min(px, y)
                return out

        passes.auto_fuse()(UnaryScalar)
        unary = json.loads((tmp_path / "grounded.dag.json").read_text())
        assert unary["vector_primitive_families"] == [
            "abs",
            "sqrt",
            "scalar_mul",
            "scalar_max",
            "scalar_min",
            "add",
            "mul",
            "add",
            "add",
        ]
        assert unary["vector_op_geometries"] == ["flat"] * 9

        @pl.program
        class Columns:
            @pl.function(attrs={"auto_fuse": True})
            def columns(
                self, x: pl.Tensor[[64, 64], pl.FP32]
            ) -> tuple[pl.Tensor[[1, 64], pl.FP32], pl.Tensor[[1, 64], pl.FP32]]:
                total: pl.Tensor[[1, 64], pl.FP32] = pl.col_sum(x)
                maximum: pl.Tensor[[1, 64], pl.FP32] = pl.col_max(x)
                return total, maximum

        passes.auto_fuse()(Columns)
        columns = json.loads((tmp_path / "columns.dag.json").read_text())
        assert columns["vector_primitive_families"] == ["col_sum", "col_extrema"]

        @pl.program
        class HighPrecision:
            @pl.function(attrs={"auto_fuse": True})
            def high_precision(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[16, 64], pl.FP32]:
                out: pl.Tensor[[16, 64], pl.FP32] = pl.rsqrt(x, high_precision=True)
                return out

        passes.auto_fuse()(HighPrecision)
        hp = json.loads((tmp_path / "high_precision.dag.json").read_text())
        assert hp["vector_primitive_families"] == ["generic"]
        assert hp["vector_op_geometries"] == ["generic"]

    def test_cube_plan_delegates_large_intermediate_l0_tiling(self):
        """A multi-L0 intermediate uses one planned L1 scratch and shared L0 children.

        The forced work unit requests a ``[64,1536]`` bf16 intermediate (192 KiB):
        larger than L0C but within the solver's L1 pebble budget.  AutoFuse realizes
        the CubeSchedulePlan's GM/L1 lifetime as one L1 scratch and replays output-tile
        calls carrying child plans from the shared L0 chooser. AutoTileMatmulL0 consumes
        those records and alone lowers each call to Left/Right/Acc/FIXPIPE operations.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, ir, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def ch(
                    self,
                    a: pl.Tensor[[128, 256], pl.BF16],
                    b: pl.Tensor[[256, 1536], pl.BF16],
                    d: pl.Tensor[[1536, 64], pl.BF16],
                ) -> pl.Tensor[[128, 64], pl.BF16]:
                    t: pl.Tensor[[128, 1536], pl.BF16] = pl.matmul(a, b)
                    out: pl.Tensor[[128, 64], pl.BF16] = pl.matmul(t, d)
                    return out

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "ch").as_python()
            assert "pl.spmd(2" in body and body.count("pl.spmd(") == 1
            assert body.count("pl.tensor.create_l1(") == 1
            assert body.count("pl.tensor.matmul(") >= 2
            assert body.count("pl.tensor.assemble(") >= 2  # internal L1 and boundary drains
            assert "AtomicType.Add" not in body

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(128, 256, dtype=torch.bfloat16) * 0.03
            b = torch.randn(256, 1536, dtype=torch.bfloat16) * 0.03
            d = torch.randn(1536, 64, dtype=torch.bfloat16) * 0.03
            actual = namespace["ch"](a, b, d)
            expected = (a @ b) @ d
            assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3), (
                actual - expected
            ).abs().max().item()

            # The production cube path uses PTOAS for interval/sub-offset L0
            # placement. The legacy PyPTO planner cannot pack two 32 KiB
            # ping-pong children into a later 64 KiB allocation even though the
            # phase lifetimes are disjoint, so it is not the allocator contract
            # this handoff test is intended to exercise.
            with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
                lowered = PassManager.get_strategy(
                    OptimizationStrategy.Default
                ).run_passes(Prog)
            incores = [
                f for _, f in lowered.functions.items() if ir.is_incore_type(f.func_type)
            ]
            assert len(incores) == 1
            assert str(incores[0].func_type) == "FunctionType.AIC"
            lowered_body = incores[0].as_python()
            assert "__autofuse_l0_matmul_plan" not in lowered_body
            assert "pl.tile.extract(" in lowered_body
            assert "target_memory=pl.Mem.Left" in lowered_body
            assert "target_memory=pl.Mem.Right" in lowered_body
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "64,64,1,2,1"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_lone_cube_plan_delegates_l0_dimension_bounds(self):
        """L0 M/N bounds are delegated even when the full region fits by Acc bytes.

        A ``[32,512]`` fp32 output is only 64 KiB, below the 128 KiB Acc capacity,
        while the backend chooser may still impose a smaller L0-N tile. AutoFuse
        must keep the requested GM/L1 region whole; AutoTileMatmulL0 then consumes
        and realizes the attached backend-specific plan. Exact tile geometry is
        covered by the shared chooser tests rather than duplicated here.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, ir, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[32, 64], pl.FP32],
                    b: pl.Tensor[[64, 512], pl.FP32],
                ) -> pl.Tensor[[32, 512], pl.FP32]:
                    out: pl.Tensor[[32, 512], pl.FP32] = pl.matmul(a, b)
                    return out

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "mm").as_python()
            assert "pl.spmd(1" in body and body.count("pl.spmd(") == 1
            assert body.count("pl.tensor.matmul(") == 1
            assert body.count("pl.tensor.assemble(") == 1

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(32, 64, dtype=torch.float32) * 0.05
            b = torch.randn(64, 512, dtype=torch.float32) * 0.05
            actual = namespace["mm"](a, b)
            expected = a @ b
            assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)

            with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
                lowered = PassManager.get_strategy(
                    OptimizationStrategy.Default
                ).run_passes(Prog)
            incores = [
                f for _, f in lowered.functions.items() if ir.is_incore_type(f.func_type)
            ]
            assert len(incores) == 1
            assert str(incores[0].func_type) == "FunctionType.AIC"
            lowered_body = incores[0].as_python()
            assert "__autofuse_l0_matmul_plan" not in lowered_body
            assert "pl.tile.matmul(" in lowered_body
            assert "target_memory=pl.Mem.Left" in lowered_body
            assert "target_memory=pl.Mem.Right" in lowered_body
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "512,32,1,1,1"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_exact_cube_plan_retains_boundary_lhs_across_l0_n_tiles(self):
        """One GM→L1 LHS panel feeds every L0-N output tile in the region.

        The exact CubeSchedulePlan chooses this algorithm only after pricing
        the serial full-panel preload against the repeated pipelined feeds.
        Emission must therefore hoist one boundary slice outside all eight
        output-tile K loops, while each loop takes local L1 extracts from it.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, ir, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[256, 64], pl.FP32],
                    b: pl.Tensor[[64, 1024], pl.FP32],
                ) -> pl.Tensor[[256, 1024], pl.FP32]:
                    out: pl.Tensor[[256, 1024], pl.FP32] = pl.matmul(a, b)
                    return out

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "mm").as_python()
            assert body.count("pl.tensor.slice(a, [256, 64]") == 1
            assert body.count("pl.tensor.slice(fused_0_i0_lhs_l1") == 8
            assert body.count("pl.tensor.slice(b, [16, 128]") == 8
            assert body.count("pl.pipeline(0, 64, 16, stage=2") == 8

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(256, 64, dtype=torch.float32) * 0.05
            b = torch.randn(64, 1024, dtype=torch.float32) * 0.05
            actual = namespace["mm"](a, b)
            expected = a @ b
            assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)

            with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
                lowered = PassManager.get_strategy(
                    OptimizationStrategy.Default
                ).run_passes(Prog)
            incores = [
                f for _, f in lowered.functions.items() if ir.is_incore_type(f.func_type)
            ]
            assert len(incores) == 1
            assert str(incores[0].func_type) == "FunctionType.AIC"
            lowered_body = incores[0].as_python()
            assert lowered_body.count("fused_0_i0_lhs_l1__tile:") == 1
            assert (
                "fused_0_i0_lhs_l1__tile" in lowered_body
                and "target_memory=pl.Mem.Mat" in lowered_body
            )
            assert (
                lowered_body.count(
                    "pl.tile.extract(fused_0_i0_lhs_l1__tile"
                )
                >= 8
            )
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "1024,256,1,1,1"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_multi_window_root_keeps_each_output_tile_in_l0c_then_drains(self):
        """An oversized root nests output/L0C tiles outside GM K windows.

        A2/A3 has no Mat-to-Acc reload. The requested 256x256 FP32 region is
        therefore split into L0C-resident output tiles; each tile runs its own
        stage-2 GM-to-L1 K loop, keeps the accumulator in Acc throughout, and
        drains directly to GM once. AutoTileMatmulL0 alone realizes the nested
        L1-to-L0 schedule and PTOAS must not multiply the two pipeline depths.
        """
        script = textwrap.dedent(
            """
            import pypto.language as pl
            from pypto import backend, ir, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[256, 512], pl.FP32],
                    b: pl.Tensor[[512, 256], pl.FP32],
                ) -> pl.Tensor[[256, 256], pl.FP32]:
                    out: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b)
                    return out

            fused = passes.auto_fuse()(Prog)
            body = next(f for _, f in fused.functions.items() if f.name == "mm").as_python()
            assert body.count("pl.spmd(") == 1, body
            assert "pl.pipeline(" in body, body
            assert "pl.tensor.matmul_acc(" in body, body
            assert body.count("pl.tensor.assemble(") >= 2, body
            assert body.count("__autofuse_l0_matmul_plan") >= 2, body

            with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
                lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            aic = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIC"]
            assert len(aic) == 1, [f.name for _, f in lowered.functions.items()]
            lowered_body = aic[0].as_python()
            assert "__autofuse_l0_matmul_plan" not in lowered_body, lowered_body
            assert "pl.tile.matmul_acc(" in lowered_body, lowered_body
            assert "target_memory=pl.Mem.Mat" in lowered_body, lowered_body
            assert lowered_body.count("pl.tile.store(") >= 2, lowered_body
            assert "pl.tile.tpush_to_aiv(" not in lowered_body, lowered_body
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "256,256,1,1,1"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_low_precision_cube_uses_fp32_accumulator_and_final_narrowing(self):
        """A low-precision result computes in FP32 Acc and narrows only at drain.

        This is the PTO A2/A3 GEMM contract: BF16/FP16 operands feed an FP32 L0C
        accumulator, and FIXPIPE converts that completed tile to the requested output
        dtype. The tensor-level child must therefore be FP32 even though the function
        output is FP16; no FP16 ``matmul_acc`` carry may be constructed.
        """
        script = textwrap.dedent(
            """
            import pypto.language as pl
            from pypto import backend, ir, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mm(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP16],
                    b: pl.Tensor[[64, 64], pl.FP16],
                ) -> pl.Tensor[[64, 64], pl.FP16]:
                    out: pl.Tensor[[64, 64], pl.FP16] = pl.matmul(a, b)
                    return out

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "mm").as_python()
            assert "pl.spmd(4" in body, body
            assert "pl.pipeline(" in body, body
            assert "pl.tensor.matmul_acc(" in body, body
            assert body.count("pl.tensor.matmul(") == 1, body
            assert "pl.Tensor[[32, 32], pl.FP32]" in body, body
            assert "__autofuse_l0_matmul_plan" in body, body

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            incores = [
                f for _, f in lowered.functions.items() if ir.is_incore_type(f.func_type)
            ]
            assert len(incores) == 1
            assert str(incores[0].func_type) == "FunctionType.AIC"
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "32,32,1,2,2"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

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
        # Only the output is assembled; the intermediate stays on-chip.
        assert body.count("pl.tensor.assemble(") == 1

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        # The fused chain is ONE vector kernel, not two.
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        assert str(incores[0].func_type) == "FunctionType.AIV"

    def test_returned_consumed_intermediate_is_materialized_live_out(self, ascend_backend, monkeypatch):
        """A returned SSA value remains observable when fusion also consumes it."""
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def live_out(
                self, x: pl.Tensor[[64, 256], pl.FP32]
            ) -> tuple[
                pl.Tensor[[64, 256], pl.FP32],
                pl.Tensor[[64, 256], pl.FP32],
            ]:
                intermediate: pl.Tensor[[64, 256], pl.FP32] = pl.exp(x)
                out: pl.Tensor[[64, 256], pl.FP32] = pl.add(intermediate, 1.0)
                return intermediate, out

        fused = passes.auto_fuse()(Prog)
        body = next(f for _, f in fused.functions.items() if f.name == "live_out").as_python()
        assert "__FREE_VAR" not in body
        assert body.count("pl.tensor.assemble(") == 2

        namespace: dict = {}
        exec(torch_codegen(fused, run_all_spmd_blocks=True), namespace)  # noqa: S102
        torch.manual_seed(0)
        x = torch.randn(64, 256, dtype=torch.float32)
        intermediate, out = namespace["live_out"](x)
        expected = torch.exp(x)
        assert torch.allclose(intermediate, expected, rtol=1e-4, atol=1e-4)
        assert torch.allclose(out, expected + 1.0, rtol=1e-4, atol=1e-4)

    def test_materialized_single_tile_uses_solver_body_plan(self, ascend_backend, monkeypatch):
        """A one-task materialized vector group still uses the generic planned body.

        Falling back to the legacy plain InCore scope for a single spatial tile would
        violate A5 whenever ``VectorStreamPlan`` selected body strips. This minimal
        case has one serial strip and must therefore emit the generic ``spmd(1)`` body
        without a pipeline.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[1, 1], pl.FP32]) -> pl.Tensor[[1, 1], pl.FP32]:
                c: pl.Tensor[[1, 1], pl.FP32] = pl.add(a, 1.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw").as_python()
        assert "pl.spmd(1" in body
        assert "pl.pipeline(" not in body
        assert "pl.tensor.slice(" in body and "pl.tensor.adds(" in body

    def test_tall_pointwise_streams_within_ub(self, ascend_backend, monkeypatch):
        """A tall pointwise tile is row-STREAMED into enough pipeline strips to fit UB.

        Each vector core's tile is realized as a stage-2 pipeline over row strips. The strip
        count was a fixed heuristic (h/8) with NO UB bound, so a tall tile overflowed the
        vector buffer: at h/8 a ``[262144,64]`` problem tiles to ~``[5461,64]`` per core, whose
        ``[682,64]`` strip double-buffers to 682*64*4*2 = 698368 bytes >> the 188416-byte UB,
        and lowering crashed at ``AllocateMemoryAddr``. ``VectorStreamPlan`` now bounds and owns
        the emitted strip: it doubles the strip count past the {8,4,2} heuristic until the
        real DFS-liveness peak plus one prefetch band fits UB. So the tall tile
        streams (>8 strips) and lowers. Rows are the free axis (no col-major granule pad), and
        the ragged last strip is clamped in-bounds — an idempotent overlap for the non-atomic
        assemble, so every row is written exactly once in effect (checked numerically).
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[262144, 64], pl.FP32]) -> pl.Tensor[[262144, 64], pl.FP32]:
                b: pl.Tensor[[262144, 64], pl.FP32] = pl.add(a, 1.0)
                c: pl.Tensor[[262144, 64], pl.FP32] = pl.mul(b, 2.0)
                return c

        fused = passes.auto_fuse()(Prog)
        body = next(f for _, f in fused.functions.items() if f.name == "pw").as_python()
        # The solver-owned UB bump fired: more strips than the {8,4,2} heuristic max of 8.
        strips = [int(n) for n in re.findall(r"pl\.pipeline\((\d+)", body)]
        assert strips and strips[0] > 8, f"expected UB-bumped strip count > 8, got {strips}"

        # Lowers through the full pipeline WITHOUT overflowing UB (the AllocateMemoryAddr crash
        # this fix prevents). run_passes raises on the overflow, so reaching here is the assertion.
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)

        # ...and is numerically exact — the streamed strips + ragged clamp write every row.
        code = torch_codegen(fused, run_all_spmd_blocks=True)
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        torch.manual_seed(0)
        a = torch.randn(262144, 64, dtype=torch.float32)
        out = namespace["pw"](a)
        ref = (a + 1.0) * 2.0
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4), (
            f"max abs diff {(out - ref).abs().max().item():.3e}"
        )

    def test_c3_per_task_overhead_prefers_fewer_tiles(self, ascend_backend, monkeypatch):
        """The C3 per-task launch-overhead term steers the solver toward FEWER, larger tiles.

        A DDR-bound pointwise kernel's per-wave fill cost is flat for num_tiles <= cores, so the
        pre-C3 model tied 48 `[512,64]` tiles with 12 `[2048,64]` tiles; a device sweep found the
        12-tile plan faster (fewer host launches). C3 adds `num_tiles*split*c_task`, so best_cost
        separates them toward the 12-tile plan. It is GATED on the generic emit — only the
        streaming emit can build the larger `[2048,64]` tile (the winning plan UB-streams it
        into pipeline strips; the legacy tiler materializes the whole tile and overflows UB), so pricing
        fewer-tile plans for the legacy path would pick tiles it cannot realize. Hence with
        ``PYPTO_AUTOFUSE_GENERIC_EMIT=1``: `[4096,384]` tiles to `pl.spmd(12)` (not the legacy
        48 — see test_single_pointwise_tiles_across_vector_cores) and lowers end-to-end.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                return c

        body = next(f for _, f in passes.auto_fuse()(Prog).functions.items() if f.name == "pw").as_python()
        # C3 prefers fewer tiles: [2048,64] -> 2x6 = 12 tasks, not the pre-C3 [512,64] -> 48.
        assert "pl.spmd(12" in body and body.count("pl.spmd(") == 1
        # The larger tile lowers (row-streamed to fit UB, not materialized -> no AllocateMemoryAddr).
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)

    def test_reused_input_wide_pointwise_streams_within_ub(self, ascend_backend, monkeypatch):
        """A fused chain that REUSES an input needs MORE strip UB bands than a linear chain.

        `(a+b)*b` holds `b` live across BOTH ops (peak 3 simultaneously-live tiles vs 2 for a
        linear chain), so a strip's UB footprint is higher. On a WIDE tile — which C3's
        fewer/larger-tile bias prefers here (`[.,4096]`) — sizing the strip against a fixed 2
        bands under-counted, and the `[.,4096]` strip overflowed UB (196608 > 188416) at
        `AllocateMemoryAddr` (found on device). The emit now sizes strips by the REAL peak
        liveness (+1 prefetch band) in `VectorStreamPlan`, so the wide reused-input tile streams
        and lowers.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(
                self,
                a: pl.Tensor[[64, 4096], pl.FP32],
                b: pl.Tensor[[64, 4096], pl.FP32],
            ) -> pl.Tensor[[64, 4096], pl.FP32]:
                c: pl.Tensor[[64, 4096], pl.FP32] = pl.add(a, b)
                d: pl.Tensor[[64, 4096], pl.FP32] = pl.mul(c, b)  # b reused -> live across both ops
                return d

        fused = passes.auto_fuse()(Prog)
        # Lowers without a Vec-buffer overflow (the wide reused-input strip now fits UB).
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        # ...and is numerically exact.
        code = torch_codegen(fused, run_all_spmd_blocks=True)
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        torch.manual_seed(0)
        a = torch.randn(64, 4096, dtype=torch.float32)
        b = torch.randn(64, 4096, dtype=torch.float32)
        out = namespace["pw"](a, b)
        assert torch.allclose(out, (a + b) * b, rtol=1e-4, atol=1e-4), (
            f"max abs diff {(out - (a + b) * b).abs().max().item():.3e}"
        )

    def test_mixed_dtype_pointwise_strip_uses_per_tensor_ub_bytes(self):
        """FP32 intermediates are not sized as the narrower INT8 live-out."""
        script = textwrap.dedent(
            """
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def mixed(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.INT8]:
                    wide: pl.Tensor[[128, 8192], pl.FP32] = pl.exp(x)
                    out: pl.Tensor[[128, 8192], pl.INT8] = pl.cast(wide, pl.INT8)
                    return out

            fused = passes.auto_fuse()(Prog)
            body = next(f for _, f in fused.functions.items() if f.name == "mixed").as_python()
            assert "pl.pipeline(" in body, body
            assert "pl.spmd(4" in body, body
            PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "2048,128,1,1,4"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_split_k_matmul_seed_tiles_within_ub(self, ascend_backend, monkeypatch):
        """The split-K zero-seed tiles the [M,N] output into UB-FITTING pieces.

        A large output TILE — which C3 prefers (`[512,512]` → a `[256,256]` tile) — makes the
        per-tile seed `tensor.full([h,w])` ITSELF exceed UB (a `[256,256]` fp32 fill = 256 KB >
        188 KB) → `AllocateMemoryAddr` crash (found on device). The seed now zeroes `[M,N]` in
        `[seed_h, w]` tiles capped to fit UB, so a large-tile split-K matmul lowers, and the
        split-K partials atomic-add onto the tiled zero-seed correctly.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[512, 512], pl.FP32],
                b: pl.Tensor[[512, 512], pl.FP32],
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                c: pl.Tensor[[512, 512], pl.FP32] = pl.matmul(a, b)
                return c

        fused = passes.auto_fuse()(Prog)
        # Lowers without a seed Vec-buffer overflow (the seed tile is capped to
        # fit UB). The plan-driven cube path uses PTOAS for the nested GM/L1 and
        # L1/L0 lifetimes; the legacy host allocator cannot interval-pack this
        # split-K accumulator schedule even though the seed itself is legal.
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
            PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        # ...and is numerically exact (split-K atomic-add merge onto the tiled zero-seed).
        code = torch_codegen(fused, run_all_spmd_blocks=True)
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        torch.manual_seed(0)
        a = torch.randn(512, 512, dtype=torch.float32)
        b = torch.randn(512, 512, dtype=torch.float32)
        out = namespace["mm"](a, b)
        assert torch.allclose(out, a @ b, rtol=1e-3, atol=1e-3), (
            f"max abs diff {(out - a @ b).abs().max().item():.3e}"
        )

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
        through AllocateMemoryAddr without a Vec overflow (streaming fired), and (b) smaller
        reductions are NUMERICALLY exact across the solver-selected materialize/split/stream
        schedules — add (col_sum) and max on SIGNED data (row_max) — via torch_codegen.
        (Behind the generic-emit flag, where streaming lives.)
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")

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

        # (b) numerical exactness of the selected reduction emit (executes the emitted IR on CPU).
        def _numeric(program, entry, x, ref):
            ns: dict = {}
            exec(torch_codegen(passes.auto_fuse()(program), run_all_spmd_blocks=True), ns)  # noqa: S102
            got = ns[entry](x)
            diff = (got - ref).abs().max().item()
            assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), f"{entry}: max abs diff {diff:.3e}"

        @pl.program
        class ColSum:  # reduce M, add-merge; the exact pebble peak fits and the solver selects S2
            @pl.function(attrs={"auto_fuse": True})
            def cs(self, a: pl.Tensor[[2048, 8], pl.FP32]) -> pl.Tensor[[1, 8], pl.FP32]:
                c: pl.Tensor[[1, 8], pl.FP32] = pl.col_sum(a)
                return c

        @pl.program
        class PointwiseColSum:  # the pointwise producer is replayed inside each reduced-axis slice
            @pl.function(attrs={"auto_fuse": True})
            def pcs(self, a: pl.Tensor[[1024, 8], pl.FP32]) -> pl.Tensor[[1, 8], pl.FP32]:
                doubled: pl.Tensor[[1024, 8], pl.FP32] = pl.add(a, a)
                c: pl.Tensor[[1, 8], pl.FP32] = pl.col_sum(doubled)
                return c

        @pl.program
        class RowMax:  # reduce N, max-merge on SIGNED data (proves mask != zero-fill)
            @pl.function(attrs={"auto_fuse": True})
            def rm(self, a: pl.Tensor[[4, 2048], pl.FP32]) -> pl.Tensor[[4, 1], pl.FP32]:
                c: pl.Tensor[[4, 1], pl.FP32] = pl.row_max(a)
                return c

        @pl.program
        class ColMax:  # reduce M; split-max is unsupported, so the materialized fallback is serial
            @pl.function(attrs={"auto_fuse": True})
            def cm(self, a: pl.Tensor[[2048, 4], pl.FP32]) -> pl.Tensor[[1, 4], pl.FP32]:
                c: pl.Tensor[[1, 4], pl.FP32] = pl.col_max(a)
                return c

        @pl.program
        class RowSum:  # reduce N, add-merge; bare row reduction — guards the reduced AXIS
            @pl.function(attrs={"auto_fuse": True})
            def rs(self, a: pl.Tensor[[4, 2048], pl.FP32]) -> pl.Tensor[[4, 1], pl.FP32]:
                c: pl.Tensor[[4, 1], pl.FP32] = pl.row_sum(a)
                return c

        def _body(program, entry):
            return next(
                f for _, f in passes.auto_fuse()(program).functions.items() if f.name == entry
            ).as_python()

        # G6: the selected algorithm, not merely the numeric result, must match the cost.
        assert "atomic=pl.AtomicType.Add" in _body(ColSum, "cs")
        assert "atomic=pl.AtomicType.Add" in _body(PointwiseColSum, "pcs")
        for program, entry in [(RowMax, "rm"), (ColMax, "cm"), (RowSum, "rs")]:
            assert "atomic=pl.AtomicType.Add" not in _body(program, entry)

        x_cs = torch.arange(2048 * 8, dtype=torch.float32).reshape(2048, 8) * 0.01
        _numeric(ColSum, "cs", x_cs, x_cs.sum(dim=0, keepdim=True))
        x_pcs = torch.arange(1024 * 8, dtype=torch.float32).reshape(1024, 8) * 0.01
        _numeric(PointwiseColSum, "pcs", x_pcs, (x_pcs + x_pcs).sum(dim=0, keepdim=True))
        x_rm = (torch.arange(4 * 2048, dtype=torch.float32).reshape(4, 2048) % 97) - 48.0
        _numeric(RowMax, "rm", x_rm, x_rm.max(dim=1, keepdim=True).values)
        # col_max: MAX-merge (not add) on signed data — a sum would flip the sign of the answer.
        x_cm = (torch.arange(2048 * 4, dtype=torch.float32).reshape(2048, 4) % 97) - 48.0
        _numeric(ColMax, "cm", x_cm, x_cm.max(dim=0, keepdim=True).values)
        # bare row_sum: reduces N (width), not M — a wrong-axis reduction gives [4,1] of Σ over M.
        x_rs = torch.arange(4 * 2048, dtype=torch.float32).reshape(4, 2048) * 0.01
        _numeric(RowSum, "rs", x_rs, x_rs.sum(dim=1, keepdim=True))

    def test_streamed_reduction_apply_p2(self, ascend_backend, monkeypatch):
        """P2: a POINTWISE sink consuming a single reduction (x - row_max(x), rmsnorm) whose output
        SPANS the reduced axis. Two-pass stream: pass 0 accumulates the reduction; pass 1 re-streams
        the reduced axis, recomputes the pointwise cone with the finalized reduction substituted, and
        assembles each output chunk — the final apply CHUNKS the reduced axis (else the full-shape
        output re-overflows UB, review R3 #2). Asserts a huge case lowers and two shapes are exact
        (incl. a pre-reduction pointwise x*x streamed per stats chunk and pruned from apply).
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

        # A5 (G2): both streamed passes — the accumulate (pass 0) and the apply re-stream (pass 1) —
        # are ForKind::Pipeline (stage=2), not Sequential, so the DDR-bound reduced-axis reads overlap
        # compute (max(compute,ddr) roofline). Pre-G2 they were serial `pl.range` loops.
        big_body = next(f for _, f in passes.auto_fuse()(Big).functions.items() if f.name == "sm").as_python()
        assert big_body.count("pl.pipeline(") == 2, big_body  # accumulate + apply, both pipelined

        out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Big)
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]

        # The streamed stats/apply peak must retain the FP32 source and
        # intermediate widths even when the boundary output is narrower.
        @pl.program
        class MixedDtype:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP16]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                centered: pl.Tensor[[128, 16384], pl.FP32] = pl.sub(x, m)
                out: pl.Tensor[[128, 16384], pl.FP16] = pl.cast(centered, pl.FP16)
                return out

        mixed_body = next(
            f for _, f in passes.auto_fuse()(MixedDtype).functions.items() if f.name == "sm"
        ).as_python()
        assert mixed_body.count("pl.pipeline(") == 2, mixed_body
        mixed_out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(MixedDtype)
        mixed_incores = [f for _, f in mixed_out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(mixed_incores) == 1, [f.name for f in mixed_incores]

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
        class RmsLike:  # x*x runs per stats chunk; dependency-pruned apply needs only x and final ms
            @pl.function(attrs={"auto_fuse": True})
            def rms(self, x: pl.Tensor[[128, 16384], pl.FP32]) -> pl.Tensor[[128, 16384], pl.FP32]:
                sq: pl.Tensor[[128, 16384], pl.FP32] = pl.mul(x, x)
                ms: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(sq)
                return pl.mul(x, ms)

        x_sm = (torch.arange(128 * 16384, dtype=torch.float32).reshape(128, 16384) % 91) - 45.0
        _numeric(SubMax, "sm", x_sm, x_sm - x_sm.max(dim=1, keepdim=True).values)
        x_rms = (torch.arange(128 * 16384, dtype=torch.float32).reshape(128, 16384) % 13) * 0.1 + 0.05
        _numeric(RmsLike, "rms", x_rms, x_rms * (x_rms * x_rms).sum(dim=1, keepdim=True))

    def test_p4_online_softmax_fuses_into_one_streamed_kernel(self, ascend_backend, monkeypatch):
        """P4: a softmax over a UB-overflowing reduced axis FUSES into ONE streamed online-flash
        kernel (default-on after silicon closure), instead of the G1 cut into row_max + apply pieces.

        The emit maintains the coupled running `(m, l)` stats with the exact `exp(m_old - m_new)`
        rescale in a single streamed pass 0 (one x-slice DMA per chunk → the reduction result IS
        the finalized max/sum), then a pass-1 apply re-streams the reduced axis substituting the
        finalized `(M, L)`: `exp(x - M)/L`. Both streamed passes are stage-2 pipelined when their
        rolled loops have at least two iterations.
        The solver ranks the fused plan cheaper than the cut (~34% here), so it fires with no force.
        (NOTE: the DAG must be FULLY NAMED — a nested-argument call like `exp(row_expand_sub(x,m))`
        drops the inner op from the solver graph and misses P4; see KNOWN_ISSUES.)
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.delenv("PYPTO_AUTOFUSE_P4", raising=False)

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.FP32]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                sh: pl.Tensor[[128, 8192], pl.FP32] = pl.row_expand_sub(x, m)
                e: pl.Tensor[[128, 8192], pl.FP32] = pl.exp(sh)
                s: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                out: pl.Tensor[[128, 8192], pl.FP32] = pl.row_expand_div(e, s)
                return out

        fused = passes.auto_fuse()(Prog)
        body = next(f for _, f in fused.functions.items() if f.name == "sm").as_python()
        # ONE fused streamed kernel (not the 2-group G1 cut), carrying the online (m,l) stats.
        assert body.count("pl.spmd(") == 1, body
        assert "_m_it" in body and "_l_it" in body  # the coupled running-stats loop carries
        assert "pl.tensor.maximum(" in body  # the online running-max merge
        assert body.count("pl.pipeline(") == 2, body  # online stats + apply, both stage-2 pipelined

        # Lowers through the full pipeline (streamed, fits UB).
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)

        # ...and is numerically exact vs torch softmax (the flash rescale is exact).
        code = torch_codegen(fused, run_all_spmd_blocks=True)
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        torch.manual_seed(0)
        x = torch.randn(128, 8192, dtype=torch.float32)
        out = namespace["sm"](x)
        assert torch.allclose(out, torch.softmax(x, dim=1), rtol=1e-4, atol=1e-4), (
            f"max abs diff {(out - torch.softmax(x, dim=1)).abs().max().item():.3e}"
        )

        # The explicit differential opt-out remains available even though
        # exact softmax is now the default.
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "0")
        cut = passes.auto_fuse()(Prog)
        cut_body = next(f for _, f in cut.functions.items() if f.name == "sm").as_python()
        assert "_m_it" not in cut_body and "_l_it" not in cut_body, cut_body
        assert cut_body.count("pl.spmd(") >= 2, cut_body

    def test_p4_declines_when_internal_statistic_is_returned(self, ascend_backend, monkeypatch):
        """Returning row_max makes it a second live-out, so flash P4 must cut."""
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm_with_max(
                self, x: pl.Tensor[[32, 8192], pl.FP32]
            ) -> tuple[
                pl.Tensor[[32, 1], pl.FP32],
                pl.Tensor[[32, 8192], pl.FP32],
            ]:
                maximum: pl.Tensor[[32, 1], pl.FP32] = pl.row_max(x)
                shifted: pl.Tensor[[32, 8192], pl.FP32] = pl.row_expand_sub(x, maximum)
                exponent: pl.Tensor[[32, 8192], pl.FP32] = pl.exp(shifted)
                total: pl.Tensor[[32, 1], pl.FP32] = pl.row_sum(exponent)
                out: pl.Tensor[[32, 8192], pl.FP32] = pl.row_expand_div(exponent, total)
                return maximum, out

        fused = passes.auto_fuse()(Prog)
        body = next(f for _, f in fused.functions.items() if f.name == "sm_with_max").as_python()
        assert "_m_it" not in body and "_l_it" not in body
        assert "__FREE_VAR" not in body

        namespace: dict = {}
        exec(torch_codegen(fused, run_all_spmd_blocks=True), namespace)  # noqa: S102
        torch.manual_seed(0)
        x = torch.randn(32, 8192, dtype=torch.float32)
        maximum, out = namespace["sm_with_max"](x)
        expected_max = x.max(dim=1, keepdim=True).values
        assert torch.allclose(maximum, expected_max, rtol=1e-4, atol=1e-4)
        assert torch.allclose(out, torch.softmax(x, dim=1), rtol=1e-4, atol=1e-4)

    def test_g9_tall_softmax_uses_full_vector_wave(self, ascend_backend, monkeypatch):
        """Row-aware reduction work prevents a tall P4 kernel from under-filling AIVs.

        The 910B2 sweep scales strongly from 12 through 48 logical tasks. PTO's
        row-aware TROWSUM/TROWMAX formulas reproduce that mechanism without
        changing the queue-wave equation, so the natural host plan must reach
        the full 48-task first wave and the emitter must realize that grid.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[768, 8192], pl.FP32]) -> pl.Tensor[[768, 8192], pl.FP32]:
                m: pl.Tensor[[768, 1], pl.FP32] = pl.row_max(x)
                shifted: pl.Tensor[[768, 8192], pl.FP32] = pl.row_expand_sub(x, m)
                exponent: pl.Tensor[[768, 8192], pl.FP32] = pl.exp(shifted)
                total: pl.Tensor[[768, 1], pl.FP32] = pl.row_sum(exponent)
                out: pl.Tensor[[768, 8192], pl.FP32] = pl.row_expand_div(exponent, total)
                return out

        fused = passes.auto_fuse()(Prog)
        body = next(f for _, f in fused.functions.items() if f.name == "sm").as_python()
        assert "pl.spmd(48" in body and body.count("pl.spmd(") == 1, body

    def test_p4_nonuniform_region_count_matches_forced_solver_grid(self):
        """DMA padding must not replace the solver's logical P4 task grid.

        The 128-row axis split into 12 regions has eight 11-row and four
        10-row logical regions. FP32 reduction tiles allocate 16 padded rows,
        but the emitted kernel must still launch exactly 12 blocks. The final
        max-shape body overlaps one row idempotently and computes every output.

        ``PYPTO_AUTOFUSE_FORCE_PLAN`` is process-cached, so run in a fresh
        interpreter.
        """
        script = textwrap.dedent(
            """
            import torch

            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def sm(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.FP32]:
                    m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                    sh: pl.Tensor[[128, 8192], pl.FP32] = pl.row_expand_sub(x, m)
                    e: pl.Tensor[[128, 8192], pl.FP32] = pl.exp(sh)
                    s: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                    out: pl.Tensor[[128, 8192], pl.FP32] = pl.row_expand_div(e, s)
                    return out

            after = passes.auto_fuse()(Prog)
            body = next(f for _, f in after.functions.items() if f.name == "sm").as_python()
            assert "pl.spmd(12" in body and body.count("pl.spmd(") == 1, body

            namespace = {}
            exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            x = torch.randn(128, 8192, dtype=torch.float32)
            actual = namespace["sm"](x)
            expected = torch.softmax(x, dim=1)
            assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4), (
                actual - expected
            ).abs().max().item()
            """
        )
        env = os.environ.copy()
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_P4"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "8192,11,1,12,1"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_p4_layernorm_welford_forced_emit_is_stable(self, ascend_backend, monkeypatch):
        """P4 increment 2: a DUAL-SUM layernorm over a UB-overflowing reduced axis can emit ONE
        streamed online Welford kernel (behind PYPTO_AUTOFUSE_P4).

        The shared P4 analysis proves the exact dual-sum layernorm shape: Sx=row_sum(x) and
        Sxsq=row_sum(x^2), both derived directly from x, NOT the chained row_sum((x-mu)^2). But
        var = E[x^2] - E[x]^2 computed from those raw sums CATASTROPHICALLY CANCELS for a large input
        mean (NaN at mean >~2000). So the EMIT streams a numerically-STABLE Welford instead: pass 0
        carries a running (mean, M2, count) merged per chunk by Chan's parallel formula (chunk M2 via
        the stable row_sum((x-mean_a)^2) form), and pass 1 substitutes the FINALIZED stable mean and
        var = M2/N directly into the cone (bypassing the sx/sxsq -> var path). Both streamed passes
        are stage-2 pipelined when their rolled loops have at least two iterations.

        The DAG must be FULLY NAMED (a nested-argument call drops the inner op from the solver graph
        and misses P4; see KNOWN_ISSUES). Numerics are checked across a wide range of input means
        (randn, +100, +1000, +2000) — the +2000 case NaNs under the old dual-sum emit and is the whole
        reason for the Welford accumulation; Welford stays within tolerance and never NaNs. This
        correctness/emit test forces the exact P4 group so a later cost-model decision to prefer the
        cut does not silently remove coverage of the Welford implementation.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")
        n = 8192
        eps = 1e-5

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def ln(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                sx: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(x)  # Sx  (reduction 1)
                xsq: pl.Tensor[[128, n], pl.FP32] = pl.mul(x, x)
                sxsq: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(xsq)  # Sx^2 (reduction 2, independent)
                mean: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sx, 1.0 / n)
                msq: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sxsq, 1.0 / n)
                m2: pl.Tensor[[128, 1], pl.FP32] = pl.mul(mean, mean)
                var: pl.Tensor[[128, 1], pl.FP32] = pl.sub(msq, m2)  # E[x^2] - E[x]^2
                veps: pl.Tensor[[128, 1], pl.FP32] = pl.add(var, eps)
                inv: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(veps)
                xc: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_sub(x, mean)  # x - mean (spanning)
                out: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_mul(xc, inv)  # * inv   (spanning)
                return out

        fused = passes.auto_fuse()(Prog)
        body = next(f for _, f in fused.functions.items() if f.name == "ln").as_python()
        # The forced exact group emits one streamed kernel carrying running Welford (mean, M2, count).
        assert body.count("pl.spmd(") == 1, body
        # Welford iter-arg carries (re-pointed from the old dual-sum {_s0_it, _s1_it} markers, which the
        # unstable sum-accumulator emit no longer produces).
        assert "_wmean_it" in body and "_wM2_it" in body and "_wcnt_it" in body, body
        assert "_s0_it" not in body and "_s1_it" not in body, body
        # Welford's parallel merge divides by the running count (n_new) — a tensor.div the dual-sum emit
        # (pure adds) never had. Its presence proves the stable accumulation is what got emitted.
        assert "pl.tensor.div(" in body, body
        assert body.count("pl.pipeline(") == 2, body  # Welford stats + apply, both stage-2 pipelined

        # Lowers through the full pipeline (streamed, fits UB — no AllocateMemoryAddr overflow).
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)

        # Numerically matches a torch layernorm reference. torch_codegen runs fp32 on CPU (torch.rsqrt
        # is exact), so the algorithm itself is validated tight; the 1e-2 tol only budgets the silicon
        # HW-rsqrt approximation. Checked across a WIDE range of input means: the +1000/+2000 cases are
        # exactly where the old dual-sum var = E[x^2] - E[x]^2 loses all precision (NaN at +2000);
        # Welford stays within tolerance and never NaNs.
        code = torch_codegen(fused, run_all_spmd_blocks=True)
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102

        def _ref(x):
            return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + eps)

        torch.manual_seed(0)
        for shift in (0.0, 100.0, 1000.0, 2000.0):
            x = torch.randn(128, n, dtype=torch.float32) + shift
            out = namespace["ln"](x)
            ref = _ref(x)
            assert not bool(torch.isnan(out).any()), f"NaN at mean+{shift} (dual-sum cancellation)"
            assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), (
                f"mean+{shift}: max abs diff {(out - ref).abs().max().item():.3e}"
            )

        # Return to the natural solver decision for the structural near-miss below.
        monkeypatch.delenv("PYPTO_AUTOFUSE_FORCE_MERGE")

        # A CHAINED layernorm — row_sum((x-mu)^2) depends on the finalized mean, so the two sums are
        # NOT the exact descriptor — must DECLINE P4 and be CUT by the solver, NOT fused
        # into the two-accumulator dual-sum kernel. This is the dual-sum-vs-Welford boundary.
        @pl.program
        class Chained:
            @pl.function(attrs={"auto_fuse": True})
            def ln(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                sx: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(x)
                mu: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sx, 1.0 / n)
                xc: pl.Tensor[[128, n], pl.FP32] = pl.sub(x, mu)
                sq: pl.Tensor[[128, n], pl.FP32] = pl.mul(xc, xc)
                sv: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(sq)  # depends on mu -> sx (CHAINED)
                var: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sv, 1.0 / n)
                ve: pl.Tensor[[128, 1], pl.FP32] = pl.add(var, eps)
                inv: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(ve)
                out: pl.Tensor[[128, n], pl.FP32] = pl.mul(xc, inv)
                return out

        chained = passes.auto_fuse()(Chained)
        cbody = next(f for _, f in chained.functions.items() if f.name == "ln").as_python()
        # Declined: the group is CUT (>= 2 streamed spmd scopes), NOT the single dual-sum kernel — no
        # two-accumulator loop marker is emitted.
        assert "_s0_it" not in cbody and "_s1_it" not in cbody, cbody
        assert cbody.count("pl.spmd(") >= 2, cbody
        # And the chained cut still lowers cleanly (no overflow) through the full pipeline.
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Chained)

    def test_p4_exact_matcher_rejects_near_miss_algorithms(self, ascend_backend, monkeypatch):
        """P4 feasibility and emission consume one exact semantic descriptor.

        A temperature-scaled softmax and an independent two-sum graph with a weighted second moment
        are not the algorithms implemented by the online softmax/Welford emit. They must be cut into
        ordinary P1/P2 groups, lower cleanly, and preserve their original mathematics.
        """
        torch = pytest.importorskip("torch")
        from pypto.debug import torch_codegen  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")
        n = 8192
        eps = 1e-5

        @pl.program
        class TemperatureSoftmax:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                shifted: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_sub(x, m)
                scaled: pl.Tensor[[128, n], pl.FP32] = pl.mul(shifted, 0.5)
                e: pl.Tensor[[128, n], pl.FP32] = pl.exp(scaled)
                s: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                out: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_div(e, s)
                return out

        scaled = passes.auto_fuse()(TemperatureSoftmax)
        scaled_body = next(f for _, f in scaled.functions.items() if f.name == "sm").as_python()
        assert "_m_it" not in scaled_body and "_l_it" not in scaled_body, scaled_body
        assert scaled_body.count("pl.spmd(") >= 2, scaled_body
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(TemperatureSoftmax)

        @pl.program
        class BranchedSoftmax:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
                shifted: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_sub(x, m)
                e: pl.Tensor[[128, n], pl.FP32] = pl.exp(shifted)
                s: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                softmax: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_div(e, s)
                out: pl.Tensor[[128, n], pl.FP32] = pl.add(softmax, m)  # m escapes the P4 cone
                return out

        branched = passes.auto_fuse()(BranchedSoftmax)
        branched_body = next(f for _, f in branched.functions.items() if f.name == "sm").as_python()
        assert "_m_it" not in branched_body and "_l_it" not in branched_body, branched_body
        assert branched_body.count("pl.spmd(") >= 2, branched_body
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(BranchedSoftmax)

        @pl.program
        class WeightedSecondMoment:
            @pl.function(attrs={"auto_fuse": True})
            def norm(self, x: pl.Tensor[[128, n], pl.FP32]) -> pl.Tensor[[128, n], pl.FP32]:
                sx: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(x)
                xsq: pl.Tensor[[128, n], pl.FP32] = pl.mul(x, x)
                weighted_xsq: pl.Tensor[[128, n], pl.FP32] = pl.mul(xsq, 2.0)
                s2: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(weighted_xsq)
                mean: pl.Tensor[[128, 1], pl.FP32] = pl.mul(sx, 1.0 / n)
                second: pl.Tensor[[128, 1], pl.FP32] = pl.mul(s2, 1.0 / n)
                veps: pl.Tensor[[128, 1], pl.FP32] = pl.add(second, eps)
                inv: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(veps)
                centered: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_sub(x, mean)
                out: pl.Tensor[[128, n], pl.FP32] = pl.row_expand_mul(centered, inv)
                return out

        weighted = passes.auto_fuse()(WeightedSecondMoment)
        weighted_body = next(f for _, f in weighted.functions.items() if f.name == "norm").as_python()
        assert "_wmean_it" not in weighted_body and "_wM2_it" not in weighted_body, weighted_body
        assert weighted_body.count("pl.spmd(") >= 2, weighted_body
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(WeightedSecondMoment)

        def _numeric(program, entry, x, ref):
            namespace: dict = {}
            exec(torch_codegen(program, run_all_spmd_blocks=True), namespace)  # noqa: S102
            got = namespace[entry](x)
            assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (
                f"{entry} changed semantics: max abs diff {(got - ref).abs().max().item():.3e}"
            )

        torch.manual_seed(0)
        x = torch.randn(128, n, dtype=torch.float32)
        _numeric(scaled, "sm", x, torch.softmax(0.5 * x, dim=1))
        _numeric(
            branched,
            "sm",
            x,
            torch.softmax(x, dim=1) + x.max(dim=1, keepdim=True).values,
        )
        weighted_ref = (x - x.mean(dim=1, keepdim=True)) * torch.rsqrt(
            2.0 * (x * x).mean(dim=1, keepdim=True) + eps
        )
        _numeric(weighted, "norm", x, weighted_ref)

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
        # This is the explicit non-P4 fallback contract.  Exact softmax is
        # default-on; disabling P4 here keeps the test focused on G1 cutting
        # and inline-return wiring rather than the online-softmax fast path.
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "0")
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
        # Exercise the guarded G1 fallback explicitly.  With the default P4
        # policy, the exact-softmax cases below are intentionally streamable.
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "0")

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
            def ba(
                self,
                x: pl.Tensor[[m_, n_], pl.FP32],
                b: pl.Tensor[[1, n_], pl.FP32],
            ) -> pl.Tensor[[m_, n_], pl.FP32]:
                return pl.add(x, b)

        @pl.program
        class RowScale:  # N-broadcast [M,1]
            @pl.function(attrs={"auto_fuse": True})
            def rs(
                self,
                x: pl.Tensor[[m_, n_], pl.FP32],
                s: pl.Tensor[[m_, 1], pl.FP32],
            ) -> pl.Tensor[[m_, n_], pl.FP32]:
                return pl.mul(x, s)

        @pl.program
        class Chain:  # both broadcasts fused + a unary
            @pl.function(attrs={"auto_fuse": True})
            def ch(
                self,
                x: pl.Tensor[[m_, n_], pl.FP32],
                b: pl.Tensor[[1, n_], pl.FP32],
                s: pl.Tensor[[m_, 1], pl.FP32],
            ) -> pl.Tensor[[m_, n_], pl.FP32]:
                t: pl.Tensor[[m_, n_], pl.FP32] = pl.add(x, b)
                u: pl.Tensor[[m_, n_], pl.FP32] = pl.mul(t, s)
                return pl.exp(u)

        @pl.program
        class P2Bcast:  # reduction group taking an external [M,1] stat (a G1/G3 softmax cut piece)
            @pl.function(attrs={"auto_fuse": True})
            def sm2(
                self,
                x: pl.Tensor[[m_, n_], pl.FP32],
                mstat: pl.Tensor[[m_, 1], pl.FP32],
            ) -> pl.Tensor[[m_, n_], pl.FP32]:
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

    def test_mixed_c2v_plan_emits_two_lane_fifo_pipeline(self):
        """The buildable mixed plan survives the complete lowering pipeline.

        FORCE_PLAN is process-cached, so the fixed 6x8 experiment runs in a
        fresh interpreter.  Forty-eight logical regions become 24 mixed group
        launches with two successor items per group.  The emitted UP_DOWN split
        gives both AIV lanes real half-row work; ExpandMixedKernel owns the
        C->V FIFO decouples the two sequential per-engine item loops.  The
        outer loop must not be tagged as a generic software pipeline because
        that would multiply the nested AutoTileL0 buffers.
        """
        script = textwrap.dedent(
            """
            import torch
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[192, 64], pl.FP32],
                    b: pl.Tensor[[64, 256], pl.FP32],
                    bias: pl.Tensor[[1, 256], pl.FP32],
                ) -> pl.Tensor[[192, 256], pl.FP32]:
                    mm: pl.Tensor[[192, 256], pl.FP32] = pl.matmul(a, b)
                    out: pl.Tensor[[192, 256], pl.FP32] = pl.add(mm, bias)
                    return out

            planned = passes.auto_fuse()(Prog)
            planned_text = next(
                f for _, f in planned.functions.items() if f.name == "epilogue"
            ).as_python()
            assert "pl.spmd(24" in planned_text, planned_text
            assert "pl.range(2" in planned_text, planned_text
            assert "pl.pipeline(2, stage=2" not in planned_text, planned_text
            assert "pl.split(pl.SplitMode.UP_DOWN, slot_num=8)" in planned_text, planned_text
            assert planned_text.count("pl.tensor.matmul(") == 1, planned_text
            assert planned_text.count("pl.tensor.add(") == 1, planned_text

            namespace = {}
            exec(torch_codegen(planned, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(192, 64)
            b = torch.randn(64, 256)
            bias = torch.randn(1, 256)
            got = namespace["epilogue"](a, b, bias)
            expected = a @ b + bias
            assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4), (got - expected).abs().max()

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            text = lowered.as_python()
            assert "type=pl.FunctionType.Group" in text, text
            assert "type=pl.FunctionType.AIC" in text, text
            assert "type=pl.FunctionType.AIV" in text, text
            assert 'attrs={"core_num": 24}' in text, text
            assert text.count("pl.tile.tpush_to_aiv(") == 1, text
            assert text.count("pl.tile.tpop_from_aic(") == 1, text
            assert text.count("pl.system.tfree_to_aic(") == 1, text
            assert text.count("slot_size=4096") == 2, text
            assert text.count("slot_num=8") >= 2, text
            assert "pl.tile.get_subblock_idx()" in text, text
            assert "subblock_idx * 16" in text, text
            assert "__gm_pipe_buffer" in text, text
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "32,32,1,6,8"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    def test_cube_k_window_keeps_l0_operand_stages_disjoint(self):
        """A one-L0-tile child still needs outer-stage L0A/L0B ping-pong.

        This is the homogeneous half of the silicon M1 reproducer. The
        CubeSchedulePlan streams four 16-wide K windows through a stage-2
        GM->L1 loop, while each child already fits L0. With no inner L0 loop,
        the outer membership must own two Left/Right banks; aliasing both
        stages onto one address lets the next move overwrite a live MAD input.
        """
        script = textwrap.dedent(
            """
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[192, 64], pl.FP32],
                    b: pl.Tensor[[64, 256], pl.FP32],
                    bias: pl.Tensor[[1, 256], pl.FP32],
                ) -> pl.Tensor[[192, 256], pl.FP32]:
                    mm: pl.Tensor[[192, 256], pl.FP32] = pl.matmul(a, b)
                    out: pl.Tensor[[192, 256], pl.FP32] = pl.add(mm, bias)
                    return out

            planned = passes.auto_fuse()(Prog)
            planned_body = next(f for _, f in planned.functions.items() if f.name == "epilogue").as_python()
            assert "pl.pipeline(0, 64, 16" in planned_body, planned_body
            assert "_a_k0" not in planned_body and "_b_k0" not in planned_body, planned_body

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            aic = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIC"]
            assert len(aic) == 1, [f.name for _, f in lowered.functions.items()]
            body = aic[0].as_python()
            assert body.count("pl.tile.alloc(pl.Mem.Left") == 2, body
            assert body.count("pl.tile.alloc(pl.Mem.Right") == 2, body
            assert body.count("pl.tile.alloc(pl.Mem.Acc") == 1, body
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "0"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = "32,32,1,6,8"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    @pytest.mark.parametrize(
        ("size", "force", "expected_left", "expected_right"),
        [
            pytest.param(272, "144,144,1,2,2", None, None, id="persistent-l0c"),
            pytest.param(512, "128,128,1,4,4", 2, 2, id="serial-k-tail"),
        ],
    )
    def test_cube_exact_plan_preserves_l0_phase_lifetimes(
        self,
        size,
        force,
        expected_left,
        expected_right,
    ):
        """Exact cube plans allocate the hierarchy priced by their child L0 plan.

        The 272 case used to retain both constant branches produced by pipeline
        peeling and allocate two full L0C accumulators.  The 512 case has a
        ``64 + 64 + 32`` child K decomposition: the final 32-wide tail is a
        serial phase that must reuse one of the two rolled Left/Right slots,
        rather than being hoisted into the enclosing GM-to-L1 pipeline and
        allocating a third operand panel.

        ``PYPTO_AUTOFUSE_FORCE_PLAN`` is process-cached, so each parameter runs
        in a fresh interpreter.  Reaching the final AIC function also exercises
        ``AllocateMemoryAddr`` and proves the selected finite-cost plan is
        physically buildable.
        """
        script = textwrap.dedent(
            f"""
            import re

            import pypto.language as pl
            from pypto import backend
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={{"auto_fuse": True}})
                def mm(
                    self,
                    a: pl.Tensor[[{size}, {size}], pl.FP32],
                    b: pl.Tensor[[{size}, {size}], pl.FP32],
                ) -> pl.Tensor[[{size}, {size}], pl.FP32]:
                    out: pl.Tensor[[{size}, {size}], pl.FP32] = pl.matmul(a, b)
                    return out

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            aic = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIC"]
            aiv = [f for _, f in lowered.functions.items() if str(f.func_type) == "FunctionType.AIV"]
            assert len(aic) == 1, [f.name for _, f in lowered.functions.items()]
            assert not aiv, [f.name for _, f in lowered.functions.items()]

            body = aic[0].as_python()
            assert body.count("pl.tile.alloc(pl.Mem.Acc") == 1, body
            assert re.search(
                r"pl\\.tile\\.move\\([^)]*target_memory=pl\\.Mem\\.Acc",
                body,
                re.DOTALL,
            ) is None, body
            assert "pipeline_serial_phase" not in body, body
            """
        )
        if expected_left is not None:
            script += textwrap.dedent(
                f"""
                assert body.count("pl.tile.alloc(pl.Mem.Left") == {expected_left}, body
                assert body.count("pl.tile.alloc(pl.Mem.Right") == {expected_right}, body
                """
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "0"
        env["PYPTO_AUTOFUSE_STRICT"] = "1"
        env["PYPTO_AUTOFUSE_EXACT_L0_COST"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_PLAN"] = force
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    def test_mixed_declines_transposed_matmul_before_solving(self):
        """Mixed v0 must not rebuild a transposed matmul as default-orientation."""
        script = textwrap.dedent(
            """
            import torch
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP32],
                    b: pl.Tensor[[64, 64], pl.FP32],
                    bias: pl.Tensor[[1, 64], pl.FP32],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    mm: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b, a_trans=True)
                    out: pl.Tensor[[64, 64], pl.FP32] = pl.add(mm, bias)
                    return out

            planned = passes.auto_fuse()(Prog)
            text = next(f for _, f in planned.functions.items() if f.name == "epilogue").as_python()
            assert "a_trans=True" in text, text
            assert "pl.split(pl.SplitMode.UP_DOWN)" not in text, text

            namespace = {}
            exec(torch_codegen(planned, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(64, 64)
            b = torch.randn(64, 64)
            bias = torch.randn(1, 64)
            got = namespace["epilogue"](a, b, bias)
            expected = a.T @ b + bias
            assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4), (got - expected).abs().max()
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    def test_mixed_declines_low_precision_accumulator_handoff(self):
        """FP16 C->V waits for an explicit FP32 K carry and final narrow."""
        script = textwrap.dedent(
            """
            import os
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[64, 8192], pl.FP16],
                    b: pl.Tensor[[8192, 64], pl.FP16],
                    bias: pl.Tensor[[1, 64], pl.FP16],
                ) -> pl.Tensor[[64, 64], pl.FP16]:
                    mm: pl.Tensor[[64, 64], pl.FP16] = pl.matmul(a, b)
                    out: pl.Tensor[[64, 64], pl.FP16] = pl.add(mm, bias)
                    return out

            planned = passes.auto_fuse()(Prog)
            text = next(f for _, f in planned.functions.items() if f.name == "epilogue").as_python()
            assert "pl.split(pl.SplitMode.UP_DOWN" not in text, text
            assert "pl.tensor.matmul(" in text, text
            assert "pl.tensor.add(" in text, text

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            lowered_text = lowered.as_python()
            assert "pl.tile.tpush_to_aiv(" not in lowered_text, lowered_text

            @pl.program
            class HeteroProg:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP16],
                    b: pl.Tensor[[64, 64], pl.FP32],
                    bias: pl.Tensor[[1, 64], pl.FP32],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    mm: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                    out: pl.Tensor[[64, 64], pl.FP32] = pl.add(mm, bias)
                    return out

            hetero = passes.auto_fuse()(HeteroProg)
            hetero_text = next(
                f for _, f in hetero.functions.items() if f.name == "epilogue"
            ).as_python()
            assert "pl.split(pl.SplitMode.UP_DOWN" not in hetero_text, hetero_text

            @pl.program
            class PromotedBiasProg:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP32],
                    b: pl.Tensor[[64, 64], pl.FP32],
                    bias: pl.Tensor[[1, 64], pl.FP16],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    mm: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                    out: pl.Tensor[[64, 64], pl.FP32] = pl.add(mm, bias)
                    return out

            promoted = passes.auto_fuse()(PromotedBiasProg)
            promoted_text = next(
                f for _, f in promoted.functions.items() if f.name == "epilogue"
            ).as_python()
            assert "pl.split(pl.SplitMode.UP_DOWN" not in promoted_text, promoted_text

            @pl.program
            class SupportedFp16Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[64, 8192], pl.FP16],
                    b: pl.Tensor[[8192, 64], pl.FP16],
                    bias: pl.Tensor[[1, 64], pl.FP32],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    mm: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(
                        a, b, out_dtype=pl.FP32
                    )
                    out: pl.Tensor[[64, 64], pl.FP32] = pl.add(mm, bias)
                    return out

            os.environ["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
            supported = passes.auto_fuse()(SupportedFp16Prog)
            supported_text = next(
                f for _, f in supported.functions.items() if f.name == "epilogue"
            ).as_python()
            assert "pl.split(pl.SplitMode.UP_DOWN, slot_num=8)" in supported_text, supported_text
            assert "pl.tensor.matmul_acc(" in supported_text, supported_text
            supported_lowered = PassManager.get_strategy(
                OptimizationStrategy.Default
            ).run_passes(SupportedFp16Prog)
            supported_lowered_text = supported_lowered.as_python()
            assert "pl.tile.tpush_to_aiv(" in supported_lowered_text, supported_lowered_text
            assert "pl.tile.tpop_from_aic(" in supported_lowered_text, supported_lowered_text
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "1"
        env.pop("PYPTO_AUTOFUSE_FORCE_MERGE", None)
        env.pop("PYPTO_AUTOFUSE_FORCE_PLAN", None)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    def test_mixed_natural_grid_accounts_for_fifo_reservation(self):
        """The former natural 192x48 plan must not overflow Vec after FIFO expansion."""
        script = textwrap.dedent(
            """
            import re
            import pypto.language as pl
            from pypto import backend
            from pypto.backend import BackendType
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[192, 64], pl.FP32],
                    b: pl.Tensor[[64, 384], pl.FP32],
                    bias: pl.Tensor[[192, 384], pl.FP32],
                ) -> pl.Tensor[[192, 384], pl.FP32]:
                    mm: pl.Tensor[[192, 384], pl.FP32] = pl.matmul(a, b)
                    out: pl.Tensor[[192, 384], pl.FP32] = pl.add(mm, bias)
                    return out

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            text = lowered.as_python()
            sizes = [int(x) for x in re.findall(r"reserve_buffer\\([^\\n]*size=(\\d+)", text)]
            assert sizes, text
            assert max(sizes) <= 188416, (sizes, text)
            assert "pl.tile.tpush_to_aiv(" in text, text
            assert "pl.tile.tpop_from_aic(" in text, text
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    def test_mixed_replays_repeated_cube_value_operand(self):
        """add(mm, mm) is one internal dependency with two semantic operand uses."""
        script = textwrap.dedent(
            """
            import torch
            import pypto.language as pl
            from pypto import backend, passes
            from pypto.backend import BackendType
            from pypto.debug import torch_codegen
            from pypto.ir.pass_manager import OptimizationStrategy, PassManager

            backend.set_backend_type(BackendType.Ascend910B)

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def epilogue(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP32],
                    b: pl.Tensor[[64, 64], pl.FP32],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    mm: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                    out: pl.Tensor[[64, 64], pl.FP32] = pl.add(mm, mm)
                    return out

            planned = passes.auto_fuse()(Prog)
            text = next(f for _, f in planned.functions.items() if f.name == "epilogue").as_python()
            assert "pl.split(pl.SplitMode.UP_DOWN, slot_num=8)" in text, text
            assert "pl.tensor.add(mm_mixed_tile, mm_mixed_tile)" in text, text

            namespace = {}
            exec(torch_codegen(planned, run_all_spmd_blocks=True), namespace)
            torch.manual_seed(0)
            a = torch.randn(64, 64)
            b = torch.randn(64, 64)
            got = namespace["epilogue"](a, b)
            expected = 2 * (a @ b)
            assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4), (got - expected).abs().max()

            lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
            lowered_text = lowered.as_python()
            assert "pl.tile.tpush_to_aiv(" in lowered_text, lowered_text
            assert "pl.tile.tpop_from_aic(" in lowered_text, lowered_text
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(os.path.abspath("python"))
        env["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
        env["PYPTO_AUTOFUSE_MIXED"] = "1"
        env["PYPTO_AUTOFUSE_FORCE_MERGE"] = "all"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
