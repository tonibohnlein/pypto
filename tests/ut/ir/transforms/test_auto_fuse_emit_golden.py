# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical GOLDEN / differential-test harness for the AutoFuse emit.

This is the migration safety net for replacing the three per-shape tilers
(``TileMatmul``, ``TileChainedMatmul``, ``TilePointwiseGroup`` in
``src/ir/transforms/auto_fuse_pass.cpp``) with a single generic tile-and-fuse
driver. Every case asserts that the EMIT — the post-AutoFuse tensor IR, with ALL
SPMD blocks executed and tile-stitched via
``torch_codegen(..., run_all_spmd_blocks=True)`` — reproduces the UNFUSED
reference computation to fp tolerance.

Why numerical, not IR-structural: the refactor changes the emitted IR shape *by
design*, so an IR snapshot would churn. A tile-stitch-vs-reference check is
invariant to how the tiling is expressed, so the CURRENT legacy tilers and the
FUTURE generic driver must both pass this file identically — flip the driver
behind a flag and diff numerically, not by IR shape.

Coverage = the surface the three tilers handle today, one representative shape
per tiler plus a multi-user DAG (the diamond, which stresses on-chip
materialization of a shared intermediate) and a rectangular / non-square matmul.
All pinned to Ascend910B (the grounded cost model's backend), since the solver's
tile/split decision is backend-specific.

NOTE: the numeric check runs the post-AutoFuse *tensor* IR, not the fully-lowered
kernel — so it validates emit correctness independently of downstream lowering
limits (e.g. the chained-matmul lowering is blocked on upstream #1908, but its
emit is still checked here).
"""

import pypto.language as pl
import pytest
from pypto import passes


def _emit_matches_reference(program, entry, inputs, ref, *, rtol=1e-4, atol=1e-4):
    """Run AutoFuse, execute the emit's every SPMD block into the shared output,
    and assert the tile-stitched full result equals the unfused reference.

    Tolerance note: most paths reproduce the reference to ~1e-6 because the tiling is
    a pure partition of the same work in the same order. The exception is **split-K**:
    partial products accumulate via atomic-add across K-slice blocks, so the sum is
    reassociated vs. the single unfused matmul, and fp rounding differs. Split-K cases
    therefore pass a looser ``rtol``/``atol`` — an exact/bit-match check would be
    falsely-red. (Reductions do NOT reassociate: S2 pins the full reduced axis on one
    core, so softmax stays tight.)
    """
    torch = pytest.importorskip("torch")
    from pypto.debug import torch_codegen  # noqa: PLC0415

    after = passes.auto_fuse()(program)
    namespace: dict = {}
    exec(torch_codegen(after, run_all_spmd_blocks=True), namespace)  # noqa: S102
    out = namespace[entry](*inputs)
    diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), f"{entry}: max abs diff {diff:.3e}"


class TestAutoFuseEmitGolden:
    """Emit == unfused reference, across the three tilers' surface.

    Both the legacy per-shape tilers and the future generic tile-and-fuse driver
    must pass every case here.
    """

    # ---- TileMatmul (single matmul: split-K seed + atomic merge) ----

    def test_matmul_square(self, ascend_backend):
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        torch.manual_seed(0)
        a, b = torch.randn(64, 64), torch.randn(64, 64)
        _emit_matches_reference(Prog, "mm", (a, b), a @ b)

    def test_matmul_rectangular(self, ascend_backend):
        """Non-square M!=N and a K != M,N — exercises the [M,K]@[K,N] operand slicing."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(self, a: pl.Tensor[[128, 192], pl.FP32], b: pl.Tensor[[192, 256], pl.FP32]) -> pl.Tensor[[128, 256], pl.FP32]:
                c: pl.Tensor[[128, 256], pl.FP32] = pl.matmul(a, b)
                return c

        torch.manual_seed(1)
        a, b = torch.randn(128, 192), torch.randn(192, 256)
        _emit_matches_reference(Prog, "mm", (a, b), a @ b)

    def test_matmul_split_k(self, ascend_backend):
        """Force split-K explicitly: a small output (few spatial tiles) with a deep K, so
        the solver splits the contraction across cores to fill them — exercising the
        riskiest new path (tiled zero-seed + DDR atomic-add merge + the flat-index decode).
        Looser tolerance because atomic-add reassociates the K partials vs. the unfused
        matmul (see ``_emit_matches_reference``)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(self, a: pl.Tensor[[64, 512], pl.FP32], b: pl.Tensor[[512, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        torch.manual_seed(6)
        a, b = torch.randn(64, 512), torch.randn(512, 64)
        _emit_matches_reference(Prog, "mm", (a, b), a @ b, rtol=3e-3, atol=3e-3)

    def test_matmul_ragged(self, ascend_backend):
        """Non-power-of-two, awkwardly-divisible M/N (48x112, K=96): the matmul rule's
        ragged M/N + the sink write-back (R1's stated raggedness locus). Green today via
        the current emit; when the generic driver tiles it with valid/alloc mask-to-full,
        it must still match. Loose tolerance in case the solver split-Ks."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(self, a: pl.Tensor[[48, 96], pl.FP32], b: pl.Tensor[[96, 112], pl.FP32]) -> pl.Tensor[[48, 112], pl.FP32]:
                c: pl.Tensor[[48, 112], pl.FP32] = pl.matmul(a, b)
                return c

        torch.manual_seed(7)
        a, b = torch.randn(48, 96), torch.randn(96, 112)
        _emit_matches_reference(Prog, "mm", (a, b), a @ b, rtol=3e-3, atol=3e-3)

    # ---- TileChainedMatmul (fused chain; emit only — lowering is #1908-blocked) ----

    def test_chained_matmul(self, ascend_backend):
        torch = pytest.importorskip("torch")

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

        torch.manual_seed(2)
        a, b, d = torch.randn(128, 256), torch.randn(256, 128), torch.randn(128, 256)
        _emit_matches_reference(Prog, "chain", (a, b, d), (a @ b) @ d)

    # ---- TilePointwiseGroup (single + chained pointwise, and the multi-user diamond) ----

    def test_pointwise_single(self, ascend_backend):
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                return c

        torch.manual_seed(3)
        a = torch.randn(4096, 384)
        _emit_matches_reference(Prog, "pw", (a,), a + 1.0)

    def test_pointwise_chain(self, ascend_backend):
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw2(self, a: pl.Tensor[[4096, 384], pl.FP32]) -> pl.Tensor[[4096, 384], pl.FP32]:
                t: pl.Tensor[[4096, 384], pl.FP32] = pl.add(a, 1.0)
                c: pl.Tensor[[4096, 384], pl.FP32] = pl.mul(t, 2.0)
                return c

        torch.manual_seed(4)
        a = torch.randn(4096, 384)
        _emit_matches_reference(Prog, "pw2", (a,), (a + 1.0) * 2.0)

    def test_vector_dag_diamond(self, ascend_backend):
        """The diamond c -> {d,e} -> g -> f + skip edge c -> f: ``c`` has 3 in-group
        users, so this is the on-chip materialization / multi-user case (XLA's
        recompute trap) that tile-granularity fusion must get right."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def dag(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                c: pl.Tensor[[128, 128], pl.FP32] = pl.add(a, b)
                d: pl.Tensor[[128, 128], pl.FP32] = pl.add(c, 1.0)
                e: pl.Tensor[[128, 128], pl.FP32] = pl.add(c, 2.0)
                g: pl.Tensor[[128, 128], pl.FP32] = pl.mul(d, e)
                f: pl.Tensor[[128, 128], pl.FP32] = pl.add(g, c)
                return f

        torch.manual_seed(5)
        a, b = torch.randn(128, 128), torch.randn(128, 128)
        c = a + b
        _emit_matches_reference(Prog, "dag", (a, b), (c + 1.0) * (c + 2.0) + c)

    def test_pointwise_ragged(self, ascend_backend):
        """Non-16-aligned M AND N (130x66): the hardest tail for the future driver's
        valid/alloc mask-to-full (R1). Elementwise, so exact today; must stay exact when
        the driver tiles it raggedly."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(self, a: pl.Tensor[[130, 66], pl.FP32]) -> pl.Tensor[[130, 66], pl.FP32]:
                c: pl.Tensor[[130, 66], pl.FP32] = pl.add(a, 1.0)
                return c

        torch.manual_seed(8)
        a = torch.randn(130, 66)
        _emit_matches_reference(Prog, "pw", (a,), a + 1.0)

    # ---- Reduction rule (the v1-new reduction path: softmax) ----

    def test_softmax_reduction(self, ascend_backend):
        """Row-wise softmax — the reduction rule: pinned reduced axis + fused
        max/sub/exp/sum/div, with the intermediate on-chip. Tight tolerance: the reduced
        axis is pinned full on one core (S2), so it does NOT reassociate vs. the reference
        (unlike split-K matmul). Untiled/fallback today; the reduction rule tiles the free
        axis when the driver lands, and this must still match."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def softmax(self, a: pl.Tensor[[256, 128], pl.FP32]) -> pl.Tensor[[256, 128], pl.FP32]:
                m: pl.Tensor[[256, 1], pl.FP32] = pl.row_max(a)
                s: pl.Tensor[[256, 128], pl.FP32] = pl.row_expand_sub(a, m)
                e: pl.Tensor[[256, 128], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(e)
                c: pl.Tensor[[256, 128], pl.FP32] = pl.row_expand_div(e, d)
                return c

        torch.manual_seed(9)
        a = torch.randn(256, 128)
        _emit_matches_reference(Prog, "softmax", (a,), torch.softmax(a, dim=1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
