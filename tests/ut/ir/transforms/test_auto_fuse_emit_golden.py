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


@pytest.fixture(params=["legacy", "generic"], autouse=True)
def emit_path(request, monkeypatch):
    """Run EVERY golden case both ways — flag OFF (legacy per-shape tilers) and flag ON
    (the generic tile-and-fuse driver) — so this file is a true DIFFERENTIAL net: both
    paths must reproduce the unfused reference. This is the migration guard that lets the
    driver be promoted / the legacy tilers retired with CI confidence, not just manual
    checks. The C++ flag re-reads the env var per compile, so monkeypatch toggles it
    in-process (see GenericEmitEnabled in auto_fuse_pass.cpp)."""
    if request.param == "generic":
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        # STRICT mode turns every Tier-B decline (a plan the emitter contract forbids —
        # mis-pinned reduction, split ∤ K/16, mixed cube+vector, multi-sink) into a hard
        # failure instead of a silent legacy fallback. Enabling it on the generic golden
        # run makes CI ENFORCE the tier invariant "no Tier-B fires on the v1 surface" —
        # otherwise a future golden case (or a solver change) that trips a Tier-B would
        # pass green via the legacy fallback, masking exactly the bug the tiers exist to
        # surface. Capability declines (non-uniform grid, vector split>1) do NOT abort
        # under strict, so legitimate deferred-fidelity cases stay green.
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
    else:
        monkeypatch.delenv("PYPTO_AUTOFUSE_GENERIC_EMIT", raising=False)
        monkeypatch.delenv("PYPTO_AUTOFUSE_STRICT", raising=False)
    return request.param


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
    # A multi-return kernel yields a tuple of outputs (one per lifted Out param); compare each
    # against the matching reference. A single return is wrapped so the loop covers both.
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    assert len(outs) == len(refs), f"{entry}: expected {len(refs)} outputs, got {len(outs)}"
    for i, (o, r) in enumerate(zip(outs, refs)):
        diff = (o - r).abs().max().item()
        assert torch.allclose(o, r, rtol=rtol, atol=atol), f"{entry}: output {i} max abs diff {diff:.3e}"


class TestAutoFuseEmitGolden:
    """Emit == unfused reference, across the three tilers' surface.

    Both the legacy per-shape tilers and the future generic tile-and-fuse driver
    must pass every case here.
    """

    # ---- TileMatmul (single matmul: ordered split-K partial merge) ----

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
        ordered first-partial + atomic-rest merge and the flat-index decode.
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
        """Row-wise softmax — the reduction rule + R1 ragged: pinned reduced axis + fused
        max/sub/exp/sum/div, with the intermediate on-chip. Tight tolerance: the reduced
        axis is pinned full on one core (S2), so it does NOT reassociate vs. the reference
        (unlike split-K matmul). Shape 256x128 has a RAGGED free axis (the solver's non-
        uniform grid, h=6, 256%6!=0), so the flag-on path exercises R1 (ceil grid + clamped
        idempotent-overlap tail) tiling it across the vector cores; flag off / pre-R1 it
        lowers via the untiled fallback — both must match."""
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

    def test_softmax_reduced_axis_ragged(self, ascend_backend):
        """Softmax over a RAGGED REDUCED axis (N=66): the reduction axis itself is ragged, so
        the generic driver pads it 66->72 (valid=66) and the reductions run over the padded
        axis. A device experiment proved trowsum/tcolsum bound the reduction by valid (the
        padded lanes are excluded), so the guard that previously declined this is lifted; this
        checks the emit is numerically correct (both paths)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def softmax(self, a: pl.Tensor[[256, 66], pl.FP32]) -> pl.Tensor[[256, 66], pl.FP32]:
                m: pl.Tensor[[256, 1], pl.FP32] = pl.row_max(a)
                s: pl.Tensor[[256, 66], pl.FP32] = pl.row_expand_sub(a, m)
                e: pl.Tensor[[256, 66], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(e)
                c: pl.Tensor[[256, 66], pl.FP32] = pl.row_expand_div(e, d)
                return c

        torch.manual_seed(10)
        a = torch.randn(256, 66)
        _emit_matches_reference(Prog, "softmax", (a,), torch.softmax(a, dim=1))

    def test_col_sum_ragged_split(self, ascend_backend):
        """Bare col_sum sink whose solver plan gangs S cores over the reduced axis (split>1)
        on a RAGGED free axis (N=100) — the exact shape that made the S2 atomic-add merge
        DOUBLE-COUNT before the fix (the ceil+clamp free-axis overlap summed twice under an
        atomic assemble). The emit now declines the un-realizable ragged split to the CORRECT
        non-split body; this asserts the generic emit equals the unfused column reduction, so a
        regression that re-enables the overlapping atomic split turns this red. Tight tolerance:
        the non-split body pins the full reduced axis on one core (no atomic reassociation)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def cs(self, a: pl.Tensor[[256, 100], pl.FP32]) -> pl.Tensor[[1, 100], pl.FP32]:
                c: pl.Tensor[[1, 100], pl.FP32] = pl.col_sum(a)
                return c

        torch.manual_seed(11)
        a = torch.randn(256, 100)
        _emit_matches_reference(Prog, "cs", (a,), a.sum(dim=0, keepdim=True))

    def test_col_reduction_cone_not_split_locally(self, ascend_backend):
        """A col_sum whose cone contains a PRIOR M-reduction: m=col_max(x); y=x-m; z=col_sum(y).
        If an S2 split replayed the whole cone per M-slice, col_max would be computed LOCALLY per
        slice (each slice subtracts its own max) → wrong. Adversarial data (top half 0, bottom half
        10) makes a local-max error a large, obvious miss. Guards the S2 cone-safety gate (the emit
        must not split a cone with an upstream M-reduction) end-to-end, both legacy and generic."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def cs(self, x: pl.Tensor[[128, 256], pl.FP32]) -> pl.Tensor[[1, 256], pl.FP32]:
                m: pl.Tensor[[1, 256], pl.FP32] = pl.col_max(x)
                y: pl.Tensor[[128, 256], pl.FP32] = pl.sub(x, m)
                z: pl.Tensor[[1, 256], pl.FP32] = pl.col_sum(y)
                return z

        x = torch.zeros(128, 256)
        x[64:, :] = 10.0
        ref = (x - x.max(dim=0, keepdim=True).values).sum(dim=0, keepdim=True)  # sum(x - global_max)
        _emit_matches_reference(Prog, "cs", (x,), ref)

    def test_col_sum_split_atomic_merge(self, ascend_backend):
        """Bare col_sum sink on a CLEAN shape (M=128, N=256) where the solver splits the
        reduced axis across cores (now S=8, a divisor of the reduced fractal count 128/16=8,
        so IM/S=16 is granule-aligned and the split is emit-realizable). Exercises the S2
        atomic-add merge itself: S disjoint 16-row partials zero-seeded then atomic-added into
        the [1,256] output. Looser tolerance because atomic-add reassociates the partials vs.
        the single unfused reduction (same as split-K matmul)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def cs(self, a: pl.Tensor[[128, 256], pl.FP32]) -> pl.Tensor[[1, 256], pl.FP32]:
                c: pl.Tensor[[1, 256], pl.FP32] = pl.col_sum(a)
                return c

        torch.manual_seed(13)
        a = torch.randn(128, 256)
        _emit_matches_reference(Prog, "cs", (a,), a.sum(dim=0, keepdim=True), rtol=1e-4, atol=1e-4)

    def test_interleaved_groups(self, ascend_backend):
        """Two INDEPENDENT elementwise chains (exp->neg on x, exp->neg on y) interleaved in SSA
        order. The solver puts each chain in its own fused group, so the groups interleave in body
        order — which the contiguous-run emit used to FRAGMENT into four single-op scopes (each
        chain's intermediate round-tripping DDR). ReorderBodyByGroup now clusters each group into
        one contiguous scope; this asserts the emit still equals the two unfused chains (both
        outputs), guarding the reorder + the multi-return wiring."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def two(self, x: pl.Tensor[[256, 128], pl.FP32], y: pl.Tensor[[256, 128], pl.FP32]):
                a1: pl.Tensor[[256, 128], pl.FP32] = pl.exp(x)
                b1: pl.Tensor[[256, 128], pl.FP32] = pl.exp(y)
                a2: pl.Tensor[[256, 128], pl.FP32] = pl.neg(a1)
                b2: pl.Tensor[[256, 128], pl.FP32] = pl.neg(b1)
                return a2, b2

        torch.manual_seed(14)
        x, y = torch.randn(256, 128), torch.randn(256, 128)
        _emit_matches_reference(Prog, "two", (x, y), (-torch.exp(x), -torch.exp(y)))

    def test_attention_block_output_wired(self, ascend_backend):
        """A full single-head attention block p=softmax(q@k/sqrt(d)); out=p@v — a matmul-ending
        fused function. Its return is a `tensor.matmul` output (no create/assemble to lift), so
        MaybeLiftReturnToOutParam synthesizes an Out param + a full-copy assemble to wire it (else
        the by-position device harness sees an unwritten, all-zero output). This asserts the emit
        reproduces the reference — exercising the matmul-ending output wiring end-to-end. Loose
        tolerance: two matmuls reassociate vs. the reference."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def attn(
                self,
                q: pl.Tensor[[128, 64], pl.FP32],
                k: pl.Tensor[[64, 128], pl.FP32],
                v: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                s: pl.Tensor[[128, 128], pl.FP32] = pl.matmul(q, k)
                sc: pl.Tensor[[128, 128], pl.FP32] = pl.mul(s, 0.125)
                m: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(sc)
                dd: pl.Tensor[[128, 128], pl.FP32] = pl.row_expand_sub(sc, m)
                e: pl.Tensor[[128, 128], pl.FP32] = pl.exp(dd)
                sm: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(e)
                p: pl.Tensor[[128, 128], pl.FP32] = pl.row_expand_div(e, sm)
                o: pl.Tensor[[128, 64], pl.FP32] = pl.matmul(p, v)
                return o

        torch.manual_seed(15)
        q, k, v = torch.randn(128, 64), torch.randn(64, 128), torch.randn(128, 64)
        ref = torch.softmax((q @ k) * 0.125, dim=-1) @ v
        _emit_matches_reference(Prog, "attn", (q, k, v), ref, rtol=2e-3, atol=2e-3)

    def test_scalar_param_broadcast(self, ascend_backend):
        """A scalar In-param (broadcast scale) carried as an operand, not a tiled tensor. Before
        the fix, registering it as a solver tensor CHECK-crashed the whole compile ('not
        tensor-typed'); now it is skipped and the pointwise mul fuses with the scalar threaded
        through the emit as-is. Guards the graceful-skip path (both legacy and generic)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def scaled(self, a: pl.Tensor[[64, 64], pl.FP32], s: pl.Scalar[pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.mul(a, s)
                return c

        torch.manual_seed(12)
        a = torch.randn(64, 64)
        _emit_matches_reference(Prog, "scaled", (a, 2.5), a * 2.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
