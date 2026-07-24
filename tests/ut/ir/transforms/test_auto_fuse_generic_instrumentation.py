# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Instrumentation: prove the AutoFuse generic driver actually ENGAGES.

The differential golden net (``test_auto_fuse_emit_golden.py``) checks NUMERIC
equality of legacy vs generic emit — but numeric equality alone cannot tell a
case the generic path *owned* from one it silently *declined* (the generic
function returns ``std::nullopt`` and the legacy tiler quietly produced the same
correct output). CI would stay green either way, hiding a driver that never runs.

This file closes that gap by asserting on the current emitters' log markers
with the flag ON:

- the generic vector driver owns pointwise groups;
- ``CubeSchedulePlan`` owns uniform and clamped-overlap matmul grids.

Together with the strict-mode golden run (``PYPTO_AUTOFUSE_STRICT=1`` — Tier-B
illegal-plan conditions abort), this makes both halves observable: the driver
engages where expected, and every decline is either a logged capability gap or a
loud illegal-plan failure — never an invisible fallback.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import LogLevel, set_log_level


def _autofuse_log(program, capfd, monkeypatch) -> str:
    """Run AutoFuse on ``program`` with the generic driver ON at INFO logging,
    and return everything it wrote to stdout+stderr (where the C++ markers land)."""
    monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
    set_log_level(LogLevel.INFO)
    try:
        passes.auto_fuse()(program)
    finally:
        set_log_level(LogLevel.WARN)
    captured = capfd.readouterr()
    return captured.out + captured.err


class TestGenericDriverEngages:
    """The generic driver runs where expected — not a silent legacy fallback."""

    def test_generic_owns_elementwise_group(self, ascend_backend, capfd, monkeypatch):
        """A tiled pointwise chain routes through the generic Elementwise rule."""

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(
                self, a: pl.Tensor[[512, 512], pl.FP32], b: pl.Tensor[[512, 512], pl.FP32]
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                c: pl.Tensor[[512, 512], pl.FP32] = pl.add(a, b)
                d: pl.Tensor[[512, 512], pl.FP32] = pl.mul(c, b)
                return d

        log = _autofuse_log(Prog, capfd, monkeypatch)
        assert "tiled by the generic driver" in log, (
            "generic Elementwise rule did not own the pointwise group (silent legacy fallback?)\n" + log
        )

    def test_cube_plan_owns_uniform_matmul(self, ascend_backend, capfd, monkeypatch):
        """A square matmul routes through CubeSchedulePlan with a uniform grid."""

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        log = _autofuse_log(Prog, capfd, monkeypatch)
        assert "AutoFuse[cube-plan]" in log and "spatial_policy=uniform" in log, (
            "CubeSchedulePlan did not own the uniform matmul\n" + log
        )

    def test_cube_plan_owns_clamped_overlap_grid(self, ascend_backend, capfd, monkeypatch):
        """A non-divisor matmul grid is emitted with explicit clamped ownership."""

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self, a: pl.Tensor[[272, 272], pl.FP32], b: pl.Tensor[[272, 272], pl.FP32]
            ) -> pl.Tensor[[272, 272], pl.FP32]:
                c: pl.Tensor[[272, 272], pl.FP32] = pl.matmul(a, b)
                return c

        log = _autofuse_log(Prog, capfd, monkeypatch)
        assert "AutoFuse[cube-plan]" in log and "spatial_policy=clamped_overlap" in log, (
            "CubeSchedulePlan did not expose clamped-overlap ownership\n" + log
        )

    def test_generic_owns_multi_sink_fork(self, ascend_backend, capfd, monkeypatch):
        """A fork — two sink ops sharing an input — is one fused group with 2 live-outs, emitted
        by the multi-sink path (each sink assembled into its own output, in execution order)."""

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def fork(
                self, x: pl.Tensor[[256, 256], pl.FP32]
            ) -> tuple[pl.Tensor[[256, 256], pl.FP32], pl.Tensor[[256, 256], pl.FP32]]:
                c: pl.Tensor[[256, 256], pl.FP32] = pl.add(x, 1.0)
                a: pl.Tensor[[256, 256], pl.FP32] = pl.mul(c, 2.0)
                b: pl.Tensor[[256, 256], pl.FP32] = pl.mul(c, 3.0)
                return a, b

        log = _autofuse_log(Prog, capfd, monkeypatch)
        assert "multi-sink group" in log and "2 live-outs" in log, (
            "multi-sink fork did not route through the multi-sink path (declined or single-sink?)\n" + log
        )


class TestVectorPipelining:
    """The vector emit is software-pipelined so DMA overlaps compute — the max(compute,ddr)
    roofline the cost model prices (db_roofline), not a serial load->compute->store."""

    def test_vector_group_emits_pipeline_strips(self, ascend_backend, capfd, monkeypatch):
        """A vector group is chunked into >=2 pipeline strips (emit side)."""
        import re  # noqa: PLC0415

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(
                self, a: pl.Tensor[[512, 512], pl.FP32], b: pl.Tensor[[512, 512], pl.FP32]
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                c: pl.Tensor[[512, 512], pl.FP32] = pl.add(a, b)
                d: pl.Tensor[[512, 512], pl.FP32] = pl.mul(c, b)
                return d

        log = _autofuse_log(Prog, capfd, monkeypatch)
        m = re.search(r"(\d+) planned strips \(stage=2\)", log)
        assert m and int(m.group(1)) >= 2, f"vector group not pipelined (expected >=2 strips)\n{log}"

    def test_vector_pipeline_realizes_overlap(self, ascend_backend, tmp_path, monkeypatch):
        """The LOWERED vector kernel loads each input per-strip into distinct ping-pong buffers
        (LowerPipelineLoops unroll+tag -> CanonicalizeIOOrder cluster -> MemoryReuse), so DMA
        overlaps compute. A loop's structural presence alone is not proof of overlap; per-strip
        loading (more tloads than inputs) is the realized-pipeline signature."""
        import os  # noqa: PLC0415

        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.function(attrs={"auto_fuse": True})
        def pw(
            a: pl.Tensor[[512, 512], pl.FP32], b: pl.Tensor[[512, 512], pl.FP32]
        ) -> pl.Tensor[[512, 512], pl.FP32]:
            c: pl.Tensor[[512, 512], pl.FP32] = pl.add(a, b)
            d: pl.Tensor[[512, 512], pl.FP32] = pl.mul(c, b)
            return d

        out = tmp_path / "pw_pipe"
        ir.compile(ir.Program([pw], "pw", ir.Span.unknown()), output_dir=str(out), skip_ptoas=True)
        aiv = ""
        for root, _, files in os.walk(str(out)):
            if "aiv" in root:
                for f in files:
                    if f.endswith(".pto"):
                        aiv = open(os.path.join(root, f), encoding="utf-8").read()
        assert aiv, "no aiv (vector) kernel emitted"
        n_loads = aiv.count("pto.tload")
        # 2 inputs (a, b); a SERIAL kernel loads each once (2 tloads). A pipelined kernel loads each
        # input PER STRIP (>=2 strips -> >2 tloads) into distinct buffers -> load overlaps compute.
        assert n_loads > 2, (
            f"vector kernel not pipelined: expected per-strip loads (>2 tloads for 2 inputs), got {n_loads}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
