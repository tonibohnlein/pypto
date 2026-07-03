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

This file closes that gap by asserting on the driver's own log markers with the
flag ON:

- generic OWNS the vector + uniform-matmul surface  ("tiled by the generic driver")
- generic DECLINES non-uniform grids LOUDLY, not silently ("CAPABILITY decline")

Together with the strict-mode golden run (``PYPTO_AUTOFUSE_STRICT=1`` — Tier-B
illegal-plan conditions abort), this makes both halves observable: the driver
engages where expected, and every decline is either a logged capability gap or a
loud illegal-plan failure — never an invisible fallback.
"""

import pypto.language as pl
import pytest
from pypto import passes
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

    def test_generic_owns_uniform_matmul(self, ascend_backend, capfd, monkeypatch):
        """A square (uniform-grid) matmul routes through the generic MatMul rule."""

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        log = _autofuse_log(Prog, capfd, monkeypatch)
        assert "matmul group" in log and "tiled by the generic driver" in log, (
            "generic MatMul rule did not own the uniform matmul (silent legacy fallback?)\n" + log
        )

    def test_generic_declines_nonuniform_grid_loudly(self, ascend_backend, capfd, monkeypatch):
        """A non-uniform (parts_m/parts_n) grid is a LOGGED capability decline, not silent.

        The solver picks a non-uniform grid for a rectangular shape (e.g. parts_n=3 on
        N=256); the v1 emitter only tiles uniform grids, so it runs untiled — correctly,
        but at lower fidelity. That decline MUST be visible (so the bake window can
        measure it), never a silent nullopt.
        """

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self, a: pl.Tensor[[128, 192], pl.FP32], b: pl.Tensor[[192, 256], pl.FP32]
            ) -> pl.Tensor[[128, 256], pl.FP32]:
                c: pl.Tensor[[128, 256], pl.FP32] = pl.matmul(a, b)
                return c

        log = _autofuse_log(Prog, capfd, monkeypatch)
        assert "CAPABILITY decline" in log and "non-uniform spatial grid" in log, (
            "non-uniform grid decline was not logged (silent fallback masks the fidelity gap)\n" + log
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
