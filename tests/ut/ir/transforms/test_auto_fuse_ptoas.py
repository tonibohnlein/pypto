# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ptoas-assembly gate for the AutoFuse generic emitter — the layer the numeric net can't cover.

The differential golden net (``test_auto_fuse_emit_golden.py``) checks NUMERIC equality via
``torch_codegen``, which executes the post-AutoFuse *tensor* IR and therefore has NO model of
hardware tile-alignment. This file closes that gap: it compiles each shape END-TO-END through the
Default pipeline WITH ptoas (``skip_ptoas=False``) and asserts the emitted kernels actually
ASSEMBLE. It caught the ragged-tile alignment gap (cube: rows/cols must be ×16; vector:
contiguous-axis byte extent must be ×32) that the numeric net was blind to.

Runs only when a ptoas toolchain is available (``PTOAS_ROOT`` set) — skipped otherwise, like the
``torch`` importorskip pattern in the golden net. No device is needed: ptoas *assembles* the
kernels; it does not execute them.

Expected status (Phase 1 = vector free-axis padding):
  - vector aligned / cube aligned            -> assemble (baseline).
  - vector free-axis-ragged (pointwise, softmax) -> assemble AFTER Phase-1 padding lands
    (xfail until then).
  - cube ragged matmul (M/N/K)               -> xfail (Phase 2 — the K-contraction hazard).
  - vector ragged-REDUCED axis (row_sum N=66) -> xfail (deferred: needs an on-device proof that
    trowsum bounds the sum by valid_col before we pad a reduced axis).
"""

import os
import shutil

import pypto.language as pl
import pytest
from pypto import backend as _backend
from pypto.backend import BackendType
from pypto import ir

# ptoas is located via PTOAS_ROOT (see pto_backend._run_ptoas) or PATH.
_PTOAS = bool(os.environ.get("PTOAS_ROOT")) or shutil.which("ptoas") is not None
pytestmark = pytest.mark.skipif(not _PTOAS, reason="ptoas toolchain unavailable (set PTOAS_ROOT)")


def _assembles(fn, name, tmp_path) -> None:
    """Compile ``fn`` end-to-end WITH ptoas under the generic emitter; raise if it fails to assemble."""
    os.environ["PYPTO_AUTOFUSE_GENERIC_EMIT"] = "1"
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    try:
        prog = ir.Program([fn], name, ir.Span.unknown())
        ir.compile(prog, output_dir=str(tmp_path / name), skip_ptoas=False)
    finally:
        os.environ.pop("PYPTO_AUTOFUSE_GENERIC_EMIT", None)
        _backend.reset_for_testing()


# ---- baseline: aligned shapes assemble (both engines) ----

def test_vector_aligned_assembles(tmp_path):
    @pl.function(attrs={"auto_fuse": True})
    def pw(a: pl.Tensor[[512, 512], pl.FP32], b: pl.Tensor[[512, 512], pl.FP32]) -> pl.Tensor[[512, 512], pl.FP32]:
        c: pl.Tensor[[512, 512], pl.FP32] = pl.add(a, b)
        d: pl.Tensor[[512, 512], pl.FP32] = pl.mul(c, b)
        return d

    _assembles(pw, "pw_aligned", tmp_path)


def test_cube_aligned_assembles(tmp_path):
    @pl.function(attrs={"auto_fuse": True})
    def mm(a: pl.Tensor[[48, 64], pl.FP32], b: pl.Tensor[[64, 112], pl.FP32]) -> pl.Tensor[[48, 112], pl.FP32]:
        c: pl.Tensor[[48, 112], pl.FP32] = pl.matmul(a, b)
        return c

    _assembles(mm, "mm_aligned", tmp_path)


# ---- Phase 1 targets: vector FREE-axis-ragged — xfail until padding lands (step 3) ----

def test_vector_ragged_pointwise_assembles(tmp_path):
    @pl.function(attrs={"auto_fuse": True})
    def rpw(a: pl.Tensor[[130, 66], pl.FP32]) -> pl.Tensor[[130, 66], pl.FP32]:
        c: pl.Tensor[[130, 66], pl.FP32] = pl.add(a, 1.0)
        d: pl.Tensor[[130, 66], pl.FP32] = pl.mul(c, 2.0)
        return d

    _assembles(rpw, "rpw_ragged", tmp_path)


def test_vector_ragged_softmax_assembles(tmp_path):
    @pl.function(attrs={"auto_fuse": True})
    def sm(x: pl.Tensor[[256, 128], pl.FP32]) -> pl.Tensor[[256, 128], pl.FP32]:
        m: pl.Tensor[[256, 1], pl.FP32] = pl.row_max(x)
        s: pl.Tensor[[256, 128], pl.FP32] = pl.sub(x, m)
        e: pl.Tensor[[256, 128], pl.FP32] = pl.exp(s)
        d: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(e)
        o: pl.Tensor[[256, 128], pl.FP32] = pl.div(e, d)
        return o

    _assembles(sm, "sm_ragged", tmp_path)


# ---- ragged REDUCED axis — now padded (device-proven trowsum/tcolsum honor valid) ----

def test_vector_ragged_reduced_axis_softmax_assembles(tmp_path):
    """Softmax over a ragged REDUCED axis (N=66): the reduction axis (N) is itself ragged and
    gets padded 66->72, valid=66. Exercises BOTH row_max and row_sum over the padded reduced
    axis. A device experiment proved trowsum/tcolsum bound by valid, so the padded lanes are
    inert; the guard that previously declined this is lifted."""
    @pl.function(attrs={"auto_fuse": True})
    def sm(x: pl.Tensor[[256, 66], pl.FP32]) -> pl.Tensor[[256, 66], pl.FP32]:
        m: pl.Tensor[[256, 1], pl.FP32] = pl.row_max(x)
        s: pl.Tensor[[256, 66], pl.FP32] = pl.sub(x, m)
        e: pl.Tensor[[256, 66], pl.FP32] = pl.exp(s)
        d: pl.Tensor[[256, 1], pl.FP32] = pl.row_sum(e)
        o: pl.Tensor[[256, 66], pl.FP32] = pl.div(e, d)
        return o

    _assembles(sm, "sm_ragged_reduced", tmp_path)


# ---- deferred: cube ragged matmul (Phase 2 — tmatmul sums physical K, needs K-tail zero-fill) ----

@pytest.mark.xfail(strict=True, reason="Phase 2: ragged cube matmul needs M/N/K padding + K-tail "
                                       "zero-fill (tmatmul sums physical K). Documented limitation.")
def test_cube_ragged_matmul_assembles(tmp_path):
    @pl.function(attrs={"auto_fuse": True})
    def mm(a: pl.Tensor[[50, 66], pl.FP32], b: pl.Tensor[[66, 70], pl.FP32]) -> pl.Tensor[[50, 70], pl.FP32]:
        c: pl.Tensor[[50, 70], pl.FP32] = pl.matmul(a, b)
        return c

    _assembles(mm, "mm_ragged", tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
