# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device system tests for the AutoFuse ragged-tile padding (Phase 1).

Runs the *fully-lowered* kernels on real hardware (``--platform a2a3``) or the camodel
simulator (``--platform a2a3sim``) — the layer that ``torch_codegen`` (numeric, no hardware
model) and ptoas (assembly only) cannot cover.

TWO parts:

  Part A — the reduction-VALID PROBE (decisive).  Hand-written InCore kernels that reduce a
    tile whose ``valid`` extent is narrower than its physical (padded) extent, with a POISON
    value in the padded lanes.  Answers the one open question of the padding work: does
    ``pto.trowsum`` / ``pto.tcolsum`` bound the sum by ``valid`` (result excludes the poison)
    or by the physical extent (poison leaks in)?  This decides whether the emitter may pad a
    reduction's *reduced* axis (currently guarded/declined in EmitFusedGroupGeneric).  These
    kernels do NOT use AutoFuse — they isolate the hardware-op semantics.

  Part B — AutoFuse and whole-function AutoTile end-to-end on device. Realistic fused-vector
    kernels (ragged pointwise, softmax, RMSNorm, LayerNorm) compile with
    ``attrs={"auto_fuse": True}``. AutoTile controls use ``attrs={"auto_tile": True}`` with
    ``PYPTO_AUTOFUSE_GENERIC_EMIT`` unset and require the entire tensor DAG to lower as one group.
    Both are numerically verified against a torch reference on hardware.
    The wide P4 cases enable ``PYPTO_AUTOFUSE_P4=1`` in the test itself and cover online softmax,
    Welford layernorm, and a scaled-softmax near miss that must take the ordinary cut path.
    A wide P2 case has an apply-only bias input, giving the profiler a direct phase-traffic check.
    The return->named-output wiring is handled by the compiler (AutoFuse lifts the returned buffer
    into an appended Out param -> orchestration codegen emits the add_output write-back), so these
    return-based programs bind their output by position ([x, out]) in the harness.

RUN:
    # Part A needs no env flag; Part B needs the generic emitter enabled:
    PYPTO_AUTOFUSE_GENERIC_EMIT=1 python -m pytest tests/st/runtime/ops/test_auto_fuse_device.py \\
        --platform a2a3 -sv

NOTE TO THE DEVICE AGENT — verify the probe DSL forms (``pl.load`` / ``pl.set_validshape`` /
  ``pl.tile.row_sum(t, tmp)`` / ``pl.store``) mirror ``test_col_reduction.py`` +
  ``test_set_validshape.py``; adjust to your DSL version if a signature differs (e.g. row_sum's
  ``tmp`` tile, or set_validshape on a tile vs tensor). Part B needs no such adjustment — the
  auto_fuse programs are plain return-based functions and the output wiring is compiler-side.

THE ONE NUMBER THAT MATTERS: the Part-A ``row_sum`` device output.  ``66.0``/row => honors
valid (lift the reduced-axis guard).  ``~6e9``/row => sums physical (keep the guard / add a
K-style zero-fill).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto import backend as _backend
from pypto import passes
from pypto.backend import BackendType
from pypto.runtime.runner import RunConfig

# Silicon-appropriate tolerances. The device-free gate (torch_codegen) checks against EXACT fp32
# math and cannot see two hardware realities that the on-device golden (default rtol=atol=1e-5)
# does: (1) the Ascend HW reciprocal-sqrt (`pl.rsqrt`) is a ~12-bit approximation, ~1e-4 relative
# vs torch.rsqrt — so norms that call rsqrt need ~1e-3; (2) an end-to-end FP16 kernel is at the
# fp16 rounding floor (eps ~1e-3), so 1e-5 is ~100x tighter than the format can represent. These
# are op-precision facts, NOT emit/wiring errors (the FP32 softmax with the identical `exp` emit
# passes at 1e-5). Bit-exact rsqrt would be a separate `pl.rsqrt` Newton-refinement, out of scope.
# fp32 norms: the HW rsqrt (~1e-4 rel) compounds through sum-of-squares over the reduced axis, so
# an end-to-end norm lands at ~5-6e-3 on silicon (device run 2026-07-07: rmsnorm 4.8e-3, layernorm
# 5.8e-3) — 1e-3 was ~5x too tight. Set 1e-2 (still catches a real compute break; only masks the
# accumulated rsqrt rounding). FP32 softmax has no rsqrt and stays tight (default).
_RSQRT_TOL = RunConfig(rtol=1e-2, atol=1e-2)  # fp32 norms calling HW rsqrt (accumulated ~5e-3)
_FP16_TOL = RunConfig(rtol=1e-2, atol=1e-2)  # end-to-end fp16 (rounding floor)
_CUBE_TOL = RunConfig(rtol=1e-4, atol=1e-4)  # fp32 MAD reassociation across K windows
_BF16_CUBE_TOL = RunConfig(rtol=2e-2, atol=2e-2)  # chained BF16 drains + MAD reassociation

# Physical 8x72 tile: 72 FP32 cols = 288 bytes (32-aligned, assembles). Valid cols = 66 (264
# bytes, NOT 32-aligned) — exactly the ragged reduced axis the emitter would pad. Poison the
# padded cols [66, 72) so a physical-extent sum is unmistakable.
PHYS_R, PHYS_C, VALID_C = 8, 72, 66
POISON = 1.0e9


def _poison_cols() -> torch.Tensor:
    """1.0 in the valid cols, POISON in the padded cols — for the row_sum (trowsum) probe.

    The harness invokes a generic ``init_value`` callable with NO args
    (``TensorSpec.create_tensor`` -> ``fn()``); the shape is fixed by the module
    constants, so this takes no parameter.
    """
    t = torch.ones(PHYS_R, PHYS_C, dtype=torch.float32)
    t[:, VALID_C:] = POISON
    return t


def _poison_rows() -> torch.Tensor:
    """1.0 in the valid rows, POISON in the padded rows — for the col_sum (tcolsum) probe."""
    t = torch.ones(PHYS_C, PHYS_R, dtype=torch.float32)  # [72, 8]: reduce the 72 rows
    t[VALID_C:, :] = POISON
    return t


@pl.program
class RowSumValidProbe:
    """row_sum over a tile with valid_col=66 < physical cols=72; poison in [66,72)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[PHYS_R, PHYS_C], pl.FP32],
        out: pl.Out[pl.Tensor[[PHYS_R, 1], pl.FP32]],
    ) -> pl.Tensor[[PHYS_R, 1], pl.FP32]:
        tile: pl.Tile[[PHYS_R, PHYS_C], pl.FP32] = pl.load(x, [0, 0], [PHYS_R, PHYS_C])
        narrowed = pl.set_validshape(tile, PHYS_R, VALID_C)  # valid cols -> 66
        tmp: pl.Tile[[PHYS_R, PHYS_C], pl.FP32] = pl.tile.create(
            [PHYS_R, PHYS_C], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[PHYS_R, 1], pl.FP32] = pl.tile.row_sum(narrowed, tmp)
        return pl.store(result, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[PHYS_R, PHYS_C], pl.FP32],
        out: pl.Out[pl.Tensor[[PHYS_R, 1], pl.FP32]],
    ) -> pl.Tensor[[PHYS_R, 1], pl.FP32]:
        out = self.kernel(x, out)
        return out


@pl.program
class ColSumValidProbe:
    """col_sum over a tile with valid_row=66 < physical rows=72; poison in [66,72)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[PHYS_C, PHYS_R], pl.FP32],
        out: pl.Out[pl.Tensor[[1, PHYS_R], pl.FP32]],
    ) -> pl.Tensor[[1, PHYS_R], pl.FP32]:
        tile: pl.Tile[[PHYS_C, PHYS_R], pl.FP32] = pl.load(x, [0, 0], [PHYS_C, PHYS_R])
        narrowed = pl.set_validshape(tile, VALID_C, PHYS_R)  # valid rows -> 66
        tmp: pl.Tile[[PHYS_C, PHYS_R], pl.FP32] = pl.tile.create(
            [PHYS_C, PHYS_R], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[1, PHYS_R], pl.FP32] = pl.tile.col_sum(tile=narrowed, tmp_tile=tmp)
        return pl.store(result, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[PHYS_C, PHYS_R], pl.FP32],
        out: pl.Out[pl.Tensor[[1, PHYS_R], pl.FP32]],
    ) -> pl.Tensor[[1, PHYS_R], pl.FP32]:
        out = self.kernel(x, out)
        return out


@pl.program
class RowMaxValidProbe:
    """row_max over a tile with valid_col=66 < physical cols=72; poison (a LARGE value) in
    [66,72). Confirms MAX reductions honor valid too (the SUM proof does not cover max/min)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[PHYS_R, PHYS_C], pl.FP32],
        out: pl.Out[pl.Tensor[[PHYS_R, 1], pl.FP32]],
    ) -> pl.Tensor[[PHYS_R, 1], pl.FP32]:
        tile: pl.Tile[[PHYS_R, PHYS_C], pl.FP32] = pl.load(x, [0, 0], [PHYS_R, PHYS_C])
        narrowed = pl.set_validshape(tile, PHYS_R, VALID_C)  # valid cols -> 66
        tmp: pl.Tile[[PHYS_R, PHYS_C], pl.FP32] = pl.tile.create(
            [PHYS_R, PHYS_C], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        result: pl.Tile[[PHYS_R, 1], pl.FP32] = pl.tile.row_max(narrowed, tmp)
        return pl.store(result, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[PHYS_R, PHYS_C], pl.FP32],
        out: pl.Out[pl.Tensor[[PHYS_R, 1], pl.FP32]],
    ) -> pl.Tensor[[PHYS_R, 1], pl.FP32]:
        out = self.kernel(x, out)
        return out


class RowSumValidProbeCase(PTOTestCase):
    """PROBE: does trowsum bound the sum by valid_col? Expect 66.0/row (poison excluded)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_rowsum_valid_probe"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [PHYS_R, PHYS_C], DataType.FP32, init_value=_poison_cols),
            TensorSpec("out", [PHYS_R, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowSumValidProbe

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # Honors-valid expectation: sum of the 66 valid cols only = 66.0 per row. If the device
        # returns ~6e9 (= 66 + 6*POISON), the op sums the physical extent -> reduced-axis padding
        # is unsafe with garbage lanes (needs zero-fill).
        tensors["out"][:] = tensors["x"][:, :VALID_C].sum(dim=1, keepdim=True)


class ColSumValidProbeCase(PTOTestCase):
    """PROBE: does tcolsum bound the sum by valid_row? Expect 66.0/col (poison excluded)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_colsum_valid_probe"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [PHYS_C, PHYS_R], DataType.FP32, init_value=_poison_rows),
            TensorSpec("out", [1, PHYS_R], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSumValidProbe

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"][:VALID_C, :].sum(dim=0, keepdim=True)


class RowMaxValidProbeCase(PTOTestCase):
    """PROBE: does trowmax bound the max by valid_col? Expect 1.0/row (poison 1e9 excluded)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_rowmax_valid_probe"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [PHYS_R, PHYS_C], DataType.FP32, init_value=_poison_cols),  # valid=1.0, pad=1e9
            TensorSpec("out", [PHYS_R, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowMaxValidProbe

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # Honors-valid expectation: max over the 66 valid cols (all 1.0) = 1.0/row. If the op
        # maxes the physical extent, it would return the 1e9 poison and FAIL against 1.0.
        tensors["out"][:] = tensors["x"][:, :VALID_C].amax(dim=1, keepdim=True)


# ---- Part B: AutoFuse free-axis padding on device (needs PYPTO_AUTOFUSE_GENERIC_EMIT=1) ----

RPW_M, RPW_N = 130, 66  # ragged pointwise: N=66 free axis padded 66->72
SM_M, SM_N = 256, 128  # softmax: ragged M=256 (h tile padded); reduced N=128 aligned
RMS_M, RMS_N = 256, 512  # RMSNorm: aligned; one reduction (row_sum of squares) + broadcast
LN_M, LN_N = 256, 512  # LayerNorm: aligned; two reductions (mean + variance) + broadcast
NORM_EPS = 1.0e-6
WS_M, WS_N = 64, 4096  # wide-short pointwise: the free-axis over-pad overflow case
TL_M, TL_N = 4096, 64  # tall pointwise: many free-axis strips
SMR_M, SMR_N = 256, 66  # softmax ragged reduced N (padded reduced axis)
RRB_M, RRB_N = 256, 128  # row-reduce + broadcast (reduction intermediate, no div)
F16_M, F16_N = 256, 128  # FP16 softmax (granule g=16)
CS_M, CS_N = 128, 256  # bare col_sum sink -> S2 split-reduction (atomic-add merge)
FK_M, FK_N = 256, 256  # multi-sink fork: two live-outs sharing an input
P4_M, P4_N = 128, 8192  # reduced axis exceeds UB: exact P4 must stream online
P4_LN_SHIFT = 2000.0  # dual-sum variance cancels here; Welford must remain finite

# --- Part C: model-fragment experiments (realistic transformer components) ---
MRMS_M, MRMS_N = 256, 1024  # wider RMSNorm: exercises the sub-granule reduction-strip cap
MLN_M, MLN_N = 256, 1024  # wider LayerNorm (two reductions)
RES_M, RES_N = 256, 1024  # residual add + RMSNorm (a pre-norm block head)
SILU_M, SILU_N = 256, 1024  # SiLU/Swish activation: x*sigmoid(x) = x/(1+exp(-x))
SWG_M, SWG_N = 256, 1024  # SwiGLU FFN gating: silu(gate)*up (two inputs)
MIXED_SWG_M, MIXED_SWG_K = 32, 64
MIXED_SWG_F, MIXED_SWG_N = 128, 64  # gate/up GEMMs -> vector SwiGLU -> down GEMM
SSM_M, SSM_N = 256, 512  # scaled softmax (attention scores * 1/sqrt(d))
TWIN_M, TWIN_N = 256, 512  # two interleaved independent chains (group-reorder fix)
ATT_S, ATT_D = 128, 64  # attention block: q@k -> scaled softmax -> p@v


def _p4_shifted_layernorm_input() -> torch.Tensor:
    """High-mean input that exposes cancellation in a raw dual-sum variance."""
    return torch.randn(P4_M, P4_N, dtype=torch.float32) + P4_LN_SHIFT


def _planned_autotile_entry(program: Any, entry_name: str) -> Any:
    """Return one entry after the standalone whole-function AutoTile transform."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    try:
        planned = passes.auto_fuse()(program)
    finally:
        _backend.reset_for_testing()
    return next(func for _, func in planned.functions.items() if func.name == entry_name)


def _assert_one_autotile_group(case: PTOTestCase, entry_name: str) -> None:
    """Assert that the system-test program uses one exact AutoTile SPMD group."""
    func = _planned_autotile_entry(case.get_program(), entry_name)
    body = func.as_python()
    assert body.count("pl.spmd(") == 1, body
    assert "auto_tile" not in func.attrs


class AutoFuseRaggedPointwiseCase(PTOTestCase):
    """AutoFuse ragged pointwise [130,66]: c=a+1; d=c*2. Free-axis N padding, on device."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_ragged_pointwise_130x66"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [RPW_M, RPW_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [RPW_M, RPW_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Defined lazily so a first-of-its-kind auto_fuse<->harness mismatch fails at run
        # (device), not at import/collection. If the harness needs an explicit output write,
        # switch to the Out-param + `out = pl.assemble(out, d, [0,0]); return out` form.
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def rpw(self, a: pl.Tensor[[RPW_M, RPW_N], pl.FP32]) -> pl.Tensor[[RPW_M, RPW_N], pl.FP32]:
                c: pl.Tensor[[RPW_M, RPW_N], pl.FP32] = pl.add(a, 1.0)
                d: pl.Tensor[[RPW_M, RPW_N], pl.FP32] = pl.mul(c, 2.0)
                return d

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = (tensors["a"] + 1.0) * 2.0


class AutoFuseSoftmaxCase(PTOTestCase):
    """AutoFuse softmax [256,128]: free-axis M padding; reduced N=128 aligned. On device."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_softmax_256x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [SM_M, SM_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [SM_M, SM_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[SM_M, SM_N], pl.FP32]) -> pl.Tensor[[SM_M, SM_N], pl.FP32]:
                m: pl.Tensor[[SM_M, 1], pl.FP32] = pl.row_max(x)
                s: pl.Tensor[[SM_M, SM_N], pl.FP32] = pl.sub(x, m)
                e: pl.Tensor[[SM_M, SM_N], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[SM_M, 1], pl.FP32] = pl.row_sum(e)
                o: pl.Tensor[[SM_M, SM_N], pl.FP32] = pl.div(e, d)
                return o

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=1)


class AutoTileSoftmaxCase(AutoFuseSoftmaxCase):
    """Whole-function AutoTile softmax; all five tensor ops must remain one group."""

    def get_name(self) -> str:
        return "autotile_whole_softmax_256x128"

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": False, "auto_tile": True})
            def sm(self, x: pl.Tensor[[SM_M, SM_N], pl.FP32]) -> pl.Tensor[[SM_M, SM_N], pl.FP32]:
                m: pl.Tensor[[SM_M, 1], pl.FP32] = pl.row_max(x)
                s: pl.Tensor[[SM_M, SM_N], pl.FP32] = pl.sub(x, m)
                e: pl.Tensor[[SM_M, SM_N], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[SM_M, 1], pl.FP32] = pl.row_sum(e)
                o: pl.Tensor[[SM_M, SM_N], pl.FP32] = pl.div(e, d)
                return o

        return Prog


class AutoFuseRmsNormCase(PTOTestCase):
    """AutoFuse RMSNorm [256,512]: sq=x*x; ms=mean(sq); out=x*rsqrt(ms+eps). One
    reduction + a [M,1]-over-[M,N] broadcast — a canonical transformer norm."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_rmsnorm_256x512"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [RMS_M, RMS_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [RMS_M, RMS_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def rmsnorm(self, x: pl.Tensor[[RMS_M, RMS_N], pl.FP32]) -> pl.Tensor[[RMS_M, RMS_N], pl.FP32]:
                sq: pl.Tensor[[RMS_M, RMS_N], pl.FP32] = pl.mul(x, x)
                ss: pl.Tensor[[RMS_M, 1], pl.FP32] = pl.row_sum(sq)
                ms: pl.Tensor[[RMS_M, 1], pl.FP32] = pl.mul(ss, 1.0 / RMS_N)
                var: pl.Tensor[[RMS_M, 1], pl.FP32] = pl.add(ms, NORM_EPS)
                rms: pl.Tensor[[RMS_M, 1], pl.FP32] = pl.rsqrt(var)
                out: pl.Tensor[[RMS_M, RMS_N], pl.FP32] = pl.mul(x, rms)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS)


class AutoFuseLayerNormCase(PTOTestCase):
    """AutoFuse LayerNorm [256,512]: mu=mean(x); xc=x-mu; var=mean(xc^2); out=xc*rsqrt(var+eps).
    Two reductions (mean + variance) + broadcast — the richest fused-vector norm."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_layernorm_256x512"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [LN_M, LN_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [LN_M, LN_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def layernorm(self, x: pl.Tensor[[LN_M, LN_N], pl.FP32]) -> pl.Tensor[[LN_M, LN_N], pl.FP32]:
                sx: pl.Tensor[[LN_M, 1], pl.FP32] = pl.row_sum(x)
                mu: pl.Tensor[[LN_M, 1], pl.FP32] = pl.mul(sx, 1.0 / LN_N)
                xc: pl.Tensor[[LN_M, LN_N], pl.FP32] = pl.sub(x, mu)
                sq: pl.Tensor[[LN_M, LN_N], pl.FP32] = pl.mul(xc, xc)
                sv: pl.Tensor[[LN_M, 1], pl.FP32] = pl.row_sum(sq)
                var: pl.Tensor[[LN_M, 1], pl.FP32] = pl.mul(sv, 1.0 / LN_N)
                vare: pl.Tensor[[LN_M, 1], pl.FP32] = pl.add(var, NORM_EPS)
                inv: pl.Tensor[[LN_M, 1], pl.FP32] = pl.rsqrt(vare)
                out: pl.Tensor[[LN_M, LN_N], pl.FP32] = pl.mul(xc, inv)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        tensors["out"][:] = xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + NORM_EPS)


class AutoFuseP4SoftmaxWideCase(PTOTestCase):
    """Exact online softmax [128,8192], whose reduced axis cannot materialize in UB."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_p4_softmax_128x8192"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [P4_M, P4_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [P4_M, P4_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[P4_M, P4_N], pl.FP32]) -> pl.Tensor[[P4_M, P4_N], pl.FP32]:
                m: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_max(x)
                shifted: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_sub(x, m)
                exp: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.exp(shifted)
                total: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_sum(exp)
                out: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_div(exp, total)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=-1)


class AutoFuseP4LayerNormWideCase(PTOTestCase):
    """Canonical dual-sum layernorm [128,8192], emitted as stable online Welford."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_p4_layernorm_128x8192_shift2000"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [P4_M, P4_N], DataType.FP32, init_value=_p4_shifted_layernorm_input),
            TensorSpec("out", [P4_M, P4_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def ln(self, x: pl.Tensor[[P4_M, P4_N], pl.FP32]) -> pl.Tensor[[P4_M, P4_N], pl.FP32]:
                sx: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_sum(x)
                xsq: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.mul(x, x)
                sxsq: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_sum(xsq)
                mean: pl.Tensor[[P4_M, 1], pl.FP32] = pl.mul(sx, 1.0 / P4_N)
                mean_square: pl.Tensor[[P4_M, 1], pl.FP32] = pl.mul(sxsq, 1.0 / P4_N)
                square_mean: pl.Tensor[[P4_M, 1], pl.FP32] = pl.mul(mean, mean)
                variance: pl.Tensor[[P4_M, 1], pl.FP32] = pl.sub(mean_square, square_mean)
                variance_eps: pl.Tensor[[P4_M, 1], pl.FP32] = pl.add(variance, NORM_EPS)
                inv_std: pl.Tensor[[P4_M, 1], pl.FP32] = pl.rsqrt(variance_eps)
                centered: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_sub(x, mean)
                out: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_mul(centered, inv_std)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        tensors["out"][:] = (x - mean) * torch.rsqrt(variance + NORM_EPS)


class AutoFuseP4ScaledSoftmaxWideCase(PTOTestCase):
    """Scaled softmax [128,8192]: a deliberate P4 near miss that must cut safely."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_p4_scaled_softmax_cut_128x8192"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [P4_M, P4_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [P4_M, P4_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[P4_M, P4_N], pl.FP32]) -> pl.Tensor[[P4_M, P4_N], pl.FP32]:
                scaled: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.mul(x, 0.125)
                m: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_max(scaled)
                shifted: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_sub(scaled, m)
                exp: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.exp(shifted)
                total: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_sum(exp)
                out: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_div(exp, total)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"] * 0.125, dim=-1)


class AutoFuseP2ApplyInputWideCase(PTOTestCase):
    """Wide P2 with x read in both phases and bias read only by the apply phase."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_p2_apply_input_128x8192"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [P4_M, P4_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("bias", [P4_M, P4_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [P4_M, P4_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def submax_bias(
                self,
                x: pl.Tensor[[P4_M, P4_N], pl.FP32],
                bias: pl.Tensor[[P4_M, P4_N], pl.FP32],
            ) -> pl.Tensor[[P4_M, P4_N], pl.FP32]:
                maximum: pl.Tensor[[P4_M, 1], pl.FP32] = pl.row_max(x)
                centered: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.row_expand_sub(x, maximum)
                out: pl.Tensor[[P4_M, P4_N], pl.FP32] = pl.add(centered, bias)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = x - x.amax(dim=-1, keepdim=True) + tensors["bias"]


class AutoFusePwWideShortCase(PTOTestCase):
    """Wide-short pointwise [64,4096]: the free-axis over-pad case. Rows are the FREE
    (row-major) axis → must NOT be granule-padded; before the fix this overflowed UB."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pw_wide_short_64x4096"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [WS_M, WS_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [WS_M, WS_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [WS_M, WS_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(
                self, a: pl.Tensor[[WS_M, WS_N], pl.FP32], b: pl.Tensor[[WS_M, WS_N], pl.FP32]
            ) -> pl.Tensor[[WS_M, WS_N], pl.FP32]:
                c: pl.Tensor[[WS_M, WS_N], pl.FP32] = pl.add(a, b)
                d: pl.Tensor[[WS_M, WS_N], pl.FP32] = pl.mul(c, b)
                return d

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = (tensors["a"] + tensors["b"]) * tensors["b"]


class AutoFusePwTallCase(PTOTestCase):
    """Tall pointwise [4096,64]: many free-axis strips."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pw_tall_4096x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TL_M, TL_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TL_M, TL_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [TL_M, TL_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def pw(
                self, a: pl.Tensor[[TL_M, TL_N], pl.FP32], b: pl.Tensor[[TL_M, TL_N], pl.FP32]
            ) -> pl.Tensor[[TL_M, TL_N], pl.FP32]:
                c: pl.Tensor[[TL_M, TL_N], pl.FP32] = pl.add(a, b)
                d: pl.Tensor[[TL_M, TL_N], pl.FP32] = pl.mul(c, b)
                return d

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = (tensors["a"] + tensors["b"]) * tensors["b"]


class AutoFuseSoftmaxRaggedNCase(PTOTestCase):
    """Softmax [256,66]: ragged REDUCED axis N=66 (padded). Reduction honors valid."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_softmax_ragged_256x66"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [SMR_M, SMR_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [SMR_M, SMR_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[SMR_M, SMR_N], pl.FP32]) -> pl.Tensor[[SMR_M, SMR_N], pl.FP32]:
                m: pl.Tensor[[SMR_M, 1], pl.FP32] = pl.row_max(x)
                s: pl.Tensor[[SMR_M, SMR_N], pl.FP32] = pl.sub(x, m)
                e: pl.Tensor[[SMR_M, SMR_N], pl.FP32] = pl.exp(s)
                d: pl.Tensor[[SMR_M, 1], pl.FP32] = pl.row_sum(e)
                o: pl.Tensor[[SMR_M, SMR_N], pl.FP32] = pl.div(e, d)
                return o

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=1)


class AutoFuseRowReduceBroadcastCase(PTOTestCase):
    """row_max + broadcast subtract [256,128]: y = x - row_max(x). Reduction intermediate
    broadcast back to [M,N], no division — isolates the reduction+broadcast path."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_row_reduce_broadcast_256x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [RRB_M, RRB_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [RRB_M, RRB_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def f(self, x: pl.Tensor[[RRB_M, RRB_N], pl.FP32]) -> pl.Tensor[[RRB_M, RRB_N], pl.FP32]:
                m: pl.Tensor[[RRB_M, 1], pl.FP32] = pl.row_max(x)
                y: pl.Tensor[[RRB_M, RRB_N], pl.FP32] = pl.sub(x, m)
                return y

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = x - x.amax(dim=1, keepdim=True)


class AutoFuseFp16SoftmaxCase(PTOTestCase):
    """FP16 softmax [256,128]: exercises the FP16 granule (g=16 elements = 32 bytes)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_fp16_softmax_256x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [F16_M, F16_N], DataType.FP16, init_value=torch.randn),
            TensorSpec("out", [F16_M, F16_N], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, x: pl.Tensor[[F16_M, F16_N], pl.FP16]) -> pl.Tensor[[F16_M, F16_N], pl.FP16]:
                m: pl.Tensor[[F16_M, 1], pl.FP16] = pl.row_max(x)
                s: pl.Tensor[[F16_M, F16_N], pl.FP16] = pl.sub(x, m)
                e: pl.Tensor[[F16_M, F16_N], pl.FP16] = pl.exp(s)
                d: pl.Tensor[[F16_M, 1], pl.FP16] = pl.row_sum(e)
                o: pl.Tensor[[F16_M, F16_N], pl.FP16] = pl.div(e, d)
                return o

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = torch.softmax(x.float(), dim=1).to(x.dtype)


class AutoFuseColSumCase(PTOTestCase):
    """Bare col_sum [128,256]->[1,256]: the reduced-sink S2 split-reduction path. The solver
    splits the reduced M axis across cores; each computes a partial col_sum, the partials
    atomic-add into a zero-seeded output. Validates the atomic-add merge on hardware."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_col_sum_128x256"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [CS_M, CS_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, CS_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def f(self, x: pl.Tensor[[CS_M, CS_N], pl.FP32]) -> pl.Tensor[[1, CS_N], pl.FP32]:
                y: pl.Tensor[[1, CS_N], pl.FP32] = pl.col_sum(x)
                return y

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].sum(dim=0, keepdim=True)


class AutoFuseSingletonReductionCase(PTOTestCase):
    """Full-frame singleton axes need padding; collapsed reduction axes stay logically thin."""

    __test__ = False

    def __init__(self, orientation: str, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        assert orientation in {"row", "column"}
        self.orientation = orientation

    def get_name(self) -> str:
        return f"autofuse_singleton_{self.orientation}_reduction"

    def define_tensors(self) -> list[TensorSpec]:
        shape = [1, 8192] if self.orientation == "row" else [8192, 1]
        return [
            TensorSpec("x", shape, DataType.FP32, init_value=torch.rand),
            TensorSpec("out", [1, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        if self.orientation == "row":

            @pl.program
            class Row:
                @pl.function(attrs={"auto_fuse": True})
                def reduce(self, x: pl.Tensor[[1, 8192], pl.FP32]) -> pl.Tensor[[1, 1], pl.FP32]:
                    values: pl.Tensor[[1, 8192], pl.FP32] = pl.exp(x)
                    out: pl.Tensor[[1, 1], pl.FP32] = pl.row_sum(values)
                    return out

            return Row

        @pl.program
        class Column:
            @pl.function(attrs={"auto_fuse": True})
            def reduce(self, x: pl.Tensor[[8192, 1], pl.FP32]) -> pl.Tensor[[1, 1], pl.FP32]:
                values: pl.Tensor[[8192, 1], pl.FP32] = pl.exp(x)
                out: pl.Tensor[[1, 1], pl.FP32] = pl.col_sum(values)
                return out

        return Column

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        dim = 1 if self.orientation == "row" else 0
        tensors["out"][:] = torch.exp(tensors["x"]).sum(dim=dim, keepdim=True)


class AutoFuseForkCase(PTOTestCase):
    """Multi-sink fork [256,256] -> (a, b): a=(x+1)*2, b=(x+1)*3 share the intermediate c=x+1.
    Two live-outs in one fused group, each assembled to its own output; validates the
    multi-sink emit + the multi-RETURN -> multiple-Out-param wiring on hardware. Outputs are
    positionally [x, a, b] to match the appended Out params [x, a_out, b_out]."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_fork_256x256"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [FK_M, FK_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("outa", [FK_M, FK_N], DataType.FP32, is_output=True),
            TensorSpec("outb", [FK_M, FK_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def f(
                self, x: pl.Tensor[[FK_M, FK_N], pl.FP32]
            ) -> tuple[pl.Tensor[[FK_M, FK_N], pl.FP32], pl.Tensor[[FK_M, FK_N], pl.FP32]]:
                c: pl.Tensor[[FK_M, FK_N], pl.FP32] = pl.add(x, 1.0)
                a: pl.Tensor[[FK_M, FK_N], pl.FP32] = pl.mul(c, 2.0)
                b: pl.Tensor[[FK_M, FK_N], pl.FP32] = pl.mul(c, 3.0)
                return a, b

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["outa"][:] = (x + 1.0) * 2.0
        tensors["outb"][:] = (x + 1.0) * 3.0


# ===========================================================================
# Part C — model-fragment experiments (realistic transformer components)
# ===========================================================================


class ModelRmsNormWideCase(PTOTestCase):
    """RMSNorm at hidden=1024 — wider than the [256,512] Part-B case, so the free-axis
    reduction strips are sub-granule and the emit falls to the serial (granule-multiple)
    path. Validates the sub-granule-strip cap end-to-end on hardware."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_rmsnorm_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [MRMS_M, MRMS_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [MRMS_M, MRMS_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def rms(self, x: pl.Tensor[[MRMS_M, MRMS_N], pl.FP32]) -> pl.Tensor[[MRMS_M, MRMS_N], pl.FP32]:
                sq: pl.Tensor[[MRMS_M, MRMS_N], pl.FP32] = pl.mul(x, x)
                ss: pl.Tensor[[MRMS_M, 1], pl.FP32] = pl.row_sum(sq)
                ms: pl.Tensor[[MRMS_M, 1], pl.FP32] = pl.mul(ss, 1.0 / MRMS_N)
                var: pl.Tensor[[MRMS_M, 1], pl.FP32] = pl.add(ms, NORM_EPS)
                inv: pl.Tensor[[MRMS_M, 1], pl.FP32] = pl.rsqrt(var)
                out: pl.Tensor[[MRMS_M, MRMS_N], pl.FP32] = pl.mul(x, inv)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS)


class ModelLayerNormWideCase(PTOTestCase):
    """LayerNorm at hidden=1024 — two reductions (mean + variance) + broadcast, wide."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_layernorm_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [MLN_M, MLN_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [MLN_M, MLN_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def ln(self, x: pl.Tensor[[MLN_M, MLN_N], pl.FP32]) -> pl.Tensor[[MLN_M, MLN_N], pl.FP32]:
                sx: pl.Tensor[[MLN_M, 1], pl.FP32] = pl.row_sum(x)
                mu: pl.Tensor[[MLN_M, 1], pl.FP32] = pl.mul(sx, 1.0 / MLN_N)
                xc: pl.Tensor[[MLN_M, MLN_N], pl.FP32] = pl.sub(x, mu)
                sq: pl.Tensor[[MLN_M, MLN_N], pl.FP32] = pl.mul(xc, xc)
                sv: pl.Tensor[[MLN_M, 1], pl.FP32] = pl.row_sum(sq)
                var: pl.Tensor[[MLN_M, 1], pl.FP32] = pl.mul(sv, 1.0 / MLN_N)
                ve: pl.Tensor[[MLN_M, 1], pl.FP32] = pl.add(var, NORM_EPS)
                inv: pl.Tensor[[MLN_M, 1], pl.FP32] = pl.rsqrt(ve)
                out: pl.Tensor[[MLN_M, MLN_N], pl.FP32] = pl.mul(xc, inv)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        xc = x - x.mean(-1, keepdim=True)
        tensors["out"][:] = xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + NORM_EPS)


class ModelResidualRmsNormCase(PTOTestCase):
    """Pre-norm block head: h = x + residual; out = RMSNorm(h). A reduction over an
    elementwise-produced intermediate — the residual add and the norm fuse into one kernel."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_residual_rmsnorm_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [RES_M, RES_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("res", [RES_M, RES_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [RES_M, RES_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def resrms(
                self, x: pl.Tensor[[RES_M, RES_N], pl.FP32], res: pl.Tensor[[RES_M, RES_N], pl.FP32]
            ) -> pl.Tensor[[RES_M, RES_N], pl.FP32]:
                h: pl.Tensor[[RES_M, RES_N], pl.FP32] = pl.add(x, res)
                sq: pl.Tensor[[RES_M, RES_N], pl.FP32] = pl.mul(h, h)
                ss: pl.Tensor[[RES_M, 1], pl.FP32] = pl.row_sum(sq)
                ms: pl.Tensor[[RES_M, 1], pl.FP32] = pl.mul(ss, 1.0 / RES_N)
                var: pl.Tensor[[RES_M, 1], pl.FP32] = pl.add(ms, NORM_EPS)
                inv: pl.Tensor[[RES_M, 1], pl.FP32] = pl.rsqrt(var)
                out: pl.Tensor[[RES_M, RES_N], pl.FP32] = pl.mul(h, inv)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        h = tensors["x"] + tensors["res"]
        tensors["out"][:] = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + NORM_EPS)


class ModelSiluCase(PTOTestCase):
    """SiLU/Swish activation out = x*sigmoid(x), composed as x/(1+exp(-x)) — a pure
    pointwise chain (neg, exp, add, div) at FFN width."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_silu_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [SILU_M, SILU_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [SILU_M, SILU_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def silu(self, x: pl.Tensor[[SILU_M, SILU_N], pl.FP32]) -> pl.Tensor[[SILU_M, SILU_N], pl.FP32]:
                nx: pl.Tensor[[SILU_M, SILU_N], pl.FP32] = pl.neg(x)
                e: pl.Tensor[[SILU_M, SILU_N], pl.FP32] = pl.exp(nx)
                d: pl.Tensor[[SILU_M, SILU_N], pl.FP32] = pl.add(e, 1.0)
                out: pl.Tensor[[SILU_M, SILU_N], pl.FP32] = pl.div(x, d)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = x * torch.sigmoid(x)


class ModelSwiGluCase(PTOTestCase):
    """SwiGLU FFN gating out = silu(gate)*up — the LLaMA/PaLM FFN nonlinearity, two inputs."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_swiglu_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("gate", [SWG_M, SWG_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("up", [SWG_M, SWG_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [SWG_M, SWG_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def swiglu(
                self, gate: pl.Tensor[[SWG_M, SWG_N], pl.FP32], up: pl.Tensor[[SWG_M, SWG_N], pl.FP32]
            ) -> pl.Tensor[[SWG_M, SWG_N], pl.FP32]:
                ng: pl.Tensor[[SWG_M, SWG_N], pl.FP32] = pl.neg(gate)
                e: pl.Tensor[[SWG_M, SWG_N], pl.FP32] = pl.exp(ng)
                d: pl.Tensor[[SWG_M, SWG_N], pl.FP32] = pl.add(e, 1.0)
                s: pl.Tensor[[SWG_M, SWG_N], pl.FP32] = pl.div(gate, d)
                out: pl.Tensor[[SWG_M, SWG_N], pl.FP32] = pl.mul(s, up)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        gate, up = tensors["gate"], tensors["up"]
        tensors["out"][:] = (gate * torch.sigmoid(gate)) * up


class AutoFuseMixedDenseSwiGluCase(PTOTestCase):
    """Dense FFN round trip: two AIC projections -> AIV SwiGLU -> AIC down projection."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_mixed_dense_swiglu_32x64x128x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "x",
                [MIXED_SWG_M, MIXED_SWG_K],
                DataType.BF16,
                init_value=lambda: (torch.randn(MIXED_SWG_M, MIXED_SWG_K) * 0.1).to(torch.bfloat16),
            ),
            TensorSpec(
                "w_gate",
                [MIXED_SWG_K, MIXED_SWG_F],
                DataType.BF16,
                init_value=lambda: (torch.randn(MIXED_SWG_K, MIXED_SWG_F) * 0.1).to(torch.bfloat16),
            ),
            TensorSpec(
                "w_up",
                [MIXED_SWG_K, MIXED_SWG_F],
                DataType.BF16,
                init_value=lambda: (torch.randn(MIXED_SWG_K, MIXED_SWG_F) * 0.1).to(torch.bfloat16),
            ),
            TensorSpec(
                "w_down",
                [MIXED_SWG_F, MIXED_SWG_N],
                DataType.BF16,
                init_value=lambda: (torch.randn(MIXED_SWG_F, MIXED_SWG_N) * 0.1).to(torch.bfloat16),
            ),
            TensorSpec("out", [MIXED_SWG_M, MIXED_SWG_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mlp(
                self,
                x: pl.Tensor[[MIXED_SWG_M, MIXED_SWG_K], pl.BF16],
                w_gate: pl.Tensor[[MIXED_SWG_K, MIXED_SWG_F], pl.BF16],
                w_up: pl.Tensor[[MIXED_SWG_K, MIXED_SWG_F], pl.BF16],
                w_down: pl.Tensor[[MIXED_SWG_F, MIXED_SWG_N], pl.BF16],
            ) -> pl.Tensor[[MIXED_SWG_M, MIXED_SWG_N], pl.FP32]:
                gate = pl.matmul(x, w_gate, out_dtype=pl.FP32)
                up = pl.matmul(x, w_up, out_dtype=pl.FP32)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                activation = pl.cast(
                    pl.mul(pl.mul(gate, sigmoid), up),
                    target_type=pl.BF16,
                )
                return pl.matmul(activation, w_down, out_dtype=pl.FP32)

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # tensor.matmul(out_dtype=FP32) widens the BF16 matmul result, matching
        # torch_codegen and the emitted cube operation.
        gate = (tensors["x"] @ tensors["w_gate"]).float()
        up = (tensors["x"] @ tensors["w_up"]).float()
        activation = ((gate * torch.sigmoid(gate)) * up).to(torch.bfloat16)
        tensors["out"][:] = (activation @ tensors["w_down"]).float()


class ModelScaledSoftmaxCase(PTOTestCase):
    """Attention-score softmax: out = softmax(scores / sqrt(d)) — a scale then the numerically
    stable softmax (row_max, sub, exp, row_sum, div). The pre-scale is the attention 1/sqrt(d)."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_scaled_softmax_256x512"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("s", [SSM_M, SSM_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [SSM_M, SSM_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sm(self, s: pl.Tensor[[SSM_M, SSM_N], pl.FP32]) -> pl.Tensor[[SSM_M, SSM_N], pl.FP32]:
                sc: pl.Tensor[[SSM_M, SSM_N], pl.FP32] = pl.mul(s, 0.125)
                m: pl.Tensor[[SSM_M, 1], pl.FP32] = pl.row_max(sc)
                d: pl.Tensor[[SSM_M, SSM_N], pl.FP32] = pl.row_expand_sub(sc, m)
                e: pl.Tensor[[SSM_M, SSM_N], pl.FP32] = pl.exp(d)
                sm: pl.Tensor[[SSM_M, 1], pl.FP32] = pl.row_sum(e)
                out: pl.Tensor[[SSM_M, SSM_N], pl.FP32] = pl.row_expand_div(e, sm)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["s"] * 0.125, dim=-1)


class ModelInterleavedTwinCase(PTOTestCase):
    """Two INDEPENDENT elementwise chains (exp->neg on x, exp->mul on y) interleaved in source
    order. The solver puts each in its own group; the group-reorder fix emits each as ONE fused
    scope (was fragmented into single-op scopes spilling the intermediate to DDR). Multi-return."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_interleaved_twin_256x512"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [TWIN_M, TWIN_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [TWIN_M, TWIN_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("outa", [TWIN_M, TWIN_N], DataType.FP32, is_output=True),
            TensorSpec("outb", [TWIN_M, TWIN_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def twin(
                self, x: pl.Tensor[[TWIN_M, TWIN_N], pl.FP32], y: pl.Tensor[[TWIN_M, TWIN_N], pl.FP32]
            ) -> tuple[pl.Tensor[[TWIN_M, TWIN_N], pl.FP32], pl.Tensor[[TWIN_M, TWIN_N], pl.FP32]]:
                a: pl.Tensor[[TWIN_M, TWIN_N], pl.FP32] = pl.exp(x)
                b: pl.Tensor[[TWIN_M, TWIN_N], pl.FP32] = pl.exp(y)
                a2: pl.Tensor[[TWIN_M, TWIN_N], pl.FP32] = pl.neg(a)
                b2: pl.Tensor[[TWIN_M, TWIN_N], pl.FP32] = pl.mul(b, b)
                return a2, b2

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["outa"][:] = -torch.exp(tensors["x"])
        tensors["outb"][:] = torch.exp(tensors["y"]) ** 2


class ModelAttentionCase(PTOTestCase):
    """A full single-head attention block: p = softmax((q@k) / sqrt(d)); out = p@v. Two matmuls
    (cube) with a scaled softmax (vector) between — the matmul + vector engines composed."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "model_attention_128x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("q", [ATT_S, ATT_D], DataType.FP32, init_value=torch.randn),
            TensorSpec("k", [ATT_D, ATT_S], DataType.FP32, init_value=torch.randn),
            TensorSpec("v", [ATT_S, ATT_D], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [ATT_S, ATT_D], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def attn(
                self,
                q: pl.Tensor[[ATT_S, ATT_D], pl.FP32],
                k: pl.Tensor[[ATT_D, ATT_S], pl.FP32],
                v: pl.Tensor[[ATT_S, ATT_D], pl.FP32],
            ) -> pl.Tensor[[ATT_S, ATT_D], pl.FP32]:
                s: pl.Tensor[[ATT_S, ATT_S], pl.FP32] = pl.matmul(q, k)
                sc: pl.Tensor[[ATT_S, ATT_S], pl.FP32] = pl.mul(s, 0.125)
                m: pl.Tensor[[ATT_S, 1], pl.FP32] = pl.row_max(sc)
                dd: pl.Tensor[[ATT_S, ATT_S], pl.FP32] = pl.row_expand_sub(sc, m)
                e: pl.Tensor[[ATT_S, ATT_S], pl.FP32] = pl.exp(dd)
                sm: pl.Tensor[[ATT_S, 1], pl.FP32] = pl.row_sum(e)
                p: pl.Tensor[[ATT_S, ATT_S], pl.FP32] = pl.row_expand_div(e, sm)
                out: pl.Tensor[[ATT_S, ATT_D], pl.FP32] = pl.matmul(p, v)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        q, k, v = tensors["q"], tensors["k"], tensors["v"]
        tensors["out"][:] = torch.softmax((q @ k) * 0.125, dim=-1) @ v


# ===========================================================================
# Part D — shape x dtype SWEEP (wide model-verification coverage)
# ===========================================================================
#
# One kernel family (pointwise / softmax / RMSNorm-style) swept across the shape space and both
# dtypes, to stress the emit where fixed cases can't: ragged widths (66/130 -> padding +
# count-mode floor), wide tiles (1024 -> sub-granule strip cap), tall tiles (4096 rows -> many
# strips), the wide-short over-pad case (64x4096), and the fp32/fp16 granule (64 vs 128 elems/
# repeat). The generic sweep stays <= 1024 for reductions; dedicated P4 cases above cover the
# online-streaming regime at N=8192 with exact softmax/layernorm and a deliberate near miss.
# FP16 covers only the scalar-free reduction kernels (a fp16 tensor + a fp32 scalar const promotes
# to fp32 in the DSL, so fp16 pointwise-with-scalar is skipped — an authoring limitation, not emit).
SWEEP_GRID = [
    ("pw", 64, 64, "fp32"),
    ("pw", 130, 66, "fp32"),
    ("pw", 256, 512, "fp32"),
    ("pw", 512, 1024, "fp32"),
    ("pw", 4096, 64, "fp32"),
    ("pw", 64, 4096, "fp32"),
    ("softmax", 256, 128, "fp32"),
    ("softmax", 256, 66, "fp32"),
    ("softmax", 128, 512, "fp32"),
    ("softmax", 512, 1024, "fp32"),
    ("softmax", 64, 256, "fp32"),
    ("rms", 256, 512, "fp32"),
    ("rms", 256, 1024, "fp32"),
    ("rms", 128, 256, "fp32"),
    ("rms", 130, 128, "fp32"),
    ("softmax", 256, 128, "fp16"),
    ("softmax", 256, 512, "fp16"),
    ("softmax", 128, 256, "fp16"),
    ("rms", 256, 512, "fp16"),
    ("rms", 256, 1024, "fp16"),
    ("rms", 128, 128, "fp16"),
]


class AutoFuseSweepCase(PTOTestCase):
    """One sweep point: kernel `kernel` at shape (M,N), dtype `dt`. The RMSNorm-style kernel here
    is the bare `x * rsqrt(sum(x^2))` (reduction + broadcast + rsqrt, no mean/eps) — enough to
    stress the reduction path; the real eps/mean RMSNorm is the fixed Part-C case."""

    __test__ = False

    def __init__(self, kernel: str, M: int, N: int, dt: str, *, platform: str | None = None, config=None):
        self.kernel, self.M, self.N, self.dt = kernel, M, N, dt
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"autofuse_sweep_{self.kernel}_{self.M}x{self.N}_{self.dt}"

    def _dtype(self) -> Any:
        return DataType.FP16 if self.dt == "fp16" else DataType.FP32

    def define_tensors(self) -> list[TensorSpec]:
        d = self._dtype()
        return [
            TensorSpec("x", [self.M, self.N], d, init_value=torch.randn),
            TensorSpec("out", [self.M, self.N], d, is_output=True),
        ]

    def get_program(self) -> Any:
        M, N = self.M, self.N
        DT = pl.FP16 if self.dt == "fp16" else pl.FP32
        # DISTINCT CLASS names per kernel. `@pl.program` resolves each class by its source via
        # inspect.getsource keyed on __qualname__; three identically-named `class Prog` in this one
        # method collide, so getsource returns the FIRST (pw) — silently compiling the pointwise
        # kernel for every softmax/rms sweep point (device run4: sweep_pw.cpp vs the softmax golden).
        # The function-name rename alone did NOT fix it; the class name is the resolution key. The
        # fixed cases don't collide because each `class Prog` lives in a different case class.
        if self.kernel == "pw":

            @pl.program
            class ProgPw:
                @pl.function(attrs={"auto_fuse": True})
                def sweep_pw(self, x: pl.Tensor[[M, N], DT]) -> pl.Tensor[[M, N], DT]:
                    a: pl.Tensor[[M, N], DT] = pl.add(x, 1.0)
                    b: pl.Tensor[[M, N], DT] = pl.mul(a, 2.0)
                    return b

            return ProgPw
        if self.kernel == "softmax":

            @pl.program
            class ProgSoftmax:
                @pl.function(attrs={"auto_fuse": True})
                def sweep_softmax(self, x: pl.Tensor[[M, N], DT]) -> pl.Tensor[[M, N], DT]:
                    m: pl.Tensor[[M, 1], DT] = pl.row_max(x)
                    s: pl.Tensor[[M, N], DT] = pl.row_expand_sub(x, m)
                    e: pl.Tensor[[M, N], DT] = pl.exp(s)
                    d: pl.Tensor[[M, 1], DT] = pl.row_sum(e)
                    o: pl.Tensor[[M, N], DT] = pl.row_expand_div(e, d)
                    return o

            return ProgSoftmax

        @pl.program
        class ProgRms:
            @pl.function(attrs={"auto_fuse": True})
            def sweep_rms(self, x: pl.Tensor[[M, N], DT]) -> pl.Tensor[[M, N], DT]:
                sq: pl.Tensor[[M, N], DT] = pl.mul(x, x)
                ss: pl.Tensor[[M, 1], DT] = pl.row_sum(sq)
                inv: pl.Tensor[[M, 1], DT] = pl.rsqrt(ss)
                o: pl.Tensor[[M, N], DT] = pl.row_expand_mul(x, inv)
                return o

        return ProgRms

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"].to(torch.float32)  # reference in fp32; result cast back to the tile dtype
        if self.kernel == "pw":
            r = (x + 1.0) * 2.0
        elif self.kernel == "softmax":
            r = torch.softmax(x, dim=-1)
        else:  # bare rms: x * rsqrt(sum(x^2))
            r = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True))
        tensors["out"][:] = r.to(tensors["out"].dtype)


# Attention (q@k -> scaled softmax -> p@v) swept across seq/head shapes — the marquee model
# fragment AND the matmul-ending output-wiring fix (a matmul return copied into an appended Out
# param). Each is a full single-head attention block.
ATTN_GRID = [(128, 64), (64, 64), (256, 64), (128, 32)]

CUBE_M, CUBE_K, CUBE_N = 192, 64, 256


class AutoFusePtoIsaVectorFusionCase(PTOTestCase):
    """PTO-ISA fused_add_relu_mul analogue, expressed in tensor AutoFuse.

    PTO-ISA uses the dedicated TRELU instruction.  The tensor DSL expresses the
    same operation as maximum(x, 0), which lowers to TMAXS.  The host structural
    test verifies that this remains one pipelined AIV kernel with no intermediate
    GM round-trip; this device test verifies the equivalent algorithm numerically.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_isa_fused_add_relu_mul_512x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [512, 1024], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [512, 1024], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def fused(self, x: pl.Tensor[[512, 1024], pl.FP32]) -> pl.Tensor[[512, 1024], pl.FP32]:
                biased: pl.Tensor[[512, 1024], pl.FP32] = pl.add(x, 1.0)
                relu: pl.Tensor[[512, 1024], pl.FP32] = pl.maximum(biased, 0.0)
                out: pl.Tensor[[512, 1024], pl.FP32] = pl.mul(relu, 0.5)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.relu(tensors["x"] + 1.0) * 0.5


class AutoTileVectorFusionCase(AutoFusePtoIsaVectorFusionCase):
    """Whole-function AutoTile pointwise chain with no generic-emitter environment gate."""

    def get_name(self) -> str:
        return "autotile_whole_add_relu_mul_512x1024"

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_tile": True})
            def fused(self, x: pl.Tensor[[512, 1024], pl.FP32]) -> pl.Tensor[[512, 1024], pl.FP32]:
                biased: pl.Tensor[[512, 1024], pl.FP32] = pl.add(x, 1.0)
                relu: pl.Tensor[[512, 1024], pl.FP32] = pl.maximum(biased, 0.0)
                out: pl.Tensor[[512, 1024], pl.FP32] = pl.mul(relu, 0.5)
                return out

        return Prog


class AutoFusePtoKernelsAbsCase(PTOTestCase):
    """PTO-Kernels ``kernel_abs.cpp`` analogue: one streamed TABS."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_kernels_abs_512x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [512, 1024], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [512, 1024], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def absolute(self, x: pl.Tensor[[512, 1024], pl.FP32]) -> pl.Tensor[[512, 1024], pl.FP32]:
                out: pl.Tensor[[512, 1024], pl.FP32] = pl.abs(x)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["x"])


class AutoFusePtoKernelsSiluCase(PTOTestCase):
    """PTO-Kernels FP16 JIT SiLU analogue."""

    __test__ = False

    M = 256
    N = 1024

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_kernels_silu_fp16_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "x",
                [self.M, self.N],
                DataType.FP16,
                init_value=lambda: torch.randn(self.M, self.N, dtype=torch.float16) * 0.5,
            ),
            TensorSpec("out", [self.M, self.N], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.M, self.N

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def silu(self, x: pl.Tensor[[m, n], pl.FP16]) -> pl.Tensor[[m, n], pl.FP16]:
                neg: pl.Tensor[[m, n], pl.FP16] = pl.neg(x)
                exp: pl.Tensor[[m, n], pl.FP16] = pl.exp(neg)
                one: pl.Scalar[pl.FP16] = pl.cast(1.0, pl.FP16)
                denom: pl.Tensor[[m, n], pl.FP16] = pl.add(exp, one)
                out: pl.Tensor[[m, n], pl.FP16] = pl.div(x, denom)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"].float()
        tensors["out"][:] = (x * torch.sigmoid(x)).half()


class AutoFusePtoKernelsSwiGluCase(PTOTestCase):
    """PTO-Kernels FP16 SwiGLU analogue."""

    __test__ = False

    M = 256
    N = 1024

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_kernels_swiglu_fp16_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        shape = [self.M, self.N]
        return [
            TensorSpec(
                "gate",
                shape,
                DataType.FP16,
                init_value=lambda: torch.randn(self.M, self.N, dtype=torch.float16) * 0.5,
            ),
            TensorSpec(
                "up",
                shape,
                DataType.FP16,
                init_value=lambda: torch.randn(self.M, self.N, dtype=torch.float16) * 0.5,
            ),
            TensorSpec("out", shape, DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.M, self.N

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def swiglu(
                self,
                gate: pl.Tensor[[m, n], pl.FP16],
                up: pl.Tensor[[m, n], pl.FP16],
            ) -> pl.Tensor[[m, n], pl.FP16]:
                neg: pl.Tensor[[m, n], pl.FP16] = pl.neg(gate)
                exp: pl.Tensor[[m, n], pl.FP16] = pl.exp(neg)
                one: pl.Scalar[pl.FP16] = pl.cast(1.0, pl.FP16)
                denom: pl.Tensor[[m, n], pl.FP16] = pl.add(exp, one)
                silu: pl.Tensor[[m, n], pl.FP16] = pl.div(gate, denom)
                out: pl.Tensor[[m, n], pl.FP16] = pl.mul(silu, up)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        gate = tensors["gate"].float()
        up = tensors["up"].float()
        tensors["out"][:] = ((gate * torch.sigmoid(gate)) * up).half()


class AutoFusePtoKernelsLayerNormCase(PTOTestCase):
    """PTO-Kernels FP16 affine LayerNorm semantics.

    The host reference comparison records that AutoFuse currently emits two
    AIV kernels, whereas the hand-written PTO-Kernels implementation overlays
    the mean and variance/apply phases in one kernel.
    """

    __test__ = False

    M = 32
    N = 1024

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_kernels_layernorm_fp16_32x1024"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [self.M, self.N], DataType.FP16, init_value=torch.randn),
            TensorSpec("gamma", [1, self.N], DataType.FP16, init_value=torch.randn),
            TensorSpec("beta", [1, self.N], DataType.FP16, init_value=torch.randn),
            TensorSpec("out", [self.M, self.N], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.M, self.N

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def layernorm(
                self,
                x: pl.Tensor[[m, n], pl.FP16],
                gamma: pl.Tensor[[1, n], pl.FP16],
                beta: pl.Tensor[[1, n], pl.FP16],
            ) -> pl.Tensor[[m, n], pl.FP16]:
                x32: pl.Tensor[[m, n], pl.FP32] = pl.cast(x, pl.FP32)
                sum_x: pl.Tensor[[m, 1], pl.FP32] = pl.row_sum(x32)
                mean: pl.Tensor[[m, 1], pl.FP32] = pl.mul(sum_x, 1.0 / n)
                centered: pl.Tensor[[m, n], pl.FP32] = pl.row_expand_sub(x32, mean)
                square: pl.Tensor[[m, n], pl.FP32] = pl.mul(centered, centered)
                sum_square: pl.Tensor[[m, 1], pl.FP32] = pl.row_sum(square)
                variance: pl.Tensor[[m, 1], pl.FP32] = pl.mul(sum_square, 1.0 / n)
                variance_eps: pl.Tensor[[m, 1], pl.FP32] = pl.add(variance, 1.0e-5)
                inv_std: pl.Tensor[[m, 1], pl.FP32] = pl.rsqrt(variance_eps)
                normalized: pl.Tensor[[m, n], pl.FP32] = pl.row_expand_mul(centered, inv_std)
                gamma32: pl.Tensor[[1, n], pl.FP32] = pl.cast(gamma, pl.FP32)
                beta32: pl.Tensor[[1, n], pl.FP32] = pl.cast(beta, pl.FP32)
                scaled: pl.Tensor[[m, n], pl.FP32] = pl.mul(normalized, gamma32)
                shifted: pl.Tensor[[m, n], pl.FP32] = pl.add(scaled, beta32)
                out: pl.Tensor[[m, n], pl.FP16] = pl.cast(shifted, pl.FP16)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"].float()
        gamma = tensors["gamma"].float()
        beta = tensors["beta"].float()
        centered = x - x.mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + 1.0e-5)
        tensors["out"][:] = (normalized * gamma + beta).half()


class AutoFusePtoasFfnActivationCase(PTOTestCase):
    """PTOAS ``FFN/ffn_act.pto`` clipped-cubic activation stage."""

    __test__ = False

    M = 32
    N = 32

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_ptoas_ffn_activation_fp32_32x32"

    def define_tensors(self) -> list[TensorSpec]:
        shape = [self.M, self.N]
        return [
            TensorSpec("h1", shape, DataType.FP32, init_value=torch.randn),
            TensorSpec("h2", shape, DataType.FP32, init_value=torch.randn),
            TensorSpec("out", shape, DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.M, self.N

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def activate(
                self,
                h1: pl.Tensor[[m, n], pl.FP32],
                h2: pl.Tensor[[m, n], pl.FP32],
            ) -> pl.Tensor[[m, n], pl.FP32]:
                low: pl.Tensor[[m, n], pl.FP32] = pl.maximum(h1, -4.0001)
                clipped: pl.Tensor[[m, n], pl.FP32] = pl.minimum(low, 4.0001)
                square: pl.Tensor[[m, n], pl.FP32] = pl.mul(clipped, clipped)
                cube: pl.Tensor[[m, n], pl.FP32] = pl.mul(square, clipped)
                cubic: pl.Tensor[[m, n], pl.FP32] = pl.mul(cube, -1.0 / 48.0)
                linear: pl.Tensor[[m, n], pl.FP32] = pl.mul(clipped, 0.25)
                shifted: pl.Tensor[[m, n], pl.FP32] = pl.add(linear, 0.5)
                gate: pl.Tensor[[m, n], pl.FP32] = pl.add(shifted, cubic)
                out: pl.Tensor[[m, n], pl.FP32] = pl.mul(gate, h2)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        clipped = torch.clamp(tensors["h1"], min=-4.0001, max=4.0001)
        gate = 0.5 + 0.25 * clipped - clipped.pow(3) / 48.0
        tensors["out"][:] = gate * tensors["h2"]


class AutoFusePtoDslGeGluCase(PTOTestCase):
    """PTO-DSL tanh-based GEGLU analogue, statically shaped for AutoFuse."""

    __test__ = False

    M = 256
    N = 1024

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_dsl_geglu_fp16_256x1024"

    def define_tensors(self) -> list[TensorSpec]:
        shape = [self.M, self.N]
        return [
            TensorSpec(
                "gate",
                shape,
                DataType.FP16,
                init_value=lambda: torch.randn(self.M, self.N, dtype=torch.float16) * 0.25,
            ),
            TensorSpec(
                "up",
                shape,
                DataType.FP16,
                init_value=lambda: torch.randn(self.M, self.N, dtype=torch.float16) * 0.25,
            ),
            TensorSpec("out", shape, DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.M, self.N

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def geglu(
                self,
                gate: pl.Tensor[[m, n], pl.FP16],
                up: pl.Tensor[[m, n], pl.FP16],
            ) -> pl.Tensor[[m, n], pl.FP16]:
                zero: pl.Tensor[[m, n], pl.FP16] = pl.sub(gate, gate)
                ones: pl.Tensor[[m, n], pl.FP16] = pl.exp(zero)
                twice: pl.Tensor[[m, n], pl.FP16] = pl.add(gate, gate)
                exp_twice: pl.Tensor[[m, n], pl.FP16] = pl.exp(twice)
                numerator: pl.Tensor[[m, n], pl.FP16] = pl.sub(exp_twice, ones)
                denominator: pl.Tensor[[m, n], pl.FP16] = pl.add(exp_twice, ones)
                tanh_gate: pl.Tensor[[m, n], pl.FP16] = pl.div(numerator, denominator)
                one_plus_tanh: pl.Tensor[[m, n], pl.FP16] = pl.add(tanh_gate, ones)
                gated: pl.Tensor[[m, n], pl.FP16] = pl.mul(gate, one_plus_tanh)
                twos: pl.Tensor[[m, n], pl.FP16] = pl.add(ones, ones)
                gelu: pl.Tensor[[m, n], pl.FP16] = pl.div(gated, twos)
                out: pl.Tensor[[m, n], pl.FP16] = pl.mul(gelu, up)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        gate, up = tensors["gate"], tensors["up"]
        exp_twice = torch.exp(gate + gate)
        tanh_gate = (exp_twice - 1.0) / (exp_twice + 1.0)
        tensors["out"][:] = (0.5 * gate * (1.0 + tanh_gate)) * up


class AutoFusePtoIsaGemmCase(PTOTestCase):
    """1536^3 FP16->FP32 GEMM from the PTO-ISA performance table."""

    __test__ = False

    M = 1536

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_isa_gemm_fp16_fp32_1536"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.M], DataType.FP16, init_value=torch.randn),
            TensorSpec("b", [self.M, self.M], DataType.FP16, init_value=torch.randn),
            TensorSpec("out", [self.M, self.M], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        m = self.M

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[m, m], pl.FP16],
                b: pl.Tensor[[m, m], pl.FP16],
            ) -> pl.Tensor[[m, m], pl.FP32]:
                out: pl.Tensor[[m, m], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["a"].float() @ tensors["b"].float()


class AutoFusePtoIsaChainGemmCase(PTOTestCase):
    """PTO-ISA fused GEMM-chain analogue with an L1-resident BF16 intermediate."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_pto_isa_chain_gemm_bf16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [128, 256],
                DataType.BF16,
                init_value=lambda: torch.randn(128, 256, dtype=torch.bfloat16) * 0.05,
            ),
            TensorSpec(
                "b",
                [256, 128],
                DataType.BF16,
                init_value=lambda: torch.randn(256, 128, dtype=torch.bfloat16) * 0.05,
            ),
            TensorSpec(
                "d",
                [128, 256],
                DataType.BF16,
                init_value=lambda: torch.randn(128, 256, dtype=torch.bfloat16) * 0.05,
            ),
            TensorSpec("out", [128, 256], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.BF16],
                b: pl.Tensor[[256, 128], pl.BF16],
                d: pl.Tensor[[128, 256], pl.BF16],
            ) -> pl.Tensor[[128, 256], pl.BF16]:
                intermediate: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(a, b)
                out: pl.Tensor[[128, 256], pl.BF16] = pl.matmul(intermediate, d)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        intermediate = tensors["a"] @ tensors["b"]
        tensors["out"][:] = intermediate @ tensors["d"]


class AutoTileChainGemmCase(AutoFusePtoIsaChainGemmCase):
    """Whole-function AutoTile cube chain with an L1-resident BF16 intermediate."""

    def get_name(self) -> str:
        return "autotile_whole_chain_gemm_bf16"

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_tile": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.BF16],
                b: pl.Tensor[[256, 128], pl.BF16],
                d: pl.Tensor[[128, 256], pl.BF16],
            ) -> pl.Tensor[[128, 256], pl.BF16]:
                intermediate: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(a, b)
                out: pl.Tensor[[128, 256], pl.BF16] = pl.matmul(intermediate, d)
                return out

        return Prog


class AutoFuseCubeMatmulKRingCase(PTOTestCase):
    """Pure-cube control for the four-window GM->L1 stage ring."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_cube_matmul_k_ring_192x64x256"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [CUBE_M, CUBE_K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [CUBE_K, CUBE_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [CUBE_M, CUBE_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[CUBE_M, CUBE_K], pl.FP32],
                b: pl.Tensor[[CUBE_K, CUBE_N], pl.FP32],
            ) -> pl.Tensor[[CUBE_M, CUBE_N], pl.FP32]:
                out: pl.Tensor[[CUBE_M, CUBE_N], pl.FP32] = pl.matmul(a, b)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["a"] @ tensors["b"]


class AutoFuseCubeEpilogueKRingCase(PTOTestCase):
    """The silicon reproducer: four-window cube followed by a cut vector epilogue."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "autofuse_cube_epilogue_k_ring_192x64x256"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [CUBE_M, CUBE_K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [CUBE_K, CUBE_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("bias", [1, CUBE_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [CUBE_M, CUBE_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def epilogue(
                self,
                a: pl.Tensor[[CUBE_M, CUBE_K], pl.FP32],
                b: pl.Tensor[[CUBE_K, CUBE_N], pl.FP32],
                bias: pl.Tensor[[1, CUBE_N], pl.FP32],
            ) -> pl.Tensor[[CUBE_M, CUBE_N], pl.FP32]:
                mm: pl.Tensor[[CUBE_M, CUBE_N], pl.FP32] = pl.matmul(a, b)
                out: pl.Tensor[[CUBE_M, CUBE_N], pl.FP32] = pl.add(mm, bias)
                return out

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["a"] @ tensors["b"] + tensors["bias"]


class AutoFuseMixedC2VEpilogueCase(AutoFuseCubeEpilogueKRingCase):
    """Same graph as the cut reproducer, forced through the one-way C->V FIFO."""

    __test__ = False

    def get_name(self) -> str:
        return "autofuse_mixed_c2v_epilogue_192x64x256"


class ModelAttentionSweepCase(PTOTestCase):
    """One attention block at (seq=S, head_dim=D). Exercises the matmul-ending output wiring
    across shapes — the fix for the all-zero-output device regression."""

    __test__ = False

    def __init__(self, S: int, D: int, *, platform: str | None = None, config=None):
        self.S, self.D = S, D
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"model_attention_{self.S}x{self.D}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("q", [self.S, self.D], DataType.FP32, init_value=torch.randn),
            TensorSpec("k", [self.D, self.S], DataType.FP32, init_value=torch.randn),
            TensorSpec("v", [self.S, self.D], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [self.S, self.D], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        S, D = self.S, self.D

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def attn(
                self,
                q: pl.Tensor[[S, D], pl.FP32],
                k: pl.Tensor[[D, S], pl.FP32],
                v: pl.Tensor[[S, D], pl.FP32],
            ) -> pl.Tensor[[S, D], pl.FP32]:
                s: pl.Tensor[[S, S], pl.FP32] = pl.matmul(q, k)
                sc: pl.Tensor[[S, S], pl.FP32] = pl.mul(s, 0.125)
                m: pl.Tensor[[S, 1], pl.FP32] = pl.row_max(sc)
                dd: pl.Tensor[[S, S], pl.FP32] = pl.row_expand_sub(sc, m)
                e: pl.Tensor[[S, S], pl.FP32] = pl.exp(dd)
                sm: pl.Tensor[[S, 1], pl.FP32] = pl.row_sum(e)
                p: pl.Tensor[[S, S], pl.FP32] = pl.row_expand_div(e, sm)
                o: pl.Tensor[[S, D], pl.FP32] = pl.matmul(p, v)
                return o

        return Prog

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        q, k, v = tensors["q"], tensors["k"], tensors["v"]
        tensors["out"][:] = torch.softmax((q @ k) * 0.125, dim=-1) @ v


class TestAutoFuseDevice:
    """AutoFuse on device: the reduction-valid probe + free-axis padding numerics."""

    # -- Part A: the decisive reduction-valid probe --

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_rowsum_honors_valid(self, test_runner, platform):
        result = test_runner.run(RowSumValidProbeCase(platform=platform))
        assert result.passed, (
            "trowsum honored valid_col? If this FAILS with device out ~6e9, the op sums the "
            f"PHYSICAL extent -> reduced-axis padding needs zero-fill. {result.error}"
        )

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_rowmax_honors_valid(self, test_runner, platform):
        result = test_runner.run(RowMaxValidProbeCase(platform=platform))
        assert result.passed, (
            "trowmax honored valid_col? If this FAILS with device out ~1e9, the op maxes the "
            f"PHYSICAL extent -> max-reduced-axis padding needs an identity (-inf) fill. {result.error}"
        )

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_colsum_honors_valid(self, test_runner, platform):
        result = test_runner.run(ColSumValidProbeCase(platform=platform))
        assert result.passed, (
            "tcolsum honored valid_row? If this FAILS with device out ~6e9, the op sums the "
            f"PHYSICAL extent -> reduced-axis padding needs zero-fill. {result.error}"
        )

    # -- Part B: AutoFuse end-to-end numerics on device (set PYPTO_AUTOFUSE_GENERIC_EMIT=1) --
    #
    # The return->named-output wiring is now handled in the compiler: AutoFuse
    # (MaybeLiftReturnToOutParam) lifts a return-based fused function's output buffer into an
    # appended Out param, so orchestration codegen emits the add_output write-back the harness
    # binds by position ([in..., out]). Verified compile-side (expected_arg_count matches, the
    # output param carries the write-back). These run the fully-lowered kernels on hardware.

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_ragged_pointwise(self, test_runner, platform):
        result = test_runner.run(AutoFuseRaggedPointwiseCase(platform=platform))
        assert result.passed, f"AutoFuse ragged pointwise [130,66] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_softmax(self, test_runner, platform):
        result = test_runner.run(AutoFuseSoftmaxCase(platform=platform))
        assert result.passed, f"AutoFuse softmax [256,128] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_rmsnorm(self, test_runner, platform):
        # rtol 1e-3: RMSNorm calls HW rsqrt (~1e-4 approximation vs exact torch.rsqrt).
        result = test_runner.run(AutoFuseRmsNormCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"AutoFuse RMSNorm [256,512] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_layernorm(self, test_runner, platform):
        # rtol 1e-3: LayerNorm calls HW rsqrt (~1e-4 approximation vs exact torch.rsqrt).
        result = test_runner.run(AutoFuseLayerNormCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"AutoFuse LayerNorm [256,512] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_p4_softmax_wide(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")
        result = test_runner.run(AutoFuseP4SoftmaxWideCase(platform=platform))
        assert result.passed, f"P4 softmax [128,8192] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_p4_layernorm_wide(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")
        result = test_runner.run(AutoFuseP4LayerNormWideCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"P4 layernorm [128,8192] at mean+2000 mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_p4_scaled_softmax_wide_cuts(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_P4", "1")
        result = test_runner.run(AutoFuseP4ScaledSoftmaxWideCase(platform=platform))
        assert result.passed, f"Scaled-softmax P4 near miss [128,8192] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_p2_apply_input_wide(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFuseP2ApplyInputWideCase(platform=platform))
        assert result.passed, f"P2 apply-only input [128,8192] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_pw_wide_short(self, test_runner, platform):
        result = test_runner.run(AutoFusePwWideShortCase(platform=platform))
        assert result.passed, f"AutoFuse wide-short pointwise [64,4096] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_pw_tall(self, test_runner, platform):
        result = test_runner.run(AutoFusePwTallCase(platform=platform))
        assert result.passed, f"AutoFuse tall pointwise [4096,64] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_softmax_ragged_n(self, test_runner, platform):
        result = test_runner.run(AutoFuseSoftmaxRaggedNCase(platform=platform))
        assert result.passed, f"AutoFuse softmax ragged-N [256,66] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_row_reduce_broadcast(self, test_runner, platform):
        result = test_runner.run(AutoFuseRowReduceBroadcastCase(platform=platform))
        assert result.passed, f"AutoFuse row-reduce+broadcast [256,128] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_fp16_softmax(self, test_runner, platform):
        # rtol 1e-2: end-to-end FP16 is at the fp16 rounding floor (eps ~1e-3); the FP32 softmax
        # with the identical `exp` emit passes at 1e-5, so this is fp16 precision, not the emit.
        result = test_runner.run(AutoFuseFp16SoftmaxCase(platform=platform, config=_FP16_TOL))
        assert result.passed, f"AutoFuse FP16 softmax [256,128] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_col_sum(self, test_runner, platform):
        result = test_runner.run(AutoFuseColSumCase(platform=platform))
        assert result.passed, f"AutoFuse col_sum [128,256] (S2 split) mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("orientation", ["row", "column"])
    def test_autofuse_singleton_reduction_generated_kernel(
        self, test_runner, platform, orientation, monkeypatch
    ):
        """TestRunner reaches PTOAS, generated-kernel compilation, and silicon execution."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFuseSingletonReductionCase(orientation, platform=platform))
        assert result.passed, f"AutoFuse singleton-{orientation} reduction failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_fork(self, test_runner, platform):
        result = test_runner.run(AutoFuseForkCase(platform=platform))
        assert result.passed, f"AutoFuse multi-sink fork [256,256] mismatch on device: {result.error}"

    # -- Part B2: whole-function AutoTile end-to-end --
    #
    # FORCE_MERGE=none is deliberately hostile: AutoTile must ignore the
    # partition override, form one exact group, and engage the plan-driven
    # emitter even though PYPTO_AUTOFUSE_GENERIC_EMIT is absent.

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autotile_whole_vector_chain(self, test_runner, platform, monkeypatch):
        monkeypatch.delenv("PYPTO_AUTOFUSE_GENERIC_EMIT", raising=False)
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "none")
        case = AutoTileVectorFusionCase(platform=platform)
        _assert_one_autotile_group(case, "fused")
        result = test_runner.run(case)
        assert result.passed, f"AutoTile whole vector chain mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autotile_whole_softmax(self, test_runner, platform, monkeypatch):
        monkeypatch.delenv("PYPTO_AUTOFUSE_GENERIC_EMIT", raising=False)
        monkeypatch.delenv("PYPTO_AUTOFUSE_P4", raising=False)
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "none")
        case = AutoTileSoftmaxCase(platform=platform)
        _assert_one_autotile_group(case, "sm")
        result = test_runner.run(case)
        assert result.passed, f"AutoTile whole softmax mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autotile_whole_cube_chain(self, test_runner, platform, monkeypatch):
        monkeypatch.delenv("PYPTO_AUTOFUSE_GENERIC_EMIT", raising=False)
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "none")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "0")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        case = AutoTileChainGemmCase(platform=platform, config=_BF16_CUBE_TOL)
        _assert_one_autotile_group(case, "chain")
        result = test_runner.run(case)
        assert result.passed, f"AutoTile whole cube chain mismatch on device: {result.error}"

    # -- Part C: model-fragment experiments (transformer components) --

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_rmsnorm_wide(self, test_runner, platform):
        result = test_runner.run(ModelRmsNormWideCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"RMSNorm [256,1024] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_layernorm_wide(self, test_runner, platform):
        result = test_runner.run(ModelLayerNormWideCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"LayerNorm [256,1024] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_residual_rmsnorm(self, test_runner, platform):
        result = test_runner.run(ModelResidualRmsNormCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"Residual+RMSNorm [256,1024] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_silu(self, test_runner, platform):
        result = test_runner.run(ModelSiluCase(platform=platform))
        assert result.passed, f"SiLU [256,1024] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_swiglu(self, test_runner, platform):
        result = test_runner.run(ModelSwiGluCase(platform=platform))
        assert result.passed, f"SwiGLU [256,1024] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_scaled_softmax(self, test_runner, platform):
        result = test_runner.run(ModelScaledSoftmaxCase(platform=platform))
        assert result.passed, f"Scaled softmax [256,512] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_interleaved_twin(self, test_runner, platform):
        result = test_runner.run(ModelInterleavedTwinCase(platform=platform))
        assert result.passed, f"Interleaved twin [256,512] (group-reorder) mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_model_attention(self, test_runner, platform):
        result = test_runner.run(ModelAttentionCase(platform=platform, config=_RSQRT_TOL))
        assert result.passed, f"Attention block [128,64] mismatch on device: {result.error}"

    # -- Part D: shape x dtype sweep (wide coverage) --

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("kernel,M,N,dt", SWEEP_GRID)
    def test_autofuse_sweep(self, test_runner, platform, kernel, M, N, dt):
        # fp16 -> the rounding-floor tolerance; fp32 rms -> the HW-rsqrt tolerance; fp32 pw/softmax
        # are exact (no rsqrt), default tight tolerance.
        cfg = _FP16_TOL if dt == "fp16" else (_RSQRT_TOL if kernel == "rms" else None)
        result = test_runner.run(AutoFuseSweepCase(kernel, M, N, dt, platform=platform, config=cfg))
        assert result.passed, f"AutoFuse sweep {kernel}[{M},{N}]/{dt} mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("S,D", ATTN_GRID)
    def test_model_attention_sweep(self, test_runner, platform, S, D):
        # matmul reassociation -> the rsqrt-level tolerance; watch for actual==0.0 (the wiring fix).
        result = test_runner.run(ModelAttentionSweepCase(S, D, platform=platform, config=_RSQRT_TOL))
        assert result.passed, (
            f"AutoFuse attention [{S},{D}] mismatch on device (any actual=0.0?): {result.error}"
        )

    # -- Part E: pure-engine PTO-ISA reference analogues --
    #
    # Host tests compare the lowered instruction dataflow.  These cases retain
    # the analogous algorithms as persistent device correctness/performance
    # controls, so codegen improvements can be compared to the public PTO-ISA
    # kernels rather than only to another AutoFuse revision.

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_isa_vector_fusion(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoIsaVectorFusionCase(platform=platform))
        assert result.passed, f"PTO-ISA add/ReLU/mul analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_kernels_abs(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoKernelsAbsCase(platform=platform))
        assert result.passed, f"PTO-Kernels abs analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_kernels_silu(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoKernelsSiluCase(platform=platform, config=_FP16_TOL))
        assert result.passed, f"PTO-Kernels FP16 SiLU analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_kernels_swiglu(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoKernelsSwiGluCase(platform=platform, config=_FP16_TOL))
        assert result.passed, f"PTO-Kernels FP16 SwiGLU analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_kernels_layernorm(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoKernelsLayerNormCase(platform=platform, config=_FP16_TOL))
        assert result.passed, f"PTO-Kernels FP16 affine LayerNorm analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_ptoas_ffn_activation(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoasFfnActivationCase(platform=platform))
        assert result.passed, (
            f"PTOAS clipped-cubic FFN activation analogue mismatch on device: {result.error}"
        )

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_dsl_geglu(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        result = test_runner.run(AutoFusePtoDslGeGluCase(platform=platform, config=_FP16_TOL))
        assert result.passed, f"PTO-DSL GEGLU analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_isa_gemm(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "0")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        result = test_runner.run(AutoFusePtoIsaGemmCase(platform=platform, config=_FP16_TOL))
        assert result.passed, f"PTO-ISA FP16->FP32 GEMM analogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_autofuse_pto_isa_chain_gemm(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "0")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        result = test_runner.run(AutoFusePtoIsaChainGemmCase(platform=platform, config=_BF16_CUBE_TOL))
        assert result.passed, f"PTO-ISA fused GEMM-chain analogue mismatch on device: {result.error}"

    # Keep forced mixed/cube tests last: FORCE_PLAN is process-cached by design.
    # Device closure runs this file with --forked, so every test still gets a
    # fresh process.

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_zx_autofuse_mixed_c2v_epilogue(self, test_runner, platform, monkeypatch):
        """Exercise the generic one-way C->V FIFO with two logical items per group."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_PLAN", "32,32,1,6,8")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        result = test_runner.run(AutoFuseMixedC2VEpilogueCase(platform=platform, config=_CUBE_TOL))
        assert result.passed, f"AutoFuse mixed C->V epilogue mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.platforms("a2a3")
    def test_zy_autofuse_mixed_dense_swiglu(self, test_runner, platform, monkeypatch):
        """Exercise the exact C,C->V->C FIFO bundle and persistent down accumulator."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_MERGE", "all")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_PLAN", "64,16,1,2,1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        result = test_runner.run(AutoFuseMixedDenseSwiGluCase(platform=platform, config=_BF16_CUBE_TOL))
        assert result.passed, f"AutoFuse mixed dense SwiGLU mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_zz_autofuse_cube_matmul_k_ring(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "0")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_PLAN", "32,32,1,6,8")
        result = test_runner.run(AutoFuseCubeMatmulKRingCase(platform=platform, config=_CUBE_TOL))
        assert result.passed, f"AutoFuse forced cube K-ring mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_zz_autofuse_cube_epilogue_k_ring(self, test_runner, platform, monkeypatch):
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_MIXED", "0")
        monkeypatch.setenv("PYPTO_AUTOFUSE_STRICT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_FORCE_PLAN", "32,32,1,6,8")
        result = test_runner.run(AutoFuseCubeEpilogueKRingCase(platform=platform, config=_CUBE_TOL))
        assert result.passed, f"AutoFuse forced cube epilogue K-ring mismatch on device: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
