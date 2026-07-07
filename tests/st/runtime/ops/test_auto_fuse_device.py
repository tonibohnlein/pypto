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

  Part B — AutoFuse end-to-end on device.  Realistic fused-vector kernels (ragged pointwise,
    softmax, RMSNorm, LayerNorm) compiled with ``attrs={"auto_fuse": True}`` and
    ``PYPTO_AUTOFUSE_GENERIC_EMIT=1``, numerically verified against a torch reference on hardware.
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
_RSQRT_TOL = RunConfig(rtol=1e-2, atol=1e-2)   # fp32 norms calling HW rsqrt (accumulated ~5e-3)
_FP16_TOL = RunConfig(rtol=1e-2, atol=1e-2)    # end-to-end fp16 (rounding floor)

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

RPW_M, RPW_N = 130, 66      # ragged pointwise: N=66 free axis padded 66->72
SM_M, SM_N = 256, 128       # softmax: ragged M=256 (h tile padded); reduced N=128 aligned
RMS_M, RMS_N = 256, 512     # RMSNorm: aligned; one reduction (row_sum of squares) + broadcast
LN_M, LN_N = 256, 512       # LayerNorm: aligned; two reductions (mean + variance) + broadcast
NORM_EPS = 1.0e-6
WS_M, WS_N = 64, 4096       # wide-short pointwise: the free-axis over-pad overflow case
TL_M, TL_N = 4096, 64       # tall pointwise: many free-axis strips
SMR_M, SMR_N = 256, 66      # softmax ragged reduced N (padded reduced axis)
RRB_M, RRB_N = 256, 128     # row-reduce + broadcast (reduction intermediate, no div)
F16_M, F16_N = 256, 128     # FP16 softmax (granule g=16)
CS_M, CS_N = 128, 256       # bare col_sum sink -> S2 split-reduction (atomic-add merge)
FK_M, FK_N = 256, 256       # multi-sink fork: two live-outs sharing an input

# --- Part C: model-fragment experiments (realistic transformer components) ---
MRMS_M, MRMS_N = 256, 1024   # wider RMSNorm: exercises the sub-granule reduction-strip cap
MLN_M, MLN_N = 256, 1024     # wider LayerNorm (two reductions)
RES_M, RES_N = 256, 1024     # residual add + RMSNorm (a pre-norm block head)
SILU_M, SILU_N = 256, 1024   # SiLU/Swish activation: x*sigmoid(x) = x/(1+exp(-x))
SWG_M, SWG_N = 256, 1024     # SwiGLU FFN gating: silu(gate)*up (two inputs)
SSM_M, SSM_N = 256, 512      # scaled softmax (attention scores * 1/sqrt(d))
TWIN_M, TWIN_N = 256, 512    # two interleaved independent chains (group-reorder fix)
ATT_S, ATT_D = 128, 64       # attention block: q@k -> scaled softmax -> p@v


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
            def pw(self, a: pl.Tensor[[WS_M, WS_N], pl.FP32], b: pl.Tensor[[WS_M, WS_N], pl.FP32]) -> pl.Tensor[[WS_M, WS_N], pl.FP32]:
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
            def pw(self, a: pl.Tensor[[TL_M, TL_N], pl.FP32], b: pl.Tensor[[TL_M, TL_N], pl.FP32]) -> pl.Tensor[[TL_M, TL_N], pl.FP32]:
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
# repeat). Widths stay <= 1024 for reductions (>= 2048 fused reductions need streaming, not built).
# FP16 covers only the scalar-free reduction kernels (a fp16 tensor + a fp32 scalar const promotes
# to fp32 in the DSL, so fp16 pointwise-with-scalar is skipped — an authoring limitation, not emit).
SWEEP_GRID = [
    ("pw", 64, 64, "fp32"), ("pw", 130, 66, "fp32"), ("pw", 256, 512, "fp32"),
    ("pw", 512, 1024, "fp32"), ("pw", 4096, 64, "fp32"), ("pw", 64, 4096, "fp32"),
    ("softmax", 256, 128, "fp32"), ("softmax", 256, 66, "fp32"), ("softmax", 128, 512, "fp32"),
    ("softmax", 512, 1024, "fp32"), ("softmax", 64, 256, "fp32"),
    ("rms", 256, 512, "fp32"), ("rms", 256, 1024, "fp32"), ("rms", 128, 256, "fp32"),
    ("rms", 130, 128, "fp32"),
    ("softmax", 256, 128, "fp16"), ("softmax", 256, 512, "fp16"), ("softmax", 128, 256, "fp16"),
    ("rms", 256, 512, "fp16"), ("rms", 256, 1024, "fp16"), ("rms", 128, 128, "fp16"),
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
        # Distinct entry-function names per kernel (like the fixed cases sm/rmsnorm/silu) — a shared
        # `def f` risks the harness binding/caching the wrong kernel across sweep points.
        if self.kernel == "pw":

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def sweep_pw(self, x: pl.Tensor[[M, N], DT]) -> pl.Tensor[[M, N], DT]:
                    a: pl.Tensor[[M, N], DT] = pl.add(x, 1.0)
                    b: pl.Tensor[[M, N], DT] = pl.mul(a, 2.0)
                    return b

            return Prog
        if self.kernel == "softmax":

            @pl.program
            class Prog:
                @pl.function(attrs={"auto_fuse": True})
                def sweep_softmax(self, x: pl.Tensor[[M, N], DT]) -> pl.Tensor[[M, N], DT]:
                    m: pl.Tensor[[M, 1], DT] = pl.row_max(x)
                    s: pl.Tensor[[M, N], DT] = pl.row_expand_sub(x, m)
                    e: pl.Tensor[[M, N], DT] = pl.exp(s)
                    d: pl.Tensor[[M, 1], DT] = pl.row_sum(e)
                    o: pl.Tensor[[M, N], DT] = pl.row_expand_div(e, d)
                    return o

            return Prog

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def sweep_rms(self, x: pl.Tensor[[M, N], DT]) -> pl.Tensor[[M, N], DT]:
                sq: pl.Tensor[[M, N], DT] = pl.mul(x, x)
                ss: pl.Tensor[[M, 1], DT] = pl.row_sum(sq)
                inv: pl.Tensor[[M, 1], DT] = pl.rsqrt(ss)
                o: pl.Tensor[[M, N], DT] = pl.row_expand_mul(x, inv)
                return o

        return Prog

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
    def test_autofuse_fork(self, test_runner, platform):
        result = test_runner.run(AutoFuseForkCase(platform=platform))
        assert result.passed, f"AutoFuse multi-sink fork [256,256] mismatch on device: {result.error}"

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
        assert result.passed, f"AutoFuse attention [{S},{D}] mismatch on device (any actual=0.0?): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
