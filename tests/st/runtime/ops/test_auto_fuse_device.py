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
        result = test_runner.run(AutoFuseRmsNormCase(platform=platform))
        assert result.passed, f"AutoFuse RMSNorm [256,512] mismatch on device: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_autofuse_layernorm(self, test_runner, platform):
        result = test_runner.run(AutoFuseLayerNormCase(platform=platform))
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
        result = test_runner.run(AutoFuseFp16SoftmaxCase(platform=platform))
        assert result.passed, f"AutoFuse FP16 softmax [256,128] mismatch on device: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
