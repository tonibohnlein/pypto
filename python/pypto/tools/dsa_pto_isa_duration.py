# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pinned PTO-ISA duration estimates for DSA schedule analysis."""

import csv
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_FORMULA_RELATIVE_PATH = Path("include/pto/costmodel/a2a3/formula_costmodel/formula_params.csv")
_ARCH_RELATIVE_PATH = Path("include/pto/costmodel/arch_config.hpp")
_LIGHTWEIGHT_RELATIVE_PATH = Path("include/pto/costmodel/lightweight_costmodel.hpp")
_TRANSFER_RELATIVE_PATH = Path("include/pto/costmodel/a2a3/formula_costmodel/formula_backend_transfer.hpp")
_COMPUTE_RELATIVE_PATH = Path("include/pto/costmodel/a2a3/formula_costmodel/formula_backend_compute.hpp")
_PERF_SIM_PROVIDER_RELATIVE_PATH = Path("include/pto/costmodel/perf_sim/costmodel_provider.hpp")
_PERF_SIM_LATENCY_RELATIVE_PATH = Path("include/pto/costmodel/perf_sim/latency.hpp")
_CCE_VECTOR_COMPUTE_RELATIVE_PATH = Path(
    "include/pto/costmodel/a2a3/cce_costmodel/cce_costmodel_vector_compute.hpp"
)
_INSTRUCTION_LOWERING_RELATIVE_PATH = Path("include/pto/common/pto_instr.hpp")
_REQUIRED_PATHS = (
    _FORMULA_RELATIVE_PATH,
    _ARCH_RELATIVE_PATH,
    _LIGHTWEIGHT_RELATIVE_PATH,
    _TRANSFER_RELATIVE_PATH,
    _COMPUTE_RELATIVE_PATH,
    _PERF_SIM_PROVIDER_RELATIVE_PATH,
    _PERF_SIM_LATENCY_RELATIVE_PATH,
    _CCE_VECTOR_COMPUTE_RELATIVE_PATH,
    _INSTRUCTION_LOWERING_RELATIVE_PATH,
)

_FORMULA_OPCODE = {
    "pto.tsub": "TSUB",
    "pto.tmul": "TMUL",
    # A2/A3 lowers reciprocal to TDIVS(dst, 1, src); use the exact lowered
    # instruction family when the pinned table carries this shape.
    "pto.trecip": "TDIVS",
    "pto.tadds": "TADDS",
    "pto.tdivs": "TDIVS",
    "pto.tmuls": "TMULS",
    "pto.tmins": "TMINS",
    "pto.trowsum": "TROWSUM",
    "pto.trowmax": "TROWMAX",
    "pto.tcolsum": "TCOLSUM",
    "pto.tcolmax": "TCOLMAX",
    "pto.trowexpand": "TROWEXPAND",
    # A2/A3 implements these fused variants with the same row-broadcast
    # traversal used to calibrate the TROWEXPAND family.  They remain labelled
    # family approximations rather than exact instruction signatures.
    "pto.trowexpandadd": "TROWEXPAND",
    "pto.trowexpandsub": "TROWEXPAND",
    "pto.texp": "TEXP",
    "pto.tsqrt": "TSQRT",
}
_ANY_PARAMETER_OPS = {"TEXP", "TSQRT"}
_NEAREST_FORMULA_SHAPE_OPS = {
    "pto.trecip",
    "pto.tmins",
    "pto.tcolsum",
    "pto.trowexpandadd",
    "pto.trowexpandsub",
    "pto.trowmax",
}
_MATMUL_OPS = {"pto.tmatmul", "pto.tmatmul.acc"}
_TRANSFER_OPS = {"pto.tload": "TLOAD", "pto.tstore": "TSTORE", "pto.tmov": "TMOV"}
_SCALAR_STAGE_OPS = {"pto.load_scalar", "pto.store_scalar", "pto.tpush", "pto.tpop", "pto.tfree"}
_PERF_SIM_DEFAULT_OPS = {
    "pto.tabs",
    "pto.tadd",
    "pto.tadds",
    "pto.tci",
    "pto.tcolexpand",
    "pto.tcolexpandexpdif",
    "pto.tcolexpandmul",
    "pto.tcvt",
    "pto.texpands",
    "pto.textract",
    "pto.tfillpad",
    "pto.tgather",
    "pto.tgetval",
    "pto.tmrgsort",
    # TMUL first consults the pinned formula table. Shapes absent from that
    # table (including Gate's fp32 1x4096 tile) use Perf-Sim's deterministic
    # vector fallback instead of becoming an unprincipled local constant.
    "pto.tmul",
    "pto.tmuls",
    "pto.tneg",
    "pto.trowexpandmul",
    "pto.trowsum",
    # A2/A3's CCE-backed Perf-Sim model contains an fp32 vrsqrt fit. Its source
    # is included in the portable provider snapshot. High-precision TRSQRT
    # lowers to several instructions and remains unsupported here; calibrated
    # exact-signature overrides belong in the composite duration model.
    "pto.trsqrt",
    "pto.tscatter",
    "pto.tsetval",
    "pto.tsort32",
    "pto.tsubs",
    "pto.ttrans",
}
_PERF_SIM_SOURCE_WORK_OPS = {
    "pto.trowsum",
    "pto.trowmax",
    "pto.trowmin",
    "pto.trowprod",
    "pto.tcolsum",
    "pto.tcolmax",
    "pto.tcolmin",
    "pto.tcolprod",
}
_DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
}
_PTO_DTYPE = {
    "f32": "fp32",
    "f16": "fp16",
    "bf16": "bf16",
    "i8": "i8",
    "ui8": "u8",
    "i16": "i16",
    "ui16": "u16",
    "i32": "i32",
    "ui32": "u32",
}
_TILE_TO_SCOPE = {
    "vec": "VecTile",
    "mat": "MatTile",
    "acc": "AccTile",
    "left": "LeftTile",
    "right": "RightTile",
    "bias": "BiasTile",
    "scaling": "ScalingTile",
}
_TRANSFER_ROUTE = {
    ("TLOAD", "VecTile"): "GM_TO_UB",
    ("TLOAD", "MatTile"): "GM_TO_L1",
    ("TSTORE", "VecTile"): "UB_TO_GM",
    ("TSTORE", "MatTile"): "L1_TO_GM",
    ("TSTORE", "AccTile"): "L0C_TO_GM",
    ("TMOV", "VecTile"): "UB_TO_UB",
    ("TMOV", "MatTile"): "L0C_TO_L1",
    ("TMOV", "LeftTile"): "L1_TO_L0A",
    ("TMOV", "RightTile"): "L1_TO_L0B",
    ("TMOV", "BiasTile"): "L1_TO_BT",
    ("TMOV", "ScalingTile"): "L1_TO_FB",
}
_BANDWIDTH_NAMES = (
    "GM_TO_UB",
    "GM_TO_L1",
    "UB_TO_GM",
    "L1_TO_GM",
    "UB_TO_UB",
    "L0C_TO_GM",
    "L0C_TO_L1",
    "L1_TO_L0A",
    "L1_TO_L0B",
    "L1_TO_BT",
    "L1_TO_FB",
    "L1_FILL",
)
_FREQUENCY_RE = re.compile(r"kMainFrequencyHz\s*=\s*([0-9.eE+-]+)L?\s*;")
_ARCH_CONFIG_RE = re.compile(
    r"kA2A3ArchConfig\s*\{\s*\"a2a3\"\s*,\s*kMainFrequencyHz\s*,\s*\{(?P<body>.*?)\}\s*,?\s*\}",
    re.DOTALL,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_LEGACY_TILE_RE = re.compile(
    r"!pto\.tile_buf<(?P<scope>[a-z0-9_]+)\s*,\s*(?P<shape>(?:\d+x)*\d+x[a-z0-9]+)",
    re.IGNORECASE,
)
_KEYED_TILE_RE = re.compile(
    r"!pto\.tile_buf<[^>]*\bloc=(?P<scope>[a-z0-9_]+)[^>]*\bdtype=(?P<dtype>[a-z0-9]+)"
    r"[^>]*\brows=(?P<rows>\d+)[^>]*\bcols=(?P<cols>\d+)",
    re.IGNORECASE,
)
_SHAPE_DTYPE_RE = re.compile(r"(?P<shape>(?:\d+x)*\d+)x(?P<dtype>[a-z][a-z0-9]*)$")
_PARTITION_TYPE_RE = re.compile(
    r"!pto\.partition_tensor_view<(?P<shape>(?:\d+x)*\d+)x(?P<dtype>[a-z][a-z0-9]*)>",
    re.IGNORECASE,
)
_SCALAR_TYPE_FULL_RE = re.compile(r"(?:bf16|f\d+|(?:ui|i)\d+|index)")


@dataclass(frozen=True)
class FormulaParameter:
    """One fitted PTO-ISA formula row."""

    op: str
    dtype: str
    cols: int | None
    slope: float
    bias: float


@dataclass(frozen=True)
class TileType:
    """The shape and element type needed by PTO-ISA's lightweight model."""

    scope: str
    rows: int
    cols: int
    dtype: str


@dataclass(frozen=True)
class DurationEstimate:
    """One operation-duration estimate with auditable provenance."""

    cycles: float
    source: str
    detail: str
    evidence_class: str
    fallback: bool = False


@dataclass(frozen=True)
class PipelineEstimate:
    """Incremental pipeline state used at a synchronization boundary.

    ``startup_cycles`` is repaid by this operation when a barrier has emptied
    its execution stream. ``pending_tail_cycles`` is work left behind by this
    operation for a later barrier to drain. These are deliberately separate
    from the inclusive operation duration: a single-operation measurement can
    identify their sum, but not an arbitrary head/tail split.
    """

    startup_cycles: float
    pending_tail_cycles: float
    source: str
    detail: str


@dataclass
class PtoIsaDurationProvider:
    """Portable snapshot of the A2/A3 PTO-ISA lightweight duration inputs."""

    revision: str
    frequency_hz: float
    bandwidth_gib_per_s: dict[str, float]
    formula_parameters: list[FormulaParameter]
    source_sha256: dict[str, str]
    unsupported_policy: str = "error"
    fallback_cycles: float = 1.0
    provider_version: str = "pto_isa_a2a3_v2"

    @classmethod
    def from_checkout(
        cls,
        root: str | Path,
        *,
        expected_revision: str,
        unsupported_policy: str = "error",
        fallback_cycles: float = 1.0,
    ) -> "PtoIsaDurationProvider":
        """Load cost data only from an exact, unmodified PTO-ISA revision."""
        checkout = Path(root).resolve()
        if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
            raise ValueError(
                f"expected PTO-ISA revision must be a full 40-hex SHA, got {expected_revision!r}"
            )
        actual = _git_output(checkout, "rev-parse", "HEAD")
        if actual != expected_revision:
            raise ValueError(f"PTO-ISA checkout is off pin: expected {expected_revision}, got {actual}")
        missing = [str(path) for path in _REQUIRED_PATHS if not (checkout / path).is_file()]
        if missing:
            raise ValueError(f"PTO-ISA checkout is missing cost-model sources: {missing}")
        changed = _git_output(
            checkout,
            "status",
            "--short",
            "--",
            *(str(path) for path in _REQUIRED_PATHS),
        )
        if changed:
            raise ValueError(f"PTO-ISA cost-model sources are modified at the pinned revision: {changed}")
        _validate_policy(unsupported_policy, fallback_cycles)
        arch_text = (checkout / _ARCH_RELATIVE_PATH).read_text()
        frequency_hz, bandwidth = _parse_arch_config(arch_text)
        formula_parameters = _parse_formula_parameters(checkout / _FORMULA_RELATIVE_PATH)
        hashes = {
            str(path): hashlib.sha256((checkout / path).read_bytes()).hexdigest() for path in _REQUIRED_PATHS
        }
        return cls(
            revision=actual,
            frequency_hz=frequency_hz,
            bandwidth_gib_per_s=bandwidth,
            formula_parameters=formula_parameters,
            source_sha256=hashes,
            unsupported_policy=unsupported_policy,
            fallback_cycles=fallback_cycles,
        )

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "PtoIsaDurationProvider":
        """Restore an embedded, portable provider snapshot."""
        provider_version = value.get("provider_version")
        if value.get("schema_version") != 1 or provider_version not in {
            "pto_isa_a2a3_v1",
            "pto_isa_a2a3_v2",
        }:
            raise ValueError("unsupported PTO-ISA duration-provider schema")
        parameters = value.get("formula_parameters")
        if not isinstance(parameters, list):
            raise ValueError("PTO-ISA duration provider is missing formula_parameters")
        bandwidth = value.get("bandwidth_gib_per_s")
        hashes = value.get("source_sha256")
        if not isinstance(bandwidth, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("PTO-ISA duration provider is missing bandwidth or source hashes")
        policy = str(value.get("unsupported_policy", "error"))
        fallback_cycles = float(value.get("fallback_cycles", 1.0))
        _validate_policy(policy, fallback_cycles)
        return cls(
            revision=str(value["revision"]),
            frequency_hz=float(value["frequency_hz"]),
            bandwidth_gib_per_s={str(key): float(item) for key, item in bandwidth.items()},
            formula_parameters=[FormulaParameter(**item) for item in parameters],
            source_sha256={str(key): str(item) for key, item in hashes.items()},
            unsupported_policy=policy,
            fallback_cycles=fallback_cycles,
            provider_version=str(provider_version),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the complete portable snapshot used for predictions."""
        return {
            "schema_version": 1,
            "provider_version": self.provider_version,
            "revision": self.revision,
            "frequency_hz": self.frequency_hz,
            "bandwidth_gib_per_s": dict(sorted(self.bandwidth_gib_per_s.items())),
            "formula_parameters": [asdict(parameter) for parameter in self.formula_parameters],
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "unsupported_policy": self.unsupported_policy,
            "fallback_cycles": self.fallback_cycles,
        }

    def estimate(self, node: Mapping[str, Any], *, work_bytes: int) -> DurationEstimate:
        """Estimate one PTO operation using the pinned lightweight model inputs."""
        op_name = node.get("op_name")
        if not isinstance(op_name, str):
            raise ValueError("PTOAS operation node has no string op_name")
        operation = node.get("operation")
        if not isinstance(operation, Mapping):
            return self._unsupported(op_name, "missing PTOAS operation metadata")
        operand_types = operation.get("operand_types")
        result_types = operation.get("result_types", [])
        if (
            not isinstance(operand_types, list)
            or not all(isinstance(item, str) for item in operand_types)
            or not isinstance(result_types, list)
            or not all(isinstance(item, str) for item in result_types)
        ):
            return self._unsupported(op_name, "missing PTOAS operand types")
        tiles = [
            tile for item in [*operand_types, *result_types] if (tile := parse_tile_type(item)) is not None
        ]

        if op_name in _SCALAR_STAGE_OPS:
            return self._estimate_scalar_stage(node, op_name, operand_types, result_types)
        if op_name in _FORMULA_OPCODE:
            if op_name == "pto.trecip":
                if _INSTRUCTION_LOWERING_RELATIVE_PATH.as_posix() not in self.source_sha256:
                    return self._unsupported(op_name, "snapshot lacks pinned TRECIP-to-TDIVS lowering")
                if operation.get("attributes", {}).get("precision_type") not in {None, "default", 0}:
                    return self._unsupported(
                        op_name, "non-default reciprocal needs mode-specific calibration"
                    )
            estimate = self._estimate_formula_operation(op_name, tiles)
            if estimate is not None:
                if op_name in {"pto.trowexpandadd", "pto.trowexpandsub"}:
                    return DurationEstimate(
                        estimate.cycles,
                        "pto_isa_formula_family",
                        f"{estimate.detail}; fused_op={op_name}; family=TROWEXPAND",
                        "pinned_formula_family_approximation",
                    )
                return estimate
        if op_name == "pto.tmaxs":
            return self._estimate_calibrated_tmaxs(tiles)
        if op_name in _MATMUL_OPS:
            return self._estimate_matmul(op_name, tiles)
        if op_name in _TRANSFER_OPS:
            return self._estimate_transfer(op_name, tiles)
        if op_name in _PERF_SIM_DEFAULT_OPS:
            return self._estimate_perf_sim_default(op_name, operand_types, result_types)
        return self._unsupported(op_name, "operation is not supported by PTO-ISA lightweight A2/A3 model")

    def estimate_pipeline(self, node: Mapping[str, Any]) -> PipelineEstimate | None:
        """Return a pinned stream-start/tail split when PTO-ISA defines one.

        This mirrors the A2/A3 CCE mock's queue contract rather than deriving a
        per-pipe constant. The matmul model has an explicit six-cycle head, and
        the deterministic Perf-Sim fallback has an explicit one- or two-cycle
        head. Fitted formula bias and transfer bandwidth estimate inclusive
        cost but do not expose a startup/pending-tail split, so both fail closed
        here unless an exact-signature override is supplied by the caller.
        """
        op_name = node.get("op_name")
        operation = node.get("operation")
        if not isinstance(op_name, str) or not isinstance(operation, Mapping):
            return None
        operand_types = operation.get("operand_types")
        result_types = operation.get("result_types", [])
        if (
            not isinstance(operand_types, list)
            or not all(isinstance(item, str) for item in operand_types)
            or not isinstance(result_types, list)
            or not all(isinstance(item, str) for item in result_types)
        ):
            return None
        tiles = [
            tile for item in [*operand_types, *result_types] if (tile := parse_tile_type(item)) is not None
        ]

        if op_name in _MATMUL_OPS:
            lhs = next((tile for tile in tiles if tile.scope == "left"), None)
            rhs = next((tile for tile in tiles if tile.scope == "right"), None)
            if lhs is not None and rhs is not None and lhs.cols == rhs.rows:
                return PipelineEstimate(
                    startup_cycles=6.0,
                    pending_tail_cycles=0.0,
                    source="pto_isa_matmul_head",
                    detail="A2/A3 lightweight matmul model uses an explicit six-cycle head",
                )
        if op_name in _PERF_SIM_DEFAULT_OPS:
            if op_name in _FORMULA_OPCODE and tiles:
                tile = tiles[0]
                if self._lookup_formula(_FORMULA_OPCODE[op_name], tile.dtype, tile.cols) is not None:
                    # The fitted formula supplies only an inclusive duration;
                    # it does not expose the Perf-Sim fallback's head/tail
                    # decomposition.
                    return None
            result_tiles = [tile for item in result_types if (tile := parse_tile_type(item)) is not None]
            operand_tiles = [tile for item in operand_types if (tile := parse_tile_type(item)) is not None]
            if op_name in _PERF_SIM_SOURCE_WORK_OPS:
                work_tile = operand_tiles[0] if operand_tiles else None
            else:
                work_tile = result_tiles[0] if result_tiles else (operand_tiles[0] if operand_tiles else None)
            if work_tile is not None:
                startup = 1.0 if op_name == "pto.ttrans" else 2.0
                return PipelineEstimate(
                    startup_cycles=startup,
                    pending_tail_cycles=0.0,
                    source="pto_isa_perf_sim_default_head",
                    detail=f"{op_name} deterministic fallback head={startup}; vector tail=0",
                )
        if op_name in _SCALAR_STAGE_OPS:
            return PipelineEstimate(
                startup_cycles=1.0,
                pending_tail_cycles=0.0,
                source="pto_isa_perf_sim_scalar_stage",
                detail="scalar stage fallback is one cycle with no deferred tail",
            )
        return None

    def _estimate_scalar_stage(
        self,
        node: Mapping[str, Any],
        op_name: str,
        operand_types: list[str],
        result_types: list[str],
    ) -> DurationEstimate:
        pipe = node.get("pipe")
        if op_name in {"pto.load_scalar", "pto.store_scalar"}:
            pointer_types = [item for item in operand_types if item.startswith("!pto.ptr<")]
            scalar_types = (
                [item for item in result_types if _SCALAR_TYPE_FULL_RE.fullmatch(item)]
                if op_name == "pto.load_scalar"
                else [item for item in operand_types if _SCALAR_TYPE_FULL_RE.fullmatch(item)]
            )
            if len(pointer_types) != 1 or len(scalar_types) != 1:
                contract = (
                    "scalar load lacks one pointer and one scalar result"
                    if op_name == "pto.load_scalar"
                    else "scalar store lacks one pointer and one scalar operand"
                )
                return self._unsupported(op_name, contract)
        elif pipe not in {"PIPE_S", "PIPE_FIX", "PIPE_MTE2"}:
            return self._unsupported(op_name, f"unexpected mixed-kernel pipe {pipe}")
        return DurationEstimate(
            1.0,
            "pto_isa_perf_sim_scalar_stage",
            "Perf-Sim StaticPipeStageLookup classifies scalar operations and "
            "FallbackCycles assigns one cycle",
            "pinned_perf_sim_approximation",
        )

    def _estimate_formula_operation(self, op_name: str, tiles: list[TileType]) -> DurationEstimate | None:
        if not tiles:
            return self._unsupported(op_name, "formula operation has no static tile type")
        tile = tiles[0]
        opcode = _FORMULA_OPCODE[op_name]
        parameter = self._lookup_formula(opcode, tile.dtype, tile.cols)
        if parameter is None:
            if op_name in _NEAREST_FORMULA_SHAPE_OPS:
                return self._estimate_from_nearest_formula(op_name, opcode, tile)
            if op_name in _PERF_SIM_DEFAULT_OPS:
                return None
            return self._unsupported(
                op_name,
                f"no PTO-ISA formula for dtype={tile.dtype}, cols={tile.cols}",
            )
        cycles = _round_to_cycles(parameter.slope * tile.rows * tile.cols + parameter.bias)
        return DurationEstimate(
            float(cycles),
            "pto_isa_formula",
            f"{opcode}:{tile.dtype}:{tile.rows}x{tile.cols}; slope={parameter.slope}; bias={parameter.bias}",
            "calibrated_formula_signature",
        )

    def _estimate_from_nearest_formula(self, op_name: str, opcode: str, tile: TileType) -> DurationEstimate:
        """Extrapolate an absent shape from the nearest pinned formula fit.

        In particular, A2/A3 lowers reciprocal to ``TDIVS(dst, 1, src)`` and
        the pinned TDIVS table begins at 32 columns, while real RMS kernels use
        8- and 16-column fp32 tiles. Reusing a same-dtype fit is evidence-backed
        but explicitly remains an approximation, never an exact signature.
        """
        candidates = [
            parameter
            for parameter in self.formula_parameters
            if parameter.op == opcode and parameter.dtype == tile.dtype and parameter.cols is not None
        ]
        if not candidates:
            return self._unsupported(op_name, f"no measured {opcode} formula family for dtype={tile.dtype}")
        nearest = min(
            candidates,
            key=lambda parameter: (abs(int(parameter.cols) - tile.cols), parameter.cols),
        )
        cycles = _round_to_cycles(nearest.slope * tile.rows * tile.cols + nearest.bias)
        return DurationEstimate(
            float(cycles),
            "pto_isa_formula_nearest_shape",
            f"{op_name} lowers to {opcode}; requested={tile.dtype}:{tile.rows}x{tile.cols}; "
            f"nearest_calibrated_cols={nearest.cols}; slope={nearest.slope}; bias={nearest.bias}",
            "pinned_formula_shape_approximation",
        )

    def _estimate_calibrated_tmaxs(self, tiles: list[TileType]) -> DurationEstimate:
        """Apply the pinned A2/A3 CCE TMAXS calibration for fp32 vector tiles."""
        if _CCE_VECTOR_COMPUTE_RELATIVE_PATH.as_posix() not in self.source_sha256:
            return self._unsupported("pto.tmaxs", "snapshot lacks pinned A2/A3 CCE calibration source")
        tile = tiles[0] if tiles else None
        if tile is None or tile.scope != "vec" or tile.dtype != "fp32":
            return self._unsupported("pto.tmaxs", "calibrated TMAXS requires one fp32 vector tile")
        repeat_elements = 256 // _DTYPE_BYTES[tile.dtype]
        repeats = _ceil_div(tile.rows * tile.cols, repeat_elements)
        return DurationEstimate(
            float(23 + repeats),
            "pto_isa_a2a3_cce_vmaxs",
            f"TMAXS:{tile.dtype}:{tile.rows}x{tile.cols}; repeat_elements={repeat_elements}; "
            f"repeats={repeats}; calibrated=repeat+23",
            "calibrated_instruction_model",
        )

    def _estimate_matmul(self, op_name: str, tiles: list[TileType]) -> DurationEstimate:
        lhs = next((tile for tile in tiles if tile.scope == "left"), None)
        rhs = next((tile for tile in tiles if tile.scope == "right"), None)
        if lhs is None or rhs is None:
            return self._unsupported(op_name, "matmul has fewer than two static tile operands")
        if lhs.cols != rhs.rows or lhs.dtype not in {"bf16", "fp16", "fp32"}:
            return self._unsupported(
                op_name,
                f"unsupported matmul types {lhs.rows}x{lhs.cols}x{lhs.dtype}, "
                f"{rhs.rows}x{rhs.cols}x{rhs.dtype}",
            )
        repeats = (
            _ceil_div(lhs.rows, 16)
            * _ceil_div(lhs.cols, 32 // _DTYPE_BYTES[lhs.dtype])
            * _ceil_div(rhs.cols, 16)
        )
        cycles = 6 + (2 if lhs.dtype == "fp32" else 1) * repeats
        return DurationEstimate(
            float(cycles),
            "pto_isa_matmul_formula",
            f"m={lhs.rows}; k={lhs.cols}; n={rhs.cols}; dtype={lhs.dtype}; repeats={repeats}",
            "pinned_analytical_model",
        )

    def _estimate_transfer(self, op_name: str, tiles: list[TileType]) -> DurationEstimate:
        if not tiles:
            return self._unsupported(op_name, "transfer operation has no static tile type")
        tile = tiles[-1] if op_name == "pto.tload" else tiles[0]
        tile_type = _TILE_TO_SCOPE.get(tile.scope)
        route = _TRANSFER_ROUTE.get((_TRANSFER_OPS[op_name], tile_type or ""))
        bandwidth = self.bandwidth_gib_per_s.get(route or "")
        element_bytes = _DTYPE_BYTES.get(tile.dtype)
        transfer_bytes = tile.rows * tile.cols * element_bytes if element_bytes is not None else 0
        if route is None or bandwidth is None or bandwidth <= 0 or transfer_bytes <= 0:
            return self._unsupported(
                op_name,
                f"unsupported transfer tile={tile_type}, bytes={transfer_bytes}, route={route}",
            )
        cycles = math.floor((transfer_bytes / (1024**3)) / bandwidth * self.frequency_hz)
        return DurationEstimate(
            float(cycles),
            "pto_isa_bandwidth",
            f"route={route}; bytes={transfer_bytes}; bandwidth_gib_per_s={bandwidth}; "
            f"frequency_hz={self.frequency_hz}",
            "pinned_analytical_model",
        )

    def _estimate_perf_sim_default(
        self, op_name: str, operand_types: list[str], result_types: list[str]
    ) -> DurationEstimate:
        """Mirror pinned Perf-Sim's deterministic EstimateInstrCycles fallback."""
        result_tiles = [tile for item in result_types if (tile := parse_tile_type(item)) is not None]
        operand_tiles = [tile for item in operand_types if (tile := parse_tile_type(item)) is not None]
        if op_name in _PERF_SIM_SOURCE_WORK_OPS:
            work_tile = operand_tiles[0] if operand_tiles else None
        else:
            work_tile = result_tiles[0] if result_tiles else (operand_tiles[0] if operand_tiles else None)
        if work_tile is None:
            return self._unsupported(op_name, "Perf-Sim default operation has no static work tile")
        elements = work_tile.rows * work_tile.cols
        if op_name == "pto.trsqrt":
            high_precision_tiles = (
                [*operand_tiles, *result_tiles]
                if len(operand_tiles) == 2 and len(result_tiles) == 1
                else operand_tiles
                if len(operand_tiles) == 3 and not result_tiles
                else []
            )
            if high_precision_tiles:
                return self._unsupported(
                    op_name,
                    "high-precision TRSQRT requires a composite exact-signature calibration",
                )
            if len(result_tiles) != 1 or len(operand_tiles) != 1:
                return self._unsupported(op_name, "unrecognized TRSQRT operand contract")
            if work_tile.scope != "vec" or work_tile.dtype != "fp32":
                return self._unsupported(
                    op_name,
                    f"unsupported A2/A3 vrsqrt tile scope={work_tile.scope}, dtype={work_tile.dtype}",
                )
            if _CCE_VECTOR_COMPUTE_RELATIVE_PATH.as_posix() not in self.source_sha256:
                return self._unsupported(op_name, "A2/A3 vrsqrt source is absent from the provider snapshot")
            repeat_elements = 256 // _DTYPE_BYTES[work_tile.dtype]
            repeats = _ceil_div(elements, repeat_elements)
            return DurationEstimate(
                float(24 + repeats),
                "pto_isa_a2a3_cce_vrsqrt",
                f"TRSQRT:{work_tile.dtype}:{work_tile.rows}x{work_tile.cols}; "
                f"repeat_elements={repeat_elements}; repeats={repeats}; calibrated=repeat+24",
                "calibrated_instruction_model",
            )
        if op_name == "pto.ttrans":
            cycles = 1 + elements // 64
            stage = "MTE1"
        else:
            cycles = 2 + elements // 32
            stage = "default"
        return DurationEstimate(
            float(cycles),
            "pto_isa_perf_sim_default",
            f"{op_name}:{work_tile.dtype}:{work_tile.rows}x{work_tile.cols}; "
            f"stage={stage}; pinned EstimateInstrCycles fallback",
            "pinned_perf_sim_approximation",
        )

    def estimate_formula(self, op: str, dtype: str, rows: int, cols: int) -> DurationEstimate | None:
        """Estimate one formula opcode for direct Perf-Sim validation."""
        parameter = self._lookup_formula(op, dtype, cols)
        if parameter is None or rows <= 0 or cols <= 0:
            return None
        cycles = _round_to_cycles(parameter.slope * rows * cols + parameter.bias)
        return DurationEstimate(
            float(cycles),
            "pto_isa_formula",
            f"{op}:{dtype}:{rows}x{cols}; slope={parameter.slope}; bias={parameter.bias}",
            "calibrated_formula_signature",
        )

    def _lookup_formula(self, op: str, dtype: str, cols: int) -> FormulaParameter | None:
        candidates: list[tuple[str, int | None]] = [(dtype, cols)]
        if op in _ANY_PARAMETER_OPS:
            candidates.extend(((dtype, None), ("any", cols), ("any", None)))
        for candidate_dtype, candidate_cols in candidates:
            for parameter in self.formula_parameters:
                if (
                    parameter.op == op
                    and parameter.dtype == candidate_dtype
                    and parameter.cols == candidate_cols
                ):
                    return parameter
        return None

    def _unsupported(self, op_name: str, reason: str) -> DurationEstimate:
        if self.unsupported_policy == "error":
            raise ValueError(f"unsupported PTO-ISA duration for {op_name}: {reason}")
        return DurationEstimate(
            self.fallback_cycles,
            "unsupported_fallback",
            reason,
            "unsupported_fallback",
            fallback=True,
        )


def _validate_policy(policy: str, fallback_cycles: float) -> None:
    if policy not in {"error", "fallback"}:
        raise ValueError(f"unsupported-operation policy must be 'error' or 'fallback', got {policy!r}")
    if not math.isfinite(fallback_cycles) or fallback_cycles <= 0:
        raise ValueError(f"fallback cycles must be finite and positive, got {fallback_cycles}")


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"cannot inspect PTO-ISA checkout {root}: git {' '.join(args)}: {detail}")
    return result.stdout.strip()


def _parse_arch_config(text: str) -> tuple[float, dict[str, float]]:
    frequency_match = _FREQUENCY_RE.search(text)
    arch_match = _ARCH_CONFIG_RE.search(text)
    if frequency_match is None or arch_match is None:
        raise ValueError("cannot parse kMainFrequencyHz or kA2A3ArchConfig from PTO-ISA arch_config.hpp")
    values = [float(item) for item in _NUMBER_RE.findall(arch_match.group("body"))]
    if len(values) != len(_BANDWIDTH_NAMES):
        raise ValueError(f"expected {len(_BANDWIDTH_NAMES)} A2/A3 bandwidth entries, found {len(values)}")
    return float(frequency_match.group(1)), dict(zip(_BANDWIDTH_NAMES, values, strict=True))


def _parse_formula_parameters(path: Path) -> list[FormulaParameter]:
    parameters: list[FormulaParameter] = []
    with path.open(newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                cols = None if row["cols"] == "*" else int(row["cols"])
                parameter = FormulaParameter(
                    op=row["op"],
                    dtype=row["dtype"],
                    cols=cols,
                    slope=float(row["slope"]),
                    bias=float(row["bias"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid formula row {row}") from error
            if parameter.cols is not None and not 0 <= parameter.cols <= 65535:
                raise ValueError(f"{path}:{line_number}: cols exceed PTO-ISA uint16 range")
            parameters.append(parameter)
    if not parameters:
        raise ValueError(f"{path}: formula table is empty")
    return parameters


def parse_tile_type(value: str) -> TileType | None:
    """Parse a static PTO tile type for duration-signature construction."""
    if match := _KEYED_TILE_RE.search(value):
        dtype = _PTO_DTYPE.get(match.group("dtype").lower())
        if dtype is None:
            return None
        return TileType(
            scope=match.group("scope").lower(),
            rows=int(match.group("rows")),
            cols=int(match.group("cols")),
            dtype=dtype,
        )
    match = _LEGACY_TILE_RE.search(value)
    if match is None:
        return None
    shape_match = _SHAPE_DTYPE_RE.fullmatch(match.group("shape"))
    if shape_match is None:
        return None
    dimensions = [int(item) for item in shape_match.group("shape").split("x")]
    dtype = _PTO_DTYPE.get(shape_match.group("dtype").lower())
    if dtype is None or not dimensions:
        return None
    rows = math.prod(dimensions[:-1]) if len(dimensions) > 1 else 1
    return TileType(
        scope=match.group("scope").lower(),
        rows=rows,
        cols=dimensions[-1],
        dtype=dtype,
    )


def static_type_size_bytes(value: str) -> int | None:
    """Return the byte extent of one statically shaped PTO tile or partition type."""
    if tile := parse_tile_type(value):
        element_bytes = _DTYPE_BYTES.get(tile.dtype)
        return tile.rows * tile.cols * element_bytes if element_bytes is not None else None
    match = _PARTITION_TYPE_RE.search(value)
    if match is None:
        return None
    dtype = _PTO_DTYPE.get(match.group("dtype").lower())
    element_bytes = _DTYPE_BYTES.get(dtype or "")
    if element_bytes is None:
        return None
    dimensions = [int(item) for item in match.group("shape").split("x")]
    return math.prod(dimensions) * element_bytes


def _round_to_cycles(value: float) -> int:
    return 0 if value <= 0 else math.floor(value + 0.5)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def provider_snapshot_sha256(provider: PtoIsaDurationProvider) -> str:
    """Return the stable identity of all duration inputs and policies."""
    payload = json.dumps(provider.to_json(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
