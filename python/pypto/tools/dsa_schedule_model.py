# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Score PTOAS schedule graphs with an explicitly provisional duration model.

PTOAS's ``--pto-insert-sync-schedule-graph`` diagnostic emits the operation
streams and synchronization dependencies that InsertSync actually analyzed.
This module assigns durations to those operations and evaluates the resulting
directed acyclic graph.  It deliberately keeps three concepts separate:

* the per-pipe stream graph, which is the no-cross-pipe-sync baseline;
* singleton synchronization exposure, measured against that baseline; and
* the makespan of the complete synchronization set, which captures edge
  interactions and coalescence.

Version 0 aggregates the work of statically bounded loops for whole-function
DAG scores and models structured if/else regions as mutually exclusive
per-pipe paths. Candidate scoring additionally evaluates distance-one edges
with a version-1 loop initiation-interval lower bound: the maximum of per-pipe
work and every supported recurrence cycle. The experimental queue/event model
instead expands static loop occurrences, preserves per-pipe FIFO order, and
prices calibrated pipeline breaks at emitted barriers. Because invocation
branch outcomes are not exported, it reports all-then/all-else extremes rather
than claiming an input-specific path. This remains a structural model, not a
cycle-accurate prediction. Active Final-SyncIR records and hypothetical
candidate synchronization endpoints are reported separately: a redundant
precedence edge can have zero DAG extension while still creating synchronization
pressure.  These are explicitly pre-codegen quantities, not counts of emitted
instructions.  Scores also disclose every loop-carried, loop-marker, or omitted
barrier dependency that the collapsed operation-only DAG cannot represent.

The complete-placement model is independent of InsertSync. It rebuilds the
non-reusing base DAG from logical-root RAW/WAR/WAW dependencies and fixed pipe
order, adds every physical-placement reuse hazard with a calibrated positive
synchronization latency, and scores the union with one longest-path calculation.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
import re
import statistics
import struct
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from pypto.tools.dsa_pto_isa_duration import (
    PtoIsaDurationProvider,
    parse_tile_type,
    provider_snapshot_sha256,
    static_type_size_bytes,
)
from pypto.tools.dsa_reuse_candidates import ReuseCandidateRecord, load_candidate_records

_PIPE_ALIASES = {
    "SCALAR": "PIPE_S",
    "VECTOR": "PIPE_V",
    "VEC": "PIPE_V",
    "CUBE": "PIPE_M",
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "FIXPIPE": "PIPE_FIX",
}

_DEBUG_NODE_RE = re.compile(r"^\s*\[\s*(\d+)\]\s+COMPOUND\s+(\S+)\s+\[(\S+)\]")
_DEBUG_LOOP_RE = re.compile(r"^\s*\[\s*(\d+)\]\s+LOOP\s+(LOOP_BEGIN|LOOP_END)\s+\(begin=(\d+),\s*end=(\d+)\)")
_DEBUG_BRANCH_RE = re.compile(
    r"^\s*\[\s*(\d+)\]\s+BRANCH\s+(IF_BEGIN|ELSE_BEGIN|IF_END)\s+"
    r"\(begin=(\d+),\s*branch=(\d+),\s*end=(\d+)\)"
)
_DEBUG_PLACEHOLDER_RE = re.compile(r"^\s*\[\s*(\d+)\]\s+PLACE_HOLDER\s+\(parentScopeId=(\d+)\)")
_DEBUG_MEM_RE = re.compile(r"^\s*(def|use)=\[(.*)\]\s*$")
_DEBUG_MEM_ITEM_RE = re.compile(r"(%[^,(\s]+)\(([^)]+)\)")
_DEBUG_SYNC_RE = re.compile(
    r"^\s*(PRE|POST)\s*:?[ ]*(\w+)\s+<(\S+)\s+->\s+(\S+)>\s+idx=(\d+)"
    r"(.*)$"
)
_DEBUG_EVENT_IDS_RE = re.compile(r"\beventIds=\[([^]]*)\]")
_DEBUG_DEPENDENCY_NODE_RE = re.compile(r"\bdepNode=(\d+)\b")
_DEBUG_FOR_END_RE = re.compile(r"\bforEnd=(\d+)\b")
_ACCESS_LOCATION_RE = re.compile(r"pypto\.access\.(\d+)")
_PTO_SOURCE_LOOP_RE = re.compile(r"\bpypto\.source_loop\.(\d+)\b")
_PTO_OPERATION_RE = re.compile(r"(?<![!\w.])(pto\.[A-Za-z0-9_]+)\b")
_PTO_CONSTANT_RE = re.compile(
    r"^\s*(%[-A-Za-z0-9_.$]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(?:index|i\d+)\s*$"
)
_PTO_SCALAR_CONSTANT_RE = re.compile(
    r"^\s*(%[-A-Za-z0-9_.$]+)\s*=\s*arith\.constant\s+(.+?)\s*:\s*"
    r"(bf16|f[A-Za-z0-9_]+|index|(?:ui|i)\d+)\s*(?:loc\(.*\))?$"
)
_PTO_SSA_VALUE_RE = re.compile(r"%[-A-Za-z0-9_.$]+")
_PTO_FOR_RE = re.compile(
    r"^\s*(?:%[-A-Za-z0-9_.$]+(?:#\d+)?\s*=\s*)?scf\.for\s+"
    r"(%[-A-Za-z0-9_.$]+)\s*=\s*(%[-A-Za-z0-9_.$]+|-?\d+)\s+to\s+"
    r"(%[-A-Za-z0-9_.$]+|-?\d+)\s+step\s+(%[-A-Za-z0-9_.$]+|-?\d+)\b"
)
_PTO_IF_RE = re.compile(r"^\s*(?:(%[-A-Za-z0-9_.$]+)\s*=\s*)?scf\.if\s+(%[-A-Za-z0-9_.$]+)\b")
_PTO_ASSIGN_RE = re.compile(r"^\s*(%[-A-Za-z0-9_.$]+)\s*=\s*(.+)$")
_PTO_YIELD_RE = re.compile(r"^\s*scf\.yield\s+(%[-A-Za-z0-9_.$]+|-?\d+)\b")
_PTO_BRANCH_RE = re.compile(r"\b(?:scf\.(?:if|while|index_switch)|cf\.(?:br|cond_br|switch))\b")
_PTO_TYPE_START_RE = re.compile(r"!pto\.[A-Za-z_][A-Za-z0-9_.]*<")
_PTO_SCALAR_TYPE_RE = re.compile(
    r"(?<![A-Za-z0-9_=%])(?:bf16|f16|f32|ui8|ui16|ui32|ui64|i8|i16|i32|i64|index)\b"
)
_PTO_SCALAR_TYPE_FULL_RE = re.compile(r"(?:bf16|f\d+|(?:ui|i)\d+|index)")
_PTO_ACC_TILE_RE = re.compile(r"!pto\.tile_buf<(?:[^>]*\bloc=acc\b|\s*acc\s*,)")

# This is deliberately narrower than the recognizer's route vocabulary. The
# first structured model targets the route families already exercised by the
# runnable vector-kernel cohort; unfamiliar routes fail closed rather than
# acquiring a guessed pipeline.
_RESOURCE_PIPE = {
    "inbound_dma": "PIPE_MTE2",
    "outbound_dma": "PIPE_MTE3",
    "l0_to_external": "PIPE_FIX",
    "l1_to_l0": "PIPE_MTE1",
    "vector_compute": "PIPE_V",
    "matrix_compute": "PIPE_M",
    "scalar_access": "PIPE_S",
}

# These exceptions describe an observed product-schedule contract, not a
# guessed execution cost. PyPTO classifies TCI as vector work, and the A2/A3
# Perf-Sim implementation charges it to the vector model, while PTOAS v0.57
# places the corresponding schedule node on PIPE_S. Candidate edges must bind
# to the pipe that InsertSync actually schedules. Keep the exception narrow so
# every other route/pipe disagreement remains a hard error.
_ROUTE_PIPE_JOIN_EXCEPTIONS = {
    ("PIPE_V", "PIPE_S", "pto.tci"): "ptoas_v057_tci_schedule_pipe",
    ("PIPE_V", "PIPE_S", "pto.tsetval"): "ptoas_v057_tsetval_schedule_pipe",
}

_MAX_QUEUE_EVENT_EXPANDED_NODES = 250_000


@dataclass(frozen=True)
class PipeParameters:
    """Primitive version-0 duration parameters for one execution pipe."""

    startup_cycles: float
    bytes_per_cycle: float
    minimum_cycles: float


@dataclass(frozen=True)
class PipelineComponents:
    """Operation-specific synchronization-boundary pipeline state."""

    startup_cycles: float
    pending_tail_cycles: float


def _default_pipe_parameters() -> dict[str, PipeParameters]:
    # Generic pipe constants silently gave every unknown operation a plausible
    # duration.  A production prediction must instead use the pinned PTO-ISA
    # provider or an explicit calibrated override.
    return {}


@dataclass
class DurationModel:
    """Duration inputs used to score a schedule graph."""

    schema_version: int = 1
    model_version: str = "duration_v1"
    calibration_status: str = "unconfigured"
    sync_latency_cycles: float = 0.0
    barrier_instruction_cycles: float = 1.0
    pipe_barrier_cycles: dict[str, float] = field(default_factory=dict)
    pipe_parameters: dict[str, PipeParameters] = field(default_factory=_default_pipe_parameters)
    operation_cycles: dict[str, float] = field(default_factory=dict)
    operation_signature_cycles: dict[str, float] = field(default_factory=dict)
    operation_signature_pipeline: dict[str, PipelineComponents] = field(default_factory=dict)
    calibration_sources: list[str] = field(default_factory=list)
    pto_isa_provider: PtoIsaDurationProvider | None = None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "DurationModel":
        """Construct a model from a JSON-compatible mapping."""
        if value.get("schema_version") != 1:
            raise ValueError("duration model must have schema_version=1")
        raw_pipes = value.get("pipe_parameters")
        if not isinstance(raw_pipes, Mapping):
            raise ValueError("duration model is missing pipe_parameters")
        pipes: dict[str, PipeParameters] = {}
        for name, raw in raw_pipes.items():
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                raise ValueError("invalid pipe_parameters entry")
            pipes[name] = PipeParameters(
                startup_cycles=float(raw["startup_cycles"]),
                bytes_per_cycle=float(raw["bytes_per_cycle"]),
                minimum_cycles=float(raw["minimum_cycles"]),
            )
        raw_ops = value.get("operation_cycles", {})
        if not isinstance(raw_ops, Mapping):
            raise ValueError("operation_cycles must be an object")
        raw_signatures = value.get("operation_signature_cycles", {})
        if not isinstance(raw_signatures, Mapping):
            raise ValueError("operation_signature_cycles must be an object")
        raw_barriers = value.get("pipe_barrier_cycles", {})
        if not isinstance(raw_barriers, Mapping):
            raise ValueError("pipe_barrier_cycles must be an object")
        barrier_cycles = {str(pipe): float(cycles) for pipe, cycles in raw_barriers.items()}
        if any(not math.isfinite(cycles) or cycles < 0 for cycles in barrier_cycles.values()):
            raise ValueError("pipe_barrier_cycles must contain finite non-negative values")
        barrier_instruction_cycles = float(value.get("barrier_instruction_cycles", 1.0))
        if not math.isfinite(barrier_instruction_cycles) or barrier_instruction_cycles < 0:
            raise ValueError("barrier_instruction_cycles must be finite and non-negative")
        raw_pipeline = value.get("operation_signature_pipeline", {})
        if not isinstance(raw_pipeline, Mapping):
            raise ValueError("operation_signature_pipeline must be an object")
        pipeline: dict[str, PipelineComponents] = {}
        for signature, raw in raw_pipeline.items():
            if not isinstance(signature, str) or not isinstance(raw, Mapping):
                raise ValueError("invalid operation_signature_pipeline entry")
            components = PipelineComponents(
                startup_cycles=float(raw["startup_cycles"]),
                pending_tail_cycles=float(raw["pending_tail_cycles"]),
            )
            if any(
                not math.isfinite(item) or item < 0
                for item in (components.startup_cycles, components.pending_tail_cycles)
            ):
                raise ValueError("operation_signature_pipeline cycles must be finite and non-negative")
            pipeline[signature] = components
        raw_provider = value.get("pto_isa_provider")
        if raw_provider is not None and not isinstance(raw_provider, Mapping):
            raise ValueError("pto_isa_provider must be an object")
        return cls(
            schema_version=1,
            model_version=str(value.get("model_version", "duration_v1")),
            calibration_status=str(value.get("calibration_status", "unknown")),
            sync_latency_cycles=float(value.get("sync_latency_cycles", 0.0)),
            barrier_instruction_cycles=barrier_instruction_cycles,
            pipe_barrier_cycles=barrier_cycles,
            pipe_parameters=pipes,
            operation_cycles={str(key): float(cycles) for key, cycles in raw_ops.items()},
            operation_signature_cycles={str(key): float(cycles) for key, cycles in raw_signatures.items()},
            operation_signature_pipeline=pipeline,
            calibration_sources=[str(path) for path in value.get("calibration_sources", [])],
            pto_isa_provider=(
                PtoIsaDurationProvider.from_json(raw_provider) if raw_provider is not None else None
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        value = asdict(self)
        value["pto_isa_provider"] = (
            self.pto_isa_provider.to_json() if self.pto_isa_provider is not None else None
        )
        for parameters in value["pipe_parameters"].values():
            if math.isinf(parameters["bytes_per_cycle"]):
                parameters["bytes_per_cycle"] = "inf"
        return value


def load_schedule_graphs(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load schedule-graph JSONL keyed by function, rejecting ambiguity."""
    source = Path(path)
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError(f"{source}:{line_number}: expected schedule graph schema_version=1")
        function = record.get("function")
        if not isinstance(function, str) or not function:
            raise ValueError(f"{source}:{line_number}: missing non-empty function")
        if function in records:
            raise ValueError(f"{source}:{line_number}: duplicate function '{function}'")
        records[function] = record
    if not records:
        raise ValueError(f"{source}: no schedule graph records")
    return records


def _extract_pto_types(text: str) -> list[str]:
    """Extract PTO and scalar types in textual order without splitting nested types."""
    types: list[tuple[int, str]] = []
    cursor = 0
    while match := _PTO_TYPE_START_RE.search(text, cursor):
        depth = 1
        end = match.end()
        while end < len(text) and depth:
            if text[end] == "<":
                depth += 1
            elif text[end] == ">":
                depth -= 1
            end += 1
        if depth:
            raise ValueError(f"unterminated PTO type in operation: {text}")
        types.append((match.start(), text[match.start() : end]))
        cursor = end

    composite_ranges = [(offset, offset + len(value)) for offset, value in types]
    for match in _PTO_SCALAR_TYPE_RE.finditer(text):
        if not any(begin <= match.start() < end for begin, end in composite_ranges):
            types.append((match.start(), match.group(0)))
    return [value for _, value in sorted(types)]


def _extract_operation_region(line: str, name: str) -> str | None:
    marker = f"{name}("
    start = line.find(marker)
    if start < 0:
        return None
    cursor = start + len(marker)
    depth = 1
    while cursor < len(line) and depth:
        if line[cursor] == "(":
            depth += 1
        elif line[cursor] == ")":
            depth -= 1
        cursor += 1
    if depth:
        raise ValueError(f"unterminated {name}(...) region in raw PTO operation: {line}")
    return line[start + len(marker) : cursor - 1]


def _operation_operand_names(line: str) -> list[str]:
    """Return textual operands in operation order, excluding SSA results."""
    input_region = _extract_operation_region(line, "ins")
    if input_region is not None:
        values, _, _ = input_region.partition(":")
        return _PTO_SSA_VALUE_RE.findall(values)

    operation_match = _PTO_OPERATION_RE.search(line)
    if operation_match is None:
        return []
    operands = line[operation_match.end() :].split(":", maxsplit=1)[0]
    return _PTO_SSA_VALUE_RE.findall(operands)


def _operation_attributes(line: str) -> dict[str, Any]:
    """Extract static operation attributes that affect instruction cost."""
    attributes: dict[str, Any] = {}
    if match := re.search(r"\brmode\s*=\s*#pto<round_mode\s+([A-Z_]+)>", line):
        attributes["round_mode"] = match.group(1)
    for name in ("descending", "exhausted"):
        if match := re.search(rf"\b{name}\s*=\s*(true|false)\b", line):
            attributes[name] = match.group(1) == "true"
    return attributes


def _operation_type_metadata(line: str, constants: Mapping[str, str]) -> dict[str, Any]:
    input_region = _extract_operation_region(line, "ins")
    output_region = _extract_operation_region(line, "outs")
    if input_region is None and output_region is None:
        operation_text = line.split(" loc(", maxsplit=1)[0]
        before_result, separator, after_result = operation_text.partition(" -> ")
        operand_types = _extract_pto_types(before_result)
        result_types = _extract_pto_types(after_result) if separator else []
    else:
        operand_types = _extract_pto_types(input_region or "")
        if output_region is not None:
            result_types = _extract_pto_types(output_region)
        elif match := re.search(r"\bouts\s*:\s*(.*?)\s+loc\(", line):
            result_types = _extract_pto_types(match.group(1))
        else:
            result_types = []
    if "pto.load_scalar" in line:
        legacy_pointer = re.search(r":\s*<\s*([^,>]+)\s*,\s*([^>]+)>\s*->", line)
        if legacy_pointer is not None:
            dtype, scope = (part.strip() for part in legacy_pointer.groups())
            operand_types = [f"!pto.ptr<{dtype}, {scope}>"]
    static_sizes = [
        size for item in [*operand_types, *result_types] if (size := static_type_size_bytes(item)) is not None
    ]
    return {
        "operand_types": operand_types,
        "operand_constants": [constants.get(name) for name in _operation_operand_names(line)],
        "result_types": result_types,
        "attributes": _operation_attributes(line),
        "static_work_bytes": max(static_sizes, default=0),
    }


def _join_operation_name(name: str) -> str:
    aliases = {
        "pto.tmatmul.acc": "pto.tmatmul",
        "pto.tpush_to_aic": "pto.tpush",
        "pto.tpush_to_aiv": "pto.tpush",
        "pto.tpop_from_aic": "pto.tpop",
        "pto.tpop_from_aiv": "pto.tpop",
        "pto.tfree_from_aic": "pto.tfree",
        "pto.tfree_from_aiv": "pto.tfree",
    }
    return aliases.get(name, name)


def _operation_names_match(expected: str, actual: str, metadata: Mapping[str, Any]) -> bool:
    if expected == actual:
        return True
    if expected in {"pto.tpush", "pto.tpop", "pto.tfree"} and _join_operation_name(actual) == expected:
        return True
    if expected != "pto.tmatmul.acc" or actual != "pto.tmatmul":
        return False
    operand_types = metadata.get("operand_types", [])
    return isinstance(operand_types, list) and any(
        isinstance(item, str) and _PTO_ACC_TILE_RE.search(item) for item in operand_types
    )


def _select_raw_pto_function(pto_text: str, function: str) -> str:
    """Select one function region from a possibly mixed PTO module."""
    matches = list(re.finditer(r"(?m)^\s*func\.func\s+@([A-Za-z0-9_]+)\b", pto_text))
    if not matches:
        return pto_text
    selected = [index for index, match in enumerate(matches) if match.group(1) == function]
    if len(selected) != 1:
        raise ValueError(f"raw PTO must contain exactly one function @{function}; found {len(selected)}")
    index = selected[0]
    start = matches[index].start()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(pto_text)
    return pto_text[start:end]


def _attach_pto_access_provenance(  # noqa: PLR0912 - fail-closed unique sequence alignment
    nodes: list[dict[str, Any]], pto_text: str, *, require_access_provenance: bool = True
) -> set[int]:
    """Join legacy SyncIR nodes to raw-PTO access locations by unique order.

    PTOAS's legacy text trace omits MLIR locations.  Raw PTO emitted with
    ``PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1`` contains those locations, while the
    trace preserves the executable operation order after lowering. Structural
    PTO ops such as ``pto.alloc_tile`` are ignored because they do not appear
    among the trace's operation names. PTOAS may eliminate an executable raw
    operation. Such a gap is accepted only when the final trace has exactly one
    monotone embedding in the raw stream; ambiguous repeated-operation joins
    still fail closed. The returned access orders are therefore independently
    proven non-materialized rather than guessed missing coordinates.
    """
    operation_nodes = [node for node in nodes if node.get("kind") == "operation"]
    expected_names = [str(node["op_name"]) for node in operation_nodes]
    traced_names = {_join_operation_name(name) for name in expected_names}
    pto_operations: list[tuple[str, int | None, str, dict[str, Any]]] = []
    constants = {
        match.group(1): f"{match.group(2)} : {match.group(3)}"
        for line in pto_text.splitlines()
        if (match := _PTO_SCALAR_CONSTANT_RE.match(line))
    }

    for line_number, line in enumerate(pto_text.splitlines(), start=1):
        locations = _ACCESS_LOCATION_RE.findall(line)
        raw_names = [match.group(1) for match in _PTO_OPERATION_RE.finditer(line)]
        names = [name for name in raw_names if _join_operation_name(name) in traced_names]
        if not names and len(set(locations)) == 1:
            # A stamped operation eliminated from the final trace is not in
            # ``traced_names``. The first PTO token is the operation name;
            # later tokens on the same line may be types such as
            # ``!pto.address_space`` and must not become phantom operations.
            names = raw_names[:1]
        if not names:
            continue
        if len(names) != 1:
            raise ValueError(f"raw PTO line {line_number} contains multiple traced operations: {names}")
        if len(set(locations)) > 1 or (require_access_provenance and len(set(locations)) != 1):
            raise ValueError(
                f"raw PTO operation {names[0]} on line {line_number} has no unambiguous "
                "pypto.access.N location"
            )
        access_order = int(locations[0]) if locations else None
        location_line = line.strip()
        pto_operations.append(
            (names[0], access_order, location_line, _operation_type_metadata(location_line, constants))
        )

    expected_count = len(expected_names)
    actual_count = len(pto_operations)
    ways = [[0] * (actual_count + 1) for _ in range(expected_count + 1)]
    for actual_index in range(actual_count + 1):
        ways[expected_count][actual_index] = 1
    for expected_index in range(expected_count - 1, -1, -1):
        for actual_index in range(actual_count - 1, -1, -1):
            count = ways[expected_index][actual_index + 1]
            actual_name, _, _, metadata = pto_operations[actual_index]
            if _operation_names_match(expected_names[expected_index], actual_name, metadata):
                count += ways[expected_index + 1][actual_index + 1]
            ways[expected_index][actual_index] = min(count, 2)

    if ways[0][0] != 1:
        raise ValueError(
            "raw PTO operation sequence has no unique monotone join to the final SyncIR trace: "
            f"alignments={ways[0][0]}, counts={expected_count}->{actual_count}"
        )

    matched_operations: list[tuple[str, int | None, str, dict[str, Any]]] = []
    nonmaterialized_access_orders: set[int] = set()
    expected_index = 0
    actual_index = 0
    while actual_index < actual_count:
        actual = pto_operations[actual_index]
        matches_current = expected_index < expected_count and _operation_names_match(
            expected_names[expected_index], actual[0], actual[3]
        )
        match_ways = ways[expected_index + 1][actual_index + 1] if matches_current else 0
        skip_ways = ways[expected_index][actual_index + 1]
        if match_ways:
            if skip_ways:
                raise ValueError("internal error: supposedly unique PTO operation join is ambiguous")
            matched_operations.append(actual)
            expected_index += 1
        else:
            access_order = actual[1]
            if access_order is None:
                raise ValueError(f"eliminated raw PTO operation {actual[0]} has no stable access provenance")
            nonmaterialized_access_orders.add(access_order)
        actual_index += 1
    if expected_index != expected_count:
        raise ValueError("internal error: unique PTO operation join did not consume the final trace")

    for node, (raw_name, access_order, location_line, metadata) in zip(
        operation_nodes, matched_operations, strict=True
    ):
        operation = {
            **metadata,
            "location": location_line,
            "raw_pto_op_name": raw_name,
        }
        if access_order is not None:
            operation["pypto_access_order"] = access_order
        node["operation"] = operation
    return nonmaterialized_access_orders


def _attach_pto_static_loop_bounds(nodes: list[dict[str, Any]], pto_text: str) -> int:
    """Join raw-PTO ``scf.for`` trip-count semantics to SyncIR loops.

    The legacy PTOAS debug stream identifies loop structure but reports SyncIR
    node ranges rather than iteration bounds. Raw PTO is the product-faithful
    source for the original ``scf.for`` lower/upper/step operands. Statically
    resolved bounds become concrete trip counts. Dynamic bounds receive a
    canonical parameter identity, so two loops proven to use the same bound
    can share one symbolic trip-count parameter. The loop order is preserved
    by PTOAS; a count mismatch is therefore an ambiguous bridge and fails
    closed.

    Returns:
        The number of loops whose bounds are genuinely dynamic or unsupported.
    """
    constants: dict[str, int] = {}
    expressions: dict[str, str] = {}
    raw_loops: list[
        tuple[str, int | None, int | None, int | None, str | None, str | None, int | None, str]
    ] = []

    def resolve(operand: str) -> int | None:
        if operand.startswith("%"):
            return constants.get(operand)
        return int(operand)

    def canonical_operand(operand: str) -> str:
        value = resolve(operand)
        if value is not None:
            return f"int:{value}"
        return expressions.get(operand, f"ssa:{operand}")

    def canonical_expression(rhs: str) -> str:
        rhs = re.sub(r"\s+loc\(.*$", "", rhs).strip()
        normalized = _PTO_SSA_VALUE_RE.sub(lambda match: f"<{canonical_operand(match.group(0))}>", rhs)
        return f"expr:{hashlib.sha256(normalized.encode()).hexdigest()}"

    for line in pto_text.splitlines():
        if match := _PTO_CONSTANT_RE.match(line):
            name, value = match.groups()
            constants[name] = int(value)
            expressions[name] = f"int:{int(value)}"
            continue
        if match := _PTO_FOR_RE.match(line):
            induction_variable = match.group(1)
            raw_operands = tuple(match.group(index) for index in range(2, 5))
            lower, upper, step = (resolve(operand) for operand in raw_operands)
            trip_count = None
            if lower is not None and upper is not None and step is not None and step > 0:
                trip_count = max(0, (upper - lower + step - 1) // step)
            parameter_identity = None
            parameter_expression = None
            if trip_count is None:
                parameter_expression = (
                    "trip(" + ",".join(canonical_operand(operand) for operand in raw_operands) + ")"
                )
                parameter_identity = (
                    "loop-trip-v1:" + hashlib.sha256(parameter_expression.encode()).hexdigest()
                )
            raw_loops.append(
                (
                    induction_variable,
                    lower,
                    upper,
                    step,
                    parameter_identity,
                    parameter_expression,
                    (
                        int(source_loop.group(1))
                        if (source_loop := _PTO_SOURCE_LOOP_RE.search(line)) is not None
                        else None
                    ),
                    line.strip(),
                )
            )
            continue
        if assignment := _PTO_ASSIGN_RE.match(line):
            result, rhs = assignment.groups()
            expressions[result] = canonical_expression(rhs)

    loop_begins = [
        node for node in nodes if node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_BEGIN"
    ]
    if len(raw_loops) != len(loop_begins):
        raise ValueError(
            "raw PTO loop sequence does not match the final SyncIR trace: "
            f"counts={len(raw_loops)}->{len(loop_begins)}"
        )

    loop_semantics_by_begin: dict[int, tuple[int | None, str | None, str | None]] = {}
    for node, (
        induction_variable,
        lower,
        upper,
        step,
        parameter_identity,
        parameter_expression,
        source_loop_id,
        source_line,
    ) in zip(loop_begins, raw_loops, strict=True):
        trip_count = None
        if lower is not None and upper is not None and step is not None and step > 0:
            trip_count = max(0, (upper - lower + step - 1) // step)
        node["static_trip_count"] = trip_count
        node["operation"] = {"raw_pto_loop": source_line}
        node["raw_pto_induction_variable"] = induction_variable
        node["raw_pto_lower_bound"] = lower
        node["raw_pto_step"] = step
        if parameter_identity is not None:
            node["dynamic_trip_count_identity"] = parameter_identity
            node["dynamic_trip_count_expression"] = parameter_expression
        if source_loop_id is not None:
            node["pypto_source_loop_id"] = source_loop_id
        loop_semantics_by_begin[int(node["id"])] = (
            trip_count,
            parameter_identity,
            parameter_expression,
        )
    for node in nodes:
        if node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_END":
            semantics = loop_semantics_by_begin.get(int(node["begin"]))
            if semantics is None:
                continue
            trip_count, parameter_identity, parameter_expression = semantics
            node["static_trip_count"] = trip_count
            if parameter_identity is not None:
                node["dynamic_trip_count_identity"] = parameter_identity
                node["dynamic_trip_count_expression"] = parameter_expression

    return sum(
        lower is None or upper is None or step is None or step <= 0
        for _, lower, upper, step, _, _, _, _ in raw_loops
    )


def _pto_function_arguments_and_loop_depths(lines: Sequence[str]) -> tuple[set[str], list[int]]:
    """Return function arguments and enclosing ``scf.for`` depth per PTO line."""
    function_arguments: set[str] = set()
    for line in lines:
        if "func.func" not in line or "(" not in line:
            continue
        signature = line.split("(", maxsplit=1)[1].split(")", maxsplit=1)[0]
        function_arguments.update(_PTO_SSA_VALUE_RE.findall(signature))
        break

    loop_depth_by_line: list[int] = []
    brace_depth = 0
    loop_exit_depths: list[int] = []
    for line in lines:
        while loop_exit_depths and brace_depth <= loop_exit_depths[-1]:
            loop_exit_depths.pop()
        loop_depth_by_line.append(len(loop_exit_depths))
        if _PTO_FOR_RE.match(line) and "{" in line:
            loop_exit_depths.append(brace_depth)
        brace_depth += line.count("{") - line.count("}")
    return function_arguments, loop_depth_by_line


def _pto_comparison_boolean_origin(
    rhs: str,
    integer_values: Mapping[str, int],
    bool_origins: Mapping[str, tuple[str, bool]],
) -> tuple[str, bool] | None:
    """Resolve a comparison against zero to an existing boolean origin."""
    match = re.search(r"arith\.cmpi\s+(slt|sgt|ne|eq),\s*([^,]+),\s*([^ :]+)", rhs)
    if match is None:
        return None
    predicate, left, right = match.groups()
    left, right = left.strip(), right.strip()
    left_zero = integer_values.get(left) == 0 or left == "0"
    right_zero = integer_values.get(right) == 0 or right == "0"
    source: str | None = None
    polarity = True
    if predicate == "slt" and left_zero and right in bool_origins:
        source = right
    elif predicate == "sgt" and right_zero and left in bool_origins:
        source = left
    elif predicate in {"ne", "eq"}:
        if left_zero and right in bool_origins:
            source = right
        elif right_zero and left in bool_origins:
            source = left
        polarity = predicate == "ne"
    if source is None:
        return None
    identity, source_polarity = bool_origins[source]
    return identity, source_polarity == polarity


def _pto_loop_invariant_values(
    lines: Sequence[str],
    function_arguments: set[str],
    loop_depth_by_line: Sequence[int],
    selects: Sequence[tuple[str, str, str, str]],
) -> set[str]:
    """Prove scalar SSA values invariant across every enclosing raw-PTO loop."""
    invariant = set(function_arguments)
    allowed_inside_loop = (
        "arith.constant",
        "arith.index_cast",
        "arith.ext",
        "arith.trunc",
        "arith.cmpi",
        "arith.cmpf",
    )

    def token_is_invariant(token: str) -> bool:
        return token.lstrip("-").isdigit() or token in invariant

    changed = True
    while changed:
        changed = False
        for line_index, line in enumerate(lines):
            assignment = _PTO_ASSIGN_RE.match(line)
            if assignment is None:
                continue
            result, rhs = assignment.groups()
            if result in invariant:
                continue
            tokens = _PTO_SSA_VALUE_RE.findall(rhs)
            top_level = loop_depth_by_line[line_index] == 0
            pure_scalar = any(operation in rhs for operation in allowed_inside_loop)
            if (top_level or pure_scalar) and all(token_is_invariant(token) for token in tokens):
                invariant.add(result)
                changed = True
        for result, condition, then_value, else_value in selects:
            if result in invariant:
                continue
            if all(token_is_invariant(token) for token in (condition, then_value, else_value)):
                invariant.add(result)
                changed = True
    return invariant


def _pto_boolean_origins(
    lines: Sequence[str],
    expressions: Mapping[str, str],
    integer_values: Mapping[str, int],
    selects: Sequence[tuple[str, str, str, str]],
) -> dict[str, tuple[str, bool]]:
    """Trace materialized boolean values to one expression and polarity."""
    bool_origins: dict[str, tuple[str, bool]] = {}
    for line in lines:
        assignment = _PTO_ASSIGN_RE.match(line)
        if assignment is None:
            continue
        result, rhs = assignment.groups()
        if "arith.cmp" in rhs:
            bool_origins[result] = (f"expr:{expressions[result]}", True)

    for result, condition, then_value, else_value in selects:
        then_int = int(then_value) if then_value.lstrip("-").isdigit() else integer_values.get(then_value)
        else_int = int(else_value) if else_value.lstrip("-").isdigit() else integer_values.get(else_value)
        origin = bool_origins.get(condition, (f"expr:{expressions.get(condition, f'ssa:{condition}')}", True))
        if then_int is not None and else_int is not None:
            if then_int != 0 and else_int == 0:
                bool_origins[result] = origin
            elif then_int == 0 and else_int != 0:
                bool_origins[result] = (origin[0], not origin[1])

    # Propagate recognized materializations through casts and canonical
    # comparisons against zero.
    for line in lines:
        assignment = _PTO_ASSIGN_RE.match(line)
        if assignment is None:
            continue
        result, rhs = assignment.groups()
        tokens = _PTO_SSA_VALUE_RE.findall(rhs)
        if "arith.index_cast" in rhs and len(tokens) == 1 and tokens[0] in bool_origins:
            bool_origins[result] = bool_origins[tokens[0]]
            continue
        comparison_origin = _pto_comparison_boolean_origin(rhs, integer_values, bool_origins)
        if comparison_origin is not None:
            bool_origins[result] = comparison_origin
    return bool_origins


def _evaluate_static_pto_integer(  # noqa: PLR0912 - explicit fail-closed SSA subset
    value: str,
    definitions: Mapping[str, str],
    integer_values: Mapping[str, int],
    induction_values: Mapping[str, int],
    visiting: frozenset[str] = frozenset(),
) -> int | None:
    """Evaluate the integer SSA subset used by static loop predicates."""
    if value.lstrip("-").isdigit():
        return int(value)
    if value in induction_values:
        return induction_values[value]
    if value in integer_values:
        return integer_values[value]
    if value in visiting:
        return None
    rhs = definitions.get(value)
    if rhs is None:
        return None
    next_visiting = visiting | {value}

    cast = re.search(r"arith\.(?:index_cast|extsi|extui|trunci)\s+(%[-A-Za-z0-9_.$]+)", rhs)
    if cast is not None:
        return _evaluate_static_pto_integer(
            cast.group(1), definitions, integer_values, induction_values, next_visiting
        )

    binary = re.search(
        r"arith\.(addi|subi|muli|divsi|floordivsi|ceildivsi|remsi|minsi|maxsi)\s+"
        r"([^,\s:]+)\s*,\s*([^\s:]+)",
        rhs,
    )
    if binary is not None:
        operation, left_operand, right_operand = binary.groups()
        left = _evaluate_static_pto_integer(
            left_operand, definitions, integer_values, induction_values, next_visiting
        )
        right = _evaluate_static_pto_integer(
            right_operand, definitions, integer_values, induction_values, next_visiting
        )
        if left is None or right is None:
            return None
        if operation == "addi":
            return left + right
        if operation == "subi":
            return left - right
        if operation == "muli":
            return left * right
        if operation in {"divsi", "floordivsi"}:
            if right == 0:
                return None
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient
        if operation == "ceildivsi":
            if right == 0:
                return None
            return -(left // -right)
        if operation == "remsi":
            if right == 0:
                return None
            quotient = abs(left) // abs(right)
            quotient = -quotient if (left < 0) != (right < 0) else quotient
            return left - quotient * right
        if operation == "minsi":
            return min(left, right)
        if operation == "maxsi":
            return max(left, right)

    comparison = re.search(
        r"arith\.cmpi\s+(eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)\s*,\s*"
        r"([^,\s:]+)\s*,\s*([^\s:]+)",
        rhs,
    )
    if comparison is None:
        return None
    predicate, left_operand, right_operand = comparison.groups()
    left = _evaluate_static_pto_integer(
        left_operand, definitions, integer_values, induction_values, next_visiting
    )
    right = _evaluate_static_pto_integer(
        right_operand, definitions, integer_values, induction_values, next_visiting
    )
    if left is None or right is None:
        return None
    if predicate.startswith("u") and (left < 0 or right < 0):
        # Raw PTO does not retain the integer bit width in this compact SSA
        # evaluator. Interpreting a negative value as unsigned would require
        # that width, so fail closed instead of guessing one.
        return None
    comparisons = {
        "eq": left == right,
        "ne": left != right,
        "slt": left < right,
        "sle": left <= right,
        "sgt": left > right,
        "sge": left >= right,
        "ult": left < right,
        "ule": left <= right,
        "ugt": left > right,
        "uge": left >= right,
    }
    return int(comparisons[predicate])


def _exact_branch_iteration_profile(
    node: Mapping[str, Any],
    condition: str,
    definitions: Mapping[str, str],
    integer_values: Mapping[str, int],
    loops_by_id: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Derive an exact static-loop branch sequence from induction-variable SSA."""
    loop_stack = node.get("loop_stack", [])
    if not isinstance(loop_stack, list) or not loop_stack:
        return None
    if not all(isinstance(loop, int) for loop in loop_stack):
        raise ValueError(f"branch node {node.get('id')} has an invalid loop stack")
    loops: list[Mapping[str, Any]] = []
    counts: list[int] = []
    for loop_id in loop_stack:
        loop = loops_by_id.get(loop_id)
        count = loop.get("static_trip_count") if loop is not None else None
        induction = loop.get("raw_pto_induction_variable") if loop is not None else None
        lower = loop.get("raw_pto_lower_bound") if loop is not None else None
        step = loop.get("raw_pto_step") if loop is not None else None
        if (
            not isinstance(count, int)
            or not isinstance(induction, str)
            or not isinstance(lower, int)
            or not isinstance(step, int)
            or step <= 0
        ):
            return None
        loops.append(loop)
        counts.append(count)
    context_count = math.prod(counts)
    if context_count > _MAX_QUEUE_EVENT_EXPANDED_NODES:
        raise ValueError(
            "exact branch-profile expansion exceeds the resource-safe node budget: "
            f"branch={node.get('id')}, contexts={context_count}, "
            f"limit={_MAX_QUEUE_EVENT_EXPANDED_NODES}"
        )

    values: list[bool] = []
    for context in itertools.product(*(range(count) for count in counts)):
        induction_values = {
            str(loop["raw_pto_induction_variable"]): int(loop["raw_pto_lower_bound"])
            + iteration * int(loop["raw_pto_step"])
            for loop, iteration in zip(loops, context, strict=True)
        }
        evaluated = _evaluate_static_pto_integer(condition, definitions, integer_values, induction_values)
        if evaluated is None:
            return None
        values.append(bool(evaluated))
    return {
        "schema_version": 1,
        "evaluation": "static_integer_ssa_induction_profile_v1",
        "loop_ids": list(loop_stack),
        "iteration_counts": counts,
        "values": values,
    }


def _attach_pto_branch_predicates(  # noqa: PLR0912 - fail-closed SSA predicate parser
    nodes: list[dict[str, Any]], pto_text: str
) -> int:
    """Attach canonical predicate identity and polarity to structured IF nodes.

    The native v0.57 schedule graph records branch structure but not the SSA
    value controlling each branch. Treating every IF marker as independent
    admits impossible paths when a loop-invariant predicate is re-tested. This
    bridge follows scalar SSA definitions in the raw pre-InsertSync PTO and
    recognizes the common ``if p then 1 else 0`` materialization plus casts and
    comparisons against zero. Unrecognized predicates still receive a stable
    expression identity; only proven aliases are coalesced.

    Returns the number of IF markers whose predicate could not be recovered.
    """
    lines = pto_text.splitlines()
    expressions: dict[str, str] = {}
    integer_values: dict[str, int] = {}
    definitions: dict[str, str] = {}
    selects: list[tuple[str, str, str, str]] = []
    branch_conditions: list[str] = []

    # Record whether each raw-PTO line executes inside any scf.for body. This
    # is deliberately scope based: one scenario bit may be reused across all
    # dynamic occurrences only when its value is defined outside every loop
    # that contains the IF. A function argument and a top-level definition are
    # sufficient evidence; an unknown block argument is not.
    function_arguments, loop_depth_by_line = _pto_function_arguments_and_loop_depths(lines)

    def strip_location(text: str) -> str:
        return re.sub(r"\s+loc\(.*$", "", text).strip()

    def token_expression(token: str) -> str:
        if token.lstrip("-").isdigit():
            return f"int:{int(token)}"
        return expressions.get(token, f"ssa:{token}")

    def normalized_expression(rhs: str) -> str:
        rhs = strip_location(rhs)
        return _PTO_SSA_VALUE_RE.sub(lambda match: f"<{token_expression(match.group(0))}>", rhs)

    # First pass: record stable scalar expressions, integral values, IF
    # conditions, and result-producing IF yields.
    for index, line in enumerate(lines):
        if match := _PTO_IF_RE.match(line):
            result, condition = match.groups()
            branch_conditions.append(condition)
            if result is not None:
                then_value: str | None = None
                else_value: str | None = None
                in_else = False
                depth = line.count("{") - line.count("}")
                for nested in lines[index + 1 :]:
                    if depth == 1 and re.match(r"^\s*}\s*else\s*{\s*$", nested):
                        in_else = True
                    elif yield_match := _PTO_YIELD_RE.match(nested):
                        if depth == 1:
                            if in_else:
                                else_value = yield_match.group(1)
                            else:
                                then_value = yield_match.group(1)
                    depth += nested.count("{") - nested.count("}")
                    if depth == 0:
                        break
                if then_value is not None and else_value is not None:
                    selects.append((result, condition, then_value, else_value))
            continue
        if constant := _PTO_CONSTANT_RE.match(line):
            name, value = constant.groups()
            integer_values[name] = int(value)
            expressions[name] = f"int:{int(value)}"
            continue
        if assignment := _PTO_ASSIGN_RE.match(line):
            result, rhs = assignment.groups()
            definitions[result] = strip_location(rhs)
            expressions[result] = hashlib.sha256(normalized_expression(rhs).encode()).hexdigest()
            tokens = _PTO_SSA_VALUE_RE.findall(rhs)
            if "arith.index_cast" in rhs and len(tokens) == 1 and tokens[0] in integer_values:
                integer_values[result] = integer_values[tokens[0]]

    bool_origins = _pto_boolean_origins(lines, expressions, integer_values, selects)
    loop_invariant_values = _pto_loop_invariant_values(lines, function_arguments, loop_depth_by_line, selects)

    if_begins = [
        node for node in nodes if node.get("kind") == "branch" and node.get("branch_kind") == "IF_BEGIN"
    ]
    if len(branch_conditions) != len(if_begins):
        raise ValueError(
            "raw PTO branch sequence does not match the final SyncIR trace: "
            f"counts={len(branch_conditions)}->{len(if_begins)}"
        )
    missing = 0
    loops_by_id = {
        int(node["id"]): node
        for node in nodes
        if node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(node.get("id"), int)
    }
    for node, condition in zip(if_begins, branch_conditions, strict=True):
        origin = bool_origins.get(condition)
        if origin is None:
            expression = token_expression(condition)
            if expression.startswith("ssa:"):
                missing += 1
            origin = (f"expr:{expression}", True)
        node["predicate_identity"], node["predicate_true_value"] = origin
        exact_profile = _exact_branch_iteration_profile(
            node, condition, definitions, integer_values, loops_by_id
        )
        if exact_profile is not None:
            node["predicate_iteration_profile"] = exact_profile
            node["predicate_loop_invariant"] = len(set(exact_profile["values"])) == 1
        else:
            node["predicate_loop_invariant"] = (
                not node.get("loop_stack") or condition in loop_invariant_values or exact_profile is not None
            )
    return missing


def enrich_native_schedule_from_pto(
    record: Mapping[str, Any], pto_text: str, *, pto_source: str = "<raw-pto>"
) -> dict[str, Any]:
    """Join a native PTOAS sync graph to portable raw-PTO semantics.

    The v0.57 exporter is authoritative for InsertSync nodes and edges, but
    MLIR's destination-style operation representation places ``outs`` tile
    types among operands. Exact duration calibration is keyed by the original
    ``ins``/``outs`` syntax. Rejoin by the complete executable operation
    sequence, refusing any count or opcode mismatch, and preserve every native
    synchronization field unchanged.
    """
    enriched = copy.deepcopy(dict(record))
    raw_nodes = enriched.get("nodes")
    if not isinstance(raw_nodes, list) or not all(isinstance(node, dict) for node in raw_nodes):
        raise ValueError("native schedule must contain mutable object nodes")
    function_pto = _select_raw_pto_function(pto_text, str(enriched.get("function", "")))
    nonmaterialized_access_orders = _attach_pto_access_provenance(
        raw_nodes, function_pto, require_access_provenance=False
    )
    unresolved_loops = _attach_pto_static_loop_bounds(raw_nodes, function_pto)
    unresolved_predicates = _attach_pto_branch_predicates(raw_nodes, function_pto)
    enriched["export_source"] = "native_schedule_graph_v1+raw_pto_semantics_v1"
    enriched["raw_pto_source"] = pto_source
    enriched["raw_pto_sha256"] = hashlib.sha256(pto_text.encode()).hexdigest()
    enriched["raw_pto_operation_join"] = "unique_monotone_executable_order_v2"
    enriched["nonmaterialized_access_orders"] = sorted(nonmaterialized_access_orders)
    limitations = enriched.get("export_limitations")
    if not isinstance(limitations, Mapping):
        limitations = {}
    enriched["export_limitations"] = {
        **limitations,
        "operation_metadata_missing": 0,
        "static_loop_bounds_missing": unresolved_loops,
        "branch_predicates_missing": unresolved_predicates,
    }
    return enriched


def apply_runtime_branch_profile(  # noqa: PLR0912 - fail-closed external evidence contract
    record: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    schedule_sha256: str,
    problem_sha256: str,
    input_set_sha256: str,
    trip_metadata_sha256: str,
) -> dict[str, Any]:
    """Apply an exact, digest-bound runtime branch profile to one schedule.

    Runtime-loaded predicates cannot be inferred from raw PTO.  A profile may
    therefore specialize the analysis to one captured input set, but only when
    it records every branch outcome per loop occurrence and binds those values
    to the exact schedule, DSA problem, input manifest, and loop-trip metadata.
    Scalar values without an immutability/derivation proof are deliberately not
    accepted as a shortcut.
    """

    if profile.get("schema_version") != 1 or profile.get("contract") != "exact_runtime_branch_profile_v1":
        raise ValueError("runtime branch profile must use exact_runtime_branch_profile_v1")
    bindings = profile.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("runtime branch profile is missing bindings")
    expected_bindings = {
        "schedule_sha256": schedule_sha256,
        "problem_sha256": problem_sha256,
        "input_set_sha256": input_set_sha256,
        "trip_metadata_sha256": trip_metadata_sha256,
    }
    for key, expected in expected_bindings.items():
        actual = bindings.get(key)
        if actual != expected:
            raise ValueError(f"runtime branch profile {key} does not match the scored input")

    enriched = copy.deepcopy(dict(record))
    raw_nodes = enriched.get("nodes")
    if not isinstance(raw_nodes, list) or not all(isinstance(node, dict) for node in raw_nodes):
        raise ValueError("runtime branch profile requires mutable object nodes")
    nodes_by_id = {int(node["id"]): node for node in raw_nodes if isinstance(node.get("id"), int)}

    raw_trip_counts = profile.get("loop_trip_counts")
    if not isinstance(raw_trip_counts, list):
        raise ValueError("runtime branch profile loop_trip_counts must be a list")
    profiled_loops: set[int] = set()
    for item in raw_trip_counts:
        if not isinstance(item, Mapping):
            raise ValueError("runtime branch profile has a malformed loop trip-count entry")
        loop_id, trip_count = item.get("loop_id"), item.get("trip_count")
        if not isinstance(loop_id, int) or not isinstance(trip_count, int) or trip_count < 0:
            raise ValueError("runtime loop trip counts require a non-negative integer id and count")
        if loop_id in profiled_loops:
            raise ValueError(f"runtime branch profile repeats loop {loop_id}")
        loop = nodes_by_id.get(loop_id)
        if loop is None or loop.get("kind") != "loop" or loop.get("loop_kind") != "LOOP_BEGIN":
            raise ValueError(f"runtime branch profile references unknown LOOP_BEGIN {loop_id}")
        compile_time_count = loop.get("static_trip_count")
        if isinstance(compile_time_count, int) and compile_time_count != trip_count:
            raise ValueError(
                f"runtime trip count for loop {loop_id} contradicts its static count: "
                f"{trip_count} != {compile_time_count}"
            )
        loop["compile_time_static_trip_count"] = compile_time_count
        loop["static_trip_count"] = trip_count
        loop["runtime_profiled_trip_count"] = True
        profiled_loops.add(loop_id)

    raw_branches = profile.get("branches")
    if not isinstance(raw_branches, list):
        raise ValueError("runtime branch profile branches must be a list")
    profiled_branches: set[int] = set()
    for item in raw_branches:
        if not isinstance(item, Mapping):
            raise ValueError("runtime branch profile has a malformed branch entry")
        node_id = item.get("if_node_id")
        predicate_identity = item.get("predicate_identity")
        loop_ids = item.get("loop_ids")
        counts = item.get("iteration_counts")
        values = item.get("values")
        active_flat_indices = item.get("active_flat_indices")
        derivation = item.get("derivation")
        if not isinstance(node_id, int) or node_id in profiled_branches:
            raise ValueError("runtime branch profile requires unique integer if_node_id values")
        node = nodes_by_id.get(node_id)
        if node is None or node.get("kind") != "branch" or node.get("branch_kind") != "IF_BEGIN":
            raise ValueError(f"runtime branch profile references unknown IF_BEGIN {node_id}")
        if not isinstance(predicate_identity, str) or predicate_identity != node.get("predicate_identity"):
            raise ValueError(f"runtime branch profile predicate identity does not match IF_BEGIN {node_id}")
        if (
            not isinstance(loop_ids, list)
            or not all(isinstance(loop_id, int) for loop_id in loop_ids)
            or loop_ids != node.get("loop_stack", [])
            or not isinstance(counts, list)
            or not all(isinstance(count, int) and count >= 0 for count in counts)
            or len(loop_ids) != len(counts)
            or not isinstance(values, list)
            or not all(isinstance(value, bool) for value in values)
        ):
            raise ValueError(f"runtime branch profile for IF_BEGIN {node_id} has malformed occurrences")
        context_count = math.prod(counts)
        if active_flat_indices is None:
            active_flat_indices = list(range(context_count))
        if (
            not isinstance(active_flat_indices, list)
            or not all(isinstance(index, int) for index in active_flat_indices)
            or active_flat_indices != sorted(set(active_flat_indices))
            or any(index < 0 or index >= context_count for index in active_flat_indices)
            or len(values) != len(active_flat_indices)
        ):
            raise ValueError(
                f"runtime branch profile for IF_BEGIN {node_id} has malformed active occurrences"
            )
        for loop_id, count in zip(loop_ids, counts, strict=True):
            loop = nodes_by_id.get(loop_id)
            if loop is None or loop.get("static_trip_count") != count:
                raise ValueError(
                    f"runtime branch profile for IF_BEGIN {node_id} disagrees with loop {loop_id}"
                )
        if not isinstance(derivation, Mapping):
            raise ValueError(f"runtime branch profile for IF_BEGIN {node_id} is missing derivation proof")
        if derivation.get("kind") not in {
            "captured_branch_outcomes_v1",
            "captured_immutable_scalar_expression_v1",
        }:
            raise ValueError(f"runtime branch profile for IF_BEGIN {node_id} has unsupported derivation")
        evidence_sha256 = derivation.get("evidence_sha256")
        if not isinstance(evidence_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None:
            raise ValueError(
                f"runtime branch profile for IF_BEGIN {node_id} requires a SHA-256 derivation proof"
            )
        if derivation["kind"] == "captured_immutable_scalar_expression_v1" and (
            derivation.get("immutability_proof") not in {"function_argument_v1", "pre_loop_ssa_definition_v1"}
        ):
            raise ValueError(
                f"runtime branch profile for IF_BEGIN {node_id} lacks a supported scalar immutability proof"
            )
        exact_profile = {
            "schema_version": 1,
            "evaluation": "exact_runtime_capture_v1",
            "loop_ids": list(loop_ids),
            "iteration_counts": list(counts),
            "active_flat_indices": list(active_flat_indices),
            "values": list(values),
            "derivation": dict(derivation),
        }
        existing = node.get("predicate_iteration_profile")
        if isinstance(existing, Mapping):
            comparable_existing = {
                key: existing.get(key)
                for key in ("loop_ids", "iteration_counts", "active_flat_indices", "values")
            }
            comparable_runtime = {
                key: exact_profile[key]
                for key in ("loop_ids", "iteration_counts", "active_flat_indices", "values")
            }
            if comparable_existing["active_flat_indices"] is None:
                comparable_existing["active_flat_indices"] = list(range(math.prod(counts)))
            if comparable_existing != comparable_runtime:
                raise ValueError(f"runtime branch profile for IF_BEGIN {node_id} contradicts static analysis")
        node["predicate_iteration_profile"] = exact_profile
        node["predicate_loop_invariant"] = len(set(values)) <= 1
        profiled_branches.add(node_id)

    canonical_profile = json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    enriched["runtime_branch_profile"] = {
        "contract": "exact_runtime_branch_profile_v1",
        "semantic_sha256": hashlib.sha256(canonical_profile).hexdigest(),
        "bindings": dict(expected_bindings),
        "profiled_loop_ids": sorted(profiled_loops),
        "profiled_if_node_ids": sorted(profiled_branches),
    }
    return enriched


def apply_runtime_parallel_branch_profile(  # noqa: PLR0912 - fail-closed evidence contract
    record: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    schedule_sha256: str,
    problem_sha256: str,
    input_set_sha256: str,
    trip_metadata_sha256: str,
) -> dict[str, Any]:
    """Attach an exact, digest-bound branch profile for parallel instances.

    A kernel dispatch may execute the same structured graph on several cores
    or blocks with different loop-invariant branch outcomes.  The latency of
    that dispatch is the maximum instance makespan, not the maximum placement
    extension considered in isolation.  This profile records which complete
    branch scenarios were actually present without pretending that their
    captured values are compile-time facts.
    """

    if (
        profile.get("schema_version") != 1
        or profile.get("contract") != "exact_runtime_parallel_branch_profile_v1"
    ):
        raise ValueError("runtime parallel branch profile must use exact_runtime_parallel_branch_profile_v1")
    bindings = profile.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("runtime parallel branch profile is missing bindings")
    expected_bindings = {
        "schedule_sha256": schedule_sha256,
        "problem_sha256": problem_sha256,
        "input_set_sha256": input_set_sha256,
        "trip_metadata_sha256": trip_metadata_sha256,
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise ValueError(f"runtime parallel branch profile {key} does not match the scored input")

    derivation = profile.get("derivation")
    if (
        not isinstance(derivation, Mapping)
        or derivation.get("kind") != "captured_parallel_branch_outcomes_v1"
    ):
        raise ValueError("runtime parallel branch profile lacks captured outcome provenance")
    evidence_sha256 = derivation.get("evidence_sha256")
    if not isinstance(evidence_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None:
        raise ValueError("runtime parallel branch profile requires a SHA-256 derivation proof")

    nodes_by_id = {
        int(node["id"]): node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    raw_scenarios = profile.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("runtime parallel branch profile scenarios must be a non-empty list")
    normalized_scenarios: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, bool], ...]] = set()
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping):
            raise ValueError("runtime parallel branch profile has a malformed scenario")
        instance_count = raw.get("instance_count")
        raw_choices = raw.get("branch_choices")
        if not isinstance(instance_count, int) or isinstance(instance_count, bool) or instance_count <= 0:
            raise ValueError("parallel branch scenario instance_count must be a positive integer")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError("parallel branch scenario branch_choices must be a non-empty list")
        choices: dict[int, bool] = {}
        for item in raw_choices:
            if not isinstance(item, Mapping):
                raise ValueError("parallel branch scenario has a malformed branch choice")
            node_id = item.get("if_node_id")
            value = item.get("value")
            predicate_identity = item.get("predicate_identity")
            if not isinstance(node_id, int) or not isinstance(value, bool) or node_id in choices:
                raise ValueError(
                    "parallel branch scenario requires unique integer branch ids and bool values"
                )
            node = nodes_by_id.get(node_id)
            if node is None or node.get("kind") != "branch" or node.get("branch_kind") != "IF_BEGIN":
                raise ValueError(f"parallel branch scenario references unknown IF_BEGIN {node_id}")
            if node.get("predicate_loop_invariant") is not True:
                raise ValueError(f"parallel branch scenario IF_BEGIN {node_id} is not proven loop-invariant")
            if not isinstance(predicate_identity, str) or predicate_identity != node.get(
                "predicate_identity"
            ):
                raise ValueError(
                    f"parallel branch scenario predicate identity does not match IF_BEGIN {node_id}"
                )
            choices[node_id] = value
        key = _branch_scenario_key(choices)
        if key in seen:
            raise ValueError(f"runtime parallel branch profile repeats scenario {key}")
        seen.add(key)
        normalized_scenarios.append(
            {
                "instance_count": instance_count,
                "branch_choices": {str(node_id): value for node_id, value in key},
            }
        )

    enriched = copy.deepcopy(dict(record))
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    enriched["runtime_parallel_branch_profile"] = {
        "contract": "exact_runtime_parallel_branch_profile_v1",
        "semantic_sha256": hashlib.sha256(canonical).hexdigest(),
        "bindings": dict(expected_bindings),
        "derivation": dict(derivation),
        "parallel_instance_count": sum(row["instance_count"] for row in normalized_scenarios),
        "scenarios": sorted(
            normalized_scenarios,
            key=lambda row: _branch_scenario_key(row["branch_choices"]),
        ),
    }
    return enriched


def import_insert_sync_debug(  # noqa: PLR0912,PLR0915 - parser mirrors the debug record state machine
    text: str, *, function: str, pto_text: str | None = None
) -> dict[str, Any]:
    """Convert PTOAS's legacy level-3 final SyncIR dump to schema v1.

    This is a compatibility bridge for archived or pre-exporter runs. The
    native C++ exporter remains authoritative: the text dump omits allocation
    sizes, operation attributes, barrier dependency nodes, and static loop
    bounds. With raw PTO, the bridge recovers operand/result types, static
    operation work bytes, access provenance, and provable loop trip counts.
    The returned record names every remaining limitation explicitly.
    """
    marker = "// === [PTOInsertSync Debug] After EventId Allocation === //"
    start = text.rfind(marker)
    if start < 0:
        raise ValueError("debug log has no final 'After EventId Allocation' phase")
    phase = text[start + len(marker) :]
    end = phase.find("// ========================================= //")
    if end < 0:
        raise ValueError("final debug phase is not terminated")
    lines = phase[:end].splitlines()

    nodes: list[dict[str, Any]] = []
    stream_edges: list[dict[str, Any]] = []
    loop_stack: list[int] = []
    branch_stack: list[int] = []
    previous_by_pipe: dict[str, int] = {}
    sync_operations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    current_node: dict[str, Any] | None = None

    for line in lines:
        if match := _DEBUG_NODE_RE.match(line):
            node_id = int(match.group(1))
            pipe = match.group(3)
            current_node = {
                "id": node_id,
                "kind": "operation",
                "op_name": match.group(2),
                "pipe": pipe,
                "macro_phase": -1,
                "loop_stack": list(loop_stack),
                "branch_stack": list(branch_stack),
                "defs": [],
                "uses": [],
                "operation": {},
            }
            nodes.append(current_node)
            if pipe in previous_by_pipe:
                stream_edges.append({"source": previous_by_pipe[pipe], "target": node_id, "pipe": pipe})
            previous_by_pipe[pipe] = node_id
            continue

        if match := _DEBUG_LOOP_RE.match(line):
            node_id = int(match.group(1))
            kind = match.group(2)
            if kind == "LOOP_END" and loop_stack:
                loop_stack.pop()
            current_node = {
                "id": node_id,
                "kind": "loop",
                "loop_kind": kind,
                "begin": int(match.group(3)),
                "end": int(match.group(4)),
                "static_trip_count": None,
                "loop_stack": list(loop_stack),
                "branch_stack": list(branch_stack),
                "operation": {},
            }
            nodes.append(current_node)
            if kind == "LOOP_BEGIN":
                loop_stack.append(node_id)
            continue

        if match := _DEBUG_BRANCH_RE.match(line):
            node_id = int(match.group(1))
            kind = match.group(2)
            if kind == "IF_END" and branch_stack:
                branch_stack.pop()
            current_node = {
                "id": node_id,
                "kind": "branch",
                "branch_kind": kind,
                "begin": int(match.group(3)),
                "branch": int(match.group(4)),
                "end": int(match.group(5)),
                "loop_stack": list(loop_stack),
                "branch_stack": list(branch_stack),
                "operation": {},
            }
            nodes.append(current_node)
            if kind == "IF_BEGIN":
                branch_stack.append(node_id)
            elif kind == "ELSE_BEGIN":
                if branch_stack:
                    branch_stack.pop()
                branch_stack.append(node_id)
            continue

        if match := _DEBUG_PLACEHOLDER_RE.match(line):
            current_node = {
                "id": int(match.group(1)),
                "kind": "placeholder",
                "parent_scope": int(match.group(2)),
                "virtual_else": False,
                "loop_stack": list(loop_stack),
                "branch_stack": list(branch_stack),
                "operation": {},
            }
            nodes.append(current_node)
            continue

        if current_node is not None and (match := _DEBUG_MEM_RE.match(line)):
            key = "defs" if match.group(1) == "def" else "uses"
            entries = []
            for value, scope in _DEBUG_MEM_ITEM_RE.findall(match.group(2)):
                entries.append(
                    {
                        "base": value,
                        "root": value,
                        "scope": scope,
                        "allocate_size_bytes": 0,
                        "known_physical_addresses": False,
                        "aliases_unknown_range": False,
                        "base_addresses": [],
                    }
                )
            current_node[key] = entries
            continue

        if current_node is not None and (match := _DEBUG_SYNC_RE.match(line)):
            sync_index = int(match.group(5))
            suffix = match.group(6)
            event_ids_match = _DEBUG_EVENT_IDS_RE.search(suffix)
            event_ids = (
                [int(value) for value in event_ids_match.group(1).split(",") if value]
                if event_ids_match
                else []
            )
            dependency_match = _DEBUG_DEPENDENCY_NODE_RE.search(suffix)
            loop_end_match = _DEBUG_FOR_END_RE.search(suffix)
            sync_operations[sync_index].append(
                {
                    "placement": match.group(1),
                    "type": match.group(2),
                    "node": current_node["id"],
                    "dependency_node": (int(dependency_match.group(1)) if dependency_match else None),
                    "src_pipe": match.group(3),
                    "dst_pipe": match.group(4),
                    "loop_end": int(loop_end_match.group(1)) if loop_end_match else None,
                    "event_ids": event_ids,
                    "useless": bool(re.search(r"\buseless\b", suffix)),
                }
            )

    if not any(node["kind"] == "operation" for node in nodes):
        raise ValueError("final debug phase has no operation nodes")
    missing_static_loop_bounds = sum(
        node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_BEGIN" for node in nodes
    )
    missing_branch_predicates = sum(
        node.get("kind") == "branch" and node.get("branch_kind") == "IF_BEGIN" for node in nodes
    )
    nonmaterialized_access_orders: set[int] = set()
    function_pto = _select_raw_pto_function(pto_text, function) if pto_text is not None else None
    if function_pto is not None:
        nonmaterialized_access_orders = _attach_pto_access_provenance(nodes, function_pto)
        missing_static_loop_bounds = _attach_pto_static_loop_bounds(nodes, function_pto)
        missing_branch_predicates = _attach_pto_branch_predicates(nodes, function_pto)

    sync_groups: list[dict[str, Any]] = []
    sync_edges: list[dict[str, Any]] = []
    omitted_barriers = 0
    nodes_by_id = {node["id"]: node for node in nodes}
    loop_begin_by_end = {
        node["end"]: node["id"]
        for node in nodes
        if node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(node.get("end"), int)
    }
    for sync_index in sorted(sync_operations):
        operations = sync_operations[sync_index]
        active_operations = [operation for operation in operations if not operation["useless"]]
        representative = active_operations[0] if active_operations else operations[0]
        annotated_loop_ends = {
            operation["loop_end"]
            for operation in active_operations
            if isinstance(operation.get("loop_end"), int)
        }
        endpoint_node_ids: set[int] = set()
        for operation in active_operations:
            if operation["type"] in {"set_flag", "wait_flag"}:
                endpoint_node_ids.add(operation["node"])
            elif operation["type"].startswith("pipe_barrier"):
                endpoint_node_ids.add(operation["node"])
                dependency_node = operation.get("dependency_node")
                if isinstance(dependency_node, int):
                    endpoint_node_ids.add(dependency_node)
        endpoint_nodes = [nodes_by_id.get(node_id, {}) for node_id in sorted(endpoint_node_ids)]
        loop_carried = bool(endpoint_nodes) and any(
            all(loop_begin in node.get("loop_stack", []) for node in endpoint_nodes)
            for loop_end in annotated_loop_ends
            if (loop_begin := loop_begin_by_end.get(loop_end)) is not None
        )
        sources = sorted(
            {operation["node"] for operation in active_operations if operation["type"] == "set_flag"}
        )
        targets = sorted(
            {operation["node"] for operation in active_operations if operation["type"] == "wait_flag"}
        )
        barriers = [
            operation for operation in active_operations if operation["type"].startswith("pipe_barrier")
        ]
        for barrier in barriers:
            dependency_node = barrier.get("dependency_node")
            if isinstance(dependency_node, int):
                sources.append(dependency_node)
                targets.append(barrier["node"])
            else:
                omitted_barriers += 1
        sources = sorted(set(sources))
        targets = sorted(set(targets))
        group_id = len(sync_groups)
        sync_groups.append(
            {
                "id": group_id,
                "sync_index": sync_index,
                "src_pipe": representative["src_pipe"],
                "dst_pipe": representative["dst_pipe"],
                "loop_carried": loop_carried,
                "root_buffers": [],
                "operations": operations,
            }
        )
        for source in sources:
            for target in targets:
                if source != target:
                    sync_edges.append(
                        {
                            "source": source,
                            "target": target,
                            "group": group_id,
                            "src_pipe": representative["src_pipe"],
                            "dst_pipe": representative["dst_pipe"],
                            "loop_carried": loop_carried,
                            "root_buffers": [],
                        }
                    )

    operation_nodes = [node for node in nodes if node.get("kind") == "operation"]
    missing_operation_types = sum(
        not (node.get("operation", {}).get("operand_types") or node.get("operation", {}).get("result_types"))
        for node in operation_nodes
    )
    missing_static_work_sizes = sum(
        not node.get("operation", {}).get("static_work_bytes") for node in operation_nodes
    )
    # Current and v0.57 debug dumps print BranchInstanceElements. Older dumps
    # do not, so raw PTO remains the independent fail-closed check that the
    # reconstructed control-flow skeleton is complete.
    branch_nodes_missing = 0
    if function_pto is not None:
        raw_branch_count = sum(
            bool(_PTO_BRANCH_RE.search(line.split("//", maxsplit=1)[0])) for line in function_pto.splitlines()
        )
        exported_if_count = sum(
            node.get("kind") == "branch" and node.get("branch_kind") == "IF_BEGIN" for node in nodes
        )
        branch_nodes_missing = max(0, raw_branch_count - exported_if_count)
    return {
        "schema_version": 1,
        "function": function,
        "status": "analyzed",
        "node_count": len(nodes),
        "duration_model": "unestimated",
        "export_source": (
            "ptoas_debug_import_v0+pto_access_join_v3+static_loop_bounds_v1"
            if pto_text is not None
            else "ptoas_debug_import_v0"
        ),
        "export_limitations": {
            "allocation_sizes_missing": True,
            "operation_types_missing": missing_operation_types,
            "static_work_sizes_missing": missing_static_work_sizes,
            "static_loop_bounds_missing": missing_static_loop_bounds,
            "branch_predicates_missing": missing_branch_predicates,
            "barrier_dependency_nodes_missing": omitted_barriers,
            "branch_nodes_missing": branch_nodes_missing,
            "access_provenance_missing": pto_text is None,
        },
        "nodes": nodes,
        "nonmaterialized_access_orders": sorted(nonmaterialized_access_orders),
        "stream_edges": stream_edges,
        "sync_groups": sync_groups,
        "sync_edges": sync_edges,
    }


def _canonical_operation(name: str) -> str:
    value = name.removeprefix("pto.")
    return "".join(
        "_" if character == "." else character
        for character in value.upper()
        if character.isalnum() or character in {"_", "."}
    )


def _operation_key(pipe: str, name: str) -> str:
    return f"{pipe}:{_canonical_operation(name)}"


def _semantic_operation_text(operation: Mapping[str, Any]) -> str | None:
    """Return SSA-name- and location-independent operation syntax.

    Operand/result types alone do not encode modes such as ``descending`` or
    round/pad attributes. Keep the executable operation text in the complete
    simulator signature, while erasing only unstable SSA identifiers and the
    source location suffix.
    """
    location = operation.get("location")
    if not isinstance(location, str) or not location:
        return None
    # Native PTOAS exports an MLIR source location in this field. The legacy
    # text bridge stores the complete raw operation followed by `` loc(...)``.
    # A bare location contains no execution semantics and must not make an
    # otherwise portable duration signature depend on a checkout path.
    if location.lstrip().startswith("loc("):
        return None
    semantic = location.split(" loc(", maxsplit=1)[0]
    semantic = _PTO_SSA_VALUE_RE.sub("%value", semantic)
    return " ".join(semantic.split())


def _memory_sizes(node: Mapping[str, Any], key: str) -> list[int]:
    entries = node.get(key, [])
    if not isinstance(entries, list):
        return []
    sizes: list[int] = []
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        size = entry.get("allocate_size_bytes")
        root = entry.get("root", "")
        if isinstance(size, int) and size >= 0 and (str(root), size) not in seen:
            seen.add((str(root), size))
            sizes.append(size)
    return sizes


def _work_bytes(node: Mapping[str, Any]) -> int:
    """Estimate the bytes governing one operation's duration."""
    operation = node.get("operation")
    static_work_bytes = operation.get("static_work_bytes", 0) if isinstance(operation, Mapping) else 0
    if not isinstance(static_work_bytes, int) or static_work_bytes < 0:
        raise ValueError(f"operation node {node.get('id')} has invalid static_work_bytes")
    if static_work_bytes:
        return static_work_bytes
    defs = _memory_sizes(node, "defs")
    uses = _memory_sizes(node, "uses")
    pipe = node.get("pipe")
    if pipe in {"PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_MTE4", "PIPE_MTE5", "PIPE_FIX"}:
        def_bytes = sum(defs)
        use_bytes = sum(uses)
        if def_bytes and use_bytes:
            return min(def_bytes, use_bytes)
        return max(def_bytes, use_bytes)
    return max(defs + uses, default=0)


def operation_duration_signature(node: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete static signature used by simulator calibration.

    A family-wide median is deliberately insufficient: transfer and vector
    costs vary materially with shape, type, route, mode, and constants.  A
    calibration producer should copy this mapping from the schedule node it
    measured instead of reconstructing it heuristically from an opcode name.
    """
    pipe = node.get("pipe")
    op_name = node.get("op_name")
    operation = node.get("operation")
    if not isinstance(pipe, str) or not isinstance(op_name, str) or not isinstance(operation, Mapping):
        raise ValueError("duration signature requires pipe, op_name, and operation metadata")
    operand_types = operation.get("operand_types", [])
    result_types = operation.get("result_types", [])
    if (
        not isinstance(operand_types, list)
        or not all(isinstance(item, str) for item in operand_types)
        or not isinstance(result_types, list)
        or not all(isinstance(item, str) for item in result_types)
    ):
        raise ValueError("duration signature requires string operand_types and result_types")
    tiles = [tile for item in [*operand_types, *result_types] if (tile := parse_tile_type(item)) is not None]
    work_tile = tiles[0] if tiles else None
    constants = operation.get("operand_constants", [])
    attributes = operation.get("attributes", {})
    if not isinstance(constants, list) or not isinstance(attributes, Mapping):
        raise ValueError("duration signature has invalid constants or attributes")
    return {
        "pipe": pipe,
        "operation": _canonical_operation(op_name),
        "dtype": work_tile.dtype if work_tile is not None else None,
        "rows": work_tile.rows if work_tile is not None else None,
        "cols": work_tile.cols if work_tile is not None else None,
        "work_bytes": _work_bytes(node),
        "operand_types": operand_types,
        "result_types": result_types,
        "attributes": dict(sorted((str(key), value) for key, value in attributes.items())),
        "operand_constants": constants,
        "semantic_operation": _semantic_operation_text(operation),
    }


def _operation_signature_key(signature: Mapping[str, Any]) -> str:
    required = {"pipe", "operation", "dtype", "rows", "cols", "work_bytes"}
    if not required.issubset(signature):
        missing = sorted(required - set(signature))
        raise ValueError(f"operation signature is missing fields: {missing}")
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _canonical_duration_type(value: str) -> tuple[Any, ...]:
    if tile := parse_tile_type(value):
        return ("tile", tile.scope, tile.dtype, tile.rows, tile.cols)
    return ("other", " ".join(value.split()))


def _duration_signatures_compatible(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Accept only a representational ins/outs spelling difference.

    Raw PTO and native v0.57 MLIR print the same tile types in compact and
    keyed forms. Operation, pipe, shape, work, attributes, constants, and the
    ordered canonical operand/result types must still agree exactly. The
    semantic-operation string is intentionally omitted because it embeds the
    same non-canonical type spelling; all modeled modes are carried separately
    in attributes and operand constants.
    """
    scalar_fields = (
        "pipe",
        "operation",
        "dtype",
        "rows",
        "cols",
        "work_bytes",
        "attributes",
        "operand_constants",
    )
    if any(expected.get(field) != actual.get(field) for field in scalar_fields):
        return False
    for type_field in ("operand_types", "result_types"):
        expected_types = expected.get(type_field)
        actual_types = actual.get(type_field)
        if (
            not isinstance(expected_types, list)
            or not all(isinstance(item, str) for item in expected_types)
            or not isinstance(actual_types, list)
            or not all(isinstance(item, str) for item in actual_types)
        ):
            return False
        if [_canonical_duration_type(item) for item in expected_types] != [
            _canonical_duration_type(item) for item in actual_types
        ]:
            return False
    return True


def _compatible_signature_override(
    signature: Mapping[str, Any], overrides: Mapping[str, float]
) -> tuple[float, str] | None:
    matches: list[tuple[float, str]] = []
    for key, cycles in overrides.items():
        try:
            candidate = json.loads(key)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping) and _duration_signatures_compatible(candidate, signature):
            matches.append((cycles, key))
    if not matches:
        return None
    distinct_cycles = {cycles for cycles, _ in matches}
    if len(distinct_cycles) != 1:
        raise ValueError("ambiguous compatible exact-signature duration overrides")
    return matches[0]


def _loop_multipliers(record: Mapping[str, Any]) -> tuple[dict[int, int], list[int]]:
    loop_counts: dict[int, int] = {}
    dynamic_loops: list[int] = []
    for node in record.get("nodes", []):
        if (
            not isinstance(node, Mapping)
            or node.get("kind") != "loop"
            or node.get("loop_kind") != "LOOP_BEGIN"
        ):
            continue
        node_id = node.get("id")
        trip_count = node.get("static_trip_count")
        if not isinstance(node_id, int):
            continue
        if isinstance(trip_count, int) and trip_count >= 0:
            loop_counts[node_id] = trip_count
        else:
            dynamic_loops.append(node_id)
            loop_counts[node_id] = 1
    return loop_counts, dynamic_loops


def _node_execution_counts(record: Mapping[str, Any]) -> tuple[dict[int, int], list[int]]:
    """Return statically determined executions for every exported schedule node."""
    loop_counts, dynamic_loops = _loop_multipliers(record)
    counts: dict[int, int] = {}
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), int):
            continue
        multiplier = 1
        loop_stack = node.get("loop_stack", [])
        if not isinstance(loop_stack, list) or not all(isinstance(loop, int) for loop in loop_stack):
            raise ValueError(f"schedule node {node['id']} has an invalid loop stack")
        for loop_id in loop_stack:
            multiplier *= loop_counts.get(loop_id, 1)
        counts[node["id"]] = multiplier
    return counts, dynamic_loops


def _pre_codegen_sync_record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Count Final-SyncIR records before SyncCodegen lowering and deduplication.

    Native JSON exports already omit tombstoned records. Legacy text imports
    preserve the ``useless`` marker and exclude those records from active
    counts. Even active records are not emitted-instruction counts: SyncCodegen
    can coalesce identical set/wait records and neighboring barriers.
    """
    execution_counts, dynamic_loops = _node_execution_counts(record)
    if dynamic_loops:
        raise ValueError(
            f"static synchronization counts require bounded loops; dynamic loop nodes: {dynamic_loops[:8]}"
        )

    active_sites_by_type: Counter[str] = Counter()
    active_executions_by_type: Counter[str] = Counter()
    active_sites_by_pipe_pair: Counter[str] = Counter()
    active_executions_by_pipe_pair: Counter[str] = Counter()
    record_count = 0
    useless_record_count = 0
    group_count = 0
    active_group_count = 0
    for group in record.get("sync_groups", []):
        if not isinstance(group, Mapping):
            raise ValueError("schedule synchronization group must be an object")
        operations = group.get("operations")
        if not isinstance(operations, list):
            raise ValueError(f"schedule synchronization group {group.get('id')} has no operations")
        group_count += 1
        group_active = False
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise ValueError(f"schedule synchronization group {group.get('id')} has an invalid operation")
            node_id = operation.get("node")
            kind = operation.get("type")
            source_pipe = operation.get("src_pipe", group.get("src_pipe"))
            target_pipe = operation.get("dst_pipe", group.get("dst_pipe"))
            if (
                not isinstance(node_id, int)
                or node_id not in execution_counts
                or not isinstance(kind, str)
                or not isinstance(source_pipe, str)
                or not isinstance(target_pipe, str)
            ):
                raise ValueError(
                    f"schedule synchronization group {group.get('id')} has incomplete operation metadata"
                )
            record_count += 1
            if operation.get("useless") is True:
                useless_record_count += 1
                continue
            group_active = True
            executions = execution_counts[node_id]
            pipe_pair = f"{source_pipe}->{target_pipe}"
            active_sites_by_type[kind] += 1
            active_executions_by_type[kind] += executions
            active_sites_by_pipe_pair[pipe_pair] += 1
            active_executions_by_pipe_pair[pipe_pair] += executions
        active_group_count += int(group_active)

    return {
        "model_version": "pre_codegen_sync_record_count_v1",
        "group_count": group_count,
        "active_group_count": active_group_count,
        "record_count": record_count,
        "active_record_site_count": sum(active_sites_by_type.values()),
        "useless_record_site_count": useless_record_count,
        "active_record_execution_count": sum(active_executions_by_type.values()),
        "active_record_sites_by_type": dict(sorted(active_sites_by_type.items())),
        "active_record_executions_by_type": dict(sorted(active_executions_by_type.items())),
        "active_record_sites_by_pipe_pair": dict(sorted(active_sites_by_pipe_pair.items())),
        "active_record_executions_by_pipe_pair": dict(sorted(active_executions_by_pipe_pair.items())),
    }


def _latency_graph_completeness(
    record: Mapping[str, Any],
    edge_diagnostics: Mapping[str, int],
    loop_sync_models: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    modeled_loop_carried = sum(len(model.get("loop_carried_recurrences", [])) for model in loop_sync_models)
    non_cycle_loop_carried = sum(
        recurrence.get("cycles") is None
        for model in loop_sync_models
        for recurrence in model.get("loop_carried_recurrences", [])
        if isinstance(recurrence, Mapping)
    )
    excluded_loop_carried = edge_diagnostics.get("excluded_loop_carried_sync_edges", 0)
    unresolved_loop_carried = max(0, excluded_loop_carried - modeled_loop_carried)
    limitations = [
        name
        for name in (
            "excluded_non_operation_stream_edges",
            "excluded_non_operation_sync_edges",
        )
        if edge_diagnostics.get(name, 0) > 0
    ]
    if modeled_loop_carried != excluded_loop_carried:
        limitations.append("unresolved_loop_carried_sync_edges")
    export_limitations = record.get("export_limitations", {})
    if isinstance(export_limitations, Mapping) and export_limitations.get(
        "barrier_dependency_nodes_missing", 0
    ):
        limitations.append("export_limitations.barrier_dependency_nodes_missing")
    if isinstance(export_limitations, Mapping) and export_limitations.get("branch_nodes_missing", 0):
        limitations.append("export_limitations.branch_nodes_missing")
    return {
        "latency_graph_complete": not limitations,
        "latency_graph_limitations": limitations,
        "modeled_loop_carried_sync_edges": modeled_loop_carried,
        "unresolved_loop_carried_sync_edges": unresolved_loop_carried,
        "non_cycle_loop_carried_sync_edges": non_cycle_loop_carried,
    }


def _propagate_barrier_dependency_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize exported barrier dependency nodes as public sync edges.

    PTOAS attaches the operation drained by a barrier to the barrier operation
    record.  Older consumers looked only at ``sync_edges`` and therefore had
    to reach into private campaign helpers to recover this provenance.  Keep
    the public evaluator fail-closed, but make the exported operation field the
    authoritative source when it is present.
    """
    normalized = copy.deepcopy(dict(record))
    nodes = {
        node["id"]
        for node in normalized.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    raw_edges = normalized.get("sync_edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError("schedule sync_edges must be an array")
    edges = [dict(edge) for edge in raw_edges if isinstance(edge, Mapping)]
    existing = {(edge.get("source"), edge.get("target"), edge.get("group")) for edge in edges}
    recovered = 0
    missing = 0
    barrier_sites = 0
    for group in normalized.get("sync_groups", []):
        if not isinstance(group, Mapping):
            continue
        group_id = group.get("id")
        for operation in group.get("operations", []):
            if (
                not isinstance(operation, Mapping)
                or operation.get("useless") is True
                or not str(operation.get("type", "")).startswith("pipe_barrier")
            ):
                continue
            barrier_sites += 1
            source = operation.get("dependency_node")
            target = operation.get("node")
            if (
                not isinstance(source, int)
                or not isinstance(target, int)
                or source not in nodes
                or target not in nodes
            ):
                missing += 1
                continue
            key = (source, target, group_id)
            if key in existing:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "group": group_id,
                    "src_pipe": operation.get("src_pipe", group.get("src_pipe")),
                    "dst_pipe": operation.get("dst_pipe", group.get("dst_pipe")),
                    "loop_carried": bool(group.get("loop_carried", False)),
                    "root_buffers": list(group.get("root_buffers", [])),
                }
            )
            existing.add(key)
            recovered += 1
    normalized["sync_edges"] = edges
    limitations = dict(normalized.get("export_limitations", {}))
    if barrier_sites == 0:
        prior_missing = limitations.get("barrier_dependency_nodes_missing", 0)
        if isinstance(prior_missing, int) and prior_missing > 0:
            missing = prior_missing
    limitations["barrier_dependency_nodes_missing"] = missing
    normalized["export_limitations"] = limitations
    normalized["barrier_dependency_provenance"] = {
        "source": "sync_groups.operations.dependency_node",
        "barrier_site_count": barrier_sites,
        "recovered_sync_edge_count": recovered,
        "missing_dependency_node_count": missing,
    }
    return normalized


def estimate_node_durations(
    record: Mapping[str, Any], model: DurationModel
) -> tuple[dict[int, float], dict[int, dict[str, Any]], list[int]]:
    """Estimate operation durations, including aggregate static-loop work."""
    loop_counts, dynamic_loops = _loop_multipliers(record)
    durations: dict[int, float] = {}
    provenance: dict[int, dict[str, Any]] = {}
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("kind") != "operation":
            continue
        node_id = node.get("id")
        pipe = node.get("pipe")
        op_name = node.get("op_name")
        if not isinstance(node_id, int) or not isinstance(pipe, str) or not isinstance(op_name, str):
            raise ValueError("operation node must have integer id, pipe, and op_name")

        key = _operation_key(pipe, op_name)
        signature = operation_duration_signature(node) if model.operation_signature_cycles else None
        signature_key = _operation_signature_key(signature) if signature is not None else None
        compatible_override = (
            _compatible_signature_override(signature, model.operation_signature_cycles)
            if signature is not None and signature_key not in model.operation_signature_cycles
            else None
        )
        work_bytes = _work_bytes(node)
        if signature_key is not None and signature_key in model.operation_signature_cycles:
            base = model.operation_signature_cycles[signature_key]
            source = "simulator_complete_signature_median"
            detail = f"complete operation signature {signature_key}"
            fallback = False
        elif compatible_override is not None:
            base, matched_key = compatible_override
            source = "simulator_complete_signature_compatible_encoding"
            detail = f"compatible complete operation signature {matched_key}"
            fallback = False
        elif not model.operation_signature_cycles and key in model.operation_cycles:
            base = model.operation_cycles[key]
            source = "simulator_operation_median"
            detail = f"explicit operation override {key}"
            fallback = False
        elif model.pto_isa_provider is not None:
            estimate = model.pto_isa_provider.estimate(node, work_bytes=work_bytes)
            base = estimate.cycles
            source = estimate.source
            detail = estimate.detail
            fallback = estimate.fallback
        elif model.operation_signature_cycles:
            raise ValueError(
                f"no exact-signature duration estimate for {op_name} on {pipe}; "
                "configure the pinned PTO-ISA provider or add this complete signature"
            )
        else:
            parameters = model.pipe_parameters.get(pipe)
            if parameters is None:
                raise ValueError(
                    f"no duration estimate for {op_name} on {pipe}; configure a pinned PTO-ISA "
                    "provider, an operation override, or an explicit legacy pipe model"
                )
            transfer = (
                0.0 if math.isinf(parameters.bytes_per_cycle) else work_bytes / parameters.bytes_per_cycle
            )
            base = max(parameters.minimum_cycles, parameters.startup_cycles + transfer)
            source = "legacy_pipe_size_model"
            detail = (
                f"startup={parameters.startup_cycles}; bytes_per_cycle={parameters.bytes_per_cycle}; "
                f"minimum={parameters.minimum_cycles}"
            )
            fallback = True

        multiplier = 1
        for loop_id in node.get("loop_stack", []):
            if isinstance(loop_id, int):
                multiplier *= loop_counts.get(loop_id, 1)
        duration = base * multiplier
        durations[node_id] = duration
        provenance[node_id] = {
            "pipe": pipe,
            "op_name": op_name,
            "operation_key": key,
            "work_bytes": work_bytes,
            "base_cycles": base,
            "loop_multiplier": multiplier,
            "cycles": duration,
            "source": source,
            "detail": detail,
            "fallback": fallback,
        }
    return durations, provenance, dynamic_loops


def _prepare_control_flow_record(  # noqa: PLR0912 - structured marker state machine
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild per-pipe issue order through structured branches and loops.

    PTOAS v0.57's native exporter links operations by pipe while walking the
    linear SyncIR.  That accidentally serializes the mutually exclusive then
    and else arms.  A control marker also belongs to every affected pipe: a
    wait at ``IF_BEGIN`` gates both arms, while a set at ``IF_END`` depends on
    the taken arm only.  Represent each ``(marker, pipe)`` pair by a private
    zero-duration node so a control-flow join never creates an undocumented
    cross-pipe dependency.
    """
    original_nodes = [node for node in record.get("nodes", []) if isinstance(node, Mapping)]
    operation_pipes: set[str] = set()
    for node in original_nodes:
        pipe = node.get("pipe")
        if node.get("kind") == "operation" and isinstance(pipe, str):
            operation_pipes.add(pipe)
    sync_pipes: set[str] = set()
    for edge in record.get("sync_edges", []):
        if not isinstance(edge, Mapping):
            continue
        for key in ("src_pipe", "dst_pipe"):
            pipe = edge.get(key)
            if isinstance(pipe, str):
                sync_pipes.add(pipe)
    pipes = sorted(operation_pipes | sync_pipes)
    if not pipes:
        return dict(record)

    synthetic_nodes: list[dict[str, Any]] = []
    endpoint_by_marker_pipe: dict[tuple[int, str], int] = {}
    stream_edges: list[dict[str, Any]] = []
    seen_stream_edges: set[tuple[int, int, str]] = set()
    frontiers: dict[str, set[int]] = {pipe: set() for pipe in pipes}
    branch_frames: list[dict[str, Any]] = []
    next_synthetic_id = -1

    def connect(source: int, target: int, pipe: str) -> None:
        key = (source, target, pipe)
        if source != target and key not in seen_stream_edges:
            stream_edges.append({"source": source, "target": target, "pipe": pipe})
            seen_stream_edges.add(key)

    def add_marker(node: Mapping[str, Any], predecessors: Mapping[str, set[int]]) -> None:
        nonlocal next_synthetic_id
        node_id = node.get("id")
        if not isinstance(node_id, int):
            raise ValueError("schedule control-flow marker has no integer id")
        for pipe in pipes:
            synthetic_id = next_synthetic_id
            next_synthetic_id -= 1
            endpoint_by_marker_pipe[(node_id, pipe)] = synthetic_id
            synthetic_nodes.append(
                {
                    "id": synthetic_id,
                    "kind": "control_point",
                    "control_kind": node.get("kind"),
                    "control_subkind": node.get("branch_kind", node.get("loop_kind")),
                    "origin_node": node_id,
                    "pipe": pipe,
                    "loop_stack": list(node.get("loop_stack", [])),
                    "branch_stack": list(node.get("branch_stack", [])),
                }
            )
            for predecessor in sorted(predecessors[pipe]):
                connect(predecessor, synthetic_id, pipe)
            frontiers[pipe] = {synthetic_id}

    for node in original_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, int):
            continue
        kind = node.get("kind")
        if kind == "operation":
            pipe = node.get("pipe")
            if not isinstance(pipe, str) or pipe not in frontiers:
                raise ValueError(f"schedule operation {node_id} has an invalid pipe")
            for predecessor in sorted(frontiers[pipe]):
                connect(predecessor, node_id, pipe)
            frontiers[pipe] = {node_id}
            continue

        if kind == "branch":
            branch_kind = node.get("branch_kind")
            if branch_kind == "IF_BEGIN":
                add_marker(node, frontiers)
                branch_frames.append(
                    {
                        "begin": node_id,
                        "entry": {pipe: set(values) for pipe, values in frontiers.items()},
                        "then": None,
                        "else_seen": False,
                    }
                )
                continue
            if not branch_frames:
                raise ValueError(f"schedule branch marker {node_id} has no open IF_BEGIN")
            frame = branch_frames[-1]
            if node.get("begin") != frame["begin"]:
                raise ValueError(f"schedule branch marker {node_id} does not match IF_BEGIN {frame['begin']}")
            if branch_kind == "ELSE_BEGIN":
                frame["then"] = {pipe: set(values) for pipe, values in frontiers.items()}
                frame["else_seen"] = True
                frontiers = {pipe: set(values) for pipe, values in frame["entry"].items()}
                add_marker(node, frontiers)
                continue
            if branch_kind == "IF_END":
                branch_frames.pop()
                if frame["else_seen"]:
                    alternatives = (frame["then"], frontiers)
                else:
                    alternatives = (frontiers, frame["entry"])
                merged = {
                    pipe: set().union(*(alternative[pipe] for alternative in alternatives)) for pipe in pipes
                }
                add_marker(node, merged)
                continue
            raise ValueError(f"schedule branch marker {node_id} has invalid kind {branch_kind!r}")

        # Loop and placeholder markers are sequential control points. Static
        # loop work is already multiplied in operation durations; the markers
        # make entry/exit synchronization endpoints joinable without adding a
        # recurrence to the acyclic whole-function graph.
        add_marker(node, frontiers)

    if branch_frames:
        raise ValueError(f"schedule has unterminated IF_BEGIN {branch_frames[-1]['begin']}")

    nodes_by_id = {node["id"]: node for node in original_nodes if isinstance(node.get("id"), int)}

    def effective_endpoint(node_id: Any, pipe: Any) -> Any:
        if not isinstance(node_id, int) or not isinstance(pipe, str):
            return node_id
        node = nodes_by_id.get(node_id)
        if node is None or node.get("kind") == "operation":
            return node_id
        return endpoint_by_marker_pipe.get((node_id, pipe), node_id)

    sync_edges: list[dict[str, Any]] = []
    for edge in record.get("sync_edges", []):
        if not isinstance(edge, Mapping):
            continue
        transformed = dict(edge)
        transformed["source"] = effective_endpoint(edge.get("source"), edge.get("src_pipe"))
        transformed["target"] = effective_endpoint(edge.get("target"), edge.get("dst_pipe"))
        transformed["source_origin"] = edge.get("source")
        transformed["target_origin"] = edge.get("target")
        sync_edges.append(transformed)

    prepared = dict(record)
    prepared["nodes"] = [dict(node) for node in original_nodes] + synthetic_nodes
    prepared["stream_edges"] = stream_edges
    prepared["sync_edges"] = sync_edges
    prepared["control_flow_graph_version"] = "per_pipe_structured_control_v1"
    prepared["control_point_nodes"] = synthetic_nodes
    return prepared


def _schedule_graph_durations(
    record: Mapping[str, Any], operation_durations: Mapping[int, float]
) -> dict[int, float]:
    """Add zero-duration structural markers to operation durations."""
    durations = dict(operation_durations)
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, int):
            continue
        if node_id in durations:
            continue
        if node.get("kind") in {"loop", "branch", "placeholder", "control_point"}:
            durations[node_id] = 0.0
    return durations


def _stream_graph_edges(
    record: Mapping[str, Any], node_ids: set[int]
) -> tuple[list[tuple[int, int, float, str, int | None]], int]:
    edges: list[tuple[int, int, float, str, int | None]] = []
    seen: set[tuple[int, int, str, int | None]] = set()
    excluded = 0
    for edge in record.get("stream_edges", []):
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        key = (source, target, "stream", None)
        if source in node_ids and target in node_ids and key not in seen:
            edges.append((source, target, 0.0, "stream", None))
            seen.add(key)
        elif source not in node_ids or target not in node_ids:
            excluded += 1
    return edges, excluded


def _sync_graph_edges(
    record: Mapping[str, Any], node_ids: set[int]
) -> tuple[list[tuple[int, int, float, str, int | None]], dict[str, int]]:
    edges: list[tuple[int, int, float, str, int | None]] = []
    seen: set[tuple[int, int, str, int | None]] = set()
    resolved_loop_carried = _effective_loop_carried_edge_indices(record)
    declared_loop_carried = 0
    excluded_loop_carried = 0
    reclassified_non_recurrence = 0
    excluded_non_operation_sync_edges = 0
    excluded_sentinel_sync_edges = 0
    for edge_index, edge in enumerate(record.get("sync_edges", [])):
        if not isinstance(edge, Mapping):
            continue
        if edge.get("loop_carried"):
            declared_loop_carried += 1
        if edge_index in resolved_loop_carried:
            excluded_loop_carried += 1
            continue
        if edge.get("loop_carried"):
            reclassified_non_recurrence += 1
        source, target, group = edge.get("source"), edge.get("target"), edge.get("group")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        group_id = group if isinstance(group, int) else None
        key = (source, target, "sync", group_id)
        if source in node_ids and target in node_ids and source != target and key not in seen:
            edges.append((source, target, 0.0, "sync", group_id))
            seen.add(key)
        elif source not in node_ids or target not in node_ids:
            # PTOAS uses dependency node zero as the source of the final
            # PIPE_ALL tail barrier when the function has no real node zero.
            if (source == 0 and source not in node_ids) or (target == 0 and target not in node_ids):
                excluded_sentinel_sync_edges += 1
            else:
                excluded_non_operation_sync_edges += 1
    return edges, {
        "excluded_loop_carried_sync_edges": excluded_loop_carried,
        "declared_loop_carried_sync_edges": declared_loop_carried,
        "reclassified_non_recurrence_sync_edges": reclassified_non_recurrence,
        "excluded_non_operation_sync_edges": excluded_non_operation_sync_edges,
        "excluded_sentinel_sync_edges": excluded_sentinel_sync_edges,
    }


def _graph_edges(
    record: Mapping[str, Any], node_ids: set[int], *, include_sync: bool
) -> tuple[list[tuple[int, int, float, str, int | None]], dict[str, int]]:
    edges, excluded_stream_edges = _stream_graph_edges(record, node_ids)
    diagnostics = {
        "excluded_loop_carried_sync_edges": 0,
        "declared_loop_carried_sync_edges": 0,
        "reclassified_non_recurrence_sync_edges": 0,
        "excluded_non_operation_stream_edges": excluded_stream_edges,
        "excluded_non_operation_sync_edges": 0,
        "excluded_sentinel_sync_edges": 0,
    }
    if include_sync:
        sync_edges, sync_diagnostics = _sync_graph_edges(record, node_ids)
        edges.extend(sync_edges)
        diagnostics.update(sync_diagnostics)
    return edges, diagnostics


def _longest_path(
    durations: Mapping[int, float],
    edges: Iterable[tuple[int, int, float, str, int | None]],
) -> tuple[float, dict[int, float], dict[int, float], list[int]]:
    """Return makespan, forward/backward distances, and one critical path."""
    successors: dict[int, list[tuple[int, float]]] = {node: [] for node in durations}
    predecessors: dict[int, list[tuple[int, float]]] = {node: [] for node in durations}
    indegree = {node: 0 for node in durations}
    for source, target, latency, _, _ in edges:
        successors[source].append((target, latency))
        predecessors[target].append((source, latency))
        indegree[target] += 1

    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for target, _ in sorted(successors[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(durations):
        cyclic = sorted(node for node, degree in indegree.items() if degree)
        raise ValueError(f"schedule graph is cyclic after excluding loop-carried edges: {cyclic[:8]}")

    forward: dict[int, float] = {}
    parent: dict[int, int | None] = {}
    for node in order:
        best = 0.0
        best_parent: int | None = None
        for predecessor, latency in predecessors[node]:
            candidate = forward[predecessor] + latency
            if candidate > best:
                best = candidate
                best_parent = predecessor
        forward[node] = best + durations[node]
        parent[node] = best_parent

    backward: dict[int, float] = {}
    for node in reversed(order):
        tail = max((latency + backward[target] for target, latency in successors[node]), default=0.0)
        backward[node] = durations[node] + tail

    if not order:
        return 0.0, forward, backward, []
    end = max(order, key=lambda node: forward[node])
    makespan = forward[end]
    path: list[int] = []
    cursor: int | None = end
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return makespan, forward, backward, path


def _expanded_node_loop_stack(
    node: Mapping[str, Any], original_nodes: Mapping[int, Mapping[str, Any]]
) -> tuple[int, ...]:
    """Return the static-loop coordinates that identify one dynamic node.

    Operations and branch control points already carry their enclosing loop
    stack.  A loop begin/end control point is executed once per iteration of
    that loop, however, while PTOAS records the marker outside the loop's own
    stack.  Add that final coordinate explicitly so the expanded graph can
    connect iteration ``i`` to ``i + 1`` without guessing from node order.
    """
    raw_stack = node.get("loop_stack", [])
    if not isinstance(raw_stack, list) or not all(isinstance(loop, int) for loop in raw_stack):
        raise ValueError(f"schedule node {node.get('id')} has an invalid loop stack")
    stack = list(raw_stack)
    if node.get("kind") != "control_point" or node.get("control_kind") != "loop":
        return tuple(stack)

    origin = node.get("origin_node")
    original = original_nodes.get(origin) if isinstance(origin, int) else None
    if original is None:
        raise ValueError(f"loop control point {node.get('id')} has no original loop marker")
    loop_kind = node.get("control_subkind")
    loop_id = origin if loop_kind == "LOOP_BEGIN" else original.get("begin")
    if loop_kind not in {"LOOP_BEGIN", "LOOP_END"} or not isinstance(loop_id, int):
        raise ValueError(f"loop control point {node.get('id')} has invalid loop identity")
    stack.append(loop_id)
    return tuple(stack)


def _expanded_edge_target_context(
    source_stack: tuple[int, ...],
    source_context: tuple[int, ...],
    target_stack: tuple[int, ...],
    loop_counts: Mapping[int, int],
    *,
    recurrence_loop: int | None,
) -> tuple[int, ...] | None:
    """Map one source occurrence to the corresponding target occurrence."""
    source_values = dict(zip(source_stack, source_context, strict=True))
    target_values: list[int] = []
    for loop in target_stack:
        if loop in source_values:
            value = source_values[loop]
            if loop == recurrence_loop:
                value += 1
                if value >= loop_counts[loop]:
                    return None
            target_values.append(value)
        else:
            # Entering a nested or following loop always targets its first
            # dynamic iteration.
            target_values.append(0)

    for loop, value in source_values.items():
        if loop in target_stack:
            continue
        # An edge leaving a loop is enabled only by its final occurrence.
        if value != loop_counts[loop] - 1:
            return None
    if recurrence_loop is not None and recurrence_loop not in source_values:
        raise ValueError(f"loop-carried edge source is not inside loop {recurrence_loop}")
    if recurrence_loop is not None and recurrence_loop not in target_stack:
        raise ValueError(f"loop-carried edge target is not inside loop {recurrence_loop}")
    return tuple(target_values)


def _compact_expanded_path(
    path: Sequence[int], clone_provenance: Mapping[int, tuple[int, tuple[int, ...]]]
) -> dict[str, Any]:
    """Keep an auditable but bounded representation of a dynamic path."""

    def decode(clone: int) -> dict[str, Any]:
        node, iterations = clone_provenance[clone]
        return {"node": node, "iterations": list(iterations)}

    limit = 32
    truncated = len(path) > 2 * limit
    return {
        "node_count": len(path),
        "head": [decode(clone) for clone in path[:limit]],
        "tail": [decode(clone) for clone in path[-limit:]] if truncated else [],
        "truncated": truncated,
    }


def _branch_alternatives(record: Mapping[str, Any]) -> tuple[list[int], dict[int, tuple[int, bool]]]:
    """Return canonical predicate identities and branch-marker requirements.

    A final SyncIR trace identifies structured IF regions but does not, by
    itself, say when two IFs re-test the same scalar predicate.  The raw-PTO
    semantic bridge annotates such IF_BEGIN nodes with ``predicate_identity``
    and the polarity under which their THEN region executes.  Collapse those
    re-tests to one scenario bit.  Native records without the annotation keep
    the conservative historical behavior of one independent bit per IF.
    """
    branch_ids: list[int] = []
    markers: dict[int, tuple[int, bool]] = {}
    representative_by_predicate: dict[str, int] = {}
    if_marker_requirements: dict[int, tuple[int, bool]] = {}

    # Establish the scenario variable represented by every IF marker first;
    # ELSE_BEGIN records refer back to these ids and need the same canonical
    # identity even if they appear later in the node sequence.
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("kind") != "branch":
            continue
        node_id = node.get("id")
        branch_kind = node.get("branch_kind")
        if not isinstance(node_id, int) or branch_kind != "IF_BEGIN":
            continue
        predicate_identity = node.get("predicate_identity")
        predicate_true_value = node.get("predicate_true_value", True)
        iteration_profile = node.get("predicate_iteration_profile")
        if isinstance(iteration_profile, Mapping):
            # The profile records the actual boolean consumed by this IF for
            # every static loop occurrence. Keep the marker local to this IF;
            # it is not a free scenario bit and does not need polarity
            # canonicalization through a materialized boolean alias.
            requirement = (node_id, True)
            if_marker_requirements[node_id] = requirement
            markers[node_id] = requirement
            continue
        if not isinstance(predicate_identity, str):
            predicate_identity = f"independent-if:{node_id}"
        if not isinstance(predicate_true_value, bool):
            raise ValueError(f"IF_BEGIN marker {node_id} has invalid predicate polarity")
        representative = representative_by_predicate.get(predicate_identity)
        if representative is None:
            representative = node_id
            representative_by_predicate[predicate_identity] = representative
            branch_ids.append(representative)
        requirement = (representative, predicate_true_value)
        if_marker_requirements[node_id] = requirement
        markers[node_id] = requirement

    for node in record.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("kind") != "branch":
            continue
        node_id = node.get("id")
        begin = node.get("begin")
        if (
            not isinstance(node_id, int)
            or not isinstance(begin, int)
            or node.get("branch_kind") != "ELSE_BEGIN"
        ):
            continue
        requirement = if_marker_requirements.get(begin)
        if requirement is None:
            raise ValueError(f"ELSE_BEGIN marker {node_id} references unknown IF_BEGIN {begin}")
        representative, then_value = requirement
        markers[node_id] = (representative, not then_value)
    return sorted(branch_ids), markers


def _branch_iteration_profiles(
    record: Mapping[str, Any],
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...], dict[int, bool]]]:
    """Index exact per-iteration branch values by their IF marker."""
    profiles: dict[int, tuple[tuple[int, ...], tuple[int, ...], dict[int, bool]]] = {}
    for node in record.get("nodes", []):
        if (
            not isinstance(node, Mapping)
            or node.get("kind") != "branch"
            or node.get("branch_kind") != "IF_BEGIN"
            or not isinstance(node.get("id"), int)
        ):
            continue
        raw_profile = node.get("predicate_iteration_profile")
        if raw_profile is None:
            continue
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"IF_BEGIN marker {node['id']} has an invalid iteration profile")
        loop_ids = raw_profile.get("loop_ids")
        counts = raw_profile.get("iteration_counts")
        active_flat_indices = raw_profile.get("active_flat_indices")
        values = raw_profile.get("values")
        if (
            not isinstance(loop_ids, list)
            or not all(isinstance(loop, int) for loop in loop_ids)
            or not isinstance(counts, list)
            or not all(isinstance(count, int) and count >= 0 for count in counts)
            or len(loop_ids) != len(counts)
            or not isinstance(values, list)
            or not all(isinstance(value, bool) for value in values)
        ):
            raise ValueError(f"IF_BEGIN marker {node['id']} has a malformed iteration profile")
        context_count = math.prod(counts)
        if active_flat_indices is None:
            active_flat_indices = list(range(context_count))
        if (
            not isinstance(active_flat_indices, list)
            or not all(isinstance(index, int) for index in active_flat_indices)
            or active_flat_indices != sorted(set(active_flat_indices))
            or any(index < 0 or index >= context_count for index in active_flat_indices)
            or len(values) != len(active_flat_indices)
        ):
            raise ValueError(f"IF_BEGIN marker {node['id']} has malformed active occurrences")
        profiles[int(node["id"])] = (
            tuple(loop_ids),
            tuple(counts),
            dict(zip(active_flat_indices, values, strict=True)),
        )
    return profiles


def _branch_value_for_context(
    branch: int,
    choices: Mapping[int, bool],
    profiles: Mapping[int, tuple[tuple[int, ...], tuple[int, ...], Mapping[int, bool]]],
    context: Mapping[int, int],
) -> bool:
    """Resolve one scenario or exact mixed-iteration branch value."""
    profile = profiles.get(branch)
    if profile is None:
        if branch not in choices:
            raise ValueError(f"structured branch {branch} has neither a scenario choice nor a profile")
        return choices[branch]
    loop_ids, counts, values = profile
    flat_index = 0
    for loop_id, count in zip(loop_ids, counts, strict=True):
        iteration = context.get(loop_id)
        if iteration is None or iteration < 0 or iteration >= count:
            raise ValueError(
                f"structured branch {branch} has no iteration value for loop {loop_id}: "
                f"context={dict(context)}"
            )
        flat_index = flat_index * count + iteration
    if flat_index not in values:
        raise ValueError(f"structured branch {branch} is inactive in loop context {dict(context)}")
    return values[flat_index]


def _node_active_in_branch_scenario(
    node: Mapping[str, Any],
    choices: Mapping[int, bool],
    markers: Mapping[int, tuple[int, bool]],
    profiles: Mapping[int, tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...]]] | None = None,
    context: Mapping[int, int] | None = None,
) -> bool:
    """Return whether a structured node executes in one concrete loop context."""
    profiles = profiles or {}
    context = context or {}
    kind = node.get("kind")
    subkind = node.get("branch_kind") if kind == "branch" else node.get("control_subkind")
    marker_id = node.get("id") if kind == "branch" else node.get("origin_node")
    own_else: tuple[int, bool] | None = None
    if subkind == "ELSE_BEGIN" and isinstance(marker_id, int):
        own_else = markers.get(marker_id)
        if own_else is None:
            raise ValueError(f"ELSE_BEGIN marker {marker_id} has no branch identity")
    stack = node.get("branch_stack", [])
    if not isinstance(stack, list) or not all(isinstance(marker, int) for marker in stack):
        raise ValueError(f"schedule node {node.get('id')} has an invalid branch stack")
    for marker in stack:
        alternative = markers.get(marker)
        if alternative is None:
            raise ValueError(f"schedule node {node.get('id')} references unknown branch marker {marker}")
        begin, expected = alternative
        if own_else is not None and begin == own_else[0]:
            continue
        if _branch_value_for_context(begin, choices, profiles, context) != expected:
            return False
    if own_else is not None:
        begin, expected = own_else
        if _branch_value_for_context(begin, choices, profiles, context) != expected:
            return False
    return True


def _node_branch_requirements(
    node: Mapping[str, Any], markers: Mapping[int, tuple[int, bool]]
) -> dict[int, bool]:
    """Return the structured branch choices required to execute ``node``.

    Candidate reuse edges are meaningful only in scenarios where both access
    endpoints execute.  Keep that predicate on the edge catalog instead of
    treating mutually exclusive branch arms as a synchronization demand.
    """
    requirements: dict[int, bool] = {}
    stack = node.get("branch_stack", [])
    if not isinstance(stack, list) or not all(isinstance(marker, int) for marker in stack):
        raise ValueError(f"schedule node {node.get('id')} has an invalid branch stack")
    for marker in stack:
        alternative = markers.get(marker)
        if alternative is None:
            raise ValueError(f"schedule node {node.get('id')} references unknown branch marker {marker}")
        branch, value = alternative
        previous = requirements.get(branch)
        if previous is not None and previous != value:
            raise ValueError(f"schedule node {node.get('id')} has contradictory branch requirements")
        requirements[branch] = value
    return requirements


def _combined_branch_requirements(
    source: Mapping[str, Any], target: Mapping[str, Any], markers: Mapping[int, tuple[int, bool]]
) -> dict[int, bool] | None:
    """Return the predicate under which both endpoints execute, or ``None``."""
    combined = _node_branch_requirements(source, markers)
    for branch, value in _node_branch_requirements(target, markers).items():
        previous = combined.get(branch)
        if previous is not None and previous != value:
            return None
        combined[branch] = value
    return combined


def _structured_operation_occurrences(
    record: Mapping[str, Any],
    clone_by_context: Mapping[int, Mapping[tuple[int, ...], int]],
    stacks: Mapping[int, tuple[int, ...]],
    loop_counts: Mapping[int, int],
    active_clones: Mapping[int, bool],
) -> list[tuple[int, Mapping[str, Any]]]:
    """Return operation clones in structured execution order.

    This traversal expands the original, pre-InsertSync operation stream rather
    than inferring program order from physical addresses or synchronization
    records. Callers select a concrete branch scenario through ``active_nodes``;
    structural branch markers themselves do not represent memory accesses.
    """
    nodes = [node for node in record.get("nodes", []) if isinstance(node, Mapping)]
    positions = {node["id"]: index for index, node in enumerate(nodes) if isinstance(node.get("id"), int)}
    occurrences: list[tuple[int, Mapping[str, Any]]] = []

    def visit(start: int, stop: int, context: dict[int, int]) -> None:
        index = start
        while index < stop:
            node = nodes[index]
            node_id = node.get("id")
            if not isinstance(node_id, int):
                index += 1
                continue
            if node.get("kind") == "branch":
                index += 1
                continue
            if node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_BEGIN":
                end_id = node.get("end")
                end_index = positions.get(end_id) if isinstance(end_id, int) else None
                if end_index is None or end_index <= index or end_index >= stop:
                    raise ValueError(f"static loop {node_id} has an invalid end marker {end_id}")
                count = loop_counts.get(node_id)
                if count is None:
                    raise ValueError(f"static loop {node_id} has no resolved trip count")
                for iteration in range(count):
                    visit(index + 1, end_index, {**context, node_id: iteration})
                index = end_index + 1
                continue
            if node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_END":
                raise ValueError(f"unexpected loop-end marker {node_id} in structured traversal")
            if node.get("kind") == "operation":
                stack = stacks.get(node_id)
                if stack is None:
                    raise ValueError(f"operation {node_id} has no expanded loop stack")
                try:
                    occurrence_context = tuple(context[loop] for loop in stack)
                except KeyError as error:
                    raise ValueError(
                        f"operation {node_id} has an incomplete loop context for stack {stack}"
                    ) from error
                try:
                    clone = clone_by_context[node_id][occurrence_context]
                except KeyError as error:
                    raise ValueError(
                        f"operation {node_id} has no clone for loop context {occurrence_context}"
                    ) from error
                if active_clones.get(clone, False):
                    occurrences.append((clone, node))
            index += 1

    visit(0, len(nodes), {})
    return occurrences


def _expanded_logical_memory_edges(
    occurrences: Sequence[tuple[int, Mapping[str, Any]]],
    synchronization_latency_cycles: float,
) -> list[tuple[int, int, float, str, int | None]]:
    """Derive no-reuse RAW/WAR/WAW dependencies from logical allocation roots.

    A dependency crossing execution pipes already requires synchronization in
    the non-reusing program, so it carries the same calibrated synchronization
    latency as a placement-induced cross-pipe dependency. Same-pipe ordering is
    provided by the FIFO stream and therefore carries no extra edge latency.
    """

    def roots(node: Mapping[str, Any], field: str) -> set[str]:
        accesses = node.get(field, [])
        if not isinstance(accesses, list):
            raise ValueError(f"schedule node {node.get('id')} has invalid {field} metadata")
        result: set[str] = set()
        for access in accesses:
            if not isinstance(access, Mapping):
                raise ValueError(f"schedule node {node.get('id')} has a non-object {field} entry")
            root = access.get("root")
            if not isinstance(root, str) or not root:
                raise ValueError(f"schedule node {node.get('id')} has a {field} entry without a root")
            result.add(root)
        return result

    last_writer: dict[str, int] = {}
    readers_since_write: dict[str, set[int]] = defaultdict(set)
    dependencies: set[tuple[int, int, str]] = set()
    pipe_by_clone = {
        clone: node.get("pipe") for clone, node in occurrences if isinstance(node.get("pipe"), str)
    }
    for clone, node in occurrences:
        read_roots = roots(node, "uses")
        write_roots = roots(node, "defs")
        for root in read_roots:
            writer = last_writer.get(root)
            if writer is not None and writer != clone:
                dependencies.add((writer, clone, "ssa_raw"))
        for root in write_roots:
            writer = last_writer.get(root)
            if writer is not None and writer != clone:
                dependencies.add((writer, clone, "ssa_waw"))
            dependencies.update(
                (reader, clone, "ssa_war") for reader in readers_since_write[root] if reader != clone
            )
        for root in write_roots:
            last_writer[root] = clone
            readers_since_write[root].clear()
        for root in read_roots - write_roots:
            readers_since_write[root].add(clone)
    return [
        (
            source,
            target,
            (
                synchronization_latency_cycles
                if pipe_by_clone.get(source) != pipe_by_clone.get(target)
                else 0.0
            ),
            kind,
            None,
        )
        for source, target, kind in sorted(dependencies)
    ]


def _pipe_barrier_sites(record: Mapping[str, Any]) -> list[tuple[int, str, int]]:
    """Return unique lowered barrier sites as ``(target, pipe, group)``.

    Final-SyncIR may carry duplicate records that SyncCodegen coalesces at one
    operation site.  The queue model therefore keys a physical barrier by its
    target operation and pipe, retaining the first group only as provenance.
    """
    nodes = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    sites: dict[tuple[int, str], int] = {}
    for group in record.get("sync_groups", []):
        if not isinstance(group, Mapping):
            continue
        group_id = group.get("id")
        for operation in group.get("operations", []):
            if not isinstance(operation, Mapping) or not str(operation.get("type", "")).startswith(
                "pipe_barrier"
            ):
                continue
            target = operation.get("node")
            if not isinstance(target, int) or target not in nodes:
                continue
            pipe = group.get("src_pipe")
            if not isinstance(pipe, str):
                pipe = nodes[target].get("pipe")
            if not isinstance(pipe, str):
                raise ValueError(f"pipe barrier at node {target} has no execution pipe")
            sites.setdefault((target, pipe), group_id if isinstance(group_id, int) else -1)
    return [(target, pipe, group) for (target, pipe), group in sorted(sites.items())]


def _pipeline_components_for_node(
    node: Mapping[str, Any], model: DurationModel
) -> tuple[PipelineComponents | None, dict[str, Any]]:
    """Resolve the stream state charged at a barrier without a pipe constant."""
    try:
        signature = operation_duration_signature(node)
        signature_key = _operation_signature_key(signature)
    except ValueError as error:
        return None, {"source": "unavailable", "detail": str(error)}
    components = model.operation_signature_pipeline.get(signature_key)
    source = "exact_signature_pipeline_override"
    if components is None:
        compatible: list[tuple[PipelineComponents, str]] = []
        for key, candidate in model.operation_signature_pipeline.items():
            try:
                expected = json.loads(key)
            except json.JSONDecodeError:
                continue
            if isinstance(expected, Mapping) and _duration_signatures_compatible(expected, signature):
                compatible.append((candidate, key))
        distinct = {(item.startup_cycles, item.pending_tail_cycles) for item, _ in compatible}
        if len(distinct) > 1:
            raise ValueError("ambiguous compatible exact-signature pipeline overrides")
        if compatible:
            components = compatible[0][0]
            source = "compatible_signature_pipeline_override"
    if components is not None:
        return components, {
            "source": source,
            "detail": "complete operation-signature stream components",
            "signature": signature,
        }
    if model.pto_isa_provider is not None:
        estimate = model.pto_isa_provider.estimate_pipeline(node)
        if estimate is not None:
            return (
                PipelineComponents(
                    startup_cycles=estimate.startup_cycles,
                    pending_tail_cycles=estimate.pending_tail_cycles,
                ),
                {
                    "source": estimate.source,
                    "detail": estimate.detail,
                    "signature": signature,
                },
            )
    return None, {
        "source": "unavailable",
        "detail": "no pinned startup/pending-tail split for this complete operation signature",
        "signature": signature,
    }


def _barrier_dependency_sites(record: Mapping[str, Any], model: DurationModel) -> list[dict[str, Any]]:
    """Describe each emitted barrier as drained work plus successor restart."""
    nodes = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    loop_counts, dynamic_loops = _loop_multipliers(record)
    if dynamic_loops:
        raise ValueError(
            "queue_drain_restart_v1 requires statically bounded loops; "
            f"dynamic loop nodes: {dynamic_loops[:8]}"
        )
    sites: dict[tuple[int, int, str], dict[str, Any]] = {}
    for group in record.get("sync_groups", []):
        if not isinstance(group, Mapping):
            continue
        for operation in group.get("operations", []):
            if (
                not isinstance(operation, Mapping)
                or operation.get("useless") is True
                or not str(operation.get("type", "")).startswith("pipe_barrier")
            ):
                continue
            source = operation.get("dependency_node")
            target = operation.get("node")
            if (
                not isinstance(source, int)
                or source not in nodes
                or not isinstance(target, int)
                or target not in nodes
            ):
                continue
            pipe = operation.get("src_pipe", group.get("src_pipe", nodes[target].get("pipe")))
            if not isinstance(pipe, str):
                raise ValueError(f"pipe barrier at node {target} has no execution pipe")
            key = (source, target, pipe)
            if key in sites:
                continue
            predecessor, predecessor_provenance = _pipeline_components_for_node(nodes[source], model)
            successor, successor_provenance = _pipeline_components_for_node(nodes[target], model)
            multiplier = math.prod(
                loop_counts.get(loop, 1)
                for loop in nodes[target].get("loop_stack", [])
                if isinstance(loop, int)
            )
            complete = predecessor is not None and successor is not None
            site_cycles = None
            if predecessor is not None and successor is not None:
                site_cycles = (
                    model.barrier_instruction_cycles
                    + predecessor.pending_tail_cycles
                    + successor.startup_cycles
                )
            sites[key] = {
                "source": source,
                "target": target,
                "pipe": pipe,
                "group": group.get("id"),
                "loop_multiplier": multiplier,
                "barrier_instruction_cycles": model.barrier_instruction_cycles,
                "predecessor_pending_tail_cycles": (
                    predecessor.pending_tail_cycles if predecessor is not None else None
                ),
                "successor_restart_cycles": successor.startup_cycles if successor is not None else None,
                "site_cycles": site_cycles,
                "expanded_cycles": site_cycles * multiplier if site_cycles is not None else None,
                "predecessor_provenance": predecessor_provenance,
                "successor_provenance": successor_provenance,
                "predecessor_complete": predecessor is not None,
                "successor_complete": successor is not None,
                "complete": complete,
            }
    return list(sites.values())


def _score_queue_drain_restart(record: Mapping[str, Any], model: DurationModel) -> dict[str, Any]:
    """Price barriers from queued predecessor tail and successor restart."""
    branch_ids, markers = _branch_alternatives(record)
    branch_profiles = _branch_iteration_profiles(record)
    if len(branch_ids) > 6:
        raise ValueError(f"queue_drain_restart_v1 supports at most 6 branches, got {len(branch_ids)}")
    nodes = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    sites = _barrier_dependency_sites(record, model)
    if branch_profiles:
        # This diagnostic model aggregates each barrier by one loop
        # multiplier. That representation cannot preserve a predicate whose
        # value changes between iterations: predecessor activity and drain
        # cost must be evaluated for every concrete occurrence. The complete
        # placement graph has that expansion, but silently reusing the
        # aggregate here would manufacture a queue-drain estimate.
        return {
            "model_version": "queue_drain_successor_restart_v1",
            "cost_definition": "barrier instruction + predecessor pending tail + successor stream restart",
            "status": "INCOMPLETE",
            "limitations": ["mixed_iteration_branch_profile_not_supported_v1"],
            "scenario_count": 0,
            "sites": sites,
            "scenarios": [],
        }
    scenarios: list[dict[str, Any]] = []
    for values in itertools.product((False, True), repeat=len(branch_ids)):
        choices = dict(zip(branch_ids, values, strict=True))
        active_sites: list[dict[str, Any]] = []
        for site in sites:
            if not _node_active_in_branch_scenario(nodes[site["target"]], choices, markers):
                continue
            predecessor_active = _node_active_in_branch_scenario(nodes[site["source"]], choices, markers)
            complete = bool(site["successor_complete"]) and (
                bool(site["predecessor_complete"]) or not predecessor_active
            )
            scenario_site = dict(site)
            scenario_site["predecessor_active"] = predecessor_active
            scenario_site["complete"] = complete
            scenario_cycles = None
            if complete:
                scenario_cycles = (
                    model.barrier_instruction_cycles
                    + (float(site["predecessor_pending_tail_cycles"]) if predecessor_active else 0.0)
                    + float(site["successor_restart_cycles"])
                )
            scenario_site["site_cycles"] = scenario_cycles
            scenario_site["expanded_cycles"] = (
                scenario_cycles * int(site["loop_multiplier"]) if scenario_cycles is not None else None
            )
            active_sites.append(scenario_site)
        scenario_complete = all(site["complete"] for site in active_sites)
        scenarios.append(
            {
                "branch_choices": {str(node): choice for node, choice in sorted(choices.items())},
                "active_site_count": len(active_sites),
                "complete": scenario_complete,
                "total_cycles": (
                    sum(float(site["expanded_cycles"]) for site in active_sites)
                    if scenario_complete
                    else None
                ),
                "active_sites": active_sites,
            }
        )
    return {
        "model_version": "queue_drain_successor_restart_v1",
        "cost_definition": "barrier instruction + predecessor pending tail + successor stream restart",
        "status": "COMPLETE",
        "limitations": [],
        "scenario_count": len(scenarios),
        "sites": sites,
        "scenarios": scenarios,
    }


def _score_static_queue_event_scenario(  # noqa: PLR0912 - fail-closed graph expansion
    record: Mapping[str, Any],
    operation_durations: Mapping[int, float],
    pipe_barrier_cycles: Mapping[str, float],
    choices: Mapping[int, bool],
    *,
    logical_memory_dependencies: bool = False,
    sync_edge_origin: str | None = None,
    sync_edge_latency_cycles: float = 0.0,
    include_barrier_sites: bool = True,
) -> dict[str, Any]:
    """Evaluate static loops with explicit per-pipe FIFO and sync recurrences.

    This is a max-plus expansion of PTO-ISA's deterministic queue/event model.
    Stream edges preserve issue order on each pipe. Ordinary InsertSync edges
    bind occurrences in the same iteration; a resolved loop-carried edge binds
    the producer in iteration ``i`` to the consumer in ``i + 1``. The direct
    dependency is equivalent to a matched counter-valued set/wait pair for a
    fixed schedule, without assigning latency to the synchronization opcode.

    Operation durations are inclusive. A calibrated ``pipe_barrier_cycles``
    value represents the barrier instruction plus the pending-tail flush and
    subsequent pipeline restart that PTO-ISA charges when a stream is broken.
    Missing pipe calibration is reported instead of acquiring a guessed cost.

    Induction-variable predicates over statically bounded loops use the exact
    boolean value derived for each dynamic occurrence. Remaining predicates
    use one symbolic choice for every occurrence, yielding auditable
    all-then/all-else path extremes without inventing runtime data.
    """
    loop_counts, dynamic_loops = _loop_multipliers(record)
    if dynamic_loops:
        raise ValueError(
            f"queue_event_v1 requires statically bounded loops; dynamic loop nodes: {dynamic_loops[:8]}"
        )
    prepared = _prepare_control_flow_record(record)
    original_nodes = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    prepared_nodes = {
        node["id"]: node
        for node in prepared.get("nodes", [])
        if isinstance(node, Mapping)
        and isinstance(node.get("id"), int)
        and (node.get("kind") == "operation" or node.get("kind") == "control_point")
    }
    stacks = {
        node_id: _expanded_node_loop_stack(node, original_nodes) for node_id, node in prepared_nodes.items()
    }
    _, branch_markers = _branch_alternatives(record)
    branch_profiles = _branch_iteration_profiles(record)

    clone_by_context: dict[int, dict[tuple[int, ...], int]] = {}
    clone_provenance: dict[int, tuple[int, tuple[int, ...]]] = {}
    active_clones: dict[int, bool] = {}
    durations: dict[int, float] = {}
    next_clone = 0
    for node_id, node in prepared_nodes.items():
        stack = stacks[node_id]
        contexts = itertools.product(*(range(loop_counts[loop]) for loop in stack))
        clones: dict[tuple[int, ...], int] = {}
        if node.get("kind") == "operation":
            divisor = math.prod(loop_counts[loop] for loop in stack)
            base_duration = operation_durations[node_id] / max(divisor, 1)
        else:
            base_duration = 0.0
        for context in contexts:
            clone = next_clone
            next_clone += 1
            clones[context] = clone
            clone_provenance[clone] = (node_id, context)
            iteration_context = dict(zip(stack, context, strict=True))
            active = _node_active_in_branch_scenario(
                node, choices, branch_markers, branch_profiles, iteration_context
            )
            active_clones[clone] = active
            durations[clone] = base_duration if active else 0.0
        clone_by_context[node_id] = clones

    edges: list[tuple[int, int, float, str, int | None]] = []

    def add_expanded_edge(
        source: int,
        target: int,
        *,
        kind: str,
        group: int | None,
        recurrence_loop: int | None = None,
        latency_cycles: float = 0.0,
    ) -> None:
        if source not in clone_by_context or target not in clone_by_context:
            return
        target_clones = clone_by_context[target]
        for source_context, source_clone in clone_by_context[source].items():
            target_context = _expanded_edge_target_context(
                stacks[source],
                source_context,
                stacks[target],
                loop_counts,
                recurrence_loop=recurrence_loop,
            )
            if target_context is None:
                continue
            target_clone = target_clones.get(target_context)
            if target_clone is None:
                raise ValueError(
                    f"expanded schedule edge {source}->{target} has no target context {target_context}"
                )
            if kind == "sync" and (not active_clones[source_clone] or not active_clones[target_clone]):
                continue
            edges.append((source_clone, target_clone, latency_cycles, kind, group))

    for edge in prepared.get("stream_edges", []):
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("source"), edge.get("target")
        if isinstance(source, int) and isinstance(target, int):
            add_expanded_edge(source, target, kind="stream", group=None)

    # The prepared graph has begin->body and body->end edges for one logical
    # iteration. Connect each pipe's end marker to the next begin marker.
    control_points: dict[tuple[int, str], int] = {}
    for node_id, node in prepared_nodes.items():
        origin, pipe = node.get("origin_node"), node.get("pipe")
        if node.get("kind") == "control_point" and isinstance(origin, int) and isinstance(pipe, str):
            control_points[(origin, pipe)] = node_id
    for loop_id, count in loop_counts.items():
        if count <= 1:
            continue
        loop = original_nodes.get(loop_id)
        loop_end = loop.get("end") if isinstance(loop, Mapping) else None
        if not isinstance(loop_end, int):
            raise ValueError(f"static loop {loop_id} has no integer end marker")
        pipes = sorted(
            pipe
            for origin, pipe in control_points
            if origin == loop_id and (loop_end, pipe) in control_points
        )
        for pipe in pipes:
            add_expanded_edge(
                control_points[(loop_end, pipe)],
                control_points[(loop_id, pipe)],
                kind="pipe_iteration",
                group=None,
                recurrence_loop=loop_id,
            )

    pipe_order_edge_count = len(edges)
    logical_memory_edge_count = 0
    if logical_memory_dependencies:
        occurrences = _structured_operation_occurrences(
            record, clone_by_context, stacks, loop_counts, active_clones
        )
        memory_edges = _expanded_logical_memory_edges(occurrences, sync_edge_latency_cycles)
        logical_memory_edge_count = len(memory_edges)
        edges.extend(memory_edges)

    resolved_loop_carried = _effective_loop_carried_edge_indices(record)
    sync_edges = [edge for edge in prepared.get("sync_edges", []) if isinstance(edge, Mapping)]
    # The resolver indexes the original and prepared sync arrays identically.
    for edge_index, edge in enumerate(sync_edges):
        if sync_edge_origin is not None and edge.get("analysis_origin") != sync_edge_origin:
            continue
        source, target, group = edge.get("source"), edge.get("target"), edge.get("group")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        add_expanded_edge(
            source,
            target,
            kind="sync",
            group=group if isinstance(group, int) else None,
            recurrence_loop=resolved_loop_carried.get(edge_index),
            latency_cycles=sync_edge_latency_cycles,
        )

    baseline_edges = [edge for edge in edges if edge[3] != "sync"]
    full_durations = dict(durations)
    calibrated_barriers = 0
    uncalibrated_barriers: list[dict[str, Any]] = []
    barrier_sites = _pipe_barrier_sites(record) if include_barrier_sites else []
    for target, pipe, group in barrier_sites:
        cycles = pipe_barrier_cycles.get(pipe)
        if cycles is None:
            uncalibrated_barriers.append({"node": target, "pipe": pipe, "group": group})
            continue
        calibrated_barriers += 1
        for clone in clone_by_context.get(target, {}).values():
            if active_clones[clone]:
                full_durations[clone] += cycles

    baseline, _, _, baseline_path = _longest_path(durations, baseline_edges)
    full, _, _, full_path = _longest_path(full_durations, edges)
    return {
        "model_version": "static_unrolled_pipe_event_v2",
        "operation_duration_policy": "inclusive_cycles",
        "pipeline_break_policy": "calibrated_per_pipe_barrier_restart",
        "branch_policy": "exact_static_induction_profiles_plus_symbolic_fixed_choice_scenario",
        "branch_choices": {str(node): choice for node, choice in sorted(choices.items())},
        "exact_iteration_profile_count": len(branch_profiles),
        "expanded_node_count": len(durations),
        "expanded_stream_edge_count": pipe_order_edge_count,
        "expanded_logical_memory_edge_count": logical_memory_edge_count,
        "expanded_sync_edge_count": len(edges) - len(baseline_edges),
        "calibrated_pipe_barrier_site_count": calibrated_barriers,
        "uncalibrated_pipe_barrier_sites": uncalibrated_barriers,
        "pipeline_break_model_complete": not uncalibrated_barriers,
        "baseline_makespan_cycles": baseline,
        "full_makespan_cycles": full,
        "synchronization_exposure_cycles": full - baseline,
        "baseline_critical_path": _compact_expanded_path(baseline_path, clone_provenance),
        "full_critical_path": _compact_expanded_path(full_path, clone_provenance),
    }


def _score_static_queue_event_graph(
    record: Mapping[str, Any],
    operation_durations: Mapping[int, float],
    pipe_barrier_cycles: Mapping[str, float],
) -> dict[str, Any]:
    """Score static queues over auditable all-then/all-else branch extremes."""
    branch_ids, _ = _branch_alternatives(record)
    branch_profiles = _branch_iteration_profiles(record)
    if len(branch_ids) > 6:
        raise ValueError(
            "queue_event_v2 supports at most 6 conditional regions for exhaustive "
            f"path extremes, got {len(branch_ids)}"
        )
    loop_counts, dynamic_loops = _loop_multipliers(record)
    if dynamic_loops:
        raise ValueError(
            f"queue_event_v2 requires statically bounded loops; dynamic loop nodes: {dynamic_loops[:8]}"
        )
    prepared = _prepare_control_flow_record(record)
    original_nodes = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    nodes_per_scenario = sum(
        math.prod(loop_counts[loop] for loop in _expanded_node_loop_stack(node, original_nodes))
        for node in prepared.get("nodes", [])
        if isinstance(node, Mapping)
        and isinstance(node.get("id"), int)
        and node.get("kind") in {"operation", "control_point"}
    )
    scenario_count = 2 ** len(branch_ids)
    total_expanded_nodes = nodes_per_scenario * scenario_count
    if total_expanded_nodes > _MAX_QUEUE_EVENT_EXPANDED_NODES:
        raise ValueError(
            "queue_event_v2 expansion exceeds the resource-safe node budget: "
            f"{nodes_per_scenario} nodes/scenario * {scenario_count} scenarios = "
            f"{total_expanded_nodes}, limit {_MAX_QUEUE_EVENT_EXPANDED_NODES}"
        )
    scenarios = [
        _score_static_queue_event_scenario(
            record,
            operation_durations,
            pipe_barrier_cycles,
            dict(zip(branch_ids, values, strict=True)),
        )
        for values in itertools.product((False, True), repeat=len(branch_ids))
    ]
    full_cycles = [float(scenario["full_makespan_cycles"]) for scenario in scenarios]
    baseline_cycles = [float(scenario["baseline_makespan_cycles"]) for scenario in scenarios]
    complete = all(scenario["pipeline_break_model_complete"] for scenario in scenarios)
    return {
        "model_version": "static_unrolled_pipe_event_branch_extremes_v2",
        "branch_policy": ("exact_static_induction_profiles_plus_symbolic_fixed_choice_extremes"),
        "mixed_iteration_branch_profile_available": bool(branch_profiles),
        "exact_iteration_profile_count": len(branch_profiles),
        "scenario_count": len(scenarios),
        "expanded_node_budget": _MAX_QUEUE_EVENT_EXPANDED_NODES,
        "total_expanded_node_count": total_expanded_nodes,
        "pipeline_break_model_complete": complete,
        "baseline_makespan_cycles": max(baseline_cycles),
        "full_makespan_cycles": max(full_cycles),
        "minimum_full_makespan_cycles": min(full_cycles),
        "maximum_full_makespan_cycles": max(full_cycles),
        "synchronization_exposure_cycles": max(full_cycles) - max(baseline_cycles),
        "scenarios": scenarios,
    }


def _longest_path_between(
    durations: Mapping[int, float],
    edges: Iterable[tuple[int, int, float, str, int | None]],
    start: int,
    end: int,
) -> tuple[float, list[int]] | None:
    """Return the longest inclusive path from ``start`` to ``end`` in a DAG."""
    successors: dict[int, list[tuple[int, float]]] = {node: [] for node in durations}
    indegree = {node: 0 for node in durations}
    for source, target, latency, _, _ in edges:
        successors[source].append((target, latency))
        indegree[target] += 1

    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for target, _ in sorted(successors[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(durations):
        cyclic = sorted(node for node, degree in indegree.items() if degree)
        raise ValueError(f"loop body is cyclic before recurrence edges are added: {cyclic[:8]}")

    distance: dict[int, float] = {start: durations[start]}
    parent: dict[int, int | None] = {start: None}
    for node in order:
        if node not in distance:
            continue
        for target, latency in sorted(successors[node]):
            candidate = distance[node] + latency + durations[target]
            if candidate > distance.get(target, -math.inf):
                distance[target] = candidate
                parent[target] = node
    if end not in distance:
        return None
    path: list[int] = []
    cursor: int | None = end
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return distance[end], path


def _require_integer_edge_endpoints(edge: Mapping[str, Any], description: str) -> tuple[int, int]:
    source, target = edge.get("source"), edge.get("target")
    if not isinstance(source, int) or not isinstance(target, int):
        raise ValueError(f"{description} has invalid endpoints")
    return source, target


def _loop_recurrence_score(
    record: Mapping[str, Any],
    durations: Mapping[int, float],
    existing_edges: Sequence[tuple[int, int, float, str, int | None]],
    *,
    loop_id: int,
    source: int,
    target: int,
    candidate_latency: float,
) -> dict[str, Any]:
    """Score one distance-one edge with a loop-II lower-bound model.

    A distance-one dependency ``source(i) -> target(i+1)`` constrains the
    initiation interval only when the intra-iteration graph contains a path
    back from ``target`` to ``source``.  The resulting recurrence latency is
    the inclusive path duration plus the dependency latency.  Per-pipe work is
    an independent resource lower bound.
    """
    nodes_by_id = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    loop_nodes = {
        node_id
        for node_id, node in nodes_by_id.items()
        if loop_id in node.get("loop_stack", []) and node_id in durations
    }
    if source not in loop_nodes or target not in loop_nodes:
        raise ValueError(f"candidate edge {source}->{target} is not contained in PTOAS loop {loop_id}")

    loop_counts, dynamic_loops = _loop_multipliers(record)
    trip_count = loop_counts.get(loop_id, 1)
    iteration_durations: dict[int, float] = {}
    for node_id in loop_nodes:
        loop_stack = nodes_by_id[node_id].get("loop_stack", [])
        selected_index = loop_stack.index(loop_id)
        # Whole-function durations multiply every containing static loop. One
        # iteration of the selected loop removes its own and all ancestor
        # multipliers, while retaining work in nested loops.
        divisor = math.prod(loop_counts.get(loop, 1) for loop in loop_stack[: selected_index + 1])
        iteration_durations[node_id] = durations[node_id] / max(divisor, 1)
    body_edges = [edge for edge in existing_edges if edge[0] in loop_nodes and edge[1] in loop_nodes]
    # This validates the body once and gives the candidate's recurrence path.
    candidate_path = _longest_path_between(iteration_durations, body_edges, target, source)

    pipe_work: dict[str, float] = defaultdict(float)
    for node_id, duration in iteration_durations.items():
        node = nodes_by_id[node_id]
        pipe = node.get("pipe")
        if node.get("kind") == "operation" and isinstance(pipe, str):
            pipe_work[pipe] += duration
    resource_bound = max(pipe_work.values(), default=0.0)

    loop_end = next(
        (
            node.get("end")
            for node in record.get("nodes", [])
            if isinstance(node, Mapping)
            and node.get("kind") == "loop"
            and node.get("loop_kind") == "LOOP_BEGIN"
            and node.get("id") == loop_id
        ),
        None,
    )
    if not isinstance(loop_end, int):
        raise ValueError(f"PTOAS loop {loop_id} has no integer loop-end identity")
    groups: dict[int, Mapping[str, Any]] = {}
    for group in record.get("sync_groups", []):
        if not isinstance(group, Mapping):
            continue
        group_id = group.get("id")
        if isinstance(group_id, int):
            groups[group_id] = group
    existing_recurrences: list[dict[str, Any]] = []
    for edge in record.get("sync_edges", []):
        if not isinstance(edge, Mapping) or not edge.get("loop_carried"):
            continue
        edge_source, edge_target = _require_integer_edge_endpoints(edge, "loop-carried schedule edge")
        if edge_source not in loop_nodes or edge_target not in loop_nodes:
            continue
        edge_group = edge.get("group")
        group = groups.get(edge_group) if isinstance(edge_group, int) else None
        operations = group.get("operations", []) if isinstance(group, Mapping) else []
        loop_ends = {
            operation.get("loop_end")
            for operation in operations
            if isinstance(operation, Mapping) and isinstance(operation.get("loop_end"), int)
        }
        explicit_loop_end = edge.get("loop_end")
        if isinstance(explicit_loop_end, int):
            loop_ends.add(explicit_loop_end)
        if not loop_ends:
            raise ValueError(
                "loop-carried schedule edge has no loop identity: "
                f"group={edge.get('group')}, edge={edge_source}->{edge_target}"
            )
        if loop_end not in loop_ends:
            continue
        recurrence_path = _longest_path_between(iteration_durations, body_edges, edge_target, edge_source)
        if recurrence_path is None:
            continue
        latency, path = recurrence_path
        existing_recurrences.append(
            {
                "source": edge_source,
                "target": edge_target,
                "group": edge.get("group"),
                "cycles": latency + candidate_latency,
                "path": path,
            }
        )

    existing_recurrence_bound = max((float(item["cycles"]) for item in existing_recurrences), default=0.0)
    base_ii = max(resource_bound, existing_recurrence_bound)
    candidate_cycles = None if candidate_path is None else candidate_path[0] + candidate_latency
    with_candidate_ii = max(base_ii, candidate_cycles or 0.0)
    return {
        "model_version": "loop_recurrence_ii_lower_bound_v1",
        "loop_node": loop_id,
        "static_trip_count": None if loop_id in dynamic_loops else trip_count,
        "pipe_work_cycles": dict(sorted(pipe_work.items())),
        "resource_ii_lower_bound_cycles": resource_bound,
        "existing_recurrence_ii_lower_bound_cycles": existing_recurrence_bound,
        "existing_recurrences": existing_recurrences,
        "base_ii_lower_bound_cycles": base_ii,
        "candidate_recurrence_cycles": candidate_cycles,
        "candidate_recurrence_path": None if candidate_path is None else candidate_path[1],
        "with_candidate_ii_lower_bound_cycles": with_candidate_ii,
        "weight_cycles": max(0.0, with_candidate_ii - base_ii),
    }


def _resolve_loop_carried_sync_edges(
    sync_edges: Sequence[Mapping[str, Any]],
    groups: Mapping[int, Mapping[str, Any]],
    loops: Sequence[Mapping[str, Any]],
    schedule_ids_by_loop: Mapping[int, set[int]],
) -> dict[int, int]:
    """Resolve actual loop recurrences among edges carrying PTOAS loop metadata.

    PTOAS retains ``forEnd`` on synchronization operations after later passes
    move them outside the originating loop. Such an edge is historically
    loop-derived, but its final placement is an ordinary DAG dependency, not
    an iteration recurrence. Only endpoints that both remain inside exactly
    one annotated loop are recurrence constraints.
    """
    resolved_loop_carried: dict[int, int] = {}
    for edge_index, edge in enumerate(sync_edges):
        if not edge.get("loop_carried"):
            continue
        source, target = _require_integer_edge_endpoints(edge, f"loop-carried sync edge {edge_index}")
        edge_group = edge.get("group")
        group = groups.get(edge_group) if isinstance(edge_group, int) else None
        operations = group.get("operations", []) if isinstance(group, Mapping) else []
        loop_ends: set[int] = set()
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            operation_loop_end = operation.get("loop_end")
            if isinstance(operation_loop_end, int):
                loop_ends.add(operation_loop_end)
        explicit_loop_end = edge.get("loop_end")
        if isinstance(explicit_loop_end, int):
            loop_ends.add(explicit_loop_end)
        if not loop_ends:
            raise ValueError(
                "loop-carried schedule edge has no loop identity and does not resolve to exactly "
                "one loop model: "
                f"edge={source}->{target}, group={edge.get('group')}, loop_ends=[], matches=[]"
            )
        matching_loops = [
            loop["id"]
            for loop in loops
            if loop.get("end") in loop_ends
            and source in schedule_ids_by_loop[loop["id"]]
            and target in schedule_ids_by_loop[loop["id"]]
        ]
        if len(matching_loops) > 1:
            raise ValueError(
                "loop-carried sync edge resolves to multiple loop models: "
                f"edge={source}->{target}, group={edge.get('group')}, "
                f"loop_ends={sorted(loop_ends)}, matches={matching_loops}"
            )
        if matching_loops:
            resolved_loop_carried[edge_index] = matching_loops[0]
            continue
        annotated_loop_exists = any(loop.get("end") in loop_ends for loop in loops)
        if not annotated_loop_exists:
            raise ValueError(
                "loop-carried sync edge does not resolve to exactly one loop model: "
                f"edge={source}->{target}, group={edge.get('group')}, "
                f"loop_ends={sorted(loop_ends)}, matches=[]"
            )
    return resolved_loop_carried


def _effective_loop_carried_edge_indices(record: Mapping[str, Any]) -> dict[int, int]:
    nodes = [node for node in record.get("nodes", []) if isinstance(node, Mapping)]
    nodes_by_id = {node["id"]: node for node in nodes if isinstance(node.get("id"), int)}
    loops = [
        node
        for node in nodes
        if node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(node.get("id"), int)
    ]
    schedule_ids_by_loop = {
        loop["id"]: {
            node_id for node_id, node in nodes_by_id.items() if loop["id"] in node.get("loop_stack", [])
        }
        for loop in loops
    }
    groups = {
        group["id"]: group
        for group in record.get("sync_groups", [])
        if isinstance(group, Mapping) and isinstance(group.get("id"), int)
    }
    sync_edges = [edge for edge in record.get("sync_edges", []) if isinstance(edge, Mapping)]
    return _resolve_loop_carried_sync_edges(sync_edges, groups, loops, schedule_ids_by_loop)


def _loop_boundary_kind(
    source: int,
    target: int,
    loop_id: int,
    loop_end: Any,
    nodes_by_id: Mapping[int, Mapping[str, Any]],
) -> str | None:
    """Classify a sync edge crossing one loop's structural boundary."""
    source_node = nodes_by_id.get(source, {})
    target_node = nodes_by_id.get(target, {})
    source_inside = loop_id in source_node.get("loop_stack", [])
    target_inside = loop_id in target_node.get("loop_stack", [])
    source_origin = source_node.get("origin_node", source)
    target_origin = target_node.get("origin_node", target)
    source_marker = source_origin in {loop_id, loop_end}
    target_marker = target_origin in {loop_id, loop_end}
    if not (source_marker or target_marker or source_inside != target_inside):
        return None
    is_entry = (source_origin == loop_id and target_inside) or (
        not source_inside and target_origin == loop_id
    )
    is_entry |= not source_marker and not source_inside and target_inside
    is_exit = (source_inside and target_origin == loop_end) or (
        source_origin == loop_end and not target_inside
    )
    is_exit |= not target_marker and source_inside and not target_inside
    if is_entry and not is_exit:
        return "loop_entry"
    if is_exit and not is_entry:
        return "loop_exit"
    return "loop_boundary"


def _existing_loop_sync_models(
    record: Mapping[str, Any],
    operation_durations: Mapping[int, float],
    graph_edges: Sequence[tuple[int, int, float, str, int | None]],
    sync_latency_cycles: float,
) -> list[dict[str, Any]]:
    """Model existing loop-carried recurrences and loop-boundary sync edges."""
    nodes_by_id = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    operation_nodes = {
        node_id: node
        for node_id, node in nodes_by_id.items()
        if node.get("kind") == "operation" and node_id in operation_durations
    }
    groups: dict[int, Mapping[str, Any]] = {}
    for group in record.get("sync_groups", []):
        if not isinstance(group, Mapping):
            continue
        group_id = group.get("id")
        if isinstance(group_id, int):
            groups[group_id] = group
    loop_counts, dynamic_loops = _loop_multipliers(record)
    loops = [
        loop
        for loop in record.get("nodes", [])
        if isinstance(loop, Mapping)
        and loop.get("kind") == "loop"
        and loop.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(loop.get("id"), int)
    ]
    schedule_ids_by_loop: dict[int, set[int]] = {}
    for loop in loops:
        loop_id = loop.get("id")
        if not isinstance(loop_id, int):
            continue
        # Nested loop begin/end markers are real synchronization placement
        # sites. Include them in the enclosing loop's per-iteration graph at
        # zero duration; otherwise a recurrence from an inner loop boundary
        # to the next outer iteration cannot be joined to its loop model.
        schedule_ids_by_loop[loop_id] = {
            node_id
            for node_id, node in nodes_by_id.items()
            if loop_id in node.get("loop_stack", [])
            and (
                node.get("kind") in {"loop", "branch", "placeholder", "control_point"}
                or node_id in operation_durations
            )
        }
    sync_edges = [edge for edge in record.get("sync_edges", []) if isinstance(edge, Mapping)]
    resolved_loop_carried = _resolve_loop_carried_sync_edges(sync_edges, groups, loops, schedule_ids_by_loop)

    models: list[dict[str, Any]] = []
    for loop in loops:
        loop_id = loop["id"]
        loop_end = loop.get("end")
        loop_schedule_ids = schedule_ids_by_loop[loop_id]
        iteration_durations: dict[int, float] = {}
        pipe_work: dict[str, float] = defaultdict(float)
        for node_id in loop_schedule_ids:
            node = nodes_by_id[node_id]
            loop_stack = node.get("loop_stack", [])
            selected_index = loop_stack.index(loop_id)
            divisor = math.prod(
                loop_counts.get(containing_loop, 1) for containing_loop in loop_stack[: selected_index + 1]
            )
            duration = operation_durations.get(node_id, 0.0) / max(divisor, 1)
            iteration_durations[node_id] = duration
            pipe = node.get("pipe")
            if node.get("kind") == "operation" and isinstance(pipe, str):
                pipe_work[pipe] += duration
        body_edges = [
            edge for edge in graph_edges if edge[0] in loop_schedule_ids and edge[1] in loop_schedule_ids
        ]

        recurrences: list[dict[str, Any]] = []
        boundary_edges: list[dict[str, Any]] = []
        for edge_index, edge in enumerate(sync_edges):
            source, target = edge.get("source"), edge.get("target")
            if not isinstance(source, int) or not isinstance(target, int):
                continue
            if edge_index in resolved_loop_carried:
                if resolved_loop_carried[edge_index] != loop_id:
                    continue
                recurrence_path = _longest_path_between(iteration_durations, body_edges, target, source)
                recurrences.append(
                    {
                        "source": source,
                        "target": target,
                        "group": edge.get("group"),
                        "cycles": (
                            None if recurrence_path is None else recurrence_path[0] + sync_latency_cycles
                        ),
                        "path": None if recurrence_path is None else recurrence_path[1],
                    }
                )
                continue

            boundary_kind = _loop_boundary_kind(source, target, loop_id, loop_end, nodes_by_id)
            if boundary_kind is not None:
                boundary_edges.append(
                    {
                        "source": source,
                        "target": target,
                        "group": edge.get("group"),
                        "kind": boundary_kind,
                    }
                )

        recurrence_bound = max(
            (float(item["cycles"]) for item in recurrences if item["cycles"] is not None),
            default=0.0,
        )
        resource_bound = max(pipe_work.values(), default=0.0)
        models.append(
            {
                "model_version": "loop_sync_ii_and_boundary_v1",
                "loop_node": loop_id,
                "loop_end_node": loop_end,
                "static_trip_count": None if loop_id in dynamic_loops else loop_counts.get(loop_id, 1),
                "operation_node_count": sum(node_id in operation_nodes for node_id in loop_schedule_ids),
                "structural_node_count": sum(
                    nodes_by_id[node_id].get("kind") in {"loop", "branch", "placeholder"}
                    for node_id in loop_schedule_ids
                ),
                "control_point_node_count": sum(
                    nodes_by_id[node_id].get("kind") == "control_point" for node_id in loop_schedule_ids
                ),
                "pipe_work_cycles": dict(sorted(pipe_work.items())),
                "resource_ii_lower_bound_cycles": resource_bound,
                "recurrence_ii_lower_bound_cycles": recurrence_bound,
                "ii_lower_bound_cycles": max(resource_bound, recurrence_bound),
                "loop_carried_recurrences": recurrences,
                "loop_boundary_sync_edges": boundary_edges,
            }
        )
    return models


def score_schedule(record: Mapping[str, Any], model: DurationModel) -> dict[str, Any]:
    """Score one PTOAS schedule graph and its synchronization exposure."""
    record = _propagate_barrier_dependency_provenance(record)
    operation_durations, provenance, dynamic_loops = estimate_node_durations(record, model)
    if dynamic_loops:
        raise ValueError(
            f"duration_v0 requires statically bounded loops; dynamic loop nodes: {dynamic_loops[:8]}"
        )
    prepared_record = _prepare_control_flow_record(record)
    durations = _schedule_graph_durations(prepared_record, operation_durations)
    node_ids = set(durations)
    stream_edges, _ = _graph_edges(prepared_record, node_ids, include_sync=False)
    full_edges, edge_diagnostics = _graph_edges(prepared_record, node_ids, include_sync=True)
    full_edges = [
        (source, target, model.sync_latency_cycles if kind == "sync" else latency, kind, group)
        for source, target, latency, kind, group in full_edges
    ]
    loop_sync_models = _existing_loop_sync_models(
        prepared_record, operation_durations, full_edges, model.sync_latency_cycles
    )

    baseline, top, bottom, baseline_path = _longest_path(durations, stream_edges)
    full, _, _, full_path = _longest_path(durations, full_edges)
    loop_latency_lower_bounds: list[dict[str, Any]] = []
    for loop_model in loop_sync_models:
        trip_count = loop_model.get("static_trip_count")
        if not isinstance(trip_count, int):
            continue
        resource_ii = float(loop_model["resource_ii_lower_bound_cycles"])
        schedule_ii = float(loop_model["ii_lower_bound_cycles"])
        lower_bound = resource_ii + max(0, trip_count - 1) * schedule_ii
        loop_latency_lower_bounds.append(
            {
                "loop_node": loop_model["loop_node"],
                "static_trip_count": trip_count,
                "latency_lower_bound_cycles": lower_bound,
            }
        )
    loop_aware_makespan = max(
        [full, *(item["latency_lower_bound_cycles"] for item in loop_latency_lower_bounds)]
    )
    queue_event_score = _score_static_queue_event_graph(
        record, operation_durations, model.pipe_barrier_cycles
    )
    queue_drain_restart_score = _score_queue_drain_restart(record, model)

    edge_exposure: list[dict[str, Any]] = []
    for source, target, _, kind, group in full_edges:
        if kind != "sync":
            continue
        exposure = max(
            0.0,
            top[source] + model.sync_latency_cycles + bottom[target] - baseline,
        )
        edge_exposure.append(
            {
                "source": source,
                "target": target,
                "group": group,
                "marginal_cycles": exposure,
                "source_top_cycles": top[source],
                "target_bottom_cycles": bottom[target],
            }
        )

    exact = sum(not item["fallback"] for item in provenance.values())
    fallback = sum(item["fallback"] for item in provenance.values())
    source_counts = Counter(str(item["source"]) for item in provenance.values())
    sync_record_summary = _pre_codegen_sync_record_summary(record)
    return {
        "schema_version": 1,
        "function": record.get("function", "<unknown>"),
        "status": record.get("status", "<unknown>"),
        "schedule_export_source": record.get("export_source", "native_schedule_graph_v1"),
        "schedule_export_limitations": record.get("export_limitations", {}),
        "barrier_dependency_provenance": record["barrier_dependency_provenance"],
        "control_flow_graph_version": prepared_record.get("control_flow_graph_version"),
        "control_point_nodes": prepared_record.get("control_point_nodes", []),
        "duration_model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "loop_policy": "aggregate_static_work_v0",
        "loop_sync_model_version": "loop_sync_ii_and_boundary_v1",
        "loop_sync_models": loop_sync_models,
        "dynamic_loop_ids": dynamic_loops,
        **edge_diagnostics,
        **_latency_graph_completeness(prepared_record, edge_diagnostics, loop_sync_models),
        "operation_nodes": len(operation_durations),
        "exact_duration_nodes": exact,
        "fallback_duration_nodes": fallback,
        "exact_duration_coverage": exact / len(operation_durations) if operation_durations else 0.0,
        "duration_source_counts": dict(sorted(source_counts.items())),
        "pto_isa_provider": (
            {
                "revision": model.pto_isa_provider.revision,
                "snapshot_sha256": provider_snapshot_sha256(model.pto_isa_provider),
            }
            if model.pto_isa_provider is not None
            else None
        ),
        "baseline_makespan_cycles": baseline,
        "full_makespan_cycles": full,
        "loop_aware_makespan_cycles": loop_aware_makespan,
        "queue_event_makespan_cycles": queue_event_score["full_makespan_cycles"],
        "queue_event_baseline_makespan_cycles": queue_event_score["baseline_makespan_cycles"],
        "queue_event_synchronization_exposure_cycles": queue_event_score["synchronization_exposure_cycles"],
        "queue_event_model": queue_event_score,
        "queue_drain_restart_model": queue_drain_restart_score,
        "loop_latency_lower_bounds": loop_latency_lower_bounds,
        "synchronization_exposure_cycles": max(0.0, full - baseline),
        "baseline_critical_path": baseline_path,
        "full_critical_path": full_path,
        "sync_edge_exposure": edge_exposure,
        "pre_codegen_sync_record_summary": sync_record_summary,
        "node_durations": {str(node): value for node, value in sorted(provenance.items())},
    }


def _node_access_order(node: Mapping[str, Any]) -> int | None:
    operation = node.get("operation")
    if not isinstance(operation, Mapping):
        return None
    explicit = operation.get("pypto_access_order")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    location = operation.get("location")
    if not isinstance(location, str):
        return None
    match = _ACCESS_LOCATION_RE.search(location)
    return int(match.group(1)) if match else None


def _route_pipe(route: str) -> str:
    if "@" not in route:
        raise ValueError(f"candidate route has no resource suffix: '{route}'")
    resource = route.rsplit("@", maxsplit=1)[1]
    try:
        return _RESOURCE_PIPE[resource]
    except KeyError as error:
        raise ValueError(
            f"duration_v0 has no verified PTOAS pipe mapping for resource '{resource}'"
        ) from error


def _site_nodes(record: Mapping[str, Any]) -> dict[tuple[int, str], list[int]]:
    result: dict[tuple[int, str], list[int]] = defaultdict(list)
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("kind") != "operation":
            continue
        node_id = node.get("id")
        pipe = node.get("pipe")
        access_order = _node_access_order(node)
        if isinstance(node_id, int) and isinstance(pipe, str) and access_order is not None:
            result[(access_order, pipe)].append(node_id)
    for nodes in result.values():
        nodes.sort()
    return result


def _schedule_proves_complete_access_provenance(record: Mapping[str, Any]) -> bool:
    """Return whether absence of a ``pypto.access.N`` site proves its elimination.

    A native schedule graph alone does not establish this: an exporter may
    simply have dropped locations.  The raw-PTO semantic join is fail-closed on
    operation order and attaches every surviving access location, so a
    candidate access order absent from such a record was genuinely eliminated
    before the final pre-InsertSync operation stream.
    """
    source = record.get("export_source")
    if not isinstance(source, str):
        return False
    limitations = record.get("export_limitations", {})
    if not isinstance(limitations, Mapping):
        return False
    if "raw_pto_semantics" in source:
        return limitations.get("operation_metadata_missing", 0) == 0
    if "pto_access_join_v3" in source:
        return limitations.get("access_provenance_missing") is False
    return False


def _node_operation_name(node: Mapping[str, Any]) -> str | None:
    operation = node.get("operation")
    if isinstance(operation, Mapping):
        raw_name = operation.get("raw_pto_op_name")
        if isinstance(raw_name, str):
            return _join_operation_name(raw_name)
    op_name = node.get("op_name")
    return _join_operation_name(op_name) if isinstance(op_name, str) else None


def _resolve_candidate_site(
    indexed_nodes: Mapping[tuple[int, str], list[int]],
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    *,
    access_order: int,
    route_pipe: str,
) -> tuple[list[int], str, str | None]:
    """Bind one PyPTO access site to the pipe PTOAS actually schedules.

    Exact route/pipe agreement is the default. A small explicit exception set
    handles operations whose PyPTO execution-resource classification and
    PTOAS scheduling pipe are known to differ. Ambiguous sites and all unknown
    disagreements still fail closed.
    """
    exact = indexed_nodes.get((access_order, route_pipe), [])
    if exact:
        return exact, route_pipe, None

    matches: list[tuple[list[int], str, str]] = []
    for (site, schedule_pipe), node_ids in indexed_nodes.items():
        if site != access_order or not node_ids:
            continue
        names = {_node_operation_name(nodes_by_id[node_id]) for node_id in node_ids}
        if len(names) != 1 or None in names:
            continue
        operation_name = next(iter(names))
        if not isinstance(operation_name, str):
            continue
        reason = _ROUTE_PIPE_JOIN_EXCEPTIONS.get((route_pipe, schedule_pipe, operation_name))
        if reason is not None:
            matches.append((node_ids, schedule_pipe, reason))
    if len(matches) == 1:
        return matches[0]
    return [], route_pipe, None


def _join_candidate_access_sites(
    candidate: ReuseCandidateRecord,
    candidate_index: int,
    indexed_nodes: Mapping[tuple[int, str], list[int]],
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    materialized_access_orders: set[int],
    known_nonmaterialized_access_orders: frozenset[int],
    access_provenance_complete: bool,
) -> tuple[list[int], str, str | None, list[int], str, str | None] | dict[str, Any]:
    """Join both candidate sites or return an evidence-backed non-materialized row."""
    prior_route_pipe = _route_pipe(candidate.prior_route)
    next_route_pipe = _route_pipe(candidate.next_route)
    prior_nodes, prior_pipe, prior_pipe_override = _resolve_candidate_site(
        indexed_nodes,
        nodes_by_id,
        access_order=candidate.prior_access_order,
        route_pipe=prior_route_pipe,
    )
    next_nodes, next_pipe, next_pipe_override = _resolve_candidate_site(
        indexed_nodes,
        nodes_by_id,
        access_order=candidate.next_access_order,
        route_pipe=next_route_pipe,
    )
    for access_order, route_pipe, resolved_nodes in (
        (candidate.prior_access_order, prior_route_pipe, prior_nodes),
        (candidate.next_access_order, next_route_pipe, next_nodes),
    ):
        if access_order in materialized_access_orders and not resolved_nodes:
            raise ValueError(
                "candidate site did not join to the expected PTOAS pipe: "
                f"site={access_order}, pipe={route_pipe}"
            )
    missing_access_orders = sorted(
        {
            access_order
            for access_order in (candidate.prior_access_order, candidate.next_access_order)
            if access_order not in materialized_access_orders
        }
    )
    if missing_access_orders:
        independently_proven = set(known_nonmaterialized_access_orders)
        if access_provenance_complete:
            independently_proven.update(missing_access_orders)
        unproven = sorted(set(missing_access_orders) - independently_proven)
        if unproven:
            raise ValueError(
                "candidate access orders are absent from the lowered schedule without "
                f"non-materialization evidence: {unproven}"
            )
        return {
            "candidate_index": candidate_index,
            "first_buffer": candidate.first_buffer,
            "second_buffer": candidate.second_buffer,
            "prior_buffer": candidate.prior_buffer,
            "next_buffer": candidate.next_buffer,
            "prior_access_order": candidate.prior_access_order,
            "next_access_order": candidate.next_access_order,
            "prior_route_pipe": prior_route_pipe,
            "next_route_pipe": next_route_pipe,
            "missing_access_orders": missing_access_orders,
            "status": "not_materialized_in_schedule",
            "nonmaterialization_evidence": (
                "complete_raw_pto_access_provenance"
                if access_provenance_complete
                else "external_digest_bound_evidence"
            ),
            "weight_cycles": 0.0,
        }
    if not prior_nodes or not next_nodes:
        raise ValueError("candidate access-site join produced an empty executable endpoint")
    return (
        prior_nodes,
        prior_pipe,
        prior_pipe_override,
        next_nodes,
        next_pipe,
        next_pipe_override,
    )


def _deduplicate_scored_candidate_edges(
    rows: Sequence[dict[str, Any]], execution_counts: Mapping[int, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse candidate records that join to the same schedule edge."""
    distance_zero_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    loop_rows: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "scored":
            distance_zero_rows[(row["source_node"], row["target_node"])].append(row)
        elif row.get("status") in {"loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}:
            loop_rows[(row["loop_node"], row["source_node"], row["target_node"])].append(row)

    distance_zero_edges: list[dict[str, Any]] = []
    for (source, target), duplicates in sorted(distance_zero_rows.items()):
        weights = {float(row["weight_cycles"]) for row in duplicates}
        pipe_pairs = {(row.get("prior_pipe"), row.get("next_pipe")) for row in duplicates}
        predicates = {json.dumps(row.get("branch_predicate", {}), sort_keys=True) for row in duplicates}
        if len(weights) != 1 or len(pipe_pairs) != 1 or len(predicates) != 1:
            raise ValueError(
                "candidate records joined to one distance-zero edge but have inconsistent metadata: "
                f"edge={source}->{target}"
            )
        source_pipe, target_pipe = next(iter(pipe_pairs))
        distance_zero_edges.append(
            {
                "source_node": source,
                "target_node": target,
                "source_pipe": source_pipe,
                "target_pipe": target_pipe,
                "candidate_indices": [row["candidate_index"] for row in duplicates],
                "candidate_count": len(duplicates),
                "source_execution_count": execution_counts[source],
                "target_execution_count": execution_counts[target],
                "estimated_sync_endpoint_executions": (execution_counts[source] + execution_counts[target]),
                "weight_cycles": duplicates[0]["weight_cycles"],
                "branch_predicate": duplicates[0].get("branch_predicate", {}),
            }
        )

    loop_edges: list[dict[str, Any]] = []
    for (loop_node, source, target), duplicates in sorted(loop_rows.items()):
        weights = {float(row["weight_cycles"]) for row in duplicates}
        recurrence_cycles = {row.get("candidate_recurrence_cycles") for row in duplicates}
        weight_semantics = {row.get("weight_semantics", "loop_ii_increment_v1") for row in duplicates}
        pipe_pairs = {(row.get("prior_pipe"), row.get("next_pipe")) for row in duplicates}
        predicates = {json.dumps(row.get("branch_predicate", {}), sort_keys=True) for row in duplicates}
        if (
            len(weights) != 1
            or len(recurrence_cycles) != 1
            or len(weight_semantics) != 1
            or len(pipe_pairs) != 1
            or len(predicates) != 1
        ):
            raise ValueError(
                "candidate records joined to one recurrence edge but have inconsistent metadata: "
                f"loop={loop_node}, edge={source}->{target}"
            )
        source_pipe, target_pipe = next(iter(pipe_pairs))
        loop_edges.append(
            {
                "loop_node": loop_node,
                "source_node": source,
                "target_node": target,
                "source_pipe": source_pipe,
                "target_pipe": target_pipe,
                "candidate_indices": [row["candidate_index"] for row in duplicates],
                "candidate_count": len(duplicates),
                "source_execution_count": execution_counts[source],
                "target_execution_count": execution_counts[target],
                "estimated_sync_endpoint_executions": (execution_counts[source] + execution_counts[target]),
                "candidate_recurrence_cycles": duplicates[0]["candidate_recurrence_cycles"],
                "weight_cycles": duplicates[0]["weight_cycles"],
                **(
                    {"weight_semantics": next(iter(weight_semantics))}
                    if next(iter(weight_semantics)) != "loop_ii_increment_v1"
                    else {}
                ),
                "branch_predicate": duplicates[0].get("branch_predicate", {}),
            }
        )
    return distance_zero_edges, loop_edges


def _summarize_candidate_weights(
    distance_zero_edges: Sequence[Mapping[str, Any]],
    loop_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return stable unique-edge features for cohort evaluation."""
    positive_distance_zero = [edge for edge in distance_zero_edges if edge["weight_cycles"] > 0]
    positive_loop_recurrences = [edge for edge in loop_edges if edge["weight_cycles"] > 0]
    result = {
        "positive_distance_zero_edge_count": len(positive_distance_zero),
        "positive_loop_recurrence_edge_count": len(positive_loop_recurrences),
        "distance_zero_weight_sum_cycles": sum(edge["weight_cycles"] for edge in positive_distance_zero),
        "loop_recurrence_weight_sum_cycles": sum(edge["weight_cycles"] for edge in positive_loop_recurrences),
        "max_distance_zero_weight_cycles": max(
            (edge["weight_cycles"] for edge in positive_distance_zero), default=0.0
        ),
        "max_loop_recurrence_weight_cycles": max(
            (edge["weight_cycles"] for edge in positive_loop_recurrences), default=0.0
        ),
    }
    result["max_candidate_weight_cycles"] = max(
        result["max_distance_zero_weight_cycles"], result["max_loop_recurrence_weight_cycles"]
    )
    result["unique_positive_edge_count"] = (
        result["positive_distance_zero_edge_count"] + result["positive_loop_recurrence_edge_count"]
    )
    return result


def _buffer_pair(first: int, second: int) -> tuple[int, int]:
    return min(first, second), max(first, second)


def _load_promoted_reuse_penalty_entries(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path)
    document = json.loads(source.read_text())
    try:
        raw_penalties = document["problem"]["cost_model"]["reuse_penalties"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: missing problem.cost_model.reuse_penalties") from error
    if not isinstance(raw_penalties, list):
        raise ValueError(f"{source}: reuse_penalties must be an array")
    if not all(isinstance(raw, Mapping) for raw in raw_penalties):
        raise ValueError(f"{source}: every reuse penalty must be an object")
    return raw_penalties


def load_promoted_reuse_penalties(path: str | Path) -> dict[tuple[int, int], float]:
    """Load the pairwise reuse penalties that the DSA solver actually sees."""
    source = Path(path)
    raw_penalties = _load_promoted_reuse_penalty_entries(source)

    penalties: dict[tuple[int, int], float] = {}
    for index, raw in enumerate(raw_penalties):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: reuse penalty {index} must be an object")
        first, second, cost = raw.get("first"), raw.get("second"), _as_number(raw.get("cost"))
        if not isinstance(first, int) or not isinstance(second, int) or cost is None or cost < 0:
            raise ValueError(f"{source}: invalid reuse penalty {index}: {raw}")
        pair = _buffer_pair(first, second)
        if pair in penalties:
            raise ValueError(f"{source}: duplicate reuse penalty pair {pair}")
        penalties[pair] = cost
    return penalties


def load_promoted_reuse_penalty_reasons(path: str | Path) -> dict[tuple[int, int], str]:
    """Load the structured reason attached to every promoted penalty pair."""
    source = Path(path)
    reasons: dict[tuple[int, int], str] = {}
    for index, raw in enumerate(_load_promoted_reuse_penalty_entries(source)):
        first, second, reason = raw.get("first"), raw.get("second"), raw.get("reason")
        if not isinstance(first, int) or not isinstance(second, int) or not isinstance(reason, str):
            raise ValueError(f"{source}: invalid reuse penalty reason at index {index}: {raw}")
        pair = _buffer_pair(first, second)
        if pair in reasons:
            raise ValueError(f"{source}: duplicate reuse penalty pair {pair}")
        reasons[pair] = reason
    return reasons


def _score_penalty_pairs(
    rows: Sequence[Mapping[str, Any]],
    durations: Mapping[int, float],
    existing_edges: Sequence[tuple[int, int, float, str, int | None]],
    base_makespan: float,
    loop_counts: Mapping[int, int],
    sync_latency_cycles: float,
    promoted_penalties: Mapping[tuple[int, int], float] | None,
    promoted_penalty_reasons: Mapping[tuple[int, int], str] | None,
    distance_zero_edge_features: Mapping[tuple[int, int], Mapping[str, Any]],
    loop_edge_features: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse access-site candidates into additive buffer-pair weights."""
    by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        first, second = row.get("first_buffer"), row.get("second_buffer")
        if isinstance(first, int) and isinstance(second, int):
            by_pair[_buffer_pair(first, second)].append(row)

    scored: list[dict[str, Any]] = []
    for pair, pair_rows in sorted(by_pair.items()):
        penalty_reason = promoted_penalty_reasons.get(pair) if promoted_penalty_reasons is not None else None
        if penalty_reason == "pipeline_serialization" and not any(
            row.get("candidate_penalty_reason") == "pipeline_serialization" for row in pair_rows
        ):
            raise ValueError(
                "pipeline-serialization penalty has candidate records but none carries "
                f"pipeline provenance: pair={pair}"
            )
        distance_zero_edges = {
            (int(row["source_node"]), int(row["target_node"]))
            for row in pair_rows
            if row.get("status") == "scored"
        }
        additions = [
            (source, target, sync_latency_cycles, "candidate_sync", None)
            for source, target in sorted(distance_zero_edges)
        ]
        distance_zero_weight = 0.0
        if additions:
            with_pair, _, _, _ = _longest_path(durations, [*existing_edges, *additions])
            distance_zero_weight = max(0.0, with_pair - base_makespan)

        loop_weights: dict[int, float] = {}
        for row in pair_rows:
            if row.get("status") not in {"loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}:
                continue
            loop_id = row.get("loop_node")
            weight = _as_number(row.get("weight_cycles"))
            if not isinstance(loop_id, int) or weight is None:
                raise ValueError(f"invalid loop recurrence score for buffer pair {pair}")
            loop_weights[loop_id] = max(loop_weights.get(loop_id, 0.0), weight)
        exact_occurrence_weights = [
            float(row["weight_cycles"])
            for row in pair_rows
            if row.get("status") == "loop_carried_occurrence_profiled_v2"
        ]
        loop_total_weight = sum(
            weight * max(loop_counts.get(loop_id, 1) - 1, 0) for loop_id, weight in loop_weights.items()
        )
        if exact_occurrence_weights:
            # These weights already cover every active distance-one occurrence
            # in the captured profile and must not be multiplied by trip count
            # a second time. Multiple records are retained conservatively as a
            # sum in this legacy additive diagnostic; the complete-placement
            # DAG remains the authoritative union score.
            loop_total_weight = sum(exact_occurrence_weights)

        loop_schedule_edges = {
            (int(row["loop_node"]), int(row["source_node"]), int(row["target_node"]))
            for row in pair_rows
            if row.get("status") in {"loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}
        }
        estimated_sync_executions = sum(
            int(distance_zero_edge_features[edge]["estimated_sync_endpoint_executions"])
            for edge in distance_zero_edges
        ) + sum(
            int(loop_edge_features[edge]["estimated_sync_endpoint_executions"])
            for edge in loop_schedule_edges
        )

        promoted = promoted_penalties is None or pair in promoted_penalties
        unit_cost = 1.0 if promoted_penalties is None else promoted_penalties.get(pair, 0.0)
        executable_rows = [
            row
            for row in pair_rows
            if row.get("status")
            in {"scored", "loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}
        ]
        not_materialized_count = sum(row.get("status") == "not_materialized_in_schedule" for row in pair_rows)
        scored.append(
            {
                "first_buffer": pair[0],
                "second_buffer": pair[1],
                "promoted_to_dsa_penalty": promoted,
                "unit_cost": unit_cost,
                **({"penalty_reason": penalty_reason} if penalty_reason is not None else {}),
                "candidate_record_count": len(pair_rows),
                "executable_candidate_record_count": len(executable_rows),
                "not_materialized_candidate_record_count": not_materialized_count,
                "executable_in_lowered_schedule": bool(executable_rows),
                "model_status": (
                    "executable_with_proven_dead_records"
                    if executable_rows and not_materialized_count
                    else "executable"
                    if executable_rows
                    else "proven_nonmaterialized"
                ),
                "distance_zero_schedule_edges": [list(edge) for edge in sorted(distance_zero_edges)],
                "loop_carried_schedule_edges": [list(edge) for edge in sorted(loop_schedule_edges)],
                "estimated_sync_endpoint_executions": estimated_sync_executions,
                "distance_zero_weight_cycles": distance_zero_weight,
                "loop_ii_weight_cycles": sum(loop_weights.values()),
                "loop_total_weight_cycles": loop_total_weight,
                "critical_path_weight_cycles": distance_zero_weight + loop_total_weight,
            }
        )

    if promoted_penalties is not None:
        for pair in sorted(set(promoted_penalties) - set(by_pair)):
            reason = promoted_penalty_reasons.get(pair) if promoted_penalty_reasons is not None else None
            if reason != "pipeline_serialization":
                raise ValueError(
                    "promoted reuse penalty has no access-site candidate record and is not a "
                    f"structured pipeline-serialization penalty: pair={pair}, reason={reason!r}"
                )
            # Pipeline-intent relaxation creates these penalties from stage
            # separations. They have no operation-to-operation access record,
            # so keep them in the solver objective while reporting predictors
            # 3--5 as incomplete rather than fabricating a zero-cost edge.
            scored.append(
                {
                    "first_buffer": pair[0],
                    "second_buffer": pair[1],
                    "promoted_to_dsa_penalty": True,
                    "unit_cost": promoted_penalties[pair],
                    "candidate_record_count": 0,
                    "executable_candidate_record_count": 0,
                    "not_materialized_candidate_record_count": 0,
                    "executable_in_lowered_schedule": False,
                    "model_status": "unmodeled_pipeline_serialization",
                    "penalty_reason": reason,
                    "distance_zero_schedule_edges": [],
                    "loop_carried_schedule_edges": [],
                    "estimated_sync_endpoint_executions": 0,
                    "distance_zero_weight_cycles": 0.0,
                    "loop_ii_weight_cycles": 0.0,
                    "loop_total_weight_cycles": 0.0,
                    "critical_path_weight_cycles": 0.0,
                }
            )
        scored.sort(key=lambda row: (row["first_buffer"], row["second_buffer"]))
    return scored


def _index_problem_buffers(raw_buffers: Any) -> dict[int, Mapping[str, Any]]:
    if not isinstance(raw_buffers, list):
        raise ValueError("problem buffers must be an array")
    buffers: dict[int, Mapping[str, Any]] = {}
    for buffer in raw_buffers:
        if not isinstance(buffer, Mapping) or not isinstance(buffer.get("id"), int):
            raise ValueError(f"invalid problem buffer: {buffer!r}")
        buffer_id = buffer["id"]
        if buffer_id in buffers:
            raise ValueError(f"duplicate problem buffer id {buffer_id}")
        if not isinstance(buffer.get("size"), int) or buffer["size"] < 0:
            raise ValueError(f"invalid size for problem buffer {buffer_id}")
        buffers[buffer_id] = buffer
    return buffers


def _index_solution_placements(raw_placements: Any) -> dict[int, Mapping[str, Any]]:
    if not isinstance(raw_placements, list):
        raise ValueError("solution placements must be an array")
    placements: dict[int, Mapping[str, Any]] = {}
    for placement in raw_placements:
        if not isinstance(placement, Mapping) or not isinstance(placement.get("buffer"), int):
            raise ValueError(f"invalid solution placement: {placement!r}")
        buffer_id = placement["buffer"]
        if buffer_id in placements:
            raise ValueError(f"duplicate solution placement for buffer {buffer_id}")
        if (
            not isinstance(placement.get("pool"), int)
            or not isinstance(placement.get("offset"), int)
            or placement["offset"] < 0
        ):
            raise ValueError(f"invalid solution placement for buffer {buffer_id}")
        placements[buffer_id] = placement
    return placements


def _canonical_physical_reuse_groups(
    realized: Sequence[Mapping[str, Any]],
    distance_zero_catalog: Mapping[tuple[int, int], Mapping[str, Any]],
    loop_catalog: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups_by_range: dict[tuple[int, int, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in realized:
        pool = int(row["overlap_pool"])
        first_range = (int(row["first_placement_begin"]), int(row["first_placement_end"]))
        second_range = (int(row["second_placement_begin"]), int(row["second_placement_end"]))
        low_range, high_range = sorted((first_range, second_range))
        groups_by_range[(pool, *low_range, *high_range)].append(row)

    physical_groups: list[dict[str, Any]] = []
    for group_id, ((pool, first_begin, first_end, second_begin, second_end), group_rows) in enumerate(
        sorted(groups_by_range.items())
    ):
        overlap_begin = max(first_begin, second_begin)
        overlap_end = min(first_end, second_end)
        distance_zero_edges = {
            (int(source), int(target))
            for row in group_rows
            for source, target in row.get("distance_zero_schedule_edges", [])
        }
        loop_edges = {
            (int(loop), int(source), int(target))
            for row in group_rows
            for loop, source, target in row.get("loop_carried_schedule_edges", [])
        }
        group_pipe_pair_executions: Counter[str] = Counter()
        for edge_key in distance_zero_edges:
            edge = distance_zero_catalog[edge_key]
            group_pipe_pair_executions[f"{edge['source_pipe']}->{edge['target_pipe']}"] += int(
                edge["estimated_sync_endpoint_executions"]
            )
        for edge_key in loop_edges:
            edge = loop_catalog[edge_key]
            group_pipe_pair_executions[f"{edge['source_pipe']}->{edge['target_pipe']}"] += int(
                edge["estimated_sync_endpoint_executions"]
            )
        physical_groups.append(
            {
                "id": group_id,
                "pool": pool,
                "first_range": [first_begin, first_end],
                "second_range": [second_begin, second_end],
                "overlap_range": [overlap_begin, overlap_end],
                "shared_bytes": overlap_end - overlap_begin,
                "logical_pairs": sorted(
                    [int(row["first_buffer"]), int(row["second_buffer"])] for row in group_rows
                ),
                "logical_pair_count": len(group_rows),
                "logical_unit_cost": sum(float(row["unit_cost"]) for row in group_rows),
                "unique_induced_sync_edge_count": len(distance_zero_edges) + len(loop_edges),
                "estimated_sync_endpoint_executions": sum(group_pipe_pair_executions.values()),
                "estimated_sync_endpoint_executions_by_pipe_pair": dict(
                    sorted(group_pipe_pair_executions.items())
                ),
            }
        )
    return physical_groups


def _realized_edge_explanations(
    realized: Sequence[Mapping[str, Any]],
    physical_groups: Sequence[Mapping[str, Any]],
    candidate_scores: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join logical reuse, physical ranges, lowered sites, and sync evidence."""
    groups_by_pair: dict[tuple[int, int], Mapping[str, Any]] = {}
    for group in physical_groups:
        for raw_pair in group.get("logical_pairs", []):
            if isinstance(raw_pair, list) and len(raw_pair) == 2:
                groups_by_pair[_buffer_pair(int(raw_pair[0]), int(raw_pair[1]))] = group

    candidates_by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidate_scores.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        first, second = candidate.get("first_buffer"), candidate.get("second_buffer")
        if isinstance(first, int) and isinstance(second, int):
            candidates_by_pair[_buffer_pair(first, second)].append(candidate)

    explanations: list[dict[str, Any]] = []
    for pair_row in realized:
        pair = _buffer_pair(int(pair_row["first_buffer"]), int(pair_row["second_buffer"]))
        group = groups_by_pair.get(pair)
        if group is None:
            raise ValueError(f"realized reuse pair {pair} is absent from physical groups")
        candidates = candidates_by_pair.get(pair, [])
        if not candidates:
            model_status = pair_row.get("model_status")
            explanations.append(
                {
                    "first_buffer": pair[0],
                    "second_buffer": pair[1],
                    "physical_group_id": group["id"],
                    "physical_pool": group["pool"],
                    "physical_first_range": group["first_range"],
                    "physical_second_range": group["second_range"],
                    "physical_overlap_range": group["overlap_range"],
                    "shared_bytes": group["shared_bytes"],
                    "candidate_status": model_status,
                    "missing_access_orders": [],
                    "actual_sync_group_ids": [],
                    "critical_path_weight_cycles": 0.0,
                    "slack_basis": model_status,
                }
            )
            continue
        for candidate in candidates:
            recurrence_cycles = candidate.get("candidate_recurrence_cycles")
            base_ii_cycles = candidate.get("base_ii_lower_bound_cycles")
            loop_ii_slack = (
                max(0.0, float(base_ii_cycles) - float(recurrence_cycles))
                if isinstance(base_ii_cycles, (int, float)) and isinstance(recurrence_cycles, (int, float))
                else None
            )
            explanations.append(
                {
                    "first_buffer": pair[0],
                    "second_buffer": pair[1],
                    "physical_group_id": group["id"],
                    "physical_pool": group["pool"],
                    "physical_first_range": group["first_range"],
                    "physical_second_range": group["second_range"],
                    "physical_overlap_range": group["overlap_range"],
                    "shared_bytes": group["shared_bytes"],
                    "candidate_index": candidate.get("candidate_index"),
                    "candidate_status": candidate.get("status"),
                    "prior_access_order": candidate.get("prior_access_order"),
                    "next_access_order": candidate.get("next_access_order"),
                    "missing_access_orders": candidate.get("missing_access_orders", []),
                    "lowered_source_node": candidate.get("source_node"),
                    "lowered_source_operation": candidate.get("source_operation"),
                    "lowered_source_pipe": candidate.get("prior_pipe"),
                    "lowered_target_node": candidate.get("target_node"),
                    "lowered_target_operation": candidate.get("target_operation"),
                    "lowered_target_pipe": candidate.get("next_pipe"),
                    "actual_sync_group_ids": candidate.get("actual_sync_group_ids", []),
                    "source_loop_multiplier": candidate.get("source_execution_count"),
                    "target_loop_multiplier": candidate.get("target_execution_count"),
                    "critical_path_weight_cycles": candidate.get("weight_cycles", 0.0),
                    "critical_path_slack_cycles": candidate.get("critical_path_slack_cycles"),
                    "loop_ii_slack_cycles": loop_ii_slack,
                    "slack_basis": (
                        "loop_initiation_interval"
                        if candidate.get("status")
                        in {"loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}
                        else "whole_function_dag"
                    ),
                }
            )
    return explanations


def _static_dependency_makespan(
    record: Mapping[str, Any],
    operation_durations: Mapping[int, float],
    synchronization_latency_cycles: float,
) -> dict[str, Any]:
    """Evaluate every structured branch scenario of one concrete loop count.

    Operations are cloned for every static loop occurrence. The base graph is
    rebuilt from fixed per-pipe FIFO order and logical-root RAW/WAR/WAW
    dependencies. Existing InsertSync edges and barrier records are ignored.
    Only edges explicitly tagged ``realized_reuse_candidate`` are added, each
    with the calibrated synchronization latency supplied by the duration
    model.
    """
    branch_ids, _ = _branch_alternatives(record)
    if len(branch_ids) > 6:
        raise ValueError(
            f"complete_placement_dag_v5 supports at most 6 symbolic structured branches; "
            f"got {len(branch_ids)}"
        )
    loop_counts, dynamic_loops = _loop_multipliers(record)
    if dynamic_loops:
        raise ValueError(
            "complete_placement_dag_v5 requires concrete loop counts internally; "
            f"dynamic loop nodes: {dynamic_loops[:8]}"
        )
    prepared = _prepare_control_flow_record(record)
    original_nodes = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), int)
    }
    expanded_nodes = sum(
        math.prod(loop_counts[loop] for loop in _expanded_node_loop_stack(node, original_nodes))
        for node in prepared.get("nodes", [])
        if isinstance(node, Mapping)
        and isinstance(node.get("id"), int)
        and node.get("kind") in {"operation", "control_point"}
    )
    if expanded_nodes * (2 ** len(branch_ids)) > _MAX_QUEUE_EVENT_EXPANDED_NODES:
        raise ValueError(
            "complete_placement_dag_v5 expansion exceeds the resource-safe node budget: "
            f"{expanded_nodes} nodes/scenario * {2 ** len(branch_ids)} scenarios, "
            f"limit {_MAX_QUEUE_EVENT_EXPANDED_NODES}"
        )
    scenarios = [
        _score_static_queue_event_scenario(
            record,
            operation_durations,
            {},
            dict(zip(branch_ids, values, strict=True)),
            logical_memory_dependencies=True,
            sync_edge_origin="realized_reuse_candidate",
            sync_edge_latency_cycles=synchronization_latency_cycles,
            include_barrier_sites=False,
        )
        for values in itertools.product((False, True), repeat=len(branch_ids))
    ]
    winner = max(scenarios, key=lambda scenario: float(scenario["full_makespan_cycles"]))
    return {
        **winner,
        "model_version": "complete_placement_dependency_scenarios_v5",
        "branch_policy": "exact_static_induction_profiles_plus_symbolic_path_extremes",
        "exact_iteration_profile_count": len(_branch_iteration_profiles(record)),
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                "branch_choices": scenario["branch_choices"],
                "full_makespan_cycles": scenario["full_makespan_cycles"],
                "full_critical_path": scenario["full_critical_path"],
            }
            for scenario in scenarios
        ],
    }


def _schedule_with_realized_candidate_edges(
    record: Mapping[str, Any],
    distance_zero_edges: Sequence[Mapping[str, Any]],
    loop_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a schedule containing the union of realized candidate edges."""
    augmented = copy.deepcopy(record)
    raw_sync_edges = augmented.setdefault("sync_edges", [])
    if not isinstance(raw_sync_edges, list):
        raise ValueError("schedule sync_edges must be an array")
    group_ids = [
        group.get("id")
        for group in augmented.get("sync_groups", [])
        if isinstance(group, Mapping) and isinstance(group.get("id"), int)
    ]
    next_group = max(group_ids, default=-1) + 1
    loop_ends = {
        node["id"]: node.get("end")
        for node in augmented.get("nodes", [])
        if isinstance(node, Mapping)
        and node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(node.get("id"), int)
    }

    for edge in distance_zero_edges:
        raw_sync_edges.append(
            {
                "source": int(edge["source_node"]),
                "target": int(edge["target_node"]),
                "group": next_group,
                "src_pipe": edge["source_pipe"],
                "dst_pipe": edge["target_pipe"],
                "loop_carried": False,
                "root_buffers": [],
                "analysis_origin": "realized_reuse_candidate",
                "synchronization_latency_cycles": edge.get("synchronization_latency_cycles"),
            }
        )
        next_group += 1
    for edge in loop_edges:
        loop_id = int(edge["loop_node"])
        loop_end = loop_ends.get(loop_id)
        if not isinstance(loop_end, int):
            raise ValueError(f"realized loop candidate refers to unknown loop {loop_id}")
        raw_sync_edges.append(
            {
                "source": int(edge["source_node"]),
                "target": int(edge["target_node"]),
                "group": next_group,
                "src_pipe": edge["source_pipe"],
                "dst_pipe": edge["target_pipe"],
                "loop_carried": True,
                "loop_end": loop_end,
                "root_buffers": [],
                "analysis_origin": "realized_reuse_candidate",
                "synchronization_latency_cycles": edge.get("synchronization_latency_cycles"),
            }
        )
        next_group += 1
    return augmented


def _exact_profiled_loop_candidate_score(
    record: Mapping[str, Any],
    operation_durations: Mapping[int, float],
    *,
    loop_id: int,
    source: int,
    target: int,
    source_pipe: str,
    target_pipe: str,
    synchronization_latency_cycles: float,
) -> dict[str, Any]:
    """Score one distance-one edge over exact per-occurrence branch outcomes."""
    edge = {
        "loop_node": loop_id,
        "source_node": source,
        "target_node": target,
        "source_pipe": source_pipe,
        "target_pipe": target_pipe,
        "synchronization_latency_cycles": synchronization_latency_cycles,
    }
    augmented = _schedule_with_realized_candidate_edges(record, [], [edge])
    baseline = _static_dependency_makespan(record, operation_durations, synchronization_latency_cycles)
    with_candidate = _static_dependency_makespan(
        augmented, operation_durations, synchronization_latency_cycles
    )
    extensions = _scenario_extension_rows(baseline, with_candidate)
    weights = [float(row["critical_path_extension_cycles"]) for row in extensions]
    return {
        "model_version": "exact_profiled_distance_one_v2",
        "loop_node": loop_id,
        "static_trip_count": next(
            (
                node.get("static_trip_count")
                for node in record.get("nodes", [])
                if isinstance(node, Mapping) and node.get("id") == loop_id
            ),
            None,
        ),
        "candidate_recurrence_cycles": None,
        "candidate_recurrence_path": None,
        "base_ii_lower_bound_cycles": None,
        "with_candidate_ii_lower_bound_cycles": None,
        "weight_cycles": max(weights, default=0.0),
        "weight_semantics": "whole_execution_exact_occurrence_extension_v2",
        "scenario_extensions": extensions,
    }


def _dynamic_loop_parameter_groups(
    record: Mapping[str, Any], dynamic_loop_ids: Sequence[int]
) -> dict[str, list[int]]:
    """Group dynamic loops proven to share one raw-PTO trip-count expression.

    A lone legacy dynamic loop remains representable by its loop id. Multiple
    loops require raw-PTO identities: silently treating unrelated loop bounds
    as one parameter would manufacture a runtime relationship that the IR does
    not establish.
    """
    begins = {
        int(node["id"]): node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping)
        and node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(node.get("id"), int)
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for loop_id in dynamic_loop_ids:
        node = begins.get(loop_id)
        if node is None:
            raise ValueError(f"dynamic loop {loop_id} has no LOOP_BEGIN record")
        identity = node.get("dynamic_trip_count_identity")
        if not isinstance(identity, str):
            if len(dynamic_loop_ids) != 1:
                raise ValueError(
                    f"multiple dynamic loops require raw-PTO trip-count identities: missing loop {loop_id}"
                )
            identity = f"legacy-loop-id:{loop_id}"
        groups[identity].append(loop_id)
    return {identity: sorted(loop_ids) for identity, loop_ids in sorted(groups.items())}


def _with_concrete_dynamic_trip_count(
    record: Mapping[str, Any], loop_ids: Sequence[int], trip_count: int
) -> dict[str, Any]:
    """Materialize one probe count for a correlated dynamic-loop group."""
    requested = set(loop_ids)
    if not requested:
        raise ValueError("dynamic-loop concretization requires at least one loop")
    concrete = copy.deepcopy(dict(record))
    matched: set[int] = set()
    for node in concrete.get("nodes", []):
        if (
            isinstance(node, dict)
            and node.get("kind") == "loop"
            and node.get("loop_kind") == "LOOP_BEGIN"
            and node.get("id") in requested
        ):
            node["static_trip_count"] = trip_count
            matched.add(int(node["id"]))
    if matched != requested:
        raise ValueError(
            "dynamic-loop group did not identify every LOOP_BEGIN: "
            f"requested={sorted(requested)}, matched={sorted(matched)}"
        )
    return concrete


def _fit_affine_trip_profile(values: Sequence[float]) -> dict[str, float] | None:
    """Fit ``startup + (N - 1) * steady`` only when all probes agree exactly."""
    if len(values) < 3:
        raise ValueError("an affine trip profile requires at least three probes")
    deltas = [float(right) - float(left) for left, right in itertools.pairwise(values)]
    tolerance = 1e-9 * max(1.0, *(abs(value) for value in values))
    if max(deltas) - min(deltas) > tolerance:
        return None
    return {
        "startup_cycles_at_trip_count_1": float(values[0]),
        "steady_state_cycles_per_additional_iteration": statistics.mean(deltas),
    }


def _branch_scenario_key(choices: Mapping[str | int, Any]) -> tuple[tuple[int, bool], ...]:
    """Return a stable key for one structured branch scenario."""
    normalized: list[tuple[int, bool]] = []
    for branch, value in choices.items():
        if isinstance(branch, str) and branch.isdigit():
            branch = int(branch)
        if not isinstance(branch, int) or not isinstance(value, bool):
            raise ValueError(f"invalid branch scenario choice {branch!r}={value!r}")
        normalized.append((branch, value))
    return tuple(sorted(normalized))


def _scenario_makespans(result: Mapping[str, Any]) -> dict[tuple[tuple[int, bool], ...], float]:
    """Index one concrete-loop score by the branch choices that produced it."""
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("concrete schedule score has no branch scenarios")
    indexed: dict[tuple[tuple[int, bool], ...], float] = {}
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("branch scenario must be an object")
        choices = scenario.get("branch_choices")
        makespan = scenario.get("full_makespan_cycles")
        if not isinstance(choices, Mapping) or not isinstance(makespan, (int, float)):
            raise ValueError("branch scenario is missing choices or makespan")
        key = _branch_scenario_key(choices)
        if key in indexed:
            raise ValueError(f"duplicate branch scenario {key}")
        indexed[key] = float(makespan)
    return indexed


def _scenario_extension_rows(base: Mapping[str, Any], complete: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare base and placement makespans under identical branch choices."""
    base_spans = _scenario_makespans(base)
    complete_spans = _scenario_makespans(complete)
    if base_spans.keys() != complete_spans.keys():
        raise ValueError("base and placement expose different branch scenarios")
    return [
        {
            "branch_choices": {str(branch): value for branch, value in key},
            "base_makespan_cycles": base_spans[key],
            "placement_makespan_cycles": complete_spans[key],
            "critical_path_extension_cycles": max(0.0, complete_spans[key] - base_spans[key]),
        }
        for key in sorted(base_spans)
    ]


def _runtime_parallel_dispatch_score(
    record: Mapping[str, Any], scenario_extensions: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Aggregate observed parallel branch instances into one dispatch score."""

    profile = record.get("runtime_parallel_branch_profile")
    if profile is None:
        return None
    if not isinstance(profile, Mapping):
        raise ValueError("runtime_parallel_branch_profile must be an object")
    raw_scenarios = profile.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("runtime parallel branch profile has no scenarios")

    modeled: dict[tuple[tuple[int, bool], ...], Mapping[str, Any]] = {}
    for row in scenario_extensions:
        choices = row.get("branch_choices")
        if not isinstance(choices, Mapping):
            raise ValueError("modeled branch scenario has no branch choices")
        key = _branch_scenario_key(choices)
        if key in modeled:
            raise ValueError(f"modeled branch scenarios repeat {key}")
        modeled[key] = row

    observed: list[dict[str, Any]] = []
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping):
            raise ValueError("runtime parallel branch profile has a malformed scenario")
        choices = raw.get("branch_choices")
        instance_count = raw.get("instance_count")
        if not isinstance(choices, Mapping) or not isinstance(instance_count, int) or instance_count <= 0:
            raise ValueError("runtime parallel branch profile has an invalid scenario")
        key = _branch_scenario_key(choices)
        row = modeled.get(key)
        if row is None:
            raise ValueError(f"runtime parallel branch scenario {key} is absent from the modeled graph")
        observed.append(
            {
                "branch_choices": {str(branch): value for branch, value in key},
                "instance_count": instance_count,
                "base_makespan_cycles": float(row["base_makespan_cycles"]),
                "placement_makespan_cycles": float(row["placement_makespan_cycles"]),
                "critical_path_extension_cycles": float(row["critical_path_extension_cycles"]),
            }
        )

    base_critical = max(observed, key=lambda row: row["base_makespan_cycles"])
    placement_critical = max(observed, key=lambda row: row["placement_makespan_cycles"])
    base_cycles = float(base_critical["base_makespan_cycles"])
    placement_cycles = float(placement_critical["placement_makespan_cycles"])
    extension = max(0.0, placement_cycles - base_cycles)
    return {
        "schema_version": 1,
        "contract": "parallel_dispatch_max_instance_makespan_v1",
        "profile_semantic_sha256": profile.get("semantic_sha256"),
        "parallel_instance_count": sum(int(row["instance_count"]) for row in observed),
        "observed_scenarios": observed,
        "base_makespan_cycles": base_cycles,
        "placement_makespan_cycles": placement_cycles,
        "critical_path_extension_cycles": extension,
        "relative_critical_path_extension": extension / base_cycles if base_cycles > 0 else None,
        "base_critical_branch_choices": dict(base_critical["branch_choices"]),
        "placement_critical_branch_choices": dict(placement_critical["branch_choices"]),
    }


def _profile_dominance(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> int | None:
    """Compare fitted affine branch/loop profiles for every modeled ``N >= 1``.

    Returns ``-1`` when ``first`` is never worse and is strictly better for at
    least one scenario/trip count, ``1`` for the reverse relation, ``0`` for an
    exact tie, and ``None`` when the ordering depends on the branch or trip
    count under the fitted affine model. No branch frequency or runtime loop
    count is guessed. The caller remains responsible for reporting that the
    affine form was observed over the finite probe range rather than proved
    for the unbounded concrete expansion.
    """

    def index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[tuple[int, bool], ...], tuple[float, float]]:
        result: dict[tuple[tuple[int, bool], ...], tuple[float, float]] = {}
        for row in rows:
            choices = row.get("branch_choices", {})
            if not isinstance(choices, Mapping):
                raise ValueError("placement profile has invalid branch choices")
            startup = row.get("startup_cycles_at_trip_count_1")
            steady = row.get("steady_state_cycles_per_additional_iteration")
            if not isinstance(startup, (int, float)) or not isinstance(steady, (int, float)):
                raise ValueError("placement profile is missing affine coefficients")
            result[_branch_scenario_key(choices)] = (float(startup), float(steady))
        return result

    first_index = index(first)
    second_index = index(second)
    if first_index.keys() != second_index.keys():
        raise ValueError("placement profiles expose different branch scenarios")
    first_never_worse = True
    second_never_worse = True
    first_strict = False
    second_strict = False
    for key in first_index:
        first_startup, first_steady = first_index[key]
        second_startup, second_steady = second_index[key]
        startup_delta = first_startup - second_startup
        steady_delta = first_steady - second_steady
        # An affine delta is <= 0 for every integer N >= 1 iff both its value
        # at N=1 and its slope are <= 0.  The reverse condition is symmetric.
        if startup_delta > 1e-9 or steady_delta > 1e-9:
            first_never_worse = False
        elif startup_delta < -1e-9 or steady_delta < -1e-9:
            first_strict = True
        if startup_delta < -1e-9 or steady_delta < -1e-9:
            second_never_worse = False
        elif startup_delta > 1e-9 or steady_delta > 1e-9:
            second_strict = True
    if first_never_worse and first_strict:
        return -1
    if second_never_worse and second_strict:
        return 1
    if first_never_worse and second_never_worse:
        return 0
    return None


def compare_complete_placement_dag_scores(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two placement scores without guessing runtime control values.

    The result is suitable for frozen model evaluation: ``direction`` is -1
    when ``first`` has a lower extension for every structured branch and every
    dynamic trip count under the fitted affine model, +1 for the reverse, 0
    for a tie, and ``None`` when the ordering depends on runtime control flow.
    Dynamic-loop comparisons are explicitly labelled as relying on the affine
    extrapolation observed at trip counts one through four.
    """
    first_status = first.get("status")
    second_status = second.get("status")
    supported_statuses = {"COMPLETE", "PARAMETRIC_ASSUMPTION"}
    if first_status not in supported_statuses or second_status not in supported_statuses:
        return {
            "status": "INCOMPLETE",
            "direction": None,
            "reason": "both placement scores must be COMPLETE or PARAMETRIC_ASSUMPTION",
        }
    if first_status != second_status:
        return {
            "status": "INCOMPLETE",
            "direction": None,
            "reason": "placement scores use different completeness contracts",
        }
    first_profiles = first.get("critical_path_extension_profiles")
    second_profiles = second.get("critical_path_extension_profiles")
    if not isinstance(first_profiles, list) or not isinstance(second_profiles, list):
        raise ValueError("complete-placement score has no branch/loop extension profiles")
    direction = _profile_dominance(first_profiles, second_profiles)
    parametric = first_status == "PARAMETRIC_ASSUMPTION"
    return {
        "status": (
            "ORDERED_UNDER_PARAMETRIC_ASSUMPTION"
            if parametric and direction is not None
            else "RUNTIME_CONTROL_DEPENDENT_UNDER_PARAMETRIC_ASSUMPTION"
            if parametric
            else "ORDERED"
            if direction is not None
            else "RUNTIME_CONTROL_DEPENDENT"
        ),
        "direction": direction,
        "ordering_contract": ("all_structured_branches_and_affine_extrapolation_after_exact_N1_to_N4_probes"),
        **(
            {"parametric_assumption": "affine_extension_beyond_exact_trip_count_probes_1_to_4"}
            if parametric
            else {}
        ),
    }


def score_complete_placement_dag(  # noqa: PLR0912 - explicit evidence and model gates
    record: Mapping[str, Any],
    model: DurationModel,
    candidate_scores: Mapping[str, Any],
    realized_placement: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the complete union of dependency edges realized by a placement.

    The base graph is reconstructed independently of InsertSync from logical
    SSA/allocation roots and fixed per-pipe order. Every unique distance-zero
    and loop-carried edge induced by ``realized_placement`` is then added at
    once with ``model.sync_latency_cycles``. The result is one finite,
    longest-path calculation rather than a sum of singleton buffer-pair
    penalties. Structured branches are scored as explicit path scenarios.
    Dynamic loops proven to share one raw-PTO bound are represented by one
    affine startup/steady-state parameter when four concrete probes agree with
    that form; no runtime trip count is guessed. This is an explicitly labeled
    extrapolation model, not a proof that an arbitrary max-plus graph remains
    affine beyond the probe range. Independent dynamic parameters fail closed
    rather than being conflated.
    """
    model_version = "complete_placement_dag_v5"
    if candidate_scores.get("schema_version") != 2:
        raise ValueError("complete placement scoring requires candidate score schema_version=2")
    reference_record = copy.deepcopy(dict(record))
    export_limitations = reference_record.get("export_limitations", {})
    incomplete_reference = []
    if isinstance(export_limitations, Mapping):
        if export_limitations.get("branch_nodes_missing", 0):
            incomplete_reference.append("export_limitations.branch_nodes_missing")
    if incomplete_reference:
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "INCOMPLETE",
            "limitations": incomplete_reference,
            "critical_path_extension_cycles": None,
        }
    loop_variant_branches = [
        int(node["id"])
        for node in reference_record.get("nodes", [])
        if isinstance(node, Mapping)
        and node.get("kind") == "branch"
        and node.get("branch_kind") == "IF_BEGIN"
        and node.get("loop_stack")
        and node.get("predicate_loop_invariant") is not True
        and not isinstance(node.get("predicate_iteration_profile"), Mapping)
        and isinstance(node.get("id"), int)
    ]
    if loop_variant_branches:
        limitations = ["loop_variant_branch_profile_not_supported_v1"]
        if not realized_placement.get("synchronization_predictor_coverage_complete", False):
            limitations.append("unmodeled_pipeline_serialization")
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "INCOMPLETE",
            "limitations": limitations,
            "loop_variant_branch_nodes": loop_variant_branches,
            "critical_path_extension_cycles": None,
        }
    missing_contracts = []
    if candidate_scores.get("candidate_edge_semantics") != "pre_insert_sync_address_reuse_hazards_v1":
        missing_contracts.append("candidate_edges_not_derived_from_pre_insert_sync_access_hazards")
    if model.sync_latency_cycles <= 0:
        missing_contracts.append("positive_synchronization_latency_not_calibrated")
    if missing_contracts:
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "INCOMPLETE",
            "limitations": missing_contracts,
            "critical_path_extension_cycles": None,
        }
    pairs = realized_placement.get("pairs")
    if not isinstance(pairs, list) or not all(isinstance(pair, Mapping) for pair in pairs):
        raise ValueError("realized placement must carry an array of scored pairs")
    if not realized_placement.get("synchronization_predictor_coverage_complete", False):
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "INCOMPLETE",
            "limitations": ["unmodeled_pipeline_serialization"],
            "critical_path_extension_cycles": None,
        }

    realized = [pair for pair in pairs if pair.get("reuse_realized")]
    distance_zero_keys = {
        (int(source), int(target))
        for pair in realized
        for source, target in pair.get("distance_zero_schedule_edges", [])
    }
    loop_keys = {
        (int(loop), int(source), int(target))
        for pair in realized
        for loop, source, target in pair.get("loop_carried_schedule_edges", [])
    }
    distance_zero_catalog = {
        (int(edge["source_node"]), int(edge["target_node"])): edge
        for edge in candidate_scores.get("distance_zero_edges", [])
        if isinstance(edge, Mapping)
    }
    loop_catalog = {
        (int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])): edge
        for edge in candidate_scores.get("loop_recurrence_edges", [])
        if isinstance(edge, Mapping)
    }
    missing_distance_zero = distance_zero_keys - set(distance_zero_catalog)
    missing_loop = loop_keys - set(loop_catalog)
    if missing_distance_zero or missing_loop:
        raise ValueError(
            "complete placement refers to absent candidate edges: "
            f"distance_zero={sorted(missing_distance_zero)[:8]}, "
            f"loop={sorted(missing_loop)[:8]}"
        )

    _, _, dynamic_loops = estimate_node_durations(reference_record, model)
    try:
        dynamic_loop_groups = _dynamic_loop_parameter_groups(reference_record, dynamic_loops)
    except ValueError as error:
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "INCOMPLETE",
            "limitations": ["dynamic_loop_parameter_identity_missing"],
            "dynamic_loop_ids": dynamic_loops,
            "detail": str(error),
            "critical_path_extension_cycles": None,
        }
    if len(dynamic_loop_groups) > 1:
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "INCOMPLETE",
            "limitations": ["independent_dynamic_loop_parameters_not_supported_v1"],
            "dynamic_loop_ids": dynamic_loops,
            "dynamic_loop_groups": dynamic_loop_groups,
            "critical_path_extension_cycles": None,
        }
    selected_distance_zero = [dict(distance_zero_catalog[key]) for key in sorted(distance_zero_keys)]
    selected_loop = [dict(loop_catalog[key]) for key in sorted(loop_keys)]
    for edge in selected_distance_zero:
        edge["synchronization_latency_cycles"] = model.sync_latency_cycles
    for edge in selected_loop:
        edge["synchronization_latency_cycles"] = model.sync_latency_cycles
    augmented = _schedule_with_realized_candidate_edges(
        reference_record, selected_distance_zero, selected_loop
    )
    if dynamic_loops:
        parameter_identity, loop_ids = next(iter(dynamic_loop_groups.items()))
        probes: list[dict[str, Any]] = []
        for trip_count in range(1, 5):
            concrete_base = _with_concrete_dynamic_trip_count(reference_record, loop_ids, trip_count)
            concrete_placement = _with_concrete_dynamic_trip_count(augmented, loop_ids, trip_count)
            operation_durations, _, unresolved = estimate_node_durations(concrete_base, model)
            if unresolved:
                raise ValueError(f"dynamic-loop concretization left unresolved loops: {unresolved}")
            base_probe = _static_dependency_makespan(
                concrete_base, operation_durations, model.sync_latency_cycles
            )
            complete_probe = _static_dependency_makespan(
                concrete_placement, operation_durations, model.sync_latency_cycles
            )
            scenario_extensions = _scenario_extension_rows(base_probe, complete_probe)
            probes.append(
                {
                    "trip_count": trip_count,
                    "branch_scenarios": scenario_extensions,
                }
            )
        scenario_keys = [_branch_scenario_key(row["branch_choices"]) for row in probes[0]["branch_scenarios"]]
        profiles: list[dict[str, Any]] = []
        for scenario_index, scenario_key in enumerate(scenario_keys):
            scenario_probes = [probe["branch_scenarios"][scenario_index] for probe in probes]
            if any(_branch_scenario_key(row["branch_choices"]) != scenario_key for row in scenario_probes):
                raise ValueError("dynamic loop probes changed branch-scenario order")
            base_profile = _fit_affine_trip_profile(
                [float(row["base_makespan_cycles"]) for row in scenario_probes]
            )
            placement_profile = _fit_affine_trip_profile(
                [float(row["placement_makespan_cycles"]) for row in scenario_probes]
            )
            extension_profile = _fit_affine_trip_profile(
                [float(row["critical_path_extension_cycles"]) for row in scenario_probes]
            )
            if extension_profile is None:
                return {
                    "schema_version": 1,
                    "model_version": model_version,
                    "status": "INCOMPLETE",
                    "limitations": ["dynamic_loop_latency_not_affine_over_trip_counts_1_to_4"],
                    "dynamic_loop_ids": dynamic_loops,
                    "dynamic_loop_groups": dynamic_loop_groups,
                    "dynamic_loop_probes": probes,
                    "critical_path_extension_cycles": None,
                }
            profiles.append(
                {
                    "branch_choices": {str(branch): value for branch, value in scenario_key},
                    "base_makespan_affine": base_profile,
                    "placement_makespan_affine": placement_profile,
                    **extension_profile,
                }
            )
        if any(
            profile["startup_cycles_at_trip_count_1"] < -1e-9
            or profile["steady_state_cycles_per_additional_iteration"] < -1e-9
            for profile in profiles
        ):
            return {
                "schema_version": 1,
                "model_version": model_version,
                "status": "INCOMPLETE",
                "limitations": ["dynamic_loop_extension_not_monotone_nonnegative"],
                "dynamic_loop_ids": dynamic_loops,
                "dynamic_loop_groups": dynamic_loop_groups,
                "dynamic_loop_probes": probes,
                "critical_path_extension_cycles": None,
            }
        additive_cycles = float(realized_placement.get("critical_path_realized_cost_cycles", 0.0))
        one_profile = profiles[0] if len(profiles) == 1 else None
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "PARAMETRIC_ASSUMPTION",
            "reference_graph_contract": "non_reusing_logical_ssa_memory_plus_fixed_pipe_order",
            "candidate_edge_semantics": "pre_insert_sync_address_reuse_hazards_v1",
            "loop_policy": "correlated_dynamic_loop_affine_probe_model_v1",
            "branch_policy": "exact_static_induction_profiles_plus_symbolic_path_extremes",
            "insert_sync_policy": "not_consulted",
            "synchronization_latency_cycles": model.sync_latency_cycles,
            "dynamic_loop_ids": dynamic_loops,
            "dynamic_loop_groups": dynamic_loop_groups,
            "dynamic_trip_count_identity": parameter_identity,
            "dynamic_trip_count_symbol": (
                f"N{loop_ids[0]}"
                if parameter_identity.startswith("legacy-loop-id:")
                else f"N_{parameter_identity.removeprefix('loop-trip-v1:')[:12]}"
            ),
            "dynamic_loop_probes": probes,
            "dynamic_loop_probe_trip_counts": [1, 2, 3, 4],
            "parametric_assumption": ("placement_extension_affine_beyond_exact_trip_count_probes_1_to_4"),
            "critical_path_extension_profiles": profiles,
            "critical_path_extension_affine": (
                {
                    "startup_cycles_at_trip_count_1": one_profile["startup_cycles_at_trip_count_1"],
                    "steady_state_cycles_per_additional_iteration": one_profile[
                        "steady_state_cycles_per_additional_iteration"
                    ],
                }
                if one_profile is not None
                else None
            ),
            "parametric_ranking_key": (
                [
                    one_profile["steady_state_cycles_per_additional_iteration"],
                    one_profile["startup_cycles_at_trip_count_1"],
                ]
                if one_profile is not None
                else None
            ),
            "parametric_ranking_order": ("branch_dominance_under_fitted_affine_model_for_N_ge_1"),
            "critical_path_extension_cycles": None,
            "pairwise_additive_cost_cycles": additive_cycles,
            "nonadditive_interaction_cycles": None,
            "realized_distance_zero_edge_count": len(selected_distance_zero),
            "realized_loop_carried_edge_count": len(selected_loop),
            "distance_zero_edges": [
                [int(edge["source_node"]), int(edge["target_node"])] for edge in selected_distance_zero
            ],
            "loop_carried_edges": [
                [int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])]
                for edge in selected_loop
            ],
        }

    operation_durations, _, unresolved = estimate_node_durations(reference_record, model)
    if unresolved:
        raise ValueError(f"unexpected unresolved loops: {unresolved}")
    base = _static_dependency_makespan(reference_record, operation_durations, model.sync_latency_cycles)
    complete = _static_dependency_makespan(augmented, operation_durations, model.sync_latency_cycles)
    scenario_extensions = _scenario_extension_rows(base, complete)
    runtime_dispatch_score = _runtime_parallel_dispatch_score(reference_record, scenario_extensions)
    base_cycles = float(base["full_makespan_cycles"])
    complete_cycles = float(complete["full_makespan_cycles"])
    additive_cycles = float(realized_placement.get("critical_path_realized_cost_cycles", 0.0))
    extensions = [float(row["critical_path_extension_cycles"]) for row in scenario_extensions]
    extension = extensions[0] if len(set(extensions)) == 1 else None
    profiles = [
        {
            "branch_choices": row["branch_choices"],
            "startup_cycles_at_trip_count_1": row["critical_path_extension_cycles"],
            "steady_state_cycles_per_additional_iteration": 0.0,
        }
        for row in scenario_extensions
    ]
    return {
        "schema_version": 1,
        "model_version": model_version,
        "status": "COMPLETE",
        "reference_graph_contract": "non_reusing_logical_ssa_memory_plus_fixed_pipe_order",
        "candidate_edge_semantics": "pre_insert_sync_address_reuse_hazards_v1",
        "loop_policy": "finite_static_expansion",
        "branch_policy": "exact_static_induction_profiles_plus_symbolic_path_extremes",
        "insert_sync_policy": "not_consulted",
        "synchronization_latency_cycles": model.sync_latency_cycles,
        "base_makespan_cycles": base_cycles,
        "placement_makespan_cycles": complete_cycles,
        "critical_path_extension_cycles": extension,
        "critical_path_extension_range_cycles": [min(extensions), max(extensions)],
        "critical_path_extension_profiles": profiles,
        "branch_scenario_extensions": scenario_extensions,
        "runtime_parallel_dispatch_score": runtime_dispatch_score,
        "pairwise_additive_cost_cycles": additive_cycles,
        "nonadditive_interaction_cycles": (extension - additive_cycles if extension is not None else None),
        "realized_distance_zero_edge_count": len(selected_distance_zero),
        "realized_loop_carried_edge_count": len(selected_loop),
        "expanded_node_count": complete["expanded_node_count"],
        "expanded_pipe_order_edge_count": base["expanded_stream_edge_count"],
        "expanded_logical_memory_edge_count": base["expanded_logical_memory_edge_count"],
        "expanded_reuse_synchronization_edge_count": complete["expanded_sync_edge_count"],
        "base_critical_path": base["full_critical_path"],
        "placement_critical_path": complete["full_critical_path"],
        "distance_zero_edges": [
            [int(edge["source_node"]), int(edge["target_node"])] for edge in selected_distance_zero
        ],
        "loop_carried_edges": [
            [int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])]
            for edge in selected_loop
        ],
    }


def score_realized_reuse(
    problem_path: str | Path,
    solution_path: str | Path,
    candidate_scores: Mapping[str, Any],
    *,
    schedule_record: Mapping[str, Any] | None = None,
    model: DurationModel | None = None,
    promoted_only: bool = True,
) -> dict[str, Any]:
    """Score candidate reuse pairs physically realized by one placement.

    Production scoring uses only solver-promoted penalties. Graph-conformance
    auditing passes ``promoted_only=False`` because a complete placement graph
    must also account for realized zero-penalty candidates.
    """
    schema_version = candidate_scores.get("schema_version")
    if schema_version != 2:
        raise ValueError(
            "candidate score schema is incompatible with synchronization endpoint scoring: "
            f"expected schema_version=2, got {schema_version!r}; regenerate candidate scores"
        )
    problem_source = Path(problem_path)
    solution_source = Path(solution_path)
    problem = json.loads(problem_source.read_text())
    solution = json.loads(solution_source.read_text())
    try:
        raw_buffers = problem["problem"]["buffers"]
        raw_placements = solution["placements"]
        pair_weights = candidate_scores["penalty_pair_weights"]
    except (KeyError, TypeError) as error:
        raise ValueError("problem, solution, or candidate score has an invalid schema") from error
    buffers = _index_problem_buffers(raw_buffers)
    placements = _index_solution_placements(raw_placements)

    problem_instance = problem.get("instance")
    solution_instance = solution.get("instance")
    if isinstance(problem_instance, str) and solution_instance != problem_instance:
        raise ValueError(
            f"solution instance does not match problem: {solution_instance!r} != {problem_instance!r}"
        )
    expected_fingerprint = problem.get("problem_fingerprint")
    if expected_fingerprint is None and isinstance(problem_instance, Mapping):
        expected_fingerprint = problem_instance.get("fingerprint")
    actual_fingerprint = solution.get("problem_fingerprint")
    if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "solution problem fingerprint does not match problem: "
            f"{actual_fingerprint!r} != {expected_fingerprint!r}"
        )
    if not isinstance(pair_weights, list):
        raise ValueError("candidate score penalty_pair_weights must be an array")

    rows: list[dict[str, Any]] = []
    for pair_weight in pair_weights:
        if not isinstance(pair_weight, Mapping):
            continue
        if promoted_only and not pair_weight.get("promoted_to_dsa_penalty"):
            continue
        first, second = pair_weight.get("first_buffer"), pair_weight.get("second_buffer")
        if (
            first not in buffers
            or second not in buffers
            or first not in placements
            or second not in placements
        ):
            raise ValueError(f"placement is missing promoted reuse pair {first, second}")
        first_placement, second_placement = placements[first], placements[second]
        first_begin, second_begin = first_placement["offset"], second_placement["offset"]
        first_end = first_begin + buffers[first]["size"]
        second_end = second_begin + buffers[second]["size"]
        overlap = (
            first_placement["pool"] == second_placement["pool"]
            and first_begin < second_end
            and second_begin < first_end
        )
        overlap_begin = max(first_begin, second_begin) if overlap else None
        overlap_end = min(first_end, second_end) if overlap else None
        rows.append(
            {
                **pair_weight,
                "reuse_realized": overlap,
                "overlap_pool": int(first_placement["pool"]) if overlap else None,
                "first_placement_begin": first_begin,
                "first_placement_end": first_end,
                "second_placement_begin": second_begin,
                "second_placement_end": second_end,
                "overlap_begin": overlap_begin,
                "overlap_end": overlap_end,
                "overlap_bytes": (
                    overlap_end - overlap_begin
                    if overlap_begin is not None and overlap_end is not None
                    else 0
                ),
            }
        )

    realized = [row for row in rows if row["reuse_realized"]]
    distance_zero_catalog = {
        (int(edge["source_node"]), int(edge["target_node"])): edge
        for edge in candidate_scores.get("distance_zero_edges", [])
        if isinstance(edge, Mapping)
    }
    loop_catalog = {
        (int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])): edge
        for edge in candidate_scores.get("loop_recurrence_edges", [])
        if isinstance(edge, Mapping)
    }
    realized_distance_zero_edges = {
        (int(source), int(target))
        for row in realized
        for source, target in row.get("distance_zero_schedule_edges", [])
    }
    realized_loop_edges = {
        (int(loop), int(source), int(target))
        for row in realized
        for loop, source, target in row.get("loop_carried_schedule_edges", [])
    }
    missing_distance_zero = realized_distance_zero_edges - set(distance_zero_catalog)
    missing_loop = realized_loop_edges - set(loop_catalog)
    if missing_distance_zero or missing_loop:
        raise ValueError(
            "realized reuse refers to schedule edges absent from the candidate catalogs: "
            f"distance_zero={sorted(missing_distance_zero)[:8]}, loop={sorted(missing_loop)[:8]}"
        )

    pipe_pair_executions: Counter[str] = Counter()
    for edge_key in realized_distance_zero_edges:
        edge = distance_zero_catalog[edge_key]
        pipe_pair_executions[f"{edge['source_pipe']}->{edge['target_pipe']}"] += int(
            edge["estimated_sync_endpoint_executions"]
        )
    for edge_key in realized_loop_edges:
        edge = loop_catalog[edge_key]
        pipe_pair_executions[f"{edge['source_pipe']}->{edge['target_pipe']}"] += int(
            edge["estimated_sync_endpoint_executions"]
        )
    physical_groups = _canonical_physical_reuse_groups(realized, distance_zero_catalog, loop_catalog)
    executable_realized = [row for row in realized if row.get("executable_in_lowered_schedule")]
    unmodeled_pipeline_realized = [
        row for row in realized if row.get("model_status") == "unmodeled_pipeline_serialization"
    ]
    executable_physical_groups = _canonical_physical_reuse_groups(
        executable_realized, distance_zero_catalog, loop_catalog
    )
    edge_explanations = _realized_edge_explanations(realized, physical_groups, candidate_scores)
    realized_without_induced_edge = [
        row
        for row in realized
        if not row.get("distance_zero_schedule_edges") and not row.get("loop_carried_schedule_edges")
    ]
    realized_by_penalty_reason = Counter(str(row.get("penalty_reason", "unspecified")) for row in realized)
    executable_realized_by_penalty_reason = Counter(
        str(row.get("penalty_reason", "unspecified")) for row in executable_realized
    )
    result = {
        "schema_version": 2,
        "model_version": candidate_scores.get("model_version"),
        "problem": str(problem_source),
        "solution": str(solution_source),
        "pair_selection_policy": ("solver_promoted_penalties" if promoted_only else "all_candidate_pairs"),
        "candidate_pair_count": len(rows),
        "promoted_pair_count": sum(bool(row.get("promoted_to_dsa_penalty")) for row in rows),
        "realized_pair_count": len(realized),
        "realized_pair_count_by_penalty_reason": dict(sorted(realized_by_penalty_reason.items())),
        "canonical_physical_reuse_group_count": len(physical_groups),
        "executable_realized_pair_count": len(executable_realized),
        "executable_realized_pair_count_by_penalty_reason": dict(
            sorted(executable_realized_by_penalty_reason.items())
        ),
        "executable_canonical_physical_reuse_group_count": len(executable_physical_groups),
        "synchronization_predictor_coverage_complete": not unmodeled_pipeline_realized,
        "unmodeled_pipeline_serialization_realized_pair_count": len(unmodeled_pipeline_realized),
        "unmodeled_pipeline_serialization_realized_cost": sum(
            float(row["unit_cost"]) for row in unmodeled_pipeline_realized
        ),
        "realized_pair_count_without_induced_sync_edge": len(realized_without_induced_edge),
        "unit_realized_cost": sum(float(row["unit_cost"]) for row in realized),
        "executable_unit_realized_cost": sum(float(row["unit_cost"]) for row in executable_realized),
        "unit_realized_cost_without_induced_sync_edge": sum(
            float(row["unit_cost"]) for row in realized_without_induced_edge
        ),
        "critical_path_realized_cost_cycles": sum(
            float(row["critical_path_weight_cycles"]) for row in realized
        ),
        "unique_induced_sync_edge_count": len(realized_distance_zero_edges) + len(realized_loop_edges),
        "sync_endpoint_estimator_version": "uncoalesced_source_plus_target_static_executions_v1",
        "estimated_sync_endpoint_executions": sum(pipe_pair_executions.values()),
        "estimated_sync_endpoint_executions_by_pipe_pair": dict(sorted(pipe_pair_executions.items())),
        "canonical_physical_reuse_groups": physical_groups,
        "executable_canonical_physical_reuse_groups": executable_physical_groups,
        "edge_explanations": edge_explanations,
        "pairs": rows,
    }
    if (schedule_record is None) != (model is None):
        raise ValueError("complete placement scoring requires both schedule_record and model")
    if schedule_record is not None and model is not None:
        complete_dag = score_complete_placement_dag(schedule_record, model, candidate_scores, result)
        result["complete_placement_dag"] = complete_dag
        result["complete_placement_critical_path_cycles"] = complete_dag["critical_path_extension_cycles"]
    return result


_PTOAS_GRAPH_FUNCTION_RE = re.compile(r"^KernelScheduleGraph @(?P<function>\S+)")
_PTOAS_GRAPH_ACCESS_NODE_RE = re.compile(
    r"^\s*node\[(?P<node>\d+)\].*\bpypto_access_order=(?P<access>\d+)\b"
)


def _load_ptoas_access_node_map(path: str | Path, *, function: str) -> dict[int, int]:
    """Load a fail-closed PyPTO-access-order to PTOAS-node join from graph text."""
    source = Path(path)
    declared_function: str | None = None
    mapping: dict[int, int] = {}
    for line in source.read_text().splitlines():
        if match := _PTOAS_GRAPH_FUNCTION_RE.match(line):
            if declared_function is not None:
                raise ValueError(f"{source}: expected exactly one KernelScheduleGraph record")
            declared_function = match["function"]
            continue
        if match := _PTOAS_GRAPH_ACCESS_NODE_RE.match(line):
            node, access = int(match["node"]), int(match["access"])
            if access in mapping:
                raise ValueError(f"{source}: pypto access {access} maps to multiple PTOAS nodes")
            mapping[access] = node
    if declared_function != function:
        raise ValueError(f"{source}: graph function {declared_function!r} does not match {function!r}")
    if not mapping:
        raise ValueError(f"{source}: graph has no pypto_access_order node provenance")
    return mapping


def emit_ptoas_placement_reuse_edges(
    candidate_scores: Mapping[str, Any],
    problem_path: str | Path,
    solution_path: str | Path,
    ptoas_graph_path: str | Path,
    *,
    function: str | None = None,
) -> dict[str, Any]:
    """Emit one provenance-backed PTOAS reuse-edge file for a complete placement.

    PyPTO owns the DSA problem/solution and therefore determines which logical
    buffer pairs physically overlap. PTOAS owns the operation DAG and resolves
    the resulting access-order join to its node IDs. This adapter refuses a
    physical reuse pair whose candidate relation cannot be materialized as a
    distance-zero RAW/WAR/WAW dependency.

    Args:
        candidate_scores: Schema-v2 result from ``score-candidates``.
        problem_path: Exported DSA problem containing candidate provenance.
        solution_path: Complete DSA placement solution.
        ptoas_graph_path: Text graph from ``pto-print-kernel-schedule-graph``.
        function: Optional expected function name.

    Returns:
        JSON document accepted by PTOAS ``--placement-reuse-edges``.
    """
    if candidate_scores.get("schema_version") != 2:
        raise ValueError("PTOAS edge export requires candidate score schema_version=2")
    candidate_function = candidate_scores.get("function")
    if not isinstance(candidate_function, str) or not candidate_function:
        raise ValueError("candidate scores have no function identity")
    if function is not None and function != candidate_function:
        raise ValueError(f"requested function {function!r} does not match {candidate_function!r}")
    access_nodes = _load_ptoas_access_node_map(ptoas_graph_path, function=candidate_function)
    candidates = load_candidate_records(problem_path)
    realized = score_realized_reuse(
        problem_path,
        solution_path,
        candidate_scores,
        promoted_only=False,
    )
    realized_pairs = {
        _buffer_pair(int(pair["first_buffer"]), int(pair["second_buffer"]))
        for pair in realized["pairs"]
        if isinstance(pair, Mapping) and pair.get("reuse_realized")
    }
    rows_by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_scores.get("candidates", []):
        if not isinstance(row, Mapping):
            raise ValueError("candidate scores contain a non-object candidate row")
        first, second = row.get("first_buffer"), row.get("second_buffer")
        if isinstance(first, int) and isinstance(second, int):
            rows_by_pair[_buffer_pair(first, second)].append(row)

    by_edge: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    represented_pairs: set[tuple[int, int]] = set()
    for pair in sorted(realized_pairs):
        rows = rows_by_pair.get(pair, [])
        materialized = [row for row in rows if row.get("status") == "scored"]
        if not materialized:
            statuses = sorted({str(row.get("status")) for row in rows})
            raise ValueError(
                f"realized reuse pair {pair} has no materialized distance-zero candidate; statuses={statuses}"
            )
        represented_pairs.add(pair)
        for row in materialized:
            index = row.get("candidate_index")
            prior_access = row.get("prior_access_order")
            next_access = row.get("next_access_order")
            if (
                not isinstance(index, int)
                or not 0 <= index < len(candidates)
                or not isinstance(prior_access, int)
                or not isinstance(next_access, int)
            ):
                raise ValueError(f"realized reuse pair {pair} has incomplete candidate provenance")
            source = access_nodes.get(prior_access)
            target = access_nodes.get(next_access)
            if source is None or target is None:
                raise ValueError(
                    f"realized reuse pair {pair} cannot join access orders "
                    f"{prior_access}->{next_access} to PTOAS graph nodes"
                )
            candidate = candidates[index]
            kind = {"write_after_read": "war", "write_after_write": "waw"}.get(candidate.dependence)
            if kind is None:
                raise ValueError(f"unsupported reuse dependence {candidate.dependence!r}")
            provenance = (
                f"buffers={pair[0]},{pair[1]};candidate={index};"
                f"accesses={prior_access},{next_access}"
            )
            by_edge[(source, target, kind)].append(provenance)
    if represented_pairs != realized_pairs:
        raise ValueError("not every realized reuse pair was represented in the PTOAS edge file")
    return {
        "schema_version": 1,
        "function": candidate_function,
        "edges": [
            {
                "source_node": source,
                "target_node": target,
                "kind": kind,
                "provenance": "|".join(sorted(provenances)),
            }
            for (source, target, kind), provenances in sorted(by_edge.items())
        ],
    }


def _topology_only_duration_model(record: Mapping[str, Any]) -> DurationModel:
    """Assign uniform durations so candidate extraction depends only on topology."""
    operation_cycles = {
        _operation_key(str(node["pipe"]), str(node["op_name"])): 1.0
        for node in record.get("nodes", [])
        if isinstance(node, Mapping)
        and node.get("kind") == "operation"
        and isinstance(node.get("pipe"), str)
        and isinstance(node.get("op_name"), str)
    }
    return DurationModel(
        model_version="graph_conformance_topology_only_v1",
        calibration_status="topology_only_not_a_latency_model",
        sync_latency_cycles=1.0,
        operation_cycles=operation_cycles,
    )


def _node_level_logical_memory_edges(record: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return branch-free logical-root RAW/WAR/WAW dependencies."""

    def roots(node: Mapping[str, Any], field: str) -> set[str]:
        accesses = node.get(field, [])
        if not isinstance(accesses, list):
            raise ValueError(f"schedule node {node.get('id')} has invalid {field} metadata")
        result: set[str] = set()
        for access in accesses:
            if not isinstance(access, Mapping):
                raise ValueError(f"schedule node {node.get('id')} has a non-object {field} entry")
            root = access.get("root")
            if not isinstance(root, str) or not root:
                raise ValueError(f"schedule node {node.get('id')} has a {field} entry without a root")
            result.add(root)
        return result

    last_writer: dict[str, int] = {}
    readers_since_write: dict[str, set[int]] = defaultdict(set)
    dependencies: set[tuple[int, int]] = set()
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("kind") != "operation":
            continue
        node_id = node.get("id")
        if not isinstance(node_id, int):
            raise ValueError("schedule operation has no integer id")
        read_roots = roots(node, "uses")
        write_roots = roots(node, "defs")
        for root in read_roots:
            writer = last_writer.get(root)
            if writer is not None and writer != node_id:
                dependencies.add((writer, node_id))
        for root in write_roots:
            writer = last_writer.get(root)
            if writer is not None and writer != node_id:
                dependencies.add((writer, node_id))
            dependencies.update(
                (reader, node_id) for reader in readers_since_write[root] if reader != node_id
            )
        for root in write_roots:
            last_writer[root] = node_id
            readers_since_write[root].clear()
        for root in read_roots - write_roots:
            readers_since_write[root].add(node_id)
    return dependencies


def _edge_reachable(edges: set[tuple[int, int]], source: int, target: int) -> bool:
    """Return whether ``target`` is reachable from ``source`` in a finite graph."""
    if source == target:
        return True
    successors: dict[int, set[int]] = defaultdict(set)
    for predecessor, successor in edges:
        successors[predecessor].add(successor)
    pending = [source]
    visited = {source}
    while pending:
        current = pending.pop()
        for successor in successors[current]:
            if successor == target:
                return True
            if successor not in visited:
                visited.add(successor)
                pending.append(successor)
    return False


def _graph_conformance_base_edges(record: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Construct the node-level no-reuse graph used for conformance checks."""
    prepared = _prepare_control_flow_record(record)
    stream_edges = {
        (int(edge["source"]), int(edge["target"]))
        for edge in prepared.get("stream_edges", [])
        if isinstance(edge, Mapping)
        and isinstance(edge.get("source"), int)
        and isinstance(edge.get("target"), int)
    }
    return stream_edges | _node_level_logical_memory_edges(record)


def _realized_candidate_edge_keys(
    candidate_scores: Mapping[str, Any], realized_placement: Mapping[str, Any]
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int, int], dict[str, Any]]]:
    """Return the candidate catalogs selected by one complete placement."""
    pairs = realized_placement.get("pairs")
    if not isinstance(pairs, list) or not all(isinstance(pair, Mapping) for pair in pairs):
        raise ValueError("realized placement must carry an array of scored pairs")
    realized = [pair for pair in pairs if pair.get("reuse_realized")]
    selected_distance = {
        (int(source), int(target))
        for pair in realized
        for source, target in pair.get("distance_zero_schedule_edges", [])
    }
    selected_loops = {
        (int(loop), int(source), int(target))
        for pair in realized
        for loop, source, target in pair.get("loop_carried_schedule_edges", [])
    }
    distance_catalog = {
        (int(edge["source_node"]), int(edge["target_node"])): dict(edge)
        for edge in candidate_scores.get("distance_zero_edges", [])
        if isinstance(edge, Mapping)
    }
    loop_catalog = {
        (int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])): dict(edge)
        for edge in candidate_scores.get("loop_recurrence_edges", [])
        if isinstance(edge, Mapping)
    }
    missing_distance = selected_distance - set(distance_catalog)
    missing_loops = selected_loops - set(loop_catalog)
    if missing_distance or missing_loops:
        raise ValueError(
            "realized placement refers to absent conformance candidate edges: "
            f"distance_zero={sorted(missing_distance)[:8]}, loops={sorted(missing_loops)[:8]}"
        )
    return (
        {key: distance_catalog[key] for key in sorted(selected_distance)},
        {key: loop_catalog[key] for key in sorted(selected_loops)},
    )


def audit_graph_conformance(  # noqa: PLR0912 - explicit fail-closed classifications
    record: Mapping[str, Any],
    candidate_scores: Mapping[str, Any],
    realized_placement: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare predicted placement hazards with product post-InsertSync edges.

    Version 1 deliberately accepts only branch-free schedules with statically
    bounded loops. Distance-zero dependencies are compared modulo reachability
    in the no-reuse and product graphs. Loop-carried dependencies require an
    exact loop/source/target match; an unexplained product recurrence is
    reported as incomplete rather than guessed to be placement-induced.
    """
    if candidate_scores.get("candidate_edge_semantics") != "pre_insert_sync_address_reuse_hazards_v1":
        raise ValueError("graph conformance requires pre-InsertSync address-reuse hazards")
    classification = classify_static_schedule(record)
    limitations: list[str] = []
    if classification["branch_node_count"]:
        limitations.append("branch_dependent_reachability_not_supported_v1")
    if classification["dynamic_loop_node_count"]:
        limitations.append("dynamic_loop_reachability_not_supported_v1")
    if limitations:
        return {
            "schema_version": 1,
            "contract": "dsa_graph_conformance_v1",
            "status": "INCOMPLETE",
            "function": record.get("function", "<unknown>"),
            "limitations": limitations,
            "schedule_classification": classification,
        }

    selected_distance, selected_loops = _realized_candidate_edge_keys(candidate_scores, realized_placement)
    normalized = _propagate_barrier_dependency_provenance(record)
    prepared = _prepare_control_flow_record(normalized)
    operation_ids = {
        int(node["id"])
        for node in normalized.get("nodes", [])
        if isinstance(node, Mapping) and node.get("kind") == "operation" and isinstance(node.get("id"), int)
    }
    resolved_loop_edges = _effective_loop_carried_edge_indices(normalized)
    actual_distance: set[tuple[int, int]] = set()
    actual_graph_distance: set[tuple[int, int]] = set()
    actual_loops: set[tuple[int, int, int]] = set()
    structural_actual: list[dict[str, Any]] = []
    sync_edges = [edge for edge in normalized.get("sync_edges", []) if isinstance(edge, Mapping)]
    prepared_sync_edges = [edge for edge in prepared.get("sync_edges", []) if isinstance(edge, Mapping)]
    if len(prepared_sync_edges) != len(sync_edges):
        raise ValueError("control-flow preparation changed the number of synchronization edges")
    for edge_index, (edge, prepared_edge) in enumerate(zip(sync_edges, prepared_sync_edges, strict=True)):
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            structural_actual.append(_encoded_sync_edge(edge))
            continue
        recurrence_loop = resolved_loop_edges.get(edge_index)
        if recurrence_loop is not None:
            actual_loops.add((recurrence_loop, source, target))
        else:
            prepared_source, prepared_target = prepared_edge.get("source"), prepared_edge.get("target")
            if isinstance(prepared_source, int) and isinstance(prepared_target, int):
                actual_graph_distance.add((prepared_source, prepared_target))
            if source in operation_ids and target in operation_ids:
                actual_distance.add((source, target))
            else:
                structural_actual.append(_encoded_sync_edge(edge))

    base_edges = _graph_conformance_base_edges(normalized)
    actual_graph = base_edges | actual_graph_distance
    predicted_graph = base_edges | set(selected_distance)
    predicted_rows: list[dict[str, Any]] = []
    for (source, target), edge in selected_distance.items():
        if _edge_reachable(base_edges, source, target):
            status = "REDUNDANT_IN_NO_REUSE_GRAPH"
        elif (source, target) in actual_distance:
            status = "EXACT_PRODUCT_SYNC_EDGE"
        elif _edge_reachable(actual_graph, source, target):
            status = "IMPLIED_OR_COALESCED_BY_PRODUCT_GRAPH"
        else:
            status = "PREDICTED_EDGE_NOT_ENFORCED"
        predicted_rows.append(
            {
                "source": source,
                "target": target,
                "status": status,
                "candidate_indices": list(edge.get("candidate_indices", [])),
            }
        )

    actual_rows: list[dict[str, Any]] = []
    for source, target in sorted(actual_distance):
        if _edge_reachable(base_edges, source, target):
            status = "BASE_GRAPH_DEPENDENCY"
        elif (source, target) in selected_distance:
            status = "EXACT_PREDICTED_EDGE"
        elif _edge_reachable(predicted_graph, source, target):
            status = "IMPLIED_BY_PREDICTED_GRAPH"
        else:
            status = "COMPILER_DEPENDENCY_NOT_PREDICTED"
        actual_rows.append({"source": source, "target": target, "status": status})

    predicted_loop_rows = [
        {
            "loop": loop,
            "source": source,
            "target": target,
            "status": (
                "EXACT_PRODUCT_LOOP_SYNC_EDGE"
                if (loop, source, target) in actual_loops
                else "PREDICTED_LOOP_EDGE_CONFORMANCE_UNRESOLVED_V1"
            ),
            "candidate_indices": list(edge.get("candidate_indices", [])),
        }
        for (loop, source, target), edge in selected_loops.items()
    ]
    if any(row["status"] == "PREDICTED_LOOP_EDGE_CONFORMANCE_UNRESOLVED_V1" for row in predicted_loop_rows):
        limitations.append("loop_recurrence_reachability_not_supported_v1")
    unexplained_actual_loops = sorted(actual_loops - set(selected_loops))
    if unexplained_actual_loops:
        limitations.append("actual_loop_recurrence_origin_unclassified_v1")

    realized_pairs = [
        pair
        for pair in realized_placement.get("pairs", [])
        if isinstance(pair, Mapping) and pair.get("reuse_realized")
    ]
    lowering_eliminations = [
        {
            "first_buffer": pair.get("first_buffer"),
            "second_buffer": pair.get("second_buffer"),
            "model_status": pair.get("model_status"),
        }
        for pair in realized_pairs
        if pair.get("model_status") == "not_materialized_in_schedule"
    ]
    missing_provenance = [
        {
            "first_buffer": pair.get("first_buffer"),
            "second_buffer": pair.get("second_buffer"),
            "model_status": pair.get("model_status"),
        }
        for pair in realized_pairs
        if pair.get("model_status") == "unmodeled_pipeline_serialization"
    ]
    if missing_provenance:
        limitations.append("realized_reuse_missing_operation_provenance")

    predicted_failures = [
        row
        for row in [*predicted_rows, *predicted_loop_rows]
        if row["status"] == "PREDICTED_EDGE_NOT_ENFORCED"
    ]
    actual_failures = [row for row in actual_rows if row["status"] == "COMPILER_DEPENDENCY_NOT_PREDICTED"]
    if predicted_failures or actual_failures:
        status = "FAIL"
    elif limitations:
        status = "INCOMPLETE"
    elif not selected_distance and not selected_loops:
        status = "VACUOUS"
    else:
        status = "PASS"
    return {
        "schema_version": 1,
        "contract": "dsa_graph_conformance_v1",
        "status": status,
        "function": normalized.get("function", "<unknown>"),
        "limitations": limitations,
        "schedule_classification": classification,
        "summary": {
            "predicted_distance_zero_edge_count": len(selected_distance),
            "predicted_loop_edge_count": len(selected_loops),
            "actual_operation_sync_edge_count": len(actual_distance),
            "actual_loop_sync_edge_count": len(actual_loops),
            "predicted_failure_count": len(predicted_failures),
            "actual_unexplained_dependency_count": len(actual_failures),
            "lowering_elimination_count": len(lowering_eliminations),
            "missing_provenance_pair_count": len(missing_provenance),
            "structural_actual_sync_edge_count": len(structural_actual),
        },
        "predicted_edges": predicted_rows,
        "predicted_loop_edges": predicted_loop_rows,
        "actual_edges": actual_rows,
        "unexplained_actual_loop_edges": [
            {"loop": loop, "source": source, "target": target}
            for loop, source, target in unexplained_actual_loops
        ],
        "lowering_eliminations": lowering_eliminations,
        "missing_provenance_pairs": missing_provenance,
        "structural_actual_sync_edges": structural_actual,
    }


def audit_placement_graph_conformance(
    record: Mapping[str, Any],
    problem_path: str | Path,
    solution_path: str | Path,
    *,
    known_nonmaterialized_access_orders: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Build topology-only candidate evidence and audit one product schedule."""
    classification = classify_static_schedule(record)
    if classification["branch_node_count"] or classification["dynamic_loop_node_count"]:
        limitations = []
        if classification["branch_node_count"]:
            limitations.append("branch_dependent_reachability_not_supported_v1")
        if classification["dynamic_loop_node_count"]:
            limitations.append("dynamic_loop_reachability_not_supported_v1")
        return {
            "schema_version": 1,
            "contract": "dsa_graph_conformance_v1",
            "status": "INCOMPLETE",
            "function": record.get("function", "<unknown>"),
            "limitations": limitations,
            "schedule_classification": classification,
        }
    candidates = load_candidate_records(problem_path)
    if not candidates:
        return audit_graph_conformance(
            record,
            {
                "schema_version": 2,
                "candidate_edge_semantics": "pre_insert_sync_address_reuse_hazards_v1",
                "distance_zero_edges": [],
                "loop_recurrence_edges": [],
            },
            {"pairs": []},
        )
    promoted_penalties = load_promoted_reuse_penalties(problem_path)
    promoted_reasons = load_promoted_reuse_penalty_reasons(problem_path)
    model = _topology_only_duration_model(record)
    try:
        candidate_scores = score_reuse_candidates(
            record,
            candidates,
            model,
            promoted_penalties=promoted_penalties,
            known_nonmaterialized_access_orders=known_nonmaterialized_access_orders,
            promoted_penalty_reasons=promoted_reasons,
        )
    except ValueError as error:
        return {
            "schema_version": 1,
            "contract": "dsa_graph_conformance_v1",
            "status": "INCOMPLETE",
            "function": record.get("function", "<unknown>"),
            "limitations": ["candidate_edge_construction_failed"],
            "candidate_edge_construction_error": str(error),
            "schedule_classification": classification,
        }
    realized = score_realized_reuse(
        problem_path,
        solution_path,
        candidate_scores,
        promoted_only=False,
    )
    return audit_graph_conformance(record, candidate_scores, realized)


def score_reuse_candidates(  # noqa: PLR0912, PLR0915 - explicit provenance and fail-closed gates
    record: Mapping[str, Any],
    candidates: Sequence[ReuseCandidateRecord],
    model: DurationModel,
    promoted_penalties: Mapping[tuple[int, int], float] | None = None,
    known_nonmaterialized_access_orders: frozenset[int] = frozenset(),
    promoted_penalty_reasons: Mapping[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Score hypothetical synchronization induced by individual reuse pairs.

    The input schedule is the non-aliasing/reference placement. Existing sync
    dependencies remain in the graph. For each cross-resource candidate, the
    terminal macro phase at the prior access site is connected to the initial
    macro phase at the next access site. The non-negative penalty is the
    resulting longest-path increase. Distance-one candidates use a loop
    initiation-interval lower bound instead of being inserted into the DAG.
    Distance-zero candidates sharing a consumer are also scored as a union so
    the report exposes non-additivity/coalescence instead of pretending pair
    weights are independent.

    Access-site provenance is mandatory. An access order that survives lowering
    but lands on an unexpected pipe is an error. An absent access order also
    fails closed unless the caller supplies independent evidence that lowering
    removed it. Proven non-materialized candidates cannot induce a
    synchronization edge, but their logical unit penalty remains visible for
    comparisons with the solver objective.
    """
    branch_ids, branch_markers = _branch_alternatives(record)
    branch_profiles = _branch_iteration_profiles(record)
    operation_durations, provenance, dynamic_loops = estimate_node_durations(record, model)
    prepared_record = _prepare_control_flow_record(record)
    durations = _schedule_graph_durations(prepared_record, operation_durations)
    node_ids = set(durations)
    existing_edges, edge_diagnostics = _graph_edges(prepared_record, node_ids, include_sync=True)
    existing_edges = [
        (source, target, model.sync_latency_cycles if kind == "sync" else latency, kind, group)
        for source, target, latency, kind, group in existing_edges
    ]
    base_makespan, _, _, base_path = _longest_path(durations, existing_edges)
    indexed_nodes = _site_nodes(record)
    if not indexed_nodes:
        raise ValueError(
            "schedule graph has no pypto.access.N provenance; regenerate PTO with "
            "PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1"
        )

    nodes_by_id = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and node.get("kind") == "operation" and isinstance(node.get("id"), int)
    }
    loops_by_id = {
        node["id"]: node
        for node in record.get("nodes", [])
        if isinstance(node, Mapping)
        and node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and isinstance(node.get("id"), int)
    }
    execution_counts, _ = _node_execution_counts(record)
    actual_sync_groups_by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge in record.get("sync_edges", []):
        if not isinstance(edge, Mapping):
            continue
        source, target, group = edge.get("source"), edge.get("target"), edge.get("group")
        if isinstance(source, int) and isinstance(target, int) and isinstance(group, int):
            actual_sync_groups_by_edge[(source, target)].append(group)
    rows: list[dict[str, Any]] = []
    materialized_access_orders = {access_order for access_order, _ in indexed_nodes}
    grouped_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for index, candidate in enumerate(candidates):
        if candidate.hazard != "cross_resource":
            rows.append(
                {
                    "candidate_index": index,
                    "first_buffer": candidate.first_buffer,
                    "second_buffer": candidate.second_buffer,
                    "status": "same_resource_not_scored",
                    "weight_cycles": 0.0,
                }
            )
            continue
        joined_sites = _join_candidate_access_sites(
            candidate,
            index,
            indexed_nodes,
            nodes_by_id,
            materialized_access_orders,
            known_nonmaterialized_access_orders,
            _schedule_proves_complete_access_provenance(record),
        )
        if isinstance(joined_sites, dict):
            rows.append(joined_sites)
            continue
        (
            prior_nodes,
            prior_pipe,
            prior_pipe_override,
            next_nodes,
            next_pipe,
            next_pipe_override,
        ) = joined_sites
        prior_route_pipe = _route_pipe(candidate.prior_route)
        next_route_pipe = _route_pipe(candidate.next_route)
        source = prior_nodes[-1]
        target = next_nodes[0]
        branch_requirements = _combined_branch_requirements(
            nodes_by_id[source], nodes_by_id[target], branch_markers
        )
        if candidate.loop_carried:
            source_loop_stack = nodes_by_id[source].get("loop_stack", [])
            target_loops = set(nodes_by_id[target].get("loop_stack", []))
            common_loops = [loop for loop in source_loop_stack if loop in target_loops]
            explicitly_identified = [
                loop
                for loop in common_loops
                if loops_by_id.get(loop, {}).get("pypto_source_loop_id") == candidate.source_loop_id
            ]
            loops_with_source_identity = [
                loop
                for loop in common_loops
                if isinstance(loops_by_id.get(loop, {}).get("pypto_source_loop_id"), int)
            ]
            if explicitly_identified:
                resolved_loops = explicitly_identified
            elif not loops_with_source_identity and len(common_loops) == 1:
                # Backward-compatible schedules can still prove an unambiguous
                # single-loop recurrence without relying on positional ids.
                resolved_loops = common_loops
            else:
                resolved_loops = []
            if len(resolved_loops) != 1:
                raise ValueError(
                    f"loop-carried candidate {index} does not resolve to exactly one lowered loop; "
                    "the original-loop identity was not preserved through lowering: "
                    f"edge {source}->{target}, source_loops={source_loop_stack}, "
                    f"target_loops={sorted(target_loops)}, source_loop_id={candidate.source_loop_id}, "
                    f"matching_lowered_loops={explicitly_identified}"
                )
            recurrence_loop = resolved_loops[0]
            source_requirements = _node_branch_requirements(nodes_by_id[source], branch_markers)
            target_requirements = _node_branch_requirements(nodes_by_id[target], branch_markers)
            contradictory_branches = {
                branch
                for branch, value in source_requirements.items()
                if branch in target_requirements and target_requirements[branch] != value
            }
            occurrence_profiled = bool(contradictory_branches) and contradictory_branches <= set(
                branch_profiles
            )
            if branch_requirements is None and not occurrence_profiled:
                raise ValueError(
                    f"loop-carried candidate {index} crosses mutually exclusive branch arms; "
                    "occurrence-aware source/target branch provenance is required: "
                    f"edge {source}->{target}, loop={recurrence_loop}"
                )
            branch_predicate = (
                {str(branch): value for branch, value in sorted(branch_requirements.items())}
                if branch_requirements is not None
                else {}
            )
            if occurrence_profiled:
                recurrence = _exact_profiled_loop_candidate_score(
                    record,
                    operation_durations,
                    loop_id=recurrence_loop,
                    source=source,
                    target=target,
                    source_pipe=prior_pipe,
                    target_pipe=next_pipe,
                    synchronization_latency_cycles=model.sync_latency_cycles,
                )
                recurrence_status = "loop_carried_occurrence_profiled_v2"
            else:
                recurrence = _loop_recurrence_score(
                    prepared_record,
                    operation_durations,
                    existing_edges,
                    loop_id=recurrence_loop,
                    source=source,
                    target=target,
                    candidate_latency=model.sync_latency_cycles,
                )
                recurrence_status = "loop_carried_scored_v1"
            rows.append(
                {
                    "candidate_index": index,
                    "first_buffer": candidate.first_buffer,
                    "second_buffer": candidate.second_buffer,
                    "prior_buffer": candidate.prior_buffer,
                    "next_buffer": candidate.next_buffer,
                    "prior_access_order": candidate.prior_access_order,
                    "next_access_order": candidate.next_access_order,
                    "prior_route_pipe": prior_route_pipe,
                    "next_route_pipe": next_route_pipe,
                    "prior_pipe": prior_pipe,
                    "next_pipe": next_pipe,
                    "prior_pipe_override": prior_pipe_override,
                    "next_pipe_override": next_pipe_override,
                    "source_node": source,
                    "target_node": target,
                    "source_operation": _node_operation_name(nodes_by_id[source]),
                    "target_operation": _node_operation_name(nodes_by_id[target]),
                    "source_execution_count": execution_counts[source],
                    "target_execution_count": execution_counts[target],
                    "actual_sync_group_ids": sorted(actual_sync_groups_by_edge[(source, target)]),
                    "source_macro_nodes": prior_nodes,
                    "target_macro_nodes": next_nodes,
                    "common_loop_nodes": common_loops,
                    "source_loop_id": candidate.source_loop_id,
                    "resolved_recurrence_loop_node": recurrence_loop,
                    "branch_predicate": branch_predicate,
                    "occurrence_profiled_branch_ids": sorted(contradictory_branches),
                    "status": recurrence_status,
                    **recurrence,
                }
            )
            continue
        if branch_requirements is None:
            rows.append(
                {
                    "candidate_index": index,
                    "first_buffer": candidate.first_buffer,
                    "second_buffer": candidate.second_buffer,
                    "prior_buffer": candidate.prior_buffer,
                    "next_buffer": candidate.next_buffer,
                    "prior_access_order": candidate.prior_access_order,
                    "next_access_order": candidate.next_access_order,
                    "prior_route_pipe": prior_route_pipe,
                    "next_route_pipe": next_route_pipe,
                    "source_node": source,
                    "target_node": target,
                    "status": "mutually_exclusive_branch_sites",
                    "branch_predicate": None,
                    "weight_cycles": 0.0,
                }
            )
            continue
        branch_predicate = {str(branch): value for branch, value in sorted(branch_requirements.items())}
        hypothetical = (source, target, model.sync_latency_cycles, "candidate_sync", index)
        try:
            with_candidate, forward, backward, path = _longest_path(
                durations, [*existing_edges, hypothetical]
            )
        except ValueError as error:
            raise ValueError(
                f"candidate {index} creates a non-loop cycle at schedule edge {source}->{target}"
            ) from error
        weight = max(0.0, with_candidate - base_makespan)
        edge_path_cycles = forward[source] + model.sync_latency_cycles + backward[target]
        grouped_edges[target].add((source, target))
        rows.append(
            {
                "candidate_index": index,
                "first_buffer": candidate.first_buffer,
                "second_buffer": candidate.second_buffer,
                "prior_buffer": candidate.prior_buffer,
                "next_buffer": candidate.next_buffer,
                "prior_access_order": candidate.prior_access_order,
                "next_access_order": candidate.next_access_order,
                "prior_route_pipe": prior_route_pipe,
                "next_route_pipe": next_route_pipe,
                "prior_pipe": prior_pipe,
                "next_pipe": next_pipe,
                "prior_pipe_override": prior_pipe_override,
                "next_pipe_override": next_pipe_override,
                "source_node": source,
                "target_node": target,
                "source_operation": _node_operation_name(nodes_by_id[source]),
                "target_operation": _node_operation_name(nodes_by_id[target]),
                "source_execution_count": execution_counts[source],
                "target_execution_count": execution_counts[target],
                "actual_sync_group_ids": sorted(actual_sync_groups_by_edge[(source, target)]),
                "source_macro_nodes": prior_nodes,
                "target_macro_nodes": next_nodes,
                "branch_predicate": branch_predicate,
                "status": "scored",
                "weight_cycles": weight,
                "makespan_with_candidate_cycles": with_candidate,
                "candidate_edge_path_cycles": edge_path_cycles,
                "critical_path_slack_cycles": max(0.0, with_candidate - edge_path_cycles),
                "critical_path_with_candidate": path,
            }
        )

    for row in rows:
        candidate_index = row.get("candidate_index")
        if not isinstance(candidate_index, int) or not 0 <= candidate_index < len(candidates):
            raise ValueError("candidate score row has no valid source candidate index")
        candidate_penalty_reason = candidates[candidate_index].penalty_reason
        if candidate_penalty_reason != "reuse_recognizer":
            row["candidate_penalty_reason"] = candidate_penalty_reason

    consumer_groups: list[dict[str, Any]] = []
    for target, edges in sorted(grouped_edges.items()):
        additions = [
            (source, sink, model.sync_latency_cycles, "candidate_sync", None)
            for source, sink in sorted(edges)
        ]
        combined, _, _, path = _longest_path(durations, [*existing_edges, *additions])
        singleton_weights = {
            (row["source_node"], row["target_node"]): float(row["weight_cycles"])
            for row in rows
            if row.get("status") == "scored" and row.get("target_node") == target
        }
        singleton_sum = sum(singleton_weights.values())
        consumer_groups.append(
            {
                "target_node": target,
                "source_nodes": sorted(source for source, _ in edges),
                "candidate_count": len(edges),
                "combined_weight_cycles": max(0.0, combined - base_makespan),
                "singleton_weight_sum_cycles": singleton_sum,
                "coalescence_cycles": max(0.0, singleton_sum - max(0.0, combined - base_makespan)),
                "critical_path_with_group": path,
            }
        )

    execution_counts, _ = _node_execution_counts(record)
    distance_zero_edges, loop_edge_groups = _deduplicate_scored_candidate_edges(rows, execution_counts)
    weight_summary = _summarize_candidate_weights(distance_zero_edges, loop_edge_groups)
    loop_counts, _ = _loop_multipliers(record)
    penalty_pair_weights = _score_penalty_pairs(
        rows,
        durations,
        existing_edges,
        base_makespan,
        loop_counts,
        model.sync_latency_cycles,
        promoted_penalties,
        promoted_penalty_reasons,
        {(int(edge["source_node"]), int(edge["target_node"])): edge for edge in distance_zero_edges},
        {
            (int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])): edge
            for edge in loop_edge_groups
        },
    )
    baseline_loop_sync_models = _existing_loop_sync_models(
        prepared_record, operation_durations, existing_edges, model.sync_latency_cycles
    )
    if dynamic_loops:
        pre_codegen_summary: dict[str, Any] = {
            "status": "PARAMETRIC_DYNAMIC_LOOP",
            "dynamic_loop_ids": dynamic_loops,
            "detail": "static execution totals are intentionally not fabricated",
        }
    else:
        pre_codegen_summary = _pre_codegen_sync_record_summary(record)

    return {
        "schema_version": 2,
        "model_version": "reuse_penalty_critical_path_v2",
        "candidate_edge_semantics": "pre_insert_sync_address_reuse_hazards_v1",
        "sync_endpoint_estimator_version": "uncoalesced_source_plus_target_static_executions_v1",
        "function": record.get("function", "<unknown>"),
        "duration_model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "base_makespan_cycles": base_makespan,
        "base_critical_path": base_path,
        "dynamic_loop_ids": dynamic_loops,
        "branch_node_ids": branch_ids,
        "branch_edge_policy": "endpoint_predicate_intersection_v1",
        "loop_sync_model_version": "loop_sync_ii_and_boundary_v1",
        "baseline_loop_sync_models": baseline_loop_sync_models,
        **edge_diagnostics,
        **_latency_graph_completeness(prepared_record, edge_diagnostics, baseline_loop_sync_models),
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(
            row.get("status") in {"scored", "loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}
            for row in rows
        ),
        "scored_distance_zero_candidate_count": sum(row.get("status") == "scored" for row in rows),
        "scored_loop_carried_candidate_count": sum(
            row.get("status") in {"loop_carried_scored_v1", "loop_carried_occurrence_profiled_v2"}
            for row in rows
        ),
        "occurrence_profiled_loop_carried_candidate_count": sum(
            row.get("status") == "loop_carried_occurrence_profiled_v2" for row in rows
        ),
        "unscored_loop_carried_candidate_count": 0,
        "not_materialized_candidate_count": sum(
            row.get("status") == "not_materialized_in_schedule" for row in rows
        ),
        "unmodeled_pipeline_serialization_pair_count": sum(
            row["model_status"] == "unmodeled_pipeline_serialization" for row in penalty_pair_weights
        ),
        "candidates": rows,
        "consumer_groups": consumer_groups,
        "distance_zero_edges": distance_zero_edges,
        "loop_recurrence_edges": loop_edge_groups,
        "penalty_pair_weights": penalty_pair_weights,
        "candidate_weight_summary": weight_summary,
        "baseline_pre_codegen_sync_record_summary": pre_codegen_summary,
        "node_durations": {str(node): value for node, value in sorted(provenance.items())},
    }


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def calibrate_from_metrics(paths: Sequence[str | Path], base: DurationModel | None = None) -> DurationModel:
    """Calibrate complete operation signatures from cleaned simulator metrics."""
    model = base or DurationModel()
    by_signature: dict[str, list[float]] = defaultdict(list)
    by_pipe: dict[str, list[float]] = defaultdict(list)
    used_sources: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        instructions = payload.get("instructions")
        if not isinstance(instructions, Mapping):
            raise ValueError(f"{path}: expected cleaned metrics with an instructions object")
        source_used = False
        for records in instructions.values():
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                cycles = _as_number(record.get("cycles"))
                raw_pipe = record.get("pipe")
                if cycles is None or cycles <= 0 or not isinstance(raw_pipe, str):
                    continue
                pipe = _PIPE_ALIASES.get(raw_pipe.upper(), raw_pipe.upper())
                raw_signature = record.get("operation_signature")
                if not isinstance(raw_signature, Mapping):
                    continue
                signature = dict(raw_signature)
                signature["pipe"] = _PIPE_ALIASES.get(
                    str(signature.get("pipe", "")).upper(), signature.get("pipe")
                )
                signature_key = _operation_signature_key(signature)
                by_pipe[pipe].append(cycles)
                by_signature[signature_key].append(cycles)
                source_used = True
        if source_used:
            used_sources.append(str(path))
    if not used_sources:
        raise ValueError("no finite positive complete-signature instruction cycle samples found")

    calibrated_pipes = dict(model.pipe_parameters)
    for pipe, samples in by_pipe.items():
        previous = calibrated_pipes.get(pipe, PipeParameters(0.0, math.inf, 1.0))
        calibrated_pipes[pipe] = PipeParameters(
            startup_cycles=previous.startup_cycles,
            bytes_per_cycle=previous.bytes_per_cycle,
            minimum_cycles=statistics.median(samples),
        )
    signature_cycles = dict(model.operation_signature_cycles)
    signature_cycles.update({key: statistics.median(samples) for key, samples in by_signature.items()})
    return DurationModel(
        schema_version=1,
        model_version="duration_v1",
        calibration_status=(
            "simulator_signature_overrides+pto_isa"
            if model.pto_isa_provider
            else "simulator_complete_signature_medians"
        ),
        sync_latency_cycles=model.sync_latency_cycles,
        barrier_instruction_cycles=model.barrier_instruction_cycles,
        pipe_barrier_cycles=dict(model.pipe_barrier_cycles),
        pipe_parameters=calibrated_pipes,
        operation_cycles=dict(model.operation_cycles),
        operation_signature_cycles=signature_cycles,
        operation_signature_pipeline=dict(model.operation_signature_pipeline),
        calibration_sources=sorted(used_sources),
        pto_isa_provider=model.pto_isa_provider,
    )


_PERF_SIM_PIPE_FROM_EVENT = {
    "Scalar": "PIPE_S",
    "VEC": "PIPE_V",
    "MTE2_AIV": "PIPE_MTE2",
    "MTE2_AIC": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "MTE1": "PIPE_MTE1",
    "FIXP": "PIPE_FIX",
    "CUBE": "PIPE_M",
}
_PERF_SIM_SEQUENCE_RE = re.compile(r":\d+$")
_PERF_SIM_SIGNATURE_EVENT_RE = re.compile(
    r"^(?P<operation>[A-Z0-9_]+)\((?P<rows>\d+)x(?P<cols>\d+),(?P<dtype>[A-Za-z0-9_]+)\)"
    r"\{pipe=(?P<pipe>[^;}]+)(?:;tiles=(?P<tiles>[^;}]*))?(?:;scalars=(?P<scalars>[^}]*))?\}$"
)
_PERF_SIM_TILE_RE = re.compile(
    r"^(?P<dtype>[A-Za-z0-9_]+):(?P<rows>\d+)x(?P<cols>\d+)"
    r":loc=(?P<loc>\d+):storage=(?P<storage_rows>\d+)x(?P<storage_cols>\d+)"
    r":b=(?P<block_layout>\d+):s=(?P<storage_layout>\d+)"
    r":pad=(?P<pad>\d+):compact=(?P<compact>\d+)$"
)
_PERF_SIM_DTYPE_CANONICAL = {
    "f16": "fp16",
    "f32": "fp32",
    "i8": "i8",
    "int8": "i8",
    "ui8": "u8",
    "u8": "u8",
    "uint8": "u8",
    "i16": "i16",
    "int16": "i16",
    "ui16": "u16",
    "u16": "u16",
    "uint16": "u16",
    "i32": "i32",
    "int32": "i32",
    "ui32": "u32",
    "u32": "u32",
    "uint32": "u32",
}
_CALIBRATION_PIPE_MISMATCH_EXCEPTIONS = {
    ("TCI", "PIPE_S", "PIPE_V", "ptoas_v057_tci_schedule_pipe"),
}
_PERF_SIM_SOURCE_WORK_OPERATIONS = {
    "TROWSUM",
    "TROWMAX",
    "TROWMIN",
    "TROWPROD",
    "TCOLSUM",
    "TCOLMAX",
    "TCOLMIN",
    "TCOLPROD",
}
_PERF_SIM_LOWERED_OPERATION = {
    # On A2/A3, PTO-ISA implements TRECIP(dst, src) as TDIVS(dst, 1, src).
    # Keep TRECIP in the schedule-duration key, but require the exact lowered
    # event (including the divisor below) when calibrating it from Perf-Sim.
    "TRECIP": "TDIVS",
}
_TILE_SCOPE_CODE = {
    "vec": 0,
    "mat": 1,
    "left": 2,
    "right": 3,
    "acc": 4,
    "bias": 5,
    "scaling": 6,
    "scale_left": 7,
    "scale_right": 8,
    "ctrl": 9,
}
_BLOCK_LAYOUT_CODE = {"row_major": 0, "col_major": 1}
_STORAGE_LAYOUT_CODE = {"none_box": 0, "row_major": 1, "col_major": 2}
_KEYED_TILE_FIELD_RE = re.compile(r"\b([a-z_]+)=([^,>]+)", re.IGNORECASE)


def _parse_perf_sim_event_prefix(prefix: str) -> dict[str, Any]:
    match = _PERF_SIM_SIGNATURE_EVENT_RE.fullmatch(prefix)
    if match is None:
        raise ValueError(f"malformed Perf-Sim event prefix: {prefix}")
    tiles: list[dict[str, Any]] = []
    raw_tiles = match.group("tiles")
    if raw_tiles:
        for raw_tile in raw_tiles.split(","):
            tile_match = _PERF_SIM_TILE_RE.fullmatch(raw_tile)
            if tile_match is None:
                raise ValueError(f"malformed Perf-Sim tile signature: {raw_tile}")
            tile = tile_match.groupdict()
            tiles.append(
                {
                    "dtype": _PERF_SIM_DTYPE_CANONICAL.get(tile["dtype"], tile["dtype"]),
                    "rows": int(tile["rows"]),
                    "cols": int(tile["cols"]),
                    "loc": int(tile["loc"]),
                    "storage_rows": int(tile["storage_rows"]),
                    "storage_cols": int(tile["storage_cols"]),
                    "block_layout": int(tile["block_layout"]),
                    "storage_layout": int(tile["storage_layout"]),
                    "pad": int(tile["pad"]),
                    "compact": int(tile["compact"]),
                }
            )
    return {
        "operation": match.group("operation"),
        "rows": int(match.group("rows")),
        "cols": int(match.group("cols")),
        "dtype": _PERF_SIM_DTYPE_CANONICAL.get(match.group("dtype"), match.group("dtype")),
        "pipe": match.group("pipe"),
        "tiles": tiles,
        "scalars": match.group("scalars").split(",") if match.group("scalars") else [],
    }


def _canonical_perf_sim_dtype(dtype: str) -> str:
    return _PERF_SIM_DTYPE_CANONICAL.get(dtype.lower(), dtype.lower())


def _expected_perf_sim_tile(raw_type: str) -> dict[str, Any] | None:
    tile = parse_tile_type(raw_type)
    if tile is None:
        return None
    fields = {key.lower(): value.strip().lower() for key, value in _KEYED_TILE_FIELD_RE.findall(raw_type)}
    required = {"loc", "blayout", "slayout", "pad"}
    if not required.issubset(fields):
        raise ValueError(f"calibration requires a keyed tile type with layout and pad metadata: {raw_type}")
    try:
        return {
            "dtype": _canonical_perf_sim_dtype(tile.dtype),
            "rows": tile.rows,
            "cols": tile.cols,
            "loc": _TILE_SCOPE_CODE[fields["loc"]],
            "storage_rows": tile.rows,
            "storage_cols": tile.cols,
            "block_layout": _BLOCK_LAYOUT_CODE[fields["blayout"]],
            "storage_layout": _STORAGE_LAYOUT_CODE[fields["slayout"]],
            "pad": int(fields["pad"], 0),
            "compact": int(fields.get("compact", "0"), 0),
        }
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported keyed tile metadata in calibration type: {raw_type}") from error


def _expected_perf_sim_scalar(raw_constant: Any) -> str | None:
    if not isinstance(raw_constant, str):
        return None
    match = re.fullmatch(r"\s*(.+?)\s*:\s*(f32|f64|i\d+|ui\d+|index)\s*", raw_constant)
    if match is None:
        raise ValueError(f"unsupported static calibration constant: {raw_constant}")
    value, dtype = match.groups()
    if dtype == "f32":
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return f"f32:0x{bits:x}"
    if dtype == "f64":
        bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        return f"f64:0x{bits:x}"
    prefix = "u" if dtype.startswith("ui") else "i"
    return f"{prefix}:{int(value, 0)}"


def _semantic_perf_sim_scalars(operation: Mapping[str, Any], op_name: str) -> list[str]:
    if op_name == "pto.trecip":
        return ["i:1"]
    if op_name == "pto.tcvt":
        attributes = operation.get("attributes", {})
        mode = attributes.get("round_mode") if isinstance(attributes, Mapping) else None
        if mode is None and isinstance((location := operation.get("location")), str):
            match = re.search(r"rmode\s*=\s*#pto<round_mode\s+([A-Z_]+)>", location)
            mode = match.group(1) if match is not None else None
        if not isinstance(mode, str):
            raise ValueError("TCVT calibration requires a static round mode")
        round_modes = {"NONE": 0, "RINT": 1, "ROUND": 2, "FLOOR": 3, "CEIL": 4, "TRUNC": 5, "ODD": 6}
        if mode not in round_modes:
            raise ValueError(f"unsupported TCVT round mode in calibration: {mode}")
        return [f"enum:{round_modes[mode]}"]
    return []


def _expected_perf_sim_tiles(operation: Mapping[str, Any], op_name: str) -> list[dict[str, Any]]:
    """Return tiles in the PTO-ISA recorder's semantic argument order."""
    operand_types = operation.get("operand_types", [])
    result_types = operation.get("result_types", [])
    if not isinstance(operand_types, list) or not isinstance(result_types, list):
        raise ValueError("operation has invalid operand or result types")
    result_tiles = [
        tile
        for raw_type in result_types
        if isinstance(raw_type, str) and (tile := _expected_perf_sim_tile(raw_type)) is not None
    ]
    operand_tiles = [
        tile
        for raw_type in operand_types
        if isinstance(raw_type, str) and (tile := _expected_perf_sim_tile(raw_type)) is not None
    ]
    # The multi-source TMRGSORT C++ API orders (dst, tmp, src...), whereas raw
    # PTO orders ins(src..., tmp) and outs(dst). Canonicalize both to the API
    # order emitted by Perf-Sim. The two-tile block-length form is unchanged.
    if op_name == "pto.tmrgsort" and len(result_tiles) == 1 and len(operand_tiles) >= 3:
        return [result_tiles[0], operand_tiles[-1], *operand_tiles[:-1]]
    return [*result_tiles, *operand_tiles]


def _dynamic_perf_sim_scalar_pattern(raw_type: str) -> str:
    dtype = raw_type.strip().lower()
    if dtype == "index" or re.fullmatch(r"(?:ui|i)\d+", dtype):
        return "integer:*"
    scalar_dtype = {"f16": "fp16", "bf16": "bf16"}.get(dtype, dtype)
    return f"{scalar_dtype}:*"


def _expected_perf_sim_scalar_pattern(operation: Mapping[str, Any], op_name: str) -> list[str]:
    """Return exact static scalars and wildcards for dynamic scalar operands."""
    operand_types = operation.get("operand_types", [])
    constants = operation.get("operand_constants", [])
    if not isinstance(operand_types, list) or not isinstance(constants, list):
        raise ValueError("operation has invalid operand constants or types")
    if len(constants) != len(operand_types):
        raise ValueError(
            "operation operand_constants must align with operand_types: "
            f"{len(constants)} != {len(operand_types)}"
        )

    scalar_operands: list[tuple[str, Any]] = [
        (raw_type, constant)
        for raw_type, constant in zip(operand_types, constants, strict=True)
        if isinstance(raw_type, str) and _PTO_SCALAR_TYPE_FULL_RE.fullmatch(raw_type.strip())
    ]
    # TSETVAL records the tile offset, not the scalar value being written. The
    # value type remains part of operand_types in the complete schedule key.
    if op_name == "pto.tsetval":
        scalar_operands = scalar_operands[:1]
    pattern = [
        _expected_perf_sim_scalar(constant) or _dynamic_perf_sim_scalar_pattern(raw_type)
        for raw_type, constant in scalar_operands
    ]
    pattern.extend(_semantic_perf_sim_scalars(operation, op_name))
    return pattern


def _canonical_perf_sim_scalar(value: str) -> str:
    """Canonicalize equivalent signed/unsigned spellings when type is separate."""
    match = re.fullmatch(r"[iu]:(-?\d+)", value)
    return f"integer:{int(match.group(1))}" if match is not None else value


def _perf_sim_scalars_match(actual: Sequence[str], expected: Sequence[str]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        (
            _canonical_perf_sim_scalar(got).startswith(f"{wanted.removesuffix(':*')}:")
            if wanted.endswith(":*")
            else _canonical_perf_sim_scalar(got) == _canonical_perf_sim_scalar(wanted)
        )
        for got, wanted in zip(actual, expected, strict=True)
    )


def expected_perf_sim_event_signature(node: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact Perf-Sim event contract for one schedule operation."""
    signature = operation_duration_signature(node)
    operation = node.get("operation")
    op_name = node.get("op_name")
    if not isinstance(operation, Mapping) or not isinstance(op_name, str):
        raise ValueError("schedule node lacks operation metadata")
    tiles = _expected_perf_sim_tiles(operation, op_name)
    if not tiles:
        raise ValueError("schedule node has no statically typed tile")
    work_index = 1 if signature["operation"] in _PERF_SIM_SOURCE_WORK_OPERATIONS and len(tiles) > 1 else 0
    work_tile = tiles[work_index]
    return {
        "operation": _PERF_SIM_LOWERED_OPERATION.get(signature["operation"], signature["operation"]),
        "rows": work_tile["rows"],
        "cols": work_tile["cols"],
        "dtype": _canonical_perf_sim_dtype(work_tile["dtype"]),
        "tiles": tiles,
        "scalars": _expected_perf_sim_scalar_pattern(operation, op_name),
    }


def _validate_perf_sim_event_signature(
    event: Mapping[str, Any], node: Mapping[str, Any], *, measurement_index: int, manifest: Path
) -> None:
    operation = node.get("operation")
    if not isinstance(operation, Mapping):
        raise ValueError(f"{manifest}: measurement {measurement_index} lacks operation metadata")
    expected = expected_perf_sim_event_signature(node)
    expected_header = {key: expected[key] for key in ("operation", "rows", "cols", "dtype")}
    actual_header = {key: event[key] for key in expected_header}
    if actual_header != expected_header:
        raise ValueError(
            f"{manifest}: measurement {measurement_index} event work signature differs from schedule: "
            f"expected={expected_header}, actual={actual_header}"
        )
    if event["tiles"] != expected["tiles"]:
        raise ValueError(
            f"{manifest}: measurement {measurement_index} event tile roles/layouts differ from schedule: "
            f"expected={expected['tiles']}, actual={event['tiles']}"
        )

    if not _perf_sim_scalars_match(event["scalars"], expected["scalars"]):
        raise ValueError(
            f"{manifest}: measurement {measurement_index} event constants differ from schedule: "
            f"expected={expected['scalars']}, actual={event['scalars']}"
        )


def extract_signature_calibration(  # noqa: PLR0912 - fail-closed manifest validation is explicit
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Bind Perf-Sim events to exact schedule-node signatures.

    The manifest is deliberately explicit: each measurement names a schedule
    node and the full Perf-Sim event prefix emitted by its matching
    microkernel. This prevents an opcode-family or shape-only join from
    silently merging different modes, constants, layouts, or operand roles.
    """
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{path}: expected calibration manifest schema_version=1")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"{path}: measurements must be a non-empty array")

    instructions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: dict[Path, str] = {}
    schedule_sources: dict[Path, str] = {}
    for index, raw in enumerate(measurements):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: measurement {index} must be an object")
        raw_schedule = raw.get("schedule")
        raw_trace = raw.get("trace")
        function = raw.get("function")
        node_id = raw.get("node_id")
        event_prefix = raw.get("event_name_prefix")
        if (
            not isinstance(raw_schedule, str)
            or not isinstance(raw_trace, str)
            or (function is not None and not isinstance(function, str))
            or not isinstance(node_id, int)
            or not isinstance(event_prefix, str)
            or not event_prefix
        ):
            raise ValueError(f"{path}: measurement {index} has invalid fields")

        schedule_path = Path(raw_schedule)
        trace_path = Path(raw_trace)
        if not schedule_path.is_absolute():
            schedule_path = (path.parent / schedule_path).resolve()
        if not trace_path.is_absolute():
            trace_path = (path.parent / trace_path).resolve()
        record, resolved_function = _resolve_schedule_record(schedule_path, function)
        nodes = [
            node
            for node in record.get("nodes", [])
            if isinstance(node, Mapping) and node.get("id") == node_id and node.get("kind") == "operation"
        ]
        if len(nodes) != 1:
            raise ValueError(f"{path}: measurement {index} does not identify one operation node {node_id}")
        node = nodes[0]
        expected_opcode = _canonical_operation(str(node.get("op_name", "")))
        event_signature = _parse_perf_sim_event_prefix(event_prefix)
        _validate_perf_sim_event_signature(event_signature, node, measurement_index=index, manifest=path)

        trace = json.loads(trace_path.read_text())
        if not isinstance(trace, list):
            raise ValueError(f"{trace_path}: expected a Perf-Sim Chrome trace array")
        samples: list[tuple[float, str]] = []
        for event in trace:
            if not isinstance(event, Mapping) or event.get("ph") != "X":
                continue
            name = event.get("name")
            cycles = _as_number(event.get("dur"))
            if not isinstance(name, str) or cycles is None or cycles <= 0:
                continue
            if _PERF_SIM_SEQUENCE_RE.sub("", name) == event_prefix:
                samples.append((cycles, name))
        if not samples:
            raise ValueError(f"{path}: measurement {index} matched no positive Perf-Sim events")

        event_pipe_name = event_signature["pipe"]
        if event_pipe_name not in _PERF_SIM_PIPE_FROM_EVENT:
            raise ValueError(f"{path}: measurement {index} event has no recognized pipe")
        event_pipe = _PERF_SIM_PIPE_FROM_EVENT[event_pipe_name]
        schedule_pipe = str(node.get("pipe"))
        pipe_mismatch = event_pipe != schedule_pipe
        mismatch_reason = raw.get("pipe_mismatch_reason")
        mismatch_key = (expected_opcode, schedule_pipe, event_pipe, mismatch_reason)
        if pipe_mismatch and mismatch_key not in _CALIBRATION_PIPE_MISMATCH_EXCEPTIONS:
            raise ValueError(
                f"{path}: measurement {index} event pipe {event_pipe} differs from "
                f"schedule pipe {schedule_pipe} without an exact allowlisted exception"
            )
        if not pipe_mismatch and mismatch_reason is not None:
            raise ValueError(f"{path}: measurement {index} declares a vacuous pipe mismatch")

        signature = operation_duration_signature(node)
        key = f"{schedule_path}:{resolved_function}:{node_id}"
        for cycles, event_name in samples:
            instructions[key].append(
                {
                    "pipe": schedule_pipe,
                    "cycles": cycles,
                    "operation_signature": signature,
                    "schedule_node_id": node_id,
                    "event_name": event_name,
                    "perf_sim_pipe": event_pipe,
                    "pipe_mismatch_reason": mismatch_reason,
                }
            )
        sources[trace_path] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        schedule_sources[schedule_path] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()

    return {
        "schema_version": 1,
        "calibration_scope": "complete_schedule_signature",
        "manifest": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "sources": [{"path": str(source), "sha256": digest} for source, digest in sorted(sources.items())],
        "schedule_sources": [
            {"path": str(source), "sha256": digest} for source, digest in sorted(schedule_sources.items())
        ],
        "instructions": dict(sorted(instructions.items())),
    }


def _load_model(
    path: Path | None,
    *,
    pto_isa_root: Path | None = None,
    pto_isa_revision: str | None = None,
    unsupported_policy: str = "error",
    fallback_cycles: float = 1.0,
) -> DurationModel:
    if path is not None and pto_isa_root is not None:
        raise ValueError("choose either --model or --pto-isa-root, not both")
    if path is None:
        if pto_isa_root is None:
            raise ValueError("duration scoring requires --model or --pto-isa-root")
        revision = pto_isa_revision or _source_checkout_pto_isa_pin()
        provider = PtoIsaDurationProvider.from_checkout(
            pto_isa_root,
            expected_revision=revision,
            unsupported_policy=unsupported_policy,
            fallback_cycles=fallback_cycles,
        )
        return DurationModel(
            model_version="duration_v1",
            calibration_status="pto_isa_pinned",
            pto_isa_provider=provider,
            calibration_sources=[f"pto-isa:{provider.revision}:{provider_snapshot_sha256(provider)}"],
        )
    return DurationModel.from_json(json.loads(path.read_text()))


def _source_checkout_pto_isa_pin() -> str:
    pin_path = Path(__file__).resolve().parents[3] / "runtime" / "pto_isa.pin"
    try:
        revision = pin_path.read_text().strip()
    except OSError as error:
        raise ValueError(
            "cannot infer the PTO-ISA pin outside a PyPTO source checkout; pass --pto-isa-revision"
        ) from error
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{pin_path}: expected one full 40-hex PTO-ISA revision")
    return revision


def _write_json(path: Path | None, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(rendered)
    else:
        path.write_text(rendered)


def freeze_predictions(
    predictions: Mapping[str, Any],
    *,
    cohort: str,
    source_paths: Sequence[Path],
    frozen_before_device_timing: bool = True,
    freeze_context: str | None = None,
) -> dict[str, Any]:
    """Wrap predictions in a content-addressed holdout record."""
    context = freeze_context or (
        "prospective_holdout" if frozen_before_device_timing else "retrospective_before_timing_join"
    )
    canonical = json.dumps(predictions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "cohort": cohort,
        "frozen_before_device_timing": frozen_before_device_timing,
        "freeze_context": context,
        "prediction_sha256": hashlib.sha256(canonical).hexdigest(),
        "schedule_sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in source_paths
        ],
        "predictions": predictions,
    }


def _resolve_schedule_record(path: Path, function: str | None) -> tuple[dict[str, Any], str]:
    records = load_schedule_graphs(path)
    if function is None:
        if len(records) != 1:
            names = ", ".join(sorted(records))
            raise ValueError(f"{path}: function is required; schedule contains: {names}")
        function = next(iter(records))
    if function not in records:
        raise ValueError(f"{path}: function '{function}' is not present")
    return records[function], function


def classify_static_schedule(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify schedules supported by the structured control-flow model."""
    nodes = [node for node in record.get("nodes", []) if isinstance(node, Mapping)]
    operation_ids = [node.get("id") for node in nodes if node.get("kind") == "operation"]
    branch_ids = [node.get("id") for node in nodes if node.get("kind") == "branch"]
    loop_ids = [node.get("id") for node in nodes if node.get("kind") == "loop"]
    dynamic_loop_ids = [
        node.get("id")
        for node in nodes
        if node.get("kind") == "loop"
        and node.get("loop_kind") == "LOOP_BEGIN"
        and (type(node.get("static_trip_count")) is not int or node["static_trip_count"] < 0)
    ]
    if dynamic_loop_ids:
        status = "DYNAMIC_LOOP_EXCLUDED"
    elif branch_ids:
        status = "STATIC_BRANCH_SCHEDULE"
    else:
        status = "STATIC_SCHEDULE"
    return {
        "policy": "structured_branch_static_loop_v2",
        "eligible": status != "DYNAMIC_LOOP_EXCLUDED",
        "status": status,
        "operation_count": len(operation_ids),
        "branch_node_count": len(branch_ids),
        "loop_node_count": len(loop_ids),
        "dynamic_loop_node_count": len(dynamic_loop_ids),
        "branch_node_ids": branch_ids,
        "loop_node_ids": loop_ids,
        "dynamic_loop_node_ids": dynamic_loop_ids,
    }


def qualify_schedule_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Return timing-blind static-schedule eligibility for schedule graphs."""
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        for function, record in sorted(load_schedule_graphs(path).items()):
            rows.append(
                {
                    "source": str(path.resolve()),
                    "source_sha256": source_sha256,
                    "function": function,
                    **classify_static_schedule(record),
                }
            )
    return {
        "schema_version": 1,
        "selection_policy": "structured_branch_static_loop_v2",
        "timing_blind": True,
        "schedule_count": len(rows),
        "eligible_count": sum(row["eligible"] for row in rows),
        "schedules": rows,
    }


def _effect_sign(value: float) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _summarize_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [row for row in rows if row["observed_delta_us"] is not None]
    directional = [
        row for row in observed if row["predicted_direction"] != 0 and row["observed_direction"] != 0
    ]
    predicted_deltas = [float(row["predicted_relative_delta"]) for row in observed]
    observed_deltas = [float(row["observed_relative_delta"]) for row in observed]
    duration_coverages = [
        float(row[key])
        for row in rows
        for key in ("baseline_exact_duration_coverage", "candidate_exact_duration_coverage")
    ]
    direction_correct = sum(row["direction_correct"] is True for row in directional)
    final_edge_directional = [
        row
        for row in observed
        if _effect_sign(float(row["final_edge_independent_sum_cycles"])) != 0
        and row["observed_direction"] != 0
    ]
    final_edge_direction_correct = sum(
        _effect_sign(float(row["final_edge_independent_sum_cycles"])) == row["observed_direction"]
        for row in final_edge_directional
    )
    interactions = [abs(float(row["final_edge_interaction_cycles"])) for row in rows]
    return {
        "comparison_count": len(rows),
        "observed_comparison_count": len(observed),
        "directional_comparison_count": len(directional),
        "direction_correct_count": direction_correct,
        "direction_accuracy": direction_correct / len(directional) if directional else None,
        "minimum_exact_duration_coverage": min(duration_coverages),
        "mean_exact_duration_coverage": statistics.mean(duration_coverages),
        "spearman_relative_delta": _pearson(
            _average_ranks(predicted_deltas), _average_ranks(observed_deltas)
        ),
        "final_edge_independent_directional_comparison_count": len(final_edge_directional),
        "final_edge_independent_direction_correct_count": final_edge_direction_correct,
        "final_edge_independent_direction_accuracy": (
            final_edge_direction_correct / len(final_edge_directional) if final_edge_directional else None
        ),
        "nonzero_final_edge_interaction_count": sum(value > 1e-9 for value in interactions),
        "maximum_absolute_final_edge_interaction_cycles": max(interactions, default=0.0),
    }


def _operation_stream_signature(record: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            node.get("id"),
            node.get("kind"),
            node.get("op_name"),
            node.get("pipe"),
            tuple(node.get("loop_stack", [])),
            tuple(node.get("branch_stack", [])),
        )
        for node in record.get("nodes", [])
        if isinstance(node, Mapping)
    ]


def _sync_edge_signature(edge: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        edge.get("source"),
        edge.get("target"),
        edge.get("src_pipe"),
        edge.get("dst_pipe"),
        bool(edge.get("loop_carried", False)),
    )


def _sync_edge_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_counts = Counter(
        _sync_edge_signature(edge) for edge in baseline.get("sync_edges", []) if isinstance(edge, Mapping)
    )
    candidate_counts = Counter(
        _sync_edge_signature(edge) for edge in candidate.get("sync_edges", []) if isinstance(edge, Mapping)
    )

    def encode(delta: Counter[tuple[Any, ...]]) -> list[dict[str, Any]]:
        return [
            {
                "source": signature[0],
                "target": signature[1],
                "src_pipe": signature[2],
                "dst_pipe": signature[3],
                "loop_carried": signature[4],
                "count": count,
            }
            for signature, count in sorted(delta.items(), key=lambda item: repr(item[0]))
            if count > 0
        ]

    return encode(candidate_counts - baseline_counts), encode(baseline_counts - candidate_counts)


def _sync_edge_objects_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return concrete added and removed sync-edge occurrences.

    Group identifiers are deliberately excluded from edge identity: they are
    allocation details of one InsertSync run, while source, target, pipes, and
    loop distance define the dependency seen by the latency graph.  Concrete
    edge objects are retained so the caller can replay the delta without
    inventing provenance.
    """

    def difference(minuend: Mapping[str, Any], subtrahend: Mapping[str, Any]) -> list[dict[str, Any]]:
        groups = {
            group.get("id"): group
            for group in minuend.get("sync_groups", [])
            if isinstance(group, Mapping) and isinstance(group.get("id"), int)
        }
        remaining = Counter(
            _sync_edge_signature(edge)
            for edge in subtrahend.get("sync_edges", [])
            if isinstance(edge, Mapping)
        )
        result: list[dict[str, Any]] = []
        edges = [edge for edge in minuend.get("sync_edges", []) if isinstance(edge, Mapping)]
        for edge in sorted(edges, key=lambda item: repr((_sync_edge_signature(item), item.get("group")))):
            signature = _sync_edge_signature(edge)
            if remaining[signature] > 0:
                remaining[signature] -= 1
            else:
                copied = dict(edge)
                if copied.get("loop_carried") and not isinstance(copied.get("loop_end"), int):
                    group = groups.get(copied.get("group"))
                    operations = group.get("operations", []) if isinstance(group, Mapping) else []
                    loop_ends = {
                        operation.get("loop_end")
                        for operation in operations
                        if isinstance(operation, Mapping) and isinstance(operation.get("loop_end"), int)
                    }
                    if len(loop_ends) == 1:
                        copied["loop_end"] = loop_ends.pop()
                result.append(copied)
        return result

    return difference(candidate, baseline), difference(baseline, candidate)


def _dependency_only_schedule(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the final dependency graph while disabling provenance recovery.

    The normalized sync edges already contain every dependency exported by
    InsertSync.  Clearing ``sync_groups`` prevents a leave-one-edge-out
    ablation from being silently undone when barrier provenance is propagated
    again.  This representation is used only for dependency-makespan scoring;
    instruction-count and queue-drain models continue to use the unmodified
    schedule records.
    """
    normalized = _propagate_barrier_dependency_provenance(record)
    groups: list[dict[str, Any]] = []
    for group in normalized.get("sync_groups", []):
        if not isinstance(group, Mapping):
            continue
        operations = []
        for operation in group.get("operations", []):
            if not isinstance(operation, Mapping):
                continue
            copied = dict(operation)
            if str(copied.get("type", "")).startswith("pipe_barrier"):
                copied["useless"] = True
            operations.append(copied)
        groups.append({**dict(group), "operations": operations})
    normalized["sync_groups"] = groups
    limitations = dict(normalized.get("export_limitations", {}))
    limitations["barrier_dependency_nodes_missing"] = 0
    normalized["export_limitations"] = limitations
    return normalized


def _score_dependency_only_schedule(
    record: Mapping[str, Any], model: DurationModel
) -> tuple[dict[str, Any], float]:
    score = score_schedule(record, model)
    if not score["latency_graph_complete"]:
        raise ValueError(
            f"post-InsertSync dependency graph is incomplete: {score['latency_graph_limitations']}"
        )
    return score, float(score["loop_aware_makespan_cycles"])


def _schedule_with_sync_edges(
    record: Mapping[str, Any], edges: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    changed = copy.deepcopy(dict(record))
    changed["sync_edges"] = [dict(edge) for edge in edges]
    return changed


def _encoded_sync_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": edge.get("source"),
        "target": edge.get("target"),
        "src_pipe": edge.get("src_pipe"),
        "dst_pipe": edge.get("dst_pipe"),
        "loop_carried": bool(edge.get("loop_carried", False)),
        "root_buffers": list(edge.get("root_buffers", [])),
    }


def score_post_insert_sync_marginal(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    model: DurationModel,
) -> dict[str, Any]:
    """Score an exact complete-placement oracle and a sparse edge proxy.

    Both inputs must be actual post-InsertSync schedules for complete legal
    placements with the same operation stream.  The exact signed marginal is
    ``L(candidate) - L(baseline)``.  The sparse proxy independently removes
    or adds each changed final synchronization edge in the baseline context.
    Its residual against the exact marginal is reported as interaction rather
    than hidden inside an allegedly additive penalty.
    """
    normalized_baseline = _propagate_barrier_dependency_provenance(baseline)
    normalized_candidate = _propagate_barrier_dependency_provenance(candidate)
    if _operation_stream_signature(normalized_baseline) != _operation_stream_signature(normalized_candidate):
        raise ValueError("post-InsertSync marginal requires identical operation streams")

    baseline_score = score_schedule(normalized_baseline, model)
    candidate_score = score_schedule(normalized_candidate, model)
    for role, score in (("baseline", baseline_score), ("candidate", candidate_score)):
        if not score["latency_graph_complete"]:
            raise ValueError(
                f"incomplete {role} latency graph for post-InsertSync schedule: "
                f"{score['latency_graph_limitations']}"
            )

    dependency_baseline = _dependency_only_schedule(normalized_baseline)
    dependency_candidate = _dependency_only_schedule(normalized_candidate)
    _, baseline_cycles = _score_dependency_only_schedule(dependency_baseline, model)
    _, candidate_cycles = _score_dependency_only_schedule(dependency_candidate, model)
    added, removed = _sync_edge_objects_delta(dependency_baseline, dependency_candidate)

    baseline_edges = [
        dict(edge) for edge in dependency_baseline.get("sync_edges", []) if isinstance(edge, Mapping)
    ]
    reconstructed_edges = list(baseline_edges)
    for edge in removed:
        signature = _sync_edge_signature(edge)
        index = next(
            (
                edge_index
                for edge_index, existing in enumerate(reconstructed_edges)
                if _sync_edge_signature(existing) == signature
            ),
            None,
        )
        if index is None:
            raise ValueError(f"cannot remove absent final sync edge {signature}")
        reconstructed_edges.pop(index)
    reconstructed_edges.extend(added)
    reconstructed = _schedule_with_sync_edges(dependency_baseline, reconstructed_edges)
    _, reconstructed_cycles = _score_dependency_only_schedule(reconstructed, model)
    if not math.isclose(reconstructed_cycles, candidate_cycles, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "final sync-edge delta does not reconstruct the candidate dependency makespan: "
            f"{reconstructed_cycles} != {candidate_cycles}"
        )

    independent_rows: list[dict[str, Any]] = []
    for effect, edge in (("remove", edge) for edge in removed):
        signature = _sync_edge_signature(edge)
        changed_edges = list(baseline_edges)
        index = next(
            edge_index
            for edge_index, existing in enumerate(changed_edges)
            if _sync_edge_signature(existing) == signature
        )
        changed_edges.pop(index)
        _, changed_cycles = _score_dependency_only_schedule(
            _schedule_with_sync_edges(dependency_baseline, changed_edges), model
        )
        independent_rows.append(
            {
                "effect": effect,
                "edge": _encoded_sync_edge(edge),
                "signed_marginal_cycles": changed_cycles - baseline_cycles,
            }
        )
    for effect, edge in (("add", edge) for edge in added):
        changed_edges = [*baseline_edges, edge]
        _, changed_cycles = _score_dependency_only_schedule(
            _schedule_with_sync_edges(dependency_baseline, changed_edges), model
        )
        independent_rows.append(
            {
                "effect": effect,
                "edge": _encoded_sync_edge(edge),
                "signed_marginal_cycles": changed_cycles - baseline_cycles,
            }
        )

    sequential_rows: list[dict[str, Any]] = []
    current_edges = list(baseline_edges)
    current_cycles = baseline_cycles
    for effect, edge in [
        *(("remove", edge) for edge in removed),
        *(("add", edge) for edge in added),
    ]:
        if effect == "remove":
            signature = _sync_edge_signature(edge)
            index = next(
                edge_index
                for edge_index, existing in enumerate(current_edges)
                if _sync_edge_signature(existing) == signature
            )
            current_edges.pop(index)
        else:
            current_edges.append(edge)
        _, next_cycles = _score_dependency_only_schedule(
            _schedule_with_sync_edges(dependency_baseline, current_edges), model
        )
        sequential_rows.append(
            {
                "effect": effect,
                "edge": _encoded_sync_edge(edge),
                "signed_marginal_cycles": next_cycles - current_cycles,
            }
        )
        current_cycles = next_cycles
    if not math.isclose(current_cycles, candidate_cycles, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("sequential final-edge attribution does not telescope to the exact marginal")

    exact_marginal = candidate_cycles - baseline_cycles
    independent_sum = sum(float(row["signed_marginal_cycles"]) for row in independent_rows)
    return {
        "schema_version": 1,
        "model_version": "post_insert_sync_signed_marginal_v1",
        "oracle_contract": "complete_legal_placements_with_actual_post_insert_sync_schedules",
        "baseline_cycles": baseline_cycles,
        "candidate_cycles": candidate_cycles,
        "exact_signed_marginal_cycles": exact_marginal,
        "exact_relative_delta": exact_marginal / baseline_cycles if baseline_cycles else 0.0,
        "added_final_sync_edges": [_encoded_sync_edge(edge) for edge in added],
        "removed_final_sync_edges": [_encoded_sync_edge(edge) for edge in removed],
        "candidate_reconstructed_from_final_edge_delta": True,
        "final_edge_independent_signed_marginals": independent_rows,
        "final_edge_independent_sum_cycles": independent_sum,
        "final_edge_interaction_cycles": exact_marginal - independent_sum,
        "final_edge_sequential_signed_marginals": sequential_rows,
        "duration_coverage": {
            "baseline_exact": baseline_score["exact_duration_coverage"],
            "candidate_exact": candidate_score["exact_duration_coverage"],
        },
    }


def _queue_event_signed_marginal(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare matching branch extremes of two queue/event scores."""

    baseline_mixed = bool(baseline.get("mixed_iteration_branch_profile_available"))
    candidate_mixed = bool(candidate.get("mixed_iteration_branch_profile_available"))
    if baseline_mixed != candidate_mixed:
        raise ValueError("planner arms disagree on mixed-iteration branch-profile availability")

    def by_choice(score: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        scenarios = score.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("queue/event score has no branch scenarios")
        result: dict[str, Mapping[str, Any]] = {}
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                raise ValueError("queue/event branch scenario must be an object")
            choices = scenario.get("branch_choices")
            if not isinstance(choices, Mapping):
                raise ValueError("queue/event branch scenario has no choices")
            key = json.dumps(choices, sort_keys=True, separators=(",", ":"))
            if key in result:
                raise ValueError(f"duplicate queue/event branch scenario {key}")
            result[key] = scenario
        return result

    baseline_scenarios = by_choice(baseline)
    candidate_scenarios = by_choice(candidate)
    if baseline_scenarios.keys() != candidate_scenarios.keys():
        raise ValueError("planner arms expose different structured branch scenarios")
    complete = bool(baseline.get("pipeline_break_model_complete")) and bool(
        candidate.get("pipeline_break_model_complete")
    )
    deltas = [
        {
            "branch_choices": dict(baseline_scenarios[key]["branch_choices"]),
            "baseline_cycles": float(baseline_scenarios[key]["full_makespan_cycles"]),
            "candidate_cycles": float(candidate_scenarios[key]["full_makespan_cycles"]),
            "delta_cycles": float(candidate_scenarios[key]["full_makespan_cycles"])
            - float(baseline_scenarios[key]["full_makespan_cycles"]),
        }
        for key in sorted(baseline_scenarios)
    ]
    minimum = min(row["delta_cycles"] for row in deltas)
    maximum = max(row["delta_cycles"] for row in deltas)
    if not complete:
        conclusion = "PIPELINE_BREAK_CALIBRATION_INCOMPLETE"
    elif maximum < 0:
        conclusion = "BENEFICIAL_ALL_BRANCH_EXTREMES"
    elif minimum > 0:
        conclusion = "HARMFUL_ALL_BRANCH_EXTREMES"
    elif minimum == maximum == 0:
        conclusion = "TIE_ALL_BRANCH_EXTREMES"
    else:
        conclusion = "BRANCH_PATH_DEPENDENT"
    return {
        "model_version": "static_unrolled_pipe_event_branch_extremes_v2",
        "pipeline_break_model_complete": complete,
        "mixed_iteration_branch_profile_available": baseline_mixed,
        "minimum_delta_cycles": minimum,
        "maximum_delta_cycles": maximum,
        "direction_conclusion": conclusion,
        "scenarios": deltas,
    }


def _queue_drain_restart_signed_marginal(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare only changed barrier sites, allowing a signed marginal cost."""

    if baseline.get("status", "COMPLETE") != "COMPLETE" or candidate.get("status", "COMPLETE") != "COMPLETE":
        limitations = sorted(
            {
                str(limitation)
                for score in (baseline, candidate)
                for limitation in score.get("limitations", [])
            }
        )
        return {
            "model_version": "queue_drain_successor_restart_signed_marginal_v1",
            "complete": False,
            "limitations": limitations or ["queue_drain_restart_model_incomplete"],
            "minimum_delta_cycles": None,
            "maximum_delta_cycles": None,
            "direction_conclusion": "QUEUE_DRAIN_RESTART_MODEL_INCOMPLETE",
            "scenarios": [],
        }

    def scenarios_by_choice(score: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for scenario in score.get("scenarios", []):
            if not isinstance(scenario, Mapping) or not isinstance(scenario.get("branch_choices"), Mapping):
                raise ValueError("queue-drain score has an invalid branch scenario")
            key = json.dumps(scenario["branch_choices"], sort_keys=True, separators=(",", ":"))
            if key in result:
                raise ValueError(f"duplicate queue-drain branch scenario {key}")
            result[key] = scenario
        if not result:
            raise ValueError("queue-drain score has no branch scenarios")
        return result

    def active_sites(scenario: Mapping[str, Any]) -> dict[tuple[int, int, str], Mapping[str, Any]]:
        result: dict[tuple[int, int, str], Mapping[str, Any]] = {}
        for site in scenario.get("active_sites", []):
            if not isinstance(site, Mapping):
                raise ValueError("queue-drain active site must be an object")
            source = site.get("source")
            target = site.get("target")
            pipe = site.get("pipe")
            if not isinstance(source, int) or not isinstance(target, int) or not isinstance(pipe, str):
                raise ValueError("queue-drain active site has invalid provenance")
            key = (source, target, pipe)
            result[key] = site
        return result

    baseline_scenarios = scenarios_by_choice(baseline)
    candidate_scenarios = scenarios_by_choice(candidate)
    if baseline_scenarios.keys() != candidate_scenarios.keys():
        raise ValueError("planner arms expose different queue-drain branch scenarios")
    rows: list[dict[str, Any]] = []
    for key in sorted(baseline_scenarios):
        before = active_sites(baseline_scenarios[key])
        after = active_sites(candidate_scenarios[key])
        added = [after[site] for site in sorted(after.keys() - before.keys())]
        removed = [before[site] for site in sorted(before.keys() - after.keys())]
        for common in before.keys() & after.keys():
            if before[common].get("expanded_cycles") != after[common].get("expanded_cycles"):
                raise ValueError(f"common barrier site {common} has different pipeline costs across arms")
        complete = all(site.get("complete") is True for site in [*added, *removed])
        delta = (
            sum(float(site["expanded_cycles"]) for site in added)
            - sum(float(site["expanded_cycles"]) for site in removed)
            if complete
            else None
        )
        rows.append(
            {
                "branch_choices": dict(baseline_scenarios[key]["branch_choices"]),
                "complete": complete,
                "delta_cycles": delta,
                "added_sites": added,
                "removed_sites": removed,
            }
        )
    complete = all(row["complete"] for row in rows)
    deltas = [float(row["delta_cycles"]) for row in rows if row["delta_cycles"] is not None]
    minimum = min(deltas) if complete else None
    maximum = max(deltas) if complete else None
    if not complete:
        conclusion = "PIPELINE_COMPONENTS_INCOMPLETE"
    elif maximum is not None and maximum < 0:
        conclusion = "BENEFICIAL_ALL_BRANCH_EXTREMES"
    elif minimum is not None and minimum > 0:
        conclusion = "HARMFUL_ALL_BRANCH_EXTREMES"
    elif minimum == maximum == 0:
        conclusion = "TIE_ALL_BRANCH_EXTREMES"
    else:
        conclusion = "BRANCH_PATH_DEPENDENT"
    return {
        "model_version": "queue_drain_successor_restart_signed_marginal_v1",
        "complete": complete,
        "minimum_delta_cycles": minimum,
        "maximum_delta_cycles": maximum,
        "direction_conclusion": conclusion,
        "scenarios": rows,
    }


def evaluate_arm_manifest(manifest_path: str | Path, model: DurationModel) -> dict[str, Any]:  # noqa: PLR0912 - validation is deliberately fail-closed
    """Score paired planner arms and compare them with optional observations.

    Schedule paths are resolved relative to the manifest.  A held-out manifest
    can omit both observed latencies; supplying only one is rejected.  The
    relative delta convention is candidate/baseline - 1, so negative means the
    candidate is predicted or observed to be faster.
    """
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{path}: expected comparison manifest schema_version=1")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError(f"{path}: comparisons must be a non-empty array")

    rows: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    for index, item in enumerate(comparisons):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: comparison {index} must be an object")
        label = item.get("case")
        split = item.get("split")
        baseline_arm = item.get("baseline_arm")
        candidate_arm = item.get("candidate_arm")
        if not all(isinstance(value, str) and value for value in (label, split, baseline_arm, candidate_arm)):
            raise ValueError(f"{path}: comparison {index} has missing case, split, or arm")

        arm_records: dict[str, dict[str, Any]] = {}
        arm_results: dict[str, dict[str, Any]] = {}
        arm_sources: dict[str, dict[str, str]] = {}
        for role in ("baseline", "candidate"):
            raw_schedule = item.get(f"{role}_schedule")
            function = item.get(f"{role}_function")
            if not isinstance(raw_schedule, str) or not raw_schedule:
                raise ValueError(f"{path}: comparison {label} is missing {role}_schedule")
            if function is not None and not isinstance(function, str):
                raise ValueError(f"{path}: comparison {label} has invalid {role}_function")
            schedule = Path(raw_schedule)
            if not schedule.is_absolute():
                schedule = path.parent / schedule
            schedule = schedule.resolve()
            record, resolved_function = _resolve_schedule_record(schedule, function)
            arm_records[role] = record
            arm_sources[role] = {
                "path": str(schedule),
                "function": resolved_function,
                "sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
            }
            source_paths.add(schedule)

        if _operation_stream_signature(arm_records["baseline"]) != _operation_stream_signature(
            arm_records["candidate"]
        ):
            raise ValueError(
                f"{path}: comparison {label} has different operation streams across planner arms"
            )
        marginal = score_post_insert_sync_marginal(arm_records["baseline"], arm_records["candidate"], model)
        for role in ("baseline", "candidate"):
            arm_results[role] = score_schedule(arm_records[role], model)
        added_sync_edges, removed_sync_edges = _sync_edge_delta(
            arm_records["baseline"], arm_records["candidate"]
        )

        baseline_cycles = float(marginal["baseline_cycles"])
        candidate_cycles = float(marginal["candidate_cycles"])
        queue_event_marginal = _queue_event_signed_marginal(
            arm_results["baseline"]["queue_event_model"],
            arm_results["candidate"]["queue_event_model"],
        )
        queue_drain_restart_marginal = _queue_drain_restart_signed_marginal(
            arm_results["baseline"]["queue_drain_restart_model"],
            arm_results["candidate"]["queue_drain_restart_model"],
        )
        if baseline_cycles <= 0:
            raise ValueError(f"{path}: comparison {label} has a non-positive baseline prediction")
        predicted_delta = candidate_cycles - baseline_cycles
        predicted_relative_delta = candidate_cycles / baseline_cycles - 1.0

        observed_values = [item.get("baseline_latency_us"), item.get("candidate_latency_us")]
        present = [value is not None for value in observed_values]
        if any(present) and not all(present):
            raise ValueError(f"{path}: comparison {label} must provide both observed latencies or neither")
        observed_baseline = _as_number(observed_values[0])
        observed_candidate = _as_number(observed_values[1])
        if all(present) and (
            observed_baseline is None
            or observed_candidate is None
            or observed_baseline <= 0
            or observed_candidate <= 0
        ):
            raise ValueError(f"{path}: comparison {label} has invalid observed latencies")

        observed_delta = None
        observed_relative_delta = None
        observed_direction = None
        direction_correct = None
        if observed_baseline is not None and observed_candidate is not None:
            observed_delta = observed_candidate - observed_baseline
            observed_relative_delta = observed_candidate / observed_baseline - 1.0
            observed_direction = _effect_sign(observed_delta)
            direction_correct = _effect_sign(predicted_delta) == observed_direction

        rows.append(
            {
                "case": label,
                "split": split,
                "baseline_arm": baseline_arm,
                "candidate_arm": candidate_arm,
                "baseline_source": arm_sources["baseline"],
                "candidate_source": arm_sources["candidate"],
                "operation_stream_comparable": True,
                "added_sync_edges": added_sync_edges,
                "removed_sync_edges": removed_sync_edges,
                "baseline_cycles": baseline_cycles,
                "candidate_cycles": candidate_cycles,
                "baseline_exact_duration_coverage": arm_results["baseline"]["exact_duration_coverage"],
                "candidate_exact_duration_coverage": arm_results["candidate"]["exact_duration_coverage"],
                "baseline_duration_source_counts": arm_results["baseline"]["duration_source_counts"],
                "candidate_duration_source_counts": arm_results["candidate"]["duration_source_counts"],
                "baseline_full_critical_path": arm_results["baseline"]["full_critical_path"],
                "candidate_full_critical_path": arm_results["candidate"]["full_critical_path"],
                "baseline_sync_edge_exposure": arm_results["baseline"]["sync_edge_exposure"],
                "candidate_sync_edge_exposure": arm_results["candidate"]["sync_edge_exposure"],
                "baseline_loop_sync_models": arm_results["baseline"]["loop_sync_models"],
                "candidate_loop_sync_models": arm_results["candidate"]["loop_sync_models"],
                "baseline_pre_codegen_sync_record_summary": arm_results["baseline"][
                    "pre_codegen_sync_record_summary"
                ],
                "candidate_pre_codegen_sync_record_summary": arm_results["candidate"][
                    "pre_codegen_sync_record_summary"
                ],
                "queue_event_signed_marginal": queue_event_marginal,
                "queue_drain_restart_signed_marginal": queue_drain_restart_marginal,
                "post_insert_sync_signed_marginal": marginal,
                "final_edge_independent_sum_cycles": marginal["final_edge_independent_sum_cycles"],
                "final_edge_interaction_cycles": marginal["final_edge_interaction_cycles"],
                "predicted_delta_cycles": predicted_delta,
                "signed_marginal_sync_cost_cycles": predicted_delta,
                "predicted_relative_delta": predicted_relative_delta,
                "predicted_direction": _effect_sign(predicted_delta),
                "baseline_latency_us": observed_baseline,
                "candidate_latency_us": observed_candidate,
                "observed_delta_us": observed_delta,
                "observed_relative_delta": observed_relative_delta,
                "observed_direction": observed_direction,
                "direction_correct": direction_correct,
            }
        )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    canonical_predictions = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "prediction_metric": "loop_aware_makespan_cycles",
        "marginal_cost_metric": (
            "L(InsertSync(candidate placement)) - L(InsertSync(baseline placement)); "
            "negative values are permitted"
        ),
        "sparse_approximation_metric": (
            "sum of independently scored added/removed final InsertSync dependencies in the "
            "baseline placement context"
        ),
        "relative_delta_convention": "candidate/baseline - 1; negative is candidate faster",
        "frozen_before_device_timing": bool(payload.get("frozen_before_device_timing", False)),
        "manifest": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "schedule_sources": [
            {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            for source in sorted(source_paths)
        ],
        "prediction_sha256": hashlib.sha256(canonical_predictions).hexdigest(),
        "summary": _summarize_comparisons(rows),
        "summary_by_split": {
            split: _summarize_comparisons(split_rows) for split, split_rows in sorted(by_split.items())
        },
        "comparisons": rows,
    }


_PERF_SIM_EVENT_RE = re.compile(
    r"^(?P<op>[A-Z][A-Z0-9_]*)\((?P<rows>\d+)x(?P<cols>\d+),(?P<dtype>fp16|fp32)\)"
)


def validate_pto_isa_formulas_against_perf_sim(
    paths: Sequence[str | Path], provider: PtoIsaDurationProvider
) -> dict[str, Any]:
    """Compare supported PTO-ISA formulas with effective Perf-Sim event cycles.

    Perf-Sim prefers cycles recorded by the richer CCE mock when they are
    available, and otherwise uses the same lightweight formula API represented
    by ``provider``.  This is therefore a deliberately strict cross-model
    comparison, not a round-trip test of the CSV parser.  The result quantifies
    when a lightweight analytical duration is an adequate stand-in for the
    effective simulator duration.  Complete schedule makespans remain validated
    separately through ``evaluate_arm_manifest`` against device observations.
    """
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a Perf-Sim Chrome trace array")
        source_rows.append(
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        for event_index, event in enumerate(payload):
            if not isinstance(event, Mapping) or event.get("ph") != "X":
                continue
            name = event.get("name")
            observed = _as_number(event.get("dur"))
            if not isinstance(name, str) or observed is None or observed < 0:
                continue
            match = _PERF_SIM_EVENT_RE.match(name)
            if match is None:
                continue
            estimate = provider.estimate_formula(
                match.group("op"),
                match.group("dtype"),
                int(match.group("rows")),
                int(match.group("cols")),
            )
            if estimate is None:
                continue
            error = estimate.cycles - observed
            rows.append(
                {
                    "source": str(path.resolve()),
                    "event_index": event_index,
                    "event_name": name,
                    "op": match.group("op"),
                    "dtype": match.group("dtype"),
                    "rows": int(match.group("rows")),
                    "cols": int(match.group("cols")),
                    "predicted_cycles": estimate.cycles,
                    "perf_sim_cycles": observed,
                    "error_cycles": error,
                    "absolute_error_cycles": abs(error),
                    "absolute_percentage_error": abs(error) / observed if observed > 0 else None,
                }
            )
    if not rows:
        raise ValueError("Perf-Sim traces contain no events supported by the pinned formula table")
    percentage_errors = [
        float(row["absolute_percentage_error"])
        for row in rows
        if row["absolute_percentage_error"] is not None
    ]
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operation[str(row["op"])].append(row)

    def summarize(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        sample_percentage_errors = [
            float(row["absolute_percentage_error"])
            for row in samples
            if row["absolute_percentage_error"] is not None
        ]
        return {
            "event_count": len(samples),
            "mean_predicted_cycles": statistics.mean(float(row["predicted_cycles"]) for row in samples),
            "mean_perf_sim_cycles": statistics.mean(float(row["perf_sim_cycles"]) for row in samples),
            "mean_error_cycles": statistics.mean(float(row["error_cycles"]) for row in samples),
            "mean_absolute_error_cycles": statistics.mean(
                float(row["absolute_error_cycles"]) for row in samples
            ),
            "median_absolute_error_cycles": statistics.median(
                float(row["absolute_error_cycles"]) for row in samples
            ),
            "mean_absolute_percentage_error": statistics.mean(sample_percentage_errors),
            "median_absolute_percentage_error": statistics.median(sample_percentage_errors),
        }

    return {
        "schema_version": 1,
        "validation_scope": "lightweight_formula_vs_perf_sim_effective_events",
        "comparison_semantics": (
            "Perf-Sim uses CCE-recorded cycles when available and otherwise the lightweight model"
        ),
        "pto_isa_revision": provider.revision,
        "provider_snapshot_sha256": provider_snapshot_sha256(provider),
        "sources": source_rows,
        "event_count": len(rows),
        "operation_count": len({row["op"] for row in rows}),
        "mean_absolute_error_cycles": statistics.mean(float(row["absolute_error_cycles"]) for row in rows),
        "median_absolute_error_cycles": statistics.median(
            float(row["absolute_error_cycles"]) for row in rows
        ),
        "mean_absolute_percentage_error": statistics.mean(percentage_errors),
        "median_absolute_percentage_error": statistics.median(percentage_errors),
        "by_operation": {
            operation: summarize(samples) for operation, samples in sorted(by_operation.items())
        },
        "events": rows,
    }


def _add_duration_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="portable duration-model JSON")
    source.add_argument("--pto-isa-root", type=Path, help="exact pinned PTO-ISA checkout")
    parser.add_argument(
        "--pto-isa-revision",
        help="full expected revision; defaults to runtime/pto_isa.pin in a source checkout",
    )
    parser.add_argument(
        "--unsupported-policy",
        choices=("error", "fallback"),
        default="error",
        help="fail closed by default; fallback is explicit and reported per node",
    )
    parser.add_argument("--fallback-cycles", type=float, default=1.0)


def _model_from_args(args: argparse.Namespace) -> DurationModel:
    return _load_model(
        args.model,
        pto_isa_root=args.pto_isa_root,
        pto_isa_revision=args.pto_isa_revision,
        unsupported_policy=args.unsupported_policy,
        fallback_cycles=args.fallback_cycles,
    )


def _load_nonmaterialized_access_evidence(
    path: Path | None,
    *,
    schedule_path: Path,
    problem_path: Path,
) -> frozenset[int]:
    if path is None:
        return frozenset()
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{path}: expected non-materialized access evidence schema_version=1")
    expected = {
        "schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "problem_sha256": hashlib.sha256(problem_path.read_bytes()).hexdigest(),
    }
    for key, digest in expected.items():
        if payload.get(key) != digest:
            raise ValueError(f"{path}: {key} does not match the scored input")
    orders = payload.get("nonmaterialized_access_orders")
    if not isinstance(orders, list) or not all(isinstance(order, int) and order >= 0 for order in orders):
        raise ValueError(f"{path}: nonmaterialized_access_orders must be non-negative integers")
    if len(set(orders)) != len(orders):
        raise ValueError(f"{path}: nonmaterialized_access_orders contains duplicates")
    return frozenset(orders)


def _hashed_input(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _add_runtime_branch_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-branch-profile",
        type=Path,
        help="exact digest-bound per-occurrence branch profile",
    )
    parser.add_argument(
        "--runtime-parallel-branch-profile",
        type=Path,
        help="exact digest-bound branch scenarios across parallel dispatch instances",
    )
    parser.add_argument(
        "--runtime-input-manifest",
        type=Path,
        help="captured input manifest bound by --runtime-branch-profile",
    )
    parser.add_argument(
        "--runtime-trip-metadata",
        type=Path,
        help="captured loop-trip metadata bound by --runtime-branch-profile",
    )


def _apply_runtime_profile_from_args(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    *,
    schedule_path: Path,
    problem_path: Path,
) -> dict[str, Any]:
    profile_path = getattr(args, "runtime_branch_profile", None)
    parallel_profile_path = getattr(args, "runtime_parallel_branch_profile", None)
    input_manifest = getattr(args, "runtime_input_manifest", None)
    trip_metadata = getattr(args, "runtime_trip_metadata", None)
    if profile_path is None and parallel_profile_path is None:
        if input_manifest is not None or trip_metadata is not None:
            raise ValueError("runtime input metadata requires a runtime branch profile")
        return dict(record)
    if input_manifest is None or trip_metadata is None:
        raise ValueError(
            "runtime branch specialization requires a branch profile, "
            "--runtime-input-manifest, and --runtime-trip-metadata together"
        )
    digests = {
        "schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "problem_sha256": hashlib.sha256(problem_path.read_bytes()).hexdigest(),
        "input_set_sha256": hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
        "trip_metadata_sha256": hashlib.sha256(trip_metadata.read_bytes()).hexdigest(),
    }
    enriched = dict(record)
    if profile_path is not None:
        payload = json.loads(profile_path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{profile_path}: runtime branch profile must be an object")
        enriched = apply_runtime_branch_profile(enriched, payload, **digests)
    if parallel_profile_path is not None:
        payload = json.loads(parallel_profile_path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{parallel_profile_path}: runtime parallel branch profile must be an object")
        enriched = apply_runtime_parallel_branch_profile(enriched, payload, **digests)
    return enriched


def _duration_model_provenance(model: DurationModel) -> dict[str, Any]:
    snapshot = model.to_json()
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    provider = model.pto_isa_provider
    return {
        "semantic_sha256": hashlib.sha256(canonical).hexdigest(),
        "model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "calibration_sources": list(model.calibration_sources),
        "pto_isa_provider": (
            {
                "revision": provider.revision,
                "provider_snapshot_sha256": provider_snapshot_sha256(provider),
                "unsupported_policy": provider.unsupported_policy,
                "fallback_cycles": provider.fallback_cycles,
                "source_sha256": dict(sorted(provider.source_sha256.items())),
            }
            if provider is not None
            else None
        ),
    }


def _run_direct_model_command(args: argparse.Namespace) -> bool:
    """Run small CLI actions kept outside the main dispatch."""
    if args.command == "snapshot-duration":
        _write_json(args.output, _model_from_args(args).to_json())
        return True
    if args.command == "qualify":
        _write_json(args.output, qualify_schedule_files(args.schedules))
        return True
    if args.command == "audit-conformance":
        record, function = _resolve_schedule_record(args.schedule, args.function)
        nonmaterialized_access_orders = _load_nonmaterialized_access_evidence(
            args.nonmaterialized_access_evidence,
            schedule_path=args.schedule,
            problem_path=args.problem,
        )
        result = audit_placement_graph_conformance(
            record,
            args.problem,
            args.solution,
            known_nonmaterialized_access_orders=nonmaterialized_access_orders,
        )
        result["input"] = {
            "schedule": _hashed_input(args.schedule),
            "problem": _hashed_input(args.problem),
            "solution": _hashed_input(args.solution),
            "function": function,
            "nonmaterialized_access_evidence": (
                _hashed_input(args.nonmaterialized_access_evidence)
                if args.nonmaterialized_access_evidence is not None
                else None
            ),
        }
        _write_json(args.output, result)
        return True
    if args.command == "rescore-realized":
        record, _ = _resolve_schedule_record(args.schedule, args.function)
        candidate_scores = json.loads(args.candidate_score.read_text())
        if not isinstance(candidate_scores, Mapping):
            raise ValueError("candidate score document must be an object")
        realized_placement = candidate_scores.get("realized_placement")
        if not isinstance(realized_placement, Mapping):
            raise ValueError("candidate score document has no realized_placement object")
        result = score_complete_placement_dag(
            record,
            _model_from_args(args),
            candidate_scores,
            realized_placement,
        )
        result["input"] = {
            "schedule": str(args.schedule),
            "candidate_score": str(args.candidate_score),
        }
        _write_json(args.output, result)
        return True
    if args.command == "score-realized-grid":
        record, function = _resolve_schedule_record(args.schedule, args.function)
        record = _apply_runtime_profile_from_args(
            args,
            record,
            schedule_path=args.schedule,
            problem_path=args.problem,
        )
        raw_weights = [item.strip() for item in args.sync_latency_grid.split(",")]
        try:
            weights = sorted({float(item) for item in raw_weights if item})
        except ValueError as error:
            raise ValueError("sync latency grid must contain comma-separated numbers") from error
        if not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("sync latency grid must contain finite positive values")
        nonmaterialized_access_orders = _load_nonmaterialized_access_evidence(
            args.nonmaterialized_access_evidence,
            schedule_path=args.schedule,
            problem_path=args.problem,
        )
        base_model = _model_from_args(args)
        if (
            base_model.pto_isa_provider is not None
            and base_model.pto_isa_provider.unsupported_policy != "error"
        ):
            raise ValueError("score-realized-grid requires fail-closed PTO-ISA durations")
        _, duration_provenance, dynamic_loop_nodes = estimate_node_durations(record, base_model)
        fallback_nodes = sorted(
            node_id for node_id, provenance in duration_provenance.items() if provenance["fallback"]
        )
        if fallback_nodes:
            raise ValueError(
                "score-realized-grid requires exact or pinned non-fallback durations; "
                f"fallback operation nodes: {fallback_nodes}"
            )
        duration_coverage = {
            "operation_node_count": len(duration_provenance),
            "non_fallback_node_count": len(duration_provenance) - len(fallback_nodes),
            "fallback_node_count": len(fallback_nodes),
            "fallback_node_ids": fallback_nodes,
            "duration_sources": dict(
                sorted(Counter(str(item["source"]) for item in duration_provenance.values()).items())
            ),
            "dynamic_loop_node_ids": dynamic_loop_nodes,
        }
        duration_model = _duration_model_provenance(base_model)
        candidates = load_candidate_records(args.problem)
        promoted_penalties = load_promoted_reuse_penalties(args.problem)
        promoted_reasons = load_promoted_reuse_penalty_reasons(args.problem)
        results = []
        for weight in weights:
            model = replace(base_model, sync_latency_cycles=weight)
            candidate_scores = score_reuse_candidates(
                record,
                candidates,
                model,
                promoted_penalties=promoted_penalties,
                known_nonmaterialized_access_orders=nonmaterialized_access_orders,
                promoted_penalty_reasons=promoted_reasons,
            )
            realized = score_realized_reuse(
                args.problem,
                args.solution,
                candidate_scores,
                schedule_record=record,
                model=model,
            )
            results.append(
                {
                    "sync_latency_cycles": weight,
                    "unit_realized_cost": realized["unit_realized_cost"],
                    "canonical_physical_reuse_group_count": realized["canonical_physical_reuse_group_count"],
                    "unique_induced_sync_edge_count": realized["unique_induced_sync_edge_count"],
                    "realized_pair_count": realized["realized_pair_count"],
                    "realized_pair_count_by_penalty_reason": realized[
                        "realized_pair_count_by_penalty_reason"
                    ],
                    "executable_realized_pair_count_by_penalty_reason": realized[
                        "executable_realized_pair_count_by_penalty_reason"
                    ],
                    "synchronization_predictor_coverage_complete": realized[
                        "synchronization_predictor_coverage_complete"
                    ],
                    "score": realized["complete_placement_dag"],
                }
            )
        _write_json(
            args.output,
            {
                "schema_version": 1,
                "model_version": "complete_placement_dag_global_sync_weight_grid_v1",
                "input": {
                    "schedule": _hashed_input(args.schedule),
                    "problem": _hashed_input(args.problem),
                    "solution": _hashed_input(args.solution),
                    "function": function,
                    "nonmaterialized_access_evidence": (
                        _hashed_input(args.nonmaterialized_access_evidence)
                        if args.nonmaterialized_access_evidence is not None
                        else None
                    ),
                    "runtime_branch_profile": (
                        _hashed_input(args.runtime_branch_profile)
                        if args.runtime_branch_profile is not None
                        else None
                    ),
                    "runtime_parallel_branch_profile": (
                        _hashed_input(args.runtime_parallel_branch_profile)
                        if args.runtime_parallel_branch_profile is not None
                        else None
                    ),
                    "runtime_input_manifest": (
                        _hashed_input(args.runtime_input_manifest)
                        if args.runtime_input_manifest is not None
                        else None
                    ),
                    "runtime_trip_metadata": (
                        _hashed_input(args.runtime_trip_metadata)
                        if args.runtime_trip_metadata is not None
                        else None
                    ),
                    "duration_model_source": (
                        _hashed_input(args.model)
                        if args.model is not None
                        else {
                            "pto_isa_root": str(args.pto_isa_root.resolve()),
                            "expected_revision": args.pto_isa_revision,
                        }
                    ),
                },
                "duration_model": duration_model,
                "duration_coverage": duration_coverage,
                "duration_policy": "fail_closed_no_fallback",
                "results": results,
            },
        )
        return True
    if args.command == "aggregate-dispatch-grid":
        _write_json(args.output, aggregate_static_dispatch_grid(args.manifest))
        return True
    return False


def _dispatch_grid_score_makespans(score: Mapping[str, Any]) -> tuple[float, float, str, bool]:
    """Resolve one function score to concrete dispatch makespans.

    A captured parallel-branch profile is useful for retrospective analysis,
    but it is not information an address planner may consult.  The returned
    boolean therefore distinguishes a purely static score from an
    input-profile-dependent one.
    """

    if score.get("status") != "COMPLETE":
        raise ValueError(
            f"static dispatch aggregation requires COMPLETE per-function scores; got {score.get('status')!r}"
        )
    runtime_score = score.get("runtime_parallel_dispatch_score")
    if runtime_score is not None:
        if not isinstance(runtime_score, Mapping):
            raise ValueError("runtime_parallel_dispatch_score must be an object")
        base = runtime_score.get("base_makespan_cycles")
        placement = runtime_score.get("placement_makespan_cycles")
        source = "runtime_parallel_branch_profile"
        planner_eligible = False
    else:
        base = score.get("base_makespan_cycles")
        placement = score.get("placement_makespan_cycles")
        source = "static_worst_case_branch_envelope"
        planner_eligible = True
    if not isinstance(base, (int, float)) or not isinstance(placement, (int, float)):
        raise ValueError("complete per-function score has no concrete base/placement makespan")
    base_cycles = float(base)
    placement_cycles = float(placement)
    if not math.isfinite(base_cycles) or not math.isfinite(placement_cycles):
        raise ValueError("per-function makespans must be finite")
    if base_cycles <= 0 or placement_cycles <= 0:
        raise ValueError("per-function makespans must be positive")
    return base_cycles, placement_cycles, source, planner_eligible


def _dispatch_dag_makespan(
    task_durations: Mapping[str, float], edges: Sequence[tuple[str, str]]
) -> tuple[float, list[str]]:
    """Compute a node-weighted task-DAG makespan and one critical path."""

    successors: dict[str, list[str]] = {task: [] for task in task_durations}
    predecessors: dict[str, list[str]] = {task: [] for task in task_durations}
    indegree = {task: 0 for task in task_durations}
    for source, target in edges:
        successors[source].append(target)
        predecessors[target].append(source)
        indegree[target] += 1
    ready = deque(sorted(task for task, degree in indegree.items() if degree == 0))
    finish: dict[str, float] = {}
    critical_predecessor: dict[str, str | None] = {}
    visited = 0
    while ready:
        task = ready.popleft()
        visited += 1
        parent = max(predecessors[task], key=finish.__getitem__) if predecessors[task] else None
        critical_predecessor[task] = parent
        finish[task] = task_durations[task] + (finish[parent] if parent is not None else 0.0)
        for successor in sorted(successors[task]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(task_durations):
        raise ValueError("static dispatch graph contains a cycle")
    last = max(finish, key=finish.__getitem__)
    path: list[str] = []
    cursor: str | None = last
    while cursor is not None:
        path.append(cursor)
        cursor = critical_predecessor[cursor]
    path.reverse()
    return finish[last], path


def _static_dispatch_duration_contract(path: Path, document: Mapping[str, Any]) -> dict[str, str]:
    duration_model = document.get("duration_model")
    semantic_sha256 = duration_model.get("semantic_sha256") if isinstance(duration_model, Mapping) else None
    if not isinstance(semantic_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", semantic_sha256) is None:
        raise ValueError(f"{path}: duration model has no valid semantic_sha256")
    duration_policy = document.get("duration_policy")
    if duration_policy != "fail_closed_no_fallback":
        raise ValueError(f"{path}: duration policy must be fail_closed_no_fallback")
    duration_coverage = document.get("duration_coverage")
    if not isinstance(duration_coverage, Mapping):
        raise ValueError(f"{path}: duration coverage must be an object")
    if duration_coverage.get("fallback_node_count") != 0 or duration_coverage.get("fallback_node_ids") != []:
        raise ValueError(f"{path}: static dispatch aggregation forbids fallback duration nodes")
    return {"semantic_sha256": semantic_sha256, "duration_policy": duration_policy}


def _static_dispatch_weights(path: Path, results: Sequence[Any]) -> list[float]:
    weights: list[float] = []
    for index, row in enumerate(results):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: result {index} must be an object")
        weight = _as_number(row.get("sync_latency_cycles"))
        if weight is None or weight <= 0:
            raise ValueError(f"{path}: result {index} weight must be finite and positive")
        score = row.get("score")
        if not isinstance(score, Mapping):
            raise ValueError(f"{path}: result {index} has no score object")
        inner_weight = _as_number(score.get("synchronization_latency_cycles"))
        if inner_weight != weight:
            raise ValueError(
                f"{path}: result {index} inner synchronization weight {inner_weight!r} "
                f"does not match outer weight {weight}"
            )
        weights.append(weight)
    if len(weights) != len(set(weights)) or weights != sorted(weights):
        raise ValueError(f"{path}: placement grid weights must be unique and increasing")
    return weights


def _load_static_dispatch_grids(
    manifest_path: Path, raw_functions: Mapping[str, Any]
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]],
    list[float],
    dict[str, str],
]:
    grids: dict[str, Mapping[str, Any]] = {}
    grid_inputs: dict[str, dict[str, Any]] = {}
    expected_weights: list[float] | None = None
    expected_duration_contract: dict[str, str] | None = None
    for function, relative_path in sorted(raw_functions.items()):
        if not isinstance(function, str) or not function or not isinstance(relative_path, str):
            raise ValueError("static dispatch functions must map non-empty names to paths")
        path = (manifest_path.parent / relative_path).resolve()
        document = json.loads(path.read_text())
        if not isinstance(document, Mapping):
            raise ValueError(f"{path}: placement grid must be an object")
        if (
            document.get("schema_version") != 1
            or document.get("model_version") != "complete_placement_dag_global_sync_weight_grid_v1"
        ):
            raise ValueError(f"{path}: incompatible placement-grid schema")
        input_record = document.get("input")
        if not isinstance(input_record, Mapping) or input_record.get("function") != function:
            raise ValueError(f"{path}: grid function does not match manifest key {function!r}")
        duration_contract = _static_dispatch_duration_contract(path, document)
        if expected_duration_contract is None:
            expected_duration_contract = duration_contract
        elif duration_contract != expected_duration_contract:
            raise ValueError(f"{path}: duration-model contract differs from its siblings")
        results = document.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"{path}: placement grid has no results")
        weights = _static_dispatch_weights(path, results)
        if expected_weights is None:
            expected_weights = weights
        elif weights != expected_weights:
            raise ValueError(f"{path}: synchronization-weight grid differs from its siblings")
        grids[function] = document
        grid_inputs[function] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert expected_weights is not None
    assert expected_duration_contract is not None
    return grids, grid_inputs, expected_weights, expected_duration_contract


def _parse_static_dispatch_tasks(raw_tasks: Sequence[Any], functions: set[str]) -> dict[str, tuple[str, ...]]:
    tasks: dict[str, tuple[str, ...]] = {}
    function_owners: dict[str, str] = {}
    for raw_task in raw_tasks:
        if not isinstance(raw_task, Mapping):
            raise ValueError("static dispatch task must be an object")
        task_id = raw_task.get("id")
        task_functions = raw_task.get("functions")
        aggregation = raw_task.get("aggregation", "max")
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("static dispatch task ids must be unique non-empty strings")
        if aggregation != "max":
            raise ValueError(f"task {task_id!r}: only co-scheduled max aggregation is supported")
        if (
            not isinstance(task_functions, list)
            or not task_functions
            or not all(isinstance(function, str) and function for function in task_functions)
            or len(task_functions) != len(set(task_functions))
        ):
            raise ValueError(f"task {task_id!r}: functions must be a unique non-empty string array")
        unknown = sorted(set(task_functions) - functions)
        if unknown:
            raise ValueError(f"task {task_id!r}: unknown functions: {', '.join(unknown)}")
        for function in task_functions:
            owner = function_owners.setdefault(function, task_id)
            if owner != task_id:
                raise ValueError(f"function {function!r} appears in multiple tasks")
        tasks[task_id] = tuple(task_functions)
    missing = sorted(functions - set(function_owners))
    if missing:
        raise ValueError(f"static dispatch graph omits functions: {', '.join(missing)}")
    return tasks


def _parse_static_dispatch_edges(
    raw_edges: Sequence[Any], tasks: Mapping[str, tuple[str, ...]]
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for raw_edge in raw_edges:
        if (
            not isinstance(raw_edge, list)
            or len(raw_edge) != 2
            or not all(isinstance(task, str) for task in raw_edge)
        ):
            raise ValueError("static dispatch edges must be [source, target] string pairs")
        source_task, target_task = raw_edge
        if source_task not in tasks or target_task not in tasks:
            raise ValueError(f"static dispatch edge names an unknown task: {raw_edge}")
        if source_task == target_task:
            raise ValueError(f"static dispatch task {source_task!r} cannot depend on itself")
        edges.append((source_task, target_task))
    if len(edges) != len(set(edges)):
        raise ValueError("static dispatch graph contains duplicate edges")
    return edges


def _aggregate_static_dispatch_weight(
    grids: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, tuple[str, ...]],
    edges: Sequence[tuple[str, str]],
    index: int,
    weight: float,
) -> dict[str, Any]:
    function_rows: dict[str, dict[str, Any]] = {}
    planner_eligible = True
    for function, grid in grids.items():
        raw_result = grid["results"][index]
        score = raw_result.get("score") if isinstance(raw_result, Mapping) else None
        if not isinstance(score, Mapping):
            raise ValueError(f"function {function!r} weight {weight}: missing score object")
        base, placement, selection_source, function_planner_eligible = _dispatch_grid_score_makespans(score)
        planner_eligible &= function_planner_eligible
        function_rows[function] = {
            "base_makespan_cycles": base,
            "placement_makespan_cycles": placement,
            "selection_source": selection_source,
            "planner_eligible_static": function_planner_eligible,
        }
    base_tasks = {
        task: max(function_rows[function]["base_makespan_cycles"] for function in functions)
        for task, functions in tasks.items()
    }
    placement_tasks = {
        task: max(function_rows[function]["placement_makespan_cycles"] for function in functions)
        for task, functions in tasks.items()
    }
    base_makespan, base_path = _dispatch_dag_makespan(base_tasks, edges)
    placement_makespan, placement_path = _dispatch_dag_makespan(placement_tasks, edges)
    extension = placement_makespan - base_makespan
    return {
        "sync_latency_cycles": weight,
        "status": "COMPLETE",
        "planner_eligible_static": planner_eligible,
        "analysis_only_runtime_profile_used": not planner_eligible,
        "base_makespan_cycles": base_makespan,
        "placement_makespan_cycles": placement_makespan,
        "critical_path_extension_cycles": extension,
        "relative_critical_path_extension": extension / base_makespan,
        "base_critical_task_path": base_path,
        "placement_critical_task_path": placement_path,
        "base_task_durations_cycles": base_tasks,
        "placement_task_durations_cycles": placement_tasks,
        "functions": function_rows,
    }


def aggregate_static_dispatch_grid(manifest_path: str | Path) -> dict[str, Any]:
    """Aggregate per-function placement grids over a static runtime task DAG.

    The manifest describes which functions are co-scheduled in one runtime
    task and the dependency edges between tasks.  Co-scheduled functions are
    concurrent and therefore aggregate by ``max``.  Dependent tasks aggregate
    through a node-weighted longest path.  Every per-function grid must use
    the same synchronization-weight grid; incomplete or parametric scores fail
    closed rather than being silently coerced into a concrete latency.
    """

    source = Path(manifest_path)
    manifest = json.loads(source.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("static dispatch manifest must be an object")
    if manifest.get("schema_version") != 1 or manifest.get("contract") != "static_dispatch_graph_v1":
        raise ValueError("static dispatch manifest must use static_dispatch_graph_v1 schema_version=1")
    raw_functions = manifest.get("functions")
    raw_tasks = manifest.get("tasks")
    raw_edges = manifest.get("edges", [])
    if not isinstance(raw_functions, Mapping) or not raw_functions:
        raise ValueError("static dispatch manifest requires a non-empty functions object")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("static dispatch manifest requires a non-empty tasks array")
    if not isinstance(raw_edges, list):
        raise ValueError("static dispatch manifest edges must be an array")

    grids, grid_inputs, weights, duration_contract = _load_static_dispatch_grids(source, raw_functions)
    tasks = _parse_static_dispatch_tasks(raw_tasks, set(grids))
    edges = _parse_static_dispatch_edges(raw_edges, tasks)
    aggregated_results = [
        _aggregate_static_dispatch_weight(grids, tasks, edges, index, weight)
        for index, weight in enumerate(weights)
    ]
    return {
        "schema_version": 1,
        "model_version": "static_dispatch_complete_placement_grid_v1",
        "aggregation_contract": {
            "co_scheduled_functions": "max",
            "dependent_tasks": "node_weighted_longest_path",
            "runtime_profiles": "analysis_only_not_planner_eligible",
        },
        "duration_contract": duration_contract,
        "input": {
            "manifest": _hashed_input(source),
            "function_grids": grid_inputs,
        },
        "tasks": [
            {"id": task, "functions": list(functions), "aggregation": "max"}
            for task, functions in tasks.items()
        ],
        "edges": [list(edge) for edge in edges],
        "results": aggregated_results,
    }


def _add_score_realized_grid_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "score-realized-grid",
        help="score one placement over a global synchronization-latency grid",
    )
    parser.add_argument("schedule", type=Path)
    parser.add_argument("problem", type=Path)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--function")
    parser.add_argument("--sync-latency-grid", required=True)
    parser.add_argument("--nonmaterialized-access-evidence", type=Path)
    _add_runtime_branch_profile_arguments(parser)
    _add_duration_arguments(parser)
    parser.add_argument("-o", "--output", type=Path)


def _add_aggregate_dispatch_grid_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "aggregate-dispatch-grid",
        help="aggregate per-function placement grids over a static runtime task DAG",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0912,PLR0915 - explicit CLI dispatch
    parser = argparse.ArgumentParser(prog="dsa_schedule_model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score schedule-graph JSONL files")
    score_parser.add_argument("schedules", nargs="+", type=Path)
    _add_duration_arguments(score_parser)
    score_parser.add_argument("--freeze-cohort")
    score_parser.add_argument("-o", "--output", type=Path)

    calibrate_parser = subparsers.add_parser("calibrate", help="calibrate from instr_metrics.json files")
    calibrate_parser.add_argument("metrics", nargs="+", type=Path)
    calibrate_parser.add_argument("--base-model", type=Path)
    calibrate_parser.add_argument("-o", "--output", type=Path, required=True)

    extract_calibration_parser = subparsers.add_parser(
        "extract-calibration",
        help="bind Perf-Sim events to exact schedule-node duration signatures",
    )
    extract_calibration_parser.add_argument("manifest", type=Path)
    extract_calibration_parser.add_argument("-o", "--output", type=Path, required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot-duration", help="write a portable pinned duration-model snapshot"
    )
    _add_duration_arguments(snapshot_parser)
    snapshot_parser.add_argument("-o", "--output", type=Path, required=True)

    import_parser = subparsers.add_parser(
        "import-debug", help="convert a legacy PTOAS level-3 debug log to schedule JSONL"
    )
    import_parser.add_argument("log", type=Path)
    import_parser.add_argument("--function", required=True)
    import_parser.add_argument(
        "--pto",
        type=Path,
        help="raw PTO carrying pypto.access.N locations; joined fail-closed by operation order",
    )
    import_parser.add_argument("-o", "--output", type=Path, required=True)

    enrich_parser = subparsers.add_parser(
        "enrich-native",
        help="join a native PTOAS schedule graph to exact raw-PTO operation semantics",
    )
    enrich_parser.add_argument("schedule", type=Path)
    enrich_parser.add_argument("--pto", type=Path, required=True)
    enrich_parser.add_argument("--function")
    enrich_parser.add_argument("-o", "--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="score paired planner arms and compare optional observed latencies"
    )
    evaluate_parser.add_argument("manifest", type=Path)
    _add_duration_arguments(evaluate_parser)
    evaluate_parser.add_argument("-o", "--output", type=Path)

    qualify_parser = subparsers.add_parser(
        "qualify", help="classify schedule graphs using timing-blind straight-line eligibility"
    )
    qualify_parser.add_argument("schedules", nargs="+", type=Path)
    qualify_parser.add_argument("-o", "--output", type=Path)

    audit_parser = subparsers.add_parser(
        "audit-conformance",
        help="compare realized pre-InsertSync reuse hazards with product synchronization edges",
    )
    audit_parser.add_argument("schedule", type=Path)
    audit_parser.add_argument("problem", type=Path)
    audit_parser.add_argument("solution", type=Path)
    audit_parser.add_argument("--function")
    audit_parser.add_argument(
        "--nonmaterialized-access-evidence",
        type=Path,
        help="digest-bound proof for candidate accesses removed before the lowered schedule",
    )
    audit_parser.add_argument("-o", "--output", type=Path)

    candidate_parser = subparsers.add_parser(
        "score-candidates", help="join raw DSA candidates to a schedule and derive critical-path weights"
    )
    candidate_parser.add_argument("schedule", type=Path)
    candidate_parser.add_argument("problem", type=Path)
    candidate_parser.add_argument("--function")
    _add_duration_arguments(candidate_parser)
    candidate_parser.add_argument("--solution", type=Path)
    candidate_parser.add_argument(
        "--nonmaterialized-access-evidence",
        type=Path,
        help="digest-bound proof for candidate accesses removed before the lowered schedule",
    )
    _add_runtime_branch_profile_arguments(candidate_parser)
    candidate_parser.add_argument("-o", "--output", type=Path)

    ptoas_edges_parser = subparsers.add_parser(
        "emit-ptoas-reuse-edges",
        help="translate one complete DSA placement into PTOAS reuse-edge input",
    )
    ptoas_edges_parser.add_argument("candidate_score", type=Path)
    ptoas_edges_parser.add_argument("problem", type=Path)
    ptoas_edges_parser.add_argument("solution", type=Path)
    ptoas_edges_parser.add_argument("ptoas_graph", type=Path)
    ptoas_edges_parser.add_argument("--function")
    ptoas_edges_parser.add_argument("-o", "--output", type=Path, required=True)

    rescore_parser = subparsers.add_parser(
        "rescore-realized",
        help="recompute the exact complete-placement DAG score from archived candidate evidence",
    )
    rescore_parser.add_argument("schedule", type=Path)
    rescore_parser.add_argument("candidate_score", type=Path)
    rescore_parser.add_argument("--function")
    _add_duration_arguments(rescore_parser)
    rescore_parser.add_argument("-o", "--output", type=Path)

    _add_score_realized_grid_parser(subparsers)
    _add_aggregate_dispatch_grid_parser(subparsers)

    perf_sim_parser = subparsers.add_parser(
        "validate-perf-sim", help="compare pinned formulas with Perf-Sim trace events"
    )
    perf_sim_parser.add_argument("traces", nargs="+", type=Path)
    _add_duration_arguments(perf_sim_parser)
    perf_sim_parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args(argv)
    try:
        if _run_direct_model_command(args):
            return 0
        if args.command == "calibrate":
            base = (
                DurationModel.from_json(json.loads(args.base_model.read_text())) if args.base_model else None
            )
            calibrated = calibrate_from_metrics(args.metrics, base)
            _write_json(args.output, calibrated.to_json())
            return 0
        if args.command == "extract-calibration":
            _write_json(args.output, extract_signature_calibration(args.manifest))
            return 0
        if args.command == "import-debug":
            record = import_insert_sync_debug(
                args.log.read_text(),
                function=args.function,
                pto_text=args.pto.read_text() if args.pto is not None else None,
            )
            args.output.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            return 0
        if args.command == "enrich-native":
            record, _ = _resolve_schedule_record(args.schedule, args.function)
            enriched = enrich_native_schedule_from_pto(
                record,
                args.pto.read_text(),
                pto_source=str(args.pto.resolve()),
            )
            args.output.write_text(json.dumps(enriched, sort_keys=True, allow_nan=False) + "\n")
            return 0
        if args.command == "evaluate":
            evaluated = evaluate_arm_manifest(args.manifest, _model_from_args(args))
            _write_json(args.output, evaluated)
            return 0
        if args.command == "score-candidates":
            record, _ = _resolve_schedule_record(args.schedule, args.function)
            record = _apply_runtime_profile_from_args(
                args,
                record,
                schedule_path=args.schedule,
                problem_path=args.problem,
            )
            nonmaterialized_access_orders = _load_nonmaterialized_access_evidence(
                args.nonmaterialized_access_evidence,
                schedule_path=args.schedule,
                problem_path=args.problem,
            )
            model = _model_from_args(args)
            result = score_reuse_candidates(
                record,
                load_candidate_records(args.problem),
                model,
                promoted_penalties=load_promoted_reuse_penalties(args.problem),
                known_nonmaterialized_access_orders=nonmaterialized_access_orders,
                promoted_penalty_reasons=load_promoted_reuse_penalty_reasons(args.problem),
            )
            if isinstance(record.get("runtime_branch_profile"), Mapping):
                result["runtime_branch_profile"] = dict(record["runtime_branch_profile"])
            if isinstance(record.get("runtime_parallel_branch_profile"), Mapping):
                result["runtime_parallel_branch_profile"] = dict(record["runtime_parallel_branch_profile"])
            if args.solution is not None:
                result["realized_placement"] = score_realized_reuse(
                    args.problem,
                    args.solution,
                    result,
                    schedule_record=record,
                    model=model,
                )
            _write_json(args.output, result)
            return 0
        if args.command == "emit-ptoas-reuse-edges":
            candidate_scores = json.loads(args.candidate_score.read_text())
            if not isinstance(candidate_scores, Mapping):
                raise ValueError("candidate score input must be a JSON object")
            _write_json(
                args.output,
                emit_ptoas_placement_reuse_edges(
                    candidate_scores,
                    args.problem,
                    args.solution,
                    args.ptoas_graph,
                    function=args.function,
                ),
            )
            return 0
        if args.command == "validate-perf-sim":
            model = _model_from_args(args)
            if model.pto_isa_provider is None:
                raise ValueError("Perf-Sim formula validation requires a model with PTO-ISA provenance")
            result = validate_pto_isa_formulas_against_perf_sim(args.traces, model.pto_isa_provider)
            _write_json(args.output, result)
            return 0

        model = _model_from_args(args)
        predictions: dict[str, Any] = {}
        for schedule_path in args.schedules:
            for function, record in load_schedule_graphs(schedule_path).items():
                key = f"{schedule_path}:{function}"
                predictions[key] = score_schedule(record, model)
        value: Any = predictions
        if args.freeze_cohort:
            value = freeze_predictions(predictions, cohort=args.freeze_cohort, source_paths=args.schedules)
        _write_json(args.output, value)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
