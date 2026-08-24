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
DAG scores.  Candidate scoring additionally evaluates distance-one edges with
a version-1 loop initiation-interval lower bound: the maximum of per-pipe work
and every supported recurrence cycle.  This remains a structural model, not a
cycle-accurate prediction.  Active Final-SyncIR records and hypothetical
candidate synchronization endpoints are reported separately: a redundant
precedence edge can have zero DAG extension while still creating synchronization
pressure.  These are explicitly pre-codegen quantities, not counts of emitted
instructions.  Scores also disclose every loop-carried, loop-marker, or omitted
barrier dependency that the collapsed operation-only DAG cannot represent.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
    r"^\s*(?:%[-A-Za-z0-9_.$]+(?:#\d+)?\s*=\s*)?scf\.for\s+%\S+\s*=\s*"
    r"(%[-A-Za-z0-9_.$]+|-?\d+)\s+to\s+"
    r"(%[-A-Za-z0-9_.$]+|-?\d+)\s+step\s+(%[-A-Za-z0-9_.$]+|-?\d+)\b"
)
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


@dataclass(frozen=True)
class PipeParameters:
    """Primitive version-0 duration parameters for one execution pipe."""

    startup_cycles: float
    bytes_per_cycle: float
    minimum_cycles: float


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
    pipe_parameters: dict[str, PipeParameters] = field(default_factory=_default_pipe_parameters)
    operation_cycles: dict[str, float] = field(default_factory=dict)
    operation_signature_cycles: dict[str, float] = field(default_factory=dict)
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
        raw_provider = value.get("pto_isa_provider")
        if raw_provider is not None and not isinstance(raw_provider, Mapping):
            raise ValueError("pto_isa_provider must be an object")
        return cls(
            schema_version=1,
            model_version=str(value.get("model_version", "duration_v1")),
            calibration_status=str(value.get("calibration_status", "unknown")),
            sync_latency_cycles=float(value.get("sync_latency_cycles", 0.0)),
            pipe_parameters=pipes,
            operation_cycles={str(key): float(cycles) for key, cycles in raw_ops.items()},
            operation_signature_cycles={str(key): float(cycles) for key, cycles in raw_signatures.items()},
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
    }
    return aliases.get(name, name)


def _operation_names_match(expected: str, actual: str, metadata: Mapping[str, Any]) -> bool:
    if expected == actual:
        return True
    if expected in {"pto.tpush", "pto.tpop"} and _join_operation_name(actual) == expected:
        return True
    if expected != "pto.tmatmul.acc" or actual != "pto.tmatmul":
        return False
    operand_types = metadata.get("operand_types", [])
    return isinstance(operand_types, list) and any(
        isinstance(item, str) and _PTO_ACC_TILE_RE.search(item) for item in operand_types
    )


def _attach_pto_access_provenance(nodes: list[dict[str, Any]], pto_text: str) -> None:
    """Join legacy SyncIR nodes to raw-PTO access locations by exact op order.

    PTOAS's legacy text trace omits MLIR locations.  Raw PTO emitted with
    ``PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1`` contains those locations, while the
    trace preserves the same executable operation order.  Structural PTO ops
    such as ``pto.alloc_tile`` are ignored because they do not appear among the
    trace's operation names.  Any missing location or sequence mismatch is a
    hard error; this bridge never guesses a coordinate.
    """
    operation_nodes = [node for node in nodes if node.get("kind") == "operation"]
    expected_names = [str(node["op_name"]) for node in operation_nodes]
    traced_names = {_join_operation_name(name) for name in expected_names}
    pto_operations: list[tuple[str, int, str, dict[str, Any]]] = []
    constants = {
        match.group(1): f"{match.group(2)} : {match.group(3)}"
        for line in pto_text.splitlines()
        if (match := _PTO_SCALAR_CONSTANT_RE.match(line))
    }

    for line_number, line in enumerate(pto_text.splitlines(), start=1):
        names = [match.group(1) for match in _PTO_OPERATION_RE.finditer(line)]
        names = [name for name in names if _join_operation_name(name) in traced_names]
        if not names:
            continue
        if len(names) != 1:
            raise ValueError(f"raw PTO line {line_number} contains multiple traced operations: {names}")
        locations = _ACCESS_LOCATION_RE.findall(line)
        if len(set(locations)) != 1:
            raise ValueError(
                f"raw PTO operation {names[0]} on line {line_number} has no unambiguous "
                "pypto.access.N location"
            )
        access_order = int(locations[0])
        location_line = line.strip()
        pto_operations.append(
            (names[0], access_order, location_line, _operation_type_metadata(location_line, constants))
        )

    actual_names = [name for name, _, _, _ in pto_operations]
    matches = [
        _operation_names_match(expected, actual, metadata)
        for expected, (actual, _, _, metadata) in zip(expected_names, pto_operations, strict=False)
    ]
    if len(actual_names) != len(expected_names) or not all(matches):
        mismatch = next(
            (index for index, matched in enumerate(matches) if not matched),
            min(len(expected_names), len(actual_names)),
        )
        expected = expected_names[mismatch] if mismatch < len(expected_names) else "<end>"
        actual = actual_names[mismatch] if mismatch < len(actual_names) else "<end>"
        raise ValueError(
            "raw PTO operation sequence does not match the final SyncIR trace at "
            f"operation {mismatch}: expected {expected}, found {actual}; "
            f"counts={len(expected_names)}->{len(actual_names)}"
        )

    for node, (raw_name, access_order, location_line, metadata) in zip(
        operation_nodes, pto_operations, strict=True
    ):
        node["operation"] = {
            **metadata,
            "pypto_access_order": access_order,
            "location": location_line,
            "raw_pto_op_name": raw_name,
        }


def _attach_pto_static_loop_bounds(nodes: list[dict[str, Any]], pto_text: str) -> int:
    """Join statically provable raw-PTO ``scf.for`` trip counts to SyncIR loops.

    The legacy PTOAS debug stream identifies loop structure but reports SyncIR
    node ranges rather than iteration bounds. Raw PTO is the product-faithful
    source for the original ``scf.for`` lower/upper/step operands. The loop
    order is preserved by PTOAS; a count mismatch is therefore an ambiguous
    bridge and fails closed.

    Returns:
        The number of loops whose bounds are genuinely dynamic or unsupported.
    """
    constants: dict[str, int] = {}
    raw_loops: list[tuple[int | None, str]] = []

    def resolve(operand: str) -> int | None:
        if operand.startswith("%"):
            return constants.get(operand)
        return int(operand)

    for line in pto_text.splitlines():
        if match := _PTO_CONSTANT_RE.match(line):
            constants[match.group(1)] = int(match.group(2))
            continue
        if not (match := _PTO_FOR_RE.match(line)):
            continue
        lower, upper, step = (resolve(match.group(index)) for index in range(1, 4))
        trip_count = None
        if lower is not None and upper is not None and step is not None and step > 0:
            trip_count = max(0, (upper - lower + step - 1) // step)
        raw_loops.append((trip_count, line.strip()))

    loop_begins = [
        node for node in nodes if node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_BEGIN"
    ]
    if len(raw_loops) != len(loop_begins):
        raise ValueError(
            "raw PTO loop sequence does not match the final SyncIR trace: "
            f"counts={len(raw_loops)}->{len(loop_begins)}"
        )

    trip_count_by_begin: dict[int, int | None] = {}
    for node, (trip_count, source_line) in zip(loop_begins, raw_loops, strict=True):
        node["static_trip_count"] = trip_count
        node["operation"] = {"raw_pto_loop": source_line}
        trip_count_by_begin[int(node["id"])] = trip_count
    for node in nodes:
        if node.get("kind") == "loop" and node.get("loop_kind") == "LOOP_END":
            node["static_trip_count"] = trip_count_by_begin.get(int(node["begin"]))

    return sum(trip_count is None for trip_count, _ in raw_loops)


def import_insert_sync_debug(  # noqa: PLR0912 - stateful line parser mirrors the four debug record kinds
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
                "branch_stack": [],
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
                "branch_stack": [],
                "operation": {},
            }
            nodes.append(current_node)
            if kind == "LOOP_BEGIN":
                loop_stack.append(node_id)
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
    if pto_text is not None:
        _attach_pto_access_provenance(nodes, pto_text)
        missing_static_loop_bounds = _attach_pto_static_loop_bounds(nodes, pto_text)

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
    # The legacy text dump does not print BranchInstanceElements. Without an
    # explicit limitation, both arms appear as one serial stream and can yield
    # a plausible but invalid latency DAG. The native schedule exporter remains
    # the authoritative path for branch-aware records.
    branch_nodes_missing = 0
    if pto_text is not None:
        branch_nodes_missing = sum(
            bool(_PTO_BRANCH_RE.search(line.split("//", maxsplit=1)[0])) for line in pto_text.splitlines()
        )
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
            "barrier_dependency_nodes_missing": omitted_barriers,
            "branch_nodes_missing": branch_nodes_missing,
            "access_provenance_missing": pto_text is None,
        },
        "nodes": nodes,
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
        signature_key = (
            _operation_signature_key(operation_duration_signature(node))
            if model.operation_signature_cycles
            else None
        )
        work_bytes = _work_bytes(node)
        if signature_key is not None and signature_key in model.operation_signature_cycles:
            base = model.operation_signature_cycles[signature_key]
            source = "simulator_complete_signature_median"
            detail = f"complete operation signature {signature_key}"
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


def _schedule_graph_durations(
    record: Mapping[str, Any], operation_durations: Mapping[int, float]
) -> dict[int, float]:
    """Add zero-duration structural loop markers to operation durations."""
    durations = dict(operation_durations)
    for node in record.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, int):
            continue
        if node_id in durations:
            continue
        if node.get("kind") == "loop":
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
        if isinstance(node, Mapping) and node.get("kind") == "operation" and isinstance(node.get("id"), int)
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
        pipe = nodes_by_id[node_id].get("pipe")
        if isinstance(pipe, str):
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
    source_marker = source in {loop_id, loop_end}
    target_marker = target in {loop_id, loop_end}
    if not (source_marker or target_marker or source_inside != target_inside):
        return None
    is_entry = (source == loop_id and target_inside) or (not source_inside and target == loop_id)
    is_entry |= not source_marker and not source_inside and target_inside
    is_exit = (source_inside and target == loop_end) or (source == loop_end and not target_inside)
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
            and (node.get("kind") == "loop" or node_id in operation_durations)
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
                    nodes_by_id[node_id].get("kind") == "loop" for node_id in loop_schedule_ids
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
    branch_ids = [
        node.get("id")
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and node.get("kind") == "branch"
    ]
    if branch_ids:
        raise ValueError(
            "duration_v0 does not model mutually exclusive control-flow branches; "
            f"branch nodes: {branch_ids[:8]}"
        )
    operation_durations, provenance, dynamic_loops = estimate_node_durations(record, model)
    if dynamic_loops:
        raise ValueError(
            f"duration_v0 requires statically bounded loops; dynamic loop nodes: {dynamic_loops[:8]}"
        )
    durations = _schedule_graph_durations(record, operation_durations)
    node_ids = set(durations)
    stream_edges, _ = _graph_edges(record, node_ids, include_sync=False)
    full_edges, edge_diagnostics = _graph_edges(record, node_ids, include_sync=True)
    full_edges = [
        (source, target, model.sync_latency_cycles if kind == "sync" else latency, kind, group)
        for source, target, latency, kind, group in full_edges
    ]
    loop_sync_models = _existing_loop_sync_models(
        record, operation_durations, full_edges, model.sync_latency_cycles
    )

    baseline, top, bottom, baseline_path = _longest_path(durations, stream_edges)
    full, _, _, full_path = _longest_path(durations, full_edges)

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
        "duration_model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "loop_policy": "aggregate_static_work_v0",
        "loop_sync_model_version": "loop_sync_ii_and_boundary_v1",
        "loop_sync_models": loop_sync_models,
        "dynamic_loop_ids": dynamic_loops,
        **edge_diagnostics,
        **_latency_graph_completeness(record, edge_diagnostics, loop_sync_models),
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


def _deduplicate_scored_candidate_edges(
    rows: Sequence[dict[str, Any]], execution_counts: Mapping[int, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse candidate records that join to the same schedule edge."""
    distance_zero_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    loop_rows: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "scored":
            distance_zero_rows[(row["source_node"], row["target_node"])].append(row)
        elif row.get("status") == "loop_carried_scored_v1":
            loop_rows[(row["loop_node"], row["source_node"], row["target_node"])].append(row)

    distance_zero_edges: list[dict[str, Any]] = []
    for (source, target), duplicates in sorted(distance_zero_rows.items()):
        weights = {float(row["weight_cycles"]) for row in duplicates}
        pipe_pairs = {(row.get("prior_pipe"), row.get("next_pipe")) for row in duplicates}
        if len(weights) != 1 or len(pipe_pairs) != 1:
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
            }
        )

    loop_edges: list[dict[str, Any]] = []
    for (loop_node, source, target), duplicates in sorted(loop_rows.items()):
        weights = {float(row["weight_cycles"]) for row in duplicates}
        recurrence_cycles = {row["candidate_recurrence_cycles"] for row in duplicates}
        pipe_pairs = {(row.get("prior_pipe"), row.get("next_pipe")) for row in duplicates}
        if len(weights) != 1 or len(recurrence_cycles) != 1 or len(pipe_pairs) != 1:
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


def load_promoted_reuse_penalties(path: str | Path) -> dict[tuple[int, int], float]:
    """Load the pairwise reuse penalties that the DSA solver actually sees."""
    source = Path(path)
    document = json.loads(source.read_text())
    try:
        raw_penalties = document["problem"]["cost_model"]["reuse_penalties"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: missing problem.cost_model.reuse_penalties") from error
    if not isinstance(raw_penalties, list):
        raise ValueError(f"{source}: reuse_penalties must be an array")

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


def _score_penalty_pairs(
    rows: Sequence[Mapping[str, Any]],
    durations: Mapping[int, float],
    existing_edges: Sequence[tuple[int, int, float, str, int | None]],
    base_makespan: float,
    loop_counts: Mapping[int, int],
    sync_latency_cycles: float,
    promoted_penalties: Mapping[tuple[int, int], float] | None,
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
            if row.get("status") != "loop_carried_scored_v1":
                continue
            loop_id = row.get("loop_node")
            weight = _as_number(row.get("weight_cycles"))
            if not isinstance(loop_id, int) or weight is None:
                raise ValueError(f"invalid loop recurrence score for buffer pair {pair}")
            loop_weights[loop_id] = max(loop_weights.get(loop_id, 0.0), weight)
        loop_total_weight = sum(
            weight * max(loop_counts.get(loop_id, 1) - 1, 0) for loop_id, weight in loop_weights.items()
        )

        loop_schedule_edges = {
            (int(row["loop_node"]), int(row["source_node"]), int(row["target_node"]))
            for row in pair_rows
            if row.get("status") == "loop_carried_scored_v1"
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
        scored.append(
            {
                "first_buffer": pair[0],
                "second_buffer": pair[1],
                "promoted_to_dsa_penalty": promoted,
                "unit_cost": unit_cost,
                "candidate_record_count": len(pair_rows),
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
        missing = sorted(set(promoted_penalties) - set(by_pair))
        if missing:
            raise ValueError(f"promoted reuse penalties have no access-site candidate records: {missing[:8]}")
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


def score_realized_reuse(
    problem_path: str | Path,
    solution_path: str | Path,
    candidate_scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the promoted reuse pairs physically realized by one placement."""
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
        if not isinstance(pair_weight, Mapping) or not pair_weight.get("promoted_to_dsa_penalty"):
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
        rows.append(
            {
                **pair_weight,
                "reuse_realized": overlap,
                "overlap_bytes": max(0, min(first_end, second_end) - max(first_begin, second_begin))
                if overlap
                else 0,
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
    realized_without_induced_edge = [
        row
        for row in realized
        if not row.get("distance_zero_schedule_edges") and not row.get("loop_carried_schedule_edges")
    ]
    return {
        "schema_version": 2,
        "model_version": candidate_scores.get("model_version"),
        "problem": str(problem_source),
        "solution": str(solution_source),
        "promoted_pair_count": len(rows),
        "realized_pair_count": len(realized),
        "realized_pair_count_without_induced_sync_edge": len(realized_without_induced_edge),
        "unit_realized_cost": sum(float(row["unit_cost"]) for row in realized),
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
        "pairs": rows,
    }


def score_reuse_candidates(
    record: Mapping[str, Any],
    candidates: Sequence[ReuseCandidateRecord],
    model: DurationModel,
    promoted_penalties: Mapping[tuple[int, int], float] | None = None,
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

    Access-site provenance is mandatory. A missing/ambiguous route mapping or
    missing tagged schedule node is an error, never a zero-cost result.
    """
    branch_ids = [
        node.get("id")
        for node in record.get("nodes", [])
        if isinstance(node, Mapping) and node.get("kind") == "branch"
    ]
    if branch_ids:
        raise ValueError(
            "duration_v0 does not model mutually exclusive control-flow branches; "
            f"branch nodes: {branch_ids[:8]}"
        )
    operation_durations, provenance, dynamic_loops = estimate_node_durations(record, model)
    if dynamic_loops:
        raise ValueError(
            f"duration_v0 requires statically bounded loops; dynamic loop nodes: {dynamic_loops[:8]}"
        )
    durations = _schedule_graph_durations(record, operation_durations)
    node_ids = set(durations)
    existing_edges, edge_diagnostics = _graph_edges(record, node_ids, include_sync=True)
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
    rows: list[dict[str, Any]] = []
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
        if not prior_nodes or not next_nodes:
            raise ValueError(
                "candidate site did not join to the expected PTOAS pipe: "
                f"sites={candidate.prior_access_order}->{candidate.next_access_order}, "
                f"pipes={prior_route_pipe}->{next_route_pipe}, found={prior_nodes}->{next_nodes}"
            )
        source = prior_nodes[-1]
        target = next_nodes[0]
        if candidate.loop_carried:
            source_loop_stack = nodes_by_id[source].get("loop_stack", [])
            target_loops = set(nodes_by_id[target].get("loop_stack", []))
            common_loops = [loop for loop in source_loop_stack if loop in target_loops]
            if not common_loops:
                raise ValueError(
                    f"loop-carried candidate {index} sites do not share a PTOAS loop: edge {source}->{target}"
                )
            recurrence = _loop_recurrence_score(
                record,
                operation_durations,
                existing_edges,
                loop_id=common_loops[-1],
                source=source,
                target=target,
                candidate_latency=model.sync_latency_cycles,
            )
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
                    "source_macro_nodes": prior_nodes,
                    "target_macro_nodes": next_nodes,
                    "common_loop_nodes": common_loops,
                    "status": "loop_carried_scored_v1",
                    **recurrence,
                }
            )
            continue
        hypothetical = (source, target, model.sync_latency_cycles, "candidate_sync", index)
        try:
            with_candidate, _, _, path = _longest_path(durations, [*existing_edges, hypothetical])
        except ValueError as error:
            raise ValueError(
                f"candidate {index} creates a non-loop cycle at schedule edge {source}->{target}"
            ) from error
        weight = max(0.0, with_candidate - base_makespan)
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
                "source_macro_nodes": prior_nodes,
                "target_macro_nodes": next_nodes,
                "status": "scored",
                "weight_cycles": weight,
                "makespan_with_candidate_cycles": with_candidate,
                "critical_path_with_candidate": path,
            }
        )

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
        {(int(edge["source_node"]), int(edge["target_node"])): edge for edge in distance_zero_edges},
        {
            (int(edge["loop_node"]), int(edge["source_node"]), int(edge["target_node"])): edge
            for edge in loop_edge_groups
        },
    )
    baseline_loop_sync_models = _existing_loop_sync_models(
        record, operation_durations, existing_edges, model.sync_latency_cycles
    )

    return {
        "schema_version": 2,
        "model_version": "reuse_penalty_critical_path_v1",
        "sync_endpoint_estimator_version": "uncoalesced_source_plus_target_static_executions_v1",
        "function": record.get("function", "<unknown>"),
        "duration_model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "base_makespan_cycles": base_makespan,
        "base_critical_path": base_path,
        "dynamic_loop_ids": dynamic_loops,
        "loop_sync_model_version": "loop_sync_ii_and_boundary_v1",
        "baseline_loop_sync_models": baseline_loop_sync_models,
        **edge_diagnostics,
        **_latency_graph_completeness(record, edge_diagnostics, baseline_loop_sync_models),
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(
            row.get("status") in {"scored", "loop_carried_scored_v1"} for row in rows
        ),
        "scored_distance_zero_candidate_count": sum(row.get("status") == "scored" for row in rows),
        "scored_loop_carried_candidate_count": sum(
            row.get("status") == "loop_carried_scored_v1" for row in rows
        ),
        "unscored_loop_carried_candidate_count": 0,
        "candidates": rows,
        "consumer_groups": consumer_groups,
        "distance_zero_edges": distance_zero_edges,
        "loop_recurrence_edges": loop_edge_groups,
        "penalty_pair_weights": penalty_pair_weights,
        "candidate_weight_summary": weight_summary,
        "baseline_pre_codegen_sync_record_summary": _pre_codegen_sync_record_summary(record),
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
        pipe_parameters=calibrated_pipes,
        operation_cycles=dict(model.operation_cycles),
        operation_signature_cycles=signature_cycles,
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
    """Classify schedules whose execution count is statically determined."""
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
    if branch_ids:
        status = "BRANCH_EXCLUDED"
    elif dynamic_loop_ids:
        status = "DYNAMIC_LOOP_EXCLUDED"
    else:
        status = "STATIC_SCHEDULE"
    return {
        "policy": "static_loop_v1",
        "eligible": status == "STATIC_SCHEDULE",
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
        "selection_policy": "static_loop_v1",
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
        for role in ("baseline", "candidate"):
            arm_results[role] = score_schedule(arm_records[role], model)
        added_sync_edges, removed_sync_edges = _sync_edge_delta(
            arm_records["baseline"], arm_records["candidate"]
        )

        baseline_cycles = float(arm_results["baseline"]["full_makespan_cycles"])
        candidate_cycles = float(arm_results["candidate"]["full_makespan_cycles"])
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
                "predicted_delta_cycles": predicted_delta,
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
        "prediction_metric": "full_makespan_cycles",
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


def main(argv: Sequence[str] | None = None) -> int:
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

    candidate_parser = subparsers.add_parser(
        "score-candidates", help="join raw DSA candidates to a schedule and derive critical-path weights"
    )
    candidate_parser.add_argument("schedule", type=Path)
    candidate_parser.add_argument("problem", type=Path)
    candidate_parser.add_argument("--function")
    _add_duration_arguments(candidate_parser)
    candidate_parser.add_argument("--solution", type=Path)
    candidate_parser.add_argument("-o", "--output", type=Path)

    perf_sim_parser = subparsers.add_parser(
        "validate-perf-sim", help="compare pinned formulas with Perf-Sim trace events"
    )
    perf_sim_parser.add_argument("traces", nargs="+", type=Path)
    _add_duration_arguments(perf_sim_parser)
    perf_sim_parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot-duration":
            _write_json(args.output, _model_from_args(args).to_json())
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
        if args.command == "evaluate":
            evaluated = evaluate_arm_manifest(args.manifest, _model_from_args(args))
            _write_json(args.output, evaluated)
            return 0
        if args.command == "qualify":
            _write_json(args.output, qualify_schedule_files(args.schedules))
            return 0
        if args.command == "score-candidates":
            record, _ = _resolve_schedule_record(args.schedule, args.function)
            result = score_reuse_candidates(
                record,
                load_candidate_records(args.problem),
                _model_from_args(args),
                load_promoted_reuse_penalties(args.problem),
            )
            if args.solution is not None:
                result["realized_placement"] = score_realized_reuse(args.problem, args.solution, result)
            _write_json(args.output, result)
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
