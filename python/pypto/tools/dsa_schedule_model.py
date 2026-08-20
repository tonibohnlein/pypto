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
cycle-accurate prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    r"^\s*(PRE|POST)\s*:?[ ]*(\w+)\s+<(\S+)\s+->\s+(\S+)>\s+idx=(\d+)(?:\s+forEnd=(\d+))?"
)
_ACCESS_LOCATION_RE = re.compile(r"pypto\.access\.(\d+)")
_PTO_OPERATION_RE = re.compile(r"(?<![!\w.])(pto\.[A-Za-z0-9_]+)\b")

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


@dataclass(frozen=True)
class PipeParameters:
    """Primitive version-0 duration parameters for one execution pipe."""

    startup_cycles: float
    bytes_per_cycle: float
    minimum_cycles: float


def _default_pipe_parameters() -> dict[str, PipeParameters]:
    # These defaults are intentionally coarse initial values.  Every result
    # using them is labelled uncalibrated; simulator-derived calibration can
    # replace either exact operation medians or the pipe fallbacks.
    return {
        "PIPE_S": PipeParameters(4.0, math.inf, 4.0),
        "PIPE_V": PipeParameters(12.0, 64.0, 12.0),
        "PIPE_M": PipeParameters(32.0, 128.0, 32.0),
        "PIPE_MTE1": PipeParameters(20.0, 96.0, 20.0),
        "PIPE_MTE2": PipeParameters(24.0, 64.0, 24.0),
        "PIPE_MTE3": PipeParameters(24.0, 64.0, 24.0),
        "PIPE_FIX": PipeParameters(20.0, 64.0, 20.0),
        "PIPE_V2": PipeParameters(12.0, 64.0, 12.0),
        "PIPE_MTE4": PipeParameters(24.0, 64.0, 24.0),
        "PIPE_MTE5": PipeParameters(24.0, 64.0, 24.0),
    }


@dataclass
class DurationModel:
    """Duration inputs used to score a schedule graph."""

    schema_version: int = 1
    model_version: str = "duration_v0"
    calibration_status: str = "uncalibrated_defaults"
    sync_latency_cycles: float = 0.0
    pipe_parameters: dict[str, PipeParameters] = field(default_factory=_default_pipe_parameters)
    operation_cycles: dict[str, float] = field(default_factory=dict)
    calibration_sources: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> DurationModel:
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
        return cls(
            schema_version=1,
            model_version=str(value.get("model_version", "duration_v0")),
            calibration_status=str(value.get("calibration_status", "unknown")),
            sync_latency_cycles=float(value.get("sync_latency_cycles", 0.0)),
            pipe_parameters=pipes,
            operation_cycles={str(key): float(cycles) for key, cycles in raw_ops.items()},
            calibration_sources=[str(path) for path in value.get("calibration_sources", [])],
        )

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        value = asdict(self)
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
    traced_names = set(expected_names)
    pto_operations: list[tuple[str, int, str]] = []

    for line_number, line in enumerate(pto_text.splitlines(), start=1):
        names = [match.group(1) for match in _PTO_OPERATION_RE.finditer(line)]
        names = [name for name in names if name in traced_names]
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
        pto_operations.append((names[0], access_order, line.strip()))

    actual_names = [name for name, _, _ in pto_operations]
    if actual_names != expected_names:
        mismatch = next(
            (
                index
                for index, (expected, actual) in enumerate(zip(expected_names, actual_names, strict=False))
                if expected != actual
            ),
            min(len(expected_names), len(actual_names)),
        )
        expected = expected_names[mismatch] if mismatch < len(expected_names) else "<end>"
        actual = actual_names[mismatch] if mismatch < len(actual_names) else "<end>"
        raise ValueError(
            "raw PTO operation sequence does not match the final SyncIR trace at "
            f"operation {mismatch}: expected {expected}, found {actual}; "
            f"counts={len(expected_names)}->{len(actual_names)}"
        )

    for node, (_, access_order, location_line) in zip(operation_nodes, pto_operations, strict=True):
        node["operation"] = {
            "pypto_access_order": access_order,
            "location": location_line,
        }


def import_insert_sync_debug(  # noqa: PLR0912 - stateful line parser mirrors the four debug record kinds
    text: str, *, function: str, pto_text: str | None = None
) -> dict[str, Any]:
    """Convert PTOAS's legacy level-3 final SyncIR dump to schema v1.

    This is a compatibility bridge for archived or pre-exporter runs. The
    native C++ exporter remains authoritative: the text dump omits allocation
    sizes, operation attributes, barrier dependency nodes, and static loop
    bounds. The returned record names that limitation explicitly.
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
            sync_operations[sync_index].append(
                {
                    "placement": match.group(1),
                    "type": match.group(2),
                    "node": current_node["id"],
                    "src_pipe": match.group(3),
                    "dst_pipe": match.group(4),
                    "loop_end": int(match.group(6)) if match.group(6) else None,
                }
            )

    if not any(node["kind"] == "operation" for node in nodes):
        raise ValueError("final debug phase has no operation nodes")
    if pto_text is not None:
        _attach_pto_access_provenance(nodes, pto_text)

    sync_groups: list[dict[str, Any]] = []
    sync_edges: list[dict[str, Any]] = []
    omitted_barriers = 0
    for sync_index in sorted(sync_operations):
        operations = sync_operations[sync_index]
        representative = operations[0]
        loop_carried = any(operation["loop_end"] is not None for operation in operations)
        sources = sorted({operation["node"] for operation in operations if operation["type"] == "set_flag"})
        targets = sorted({operation["node"] for operation in operations if operation["type"] == "wait_flag"})
        if not sources and any(operation["type"] == "pipe_barrier" for operation in operations):
            omitted_barriers += 1
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

    return {
        "schema_version": 1,
        "function": function,
        "status": "analyzed",
        "node_count": len(nodes),
        "duration_model": "unestimated",
        "export_source": (
            "ptoas_debug_import_v0+pto_access_join_v1" if pto_text is not None else "ptoas_debug_import_v0"
        ),
        "export_limitations": {
            "allocation_sizes_missing": True,
            "static_loop_bounds_missing": True,
            "barrier_dependency_nodes_missing": omitted_barriers,
            "access_provenance_missing": pto_text is None,
        },
        "nodes": nodes,
        "stream_edges": stream_edges,
        "sync_groups": sync_groups,
        "sync_edges": sync_edges,
    }


def _canonical_operation(name: str) -> str:
    value = name.rsplit(".", maxsplit=1)[-1]
    return "".join(character for character in value.upper() if character.isalnum() or character == "_")


def _operation_key(pipe: str, name: str) -> str:
    return f"{pipe}:{_canonical_operation(name)}"


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
        work_bytes = _work_bytes(node)
        if key in model.operation_cycles:
            base = model.operation_cycles[key]
            source = "simulator_operation_median"
        else:
            parameters = model.pipe_parameters.get(pipe)
            if parameters is None:
                raise ValueError(f"no duration parameters for pipe '{pipe}'")
            transfer = (
                0.0 if math.isinf(parameters.bytes_per_cycle) else work_bytes / parameters.bytes_per_cycle
            )
            base = max(parameters.minimum_cycles, parameters.startup_cycles + transfer)
            source = "pipe_size_model"

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
        }
    return durations, provenance, dynamic_loops


def _graph_edges(
    record: Mapping[str, Any], node_ids: set[int], *, include_sync: bool
) -> tuple[list[tuple[int, int, float, str, int | None]], int]:
    edges: list[tuple[int, int, float, str, int | None]] = []
    seen: set[tuple[int, int, str, int | None]] = set()
    for edge in record.get("stream_edges", []):
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("source"), edge.get("target")
        key = (source, target, "stream", None)
        if (
            isinstance(source, int)
            and isinstance(target, int)
            and source in node_ids
            and target in node_ids
            and key not in seen
        ):
            edges.append((source, target, 0.0, "stream", None))
            seen.add(key)

    excluded_loop_carried = 0
    if include_sync:
        for edge in record.get("sync_edges", []):
            if not isinstance(edge, Mapping):
                continue
            if edge.get("loop_carried"):
                excluded_loop_carried += 1
                continue
            source, target, group = edge.get("source"), edge.get("target"), edge.get("group")
            group_id = group if isinstance(group, int) else None
            key = (source, target, "sync", group_id)
            if (
                isinstance(source, int)
                and isinstance(target, int)
                and source in node_ids
                and target in node_ids
                and source != target
                and key not in seen
            ):
                edges.append((source, target, 0.0, "sync", group_id))
                seen.add(key)
    return edges, excluded_loop_carried


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
    groups = {
        group.get("id"): group
        for group in record.get("sync_groups", [])
        if isinstance(group, Mapping) and isinstance(group.get("id"), int)
    }
    existing_recurrences: list[dict[str, Any]] = []
    for edge in record.get("sync_edges", []):
        if not isinstance(edge, Mapping) or not edge.get("loop_carried"):
            continue
        edge_source, edge_target = edge.get("source"), edge.get("target")
        if edge_source not in loop_nodes or edge_target not in loop_nodes:
            continue
        group = groups.get(edge.get("group"))
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
    durations, provenance, dynamic_loops = estimate_node_durations(record, model)
    node_ids = set(durations)
    stream_edges, _ = _graph_edges(record, node_ids, include_sync=False)
    full_edges, excluded_loop_carried = _graph_edges(record, node_ids, include_sync=True)
    full_edges = [
        (source, target, model.sync_latency_cycles if kind == "sync" else latency, kind, group)
        for source, target, latency, kind, group in full_edges
    ]

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

    covered = sum(item["source"] == "simulator_operation_median" for item in provenance.values())
    return {
        "schema_version": 1,
        "function": record.get("function", "<unknown>"),
        "status": record.get("status", "<unknown>"),
        "schedule_export_source": record.get("export_source", "native_schedule_graph_v1"),
        "schedule_export_limitations": record.get("export_limitations", {}),
        "duration_model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "loop_policy": "aggregate_static_work_v0",
        "dynamic_loop_ids": dynamic_loops,
        "excluded_loop_carried_sync_edges": excluded_loop_carried,
        "operation_nodes": len(durations),
        "calibrated_operation_nodes": covered,
        "calibration_coverage": covered / len(durations) if durations else 0.0,
        "baseline_makespan_cycles": baseline,
        "full_makespan_cycles": full,
        "synchronization_exposure_cycles": max(0.0, full - baseline),
        "baseline_critical_path": baseline_path,
        "full_critical_path": full_path,
        "sync_edge_exposure": edge_exposure,
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


def _deduplicate_scored_candidate_edges(
    rows: Sequence[dict[str, Any]],
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
        if len(weights) != 1:
            raise ValueError(
                "candidate records joined to one distance-zero edge but received different scores: "
                f"edge={source}->{target}"
            )
        distance_zero_edges.append(
            {
                "source_node": source,
                "target_node": target,
                "candidate_indices": [row["candidate_index"] for row in duplicates],
                "candidate_count": len(duplicates),
                "weight_cycles": duplicates[0]["weight_cycles"],
            }
        )

    loop_edges: list[dict[str, Any]] = []
    for (loop_node, source, target), duplicates in sorted(loop_rows.items()):
        weights = {float(row["weight_cycles"]) for row in duplicates}
        recurrence_cycles = {row["candidate_recurrence_cycles"] for row in duplicates}
        if len(weights) != 1 or len(recurrence_cycles) != 1:
            raise ValueError(
                "candidate records joined to one recurrence edge but received different scores: "
                f"loop={loop_node}, edge={source}->{target}"
            )
        loop_edges.append(
            {
                "loop_node": loop_node,
                "source_node": source,
                "target_node": target,
                "candidate_indices": [row["candidate_index"] for row in duplicates],
                "candidate_count": len(duplicates),
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


def score_reuse_candidates(
    record: Mapping[str, Any], candidates: Sequence[ReuseCandidateRecord], model: DurationModel
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
    durations, provenance, dynamic_loops = estimate_node_durations(record, model)
    node_ids = set(durations)
    existing_edges, excluded_loop_carried = _graph_edges(record, node_ids, include_sync=True)
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
        prior_pipe = _route_pipe(candidate.prior_route)
        next_pipe = _route_pipe(candidate.next_route)
        prior_nodes = indexed_nodes.get((candidate.prior_access_order, prior_pipe), [])
        next_nodes = indexed_nodes.get((candidate.next_access_order, next_pipe), [])
        if not prior_nodes or not next_nodes:
            raise ValueError(
                "candidate site did not join to the expected PTOAS pipe: "
                f"sites={candidate.prior_access_order}->{candidate.next_access_order}, "
                f"pipes={prior_pipe}->{next_pipe}, found={prior_nodes}->{next_nodes}"
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
                durations,
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
                    "prior_pipe": prior_pipe,
                    "next_pipe": next_pipe,
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
                "prior_pipe": prior_pipe,
                "next_pipe": next_pipe,
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

    distance_zero_edges, loop_edge_groups = _deduplicate_scored_candidate_edges(rows)
    weight_summary = _summarize_candidate_weights(distance_zero_edges, loop_edge_groups)

    return {
        "schema_version": 1,
        "model_version": "reuse_penalty_critical_path_v1",
        "function": record.get("function", "<unknown>"),
        "duration_model_version": model.model_version,
        "calibration_status": model.calibration_status,
        "base_makespan_cycles": base_makespan,
        "base_critical_path": base_path,
        "dynamic_loop_ids": dynamic_loops,
        "excluded_loop_carried_sync_edges": excluded_loop_carried,
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
        "candidate_weight_summary": weight_summary,
        "node_durations": {str(node): value for node, value in sorted(provenance.items())},
    }


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def calibrate_from_metrics(paths: Sequence[str | Path], base: DurationModel | None = None) -> DurationModel:
    """Calibrate operation and pipe medians from cleaned simulator metrics."""
    model = base or DurationModel()
    by_operation: dict[str, list[float]] = defaultdict(list)
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
                raw_name = next(
                    (
                        record.get(key)
                        for key in ("instruction", "instruction_name", "name", "opcode", "api")
                        if isinstance(record.get(key), str) and record.get(key)
                    ),
                    None,
                )
                if cycles is None or cycles <= 0 or not isinstance(raw_pipe, str):
                    continue
                pipe = _PIPE_ALIASES.get(raw_pipe.upper(), raw_pipe.upper())
                by_pipe[pipe].append(cycles)
                if raw_name is not None:
                    by_operation[_operation_key(pipe, raw_name)].append(cycles)
                source_used = True
        if source_used:
            used_sources.append(str(path))
    if not used_sources:
        raise ValueError("no finite positive instruction cycle samples found")

    calibrated_pipes = dict(model.pipe_parameters)
    for pipe, samples in by_pipe.items():
        previous = calibrated_pipes.get(pipe, PipeParameters(0.0, math.inf, 1.0))
        calibrated_pipes[pipe] = PipeParameters(
            startup_cycles=previous.startup_cycles,
            bytes_per_cycle=previous.bytes_per_cycle,
            minimum_cycles=statistics.median(samples),
        )
    operation_cycles = dict(model.operation_cycles)
    operation_cycles.update({key: statistics.median(samples) for key, samples in by_operation.items()})
    return DurationModel(
        schema_version=1,
        model_version="duration_v0",
        calibration_status="simulator_instruction_medians",
        sync_latency_cycles=model.sync_latency_cycles,
        pipe_parameters=calibrated_pipes,
        operation_cycles=operation_cycles,
        calibration_sources=sorted(used_sources),
    )


def _load_model(path: Path | None) -> DurationModel:
    if path is None:
        return DurationModel()
    return DurationModel.from_json(json.loads(path.read_text()))


def _write_json(path: Path | None, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(rendered)
    else:
        path.write_text(rendered)


def freeze_predictions(
    predictions: Mapping[str, Any], *, cohort: str, source_paths: Sequence[Path]
) -> dict[str, Any]:
    """Wrap predictions in a content-addressed holdout record."""
    canonical = json.dumps(predictions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "cohort": cohort,
        "frozen_before_device_timing": True,
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


def classify_straight_line_schedule(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify whether a schedule has no structured control-flow nodes.

    The first device calibration cohort deliberately excludes every branch and
    loop rather than approximating their execution frequency.  This predicate
    depends only on the exported schedule structure; it never reads solver
    objectives or device timings.
    """
    nodes = [node for node in record.get("nodes", []) if isinstance(node, Mapping)]
    operation_ids = [node.get("id") for node in nodes if node.get("kind") == "operation"]
    branch_ids = [node.get("id") for node in nodes if node.get("kind") == "branch"]
    loop_ids = [node.get("id") for node in nodes if node.get("kind") == "loop"]
    control_flow_ids = [*branch_ids, *loop_ids]
    return {
        "policy": "straight_line_v1",
        "eligible": not control_flow_ids,
        "status": "STRAIGHT_LINE" if not control_flow_ids else "CONTROL_FLOW_EXCLUDED",
        "operation_count": len(operation_ids),
        "branch_node_count": len(branch_ids),
        "loop_node_count": len(loop_ids),
        "branch_node_ids": branch_ids,
        "loop_node_ids": loop_ids,
    }


def qualify_schedule_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Return timing-blind straight-line eligibility for schedule graphs."""
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
                    **classify_straight_line_schedule(record),
                }
            )
    return {
        "schema_version": 1,
        "selection_policy": "straight_line_v1",
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
    direction_correct = sum(row["direction_correct"] is True for row in directional)
    return {
        "comparison_count": len(rows),
        "observed_comparison_count": len(observed),
        "directional_comparison_count": len(directional),
        "direction_correct_count": direction_correct,
        "direction_accuracy": direction_correct / len(directional) if directional else None,
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
            arm_results[role] = score_schedule(record, model)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsa_schedule_model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score schedule-graph JSONL files")
    score_parser.add_argument("schedules", nargs="+", type=Path)
    score_parser.add_argument("--model", type=Path)
    score_parser.add_argument("--freeze-cohort")
    score_parser.add_argument("-o", "--output", type=Path)

    calibrate_parser = subparsers.add_parser("calibrate", help="calibrate from instr_metrics.json files")
    calibrate_parser.add_argument("metrics", nargs="+", type=Path)
    calibrate_parser.add_argument("--base-model", type=Path)
    calibrate_parser.add_argument("-o", "--output", type=Path, required=True)

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
    evaluate_parser.add_argument("--model", type=Path)
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
    candidate_parser.add_argument("--model", type=Path)
    candidate_parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "calibrate":
            calibrated = calibrate_from_metrics(args.metrics, _load_model(args.base_model))
            _write_json(args.output, calibrated.to_json())
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
            evaluated = evaluate_arm_manifest(args.manifest, _load_model(args.model))
            _write_json(args.output, evaluated)
            return 0
        if args.command == "qualify":
            _write_json(args.output, qualify_schedule_files(args.schedules))
            return 0
        if args.command == "score-candidates":
            record, _ = _resolve_schedule_record(args.schedule, args.function)
            result = score_reuse_candidates(
                record, load_candidate_records(args.problem), _load_model(args.model)
            )
            _write_json(args.output, result)
            return 0

        model = _load_model(args.model)
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
