# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
"""Canonicalize DSA reuse by physical ranges and compare address layouts.

The DSA objective is expressed as logical buffer pairs. Several of those pairs
can collapse onto one physical byte range, so treating every active edge as an
independent hardware event over-counts the realized layout. This tool reports
both the logical edges and their canonical physical overlap groups.

The optional interleave table is deliberately a sensitivity analysis. It does
not claim an undocumented hardware bank mapping: each supplied period is a
hypothesis, and the output describes address-residue concentration under it.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _named_paths(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"{option} repeats name {name!r}")
        result[name] = Path(raw_path)
    return result


def _ranges_overlap(first_begin: int, first_end: int, second_begin: int, second_end: int) -> bool:
    return first_begin < second_end and second_begin < first_end


def _placements(problem: dict[str, Any], solution: dict[str, Any]) -> dict[int, dict[str, int]]:
    buffers = {int(buffer["id"]): buffer for buffer in problem["problem"]["buffers"]}
    result: dict[int, dict[str, int]] = {}
    for placement in solution["placements"]:
        buffer_id = int(placement["buffer"])
        if buffer_id in result:
            raise ValueError(f"solution repeats buffer {buffer_id}")
        if buffer_id not in buffers:
            raise ValueError(f"solution references unknown buffer {buffer_id}")
        result[buffer_id] = {
            "pool": int(placement["pool"]),
            "offset": int(placement["offset"]),
            "size": int(buffers[buffer_id]["size"]),
        }
    if set(result) != set(buffers):
        raise ValueError("solution and problem buffer sets differ")
    return result


def _physical_components(placements: dict[int, dict[str, int]]) -> list[dict[str, Any]]:
    adjacency = {buffer_id: set() for buffer_id in placements}
    ordered = sorted(placements)
    for index, first_id in enumerate(ordered):
        first = placements[first_id]
        for second_id in ordered[index + 1 :]:
            second = placements[second_id]
            if first["pool"] != second["pool"]:
                continue
            if _ranges_overlap(
                first["offset"],
                first["offset"] + first["size"],
                second["offset"],
                second["offset"] + second["size"],
            ):
                adjacency[first_id].add(second_id)
                adjacency[second_id].add(first_id)

    components: list[dict[str, Any]] = []
    unseen = set(placements)
    while unseen:
        pending = [min(unseen)]
        members: set[int] = set()
        while pending:
            buffer_id = pending.pop()
            if buffer_id in members:
                continue
            members.add(buffer_id)
            pending.extend(adjacency[buffer_id] - members)
        unseen -= members
        pools = {placements[buffer_id]["pool"] for buffer_id in members}
        if len(pools) != 1:
            raise ValueError("one physical overlap component spans multiple pools")
        begin = min(placements[buffer_id]["offset"] for buffer_id in members)
        end = max(placements[buffer_id]["offset"] + placements[buffer_id]["size"] for buffer_id in members)
        components.append(
            {
                "pool": pools.pop(),
                "begin": begin,
                "end": end,
                "span_bytes": end - begin,
                "buffers": sorted(members),
            }
        )
    components.sort(key=lambda component: (component["pool"], component["begin"], component["end"]))
    for index, component in enumerate(components):
        component["id"] = index
    return components


def _address_groups(placements: dict[int, dict[str, int]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for buffer_id, placement in placements.items():
        grouped[(placement["pool"], placement["offset"])].append(buffer_id)
    result = []
    for index, ((pool, begin), members) in enumerate(sorted(grouped.items())):
        end = max(begin + placements[buffer_id]["size"] for buffer_id in members)
        result.append(
            {
                "id": index,
                "pool": pool,
                "begin": begin,
                "end": end,
                "span_bytes": end - begin,
                "buffers": sorted(members),
            }
        )
    return result


def _operation_accesses(schedule: dict[str, Any] | None) -> list[dict[str, Any]]:
    if schedule is None:
        return []
    accesses = []
    for node in schedule.get("nodes", []):
        if node.get("kind") != "operation":
            continue
        addresses: set[int] = set()
        for value in [*(node.get("uses") or []), *(node.get("defs") or [])]:
            if value.get("known_physical_addresses") and value.get("scope") != "GM":
                addresses.update(int(address) for address in value.get("base_addresses", []))
        if addresses:
            accesses.append(
                {
                    "node": int(node["id"]),
                    "op": node.get("op_name", ""),
                    "pipe": node.get("pipe", ""),
                    "order": (node.get("operation") or {}).get("pypto_access_order"),
                    "addresses": sorted(addresses),
                }
            )
    return accesses


def _component_for_range(components: list[dict[str, Any]], pool: int, begin: int, end: int) -> dict[str, Any]:
    matches = [
        component
        for component in components
        if component["pool"] == pool and component["begin"] <= begin and end <= component["end"]
    ]
    if len(matches) != 1:
        raise ValueError(f"physical range [{begin},{end}) belongs to {len(matches)} components")
    return matches[0]


def _canonical_relations(
    problem: dict[str, Any],
    placements: dict[int, dict[str, int]],
    components: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logical = []
    for edge in problem["problem"].get("cost_model", {}).get("reuse_penalties", []):
        first_id, second_id = int(edge["first"]), int(edge["second"])
        first, second = placements[first_id], placements[second_id]
        if first["pool"] != second["pool"]:
            continue
        begin = max(first["offset"], second["offset"])
        end = min(first["offset"] + first["size"], second["offset"] + second["size"])
        if begin >= end:
            continue
        component = _component_for_range(components, first["pool"], begin, end)
        logical.append(
            {
                "first": first_id,
                "second": second_id,
                "cost": int(edge["cost"]),
                "reason": edge.get("reason", ""),
                "pool": first["pool"],
                "begin": begin,
                "end": end,
                "shared_bytes": end - begin,
                "physical_component": component["id"],
            }
        )

    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in logical:
        key = (edge["physical_component"], edge["pool"], edge["begin"], edge["end"])
        grouped[key].append(edge)
    canonical = []
    for (component_id, pool, begin, end), edges in sorted(grouped.items()):
        canonical.append(
            {
                "physical_component": component_id,
                "pool": pool,
                "begin": begin,
                "end": end,
                "shared_bytes": end - begin,
                "logical_edges": [[edge["first"], edge["second"]] for edge in edges],
                "logical_edge_count": len(edges),
                "logical_cost": sum(edge["cost"] for edge in edges),
                "reasons": sorted({edge["reason"] for edge in edges}),
            }
        )
    return logical, canonical


def _interleave_sensitivity(
    address_groups: list[dict[str, Any]],
    accesses: list[dict[str, Any]],
    periods: list[int],
) -> list[dict[str, Any]]:
    group_accesses = Counter()
    groups_by_begin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in address_groups:
        groups_by_begin[group["begin"]].append(group)
    for access in accesses:
        for address in access["addresses"]:
            matches = groups_by_begin[address]
            if len(matches) != 1:
                raise ValueError(f"operation address {address} matches {len(matches)} address groups")
            group_accesses[matches[0]["id"]] += 1

    result = []
    for period in periods:
        if period <= 0:
            raise ValueError(f"interleave period must be positive, got {period}")
        groups_by_residue: dict[int, list[int]] = defaultdict(list)
        access_weight_by_residue = Counter()
        for group in address_groups:
            residue = group["begin"] % period
            groups_by_residue[residue].append(group["id"])
            access_weight_by_residue[residue] += group_accesses[group["id"]]
        weights = list(access_weight_by_residue.values())
        total_weight = sum(weights)
        concentration = (
            sum(weight * weight for weight in weights) / (total_weight * total_weight)
            if total_weight
            else None
        )
        same_residue_transitions = 0
        transition_count = 0
        prior_residues: set[int] | None = None
        for access in accesses:
            residues = {address % period for address in access["addresses"]}
            if prior_residues is not None:
                transition_count += 1
                same_residue_transitions += int(bool(prior_residues & residues))
            prior_residues = residues
        result.append(
            {
                "period_bytes": period,
                "unique_group_residues": len(groups_by_residue),
                "max_groups_per_residue": max(map(len, groups_by_residue.values()), default=0),
                "group_collision_pairs": sum(
                    len(groups) * (len(groups) - 1) // 2 for groups in groups_by_residue.values()
                ),
                "access_weight_concentration": concentration,
                "same_residue_access_transitions": same_residue_transitions,
                "access_transitions": transition_count,
                "residue_groups": {
                    str(residue): groups for residue, groups in sorted(groups_by_residue.items())
                },
            }
        )
    return result


def analyze(
    problem: dict[str, Any],
    solutions: dict[str, dict[str, Any]],
    schedules: dict[str, dict[str, Any]],
    periods: list[int],
) -> dict[str, Any]:
    if set(schedules) - set(solutions):
        raise ValueError(f"schedule names have no solution: {sorted(set(schedules) - set(solutions))}")
    arms = {}
    for name, solution in solutions.items():
        placements = _placements(problem, solution)
        components = _physical_components(placements)
        address_groups = _address_groups(placements)
        accesses = _operation_accesses(schedules.get(name))
        logical, canonical = _canonical_relations(problem, placements, components)
        arms[name] = {
            "placement": [
                {"buffer": buffer_id, **placement} for buffer_id, placement in sorted(placements.items())
            ],
            "physical_components": components,
            "address_groups": address_groups,
            "active_logical_reuse_edges": logical,
            "canonical_physical_reuse_groups": canonical,
            "logical_reuse_cost": sum(edge["cost"] for edge in logical),
            "canonical_reuse_group_count": len(canonical),
            "operation_accesses": accesses,
            "interleave_sensitivity": _interleave_sensitivity(address_groups, accesses, periods),
        }
    return {
        "schema_version": 1,
        "instance": problem.get("instance"),
        "interleave_interpretation": "hypothesis_only_no_hardware_bank_mapping_assumed",
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--solution", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--schedule", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--period", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        solutions = {
            name: _read_object(path) for name, path in _named_paths(args.solution, "--solution").items()
        }
        schedules = {
            name: _read_object(path) for name, path in _named_paths(args.schedule, "--schedule").items()
        }
        if not solutions:
            raise ValueError("at least one --solution is required")
        report = analyze(
            _read_object(args.problem),
            solutions,
            schedules,
            args.period or [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
