# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze one timing-blind, reuse-opportunity capacity per verified workload."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
PRIMARY_ARMS = ("geometry_ff", "cypress", "dsa_rp_cg")
CAPACITIES_TIGHTEST_FIRST = ("tight", "q1", "half", "native")
VERIFIED_STATUS = "VERIFIED_ALL_CAPACITIES"
OPPORTUNITY_POLICY = "cypress_dsa_rp_penalty_opportunity_v1"
LEGACY_TIGHTEST_POLICY = "tightest_reuse_pressure_v1"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return [dict(row) for row in csv.DictReader(source, delimiter="\t")]


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _selection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), separators=(",", ":"), sort_keys=True).encode()
    return _sha256_bytes(payload)


def _capacity_profile(text: str) -> dict[int, int]:
    profile: dict[int, int] = {}
    for item in text.split(","):
        pool, separator, capacity = item.partition("=")
        if not separator:
            raise ValueError(f"Malformed capacity profile item {item!r}")
        pool_id = int(pool)
        if pool_id in profile:
            raise ValueError(f"Capacity profile repeats pool {pool_id}")
        profile[pool_id] = int(capacity)
    if not profile:
        raise ValueError("Capacity profile is empty")
    return profile


def _named_values(text: str, *, field: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in text.split(";"):
        name, separator, value = item.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"Malformed {field} item {item!r}")
        if name in values:
            raise ValueError(f"{field} repeats {name!r}")
        values[name] = value
    if not values:
        raise ValueError(f"{field} is empty")
    return values


def _verify_workload_problem_identity(
    workload: Mapping[str, str], instances: Sequence[Mapping[str, str]]
) -> None:
    expected = _named_values(workload["problem_fingerprints"], field="problem_fingerprints")
    observed = {row["instance"]: row["problem_fingerprint"] for row in instances}
    if len(observed) != len(instances):
        raise ValueError(f"Fresh instance inventory repeats an instance for {workload['script']}")
    if observed != expected:
        raise ValueError(f"Fresh DSA problems drifted for {workload['script']}: {observed} != {expected}")
    if int(workload["dsa_instance_count"]) != len(observed):
        raise ValueError(
            f"Frozen DSA instance count drifted for {workload['script']}: "
            f"{workload['dsa_instance_count']} != {len(observed)}"
        )


def _colocation_roots(problem: Mapping[str, Any]) -> dict[int, int]:
    buffers = {int(buffer["id"]) for buffer in problem["buffers"]}
    parents = {buffer: buffer for buffer in buffers}

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for relation in problem.get("constraints", {}).get("colocations", []):
        first, second = int(relation["first"]), int(relation["second"])
        if first not in buffers or second not in buffers:
            raise ValueError(f"Colocation references unknown buffers {first}, {second}")
        union(first, second)
    return {buffer: find(buffer) for buffer in buffers}


def mandatory_disjoint_bytes_by_pool(problem: Mapping[str, Any]) -> dict[int, int]:
    """Return a conservative disjoint-size lower bound for each memory pool."""
    roots = _colocation_roots(problem)
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for buffer in problem["buffers"]:
        groups[roots[int(buffer["id"])]].append(buffer)

    lower_bounds: dict[int, int] = defaultdict(int)
    for group in groups.values():
        allowed = set(int(pool) for pool in group[0]["allowed_pools"])
        for buffer in group[1:]:
            allowed &= {int(pool) for pool in buffer["allowed_pools"]}
        if not allowed:
            ids = sorted(int(buffer["id"]) for buffer in group)
            raise ValueError(f"Colocated buffers {ids} have no common allowed pool")
        if len(allowed) == 1:
            lower_bounds[next(iter(allowed))] += max(int(buffer["size"]) for buffer in group)
    return dict(lower_bounds)


def _overlap_relations(
    problem: Mapping[str, Any], solution: Mapping[str, Any]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    buffers = {int(buffer["id"]): buffer for buffer in problem["buffers"]}
    roots = _colocation_roots(problem)
    placements = {
        int(placement["buffer"]): (int(placement["pool"]), int(placement["offset"]))
        for placement in solution.get("placements", [])
    }
    if set(placements) != set(buffers):
        raise ValueError("Solution placement coverage does not match the DSA problem")

    overlaps: set[tuple[int, int]] = set()
    for first in sorted(buffers):
        first_pool, first_offset = placements[first]
        first_end = first_offset + int(buffers[first]["size"])
        for second in sorted(buffer for buffer in buffers if buffer > first):
            second_pool, second_offset = placements[second]
            second_end = second_offset + int(buffers[second]["size"])
            if (
                first_pool == second_pool
                and first_offset < second_end
                and second_offset < first_end
                and roots[first] != roots[second]
            ):
                overlaps.add((first, second))

    penalty_pairs = {
        tuple(sorted((int(penalty["first"]), int(penalty["second"]))))
        for penalty in (problem.get("cost_model") or {}).get("reuse_penalties", [])
    }
    return overlaps, overlaps & penalty_pairs


def _load_problem(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError(f"Problem document hash mismatch for {path}: {observed} != {expected_sha256}")
    return json.loads(payload)


def _resolve_map_directory(replay_root: Path, relative: str) -> Path:
    direct = replay_root / relative
    if direct.is_dir():
        return direct
    path = Path(relative)
    if path.parts and path.parts[0] == "artifacts":
        archived = replay_root.joinpath(*path.parts[1:])
        if archived.is_dir():
            return archived
    raise FileNotFoundError(f"Replay map directory does not exist under {replay_root}: {relative}")


def _capacity_facts(
    script: str,
    capacity: str,
    instances: Sequence[Mapping[str, str]],
    feasibility: Mapping[tuple[str, str, str, str], Mapping[str, str]],
    maps: Mapping[tuple[str, str, str], Mapping[str, str]],
    corpus_root: Path,
    replay_root: Path,
) -> tuple[str, dict[str, Any]]:
    map_rows = {arm: maps.get((script, capacity, arm)) for arm in ARMS}
    if any(row is None for row in map_rows.values()):
        return "INCOMPLETE_FOUR_ARM_MAP", {}
    assert all(row is not None for row in map_rows.values())
    if any(row["provenance_verified"] != "YES" for row in map_rows.values()):
        return "UNVERIFIED_MAP_PROVENANCE", {}

    complete_map_digests = {arm: str(map_rows[arm]["map_digest"]) for arm in PRIMARY_ARMS}
    distinct_maps = len(set(complete_map_digests.values()))
    cypress_vs_rp_distinct = complete_map_digests["cypress"] != complete_map_digests["dsa_rp_cg"]
    forced_reuse_bytes = 0
    forced_pools = 0
    cypress_alias_pairs = 0
    cypress_penalized_alias_pairs = 0
    cypress_reuse_cost = 0
    dsa_rp_reuse_cost = 0
    cypress_relations: set[tuple[str, int, int]] = set()
    dsa_rp_relations: set[tuple[str, int, int]] = set()
    cypress_penalized_relations: set[tuple[str, int, int]] = set()
    dsa_rp_penalized_relations: set[tuple[str, int, int]] = set()

    cypress_map = _resolve_map_directory(replay_root, str(map_rows["cypress"]["map_dir"]))
    dsa_rp_map = _resolve_map_directory(replay_root, str(map_rows["dsa_rp_cg"]["map_dir"]))
    for instance in instances:
        rows = {arm: feasibility.get((script, instance["instance"], capacity, arm)) for arm in ARMS}
        if any(row is None for row in rows.values()):
            return "INCOMPLETE_FOUR_ARM_INSTANCE", {}
        assert all(row is not None for row in rows.values())
        if any(row["status"] != "feasible" or row["validation"] != "VALID" for row in rows.values()):
            return "FOUR_ARM_INSTANCE_INFEASIBLE", {}
        profiles = {row["capacity_profile"] for row in rows.values()}
        if len(profiles) != 1:
            return "CAPACITY_PROFILE_MISMATCH", {}

        document = _load_problem(corpus_root / instance["document"], instance["document_sha256"])
        problem = document["problem"]
        profile = _capacity_profile(next(iter(profiles)))
        for pool, lower_bound in mandatory_disjoint_bytes_by_pool(problem).items():
            if pool not in profile:
                raise ValueError(f"Capacity profile for {script}:{instance['instance']} omits pool {pool}")
            shortage = max(0, lower_bound - profile[pool])
            forced_reuse_bytes += shortage
            forced_pools += shortage > 0

        solution_name = f"pypto_{instance['instance']}.dsa.solution.json"
        cypress_solution_path = cypress_map / solution_name
        dsa_rp_solution_path = dsa_rp_map / solution_name
        for arm, solution_path in (
            ("Cypress", cypress_solution_path),
            ("DSA-RP", dsa_rp_solution_path),
        ):
            if not solution_path.is_file():
                raise FileNotFoundError(f"{arm} replay map omits {solution_path.name}")
        cypress_overlaps, cypress_penalized = _overlap_relations(
            problem, json.loads(cypress_solution_path.read_text(encoding="utf-8"))
        )
        dsa_rp_overlaps, dsa_rp_penalized = _overlap_relations(
            problem, json.loads(dsa_rp_solution_path.read_text(encoding="utf-8"))
        )
        instance_name = instance["instance"]
        cypress_relations.update((instance_name, first, second) for first, second in cypress_overlaps)
        dsa_rp_relations.update((instance_name, first, second) for first, second in dsa_rp_overlaps)
        cypress_penalized_relations.update(
            (instance_name, first, second) for first, second in cypress_penalized
        )
        dsa_rp_penalized_relations.update(
            (instance_name, first, second) for first, second in dsa_rp_penalized
        )
        cypress_alias_pairs += len(cypress_overlaps)
        cypress_penalized_alias_pairs += len(cypress_penalized)
        cypress_reuse_cost += int(rows["cypress"]["reuse_cost"])
        dsa_rp_reuse_cost += int(rows["dsa_rp_cg"]["reuse_cost"])

    objective_gap = cypress_reuse_cost - dsa_rp_reuse_cost
    facts = {
        "capacity": capacity,
        "forced_reuse_bytes": forced_reuse_bytes,
        "forced_pools": forced_pools,
        "cypress_alias_pairs": cypress_alias_pairs,
        "cypress_penalized_alias_pairs": cypress_penalized_alias_pairs,
        "cypress_reuse_cost": cypress_reuse_cost,
        "dsa_rp_reuse_cost": dsa_rp_reuse_cost,
        "cypress_minus_dsa_rp_reuse_cost": objective_gap,
        "dsa_rp_minus_cypress_reuse_cost": dsa_rp_reuse_cost - cypress_reuse_cost,
        "penalized_relation_disagreement": len(
            cypress_penalized_relations.symmetric_difference(dsa_rp_penalized_relations)
        ),
        "reuse_relation_disagreement": len(cypress_relations.symmetric_difference(dsa_rp_relations)),
        "distinct_primary_maps": distinct_maps,
        "cypress_vs_dsa_rp_distinct": cypress_vs_rp_distinct,
        **{f"{arm}_map_digest": digest for arm, digest in complete_map_digests.items()},
    }
    if forced_pools == 0:
        return "CAPACITY_DOES_NOT_FORCE_REUSE", facts
    if cypress_alias_pairs == 0 or cypress_penalized_alias_pairs == 0:
        return "CYPRESS_DOES_NOT_REALIZE_PENALIZED_REUSE", facts
    if not cypress_vs_rp_distinct:
        return "CYPRESS_DSA_RP_IDENTICAL", facts
    if distinct_maps < 2:
        return "NO_POLICY_SEPARATION", facts
    return ("PRIMARY_THREE_WAY" if distinct_maps == 3 else "PRIMARY_TWO_WAY"), facts


def _opportunity_status(availability_status: str, facts: Mapping[str, Any]) -> str:
    if not facts:
        return availability_status
    if int(facts["cypress_penalized_alias_pairs"]) == 0:
        return "CYPRESS_DOES_NOT_REALIZE_PENALIZED_REUSE"
    if int(facts["distinct_primary_maps"]) != 3:
        return "GEOMETRY_CYPRESS_DSA_RP_NOT_DISTINCT"
    if int(facts["cypress_minus_dsa_rp_reuse_cost"]) <= 0:
        return "DSA_RP_OBJECTIVE_NOT_STRICTLY_BETTER"
    return "OPPORTUNITY_PRIMARY"


def _opportunity_score(facts: Mapping[str, Any], capacity: str) -> tuple[int, int, int, int]:
    """Rank eligible capacities without consulting latency; larger is better."""
    return (
        int(facts["cypress_minus_dsa_rp_reuse_cost"]),
        int(facts["penalized_relation_disagreement"]),
        int(facts["reuse_relation_disagreement"]),
        -CAPACITIES_TIGHTEST_FIRST.index(capacity),
    )


def _diagnostic_score(facts: Mapping[str, Any], capacity: str) -> tuple[int, ...]:
    """Choose a deterministic structural null when no opportunity exists."""
    return (
        int(facts["cypress_penalized_alias_pairs"] > 0),
        int(facts["distinct_primary_maps"] == 3),
        max(0, int(facts["cypress_minus_dsa_rp_reuse_cost"])),
        int(facts["penalized_relation_disagreement"]),
        int(facts["reuse_relation_disagreement"]),
        -CAPACITIES_TIGHTEST_FIRST.index(capacity),
    )


def select_workload_capacities(  # noqa: PLR0912, PLR0913
    cohort_path: str | Path,
    instances_path: str | Path,
    feasibility_path: str | Path,
    maps_path: str | Path,
    workload_status_path: str | Path,
    corpus_root: str | Path,
    replay_root: str | Path,
    output_root: str | Path,
    *,
    source_archive: str = "",
    source_archive_sha256: str = "",
    policy: str = OPPORTUNITY_POLICY,
    exclusions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select and freeze one structurally defined capacity per workload."""
    if policy not in {OPPORTUNITY_POLICY, LEGACY_TIGHTEST_POLICY}:
        raise ValueError(f"Unknown capacity-selection policy {policy!r}")
    exclusions = dict(exclusions or {})
    cohort = _read_tsv(Path(cohort_path))
    instances = _read_tsv(Path(instances_path))
    feasibility_rows = _read_tsv(Path(feasibility_path))
    map_rows = _read_tsv(Path(maps_path))
    statuses = {row["script"]: row["terminal_status"] for row in _read_tsv(Path(workload_status_path))}
    by_script: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in instances:
        by_script[row["script"]].append(row)
    feasibility = {
        (row["script"], row["instance"], row["capacity_label"], row["arm"]): row for row in feasibility_rows
    }
    maps = {(row["script"], row["capacity_label"], row["arm"]): row for row in map_rows}

    selection: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for workload in sorted(cohort, key=lambda row: row["script"]):
        script = workload["script"]
        if script in exclusions:
            excluded.append({"script": script, "reason": exclusions[script]})
            continue
        if statuses.get(script) != VERIFIED_STATUS:
            raise ValueError(f"Workload {script} is not verified at all capacities: {statuses.get(script)}")
        _verify_workload_problem_identity(workload, by_script[script])
        attempts: list[str] = []
        selected_status = ""
        selected_facts: dict[str, Any] = {}
        candidates: list[tuple[str, str, dict[str, Any]]] = []
        for capacity in CAPACITIES_TIGHTEST_FIRST:
            availability_status, facts = _capacity_facts(
                script,
                capacity,
                by_script[script],
                feasibility,
                maps,
                Path(corpus_root),
                Path(replay_root),
            )
            status = (
                _opportunity_status(availability_status, facts)
                if policy == OPPORTUNITY_POLICY
                else availability_status
            )
            attempts.append(f"{capacity}:{status}")
            if facts:
                candidates.append((capacity, status, facts))

        if policy == OPPORTUNITY_POLICY:
            opportunities = [candidate for candidate in candidates if candidate[1] == "OPPORTUNITY_PRIMARY"]
            if opportunities:
                capacity, selected_status, selected_facts = max(
                    opportunities, key=lambda candidate: _opportunity_score(candidate[2], candidate[0])
                )
                assert capacity == selected_facts["capacity"]
            elif candidates:
                capacity, reason, selected_facts = max(
                    candidates, key=lambda candidate: _diagnostic_score(candidate[2], candidate[0])
                )
                selected_status = f"NULL_CONTROL_{reason}"
                assert capacity == selected_facts["capacity"]
            else:
                raise ValueError(f"Workload {script} has no complete four-arm capacity: {attempts}")
        else:
            null_control: tuple[str, dict[str, Any]] | None = None
            for _capacity, status, facts in candidates:
                if status.startswith("PRIMARY_"):
                    selected_status, selected_facts = status, facts
                    break
                if status == "CYPRESS_DSA_RP_IDENTICAL" and null_control is None:
                    null_control = (status, facts)
            if not selected_status:
                if null_control is None:
                    raise ValueError(
                        f"Workload {script} has no primary capacity or explicit null control: {attempts}"
                    )
                selected_status, selected_facts = (
                    "NULL_CONTROL_CYPRESS_DSA_RP_IDENTICAL",
                    null_control[1],
                )

        selection.append(
            {
                "script": script,
                "measurement_unit": workload["measurement_unit"],
                "dsa_instances": workload["dsa_instance_count"],
                "problem_fingerprints": workload["problem_fingerprints"],
                "selection_status": selected_status,
                "evaluation_capacity": selected_facts["capacity"],
                "forced_reuse_bytes": selected_facts["forced_reuse_bytes"],
                "forced_pools": selected_facts["forced_pools"],
                "cypress_alias_pairs": selected_facts["cypress_alias_pairs"],
                "cypress_penalized_alias_pairs": selected_facts["cypress_penalized_alias_pairs"],
                "cypress_reuse_cost": selected_facts["cypress_reuse_cost"],
                "dsa_rp_reuse_cost": selected_facts["dsa_rp_reuse_cost"],
                "cypress_minus_dsa_rp_reuse_cost": selected_facts["cypress_minus_dsa_rp_reuse_cost"],
                "dsa_rp_minus_cypress_reuse_cost": selected_facts["dsa_rp_minus_cypress_reuse_cost"],
                "penalized_relation_disagreement": selected_facts["penalized_relation_disagreement"],
                "reuse_relation_disagreement": selected_facts["reuse_relation_disagreement"],
                "distinct_primary_maps": selected_facts["distinct_primary_maps"],
                "geometry_ff_map_digest": selected_facts["geometry_ff_map_digest"],
                "cypress_map_digest": selected_facts["cypress_map_digest"],
                "dsa_rp_cg_map_digest": selected_facts["dsa_rp_cg_map_digest"],
                "capacity_attempts": ";".join(attempts),
            }
        )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    table_path = output / "evaluation-workloads.tsv"
    _write_tsv(table_path, selection)
    selection_sha = _selection_sha256(selection)
    table_sha = _sha256_bytes(table_path.read_bytes())
    (output / "evaluation-workloads.tsv.sha256").write_text(
        f"{table_sha}  evaluation-workloads.tsv\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 2,
        "selection_policy": policy,
        "uses_device_latency": False,
        "source_archive": source_archive,
        "source_archive_sha256": source_archive_sha256,
        "selection_sha256": selection_sha,
        "evaluation_workloads_tsv_sha256": table_sha,
        "workload_count": len(selection),
        "primary_count": sum(
            row["selection_status"] == "OPPORTUNITY_PRIMARY" or row["selection_status"].startswith("PRIMARY_")
            for row in selection
        ),
        "null_control_count": sum(row["selection_status"].startswith("NULL_CONTROL_") for row in selection),
        "excluded_count": len(excluded),
        "excluded_workloads": excluded,
        "capacity_counts": {
            capacity: sum(row["evaluation_capacity"] == capacity for row in selection)
            for capacity in CAPACITIES_TIGHTEST_FIRST
        },
        "workloads": selection,
    }
    freeze_path = output / "evaluation-freeze.json"
    freeze_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["evaluation_freeze_sha256"] = _sha256_bytes(freeze_path.read_bytes())
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--feasibility", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--workload-status", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-archive", default="")
    parser.add_argument("--source-archive-sha256", default="")
    parser.add_argument(
        "--policy",
        choices=(OPPORTUNITY_POLICY, LEGACY_TIGHTEST_POLICY),
        default=OPPORTUNITY_POLICY,
    )
    parser.add_argument(
        "--exclude-script",
        action="append",
        default=[],
        metavar="SCRIPT=REASON",
        help="Exclude a correctness-blocked workload, recording the reason in the freeze",
    )
    args = parser.parse_args(argv)
    exclusions = (
        _named_values(";".join(args.exclude_script), field="exclude_script") if args.exclude_script else {}
    )
    summary = select_workload_capacities(
        args.cohort,
        args.instances,
        args.feasibility,
        args.maps,
        args.workload_status,
        args.corpus_root,
        args.replay_root,
        args.output_root,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
        policy=args.policy,
        exclusions=exclusions,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
