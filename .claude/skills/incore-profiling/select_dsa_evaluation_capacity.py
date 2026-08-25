# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Choose one timing-blind, reuse-pressured capacity per DSA problem."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
PRIMARY_ARMS = ("geometry_ff", "cypress", "dsa_rp_cg")
CAPACITIES = ("native", "half", "q1", "tight")
DEVICE_MEASURABLE_STATUS = "MEASURED"
_FINGERPRINT_RE = re.compile(r"-([0-9a-f]{16})\.dsa\.json$")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return [dict(row) for row in csv.DictReader(source, delimiter="\t")]


def _write_tsv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _selection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(list(rows), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _problem_screen_rows(
    screen_by_cell: Mapping[tuple[str, str, str], list[dict[str, str]]],
    problem: Mapping[str, str],
    capacity: str,
) -> list[dict[str, str]]:
    key = (problem["problem_fingerprint"][:16], problem["pool_id"], capacity)
    return screen_by_cell.get(key, [])


def _problem_documents_by_fingerprint(problems_directory: Path) -> dict[str, list[Path]]:
    documents: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(problems_directory.rglob("*.dsa.json")):
        match = _FINGERPRINT_RE.search(path.name)
        if match is not None:
            documents[match.group(1)].append(path)
    return documents


def _mandatory_disjoint_bytes_lower_bound(
    documents_by_fingerprint: Mapping[str, Sequence[Path]],
    problem: Mapping[str, str],
) -> int:
    fingerprint = problem["problem_fingerprint"][:16]
    paths = documents_by_fingerprint.get(fingerprint, ())
    if not paths:
        raise FileNotFoundError(
            f"No DSA problem document ending in -{fingerprint}.dsa.json under the problems directory"
        )

    pool_id = int(problem["pool_id"])
    lower_bounds: set[int] = set()
    for path in paths:
        document = json.loads(path.read_text())
        pools = {int(pool["id"]) for pool in document["problem"]["pools"]}
        if pool_id not in pools:
            raise ValueError(f"Problem {path} does not define selected pool {pool_id}")
        colocations = document["problem"].get("constraints", {}).get("colocations", [])
        if colocations:
            raise ValueError(
                f"Problem {path} has hard colocations; summing logical buffer sizes would "
                "overstate its disjoint lower bound"
            )
        lower_bounds.add(
            sum(
                int(buffer["size"])
                for buffer in document["problem"]["buffers"]
                if [int(candidate) for candidate in buffer["allowed_pools"]] == [pool_id]
            )
        )

    if len(lower_bounds) != 1:
        raise ValueError(
            f"Problem documents for fingerprint {fingerprint} disagree on the disjoint lower bound: "
            f"{sorted(lower_bounds)}"
        )
    lower_bound = next(iter(lower_bounds))
    if lower_bound <= 0:
        raise ValueError(
            f"Problem {fingerprint} has no buffers fixed to selected pool {pool_id}; "
            "forced reuse cannot be proven"
        )
    return lower_bound


def _device_status_by_problem(problem_status_path: Path) -> dict[tuple[str, str, str], str]:
    statuses: dict[tuple[str, str, str], str] = {}
    for row in _read_tsv(problem_status_path):
        key = (row["driver_id"], row["instance"], row["problem_fingerprint"])
        if key in statuses:
            raise ValueError(f"Duplicate device-status row for {key}")
        statuses[key] = row["status"]
    return statuses


def _classify_capacity(
    rows: Sequence[Mapping[str, str]],
    mandatory_disjoint_bytes: int,
    minimum_forced_reuse_percent: int,
) -> tuple[str, dict[str, str]]:
    by_arm = {row["arm"]: row for row in rows}
    if set(by_arm) != set(ARMS):
        return "INCOMPLETE_FOUR_ARM_CELL", {}
    if any(row["status"].upper() != "FEASIBLE" for row in by_arm.values()):
        return "FOUR_ARM_INFEASIBLE", {}

    capacities = {row["capacity"] for row in by_arm.values()}
    if len(capacities) != 1:
        return "CAPACITY_MISMATCH", {}
    capacity_bytes = int(next(iter(capacities)))
    if capacity_bytes >= mandatory_disjoint_bytes:
        return "CAPACITY_DOES_NOT_FORCE_REUSE", {}
    forced_reuse_bytes = mandatory_disjoint_bytes - capacity_bytes
    if forced_reuse_bytes * 100 < mandatory_disjoint_bytes * minimum_forced_reuse_percent:
        return "INSUFFICIENT_FORCED_REUSE_PRESSURE", {}
    cypress_aliases = int(by_arm["cypress"].get("cypress_actual_alias_pairs") or 0)
    if cypress_aliases == 0:
        return "NO_CYPRESS_REUSE_PRESSURE", {}

    placements = {arm: by_arm[arm].get("pool_placement_sha256", "") for arm in PRIMARY_ARMS}
    if any(not digest for digest in placements.values()):
        return "MISSING_POOL_PLACEMENT_DIGEST", {}
    if len(set(placements.values())) != len(PRIMARY_ARMS):
        return "NO_THREE_WAY_PLACEMENT_SEPARATION", {}

    reuse_costs = {arm: int(by_arm[arm]["reuse_cost"]) for arm in PRIMARY_ARMS}

    return (
        "PRIMARY",
        {
            "capacity_bytes": str(capacity_bytes),
            "mandatory_disjoint_bytes_lower_bound": str(mandatory_disjoint_bytes),
            "forced_reuse_bytes": str(forced_reuse_bytes),
            "forced_reuse_percent": f"{100 * forced_reuse_bytes / mandatory_disjoint_bytes:.6f}",
            "cypress_actual_alias_pairs": str(cypress_aliases),
            "geometry_ff_reuse_cost": str(reuse_costs["geometry_ff"]),
            "cypress_reuse_cost": str(reuse_costs["cypress"]),
            "dsa_rp_cg_reuse_cost": str(reuse_costs["dsa_rp_cg"]),
            "dsa_rp_minus_cypress_reuse_cost": str(reuse_costs["dsa_rp_cg"] - reuse_costs["cypress"]),
            "geometry_ff_pool_placement_sha256": placements["geometry_ff"],
            "cypress_pool_placement_sha256": placements["cypress"],
            "dsa_rp_cg_pool_placement_sha256": placements["dsa_rp_cg"],
        },
    )


def select_evaluation_capacities(
    problems_path: str | Path,
    screen_results_path: str | Path,
    problems_directory: str | Path,
    problem_status_path: str | Path,
    output_directory: str | Path,
    minimum_forced_reuse_percent: int = 25,
) -> dict[str, Any]:
    """Freeze the least restrictive uniformly pressured capacity per problem."""
    if not 0 <= minimum_forced_reuse_percent <= 100:
        raise ValueError(
            f"minimum_forced_reuse_percent must be between 0 and 100, got {minimum_forced_reuse_percent}"
        )
    problems = _read_tsv(Path(problems_path))
    screen_rows = _read_tsv(Path(screen_results_path))
    documents_by_fingerprint = _problem_documents_by_fingerprint(Path(problems_directory))
    device_statuses = _device_status_by_problem(Path(problem_status_path))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    screen_by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in screen_rows:
        fingerprint = row["tag"].rsplit("-", 1)[-1]
        screen_by_cell[(fingerprint, row["pool_id"], row["capacity_label"])].append(row)

    selection_rows: list[dict[str, str]] = []
    for problem in sorted(
        problems,
        key=lambda row: (row["driver_id"], row["instance"], row["problem_fingerprint"]),
    ):
        problem_key = (problem["driver_id"], problem["instance"], problem["problem_fingerprint"])
        device_status = device_statuses.get(problem_key, "MISSING_DEVICE_STATUS")
        mandatory_disjoint_bytes: int | None = None
        if device_status == DEVICE_MEASURABLE_STATUS:
            mandatory_disjoint_bytes = _mandatory_disjoint_bytes_lower_bound(
                documents_by_fingerprint, problem
            )
        attempts: list[str] = []
        selected_capacity = ""
        selected_facts: dict[str, str] = {}
        if device_status == DEVICE_MEASURABLE_STATUS:
            assert mandatory_disjoint_bytes is not None
            for capacity in CAPACITIES:
                status, facts = _classify_capacity(
                    _problem_screen_rows(screen_by_cell, problem, capacity),
                    mandatory_disjoint_bytes,
                    minimum_forced_reuse_percent,
                )
                attempts.append(f"{capacity}:{status}")
                if status == "PRIMARY":
                    selected_capacity = capacity
                    selected_facts = facts
                    break

        selection_rows.append(
            {
                "tier": problem["tier"],
                "driver_id": problem["driver_id"],
                "instance": problem["instance"],
                "problem_fingerprint": problem["problem_fingerprint"],
                "operation_class": problem["operation_class"],
                "pool": problem["pool"],
                "pool_id": problem["pool_id"],
                "buffers": problem["buffers"],
                "reuse_penalties": problem["reuse_penalties"],
                "device_status": device_status,
                "minimum_forced_reuse_percent": str(minimum_forced_reuse_percent),
                "selection_status": (
                    "PRIMARY"
                    if selected_capacity
                    else (
                        "NOT_DEVICE_MEASURABLE"
                        if device_status != DEVICE_MEASURABLE_STATUS
                        else attempts[-1].split(":", 1)[1]
                    )
                ),
                "evaluation_capacity": selected_capacity,
                "capacity_bytes": selected_facts.get("capacity_bytes", ""),
                "mandatory_disjoint_bytes_lower_bound": (
                    str(mandatory_disjoint_bytes) if mandatory_disjoint_bytes is not None else ""
                ),
                "forced_reuse_bytes": selected_facts.get("forced_reuse_bytes", ""),
                "forced_reuse_percent": selected_facts.get("forced_reuse_percent", ""),
                "cypress_actual_alias_pairs": selected_facts.get("cypress_actual_alias_pairs", ""),
                "geometry_ff_reuse_cost": selected_facts.get("geometry_ff_reuse_cost", ""),
                "cypress_reuse_cost": selected_facts.get("cypress_reuse_cost", ""),
                "dsa_rp_cg_reuse_cost": selected_facts.get("dsa_rp_cg_reuse_cost", ""),
                "dsa_rp_minus_cypress_reuse_cost": selected_facts.get("dsa_rp_minus_cypress_reuse_cost", ""),
                "geometry_ff_pool_placement_sha256": selected_facts.get(
                    "geometry_ff_pool_placement_sha256", ""
                ),
                "cypress_pool_placement_sha256": selected_facts.get("cypress_pool_placement_sha256", ""),
                "dsa_rp_cg_pool_placement_sha256": selected_facts.get("dsa_rp_cg_pool_placement_sha256", ""),
                "capacity_attempts": ";".join(attempts),
            }
        )

    columns = tuple(selection_rows[0]) if selection_rows else ()
    selection_path = output / "evaluation-instances.tsv"
    _write_tsv(selection_path, columns, selection_rows)
    selection_sha = _selection_sha256(selection_rows)
    (output / "evaluation-instances.tsv.sha256").write_text(
        f"{_file_sha256(selection_path)}  evaluation-instances.tsv\n"
    )
    reuse_cost_relations = {"lower": 0, "equal": 0, "higher": 0}
    for row in selection_rows:
        if row["selection_status"] != "PRIMARY":
            continue
        delta = int(row["dsa_rp_minus_cypress_reuse_cost"])
        relation = "lower" if delta < 0 else "higher" if delta > 0 else "equal"
        reuse_cost_relations[relation] += 1

    summary = {
        "verdict": "EVALUATION_CAPACITIES_SELECTED",
        "selection_sha256": selection_sha,
        "problem_count": len(selection_rows),
        "primary_count": sum(row["selection_status"] == "PRIMARY" for row in selection_rows),
        "capacity_counts": {
            capacity: sum(row["evaluation_capacity"] == capacity for row in selection_rows)
            for capacity in CAPACITIES
        },
        "dsa_rp_vs_cypress_reuse_cost": reuse_cost_relations,
        "selection_rule": (
            "least_restrictive_capacity_meeting_uniform_forced_reuse_pressure_with_"
            "cypress_reuse_and_distinct_geometry_cypress_dsa_rp_pool_placements"
        ),
        "minimum_forced_reuse_percent": minimum_forced_reuse_percent,
        "uses_device_latency": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--problems-dir", required=True, type=Path)
    parser.add_argument("--problem-status", required=True, type=Path)
    parser.add_argument("--screen-results", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--minimum-forced-reuse-percent",
        type=int,
        default=25,
        help="Minimum percentage of the disjoint-size lower bound that must not fit (default: 25)",
    )
    args = parser.parse_args(argv)
    select_evaluation_capacities(
        args.problems,
        args.screen_results,
        args.problems_dir,
        args.problem_status,
        args.output_root,
        args.minimum_forced_reuse_percent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
