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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
PRIMARY_ARMS = ("geometry_ff", "cypress", "dsa_rp_cg")
CAPACITIES = ("native", "half", "q1", "tight")


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


def _classify_capacity(rows: Sequence[Mapping[str, str]]) -> tuple[str, dict[str, str]]:
    by_arm = {row["arm"]: row for row in rows}
    if set(by_arm) != set(ARMS):
        return "INCOMPLETE_FOUR_ARM_CELL", {}
    if any(row["status"].upper() != "FEASIBLE" for row in by_arm.values()):
        return "FOUR_ARM_INFEASIBLE", {}

    capacities = {row["capacity"] for row in by_arm.values()}
    if len(capacities) != 1:
        return "CAPACITY_MISMATCH", {}
    cypress_aliases = int(by_arm["cypress"].get("cypress_actual_alias_pairs") or 0)
    if cypress_aliases == 0:
        return "NO_CYPRESS_REUSE_PRESSURE", {}

    placements = {arm: by_arm[arm].get("pool_placement_sha256", "") for arm in PRIMARY_ARMS}
    if any(not digest for digest in placements.values()):
        return "MISSING_POOL_PLACEMENT_DIGEST", {}
    if len(set(placements.values())) != len(PRIMARY_ARMS):
        return "NO_THREE_WAY_PLACEMENT_SEPARATION", {}

    return (
        "PRIMARY",
        {
            "capacity_bytes": next(iter(capacities)),
            "cypress_actual_alias_pairs": str(cypress_aliases),
            "geometry_ff_pool_placement_sha256": placements["geometry_ff"],
            "cypress_pool_placement_sha256": placements["cypress"],
            "dsa_rp_cg_pool_placement_sha256": placements["dsa_rp_cg"],
        },
    )


def select_evaluation_capacities(
    problems_path: str | Path,
    screen_results_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Freeze the least restrictive structurally nontrivial capacity per problem."""
    problems = _read_tsv(Path(problems_path))
    screen_rows = _read_tsv(Path(screen_results_path))
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
        attempts: list[str] = []
        selected_capacity = ""
        selected_facts: dict[str, str] = {}
        for capacity in CAPACITIES:
            status, facts = _classify_capacity(_problem_screen_rows(screen_by_cell, problem, capacity))
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
                "selection_status": "PRIMARY" if selected_capacity else attempts[-1].split(":", 1)[1],
                "evaluation_capacity": selected_capacity,
                "capacity_bytes": selected_facts.get("capacity_bytes", ""),
                "cypress_actual_alias_pairs": selected_facts.get("cypress_actual_alias_pairs", ""),
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
    summary = {
        "verdict": "EVALUATION_CAPACITIES_SELECTED",
        "selection_sha256": selection_sha,
        "problem_count": len(selection_rows),
        "primary_count": sum(row["selection_status"] == "PRIMARY" for row in selection_rows),
        "capacity_counts": {
            capacity: sum(row["evaluation_capacity"] == capacity for row in selection_rows)
            for capacity in CAPACITIES
        },
        "selection_rule": (
            "least_restrictive_capacity_with_cypress_reuse_and_distinct_"
            "geometry_cypress_dsa_rp_pool_placements"
        ),
        "uses_device_latency": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--screen-results", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    select_evaluation_capacities(args.problems, args.screen_results, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
