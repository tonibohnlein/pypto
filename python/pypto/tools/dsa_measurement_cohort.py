# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Freeze a timing-blind four-arm, four-capacity DSA measurement cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
CAPACITIES = ("native", "half", "q1", "tight")
REQUIRED_COLUMNS = (
    "case_id",
    "family",
    "parent",
    "function",
    "capacity",
    "arm",
    "solve_status",
    "launch_status",
    "correctness_status",
    "endpoint_digest",
)
_FORBIDDEN_COLUMN_FRAGMENTS = (
    "latency",
    "timing",
    "speedup",
    "effect",
    "objective",
    "reuse_cost",
    "penalty_sum",
    "makespan",
    "critical_path",
)


def _read_preflight(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        columns = reader.fieldnames or []
        missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        forbidden = sorted(
            column
            for column in columns
            if any(fragment in column.lower() for fragment in _FORBIDDEN_COLUMN_FRAGMENTS)
        )
        if forbidden:
            raise ValueError(
                f"{path}: performance/objective columns are forbidden in cohort selection: "
                f"{', '.join(forbidden)}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: preflight table is empty")
    return rows


def _case_reason(rows: Sequence[Mapping[str, str]]) -> str:
    expected = {(capacity, arm) for capacity in CAPACITIES for arm in ARMS}
    observed = [(row["capacity"], row["arm"]) for row in rows]
    if len(observed) != len(set(observed)):
        return "DUPLICATE_CELL"
    if set(observed) != expected:
        return "INCOMPLETE_FOUR_ARM_FOUR_CAPACITY_MATRIX"
    if any(row["solve_status"] != "FEASIBLE" for row in rows):
        return "ARM_OR_CAPACITY_INFEASIBLE"
    if any(row["launch_status"] != "RUNNABLE" for row in rows):
        return "NOT_RUNNABLE"
    if any(row["correctness_status"] != "PASS" for row in rows):
        return "CORRECTNESS_NOT_ESTABLISHED"
    if any(not row["endpoint_digest"] for row in rows):
        return "ENDPOINT_NOT_EMITTED"
    return "ELIGIBLE"


def _select_balanced(eligible: Sequence[tuple[str, Mapping[str, str]]], maximum: int) -> list[str]:
    buckets: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for case_id, metadata in sorted(
        eligible, key=lambda item: (item[1]["family"], item[1]["parent"], item[0])
    ):
        buckets[(metadata["family"], metadata["parent"])].append(case_id)
    selected: list[str] = []
    while len(selected) < maximum and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                selected.append(buckets[key].popleft())
                if len(selected) == maximum:
                    break
    return selected


def construct_cohort(
    rows: Sequence[Mapping[str, str]], *, minimum: int = 20, maximum: int = 40
) -> dict[str, Any]:
    """Classify and select cases without consulting performance or objectives."""
    if minimum <= 0 or maximum < minimum:
        raise ValueError("expected 0 < minimum <= maximum")
    by_case: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    audit: list[dict[str, str]] = []
    eligible: list[tuple[str, Mapping[str, str]]] = []
    for case_id, case_rows in sorted(by_case.items()):
        metadata = case_rows[0]
        for field in ("family", "parent", "function"):
            if any(row[field] != metadata[field] for row in case_rows):
                raise ValueError(f"case {case_id}: inconsistent {field}")
        reason = _case_reason(case_rows)
        audit.append(
            {
                "case_id": case_id,
                "family": metadata["family"],
                "parent": metadata["parent"],
                "function": metadata["function"],
                "status": reason,
            }
        )
        if reason == "ELIGIBLE":
            eligible.append((case_id, metadata))

    selected_ids = set(_select_balanced(eligible, maximum))
    selected = [
        {
            "case_id": case_id,
            "family": metadata["family"],
            "parent": metadata["parent"],
            "function": metadata["function"],
        }
        for case_id, metadata in eligible
        if case_id in selected_ids
    ]
    selected.sort(key=lambda row: row["case_id"])
    return {
        "schema_version": 1,
        "selection_policy": "four_arm_four_capacity_measurability_v1",
        "timing_blind": True,
        "required_arms": list(ARMS),
        "required_capacities": list(CAPACITIES),
        "candidate_count": len(by_case),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "verdict": "COHORT_FROZEN" if len(selected) >= minimum else "CORPUS_TOO_SMALL",
        "selected": selected,
        "audit": audit,
    }


def _write_tsv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_cohort(
    preflight_path: str | Path,
    output_directory: str | Path,
    *,
    minimum: int = 20,
    maximum: int = 40,
) -> dict[str, Any]:
    """Construct and write a content-addressed cohort and selection audit."""
    preflight = Path(preflight_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = construct_cohort(_read_preflight(preflight), minimum=minimum, maximum=maximum)
    _write_tsv(
        output / "selection-audit.tsv",
        ("case_id", "family", "parent", "function", "status"),
        result["audit"],
    )
    if result["verdict"] == "COHORT_FROZEN":
        cohort_path = output / "cohort-frozen.tsv"
        _write_tsv(
            cohort_path,
            ("case_id", "family", "parent", "function"),
            result["selected"],
        )
        digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
        (output / "cohort-frozen.tsv.sha256").write_text(f"{digest}  cohort-frozen.tsv\n")
        result["cohort_sha256"] = digest
    else:
        # A failed rerun must not leave a prior frozen cohort looking current.
        for name in ("cohort-frozen.tsv", "cohort-frozen.tsv.sha256"):
            (output / name).unlink(missing_ok=True)
    summary = {key: value for key, value in result.items() if key not in {"selected", "audit"}}
    summary["preflight_sha256"] = hashlib.sha256(preflight.read_bytes()).hexdigest()
    (output / "cohort-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preflight", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--minimum", type=int, default=20)
    parser.add_argument("--maximum", type=int, default=40)
    args = parser.parse_args(argv)
    result = freeze_cohort(args.preflight, args.output_directory, minimum=args.minimum, maximum=args.maximum)
    return 0 if result["verdict"] == "COHORT_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
