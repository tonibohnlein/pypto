# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Evaluate a sealed opportunity-capacity freeze against existing device timings.

This is deliberately a second-stage development analysis. The freeze must have
been produced without latency, and this tool never changes its selected rows.
"""

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from select_dsa_workload_capacity import OPPORTUNITY_POLICY

COMPARISONS = (
    "cypress vs geometry_ff",
    "dsa_rp_cg vs geometry_ff",
    "dsa_rp_cg vs cypress",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return [dict(row) for row in csv.DictReader(source, delimiter="\t")]


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Development evaluation produced no rows")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second, strict=True))
    first_norm = sum((x - first_mean) ** 2 for x in first)
    second_norm = sum((y - second_mean) ** 2 for y in second)
    if first_norm == 0 or second_norm == 0:
        return None
    return numerator / math.sqrt(first_norm * second_norm)


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(first), _average_ranks(second))


def _candidate_direction(row: Mapping[str, str]) -> str:
    first = float(row["A_percent_change"])
    second = float(row["B_percent_change"])
    if first < 0 and second < 0:
        return "CANDIDATE_FASTER"
    if first > 0 and second > 0:
        return "BASELINE_FASTER"
    return "DEVICE_DIRECTIONS_DISAGREE"


def evaluate(
    freeze_path: str | Path,
    pairwise_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Join a timing-blind structural freeze to a previously measured timing table."""
    freeze_payload = Path(freeze_path).read_bytes()
    freeze = json.loads(freeze_payload)
    if freeze.get("selection_policy") != OPPORTUNITY_POLICY:
        raise ValueError(f"Unexpected selection policy: {freeze.get('selection_policy')}")
    if freeze.get("uses_device_latency") is not False:
        raise ValueError("Opportunity freeze is not timing-blind")

    pairwise_payload = Path(pairwise_path).read_bytes()
    timings: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_tsv(Path(pairwise_path)):
        if row["comparison"] not in COMPARISONS:
            continue
        key = (row["script"], row["capacity"], row["comparison"])
        if key in timings:
            raise ValueError(f"Existing timing table repeats comparison {key}")
        timings[key] = row
    rows: list[dict[str, Any]] = []
    for selected in freeze["workloads"]:
        key = (selected["script"], selected["evaluation_capacity"])
        selected_timings = {}
        for comparison in COMPARISONS:
            comparison_key = (*key, comparison)
            if comparison_key not in timings:
                raise ValueError(f"Existing timing table omits selected comparison {comparison_key}")
            selected_timings[comparison] = timings[comparison_key]
        timing = selected_timings["dsa_rp_cg vs cypress"]
        opportunity = selected["selection_status"] == "OPPORTUNITY_PRIMARY"
        candidate_direction = _candidate_direction(timing)
        direction = {
            "CANDIDATE_FASTER": "DSA_RP_FASTER",
            "BASELINE_FASTER": "CYPRESS_FASTER",
            "DEVICE_DIRECTIONS_DISAGREE": "DEVICE_DIRECTIONS_DISAGREE",
        }[candidate_direction]
        if not opportunity:
            assessment = "NULL_CONTROL"
        elif timing["verdict"] != "CONFIRMED":
            assessment = "UNRESOLVED"
        elif direction == "DSA_RP_FASTER":
            assessment = "OBJECTIVE_ORDER_AGREES"
        elif direction == "CYPRESS_FASTER":
            assessment = "OBJECTIVE_ORDER_DISAGREES"
        else:
            assessment = "UNRESOLVED"
        first_change = float(timing["A_percent_change"])
        second_change = float(timing["B_percent_change"])
        cypress_geometry = selected_timings["cypress vs geometry_ff"]
        dsa_rp_geometry = selected_timings["dsa_rp_cg vs geometry_ff"]
        full_order = all(
            comparison["verdict"] == "CONFIRMED" and _candidate_direction(comparison) == "CANDIDATE_FASTER"
            for comparison in (cypress_geometry, dsa_rp_geometry, timing)
        )
        penalty_aware_beat_geometry = all(
            comparison["verdict"] == "CONFIRMED" and _candidate_direction(comparison) == "CANDIDATE_FASTER"
            for comparison in (cypress_geometry, dsa_rp_geometry)
        )
        rows.append(
            {
                "script": selected["script"],
                "measurement_unit": selected["measurement_unit"],
                "capacity": selected["evaluation_capacity"],
                "selection_status": selected["selection_status"],
                "cypress_minus_dsa_rp_reuse_cost": selected["cypress_minus_dsa_rp_reuse_cost"],
                "penalized_relation_disagreement": selected["penalized_relation_disagreement"],
                "reuse_relation_disagreement": selected["reuse_relation_disagreement"],
                "A_dsa_rp_vs_cypress_percent": timing["A_percent_change"],
                "B_dsa_rp_vs_cypress_percent": timing["B_percent_change"],
                "A_cypress_vs_geometry_percent": cypress_geometry["A_percent_change"],
                "B_cypress_vs_geometry_percent": cypress_geometry["B_percent_change"],
                "cypress_vs_geometry_verdict": cypress_geometry["verdict"],
                "A_dsa_rp_vs_geometry_percent": dsa_rp_geometry["A_percent_change"],
                "B_dsa_rp_vs_geometry_percent": dsa_rp_geometry["B_percent_change"],
                "dsa_rp_vs_geometry_verdict": dsa_rp_geometry["verdict"],
                "mean_dsa_rp_advantage_percent": -(first_change + second_change) / 2,
                "device_direction": direction,
                "timing_verdict": timing["verdict"],
                "development_assessment": assessment,
                "full_geometry_cypress_dsa_rp_order": "YES" if full_order else "NO",
                "both_penalty_aware_beat_geometry": "YES" if penalty_aware_beat_geometry else "NO",
            }
        )

    primary = [row for row in rows if row["selection_status"] == "OPPORTUNITY_PRIMARY"]
    objective_gaps = [float(row["cypress_minus_dsa_rp_reuse_cost"]) for row in primary]
    penalty_disagreements = [float(row["penalized_relation_disagreement"]) for row in primary]
    advantages = [float(row["mean_dsa_rp_advantage_percent"]) for row in primary]
    summary = {
        "schema_version": 1,
        "analysis_kind": "development_set_post_freeze_join",
        "prospective_evidence": False,
        "selection_policy": OPPORTUNITY_POLICY,
        "freeze_sha256": hashlib.sha256(freeze_payload).hexdigest(),
        "pairwise_effects_sha256": hashlib.sha256(pairwise_payload).hexdigest(),
        "workload_count": len(rows),
        "opportunity_count": len(primary),
        "null_control_count": len(rows) - len(primary),
        "confirmed_count": sum(row["timing_verdict"] == "CONFIRMED" for row in primary),
        "objective_order_agrees": sum(
            row["development_assessment"] == "OBJECTIVE_ORDER_AGREES" for row in primary
        ),
        "objective_order_disagrees": sum(
            row["development_assessment"] == "OBJECTIVE_ORDER_DISAGREES" for row in primary
        ),
        "unresolved": sum(row["development_assessment"] == "UNRESOLVED" for row in primary),
        "full_geometry_cypress_dsa_rp_order_confirmed": sum(
            row["full_geometry_cypress_dsa_rp_order"] == "YES" for row in primary
        ),
        "both_penalty_aware_beat_geometry": sum(
            row["both_penalty_aware_beat_geometry"] == "YES" for row in primary
        ),
        "spearman_objective_gap_vs_dsa_rp_advantage": _spearman(objective_gaps, advantages),
        "spearman_penalized_disagreement_vs_dsa_rp_advantage": _spearman(penalty_disagreements, advantages),
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    _write_tsv(output / "development-evaluation.tsv", rows)
    (output / "development-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--pairwise-effects", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.freeze, args.pairwise_effects, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
