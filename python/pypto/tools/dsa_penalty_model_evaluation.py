# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare DSA penalty models with observed four-arm device ordering."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
METRICS = (
    "unit_realized_cost",
    "canonical_physical_reuse_group_count",
    "unique_induced_sync_edge_count",
    "estimated_sync_endpoint_executions",
    "critical_path_realized_cost_cycles",
    "complete_placement_critical_path_cycles",
)
REQUIRED_COLUMNS = (
    "case_id",
    "capacity",
    "device",
    "arm",
    "analysis_status",
    *METRICS,
    "latency_us",
    "cypress_auxiliary_edges",
    "cypress_relaxed_edges",
    "cypress_actual_alias_pairs",
    "cypress_packing_attempts",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: input table is empty")
    return rows


def _number(value: str, *, field: str, allow_empty: bool = False) -> float | None:
    if allow_empty and value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _direction(value: float) -> int:
    return (value > 0) - (value < 0)


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


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _metric_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    observed = [row for row in rows if row["observed_relative_delta"] is not None]
    directional = [
        row
        for row in observed
        if row[f"{metric}_predicted_direction"] != 0 and row["observed_direction"] != 0
    ]
    correct = sum(row[f"{metric}_direction_correct"] is True for row in directional)
    predicted = [float(row[f"{metric}_relative_delta"]) for row in observed]
    actual = [float(row["observed_relative_delta"]) for row in observed]
    return {
        "observed_comparisons": len(observed),
        "directional_comparisons": len(directional),
        "direction_correct": correct,
        "direction_accuracy": correct / len(directional) if directional else None,
        "spearman_relative_delta": _correlation(_average_ranks(predicted), _average_ranks(actual)),
    }


def _cypress_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [group for group in groups if group["cypress_vs_dsa_rp_relative_delta"] is not None]
    feature_names = (
        "cypress_auxiliary_edges",
        "cypress_relaxed_edges",
        "cypress_actual_alias_pairs",
        "cypress_packing_attempts",
        "cypress_relaxed_fraction",
    )
    return {
        "observed_cells": len(rows),
        "feature_spearman_vs_cypress_over_dsa_rp_latency": {
            feature: _correlation(
                _average_ranks([float(row[feature]) for row in rows]),
                _average_ranks([float(row["cypress_vs_dsa_rp_relative_delta"]) for row in rows]),
            )
            if len(rows) >= 2
            else None
            for feature in feature_names
        },
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, str]], *, split: str, freeze_before_timing: bool
) -> dict[str, Any]:
    """Evaluate additive penalty predictions without selecting on latency."""
    by_cell: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["case_id"], row["capacity"], row["device"])].append(row)

    comparisons: list[dict[str, Any]] = []
    cypress_groups: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    predictions: list[dict[str, Any]] = []
    for (case_id, capacity, device), cell_rows in sorted(by_cell.items()):
        if len(cell_rows) != len(ARMS):
            raise ValueError(
                f"{case_id}/{capacity}/{device}: expected {len(ARMS)} rows, got {len(cell_rows)}"
            )
        by_arm = {row["arm"]: row for row in cell_rows}
        if set(by_arm) != set(ARMS):
            raise ValueError(f"{case_id}/{capacity}/{device}: expected exactly four arms")
        statuses = {row["analysis_status"] for row in cell_rows}
        if statuses != {"MODELED"}:
            excluded.append(
                {
                    "case_id": case_id,
                    "capacity": capacity,
                    "device": device,
                    "reason": ",".join(sorted(statuses)),
                }
            )
            continue

        scores = {
            arm: {
                metric: _number(by_arm[arm][metric], field=f"{case_id}/{arm}/{metric}") for metric in METRICS
            }
            for arm in ARMS
        }
        latencies = {
            arm: _number(
                by_arm[arm]["latency_us"],
                field=f"{case_id}/{arm}/latency_us",
                allow_empty=True,
            )
            for arm in ARMS
        }
        if freeze_before_timing and any(value is not None for value in latencies.values()):
            raise ValueError("frozen holdout predictions must not contain device latency")
        if any(value is None for value in latencies.values()) and not all(
            value is None for value in latencies.values()
        ):
            raise ValueError(f"{case_id}/{capacity}/{device}: latencies must be all present or all absent")

        predictions.append(
            {
                "case_id": case_id,
                "capacity": capacity,
                "device": device,
                "arm_scores": scores,
            }
        )
        for baseline, candidate in combinations(ARMS, 2):
            comparison: dict[str, Any] = {
                "case_id": case_id,
                "capacity": capacity,
                "device": device,
                "baseline_arm": baseline,
                "candidate_arm": candidate,
            }
            for metric in METRICS:
                baseline_score = float(scores[baseline][metric])
                candidate_score = float(scores[candidate][metric])
                delta = candidate_score - baseline_score
                relative_delta = delta / baseline_score if baseline_score else delta
                comparison[f"{metric}_delta"] = delta
                comparison[f"{metric}_relative_delta"] = relative_delta
                comparison[f"{metric}_predicted_direction"] = _direction(delta)

            baseline_latency, candidate_latency = latencies[baseline], latencies[candidate]
            observed_relative_delta = None
            observed_direction = None
            if baseline_latency is not None and candidate_latency is not None:
                observed_relative_delta = candidate_latency / baseline_latency - 1.0
                observed_direction = _direction(observed_relative_delta)
            comparison["observed_relative_delta"] = observed_relative_delta
            comparison["observed_direction"] = observed_direction
            for metric in METRICS:
                comparison[f"{metric}_direction_correct"] = (
                    comparison[f"{metric}_predicted_direction"] == observed_direction
                    if observed_direction is not None
                    else None
                )
            comparisons.append(comparison)

        cypress = by_arm["cypress"]
        auxiliary = _number(cypress["cypress_auxiliary_edges"], field="cypress_auxiliary_edges")
        relaxed = _number(cypress["cypress_relaxed_edges"], field="cypress_relaxed_edges")
        cypress_groups.append(
            {
                "case_id": case_id,
                "capacity": capacity,
                "device": device,
                "cypress_auxiliary_edges": auxiliary,
                "cypress_relaxed_edges": relaxed,
                "cypress_actual_alias_pairs": _number(
                    cypress["cypress_actual_alias_pairs"], field="cypress_actual_alias_pairs"
                ),
                "cypress_packing_attempts": _number(
                    cypress["cypress_packing_attempts"], field="cypress_packing_attempts"
                ),
                "cypress_relaxed_fraction": relaxed / auxiliary if auxiliary else 0.0,
                "cypress_vs_dsa_rp_relative_delta": (
                    latencies["cypress"] / latencies["dsa_rp_cg"] - 1.0
                    if latencies["cypress"] is not None and latencies["dsa_rp_cg"] is not None
                    else None
                ),
            }
        )

    canonical = json.dumps(predictions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "split": split,
        "frozen_before_device_timing": freeze_before_timing,
        "prediction_sha256": hashlib.sha256(canonical).hexdigest(),
        "modeled_cell_count": len(predictions),
        "excluded_cell_count": len(excluded),
        "summaries": {metric: _metric_summary(comparisons, metric) for metric in METRICS},
        "cypress_metrics": _cypress_summary(cypress_groups),
        "predictions": predictions,
        "comparisons": comparisons,
        "excluded": excluded,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--freeze-before-timing", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_rows(
            _read_rows(args.input),
            split=args.split,
            freeze_before_timing=args.freeze_before_timing,
        )
        result["input"] = {
            "path": str(args.input.resolve()),
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered)
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
