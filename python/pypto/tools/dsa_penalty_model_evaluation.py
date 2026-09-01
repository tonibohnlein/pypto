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
SYNC_WEIGHT_COLUMNS = (
    "workload_id",
    "case_id",
    "capacity",
    "device",
    "weight_cycles",
    "analysis_status",
    "cypress_penalty_cycles",
    "dsa_rp_penalty_cycles",
    "cypress_latency_us",
    "dsa_rp_latency_us",
)


def _read_table(path: Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        missing = sorted(set(required_columns) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: input table is empty")
    return rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    return _read_table(path, REQUIRED_COLUMNS)


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


def _text(value: str, *, field: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _boolean(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be 'true' or 'false', got {value!r}")


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


def _observed_classification(
    device_effects: Mapping[str, float], minimum_device_effect: float
) -> tuple[str, int | None]:
    directions = {_direction(effect) for effect in device_effects.values()}
    if len(directions) != 1:
        return "DEVICE_CONFLICT", None
    direction = next(iter(directions))
    if direction == 0:
        return "EXACT_TIE", 0
    if len(device_effects) >= 2 and all(
        abs(effect) >= minimum_device_effect for effect in device_effects.values()
    ):
        return "MULTI_DEVICE_THRESHOLD_CLEARED", direction
    if all(abs(effect) < minimum_device_effect for effect in device_effects.values()):
        return "ALL_DEVICES_BELOW_THRESHOLD", direction
    return "MIXED_DEVICE_THRESHOLD", direction


def _weight_summary(rows: Sequence[Mapping[str, Any]], observed_class: str) -> dict[str, Any]:
    selected = [row for row in rows if row["observed_class"] == observed_class]
    strict = [row for row in selected if row["predicted_direction"] != 0]
    correct = sum(row["predicted_direction"] == row["observed_direction"] for row in strict)
    wrong = len(strict) - correct
    calibration_utility = correct - wrong
    return {
        "observed_cell_count": len(selected),
        "predicted_strict_count": len(strict),
        "predicted_silent_count": len(selected) - len(strict),
        "correct_count": correct,
        "wrong_count": wrong,
        "calibration_utility": calibration_utility,
        "strict_accuracy": correct / len(strict) if strict else None,
        "prediction_coverage": len(strict) / len(selected) if selected else None,
    }


def _best_weight_rows(
    rows_by_weight: Mapping[float, Sequence[Mapping[str, Any]]],
) -> tuple[list[float], dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, int], float, dict[str, Any]]] = []
    for weight, rows in sorted(rows_by_weight.items()):
        summary = _weight_summary(rows, "MULTI_DEVICE_THRESHOLD_CLEARED")
        accuracy = summary["strict_accuracy"]
        if accuracy is None:
            continue
        # One global utility is fixed before leave-one-workload-out calibration:
        # correct strict ordering = +1, wrong strict ordering = -1, silence = 0.
        # Coverage then breaks utility ties, so a selectively silent 1/1 model
        # cannot outrank a broadly correct model merely through higher accuracy.
        key = (
            int(summary["calibration_utility"]),
            int(summary["correct_count"]),
            int(summary["predicted_strict_count"]),
            -int(summary["wrong_count"]),
        )
        candidates.append((key, weight, summary))
    if not candidates:
        return [], None
    best_key = max(key for key, _, _ in candidates)
    best = [(weight, summary) for key, weight, summary in candidates if key == best_key]
    return [weight for weight, _ in best], best[0][1]


def _ranking_intervals(
    weights: Sequence[float], rows_by_weight: Mapping[float, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    signatures = {
        weight: tuple(
            (row["workload_id"], row["case_id"], row["capacity"], row["predicted_direction"])
            for row in sorted(
                rows_by_weight[weight],
                key=lambda item: (item["workload_id"], item["case_id"], item["capacity"]),
            )
        )
        for weight in weights
    }
    intervals: list[dict[str, Any]] = []
    begin = 0
    for index in range(1, len(weights) + 1):
        if index < len(weights) and signatures[weights[index]] == signatures[weights[begin]]:
            continue
        intervals.append(
            {
                "minimum_weight_cycles": weights[begin],
                "maximum_weight_cycles": weights[index - 1],
                "grid_point_count": index - begin,
                "strict_prediction_count": sum(
                    direction != 0 for *_, direction in signatures[weights[begin]]
                ),
            }
        )
        begin = index
    return intervals


def _stable_weight_calibration(
    folds: Sequence[Mapping[str, Any]], all_data_optimal_weights: Sequence[float]
) -> dict[str, Any]:
    """Intersect optimal weights across every LOOW fold and the full dataset."""

    incomplete = [str(fold.get("held_out_workload")) for fold in folds if fold.get("status") != "CALIBRATED"]
    if incomplete or not all_data_optimal_weights:
        return {
            "status": "INCOMPLETE",
            "stable_weight_intersection_cycles": [],
            "selected_weight_cycles": None,
            "incomplete_folds": incomplete,
        }
    optimal_sets = [set(float(weight) for weight in all_data_optimal_weights)]
    optimal_sets.extend(set(float(weight) for weight in fold["optimal_training_weights"]) for fold in folds)
    intersection = sorted(set.intersection(*optimal_sets))
    return {
        "status": "STABLE" if intersection else "NO_STABLE_WEIGHT",
        "stable_weight_intersection_cycles": intersection,
        # The smallest stable weight is the deterministic conservative choice:
        # it gives the same calibrated ordering with the least synchronization
        # latency assigned by the provisional model.
        "selected_weight_cycles": intersection[0] if intersection else None,
        "incomplete_folds": [],
    }


def _planner_admission_limits(requirements: Mapping[str, Any]) -> dict[str, int]:
    fields = (
        "minimum_directional_workloads",
        "minimum_dsa_rp_wins",
        "minimum_cypress_wins",
        "minimum_correct_directional_workloads",
        "maximum_wrong_directional_workloads",
        "minimum_calibration_utility",
    )
    limits: dict[str, int] = {}
    for field in fields:
        value = requirements.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be an explicitly declared non-negative integer")
        limits[field] = value
    return limits


def _planner_directional_workloads(
    selected_rows: Sequence[Mapping[str, Any]], failures: list[str]
) -> tuple[dict[str, int], list[str], list[str], dict[str, Any]]:
    directions_by_workload: dict[str, set[int]] = defaultdict(set)
    rows_by_workload: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if row.get("observed_class") != "MULTI_DEVICE_THRESHOLD_CLEARED":
            continue
        direction = row.get("observed_direction")
        workload = row.get("workload_id")
        if isinstance(workload, str) and isinstance(direction, int) and not isinstance(direction, bool):
            directions_by_workload[workload].add(direction)
            rows_by_workload[workload].append(row)
    ambiguous = sorted(
        workload for workload, directions in directions_by_workload.items() if len(directions) != 1
    )
    if ambiguous:
        failures.append("direction changes within workload: " + ", ".join(ambiguous))
    directional = {
        workload: next(iter(directions))
        for workload, directions in directions_by_workload.items()
        if len(directions) == 1
    }
    dsa_wins = sorted(workload for workload, direction in directional.items() if direction < 0)
    cypress_wins = sorted(workload for workload, direction in directional.items() if direction > 0)
    correct: list[str] = []
    wrong: list[str] = []
    silent: list[str] = []
    for workload, observed_direction in directional.items():
        predictions = [row.get("predicted_direction") for row in rows_by_workload[workload]]
        if not all(
            isinstance(prediction, int) and not isinstance(prediction, bool) and prediction in {-1, 0, 1}
            for prediction in predictions
        ):
            raise ValueError(f"{workload}: predicted directions must be -1, 0, or 1")
        if any(prediction not in {0, observed_direction} for prediction in predictions):
            wrong.append(workload)
        elif all(prediction == observed_direction for prediction in predictions):
            correct.append(workload)
        else:
            silent.append(workload)
    quality = {
        "correct_workloads": sorted(correct),
        "wrong_workloads": sorted(wrong),
        "silent_workloads": sorted(silent),
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "silent_count": len(silent),
        "calibration_utility": len(correct) - len(wrong),
    }
    return directional, dsa_wins, cypress_wins, quality


def _manifest_text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return _text(value, field=f"{context}.{field}")


def _evaluate_planner_representative(
    raw: Mapping[str, Any], row_index: Mapping[tuple[Any, Any, Any], Mapping[str, Any]]
) -> tuple[dict[str, Any], str | None]:
    name = _manifest_text(raw, "name", "representative")
    workload = _manifest_text(raw, "workload_id", name)
    case_id = _manifest_text(raw, "case_id", name)
    capacity = _manifest_text(raw, "capacity", name)
    expected = raw.get("expected_ordering")
    valid_expected = {"dsa_rp_faster": -1, "cypress_faster": 1, "tie": 0}
    if expected not in valid_expected:
        raise ValueError(f"{name}: expected_ordering must be one of {sorted(valid_expected)}")
    row = row_index.get((workload, case_id, capacity))
    if row is None:
        return (
            {"name": name, "status": "MISSING", "expected_ordering": expected},
            f"representative {name} is absent at the stable weight",
        )
    expected_direction = valid_expected[expected]
    observed_class = row.get("observed_class")
    observed_direction = row.get("observed_direction")
    predicted_direction = row.get("predicted_direction")
    observed_ok = (
        observed_class in {"EXACT_TIE", "ALL_DEVICES_BELOW_THRESHOLD"}
        if expected_direction == 0
        else observed_class == "MULTI_DEVICE_THRESHOLD_CLEARED" and observed_direction == expected_direction
    )
    status = "PASS" if observed_ok and predicted_direction == expected_direction else "FAIL"
    return (
        {
            "name": name,
            "status": status,
            "expected_ordering": expected,
            "observed_class": observed_class,
            "observed_direction": observed_direction,
            "predicted_direction": predicted_direction,
        },
        None if status == "PASS" else f"representative {name} does not satisfy {expected}",
    )


def _selected_planner_rows(
    evaluation: Mapping[str, Any], failures: list[str]
) -> tuple[float | None, list[Mapping[str, Any]]]:
    stable = evaluation.get("stable_weight_calibration")
    if not isinstance(stable, Mapping) or stable.get("status") != "STABLE":
        failures.append("no stable synchronization weight across all LOOW folds")
        return None, []
    selected_weight = stable.get("selected_weight_cycles")
    if (
        not isinstance(selected_weight, (int, float))
        or isinstance(selected_weight, bool)
        or not math.isfinite(float(selected_weight))
    ):
        failures.append("stable synchronization weight is not finite")
        return None, []
    rows = [
        row
        for row in evaluation.get("comparisons", [])
        if isinstance(row, Mapping) and row.get("weight_cycles") == selected_weight
    ]
    return float(selected_weight), rows


def _validate_planner_score_contract(selected_rows: Sequence[Mapping[str, Any]], failures: list[str]) -> None:
    non_total_contract = sorted(
        {
            str(row.get("workload_id"))
            for row in selected_rows
            if row.get("prediction_contract") != "total_modeled_latency_relative_effect_gate_v1"
        }
    )
    if non_total_contract:
        failures.append(
            "planner admission requires total modeled latency for workloads: " + ", ".join(non_total_contract)
        )
    non_static = sorted(
        {
            str(row.get("workload_id"))
            for row in selected_rows
            if row.get("planner_eligible_static") is not True
        }
    )
    if non_static:
        failures.append(
            "planner admission requires static planner-eligible scores for workloads: "
            + ", ".join(non_static)
        )


def _append_planner_minimum_failures(
    limits: Mapping[str, int],
    directional: Mapping[str, int],
    dsa_wins: Sequence[str],
    cypress_wins: Sequence[str],
    quality: Mapping[str, Any],
    failures: list[str],
) -> None:
    checks = (
        (
            len(directional),
            limits["minimum_directional_workloads"],
            "directional workloads",
        ),
        (len(dsa_wins), limits["minimum_dsa_rp_wins"], "DSA-RP wins"),
        (len(cypress_wins), limits["minimum_cypress_wins"], "Cypress wins"),
        (
            int(quality["correct_count"]),
            limits["minimum_correct_directional_workloads"],
            "correctly predicted directional workloads",
        ),
    )
    for actual, required, label in checks:
        if actual < required:
            failures.append(f"only {actual} {label}; require {required}")
    wrong_count = int(quality["wrong_count"])
    if wrong_count > limits["maximum_wrong_directional_workloads"]:
        failures.append(
            f"{wrong_count} wrongly predicted directional workloads; "
            f"allow at most {limits['maximum_wrong_directional_workloads']}"
        )
    utility = int(quality["calibration_utility"])
    if utility < limits["minimum_calibration_utility"]:
        failures.append(f"calibration utility {utility}; require {limits['minimum_calibration_utility']}")


def _planner_row_index(
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, Any, Any], Mapping[str, Any]]:
    row_index: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for row in selected_rows:
        key = (row.get("workload_id"), row.get("case_id"), row.get("capacity"))
        if key in row_index:
            raise ValueError(f"duplicate planner-admission comparison row for {key}")
        row_index[key] = row
    return row_index


def _planner_representative_results(
    raw_representatives: Sequence[Any],
    row_index: Mapping[tuple[Any, Any, Any], Mapping[str, Any]],
    failures: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw in raw_representatives:
        if not isinstance(raw, Mapping):
            raise ValueError("planner-admission representative must be an object")
        result, failure = _evaluate_planner_representative(raw, row_index)
        results.append(result)
        if failure is not None:
            failures.append(failure)
    return results


def evaluate_planner_admission(
    evaluation: Mapping[str, Any], requirements: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the predeclared scientific gate before implementing a planner.

    The gate intentionally uses only the stable global weight and treats a
    device result as a null only when every device remains below threshold.
    Representative cases must agree with both the measured class and the model
    prediction, so a false-confident Gumbel-style result blocks admission even
    when aggregate accuracy is high.
    """

    if requirements.get("schema_version") != 1 or requirements.get("contract") != (
        "complete_placement_planner_admission_v1"
    ):
        raise ValueError(
            "planner-admission requirements must use complete_placement_planner_admission_v1 schema_version=1"
        )
    limits = _planner_admission_limits(requirements)
    raw_representatives = requirements.get("representatives")
    if not isinstance(raw_representatives, list) or not raw_representatives:
        raise ValueError("planner-admission requirements need representative cases")

    failures: list[str] = []
    selected_weight, selected_rows = _selected_planner_rows(evaluation, failures)
    _validate_planner_score_contract(selected_rows, failures)

    directional, dsa_wins, cypress_wins, quality = _planner_directional_workloads(selected_rows, failures)
    _append_planner_minimum_failures(limits, directional, dsa_wins, cypress_wins, quality, failures)
    representative_results = _planner_representative_results(
        raw_representatives, _planner_row_index(selected_rows), failures
    )

    return {
        "schema_version": 1,
        "contract": "complete_placement_planner_admission_v1",
        "status": "PASS" if not failures else "FAIL",
        "selected_weight_cycles": selected_weight,
        "requirements": limits,
        "directional_workload_count": len(directional),
        "directional_workloads": dict(sorted(directional.items())),
        "dsa_rp_win_workloads": dsa_wins,
        "cypress_win_workloads": cypress_wins,
        "model_quality": quality,
        "representatives": representative_results,
        "failures": failures,
    }


def _planner_eligibility_for_cell(
    cell_rows: Sequence[Mapping[str, str]], context: str
) -> tuple[dict[str, bool] | None, bool | None]:
    fields = ("cypress_planner_eligible_static", "dsa_rp_planner_eligible_static")
    presence = [tuple(row.get(field, "") != "" for field in fields) for row in cell_rows]
    if not any(any(row_presence) for row_presence in presence):
        return None, None
    if not all(all(row_presence) for row_presence in presence):
        raise ValueError(f"{context}: planner eligibility is incomplete across devices")

    def parse(row: Mapping[str, str]) -> dict[str, bool]:
        return {
            "cypress": _boolean(
                row["cypress_planner_eligible_static"],
                field="cypress_planner_eligible_static",
            ),
            "dsa_rp_cg": _boolean(
                row["dsa_rp_planner_eligible_static"],
                field="dsa_rp_planner_eligible_static",
            ),
        }

    eligibility = parse(cell_rows[0])
    if any(parse(row) != eligibility for row in cell_rows[1:]):
        raise ValueError(f"{context}: planner eligibility varies by device")
    return eligibility, all(eligibility.values())


def _modeled_weight_comparison(
    workload: str,
    case_id: str,
    capacity: str,
    weight: float,
    cell_rows: Sequence[Mapping[str, str]],
    observed_by_device: dict[tuple[str, str, str, str], tuple[float, float]],
    minimum_device_effect: float,
    minimum_model_effect: float,
) -> tuple[dict[str, Any], set[str]]:
    effects: dict[str, float] = {}
    for row in cell_rows:
        device = row["device"]
        cypress_latency = _number(row["cypress_latency_us"], field="cypress_latency_us")
        dsa_latency = _number(row["dsa_rp_latency_us"], field="dsa_rp_latency_us")
        assert cypress_latency is not None and dsa_latency is not None
        if cypress_latency <= 0 or dsa_latency <= 0:
            raise ValueError("device latencies must be positive")
        effect = dsa_latency / cypress_latency - 1.0
        observation_key = (workload, case_id, capacity, device)
        latency_pair = (cypress_latency, dsa_latency)
        previous = observed_by_device.setdefault(observation_key, latency_pair)
        if previous != latency_pair:
            raise ValueError(f"{observation_key}: exact device latencies change across weights")
        if device in effects:
            raise ValueError(f"{observation_key}: duplicate device row at weight {weight}")
        effects[device] = effect

    first = cell_rows[0]
    cypress_score = _number(first["cypress_penalty_cycles"], field="cypress_penalty_cycles")
    dsa_score = _number(first["dsa_rp_penalty_cycles"], field="dsa_rp_penalty_cycles")
    assert cypress_score is not None and dsa_score is not None
    if cypress_score < 0 or dsa_score < 0:
        raise ValueError("modeled penalty cycles must be non-negative")
    for row in cell_rows[1:]:
        other_cypress = _number(row["cypress_penalty_cycles"], field="cypress_penalty_cycles")
        other_dsa = _number(row["dsa_rp_penalty_cycles"], field="dsa_rp_penalty_cycles")
        if other_cypress != cypress_score or other_dsa != dsa_score:
            raise ValueError(f"{workload}/{case_id}/{capacity}/{weight}: model score varies by device")

    context = f"{workload}/{case_id}/{capacity}/{weight}"
    planner_eligibility_by_arm, planner_eligible_static = _planner_eligibility_for_cell(cell_rows, context)

    modeled_latency_fields = ("cypress_modeled_latency_cycles", "dsa_rp_modeled_latency_cycles")
    modeled_latency_presence = [
        tuple(row.get(field, "") != "" for field in modeled_latency_fields) for row in cell_rows
    ]
    any_modeled_latency = any(any(presence) for presence in modeled_latency_presence)
    complete_modeled_latency = all(all(presence) for presence in modeled_latency_presence)
    if any_modeled_latency and not complete_modeled_latency:
        raise ValueError(
            f"{workload}/{case_id}/{capacity}/{weight}: modeled latency is incomplete across devices"
        )
    if complete_modeled_latency:
        cypress_modeled_latency = _number(
            first["cypress_modeled_latency_cycles"], field="cypress_modeled_latency_cycles"
        )
        dsa_modeled_latency = _number(
            first["dsa_rp_modeled_latency_cycles"], field="dsa_rp_modeled_latency_cycles"
        )
        assert cypress_modeled_latency is not None and dsa_modeled_latency is not None
        if cypress_modeled_latency <= 0 or dsa_modeled_latency <= 0:
            raise ValueError("modeled total latency cycles must be positive")
        for row in cell_rows[1:]:
            if (
                _number(
                    row["cypress_modeled_latency_cycles"],
                    field="cypress_modeled_latency_cycles",
                )
                != cypress_modeled_latency
                or _number(
                    row["dsa_rp_modeled_latency_cycles"],
                    field="dsa_rp_modeled_latency_cycles",
                )
                != dsa_modeled_latency
            ):
                raise ValueError(
                    f"{workload}/{case_id}/{capacity}/{weight}: modeled latency varies by device"
                )
        modeled_effect = dsa_modeled_latency / cypress_modeled_latency - 1.0
        raw_predicted_direction = _direction(modeled_effect)
        predicted_direction = raw_predicted_direction if abs(modeled_effect) >= minimum_model_effect else 0
        prediction_contract = "total_modeled_latency_relative_effect_gate_v1"
    else:
        cypress_modeled_latency = None
        dsa_modeled_latency = None
        modeled_effect = None
        raw_predicted_direction = _direction(dsa_score - cypress_score)
        predicted_direction = raw_predicted_direction
        prediction_contract = "legacy_penalty_cycle_strict_order_v1"
    observed_class, observed_direction = _observed_classification(effects, minimum_device_effect)
    return (
        {
            "workload_id": workload,
            "case_id": case_id,
            "capacity": capacity,
            "weight_cycles": weight,
            "cypress_penalty_cycles": cypress_score,
            "dsa_rp_penalty_cycles": dsa_score,
            "cypress_modeled_latency_cycles": cypress_modeled_latency,
            "dsa_rp_modeled_latency_cycles": dsa_modeled_latency,
            "modeled_relative_effect": modeled_effect,
            "raw_predicted_direction": raw_predicted_direction,
            "predicted_direction": predicted_direction,
            "prediction_contract": prediction_contract,
            "planner_eligible_static": planner_eligible_static,
            "planner_eligibility_by_arm": planner_eligibility_by_arm,
            "observed_class": observed_class,
            "observed_direction": observed_direction,
            "device_effects": dict(sorted(effects.items())),
        },
        set(effects),
    )


def _grid_cell_identity(
    case_key: tuple[str, str, str],
    weight: float,
    cell_rows: Sequence[Mapping[str, str]],
    required_device_count: int,
) -> tuple[set[str], set[str]]:
    devices = [row["device"] for row in cell_rows]
    if len(devices) != len(set(devices)):
        raise ValueError(f"{case_key}: duplicate device row at weight {weight}")
    if len(devices) != required_device_count:
        raise ValueError(
            f"{case_key}: expected exactly {required_device_count} devices at weight {weight}, "
            f"got {len(devices)}"
        )
    statuses = {row["analysis_status"] for row in cell_rows}
    if len(statuses) != 1:
        raise ValueError(f"{case_key}: analysis status varies by device at weight {weight}")
    return set(devices), statuses


def evaluate_sync_weight_grid(
    rows: Sequence[Mapping[str, str]],
    *,
    minimum_device_effect: float = 0.02,
    minimum_model_effect: float = 0.02,
    required_device_count: int = 2,
) -> dict[str, Any]:
    """Evaluate one global synchronization weight with leave-one-workload-out calibration.

    The input contains Cypress and DSA-RP scores for every weight and repeats the
    same frozen device observations at each weight.  A device result is treated
    as threshold-cleared only when the required devices agree in direction and
    every magnitude reaches ``minimum_device_effect``.  This is a deterministic
    effect-size gate, not a confidence-interval claim.  Calibration never falls
    back to the below-threshold rows.
    """
    if not math.isfinite(minimum_device_effect) or minimum_device_effect < 0:
        raise ValueError("minimum_device_effect must be finite and non-negative")
    if not math.isfinite(minimum_model_effect) or minimum_model_effect < 0:
        raise ValueError("minimum_model_effect must be finite and non-negative")
    if not isinstance(required_device_count, int) or required_device_count < 2:
        raise ValueError("required_device_count must be an integer of at least two")

    grouped: dict[tuple[str, str, str, float], list[Mapping[str, str]]] = defaultdict(list)
    weights: set[float] = set()
    for row in rows:
        workload = _text(row["workload_id"], field="workload_id")
        case_id = _text(row["case_id"], field="case_id")
        capacity = _text(row["capacity"], field="capacity")
        device = _text(row["device"], field="device")
        status = _text(row["analysis_status"], field="analysis_status")
        weight = _number(row["weight_cycles"], field="weight_cycles")
        assert weight is not None
        if weight <= 0:
            raise ValueError("weight_cycles must be positive")
        weights.add(weight)
        normalized = dict(row)
        normalized.update(
            workload_id=workload,
            case_id=case_id,
            capacity=capacity,
            device=device,
            analysis_status=status,
        )
        grouped[(workload, case_id, capacity, weight)].append(normalized)
    ordered_weights = sorted(weights)

    comparisons: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    observed_by_device: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    expected_weights: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    statuses_by_case: dict[tuple[str, str, str], set[str]] = {}
    devices_by_case: dict[tuple[str, str, str], set[str]] = {}
    for (workload, case_id, capacity, weight), cell_rows in sorted(grouped.items()):
        case_key = (workload, case_id, capacity)
        expected_weights[case_key].add(weight)
        device_set, statuses = _grid_cell_identity(case_key, weight, cell_rows, required_device_count)
        previous_devices = devices_by_case.setdefault(case_key, device_set)
        if device_set != previous_devices:
            raise ValueError(f"{case_key}: device set changes across weights")
        previous_statuses = statuses_by_case.setdefault(case_key, statuses)
        if statuses != previous_statuses:
            raise ValueError(f"{case_key}: analysis status changes across weights")
        if statuses != {"MODELED"}:
            excluded.append(
                {
                    "workload_id": workload,
                    "case_id": case_id,
                    "capacity": capacity,
                    "weight_cycles": weight,
                    "reason": ",".join(sorted(statuses)),
                }
            )
            continue
        comparison, modeled_devices = _modeled_weight_comparison(
            workload,
            case_id,
            capacity,
            weight,
            cell_rows,
            observed_by_device,
            minimum_device_effect,
            minimum_model_effect,
        )
        if modeled_devices != device_set:
            raise ValueError(f"{case_key}: modeled device set does not match the declared cell")
        comparisons.append(comparison)

    for key, actual_weights in expected_weights.items():
        if actual_weights != weights:
            raise ValueError(f"{key}: incomplete weight grid")

    rows_by_weight = {
        weight: [row for row in comparisons if row["weight_cycles"] == weight] for weight in ordered_weights
    }
    sensitivity = [
        {
            "weight_cycles": weight,
            "threshold_cleared": _weight_summary(rows_by_weight[weight], "MULTI_DEVICE_THRESHOLD_CLEARED"),
            "all_devices_below_threshold": _weight_summary(
                rows_by_weight[weight], "ALL_DEVICES_BELOW_THRESHOLD"
            ),
            "mixed_device_threshold": _weight_summary(rows_by_weight[weight], "MIXED_DEVICE_THRESHOLD"),
        }
        for weight in ordered_weights
    ]

    workloads = sorted({row["workload_id"] for row in comparisons})
    folds: list[dict[str, Any]] = []
    for held_out in workloads:
        training = {
            weight: [row for row in weight_rows if row["workload_id"] != held_out]
            for weight, weight_rows in rows_by_weight.items()
        }
        best_weights, training_summary = _best_weight_rows(training)
        if not best_weights:
            folds.append(
                {
                    "held_out_workload": held_out,
                    "status": "INSUFFICIENT_THRESHOLD_CLEARED_TRAINING_COMPARISONS",
                }
            )
            continue
        selected_weight = best_weights[len(best_weights) // 2]
        held_out_rows = [row for row in rows_by_weight[selected_weight] if row["workload_id"] == held_out]
        folds.append(
            {
                "held_out_workload": held_out,
                "status": "CALIBRATED",
                "optimal_training_weights": best_weights,
                "selected_weight_cycles": selected_weight,
                "training": training_summary,
                "held_out_threshold_cleared": _weight_summary(
                    held_out_rows, "MULTI_DEVICE_THRESHOLD_CLEARED"
                ),
            }
        )

    best_weights, all_data_summary = _best_weight_rows(rows_by_weight)
    stable_calibration = _stable_weight_calibration(folds, best_weights)
    intervals = _ranking_intervals(ordered_weights, rows_by_weight) if comparisons else []
    eligible_cases = {(row["workload_id"], row["case_id"], row["capacity"]) for row in comparisons}
    return {
        "schema_version": 1,
        "model": "complete_placement_dag_global_sync_weight_grid_v1",
        "minimum_device_effect": minimum_device_effect,
        "minimum_model_effect": minimum_model_effect,
        "required_device_count": required_device_count,
        "calibration_objective": {
            "correct_strict_ordering": 1,
            "wrong_strict_ordering": -1,
            "silent_prediction": 0,
            "tie_break": "more correct, then more strict predictions, then fewer wrong",
        },
        "weights_cycles": ordered_weights,
        "eligible_workload_count": len(workloads),
        "declared_case_count": len(expected_weights),
        "eligible_case_count": len(eligible_cases),
        "excluded_case_count": len(expected_weights) - len(eligible_cases),
        "excluded_grid_row_count": len(excluded),
        "sensitivity": sensitivity,
        "ranking_intervals": intervals,
        "rankings_stable_across_full_grid": len(intervals) == 1,
        "leave_one_workload_out": folds,
        "all_development_data_optimal_weights": best_weights,
        "all_development_data_summary": all_data_summary,
        "stable_weight_calibration": stable_calibration,
        "comparisons": comparisons,
        "excluded": excluded,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--freeze-before-timing", action="store_true")
    parser.add_argument(
        "--sync-weight-grid",
        action="store_true",
        help="evaluate Cypress-versus-DSA-RP rows over one global synchronization-weight grid",
    )
    parser.add_argument(
        "--minimum-device-effect",
        type=float,
        default=0.02,
        help="per-device relative-effect floor for a threshold-cleared ordering (default: 0.02)",
    )
    parser.add_argument(
        "--minimum-model-effect",
        type=float,
        default=0.02,
        help="relative modeled-latency floor for a strict prediction (default: 0.02)",
    )
    parser.add_argument(
        "--required-device-count",
        type=int,
        default=2,
        help="exact number of devices required in every grid cell (default: 2)",
    )
    parser.add_argument(
        "--planner-admission",
        type=Path,
        help="predeclared representative and corpus gate for experimental planner admission",
    )
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.sync_weight_grid:
            if args.freeze_before_timing:
                raise ValueError("a synchronization-weight grid requires observed development timing")
            result = evaluate_sync_weight_grid(
                _read_table(args.input, SYNC_WEIGHT_COLUMNS),
                minimum_device_effect=args.minimum_device_effect,
                minimum_model_effect=args.minimum_model_effect,
                required_device_count=args.required_device_count,
            )
            if args.planner_admission is not None:
                requirements = json.loads(args.planner_admission.read_text())
                if not isinstance(requirements, Mapping):
                    raise ValueError("planner-admission requirements must be an object")
                result["planner_admission"] = evaluate_planner_admission(result, requirements)
                result["planner_admission_input"] = {
                    "path": str(args.planner_admission.resolve()),
                    "sha256": hashlib.sha256(args.planner_admission.read_bytes()).hexdigest(),
                }
            result["split"] = args.split
        else:
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
