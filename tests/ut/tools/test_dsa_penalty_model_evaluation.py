# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for development and held-out DSA penalty-model evaluation."""

import csv
import json

import pytest
from pypto.tools import dsa_penalty_model_evaluation as evaluation


def _cell(
    case_id: str,
    *,
    latencies: dict[str, str] | None = None,
    status: str = "MODELED",
) -> list[dict[str, str]]:
    unit = {"geometry_ff": "2", "geometry_cg": "2", "cypress": "0", "dsa_rp_cg": "1"}
    critical_path = {
        "geometry_ff": "20",
        "geometry_cg": "20",
        "cypress": "10",
        "dsa_rp_cg": "5",
    }
    complete_critical_path = {
        "geometry_ff": "18",
        "geometry_cg": "18",
        "cypress": "9",
        "dsa_rp_cg": "4",
    }
    physical_groups = {"geometry_ff": "2", "geometry_cg": "2", "cypress": "0", "dsa_rp_cg": "1"}
    sync_edges = {"geometry_ff": "2", "geometry_cg": "2", "cypress": "0", "dsa_rp_cg": "1"}
    sync_endpoints = {"geometry_ff": "4", "geometry_cg": "4", "cypress": "0", "dsa_rp_cg": "2"}
    latencies = latencies or {arm: "" for arm in evaluation.ARMS}
    return [
        {
            "case_id": case_id,
            "capacity": "half",
            "device": "dev0",
            "arm": arm,
            "analysis_status": status,
            "unit_realized_cost": unit[arm],
            "canonical_physical_reuse_group_count": physical_groups[arm],
            "unique_induced_sync_edge_count": sync_edges[arm],
            "estimated_sync_endpoint_executions": sync_endpoints[arm],
            "critical_path_realized_cost_cycles": critical_path[arm],
            "complete_placement_critical_path_cycles": complete_critical_path[arm],
            "latency_us": latencies[arm],
            "cypress_auxiliary_edges": "100" if arm == "cypress" else "",
            "cypress_relaxed_edges": "20" if arm == "cypress" else "",
            "cypress_actual_alias_pairs": "8" if arm == "cypress" else "",
            "cypress_packing_attempts": "21" if arm == "cypress" else "",
        }
        for arm in evaluation.ARMS
    ]


def test_development_evaluation_compares_unit_and_critical_path_ordering():
    rows = _cell(
        "case-a",
        latencies={"geometry_ff": "10", "geometry_cg": "10", "cypress": "8", "dsa_rp_cg": "7"},
    )

    result = evaluation.evaluate_rows(rows, split="development", freeze_before_timing=False)

    assert result["modeled_cell_count"] == 1
    rp_vs_cypress = next(
        row
        for row in result["comparisons"]
        if row["baseline_arm"] == "cypress" and row["candidate_arm"] == "dsa_rp_cg"
    )
    assert rp_vs_cypress["observed_direction"] == -1
    assert rp_vs_cypress["unit_realized_cost_predicted_direction"] == 1
    assert rp_vs_cypress["unit_realized_cost_direction_correct"] is False
    assert rp_vs_cypress["critical_path_realized_cost_cycles_predicted_direction"] == -1
    assert rp_vs_cypress["critical_path_realized_cost_cycles_direction_correct"] is True


def test_holdout_freeze_is_content_addressed_and_contains_no_latency():
    result = evaluation.evaluate_rows(_cell("holdout"), split="holdout", freeze_before_timing=True)

    assert result["frozen_before_device_timing"] is True
    assert len(result["prediction_sha256"]) == 64
    assert all(row["observed_relative_delta"] is None for row in result["comparisons"])


def test_holdout_freeze_rejects_observed_latency():
    rows = _cell(
        "leaked",
        latencies={"geometry_ff": "10", "geometry_cg": "10", "cypress": "8", "dsa_rp_cg": "7"},
    )

    with pytest.raises(ValueError, match="must not contain device latency"):
        evaluation.evaluate_rows(rows, split="holdout", freeze_before_timing=True)


def test_model_ineligible_cell_is_excluded_without_affecting_measurability():
    result = evaluation.evaluate_rows(
        _cell("dynamic", status="DYNAMIC_CONTROL_FLOW_EXCLUDED"),
        split="development",
        freeze_before_timing=False,
    )

    assert result["modeled_cell_count"] == 0
    assert result["excluded"] == [
        {
            "case_id": "dynamic",
            "capacity": "half",
            "device": "dev0",
            "reason": "DYNAMIC_CONTROL_FLOW_EXCLUDED",
        }
    ]


def test_duplicate_arm_row_is_rejected():
    rows = _cell("duplicate")
    rows[-1] = dict(rows[0])

    with pytest.raises(ValueError, match="expected exactly four arms"):
        evaluation.evaluate_rows(rows, split="development", freeze_before_timing=False)


def _weight_grid_cell(
    workload: str,
    *,
    observed_effect: float,
    scores: dict[float, tuple[float, float]],
    modeled_latencies: dict[float, tuple[float, float]] | None = None,
    status: str = "MODELED",
) -> list[dict[str, str]]:
    rows = []
    for weight, (cypress_score, dsa_score) in scores.items():
        for device, cypress_latency in (("dev0", 10.0), ("dev1", 11.0)):
            rows.append(
                {
                    "workload_id": workload,
                    "case_id": f"{workload}-case",
                    "capacity": "half",
                    "device": device,
                    "weight_cycles": str(weight),
                    "analysis_status": status,
                    "cypress_penalty_cycles": str(cypress_score),
                    "dsa_rp_penalty_cycles": str(dsa_score),
                    "cypress_latency_us": str(cypress_latency),
                    "dsa_rp_latency_us": str(cypress_latency * (1.0 + observed_effect)),
                    **(
                        {
                            "cypress_modeled_latency_cycles": str(modeled_latencies[weight][0]),
                            "dsa_rp_modeled_latency_cycles": str(modeled_latencies[weight][1]),
                        }
                        if modeled_latencies is not None
                        else {}
                    ),
                }
            )
    return rows


def test_sync_weight_grid_uses_one_training_weight_per_leave_one_workload_out_fold():
    scores = {16.0: (0.0, 0.0), 64.0: (10.0, 0.0), 160.0: (0.0, 10.0)}
    rows = [
        row
        for workload in ("workload-a", "workload-b", "workload-c")
        for row in _weight_grid_cell(workload, observed_effect=-0.05, scores=scores)
    ]

    result = evaluation.evaluate_sync_weight_grid(rows)

    assert result["eligible_workload_count"] == 3
    assert result["rankings_stable_across_full_grid"] is False
    assert result["all_development_data_optimal_weights"] == [64.0]
    assert all(fold["status"] == "CALIBRATED" for fold in result["leave_one_workload_out"])
    assert {fold["selected_weight_cycles"] for fold in result["leave_one_workload_out"]} == {64.0}
    assert all(
        fold["held_out_threshold_cleared"]["strict_accuracy"] == 1.0
        for fold in result["leave_one_workload_out"]
    )


def test_sync_weight_grid_does_not_calibrate_on_below_threshold_effects():
    scores = {16.0: (1.0, 0.0), 64.0: (2.0, 0.0)}
    rows = [
        row
        for workload in ("workload-a", "workload-b")
        for row in _weight_grid_cell(workload, observed_effect=-0.01, scores=scores)
    ]

    result = evaluation.evaluate_sync_weight_grid(rows)

    assert result["all_development_data_optimal_weights"] == []
    assert all(
        fold["status"] == "INSUFFICIENT_THRESHOLD_CLEARED_TRAINING_COMPARISONS"
        for fold in result["leave_one_workload_out"]
    )
    assert all(row["sign_consistent_below_threshold"]["correct_count"] == 2 for row in result["sensitivity"])


def test_sync_weight_grid_treats_subthreshold_modeled_latency_effect_as_tie():
    rows = _weight_grid_cell(
        "workload-a",
        observed_effect=0.001,
        scores={64.0: (1000.0, 0.0)},
        modeled_latencies={64.0: (400000.0, 400041.0)},
    )

    result = evaluation.evaluate_sync_weight_grid(rows)
    comparison = result["comparisons"][0]

    assert comparison["prediction_contract"] == "total_modeled_latency_relative_effect_gate_v1"
    assert comparison["raw_predicted_direction"] == 1
    assert comparison["predicted_direction"] == 0
    assert comparison["modeled_relative_effect"] == pytest.approx(41.0 / 400000.0)


def test_sync_weight_grid_rejects_partial_modeled_latency_pair():
    rows = _weight_grid_cell(
        "workload-a",
        observed_effect=0.001,
        scores={64.0: (1000.0, 0.0)},
    )
    for row in rows:
        row["cypress_modeled_latency_cycles"] = "400000"

    with pytest.raises(ValueError, match="modeled latency is incomplete"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_rejects_exact_latency_drift_even_when_effect_is_unchanged():
    rows = _weight_grid_cell(
        "workload-a",
        observed_effect=-0.05,
        scores={16.0: (1.0, 0.0), 64.0: (2.0, 0.0)},
    )
    rows[-1]["cypress_latency_us"] = "22.0"
    rows[-1]["dsa_rp_latency_us"] = str(22.0 * 0.95)

    with pytest.raises(ValueError, match="exact device latencies change across weights"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_rejects_device_set_drift_across_weights():
    rows = _weight_grid_cell(
        "workload-a",
        observed_effect=-0.05,
        scores={16.0: (1.0, 0.0), 64.0: (2.0, 0.0)},
    )
    rows[-1]["device"] = "dev2"

    with pytest.raises(ValueError, match="device set changes across weights"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_rejects_excluded_case_device_set_drift():
    rows = _weight_grid_cell(
        "workload-a",
        observed_effect=0.0,
        scores={16.0: (0.0, 0.0), 64.0: (0.0, 0.0)},
        status="MODEL_INELIGIBLE_BRANCH_JOIN_V1",
    )
    rows[-1]["device"] = "dev2"

    with pytest.raises(ValueError, match="device set changes across weights"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_rejects_analysis_status_drift_across_weights():
    rows = _weight_grid_cell(
        "workload-a",
        observed_effect=-0.05,
        scores={16.0: (1.0, 0.0), 64.0: (2.0, 0.0)},
    )
    for row in rows:
        if row["weight_cycles"] == "64.0":
            row["analysis_status"] = "MODEL_INELIGIBLE_BRANCH_JOIN_V1"

    with pytest.raises(ValueError, match="analysis status changes across weights"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_reports_ineligible_case_coverage():
    modeled = _weight_grid_cell(
        "workload-a",
        observed_effect=-0.05,
        scores={16.0: (1.0, 0.0), 64.0: (2.0, 0.0)},
    )
    ineligible = _weight_grid_cell(
        "workload-b",
        observed_effect=0.0,
        scores={16.0: (0.0, 0.0), 64.0: (0.0, 0.0)},
        status="MODEL_INELIGIBLE_BRANCH_JOIN_V1",
    )

    result = evaluation.evaluate_sync_weight_grid([*modeled, *ineligible])

    assert result["declared_case_count"] == 2
    assert result["eligible_case_count"] == 1
    assert result["excluded_case_count"] == 1
    assert result["excluded_grid_row_count"] == 2


def test_sync_weight_grid_global_utility_does_not_reward_selective_silence():
    observed = "MULTI_DEVICE_THRESHOLD_CLEARED"
    sparse = [
        {"observed_class": observed, "predicted_direction": -1, "observed_direction": -1},
        *[{"observed_class": observed, "predicted_direction": 0, "observed_direction": -1} for _ in range(3)],
    ]
    broad = [
        *[
            {"observed_class": observed, "predicted_direction": -1, "observed_direction": -1}
            for _ in range(3)
        ],
        {"observed_class": observed, "predicted_direction": 1, "observed_direction": -1},
    ]

    weights, summary = evaluation._best_weight_rows({16.0: sparse, 64.0: broad})

    assert weights == [64.0]
    assert summary is not None
    assert summary["calibration_utility"] == 2


@pytest.mark.parametrize("field", ["workload_id", "case_id", "capacity", "device", "analysis_status"])
def test_sync_weight_grid_rejects_blank_identifiers(field):
    rows = _weight_grid_cell("workload-a", observed_effect=-0.05, scores={16.0: (1.0, 0.0)})
    rows[0][field] = "  "

    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_rejects_negative_model_score():
    rows = _weight_grid_cell("workload-a", observed_effect=-0.05, scores={16.0: (-1.0, 0.0)})

    with pytest.raises(ValueError, match="modeled penalty cycles must be non-negative"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_sync_weight_grid_requires_exactly_two_devices_by_default():
    rows = _weight_grid_cell("workload-a", observed_effect=-0.05, scores={16.0: (1.0, 0.0)})
    rows.pop()

    with pytest.raises(ValueError, match="expected exactly 2 devices"):
        evaluation.evaluate_sync_weight_grid(rows)


def test_cli_writes_input_hash(tmp_path):
    source = tmp_path / "scores.tsv"
    output = tmp_path / "evaluation.json"
    rows = _cell("holdout")
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=evaluation.REQUIRED_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    assert (
        evaluation.main([str(source), "--split", "holdout", "--freeze-before-timing", "-o", str(output)]) == 0
    )
    result = json.loads(output.read_text())
    assert result["input"]["path"] == str(source.resolve())
    assert len(result["input"]["sha256"]) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
