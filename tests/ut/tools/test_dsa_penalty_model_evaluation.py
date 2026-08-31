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
