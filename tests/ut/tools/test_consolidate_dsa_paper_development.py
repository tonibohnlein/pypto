# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Accounting, timing-unit and selection controls for retrospective paper tables."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "consolidate_dsa_paper_development",
    Path(__file__).resolve().parents[2] / "tools/consolidate_dsa_paper_development.py",
)
assert _SPEC and _SPEC.loader
model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model)


def _row(name: str = "native", /, **updates):
    return {
        "script": "parent.py",
        "argv_json": "[]",
        "artifact_id": name,
        "opportunity": True,
        "unit_gap": 5,
        "relation_disagreement": 7,
        "capacity_bytes": 100,
        **updates,
    }


def test_capacity_selection_counts_driver_not_child_and_maximizes_gap():
    first = _row("child_a/native", target="a")
    second = _row("child_b/tight", target="b", unit_gap=3, capacity_bytes=50)
    selected, audit = model.choose_primary([second, first])
    assert selected == [first]
    assert [row["role"] for row in audit] == ["PRIMARY", "SENSITIVITY"]


def test_capacity_tiebreak_and_null_controls_are_retained():
    rows = [_row("native", opportunity=False), _row("tight", opportunity=False, capacity_bytes=50)]
    primary, audit = model.choose_primary(rows)
    assert primary[0]["artifact_id"] == "tight"
    assert all(row["selection"] == "CONTROL" for row in audit)


def test_distinct_arguments_are_distinct_workloads():
    selected, _ = model.choose_primary([_row(), _row("decode", argv_json='["--mode","decode"]')])
    assert len(selected) == 2


@pytest.mark.parametrize("field", ["latency_us", "median_us", "samples_us"])
def test_capacity_selector_refuses_timing_fields(field):
    with pytest.raises(ValueError, match="latency-bearing"):
        model.choose_primary([_row(**{field: 3})])


def _record():
    return {
        "maps": {"geometry_ff": {"map_digest": "same"}, "geometry_cg": {"map_digest": "same"}},
        "launches": [
            {
                "device": 0,
                "rep": rep,
                "map_digest": "same",
                "samples_us": [1.0, 2.0],
                "status": "PASS",
                "replay_provenance": "PASS",
            }
            for rep in range(3)
        ],
    }


def test_identical_maps_share_measurements_and_are_physical_nulls():
    record = _record()
    first = model.launch_samples(record, "geometry_ff", 0)
    second = model.launch_samples(record, "geometry_cg", 0)
    assert model.contrast(first, second, identical=True)["latency_change_pct"] == 0
    with pytest.raises(ValueError, match="different shared"):
        model.contrast(first, [[2.0]] * 3, identical=True)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "zero", "unverified"])
def test_timing_integrity_controls(mutation):
    record = _record()
    if mutation == "missing":
        record["launches"].pop()
    elif mutation == "duplicate":
        record["launches"].append(dict(record["launches"][0]))
    elif mutation == "zero":
        record["launches"][0]["samples_us"] = [0.0]
    else:
        record["launches"][0]["replay_provenance"] = "NOT_CHECKED"
    with pytest.raises(ValueError):
        model.launch_samples(record, "geometry_ff", 0)


def test_direction_requires_both_devices_and_keeps_misses_separate():
    win = {"latency_change_pct": -3, "ci_low_pct": -4, "ci_high_pct": -2}
    loss = {"latency_change_pct": 3, "ci_low_pct": 2, "ci_high_pct": 4}
    assert model.classify_device([win, win]) == "DSA_RP_FASTER"
    assert model.classify_device([win, loss]) == "UNCONFIRMED_OR_DEVICE_DISAGREEMENT"
    assert model.assessment("TIE", "DSA_RP_FASTER") == "MISSED_DIRECTION"
    assert model.assessment("CYPRESS_FASTER", "DSA_RP_FASTER") == "WRONG"
    assert model.assessment("DSA_RP_FASTER", "SMALL_OR_NULL") == "STRICT_ON_SMALL_OR_NULL"


def test_matched_summary_uses_same_rows_for_all_predictors():
    records = [
        {
            "artifact_id": case,
            "role": "PRIMARY",
            "predictor": predictor,
            "sync_weight_cycles": 8 if predictor == "dag" else 0,
            "predicted": "UNAVAILABLE" if predictor == "dag" and case == "parent" else "TIE",
            "assessment": "UNAVAILABLE" if predictor == "dag" and case == "parent" else "NULL_TIE",
        }
        for case in ("single", "parent")
        for predictor in ("unit_cost", "dag")
    ]
    matched = model.summarize_predictors(records, matched=True)
    assert len(matched) == 2
    assert all(row["NULL_TIE"] == 1 and row["UNAVAILABLE"] == 0 for row in matched)


def test_intersections_use_half_open_ranges_and_pool_identity():
    document = {"problem": {"buffers": [{"id": index, "size": 10} for index in range(4)]}}
    solution = {
        "placements": [
            {"buffer": 0, "pool": 0, "offset": 0},
            {"buffer": 1, "pool": 0, "offset": 10},
            {"buffer": 2, "pool": 0, "offset": 5},
            {"buffer": 3, "pool": 1, "offset": 0},
        ]
    }
    assert model.placement_relations(document, solution) == {(0, 2), (1, 2)}
    solution["placements"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        model.placement_relations(document, solution)


def test_reserved_alias_rows_do_not_claim_independent_workloads():
    candidates = [
        {"family": "build_bias", "instance": "build_bias"},
        {"family": "build_bias", "instance": "build_bias_0"},
        {"family": "", "instance": "kv_proj_matmul"},
    ]
    rows = model.audit_reserved_candidates(candidates, {"build_bias"})
    assert [row["rows_in_family"] for row in rows] == [2, 2, 1]
    assert [row["development_family_overlap"] for row in rows] == [True, True, False]
    assert not any(row["independent_workload_proven"] for row in rows)


def test_parent_timing_is_not_compared_with_a_child_graph():
    record = _record()
    record["maps"] = {arm: {"map_digest": "same"} for arm in model.ARMS}
    record["launches"] += [{**row, "device": 1} for row in record["launches"]]
    row = _row(
        "parent",
        target="child",
        role="PRIMARY",
        selection="CONTROL",
        capacity="native",
        child_instances=2,
        tightened_pool_id=1,
    )
    metrics = {
        ("parent", arm): {
            "map_digest": "same",
            "unit_cost": 0,
            "range_intersection_pairs": 0,
            "peak_sum_bytes": 100,
        }
        for arm in model.ARMS
    }
    timings, _, predictions, _ = model.evaluate_cells(
        [row],
        {"parent": record},
        {"parent.py": {"emitted_kernels": "2"}},
        metrics,
        {("parent.py", "same"): [{"model_status": "MODEL_ELIGIBLE"}]},
    )
    assert len(timings) == 2
    assert all(row["measurement_mode"] == "PARENT_PROGRAM_WINDOW" for row in timings)
    dag = [row for row in predictions if row["predictor"] == "dag"]
    assert len(dag) == len(model.WEIGHTS)
    assert all(row["reason"] == "PARENT_GRAPH_COMPOSITION_REQUIRED" for row in dag)


def test_table_empty_final_fields_round_trip_without_trailing_whitespace(tmp_path):
    path = tmp_path / "table.tsv"
    model.write_tsv(path, [{"case": "one", "reason": ""}, {"case": "two", "reason": "unsupported"}])
    assert model.read_tsv(path) == [{"case": "one", "reason": ""}, {"case": "two", "reason": "unsupported"}]
    assert all(line == line.rstrip() for line in path.read_text().splitlines())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
