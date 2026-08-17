# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for duration-model v0 and schedule-graph scoring."""

import json

import pytest
from pypto.tools import dsa_reuse_candidates, dsa_schedule_model


def _memory(root: str, size: int) -> dict:
    return {"root": root, "allocate_size_bytes": size}


def _operation(node_id: int, pipe: str, op_name: str, loop_stack: list[int] | None = None) -> dict:
    return {
        "id": node_id,
        "kind": "operation",
        "pipe": pipe,
        "op_name": op_name,
        "loop_stack": loop_stack or [],
        "defs": [_memory(f"%d{node_id}", 640)],
        "uses": [_memory(f"%u{node_id}", 640)],
    }


def _with_access(node: dict, order: int, *, explicit: bool = True) -> dict:
    node = dict(node)
    node["operation"] = (
        {"pypto_access_order": order, "location": f'loc("pypto.access.{order}")'}
        if explicit
        else {"location": f'loc("pypto.access.{order}"("kernel.py":10:3))'}
    )
    return node


def _candidate(*, prior_site: int = 3, next_site: int = 7, distance: int = 0):
    return dsa_reuse_candidates.parse_candidate_record(
        "0,1,0->1,ub->ub@vector_compute=>external->ub@inbound_dma,arenas=Vec->Vec,"
        "write_after_read,no_logical_order,inter_operation,full_allocation,complete_access_set,"
        f"verified_initial_write,in_loop,distance_{distance},sites={prior_site}->{next_site},"
        "ranges=0+640->0+640,hazard=cross_resource,dag_path=none"
    )


def _record(*, sync_edges: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [
            _operation(0, "PIPE_V", "pto.tadd"),
            _operation(1, "PIPE_MTE2", "pto.tload"),
            _operation(2, "PIPE_V", "pto.tmuls"),
            _operation(3, "PIPE_MTE2", "pto.tload"),
        ],
        "stream_edges": [
            {"source": 0, "target": 2, "pipe": "PIPE_V"},
            {"source": 1, "target": 3, "pipe": "PIPE_MTE2"},
        ],
        "sync_edges": sync_edges or [],
    }


def _ten_cycle_model() -> dsa_schedule_model.DurationModel:
    model = dsa_schedule_model.DurationModel()
    model.calibration_status = "test"
    model.operation_cycles = {
        "PIPE_V:TADD": 10.0,
        "PIPE_V:TMULS": 10.0,
        "PIPE_MTE2:TLOAD": 10.0,
    }
    return model


def test_score_computes_full_and_singleton_exposure():
    record = _record(sync_edges=[{"source": 2, "target": 1, "group": 7, "loop_carried": False}])

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["baseline_makespan_cycles"] == 20.0
    assert result["full_makespan_cycles"] == 40.0
    assert result["synchronization_exposure_cycles"] == 20.0
    assert result["sync_edge_exposure"] == [
        {
            "source": 2,
            "target": 1,
            "group": 7,
            "marginal_cycles": 20.0,
            "source_top_cycles": 20.0,
            "target_bottom_cycles": 20.0,
        }
    ]
    assert result["full_critical_path"] == [0, 2, 1, 3]


def test_score_aggregates_static_loop_work_and_discloses_dynamic_loops():
    record = _record()
    record["nodes"] = [
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "static_trip_count": 4,
        },
        {
            "id": 11,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "static_trip_count": None,
        },
        _operation(0, "PIPE_V", "pto.tadd", [10, 11]),
    ]
    record["stream_edges"] = []

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["baseline_makespan_cycles"] == 40.0
    assert result["dynamic_loop_ids"] == [11]
    assert result["node_durations"]["0"]["loop_multiplier"] == 4
    assert result["loop_policy"] == "aggregate_static_work_v0"


def test_loop_carried_sync_is_excluded_and_reported():
    record = _record(sync_edges=[{"source": 2, "target": 1, "group": 4, "loop_carried": True}])

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["excluded_loop_carried_sync_edges"] == 1
    assert result["full_makespan_cycles"] == result["baseline_makespan_cycles"]
    assert result["sync_edge_exposure"] == []


def test_score_rejects_non_loop_cycle():
    record = _record(sync_edges=[{"source": 2, "target": 0, "group": 4, "loop_carried": False}])

    with pytest.raises(ValueError, match="schedule graph is cyclic"):
        dsa_schedule_model.score_schedule(record, _ten_cycle_model())


def test_candidate_score_joins_access_sites_and_derives_non_negative_weight():
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(record["nodes"][1], 7, explicit=False),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]

    result = dsa_schedule_model.score_reuse_candidates(record, [_candidate()], _ten_cycle_model())

    assert result["base_makespan_cycles"] == 20.0
    assert result["scored_candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["source_node"] == 2
    assert candidate["target_node"] == 1
    assert candidate["weight_cycles"] == 20.0
    assert candidate["makespan_with_candidate_cycles"] == 40.0
    assert result["consumer_groups"][0]["combined_weight_cycles"] == 20.0
    assert result["candidate_weight_summary"] == {
        "positive_distance_zero_edge_count": 1,
        "positive_loop_recurrence_edge_count": 0,
        "distance_zero_weight_sum_cycles": 20.0,
        "loop_recurrence_weight_sum_cycles": 0,
        "max_distance_zero_weight_cycles": 20.0,
        "max_loop_recurrence_weight_cycles": 0.0,
        "max_candidate_weight_cycles": 20.0,
        "unique_positive_edge_count": 1,
    }


def test_duplicate_distance_zero_candidates_do_not_inflate_group_or_summary():
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(record["nodes"][1], 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]

    result = dsa_schedule_model.score_reuse_candidates(
        record, [_candidate(), _candidate()], _ten_cycle_model()
    )

    assert result["distance_zero_edges"] == [
        {
            "source_node": 2,
            "target_node": 1,
            "candidate_indices": [0, 1],
            "candidate_count": 2,
            "weight_cycles": 20.0,
        }
    ]
    assert result["consumer_groups"][0]["singleton_weight_sum_cycles"] == 20.0
    assert result["candidate_weight_summary"]["distance_zero_weight_sum_cycles"] == 20.0
    assert result["candidate_weight_summary"]["unique_positive_edge_count"] == 1


def _loop_candidate_record(*, with_return_path: bool = False) -> dict:
    record = _record()
    record["nodes"] = [
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 10,
            "end": 20,
            "static_trip_count": 4,
        },
        _with_access(_operation(0, "PIPE_V", "pto.tadd", [10]), 1),
        _with_access(_operation(1, "PIPE_MTE2", "pto.tload", [10]), 7),
        _with_access(_operation(2, "PIPE_V", "pto.tmuls", [10]), 3),
        _with_access(_operation(3, "PIPE_MTE2", "pto.tload", [10]), 9),
    ]
    if with_return_path:
        record["sync_edges"] = [{"source": 1, "target": 0, "group": 7, "loop_carried": False}]
    return record


def test_candidate_score_reports_zero_loop_recurrence_weight_without_return_path():
    record = _loop_candidate_record()

    result = dsa_schedule_model.score_reuse_candidates(record, [_candidate(distance=1)], _ten_cycle_model())

    assert result["scored_candidate_count"] == 1
    assert result["scored_loop_carried_candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "loop_carried_scored_v1"
    assert candidate["weight_cycles"] == 0.0
    assert candidate["common_loop_nodes"] == [10]
    assert candidate["resource_ii_lower_bound_cycles"] == 20.0
    assert candidate["candidate_recurrence_cycles"] is None


def test_candidate_score_derives_loop_recurrence_ii_weight():
    record = _loop_candidate_record(with_return_path=True)

    result = dsa_schedule_model.score_reuse_candidates(record, [_candidate(distance=1)], _ten_cycle_model())

    candidate = result["candidates"][0]
    assert candidate["base_ii_lower_bound_cycles"] == 20.0
    assert candidate["candidate_recurrence_cycles"] == 30.0
    assert candidate["candidate_recurrence_path"] == [1, 0, 2]
    assert candidate["with_candidate_ii_lower_bound_cycles"] == 30.0
    assert candidate["weight_cycles"] == 10.0


def test_inner_loop_ii_removes_static_outer_and_inner_trip_multipliers():
    record = _loop_candidate_record(with_return_path=True)
    record["nodes"].insert(
        0,
        {
            "id": 9,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 9,
            "end": 21,
            "static_trip_count": 3,
        },
    )
    for node in record["nodes"]:
        if node.get("kind") == "operation":
            node["loop_stack"] = [9, 10]

    result = dsa_schedule_model.score_reuse_candidates(record, [_candidate(distance=1)], _ten_cycle_model())

    candidate = result["candidates"][0]
    assert candidate["loop_node"] == 10
    assert candidate["pipe_work_cycles"] == {"PIPE_MTE2": 20.0, "PIPE_V": 20.0}
    assert candidate["candidate_recurrence_cycles"] == 30.0
    assert candidate["weight_cycles"] == 10.0


def test_existing_loop_recurrence_suppresses_duplicate_candidate_weight():
    record = _loop_candidate_record(with_return_path=True)
    record["sync_edges"].append({"source": 2, "target": 1, "group": 8, "loop_carried": True})
    record["sync_groups"] = [
        {
            "id": 8,
            "operations": [
                {"node": 2, "type": "set_flag", "loop_end": 20},
                {"node": 1, "type": "wait_flag", "loop_end": 20},
            ],
        }
    ]

    result = dsa_schedule_model.score_reuse_candidates(record, [_candidate(distance=1)], _ten_cycle_model())

    candidate = result["candidates"][0]
    assert candidate["existing_recurrence_ii_lower_bound_cycles"] == 30.0
    assert candidate["base_ii_lower_bound_cycles"] == 30.0
    assert candidate["candidate_recurrence_cycles"] == 30.0
    assert candidate["weight_cycles"] == 0.0


def test_loop_candidate_fails_closed_when_existing_recurrence_has_no_loop_identity():
    record = _loop_candidate_record(with_return_path=True)
    record["sync_edges"].append({"source": 2, "target": 1, "group": 8, "loop_carried": True})

    with pytest.raises(ValueError, match="has no loop identity"):
        dsa_schedule_model.score_reuse_candidates(record, [_candidate(distance=1)], _ten_cycle_model())


def test_duplicate_loop_candidates_collapse_to_one_scored_recurrence_edge():
    record = _loop_candidate_record(with_return_path=True)

    result = dsa_schedule_model.score_reuse_candidates(
        record, [_candidate(distance=1), _candidate(distance=1)], _ten_cycle_model()
    )

    assert result["scored_loop_carried_candidate_count"] == 2
    assert result["loop_recurrence_edges"] == [
        {
            "loop_node": 10,
            "source_node": 2,
            "target_node": 1,
            "candidate_indices": [0, 1],
            "candidate_count": 2,
            "candidate_recurrence_cycles": 30.0,
            "weight_cycles": 10.0,
        }
    ]


def test_candidate_score_fails_closed_without_access_provenance():
    with pytest.raises(ValueError, match="PYPTO_EMIT_DSA_ACCESS_PROVENANCE"):
        dsa_schedule_model.score_reuse_candidates(_record(), [_candidate()], _ten_cycle_model())


def test_candidate_score_fails_closed_when_site_pipe_does_not_join():
    record = _record()
    record["nodes"] = [_with_access(node, index) for index, node in enumerate(record["nodes"])]

    with pytest.raises(ValueError, match="did not join"):
        dsa_schedule_model.score_reuse_candidates(record, [_candidate()], _ten_cycle_model())


def test_main_scores_candidate_problem(tmp_path):
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(record["nodes"][1], 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]
    schedule = tmp_path / "schedule.jsonl"
    problem = tmp_path / "problem.dsa.json"
    output = tmp_path / "weights.json"
    schedule.write_text(json.dumps(record) + "\n")
    candidate_fields = ",".join(_candidate().fields)
    problem.write_text(
        json.dumps(
            {
                "metadata": {
                    "recognized_reuse_candidates": "1",
                    "recognized_reuse_candidate_records_v4": candidate_fields,
                }
            }
        )
    )

    assert dsa_schedule_model.main(["score-candidates", str(schedule), str(problem), "-o", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["candidates"][0]["weight_cycles"] > 0
    assert result["model_version"] == "reuse_penalty_critical_path_v1"


def test_score_fails_closed_on_control_flow_branches():
    record = _record()
    record["nodes"].append({"id": 4, "kind": "branch", "branch_kind": "IF_BEGIN"})

    with pytest.raises(ValueError, match="does not model mutually exclusive control-flow branches"):
        dsa_schedule_model.score_schedule(record, _ten_cycle_model())


def test_calibrate_uses_per_operation_and_pipe_medians(tmp_path):
    metrics = tmp_path / "instr_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "instructions": {
                    "core0": [
                        {"pipe": "VECTOR", "name": "TADD", "cycles": 8},
                        {"pipe": "VECTOR", "name": "TADD", "cycles": 12},
                    ],
                    "core1": [{"pipe": "MTE2", "instruction": "TLOAD", "cycles": 30}],
                }
            }
        )
    )

    model = dsa_schedule_model.calibrate_from_metrics([metrics])

    assert model.calibration_status == "simulator_instruction_medians"
    assert model.operation_cycles["PIPE_V:TADD"] == 10.0
    assert model.operation_cycles["PIPE_MTE2:TLOAD"] == 30.0
    assert model.pipe_parameters["PIPE_V"].minimum_cycles == 10.0
    assert model.calibration_sources == [str(metrics)]


def test_load_and_freeze_predictions_are_content_addressed(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps(_record()) + "\n")

    records = dsa_schedule_model.load_schedule_graphs(schedule)
    predictions = {"kernel": dsa_schedule_model.score_schedule(records["kernel"], _ten_cycle_model())}
    frozen = dsa_schedule_model.freeze_predictions(predictions, cohort="holdout-v0", source_paths=[schedule])

    assert frozen["cohort"] == "holdout-v0"
    assert frozen["frozen_before_device_timing"] is True
    assert len(frozen["prediction_sha256"]) == 64
    assert frozen["schedule_sources"][0]["path"] == str(schedule)
    assert len(frozen["schedule_sources"][0]["sha256"]) == 64


def test_main_scores_and_writes_frozen_record(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    output = tmp_path / "frozen.json"
    schedule.write_text(json.dumps(_record()) + "\n")

    assert (
        dsa_schedule_model.main(["score", str(schedule), "--freeze-cohort", "held-out", "-o", str(output)])
        == 0
    )
    result = json.loads(output.read_text())
    assert result["cohort"] == "held-out"
    prediction = result["predictions"][f"{schedule}:kernel"]
    assert prediction["calibration_status"] == "uncalibrated_defaults"


def test_import_legacy_debug_reconstructs_streams_and_event_edges():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
// nodes=5, syncGroups=1, activeOps=2
[   0] COMPOUND pto.tload [PIPE_MTE2]
  def=[%7(VEC)]
  use=[%arg0(GM)]
  POST: set_flag <PIPE_MTE2 -> PIPE_V> idx=3 eventIds=[0]
[   1] LOOP LOOP_BEGIN (begin=1, end=3)
  [   2] COMPOUND pto.tadd [PIPE_V]
    def=[%8(VEC)]
    use=[%7(VEC)]
    PRE : wait_flag <PIPE_MTE2 -> PIPE_V> idx=3 eventIds=[0]
[   3] LOOP LOOP_END (begin=1, end=3)
[   4] COMPOUND pto.tstore [PIPE_MTE3]
  def=[%arg1(GM)]
  use=[%8(VEC)]
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")

    assert record["export_source"] == "ptoas_debug_import_v0"
    assert record["nodes"][2]["loop_stack"] == [1]
    assert record["nodes"][0]["defs"][0]["root"] == "%7"
    assert record["sync_edges"] == [
        {
            "source": 0,
            "target": 2,
            "group": 0,
            "src_pipe": "PIPE_MTE2",
            "dst_pipe": "PIPE_V",
            "loop_carried": False,
            "root_buffers": [],
        }
    ]


def test_import_legacy_debug_rejects_incomplete_log():
    with pytest.raises(ValueError, match="no final"):
        dsa_schedule_model.import_insert_sync_debug("no phases", function="kernel")


def test_import_legacy_debug_joins_raw_pto_access_provenance():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tload [PIPE_MTE2]
[   1] COMPOUND pto.tadd [PIPE_V]
[   2] COMPOUND pto.tstore [PIPE_MTE3]
// ========================================= //
"""
    pto = """
%tile = pto.alloc_tile addr = %c0 : !pto.tile_buf loc("pypto.access.3")
pto.tload ins(%arg0) outs(%tile) loc("pypto.access.3")
pto.tadd ins(%tile, %tile) outs(%tile) loc("pypto.access.7")
pto.tstore ins(%tile) outs(%arg1) loc("pypto.access.9")
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert record["export_source"] == "ptoas_debug_import_v0+pto_access_join_v1"
    assert [node["operation"]["pypto_access_order"] for node in record["nodes"]] == [3, 7, 9]
    assert record["export_limitations"]["access_provenance_missing"] is False


def test_import_legacy_debug_rejects_raw_pto_operation_mismatch():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tload [PIPE_MTE2]
[   1] COMPOUND pto.tadd [PIPE_V]
// ========================================= //
"""
    pto = """
pto.tload ins(%arg0) outs(%tile) loc("pypto.access.3")
pto.tmul ins(%tile, %tile) outs(%tile) loc("pypto.access.7")
"""

    with pytest.raises(ValueError, match="operation sequence does not match"):
        dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)


def test_evaluate_arm_manifest_scores_direction_and_rank(tmp_path):
    baseline_a = tmp_path / "baseline_a.jsonl"
    candidate_a = tmp_path / "candidate_a.jsonl"
    baseline_b = tmp_path / "baseline_b.jsonl"
    candidate_b = tmp_path / "candidate_b.jsonl"
    baseline_a.write_text(json.dumps(_record()) + "\n")
    candidate_a.write_text(
        json.dumps(_record(sync_edges=[{"source": 2, "target": 1, "group": 7, "loop_carried": False}])) + "\n"
    )
    baseline_b.write_text(json.dumps(_record()) + "\n")
    candidate_b.write_text(json.dumps(_record()) + "\n")
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frozen_before_device_timing": True,
                "comparisons": [
                    {
                        "case": "case-a",
                        "split": "development",
                        "baseline_arm": "geometry_ff",
                        "candidate_arm": "dsa_rp_cg",
                        "baseline_schedule": baseline_a.name,
                        "candidate_schedule": candidate_a.name,
                        "baseline_latency_us": 20,
                        "candidate_latency_us": 40,
                    },
                    {
                        "case": "case-b",
                        "split": "holdout",
                        "baseline_arm": "geometry_ff",
                        "candidate_arm": "dsa_rp_cg",
                        "baseline_schedule": baseline_b.name,
                        "candidate_schedule": candidate_b.name,
                    },
                ],
            }
        )
    )

    result = dsa_schedule_model.evaluate_arm_manifest(manifest, _ten_cycle_model())

    assert result["frozen_before_device_timing"] is True
    assert result["summary"]["comparison_count"] == 2
    assert result["summary"]["observed_comparison_count"] == 1
    assert result["summary"]["direction_accuracy"] == 1.0
    assert result["comparisons"][0]["predicted_relative_delta"] == 1.0
    assert result["comparisons"][0]["observed_relative_delta"] == 1.0
    assert result["comparisons"][0]["direction_correct"] is True
    assert result["comparisons"][0]["added_sync_edges"] == [
        {
            "source": 2,
            "target": 1,
            "src_pipe": None,
            "dst_pipe": None,
            "loop_carried": False,
            "count": 1,
        }
    ]
    assert result["comparisons"][0]["removed_sync_edges"] == []
    assert result["comparisons"][1]["direction_correct"] is None
    assert len(result["prediction_sha256"]) == 64


def test_evaluate_arm_manifest_rejects_one_sided_observation(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps(_record()) + "\n")
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparisons": [
                    {
                        "case": "case",
                        "split": "holdout",
                        "baseline_arm": "geometry_ff",
                        "candidate_arm": "dsa_rp_cg",
                        "baseline_schedule": schedule.name,
                        "candidate_schedule": schedule.name,
                        "baseline_latency_us": 10,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="both observed latencies or neither"):
        dsa_schedule_model.evaluate_arm_manifest(manifest, _ten_cycle_model())


def test_evaluate_arm_manifest_rejects_different_operation_streams(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(json.dumps(_record()) + "\n")
    changed = _record()
    changed["nodes"][0]["op_name"] = "pto.tmul"
    candidate.write_text(json.dumps(changed) + "\n")
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparisons": [
                    {
                        "case": "case",
                        "split": "holdout",
                        "baseline_arm": "geometry_ff",
                        "candidate_arm": "dsa_rp_cg",
                        "baseline_schedule": baseline.name,
                        "candidate_schedule": candidate.name,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="different operation streams"):
        dsa_schedule_model.evaluate_arm_manifest(manifest, _ten_cycle_model())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
