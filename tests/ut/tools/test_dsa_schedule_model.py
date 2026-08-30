# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for duration-model v0 and schedule-graph scoring."""

import hashlib
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


def test_complete_signature_mode_does_not_fall_back_to_family_median():
    record = _record()
    for index, node in enumerate(record["nodes"]):
        record["nodes"][index] = _with_access(node, index)
    model = _ten_cycle_model()
    model.operation_signature_cycles = {"unrelated-complete-signature": 1.0}

    with pytest.raises(ValueError, match="no exact-signature duration estimate"):
        dsa_schedule_model.score_schedule(record, model)


def test_complete_signature_accepts_equivalent_compact_tile_type_encoding():
    compact_type = "!pto.tile_buf<vec, 8x8xi32, valid=?x?>"
    keyed_type = (
        "!pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=8, v_row=?, v_col=?, "
        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )
    node = _operation(0, "PIPE_V", "pto.tadd")
    node["defs"] = []
    node["uses"] = []
    node["operation"] = {
        "location": f"pto.tadd ins(%lhs, %rhs : {compact_type}, {compact_type}) outs(%out : {compact_type})",
        "operand_types": [compact_type, compact_type],
        "result_types": [compact_type],
        "operand_constants": [None, None],
        "attributes": {},
        "static_work_bytes": 256,
    }
    expected_signature = {
        **dsa_schedule_model.operation_duration_signature(node),
        "operand_types": [keyed_type, keyed_type],
        "result_types": [keyed_type],
        "semantic_operation": "keyed spelling intentionally differs",
    }
    model = dsa_schedule_model.DurationModel(
        calibration_status="test",
        operation_signature_cycles={dsa_schedule_model._operation_signature_key(expected_signature): 37.0},
    )
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [node],
        "stream_edges": [],
        "sync_edges": [],
    }

    result = dsa_schedule_model.score_schedule(record, model)

    assert result["full_makespan_cycles"] == 37.0
    assert result["duration_source_counts"] == {"simulator_complete_signature_compatible_encoding": 1}


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
    assert result["latency_graph_complete"] is True
    assert result["latency_graph_limitations"] == []


def test_score_aggregates_static_loop_work():
    record = _record()
    record["nodes"] = [
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 10,
            "end": 11,
            "static_trip_count": 4,
        },
        _operation(0, "PIPE_V", "pto.tadd", [10]),
    ]
    record["stream_edges"] = []

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["baseline_makespan_cycles"] == 40.0
    assert result["loop_aware_makespan_cycles"] == 40.0
    assert result["dynamic_loop_ids"] == []
    assert result["node_durations"]["0"]["loop_multiplier"] == 4
    assert result["loop_policy"] == "aggregate_static_work_v0"


def test_score_reports_pre_codegen_sync_records_separately_from_latency():
    record = _record(sync_edges=[{"source": 0, "target": 1, "group": 7, "loop_carried": False}])
    record["nodes"].insert(
        0,
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 10,
            "end": 11,
            "static_trip_count": 4,
            "loop_stack": [],
        },
    )
    record["nodes"].append(
        {
            "id": 11,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 10,
            "end": 11,
            "static_trip_count": 4,
            "loop_stack": [],
        }
    )
    for node in record["nodes"]:
        if node.get("kind") == "operation":
            node["loop_stack"] = [10]
    record["sync_groups"] = [
        {
            "id": 7,
            "operations": [
                {
                    "node": 0,
                    "type": "set_flag",
                    "src_pipe": "PIPE_V",
                    "dst_pipe": "PIPE_MTE2",
                },
                {
                    "node": 1,
                    "type": "wait_flag",
                    "src_pipe": "PIPE_V",
                    "dst_pipe": "PIPE_MTE2",
                },
            ],
        }
    ]
    record["sync_edges"].append({"source": 11, "target": 1, "group": 8, "loop_carried": False})

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["pre_codegen_sync_record_summary"] == {
        "model_version": "pre_codegen_sync_record_count_v1",
        "group_count": 1,
        "active_group_count": 1,
        "record_count": 2,
        "active_record_site_count": 2,
        "useless_record_site_count": 0,
        "active_record_execution_count": 8,
        "active_record_sites_by_type": {"set_flag": 1, "wait_flag": 1},
        "active_record_executions_by_type": {"set_flag": 4, "wait_flag": 4},
        "active_record_sites_by_pipe_pair": {"PIPE_V->PIPE_MTE2": 2},
        "active_record_executions_by_pipe_pair": {"PIPE_V->PIPE_MTE2": 8},
    }
    assert result["excluded_non_operation_sync_edges"] == 0
    assert result["latency_graph_complete"] is True
    assert result["latency_graph_limitations"] == []
    assert result["loop_sync_models"][0]["loop_boundary_sync_edges"] == [
        {"source": 11, "target": 1, "group": 8, "kind": "loop_boundary"}
    ]


def test_pre_codegen_sync_summary_keeps_duplicate_barrier_records_distinct():
    record = _record()
    record["sync_groups"] = [
        {
            "id": group_id,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_V",
            "operations": [{"node": 1, "type": "pipe_barrier"}],
        }
        for group_id in (11, 12)
    ]

    summary = dsa_schedule_model.score_schedule(record, _ten_cycle_model())["pre_codegen_sync_record_summary"]

    # These are two Final-SyncIR records. SyncCodegen may merge them into one
    # emitted barrier, which is why this summary deliberately does not claim an
    # instruction count.
    assert summary["group_count"] == 2
    assert summary["active_record_site_count"] == 2


def test_latency_graph_is_incomplete_when_export_omits_barrier_dependencies():
    record = _record()
    record["export_limitations"] = {"barrier_dependency_nodes_missing": 1}

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["latency_graph_complete"] is False
    assert result["latency_graph_limitations"] == ["export_limitations.barrier_dependency_nodes_missing"]


def test_score_fails_closed_on_dynamic_loop():
    record = _record()
    record["nodes"].insert(
        0,
        {"id": 11, "kind": "loop", "loop_kind": "LOOP_BEGIN", "static_trip_count": None},
    )
    for node in record["nodes"]:
        if node.get("kind") == "operation":
            node["loop_stack"] = [11]

    with pytest.raises(ValueError, match="requires statically bounded loops"):
        dsa_schedule_model.score_schedule(record, _ten_cycle_model())


def test_loop_carried_sync_without_loop_identity_fails_closed():
    record = _record(sync_edges=[{"source": 2, "target": 1, "group": 4, "loop_carried": True}])

    with pytest.raises(ValueError, match="does not resolve to exactly one loop model"):
        dsa_schedule_model.score_schedule(record, _ten_cycle_model())


def test_score_models_existing_loop_recurrence_and_boundary_edges():
    record = _record()
    record["nodes"] = [
        _operation(4, "PIPE_V", "pto.tadd"),
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 10,
            "end": 11,
            "static_trip_count": 4,
            "loop_stack": [],
        },
        _operation(0, "PIPE_V", "pto.tadd", [10]),
        _operation(1, "PIPE_MTE2", "pto.tload", [10]),
        _operation(2, "PIPE_V", "pto.tmuls", [10]),
        _operation(3, "PIPE_MTE2", "pto.tload", [10]),
        {
            "id": 11,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 10,
            "end": 11,
            "static_trip_count": 4,
            "loop_stack": [],
        },
        _operation(5, "PIPE_MTE2", "pto.tload"),
    ]
    record["sync_edges"] = [
        {"source": 4, "target": 10, "group": 0, "loop_carried": False},
        {"source": 10, "target": 1, "group": 1, "loop_carried": False},
        {"source": 1, "target": 0, "group": 4, "loop_carried": False},
        {"source": 2, "target": 1, "group": 2, "loop_carried": True},
        {"source": 2, "target": 11, "group": 3, "loop_carried": False},
        {"source": 11, "target": 5, "group": 5, "loop_carried": False},
    ]
    record["sync_groups"] = [
        {
            "id": 2,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_MTE2",
            "operations": [
                {"node": 2, "type": "set_flag", "loop_end": 11},
                {"node": 1, "type": "wait_flag", "loop_end": 11},
            ],
        }
    ]

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    model = result["loop_sync_models"][0]
    assert model["static_trip_count"] == 4
    assert model["recurrence_ii_lower_bound_cycles"] == 30.0
    assert model["ii_lower_bound_cycles"] == 30.0
    assert model["loop_carried_recurrences"] == [
        {"source": 2, "target": 1, "group": 2, "cycles": 30.0, "path": [1, 0, 2]}
    ]
    assert model["loop_boundary_sync_edges"] == [
        {"source": 4, "target": 10, "group": 0, "kind": "loop_entry"},
        {"source": 10, "target": 1, "group": 1, "kind": "loop_entry"},
        {"source": 2, "target": 11, "group": 3, "kind": "loop_exit"},
        {"source": 11, "target": 5, "group": 5, "kind": "loop_exit"},
    ]
    assert result["excluded_loop_carried_sync_edges"] == 1
    assert result["modeled_loop_carried_sync_edges"] == 1
    assert result["unresolved_loop_carried_sync_edges"] == 0
    assert result["latency_graph_complete"] is True
    assert result["excluded_non_operation_sync_edges"] == 0


def test_moved_loop_sync_metadata_is_reclassified_as_ordinary_dependency():
    record = _record()
    record["nodes"] = [
        _operation(0, "PIPE_V", "pto.tadd"),
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 10,
            "end": 13,
            "static_trip_count": 4,
            "loop_stack": [],
        },
        _operation(11, "PIPE_MTE2", "pto.tload", [10]),
        _operation(12, "PIPE_V", "pto.tmuls", [10]),
        {
            "id": 13,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 10,
            "end": 13,
            "static_trip_count": 4,
            "loop_stack": [],
        },
    ]
    record["stream_edges"] = [{"source": 0, "target": 12, "pipe": "PIPE_V"}]
    record["sync_edges"] = [{"source": 0, "target": 13, "group": 4, "loop_carried": True}]
    record["sync_groups"] = [
        {
            "id": 4,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_MTE2",
            "operations": [
                {"node": 0, "type": "set_flag", "loop_end": 13},
                {"node": 13, "type": "wait_flag", "loop_end": 13},
            ],
        }
    ]

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["declared_loop_carried_sync_edges"] == 1
    assert result["excluded_loop_carried_sync_edges"] == 0
    assert result["reclassified_non_recurrence_sync_edges"] == 1
    assert result["modeled_loop_carried_sync_edges"] == 0
    assert result["latency_graph_complete"] is True


def test_score_models_outer_recurrence_from_nested_loop_end():
    record = _record()
    record["nodes"] = [
        {
            "id": 16,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 16,
            "end": 37,
            "static_trip_count": 3,
            "loop_stack": [],
        },
        _operation(17, "PIPE_MTE2", "pto.tload", [16]),
        {
            "id": 21,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 21,
            "end": 28,
            "static_trip_count": 2,
            "loop_stack": [16],
        },
        _operation(24, "PIPE_V", "pto.tmuls", [16, 21]),
        {
            "id": 28,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 21,
            "end": 28,
            "static_trip_count": 2,
            "loop_stack": [16],
        },
        {
            "id": 37,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 16,
            "end": 37,
            "static_trip_count": 3,
            "loop_stack": [],
        },
    ]
    record["stream_edges"] = []
    record["sync_edges"] = [
        {"source": 17, "target": 21, "group": 1, "loop_carried": False},
        {"source": 21, "target": 24, "group": 2, "loop_carried": False},
        {"source": 24, "target": 28, "group": 3, "loop_carried": False},
        {"source": 28, "target": 17, "group": 4, "loop_carried": True},
    ]
    record["sync_groups"] = [
        {
            "id": 4,
            "src_pipe": "PIPE_M",
            "dst_pipe": "PIPE_MTE2",
            "operations": [
                {"node": 28, "type": "set_flag", "loop_end": 37},
                {"node": 17, "type": "wait_flag", "loop_end": 37},
            ],
        }
    ]

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    outer = next(model for model in result["loop_sync_models"] if model["loop_node"] == 16)
    assert outer["structural_node_count"] == 2
    assert outer["loop_carried_recurrences"] == [
        {"source": 28, "target": 17, "group": 4, "cycles": 30.0, "path": [17, 21, 24, 28]}
    ]
    assert result["latency_graph_complete"] is True


def test_non_cycle_loop_carried_sync_is_resolved_without_ii_constraint():
    record = _loop_candidate_record()
    record["sync_edges"] = [{"source": 1, "target": 2, "group": 4, "loop_carried": True}]
    record["sync_groups"] = [
        {
            "id": 4,
            "src_pipe": "PIPE_MTE2",
            "dst_pipe": "PIPE_V",
            "operations": [
                {"node": 1, "type": "set_flag", "loop_end": 20},
                {"node": 2, "type": "wait_flag", "loop_end": 20},
            ],
        }
    ]

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["modeled_loop_carried_sync_edges"] == 1
    assert result["non_cycle_loop_carried_sync_edges"] == 1
    assert result["unresolved_loop_carried_sync_edges"] == 0
    assert result["latency_graph_complete"] is True


def test_final_tail_dependency_zero_is_a_structural_sentinel():
    record = _record()
    record["nodes"] = [_operation(1, "PIPE_V", "pto.tadd"), _operation(2, "PIPE_V", "pto.tmuls")]
    record["stream_edges"] = [{"source": 1, "target": 2, "pipe": "PIPE_V"}]
    record["sync_edges"] = [{"source": 0, "target": 2, "group": 7, "loop_carried": False}]

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    assert result["excluded_sentinel_sync_edges"] == 1
    assert result["excluded_non_operation_sync_edges"] == 0
    assert result["latency_graph_complete"] is True


def test_cross_loop_end_to_begin_is_exit_then_entry():
    record = _record()
    record["nodes"] = [
        {
            "id": 10,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 10,
            "end": 11,
            "static_trip_count": 2,
            "loop_stack": [],
        },
        _operation(0, "PIPE_V", "pto.tadd", [10]),
        {
            "id": 11,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 10,
            "end": 11,
            "static_trip_count": 2,
            "loop_stack": [],
        },
        {
            "id": 20,
            "kind": "loop",
            "loop_kind": "LOOP_BEGIN",
            "begin": 20,
            "end": 21,
            "static_trip_count": 3,
            "loop_stack": [],
        },
        _operation(1, "PIPE_MTE2", "pto.tload", [20]),
        {
            "id": 21,
            "kind": "loop",
            "loop_kind": "LOOP_END",
            "begin": 20,
            "end": 21,
            "static_trip_count": 3,
            "loop_stack": [],
        },
    ]
    record["stream_edges"] = []
    record["sync_edges"] = [{"source": 11, "target": 20, "group": 7, "loop_carried": False}]

    result = dsa_schedule_model.score_schedule(record, _ten_cycle_model())

    first, second = result["loop_sync_models"]
    assert first["loop_boundary_sync_edges"] == [
        {"source": 11, "target": 20, "group": 7, "kind": "loop_exit"}
    ]
    assert second["loop_boundary_sync_edges"] == [
        {"source": 11, "target": 20, "group": 7, "kind": "loop_entry"}
    ]


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
    assert result["penalty_pair_weights"] == [
        {
            "first_buffer": 0,
            "second_buffer": 1,
            "promoted_to_dsa_penalty": True,
            "unit_cost": 1.0,
            "candidate_record_count": 1,
            "executable_candidate_record_count": 1,
            "not_materialized_candidate_record_count": 0,
            "executable_in_lowered_schedule": True,
            "model_status": "executable",
            "distance_zero_schedule_edges": [[2, 1]],
            "loop_carried_schedule_edges": [],
            "estimated_sync_endpoint_executions": 2,
            "distance_zero_weight_cycles": 20.0,
            "loop_ii_weight_cycles": 0,
            "loop_total_weight_cycles": 0,
            "critical_path_weight_cycles": 20.0,
        }
    ]
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


def test_candidate_score_fails_closed_on_conditional_endpoint_lifting():
    record = _record()
    record["nodes"].extend(
        [
            {"id": 10, "kind": "branch", "branch_kind": "IF_BEGIN"},
            {"id": 11, "kind": "branch", "branch_kind": "IF_END"},
        ]
    )

    with pytest.raises(ValueError, match="does not lift conditional access endpoints"):
        dsa_schedule_model.score_reuse_candidates(record, [_candidate()], _ten_cycle_model())


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
            "source_pipe": "PIPE_V",
            "target_pipe": "PIPE_MTE2",
            "candidate_indices": [0, 1],
            "candidate_count": 2,
            "source_execution_count": 1,
            "target_execution_count": 1,
            "estimated_sync_endpoint_executions": 2,
            "weight_cycles": 20.0,
        }
    ]
    assert result["consumer_groups"][0]["singleton_weight_sum_cycles"] == 20.0
    assert result["penalty_pair_weights"][0]["estimated_sync_endpoint_executions"] == 2
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


@pytest.mark.parametrize("defect", ["missing_group", "unmatched_loop_end"])
def test_existing_loop_recurrence_metadata_fails_closed(defect):
    record = _loop_candidate_record(with_return_path=True)
    source, target = 2, 1
    loop_end = 20
    record["sync_edges"].append({"source": source, "target": target, "group": 8, "loop_carried": True})
    if defect != "missing_group":
        record["sync_groups"] = [
            {
                "id": 8,
                "src_pipe": "PIPE_V",
                "dst_pipe": "PIPE_MTE2",
                "operations": [
                    {
                        "node": source,
                        "type": "set_flag",
                        "loop_end": 99 if defect == "unmatched_loop_end" else loop_end,
                    },
                    {
                        "node": target,
                        "type": "wait_flag",
                        "loop_end": 99 if defect == "unmatched_loop_end" else loop_end,
                    },
                ],
            }
        ]

    with pytest.raises(ValueError, match="does not resolve to exactly one loop model"):
        dsa_schedule_model.score_schedule(record, _ten_cycle_model())


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
    assert result["penalty_pair_weights"] == [
        {
            "first_buffer": 0,
            "second_buffer": 1,
            "promoted_to_dsa_penalty": True,
            "unit_cost": 1.0,
            "candidate_record_count": 1,
            "executable_candidate_record_count": 1,
            "not_materialized_candidate_record_count": 0,
            "executable_in_lowered_schedule": True,
            "model_status": "executable",
            "distance_zero_schedule_edges": [],
            "loop_carried_schedule_edges": [[10, 2, 1]],
            "estimated_sync_endpoint_executions": 8,
            "distance_zero_weight_cycles": 0.0,
            "loop_ii_weight_cycles": 10.0,
            "loop_total_weight_cycles": 30.0,
            "critical_path_weight_cycles": 30.0,
        }
    ]


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
                {
                    "node": 2,
                    "type": "set_flag",
                    "loop_end": 20,
                    "src_pipe": "PIPE_V",
                    "dst_pipe": "PIPE_MTE2",
                },
                {
                    "node": 1,
                    "type": "wait_flag",
                    "loop_end": 20,
                    "src_pipe": "PIPE_V",
                    "dst_pipe": "PIPE_MTE2",
                },
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
            "source_pipe": "PIPE_V",
            "target_pipe": "PIPE_MTE2",
            "candidate_indices": [0, 1],
            "candidate_count": 2,
            "source_execution_count": 4,
            "target_execution_count": 4,
            "estimated_sync_endpoint_executions": 8,
            "candidate_recurrence_cycles": 30.0,
            "weight_cycles": 10.0,
        }
    ]
    assert result["penalty_pair_weights"][0]["estimated_sync_endpoint_executions"] == 8


def test_candidate_score_fails_closed_without_access_provenance():
    with pytest.raises(ValueError, match="PYPTO_EMIT_DSA_ACCESS_PROVENANCE"):
        dsa_schedule_model.score_reuse_candidates(_record(), [_candidate()], _ten_cycle_model())


def test_candidate_score_fails_closed_when_site_pipe_does_not_join():
    record = _record()
    record["nodes"] = [_with_access(node, index) for index, node in enumerate(record["nodes"])]

    with pytest.raises(ValueError, match="did not join"):
        dsa_schedule_model.score_reuse_candidates(record, [_candidate()], _ten_cycle_model())


def test_candidate_score_classifies_access_removed_before_lowered_schedule():
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(record["nodes"][1], 7),
        # Access 3 is intentionally absent: it was removed by a lowering pass.
        _with_access(record["nodes"][2], 11),
        _with_access(record["nodes"][3], 9),
    ]

    with pytest.raises(ValueError, match="without non-materialization evidence: \\[3\\]"):
        dsa_schedule_model.score_reuse_candidates(record, [_candidate()], _ten_cycle_model())

    result = dsa_schedule_model.score_reuse_candidates(
        record,
        [_candidate()],
        _ten_cycle_model(),
        known_nonmaterialized_access_orders=frozenset({3}),
    )

    assert result["scored_candidate_count"] == 0
    assert result["not_materialized_candidate_count"] == 1
    assert result["candidates"][0] == {
        "candidate_index": 0,
        "first_buffer": 0,
        "second_buffer": 1,
        "prior_buffer": 0,
        "next_buffer": 1,
        "prior_access_order": 3,
        "next_access_order": 7,
        "prior_route_pipe": "PIPE_V",
        "next_route_pipe": "PIPE_MTE2",
        "missing_access_orders": [3],
        "status": "not_materialized_in_schedule",
        "weight_cycles": 0.0,
    }
    assert result["penalty_pair_weights"] == [
        {
            "first_buffer": 0,
            "second_buffer": 1,
            "promoted_to_dsa_penalty": True,
            "unit_cost": 1.0,
            "candidate_record_count": 1,
            "executable_candidate_record_count": 0,
            "not_materialized_candidate_record_count": 1,
            "executable_in_lowered_schedule": False,
            "model_status": "proven_nonmaterialized",
            "distance_zero_schedule_edges": [],
            "loop_carried_schedule_edges": [],
            "estimated_sync_endpoint_executions": 0,
            "distance_zero_weight_cycles": 0.0,
            "loop_ii_weight_cycles": 0,
            "loop_total_weight_cycles": 0,
            "critical_path_weight_cycles": 0.0,
        }
    ]


def test_candidate_score_preserves_unmodeled_pipeline_penalty_without_access_record(tmp_path):
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(record["nodes"][1], 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]

    penalties = {(0, 1): 1.0, (2, 3): 4.0}
    with pytest.raises(ValueError, match="not a structured pipeline-serialization penalty"):
        dsa_schedule_model.score_reuse_candidates(
            record,
            [_candidate()],
            _ten_cycle_model(),
            promoted_penalties=penalties,
        )

    result = dsa_schedule_model.score_reuse_candidates(
        record,
        [_candidate()],
        _ten_cycle_model(),
        promoted_penalties=penalties,
        promoted_penalty_reasons={(0, 1): "cross_pipe", (2, 3): "pipeline_serialization"},
    )

    assert result["unmodeled_pipeline_serialization_pair_count"] == 1
    assert result["penalty_pair_weights"][1] == {
        "first_buffer": 2,
        "second_buffer": 3,
        "promoted_to_dsa_penalty": True,
        "unit_cost": 4.0,
        "candidate_record_count": 0,
        "executable_candidate_record_count": 0,
        "not_materialized_candidate_record_count": 0,
        "executable_in_lowered_schedule": False,
        "model_status": "unmodeled_pipeline_serialization",
        "penalty_reason": "pipeline_serialization",
        "distance_zero_schedule_edges": [],
        "loop_carried_schedule_edges": [],
        "estimated_sync_endpoint_executions": 0,
        "distance_zero_weight_cycles": 0.0,
        "loop_ii_weight_cycles": 0.0,
        "loop_total_weight_cycles": 0.0,
        "critical_path_weight_cycles": 0.0,
    }

    problem = tmp_path / "problem.dsa.json"
    solution = tmp_path / "solution.dsa.solution.json"
    problem.write_text(
        json.dumps({"problem": {"buffers": [{"id": buffer_id, "size": 64} for buffer_id in range(4)]}})
    )
    solution.write_text(
        json.dumps({"placements": [{"buffer": buffer_id, "pool": 1, "offset": 0} for buffer_id in range(4)]})
    )

    realized = dsa_schedule_model.score_realized_reuse(problem, solution, result)

    assert realized["synchronization_predictor_coverage_complete"] is False
    assert realized["unmodeled_pipeline_serialization_realized_pair_count"] == 1
    assert realized["unmodeled_pipeline_serialization_realized_cost"] == 4.0
    assert realized["unit_realized_cost"] == 5.0
    assert realized["executable_unit_realized_cost"] == 1.0


def test_nonmaterialized_access_evidence_is_bound_to_scored_inputs(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    problem = tmp_path / "problem.json"
    evidence = tmp_path / "evidence.json"
    schedule.write_text("schedule\n")
    problem.write_text("problem\n")
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schedule_sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
                "problem_sha256": hashlib.sha256(problem.read_bytes()).hexdigest(),
                "nonmaterialized_access_orders": [3, 9],
            }
        )
    )

    assert dsa_schedule_model._load_nonmaterialized_access_evidence(
        evidence,
        schedule_path=schedule,
        problem_path=problem,
    ) == frozenset({3, 9})

    schedule.write_text("changed\n")
    with pytest.raises(ValueError, match="schedule_sha256 does not match"):
        dsa_schedule_model._load_nonmaterialized_access_evidence(
            evidence,
            schedule_path=schedule,
            problem_path=problem,
        )


def test_candidate_score_records_narrow_tci_schedule_pipe_override():
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(_operation(1, "PIPE_S", "pto.tci"), 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]
    record["stream_edges"] = [{"source": 0, "target": 2, "pipe": "PIPE_V"}]
    model = _ten_cycle_model()
    model.operation_cycles["PIPE_S:TCI"] = 10.0
    vector_target = dsa_reuse_candidates.parse_candidate_record(
        "0,1,0->1,ub->ub@vector_compute=>ub->ub@vector_compute,arenas=Vec->Vec,"
        "write_after_read,no_logical_order,inter_operation,full_allocation,complete_access_set,"
        "verified_initial_write,in_loop,distance_0,sites=3->7,"
        "ranges=0+640->0+640,hazard=cross_resource,dag_path=none"
    )

    result = dsa_schedule_model.score_reuse_candidates(record, [vector_target], model)

    candidate = result["candidates"][0]
    assert candidate["next_route_pipe"] == "PIPE_V"
    assert candidate["next_pipe"] == "PIPE_S"
    assert candidate["next_pipe_override"] == "ptoas_v057_tci_schedule_pipe"


def test_candidate_score_rejects_unknown_schedule_pipe_override():
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(_operation(1, "PIPE_S", "pto.tadd"), 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]
    record["stream_edges"] = [{"source": 0, "target": 2, "pipe": "PIPE_V"}]
    model = _ten_cycle_model()
    model.operation_cycles["PIPE_S:TADD"] = 10.0
    vector_target = dsa_reuse_candidates.parse_candidate_record(
        "0,1,0->1,ub->ub@vector_compute=>ub->ub@vector_compute,arenas=Vec->Vec,"
        "write_after_read,no_logical_order,inter_operation,full_allocation,complete_access_set,"
        "verified_initial_write,in_loop,distance_0,sites=3->7,"
        "ranges=0+640->0+640,hazard=cross_resource,dag_path=none"
    )

    with pytest.raises(ValueError, match="did not join"):
        dsa_schedule_model.score_reuse_candidates(record, [vector_target], model)


def test_candidate_score_records_tsetval_schedule_pipe_override():
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(_operation(1, "PIPE_S", "pto.tsetval"), 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]
    record["stream_edges"] = [{"source": 0, "target": 2, "pipe": "PIPE_V"}]
    model = _ten_cycle_model()
    model.operation_cycles["PIPE_S:TSETVAL"] = 1.0
    vector_target = dsa_reuse_candidates.parse_candidate_record(
        "0,1,0->1,ub->ub@vector_compute=>ub->ub@vector_compute,arenas=Vec->Vec,"
        "write_after_read,no_logical_order,inter_operation,full_allocation,complete_access_set,"
        "verified_initial_write,in_loop,distance_0,sites=3->7,"
        "ranges=0+640->0+640,hazard=cross_resource,dag_path=none"
    )

    result = dsa_schedule_model.score_reuse_candidates(record, [vector_target], model)

    candidate = result["candidates"][0]
    assert candidate["next_pipe"] == "PIPE_S"
    assert candidate["next_pipe_override"] == "ptoas_v057_tsetval_schedule_pipe"


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
    model = tmp_path / "duration-model.json"
    schedule.write_text(json.dumps(record) + "\n")
    model.write_text(json.dumps(_ten_cycle_model().to_json()))
    candidate_fields = ",".join(_candidate().fields)
    problem.write_text(
        json.dumps(
            {
                "metadata": {
                    "recognized_reuse_candidates": "1",
                    "recognized_reuse_candidate_records_v4": candidate_fields,
                },
                "problem": {
                    "cost_model": {
                        "reuse_penalties": [{"first": 0, "second": 1, "cost": 1, "reason": "cross_pipe"}]
                    }
                },
            }
        )
    )

    assert (
        dsa_schedule_model.main(
            [
                "score-candidates",
                str(schedule),
                str(problem),
                "--model",
                str(model),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["candidates"][0]["weight_cycles"] > 0
    assert result["model_version"] == "reuse_penalty_critical_path_v1"


def test_realized_placement_scores_only_physical_reuse(tmp_path):
    record = _record()
    record["nodes"] = [
        _with_access(record["nodes"][0], 1),
        _with_access(record["nodes"][1], 7),
        _with_access(record["nodes"][2], 3),
        _with_access(record["nodes"][3], 9),
    ]
    candidate_scores = dsa_schedule_model.score_reuse_candidates(
        record,
        [_candidate()],
        _ten_cycle_model(),
        {(0, 1): 3.0},
    )
    problem = tmp_path / "problem.dsa.json"
    solution = tmp_path / "solution.dsa.solution.json"
    problem.write_text(
        json.dumps(
            {
                "problem": {
                    "buffers": [
                        {"id": 0, "size": 64},
                        {"id": 1, "size": 64},
                    ]
                }
            }
        )
    )
    solution.write_text(
        json.dumps(
            {
                "placements": [
                    {"buffer": 0, "pool": 1, "offset": 0},
                    {"buffer": 1, "pool": 1, "offset": 32},
                ]
            }
        )
    )

    result = dsa_schedule_model.score_realized_reuse(problem, solution, candidate_scores)

    assert result["realized_pair_count"] == 1
    assert result["canonical_physical_reuse_group_count"] == 1
    assert result["executable_realized_pair_count"] == 1
    assert result["executable_canonical_physical_reuse_group_count"] == 1
    assert result["synchronization_predictor_coverage_complete"] is True
    assert result["unmodeled_pipeline_serialization_realized_pair_count"] == 0
    assert result["unmodeled_pipeline_serialization_realized_cost"] == 0
    assert result["realized_pair_count_without_induced_sync_edge"] == 0
    assert result["unit_realized_cost"] == 3.0
    assert result["executable_unit_realized_cost"] == 3.0
    assert result["unit_realized_cost_without_induced_sync_edge"] == 0
    assert result["critical_path_realized_cost_cycles"] == 20.0
    assert result["unique_induced_sync_edge_count"] == 1
    assert result["estimated_sync_endpoint_executions"] == 2
    assert result["estimated_sync_endpoint_executions_by_pipe_pair"] == {"PIPE_V->PIPE_MTE2": 2}
    assert result["canonical_physical_reuse_groups"] == [
        {
            "id": 0,
            "pool": 1,
            "first_range": [0, 64],
            "second_range": [32, 96],
            "overlap_range": [32, 64],
            "shared_bytes": 32,
            "logical_pairs": [[0, 1]],
            "logical_pair_count": 1,
            "logical_unit_cost": 3.0,
            "unique_induced_sync_edge_count": 1,
            "estimated_sync_endpoint_executions": 2,
            "estimated_sync_endpoint_executions_by_pipe_pair": {"PIPE_V->PIPE_MTE2": 2},
        }
    ]
    assert result["executable_canonical_physical_reuse_groups"] == result["canonical_physical_reuse_groups"]
    assert result["edge_explanations"] == [
        {
            "first_buffer": 0,
            "second_buffer": 1,
            "physical_group_id": 0,
            "physical_pool": 1,
            "physical_first_range": [0, 64],
            "physical_second_range": [32, 96],
            "physical_overlap_range": [32, 64],
            "shared_bytes": 32,
            "candidate_index": 0,
            "candidate_status": "scored",
            "prior_access_order": 3,
            "next_access_order": 7,
            "missing_access_orders": [],
            "lowered_source_node": 2,
            "lowered_source_operation": "pto.tmuls",
            "lowered_source_pipe": "PIPE_V",
            "lowered_target_node": 1,
            "lowered_target_operation": "pto.tload",
            "lowered_target_pipe": "PIPE_MTE2",
            "actual_sync_group_ids": [],
            "source_loop_multiplier": 1,
            "target_loop_multiplier": 1,
            "critical_path_weight_cycles": 20.0,
            "critical_path_slack_cycles": 0.0,
            "loop_ii_slack_cycles": None,
            "slack_basis": "whole_function_dag",
        }
    ]
    assert result["pairs"][0]["overlap_bytes"] == 32

    no_edge_scores = {
        **candidate_scores,
        "distance_zero_edges": [],
        "loop_recurrence_edges": [],
        "penalty_pair_weights": [
            {
                **candidate_scores["penalty_pair_weights"][0],
                "distance_zero_schedule_edges": [],
                "loop_carried_schedule_edges": [],
                "estimated_sync_endpoint_executions": 0,
            }
        ],
    }
    no_edge_result = dsa_schedule_model.score_realized_reuse(problem, solution, no_edge_scores)
    assert no_edge_result["realized_pair_count_without_induced_sync_edge"] == 1
    assert no_edge_result["unit_realized_cost_without_induced_sync_edge"] == 3.0

    solution.write_text(
        json.dumps(
            {
                "placements": [
                    {"buffer": 0, "pool": 1, "offset": 0},
                    {"buffer": 1, "pool": 1, "offset": 64},
                ]
            }
        )
    )
    disjoint = dsa_schedule_model.score_realized_reuse(problem, solution, candidate_scores)
    assert disjoint["realized_pair_count"] == 0
    assert disjoint["canonical_physical_reuse_group_count"] == 0
    assert disjoint["realized_pair_count_without_induced_sync_edge"] == 0
    assert disjoint["unit_realized_cost"] == 0
    assert disjoint["critical_path_realized_cost_cycles"] == 0
    assert disjoint["unique_induced_sync_edge_count"] == 0
    assert disjoint["estimated_sync_endpoint_executions"] == 0
    assert disjoint["canonical_physical_reuse_groups"] == []


def test_realized_placement_collapses_logical_pairs_by_physical_range(tmp_path):
    problem = tmp_path / "problem.dsa.json"
    solution = tmp_path / "solution.dsa.solution.json"
    problem.write_text(
        json.dumps(
            {
                "problem": {
                    "buffers": [
                        {"id": 0, "size": 64},
                        {"id": 1, "size": 64},
                        {"id": 2, "size": 64},
                    ]
                }
            }
        )
    )
    solution.write_text(
        json.dumps(
            {
                "placements": [
                    {"buffer": 0, "pool": 1, "offset": 0},
                    {"buffer": 1, "pool": 1, "offset": 0},
                    {"buffer": 2, "pool": 1, "offset": 0},
                ]
            }
        )
    )
    pair_template = {
        "promoted_to_dsa_penalty": True,
        "distance_zero_schedule_edges": [],
        "loop_carried_schedule_edges": [],
        "estimated_sync_endpoint_executions": 0,
        "critical_path_weight_cycles": 0,
    }
    scores = {
        "schema_version": 2,
        "distance_zero_edges": [],
        "loop_recurrence_edges": [],
        "penalty_pair_weights": [
            {**pair_template, "first_buffer": 0, "second_buffer": 1, "unit_cost": 1},
            {**pair_template, "first_buffer": 0, "second_buffer": 2, "unit_cost": 2},
        ],
    }

    result = dsa_schedule_model.score_realized_reuse(problem, solution, scores)

    assert result["realized_pair_count"] == 2
    assert result["canonical_physical_reuse_group_count"] == 1
    assert result["canonical_physical_reuse_groups"][0]["logical_pairs"] == [[0, 1], [0, 2]]
    assert result["canonical_physical_reuse_groups"][0]["logical_unit_cost"] == 3


def test_realized_placement_keeps_distinct_tile_pairs_with_same_intersection(tmp_path):
    problem = tmp_path / "problem.dsa.json"
    solution = tmp_path / "solution.dsa.solution.json"
    problem.write_text(
        json.dumps(
            {
                "problem": {
                    "buffers": [
                        {"id": 0, "size": 64},
                        {"id": 1, "size": 64},
                        {"id": 2, "size": 80},
                        {"id": 3, "size": 32},
                    ]
                }
            }
        )
    )
    solution.write_text(
        json.dumps(
            {
                "placements": [
                    {"buffer": 0, "pool": 1, "offset": 0},
                    {"buffer": 1, "pool": 1, "offset": 32},
                    {"buffer": 2, "pool": 1, "offset": 0},
                    {"buffer": 3, "pool": 1, "offset": 32},
                ]
            }
        )
    )
    pair_template = {
        "promoted_to_dsa_penalty": True,
        "distance_zero_schedule_edges": [],
        "loop_carried_schedule_edges": [],
        "estimated_sync_endpoint_executions": 0,
        "critical_path_weight_cycles": 0,
        "unit_cost": 1,
    }
    scores = {
        "schema_version": 2,
        "distance_zero_edges": [],
        "loop_recurrence_edges": [],
        "penalty_pair_weights": [
            {**pair_template, "first_buffer": 0, "second_buffer": 1},
            {**pair_template, "first_buffer": 2, "second_buffer": 3},
        ],
    }

    result = dsa_schedule_model.score_realized_reuse(problem, solution, scores)

    assert result["realized_pair_count"] == 2
    assert result["canonical_physical_reuse_group_count"] == 2
    assert {tuple(group["overlap_range"]) for group in result["canonical_physical_reuse_groups"]} == {
        (32, 64)
    }


def test_realized_reuse_rejects_duplicate_placements(tmp_path):
    problem = tmp_path / "problem.dsa.json"
    solution = tmp_path / "solution.json"
    problem.write_text(
        json.dumps(
            {
                "problem": {
                    "buffers": [
                        {"id": 0, "size": 64},
                        {"id": 1, "size": 64},
                    ]
                }
            }
        )
    )
    solution.write_text(
        json.dumps(
            {
                "placements": [
                    {"buffer": 0, "pool": 0, "offset": 0},
                    {"buffer": 0, "pool": 0, "offset": 64},
                    {"buffer": 1, "pool": 0, "offset": 128},
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate solution placement"):
        dsa_schedule_model.score_realized_reuse(
            problem,
            solution,
            {
                "schema_version": 2,
                "penalty_pair_weights": [
                    {
                        "first_buffer": 0,
                        "second_buffer": 1,
                        "promoted_to_dsa_penalty": True,
                        "unit_cost": 1,
                        "critical_path_weight_cycles": 10,
                    }
                ],
            },
        )


def test_realized_reuse_rejects_legacy_candidate_schema(tmp_path):
    problem = tmp_path / "problem.dsa.json"
    solution = tmp_path / "solution.json"
    problem.write_text(json.dumps({"problem": {"buffers": []}}))
    solution.write_text(json.dumps({"placements": []}))

    with pytest.raises(ValueError, match="expected schema_version=2, got 1"):
        dsa_schedule_model.score_realized_reuse(
            problem,
            solution,
            {"schema_version": 1, "penalty_pair_weights": []},
        )


def test_score_uses_maximum_branch_arm_instead_of_serializing_arms():
    record = {
        "schema_version": 1,
        "function": "branch_kernel",
        "status": "analyzed",
        "nodes": [
            _operation(0, "PIPE_V", "pto.tadd"),
            {"id": 1, "kind": "branch", "branch_kind": "IF_BEGIN", "begin": 1, "branch": 3, "end": 5},
            {**_operation(2, "PIPE_V", "pto.tadd"), "branch_stack": [1]},
            {"id": 3, "kind": "branch", "branch_kind": "ELSE_BEGIN", "begin": 1, "branch": 3, "end": 5},
            {**_operation(4, "PIPE_V", "pto.texp"), "branch_stack": [3]},
            {"id": 5, "kind": "branch", "branch_kind": "IF_END", "begin": 1, "branch": 3, "end": 5},
            _operation(6, "PIPE_V", "pto.tadd"),
        ],
        # This is the incorrect v0.57 exporter chain. The scorer must rebuild
        # structured issue order rather than serialize node 2 before node 4.
        "stream_edges": [
            {"source": 0, "target": 2, "pipe": "PIPE_V"},
            {"source": 2, "target": 4, "pipe": "PIPE_V"},
            {"source": 4, "target": 6, "pipe": "PIPE_V"},
        ],
        "sync_edges": [],
    }
    model = _ten_cycle_model()
    model.operation_cycles["PIPE_V:TEXP"] = 20.0

    result = dsa_schedule_model.score_schedule(record, model)

    assert result["baseline_makespan_cycles"] == 40.0
    assert result["full_makespan_cycles"] == 40.0
    assert result["control_flow_graph_version"] == "per_pipe_structured_control_v1"


def test_branch_boundary_sync_joins_source_and_target_pipes():
    record = {
        "schema_version": 1,
        "function": "branch_sync_kernel",
        "status": "analyzed",
        "nodes": [
            _operation(0, "PIPE_MTE2", "pto.tload"),
            {"id": 1, "kind": "branch", "branch_kind": "IF_BEGIN", "begin": 1, "branch": 3, "end": 5},
            {**_operation(2, "PIPE_V", "pto.tadd"), "branch_stack": [1]},
            {"id": 3, "kind": "branch", "branch_kind": "ELSE_BEGIN", "begin": 1, "branch": 3, "end": 5},
            {**_operation(4, "PIPE_V", "pto.texp"), "branch_stack": [3]},
            {"id": 5, "kind": "branch", "branch_kind": "IF_END", "begin": 1, "branch": 3, "end": 5},
            _operation(6, "PIPE_MTE2", "pto.tload"),
        ],
        "stream_edges": [],
        "sync_edges": [
            {
                "source": 0,
                "target": 1,
                "group": 0,
                "src_pipe": "PIPE_MTE2",
                "dst_pipe": "PIPE_V",
                "loop_carried": False,
            },
            {
                "source": 5,
                "target": 6,
                "group": 1,
                "src_pipe": "PIPE_V",
                "dst_pipe": "PIPE_MTE2",
                "loop_carried": False,
            },
        ],
    }
    model = _ten_cycle_model()
    model.operation_cycles["PIPE_V:TEXP"] = 20.0

    result = dsa_schedule_model.score_schedule(record, model)

    assert result["baseline_makespan_cycles"] == 20.0
    assert result["full_makespan_cycles"] == 40.0
    assert result["latency_graph_complete"] is True
    assert result["excluded_non_operation_sync_edges"] == 0


def test_static_loop_qualification_accepts_bounded_loops_and_structured_branches():
    straight = dsa_schedule_model.classify_static_schedule(_record())
    assert straight == {
        "policy": "structured_branch_static_loop_v2",
        "eligible": True,
        "status": "STATIC_SCHEDULE",
        "operation_count": 4,
        "branch_node_count": 0,
        "loop_node_count": 0,
        "dynamic_loop_node_count": 0,
        "branch_node_ids": [],
        "loop_node_ids": [],
        "dynamic_loop_node_ids": [],
    }

    with_static_loop = _record()
    with_static_loop["nodes"].extend(
        [
            {"id": 11, "kind": "loop", "loop_kind": "LOOP_BEGIN", "static_trip_count": 4},
            {"id": 12, "kind": "loop", "loop_kind": "LOOP_END", "static_trip_count": None},
        ]
    )
    eligible = dsa_schedule_model.classify_static_schedule(with_static_loop)
    assert eligible["eligible"] is True
    assert eligible["status"] == "STATIC_SCHEDULE"
    assert eligible["loop_node_ids"] == [11, 12]
    assert eligible["dynamic_loop_node_ids"] == []

    with_branch = _record()
    with_branch["nodes"].extend(
        [
            {"id": 10, "kind": "branch", "branch_kind": "IF_BEGIN"},
            {"id": 11, "kind": "branch", "branch_kind": "IF_END"},
        ]
    )
    branch_eligible = dsa_schedule_model.classify_static_schedule(with_branch)
    assert branch_eligible["eligible"] is True
    assert branch_eligible["status"] == "STATIC_BRANCH_SCHEDULE"
    assert branch_eligible["branch_node_ids"] == [10, 11]

    with_dynamic_loop = _record()
    with_dynamic_loop["nodes"].append(
        {"id": 13, "kind": "loop", "loop_kind": "LOOP_BEGIN", "static_trip_count": None}
    )
    dynamic_excluded = dsa_schedule_model.classify_static_schedule(with_dynamic_loop)
    assert dynamic_excluded["eligible"] is False
    assert dynamic_excluded["status"] == "DYNAMIC_LOOP_EXCLUDED"
    assert dynamic_excluded["dynamic_loop_node_ids"] == [13]


def test_qualify_command_is_timing_blind_and_hashes_sources(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps(_record()) + "\n")
    output = tmp_path / "qualification.json"

    assert dsa_schedule_model.main(["qualify", str(schedule), "-o", str(output)]) == 0

    result = json.loads(output.read_text())
    assert result["selection_policy"] == "structured_branch_static_loop_v2"
    assert result["timing_blind"] is True
    assert result["schedule_count"] == 1
    assert result["eligible_count"] == 1
    assert result["schedules"][0]["function"] == "kernel"
    assert result["schedules"][0]["status"] == "STATIC_SCHEDULE"
    assert len(result["schedules"][0]["source_sha256"]) == 64


def test_calibrate_uses_complete_signatures_instead_of_family_medians(tmp_path):
    metrics = tmp_path / "instr_metrics.json"
    small = {
        "pipe": "PIPE_MTE2",
        "operation": "TLOAD",
        "dtype": "fp32",
        "rows": 1,
        "cols": 32,
        "work_bytes": 128,
        "operand_types": ["!pto.partition_tensor_view<1x32xf32>"],
        "result_types": ["!pto.tile_buf<vec, 1x32xf32, valid=?x?>"],
        "attributes": {},
        "operand_constants": [None],
    }
    large = {**small, "rows": 16, "work_bytes": 2048}
    large["operand_types"] = ["!pto.partition_tensor_view<16x32xf32>"]
    large["result_types"] = ["!pto.tile_buf<vec, 16x32xf32, valid=?x?>"]
    metrics.write_text(
        json.dumps(
            {
                "instructions": {
                    "core0": [
                        {"pipe": "MTE2", "cycles": 30, "operation_signature": small},
                        {"pipe": "MTE2", "cycles": 34, "operation_signature": small},
                    ],
                    "core1": [{"pipe": "MTE2", "cycles": 1672, "operation_signature": large}],
                }
            }
        )
    )

    model = dsa_schedule_model.calibrate_from_metrics([metrics])

    assert model.calibration_status == "simulator_complete_signature_medians"
    assert model.operation_cycles == {}
    assert model.operation_signature_cycles[dsa_schedule_model._operation_signature_key(small)] == 32.0
    assert model.operation_signature_cycles[dsa_schedule_model._operation_signature_key(large)] == 1672.0
    assert model.pipe_parameters["PIPE_MTE2"].minimum_cycles == 34.0
    assert model.calibration_sources == [str(metrics)]


def test_calibrate_rejects_family_only_samples(tmp_path):
    metrics = tmp_path / "instr_metrics.json"
    metrics.write_text(
        json.dumps({"instructions": {"core0": [{"pipe": "MTE2", "name": "TLOAD", "cycles": 30}]}})
    )

    with pytest.raises(ValueError, match="complete-signature"):
        dsa_schedule_model.calibrate_from_metrics([metrics])


def test_complete_signature_preserves_modes_but_erases_ssa_names():
    node = {
        "id": 6,
        "kind": "operation",
        "pipe": "PIPE_S",
        "op_name": "pto.tci",
        "defs": [],
        "uses": [],
        "operation": {
            "location": (
                "pto.tci ins(%start : ui32) outs(%indices : "
                "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=512, v_row=?, v_col=?, "
                "blayout=row_major, slayout=none_box, fractal=512, pad=0>) "
                '{descending = false} loc("pypto.access.6")'
            ),
            "operand_types": ["ui32"],
            "result_types": [
                "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=512, v_row=?, v_col=?, "
                "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
            ],
            "operand_constants": [None],
            "attributes": {},
            "static_work_bytes": 2048,
        },
    }

    signature = dsa_schedule_model.operation_duration_signature(node)

    assert "descending = false" in signature["semantic_operation"]
    assert "%start" not in signature["semantic_operation"]
    assert "%indices" not in signature["semantic_operation"]
    assert "pypto.access" not in signature["semantic_operation"]


def test_complete_signature_ignores_native_source_location():
    node = _operation(0, "PIPE_V", "pto.tadd")
    node["operation"] = {
        "location": 'loc("/checkout/model.py":10:4)',
        "operand_types": [],
        "result_types": [],
        "operand_constants": [],
        "attributes": {},
    }

    assert dsa_schedule_model.operation_duration_signature(node)["semantic_operation"] is None


def test_extract_calibration_binds_trace_to_exact_schedule_node(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    trace = tmp_path / "trace.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "metrics.json"
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [
            {
                "id": 0,
                "kind": "operation",
                "pipe": "PIPE_V",
                "op_name": "pto.texpands",
                "defs": [],
                "uses": [],
                "operation": {
                    "location": (
                        "pto.texpands ins(%cst : f32) outs(%tile : "
                        "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, "
                        'blayout=row_major, slayout=none_box, fractal=512, pad=0>) loc("x")'
                    ),
                    "operand_types": ["f32"],
                    "result_types": [
                        "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, "
                        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
                    ],
                    "operand_constants": ["0.0 : f32"],
                    "attributes": {},
                    "static_work_bytes": 128,
                },
            }
        ],
        "stream_edges": [],
        "sync_edges": [],
    }
    prefix = (
        "TEXPANDS(1x32,fp32){pipe=VEC;"
        "tiles=fp32:1x32:loc=0:storage=1x32:b=0:s=0:pad=0:compact=0;scalars=f32:0x0}"
    )
    schedule.write_text(json.dumps(record) + "\n")
    trace.write_text(
        json.dumps(
            [
                {"ph": "X", "name": f"{prefix}:7", "dur": 15},
                {"ph": "X", "name": f"{prefix}:18", "dur": 17},
            ]
        )
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurements": [
                    {
                        "schedule": schedule.name,
                        "function": "kernel",
                        "node_id": 0,
                        "trace": trace.name,
                        "event_name_prefix": prefix,
                    }
                ],
            }
        )
    )

    assert dsa_schedule_model.main(["extract-calibration", str(manifest), "-o", str(output)]) == 0
    metrics = json.loads(output.read_text())
    records = next(iter(metrics["instructions"].values()))
    assert [row["cycles"] for row in records] == [15.0, 17.0]
    assert records[0]["operation_signature"]["operand_constants"] == ["0.0 : f32"]
    assert records[0]["perf_sim_pipe"] == "PIPE_V"

    bad_shape = prefix.replace("1x32", "1x64")
    trace.write_text(json.dumps([{"ph": "X", "name": f"{bad_shape}:1", "dur": 2}]))
    payload = json.loads(manifest.read_text())
    payload["measurements"][0]["event_name_prefix"] = bad_shape
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="work signature differs"):
        dsa_schedule_model.extract_signature_calibration(manifest)

    bad_constant = prefix.replace("f32:0x0", "f32:0x3f800000")
    trace.write_text(json.dumps([{"ph": "X", "name": f"{bad_constant}:1", "dur": 2}]))
    payload["measurements"][0]["event_name_prefix"] = bad_constant
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="event constants differ"):
        dsa_schedule_model.extract_signature_calibration(manifest)


@pytest.mark.parametrize(
    ("operand_dtype", "result_dtype", "round_mode", "round_value"),
    [("bf16", "f32", "ROUND", 2), ("f32", "bf16", "RINT", 1)],
)
def test_perf_sim_header_uses_result_first_for_mixed_dtype_tcvt(
    tmp_path, operand_dtype, result_dtype, round_mode, round_value
):
    def tile(dtype):
        return (
            f"!pto.tile_buf<loc=vec, dtype={dtype}, rows=8, cols=128, v_row=?, v_col=?, "
            "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
        )

    operand = tile(operand_dtype)
    result = tile(result_dtype)
    node = {
        "id": 0,
        "kind": "operation",
        "pipe": "PIPE_V",
        "op_name": "pto.tcvt",
        "defs": [],
        "uses": [],
        "operation": {
            "location": f"pto.tcvt ins(%src {{rmode = #pto<round_mode {round_mode}>}} : {operand}) "
            f"outs(%dst : {result})",
            "operand_types": [operand],
            "result_types": [result],
            "operand_constants": [None],
            "attributes": {},
            "static_work_bytes": 4096,
        },
    }
    dtype_name = {"f32": "fp32", "bf16": "bf16"}
    prefix = (
        f"TCVT(8x128,{dtype_name[result_dtype]}){{pipe=VEC;"
        f"tiles={dtype_name[result_dtype]}:8x128:loc=0:storage=8x128:b=0:s=0:pad=0:compact=0,"
        f"{dtype_name[operand_dtype]}:8x128:loc=0:storage=8x128:b=0:s=0:pad=0:compact=0;"
        f"scalars=enum:{round_value}}}"
    )

    event = dsa_schedule_model._parse_perf_sim_event_prefix(prefix)
    dsa_schedule_model._validate_perf_sim_event_signature(
        event, node, measurement_index=0, manifest=tmp_path / "manifest.json"
    )


def test_perf_sim_reciprocal_requires_exact_tdivs_by_one_lowering():
    tile = (
        "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=?, v_col=?, "
        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )
    node = {
        "id": 0,
        "kind": "operation",
        "pipe": "PIPE_V",
        "op_name": "pto.trecip",
        "defs": [],
        "uses": [],
        "operation": {
            "location": f"pto.trecip ins(%src : {tile}) outs(%dst : {tile})",
            "operand_types": [tile],
            "result_types": [tile],
            "operand_constants": [None],
            "attributes": {},
            "static_work_bytes": 64,
        },
    }

    event = dsa_schedule_model.expected_perf_sim_event_signature(node)
    assert event["operation"] == "TDIVS"
    assert event["scalars"] == ["i:1"]
    assert dsa_schedule_model.operation_duration_signature(node)["operation"] == "TRECIP"


def test_perf_sim_signature_tracks_inline_round_mode_and_dynamic_scalar_indices(tmp_path):
    tile = (
        "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32, v_row=?, v_col=?, "
        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )
    tci = {
        "id": 0,
        "kind": "operation",
        "pipe": "PIPE_S",
        "op_name": "pto.tci",
        "defs": [],
        "uses": [],
        "operation": {
            "location": f"pto.tci ins(%start : ui32) outs(%dst : {tile}) {{descending = false}}",
            "operand_types": ["ui32"],
            "result_types": [tile],
            "operand_constants": [None],
            "attributes": {"descending": False},
            "static_work_bytes": 128,
        },
    }
    event = dsa_schedule_model._parse_perf_sim_event_prefix(
        "TCI(1x32,uint32){pipe=VEC;tiles=uint32:1x32:loc=0:storage=1x32:b=0:s=0:pad=0:compact=0;scalars=u:17}"
    )

    expected = dsa_schedule_model.expected_perf_sim_event_signature(tci)
    assert expected["dtype"] == "u32"
    assert expected["scalars"] == ["integer:*"]
    dsa_schedule_model._validate_perf_sim_event_signature(
        event, tci, measurement_index=0, manifest=tmp_path / "manifest.json"
    )
    wrong_scalar_type = {**event, "scalars": ["f32:0x0"]}
    with pytest.raises(ValueError, match="event constants differ"):
        dsa_schedule_model._validate_perf_sim_event_signature(
            wrong_scalar_type, tci, measurement_index=0, manifest=tmp_path / "manifest.json"
        )

    tcvt = {
        **tci,
        "op_name": "pto.tcvt",
        "pipe": "PIPE_V",
        "operation": {
            "location": (
                "pto.tcvt ins(%src {rmode = #pto<round_mode ROUND>} : "
                "!pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32, v_row=?, v_col=?, "
                "blayout=row_major, slayout=none_box, fractal=512, pad=0>) "
                "outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, "
                "v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)"
            ),
            "operand_types": [
                "!pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32, v_row=?, v_col=?, "
                "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
            ],
            "result_types": [
                "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, "
                "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
            ],
            "operand_constants": [None],
            "attributes": {"round_mode": "ROUND"},
            "static_work_bytes": 128,
        },
    }
    assert dsa_schedule_model.expected_perf_sim_event_signature(tcvt)["scalars"] == ["enum:2"]


def test_perf_sim_signature_records_only_tsetval_offset(tmp_path):
    tile = (
        "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, "
        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )
    node = {
        "id": 0,
        "kind": "operation",
        "pipe": "PIPE_S",
        "op_name": "pto.tsetval",
        "defs": [],
        "uses": [],
        "operation": {
            "location": f"pto.tsetval ins(%offset, %value : index, f32) outs(%dst : {tile})",
            "operand_types": ["index", "f32"],
            "result_types": [tile],
            "operand_constants": [None, None],
            "attributes": {},
            "static_work_bytes": 128,
        },
    }
    event = dsa_schedule_model._parse_perf_sim_event_prefix(
        "TSETVAL(1x32,fp32){pipe=Scalar;"
        "tiles=fp32:1x32:loc=0:storage=1x32:b=0:s=0:pad=0:compact=0;scalars=u:9}"
    )

    assert dsa_schedule_model.expected_perf_sim_event_signature(node)["scalars"] == ["integer:*"]
    dsa_schedule_model._validate_perf_sim_event_signature(
        event, node, measurement_index=0, manifest=tmp_path / "manifest.json"
    )


def test_perf_sim_signature_reorders_multi_source_mrgsort_and_matches_integer_value(tmp_path):
    def tile(cols):
        return (
            f"!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols={cols}, v_row=?, v_col=?, "
            "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
        )

    node = {
        "id": 0,
        "kind": "operation",
        "pipe": "PIPE_V",
        "op_name": "pto.tmrgsort",
        "defs": [],
        "uses": [],
        "operation": {
            "location": (
                f"pto.tmrgsort ins(%src0, %src1, %tmp : {tile(64)}, {tile(64)}, {tile(128)}) "
                f"outs(%dst : {tile(128)}) {{exhausted = false}}"
            ),
            "operand_types": [tile(64), tile(64), tile(128)],
            "result_types": [tile(128)],
            "operand_constants": [None, None, None],
            "attributes": {"exhausted": False},
            "static_work_bytes": 512,
        },
    }
    event = dsa_schedule_model._parse_perf_sim_event_prefix(
        "TMRGSORT(1x128,fp32){pipe=VEC;"
        "tiles=fp32:1x128:loc=0:storage=1x128:b=0:s=0:pad=0:compact=0,"
        "fp32:1x128:loc=0:storage=1x128:b=0:s=0:pad=0:compact=0,"
        "fp32:1x64:loc=0:storage=1x64:b=0:s=0:pad=0:compact=0,"
        "fp32:1x64:loc=0:storage=1x64:b=0:s=0:pad=0:compact=0}"
    )
    dsa_schedule_model._validate_perf_sim_event_signature(
        event, node, measurement_index=0, manifest=tmp_path / "manifest.json"
    )

    block_node = {
        **node,
        "operation": {
            **node["operation"],
            "location": f"pto.tmrgsort ins(%src, %len : {tile(4096)}, i32) outs(%dst : {tile(4096)})",
            "operand_types": [tile(4096), "i32"],
            "result_types": [tile(4096)],
            "operand_constants": [None, "64 : i32"],
            "attributes": {},
            "static_work_bytes": 16384,
        },
    }
    block_event = dsa_schedule_model._parse_perf_sim_event_prefix(
        "TMRGSORT(1x4096,fp32){pipe=VEC;"
        "tiles=fp32:1x4096:loc=0:storage=1x4096:b=0:s=0:pad=0:compact=0,"
        "fp32:1x4096:loc=0:storage=1x4096:b=0:s=0:pad=0:compact=0;scalars=u:64}"
    )
    dsa_schedule_model._validate_perf_sim_event_signature(
        block_event, block_node, measurement_index=0, manifest=tmp_path / "manifest.json"
    )


@pytest.mark.parametrize("mismatch_reason", [None, "invented_reason"])
def test_extract_calibration_rejects_unallowlisted_pipe_mismatch(tmp_path, mismatch_reason):
    schedule = tmp_path / "schedule.jsonl"
    trace = tmp_path / "trace.json"
    manifest = tmp_path / "manifest.json"
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [
            {
                "id": 0,
                "kind": "operation",
                "pipe": "PIPE_S",
                "op_name": "pto.tci",
                "defs": [],
                "uses": [],
                "operation": {
                    "location": (
                        "pto.tci ins(%start : ui32) outs(%tile : "
                        "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32, v_row=?, v_col=?, "
                        "blayout=row_major, slayout=none_box, fractal=512, pad=0>) {descending = false}"
                    ),
                    "operand_types": ["ui32"],
                    "result_types": [
                        "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32, v_row=?, v_col=?, "
                        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
                    ],
                    "operand_constants": [None],
                    "attributes": {"descending": False},
                    "static_work_bytes": 128,
                },
            }
        ],
        "stream_edges": [],
        "sync_edges": [],
    }
    prefix = (
        "TCI(1x32,uint32){pipe=VEC;tiles=uint32:1x32:loc=0:storage=1x32:b=0:s=0:pad=0:compact=0;scalars=u:0}"
    )
    schedule.write_text(json.dumps(record) + "\n")
    trace.write_text(json.dumps([{"ph": "X", "name": f"{prefix}:1", "dur": 2}]))
    measurement = {
        "schedule": schedule.name,
        "node_id": 0,
        "trace": trace.name,
        "event_name_prefix": prefix,
    }
    if mismatch_reason is not None:
        measurement["pipe_mismatch_reason"] = mismatch_reason
    manifest.write_text(json.dumps({"schema_version": 1, "measurements": [measurement]}))

    with pytest.raises(ValueError, match="without an exact allowlisted exception"):
        dsa_schedule_model.extract_signature_calibration(manifest)


def test_extract_calibration_accepts_exact_tci_pipe_exception(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    trace = tmp_path / "trace.json"
    manifest = tmp_path / "manifest.json"
    tile = (
        "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32, v_row=?, v_col=?, "
        "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [
            {
                "id": 0,
                "kind": "operation",
                "pipe": "PIPE_S",
                "op_name": "pto.tci",
                "defs": [],
                "uses": [],
                "operation": {
                    "location": f"pto.tci ins(%start : ui32) outs(%tile : {tile}) {{descending = false}}",
                    "operand_types": ["ui32"],
                    "result_types": [tile],
                    "operand_constants": [None],
                    "attributes": {"descending": False},
                    "static_work_bytes": 128,
                },
            }
        ],
        "stream_edges": [],
        "sync_edges": [],
    }
    prefix = (
        "TCI(1x32,uint32){pipe=VEC;tiles=uint32:1x32:loc=0:storage=1x32:b=0:s=0:pad=0:compact=0;scalars=u:0}"
    )
    schedule.write_text(json.dumps(record) + "\n")
    trace.write_text(json.dumps([{"ph": "X", "name": f"{prefix}:1", "dur": 2}]))
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurements": [
                    {
                        "schedule": schedule.name,
                        "node_id": 0,
                        "trace": trace.name,
                        "event_name_prefix": prefix,
                        "pipe_mismatch_reason": "ptoas_v057_tci_schedule_pipe",
                    }
                ],
            }
        )
    )

    result = dsa_schedule_model.extract_signature_calibration(manifest)
    assert next(iter(result["instructions"].values()))[0]["pipe_mismatch_reason"] == (
        "ptoas_v057_tci_schedule_pipe"
    )


def test_accumulating_matmul_keeps_full_calibration_key():
    assert dsa_schedule_model._operation_key("PIPE_M", "pto.tmatmul.acc") == "PIPE_M:TMATMUL_ACC"


def test_load_and_freeze_predictions_are_content_addressed(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps(_record()) + "\n")

    records = dsa_schedule_model.load_schedule_graphs(schedule)
    predictions = {"kernel": dsa_schedule_model.score_schedule(records["kernel"], _ten_cycle_model())}
    frozen = dsa_schedule_model.freeze_predictions(predictions, cohort="holdout-v0", source_paths=[schedule])

    assert frozen["cohort"] == "holdout-v0"
    assert frozen["frozen_before_device_timing"] is True
    assert frozen["freeze_context"] == "prospective_holdout"
    assert len(frozen["prediction_sha256"]) == 64
    assert frozen["schedule_sources"][0]["path"] == str(schedule)
    assert len(frozen["schedule_sources"][0]["sha256"]) == 64


def test_freeze_predictions_labels_retrospective_join_honestly(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps(_record()) + "\n")

    frozen = dsa_schedule_model.freeze_predictions(
        {"kernel": {"status": "MODEL_ELIGIBLE"}},
        cohort="existing-measurements-v1",
        source_paths=[schedule],
        frozen_before_device_timing=False,
    )

    assert frozen["frozen_before_device_timing"] is False
    assert frozen["freeze_context"] == "retrospective_before_timing_join"


def test_main_scores_and_writes_frozen_record(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    output = tmp_path / "frozen.json"
    model = tmp_path / "duration-model.json"
    schedule.write_text(json.dumps(_record()) + "\n")
    model.write_text(json.dumps(_ten_cycle_model().to_json()))

    assert (
        dsa_schedule_model.main(
            [
                "score",
                str(schedule),
                "--model",
                str(model),
                "--freeze-cohort",
                "held-out",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["cohort"] == "held-out"
    prediction = result["predictions"][f"{schedule}:kernel"]
    assert prediction["calibration_status"] == "test"


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
    assert record["sync_groups"][0]["operations"][0]["event_ids"] == [0]
    assert record["sync_groups"][0]["operations"][0]["useless"] is False


def test_import_legacy_debug_preserves_but_does_not_activate_useless_sync():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tload [PIPE_MTE2]
  POST: set_flag <PIPE_MTE2 -> PIPE_V> idx=3 useless eventIds=[0,1]
[   1] COMPOUND pto.tadd [PIPE_V]
  PRE : wait_flag <PIPE_MTE2 -> PIPE_V> idx=3 eventIds=[0,1]
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")
    group = record["sync_groups"][0]

    assert group["operations"][0]["useless"] is True
    assert group["operations"][0]["event_ids"] == [0, 1]
    assert record["sync_edges"] == []
    summary = dsa_schedule_model._pre_codegen_sync_record_summary(record)
    assert summary["record_count"] == 2
    assert summary["active_record_site_count"] == 1
    assert summary["useless_record_site_count"] == 1


def test_import_legacy_debug_exports_barrier_dependency_edge():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tadds [PIPE_V]
[   1] COMPOUND pto.tmul [PIPE_V]
  PRE : pipe_barrier <PIPE_V -> PIPE_V> idx=3 depNode=0
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")

    assert record["sync_edges"] == [
        {
            "source": 0,
            "target": 1,
            "group": 0,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_V",
            "loop_carried": False,
            "root_buffers": [],
        }
    ]
    assert record["sync_groups"][0]["operations"][0]["dependency_node"] == 0
    assert record["export_limitations"]["barrier_dependency_nodes_missing"] == 0


def test_import_legacy_debug_marks_barrier_without_dependency_incomplete():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tadds [PIPE_V]
[   1] COMPOUND pto.tmul [PIPE_V]
  PRE : pipe_barrier <PIPE_V -> PIPE_V> idx=3
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")

    assert record["sync_edges"] == []
    assert record["export_limitations"]["barrier_dependency_nodes_missing"] == 1


def test_import_legacy_debug_classifies_loop_carried_barrier_dependency():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] LOOP LOOP_BEGIN (begin=0, end=3)
  [   1] COMPOUND pto.tadds [PIPE_V]
    PRE : pipe_barrier <PIPE_V -> PIPE_V> idx=3 depNode=2 forEnd=3
  [   2] COMPOUND pto.tmul [PIPE_V]
[   3] LOOP LOOP_END (begin=0, end=3)
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")

    assert record["sync_groups"][0]["loop_carried"] is True
    assert record["sync_edges"] == [
        {
            "source": 2,
            "target": 1,
            "group": 0,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_V",
            "loop_carried": True,
            "root_buffers": [],
        }
    ]


def test_import_legacy_debug_does_not_treat_loop_boundary_lifecycle_as_recurrence():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.texpands [PIPE_V]
  PRE : set_flag <PIPE_V -> PIPE_MTE2> idx=3 forEnd=3 eventIds=[0]
[   1] LOOP LOOP_BEGIN (begin=1, end=3)
  [   2] COMPOUND pto.tload [PIPE_MTE2]
[   3] LOOP LOOP_END (begin=1, end=3)
  POST: wait_flag <PIPE_V -> PIPE_MTE2> idx=3 forEnd=3 eventIds=[0]
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")

    assert record["sync_groups"][0]["loop_carried"] is False
    assert record["sync_edges"] == [
        {
            "source": 0,
            "target": 3,
            "group": 0,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_MTE2",
            "loop_carried": False,
            "root_buffers": [],
        }
    ]


def test_import_legacy_debug_preserves_genuine_loop_carried_recurrence():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] LOOP LOOP_BEGIN (begin=0, end=3)
  [   1] COMPOUND pto.tload [PIPE_MTE2]
    PRE : wait_flag <PIPE_V -> PIPE_MTE2> idx=3 depNode=2 forEnd=3 eventIds=[0]
  [   2] COMPOUND pto.texpands [PIPE_V]
    POST: set_flag <PIPE_V -> PIPE_MTE2> idx=3 depNode=2 forEnd=3 eventIds=[0]
[   3] LOOP LOOP_END (begin=0, end=3)
// ========================================= //
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel")

    assert record["sync_groups"][0]["loop_carried"] is True
    assert record["sync_edges"] == [
        {
            "source": 2,
            "target": 1,
            "group": 0,
            "src_pipe": "PIPE_V",
            "dst_pipe": "PIPE_MTE2",
            "loop_carried": True,
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
[   1] COMPOUND pto.tadds [PIPE_V]
[   2] COMPOUND pto.tstore [PIPE_MTE3]
// ========================================= //
"""
    tile_type = "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128>"
    partition_type = "!pto.partition_tensor_view<8x128xf32>"
    pto = "\n".join(
        [
            "%scale = arith.constant 1.250000e+00 : f32",
            f"%tile = pto.alloc_tile addr = %c0 : {tile_type}",
            f'pto.tload ins(%arg0 : {partition_type}) outs(%tile : {tile_type}) loc("pypto.access.3")',
            f"pto.tadds ins(%tile, %scale : {tile_type}, f32) outs(%tile : {tile_type}) "
            'loc("pypto.access.7")',
            f'pto.tstore ins(%tile : {tile_type}) outs(%arg1 : {partition_type}) loc("pypto.access.9")',
        ]
    )

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert record["export_source"] == ("ptoas_debug_import_v0+pto_access_join_v3+static_loop_bounds_v1")
    assert [node["operation"]["pypto_access_order"] for node in record["nodes"]] == [3, 7, 9]
    assert record["nodes"][0]["operation"]["operand_types"] == ["!pto.partition_tensor_view<8x128xf32>"]
    assert record["nodes"][0]["operation"]["result_types"] == [
        "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128>"
    ]
    assert record["nodes"][0]["operation"]["operand_constants"] == [None]
    assert record["nodes"][1]["operation"]["operand_constants"] == [None, "1.250000e+00 : f32"]
    assert [node["operation"]["static_work_bytes"] for node in record["nodes"]] == [
        4096,
        4096,
        4096,
    ]
    assert record["export_limitations"]["access_provenance_missing"] is False
    assert record["export_limitations"]["operation_types_missing"] == 0
    assert record["export_limitations"]["static_work_sizes_missing"] == 0
    assert record["export_limitations"]["static_loop_bounds_missing"] == 0


def test_import_legacy_debug_preserves_scalar_constant_mutations():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tadds [PIPE_V]
// ========================================= //
"""
    tile_type = "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128>"

    def import_constant(value: str) -> list[str | None]:
        pto = "\n".join(
            [
                f"%scale = arith.constant {value} : f32",
                f"pto.tadds ins(%tile, %scale : {tile_type}, f32) outs(%tile : {tile_type}) "
                'loc("pypto.access.7")',
            ]
        )
        record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)
        return record["nodes"][0]["operation"]["operand_constants"]

    assert import_constant("1.250000e+00") == [None, "1.250000e+00 : f32"]
    assert import_constant("2.500000e+00") == [None, "2.500000e+00 : f32"]


def test_import_legacy_debug_preserves_unsigned_constants_and_inline_attributes():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tci [PIPE_S]
[   1] COMPOUND pto.tcvt [PIPE_V]
// ========================================= //
"""
    ui32_tile = "!pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32>"
    bf16_tile = "!pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32>"
    f32_tile = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>"
    pto = "\n".join(
        [
            "%start = arith.constant 0 : ui32",
            f"pto.tci ins(%start : ui32) outs(%indices : {ui32_tile}) "
            '{descending = false} loc("pypto.access.1")',
            f"pto.tcvt ins(%source {{rmode = #pto<round_mode ROUND>}} : {bf16_tile}) "
            f"outs(%result : {f32_tile}) "
            'loc("pypto.access.2")',
        ]
    )

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert record["nodes"][0]["operation"]["operand_constants"] == ["0 : ui32"]
    assert record["nodes"][0]["operation"]["attributes"] == {"descending": False}
    assert record["nodes"][1]["operation"]["attributes"] == {"round_mode": "ROUND"}


def test_import_legacy_debug_accepts_semantic_accumulating_matmul_name():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tmatmul.acc [PIPE_M]
// ========================================= //
"""
    acc_type = "!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32>"
    left_type = "!pto.tile_buf<loc=left, dtype=bf16, rows=16, cols=64>"
    right_type = "!pto.tile_buf<loc=right, dtype=bf16, rows=64, cols=32>"
    pto = (
        f"pto.tmatmul ins(%acc, %lhs, %rhs : {acc_type}, {left_type}, {right_type}) "
        f"outs(%acc : {acc_type}) "
        'loc("pypto.access.4")'
    )

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    operation = record["nodes"][0]["operation"]
    assert record["nodes"][0]["op_name"] == "pto.tmatmul.acc"
    assert operation["raw_pto_op_name"] == "pto.tmatmul"
    assert operation["static_work_bytes"] == 4096


@pytest.mark.parametrize(
    ("trace_name", "raw_name"),
    [
        ("pto.tpush", "pto.tpush_to_aiv"),
        ("pto.tpush", "pto.tpush_to_aic"),
        ("pto.tpop", "pto.tpop_from_aiv"),
        ("pto.tpop", "pto.tpop_from_aic"),
    ],
)
def test_import_legacy_debug_accepts_mixed_kernel_operation_names(trace_name, raw_name):
    log = f"""
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND {trace_name} [PIPE_FIX]
// ========================================= //
"""
    tile_type = "!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=128>"
    pto = f'{raw_name}(%tile : {tile_type}) loc("pypto.access.4")'

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    operation = record["nodes"][0]["operation"]
    assert operation["raw_pto_op_name"] == raw_name
    assert operation["pypto_access_order"] == 4


def test_import_legacy_debug_rejects_non_accumulating_matmul_name_mismatch():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tmatmul.acc [PIPE_M]
// ========================================= //
"""
    acc_type = "!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32>"
    left_type = "!pto.tile_buf<loc=left, dtype=bf16, rows=16, cols=64>"
    right_type = "!pto.tile_buf<loc=right, dtype=bf16, rows=64, cols=32>"
    pto = (
        f"pto.tmatmul ins(%lhs, %rhs : {left_type}, {right_type}) outs(%acc : {acc_type}) "
        'loc("pypto.access.4")'
    )

    with pytest.raises(ValueError, match="operation sequence does not match"):
        dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)


def test_import_legacy_debug_extracts_non_ins_outs_operation_types():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.load_scalar [PIPE_S]
// ========================================= //
"""
    pto = '%value = pto.load_scalar %arg0[%index] : !pto.ptr<i32> -> i32 loc("pypto.access.5")'

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    operation = record["nodes"][0]["operation"]
    assert operation["operand_types"] == ["!pto.ptr<i32>"]
    assert operation["result_types"] == ["i32"]
    assert operation["static_work_bytes"] == 0


def test_enrich_native_schedule_preserves_sync_graph_and_joins_legacy_pointer_type():
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [
            {
                "id": 0,
                "kind": "operation",
                "op_name": "pto.load_scalar",
                "pipe": "PIPE_S",
                "loop_stack": [],
                "branch_stack": [],
                "defs": [],
                "uses": [],
                "operation": {"location": 'loc("kernel.pto":1:1)'},
            }
        ],
        "stream_edges": [],
        "sync_edges": [{"source": 0, "target": 0, "group": 7, "loop_carried": False}],
        "sync_groups": [
            {
                "id": 7,
                "src_pipe": "PIPE_S",
                "dst_pipe": "PIPE_S",
                "operations": [{"type": "pipe_barrier", "node": 0, "dependency_node": 0}],
            }
        ],
    }
    pto = "%value = pto.load_scalar %arg0[%index] : <f32, gm> -> f32"

    enriched = dsa_schedule_model.enrich_native_schedule_from_pto(record, pto, pto_source="kernel.pto")

    operation = enriched["nodes"][0]["operation"]
    assert operation["operand_types"] == ["!pto.ptr<f32, gm>"]
    assert operation["result_types"] == ["f32"]
    assert "pypto_access_order" not in operation
    assert enriched["sync_edges"] == record["sync_edges"]
    assert enriched["sync_groups"] == record["sync_groups"]
    assert enriched["raw_pto_operation_join"] == "exact_executable_order_v1"
    assert enriched["export_limitations"]["operation_metadata_missing"] == 0


def test_import_legacy_debug_extracts_scalar_outs_type():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] COMPOUND pto.tgetval [PIPE_S]
// ========================================= //
"""
    tile_type = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>"
    pto = f'%value = pto.tgetval ins(%tile, %index : {tile_type}, index) outs : f32 loc("pypto.access.6")'

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    operation = record["nodes"][0]["operation"]
    assert operation["operand_types"] == [tile_type, "index"]
    assert operation["result_types"] == ["f32"]
    assert operation["static_work_bytes"] == 128


def test_import_legacy_debug_joins_static_raw_pto_loop_bounds():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] LOOP LOOP_BEGIN (begin=0, end=2)
  [   1] COMPOUND pto.tadd [PIPE_V]
[   2] LOOP LOOP_END (begin=0, end=2)
// ========================================= //
"""
    pto = """
%c0_index = arith.constant 0 : index
%c2_index = arith.constant 2 : index
%c32_index = arith.constant 32 : index
scf.for %i = %c0_index to %c32_index step %c2_index {
  pto.tadd ins(%tile, %tile) outs(%tile) loc("pypto.access.7")
}
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert [node["static_trip_count"] for node in record["nodes"] if node["kind"] == "loop"] == [
        16,
        16,
    ]
    assert record["export_limitations"]["static_loop_bounds_missing"] == 0
    assert dsa_schedule_model.classify_static_schedule(record)["status"] == "STATIC_SCHEDULE"


def test_import_legacy_debug_joins_result_producing_static_loop_bounds():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] LOOP LOOP_BEGIN (begin=0, end=2)
  [   1] COMPOUND pto.tadd [PIPE_V]
[   2] LOOP LOOP_END (begin=0, end=2)
// ========================================= //
"""
    pto = """
%c0_index = arith.constant 0 : index
%c2_index = arith.constant 2 : index
%c32_index = arith.constant 32 : index
%result = scf.for %i = %c0_index to %c32_index step %c2_index iter_args(%value = %initial) -> (i32) {
  pto.tadd ins(%tile, %tile) outs(%tile) loc("pypto.access.7")
}
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert [node["static_trip_count"] for node in record["nodes"] if node["kind"] == "loop"] == [
        16,
        16,
    ]
    assert record["export_limitations"]["static_loop_bounds_missing"] == 0


def test_import_legacy_debug_reconstructs_branch_nodes_and_arm_stacks():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] BRANCH IF_BEGIN (begin=0, branch=3, end=5)
  [   1] COMPOUND pto.tadd [PIPE_V]
  [   2] PLACE_HOLDER (parentScopeId=0)
[   3] BRANCH ELSE_BEGIN (begin=0, branch=3, end=5)
  [   4] COMPOUND pto.tmul [PIPE_V]
[   5] BRANCH IF_END (begin=0, branch=3, end=5)
// ========================================= //
"""
    pto = """
func.func @kernel(%condition: i1) {
  scf.if %condition {
    pto.tadd ins(%tile, %tile) outs(%tile) loc("pypto.access.7")
  } else {
    pto.tmul ins(%tile, %tile) outs(%tile) loc("pypto.access.8")
  }
  return
}
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert record["export_limitations"]["branch_nodes_missing"] == 0
    assert [node["branch_kind"] for node in record["nodes"] if node["kind"] == "branch"] == [
        "IF_BEGIN",
        "ELSE_BEGIN",
        "IF_END",
    ]
    assert record["nodes"][1]["branch_stack"] == [0]
    assert record["nodes"][4]["branch_stack"] == [3]
    assert record["nodes"][5]["branch_stack"] == []


def test_import_legacy_debug_preserves_genuinely_dynamic_loop():
    log = """
// === [PTOInsertSync Debug] After EventId Allocation === //
[   0] LOOP LOOP_BEGIN (begin=0, end=2)
  [   1] COMPOUND pto.tadd [PIPE_V]
[   2] LOOP LOOP_END (begin=0, end=2)
// ========================================= //
"""
    pto = """
%c0_index = arith.constant 0 : index
%c1_index = arith.constant 1 : index
scf.for %i = %c0_index to %arg0 step %c1_index {
  pto.tadd ins(%tile, %tile) outs(%tile) loc("pypto.access.7")
}
"""

    record = dsa_schedule_model.import_insert_sync_debug(log, function="kernel", pto_text=pto)

    assert record["export_limitations"]["static_loop_bounds_missing"] == 1
    assert dsa_schedule_model.classify_static_schedule(record)["status"] == "DYNAMIC_LOOP_EXCLUDED"


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
    assert result["comparisons"][0]["signed_marginal_sync_cost_cycles"] == 20.0
    assert result["comparisons"][0]["observed_relative_delta"] == 1.0
    assert result["comparisons"][0]["direction_correct"] is True
    assert result["comparisons"][0]["baseline_exact_duration_coverage"] == 1.0
    assert result["comparisons"][0]["candidate_exact_duration_coverage"] == 1.0
    assert result["comparisons"][0]["baseline_duration_source_counts"] == {"simulator_operation_median": 4}
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


def test_evaluate_arm_manifest_preserves_negative_marginal_sync_cost(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(
        json.dumps(_record(sync_edges=[{"source": 2, "target": 1, "group": 7, "loop_carried": False}])) + "\n"
    )
    candidate.write_text(json.dumps(_record()) + "\n")
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparisons": [
                    {
                        "case": "barrier-removed",
                        "split": "development",
                        "baseline_arm": "D0",
                        "candidate_arm": "O0",
                        "baseline_schedule": baseline.name,
                        "candidate_schedule": candidate.name,
                    }
                ],
            }
        )
    )

    result = dsa_schedule_model.evaluate_arm_manifest(manifest, _ten_cycle_model())

    row = result["comparisons"][0]
    assert row["baseline_cycles"] == 40.0
    assert row["candidate_cycles"] == 20.0
    assert row["signed_marginal_sync_cost_cycles"] == -20.0
    assert row["predicted_direction"] == -1
    assert "negative values are permitted" in result["marginal_cost_metric"]


def test_evaluate_arm_manifest_rejects_incomplete_latency_graph(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    record = _record()
    record["export_limitations"] = {"barrier_dependency_nodes_missing": 1}
    schedule.write_text(json.dumps(record) + "\n")
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparisons": [
                    {
                        "case": "incomplete",
                        "split": "development",
                        "baseline_arm": "D0",
                        "candidate_arm": "O0",
                        "baseline_schedule": schedule.name,
                        "candidate_schedule": schedule.name,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="incomplete baseline latency graph"):
        dsa_schedule_model.evaluate_arm_manifest(manifest, _ten_cycle_model())


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
