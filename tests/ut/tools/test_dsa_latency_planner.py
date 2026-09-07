# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Legality, objective-selection and complete physical-DAG planner controls."""

import copy
import json

import pytest
from pypto.tools import dsa_latency_planner as planner
from pypto.tools import dsa_schedule_model as model


def _document():
    return {
        "schema_version": 1,
        "profile": "pypto_research_v1",
        "instance": "kernel",
        "problem": {
            "pools": [{"id": 1, "name": "UB", "capacity": 32, "reserved_ranges": []}],
            "buffers": [
                {
                    "id": b,
                    "name": f"b{b}",
                    "size": 8,
                    "alignment": 8,
                    "allowed_pools": [1],
                    "live_intervals": [{"lower": 4 * b + 1, "upper": 4 * b + 4}],
                }
                for b in range(3)
            ],
            "constraints": {
                "colocations": [],
                "separations": [],
                "no_partial_overlaps": [],
                "temporal_exclusions": [],
                "pinned_allocations": [],
            },
            "cost_model": {
                "reuse_penalties": [
                    {"first": 0, "second": 1, "cost": 2, "reason": "cross_pipe"},
                    {"first": 1, "second": 2, "cost": 1, "reason": "cross_pipe"},
                ]
            },
        },
    }


def _solution(placements, document=None):
    return {
        "schema_version": 1,
        "profile": "pypto_research_v1",
        "instance": "kernel",
        "problem_fingerprint": planner.problem_fingerprint(document or _document()),
        "metadata": {"solver": "first-fit", "obsolete_objective": "99"},
        "placements": [{"buffer": b, "pool": p, "offset": o} for b, (p, o) in placements.items()],
    }


def test_objective_really_changes_selected_placement():
    document = _document()
    document["problem"]["pools"][0]["capacity"] = 16
    problem = planner.PlacementProblem(document)
    seed = {0: (1, 0), 1: (1, 0), 2: (1, 8)}
    structural = planner.refine_placement(problem, seed, problem.structural_cost)

    # A hazard on a slack path costs zero; a different overlap exposes 100 cycles.
    def latency(p):
        return 100.0 if p[1] == p[2] else 0.0

    timed = planner.refine_placement(problem, seed, latency)
    assert structural["final_score"] == 0
    assert timed["final_score"] == 0
    assert structural["placements"] != timed["placements"]
    assert timed["placements"] == seed
    assert problem.valid(structural["placements"])


def test_cache_and_final_uncached_verification():
    problem = planner.PlacementProblem(_document())
    seed = {b: (1, 0) for b in range(3)}
    calls = []

    def objective(p):
        calls.append(planner.placement_key(p))
        return problem.structural_cost(p)

    result = planner.refine_placement(problem, seed, objective)
    assert result["final_score"] == 0
    assert result["cache_hits"] > 0
    assert len(calls) == result["score_evaluations"] + 1
    assert calls[-1] == planner.placement_key(result["placements"])
    assert all(move["after"] < move["before"] for move in result["accepted_moves"])


def test_budget_returns_legal_incumbent_without_claiming_optimum():
    problem = planner.PlacementProblem(_document())
    seed = {b: (1, 0) for b in range(3)}
    result = planner.refine_placement(problem, seed, problem.structural_cost, max_evaluations=1)
    assert result["status"] == "BUDGET_EXHAUSTED"
    assert result["placements"] == seed
    assert result["score_evaluations"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_objective_rejected(value):
    problem = planner.PlacementProblem(_document())
    with pytest.raises(ValueError, match="invalid placement score"):
        planner.refine_placement(problem, {b: (1, 0) for b in range(3)}, lambda p: value)


def test_unknown_hard_constraint_fails_closed():
    document = _document()
    document["problem"]["constraints"]["unknown"] = []
    with pytest.raises(ValueError, match="unknown hard constraints"):
        planner.PlacementProblem(document)


@pytest.mark.parametrize(
    "mutation",
    ["capacity", "alignment", "pool", "missing", "separation", "lifetime", "partial", "pin", "reserved"],
)
def test_illegal_placements(mutation):
    doc = _document()
    p = {0: (1, 0), 1: (1, 8), 2: (1, 16)}
    constraints = doc["problem"]["constraints"]
    if mutation == "capacity":
        p[2] = (1, 32)
    elif mutation == "alignment":
        p[2] = (1, 17)
    elif mutation == "pool":
        p[2] = (99, 0)
    elif mutation == "missing":
        del p[2]
    elif mutation == "separation":
        constraints["separations"] = [{"first": 0, "second": 1}]
        p[1] = p[0]
    elif mutation == "lifetime":
        doc["problem"]["buffers"][1]["live_intervals"] = [{"lower": 1, "upper": 9}]
        p[1] = p[0]
    elif mutation == "partial":
        constraints["no_partial_overlaps"] = [{"first": 0, "second": 1}]
        doc["problem"]["buffers"][0]["size"] = 16
    elif mutation == "pin":
        constraints["pinned_allocations"] = [
            {"buffer": 0, "pool": 1, "offset": 24, "exclusive_for_all_time": False}
        ]
    else:
        doc["problem"]["pools"][0]["reserved_ranges"] = [{"begin": 8, "end": 16}]
    assert not planner.PlacementProblem(doc).valid(p)


def test_colocation_moves_as_class_and_respects_largest_extent():
    doc = _document()
    doc["problem"]["constraints"]["colocations"] = [{"first": 0, "second": 1}]
    doc["problem"]["buffers"][0]["size"] = 16
    doc["problem"]["buffers"][1]["live_intervals"] = [{"lower": 1, "upper": 20}]
    doc["problem"]["buffers"][2]["live_intervals"] = [{"lower": 15, "upper": 25}]
    problem = planner.PlacementProblem(doc)
    assert not problem.valid({0: (1, 0), 1: (1, 0), 2: (1, 8)})
    seed = {0: (1, 0), 1: (1, 0), 2: (1, 16)}
    assert problem.valid(seed)
    for candidate in problem.candidates(seed):
        assert candidate[0] == candidate[1]


def test_temporal_exclusion_and_exclusive_pin():
    doc = _document()
    doc["problem"]["buffers"][1]["live_intervals"] = [{"lower": 1, "upper": 8}]
    doc["problem"]["constraints"]["temporal_exclusions"] = [{"first": 0, "second": 1}]
    seed = {0: (1, 0), 1: (1, 0), 2: (1, 8)}
    assert planner.PlacementProblem(doc).valid(seed)
    doc["problem"]["constraints"]["pinned_allocations"] = [
        {"buffer": 0, "pool": 1, "offset": 0, "exclusive_for_all_time": True}
    ]
    assert not planner.PlacementProblem(doc).valid(seed)


def test_identity_and_replay_metadata():
    problem = planner.PlacementProblem(_document())
    seed = _solution({b: (1, 0) for b in range(3)})
    decoded = problem.read_solution(seed)
    output = planner.solution_document(seed, decoded, "dsa-rp-latency-relocation")
    assert output["problem_fingerprint"] == seed["problem_fingerprint"]
    assert output["metadata"] == {"solver": "dsa-rp-latency-relocation"}
    assert seed["metadata"]["obsolete_objective"] == "99"
    seed["placements"].append(seed["placements"][0])
    with pytest.raises(ValueError, match="duplicate"):
        problem.read_solution(seed)


def _physical_fixture(tmp_path):
    doc = _document()
    doc["problem"]["buffers"] = doc["problem"]["buffers"][:2]
    doc["problem"]["cost_model"]["reuse_penalties"] = []  # Not a candidate-catalog test.
    doc["metadata"] = {
        "allocation_accesses_v1": json.dumps(
            [
                {
                    "buffer": 0,
                    "complete": True,
                    "accesses": [
                        {"order": 0, "pool": 1, "mode": "write"},
                        {"order": 1, "pool": 1, "mode": "read"},
                    ],
                },
                {
                    "buffer": 1,
                    "complete": True,
                    "accesses": [
                        {"order": 2, "pool": 1, "mode": "write"},
                        {"order": 3, "pool": 1, "mode": "read"},
                    ],
                },
            ]
        )
    }
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [],
        "stream_edges": [],
        "sync_edges": [],
    }
    for i, pipe in enumerate(("PIPE_V", "PIPE_V", "PIPE_MTE2", "PIPE_MTE2")):
        memory = {"root": f"%allocation{i // 2}", "scope": "UB"}
        record["nodes"].append(
            {
                "id": i,
                "kind": "operation",
                "op_name": "pto.tadd" if i < 2 else "pto.tload",
                "pipe": pipe,
                "loop_stack": [],
                "branch_stack": [],
                "defs": [memory] if i % 2 == 0 else [],
                "uses": [memory] if i % 2 else [],
                "operation": {"pypto_access_order": i},
            }
        )
    graph = tmp_path / "graph.txt"
    graph.write_text(
        "KernelScheduleGraph @kernel nodes=4 dag_edges=2 dependencies=2\n"
        + "".join(
            f"  node[{i}] op={n['op_name']} pypto_access_order={i}\n" for i, n in enumerate(record["nodes"])
        )
    )
    problem = tmp_path / "problem.json"
    problem.write_text(json.dumps(doc))
    durations = model.DurationModel(
        sync_latency_cycles=8,
        calibration_status="test",
        operation_cycles={"PIPE_V:TADD": 10, "PIPE_MTE2:TLOAD": 10},
    )
    return doc, record, durations, problem, graph


def test_real_dag_oracle_changes_search_with_empty_penalty_catalog(tmp_path):
    doc, record, durations, problem_path, graph = _physical_fixture(tmp_path)
    seed = {0: (1, 0), 1: (1, 0)}
    oracle = planner.LatencyObjective(
        record, durations, problem_path, graph, _solution(seed, doc), tmp_path / "work"
    )
    original = copy.deepcopy(record)
    assert oracle(seed) == 28  # 20-cycle parallel paths become 40 + 8.
    problem = planner.PlacementProblem(doc)
    result = planner.refine_placement(problem, seed, oracle, max_evaluations=16)
    assert result["initial_score"] == 28
    assert result["final_score"] == 0
    assert result["placements"][0] != result["placements"][1]
    assert result["accepted_moves"]
    assert record == original


def test_physical_dag_scores_combined_edges_not_sum_of_pairs(tmp_path):
    doc, record, durations, problem_path, graph = _physical_fixture(tmp_path)
    doc["problem"]["buffers"].append(_document()["problem"]["buffers"][2])
    accesses = json.loads(doc["metadata"]["allocation_accesses_v1"])
    accesses.append(
        {
            "buffer": 2,
            "complete": True,
            "accesses": [{"order": 4, "pool": 1, "mode": "write"}, {"order": 5, "pool": 1, "mode": "read"}],
        }
    )
    doc["metadata"]["allocation_accesses_v1"] = json.dumps(accesses)
    for i in (4, 5):
        memory = {"root": "%allocation2", "scope": "UB"}
        record["nodes"].append(
            {
                "id": i,
                "kind": "operation",
                "op_name": "pto.tadd",
                "pipe": "PIPE_V",
                "loop_stack": [],
                "branch_stack": [],
                "defs": [memory] if i == 4 else [],
                "uses": [memory] if i == 5 else [],
                "operation": {"pypto_access_order": i},
            }
        )
    problem_path.write_text(json.dumps(doc))
    graph.write_text(
        "KernelScheduleGraph @kernel nodes=6 dag_edges=3 dependencies=3\n"
        + "".join(
            f"  node[{i}] op={n['op_name']} pypto_access_order={i}\n" for i, n in enumerate(record["nodes"])
        )
    )
    combined = {0: (1, 0), 1: (1, 0), 2: (1, 0)}
    objective = planner.LatencyObjective(
        record, durations, problem_path, graph, _solution(combined, doc), tmp_path / "oracle"
    )
    first = objective({0: (1, 0), 1: (1, 0), 2: (1, 8)})
    second = objective({0: (1, 8), 1: (1, 0), 2: (1, 0)})
    assert first == second == 8
    assert objective(combined) == 36
    assert objective(combined) != first + second


def test_runtime_profile_cannot_be_used_as_planner_input(tmp_path):
    doc, record, durations, problem_path, graph = _physical_fixture(tmp_path)
    record["runtime_branch_profile"] = {"captured": True}
    with pytest.raises(ValueError, match="analysis-only"):
        planner.LatencyObjective(
            record, durations, problem_path, graph, _solution({0: (1, 0), 1: (1, 8)}, doc), tmp_path / "work"
        )


def test_seed_semantic_drift_is_rejected():
    doc = _document()
    seed = _solution({b: (1, 0) for b in range(3)})
    doc["problem"]["buffers"][0]["size"] *= 2
    with pytest.raises(ValueError, match="actual problem semantics"):
        planner.PlacementProblem(doc).read_solution(seed)


def test_frozen_graph_drift_is_rejected(tmp_path):
    doc, record, durations, problem_path, graph = _physical_fixture(tmp_path)
    seed = {0: (1, 0), 1: (1, 0)}
    objective = planner.LatencyObjective(
        record, durations, problem_path, graph, _solution(seed, doc), tmp_path / "work"
    )
    graph.write_text(graph.read_text().replace("@kernel", "@different"))
    with pytest.raises(ValueError, match="changed during search"):
        objective(seed)


def test_pipeline_members_are_fixed_without_blocking_other_moves():
    doc = _document()
    doc["problem"]["pypto_structure"] = {
        "alias_classes": [],
        "pipeline_groups": [{"members": [{"buffer": 0}]}],
    }
    problem = planner.PlacementProblem(doc)
    seed = {b: (1, 0) for b in range(3)}
    candidates = list(problem.candidates(seed))
    assert candidates
    assert all(candidate[0] == seed[0] for candidate in candidates)


def test_cli_structural_emits_replay_solution(tmp_path):
    problem = tmp_path / "problem.json"
    seed = tmp_path / "seed.json"
    out = tmp_path / "out"
    problem.write_text(json.dumps(_document()))
    seed.write_text(json.dumps(_solution({b: (1, 0) for b in range(3)})))
    assert (
        planner.main(
            [
                "--problem",
                str(problem),
                "--seed-solution",
                str(seed),
                "--objective",
                "structural",
                "--output-root",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads((out / "search.json").read_text())
    assert result["final_score"] == 0
    assert planner.PlacementProblem(_document()).read_solution(
        json.loads((out / "solution.json").read_text())
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
