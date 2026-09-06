"""Conformance diagnostics must not manufacture allocation identity or latency."""

import json

import pytest
from pypto.tools.dsa_graph_audit import (
    dag_witness,
    frontend_c2v_dependencies,
    orchestration_tensor_dependencies,
    parse_weighted_graph,
    single_writer_raw_dependencies,
)
from pypto.tools.dsa_schedule_model import emit_ptoas_logical_memory_topology


def test_dag_witness_parallel_slack_and_recurrence():
    nodes = {0: {"cycles": 10}, 1: {"cycles": 3}, 2: {"cycles": 2}}
    edges = [dict(source=1, target=2, latency=1), dict(source=2, target=1, distance=1)]
    assert dag_witness(nodes, edges)["longest_path_cycles"] == 10
    edges.append(dict(source=1, target=2, latency=8))
    result = dag_witness(nodes, edges)
    assert result["longest_path_cycles"] == 13
    assert result["critical_path"] == [1, 2]
    assert result["suffix"][1] == 13
    edges.append(dict(source=2, target=1))
    with pytest.raises(ValueError, match="cycle"):
        dag_witness(nodes, edges)


def test_parse_weighted_graph_refuses_partial_or_multiple_graphs():
    text = """KernelScheduleGraph @test nodes=1 dag_edges=0 dependencies=0
  node[0] op=x pipe=P loop_depth=0 duration_cycles=5 duration_resolved=true pypto_access_order=7
"""
    nodes, edges = parse_weighted_graph(text)
    assert dag_witness(nodes, edges)["longest_path_cycles"] == 5
    with pytest.raises(ValueError, match="counts"):
        parse_weighted_graph(text.replace("nodes=1", "nodes=2"))
    with pytest.raises(ValueError, match="one native graph"):
        parse_weighted_graph(text + text)
    with pytest.raises(ValueError, match="unresolved"):
        parse_weighted_graph(text.replace("duration_resolved=true", "duration_resolved=false"))


def test_logical_raw_witness_uses_allocation_identity_and_scope():
    nodes = {0: dict(access=24), 1: dict(access=27)}
    accesses = [
        dict(order=24, mode="write", pool=4, offset=0, size=64, range_known=True),
        dict(order=27, mode="read", pool=4, offset=0, size=64, range_known=True),
    ]
    catalog = [dict(buffer=5, complete=True, accesses=accesses)]
    schedule = {"nodes": [dict(kind="operation", operation=dict(pypto_access_order=n)) for n in (24, 27)]}
    result = single_writer_raw_dependencies(nodes, catalog, schedule)
    assert len(result) == 1
    assert result[0]["allocation"] == 5
    assert (result[0]["source"], result[0]["target"]) == (0, 1)
    # Same offsets in different logical allocations do not establish a RAW.
    separate = [dict(buffer=i, complete=True, accesses=[a]) for i, a in enumerate(accesses)]
    assert single_writer_raw_dependencies(nodes, separate, schedule) == []
    # A conditional writer does not dominate an unconditional later reader.
    schedule["nodes"][0]["branch_stack"] = [3]
    assert single_writer_raw_dependencies(nodes, catalog, schedule) == []
    schedule["nodes"][0]["branch_stack"] = []
    accesses[0]["size"] = 32
    assert single_writer_raw_dependencies(nodes, catalog, schedule) == []


def test_logical_memory_document_preserves_views_and_recurrence(tmp_path):
    record = {
        "function": "views",
        "nodes": [
            dict(id=0, kind="loop", loop_kind="LOOP_BEGIN", begin=0, end=3),
            dict(
                id=1,
                kind="operation",
                op_name="pto.trowmax",
                pipe="PIPE_V",
                loop_stack=[0],
                branch_stack=[],
                operation=dict(pypto_access_order=24),
            ),
            dict(
                id=2,
                kind="operation",
                op_name="pto.tmax",
                pipe="PIPE_V",
                loop_stack=[0],
                branch_stack=[],
                operation=dict(pypto_access_order=27),
            ),
            dict(id=3, kind="loop", loop_kind="LOOP_END", begin=0, end=3),
        ],
    }
    accesses = [
        dict(order=o, mode=m, pool=1, offset=0, size=64, range_known=True)
        for o, m in [(24, "write"), (27, "read")]
    ]
    problem = dict(
        instance="views",
        problem=dict(buffers=[dict(id=5, size=128)]),
        metadata={"allocation_accesses_v1": json.dumps([dict(buffer=5, complete=True, accesses=accesses)])},
    )
    source, graph = tmp_path / "problem.json", tmp_path / "graph.txt"
    source.write_text(json.dumps(problem))
    graph.write_text(
        "KernelScheduleGraph @views nodes=2 dag_edges=0 dependencies=0\n"
        "  node[0] op=pto.trowmax pypto_access_order=24\n"
        "  node[1] op=pto.tmax pypto_access_order=27\n"
    )
    result = emit_ptoas_logical_memory_topology(record, source, graph)
    assert result["contract"] == "ptoas_logical_memory_topology_v1"
    assert {
        (e["source_node"], e["target_node"], e["kind"], e["iteration_distance"]) for e in result["edges"]
    } == {(0, 1, "raw", 0), (0, 0, "waw", 1), (1, 0, "war", 1)}
    # Disjoint ranges in the same logical allocation must not create RAW/WAR.
    accesses[1]["offset"] = 64
    problem["metadata"]["allocation_accesses_v1"] = json.dumps(
        [dict(buffer=5, complete=True, accesses=accesses)]
    )
    source.write_text(json.dumps(problem))
    result = emit_ptoas_logical_memory_topology(record, source, graph)
    assert len(result["edges"]) == 1 and result["edges"][0]["kind"] == "waw"
    accesses[1]["range_known"] = False
    problem["metadata"]["allocation_accesses_v1"] = json.dumps(
        [dict(buffer=5, complete=True, accesses=accesses)]
    )
    source.write_text(json.dumps(problem))
    with pytest.raises(ValueError, match="unresolved access range"):
        emit_ptoas_logical_memory_topology(record, source, graph)
    conservative = emit_ptoas_logical_memory_topology(record, source, graph, conservative_ranges=True)
    assert conservative["range_precision"] == "conservative_allocation_envelope"
    assert conservative["conservative_accesses"] == [dict(allocation=5, access_order=27)]
    assert any(e["kind"] == "raw" for e in conservative["edges"])
    with pytest.raises(ValueError, match="function differs"):
        emit_ptoas_logical_memory_topology(record, source, graph, function="other")


def test_orchestration_dependencies_keep_mixed_group_and_source_sites():
    text = """
    params0.add_input(x);
    params0.add_output(y);
    rt_submit_aiv_task(0, params0);
    params1.add_input(y);
    params1.add_output(z);
    MixedKernels group = {1, 2, 2};
    rt_submit_task(group, params1);
    rt_submit_dummy_task(fence);
    """
    names = {0: "norm", 1: "cube", 2: "vec"}
    result = orchestration_tensor_dependencies(text, names)
    assert result["tasks"][1]["kernels"] == ["cube", "vec", "vec"]
    assert result["dependencies"] == [dict(source=0, target=1, kind="raw", tensor="y")]
    assert result["explicit_dummy_sites"] == 1 and not result["invocation_model_complete"]
    with pytest.raises(ValueError, match="unbound kernel"):
        orchestration_tensor_dependencies(text, {0: "norm"})
    with pytest.raises(ValueError, match="direct bound variable"):
        orchestration_tensor_dependencies(text.replace("add_input(y)", "add_input(y.slice(0))"), names)


def test_mixed_c2v_dependency_requires_named_channel_and_all_sites():
    pto = """func.func @cube() {
    %peer = pto.import_reserved_buffer {name = "slots", peer_func = @vec} -> i32
    pto.aic_initialize_pipe {dir_mask = 1, slot_size = 1024, slot_num = 2}
      (c2v_consumer_buf = %peer : i32)
    pto.tpush_to_aiv(%tile) {split = 0} loc("pypto.access.75")
    }
    func.func @vec() {
    %local = pto.reserve_buffer {name = "slots", size = 2048} -> i32
    pto.aiv_initialize_pipe {dir_mask = 1, slot_size = 1024, slot_num = 2}
      (c2v_consumer_buf = %local : i32)
    %tile = pto.tpop_from_aic {split = 0} loc("pypto.access.57")
    pto.tfree_from_aic {split = 0} loc("pypto.access.59")
    }"""
    schedules, graphs = {}, {}
    for name, accesses in (("cube", (75,)), ("vec", (57, 59))):
        schedules[name] = dict(
            nodes=[dict(kind="operation", operation=dict(pypto_access_order=o)) for o in accesses]
        )
        graphs[name] = f"KernelScheduleGraph @{name} nodes={len(accesses)} dag_edges=0 dependencies=0\n"
        graphs[name] += "".join(
            f"  node[{i}] op=pto.tpush pipe=PIPE_MTE3 loop_depth=0 duration_cycles=1 "
            f"duration_resolved=true pypto_access_order={o}\n"
            for i, o in enumerate(accesses)
        )
    result = frontend_c2v_dependencies(pto, "cube", "vec", schedules, graphs)
    assert [e["kind"] for e in result["edges"]] == ["mixed-c2v-availability", "mixed-slot-release"]
    assert result["edges"][0]["source"]["access_order"] == 75
    assert result["latency_cycles"] is None and not result["invocation_model_complete"]
    with pytest.raises(ValueError, match="slot configurations"):
        frontend_c2v_dependencies(
            pto.replace("slot_num = 2", "slot_num = 3", 1), "cube", "vec", schedules, graphs
        )
    with pytest.raises(ValueError, match="not bound"):
        frontend_c2v_dependencies(
            pto.replace("c2v_consumer_buf = %peer", "c2v_consumer_buf = %unused"),
            "cube",
            "vec",
            schedules,
            graphs,
        )
    with pytest.raises(ValueError, match="missing from the schedule"):
        frontend_c2v_dependencies(
            pto.replace("pypto.access.75", "pypto.access.76"), "cube", "vec", schedules, graphs
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
