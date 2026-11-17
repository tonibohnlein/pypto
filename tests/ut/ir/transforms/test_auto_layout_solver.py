# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate greedy and treewidth-DP layout solvers against an exact oracle."""

import itertools
import random
from dataclasses import replace

import pytest
from pypto import DataType, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir import auto_layout_bench
from pypto.ir.auto_layout import (
    COL_MAJOR_FLAT,
    ROW_MAJOR_FLAT,
    CostBreakdown,
    LayoutAssignment,
    LayoutChoice,
    LayoutCostModel,
    LayoutOperationNode,
    LayoutProblemInstance,
    LayoutValueNode,
    OperatorConfig,
    apply_layout_assignment,
    enumerate_assignments,
    evaluate_assignment,
    extract_layout_instance,
    make_solver_auto_layout_pass,
    run_post_layout_passes,
    run_pre_layout_passes,
)
from pypto.ir.auto_layout_solver import (
    LayoutSolverConfig,
    _Problem,
    _State,
    compute_treewidth,
    solve_auto_layout,
    solve_greedy,
    solve_treewidth_dp,
)
from pypto.pypto_core import codegen, passes

ND = ir.TensorLayout.ND
NZ = ir.TensorLayout.NZ


@pytest.fixture(autouse=True)
def _reset_backend():
    reset_for_testing()
    yield
    reset_for_testing()


def _value(
    name: str,
    producer: str | None = None,
    *,
    memory_space: ir.MemorySpace = ir.MemorySpace.DDR,
    reusable: bool = False,
    external_output: bool = False,
) -> LayoutValueNode:
    layouts = (ND, NZ) if memory_space == ir.MemorySpace.DDR else (ROW_MAJOR_FLAT, COL_MAJOR_FLAT)
    return LayoutValueNode(
        name=name,
        shape=(16, 16),
        dtype=DataType.FP16,
        memory_space=memory_space,
        current_layout=layouts[0],
        admissible_layouts=layouts,
        producer_op_id=producer,
        is_reusable_weight=reusable,
        is_external_output=external_output,
    )


def _operation(
    op_id: str,
    inputs: tuple[str, ...],
    output: str,
    choices: tuple[tuple[tuple[LayoutChoice, ...], LayoutChoice, float], ...],
    *,
    invocations: int = 1,
) -> LayoutOperationNode:
    return LayoutOperationNode(
        op_id=op_id,
        op_name="synthetic",
        inputs=inputs,
        output=output,
        invocations=invocations,
        configs=tuple(
            OperatorConfig(op_id, "synthetic", input_layouts, output_layout, cost)
            for input_layouts, output_layout, cost in choices
        ),
    )


def _assignment(instance: LayoutProblemInstance, indices: tuple[int, ...]) -> LayoutAssignment:
    return LayoutAssignment(
        value_layouts={},
        op_configs={
            op.op_id: op.configs[index] for op, index in zip(instance.operations, indices, strict=True)
        },
    )


def _exact(instance: LayoutProblemInstance) -> CostBreakdown:
    """Enumerate complete configurations, including internal conversion solutions."""
    candidates = (
        evaluate_assignment(
            instance,
            LayoutAssignment(
                value_layouts={},
                op_configs={op.op_id: cfg for op, cfg in zip(instance.operations, choices, strict=True)},
            ),
        )
        for choices in itertools.product(*(op.configs for op in instance.operations))
    )
    return min(candidates, key=lambda candidate: candidate.total_cost_cycles)


def _barrier(*, repeated_operand: bool = False) -> tuple[LayoutProblemInstance, LayoutAssignment]:
    """The two-consumer landscape is BB=6, AB=BA=8, AA=4."""
    inputs = ("x", "x") if repeated_operand else ("x",)
    first = _operation("Q1", inputs, "y", (((NZ,) * len(inputs), ND, 0.0), ((ND,) * len(inputs), ND, 2.0)))
    second = _operation("Q2", ("x",), "z", (((NZ,), ND, 0.0), ((ND,), ND, 2.0)))
    instance = LayoutProblemInstance(
        values={node.name: node for node in (_value("x"), _value("y", "Q1"), _value("z", "Q2"))},
        operations=[first, second],
        cost_model=LayoutCostModel(gm_pack_fixed_cycles=6.0, gm_pack_bytes_per_cycle=1e18),
    )
    return instance, _assignment(instance, (0, 0))


def test_greedy_initializes_topologically_then_refines_the_full_objective():
    producer = _operation("P", (), "t", (((), ND, 0.0), ((), NZ, 3.0)))
    consumer = _operation("Q", ("t",), "y", (((ND,), ND, 10.0), ((NZ,), ND, 0.0)))
    instance = LayoutProblemInstance(
        values={"t": _value("t", "P"), "y": _value("y", "Q")},
        operations=[consumer, producer],
        cost_model=LayoutCostModel(gm_pack_fixed_cycles=6.0, gm_pack_bytes_per_cycle=1e18),
    )
    seeded = solve_greedy(instance, config=LayoutSolverConfig(max_evaluations=4))
    assert seeded.assignment.op_configs["P"].output_layout == ND
    assert seeded.assignment.op_configs["Q"].input_layouts == (NZ,)
    assert seeded.total_cost_cycles == pytest.approx(6.0)
    assert seeded.stats.termination_reason == "evaluation_budget"

    refined = solve_greedy(instance)
    assert refined.assignment.op_configs["P"].output_layout == NZ
    assert refined.total_cost_cycles == pytest.approx(3.0)
    assert refined.stats.termination_reason == "local_optimum"


@pytest.mark.parametrize("repeated_operand", [False, True])
def test_treewidth_dp_solves_shared_conversion_barrier_globally(repeated_operand: bool):
    instance, initial = _barrier(repeated_operand=repeated_operand)
    greedy = solve_greedy(instance, initial=initial)
    dp_result = solve_treewidth_dp(instance)

    assert greedy.total_cost_cycles == pytest.approx(6.0)
    assert dp_result.total_cost_cycles == pytest.approx(4.0)
    assert dp_result.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)
    assert dp_result.stats.termination_reason == "optimal"
    assert dp_result.stats.treewidth == 1
    assert all(
        config.input_layouts == (ND,) * len(config.input_layouts)
        for config in dp_result.assignment.op_configs.values()
    )
    assert not dp_result.conversions


def test_treewidth_stats_on_chain_tree_and_diamond():
    # 1. Chain P -> Q -> R: treewidth = 1
    p = _operation("P", ("x",), "t1", (((ND,), ND, 1.0), ((NZ,), NZ, 1.0)))
    q = _operation("Q", ("t1",), "t2", (((ND,), ND, 1.0), ((NZ,), NZ, 1.0)))
    r = _operation("R", ("t2",), "y", (((ND,), ND, 1.0), ((NZ,), NZ, 1.0)))
    chain_inst = LayoutProblemInstance(
        values={"x": _value("x"), "t1": _value("t1", "P"), "t2": _value("t2", "Q"), "y": _value("y", "R")},
        operations=[p, q, r],
    )
    tw_chain = compute_treewidth(chain_inst)
    assert tw_chain.dataflow_treewidth == 1
    assert tw_chain.primal_treewidth == 1
    assert tw_chain.active_primal_treewidth == 1
    assert tw_chain.max_bag_size == 2
    assert tw_chain.max_bag_domain_product == 4

    # 2. Diamond: P produces t1; Q1, Q2 both consume t1; R consumes u1, u2 -> K_4 minus (P,R), treewidth = 2
    q1 = _operation("Q1", ("t1",), "u1", (((ND,), ND, 1.0), ((NZ,), NZ, 1.0)))
    q2 = _operation("Q2", ("t1",), "u2", (((ND,), ND, 1.0), ((NZ,), NZ, 1.0)))
    r_join = _operation("R", ("u1", "u2"), "y", (((ND, ND), ND, 1.0), ((NZ, NZ), NZ, 1.0)))
    diamond_inst = LayoutProblemInstance(
        values={
            "x": _value("x"),
            "t1": _value("t1", "P"),
            "u1": _value("u1", "Q1"),
            "u2": _value("u2", "Q2"),
            "y": _value("y", "R"),
        },
        operations=[p, q1, q2, r_join],
    )
    tw_diamond = compute_treewidth(diamond_inst)
    assert tw_diamond.dataflow_treewidth == 2
    assert tw_diamond.primal_treewidth == 2
    assert tw_diamond.max_bag_size == 3
    assert tw_diamond.max_bag_domain_product == 8


@pytest.mark.parametrize("producer_nz_cost, expected", [(1.0, 1.0), (5.0, 3.0)])
def test_producer_output_flip_and_internal_conversion_match_exact_oracle(
    producer_nz_cost: float, expected: float
):
    producer = _operation("P", (), "t", (((), ND, 0.0), ((), NZ, producer_nz_cost)))
    consumer = _operation("Q", ("t",), "y", (((ND,), ND, 9.0), ((NZ,), ND, 0.0)))
    instance = LayoutProblemInstance(
        values={"t": _value("t", "P"), "y": _value("y", "Q")},
        operations=[producer, consumer],
        cost_model=LayoutCostModel(gm_pack_fixed_cycles=3.0, gm_pack_bytes_per_cycle=1e18),
    )
    result = solve_greedy(instance, initial=_assignment(instance, (0, 1)))
    dp_result = solve_treewidth_dp(instance)

    assert result.total_cost_cycles == pytest.approx(expected)
    assert dp_result.total_cost_cycles == pytest.approx(expected)
    assert result.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)
    if producer_nz_cost == 1.0:
        assert result.assignment.op_configs["P"].output_layout == NZ
        assert not result.conversions
    else:
        assert enumerate_assignments(instance)[0].total_cost_cycles == pytest.approx(5.0)
        assert len(result.conversions) == 1
        conversion = result.conversions[0]
        assert (conversion.value_name, conversion.source_layout, conversion.target_layout) == ("t", ND, NZ)
        assert conversion.cost_cycles == pytest.approx(3.0)
        assert conversion.consumer_op_ids == ("Q",)
        assert not conversion.is_boundary
        assert result.writeback_issues
        with pytest.raises(ValueError):
            result.writeback_assignment()


def test_greedy_initialization_respects_dependencies_not_storage_order():
    producer = _operation("P", (), "t", (((), ND, 20.0), ((), NZ, 0.0)))
    consumer = _operation("Q", ("t",), "y", (((ND,), ND, 10.0), ((NZ,), ND, 0.0)))
    instance = LayoutProblemInstance(
        values={"t": _value("t", "P"), "y": _value("y", "Q")},
        operations=[consumer, producer],
    )
    result = solve_greedy(instance)
    assert result.total_cost_cycles == pytest.approx(0.0)
    assert result.assignment.value_layouts["t"] == NZ
    assert result.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)


@pytest.mark.parametrize("seed", range(8))
def test_small_random_dags_preserve_exact_objective_and_dp_optimality(seed: int):
    rng = random.Random(seed)
    values = {"x": _value("x")}
    operations = []
    for index in range(5):
        op_id, output = f"P{index}", f"v{index}"
        inputs = tuple(rng.choice(tuple(values)) for _ in range(rng.randrange(1, 3)))
        choices = tuple(
            (tuple(rng.choice((ND, NZ)) for _ in inputs), rng.choice((ND, NZ)), float(rng.randrange(9)))
            for _ in range(2)
        )
        operations.append(_operation(op_id, inputs, output, choices))
        values[output] = _value(output, op_id)
    instance = LayoutProblemInstance(
        values=values,
        operations=operations,
        cost_model=LayoutCostModel(gm_pack_fixed_cycles=4.0, gm_pack_bytes_per_cycle=1e18),
    )
    optimum = _exact(instance)
    greedy = solve_greedy(instance)
    dp_res = solve_treewidth_dp(instance)

    assert dp_res.total_cost_cycles == pytest.approx(optimum.total_cost_cycles)
    assert dp_res.total_cost_cycles <= greedy.total_cost_cycles + 1e-8
    for result in (greedy, dp_res):
        fresh = evaluate_assignment(instance, result.assignment)
        assert result.total_cost_cycles == pytest.approx(fresh.total_cost_cycles)
        assert result.breakdown.per_op_costs == pytest.approx(fresh.per_op_costs)
        assert result.breakdown.per_value_conversion_costs == pytest.approx(fresh.per_value_conversion_costs)
        assert sum(conversion.cost_cycles for conversion in result.conversions) == pytest.approx(
            result.breakdown.conversion_cost_cycles
        )
    for op in operations:
        for candidate in op.configs:
            configs = dict(greedy.assignment.op_configs)
            configs[op.op_id] = candidate
            trial = evaluate_assignment(instance, LayoutAssignment(value_layouts={}, op_configs=configs))
            assert trial.total_cost_cycles + 1e-8 >= greedy.total_cost_cycles


class _AsymmetricCostModel(LayoutCostModel):
    """Three physical layouts with nonmetric, directional conversion prices."""

    def estimate_conversion_cost(
        self,
        val_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        if src_layout == dst_layout:
            return 0.0
        assert isinstance(src_layout, ir.TensorLayout)
        assert isinstance(dst_layout, ir.TensorLayout)
        rank = {ND: 0, NZ: 1, ir.TensorLayout.DN: 2}
        return float(7 * (rank[src_layout] + 1) + 3 * (rank[dst_layout] + 1) + len(val_node.name))


def test_incremental_gain_apply_and_undo_match_full_objective_on_all_small_states():
    dn = ir.TensorLayout.DN
    operations = [
        _operation("P", ("x",), "t", (((ND,), ND, 3.0), ((NZ,), NZ, 1.0), ((dn,), dn, 2.0)), invocations=3),
        _operation(
            "Q",
            ("t", "t"),
            "u",
            (((ND, ND), NZ, 5.0), ((NZ, dn), ND, 2.0), ((dn, NZ), dn, 4.0)),
            invocations=6,
        ),
        _operation(
            "R",
            ("t", "u", "t"),
            "y",
            (((NZ, ND, NZ), NZ, 0.0), ((dn, dn, ND), dn, 1.0), ((ND, NZ, dn), ND, 3.0)),
            invocations=3,
        ),
    ]
    nodes = (
        _value("x", reusable=True),
        _value("t", "P"),
        _value("u", "Q"),
        _value("y", "R", external_output=True),
    )
    instance = LayoutProblemInstance(
        values={node.name: replace(node, admissible_layouts=(ND, NZ, dn)) for node in nodes},
        operations=operations,
        cost_model=_AsymmetricCostModel(expected_uses=3),
    )
    dp_res = solve_treewidth_dp(instance)
    assert dp_res.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)

    problem = _Problem(instance)
    for choices in itertools.product(range(3), repeat=3):
        state = _State(problem, list(choices))
        baseline = evaluate_assignment(instance, _assignment(instance, choices)).total_cost_cycles
        counts_before = {name: counts.copy() for name, counts in state.counts.items()}
        source_costs_before = {name: costs.copy() for name, costs in state.source_costs.items()}
        sources_before = state.sources.copy()
        assert state.cost == pytest.approx(baseline)
        for index, old_choice in enumerate(choices):
            for candidate in range(3):
                if candidate == old_choice:
                    continue
                changed = list(choices)
                changed[index] = candidate
                expected = evaluate_assignment(
                    instance, _assignment(instance, tuple(changed))
                ).total_cost_cycles
                assert state.gain(index, candidate) == pytest.approx(baseline - expected)
                state.apply(index, candidate)
                assert state.cost == pytest.approx(expected)
                assert state.choices == changed
                for next_index in range(3):
                    for next_choice in range(3):
                        next_choices = changed.copy()
                        next_choices[next_index] = next_choice
                        next_cost = evaluate_assignment(
                            instance, _assignment(instance, tuple(next_choices))
                        ).total_cost_cycles
                        assert state.gain(next_index, next_choice) == pytest.approx(expected - next_cost)
                state.apply(index, old_choice)
                assert state.cost == pytest.approx(baseline)
                assert state.choices == list(choices)
                assert state.counts == counts_before
                assert state.sources == sources_before
                assert state.source_costs == source_costs_before


@pytest.mark.parametrize("expected_uses", [1, 3])
def test_solver_uses_l2_reread_weight_in_operator_costs(expected_uses: int):
    op = LayoutOperationNode(
        op_id="load",
        op_name="tile.load",
        inputs=("x",),
        output="tile",
        invocations=4 * expected_uses,
        configs=(
            OperatorConfig("load", "tile.load", (ND,), ROW_MAJOR_FLAT, 10.0),
            OperatorConfig("load", "tile.load", (NZ,), ROW_MAJOR_FLAT, 8.0),
        ),
    )
    instance = LayoutProblemInstance(
        values={"x": _value("x"), "tile": _value("tile", "load", memory_space=ir.MemorySpace.Mat)},
        operations=[op],
        cost_model=LayoutCostModel(
            expected_uses=expected_uses,
            mte2_l2_reread_discount=0.3,
            gm_pack_fixed_cycles=0.0,
            gm_pack_bytes_per_cycle=1e18,
        ),
    )
    expected = expected_uses * (1.0 + 3 * 0.3) * 8.0
    for solve in (solve_greedy, solve_treewidth_dp):
        result = solve(instance)
        assert result.assignment.op_configs["load"].input_layouts == (NZ,)
        assert result.breakdown.per_op_costs["load"] == pytest.approx(expected)
        assert result.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)


def test_internal_tile_conversion_uses_producer_invocation_count():
    producer = _operation("P", (), "t", (((), ROW_MAJOR_FLAT, 0.0),), invocations=7)
    consumer = _operation("Q", ("t",), "y", (((COL_MAJOR_FLAT,), ROW_MAJOR_FLAT, 0.0),), invocations=7)
    instance = LayoutProblemInstance(
        values={
            "t": _value("t", "P", memory_space=ir.MemorySpace.Vec),
            "y": _value("y", "Q", memory_space=ir.MemorySpace.Vec),
        },
        operations=[producer, consumer],
        cost_model=LayoutCostModel(expected_uses=1, onchip_relayout_bytes_per_cycle=128.0),
    )
    for solve in (solve_greedy, solve_treewidth_dp):
        result = solve(instance)
        assert result.breakdown.conversion_cost_cycles == pytest.approx(7 * 512 / 128)
        assert result.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)
        with pytest.raises(ValueError):
            result.writeback_assignment()


@pytest.mark.parametrize("target", [ROW_MAJOR_FLAT, COL_MAJOR_FLAT])
def test_loop_exit_conversion_requires_execution_domain_information(target: LayoutChoice):
    producer = _operation("P", (), "t", (((), ROW_MAJOR_FLAT, 0.0),), invocations=7)
    consumer = _operation("Q", ("t",), "y", (((target,), ROW_MAJOR_FLAT, 0.0),), invocations=1)
    instance = LayoutProblemInstance(
        values={
            "t": _value("t", "P", memory_space=ir.MemorySpace.Vec),
            "y": _value("y", "Q", memory_space=ir.MemorySpace.Vec),
        },
        operations=[producer, consumer],
    )
    if target == ROW_MAJOR_FLAT:
        assert solve_greedy(instance).total_cost_cycles == 0.0
        assert solve_treewidth_dp(instance).total_cost_cycles == 0.0
    else:
        with pytest.raises(ValueError, match="execution counts"):
            solve_greedy(instance)
        with pytest.raises(ValueError, match="execution counts"):
            solve_treewidth_dp(instance)
        with pytest.raises(ValueError, match="execution counts"):
            evaluate_assignment(instance, _assignment(instance, (0, 0)))


@pytest.mark.parametrize("reusable, expected_conversion", [(False, 60.0), (True, 6.0)])
def test_boundary_packing_reuse_and_single_layout_writeback(reusable: bool, expected_conversion: float):
    op = _operation("Q", ("x",), "y", (((NZ,), ND, 2.0),), invocations=10)
    instance = LayoutProblemInstance(
        values={"x": _value("x", reusable=reusable), "y": _value("y", "Q")},
        operations=[op],
        cost_model=LayoutCostModel(expected_uses=10, gm_pack_fixed_cycles=6.0, gm_pack_bytes_per_cycle=1e18),
    )
    for solve in (solve_greedy, solve_treewidth_dp):
        result = solve(instance)
        assert result.total_cost_cycles == pytest.approx(20.0 + expected_conversion)
        assert len(result.conversions) == 1
        assert result.conversions[0].is_boundary
        assert result.conversions[0].cost_cycles == pytest.approx(expected_conversion)
        assert not result.writeback_issues
        assert result.writeback_assignment().tensor_layouts["x"] == NZ


def test_writeback_rejects_two_representations_of_one_external_input():
    first = _operation("Q1", ("x",), "y", (((ND,), ND, 0.0),))
    second = _operation("Q2", ("x",), "z", (((NZ,), ND, 0.0),))
    instance = LayoutProblemInstance(
        values={node.name: node for node in (_value("x"), _value("y", "Q1"), _value("z", "Q2"))},
        operations=[first, second],
    )
    for solve in (solve_greedy, solve_treewidth_dp):
        result = solve(instance)
        assert result.writeback_issues
        with pytest.raises(ValueError):
            result.writeback_assignment()


@pytest.mark.parametrize("allow_relayout, expected", [(False, 6.0), (True, 0.0)])
def test_external_output_layout_requirement_contributes_to_objective(allow_relayout: bool, expected: float):
    op = _operation("P", (), "y", (((), NZ, 0.0),))
    instance = LayoutProblemInstance(
        values={"y": _value("y", "P", external_output=True)},
        operations=[op],
        cost_model=LayoutCostModel(
            allow_external_output_relayout=allow_relayout,
            gm_pack_fixed_cycles=6.0,
            gm_pack_bytes_per_cycle=1e18,
        ),
    )
    for solve in (solve_greedy, solve_treewidth_dp):
        result = solve(instance)
        assert result.total_cost_cycles == pytest.approx(expected)
        assert result.total_cost_cycles == pytest.approx(_exact(instance).total_cost_cycles)
        assert all(conversion.is_boundary for conversion in result.conversions)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_input",
        "empty_catalog",
        "foreign_config",
        "wrong_arity",
        "nan",
        "infinity",
        "multiple_producers",
        "cycle",
    ],
)
def test_invalid_problem_is_rejected_before_search(defect: str):
    producer = _operation("P", ("x",), "t", (((ND,), ND, 0.0),))
    consumer = _operation("Q", ("t",), "y", (((ND,), ND, 0.0),))
    if defect == "unknown_input":
        producer = replace(producer, inputs=("missing",))
    elif defect == "empty_catalog":
        producer = replace(producer, configs=())
    elif defect == "foreign_config":
        producer = replace(producer, configs=(replace(producer.configs[0], op_id="other"),))
    elif defect == "wrong_arity":
        producer = replace(producer, configs=(replace(producer.configs[0], input_layouts=()),))
    elif defect in ("nan", "infinity"):
        invalid_cost = float("nan") if defect == "nan" else float("inf")
        producer = replace(producer, configs=(replace(producer.configs[0], op_cost_cycles=invalid_cost),))
    elif defect == "multiple_producers":
        consumer = replace(consumer, output="t")
    elif defect == "cycle":
        producer = replace(producer, inputs=("y",))
    instance = LayoutProblemInstance(
        values={"x": _value("x"), "t": _value("t", "P"), "y": _value("y", "Q")},
        operations=[producer, consumer],
    )
    with pytest.raises(ValueError):
        solve_greedy(instance)
    with pytest.raises(ValueError):
        solve_treewidth_dp(instance)


@pytest.mark.parametrize("defect", ["foreign_config", "contradictory_output"])
def test_public_evaluator_rejects_assignment_that_disagrees_with_scored_configuration(defect: str):
    op = _operation("P", ("x",), "y", (((ND,), ND, 0.0),))
    instance = LayoutProblemInstance(values={"x": _value("x"), "y": _value("y", "P")}, operations=[op])
    selected = op.configs[0]
    layouts: dict[str, LayoutChoice] = {}
    if defect == "foreign_config":
        selected = replace(selected, op_cost_cycles=1.0)
    else:
        layouts["y"] = NZ
    assignment = LayoutAssignment(value_layouts=layouts, op_configs={"P": selected})
    with pytest.raises(ValueError):
        evaluate_assignment(instance, assignment)


def test_empty_problem_has_zero_cost():
    for solve in (solve_greedy, solve_treewidth_dp):
        result = solve(LayoutProblemInstance())
        assert result.total_cost_cycles == 0.0
        assert not result.assignment.op_configs
        assert not result.conversions


def test_extracted_matmul_solver_matches_calibration_enumerator_and_lowers():
    set_backend_type(BackendType.Ascend910)
    case = auto_layout_bench.build_single_matmul_case(64, 128, 64, expected_uses=100)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        pre_program = run_pre_layout_passes(case.program)
        instance = extract_layout_instance(
            pre_program,
            cost_model=LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910),
            reusable_tensors=case.reusable_tensors,
        )
        exact_variants = enumerate_assignments(instance)
        greedy_res = solve_greedy(instance)
        dp_res = solve_treewidth_dp(instance)
        assert greedy_res.total_cost_cycles == pytest.approx(exact_variants[0].total_cost_cycles)
        assert dp_res.total_cost_cycles == pytest.approx(exact_variants[0].total_cost_cycles)
        assert dp_res.treewidth_stats is not None
        assert dp_res.treewidth_stats.primal_treewidth == 1
        assert dp_res.treewidth_stats.active_primal_treewidth == 1
        assignment = dp_res.writeback_assignment()
        assert assignment.tensor_layouts["w"] == NZ
        rewritten = apply_layout_assignment(pre_program, assignment)
        ir.assert_structural_equal(passes.resolve_backend_op_layouts()(rewritten), rewritten)
        lowered = run_post_layout_passes(rewritten)
        texts = []
        for function in lowered.functions.values():
            if function.func_type in (ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV):
                single = ir.Program([function], function.name, lowered.span)
                emitted = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
                texts.append(emitted if isinstance(emitted, str) else "".join(emitted.values()))
        assert any("pto.tmatmul" in text for text in texts)


def test_solve_auto_layout_dispatches_by_max_treewidth_and_shared_qkv_case():
    set_backend_type(BackendType.Ascend910)
    instance, initial = _barrier()
    # Default max_treewidth=8 uses exact treewidth DP (`optimal`).
    auto_dp = solve_auto_layout(instance, initial=initial)
    assert auto_dp.stats.termination_reason == "optimal"
    assert auto_dp.total_cost_cycles == pytest.approx(4.0)

    # Setting max_treewidth=0 forces fallback to greedy coordinate descent (`local_optimum`).
    auto_greedy = solve_auto_layout(instance, initial=initial, config=LayoutSolverConfig(max_treewidth=0))
    assert auto_greedy.stats.termination_reason == "local_optimum"
    assert auto_greedy.total_cost_cycles == pytest.approx(6.0)
    with pytest.raises(ValueError, match="max_treewidth"):
        solve_treewidth_dp(instance, config=LayoutSolverConfig(max_treewidth=0))

    # Verify the 3-projection shared-x QKV case separates DP/auto_layout from greedy.
    qkv_case = auto_layout_bench.build_shared_input_qkv_case(64, 1024, 64, expected_uses=100)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        pre_qkv = run_pre_layout_passes(qkv_case.program)
        qkv_inst = extract_layout_instance(
            pre_qkv,
            cost_model=LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910),
            reusable_tensors=qkv_case.reusable_tensors,
        )
        greedy_qkv = solve_greedy(qkv_inst)
        dp_qkv = solve_treewidth_dp(qkv_inst)
        auto_qkv = solve_auto_layout(qkv_inst)
        assert greedy_qkv.assignment.tensor_layouts["x"] == ND
        assert dp_qkv.assignment.tensor_layouts["x"] == NZ
        assert auto_qkv.assignment.tensor_layouts["x"] == NZ
        assert dp_qkv.total_cost_cycles < greedy_qkv.total_cost_cycles
        assert auto_qkv.total_cost_cycles == pytest.approx(dp_qkv.total_cost_cycles)

        # Verify `make_solver_auto_layout_pass` rewrites `x`, `wq`, `wk`, `wv` to NZ.
        solver_pass = make_solver_auto_layout_pass(
            LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910),
            reusable_tensors=qkv_case.reusable_tensors,
            solver="auto",
        )
        rewritten_qkv = solver_pass(pre_qkv)
        lowered_qkv = run_post_layout_passes(rewritten_qkv)
        assert len(lowered_qkv.functions) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
