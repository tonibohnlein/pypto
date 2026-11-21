# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent emitted-IR checks for the bounded scheduling catalogue.

An unavailable forced mode is a reported refusal, never a successful ablation.
The token/access/flag interpreters do not consume scheduler cost certificates.
"""

from typing import cast

import pytest
from pypto import ir, passes, testing

from .sync_control_metrics import COMMANDS, statements, trace_control
from .sync_physical_oracle import run_physical
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_autosync_switch import clone
from .test_nested_sync_repetition import BOUNDS as NESTED_DMA_BOUNDS
from .test_nested_sync_repetition import nested_fixture
from .test_parameterized_token_sync import check, worker
from .test_streamed_slot_composition import streamed_fixture
from .test_sync_case_only_schedule import case_only_fixture
from .test_sync_command_control import command, guarded, seq
from .test_sync_range_proofs import caller_program
from .test_token_sync_lowering import _apply, _check_actual_tokens, _two_streams

pytestmark = pytest.mark.usefixtures("ascend_backend")

MODES = (
    "guarded",
    "case",
    "boundary",
    "periodic",
    "case_boundary",
    "case_periodic",
    "boundary_periodic",
    "combined",
)
WORK_KEYS = (
    "scalar_arithmetic_ops",
    "scalar_comparisons",
    "scalar_boolean_ops",
    "conditions",
    "loop_header_work",
    "scalar_evaluations",
)


class ExpandedCensus:
    """Walk occurrences explicitly, independently of the C++ visitor callbacks."""

    def __init__(self):
        self.nodes = 0
        self.payloads = 0
        self.commands = 0

    def attributes(self, attributes):
        for value in dict(attributes).values():
            if isinstance(value, ir.Expr):
                self.visit_expr(value)
            elif isinstance(value, (tuple, list)):
                for element in value:
                    if isinstance(element, ir.Expr):
                        self.visit_expr(element)

    def visit_function(self, function):
        for parameter in function.params:
            self.visit_expr(parameter)
        self.attributes(function.attrs)
        self.visit_stmt(function.body)

    def visit_expr(self, op):
        self.nodes += 1
        if isinstance(op, (ir.MemRef, ir.WindowBuffer)):
            return
        if isinstance(op, ir.Var):
            if isinstance(op.type, ir.TensorType):
                for dimension in op.type.shape:
                    self.visit_expr(dimension)
            if isinstance(op, ir.IterArg):
                self.visit_expr(op.initValue)
        elif isinstance(op, ir.BinaryExpr):
            self.visit_expr(op.left)
            self.visit_expr(op.right)
        elif isinstance(op, ir.UnaryExpr):
            self.visit_expr(op.operand)
        elif isinstance(op, ir.Call):
            for argument in op.args:
                self.visit_expr(argument)
            self.attributes(op.kwargs)
        elif isinstance(op, ir.MakeTuple):
            for element in op.elements:
                self.visit_expr(element)
        else:
            assert isinstance(op, (ir.ConstInt, ir.ConstFloat, ir.ConstBool)), type(op)

    def visit_stmt(self, op):
        self.nodes += 1
        if isinstance(op, ir.SeqStmts):
            for child in op.stmts:
                self.visit_stmt(child)
        elif isinstance(op, ir.AssignStmt):
            self.visit_expr(op.var)
            self.visit_expr(op.value)
        elif isinstance(op, ir.IfStmt):
            self.visit_expr(op.condition)
            self.visit_stmt(op.then_body)
            if op.else_body is not None:
                self.visit_stmt(op.else_body)
            for variable in op.return_vars:
                self.visit_expr(variable)
        elif isinstance(op, ir.ForStmt):
            for expression in (op.loop_var, op.start, op.stop, op.step, *op.iter_args):
                self.visit_expr(expression)
            self.visit_stmt(op.body)
            for variable in op.return_vars:
                self.visit_expr(variable)
        elif isinstance(op, ir.EvalStmt):
            if isinstance(op.expr, ir.Call):
                if op.expr.op.name in COMMANDS:
                    self.commands += 1
                else:
                    self.payloads += 1
            self.visit_expr(op.expr)
        else:
            assert isinstance(op, (ir.YieldStmt, ir.ReturnStmt)), type(op)
            for value in op.value:
                self.visit_expr(value)


def census(function):
    result = ExpandedCensus()
    result.visit_function(function)
    return result


def emitted_work(function, parameters=None, block=None):
    state = trace_control(function, parameters, block)
    return state, tuple(state.counts[key] for key in WORK_KEYS)


def fingerprint(result):
    # Independent lowering transactions own different stripped loop objects.
    # Compare their structure, not Python wrapper or C++ allocation addresses.
    return [
        (
            [
                {
                    **family,
                    "axes": [
                        (ir.structural_hash(frame, True), count, source_stride, target_stride)
                        for frame, count, source_stride, target_stride in family["axes"]
                    ],
                }
                for family in case["families"]
            ],
            case["native_reserved_ids"],
        )
        for case in result["physical_cases"]
    ]


def lower(logical, mode, *, parameter_bounds=None, block_bounds=None, **policy):
    # Ablation explicitly opts into bounded duplication; production defaults
    # are checked separately and never inferred from these forced modes.
    policy.setdefault("no_clone", False)
    return testing.lower_sync_schedule(
        logical,
        parameter_bounds=parameter_bounds or {},
        block_bounds=block_bounds,
        schedule_policy=testing.SyncSchedulePolicy(mode=mode, **policy),
    )


def selected(result):
    return [row for row in result["schedule"]["choices"] if row["selected"]]


def assert_receipt(result, requested):
    rows = result["schedule"]["choices"]
    assert selected(result)
    for row in rows:
        assert row["model_scope"] == "standalone_candidate_before_parent_gap_composition"
    labels = {row["loop"] for row in selected(result)}
    for gap in result["schedule"]["parent_gaps"]:
        assert gap["loop"] in labels
        assert gap["scope"] == "parent_gap_after_candidate_selection"
        assert gap["count_order"] == ("guards", "expressions", "statements")
        assert gap["command_sites"] > 0
        assert gap["after"][0] < gap["before"][0]
        assert all(a <= b for a, b in zip(gap["after"], gap["before"], strict=True))
    for row in selected(result):
        assert row["status"] in ("selected", "selected_forced", "fallback")
        if requested != "auto" and row["mode"] != requested:
            assert row["mode"] == "guarded" and row["status"] == "fallback"
            assert any(
                other["loop"] == row["loop"]
                and other["mode"] == requested
                and not other["selected"]
                and other["status"] in ("ineligible", "over_budget")
                and other["reason"]
                for other in rows
            )
    if result["physical_cases"]:
        actual = census(result["function"])
        assert actual.nodes == result["schedule"]["emitted_control_nodes"]
        assert actual.payloads == result["schedule"]["emitted_payload_sites"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count,start,step", [(1, 0, 1), (2, 0, 1), (4, 5, 2), (7, 0, 1), (11, 5, 2)])
def test_forced_catalogue_preserves_actual_tokens_operands_and_order(mode, count, start, step):
    original = streamed_fixture(count, start=start, step=step)
    logical = _apply(passes.insert_sync, original)
    baseline = lower(logical, "guarded")
    result = lower(logical, mode)
    assert_receipt(result, mode)
    assert fingerprint(result) == fingerprint(baseline)
    _check_actual_tokens(original, logical, result["function"], check_codegen=False)
    assert trace_control(result["function"]).trace == trace_control(baseline["function"]).trace


@pytest.mark.parametrize("mode", MODES)
def test_case_only_matrix_preserves_extra_actual_token_uses(mode):
    original, logical = case_only_fixture()
    parameters = {4: (0, 0)}
    blocks = (0, 1)
    baseline = lower(logical, "guarded", parameter_bounds=parameters, block_bounds=blocks)
    result = lower(logical, mode, parameter_bounds=parameters, block_bounds=blocks)
    assert_receipt(result, mode)
    assert fingerprint(result) == fingerprint(baseline)
    for block in (0, 1):
        check(original, logical, result["function"], {4: 0}, block=block, minimum_protocol=False)
        assert (
            trace_control(result["function"], {4: 0}, block).trace
            == trace_control(baseline["function"], {4: 0}, block).trace
        )
    if mode == "case":
        assert any(row["mode"] == mode for row in selected(result))
    if mode in ("case_periodic", "combined"):
        assert not any(row["mode"] == mode for row in selected(result))


@pytest.mark.parametrize("mode", ["boundary", "periodic", "boundary_periodic"])
def test_period_and_interval_candidates_are_independent(mode):
    original = streamed_fixture(11, start=5, step=2)
    logical = _apply(passes.insert_sync, original)
    result = lower(logical, mode, boundary_motion=False)
    assert any(row["mode"] == mode for row in selected(result))
    if mode == "periodic":
        loops = [s for s in statements(result["function"].body) if isinstance(s, ir.ForStmt)]
        expanded = [loop for loop in loops if isinstance(loop.step, ir.ConstInt) and loop.step.value > 2]
        assert expanded
        assert any(isinstance(s, ir.IfStmt) for loop in expanded for s in statements(loop.body))
    assert any(len(family["ids"]) > 1 for case in result["physical_cases"] for family in case["families"])
    _check_actual_tokens(original, logical, result["function"], check_codegen=False)


@pytest.mark.parametrize(
    "policy", [{"no_clone": True}, {"max_extra_payload_sites": 0}, {"max_extra_control_nodes": 0}]
)
def test_forced_mode_never_bypasses_growth_policy(policy):
    original, logical = case_only_fixture()
    parameters = {4: (0, 0)}
    blocks = (0, 1)
    result = lower(logical, "case", parameter_bounds=parameters, block_bounds=blocks, **policy)
    assert_receipt(result, "case")
    assert not any(row["mode"] == "case" for row in selected(result))
    assert any(
        row["mode"] == "case" and row["status"] == "over_budget" for row in result["schedule"]["choices"]
    )
    for block in (0, 1):
        check(original, logical, result["function"], {4: 0}, block=block, minimum_protocol=False)


@pytest.mark.parametrize("count", [2, 4, 7, 11, 17])
def test_automatic_choice_does_not_increase_executed_source_work(count):
    original = streamed_fixture(count, start=5, step=2)
    logical = _apply(passes.insert_sync, original)
    baseline = lower(logical, "guarded")
    result = lower(logical, "auto")
    assert_receipt(result, "auto")
    before, baseline_cost = emitted_work(baseline["function"])
    after, selected_cost = emitted_work(result["function"])
    assert before.trace == after.trace
    assert sum(selected_cost) <= sum(baseline_cost)


def projection_schedule_inputs(name, disjoint):
    program = capture_qwen_projection(name, layers=2, hidden=5120, band=0)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    with passes.PassContext(
        [],
        enable_pypto_autosync=True,
        autosync_disjoint_gm=disjoint,
        autosync_policy="exact" if disjoint else "conservative_prefix",
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        logical = worker(passes.optimize_sync()(passes.insert_sync()(program)), original.name)
        optimized = worker(passes.realize_sync()(program), original.name)
    evidence = testing.sync_invocation_evidence(program)
    bounds = {
        "parameter_bounds": evidence["parameters"][original.name],
        "block_bounds": tuple(evidence["block_indices"][original.name]),
    }
    return logical, optimized, bounds


@pytest.mark.parametrize("name,block", [("output", 0), ("gate_up", 7)])
@pytest.mark.parametrize("disjoint", [True, False])
def test_projection_production_boundary_phases_preserve_order_with_less_control(name, block, disjoint):
    logical, optimized, bounds = projection_schedule_inputs(name, disjoint)
    baseline = lower(logical, "guarded", **bounds)["function"]
    before, before_work = emitted_work(baseline, {3: 0}, block=block)
    after, after_work = emitted_work(optimized, {3: 0}, block=block)
    # Production appends a drain. The remaining ordered commands, event IDs,
    # and payload cuts must match guarded lowering through all phase edges.
    assert after.trace[-1] == ("system.bar_all", before.payload_count)
    assert before.trace == after.trace[:-1]
    assert sum(after_work) < sum(before_work)
    assert (
        after.counts["conditions"] + after.counts["loop_header_work"]
        < (before.counts["conditions"] + before.counts["loop_header_work"]) * 0.65
    )
    before_size, after_size = census(baseline), census(optimized)
    assert 1 < after_size.payloads - before_size.payloads <= 57  # Includes the production drain.
    assert after_size.nodes < 2 * before_size.nodes


@pytest.mark.parametrize(
    "budget", [{"no_clone": True}, {"max_extra_payload_sites": 0}, {"max_extra_control_nodes": 0}]
)
def test_projection_boundary_growth_limits_preserve_guarded_schedule(budget):
    logical, _, bounds = projection_schedule_inputs("output", True)
    baseline = lower(logical, "guarded", **bounds)["function"]
    result = lower(logical, "auto_boundary", **({"no_clone": False} | bounds | budget))
    ir.assert_structural_equal(result["function"], baseline)
    assert all(row["mode"] == "guarded" for row in selected(result))
    assert any(
        row["mode"] == "boundary" and row["status"] == "over_budget" for row in result["schedule"]["choices"]
    )


def test_retained_projection_children_share_a_payload_growth_budget():
    logical, _, bounds = projection_schedule_inputs("gate_up", False)
    retained = [
        s
        for s in statements(logical.body)
        if isinstance(s, ir.ForStmt) and "pypto.closed_sync_child" in s.attrs
    ]
    assert len(retained) > 1
    baseline = lower(logical, "guarded", **bounds)
    budget = 14
    limited = lower(logical, "auto_boundary", max_extra_payload_sites=budget, **bounds)
    before, after = baseline["function"], limited["function"]
    growth = census(after).payloads - census(before).payloads
    assert 0 < growth <= budget
    schedule = limited["schedule"]
    assert schedule["emitted_payload_sites"] - baseline["schedule"]["emitted_payload_sites"] == growth
    assert schedule["payload_growth_debit"] == growth
    assert baseline["schedule"]["payload_growth_debit"] == 0
    assert 0 < schedule["control_growth_debit"] <= limited["schedule_policy"]["max_extra_control_nodes"]

    # The growth belongs to the selected child boundary schedule. A guarded
    # identity for the parent alone used to hide the child choices and savings.
    chosen = selected(limited)
    labels = [row["loop"] for row in chosen]
    assert len(set(labels)) == len(labels)
    assert list(dict.fromkeys(label.split("/", 1)[0] for label in labels)) == [
        f"retained[{ordinal}]:{child.loop_var.name_hint}" for ordinal, child in enumerate(retained)
    ]
    guarded_payloads = {row["loop"]: row["payload_sites"] for row in selected(baseline)}
    assert set(guarded_payloads) == set(labels)
    assert all(row["mode"] == "guarded" for row in selected(baseline))
    boundary_loops = sum(row["mode"] == "boundary" for row in chosen)
    assert boundary_loops > 0
    assert sum(row["payload_sites"] - guarded_payloads[row["loop"]] for row in chosen) == growth
    work = limited["physical_cases"][0]
    assert work["regular_schedule_loops"] == boundary_loops
    assert work["regular_schedule_candidates"] >= boundary_loops
    assert work["regular_schedule_payload_copies"] >= growth
    assert work["regular_schedule_model_saving"] > 0
    assert baseline["physical_cases"][0]["regular_schedule_model_saving"] == 0
    assert {gap["loop"] for gap in schedule["parent_gaps"]}.issubset(labels)
    for block in (0, 7, 23):
        assert trace_control(before, {3: 1}, block).trace == trace_control(after, {3: 1}, block).trace


@pytest.mark.parametrize("count", [2, 7])
def test_boundary_auto_leaves_flat_or_short_repetition_guarded(count):
    logical = _apply(passes.insert_sync, streamed_fixture(count))
    result = lower(logical, "auto_boundary", no_clone=False)
    assert all(row["mode"] == "guarded" for row in selected(result))
    ir.assert_structural_equal(lower(logical, "guarded")["function"], result["function"])


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("two_tiles", [False, True])
def test_boundary_auto_short_single_visit_phases(count, two_tiles):
    original = nested_fixture((4, count), repeats=2)
    if two_tiles:
        original = add_independent_dma_tile(original)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        logical = worker(passes.insert_sync()(caller_program(original)), original.name)
    bounds = {"parameter_bounds": NESTED_DMA_BOUNDS}
    baseline = lower(logical, "guarded", **bounds)
    growth_per_loop = (4 if two_tiles else 2) * (count - 1)
    result = lower(logical, "auto_boundary", max_extra_payload_sites=growth_per_loop, **bounds)

    choices = result["schedule"]["choices"]
    boundary = [row for row in choices if row["mode"] == "boundary"]
    assert len(boundary) == 2
    if not two_tiles:
        # Extra singleton headers consume the proved decision saving for one tile.
        assert all(row["status"] in ("not_profitable", "cost_unproved") for row in boundary)
        ir.assert_structural_equal(result["function"], baseline["function"])
        return

    parameters = {1: 16, 3: 64}
    before, before_work = emitted_work(baseline["function"], parameters)
    after, after_work = emitted_work(result["function"], parameters)

    # The nonzero, nonunit inner counter indexes the original scope. Both
    # qualifying loops compete for one leaf budget, so exactly one can split.
    assert [row["status"] for row in boundary] == ["selected", "over_budget"]
    assert result["schedule"]["payload_growth_debit"] == growth_per_loop
    assert census(result["function"]).payloads - census(baseline["function"]).payloads == growth_per_loop
    assert before.trace == after.trace
    assert sum(after_work) < sum(before_work)
    run_physical(original, result["function"], range(6), parameters)


def add_independent_dma_tile(function):
    """Copy each DMA and vector access into an independent UB tile and GM column range."""
    span = function.span
    allocation, *rest = cast(ir.SeqStmts, function.body).stmts
    assert isinstance(allocation, ir.AssignStmt) and isinstance(allocation.value, ir.Call)
    tile = allocation.var
    other = ir.Var("second_tile", tile.type, span)
    second_address = ir.ConstInt(4096, ir.DataType.INDEX, span)
    second_alloc_call = ir.Call(
        allocation.value.op, [allocation.value.args[0], second_address], other.type, span
    )
    second_allocation = ir.AssignStmt(other, second_alloc_call, span)
    dma_ops = {ir.get_op(name).name for name in ("buffer.load", "buffer.store")}

    def duplicate_access(stmt):
        assert isinstance(stmt.expr, ir.Call)
        call = stmt.expr
        args = [other if operand.same_as(tile) else operand for operand in call.args]
        if call.op.name in dma_ops:
            offsets = cast(ir.MakeTuple, args[1])
            args[1] = ir.MakeTuple([offsets.elements[0], ir.ConstInt(32, ir.DataType.INDEX, span)], span)
        return ir.EvalStmt(ir.Call(call.op, args, call.type, span), span)

    def duplicate_payload(body):
        # Keep each loop body flat so the placed-sync importer sees the
        # repeated access sites at their original loop depth.
        output = []
        for stmt in body:
            if isinstance(stmt, ir.ForStmt):
                nested = duplicate_payload(cast(ir.SeqStmts, stmt.body).stmts)
                output.append(
                    ir.ForStmt(stmt.loop_var, stmt.start, stmt.stop, stmt.step, [], nested, [], span)
                )
            else:
                output.append(stmt)
                if isinstance(stmt, ir.EvalStmt):
                    output.append(duplicate_access(stmt))
        return ir.SeqStmts(output, span)

    payload = duplicate_payload(rest)
    return clone(function, body=ir.SeqStmts([allocation, second_allocation, *payload.stmts], span))


def test_no_clone_policy_keeps_static_payload_sites_for_every_mode():
    original = streamed_fixture(11)
    logical = _apply(passes.insert_sync, original)
    baseline = lower(logical, "guarded")
    for mode in MODES:
        result = lower(logical, mode, no_clone=True)
        assert census(result["function"]).payloads <= census(baseline["function"]).payloads
        assert_receipt(result, mode)


@pytest.mark.parametrize("fixture", ["streamed", "case"])
def test_production_default_keeps_payload_body_count_and_actual_semantics(fixture):
    if fixture == "streamed":
        original = streamed_fixture(11, start=5, step=2)
        logical = _apply(passes.insert_sync, original)
        parameters = {}
        blocks = None
    else:
        original, logical = case_only_fixture()
        parameters = {4: (0, 0)}
        blocks = (0, 1)
    result = testing.lower_sync_schedule(logical, parameter_bounds=parameters, block_bounds=blocks)
    baseline = lower(logical, "guarded", parameter_bounds=parameters, block_bounds=blocks)
    assert result["schedule_policy"]["no_clone"] is True
    assert census(result["function"]).payloads <= census(baseline["function"]).payloads
    if fixture == "streamed":
        _check_actual_tokens(original, logical, result["function"], check_codegen=False)
    else:
        for block in (0, 1):
            check(original, logical, result["function"], {4: 0}, block=block, minimum_protocol=False)


@pytest.mark.parametrize("mode", MODES)
def test_selected_loop_bound_dominates_independent_expanded_ir_census(mode):
    original = streamed_fixture(11, start=5, step=2)
    original_loop = next(s for s in statements(original.body) if isinstance(s, ir.ForStmt))
    loop_size = ExpandedCensus()
    loop_size.visit_stmt(original_loop)
    outside = census(original).nodes - loop_size.nodes
    result = lower(_apply(passes.insert_sync, original), mode)
    choices = selected(result)
    assert len(choices) == 1
    assert census(result["function"]).nodes <= outside + choices[0]["control_node_bound"]
    assert census(result["function"]).payloads == choices[0]["payload_sites"]


@pytest.mark.parametrize("source", ["finite_tokens", "no_tokens", "empty_loop"])
def test_unavailable_forced_mode_has_explicit_fallback_without_a_plannable_loop(source):
    original, logical = _two_streams(strengthen=True)
    if source == "no_tokens":
        logical = original
    elif source == "empty_loop":
        original = streamed_fixture(0)
        logical = _apply(passes.insert_sync, original)
    result = lower(logical, "periodic")
    assert_receipt(result, "periodic")
    assert any(
        row["mode"] == "periodic" and row["status"] == "ineligible" for row in result["schedule"]["choices"]
    )
    assert any(row["mode"] == "guarded" and row["status"] == "fallback" for row in selected(result))
    if source == "no_tokens":
        ir.assert_structural_equal(original.body, result["function"].body)
    else:
        _check_actual_tokens(original, logical, result["function"], check_codegen=False)


def predicate_sharing_pair():
    """Emitter-only scalar sharing with one unchanged payload, not loop versioning."""
    original, _ = _two_streams()
    span = original.span
    iv = ir.Var("coordinate", ir.ScalarType(ir.DataType.INDEX), span)

    def integer(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    conditions = seq(
        *[
            guarded(
                ir.Lt(
                    ir.FloorDiv(iv, integer(2), ir.DataType.INDEX, span),
                    integer(bound),
                    ir.DataType.BOOL,
                    span,
                ),
                command(bound),
            )
            for bound in (2, 3, 4)
        ]
    )
    payload = cast(ir.SeqStmts, original.body).stmts[2]

    def owner(commands):
        return ir.Function(
            "predicate_sharing",
            [*zip(original.params, original.param_directions), iv],
            [],
            seq(*cast(ir.SeqStmts, original.body).stmts[:2], commands, payload, ir.ReturnStmt([], span)),
            span,
            original.func_type,
            ir_stage=original.ir_stage,
        )

    baseline = owner(conditions)
    output = owner(testing.materialize_sync_command_values(baseline, conditions))
    return baseline, output


def test_predicate_sharing_is_a_separate_no_clone_control_measurement():
    baseline, output = predicate_sharing_pair()
    assert census(baseline).payloads == census(output).payloads == 1
    for coordinate in (-1, 0, 3, 8, 10):
        before, old_cost = emitted_work(baseline, {4: coordinate})
        after, new_cost = emitted_work(output, {4: coordinate})
        assert before.trace == after.trace
        assert sum(new_cost) < sum(old_cost)


def invocation_scaling_fixture(case_count):
    """Empty activation frames partition cases without adding payload hazards."""
    original, logical = case_only_fixture()
    span = original.span
    block = cast(ir.AssignStmt, cast(ir.SeqStmts, original.body).stmts[0]).var
    prefixes = []
    for boundary in range(1, case_count):
        iv = ir.Var(f"activation_{boundary}", block.type, span)
        prefixes.append(
            ir.ForStmt(
                iv,
                block,
                ir.ConstInt(boundary, ir.DataType.INDEX, span),
                ir.ConstInt(case_count, ir.DataType.INDEX, span),
                [],
                ir.SeqStmts([ir.YieldStmt([], span)], span),
                [],
                span,
            )
        )

    def replace(function):
        body = ir.SeqStmts(
            [
                *cast(ir.SeqStmts, function.body).stmts[:3],
                *prefixes,
                *cast(ir.SeqStmts, function.body).stmts[-2:],
            ],
            span,
        )
        return clone(function, body=body)

    return replace(original), replace(logical)


@pytest.mark.parametrize("case_count", [2, 4, 8, 16])
def test_invocation_profile_scaling_keeps_actual_case_protocols(case_count):
    original, logical = invocation_scaling_fixture(case_count)
    parameters = {4: (0, 0)}
    blocks = (0, case_count - 1)
    baseline = lower(logical, "guarded", parameter_bounds=parameters, block_bounds=blocks)
    result = lower(logical, "case", parameter_bounds=parameters, block_bounds=blocks)
    assert len(result["physical_cases"]) == case_count
    assert any(row["mode"] == "case" for row in selected(result))
    assert fingerprint(result) == fingerprint(baseline)
    for block in (0, 1, case_count - 1):
        check(original, logical, result["function"], {4: 0}, block=block, minimum_protocol=False)
        assert (
            trace_control(result["function"], {4: 0}, block).trace
            == trace_control(baseline["function"], {4: 0}, block).trace
        )


def test_growth_budget_is_cumulative_and_source_order_is_stable_after_serialization():
    original, logical = case_only_fixture()

    def twice(function):
        loop = cast(ir.SeqStmts, function.body).stmts[-2]
        copied, _ = ir.deep_clone(loop)

        class DistinctCalls(ir.IRMutator):
            def visit_call(self, op):
                return ir.Call(op.op, op.args, op.kwargs, op.type, op.span)

        # DeepClone can share immutable calls whose captured operands do not
        # change. Each repeated payload needs its own source cut identity.
        copied = DistinctCalls().visit_stmt(copied)
        return clone(
            function,
            body=ir.SeqStmts(
                [
                    *cast(ir.SeqStmts, function.body).stmts[:-1],
                    copied,
                    cast(ir.SeqStmts, function.body).stmts[-1],
                ],
                function.span,
            ),
        )

    original, logical = twice(original), twice(logical)
    first_loop = cast(ir.SeqStmts, original.body).stmts[-3]
    extra = ExpandedCensus()
    extra.visit_stmt(first_loop)
    policy = {
        "max_extra_payload_sites": extra.payloads,
        "parameter_bounds": {4: (0, 0)},
        "block_bounds": (0, 1),
    }
    result = lower(logical, "case", **policy)
    chosen = selected(result)
    assert [row["mode"] for row in chosen] == ["case", "guarded"]
    assert len({row["loop"] for row in chosen}) == 2
    assert any(
        row["mode"] == "case" and row["status"] == "over_budget" for row in result["schedule"]["choices"]
    )
    restored = ir.deserialize(ir.serialize(logical))
    assert isinstance(restored, ir.Function)
    repeated = lower(restored, "case", **policy)
    assert result["schedule"]["choices"] == repeated["schedule"]["choices"]
    for block in (0, 1):
        check(original, logical, result["function"], {4: 0}, block=block, minimum_protocol=False)


@pytest.mark.parametrize("block", [0, 1])
def test_complete_case_cost_delta_matches_independent_active_execution(block):
    original, logical = case_only_fixture()
    parameters = {4: (0, 0)}
    blocks = (0, 1)
    baseline = lower(logical, "guarded", parameter_bounds=parameters, block_bounds=blocks)
    candidate = lower(logical, "case", parameter_bounds=parameters, block_bounds=blocks)
    before_rows, after_rows = selected(baseline), selected(candidate)
    assert len(before_rows) == len(after_rows) == 1
    before, after = before_rows[0], after_rows[0]
    assert after["mode"] == "case" and after["model_comparable"]
    assert before["model_exact"] and after["model_exact"]
    # The main loop executes once in both valuations. Common prefix/payload
    # work cancels; a conditional per-entry model is not an invocation total.
    old, old_work = emitted_work(baseline["function"], {4: 0}, block)
    new, new_work = emitted_work(candidate["function"], {4: 0}, block)
    assert old.trace == new.trace
    expected = sum(before["model_cost"][block]) - sum(after["model_cost"][block])
    assert expected > 0
    assert sum(old_work) - sum(new_work) == expected
    check(original, logical, candidate["function"], {4: 0}, block=block, minimum_protocol=False)


@pytest.mark.parametrize("mode", ["auto", "case"])
def test_actual_token_lowering_refuses_interior_launch_predicate_before_scheduling(mode):
    _, logical = case_only_fixture()
    block = cast(ir.AssignStmt, cast(ir.SeqStmts, logical.body).stmts[0]).var

    class InteriorPredicate(ir.IRMutator):
        def visit_for_stmt(self, op):
            if op.start.same_as(block):
                # On blocks 0..3 this prefix executes only at block zero.
                # Step four preserves those visits and admits the full launch
                # interval, so this test reaches the interior guard contract.
                return ir.ForStmt(
                    op.loop_var,
                    op.start,
                    op.stop,
                    ir.ConstInt(4, ir.DataType.INDEX, op.span),
                    op.iter_args,
                    self.visit_stmt(op.body),
                    op.return_vars,
                    op.span,
                )
            return super().visit_for_stmt(op)

        def visit_if_stmt(self, op):
            condition = op.condition
            if isinstance(condition, ir.Lt) and condition.left.same_as(block):
                condition = ir.Eq(
                    block, ir.ConstInt(2, ir.DataType.INDEX, op.span), ir.DataType.BOOL, op.span
                )
            return ir.IfStmt(condition, op.then_body, op.else_body, op.return_vars, op.span)

    logical = InteriorPredicate().visit_function(logical)
    # The activation prefix partitions [0,3] into [0,0] and [1,3]. Equality
    # at 2 splits the latter's interior; neither endpoint sample proves it.
    # Actual-token qualification refuses before candidate costing, even forced.
    with pytest.raises(ValueError, match="Scoped invocation guard is not uniform over the complete case"):
        lower(logical, mode, parameter_bounds={4: (0, 0)}, block_bounds=(0, 3))


@pytest.mark.parametrize("kind", ["versioned", "multiphase"])
def test_parent_gap_does_not_cross_dispatch_or_multiple_phases(kind):
    if kind == "versioned":
        original, logical = case_only_fixture()
        mode = "case"
        parameters = {4: (0, 0)}
        blocks = (0, 1)
        values = [({4: 0}, block) for block in (0, 1)]
    else:
        original = streamed_fixture(7, start=5, step=2)
        logical = _apply(passes.insert_sync, original)
        mode, values = "boundary_periodic", [({}, None)]
        parameters = {}
        blocks = None
    result = lower(logical, mode, parameter_bounds=parameters, block_bounds=blocks)
    assert_receipt(result, mode)
    assert any(row["mode"] == mode for row in selected(result))
    assert not result["schedule"]["parent_gaps"]
    for parameters, block in values:
        check(
            original,
            logical,
            result["function"],
            parameters,
            block=block,
            minimum_protocol=kind != "versioned",
        )


def test_repeated_static_loop_pointer_is_rejected_before_parent_composition():
    _, logical = case_only_fixture()
    loop = cast(ir.SeqStmts, logical.body).stmts[-2]
    repeated = clone(
        logical,
        body=ir.SeqStmts(
            [*cast(ir.SeqStmts, logical.body).stmts[:-1], loop, cast(ir.SeqStmts, logical.body).stmts[-1]],
            logical.span,
        ),
    )
    snapshot = ir.serialize(repeated)
    # This malformed actual-token program fails SSA admission first. It does
    # not directly exercise the defensive static-loop census in the planner.
    with pytest.raises(ValueError, match="multiple SSA definitions"):
        lower(repeated, "guarded", parameter_bounds={4: (0, 0)}, block_bounds=(0, 1))
    assert ir.serialize(repeated) == snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
