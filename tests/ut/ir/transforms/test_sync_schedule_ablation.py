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

import pytest
from pypto import ir, passes, testing

from .sync_control_metrics import COMMANDS, statements, trace_control
from .test_autosync_switch import clone
from .test_parameterized_token_sync import check
from .test_streamed_slot_composition import streamed_fixture
from .test_sync_case_only_schedule import case_only_fixture
from .test_sync_command_control import command, guarded, seq
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
                self.visit_expr(op.init_value)
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
    evidence = {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, 1)}
    baseline = lower(logical, "guarded", **evidence)
    result = lower(logical, mode, **evidence)
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
    evidence = {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, 1)}
    result = lower(logical, "case", **evidence, **policy)
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
        evidence = {}
    else:
        original, logical = case_only_fixture()
        evidence = {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, 1)}
    result = testing.lower_sync_schedule(logical, **evidence)
    baseline = lower(logical, "guarded", **evidence)
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
    payload = original.body.stmts[2]

    def owner(commands):
        return ir.Function(
            "predicate_sharing",
            [*zip(original.params, original.param_directions), iv],
            [],
            seq(*original.body.stmts[:2], commands, payload, ir.ReturnStmt([], span)),
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
    block = original.body.stmts[0].var
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
        body = ir.SeqStmts([*function.body.stmts[:3], *prefixes, *function.body.stmts[-2:]], span)
        return clone(function, body=body)

    return replace(original), replace(logical)


@pytest.mark.parametrize("case_count", [2, 4, 8, 16])
def test_invocation_profile_scaling_keeps_actual_case_protocols(case_count):
    original, logical = invocation_scaling_fixture(case_count)
    evidence = {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, case_count - 1)}
    baseline = lower(logical, "guarded", **evidence)
    result = lower(logical, "case", **evidence)
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
        loop = function.body.stmts[-2]
        copied, _ = ir.deep_clone(loop)

        class DistinctCalls(ir.IRMutator):
            def visit_call(self, op):
                return ir.Call(op.op, op.args, op.kwargs, op.type, op.span)

        # DeepClone can share immutable calls whose captured operands do not
        # change. Each repeated payload needs its own source cut identity.
        copied = DistinctCalls().visit_stmt(copied)
        return clone(
            function,
            body=ir.SeqStmts([*function.body.stmts[:-1], copied, function.body.stmts[-1]], function.span),
        )

    original, logical = twice(original), twice(logical)
    first_loop = original.body.stmts[-3]
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
