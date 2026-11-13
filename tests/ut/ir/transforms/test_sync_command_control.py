# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Control sharing preserves ordered consuming commands, not just graph closure."""

import pytest
from pypto import codegen, ir, passes, testing

from .qualify_token_production import program_of
from .sync_control_metrics import ControlTrace, static_control, trace_control
from .sync_physical_oracle import run_physical
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_parameterized_token_sync import apply, check, worker
from .test_qwen_prefill_pipeline import activation_fixture
from .test_streamed_slot_composition import streamed_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def command(identifier=0, source=ir.PipeType.MTE2, destination=ir.PipeType.M):
    return ir.EvalStmt(
        ir.Call(
            ir.get_op("system.set_flag"),
            [],
            {
                "src_pipe": int(source),
                "dst_pipe": int(destination),
                "event_id": identifier,
                "core_type": "aic",
            },
            ir.TupleType([]),
            SPAN,
        ),
        SPAN,
    )


def seq(*children):
    return ir.SeqStmts(list(children), SPAN)


def guarded(condition, *children):
    return ir.IfStmt(condition, seq(*children), None, [], SPAN)


def evaluate(statement, environment):
    state = ControlTrace()
    state.execute(statement, dict(environment))
    return state


def test_consecutive_prefixes_keep_duplicate_leaves_and_guard_order():
    a = ir.Var("a", ir.ScalarType(ir.DataType.BOOL), SPAN)
    b = ir.Var("b", a.type, SPAN)
    original = seq(
        guarded(a, command()),
        guarded(a, guarded(b, command(1))),
        guarded(a, guarded(b, command(1))),
        guarded(b, command(2)),
        guarded(a, command(3)),
    )
    output = testing.share_sync_command_guards(original)
    assert isinstance(output, ir.SeqStmts)
    # Share the initial A prefix and consecutive nested B prefixes, but never
    # collect the trailing A through B. Both identical ID-1 leaves survive.
    assert len(output.stmts) == 3
    first = output.stmts[0]
    assert isinstance(first, ir.IfStmt)
    assert isinstance(first.then_body, ir.SeqStmts)
    assert len(first.then_body.stmts) == 2
    nested = first.then_body.stmts[1]
    assert isinstance(nested, ir.IfStmt)
    assert isinstance(nested.then_body, ir.SeqStmts)
    assert len(nested.then_body.stmts) == 2
    for av in (False, True):
        for bv in (False, True):
            before, after = evaluate(original, {a: av, b: bv}), evaluate(output, {a: av, b: bv})
            assert before.trace == after.trace
            assert after.counts["conditions"] <= before.counts["conditions"]


def test_free_variable_identity_and_shared_else():
    a = ir.Var("same", ir.ScalarType(ir.DataType.BOOL), SPAN)
    b = ir.Var("same", a.type, SPAN)
    choice = ir.IfStmt(a, command(1), command(2), [], SPAN)
    original = seq(guarded(a, command()), guarded(b, command()), choice, guarded(a, command(3)))
    output = testing.share_sync_command_guards(original)
    assert isinstance(output, ir.SeqStmts)
    assert len(output.stmts) == 3
    for av in (False, True):
        for bv in (False, True):
            assert evaluate(original, {a: av, b: bv}).trace == evaluate(output, {a: av, b: bv}).trace


def test_adjacent_nested_decisions_share_both_arms_without_deduplication():
    a = ir.Var("outer", ir.ScalarType(ir.DataType.BOOL), SPAN)
    b = ir.Var("inner", a.type, SPAN)

    def choice(condition, yes, no):
        return ir.IfStmt(condition, yes, no, [], SPAN)

    original = seq(
        choice(a, command(0), choice(b, command(1), command(2))),
        choice(a, command(0), choice(b, command(3), guarded(a, command(4)))),
        choice(a, command(5), guarded(b, command(1))),
    )
    output = testing.share_sync_command_guards(original)
    assert len(output.stmts) == 1
    assert len(output.stmts[0].then_body.stmts) == 3
    assert len(output.stmts[0].else_body.stmts) == 1
    for av in (False, True):
        for bv in (False, True):
            before, after = evaluate(original, {a: av, b: bv}), evaluate(output, {a: av, b: bv})
            assert before.trace == after.trace
            assert after.counts["conditions"] < before.counts["conditions"]


def test_unconditional_command_fences_decision_sharing():
    a = ir.Var("a", ir.ScalarType(ir.DataType.BOOL), SPAN)
    decision = ir.IfStmt(a, command(1), command(2), [], SPAN)
    original = seq(decision, command(3), decision)
    output = testing.share_sync_command_guards(original)
    assert len(output.stmts) == 3
    for av in (False, True):
        assert evaluate(original, {a: av}).trace == evaluate(output, {a: av}).trace


def test_partial_residue_decisions_do_not_acquire_an_exhaustive_else():
    # Only the ID-table producer knows that every residue was enumerated.
    # The generic sharing service must preserve holes even for modulo syntax.
    x = ir.Var("x", ir.ScalarType(ir.DataType.INDEX), SPAN)

    def literal(value):
        return ir.ConstInt(value, ir.DataType.INDEX, SPAN)

    residue = ir.FloorMod(x, literal(3), ir.DataType.INDEX, SPAN)
    zero = ir.Eq(residue, literal(0), ir.DataType.BOOL, SPAN)
    two = ir.Eq(residue, literal(2), ir.DataType.BOOL, SPAN)
    decision = ir.IfStmt(zero, command(0), guarded(two, command(2)), [], SPAN)
    original = seq(decision, decision)
    output = testing.share_sync_command_guards(original)
    for value in range(9):
        before, after = evaluate(original, {x: value}), evaluate(output, {x: value})
        assert before.trace == after.trace
        if value % 3 == 1:
            assert not after.trace


def test_unproved_decision_expression_keeps_separate_trees():
    x = ir.Var("x", ir.ScalarType(ir.DataType.INDEX), SPAN)
    divisor = ir.Var("divisor", x.type, SPAN)
    residue = ir.FloorMod(x, divisor, ir.DataType.INDEX, SPAN)
    condition = ir.Eq(residue, ir.ConstInt(0, ir.DataType.INDEX, SPAN), ir.DataType.BOOL, SPAN)
    choice = ir.IfStmt(condition, command(0), command(1), [], SPAN)
    original = seq(choice, choice)
    output = testing.share_sync_command_guards(original)
    # The descriptor service has no positive-divisor premise for this value.
    assert len(output.stmts) == 2
    for value in range(6):
        for period in (1, 2, 3):
            env = {x: value, divisor: period}
            assert evaluate(original, env).trace == evaluate(output, env).trace


def test_unavailable_structural_comparison_keeps_unfactored_plan():
    x = ir.Var("x", ir.ScalarType(ir.DataType.INDEX), SPAN)
    expression = x
    for _ in range(520):
        expression = ir.Add(expression, ir.ConstInt(1, ir.DataType.INDEX, SPAN), ir.DataType.INDEX, SPAN)
    condition = ir.Lt(expression, ir.ConstInt(999, ir.DataType.INDEX, SPAN), ir.DataType.BOOL, SPAN)
    original = seq(guarded(condition, command()), guarded(condition, command(1)))
    output = testing.share_sync_command_guards(original)
    assert isinstance(output, ir.SeqStmts)
    # Optional equality has its own traversal limit. Do not turn that into a
    # new physical-export refusal or assume equality without a proof.
    assert len(output.stmts) == 2


def test_invocation_hole_is_not_coalesced_into_active_domain():
    original = activation_fixture()
    owner = testing.analyze_sync_invocation(original, True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 7))
    assert [c["block_bounds"] for c in owner.describe()["cases"]] == [(0, 1), (2, 4), (5, 7)]
    result = owner.realize_physical("a2a3", list(range(6)))
    for block in range(8):
        state, _, _ = run_physical(original, result["function"], range(6), {1: 0, 3: 32}, block_index=block)
        assert bool(state.records) == (block < 2 or block >= 5)
    stats = static_control(result["function"])
    assert result["if_sites"] == stats["if_sites"]
    assert result["set_sites"] == stats[ir.get_op("system.set_flag").name]


@pytest.mark.parametrize("count", [0, 1, 7, 70, 7000, 1000000])
def test_residue_selection_stays_compact_and_matches_original_accesses(count):
    original = streamed_fixture(count, start=5, step=2)
    physical = worker(apply(passes.realize_sync, program_of(original)), original.name)
    if count == 0:
        # Existing empty rotating ownership delegates transactionally. Guard
        # normalization must neither invent commands nor broaden admission.
        assert not dict(physical.attrs)["pypto_autosync"]
        ir.assert_structural_equal(original.body, physical.body)
        return
    assert dict(physical.attrs)["pypto_autosync"]
    stats = static_control(physical)
    # The no-clone production path retains compact per-cut ID dispatch;
    # this inherited branch bound is conservative for the guarded default.
    assert stats["if_sites"] <= 3 * 20
    if count <= 7:
        run_physical(original, physical, range(6))
    if count == 7:
        state = trace_control(physical)
        # Ten executions reach the exhaustive final ID arm; its equality
        # test is no longer constructed. Endpoint presence tests remain.
        assert state.counts["conditions"] == 70
        logical = worker(apply(passes.insert_sync, program_of(original)), original.name)
        specialized = testing.lower_sync_schedule(
            logical,
            schedule_policy=testing.SyncSchedulePolicy(mode="boundary_periodic", no_clone=False),
        )["function"]
        run_physical(original, specialized, range(6))
        optimized = trace_control(specialized)
        assert optimized.trace == state.trace
        assert optimized.counts["conditions"] == 31


@pytest.mark.parametrize("band,bound", [(0, 111), (1, 115), (2, 147), (3, 149)])
def test_qwen_whole_cut_coalescing_reduces_control_without_new_coloring(band, bound):
    program = capture_qwen_projection("gate_up", layers=2, hidden=5120, band=band)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    physical = worker(apply(passes.realize_sync, program), original.name)
    stats = static_control(physical)
    assert stats["if_sites"] <= bound
    # These conservative site bounds also covered the prior versioned output.
    # Separate exact regressions enforce the new no-clone default.
    command_bound = [92, 92, 96, 96][band]
    assert stats["system.set_flag"] <= command_bound
    assert stats["system.wait_flag"] <= command_bound
    if band == 2:
        # The guarded default retains this band's residual guards; bounded
        # source control complements the independent domain/trace checks.
        assert stats["system.set_flag"] <= 96
        assert stats["guard_expression_visits"] <= 485
    source = codegen.PTOCodegen().generate(program_of(physical))
    assert source.count("scf.if") == stats["if_sites"]
    # Large cases use source-control enumeration here. Bounded independent
    # access/matching/reuse checks remain in test_invocation_token_sync.
    for block in (0, 11, 19, 23):
        state = trace_control(physical, {3: block % 2}, block)
        assert (
            state.counts[ir.get_op("system.set_flag").name]
            == state.counts[ir.get_op("system.wait_flag").name]
        )


def test_qwen_selected_loop_entry_shares_decision_without_changing_actual_tokens():
    program = capture_qwen_projection("gate_up", layers=2, hidden=512, band=2)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    logical = worker(apply(passes.insert_sync, program), original.name)
    evidence = testing.sync_invocation_evidence(program)
    options = {
        "parameter_bounds": evidence["parameters"][original.name],
        "block_bounds": tuple(evidence["block_indices"][original.name]),
    }
    result = testing.lower_sync_schedule(logical, **options)
    # Without boundary extraction the command remains inside the first visit;
    # ordered payload/command traces still provide an independent placement check.
    unhoisted = testing.lower_sync_schedule(
        logical, **options, schedule_policy=testing.SyncSchedulePolicy(boundary_motion=False)
    )
    assert not unhoisted["schedule"]["parent_gaps"]
    gaps = result["schedule"]["parent_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["before"] == [4, 18, 14]
    assert gaps[0]["after"] == [2, 12, 12]
    assert gaps[0]["command_sites"] == 6
    chosen = {row["loop"]: row for row in result["schedule"]["choices"] if row["selected"]}
    assert chosen[gaps[0]["loop"]]["model_scope"] == "standalone_candidate_before_parent_gap_composition"
    assert static_control(result["function"])["if_sites"] == 67
    for layer in (0, 1):
        for block in range(24):
            actual = trace_control(result["function"], {3: layer}, block)
            assert actual.trace == trace_control(unhoisted["function"], {3: layer}, block).trace
        check(original, logical, result["function"], {3: layer}, block=0 if layer == 0 else 23)


def test_unsupported_control_before_loop_remains_unavailable():
    program = capture_qwen_projection("gate_up", layers=2, hidden=512, band=2)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    logical = worker(apply(passes.insert_sync, program), original.name)
    evidence = testing.sync_invocation_evidence(program)

    class FenceLoops(ir.IRMutator):
        def visit_for_stmt(self, op):
            nested = super().visit_for_stmt(op)
            constant = ir.ConstInt(0, ir.DataType.INDEX, SPAN)
            return seq(guarded(ir.Eq(logical.params[3], constant, ir.DataType.BOOL, SPAN)), nested)

    fenced = FenceLoops().visit_function(logical)
    # This guard is rejected by scoped-token admission, before parent-gap
    # planning. Retained statement fences are tested independently by the
    # executable-gap tests in test_sync_boundary_schedule.
    with pytest.raises(ValueError, match="Scoped token guard requires an active original coordinate"):
        testing.lower_sync_schedule(
            fenced,
            parameter_bounds=evidence["parameters"][original.name],
            block_bounds=tuple(evidence["block_indices"][original.name]),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
