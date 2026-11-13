# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualified boundary scheduling preserves complete ordered execution traces.

Synthetic cut tests exercise lowering only; the production campaign separately
checks matching and physical reuse against independently recovered accesses.
"""

import pytest
from pypto import ir, testing

from .sync_control_metrics import statements
from .test_sync_command_control import SPAN, command, evaluate, guarded, seq
from .test_sync_cut_control import ci, owner, variable

pytestmark = pytest.mark.usefixtures("ascend_backend")


def payload():
    # A distinct supplied anchor: this diagnostic never invokes access recovery.
    return ir.EvalStmt(ir.Call(ir.get_op("buffer.matmul"), [], {}, ir.TupleType([]), SPAN), SPAN)


def eq(x, n):
    return ir.Eq(x, ci(n), ir.DataType.BOOL, SPAN)


def loop(i, start, stop, step, body):
    return ir.ForStmt(i, ci(start), ci(stop), ci(step), [], body, [], SPAN)


def emit(body, cuts, enabled=True):
    return testing.emit_sync_boundary_commands(owner(body), cuts, enabled)


def compare(body, cuts, environments=({},)):
    before = emit(body, cuts, False)
    after = emit(body, cuts)
    for environment in environments:
        original = evaluate(before["function"].body, environment)
        revised = evaluate(after["function"].body, environment)
        assert original.trace == revised.trace
        assert revised.counts["conditions"] <= original.counts["conditions"]
    return after


@pytest.mark.parametrize("start,stop,step", [(0, 4, 1), (7, 23, 5), (-9, 4, 3)])
def test_first_last_commands_keep_nonunit_original_lattice_and_group_order(start, stop, step):
    i = variable("i")
    a, b = payload(), payload()
    last = list(range(start, stop, step))[-1]
    before = seq(
        guarded(ir.Gt(i, ci(start), ir.DataType.BOOL, SPAN), command(0)),
        guarded(eq(i, start), command(1), command(2)),
    )
    after = seq(
        guarded(ir.Lt(i, ci(last), ir.DataType.BOOL, SPAN), command(3)),
        guarded(eq(i, last), command(4), command(5)),
    )
    result = compare(loop(i, start, stop, step, seq(a, b)), [(a.expr, before, seq()), (b.expr, seq(), after)])
    assert result["scheduled_loops"] == 1
    assert result["schedule_boundary_commands"] == 4
    assert sum(isinstance(s, ir.ForStmt) for s in statements(result["function"].body)) == 1


@pytest.mark.parametrize("trips", [0, 1, 2, 9])
def test_outer_guards_skips_and_counter_resets(trips):
    i, outer = variable("i"), variable("visit")
    active = ir.Eq(ir.FloorMod(outer, ci(2), ir.DataType.INDEX, SPAN), ci(0), ir.DataType.BOOL, SPAN)
    a = payload()
    cuts = [(a.expr, guarded(eq(i, 0), command(1)), guarded(active, guarded(eq(i, trips - 1), command(2))))]
    body = loop(outer, 0, 4, 1, guarded(active, loop(i, 0, trips, 1, seq(a))))
    result = compare(body, cuts)
    assert result["scheduled_loops"] == int(trips >= 2)


def test_single_payload_distinct_entry_exit_and_surviving_outer_predicate():
    i, outer = variable("i"), variable("outer")
    a = payload()
    cuts = [
        (
            a.expr,
            guarded(eq(i, 0), command(1)),
            guarded(eq(outer, 1), guarded(eq(i, 3), command(2), command(3))),
        )
    ]
    result = compare(loop(i, 0, 4, 1, seq(a)), cuts, [{outer: x} for x in range(3)])
    assert result["schedule_boundary_commands"] == 3


def test_active_command_fences_boundary_motion():
    i = variable("i")
    a = payload()
    cuts = [
        (
            a.expr,
            seq(command(0), guarded(eq(i, 0), command(1))),
            seq(guarded(eq(i, 3), command(2)), command(3)),
        )
    ]
    result = compare(loop(i, 0, 4, 1, seq(a)), cuts)
    assert result["scheduled_loops"] == 0


def test_second_payload_readiness_cannot_cross_first_payload():
    i = variable("i")
    a, b = payload(), payload()
    result = compare(
        loop(i, 0, 4, 1, seq(a, b)), [(a.expr, seq(), seq()), (b.expr, guarded(eq(i, 0), command(1)), seq())]
    )
    assert result["scheduled_loops"] == 0


def test_guard_holes_and_else_commands_do_not_become_boundary_only():
    i = variable("i")
    a = payload()
    sparse = ir.Or(eq(i, 0), eq(i, 2), ir.DataType.BOOL, SPAN)
    choice = ir.IfStmt(eq(i, 0), command(2), command(3), [], SPAN)
    cuts = [(a.expr, seq(guarded(sparse, command(1)), choice), seq())]
    result = compare(loop(i, 0, 4, 1, seq(a)), cuts)
    assert result["scheduled_loops"] == 0


@pytest.mark.parametrize("unsupported", ["definition", "unknown_effect", "conditional", "duplicate_anchor"])
def test_unproved_body_scope_keeps_original_cuts(unsupported):
    i, x = variable("i"), variable("local")
    a, unknown = payload(), payload()
    middle = {
        "definition": ir.AssignStmt(x, i, SPAN),
        "unknown_effect": unknown,
        "conditional": guarded(eq(i, 1), unknown),
        "duplicate_anchor": a,
    }[unsupported]
    cuts = [(a.expr, guarded(eq(i, 0), command(1)), seq())]
    result = compare(loop(i, 0, 4, 1, seq(middle, a)), cuts)
    assert result["scheduled_loops"] == 0


def test_retained_control_and_inspected_work_do_not_expand_trip_counts():
    measurements = []
    for trips in (4, 400, 4000000):
        i, a = variable("i"), payload()
        result = emit(
            loop(i, 0, trips, 1, seq(a)),
            [(a.expr, guarded(eq(i, 0), command(1)), guarded(eq(i, trips - 1), command(2)))],
        )
        assert result["scheduled_loops"] == 1
        measurements.append(
            tuple(
                result[name]
                for name in (
                    "schedule_body_visits",
                    "schedule_descriptor_visits",
                    "materialized_control_statements",
                    "schedule_boundary_commands",
                )
            )
        )
    assert measurements[0] == measurements[1] == measurements[2]


def test_symbolic_and_overflowing_iteration_domains_remain_unproved():
    i, stop = variable("i"), variable("stop")
    a = payload()
    cuts = [(a.expr, guarded(eq(i, 0), command(1)), seq())]
    symbolic = ir.ForStmt(i, ci(0), stop, ci(1), [], seq(a), [], SPAN)
    assert emit(symbolic, cuts)["scheduled_loops"] == 0
    maximum = (1 << 63) - 1
    overflowing = loop(i, maximum - 5, maximum, 4, seq(a))
    # Do not execute an overflowing reference loop: only check conservative refusal.
    assert (
        emit(overflowing, [(a.expr, guarded(eq(i, maximum - 5), command(1)), seq())])["scheduled_loops"] == 0
    )


def test_negative_step_keeps_original_lowering():
    i, a = variable("i"), payload()
    result = compare(loop(i, 4, 0, -1, seq(a)), [(a.expr, guarded(eq(i, 4), command(1)), seq())])
    assert result["scheduled_loops"] == 0


@pytest.mark.parametrize("fence", ["none", "definition", "payload", "branch", "duplicate"])
def test_decision_sharing_only_across_one_executable_command_gap(fence):
    a, b, unrelated = payload(), payload(), payload()
    choice = variable("choice")
    condition = eq(choice, 1)
    left = ir.IfStmt(condition, command(0), command(1), [], SPAN)
    right = ir.IfStmt(condition, command(2), command(3), [], SPAN)
    between = {
        "none": [],
        "definition": [ir.AssignStmt(variable("local"), ci(1), SPAN)],
        "payload": [unrelated],
        "branch": [guarded(condition, unrelated)],
        "duplicate": [a],
    }[fence]
    body = seq(a, *between, b)
    cuts = [(a.expr, seq(), left), (b.expr, right, seq())]
    result = compare(body, cuts, [{choice: value} for value in (0, 1, 2)])
    assert result["shared_composed_guard_sites"] == int(fence == "none")


@pytest.mark.parametrize("kind", ["active", "foreign", "definition", "private", "wrong_cut"])
def test_command_scope_uses_original_ready_coordinates(kind):
    i, other, scalar = variable("i"), variable("i"), variable("loaded")
    a = payload()
    predicate = eq(i if kind in ("active", "wrong_cut") else other if kind == "foreign" else scalar, 0)
    body = loop(i, 0, 3, 1, seq(a))
    if kind == "definition":
        body = seq(ir.AssignStmt(scalar, ci(0), SPAN), body)
    elif kind == "private":
        body = seq(guarded(eq(i, 0), ir.AssignStmt(scalar, ci(0), SPAN)), body)
    cut = payload().expr if kind == "wrong_cut" else a.expr
    supplied = [(cut, guarded(predicate, command()), seq())]
    if kind == "active":
        result = testing.emit_sync_boundary_commands(owner(body), supplied, validate_scope=True)
        assert result["scheduled_loops"] == 1
    else:
        reason = "no original owner scope" if kind == "wrong_cut" else "unavailable original-cut coordinate"
        with pytest.raises(ValueError, match=reason):
            testing.emit_sync_boundary_commands(owner(body), supplied, validate_scope=True)


def test_command_descriptors_reject_effectful_guards_and_payload_leaves():
    a = payload()
    effect = ir.Call(ir.get_op("buffer.get_block_idx"), [], {}, ir.ScalarType(ir.DataType.INDEX), SPAN)
    with pytest.raises(ValueError, match="pure supported coordinate expression"):
        emit(seq(a), [(a.expr, guarded(eq(effect, 0), command()), seq())])
    with pytest.raises(ValueError, match="physical synchronization leaf"):
        emit(seq(a), [(a.expr, seq(payload()), seq())])


def test_unproved_arithmetic_stays_opaque_and_cannot_gain_production_scope():
    i, divisor = variable("i"), variable("divisor")
    condition = eq(ir.FloorMod(i, divisor, ir.DataType.INDEX, SPAN), 0)
    commands = seq(guarded(condition, command(0)), guarded(condition, command(1)))
    # Even a then-only tree cannot bypass the missing positive-divisor proof.
    # Diagnostic comparison preserves the original plan; production refuses it.
    assert len(testing.share_sync_command_guards(commands).stmts) == 2
    a = payload()
    with pytest.raises(ValueError, match="Unproved command predicate arithmetic"):
        testing.emit_sync_boundary_commands(
            owner(loop(i, 0, 3, 1, seq(a))), [(a.expr, commands, seq())], validate_scope=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
