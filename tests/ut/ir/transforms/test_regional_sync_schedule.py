# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A shared symbolic schedule precedes physical command expansion."""

import pytest
from pypto import ir, testing

from .sync_control_metrics import statements
from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import hardware_reference, interpret
from .sync_physical_oracle import run_physical
from .test_rotating_sync_analysis import rotating_fixture
from .test_streamed_slot_composition import streamed_fixture
from .test_sync_command_control import SPAN, command, guarded, seq
from .test_sync_cut_control import ci, owner, variable

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("count", [7, 70, 7000, 1000000])
def test_symbolic_inventory_is_independent_of_id_and_trip_expansion(count):
    original = streamed_fixture(count)
    retained = testing.analyze_placed_sync(original, True)
    description = retained.describe()
    physical = retained.realize_physical(
        "a2a3",
        list(range(6)),
        schedule_policy=testing.SyncSchedulePolicy(mode="boundary_periodic", no_clone=False),
    )
    assert len(physical["families"]) == 3
    assert physical["assigned_schedule_endpoints"] == 6
    # P=3 bodies plus at most three guarded boundary bodies, each with at
    # most nine commands per side. This is independent of the trip count.
    assert physical["set_sites"] <= 36 and physical["wait_sites"] <= 36
    assert physical["regular_schedule_body_copies"] <= 6
    assert physical["regular_schedule_loops"] == 1
    assert physical["regular_schedule_model_saving"] > 0
    assert retained.describe() == description
    if count == 7:
        run_physical(original, physical["function"], range(6))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6, 7, 8, 11, 17])
@pytest.mark.parametrize("start,step", [(0, 1), (5, 2)])
def test_phase_and_residue_schedule_preserves_accesses_matching_and_reuse(count, start, step):
    original = streamed_fixture(count, start=start, step=step)
    result = testing.analyze_placed_sync(original, True).realize_physical(
        "a2a3",
        list(range(6)),
        schedule_policy=testing.SyncSchedulePolicy(mode="boundary_periodic", no_clone=False),
    )
    run_physical(original, result["function"], range(6))
    steady = [
        s
        for s in statements(result["function"].body)
        if isinstance(s, ir.ForStmt) and "_sync_steady_" in s.loop_var.name_hint
    ]
    if result["regular_schedule_loops"]:
        assert len(steady) == 1
        # This fixture has no payload branches. Its steady body contains only
        # literal commands, the original slot accesses and coordinate bindings.
        assert not any(isinstance(s, ir.IfStmt) for s in statements(steady[0].body))
        assert result["regular_schedule_payload_copies"] == 3 * result["regular_schedule_body_copies"]
    else:
        assert not steady


def test_operand_oracle_rejects_changed_slot_even_with_same_command_trace():
    original = streamed_fixture(17)
    result = testing.analyze_placed_sync(original, True).realize_physical(
        "a2a3",
        list(range(6)),
        schedule_policy=testing.SyncSchedulePolicy(mode="boundary_periodic", no_clone=False),
    )

    class WrongSlot(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.slot").name:
                return ir.Call(
                    op.op, [op.args[0], ir.ConstInt(0, ir.DataType.INDEX, op.span)], op.type, op.span
                )
            return super().visit_call(op)

    with pytest.raises(AssertionError, match="changed executed payload operands"):
        run_physical(original, WrongSlot().visit_function(result["function"]), range(6))


@pytest.mark.parametrize("protected", [False, True])
def test_reporting_reference_retains_original_effects_and_hardware_scope(protected):
    original = staged_fixture(3, m=32 if protected else 16, n=80 if protected else 16)
    emitted = testing.analyze_placed_sync(original, True).realize_physical("a2a3", list(range(6)))
    state, _, order = run_physical(original, emitted["function"], range(6))
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    assert state.required_reference == reference
    assert state.required_reference[2] == order
    assert any(c[3] for c in state.original_calls)
    assert any(c[4] for c in state.original_calls)
    for before, retained in zip(calls, state.original_calls, strict=True):
        assert before[0].same_as(retained[0])
        assert before[3:] == retained[3:]


def invocation_owner():
    block = variable()
    return block, owner(
        seq(ir.AssignStmt(block, ir.Call(ir.get_op("buffer.get_block_idx"), [], {}, SPAN), SPAN))
    )


@pytest.mark.parametrize("middle", ["different_id", "empty"])
def test_complete_case_schedule_group_preserves_holes_and_empty_commands(middle):
    _, function = invocation_owner()
    a, b = seq(command(0)), seq(command(1)) if middle == "different_id" else seq()
    result = testing.inspect_sync_invocation_groups(function, [(0, 1, a), (2, 3, b), (4, 5, a)])
    assert result["groups"] == [[(0, 1), (4, 5)], [(2, 3)]]
    # Count descriptor import work, not only subsequent grouping work.
    assert result["descriptor_nodes"] >= 5


def test_grouping_preserves_identical_retained_launch_predicates():
    block, function = invocation_owner()
    body = guarded(ir.Lt(block, ci(2), ir.DataType.BOOL, SPAN), command())
    result = testing.inspect_sync_invocation_groups(function, [(0, 1, body), (2, 3, body)])
    assert result["groups"] == [[(0, 3)]]
    assert set(result) == {"groups", "descriptor_nodes", "comparisons"}


def test_grouping_retains_predicates_that_split_the_interior_of_a_case():
    block, function = invocation_owner()
    body = guarded(ir.Eq(block, ci(3), ir.DataType.BOOL, SPAN), command())
    result = testing.inspect_sync_invocation_groups(function, [(0, 6, body), (7, 8, seq(command(1)))])
    # Structural grouping keeps the equality guard intact; it cannot prove
    # its interior truth value or make a profitability decision.
    assert result["groups"] == [[(0, 6)], [(7, 8)]]


def test_missing_dominating_launch_value_keeps_cut_dispatch():
    result = testing.inspect_sync_invocation_groups(
        owner(seq()), [(0, 0, seq(command(0))), (1, 1, seq(command(1)))]
    )
    assert result["groups"] == []


@pytest.mark.parametrize("count", [2, 4, 8, 16, 32])
def test_structural_grouping_does_not_enumerate_launch_values(count):
    _, function = invocation_owner()
    cases: list[tuple[int, int, ir.Stmt]] = [
        (i * 1000000, (i + 1) * 1000000 - 1, seq(command(i % 2))) for i in range(count)
    ]
    result = testing.inspect_sync_invocation_groups(function, cases)
    assert len(result["groups"]) == 2
    assert sum(map(len, result["groups"])) == count
    assert result["descriptor_nodes"] == 2 * count
    assert result["comparisons"] <= 2 * count


def test_grouping_preserves_boolean_predicates_and_merges_only_equal_schedules():
    block, function = invocation_owner()
    retained = ir.Or(
        ir.Lt(block, ci(2), ir.DataType.BOOL, SPAN),
        ir.And(
            ir.Ge(block, ci(4), ir.DataType.BOOL, SPAN),
            ir.Eq(block, ci(5), ir.DataType.BOOL, SPAN),
            ir.DataType.BOOL,
            SPAN,
        ),
        ir.DataType.BOOL,
        SPAN,
    )
    a, b = guarded(retained, command(0)), seq(command(1))
    cases: list[tuple[int, int, ir.Stmt]] = [(0, 1, a), (2, 3, b), (4, 4, a), (5, 5, a)]
    result = testing.inspect_sync_invocation_groups(function, cases)
    assert result["groups"] == [[(0, 1), (4, 5)], [(2, 3)]]


@pytest.mark.parametrize("stages", [1, 2, 4, 8])
def test_static_stage_growth_is_charged_separately_from_trip_count(stages):
    counts = []
    for trips in (17, 1700):
        original = rotating_fixture(
            trips,
            [(3, 1)],
            [("load", 0, 0), *[("rmw", 0, 0)] * stages, ("store", 0, 0)],
            stream_input=True,
            stream_output=True,
        )
        result = testing.analyze_placed_sync(original, True).realize_physical(
            "a2a3",
            list(range(6)),
            schedule_policy=testing.SyncSchedulePolicy(mode="boundary_periodic", no_clone=False),
        )
        assert result["regular_schedule_loops"] == 1
        assert result["regular_schedule_body_copies"] <= 6
        assert result["regular_schedule_payload_copies"] <= 6 * (stages + 2)
        counts.append((result["assigned_schedule_endpoints"], result["regular_schedule_payload_copies"]))
        if trips == 17:
            run_physical(original, result["function"], range(6))
    assert counts[0] == counts[1]


def test_schedule_name_inventory_is_shared_across_emitted_copies():
    visits = []
    for count in (7, 70, 7000):
        original = streamed_fixture(count)
        result = testing.analyze_placed_sync(original, True).realize_physical(
            "a2a3",
            list(range(6)),
            schedule_policy=testing.SyncSchedulePolicy(mode="boundary_periodic", no_clone=False),
        )
        assert result["regular_schedule_body_copies"] > 1
        assert result["schedule_name_scans"] == 1
        visits.append(result["schedule_name_visits"])
    assert visits[0] > 0
    assert len(set(visits)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
