# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capture-aware loop closure checked by replaying one pipe fact around each backedge."""

from itertools import permutations

import pytest
from pypto import ir, testing
from pypto.ir.op import sync

from .sync_closed_command_oracle import ClosedCommands

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()
LOAD, VECTOR, STORE = ir.PipeType.MTE2, ir.PipeType.V, ir.PipeType.MTE3
DIRECTIONS = {1: (LOAD, VECTOR), 2: (VECTOR, STORE), 3: (STORE, LOAD)}
SET = ir.get_op("system.set_flag").name
WAIT = ir.get_op("system.wait_flag").name


def seq(*statements: ir.Stmt) -> ir.SeqStmts:
    return ir.SeqStmts(list(statements), SPAN)


def record_and_wait(name: str, source: ir.PipeType, destination: ir.PipeType) -> tuple[ir.Stmt, ir.Stmt]:
    token = ir.Var(name, ir.SyncTokenType(source), SPAN)
    return ir.AssignStmt(token, sync.record(source), SPAN), ir.EvalStmt(sync.wait(token, destination), SPAN)


def loop_function(body: ir.SeqStmts, params: list[ir.Var]) -> ir.Function:
    index = ir.DataType.INDEX
    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(index), SPAN),
        ir.ConstInt(0, index, SPAN),
        ir.ConstInt(3, index, SPAN),
        ir.ConstInt(1, index, SPAN),
        [],
        seq(*body, ir.YieldStmt([], SPAN)),
        [],
        SPAN,
    )
    return ir.Function(
        "capture_cuts",
        params,
        [],
        seq(loop, ir.ReturnStmt([], SPAN)),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def scan_fact(segment: tuple[int, ...], reached: set[ir.PipeType]) -> None:
    """Advance one fact within a segment, starting with no captures from outside it."""
    captured = set()
    for event in segment:
        source, destination = DIRECTIONS[abs(event)]
        if event > 0 and source in reached:
            captured.add(event)
        elif event < 0 and -event in captured:
            reached.add(destination)


def expected_commands(events: tuple[int, ...]) -> list[tuple[str, ir.PipeType, ir.PipeType]]:
    commands = []
    for position, event in enumerate(events):
        source, destination = DIRECTIONS[abs(event)]
        if event > 0:
            commands.append((SET, source, destination))
            continue
        commands.append((WAIT, source, destination))
        reached = {destination}
        scan_fact(events[position + 1 :], reached)
        scan_fact(events[: events.index(-event)], reached)
        if source not in reached:
            commands.extend([(SET, destination, source), (WAIT, destination, source)])
    return commands


def loop_commands(function: ir.Function) -> list[tuple[str, ir.PipeType, ir.PipeType]]:
    assert isinstance(function.body, ir.SeqStmts)
    loop = function.body.stmts[0]
    assert isinstance(loop, ir.ForStmt)
    assert isinstance(loop.body, ir.SeqStmts)
    commands = []
    for statement in loop.body:
        if isinstance(statement, ir.YieldStmt):
            continue
        assert isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call)
        call = statement.expr
        commands.append(
            (call.op.name, ir.PipeType(call.kwargs["src_pipe"]), ir.PipeType(call.kwargs["dst_pipe"]))
        )
    return commands


def test_loop_suffix_matches_forward_capture_replay():
    # Exhaust all legal interleavings of the directed three-pipe cycle. Varying
    # the three capture positions also varies the wait edges that cross each
    # transaction's suffix boundary and the chains that traverse multiple waits.
    for events in permutations((1, -1, 2, -2, 3, -3)):
        if any(events.index(token) > events.index(-token) for token in DIRECTIONS):
            continue
        operations = {}
        for token, direction in DIRECTIONS.items():
            operations[token], operations[-token] = record_and_wait(f"t{token}", *direction)
        logical = loop_function(seq(*(operations[event] for event in events)), [])
        physical = testing.lower_sync_schedule(logical)["function"]
        assert loop_commands(physical) == expected_commands(events), events


@pytest.mark.parametrize("groups", [32, 256])
def test_staggered_capture_suffix_work_is_linear(groups: int):
    body = []
    for group in range(groups):
        a_record, a_wait = record_and_wait(f"a{group}", LOAD, VECTOR)
        b_record, b_wait = record_and_wait(f"b{group}", VECTOR, LOAD)
        body.extend([a_record, b_record, a_wait, b_wait])
    result = testing.lower_sync_schedule(loop_function(seq(*body), []))
    (physical,) = result["physical_cases"]
    assert physical["set_sites"] == physical["wait_sites"] == 2 * groups

    # The outer sequence contains a loop and return. Its loop body adds a yield
    # and four cuts per group. The two captures in each group add two more visits.
    assert physical["closed_prefix_suffix_visits"] == 3 + 6 * groups


@pytest.mark.parametrize("late_capture", [False, True])
@pytest.mark.parametrize("both_arms", [False, True])
def test_loop_suffix_capture_spanning_nested_branch(late_capture: bool, both_arms: bool):
    probe_record, probe_wait = record_and_wait("probe", LOAD, VECTOR)
    return_record, return_wait = record_and_wait("return_token", STORE, LOAD)
    choice = ir.Var("choice", ir.ScalarType(ir.DataType.BOOL), SPAN)
    then_body = seq(*record_and_wait("then_transfer", VECTOR, STORE))
    else_body = seq(*record_and_wait("else_transfer", VECTOR, STORE)) if both_arms else None
    branch = ir.IfStmt(choice, then_body, else_body, [], SPAN)
    middle = [branch, return_record] if late_capture else [return_record, branch]
    logical = loop_function(seq(probe_record, probe_wait, *middle, return_wait), [choice])
    result = testing.lower_sync_schedule(logical)

    # A capture before the branch cannot forward the branch's completion fact.
    # A late capture lets each arm's transfer close via the outer edges. Those
    # outer edges also close when the branch provides V -> MTE3 on both paths.
    transfers = 4 if both_arms else 3
    acknowledgments = transfers if not late_capture else (0 if both_arms else 2)
    assert result["set_sites"] == result["wait_sites"] == transfers + acknowledgments
    for condition in (False, True):
        ClosedCommands(result["function"], {0: condition}, range(8)).run()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
