# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check completion and consuming event lifetimes across retained child loops."""

import pytest
from pypto import ir, passes, testing

from .sync_closed_command_oracle import check_closed_commands
from .sync_matmul_oracle import sequence_stmts
from .test_closed_prefix_sync import Services, ci
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("mode", ["concrete", "unknown_guard", "reuse"])
def test_retained_region_handoff_preserves_selected_payload_order(mode):
    original = vec_fixture(operation="buffer.abs")
    span = original.span
    *allocations, load, _, compute, _, store, terminal = sequence_stmts(original.body)
    source, consumer, output = ir.PipeType.MTE2, ir.PipeType.V, ir.PipeType.MTE3
    set_flag = ir.get_op("system.set_flag").name
    wait_flag = ir.get_op("system.wait_flag").name
    barrier = ir.get_op("system.bar_v").name
    services = Services()

    def region(name, payload):
        return ir.ForStmt(
            ir.Var(name, ir.ScalarType(ir.DataType.INDEX), span),
            ci(0),
            ci(1),
            ci(1),
            [],
            ir.SeqStmts([payload, ir.YieldStmt([], span)], span),
            [],
            span,
            attrs={"pypto.closed_sync_child": 1, "pypto.closed_sync_child_first_visit": True},
        )

    first, second = region("produce", load), region("consume", compute)
    statements = [
        *allocations,
        first,
        *services.closed(consumer, consumer),
        *services.closed(source, consumer),
        second,
        *(services.closed(source, consumer) if mode == "reuse" else []),
        *services.closed(consumer, output),
        store,
    ]
    parameters = list(zip(original.params, original.param_directions))
    if mode == "unknown_guard":
        coordinate = ir.Var("caller_coordinate", ir.ScalarType(ir.DataType.INDEX), span)
        parameters.append((coordinate, ir.ParamDirection.In))
        # A proof must not use x < x+1 as an identity without a bound on x.
        successor = ir.Add(coordinate, ci(1), ir.DataType.INDEX, span)
        guard = ir.Lt(coordinate, successor, ir.DataType.BOOL, span)
        statements.append(ir.IfStmt(guard, ir.SeqStmts([], span), None, [], span))
    logical = ir.Function(
        original.name,
        parameters,
        original.return_types,
        ir.SeqStmts([*statements, terminal], span),
        span,
        type=original.func_type,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=original.ir_stage,
    )
    program = ir.Program([logical], "RetainedRegionHandoff", span)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        baseline = testing.lower_sync_schedule(logical)["function"]
        physical = next(iter(passes.lower_sync()(program).functions.values()))
    assert physical.attrs["pypto_autosync"] is True

    def handoff(function):
        body = list(sequence_stmts(function.body))
        before, after = [i for i, stmt in enumerate(body) if isinstance(stmt, ir.ForStmt)]
        commands = []
        for stmt in body[before + 1 : after]:
            assert isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
            commands.append(stmt.expr)
        return commands

    baseline_handoff = [call.op.name for call in handoff(baseline)]
    optimized_handoff = handoff(physical)
    assert baseline_handoff == [barrier, set_flag, wait_flag, set_flag, wait_flag]
    optimized_names = [call.op.name for call in optimized_handoff]
    if mode == "unknown_guard":
        # The candidate keeps the whole handoff when the replay cannot resolve
        # later control for every invocation.
        assert optimized_names == baseline_handoff
        return

    if mode == "reuse":
        # Both publications use the same MTE2→V bank. The reverse relay
        # conveys consumption of the first wait to the second publication.
        forward_ids = [
            call.kwargs["event_id"]
            for stmt in sequence_stmts(physical.body)
            if isinstance(stmt, ir.EvalStmt)
            and isinstance(call := stmt.expr, ir.Call)
            and call.op.name == set_flag
            and call.kwargs["src_pipe"] == int(source)
            and call.kwargs["dst_pipe"] == int(consumer)
        ]
        assert len(forward_ids) == 2 and forward_ids[0] == forward_ids[1]
        assert optimized_names == baseline_handoff[1:]
    else:
        # The empty V barrier and reverse relay convey no completion to an
        # issued payload. MTE2→V is the only path from the load to the consume.
        assert optimized_names == [set_flag, wait_flag]
        assert all(
            call.kwargs["src_pipe"] == int(source) and call.kwargs["dst_pipe"] == int(consumer)
            for call in optimized_handoff
        )
    check_closed_commands(baseline, physical, {}, compare_commands=False)
