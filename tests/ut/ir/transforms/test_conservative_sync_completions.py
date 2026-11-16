# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Construct outstanding storage services without changing actual-token lowering."""

from itertools import product

import pytest
from pypto import ir, passes, testing

from .sync_closed_command_oracle import ClosedCommands
from .test_autosync_switch import clone
from .test_closed_prefix_sync import Services, lower
from .test_conservative_sync_dataflow import conservative_fixture
from .test_conservative_sync_storage import fixture
from .test_logical_sync_realization import verify
from .test_sync_scalar_payload import scalar_payload_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def baseline_program(function, analysis):
    """Build the unchanged whole-service reference at the actual input cuts."""
    services = Services()
    indexed = {site["cut"]: site for site in analysis["sites"]}

    class Baseline(ir.IRMutator):
        def wrap(self, statement):
            site = indexed.get(statement)
            if site is None:
                return statement
            commands = [
                command
                for source in site["required_source_pipes"]
                for command in services.closed(source, site["pipe"])
            ]
            return ir.SeqStmts([*commands, statement], statement.span) if commands else statement

        def visit_eval_stmt(self, op):
            return self.wrap(op)

        def visit_assign_stmt(self, op):
            return self.wrap(op)

    return Baseline().visit_function(function)


def programs(function, disjoint=False):
    before = ir.serialize(function)
    with passes.PassContext([], autosync_disjoint_gm=disjoint):
        analysis = testing.analyze_conservative_sync(function, retain_incoming_diagnostics=False)
        baseline = baseline_program(function, analysis)
        candidate = testing.insert_conservative_prefixes(function)
        verify(baseline)
        verify(candidate)
        baseline_physical = lower(baseline, eligible=list(range(6)))
        candidate_physical = lower(candidate, eligible=list(range(6)))
    assert ir.serialize(function) == before
    cost = analysis["cost"]
    assert cost["baseline_services"] == cost["services"] + cost["suppressed_services"]
    assert cost["certificate_omissions"] == cost["suppressed_services"]
    assert cost["certificate_visits"] > 0
    for site in analysis["sites"]:
        assert set(site["source_pipes"]) <= set(site["required_source_pipes"])
    return analysis, baseline, candidate, baseline_physical, candidate_physical


def check_execution(programs_, parameters, *, tensor_bases=None, memory=None):
    _, baseline, candidate, baseline_physical, candidate_physical = programs_
    executions = [
        ClosedCommands(f, parameters, range(6), tensor_bases, scalar_memory=memory)
        for f in (baseline, candidate, baseline_physical, candidate_physical)
    ]
    orders = [execution.run() for execution in executions]
    assert all(execution.payloads == executions[0].payloads for execution in executions)
    assert all(execution.scalar_memory == executions[0].scalar_memory for execution in executions)
    assert all(execution.descriptors == executions[0].descriptors for execution in executions)
    assert all(execution.while_tests == executions[0].while_tests for execution in executions)
    pairs = [
        {(a, b) for a, b in order if a[0] == b[0] == "payload" and a[2] == "C" and b[2] == "I"}
        for order in orders
    ]
    # Lowering preserves each actual supplied token program. Construction can
    # remove order; it may never introduce a prerequisite absent from baseline.
    assert pairs[0] == pairs[2] and pairs[1] == pairs[3]
    assert pairs[3] <= pairs[2]
    actual = executions[3]
    assert not actual.live and all(pair in orders[3] for pair in actual.reuses)
    assert all(ids <= set(range(6)) for ids in actual.ids.values())
    for index, (_, _, reads, writes) in enumerate(actual.payloads):
        for previous in range(index):
            _, _, earlier_reads, earlier_writes = actual.payloads[previous]
            if earlier_writes & (reads | writes) or earlier_reads & writes:
                assert (("payload", previous, "C"), ("payload", index, "I")) in pairs[3]
    return executions


@pytest.mark.parametrize("kind", ["for", "while"])
@pytest.mark.parametrize("disjoint", [False, True])
def test_skipped_writers_empty_loops_repeated_visits_and_resets(kind, disjoint):
    result = programs(conservative_fixture(kind), disjoint)
    for count, visits, choice in product((0, 1, 3), (0, 1, 2), (0, 1)):
        check_execution(result, {3: count, 4: visits, 5: choice})
        if not disjoint:
            check_execution(result, {3: count, 4: visits, 5: choice}, tensor_bases={2: 4096})


def repeated_scalar_writes():
    original = scalar_payload_fixture()
    assert isinstance(original.body, ir.SeqStmts)
    definitions, load = original.body.stmts[:2]
    assert isinstance(definitions, ir.AssignStmt)
    local = definitions.var
    span = original.span
    index = ir.DataType.INDEX
    point = ir.MakeTuple([ir.ConstInt(0, index, span)] * 2, span)
    writes = [
        ir.EvalStmt(
            ir.Call(
                ir.get_op("buffer.write"),
                [local, point, ir.ConstFloat(float(number), ir.DataType.FP32, span)],
                ir.VoidType(),
                span,
            ),
            span,
        )
        for number in (1, 2)
    ]
    final = original.body.stmts[-2:]
    return clone(original, body=ir.SeqStmts([definitions, load, *writes, *final], span))


def test_retained_acquire_covers_later_scalar_consumer_without_republishing():
    result = programs(repeated_scalar_writes(), True)
    analysis = result[0]
    writes = [
        site for site in analysis["sites"] if site["operation"].op.name == ir.get_op("buffer.write").name
    ]
    assert len(writes) == 2
    assert ir.PipeType.MTE2 in writes[1]["required_source_pipes"]
    assert ir.PipeType.MTE2 not in writes[1]["source_pipes"]
    assert analysis["cost"]["suppressed_services"] > 0
    # Both sources are baseline prerequisites at the final MTE3 cut. The
    # later S descriptor already advertises current MTE2 completion, so the
    # builder must not first emit a redundant earlier MTE2 descriptor.
    final_store = analysis["sites"][-1]
    assert final_store["required_source_pipes"] == [ir.PipeType.MTE2, ir.PipeType.S]
    assert final_store["source_pipes"] == [ir.PipeType.S]
    check_execution(result, {2: 0, 3: 0}, memory={("gm", i): 7.0 for i in range(64)})


def destination_still_outstanding():
    original = fixture()
    assert isinstance(original.body, ir.SeqStmts)
    statements = list(original.body.stmts)
    load = statements[-3]
    previous_compute = statements[-4]
    assert isinstance(load, ir.EvalStmt) and isinstance(load.expr, ir.Call)
    assert isinstance(previous_compute, ir.EvalStmt) and isinstance(previous_compute.expr, ir.Call)
    new_compute = ir.EvalStmt(
        ir.Call(
            ir.get_op("buffer.abs"),
            [load.expr.args[3], previous_compute.expr.args[1]],
            ir.VoidType(),
            original.span,
        ),
        original.span,
    )
    return clone(original, body=ir.SeqStmts([*statements[:-2], new_compute, *statements[-2:]], original.span))


def test_ack_publication_does_not_complete_its_own_destination_payloads():
    result = programs(destination_still_outstanding(), True)
    # p=MTE2 is serviced first. Its forward acquire/return publication cannot
    # replace the subsequent V barrier protecting the earlier V write.
    site = result[0]["sites"][3]
    assert site["required_source_pipes"] == [ir.PipeType.MTE2, ir.PipeType.V]
    assert ir.PipeType.V in site["source_pipes"]
    check_execution(result, {})


@pytest.mark.parametrize("initial", [-1.0, 1.0])
def test_scalar_produced_control_is_protected_with_both_branch_outcomes(initial):
    result = programs(scalar_payload_fixture(), True)
    for count, visits in product((0, 1, 3), (0, 1, 2)):
        check_execution(result, {2: count, 3: visits}, memory={("gm", i): initial for i in range(64)})


@pytest.mark.parametrize("optimize", [False, True])
def test_public_lowering_preserves_three_pipe_wait_record_forwarding(optimize):
    original = fixture()
    assert isinstance(original.body, ir.SeqStmts)
    statements = list(original.body.stmts)
    load, compute, store = statements[3], statements[4], statements[-2]
    assert isinstance(load, ir.EvalStmt) and isinstance(compute, ir.EvalStmt)
    assert isinstance(store, ir.EvalStmt)
    services = Services()
    p, r, q = ir.PipeType.MTE2, ir.PipeType.V, ir.PipeType.MTE3
    prefix = [
        *statements[:3],
        load,
        *services.closed(p, r),
        compute,
        *services.closed(p, q),  # Old p prefix does not contain r's compute.
        *services.closed(r, p),  # p imports r with no new p payload.
    ]
    old = clone(original, body=ir.SeqStmts([*prefix, store, statements[-1]], original.span))
    logical = clone(
        original,
        body=ir.SeqStmts(
            [*prefix, *services.closed(p, q), store, statements[-1]],
            original.span,
        ),
    )
    verify(logical)
    program = ir.Program([logical], "ForwardedCompletion", original.span)
    if optimize:
        program = passes.optimize_sync()(program)
    physical_program = passes.lower_sync()(program)
    physical = physical_program.get_function(original.name)
    assert physical is not None
    old_prefix = ClosedCommands(old, {}, range(6))
    expected = ClosedCommands(logical, {}, range(6))
    actual = ClosedCommands(physical, {}, range(6))
    stale, before, after = old_prefix.run(), expected.run(), actual.run()
    required = (("payload", 1, "C"), ("payload", 2, "I"))
    assert required not in stale  # Immutable prefix facts differ despite same p payload ordinal.
    assert required in before and required in after
    assert len(expected.payloads) == len(actual.payloads) == 3
    assert expected.payloads == actual.payloads
    assert expected.trace == actual.trace
    assert not actual.live and all(pair in after for pair in actual.reuses)


def branch_completion_fixture(kind):
    """Distinct original payload nodes make equal per-arm issue counts adversarial."""
    original = scalar_payload_fixture()
    assert isinstance(original.body, ir.SeqStmts)
    span = original.span
    definition, load = original.body.stmts[:2]
    assert isinstance(definition, ir.AssignStmt)
    assert isinstance(load, ir.EvalStmt) and isinstance(load.expr, ir.Call)
    load_arguments = list(load.expr.args)
    local = definition.var
    index = ir.DataType.INDEX
    point = ir.MakeTuple([ir.ConstInt(0, index, span)] * 2, span)

    def payload(name, args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), args, ir.VoidType(), span), span)

    def write():
        return payload("buffer.write", [local, point, ir.ConstFloat(1.0, ir.DataType.FP32, span)])

    def read(name):
        value = ir.Var(name, ir.ScalarType(ir.DataType.FP32), span)
        return ir.AssignStmt(value, ir.Call(ir.get_op("buffer.read"), [local, point], value.type, span), span)

    def reload():
        return payload("buffer.load", load_arguments)

    def sequence(*statements):
        return ir.SeqStmts(list(statements), span)

    condition = ir.Gt(original.params[2], ir.ConstInt(0, index, span), ir.DataType.BOOL, span)
    harmless = ir.AssignStmt(
        ir.Var("unrelated", ir.ScalarType(index), span), ir.ConstInt(7, index, span), span
    )
    prefix = [definition, load, write()]
    final = write()
    alternate = None
    if kind == "unrelated":
        arm = harmless
    elif kind == "nested":
        arm = ir.IfStmt(condition, harmless, None, [], span)
    elif kind == "new_write":
        arm = reload()
    elif kind == "new_read":
        # The reload acknowledges the first S read. A later pure S read must
        # invalidate that completion before another MTE2 overwrite (WAR).
        prefix = [definition, load, read("first_read"), reload()]
        arm, final = read("branch_read"), reload()
    elif kind == "both_complete":
        arm = sequence(reload(), write())
        alternate = sequence(reload(), write())
    elif kind == "equal_count_one_complete":
        arm = sequence(reload(), write())
        alternate = reload()
    else:
        raise AssertionError(kind)
    branch = ir.IfStmt(condition, arm, alternate, [], span)
    body = sequence(*prefix, branch, final, *original.body.stmts[-2:])
    return clone(original, body=body), final


@pytest.mark.parametrize(
    "kind,source,required",
    [
        ("unrelated", ir.PipeType.MTE2, False),
        ("nested", ir.PipeType.MTE2, False),
        ("new_write", ir.PipeType.MTE2, True),
        ("new_read", ir.PipeType.S, True),
        ("both_complete", ir.PipeType.MTE2, False),
        ("equal_count_one_complete", ir.PipeType.MTE2, True),
    ],
)
def test_branch_must_completion_uses_current_prefix_on_every_path(kind, source, required):
    original, final = branch_completion_fixture(kind)
    result = programs(original, True)
    site = next(site for site in result[0]["sites"] if site["cut"] == final)
    assert source in site["required_source_pipes"]
    assert (source in site["source_pipes"]) == required
    for choice in (0, 1):
        # The independent interpreter replays the actual logical AND physical
        # packets, checks every original byte conflict, and matches/reuses IDs.
        check_execution(result, {2: choice, 3: 0}, memory={("gm", i): 7.0 for i in range(64)})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
