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


def loop_fixture_parts(function: ir.Function):
    assert isinstance(function.body, ir.SeqStmts)
    x, z, y, load, reader, _, store, ret = function.body.stmts
    assert isinstance(x, ir.AssignStmt) and isinstance(z, ir.AssignStmt) and isinstance(y, ir.AssignStmt)
    assert isinstance(load, ir.EvalStmt)
    assert isinstance(store, ir.EvalStmt) and isinstance(store.expr, ir.Call)
    return x, z, y, load, reader, store.expr, ret


def producer_capture_loop(late_reader):
    """Two vector operations on separate slots, with the input reloaded each visit."""
    original = fixture()
    span = original.span
    index = ir.DataType.INDEX

    def ci(n):
        return ir.ConstInt(n, index, span)

    def payload(op, *args):
        return ir.EvalStmt(ir.Call(ir.get_op(op), list(args), ir.VoidType(), span), span)

    alloc_x, alloc_z, alloc_y, load, first_reader, store_z, ret = loop_fixture_parts(original)
    x, z, y = alloc_x.var, alloc_z.var, alloc_y.var
    definitions = [alloc_x, alloc_z, alloc_y]
    extra_operand = z
    if late_reader:
        # A distinct SSA handle placed over the input's bytes.
        extra_operand = ir.Var("input_alias", x.type, span)
        definitions.append(ir.AssignStmt(extra_operand, alloc_x.value, span))
    later = payload("buffer.add", z, extra_operand, y)
    store_y = payload("buffer.store", y, *store_z.args[1:])
    active = ir.Var("active", ir.ScalarType(ir.DataType.BOOL), span)
    visits = ir.Var("visits", ir.ScalarType(index), span)
    fill = payload("buffer.full", ir.ConstFloat(0.0, ir.DataType.FP32, span), z)
    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(index), span),
        ci(0),
        visits,
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.IfStmt(active, load, None, [], span),
                ir.IfStmt(active, first_reader, fill, [], span),
                later,
                store_y,
            ],
            span,
        ),
        [],
        span,
    )
    return ir.Function(
        "guarded_vector_work",
        [*zip(original.params, original.param_directions, strict=True), active, visits],
        original.return_types,
        ir.SeqStmts([*definitions, loop, ret], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("late_reader", [False, True])
def test_loop_capture_orders_each_reader_before_input_reuse(late_reader):
    # The later ADD always uses z. If it also reads x, its completion is
    # required. Otherwise that ADD and the next load into x can overlap.
    f = producer_capture_loop(late_reader)
    plans = programs(f, disjoint=True)
    first_reader = plans[0]["sites"][1]
    assert (first_reader["post_destination"] == ir.PipeType.MTE2) == (not late_reader)
    for active, visits in product((False, True), (0, 1, 4)):
        if late_reader and not active:
            continue  # The test's x input is defined by the active arm.
        execution = check_execution(plans, {3: active, 4: visits})[-1]
        if active:
            for i in range(visits - 1):
                next_load = ("payload", 4 * i + 4, "I")
                assert (("payload", 4 * i + 1, "C"), next_load) in execution.order
                assert ((("payload", 4 * i + 2, "C"), next_load) in execution.order) == late_reader


def test_producer_capture_preserves_a_possible_gm_relay():
    function = producer_capture_loop(late_reader=False)
    plans = programs(function, disjoint=False)
    # Alias the output and input tensors so the store must finish before the
    # next load, even though the earlier vector reader has released its slot.
    execution = check_execution(plans, {3: True, 4: 4}, tensor_bases={2: 0})[-1]
    for i in range(3):
        assert (("payload", 4 * i + 3, "C"), ("payload", 4 * i + 4, "I")) in execution.order


def test_store_release_allows_next_input_load_to_overlap_an_independent_store():
    original = fixture()
    span = original.span
    alloc_x, alloc_z, _, load, _, store, ret = loop_fixture_parts(original)
    x, z = alloc_x.var, alloc_z.var
    _, offset, valid, out = store.args
    assert isinstance(out, ir.Var)

    def payload(name, *args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), list(args), ir.VoidType(), span), span)

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(ir.DataType.INDEX), span),
        ci(0),
        ci(4),
        ci(1),
        [],
        ir.SeqStmts(
            [
                payload("buffer.full", ir.ConstFloat(7.0, ir.DataType.FP32, span), z),
                load,
                payload("buffer.store", x, offset, valid, out),
                payload("buffer.store", z, offset, valid, original.params[1]),
            ],
            span,
        ),
        [],
        span,
    )
    function = ir.Function(
        "two_output_stores",
        [original.params[0], (original.params[1], ir.ParamDirection.Out), (out, ir.ParamDirection.Out)],
        original.return_types,
        ir.SeqStmts([alloc_x, alloc_z, loop, ret], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    plans = programs(function, disjoint=True)
    first_store = plans[0]["sites"][2]
    assert first_store["pipe"] == ir.PipeType.MTE3 and first_store["post_destination"] == ir.PipeType.MTE2
    execution = check_execution(plans, {})[-1]
    for i in range(3):
        next_load = ("payload", 4 * i + 5, "I")
        assert (("payload", 4 * i + 2, "C"), next_load) in execution.order
        assert (("payload", 4 * i + 3, "C"), next_load) not in execution.order


def loop_entry_reader_fixture(revisit_reader=False, *, skip_first=False, preceding_store=False):
    source = fixture()
    alloc_x, alloc_z, alloc_y, load, read_x, store, ret = loop_fixture_parts(source)
    assert isinstance(load.expr, ir.Call)
    span = source.span
    idx = ir.ScalarType(ir.DataType.INDEX)
    iv, stop = (ir.Var(name, idx, span) for name in ("iv", "stop"))
    zero = ir.ConstInt(0, ir.DataType.INDEX, span)
    one = ir.ConstInt(1, ir.DataType.INDEX, span)

    def payload(name, *args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), list(args), ir.VoidType(), span), span)

    # A vector operation reads x before entry. Every visit overwrites x on
    # MTE2 and drains it to GM on MTE3; a later visit can reuse x only after
    # that store completes. The condition can reintroduce the vector reader.
    store_x = payload("buffer.store", alloc_x.var, *store.args[1:])
    statements: list[ir.Stmt] = [payload("buffer.load", *load.expr.args), store_x]
    if revisit_reader:
        alias = ir.Var("same_slot", alloc_x.var.type, span)
        definitions = [alloc_x, alloc_z, ir.AssignStmt(alias, alloc_x.value, span)]
        statements.append(
            ir.IfStmt(
                ir.Eq(iv, one, ir.DataType.BOOL, span),
                payload("buffer.abs", alias, alloc_z.var),
                None,
                [],
                span,
            )
        )
    else:
        definitions = [alloc_x, alloc_z]
    body: ir.Stmt = ir.SeqStmts(statements, span)
    if skip_first:
        body = ir.IfStmt(ir.Gt(iv, zero, ir.DataType.BOOL, span), body, None, [], span)
    loop = ir.ForStmt(iv, zero, stop, one, [], body, [], span)
    prologue: list[ir.Stmt] = list(definitions)
    if preceding_store:
        _, offsets, valid, _ = store.args
        prologue.extend(
            [
                alloc_y,
                payload("buffer.load", source.params[0], offsets, valid, alloc_y.var),
                payload("buffer.store", alloc_y.var, offsets, valid, source.params[1]),
            ]
        )
    prologue.extend([load, read_x])
    return ir.Function(
        "loop_entry_reader",
        [
            source.params[0],
            (source.params[1], ir.ParamDirection.InOut),
            (source.params[2], ir.ParamDirection.InOut),
            stop,
        ],
        source.return_types,
        ir.SeqStmts([*prologue, loop, ret], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("revisit_reader", [False, True])
def test_loop_entry_reader_and_backedge_have_distinct_release_obligations(revisit_reader):
    plans = programs(loop_entry_reader_fixture(revisit_reader), disjoint=True)
    for visits in (0, 1, 4):
        actual = check_execution(plans, {3: visits})[-1]
        if revisit_reader and visits > 2:
            # At i=1 V also reads the bytes through a different handle. The
            # i=2 load must respect that read as well as the previous store.
            assert (("payload", 6, "C"), ("payload", 7, "I")) in actual.order
        elif not revisit_reader:
            assert actual.trace.count(("acquire", ir.PipeType.V, ir.PipeType.MTE2)) == int(visits > 0)
            assert len(actual.publications) == 1 + int(visits > 0) + 2 * visits
            if visits:
                assert (("payload", 1, "C"), ("payload", 2, "I")) in actual.order
            for i in range(1, visits):
                assert (("payload", 2 * i + 1, "C"), ("payload", 2 * i + 2, "I")) in actual.order


def test_first_executed_use_can_be_later_than_the_first_loop_iteration():
    plans = programs(loop_entry_reader_fixture(skip_first=True), disjoint=True)
    for visits in (0, 1, 2, 4):
        actual = check_execution(plans, {3: visits})[-1]
        assert len(actual.payloads) == 2 + 2 * max(visits - 1, 0)
        if visits <= 1:
            assert ("acquire", ir.PipeType.V, ir.PipeType.MTE2) not in actual.trace
        else:
            assert (("payload", 1, "C"), ("payload", 2, "I")) in actual.order


def test_steady_wait_does_not_capture_independent_pre_loop_store():
    plans = programs(loop_entry_reader_fixture(preceding_store=True), disjoint=True)
    actual = check_execution(plans, {3: 4})[-1]
    # The pre-loop store reads y and writes a separate GM allocation. It can
    # overlap the first load into x. The stores of x inside the loop cannot
    # overlap the next overwrite of x.
    assert (("payload", 1, "C"), ("payload", 4, "I")) not in actual.order
    assert (("payload", 5, "C"), ("payload", 6, "I")) in actual.order


def nested_loop_entry_fixture():
    source = fixture()
    alloc_x, alloc_z, _, load, read_x, store, ret = loop_fixture_parts(source)
    span = source.span
    idx = ir.ScalarType(ir.DataType.INDEX)
    row, col, tail, rows, visits = (
        ir.Var(name, idx, span) for name in ("row", "col", "tail", "rows", "visits")
    )
    zero, one, two, three = (ir.ConstInt(n, ir.DataType.INDEX, span) for n in range(4))

    # Only the second loop issues stores, but an earlier row's store can
    # still be reading x when the first loop starts overwriting it.
    reduce = ir.ForStmt(col, zero, visits, one, [], ir.SeqStmts([load, read_x], span), [], span)
    tail_load = ir.EvalStmt(load.expr, span)
    tail_store = ir.EvalStmt(
        ir.Call(ir.get_op("buffer.store"), [alloc_x.var, *store.args[1:]], ir.VoidType(), span),
        span,
    )
    write = ir.ForStmt(tail, zero, two, one, [], ir.SeqStmts([tail_load, tail_store], span), [], span)
    body = ir.IfStmt(
        ir.Lt(row, rows, ir.DataType.BOOL, span), ir.SeqStmts([reduce, write], span), None, [], span
    )
    outer = ir.ForStmt(row, zero, three, one, [], body, [], span)
    final_load = ir.EvalStmt(load.expr, span)
    return ir.Function(
        "nested_loop_entry",
        [source.params[0], source.params[1], (source.params[2], ir.ParamDirection.InOut), rows, visits],
        source.return_types,
        ir.SeqStmts([alloc_x, alloc_z, outer, final_load, ret], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("disjoint", [False, True])
def test_nested_loop_delivers_idle_store_completion_at_entry(disjoint):
    plans = programs(nested_loop_entry_fixture(), disjoint)

    flag_ops = {ir.get_op("system.set_flag").name, ir.get_op("system.wait_flag").name}

    class FlagDirections(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.flags = []
            self.conditionals = 0

        def visit_if_stmt(self, op):
            self.conditionals += 1
            super().visit_if_stmt(op)

        def visit_call(self, op):
            if op.op.name in flag_ops:
                self.flags.append((ir.PipeType(op.kwargs["src_pipe"]), ir.PipeType(op.kwargs["dst_pipe"])))

    def find_reduction(stmt):
        if isinstance(stmt, ir.ForStmt):
            if stmt.loop_var.name_hint == "col":
                yield stmt
            yield from find_reduction(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            yield from find_reduction(stmt.then_body)
            if stmt.else_body is not None:
                yield from find_reduction(stmt.else_body)
        elif isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                yield from find_reduction(child)

    (reduction,) = find_reduction(plans[-1].body)
    commands = FlagDirections()
    commands.visit_stmt(reduction.body)
    # MTE3 delivers its prefix at entry and issues no payload here. The
    # physical loop needs neither the forward flag nor its return ACK.
    assert all(ir.PipeType.MTE3 not in direction for direction in commands.flags)
    # The two V/MTE2 services continue to establish each other's event reuse.
    # Guarding the first V wait would require a second V/MTE2 pair as an ACK.
    assert commands.conditionals == 0
    assert commands.flags.count((ir.PipeType.V, ir.PipeType.MTE2)) == 2
    assert commands.flags.count((ir.PipeType.MTE2, ir.PipeType.V)) == 2
    tensor_bases = None if disjoint else {2: 0}
    for rows, visits in product((0, 1, 3), (0, 1, 5)):
        execution = check_execution(plans, {3: rows, 4: visits}, tensor_bases=tensor_bases)[-1]
        per_row = 2 * visits + 4
        assert len(execution.payloads) == rows * per_row + 1
        # Consecutive row blocks and the final load must respect the final
        # store even when an inner loop or an outer arm executes zero trips.
        for after_store in range(per_row, len(execution.payloads), per_row):
            assert (("payload", after_store - 1, "C"), ("payload", after_store, "I")) in execution.order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
