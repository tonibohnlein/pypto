# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Closed current-prefix protocols: actual commands, matching and causal reuse."""

from itertools import product

import pytest
from pypto import ir, passes, testing
from pypto.ir.op import sync

from .sync_closed_command_oracle import ClosedCommands, check_closed_commands, opname
from .test_autosync_switch import clone
from .test_conservative_sync_dataflow import conservative_fixture
from .test_logical_sync_realization import verify

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()
PIPES = (ir.PipeType.MTE2, ir.PipeType.V, ir.PipeType.MTE3)
SIX_IDS = [0, 2, 3, 5, 6, 7]


def ci(value):
    return ir.ConstInt(value, ir.DataType.INDEX, SPAN)


def seq(*statements):
    return ir.SeqStmts(list(statements), SPAN)


class Services:
    def __init__(self):
        self.number = 0

    def closed(self, source, destination):
        token = ir.Var(f"prefix{self.number}", ir.SyncTokenType(source), SPAN)
        self.number += 1
        result: list[ir.Stmt] = [
            ir.AssignStmt(token, sync.record(source), SPAN),
            ir.EvalStmt(sync.wait(token, destination), SPAN),
        ]
        if source != destination:
            acknowledgment = ir.Var(f"ack{self.number}", ir.SyncTokenType(destination), SPAN)
            self.number += 1
            result.extend(
                [
                    ir.AssignStmt(acknowledgment, sync.record(destination), SPAN),
                    ir.EvalStmt(sync.wait(acknowledgment, source), SPAN),
                ]
            )
        return result


def command_fixture(kind="for", orientations="both"):
    count, visits, choice = [
        ir.Var(n, ir.ScalarType(ir.DataType.INDEX), SPAN) for n in ("count", "visits", "choice")
    ]
    services = Services()
    commands = services.closed(PIPES[0], PIPES[1])
    if orientations == "both":
        commands += services.closed(PIPES[1], PIPES[0])
        commands += services.closed(PIPES[0], PIPES[2])
    iv = (
        ir.IterArg("i", ir.ScalarType(ir.DataType.INDEX), ci(0), SPAN)
        if kind == "while"
        else ir.Var("i", ir.ScalarType(ir.DataType.INDEX), SPAN)
    )
    active = ir.Eq(ir.FloorMod(iv, ci(2), ir.DataType.INDEX, SPAN), choice, ir.DataType.BOOL, SPAN)
    body = ir.IfStmt(active, seq(*commands), None, [], SPAN)
    if kind == "while":
        assert isinstance(iv, ir.IterArg)
        final = ir.Var("final", ir.ScalarType(ir.DataType.INDEX), SPAN)
        inner = ir.WhileStmt(
            ir.Lt(iv, count, ir.DataType.BOOL, SPAN),
            [iv],
            seq(body, ir.YieldStmt([ir.Add(iv, ci(1), ir.DataType.INDEX, SPAN)], SPAN)),
            [final],
            SPAN,
        )
    else:
        inner = ir.ForStmt(iv, ci(0), count, ci(1), [], seq(body), [], SPAN)
    outer = ir.ForStmt(
        ir.Var("visit", ir.ScalarType(ir.DataType.INDEX), SPAN), ci(0), visits, ci(1), [], inner, [], SPAN
    )
    return ir.Function(
        "empty_prefixes",
        [count, visits, choice],
        [],
        seq(outer, ir.ReturnStmt([], SPAN)),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def manual_payload_protocol(kind):
    original = conservative_fixture(kind)
    services = Services()

    class Prefixes(ir.IRMutator):
        def visit_eval_stmt(self, op):
            assert isinstance(op.expr, ir.Call)
            name = op.expr.op.name
            destination = {
                opname("buffer.load"): PIPES[0],
                opname("buffer.abs"): PIPES[1],
                opname("buffer.store"): PIPES[2],
            }[name]
            commands = [cut for pipe in PIPES for cut in services.closed(pipe, destination)]
            return seq(*commands, op)

    return Prefixes().visit_function(original)


def lower(function, eligible=SIX_IDS):
    before = ir.serialize(function)
    result = testing.lower_sync_schedule(function, eligible_ids=eligible)
    assert ir.serialize(function) == before, "lowering mutated original logical evidence"
    return result["function"]


@pytest.mark.parametrize("kind", ["for", "while"])
def test_empty_source_wait_forwarding_and_command_only_acknowledgments(kind):
    logical = command_fixture(kind)
    physical = lower(logical)
    assert "sync.record" not in str(physical)
    for count, visits, choice in product((0, 1, 3), (0, 1, 2), (0, 1)):
        before, after = check_closed_commands(logical, physical, {0: count, 1: visits, 2: choice}, SIX_IDS)
        assert not before.payloads and not after.payloads
        assert len(after.publications) == len(after.acquisitions)
        assert all(len(ids) <= 2 for ids in after.ids.values())


@pytest.mark.parametrize("kind", ["for", "while"])
def test_manual_closed_services_preserve_payloads_bytes_and_actual_token_order(kind):
    logical = manual_payload_protocol(kind)
    physical = lower(logical)
    for count, visits, choice, alias in product((0, 1, 3), (0, 1, 2), (0, 1), (False, True)):
        parameters = {3: count, 4: visits, 5: choice}
        bases = {2: 4096} if alias else None  # Output aliases the second input.
        before, after = check_closed_commands(
            logical, physical, parameters, SIX_IDS, tensor_bases=bases, compare_commands=False
        )
        assert before.payloads and after.barriers


@pytest.mark.parametrize("kind", ["for", "while"])
def test_actual_dataflow_producer_to_physical_commands(kind):
    original = conservative_fixture(kind)
    before = ir.serialize(original)
    logical = testing.insert_conservative_prefixes(original)
    assert ir.serialize(original) == before
    restored = ir.deserialize(ir.serialize(logical))
    assert isinstance(restored, ir.Function)
    physical = lower(restored)
    for count, visits, choice, alias in product((0, 1, 3), (0, 1, 2), (0, 1), (False, True)):
        parameters = {3: count, 4: visits, 5: choice}
        bases = {2: 4096} if alias else None
        original_trace = ClosedCommands(original, parameters, SIX_IDS, bases)
        original_trace.run()
        actual, _ = check_closed_commands(
            restored, physical, parameters, SIX_IDS, tensor_bases=bases, compare_commands=False
        )
        assert original_trace.payloads == actual.payloads
    # Also exercise the public pass on serialized logical IR, without invoking
    # a diagnostic analysis or reconstructing the memory-generated token graph.
    program = ir.Program([restored], "ClosedPrefixes", SPAN)
    with passes.PassContext([], enable_pypto_autosync=True):
        output = passes.lower_sync()(program)
    emitted = next(iter(output.functions.values()))
    _, completed = check_closed_commands(
        restored, emitted, {3: 2, 4: 2, 5: 1}, range(8), compare_commands=False
    )
    assert completed.last_all is not None and len(completed.returns) == 1
    assert all(
        (("payload", index, "C"), completed.returns[0]) in completed.order
        for index in range(len(completed.payloads))
    )


def test_pool_policy_refusal_is_transactional_and_not_a_minimum_claim():
    logical = command_fixture()
    before = ir.serialize(logical)
    with pytest.raises(ValueError, match="role reservation unavailable.*not.*lower bound"):
        lower(logical, [5])
    assert ir.serialize(logical) == before
    physical = lower(logical, [2, 5])
    check_closed_commands(logical, physical, {0: 3, 1: 2, 2: 0}, [2, 5])


@pytest.mark.parametrize("failure", ["parallel", "opaque_initializer", "inline_scope", "attributes"])
def test_unsupported_native_owners_and_controls_are_refused(failure):
    logical = manual_payload_protocol("for")

    class BreakQualification(ir.IRMutator):
        changed = False

        def visit_for_stmt(self, op):
            if self.changed or failure not in ("parallel", "opaque_initializer"):
                return super().visit_for_stmt(op)
            self.changed = True
            if failure == "parallel":
                return ir.ForStmt(
                    op.loop_var,
                    op.start,
                    op.stop,
                    op.step,
                    op.iter_args,
                    op.body,
                    op.return_vars,
                    op.span,
                    ir.ForKind.Parallel,
                )
            initial = ir.Call(ir.GlobalVar("opaque"), [], ir.ScalarType(ir.DataType.INDEX), SPAN)
            carry = ir.IterArg("carry", initial.type, initial, SPAN)
            final = ir.Var("final", initial.type, SPAN)
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                [carry],
                seq(op.body, ir.YieldStmt([carry], SPAN)),
                [final],
                SPAN,
            )

        def visit_call(self, op):
            if failure == "attributes" and not self.changed and op.op.name == opname("buffer.load"):
                self.changed = True
                return ir.Call(op.op, op.args, dict(op.kwargs), {"unqualified": True}, op.type, op.span)
            return op

    candidate = BreakQualification().visit_function(logical)
    if failure == "inline_scope":
        candidate = clone(candidate, body=ir.InCoreScopeStmt(body=candidate.body, span=SPAN))
    before = ir.serialize(candidate)
    with pytest.raises(ValueError):
        lower(candidate)
    assert ir.serialize(candidate) == before


def test_edited_token_partner_is_refused():
    cuts = Services().closed(PIPES[0], PIPES[1])
    acknowledgment = cuts[2]
    assert isinstance(acknowledgment, ir.AssignStmt)
    cuts[1] = ir.EvalStmt(sync.wait(acknowledgment.var, PIPES[1]), SPAN)
    logical = clone(command_fixture(), body=seq(*cuts))
    with pytest.raises(ValueError):
        lower(logical)


def test_finite_token_fanout_uses_safe_existing_lowering():
    # Fanout is legal logical SSA, but is outside the closed-transaction
    # recognizer. Its existing finite fallback must preserve the actual order,
    # rather than consume one physical flag twice or drop a payload obligation.
    original = conservative_fixture()
    assert isinstance(original.body, ir.SeqStmts)
    allocations = list(original.body.stmts[:3])
    loaded = original.body.stmts[3]
    first_allocation, last_allocation = allocations[0], allocations[2]
    assert isinstance(first_allocation, ir.AssignStmt) and isinstance(last_allocation, ir.AssignStmt)
    x, z = first_allocation.var, last_allocation.var
    computed = ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, z], ir.VoidType(), SPAN), SPAN)
    final_statement = original.body.stmts[5]
    assert isinstance(final_statement, ir.EvalStmt)
    final = final_statement.expr
    assert isinstance(final, ir.Call)
    stored = ir.EvalStmt(ir.Call(final.op, [z, *final.args[1:]], ir.VoidType(), SPAN), SPAN)
    services = Services()
    first = services.closed(PIPES[0], PIPES[1])
    publication = first[0]
    assert isinstance(publication, ir.AssignStmt)
    duplicate = ir.EvalStmt(sync.wait(publication.var, PIPES[1]), SPAN)
    logical = clone(
        original,
        body=seq(
            *allocations,
            loaded,
            *first,
            duplicate,
            computed,
            *services.closed(PIPES[1], PIPES[2]),
            stored,
            original.body.stmts[-1],
        ),
    )
    physical = lower(logical, [2, 3, 5])
    before, after = check_closed_commands(
        logical, physical, {3: 0, 4: 0, 5: 0}, [2, 3, 5], compare_commands=False
    )
    assert len(before.acquisitions) > len(before.publications)
    assert len(after.publications) == len(after.acquisitions)
    assert len(before.payloads) == len(after.payloads) == 3


def test_local_empty_prefix_needs_no_event_ids():
    services = Services()
    logical = clone(
        command_fixture(),
        body=seq(*services.closed(PIPES[0], PIPES[0]), *services.closed(PIPES[1], PIPES[1])),
    )
    physical = lower(logical, [])
    before, after = check_closed_commands(logical, physical, {0: 0, 1: 0, 2: 0}, [], compare_commands=False)
    assert len(before.publications) == 2 and after.barriers == 2
    assert not after.publications and not after.acquisitions and not after.ids


def test_oracle_rejects_wrong_flag_partner():
    logical = command_fixture()
    physical = lower(logical)

    class WrongId(ir.IRMutator):
        changed = False

        def visit_call(self, op):
            if not self.changed and op.op.name == opname("system.wait_flag"):
                self.changed = True
                kwargs = dict(op.kwargs)
                kwargs["event_id"] = 7
                return ir.Call(op.op, op.args, kwargs, None, op.type, op.span)
            return op

    with pytest.raises(AssertionError, match="matching publication"):
        check_closed_commands(logical, WrongId().visit_function(physical), {0: 2, 1: 1, 2: 0}, SIX_IDS)


def test_textual_flag_reuse_without_acknowledgment_is_not_causal_reuse():
    logical = command_fixture(orientations="forward")
    physical = lower(logical)

    class DropAcknowledgment(ir.IRMutator):
        removed = 0

        def visit_eval_stmt(self, op):
            call = op.expr
            if (
                isinstance(call, ir.Call)
                and call.op.name
                in (
                    opname("system.set_flag"),
                    opname("system.wait_flag"),
                )
                and ir.PipeType(call.kwargs["src_pipe"]) == PIPES[1]
                and ir.PipeType(call.kwargs["dst_pipe"]) == PIPES[0]
            ):
                self.removed += 1
                return seq()
            return op

    mutation = DropAcknowledgment()
    broken = mutation.visit_function(physical)
    assert mutation.removed == 2, "negative control must remove the actual return SET/WAIT pair"
    with pytest.raises(AssertionError, match="consumption does not precede republication"):
        ClosedCommands(broken, {0: 3, 1: 1, 2: 0}, SIX_IDS).run()


def test_all_completion_does_not_consume_outstanding_flags():
    # A terminal ALL establishes completion, not event-slot consumption.
    publication = ir.EvalStmt(
        ir.Call(
            ir.get_op("system.set_flag"),
            [],
            {
                "src_pipe": int(PIPES[0]),
                "dst_pipe": int(PIPES[1]),
                "event_id": 2,
                "core_type": "aiv",
            },
            ir.VoidType(),
            SPAN,
        ),
        SPAN,
    )
    barrier = ir.EvalStmt(
        ir.Call(ir.get_op("system.bar_all"), [], {"core_type": "aiv"}, ir.VoidType(), SPAN), SPAN
    )
    invalid = clone(command_fixture(), body=seq(publication, barrier, ir.ReturnStmt([], SPAN)))
    with pytest.raises(AssertionError, match="outstanding flags"):
        ClosedCommands(invalid, {0: 0, 1: 0, 2: 0}, SIX_IDS).run()


@pytest.mark.parametrize("payloads", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_completed_alias_chain_is_eliminated_before_ownership(payloads, optimized, roundtrip):
    # Empty facts and their unused aliases are valid unoptimized token IR.
    # Erasing only the original definition used to publish an owned function
    # containing a free token variable; verify the actual physical output too.
    original = manual_payload_protocol("for") if payloads else command_fixture()
    aliases = []
    token = ir.Var("empty_fact", ir.SyncTokenType(PIPES[0]), SPAN)
    aliases.append(ir.AssignStmt(token, sync.completed(PIPES[0]), SPAN))
    for index in range(3):
        alias = ir.Var(f"empty_alias_{index}", token.type, SPAN)
        aliases.append(ir.AssignStmt(alias, token, SPAN))
        token = alias
    logical = clone(original, body=seq(*aliases, original.body))
    if roundtrip:
        logical = ir.deserialize(ir.serialize(logical))
        assert isinstance(logical, ir.Function)
    verify(logical)
    program = ir.Program([logical], "CompletedAliases", SPAN)
    with passes.PassContext([], enable_pypto_autosync=True):
        if optimized:
            program = passes.optimize_sync()(program)
        output = passes.lower_sync()(program)
    physical = next(iter(output.functions.values()))
    verify(physical)
    assert "SyncToken" not in str(physical)
    assert "empty_alias" not in str(physical)
    parameters = {3: 3, 4: 2, 5: 1} if payloads else {0: 3, 1: 2, 2: 1}
    check_closed_commands(logical, physical, parameters, range(8), compare_commands=False)


@pytest.mark.parametrize("depth", [0, 28])
def test_closed_lowering_preserves_shared_signature_metadata(depth):
    """Signatures stay original even when rewriting a command-bearing body."""
    original = command_fixture()
    length = ir.Var("length", ir.ScalarType(ir.DataType.INDEX), SPAN)
    extent = length
    for _ in range(depth):
        extent = ir.Min(extent, extent, ir.DataType.INDEX, SPAN)
    tensor = ir.Var("metadata_only", ir.TensorType([extent, ci(8)], ir.DataType.FP32), SPAN)
    logical = ir.Function(
        original.name,
        [*original.params, length, tensor],
        original.return_types,
        original.body,
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    verify(logical)
    physical = lower(logical)
    verify(physical)
    assert physical.params[-1].same_as(tensor)
    result_type = physical.params[-1].type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.shape[0].same_as(extent)


def test_straight_line_and_branch_one_way_tokens_lower_to_two_commands():
    cond = ir.Var("cond", ir.ScalarType(ir.DataType.BOOL), SPAN)
    t0 = ir.Var("t0", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    t1 = ir.Var("t1", ir.SyncTokenType(ir.PipeType.V), SPAN)
    branch = ir.IfStmt(
        cond,
        seq(ir.AssignStmt(t1, sync.record(ir.PipeType.V), SPAN), ir.EvalStmt(sync.wait(t1, ir.PipeType.MTE3), SPAN)),
        None,
        [],
        SPAN,
    )
    logical = ir.Function(
        "one_way_straight",
        [cond],
        [],
        seq(
            ir.AssignStmt(t0, sync.record(ir.PipeType.MTE2), SPAN),
            ir.EvalStmt(sync.wait(t0, ir.PipeType.V), SPAN),
            branch,
            ir.ReturnStmt([], SPAN),
        ),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = testing.lower_sync_schedule(logical, eligible_ids=[0])
    assert result["set_sites"] == 2 and result["wait_sites"] == 2
    physical = result["function"]
    for flag in (False, True):
        before, after = check_closed_commands(logical, physical, {0: flag}, [0], compare_commands=True)
        expected_pairs = 2 if flag else 1
        assert len(after.publications) == expected_pairs
        assert len(after.acquisitions) == expected_pairs


def test_optimize_sync_eliminates_redundant_waits_joins_and_dead_bindings():
    c = ir.Var("empty_fact", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    c_alias = ir.Var("empty_alias", c.type, SPAN)
    r1 = ir.Var("rec1", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    r2 = ir.Var("rec2", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    j = ir.Var("joined", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    dead_r = ir.Var("dead_vec_rec", ir.SyncTokenType(ir.PipeType.V), SPAN)
    logical = ir.Function(
        "opt_tokens",
        [],
        [],
        seq(
            ir.AssignStmt(c, sync.completed(ir.PipeType.MTE2), SPAN),
            ir.AssignStmt(c_alias, c, SPAN),
            ir.EvalStmt(sync.wait(c_alias, ir.PipeType.V), SPAN),
            ir.AssignStmt(r1, sync.record(ir.PipeType.MTE2), SPAN),
            ir.AssignStmt(r2, sync.record(ir.PipeType.MTE2), SPAN),
            ir.AssignStmt(j, sync.join(c, r2), SPAN),
            # Adjacent same-pipe waits: later wait on `j` (`r2`) FIFO-subsumes wait on `r1`.
            ir.EvalStmt(sync.wait(r1, ir.PipeType.V), SPAN),
            ir.EvalStmt(sync.wait(j, ir.PipeType.V), SPAN),
            # Dominated duplicate wait on `r2`.
            ir.EvalStmt(sync.wait(r2, ir.PipeType.V), SPAN),
            ir.AssignStmt(dead_r, sync.record(ir.PipeType.V), SPAN),
            ir.ReturnStmt([], SPAN),
        ),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    verify(logical)
    program = ir.Program([logical], "OptimizeTokens", SPAN)
    with passes.PassContext([], enable_pypto_autosync=True):
        optimized = passes.optimize_sync()(program)
        lowered = passes.lower_sync()(optimized)
    opt_fn = next(iter(optimized.functions.values()))
    opt_text = str(opt_fn)
    assert "empty_alias" not in opt_text
    assert "sync.completed" not in opt_text
    assert "sync.join" not in opt_text
    assert "dead_vec_rec" not in opt_text
    assert opt_text.count("sync.wait") == 1
    phys_fn = next(iter(lowered.functions.values()))
    verify(phys_fn)
    _, after = check_closed_commands(opt_fn, phys_fn, {}, range(8), compare_commands=False)
    assert len(after.publications) == 1 and len(after.acquisitions) == 1


@pytest.mark.parametrize("closed_cycle", [True, False])
def test_loop_one_way_closed_cycle_stays_two_cmd_and_unclosed_falls_back(closed_cycle):
    count = ir.Var("count", ir.ScalarType(ir.DataType.INDEX), SPAN)
    iv = ir.Var("i", ir.ScalarType(ir.DataType.INDEX), SPAN)
    t_fwd = ir.Var("t_fwd", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    body_stmts: list[ir.Stmt] = [
        ir.AssignStmt(t_fwd, sync.record(ir.PipeType.MTE2), SPAN),
        ir.EvalStmt(sync.wait(t_fwd, ir.PipeType.V), SPAN),
    ]
    if closed_cycle:
        t_rev = ir.Var("t_rev", ir.SyncTokenType(ir.PipeType.V), SPAN)
        body_stmts.extend(
            [
                ir.AssignStmt(t_rev, sync.record(ir.PipeType.V), SPAN),
                ir.EvalStmt(sync.wait(t_rev, ir.PipeType.MTE2), SPAN),
            ]
        )
    body_stmts.append(ir.YieldStmt([], SPAN))
    loop = ir.ForStmt(iv, ci(0), count, ci(1), [], seq(*body_stmts), [], SPAN)
    logical = ir.Function(
        "loop_cycle_mode",
        [count],
        [],
        seq(loop, ir.ReturnStmt([], SPAN)),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = testing.lower_sync_schedule(logical, eligible_ids=[0, 1])
    # Both cases emit 2 set_flag + 2 wait_flag in body: closed_cycle via two 1-way edges,
    # unclosed via one 1-way edge plus its constant-role SetWaitAck fallback.
    assert result["set_sites"] == 2 and result["wait_sites"] == 2
    physical = result["function"]
    for c in (0, 1, 3):
        _, after = check_closed_commands(logical, physical, {0: c}, [0, 1], compare_commands=False)
        assert len(after.publications) == 2 * c and len(after.acquisitions) == 2 * c


def test_parent_child_per_direction_reservations_allow_disjoint_direction_id_reuse():
    # Parent overlaps two concurrent 1-way (MTE2 -> MTE3) edges, occupying BOTH IDs [0, 1]
    # on (MTE2, MTE3). Child operates on disjoint direction (MTE2 -> V) with entry acks
    # {(MTE2, V), (V, MTE2)} using ID 0 on (V, MTE2); with per-direction reservations,
    # eligible_ids=[0, 1] still suffices without false global ID subtraction.
    p0 = ir.Var("p0", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    p1 = ir.Var("p1", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    e0 = ir.Var("e0", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    e1 = ir.Var("e1", ir.SyncTokenType(ir.PipeType.V), SPAN)
    c_init = ir.Var("c_init", ir.SyncTokenType(ir.PipeType.V), SPAN)
    c_carry = ir.IterArg("c_carry", ir.SyncTokenType(ir.PipeType.V), c_init, SPAN)
    c_tok = ir.Var("c_tok", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    c_end = ir.Var("c_end", ir.SyncTokenType(ir.PipeType.V), SPAN)
    c_ret = ir.Var("c_ret", ir.SyncTokenType(ir.PipeType.V), SPAN)

    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)
    a = ir.Var("a", gm_t, SPAN)
    buf = ir.Var("buf", vec_t, SPAN)
    out_buf = ir.Var("out_buf", vec_t, SPAN)
    out_gm = ir.Var("out_gm", gm_t, SPAN)
    iv = ir.Var("k", ir.ScalarType(ir.DataType.INDEX), SPAN)

    child = ir.ForStmt(
        iv,
        ci(0),
        ci(2),
        ci(1),
        [c_carry],
        seq(
            ir.IfStmt(ir.Gt(iv, ci(0), ir.DataType.BOOL, SPAN), ir.EvalStmt(sync.wait(c_carry, ir.PipeType.MTE2), SPAN), None, [], SPAN),
            ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [a, ir.MakeTuple([ci(0), ci(0)], SPAN), ir.MakeTuple([ci(2), ci(8)], SPAN), buf], ir.VoidType(), SPAN), SPAN),
            ir.AssignStmt(c_tok, sync.record(ir.PipeType.MTE2), SPAN),
            ir.EvalStmt(sync.wait(c_tok, ir.PipeType.V), SPAN),
            ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [buf, out_buf], ir.VoidType(), SPAN), SPAN),
            ir.AssignStmt(c_end, sync.record(ir.PipeType.V), SPAN),
            ir.YieldStmt([c_end], SPAN),
        ),
        [c_ret],
        SPAN,
        attrs={"pypto.closed_sync_child": 1},
    )
    logical = ir.Function(
        "per_dir_child_res",
        [a, (out_gm, ir.ParamDirection.InOut)],
        [out_gm.type],
        seq(
            ir.AssignStmt(buf, ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], SPAN), ci(0)], vec_t, SPAN), SPAN),
            ir.AssignStmt(out_buf, ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], SPAN), ci(128)], vec_t, SPAN), SPAN),
            ir.AssignStmt(p0, sync.record(ir.PipeType.MTE2), SPAN),
            ir.AssignStmt(p1, sync.record(ir.PipeType.MTE2), SPAN),
            ir.EvalStmt(sync.wait(p0, ir.PipeType.MTE3), SPAN),
            ir.EvalStmt(sync.wait(p1, ir.PipeType.MTE3), SPAN),
            ir.AssignStmt(c_init, sync.completed(ir.PipeType.V), SPAN),
            ir.AssignStmt(e0, sync.record(ir.PipeType.MTE2), SPAN),
            ir.EvalStmt(sync.wait(e0, ir.PipeType.V), SPAN),
            ir.AssignStmt(e1, sync.record(ir.PipeType.V), SPAN),
            ir.EvalStmt(sync.wait(e1, ir.PipeType.MTE2), SPAN),
            child,
            ir.ReturnStmt([out_gm], SPAN),
        ),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = testing.lower_sync_schedule(logical, eligible_ids=[0, 1])
    physical = result["function"]
    verify(physical)
    _, after = check_closed_commands(logical, physical, {}, [0, 1], compare_commands=False)
    assert after.ids[ir.PipeType.MTE2, ir.PipeType.MTE3] == {0, 1}
    assert after.ids[ir.PipeType.MTE2, ir.PipeType.V].issubset({0, 1})

