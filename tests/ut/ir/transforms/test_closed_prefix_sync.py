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
import time

import pytest
from pypto import ir, passes, testing
from pypto.backend import pto_backend
from pypto.ir.op import sync

from .sync_closed_command_oracle import ClosedCommands, check_closed_commands, opname
from .test_autosync_switch import clone
from .test_conservative_sync_dataflow import conservative_fixture
from .test_logical_sync_realization import verify
from .test_token_sync_lowering import _check_actual_tokens

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



def test_zero_trip_loop_prevents_post_loop_id_reuse():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    n = ir.Var("n", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)
    ci = lambda v: ir.ConstInt(v, idx, span)

    t0 = ir.Var("t0", ir.SyncTokenType(ir.PipeType.MTE2), span)
    t_loop = ir.Var("t_loop", ir.SyncTokenType(ir.PipeType.V), span)
    t1 = ir.Var("t1", ir.SyncTokenType(ir.PipeType.MTE2), span)

    # Reverse edge V -> MTE2 lives ONLY inside a symbolic-bound ForStmt that may run 0 trips.
    # Post-loop MTE2 -> V must NOT reuse t0's event_id=0 assuming V -> MTE2 ran inside the loop.
    loop = ir.ForStmt(
        k,
        ci(0),
        n,
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(t_loop, sync.record(ir.PipeType.V), span),
                ir.EvalStmt(sync.wait(t_loop, ir.PipeType.MTE2), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    fn = ir.Function(
        "zero_trip_no_reuse",
        [n],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(t0, sync.record(ir.PipeType.MTE2), span),
                ir.EvalStmt(sync.wait(t0, ir.PipeType.V), span),
                loop,
                ir.AssignStmt(t1, sync.record(ir.PipeType.MTE2), span),
                ir.EvalStmt(sync.wait(t1, ir.PipeType.V), span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([fn], "ZeroTripNoReuse", span)
    with passes.PassContext([]):
        physical = passes.lower_sync()(program)
    phys_fn = next(iter(physical.functions.values()))
    text = str(phys_fn)
    assert "event_id=0" in text
    assert "event_id=1" in text
    for trips in (0, 2):
        check_closed_commands(fn, phys_fn, {0: trips}, range(6), compare_commands=False)


def test_unbalanced_bare_unwaited_and_carried_syntax_routed_safely():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    cond = ir.Var("cond", ir.ScalarType(ir.DataType.BOOL), span)
    k = ir.Var("k", ir.ScalarType(idx), span)
    ci = lambda v: ir.ConstInt(v, idx, span)

    # 1a. Bare EvalStmt(sync.wait) directly inside IfStmt (no SeqStmts wrapper) must be flagged unsupported.
    t0 = ir.Var("t0", ir.SyncTokenType(ir.PipeType.MTE2), span)
    t_other = ir.Var("t_other", ir.SyncTokenType(ir.PipeType.MTE2), span)
    bare_if = ir.Function(
        "bare_arm_wait",
        [cond],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(t0, sync.record(ir.PipeType.MTE2), span),
                ir.IfStmt(
                    cond,
                    ir.EvalStmt(sync.wait(t0, ir.PipeType.V), span),
                    None,
                    [],
                    span,
                ),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="LowerSync structured token transport"):
        with passes.PassContext([]):
            passes.lower_sync()(ir.Program([bare_if], "BareArmWait", span))

    # 1b. Cross-control record/wait (record outside IfStmt, wait inside SeqStmts arm) routes to fallback cleanly.
    t_cross = ir.Var("t_cross", ir.SyncTokenType(ir.PipeType.MTE2), span)
    cross_if = ir.Function(
        "cross_arm_wait",
        [cond],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(t_cross, sync.record(ir.PipeType.MTE2), span),
                ir.IfStmt(
                    cond,
                    ir.SeqStmts([ir.EvalStmt(sync.wait(t_cross, ir.PipeType.V), span)], span),
                    None,
                    [],
                    span,
                ),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="LowerSync structured token transport"):
        with passes.PassContext([]):
            passes.lower_sync()(ir.Program([cross_if], "CrossArmWait", span))

    # 2. Un-waited sync.record must refuse closed-prefix syntax and still lower cleanly via fallback.
    t_used = ir.Var("t_used", ir.SyncTokenType(ir.PipeType.MTE2), span)
    unwaited = ir.Function(
        "unwaited_rec",
        [],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(t_used, sync.record(ir.PipeType.MTE2), span),
                ir.EvalStmt(sync.wait(t_used, ir.PipeType.V), span),
                ir.AssignStmt(t_other, sync.record(ir.PipeType.MTE2), span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        phys_unwaited = passes.lower_sync()(ir.Program([unwaited], "UnwaitedRec", span))
    assert "sync.record" not in str(next(iter(phys_unwaited.functions.values())))
    assert "sync.wait" not in str(next(iter(phys_unwaited.functions.values())))

    # 3. ForStmt carrying a SyncTokenType iter_arg routes away from closed prefix to structured lowering.
    idx_vec = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)
    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    x = ir.Var("x", idx_vec, span)
    y = ir.Var("y", idx_vec, span)
    tup = lambda vals: ir.MakeTuple([ci(v) for v in vals], span)
    t_init = ir.Var("t_init", ir.SyncTokenType(ir.PipeType.MTE2), span)
    t_iter = ir.IterArg("t_iter", ir.SyncTokenType(ir.PipeType.MTE2), t_init, span)
    t_out = ir.Var("t_out", ir.SyncTokenType(ir.PipeType.MTE2), span)
    t_next = ir.Var("t_next", ir.SyncTokenType(ir.PipeType.MTE2), span)
    carried = ir.Function(
        "carried_loop_token",
        [src, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], idx_vec, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], idx_vec, span), span),
                ir.AssignStmt(t_init, sync.completed(ir.PipeType.MTE2), span),
                ir.ForStmt(
                    k,
                    ci(0),
                    ci(2),
                    ci(1),
                    [t_iter],
                    ir.SeqStmts(
                        [
                            ir.EvalStmt(sync.wait(t_iter, ir.PipeType.V), span),
                            ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span),
                            ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                            ir.AssignStmt(t_next, sync.record(ir.PipeType.MTE2), span),
                            ir.YieldStmt([t_next], span),
                        ],
                        span,
                    ),
                    [t_out],
                    span,
                ),
                ir.EvalStmt(sync.wait(t_out, ir.PipeType.V), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [y, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="Unsupported token activation"):
        with passes.PassContext([]):
            passes.lower_sync()(ir.Program([carried], "CarriedLoopToken", span))

    # 4. TupleType([SyncTokenType]) containers (e.g. MakeTuple([completed])) must also be rejected
    # consistently by both HasOnlyClosedPrefixSyntax and TryLowerClosedPrefixSync.
    c_tok = ir.Var("c_tok", ir.SyncTokenType(ir.PipeType.MTE2), span)
    t_pair = ir.Var("t_pair", ir.SyncTokenType(ir.PipeType.MTE2), span)
    tup_var = ir.Var("tup_tok", ir.TupleType([ir.SyncTokenType(ir.PipeType.MTE2)]), span)
    tup_fn = ir.Function(
        "tuple_token_container",
        [],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(t_pair, sync.record(ir.PipeType.MTE2), span),
                ir.EvalStmt(sync.wait(t_pair, ir.PipeType.V), span),
                ir.AssignStmt(c_tok, sync.completed(ir.PipeType.MTE2), span),
                ir.AssignStmt(tup_var, ir.MakeTuple([c_tok], span), span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError):
        with passes.PassContext([]):
            passes.lower_sync()(ir.Program([tup_fn], "TupleTokenContainer", span))


def test_intra_loop_set_wait_ack_reuses_ids_without_exhaustion():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    n = ir.Var("n", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)
    ci = lambda v: ir.ConstInt(v, idx, span)

    # 10 open MTE2 -> V edges inside one loop body exceed the 8-ID hardware pool without intra-loop reuse.
    stmts = []
    for i in range(10):
        tok = ir.Var(f"tok_{i}", ir.SyncTokenType(ir.PipeType.MTE2), span)
        stmts.append(ir.AssignStmt(tok, sync.record(ir.PipeType.MTE2), span))
        stmts.append(ir.EvalStmt(sync.wait(tok, ir.PipeType.V), span))
    stmts.append(ir.YieldStmt([], span))

    fn = ir.Function(
        "intra_loop_reuse",
        [n],
        [],
        ir.SeqStmts(
            [
                ir.ForStmt(k, ci(0), n, ci(1), [], ir.SeqStmts(stmts, span), [], span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        physical = passes.lower_sync()(ir.Program([fn], "IntraLoopReuse", span))
    phys_fn = next(iter(physical.functions.values()))
    # All 10 SetWaitAck sites inside the loop should reuse event_id=0 without exhausting the pool.
    text = str(phys_fn)
    assert "event_id=0" in text
    assert "event_id=9" not in text
    check_closed_commands(fn, phys_fn, {0: 3}, range(8), compare_commands=False)


def test_loop_if_else_cross_arm_payloads_and_open_edges_never_collide():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    ci = lambda v: ir.ConstInt(v, idx, span)
    tup = lambda vals: ir.MakeTuple([ci(v) for v in vals], span)
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)

    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    n = ir.Var("n", ir.ScalarType(idx), span)
    choice = ir.Var("choice", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)
    x = ir.Var("x", vec_t, span)
    y = ir.Var("y", vec_t, span)
    z = ir.Var("z", vec_t, span)
    w = ir.Var("w", vec_t, span)

    # 1. Pure-payload BufferIR reproducer: MTE2->V in then-arm, V->MTE2 in else-arm inside a loop.
    # Both arms have closed cycle in loop; forward IDs must stay 2-cmd SetWait with no cross-arm race.
    cond = ir.Eq(ir.FloorMod(k, ci(2), idx, span), choice, ir.DataType.BOOL, span)
    payload_loop = ir.ForStmt(
        k,
        ci(0),
        n,
        ci(1),
        [],
        seq(
            ir.IfStmt(
                cond,
                seq(
                    ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), z], ir.VoidType(), span), span),
                    ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [z, w], ir.VoidType(), span), span),
                ),
                seq(
                    ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span),
                    ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ),
                [],
                span,
            ),
            ir.YieldStmt([], span),
        ),
        [],
        span,
    )
    payload_fn = ir.Function(
        "loop_if_else_payloads",
        [src, (out, ir.ParamDirection.InOut), n, choice],
        [out.type],
        seq(
            ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
            ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
            ir.AssignStmt(z, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(256)], vec_t, span), span),
            ir.AssignStmt(w, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(384)], vec_t, span), span),
            ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
            payload_loop,
            ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [w, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
            ir.ReturnStmt([out], span),
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    prog = ir.Program([payload_fn], "LoopIfElsePayloads", span)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_policy="conservative_prefix"):
        opt = passes.optimize_sync()(passes.insert_sync()(prog))
        p3 = next(iter(passes.lower_sync()(opt).functions.values()))
        p1 = next(iter(passes.realize_sync()(prog).functions.values()))
    log_fn = next(iter(opt.functions.values()))
    for phys in (p3, p1):
        for n_val, ch_val in product((0, 1, 2, 4), (0, 1)):
            check_closed_commands(log_fn, phys, {2: n_val, 3: ch_val}, range(8), compare_commands=False)

    # 2. Open-cycle cross-arm tokens: SetWaitAck ack_ids must not collide with opposite-arm forward_ids.
    t_then = ir.Var("t_then", ir.SyncTokenType(ir.PipeType.MTE2), span)
    t_else = ir.Var("t_else", ir.SyncTokenType(ir.PipeType.V), span)
    open_fn = ir.Function(
        "loop_if_else_open_tokens",
        [n, choice],
        [],
        seq(
            ir.ForStmt(
                k,
                ci(0),
                n,
                ci(1),
                [],
                seq(
                    ir.IfStmt(
                        cond,
                        seq(
                            ir.AssignStmt(t_then, sync.record(ir.PipeType.MTE2), span),
                            ir.EvalStmt(sync.wait(t_then, ir.PipeType.V), span),
                        ),
                        seq(
                            ir.AssignStmt(t_else, sync.record(ir.PipeType.V), span),
                            ir.EvalStmt(sync.wait(t_else, ir.PipeType.MTE2), span),
                        ),
                        [],
                        span,
                    ),
                    ir.YieldStmt([], span),
                ),
                [],
                span,
            ),
            ir.ReturnStmt([], span),
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    phys_open = lower(open_fn, [0, 1])
    for n_val, ch_val in product((0, 1, 2, 4), (0, 1)):
        _, after = check_closed_commands(open_fn, phys_open, {0: n_val, 1: ch_val}, [0, 1], compare_commands=False)
        assert not after.live


@pytest.mark.parametrize("kind", ["for", "while"])
def test_ten_sequential_unpromoted_loops_reuse_single_event_id(kind):
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    ci = lambda v: ir.ConstInt(v, idx, span)
    n = ir.Var("n", ir.ScalarType(idx), span)

    loops: list[ir.Stmt] = []
    for li in range(10):
        tok = ir.Var(f"tok_{li}", ir.SyncTokenType(ir.PipeType.MTE2), span)
        body_stmts: list[ir.Stmt] = [
            ir.AssignStmt(tok, sync.record(ir.PipeType.MTE2), span),
            ir.EvalStmt(sync.wait(tok, ir.PipeType.V), span),
        ]
        if kind == "while":
            iv = ir.IterArg(f"i_{li}", ir.ScalarType(idx), ci(0), span)
            final = ir.Var(f"fin_{li}", ir.ScalarType(idx), span)
            body_stmts.append(ir.YieldStmt([ir.Add(iv, ci(1), idx, span)], span))
            loops.append(
                ir.WhileStmt(
                    ir.Lt(iv, n, ir.DataType.BOOL, span), [iv], seq(*body_stmts), [final], span
                )
            )
        else:
            iv = ir.Var(f"i_{li}", ir.ScalarType(idx), span)
            body_stmts.append(ir.YieldStmt([], span))
            loops.append(ir.ForStmt(iv, ci(0), n, ci(1), [], seq(*body_stmts), [], span))

    fn = ir.Function(
        f"ten_seq_{kind}_loops",
        [n],
        [],
        seq(*loops, ir.ReturnStmt([], span)),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([]):
        phys = next(iter(passes.lower_sync()(ir.Program([fn], "TenLoops", span)).functions.values()))
    _, after = check_closed_commands(fn, phys, {0: 2}, range(8), compare_commands=False)
    assert after.ids[ir.PipeType.MTE2, ir.PipeType.V] == {0}


@pytest.mark.parametrize("control", ["straight", "while", "if"])
def test_closed_prefix_refuses_stale_same_pipe_barrier_with_intervening_payload(control):
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    ci = lambda v: ir.ConstInt(v, idx, span)
    tup = lambda vals: ir.MakeTuple([ci(v) for v in vals], span)
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)

    src = ir.Var("src", gm_t, span)
    cond = ir.Var("cond", ir.ScalarType(ir.DataType.BOOL), span)
    x = ir.Var("x", vec_t, span)
    z = ir.Var("z", vec_t, span)
    tok = ir.Var("tok", ir.SyncTokenType(ir.PipeType.MTE2), span)

    stale_seq = seq(
        ir.AssignStmt(tok, sync.record(ir.PipeType.MTE2), span),
        ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), z], ir.VoidType(), span), span),
        ir.EvalStmt(sync.wait(tok, ir.PipeType.MTE2), span),
    )
    if control == "straight":
        middle = stale_seq
    elif control == "while":
        middle = ir.WhileStmt(cond, [], seq(*stale_seq.stmts, ir.YieldStmt([], span)), [], span)
    else:
        middle = ir.IfStmt(cond, stale_seq, None, [], span)

    fn = ir.Function(
        f"stale_same_pipe_{control}",
        [src, cond],
        [],
        seq(
            ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
            ir.AssignStmt(z, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
            ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
            middle,
            ir.ReturnStmt([], span),
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    before = ir.serialize(fn)
    with pytest.raises(ValueError, match="LowerSync same-pipe barrier would strengthen the recorded prefix"):
        with passes.PassContext([]):
            passes.lower_sync()(ir.Program([fn], "StaleSamePipe", span))
    assert ir.serialize(fn) == before


@pytest.mark.parametrize("in_loop", [False, True])
def test_128_sequential_one_sided_ifs_reuse_ids_without_pool_exhaustion(in_loop):
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX

    def ci(v):
        return ir.ConstInt(v, idx, span)

    n = ir.Var("n", ir.ScalarType(idx), span)
    choice = ir.Var("choice", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)

    def make_arm(i):
        t_32 = ir.Var(f"t32_{i}", ir.SyncTokenType(ir.PipeType.MTE3), span)
        t_2s = ir.Var(f"t2s_{i}", ir.SyncTokenType(ir.PipeType.MTE2), span)
        t_s3a = ir.Var(f"ts3a_{i}", ir.SyncTokenType(ir.PipeType.S), span)
        t_3s = ir.Var(f"t3s_{i}", ir.SyncTokenType(ir.PipeType.MTE3), span)
        t_s3b = ir.Var(f"ts3b_{i}", ir.SyncTokenType(ir.PipeType.S), span)
        return seq(
            ir.AssignStmt(t_32, sync.record(ir.PipeType.MTE3), span),
            ir.EvalStmt(sync.wait(t_32, ir.PipeType.MTE2), span),
            ir.AssignStmt(t_2s, sync.record(ir.PipeType.MTE2), span),
            ir.EvalStmt(sync.wait(t_2s, ir.PipeType.S), span),
            ir.AssignStmt(t_s3a, sync.record(ir.PipeType.S), span),
            ir.EvalStmt(sync.wait(t_s3a, ir.PipeType.MTE3), span),
            ir.AssignStmt(t_3s, sync.record(ir.PipeType.MTE3), span),
            ir.EvalStmt(sync.wait(t_3s, ir.PipeType.S), span),
            ir.AssignStmt(t_s3b, sync.record(ir.PipeType.S), span),
            ir.EvalStmt(sync.wait(t_s3b, ir.PipeType.MTE3), span),
        )

    init_tok = ir.Var("t_init", ir.SyncTokenType(ir.PipeType.S), span)
    branches = []
    for i in range(128):
        lhs = ci(i) if not in_loop else ir.Add(k, ci(i), idx, span)
        cond = ir.Eq(ir.FloorMod(lhs, ci(2), idx, span), choice, ir.DataType.BOOL, span)
        branches.append(ir.IfStmt(cond, make_arm(i), None, [], span))

    if in_loop:
        body = seq(ir.ForStmt(k, ci(0), n, ci(1), [], seq(*branches, ir.YieldStmt([], span)), [], span))
    else:
        body = seq(*branches)

    fn = ir.Function(
        f"qkpv_branch_reuse_{'loop' if in_loop else 'flat'}",
        [n, choice],
        [],
        seq(
            ir.AssignStmt(init_tok, sync.record(ir.PipeType.S), span),
            ir.EvalStmt(sync.wait(init_tok, ir.PipeType.MTE3), span),
            body,
            ir.ReturnStmt([], span),
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    phys = lower(fn, eligible=range(8))
    max_id_limit = 2 if in_loop else 1
    for n_val, ch_val in product((0, 1, 2) if in_loop else (1,), (0, 1)):
        _, after = check_closed_commands(fn, phys, {0: n_val, 1: ch_val}, range(8), compare_commands=False)
        assert not after.live
        assert all(max(ids) < max_id_limit for ids in after.ids.values() if ids)


def test_loop_one_sided_if_set_wait_ack_propagates_relay_and_reuses_ids():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX

    def ci(v):
        return ir.ConstInt(v, idx, span)

    n = ir.Var("n", ir.ScalarType(idx), span)
    choice = ir.Var("choice", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)

    branches = []
    for i in range(20):
        t_s3 = ir.Var(f"ts3_{i}", ir.SyncTokenType(ir.PipeType.S), span)
        t_3s = ir.Var(f"t3s_{i}", ir.SyncTokenType(ir.PipeType.MTE3), span)
        cond = ir.Eq(
            ir.FloorMod(ir.Add(k, ci(i), idx, span), ci(2), idx, span), choice, ir.DataType.BOOL, span
        )
        branches.append(
            ir.IfStmt(
                cond,
                seq(
                    ir.AssignStmt(t_s3, sync.record(ir.PipeType.S), span),
                    ir.EvalStmt(sync.wait(t_s3, ir.PipeType.MTE3), span),
                    ir.AssignStmt(t_3s, sync.record(ir.PipeType.MTE3), span),
                    ir.EvalStmt(sync.wait(t_3s, ir.PipeType.S), span),
                ),
                None,
                [],
                span,
            )
        )

    fn = ir.Function(
        "loop_set_wait_ack_branch_relay",
        [n, choice],
        [],
        seq(
            ir.ForStmt(k, ci(0), n, ci(1), [], seq(*branches, ir.YieldStmt([], span)), [], span),
            ir.ReturnStmt([], span),
        ),
        span,
        type=ir.FunctionType.AIV,
        attrs={"autosync_analysis_policy": "conservative_prefix"},
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    phys = lower(fn, eligible=[0, 1])
    for n_val, ch_val in product((0, 1, 3), (0, 1)):
        _, after = check_closed_commands(fn, phys, {0: n_val, 1: ch_val}, [0, 1], compare_commands=False)
        assert not after.live


def test_interleaved_loop_records_require_return_ack_and_reviewer_probes():
    # 1. Interleaved loop record/wait (b recorded before wait(a, V) cannot relay wait(a, V))
    a = ir.Var("a", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    b = ir.Var("b", ir.SyncTokenType(ir.PipeType.V), SPAN)
    iv = ir.Var("i", ir.ScalarType(ir.DataType.INDEX), SPAN)
    loop_body = seq(
        ir.AssignStmt(a, sync.record(ir.PipeType.MTE2), SPAN),
        ir.AssignStmt(b, sync.record(ir.PipeType.V), SPAN),
        ir.EvalStmt(sync.wait(a, ir.PipeType.V), SPAN),
        ir.EvalStmt(sync.wait(b, ir.PipeType.MTE2), SPAN),
        ir.YieldStmt([], SPAN),
    )
    for_loop = ir.ForStmt(iv, ci(0), ci(3), ci(1), [], loop_body, [], SPAN)
    fn_for = ir.Function(
        "interleaved_for",
        [],
        [],
        seq(for_loop, ir.ReturnStmt([], SPAN)),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    verify(fn_for)
    phys_for = lower(fn_for)
    ClosedCommands(phys_for, {}, range(8)).run()
    with passes.PassContext([], enable_pypto_autosync=True):
        pub_for = passes.lower_sync()(ir.Program([fn_for], "P1", SPAN)).get_function("interleaved_for")
    assert pub_for is not None
    ClosedCommands(pub_for, {}, range(8)).run()

    # 2. Manual barrier-only requirement on disjoint buffers in default LowerSync
    buf_type = ir.BufferType([16, 32], ir.DataType.FP32, ir.MemorySpace.Vec, [16, 32])
    x = ir.Var("x", buf_type, SPAN)
    y = ir.Var("y", buf_type, SPAN)
    tok = ir.Var("t", ir.SyncTokenType(ir.PipeType.V), SPAN)
    alloc_x = ir.AssignStmt(
        x, ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], SPAN), ci(0)], {}, None, buf_type, SPAN), SPAN
    )
    alloc_y = ir.AssignStmt(
        y,
        ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], SPAN), ci(4096)], {}, None, buf_type, SPAN),
        SPAN,
    )
    full_x = ir.EvalStmt(
        ir.Call(
            ir.get_op("buffer.full"),
            [ir.ConstFloat(1.0, ir.DataType.FP32, SPAN), x],
            {},
            ir.VoidType(),
            SPAN,
        ),
        SPAN,
    )
    full_y = ir.EvalStmt(
        ir.Call(
            ir.get_op("buffer.full"),
            [ir.ConstFloat(2.0, ir.DataType.FP32, SPAN), y],
            {},
            ir.VoidType(),
            SPAN,
        ),
        SPAN,
    )
    fn_orig = ir.Function(
        "manual_bar",
        [],
        [],
        seq(alloc_x, alloc_y, full_x, full_y, ir.ReturnStmt([], SPAN)),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    fn_manual = ir.Function(
        "manual_bar",
        [],
        [],
        seq(
            alloc_x,
            alloc_y,
            full_x,
            ir.AssignStmt(tok, sync.record(ir.PipeType.V), SPAN),
            ir.EvalStmt(sync.wait(tok, ir.PipeType.V), SPAN),
            full_y,
            ir.ReturnStmt([], SPAN),
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        lowered_manual = passes.lower_sync(require_whole_module=False)(
            ir.Program([fn_manual], "P2", SPAN)
        ).get_function("manual_bar")
    assert lowered_manual is not None
    _check_actual_tokens(fn_orig, fn_manual, lowered_manual)
    assert dict(lowered_manual.attrs)["pypto_autosync"] is False

    # 3. Buffer BYPASS load wrapper detection
    tensor_a = ir.Var("a", ir.TensorType([ci(16), ci(32)], ir.DataType.FP32), SPAN)
    load_bypass = ir.EvalStmt(
        ir.Call(
            ir.get_op("buffer.load"),
            [tensor_a, ir.MakeTuple([ci(0), ci(0)], SPAN), ir.MakeTuple([ci(16), ci(32)], SPAN), x],
            {"cache": int(ir.CachePolicy.BYPASS)},
            ir.VoidType(),
            SPAN,
        ),
        SPAN,
    )
    fn_bypass = ir.Function(
        "bypass_buf",
        [tensor_a],
        [],
        seq(alloc_x, load_bypass, ir.ReturnStmt([], SPAN)),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    assert pto_backend._uses_l2_cache_offset(fn_bypass) is True

    # 4. OptimizeSync nested join liveness
    ta = ir.Var("ta", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    tb = ir.Var("tb", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    tj = ir.Var("tj", ir.SyncTokenType(ir.PipeType.MTE2), SPAN)
    fn_join = ir.Function(
        "nested_join",
        [],
        [],
        seq(
            ir.AssignStmt(ta, sync.record(ir.PipeType.MTE2), SPAN),
            ir.AssignStmt(tb, sync.record(ir.PipeType.MTE2), SPAN),
            ir.AssignStmt(tj, sync.join(sync.join(ta, tb), sync.completed(ir.PipeType.MTE2)), SPAN),
            ir.EvalStmt(sync.wait(tj, ir.PipeType.V), SPAN),
            ir.ReturnStmt([], SPAN),
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    verify(fn_join)
    opt_join = passes.optimize_sync()(ir.Program([fn_join], "P4", SPAN)).get_function("nested_join")
    assert opt_join is not None
    verify(opt_join)

    # 5. Vec->Vec extract enclosure precision
    src_t = ir.BufferType([32, 64], ir.DataType.FP32, ir.MemorySpace.Vec, [32, 64])
    dst_t = ir.BufferType([16, 32], ir.DataType.FP32, ir.MemorySpace.Vec, [16, 32])
    v_src = ir.Var("v_src", src_t, SPAN)
    v_dst = ir.Var("v_dst", dst_t, SPAN)
    fn_ext = ir.Function(
        "vec_ext",
        [tensor_a],
        [],
        seq(
            ir.AssignStmt(
                v_src,
                ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], SPAN), ci(0)], {}, None, src_t, SPAN),
                SPAN,
            ),
            ir.AssignStmt(
                v_dst,
                ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], SPAN), ci(8192)], {}, None, dst_t, SPAN),
                SPAN,
            ),
            ir.EvalStmt(
                ir.Call(
                    ir.get_op("buffer.extract"), [v_src, ci(8), ci(16), v_dst], {}, ir.VoidType(), SPAN
                ),
                SPAN,
            ),
            ir.ReturnStmt([], SPAN),
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    retained_ext = testing.analyze_placed_sync(fn_ext, True).describe()
    ext_site = retained_ext["regions"][0]["sites"][0]
    assert ext_site["access_precision"] == ["enclosure", "exact"]
    assert ext_site["physical_intervals"][0] is None

    # 6. Conservative solver scaling on 512 independent windows
    t0 = time.perf_counter()
    gm_512 = ir.Var("source", ir.TensorType([2, 8], ir.DataType.FP32), SPAN)
    local_512 = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    stmts_512 = []
    zero_tup = ir.MakeTuple([ci(0), ci(0)], SPAN)
    shape_tup = ir.MakeTuple([ci(2), ci(8)], SPAN)
    empty_tup = ir.MakeTuple([], SPAN)
    for idx in range(512):
        s_var = ir.Var(f"x_{idx}", local_512, SPAN)
        t_var = ir.Var(f"z_{idx}", local_512, SPAN)
        for var, addr in ((s_var, 128 * idx), (t_var, 128 * idx + 64)):
            stmts_512.append(
                ir.AssignStmt(
                    var, ir.Call(ir.get_op("buffer.alloc"), [empty_tup, ci(addr)], local_512, SPAN), SPAN
                )
            )
        stmts_512.append(
            ir.EvalStmt(
                ir.Call(ir.get_op("buffer.load"), [gm_512, zero_tup, shape_tup, s_var], ir.VoidType(), SPAN),
                SPAN,
            )
        )
        stmts_512.append(
            ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [s_var, t_var], ir.VoidType(), SPAN), SPAN)
        )
    stmts_512.append(ir.ReturnStmt([], SPAN))
    fn_512 = ir.Function(
        "independent_windows",
        [gm_512],
        [],
        ir.SeqStmts(stmts_512, SPAN),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    res_512 = testing.analyze_conservative_sync(
        fn_512, automatic_children=False, retain_incoming_diagnostics=False
    )
    dt = time.perf_counter() - t0
    assert res_512["cost"]["retained_facts"] == 2048
    assert dt < 0.25
