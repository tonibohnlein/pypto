# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lower actual finite completion-token semantics, independently of demand reduction.

The original access interpreter supplies payload identities and concrete hazards.
The command interpreters supply order from the actual tokens and actual flags.
In particular, manually strengthened token programs are compared to their token
order, not silently replaced by minimum demands for the underlying accesses.
"""

import pytest
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.op import sync

from .sync_logical_oracle import CommandExecution, Token, value
from .sync_matmul_cases import capture, staged_fixture
from .sync_matmul_oracle import hardware_reference, interpret
from .sync_physical_oracle import PhysicalExecution, directional_widths
from .test_autosync_switch import clone
from .test_finite_sync_analysis import closure
from .test_logical_sync_realization import verify
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


class FiniteTokenExecution(CommandExecution):
    """Extend the independent oracle to non-consuming joins and empty facts."""

    def assign(self, stmt, env):
        call = stmt.value
        if isinstance(call, ir.Call) and call.op.name == ir.get_op("sync.join").name:
            members = []
            for argument in call.args:
                fact = value(argument, env)
                members.extend(fact if isinstance(fact, tuple) else (fact,))
            env[stmt.var] = tuple(members)
        else:
            super().assign(stmt, env)

    def wait(self, call, env):
        fact = value(call.args[0], env)
        members = fact if isinstance(fact, tuple) else (fact,)
        for member in members:
            assert isinstance(member, Token)
            if member.node is None:
                continue
            local = dict(env)
            local[call.args[0]] = member
            super().wait(call, local)


def _apply(factory, function):
    program = ir.Program([function], "TokenPipeline", function.span)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        result = factory()(program)
    return next(iter(result.functions.values()))


def _project(vertices, edges, payload_vertices):
    order = closure(vertices, edges)
    assert not any(a == b for a, b in order), "cyclic emitted command graph"
    payloads = set(payload_vertices)
    return order, {(a, b) for a, b in order if a in payloads and b in payloads}


def _check_actual_tokens(original, logical, physical):
    """Check all token-induced order, matching and reuse without compiler families."""
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    token_state = FiniteTokenExecution(calls, reference[0])
    token_state.execute(logical.body, {})
    _, expected = _project(token_state.vertices, token_state.edges, reference[0])
    flag_state = PhysicalExecution(calls, reference[0], range(6))
    flag_state.execute(physical.body, {})
    order, actual = _project(flag_state.vertices, flag_state.edges, reference[0])
    assert token_state.payload_count == flag_state.payload_count == len(calls)
    assert not flag_state.live, "unconsumed physical notification at invocation exit"
    assert len(flag_state.records) == len(flag_state.waits)
    assert all(pair in order for pair in flag_state.reuses), "unsafe physical flag reuse"
    assert actual == expected, ("token lowering changed payload order", actual - expected, expected - actual)
    assert reference[2] <= actual, ("missing modeled access coverage", reference[2] - actual)
    assert not any(flag_state.pending.values()), "physical wait lacks its intended payload consumer"
    assert all(ids <= set(range(6)) for ids in flag_state.ids.values())
    # The fixed graph is acyclic and matched, so the directional chain-width
    # theorem applies. This is not minimum over alternative token protocols.
    assert directional_widths(flag_state, order) == {
        direction: len(ids) for direction, ids in flag_state.ids.items()
    }
    assert len(token_state.actual_allocations) == len(flag_state.actual_allocations)
    program = verify(physical)
    text = codegen.PTOCodegen().generate(program)
    assert "sync.record" not in text and "!pto.sync_token" not in text
    return actual, reference[2], flag_state


@pytest.mark.parametrize("name", ["matmul_64", "matmul_acc_64"])
def test_public_finite_matmul_pipeline(name):
    original = capture(name)
    logical = _apply(passes.insert_sync, original)
    verify(logical)
    optimized = _apply(passes.optimize_sync, logical)
    ir.assert_structural_equal(logical, optimized)
    physical = _apply(passes.lower_sync, optimized)
    actual, required, state = _check_actual_tokens(original, logical, physical)
    assert actual == required
    assert state.records


@pytest.mark.parametrize("view", ["dense", "overlap", "nested", "reshape", "byte_root"])
def test_public_finite_vec_pipeline(view):
    original = vec_fixture(view=view)
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def _two_streams(*, imported_prefix=False, strengthen=False, repeated_wait=False, joined=False):
    """Two disjoint streams with explicit tokens and no demand-analysis sidecar.

    The imported-prefix variant records MTE2 immediately after importing the
    vector completion. There is no intervening MTE2 payload: associating that
    record only with its latest load would drop the required abs->store edge.
    """
    span = ir.Span.unknown()
    fp, index = ir.DataType.FP32, ir.DataType.INDEX

    def ci(number):
        return ir.ConstInt(number, index, span)

    def tup(numbers):
        return ir.MakeTuple([ci(number) for number in numbers], span)

    def payload(name, args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), args, ir.VoidType(), span), span)

    def token(name, pipe):
        return ir.Var(name, ir.SyncTokenType(pipe), span)

    def record(var):
        return ir.AssignStmt(var, sync.record(var.type.pipe), span)

    def wait(var, pipe):
        return ir.EvalStmt(sync.wait(var, pipe), span)

    tensor_type = ir.TensorType([2, 8], fp)
    a, b, out_x, out_y = [ir.Var(name, tensor_type, span) for name in ["a", "b", "out_x", "out_y"]]
    buffer_type = ir.BufferType([2, 8], fp, ir.MemorySpace.Vec)
    x, y = [ir.Var(name, buffer_type, span) for name in ["x", "y"]]
    allocations = [
        ir.AssignStmt(
            var, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(address)], buffer_type, span), span
        )
        for var, address in [(x, 0), (y, 1024)]
    ]
    load_x = payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x])
    compute_x = payload("buffer.abs", [x, x])
    store_x = payload("buffer.store", [x, tup([0, 0]), tup([2, 8]), out_x])
    load_y = payload("buffer.load", [b, tup([0, 0]), tup([2, 8]), y])
    store_y = payload("buffer.store", [y, tup([0, 0]), tup([2, 8]), out_y])
    finish = ir.ReturnStmt([out_x, out_y], span)
    original = ir.Function(
        "two_streams",
        [a, b, (out_x, ir.ParamDirection.InOut), (out_y, ir.ParamDirection.InOut)],
        [tensor_type, tensor_type],
        ir.SeqStmts([*allocations, load_x, compute_x, store_x, load_y, store_y, finish], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    loaded_x = token("loaded_x", ir.PipeType.MTE2)
    computed_x = token("computed_x", ir.PipeType.V)
    loaded_y = token("loaded_y", ir.PipeType.MTE2)
    body: list[ir.Stmt] = [*allocations, load_x, record(loaded_x)]
    input_token = loaded_x
    if joined:
        empty = token("empty", ir.PipeType.MTE2)
        combined = token("combined", ir.PipeType.MTE2)
        body.extend(
            [
                ir.AssignStmt(empty, sync.completed(ir.PipeType.MTE2), span),
                wait(empty, ir.PipeType.V),
                ir.AssignStmt(combined, sync.join(loaded_x, empty), span),
            ]
        )
        input_token = combined
    body.append(wait(input_token, ir.PipeType.V))
    if repeated_wait:
        body.append(wait(input_token, ir.PipeType.V))
    body.extend([compute_x, record(computed_x)])
    if imported_prefix:
        imported = token("imported", ir.PipeType.MTE2)
        body.extend([wait(computed_x, ir.PipeType.MTE2), record(imported), wait(imported, ir.PipeType.MTE3)])
    else:
        body.append(wait(computed_x, ir.PipeType.MTE3))
    # Fanout of the earlier immutable token is legal even after its first wait.
    if repeated_wait:
        body.append(wait(input_token, ir.PipeType.MTE3))
    body.append(store_x)
    if strengthen:
        stored_x = token("stored_x", ir.PipeType.MTE3)
        body.extend([record(stored_x), wait(stored_x, ir.PipeType.MTE2)])
    body.extend([load_y, record(loaded_y)])
    output_token = loaded_y
    if joined:
        both_loads = token("both_loads", ir.PipeType.MTE2)
        body.append(ir.AssignStmt(both_loads, sync.join(loaded_x, loaded_y), span))
        output_token = both_loads
    body.extend([wait(output_token, ir.PipeType.MTE3), store_y, finish])
    return original, clone(original, body=ir.SeqStmts(body, span))


@pytest.mark.parametrize("repeated_wait", [False, True])
@pytest.mark.parametrize("joined", [False, True])
def test_handwritten_completed_join_fanout_and_repeated_waits(repeated_wait, joined):
    original, logical = _two_streams(repeated_wait=repeated_wait, joined=joined)
    verify(logical)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def test_record_captures_imported_wait_without_an_intervening_payload():
    original, logical = _two_streams(imported_prefix=True)
    physical = _apply(passes.lower_sync, logical)
    actual, _, _ = _check_actual_tokens(original, logical, physical)
    assert ((1, True), (2, False)) in actual


def test_manual_additional_prerequisite_is_not_reduced_to_access_demands():
    original, logical = _two_streams(strengthen=True)
    physical = _apply(passes.lower_sync, logical)
    actual, required, _ = _check_actual_tokens(original, logical, physical)
    # Separate buffers do not require this order. The user explicitly requested
    # it through tokens, so LowerSync must preserve it even though InsertSync
    # would not generate it from the original access model.
    prerequisite = ((2, True), (3, False))
    assert prerequisite in actual and prerequisite not in required


def test_serialized_tokens_lower_without_retained_analysis(tmp_path):
    original, logical = _two_streams(imported_prefix=True, repeated_wait=True, joined=True)
    path = tmp_path / "token-input.msgpack"
    ir.serialize_to_file(logical, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    physical = _apply(passes.lower_sync, restored)
    _check_actual_tokens(original, restored, physical)


def test_insert_sync_rejects_already_logical_input():
    _, logical = _two_streams()
    with pytest.raises(ValueError, match="[Ll]ogical|[Tt]oken|[Ss]ynchronization"):
        _apply(passes.insert_sync, logical)


def test_noncanonical_structured_guard_refuses_without_mutation():
    original = staged_fixture(2)
    logical = _apply(passes.insert_sync, original)

    class ChangeGuard(ir.IRMutator):
        def visit_gt(self, op):
            return ir.Ge(op.left, op.right, ir.DataType.BOOL, op.span)

    logical = ChangeGuard().visit_function(logical)
    before = str(logical)
    with pytest.raises(ValueError, match="Unsupported token guard"):
        _apply(passes.lower_sync, logical)
    assert str(logical) == before


def test_same_pipe_stale_prefix_refuses_instead_of_completing_later_work():
    _, logical = _two_streams()
    assert isinstance(logical.body, ir.SeqStmts)
    statements = list(logical.body.stmts)
    computed = next(
        statement.var
        for statement in statements
        if isinstance(statement, ir.AssignStmt) and statement.var.name_hint == "computed_x"
    )
    store_index = next(
        index
        for index, statement in enumerate(statements)
        if isinstance(statement, ir.EvalStmt)
        and isinstance(statement.expr, ir.Call)
        and statement.expr.op.name == ir.get_op("buffer.store").name
    )
    store = statements[store_index]
    assert isinstance(store, ir.EvalStmt) and isinstance(store.expr, ir.Call)
    stored_buffer = store.expr.args[0]
    span = logical.span
    later_work = ir.EvalStmt(
        ir.Call(ir.get_op("buffer.abs"), [stored_buffer, stored_buffer], ir.VoidType(), span), span
    )
    stale_wait = ir.EvalStmt(sync.wait(computed, ir.PipeType.V), span)
    # The earlier V record excludes later_work. A barrier after later_work
    # would wait for more than the input token requests. The initial lowering
    # has no selective same-pipe completion primitive and must refuse.
    following_work = ir.EvalStmt(
        ir.Call(ir.get_op("buffer.abs"), [stored_buffer, stored_buffer], ir.VoidType(), span), span
    )
    statements[store_index:store_index] = [later_work, stale_wait, following_work]
    logical = clone(logical, body=ir.SeqStmts(statements, span))
    verify(logical)
    before = str(logical)
    with pytest.raises(ValueError, match="[Ss]ame.pipe|[Pp]refix"):
        _apply(passes.lower_sync, logical)
    assert str(logical) == before, "failed physical qualification mutated the source token program"


@pytest.mark.parametrize(
    "kind", [ir.FunctionType.InCore, ir.FunctionType.Orchestration, ir.FunctionType.SIMT]
)
def test_lower_sync_rejects_tokens_outside_specialized_leaf_scope(kind):
    _, logical = _two_streams()
    body = logical.body
    returns = logical.return_types
    if kind == ir.FunctionType.SIMT:
        assert isinstance(body, ir.SeqStmts)
        body = ir.SeqStmts([*body.stmts[:-1], ir.ReturnStmt([], logical.span)], logical.span)
        returns = []
    invalid = ir.Function(
        logical.name,
        list(zip(logical.params, logical.param_directions)),
        returns,
        body,
        logical.span,
        type=kind,
        ir_stage=logical.ir_stage,
    )
    before = str(invalid)
    with pytest.raises((ValueError, RuntimeError), match="[Ss]ync|[Tt]oken|AIC|AIV|[Cc]ore"):
        _apply(passes.lower_sync, invalid)
    assert str(invalid) == before


def test_shared_join_dag_is_not_expanded():
    original, logical = _two_streams()

    class SharedJoin(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                token = op.expr.args[0]
                # Linear DAG, exponentially many paths. Token interpretation and
                # preflight scans must visit distinct nodes, not expand paths.
                for _ in range(28):
                    token = sync.join(token, token)
                return ir.EvalStmt(sync.wait(token, ir.PipeType(op.expr.kwargs["dst_pipe"])), op.span)
            return op

    dag = SharedJoin().visit_function(logical)
    physical = _apply(passes.lower_sync, dag)
    # Compare to the unduplicated token graph; the bounded oracle deliberately
    # does not expand the exponential paths in the shared synthetic join DAG.
    _check_actual_tokens(original, logical, physical)


def test_unreachable_token_cuts_refuse_without_mutation():
    original, logical = _two_streams()
    assert isinstance(logical.body, ir.SeqStmts)
    body = list(logical.body.stmts)
    body.append(
        next(
            statement
            for statement in reversed(body)
            if isinstance(statement, ir.EvalStmt)
            and isinstance(statement.expr, ir.Call)
            and statement.expr.op.name == ir.get_op("sync.wait").name
        )
    )
    invalid = clone(logical, body=ir.SeqStmts(body, logical.span))
    before = str(invalid)
    with pytest.raises(ValueError, match="terminal return|unreachable"):
        _apply(passes.lower_sync, invalid)
    assert str(invalid) == before


def test_empty_record_is_satisfied_without_a_physical_publication():
    original = vec_fixture()
    empty = ir.Var("empty_prefix", ir.SyncTokenType(ir.PipeType.MTE2), original.span)
    logical = clone(
        original,
        body=ir.SeqStmts(
            [
                ir.AssignStmt(empty, sync.record(ir.PipeType.MTE2), original.span),
                ir.EvalStmt(sync.wait(empty, ir.PipeType.V), original.span),
                original.body,
            ],
            original.span,
        ),
    )
    physical = _apply(passes.lower_sync, logical)
    # Lowering does not fill in absent storage dependencies on behalf of a
    # manual program. This test only establishes empty-token erasure.
    ir.assert_structural_equal(original.body, physical.body)


def test_single_statement_completed_wait_root_lowers():
    span = ir.Span.unknown()
    function = ir.Function(
        "empty_wait",
        [],
        [],
        ir.EvalStmt(sync.wait(sync.completed(ir.PipeType.MTE2), ir.PipeType.V), span),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    physical = _apply(passes.lower_sync, function)
    assert isinstance(physical.body, ir.SeqStmts) and not physical.body.stmts


@pytest.mark.parametrize("depth", [1, 16, 128])
def test_nested_statement_sequences_preserve_token_order(depth):
    original, logical = _two_streams(imported_prefix=True)
    body = logical.body
    for index in range(depth):
        token = ir.Var(f"unused_completed_{index}", ir.SyncTokenType(ir.PipeType.V), logical.span)
        body = ir.SeqStmts(
            [ir.AssignStmt(token, sync.completed(ir.PipeType.V), logical.span), body], logical.span
        )
    nested = clone(logical, body=body)
    physical = _apply(passes.lower_sync, nested)
    _check_actual_tokens(original, logical, physical)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("name", ["matmul_64", "matmul_acc_64", "finite_vec"])
def test_token_pipeline_generates_cpp(name, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    original = vec_fixture() if name == "finite_vec" else capture(name)
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _, _, state = _check_actual_tokens(original, logical, physical)
    source, output = tmp_path / "tokens.pto", tmp_path / "tokens.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([physical], "Tokens", physical.span), emit_tile_addr=True, emit_source_loc=False
        )
    )
    _run_ptoas(
        str(source),
        str(output),
        ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"],
    )
    kernel_body = output.read_text().split("AICORE void ", 1)[1]
    assert "ptoas_auto_sync_tail(" not in kernel_body
    assert kernel_body.count("set_flag(") == len(state.records)
    assert kernel_body.count("wait_flag(") == len(state.waits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
