# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Interpret emitted token SSA and build a command graph without demand edges.

Physical access conflicts come from the separate original-IR interpreter. Token
meaning comes only from executed record/wait calls, branches and loop carries.
This tests logical completion prefixes, not consuming flags or native lowering.
"""

from dataclasses import dataclass
from typing import Any

from pypto import ir

from .sync_matmul_oracle import cast_integer, hardware_reference, interpret
from .test_finite_sync_analysis import closure


@dataclass(frozen=True)
class Token:
    pipe: ir.PipeType
    node: tuple | None
    source: int | None


def value(expr, env):
    if isinstance(expr, (ir.ConstInt, ir.ConstBool)):
        return expr.value
    if isinstance(expr, ir.Cast):
        assert isinstance(expr.type, ir.ScalarType)
        return cast_integer(value(expr.operand, env), expr.type.dtype)
    if isinstance(expr, ir.Var):
        assert expr in env, f"unbound value: {expr.name_hint}"
        return env[expr]
    if isinstance(expr, (ir.Add, ir.Sub, ir.Mul, ir.FloorDiv, ir.FloorMod, ir.Eq, ir.Gt, ir.Ge, ir.Lt)):
        a, b = value(expr.left, env), value(expr.right, env)
        if isinstance(expr, ir.Add):
            return a + b
        if isinstance(expr, ir.Sub):
            return a - b
        if isinstance(expr, ir.Mul):
            return a * b
        if isinstance(expr, ir.FloorDiv):
            return a // b
        if isinstance(expr, ir.FloorMod):
            return a % b
        if isinstance(expr, ir.Ge):
            return a >= b
        if isinstance(expr, ir.Lt):
            return a < b
        return a == b if isinstance(expr, ir.Eq) else a > b
    if isinstance(expr, ir.Call) and expr.op.name == ir.get_op("sync.completed").name:
        return Token(ir.PipeType(expr.kwargs["pipe"]), None, None)
    raise AssertionError(f"Unsupported logical oracle expression: {type(expr)}")


def same_payload(before, after):
    class Names(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.names = []

        def visit_var(self, op):
            self.names.append(op.name_hint)

    a, b = Names(), Names()
    if isinstance(before, ir.Expr):
        a.visit_expr(before)
        b.visit_expr(after)
    else:
        a.visit_stmt(before)
        b.visit_stmt(after)
    assert a.names == b.names, "changed original storage or coordinate identity"
    ir.assert_structural_equal(before, after, enable_auto_mapping=True)


class CommandExecution:
    def __init__(self, calls, vertices):
        self.calls = calls
        self.vertices, self.edges = set(vertices), set()
        self.latest_payload, self.gate, self.pending = {}, {}, {}
        self.records, self.waits, self.matched = [], [], set()
        self.payload_count = self.branches = self.same_pipe_waits = 0
        self.actual_allocations = []
        self.actual_slots = []

    def assign(self, stmt, env):
        call = stmt.value
        if isinstance(call, ir.Call) and call.op.name in {
            ir.get_op("buffer.alloc").name,
            ir.get_op("buffer.alloc_multi").name,
            ir.get_op("buffer.subview").name,
            ir.get_op("buffer.reshape").name,
        }:
            self.actual_allocations.append(stmt)
        elif isinstance(call, ir.Call) and call.op.name == ir.get_op("buffer.slot").name:
            # Selection has no payload effect. Preserve both its original IR
            # and the selected dynamic slot across inserted command guards.
            self.actual_slots.append((stmt, value(call.args[1], env)))
        elif isinstance(call, ir.Call) and call.op.name == ir.get_op("sync.record").name:
            pipe = ir.PipeType(call.kwargs["pipe"])
            assert pipe in self.latest_payload, "record has no source payload"
            source = self.latest_payload[pipe]
            node = ("record", len(self.records))
            self.vertices.add(node)
            self.edges.add(((source, True), node))
            if pipe in self.gate:
                self.edges.add((self.gate[pipe], node))
            assert isinstance(stmt.var.type, ir.SyncTokenType) and stmt.var.type.pipe == pipe
            env[stmt.var] = Token(pipe, node, source)
            self.records.append(node)
        else:
            env[stmt.var] = value(stmt.value, env)

    def wait(self, call, env):
        token = value(call.args[0], env)
        assert isinstance(token, Token)
        # This construction guards every absent predecessor. Completed
        # carries are initial SSA values, never an active demand source.
        assert token.node is not None and token.source is not None, "wait on empty token"
        destination = ir.PipeType(call.kwargs["dst_pipe"])
        assert isinstance(call.args[0].type, ir.SyncTokenType)
        assert call.args[0].type.pipe == token.pipe, "source pipeline mismatch"
        node = ("wait", len(self.waits))
        self.vertices.add(node)
        self.edges.add((token.node, node))
        if destination in self.gate:
            self.edges.add((self.gate[destination], node))
        if destination in self.latest_payload:
            self.edges.add(((self.latest_payload[destination], False), node))
        self.gate[destination] = node
        self.pending.setdefault(destination, []).append(token.source)
        self.waits.append(token.node)
        self.same_pipe_waits += destination == token.pipe

    def payload(self, call, env):
        assert self.payload_count < len(self.calls), "extra payload"
        original_call, coordinates, pipe, _, _ = self.calls[self.payload_count]
        same_payload(call, original_call)
        for name, expected in coordinates.items():
            values = [v for var, v in env.items() if var.name_hint == name]
            assert values == [expected], "changed original occurrence coordinates"
        start, completion = (self.payload_count, False), (self.payload_count, True)
        self.edges.add((start, completion))
        if pipe in self.latest_payload:
            previous = self.latest_payload[pipe]
            self.edges.update([((previous, False), start), ((previous, True), completion)])
        if pipe in self.gate:
            self.edges.add((self.gate[pipe], start))
        for source in self.pending.pop(pipe, []):
            self.matched.add((source, self.payload_count))
        self.latest_payload[pipe] = self.payload_count
        self.payload_count += 1

    def execute(self, stmt, env):
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                yielded = self.execute(child, env)
                if yielded is not None:
                    return yielded
        elif isinstance(stmt, ir.AssignStmt):
            self.assign(stmt, env)
        elif isinstance(stmt, ir.ForStmt):
            carried = tuple(value(arg.initValue, env) for arg in stmt.iter_args)
            start, stop, step = value(stmt.start, env), value(stmt.stop, env), value(stmt.step, env)
            assert isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)
            for index in range(start, stop, step):
                local = dict(env)
                local[stmt.loop_var] = index
                local.update(zip(stmt.iter_args, carried))
                yielded = self.execute(stmt.body, local)
                carried = () if yielded is None else yielded
                assert len(carried) == len(stmt.iter_args), "loop carry arity mismatch"
            assert len(carried) == len(stmt.return_vars)
            env.update(zip(stmt.return_vars, carried))
        elif isinstance(stmt, ir.IfStmt):
            self.branches += 1
            arm = stmt.then_body if value(stmt.condition, env) else stmt.else_body
            yielded = self.execute(arm, dict(env)) if arm is not None else ()
            values = () if yielded is None else yielded
            assert len(values) == len(stmt.return_vars)
            env.update(zip(stmt.return_vars, values))
        elif isinstance(stmt, ir.YieldStmt):
            return tuple(value(x, env) for x in stmt.value)
        elif isinstance(stmt, ir.ReturnStmt):
            return ()
        elif isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
            assert isinstance(call, ir.Call)
            if call.op.name == ir.get_op("sync.wait").name:
                self.wait(call, env)
            else:
                self.payload(call, env)
        else:
            raise AssertionError(f"Unsupported logical oracle statement: {type(stmt)}")
        return None


def validate_logical(original, emitted, *, require_equality=True):
    _, calls = interpret(original)
    reference_graph = hardware_reference(original, calls)
    state = CommandExecution(calls, reference_graph[0])
    state.execute(emitted.body, {})
    assert state.payload_count == len(calls), "missing payload"
    assert not any(state.pending.values()), "wait has no intended consumer"
    original_allocations = []

    class Allocations(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            original_allocations.append(op)

    Allocations().visit_function(original)
    assert len(original_allocations) == len(state.actual_allocations)
    for before, after in zip(original_allocations, state.actual_allocations):
        same_payload(before, after)
    return check_graph(state, reference_graph, require_equality)


def check_graph(state, reference_graph, require_equality):
    payload_vertices, _, required, covers = reference_graph
    order = closure(state.vertices, state.edges)
    assert not any(a == b for a, b in order), "cyclic command graph"
    payload_set = set(payload_vertices)
    projected = {(a, b) for a, b in order if a in payload_set and b in payload_set}
    assert required <= projected, ("missing coverage", required - projected)
    assert state.matched == covers, (
        "wrong dynamic token matching",
        state.matched - covers,
        covers - state.matched,
    )
    assert len(state.waits) == len(covers), "duplicate logical demand waits"
    extra = projected - required
    if require_equality:
        assert not extra, ("extra payload order", extra)
    result: dict[str, Any] = {
        "payloads": state.payload_count,
        "records": len(state.records),
        "waits": len(state.waits),
        "same_pipe_waits": state.same_pipe_waits,
        "matched_demands": len(state.matched),
        "extra_event_pairs": len(extra),
        "extra_completion_start_pairs": sum(a[1] is True and b[1] is False for a, b in extra),
        "branches_evaluated": state.branches,
        "unused_records": len(set(state.records) - set(state.waits)),
        "max_wait_fanout": max((state.waits.count(r) for r in state.records), default=0),
        "acyclic": True,
    }
    return result
