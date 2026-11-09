# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent consuming-flag execution; never add reuse edges to make it pass.

Accesses are interpreted from the original payload IR, not from compiler cells
or demand families. Physical matching is read from actual commands. The graph
implements the draft's SET/WAIT/barrier table, including nonblocking SETs.
"""

from collections import defaultdict
from itertools import combinations

from pypto import ir

from .sync_logical_oracle import CommandExecution, same_payload
from .sync_matmul_oracle import interpret
from .test_finite_sync_analysis import closure, reference


class PhysicalExecution(CommandExecution):
    def __init__(self, calls, vertices, eligible):
        super().__init__(calls, vertices)
        self.eligible = set(eligible)
        self.commands = defaultdict(list)
        self.live = {}
        self.last_consumption = {}
        self.reuses = []
        self.handoffs = []
        self.ids = defaultdict(set)
        self.barriers = []

    def command(self, call):
        name = call.op.name
        is_set = name == ir.get_op("system.set_flag").name
        is_wait = name == ir.get_op("system.wait_flag").name
        if is_set or is_wait:
            assert not call.args, "prototype must bind literal IDs"
            src, dst = ir.PipeType(call.kwargs["src_pipe"]), ir.PipeType(call.kwargs["dst_pipe"])
            key = (src, dst, call.kwargs["event_id"])
            assert src != dst and key[2] in self.eligible, "ineligible directional ID"
            self.ids[src, dst].add(key[2])
            pipe = src if is_set else dst
        else:
            pipe = ir.PipeType.M if name == ir.get_op("system.bar_m").name else ir.PipeType.V
        node = ("physical", sum(map(len, self.commands.values())))
        self.vertices.add(node)
        if pipe in self.latest_payload:
            self.edges.add(((self.latest_payload[pipe], False), node))
        if pipe in self.gate:
            self.edges.add((self.gate[pipe], node))
        if not is_wait:
            for previous in self.commands[pipe]:
                self.edges.add((previous, node))
            if pipe in self.latest_payload:
                self.edges.add(((self.latest_payload[pipe], True), node))
        if is_set:
            assert key not in self.live, "publication overwrites an unconsumed flag"
            assert pipe in self.latest_payload, "publication has no source payload"
            self.live[key] = (node, self.latest_payload[pipe])
            if key in self.last_consumption:
                self.reuses.append((self.last_consumption[key], node))
            self.records.append(node)
        elif is_wait:
            assert key in self.live, "wait without a matching publication"
            publication, source = self.live.pop(key)
            self.edges.add((publication, node))
            self.last_consumption[key] = node
            self.pending.setdefault(pipe, []).append(source)
            self.waits.append(publication)
            self.handoffs.append((key[:2], publication, node))
        else:
            self.barriers.append(node)
        if not is_set:
            self.gate[pipe] = node
        self.commands[pipe].append(node)

    def execute(self, stmt, env):
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
            if stmt.expr.op.name in {
                ir.get_op(name).name
                for name in ["system.set_flag", "system.wait_flag", "system.bar_m", "system.bar_v"]
            }:
                self.command(stmt.expr)
                return None
        return super().execute(stmt, env)


def run_physical(original, emitted, eligible=range(8)):
    _, calls = interpret(original)
    effects = [(pipe, True, reads, writes) for _, _, pipe, reads, writes in calls]
    ref = reference(effects)
    state = PhysicalExecution(calls, ref[0], eligible)

    class CoreOwnership(ir.IRVisitor):
        def visit_call(self, op):
            call = op
            if call.op.name in {ir.get_op("system.set_flag").name, ir.get_op("system.wait_flag").name}:
                expected = "aiv" if original.func_type == ir.FunctionType.AIV else "aic"
                assert call.kwargs["core_type"] == expected, "flag has the wrong owning core"

    CoreOwnership().visit_function(emitted)
    state.execute(emitted.body, {})
    assert state.payload_count == len(calls), "missing payload"
    assert not state.live, "outstanding notifications at invocation exit"
    assert not any(state.pending.values()), "wait has no payload consumer"
    allocations = []

    class Allocations(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            allocations.append(op)

    Allocations().visit_function(original)
    assert len(allocations) == len(state.actual_allocations)
    for a, b in zip(allocations, state.actual_allocations):
        same_payload(a, b)
    order = closure(state.vertices, state.edges)
    assert not any(a == b for a, b in order), "cyclic physical command graph"
    assert all(pair in order for pair in state.reuses), "consumption does not causally precede republication"
    vertices = set(ref[0])
    projected = {(a, b) for a, b in order if a in vertices and b in vertices}
    assert ref[2] <= projected, ("missing physical coverage", ref[2] - projected)
    assert projected == ref[2], ("extra physical payload order", projected - ref[2])
    expected = {(a, b) for a, b in ref[3] if calls[a][2] != calls[b][2]}
    assert state.matched == expected, ("wrong handoff identities", state.matched ^ expected)
    assert len(state.waits) == len(expected), "duplicate physical handoff"
    return state, order, projected


def directional_widths(state, order):
    """Independent finite poset width by maximum bipartite matching (test only).

    Premises were checked by run_physical: acyclic, complete matching, fresh
    entry and closed notification exit. No schedule timestamps are used.
    """
    directions = defaultdict(list)
    for direction, publication, consumption in state.handoffs:
        directions[direction].append((publication, consumption))
    widths = {}
    for direction, handoffs in directions.items():
        edges = {
            i: [j for j, (s, _) in enumerate(handoffs) if (w, s) in order]
            for i, (_, w) in enumerate(handoffs)
        }
        matched = {}

        def augment(i, seen):
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in matched or augment(matched[j], seen):
                    matched[j] = i
                    return True
            return False

        for i in range(len(handoffs)):
            augment(i, set())
        widths[direction] = len(handoffs) - len(matched)
        # Tiny graphs additionally check width directly without matching.
        if len(handoffs) <= 12:
            brute = max(
                (
                    len(subset)
                    for n in range(len(handoffs) + 1)
                    for subset in combinations(range(len(handoffs)), n)
                    if all(j not in edges[i] and i not in edges[j] for i, j in combinations(subset, 2))
                ),
                default=0,
            )
            assert widths[direction] == brute
    return widths
