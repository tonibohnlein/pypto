# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source-control census and ordered trace comparison, independent of family inventories.

This interpreter counts source evaluations, not optimized native instructions.
It does not replace the independent access/command graph oracle.
"""

from collections import Counter

from pypto import ir

from .sync_logical_oracle import CommandExecution, value

COMMANDS = frozenset(
    ir.get_op(name).name for name in ("system.set_flag", "system.wait_flag", "system.bar_m", "system.bar_v")
)


def statements(statement):
    """Visit every structured statement occurrence, including both alternatives."""
    yield statement
    if isinstance(statement, ir.SeqStmts):
        for child in statement.stmts:
            yield from statements(child)
    elif isinstance(statement, ir.IfStmt):
        yield from statements(statement.then_body)
        if statement.else_body is not None:
            yield from statements(statement.else_body)
    elif isinstance(statement, ir.ForStmt):
        yield from statements(statement.body)


def expression_nodes(expression):
    """Count expression occurrences rather than confusing them with shared nodes."""
    yield expression
    if isinstance(expression, ir.BinaryExpr):
        yield from expression_nodes(expression.left)
        yield from expression_nodes(expression.right)
    elif isinstance(expression, ir.UnaryExpr):
        yield from expression_nodes(expression.operand)


def static_control(function):
    """Return static branch and physical-command counts for the actual function."""
    counts = Counter()
    guards = set()
    for statement in statements(function.body):
        if isinstance(statement, ir.IfStmt):
            counts["if_sites"] += 1
            size = sum(1 for _ in expression_nodes(statement.condition))
            counts["guard_expression_visits"] += size
            counts["largest_guard_nodes"] = max(counts["largest_guard_nodes"], size)
            guards.add(ir.structural_hash(statement.condition, False))
        elif isinstance(statement, ir.AssignStmt) and isinstance(statement.var.type, ir.ScalarType):
            # Count all scalar work, including original control: compare with
            # the same unsynchronized input. Never hide predicates in temps.
            counts["scalar_definitions"] += 1
            counts["scalar_expression_visits"] += sum(1 for _ in expression_nodes(statement.value))
        elif isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call):
            if statement.expr.op.name in COMMANDS:
                counts[statement.expr.op.name] += 1
    counts["distinct_guard_expressions"] = len(guards)
    return counts


class ControlTrace(CommandExecution):
    """Execute original scalar control and record commands at exact payload cuts.

    Buffer definitions are not evaluated here: the separate access oracle owns
    them. An unexpected scalar expression fails in value(), not a guessed value.
    """

    def __init__(self):
        super().__init__([], [])
        self.trace = []
        self.counts = Counter()

    def assign(self, stmt, env):
        if isinstance(stmt.var.type, ir.ScalarType):
            self.counts["scalar_evaluations"] += 1
            self.counts["scalar_value_nodes"] += sum(1 for _ in expression_nodes(stmt.value))
            env[stmt.var] = value(stmt.value, env)

    def execute(self, stmt, env):
        if isinstance(stmt, ir.IfStmt):
            self.counts["conditions"] += 1
            self.counts["inactive_conditions"] += not value(stmt.condition, env)
            self.counts["condition_nodes"] += sum(1 for _ in expression_nodes(stmt.condition))
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
            call = stmt.expr
            if call.op.name in COMMANDS:
                self.trace.append((call.op.name, tuple(sorted(call.kwargs.items())), self.payload_count))
                self.counts[call.op.name] += 1
            else:
                self.trace.append((call.op.name, self.payload_count))
                self.payload_count += 1
            return None
        return super().execute(stmt, env)


def trace_control(function, parameters=None, block=None):
    """Bind the function's own original parameters and execute one invocation."""
    environment = {function.params[i]: n for i, n in (parameters or {}).items()}
    if block is not None:
        environment["__block_index__"] = block
    state = ControlTrace()
    state.execute(function.body, environment)
    return state
