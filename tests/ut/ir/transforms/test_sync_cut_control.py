# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generic execution-domain and first-use control-value regressions."""

import pytest
from pypto import ir, passes, testing

from .sync_control_metrics import statements
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_parameterized_token_sync import apply
from .test_sync_command_control import SPAN, command, evaluate, guarded, seq

pytestmark = pytest.mark.usefixtures("ascend_backend")


def variable(name="block"):
    return ir.Var(name, ir.ScalarType(ir.DataType.INDEX), SPAN)


def ci(n):
    return ir.ConstInt(n, ir.DataType.INDEX, SPAN)


def owner(body):
    return ir.Function("control", [], [], body, SPAN, ir.FunctionType.AIC)


@pytest.mark.parametrize("upper", [7, 7000, 7000000])
def test_absent_gap_is_not_a_present_no_command_gap(upper):
    block = variable()
    body = seq(command())
    cases: list[tuple[int, int, ir.Stmt]] = [(0, 1, body), (5, upper, body)]
    absent = testing.emit_sync_command_cases(block, [(0, 1), (5, upper)], cases)
    possible = testing.emit_sync_command_cases(block, [(0, upper)], cases)
    assert sum(isinstance(s, ir.IfStmt) for s in statements(absent)) == 0
    assert sum(isinstance(s, ir.IfStmt) for s in statements(possible)) == 2
    for b in [0, 1, 2, 4, 5, upper]:
        original = seq(
            guarded(ir.Lt(block, ci(2), ir.DataType.BOOL, SPAN), command()),
            guarded(ir.Ge(block, ci(5), ir.DataType.BOOL, SPAN), command()),
        )
        assert evaluate(original, {block: b}).trace == evaluate(possible, {block: b}).trace
        if b < 2 or b >= 5:
            assert evaluate(original, {block: b}).trace == evaluate(absent, {block: b}).trace


def test_unknown_domain_is_full_and_different_commands_do_not_merge():
    block = variable()
    cases: list[tuple[int, int, ir.Stmt]] = [(0, 1, seq(command(0))), (5, 7, seq(command(1)))]
    result = testing.emit_sync_command_cases(block, [(0, 1), (5, 7)], cases)
    assert sum(isinstance(s, ir.IfStmt) for s in statements(result)) == 2
    assert evaluate(result, {block: 0}).trace[0][1] != evaluate(result, {block: 7}).trace[0][1]
    assert evaluate(testing.emit_sync_command_cases(block, [(0, 7)], cases), {block: 3}).trace == []
    assert evaluate(testing.emit_sync_command_cases(block, [], cases), {}).trace == []
    with pytest.raises(ValueError, match="domain"):
        testing.emit_sync_command_cases(block, [(5, 7), (0, 1)], cases)
    with pytest.raises(ValueError, match="cases"):
        testing.emit_sync_command_cases(block, [(0, 7)], [cases[1], cases[0]])


def test_shared_quotient_is_defined_at_first_use_and_not_hoisted_from_guard():
    index = variable("i")
    active = ir.Var("active", ir.ScalarType(ir.DataType.BOOL), SPAN)
    quotient = ir.FloorDiv(index, ci(4), ir.DataType.INDEX, SPAN)
    inner = seq(
        guarded(ir.Ge(quotient, ci(1), ir.DataType.BOOL, SPAN), command(0)),
        guarded(ir.Eq(quotient, ci(0), ir.DataType.BOOL, SPAN), command(1)),
    )
    original = seq(command(2), guarded(active, inner))
    output = testing.materialize_sync_command_values(owner(original), original)
    definitions = [s for s in statements(output) if isinstance(s, ir.AssignStmt)]
    assert len(definitions) == 1
    # The undefined i must not be evaluated on the skipped branch.
    assert evaluate(output, {active: False}).trace == evaluate(original, {active: False}).trace
    for i in [0, 3, 4, 8]:
        assert (
            evaluate(output, {active: True, index: i}).trace
            == evaluate(original, {active: True, index: i}).trace
        )


def test_value_sharing_keeps_ssa_identity_and_reset_scopes():
    a, b = variable("same"), variable("same")
    first = ir.Lt(a, ci(2), ir.DataType.BOOL, SPAN)
    second = ir.Lt(b, ci(2), ir.DataType.BOOL, SPAN)
    original = seq(guarded(first, command(0)), guarded(second, command(1)))
    output = testing.materialize_sync_command_values(owner(original), original)
    assert not any(isinstance(s, ir.AssignStmt) for s in statements(output))
    for av, bv in [(0, 3), (3, 0), (3, 3)]:
        assert evaluate(output, {a: av, b: bv}).trace == evaluate(original, {a: av, b: bv}).trace
    # A new visit gets a new execution of the definition; no retained value
    # from a previous invocation or branch is reused by the compiler.
    original = seq(guarded(first, command(0)), guarded(first, command(1)))
    output = testing.materialize_sync_command_values(owner(original), original)
    for av in [0, 3, 0]:
        assert evaluate(output, {a: av}).trace == evaluate(original, {a: av}).trace


def test_short_circuit_and_unknown_comparison_keep_original_control():
    a = variable()
    active = ir.Var("active", ir.ScalarType(ir.DataType.BOOL), SPAN)
    condition = ir.And(active, ir.Lt(a, ci(2), ir.DataType.BOOL, SPAN), ir.DataType.BOOL, SPAN)
    original = seq(guarded(condition, command()), guarded(condition, command(1)))
    output = testing.materialize_sync_command_values(owner(original), original)
    assert evaluate(output, {active: False}).trace == []
    expression = a
    for _ in range(520):
        expression = ir.Add(expression, ci(1), ir.DataType.INDEX, SPAN)
    deep = ir.Lt(expression, ci(999), ir.DataType.BOOL, SPAN)
    original = seq(guarded(deep, command()), guarded(deep, command(1)))
    output = testing.materialize_sync_command_values(owner(original), original)
    assert evaluate(output, {a: 0}).trace == evaluate(original, {a: 0}).trace


def test_control_names_are_reserved_and_deterministic():
    a = variable("sync_control_0")
    condition = ir.Lt(a, ci(2), ir.DataType.BOOL, SPAN)
    original = seq(guarded(condition, command()), guarded(condition, command(1)))
    outputs = [testing.materialize_sync_command_values(owner(original), original) for _ in range(3)]
    assert len({ir.python_print(x, format=False) for x in outputs}) == 1
    definition = next(s for s in statements(outputs[0]) if isinstance(s, ir.AssignStmt))
    assert definition.var.name_hint != a.name_hint


def test_production_control_is_deterministic_after_serialization():
    program = capture_qwen_projection("gate_up", layers=2, hidden=512, band=2)
    first = apply(passes.realize_sync, program)
    cloned = ir.deserialize(ir.serialize(program))
    assert isinstance(cloned, ir.Program)
    second = apply(passes.realize_sync, cloned)
    assert ir.python_print(first, format=False) == ir.python_print(second, format=False)
