# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generic execution-domain and first-use control-value regressions."""

import random

import pytest
from pypto import ir, passes, testing

from .qualify_sync_accounting import family_key, guard_accounting
from .sync_control_metrics import scalar_work, statements
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_parameterized_token_sync import apply
from .test_sync_command_control import SPAN, command, evaluate, guarded, seq

pytestmark = pytest.mark.usefixtures("ascend_backend")


def test_accounting_singleton_axes_preserve_occurrence_coordinates():
    loop = ir.ForStmt(variable("i"), ci(0), ci(3), ci(1), [], seq(), [], SPAN)
    base = {"source": (0, 7, 1), "target": (1, 2, 0), "axes": []}
    singleton = {**base, "axes": [(loop, 1, 100, 20)]}
    assert family_key(base) == family_key(singleton)
    assert family_key(base) != family_key({**base, "target": (1, 3, 0)})
    assert family_key(base) != family_key({**base, "axes": [(loop, 2, 100, 20)]})


def test_accounting_tracks_condition_definitions_and_executed_roles():
    launch = variable("block")
    temporary = ir.Var("predicate", ir.ScalarType(ir.DataType.BOOL), SPAN)
    original = owner(seq())
    physical = owner(
        seq(
            ir.AssignStmt(launch, ir.Call(ir.get_op("buffer.get_block_idx"), [], {}, SPAN), SPAN),
            ir.AssignStmt(temporary, ir.Lt(launch, ci(2), ir.DataType.BOOL, SPAN), SPAN),
            guarded(temporary, command()),
        )
    )
    result = guard_accounting(original, physical, [({}, 0), ({}, 3)])
    assert result["static"] == {"launch": 1}
    assert result["executions"][0]["false"] == {"launch": 0}
    assert result["executions"][1]["false"] == {"launch": 1}


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
    assert sum(isinstance(s, ir.IfStmt) for s in statements(possible)) == 1
    for b in [0, 1, 2, 4, 5, upper]:
        original = seq(
            guarded(ir.Lt(block, ci(2), ir.DataType.BOOL, SPAN), command()),
            guarded(ir.Ge(block, ci(5), ir.DataType.BOOL, SPAN), command()),
        )
        assert evaluate(original, {block: b}).trace == evaluate(possible, {block: b}).trace
        if b < 2 or b >= 5:
            assert evaluate(original, {block: b}).trace == evaluate(absent, {block: b}).trace


def test_disjoint_complete_sequence_preserves_holes_and_multiplicity():
    block = variable()
    body = seq(command(0), command(1), command(0))
    cases: list[tuple[int, int, ir.Stmt]] = [(0, 2, body), (9, 11, body)]
    result = testing.emit_sync_command_cases(block, [(0, 11)], cases)
    assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == 3
    assert sum(isinstance(s, ir.IfStmt) for s in statements(result)) == 1
    for b in range(12):
        assert (
            evaluate(result, {block: b}).trace == evaluate(unfactored_cases(block, cases), {block: b}).trace
        )


@pytest.mark.parametrize("last", [(1, 2), (1, 1, 2), (2, 1)])
def test_residual_sequence_union_preserves_order_and_multiplicity(last):
    block = variable()
    cases: list[tuple[int, int, ir.Stmt]] = [
        (0, 1, seq(command(0), command(1), command(2), command(3))),
        (2, 5, seq(command(0), command(3))),
        (6, 7, seq(command(0), *(command(i) for i in last), command(3))),
    ]
    result = testing.emit_sync_command_cases(block, [(0, 7)], cases)
    if last == (1, 2):
        assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == 4
        assert sum(isinstance(s, ir.IfStmt) for s in statements(result)) == 1
    for b in range(8):
        assert (
            evaluate(result, {block: b}).trace == evaluate(unfactored_cases(block, cases), {block: b}).trace
        )


def test_residual_prefix_cannot_cross_uncovered_cut_domain():
    block = variable()
    a, b = seq(command(0), command(1)), seq(command(0))
    cases: list[tuple[int, int, ir.Stmt]] = [(1, 2, a), (3, 5, b), (6, 7, a)]
    result = testing.emit_sync_command_cases(block, [(0, 8)], cases)
    for value in range(9):
        assert (
            evaluate(result, {block: value}).trace
            == evaluate(unfactored_cases(block, cases), {block: value}).trace
        )


def test_disjoint_unknown_inner_domain_keeps_incumbent():
    block, inner = variable(), variable("inner")
    body = seq(guarded(ir.Eq(inner, ci(0), ir.DataType.BOOL, SPAN), command()))
    result = testing.emit_sync_command_cases(block, [(0, 7)], [(0, 1, body), (5, 7, body)])
    # The standalone helper has no original-loop domain proof. Identical
    # predicates alone cannot justify cancellation of their worst-case work.
    assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == 2
    for b in range(8):
        for i in range(2):
            expected = [command()] if (b < 2 or b >= 5) and i == 0 else []
            assert evaluate(result, {block: b, inner: i}).trace == evaluate(seq(*expected), {}).trace


def test_union_accounting_counts_eager_boolean_work():
    block = variable()
    expr = ir.Or(
        ir.Lt(block, ci(2), ir.DataType.BOOL, SPAN),
        ir.Ge(block, ci(5), ir.DataType.BOOL, SPAN),
        ir.DataType.BOOL,
        SPAN,
    )
    assert scalar_work(expr) == {"scalar_comparisons": 2, "scalar_boolean_ops": 1}
    original = seq(
        guarded(ir.Lt(block, ci(2), ir.DataType.BOOL, SPAN), command()),
        guarded(ir.Ge(block, ci(5), ir.DataType.BOOL, SPAN), command()),
    )
    shared = seq(guarded(expr, command()))
    for b in range(8):
        before, after = evaluate(original, {block: b}), evaluate(shared, {block: b})
        assert before.trace == after.trace

        def cost(state):
            return sum(state.counts[k] for k in ("conditions", "scalar_comparisons", "scalar_boolean_ops"))

        assert cost(before) == cost(after) == 4


def test_disjoint_interior_membership_preserves_all_four_boundaries():
    block = variable()
    body = seq(command(0), command(1), command(2), command(3))
    result = testing.emit_sync_command_cases(block, [(0, 9)], [(2, 3, body), (6, 7, body)])
    # Four comparisons, two eager ANDs, one OR and one decision cost eight,
    # equal to the two independent interval decisions in the incumbent.
    # Neither outside tail nor the interior hole may execute a command.
    assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == 4
    for b in range(10):
        assert evaluate(result, {block: b}).trace == evaluate(body if b in (2, 3, 6, 7) else seq(), {}).trace


def test_unknown_domain_is_full_and_different_commands_do_not_merge():
    block = variable()
    cases: list[tuple[int, int, ir.Stmt]] = [(0, 1, seq(command(0))), (5, 7, seq(command(1)))]
    result = testing.emit_sync_command_cases(block, [(0, 1), (5, 7)], cases)
    # The disjoint possible domain permits one binary decision, but the two
    # differently assigned commands must remain distinct alternatives.
    assert sum(isinstance(s, ir.IfStmt) for s in statements(result)) == 1
    assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == 2
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


def unfactored_cases(block, cases):
    return seq(
        *[
            guarded(
                ir.And(
                    ir.Ge(block, ci(lo), ir.DataType.BOOL, SPAN),
                    ir.Le(block, ci(hi), ir.DataType.BOOL, SPAN),
                    ir.DataType.BOOL,
                    SPAN,
                ),
                body,
            )
            for lo, hi, body in cases
        ]
    )


def test_case_tree_shares_ordered_prefix_and_suffix_without_recoloring():
    block = variable()
    cases: list[tuple[int, int, ir.Stmt]] = [
        (0, 3, seq(command(0), command(1), command(3))),
        (4, 7, seq(command(0), command(2), command(3))),
    ]
    result = testing.emit_sync_command_cases(block, [(0, 7)], cases)
    assert sum(isinstance(s, ir.IfStmt) for s in statements(result)) == 1
    assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == 4
    for b in range(8):
        assert (
            evaluate(result, {block: b}).trace == evaluate(unfactored_cases(block, cases), {block: b}).trace
        )


def test_case_tree_keeps_executing_empty_cases_and_command_multiplicity():
    block = variable()
    cases: list[tuple[int, int, ir.Stmt]] = [
        (2, 3, seq(command(0), command(0))),
        (6, 7, seq(command(0), command(1), command(0))),
    ]
    result = testing.emit_sync_command_cases(block, [(0, 9)], cases)
    for b in range(10):
        assert (
            evaluate(result, {block: b}).trace == evaluate(unfactored_cases(block, cases), {block: b}).trace
        )
    # A different order is not a shared subsequence that may be reordered.
    cases: list[tuple[int, int, ir.Stmt]] = [
        (0, 3, seq(command(0), command(1))),
        (4, 7, seq(command(1), command(0))),
    ]
    result = testing.emit_sync_command_cases(block, [(0, 7)], cases)
    for b in range(8):
        assert (
            evaluate(result, {block: b}).trace == evaluate(unfactored_cases(block, cases), {block: b}).trace
        )


def test_case_tree_preserves_guard_availability_and_original_ssa_identity():
    block, a, b = variable(), variable("same"), variable("same")
    active = ir.Var("active", ir.ScalarType(ir.DataType.BOOL), SPAN)
    common = guarded(active, guarded(ir.Lt(a, ci(2), ir.DataType.BOOL, SPAN), command(0)))
    other = guarded(active, guarded(ir.Lt(b, ci(2), ir.DataType.BOOL, SPAN), command(0)))
    cases: list[tuple[int, int, ir.Stmt]] = [
        (0, 3, seq(common, command(1), common)),
        (4, 7, seq(other, command(2), common)),
    ]
    result = testing.emit_sync_command_cases(block, [(0, 7)], cases)
    for visit in [0, 7, 0]:
        for env in [{active: False}, {active: True, a: 0, b: 3}, {active: True, a: 3, b: 0}]:
            values = {block: visit, **env}
            assert evaluate(result, values).trace == evaluate(unfactored_cases(block, cases), values).trace


@pytest.mark.parametrize("count", [8, 32, 128])
def test_case_tree_has_linear_static_size_without_launch_enumeration(count):
    block = variable()
    width = 1000000
    cases: list[tuple[int, int, ir.Stmt]] = [
        (i * width, (i + 1) * width - 1, seq(command(0), command(1 + i % 2), command(3)))
        for i in range(count)
    ]
    result = testing.emit_sync_command_cases(block, [(0, count * width - 1)], cases)
    assert sum(isinstance(s, ir.IfStmt) for s in statements(result)) == count - 1
    assert sum(isinstance(s, ir.EvalStmt) for s in statements(result)) == count + 2
    for b in [0, width - 1, width, count * width - 1]:
        assert (
            evaluate(result, {block: b}).trace == evaluate(unfactored_cases(block, cases), {block: b}).trace
        )


def test_generated_case_sequences_match_on_holes_empty_paths_and_resets():
    rng = random.Random(1821)
    block = variable()
    for _ in range(100):
        cases: list[tuple[int, int, ir.Stmt]] = []
        for i in range(8):
            if rng.randrange(3):
                cases.append(
                    (
                        3 * i,
                        3 * i + rng.randrange(3),
                        seq(*[command(rng.randrange(4)) for _ in range(rng.randrange(5))]),
                    )
                )
        possible = [(0, 23)] if rng.randrange(2) else [(0, 5), (9, 17), (21, 23)]
        result = testing.emit_sync_command_cases(block, possible, cases)
        reference = unfactored_cases(block, cases)
        for lo, hi in possible:
            for b in range(lo, hi + 1):
                assert evaluate(result, {block: b}).trace == evaluate(reference, {block: b}).trace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
