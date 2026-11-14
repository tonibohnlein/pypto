# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Actual scoped SSA uses versus compact handoff families, without demand sidecars."""

from collections import Counter
from itertools import product
from typing import cast

import pytest
from pypto import ir, passes, testing
from pypto.ir.op import sync

from .sync_matmul_oracle import hardware_reference, integer, interpret
from .test_collective_sync_realization import nested_collective, repeat_collective
from .test_nested_phase_sync import nested_matrix_fixture
from .test_nested_sync_repetition import resident_pipeline_tree
from .test_scoped_logical_sync import wrapped_collective
from .test_token_sync_lowering import FiniteTokenExecution, _apply

pytestmark = pytest.mark.usefixtures("ascend_backend")


class Uses(FiniteTokenExecution):
    def __init__(self, calls, vertices):
        super().__init__(calls, vertices)
        self.uses = Counter()

    def payload(self, call, env):
        pipe = self.calls[self.payload_count][2]
        self.uses.update((source, self.payload_count) for source in self.pending.get(pipe, []))
        super().payload(call, env)


def occurrence_indices(result, calls):
    sites = {}
    for r, region in enumerate(result["regions"]):
        for s, (first, continuation) in enumerate(region["sites"]):
            sites[first] = (r, s)
            if continuation is not None:
                sites[continuation] = (r, s)
    occurrences = {}
    for index, (call, env, _, _, _) in enumerate(calls):
        r, s = sites[call]
        visit = 0
        for frame, count, _, _ in result["regions"][r]["frames"]:
            start, step = integer(frame.start, env), integer(frame.step, env)
            digit = (integer(frame.loop_var, env) - start) // step
            assert 0 <= digit < count
            visit = visit * count + digit
        assert (r, visit, s) not in occurrences
        occurrences[r, visit, s] = index
    return occurrences


def compare(logical, *, prepare=False):
    snapshot = ir.serialize(ir.Program([logical], "Before", logical.span))
    result = testing.inspect_scoped_token_families(logical, prepare=prepare)
    assert snapshot == ir.serialize(ir.Program([logical], "Before", logical.span))
    original = result["function"]
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    state = Uses(calls, reference[0])
    state.execute(logical.body, {})
    assert state.payload_count == len(calls)
    assert not any(state.pending.values())
    occurrences = occurrence_indices(result, calls)
    expanded = Counter()
    for family in result["families"]:
        sr, sv, ss = family["source"]
        tr, tv, ts = family["target"]
        count = 0
        selected_source, selected_target = set(), set()
        for digits in product(*(range(axis[1]) for axis in family["axes"])):
            source = sv + sum(digit * axis[2] for digit, axis in zip(digits, family["axes"]))
            target = tv + sum(digit * axis[3] for digit, axis in zip(digits, family["axes"]))
            selected_source.add(source)
            selected_target.add(target)
            expanded[occurrences[sr, source, ss], occurrences[tr, target, ts]] += 1
            count += 1
        assert count == family["count"]
        for key, region, selected in [
            ("source_domain", sr, selected_source),
            ("target_domain", tr, selected_target),
        ]:
            retained = set()
            frames = result["regions"][region]["frames"]
            steps = family[key.replace("domain", "steps")] or [1] * len(family[key])
            for digits in product(
                *(range(lo, hi, step) for (lo, hi), step in zip(family[key], steps, strict=True))
            ):
                ordinal = 0
                for digit, (_, size, _, _) in zip(digits, frames, strict=True):
                    assert 0 <= digit < size
                    ordinal = ordinal * size + digit
                retained.add(ordinal)
            assert retained == selected
    assert expanded == state.uses, (expanded - state.uses, state.uses - expanded)
    return result, state


@pytest.mark.parametrize("count,visits", [(0, 2), (1, 0), (1, 1), (1, 2), (2, 2), (3, 3), (5, 2)])
def test_phase_capture_actual_token_uses(count, visits):
    original = nested_matrix_fixture(count, visits=visits, tile_columns=2)
    logical = testing.analyze_placed_sync(original, True).realize()["function"]
    compare(logical)
    compare(ir.deserialize(ir.serialize(logical)))


@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_collective_actual_uses(rows, visits):
    compare(_apply(passes.insert_sync, repeat_collective(rows, visits)))


@pytest.mark.parametrize("counts", [(0, 2), (2, 0), (1, 1), (2, 3), (2, 2, 2)])
def test_nested_resets(counts):
    compare(_apply(passes.insert_sync, nested_collective(3, counts)))


@pytest.mark.parametrize("inner,outer,replay", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (2, 2, 2)])
def test_resident_tokens(inner, outer, replay):
    compare(_apply(passes.insert_sync, resident_pipeline_tree(inner, outer, replay)))


@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_original_nonunit_frames_and_interior_capture(visits):
    compare(_apply(passes.insert_sync, wrapped_collective(visits)))


def test_serialized_actual_tokens(tmp_path):
    logical = _apply(passes.insert_sync, wrapped_collective(3))
    path = tmp_path / "scoped.msgpack"
    ir.serialize_to_file(logical, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    compare(restored)


def _symbolic_start_guard_fixture(
    offset=2, foreign_origin: bool | str = False, comparison: type[ir.Eq] | type[ir.Lt] = ir.Eq
):
    """Author an additional token use on a symbolic-start frame, without analysis."""
    # The shared fixture imports Uses through its independent command checker.
    from .test_sync_case_only_schedule import case_only_fixture  # noqa: PLC0415

    _, logical = case_only_fixture(4)
    block = cast(ir.AssignStmt, cast(ir.SeqStmts, logical.body).stmts[0]).var
    span = logical.span
    other = ir.Var(block.name_hint, block.type, span)

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    class SymbolicStart(ir.IRMutator):
        def visit_assign_stmt(self, op):
            if foreign_origin == "binding" and op.var.same_as(block):
                # A valid, same-named alias is still a different SSA binding.
                return ir.SeqStmts([op, ir.AssignStmt(other, block, span)], span)
            return op

        def visit_for_stmt(self, op):
            if op.loop_var.name_hint != "i":
                return op
            start = ir.Add(block, ci(5), ir.DataType.INDEX, span)
            # The foreign origin has different arithmetic, not just a different
            # Python wrapper for the same underlying expression.
            origin = start
            if foreign_origin == "binding":
                origin = ir.Add(other, ci(5), ir.DataType.INDEX, span)
            elif foreign_origin:
                origin = ir.Add(block, ci(6), ir.DataType.INDEX, span)
            threshold = origin if offset == 0 else ir.Add(origin, ci(offset), ir.DataType.INDEX, span)
            statements = list(cast(ir.SeqStmts, op.body).stmts)
            at = next(
                i
                for i, stmt in enumerate(statements)
                if isinstance(stmt, ir.EvalStmt)
                and isinstance(stmt.expr, ir.Call)
                and stmt.expr.op.name == ir.get_op("sync.wait").name
            )
            statements.insert(
                at + 1,
                ir.IfStmt(
                    comparison(op.loop_var, threshold, ir.DataType.BOOL, span), statements[at], None, [], span
                ),
            )
            return ir.ForStmt(
                op.loop_var,
                start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(statements, span),
                op.return_vars,
                span,
            )

    return SymbolicStart().visit_function(logical)


@pytest.mark.parametrize("offset", [0, 2, 4])
def test_symbolic_start_actual_token_guard_roundtrip(offset):
    logical = _symbolic_start_guard_fixture(offset)
    original = testing.inspect_scoped_token_families(
        logical, parameter_bounds={4: (0, 0)}, block_bounds=(0, 0)
    )
    digit = offset // 2
    selected = [f for f in original["families"] if f["target_domain"] == [(digit, digit + 1)]]
    assert len(selected) == 1
    assert selected[0]["count"] == 1
    assert selected[0]["source_domain"] == [(digit, digit + 1)]
    assert selected[0]["source"][1] == selected[0]["target"][1] == digit
    restored = ir.deserialize(ir.serialize(logical))
    assert isinstance(restored, ir.Function)
    reparsed = testing.inspect_scoped_token_families(
        restored, parameter_bounds={4: (0, 0)}, block_bounds=(0, 0)
    )

    # Serialization may rebuild expression nodes. It must preserve the original
    # SSA bindings and every actual use, including the explicitly added wait.
    def inventory(result):
        return [
            {**family, "axes": [tuple(axis[1:]) for axis in family["axes"]]} for family in result["families"]
        ]

    assert inventory(reparsed) == inventory(original)


@pytest.mark.parametrize("offset,foreign", [(1, False), (10, False), (2, True), (2, "binding")])
def test_symbolic_start_token_guard_requires_original_lattice(offset, foreign):
    logical = _symbolic_start_guard_fixture(offset, foreign)
    before = ir.serialize(logical)
    with pytest.raises(ValueError, match="symbolic guard"):
        testing.inspect_scoped_token_families(logical, parameter_bounds={4: (0, 0)}, block_bounds=(0, 0))
    assert ir.serialize(logical) == before


def test_symbolic_start_one_past_guard_is_half_open():
    empty = testing.inspect_scoped_token_families(
        _symbolic_start_guard_fixture(8), parameter_bounds={4: (0, 0)}, block_bounds=(0, 0)
    )
    full = testing.inspect_scoped_token_families(
        _symbolic_start_guard_fixture(8, comparison=ir.Lt), parameter_bounds={4: (0, 0)}, block_bounds=(0, 0)
    )
    assert len(full["families"]) == len(empty["families"]) + 1
    assert sum(f["count"] for f in full["families"]) == sum(f["count"] for f in empty["families"]) + 4
    assert all(f["target_domain"] == [(0, 4)] for f in full["families"])


def test_family_size_independent_of_trip_counts():
    sizes = []
    for rows, counts in [(3, (3, 3)), (1000, (3, 3)), (1000000, (3, 3)), (3, (1000, 3)), (3, (3, 1000000))]:
        logical = _apply(passes.insert_sync, nested_collective(rows, counts))
        result = testing.inspect_scoped_token_families(logical)
        sizes.append((len(result["families"]), sum(len(f["axes"]) for f in result["families"])))
    assert len(set(sizes)) == 1


def test_duplicate_waits_preserve_each_actual_use():
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    class Duplicate(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                return ir.SeqStmts([op, op], op.span)
            return op

    before, state = compare(logical)
    after, edited = compare(Duplicate().visit_function(logical))
    assert len(after["families"]) == 2 * len(before["families"])
    assert edited.uses == Counter({pair: 2 * count for pair, count in state.uses.items()})


@pytest.mark.parametrize("selected", [0, 2, 4, 5])
def test_actual_capture_coordinate_controls_family(selected):
    # This intentionally edits the selected order. A demand sidecar would retain
    # row 1 and fail this comparison; an absent capture yields completed.
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    class Select(ir.IRMutator):
        changed = 0

        def visit_if_stmt(self, op):
            if (
                op.return_vars
                and isinstance(op.return_vars[0].type, ir.SyncTokenType)
                and isinstance(op.condition, ir.Eq)
            ):
                self.changed += 1
                guard = ir.Eq(
                    op.condition.left,
                    ir.ConstInt(selected, ir.DataType.INDEX, op.span),
                    ir.DataType.BOOL,
                    op.span,
                )
                return ir.IfStmt(guard, op.then_body, op.else_body, op.return_vars, op.span)
            return super().visit_if_stmt(op)

    edit = Select()
    changed = edit.visit_function(logical)
    assert edit.changed
    compare(changed)


@pytest.mark.parametrize("mode", ["unguarded", "first_only", "last_only"])
def test_history_activity_is_exact_not_assumed_uniform(mode):
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    class Guard(ir.IRMutator):
        def visit_if_stmt(self, op):
            if not op.return_vars and isinstance(op.condition, ir.And):
                assert isinstance(op.condition.left, ir.Gt)
                outer = op.condition.left.left
                row = op.condition.right
                if mode == "unguarded":
                    guard = row
                else:
                    guard = ir.And(
                        ir.Eq(
                            outer,
                            ir.ConstInt(1 if mode == "first_only" else 5, ir.DataType.INDEX, op.span),
                            ir.DataType.BOOL,
                            op.span,
                        ),
                        row,
                        ir.DataType.BOOL,
                        op.span,
                    )
                return ir.IfStmt(guard, op.then_body, op.else_body, op.return_vars, op.span)
            return super().visit_if_stmt(op)

    result, _ = compare(Guard().visit_function(logical))
    # First-visit carried facts are completed; they cannot certify a seam.
    if mode == "first_only":
        assert len(result["families"]) == 2


@pytest.mark.parametrize("mode", ["wrong_carry", "nonrectangular", "nonpersistent_capture"])
def test_unsupported_equations_refuse_without_mutation(mode):
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    carries = []

    def collect(stmt):
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                collect(child)
        elif isinstance(stmt, ir.ForStmt):
            carries.extend(stmt.iter_args)
            collect(stmt.body)

    collect(logical.body)

    class Invalid(ir.IRMutator):
        changed = False

        def visit_if_stmt(self, op):
            if mode == "nonrectangular" and not op.return_vars and isinstance(op.condition, ir.And):
                self.changed = True
                guard = ir.Or(op.condition.left, op.condition.right, ir.DataType.BOOL, op.span)
                return ir.IfStmt(guard, op.then_body, op.else_body, op.return_vars, op.span)
            if op.return_vars and isinstance(op.return_vars[0].type, ir.SyncTokenType):
                if mode == "wrong_carry":
                    assert isinstance(op.else_body, ir.YieldStmt)
                    previous = op.else_body.value[0]
                    previous_type = previous.type
                    assert isinstance(previous_type, ir.SyncTokenType)
                    other = next(
                        c
                        for c in carries
                        if isinstance(c.type, ir.SyncTokenType)
                        and c.type.pipe == previous_type.pipe
                        and not c.same_as(previous)
                    )
                    self.changed = True
                    return ir.IfStmt(
                        op.condition, op.then_body, ir.YieldStmt([other], op.span), op.return_vars, op.span
                    )
                if mode == "nonpersistent_capture":
                    # The false arm no longer preserves the selected visit.
                    # This describes a different token, not an own-carry capture.
                    assert isinstance(op.else_body, ir.YieldStmt)
                    self.changed = True
                    return ir.IfStmt(op.condition, op.then_body, op.then_body, op.return_vars, op.span)
            return super().visit_if_stmt(op)

    edit = Invalid()
    invalid = edit.visit_function(logical)
    assert edit.changed
    before = ir.serialize(ir.Program([invalid], "Invalid", invalid.span))
    with pytest.raises(ValueError, match="capture|guard|coordinate"):
        testing.inspect_scoped_token_families(invalid)
    assert before == ir.serialize(ir.Program([invalid], "Invalid", invalid.span))


@pytest.mark.parametrize("mode", ["record_cut", "wait_cut", "intermediate_capture"])
def test_unsupported_cuts(mode):
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    class Invalid(ir.IRMutator):
        done = False

        def scalar(self, span):
            var = ir.Var("intervening", ir.ScalarType(ir.DataType.INDEX), span)
            return ir.AssignStmt(var, ir.ConstInt(0, ir.DataType.INDEX, span), span)

        def visit_assign_stmt(self, op):
            if mode == "record_cut" and not self.done and isinstance(op.value, ir.Call):
                if op.value.op.name == ir.get_op("sync.record").name:
                    self.done = True
                    return ir.SeqStmts([self.scalar(op.span), op], op.span)
            return op

        def visit_eval_stmt(self, op):
            if mode == "wait_cut" and not self.done and isinstance(op.expr, ir.Call):
                if op.expr.op.name == ir.get_op("sync.wait").name:
                    self.done = True
                    return ir.SeqStmts([op, self.scalar(op.span)], op.span)
            return op

        def visit_if_stmt(self, op):
            if mode == "intermediate_capture" and op.return_vars and not self.done:
                if isinstance(op.return_vars[0].type, ir.SyncTokenType):
                    assert isinstance(op.else_body, ir.YieldStmt)
                    self.done = True
                    # Feeding the captured value back through another carry
                    # would require interpreting its piecewise intermediate state.

                    wait = ir.EvalStmt(sync.wait(op.return_vars[0], ir.PipeType.MTE2), op.span)
                    return ir.SeqStmts([op, wait], op.span)
            return super().visit_if_stmt(op)

    edit = Invalid()
    invalid = edit.visit_function(logical)
    assert edit.done
    with pytest.raises(ValueError, match="cut|wait|Wait|capture|Scalar"):
        testing.inspect_scoped_token_families(invalid)


def test_cyclic_outer_history_is_explicitly_unavailable():
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    class Cycle(ir.IRMutator):
        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            statements = list(cast(ir.SeqStmts, op.body).stmts)
            statements[-1] = ir.YieldStmt(list(op.iter_args), op.span)
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(statements, op.span),
                op.return_vars,
                op.span,
            )

    with pytest.raises(ValueError, match="Cyclic scoped token equation"):
        testing.inspect_scoped_token_families(Cycle().visit_function(logical))


@pytest.mark.parametrize("mode", ["foreign_origin", "zero_modulus", "missing_consumer_residue"])
def test_phase_capture_requires_exact_original_lattice(mode):
    logical = testing.analyze_placed_sync(nested_matrix_fixture(5, visits=2, tile_columns=2), True).realize()[
        "function"
    ]

    class Change(ir.IRMutator):
        changed = False

        def visit_if_stmt(self, op):
            capture = bool(op.return_vars)
            if capture == (mode == "missing_consumer_residue"):
                return super().visit_if_stmt(op)

            def alter(expr):
                if isinstance(expr, ir.And):
                    left, right = alter(expr.left), alter(expr.right)
                    if left is None:
                        return right
                    if right is None:
                        return left
                    return ir.And(left, right, ir.DataType.BOOL, expr.span)
                if not isinstance(expr, ir.Eq) or not isinstance(expr.left, ir.FloorMod):
                    return expr
                self.changed = True
                if mode == "missing_consumer_residue":
                    return None
                mod = expr.left
                operand, divisor = mod.left, mod.right
                assert isinstance(operand, ir.Sub)
                if mode == "foreign_origin":
                    operand = ir.Sub(
                        operand.left,
                        ir.Add(
                            operand.right,
                            ir.ConstInt(1, ir.DataType.INDEX, expr.span),
                            ir.DataType.INDEX,
                            expr.span,
                        ),
                        ir.DataType.INDEX,
                        expr.span,
                    )
                else:
                    divisor = ir.ConstInt(0, ir.DataType.INDEX, expr.span)
                return ir.Eq(
                    ir.FloorMod(operand, divisor, ir.DataType.INDEX, expr.span),
                    expr.right,
                    ir.DataType.BOOL,
                    expr.span,
                )

            condition = alter(op.condition)
            if condition is None:
                return op.then_body
            return ir.IfStmt(condition, op.then_body, op.else_body, op.return_vars, op.span)

    edit = Change()
    changed = edit.visit_function(logical)
    assert edit.changed
    before = ir.serialize(changed)
    with pytest.raises(ValueError, match="residue|guard|latest-source|coordinate"):
        testing.inspect_scoped_token_families(changed)
    assert ir.serialize(changed) == before


def test_capture_followed_by_second_same_frame_history_requires_proof():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(5, visits=2, tile_columns=2), True).realize()[
        "function"
    ]

    class ExtraHistory(ir.IRMutator):
        changed = False

        def visit_for_stmt(self, op):
            if self.changed or not isinstance(op.body, ir.SeqStmts):
                return super().visit_for_stmt(op)
            captures = [
                stmt.return_vars[0]
                for stmt in cast(ir.SeqStmts, op.body).stmts
                if isinstance(stmt, ir.IfStmt) and stmt.return_vars
            ]
            yielded = cast(ir.SeqStmts, op.body).stmts[-1]
            assert isinstance(yielded, ir.YieldStmt)

            class WaitArguments(ir.IRVisitor):
                def __init__(self):
                    super().__init__()
                    self.values = []

                def visit_eval_stmt(self, op):
                    statement = op
                    call = statement.expr
                    if isinstance(call, ir.Call) and call.op.name == ir.get_op("sync.wait").name:
                        self.values.append(call.args[0])
                    super().visit_eval_stmt(statement)

            arguments = WaitArguments()
            arguments.visit_stmt(op.body)
            candidates = [
                i
                for i, expr in enumerate(yielded.value)
                if any(expr.same_as(capture) for capture in captures)
                and any(op.iter_args[i].same_as(value) for value in arguments.values)
            ]
            if not candidates:
                return super().visit_for_stmt(op)
            previous = op.iter_args[candidates[0]]
            older = ir.IterArg("test_older", previous.type, previous.initValue, op.span)
            result = ir.Var("test_older_result", previous.type, op.span)

            class ReplaceWait(ir.IRMutator):
                count = 0

                def visit_eval_stmt(self, op):
                    statement = op
                    call = statement.expr
                    if (
                        isinstance(call, ir.Call)
                        and call.op.name == ir.get_op("sync.wait").name
                        and call.args[0].same_as(previous)
                    ):
                        self.count += 1
                        return ir.EvalStmt(
                            ir.Call(call.op, [older], dict(call.kwargs), None, call.type, call.span),
                            statement.span,
                        )
                    return statement

            replace = ReplaceWait()
            body = replace.visit_stmt(op.body)
            assert replace.count
            assert isinstance(body, ir.SeqStmts)
            body = ir.SeqStmts([*body.stmts[:-1], ir.YieldStmt([*yielded.value, previous], op.span)], op.span)
            self.changed = True
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                [*op.iter_args, older],
                body,
                [*op.return_vars, result],
                op.span,
            )

    edit = ExtraHistory()
    changed = edit.visit_function(logical)
    assert edit.changed
    before = ir.serialize(changed)
    with pytest.raises(ValueError, match="Composed captured history"):
        testing.inspect_scoped_token_families(changed)
    assert ir.serialize(changed) == before


@pytest.mark.parametrize("count,visits", [(1, 1), (2, 2), (3, 3), (5, 2), (8, 2)])
def test_prepared_phase_family_unions_preserve_actual_records(count, visits):
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(count, visits=visits, tile_columns=2), True
    ).realize()["function"]
    raw, before = compare(logical)
    prepared, after = compare(logical, prepare=True)
    assert before.uses == after.uses
    assert prepared["raw_family_count"] == len(raw["families"])
    assert len(prepared["families"]) <= len(raw["families"])
    assert [row["protocol_uses"] for row in raw["families"]] == [[i] for i in range(len(raw["families"]))]
    assert sorted(i for row in prepared["families"] for i in row["protocol_uses"]) == list(
        range(len(raw["families"]))
    )
    assert prepared["protocol_cuts"] == raw["protocol_cuts"]
    # compare expands each exact endpoint progression and every paired record;
    # neither an interval hull nor deduplication can pass these comparisons.
    if count >= 5:
        assert prepared["work"]["union_candidates"] > 0
        assert prepared["work"]["union_reuse_checks"] > 0


def test_prepared_family_union_preserves_duplicate_token_uses():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(5, visits=2, tile_columns=2), True).realize()[
        "function"
    ]

    class Duplicate(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                return ir.SeqStmts([op, op], op.span)
            return op

    _, before = compare(logical, prepare=True)
    _, after = compare(Duplicate().visit_function(logical), prepare=True)
    assert after.uses == Counter({pair: 2 * count for pair, count in before.uses.items()})


def test_retained_protocol_rejects_an_equivalent_but_different_token_owner():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(2, visits=1), True).realize()["function"]
    order = testing.analyze_scoped_token_order(logical)
    order.validate_token_owner(logical)
    replacement = ir.deserialize(ir.serialize(logical))
    assert isinstance(replacement, ir.Function)
    ir.assert_structural_equal(logical, replacement)
    with pytest.raises(ValueError, match="different token IR"):
        order.validate_token_owner(replacement)
