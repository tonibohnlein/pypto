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

import pytest
from pypto import ir, passes, testing
from pypto.ir.op import sync

from .sync_matmul_oracle import hardware_reference, integer, interpret
from .test_collective_sync_realization import nested_collective, repeat_collective
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


def compare(logical):
    snapshot = ir.serialize(ir.Program([logical], "Before", logical.span))
    result = testing.inspect_scoped_token_families(logical)
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
            for digits in product(*(range(lo, hi) for lo, hi in family[key])):
                ordinal = 0
                for digit, (_, size, _, _) in zip(digits, frames, strict=True):
                    assert 0 <= digit < size
                    ordinal = ordinal * size + digit
                retained.add(ordinal)
            assert retained == selected
    assert expanded == state.uses, (expanded - state.uses, state.uses - expanded)
    return result, state


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


def _symbolic_start_guard_fixture(offset=2, foreign_origin=False, comparison=ir.Eq):
    """Author an additional token use on a symbolic-start frame, without analysis."""
    # The shared fixture imports Uses through its independent command checker.
    from .test_sync_case_only_schedule import case_only_fixture  # noqa: PLC0415

    _, logical = case_only_fixture(4)
    block = logical.body.stmts[0].var
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
            statements = list(op.body.stmts)
            at = next(
                i
                for i, stmt in enumerate(statements)
                if isinstance(stmt, ir.EvalStmt) and stmt.expr.op.name == ir.get_op("sync.wait").name
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
    options = {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, 0)}
    original = testing.inspect_scoped_token_families(logical, **options)
    digit = offset // 2
    selected = [f for f in original["families"] if f["target_domain"] == [(digit, digit + 1)]]
    assert len(selected) == 1
    assert selected[0]["count"] == 1
    assert selected[0]["source_domain"] == [(digit, digit + 1)]
    assert selected[0]["source"][1] == selected[0]["target"][1] == digit
    restored = ir.deserialize(ir.serialize(logical))
    reparsed = testing.inspect_scoped_token_families(restored, **options)

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
    options = {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, 0)}
    empty = testing.inspect_scoped_token_families(_symbolic_start_guard_fixture(8), **options)
    full = testing.inspect_scoped_token_families(
        _symbolic_start_guard_fixture(8, comparison=ir.Lt), **options
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
        def visit_if_stmt(self, op):
            if op.return_vars and isinstance(op.return_vars[0].type, ir.SyncTokenType):
                assert isinstance(op.condition, ir.Eq)
                guard = ir.Eq(
                    op.condition.left,
                    ir.ConstInt(selected, ir.DataType.INDEX, op.span),
                    ir.DataType.BOOL,
                    op.span,
                )
                return ir.IfStmt(guard, op.then_body, op.else_body, op.return_vars, op.span)
            return super().visit_if_stmt(op)

    compare(Select().visit_function(logical))


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
            statements = list(op.body.stmts)
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
