# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original-occurrence captures across rotating regions, without token sidecars."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.op import sync

from .sync_matmul_oracle import sequence_stmts, validate
from .test_buffered_sync_realization import local_sequence, resident_streamed_sequence
from .test_regional_slot_composition import regional_fixture
from .test_token_sync_lowering import _apply, _check_actual_tokens

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _capture(branch):
    return bool(branch.return_vars) and isinstance(branch.return_vars[0].type, ir.SyncTokenType)


def _capture_count(function):
    class Count(ir.IRVisitor):
        count = 0

        def visit_if_stmt(self, op):
            self.count += int(_capture(op))
            super().visit_if_stmt(op)

    visitor = Count()
    visitor.visit_function(function)
    return visitor.count


@pytest.mark.parametrize("counts", [(0, 0), (1, 1), (2, 0, 3), (6, 5), (4, 5, 2)])
@pytest.mark.parametrize("start,step", [(0, 1), (1, 2)])
def test_actual_interior_crossings_and_resets(counts, start, step):
    original = regional_fixture(
        counts, start=start, step=step, schedules=[[("load", 0, 0), ("rmw", 0, 0)]] * len(counts)
    )
    validate(original, testing.analyze_placed_sync(original, True))
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical, check_codegen=any(counts))
    if counts == (6, 5):
        assert _capture_count(logical) > 0


@pytest.mark.parametrize("count", [0, 1, 2, 5, 8])
def test_resident_prologue_stream_tail(count):
    original = resident_streamed_sequence(count)
    validate(original, testing.analyze_placed_sync(original, True))
    logical = _apply(passes.insert_sync, original)
    _check_actual_tokens(original, logical, _apply(passes.lower_sync, logical))


def test_capture_serialization_and_fanout(tmp_path):
    original = local_sequence((6, 5))
    logical = _apply(passes.insert_sync, original)
    loops = [s for s in sequence_stmts(logical.body) if isinstance(s, ir.ForStmt)]
    first = loops[0]
    statements = list(sequence_stmts(first.body))
    selected = next(s.return_vars[0] for s in statements if isinstance(s, ir.IfStmt) and _capture(s))
    last = statements[-1]
    assert isinstance(last, ir.YieldStmt)
    result = first.return_vars[next(i for i, v in enumerate(last.value) if v.same_as(selected))]

    class Fanout(ir.IRMutator):
        done = False

        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                if not self.done and op.expr.args[0].same_as(result):
                    self.done = True
                    return ir.SeqStmts([op, op], op.span)
            return op

    mutation = Fanout()
    duplicated = mutation.visit_function(logical)
    assert mutation.done
    path = tmp_path / "captures.msgpack"
    ir.serialize_to_file(duplicated, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    _, _, baseline = _check_actual_tokens(original, logical, _apply(passes.lower_sync, logical))
    _, _, candidate = _check_actual_tokens(original, restored, _apply(passes.lower_sync, restored))
    assert len(candidate.records) == len(baseline.records) + 1


@pytest.mark.parametrize("mutation", ["wrong_carry", "wrong_guard", "effectful_arm", "body_use"])
def test_unqualified_capture_refuses_without_mutation(mutation):
    logical = _apply(passes.insert_sync, local_sequence((6, 5)))

    class Invalid(ir.IRMutator):
        done = False
        payload = None
        carries = []

        def visit_for_stmt(self, op):
            self.carries = op.iter_args
            return super().visit_for_stmt(op)

        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("buffer.add").name:
                self.payload = op.expr
            return op

        def visit_if_stmt(self, op):
            if self.done or not _capture(op):
                return super().visit_if_stmt(op)
            self.done = True
            assert isinstance(op.then_body, ir.YieldStmt)
            assert isinstance(op.else_body, ir.YieldStmt)
            if mutation == "wrong_carry":
                previous = op.else_body.value[0]
                other = next(c for c in self.carries if c.type == previous.type and not c.same_as(previous))
                wrong = ir.YieldStmt([other], op.span)
                return ir.IfStmt(op.condition, op.then_body, wrong, op.return_vars, op.span)
            if mutation == "wrong_guard":
                assert isinstance(op.condition, ir.Eq)
                guard = ir.Gt(op.condition.left, op.condition.right, ir.DataType.BOOL, op.span)
                return ir.IfStmt(guard, op.then_body, op.else_body, op.return_vars, op.span)
            assert self.payload is not None
            call = self.payload
            payload = ir.EvalStmt(ir.Call(call.op, call.args, call.type, call.span), call.span)
            if mutation == "effectful_arm":
                arm = ir.SeqStmts([payload, op.then_body], op.span)
                return ir.IfStmt(op.condition, arm, op.else_body, op.return_vars, op.span)
            wait = ir.EvalStmt(sync.wait(op.return_vars[0], ir.PipeType.V), op.span)
            return ir.SeqStmts([op, wait, payload], op.span)

    edit = Invalid()
    invalid = edit.visit_function(logical)
    assert edit.done
    before = ir.python_print(invalid, format=False)
    with pytest.raises(ValueError, match="capture|Capture|conditional|branch|accumulation"):
        _apply(passes.lower_sync, invalid)
    assert ir.python_print(invalid, format=False) == before


@pytest.mark.parametrize("which,count", [(0, 0), (1, 0), (1, 1), (1, 2)])
def test_existing_capture_with_absent_source_or_target(which, count):
    original = local_sequence((6, 5))
    logical = _apply(passes.insert_sync, original)

    class EmptyOne(ir.IRMutator):
        index = 0

        def visit_for_stmt(self, op):
            index = self.index
            self.index += 1
            if index != which:
                return op
            return ir.ForStmt(
                op.loop_var,
                op.start,
                ir.ConstInt(count, ir.DataType.INDEX, op.span),
                op.step,
                op.iter_args,
                op.body,
                op.return_vars,
                op.span,
            )

    original = EmptyOne().visit_function(original)
    logical = EmptyOne().visit_function(logical)
    _check_actual_tokens(original, logical, _apply(passes.lower_sync, logical))


def test_fixed_slot_interface_does_not_shift_entire_trip_history():
    sizes = []
    for count in [6, 1023, 999999]:
        logical = _apply(passes.insert_sync, local_sequence((count, 5)))
        physical = _apply(passes.lower_sync, logical)
        captures = _capture_count(logical)
        loops = [s for s in sequence_stmts(logical.body) if isinstance(s, ir.ForStmt)]
        text = ir.python_print(physical, format=False)
        sizes.append((captures, sum(len(s.iter_args) for s in loops), text.count("set_flag(")))
    assert sizes[0][:2] == sizes[1][:2] == sizes[2][:2]
    # Phase-specialized output can use a smaller short-trip layout. Retained
    # captures stay constant, and steady output does not grow with trips.
    assert sizes[1][2] == sizes[2][2]
    assert sizes[0][2] <= sizes[1][2]
    assert sizes[0][0] > 0


@pytest.mark.ptoas_compile
def test_interior_crossing_cpp(tmp_path):
    original = local_sequence((6, 5))
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)
    source, output = tmp_path / "cross.pto", tmp_path / "cross.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([physical], "Cross", physical.span), emit_tile_addr=True, emit_source_loc=False
        )
    )
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    body = output.read_text().split("AICORE void ", 1)[1]
    assert "ptoas_auto_sync_tail(" not in body
    assert "set_flag(" in body and "wait_flag(" in body


def test_edited_capture_uses_actual_later_prefix():
    original = local_sequence((6, 5))
    logical = _apply(passes.insert_sync, original)

    class Later(ir.IRMutator):
        done = False

        def visit_if_stmt(self, op):
            if self.done or not _capture(op):
                return super().visit_if_stmt(op)
            self.done = True
            assert isinstance(op.condition, ir.Eq) and isinstance(op.condition.right, ir.ConstInt)
            coordinate = ir.ConstInt(op.condition.right.value + 1, ir.DataType.INDEX, op.span)
            condition = ir.Eq(op.condition.left, coordinate, ir.DataType.BOOL, op.span)
            return ir.IfStmt(condition, op.then_body, op.else_body, op.return_vars, op.span)

    edit = Later()
    stronger = edit.visit_function(logical)
    assert edit.done
    old, _, _ = _check_actual_tokens(original, logical, _apply(passes.lower_sync, logical))
    new, _, _ = _check_actual_tokens(original, stronger, _apply(passes.lower_sync, stronger))
    assert old < new, "lowering must preserve the manually strengthened prefix"


def test_shortened_capture_preserves_tokens_without_repairing_missing_demands():
    original = local_sequence((6, 5))
    logical = _apply(passes.insert_sync, original)

    class Shorten(ir.IRMutator):
        done = False

        def visit_for_stmt(self, op):
            if self.done:
                return op
            self.done = True
            return ir.ForStmt(
                op.loop_var,
                op.start,
                ir.ConstInt(2, ir.DataType.INDEX, op.span),
                op.step,
                op.iter_args,
                op.body,
                op.return_vars,
                op.span,
            )

    original = Shorten().visit_function(original)
    logical = Shorten().visit_function(logical)
    # The old slot-boundary captures are now absent. This edited logical program
    # is well-formed but no longer covers the changed access problem. LowerSync
    # must faithfully lower it, rather than silently repair it with reanalysis.
    actual, required, state = _check_actual_tokens(
        original, logical, _apply(passes.lower_sync, logical), require_access_coverage=False
    )
    assert required - actual
    assert not state.live
