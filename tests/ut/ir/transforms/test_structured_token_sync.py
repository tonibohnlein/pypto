# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent original-frame checks for canonical structured token transport."""

import pytest
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.op import sync

from .sync_matmul_cases import capture, staged_fixture
from .test_autosync_switch import clone
from .test_token_sync_lowering import _apply, _check_actual_tokens
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("visits", [1, 2, 3])
def test_original_loop_frames_and_resets(count, visits):
    original = staged_fixture(count, visits=visits)
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


@pytest.mark.parametrize("start,step", [(0, 2), (5, 3), (3, 1)])
def test_original_nonunit_loop_coordinates(start, step):
    original = staged_fixture(3, visits=2, start=start, step=step)
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def test_split_k_uses_standalone_token_program():
    original = capture("ddr_split_k")
    logical = _apply(passes.insert_sync, original)
    logical = _apply(passes.optimize_sync, logical)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def test_structured_serialization_without_sidecar(tmp_path):
    original = staged_fixture(3, visits=2)
    logical = _apply(passes.insert_sync, original)
    path = tmp_path / "loop-tokens.msgpack"
    ir.serialize_to_file(logical, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    physical = _apply(passes.lower_sync, restored)
    _check_actual_tokens(original, restored, physical)


def test_actual_repeated_waits_are_distinct_consuming_handoffs():
    original = staged_fixture(3, visits=2)
    logical = _apply(passes.insert_sync, original)

    class RepeatWait(ir.IRMutator):
        done = False

        def visit_eval_stmt(self, op):
            if (
                not self.done
                and isinstance(op.expr, ir.Call)
                and op.expr.op.name == ir.get_op("sync.wait").name
            ):
                self.done = True
                return ir.SeqStmts([op, op], op.span)
            return op

    doubled = RepeatWait().visit_function(logical)
    physical = _apply(passes.lower_sync, doubled)
    _, _, state = _check_actual_tokens(original, doubled, physical)
    baseline = _apply(passes.lower_sync, logical)
    _, _, before = _check_actual_tokens(original, logical, baseline)
    assert len(state.records) == len(before.records) + 2


def test_trip_counts_do_not_expand_physical_or_token_schemas():
    sizes = []
    for count in [2, 1024, 1000000]:
        original = staged_fixture(count, visits=2)
        logical = _apply(passes.insert_sync, original)
        physical = _apply(passes.lower_sync, logical)
        token_text = ir.python_print(logical, format=False)
        flag_text = ir.python_print(physical, format=False)
        sizes.append(
            (token_text.count("pl.sync."), flag_text.count("set_flag("), flag_text.count("wait_flag("))
        )
    assert sizes[0] == sizes[1] == sizes[2]


def test_previous_record_yield_is_not_assumed_to_be_current():
    original = staged_fixture(3)
    logical = _apply(passes.insert_sync, original)

    class PreviousYield(ir.IRMutator):
        args = []

        def visit_for_stmt(self, op):
            self.args = op.iter_args
            return super().visit_for_stmt(op)

        def visit_yield_stmt(self, op):
            return ir.YieldStmt(list(self.args), op.span)

    invalid = PreviousYield().visit_function(logical)
    before = str(invalid)
    with pytest.raises(
        ValueError, match="structured token binding|unconditional current record|Cyclic token history"
    ):
        _apply(passes.lower_sync, invalid)
    assert str(invalid) == before


def test_noncompleted_entry_carry_remains_unavailable():
    original = staged_fixture(3, visits=2)
    logical = _apply(passes.insert_sync, original)
    assert isinstance(logical.body, ir.SeqStmts)
    loops = [s for s in logical.body.stmts if isinstance(s, ir.ForStmt)]
    incoming = loops[1].iter_args[0].initValue
    previous = loops[0].return_vars[0]

    class ChangeInit(ir.IRMutator):
        def visit_assign_stmt(self, op):
            return ir.AssignStmt(op.var, previous, op.span) if op.var.same_as(incoming) else op

    invalid = ChangeInit().visit_function(logical)
    before = str(invalid)
    with pytest.raises(ValueError, match="completed entry"):
        _apply(passes.lower_sync, invalid)
    assert str(invalid) == before


def test_zero_trip_export_is_completed_before_a_live_region():
    original = vec_fixture()
    logical = _apply(passes.insert_sync, original)
    span = original.span
    token_type = ir.SyncTokenType(ir.PipeType.MTE2)
    incoming = ir.Var("zero_initial", token_type, span)
    carry = ir.IterArg("zero_carry", token_type, incoming, span)
    recorded = ir.Var("zero_record", token_type, span)
    final = ir.Var("zero_result", token_type, span)
    iv = ir.Var("zero_iv", ir.ScalarType(ir.DataType.INDEX), span)
    zero = ir.ConstInt(0, ir.DataType.INDEX, span)
    one = ir.ConstInt(1, ir.DataType.INDEX, span)
    empty_loop = ir.ForStmt(iv, zero, zero, one, [], ir.YieldStmt([], span), [], span)
    token_loop = ir.ForStmt(
        iv,
        zero,
        zero,
        one,
        [carry],
        ir.SeqStmts(
            [ir.AssignStmt(recorded, sync.record(ir.PipeType.MTE2), span), ir.YieldStmt([recorded], span)],
            span,
        ),
        [final],
        span,
    )

    def inject(function, commands):
        assert isinstance(function.body, ir.SeqStmts)
        statements = list(function.body.stmts)
        index = next(i for i, s in enumerate(statements) if isinstance(s, ir.EvalStmt))
        return clone(function, body=ir.SeqStmts(statements[:index] + commands + statements[index:], span))

    original = inject(original, [empty_loop])
    logical = inject(
        logical,
        [
            ir.AssignStmt(incoming, sync.completed(ir.PipeType.MTE2), span),
            token_loop,
            ir.EvalStmt(sync.wait(final, ir.PipeType.MTE2), span),
        ],
    )
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def test_nested_sequences_do_not_expand_structured_transport():
    original = staged_fixture(3)
    logical = _apply(passes.insert_sync, original)
    body = logical.body
    for i in range(64):
        empty = ir.Var(f"empty_nested_{i}", ir.SyncTokenType(ir.PipeType.MTE2), logical.span)
        body = ir.SeqStmts(
            [ir.AssignStmt(empty, sync.completed(ir.PipeType.MTE2), logical.span), body], logical.span
        )
    logical = clone(logical, body=body)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def test_record_displaced_after_a_pending_wait_refuses():
    original = staged_fixture(3)
    logical = _apply(passes.insert_sync, original)

    class MoveRecord(ir.IRMutator):
        changed = False

        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            body = list(op.body.stmts)
            for i, stmt in enumerate(body[:-1]):
                if (
                    not self.changed
                    and isinstance(stmt, ir.AssignStmt)
                    and isinstance(stmt.value, ir.Call)
                    and stmt.value.op.name == ir.get_op("sync.record").name
                    and isinstance(body[i + 1], ir.IfStmt)
                ):
                    body[i], body[i + 1] = body[i + 1], body[i]
                    self.changed = True
                    break
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(body, op.span),
                op.return_vars,
                op.span,
            )

    mutation = MoveRecord()
    invalid = mutation.visit_function(logical)
    assert mutation.changed
    before = str(invalid)
    with pytest.raises(ValueError, match="direct payload cut"):
        _apply(passes.lower_sync, invalid)
    assert str(invalid) == before


@pytest.mark.ptoas_compile
def test_split_k_cpp(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    original = capture("ddr_split_k")
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)
    source, output = tmp_path / "split.pto", tmp_path / "split.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([physical], "Split", physical.span), emit_tile_addr=True, emit_source_loc=False
        )
    )
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    kernel_body = output.read_text().split("AICORE void ", 1)[1]
    assert "ptoas_auto_sync_tail(" not in kernel_body
    assert "set_flag(" in kernel_body and "wait_flag(" in kernel_body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
