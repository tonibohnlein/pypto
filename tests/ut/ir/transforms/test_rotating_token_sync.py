# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Actual RFC shift carries: independent accesses, token order and consuming flags."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import sequence_stmts, validate
from .test_autosync_switch import PYPTO_AUTOSYNC_ATTR
from .test_rotating_sync_analysis import rotating_fixture
from .test_streamed_slot_composition import SCHEDULE, streamed_fixture
from .test_token_sync_lowering import _apply, _check_actual_tokens

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
@pytest.mark.parametrize("start,step", [(0, 1), (1, 2), (5, 3)])
def test_shift_history_original_coordinates(count, start, step):
    original = streamed_fixture(count, start=start, step=step)
    if not count and start:
        # Existing importer validates the representative footprint even for an
        # empty loop; this fixture has no row at its positive start.
        with pytest.raises(ValueError, match="exceeds its proved allocation"):
            _apply(passes.insert_sync, original)
        return
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)  # Independent original accesses, covers, queries and selectors.
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical, check_codegen=bool(count))
    if count:
        assert dict(physical.attrs)[PYPTO_AUTOSYNC_ATTR]
    else:
        assert not dict(physical.attrs).get(PYPTO_AUTOSYNC_ATTR, False)
        # There is no manual SET/WAIT ownership marker. The downstream delegated
        # slot-admission restriction remains explicit, even for a zero trip.
        with pytest.raises(ValueError, match="delegated slot selection"):
            codegen.PTOCodegen().generate(ir.Program([physical], "Zero", physical.span))


@pytest.mark.parametrize("slots,stride", [(1, 0), (2, 1), (3, 1), (6, 2)])
def test_active_slot_recurrence_and_tail(slots, stride):
    count = 2 * slots + 1
    original = rotating_fixture(
        count, [(slots, stride)], SCHEDULE, stream_input=True, stream_output=slots > 1
    )
    validate(original, testing.analyze_placed_sync(original, True))
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)


def _loop(function):
    return next(s for s in sequence_stmts(function.body) if isinstance(s, ir.ForStmt))


def test_unused_large_period_does_not_construct_history():
    original = rotating_fixture(5, [(1000003, 1)], SCHEDULE, stream_input=True, stream_output=True)
    logical = _apply(passes.insert_sync, original)
    assert not _loop(logical).iter_args
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical, check_codegen=False)
    with pytest.raises(ValueError, match="at most 16 slots"):
        codegen.PTOCodegen().generate(ir.Program([physical], "LargePeriod", physical.span))


def test_fixed_lag_token_and_flag_size_independent_of_trip_count():
    sizes = []
    for count in [7, 1024, 1000000]:
        original = streamed_fixture(count)
        logical = _apply(passes.insert_sync, original)
        physical = _apply(passes.lower_sync, logical)
        text = ir.python_print(physical, format=False)
        sizes.append((len(_loop(logical).iter_args), text.count("set_flag("), text.count("wait_flag(")))
    assert sizes[0] == sizes[1] == sizes[2]
    assert sizes[0][0] == 3  # Scalar RFC state is linear in live lag, not its bit length.


def test_permuted_carries_keep_simultaneous_update_and_serialization(tmp_path):
    original = streamed_fixture(7)
    logical = _apply(passes.insert_sync, original)

    class Permute(ir.IRMutator):
        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            statements = list(op.body.stmts)
            last = statements[-1]
            assert isinstance(last, ir.YieldStmt)
            statements[-1] = ir.YieldStmt(list(reversed(last.value)), last.span)
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                list(reversed(op.iter_args)),
                ir.SeqStmts(statements, op.span),
                list(reversed(op.return_vars)),
                op.span,
            )

    logical = Permute().visit_function(logical)
    path = tmp_path / "rotating.msgpack"
    ir.serialize_to_file(logical, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    _check_actual_tokens(original, restored, _apply(passes.lower_sync, restored))


def test_history_guard_must_match_the_carried_lag():
    original = streamed_fixture(7, start=1, step=2)
    logical = _apply(passes.insert_sync, original)

    class WrongGuard(ir.IRMutator):
        def visit_if_stmt(self, op):
            assert isinstance(op.condition, ir.Gt)
            wrong = ir.Gt(
                op.condition.left, ir.ConstInt(1, ir.DataType.INDEX, op.span), ir.DataType.BOOL, op.span
            )
            return ir.IfStmt(wrong, op.then_body, op.else_body, op.return_vars, op.span)

    invalid = WrongGuard().visit_function(logical)
    before = ir.python_print(invalid, format=False)
    with pytest.raises(ValueError, match="guard|activation"):
        _apply(passes.lower_sync, invalid)
    assert ir.python_print(invalid, format=False) == before


@pytest.mark.ptoas_compile
def test_rotating_cpp_uses_manual_ownership(tmp_path):
    original = streamed_fixture(7)
    logical = _apply(passes.insert_sync, original)
    physical = _apply(passes.lower_sync, logical)
    _check_actual_tokens(original, logical, physical)
    source, output = tmp_path / "ring.pto", tmp_path / "ring.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([physical], "Ring", physical.span), emit_tile_addr=True, emit_source_loc=False
        )
    )
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    body = output.read_text().split("AICORE void ", 1)[1]
    assert "ptoas_auto_sync_tail(" not in body
    assert "set_flag(" in body and "wait_flag(" in body


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_existing_history_shortened_without_reanalysis(count):
    original = streamed_fixture(7)
    logical = _apply(passes.insert_sync, original)
    assert len(_loop(logical).iter_args) == 3

    class Shorten(ir.IRMutator):
        def visit_for_stmt(self, op):
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

    original = Shorten().visit_function(original)
    logical = Shorten().visit_function(logical)
    _check_actual_tokens(original, logical, _apply(passes.lower_sync, logical), check_codegen=bool(count))


def test_cyclic_shift_equations_refuse_transactionally():
    logical = _apply(passes.insert_sync, streamed_fixture(7))

    class Cycle(ir.IRMutator):
        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            statements = list(op.body.stmts)
            statements[-1] = ir.YieldStmt([op.iter_args[-1], *op.iter_args[:-1]], op.span)
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

    invalid = Cycle().visit_function(logical)
    before = ir.python_print(invalid, format=False)
    with pytest.raises(ValueError, match="Cyclic token history"):
        _apply(passes.lower_sync, invalid)
    assert ir.python_print(invalid, format=False) == before


def test_live_older_loop_result_is_not_the_final_source_visit():
    logical = _apply(passes.insert_sync, staged_fixture(3, visits=2))
    first = _loop(logical)
    arg = first.iter_args[0]
    older = ir.IterArg("older", arg.type, arg.initValue, first.span)
    result = ir.Var("older_final", arg.type, first.span)

    class OlderResult(ir.IRMutator):
        def visit_for_stmt(self, op):
            if op.same_as(first):
                assert isinstance(op.body, ir.SeqStmts)
                statements = list(op.body.stmts)
                last = statements[-1]
                assert isinstance(last, ir.YieldStmt)
                statements[-1] = ir.YieldStmt([*last.value, arg], last.span)
                return ir.ForStmt(
                    op.loop_var,
                    op.start,
                    op.stop,
                    op.step,
                    [*op.iter_args, older],
                    ir.SeqStmts(statements, op.span),
                    [*op.return_vars, result],
                    op.span,
                )
            return super().visit_for_stmt(op)

        def visit_var(self, op):
            return result if op.same_as(first.return_vars[0]) else op

    invalid = OlderResult().visit_function(logical)
    before = ir.python_print(invalid, format=False)
    with pytest.raises(ValueError, match="cross-invocation token frame"):
        _apply(passes.lower_sync, invalid)
    assert ir.python_print(invalid, format=False) == before


def test_coprime_families_keep_separate_histories():
    original = rotating_fixture(
        7,
        [(2, 1), (3, 1)],
        [("load", 0, 0), ("rmw", 0, 0), ("load", 1, 0), ("rmw", 1, 0), ("store", 1, 0)],
        stream_input=True,
        stream_output=True,
    )
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)
    logical = _apply(passes.insert_sync, original)
    # Each source keeps its own required history, rather than six joint phases.
    assert len(_loop(logical).iter_args) == 5
    _check_actual_tokens(original, logical, _apply(passes.lower_sync, logical))
