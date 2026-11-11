# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scoped logical token transport; physical certification is a separate stage."""

import pytest
from pypto import ir, passes, testing

from .sync_logical_oracle import CommandExecution, check_graph
from .sync_matmul_oracle import hardware_reference, interpret, validate
from .test_collective_sync_interface import collective_fixture
from .test_collective_sync_realization import nested_collective, repeat_collective
from .test_logical_sync_realization import verify
from .test_nested_sync_repetition import resident_pipeline_tree
from .test_token_sync_lowering import _apply

pytestmark = pytest.mark.usefixtures("ascend_backend")


def check(original):
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)  # Original effects -> actual covers, queries and selectors.
    original_snapshot = ir.serialize(ir.Program([original], "Original", original.span))
    logical = retained.realize()
    assert ir.serialize(ir.Program([original], "Original", original.span)) == original_snapshot
    verify(logical["function"])
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    state = CommandExecution(calls, reference[0])
    state.execute(logical["function"].body, {})
    assert state.payload_count == len(calls)
    assert not any(state.pending.values())
    checked = check_graph(state, reference, require_equality=True)
    return logical, checked


@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_collective_tokens_match_actual_covers(rows, visits):
    original = repeat_collective(rows, visits)
    logical, checked = check(original)
    assert checked["extra_event_pairs"] == 0
    inserted = _apply(passes.insert_sync, original)
    ir.assert_structural_equal(inserted, logical["function"])
    # This increment qualifies token insertion, not nested consuming flags.
    if visits and rows > 1:
        with pytest.raises(ValueError, match="nested token frames"):
            _apply(passes.lower_sync, inserted)


@pytest.mark.parametrize("counts", [(0, 2), (2, 0), (1, 1), (2, 3), (2, 2, 2)])
def test_nested_invocation_resets_preserve_token_identity(counts):
    check(nested_collective(3, counts))


@pytest.mark.parametrize("inner,outer,replay", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (2, 2, 2)])
def test_resident_operands_and_accumulator_tokens(inner, outer, replay):
    check(resident_pipeline_tree(inner, outer, replay))


def test_scoped_token_state_does_not_unfold_rows_or_visits():
    sizes = []
    for rows, counts in [(3, (2, 3)), (1000, (2, 3)), (1000000, (2, 3)), (3, (1000, 3)), (3, (2, 1000000))]:
        retained = testing.analyze_placed_sync(nested_collective(rows, counts), True)
        logical = retained.realize()
        verify(logical["function"])
        sizes.append(
            tuple(logical[k] for k in ("record_sites", "wait_sites", "guarded_wait_sites", "carried_tokens"))
        )
    assert len(set(sizes)) == 1


def test_scoped_token_serialization_and_repeat_preparation(tmp_path):
    original = nested_collective(3, (2, 2))
    logical, _ = check(original)
    first = logical["function"]
    path = tmp_path / "nested-tokens.msgpack"
    ir.serialize_to_file(first, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Function)
    verify(restored)
    ir.assert_structural_equal(first, restored)
    retained = testing.analyze_placed_sync(original, True)
    ir.assert_structural_equal(first, retained.realize()["function"])
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    state = CommandExecution(calls, reference[0])
    state.execute(restored.body, {})
    check_graph(state, reference, require_equality=True)


def wrapped_collective(visits=2):
    original = collective_fixture(5, [[(0, 2), (2, 5)]])
    assert isinstance(original.body, ir.SeqStmts)
    definition, *body, ret = original.body.stmts
    span = original.span
    iv = ir.Var("outer", ir.ScalarType(ir.DataType.INDEX), span)
    bounds = [ir.ConstInt(x, ir.DataType.INDEX, span) for x in [1, 1 + 2 * visits, 2]]
    loop = ir.ForStmt(iv, *bounds, [], ir.SeqStmts([*body, ir.YieldStmt([], span)], span), [], span)
    return ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([definition, loop, ret], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_nonfinal_row_capture_and_outer_seam_reset(visits):
    original = wrapped_collective(visits)
    logical, _ = check(original)
    if visits:

        class Captures(ir.IRVisitor):
            found = 0

            def visit_if_stmt(self, op):
                if op.return_vars and isinstance(op.return_vars[0].type, ir.SyncTokenType):
                    self.found += 1
                super().visit_if_stmt(op)

        visitor = Captures()
        visitor.visit_function(logical["function"])
        assert visitor.found > 0, "fixture must retain an interior row completion"


def test_dynamic_start_refusal_preserves_retained_mathematics():
    original = repeat_collective(3, 1)
    assert isinstance(original.body, ir.SeqStmts)
    definition, loop, ret = original.body.stmts
    assert isinstance(loop, ir.ForStmt)
    span = original.span
    start = ir.Var("start", ir.ScalarType(ir.DataType.INDEX), span)
    replacement = ir.ForStmt(
        loop.loop_var,
        start,
        ir.ConstInt(9, ir.DataType.INDEX, span),
        ir.ConstInt(10, ir.DataType.INDEX, span),
        [],
        loop.body,
        [],
        span,
    )
    function = ir.Function(
        original.name,
        [*zip(original.params, original.param_directions), start],
        original.return_types,
        ir.SeqStmts([definition, replacement, ret], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    retained = testing.analyze_placed_sync(function, True, parameter_bounds={2: (0, 8)})
    covers = retained.expand_retained_demands(32)
    snapshot = ir.serialize(ir.Program([function], "Dynamic", span))
    with pytest.raises(ValueError, match="constant original frame bounds"):
        retained.realize()
    assert retained.expand_retained_demands(32) == covers
    assert ir.serialize(ir.Program([function], "Dynamic", span)) == snapshot
