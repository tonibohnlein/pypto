# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Visit-owned streamed storage with independent original-access checks."""

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import sequence_stmts, validate
from .test_rotating_sync_analysis import rotating_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
SCHEDULE = [("load", 0, 0), ("rmw", 0, 0), ("store", 0, 0)]


def streamed_fixture(count, start=0, step=1, shared=False, shift=0, duplicate_store=False):
    schedule = SCHEDULE + ([("store", 0, 0)] if duplicate_store else [])
    function = rotating_fixture(
        count, [(3, 1)], schedule, start=start, step=step, stream_input=True, stream_output=True
    )
    if not shared:
        return function
    # An InOut tensor is both read and written. This is actual common storage,
    # not a certificate inferred from distinct parameter names.
    output = ir.Var(
        "out", ir.TensorType([max(1, start + count * step + shift), 32], ir.DataType.FP32), function.span
    )

    class Share(ir.IRMutator):
        def visit_var(self, op):
            return output if any(op.same_as(p) for p in function.params) else op

        def visit_call(self, op):
            op = super().visit_call(op)
            assert isinstance(op, ir.Call)
            if shift and op.op.name == ir.get_op("buffer.load").name:
                args = list(op.args)
                assert isinstance(args[1], ir.MakeTuple)
                offsets = list(args[1].elements)
                offsets[0] = ir.Add(
                    offsets[0], ir.ConstInt(shift, ir.DataType.INDEX, op.span), ir.DataType.INDEX, op.span
                )
                args[1] = ir.MakeTuple(offsets, op.span)
                return ir.Call(op.op, args, op.type, op.span)
            return op

    return ir.Function(
        function.name,
        [function.params[0], (output, ir.ParamDirection.InOut)],
        [output.type],
        Share().visit_stmt(function.body),
        function.span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def streamed_sequence(count, outside_reader=False, resident_reader=False):
    """Resident local prologue/tail surrounding one owning streamed child."""
    function = streamed_fixture(count)
    shared = {v.name_hint: v for v in function.params}
    for stmt in sequence_stmts(function.body)[:-2]:
        assert isinstance(stmt, ir.AssignStmt)
        shared[stmt.var.name_hint] = stmt.var

    def finite_body(operation, suffix, read_output=False):
        other = rotating_fixture(1, [(3, 1)], [(operation, 0, 0)], suffix=suffix)
        loop = sequence_stmts(other.body)[-2]
        assert isinstance(loop, ir.ForStmt)

        class Bind(ir.IRMutator):
            def visit_var(self, op):
                if op.name_hint == loop.loop_var.name_hint:
                    return loop.start
                if read_output and op.name_hint == "a":
                    return function.params[1]
                return shared.get(op.name_hint, op)

        body = ir.SeqStmts(list(sequence_stmts(loop.body)[:-1]), function.span)
        return list(sequence_stmts(Bind().visit_stmt(body)))

    roots = [
        *sequence_stmts(function.body)[:-2],
        *finite_body("load", "_prologue"),
        sequence_stmts(function.body)[-2],
    ]
    if resident_reader:
        # Separate qualified fixture: the prologue reader creates the needed
        # cross-pipe release before the next DMA overwrites its resident slot.
        # The original writer-only prologue remains a refusal regression.
        roots[-1:-1] = finite_body("rmw", "_resident")
    roots += finite_body("load" if outside_reader else "rmw", "_tail", outside_reader)
    return ir.Function(
        "streamed_sequence",
        [function.params[0], (function.params[1], ir.ParamDirection.InOut)],
        function.return_types,
        ir.SeqStmts([*roots, sequence_stmts(function.body)[-1]], function.span),
        function.span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def streamed_reset(inner, outer):
    function = streamed_fixture(inner)
    span, index = function.span, ir.DataType.INDEX
    iv = ir.Var("outer", ir.ScalarType(index), span)
    loop = ir.ForStmt(
        iv,
        ir.ConstInt(0, index, span),
        ir.ConstInt(outer, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts([sequence_stmts(function.body)[-2], ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return ir.Function(
        "streamed_reset",
        [function.params[0], (function.params[1], ir.ParamDirection.InOut)],
        function.return_types,
        ir.SeqStmts([*sequence_stmts(function.body)[:-2], loop, sequence_stmts(function.body)[-1]], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("count", [0, 1, 2, 4, 7])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("duplicate_store", [False, True])
def test_streamed_covers_queries_and_exact_membership(count, shared, duplicate_store):
    function = streamed_fixture(count, shared=shared, duplicate_store=duplicate_store)
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("count", [1, 2, 5])
def test_original_stride_and_absent_rows(count):
    function = streamed_fixture(count, start=1, step=2)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    absent = result.boundary(1, 0, 0)
    assert absent is not None and absent["first_writer"] is None
    present = result.boundary(1, 1, 0)
    assert present is not None
    assert present["first_writer"] is not None
    absent_writer = result.boundary(1, 2, 0)
    assert absent_writer is not None
    assert absent_writer["first_writer"] is None


@pytest.mark.parametrize("count", [0, 1, 2, 5])
@pytest.mark.parametrize("order", ["balanced", "left", "right"])
def test_streamed_owner_with_resident_local_boundaries(count, order):
    function = streamed_sequence(count)
    result = testing.analyze_placed_sync(function, True, composition_order=order)
    validate(function, result)
    assert result.describe()["hierarchical"]


def test_reader_in_another_child_is_not_exclusive_ownership():
    with pytest.raises(ValueError, match="(Moving|moving|owner|owned|selector|stream)"):
        testing.analyze_placed_sync(streamed_sequence(3, outside_reader=True), True)


def test_counter_reset_does_not_promote_local_disjointness():
    with pytest.raises(ValueError, match="(Nested|nested|invariant|reset|stream|owned)"):
        testing.analyze_placed_sync(streamed_reset(3, 2), True)


def test_union_of_all_accesses_must_fit_translation():
    # Reader of row i+1 overlaps the next visit's writer. Each one-row access
    # fits the unit stride separately; their union does not.
    with pytest.raises(ValueError, match="(Moving|moving|owned|separat|stream)"):
        testing.analyze_placed_sync(streamed_fixture(3, shared=True, shift=1), True)


@pytest.mark.parametrize("count", [7, 1000, 1000000])
def test_streamed_output_does_not_materialize_one_cell_per_visit(count):
    result = testing.analyze_placed_sync(streamed_sequence(count), True)
    desc = result.describe()
    assert desc["omitted_bridge_domains"] == 1
    assert sum(len(region["visit_owned"]) for region in desc["regions"]) == 1
    assert desc["analyzed_occurrences"] <= 5
    assert desc["interface_cells"] <= 8
    baseline = testing.analyze_placed_sync(streamed_sequence(7), True).describe()
    assert desc["interface_entries"] == baseline["interface_entries"]
    assert desc["interface_entries"] <= 4096
    assert desc["merge_nodes"] <= 3
    boundary = result.boundary(1, count - 1, 0)
    assert boundary is not None
    assert boundary["last_writer"] is not None


def test_mismatched_read_write_translation_refuses():
    function = streamed_fixture(3, shared=True)

    class StationaryRead(ir.IRMutator):
        def visit_call(self, op):
            op = super().visit_call(op)
            assert isinstance(op, ir.Call)
            if op.op.name == ir.get_op("buffer.load").name:
                args = list(op.args)
                zero = ir.ConstInt(0, ir.DataType.INDEX, op.span)
                args[1] = ir.MakeTuple([zero, zero], op.span)
                return ir.Call(op.op, args, op.type, op.span)
            return op

    with pytest.raises(ValueError, match="common translation"):
        testing.analyze_placed_sync(StationaryRead().visit_function(function), True)


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_column_translation_agrees_with_original_accesses(count):
    function = streamed_fixture(count)
    tensor = ir.TensorType([1, 32 * max(1, count)], ir.DataType.FP32)
    replacements = {p.name_hint: ir.Var(p.name_hint, tensor, function.span) for p in function.params}

    class Columns(ir.IRMutator):
        def visit_var(self, op):
            return replacements.get(op.name_hint, op)

        def visit_call(self, op):
            op = super().visit_call(op)
            assert isinstance(op, ir.Call)
            if op.op.name in {ir.get_op("buffer.load").name, ir.get_op("buffer.store").name}:
                args = list(op.args)
                assert isinstance(args[1], ir.MakeTuple)
                column = ir.Mul(
                    args[1].elements[0],
                    ir.ConstInt(32, ir.DataType.INDEX, op.span),
                    ir.DataType.INDEX,
                    op.span,
                )
                args[1] = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, op.span), column], op.span)
                return ir.Call(op.op, args, op.type, op.span)
            return op

    a, output = (replacements[p.name_hint] for p in function.params)
    modified = ir.Function(
        function.name,
        [a, (output, ir.ParamDirection.InOut)],
        [tensor],
        Columns().visit_stmt(function.body),
        function.span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    validate(modified, testing.analyze_placed_sync(modified, True))
