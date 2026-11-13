# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Collective row interfaces checked against original accesses, never sampled cells."""

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import validate

pytestmark = pytest.mark.usefixtures("ascend_backend")


def collective_fixture(rows=8, patterns=None, *, incomplete=False, invalid_view=False, nested=None):
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    patterns = patterns or [[(0, rows)], [(0, rows)]]
    ty = ir.BufferType([rows, 8], fp, ir.MemorySpace.Vec)
    row_type = ir.BufferType([1, 8], fp, ir.MemorySpace.Vec)
    buffer = ir.Var("resident", ty, span)
    a = ir.Var("a", ir.TensorType([rows, 8], fp), span)
    out = ir.Var("out", a.type, span)
    statements: list[ir.Stmt] = [ir.AssignStmt(buffer, call("buffer.alloc", [tup([]), ci(0)], ty), span)]
    for phase, pattern in enumerate(patterns):
        iv = ir.Var(f"row_{phase}", ir.ScalarType(index), span)
        row = ir.Var(f"window_{phase}", row_type, span)
        position = ir.Add(iv, ci(1), index, span) if invalid_view else iv
        offsets = ir.MakeTuple([position, ci(0)], span)
        body_statements: list[ir.Stmt] = [
            ir.AssignStmt(row, call("buffer.subview", [buffer, offsets], row_type), span)
        ]
        selected = row
        if nested:
            selected = ir.Var(f"nested_{phase}", row_type, span)
            args = [row, tup([0, 0])] if nested == "subview" else [row]
            body_statements.append(ir.AssignStmt(selected, call(f"buffer.{nested}", args, row_type), span))
        body_statements.extend(
            [
                ir.EvalStmt(
                    call("buffer.load", [a, ir.MakeTuple([iv, ci(0)], span), tup([1, 8]), selected]), span
                ),
                ir.YieldStmt([], span),
            ]
        )
        body = ir.SeqStmts(body_statements, span)
        statements.append(
            ir.ForStmt(iv, ci(0), ci(rows - 1 if incomplete else rows), ci(1), [], body, [], span)
        )
        for reader, (lo, hi) in enumerate(pattern):
            reader_type = ir.BufferType([hi - lo, 8], fp, ir.MemorySpace.Vec)
            view = ir.Var(f"reader_{phase}_{reader}", reader_type, span)
            statements.extend(
                [
                    ir.AssignStmt(view, call("buffer.subview", [buffer, tup([lo, 0])], reader_type), span),
                    ir.EvalStmt(call("buffer.store", [view, tup([lo, 0]), tup([hi - lo, 8]), out]), span),
                ]
            )
    statements.append(ir.ReturnStmt([out], span))
    return ir.Function(
        "collective",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("rows", [1, 2, 5, 9])
@pytest.mark.parametrize("order", ["left", "right", "balanced"])
def test_collective_original_covers_queries_selectors(rows, order):
    function = collective_fixture(rows)
    result = testing.analyze_placed_sync(function, True, composition_order=order)
    assert result.describe()["collective_domains"] == 1
    validate(function, result)


def test_collective_reader_intervals_and_other_domain_effects():
    # Reader writes to GM overlap too; the row formulas are generators and must
    # be reduced jointly with those independent GM conflicts.
    patterns = [[(0, 5), (2, 8)], [(4, 8), (0, 6)], [(0, 8)]]
    function = collective_fixture(8, patterns)
    for order in ["left", "right", "balanced"]:
        result = testing.analyze_placed_sync(function, True, composition_order=order)
        validate(function, result)


def test_collective_frame_size_does_not_enumerate_rows():
    signatures = []
    for rows in [8, 1000, 1000000]:
        result = testing.analyze_placed_sync(collective_fixture(rows), True)
        d = result.describe()
        signatures.append((d["collective_witnesses"], d["collective_access_visits"], d["interface_cells"]))
        assert d["collective_witnesses"] == 2
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("case", ["incomplete_fill", "missing_read", "outside_view"])
def test_collective_missing_premises_remain_unavailable(case):
    function = collective_fixture(
        8,
        [[(0, 7)]] if case == "missing_read" else None,
        incomplete=case == "incomplete_fill",
        invalid_view=case == "outside_view",
    )
    with pytest.raises(ValueError, match="(Visit-owned|proved parent window|selector partition)"):
        testing.analyze_placed_sync(function, True)


def test_collective_moving_view_cannot_escape_its_frame():
    class Escape(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.window = None

        def visit_assign_stmt(self, op):
            if op.var.name_hint == "window_0":
                self.window = op.var
            return super().visit_assign_stmt(op)

        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.store").name:
                assert self.window is not None
                # Use the row handle from a completed fill loop in an outer
                # consumer. SSA scope and occurrence binding must not be guessed.
                size = ir.MakeTuple([ir.ConstInt(v, ir.DataType.INDEX, op.span) for v in [1, 8]], op.span)
                return ir.Call(op.op, [self.window, op.args[1], size, op.args[3]], op.type, op.span)
            return super().visit_call(op)

    with pytest.raises(ValueError, match="(occurrence frame|scope|defined)"):
        testing.analyze_placed_sync(Escape().visit_function(collective_fixture(4, [[(0, 4)]])), True)


@pytest.mark.parametrize("nested", ["subview", "reshape"])
def test_collective_nested_views_preserve_pitch_and_original_frame(nested):
    function = collective_fixture(5, nested=nested)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    assert result.describe()["collective_domains"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
