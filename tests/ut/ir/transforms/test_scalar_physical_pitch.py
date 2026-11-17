# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar native flat offsets require inherited physical row-pitch evidence."""

import pytest
from pypto import codegen, ir, passes, testing

from .test_closed_prefix_sync import Services
from .test_runtime_valid_broadcast import diagnostics

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def scalar_window(width=8, *, write=False, borrowed=False, commands=False):
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(n):
        return ir.ConstInt(n, index, SPAN)

    def tup(ns):
        return ir.MakeTuple([ci(n) for n in ns], SPAN)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), SPAN)

    root_type = ir.BufferType([4, 16], fp, ir.MemorySpace.Vec)
    kind = ir.BufferType([2, width], fp, ir.MemorySpace.Vec)
    root, view = ir.Var("root", root_type, SPAN), ir.Var("view", kind, SPAN)
    alias = ir.Var("overlapping_alias", kind, SPAN)
    incoming = ir.Var("borrowed", ir.TupleType([kind]), SPAN)
    # A direct Buffer parameter has an ABI layout contract. A tuple-projected
    # borrowed handle has no retained physical-window evidence for this client.
    setup = (
        [ir.AssignStmt(view, ir.TupleGetItemExpr(incoming, 0, SPAN), SPAN)]
        if borrowed
        else [
            ir.AssignStmt(root, call("buffer.alloc", [tup([]), ci(0)], root_type), SPAN),
            ir.AssignStmt(view, call("buffer.subview", [root, tup([1, 0])], kind), SPAN),
            # This second explicit alias retains the same parent pitch and bytes.
            ir.AssignStmt(alias, call("buffer.subview", [root, tup([1, 0])], kind), SPAN),
        ]
    )
    operand = view if borrowed else alias
    if write:
        scalar = ir.EvalStmt(call("buffer.write", [operand, tup([1, 0]), ir.ConstFloat(3.0, fp, SPAN)]), SPAN)
    else:
        result = ir.Var("scalar", ir.ScalarType(fp), SPAN)
        scalar = ir.AssignStmt(result, call("buffer.read", [operand, tup([1, 0])], result.type), SPAN)
    protocol = Services().closed(ir.PipeType.S, ir.PipeType.V) if commands else []
    return ir.Function(
        "scalar_window",
        [incoming] if borrowed else [],
        [],
        ir.SeqStmts([*setup, scalar, *protocol, ir.ReturnStmt([], SPAN)], SPAN),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def program(function):
    return ir.Program([function], "ScalarPitch", SPAN)


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("borrowed", [False, True])
def test_scalar_pitch_is_checked_by_all_ownership_paths(write, borrowed):
    function = scalar_window(write=write, borrowed=borrowed)
    before = ir.serialize(function)
    assert any("dense physical row pitch" in item.message for item in diagnostics(function))
    with pytest.raises(ValueError, match="dense physical row pitch"):
        testing.analyze_placed_sync(function, True)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        with pytest.raises(ValueError, match="dense physical row pitch"):
            passes.insert_sync()(program(function))
    # Both the no-token and the closed-prefix physical paths must enforce the
    # native layout premise before publishing an owned plan.
    for source in (function, scalar_window(write=write, borrowed=borrowed, commands=True)):
        with pytest.raises(ValueError, match="dense physical row pitch"):
            passes.lower_sync()(program(source))
    # Logical token IR is deliberately not a code-generation input. Use the
    # token-free descriptor fixture to inspect native layout qualification.
    with pytest.raises(ValueError, match="dense physical row pitch"):
        codegen.PTOCodegen().generate(program(function))
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", autosync_disjoint_gm=True
    ):
        result = passes.realize_sync()(program(function)).get_function(function.name)
    assert result is not None and not result.attrs.get("pypto_autosync", False)
    ir.assert_structural_equal(result.body, function.body)
    assert ir.serialize(function) == before


@pytest.mark.parametrize("write", [False, True])
def test_full_width_overlapping_windows_have_the_proved_dense_pitch(write):
    function = scalar_window(16, write=write, commands=True)
    assert not diagnostics(function)
    with passes.PassContext([]):
        physical = passes.lower_sync()(program(function))
    result = physical.get_function(function.name)
    assert result is not None and not diagnostics(result)
    assert ("pto.tsetval" if write else "pto.tgetval") in codegen.PTOCodegen().generate(physical)


def test_malformed_discarded_buffer_read_is_refused_without_a_dummy_value():
    original = scalar_window(16)

    class DiscardRead(ir.IRMutator):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("buffer.read").name:
                return ir.EvalStmt(op.value, op.span)
            return super().visit_assign_stmt(op)

    function = DiscardRead().visit_function(original)
    snapshot = ir.serialize(function)
    assert any("SSA results require a direct AssignStmt" in item.message for item in diagnostics(function))
    with pytest.raises(ValueError, match="SSA results require a direct AssignStmt"):
        passes.lower_sync()(program(function))
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        with pytest.raises(ValueError, match="SSA results require a direct AssignStmt"):
            passes.insert_sync()(program(function))
    assert ir.serialize(function) == snapshot
