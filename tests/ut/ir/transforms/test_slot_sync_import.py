# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Slot recognition checks; no reduction is inferred from successful import."""

import pytest
from pypto import ir, testing

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import sequence_stmts
from .test_indexed_sync_analysis import indexed_fixture
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def slot_fixture(count=5, slots=3, stride=1, phase=0, start=0, step=1, alias=False):
    """One explicit Vec slot family; addresses and selection are original IR."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    ty = ir.BufferType([1, 32], fp, ir.MemorySpace.Vec)
    group_ty = ir.MultiBufferType(ty, slots)
    group = ir.Var("ring", group_ty, span)
    a = ir.Var("a", ir.TensorType([1, 32], fp), span)
    out = ir.Var("out", a.type, span)
    definitions = [ir.AssignStmt(group, call("buffer.alloc_multi", [tup([]), ci(0)], group_ty), span)]
    if alias:
        group2 = ir.Var("alias", group_ty, span)
        definitions.append(
            ir.AssignStmt(group2, call("buffer.alloc_multi", [tup([]), ci(0)], group_ty), span)
        )
    else:
        group2 = group
    iv = ir.Var("i", ir.ScalarType(index), span)
    expression = ir.Add(ir.Mul(iv, ci(stride), index, span), ci(phase), index, span)
    selection = ir.FloorMod(expression, ci(slots), index, span)
    x, y = ir.Var("x", ty, span), ir.Var("y", ty, span)
    body: list[ir.Stmt] = [
        ir.AssignStmt(x, call("buffer.slot", [group, selection], ty), span),
        ir.AssignStmt(y, call("buffer.slot", [group2, selection], ty), span),
        ir.EvalStmt(call("buffer.load", [a, tup([0, 0]), tup([1, 32]), x]), span),
        ir.EvalStmt(call("buffer.abs", [x, x]), span),
        ir.EvalStmt(call("buffer.store", [y, tup([0, 0]), tup([1, 32]), out]), span),
        ir.YieldStmt([], span),
    ]
    loop = ir.ForStmt(
        iv, ci(start), ci(start + count * step), ci(step), [], ir.SeqStmts(body, span), [], span
    )
    return ir.Function(
        "rotating_vec",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts([*definitions, loop, ir.ReturnStmt([out], span)], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize(("slots", "stride", "phase"), [(1, 0, 0), (3, 1, 2), (6, 2, 3), (7, 0, 2)])
@pytest.mark.parametrize(("count", "start", "step"), [(0, 0, 1), (1, 3, 2), (9, 1, 3)])
def test_slot_access_description_matches_original_addresses(slots, stride, phase, count, start, step):
    function = slot_fixture(count, slots, stride, phase, start, step, alias=True)
    result = testing.describe_placed_sync_input(function, True)
    assert "not requested" in result["guarantee"]
    assert len(result["storage"]) == 3, "two handles of the same ring are one physical family"
    ring = result["storage"][2]
    assert (ring["slot_count"], ring["slot_bytes"], ring["bytes"]) == (slots, 128, slots * 128)
    for site in result["regions"][0]["sites"]:
        for access, selection in zip(site["accesses"], site["slot_selections"]):
            if access[0] != 2:
                continue
            for visit in range(count):
                # Original source formula, independent of importer normalization.
                actual = ((start + visit * step) * stride + phase) % slots
                selected = (selection[0] + visit * selection[1]) % slots
                assert ring["address"] + 128 * selected == 128 * actual
    assert not result["analysis_available"]
    assert result["regions"][0]["demands"] is None


@pytest.mark.parametrize("count, slots", [(0, 3), (7, 6), (1000000, 1000003)])
def test_exact_interval_evidence_retains_physical_aliases(count, slots):
    description = testing.describe_placed_sync_input(slot_fixture(count, slots, alias=True), True)
    visits = 0
    for site in description["regions"][0]["sites"]:
        for access, interval in zip(site["accesses"], site["physical_intervals"], strict=True):
            visits += 1
            # All accesses in this original fixture cover 32 contiguous FP32
            # elements. Aliasing handles retain the same family and byte range.
            assert interval == (0, 128)
    assert description["interval_access_visits"] == visits
    assert description["qualified_intervals"] == visits
    assert not description["analysis_available"]


def interval_extract_fixture(major, boxed, shape, offset):
    """Ready full Mat fill followed by an original partial operand extraction."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    minor = ir.TileLayout.row_major if boxed else ir.TileLayout.none_box
    source_type = ir.BufferType([32, 32], fp, ir.MemorySpace.Mat, blayout=major, slayout=minor, fractal=512)
    target_type = ir.BufferType(shape, fp, ir.MemorySpace.Left, blayout=major, slayout=minor, fractal=512)
    source, target = ir.Var("source", source_type, span), ir.Var("target", target_type, span)
    arg = ir.Var("arg", ir.TensorType([32, 32], fp), span)
    statements: list[ir.Stmt] = [
        ir.AssignStmt(source, call("buffer.alloc", [tup([]), ci(0)], source_type), span),
        ir.AssignStmt(target, call("buffer.alloc", [tup([]), ci(0)], target_type), span),
        ir.EvalStmt(call("buffer.load", [arg, tup([0, 0]), tup([32, 32]), source]), span),
        ir.EvalStmt(call("buffer.extract", [source, ci(offset[0]), ci(offset[1]), target]), span),
        ir.ReturnStmt([], span),
    ]
    return ir.Function(
        "interval_extract",
        [arg],
        [],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize(
    "major, boxed, shape, offset, expected",
    [
        (ir.TileLayout.row_major, False, [16, 32], [16, 0], (2048, 4096)),
        (ir.TileLayout.col_major, False, [32, 16], [0, 16], (2048, 4096)),
        (ir.TileLayout.row_major, False, [16, 16], [16, 16], None),
        (ir.TileLayout.col_major, False, [16, 16], [16, 16], None),
        (ir.TileLayout.col_major, True, [16, 16], [16, 16], None),
        (ir.TileLayout.col_major, True, [32, 32], [0, 0], (0, 4096)),
    ],
)
def test_partial_intervals_require_exact_layout(major, boxed, shape, offset, expected):
    description = testing.describe_placed_sync_input(
        interval_extract_fixture(major, boxed, shape, offset), True
    )
    source_access = description["regions"][0]["sites"][-1]
    assert source_access["physical_intervals"][0] == expected
    assert source_access["access_precision"][0] == "exact"
    assert source_access["accesses"][0][2:4] == tuple(offset)
    if expected is not None and not boxed:
        # Independent element-address enumeration; no compiler cell partition.
        addresses = set()
        for row in range(offset[0], offset[0] + shape[0]):
            for col in range(offset[1], offset[1] + shape[1]):
                element = row * 32 + col if major == ir.TileLayout.row_major else col * 32 + row
                addresses.update(range(4 * element, 4 * element + 4))
        assert addresses == set(range(*expected))


def test_interval_unavailability_preserves_enclosed_and_moving_effects():
    enclosed = testing.describe_placed_sync_input(indexed_fixture(), True)
    moving = testing.describe_placed_sync_input(staged_fixture(2), True)
    counts = [0, 0]
    for kind, description in enumerate([enclosed, moving]):
        for region in description["regions"]:
            for site in region["sites"]:
                for access, precision, interval in zip(
                    site["accesses"], site["access_precision"], site["physical_intervals"], strict=True
                ):
                    if (kind == 0 and precision == "enclosure") or (kind == 1 and any(access[4:6])):
                        assert interval is None
                        assert access[6] > 0 and access[7] > 0
                        counts[kind] += 1
    assert all(counts)


def test_interval_offsets_preserve_strided_view_gaps():
    description = testing.describe_placed_sync_input(vec_fixture(view="nested"), True)
    # Existing recovery lists the two rows. Qualification must not fill the gap.
    site = description["regions"][0]["sites"][2]
    assert site["physical_intervals"][:2] == [(288, 320), (416, 448)]


def test_recognition_does_not_enumerate_slots_or_trips():
    signatures = []
    for count, slots in [(3, 2), (1000, 7), (1000000, 1000003)]:
        d = testing.describe_placed_sync_input(slot_fixture(count, slots), True)
        signatures.append((len(d["storage"]), len(d["regions"][0]["sites"]), d["coordinate_nodes"]))
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("bad", ["overlap", "modulus", "overflow", "unbounded"])
def test_unproved_slot_geometry_or_selection_refuses(bad):
    class Change(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name != ir.get_op("buffer.slot").name:
                return super().visit_call(op)
            span = op.span
            index = ir.DataType.INDEX
            selection = op.args[1]
            assert isinstance(selection, ir.FloorMod)
            if bad == "modulus":
                selection = ir.FloorMod(selection.left, ir.ConstInt(2, index, span), index, span)
            elif bad == "overflow":
                selection = ir.FloorMod(
                    ir.Mul(selection.left, ir.ConstInt(2**62, index, span), index, span),
                    selection.right,
                    index,
                    span,
                )
            elif bad == "unbounded":
                selection = selection.left
            return ir.Call(op.op, [op.args[0], selection], op.type, span)

        def visit_assign_stmt(self, op):
            if bad == "overlap" and op.var.name_hint == "alias":
                value = op.value
                assert isinstance(value, ir.Call)
                address = ir.ConstInt(128, ir.DataType.INDEX, op.span)
                return ir.AssignStmt(
                    op.var,
                    ir.Call(value.op, [value.args[0], address], value.type, value.span),
                    op.span,
                )
            return super().visit_assign_stmt(op)

    f = Change().visit_function(slot_fixture(alias=True))
    with pytest.raises(ValueError, match="(Overlapping|modulus|nonoverflowing|Slot selection)"):
        testing.describe_placed_sync_input(f, True)


@pytest.mark.parametrize("ordinary_first", [False, True])
@pytest.mark.parametrize("address", [0, 32])
def test_one_slot_ring_never_hides_ordinary_vec_alias(ordinary_first, address):
    # A singleton MultiBuffer is still a family-domain descriptor. Its overlap
    # with the shared byte-domain Vec representation must not become disjoint.
    f = slot_fixture(slots=1)
    span = f.span
    ty = ir.BufferType([1, 32], ir.DataType.FP32, ir.MemorySpace.Vec)
    var = ir.Var("ordinary", ty, span)
    call = ir.Call(
        ir.get_op("buffer.alloc"),
        [ir.MakeTuple([], span), ir.ConstInt(address, ir.DataType.INDEX, span)],
        ty,
        span,
    )
    statements = list(sequence_stmts(f.body))
    statements.insert(0 if ordinary_first else 1, ir.AssignStmt(var, call, span))
    changed = ir.Function(
        f.name,
        list(zip(f.params, f.param_directions)),
        f.return_types,
        ir.SeqStmts(statements, span),
        span,
        f.func_type,
        ir_stage=f.ir_stage,
    )
    with pytest.raises(ValueError, match="Overlapping"):
        testing.describe_placed_sync_input(changed, True)


def test_narrowing_before_modulo_is_not_hidden_by_slot_bound():
    class Narrow(ir.IRMutator):
        def visit_floor_mod(self, op):
            narrow = ir.Cast(op.left, ir.DataType.INT8, op.span)
            wide = ir.Cast(narrow, ir.DataType.INDEX, op.span)
            return ir.FloorMod(wide, op.right, ir.DataType.INDEX, op.span)

    with pytest.raises(ValueError, match="cast may change"):
        testing.describe_placed_sync_input(Narrow().visit_function(slot_fixture(count=200)), True)


def test_partial_slot_view_is_unavailable():
    class Partial(ir.IRMutator):
        def visit_assign_stmt(self, op):
            if op.var.name_hint != "x":
                return super().visit_assign_stmt(op)
            ty = ir.BufferType([1, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
            var = ir.Var("partial", ty, op.span)
            offsets = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, op.span)] * 2, op.span)
            view = ir.Call(ir.get_op("buffer.subview"), [op.var, offsets], ty, op.span)
            return ir.SeqStmts([op, ir.AssignStmt(var, view, op.span)], op.span)

    with pytest.raises(ValueError, match="(storage/view|Nested or data-dependent control)"):
        testing.describe_placed_sync_input(Partial().visit_function(slot_fixture()), True)


def test_native_padding_cannot_change_the_imported_slot_stride():
    span = ir.Span.unknown()
    element = ir.BufferType([1, 4], ir.DataType.FP32, ir.MemorySpace.Vec)
    group_type = ir.MultiBufferType(element, 2)
    group = ir.Var("ring", group_type, span)
    alloc = ir.Call(
        ir.get_op("buffer.alloc_multi"),
        [ir.MakeTuple([], span), ir.ConstInt(0, ir.DataType.INDEX, span)],
        group_type,
        span,
    )
    f = ir.Function(
        "unaligned",
        [],
        [],
        ir.AssignStmt(group, alloc, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    # Native multi-tile stride rounds 16 to 32. Never report a 16-byte physical
    # slot stride as qualified while leaving that padding unrepresented.
    with pytest.raises(ValueError, match="aligned whole-slot"):
        testing.describe_placed_sync_input(f, True)


def test_selection_from_another_occurrence_frame_is_refused():
    f = slot_fixture()
    span = f.span
    original = sequence_stmts(f.body)[-2]
    assert isinstance(original, ir.ForStmt)
    iv = ir.Var("other", ir.ScalarType(ir.DataType.INDEX), span)
    # Deliberately retain a selected handle from the previous loop. Even when
    # its descriptor and modulus agree, it is not a selection in this visit.
    store = sequence_stmts(original.body)[-2]
    second = ir.ForStmt(
        iv,
        original.start,
        original.stop,
        original.step,
        [],
        ir.SeqStmts([store, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    changed = ir.Function(
        f.name,
        list(zip(f.params, f.param_directions)),
        f.return_types,
        ir.SeqStmts([*sequence_stmts(f.body)[:-1], second, sequence_stmts(f.body)[-1]], span),
        span,
        f.func_type,
        ir_stage=f.ir_stage,
    )
    with pytest.raises(ValueError, match="outside its defining scope"):
        testing.describe_placed_sync_input(changed, True)


def test_different_strides_do_not_supply_a_common_stride_certificate():
    class Mixed(ir.IRMutator):
        def visit_assign_stmt(self, op):
            if op.var.name_hint != "y":
                return super().visit_assign_stmt(op)
            call = op.value
            assert isinstance(call, ir.Call)
            index = ir.DataType.INDEX
            expr = call.args[1]
            assert isinstance(expr, ir.FloorMod)
            selection = ir.FloorMod(
                ir.Mul(expr.left, ir.ConstInt(2, index, op.span), index, op.span),
                expr.right,
                index,
                op.span,
            )
            return ir.AssignStmt(
                op.var, ir.Call(call.op, [call.args[0], selection], call.type, op.span), op.span
            )

    with pytest.raises(ValueError, match="one common stride"):
        testing.describe_placed_sync_input(Mixed().visit_function(slot_fixture()), True)
