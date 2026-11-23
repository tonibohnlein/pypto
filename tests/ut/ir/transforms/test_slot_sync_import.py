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
from pypto import ir, passes, testing
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR

from .sync_closed_command_oracle import ClosedCommands, opname
from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import check_imported_accesses, interpret, sequence_stmts
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


def test_slot_reuse_waits_for_both_readers_and_leaves_the_other_slot_available():
    original = slot_fixture(count=1, slots=2)
    allocation, loop, ret = sequence_stmts(original.body)
    select, _, load, vector, store, _ = sequence_stmts(loop.body)
    group = allocation.var
    span = original.span
    ty = select.var.type

    def ci(x):
        return ir.ConstInt(x, ir.DataType.INDEX, span)

    def payload(template, args):
        return ir.EvalStmt(ir.Call(template.expr.op, args, template.expr.type, span), span)

    slots = [select.var, ir.Var("other", ty, span)]
    definitions = [allocation]
    for i, slot in enumerate(slots):
        definitions.append(ir.AssignStmt(slot, ir.Call(select.value.op, [group, ci(i)], ty, span), span))
    scratch = ir.Var("result", ty, span)
    alloc = ir.Call(ir.get_op("buffer.alloc"), [allocation.value.args[0], ci(256)], ty, span)
    definitions.append(ir.AssignStmt(scratch, alloc, span))
    payloads = [
        payload(load, [*load.expr.args[:-1], slots[0]]),
        payload(vector, [slots[0], scratch]),
        payload(store, [slots[0], *store.expr.args[1:]]),
        payload(load, [*load.expr.args[:-1], slots[1]]),
        payload(load, [*load.expr.args[:-1], slots[0]]),
    ]
    function = ir.Function(
        original.name,
        original.params,
        original.return_types,
        ir.SeqStmts([*definitions, *payloads, ret], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([function], "slot_wrap", span)
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", autosync_disjoint_gm=True
    ):
        result = passes.realize_sync()(program)[function.name]
    assert result.attrs[PYPTO_AUTOSYNC_ATTR]
    commands = ClosedCommands(result, {}, range(8), tensor_bases={0: 0, 1: 4096})
    order = commands.run()
    complete = [("payload", i, "C") for i in (1, 2)]
    assert [item[1] for item in commands.payloads[1:3]] == [ir.PipeType.V, ir.PipeType.MTE3]
    # Both readers touch slot zero. The next issue can use slot one while
    # they run, but wrapping to slot zero must wait for both completions.
    assert all((reader, ("payload", 3, "I")) not in order for reader in complete)
    assert all((reader, ("payload", 4, "I")) in order for reader in complete)

    class DropRelease(ClosedCommands):
        def __init__(self, source):
            super().__init__(result, {}, range(8), tensor_bases={0: 0, 1: 4096})
            self.source = source

        def physical(self, call):
            if call.op.name in (opname("system.set_flag"), opname("system.wait_flag")):
                if (ir.PipeType(call.kwargs["src_pipe"]), ir.PipeType(call.kwargs["dst_pipe"])) == (
                    self.source,
                    ir.PipeType.MTE2,
                ):
                    return
            super().physical(call)

    for source, reader in zip((ir.PipeType.V, ir.PipeType.MTE3), complete, strict=True):
        assert (reader, ("payload", 4, "I")) not in DropRelease(source).run()


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


def nested_slot_fixture(outer_count=3, inner_count=4, *, coupled=False, escaped=False, constant=False):
    """An ancestor selects a resident slot; the inner counter drives GM rows."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(value) for value in values], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    ty = ir.BufferType([1, 32], fp, ir.MemorySpace.Vec)
    group_ty = ir.MultiBufferType(ty, 3)
    group = ir.Var("outer_ring", group_ty, span)
    source = ir.Var("source", ir.TensorType([max(1, inner_count), 32], fp), span)
    output = ir.Var("output", source.type, span)
    outer_iv, inner_iv = [ir.Var(name, ir.ScalarType(index), span) for name in ("outer", "inner")]
    selected = ir.Var("selected", ty, span)
    named_slot = ir.Var("slot_index", ir.ScalarType(index), span)
    numerator = ir.Add(ir.Mul(outer_iv, ci(2), index, span), ci(1), index, span)
    if coupled:
        numerator = ir.Add(numerator, inner_iv, index, span)
    selection = ci(1) if constant else ir.FloorMod(numerator, ci(3), index, span)
    definitions = [
        ir.AssignStmt(named_slot, selection, span),
        ir.AssignStmt(selected, call("buffer.slot", [group, named_slot], ty), span),
    ]
    offsets = ir.MakeTuple([inner_iv, ci(0)], span)
    body = [
        ir.EvalStmt(call("buffer.load", [source, offsets, tup([1, 32]), selected]), span),
        ir.EvalStmt(call("buffer.abs", [selected, selected]), span),
        ir.EvalStmt(call("buffer.store", [selected, offsets, tup([1, 32]), output]), span),
        ir.YieldStmt([], span),
    ]
    inner = ir.ForStmt(
        inner_iv,
        ci(0),
        ci(inner_count),
        ci(1),
        [],
        ir.SeqStmts([*(definitions if coupled else []), *body], span),
        [],
        span,
    )
    outer = ir.ForStmt(
        outer_iv,
        ci(1),
        ci(1 + 2 * outer_count),
        ci(2),
        [],
        ir.SeqStmts([*([] if coupled else definitions), inner, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    statements: list[ir.Stmt] = [
        ir.AssignStmt(group, call("buffer.alloc_multi", [tup([]), ci(0)], group_ty), span),
        outer,
    ]
    if escaped:
        statements.append(ir.EvalStmt(call("buffer.abs", [selected, selected]), span))
    statements.append(ir.ReturnStmt([output], span))
    function = ir.Function(
        "nested_slot_frames",
        [source, (output, ir.ParamDirection.InOut)],
        [output.type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return function, outer, inner


@pytest.mark.parametrize("outer_count,inner_count", [(0, 4), (1, 1), (3, 0), (3, 4), (4, 2)])
def test_ancestor_slot_import_preserves_original_frame_and_access_agreement(outer_count, inner_count):
    function, outer, inner = nested_slot_fixture(outer_count, inner_count)
    description = testing.describe_placed_sync_input(function, True)
    storage, calls = interpret(function)
    check_imported_accesses(description, storage, calls)
    assert len(calls) == 3 * outer_count * inner_count
    for region in description["regions"]:
        for site in region["sites"]:
            for access, slot, slot_frame, address_frame in zip(
                site["accesses"],
                site["slot_selections"],
                site["slot_frames"],
                site["access_frames"],
                strict=True,
            ):
                domain = description["storage"][access[0]]
                if domain["slot_count"] > 1:
                    assert slot == (0, 1), "(2*(1+2*visit)+1)%3 uses the original outer ordinal"
                    assert slot_frame.same_as(outer)
                    assert address_frame is None
                elif access[4] or access[5]:
                    assert address_frame.same_as(inner)
                    assert slot_frame is None
    # Slot phases are now retained, but these moving GM writes reuse the same
    # rows in every outer visit. A visit-local disjointness argument is not an
    # external replay certificate; preserve that independent export refusal.
    with pytest.raises(
        ValueError, match="Phased moving GM writes need a scoped external ownership certificate"
    ):
        testing.analyze_placed_sync(function, True)


@pytest.mark.parametrize("constant", [False, True])
def test_slot_selection_cannot_escape_its_defining_invocation(constant):
    function, _, _ = nested_slot_fixture(escaped=True, constant=constant)
    # The shared IR verifier rejects this before descriptor recovery, including
    # the constant-selector case. Do not bypass it to test a later check.
    with pytest.raises(ValueError, match="used outside its defining scope"):
        testing.describe_placed_sync_input(function, True)


def test_slot_selection_with_coupled_enclosing_coordinates_stays_unavailable():
    function, _, _ = nested_slot_fixture(coupled=True)
    with pytest.raises(ValueError, match="Coupled slot coordinates require a multidimensional"):
        testing.describe_placed_sync_input(function, True)


@pytest.mark.parametrize("boxed", [False, True])
@pytest.mark.parametrize("start, step", [(0, 1), (1, 3)])
def test_framed_moving_slot_rectangle_is_exact_without_an_interval_export(boxed, start, step):
    original = interval_extract_fixture(ir.TileLayout.row_major, boxed, (32, 16), (0, 0))
    span, index = original.span, ir.DataType.INDEX
    source_alloc, target_alloc, load, extract, result = sequence_stmts(original.body)
    assert isinstance(source_alloc, ir.AssignStmt) and isinstance(source_alloc.value, ir.Call)
    assert isinstance(target_alloc, ir.AssignStmt)
    assert isinstance(load, ir.EvalStmt) and isinstance(load.expr, ir.Call)
    source_type = source_alloc.var.type
    assert isinstance(source_type, ir.BufferType)
    multi = ir.MultiBufferType(source_type, 2)
    group, selected = ir.Var("ring", multi, span), ir.Var("selected", source_type, span)
    outer_iv, inner_iv = [ir.Var(name, ir.ScalarType(index), span) for name in ("outer", "inner")]

    def ci(value):
        return ir.ConstInt(value, index, span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    allocation = ir.AssignStmt(group, call("buffer.alloc_multi", source_alloc.value.args, multi), span)
    select = ir.AssignStmt(
        selected, call("buffer.slot", [group, ir.FloorMod(outer_iv, ci(2), index, span)], source_type), span
    )
    load = ir.EvalStmt(call("buffer.load", [*load.expr.args[:-1], selected]), span)
    extract = ir.EvalStmt(
        call("buffer.extract", [selected, ci(0), ir.Mul(inner_iv, ci(16), index, span), target_alloc.var]),
        span,
    )
    inner = ir.ForStmt(
        inner_iv, ci(0), ci(2), ci(1), [], ir.SeqStmts([extract, ir.YieldStmt([], span)], span), [], span
    )
    outer = ir.ForStmt(
        outer_iv,
        ci(start),
        ci(start + 3 * step),
        ci(step),
        [],
        ir.SeqStmts([select, load, inner, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    function = ir.Function(
        "framed_rectangles",
        original.params,
        [],
        ir.SeqStmts([allocation, target_alloc, outer, result], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    description = testing.describe_placed_sync_input(function, True)
    storage, calls = interpret(function)
    check_imported_accesses(description, storage, calls)
    extraction = next(
        site
        for region in description["regions"]
        for site in region["sites"]
        if site["original"].same_as(extract.expr)
    )
    assert extraction["access_precision"][0] == "exact"
    assert extraction["physical_intervals"][0] is None
    assert extraction["slot_frames"][0].same_as(outer)
    assert extraction["access_frames"][0].same_as(inner)


@pytest.mark.parametrize("kind", ["equivalent", "different_slot", "extra_payload"])
def test_initialization_macro_qualifies_arm_local_slot_definitions(kind):
    original = staged_fixture(2, stages=1)
    span, index = original.span, ir.DataType.INDEX
    statements = sequence_stmts(original.body)
    loop = next(statement for statement in statements if isinstance(statement, ir.ForStmt))
    branch = next(statement for statement in sequence_stmts(loop.body) if isinstance(statement, ir.IfStmt))
    assert isinstance(branch.then_body, ir.EvalStmt) and isinstance(branch.then_body.expr, ir.Call)
    element = branch.then_body.expr.args[0].type
    assert isinstance(element, ir.BufferType)
    multi = ir.MultiBufferType(element, 2)
    group = ir.Var("arm_ring", multi, span)

    def ci(value):
        return ir.ConstInt(value, index, span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    allocation = ir.AssignStmt(
        group, call("buffer.alloc_multi", [ir.MakeTuple([], span), ci(8192)], multi), span
    )

    def arm(body, name, slot):
        selected = ir.Var(name, element, span)
        prefix = ir.AssignStmt(selected, call("buffer.slot", [group, ci(slot)], element), span)
        payload = ir.EvalStmt(
            ir.Call(body.expr.op, [selected, *body.expr.args[1:]], body.expr.type, span), span
        )
        return ir.SeqStmts([prefix, *([payload] if kind == "extra_payload" else []), payload], span)

    class Replace(ir.IRMutator):
        def visit_if_stmt(self, op):
            return ir.IfStmt(
                op.condition,
                arm(op.then_body, "first", 0),
                arm(op.else_body, "later", 1 if kind == "different_slot" else 0),
                [],
                span,
            )

    changed = Replace().visit_stmt(original.body)
    function = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions, strict=True)),
        original.return_types,
        ir.SeqStmts([allocation, *sequence_stmts(changed)], span),
        span,
        original.func_type,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    if kind != "equivalent":
        message = "preserve operands" if kind == "different_slot" else "qualified scalar/slot definitions"
        with pytest.raises(ValueError, match=message):
            testing.describe_placed_sync_input(function, True)
        return
    description = testing.describe_placed_sync_input(function, True)
    storage, calls = interpret(function)
    check_imported_accesses(description, storage, calls)
    assert any(
        site["continuation"] is not None for region in description["regions"] for site in region["sites"]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
