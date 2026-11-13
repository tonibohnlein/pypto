# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Supplied scoped order versus an independently expanded selected proof graph."""

from itertools import product

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_control_metrics import expression_nodes, statements
from .sync_matmul_oracle import hardware_reference, interpret
from .sync_physical_oracle import PhysicalExecution, checked_payload_calls
from .test_collective_sync_realization import nested_collective, repeat_collective
from .test_finite_sync_analysis import closure
from .test_nested_sync_repetition import resident_pipeline_tree
from .test_scoped_logical_sync import wrapped_collective
from .test_scoped_token_families import Uses, occurrence_indices
from .test_token_sync_lowering import _apply, _project
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def check_order(logical, *, physical=True):
    result = testing.inspect_scoped_token_families(logical)
    original = result["function"]
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    state = Uses(calls, reference[0])
    state.execute(logical.body, {})
    actual, actual_payload = _project(state.vertices, state.edges, reference[0])
    occurrences = occurrence_indices(result, calls)
    order = testing.analyze_scoped_token_order(logical)
    edges = {((i, False), (i, True)) for i in range(len(calls))}
    latest = {}
    predecessor = {}
    for i, (_, _, pipe, _, _) in enumerate(calls):
        if pipe in latest:
            predecessor[i] = latest[pipe]
            edges.update(((latest[pipe], kind), (i, kind)) for kind in (False, True))
        latest[pipe] = i
    for index, family in enumerate(result["families"]):
        pairs = []
        sr, sv, ss = family["source"]
        tr, tv, ts = family["target"]
        for digits in product(*(range(axis[1]) for axis in family["axes"])):
            source = sv + sum(digit * axis[2] for digit, axis in zip(digits, family["axes"]))
            target = tv + sum(digit * axis[3] for digit, axis in zip(digits, family["axes"]))
            pair = (occurrences[sr, source, ss], occurrences[tr, target, ts])
            assert state.uses[pair] > 0
            pairs.append(pair)
            if order.uniform(index):
                edges.add(((pair[0], True), (pair[1], False)))
        if order.reuse(index):
            assert all(((h[1], True), (g[0], False)) in actual_payload for h, g in zip(pairs, pairs[1:]))
        if pairs and calls[pairs[0][0]][2] == calls[pairs[0][1]][2] and order.barrier(index):
            assert all(predecessor.get(target) == source for source, target in pairs)
    expected = closure(reference[0], edges)
    assert not any(a == b for a, b in expected)
    assert expected <= actual_payload
    for a, ai in occurrences.items():
        for b, bi in occurrences.items():
            for ac, bc in product((False, True), repeat=2):
                assert order.precedes(a, ac, b, bc) == (
                    (ai == bi and ac == bc) or ((ai, ac), (bi, bc)) in expected
                ), (a, ac, b, bc)
    if physical and calls:
        diagnostic = order.realize()
        lowered = _apply(passes.lower_sync, logical)
        # Production adds ownership metadata; commands must remain identical.
        assert ir.python_print(diagnostic.body, format=False) == ir.python_print(lowered.body, format=False)
        _, emitted_calls = checked_payload_calls(original, lowered)
        flags = PhysicalExecution(emitted_calls, reference[0], range(6))
        flags.execute(lowered.body, {})
        graph, projection = _project(flags.vertices, flags.edges, reference[0])
        assert flags.payload_count == len(calls)
        assert not flags.live and not any(flags.pending.values())
        assert len(flags.records) == len(flags.waits)
        assert all(pair in graph for pair in flags.reuses)
        assert projection == actual_payload
        assert reference[2] <= projection
        assert all(ids <= set(range(6)) for ids in flags.ids.values())
        text = codegen.PTOCodegen().generate(ir.Program([lowered], "Scoped", lowered.span))
        assert "sync.record" not in text
    return result, order


@pytest.mark.parametrize(
    "start,step,trips,trim",
    [
        (0, 64, 4, 0),
        (-7, 3, 3, 1),
        (-(1 << 62), 1 << 62, 3, 1),  # Endpoints fit, normalized subtraction does not.
        (7, 3, 3, 1),
        (5, 4, 1, 2),
        (9, 5, 0, 0),
        ((1 << 63) - 22, 3, 3, 1),
    ],
)
def test_original_lattice_bounds_preserve_matching_and_reset(start, step, trips, trim):
    original = wrapped_collective(trips)

    class Bounds(ir.IRMutator):
        def visit_for_stmt(self, op):
            if op.loop_var.name_hint != "outer":
                return super().visit_for_stmt(op)

            def ci(n):
                return ir.ConstInt(n, ir.DataType.INDEX, op.span)

            return ir.ForStmt(
                op.loop_var,
                ci(start),
                ci(start + step * trips - trim),
                ci(step),
                op.iter_args,
                op.body,
                op.return_vars,
                op.span,
            )

    original = Bounds().visit_function(original)
    if start < 0:
        # Import admission is unchanged. Do not broaden it merely to exercise
        # the emitter's signed arithmetic guard.
        with pytest.raises(ValueError, match="nonnegative nonoverflowing"):
            _apply(passes.insert_sync, original)
        return
    logical = _apply(passes.insert_sync, original)
    check_order(logical)
    lowered = _apply(passes.lower_sync, logical)
    # On the qualified original lattice all threshold guards use the IV
    # directly. In particular, last is start+(trips-1)*step, not stop-step.
    for stmt in statements(lowered.body):
        if isinstance(stmt, ir.IfStmt):
            assert not any(isinstance(node, ir.FloorDiv) for node in expression_nodes(stmt.condition))


@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_collective_order_and_flags(rows, visits):
    check_order(_apply(passes.insert_sync, repeat_collective(rows, visits)))


@pytest.mark.parametrize("counts", [(0, 2), (2, 0), (1, 1), (2, 3), (2, 2, 2)])
def test_original_reset_tree(counts):
    check_order(_apply(passes.insert_sync, nested_collective(3, counts)))


@pytest.mark.parametrize("inner,outer,replay", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (2, 2, 2)])
def test_resident_order(inner, outer, replay):
    _, order = check_order(
        _apply(passes.insert_sync, resident_pipeline_tree(inner, outer, replay)),
        physical=bool(inner and outer and replay == 1),
    )
    if replay > 1:
        # Existing DMA same-pipe mechanism limitation; never drop the token
        # merely to obtain a flag plan for a zero-inner-body case.
        with pytest.raises(ValueError, match="same-pipe barrier mechanism"):
            order.realize()


@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_captured_rows_across_original_nonunit_frames(visits):
    check_order(_apply(passes.insert_sync, wrapped_collective(visits)))


def test_shared_summaries_scale_with_count_bits_not_visits():
    entries = []
    for count in [3, 1023, 1048575]:
        logical = _apply(passes.insert_sync, nested_collective(3, (count, 3)))
        order = testing.analyze_scoped_token_order(logical)
        entries.append(order.summary_nodes)
        order.realize()
    assert entries[-1] < 12 * entries[0]


@pytest.mark.parametrize("mode", ["first_active", "last_only", "clipped", "no_seam"])
def test_exceptional_reset_edge_does_not_certify_all_visits(mode):
    logical = _apply(passes.insert_sync, wrapped_collective(4))

    class Restrict(ir.IRMutator):
        def visit_if_stmt(self, op):
            if not op.return_vars and isinstance(op.condition, ir.And):
                if mode == "no_seam":
                    return ir.SeqStmts([], op.span)
                assert isinstance(op.condition.left, ir.Gt)
                comparison = ir.Gt if mode == "clipped" else ir.Eq
                point = comparison(
                    op.condition.left.left,
                    ir.ConstInt(3 if mode in ("first_active", "clipped") else 7, ir.DataType.INDEX, op.span),
                    ir.DataType.BOOL,
                    op.span,
                )
                guard = ir.And(point, op.condition.right, ir.DataType.BOOL, op.span)
                return ir.IfStmt(guard, op.then_body, op.else_body, op.return_vars, op.span)
            return super().visit_if_stmt(op)

    result, order = check_order(Restrict().visit_function(logical), physical=False)
    assert any(not order.reuse(i) for i, f in enumerate(result["families"]) if f["count"] > 1)
    if mode != "no_seam":
        assert any(not order.uniform(i) for i in range(len(result["families"])))
    with pytest.raises(ValueError, match="reuse is unproved"):
        order.realize()
    amended = Restrict().visit_function(logical)
    snapshot = ir.serialize(ir.Program([amended], "Before", amended.span))
    with pytest.raises(ValueError, match="reuse is unproved"):
        _apply(passes.lower_sync, amended)
    assert snapshot == ir.serialize(ir.Program([amended], "Before", amended.span))


def test_duplicate_handoffs_share_proof_not_notification_identity():
    logical = _apply(passes.insert_sync, wrapped_collective(3))

    class Duplicate(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                return ir.SeqStmts([op, op], op.span)
            return op

    before, _ = check_order(logical)
    after, _ = check_order(Duplicate().visit_function(logical))
    assert len(after["families"]) == 2 * len(before["families"])


def test_equal_counts_do_not_share_distinct_reset_orders():
    original = wrapped_collective(3, [[(0, 1), (1, 5)], [(0, 3), (3, 5)]])
    check_order(_apply(passes.insert_sync, original))


def test_same_pipe_certificate_uses_original_native_predecessor():
    original = vec_fixture(count=2)
    assert isinstance(original.body, ir.SeqStmts)
    span = original.span
    prefix, inner, ret = original.body.stmts[:-2], original.body.stmts[-2], original.body.stmts[-1]
    iv = ir.Var("outer", ir.ScalarType(ir.DataType.INDEX), span)
    marker = ir.Var("marker", ir.ScalarType(ir.DataType.INDEX), span)
    bounds = [ir.ConstInt(i, ir.DataType.INDEX, span) for i in (0, 2, 1)]
    body = ir.SeqStmts([ir.AssignStmt(marker, bounds[0], span), inner, ir.YieldStmt([], span)], span)
    outer = ir.ForStmt(iv, *bounds, [], body, [], span)
    nested = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([*prefix, outer, ret], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    result, order = check_order(_apply(passes.insert_sync, nested))
    assert any(order.barrier(i) for i in range(len(result["families"])))


@pytest.mark.parametrize("fixture", ["collective", "captured"])
def test_public_scoped_cpp_and_serialization(tmp_path, fixture):
    original = nested_collective(3, (2, 3)) if fixture == "collective" else wrapped_collective(3)
    logical = _apply(passes.insert_sync, original)
    archive = tmp_path / "tokens.msgpack"
    ir.serialize_to_file(logical, str(archive))
    restored = ir.deserialize_from_file(str(archive))
    assert isinstance(restored, ir.Function)
    check_order(restored)
    lowered = _apply(passes.lower_sync, restored)
    source, output = tmp_path / "scoped.pto", tmp_path / "scoped.cpp"
    source.write_text(codegen.PTOCodegen().generate(ir.Program([lowered], "Scoped", lowered.span)))
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    body = output.read_text().split("AICORE void ", 1)[1]
    assert "set_flag(" in body and "wait_flag(" in body
    assert "ptoas_auto_sync_tail(" not in body
