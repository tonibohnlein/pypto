# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scoped order certificates versus independently interpreted actual tokens."""

from itertools import product

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_control_metrics import expression_nodes, statements
from .sync_matmul_oracle import hardware_reference, interpret
from .sync_physical_oracle import PhysicalExecution, checked_payload_calls, run_physical
from .sync_shenggan_cases import capture_shenggan
from .test_collective_sync_realization import nested_collective, repeat_collective
from .test_finite_sync_analysis import closure
from .test_nested_phase_sync import nested_matrix_fixture
from .test_nested_sync_repetition import resident_pipeline_tree
from .test_scoped_logical_sync import wrapped_collective
from .test_scoped_palette_realization import check_physical
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
                uniform_answer = (ai == bi and ac == bc) or ((ai, ac), (bi, bc)) in expected
                # Exceptional/periodic tokens may provide a concrete native ->
                # token -> native witness even when they cannot be installed in
                # the uniform repeat summary. Derive that witness from executed
                # SSA uses, independently of the parser's family progression.
                one_edge = any(
                    calls[source][2] == calls[ai][2]
                    and calls[target][2] == calls[bi][2]
                    and ai <= source
                    and target <= bi
                    for source, target in state.uses
                )
                answer = order.precedes(a, ac, b, bc)
                assert answer == (uniform_answer or one_edge), (a, ac, b, bc)
                assert not answer or (ai == bi and ac == bc) or ((ai, ac), (bi, bc)) in actual_payload
    if physical and calls:
        diagnostic = order.realize()
        lowered = _apply(passes.lower_sync, logical)

        # Whole-kernel ownership adds a terminal drain; child lowering must
        # retain pending payloads. Permit only that precise boundary difference.
        class TerminalDrain(ir.IRMutator):
            removed = 0

            def visit_seq_stmts(self, op):
                updated = super().visit_seq_stmts(op)
                assert isinstance(updated, ir.SeqStmts)
                kept = []
                for i, child in enumerate(updated.stmts):
                    if (
                        isinstance(child, ir.EvalStmt)
                        and isinstance(child.expr, ir.Call)
                        and child.expr.op.name == ir.get_op("system.bar_all").name
                    ):
                        assert i + 1 < len(updated.stmts)
                        assert isinstance(updated.stmts[i + 1], ir.ReturnStmt)
                        self.removed += 1
                    else:
                        kept.append(child)
                return ir.SeqStmts(kept, updated.span)

        boundary = TerminalDrain()
        internal = boundary.visit_function(lowered)
        assert ir.python_print(diagnostic.body, format=False) == ir.python_print(internal.body, format=False)
        _, emitted_calls = checked_payload_calls(original, lowered)
        flags = PhysicalExecution(emitted_calls, reference[0], range(6))
        flags.execute(lowered.body, {})
        graph, projection = _project(flags.vertices, flags.edges, reference[0])
        assert flags.payload_count == len(calls)
        assert boundary.removed == bool(flags.records and flags.waits)
        if boundary.removed:
            assert flags.exits and all(
                ((p, True), exit_node) in graph for p in range(len(calls)) for exit_node in flags.exits
            )
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
        order.realize_legacy()
    amended = Restrict().visit_function(logical)
    snapshot = ir.serialize(ir.Program([amended], "Before", amended.span))
    # The period-one premise remains unproved. A complete directional palette
    # can nevertheless certify the actual exceptional protocol with more lanes.
    # This deliberate removal of token waits is not a storage-coverage claim.

    if mode == "no_seam":
        # This intentionally weakened protocol remains uncertified in the
        # explicit six-ID pool. Its existing default eight-ID configuration is
        # tested separately; no production resource pool is enlarged.
        with pytest.raises(ValueError, match="reuse is unproved"):
            order.realize(eligible_ids=list(range(6)))
    else:
        check_physical(amended, range(6), require_coverage=False)
    lowered = _apply(passes.lower_sync, amended)
    check_physical(amended, range(8), require_coverage=False, emitted=lowered)
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


@pytest.mark.parametrize("count,visits", [(1, 1), (2, 1), (3, 2)])
def test_periodic_token_order_witnesses_are_sound_across_resets(count, visits):
    original = nested_matrix_fixture(count, visits=visits, slices=4, tile_columns=2)
    logical = testing.analyze_placed_sync(original, True).realize()["function"]
    # This independently checks all four event combinations and every accepted
    # reuse certificate against executed tokens. Physical acceptance is separate.
    _, order = check_order(logical, physical=False)
    assert order.witness_attempts > 0
    assert order.candidate_visits > 0
    assert order.binary_probes >= order.candidate_visits


@pytest.mark.parametrize("count,visits", [(0, 2), (1, 0), (1, 1), (1, 2), (2, 2), (3, 3), (5, 2)])
def test_nested_phase_physical_matching_and_reuse(count, visits):
    original = nested_matrix_fixture(count, visits=visits, tile_columns=2)
    logical = testing.analyze_placed_sync(original, True).realize()["function"]
    _, order = check_order(logical)
    if count and visits:
        # Explicit requested pool, independent of the current native catalog.
        physical = order.realize(eligible_ids=list(range(6)))

        run_physical(original, physical, eligible=range(6))


def test_shenggan_original_tiles_single_invocation_cpp(tmp_path):
    # Bounded source derivative: original tile sizes and four inner slices.
    # This does not stand in for the full original kernel's resource outcome.
    program = capture_shenggan(tmp_path / "capture", m=128, n=256, k=512, workers=1)
    with passes.PassContext(
        [],
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        logical = passes.insert_sync()(program)
        physical = passes.lower_sync()(logical)
    worker = next(f for f in logical.functions.values() if f.func_type == ir.FunctionType.AIC)
    before = testing.inspect_scoped_token_families(worker, block_bounds=(0, 0))["function"]
    after = next(f for f in physical.functions.values() if f.func_type == ir.FunctionType.AIC)
    # Full program supplies the fixed launch count; use its resulting worker
    # for physical checks with explicit worker zero.
    run_physical(before, after, eligible=range(6), block_index=0)
    text = codegen.PTOCodegen().generate(ir.Program([after], "Shenggan", after.span))
    path = tmp_path / "shenggan.pto"
    path.write_text(text)
    cpp = tmp_path / "shenggan.cpp"
    _run_ptoas(str(path), str(cpp), ["--enable-insert-sync=true"])
    assert cpp.exists() and cpp.stat().st_size > 0


def test_shenggan_collapsed_single_k_startup_remains_explicit(tmp_path):
    program = capture_shenggan(tmp_path / "capture", m=128, n=256, k=256, workers=1)
    before = ir.serialize(program)
    # Upstream specialization removes the K frame. The older rotating-leaf
    # route does not yet export its initialization/continuation macro.
    with passes.PassContext(
        [],
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        with pytest.raises(ValueError, match="accumulator/continuation export unavailable"):
            passes.insert_sync()(program)
    assert ir.serialize(program) == before


@pytest.mark.parametrize("count,visits", [(2, 1), (3, 2)])
def test_resource_transition_queries_use_actual_tokens(count, visits):
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(count, visits=visits, slices=4, tile_columns=2), True
    ).realize()["function"]
    result, order = check_order(logical, physical=False)
    for family in result["families"]:
        # The parser's family is an independently checked rectangle of actual
        # token edges. Asking about that rectangle must not install another edge.
        axes = []
        sr, _, _ = family["source"]
        tr, _, _ = family["target"]
        source_frames = result["regions"][sr]["frames"]
        target_frames = result["regions"][tr]["frames"]
        for i, (frame, length, source_stride, target_stride) in enumerate(family["axes"]):
            source_unit = 1
            target_unit = 1
            for child in source_frames[i + 1 :]:
                source_unit *= child[1]
            for child in target_frames[i + 1 :]:
                target_unit *= child[1]
            axes.append(
                (
                    order.region_frames[sr][i],
                    length,
                    source_stride,
                    target_stride,
                    source_stride // source_unit,
                    target_stride // target_unit,
                )
            )
        assert order.prove_transition(family["source"], family["target"], axes)
        assert not order.prove_transition(family["target"], family["source"], [])
        if axes:
            corrupted = list(axes)
            frame, length, source_stride, target_stride, source_step, target_step = corrupted[0]
            corrupted[0] = (frame, length, source_stride, target_stride, source_step + 1, target_step)
            with pytest.raises(ValueError, match="transition step"):
                order.prove_transition(family["source"], family["target"], corrupted)
    # Cross-record resource questions are distinct from the supplied token edges.
    # Point answers agree with the separately checked actual-token order view.
    for previous, following in product(result["families"][:8], repeat=2):
        consumer, producer = previous["target"], following["source"]
        expected = order.precedes(consumer, True, producer, False)
        assert order.prove_transition(consumer, producer, []) == expected
        assert order.precedes(consumer, True, producer, False) == expected
