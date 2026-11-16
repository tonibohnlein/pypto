# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A checked coordinate bijection preserves original payloads and query bindings."""

from itertools import product

import pytest
from pypto import ir, passes, testing

from .sync_matmul_oracle import (
    check_imported_accesses,
    hardware_reference,
    interpret,
    sequence_stmts,
    translate_access_description,
    validate_boundaries,
)
from .sync_physical_oracle import run_physical
from .test_parameterized_token_sync import apply, callers, check, worker

pytestmark = pytest.mark.usefixtures("ascend_backend")


def rotated_dma(launch=8, stop=5, *, unrelated=False, foreign=False, narrow=False):
    """One original loop; accessed rows depend on its actual effective start."""
    span, index = ir.Span.unknown(), ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(v) if isinstance(v, int) else v for v in values], span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    tensor = ir.TensorType([ci(16 * max(launch, stop)), ci(32)], ir.DataType.FP32)
    source, output = (ir.Var(name, tensor, span) for name in ("source", "output"))
    rotation = ir.Var("rotation", ir.ScalarType(index), span)
    block = ir.Var("block", ir.ScalarType(index), span)
    effective = ir.Var("effective", ir.ScalarType(index), span)
    visit = ir.Var("visit", ir.ScalarType(index), span)
    tile_type = ir.BufferType([16, 32], ir.DataType.FP32, ir.MemorySpace.Vec)
    tile = ir.Var("tile", tile_type, span)
    offset = ir.Var("rotation", rotation.type, span) if foreign else rotation
    addition = ir.Add(block, offset, index, span)
    coordinate = ir.FloorMod(addition, ci(launch), index, span)
    if narrow:
        coordinate = ir.Cast(
            ir.FloorMod(
                ir.Add(
                    ir.Cast(block, ir.DataType.INT32, span),
                    ir.Cast(rotation, ir.DataType.INT32, span),
                    ir.DataType.INT32,
                    span,
                ),
                ir.ConstInt(launch, ir.DataType.INT32, span),
                ir.DataType.INT32,
                span,
            ),
            index,
            span,
        )
    row = ir.Mul(block if unrelated else visit, ci(16), index, span)
    body = ir.SeqStmts(
        [
            ir.EvalStmt(call("buffer.load", [source, tup([row, 0]), tup([16, 32]), tile]), span),
            ir.EvalStmt(call("buffer.store", [tile, tup([row, 0]), tup([16, 32]), output]), span),
        ],
        span,
    )
    loop = ir.ForStmt(visit, effective, ci(stop), ci(launch), [], body, [], span)
    function = ir.Function(
        "rotated_dma",
        [source, (output, ir.ParamDirection.InOut), rotation],
        [tensor],
        ir.SeqStmts(
            [
                ir.AssignStmt(tile, call("buffer.alloc", [tup([]), ci(0)], tile_type), span),
                ir.AssignStmt(block, call("buffer.get_block_idx", [], block.type), span),
                ir.AssignStmt(effective, coordinate, span),
                loop,
                ir.ReturnStmt([output], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return function, loop


def analyze(function, launch=8, bounds=(0, 7), *, blocks=None, count=None):
    return testing.analyze_sync_invocation(
        function,
        True,
        "a2a3",
        {2: bounds},
        (0, launch - 1) if blocks is None else blocks,
        launch_count=launch if count is None else count,
    )


def assert_queries(original, selected, coordinates, stop, trace):
    """Original byte-access sets supply selectors, independent of region names."""
    effective, original_block, rotation = coordinates
    storage, calls = trace
    description = selected.describe()
    # Translate only claimed model geometry using its effective coordinate.
    # Concrete accesses and public queries continue to use the original worker
    # and rotation valuation; neither oracle specializes the retained recipe.
    translate_access_description(description, {2: rotation}, effective)
    occurrences, effects = check_imported_accesses(description, storage, calls, {2: rotation})
    assert bool(occurrences) == (effective < stop)
    vertices, _, order, covers = hardware_reference(original, calls)
    if description.get("hierarchical"):
        retained = set(selected.expand_retained_demands(len(calls)))
    else:
        retained = set(description["crossing_demands"])
        for region_index, region in enumerate(description["regions"]):
            for source, target, distance in region["demands"]:
                for visit in range(region["count"] - distance):
                    retained.add(((region_index, visit, source), (region_index, visit + distance, target)))
    assert retained == {(occurrences[a], occurrences[b]) for a, b in covers}
    for a, ac in vertices:
        for b, bc in vertices:
            assert selected.precedes(occurrences[a], ac, occurrences[b], bc) == (((a, ac), (b, bc)) in order)
    checks = validate_boundaries(
        description, storage, effects, occurrences, selected, {2: rotation}, original_block
    )
    assert checks > 0


@pytest.mark.parametrize("launch,stop", [(8, 5), (24, 17)])
def test_coordinate_queries_and_physical_trace_use_original_valuation(launch, stop):
    original, loop = rotated_dma(launch, stop)
    owner = analyze(original, launch, (0, launch - 1))
    description = owner.describe()
    assert description["effective_coordinate"].same_as(loop.start)
    assert [c["block_bounds"] for c in description["cases"]] == [(0, stop - 1), (stop, launch - 1)]
    physical = owner.realize_physical("a2a3", [2, 5])["function"]
    for rotation, block in product((0, 1, launch - 1), range(launch)):
        selected = owner.select_case(block, {2: rotation})
        assert selected is not None
        effective = (block + rotation) % launch
        trace = interpret(original, {2: rotation}, block_index=block)
        assert len(trace[1]) == (2 if effective < stop else 0)
        assert_queries(original, selected, (effective, block, rotation), stop, trace)
        run_physical(original, physical, range(2, 6, 3), {2: rotation}, block_index=block)
    # Retained owners never guess a missing rotation or choose a representative
    # original worker for an effective case.
    assert owner.select_case(0, {}) is None
    assert owner.select_case(-1, {2: 0}) is None
    assert owner.select_case(launch, {2: 0}) is None
    assert owner.select_case(0, {2: launch}) is None
    active = owner.select_case(0, {2: 0})
    assert active is not None
    assert active.boundary(0, 0, 0, {}, 0) is None
    assert active.boundary(0, 0, 0, {2: 0}) is None
    assert active.boundary(0, 0, 0, {2: 0}, launch - 1) is None


def test_public_saved_pipeline_collects_actual_cardinality(tmp_path):
    original, loop = rotated_dma()
    program = callers(original, starts=(0, 7), blocks=8)
    evidence = testing.sync_invocation_evidence(program)
    assert evidence["block_counts"][original.name] == (8, 8)
    assert evidence["parameters"][original.name][2] == (0, 7)
    logical_program = apply(passes.insert_sync, program)
    logical = worker(logical_program, original.name)
    archive = tmp_path / "effective.msgpack"
    ir.serialize_to_file(logical_program, str(archive))
    saved = ir.deserialize_from_file(str(archive))
    assert isinstance(saved, ir.Program)
    physical = worker(apply(passes.lower_sync, apply(passes.optimize_sync, saved)), original.name)
    for rotation, block in product((0, 1, 7), range(8)):
        check(original, logical, physical, {2: rotation}, block=block)
    # The original payload IV/start are retained, not replaced by a chosen
    # numerical representative or a rewritten global block lookup.
    logical_loops = [s for s in sequence_stmts(logical.body) if isinstance(s, ir.ForStmt)]
    assert len(logical_loops) == 1
    assert logical_loops[0].loop_var.same_as(loop.loop_var)
    assert logical_loops[0].start.same_as(loop.start)
    with pytest.raises(ValueError, match="cardinality"):
        testing.lower_sync_schedule(logical, parameter_bounds={2: (0, 7)}, block_bounds=(0, 7))


@pytest.mark.parametrize("count,blocks", [(None, (0, 7)), (7, (0, 7)), (9, (0, 7)), (8, (1, 7))])
def test_effective_hull_is_not_actual_launch_cardinality(count, blocks):
    original, _ = rotated_dma()
    with pytest.raises(ValueError, match="independent full launch cardinality"):
        testing.analyze_sync_invocation(original, True, "a2a3", {2: (0, 7)}, blocks, launch_count=count)


@pytest.mark.parametrize("edit", ["unrelated", "foreign", "narrow"])
def test_effective_coordinate_does_not_reinterpret_other_original_values(edit):
    original, _ = rotated_dma(**{edit: True})
    with pytest.raises(ValueError, match="unrelated original block|ready invariant|proved fixed"):
        analyze(original)


def test_original_addition_overflow_is_not_hidden_by_remainder():
    original, _ = rotated_dma()
    with pytest.raises(ValueError, match="overflow its original addition"):
        analyze(original, bounds=(0, (1 << 63) - 1))


def test_foreign_loop_frame_and_unknown_caller_do_not_gain_evidence():
    original, loop = rotated_dma()
    foreign = ir.Var(loop.loop_var.name_hint, loop.loop_var.type, loop.span)

    class ForeignFrame(ir.IRMutator):
        def visit_var(self, op):
            return foreign if op.same_as(loop.loop_var) else op

    changed_loop = ir.ForStmt(
        loop.loop_var,
        loop.start,
        loop.stop,
        loop.step,
        loop.iter_args,
        ForeignFrame().visit_stmt(loop.body),
        loop.return_vars,
        loop.span,
    )
    statements = sequence_stmts(original.body)
    changed = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([*statements[:-2], changed_loop, statements[-1]], original.span),
        original.span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    with pytest.raises(ValueError, match="unavailable|Unavailable|scope|coordinate"):
        analyze(changed)
    unknown = callers(original, starts=(0, 7), unknown=True, blocks=8)
    evidence = testing.sync_invocation_evidence(unknown)
    assert 2 not in evidence["parameters"].get(original.name, {})
    with pytest.raises(ValueError, match="fixed|range"):
        apply(passes.insert_sync, unknown)


def test_non_singleton_actual_launch_count_refuses_permutation():
    original, _ = rotated_dma()
    program = callers(original, starts=(0, 7), blocks=8)
    wrapper, main = worker(program, "launch"), worker(program, "main")
    count = ir.ConstInt(9, ir.DataType.INDEX, original.span)
    other = ir.Function(
        "other_launch",
        list(zip(wrapper.params, wrapper.param_directions)),
        wrapper.return_types,
        wrapper.body,
        wrapper.span,
        wrapper.func_type,
        attrs={"core_num": count},
    )
    call = ir.EvalStmt(
        ir.Call(
            ir.GlobalVar(other.name),
            [*main.params, ir.ConstInt(7, ir.DataType.INDEX, original.span)],
            original.return_types[0],
            original.span,
        ),
        original.span,
    )
    body = sequence_stmts(main.body)
    caller = ir.Function(
        main.name,
        main.params,
        main.return_types,
        ir.SeqStmts([*body[:-1], call, body[-1]], main.span),
        main.span,
        main.func_type,
    )
    varying = ir.Program([original, wrapper, other, caller], "VaryingLaunch", original.span)
    evidence = testing.sync_invocation_evidence(varying)
    assert evidence["block_counts"][original.name] == (8, 9)
    assert evidence["block_indices"][original.name] == (0, 8)
    with pytest.raises(ValueError, match="independent full launch cardinality"):
        apply(passes.insert_sync, varying)


def test_partition_size_does_not_enumerate_workers_or_rotation_values():
    work = []
    for launch in (8, 24, 1024):
        original, _ = rotated_dma(launch, launch - 1)
        description = analyze(original, launch, (0, launch - 1)).describe()
        assert len(description["cases"]) == 2
        work.append((description["coordinate_nodes"], description["coordinate_pieces"]))
    # These named counters measure analytic partition work, not all compiler
    # work or allocations. The growing launch domain supplies no extra pieces.
    assert len(set(work)) == 1


@pytest.mark.parametrize("stop", [9, 16, 19])
def test_rotated_multi_trip_covers_queries_selectors_and_physical_reuse(stop):
    launch = 8
    original, _ = rotated_dma(launch, stop)
    owner = analyze(original, launch, (0, launch - 1))
    physical = owner.realize_physical("a2a3", [2, 5])["function"]
    for rotation, block in product((0, 1, 7), range(launch)):
        selected = owner.select_case(block, {2: rotation})
        assert selected is not None
        effective = (block + rotation) % launch
        trace = interpret(original, {2: rotation}, block_index=block)
        assert len(trace[1]) == 2 * len(range(effective, stop, launch))
        assert_queries(original, selected, (effective, block, rotation), stop, trace)
        # The actual original invocation executes all visits, including the
        # final short orbit, under the same retained physical assignment.
        run_physical(original, physical, range(2, 6, 3), {2: rotation}, block_index=block)


def with_body(function, statements):
    return ir.Function(
        function.name,
        list(zip(function.params, function.param_directions)),
        function.return_types,
        ir.SeqStmts(statements, function.span),
        function.span,
        function.func_type,
        ir_stage=function.ir_stage,
    )


def test_coordinate_preflight_shares_aliases_and_ignores_unneeded_scalar_dags():
    original, loop = rotated_dma()
    setup = list(sequence_stmts(original.body)[:-2])
    alias = loop.start
    for i in range(512):
        following = ir.Var(f"alias{i}", alias.type, original.span)
        setup.append(ir.AssignStmt(following, alias, original.span))
        alias = following
    loops = [
        ir.ForStmt(
            ir.Var(f"visit{i}", loop.loop_var.type, original.span),
            alias,
            loop.stop,
            loop.step,
            [],
            ir.SeqStmts([], original.span),
            [],
            original.span,
        )
        for i in range(512)
    ]
    predicate = ir.ConstBool(True, original.span)
    for _ in range(28):
        predicate = ir.And(predicate, predicate, ir.DataType.BOOL, original.span)
    unrelated = ir.IfStmt(predicate, ir.SeqStmts([], original.span), None, [], original.span)
    function = with_body(original, [*setup, *loops, unrelated, sequence_stmts(original.body)[-1]])
    # The deliberately missing independent launch certificate stops before
    # access extraction. Discovery itself must process distinct definitions,
    # loops and alias links, not unfold the shared Boolean expression tree.
    with pytest.raises(ValueError, match="independent full launch cardinality"):
        testing.analyze_sync_invocation(function, True, "a2a3", {2: (0, 7)}, (0, 7), launch_count=None)


@pytest.mark.parametrize("kind", ["future", "cycle"])
def test_global_alias_memo_never_establishes_lexical_readiness(kind):
    original, loop = rotated_dma()
    span = original.span
    setup = list(sequence_stmts(original.body)[:-2])
    effective = setup[-1]
    assert isinstance(effective, ir.AssignStmt)
    if kind == "future":
        late = ir.Var("future_addition", loop.start.type, span)
        remainder = effective.value
        assert isinstance(remainder, ir.FloorMod)
        setup[-1] = ir.AssignStmt(
            effective.var, ir.FloorMod(late, remainder.right, ir.DataType.INDEX, span), span
        )
        setup.append(ir.AssignStmt(late, remainder.left, span))
        start = loop.start
        reason = "dominating original definitions"
    else:
        first = ir.Var("cycle_first", loop.start.type, span)
        second = ir.Var("cycle_second", loop.start.type, span)
        setup.extend([ir.AssignStmt(first, second, span), ir.AssignStmt(second, first, span)])
        start = first
        reason = "Cyclic invocation coordinate"
    changed_loop = ir.ForStmt(loop.loop_var, start, loop.stop, loop.step, [], loop.body, [], span)
    function = with_body(original, [*setup, changed_loop, sequence_stmts(original.body)[-1]])
    with pytest.raises(ValueError, match=reason):
        analyze(function)
