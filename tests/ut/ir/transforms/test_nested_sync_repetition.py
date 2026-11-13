# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical nested repetition preserves original frames without unfolding."""

from itertools import product

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import sequence_stmts, validate
from .sync_physical_oracle import run_physical
from .test_sync_range_proofs import caller_program, range_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
BOUNDS = {1: (0, 16), 3: (64, 128)}


def nested_fixture(counts, repeats=1, moving=False, parent_consumers=True, outer_moving=False):
    original = range_fixture()
    span, index = original.span, ir.DataType.INDEX
    assert isinstance(original.body, ir.SeqStmts)
    alloc, load, store, ret = sequence_stmts(original.body)
    assert isinstance(alloc, ir.AssignStmt)
    assert isinstance(load, ir.EvalStmt) and isinstance(store, ir.EvalStmt)

    def ci(n):
        return ir.ConstInt(n, index, span)

    def clone(stmt):
        call = stmt.expr
        return ir.EvalStmt(ir.Call(call.op, call.args, call.type, span), span)

    # Parent accesses surround the child. The prefix initializes local storage,
    # including zero-trip executions, and the suffix leaves payload work pending.
    def consume():
        tile = alloc.var
        return ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [tile, tile], ir.VoidType(), span), span)

    body: list[ir.Stmt] = [alloc, clone(load)]
    if parent_consumers:
        body.append(consume())
    for invocation in range(repeats):
        variables = [
            ir.Var(f"iv_{invocation}_{depth}", ir.ScalarType(index), span) for depth in range(len(counts))
        ]
        child_load = clone(load)
        if moving or outer_moving:
            call = child_load.expr
            assert isinstance(call, ir.Call)
            assert isinstance(call.args[1], ir.MakeTuple)
            args = list(call.args)
            args[1] = ir.MakeTuple(
                [
                    ir.Add(
                        call.args[1].elements[0],
                        (variables[0] if outer_moving else variables[-1]),
                        index,
                        span,
                    ),
                    ci(0),
                ],
                span,
            )
            child_load = ir.EvalStmt(ir.Call(call.op, args, call.type, span), span)
        child = ir.SeqStmts([child_load, clone(store), ir.YieldStmt([], span)], span)
        for depth in reversed(range(len(counts))):
            # Distinct, nonzero/nonunit coordinates make accidental reuse of
            # the innermost counter as the global ordinal observable.
            start, step = depth + 1, depth + 2
            loop = ir.ForStmt(
                variables[depth], ci(start), ci(start + counts[depth] * step), ci(step), [], child, [], span
            )
            child = ir.SeqStmts([loop, ir.YieldStmt([], span)], span)
        body.append(loop)
        if parent_consumers:
            body.append(consume())
        body.append(clone(store))
    body.append(ret)
    return ir.Function(
        "nested_dma",
        [(p, d) for p, d in zip(original.params, original.param_directions)],
        original.return_types,
        ir.SeqStmts(body, span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize("outer,inner", [(0, 3), (2, 0), (1, 1), (1, 3), (2, 1), (2, 3)])
@pytest.mark.parametrize("repeats", [1, 2])
def test_nested_repetition_covers_boundaries_and_physical_reuse(outer, inner, repeats):
    function = nested_fixture([outer, inner], repeats)
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    before = retained.describe()
    parameters = {1: 16, 3: 64}
    validate(function, retained, parameters=parameters)
    physical = retained.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6), parameters=parameters)
    after = retained.describe()
    for counter in ("scoped_export_builds", "scoped_reuse_queries", "hierarchy_query_steps"):
        assert after.pop(counter) >= before.pop(counter)
    assert after == before
    regions = [r for r in before["regions"] if r["nested_frames"]]
    assert len(regions) == repeats
    assert all(r["count"] == outer * inner and len(r["nested_frames"]) == 2 for r in regions)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        logical = passes.insert_sync()(caller_program(function))
        lowered = passes.lower_sync()(logical)
    worker = next(f for f in lowered.functions.values() if f.name == function.name)
    run_physical(function, worker, range(6), parameters=parameters)


def test_three_nested_frames_and_production_selection():
    function = nested_fixture([2, 2, 2])
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    validate(function, retained, parameters={1: 0, 3: 128})
    physical = retained.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6), parameters={1: 0, 3: 128})
    program = caller_program(function)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        synchronized = passes.realize_sync()(program)
    result = next(f for f in synchronized.functions.values() if f.name == function.name)
    assert dict(result.attrs)["pypto_autosync"]
    run_physical(function, result, range(6), parameters={1: 16, 3: 128})


def test_nested_trip_growth_does_not_expand_retained_analysis():
    sizes = []
    for counts in [[2, 3], [2, 1000], [1000, 3], [1000000, 1000000]]:
        retained = testing.analyze_placed_sync(nested_fixture(counts), True, parameter_bounds=BOUNDS)
        d = retained.describe()
        physical = retained.realize_physical("a2a3", list(range(6)))
        sizes.append(
            (
                d["analyzed_occurrences"],
                d["cells"],
                [(len(r["sites"]), len(r["demands"]), r["distance_entries"]) for r in d["regions"]],
                physical["set_sites"],
                physical["wait_sites"],
                physical["reserved_lanes"],
            )
        )
    assert all(size == sizes[0] for size in sizes)


@pytest.mark.parametrize("counts", [[0, 3], [2, 0], [1, 3], [2, 3], [2, 2, 3]])
def test_inner_moving_reads_preserve_point_membership_across_resets(counts):
    function = nested_fixture(counts, moving=True)
    result = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    validate(function, result, parameters={1: 0, 3: 64})
    region = next(r for r in result.describe()["regions"] if r["nested_frames"])
    assert region["reset_period"] == counts[-1]
    physical = result.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6), parameters={1: 0, 3: 64})


def test_nested_ordinal_product_must_fit_index():
    with pytest.raises(ValueError, match="nonnegative nonoverflowing index arithmetic"):
        testing.analyze_placed_sync(nested_fixture([2**32, 2**32]), True, parameter_bounds=BOUNDS)


@pytest.mark.ptoas_compile
def test_nested_repetition_cpp_preserves_original_loops(tmp_path):
    function = nested_fixture([2, 3])
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    physical = retained.realize_physical("a2a3", list(range(6)))
    program = ir.Program([physical["function"]], "NestedDMA", function.span)
    source, output = tmp_path / "nested.pto", tmp_path / "nested.cpp"
    source.write_text(codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False))
    assert source.read_text().count("scf.for ") == 2
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    assert "set_flag(" in output.read_text() and "wait_flag(" in output.read_text()


def test_nested_mathematics_does_not_hide_unavailable_dma_barrier():
    function = nested_fixture([2, 3], parent_consumers=False)
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    validate(function, retained, parameters={1: 0, 3: 64})
    before = retained.describe()
    with pytest.raises(ValueError, match="same-pipe barrier mechanism"):
        retained.realize_physical("a2a3", list(range(6)))
    after = retained.describe()
    assert after.pop("scoped_export_builds") == 1
    assert before.pop("scoped_export_builds") == 0
    assert after == before


def test_outer_coordinate_access_uses_original_nested_interface():
    function = nested_fixture([2, 3], outer_moving=True)
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    validate(function, retained, parameters={1: 0, 3: 128})


def test_accumulator_reset_is_not_global_first_visit():
    original = staged_fixture(2, stages=1)
    span, index = original.span, ir.DataType.INDEX

    class FixedOperands(ir.IRMutator):
        def visit_call(self, op):
            call = op
            if call.op.name != ir.get_op("buffer.load").name:
                return call
            args = list(call.args)
            args[1] = ir.MakeTuple([ir.ConstInt(0, index, span), ir.ConstInt(0, index, span)], span)
            return ir.Call(call.op, args, call.type, span)

    original = FixedOperands().visit_function(original)
    assert isinstance(original.body, ir.SeqStmts)
    body = []
    for stmt in sequence_stmts(original.body):
        if isinstance(stmt, ir.ForStmt):
            outer = ir.Var("outer", ir.ScalarType(index), span)
            stmt = ir.ForStmt(
                outer,
                ir.ConstInt(0, index, span),
                ir.ConstInt(2, index, span),
                ir.ConstInt(1, index, span),
                [],
                ir.SeqStmts([stmt, ir.YieldStmt([], span)], span),
                [],
                span,
            )
        body.append(stmt)
    function = ir.Function(
        original.name,
        [(p, d) for p, d in zip(original.params, original.param_directions)],
        original.return_types,
        ir.SeqStmts(body, span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    retained = testing.analyze_placed_sync(function, True)
    validate(function, retained)
    physical = retained.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6))


def test_empty_middle_frame_preserves_parent_work():
    function = nested_fixture([2, 0, 3])
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    validate(function, retained, parameters={1: 0, 3: 64})
    physical = retained.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6), parameters={1: 0, 3: 64})


def test_original_frame_work_and_resource_limits_with_depth():
    measurements = []
    for depth in [2, 4, 8, 16]:
        function = nested_fixture([2] * depth)
        retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
        description = retained.describe()
        logical = retained.realize()
        region = next(r for r in description["regions"] if r["nested_frames"])
        assert len(region["nested_frames"]) == depth
        measurements.append(
            (depth, description["source_coordinate_entries"], description["source_region_nodes"])
        )
        assert logical["record_sites"] == 5
        assert logical["wait_sites"] == depth + 5
        if depth <= 4:
            physical = retained.realize_physical("a2a3", list(range(6)))
            run_physical(function, physical["function"], range(6), parameters={1: 0, 3: 128})
        else:
            # The original-frame allocator does not certify these deep nests.
            # A former flattened diagnostic assignment was not production evidence.
            with pytest.raises(ValueError, match="Allocation unproved by whole-span reservation"):
                retained.realize_physical("a2a3", list(range(6)))
    assert all(entries == depth + 5 and nodes == depth + 3 for depth, entries, nodes in measurements)


def resident_read_nest(inner, outer, refill=True):
    """Whole L1 fills surround sliced reads; L0/Acc still have internal writes.

    The first/last reader of a queried point can occur at an interior inner
    coordinate, and that coordinate resets at each outer visit.
    """
    span, fp, index = ir.Span.unknown(), ir.DataType.FP32, ir.DataType.INDEX
    width = max(1, inner) * 32

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(v) if isinstance(v, int) else v for v in values], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    a = ir.Var("a", ir.TensorType([32, width], fp), span)
    b = ir.Var("b", ir.TensorType([width, 32], fp), span)
    out = ir.Var("out", ir.TensorType([32, 32], fp), span)
    definitions = []

    def allocation(name, shape, space, address):
        ty = ir.BufferType(
            shape,
            fp,
            space,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024 if space == ir.MemorySpace.Acc else 512,
        )
        var = ir.Var(name, ty, span)
        definitions.append(ir.AssignStmt(var, call("buffer.alloc", [tup([]), ci(address)], ty), span))
        return var

    a1 = allocation("a1", [32, width], ir.MemorySpace.Mat, 0)
    b1 = allocation("b1", [width, 32], ir.MemorySpace.Mat, 32 * width * 4)
    left = allocation("left", [32, 32], ir.MemorySpace.Left, 0)
    right = allocation("right", [32, 32], ir.MemorySpace.Right, 0)
    acc = allocation("acc", [32, 32], ir.MemorySpace.Acc, 0)
    j = ir.Var("reader", ir.ScalarType(index), span)
    v = ir.Var("visit", ir.ScalarType(index), span)
    offset = ir.Mul(j, ci(32), index, span)
    body = ir.SeqStmts(
        [
            payload("buffer.extract", [a1, ci(0), offset, left]),
            payload("buffer.extract", [b1, offset, ci(0), right]),
            payload("buffer.matmul", [left, right, acc]),
            ir.YieldStmt([], span),
        ],
        span,
    )
    loop = ir.ForStmt(j, ci(0), ci(inner), ci(1), [], body, [], span)
    nest = ir.ForStmt(
        v, ci(0), ci(outer), ci(1), [], ir.SeqStmts([loop, ir.YieldStmt([], span)], span), [], span
    )

    def fills():
        return [
            payload("buffer.load", [a, tup([0, 0]), tup([32, width]), a1]),
            payload("buffer.load", [b, tup([0, 0]), tup([width, 32]), b1]),
        ]

    statements: list[ir.Stmt] = [*definitions, *fills(), nest]
    if inner and outer:
        statements.append(payload("buffer.store", [acc, tup([0, 0]), tup([32, 32]), out]))
    if refill:
        statements.extend(fills())
    statements.append(ir.ReturnStmt([out], span))
    return ir.Function(
        "resident_nest",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("inner,outer", [(0, 2), (2, 0), (1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("refill", [False, True])
def test_resident_read_boxes_and_local_writes(inner, outer, refill):
    function = resident_read_nest(inner, outer, refill)
    retained = testing.analyze_placed_sync(function, True)
    validate(function, retained)
    # Different slices have different point answers; all reads nevertheless
    # conflict with each whole fill. Empty visits contribute no reader.
    d = retained.describe()
    domain = next(i for i, storage in enumerate(d["storage"]) if storage["name"] == "a1")
    for part in range(max(1, inner)):
        boundary = retained.boundary(domain, 0, part * 32)
        assert boundary is not None
        readers = boundary["last_readers"]
        if refill or not inner or not outer:
            assert not any(readers)
        else:
            region = next(i for i, r in enumerate(d["regions"]) if r["nested_frames"])
            assert readers[int(ir.PipeType.MTE1)] == (region, (outer - 1) * inner + part, 0)


def test_resident_outer_repetition_has_constant_retained_size():
    sizes = []
    for outer in [1, 2, 1000, 1000000]:
        retained = testing.analyze_placed_sync(resident_read_nest(4, outer), True)
        d = retained.describe()
        sizes.append(
            (
                d["analyzed_occurrences"],
                d["cells"],
                d["boundary_graph_edges"],
                sum(len(r["demands"]) for r in d["regions"]),
            )
        )
    assert all(size == sizes[0] for size in sizes)


def resident_pipeline_tree(inner, outer, replay):
    """Interleave fills, a resident nested child, store and refill at each reset."""
    original = resident_read_nest(inner, outer)
    span, index = original.span, ir.DataType.INDEX
    allocations, body = [], []
    for statement in sequence_stmts(original.body)[:-1]:
        (allocations if isinstance(statement, ir.AssignStmt) else body).append(statement)
    iv = ir.Var("replay", ir.ScalarType(index), span)
    loop = ir.ForStmt(
        iv,
        ir.ConstInt(2, index, span),
        ir.ConstInt(2 + replay * 3, index, span),
        ir.ConstInt(3, index, span),
        [],
        ir.SeqStmts([*body, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return ir.Function(
        "resident_pipeline_tree",
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([*allocations, loop, sequence_stmts(original.body)[-1]], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize(
    "inner,outer,replay", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (2, 1, 2), (2, 2, 2)]
)
@pytest.mark.parametrize("order", ["left", "right", "balanced"])
def test_nonperfect_source_tree_original_occurrences_and_covers(inner, outer, replay, order):
    original = resident_pipeline_tree(inner, outer, replay)
    retained = testing.analyze_placed_sync(original, True, composition_order=order)
    assert retained.describe()["hierarchical"]
    validate(original, retained)
    # Physical transport has a separate certificate and emitted-code gate.
    # This test establishes the numerical source-tree contract only.


@pytest.mark.parametrize("axis", ["inner", "outer", "replay"])
def test_nonperfect_source_tree_size_tracks_definitions_and_log_repeats(axis):
    rows = []
    # Large inner changes L1 allocation size, so keep it within the explicit
    # modeled allocation contract. Large replay/outer counts do not resize it.
    counts = [1, 2, 4] if axis == "inner" else [1, 10, 1000, 1000000]
    for count in counts:
        dimensions = {"inner": 2, "outer": 2, "replay": 2}
        dimensions[axis] = count
        retained = testing.analyze_placed_sync(resident_pipeline_tree(**dimensions), True)
        d = retained.describe()
        rows.append((d["analyzed_occurrences"], d["interface_cells"], d["merge_nodes"]))
        assert d["merge_nodes"] < 8 * count.bit_length() + 32
    assert all(row[:2] == rows[0][:2] for row in rows)


def replay_staged_body(count, replay, m=16, n=16):
    original = staged_fixture(count, stages=1, m=m, n=n)
    span, index = original.span, ir.DataType.INDEX
    allocations, body = [], []
    for statement in sequence_stmts(original.body)[:-1]:
        (allocations if isinstance(statement, ir.AssignStmt) else body).append(statement)
    iv = ir.Var("invocation", ir.ScalarType(index), span)
    marker = ir.AssignStmt(
        ir.Var("entry_coordinate", ir.ScalarType(index), span), ir.ConstInt(0, index, span), span
    )
    loop = ir.ForStmt(
        iv,
        ir.ConstInt(0, index, span),
        ir.ConstInt(replay, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts([marker, *body, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return ir.Function(
        "startup_replay",
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([*allocations, loop, sequence_stmts(original.body)[-1]], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize("count,replay", [(0, 2), (1, 1), (2, 0), (2, 2), (3, 2)])
def test_nonperfect_startup_is_per_original_child_invocation(count, replay):
    original = replay_staged_body(count, replay)
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)


def test_nonperfect_protected_continuation_preserves_scoped_resets():
    # Existing Mmad footprint protection is retained, but each invocation's
    # initialization still terminates the previous protected chain.
    original = replay_staged_body(2, 2, m=64, n=64)
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)


def test_nested_moving_writer_remains_unavailable():
    original = nested_fixture([2, 3], moving=True)
    outer_loop = next(s for s in sequence_stmts(original.body) if isinstance(s, ir.ForStmt))
    target = sequence_stmts(outer_loop.body)[0]
    assert isinstance(target, ir.ForStmt)

    class MoveWrite(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name != ir.get_op("buffer.store").name:
                return op
            args = list(op.args)
            args[1] = ir.MakeTuple([target.loop_var, ir.ConstInt(0, ir.DataType.INDEX, op.span)], op.span)
            return ir.Call(op.op, args, op.type, op.span)

    # Change only the nested store; outside stores have no child coordinate.
    outer = next(s for s in sequence_stmts(original.body) if isinstance(s, ir.ForStmt))
    moved = ir.ForStmt(
        outer.loop_var,
        outer.start,
        outer.stop,
        outer.step,
        [],
        MoveWrite().visit_stmt(outer.body),
        [],
        outer.span,
    )
    body = ir.SeqStmts([moved if s is outer else s for s in sequence_stmts(original.body)], original.span)
    function = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        body,
        original.span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    with pytest.raises(ValueError, match="GM accesses need a common invocation translation"):
        testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)


def two_phase_accumulator_tree(count, replay, intermediate_read=False, second_count=None):
    second_count = count if second_count is None else second_count
    original = staged_fixture(max(1, count, second_count), stages=1, m=64, n=64)
    span, index = original.span, ir.DataType.INDEX
    loop = next(s for s in sequence_stmts(original.body) if isinstance(s, ir.ForStmt))
    loop = ir.ForStmt(
        loop.loop_var, loop.start, ir.ConstInt(count, index, span), loop.step, [], loop.body, [], span
    )
    second_iv = ir.Var("second_phase", ir.ScalarType(index), span)

    class Continuation(ir.IRMutator):
        def visit_var(self, op):
            return second_iv if op.same_as(loop.loop_var) else op

        def visit_call(self, op):
            return ir.Call(op.op, [self.visit_expr(a) for a in op.args], op.type, op.span)

        def visit_if_stmt(self, op):
            assert op.else_body is not None
            assert op.else_body is not None
            return self.visit_stmt(op.else_body)

    second = ir.ForStmt(
        second_iv,
        loop.start,
        ir.ConstInt(second_count, index, span),
        loop.step,
        [],
        Continuation().visit_stmt(loop.body),
        [],
        span,
    )
    store = sequence_stmts(original.body)[-2] if count else ir.SeqStmts([], span)
    body: list[ir.Stmt] = [loop]
    if intermediate_read and count:
        assert isinstance(store, ir.EvalStmt)
        assert isinstance(store.expr, ir.Call)
        body.append(ir.EvalStmt(ir.Call(store.expr.op, store.expr.args, store.expr.type, span), span))
    body.extend([second, store, ir.YieldStmt([], span)])
    iv = ir.Var("replay", ir.ScalarType(index), span)
    repeat = ir.ForStmt(
        iv,
        ir.ConstInt(0, index, span),
        ir.ConstInt(replay, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts(body, span),
        [],
        span,
    )
    return ir.Function(
        "two_accumulator_phases",
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([*sequence_stmts(original.body)[:-3], repeat, sequence_stmts(original.body)[-1]], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize("read", [False, True])
@pytest.mark.parametrize("replay", [0, 1, 2])
def test_native_accumulator_protection_across_phases_not_reads_or_resets(read, replay):
    original = two_phase_accumulator_tree(2, replay, read)
    result = testing.analyze_placed_sync(original, True)
    validate(original, result)
    # This remains the protected native contract, not a new implicit C->I edge.
    assert result.describe()["hardware_target"] == "a2a3"


def scoped_startup_tree(
    inner, k_count, replay, intermediate_read=False, *, bad_guard=False, prior_read=False
):
    """Resident accumulator initialization spans K and inner, with outer resets."""
    original = two_phase_accumulator_tree(inner, replay, intermediate_read)
    span, index = original.span, ir.DataType.INDEX
    invocation = next(s for s in sequence_stmts(original.body) if isinstance(s, ir.ForStmt))
    phases = list(sequence_stmts(invocation.body)[:-2])
    # No accumulator read is valid if the complete K invocation was skipped.
    store = sequence_stmts(invocation.body)[-2] if k_count else ir.SeqStmts([], span)
    k_var = ir.Var("k_pair", ir.ScalarType(index), span)
    k_first = ir.Eq(k_var, ir.ConstInt(3, index, span), ir.DataType.BOOL, span)

    class ScopedInitialization(ir.IRMutator):
        def visit_if_stmt(self, op):
            condition = k_first if bad_guard else ir.And(k_first, op.condition, ir.DataType.BOOL, span)
            return ir.IfStmt(condition, op.then_body, op.else_body, [], span)

    phases[0] = ScopedInitialization().visit_stmt(phases[0])
    if prior_read:
        assert isinstance(store, ir.EvalStmt)
        assert isinstance(store.expr, ir.Call)
        phases.insert(0, ir.EvalStmt(ir.Call(store.expr.op, store.expr.args, store.expr.type, span), span))
    k_loop = ir.ForStmt(
        k_var,
        ir.ConstInt(3, index, span),
        ir.ConstInt(3 + 2 * k_count, index, span),
        ir.ConstInt(2, index, span),
        [],
        ir.SeqStmts([*phases, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    replacement = ir.ForStmt(
        invocation.loop_var,
        invocation.start,
        invocation.stop,
        invocation.step,
        [],
        ir.SeqStmts([k_loop, store, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return ir.Function(
        "scoped_startup",
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([replacement if s is invocation else s for s in sequence_stmts(original.body)], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize(
    "inner,k_count,replay", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (1, 2, 2), (2, 2, 2)]
)
@pytest.mark.parametrize("intermediate_read", [False, True])
def test_scoped_startup_views_preserve_covers_queries_and_resets(inner, k_count, replay, intermediate_read):
    original = scoped_startup_tree(inner, k_count, replay, intermediate_read)
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)
    assert any(
        s["initialization_scope"] is not None for r in retained.describe()["regions"] for s in r["sites"]
    )


@pytest.mark.parametrize(
    "option,error", [("prior_read", "first-touch"), ("bad_guard", "consecutive first-visit")]
)
def test_scoped_startup_requires_first_touch_and_all_inner_coordinates(option, error):
    with pytest.raises(ValueError, match=error):
        testing.analyze_placed_sync(scoped_startup_tree(2, 2, 2, **{option: True}), True)


@pytest.mark.parametrize("middle_count", [0, 2])
def test_scoped_startup_refuses_unrepresented_intermediate_seams(middle_count):
    original = scoped_startup_tree(1, 2, 2)
    span, index = original.span, ir.DataType.INDEX
    iv = ir.Var("intermediate", ir.ScalarType(index), span)

    invocation = next(stmt for stmt in sequence_stmts(original.body) if isinstance(stmt, ir.ForStmt))
    loop = sequence_stmts(invocation.body)[0]
    assert isinstance(loop, ir.ForStmt)
    first, *rest = sequence_stmts(loop.body)

    class Guard(ir.IRMutator):
        def visit_if_stmt(self, op):
            condition = ir.And(
                op.condition,
                ir.Eq(iv, ir.ConstInt(0, index, span), ir.DataType.BOOL, span),
                ir.DataType.BOOL,
                span,
            )
            return ir.IfStmt(condition, op.then_body, op.else_body, [], span)

    empty_or_repeated = ir.ForStmt(
        iv,
        ir.ConstInt(0, index, span),
        ir.ConstInt(middle_count, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts([Guard().visit_stmt(first), ir.YieldStmt([], span)], span),
        [],
        span,
    )
    k_loop = ir.ForStmt(
        loop.loop_var,
        loop.start,
        loop.stop,
        loop.step,
        [],
        ir.SeqStmts([empty_or_repeated, *rest], span),
        [],
        span,
    )
    outer = ir.ForStmt(
        invocation.loop_var,
        invocation.start,
        invocation.stop,
        invocation.step,
        [],
        ir.SeqStmts([k_loop, *sequence_stmts(invocation.body)[1:]], span),
        [],
        span,
    )
    changed = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([outer if stmt is invocation else stmt for stmt in sequence_stmts(original.body)], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    with pytest.raises(ValueError, match="intermediate repetitions lacks continuation seam evidence"):
        testing.analyze_placed_sync(changed, True)


def structured_scaling_fixture(axis, size):
    original = staged_fixture(
        2, stages=size if axis == "classes" else 1, visits=size if axis == "children" else 1, m=64, n=64
    )
    span, index = original.span, ir.DataType.INDEX

    class Clone(ir.IRMutator):
        def visit_call(self, op):
            return ir.Call(op.op, [self.visit_expr(a) for a in op.args], op.type, op.span)

        def visit_if_stmt(self, op):
            assert op.else_body is not None
            return self.visit_stmt(op.else_body)

    body = []
    allocations = []
    for stmt in sequence_stmts(original.body)[:-1]:
        if isinstance(stmt, ir.AssignStmt):
            allocations.append(stmt)
            continue
        if axis == "sites" and isinstance(stmt, ir.ForStmt):
            payloads = list(sequence_stmts(stmt.body)[:-1])
            for _ in range(size - 1):
                payloads.extend(Clone().visit_stmt(p) for p in sequence_stmts(stmt.body)[:-1])
            stmt = ir.ForStmt(
                stmt.loop_var,
                stmt.start,
                stmt.stop,
                stmt.step,
                [],
                ir.SeqStmts([*payloads, ir.YieldStmt([], span)], span),
                [],
                span,
            )
        body.append(stmt)
    iv = ir.Var("scaling_replay", ir.ScalarType(index), span)
    outer = ir.ForStmt(
        iv,
        ir.ConstInt(0, index, span),
        ir.ConstInt(2, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts([*body, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([*allocations, outer, sequence_stmts(original.body)[-1]], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize("axis", ["sites", "children", "classes"])
def test_numerical_static_scaling_tracks_supplied_interfaces(axis):
    records = []
    for size in [1, 2, 4, 8, 16]:
        result = testing.analyze_placed_sync(structured_scaling_fixture(axis, size), True).describe()
        records.append(result)
        for region in result["regions"]:
            if region["loop"] is None:
                continue
            pipes = len({s["pipe"] for s in region["sites"]})
            assert region["core_frontiers"] and region["distance_entries"] == 0
            assert region["shortest_path_runs"] <= 2 * pipes
            assert region["retained_frontier_entries"] <= 4 * pipes * len(region["sites"])
    if axis == "sites":
        assert len({r["cells"] for r in records}) == 1
        assert len({r["source_region_nodes"] for r in records}) == 1
    elif axis == "children":
        assert len({r["cells"] for r in records}) == 1
        for size, result in zip([1, 2, 4, 8, 16], records, strict=True):
            assert result["source_coordinate_entries"] <= 8 * size + 8
    else:
        # More written domains create more supplied classes. No claim that
        # Cartesian partitioning or full per-leaf interfaces are globally linear.
        assert records[-1]["cells"] > records[0]["cells"]


def expand_scoped_families(export):
    """Interpret exported coordinate maps, independently of compiler expansion."""
    records = []
    for family in export["families"]:
        sr, sv, ss = family["source"]
        tr, tv, ts = family["target"]
        axes = family["axes"]
        assert len({axis[0] for axis in axes}) == len(axes)
        for digits in product(*(range(axis[1]) for axis in axes)):
            source = (sr, sv + sum(d * a[2] for d, a in zip(digits, axes, strict=True)), ss)
            target = (tr, tv + sum(d * a[3] for d, a in zip(digits, axes, strict=True)), ts)
            records.append((source, target))
    assert len(records) == len(set(records)), "Duplicate logical publication identities"
    return set(records)


@pytest.mark.parametrize("fixture", ["resident", "startup", "startup_fix"])
@pytest.mark.parametrize(
    "inner,k_count,replay",
    [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (2, 2, 2), (1, 3, 2), (1, 4, 2), (1, 5, 2)],
)
def test_scoped_families_equal_actual_covers_and_keep_original_frames(fixture, inner, k_count, replay):
    original = (
        resident_pipeline_tree(inner, k_count, replay)
        if fixture == "resident"
        else scoped_startup_tree(inner, k_count, replay, intermediate_read=fixture == "startup_fix")
    )
    retained = testing.analyze_placed_sync(original, True)
    checked = validate(original, retained)
    before = retained.describe()
    export = retained.export_scoped_demands()
    assert export is not None
    assert expand_scoped_families(export) == set(retained.expand_retained_demands(checked["payloads"]))
    for family in export["families"]:
        for frame, _, _, _ in family["axes"]:
            for endpoint in [family["source"], family["target"]]:
                original_frames = before["regions"][endpoint[0]]["nested_frames"]
                assert any(frame.same_as(f[0]) for f in original_frames)
    assert retained.export_scoped_demands() == export
    after = retained.describe()
    assert after.pop("scoped_export_builds") == 1
    assert before.pop("scoped_export_builds") == 0
    assert before == after, "An endpoint export must not reconstruct retained mathematics"


def test_scoped_export_size_does_not_enumerate_visits():
    sizes = []
    for count in [1, 2, 10, 1000, 1000000]:
        retained = testing.analyze_placed_sync(resident_pipeline_tree(2, count, count), True)
        export = retained.export_scoped_demands()
        assert export is not None
        sizes.append((len(export["families"]), export["visited_source_nodes"], export["coordinate_entries"]))
    assert all(row == sizes[1] for row in sizes[1:])
    assert sizes[0][0] <= sizes[1][0]  # Singleton repetition has no inter-visit seam.


@pytest.mark.parametrize("order", ["left", "right", "balanced"])
@pytest.mark.parametrize("first,second", [(1, 3), (3, 1), (2, 0)])
def test_scoped_export_unequal_and_empty_siblings(first, second, order):
    original = two_phase_accumulator_tree(first, 2, second_count=second)
    retained = testing.analyze_placed_sync(original, True, composition_order=order)
    checked = validate(original, retained)
    export = retained.export_scoped_demands()
    assert expand_scoped_families(export) == set(retained.expand_retained_demands(checked["payloads"]))


def with_launch_assignment(original, variable, body):
    span = original.span
    definition = ir.AssignStmt(
        variable, ir.Call(ir.get_op("buffer.get_block_idx"), [], variable.type, span), span
    )
    return ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([definition, *sequence_stmts(body)], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


def test_block_dependent_local_extract_is_not_silently_zero_translated():
    original = resident_pipeline_tree(2, 1, 2)
    span, index = original.span, ir.DataType.INDEX
    block = ir.Var("block", ir.ScalarType(index), span)

    class Shift(ir.IRMutator):
        def visit_call(self, op):
            if (
                op.op.name == ir.get_op("buffer.extract").name
                and isinstance(op.args[0], ir.Var)
                and op.args[0].name_hint == "a1"
            ):
                args = list(op.args)
                # Both offsets are inside the 64-wide resident tile. Bounds
                # alone cannot justify erasing the invocation translation.
                args[2] = ir.Mul(block, ir.ConstInt(32, index, span), index, span)
                return ir.Call(op.op, args, op.type, op.span)
            return op

    changed = with_launch_assignment(original, block, Shift().visit_stmt(original.body))
    with pytest.raises(ValueError, match="Parameterized local extraction"):
        testing.analyze_placed_sync(changed, True, block_bounds=(0, 1))


def test_symbolic_singleton_start_is_not_constant_zero_initialization():
    original = scoped_startup_tree(2, 2, 2)
    span, index = original.span, ir.DataType.INDEX
    block = ir.Var("block", ir.ScalarType(index), span)
    outer = next(stmt for stmt in sequence_stmts(original.body) if isinstance(stmt, ir.ForStmt))
    k_loop = sequence_stmts(outer.body)[0]
    assert isinstance(k_loop, ir.ForStmt)

    class Guard(ir.IRMutator):
        def visit_if_stmt(self, op):
            assert isinstance(op.condition, ir.And)
            condition = ir.And(
                ir.Eq(k_loop.loop_var, ir.ConstInt(0, index, span), ir.DataType.BOOL, span),
                op.condition.right,
                ir.DataType.BOOL,
                span,
            )
            return ir.IfStmt(condition, op.then_body, op.else_body, [], span)

    singleton = ir.ForStmt(
        k_loop.loop_var,
        block,
        ir.ConstInt(2, index, span),
        ir.ConstInt(2, index, span),
        [],
        Guard().visit_stmt(k_loop.body),
        [],
        span,
    )
    repeated = ir.ForStmt(
        outer.loop_var,
        outer.start,
        outer.stop,
        outer.step,
        [],
        ir.SeqStmts([singleton, *sequence_stmts(outer.body)[1:]], span),
        [],
        span,
    )
    body = ir.SeqStmts([repeated if stmt is outer else stmt for stmt in sequence_stmts(original.body)], span)
    changed = with_launch_assignment(original, block, body)
    # block1 executes once, but its initializer condition is false. The source
    # frame's normalized ordinal0 is not proof that original k_loop IV is0.
    with pytest.raises(ValueError, match="Symbolic first-visit initialization"):
        testing.analyze_placed_sync(changed, True, block_bounds=(0, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
