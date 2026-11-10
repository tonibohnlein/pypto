# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical nested repetition preserves original frames without unfolding."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .test_sync_range_proofs import caller_program, range_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
BOUNDS = {1: (0, 16), 3: (64, 128)}


def nested_fixture(counts, repeats=1, moving=False, parent_consumers=True, outer_moving=False):
    original = range_fixture()
    span, index = original.span, ir.DataType.INDEX
    assert isinstance(original.body, ir.SeqStmts)
    alloc, load, store, ret = original.body.stmts
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
    assert retained.describe() == before
    regions = [r for r in before["regions"] if r["nested_frames"]]
    assert len(regions) == repeats
    assert all(r["count"] == outer * inner and len(r["nested_frames"]) == 2 for r in regions)
    # This exporter needs carried tokens at each enclosing loop. A qualified
    # physical export is not permission to manufacture a logical token scope.
    with pytest.raises(ValueError, match="token transport across every enclosing frame"):
        retained.realize()


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


def test_inner_moving_access_is_not_mistaken_for_invariant_repetition():
    with pytest.raises(ValueError, match="Nested moving accesses require a child-boundary summary"):
        testing.analyze_placed_sync(nested_fixture([2, 3], moving=True), True, parameter_bounds=BOUNDS)


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
    with pytest.raises(ValueError, match="Unavailable same-pipe barrier mechanism"):
        retained.realize_physical("a2a3", list(range(6)))
    assert retained.describe() == before


def test_outer_coordinate_access_requires_general_nested_interface():
    with pytest.raises(ValueError, match="Unavailable affine synchronization access"):
        testing.analyze_placed_sync(nested_fixture([2, 3], outer_moving=True), True, parameter_bounds=BOUNDS)


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
    for stmt in original.body.stmts:
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
    with pytest.raises(ValueError, match="Nested initialization requires a per-invocation summary"):
        testing.analyze_placed_sync(function, True)


def test_empty_middle_frame_preserves_parent_work():
    function = nested_fixture([2, 0, 3])
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    validate(function, retained, parameters={1: 0, 3: 64})
    physical = retained.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6), parameters={1: 0, 3: 64})


def test_original_frame_and_ordinal_work_scale_with_depth():
    def guard_nodes(body):
        statements, expressions, seen = [body], [], set()
        while statements:
            statement = statements.pop()
            if isinstance(statement, ir.SeqStmts):
                statements.extend(statement.stmts)
            elif isinstance(statement, ir.ForStmt):
                statements.append(statement.body)
            elif isinstance(statement, ir.IfStmt):
                expressions.append(statement.condition)
                statements.append(statement.then_body)
                if statement.else_body is not None:
                    statements.append(statement.else_body)
        while expressions:
            expression = expressions.pop()
            if not isinstance(expression, ir.BinaryExpr) or expression in seen:
                continue
            seen.add(expression)
            expressions.extend([expression.left, expression.right])
        return len(seen)

    measurements = []
    for depth in [2, 4, 8, 16]:
        function = nested_fixture([2] * depth)
        retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
        physical = retained.realize_physical("a2a3", list(range(6)))
        nodes = guard_nodes(physical["function"].body)
        region = next(r for r in retained.describe()["regions"] if r["nested_frames"])
        assert len(region["nested_frames"]) == depth
        measurements.append((depth, nodes))
    # Each original frame adds normalized-digit and mixed-radix arithmetic at
    # the fixed command sites. Count distinct generated guard nodes, not elapsed time.
    slopes = [(b - a) // (db - da) for (da, a), (db, b) in zip(measurements, measurements[1:])]
    assert slopes[0] > 0 and len(set(slopes)) == 1
    assert all((b - a) == slopes[0] * (db - da) for (da, a), (db, b) in zip(measurements, measurements[1:]))
