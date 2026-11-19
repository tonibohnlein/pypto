# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unswizzled row-major external geometry, independent of kernel names or flags."""

import pytest
from pypto import ir, passes, testing

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import check_imported_accesses, interpret, validate
from .sync_physical_oracle import run_physical
from .test_nested_phase_sync import nested_matrix_fixture
from .test_nested_sync_repetition import expand_scoped_families

pytestmark = pytest.mark.usefixtures("ascend_backend")


def unavailable_read_domains(description):
    """Declare only this milestone's explicit nonlinear readonly selector gap."""
    reads, writes, external = set(), set(), set()
    for region in description["regions"]:
        for site in region["sites"]:
            for access, geometry in zip(site["accesses"], site["external_geometry"], strict=True):
                (writes if access[1] else reads).add(access[0])
                if geometry is not None:
                    external.add(access[0])
    return (reads - writes) & external


@pytest.mark.parametrize("count,visits", [(0, 2), (1, 0), (1, 1), (1, 2), (2, 3), (3, 3)])
def test_row_major_external_accesses_and_retained_order(count, visits):
    function = nested_matrix_fixture(count, visits=visits, tile_columns=2)
    description = testing.describe_placed_sync_input(function, True)
    storage, calls = interpret(function)
    check_imported_accesses(description, storage, calls)
    result = testing.analyze_placed_sync(function, True)
    checks = validate(function, result, unavailable_boundaries=unavailable_read_domains(description))
    assert expand_scoped_families(result.export_scoped_demands()) == set(
        result.expand_retained_demands(checks["payloads"])
    )
    # The final allocated row may contain an untouched tile: proven absence is
    # distinct from the explicitly unavailable readonly input selectors above.
    if count and visits == 3:
        output = next(i for i, domain in enumerate(description["storage"]) if domain["name"] == "out")
        absent = result.boundary(output, 16, 16)
        assert absent is not None
        assert absent["first_writer"] is None and absent["last_writer"] is None
        for row, col in [(0, 0), (0, 16), (16, 0)]:
            answer = result.boundary(output, row, col)
            assert answer is not None and answer["first_writer"] == answer["last_writer"]
            assert answer["first_writer"] is not None


def test_row_major_external_domains_require_explicit_alias_contract():
    with pytest.raises(ValueError, match="Disjoint GM argument contract"):
        testing.analyze_placed_sync(nested_matrix_fixture(2, visits=3, tile_columns=2), False)


def test_output_tile_map_cannot_discharge_enclosing_replay_conflicts():
    function = nested_matrix_fixture(2, visits=3, tile_columns=2)
    assert isinstance(function.body, ir.SeqStmts)
    statements = function.body.stmts
    outer = statements[-2]
    span = function.span

    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, span)

    repeat = ir.Var("repeat", ir.ScalarType(ir.DataType.INDEX), span)
    loop = ir.ForStmt(
        repeat, ci(0), ci(2), ci(1), [], ir.SeqStmts([outer, ir.YieldStmt([], span)], span), [], span
    )
    changed = ir.Function(
        function.name,
        list(zip(function.params, function.param_directions, strict=True)),
        function.return_types,
        ir.SeqStmts([*statements[:-2], loop, statements[-1]], span),
        span,
        function.func_type,
        ir_stage=function.ir_stage,
    )
    with pytest.raises(ValueError, match="ownership|replay|invocation"):
        testing.analyze_placed_sync(changed, True)


@pytest.mark.parametrize("count", [3, 16, 257, 1000000])
def test_external_geometry_and_phase_size_do_not_unfold_k(count):
    function = nested_matrix_fixture(count, visits=3, tile_columns=2)
    result = testing.analyze_placed_sync(function, True)
    description = result.describe()
    assert description["phase_variants"] == 4
    assert description["finite_microbody_occurrences"] == 25
    assert sum(len(program) for program in description["coordinate_programs"]) < 40
    assert description["merge_nodes"] <= 16 * count.bit_length() + 32


def test_original_void_initialization_arms_may_end_in_empty_yield():
    original = nested_matrix_fixture(2, visits=3, tile_columns=2)

    class ExplicitYield(ir.IRMutator):
        def visit_if_stmt(self, op):
            def arm(body):
                return ir.SeqStmts([*body.stmts, ir.YieldStmt([], op.span)], op.span)

            return ir.IfStmt(op.condition, arm(op.then_body), arm(op.else_body), [], op.span)

    changed = ExplicitYield().visit_function(original)
    description = testing.describe_placed_sync_input(changed, True)
    result = testing.analyze_placed_sync(changed, True)
    validate(changed, result, unavailable_boundaries=unavailable_read_domains(description))


def test_external_geometry_does_not_invent_output_tile_ownership():
    original = nested_matrix_fixture(2, visits=3, tile_columns=2)

    class CollidingOutput(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.store").name:
                assert isinstance(op.args[1], ir.MakeTuple)
                offsets = ir.MakeTuple(
                    [op.args[1].elements[0], ir.ConstInt(0, ir.DataType.INDEX, op.span)], op.span
                )
                return ir.Call(op.op, [op.args[0], offsets, *op.args[2:]], op.type, op.span)
            return super().visit_call(op)

    changed = CollidingOutput().visit_function(original)
    description = testing.describe_placed_sync_input(changed, True)
    storage, calls = interpret(changed)
    check_imported_accesses(description, storage, calls)
    with pytest.raises(ValueError, match="ownership|tile map"):
        testing.analyze_placed_sync(changed, True)


def test_runtime_divisor_is_not_admitted_as_a_row_major_certificate():
    original = nested_matrix_fixture(2, visits=3, tile_columns=2)
    assert isinstance(original.body, ir.SeqStmts)
    outer = original.body.stmts[-2]
    assert isinstance(outer, ir.ForStmt)

    class RuntimeDivisor(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.store").name:
                one = ir.ConstInt(1, ir.DataType.INDEX, op.span)
                divisor = ir.Add(outer.loop_var, one, ir.DataType.INDEX, op.span)
                row = ir.FloorDiv(outer.loop_var, divisor, ir.DataType.INDEX, op.span)
                offsets = ir.MakeTuple([row, ir.ConstInt(0, ir.DataType.INDEX, op.span)], op.span)
                return ir.Call(op.op, [op.args[0], offsets, *op.args[2:]], op.type, op.span)
            return super().visit_call(op)

    with pytest.raises(ValueError, match="positive constant divisor"):
        testing.describe_placed_sync_input(RuntimeDivisor().visit_function(original), True)


def with_unbounded_dma_coordinate(
    *,
    writable_shape: bool = False,
    writable_offset: bool = False,
    shared_base: bool = False,
    runtime_control: bool = False,
    input_write: bool = False,
) -> ir.Function:
    source = staged_fixture(2, stages=1, m=64, n=64)
    a, b, out = source.params
    span = source.span
    index = ir.DataType.INDEX
    scalar = ir.ScalarType(index)
    base = ir.Var("row_base", scalar, span)
    extent = ir.Var("row_extent", scalar, span)
    common = ir.MemRef("shared", 0, 0, span) if shared_base else None
    assert isinstance(b.type, ir.TensorType) and isinstance(out.type, ir.TensorType)
    weights = ir.Var(b.name_hint, ir.TensorType([extent, b.type.shape[1]], b.type.dtype, common), span)
    output = ir.Var(
        out.name_hint,
        ir.TensorType(
            [extent if writable_shape else out.type.shape[0], out.type.shape[1]], out.type.dtype, common
        ),
        span,
    )
    body = ir.substitute_stmt(source.body, [(b, weights), (out, output)])

    class DynamicOffset(ir.IRMutator):
        def visit_if_stmt(self, op: ir.IfStmt) -> ir.Stmt:
            if not runtime_control:
                return super().visit_if_stmt(op)
            gate = ir.Eq(base, ir.ConstInt(0, index, span), ir.DataType.BOOL, span)
            condition = ir.And(op.condition, gate, ir.DataType.BOOL, span)
            return ir.IfStmt(condition, op.then_body, op.else_body, op.return_vars, span)

        def visit_call(self, op: ir.Call) -> ir.Expr:
            if (
                op.op.name == ir.get_op("buffer.load").name
                and op.args[0] is weights
                or writable_offset
                and op.op.name == ir.get_op("buffer.store").name
            ):
                offsets = op.args[1]
                assert isinstance(offsets, ir.MakeTuple)
                row, col = offsets.elements
                new = ir.MakeTuple([ir.Add(base, row, index, span), col], span)
                return ir.Call(op.op, [op.args[0], new, *op.args[2:]], op.type, span)
            return super().visit_call(op)

    body = DynamicOffset().visit_stmt(body)
    direction = ir.ParamDirection.In if input_write else ir.ParamDirection.InOut
    return ir.Function(
        source.name,
        [a, weights, (output, direction), base, extent],
        [output.type],
        body,
        span,
        source.func_type,
        ir_stage=source.ir_stage,
    )


def realize_exact_projection(function: ir.Function, *, disjoint: bool, split_passes: bool) -> ir.Function:
    program = ir.Program([function], "DisjointReadOnlyProjection", function.span)
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", autosync_disjoint_gm=disjoint
    ):
        if split_passes:
            result = passes.lower_sync()(passes.optimize_sync()(passes.insert_sync()(program)))
        else:
            result = passes.realize_sync()(program)
    return next(iter(result.functions.values()))


@pytest.mark.parametrize("split_passes", [False, True])
def test_disjoint_readonly_projection_keeps_original_unbounded_coordinates(split_passes):
    original = with_unbounded_dma_coordinate()
    before = ir.serialize(original)
    with pytest.raises(ValueError, match="caller range"):
        testing.analyze_placed_sync(original, True)
    physical = realize_exact_projection(original, disjoint=True, split_passes=split_passes)
    assert physical.attrs["pypto_autosync"]
    assert "autosync_analysis_policy" not in physical.attrs
    assert ir.serialize(original) == before
    for expected, actual in zip(original.params, physical.params, strict=True):
        ir.assert_structural_equal(expected.type, actual.type)
    run_physical(original, physical, range(6), {3: 1, 4: 65})


@pytest.mark.parametrize(
    "disjoint,unsafe",
    [
        (False, {}),
        (True, {"writable_shape": True}),
        (True, {"writable_offset": True}),
        (True, {"shared_base": True}),
        (True, {"runtime_control": True}),
        (True, {"input_write": True}),
    ],
)
def test_disjoint_readonly_projection_requires_qualified_writable_and_distinct_roots(disjoint, unsafe):
    original = with_unbounded_dma_coordinate(**unsafe)
    physical = realize_exact_projection(original, disjoint=disjoint, split_passes=False)
    assert (
        not physical.attrs["pypto_autosync"]
        or physical.attrs["autosync_analysis_policy"] == "conservative_prefix"
    )


def test_symbolic_innermost_frame_keeps_original_start_for_branch_proof():
    source = staged_fixture(2, stages=1, m=64, n=64)
    span = source.span
    index = ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index, span)

    block = ir.Var("block", ir.ScalarType(index), span)
    get_block = ir.AssignStmt(block, ir.Call(ir.get_op("buffer.get_block_idx"), [], block.type, span), span)
    assert isinstance(source.body, ir.SeqStmts)
    original_loop, _, returned = source.body.stmts[-3:]
    assert isinstance(original_loop, ir.ForStmt) and isinstance(original_loop.body, ir.SeqStmts)
    var = original_loop.loop_var
    payload = original_loop.body.stmts[0]
    assert isinstance(payload, ir.EvalStmt) and isinstance(payload.expr, ir.Call)
    load = payload.expr
    offsets = ir.MakeTuple([ci(0), ci(0)], span)
    first = ir.EvalStmt(ir.Call(load.op, [load.args[0], offsets, *load.args[2:]], load.type, span), span)
    second = ir.EvalStmt(ir.Call(load.op, [load.args[0], offsets, *load.args[2:]], load.type, span), span)
    branch = ir.IfStmt(ir.Gt(var, ci(24), ir.DataType.BOOL, span), second, None, [], span)
    loop = ir.ForStmt(
        var, block, ci(68), ci(24), [], ir.SeqStmts([first, branch, ir.YieldStmt([], span)], span), [], span
    )
    changed = ir.Function(
        source.name,
        list(zip(source.params, source.param_directions, strict=True)),
        source.return_types,
        ir.SeqStmts([*source.body.stmts[:-3], get_block, loop, returned], span),
        span,
        source.func_type,
        ir_stage=source.ir_stage,
    )
    # The first trip uses an IV between 20 and 23 and the second between 44
    # and 47. A zero-based [0, 24] proof would incorrectly discard the branch.
    with pytest.raises(ValueError, match="Only first-visit initialization"):
        testing.analyze_sync_invocation(changed, True, "a2a3", {}, (20, 23))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
