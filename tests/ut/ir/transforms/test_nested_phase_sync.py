# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original-coordinate nested ring snapshots, independently checked from Buffer IR."""

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import check_imported_accesses, interpret, sequence_stmts, validate
from .test_nested_sync_repetition import expand_scoped_families

pytestmark = pytest.mark.usefixtures("ascend_backend")


def nested_matrix_fixture(count, visits=1, slices=2, m=16, n=16, micro_repeats=None, tile_columns=None):
    """Repeated invocations share L1/L0/ACC and GM storage; K and slice IVs reset.

    This derivative preserves the original matrix structure, not the original
    multicore tile permutation. Its parameter domain is literal nonnegative
    counts, two explicit rings, and full 16-wide slices. No padding assumptions
    are introduced: every logical element has a checked physical byte address.
    """
    span = ir.Span.unknown()
    dtype, index = ir.DataType.FP16, ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(v) if isinstance(v, int) else v for v in values], span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    def mul(a, b):
        return ir.Mul(a, ci(b), index, span)

    chunk = 16 * slices
    output_rows = m if tile_columns is None else max(1, (visits + tile_columns - 1) // tile_columns) * m
    output_columns = n if tile_columns is None else tile_columns * n
    a = ir.Var("a", ir.TensorType([output_rows, max(1, count) * chunk], dtype), span)
    b = ir.Var("b", ir.TensorType([max(1, count) * chunk, output_columns], dtype), span)
    out = ir.Var("out", ir.TensorType([output_rows, output_columns], ir.DataType.FP32), span)
    allocations = []

    def allocate(name, shape, space, address, slots=2):
        element = ir.BufferType(
            shape,
            ir.DataType.FP32 if space == ir.MemorySpace.Acc else dtype,
            space,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024 if space == ir.MemorySpace.Acc else 512,
        )
        ty = ir.MultiBufferType(element, slots) if slots > 1 else element
        var = ir.Var(name, ty, span)
        allocation = "buffer.alloc_multi" if slots > 1 else "buffer.alloc"
        allocations.append(ir.AssignStmt(var, call(allocation, [tup([]), ci(address)], ty), span))
        return var

    mat_a = allocate("mat_a", [m, chunk], ir.MemorySpace.Mat, 0)
    mat_b = allocate("mat_b", [chunk, n], ir.MemorySpace.Mat, 2 * m * chunk * 2)
    left = allocate("left", [m, 16], ir.MemorySpace.Left, 0)
    right = allocate("right", [16, n], ir.MemorySpace.Right, 0)
    acc = allocate("acc", [m, n], ir.MemorySpace.Acc, 0, slots=1)
    outer, k, inner = [ir.Var(name, ir.ScalarType(index), span) for name in ("visit", "k", "slice")]

    row_offset, column_offset = ci(0), ci(0)
    if tile_columns is not None:
        row_offset = mul(ir.FloorDiv(outer, ci(tile_columns), index, span), m)
        column_offset = mul(ir.FloorMod(outer, ci(tile_columns), index, span), n)

    def slot(group, iv, name):
        selected = ir.Var(name, group.type.element_type, span)
        definition = ir.AssignStmt(
            selected, call("buffer.slot", [group, ir.FloorMod(iv, ci(2), index, span)], selected.type), span
        )
        return selected, definition

    a_slot, a_def = slot(mat_a, k, "a_slot")
    b_slot, b_def = slot(mat_b, k, "b_slot")
    l_slot, l_def = slot(left, inner, "l_slot")
    r_slot, r_def = slot(right, inner, "r_slot")

    def matrix_arm(name, accumulate):
        # Real lowering emits fresh descriptors inside each original arm.
        lhs, ld = slot(left, inner, name + "_left")
        rhs, rd = slot(right, inner, name + "_right")
        return ir.SeqStmts(
            [ld, rd, payload("buffer.matmul_acc" if accumulate else "buffer.matmul", [lhs, rhs, acc])], span
        )

    startup = ir.And(
        ir.Eq(k, ci(0), ir.DataType.BOOL, span),
        ir.Eq(inner, ci(0), ir.DataType.BOOL, span),
        ir.DataType.BOOL,
        span,
    )
    if micro_repeats is not None:
        startup = ir.Eq(inner, ci(0), ir.DataType.BOOL, span)
    inner_body = ir.SeqStmts(
        [
            l_def,
            r_def,
            payload("buffer.extract", [a_slot, ci(0), mul(inner, 16), l_slot]),
            payload("buffer.extract", [b_slot, mul(inner, 16), ci(0), r_slot]),
            ir.IfStmt(startup, matrix_arm("initialize", False), matrix_arm("accumulate", True), [], span),
            ir.YieldStmt([], span),
        ],
        span,
    )
    inner_loop = ir.ForStmt(inner, ci(0), ci(slices), ci(1), [], inner_body, [], span)
    if micro_repeats is not None:
        wrapper = ir.Var("micro_repeat", ir.ScalarType(index), span)
        inner_loop = ir.ForStmt(
            wrapper,
            ci(0),
            ci(micro_repeats),
            ci(1),
            [],
            ir.SeqStmts([inner_loop, ir.YieldStmt([], span)], span),
            [],
            span,
        )
    k_body = ir.SeqStmts(
        [
            a_def,
            b_def,
            payload("buffer.load", [a, tup([row_offset, mul(k, chunk)]), tup([m, chunk]), a_slot]),
            payload("buffer.load", [b, tup([mul(k, chunk), column_offset]), tup([chunk, n]), b_slot]),
            inner_loop,
            ir.YieldStmt([], span),
        ],
        span,
    )
    k_loop = ir.ForStmt(k, ci(0), ci(count), ci(1), [], k_body, [], span)
    outer_body: list[ir.Stmt] = [k_loop]
    # Zero K does not read an uninitialized accumulator.
    if count:
        outer_body.append(payload("buffer.store", [acc, tup([row_offset, column_offset]), tup([m, n]), out]))
    outer_body.append(ir.YieldStmt([], span))
    outer_loop = ir.ForStmt(outer, ci(0), ci(visits), ci(1), [], ir.SeqStmts(outer_body, span), [], span)
    return ir.Function(
        "nested_matrix_phase",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts([*allocations, outer_loop, ir.ReturnStmt([out], span)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("count,visits", [(0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)])
def test_nested_matrix_original_accesses(count, visits):
    function = nested_matrix_fixture(count, visits)
    description = testing.describe_placed_sync_input(function, True)
    storage, calls = interpret(function)
    check_imported_accesses(description, storage, calls)
    assert len(calls) == visits * (count * 8 + (count > 0))


@pytest.mark.parametrize("count,visits", [(0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)])
def test_nested_matrix_actual_covers_queries_selectors(count, visits):
    function = nested_matrix_fixture(count, visits)
    result = testing.analyze_placed_sync(function, True)
    checked = validate(function, result)
    exported = result.export_scoped_demands()
    assert exported is not None
    assert expand_scoped_families(exported) == set(result.expand_retained_demands(checked["payloads"]))


def test_nested_matrix_native_accumulator_threshold():
    # 4*3 >= 10 admits the existing hardware protection. Tiny 1*1 cases
    # above deliberately do not satisfy it and must retain same-pipe demands.
    function = nested_matrix_fixture(2, visits=2, m=64, n=48)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)


@pytest.mark.parametrize("count", [3, 16, 257, 1000000])
def test_nested_matrix_retains_phases_instead_of_unfolding_k(count):
    result = testing.analyze_placed_sync(nested_matrix_fixture(count, visits=3), True)
    description = result.describe()
    assert description["hierarchical"]
    assert description["analyzed_occurrences"] <= 64
    assert description["phase_variants"] == 4
    assert description["finite_microbody_occurrences"] == description["analyzed_occurrences"]
    assert description["finite_microbody_occurrences"] == 25
    assert description["merge_nodes"] <= 16 * count.bit_length() + 32


@pytest.mark.parametrize("coordinate", ["k", "visit"])
def test_local_geometry_cannot_borrow_a_representative_phase_binding(coordinate):
    function = nested_matrix_fixture(4, visits=4, slices=4)
    frames = {}

    def collect(stmt):
        if isinstance(stmt, ir.ForStmt):
            frames[stmt.loop_var.name_hint] = stmt.loop_var
            collect(stmt.body)
        elif isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                collect(child)

    collect(function.body)

    class MoveRead(ir.IRMutator):
        def visit_call(self, op):
            if (
                op.op.name == ir.get_op("buffer.extract").name
                and isinstance(op.args[0], ir.Var)
                and op.args[0].name_hint == "a_slot"
            ):
                offset = ir.Mul(
                    frames[coordinate],
                    ir.ConstInt(16, ir.DataType.INDEX, op.span),
                    ir.DataType.INDEX,
                    op.span,
                )
                return ir.Call(op.op, [op.args[0], op.args[1], offset, op.args[3]], op.type, op.span)
            return super().visit_call(op)

    moved = MoveRead().visit_function(function)
    # All four rectangles fit the original buffer, but neither K nor the
    # enclosing visit is enumerated by the finite inner microbody. Sampling
    # either coordinate as zero would turn a missing proof into a false export.
    description = testing.describe_placed_sync_input(moved, True)
    storage, calls = interpret(moved)
    check_imported_accesses(description, storage, calls)
    with pytest.raises(ValueError, match="Phase-local geometry varies outside"):
        testing.analyze_placed_sync(moved, True)


def test_phase_provider_cannot_assume_an_outer_startup_is_reset_locally():
    function = nested_matrix_fixture(3, visits=2, m=64, n=48)
    outer = None

    class Collect(ir.IRVisitor):
        def visit_for_stmt(self, op):
            nonlocal outer
            if op.loop_var.name_hint == "visit":
                outer = op.loop_var
            super().visit_for_stmt(op)

    Collect().visit_function(function)

    class OuterStartup(ir.IRMutator):
        def visit_if_stmt(self, op):
            assert isinstance(outer, ir.Var)
            first = ir.Eq(outer, ir.ConstInt(0, ir.DataType.INDEX, op.span), ir.DataType.BOOL, op.span)
            return ir.IfStmt(
                ir.And(first, op.condition, ir.DataType.BOOL, op.span),
                op.then_body,
                op.else_body,
                op.return_vars,
                op.span,
            )

    # The importer may reject the unsupported three-frame first-touch scope
    # before phase construction. Either layer must retain an explicit refusal.
    with pytest.raises(
        ValueError,
        match=(
            "outer startup scope|unbound enclosing invocation|Initialization across intermediate repetitions"
        ),
    ):
        testing.analyze_placed_sync(OuterStartup().visit_function(function), True)


def test_phase_snapshot_refuses_a_nested_repeated_subregion():
    function = nested_matrix_fixture(3, visits=2, micro_repeats=3)
    outer = next(stmt for stmt in sequence_stmts(function.body) if isinstance(stmt, ir.ForStmt))
    k_loop = next(stmt for stmt in sequence_stmts(outer.body) if isinstance(stmt, ir.ForStmt))
    wrapper = next(stmt for stmt in sequence_stmts(k_loop.body) if isinstance(stmt, ir.ForStmt))
    assert wrapper.loop_var.name_hint == "micro_repeat"
    with pytest.raises(ValueError, match="Nested repeated microbody"):
        testing.analyze_placed_sync(function, True)


def test_unprotected_unconditional_matmul_remains_an_invariant_rotating_payload():
    original = nested_matrix_fixture(1, visits=1, slices=3)
    statements = sequence_stmts(original.body)
    outer = statements[-2]
    assert isinstance(outer, ir.ForStmt)
    k_loop, store, _ = sequence_stmts(outer.body)
    assert isinstance(k_loop, ir.ForStmt)
    k_body = sequence_stmts(k_loop.body)

    class SingleInvocation(ir.IRMutator):
        def visit_var(self, op):
            return ir.ConstInt(0, ir.DataType.INDEX, op.span) if op.same_as(k_loop.loop_var) else op

        def visit_if_stmt(self, op):
            # Every matrix operation overwrites ACC. This has invariant effects
            # and no >=10 native exemption, unlike conditional accumulation.
            return self.visit_stmt(op.then_body)

        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.extract").name:
                # Use the already supported stationary source rectangle; a
                # moving Mat selector is a separate phase-interface contract.
                zero = ir.ConstInt(0, ir.DataType.INDEX, op.span)
                return ir.Call(op.op, [op.args[0], zero, zero, op.args[3]], op.type, op.span)
            return super().visit_call(op)

    body = SingleInvocation().visit_stmt(
        ir.SeqStmts([*statements[:-2], *k_body[:-1], store, statements[-1]], original.span)
    )
    function = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions, strict=True)),
        original.return_types,
        body,
        original.span,
        original.func_type,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    regions = result.describe()["regions"]
    assert any(region["rotating"] and region["count"] == 3 for region in regions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
