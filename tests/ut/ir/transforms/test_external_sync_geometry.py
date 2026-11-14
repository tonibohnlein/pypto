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
from pypto import ir, testing

from .sync_matmul_oracle import check_imported_accesses, interpret, validate
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
