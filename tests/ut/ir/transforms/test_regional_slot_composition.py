# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Retained numerical composition checked across original intervals and resets."""

import pytest
from pypto import ir, testing

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import sequence_stmts, validate
from .test_rotating_sync_analysis import rotating_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
SCHEDULE = [("load", 0, 0), ("rmw", 0, 1), ("store", 0, 2)]


def regional_fixture(counts, slots=3, start=0, step=1, schedules=None):
    """Sibling loop invocations share one physical family; each IV resets."""
    functions = [
        rotating_fixture(
            1 if count is None else count,
            [(slots, 1)],
            schedules[i] if schedules is not None else SCHEDULE,
            start=start,
            step=step,
            suffix=f"_{i}",
        )
        for i, count in enumerate(counts)
    ]
    first = functions[0]
    allocations = list(sequence_stmts(first.body)[:-2])
    shared = {var.name_hint: var for var in first.params}
    for stmt in allocations:
        assert isinstance(stmt, ir.AssignStmt)
        shared[stmt.var.name_hint] = stmt.var

    class Bind(ir.IRMutator):
        def visit_var(self, op):
            return shared.get(op.name_hint, op)

    loops = []
    for count, function in zip(counts, functions, strict=True):
        loop = Bind().visit_stmt(sequence_stmts(function.body)[-2])
        assert isinstance(loop, ir.ForStmt)
        if count is not None:
            loops.append(loop)
        else:
            loop_start, loop_var = loop.start, loop.loop_var

            class Resolve(ir.IRMutator):
                def visit_var(self, op):
                    return loop_start if op.name_hint == loop_var.name_hint else op

            finite = Resolve().visit_stmt(ir.SeqStmts(list(sequence_stmts(loop.body)[:-1]), loop.span))
            loops.extend(sequence_stmts(finite))
    return ir.Function(
        "rotating_sequence",
        [first.params[0], (first.params[1], ir.ParamDirection.InOut)],
        first.return_types,
        ir.SeqStmts([*allocations, *loops, sequence_stmts(first.body)[-1]], first.span),
        first.span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def repeated_fixture(inner, outer, slots=3, inner_start=0, inner_step=1):
    """Perfect nest with a real inner modular-counter reset at each outer visit."""
    function = rotating_fixture(inner, [(slots, 1)], SCHEDULE, start=inner_start, step=inner_step)
    span, index = function.span, ir.DataType.INDEX
    iv = ir.Var("outer", ir.ScalarType(index), span)
    loop = ir.ForStmt(
        iv,
        ir.ConstInt(2, index, span),
        ir.ConstInt(2 + outer * 3, index, span),
        ir.ConstInt(3, index, span),
        [],
        ir.SeqStmts([sequence_stmts(function.body)[-2], ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return ir.Function(
        "rotating_replay",
        [function.params[0], (function.params[1], ir.ParamDirection.InOut)],
        function.return_types,
        ir.SeqStmts([*sequence_stmts(function.body)[:-2], loop, sequence_stmts(function.body)[-1]], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("counts", [(0, 0), (1, 1), (2, 0, 3), (4, 1, 2), (0, 3, 0), (2, 3, 2, 1)])
@pytest.mark.parametrize("slots", [2, 3, 5])
def test_actual_covers_across_sibling_resets(counts, slots):
    function = regional_fixture(counts, slots, start=1, step=2)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)


@pytest.mark.parametrize("inner", [0, 1, 2, 4])
@pytest.mark.parametrize("outer", [0, 1, 2, 3, 5])
def test_shared_repetition_original_coordinates_and_actual_covers(inner, outer):
    function = repeated_fixture(inner, outer, inner_start=1, inner_step=2)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    if inner and outer:
        d = result.describe()
        assert d["regions"][0]["count"] == inner * outer
        assert d["regions"][0]["reset_period"] == inner


def test_audit_expansion_has_an_explicit_budget():
    result = testing.analyze_placed_sync(repeated_fixture(4, 5), True)
    before = result.describe()
    with pytest.raises(ValueError, match="(budget|payload|limit)"):
        result.expand_retained_demands(2)
    assert result.describe() == before
    assert result.expand_retained_demands(60)


@pytest.mark.parametrize("inner", [4, 1000, 1000000])
def test_inner_count_does_not_expand_interfaces(inner):
    result = testing.analyze_placed_sync(repeated_fixture(inner, 7), True)
    d = result.describe()
    assert d["analyzed_occurrences"] <= 3
    assert d["interface_slot_visits"] <= 64
    assert d["interface_entries"] <= 10000
    assert d["merge_nodes"] <= 32


@pytest.mark.parametrize("outer", [2, 7, 64, 1000000])
def test_outer_count_shares_definitions_instead_of_unfolding(outer):
    result = testing.analyze_placed_sync(repeated_fixture(4, outer), True)
    d = result.describe()
    assert d["analyzed_occurrences"] <= 3
    assert d["merge_nodes"] <= 2 * outer.bit_length() + 2
    assert d["interface_entries"] <= 2048 * outer.bit_length()


@pytest.mark.parametrize("order", ["left", "right", "balanced"])
@pytest.mark.parametrize("counts", [(2, 0, 3, 1), (1, 2, 1, 2)])
def test_parenthesizations_preserve_original_covers_and_queries(order, counts):
    function = regional_fixture(counts, slots=3)
    result = testing.analyze_placed_sync(function, True, composition_order=order)
    validate(function, result)


def test_unknown_partition_policy_does_not_select_another_algorithm():
    with pytest.raises(ValueError, match="(composition|order)"):
        testing.analyze_placed_sync(regional_fixture((2, 3)), True, composition_order="unknown")


@pytest.mark.parametrize("order", ["balanced", "left", "right"])
@pytest.mark.parametrize("middle", [0, 1, 3])
def test_asymmetric_prologue_tail_missing_pipes_and_empty_child(order, middle):
    function = regional_fixture(
        (None, 2, 0, middle, None),
        schedules=[[("load", 0, 0)], [("rmw", 0, 0)], [("load", 0, 1)], [("store", 0, 1)], [("store", 0, 0)]],
    )
    result = testing.analyze_placed_sync(function, True, composition_order=order)
    validate(function, result)
    d = result.describe()
    assert d["hierarchical"]
    assert d["crossing_generators"] is None
    assert d["crossing_demands"] is None
    assert result.expand_retained_demands(32)


def test_unused_slot_allocation_does_not_change_protected_matmul_route():
    original = staged_fixture(3)
    span = original.span
    element = ir.BufferType(
        [64, 32],
        ir.DataType.FP32,
        ir.MemorySpace.Mat,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=512,
    )
    group_type = ir.MultiBufferType(element, 3)
    group = ir.Var("unused_ring", group_type, span)
    allocation = ir.AssignStmt(
        group,
        ir.Call(
            ir.get_op("buffer.alloc_multi"),
            [ir.MakeTuple([], span), ir.ConstInt(32768, ir.DataType.INDEX, span)],
            group_type,
            span,
        ),
        span,
    )
    function = ir.Function(
        original.name,
        [*original.params[:-1], (original.params[-1], ir.ParamDirection.InOut)],
        original.return_types,
        ir.SeqStmts([allocation, *sequence_stmts(original.body)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    assert not result.describe()["hierarchical"]
    assert all(not region["rotating"] for region in result.describe()["regions"])
