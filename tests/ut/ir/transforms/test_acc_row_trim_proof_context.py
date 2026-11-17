# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Actual Buffer metadata proofs: shared DAGs and lexical fact boundaries.

These fixtures exercise proof admission, not payload initializedness or access
coverage: they contain no producer for the supplied Acc bytes. End-to-end
payload, matching and reuse checks live in test_acc_row_trim_bridge.py.
"""

import struct

import pytest
from pypto import backend, ir, passes, testing
from pypto.backend import BackendType

from .buffer_test_utils import statements
from .test_sync_schedule_ablation import census

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()
INDEX = ir.DataType.INDEX
BOOL = ir.DataType.BOOL


def ci(number):
    return ir.ConstInt(number, INDEX, SPAN)


def tup(items):
    return ir.MakeTuple([ci(item) if isinstance(item, int) else item for item in items], SPAN)


def call(name, args, kind=None):
    return ir.Call(ir.get_op(name), args, kind or ir.VoidType(), SPAN)


def evaluate(name, args):
    return ir.EvalStmt(call(name, args), SPAN)


def seq(items):
    return ir.SeqStmts(items, SPAN)


def fixture(make_body, m=32, n=64):
    row = ir.Var("rows", ir.ScalarType(INDEX), SPAN)
    out = ir.Var("out", ir.TensorType([m, n], ir.DataType.FP32), SPAN)
    kind = ir.BufferType(
        [m, n],
        ir.DataType.FP32,
        ir.MemorySpace.Acc,
        valid_shape=[-1, -1],
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=1024,
    )
    acc = ir.Var("store_alias", kind, SPAN)

    def store(rows):
        return seq(
            [
                evaluate("buffer.set_validshape", [acc, tup([rows, n])]),
                evaluate("buffer.store", [acc, tup([0, 0]), tup([rows, n]), out]),
            ]
        )

    body = seq(
        [
            ir.AssignStmt(acc, call("buffer.alloc", [tup([m, n]), ci(0)], kind), SPAN),
            *make_body(row, store),
            ir.ReturnStmt([out], SPAN),
        ]
    )
    return ir.Function(
        "acc_metadata_context",
        [row, (out, ir.ParamDirection.InOut)],
        [out.type],
        body,
        SPAN,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def lower(function):
    program = ir.Program([function], "AccMetadataContexts", SPAN)
    with passes.PassContext([], enable_pypto_autosync=True):
        return passes.lower_sync()(program)


def bounded_guard(row, m, inverse=False):
    if inverse:
        return ir.Or(
            ir.Le(row, ci(0), BOOL, SPAN),
            ir.Gt(row, ci(m), BOOL, SPAN),
            BOOL,
            SPAN,
        )
    return ir.And(
        ir.Gt(row, ci(0), BOOL, SPAN),
        ir.Le(row, ci(m), BOOL, SPAN),
        BOOL,
        SPAN,
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_public_lower_shared_boolean_dag_keeps_atomic_guard_facts(inverse, named):
    def body(row, store):
        guard = bounded_guard(row, 32, inverse)
        # Linear DAG with exponentially many paths: lowering must visit distinct
        # syntax, and must not hand the shared tree to a recursive simplifier.
        for _ in range(28):
            guard = (ir.Or if inverse else ir.And)(guard, guard, BOOL, SPAN)
        definitions = []
        if named:
            for i in range(28):
                alias = ir.Var(f"guard_{i}", ir.ScalarType(BOOL), SPAN)
                definitions.append(ir.AssignStmt(alias, guard, SPAN))
                guard = alias
        native = store(row)
        empty = seq([])
        branch = ir.IfStmt(guard, empty if inverse else native, native if inverse else None, [], SPAN)
        return [*definitions, branch]

    result = lower(fixture(body))
    stores = [
        statement
        for statement in statements(result)
        if isinstance(statement, ir.EvalStmt)
        and isinstance(statement.expr, ir.Call)
        and statement.expr.op.name == ir.get_op("buffer.store").name
    ]
    assert len(stores) == 1


def test_public_lower_shared_min_max_dag_has_many_consumer_definitions():
    def body(row, store):
        expression = ir.Min(ir.Max(row, ci(1), INDEX, SPAN), ci(32), INDEX, SPAN)
        for i in range(28):
            expression = (ir.Min if i % 2 else ir.Max)(expression, expression, INDEX, SPAN)
        definitions, consumers = [], []
        for i in range(32):
            value = ir.Var(f"consumer_rows_{i}", ir.ScalarType(INDEX), SPAN)
            definitions.append(ir.AssignStmt(value, expression, SPAN))
            consumers.append(store(value))
        return [*definitions, *consumers]

    result = lower(fixture(body))
    assert (
        sum(
            isinstance(statement, ir.EvalStmt)
            and isinstance(statement.expr, ir.Call)
            and statement.expr.op.name == ir.get_op("buffer.store").name
            for statement in statements(result)
        )
        == 32
    )


@pytest.mark.parametrize("location", ["after", "else"])
def test_guard_bounds_do_not_leak_to_unproved_store_context(location):
    def body(row, store):
        good = store(row)
        # A fresh setter establishes actual metadata in the bad context too, so
        # the failure must be the missing positive/bounded-row proof, not stale
        # metadata ownership or the absence of a dominating setter.
        bad = store(row)
        branch = ir.IfStmt(bounded_guard(row, 32), good, bad if location == "else" else None, [], SPAN)
        return [branch, bad] if location == "after" else [branch]

    function = fixture(body)
    original = function.body
    with pytest.raises(ValueError, match="Runtime Acc store requires proved positive rows"):
        lower(function)
    assert function.body.same_as(original)


@pytest.mark.parametrize("m,n", [(8208, 16), (16, 4096)])
def test_native_valid_size_fields_are_not_inferred_from_allocation_capacity(m, n):
    def body(row, store):
        fixed = ir.Var("one_row", ir.ScalarType(INDEX), SPAN)
        return [ir.AssignStmt(fixed, ci(1), SPAN), store(fixed)]

    with pytest.raises(ValueError, match="Runtime Acc aliases require the qualified row-only"):
        lower(fixture(body, m, n))


def metric_fixture(depth):
    def body(row, store):
        expression = ir.Min(ir.Max(row, ci(1), INDEX, SPAN), ci(32), INDEX, SPAN)
        for _ in range(depth):
            expression = ir.Min(expression, expression, INDEX, SPAN)
        rows = ir.Var("bounded_rows", ir.ScalarType(INDEX), SPAN)
        return [ir.AssignStmt(rows, expression, SPAN), store(rows)]

    return fixture(body)


@pytest.mark.parametrize("depth", [0, 3, 8])
def test_weighted_schedule_size_preserves_small_expanded_census(depth):
    result = testing.lower_sync_schedule(metric_fixture(depth))
    expected = census(result["function"])
    assert result["schedule"]["emitted_control_nodes"] == expected.nodes
    assert result["schedule"]["emitted_payload_sites"] == expected.payloads


def test_weighted_schedule_size_counts_deep_shared_dag_without_expanding_it():
    result = testing.lower_sync_schedule(metric_fixture(28))
    assert result["schedule"]["emitted_control_nodes"] > 1 << 28
    assert result["schedule"]["emitted_control_nodes"] < 1 << 40


def test_weighted_schedule_size_overflow_remains_explicitly_unproved():
    result = testing.lower_sync_schedule(metric_fixture(80))
    maximum = (1 << (8 * struct.calcsize("P"))) - 1
    assert result["schedule"]["emitted_control_nodes"] == maximum
    assert any(
        row["selected"] and row["status"] == "cost_unproved" and "size" in row["reason"]
        for row in result["schedule"]["choices"]
    )


def test_original_counted_coordinate_is_ready_without_inferred_trip_bounds():
    def body(row, store):
        coordinate = ir.Var("original_iteration", ir.ScalarType(INDEX), SPAN)
        rows = ir.Var("iteration_rows", ir.ScalarType(INDEX), SPAN)
        bounded = ir.Min(ir.Max(coordinate, ci(1), INDEX, SPAN), ci(32), INDEX, SPAN)
        return [
            ir.ForStmt(
                coordinate,
                ci(0),
                row,
                ci(1),
                [],
                seq([ir.AssignStmt(rows, bounded, SPAN), store(rows)]),
                [],
                SPAN,
            )
        ]

    result = lower(fixture(body))
    assert any(isinstance(statement, ir.ForStmt) for statement in statements(result))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_aic_core_kind_does_not_qualify_runtime_acc_store_on_a5(ascend_backend):
    assert ascend_backend == BackendType.Ascend950
    with pytest.raises(ValueError, match="Runtime Acc row stores require a qualified selected A2/A3"):
        lower(metric_fixture(0))


def test_unconfigured_backend_keeps_runtime_acc_native_contract_unproved():
    function = metric_fixture(0)
    backend.reset_for_testing()
    try:
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.BufferIR)
        diagnostics = passes.PropertyVerifierRegistry.verify(
            properties,
            ir.Program([function], "UnqualifiedAccNativeTarget", SPAN),
        )
        assert any("qualified selected A2/A3" in diagnostic.message for diagnostic in diagnostics)
    finally:
        backend.set_backend_type(BackendType.Ascend910)
