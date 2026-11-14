# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static-site scaling, independently of trip count and storage geometry."""

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import validate

pytestmark = pytest.mark.usefixtures("ascend_backend")


def scaling_fixture(size, *, diagonal=False, count=3, offsets=None, read_hole=False):
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    def emit(name, args):
        return ir.EvalStmt(call(name, args), span)

    a = ir.Var("input", ir.TensorType([2, 8], fp), span)
    out = ir.Var("output", ir.TensorType([2 * size, 8 * size] if diagonal else [2, 8], fp), span)
    ty = ir.BufferType([2, 8], fp, ir.MemorySpace.Vec)
    buf = ir.Var("buf", ty, span)
    body: list[ir.Stmt] = [
        ir.AssignStmt(buf, call("buffer.alloc", [tup([]), ci(0)], ty), span),
        emit("buffer.load", [a, tup([0, 0]), tup([2, 8]), buf]),
    ]
    if diagonal:
        offsets = offsets or [(2 * i, 8 * i) for i in range(size)]
        body.extend(emit("buffer.store", [buf, tup(offset), tup([2, 8]), out]) for offset in offsets)
        if read_hole:
            body.append(emit("buffer.load", [out, tup([0, 8 * (size - 1)]), tup([2, 8]), buf]))
    else:
        iv = ir.Var("i", ir.ScalarType(index), span)
        loop = [emit("buffer.abs", [buf, buf]) for _ in range(size)]
        body.append(
            ir.ForStmt(
                iv, ci(0), ci(count), ci(1), [], ir.SeqStmts([*loop, ir.YieldStmt([], span)], span), [], span
            )
        )
        body.append(emit("buffer.store", [buf, tup([0, 0]), tup([2, 8]), out]))
    body.append(ir.ReturnStmt([out], span))
    return ir.Function(
        "scaling",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(body, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_flat_frontiers_match_expanded_access_oracle(size, count):
    function = scaling_fixture(size, count=count)
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_sparse_partition_preserves_selectors_and_covers(size):
    function = scaling_fixture(size, diagonal=True)
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("size", [8, 16, 32, 64, 128, 256])
def test_static_loop_frontier_state_is_linear(size):
    description = testing.analyze_placed_sync(scaling_fixture(size), True).describe()
    loop = next(r for r in description["regions"] if len(r["sites"]) == size)
    assert loop["core_frontiers"]
    assert loop["distance_entries"] == 0
    assert loop["retained_frontier_entries"] == 4 * size
    assert loop["shortest_path_runs"] == 2
    # Composition retains actual crossing/native endpoints, not each site's
    # first/final occurrence. The load/loop/store interface is constant here.
    assert description["port_pair_tests"] == 24
    assert description["boundary_graph_edges"] <= 24


@pytest.mark.parametrize("size", [8, 16, 32, 64, 128, 256])
def test_diagonal_store_partition_and_membership_are_output_sensitive(size):
    description = testing.analyze_placed_sync(scaling_fixture(size, diagonal=True), True).describe()
    assert description["cells"] <= 3 * size + 2
    assert description["class_tests"] <= 3 * size + 4
    assert description["class_lookup_tests"] <= 32 * size * (size.bit_length() + 1)


@pytest.mark.parametrize(
    "offsets",
    [
        [(0, 0), (1, 0), (2, 4), (2, 0)],
        [(0, 0), (0, 0), (2, 0), (4, 8)],
        [(0, 0), (0, 4), (1, 4), (2, 8)],
    ],
)
def test_active_column_multiplicity_and_read_only_holes(offsets):
    # Shared column boundaries must survive other writers ending. The last
    # read lies in an unwritten GM hole and must retain its reader selector.
    function = scaling_fixture(4, diagonal=True, offsets=offsets, read_hole=True)
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("count", [0, 1, 3])
def test_interior_crossing_ports_and_empty_middle_region(count):
    # The second buffer's V writer is inside the pipe chain, not its first or
    # last site. Its explicit crossing endpoints must survive port compaction.
    function = scaling_fixture(2, count=count)
    span = function.span
    assert isinstance(function.body, ir.SeqStmts)
    body = list(function.body.stmts)
    assert isinstance(body[0], ir.AssignStmt)
    buf = body[0].var
    other = ir.Var("other", buf.type, span)
    allocation = body[0].value
    assert isinstance(allocation, ir.Call)
    second = ir.AssignStmt(
        other,
        ir.Call(
            allocation.op, [allocation.args[0], ir.ConstInt(1024, ir.DataType.INDEX, span)], buf.type, span
        ),
        span,
    )
    assert isinstance(body[1], ir.EvalStmt)
    load = body[1].expr
    assert isinstance(load, ir.Call)
    second_load = ir.EvalStmt(ir.Call(load.op, [*load.args[:-1], other], load.type, span), span)
    loop = body[2]
    assert isinstance(loop, ir.ForStmt) and isinstance(loop.body, ir.SeqStmts)
    middle = ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [other, other], ir.VoidType(), span), span)
    loop = ir.ForStmt(
        loop.loop_var,
        loop.start,
        loop.stop,
        loop.step,
        [],
        ir.SeqStmts([loop.body.stmts[0], middle, *loop.body.stmts[1:]], span),
        [],
        span,
    )
    assert isinstance(body[3], ir.EvalStmt)
    store = body[3].expr
    assert isinstance(store, ir.Call)
    second_store = ir.EvalStmt(ir.Call(store.op, [other, *store.args[1:]], store.type, span), span)
    body = [body[0], second, body[1], second_load, loop, second_store, *body[3:]]
    from_function = ir.Function(
        function.name,
        [(p, d) for p, d in zip(function.params, function.param_directions, strict=True)],
        function.return_types,
        ir.SeqStmts(body, span),
        span,
        function.func_type,
        ir_stage=function.ir_stage,
    )
    validate(from_function, testing.analyze_placed_sync(from_function, True))
