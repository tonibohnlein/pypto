# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent concrete effects versus structural conservative prefix selection.

These tests do not claim physical realization. The oracle uses original fixture
byte ranges, not compiler partitions, registry effects, or generated facts.
"""

import pytest
from pypto import ir, testing
from pypto.ir.op import sync

from .test_autosync_switch import clone
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def conservative_fixture(kind="for", literal_count=None):
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(value) for value in values], span)

    def var(name, ty):
        return ir.Var(name, ty, span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    a, b, out = [var(name, ir.TensorType([2, 8], fp)) for name in ("a", "b", "out")]
    count, visits, choice = [var(name, ir.ScalarType(index)) for name in ("count", "visits", "choice")]
    local = ir.BufferType([2, 8], fp, ir.MemorySpace.Vec)
    x, alias, z = [var(name, local) for name in ("x", "alias", "z")]
    definitions = [
        ir.AssignStmt(value, call("buffer.alloc", [tup([]), ci(address)], local), span)
        for value, address in ((x, 0), (alias, 0), (z, 128))
    ]
    initial = payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x])
    iv = (
        ir.IterArg("i", ir.ScalarType(index), ci(0), span)
        if kind == "while"
        else var("i", ir.ScalarType(index))
    )
    selected = ir.Eq(ir.FloorMod(iv, ci(2), index, span), choice, ir.DataType.BOOL, span)
    conditional_load = payload("buffer.load", [b, tup([0, 0]), tup([2, 8]), alias])
    compute = payload("buffer.abs", [x, z])
    store = payload("buffer.store", [z, tup([0, 0]), tup([2, 8]), out])
    conditional_write = payload("buffer.abs", [x, x])
    body: list[ir.Stmt] = [
        ir.IfStmt(selected, conditional_load, None, [], span),
        compute,
        store,
        ir.IfStmt(selected, ir.SeqStmts([], span), conditional_write, [], span),
    ]
    stop = count if literal_count is None else ci(literal_count)
    if kind == "while":
        assert isinstance(iv, ir.IterArg)
        final = var("final_i", ir.ScalarType(index))
        inner = ir.WhileStmt(
            ir.Lt(iv, stop, ir.DataType.BOOL, span),
            [iv],
            ir.SeqStmts([*body, ir.YieldStmt([ir.Add(iv, ci(1), index, span)], span)], span),
            [final],
            span,
        )
    else:
        inner = ir.ForStmt(iv, ci(0), stop, ci(1), [], ir.SeqStmts(body, span), [], span)
    outer = ir.ForStmt(var("visit", ir.ScalarType(index)), ci(0), visits, ci(1), [], inner, [], span)
    final_store = payload("buffer.store", [x, tup([0, 0]), tup([2, 8]), out])
    return ir.Function(
        "conservative",
        [a, b, (out, ir.ParamDirection.InOut), count, visits, choice],
        [out.type],
        ir.SeqStmts([*definitions, initial, outer, final_store, ir.ReturnStmt([out], span)], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def _trace(visits, count, choice):
    result = [0]
    for _ in range(visits):
        for index in range(count):
            if index % 2 == choice:
                result.append(1)
            result.extend((2, 3))
            if index % 2 != choice:
                result.append(4)
    return [*result, 5]


def _coverage(sites, trace, overlap):
    # Explicit concrete byte addresses. x and alias are distinct SSA roots at
    # the same Vec address. GM out may overlap input b; no disjoint promise.
    x, z = set(range(64)), set(range(128, 192))
    a, b = set(range(1024, 1088)), set(range(2048, 2112))
    out = b if overlap else set(range(3072, 3136))
    operations = [
        (ir.PipeType.MTE2, {("gm", byte) for byte in a}, {("vec", byte) for byte in x}),
        (ir.PipeType.MTE2, {("gm", byte) for byte in b}, {("vec", byte) for byte in x}),
        (ir.PipeType.V, {("vec", byte) for byte in x}, {("vec", byte) for byte in z}),
        (ir.PipeType.MTE3, {("vec", byte) for byte in z}, {("gm", byte) for byte in out}),
        (ir.PipeType.V, {("vec", byte) for byte in x}, {("vec", byte) for byte in x}),
        (ir.PipeType.MTE3, {("vec", byte) for byte in x}, {("gm", byte) for byte in out}),
    ]
    # Replay payload identities carried by each actual closed service. Forward
    # WAIT imports the source publication; return WAIT imports the destination
    # publication AFTER that acquire. The return SET alone does not complete
    # destination payloads on their own pipe. Unlike the compiler's must matrix,
    # these sets contain individual executed payload IDs on one concrete trace.
    ancestors = []
    issued, imported = {}, {}

    def publication(pipe):
        completed = set(imported.get(pipe, ()))
        for payload in issued.get(pipe, ()):
            completed.add(payload)
            completed.update(ancestors[payload])
        return completed

    for index, label in enumerate(trace):
        pipe, reads, writes = operations[label]
        assert sites[label]["pipe"] == pipe
        for source in sites[label]["source_pipes"]:
            imported.setdefault(pipe, set()).update(publication(source))
            if source != pipe:
                imported.setdefault(source, set()).update(publication(pipe))
        predecessors = set(imported.get(pipe, ()))
        assert all(previous < index for previous in predecessors)
        for previous in range(index):
            _, prior_reads, prior_writes = operations[trace[previous]]
            if prior_writes & (reads | writes) or prior_reads & writes:
                assert previous in predecessors, (previous, index, trace)
        ancestors.append(predecessors)
        issued.setdefault(pipe, []).append(index)


@pytest.mark.parametrize("kind", ["for", "while"])
def test_structural_services_cover_independent_effects(kind):
    function = conservative_fixture(kind)
    before = ir.serialize(function)
    analysis = testing.analyze_conservative_sync(function)
    assert ir.serialize(function) == before
    assert analysis["precision"] == "storage_domain_no_kill"
    assert not analysis["exact_storage_selectors"] and not analysis["executable_protocol"]
    sites = analysis["sites"]
    assert len(sites) == 6
    assert [len(site["control"]) for site in sites] == [0, 3, 2, 2, 3, 0]
    # Conditional/skipped writes cannot erase the initial MTE2 load obligation.
    assert ir.PipeType.MTE2 in sites[2]["source_pipes"]
    for visits in (0, 1, 3):
        for count in (0, 1, 2, 4):
            for choice in (0, 1):
                for overlap in (False, True):
                    _coverage(sites, _trace(visits, count, choice), overlap)


@pytest.mark.parametrize("kind", ["for", "while"])
def test_counts_do_not_expand_conservative_representation(kind):
    costs = [
        testing.analyze_conservative_sync(conservative_fixture(kind, count))["cost"]
        for count in (0, 1, 7, 1000000)
    ]
    assert all(cost == costs[0] for cost in costs)


@pytest.mark.parametrize("view", ["dense", "overlap", "nested", "reshape", "byte_root"])
def test_existing_descriptor_forms_keep_data_effects(view):
    result = testing.analyze_conservative_sync(vec_fixture(view=view))
    assert len(result["sites"]) == (6 if view == "overlap" else 5)
    assert {fact["space"] for fact in result["exit_effects"]} == {"DDR", "Vec"}


@pytest.mark.parametrize("condition", ["unbound", "memory"])
def test_unproved_control_readiness_is_refused(condition):
    function = vec_fixture()
    span = function.span
    if condition == "unbound":
        value = ir.Var("not_ready", ir.ScalarType(ir.DataType.BOOL), span)
    else:
        value = ir.Call(
            ir.get_op("tensor.read"),
            [function.params[0], ir.MakeTuple([], span)],
            ir.ScalarType(ir.DataType.BOOL),
            span,
        )
    candidate = clone(function, body=ir.IfStmt(value, function.body, None, [], span))
    before = ir.serialize(candidate)
    with pytest.raises(ValueError, match="ready definition|readiness.*visibility"):
        testing.analyze_conservative_sync(candidate)
    assert ir.serialize(candidate) == before


def test_unqualified_target_and_protocol_remain_unavailable():
    function = vec_fixture()
    with pytest.raises(ValueError, match="target"):
        testing.analyze_conservative_sync(function, "a5")
    token = ir.Var("already_recorded", ir.SyncTokenType(ir.PipeType.MTE2), function.span)
    record = ir.AssignStmt(token, sync.record(ir.PipeType.MTE2), function.span)
    candidate = clone(function, body=ir.SeqStmts([record, function.body], function.span))
    with pytest.raises(ValueError, match="[Bb]uffer|instruction contract"):
        testing.analyze_conservative_sync(candidate)


def test_nested_control_uses_one_retained_parent_index():
    base = vec_fixture()
    span = base.span
    assert isinstance(base.body, ir.SeqStmts)
    statements = list(base.body.stmts)
    definitions = [statement for statement in statements if isinstance(statement, ir.AssignStmt)]
    payloads = [statement for statement in statements if isinstance(statement, ir.EvalStmt)]
    compute = next(
        statement.expr
        for statement in payloads
        if isinstance(statement.expr, ir.Call) and statement.expr.op.name == ir.get_op("buffer.add").name
    )
    costs = []
    for depth in (8, 16, 32):
        nested = ir.SeqStmts([], span)
        for level in range(depth):
            variable = ir.Var(f"level_{level}", ir.ScalarType(ir.DataType.INDEX), span)
            nested = ir.ForStmt(
                variable,
                ir.ConstInt(0, ir.DataType.INDEX, span),
                ir.ConstInt(1, ir.DataType.INDEX, span),
                ir.ConstInt(1, ir.DataType.INDEX, span),
                [],
                ir.SeqStmts([ir.EvalStmt(compute, span), nested], span),
                [],
                span,
            )
        function = clone(
            base, body=ir.SeqStmts([*definitions, *payloads[:2], nested, payloads[-1], statements[-1]], span)
        )
        result = testing.analyze_conservative_sync(function)
        cost = result["cost"]
        assert cost["parent_links"] == cost["statements"] - 1
        # Inspection deliberately materializes paths; production retains no such
        # per-site vectors. Charge that separate diagnostic output honestly.
        assert cost["diagnostic_control_output_nodes"] == depth * (depth + 1) // 2
        costs.append(cost)
    for counter in ("statements", "parent_links", "retained_facts"):
        assert costs[2][counter] - costs[1][counter] == 2 * (costs[1][counter] - costs[0][counter])
