# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Provider-owned native child: original frames, pending effects and uniform proof.

The generic resident-operand loop exercises the existing native accumulation
contract. It does not import an eventful child or establish arbitrary caller
bounds. All requests reuse one whole-function retained exact analysis.
"""

import pytest
from pypto import ir, testing

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import validate
from .test_autosync_switch import clone
from .test_nested_phase_sync import nested_matrix_fixture
from .test_row_sync_analysis import row_fixture
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def resident_child_fixture(count=3, visits=2, m=64, n=64, initialize_in_child=False):
    base = staged_fixture(1, stages=1, m=m, n=n)
    span = ir.Span.unknown()
    index = ir.DataType.INDEX

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def payload(name, args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), args, ir.VoidType(), span), span)

    assert isinstance(base.body, ir.SeqStmts)
    allocations = list(base.body.stmts[:5])
    assert all(isinstance(statement, ir.AssignStmt) for statement in allocations)
    mat_a, mat_b, left, right, acc = [
        statement.var for statement in allocations if isinstance(statement, ir.AssignStmt)
    ]
    a, b, out = base.params
    args = [left, right, acc]
    coordinate = ir.Var("k", ir.ScalarType(index), span)
    compute: ir.Stmt = payload("buffer.matmul_acc", args)
    if initialize_in_child:
        compute = ir.IfStmt(
            ir.Eq(coordinate, ci(0), ir.DataType.BOOL, span),
            payload("buffer.matmul", args),
            compute,
            [],
            span,
        )
    child = ir.ForStmt(
        coordinate,
        ci(0),
        ci(count),
        ci(1),
        [],
        ir.SeqStmts([compute, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    outer = ir.ForStmt(
        ir.Var("visit", ir.ScalarType(index), span),
        ci(0),
        ci(visits),
        ci(1),
        [],
        ir.SeqStmts(
            [
                payload("buffer.load", [a, tup([0, 0]), tup([m, 32]), mat_a]),
                payload("buffer.load", [b, tup([0, 0]), tup([32, n]), mat_b]),
                payload("buffer.copy", [mat_a, left]),
                payload("buffer.copy", [mat_b, right]),
                payload("buffer.matmul", args),
                child,
                payload("buffer.store", [acc, tup([0, 0]), tup([m, n]), out]),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    function = ir.Function(
        "resident_child",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts([*allocations, outer, ir.ReturnStmt([out], span)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return function, child


def loop_scopes(statement):
    """Walk actual structured children directly, including nested For bodies."""
    if isinstance(statement, ir.ForStmt):
        yield statement
        yield from loop_scopes(statement.body)
    elif isinstance(statement, ir.SeqStmts):
        for child in statement.stmts:
            yield from loop_scopes(child)
    elif isinstance(statement, ir.IfStmt):
        yield from loop_scopes(statement.then_body)
        if statement.else_body is not None:
            yield from loop_scopes(statement.else_body)


def exported(function, child):
    mathematics = testing.analyze_placed_sync(function, True)
    result = testing.export_native_sync_child(mathematics, child)
    assert result["status"] == "ready", result
    return result, mathematics


@pytest.mark.parametrize("count", [0, 1, 3, 17])
def test_retained_native_child_has_pending_effects_and_preserves_original_scope(count):
    function, scope = resident_child_fixture(count)
    before = ir.serialize(function)
    export, _ = exported(function, scope)
    child = export["child"]
    assert child.owner is function and child.scope is scope
    baseline = testing.analyze_conservative_sync(function)
    candidate = testing.analyze_conservative_sync(function, children=[child])
    assert len(candidate["children"]) == 1
    entry = candidate["children"][0]
    assert entry["effects"] and entry["incoming"] and entry["services"]
    assert len(candidate["sites"]) + 1 == len(baseline["sites"])
    assert candidate["exit_effects"] == baseline["exit_effects"]
    logical = testing.insert_conservative_prefixes(function, children=[child])
    assert any(item is scope for item in loop_scopes(logical.body))
    assert ir.serialize(function) == before


def test_native_child_context_scope_and_target_are_not_transplantable():
    function, scope = resident_child_fixture()
    export, mathematics = exported(function, scope)
    child = export["child"]
    with pytest.raises(ValueError, match="different original function"):
        testing.analyze_conservative_sync(clone(function), children=[child])
    with pytest.raises(ValueError, match="Unsupported physical target"):
        testing.analyze_conservative_sync(function, "a5", children=[child])
    with pytest.raises(ValueError, match="duplicate"):
        testing.analyze_conservative_sync(function, children=[child, child])
    _, foreign_scope = resident_child_fixture()
    unavailable = testing.export_native_sync_child(mathematics, foreign_scope)
    assert unavailable["status"] == "missing_retained_scope" and unavailable["child"] is None
    assert testing.export_native_sync_child(mathematics, scope, "a5")["status"] == "foreign_context"


def test_internal_nonnative_demands_require_an_eventful_child_contract():
    function, child = resident_child_fixture(m=16, n=16)
    mathematics = testing.analyze_placed_sync(function, True)
    result = testing.export_native_sync_child(mathematics, child)
    assert result["status"] == "nonnative_internal_demands"
    assert result["child"] is None


def test_retained_native_child_summary_does_not_unfold_k():
    observations = []
    for count in (3, 31, 1023):
        function, child = resident_child_fixture(count)
        result, _ = exported(function, child)
        observations.append(result)
    assert len({item["source_nodes"] for item in observations}) == 1
    assert len({item["operations"] for item in observations}) == 1
    assert observations[-1]["summary_nodes"] < 64


def test_caller_specialization_is_not_implicitly_inherited_by_parent():
    base, scope = resident_child_fixture()
    unused = ir.Var("caller_coordinate", ir.ScalarType(ir.DataType.INDEX), base.span)
    function = ir.Function(
        base.name,
        [*list(zip(base.params, base.param_directions)), unused],
        base.return_types,
        base.body,
        base.span,
        base.func_type,
        ir_stage=base.ir_stage,
    )
    retained = testing.analyze_placed_sync(function, True, parameter_bounds={3: (0, 1)})
    result = testing.export_native_sync_child(retained, scope)
    assert result["status"] == "access_premises" and result["child"] is None
    assert "caller/launch" in result["reason"]


@pytest.mark.parametrize("uses_native_event", [False, True])
def test_compound_native_flag_claims_prevent_resource_empty_export(uses_native_event):
    base = row_fixture()
    span = base.span

    def ci(x):
        return ir.ConstInt(x, ir.DataType.INDEX, span)

    assert isinstance(base.body, ir.SeqStmts)
    statements = list(base.body.stmts)
    definitions, load, row = statements[:4], statements[4], statements[5]
    if uses_native_event:
        # A2/A3 TRECIP uses a private V/S event. The FP32 TROWMAX control
        # uses explicit scratch and vector instructions only.
        src, dst = definitions[0].var, definitions[1].var
        row = ir.EvalStmt(ir.Call(ir.get_op("buffer.recip"), [src, dst], ir.VoidType(), span), span)
    child = ir.ForStmt(
        ir.Var("once", ir.ScalarType(ir.DataType.INDEX), span),
        ci(0),
        ci(1),
        ci(1),
        [],
        ir.SeqStmts([row, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    outer = ir.ForStmt(
        ir.Var("visit", ir.ScalarType(ir.DataType.INDEX), span),
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts([load, child, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    function = clone(base, body=ir.SeqStmts([*definitions, outer, statements[-1]], span))
    result = testing.export_native_sync_child(testing.analyze_placed_sync(function, True), child)
    assert result["status"] == ("native_resources" if uses_native_event else "ready")
    assert (result["child"] is None) == uses_native_event


def test_missing_phase_inner_summary_remains_unavailable():
    function = nested_matrix_fixture(3, visits=2, m=64, n=64)
    retained = testing.analyze_placed_sync(function, True)
    results = [testing.export_native_sync_child(retained, loop) for loop in loop_scopes(function.body)]
    assert any(result["status"] == "unavailable_summary" for result in results)
    assert all(result["child"] is None for result in results)


def test_external_storage_does_not_inherit_disjoint_gm_from_mathematics():
    base = vec_fixture()
    span = base.span

    def ci(x):
        return ir.ConstInt(x, ir.DataType.INDEX, span)

    assert isinstance(base.body, ir.SeqStmts)
    statements = list(base.body.stmts)
    child = ir.ForStmt(
        ir.Var("once", ir.ScalarType(ir.DataType.INDEX), span),
        ci(0),
        ci(1),
        ci(1),
        [],
        ir.SeqStmts([statements[3], ir.YieldStmt([], span)], span),
        [],
        span,
    )
    outer = ir.ForStmt(
        ir.Var("visit", ir.ScalarType(ir.DataType.INDEX), span),
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts([child, *statements[4:-1], ir.YieldStmt([], span)], span),
        [],
        span,
    )
    function = clone(base, body=ir.SeqStmts([*statements[:3], outer, statements[-1]], span))
    result = testing.export_native_sync_child(testing.analyze_placed_sync(function, True), child)
    assert result["status"] == "access_premises" and result["child"] is None
    assert "external-storage alias" in result["reason"]


@pytest.mark.parametrize("count,visits", [(0, 2), (1, 0), (1, 1), (3, 2)])
def test_original_retained_child_demands_and_interfaces_match_independent_access_oracle(count, visits):
    function, scope = resident_child_fixture(count, visits)
    result, mathematics = exported(function, scope)
    assert result["child"] is not None
    validate(function, mathematics)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_native_child_preserves_first_visit_initialization_branch(count):
    function, scope = resident_child_fixture(count, initialize_in_child=True)
    result, mathematics = exported(function, scope)
    validate(function, mathematics)
    analysis = testing.analyze_conservative_sync(function, children=[result["child"]])
    assert analysis["children"][0]["scope"] is scope
    assert "if" in str(scope)
