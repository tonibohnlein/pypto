# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller-local correlations refine numerical evidence without ABI assumptions."""

import pytest
from pypto import ir, passes, testing

from .sync_physical_oracle import run_physical
from .test_sync_range_proofs import range_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


class CallerFixture:
    """Build real caller cuts and distinct scalar callees for range assertions."""

    def __init__(self, dtype=ir.DataType.INT32, submit=False):
        self.dtype = dtype
        self.submit = submit
        self.iv = self.var("band")
        self.external = self.var("band")  # Distinct identity deliberately shares its name.
        self.children = {}

    def ci(self, value):
        return ir.ConstInt(value, self.dtype, SPAN)

    def var(self, name):
        return ir.Var(name, ir.ScalarType(self.dtype), SPAN)

    def binary(self, operation, left, right):
        return operation(left, right, self.dtype, SPAN)

    def call(self, name, value):
        if name not in self.children:
            parameter = self.var("argument")
            self.children[name] = ir.Function(
                name, [parameter], [], ir.ReturnStmt([], SPAN), SPAN, ir.FunctionType.AIV
            )
        if self.submit:
            value = ir.Submit(
                ir.GlobalVar(name),
                [value],
                [],
                {},
                {},
                ir.TupleType([ir.ScalarType(ir.DataType.TASK_ID)]),
                SPAN,
            )
        else:
            value = ir.Call(ir.GlobalVar(name), [value], ir.VoidType(), SPAN)
        return ir.EvalStmt(value, SPAN)

    def program(self, statements, stop=4, after=()):
        loop = ir.ForStmt(
            self.iv,
            self.ci(0),
            self.ci(stop),
            self.ci(1),
            [],
            ir.SeqStmts([*statements, ir.YieldStmt([], SPAN)], SPAN),
            [],
            SPAN,
        )
        main = ir.Function(
            "main",
            [self.external],
            [],
            ir.SeqStmts([loop, *after, ir.ReturnStmt([], SPAN)], SPAN),
            SPAN,
            ir.FunctionType.Orchestration,
        )
        return ir.Program([*self.children.values(), main], "CallerCorrelations", SPAN)


def _bounds(program, name):
    return testing.sync_parameter_bounds(program).get(name, {})


@pytest.mark.parametrize("submit", [False, True])
def test_original_definition_relations_reach_call_and_submit(submit):
    f = CallerFixture(submit=submit)
    blocks, rotation = f.var("blocks"), f.var("rotation")
    statements = [
        ir.AssignStmt(blocks, f.binary(ir.Mul, f.ci(17), f.iv), SPAN),
        ir.AssignStmt(rotation, f.binary(ir.Mul, f.ci(6), f.iv), SPAN),
        f.call("difference", f.binary(ir.Sub, blocks, rotation)),
        f.call("fixed", f.binary(ir.Sub, f.binary(ir.Add, blocks, f.ci(16)), blocks)),
    ]
    program = f.program(statements)
    assert _bounds(program, "difference") == {0: (0, 33)}
    assert _bounds(program, "fixed") == {0: (16, 16)}


@pytest.mark.parametrize("dtype", [ir.DataType.INT32, ir.DataType.UINT32, ir.DataType.INDEX])
def test_original_comparison_scopes_and_false_arm_restore(dtype):
    f = CallerFixture(dtype)
    condition = ir.Lt(f.iv, f.ci(2), ir.DataType.BOOL, SPAN)
    branch = ir.IfStmt(condition, f.call("then_arm", f.iv), f.call("else_arm", f.iv), [], SPAN)
    program = f.program([branch, f.call("after_join", f.iv)])
    assert _bounds(program, "then_arm") == {0: (0, 1)}
    assert _bounds(program, "else_arm") == {0: (2, 3)}
    assert _bounds(program, "after_join") == {0: (0, 3)}


def test_branch_local_definition_and_nested_constraint_do_not_escape():
    f = CallerFixture()
    local = f.var("local")
    inner = ir.IfStmt(
        ir.Ge(f.iv, f.ci(1), ir.DataType.BOOL, SPAN),
        f.call("nested", f.iv),
        None,
        [],
        SPAN,
    )
    branch = ir.IfStmt(
        ir.Lt(f.iv, f.ci(2), ir.DataType.BOOL, SPAN),
        ir.SeqStmts(
            [
                ir.AssignStmt(local, f.binary(ir.Add, f.iv, f.ci(10)), SPAN),
                f.call("local", local),
                inner,
                f.call("after_inner", f.iv),
            ],
            SPAN,
        ),
        f.call("sibling", f.iv),
        [],
        SPAN,
    )
    # Deliberately shared out-of-scope node: it must not acquire a numerical
    # certificate just because the producer previously visited its definition.
    program = f.program([branch, f.call("escaped", local), f.call("joined", f.iv)])
    assert _bounds(program, "local") == {0: (10, 11)}
    assert _bounds(program, "nested") == {0: (1, 1)}
    assert _bounds(program, "after_inner") == {0: (0, 1)}
    assert _bounds(program, "sibling") == {0: (2, 3)}
    assert _bounds(program, "escaped") == {}
    assert _bounds(program, "joined") == {0: (0, 3)}


@pytest.mark.parametrize("kind", ["same_name", "negative", "narrow", "overflow", "overflow_alias"])
def test_unproved_original_operations_are_not_erased_by_simplification(kind):
    f = CallerFixture(ir.DataType.INT8)
    statements = []
    if kind == "same_name":
        expression = f.binary(ir.Sub, f.iv, f.external)
    elif kind == "negative":
        expression = f.binary(ir.Sub, f.iv, f.ci(3))
    elif kind == "narrow":
        wide = ir.Add(
            ir.Cast(f.iv, ir.DataType.INT32, SPAN),
            ir.ConstInt(128, ir.DataType.INT32, SPAN),
            ir.DataType.INT32,
            SPAN,
        )
        narrowed = ir.Cast(wide, ir.DataType.INT8, SPAN)
        expression = f.binary(ir.Sub, narrowed, narrowed)
    else:
        value = f.binary(ir.Add, f.iv, f.ci(127))
        if kind == "overflow_alias":
            alias = f.var("overflowing")
            statements.append(ir.AssignStmt(alias, value, SPAN))
            value = alias
        expression = f.binary(ir.Sub, value, value)
    statements.append(f.call("child", expression))
    assert _bounds(f.program(statements), "child") == {}


def test_later_branch_cannot_legitimize_an_earlier_overflowing_definition():
    f = CallerFixture(ir.DataType.INT8)
    bad = f.var("earlier")
    good = f.var("inside")
    condition = ir.Eq(f.iv, f.ci(0), ir.DataType.BOOL, SPAN)
    statements = [
        ir.AssignStmt(bad, f.binary(ir.Add, f.iv, f.ci(127)), SPAN),
        ir.IfStmt(
            condition,
            ir.SeqStmts(
                [
                    f.call("unproved", f.binary(ir.Sub, bad, bad)),
                    ir.AssignStmt(good, f.binary(ir.Add, f.iv, f.ci(127)), SPAN),
                    f.call("proved", f.binary(ir.Sub, good, good)),
                ],
                SPAN,
            ),
            None,
            [],
            SPAN,
        ),
    ]
    program = f.program(statements)
    assert _bounds(program, "unproved") == {}
    assert _bounds(program, "proved") == {0: (0, 0)}


def test_unknown_second_call_still_poisons_joined_evidence():
    f = CallerFixture()
    statements = [f.call("child", f.binary(ir.Sub, f.binary(ir.Add, f.iv, f.ci(16)), f.iv))]
    unknown_call = f.call("child", f.external)
    assert _bounds(f.program(statements, after=[unknown_call]), "child") == {}


def test_unknown_predicate_does_not_create_caller_assumptions():
    f = CallerFixture()
    branch = ir.IfStmt(
        ir.Lt(f.external, f.ci(2), ir.DataType.BOOL, SPAN),
        f.call("child", f.external),
        None,
        [],
        SPAN,
    )
    assert _bounds(f.program([branch]), "child") == {}


def test_correlated_caller_offset_reaches_real_dma_sync_consumer():
    function = range_fixture()
    dtype, index = ir.DataType.INT32, ir.DataType.INDEX
    rows = ir.ConstInt(32, index, SPAN)
    tensor_type = ir.TensorType([32, 64], ir.DataType.FP32)
    source, output = [ir.Var(name, tensor_type, SPAN) for name in ["source", "output"]]
    iv = ir.Var("base", ir.ScalarType(dtype), SPAN)
    shifted = ir.Var("shifted", ir.ScalarType(dtype), SPAN)

    def ci(value):
        return ir.ConstInt(value, dtype, SPAN)

    offset = ir.Sub(shifted, iv, dtype, SPAN)
    loop = ir.ForStmt(
        iv,
        ci(0),
        ci(4),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(shifted, ir.Add(iv, ci(16), dtype, SPAN), SPAN),
                ir.EvalStmt(
                    ir.Call(ir.GlobalVar(function.name), [source, offset, output, rows], tensor_type, SPAN),
                    SPAN,
                ),
                ir.YieldStmt([], SPAN),
            ],
            SPAN,
        ),
        [],
        SPAN,
    )
    caller = ir.Function("main", [source, output], [], loop, SPAN, ir.FunctionType.Orchestration)
    program = ir.Program([function, caller], "CorrelatedDMA", SPAN)
    assert _bounds(program, function.name) == {1: (16, 16), 3: (32, 32)}
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        physical = passes.realize_sync()(program)
    actual = next(f for f in physical.functions.values() if f.name == function.name)
    assert dict(actual.attrs)["pypto_autosync"]
    # Independent finite token occupancy, payload-order and storage oracle.
    run_physical(function, actual, range(6), parameters={1: 16, 3: 32})


def test_while_carried_and_phi_values_remain_unproved():
    f = CallerFixture()
    carried = ir.IterArg("carried", ir.ScalarType(f.dtype), f.ci(0), SPAN)
    final = f.var("final")
    loop = ir.WhileStmt(
        ir.Lt(carried, f.ci(3), ir.DataType.BOOL, SPAN),
        [carried],
        ir.SeqStmts(
            [
                f.call("carried_child", carried),
                ir.YieldStmt([f.binary(ir.Add, carried, f.ci(1))], SPAN),
            ],
            SPAN,
        ),
        [final],
        SPAN,
    )
    program = f.program([loop, f.call("phi_child", final)])
    assert _bounds(program, "carried_child") == {}
    assert _bounds(program, "phi_child") == {}
