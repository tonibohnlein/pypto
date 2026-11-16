# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Expression reuse cannot hide changed lexical or representation contexts.

Depth28 tests deliberately retain the shared graph; neither a printed expanded
function nor an unfolded tree is needed to validate it. External watchdogs own
scaling qualification rather than machine-sensitive timing assertions.
"""

import pytest
from pypto import ir, passes

SPAN = ir.Span.unknown()
INDEX = ir.ScalarType(ir.DataType.INDEX)
pytestmark = pytest.mark.usefixtures("ascend_backend")


def ci(number):
    return ir.ConstInt(number, ir.DataType.INDEX, SPAN)


def seq(items):
    return ir.SeqStmts(items, SPAN)


def guard(variable, depth):
    result = ir.Gt(variable, ci(0), ir.DataType.BOOL, SPAN)
    for _ in range(depth):
        result = ir.And(result, result, ir.DataType.BOOL, SPAN)
    return result


def branch(condition, body=None, other=None):
    return ir.IfStmt(condition, body if body is not None else seq([]), other, [], SPAN)


def program(body, parameters=(), *, selected=False, function_type=ir.FunctionType.AIV):
    function = ir.Function(
        "dag_kernel",
        list(parameters),
        [],
        seq([*body, ir.ReturnStmt(SPAN)]),
        SPAN,
        type=function_type,
        ir_stage=ir.FunctionIRStage.Buffer,
        attrs={"buffer_ir_selected": True} if selected else None,
    )
    return ir.Program([function], "VerifierDag", SPAN)


def verify(source, property_=passes.IRProperty.BufferIR):
    properties = passes.IRPropertySet()
    properties.insert(property_)
    return passes.PropertyVerifierRegistry.verify(properties, source)


@pytest.mark.parametrize("depth", [0, 4, 28])
@pytest.mark.parametrize("selected", [False, True])
def test_public_buffer_verification_accepts_shared_guard(depth, selected):
    coordinate = ir.Var("coordinate", INDEX, SPAN)
    condition = guard(coordinate, depth)
    source = program([branch(condition), branch(condition)], [coordinate], selected=selected)
    assert not verify(source)
    function = source.get_function("dag_kernel")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    first, second = function.body.stmts[:2]
    assert isinstance(first, ir.IfStmt) and isinstance(second, ir.IfStmt)
    assert first.condition.same_as(condition)
    assert second.condition.same_as(condition)


@pytest.mark.parametrize(
    "property_", [passes.IRProperty.BufferIR, passes.IRProperty.SSAForm, passes.IRProperty.UseAfterDef]
)
@pytest.mark.parametrize("depth", [0, 28])
def test_shared_guard_before_definition_still_rejects_after_later_valid_use(depth, property_):
    late = ir.Var("late", INDEX, SPAN)
    condition = guard(late, depth)
    source = program([branch(condition), ir.AssignStmt(late, ci(1), SPAN), branch(condition)])
    diagnostics = verify(source, property_)
    assert any("late" in item.message for item in diagnostics)
    # Do not expand the entire shared expression when reporting its first use.
    assert all(len(item.message) < 1000 for item in diagnostics)


@pytest.mark.parametrize("depth", [0, 28])
@pytest.mark.parametrize("sibling_first", [False, True])
def test_valid_branch_use_cannot_hide_sibling_or_join_scope_error(depth, sibling_first):
    local = ir.Var("local", INDEX, SPAN)
    condition = guard(local, depth)
    defined = seq([ir.AssignStmt(local, ci(1), SPAN), branch(condition)])
    unbound = seq([branch(condition)])
    then, otherwise = (unbound, defined) if sibling_first else (defined, unbound)
    source = program([branch(ir.ConstBool(True, SPAN), then, otherwise), branch(condition)])
    diagnostics = verify(source)
    assert any(item.rule_name == "SSAVerify" and "local" in item.message for item in diagnostics)
    assert any(item.rule_name == "UseAfterDefCheck" and "local" in item.message for item in diagnostics)


@pytest.mark.parametrize("depth", [0, 28])
def test_loop_local_guard_does_not_become_visible_after_reset(depth):
    coordinate = ir.Var("loop_index", INDEX, SPAN)
    condition = guard(coordinate, depth)
    loop = ir.ForStmt(coordinate, ci(0), ci(2), ci(1), [], seq([branch(condition)]), [], SPAN)
    diagnostics = verify(program([loop, branch(condition)]))
    assert any("loop_index" in item.message for item in diagnostics)


def test_same_allocation_call_requires_assign_at_every_statement_occurrence():
    kind = ir.BufferType([2, 8], ir.DataType.INT32, ir.MemorySpace.Vec)
    destination = ir.Var("destination", kind, SPAN)
    allocation = ir.Call(ir.Op("buffer.alloc"), [ir.MakeTuple([], SPAN)], {}, None, kind, SPAN)
    source = program([ir.AssignStmt(destination, allocation, SPAN), ir.EvalStmt(allocation, SPAN)])
    assert any("direct AssignStmt" in item.message for item in verify(source))


def test_reused_native_call_and_definition_occurrences_are_not_memoized():
    kind = ir.BufferType([2, 8], ir.DataType.INT32, ir.MemorySpace.Vec)
    destination = ir.Var("destination", kind, SPAN)
    allocation = ir.Call(ir.Op("buffer.alloc"), [ir.MakeTuple([], SPAN)], {}, None, kind, SPAN)
    assigned = ir.AssignStmt(destination, allocation, SPAN)
    source = program([assigned, assigned])
    assert any(
        item.rule_name == "SSAVerify" and "assigned more than once" in item.message for item in verify(source)
    )


def test_native_operand_use_does_not_authorize_same_expression_in_attributes():
    kind = ir.BufferType([2, 8], ir.DataType.INT32, ir.MemorySpace.Vec)
    destination = ir.Var("destination", kind, SPAN)
    scalar = ir.Var("scalar", ir.ScalarType(ir.DataType.INT32), SPAN)
    read = ir.Call(
        ir.Op("buffer.read"), [destination, ir.MakeTuple([ci(0), ci(0)], SPAN)], {}, None, scalar.type, SPAN
    )
    hidden = ir.Call(ir.Op("system.bar_all"), [], {"core_type": "aiv"}, {"hidden": read}, ir.VoidType(), SPAN)
    diagnostics = verify(
        program([ir.AssignStmt(scalar, read, SPAN), ir.EvalStmt(hidden, SPAN)], [destination])
    )
    assert any("cannot carry buffer handles" in item.message for item in diagnostics)


@pytest.mark.parametrize("depth", [0, 28])
def test_scalar_metadata_graph_keeps_its_original_type_use_contract(depth):
    length = ir.Var("length", INDEX, SPAN)
    expression = length
    for _ in range(depth):
        expression = ir.Min(expression, expression, ir.DataType.INDEX, SPAN)
    tensor = ir.Var("tensor", ir.TensorType([expression, ci(8)], ir.DataType.FP32), SPAN)
    # Signature dimensions are established entry bindings by the existing ABI
    # contract; this test introduces no range or caller-value assumptions.
    source = program([branch(guard(length, depth))], [tensor], selected=True)
    assert not verify(source)


@pytest.mark.parametrize("depth", [0, 28])
def test_nested_call_prohibition_survives_prior_exempt_submit_predicate(depth):
    coordinate = ir.Var("coordinate", INDEX, SPAN)
    call = ir.Call(ir.Op("test.scalar"), [coordinate], {}, None, INDEX, SPAN)
    condition = guard(call, depth)
    # NoNestedCall deliberately skips Submit.predicate; the same node in an
    # evaluated branch must nevertheless be checked from a fresh use/context.
    task = ir.Submit(
        ir.GlobalVar("external"),
        [],
        [],
        {},
        None,
        ir.ScalarType(ir.DataType.TASK_ID),
        SPAN,
        predicate=condition,
    )
    source = program([ir.EvalStmt(task, SPAN), branch(condition)], [coordinate])
    diagnostics = verify(source, passes.IRProperty.NoNestedCalls)
    assert any(item.rule_name == "NoNestedCall" for item in diagnostics)


def test_one_call_root_distinguishes_ordinary_operand_from_hidden_attribute():
    kind = ir.BufferType([2, 8], ir.DataType.INT32, ir.MemorySpace.Vec)
    destination = ir.Var("destination", kind, SPAN)
    scalar = ir.Var("scalar", ir.ScalarType(ir.DataType.INT32), SPAN)
    # The operand is visited first in the same outer Call traversal. Its valid
    # ordinary role must not authorize the attribute's forbidden storage use.
    read = ir.Call(
        ir.Op("buffer.read"),
        [destination, ir.MakeTuple([ci(0), ci(0)], SPAN)],
        {},
        {"hidden": destination},
        scalar.type,
        SPAN,
    )
    diagnostics = verify(program([ir.AssignStmt(scalar, read, SPAN)], [destination]))
    assert any("Call attribute 'hidden' cannot carry buffer handles" in item.message for item in diagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_direct_buffer_call_cannot_borrow_statement_role_in_result_metadata():
    scalar = ir.Call(ir.Op("buffer.get_block_idx"), [], {}, None, INDEX, SPAN)
    result = ir.Var("result", ir.TensorType([scalar, ci(8)], ir.DataType.FP32), SPAN)
    diagnostics = verify(program([ir.AssignStmt(result, scalar, SPAN)]))
    assert any("require a direct AssignStmt" in item.message for item in diagnostics)
