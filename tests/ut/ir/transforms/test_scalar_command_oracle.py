# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The independent command oracle must not invent scalar-to-all-pipe fences."""

import pytest
from pypto import ir

from .sync_closed_command_oracle import ClosedCommands, check_closed_commands
from .test_closed_prefix_sync import Services

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def oracle_fixture(kind, count=1):
    """Explicit local initial bytes are supplied by the test, not inferred from IR."""
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(number):
        return ir.ConstInt(number, index, SPAN)

    def tup(numbers):
        return ir.MakeTuple([ci(number) for number in numbers], SPAN)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), SPAN)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), SPAN)

    scalar_dtype = ir.DataType.INT32 if kind in ("for", "while") else fp
    x_type = ir.BufferType([2, 8], scalar_dtype, ir.MemorySpace.Vec)
    y_type = ir.BufferType([2, 8], fp, ir.MemorySpace.Vec)
    x, y = ir.Var("x", x_type, SPAN), ir.Var("y", y_type, SPAN)
    scalar = ir.Var("scalar", ir.ScalarType(scalar_dtype), SPAN)
    setup = [
        ir.AssignStmt(x, call("buffer.alloc", [tup([]), ci(0)], x_type), SPAN),
        ir.AssignStmt(y, call("buffer.alloc", [tup([]), ci(128)], y_type), SPAN),
    ]
    read = ir.AssignStmt(scalar, call("buffer.read", [x, tup([0, 0])], scalar.type), SPAN)
    write = payload("buffer.write", [x, tup([0, 0]), ir.ConstFloat(1.0, fp, SPAN)])
    independent = payload("buffer.abs", [y, y])
    services = Services()
    if kind == "write_hazard":
        body = [write, payload("buffer.abs", [x, x])]
    elif kind == "write_independent":
        body = [write, independent]
    elif kind == "unused_read":
        body = [read, independent]
    elif kind == "same_pipe":
        body = [write, read]
    elif kind == "value":
        body = [read, payload("buffer.adds", [y, scalar, y])]
    elif kind == "branch":
        condition = ir.Gt(scalar, ir.ConstFloat(0.0, fp, SPAN), ir.DataType.BOOL, SPAN)
        selected = ir.SeqStmts(services.closed(ir.PipeType.MTE2, ir.PipeType.V), SPAN)
        body = [read, ir.IfStmt(condition, selected, None, [], SPAN), independent]
    else:
        assert kind in ("for", "while")
        initial = ir.Cast(scalar, index, SPAN)
        commands = services.closed(ir.PipeType.MTE2, ir.PipeType.V)
        if kind == "for":
            loop = ir.ForStmt(
                ir.Var("i", ir.ScalarType(index), SPAN),
                ci(0),
                initial,
                ci(1),
                [],
                ir.SeqStmts([*commands, ir.YieldStmt([], SPAN)], SPAN),
                [],
                SPAN,
            )
        else:
            carry = ir.IterArg("remaining", ir.ScalarType(index), initial, SPAN)
            result = ir.Var("done_remaining", ir.ScalarType(index), SPAN)
            condition = ir.Gt(carry, ci(0), ir.DataType.BOOL, SPAN)
            loop_body = ir.SeqStmts(
                [*commands, ir.YieldStmt([ir.Sub(carry, ci(1), index, SPAN)], SPAN)], SPAN
            )
            loop = ir.WhileStmt(condition, [carry], loop_body, [result], SPAN)
        body = [read, loop, independent]
    function = ir.Function(
        "oracle_scalar",
        [],
        [],
        ir.SeqStmts([*setup, *body, ir.ReturnStmt([], SPAN)], SPAN),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    values = {(ir.MemorySpace.Vec, address): count for address in range(256)}
    return function, values


@pytest.mark.parametrize("kind", ["write_independent", "unused_read"])
def test_scalar_operation_does_not_fence_an_independent_pipe(kind):
    function, memory = oracle_fixture(kind)
    execution = ClosedCommands(function, {}, range(6), scalar_memory=memory)
    order = execution.run()
    assert (("payload", 0, "C"), ("payload", 1, "I")) not in order


def test_missing_scalar_write_service_is_exposed_by_hazard_check():
    function, _ = oracle_fixture("write_hazard")
    with pytest.raises(AssertionError, match="unordered physical-byte conflict"):
        check_closed_commands(function, function, {}, range(6))


@pytest.mark.parametrize("kind", ["same_pipe", "value"])
def test_scalar_completion_reaches_actual_same_pipe_or_value_consumer(kind):
    function, memory = oracle_fixture(kind)
    execution = ClosedCommands(function, {}, range(6), scalar_memory=memory)
    order = execution.run()
    assert (("payload", 0, "C"), ("payload", 1, "I")) in order


@pytest.mark.parametrize("kind", ["branch", "for", "while"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_scalar_control_reaches_commands_and_empty_path_continuation(kind, count):
    function, memory = oracle_fixture(kind, count)
    execution = ClosedCommands(function, {}, range(6), scalar_memory=memory)
    order = execution.run()
    source = ("payload", 0, "C")
    assert (source, ("payload", 1, "I")) in order
    assert all((source, publication) in order for publication in execution.publications)
    expected = int(count > 0) if kind == "branch" else count
    assert len(execution.publications) == 2 * expected
    if kind == "while":
        assert execution.while_tests == count + 1
