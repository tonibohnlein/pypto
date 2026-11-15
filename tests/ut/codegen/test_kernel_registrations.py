# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Registration identity follows the final payload, including wrapper dispatch."""

import pytest
from _orchestration_codegen_common import _finalize_for_codegen
from pypto import DataType as D
from pypto import backend, codegen, ir
from pypto.backend import BackendType

S = ir.Span.unknown()


def parameter(name):
    return ir.Var(name, ir.TensorType([16], D.FP32), S)


def function(name, kind, param, body):
    return ir.Function(name, [param], [], body, S, type=kind)


def call(name, argument, direction):
    return ir.Call(ir.GlobalVar(name), [argument], {}, {"arg_directions": [direction]}, ir.TupleType([]), S)


@pytest.mark.parametrize("wrapper", [None, ir.FunctionType.Spmd, ir.FunctionType.Group, "mixed"])
@pytest.mark.parametrize("submit", [False, True])
def test_same_code_different_tensor_directions_get_distinct_ids(wrapper, submit):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    x = parameter("kernel_input")
    functions = [function("vector", ir.FunctionType.AIV, x, ir.ReturnStmt([], S))]
    target = "vector"
    if wrapper is not None:
        w = parameter("wrapper_input")
        body: list[ir.Stmt] = [ir.EvalStmt(call("vector", w, ir.ArgDirection.Input), S)]
        if wrapper == "mixed":
            cube_param = parameter("cube_input")
            functions.append(function("cube", ir.FunctionType.AIC, cube_param, ir.ReturnStmt([], S)))
            body.insert(0, ir.EvalStmt(call("cube", w, ir.ArgDirection.Input), S))
        body.append(ir.ReturnStmt([], S))
        functions.append(
            function(
                "wrapper", ir.FunctionType.Group if wrapper == "mixed" else wrapper, w, ir.SeqStmts(body, S)
            )
        )
        target = "wrapper"
    actual = parameter("actual")
    body = ir.SeqStmts(
        [
            ir.EvalStmt(
                ir.Submit(
                    ir.GlobalVar(target),
                    [actual],
                    [],
                    {},
                    {"arg_directions": [ir.ArgDirection.Input]},
                    ir.TupleType([ir.ScalarType(D.TASK_ID)]),
                    S,
                )
                if submit
                else call(target, actual, ir.ArgDirection.Input),
                S,
            ),
            ir.EvalStmt(call(target, actual, ir.ArgDirection.NoDep), S),
            ir.EvalStmt(call(target, actual, ir.ArgDirection.Input), S),
            ir.ReturnStmt([], S),
        ],
        S,
    )
    functions.append(function("main", ir.FunctionType.Orchestration, actual, body))
    source = _finalize_for_codegen(ir.Program(functions, "test", S))
    main = source.get_function("main")
    assert main is not None
    result = codegen.generate_orchestration(source, main)
    registrations = result.kernel_registrations
    expected_names = {"vector", "cube"} if wrapper == "mixed" else {"vector"}
    assert len(registrations) == 2 * len(expected_names)
    assert len({r.func_id for r in registrations}) == len(registrations)
    for name in expected_names:
        assert {tuple(r.signature) for r in registrations if r.code_name == name} == {("IN",), ("INOUT",)}
    assert result.code.count("add_no_dep") == 1
    expected_domains = {"main": ir.FunctionType.Orchestration}
    if wrapper is not None:
        expected_domains["wrapper"] = ir.FunctionType.Group if wrapper == "mixed" else wrapper
    assert result.task_domains == expected_domains
    # The public property is a detached read-only publication, not a mutable
    # registration side channel. Local kernels never acquire this owner.
    result.task_domains.clear()
    assert result.task_domains == expected_domains


def test_referenced_graph_and_unused_wrapper_have_distinct_owners():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    x, g, w, actual = [parameter(name) for name in ["x", "g", "w", "actual"]]
    source = ir.Program(
        [
            function("vector", ir.FunctionType.AIV, x, ir.ReturnStmt([], S)),
            function(
                "graph",
                ir.FunctionType.Graph,
                g,
                ir.SeqStmts(
                    [ir.EvalStmt(call("vector", g, ir.ArgDirection.Input), S), ir.ReturnStmt([], S)], S
                ),
            ),
            function(
                "unused_wrapper",
                ir.FunctionType.Spmd,
                w,
                ir.SeqStmts(
                    [ir.EvalStmt(call("vector", w, ir.ArgDirection.Input), S), ir.ReturnStmt([], S)], S
                ),
            ),
            function(
                "main",
                ir.FunctionType.Orchestration,
                actual,
                ir.SeqStmts(
                    [ir.EvalStmt(call("graph", actual, ir.ArgDirection.Input), S), ir.ReturnStmt([], S)], S
                ),
            ),
        ],
        "GraphOwners",
        S,
    )
    source = _finalize_for_codegen(source)
    main = source.get_function("main")
    assert main is not None
    result = codegen.generate_orchestration(source, main)
    assert result.task_domains == {"main": ir.FunctionType.Orchestration, "graph": ir.FunctionType.Graph}
    assert [entry.code_name for entry in result.kernel_registrations] == ["vector"]
    assert "rt_submit_graph" in result.code and "rt_submit_aiv_task" in result.code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
