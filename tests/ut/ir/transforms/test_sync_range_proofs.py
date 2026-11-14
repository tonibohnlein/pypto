# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original-width coordinate proofs and explicitly bounded dynamic allocations."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical

pytestmark = pytest.mark.usefixtures("ascend_backend")


def range_fixture(expression=None):
    """A generic DMA round trip; shape identity is an explicit ABI contract."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(v) if isinstance(v, int) else v for v in values], span)

    rows = ir.Var("rows", ir.ScalarType(index), span)
    offset = ir.Var("offset", ir.ScalarType(ir.DataType.INT32), span)
    tensor_type = ir.TensorType([rows, ci(64)], fp)
    source, output = (ir.Var(name, tensor_type, span) for name in ["source", "output"])
    tile_type = ir.BufferType([16, 32], fp, ir.MemorySpace.Vec)
    tile = ir.Var("tile", tile_type, span)
    coordinate = ir.Cast(offset if expression is None else expression(offset, span), index, span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(tile, call("buffer.alloc", [tup([]), ci(0)], tile_type), span),
            ir.EvalStmt(call("buffer.load", [source, tup([coordinate, 0]), tup([16, 32]), tile]), span),
            ir.EvalStmt(call("buffer.store", [tile, tup([coordinate, 0]), tup([16, 32]), output]), span),
            ir.ReturnStmt([output], span),
        ],
        span,
    )
    # Intentionally interleave scalar/tensor formals; indices are original ABI positions.
    return ir.Function(
        "bounded_dma",
        [source, offset, (output, ir.ParamDirection.InOut), rows],
        [tensor_type],
        body,
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("rows", [32, 64, 128])
@pytest.mark.parametrize("offset", [0, 16])
def test_dynamic_extent_and_original_width_coordinates(rows, offset):
    function = range_fixture()
    result = testing.analyze_placed_sync(function, True, parameter_bounds={1: (0, 16), 3: (32, 128)})
    parameters = {1: offset, 3: rows}
    validate(function, result, parameters=parameters)
    physical = result.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6), parameters=parameters)
    assert physical["set_sites"] == physical["wait_sites"] == 1
    assert result.describe()["storage"][0]["row_extent_parameter"] == 3
    assert result.boundary(0, rows, 0, parameters) is None
    assert result.boundary(0, 0, 0, {1: offset}) is None
    assert result.boundary(0, 0, 0, {1: offset, 3: 129}) is None
    absent = result.boundary(0, 0, 63, parameters)
    assert absent is not None and absent["first_writer"] is None and not any(absent["first_readers"])
    # Local export does not depend on an unrelated GM extent/translation.
    assert result.boundary(2, 0, 0) is not None


def test_dynamic_extent_uses_smallest_admitted_allocation():
    with pytest.raises(ValueError, match="exceeds its proved allocation"):
        testing.analyze_placed_sync(range_fixture(), True, parameter_bounds={1: (0, 16), 3: (31, 128)})
    with pytest.raises(ValueError, match="no established caller range"):
        testing.analyze_placed_sync(range_fixture(), True, parameter_bounds={1: (0, 16)})


@pytest.mark.parametrize("overflow_then_zero", [False, True])
def test_original_integer_overflow_is_not_hidden_by_index_cast(overflow_then_zero):
    def expression(offset, span):
        dtype = ir.DataType.INT32
        overflow = ir.Add(offset, ir.ConstInt(2147483647, dtype, span), dtype, span)
        return ir.Mul(overflow, ir.ConstInt(0, dtype, span), dtype, span) if overflow_then_zero else overflow

    with pytest.raises(ValueError, match="overflow its source type"):
        testing.analyze_placed_sync(
            range_fixture(expression), True, parameter_bounds={1: (0, 16), 3: (32, 128)}
        )


def test_narrowing_cast_is_only_accepted_when_value_preserving():
    function = range_fixture(lambda offset, span: ir.Cast(offset, ir.DataType.INT8, span))
    result = testing.analyze_placed_sync(function, True, parameter_bounds={1: (0, 16), 3: (32, 128)})
    validate(function, result, parameters={1: 16, 3: 32})
    with pytest.raises(ValueError, match="cast may change its value"):
        testing.analyze_placed_sync(function, True, parameter_bounds={1: (0, 128), 3: (256, 256)})


def caller_program(function, unknown=False, overflow=False):
    """Actual call sites establish scalar ranges, independently of diagnostic bounds."""
    span = function.span
    rows_unknown = ir.Var("unknown", ir.ScalarType(ir.DataType.INDEX), span)
    params, statements = [rows_unknown], []
    for rows, offset in [(32, 0), (128, 16)]:
        tensor_type = ir.TensorType([rows, 64], ir.DataType.FP32)
        source, output = (ir.Var(f"{name}{rows}", tensor_type, span) for name in ["source", "output"])
        params.extend([source, output])
        offset_arg = ir.ConstInt(offset, ir.DataType.INT32, span)
        if overflow:
            offset_arg = ir.Mul(
                ir.Add(offset_arg, ir.ConstInt(2147483647, ir.DataType.INT32, span), ir.DataType.INT32, span),
                ir.ConstInt(0, ir.DataType.INT32, span),
                ir.DataType.INT32,
                span,
            )
        rows_arg = rows_unknown if unknown else ir.ConstInt(rows, ir.DataType.INDEX, span)
        statements.append(
            ir.EvalStmt(
                ir.Call(
                    ir.GlobalVar(function.name), [source, offset_arg, output, rows_arg], tensor_type, span
                ),
                span,
            )
        )
    statements.append(ir.ReturnStmt([], span))
    caller = ir.Function(
        "main", params, [], ir.SeqStmts(statements, span), span, ir.FunctionType.Orchestration
    )
    return ir.Program([function, caller], "RangeCalls", span)


def test_original_callers_supply_production_bounds():
    function = range_fixture()
    program = caller_program(function)
    assert testing.sync_parameter_bounds(program)[function.name] == {1: (0, 16), 3: (32, 128)}
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        output = passes.realize_sync()(program)
    actual = next(f for f in output.functions.values() if f.name == function.name)
    assert dict(actual.attrs)["pypto_autosync"]
    for rows, offset in [(32, 0), (128, 16)]:
        run_physical(function, actual, range(6), parameters={1: offset, 3: rows})


@pytest.mark.parametrize("unknown,overflow,missing", [(True, False, 3), (False, True, 1)])
def test_unknown_and_wrapping_callers_remain_unproved(unknown, overflow, missing):
    function = range_fixture()
    program = caller_program(function, unknown, overflow)
    assert missing not in testing.sync_parameter_bounds(program)[function.name]
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        output = passes.realize_sync()(program)
    actual = next(f for f in output.functions.values() if f.name == function.name)
    assert not dict(actual.attrs).get("pypto_autosync", False)
    ir.assert_structural_equal(actual.body, function.body)


@pytest.mark.ptoas_compile
def test_dynamic_extent_physical_source_generation(tmp_path):
    function = range_fixture()
    retained = testing.analyze_placed_sync(function, True, parameter_bounds={1: (0, 16), 3: (32, 128)})
    physical = retained.realize_physical("a2a3", list(range(6)))
    program = ir.Program([physical["function"]], "BoundedDMA", function.span)
    source, output = tmp_path / "bounded.pto", tmp_path / "bounded.cpp"
    source.write_text(codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False))
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    assert "set_flag(" in output.read_text()
    assert "wait_flag(" in output.read_text()


@pytest.mark.parametrize("dtype,stop", [(ir.DataType.INT8, 124), (ir.DataType.UINT8, 252)])
@pytest.mark.parametrize("overflow", [False, True])
def test_caller_loop_increment_is_checked_in_original_width(dtype, stop, overflow):
    span, index = ir.Span.unknown(), ir.DataType.INDEX
    parameter = ir.Var("coordinate", ir.ScalarType(index), span)
    child = ir.Function("child", [parameter], [], ir.ReturnStmt([], span), span, ir.FunctionType.AIC)
    iv = ir.Var("i", ir.ScalarType(dtype), span)
    call = ir.Call(ir.GlobalVar("child"), [ir.Cast(iv, index, span)], ir.VoidType(), span)
    bound = stop + 3 if overflow else stop
    loop = ir.ForStmt(
        iv,
        ir.ConstInt(0, dtype, span),
        ir.ConstInt(bound, dtype, span),
        ir.ConstInt(2, dtype, span),
        [],
        ir.SeqStmts([ir.EvalStmt(call, span), ir.YieldStmt([], span)], span),
        [],
        span,
    )
    caller = ir.Function("main", [], [], loop, span, ir.FunctionType.Orchestration)
    program = ir.Program([child, caller], "NarrowLoop", span)
    assert testing.sync_parameter_bounds(program).get("child", {}) == ({} if overflow else {0: (0, stop - 2)})


def invocation_chain(
    depth=3, unknown=False, cycle=False, launches=(20, 24), submit=False, worker_kind=ir.FunctionType.AIC
):
    """Multiple real launch sites forward a parameter and logical block identity."""
    span, index = ir.Span.unknown(), ir.DataType.INDEX
    ty = ir.ScalarType(index)

    def ci(value):
        return ir.ConstInt(value, index, span)

    functions = []
    for number in range(depth):
        value = ir.Var(f"value{number}", ty, span)
        if number + 1 < depth:
            call = ir.Call(ir.GlobalVar(f"f{number + 1}"), [value], ir.VoidType(), span)
            body = ir.EvalStmt(call, span)
        else:
            body = ir.ReturnStmt([], span)
        if cycle and number == depth - 1:
            body = ir.EvalStmt(ir.Call(ir.GlobalVar("f0"), [value], ir.VoidType(), span), span)
        functions.append(
            ir.Function(
                f"f{number}",
                [value],
                [],
                body,
                span,
                worker_kind if number == depth - 1 else ir.FunctionType.Spmd,
            )
        )
    external = ir.Var("unknown", ty, span)
    calls = []
    for i, count in enumerate(launches):
        argument = external if unknown and i == len(launches) - 1 else ci(i * 5120)
        attrs = {} if count is None else {"core_num": ci(count)}
        if submit:
            call = ir.Submit(
                ir.GlobalVar("f0"),
                [argument],
                [],
                {},
                {},
                ir.TupleType([ir.ScalarType(ir.DataType.TASK_ID)]),
                span,
                core_num=None if count is None else ci(count),
            )
        else:
            call = ir.Call(ir.GlobalVar("f0"), [argument], {}, attrs, ir.VoidType(), span)
        calls.append(ir.EvalStmt(call, span))
    functions.append(
        ir.Function("main", [external], [], ir.SeqStmts(calls, span), span, ir.FunctionType.Orchestration)
    )
    return ir.Program(functions, "ForwardedEvidence", span)


@pytest.mark.parametrize("submit", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("worker_kind", [ir.FunctionType.AIC, ir.FunctionType.AIV])
def test_forwarded_ranges_join_all_original_callers(submit, unknown, worker_kind):
    evidence = testing.sync_invocation_evidence(
        invocation_chain(2, unknown=unknown, submit=submit, worker_kind=worker_kind)
    )
    for name in ["f0", "f1"]:
        assert evidence["parameters"].get(name, {}) == ({} if unknown else {0: (0, 5120)})
        assert evidence["block_indices"][name] == (0, 23)
    assert evidence["call_graph_nodes"] == 3
    assert evidence["call_graph_edges"] == 2
    assert evidence["processed_functions"] == 3


@pytest.mark.parametrize("launches", [(20, None), (20, 0), (20, -1)])
@pytest.mark.parametrize("worker_kind", [ir.FunctionType.AIC, ir.FunctionType.AIV])
def test_unknown_launch_does_not_inherit_another_callers_width(launches, worker_kind):
    evidence = testing.sync_invocation_evidence(
        invocation_chain(2, launches=launches, worker_kind=worker_kind)
    )
    assert evidence["block_indices"] == {}
    assert evidence["parameters"]["f1"] == {0: (0, 5120)}


def test_recursive_invocation_evidence_remains_unproved():
    evidence = testing.sync_invocation_evidence(invocation_chain(3, cycle=True))
    assert evidence["parameters"] == {}
    assert evidence["block_indices"] == {}
    assert evidence["processed_functions"] == 1


def test_wrapper_proof_work_tracks_static_call_graph():
    counts = []
    for depth in [8, 16, 32]:
        evidence = testing.sync_invocation_evidence(invocation_chain(depth))
        assert evidence["parameters"][f"f{depth - 1}"] == {0: (0, 5120)}
        assert evidence["call_graph_nodes"] == depth + 1
        assert evidence["call_graph_edges"] == depth
        counts.append(evidence["expression_visits"])
    assert counts[1] <= 2 * counts[0] and counts[2] <= 2 * counts[1]


@pytest.mark.parametrize("override,expected", [(None, 19), (8, 7), (0, None)])
@pytest.mark.parametrize(
    "kind,dual", [(ir.FunctionType.AIC, False), (ir.FunctionType.AIV, False), (ir.FunctionType.AIV, True)]
)
def test_launch_call_override_and_function_default(override, expected, kind, dual):
    span, dtype = ir.Span.unknown(), ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, dtype, span)

    worker = ir.Function(
        "worker",
        [],
        [],
        ir.ReturnStmt([], span),
        span,
        kind,
        attrs={"dual_aiv_dispatch": True} if dual else {},
    )
    wrapper = ir.Function(
        "wrapper",
        [],
        [],
        ir.EvalStmt(ir.Call(ir.GlobalVar("worker"), [], ir.VoidType(), span), span),
        span,
        ir.FunctionType.Spmd,
        attrs={"core_num": ci(20)},
    )
    attrs = {} if override is None else {"core_num": ci(override)}
    call = ir.Call(ir.GlobalVar("wrapper"), [], {}, attrs, ir.VoidType(), span)
    root = ir.Function("main", [], [], ir.EvalStmt(call, span), span, ir.FunctionType.Orchestration)
    evidence = testing.sync_invocation_evidence(ir.Program([root, wrapper, worker], "launch", span))
    assert evidence["block_indices"] == (
        {} if expected is None else {"wrapper": (0, expected), "worker": (0, expected)}
    )
    # A second direct orchestration->worker call has no established block semantics.
    root = ir.Function(
        "main",
        [],
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(call, span),
                ir.EvalStmt(ir.Call(ir.GlobalVar("worker"), [], ir.VoidType(), span), span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )
    evidence = testing.sync_invocation_evidence(ir.Program([root, wrapper, worker], "direct", span))
    assert "worker" not in evidence["block_indices"]


@pytest.mark.parametrize(
    "divisor,offset,expected",
    [(24, 6, (0, 23)), (48, 6, (6, 29)), (0, 6, None), (-1, 6, None), (24, 2**63 - 24, None)],
)
def test_launch_coordinate_modulo_preserves_original_width(divisor, offset, expected):
    span, dtype = ir.Span.unknown(), ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, dtype, span)

    ty = ir.ScalarType(dtype)
    coordinate = ir.Var("coordinate", ty, span)
    worker = ir.Function("worker", [coordinate], [], ir.ReturnStmt([], span), span, ir.FunctionType.AIC)
    block = ir.Call(ir.get_op("buffer.get_block_idx"), [], ty, span)
    expression = ir.FloorMod(ir.Add(block, ci(offset), dtype, span), ci(divisor), dtype, span)
    wrapper = ir.Function(
        "wrapper",
        [],
        [],
        ir.EvalStmt(ir.Call(ir.GlobalVar("worker"), [expression], ir.VoidType(), span), span),
        span,
        ir.FunctionType.Spmd,
    )
    call = ir.Call(ir.GlobalVar("wrapper"), [], {}, {"core_num": ci(24)}, ir.VoidType(), span)
    root = ir.Function("main", [], [], ir.EvalStmt(call, span), span, ir.FunctionType.Orchestration)
    evidence = testing.sync_invocation_evidence(ir.Program([root, wrapper, worker], "modulo", span))
    assert evidence["parameters"].get("worker", {}) == ({} if expected is None else {0: expected})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
