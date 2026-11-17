# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Blocking local scalar values remain payloads through prefix/token lowering."""

import pytest
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.pypto_core import ir as core_ir

from .sync_closed_command_oracle import ClosedCommands

pytestmark = pytest.mark.usefixtures("ascend_backend")


def scalar_payload_fixture():
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(v):
        return ir.ConstInt(v, index, span)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], span)

    def var(name, ty):
        return ir.Var(name, ty, span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    def op(name, args):
        return ir.EvalStmt(call(name, args), span)

    a, out = [var(name, ir.TensorType([2, 8], fp)) for name in ("a", "out")]
    count, visits = [var(name, ir.ScalarType(index)) for name in ("count", "visits")]
    local = ir.BufferType([2, 8], fp, ir.MemorySpace.Vec)
    x = var("x", local)
    scalar = var("loaded_control", ir.ScalarType(fp))
    body = ir.SeqStmts(
        [
            ir.AssignStmt(scalar, call("buffer.read", [x, tup([0, 0])], scalar.type), span),
            ir.IfStmt(
                ir.Gt(scalar, ir.ConstFloat(0.0, fp, span), ir.DataType.BOOL, span),
                op("buffer.write", [x, tup([0, 0]), ir.ConstFloat(-1.0, fp, span)]),
                op("buffer.write", [x, tup([0, 0]), ir.ConstFloat(1.0, fp, span)]),
                [],
                span,
            ),
            op("buffer.store", [x, tup([0, 0]), tup([2, 8]), out]),
        ],
        span,
    )
    inner = ir.ForStmt(var("i", ir.ScalarType(index)), ci(0), count, ci(1), [], body, [], span)
    outer = ir.ForStmt(var("visit", ir.ScalarType(index)), ci(0), visits, ci(1), [], inner, [], span)
    return ir.Function(
        "scalar_payload",
        [a, (out, ir.ParamDirection.InOut), count, visits],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, call("buffer.alloc", [tup([]), ci(0)], local), span),
                op("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                outer,
                op("buffer.store", [x, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def scalar_pipeline(optimize):
    source = scalar_payload_fixture()
    program = ir.Program([source], "ScalarPayload", source.span)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        logical = passes.insert_sync()(program)
        if optimize:
            logical = passes.optimize_sync()(logical)
    # Reparse the actual tokens: no insertion-time readiness sidecar may be
    # required to implement the original scalar definition and branch.
    logical = ir.deserialize(ir.serialize(logical))
    assert isinstance(logical, ir.Program)
    with passes.PassContext([]):
        physical = passes.lower_sync()(logical)
    return logical, physical


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("count,visits", [(0, 0), (0, 2), (1, 1), (3, 1), (1, 3), (3, 2)])
@pytest.mark.parametrize("initial", [-1.0, 1.0])
def test_scalar_control_matching_coverage_and_counter_resets(optimize, count, visits, initial):
    logical, physical = scalar_pipeline(optimize)
    first = next(iter(logical.functions.values()))
    second = next(iter(physical.functions.values()))
    inputs = {("gm", i): initial for i in range(64)}
    before = ClosedCommands(first, {2: count, 3: visits}, range(6), scalar_memory=inputs)
    after = ClosedCommands(second, {2: count, 3: visits}, range(6), scalar_memory=inputs)
    expected, actual = before.run(), after.run()
    assert before.payloads == after.payloads
    assert before.scalar_memory == after.scalar_memory
    common = {node for node in before.vertices if node[0] == "payload"}
    assert {(a, b) for a, b in expected if a in common and b in common} == {
        (a, b) for a, b in actual if a in common and b in common
    }
    for i, (_, _, reads, writes) in enumerate(after.payloads):
        for j in range(i):
            _, _, r, w = after.payloads[j]
            if w & (reads | writes) or r & writes:
                assert (("payload", j, "C"), ("payload", i, "I")) in actual
    reads = [entry[0] for entry in after.payloads if entry[0][0] == ir.get_op("buffer.read").name]
    assert len(reads) == count * visits
    assert not any(a == b for a, b in actual)
    assert all(pair in actual for pair in after.reuses)
    assert all(ids <= set(range(6)) for ids in after.ids.values())


@pytest.mark.ptoas_compile
def test_scalar_payload_cpp_and_existing_native_scalar_rule(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    _, physical = scalar_pipeline(True)
    source, output = tmp_path / "scalar.pto", tmp_path / "scalar.cpp"
    text = codegen.PTOCodegen().generate(physical, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.tgetval" in text and "pto.tsetval" in text
    source.write_text(text)
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    cpp = output.read_text()
    assert ".GetValue(" in cpp and ".SetValue(" in cpp
    assert "pipe_barrier(PIPE_S)" not in cpp


def test_scalar_payload_rejects_unknown_metadata_or_nonlocal_storage():
    span = ir.Span.unknown()
    coords = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, span)] * 2, span)
    for space, valid in [(ir.MemorySpace.Mat, [2, 8]), (ir.MemorySpace.Vec, [-1, 8])]:
        descriptor = ir.BufferType([2, 8], ir.DataType.FP32, space, valid_shape=valid)
        source = ir.Var("source", descriptor, span)
        with pytest.raises(ValueError, match="Buffer scalar"):
            core_ir._create_internal_op_call("buffer.read", [source, coords], {}, span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
