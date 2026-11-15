# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Packed GM columns use the existing DN-to-DN transfer and physical byte model."""

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import ir as core_ir

from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical

pytestmark = pytest.mark.usefixtures("ascend_backend")


def column_program(rows, visits, reduction=False, replay=1, start=0, stream=True):
    total = rows * max(1, visits) + start
    columns = 8 if reduction else 1
    offset = f"visit * {rows} + {start}" if stream else str(start)
    body = (
        f"scratch = pl.create_tile([{rows}, 8], dtype=pl.FP32)\n"
        "            result = pl.row_sum(source, scratch)"
        if reduction
        else f"row = pl.reshape(source, [1, {rows}])\n"
        "            changed = pl.abs(row)\n"
        f"            result = pl.reshape(changed, [{rows}, 1])"
    )
    program = pl.parse_program(f"""
@pl.program
class PackedColumns:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[{total}, {columns}], pl.FP32],
               out: pl.InOut[pl.Tensor[[{total}, 1], pl.FP32]]) -> pl.Tensor[[{total}, 1], pl.FP32]:
        for visit in pl.range({visits}):
            source = pl.load(a, [{offset}, 0], [{rows}, {columns}])
            {body}
            out = pl.store(result, [{offset}, 0], out)
        return out
""")
    if replay > 1:

        class Replay(ir.IRMutator):
            def visit_for_stmt(self, op):
                span = op.span
                iv = ir.Var("replay", ir.ScalarType(ir.DataType.INDEX), span)

                def ci(value):
                    return ir.ConstInt(value, ir.DataType.INDEX, span)

                return ir.ForStmt(
                    iv,
                    ci(0),
                    ci(replay),
                    ci(1),
                    [],
                    ir.SeqStmts([op, ir.YieldStmt([], span)], span),
                    [],
                    span,
                )

        program = Replay().visit_program(program)
    return program


@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize("visits,replay,stream", [(1, 1, True), (3, 1, True), (1, 2, True), (3, 2, False)])
@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("start", [0, 1])
def test_packed_column_original_accesses_covers_and_reuse(
    rows, visits, reduction, replay, stream, start, tmp_path
):
    # Cross-reset positives reuse the original constant GM slice. Streaming GM
    # writes across a reset require a distinct regional membership export below.
    program = column_program(rows, visits, reduction, replay, start, stream)
    functions = []
    for enabled in [False, True]:
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=enabled,
            autosync_disjoint_gm=True,
        ):
            lowered = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
        function = lowered.get_function("kernel")
        assert function is not None
        assert function.ir_stage == ir.FunctionIRStage.Buffer
        functions.append(function)
    original, emitted = functions
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    retained = testing.analyze_placed_sync(original, True)
    # This interprets addresses from original Buffer IR, not compiler cells.
    validate(original, retained)
    run_physical(original, emitted, range(6))
    restored = ir.deserialize(ir.serialize(emitted))
    ir.assert_structural_equal(emitted, restored)
    text = codegen.PTOCodegen().generate(ir.Program([emitted], "PackedColumns", emitted.span))
    if find_ptoas_binary() is not None:
        source, target = tmp_path / "column.pto", tmp_path / "column.cpp"
        source.write_text(text)
        _run_ptoas(str(source), str(target), ["--pto-level=level3", "--pto-arch=a2"])
        native = target.read_text()
        assert "TSTORE" in native and "Layout::DN" in native


@pytest.mark.parametrize("columns,valid", [(8, [16, 1]), (1, [8, 1])])
def test_column_transfer_rejects_wide_gm_or_partial_local(columns, valid):
    span = ir.Span.unknown()
    source = ir.Var("a", ir.TensorType([32, columns], ir.DataType.FP32), span)
    local = ir.Var(
        "v",
        ir.BufferType(
            [16, 1], ir.DataType.FP32, ir.MemorySpace.Vec, valid_shape=valid, blayout=ir.TileLayout.col_major
        ),
        span,
    )

    def tup(values):
        return ir.MakeTuple([ir.ConstInt(v, ir.DataType.INDEX, span) for v in values], span)

    with pytest.raises(ValueError, match="packed singleton-column"):
        core_ir._create_internal_op_call("buffer.load", [source, tup([0, 0]), tup(valid), local], {}, span)


@pytest.mark.parametrize("kind", ["misaligned", "symbolic_rows", "large_rows", "strided", "burst_overflow"])
def test_column_transfer_requires_native_shape_and_packed_stride(kind):
    span = ir.Span.unknown()

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    rows = (
        ir.Var("rows", ir.ScalarType(ir.DataType.INDEX), span)
        if kind == "symbolic_rows"
        else ci((1 << 30) if kind == "large_rows" else 524288 if kind == "burst_overflow" else 32)
    )
    view = ir.TensorView([ci(2), ci(1)], ir.TensorLayout.ND) if kind == "strided" else None
    source = ir.Var("a", ir.TensorType([rows, ci(1)], ir.DataType.FP32, tensor_view=view), span)
    size = 7 if kind == "misaligned" else 524288 if kind == "burst_overflow" else 16
    local = ir.Var(
        "v",
        ir.BufferType([size, 1], ir.DataType.FP32, ir.MemorySpace.Vec, blayout=ir.TileLayout.col_major),
        span,
    )
    with pytest.raises(ValueError, match="packed singleton-column"):
        core_ir._create_internal_op_call(
            "buffer.load",
            [source, ir.MakeTuple([ci(0), ci(0)], span), ir.MakeTuple([ci(size), ci(1)], span), local],
            {},
            span,
        )


def test_explicit_packed_column_stride_preserves_transfer_contract():
    span = ir.Span.unknown()

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    view = ir.TensorView([ci(1), ci(1)], ir.TensorLayout.ND)
    source = ir.Var("a", ir.TensorType([32, 1], ir.DataType.FP32, tensor_view=view), span)
    local = ir.Var(
        "v",
        ir.BufferType([16, 1], ir.DataType.FP32, ir.MemorySpace.Vec, blayout=ir.TileLayout.col_major),
        span,
    )
    call = core_ir._create_internal_op_call(
        "buffer.load",
        [source, ir.MakeTuple([ci(8), ci(0)], span), ir.MakeTuple([ci(16), ci(1)], span), local],
        {},
        span,
    )
    assert isinstance(call.type, ir.VoidType)


def test_empty_column_loop_does_not_publish_notifications():
    program = column_program(8, 0)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
    ):
        result = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
    function = result.get_function("kernel")
    assert function is not None
    # No iteration means no load/store completion to publish. Empty-plan
    # ownership remains governed by the existing downstream agreement.
    assert not dict(function.attrs).get(PYPTO_AUTOSYNC_ATTR, False)

    class Commands(ir.IRVisitor):
        count = 0

        def visit_call(self, op):
            self.count += op.op.name in {
                ir.get_op("system.set_flag").name,
                ir.get_op("system.wait_flag").name,
            }
            super().visit_call(op)

    commands = Commands()
    commands.visit_function(function)
    assert commands.count == 0


def test_moving_column_writes_across_resets_retain_export_refusal():
    program = column_program(8, 3, replay=2, stream=True)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=False,
        autosync_disjoint_gm=True,
    ):
        placed = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
    original = placed.get_function("kernel")
    assert original is not None and original.ir_stage == ir.FunctionIRStage.Buffer
    before = ir.serialize(placed)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
    ):
        result = passes.realize_sync()(placed)
    candidate = result.get_function("kernel")
    assert candidate is not None
    assert not dict(candidate.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert "Visit-owned storage cannot be omitted" in dict(candidate.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    # Compare one immutable placed input with its refused transaction. Separate
    # frontend lowerings allocate fresh symbolic roots and are not an identity
    # test for rollback, even when their pretty-printed payloads match.
    assert ir.serialize(placed) == before
    ir.assert_structural_equal(original.body, candidate.body)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_explicit_column_transfer_does_not_gain_unqualified_a5_codegen(ascend_backend):
    assert ascend_backend == BackendType.Ascend950
    span = ir.Span.unknown()

    def ci(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    def tup(values):
        return ir.MakeTuple([ci(value) for value in values], span)

    source = ir.Var("a", ir.TensorType([32, 1], ir.DataType.FP32), span)
    output = ir.Var("out", source.type, span)
    local = ir.Var(
        "v",
        ir.BufferType([16, 1], ir.DataType.FP32, ir.MemorySpace.Vec, blayout=ir.TileLayout.col_major),
        span,
    )
    allocation = ir.AssignStmt(
        local, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], local.type, span), span
    )
    load = core_ir._create_internal_op_call(
        "buffer.load", [source, tup([0, 0]), tup([16, 1]), local], {}, span
    )
    store = core_ir._create_internal_op_call(
        "buffer.store", [local, tup([0, 0]), tup([16, 1]), output], {}, span
    )
    function = ir.Function(
        "column",
        [source, (output, ir.ParamDirection.InOut)],
        [output.type],
        ir.SeqStmts(
            [allocation, ir.EvalStmt(load, span), ir.EvalStmt(store, span), ir.ReturnStmt([output], span)],
            span,
        ),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="qualified A2/A3"):
        codegen.PTOCodegen().generate(ir.Program([function], "Column", span))
