# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit Buffer calls cannot bypass runtime metadata or native encodings."""

import pytest
from pypto import codegen, ir
from pypto.backend import BackendType

from .test_runtime_valid_broadcast import diagnostics

pytestmark = pytest.mark.usefixtures("ascend_backend")
S = ir.Span.unknown()


def metadata_function(operation="buffer.cast", kind="matched", rows=8, columns=32):
    """All valid extents carry the existing 0<=extent<=capacity input contract."""

    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, S)

    def tup(xs):
        return ir.MakeTuple([ci(x) if isinstance(x, int) else x for x in xs], S)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), S)

    def evaluate(name, args):
        return ir.EvalStmt(call(name, args), S)

    r1, r2 = [ir.Var(n, ir.ScalarType(ir.DataType.INDEX), S) for n in ["r1", "r2"]]
    source_dtype = ir.DataType.BF16 if operation == "buffer.cast" else ir.DataType.FP32
    matrix = ir.BufferType([rows, columns], source_dtype, ir.MemorySpace.Vec, valid_shape=[-1, -1])
    source = ir.Var("source", matrix, S)
    reduction = operation in {"buffer.row_sum", "buffer.row_max"}
    target_shape = [rows, 1] if reduction else [rows, columns]
    target_type = ir.BufferType(
        target_shape,
        ir.DataType.FP32,
        ir.MemorySpace.Vec,
        valid_shape=[-1, -1],
        blayout=ir.TileLayout.col_major if reduction else ir.TileLayout.row_major,
    )
    target = ir.Var("target", target_type, S)
    source_cols = columns - 16 if kind == "partial_columns" else columns
    target_cols = 1 if reduction else source_cols
    target_rows = r2 if kind == "mismatch" else r1
    source_bytes = rows * columns * (2 if operation == "buffer.cast" else 4)
    target_bytes = rows * target_shape[1] * 4
    target_address = ((source_bytes + 31) // 32) * 32 + 32
    workspace_address = ((target_address + target_bytes + 31) // 32) * 32 + 32
    definitions: list[ir.Stmt] = [
        ir.AssignStmt(source, call("buffer.alloc", [tup([r1, columns]), ci(0)], matrix), S),
        ir.AssignStmt(
            target,
            call(
                "buffer.alloc",
                [tup([r1, target_shape[1]]), ci(0 if kind == "alias" else target_address)],
                target_type,
            ),
            S,
        ),
    ]
    arguments: list[ir.Expr] = [source, target]
    if operation != "buffer.cast":
        carrier_shape = [rows, columns] if reduction else [rows, 1]
        carrier_type = ir.BufferType(
            carrier_shape,
            ir.DataType.FP32,
            ir.MemorySpace.Vec,
            valid_shape=carrier_shape if reduction else [-1, -1],
            blayout=ir.TileLayout.row_major if reduction else ir.TileLayout.col_major,
        )
        carrier = ir.Var("workspace" if reduction else "carrier", carrier_type, S)
        definitions.append(
            ir.AssignStmt(
                carrier,
                call(
                    "buffer.alloc", [tup([] if reduction else [r1, 1]), ci(workspace_address)], carrier_type
                ),
                S,
            )
        )
        if not reduction:
            definitions.append(evaluate("buffer.set_validshape", [carrier, tup([r1, 1])]))
        arguments = [source, carrier, target]
    definitions.extend(
        [
            evaluate("buffer.set_validshape", [source, tup([r1, source_cols])]),
            evaluate("buffer.set_validshape", [target, tup([target_rows, target_cols])]),
        ]
    )
    native: ir.Stmt = evaluate(operation, arguments)
    if kind != "unguarded":
        native = ir.IfStmt(ir.Gt(r1, ci(0), ir.DataType.BOOL, S), native, None, [], S)
    return ir.Function(
        "metadata",
        [r1, r2],
        [],
        ir.SeqStmts([*definitions, native, ir.ReturnStmt([], S)], S),
        S,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize(
    "operation", ["buffer.cast", "buffer.row_sum", "buffer.row_max", "buffer.row_expand_mul"]
)
@pytest.mark.parametrize(
    "kind,reason",
    [
        ("mismatch", "matching runtime rows"),
        ("unguarded", "positive-row guard"),
        ("partial_columns", "fixed full columns"),
    ],
)
def test_explicit_runtime_calls_require_actual_use_site_metadata(operation, kind, reason):
    function = metadata_function(operation, kind)
    errors = diagnostics(function)
    assert errors and any(reason in error.message for error in errors)
    with pytest.raises(ValueError, match=reason):
        codegen.PTOCodegen().generate(ir.Program([function], "Metadata", S))


@pytest.mark.parametrize(
    "operation", ["buffer.cast", "buffer.row_sum", "buffer.row_max", "buffer.row_expand_mul"]
)
def test_supported_explicit_runtime_metadata_generates_native_operation(operation):
    function = metadata_function(operation)
    assert not diagnostics(function)
    assert "pto." in codegen.PTOCodegen().generate(ir.Program([function], "Metadata", S))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize(
    "operation", ["buffer.cast", "buffer.row_sum", "buffer.row_max", "buffer.row_expand_mul"]
)
def test_new_runtime_families_remain_target_qualified(ascend_backend, operation):
    assert ascend_backend == BackendType.Ascend950
    function = metadata_function(operation)
    with pytest.raises(ValueError, match="A2/A3"):
        codegen.PTOCodegen().generate(ir.Program([function], "Metadata", S))


@pytest.mark.parametrize(
    "operation,rows,columns,reason",
    [
        ("buffer.cast", 8, 65536, "native uint16"),
        ("buffer.cast", 1 << 28, 32, "uint32"),
        ("buffer.row_sum", 65536, 32, "native uint16"),
        ("buffer.row_max", 8, 524288, "native uint16"),
        ("buffer.row_expand_mul", 256, 2048, "native uint8"),
    ],
)
def test_runtime_native_encoding_overflow_remains_unavailable(operation, rows, columns, reason):
    function = metadata_function(operation, rows=rows, columns=columns)
    # Synthetic addressed windows are pairwise disjoint. Require the native
    # field diagnostic, not an unrelated physical-alias/placement refusal.
    with pytest.raises(ValueError, match=reason):
        codegen.PTOCodegen().generate(ir.Program([function], "Metadata", S))


def test_runtime_cast_cannot_use_overlapping_physical_windows():
    function = metadata_function(kind="alias")
    assert diagnostics(function)
    with pytest.raises(ValueError):
        codegen.PTOCodegen().generate(ir.Program([function], "Metadata", S))


@pytest.mark.parametrize("operation", ["buffer.load", "buffer.store"])
@pytest.mark.parametrize("gm_columns", [1, 16])
def test_dynamic_column_transfers_do_not_enter_unqualified_native_variants(operation, gm_columns):
    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, S)

    def tup(xs):
        return ir.MakeTuple([ci(x) if isinstance(x, int) else x for x in xs], S)

    rows = ir.Var("rows", ir.ScalarType(ir.DataType.INDEX), S)
    gm = ir.Var("gm", ir.TensorType([8, gm_columns], ir.DataType.FP32), S)
    ty = ir.BufferType(
        [8, 1], ir.DataType.FP32, ir.MemorySpace.Vec, valid_shape=[-1, -1], blayout=ir.TileLayout.col_major
    )
    column = ir.Var("column", ty, S)
    allocation = ir.AssignStmt(column, ir.Call(ir.get_op("buffer.alloc"), [tup([rows, 1]), ci(0)], ty, S), S)
    args: list[ir.Expr] = (
        [gm, tup([0, 0]), tup([rows, 1]), column]
        if operation == "buffer.load"
        else [column, tup([0, 0]), tup([rows, 1]), gm]
    )
    transfer = ir.EvalStmt(ir.Call(ir.get_op(operation), args, ir.VoidType(), S), S)
    function = ir.Function(
        "transfer",
        [gm, rows],
        [],
        ir.SeqStmts([allocation, transfer, ir.ReturnStmt([], S)], S),
        S,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([function], "Transfer", S)
    original = ir.serialize(program)
    with pytest.raises(ValueError, match="column|Column"):
        codegen.PTOCodegen().generate(program)
    assert ir.serialize(program) == original


@pytest.mark.parametrize("operation", ["buffer.add", "buffer.mul"])
def test_dynamic_column_elementwise_requires_qualified_axis_interpretation(operation):
    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, S)

    rows = ir.Var("rows", ir.ScalarType(ir.DataType.INDEX), S)
    ty = ir.BufferType(
        [8, 1], ir.DataType.FP32, ir.MemorySpace.Vec, valid_shape=[-1, -1], blayout=ir.TileLayout.col_major
    )
    buffers = [ir.Var(n, ty, S) for n in ["a", "b", "out"]]
    statements: list[ir.Stmt] = [
        ir.AssignStmt(
            v, ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([rows, ci(1)], S), ci(index * 64)], ty, S), S
        )
        for index, v in enumerate(buffers)
    ]
    native = ir.EvalStmt(
        ir.Call(ir.get_op(operation), [buffers[0], buffers[1], buffers[2]], ir.VoidType(), S), S
    )
    statements.append(ir.IfStmt(ir.Gt(rows, ci(0), ir.DataType.BOOL, S), native, None, [], S))
    statements.append(ir.ReturnStmt([], S))
    function = ir.Function(
        "columns",
        [rows],
        [],
        ir.SeqStmts(statements, S),
        S,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([function], "Columns", S)
    original = ir.serialize(program)
    with pytest.raises(ValueError, match="runtime column elementwise"):
        codegen.PTOCodegen().generate(program)
    assert ir.serialize(program) == original
