# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pinned example capture helpers for the read-only synchronization prototype."""

import importlib.util
from pathlib import Path

import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.jit.decorator import JITFunction
from pypto.runtime import RunConfig

ROOT = Path(__file__).resolve().parents[4]
CASES = {
    "matmul_64": ("examples/beginner/05_matmul.py", (64, 64, 64)),
    "matmul_acc_64": ("examples/intermediate/04_matmul_acc.py", (64, 64, 64)),
    "ddr_split_k": ("examples/advanced/02_auto_tile_matmul.py", (256, 128, 256)),
}


def load_case(name: str) -> JITFunction:
    """Load an original example without executing its main block."""
    path, _ = CASES[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def capture(name: str, shape: tuple[int, int, int] | None = None) -> ir.Function:
    """Run the real frontend and placement; never run PTOAS or a device."""
    _, default_shape = CASES[name]
    m, k, n = shape or default_shape
    kernel = load_case(name)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        program = kernel.lower(
            torch.empty(m, k),
            torch.empty(k, n),
            torch.empty(m, n),
            config=RunConfig(arch=BackendType.Ascend910),
        )
    return next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)


def staged_fixture(
    count: int, stages: int = 2, visits: int = 1, start: int = 0, step: int = 1
) -> ir.Function:
    """Explicit placed derivative: fixed local slots, streamed GM operand chunks.

    This is not a different AutoTile choice. GM K grows with count, while
    local placement and per-body operations stay fixed. A zero-trip visit
    leaves the caller-initialized output untouched (no uninitialized store).
    """
    span = ir.Span.unknown()
    fp, index = ir.DataType.FP32, ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(x) if isinstance(x, int) else x for x in values], span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    k = 32 * stages * max(1, start + count * step)
    a = ir.Var("a", ir.TensorType([16, k], fp), span)
    b = ir.Var("b", ir.TensorType([k, 16], fp), span)
    out = ir.Var("out", ir.TensorType([16, 16], fp), span)
    allocations, left, right, mat_a, mat_b = [], [], [], [], []

    def alloc(name, shape, space, address):
        ty = ir.BufferType(
            shape,
            fp,
            space,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024 if space == ir.MemorySpace.Acc else 512,
        )
        var = ir.Var(name, ty, span)
        allocations.append(ir.AssignStmt(var, call("buffer.alloc", [tup([]), ci(address)], ty), span))
        return var

    for stage in range(stages):
        mat_a.append(alloc(f"a{stage}", [16, 32], ir.MemorySpace.Mat, stage * 4096))
        mat_b.append(alloc(f"b{stage}", [32, 16], ir.MemorySpace.Mat, stage * 4096 + 2048))
        left.append(alloc(f"l{stage}", [16, 32], ir.MemorySpace.Left, stage * 2048))
        right.append(alloc(f"r{stage}", [32, 16], ir.MemorySpace.Right, stage * 2048))
    acc = alloc("acc", [16, 16], ir.MemorySpace.Acc, 0)
    body = list(allocations)
    for invocation in range(visits):
        iv = ir.Var(f"k{invocation}", ir.ScalarType(index), span)
        loop_body = []
        for stage in range(stages):
            offset = ir.Add(ir.Mul(iv, ci(32 * stages), index, span), ci(stage * 32), index, span)
            loop_body += [
                payload("buffer.load", [a, tup([0, offset]), tup([16, 32]), mat_a[stage]]),
                payload("buffer.load", [b, tup([offset, 0]), tup([32, 16]), mat_b[stage]]),
                payload("buffer.copy", [mat_a[stage], left[stage]]),
                payload("buffer.copy", [mat_b[stage], right[stage]]),
            ]
            args = [left[stage], right[stage], acc]
            if stage == 0:
                loop_body.append(
                    ir.IfStmt(
                        ir.Eq(iv, ci(start), ir.DataType.BOOL, span),
                        payload("buffer.matmul", args),
                        payload("buffer.matmul_acc", args),
                        [],
                        span,
                    )
                )
            else:
                loop_body.append(payload("buffer.matmul_acc", args))
        loop_body.append(ir.YieldStmt([], span))
        body.append(
            ir.ForStmt(
                iv, ci(start), ci(start + count * step), ci(step), [], ir.SeqStmts(loop_body, span), [], span
            )
        )
        if count:
            body.append(payload("buffer.store", [acc, tup([0, 0]), tup([16, 16]), out]))
    body.append(ir.ReturnStmt([out], span))
    return ir.Function(
        "staged_matmul_derivative",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(body, span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
