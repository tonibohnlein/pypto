# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression suite for hand-tuned TileLang and PTOAS-prototype GEMM kernels on PyPTO AutoSync.

Covers the TileLang (`tile-ai/tilelang-ascend`) and PTOAS `FrontierSynch` prototype GEMM corpus:
1. `tilelang_gemm_lookahead` & `jit_tilelang_gemm_{2tile,grid}`:
   `examples/gemm/example_gemm_intrinsic.py` @ 1986dd7 (`prepared/tilelang/input.pto`, 256x256x4096,
   block 128x256x64, K_L1=256, 2-stage L1 lookahead + 2-stage L0A/L0B, FP16 out).
   PTOAS baselines: Hand-Expert s/w=13/13 dyn, bar=0+2, if=4; Upstream InsertSync s/w=14/14, bar=1+1,
   if=2 (stalls lookahead & serializes L0, PTOAS #1519); FrontierSynch s/w=15/15 dyn, bar=0+2, if=25.
2. `tilelang_persistent_lookahead`, `jit_tilelang_persistent_gemm`, & `jit_tilelang_wave_guarded_gemm`:
   `examples/gemm/example_gemm_intrinsic_persistent.py` @ cbc67f2 (`prepared/persistent_gemm/input.pto`,
   8192x1024x8192, 20 cores, 13 waves). PTOAS FrontierSynch baselines: Oct 6 rejected; Oct 8 s/w=257/257,
   if=507 (2951 LOC); Oct 16 s/w=57/57, if=115 (1318 LOC).
3. `jit_tilelang_swizzle_gemm` & `jit_tilelang_swizzle_tail_gemm`:
   `examples/gemm/example_gemm_intrinsic.py` @ cbc67f2 (`T.use_swizzle(..., off=3)`, panel width 8 +
   M-zigzag, full panels -> `whole-exact` in M2/M3; variable-width tail panel -> `EventfulSyncChild`).
4. `jit_tilelang_pipelined_gemm`:
   `examples/pipeline/gemm_v0_pipeline.py` (`T.Pipelined` 2-stage L1 & L0 -> `pl.pipeline(stage=2)`).
5. `jit_tilelang_dual_gemm` & `jit_tilelang_gate_up_seq_gemm`:
   `examples/fusedmoe/example_fusedmoe.py:dual_gemm_kernel` (`test/samples/FFN/ffn_fc1_splitk.pto`).
6. `jit_tilelang_tail_k_gemm`:
   `examples/gemm/example_gemm_tail_block_developer.py` (unaligned K = 4*256 + 64 tail epilogue).
7. `jit_tilelang_splitk_gemm`:
   `examples_experiment/gemm_splitk/example_tilelang_gemm_splitk.py` (Split-K=4 with atomic FP32 add).
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType, pto_backend
from pypto.debug import torch_codegen
from pypto.pypto_core import ir as core_ir
from pypto.runtime import RunConfig

from .sync_closed_command_oracle import (
    ClosedCommands,
    check_closed_commands,
    opname,
)
from .sync_matmul_oracle import (
    define_slot_storage,
    protected_accumulator_pairs,
)
from .sync_physical_oracle import run_physical

pytestmark = pytest.mark.usefixtures("ascend_backend")

MODES: tuple[tuple[str, bool, str, bool], ...] = (
    ("M0_ptoas", False, "exact", False),
    ("M1_prefix_safe", True, "conservative_prefix", False),
    ("M2_prefix_disjoint", True, "conservative_prefix", True),
    ("M3_exact_disjoint", True, "exact", True),
)

LOCAL_BARRIER_PIPES = {
    opname("system.bar_mte2"): ir.PipeType.MTE2,
    opname("system.bar_mte3"): ir.PipeType.MTE3,
    opname("system.bar_v"): ir.PipeType.V,
    opname("system.bar_mte1"): ir.PipeType.MTE1,
    opname("system.bar_m"): ir.PipeType.M,
    opname("system.bar_fix"): ir.PipeType.FIX,
}


def walk_stmts(stmt: ir.Stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from walk_stmts(child)
    elif isinstance(stmt, ir.IfStmt):
        yield from walk_stmts(stmt.then_body)
        if stmt.else_body is not None:
            yield from walk_stmts(stmt.else_body)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield from walk_stmts(stmt.body)


def walk_exprs(expr: ir.Expr | None):
    if expr is None:
        return
    yield expr
    if isinstance(expr, ir.BinaryExpr):
        yield from walk_exprs(expr.left)
        yield from walk_exprs(expr.right)
    elif isinstance(expr, ir.UnaryExpr):
        yield from walk_exprs(expr.operand)
    elif isinstance(expr, ir.Call):
        for arg in expr.args:
            yield from walk_exprs(arg)
        for val in expr.kwargs.values():
            if isinstance(val, ir.Expr):
                yield from walk_exprs(val)
    elif isinstance(expr, ir.MakeTuple):
        for elem in expr.elements:
            yield from walk_exprs(elem)


def _count_div_mod(expr: ir.Expr | None, counts: Counter) -> None:
    for node in walk_exprs(expr):
        if isinstance(node, ir.FloorDiv):
            counts["fdiv"] += 1
        elif isinstance(node, ir.FloorMod):
            counts["fmod"] += 1


def collect_sync_metrics(fn: ir.Function) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    event_ids: set[int] = set()
    for stmt in walk_stmts(fn.body):
        if isinstance(stmt, ir.IfStmt):
            counts["phys_if"] += 1
            _count_div_mod(stmt.condition, counts)
        elif isinstance(stmt, ir.ForStmt):
            for bound in (stmt.start, stmt.stop, stmt.step):
                _count_div_mod(bound, counts)
        elif isinstance(stmt, ir.AssignStmt):
            _count_div_mod(stmt.value, counts)
        elif isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
            call = stmt.expr
            op = call.op.name
            if op in ("system.set_flag", "system.wait_flag"):
                counts["set_flag" if op == "system.set_flag" else "wait_flag"] += 1
                ev = call.kwargs["event_id"]
                event_ids.add(int(ev.value) if isinstance(ev, ir.ConstInt) else int(ev))
            elif op in {"system.bar_all", "system.barrier"}:
                counts["bar_all"] += 1
            elif op.startswith("system.bar_"):
                counts["bar_local"] += 1
    return {
        "phys_if": counts["phys_if"],
        "set_flag": counts["set_flag"],
        "wait_flag": counts["wait_flag"],
        "bar_local": counts["bar_local"],
        "bar_all": counts["bar_all"],
        "fdiv": counts["fdiv"],
        "fmod": counts["fmod"],
        "event_ids": sorted(event_ids),
        "event_id_count": len(event_ids),
    }


def aic_function(program: ir.Program) -> ir.Function:
    return next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.AIC)


def lower_jit(jit_fn: Any, shapes: list[tuple[tuple[int, ...], torch.dtype]]) -> ir.Program:
    meta_args = [torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in shapes]
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        return jit_fn.lower(*meta_args, config=RunConfig(arch=BackendType.Ascend910))


def run_mode(
    program: ir.Program, *, enable_sync: bool, policy: str, disjoint_gm: bool
) -> tuple[ir.Program, ir.Program | None]:
    if not enable_sync:
        return program, None
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy=policy,
        autosync_disjoint_gm=disjoint_gm,
    ):
        realized = passes.realize_sync()(program)
        if bool(aic_function(realized).attrs.get(PYPTO_AUTOSYNC_ATTR, False)):
            three_pass = passes.lower_sync()(passes.optimize_sync()(passes.insert_sync()(program)))
        else:
            three_pass = None
    return realized, three_pass


def specialize_launch_bounds(fn: ir.Function, *, block_idx: int, block_num: int) -> ir.Function:
    """Bind symbolic SPMD launch queries to concrete core coordinates for oracle execution."""
    span = fn.span
    idx_ty = ir.DataType.INDEX
    idx_op = opname("buffer.get_block_idx")
    num_op = opname("buffer.get_block_num")

    class BindLaunch(ir.IRMutator):
        def visit_call(self, op: ir.Call):
            if op.op.name == idx_op:
                return ir.ConstInt(block_idx, idx_ty, span)
            if op.op.name == num_op:
                return ir.ConstInt(block_num, idx_ty, span)
            return super().visit_call(op)

    return BindLaunch().visit_function(fn)


class MultiSlotClosedCommands(ClosedCommands):
    """ClosedCommands extension adding `buffer.alloc_multi` + `buffer.slot` via `define_slot_storage`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.slot_pools: dict[Any, Any] = {}

    def physical(self, call: ir.Call) -> None:
        if call.op.name in LOCAL_BARRIER_PIPES and "core_type" not in call.kwargs:
            self.barrier(LOCAL_BARRIER_PIPES[call.op.name])
            return
        super().physical(call)

    def assign(self, statement: ir.AssignStmt, env: dict[Any, Any]) -> None:
        call = statement.value
        if isinstance(call, ir.Call) and define_slot_storage(statement, env, self.slot_pools):
            entry = self.slot_pools[statement.var]
            space, base, height, width, layout, scalar_bytes = entry
            if call.op.name == opname("buffer.alloc_multi"):
                assert isinstance(statement.var.type, ir.MultiBufferType)
                self.descriptors.append(
                    ("alloc_multi", space, base, height, width, scalar_bytes, statement.var.type.slot_count)
                )
            else:
                self.storage[statement.var] = (space, base, height, width, scalar_bytes)
                self.layouts[statement.var] = layout
            return
        super().assign(statement, env)


def check_multislot_closed_commands(
    logical: ir.Function,
    physical: ir.Function,
    parameters: dict[int, Any],
    *,
    hardware_source: ir.Function,
    tensor_bases: dict[int, int],
    block_index: int | None = None,
) -> None:
    before = MultiSlotClosedCommands(logical, parameters, range(8), tensor_bases, block_index=block_index)
    after = MultiSlotClosedCommands(physical, parameters, range(8), tensor_bases, block_index=block_index)
    expected, actual = before.run(), after.run()
    assert before.payloads == after.payloads, "changed payload operands or physical bytes"
    assert before.descriptors == after.descriptors, "changed original allocations"
    common = {node for node in before.vertices if node[0] == "payload"}
    assert {(a, b) for a, b in expected if a in common and b in common} <= {
        (a, b) for a, b in actual if a in common and b in common
    }, "lost actual-token payload order"
    original = MultiSlotClosedCommands(
        hardware_source, parameters, range(8), tensor_bases, block_index=block_index
    )
    original.run()
    assert original.payloads == before.payloads, "changed original Cube payload geometry"
    protected = protected_accumulator_pairs(hardware_source, original.calls)
    for i, (_, _, reads, writes) in enumerate(before.payloads):
        for j in range(i):
            _, _, earlier_reads, earlier_writes = before.payloads[j]
            if (earlier_writes & (reads | writes) or earlier_reads & writes) and (j, i) not in protected:
                assert (("payload", j, "C"), ("payload", i, "I")) in actual, (
                    "unordered physical-byte conflict",
                    j,
                    i,
                )


# ==============================================================================
# 1. Exact Buffer-IR Ports of `tilelang_gemm` & `persistent_gemm` (L1 Lookahead)
# ==============================================================================


@dataclass(frozen=True)
class LookaheadConfig:
    name: str
    m: int = 256
    n: int = 256
    k_total: int = 4096
    block_m: int = 128
    block_n: int = 256
    k_l1: int = 256
    block_k: int = 64
    persistent_cores: int | None = None


def _make_lookahead_tile_body(
    cfg: LookaheadConfig,
    pools: tuple[ir.Var, ir.Var, ir.Var, ir.Var, ir.Var],
    params: tuple[ir.Var, ir.Var, ir.Var],
    coords: tuple[ir.Var, ir.Var],
) -> list[ir.Stmt]:
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    c_param, a_param, b_param = params
    acc, a_l1_pool, b_l1_pool, a_l0_pool, b_l0_pool = pools
    row_var, col_var = coords

    def pool_element(pool: ir.Var) -> ir.BufferType:
        typ = pool.type
        assert isinstance(typ, ir.MultiBufferType)
        return typ.element_type

    a_mat_t, b_mat_t, left_t, right_t = map(pool_element, (a_l1_pool, b_l1_pool, a_l0_pool, b_l0_pool))

    def ci(val: int) -> ir.ConstInt:
        return ir.ConstInt(val, idx, span)

    def tup(vals: list[Any]) -> ir.MakeTuple:
        return ir.MakeTuple([ci(v) if isinstance(v, int) else v for v in vals], span)

    def call_op(op_name: str, args: list[Any], ret_type: Any = None) -> ir.Call:
        if ret_type is None:
            return core_ir._create_internal_op_call(op_name, args, {}, span)
        return core_ir._create_internal_op_call(op_name, args, {}, ret_type, span)

    loop_k = cfg.k_total // cfg.k_l1
    loop_kk = cfg.k_l1 // cfg.block_k
    a0, b0 = ir.Var("a_first", a_mat_t, span), ir.Var("b_first", b_mat_t, span)
    k_iv, kk_iv = ir.Var("k", ir.ScalarType(idx), span), ir.Var("kk", ir.ScalarType(idx), span)
    next_k, next_slot = (
        ir.Var("next_k", ir.ScalarType(idx), span),
        ir.Var("next_slot", ir.ScalarType(idx), span),
    )
    next_col, slot = ir.Var("next_col", ir.ScalarType(idx), span), ir.Var("slot", ir.ScalarType(idx), span)
    a_next, b_next = ir.Var("a_next", a_mat_t, span), ir.Var("b_next", b_mat_t, span)
    a_cur, b_cur = ir.Var("a_cur", a_mat_t, span), ir.Var("b_cur", b_mat_t, span)
    side, ko = ir.Var("side", ir.ScalarType(idx), span), ir.Var("ko", ir.ScalarType(idx), span)
    a_l0, b_l0 = ir.Var("a_l0", left_t, span), ir.Var("b_l0", right_t, span)

    prefetch_body = ir.SeqStmts(
        [
            ir.AssignStmt(next_slot, ir.FloorMod(next_k, ci(2), idx, span), span),
            ir.AssignStmt(next_col, ir.Mul(next_k, ci(cfg.k_l1), idx, span), span),
            ir.AssignStmt(a_next, call_op("buffer.slot", [a_l1_pool, next_slot], a_mat_t), span),
            ir.AssignStmt(b_next, call_op("buffer.slot", [b_l1_pool, next_slot], b_mat_t), span),
            ir.EvalStmt(
                call_op(
                    "buffer.load", [a_param, tup([row_var, next_col]), tup([cfg.block_m, cfg.k_l1]), a_next]
                ),
                span,
            ),
            ir.EvalStmt(
                call_op(
                    "buffer.load", [b_param, tup([next_col, col_var]), tup([cfg.k_l1, cfg.block_n]), b_next]
                ),
                span,
            ),
            ir.YieldStmt([], span),
        ],
        span,
    )
    kk_loop = ir.ForStmt(
        kk_iv,
        ci(0),
        ci(loop_kk),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(side, ir.FloorMod(kk_iv, ci(2), idx, span), span),
                ir.AssignStmt(ko, ir.Mul(kk_iv, ci(cfg.block_k), idx, span), span),
                ir.AssignStmt(a_l0, call_op("buffer.slot", [a_l0_pool, side], left_t), span),
                ir.AssignStmt(b_l0, call_op("buffer.slot", [b_l0_pool, side], right_t), span),
                ir.EvalStmt(call_op("buffer.extract", [a_cur, ci(0), ko, a_l0]), span),
                ir.EvalStmt(call_op("buffer.extract", [b_cur, ko, ci(0), b_l0]), span),
                ir.IfStmt(
                    ir.And(
                        ir.Eq(k_iv, ci(0), ir.DataType.BOOL, span),
                        ir.Eq(kk_iv, ci(0), ir.DataType.BOOL, span),
                        ir.DataType.BOOL,
                        span,
                    ),
                    ir.SeqStmts(
                        [
                            ir.EvalStmt(call_op("buffer.matmul", [a_l0, b_l0, acc]), span),
                            ir.YieldStmt([], span),
                        ],
                        span,
                    ),
                    ir.SeqStmts(
                        [
                            ir.EvalStmt(call_op("buffer.matmul_acc", [a_l0, b_l0, acc]), span),
                            ir.YieldStmt([], span),
                        ],
                        span,
                    ),
                    [],
                    span,
                ),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    k_loop = ir.ForStmt(
        k_iv,
        ci(0),
        ci(loop_k),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(slot, ir.FloorMod(k_iv, ci(2), idx, span), span),
                ir.AssignStmt(next_k, ir.Add(k_iv, ci(1), idx, span), span),
                ir.IfStmt(ir.Lt(next_k, ci(loop_k), ir.DataType.BOOL, span), prefetch_body, None, [], span),
                ir.AssignStmt(a_cur, call_op("buffer.slot", [a_l1_pool, slot], a_mat_t), span),
                ir.AssignStmt(b_cur, call_op("buffer.slot", [b_l1_pool, slot], b_mat_t), span),
                kk_loop,
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    return [
        ir.AssignStmt(a0, call_op("buffer.slot", [a_l1_pool, ci(0)], a_mat_t), span),
        ir.AssignStmt(b0, call_op("buffer.slot", [b_l1_pool, ci(0)], b_mat_t), span),
        ir.EvalStmt(
            call_op("buffer.load", [a_param, tup([row_var, ci(0)]), tup([cfg.block_m, cfg.k_l1]), a0]), span
        ),
        ir.EvalStmt(
            call_op("buffer.load", [b_param, tup([ci(0), col_var]), tup([cfg.k_l1, cfg.block_n]), b0]), span
        ),
        k_loop,
        ir.EvalStmt(
            call_op("buffer.store", [acc, tup([row_var, col_var]), tup([cfg.block_m, cfg.block_n]), c_param]),
            span,
        ),
    ]


def build_tilelang_lookahead_buffer_ir(cfg: LookaheadConfig) -> ir.Program:
    """Build exact Buffer-IR port of `prepared/tilelang/input.pto` or `prepared/persistent_gemm/input.pto`."""
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX

    def ci(val: int) -> ir.ConstInt:
        return ir.ConstInt(val, idx, span)

    def tup(vals: list[Any]) -> ir.MakeTuple:
        return ir.MakeTuple([ci(v) if isinstance(v, int) else v for v in vals], span)

    def call_op(op_name: str, args: list[Any], ret_type: Any = None) -> ir.Call:
        if ret_type is None:
            return core_ir._create_internal_op_call(op_name, args, {}, span)
        return core_ir._create_internal_op_call(op_name, args, {}, ret_type, span)

    a_mat_t = ir.BufferType(
        [cfg.block_m, cfg.k_l1],
        ir.DataType.FP16,
        ir.MemorySpace.Mat,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=512,
    )
    b_mat_t = ir.BufferType(
        [cfg.k_l1, cfg.block_n],
        ir.DataType.FP16,
        ir.MemorySpace.Mat,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=512,
    )
    left_t = ir.BufferType(
        [cfg.block_m, cfg.block_k],
        ir.DataType.FP16,
        ir.MemorySpace.Left,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.row_major,
        fractal=512,
    )
    right_t = ir.BufferType(
        [cfg.block_k, cfg.block_n],
        ir.DataType.FP16,
        ir.MemorySpace.Right,
        slayout=ir.TileLayout.col_major,
        blayout=ir.TileLayout.row_major,
        fractal=512,
    )
    acc_t = ir.BufferType(
        [cfg.block_m, cfg.block_n],
        ir.DataType.FP32,
        ir.MemorySpace.Acc,
        slayout=ir.TileLayout.row_major,
        blayout=ir.TileLayout.col_major,
        fractal=1024,
    )

    c_param = ir.Var("c", ir.TensorType([cfg.m, cfg.n], ir.DataType.FP16), span)
    a_param = ir.Var("a", ir.TensorType([cfg.m, cfg.k_total], ir.DataType.FP16), span)
    b_param = ir.Var("b", ir.TensorType([cfg.k_total, cfg.n], ir.DataType.FP16), span)
    acc = ir.Var("acc", acc_t, span)
    a_l1_pool = ir.Var("a_l1_pool", ir.MultiBufferType(a_mat_t, 2), span)
    b_l1_pool = ir.Var("b_l1_pool", ir.MultiBufferType(b_mat_t, 2), span)
    a_l0_pool = ir.Var("a_l0_pool", ir.MultiBufferType(left_t, 2), span)
    b_l0_pool = ir.Var("b_l0_pool", ir.MultiBufferType(right_t, 2), span)

    b_l1_offset = 2 * cfg.block_m * cfg.k_l1 * 2
    allocs: list[ir.Stmt] = [
        ir.AssignStmt(acc, call_op("buffer.alloc", [tup([]), ci(0)], acc_t), span),
        ir.AssignStmt(a_l1_pool, call_op("buffer.alloc_multi", [tup([]), ci(0)], a_l1_pool.type), span),
        ir.AssignStmt(
            b_l1_pool, call_op("buffer.alloc_multi", [tup([]), ci(b_l1_offset)], b_l1_pool.type), span
        ),
        ir.AssignStmt(a_l0_pool, call_op("buffer.alloc_multi", [tup([]), ci(0)], a_l0_pool.type), span),
        ir.AssignStmt(b_l0_pool, call_op("buffer.alloc_multi", [tup([]), ci(0)], b_l0_pool.type), span),
    ]
    pools = (acc, a_l1_pool, b_l1_pool, a_l0_pool, b_l0_pool)
    params = (c_param, a_param, b_param)
    row_var, col_var = ir.Var("row", ir.ScalarType(idx), span), ir.Var("col", ir.ScalarType(idx), span)

    if cfg.persistent_cores is None:
        tile_iv = ir.Var("tile", ir.ScalarType(idx), span)
        outer = ir.ForStmt(
            tile_iv,
            ci(0),
            ci(cfg.m // cfg.block_m),
            ci(1),
            [],
            ir.SeqStmts(
                [
                    ir.AssignStmt(row_var, ir.Mul(tile_iv, ci(cfg.block_m), idx, span), span),
                    ir.AssignStmt(col_var, ci(0), span),
                    *_make_lookahead_tile_body(cfg, pools, params, (row_var, col_var)),
                    ir.YieldStmt([], span),
                ],
                span,
            ),
            [],
            span,
        )
        body = ir.SeqStmts([*allocs, outer, ir.ReturnStmt([c_param], span)], span)
    else:
        cid = ir.Var("cid", ir.ScalarType(idx), span)
        wave_iv = ir.Var("wave", ir.ScalarType(idx), span)
        task = ir.Var("task", ir.ScalarType(idx), span)
        n_blocks = cfg.n // cfg.block_n
        total_tiles = (cfg.m // cfg.block_m) * n_blocks
        waves = (total_tiles + cfg.persistent_cores - 1) // cfg.persistent_cores
        outer = ir.ForStmt(
            wave_iv,
            ci(0),
            ci(waves),
            ci(1),
            [],
            ir.SeqStmts(
                [
                    ir.AssignStmt(
                        task,
                        ir.Add(ir.Mul(wave_iv, ci(cfg.persistent_cores), idx, span), cid, idx, span),
                        span,
                    ),
                    ir.IfStmt(
                        ir.Lt(task, ci(total_tiles), ir.DataType.BOOL, span),
                        ir.SeqStmts(
                            [
                                ir.AssignStmt(
                                    row_var,
                                    ir.Mul(
                                        ir.FloorDiv(task, ci(n_blocks), idx, span), ci(cfg.block_m), idx, span
                                    ),
                                    span,
                                ),
                                ir.AssignStmt(
                                    col_var,
                                    ir.Mul(
                                        ir.FloorMod(task, ci(n_blocks), idx, span), ci(cfg.block_n), idx, span
                                    ),
                                    span,
                                ),
                                *_make_lookahead_tile_body(cfg, pools, params, (row_var, col_var)),
                                ir.YieldStmt([], span),
                            ],
                            span,
                        ),
                        None,
                        [],
                        span,
                    ),
                    ir.YieldStmt([], span),
                ],
                span,
            ),
            [],
            span,
        )
        body = ir.SeqStmts(
            [
                ir.AssignStmt(cid, call_op("buffer.get_block_idx", []), span),
                *allocs,
                outer,
                ir.ReturnStmt([c_param], span),
            ],
            span,
        )

    fn = ir.Function(
        cfg.name,
        [(c_param, ir.ParamDirection.Out), (a_param, ir.ParamDirection.In), (b_param, ir.ParamDirection.In)],
        [c_param.type],
        body,
        span,
        type=ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([fn], cfg.name + "_prog", span)


# ==============================================================================
# 2. Full-Scale `@pl.jit` PyPTO TileLang GEMM Variants
# ==============================================================================


@pl.jit
def jit_tilelang_gemm_2tile(
    a: pl.Tensor[[256, 4096], pl.FP16],
    b: pl.Tensor[[4096, 256], pl.FP16],
    c: pl.Out[pl.Tensor[[256, 256], pl.FP16]],
):
    with pl.spmd(1, name_hint="tl_gemm_2tile", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for tile in pl.range(cid, 2, cnum):
            row = tile * 128
            acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 16):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, 0], [256, 256], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, 0], c)
    return c


@pl.jit
def jit_tilelang_gemm_grid(
    a: pl.Tensor[[1024, 1024], pl.FP16],
    b: pl.Tensor[[1024, 1024], pl.FP16],
    c: pl.Out[pl.Tensor[[1024, 1024], pl.FP16]],
):
    with pl.spmd(8, name_hint="tl_gemm_grid", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 32, cnum):
            row = (task // 4) * 128
            col = (task % 4) * 256
            acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 4):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [256, 256], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def jit_tilelang_persistent_gemm(
    a: pl.Tensor[[8192, 8192], pl.FP16],
    b: pl.Tensor[[8192, 1024], pl.FP16],
    c: pl.Out[pl.Tensor[[8192, 1024], pl.FP16]],
):
    with pl.spmd(20, name_hint="tl_persistent_gemm", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 256, cnum):
            row = (task // 4) * 128
            col = (task % 4) * 256
            acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 32):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [256, 256], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def jit_tilelang_wave_guarded_gemm(
    a: pl.Tensor[[8192, 8192], pl.FP16],
    b: pl.Tensor[[8192, 1024], pl.FP16],
    c: pl.Out[pl.Tensor[[8192, 1024], pl.FP16]],
):
    for cid in pl.spmd(20, name_hint="tl_wave_guarded"):
        for wave in pl.range(0, 13):
            task = wave * 20 + cid
            if task < 256:
                row = (task // 4) * 128
                col = (task % 4) * 256
                acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                mat_slot = 1
                for k in pl.range(0, 32):
                    mat_slot = (mat_slot + 1) % 2
                    k0 = k * 256
                    a_mat: pl.Tile[
                        [128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat
                    ] = pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                    b_mat: pl.Tile[
                        [256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat
                    ] = pl.load(b, [k0, col], [256, 256], target_memory=pl.Mem.Mat)
                    l0_slot = 1
                    for kk in pl.range(0, 4):
                        l0_slot = (l0_slot + 1) % 2
                        ko = kk * 64
                        a_l0: pl.Tile[
                            [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                        ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                        b_l0: pl.Tile[
                            [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                        ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                        if k == 0 and kk == 0:
                            acc = pl.matmul(a_l0, b_l0)
                        else:
                            acc = pl.matmul_acc(acc, a_l0, b_l0)
                c = pl.store(acc, [row, col], c)
    return c


SW_FULL_M = 32
SW_FULL_N = 32
SW_TAIL_N = 10
SW_W = 8


@pl.jit.inline
def _tl_swizzle_full(tile_idx: pl.Scalar[pl.INDEX]):
    span = SW_W * SW_FULL_M
    num_groups = (SW_FULL_N + SW_W - 1) // SW_W
    group_idx = tile_idx // span
    in_group = tile_idx % span
    if group_idx == num_groups - 1:
        cols = SW_FULL_N - SW_W * group_idx
    else:
        cols = SW_W
    base_m = in_group // cols
    by = group_idx * SW_W + (in_group % cols)
    if (group_idx % 2) == 1:
        bx = SW_FULL_M - base_m - 1
    else:
        bx = base_m
    return bx, by


@pl.jit.inline
def _tl_swizzle_tail(tile_idx: pl.Scalar[pl.INDEX]):
    span = SW_W * SW_FULL_M
    num_groups = (SW_TAIL_N + SW_W - 1) // SW_W
    group_idx = tile_idx // span
    in_group = tile_idx % span
    if group_idx == num_groups - 1:
        cols = SW_TAIL_N - SW_W * group_idx
    else:
        cols = SW_W
    base_m = in_group // cols
    by = group_idx * SW_W + (in_group % cols)
    if (group_idx % 2) == 1:
        bx = SW_FULL_M - base_m - 1
    else:
        bx = base_m
    return bx, by


@pl.jit
def jit_tilelang_swizzle_gemm(
    a: pl.Tensor[[4096, 4096], pl.FP16],
    b: pl.Tensor[[4096, 8192], pl.FP16],
    c: pl.Out[pl.Tensor[[4096, 8192], pl.FP16]],
):
    with pl.spmd(20, name_hint="tl_swizzle_gemm", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 1024, cnum):
            bx, by = _tl_swizzle_full(task)
            row = bx * 128
            col = by * 256
            acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 16):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [256, 256], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def jit_tilelang_swizzle_tail_gemm(
    a: pl.Tensor[[4096, 4096], pl.FP16],
    b: pl.Tensor[[4096, 2560], pl.FP16],
    c: pl.Out[pl.Tensor[[4096, 2560], pl.FP16]],
):
    with pl.spmd(20, name_hint="tl_swizzle_tail_gemm", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 320, cnum):
            bx, by = _tl_swizzle_tail(task)
            row = bx * 128
            col = by * 256
            acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 16):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [256, 256], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def jit_tilelang_pipelined_gemm(
    a: pl.Tensor[[1024, 1024], pl.FP16],
    b: pl.Tensor[[1024, 1024], pl.FP16],
    c: pl.Out[pl.Tensor[[1024, 1024], pl.FP16]],
):
    for mb in pl.parallel(0, 1024, 128):
        for nb in pl.parallel(0, 1024, 128):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tl_pipelined_tile"):
                acc = pl.create_tile([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for k in pl.pipeline(0, 4, stage=2):
                    k0 = k * 256
                    a_l1: pl.Tile[[128, 256], pl.FP16, pl.Mem.Mat] = pl.load(
                        a, [mb, k0], [128, 256], target_memory=pl.Mem.Mat
                    )
                    b_l1: pl.Tile[[256, 128], pl.FP16, pl.Mem.Mat] = pl.load(
                        b, [k0, nb], [256, 128], target_memory=pl.Mem.Mat
                    )
                    for kk in pl.pipeline(0, 4, stage=2):
                        ko = kk * 64
                        a_l0: pl.Tile[[128, 64], pl.FP16, pl.Mem.Left] = pl.tile.extract(
                            a_l1, 0, ko, [128, 64], target_memory=pl.Mem.Left
                        )
                        b_l0: pl.Tile[[64, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(
                            b_l1, ko, 0, [64, 128], target_memory=pl.Mem.Right
                        )
                        if k == 0 and kk == 0:
                            acc = pl.matmul(a_l0, b_l0)
                        else:
                            acc = pl.matmul_acc(acc, a_l0, b_l0)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def jit_tilelang_dual_gemm(
    x: pl.Tensor[[256, 1024], pl.FP16],
    w_gate: pl.Tensor[[1024, 512], pl.FP16],
    w_up: pl.Tensor[[1024, 512], pl.FP16],
    gate_out: pl.Out[pl.Tensor[[256, 512], pl.FP16]],
    up_out: pl.Out[pl.Tensor[[256, 512], pl.FP16]],
):
    with pl.spmd(8, name_hint="tl_dual_gemm", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 16, cnum):
            row = (task // 4) * 64
            col = (task % 4) * 128
            gate_acc = pl.create_tile([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            up_acc = pl.create_tile([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 4):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[64, 256], pl.FP16, pl.MemRef("a_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(x, [row, k0], [64, 256], target_memory=pl.Mem.Mat)
                )
                bg_mat: pl.Tile[[256, 128], pl.FP16, pl.MemRef("bg_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(w_gate, [k0, col], [256, 128], target_memory=pl.Mem.Mat)
                )
                for kk in pl.range(0, 4):
                    ko = kk * 64
                    a_l0: pl.Tile[[64, 64], pl.FP16, pl.MemRef("a_l0", slots=2)[kk % 2], pl.Mem.Left] = (
                        pl.tile.extract(a_mat, 0, ko, [64, 64], target_memory=pl.Mem.Left)
                    )
                    bg_l0: pl.Tile[[64, 128], pl.FP16, pl.MemRef("bg_l0", slots=2)[kk % 2], pl.Mem.Right] = (
                        pl.tile.extract(bg_mat, ko, 0, [64, 128], target_memory=pl.Mem.Right)
                    )
                    if k == 0 and kk == 0:
                        gate_acc = pl.matmul(a_l0, bg_l0)
                    else:
                        gate_acc = pl.matmul_acc(gate_acc, a_l0, bg_l0)
                bu_mat: pl.Tile[[256, 128], pl.FP16, pl.MemRef("bu_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(w_up, [k0, col], [256, 128], target_memory=pl.Mem.Mat)
                )
                for kk2 in pl.range(0, 4):
                    ko2 = kk2 * 64
                    a_l0_2: pl.Tile[[64, 64], pl.FP16, pl.MemRef("a_l0_u", slots=2)[kk2 % 2], pl.Mem.Left] = (
                        pl.tile.extract(a_mat, 0, ko2, [64, 64], target_memory=pl.Mem.Left)
                    )
                    bu_l0: pl.Tile[[64, 128], pl.FP16, pl.MemRef("bu_l0", slots=2)[kk2 % 2], pl.Mem.Right] = (
                        pl.tile.extract(bu_mat, ko2, 0, [64, 128], target_memory=pl.Mem.Right)
                    )
                    if k == 0 and kk2 == 0:
                        up_acc = pl.matmul(a_l0_2, bu_l0)
                    else:
                        up_acc = pl.matmul_acc(up_acc, a_l0_2, bu_l0)
            gate_out = pl.store(gate_acc, [row, col], gate_out)
            up_out = pl.store(up_acc, [row, col], up_out)
    return gate_out, up_out


@pl.jit
def jit_tilelang_gate_up_seq_gemm(
    x: pl.Tensor[[256, 1024], pl.FP16],
    w_gate: pl.Tensor[[1024, 512], pl.FP16],
    w_up: pl.Tensor[[1024, 512], pl.FP16],
    gate_out: pl.Out[pl.Tensor[[256, 512], pl.FP16]],
    up_out: pl.Out[pl.Tensor[[256, 512], pl.FP16]],
):
    with pl.spmd(8, name_hint="tl_gate_up_seq", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 8, cnum):
            row = (task // 4) * 128
            col = (task % 4) * 128
            gate_acc = pl.create_tile([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for kg in pl.range(0, 4):
                mat_slot = (mat_slot + 1) % 2
                k0 = kg * 256
                ag_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("ag_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(x, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                bg_mat: pl.Tile[[256, 128], pl.FP16, pl.MemRef("bg_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(w_gate, [k0, col], [256, 128], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kkg in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kkg * 64
                    ag_l0: pl.Tile[[128, 64], pl.FP16, pl.MemRef("ag_l0", slots=2)[l0_slot], pl.Mem.Left] = (
                        pl.tile.extract(ag_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    )
                    bg_l0: pl.Tile[[64, 128], pl.FP16, pl.MemRef("bg_l0", slots=2)[l0_slot], pl.Mem.Right] = (
                        pl.tile.extract(bg_mat, ko, 0, [64, 128], target_memory=pl.Mem.Right)
                    )
                    if kg == 0 and kkg == 0:
                        gate_acc = pl.matmul(ag_l0, bg_l0)
                    else:
                        gate_acc = pl.matmul_acc(gate_acc, ag_l0, bg_l0)
            gate_out = pl.store(gate_acc, [row, col], gate_out)

            up_acc = pl.create_tile([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot_u = 1
            for ku in pl.range(0, 4):
                mat_slot_u = (mat_slot_u + 1) % 2
                k0u = ku * 256
                au_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("au_l1", slots=2)[mat_slot_u], pl.Mem.Mat] = (
                    pl.load(x, [row, k0u], [128, 256], target_memory=pl.Mem.Mat)
                )
                bu_mat: pl.Tile[[256, 128], pl.FP16, pl.MemRef("bu_l1", slots=2)[mat_slot_u], pl.Mem.Mat] = (
                    pl.load(w_up, [k0u, col], [256, 128], target_memory=pl.Mem.Mat)
                )
                l0_slot_u = 1
                for kku in pl.range(0, 4):
                    l0_slot_u = (l0_slot_u + 1) % 2
                    kou = kku * 64
                    au_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("au_l0", slots=2)[l0_slot_u], pl.Mem.Left
                    ] = pl.tile.extract(au_mat, 0, kou, [128, 64], target_memory=pl.Mem.Left)
                    bu_l0: pl.Tile[
                        [64, 128], pl.FP16, pl.MemRef("bu_l0", slots=2)[l0_slot_u], pl.Mem.Right
                    ] = pl.tile.extract(bu_mat, kou, 0, [64, 128], target_memory=pl.Mem.Right)
                    if ku == 0 and kku == 0:
                        up_acc = pl.matmul(au_l0, bu_l0)
                    else:
                        up_acc = pl.matmul_acc(up_acc, au_l0, bu_l0)
            up_out = pl.store(up_acc, [row, col], up_out)
    return gate_out, up_out


@pl.jit
def jit_tilelang_tail_k_gemm(
    a: pl.Tensor[[256, 1088], pl.FP16],
    b: pl.Tensor[[1088, 512], pl.FP16],
    c: pl.Out[pl.Tensor[[256, 512], pl.FP16]],
):
    with pl.spmd(8, name_hint="tl_tail_k_gemm", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 8, cnum):
            row = (task // 4) * 128
            col = (task % 4) * 128
            acc = pl.create_tile([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 4):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 128], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [256, 128], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 128], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 128], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            a_tail_mat: pl.Tile[[128, 64], pl.FP16, pl.Mem.Mat] = pl.load(
                a, [row, 1024], [128, 64], target_memory=pl.Mem.Mat
            )
            b_tail_mat: pl.Tile[[64, 128], pl.FP16, pl.Mem.Mat] = pl.load(
                b, [1024, col], [64, 128], target_memory=pl.Mem.Mat
            )
            a_tail_l0: pl.Tile[[128, 64], pl.FP16, pl.Mem.Left] = pl.tile.extract(
                a_tail_mat, 0, 0, [128, 64], target_memory=pl.Mem.Left
            )
            b_tail_l0: pl.Tile[[64, 128], pl.FP16, pl.Mem.Right] = pl.tile.extract(
                b_tail_mat, 0, 0, [64, 128], target_memory=pl.Mem.Right
            )
            acc = pl.matmul_acc(acc, a_tail_l0, b_tail_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def jit_tilelang_splitk_gemm(
    a: pl.Tensor[[512, 4096], pl.FP16],
    b: pl.Tensor[[4096, 512], pl.FP16],
    c: pl.InOut[pl.Tensor[[512, 512], pl.FP32]],
):
    with pl.spmd(16, name_hint="tl_splitk_gemm", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 32, cnum):
            tile_id = task // 4
            sk = task % 4
            row = (tile_id // 2) * 128
            col = (tile_id % 2) * 256
            k_base = sk * 1024
            acc = pl.create_tile([128, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 4):
                mat_slot = (mat_slot + 1) % 2
                k0 = k_base + k * 256
                a_mat: pl.Tile[[128, 256], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [128, 256], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[256, 256], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [256, 256], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 4):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 64
                    a_l0: pl.Tile[
                        [128, 64], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [64, 256], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [64, 256], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c, atomic=pl.AtomicType.Add)
    return c


# ==============================================================================
# 3. Compact Oracle & Numerical-Golden `@pl.jit` Kernels
# ==============================================================================

COMPACT_SW_M = 4
COMPACT_SW_N = 8
COMPACT_SW_W = 2


@pl.jit.inline
def _compact_swizzle_index(tile_idx: pl.Scalar[pl.INDEX]):
    span = COMPACT_SW_W * COMPACT_SW_M
    num_groups = (COMPACT_SW_N + COMPACT_SW_W - 1) // COMPACT_SW_W
    group_idx = tile_idx // span
    in_group = tile_idx % span
    if group_idx == num_groups - 1:
        cols = COMPACT_SW_N - COMPACT_SW_W * group_idx
    else:
        cols = COMPACT_SW_W
    base_m = in_group // cols
    by = group_idx * COMPACT_SW_W + (in_group % cols)
    if (group_idx % 2) == 1:
        bx = COMPACT_SW_M - base_m - 1
    else:
        bx = base_m
    return bx, by


@pl.jit
def compact_pipelined_gemm(
    a: pl.Tensor[[32, 64], pl.FP16],
    b: pl.Tensor[[64, 32], pl.FP16],
    c: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
):
    for mb in pl.parallel(0, 32, 16):
        for nb in pl.parallel(0, 32, 16):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="compact_pipe"):
                acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for k in pl.pipeline(0, 2, stage=2):
                    k0 = k * 32
                    a_l1: pl.Tile[[16, 32], pl.FP16, pl.Mem.Mat] = pl.load(
                        a, [mb, k0], [16, 32], target_memory=pl.Mem.Mat
                    )
                    b_l1: pl.Tile[[32, 16], pl.FP16, pl.Mem.Mat] = pl.load(
                        b, [k0, nb], [32, 16], target_memory=pl.Mem.Mat
                    )
                    for kk in pl.pipeline(0, 2, stage=2):
                        ko = kk * 16
                        a_l0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left] = pl.tile.extract(
                            a_l1, 0, ko, [16, 16], target_memory=pl.Mem.Left
                        )
                        b_l0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right] = pl.tile.extract(
                            b_l1, ko, 0, [16, 16], target_memory=pl.Mem.Right
                        )
                        if k == 0 and kk == 0:
                            acc = pl.matmul(a_l0, b_l0)
                        else:
                            acc = pl.matmul_acc(acc, a_l0, b_l0)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def compact_swizzle_gemm(
    a: pl.Tensor[[64, 64], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[64, 128], pl.FP16]],
):
    with pl.spmd(4, name_hint="compact_swizzle", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 32, cnum):
            bx, by = _compact_swizzle_index(task)
            row = bx * 16
            col = by * 16
            acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 2):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 32
                a_mat: pl.Tile[[16, 32], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [16, 32], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[32, 16], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [32, 16], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 2):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 16
                    a_l0: pl.Tile[
                        [16, 16], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [16, 16], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [16, 16], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [16, 16], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def compact_wave_guarded_gemm(
    a: pl.Tensor[[48, 64], pl.FP16],
    b: pl.Tensor[[64, 32], pl.FP16],
    c: pl.Out[pl.Tensor[[48, 32], pl.FP16]],
):
    for cid in pl.spmd(4, name_hint="compact_wave"):
        for wave in pl.range(0, 2):
            task = wave * 4 + cid
            if task < 6:
                row = (task // 2) * 16
                col = (task % 2) * 16
                acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                mat_slot = 1
                for k in pl.range(0, 2):
                    mat_slot = (mat_slot + 1) % 2
                    k0 = k * 32
                    a_mat: pl.Tile[
                        [16, 32], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat
                    ] = pl.load(a, [row, k0], [16, 32], target_memory=pl.Mem.Mat)
                    b_mat: pl.Tile[
                        [32, 16], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat
                    ] = pl.load(b, [k0, col], [32, 16], target_memory=pl.Mem.Mat)
                    l0_slot = 1
                    for kk in pl.range(0, 2):
                        l0_slot = (l0_slot + 1) % 2
                        ko = kk * 16
                        a_l0: pl.Tile[
                            [16, 16], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                        ] = pl.tile.extract(a_mat, 0, ko, [16, 16], target_memory=pl.Mem.Left)
                        b_l0: pl.Tile[
                            [16, 16], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                        ] = pl.tile.extract(b_mat, ko, 0, [16, 16], target_memory=pl.Mem.Right)
                        if k == 0 and kk == 0:
                            acc = pl.matmul(a_l0, b_l0)
                        else:
                            acc = pl.matmul_acc(acc, a_l0, b_l0)
                c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def compact_tail_k_gemm(
    a: pl.Tensor[[32, 80], pl.FP16],
    b: pl.Tensor[[80, 32], pl.FP16],
    c: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
):
    with pl.spmd(2, name_hint="compact_tail_k", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 4, cnum):
            row = (task // 2) * 16
            col = (task % 2) * 16
            acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 2):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 32
                a_mat: pl.Tile[[16, 32], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [16, 32], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[32, 16], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [32, 16], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 2):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 16
                    a_l0: pl.Tile[
                        [16, 16], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [16, 16], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [16, 16], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [16, 16], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            a_tail_mat: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.load(
                a, [row, 64], [16, 16], target_memory=pl.Mem.Mat
            )
            b_tail_mat: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.load(
                b, [64, col], [16, 16], target_memory=pl.Mem.Mat
            )
            a_tail_l0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left] = pl.tile.extract(
                a_tail_mat, 0, 0, [16, 16], target_memory=pl.Mem.Left
            )
            b_tail_l0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right] = pl.tile.extract(
                b_tail_mat, 0, 0, [16, 16], target_memory=pl.Mem.Right
            )
            acc = pl.matmul_acc(acc, a_tail_l0, b_tail_l0)
            c = pl.store(acc, [row, col], c)
    return c


@pl.jit
def compact_dual_gemm(
    x: pl.Tensor[[32, 64], pl.FP16],
    w_gate: pl.Tensor[[64, 32], pl.FP16],
    w_up: pl.Tensor[[64, 32], pl.FP16],
    gate_out: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
    up_out: pl.Out[pl.Tensor[[32, 32], pl.FP16]],
):
    with pl.spmd(2, name_hint="compact_dual", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 4, cnum):
            row = (task // 2) * 16
            col = (task % 2) * 16
            gate_acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            up_acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 2):
                mat_slot = (mat_slot + 1) % 2
                k0 = k * 32
                a_mat: pl.Tile[[16, 32], pl.FP16, pl.MemRef("a_l1", slots=2)[mat_slot], pl.Mem.Mat] = pl.load(
                    x, [row, k0], [16, 32], target_memory=pl.Mem.Mat
                )
                bg_mat: pl.Tile[[32, 16], pl.FP16, pl.MemRef("bg_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(w_gate, [k0, col], [32, 16], target_memory=pl.Mem.Mat)
                )
                for kk in pl.range(0, 2):
                    ko = kk * 16
                    a_l0: pl.Tile[[16, 16], pl.FP16, pl.MemRef("a_l0", slots=2)[kk % 2], pl.Mem.Left] = (
                        pl.tile.extract(a_mat, 0, ko, [16, 16], target_memory=pl.Mem.Left)
                    )
                    bg_l0: pl.Tile[[16, 16], pl.FP16, pl.MemRef("bg_l0", slots=2)[kk % 2], pl.Mem.Right] = (
                        pl.tile.extract(bg_mat, ko, 0, [16, 16], target_memory=pl.Mem.Right)
                    )
                    if k == 0 and kk == 0:
                        gate_acc = pl.matmul(a_l0, bg_l0)
                    else:
                        gate_acc = pl.matmul_acc(gate_acc, a_l0, bg_l0)
                bu_mat: pl.Tile[[32, 16], pl.FP16, pl.MemRef("bu_l1", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(w_up, [k0, col], [32, 16], target_memory=pl.Mem.Mat)
                )
                for kk2 in pl.range(0, 2):
                    ko2 = kk2 * 16
                    a_l0_2: pl.Tile[[16, 16], pl.FP16, pl.MemRef("a_l0_u", slots=2)[kk2 % 2], pl.Mem.Left] = (
                        pl.tile.extract(a_mat, 0, ko2, [16, 16], target_memory=pl.Mem.Left)
                    )
                    bu_l0: pl.Tile[[16, 16], pl.FP16, pl.MemRef("bu_l0", slots=2)[kk2 % 2], pl.Mem.Right] = (
                        pl.tile.extract(bu_mat, ko2, 0, [16, 16], target_memory=pl.Mem.Right)
                    )
                    if k == 0 and kk2 == 0:
                        up_acc = pl.matmul(a_l0_2, bu_l0)
                    else:
                        up_acc = pl.matmul_acc(up_acc, a_l0_2, bu_l0)
            gate_out = pl.store(gate_acc, [row, col], gate_out)
            up_out = pl.store(up_acc, [row, col], up_out)
    return gate_out, up_out


@pl.jit
def compact_splitk_gemm(
    a: pl.Tensor[[32, 128], pl.FP16],
    b: pl.Tensor[[128, 32], pl.FP16],
    c: pl.InOut[pl.Tensor[[32, 32], pl.FP32]],
):
    with pl.spmd(4, name_hint="compact_splitk", sync_start=True):
        cid = pl.tile.get_block_idx()
        cnum = pl.tile.get_block_num()
        for task in pl.range(cid, 8, cnum):
            tile_id = task // 2
            sk = task % 2
            row = (tile_id // 2) * 16
            col = (tile_id % 2) * 16
            k_base = sk * 64
            acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            mat_slot = 1
            for k in pl.range(0, 2):
                mat_slot = (mat_slot + 1) % 2
                k0 = k_base + k * 32
                a_mat: pl.Tile[[16, 32], pl.FP16, pl.MemRef("a_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(a, [row, k0], [16, 32], target_memory=pl.Mem.Mat)
                )
                b_mat: pl.Tile[[32, 16], pl.FP16, pl.MemRef("b_l1_pool", slots=2)[mat_slot], pl.Mem.Mat] = (
                    pl.load(b, [k0, col], [32, 16], target_memory=pl.Mem.Mat)
                )
                l0_slot = 1
                for kk in pl.range(0, 2):
                    l0_slot = (l0_slot + 1) % 2
                    ko = kk * 16
                    a_l0: pl.Tile[
                        [16, 16], pl.FP16, pl.MemRef("a_l0_pool", slots=2)[l0_slot], pl.Mem.Left
                    ] = pl.tile.extract(a_mat, 0, ko, [16, 16], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[
                        [16, 16], pl.FP16, pl.MemRef("b_l0_pool", slots=2)[l0_slot], pl.Mem.Right
                    ] = pl.tile.extract(b_mat, ko, 0, [16, 16], target_memory=pl.Mem.Right)
                    if k == 0 and kk == 0:
                        acc = pl.matmul(a_l0, b_l0)
                    else:
                        acc = pl.matmul_acc(acc, a_l0, b_l0)
            c = pl.store(acc, [row, col], c, atomic=pl.AtomicType.Add)
    return c


# ==============================================================================
# 4. Regression Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("cfg", "base_ifs"),
    [
        (LookaheadConfig(name="tilelang_gemm_lookahead"), 2),
        (
            LookaheadConfig(
                name="tilelang_persistent_lookahead", m=8192, n=1024, k_total=8192, persistent_cores=20
            ),
            3,
        ),
    ],
)
def test_tilelang_lookahead_four_modes_and_ptoas_1519_guard(cfg: LookaheadConfig, base_ifs: int):
    program = build_tilelang_lookahead_buffer_ir(cfg)

    for mode_name, en_sync, policy, disjoint in MODES:
        realized, three_pass = run_mode(program, enable_sync=en_sync, policy=policy, disjoint_gm=disjoint)
        fn = aic_function(realized)
        owned = bool(fn.attrs.get(PYPTO_AUTOSYNC_ATTR, False))
        metrics = collect_sync_metrics(fn)

        if mode_name in ("M1_prefix_safe", "M2_prefix_disjoint"):
            assert owned is True
            assert three_pass is not None
            assert collect_sync_metrics(aic_function(three_pass)) == metrics
            # Zero synthetic `if` bloat (vs FrontierSynch's 25 on tilelang_gemm, 507/115 on persistent_gemm).
            assert metrics["phys_if"] == base_ifs
            expected_sw = 8 if mode_name == "M1_prefix_safe" else 12
            expected_ids = [0, 1] if mode_name == "M1_prefix_safe" else [0, 1, 2]
            assert (metrics["set_flag"], metrics["wait_flag"]) == (expected_sw, expected_sw)
            assert metrics["bar_all"] == 1
            assert metrics["event_ids"] == expected_ids
            pto_text = codegen.PTOCodegen().generate(
                ir.Program([fn], fn.name + "_owned", fn.span), emit_tile_addr=True, emit_source_loc=False
            )
            assert "pto.set_flag" in pto_text and "pto.wait_flag" in pto_text
        else:
            assert owned is False
            assert (metrics["set_flag"], metrics["wait_flag"]) == (0, 0)
            # PTOCodegen explicitly refuses un-owned co-live multi-buffer slots due to PTOAS #1519.
            with pytest.raises(ValueError, match="PTOAS #1519"):
                codegen.PTOCodegen().generate(
                    ir.Program([fn], fn.name + "_unowned", fn.span),
                    emit_tile_addr=True,
                    emit_source_loc=False,
                )


def test_tilelang_lookahead_compact_closed_command_hazards():
    bases = {0: 0, 1: 1 << 20, 2: 2 << 20}
    single_cfg = LookaheadConfig(
        name="compact_tl_lookahead", m=64, n=32, k_total=128, block_m=32, block_n=32, k_l1=64, block_k=32
    )
    single_prog = build_tilelang_lookahead_buffer_ir(single_cfg)
    single_orig = aic_function(single_prog)

    for disjoint in (False, True):
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy="conservative_prefix",
            autosync_disjoint_gm=disjoint,
        ):
            opt_log = passes.optimize_sync()(passes.insert_sync()(single_prog))
            lowered = passes.lower_sync()(opt_log)
            realized = passes.realize_sync()(single_prog)
        for cand in (aic_function(lowered), aic_function(realized)):
            check_multislot_closed_commands(
                aic_function(opt_log), cand, {}, hardware_source=single_orig, tensor_bases=bases
            )

    pers_cfg = LookaheadConfig(
        name="compact_tl_persistent",
        m=64,
        n=64,
        k_total=128,
        block_m=32,
        block_n=32,
        k_l1=64,
        block_k=32,
        persistent_cores=3,
    )
    pers_prog = build_tilelang_lookahead_buffer_ir(pers_cfg)
    pers_orig = aic_function(pers_prog)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        pers_log = passes.optimize_sync()(passes.insert_sync()(pers_prog))
        pers_real = passes.realize_sync()(pers_prog)
    # Exercise both a 2-wave active core (cid=0: tasks 0, 3) and a partial-wave core (cid=2: task 2, skip 5).
    for cid in (0, 2):
        check_multislot_closed_commands(
            aic_function(pers_log),
            aic_function(pers_real),
            {},
            hardware_source=pers_orig,
            tensor_bases=bases,
            block_index=cid,
        )


JIT_SUITE_SPECS: tuple[tuple[str, Any, list[tuple[tuple[int, ...], torch.dtype]], dict[str, Any]], ...] = (
    (
        "tilelang_gemm_2tile",
        jit_tilelang_gemm_2tile,
        [((256, 4096), torch.float16), ((4096, 256), torch.float16), ((256, 256), torch.float16)],
        {"m1_children": 1, "m2_route": "prefix", "m3_owned": False, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
    (
        "tilelang_gemm_grid",
        jit_tilelang_gemm_grid,
        [((1024, 1024), torch.float16)] * 3,
        {"m1_children": 1, "m2_route": "whole-exact", "m3_owned": True, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
    (
        "tilelang_persistent_gemm",
        jit_tilelang_persistent_gemm,
        [((8192, 8192), torch.float16), ((8192, 1024), torch.float16), ((8192, 1024), torch.float16)],
        {"m1_children": 1, "m2_route": "whole-exact", "m3_owned": True, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
    (
        "tilelang_wave_guarded_gemm",
        jit_tilelang_wave_guarded_gemm,
        [((8192, 8192), torch.float16), ((8192, 1024), torch.float16), ((8192, 1024), torch.float16)],
        {"m1_children": 1, "m2_route": "prefix", "m3_owned": False, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
    (
        "tilelang_swizzle_gemm",
        jit_tilelang_swizzle_gemm,
        [((4096, 4096), torch.float16), ((4096, 8192), torch.float16), ((4096, 8192), torch.float16)],
        {"m1_children": 1, "m2_route": "whole-exact", "m3_owned": True, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
    (
        "tilelang_swizzle_tail_gemm",
        jit_tilelang_swizzle_tail_gemm,
        [((4096, 4096), torch.float16), ((4096, 2560), torch.float16), ((4096, 2560), torch.float16)],
        {"m1_children": 1, "m2_route": "prefix", "m3_owned": False, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
    (
        "tilelang_pipelined_gemm",
        jit_tilelang_pipelined_gemm,
        [((1024, 1024), torch.float16)] * 3,
        {"m1_children": 1, "m2_route": "whole-exact", "m3_owned": True, "m1_max_ids": 4, "m1_sw": (18, 18)},
    ),
    (
        "tilelang_dual_gemm",
        jit_tilelang_dual_gemm,
        [
            ((256, 1024), torch.float16),
            ((1024, 512), torch.float16),
            ((1024, 512), torch.float16),
            ((256, 512), torch.float16),
            ((256, 512), torch.float16),
        ],
        {
            "m1_children": 0,
            "m2_route": "prefix",
            "m3_owned": False,
            "m1_max_ids": 4,
            "m2_max_ids": 5,
            "m1_sw": (13, 13),
        },
    ),
    (
        "tilelang_gate_up_seq_gemm",
        jit_tilelang_gate_up_seq_gemm,
        [
            ((256, 1024), torch.float16),
            ((1024, 512), torch.float16),
            ((1024, 512), torch.float16),
            ((256, 512), torch.float16),
            ((256, 512), torch.float16),
        ],
        {"m1_children": 2, "m2_route": "whole-exact", "m3_owned": True, "m1_max_ids": 5, "m1_sw": (44, 44)},
    ),
    (
        "tilelang_tail_k_gemm",
        jit_tilelang_tail_k_gemm,
        [((256, 1088), torch.float16), ((1088, 512), torch.float16), ((256, 512), torch.float16)],
        # Tier-2 Class-A windowing absorbs [k, mem_mat_8, mem_mat_9] to cut 25/25 -> 24/24 flags;
        # parent pre-entry rearm reserves ID 0 while tail L1 loads use fresh (MTE2, MTE1) IDs {5, 6}.
        {"m1_children": 1, "m2_route": "whole-exact", "m3_owned": True, "m1_max_ids": 7, "m1_sw": (24, 24)},
    ),
    (
        "tilelang_splitk_gemm",
        jit_tilelang_splitk_gemm,
        [((512, 4096), torch.float16), ((4096, 512), torch.float16), ((512, 512), torch.float32)],
        {"m1_children": 1, "m2_route": "prefix", "m3_owned": False, "m1_max_ids": 5, "m1_sw": (22, 22)},
    ),
)


@pytest.mark.parametrize(
    ("name", "jit_fn", "shapes", "spec"),
    JIT_SUITE_SPECS,
    ids=[entry[0] for entry in JIT_SUITE_SPECS],
)
def test_tilelang_jit_gemm_suite_across_four_modes(
    name: str,
    jit_fn: Any,
    shapes: list[tuple[tuple[int, ...], torch.dtype]],
    spec: dict[str, Any],
):
    program = lower_jit(jit_fn, shapes)
    orig_fn = aic_function(program)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
    ):
        diag = testing.analyze_conservative_sync(
            orig_fn, automatic_children=True, retain_incoming_diagnostics=False
        )
    ready_children = [a for a in diag["child_attempts"] if a["stage"] == "logical_export_ready"]
    assert len(ready_children) == spec["m1_children"]

    for mode_name, en_sync, policy, disjoint in MODES:
        realized, three_pass = run_mode(program, enable_sync=en_sync, policy=policy, disjoint_gm=disjoint)
        fn = aic_function(realized)
        owned = bool(fn.attrs.get(PYPTO_AUTOSYNC_ATTR, False))
        metrics = collect_sync_metrics(fn)

        if mode_name == "M0_ptoas":
            assert owned is False
            assert (metrics["set_flag"], metrics["wait_flag"]) == (0, 0)
            continue

        if mode_name == "M3_exact_disjoint" and not spec["m3_owned"]:
            assert owned is False
            assert "autosync_fallback_reason" in fn.attrs
            continue

        assert owned is True
        assert three_pass is not None
        assert collect_sync_metrics(aic_function(three_pass)) == metrics
        assert metrics["bar_all"] == 1
        assert "% 15" not in str(fn)

        if mode_name == "M1_prefix_safe":
            assert fn.attrs.get("autosync_retained_child_count") == spec["m1_children"]
            assert (metrics["set_flag"], metrics["wait_flag"]) == spec["m1_sw"]
            assert metrics["event_id_count"] <= spec["m1_max_ids"]
        elif spec["m2_route"] == "whole-exact":
            assert fn.attrs.get("autosync_retained_child_count") is None
            assert metrics["bar_local"] == 0
            assert metrics["event_id_count"] <= 8
        else:
            assert fn.attrs.get("autosync_retained_child_count") == spec["m1_children"]
            assert metrics["event_id_count"] <= spec.get("m2_max_ids", spec["m1_max_ids"])

        pto_text = codegen.PTOCodegen().generate(
            ir.Program([fn], fn.name + "_incore", fn.span), emit_tile_addr=True, emit_source_loc=False
        )
        assert "pto.set_flag" in pto_text and "pto.wait_flag" in pto_text
        if name == "tilelang_splitk_gemm":
            assert "atomic_add" in pto_text


def test_tilelang_compact_closed_commands_and_physical_oracles():
    pipe_prog = lower_jit(
        compact_pipelined_gemm,
        [((32, 64), torch.float16), ((64, 32), torch.float16), ((32, 32), torch.float16)],
    )
    pipe_orig = aic_function(pipe_prog)
    pipe_params = {1: 0, 3: 0}
    pipe_bases = {0: 0, 2: 1 << 20, 4: 2 << 20}

    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
    ):
        pipe_m1_log = passes.optimize_sync()(passes.insert_sync()(pipe_prog))
        pipe_m1_real = passes.realize_sync()(pipe_prog)
    check_closed_commands(
        aic_function(pipe_m1_log),
        aic_function(pipe_m1_real),
        pipe_params,
        range(8),
        tensor_bases=pipe_bases,
        compare_commands=False,
        hardware_source=pipe_orig,
    )

    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="exact",
        autosync_disjoint_gm=True,
    ):
        pipe_m3_real = passes.realize_sync()(pipe_prog)
    run_physical(pipe_orig, aic_function(pipe_m3_real), eligible=range(8), parameters=pipe_params)

    # Swizzled & Tail-K whole-exact physical + M1 closed-command checks across concrete worker cores.
    for jit_fn, shapes, cores in (
        (
            compact_swizzle_gemm,
            [((64, 64), torch.float16), ((64, 128), torch.float16), ((64, 128), torch.float16)],
            4,
        ),
        (
            compact_tail_k_gemm,
            [((32, 80), torch.float16), ((80, 32), torch.float16), ((32, 32), torch.float16)],
            2,
        ),
    ):
        prog = lower_jit(jit_fn, shapes)
        orig = aic_function(prog)
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy="conservative_prefix",
            autosync_disjoint_gm=False,
        ):
            m1_log = aic_function(passes.optimize_sync()(passes.insert_sync()(prog)))
            m1_real = aic_function(passes.realize_sync()(prog))
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy="exact",
            autosync_disjoint_gm=True,
        ):
            m3_real = aic_function(passes.realize_sync()(prog))
        assert m3_real.attrs.get(PYPTO_AUTOSYNC_ATTR) is True
        assert m3_real.attrs.get("autosync_retained_child_count") is None
        for cid in (0, cores - 1):
            s_orig = specialize_launch_bounds(orig, block_idx=cid, block_num=cores)
            s_log = specialize_launch_bounds(m1_log, block_idx=cid, block_num=cores)
            s_m1 = specialize_launch_bounds(m1_real, block_idx=cid, block_num=cores)
            s_m3 = specialize_launch_bounds(m3_real, block_idx=cid, block_num=cores)
            check_multislot_closed_commands(
                s_log, s_m1, {}, hardware_source=s_orig, tensor_bases={0: 0, 1: 1 << 20, 2: 2 << 20}
            )
            run_physical(s_orig, s_m3, eligible=range(8))

    # Dual Gate+Up & Split-K atomic ReadWrite closed-command hazard checks.
    for jit_fn, shapes, cores, bases in (
        (
            compact_dual_gemm,
            [
                ((32, 64), torch.float16),
                ((64, 32), torch.float16),
                ((64, 32), torch.float16),
                ((32, 32), torch.float16),
                ((32, 32), torch.float16),
            ],
            2,
            {0: 0, 1: 1 << 20, 2: 2 << 20, 3: 3 << 20, 4: 4 << 20},
        ),
        (
            compact_splitk_gemm,
            [((32, 128), torch.float16), ((128, 32), torch.float16), ((32, 32), torch.float32)],
            4,
            {0: 0, 1: 1 << 20, 2: 2 << 20},
        ),
    ):
        prog = lower_jit(jit_fn, shapes)
        orig = aic_function(prog)
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy="conservative_prefix",
            autosync_disjoint_gm=True,
        ):
            m2_log = aic_function(passes.optimize_sync()(passes.insert_sync()(prog)))
            m2_real = aic_function(passes.realize_sync()(prog))
        for cid in (0, cores - 1):
            check_multislot_closed_commands(
                specialize_launch_bounds(m2_log, block_idx=cid, block_num=cores),
                specialize_launch_bounds(m2_real, block_idx=cid, block_num=cores),
                {},
                hardware_source=specialize_launch_bounds(orig, block_idx=cid, block_num=cores),
                tensor_bases=bases,
            )


def _lower_tile_reference(
    jit_fn: Any,
    shapes: list[tuple[tuple[int, ...], torch.dtype]],
    *,
    block_idx: int = 0,
    cores: int = 1,
) -> ir.Program:
    meta_args = [torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in shapes]
    with passes.PassContext([], enable_buffer_ir=False):
        prog = jit_fn.lower(*meta_args, config=RunConfig(arch=BackendType.Ascend910))
    bidx_op = opname("tile.get_block_idx")
    bnum_op = opname("tile.get_block_num")

    class BindTileLaunch(ir.IRMutator):
        def visit_call(self, op: ir.Call):
            if op.op.name == bidx_op:
                return ir.ConstInt(block_idx, ir.DataType.INDEX, op.span)
            if op.op.name == bnum_op:
                return ir.ConstInt(cores, ir.DataType.INDEX, op.span)
            return super().visit_call(op)

    return ir.Program(
        [BindTileLaunch().visit_function(f) for f in prog.functions.values()], prog.name, prog.span
    )


def _run_spmd_torch_reference(
    fn_name: str,
    jit_fn: Any,
    shapes: list[tuple[tuple[int, ...], torch.dtype]],
    args: list[torch.Tensor],
    *,
    cores: int,
) -> None:
    for cid in range(cores):
        ref_prog = _lower_tile_reference(jit_fn, shapes, block_idx=cid, cores=cores)
        ns: dict[str, Any] = {}
        exec(torch_codegen(ref_prog), ns)  # noqa: S102
        ns[fn_name](*args)


def test_tilelang_gemm_torch_numerical_goldens():
    torch.manual_seed(2026)

    # 1. Post-RealizeSync Buffer-IR execution via torch_codegen (`pl.pipeline` uses `buffer.alloc`).
    pipe_prog = lower_jit(
        compact_pipelined_gemm,
        [((32, 64), torch.float16), ((64, 32), torch.float16), ((32, 32), torch.float16)],
    )
    pipe_synced, _ = run_mode(pipe_prog, enable_sync=True, policy="exact", disjoint_gm=True)
    assert aic_function(pipe_synced).attrs.get(PYPTO_AUTOSYNC_ATTR) is True
    ns_pipe: dict[str, Any] = {}
    exec(torch_codegen(pipe_synced), ns_pipe)  # noqa: S102
    pa = torch.randn(32, 64, dtype=torch.float16) * 0.1
    pb = torch.randn(64, 32, dtype=torch.float16) * 0.1
    pc = torch.zeros(32, 32, dtype=torch.float16)
    ns_pipe["compact_pipelined_gemm"](pa, pb, pc)
    torch.testing.assert_close(pc, (pa.float() @ pb.float()).half(), rtol=1e-2, atol=1e-2)

    # 2. Numerical golden check across swizzled, wave-guarded, and tail-K `@pl.jit` kernels.
    for fn_name, jit_fn, (m, k, n), cores in (
        ("compact_swizzle_gemm", compact_swizzle_gemm, (64, 64, 128), 4),
        ("compact_wave_guarded_gemm", compact_wave_guarded_gemm, (48, 64, 32), 4),
        ("compact_tail_k_gemm", compact_tail_k_gemm, (32, 80, 32), 2),
    ):
        shapes = [((m, k), torch.float16), ((k, n), torch.float16), ((m, n), torch.float16)]
        a = (torch.randn(m, k, dtype=torch.float16) * 0.1).contiguous()
        b = (torch.randn(k, n, dtype=torch.float16) * 0.1).contiguous()
        c = torch.zeros(m, n, dtype=torch.float16)
        _run_spmd_torch_reference(fn_name, jit_fn, shapes, [a, b, c], cores=cores)
        torch.testing.assert_close(c, (a.float() @ b.float()).half(), rtol=1e-2, atol=1e-2)

    # 3. Dual Gate+Up numerical golden across 2 SPMD workers.
    dual_shapes = [
        ((32, 64), torch.float16),
        ((64, 32), torch.float16),
        ((64, 32), torch.float16),
        ((32, 32), torch.float16),
        ((32, 32), torch.float16),
    ]
    x = torch.randn(32, 64, dtype=torch.float16) * 0.1
    wg = torch.randn(64, 32, dtype=torch.float16) * 0.1
    wu = torch.randn(64, 32, dtype=torch.float16) * 0.1
    g_out = torch.zeros(32, 32, dtype=torch.float16)
    u_out = torch.zeros(32, 32, dtype=torch.float16)
    _run_spmd_torch_reference(
        "compact_dual_gemm", compact_dual_gemm, dual_shapes, [x, wg, wu, g_out, u_out], cores=2
    )
    torch.testing.assert_close(g_out, (x.float() @ wg.float()).half(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(u_out, (x.float() @ wu.float()).half(), rtol=1e-2, atol=1e-2)

    # 4. Split-K atomic Add numerical golden across 4 SPMD workers (including non-zero initial C).
    sk_shapes = [((32, 128), torch.float16), ((128, 32), torch.float16), ((32, 32), torch.float32)]
    sa = torch.randn(32, 128, dtype=torch.float16) * 0.1
    sb = torch.randn(128, 32, dtype=torch.float16) * 0.1
    sc = torch.full((32, 32), 2.5, dtype=torch.float32)
    _run_spmd_torch_reference("compact_splitk_gemm", compact_splitk_gemm, sk_shapes, [sa, sb, sc], cores=4)
    torch.testing.assert_close(sc, sa.float() @ sb.float() + 2.5, rtol=1e-2, atol=1e-2)


def test_tilelang_end_to_end_cce_wrapper_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen_pto: list[str] = []

    def stub_ptoas(input_path: str, output_path: str, *, ptoas_flags: list[str] | None = None) -> None:
        pto_text = Path(input_path).read_text()
        seen_pto.append(pto_text)
        assert ptoas_flags is not None and "--enable-insert-sync" in ptoas_flags
        assert "pto.set_flag" in pto_text and "pto.wait_flag" in pto_text
        Path(output_path).write_text("// Stubbed CCE kernel body\n")

    monkeypatch.setattr("pypto.jit.decorator._ptoas_available", lambda: True)
    monkeypatch.setattr(pto_backend, "_find_ptoas_binary", lambda: "/stub/ptoas")
    monkeypatch.setattr(pto_backend, "_run_ptoas", stub_ptoas)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="exact",
        autosync_disjoint_gm=True,
    ):
        compiled = jit_tilelang_swizzle_gemm.compile(
            torch.empty((4096, 4096), dtype=torch.float16, device="meta"),
            torch.empty((4096, 8192), dtype=torch.float16, device="meta"),
            torch.empty((4096, 8192), dtype=torch.float16, device="meta"),
            config=RunConfig(
                arch=BackendType.Ascend910,
                codegen_only=True,
                save_kernels_dir=str(tmp_path),
            ),
        )
    assert compiled.program is not None
    assert len(seen_pto) == 1
    report = json.loads((tmp_path / "report" / "autosync.json").read_text())
    assert report["kernels"][0]["selected_route"] == "pypto"
    cpp_files = list(tmp_path.rglob("*.cpp"))
    assert cpp_files, "expected C++ kernel/orchestration sources to be emitted"
