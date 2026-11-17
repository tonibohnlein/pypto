# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sync qualification and exact/conservative/child realization for merged Buffer IR ops."""

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .test_implicit_sync_effects import SCRATCH, independent_broadcasts
from .test_logical_sync_realization import verify

pytestmark = pytest.mark.usefixtures("ascend_backend")

_POLICIES = (
    ("conservative_prefix", False),
    ("exact", True),
)


def _lower_to_buffer(program: ir.Program) -> ir.Program:
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _kernel(program: ir.Program) -> ir.Function:
    fn = program.get_function("kernel")
    assert fn is not None
    assert fn.ir_stage == ir.FunctionIRStage.Buffer
    return fn


def _realize(program: ir.Program, policy: str, disjoint_gm: bool) -> ir.Function:
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy=policy,
        autosync_disjoint_gm=disjoint_gm,
    ):
        realized = passes.realize_sync()(program)
    fn = realized.get_function("kernel")
    assert fn is not None
    assert dict(fn.attrs)[PYPTO_AUTOSYNC_ATTR] is True
    return fn


def _insert_optimize_lower(program: ir.Program, policy: str, disjoint_gm: bool) -> ir.Function:
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy=policy,
        autosync_disjoint_gm=disjoint_gm,
    ):
        logical = passes.insert_sync()(program)
        optimized = passes.optimize_sync()(logical)
        physical = passes.lower_sync()(optimized)
    fn = physical.get_function("kernel")
    assert fn is not None
    assert dict(fn.attrs)[PYPTO_AUTOSYNC_ATTR] is True
    return fn


def _row_expand_program(broadcast: str, *, loop_visits: int = 2, store_inside: bool = True) -> ir.Program:
    if store_inside:
        return pl.parse_program(f"""
@pl.program
class RowExpandKernel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[32, 64], pl.FP32]]) -> pl.Tensor[[32, 64], pl.FP32]:
        for _ in pl.range({loop_visits}):
            source = pl.load(a, [0, 0], [16, 64])
            scratch = pl.create_tile([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            column = pl.tile.row_min(source, scratch)
            result = pl.tile.row_expand_{broadcast}(source, column)
            out = pl.store(result, [0, 0], out)
            out = pl.store(source, [16, 0], out)
        return out
""")
    return pl.parse_program(f"""
@pl.program
class RowExpandChildKernel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        initial = pl.load(a, [0, 0], [16, 64])
        scratch = pl.create_tile([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        column = pl.tile.row_min(initial, scratch)
        acc = pl.tile.full([16, 64], pl.FP32, 0.0)
        for _ in pl.range({loop_visits}):
            source = pl.load(a, [0, 0], [16, 64])
            step = pl.tile.row_expand_{broadcast}(source, column)
            acc = pl.tile.add(acc, step)
        return pl.store(acc, [0, 0], out)
""")


def _transpose_program(*, loop_visits: int = 2, store_inside: bool = True) -> ir.Program:
    if store_inside:
        return pl.parse_program(f"""
@pl.program
class TransposeKernel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32],
               out: pl.InOut[pl.Tensor[[32, 16], pl.FP32]],
               old: pl.InOut[pl.Tensor[[16, 32], pl.FP32]]) -> tuple[
                   pl.Tensor[[32, 16], pl.FP32], pl.Tensor[[16, 32], pl.FP32]]:
        for _ in pl.range({loop_visits}):
            source = pl.load(a, [0, 0], [16, 32])
            result = pl.tile.transpose(source, -2, -1)
            out = pl.store(result, [0, 0], out)
            old = pl.store(source, [0, 0], old)
        return out, old
""")
    return pl.parse_program(f"""
@pl.program
class TransposeChildKernel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32],
               out: pl.InOut[pl.Tensor[[32, 16], pl.FP32]]) -> pl.Tensor[[32, 16], pl.FP32]:
        acc = pl.tile.full([32, 16], pl.FP32, 0.0)
        for _ in pl.range({loop_visits}):
            source = pl.load(a, [0, 0], [16, 32])
            result = pl.tile.transpose(source, -2, -1)
            acc = pl.tile.add(acc, result)
        return pl.store(acc, [0, 0], out)
""")


def _gather_compare_program(
    dtype: str = "FP32",
    mode: str = "gt",
    *,
    reshape_workspace: bool = False,
    loop_visits: int = 2,
    store_inside: bool = True,
) -> ir.Program:
    threshold = "1" if dtype == "INT32" else "1.0"
    workspace = (
        "backing = pl.create_tile([32, 256], dtype=pl.UINT8)\n"
        "            scratch = pl.tile.reshape(backing, [1, 8192])"
        if reshape_workspace
        else "scratch = pl.create_tile([1, 8192], dtype=pl.UINT8)"
    )
    if store_inside:
        return pl.parse_program(f"""
@pl.program
class GatherCompareKernel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[8, 128], pl.{dtype}],
               index_out: pl.InOut[pl.Tensor[[8, 128], pl.INT32]],
               count_out: pl.InOut[pl.Tensor[[1, 8], pl.INT32]]) -> tuple[
                   pl.Tensor[[8, 128], pl.INT32], pl.Tensor[[1, 8], pl.INT32]]:
        for _ in pl.range({loop_visits}):
            source = pl.load(a, [0, 0], [8, 128])
            {workspace}
            indices, counts = pl.tile.gather_compare(
                source, pl.const({threshold}, pl.{dtype}), scratch, cmp_mode="{mode}", out_cols=128, offset=7
            )
            index_out = pl.store(indices, [0, 0], index_out)
            count_out = pl.store(counts, [0, 0], count_out)
        return index_out, count_out
""")
    return pl.parse_program(f"""
@pl.program
class GatherCompareChildKernel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[8, 128], pl.{dtype}],
               index_out: pl.InOut[pl.Tensor[[8, 128], pl.INT32]],
               count_out: pl.InOut[pl.Tensor[[1, 8], pl.INT32]]) -> tuple[
                   pl.Tensor[[8, 128], pl.INT32], pl.Tensor[[1, 8], pl.INT32]]:
        acc_idx = pl.tile.full([8, 128], pl.INT32, 0)
        acc_cnt = pl.tile.full([1, 8], pl.INT32, 0)
        for _ in pl.range({loop_visits}):
            source = pl.load(a, [0, 0], [8, 128])
            {workspace}
            indices, counts = pl.tile.gather_compare(
                source, pl.const({threshold}, pl.{dtype}), scratch, cmp_mode="{mode}", out_cols=128, offset=7
            )
            acc_idx = pl.tile.add(acc_idx, indices)
            acc_cnt = pl.tile.add(acc_cnt, counts)
        index_out = pl.store(acc_idx, [0, 0], index_out)
        count_out = pl.store(acc_cnt, [0, 0], count_out)
        return index_out, count_out
""")


def _wrap_inner_loop_in_while(function: ir.Function) -> tuple[ir.Function, ir.ForStmt]:
    """Wrap the inner ForStmt inside a dynamic WhileStmt so EventfulSyncChild is exercised."""
    assert isinstance(function.body, ir.SeqStmts)
    stmts = list(function.body.stmts)
    for_idx = next(i for i, s in enumerate(stmts) if isinstance(s, ir.ForStmt))
    inner = stmts[for_idx]
    assert isinstance(inner, ir.ForStmt)
    span = function.span
    index = ir.DataType.INDEX
    limit = ir.Var("outer_trips", ir.ScalarType(index), span)
    counter = ir.IterArg("outer_iter", ir.ScalarType(index), ir.ConstInt(0, index, span), span)
    final_counter = ir.Var("outer_final", ir.ScalarType(index), span)
    cond = ir.Lt(counter, limit, ir.DataType.BOOL, span)
    next_counter = ir.Add(counter, ir.ConstInt(1, index, span), index, span)
    while_stmt = ir.WhileStmt(
        cond, [counter], ir.SeqStmts([inner, ir.YieldStmt([next_counter], span)], span), [final_counter], span
    )
    new_body = ir.SeqStmts([*stmts[:for_idx], while_stmt, *stmts[for_idx + 1 :]], span)
    wrapped = ir.Function(
        function.name,
        [*zip(function.params, function.param_directions), (limit, ir.ParamDirection.In)],
        function.return_types,
        new_body,
        span,
        function.func_type,
        ir_stage=function.ir_stage,
    )
    return wrapped, inner


def _verify_emitted(
    original: ir.Function, realized: ir.Function, lowered: ir.Function, disjoint_gm: bool
) -> None:
    verify(realized)
    verify(lowered)
    if disjoint_gm:
        run_physical(original, realized, range(6))
        run_physical(original, lowered, range(6))
    else:
        for fn in (realized, lowered):
            text = codegen.PTOCodegen().generate(ir.Program([fn], "Owned", fn.span), emit_tile_addr=True)
            assert "pto.set_flag" in text and "pto.wait_flag" in text


@pytest.mark.parametrize("broadcast", ["add", "max", "min"])
def test_row_expand_variants_contract_and_implicit_scratch(broadcast):
    function, calls, _ = independent_broadcasts((broadcast,))
    info = testing.describe_buffer_sync_operation(calls[0], function.func_type)
    assert [item["data"] for item in info["arguments"]] == ["read", "read", "write"]
    assert info["implicit_effects"] == [
        {
            "space": ir.MemorySpace.Vec,
            "begin": SCRATCH[0],
            "end": SCRATCH[1],
            "data": "read_write",
            "provenance": "a2a3-row-broadcast-mode1-implicit-scratch",
            "exact_footprint": False,
        }
    ]

    aliased, _, _ = independent_broadcasts((broadcast,), addresses={"x0": SCRATCH[0]})
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        realized = passes.realize_sync()(ir.Program([aliased], "AliasedScratch", aliased.span))
    preserved = next(iter(realized.functions.values()))
    assert not dict(preserved.attrs).get(PYPTO_AUTOSYNC_ATTR, False)
    assert (
        "Native implicit workspace separation is unproved"
        in dict(preserved.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    )


@pytest.mark.parametrize("broadcast", ["add", "max", "min"])
@pytest.mark.parametrize("policy,disjoint_gm", _POLICIES)
def test_row_expand_variants_realize_and_lower(broadcast, policy, disjoint_gm):
    program = _lower_to_buffer(_row_expand_program(broadcast))
    original = _kernel(program)
    if disjoint_gm:
        retained = testing.analyze_placed_sync(original, True)
        assert retained.describe()["regions"]
        validate(original, retained)
    realized = _realize(program, policy, disjoint_gm)
    lowered = _insert_optimize_lower(program, policy, disjoint_gm)
    _verify_emitted(original, realized, lowered, disjoint_gm)


@pytest.mark.parametrize("broadcast", ["add", "max", "min"])
def test_row_expand_variants_eventful_child_promotion(broadcast):
    program = _lower_to_buffer(_row_expand_program(broadcast, loop_visits=3, store_inside=False))
    wrapped, scope = _wrap_inner_loop_in_while(_kernel(program))
    with passes.PassContext(
        [], memory_planner=passes.MemoryPlanner.PYPTO, autosync_policy="conservative_prefix"
    ):
        analysis = testing.analyze_conservative_sync(
            wrapped, automatic_children=True, retain_incoming_diagnostics=False
        )
    attempts = analysis["child_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["scope"] is scope
    assert attempts[0]["stage"] == "logical_export_ready"

    wrapped_program = ir.Program([wrapped], "RowExpandChild", wrapped.span)
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        logical = passes.insert_sync()(wrapped_program)
        optimized = passes.optimize_sync()(logical)
        physical = passes.lower_sync()(optimized)
        realized = passes.realize_sync()(wrapped_program)
    logical_fn = logical.get_function("kernel")
    physical_fn = physical.get_function("kernel")
    realized_fn = realized.get_function("kernel")
    assert logical_fn is not None and physical_fn is not None and realized_fn is not None
    assert dict(logical_fn.attrs)["autosync_retained_child_count"] == 1
    assert dict(physical_fn.attrs)[PYPTO_AUTOSYNC_ATTR] is True
    assert dict(realized_fn.attrs)[PYPTO_AUTOSYNC_ATTR] is True


@pytest.mark.parametrize("policy,disjoint_gm", _POLICIES)
def test_transpose_realize_and_lower(policy, disjoint_gm):
    program = _lower_to_buffer(_transpose_program())
    original = _kernel(program)
    if disjoint_gm:
        retained = testing.analyze_placed_sync(original, True)
        validate(original, retained)
        sites = [
            site
            for region in retained.describe()["regions"]
            for site in region["sites"]
            if site["original"].op.name == ir.get_op("buffer.transpose").name
        ]
        assert len(sites) == 1
        assert sites[0]["access_precision"] == ["exact", "enclosure", "exact"]
    realized = _realize(program, policy, disjoint_gm)
    lowered = _insert_optimize_lower(program, policy, disjoint_gm)
    _verify_emitted(original, realized, lowered, disjoint_gm)


def test_transpose_eventful_child_analysis_succeeds():
    program = _lower_to_buffer(_transpose_program(loop_visits=3, store_inside=False))
    wrapped, scope = _wrap_inner_loop_in_while(_kernel(program))
    with passes.PassContext(
        [], memory_planner=passes.MemoryPlanner.PYPTO, autosync_policy="conservative_prefix"
    ):
        analysis = testing.analyze_conservative_sync(
            wrapped, automatic_children=True, retain_incoming_diagnostics=False
        )
    attempts = analysis["child_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["scope"] is scope
    # Placed sync import + exact compact analysis succeed; qualification declines
    # only because TTRANS owns V<->S ID0.
    assert attempts[0]["mathematics"] is not None
    assert attempts[0]["mathematics"].describe()["regions"]
    assert attempts[0]["stage"] == "qualification"
    assert "owns flags" in attempts[0]["reason"]

    wrapped_program = ir.Program([wrapped], "TransposeChild", wrapped.span)
    realized = _realize(wrapped_program, "conservative_prefix", False)
    lowered = _insert_optimize_lower(wrapped_program, "conservative_prefix", False)
    assert dict(realized.attrs)[PYPTO_AUTOSYNC_ATTR] is True
    assert dict(lowered.attrs)[PYPTO_AUTOSYNC_ATTR] is True


@pytest.mark.parametrize(
    "dtype,mode,reshape_workspace",
    [
        ("FP32", "gt", False),
        ("FP32", "gt", True),
        ("FP16", "eq", False),
        ("INT32", "eq", False),
    ],
)
@pytest.mark.parametrize("policy,disjoint_gm", _POLICIES)
def test_gather_compare_realize_and_lower(reshape_workspace, dtype, mode, policy, disjoint_gm):
    program = _lower_to_buffer(_gather_compare_program(dtype, mode, reshape_workspace=reshape_workspace))
    original = _kernel(program)
    if disjoint_gm:
        retained = testing.analyze_placed_sync(original, True)
        validate(original, retained)
        sites = [
            site
            for region in retained.describe()["regions"]
            for site in region["sites"]
            if site["original"].op.name == ir.get_op("buffer.gather_compare").name
        ]
        assert len(sites) == 1
        assert sites[0]["access_precision"] == ["exact", "enclosure", "enclosure", "exact", "exact"]
    realized = _realize(program, policy, disjoint_gm)
    lowered = _insert_optimize_lower(program, policy, disjoint_gm)
    _verify_emitted(original, realized, lowered, disjoint_gm)


@pytest.mark.parametrize("reshape_workspace", [False, True])
def test_gather_compare_eventful_child_analysis_succeeds(reshape_workspace):
    program = _lower_to_buffer(
        _gather_compare_program(reshape_workspace=reshape_workspace, loop_visits=3, store_inside=True)
    )
    wrapped, scope = _wrap_inner_loop_in_while(_kernel(program))
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        analysis = testing.analyze_conservative_sync(
            wrapped, automatic_children=True, retain_incoming_diagnostics=False
        )
    attempts = analysis["child_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["scope"] is scope
    # Placed sync import + exact compact analysis succeed (including reshaping the
    # 1-row mask prefix); qualification declines only because TGATHER owns V<->S ID0.
    assert attempts[0]["mathematics"] is not None
    assert attempts[0]["mathematics"].describe()["regions"]
    assert attempts[0]["stage"] == "qualification"
    assert "owns flags" in attempts[0]["reason"]

    wrapped_program = ir.Program([wrapped], "GatherCompareChild", wrapped.span)
    realized = _realize(wrapped_program, "conservative_prefix", True)
    lowered = _insert_optimize_lower(wrapped_program, "conservative_prefix", True)
    assert dict(realized.attrs)[PYPTO_AUTOSYNC_ATTR] is True
    assert dict(lowered.attrs)[PYPTO_AUTOSYNC_ATTR] is True


def test_noncontiguous_multirow_reshape_remains_rejected_in_exact_import():
    span = ir.Span.unknown()
    index = ir.DataType.INDEX

    def ci(val):
        return ir.ConstInt(val, index, span)

    def tup(vals):
        return ir.MakeTuple([ci(v) for v in vals], span)

    parent_ty = ir.BufferType([4, 16], ir.DataType.FP32, ir.MemorySpace.Vec)
    subview_ty = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    reshaped_ty = ir.BufferType([1, 16], ir.DataType.FP32, ir.MemorySpace.Vec)
    parent = ir.Var("parent", parent_ty, span)
    subview = ir.Var("subview", subview_ty, span)
    reshaped = ir.Var("reshaped", reshaped_ty, span)
    out = ir.Var("out", ir.TensorType([1, 16], ir.DataType.FP32), span)
    fn = ir.Function(
        "kernel",
        [(out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    parent,
                    ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], parent_ty, span),
                    span,
                ),
                ir.AssignStmt(
                    subview,
                    ir.Call(ir.get_op("buffer.subview"), [parent, tup([0, 0])], subview_ty, span),
                    span,
                ),
                ir.AssignStmt(
                    reshaped,
                    ir.Call(ir.get_op("buffer.reshape"), [subview], reshaped_ty, span),
                    span,
                ),
                ir.EvalStmt(
                    ir.Call(
                        ir.get_op("buffer.store"),
                        [reshaped, tup([0, 0]), tup([1, 16]), out],
                        ir.VoidType(),
                        span,
                    ),
                    span,
                ),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="cannot reshape a noncontiguous window"):
        testing.analyze_placed_sync(fn, True)


def test_gather_compare_unready_scalar_threshold_rejected_in_exact_import():
    program = _lower_to_buffer(_gather_compare_program("INT32", "eq", loop_visits=1))
    original = _kernel(program)
    span = original.span
    index = ir.DataType.INDEX
    scalar_buf_ty = ir.BufferType([1, 8], ir.DataType.INT32, ir.MemorySpace.Vec)
    scalar_buf = ir.Var("threshold_buf", scalar_buf_ty, span)
    threshold = ir.Var("runtime_threshold", ir.ScalarType(ir.DataType.INT32), span)
    alloc_stmt = ir.AssignStmt(
        scalar_buf,
        ir.Call(
            ir.get_op("buffer.alloc"),
            [ir.MakeTuple([], span), ir.ConstInt(65536, index, span)],
            scalar_buf_ty,
            span,
        ),
        span,
    )
    read_stmt = ir.AssignStmt(
        threshold,
        ir.Call(
            ir.get_op("buffer.read"),
            [scalar_buf, ir.MakeTuple([ir.ConstInt(0, index, span), ir.ConstInt(0, index, span)], span)],
            threshold.type,
            span,
        ),
        span,
    )

    class ReplaceThreshold(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.gather_compare").name:
                args = list(op.args)
                args[1] = threshold
                return ir.Call(op.op, args, dict(op.kwargs), None, op.type, op.span)
            return super().visit_call(op)

    mutated = ReplaceThreshold().visit_function(original)
    assert isinstance(mutated.body, ir.SeqStmts)
    mutated = ir.Function(
        mutated.name,
        [*zip(mutated.params, mutated.param_directions)],
        mutated.return_types,
        ir.SeqStmts([alloc_stmt, read_stmt, *mutated.body.stmts], span),
        span,
        mutated.func_type,
        ir_stage=mutated.ir_stage,
    )
    with pytest.raises(ValueError, match="Unavailable affine synchronization access"):
        testing.analyze_placed_sync(mutated, True)
