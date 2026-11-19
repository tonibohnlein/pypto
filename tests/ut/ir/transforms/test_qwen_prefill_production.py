# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Original Qwen bodies through the ordinary opt-in whole-program pass."""

import ast
import inspect

import pytest
from pypto import codegen, ir, passes, testing
from pypto import language as pl
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.runtime import RunConfig

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import sequence_stmts
from .sync_physical_oracle import run_physical
from .sync_qwen_prefill_cases import FIXTURES, capture_qwen_projection, control_calls, source_body_ast

pytestmark = pytest.mark.usefixtures("ascend_backend")


def realize(program, disjoint=True, policy="exact"):
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=disjoint,
        autosync_policy=policy,
    ):
        return passes.realize_sync()(program)


MLP_M_TILE, MLP_BAND_WIDTH, SILU_OUT_CHUNK, SILU_SPMD_BLOCKS = 128, 4352, 64, 24


@pl.jit(auto_scope=False)
def silu_prefill(
    mlp_silu_b: pl.Out[pl.Tensor[[MLP_M_TILE, MLP_BAND_WIDTH], pl.BF16]],
    gate_up_acc_b: pl.Tensor[[2 * MLP_M_TILE, MLP_BAND_WIDTH], pl.FP32],
) -> pl.Tensor[[MLP_M_TILE, MLP_BAND_WIDTH], pl.BF16]:
    silu_band_blocks = MLP_BAND_WIDTH // SILU_OUT_CHUNK
    with pl.manual_scope():
        with pl.spmd(SILU_SPMD_BLOCKS, name_hint="silu_spmd"):
            silu_core = pl.tile.get_block_idx()
            for rel_sb in pl.range(silu_core, silu_band_blocks, SILU_SPMD_BLOCKS):
                so0 = rel_sb * SILU_OUT_CHUNK
                silu_gate = pl.slice(gate_up_acc_b, [MLP_M_TILE, SILU_OUT_CHUNK], [0, so0])
                silu_up = pl.slice(gate_up_acc_b, [MLP_M_TILE, SILU_OUT_CHUNK], [MLP_M_TILE, so0])
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(silu_gate)), 1.0))
                mlp_chunk = pl.mul(pl.mul(silu_gate, sigmoid), silu_up)
                mlp_silu_b = pl.assemble(mlp_silu_b, pl.cast(mlp_chunk, target_type=pl.BF16), [0, so0])
    return mlp_silu_b


@pytest.mark.parametrize("stop", [68, 44])
def test_silu_prefill_uses_original_loop_boundaries(stop):
    def launch_body(source):
        return next(
            source_body_ast(ast.Module(body=node.body, type_ignores=[]))
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.With)
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
            for keyword in item.context_expr.keywords
            if keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "silu_spmd"
        )

    assert launch_body(inspect.getsource(silu_prefill._func)) == launch_body(
        (FIXTURES / "prefill_fwd.py.source").read_text()
    )
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        original = silu_prefill.lower(config=RunConfig(arch=BackendType.Ascend910))

    class LimitVisits(ir.IRMutator):
        def visit_for_stmt(self, op):
            # The standalone launch has one loop. A stop of 44 retains the
            # placed payload but gives the last four workers a single visit.
            assert isinstance(op.stop, ir.ConstInt) and op.stop.value == 68
            return ir.ForStmt(
                op.loop_var,
                op.start,
                ir.ConstInt(stop, ir.DataType.INDEX, op.span),
                op.step,
                op.iter_args,
                op.body,
                op.return_vars,
                op.span,
                op.kind,
                op.attrs,
                op.hardware_loop_policy,
            )

    if stop != 68:
        original = LimitVisits().visit_program(original)
    physical = realize(original, policy="conservative_prefix")
    worker = next(f for f in physical.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert dict(worker.attrs)[PYPTO_AUTOSYNC_ATTR], dict(worker.attrs)
    pto = codegen.PTOCodegen().generate(ir.Program([worker], "SiLU", worker.span), emit_tile_addr=True)
    assert pto.count("scf.if") == 4
    assert "arith.cmpi ne," in pto and "arith.cmpi slt," in pto
    assert "arith.floordivsi" not in pto

    def flag(kind, source, target, event=0):
        return (ir.get_op(f"system.{kind}_flag").name, int(source), int(target), event)

    barrier = ir.get_op("system.bar_v").name
    load, neg, exp, adds, recip, mul, cast, store = (
        ir.get_op(f"buffer.{name}").name
        for name in ("load", "neg", "exp", "adds", "recip", "mul", "cast", "store")
    )
    v, mte2, mte3 = ir.PipeType.V, ir.PipeType.MTE2, ir.PipeType.MTE3
    for block in range(SILU_SPMD_BLOCKS):
        expected = []
        for i in range(block, stop, SILU_SPMD_BLOCKS):
            if i != block:
                expected.append(flag("wait", v, mte2))
            expected += [load, flag("set", mte2, v)]
            if i != block:
                expected.append(flag("wait", mte3, mte2))
            expected += [
                load,
                flag("set", mte2, v, 1),
                flag("wait", mte2, v),
                neg,
                barrier,
                exp,
                barrier,
                adds,
                barrier,
                recip,
                barrier,
                mul,
                barrier,
                flag("wait", mte2, v, 1),
                mul,
                barrier,
                cast,
            ]
            has_next = i + SILU_SPMD_BLOCKS < stop
            if has_next:
                expected.append(flag("set", v, mte2))
            expected += [flag("set", v, mte3), flag("wait", v, mte3), store]
            if has_next:
                expected.append(flag("set", mte3, mte2))
        expected.append(ir.get_op("system.bar_all").name)
        actual = [
            (call.op.name, *(call.kwargs[key] for key in ("src_pipe", "dst_pipe", "event_id")))
            if "event_id" in call.kwargs
            else call.op.name
            for call, _ in control_calls(worker, block)
        ]
        assert actual == expected


@pytest.mark.parametrize("name,band", [("output", 3), *(("gate_up", band) for band in range(4))])
def test_full_original_qwen_production_selects_pypto(name, band):
    original = capture_qwen_projection(name, layers=2, band=band)
    result = realize(original)
    workers = [f for f in result.functions.values() if f.func_type == ir.FunctionType.AIC]
    assert len(workers) == 1
    assert dict(workers[0].attrs)[PYPTO_AUTOSYNC_ATTR]
    assert AUTOSYNC_FALLBACK_REASON_ATTR not in dict(workers[0].attrs)


@pytest.mark.parametrize("name,blocks", [("output", [0, 2, 19]), ("gate_up", [0, 6, 13, 23])])
def test_bounded_original_production_commands_preserve_order(name, blocks):
    original = capture_qwen_projection(name, layers=2, band=3, hidden=512)
    worker = next(f for f in original.functions.values() if f.func_type == ir.FunctionType.AIC)
    result = realize(original)
    emitted = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.AIC)
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    for i, block in enumerate(blocks):
        run_physical(worker, emitted, range(6), {3: i % 2}, block_index=block)


@pytest.mark.parametrize("name", ["output", "gate_up"])
def test_missing_alias_or_caller_contract_delegates_unchanged(name):
    configured = capture_qwen_projection(name, layers=2, band=3)
    worker = next(f for f in configured.functions.values() if f.func_type == ir.FunctionType.AIC)
    for program, disjoint in [(configured, False), (ir.Program([worker], "NoCaller", worker.span), True)]:
        result = realize(program, disjoint)
        emitted = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.AIC)
        assert not dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
        assert dict(emitted.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
        ir.assert_structural_equal(worker.body, emitted.body)


@pytest.mark.parametrize("symbolic_start", [False, True])
def test_unused_launch_evidence_preserves_uniform_scalar_loop(symbolic_start):
    original = staged_fixture(2, stages=1, m=64, n=64)
    span, index = original.span, ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index, span)

    parameter = ir.Var("start", ir.ScalarType(index), span)
    body = []
    for statement in sequence_stmts(original.body):
        if isinstance(statement, ir.ForStmt):
            # Exactly one visit under the caller domain. It initializes its own
            # accumulator irrespective of its original starting coordinate.
            payloads = [
                s.then_body if isinstance(s, ir.IfStmt) else s for s in sequence_stmts(statement.body)
            ]
            statement = ir.ForStmt(
                statement.loop_var,
                parameter if symbolic_start else ci(0),
                ci(2),
                ci(2),
                [],
                ir.SeqStmts(payloads, span),
                [],
                span,
            )
        body.append(statement)
    worker = ir.Function(
        original.name,
        [*zip(original.params, original.param_directions), parameter],
        original.return_types,
        ir.SeqStmts(body, span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    params = [ir.Var(p.name_hint + "_forward", p.type, span) for p in worker.params]
    wrapper = ir.Function(
        "launch_wrapper",
        list(zip(params, worker.param_directions)),
        worker.return_types,
        ir.EvalStmt(ir.Call(ir.GlobalVar(worker.name), params, worker.return_types[0], span), span),
        span,
        ir.FunctionType.Spmd,
        attrs={"core_num": ci(20)},
    )
    roots = [ir.Var(p.name_hint + "_input", p.type, span) for p in original.params]
    calls = [
        ir.EvalStmt(
            ir.Call(ir.GlobalVar(wrapper.name), [*roots, ci(start)], worker.return_types[0], span), span
        )
        for start in [0, 1]
    ]
    caller = ir.Function(
        "main",
        roots,
        [],
        ir.SeqStmts([*calls, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )
    program = ir.Program([worker, wrapper, caller], "UniformScalarLaunch", span)
    evidence = testing.sync_invocation_evidence(program)
    assert evidence["block_indices"][worker.name] == (0, 19)
    assert evidence["parameters"][worker.name][3] == (0, 1)
    result = realize(program)
    actual = next(f for f in result.functions.values() if f.name == worker.name)
    assert dict(actual.attrs)[PYPTO_AUTOSYNC_ATTR]
    for start in [0, 1]:
        run_physical(worker, actual, range(6), {3: start}, block_index=7)
