# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Original Qwen bodies through the ordinary opt-in whole-program pass."""

import pytest
from pypto import ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import sequence_stmts
from .sync_physical_oracle import run_physical
from .sync_qwen_prefill_cases import capture_qwen_projection

pytestmark = pytest.mark.usefixtures("ascend_backend")


def realize(program, disjoint=True):
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=disjoint,
    ):
        return passes.realize_sync()(program)


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
