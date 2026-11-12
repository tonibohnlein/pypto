# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller evidence is re-established from the actual serialized token program."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_cases import capture
from .sync_matmul_oracle import hardware_reference, interpret, validate
from .sync_physical_oracle import PhysicalExecution, run_physical
from .sync_prefill_cases import capture_prefill
from .test_nested_sync_repetition import resident_pipeline_tree
from .test_scoped_logical_sync import dynamic_collective
from .test_scoped_token_families import Uses
from .test_token_sync_lowering import _apply, _project

pytestmark = pytest.mark.usefixtures("ascend_backend")


def apply(factory, program):
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        return factory()(program)


def worker(program, name):
    return next(f for f in program.functions.values() if f.name == name)


def check(original, logical, physical, parameters, *, block=None, minimum_protocol=True):
    _, calls = interpret(original, parameters, block_index=block)
    reference = hardware_reference(original, calls)
    tokens = Uses(calls, reference[0])
    flags = PhysicalExecution(calls, reference[0], range(6))
    for function, state in [(logical, tokens), (physical, flags)]:
        environment = {function.params[i]: v for i, v in parameters.items()}
        if block is not None:
            environment["__block_index__"] = block
        state.execute(function.body, environment)
    _, expected = _project(tokens.vertices, tokens.edges, reference[0])
    graph, actual = _project(flags.vertices, flags.edges, reference[0])
    assert actual == expected
    assert reference[2] <= actual
    assert tokens.payload_count == flags.payload_count == len(calls)
    assert not flags.live and not any(flags.pending.values())
    assert len(flags.records) == len(flags.waits) == sum(tokens.uses.values())
    assert all(pair in graph for pair in flags.reuses)
    assert all(ids <= set(range(6)) for ids in flags.ids.values())
    if minimum_protocol:
        run_physical(original, physical, range(6), parameters, block_index=block)


def callers(function, starts=(0, 8), *, unknown=False, blocks=None):
    span = function.span

    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, span)

    params = [ir.Var(p.name_hint + "_argument", p.type, span) for p in function.params[:-1]]
    unresolved = ir.Var("unresolved", function.params[-1].type, span)
    calls = [
        ir.EvalStmt(
            ir.Call(ir.GlobalVar(function.name), [*params, ci(n)], function.return_types[0], span), span
        )
        for n in starts
    ]
    if unknown:
        calls.append(
            ir.EvalStmt(
                ir.Call(ir.GlobalVar(function.name), [*params, unresolved], function.return_types[0], span),
                span,
            )
        )
    caller = ir.Function(
        "main",
        [*params, *([unresolved] if unknown else [])],
        [],
        ir.SeqStmts([*calls, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )
    functions = [function, caller]
    if blocks is not None:
        wrapper = ir.Function(
            "launch",
            list(zip(function.params, function.param_directions)),
            function.return_types,
            ir.EvalStmt(
                ir.Call(ir.GlobalVar(function.name), function.params, function.return_types[0], span), span
            ),
            span,
            ir.FunctionType.Spmd,
            attrs={"core_num": ci(blocks)},
        )

        class Redirect(ir.IRMutator):
            def visit_call(self, op):
                if isinstance(op.op, ir.GlobalVar) and op.op.name == function.name:
                    return ir.Call(ir.GlobalVar(wrapper.name), op.args, op.type, op.span)
                return super().visit_call(op)

        functions = [function, wrapper, Redirect().visit_function(caller)]
    return ir.Program(functions, "CallerEvidence", span)


@pytest.mark.parametrize("name", ["matmul", "gemm"])
def test_original_pypto_lib_token_pipeline(name, tmp_path):
    program = capture_prefill(name)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    inserted = apply(passes.insert_sync, program)
    path = tmp_path / "logical.msgpack"
    ir.serialize_to_file(inserted, str(path))
    restored = ir.deserialize_from_file(str(path))
    assert isinstance(restored, ir.Program)
    logical = apply(passes.optimize_sync, restored)
    physical = apply(passes.lower_sync, logical)
    bounds = testing.sync_parameter_bounds(program)[original.name]
    retained = testing.analyze_placed_sync(original, True, parameter_bounds=bounds)
    for row, col in [(0, 0), (64, 128), (192, 192)]:
        validate(original, retained, parameters={1: row, 3: col})
        check(original, worker(logical, original.name), worker(physical, original.name), {1: row, 3: col})
    source = tmp_path / "kernel.pto"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([worker(physical, original.name)], "Physical", original.span)
        )
    )
    _run_ptoas(
        str(source),
        str(tmp_path / "kernel.cpp"),
        ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"],
    )
    assert (tmp_path / "kernel.cpp").stat().st_size > 0
    # Removing callers must remove their authority, even after successful insertion.
    alone = ir.Program([worker(logical, original.name)], "Alone", original.span)
    with pytest.raises(ValueError, match="established caller range"):
        apply(passes.lower_sync, alone)


@pytest.mark.parametrize("starts", [(0, 8), (9, 12)])
@pytest.mark.parametrize("blocks", [None, 20])
def test_original_dynamic_zero_or_singleton_frame(starts, blocks):
    original = dynamic_collective()
    program = callers(original, starts, blocks=blocks)
    logical = apply(passes.insert_sync, program)
    physical = apply(passes.lower_sync, logical)
    for start in range(starts[0], starts[-1] + 1):
        check(
            original,
            worker(logical, original.name),
            worker(physical, original.name),
            {2: start},
            block=7 if blocks else None,
        )
    # No numerical-zero placeholder may replace the original dynamic start.
    for result in [logical, physical]:
        text = ir.python_print(worker(result, original.name), format=False)
        assert "range(start, 9, 10" in text


@pytest.mark.parametrize("unknown,starts", [(True, (0, 8)), (False, (0, 9))])
def test_changed_caller_contract_refuses_atomically(unknown, starts):
    original = dynamic_collective()
    logical = apply(passes.insert_sync, callers(original))
    program = callers(worker(logical, original.name), starts, unknown=unknown)
    snapshot = ir.serialize(program)
    with pytest.raises(ValueError, match="established caller range|guarded regional activation"):
        apply(passes.lower_sync, program)
    assert ir.serialize(program) == snapshot


def test_actual_duplicate_uses_survive_caller_transport():
    original = dynamic_collective()
    logical = apply(passes.insert_sync, callers(original))

    class Duplicate(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                return ir.SeqStmts([op, op], op.span)
            return op

    edited = Duplicate().visit_function(worker(logical, original.name))
    modified = callers(edited)
    lowered = apply(passes.lower_sync, modified)
    for start in (0, 4, 8):
        check(original, edited, worker(lowered, original.name), {2: start}, minimum_protocol=False)


def test_dynamic_coordinate_guard_does_not_use_placeholder_zero():
    original = dynamic_collective()
    logical = worker(apply(passes.insert_sync, callers(original)), original.name)
    assert isinstance(logical.body, ir.SeqStmts)
    outer = logical.body.stmts[-2]
    assert isinstance(outer, ir.ForStmt)

    class Guard(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                condition = ir.Eq(
                    outer.loop_var, ir.ConstInt(0, ir.DataType.INDEX, op.span), ir.DataType.BOOL, op.span
                )
                return ir.IfStmt(condition, op, None, [], op.span)
            return op

    modified = callers(Guard().visit_function(logical))
    with pytest.raises(ValueError, match="dynamic original frame"):
        apply(passes.lower_sync, modified)


@pytest.mark.parametrize("blocks", [8, 10])
def test_block_dependent_start_requires_uniform_activation(blocks):
    original = resident_pipeline_tree(2, 2, 1)
    span = original.span
    parameter = ir.Var("unused_start", ir.ScalarType(ir.DataType.INDEX), span)
    block_var = ir.Var("block", parameter.type, span)
    assert isinstance(original.body, ir.SeqStmts)
    outer = original.body.stmts[-2]
    assert isinstance(outer, ir.ForStmt)
    dynamic = ir.ForStmt(
        outer.loop_var,
        block_var,
        ir.ConstInt(9, ir.DataType.INDEX, span),
        ir.ConstInt(10, ir.DataType.INDEX, span),
        [],
        outer.body,
        [],
        span,
    )
    definition = ir.AssignStmt(
        block_var, ir.Call(ir.get_op("buffer.get_block_idx"), [], block_var.type, span), span
    )
    function = ir.Function(
        original.name,
        [*zip(original.params, original.param_directions), parameter],
        original.return_types,
        ir.SeqStmts([definition, *original.body.stmts[:-2], dynamic, original.body.stmts[-1]], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    # The unused scalar parameter remains qualified; the block start now needs
    # its own explicit launch range. Eight blocks are all active; ten straddle.
    program = callers(function, blocks=blocks)
    if blocks == 10:
        with pytest.raises(ValueError, match="guarded regional activation"):
            apply(passes.insert_sync, program)
    else:
        logical = apply(passes.insert_sync, program)
        physical = apply(passes.lower_sync, logical)
        for block in (0, 4, 7):
            check(
                function,
                worker(logical, function.name),
                worker(physical, function.name),
                {len(function.params) - 1: 0},
                block=block,
            )


def test_parameterized_gemm_token_size_is_trip_compact():
    counts = []
    for trips in (4, 8, 64, 1000000):
        program = capture_prefill("gemm", trips)
        logical = apply(passes.insert_sync, program)
        physical = apply(passes.lower_sync, logical)
        summary = []

        class Count(ir.IRVisitor):
            def visit_call(self, op):
                if op.op.name in {
                    ir.get_op(n).name
                    for n in ("sync.record", "sync.wait", "system.set_flag", "system.wait_flag")
                }:
                    summary.append(op.op.name)
                super().visit_call(op)

        for result in (logical, physical):
            for function in result.functions.values():
                if function.func_type == ir.FunctionType.AIC:
                    Count().visit_function(function)
        counts.append(sorted(summary))
    assert all(count == counts[0] for count in counts)


def test_multifunction_refusal_preserves_whole_program():
    finite = _apply(passes.insert_sync, capture("matmul_64"))
    original = dynamic_collective()
    qualified = apply(passes.insert_sync, callers(original))
    unqualified = callers(worker(qualified, original.name), unknown=True)
    program = ir.Program([finite, *unqualified.functions.values()], "AtomicLowering", original.span)
    snapshot = ir.serialize(program)
    with pytest.raises(ValueError, match="established caller range"):
        apply(passes.lower_sync, program)
    assert ir.serialize(program) == snapshot


def test_caller_reactivation_does_not_regenerate_missing_tokens():
    original = dynamic_collective()
    absent = apply(passes.insert_sync, callers(original, (9, 12)))
    supplied = worker(absent, original.name)
    # Editing callers can invalidate demand coverage of already inserted IR.
    # LowerSync is a token interpreter, not a fresh dependence analysis.
    active = callers(supplied, (0, 8))
    lowered = worker(apply(passes.lower_sync, active), original.name)
    _, calls = interpret(original, {2: 0})
    reference = hardware_reference(original, calls)
    flags = PhysicalExecution(calls, reference[0], range(6))
    flags.execute(lowered.body, {lowered.params[2]: 0})
    _, actual = _project(flags.vertices, flags.edges, reference[0])
    assert reference[2] - actual
    assert not dict(lowered.attrs).get("pypto_autosync", False)
