# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Production prefix requests select whole closed invocations through existing providers."""

import pytest
from pypto import ir, passes, testing

from .qualify_captured_child_commands import execution_case, logical_plan, parse_arguments
from .sync_closed_command_oracle import check_closed_commands
from .sync_matmul_oracle import sequence_stmts, validate
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_autosync_switch import clone
from .test_eventful_sync_children import BASES, eventful_fixture, readonly_external_fixture
from .test_logical_sync_realization import verify
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def inspect(function):
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        return testing.analyze_conservative_sync(
            function, automatic_children=True, retain_incoming_diagnostics=False
        )


def construct(function):
    program = ir.Program([function], "AutomaticChildren", function.span)
    before = ir.serialize(program)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
        physical_program = passes.lower_sync()(logical_program)
    assert ir.serialize(program) == before
    logical = next(iter(logical_program.functions.values()))
    physical = next(iter(physical_program.functions.values()))
    verify(logical)
    verify(physical)
    return logical, physical


@pytest.mark.parametrize("count", [0, 1, 3])
def test_public_prefix_retains_original_whole_readonly_invocation(count):
    original, scope, oracle = readonly_external_fixture(count)
    analysis = inspect(original)
    attempts = analysis["child_attempts"]
    ready = [attempt for attempt in attempts if attempt["stage"] == "logical_export_ready"]
    assert len(ready) == 1 and ready[0]["scope"] is scope
    assert ready[0]["physical_qualification"] == "pending_whole_plan"
    retained = ready[0]["mathematics"]
    unavailable = [
        index
        for index, domain in enumerate(retained.describe()["storage"])
        if domain["space"] == ir.MemorySpace.DDR
    ]
    assert unavailable and all(retained.boundary(index, 0, 0) is None for index in unavailable)
    # Actual retained covers, four-kind queries and local selectors are checked
    # independently; projected GM selectors remain unavailable, never empty.
    validate(
        oracle,
        retained,
        parameters={3: 1, 4: True, 5: 0},
        unavailable_boundaries=unavailable,
        readonly_external_projection=True,
    )
    logical, physical = construct(original)
    assert dict(logical.attrs)["autosync_retained_child_count"] == 1
    assert dict(physical.attrs)["pypto_autosync"] is True
    for visits in (0, 1, 3):
        for execute in (False, True):
            for bases in (BASES, {0: 0, 1: 0, 2: 2 << 20}):
                check_closed_commands(
                    logical,
                    physical,
                    {3: visits, 4: execute},
                    range(6),
                    tensor_bases=bases,
                    compare_commands=False,
                    hardware_source=original,
                )


def vector_child(count, operation="buffer.abs"):
    original = vec_fixture()
    statements = list(sequence_stmts(original.body))
    span = original.span
    # Keep initialization and output outside the whole child. Its optional GM
    # load is read-only, while all reused destination storage remains local.
    original_load = statements[3]
    assert isinstance(original_load, ir.EvalStmt) and isinstance(original_load.expr, ir.Call)
    call = original_load.expr
    load = ir.EvalStmt(ir.Call(call.op, call.args, dict(call.kwargs), None, call.type, call.span), span)
    first_compute = statements[5]
    assert isinstance(first_compute, ir.EvalStmt) and isinstance(first_compute.expr, ir.Call)
    source, _, destination = first_compute.expr.args
    compute = ir.EvalStmt(ir.Call(ir.get_op(operation), [source, destination], ir.VoidType(), span), span)
    index = ir.DataType.INDEX
    loop = ir.ForStmt(
        ir.Var("repeat", ir.ScalarType(index), span),
        ir.ConstInt(0, index, span),
        ir.ConstInt(count, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts([load, compute, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    return clone(original, body=ir.SeqStmts([*statements[:6], loop, *statements[6:]], span)), loop


@pytest.mark.parametrize("count", [0, 1, 3])
def test_vector_child_uses_the_same_automatic_provider(count):
    original, loop = vector_child(count)
    analysis = inspect(original)
    assert analysis["child_attempts"][0]["scope"] is loop
    assert analysis["child_attempts"][0]["stage"] == "logical_export_ready"
    logical, physical = construct(original)
    check_closed_commands(logical, physical, {}, range(6), tensor_bases=BASES, compare_commands=False)


def test_failed_native_export_retains_successful_mathematics():
    original, loop = vector_child(3, "buffer.recip")
    attempt = inspect(original)["child_attempts"][0]
    assert attempt["scope"] is loop
    assert attempt["stage"] == "qualification"
    assert "owns flags" in attempt["reason"]
    assert attempt["child"] is None
    assert attempt["mathematics"].describe()["regions"]
    # Failure preserves an explicit conservative request, not an invented
    # resource-free child or a weakened exact result.
    logical, physical = construct(original)
    assert dict(logical.attrs)["autosync_retained_child_count"] == 0
    check_closed_commands(logical, physical, {}, range(6), tensor_bases=BASES, compare_commands=False)


def test_external_write_is_an_effect_refusal_without_invented_alias_premises():
    original = vec_fixture(count=2)
    attempt = inspect(original)["child_attempts"][0]
    assert attempt["stage"] == "effects"
    assert "not uniformly read-only" in attempt["reason"]
    assert attempt["mathematics"] is None and attempt["child"] is None
    logical, _ = construct(original)
    assert dict(logical.attrs)["autosync_retained_child_count"] == 0


def test_automatic_preparation_does_not_unfold_count():
    signatures = []
    for count in (2, 16, 1024):
        original, scope = eventful_fixture(count)
        analysis = inspect(original)
        attempts = analysis["child_attempts"]
        ready = [attempt for attempt in attempts if attempt["stage"] == "logical_export_ready"]
        assert len(ready) == 1 and ready[0]["scope"] is scope
        description = ready[0]["mathematics"].describe()
        signatures.append(
            (
                analysis["cost"]["child_discovery_nodes"],
                analysis["cost"]["child_index_nodes"],
                description["analyzed_occurrences"],
                description["cells"],
                [(len(r["sites"]), len(r["demands"])) for r in description["regions"]],
            )
        )
    assert all(signature == signatures[0] for signature in signatures)


def test_disjoint_children_share_one_signature_index():
    observations = []
    for repetitions in (2, 8, 32):
        original, template = vector_child(2)
        span = original.span
        loops = []
        for number in range(repetitions):
            body = []
            for statement in sequence_stmts(template.body):
                if isinstance(statement, ir.EvalStmt):
                    call = statement.expr
                    assert isinstance(call, ir.Call)
                    body.append(
                        ir.EvalStmt(
                            ir.Call(call.op, call.args, dict(call.kwargs), None, call.type, call.span), span
                        )
                    )
                else:
                    assert isinstance(statement, ir.YieldStmt)
                    body.append(ir.YieldStmt([], span))
            loops.append(
                ir.ForStmt(
                    ir.Var(f"child_{number}", template.loop_var.type, span),
                    template.start,
                    template.stop,
                    template.step,
                    [],
                    ir.SeqStmts(body, span),
                    [],
                    span,
                )
            )
        body = []
        for statement in sequence_stmts(original.body):
            body.extend(loops if statement is template else [statement])
        parameters = list(zip(original.params, original.param_directions))
        parameters.extend(
            (ir.Var(f"unused_{number}", ir.ScalarType(ir.DataType.INDEX), span), ir.ParamDirection.In)
            for number in range(32)
        )
        function = ir.Function(
            original.name,
            parameters,
            original.return_types,
            ir.SeqStmts(body, span),
            span,
            type=original.func_type,
            level=original.level,
            role=original.role,
            attrs=dict(original.attrs),
            ir_stage=original.ir_stage,
        )
        analysis = inspect(function)
        attempts = analysis["child_attempts"]
        assert len(attempts) == repetitions
        assert all(attempt["stage"] == "logical_export_ready" for attempt in attempts)
        assert analysis["cost"]["child_parameter_nodes"] == len(parameters)
        observations.append(
            (
                repetitions,
                analysis["cost"]["child_index_nodes"],
                sum(attempt["binding_definitions"] for attempt in attempts),
            )
        )
    # Shared owner indexing is linear in original statements; per-child work
    # accounts for the descriptors actually exported, not all unused formals.
    assert len({(nodes - observations[0][1]) // (count - 2) for count, nodes, _ in observations[1:]}) == 1
    assert len({definitions // count for count, _, definitions in observations}) == 1


def test_automatic_child_physical_refusal_does_not_mutate_logical_program():
    original, _ = eventful_fixture(3)
    logical, _ = construct(original)
    before = ir.serialize(logical)
    original_before = ir.serialize(original)
    # A bounded resource derivative: the primitive parent protocol fits two
    # IDs, but its reserved roles leave no private palette for this child.
    # That refusal is not a lower bound for other plans or the six-ID pool.
    baseline = testing.insert_conservative_prefixes(original)
    baseline_physical = testing.lower_sync_schedule(baseline, eligible_ids=[0, 1])["function"]
    check_closed_commands(
        baseline,
        baseline_physical,
        {3: 2, 4: True},
        range(2),
        tensor_bases=BASES,
        compare_commands=False,
        hardware_source=original,
    )
    with pytest.raises(ValueError):
        testing.lower_sync_schedule(logical, eligible_ids=[0, 1])
    assert ir.serialize(logical) == before
    assert ir.serialize(original) == original_before


def test_original_qwen_projection_uses_original_leaf_loop_boundaries():
    program = capture_qwen_projection("output", layers=2, band=3, hidden=512)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    analysis = inspect(original)
    assert analysis["child_attempts"]
    assert any(attempt["stage"] == "logical_export_ready" for attempt in analysis["child_attempts"])
    logical, physical = construct(original)
    assert dict(logical.attrs)["autosync_retained_child_count"] > 0
    assert dict(physical.attrs)["pypto_autosync"] is True


@pytest.mark.parametrize("extra", [["--automatic", "--scope", "k"], ["--automatic", "--check-retained"], []])
def test_command_audit_refuses_ambiguous_mode_before_output(tmp_path, extra):
    output = tmp_path / "not_created"
    argv = [
        "--input",
        str(tmp_path / "not_read"),
        "--function",
        "kernel",
        "--parameters",
        "{}",
        "--shape-values",
        "{}",
        "--block-index",
        "0",
        "--output",
        str(output),
        *extra,
    ]
    with pytest.raises(SystemExit) as error:
        parse_arguments(argv)
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("automatic", [False, True])
def test_command_audit_mode_is_explicit_and_does_not_create_output(tmp_path, automatic):
    output = tmp_path / "not_created"
    argv = [
        "--input",
        str(tmp_path / "not_read"),
        "--function",
        "kernel",
        "--parameters",
        "{}",
        "--shape-values",
        "{}",
        "--block-index",
        "0",
        "--output",
        str(output),
    ]
    argv += ["--automatic"] if automatic else ["--scope", "k"]
    args = parse_arguments(argv)
    assert args.automatic == automatic and args.check_retained is False
    assert not output.exists()


def test_historical_physical_audit_does_not_fabricate_a_logical_baseline():
    original, _ = eventful_fixture(1)
    baseline = logical_plan(original)
    candidate = logical_plan(original, automatic=True)
    with passes.PassContext([]):
        old = passes.lower_sync()(baseline)
        new = passes.lower_sync()(candidate)
    summary = execution_case(
        original,
        [
            (None, next(iter(old.functions.values()))),
            (next(iter(candidate.functions.values())), next(iter(new.functions.values()))),
        ],
        {3: 1, 4: True},
        {},
        BASES,
        0,
    )
    assert summary["actual_token_all_event_order_preserved"] is None
    assert summary["actual_token_order_checked_by_arm"] == [False, True]
    payloads = summary["payloads"]
    assert isinstance(payloads, int) and payloads > 0


def test_top_down_region_discovery_and_guarded_exact_if():
    span = ir.Span.unknown()

    def ci(number):
        return ir.ConstInt(number, ir.DataType.INDEX, span)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    # 1. Nested 2-level sequential ForStmt nest inside a dynamic WhileStmt:
    #    top-down discovery must promote the enclosing 2-level ForStmt nest once,
    #    retiring the leaf-only heuristic.
    gm = ir.TensorType([2, 8], ir.DataType.FP32)
    src = ir.Var("src", gm, span)
    out = ir.Var("out", gm, span)
    ub = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    x = ir.Var("x", ub, span)
    y = ir.Var("y", ub, span)
    cond_flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)
    i_var = ir.Var("outer_i", ir.ScalarType(ir.DataType.INDEX), span)
    j_var = ir.Var("inner_j", ir.ScalarType(ir.DataType.INDEX), span)
    inner_loop = ir.ForStmt(
        j_var,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts([payload("buffer.abs", [x, y]), ir.YieldStmt([], span)], span),
        [],
        span,
    )
    outer_exact_nest = ir.ForStmt(
        i_var,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts([inner_loop, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    while_wrapper = ir.WhileStmt(
        cond_flag,
        [],
        ir.SeqStmts([outer_exact_nest, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    mixed_fn = ir.Function(
        "nested_exact_in_while",
        [src, cond_flag, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, call("buffer.alloc", [tup([]), ci(0)], ub), span),
                ir.AssignStmt(y, call("buffer.alloc", [tup([]), ci(128)], ub), span),
                payload("buffer.load", [src, tup([0, 0]), tup([2, 8]), x]),
                while_wrapper,
                payload("buffer.store", [y, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    analysis = inspect(mixed_fn)
    ready = [a for a in analysis["child_attempts"] if a["stage"] == "logical_export_ready"]
    assert len(ready) == 1 and ready[0]["scope"] is outer_exact_nest
    assert [c["scope"] for c in analysis["children"]] == [outer_exact_nest]
    mixed_prog = ir.Program([mixed_fn], "MixedNestedWhile", span)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_policy="conservative_prefix"):
        mixed_logical = passes.insert_sync()(mixed_prog)
        mixed_physical = passes.lower_sync()(mixed_logical)
    assert dict(next(iter(mixed_logical.functions.values())).attrs)["autosync_retained_child_count"] == 1
    assert dict(next(iter(mixed_physical.functions.values())).attrs)["pypto_autosync"] is True

    # 2. Guarded statically-resolved IfStmt inside Exact (`autosync_policy="exact"`):
    #    provably true branch (`i_var < 4` in loop 0..2) must remain 100% Exact and lower cleanly.
    dead_store = payload("buffer.load", [src, tup([0, 0]), tup([2, 8]), y])
    guarded_fn = ir.Function(
        "guarded_exact_if",
        [src, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, call("buffer.alloc", [tup([]), ci(0)], ub), span),
                ir.AssignStmt(y, call("buffer.alloc", [tup([]), ci(128)], ub), span),
                payload("buffer.load", [src, tup([0, 0]), tup([2, 8]), x]),
                ir.ForStmt(
                    i_var,
                    ci(0),
                    ci(2),
                    ci(1),
                    [],
                    ir.SeqStmts(
                        [
                            ir.IfStmt(
                                ir.Lt(i_var, ci(4), ir.DataType.BOOL, span),
                                payload("buffer.abs", [x, y]),
                                dead_store,
                                [],
                                span,
                            ),
                            ir.YieldStmt([], span),
                        ],
                        span,
                    ),
                    [],
                    span,
                ),
                payload("buffer.store", [y, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([guarded_fn], "GuardedExactIf", span)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True, autosync_policy="exact"):
        physical = next(iter(passes.realize_sync()(program).functions.values()))
    assert dict(physical.attrs)["pypto_autosync"] is True
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_disjoint_gm=True, autosync_policy="conservative_prefix"
    ):
        unified_physical = next(iter(passes.realize_sync()(program).functions.values()))
    assert dict(unified_physical.attrs)["pypto_autosync"] is True


def test_parent_scalar_parameter_and_var_alias_promote_disjoint_gm_ddr_store_child():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    ci = lambda v: ir.ConstInt(v, idx, span)
    tup = lambda vals: ir.MakeTuple([ci(v) for v in vals], span)
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)

    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    cond = ir.Var("cond", ir.ScalarType(ir.DataType.BOOL), span)
    n = ir.Var("n", ir.ScalarType(idx), span)
    n_alias = ir.Var("n_alias", ir.ScalarType(idx), span)
    k = ir.Var("k", ir.ScalarType(idx), span)
    x = ir.Var("x", vec_t, span)
    y = ir.Var("y", vec_t, span)

    # Sequential child loop using parent scalar param `n`, `Var`-to-`Var` alias `n_alias = n`,
    # and performing a DDR store under autosync_disjoint_gm=True.
    child = ir.ForStmt(
        k,
        n,
        ci(4),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ir.IfStmt(
                    ir.Gt(n_alias, ci(0), ir.DataType.BOOL, span),
                    ir.SeqStmts([ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span)], span),
                    None,
                    [],
                    span,
                ),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [y, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    # WhileStmt forces MixedCompose + automatic child discovery instead of whole-function Exact.
    while_stmt = ir.WhileStmt(
        cond,
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    worker = ir.Function(
        "worker_ddr_store_child",
        [src, (out, ir.ParamDirection.InOut), cond, n],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(n_alias, n, span),
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                while_stmt,
                child,
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        level=ir.Level.AIV,
        role=ir.Role.SubWorker,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    gv_worker = ir.GlobalVar(worker.name)
    entry_src = ir.Var("e_src", gm_t, span)
    entry_out = ir.Var("e_out", gm_t, span)
    called_res = ir.Var("called_res", out.type, span)
    entry = ir.Function(
        "entry_caller",
        [entry_src, (entry_out, ir.ParamDirection.InOut)],
        [entry_out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    called_res,
                    ir.Call(
                        gv_worker,
                        [entry_src, entry_out, ir.ConstBool(False, span), ci(2)],
                        out.type,
                        span,
                    ),
                    span,
                ),
                ir.ReturnStmt([called_res], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.Opaque,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    prog = ir.Program([worker, entry], "DisjointGmDdrStoreChild", span)

    with passes.PassContext([], autosync_disjoint_gm=True, autosync_policy="conservative_prefix"):
        logical = passes.insert_sync()(prog)
        optimized = passes.optimize_sync()(logical)
    with passes.PassContext([], autosync_disjoint_gm=True):
        physical_3pass = passes.lower_sync()(optimized)
    with passes.PassContext([], autosync_disjoint_gm=False):
        physical_standalone = passes.lower_sync()(optimized)
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_disjoint_gm=True, autosync_policy="conservative_prefix"
    ):
        physical_1pass = passes.realize_sync()(prog)

    log_fn = next(f for g, f in optimized.functions.items() if g.name == "worker_ddr_store_child")
    assert dict(log_fn.attrs)["autosync_retained_child_count"] == 1
    p3 = next(f for g, f in physical_3pass.functions.items() if g.name == "worker_ddr_store_child")
    ps = next(f for g, f in physical_standalone.functions.items() if g.name == "worker_ddr_store_child")
    p1 = next(f for g, f in physical_1pass.functions.items() if g.name == "worker_ddr_store_child")
    assert dict(p3.attrs)["pypto_autosync"] is True
    assert dict(ps.attrs)["pypto_autosync"] is True
    assert dict(p1.attrs)["pypto_autosync"] is True
    ir.assert_structural_equal(p3.body, ps.body)
    for phys in (p3, p1):
        check_closed_commands(
            log_fn,
            phys,
            {2: False, 3: 2},
            range(6),
            tensor_bases={0: 0, 1: 1 << 20},
            compare_commands=False,
            hardware_source=worker,
        )


def test_prescreen_skips_outer_loop_with_while_without_losing_inner_promotion():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    ci = lambda v: ir.ConstInt(v, idx, span)
    tup = lambda vals: ir.MakeTuple([ci(v) for v in vals], span)
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)

    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    cond = ir.Var("cond", ir.ScalarType(ir.DataType.BOOL), span)
    x = ir.Var("x", vec_t, span)
    y = ir.Var("y", vec_t, span)
    outer_i = ir.Var("outer_i", ir.ScalarType(idx), span)
    inner_k = ir.Var("inner_k", ir.ScalarType(idx), span)

    # Eligible inner ForStmt inside a WhileStmt wrapped in an outer ForStmt.
    inner_child = ir.ForStmt(
        inner_k,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    outer_for = ir.ForStmt(
        outer_i,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.WhileStmt(cond, [], ir.SeqStmts([inner_child, ir.YieldStmt([], span)], span), [], span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    fn = ir.Function(
        "outer_for_wrapping_while",
        [src, (out, ir.ParamDirection.InOut), cond],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                outer_for,
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [y, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    analysis = inspect(fn)
    attempts = analysis["child_attempts"]
    # Outer loop containing WhileStmt must be skipped by pre-screen without attempting Bind/Analysis,
    # while inner eligible ForStmt inside WhileStmt is promoted cleanly.
    assert len(attempts) == 1
    assert attempts[0]["scope"] is inner_child
    assert attempts[0]["stage"] == "logical_export_ready"

    # Outer loop containing only WhileStmt with no inner ForStmt builds zero SyncRegionBindings index.
    empty_outer = ir.ForStmt(
        outer_i,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.WhileStmt(
                    cond,
                    [],
                    ir.SeqStmts(
                        [
                            ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
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
    no_inner_fn = clone(
        fn,
        body=ir.SeqStmts(
            [
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                empty_outer,
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [y, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
    )
    no_inner_analysis = inspect(no_inner_fn)
    assert no_inner_analysis["child_attempts"] == []
    assert no_inner_analysis["cost"]["child_index_nodes"] == 0


def test_local_only_and_readonly_children_receive_caller_bounds_without_disjoint_gm():
    span = ir.Span.unknown()
    idx = ir.DataType.INDEX
    ci = lambda v: ir.ConstInt(v, idx, span)
    tup = lambda vals: ir.MakeTuple([ci(v) for v in vals], span)
    vec_t = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm_t = ir.TensorType([2, 8], ir.DataType.FP32)

    src = ir.Var("src", gm_t, span)
    out = ir.Var("out", gm_t, span)
    n = ir.Var("n", ir.ScalarType(idx), span)
    x = ir.Var("x", vec_t, span)
    y = ir.Var("y", vec_t, span)
    k = ir.Var("k", ir.ScalarType(idx), span)

    # Child loop uses caller-bounded `n` in [2, 2] (`range(n, 4)`) and reads GM `src` (readonly external effect).
    child = ir.ForStmt(
        k,
        n,
        ci(4),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.get_op("buffer.load"), [src, tup([0, 0]), tup([2, 8]), x], ir.VoidType(), span), span),
                ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [x, y], ir.VoidType(), span), span),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    worker = ir.Function(
        "worker_caller_bounds_no_disjoint",
        [src, (out, ir.ParamDirection.InOut), n],
        [out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], vec_t, span), span),
                ir.AssignStmt(y, ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(128)], vec_t, span), span),
                child,
                ir.EvalStmt(ir.Call(ir.get_op("buffer.store"), [y, tup([0, 0]), tup([2, 8]), out], ir.VoidType(), span), span),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.AIV,
        level=ir.Level.AIV,
        role=ir.Role.SubWorker,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    gv_worker = ir.GlobalVar(worker.name)
    e_src = ir.Var("e_src", gm_t, span)
    e_out = ir.Var("e_out", gm_t, span)
    e_res = ir.Var("e_res", out.type, span)
    caller = ir.Function(
        "caller_entry",
        [e_src, (e_out, ir.ParamDirection.InOut)],
        [e_out.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(e_res, ir.Call(gv_worker, [e_src, e_out, ci(2)], out.type, span), span),
                ir.ReturnStmt([e_res], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.Opaque,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    prog = ir.Program([worker, caller], "CallerBoundsNoDisjointGm", span)

    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", autosync_disjoint_gm=False
    ):
        opt = passes.optimize_sync()(passes.insert_sync()(prog))
        p3 = passes.lower_sync()(opt)
        p1 = passes.realize_sync()(prog)

    log_fn = next(f for g, f in opt.functions.items() if g.name == "worker_caller_bounds_no_disjoint")
    assert dict(log_fn.attrs)["autosync_retained_child_count"] == 1
    for phys_prog in (p3, p1):
        pf = next(f for g, f in phys_prog.functions.items() if g.name == "worker_caller_bounds_no_disjoint")
        assert dict(pf.attrs)["pypto_autosync"] is True
        check_closed_commands(log_fn, pf, {2: 2}, range(6), compare_commands=False, hardware_source=worker)


@pytest.mark.parametrize("count", [None, 3], ids=["straight-line", "loop"])
def test_inline_entry_reuses_delivered_completions(count):
    # The two exact slices use overlapping Vec windows. A may-alias GM store
    # separates the slices and publishes completions the next slice inherits.
    fn = vec_fixture(count=count, visits=2, view="overlap")
    analysis = inspect(fn)
    assert len(analysis["children"]) == 2
    assert all(child["scope"] is None for child in analysis["children"])
    assert any(len(c["services"]) < len(c["required_services"]) for c in analysis["children"])
    logical, physical = construct(fn)
    for output in (0, 1 << 20, 2 << 20):
        check_closed_commands(
            logical,
            physical,
            {},
            range(8),
            tensor_bases={0: 0, 1: 1 << 20, 2: output},
            compare_commands=False,
        )


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint-loads", "aliased-loads"])
def test_inline_first_use_orders_prefetch_across_guarded_visits(overlap):
    # The first vector consumer of prefetched runs after the slice's internal
    # load-to-vector wait. The first load itself needs an entry barrier only
    # when its destination overlaps the prefetch.
    original = vec_fixture(view="overlap")
    span = original.span
    *allocs, load_x, abs_x, load_y, add, abs_z, store, ret = original.body.stmts
    root = load_x.expr.args[-1]
    idx = ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, idx, span)

    def payload(name, args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), args, ir.VoidType(), span), span)

    prefetched = ir.Var("prefetched", root.type, span)
    alloc = ir.AssignStmt(
        prefetched,
        ir.Call(
            ir.get_op("buffer.alloc"), [allocs[0].value.args[0], ci(0 if overlap else 3072)], root.type, span
        ),
        span,
    )
    prefetch = payload("buffer.load", [*load_x.expr.args[:-1], prefetched])
    consume = payload("buffer.add", [prefetched, *abs_z.expr.args])
    active = ir.Var("active", ir.ScalarType(ir.DataType.BOOL), span)
    trips = ir.Var("trips", ir.ScalarType(idx), span)
    branch = ir.IfStmt(
        active, ir.SeqStmts([load_x, abs_x, load_y, add, consume, store], span), None, [], span
    )
    loop = ir.ForStmt(
        ir.Var("i", trips.type, span),
        ci(0),
        trips,
        ci(1),
        [],
        ir.SeqStmts([prefetch, branch, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    fn = ir.Function(
        "guarded_prefetch",
        [*zip(original.params, original.param_directions, strict=True), active, trips],
        original.return_types,
        ir.SeqStmts([*allocs, alloc, loop, ret], span),
        span,
        type=original.func_type,
        ir_stage=original.ir_stage,
    )

    (child,) = inspect(fn)["children"]
    dma = ir.PipeType.MTE2
    assert (dma, dma) in child["required_services"] and (dma, ir.PipeType.V) in child["required_services"]
    assert ((dma, dma) in child["services"]) == overlap
    assert (dma, ir.PipeType.V) not in child["services"]
    logical, physical = construct(fn)
    for taken, count in ((True, 0), (False, 3), (True, 1), (True, 3)):
        _, commands = check_closed_commands(
            logical,
            physical,
            {3: taken, 4: count},
            range(8),
            tensor_bases={0: 0, 1: 1 << 20, 2: 2 << 20},
            compare_commands=False,
        )
        if taken:
            for start in range(0, len(commands.payloads), 7):
                complete = ("payload", start, "C")
                assert ((complete, ("payload", start + 1, "I")) in commands.order) == overlap
                assert (complete, ("payload", start + 5, "I")) in commands.order
