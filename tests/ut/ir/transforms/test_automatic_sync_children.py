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
    assert len(attempts) == 1 and attempts[0]["scope"] is scope
    assert attempts[0]["stage"] == "logical_export_ready"
    assert attempts[0]["physical_qualification"] == "pending_whole_plan"
    retained = attempts[0]["mathematics"]
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


def test_automatic_preparation_does_not_unfold_count_or_retry_enclosing_loops():
    signatures = []
    for count in (2, 16, 1024):
        original, scope = eventful_fixture(count)
        analysis = inspect(original)
        attempts = analysis["child_attempts"]
        assert len(attempts) == 1 and attempts[0]["scope"] is scope
        assert attempts[0]["stage"] == "logical_export_ready"
        description = attempts[0]["mathematics"].describe()
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
