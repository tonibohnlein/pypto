# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Original-frame physical transport with independently checked command graphs."""

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import sequence_stmts, validate
from .sync_physical_oracle import run_physical
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_nested_sync_repetition import resident_pipeline_tree, scoped_startup_tree
from .test_qwen_prefill_pipeline import activation_fixture
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize(
    "inner,outer,replay",
    [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (2, 2, 1), (2, 2, 2), (1, 3, 2), (1, 4, 2), (1, 5, 2)],
)
@pytest.mark.parametrize("kind", ["resident", "startup", "fix"])
def test_scoped_physical_matching_and_reuse_across_every_carry(kind, inner, outer, replay):
    original = (
        resident_pipeline_tree(inner, outer, replay)
        if kind == "resident"
        else scoped_startup_tree(inner, outer, replay, intermediate_read=kind == "fix")
    )
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)
    description = retained.describe()
    unsupported_barriers = []
    export = retained.export_scoped_demands()
    assert export is not None
    for family in export["families"]:
        sr, _, ss = family["source"]
        tr, _, ts = family["target"]
        source = description["regions"][sr]["sites"][ss]["pipe"]
        target = description["regions"][tr]["sites"][ts]["pipe"]
        if source == target and source not in {ir.PipeType.M, ir.PipeType.V}:
            unsupported_barriers.append((family, source))
    if unsupported_barriers:
        # Independent cover validation above establishes these obligations.
        # The admitted target has no qualified DMA/FIX barrier in this path.
        with pytest.raises(ValueError, match="Scoped same-pipe barrier mechanism"):
            retained.realize_physical("a2a3", list(range(6)))
        return
    if kind in {"startup", "fix"} and inner == outer == 2 and replay > 0:
        # The existing whole-span reservation is sufficient, not optimal. This
        # nested sharing pattern needs a stronger inter-family certificate;
        # retain the refusal rather than changing the allocator or ID pool.
        with pytest.raises(ValueError, match="Allocation unproved by whole-span reservation"):
            retained.realize_physical("a2a3", list(range(6)))
        return
    physical = retained.realize_physical("a2a3", list(range(6)))
    run_physical(original, physical["function"], range(6))


@pytest.mark.parametrize("hidden,block,layer", [(512, 0, 0), (512, 2, 1), (512, 19, 0), (1024, 3, 1)])
def test_original_qwen_output_scoped_commands(hidden, block, layer):
    program = capture_qwen_projection("output", layers=2, hidden=hidden, band=3)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    owner = testing.analyze_sync_invocation(worker, True, "a2a3", {3: (0, 1)}, (0, 19))
    retained = owner.select_case(block, {3: layer})
    assert retained is not None
    physical = retained.realize_physical("a2a3", list(range(6)))
    state, _, _ = run_physical(worker, physical["function"], range(6), {3: layer}, block_index=block)
    assert len(state.waits) == len(state.records)
    assert physical["barrier_sites"] == 0


@pytest.mark.parametrize("band,blocks", [(0, [0, 6, 7, 16, 17, 23]), (3, [0, 5, 6, 12, 13, 22, 23])])
def test_one_original_gate_up_program_covers_all_activation_boundaries(band, blocks):
    program = capture_qwen_projection("gate_up", layers=2, hidden=512, band=band)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    owner = testing.analyze_sync_invocation(worker, True, "a2a3", {3: (0, 1)}, (0, 23))
    physical = owner.realize_physical("a2a3", list(range(6)))
    assert len(physical["cases"]) == len(owner.describe()["cases"])
    for i, block in enumerate(blocks):
        run_physical(worker, physical["function"], range(6), {3: i % 2}, block_index=block)

    # The per-execution oracle above establishes exact operand and payload
    # preservation. Static phase/case copies are charged, not forbidden.
    class Payloads(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, op):
            if op.op.name.startswith("buffer.") and op.op.name not in {
                ir.get_op("buffer.alloc").name,
                ir.get_op("buffer.get_block_idx").name,
            }:
                self.count += 1

    before, after = Payloads(), Payloads()
    before.visit_function(worker)
    after.visit_function(physical["function"])
    # Keep the inherited conservative site bound. Exact no-clone payload
    # equality is checked independently in test_sync_schedule_ablation.
    assert before.count <= after.count <= 6 * before.count


def test_scoped_palette_refusal_is_not_claimed_as_minimum_capacity():
    program = capture_qwen_projection("output", layers=2, band=3)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    owner = testing.analyze_sync_invocation(worker, True, "a2a3", {3: (0, 1)}, (0, 19))
    with pytest.raises(ValueError, match="Allocation unproved by whole-span reservation"):
        owner.realize_physical("a2a3", [0, 1, 2])
    result = owner.realize_physical("a2a3", list(range(6)))
    assert result["schedule_policy"]["no_clone"]
    assert result["set_sites"] == result["wait_sites"] == 17
    assert not result["cases"][0]["finite_minimum"]
    specialized = owner.realize_physical(
        "a2a3",
        list(range(6)),
        schedule_policy=testing.SyncSchedulePolicy(mode="boundary", no_clone=False),
    )
    assert specialized["set_sites"] == specialized["wait_sites"] == 29
    assert not specialized["cases"][0]["finite_minimum"]
    assert [case["families"] for case in specialized["cases"]] == [
        case["families"] for case in result["cases"]
    ]


def test_activation_commands_preserve_empty_cases_and_original_payloads():
    original = activation_fixture()
    owner = testing.analyze_sync_invocation(original, True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 7))
    physical = owner.realize_physical("a2a3", list(range(6)))
    for block in range(8):
        run_physical(original, physical["function"], range(6), {1: 0, 3: 32}, block_index=block)


def test_invocation_case_guard_requires_dominating_original_binding():
    original = activation_fixture()
    block, allocation, loop, ret = sequence_stmts(original.body)
    assert isinstance(loop, ir.ForStmt)
    # A real payload precedes the block definition; no universal cut dominance
    # can be claimed by the current adapter. Mathematics remains available.
    body = ir.SeqStmts([allocation, sequence_stmts(loop.body)[0], block, loop, ret], original.span)
    changed = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        body,
        original.span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    owner = testing.analyze_sync_invocation(changed, True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 7))
    with pytest.raises(ValueError, match="Original dominating invocation block binding"):
        owner.realize_physical("a2a3", list(range(6)))
    selected = owner.select_case(0, {1: 0, 3: 32})
    assert selected is not None
    assert selected.export_scoped_demands() is not None
    assert changed.body.same_as(body)


def test_scoped_reuse_proof_and_command_size_do_not_unfold_trip_counts():
    summaries = []
    for count in [3, 16, 1024, 1000000]:
        original = scoped_startup_tree(1, count, count)
        retained = testing.analyze_placed_sync(original, True)
        physical = retained.realize_physical("a2a3", list(range(6)))
        d = retained.describe()
        summaries.append(
            (
                d["analyzed_occurrences"],
                d["scoped_reuse_queries"],
                physical["set_sites"],
                physical["wait_sites"],
                physical["guard_sites"],
            )
        )
        # Cached certificate answers must not repeat the proof queries.
        retained.realize_physical("a2a3", list(range(6)))
        assert retained.describe()["scoped_reuse_queries"] == d["scoped_reuse_queries"]
    assert len(set(summaries)) == 1


@pytest.mark.parametrize("intervening", [False, True])
def test_scoped_barrier_requires_native_adjacency(intervening):
    base = vec_fixture(count=2)
    span, dtype = base.span, ir.DataType.INDEX

    def ci(x):
        return ir.ConstInt(x, dtype, span)

    definitions = list(sequence_stmts(base.body)[:-2])
    inner = sequence_stmts(base.body)[-2]
    assert isinstance(inner, ir.ForStmt)
    statements = list(sequence_stmts(inner.body))
    if intervening:
        z = definitions[-1]
        assert isinstance(z, ir.AssignStmt)
        assert isinstance(z.value, ir.Call)
        spare = ir.Var("independent", z.var.type, span)
        args = list(z.value.args)
        args[-1] = ci(4096)
        definitions.append(ir.AssignStmt(spare, ir.Call(z.value.op, args, spare.type, span), span))
        load_stmt = statements[0]
        assert isinstance(load_stmt, ir.EvalStmt)
        load = load_stmt.expr
        assert isinstance(load, ir.Call)
        load_args = list(load.args)
        load_args[-1] = spare
        definitions.append(ir.EvalStmt(ir.Call(load.op, load_args, load.type, span), span))
        statements.insert(
            3, ir.EvalStmt(ir.Call(ir.get_op("buffer.abs"), [spare, spare], ir.VoidType(), span), span)
        )
    inner = ir.ForStmt(
        inner.loop_var, inner.start, inner.stop, inner.step, [], ir.SeqStmts(statements, span), [], span
    )
    outer = ir.ForStmt(
        ir.Var("outer", ir.ScalarType(dtype), span),
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(ir.Var("marker", ir.ScalarType(dtype), span), ci(0), span),
                inner,
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    original = ir.Function(
        base.name,
        list(zip(base.params, base.param_directions)),
        base.return_types,
        ir.SeqStmts([*definitions, outer, sequence_stmts(base.body)[-1]], span),
        span,
        base.func_type,
        ir_stage=base.ir_stage,
    )
    retained = testing.analyze_placed_sync(original, True)
    assert retained.describe()["source_region_nodes"] > 0
    validate(original, retained)
    if intervening:
        with pytest.raises(ValueError, match="Scoped same-pipe barrier adjacency is unproved"):
            retained.realize_physical("a2a3", list(range(6)))
    else:
        physical = retained.realize_physical("a2a3", list(range(6)))
        assert physical["barrier_sites"] > 0
        run_physical(original, physical["function"], range(6))
