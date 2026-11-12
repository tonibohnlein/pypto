# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Original Qwen projection input contracts and staged qualification."""

import ast

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import interpret, sequence_stmts, validate
from .sync_qwen_prefill_cases import (
    assert_gate_up_body_identity,
    assert_output_body_identity,
    capture_qwen_projection,
    original_control_trace,
    source_body_ast,
)
from .test_nested_sync_repetition import expand_scoped_families
from .test_sync_range_proofs import range_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def test_original_gate_up_compute_body_is_unchanged():
    assert_gate_up_body_identity()
    assert_output_body_identity()


def test_source_body_fingerprint_preserves_empty_fields_and_control():
    # Empty call keywords/else arms disappeared from the default dump on the
    # newer host interpreter, although the device still included them.
    tree = ast.parse("if ready:\n    consume()\n")
    fingerprint = source_body_ast(tree)
    assert "keywords=[]" in fingerprint
    assert "orelse=[]" in fingerprint
    assert "type_ignores=[]" in fingerprint
    assert fingerprint != source_body_ast(ast.parse("if not ready:\n    consume()\n"))
    assert fingerprint != source_body_ast(ast.parse("if ready:\n    publish()\n"))


@pytest.mark.parametrize("name,blocks", [("output", 20), ("gate_up", 24)])
def test_qwen_projection_retains_actual_launch_contract(name, blocks):
    program = capture_qwen_projection(name)
    evidence = testing.sync_invocation_evidence(program)
    workers = [f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC]
    assert len(workers) == 1
    worker = workers[0]
    assert evidence["block_indices"][worker.name] == (0, blocks - 1)
    # The original symbolic ABI supplies no layer or band bound. Launch evidence
    # alone must not make the old flat importer accept these nested kernels.
    assert not evidence["parameters"].get(worker.name)
    with pytest.raises(ValueError, match="caller range|constant extent"):
        testing.analyze_placed_sync(worker, True)
    assert len(loop_inventory(worker.body)) == (4 if name == "output" else 8)


def loop_inventory(stmt):
    if isinstance(stmt, ir.ForStmt):
        return [stmt, *loop_inventory(stmt.body)]
    if isinstance(stmt, ir.SeqStmts):
        return [loop for child in sequence_stmts(stmt) for loop in loop_inventory(child)]
    if isinstance(stmt, ir.IfStmt):
        return loop_inventory(stmt.then_body) + (loop_inventory(stmt.else_body) if stmt.else_body else [])
    return []


@pytest.mark.parametrize("name,blocks", [("output", 20), ("gate_up", 24)])
def test_configured_qwen_has_explicit_extent_and_forwarded_layer_bounds(name, blocks):
    program = capture_qwen_projection(name, layers=2, band=3)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    evidence = testing.sync_invocation_evidence(program)
    assert evidence["parameters"][worker.name] == {3: (0, 1)}
    assert evidence["block_indices"][worker.name] == (0, blocks - 1)
    weights = [
        p
        for p in worker.params
        if isinstance(p.type, ir.TensorType)
        and isinstance(p.type.shape[0], ir.ConstInt)
        and p.type.shape[0].value == 10240
    ]
    assert len(weights) == (1 if name == "output" else 2)
    # The explicit caller proves inputs; nested import is a distinct later gate.
    with pytest.raises(ValueError, match="Launch coordinate has no established caller range"):
        testing.analyze_placed_sync(worker, True, parameter_bounds=evidence["parameters"][worker.name])


@pytest.mark.parametrize(
    "name,core",
    [("output", 0), ("output", 1), ("output", 19), ("gate_up", 0), ("gate_up", 7), ("gate_up", 23)],
)
@pytest.mark.parametrize("band,layer", [(0, 1), (3, 0)])
def test_small_qwen_original_accesses_initialization_and_skips(name, core, band, layer):
    # Named hidden-size derivative: same original body, stage geometry and launch,
    # one two-stage K pair. The interpreter receives actual input values and
    # does not consume compiler range evidence, storage cells or demand records.
    program = capture_qwen_projection(name, layers=2, hidden=512, band=band)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    _, calls = interpret(worker, {3: layer}, block_index=core)
    gu = (core + band * 6) % 24
    expected = int(core < 2) if name == "output" else int(gu < 17) + int((gu + 17) % 24 < 17)
    assert len(calls) == 53 * expected
    assert sum(call[0].op.name == ir.get_op("buffer.matmul").name for call in calls) == expected
    assert sum(call[0].op.name == ir.get_op("buffer.store").name for call in calls) == expected
    # All reads/writes are recovered from original physical views by interpret;
    # address_set checks every touched element is inside its declared allocation.
    assert all(reads and writes for _, _, _, reads, writes in calls)


@pytest.mark.parametrize("name,blocks", [("output", 20), ("gate_up", 24)])
def test_all_original_workers_layers_and_gate_up_bands(name, blocks):
    for band in range(4) if name == "gate_up" else [0]:
        program = capture_qwen_projection(name, layers=2, band=band)
        worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
        for layer in [0, 1]:
            total_stores = 0
            for block in range(blocks):
                events = original_control_trace(worker, layer, block)
                gu = (block + band * 6) % 24
                phases = 1 if name == "output" else int(gu < 17) + int((gu + 17) % 24 < 17)
                assert len(events) == phases * 521
                assert sum(op == ir.get_op("buffer.matmul").name for op, _ in events) == phases
                stores = sum(op == ir.get_op("buffer.store").name for op, _ in events)
                assert stores == phases
                total_stores += stores
            assert total_stores == (20 if name == "output" else 34)


def test_full_output_projection_admits_proved_singleton_launch_frame():
    program = capture_qwen_projection("output", layers=2, band=3)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    evidence = testing.sync_invocation_evidence(program)
    retained = testing.analyze_placed_sync(
        worker,
        True,
        parameter_bounds=evidence["parameters"][worker.name],
        block_bounds=evidence["block_indices"][worker.name],
    )
    assert retained.describe()["hierarchical"]
    assert retained.export_scoped_demands() is not None
    physical = retained.realize_physical("a2a3", list(range(6)))
    assert physical["schedule_policy"]["no_clone"]
    assert physical["set_sites"] == physical["wait_sites"] == 17
    specialized = retained.realize_physical(
        "a2a3",
        list(range(6)),
        schedule_policy=testing.SyncSchedulePolicy(mode="boundary", no_clone=False),
    )
    # Explicitly requested boundary phases keep the same allocated handoffs.
    assert specialized["set_sites"] == specialized["wait_sites"] == 29
    assert specialized["families"] == physical["families"]


@pytest.mark.parametrize("block,layer", [(0, 0), (3, 1)])
def test_small_original_output_covers_and_selectors_under_explicit_active_domain(block, layer):
    program = capture_qwen_projection("output", layers=2, hidden=1024, band=3)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    evidence = testing.sync_invocation_evidence(program)
    # Diagnostic domain restriction, supplied explicitly. The complete original
    # 20-worker launch has skipped workers at this smaller hidden-size derivative;
    # admitting those needs the separate activation interface, not this proof.
    retained = testing.analyze_placed_sync(
        worker,
        True,
        parameter_bounds=evidence["parameters"][worker.name],
        block_bounds=(0, 3),
    )
    checked = validate(worker, retained, parameters={3: layer}, block_index=block)
    assert checked["payloads"] == 105
    assert expand_scoped_families(retained.export_scoped_demands()) == set(
        retained.expand_retained_demands(checked["payloads"])
    )
    description = retained.describe()
    translated = [i for i, domain in enumerate(description["storage"]) if domain["column_block"]]
    assert translated
    for domain in translated:
        assert retained.boundary(domain, 0, 0, {3: layer}) is None
        assert retained.boundary(domain, 0, 0, {3: layer}, 4) is None
    with pytest.raises(ValueError, match="Optional or varying outer visits"):
        testing.analyze_placed_sync(worker, True, parameter_bounds={3: (0, 1)}, block_bounds=(0, 19))


@pytest.mark.parametrize("name", ["output", "gate_up"])
@pytest.mark.parametrize("band", [0, 1, 2, 3])
def test_invocation_partition_covers_original_launch_without_worker_enumeration(name, band):
    program = capture_qwen_projection(name, layers=2, hidden=512, band=band)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    evidence = testing.sync_invocation_evidence(program)
    bounds = evidence["block_indices"][worker.name]
    retained = testing.analyze_sync_invocation(
        worker, True, "a2a3", evidence["parameters"][worker.name], bounds
    )
    description = retained.describe()
    assert description["cases"][0]["block_bounds"][0] == bounds[0]
    assert description["cases"][-1]["block_bounds"][1] == bounds[1]
    for a, b in zip(description["cases"], description["cases"][1:]):
        assert a["block_bounds"][1] + 1 == b["block_bounds"][0]
    assert len(description["cases"]) <= 6
    for block in range(bounds[1] + 1):
        selected = retained.select_case(block, {3: 0})
        assert selected is not None
        events = original_control_trace(worker, 0, block)
        assert sum(r["count"] * len(r["sites"]) for r in selected.describe()["regions"]) == len(events)
    assert retained.select_case(-1, {3: 0}) is None
    assert retained.select_case(bounds[1] + 1, {3: 0}) is None
    assert retained.select_case(0, {}) is None
    assert retained.select_case(0, {3: 2}) is None


@pytest.mark.parametrize(
    "name,band,block", [("output", 0, 2), ("gate_up", 0, 0), ("gate_up", 1, 0), ("gate_up", 3, 0)]
)
def test_invocation_cases_against_original_access_and_cover_oracle(name, band, block):
    program = capture_qwen_projection(name, layers=2, hidden=512, band=band)
    worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    evidence = testing.sync_invocation_evidence(program)
    owner = testing.analyze_sync_invocation(
        worker, True, "a2a3", evidence["parameters"][worker.name], evidence["block_indices"][worker.name]
    )
    retained = owner.select_case(block, {3: 1})
    assert retained is not None
    checked = validate(worker, retained, parameters={3: 1}, block_index=block)
    assert expand_scoped_families(retained.export_scoped_demands()) == set(
        retained.expand_retained_demands(checked["payloads"])
    )


def activation_fixture(offset=3, modulus=8, stop=5, overflow=False, stride=1):
    original = range_fixture()
    span, dtype = original.span, ir.DataType.INT32
    block = ir.Var("block", ir.ScalarType(ir.DataType.INDEX), span)

    def ci(value):
        return ir.ConstInt(value, dtype, span)

    added = ir.Add(
        ir.Mul(ir.Cast(block, dtype, span), ci(stride), dtype, span),
        ci(2147483647 if overflow else offset),
        dtype,
        span,
    )
    start = ir.FloorMod(added, ci(modulus), dtype, span)
    loop = ir.ForStmt(
        ir.Var("visit", ir.ScalarType(ir.DataType.INDEX), span),
        ir.Cast(start, ir.DataType.INDEX, span),
        ir.ConstInt(stop, ir.DataType.INDEX, span),
        ir.ConstInt(modulus, ir.DataType.INDEX, span),
        [],
        ir.SeqStmts(list(sequence_stmts(original.body)[1:3]), span),
        [],
        span,
    )
    body = ir.SeqStmts(
        [
            ir.AssignStmt(block, ir.Call(ir.get_op("buffer.get_block_idx"), [], block.type, span), span),
            sequence_stmts(original.body)[0],
            loop,
            sequence_stmts(original.body)[-1],
        ],
        span,
    )
    return ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        body,
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


def test_activation_modulo_partition_is_exhaustive_at_boundaries_and_keeps_absence():
    original = activation_fixture()
    result = testing.analyze_sync_invocation(original, True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 7))
    assert [c["block_bounds"] for c in result.describe()["cases"]] == [(0, 1), (2, 4), (5, 7)]
    for block in range(8):
        selected = result.select_case(block, {1: 0, 3: 32})
        assert selected is not None
        checked = validate(original, selected, parameters={1: 0, 3: 32}, block_index=block)
        assert checked["payloads"] == 2 * int((block + 3) % 8 < 5)
        assert expand_scoped_families(selected.export_scoped_demands()) == set(
            selected.expand_retained_demands(checked["payloads"])
        )
        if checked["payloads"] == 0:
            with pytest.raises(ValueError, match="Invalid numerical region"):
                selected.precedes((0, 0, 0), False, (0, 0, 0), True)


def test_activation_original_overflow_is_not_hidden_by_modulo():
    with pytest.raises(ValueError, match="overflow or become negative"):
        testing.analyze_sync_invocation(
            activation_fixture(overflow=True), True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 7)
        )


def test_activation_missing_fixed_offset_or_single_orbit_stays_unavailable():
    original = activation_fixture()

    class RuntimeOffset(ir.IRMutator):
        def visit_add(self, op):
            return ir.Add(op.left, original.params[1], ir.DataType.INT32, op.span)

    changed = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        RuntimeOffset().visit_stmt(original.body),
        original.span,
        original.func_type,
        ir_stage=original.ir_stage,
    )
    with pytest.raises(ValueError, match="not a proved fixed value"):
        testing.analyze_sync_invocation(changed, True, "a2a3", {1: (0, 3), 3: (32, 32)}, (0, 7))
    with pytest.raises(ValueError, match="single-orbit"):
        testing.analyze_sync_invocation(original, True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 15))
    with pytest.raises(ValueError, match="single-orbit"):
        testing.analyze_sync_invocation(
            activation_fixture(stride=2), True, "a2a3", {1: (0, 0), 3: (32, 32)}, (0, 3)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
