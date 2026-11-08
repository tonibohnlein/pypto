# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Realize retained demands and independently check the emitted token graph."""

import pytest
from pypto import codegen, ir, passes, testing

from . import run_sync_matmul_prototype as campaign
from .sync_logical_oracle import validate_logical
from .sync_matmul_cases import CASES, capture, staged_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def verify(function):
    properties = passes.IRPropertySet()
    for name in ["SSAForm", "UseAfterDef", "TypeChecked", "BufferIR", "LogicalSyncValid"]:
        properties.insert(getattr(passes.IRProperty, name))
    program = ir.Program([function], "LogicalMatmul", function.span)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
    assert not diagnostics, [d.message for d in diagnostics]
    return program


def check(function):
    retained = testing.analyze_placed_sync(function, True)
    before = retained.describe()
    first = retained.realize()
    verify(first["function"])
    result = validate_logical(function, first["function"])
    assert result["waits"] == result["matched_demands"]
    assert retained.describe() == before
    second = retained.realize()
    ir.assert_structural_equal(first["function"], second["function"])
    return first, result


@pytest.mark.parametrize("name", CASES)
def test_real_matmul_logical_commands(name):
    _, result = check(capture(name))
    assert result["records"] and result["waits"]
    assert result["extra_event_pairs"] == 0


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("visits", [1, 2, 3])
def test_logical_carries_and_counter_resets(count, visits):
    _, result = check(staged_fixture(count, visits=visits))
    assert bool(result["records"]) == bool(count)


@pytest.mark.parametrize(("start", "step"), [(3, 1), (0, 2), (3, 2)])
def test_logical_original_coordinates(start, step):
    check(staged_fixture(3, visits=2, start=start, step=step))


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_logical_static_stages_and_fanout(stages):
    _, result = check(staged_fixture(3, stages=stages, visits=2))
    assert (result["same_pipe_waits"] > 0) == (stages > 1)


def test_logical_constant_construction_through_large_trip_counts():
    signatures = []
    for count in [2, 4, 8, 16, 64, 1024, 1000000]:
        result = testing.analyze_placed_sync(staged_fixture(count, visits=2), True).realize()
        verify(result["function"])
        signatures.append({k: v for k, v in result.items() if k != "function"})
    assert all(s == signatures[0] for s in signatures)


def test_optional_original_empty_yield():
    class RemoveYield(ir.IRMutator):
        removed = 0

        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            body = [s for s in op.body.stmts if not isinstance(s, ir.YieldStmt)]
            self.removed += len(op.body.stmts) - len(body)
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(body, op.span),
                op.return_vars,
                op.span,
            )

    mutation = RemoveYield()
    function = mutation.visit_function(staged_fixture(3, visits=2))
    assert mutation.removed == 2
    check(function)


def test_logical_shared_physical_storage_and_name_collision():
    class Alias(ir.IRMutator):
        def visit_call(self, op):
            if (
                op.op.name == ir.get_op("buffer.alloc").name
                and isinstance(op.type, ir.BufferType)
                and op.type.memory_space == ir.MemorySpace.Mat
            ):
                assert isinstance(op.args[1], ir.ConstInt)
                if op.args[1].value == 4096:
                    return ir.Call(
                        op.op, [op.args[0], ir.ConstInt(0, ir.DataType.INDEX, op.span)], op.type, op.span
                    )
            return op

    check(Alias().visit_function(staged_fixture(3, visits=2)))

    f = staged_fixture(3)
    replacements = {}

    class Rename(ir.IRMutator):
        def visit_var(self, op):
            if op.name_hint == "k0":
                if op not in replacements:
                    replacements[op] = ir.Var("sync_completion_0", op.type, op.span)
                return replacements[op]
            return op

    check(Rename().visit_function(f))


def test_logical_serialization_and_codegen_boundary(tmp_path):
    function = staged_fixture(3, visits=2)
    result, _ = check(function)
    path = tmp_path / "logical.msgpack"
    ir.serialize_to_file(result["function"], str(path))
    decoded = ir.deserialize_from_file(str(path))
    assert isinstance(decoded, ir.Function)
    ir.assert_structural_equal(result["function"], decoded)
    program = verify(decoded)
    validate_logical(function, decoded)
    with pytest.raises(ValueError, match="Logical synchronization requires LowerSync"):
        codegen.PTOCodegen().generate(program)


def test_oracle_rejects_missing_wait():
    class DropWait(ir.IRMutator):
        dropped = False

        def visit_eval_stmt(self, op):
            if (
                not self.dropped
                and isinstance(op.expr, ir.Call)
                and op.expr.op.name == ir.get_op("sync.wait").name
            ):
                self.dropped = True
                return ir.SeqStmts([], op.span)
            return op

    function = staged_fixture(3, visits=2)
    emitted = testing.analyze_placed_sync(function, True).realize()["function"]
    with pytest.raises(AssertionError, match="missing coverage"):
        validate_logical(function, DropWait().visit_function(emitted))


def test_oracle_rejects_wrong_carried_source():
    class WrongSource(ir.IRMutator):
        def visit_yield_stmt(self, op):
            if len(op.value) > 1:
                return ir.YieldStmt([op.value[1], *op.value[1:]], op.span)
            return op

    function = staged_fixture(3, visits=2)
    emitted = testing.analyze_placed_sync(function, True).realize()["function"]
    with pytest.raises(AssertionError):
        validate_logical(function, WrongSource().visit_function(emitted))


def test_oracle_rejects_late_record():
    class LateRecord(ir.IRMutator):
        changed = False

        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            statements = list(op.body.stmts)
            for i, s in enumerate(statements[:-1]):
                following = statements[i + 1]
                if (
                    not self.changed
                    and isinstance(s, ir.AssignStmt)
                    and isinstance(s.value, ir.Call)
                    and s.value.op.name == ir.get_op("sync.record").name
                    and isinstance(following, ir.EvalStmt)
                    and isinstance(following.expr, ir.Call)
                    and following.expr.op.name == ir.get_op("buffer.load").name
                ):
                    statements[i], statements[i + 1] = statements[i + 1], s
                    self.changed = True
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(statements, op.span),
                op.return_vars,
                op.span,
            )

    function = staged_fixture(1, visits=2)
    emitted = testing.analyze_placed_sync(function, True).realize()["function"]
    mutation = LateRecord()
    damaged = mutation.visit_function(emitted)
    assert mutation.changed
    with pytest.raises(AssertionError):
        validate_logical(function, damaged)


def test_shared_affine_dag_name_collection_is_linear():
    scans = []
    for depth in [0, 4, 8, 12]:
        span = ir.Span.unknown()
        zero = ir.ConstInt(0, ir.DataType.INDEX, span)
        for _ in range(depth):
            zero = ir.Add(zero, zero, ir.DataType.INDEX, span)

        class SharedOffset(ir.IRMutator):
            def visit_add(self, op):
                return ir.Add(op, zero, ir.DataType.INDEX, op.span)

        f = SharedOffset().visit_function(staged_fixture(2))
        retained = testing.analyze_placed_sync(f, True)
        result = retained.realize()
        scans.append((result["name_scan_nodes"], result["name_scan_visits"]))
        # Only the added realizer traversal is qualified here. The importer
        # still has its separately documented expression traversal costs.
    assert all(b[0] - a[0] == 4 and b[1] - a[1] == 8 for a, b in zip(scans, scans[1:]))


def test_realization_refusal_retains_mathematics():
    class SharedCall(ir.IRMutator):
        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            body = [op.body.stmts[0], *op.body.stmts]
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(body, op.span),
                op.return_vars,
                op.span,
            )

    f = SharedCall().visit_function(staged_fixture(2))
    retained = testing.analyze_placed_sync(f, True)
    before = retained.describe()
    with pytest.raises(ValueError, match="Shared payload call"):
        retained.realize()
    assert retained.describe() == before
    assert retained.precedes((0, 0, 0), False, (0, 0, 0), True)


def test_campaign_retains_analysis_on_logical_refusal(monkeypatch):
    f = staged_fixture(2)
    retained = testing.analyze_placed_sync(f, True)

    class Refusal:
        def realize(self):
            raise ValueError("missing export")

        def __getattr__(self, name):
            return getattr(retained, name)

    monkeypatch.setattr(campaign, "capture", lambda *args: f)
    monkeypatch.setattr(campaign.testing, "analyze_placed_sync", lambda *args: Refusal())
    result = campaign.autotile_case(64, True)
    assert result["status"] == "logical_unavailable"
    assert result["analysis"] and result["checks"]["covers"] > 0
    assert result["logical"]["status"] == "unavailable"
    assert result["logical"]["reason"] == "missing export"


def test_campaign_checker_failure_is_not_export_refusal(monkeypatch):
    def failed_check(*args):
        raise ValueError("incorrect emitted payload")

    f = staged_fixture(2)
    monkeypatch.setattr(campaign, "capture", lambda *args: f)
    monkeypatch.setattr(campaign, "validate_logical", failed_check)
    with pytest.raises(ValueError, match="incorrect emitted payload"):
        campaign.autotile_case(64, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
