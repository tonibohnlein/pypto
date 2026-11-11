# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One original payload tree with analytically partitioned token wait guards."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_parameterized_token_sync import apply, block_resident, callers, check, worker

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("name,band", [("output", 3), *(("gate_up", b) for b in range(4))])
def test_full_qwen_actual_tokens(name, band, tmp_path):
    program = capture_qwen_projection(name, layers=2, band=band, hidden=512)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    logical = apply(passes.insert_sync, program)
    archive = tmp_path / "logical.msgpack"
    ir.serialize_to_file(logical, str(archive))
    restored = ir.deserialize_from_file(str(archive))
    assert isinstance(restored, ir.Program)
    physical = apply(passes.lower_sync, apply(passes.optimize_sync, restored))
    assert dict(worker(physical, original.name).attrs)["pypto_autosync"]
    evidence = testing.sync_invocation_evidence(program)
    qualified = testing.analyze_sync_invocation(
        original,
        True,
        "a2a3",
        evidence["parameters"][original.name],
        evidence["block_indices"][original.name],
    )
    intervals = [tuple(case["block_bounds"]) for case in qualified.describe()["cases"]]
    # Both sides of every analytic activation boundary, including empty cases.
    for block in sorted({x for interval in intervals for x in interval}):
        check(
            original,
            worker(restored, original.name),
            worker(physical, original.name),
            {3: block % 2},
            block=block,
        )
    source, output = tmp_path / "qwen.pto", tmp_path / "qwen.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([worker(physical, original.name)], "Physical", original.span)
        )
    )
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    assert "ptoas_auto_sync_tail(" not in output.read_text().split("AICORE void ", 1)[1]


@pytest.mark.parametrize("edit", ["duplicate", "unguard", "straddle"])
def test_actual_case_wait_edits(edit):
    original, program = block_resident()
    inserted = apply(passes.insert_sync, program)
    logical = worker(inserted, original.name)
    changed = 0

    class Edit(ir.IRMutator):
        def visit_eval_stmt(self, op):
            nonlocal changed
            if (
                edit == "duplicate"
                and isinstance(op.expr, ir.Call)
                and op.expr.op.name == ir.get_op("sync.wait").name
            ):
                changed += 1
                return ir.SeqStmts([op, op], op.span)
            return op

        def visit_if_stmt(self, op):
            nonlocal changed
            if (
                isinstance(op.condition, ir.Lt)
                and isinstance(op.condition.right, ir.ConstInt)
                and op.condition.right.value == 9
            ):
                changed += 1
                if edit == "unguard":
                    return op.then_body
                if edit == "straddle":
                    condition = ir.Lt(
                        op.condition.left,
                        ir.ConstInt(8, ir.DataType.INDEX, op.span),
                        ir.DataType.BOOL,
                        op.span,
                    )
                    return ir.IfStmt(condition, op.then_body, op.else_body, op.return_vars, op.span)
            return super().visit_if_stmt(op)

    edited = Edit().visit_function(logical)
    assert changed
    modified = callers(edited, blocks=10)
    snapshot = ir.serialize(modified)
    if edit == "straddle":
        with pytest.raises(ValueError, match="not uniform over the complete case"):
            apply(passes.lower_sync, modified)
        assert ir.serialize(modified) == snapshot
    else:
        physical = apply(passes.lower_sync, modified)
        for block in (0, 8, 9):
            check(
                original,
                edited,
                worker(physical, original.name),
                {len(original.params) - 1: 0},
                block=block,
                minimum_protocol=edit != "duplicate",
            )


def test_case_without_payload_or_commands_and_changed_launch():
    original, program = block_resident()
    inserted = apply(passes.insert_sync, program)
    logical = worker(inserted, original.name)
    params = {len(original.params) - 1: (0, 8)}
    active = testing.inspect_scoped_token_families(logical, parameter_bounds=params, block_bounds=(0, 8))
    absent = testing.inspect_scoped_token_families(logical, parameter_bounds=params, block_bounds=(9, 9))
    assert active["families"] and not absent["families"]
    assert active["work"]["read_statement_visits"] > absent["work"]["read_statement_visits"]
    # Repartitioning a narrower launch still consumes the actual existing IR.
    for blocks in (1, 8):
        narrowed = callers(logical, blocks=blocks)
        physical = apply(passes.lower_sync, narrowed)
        check(
            original,
            logical,
            worker(physical, original.name),
            {len(original.params) - 1: 0},
            block=blocks - 1,
        )
