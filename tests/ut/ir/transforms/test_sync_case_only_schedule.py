# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Production case versioning must not require an endpoint-boundary phase."""

import pytest
from pypto import ir, testing
from pypto.ir.op import sync

from .sync_control_metrics import statements
from .test_parameterized_token_sync import check
from .test_token_sync_lowering import _two_streams

pytestmark = pytest.mark.usefixtures("ascend_backend")


def case_only_fixture(trips=12):
    original, logical = _two_streams(strengthen=True)
    span = original.span
    index = ir.ScalarType(ir.DataType.INDEX)
    block, unused, iv, prefix_iv = [ir.Var(n, index, span) for n in ("block", "unused", "i", "prefix")]

    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, span)

    definition = ir.AssignStmt(block, ir.Call(ir.get_op("buffer.get_block_idx"), [], index, span), span)
    # This original one-visit/absent prefix establishes two invocation cases.
    # The repeated main loop itself executes with identical bounds in both.
    prefix_load = original.body.stmts[5].expr
    prefix_payload = ir.EvalStmt(ir.Call(prefix_load.op, prefix_load.args, prefix_load.type, span), span)
    prefix = ir.ForStmt(
        prefix_iv,
        block,
        ci(1),
        ci(2),
        [],
        ir.SeqStmts([prefix_payload, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    # The terminal MTE2 payload consumes the closing wait on y, so the next
    # iteration's x load does not introduce an unrepresented same-pipe WAW.
    last_load = original.body.stmts[5].expr
    terminal = ir.EvalStmt(ir.Call(last_load.op, last_load.args, last_load.type, span), span)
    closed = ir.Var("closed", ir.SyncTokenType(ir.PipeType.MTE3), span)
    repeated = list(logical.body.stmts[2:-1])
    # Preserve a legal extra consuming use in the first case. The original
    # actual-token parser, rather than a demand recomputation, owns both cases.
    first_wait = next(
        s for s in repeated if isinstance(s, ir.EvalStmt) and s.expr.op.name == ir.get_op("sync.wait").name
    )
    repeated.insert(
        repeated.index(first_wait) + 1,
        ir.IfStmt(ir.Lt(block, ci(1), ir.DataType.BOOL, span), first_wait, None, [], span),
    )
    repeated.extend(
        [
            ir.AssignStmt(closed, sync.record(ir.PipeType.MTE3), span),
            ir.EvalStmt(sync.wait(closed, ir.PipeType.MTE2), span),
            terminal,
        ]
    )

    def make(body):
        loop = ir.ForStmt(
            iv,
            ci(5),
            ci(5 + 2 * trips - 1),
            ci(2),
            [],
            ir.SeqStmts([*body, ir.YieldStmt([], span)], span),
            [],
            span,
        )
        return ir.Function(
            original.name,
            [*zip(original.params, original.param_directions), unused],
            original.return_types,
            ir.SeqStmts([definition, *original.body.stmts[:2], prefix, loop, original.body.stmts[-1]], span),
            span,
            original.func_type,
            ir_stage=original.ir_stage,
        )

    return make([*original.body.stmts[2:-1], terminal]), make(repeated)


def test_case_only_production_keeps_original_loop_bounds_and_actual_uses():
    original, logical = case_only_fixture()
    # Every repeated endpoint occupies the whole main loop: there is no legal
    # internal demand boundary to supply the old regular-scheduling prerequisite.
    for bounds in ((0, 0), (1, 1)):
        description = testing.inspect_scoped_token_families(
            logical, parameter_bounds={4: (0, 0)}, block_bounds=bounds
        )
        assert description["families"]
        for family in description["families"]:
            for key in ("source_domain", "target_domain"):
                assert all(lo == 0 and hi == 12 for lo, hi in family[key])
    # The caller-evidence collector currently forwards launch bounds only to
    # AIC workers. Supply this AIV fixture's explicit launch domain to the same
    # production token-lowering entry point, without changing that collector.
    physical = testing.lower_sync_schedule(
        logical,
        parameter_bounds={4: (0, 0)},
        block_bounds=(0, 1),
        schedule_policy=testing.SyncSchedulePolicy(mode="case", no_clone=False),
    )["function"]
    loops = [
        s
        for s in statements(physical.body)
        if isinstance(s, ir.ForStmt) and isinstance(s.start, ir.ConstInt) and s.start.value == 5
    ]
    assert len(loops) == 2  # One unchanged loop version per incompatible case.
    assert all(s.stop.value == 28 and s.step.value == 2 for s in loops)
    for block in (0, 1):
        check(original, logical, physical, {4: 0}, block=block, minimum_protocol=False)
