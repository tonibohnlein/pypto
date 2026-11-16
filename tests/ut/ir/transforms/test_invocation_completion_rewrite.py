# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The owned-return wrapper changes statement cuts, preserving expression DAGs.

Acc bytes in the supplied descriptor are an input premise. These are command
and rewrite checks, not a new initializedness or numeric matmul qualification.
"""

import pytest
from pypto import ir, passes

from .buffer_test_utils import statements
from .sync_closed_command_oracle import ClosedCommands, opname
from .test_acc_row_trim_proof_context import SPAN, bounded_guard, fixture
from .test_closed_prefix_sync import Services

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("depth", [0, 4, 28])
def test_owned_completion_preserves_shared_guard_and_original_frame(depth):
    guards = []

    def body(row, store):
        guard = bounded_guard(row, 32)
        for _ in range(depth):
            guard = ir.And(guard, guard, ir.DataType.BOOL, SPAN)
        guards.append(guard)
        return [
            *Services().closed(ir.PipeType.MTE2, ir.PipeType.MTE1),
            ir.IfStmt(guard, store(row), None, [], SPAN),
        ]

    function = fixture(body)
    with passes.PassContext([], enable_pypto_autosync=True):
        lowered = passes.lower_sync()(ir.Program([function], "OwnedReturnDag", SPAN))
    actual = lowered.get_function(function.name)
    assert actual is not None and actual.attrs["pypto_autosync"]
    assert all(a.same_as(b) for a, b in zip(actual.params, function.params, strict=True))
    branches = [s for s in statements(lowered) if isinstance(s, ir.IfStmt)]
    assert len(branches) == 1 and branches[0].condition.same_as(guards[0])
    # A false leftmost atom skips the uninitialized payload and evaluates a
    # shared conjunction in linear depth; no expanded-DAG execution is claimed.
    execution = ClosedCommands(actual, {0: 0}, range(6))
    order = execution.run()
    assert not execution.payloads and not execution.live
    assert len(execution.publications) == len(execution.acquisitions) == 2
    assert execution.last_all is not None
    assert (execution.last_all, execution.returns[-1]) in order
    calls = [
        s.expr for s in statements(lowered) if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call)
    ]
    assert sum(call.op.name == opname("system.bar_all") for call in calls) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
