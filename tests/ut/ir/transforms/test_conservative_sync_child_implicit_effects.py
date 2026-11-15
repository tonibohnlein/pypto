# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Retained native children preserve complete implicit native effects."""

import pytest
from pypto import ir, testing

from .test_autosync_switch import clone
from .test_implicit_sync_effects import independent_broadcasts

pytestmark = pytest.mark.usefixtures("ascend_backend")


def hierarchical_broadcasts(names=("sub", "div"), count=1):
    function, _, scope = independent_broadcasts(names, count=count)
    assert scope is not None
    span = function.span
    index = ir.DataType.INDEX
    outer = ir.ForStmt(
        ir.Var("visit", ir.ScalarType(index), span),
        ir.ConstInt(0, index, span),
        ir.ConstInt(2, index, span),
        ir.ConstInt(1, index, span),
        [],
        ir.SeqStmts([scope, ir.YieldStmt([], span)], span),
        [],
        span,
    )
    assert isinstance(function.body, ir.SeqStmts)
    body = ir.SeqStmts(
        [outer if statement.same_as(scope) else statement for statement in function.body.stmts], span
    )
    # This is the complete fixture's original owner, including outer reset
    # visits. Its retained mathematics is constructed below, never detached.
    return clone(function, body=body), scope


def test_conservative_child_keeps_pending_implicit_effects_and_refuses_open_internal_order():
    function, scope = hierarchical_broadcasts(("sub",), count=1)
    retained = testing.analyze_placed_sync(function, True)
    exported = testing.export_native_sync_child(retained, scope)
    assert exported["status"] == "ready", exported
    baseline = testing.analyze_conservative_sync(function)
    candidate = testing.analyze_conservative_sync(function, children=[exported["child"]])
    assert candidate["exit_effects"] == baseline["exit_effects"]
    assert {(f["space"], f["write"]) for f in candidate["children"][0]["effects"]} == {
        ("Vec", False),
        ("Vec", True),
    }
    for count in (1, 3):
        function, scope = hierarchical_broadcasts(count=count)
        exported = testing.export_native_sync_child(testing.analyze_placed_sync(function, True), scope)
        assert exported["status"] == "nonnative_internal_demands", exported
        assert exported["child"] is None
