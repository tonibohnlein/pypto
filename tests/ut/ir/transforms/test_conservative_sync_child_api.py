# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public InsertSync consumes immutable provider capabilities transactionally."""

from typing import Any

import pytest
from pypto import ir, passes

from .test_autosync_switch import clone
from .test_conservative_sync_children import exported, loop_scopes, resident_child_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def apply(program, transform, policy="conservative_prefix"):
    with passes.PassContext([], autosync_policy=policy, autosync_disjoint_gm=True):
        return transform(program)


def test_public_child_capability_has_no_mutable_or_fabricated_constructor():
    function, scope = resident_child_fixture()
    result, _ = exported(function, scope)
    child = result["child"]
    assert isinstance(child, passes.NativeSyncChild)
    with pytest.raises(TypeError):
        passes.NativeSyncChild()
    for attribute, value in (("owner", function), ("scope", scope)):
        with pytest.raises(AttributeError):
            setattr(child, attribute, value)


def test_public_insert_retains_accepted_children_by_value_and_original_scope():
    function, scope = resident_child_fixture()
    result, _ = exported(function, scope)
    accepted = [result["child"]]
    transform = passes.insert_sync(accepted_children=accepted)
    accepted.clear()
    program = ir.Program([function], "NativeChild", function.span)
    before = ir.serialize(program)
    logical = apply(program, transform)
    lowered = next(iter(logical.functions.values()))
    assert any(loop is scope for loop in loop_scopes(lowered.body))
    assert ir.serialize(program) == before


def test_public_insert_rejects_exact_policy_and_foreign_original_owner():
    function, scope = resident_child_fixture()
    result, _ = exported(function, scope)
    transform = passes.insert_sync(accepted_children=[result["child"]])
    original = ir.Program([function], "Original", function.span)
    before = ir.serialize(original)
    with pytest.raises(ValueError, match="explicit conservative_prefix policy"):
        apply(original, transform, policy="exact")
    foreign = ir.Program([clone(function)], "Foreign", function.span)
    with pytest.raises(ValueError, match="unique original owner"):
        apply(foreign, transform)
    assert ir.serialize(original) == before


def test_public_insert_routes_multiple_owners_without_conflating_shared_subtrees():
    first, scope = resident_child_fixture()
    second = clone(first, name="second")
    first_export, _ = exported(first, scope)
    second_export, _ = exported(second, scope)
    transform = passes.insert_sync(accepted_children=[first_export["child"], second_export["child"]])
    program = ir.Program([first, second], "TwoOwners", first.span)
    logical = apply(program, transform)
    assert len(logical.functions) == 2
    for function in logical.functions.values():
        assert any(loop is scope for loop in loop_scopes(function.body))


def test_public_insert_refuses_duplicate_and_unavailable_capabilities():
    function, scope = resident_child_fixture()
    result, _ = exported(function, scope)
    program = ir.Program([function], "Original", function.span)
    before = ir.serialize(program)
    with pytest.raises(ValueError, match="duplicate accepted child scopes"):
        apply(program, passes.insert_sync(accepted_children=[result["child"], result["child"]]))
    with pytest.raises(ValueError, match="unavailable child capability"):
        # Deliberately exercise malformed dynamic input at the Python boundary.
        unavailable: list[Any] = [None]
        apply(program, passes.insert_sync(accepted_children=unavailable))
    assert ir.serialize(program) == before


def test_public_insert_program_failure_does_not_publish_partial_child_import():
    function, scope = resident_child_fixture()
    result, _ = exported(function, scope)
    call = ir.Call(ir.get_op("system.bar_all"), [], ir.VoidType(), function.span)
    invalid = clone(
        function,
        name="unsupported",
        body=ir.SeqStmts(
            [
                ir.EvalStmt(call, function.span),
                ir.ReturnStmt([function.params[-1]], function.span),
            ],
            function.span,
        ),
    )
    program = ir.Program([function, invalid], "Rollback", function.span)
    before = ir.serialize(program)
    with pytest.raises(ValueError):
        apply(program, passes.insert_sync(accepted_children=[result["child"]]))
    assert ir.serialize(program) == before
