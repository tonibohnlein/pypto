# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ordering quality is not inferred from command or excess counts alone."""

import pytest

from .sync_order_metrics import compare_order, prerequisite_profile


def edge(a, b):
    return ((a, True), (b, False))


def test_equal_excess_can_hide_incomparable_profiles():
    # Two barrier placements on A;X;B: early adds A->X, late adds X->B.
    required = {edge(0, 2)}
    early = required | {edge(0, 1)}
    late = required | {edge(1, 2)}
    result = compare_order([0, 0, 0], required, early, late)
    assert result["relation"] == "incomparable"
    assert result["before"]["excess_completion_start"] == 1
    assert result["after"]["excess_completion_start"] == 1
    assert result["added"] == [(1, 2)]
    assert result["removed"] == [(0, 1)]


def test_dominance_equality_missing_coverage_and_empty_execution():
    required = {edge(0, 2)}
    extra = required | {edge(1, 2)}
    assert compare_order([0, 0, 1], required, extra, required)["relation"] == "after-dominates"
    assert compare_order([0, 0, 1], required, required, required)["relation"] == "equal"
    assert not compare_order([0, 0, 1], required, required, set())["after"]["coverage"]
    assert compare_order([], set(), set(), set())["after"]["excess_completion_start"] == 0


def test_profile_requires_native_prefix_closure():
    with pytest.raises(AssertionError, match="non-prefix"):
        prerequisite_profile([0, 0, 1], {edge(1, 2)})
