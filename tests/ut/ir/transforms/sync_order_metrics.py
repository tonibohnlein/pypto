# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Draft prerequisite profiles from independently constructed payload closures.

Boolean event kind is False for start and True for completion. This module
consumes graphs, not compiler demand families or counts of inserted commands.
"""

from collections import Counter
from typing import Any


def completion_start_pairs(order):
    """Project a payload closure onto completion-to-start prerequisites."""
    return {(a[0], b[0]) for a, b in order if a[1] is True and b[1] is False}


def prerequisite_profile(pipes, order):
    """Return sparse K(b,p), checking the native completion-prefix premise."""
    counts = Counter()
    ranks = []
    for pipe in pipes:
        counts[pipe] += 1
        ranks.append(counts[pipe])
    pairs = completion_start_pairs(order)
    profile = {}
    reached = Counter()
    for a, b in pairs:
        key = (b, pipes[a])
        profile[key] = max(profile.get(key, 0), ranks[a])
        reached[key] += 1
    assert all(reached[key] == rank for key, rank in profile.items()), "non-prefix prerequisite graph"
    return profile


def compare_order(pipes, required, before, after) -> dict[str, Any]:
    """Report coverage, excess and dominance, including witnesses both ways.

    The sum of profile differences equals the number of extra C-to-I pairs
    only under the checked native completion-prefix premise. Full payload-event
    coverage is checked separately so missing I-to-I/C-to-C order is not hidden.
    """
    reference = completion_start_pairs(required)
    old = completion_start_pairs(before)
    new = completion_start_pairs(after)
    profiles = [prerequisite_profile(pipes, graph) for graph in (required, before, after)]
    removed, added = old - new, new - old
    relation = (
        "equal"
        if not added and not removed
        else ("after-dominates" if not added else "before-dominates" if not removed else "incomparable")
    )
    result: dict[str, Any] = {"relation": relation, "removed": sorted(removed), "added": sorted(added)}
    for name, graph, pairs, profile in zip(("before", "after"), (before, after), (old, new), profiles[1:]):
        missing = required - graph
        result[name] = {
            "coverage": not missing,
            "missing_event_pairs": sorted(missing),
            "excess_completion_start": len(pairs - reference),
            "profile": [(b, int(p), rank) for (b, p), rank in sorted(profile.items())],
        }
        if not missing:
            assert sum(profile.values()) - sum(profiles[0].values()) == len(pairs - reference)
    return result
