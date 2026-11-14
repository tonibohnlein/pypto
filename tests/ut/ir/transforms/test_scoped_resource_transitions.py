# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Uniform resource questions checked against independently executed token IR."""

from itertools import product
from math import prod

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import hardware_reference, interpret
from .test_nested_phase_sync import nested_matrix_fixture
from .test_scoped_token_families import Uses, occurrence_indices
from .test_token_sync_lowering import _project

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("count,visits", [(2, 2), (3, 3)])
def test_uniform_resource_rectangles_are_sound(count, visits):
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(count, visits=visits, slices=2), True
    ).realize()["function"]
    inventory = testing.inspect_scoped_token_families(logical)
    original = inventory["function"]
    _, calls = interpret(original)
    vertices = hardware_reference(original, calls)[0]
    executed = Uses(calls, vertices)
    executed.execute(logical.body, {})
    _, closure = _project(executed.vertices, executed.edges, vertices)
    occurrences = occurrence_indices(inventory, calls)
    order = testing.analyze_scoped_token_order(logical)
    owner_frames = order.region_frames
    accepted = rejected = 0
    # All static endpoint pairs, not only relations supplied by the parser.
    # Each range varies one original frame and fixes the other prefix frames;
    # suffixes stay in the occurrence bases. Step two tests activity holes.
    endpoints = sorted(
        (r, s) for r, region in enumerate(inventory["regions"]) for s in range(len(region["sites"]))
    )
    for (sr, ss), (tr, ts) in product(endpoints, repeat=2):
        sf, tf = owner_frames[sr], owner_frames[tr]
        sc = [item[1] for item in inventory["regions"][sr]["frames"]]
        tc = [item[1] for item in inventory["regions"][tr]["frames"]]
        common = 0
        while common < min(len(sf), len(tf)) and sf[common].same_as(tf[common]):
            common += 1
        for varying in range(common):
            for step in (1, 2):
                length = min(2, (min(sc[varying], tc[varying]) - 1) // step + 1)
                if length < 2:
                    continue
                axes = []
                for i in range(varying + 1):
                    delta = step if i == varying else 1
                    axes.append(
                        (
                            sf[i],
                            length if i == varying else 1,
                            prod(sc[i + 1 :]) * delta,
                            prod(tc[i + 1 :]) * delta,
                            delta,
                            delta,
                        )
                    )
                source, target = (sr, 0, ss), (tr, 0, ts)
                expected = []
                for digits in product(*(range(axis[1]) for axis in axes)):
                    a = sum(d * axis[2] for d, axis in zip(digits, axes, strict=True))
                    b = sum(d * axis[3] for d, axis in zip(digits, axes, strict=True))
                    ai, bi = occurrences[sr, a, ss], occurrences[tr, b, ts]
                    expected.append(((ai, True), (bi, False)) in closure)
                answer = order.prove_transition(source, target, axes)
                assert not answer or all(expected), (source, target, axes, expected)
                accepted += answer
                rejected += not answer
                if expected[0] and not all(expected):
                    assert not answer
                malformed = list(axes)
                frame, size, a, b, astep, bstep = malformed[-1]
                malformed[-1] = (frame, size, a, b, astep + 1, bstep)
                with pytest.raises(ValueError, match="transition step"):
                    order.prove_transition(source, target, malformed)
    assert accepted > 0 and rejected > 0
    # A separately parsed owner has structurally equal loops, not the same
    # original frame identities. It cannot supply a certificate for this owner.
    for r, frames in enumerate(owner_frames):
        if not frames:
            continue
        size = prod(item[1] for item in inventory["regions"][r]["frames"][1:])
        foreign = inventory["regions"][r]["frames"][0][0]
        with pytest.raises(ValueError, match="incompatible axes"):
            order.prove_transition((r, 0, 0), (r, 0, 0), [(foreign, 1, size, size, 1, 1)])
        count = inventory["regions"][r]["frames"][0][1]
        with pytest.raises(ValueError, match="incompatible axes"):
            order.prove_transition((r, 0, 0), (r, 0, 0), [(frames[0], count + 1, size, size, 1, 1)])
        break


def test_single_visit_edge_is_not_a_uniform_reset_proof():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(2, visits=3, slices=2), True).realize()[
        "function"
    ]

    class FirstVisitWaits(ir.IRMutator):
        """Restrict actual waits, leaving later publications without that order."""

        visit = None

        def visit_for_stmt(self, op):
            saved = self.visit
            if op.loop_var.name_hint == "visit":
                self.visit = op.loop_var
            result = super().visit_for_stmt(op)
            self.visit = saved
            return result

        def visit_eval_stmt(self, op):
            if (
                self.visit is not None
                and isinstance(op.expr, ir.Call)
                and op.expr.op.name == ir.get_op("sync.wait").name
            ):
                zero = ir.ConstInt(0, ir.DataType.INDEX, op.span)
                guard = ir.Eq(self.visit, zero, ir.DataType.BOOL, op.span)
                return ir.IfStmt(guard, op, None, [], op.span)
            return op

    logical = FirstVisitWaits().visit_function(logical)
    inventory = testing.inspect_scoped_token_families(logical)
    _, calls = interpret(inventory["function"])
    vertices = hardware_reference(inventory["function"], calls)[0]
    executed = Uses(calls, vertices)
    executed.execute(logical.body, {})
    _, closure = _project(executed.vertices, executed.edges, vertices)
    occurrences = occurrence_indices(inventory, calls)
    order = testing.analyze_scoped_token_order(logical)
    checked = 0
    for family in inventory["families"]:
        sr, sv, ss = family["source"]
        tr, tv, ts = family["target"]
        sf, tf = order.region_frames[sr], order.region_frames[tr]
        if not sf or not tf or not sf[0].same_as(tf[0]) or sf[0].loop_var.name_hint != "visit":
            continue
        a = prod(item[1] for item in inventory["regions"][sr]["frames"][1:])
        b = prod(item[1] for item in inventory["regions"][tr]["frames"][1:])
        first = ((occurrences[sr, sv, ss], True), (occurrences[tr, tv, ts], False))
        later = ((occurrences[sr, sv + a, ss], True), (occurrences[tr, tv + b, ts], False))
        if first not in closure or later in closure:
            continue
        assert order.prove_transition(family["source"], family["target"], [])
        assert not order.prove_transition(family["source"], family["target"], [(sf[0], 2, a, b, 1, 1)])
        checked += 1
    assert checked > 0
