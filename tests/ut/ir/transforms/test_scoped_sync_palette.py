# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Publication ranks and resource certificates versus executed SSA token uses."""

from collections import Counter
from itertools import product

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import hardware_reference, interpret
from .sync_physical_oracle import PhysicalExecution, checked_payload_calls
from .test_nested_phase_sync import nested_matrix_fixture
from .test_nested_sync_repetition import BOUNDS, nested_fixture
from .test_scoped_token_families import Uses, occurrence_indices
from .test_token_sync_lowering import _project

pytestmark = pytest.mark.usefixtures("ascend_backend")


def ordinal(digits, frames):
    value = 0
    for digit, (_, count, _, _) in zip(digits, frames, strict=True):
        value = value * count + digit
    return value


@pytest.mark.parametrize("count,visits", [(0, 1), (2, 0), (1, 1), (2, 2), (3, 2), (4, 3)])
def test_palette_ranks_preserve_actual_handoffs(count, visits):
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(count, visits=visits, slices=2), True
    ).realize()["function"]
    check_palette(logical, require_reuse=count >= 2 and visits > 0)


def check_palette(logical, *, require_reuse=False, parameter_bounds=None, parameters=None):
    parameter_bounds = parameter_bounds or {}
    parameters = parameters or {}
    inventory = testing.inspect_scoped_token_families(logical, parameter_bounds=parameter_bounds)
    _, calls = interpret(inventory["function"], parameters)
    vertices = hardware_reference(inventory["function"], calls)[0]
    executed = Uses(calls, vertices)
    executed.execute(logical.body, {logical.params[i]: value for i, value in parameters.items()})
    _, closure = _project(executed.vertices, executed.edges, vertices)
    occurrences = occurrence_indices(inventory, calls)
    order = testing.analyze_scoped_token_order(logical, parameter_bounds=parameter_bounds)
    cut_order = {use: j for _, _, _, after in inventory["protocol_cuts"] for j, use in enumerate(after)}
    by_direction = {}
    all_uses = Counter()
    for raw, family in enumerate(inventory["families"]):
        sr, sv, ss = family["source"]
        tr, tv, ts = family["target"]
        for digits in product(*(range(a[1]) for a in family["axes"])):
            a = occurrences[sr, sv + sum(d * x[2] for d, x in zip(digits, family["axes"], strict=True)), ss]
            b = occurrences[tr, tv + sum(d * x[3] for d, x in zip(digits, family["axes"], strict=True)), ts]
            all_uses[a, b] += 1
            direction = calls[a][2], calls[b][2]
            if direction[0] != direction[1]:
                by_direction.setdefault(direction, []).append((a, cut_order[raw], raw, b))
    assert all_uses == executed.uses
    certified_nontrivial = 0
    for direction, publications in by_direction.items():
        expected = sorted(publications)
        for budget in (1, 2, 4, 6):
            result = order.palette(*direction, budget)
            assert result["status"] in ("certified", "unproved-reuse"), result
            assert result["publications"] == len(expected)
            actual = []
            for piece in result["pieces"]:
                family = inventory["families"][piece["raw_use"]]
                sr, _, ss = family["source"]
                tr, _, ts = family["target"]
                for digits in product(*(range(d[2]) for d in piece["source"])):
                    source = [
                        first + step * d for (first, step, _), d in zip(piece["source"], digits, strict=True)
                    ]
                    target = [
                        first + (step * digits[i] if size > 1 else 0)
                        for i, (first, step, size) in enumerate(piece["target"])
                    ]
                    rank = piece["rank_base"] + sum(
                        d * c for d, c in zip(digits, piece["coefficients"], strict=True)
                    )
                    a = occurrences[sr, ordinal(source, inventory["regions"][sr]["frames"]), ss]
                    b = occurrences[tr, ordinal(target, inventory["regions"][tr]["frames"]), ts]
                    actual.append((rank, a, cut_order[piece["raw_use"]], piece["raw_use"], b))
            assert sorted(actual) == [(rank, *p) for rank, p in enumerate(expected)]
            if result["status"] == "certified":
                certified_nontrivial += len(expected) > budget
                assert all(
                    ((h[3], True), (g[0], False)) in closure for h, g in zip(expected, expected[budget:])
                )

    if require_reuse:
        assert certified_nontrivial > 0


def test_palette_rejects_invalid_resource_questions():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(2, slices=2), True).realize()["function"]
    order = testing.analyze_scoped_token_order(logical)
    assert order.palette(ir.PipeType.MTE2, ir.PipeType.MTE1, 0)["status"] == "unsupported-grammar"
    assert order.palette(ir.PipeType.MTE2, ir.PipeType.MTE2, 2)["status"] == "unsupported-grammar"


def test_palette_grammar_does_not_unfold_trips():
    summaries = []
    for count, visits in ((100, 3), (10000, 3), (100, 10000)):
        logical = testing.analyze_placed_sync(
            nested_matrix_fixture(count, visits=visits, slices=2), True
        ).realize()["function"]
        order = testing.analyze_scoped_token_order(logical)
        result = order.palette(ir.PipeType.MTE1, ir.PipeType.M, 6)
        assert result["status"] in ("certified", "unproved-reuse"), result
        summaries.append((result["grammar_nodes"], result["phase_visits"], len(result["pieces"])))
    assert summaries[0] == summaries[1] == summaries[2]


def test_palette_preserves_overlapping_residue_uses_and_holes():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(2, visits=7, slices=2), True).realize()[
        "function"
    ]

    class ResidueUses(ir.IRMutator):
        frame = None

        def visit_for_stmt(self, op):
            saved = self.frame
            if op.loop_var.name_hint == "visit":
                self.frame = op.loop_var
            result = super().visit_for_stmt(op)
            self.frame = saved
            return result

        def visit_eval_stmt(self, op):
            if (
                self.frame is None
                or not isinstance(op.expr, ir.Call)
                or op.expr.op.name != ir.get_op("sync.wait").name
            ):
                return op
            branches = []
            for period in (2, 3):
                divisor = ir.ConstInt(period, ir.DataType.INDEX, op.span)
                zero = ir.ConstInt(0, ir.DataType.INDEX, op.span)
                normalized = ir.Sub(self.frame, zero, ir.DataType.INDEX, op.span)
                residue = ir.FloorMod(normalized, divisor, ir.DataType.INDEX, op.span)
                guard = ir.Eq(residue, zero, ir.DataType.BOOL, op.span)
                branches.append(ir.IfStmt(guard, op, None, [], op.span))
            return ir.SeqStmts(branches, op.span)

    check_palette(ResidueUses().visit_function(logical))


def test_palette_reuses_schedule_and_rejects_foreign_protocol():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(3, visits=2, slices=2), True).realize()[
        "function"
    ]
    order = testing.analyze_scoped_token_order(logical)
    schedule = order.publication_schedule(ir.PipeType.MTE1, ir.PipeType.M)
    assert schedule.publications > 0
    pieces = None
    accepted = False
    for budget in (1, 2, 4, 6):
        result = schedule.certify(order, budget)
        assert result["status"] in ("certified", "unproved-reuse")
        assert result["grammar_nodes"] == result["phase_visits"] == result["domain_visits"] == 0
        assert result["routing_entries"] == 0
        if pieces is None:
            pieces = result["pieces"]
        assert pieces == result["pieces"]
        accepted |= result["status"] == "certified"
    assert accepted
    foreign = testing.analyze_scoped_token_order(logical)
    refusal = schedule.certify(foreign, 6)
    assert refusal["status"] == "unsupported-grammar"
    assert "different immutable token protocol" in refusal["reason"]


@pytest.mark.parametrize("depth", [2, 4, 8, 12, 16])
def test_successor_palette_avoids_cartesian_nested_boundary_cells(depth):
    original = nested_fixture([2] * depth)
    logical = testing.analyze_placed_sync(original, True, parameter_bounds=BOUNDS).realize()["function"]
    order = testing.analyze_scoped_token_order(logical, parameter_bounds=BOUNDS)
    result = order.palette(ir.PipeType.MTE3, ir.PipeType.MTE2, 1)
    assert result["status"] == "certified", result
    assert result["publications"] == 2**depth - 1
    assert len(result["pieces"]) == depth
    assert result["grammar_nodes"] == result["phase_visits"] == result["routing_entries"] == 0
    # The public one-shot receipt reports certification transition queries.
    # Construction prequalification uses the same queries separately.
    assert result["transition_queries"] == depth
    # Public LowerSync consumes these same original tokens and their IDs. The
    # output must not have one literal SET for every dynamic publication.
    physical = testing.lower_sync_schedule(logical, parameter_bounds=BOUNDS, eligible_ids=list(range(6)))
    assert physical["set_sites"] <= 2 * depth + 8
    assert physical["wait_sites"] <= 2 * depth + 8


@pytest.mark.parametrize("counts", [[2, 3], [3, 2, 2], [1, 2, 1, 2]])
def test_successor_palette_ranks_labels_matching_and_reuse_independently(counts):
    original = nested_fixture(counts)
    logical = testing.analyze_placed_sync(original, True, parameter_bounds=BOUNDS).realize()["function"]
    parameters = {1: 0, 3: 64}
    # Expands actual token uses, every emitted affine rank and all raw labels;
    # compares ranks with concrete original occurrences independently of cells.
    check_palette(logical, parameter_bounds=BOUNDS, parameters=parameters, require_reuse=True)
    physical = testing.lower_sync_schedule(logical, parameter_bounds=BOUNDS, eligible_ids=list(range(6)))
    emitted = physical["function"]
    calls, emitted_calls = checked_payload_calls(original, emitted, parameters)
    reference = hardware_reference(original, calls)
    tokens = Uses(calls, reference[0])
    tokens.execute(logical.body, {logical.params[i]: value for i, value in parameters.items()})
    flags = PhysicalExecution(emitted_calls, reference[0], range(6))
    flags.execute(emitted.body, {emitted.params[i]: value for i, value in parameters.items()})
    before, token_order = _project(tokens.vertices, tokens.edges, reference[0])
    after, flag_order = _project(flags.vertices, flags.edges, reference[0])
    assert token_order == flag_order
    assert reference[2] <= flag_order
    assert not tokens.pending or not any(tokens.pending.values())
    assert not flags.live and not any(flags.pending.values())
    assert len(flags.records) == len(flags.waits)
    assert all(pair in after for pair in flags.reuses)
    assert all(ids <= set(range(6)) for ids in flags.ids.values())
    assert not any(a == b for a, b in before | after)


def test_successor_palette_retains_original_frames_as_counts_grow():
    sizes = []
    for count in [2, 17, 1000000]:
        logical = testing.analyze_placed_sync(
            nested_fixture([count, 3, 2]), True, parameter_bounds=BOUNDS
        ).realize()["function"]
        order = testing.analyze_scoped_token_order(logical, parameter_bounds=BOUNDS)
        result = order.palette(ir.PipeType.MTE3, ir.PipeType.MTE2, 1)
        assert result["status"] == "certified", result
        assert result["publications"] == count * 6 - 1
        sizes.append((len(result["pieces"]), result["grammar_nodes"], result["phase_visits"]))
    assert sizes == [(3, 0, 0)] * 3


@pytest.mark.parametrize("kind", ["duplicate", "hole", "missing_reuse_path"])
def test_successor_normalization_retains_general_grammar_for_unproved_shapes(kind):
    logical = testing.analyze_placed_sync(nested_fixture([2, 3]), True, parameter_bounds=BOUNDS).realize()[
        "function"
    ]

    class AlterUses(ir.IRMutator):
        frame = None
        changed = 0

        def visit_for_stmt(self, op):
            saved = self.frame
            self.frame = op
            result = super().visit_for_stmt(op)
            self.frame = saved
            return result

        def visit_eval_stmt(self, op):
            call = op.expr
            if not isinstance(call, ir.Call) or call.op.name != ir.get_op("sync.wait").name:
                return op
            token = call.args[0].type
            assert isinstance(token, ir.SyncTokenType)
            destination = ir.PipeType(call.kwargs["dst_pipe"])
            if kind == "missing_reuse_path":
                if token.pipe == ir.PipeType.MTE2 and destination == ir.PipeType.MTE3:
                    self.changed += 1
                    return ir.SeqStmts([], op.span)
                return op
            if token.pipe != ir.PipeType.MTE3 or destination != ir.PipeType.MTE2:
                return op
            if kind == "duplicate":
                self.changed += 1
                return ir.SeqStmts([op, op], op.span)
            if self.frame is None:
                return op
            self.changed += 1
            frame = self.frame
            assert isinstance(frame.start, ir.ConstInt) and isinstance(frame.step, ir.ConstInt)
            # Express the same alternating original-occurrence domain through
            # the supported direct-IV residue syntax. With step=3 this selects
            # IVs 2 and 8 while excluding IV 5, preserving an actual hole.
            modulus = 2 * frame.step.value
            residue = ir.FloorMod(
                ir.Sub(frame.loop_var, frame.start, ir.DataType.INDEX, op.span),
                ir.ConstInt(modulus, ir.DataType.INDEX, op.span),
                ir.DataType.INDEX,
                op.span,
            )
            guard = ir.Eq(
                residue,
                ir.ConstInt(0, ir.DataType.INDEX, op.span),
                ir.DataType.BOOL,
                op.span,
            )
            return ir.IfStmt(guard, op, None, [], op.span)

    altered = AlterUses()
    actual = altered.visit_function(logical)
    assert altered.changed
    order = testing.analyze_scoped_token_order(actual, parameter_bounds=BOUNDS)
    result = order.palette(ir.PipeType.MTE3, ir.PipeType.MTE2, 1)
    assert result["grammar_nodes"] > 0, "inapplicable compact normalization must retain the original grammar"
    if kind == "hole":
        assert 0 < result["publications"] < 5, "the residue must remove a proper subset of five successors"
    if kind == "missing_reuse_path":
        # This deliberately under-synchronized token program need not cover
        # buffer hazards. Lowering must not invent its missing readiness edge
        # to justify a resource assignment; actual-token order is its owner.
        assert result["status"] == "unproved-reuse"
    else:
        check_palette(actual, parameter_bounds=BOUNDS, parameters={1: 0, 3: 64})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
