# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent event-graph oracle for the finite numerical synchronization service."""

import itertools
import random

import pytest
from pypto import ir, testing

PIPES = [
    ir.PipeType.MTE1,
    ir.PipeType.MTE2,
    ir.PipeType.MTE3,
    ir.PipeType.M,
    ir.PipeType.V,
    ir.PipeType.S,
    ir.PipeType.FIX,
]


def analyze(ops, cells=(0, 1, 2), prerequisites=(), native=(), queries=(), boundary_cells=()):
    return testing.analyze_finite_sync(
        ops, list(cells), True, list(prerequisites), list(native), list(queries), list(boundary_cells)
    )


def closure(vertices, edges):
    """Graph traversal independent of rank rows and storage-frontier generation."""
    adjacency = {v: set() for v in vertices}
    for a, b in edges:
        adjacency[a].add(b)
    result = set()
    for source in vertices:
        pending = list(adjacency[source])
        seen = set()
        while pending:
            target = pending.pop()
            if target not in seen:
                seen.add(target)
                pending.extend(adjacency[target] - seen)
        result.update((source, target) for target in seen)
    return result


def reference(ops, prerequisites=(), native=()):
    """Enumerate all actual conflicts, never the implementation's generators."""
    active = [i for i, op in enumerate(ops) if op[1]]
    vertices = [(i, kind) for i in active for kind in (False, True)]
    base = {((i, False), (i, True)) for i in active}
    for pipe in PIPES:
        on_pipe = [i for i in active if ops[i][0] == pipe]
        for a, b in zip(on_pipe, on_pipe[1:]):
            base.update([((a, False), (b, False)), ((a, True), (b, True))])
    base.update(((a, True), (b, False)) for a, b in native if a in active and b in active)
    required = set(base)
    for a, b in itertools.combinations(active, 2):
        ar, aw, br, bw = (set(x) for x in (ops[a][2], ops[a][3], ops[b][2], ops[b][3]))
        if aw & (br | bw) or ar & bw:
            required.add(((a, True), (b, False)))
    required.update(((a, True), (b, False)) for a, b in prerequisites if a in active and b in active)
    order = closure(vertices, required)
    covers = {
        (a[0], b[0])
        for a, b in order - base
        if not any((a, middle) in order and (middle, b) in order for middle in vertices)
    }
    return vertices, base, order, covers


def check_case(ops, prerequisites=(), native=(), cells=(0, 1, 2)):
    queries = list(itertools.product(range(len(ops)), (False, True), range(len(ops)), (False, True)))
    result = analyze(ops, cells, prerequisites, native, queries, cells)
    vertices, base, order, covers = reference(ops, prerequisites, native)
    assert set(result["demands"]) == covers
    assert len(result["demands"]) == len(covers)
    for name in ("generators", "demands"):
        graph = base | {((a, True), (b, False)) for a, b in result[name]}
        assert closure(vertices, graph) == order
    assert result["queries"] == [((a, ac), (b, bc)) in order for a, ac, b, bc in queries]
    for cell, boundary in zip(cells, result["boundaries"]):
        assert boundary is not None
        writers = [i for i, op in enumerate(ops) if op[1] and cell in op[3]]
        assert boundary["first_writer"] == (writers[0] if writers else None)
        assert boundary["last_writer"] == (writers[-1] if writers else None)
        for pipe in PIPES:
            readers = [
                i
                for i, op in enumerate(ops)
                if op[1] and op[0] == pipe and cell in op[2] and cell not in op[3]
            ]
            before = [i for i in readers if not writers or i < writers[0]]
            after = [i for i in readers if not writers or i > writers[-1]]
            assert boundary["first_readers"][int(pipe)] == (before[0] if before else None)
            assert boundary["last_readers"][int(pipe)] == (after[-1] if after else None)
    return result


def test_independent_random_event_graphs():
    rng = random.Random(71603)
    for _ in range(500):
        ops = [
            (
                rng.choice(PIPES),
                rng.randrange(4) != 0,
                [c for c in range(3) if rng.randrange(3) == 0],
                [c for c in range(3) if rng.randrange(3) == 0],
            )
            for _ in range(rng.randrange(11))
        ]
        pairs = list(itertools.combinations(range(len(ops)), 2))
        prerequisites = [p for p in pairs if rng.randrange(12) == 0]
        native = [p for p in pairs if rng.randrange(12) == 0]
        check_case(ops, prerequisites, native)


def test_native_completion_is_not_a_same_pipe_barrier():
    ops = [(ir.PipeType.S, True, [], [0]), (ir.PipeType.S, True, [0], [])]
    assert check_case(ops)["demands"] == [(0, 1)]
    # Scalar and other native exemptions enter as qualified evidence, never
    # as an assumption that all payloads on one pipe are serialized.
    assert check_case(ops, native=[(0, 1)])["demands"] == []


def test_cross_cell_transitivity_and_rmw():
    ops = [(PIPES[0], True, [], [0]), (PIPES[1], True, [0], [0, 1]), (PIPES[2], True, [0, 1], [])]
    result = check_case(ops)
    assert set(result["demands"]) == {(0, 1), (1, 2)}
    assert (0, 2) not in result["demands"]


def test_coverage_absence_and_duplicate_accesses():
    ops = [(PIPES[0], True, [0, 0], [0, 0]), (PIPES[1], False, [1], [1])]
    result = analyze(ops, cells=[0, 1, 2], boundary_cells=[0, 1, 2, 99])
    for boundary, expected in zip(result["boundaries"][:3], [0, None, None]):
        assert boundary is not None
        assert boundary["first_writer"] == expected
    assert result["boundaries"][3] is None
    assert result["work"]["cell_visits"] == 1
    with pytest.raises(ValueError, match="outside covered storage"):
        analyze(ops, cells=[0])


@pytest.mark.parametrize("edges", [[(0, 0)], [(1, 0)], [(0, 2)]])
@pytest.mark.parametrize("kind", ["prerequisites", "native"])
def test_invalid_forward_contract(edges, kind):
    ops = [(PIPES[0], True, [], []), (PIPES[1], True, [], [])]
    with pytest.raises(ValueError, match="valid forward occurrence"):
        analyze(ops, **{kind: edges})


def test_invalid_pipe_and_query():
    with pytest.raises(ValueError, match="concrete payload pipe"):
        analyze([(ir.PipeType.ALL, True, [], [])])
    with pytest.raises(ValueError, match="query occurrence"):
        analyze([], queries=[(0, False, 0, True)])


@pytest.mark.parametrize("presence", tuple(itertools.product((False, True), repeat=4)))
def test_resolved_guards_preserve_original_indices(presence):
    ops = [
        (PIPES[i % 3], active, [0] if i % 2 else [], [0] if not i % 2 else [])
        for i, active in enumerate(presence)
    ]
    check_case(ops)


@pytest.mark.parametrize("count", [64, 256, 1024, 4096])
def test_read_heavy_inspected_work_is_linear(count):
    ops = [(PIPES[0], True, [], [0])] + [(PIPES[1], True, [0], [])] * count
    ops += [(PIPES[0], True, [], [0])]
    result = analyze(ops)
    assert result["work"]["cell_visits"] == count + 2
    assert result["work"]["reader_visits"] == count
    assert result["work"]["witnesses"] == 2 * count + 1
    assert result["work"]["candidate_tests"] == 2 * count + 1
    assert result["demands"] == [(0, 1), (count, count + 1)]
    assert len(result["starts"]) == count + 2
    assert all(len(row) == 14 for row in result["starts"])


def test_read_only_has_no_pair_work():
    result = analyze([(PIPES[i % 3], True, [0], []) for i in range(4096)])
    assert result["work"]["reader_visits"] == 0
    assert result["work"]["candidate_tests"] == 0
    assert result["demands"] == []


def test_conservative_model_is_not_promoted_to_exact():
    result = testing.analyze_finite_sync(
        [(PIPES[0], True, [], [0]), (PIPES[1], True, [0], [])], [0], False, [], [], [], []
    )
    assert result["exact_accesses"] is False
    assert result["demands"] == [(0, 1)]


def test_typed_views_use_common_physical_bytes():
    # A 2-element float32 store overlaps an int16 view starting at byte 6.
    # The independent oracle sees byte sets, not SSA roots or equal indices.
    ops = [
        (PIPES[0], True, [], list(range(8))),
        (PIPES[1], True, list(range(6, 10)), []),
        (PIPES[2], True, list(range(10, 12)), []),
    ]
    result = check_case(ops, cells=tuple(range(12)))
    assert result["demands"] == [(0, 1)]


def test_region_boundaries_reconstruct_crossings():
    ops = [
        (PIPES[0], True, [], [0]),
        (PIPES[1], True, [0], []),
        (PIPES[2], False, [], [0]),
        (PIPES[2], True, [0], []),
        (PIPES[0], True, [], [0]),
        (PIPES[1], True, [0], []),
    ]
    flat = check_case(ops)
    for cut in range(len(ops) + 1):
        left = analyze(ops[:cut], boundary_cells=[0])
        right = analyze(ops[cut:], boundary_cells=[0])
        a, b = left["boundaries"][0], right["boundaries"][0]
        assert a is not None and b is not None
        bridges = set()

        def bridge(source, target):
            if source is not None and target is not None:
                bridges.add((source, cut + target))

        bridge(a["last_writer"], b["first_writer"])
        for pipe in PIPES:
            bridge(a["last_writer"], b["first_readers"][int(pipe)])
            bridge(a["last_readers"][int(pipe)], b["first_writer"])
        demands = left["demands"] + [(cut + x, cut + y) for x, y in right["demands"]]
        # Feed only child demands and boundary crossings into the reducer.
        # This checks the interface without reusing the flat storage scan.
        composed = analyze(
            [(p, present, [], []) for p, present, _, _ in ops], prerequisites=demands + sorted(bridges)
        )
        assert set(composed["demands"]) == set(flat["demands"])
        assert composed["starts"] == flat["starts"]
        assert composed["completions"] == flat["completions"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
