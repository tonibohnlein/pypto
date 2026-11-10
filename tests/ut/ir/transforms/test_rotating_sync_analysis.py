# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Periodic contracts checked against original physical access execution."""

import math
import random

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import validate
from .test_slot_sync_import import slot_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def rotating_fixture(
    count, families, schedule, start=0, step=1, stream_input=False, suffix="", stream_output=False
):
    """Uniform Vec families, original modular accesses, no compiler generators."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32
    ty = ir.BufferType([1, 32], fp, ir.MemorySpace.Vec)

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(value) for value in values], span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    extent = max(1, start + count * step)
    a = ir.Var("a", ir.TensorType([extent if stream_input else 1, 32], fp), span)
    out = ir.Var("out", ir.TensorType([extent if stream_output else 1, 32], fp), span)
    groups, definitions, address = [], [], 0
    for family, (slots, _) in enumerate(families):
        group_type = ir.MultiBufferType(ty, slots)
        group = ir.Var(f"ring_{family}", group_type, span)
        groups.append(group)
        definitions.append(
            ir.AssignStmt(group, call("buffer.alloc_multi", [tup([]), ci(address)], group_type), span)
        )
        address += slots * 128
    iv = ir.Var(f"i{suffix}", ir.ScalarType(index), span)
    body = []
    for site, (operation, family, phase) in enumerate(schedule):
        slots, stride = families[family]
        selector = ir.FloorMod(
            ir.Add(ir.Mul(iv, ci(stride), index, span), ci(phase), index, span), ci(slots), index, span
        )
        tile = ir.Var(f"tile_{site}{suffix}", ty, span)
        body.append(ir.AssignStmt(tile, call("buffer.slot", [groups[family], selector], ty), span))
        if operation == "load":
            offsets = ir.MakeTuple([iv, ci(0)], span) if stream_input else tup([0, 0])
            payload = call("buffer.load", [a, offsets, tup([1, 32]), tile])
        elif operation == "store":
            offsets = ir.MakeTuple([iv, ci(0)], span) if stream_output else tup([0, 0])
            payload = call("buffer.store", [tile, offsets, tup([1, 32]), out])
        else:
            assert operation == "rmw"
            payload = call("buffer.add", [tile, tile, tile])
        body.append(ir.EvalStmt(payload, span))
    body.append(ir.YieldStmt([], span))
    loop = ir.ForStmt(
        iv, ci(start), ci(start + count * step), ci(step), [], ir.SeqStmts(body, span), [], span
    )
    return ir.Function(
        "rotating",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts([*definitions, loop, ir.ReturnStmt([out], span)], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
@pytest.mark.parametrize(("slots", "stride", "phase"), [(1, 0, 0), (3, 1, 2), (6, 2, 1), (7, 0, 2)])
def test_original_slot_accesses_covers_queries_and_selectors(count, slots, stride, phase):
    function = slot_fixture(count, slots, stride, phase, start=1, step=3, alias=True)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    if slots > 1:
        assert result.boundary(2, 0, 0) is None
    for slot in [-1, slots]:
        assert result.boundary_slot(2, slot, 0, 0) is None
    assert result.boundary_slot(2, 0, 1, 0) is None
    if not count:
        with pytest.raises(ValueError, match="occurrence"):
            result.precedes((0, 0, 0), False, (0, 0, 0), True)


def test_independent_random_modular_accesses():
    rng = random.Random(71011)
    for _ in range(48):
        families = [(rng.randint(1, 7), rng.randint(0, 4)) for _ in range(rng.randint(1, 2))]
        schedule = [
            (rng.choice(["load", "store", "rmw"]), rng.randrange(len(families)), rng.randint(0, 9))
            for _ in range(rng.randint(2, 6))
        ]
        function = rotating_fixture(rng.randint(0, 7), families, schedule)
        validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_coprime_banks_do_not_require_joint_period(count):
    function = rotating_fixture(
        count,
        [(3, 1), (5, 2)],
        [("load", 0, 0), ("load", 1, 1), ("rmw", 0, 1), ("store", 1, 2), ("store", 0, 2)],
    )
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("axis", ["trips", "slots"])
def test_large_binary_counts_keep_frontier_representation_compact(axis):
    sizes = []
    for count, slots in (
        [(7, 3), (10000, 3), (1000000, 3)] if axis == "trips" else [(7, 3), (7, 101), (7, 1000003)]
    ):
        result = testing.analyze_placed_sync(slot_fixture(count, slots), True)
        description = result.describe()
        region = description["regions"][0]
        assert region["distance_entries"] == 0
        assert region["retained_frontier_entries"] <= 4 * 7 * len(region["sites"])
        assert region["shortest_path_runs"] <= 2 * 7
        sizes.append((len(region["sites"]), len(region["generators"]), region["retained_frontier_entries"]))
        # On-demand selection at a large slot uses arithmetic, not slot expansion.
        assert result.boundary_slot(2, slots - 1, 0, 0) is not None
    assert len(set(sizes)) == 1


def test_duplicate_rmw_witnesses_and_read_only_orbits():
    for count in [0, 1, 5, 8]:
        function = rotating_fixture(
            count, [(6, 2)], [("store", 0, 1), ("rmw", 0, 0), ("load", 0, 2), ("store", 0, 4)]
        )
        result = testing.analyze_placed_sync(function, True)
        validate(function, result)
        assert result.boundary_slot(2, 1, 0, 0)["first_writer"] is None


def test_retained_periodic_queries_do_not_rebuild():
    result = testing.analyze_placed_sync(slot_fixture(11, 5), True)
    before = result.describe()
    for _ in range(5):
        result.precedes((0, 0, 0), True, (0, 10, 2), False)
        result.boundary_slot(2, 3, 0, 0)
    assert result.describe() == before


@pytest.mark.parametrize("count", [0, 1, 5, 8])
def test_shared_physical_family_preserves_distinct_handle_phases(count):
    class ShiftAlias(ir.IRMutator):
        def visit_call(self, call):
            if call.op.name == ir.get_op("buffer.slot").name and call.args[0].name_hint == "alias":
                original = call.args[1]
                shifted = ir.FloorMod(
                    ir.Add(
                        original.left,
                        ir.ConstInt(2, ir.DataType.INDEX, call.span),
                        ir.DataType.INDEX,
                        call.span,
                    ),
                    original.right,
                    ir.DataType.INDEX,
                    call.span,
                )
                return ir.Call(call.op, [call.args[0], shifted], call.type, call.span)
            return super().visit_call(call)

    function = ShiftAlias().visit_function(slot_fixture(count, 5, alias=True))
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("count", [0, 1, 2, 5, 8])
def test_local_ring_conflicts_without_shared_output_writes(count):
    function = rotating_fixture(count, [(3, 1)], [("load", 0, 0), ("rmw", 0, 1), ("rmw", 0, 2)])
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("stages", [1, 4, 16, 64])
def test_static_stage_growth_is_charged_in_normalized_fragments(stages):
    schedule = [(op, 0, stage % 5) for stage in range(stages) for op in ["load", "rmw", "store"]]
    result = testing.analyze_placed_sync(rotating_fixture(1000, [(5, 1)], schedule), True)
    description = result.describe()
    region = description["regions"][0]
    sites, fragments = len(region["sites"]), region["rotating_fragments"]
    assert sites == 3 * stages
    assert len(region["generators"]) <= 2 * fragments
    assert region["sweep_visits"] <= 4 * fragments
    assert region["sort_comparisons"] <= 12 * fragments * math.ceil(math.log2(fragments + 1))
    assert region["retained_frontier_entries"] <= 4 * 7 * sites
    assert description["analyzed_occurrences"] == sites


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_streaming_readonly_gm_membership_is_retained(count):
    function = rotating_fixture(
        count, [(3, 1)], [("load", 0, 0), ("rmw", 0, 0), ("store", 0, 0)], stream_input=True
    )
    validate(function, testing.analyze_placed_sync(function, True))
