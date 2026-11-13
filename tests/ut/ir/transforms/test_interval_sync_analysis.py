# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Interval generation checked from original Buffer addresses and event graphs."""

import math

import pytest
from pypto import ir, testing

from .sync_matmul_oracle import validate
from .test_rotating_sync_analysis import rotating_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def interval_pipeline(
    count=5, readers=3, slots=3, stride=1, *, boxed=False, partial_columns=False, ranges=None
):
    """Nested read prefixes make the old interval/atom incidence inventory quadratic."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    height, width = 16 * (readers + 1), 32
    minor = ir.TileLayout.row_major if boxed else ir.TileLayout.none_box
    source_type = ir.BufferType([height, width], fp, ir.MemorySpace.Mat, slayout=minor)
    group_type = ir.MultiBufferType(source_type, slots)
    group = ir.Var("ring", group_type, span)
    arg = ir.Var("arg", ir.TensorType([height, width], fp), span)
    definitions = [ir.AssignStmt(group, call("buffer.alloc_multi", [tup([]), ci(0)], group_type), span)]
    iv = ir.Var("i", ir.ScalarType(index), span)
    body: list[ir.Stmt] = []

    def select(phase, name):
        selector = ir.FloorMod(
            ir.Add(ir.Mul(iv, ci(stride), index, span), ci(phase), index, span), ci(slots), index, span
        )
        tile = ir.Var(name, source_type, span)
        body.append(ir.AssignStmt(tile, call("buffer.slot", [group, selector], source_type), span))
        return tile

    tile = select(0, "fill")
    body.append(ir.EvalStmt(call("buffer.load", [arg, tup([0, 0]), tup([height, width]), tile]), span))
    address = 0
    for reader in range(readers):
        lo, hi = ranges[reader] if ranges else (0, 16 * (reader + 1))
        shape = [hi - lo, 16 if partial_columns else width]
        target_type = ir.BufferType(shape, fp, ir.MemorySpace.Left, slayout=minor)
        target = ir.Var(f"operand_{reader}", target_type, span)
        definitions.append(
            ir.AssignStmt(target, call("buffer.alloc", [tup([]), ci(address)], target_type), span)
        )
        address += shape[0] * shape[1] * 4
        selected = select(reader + 1, f"slice_{reader}")
        body.append(ir.EvalStmt(call("buffer.extract", [selected, ci(lo), ci(0), target]), span))
    body.append(ir.YieldStmt([], span))
    loop = ir.ForStmt(iv, ci(0), ci(count), ci(1), [], ir.SeqStmts(body, span), [], span)
    return ir.Function(
        "interval_pipeline",
        [arg],
        [],
        ir.SeqStmts([*definitions, loop, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("count", [0, 1, 2, 5])
@pytest.mark.parametrize("slots,stride", [(1, 0), (3, 1), (6, 2), (5, 0)])
def test_partial_rotating_intervals_match_original_access_oracle(count, slots, stride):
    function = interval_pipeline(count, slots=slots, stride=stride)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    if slots > 1:
        region = result.describe()["regions"][0]
        assert region["interval_domains"]
        assert region["atom_tests"] == 0
        assert len(region["generators"]) <= 7 * len(region["sites"])


@pytest.mark.parametrize("boxed,partial_columns", [(True, False), (False, True)])
def test_partial_rotating_interval_requires_exact_contiguity(boxed, partial_columns):
    with pytest.raises(ValueError, match="exact contiguous within-slot"):
        testing.analyze_placed_sync(interval_pipeline(boxed=boxed, partial_columns=partial_columns), True)


def test_interval_generation_avoids_atom_incidence_expansion():
    for readers in [4, 16, 64, 128]:
        result = testing.analyze_placed_sync(interval_pipeline(1000000, readers), True)
        region = result.describe()["regions"][0]
        intervals = region["interval_fragments"]
        assert intervals == 2 * (readers + 1)
        assert region["atom_tests"] == region["rotating_fragments"] == 0
        assert region["interval_endpoints"] <= 2 * intervals
        assert region["interval_queries"] <= 2 * 7 * intervals
        assert region["interval_updates"] <= 4 * intervals
        assert region["interval_node_visits"] <= 64 * intervals * math.ceil(math.log2(intervals + 2))
        assert region["interval_tree_entries"] <= 16 * 7 * intervals
        assert len(region["generators"]) <= 7 * len(region["sites"])


def test_interval_generation_does_not_enumerate_trips_or_slots():
    signatures = []
    for count, slots in [(7, 3), (1000000, 3), (7, 1000003)]:
        region = testing.analyze_placed_sync(interval_pipeline(count, slots=slots), True).describe()[
            "regions"
        ][0]
        signatures.append(
            tuple(
                region[k]
                for k in [
                    "interval_fragments",
                    "interval_queries",
                    "interval_updates",
                    "interval_node_visits",
                    "interval_tree_entries",
                ]
            )
        )
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("count", [0, 1, 2, 7])
@pytest.mark.parametrize("families", [[(6, 2)], [(3, 1), (5, 2)]])
def test_multi_view_payload_keeps_distinct_offsets_and_family_frames(count, families):
    class Mix(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def visit_assign_stmt(self, op):
            if op.var.name_hint in {"tile_0", "tile_1"}:
                self.inputs.append(op.var)
            return super().visit_assign_stmt(op)

        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.add").name:
                # One target has operands from different phases, potentially
                # different periods. Its output aliases the first operand.
                output = self.inputs[0] if len(families) > 1 else op.args[2]
                return ir.Call(op.op, [*self.inputs, output], op.type, op.span)
            return super().visit_call(op)

    # The native Vec contract does not prove partial alias legality for
    # dynamically distinct slots. Use a disjoint destination in that case;
    # the different-family case exercises exact in-place RMW as well.
    schedule = [("load", 0, 0), ("load", len(families) - 1, 1), ("rmw", len(families), 0), ("store", 0, 2)]
    function = Mix().visit_function(rotating_fixture(count, [*families, (1, 0)], schedule))
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize("ranges", [[(0, 16), (32, 48), (16, 32)], [(0, 32), (16, 48), (48, 64)]])
def test_disjoint_and_partially_overlapping_interval_readers(ranges):
    for count in [0, 1, 5]:
        function = interval_pipeline(count, ranges=ranges)
        validate(function, testing.analyze_placed_sync(function, True))


def test_partial_interval_physical_refusal_preserves_retained_answers():
    function = interval_pipeline(5)
    retained = testing.analyze_placed_sync(function, True)
    before = retained.describe()
    with pytest.raises(ValueError, match="Unavailable same-pipe barrier mechanism"):
        retained.realize_physical("a2a3", list(range(6)))
    assert retained.describe() == before
    validate(function, retained)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
