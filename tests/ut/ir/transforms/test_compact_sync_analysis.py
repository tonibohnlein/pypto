# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end placed-input, compact reduction and independent regional checks."""

import pytest
from pypto import ir, testing

from .sync_matmul_cases import CASES, capture, staged_fixture
from .sync_matmul_oracle import validate

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("name", CASES)
def test_original_matmul_examples(name):
    function = capture(name)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    with pytest.raises(ValueError, match="Disjoint GM argument contract"):
        testing.analyze_placed_sync(function, False)
    assert result.boundary(999, 0, 0) is None


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("visits", [1, 2, 3])
def test_streamed_matmul_resets(count, visits):
    function = staged_fixture(count, visits=visits)
    validate(function, testing.analyze_placed_sync(function, True))


def test_compact_k_does_not_enumerate_iterations():
    signatures = []
    for count in [2, 4, 8, 16, 64, 1024, 1000000]:
        description = testing.analyze_placed_sync(staged_fixture(count, visits=2), True).describe()
        port_bound = sum(4 * len(r["sites"]) for r in description["regions"])
        assert description["boundary_graph_edges"] <= port_bound**2
        signatures.append(
            (
                description["analyzed_occurrences"],
                description["cells"],
                [(len(r["sites"]), len(r["demands"]), r["distance_entries"]) for r in description["regions"]],
            )
        )
    assert all(s == signatures[0] for s in signatures)


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_static_stages(stages):
    function = staged_fixture(3, stages=stages)
    validate(function, testing.analyze_placed_sync(function, True))


@pytest.mark.parametrize(("start", "step"), [(3, 1), (0, 2), (3, 2)])
def test_original_loop_coordinates(start, step):
    function = staged_fixture(3, visits=2, start=start, step=step)
    validate(function, testing.analyze_placed_sync(function, True))


def test_identical_local_allocations_share_physical_storage():
    class Alias(ir.IRMutator):
        def visit_call(self, op):
            if (
                op.op.name == ir.get_op("buffer.alloc").name
                and isinstance(op.type, ir.BufferType)
                and op.type.memory_space == ir.MemorySpace.Mat
            ):
                assert isinstance(op.args[1], ir.ConstInt)
                if op.args[1].value == 4096:
                    return ir.Call(
                        op.op, [op.args[0], ir.ConstInt(0, ir.DataType.INDEX, op.span)], op.type, op.span
                    )
            return op

    function = Alias().visit_function(staged_fixture(3, visits=2))
    validate(function, testing.analyze_placed_sync(function, True))


def test_unavailable_is_not_absence():
    f = staged_fixture(0)
    result = testing.analyze_placed_sync(f, True)
    for sid in range(len(result.describe()["storage"])):
        assert result.boundary(sid, 0, 0) == {
            "first_writer": None,
            "last_writer": None,
            "first_readers": [None] * 7,
            "last_readers": [None] * 7,
        }
    assert result.boundary(0, -1, 0) is None
    with pytest.raises(ValueError, match="occurrence"):
        result.precedes((0, 0, 0), False, (0, 0, 0), True)

    class Nested(ir.IRMutator):
        def visit_for_stmt(self, op):
            inner = ir.ForStmt(
                ir.Var("nested", op.loop_var.type, op.span),
                op.start,
                op.stop,
                op.step,
                [],
                op.body,
                [],
                op.span,
            )
            return ir.ForStmt(op.loop_var, op.start, op.stop, op.step, [], inner, [], op.span)

    with pytest.raises(ValueError, match="Nested or data-dependent"):
        testing.analyze_placed_sync(Nested().visit_function(staged_fixture(2)), True)


@pytest.mark.parametrize("overflow", [False, True])
def test_refuses_unqualified_affine_intermediates(overflow):
    class UnsupportedArithmetic(ir.IRMutator):
        def visit_mul(self, op):
            if overflow:
                intermediate = ir.Mul(
                    op.left, ir.ConstInt(2**63 - 1, ir.DataType.INDEX, op.span), ir.DataType.INDEX, op.span
                )
                return ir.Mul(
                    intermediate, ir.ConstInt(0, ir.DataType.INDEX, op.span), ir.DataType.INDEX, op.span
                )
            return ir.Mul(op.left, op.right, ir.DataType.INT32, op.span)

    expected = "nonoverflowing" if overflow else "entirely INDEX"
    with pytest.raises(ValueError, match=expected):
        testing.analyze_placed_sync(UnsupportedArithmetic().visit_function(staged_fixture(3)), True)


def test_refuses_addressless_allocation():
    class Addressless(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.alloc").name:
                return ir.Call(op.op, [op.args[0]], op.type, op.span)
            return op

    with pytest.raises(ValueError, match="placed buffer allocation addresses"):
        testing.analyze_placed_sync(Addressless().visit_function(staged_fixture(2)), True)


def test_refuses_partial_overlap_and_changed_guard():
    class Overlap(ir.IRMutator):
        def visit_call(self, op):
            if (
                op.op.name == ir.get_op("buffer.alloc").name
                and isinstance(op.type, ir.BufferType)
                and op.type.memory_space == ir.MemorySpace.Mat
            ):
                return ir.Call(
                    op.op, [op.args[0], ir.ConstInt(4, ir.DataType.INDEX, op.span)], op.type, op.span
                )
            return op

    with pytest.raises(ValueError, match="Overlapping placed buffers"):
        testing.analyze_placed_sync(Overlap().visit_function(staged_fixture(2)), True)

    class WrongGuard(ir.IRMutator):
        def visit_if_stmt(self, op):
            assert isinstance(op.condition, ir.Eq)
            return ir.IfStmt(
                ir.Eq(
                    op.condition.left, ir.ConstInt(1, ir.DataType.INDEX, op.span), ir.DataType.BOOL, op.span
                ),
                op.then_body,
                op.else_body,
                [],
                op.span,
            )

    with pytest.raises(ValueError, match="first-visit"):
        testing.analyze_placed_sync(WrongGuard().visit_function(staged_fixture(2)), True)


def test_retained_queries_do_not_rebuild_or_borrow_the_function():
    f = staged_fixture(3, visits=2)
    result = testing.analyze_placed_sync(f, True)
    before = result.describe()
    del f
    for _ in range(5):
        assert result.precedes((0, 0, 0), True, (0, 0, 4), False)
        assert result.boundary(0, 0, 0) is not None
    assert result.describe() == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
