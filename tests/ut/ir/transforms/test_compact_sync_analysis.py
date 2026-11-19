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

    with pytest.raises(ValueError, match="Initialization guard must test consecutive first-visit frames"):
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

    expected = "nonoverflowing" if overflow else "matching operand types and explicit casts"
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


def placed_tile(
    name: str, tile_type: ir.BufferType, address: int, span: ir.Span
) -> tuple[ir.Var, ir.AssignStmt]:
    tensor = ir.Var(name, tile_type, span)
    allocation = ir.Call(
        ir.get_op("buffer.alloc"),
        [ir.MakeTuple([], span), ir.ConstInt(address, ir.DataType.INDEX, span)],
        tile_type,
        span,
    )
    return tensor, ir.AssignStmt(tensor, allocation, span)


def matrix_overlap_fixture(
    inner: ir.TileLayout,
    address: int,
    column: int = 16,
    *,
    width: int = 64,
    extract_width: int = 32,
) -> tuple[ir.Function, ir.Call]:
    span = ir.Span.unknown()
    index, fp32 = ir.DataType.INDEX, ir.DataType.FP32
    row_major, col_major = ir.TileLayout.row_major, ir.TileLayout.col_major

    def ci(value: int) -> ir.ConstInt:
        return ir.ConstInt(value, index, span)

    outer = row_major if inner == col_major else col_major
    wide_type = ir.BufferType([64, width], fp32, ir.MemorySpace.Mat, blayout=outer, slayout=inner)
    narrow_type = ir.BufferType([32, width], fp32, ir.MemorySpace.Mat, blayout=col_major, slayout=row_major)
    left_type = ir.BufferType(
        [16, extract_width], fp32, ir.MemorySpace.Left, blayout=row_major, slayout=row_major
    )
    wide, alloc_wide = placed_tile("wide_mat", wide_type, 4096, span)
    narrow, alloc_narrow = placed_tile("narrow_mat", narrow_type, 4096 + address, span)
    left, alloc_left = placed_tile("left", left_type, 0, span)
    gm_wide = ir.Var("input_wide", ir.TensorType(wide_type.shape, fp32), span)
    gm_narrow = ir.Var("input_narrow", ir.TensorType(narrow_type.shape, fp32), span)

    def load(source: ir.Var, destination: ir.Var, shape: tuple[int, int]) -> ir.EvalStmt:
        return ir.EvalStmt(
            ir.Call(
                ir.get_op("buffer.load"),
                [
                    source,
                    ir.MakeTuple([ci(0), ci(0)], span),
                    ir.MakeTuple([ci(n) for n in shape], span),
                    destination,
                ],
                ir.VoidType(),
                span,
            ),
            span,
        )

    iv = ir.Var("row", ir.ScalarType(index), span)
    extract = ir.Call(ir.get_op("buffer.extract"), [wide, iv, ci(column), left], ir.VoidType(), span)
    loop = ir.ForStmt(
        iv,
        ci(16),
        ci(48),
        ci(16),
        [],
        ir.SeqStmts([ir.EvalStmt(extract, span), ir.YieldStmt([], span)], span),
        [],
        span,
    )
    body = ir.SeqStmts(
        [
            alloc_wide,
            alloc_narrow,
            alloc_left,
            load(gm_wide, wide, (64, width)),
            load(gm_narrow, narrow, (32, width)),
            loop,
            ir.ReturnStmt([], span),
        ],
        span,
    )
    return ir.Function(
        "overlapping_matrix_windows",
        [gm_wide, gm_narrow],
        [],
        body,
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    ), extract


@pytest.mark.parametrize("inner", [ir.TileLayout.row_major, ir.TileLayout.col_major])
@pytest.mark.parametrize("address", [0, 2048])
def test_overlapping_matrix_views_use_exact_native_byte_stripes(inner, address):
    function, extract = matrix_overlap_fixture(inner, address)
    data = testing.describe_placed_sync_input(function, True)
    locations = [
        (index, storage)
        for index, storage in enumerate(data["storage"])
        if storage["space"] == ir.MemorySpace.Mat
    ]
    assert len(locations) == 1
    storage_id, matrix = locations[0]
    assert matrix["byte_coordinates"]
    sites = [site for region in data["regions"] for site in region["sites"]]
    site = next(site for site in sites if site["original"].same_as(extract))
    narrow_writes = [effect for effect in sites[1]["accesses"] if effect[0] == storage_id and effect[1]]
    assert len(narrow_writes) == 1
    _, _, row, col, row_step, col_step, rows, cols, _ = narrow_writes[0]
    assert row == row_step == col_step == 0 and rows == 1
    assert matrix["address"] + col == 4096 + address
    assert cols == 32 * 64 * 4

    # Native FP32 ZN has 8x16 column-major interiors, while NZ has 16x8
    # row-major interiors. Independently enumerate individual bytes so a gap in
    # the layout, an erroneous pitch or an extra stripe cannot pass.
    zn = inner == ir.TileLayout.col_major
    box_rows, box_cols = (8, 16) if zn else (16, 8)
    for visit in range(2):
        row_begin = 16 + visit * 16
        expected = set()
        for row in range(row_begin, row_begin + 16):
            outer_row, r = divmod(row, box_rows)
            for col in range(16, 48):
                outer_col, c = divmod(col, box_cols)
                if zn:
                    block = outer_row * (64 // box_cols) + outer_col
                    element = c * box_rows + r
                else:
                    block = outer_col * (64 // box_rows) + outer_row
                    element = r * box_cols + c
                start = 4096 + block * 512 + element * 4
                expected.update(range(start, start + 4))
        actual = set()
        for domain, write, row, col, row_step, col_step, rows, cols, _ in site["accesses"]:
            if domain == storage_id:
                assert not write and row == row_step == 0 and rows == 1
                begin = matrix["address"] + col + visit * col_step
                actual.update(range(begin, begin + cols))
        assert actual == expected


def test_overlapping_zn_views_refuse_unaligned_native_box():
    function, _ = matrix_overlap_fixture(ir.TileLayout.col_major, 2048, column=8)
    with pytest.raises(ValueError, match="whole-box access"):
        testing.describe_placed_sync_input(function, True)


def test_overlapping_matrix_views_refuse_excessive_stripes():
    # NZ with FP32 16x8 boxes. A 16x1040 partial-height extract has 130
    # separated runs. The exact importer must refuse instead of truncating.
    function, _ = matrix_overlap_fixture(
        ir.TileLayout.row_major, 2048, column=0, width=1056, extract_width=1040
    )
    before = ir.serialize(function)
    with pytest.raises(ValueError, match="exact physical-byte run limit"):
        testing.describe_placed_sync_input(function, True)
    assert ir.serialize(function) == before


def test_overlapping_accumulator_views_only_protect_the_same_layout():
    span = ir.Span.unknown()
    fp32 = ir.DataType.FP32
    row_major, col_major = ir.TileLayout.row_major, ir.TileLayout.col_major
    wide_a = ir.BufferType(
        [64, 64], fp32, ir.MemorySpace.Acc, blayout=col_major, slayout=row_major, fractal=1024
    )
    narrow_a = ir.BufferType(
        [48, 64], fp32, ir.MemorySpace.Acc, blayout=col_major, slayout=row_major, fractal=1024
    )
    wide_l = ir.BufferType([64, 32], fp32, ir.MemorySpace.Left, blayout=row_major, slayout=row_major)
    narrow_l = ir.BufferType([48, 32], fp32, ir.MemorySpace.Left, blayout=row_major, slayout=row_major)
    right_type = ir.BufferType([32, 64], fp32, ir.MemorySpace.Right, blayout=row_major, slayout=col_major)
    wide, alloc_wide = placed_tile("wide_acc", wide_a, 0, span)
    narrow, alloc_narrow = placed_tile("narrow_acc", narrow_a, 0, span)
    left_wide, alloc_lw = placed_tile("wide_lhs", wide_l, 0, span)
    left_narrow, alloc_ln = placed_tile("narrow_lhs", narrow_l, 8192, span)
    right, alloc_r = placed_tile("rhs", right_type, 0, span)

    def matrix(name: str, lhs: ir.Var, dst: ir.Var) -> ir.EvalStmt:
        return ir.EvalStmt(ir.Call(ir.get_op(name), [lhs, right, dst], ir.VoidType(), span), span)

    body = ir.SeqStmts(
        [
            alloc_wide,
            alloc_narrow,
            alloc_lw,
            alloc_ln,
            alloc_r,
            matrix("buffer.matmul", left_wide, wide),
            matrix("buffer.matmul_acc", left_wide, wide),
            matrix("buffer.matmul_acc", left_narrow, narrow),
            matrix("buffer.matmul_acc", left_narrow, narrow),
            ir.ReturnStmt([], span),
        ],
        span,
    )
    function = ir.Function(
        "aliased_accumulator_windows",
        [],
        [],
        body,
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    result = testing.analyze_placed_sync(function, True)
    assert not result.precedes((0, 0, 0), True, (0, 0, 1), False)
    assert result.precedes((0, 0, 1), True, (0, 0, 2), False)
    assert not result.precedes((0, 0, 2), True, (0, 0, 3), False)


def test_overlapping_accumulator_chain_refuses_a_common_interference_domain():
    span = ir.Span.unknown()
    acc_type = ir.BufferType(
        [64, 64],
        ir.DataType.FP32,
        ir.MemorySpace.Acc,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=1024,
    )
    # Each window is 16 KiB. The middle view overlaps both ends, but a
    # whole-window read at one end does not interfere with the other end.
    allocs = [placed_tile(f"acc_{i}", acc_type, offset, span)[1] for i, offset in enumerate((0, 8192, 16384))]
    body = ir.SeqStmts([*allocs, ir.ReturnStmt([], span)], span)
    function = ir.Function(
        "accumulator_overlap_chain",
        [],
        [],
        body,
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="pairwise-intersecting whole windows"):
        testing.describe_placed_sync_input(function, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
