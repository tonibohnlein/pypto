# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact contiguous vector relabels reuse the existing physical view owner."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes, testing
from pypto.debug import torch_codegen
from pypto.pypto_core import ir as core_ir

from .sync_matmul_oracle import validate
from .test_select_buffer_ir_functions import _PLANNERS, _compile_dynamic_state_native, _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def descriptor(shape, column=False, valid=(), dtype=ir.DataType.FP32):
    return ir.BufferType(
        list(shape),
        dtype,
        ir.MemorySpace.Vec,
        list(valid),
        blayout=ir.TileLayout.col_major if column else ir.TileLayout.row_major,
    )


def reshape(source, destination):
    return core_ir._create_internal_op_call(
        "buffer.reshape", [ir.Var("source", source, SPAN)], {}, destination, SPAN
    )


@pytest.mark.parametrize("length", [8, 16, 32])
def test_contiguous_vector_relabel_preserves_bidirectional_full_byte_span(length):
    column, row = descriptor((length, 1), True), descriptor((1, length))
    for source, destination in ((column, row), (row, column)):
        call = reshape(source, destination)
        assert call.type == destination
        # Independently enumerate the two native linear element maps. This
        # compares the same bytes, not merely element counts or SSA names.
        column_bytes = {4 * r + byte for r in range(length) for byte in range(4)}
        row_bytes = {4 * c + byte for c in range(length) for byte in range(4)}
        assert column_bytes == row_bytes == set(range(4 * length))


@pytest.mark.parametrize(
    "kind", ["padded_row", "partial", "dynamic", "dtype", "different_length", "overflow"]
)
def test_unqualified_vector_relabels_remain_unavailable(kind):
    column, row = descriptor((16, 1), True), descriptor((1, 16))
    if kind == "padded_row":
        column = descriptor((16, 1))
    elif kind == "partial":
        column = descriptor((16, 1), True, valid=(8, 1))
    elif kind == "dynamic":
        row = descriptor((1, 16), valid=(-1, -1))
    elif kind == "dtype":
        row = descriptor((1, 32), dtype=ir.DataType.FP16)
    elif kind == "different_length":
        row = descriptor((1, 32))
    else:
        with pytest.raises(ValueError):
            reshape(descriptor((2**62, 1), True), descriptor((1, 2**62)))
        return
    with pytest.raises(ValueError):
        reshape(column, row)
    with pytest.raises(ValueError):
        reshape(row, column)


def reduction_vector_program():
    return pl.parse_program("""
@pl.program
class ContiguousVectors:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 128], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
        loaded = pl.load(x, [0, 0], [16, 128])
        scratch = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        sums = pl.tile.row_sum(loaded, scratch)
        row = pl.reshape(sums, [1, 16])
        incremented = pl.tile.adds(row, 2.0)
        column = pl.reshape(incremented, [16, 1])
        expanded = pl.tile.row_expand_sub(loaded, column)
        return pl.store(expanded, [0, 0], out)
""")


@pytest.mark.parametrize("planner", _PLANNERS)
def test_reduction_vector_roundtrip_through_buffer_planners_and_native_cpp(tmp_path, planner):
    program = reduction_vector_program()
    lowered = _lower(program, planner)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048).reshape(16, 128).float() / 2048
    out = torch.full_like(x, -99)
    namespace["kernel"](x, out)
    torch.testing.assert_close(out, x - (x.sum(dim=1, keepdim=True) + 2))
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    native = tmp_path / "dynamic_state.pto"
    if native.exists() and planner == passes.MemoryPlanner.PTOAS:
        assert "pto.treshape" in native.read_text(), "addressless views must retain the typed alias"
    if planner != passes.MemoryPlanner.PTOAS:
        retained = testing.analyze_placed_sync(function, True)
        validate(function, retained)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_column_relabel_does_not_admit_an_inherited_slice_window(planner):
    program = pl.parse_program("""
@pl.program
class SlicedVector:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, vector: pl.Tensor[[1, 32], pl.FP32], matrix: pl.Tensor[[16, 16], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        loaded = pl.load(vector, [0, 0], [1, 32])
        window = pl.tile.slice(loaded, [1, 16], [0, 8])
        column = pl.reshape(window, [16, 1])
        values = pl.load(matrix, [0, 0], [16, 16])
        result = pl.tile.row_expand_sub(values, column)
        return pl.store(result, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional
