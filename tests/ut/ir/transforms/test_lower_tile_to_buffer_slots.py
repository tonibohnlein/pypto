# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Uniform slots retain one allocation and original use-site coordinates."""

import pypto.language as pl
import pytest
from buffer_pipeline_utils import buffer_recipe_pipeline
from pypto import DataType, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen
from pypto.pypto_core import ir as core_ir

from .buffer_test_utils import statements

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()
RING = pl.MemRef(slots=3)
VIEW_RING = pl.MemRef(slots=2)


@pl.program
class SlottedCopy:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        source: pl.Tensor[[8, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ) -> pl.Tensor[[8, 32], pl.FP32]:
        for i in pl.range(8):
            tile: pl.Tile[[1, 32], pl.FP32, RING[i % 3], pl.Mem.Vec] = pl.load(source, [i, 0], [1, 32])
            output = pl.store(tile, [i, 0], output)
        return output


def integer(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def internal(name, args, result):
    return core_ir._create_internal_op_call(name, args, {}, result, SPAN)


def test_rotating_tile_pipeline_preserves_one_allocation_and_lexical_selection():
    original = ir.serialize(SlottedCopy)
    with passes.PassContext([], enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(SlottedCopy)
    allocations = []
    selections = []
    loops = []
    for stmt in statements(lowered):
        if isinstance(stmt, ir.ForStmt):
            loops.append(stmt)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == ir.get_op("buffer.alloc_multi").name:
                allocations.append(stmt)
            if stmt.value.op.name == ir.get_op("buffer.slot").name:
                selections.append(stmt)
    assert len(allocations) == 1
    group = allocations[0]
    assert isinstance(group.var.type, ir.MultiBufferType)
    assert group.var.type.slot_count == 3
    assert group.var.type.element_type.shape == [1, 32]
    assert len(group.value.args) == 2
    assert isinstance(group.value.args[1], ir.ConstInt)
    assert len(selections) == 2
    assert len(loops) == 1
    for selection in selections:
        assert selection.value.args[0].same_as(group.var)
        index = selection.value.args[1]
        assert isinstance(index, ir.FloorMod)
        assert index.left.same_as(loops[0].loop_var)
        assert index.right.value == 3
    assert ir.serialize(SlottedCopy) == original
    ir.assert_structural_equal(ir.deserialize(ir.serialize(lowered)), lowered)
    native = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert native.count("pto.alloc_multi_tile") == 1
    assert native.count("pto.multi_tile_get") == 2
    assert "count=3" in native


@pytest.mark.parametrize("slots", [1, 2, 17, 1000000])
def test_multibuffer_operation_preserves_numerical_count_without_enumeration(slots):
    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    multi = ir.MultiBufferType(element, slots)
    call = internal("buffer.alloc_multi", [ir.MakeTuple([], SPAN), integer(512)], multi)
    assert call.type.slot_count == slots
    group = ir.Var("ring", multi, SPAN)
    selected = internal("buffer.slot", [group, integer(slots - 1)], element)
    assert selected.args[1].value == slots - 1


@pytest.mark.parametrize("index", [-1, 3])
def test_constant_slot_index_must_be_in_range(index):
    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    group = ir.Var("ring", ir.MultiBufferType(element, 3), SPAN)
    with pytest.raises(ValueError, match="index must lie"):
        internal("buffer.slot", [group, integer(index)], element)


def test_slot_descriptor_and_metadata_cannot_change():
    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    group = ir.Var("ring", ir.MultiBufferType(element, 3), SPAN)
    smaller = ir.BufferType([1, 16], DataType.FP32, ir.MemorySpace.Vec)
    with pytest.raises(ValueError, match="element descriptor"):
        internal("buffer.slot", [group, integer(0)], smaller)
    dynamic = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec, [1, -1])
    with pytest.raises(ValueError, match="static slot valid extents"):
        internal("buffer.alloc_multi", [ir.MakeTuple([integer(16)], SPAN)], ir.MultiBufferType(dynamic, 2))


def _slot_program(index, address=512, slots=3):
    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    multi = ir.MultiBufferType(element, slots)
    group = ir.Var("ring", multi, SPAN)
    selected = ir.Var("selected", element, SPAN)
    allocation = ir.AssignStmt(
        group, internal("buffer.alloc_multi", [ir.MakeTuple([], SPAN), integer(address)], multi), SPAN
    )
    selection = ir.AssignStmt(selected, internal("buffer.slot", [group, index], element), SPAN)
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.SeqStmts([allocation, selection], SPAN),
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "Slot", SPAN)


def test_native_slot_group_uses_the_exact_placed_base():
    native = codegen.PTOCodegen().generate(_slot_program(integer(2)), emit_tile_addr=True)
    assert "512" in native
    assert "pto.alloc_multi_tile addr =" in native
    assert native.count("pto.multi_tile_get") == 1
    assert "count=3" in native


def test_native_slot_bound_remains_separate_from_ir_validity():
    program = _slot_program(integer(0), slots=17)
    with pytest.raises(ValueError, match="at most 16 slots"):
        codegen.PTOCodegen().generate(program, emit_tile_addr=True)


def test_allocation_rejects_slot_capacity_overflow():
    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    multi = ir.MultiBufferType(element, 1 << 60)
    with pytest.raises(ValueError, match="representable uniform physical"):
        internal("buffer.alloc_multi", [ir.MakeTuple([], SPAN)], multi)


def test_native_single_slot_family_uses_ordinary_allocation():
    native = codegen.PTOCodegen().generate(_slot_program(integer(0), slots=1), emit_tile_addr=True)
    assert native.count("pto.alloc_tile") == 1
    assert "pto.alloc_multi_tile" not in native
    assert "pto.multi_tile_get" not in native


def test_nested_co_live_slots_preserve_downstream_refusal():
    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    multi = ir.MultiBufferType(element, 2)
    group = ir.Var("ring", multi, SPAN)
    outer_index = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
    inner_index = ir.Var("j", ir.ScalarType(DataType.INDEX), SPAN)
    first = ir.Var("first", element, SPAN)
    second = ir.Var("second", element, SPAN)
    allocation = ir.AssignStmt(
        group, internal("buffer.alloc_multi", [ir.MakeTuple([], SPAN), integer(0)], multi), SPAN
    )
    # A slot selected in the ancestor body can remain live when the inner body
    # selects another. Checking only the innermost loop misses this case.
    select_first = ir.AssignStmt(first, internal("buffer.slot", [group, integer(0)], element), SPAN)
    select_second = ir.AssignStmt(second, internal("buffer.slot", [group, integer(1)], element), SPAN)
    inner = ir.ForStmt(inner_index, integer(0), integer(2), integer(1), [], select_second, [], SPAN)
    outer = ir.ForStmt(
        outer_index,
        integer(0),
        integer(2),
        integer(1),
        [],
        ir.SeqStmts([select_first, inner], SPAN),
        [],
        SPAN,
    )
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.SeqStmts([allocation, outer], SPAN),
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="distinct co-live loop slots"):
        codegen.PTOCodegen().generate(ir.Program([function], "NestedSlots", SPAN), emit_tile_addr=True)


def test_rotating_buffer_native_compilation_preserves_unsigned_slot_selector(tmp_path):
    with passes.PassContext([], enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(SlottedCopy)
    native = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "arith.remui" in native
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "rotating-buffer.pto"
    output = tmp_path / "rotating-buffer.cpp"
    source.write_text(native)
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    assert output.is_file()


def test_native_slot_stride_padding_is_not_silently_changed():
    # Buffer slots are tightly packed in this initial contract. Native Vec slots
    # round 16-byte elements to 32 bytes, so exporting this descriptor would move
    # slot 1 to a different address from the modeled one.
    element = ir.BufferType([1, 4], DataType.FP32, ir.MemorySpace.Vec)
    multi = ir.MultiBufferType(element, 2)
    group = ir.Var("ring", multi, SPAN)
    allocation = ir.AssignStmt(
        group, internal("buffer.alloc_multi", [ir.MakeTuple([], SPAN), integer(0)], multi), SPAN
    )
    function = ir.Function(
        "kernel",
        [],
        [],
        allocation,
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match="native-aligned physical slot stride"):
        codegen.PTOCodegen().generate(ir.Program([function], "UnalignedSlot", SPAN), emit_tile_addr=True)


def test_native_slot_base_alignment_requires_evidence():
    with pytest.raises(ValueError, match="native-aligned allocation base"):
        codegen.PTOCodegen().generate(_slot_program(integer(0), address=1), emit_tile_addr=True)


def test_ordinary_production_pipeline_selects_qualified_whole_slots():
    with passes.PassContext([], enable_buffer_ir=True):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(SlottedCopy)
    functions = [
        function for function in lowered.functions.values() if function.ir_stage == ir.FunctionIRStage.Buffer
    ]
    assert functions
    assert any(
        isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("buffer.alloc_multi").name
        for stmt in statements(lowered)
    )
    native = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.alloc_multi_tile" in native
    assert "arith.remui" in native


@pl.program
class PartialSlotView:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        source: pl.Tensor[[2, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
    ) -> pl.Tensor[[1, 32], pl.FP32]:
        tile: pl.Tile[[2, 32], pl.FP32, VIEW_RING[0], pl.Mem.Vec] = pl.load(source, [0, 0], [2, 32])
        part = pl.slice(tile, [1, 32], [1, 0])
        return pl.store(part, [0, 0], output)


@pytest.mark.parametrize("program", [PartialSlotView])
def test_ordinary_selection_retains_tile_when_slot_export_is_unqualified(program):
    with passes.PassContext([], enable_buffer_ir=True):
        prefix = passes.PassPipeline()
        for transform in PassManager(OptimizationStrategy.Default).passes:
            prefix.add_pass(transform)
            if transform.get_name() == "SelectBufferIRFunctions":
                break
        selected = prefix.run(program)
    functions = [
        function for function in selected.functions.values() if "buffer_ir_selected" in function.attrs
    ]
    assert functions
    assert all(function.attrs["buffer_ir_selected"] is False for function in functions)


@pytest.mark.parametrize("proved_zero", [False, True])
@pytest.mark.parametrize("template_count, template_offset", [(1, 0), (2, 0), (1, 1)])
def test_resolved_singleton_selection_needs_the_same_index_proof_as_emission(
    proved_zero, template_count, template_offset
):
    # The public parser intentionally disallows subscripting a one-slot MemRef.
    # Exercise the resolved IR contract directly: serializers/IR producers can
    # carry explicit singleton selection metadata independently of that syntax.
    tensor = ir.TensorType([1, 32], DataType.FP32)
    source = ir.Var("source", tensor, SPAN)
    output = ir.Var("output", tensor, SPAN)
    parameter = ir.Var("selected", ir.ScalarType(DataType.INDEX), SPAN)
    index = integer(0) if proved_zero else parameter
    base = ir.Var("ring", ir.PtrType(), SPAN)
    offset = integer(0) if proved_zero else ir.Mul(index, integer(128), DataType.INDEX, SPAN)
    memory = ir.MemRef(base, offset, 128, SPAN, slots=1, slot=index)
    descriptor = ir.TileType([1, 32], DataType.FP32, memory, memory_space=ir.MemorySpace.Vec)
    tile = ir.Var("tile", descriptor, SPAN)
    result = ir.Var("result", tensor, SPAN)
    zeros = ir.MakeTuple([integer(0), integer(0)], SPAN)
    shape = ir.MakeTuple([integer(1), integer(32)], SPAN)
    allocation = ir.Call(ir.Op("tile.alloc"), [integer(0), integer(128)], {}, None, ir.PtrType(), SPAN)
    # InitMemRef resolves the assigned TileType while leaving this operation
    # result template unresolved. A mismatched family cannot borrow its proof.
    template_memory = ir.MemRef(
        base, template_offset, 0, SPAN, is_pinned=True, slots=template_count, slot=index
    )
    template = ir.TileType([1, 32], DataType.FP32, template_memory, memory_space=ir.MemorySpace.Vec)
    load = ir.Call(ir.Op("tile.load"), [source, zeros, shape], {}, None, template, SPAN)
    store = ir.Call(ir.Op("tile.store"), [tile, zeros, output], {}, None, tensor, SPAN)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(base, allocation, SPAN),
            ir.AssignStmt(tile, load, SPAN),
            ir.AssignStmt(result, store, SPAN),
            ir.ReturnStmt([result], SPAN),
        ],
        SPAN,
    )
    function = ir.Function(
        "kernel",
        [source, (output, ir.ParamDirection.Out), parameter],
        [tensor],
        body,
        SPAN,
        type=ir.FunctionType.InCore,
    )
    with passes.PassContext([], enable_buffer_ir=True):
        selected = passes.select_buffer_ir_functions()(ir.Program([function], "Singleton", SPAN))
    assert selected.get_function("kernel").attrs["buffer_ir_selected"] is (
        proved_zero and template_count == 1 and template_offset == 0
    )
