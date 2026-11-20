# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent point-write values, bytes and actual consuming command checks."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto import language as pl
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.pass_manager import PassManager

from .buffer_test_utils import statements
from .sync_closed_command_oracle import ClosedCommands, opname
from .test_gm_scalar_staging import check_order
from .test_scalar_tile_buffer_bridge import PLANNERS, calls

pytestmark = pytest.mark.usefixtures("ascend_backend")


def write_program(
    length=9, *, assigned=False, recurrence=False, dtype="INT32", while_loop=False, inline_value=False
):
    stored = "pl.cast(pl.cast(seed, pl.INDEX) + i, pl.INT32)" if inline_value else "replacement"
    write = f"pl.tensor.write(indices, [position], {stored})"
    if assigned:
        write = "indices = " + write
    definition = (
        (
            "old = pl.tensor.read(indices, [position])\n"
            "                    replacement = old + pl.const(1, pl.INT32)"
        )
        if recurrence
        else "replacement = seed"
    )
    preamble = f'{length} = pl.dynamic("{length}")\n' if isinstance(length, str) else ""
    inner = (
        "i = pl.const(0, pl.INDEX)\n            while i < count:"
        if while_loop
        else "for i in pl.range(count):"
    )
    increment = "i = i + 1" if while_loop else ""
    return pl.parse_program(
        preamble
        + f"""
@pl.program
class GMPointWrite:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, indices: pl.InOut[pl.Tensor[[{length}], pl.{dtype}]],
               data: pl.Tensor[[2,8], pl.INT32], position: pl.Scalar[pl.INDEX],
               count: pl.Scalar[pl.INDEX], visits: pl.Scalar[pl.INDEX],
               seed: pl.Scalar[pl.INT32],
               out: pl.InOut[pl.Tensor[[2,8], pl.INT32]]) -> pl.Tensor[[2,8], pl.INT32]:
        tile = pl.load(data, [0,0], [2,8])
        for outer in pl.range(visits):
            {inner}
                if i % 2 == 0:
                    {definition}
                    {write}
                {increment}
        out = pl.store(tile, [0,0], out)
        return out
"""
    )


def lower(program, planner, *, disjoint=False, enabled=True, buffer=True, before_sync=False):
    with passes.PassContext(
        [],
        memory_planner=planner,
        enable_pypto_autosync=enabled,
        enable_buffer_ir=buffer,
        autosync_policy="conservative_prefix",
        enable_scalar_gm_dma=enabled and buffer,
        autosync_disjoint_gm=disjoint,
    ):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        if not before_sync:
            return manager.run_passes(program)
        boundary = manager.get_pass_names().index("RealizeSync")
        for transform in manager.passes[:boundary]:
            program = transform(program)
        return program


def point_stores(program):
    return [
        call
        for call in calls(program, "buffer.store")
        if isinstance(call.args[3].type, ir.TensorType) and len(call.args[3].type.shape) == 1
    ]


def memory(length):
    result = {("gm", 4 * i + b): 100 + i for i in range(length) for b in range(4)}
    result.update({("gm", 4096 + 4 * i + b): i for i in range(16) for b in range(4)})
    result.update({("gm", 6 * 4096 + 4 * i + b): -1234 for i in range(16) for b in range(4)})
    return result


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("assigned", [False, True])
@pytest.mark.parametrize("disjoint", [False, True])
def test_write_direction_keeps_original_root_index_and_four_byte_partition(planner, assigned, disjoint):
    original = write_program(assigned=assigned)
    saved = ir.serialize(original)
    program = lower(original, planner, disjoint=disjoint)
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    stores = point_stores(program)
    assert len(stores) == 1 and not calls(program, "tensor.write")
    store = stores[0]
    assert store.args[3].same_as(function.params[0])
    assert store.args[1].elements[0].same_as(function.params[2])
    assert [e.value for e in store.args[2].elements] == [1]
    assert tuple(store.args[0].type.shape) == (1, 8)
    assert tuple(store.args[0].type.valid_shape) == (1, 1)
    assert len(calls(program, "buffer.write")) == 1
    text = codegen.PTOCodegen().generate(program)
    assert "!pto.partition_tensor_view<1xi32>" in text
    assert "!pto.partition_tensor_view<8xi32>" not in text
    assert "pto.tsetval" in text
    assert ir.serialize(original) == saved
    if planner == passes.MemoryPlanner.PTOAS:
        assert not function.attrs.get("pypto_autosync", False)
        return
    assert function.attrs["pypto_autosync"]
    for position, count, visits, seed in [
        (0, 0, 0, -7),
        (1, 0, 2, 99),
        (7, 1, 1, -2147483648),
        (8, 3, 2, 2147483647),
        (0, 2, 3, -19),
    ]:
        execution = ClosedCommands(
            function, {2: position, 3: count, 4: visits, 5: seed}, range(6), scalar_memory=memory(9)
        )
        check_order(execution)
        assert execution.scalar_memory is not None
        expected = [100 + i for i in range(9)]
        if count and visits:
            expected[position] = seed
        assert [execution.scalar_memory[("gm", 4 * i)] for i in range(9)] == expected
        writes = [
            row
            for row in execution.calls
            if row[0].op.name == opname("buffer.store") and len(row[0].args[3].type.shape) == 1
        ]
        assert len(writes) == ((count + 1) // 2) * visits
        for _, _, _, reads, written in writes:
            assert len(reads) == 32
            assert written == frozenset(("gm", 4 * position + b) for b in range(4))
        for payload in range(len(execution.payloads)):
            assert (("payload", payload, "C"), execution.returns[-1]) in execution.order


@pytest.mark.parametrize("assigned", [False, True])
@pytest.mark.parametrize("disjoint", [False, True])
def test_recurrence_uses_original_gm_values_and_preserves_actual_token_order(assigned, disjoint):
    original = write_program(assigned=assigned, recurrence=True)
    program = lower(original, passes.MemoryPlanner.PYPTO, disjoint=disjoint, before_sync=True)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=disjoint):
        logical = passes.insert_sync()(program)
    logical_function = logical.get_function("kernel")
    assert logical_function is not None
    physical = testing.lower_sync_schedule(logical_function, eligible_ids=[2, 5])["function"]
    for count, visits in [(0, 2), (1, 1), (3, 2)]:
        states = [
            ClosedCommands(function, {2: 1, 3: count, 4: visits, 5: 99}, [2, 5], scalar_memory=memory(9))
            for function in (logical_function, physical)
        ]
        for state in states:
            check_order(state)
        before, after = states
        assert before.payloads == after.payloads
        endpoints = {v for v in before.vertices if v[0] == "payload"}
        assert {(a, b) for a, b in before.order if a in endpoints and b in endpoints} == {
            (a, b) for a, b in after.order if a in endpoints and b in endpoints
        }
        assert after.scalar_memory is not None
        assert after.scalar_memory[("gm", 4)] == 101 + ((count + 1) // 2) * visits
        assert [after.scalar_memory[("gm", 4 * i)] for i in range(9) if i != 1] == [
            100 + i for i in range(9) if i != 1
        ]


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("enabled,buffer", [(False, True), (False, False), (True, False)])
def test_write_staging_keeps_default_and_disabled_paths(planner, enabled, buffer):
    result = lower(write_program(), planner, enabled=enabled, buffer=buffer)
    assert calls(result, "tensor.write") and not calls(result, "buffer.write")


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", PLANNERS)
def test_write_staging_real_cpp_preserves_one_element_store_and_final_owner(planner, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")
    result = lower(write_program(assigned=True), planner)
    pto, cpp = tmp_path / "point-write.pto", tmp_path / "point-write.cpp"
    pto.write_text(codegen.PTOCodegen().generate(result))
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(pto), str(cpp), [f"--pto-level={level}", "--pto-arch=a2"])
    text = cpp.read_text()
    assert "TSTORE" in text and ".SetValue(" in text
    assert "pipe_barrier(PIPE_S)" not in text


def rank_two_point_copy(dtype, columns):
    return pl.parse_program(f"""
POINT_ROWS = pl.dynamic("POINT_ROWS")

@pl.program
class RankTwoPointCopy:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, cells: pl.InOut[pl.Tensor[[POINT_ROWS, {columns}], pl.{dtype}]],
               row: pl.Scalar[pl.INDEX], source_column: pl.Scalar[pl.INDEX],
               target_column: pl.Scalar[pl.INDEX], visits: pl.Scalar[pl.INDEX]
               ) -> pl.Tensor[[POINT_ROWS, {columns}], pl.{dtype}]:
        for i in pl.range(visits):
            point = pl.tensor.read(cells, [row, source_column])
            cells = pl.tensor.write(cells, [row, target_column], point)
        return cells
""")


@pytest.mark.parametrize("dtype,element_type", [("INT32", "i32"), ("FP32", "f32")])
@pytest.mark.parametrize("columns,source_column,target_column", [(1, 0, 0), (9, 1, 7), (9, 8, 0)])
def test_rank_two_point_copy_keeps_parent_pitch_and_single_element_effects(
    dtype, element_type, columns, source_column, target_column
):
    original = rank_two_point_copy(dtype, columns)
    result = lower(original, passes.MemoryPlanner.PYPTO)
    function = result.get_function("kernel")
    assert function is not None and function.attrs.get("pypto_autosync", False)
    assert not calls(result, "tensor.read") and not calls(result, "tensor.write")
    tensor_type = function.params[0].type
    assert isinstance(tensor_type, ir.TensorType)
    row_symbol, column_extent = tensor_type.shape
    assert isinstance(column_extent, ir.ConstInt) and column_extent.value == columns
    (load,) = calls(result, "buffer.load")
    (store,) = calls(result, "buffer.store")
    for transfer, tensor in [(load, load.args[0]), (store, store.args[3])]:
        assert tensor.same_as(function.params[0])
        assert transfer.args[1].elements[0].same_as(function.params[1])
        assert [extent.value for extent in transfer.args[2].elements] == [1, 1]
    text = codegen.PTOCodegen().generate(result)
    assert f"!pto.partition_tensor_view<1x1x{element_type}>" in text
    assert f"!pto.partition_tensor_view<1x8x{element_type}>" not in text
    # Read and write in the second row. The neighbors hold distinct values so
    # a flat-without-pitch address or widened store changes observable bytes.
    source = columns + source_column
    target = columns + target_column
    values = [index + (0.25 if dtype == "FP32" else 0) for index in range(2 * columns)]
    for visits in [0, 1, 3]:
        cells = {("gm", 4 * i + b): v for i, v in enumerate(values) for b in range(4)}
        execution = ClosedCommands(
            function,
            {1: 1, 2: source_column, 3: target_column, 4: visits},
            range(6),
            shape_parameters={row_symbol: 2},
            scalar_memory=cells,
        )
        check_order(execution)
        assert len(execution.calls) == visits * 4
        for command, _, _, reads, writes in execution.calls:
            if command.op.name == opname("buffer.load"):
                assert reads == frozenset(("gm", 4 * source + b) for b in range(4))
                assert len(writes) == 32
            elif command.op.name == opname("buffer.store"):
                assert writes == frozenset(("gm", 4 * target + b) for b in range(4))
                assert len(reads) == 32
        expected = list(values)
        if visits:
            expected[target] = expected[source]
        assert execution.scalar_memory is not None
        assert [execution.scalar_memory[("gm", 4 * i)] for i in range(2 * columns)] == expected


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("dtype,columns", [("INT32", 9), ("FP32", 1)])
def test_rank_two_scalar_copy_emits_actual_native_load_and_store(planner, dtype, columns, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")
    result = lower(rank_two_point_copy(dtype, columns), planner)
    pto, cpp = tmp_path / "rank-two-point.pto", tmp_path / "rank-two-point.cpp"
    pto.write_text(codegen.PTOCodegen().generate(result))
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(pto), str(cpp), [f"--pto-level={level}", "--pto-arch=a2"])
    text = cpp.read_text()
    assert "TLOAD" in text and "TSTORE" in text
    assert ".GetValue(" in text and ".SetValue(" in text


def test_rank_two_staging_keeps_default_native_accesses():
    original = rank_two_point_copy("INT32", 9)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_policy="conservative_prefix"):
        result = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(original)
    assert calls(result, "tensor.read") and calls(result, "tensor.write")
    assert not calls(result, "buffer.read") and not calls(result, "buffer.write")


@pytest.mark.parametrize("columns", ["POINT_ROWS", 2**30 + 1])
def test_rank_two_point_copy_delegates_unqualified_native_pitch(columns):
    result = lower(rank_two_point_copy("INT32", columns), passes.MemoryPlanner.PYPTO)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert calls(result, "tensor.read") and calls(result, "tensor.write")


def test_dynamic_trip_values_keep_one_static_write_carrier():
    result = lower(write_program(), passes.MemoryPlanner.PYPTO)
    assert len(point_stores(result)) == len(calls(result, "buffer.write")) == 1
    assert len([s for s in statements(result) if isinstance(s, ir.ForStmt)]) == 2


def test_independent_checker_rejects_removing_complete_flag_protocol():
    program = lower(write_program(), passes.MemoryPlanner.PYPTO)
    original = program.get_function("kernel")
    assert original is not None and original.attrs["pypto_autosync"]
    check_order(ClosedCommands(original, {2: 1, 3: 3, 4: 2, 5: -10}, range(6), scalar_memory=memory(9)))

    class RemoveFlags(ir.IRMutator):
        def visit_eval_stmt(self, op):
            statement = op
            if isinstance(statement.expr, ir.Call) and statement.expr.op.name in (
                opname("system.set_flag"),
                opname("system.wait_flag"),
            ):
                return ir.SeqStmts([], statement.span)
            return statement

    broken = RemoveFlags().visit_program(program).get_function("kernel")
    assert broken is not None
    execution = ClosedCommands(broken, {2: 1, 3: 3, 4: 2, 5: -10}, range(6), scalar_memory=memory(9))
    # No unmatched flag remains after removing both sides. Actual payload
    # hazards must still fail: a terminal ALL cannot order an earlier consumer.
    with pytest.raises(AssertionError):
        check_order(execution)


@pytest.mark.parametrize("planner", PLANNERS)
def test_pure_point_write_initializes_its_only_valid_lane(planner):
    original = pl.parse_program("""
@pl.program
class PurePointWrite:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, out: pl.InOut[pl.Tensor[[9],pl.INT32]],
               position: pl.Scalar[pl.INDEX], seed: pl.Scalar[pl.INT32]) -> pl.Tensor[[9],pl.INT32]:
        updated = pl.tensor.write(out,[position],seed)
        return updated
""")
    result = lower(original, planner)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    assert len(point_stores(result)) == 1
    assert not calls(result, "buffer.load") and not calls(result, "buffer.full")
    if planner == passes.MemoryPlanner.PTOAS:
        return
    execution = ClosedCommands(function, {1: 7, 2: -21}, range(6), scalar_memory=memory(9))
    check_order(execution)
    assert execution.scalar_memory is not None
    assert [execution.scalar_memory[("gm", 4 * i)] for i in range(9)] == [
        (-21 if i == 7 else 100 + i) for i in range(9)
    ]


def test_may_alias_reads_and_point_writes_share_original_physical_bytes():
    result = lower(write_program(), passes.MemoryPlanner.PYPTO, disjoint=False)
    function = result.get_function("kernel")
    assert function is not None and function.attrs["pypto_autosync"]
    initial = {("gm", 4 * i + b): 100 + i for i in range(16) for b in range(4)}
    initial.update({("gm", 6 * 4096 + 4 * i + b): -444 for i in range(16) for b in range(4)})
    execution = ClosedCommands(
        function,
        {2: 7, 3: 3, 4: 2, 5: -41},
        range(6),
        tensor_bases={0: 0, 1: 0, 6: 6 * 4096},
        scalar_memory=initial,
    )
    check_order(execution)
    assert execution.scalar_memory is not None
    assert [execution.scalar_memory[("gm", 4 * i)] for i in range(16)] == [
        (-41 if i == 7 else 100 + i) for i in range(16)
    ]
    assert [execution.scalar_memory[("gm", 6 * 4096 + 4 * i)] for i in range(16)] == [
        100 + i for i in range(16)
    ]


@pytest.mark.parametrize("operand", ["value", "index", "casted_value"])
def test_nested_effectful_write_operands_keep_original_function_on_refusal(operand):
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", enable_scalar_gm_dma=True
    ):
        program = write_program()
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        for transform in manager.passes[:boundary]:
            program = transform(program)

        class NestedRead(ir.IRMutator):
            def visit_call(self, op):
                call = op
                if call.op.name != opname("tensor.write"):
                    return call
                args = list(call.args)
                index = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, call.span)], call.span)
                nested = ir.Call(
                    ir.get_op("tensor.read"), [args[0], index], ir.ScalarType(ir.DataType.INT32), call.span
                )
                if operand in ("value", "casted_value"):
                    args[2] = nested if operand == "value" else ir.Cast(nested, ir.DataType.INT32, call.span)
                else:
                    args[1] = ir.MakeTuple([ir.Cast(nested, ir.DataType.INDEX, call.span)], call.span)
                return ir.Call(call.op, args, call.type, call.span)

        modified = NestedRead().visit_program(program)
        original = modified.get_function("kernel")
        assert original is not None
        selected = passes.select_buffer_ir_functions()(modified).get_function("kernel")
        assert selected is not None and not selected.attrs["buffer_ir_selected"]
        ir.assert_structural_equal(original.body, selected.body)


@pytest.mark.parametrize("distributed", [False, True])
def test_registered_write_result_preserves_physical_window_before_optional_staging(distributed):
    span = ir.Span.unknown()
    base = ir.Var("original_root", ir.PtrType(), span)
    memref = ir.MemRef(base, 0, 36, span)
    dims = [ir.ConstInt(9, ir.DataType.INDEX, span)]
    window = ir.WindowBuffer(base, ir.ConstInt(36, ir.DataType.INDEX, span), span=span)
    kind = (
        ir.DistributedTensorType(dims, ir.DataType.INT32, memref, None, window)
        if distributed
        else ir.TensorType(dims, ir.DataType.INT32, memref)
    )
    target = ir.Var("target", kind, span)
    seed = ir.Var("seed", ir.ScalarType(ir.DataType.INT32), span)
    write = tensor_ops.write(target, [ir.ConstInt(1, ir.DataType.INDEX, span)], seed)
    # Distributed window identity is an operation/type contract here, not
    # a local placement or peer-visibility qualification. A standalone window
    # reference has no source-level CommDomain declaration for a function pass.
    if distributed:
        assert isinstance(write.type, ir.DistributedTensorType)
        assert write.type.window_buffer is not None and write.type.window_buffer.same_as(window)
        ir.assert_structural_equal(write.type, kind)
        restored = ir.deserialize(ir.serialize(write))
        assert isinstance(restored, ir.Call)
        ir.assert_structural_equal(write, restored, enable_auto_mapping=True)
        assert isinstance(restored.type, ir.DistributedTensorType)
        assert isinstance(restored.args[0].type, ir.DistributedTensorType)
        restored_window = restored.type.window_buffer
        source_window = restored.args[0].type.window_buffer
        assert restored_window is not None and source_window is not None
        assert restored_window.same_as(source_window)
        return
    updated = ir.Var("updated", write.type, span)
    function = ir.Function(
        "write_alias",
        [(target, ir.ParamDirection.InOut), seed],
        [kind],
        ir.SeqStmts([ir.AssignStmt(updated, write, span), ir.ReturnStmt([updated], span)], span),
        span,
        ir.FunctionType.AIV,
    )
    program = passes.init_mem_ref()(ir.Program([function], "WriteAlias", span))
    result = program.get_function("write_alias")
    assert result is not None
    source = result.params[0].type
    assignments = [
        stmt
        for stmt in statements(program)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == opname("tensor.write")
    ]
    assert len(assignments) == 1
    alias = assignments[0].var.type
    assert isinstance(source, ir.TensorType) and isinstance(alias, ir.TensorType)
    assert source.memref is not None and alias.memref is not None
    assert source.memref.base_.same_as(alias.memref.base_)
    ir.assert_structural_equal(source.memref, alias.memref)


@pytest.mark.parametrize("length,actual", [(1, 1), ("POINT_LENGTH", 1), ("POINT_LENGTH", 9)])
def test_singleton_and_symbolic_gm_lengths_preserve_original_point_domain(length, actual):
    result = lower(write_program(length), passes.MemoryPlanner.PYPTO)
    function = result.get_function("kernel")
    assert function is not None and function.attrs["pypto_autosync"]
    destination_type = function.params[0].type
    assert isinstance(destination_type, ir.TensorType)
    dim = destination_type.shape[0]
    shapes = {dim: actual} if isinstance(dim, ir.Var) else None
    for count in (0, 3):
        execution = ClosedCommands(
            function,
            {2: actual - 1, 3: count, 4: 2, 5: -91},
            range(6),
            shape_parameters=shapes,
            scalar_memory=memory(actual),
        )
        check_order(execution)
        assert execution.scalar_memory is not None
        expected = [100 + i for i in range(actual)]
        if count:
            expected[-1] = -91
        assert [execution.scalar_memory[("gm", 4 * i)] for i in range(actual)] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_mixed_scalar_and_dma_writes_remain_explicitly_refused(enabled):
    original = pl.parse_program("""
@pl.program
class MixedChannels:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, dst: pl.InOut[pl.Tensor[[64],pl.INT32]],
               seed: pl.Scalar[pl.INT32]) -> pl.Tensor[[64],pl.INT32]:
        fill=pl.full([32],dtype=pl.INT32,value=-1)
        dst[0:32]=fill
        for i in pl.range(4):
            pl.tensor.write(dst,[i],seed)
        return dst
""")
    with pytest.raises(ValueError, match="mixes MTE3 and scalar stores"):
        lower(original, passes.MemoryPlanner.PYPTO, enabled=enabled)


@pytest.mark.parametrize("foreign", ["root", "offset"])
def test_explicit_inconsistent_write_result_storage_is_not_rebound(foreign):
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", enable_scalar_gm_dma=True
    ):
        program = write_program()
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        for transform in manager.passes[:boundary]:
            program = transform(program)

        class ForeignResult(ir.IRMutator):
            def visit_call(self, op):
                call = op
                if call.op.name != opname("tensor.write"):
                    return call
                tensor = call.args[0].type
                assert isinstance(tensor, ir.TensorType) and tensor.memref is not None
                original = tensor.memref
                base = (
                    ir.Var("foreign_root", ir.PtrType(), call.span) if foreign == "root" else original.base_
                )
                ref = ir.MemRef(base, 4 if foreign == "offset" else 0, original.size_, call.span)
                kind = ir.TensorType(tensor.shape, tensor.dtype, ref, tensor.tensor_view)
                return ir.Call(call.op, call.args, kind, call.span)

        modified = ForeignResult().visit_program(program)
        original = modified.get_function("kernel")
        assert original is not None
        selected = passes.select_buffer_ir_functions()(modified).get_function("kernel")
        assert selected is not None and not selected.attrs["buffer_ir_selected"]
        ir.assert_structural_equal(original.body, selected.body)


def test_while_final_false_and_outer_counter_resets_keep_point_recurrence():
    result = lower(write_program(recurrence=True, while_loop=True), passes.MemoryPlanner.PYPTO)
    function = result.get_function("kernel")
    assert function is not None and function.attrs["pypto_autosync"]
    for count, visits in [(0, 0), (0, 2), (1, 2), (3, 2)]:
        execution = ClosedCommands(
            function, {2: 7, 3: count, 4: visits, 5: 99}, range(6), scalar_memory=memory(9)
        )
        check_order(execution)
        assert execution.while_tests == (count + 1) * visits
        assert execution.scalar_memory is not None
        assert execution.scalar_memory[("gm", 28)] == 107 + ((count + 1) // 2) * visits


@pytest.mark.ptoas_compile
def test_default_off_assigned_write_keeps_functional_aliases_in_real_cpp(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")
    result = lower(write_program(assigned=True), passes.MemoryPlanner.PYPTO, enabled=False)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert calls(result, "tensor.write") and not calls(result, "buffer.write")
    pto, cpp = tmp_path / "ordinary-write.pto", tmp_path / "ordinary-write.cpp"
    pto.write_text(codegen.PTOCodegen().generate(result))
    _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    assert cpp.is_file()


@pytest.mark.parametrize("seed", [-2147483648, 2147483647])
def test_unchanged_inline_scalar_casts_preserve_machine_narrowing(seed):
    result = lower(write_program(inline_value=True), passes.MemoryPlanner.PYPTO)
    function = result.get_function("kernel")
    assert function is not None and function.attrs["pypto_autosync"]
    for count, visits in [(0, 2), (1, 1), (3, 2)]:
        execution = ClosedCommands(
            function, {2: 7, 3: count, 4: visits, 5: seed}, range(6), scalar_memory=memory(9)
        )
        check_order(execution)
        assert execution.scalar_memory is not None
        last_even = ((count - 1) // 2) * 2
        narrowed = (seed + last_even + 2**31) % 2**32 - 2**31
        assert execution.scalar_memory[("gm", 28)] == (narrowed if count and visits else 107)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
