# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One-point GM staging preserves original ABI, values and executed cuts."""

import pytest
from pypto import codegen, ir, passes
from pypto import language as pl
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import PassManager

from .buffer_test_utils import statements
from .sync_closed_command_oracle import ClosedCommands
from .test_runtime_valid_broadcast import diagnostics
from .test_scalar_tile_buffer_bridge import PLANNERS, calls

pytestmark = pytest.mark.usefixtures("ascend_backend")


def scalar_gm_program(length: int | str = 9, *, dtype="INT32", discarded=False):
    definition = (
        "pl.tensor.read(indices, [position])"
        if discarded
        else ("scalar = pl.tensor.read(indices, [position])")
    )
    update = (
        ""
        if discarded
        else """
                    if scalar > 0:
                        pl.tile.write(tile, [0,0], pl.cast(scalar, pl.INT32))
"""
    )
    preamble = f'{length} = pl.dynamic("{length}")\n' if isinstance(length, str) else ""
    return pl.parse_program(
        preamble
        + f"""
@pl.program
class GMPoint:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, indices: pl.Tensor[[{length}], pl.{dtype}],
               data: pl.Tensor[[2,8], pl.INT32], position: pl.Scalar[pl.INDEX],
               count: pl.Scalar[pl.INDEX], visits: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[2,8], pl.INT32]]) -> pl.Tensor[[2,8], pl.INT32]:
        tile = pl.load(data, [0,0], [2,8])
        for visit in pl.range(visits):
            for i in pl.range(count):
                if i % 2 == 0:
                    {definition}
{update}
        out = pl.store(tile, [0,0], out)
        return out
"""
    )


def compile_program(original, planner, *, enabled=True, buffer=True, disjoint=False):
    with passes.PassContext(
        [],
        enable_buffer_ir=buffer,
        memory_planner=planner,
        enable_pypto_autosync=enabled,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=disjoint,
    ):
        return PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(original)


def is_point_load(call):
    source_type = call.args[0].type
    return isinstance(source_type, ir.TensorType) and len(source_type.shape) == 1


def check_order(execution):
    order = execution.run()
    for i, (_, _, reads, writes) in enumerate(execution.payloads):
        for j, (_, _, earlier_reads, earlier_writes) in enumerate(execution.payloads[:i]):
            if earlier_writes & (reads | writes) or earlier_reads & writes:
                assert (("payload", j, "C"), ("payload", i, "I")) in order
    assert all(pair in order for pair in execution.reuses)
    assert all(ids <= set(range(6)) for ids in execution.ids.values())
    assert not any(a == b for a, b in order)
    return order


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("length,position", [(1, 0), (9, 0), (9, 1), (9, 7), (9, 8)])
@pytest.mark.parametrize("disjoint", [False, True])
def test_rank_one_point_abi_and_native_partition_are_preserved(planner, length, position, disjoint):
    original = scalar_gm_program(length)
    original_bytes = ir.serialize(original)
    result = compile_program(original, planner, disjoint=disjoint)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    assert not diagnostics(function)
    source = function.params[0]
    assert isinstance(source.type, ir.TensorType) and len(source.type.shape) == 1
    assert isinstance(source.type.shape[0], ir.ConstInt)
    assert source.type.shape[0].value == length
    assert not calls(result, "tensor.read")
    points = [call for call in calls(result, "buffer.load") if is_point_load(call)]
    assert len(points) == 1
    point = points[0]
    assert point.args[0].same_as(source)
    assert isinstance(point.args[1], ir.MakeTuple) and isinstance(point.args[2], ir.MakeTuple)
    assert isinstance(point.args[3].type, ir.BufferType)
    assert all(isinstance(e, ir.ConstInt) for e in point.args[2].elements)
    assert len(point.args[1].elements) == 1 and point.args[1].elements[0].same_as(function.params[2])
    assert [e.value for e in point.args[2].elements if isinstance(e, ir.ConstInt)] == [1]
    assert list(point.args[3].type.shape) == [1, 8] and list(point.args[3].type.valid_shape) == [1, 1]
    assert len(calls(result, "buffer.read")) == 1
    text = codegen.PTOCodegen().generate(result)
    # Source partition extent is one even though UB capacity is eight. Native
    # validCol must equal source Shape4, so an eight-element partition is wrong.
    assert "!pto.partition_tensor_view<1xi32>" in text
    assert "!pto.partition_tensor_view<8xi32>" not in text
    assert "pto.tgetval" in text
    assert ir.serialize(original) == original_bytes
    if planner == passes.MemoryPlanner.PTOAS:
        assert not function.attrs.get("pypto_autosync", False)
        return
    assert function.attrs["pypto_autosync"]
    values = [100 + i for i in range(length)]
    values[position] = -5 if position == 7 else 700 + position
    for count, visits in [(0, 0), (0, 2), (1, 1), (2, 1), (3, 2)]:
        memory = {("gm", i * 4 + byte): v for i, v in enumerate(values) for byte in range(4)}
        memory.update({("gm", 4096 + i * 4 + byte): i for i in range(16) for byte in range(4)})
        execution = ClosedCommands(
            function, {2: position, 3: count, 4: visits}, range(6), scalar_memory=memory
        )
        check_order(execution)
        scalar_reads = [x for x in execution.calls if x[0].op.name == ir.get_op("buffer.read").name]
        assert len(scalar_reads) == ((count + 1) // 2) * visits
        for call, env, _, reads, writes in execution.calls:
            if call.op.name == ir.get_op("buffer.load").name and is_point_load(call):
                assert reads == frozenset(("gm", position * 4 + byte) for byte in range(4))
                assert len(writes) == 32
        assert execution.scalar_memory is not None
        expected = list(range(16))
        if count and visits and values[position] > 0:
            expected[0] = values[position]
        actual = [execution.scalar_memory[("gm", 5 * 4096 + i * 4)] for i in range(16)]
        assert actual == expected


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("enabled,buffer", [(False, True), (False, False), (True, False)])
def test_staging_is_explicit_and_default_delegation_retains_original_reads(planner, enabled, buffer):
    original = scalar_gm_program()
    result = compile_program(original, planner, enabled=enabled, buffer=buffer)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert calls(result, "tensor.read") and not calls(result, "buffer.read")


@pytest.mark.parametrize("discarded,dtype", [(True, "INT32"), (False, "INT64")])
def test_unqualified_scalar_staging_retains_the_whole_function(discarded, dtype):
    original = scalar_gm_program(discarded=discarded, dtype=dtype)
    result = compile_program(original, passes.MemoryPlanner.PYPTO)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert calls(result, "tensor.read") and not calls(result, "buffer.read")


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", PLANNERS)
def test_point_staging_generates_real_cpp_through_its_selected_owner(planner, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")
    result = compile_program(scalar_gm_program(1), planner)
    source, output = tmp_path / "point.pto", tmp_path / "point.cpp"
    source.write_text(codegen.PTOCodegen().generate(result))
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a2"])
    cpp = output.read_text()
    assert "TLOAD" in cpp and ".GetValue(" in cpp
    assert "pipe_barrier(PIPE_S)" not in cpp


def test_trip_values_do_not_expand_the_production_carrier_or_commands():
    # Runtime visit values share the exact same retained function. The bounded
    # interpreter varies them; staging and physical commands are static sites.
    result = compile_program(scalar_gm_program(), passes.MemoryPlanner.PYPTO)
    assert len(calls(result, "buffer.read")) == 1
    assert len(calls(result, "buffer.load")) == 2
    assert len([s for s in statements(result) if isinstance(s, ir.ForStmt)]) == 2


@pytest.mark.parametrize("length,position", [(1, 0), (9, 1), (9, 7), (9, 8)])
@pytest.mark.parametrize("overlap", [False, True])
def test_dynamic_rank_one_extent_and_permitted_overlapping_gm_views(length, position, overlap):
    original = scalar_gm_program("POINT_LENGTH")
    result = compile_program(original, passes.MemoryPlanner.PYPTO, disjoint=False)
    function = result.get_function("kernel")
    assert function is not None and function.attrs["pypto_autosync"]
    source_type = function.params[0].type
    assert isinstance(source_type, ir.TensorType) and len(source_type.shape) == 1
    dimension = source_type.shape[0]
    assert isinstance(dimension, ir.Var)
    values = [11 + i for i in range(length)]
    output_base = 0 if overlap else 5 * 4096
    for count, visits in [(0, 0), (0, 2), (1, 1), (3, 2)]:
        memory = {("gm", i * 4 + byte): v for i, v in enumerate(values) for byte in range(4)}
        memory.update({("gm", 4096 + i * 4 + byte): i for i in range(16) for byte in range(4)})
        execution = ClosedCommands(
            function,
            {2: position, 3: count, 4: visits},
            range(6),
            tensor_bases={0: 0, 1: 4096, 5: output_base},
            shape_parameters={dimension: length},
            scalar_memory=memory,
        )
        check_order(execution)
        assert execution.scalar_memory is not None
        expected = list(range(16))
        if count and visits:
            expected[0] = values[position]
        assert [execution.scalar_memory[("gm", output_base + i * 4)] for i in range(16)] == expected
        # Real source index and loaded value drive the conditional write, not
        # a launch sample or a synthetic ready invocation parameter.
        reads = [call for call in execution.calls if call[0].op.name == ir.get_op("buffer.read").name]
        assert len(reads) == ((count + 1) // 2) * visits
