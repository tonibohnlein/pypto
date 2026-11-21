# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend scalar Tile operations use shared Buffer point contracts."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import codegen, ir, passes
from pypto import language as pl
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import PassManager

from .buffer_test_utils import statements
from .sync_closed_command_oracle import ClosedCommands
from .test_runtime_valid_broadcast import diagnostics

pytestmark = pytest.mark.usefixtures("ascend_backend")
PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


def frontend_scalar_program(assigned):
    write = (
        "changed = pl.tile.write(source, [0,0], replacement)"
        if assigned
        else "pl.tile.write(source, [0,0], replacement)"
    )
    stored = "changed" if assigned else "source"
    return pl.parse_program(
        f"""
@pl.program
class ScalarBridge:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[2,8], pl.FP32], count: pl.Scalar[pl.INDEX],
               visits: pl.Scalar[pl.INDEX], seen: pl.InOut[pl.Tensor[[2,8], pl.FP32]],
               out: pl.InOut[pl.Tensor[[2,8], pl.FP32]]) -> pl.Tensor[[2,8], pl.FP32]:
        source = pl.load(x, [0,0], [2,8])
        old_alias = source
        for visit in pl.range(visits):
            for i in pl.range(count):
                scalar = pl.tile.read(source, [0,0])
                if scalar > 0.0:
                    replacement = pl.const(-1.0, pl.FP32)
                else:
                    replacement = pl.const(1.0, pl.FP32)
                {write}
                out = pl.store({stored}, [0,0], out)
        seen = pl.store(old_alias, [0,0], seen)
        out = pl.store(source, [0,0], out)
        return out
"""
    )


def lower(program, planner):
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        return PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)


def calls(program, name):
    selected = []
    for statement in statements(program):
        expression = (
            statement.value
            if isinstance(statement, ir.AssignStmt)
            else (statement.expr if isinstance(statement, ir.EvalStmt) else None)
        )
        if isinstance(expression, ir.Call) and expression.op.name == ir.get_op(name).name:
            selected.append(expression)
    return selected


@pytest.mark.parametrize("assigned", [False, True])
@pytest.mark.parametrize("planner", PLANNERS)
def test_frontend_scalar_bridge_mutates_input_and_preserves_chaining_alias(assigned, planner):
    original = frontend_scalar_program(assigned)
    saved = ir.serialize(original)
    program = lower(original, planner)
    assert ir.serialize(original) == saved
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    assert not diagnostics(function)
    assert len(calls(program, "buffer.read")) == len(calls(program, "buffer.write")) == 1
    # The returned Tile is another name for the mutated destination. It must
    # not become a fresh allocation or a copy that leaves old_alias unchanged.
    assert not calls(program, "buffer.copy")
    emitted = codegen.PTOCodegen().generate(program)
    assert "pto.tgetval" in emitted and "pto.tsetval" in emitted
    if planner == passes.MemoryPlanner.PTOAS:
        # Downstream placement owns this addressless program. No invented local
        # addresses are supplied to the physical-byte oracle below.
        return
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        logical = passes.insert_sync()(program)
    logical = ir.deserialize(ir.serialize(logical))
    assert isinstance(logical, ir.Program)
    with passes.PassContext([]):
        physical = passes.lower_sync()(logical)
    before = logical.get_function("kernel")
    after = physical.get_function("kernel")
    assert before is not None and after is not None
    for count, visits in [(0, 0), (0, 2), (1, 1), (2, 1), (1, 3), (2, 2)]:
        for sign in [-1.0, 1.0]:
            initial = [sign, *[float(i) for i in range(1, 16)]]
            memory = {("gm", byte): initial[byte // 4] for byte in range(64)}
            expected = list(initial)
            for _ in range(count * visits):
                expected[0] = -1.0 if expected[0] > 0 else 1.0
            executions = [
                ClosedCommands(f, {1: count, 2: visits}, range(6), scalar_memory=memory)
                for f in (before, after)
            ]
            orders = [execution.run() for execution in executions]
            assert executions[0].payloads == executions[1].payloads
            common = {v for v in executions[0].vertices if v[0] == "payload"}
            assert {(a, b) for a, b in orders[0] if a in common and b in common} == {
                (a, b) for a, b in orders[1] if a in common and b in common
            }
            for execution, order in zip(executions, orders, strict=True):
                assert execution.scalar_memory is not None
                for parameter in [3, 4]:
                    actual = [execution.scalar_memory[("gm", parameter * 4096 + i * 4)] for i in range(16)]
                    assert actual == expected
                for i, (_, _, reads, writes) in enumerate(execution.payloads):
                    for j, (_, _, earlier_reads, earlier_writes) in enumerate(execution.payloads[:i]):
                        if earlier_writes & (reads | writes) or earlier_reads & writes:
                            assert (("payload", j, "C"), ("payload", i, "I")) in order
                assert all(pair in order for pair in execution.reuses)
                assert all(ids <= set(range(6)) for ids in execution.ids.values())


def scalar_metadata_program(point="[1,7]", dynamic_source=False, dtype="FP32"):
    selected = "narrow" if dynamic_source else "source"
    return pl.parse_program(
        f"""
@pl.program
class ScalarMetadata:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[2,8], pl.{dtype}], width: pl.Scalar[pl.INDEX],
               seen: pl.InOut[pl.Tensor[[2,8], pl.{dtype}]],
               out: pl.InOut[pl.Tensor[[2,8], pl.{dtype}]]) -> pl.Tensor[[2,8], pl.{dtype}]:
        source = pl.load(x, [0,0], [2,8])
        narrow = pl.tile.set_validshape(source, 2, width)
        seen = pl.store(narrow, [0,0], seen)
        scalar = pl.tile.read({selected}, {point})
        pl.tile.write(source, [1,7], scalar)
        out = pl.store(source, [0,0], out)
        return out
"""
    )


@pytest.mark.parametrize("planner", PLANNERS)
def test_static_scalar_use_bridges_a_dynamically_promoted_root(planner):
    program = lower(scalar_metadata_program(), planner)
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    assert not diagnostics(function)
    (read,) = calls(program, "buffer.read")
    (write,) = calls(program, "buffer.write")
    for call in (read, write):
        assert isinstance(call.args[0].type, ir.BufferType)
        assert list(call.args[0].type.valid_shape) == [2, 8]
    assert "pto.tgetval" in codegen.PTOCodegen().generate(program)


@pytest.mark.parametrize(
    "point,dynamic_source,dtype",
    [("[0,width]", False, "FP32"), ("[0,0]", True, "FP32"), ("[0,0]", False, "FP16")],
)
def test_unqualified_scalar_points_delegate_before_buffer_lowering(point, dynamic_source, dtype):
    original = scalar_metadata_program(point, dynamic_source, dtype)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        before = original
        for transform in manager.passes[:boundary]:
            before = transform(before)
        snapshot = ir.serialize(before)
        selected = passes.select_buffer_ir_functions()(before)
    assert ir.serialize(before) == snapshot
    function = selected.get_function("kernel")
    previous = before.get_function("kernel")
    assert function is not None and previous is not None
    assert not dict(function.attrs).get("buffer_ir_selected", False)
    ir.assert_structural_equal(function.body, previous.body, enable_auto_mapping=False)


@pytest.mark.ptoas_compile
def test_frontend_scalar_bridge_cpp_preserves_native_scalar_exemption(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    program = lower(frontend_scalar_program(True), passes.MemoryPlanner.PYPTO)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        logical = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical = passes.lower_sync()(logical)
    source, output = tmp_path / "scalar.pto", tmp_path / "scalar.cpp"
    source.write_text(codegen.PTOCodegen().generate(physical))
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    cpp = output.read_text()
    assert ".GetValue(" in cpp and ".SetValue(" in cpp
    assert "pipe_barrier(PIPE_S)" not in cpp


@pytest.mark.parametrize("width", [8, 16])
@pytest.mark.parametrize("planner", PLANNERS)
def test_scalar_tile_window_requires_the_inherited_dense_pitch(width, planner):
    original = pl.parse_program(f"""
@pl.program
class ScalarWindow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[4,16], pl.FP32],
               out: pl.InOut[pl.Tensor[[4,16], pl.FP32]]) -> pl.Tensor[[4,16], pl.FP32]:
        whole = pl.load(x, [0,0], [4,16])
        window = pl.tile.slice(whole, [2,{width}], [1,0])
        scalar = pl.tile.read(window, [1,0])
        pl.tile.write(window, [1,0], scalar)
        out = pl.store(whole, [0,0], out)
        return out
""")
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        before = original
        for transform in manager.passes[:boundary]:
            before = transform(before)
        snapshot = ir.serialize(before)
        selected = passes.select_buffer_ir_functions()(before)
    assert ir.serialize(before) == snapshot
    function = selected.get_function("kernel")
    previous = before.get_function("kernel")
    assert function is not None and previous is not None
    assert function.attrs["buffer_ir_selected"] == (width == 16)
    if width == 8:
        ir.assert_structural_equal(function.body, previous.body, enable_auto_mapping=False)
    else:
        result = lower(original, planner)
        assert not diagnostics(result.get_function("kernel"))
        text = codegen.PTOCodegen().generate(result)
        assert "pto.tgetval" in text and "pto.tsetval" in text


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("enabled", [False, True])
def test_discarded_scalar_tile_reads_delegate_before_lowering(planner, enabled):
    original = pl.parse_program("""
@pl.program
class DiscardedScalar:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[2,8], pl.FP32],
               out: pl.InOut[pl.Tensor[[2,8], pl.FP32]]) -> pl.Tensor[[2,8], pl.FP32]:
        source = pl.load(x, [0,0], [2,8])
        pl.tile.read(source, [0,0])
        out = pl.store(source, [0,0], out)
        return out
""")
    snapshot = ir.serialize(original)
    # Prepare the real selection input with the Buffer strategy. The disabled
    # strategy deliberately omits SelectBufferIRFunctions altogether.
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        before = original
        for transform in manager.passes[:boundary]:
            before = transform(before)
    with passes.PassContext([], enable_buffer_ir=enabled, memory_planner=planner):
        selected = passes.select_buffer_ir_functions()(before)
        result = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(original)
    previous = before.get_function("kernel")
    function = selected.get_function("kernel")
    lowered = result.get_function("kernel")
    assert previous is not None and function is not None and lowered is not None
    assert not function.attrs["buffer_ir_selected"]
    ir.assert_structural_equal(function.body, previous.body, enable_auto_mapping=False)
    assert lowered.ir_stage == ir.FunctionIRStage.Functional
    assert calls(result, "tile.read") and not calls(result, "buffer.read")
    assert ir.serialize(original) == snapshot


@pytest.mark.parametrize("planner", PLANNERS)
def test_nested_scalar_tile_read_requires_an_original_ssa_definition(planner):
    original = frontend_scalar_program(True)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        before = original
        for transform in manager.passes[:boundary]:
            before = transform(before)

        class NestScalarRead(ir.IRMutator):
            def visit_assign_stmt(self, op):
                if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.read").name:
                    return ir.AssignStmt(
                        op.var,
                        ir.Add(
                            op.value, ir.ConstFloat(1.0, ir.DataType.FP32, op.span), ir.DataType.FP32, op.span
                        ),
                        op.span,
                    )
                return super().visit_assign_stmt(op)

        nested = NestScalarRead().visit_program(before)
        snapshot = ir.serialize(nested)
        selected = passes.select_buffer_ir_functions()(nested)
    function = selected.get_function("kernel")
    previous = nested.get_function("kernel")
    assert function is not None and previous is not None and not function.attrs["buffer_ir_selected"]
    ir.assert_structural_equal(function.body, previous.body, enable_auto_mapping=False)
    assert ir.serialize(nested) == snapshot


def reverse_scalar_program(last_col, family="fp32_column"):
    dtype = "INT32" if family == "int32_row" else "FP32"
    reduction = (
        """
        scratch = pl.create_tile([8,16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        maximum = pl.tile.row_max(source, scratch)"""
        if family == "fp32_column"
        else ""
    )
    operands = "maximum, [row,0]" if family == "fp32_column" else "source, [row,2]"
    return pl.parse_program(f"""
@pl.program
class ReverseScalarPoints:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,16], pl.{dtype}],
               out: pl.InOut[pl.Tensor[[8,16], pl.{dtype}]]) -> pl.Tensor[[8,16], pl.{dtype}]:
        source = pl.load(x, [0,0], [8,16]){reduction}
        for i in pl.range(0, 8, 2):
            row = 6 - i
            col = pl.cast({last_col} - i, pl.INT32)
            value = pl.tile.read({operands})
            pl.tile.write(source, [row,col], value)
        out = pl.store(source, [0,0], out)
        return out
""")


@pytest.mark.parametrize("last_col", [15, 16])
def test_reverse_local_scalar_points_require_a_lexical_bound(last_col):
    program = lower(reverse_scalar_program(last_col), passes.MemoryPlanner.PYPTO)
    function = program.get_function("kernel")
    assert function is not None
    if last_col == 16:
        # The first iteration is out of range even though the scan then enters the tile.
        assert function.ir_stage == ir.FunctionIRStage.Functional
        assert calls(program, "tile.write")
        return
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    assert not diagnostics(function)
    (read,) = calls(program, "buffer.read")
    assert list(read.args[0].type.shape) == [8, 1]
    (write,) = calls(program, "buffer.write")
    assert list(write.args[0].type.shape) == [8, 16]
    emitted = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    assert "pto.tgetval" in emitted and "pto.tsetval" in emitted
    # The dense row-major write has two nonzero coordinates and needs a
    # flattened index. The reduction column needs only its runtime row.
    assert "arith.muli" in emitted and "arith.addi" in emitted


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("family", ["fp32_column", "fp32_row", "int32_row"])
def test_reverse_local_scalar_points_compile_with_native_sync(tmp_path, family):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    program = lower(reverse_scalar_program(15, family), passes.MemoryPlanner.PYPTO)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        logical = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical = passes.lower_sync()(logical)
    source, output = tmp_path / "reverse.pto", tmp_path / "reverse.cpp"
    source.write_text(codegen.PTOCodegen().generate(physical, emit_tile_addr=True, emit_source_loc=False))
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    cpp = output.read_text()
    assert ("TROWMAX(" in cpp) == (family == "fp32_column")
    assert ".GetValue(" in cpp and ".SetValue(" in cpp
    assert "pipe_barrier(PIPE_S)" not in cpp
    _check_native_scalar_values(cpp, family, tmp_path)


def _check_native_scalar_values(cpp, family, tmp_path):
    """Execute PTOAS's generated GetValue/SetValue with the real CPU ISA headers."""
    compiler = shutil.which("g++-15") or shutil.which("g++")
    isa_root = os.environ.get("PTO_ISA_ROOT")
    if compiler is None or not isa_root:
        return
    major = subprocess.check_output([compiler, "-dumpversion"], text=True).split(".")[0]
    if int(major) < 15:
        return
    include = Path(isa_root) / "include"
    assert (include / "pto/pto-inst.hpp").is_file()
    ctype = "int32_t" if family == "int32_row" else "float"
    input_col = 15 if family == "fp32_column" else 2
    # Each row has distinct values, so using the logical column as a flat
    # offset or skipping the loop stride changes the final tile.
    source = tmp_path / "probe.cpp"
    source.write_text(
        "#include <array>\n#include <cstdint>\n"
        + cpp
        + f"""
int main() {{
    pto::NPU_MEMORY_INIT();
    std::array<{ctype}, 128> input{{}}, output{{}}, expected{{}};
    for (int j = 0; j < 128; ++j) input[j] = j;
    expected = input;
    for (int i = 0; i < 8; i += 2) {{
        int row = 6 - i, col = 15 - i;
        expected[row * 16 + col] = input[row * 16 + {input_col}];
    }}
    kernel(input.data(), output.data());
    return output == expected ? 0 : 1;
}}
"""
    )
    binary = tmp_path / "probe"
    result = subprocess.run(
        [
            compiler,
            "-std=c++23",
            "-pthread",
            "-D__CPU_SIM",
            "-D__DAV_VEC__",
            "-DPTO_CPU_MAX_THREADS=1",
            f"-I{include}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
