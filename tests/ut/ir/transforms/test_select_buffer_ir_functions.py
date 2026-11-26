# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Default selection must prove a complete logical kernel before storage mutation."""

import pypto.language as pl
import pytest
import torch
from pypto import DataType, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.pypto_core import codegen
from pypto.pypto_core import ir as core_ir

from .test_runtime_valid_broadcast import check_runtime_commands

pytestmark = pytest.mark.usefixtures("ascend_backend")

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
_VECTOR = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "abs": torch.abs,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "neg": torch.neg,
    "relu": torch.relu,
    "log": torch.log,
    "recip": torch.reciprocal,
}
_BINARY = {"add", "sub", "mul", "div", "maximum", "minimum"}
_SCALAR = {
    "adds": torch.add,
    "subs": torch.sub,
    "muls": torch.mul,
    "divs": torch.div,
    "maximums": torch.maximum,
    "minimums": torch.minimum,
    "lrelu": lambda x, s: torch.where(x > 0, x, x * s),
}


def _program(operation, dtype="FP32", scalar_dtype="FP32"):
    if operation == "full":
        body = f"value = pl.tile.full([16, 32], pl.{dtype}, 0.5)"
    else:
        args = "x, y" if operation in _BINARY else "x, factor" if operation in _SCALAR else "x"
        if scalar_dtype == "INDEX":
            args = args.replace("factor", "pl.cast(factor, pl.INT64)")
        body = (
            "x = pl.load(a, [0, 0], [16, 32])\n"
            "        y = pl.load(b, [0, 0], [16, 32])\n"
            f"        value = pl.tile.{operation}({args})"
        )
    return pl.parse_program(f"""
@pl.program
class Recipes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.{dtype}], b: pl.Tensor[[16, 32], pl.{dtype}],
               factor: pl.Scalar[pl.{scalar_dtype}], out: pl.Out[pl.Tensor[[16, 32], pl.{dtype}]]
               ) -> pl.Tensor[[16, 32], pl.{dtype}]:
        {body}
        return pl.store(value, [0, 0], out)
""")


def _lower(program, planner, enabled=None):
    options = {} if enabled is None else {"enable_buffer_ir": enabled}
    with passes.PassContext([], memory_planner=planner, **options):
        return PassManager(OptimizationStrategy.Default).run_passes(program)


def _kernel(program: ir.Program) -> ir.Function:
    function = program.get_function("kernel")
    assert function is not None
    return function


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("operation", [*_VECTOR, *_SCALAR, "full"])
def test_default_recipes_select_emit_roundtrip_and_execute(planner, enabled, operation):
    program = _lower(_program(operation), planner, enabled)
    function = _kernel(program)
    assert function is not None
    expected_stage = ir.FunctionIRStage.Functional if enabled is False else ir.FunctionIRStage.Buffer
    assert function.ir_stage == expected_stage
    if enabled is False:
        return
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    text = codegen.PTOCodegen().generate(restored, emit_tile_addr=False, emit_source_loc=False)
    assert "pto.tstore" in text
    a = torch.linspace(0.25, 1.25, 512).reshape(16, 32)
    if operation in ("abs", "neg", "relu", "lrelu"):
        a -= 0.75
    b = torch.linspace(0.5, 2, 512).reshape(16, 32)
    scalar = 0.5
    out = torch.full_like(a, -777)
    if operation == "full":
        expected = torch.full_like(a, scalar)
    elif operation in _SCALAR:
        expected = _SCALAR[operation](a, torch.tensor(scalar))
    else:
        expected = _VECTOR[operation](a, b) if operation in _BINARY else _VECTOR[operation](a)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](a, b, scalar, out)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("rhs_rows", [1, 16])
def test_elementwise_selection_requires_matching_physical_shapes(planner, enabled, rhs_rows):
    program = pl.parse_program(f"""
@pl.program
class PhysicalShapes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32], valid_shape=[1, 32])
        y = pl.load(b, [0, 0], [{rhs_rows}, 32], valid_shape=[1, 32])
        value = pl.add(x, y)
        return pl.store(value, [0, 0], out)
""")
    lowered = _lower(program, planner, enabled)
    selected = enabled is not False and rhs_rows == 16
    assert _kernel(lowered).ir_stage == (
        ir.FunctionIRStage.Buffer if selected else ir.FunctionIRStage.Functional
    )
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=False, emit_source_loc=False)
    assert "pto.tadd" in text and "pto.tstore" in text


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype", ["FP16", "INT32", "BF16"])
@pytest.mark.parametrize("operation", ["add", "mul"])
def test_arithmetic_dtype_contract_is_narrower_than_transfer(planner, dtype, operation):
    program = _lower(_program(operation, dtype), planner)
    function = _kernel(program)
    assert function is not None
    assert function.ir_stage == (
        ir.FunctionIRStage.Functional if dtype == "BF16" else ir.FunctionIRStage.Buffer
    )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("scalar_dtype", ["INT32", "INT64", "INDEX", "FP16", "BF16", "FP32"])
def test_element_scalar_conversion_runs_in_debug(planner, scalar_dtype):
    program = _lower(_program("muls", scalar_dtype=scalar_dtype), planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    a = torch.linspace(-1, 1, 512).reshape(16, 32)
    out = torch.empty_like(a)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](a, a, 2, out)
    torch.testing.assert_close(out, a * 2)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype", ["FP16", "BF16", "FP32", "INT32"])
def test_partial_typed_transfer_preserves_unwritten_rows_and_columns(planner, dtype):
    program = pl.parse_program(f"""
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.{dtype}],
               out: pl.InOut[pl.Tensor[[24, 48], pl.{dtype}]]) -> pl.Tensor[[24, 48], pl.{dtype}]:
        x = pl.load(a, [0, 0], [16, 32], valid_shape=[13, 24])
        return pl.store(x, [3, 8], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
        "FP32": torch.float32,
        "INT32": torch.int32,
    }[dtype]
    a = (torch.arange(512).reshape(16, 32) * 13 - 1777).to(torch_dtype)
    out = torch.full((24, 48), -777, dtype=torch_dtype)
    expected = out.clone()
    expected[3:16, 8:32] = a[:13, :24]
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    namespace["kernel"](a, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,selected", [("FP32", True), ("FP16", False), ("BF16", False)])
def test_row_alignment_depends_on_element_width(dtype, selected):
    program = pl.parse_program(f"""
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 8], pl.{dtype}],
               out: pl.Out[pl.Tensor[[16, 8], pl.{dtype}]]) -> pl.Tensor[[16, 8], pl.{dtype}]:
        x = pl.load(a, [0, 0], [16, 8])
        return pl.store(x, [0, 0], out)
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    assert (_kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer) is selected


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype", ["FP32", "INT32"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_explicit_store_shapes_keep_partial_transfer_on_tile(planner, dtype, dynamic):
    # Explicit partition shapes add a fourth store operand. Selecting this
    # otherwise-supported kernel used to fail in LowerTileToBuffer, after
    # storage planning had already committed to the Buffer representation.
    rows = "n" if dynamic else "13"
    program = pl.parse_program(f"""
@pl.program
class ExplicitStoreShapes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.{dtype}], n: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[24, 48], pl.{dtype}]]) -> pl.Tensor[[24, 48], pl.{dtype}]:
        x = pl.load(a, [0, 0], [16, 32], valid_shape=[{rows}, 24])
        return pl.store(x, [3, 8], out, shapes=[{rows}, 24])
""")
    lowered = _lower(program, planner)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Functional
    stores: list[ir.Call] = []

    class StoreCollector(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == ir.get_op("tile.store").name:
                stores.append(op)
            super().visit_call(op)

    StoreCollector().visit_function(function)
    assert len(stores) == 1
    assert len(stores[0].args) == 4
    shapes = stores[0].args[3]
    assert isinstance(shapes, ir.MakeTuple)
    assert len(shapes.elements) == 2
    store_rows, store_cols = shapes.elements
    if dynamic:
        assert isinstance(store_rows, ir.Var)
    else:
        assert isinstance(store_rows, ir.ConstInt)
        assert store_rows.value == 13
    assert isinstance(store_cols, ir.ConstInt)
    assert store_cols.value == 24
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=False, emit_source_loc=False)
    assert "pto.tstore" in text
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    torch_dtype = torch.float32 if dtype == "FP32" else torch.int32
    a = torch.arange(512, dtype=torch_dtype).reshape(16, 32)
    for n in (7, 13):
        out = torch.full((24, 48), -777, dtype=torch_dtype)
        expected = out.clone()
        valid_rows = n if dynamic else 13
        expected[3 : 3 + valid_rows, 8:32] = a[:valid_rows, :24]
        namespace["kernel"](a, n, out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_unsupported_scalar_power_keeps_the_whole_kernel_on_tile(planner):
    program = pl.parse_program("""
@pl.program
class ScalarPower:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], factor: pl.Scalar[pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32])
        value = pl.tile.muls(x, factor ** 2.0)
        return pl.store(value, [0, 0], out)
""")
    program = _lower(program, planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Functional


def _control_flow_program(kind, step="1", change_alias=False):
    header = (
        "for left, right, row, gm in pl.while_(init_values=(lhs, rhs, 0, out)):\n"
        "            pl.cond(row < count)"
        if kind == "while"
        else f"for i, (left, right, row, gm) in pl.range(0, count, {step}, init_values=(lhs, rhs, 0, out)):"
    )
    backedge = "other" if change_alias else "stored"
    return pl.parse_program(f"""
@pl.program
class ControlFlow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               out: pl.InOut[pl.Tensor[[32, 32], pl.FP32]], other: pl.InOut[pl.Tensor[[32, 32], pl.FP32]],
               count: pl.Scalar[pl.INT64], flag: pl.Scalar[pl.BOOL], step: pl.Scalar[pl.INT64]
               ) -> tuple[pl.Tensor[[32, 32], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        {header}
            if flag:
                chosen = pl.yield_(left)
            else:
                chosen = pl.yield_(right)
            stored = pl.store(chosen, [row, 0], gm)
            final_left, final_right, final_row, final_gm = pl.yield_(right, left, row + 1, {backedge})
        value = pl.sub(final_left, pl.mul(final_right, rhs))
        result = pl.store(value, [final_row, 0], final_gm)
        saved = pl.store(lhs, [0, 0], other)
        return result, saved
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["for", "while"])
@pytest.mark.parametrize("enabled", [None, True, False])
def test_default_control_flow_preserves_swaps_live_inputs_and_gm_carries(planner, kind, enabled):
    if enabled is False:
        # Selection is disabled before planning. The legacy DSA-RP while path
        # independently rejects this storage pattern; it is not a Buffer fallback.
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
            selected = passes.select_buffer_ir_functions()(_control_flow_program(kind))
        assert _kernel(selected).attrs["buffer_ir_selected"] is False
        return
    program = _lower(_control_flow_program(kind), planner, enabled)
    assert _kernel(program).ir_stage == (
        ir.FunctionIRStage.Functional if enabled is False else ir.FunctionIRStage.Buffer
    )
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    a = torch.linspace(-1, 1, 512).reshape(16, 32)
    b = torch.linspace(0.5, 2, 512).reshape(16, 32)
    for count in range(4):
        for flag in (False, True):
            out = torch.full((32, 32), -777.0)
            saved = torch.full((32, 32), -555.0)
            expected = out.clone()
            left, right = a, b
            for row in range(count):
                expected[row : row + 16] = left if flag else right
                left, right = right, left
            expected[count : count + 16] = left - right * b
            namespace["kernel"](a, b, out, saved, count, flag, 1)
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(saved[:16], a)
            assert torch.all(saved[16:] == -555)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "kind,step,change_alias",
    [
        ("for", "step", False),
        ("for", "-1", False),
        ("for", "1", True),
        ("while", "1", True),
    ],
)
def test_unsupported_step_and_changing_gm_carry_are_rejected_before_storage(
    planner, kind, step, change_alias
):
    with passes.PassContext([], memory_planner=planner):
        manager = PassManager(OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = _control_flow_program(kind, step, change_alias)
        for transform in manager.passes[: boundary + 1]:
            program = transform(program)
    assert _kernel(program).attrs["buffer_ir_selected"] is False


@pytest.mark.parametrize("planner", _PLANNERS)
def test_positive_launch_stride_is_preserved_by_buffer_selection(planner):
    # The launch count is positive by the operation contract, even when its
    # numerical value is not available to this function. This admits Buffer
    # conversion, not an exact synchronization trip-count certificate.
    program = _lower(_control_flow_program("for", "pl.tile.get_block_num()"), planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.get_block_num" in str(program)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["for", "while"])
def test_scalar_carry_swaps_are_simultaneous_in_buffer_debug(planner, kind):
    header = (
        "for row, column, i in pl.while_(init_values=(0, 16, 0)):\n            pl.cond(i < count)"
        if kind == "while"
        else "for j, (row, column, i) in pl.range(count, init_values=(0, 16, 0)):"
    )
    program = pl.parse_program(f"""
@pl.program
class ScalarSwap:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], out: pl.InOut[pl.Tensor[[32, 64], pl.FP32]],
               count: pl.Scalar[pl.INT64]) -> pl.Tensor[[32, 64], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32])
        {header}
            final_row, final_column, final_i = pl.yield_(column, row, i + 1)
        return pl.store(x, [final_row, final_column], out)
""")
    program = _lower(program, planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    for count in range(4):
        out = torch.full((32, 64), -777.0)
        expected = out.clone()
        row, column = (0, 16) if count % 2 == 0 else (16, 0)
        expected[row : row + 16, column : column + 32] = a
        namespace["kernel"](a, out, count)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("same_alias", [True, False])
def test_branch_gm_aliases_must_resolve_to_one_parameter(planner, same_alias):
    other = "a" if same_alias else "b"
    program = pl.parse_program(f"""
@pl.program
class SelectGM:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], flag: pl.Scalar[pl.BOOL]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        if flag:
            source = pl.yield_(a)
        else:
            source = pl.yield_({other})
        x = pl.load(source, [0, 0], [16, 32])
        return pl.store(x, [0, 0], out)
""")
    result = _lower(program, planner)
    assert _kernel(result).ir_stage == (
        ir.FunctionIRStage.Buffer if same_alias else ir.FunctionIRStage.Functional
    )
    restored = ir.deserialize(ir.serialize(result))
    assert isinstance(restored, ir.Program)
    namespace = {}
    exec(torch_codegen(restored), namespace)  # noqa: S102
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    b = -a - 1
    for flag in (False, True):
        out = torch.empty_like(a)
        namespace["kernel"](a, b, out, flag)
        torch.testing.assert_close(out, a if flag or same_alias else b)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("enabled", [None, False])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_static_windows_use_default_pipeline_and_preserve_source(
    tmp_path, ascend_backend, planner, nested, enabled
):
    window = (
        "outer = pl.tile.slice(whole, [32, 48], [16, 16])\n"
        "        window = pl.tile.slice(outer, [16, 32], [0, 16])"
        if nested
        else "window = pl.tile.slice(whole, [16, 32], [16, 32])"
    )
    program = pl.parse_program(f"""
@pl.program
class StaticWindows:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
               placed_out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        {window}
        doubled = pl.add(window, window)
        canvas = pl.tile.full([32, 32], dtype=pl.FP32, value=-777.0)
        placed = pl.tile.assemble(canvas, doubled, [16, 0])
        pl.store(placed, [0, 0], placed_out)
        return pl.store(whole, [0, 0], out)
""")
    lowered = _lower(program, planner, enabled)
    assert _kernel(lowered).ir_stage == (
        ir.FunctionIRStage.Functional
        if enabled is False or ascend_backend == BackendType.Ascend950
        else ir.FunctionIRStage.Buffer
    )
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    text = codegen.PTOCodegen().generate(
        restored, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    assert "pto.subview" in text
    namespace = {}
    exec(torch_codegen(restored), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    out = torch.empty_like(x)
    placed = torch.empty((32, 32))
    namespace["kernel"](x, out, placed)
    torch.testing.assert_close(out, x)
    expected = torch.full_like(placed, -777)
    expected[16:] = x[16:32, 32:] * 2
    torch.testing.assert_close(placed, expected)
    if find_ptoas_binary() is not None:
        source = tmp_path / "window.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        arch = "a2" if ascend_backend == BackendType.Ascend910 else "a5"
        _run_ptoas(str(source), str(tmp_path / "window.cpp"), [f"--pto-arch={arch}", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("column", [0, 16, 32])
def test_overlapping_assemble_snapshots_source_before_write(tmp_path, planner, column):
    program = pl.parse_program(f"""
@pl.program
class OverlappingAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]) -> pl.Tensor[[32, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [32, 64])
        source = pl.tile.slice(whole, [32, 32], [0, 0])
        placed = pl.tile.assemble(whole, source, [0, {column}])
        return pl.store(placed, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    # Exact self-writes need no copy; other cases still have overlapping byte envelopes.
    assert text.count("pto.alloc_tile") == (1 if column == 0 else 2)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    expected = x.clone()
    expected[:, column : column + 32] = x[:, :32]
    torch.testing.assert_close(out, expected)
    if find_ptoas_binary() is not None:
        source = tmp_path / "assemble.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "assemble.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_disjoint_assemble_fits_one_large_allocation(tmp_path, planner, reverse, nested):
    row = 64 if reverse else 0
    target_row = 0 if reverse else 64
    window = (
        f"outer = pl.tile.slice(whole, [96, 256], [{row // 2}, 0])\n"
        f"        source = pl.tile.slice(outer, [64, 256], [{row // 2}, 0])"
        if nested
        else f"source = pl.tile.slice(whole, [64, 256], [{row}, 0])"
    )
    program = pl.parse_program(f"""
@pl.program
class DisjointAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[128, 256], pl.FP32],
               out: pl.Out[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:
        whole = pl.load(x, [0, 0], [128, 256])
        {window}
        placed = pl.tile.assemble(whole, source, [{target_row}, 0])
        return pl.store(placed, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    # One parent is 128 KiB; an unnecessary snapshot exceeds the 184 KiB Vec limit.
    assert text.count("pto.alloc_tile") == 1
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    expected = x.clone()
    expected[target_row : target_row + 64] = x[row : row + 64]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "disjoint.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "disjoint.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("reslice", [False, True])
def test_chained_disjoint_assembles_preserve_allocation_origin(tmp_path, planner, reverse, reslice):
    row = 96 if reverse else 0
    targets = [64, 32, 0] if reverse else [32, 64, 96]
    source = f"source = pl.tile.slice(first, [32, 256], [{targets[0]}, 0])" if reslice else ""
    source_name = "source" if reslice else "window"
    program = pl.parse_program(f"""
@pl.program
class ChainedAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[128, 256], pl.FP32],
               out: pl.Out[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:
        whole = pl.load(x, [0, 0], [128, 256])
        window = pl.tile.slice(whole, [32, 256], [{row}, 0])
        first = pl.tile.assemble(whole, window, [{targets[0]}, 0])
        {source}
        second = pl.tile.assemble(first, {source_name}, [{targets[1]}, 0])
        third = pl.tile.assemble(second, {source_name}, [{targets[2]}, 0])
        return pl.store(third, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert text.count("pto.alloc_tile") == 1
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    torch.testing.assert_close(out, x[row : row + 32].repeat(4, 1), rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source_file = tmp_path / "chained.pto"
        source_file.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source_file), str(tmp_path / "chained.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
def test_chained_overlapping_assemble_snapshots_updated_parent(tmp_path, planner):
    @pl.program
    class ChainedOverlap:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]
        ) -> pl.Tensor[[32, 64], pl.FP32]:
            whole = pl.load(x, [0, 0], [32, 64])
            window = pl.tile.slice(whole, [8, 64], [0, 0])
            first = pl.tile.assemble(whole, window, [8, 0])
            updated = pl.tile.slice(first, [16, 64], [8, 0])
            second = pl.tile.assemble(first, updated, [16, 0])
            return pl.store(second, [0, 0], out)

    lowered = _lower(ChainedOverlap, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert text.count("pto.alloc_tile") == 2
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    out = torch.empty_like(x)
    namespace["kernel"](x, out)
    expected = x.clone()
    expected[8:16] = x[:8]
    expected[16:] = expected[8:24].clone()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source_file = tmp_path / "chained_overlap.pto"
        source_file.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source_file), str(tmp_path / "chained_overlap.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("view_kind", ["allocation", "full_width"])
@pytest.mark.parametrize("source_dtype,target_dtype", [("FP32", "INT32"), ("FP16", "FP32")])
def test_contiguous_reinterpret_preserves_bit_patterns(
    tmp_path, planner, view_kind, source_dtype, target_dtype
):
    rows = 16 if view_kind == "full_width" else 32
    cols = 32 if source_dtype == "FP16" else 64
    view = {
        "allocation": "window = whole",
        "full_width": "window = pl.tile.slice(whole, [16, 64], [16, 0])",
    }[view_kind]
    program = pl.parse_program(f"""
@pl.program
class Reinterpret:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 64], pl.{source_dtype}],
               out: pl.Out[pl.Tensor[[{rows}, {cols}], pl.{target_dtype}]]
               ) -> pl.Tensor[[{rows}, {cols}], pl.{target_dtype}]:
        whole = pl.load(x, [0, 0], [32, 64])
        {view}
        bits = pl.tile.reinterpret_view(window, pl.{target_dtype})
        return pl.store(bits, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    dtype = torch.float16 if source_dtype == "FP16" else torch.float32
    target = torch.int32 if target_dtype == "INT32" else torch.float32
    x = torch.arange(2048, dtype=dtype).reshape(32, 64)
    out = torch.empty((rows, cols), dtype=target)
    namespace["kernel"](x, out)
    expected = (x[16:] if view_kind == "full_width" else x).view(target)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "reinterpret.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "reinterpret.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("operation", ["reinterpret", "reshape"])
def test_strided_relabels_remain_outside_default_buffer_contract(planner, rows, operation):
    dtype = "INT32" if operation == "reinterpret" else "FP32"
    expression = (
        "pl.tile.reinterpret_view(window, pl.INT32)"
        if operation == "reinterpret"
        else f"pl.reshape(window, [{rows}, 32])"
    )
    program = pl.parse_program(f"""
@pl.program
class StridedRelabel:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[{rows}, 32], pl.{dtype}]]
               ) -> pl.Tensor[[{rows}, 32], pl.{dtype}]:
        whole = pl.load(x, [0, 0], [32, 64])
        window = pl.tile.slice(whole, [{rows}, 32], [0, 0])
        bits = {expression}
        return pl.store(bits, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_reshaped_allocation_is_a_native_window_parent(tmp_path, planner):
    @pl.program
    class ReshapedParent:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
        ) -> pl.Tensor[[8, 64], pl.FP32]:
            whole = pl.load(x, [0, 0], [32, 64])
            parent = pl.reshape(whole, [16, 128])
            window = pl.tile.slice(parent, [8, 64], [8, 32])
            return pl.store(window, [0, 0], out)

    lowered = _lower(ReshapedParent, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    out = torch.empty((8, 64))
    namespace["kernel"](x, out)
    torch.testing.assert_close(out, x.reshape(16, 128)[8:, 32:96])
    if find_ptoas_binary() is not None:
        source = tmp_path / "parent.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "parent.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
def test_conflicting_typed_window_parents_stay_tile(planner):
    @pl.program
    class ConflictingParents:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 64], pl.FP32]]
        ) -> pl.Tensor[[32, 64], pl.FP32]:
            whole = pl.load(x, [0, 0], [64, 64])
            first = pl.tile.slice(whole, [16, 64], [0, 0])
            out = pl.store(first, [0, 0], out)
            parent = pl.reshape(whole, [32, 128])
            second = pl.tile.slice(parent, [16, 64], [16, 32])
            return pl.store(second, [16, 0], out)

    assert _kernel(_lower(ConflictingParents, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_fillpad_defines_zero_suffix_before_cast_and_store(planner):
    @pl.program
    class RuntimeRowFillPad:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tensor[[16, 32], pl.BF16],
            rows: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            prefix = pl.load(source, [0, 0], [16, 32], valid_shape=[rows, 32])
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.zero)
            widened = pl.cast(padded, pl.FP32)
            return pl.store(widened, [0, 0], output)

    program = _run_autosync(RuntimeRowFillPad, planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    pto = codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.tfillpad" in pto
    assert "pto.texpands" in pto
    assert "pto.tcvt" in pto
    assert "pad=1" in pto
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    original = torch.arange(1, 513, dtype=torch.bfloat16).reshape(16, 32)
    for rows in (0, 3, 16):
        output = torch.full((16, 32), -1.0)
        namespace["kernel"](original, rows, output)
        expected = original.float()
        expected[rows:] = 0
        torch.testing.assert_close(output, expected)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("width", [5, None], ids=["static-prefix", "unproved-runtime-prefix"])
@pytest.mark.parametrize("convert", [False, True])
def test_minimum_fillpad_before_row_max(width, convert, tmp_path):
    extent = "cols" if width is None else str(width)
    conversion = (
        "expanded = pl.exp(padded)\n        observed = pl.cast(pl.cast(expanded, pl.BF16), pl.FP32)"
        if convert
        else "observed = padded"
    )
    program = pl.parse_program(f"""
@pl.program
class MinPad:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], cols: pl.Scalar[pl.INDEX],
               output: pl.Out[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        src = pl.load(source, [0, 0], [16, 32])
        prefix = pl.set_validshape(src, 16, {extent})
        padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
        {conversion}
        scratch = pl.tile.create([16, 32], dtype=pl.FP32)
        maximum = pl.row_max(observed, scratch)
        return pl.store(maximum, [0, 0], output)
""")
    lowered = _run_autosync(program)
    pto = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.tfillpad" in pto and "pad=3" in pto and "pto.trowmax" in pto
    assert ("pto.texpands" in pto) is (width is None)
    if convert:
        assert pto.count("pto.tcvt") == 2 and "pto.texp " in pto
        assert any("dtype=bf16" in line and "pad=3" in line for line in pto.splitlines())
    pto_path = tmp_path / "min_pad.pto"
    pto_path.write_text(pto)
    _run_ptoas(
        str(pto_path),
        str(tmp_path / "min_pad.cpp"),
        ["--pto-arch=a2", "--pto-level=level3", "--enable-insert-sync=false"],
    )
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    source = -torch.arange(1, 513, dtype=torch.float32).reshape(16, 32)
    width = 5 if width is None else width
    source[:, width:] = 1
    output = torch.empty(16, 1)
    namespace["kernel"](source, width, output)
    expected = source[:, :width].amax(dim=1, keepdim=True)
    if convert:
        expected = expected.exp().bfloat16().float()
    torch.testing.assert_close(output, expected)


def test_bf16_transport_pad_tag_does_not_qualify_minimum_fillpad():
    program = pl.parse_program("""
@pl.program
class BfloatMinPad:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.BF16],
               destination: pl.Out[pl.Tensor[[16, 32], pl.BF16]]) -> pl.Tensor[[16, 32], pl.BF16]:
        loaded = pl.load(source, [0, 0], [16, 32])
        prefix = pl.set_validshape(loaded, 16, 16)
        padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
        return pl.store(padded, [0, 0], destination)
""")
    assert _kernel(_lower(program, passes.MemoryPlanner.PYPTO)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("operation", ["add", "mul"])
@pytest.mark.parametrize("explicit_padding", [False, True])
def test_full_binary_with_materialized_zero_tail_keeps_full_native_footprint(
    operation, explicit_padding, tmp_path
):
    def program(materialize):
        operand = "pl.fillpad(partial, pad_value=pl.PadValue.zero)" if materialize else "partial"
        return pl.parse_program(f"""
@pl.program
class TailArithmetic:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, full: pl.Tensor[[16, 32], pl.FP32], tail: pl.Tensor[[16, 32], pl.FP32],
               rows: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs = pl.load(full, [0, 0], [16, 32])
        partial = pl.load(tail, [0, 0], [16, 32], valid_shape=[rows, 32])
        rhs = {operand}
        return pl.store(pl.tile.{operation}(lhs, rhs), [0, 0], out)
""")

    authored = program(explicit_padding)
    original = _lower(authored, passes.MemoryPlanner.PYPTO)
    if operation == "mul" and not explicit_padding:
        refused = _kernel(original)
        assert refused.ir_stage == ir.FunctionIRStage.Functional
        assert refused.attrs["buffer_admission_operation"] == "tile.mul"
        return
    assert _kernel(original).ir_stage == ir.FunctionIRStage.Buffer
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = passes.insert_sync()(original)
    lowered = _run_autosync(authored)
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.tfillpad" in text and f"pto.t{operation}" in text and "pto.tstore" in text
    _compile_dynamic_state_native(lowered, tmp_path, passes.MemoryPlanner.PYPTO)

    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    lhs = torch.randn(16, 32)
    for rows in (0, 1, 13, 16):
        rhs = torch.rand_like(lhs)
        rhs[rows:] = torch.nan
        output = torch.empty_like(lhs)
        namespace["kernel"](lhs, rhs, rows, output)
        rhs[rows:] = 0
        torch.testing.assert_close(output, _VECTOR[operation](lhs, rhs))
        check_runtime_commands(_kernel(logical), _kernel(lowered), {2: rows})


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("pitch,selected", [(32, True), (64, False)])
def test_static_valid_cast_accepts_dense_window_and_preserves_physical_pitch(pitch, selected, tmp_path):
    program = pl.parse_program(f"""
@pl.program
class WindowCast:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, {pitch}], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.BF16]]) -> pl.Tensor[[16, 32], pl.BF16]:
        whole = pl.load(source, [0, 0], [16, {pitch}])
        for row in pl.range(16):
            window = pl.tile.slice(whole, [1, 32], [row, 0])
            converted = pl.tile.cast(window, target_type=pl.BF16)
            out = pl.store(converted, [row, 0], out)
        return out
""")
    if not selected:
        refused = _kernel(_lower(program, passes.MemoryPlanner.PYPTO))
        assert refused.ir_stage == ir.FunctionIRStage.Functional
        assert refused.attrs["buffer_admission_operation"] == "tile.cast"
        return

    lowered = _run_autosync(program)
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.subview" in text and "pto.tcvt" in text
    _compile_dynamic_state_native(lowered, tmp_path, passes.MemoryPlanner.PYPTO)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    source = torch.randn(16, pitch)
    output = torch.empty(16, 32, dtype=torch.bfloat16)
    namespace["kernel"](source, output)
    torch.testing.assert_close(output, source.bfloat16())


@pytest.mark.ptoas_compile
def test_static_vec_extract_coexists_with_runtime_valid_load(tmp_path):
    program = pl.parse_program("""
@pl.program
class ExtractPaddedRows:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 64], pl.FP32], rows: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        partial = pl.load(source, [0, 0], [16, 64], valid_shape=[rows, 64])
        padded = pl.fillpad(partial, pad_value=pl.PadValue.zero)
        for row in pl.range(16):
            if row < rows:
                patch = pl.tile.extract(padded, row, 32, [1, 32], target_memory=pl.Mem.Vec)
                out = pl.store(patch, [row, 0], out)
        return out
""")
    original = _lower(program, passes.MemoryPlanner.PYPTO)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = passes.insert_sync()(original)
    lowered = _run_autosync(program)
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.set_validshape" in text and "pto.textract" in text
    _compile_dynamic_state_native(lowered, tmp_path, passes.MemoryPlanner.PYPTO)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    source = torch.randn(16, 64)
    for rows in (0, 3, 16):
        output = torch.full((16, 32), -123.0)
        namespace["kernel"](source, rows, output)
        torch.testing.assert_close(output[:rows], source[:rows, 32:])
        assert torch.all(output[rows:] == -123.0)
        check_runtime_commands(_kernel(logical), _kernel(lowered), {1: rows})


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "adjust,unknown_stride,guarded",
    [(8, False, False), (0, False, True), (9, False, True), (8, True, False)],
)
def test_minimum_fillpad_checks_runtime_column_prefix(adjust, unknown_stride, guarded, tmp_path):
    step = "unproved" if unknown_stride else "pl.tile.get_block_num()"
    program = pl.parse_program(f"""
@pl.program
class MinimumPrefix:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], unproved: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        for work in pl.range(pl.tile.get_block_idx(), 32, {step}):
            columns = (work // 8) * 8 + {adjust}
            src = pl.load(source, [0, 0], [16, 32])
            prefix = pl.set_validshape(src, 16, columns)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            scratch = pl.tile.create([16, 32], dtype=pl.FP32)
            maximum = pl.row_max(padded, scratch)
            output = pl.store(maximum, [0, 0], output)
        return output
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    actual = _kernel(lowered)
    assert (actual.ir_stage == ir.FunctionIRStage.Buffer) is not unknown_stride
    if not unknown_stride:
        text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
        assert "pto.tfillpad" in text and "pad=3" in text and "pto.set_validshape" in text
        assert ("pto.texpands" in text) is guarded
        _compile_dynamic_state_native(lowered, tmp_path, passes.MemoryPlanner.PYPTO)


@pytest.mark.parametrize(
    "limit, predicate, proved",
    [
        pytest.param(32, "columns > 0", True, id="exact-positive-observation"),
        pytest.param(33, "columns > 0", False, id="cap-exceeds-capacity"),
        pytest.param(32, "columns >= 0", False, id="zero-still-possible"),
        pytest.param(32, "remaining > 0", False, id="guard-different-ssa"),
    ],
)
def test_minimum_fillpad_uses_guarded_observed_width(limit, predicate, proved):
    program = pl.parse_program(f"""
@pl.program
class GuardedWidth:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], remaining: pl.Scalar[pl.INDEX],
               origin: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        columns = pl.min({limit}, remaining - origin)
        if {predicate}:
            src = pl.load(source, [0, 0], [16, 32])
            prefix = pl.set_validshape(src, 16, columns)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            scratch = pl.tile.create([16, 32], dtype=pl.FP32)
            maximum = pl.row_max(padded, scratch)
            output = pl.store(maximum, [0, 0], output)
        return output
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.tfillpad" in text and "pto.set_validshape" in text
    assert ("pto.texpands" in text) is not proved


@pytest.mark.parametrize(
    ("row_guard", "selected"),
    [
        ("rows > 0 and rows <= 16", True),
        ("rows >= 0 and rows <= 16", False),
        ("rows > 0 and rows <= 17", False),
    ],
)
def test_minimum_fillpad_accepts_both_runtime_extents_under_exact_guards(row_guard, selected, tmp_path):
    program = pl.parse_program(f"""
@pl.program
class GuardedBox:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], rows: pl.Scalar[pl.INDEX],
               columns: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        if {row_guard} and columns > 0 and columns <= 32:
            src = pl.load(source, [0, 0], [16, 32])
            prefix = pl.set_validshape(src, rows, columns)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            workspace = pl.tile.create([16, 32], dtype=pl.FP32)
            maximum = pl.row_max(padded, workspace)
            output = pl.store(maximum, [0, 0], output)
        return output
""")
    planner = passes.MemoryPlanner.PYPTO
    lowered = _lower(program, planner)
    assert (_kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer) is selected
    if selected:
        text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
        assert "pto.tfillpad" in text and "pto.texpands" not in text
        _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize(
    ("case", "proved"),
    [
        ("matching", True),
        ("different_extent", False),
        ("different_divisor", False),
        ("shadowed_quotient", False),
        ("narrow_formal", True),
        ("unbounded_index", False),
    ],
)
def test_minimum_fillpad_uses_the_matching_ceil_div_tile_guard(case, proved):
    # Either the ABI or a signed table read supplies an INT32 range. An
    # unrestricted INDEX can overflow before the ceiling division.
    read_extent = (
        "formal" if case in ("narrow_formal", "unbounded_index") else "pl.tensor.read(extent_values, [0])"
    )
    formal_dtype = "INDEX" if case == "unbounded_index" else "INT32"
    tail_source = "other" if case == "different_extent" else "extent"
    divisor = 16 if case == "different_divisor" else 32
    shadow = "tiles = pl.cast(other, pl.INDEX)" if case == "shadowed_quotient" else ""
    program = pl.parse_program(f"""
@pl.program
class GuardedTail:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], extent_values: pl.Tensor[[1], pl.INT32],
               other_values: pl.Tensor[[1], pl.INT32], formal: pl.Scalar[pl.{formal_dtype}],
               output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        extent = {read_extent}
        other = pl.tensor.read(other_values, [0])
        tiles = (pl.cast(extent, pl.INDEX) + {divisor - 1}) // {divisor}
        {shadow}
        for i in pl.range(tiles + 2):
            if i < tiles:
                columns = pl.min(pl.cast({tail_source}, pl.INDEX) - i * 32, 32)
                src = pl.load(source, [0, 0], [16, 32])
                prefix = pl.set_validshape(src, 16, columns)
                padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
                workspace = pl.tile.create([16, 32], dtype=pl.FP32)
                maximum = pl.row_max(padded, workspace)
                output = pl.store(maximum, [0, 0], output)
        return output
""")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_scalar_gm_dma=True):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert ("pto.texpands" in text) is not proved


@pytest.mark.parametrize(
    "case,proved",
    [
        ("matching", True),
        ("different_page", False),
        ("unproved_product_range", False),
        ("unproved_numerator_range", False),
    ],
)
def test_minimum_fillpad_tail_proof_uses_bounded_original_scalar_values(case: str, proved: bool) -> None:
    # Floor division gives the same tail implication for negative pages.
    # These pages do not index either tensor. A width proof still requires range
    # evidence for the arithmetic before the guard and within the taken arm.
    ordinal = "unbounded" if case == "unproved_product_range" else "pl.cast(table_page, pl.INDEX) - 1"
    extent = "unbounded" if case == "unproved_numerator_range" else "pl.cast(table_extent, pl.INDEX) + 1"
    product = "page + 1" if case == "different_page" else "page"
    program = pl.parse_program(f"""
@pl.program
class SignedGuardedTail:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], table: pl.Tensor[[2], pl.INT32],
               unbounded: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        table_page = pl.tensor.read(table, [0])
        table_extent = pl.tensor.read(table, [1])
        page = {ordinal}
        extent = {extent}
        pages = (extent + 31) // 32
        if page < pages:
            origin = ({product}) * 32
            remaining = extent - origin
            columns = pl.min(remaining, 32)
            src = pl.load(source, [0, 0], [16, 32])
            prefix = pl.set_validshape(src, 16, columns)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            workspace = pl.tile.create([16, 32], dtype=pl.FP32)
            maximum = pl.row_max(padded, workspace)
            output = pl.store(maximum, [0, 0], output)
        return output
""")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_scalar_gm_dma=True):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert "arith.floordivsi" in text and "pto.tfillpad" in text
    assert ("pto.texpands" in text) is not proved


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("unknown_call", [False, True])
def test_minimum_fillpad_prefers_proved_caller_bounds(tmp_path, unknown_call):
    # Every caller uses the same standalone leaf. An unrestricted caller
    # requires the guarded form. When every caller proves the tail bounds,
    # the leaf emits a single native fillpad.
    other_base = "unbounded" if unknown_call else "0"
    program = pl.parse_program(f"""
@pl.program
class CallerBoundedFillpad:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], table: pl.Tensor[[2], pl.INT32],
               page_base: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        table_page = pl.tensor.read(table, [0])
        table_extent = pl.tensor.read(table, [1])
        page = page_base + pl.cast(table_page, pl.INDEX)
        extent = pl.cast(table_extent, pl.INDEX) + 1
        pages = (extent + 31) // 32
        if page < pages:
            origin = page * 32
            remaining = extent - origin
            columns = pl.min(remaining, 32)
            src = pl.load(source, [0, 0], [16, 32])
            prefix = pl.set_validshape(src, 16, columns)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            workspace = pl.tile.create([16, 32], dtype=pl.FP32)
            maximum = pl.row_max(padded, workspace)
            output = pl.store(maximum, [0, 0], output)
        return output

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, source: pl.Tensor[[16, 32], pl.FP32], table: pl.Tensor[[2], pl.INT32],
             unbounded: pl.Scalar[pl.INDEX],
             output: pl.InOut[pl.Tensor[[16, 1], pl.FP32]]) -> pl.Tensor[[16, 1], pl.FP32]:
        total = pl.tensor.read(table, [0])
        blocks = (pl.cast(total, pl.INDEX) + 127) // 128
        for base in pl.range(blocks):
            output = self.kernel(source, table, base, output)
        return self.kernel(source, table, {other_base}, output)
""")
    snapshots = []

    def capture(transform, current):
        if transform.get_name() == "SelectBufferIRFunctions":
            snapshots.append(current.get_function("main"))

    instrument = passes.CallbackInstrument(before_pass=capture, after_pass=capture, name="capture_selection")
    planner = passes.MemoryPlanner.PYPTO
    with passes.PassContext([instrument], memory_planner=planner, enable_scalar_gm_dma=True):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    original, selected = snapshots
    function = _kernel(lowered)
    ir.assert_structural_equal(original.body, selected.body, enable_auto_mapping=True)
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    standalone = ir.Program([function], "CallerBoundedFillpadStandalone", function.span)
    serialized = ir.deserialize(ir.serialize(standalone))
    ir.assert_structural_equal(standalone, serialized, enable_auto_mapping=True)

    class WidthWitness(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.extent = None

        def visit_assign_stmt(self, op):
            name = "fillpad_width" if unknown_call else "columns"
            if op.var.name_hint.startswith(name) and isinstance(op.value, ir.Max):
                self.extent = op.value
            super().visit_assign_stmt(op)

    witness = WidthWitness()
    assert isinstance(serialized, ir.Program)
    witness.visit_function(_kernel(serialized))
    assert isinstance(witness.extent, ir.Max)
    assert isinstance(witness.extent.right, ir.ConstInt) and witness.extent.right.value == 1
    minimum = witness.extent.left
    assert isinstance(minimum, ir.Min)
    assert any(isinstance(arg, ir.ConstInt) and arg.value == 32 for arg in (minimum.left, minimum.right))
    text = codegen.PTOCodegen().generate(serialized, emit_tile_addr=True, emit_source_loc=False)
    assert text.count("pto.tfillpad") == 1 and "arith.maxsi" in text
    assert ("pto.texpands" in text) is unknown_call
    _compile_dynamic_state_native(serialized, tmp_path, planner)


@pytest.mark.parametrize("check_width", [False, True], ids=["group-end-only", "element-width-checked"])
def test_single_page_group_end_does_not_bound_first_fill_width(check_width, tmp_path):
    # end=31 and origin=-1 for a full group of 32 positions pass the page
    # guard, but its first score width is zero. The two table values have
    # independent INT32 representations even when supplied by the same caller.
    predicate = "0 < width and width <= 32" if check_width else "True"
    program = pl.parse_program(f"""
@pl.program
class SinglePage:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], lengths: pl.Tensor[[2], pl.INT32],
               output: pl.InOut[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        end = pl.tensor.read(lengths, [0])
        group = pl.tensor.read(lengths, [1])
        origin = pl.cast(end, pl.INDEX) - pl.cast(group, pl.INDEX)
        if (origin + 32 + 31) // 32 == 1:
            for position in pl.range(32):
                width = origin + position + 1
                if {predicate}:
                    tile = pl.load(source, [0, 0], [16, 32])
                    prefix = pl.set_validshape(tile, 16, width)
                    padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
                    output = pl.store(padded, [0, 0], output)
        return output
""")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_scalar_gm_dma=True):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Buffer, str(dict(function.attrs))
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True, emit_source_loc=False)
    assert ("pto.texpands" in text) is not check_width
    _compile_dynamic_state_native(lowered, tmp_path, passes.MemoryPlanner.PYPTO)


@pytest.mark.parametrize("prefix_cols", ["32", "columns"])
def test_minimum_pad_runtime_rows_flow_through_native_payloads(prefix_cols, tmp_path):
    program = pl.parse_program(f"""
@pl.program
class PaddedRows:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 32], pl.FP32], rows: pl.Scalar[pl.INDEX],
               columns: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[16, 32], pl.BF16]]) -> pl.Tensor[[16, 32], pl.BF16]:
        if rows > 0 and rows <= 16 and columns > 0 and columns <= 32:
            src = pl.load(source, [0, 0], [16, 32])
            prefix = pl.set_validshape(src, rows, columns)
            padded = pl.fillpad(prefix, pad_value=pl.PadValue.min)
            active = pl.set_validshape(padded, rows, {prefix_cols})
            workspace = pl.tile.create([16, 32], dtype=pl.FP32)
            maximum = pl.row_max(active, workspace)
            shifted = pl.row_expand_sub(active, maximum)
            weights = pl.exp(shifted)
            copied = pl.tile.move(weights, target_memory=pl.Mem.Vec, slayout=pl.TileLayout.none_box)
            sums = pl.row_sum(copied, workspace)
            centered = pl.row_expand_sub(weights, sums)
            packed = pl.cast(centered, pl.BF16)
            output = pl.store(packed, [0, 0], output)
        return output
""")
    planner = passes.MemoryPlanner.PYPTO
    lowered = _lower(program, planner)
    actual = _kernel(lowered)
    assert (actual.ir_stage == ir.FunctionIRStage.Buffer) is (prefix_cols == "32"), str(actual)
    if prefix_cols == "32":
        text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
        assert all(
            f"pto.{op}" in text
            for op in ["tfillpad", "trowmax", "trowexpandsub", "texp", "tmov", "trowsum", "tcvt"]
        )
        _compile_dynamic_state_native(lowered, tmp_path, planner)


def test_contiguous_column_move_preserves_runtime_rows(tmp_path):
    program = pl.parse_program("""
@pl.program
class ColumnMove:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[1, 16], pl.FP32], rows: pl.Scalar[pl.INDEX],
               output: pl.InOut[pl.Tensor[[1, 16], pl.FP32]]) -> pl.Tensor[[1, 16], pl.FP32]:
        if rows > 0 and rows <= 16:
            src = pl.load(source, [0, 0], [1, 16])
            active = pl.set_validshape(src, 1, rows)
            column = pl.reshape(active, [16, 1])
            copied = pl.tile.move(column, target_memory=pl.Mem.Vec)
            copy_row = pl.reshape(copied, [1, 16])
            original_row = pl.reshape(column, [1, 16])
            changed = pl.mul(copy_row, 2.0)
            result = pl.add(original_row, changed)
            output = pl.store(result, [0, 0], output)
        return output
""")
    planner = passes.MemoryPlanner.PYPTO
    lowered = _lower(program, planner)
    actual = _kernel(lowered)
    assert actual.ir_stage == ir.FunctionIRStage.Buffer, str(actual)
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert "pto.tmov" in text
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    source = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    for rows in (3, 15):
        output = torch.full_like(source, -1)
        namespace["kernel"](source, rows, output)
        torch.testing.assert_close(output[:, :rows], 3 * source[:, :rows])
        assert torch.all(output[:, rows:] == -1)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("window", ["identity", "offset", "valid"])
def test_acc_slice_reuses_only_an_identical_buffer(planner, window, tmp_path):
    parent_rows = 32 if window == "offset" else 16
    offset = 16 if window == "offset" else 0
    validity = ", valid_shape=[8, 16]" if window == "valid" else ""
    program = pl.parse_program(f"""
@pl.program
class AccWindow:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, lhs: pl.Tensor[[{parent_rows}, 32], pl.FP16], rhs: pl.Tensor[[32, 16], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        x = pl.load(lhs, [0, 0], [{parent_rows}, 32], target_memory=pl.Mem.Mat)
        y = pl.load(rhs, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        product = pl.matmul(x, y)
        result = pl.slice(product, [16, 16], [{offset}, 0]{validity})
        return pl.store(result, [0, 0], out)
""")
    lowered = _lower(program, planner)
    actual = _kernel(lowered)
    if window != "identity":
        assert actual.ir_stage == ir.FunctionIRStage.Functional
        return
    assert actual.ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert "pto.subview" not in text
    product = next(line for line in text.splitlines() if "pto.tmatmul " in line)
    buffer = product.split("outs(", 1)[1].split(" :", 1)[0]
    assert f"pto.tstore ins({buffer} " in text
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    lhs = torch.arange(512, dtype=torch.float16).reshape(16, 32) / 512
    rhs = torch.ones((32, 16), dtype=torch.float16)
    out = torch.empty((16, 16))
    namespace["kernel"](lhs, rhs, out)
    torch.testing.assert_close(out, lhs.float() @ rhs.float())


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("column", [False, True], ids=["row-major", "column-major"])
def test_full_vec_slice_keeps_the_source_buffer(planner, column, tmp_path):
    cols = 1 if column else 16
    producer = (
        "scratch = pl.tile.create([16, 16], dtype=pl.FP32)\n        data = pl.row_sum(loaded, scratch)"
        if column
        else "data = pl.add(loaded, loaded)"
    )
    program = pl.parse_program(f"""
@pl.program
class VecWindow:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, 16], pl.FP32],
               out: pl.Out[pl.Tensor[[16, {cols}], pl.FP32]]) -> pl.Tensor[[16, {cols}], pl.FP32]:
        loaded = pl.load(source, [0, 0], [16, 16])
        {producer}
        full = pl.slice(data, [16, {cols}], [0, 0])
        return pl.store(full, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert "pto.subview" not in text
    op = "pto.trowsum" if column else "pto.tadd"
    producer_line = next(line for line in text.splitlines() if op in line)
    buffer = producer_line.split("outs(", 1)[1].split(" :", 1)[0]
    assert f"pto.tstore ins({buffer} " in text
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    source = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    out = torch.empty((16, cols))
    namespace["kernel"](source, out)
    torch.testing.assert_close(out, source.sum(dim=1, keepdim=True) if column else source * 2)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("strided_parent", [False, True], ids=["dense-window", "strided-window"])
def test_static_dense_window_reshape_in_kernel_with_runtime_validity(strided_parent, tmp_path):
    columns = 64 if strided_parent else 32
    program = pl.parse_program(f"""
@pl.program
class DenseWindow:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, source: pl.Tensor[[16, {columns}], pl.BF16],
               partial: pl.InOut[pl.Tensor[[16, {columns}], pl.BF16]],
               destination: pl.InOut[pl.Tensor[[1, 160], pl.BF16]]) -> pl.Tensor[[1, 160], pl.BF16]:
        loaded = pl.load(source, [0, 0], [16, {columns}])
        for rows in pl.range(5, 9):
            prefix = pl.set_validshape(loaded, rows, {columns})
            partial = pl.store(prefix, [0, 0], partial)
            window = pl.slice(loaded, [5, 32], [0, 0])
            flattened = pl.reshape(window, [1, 160])
            destination = pl.store(flattened, [0, 0], destination)
        return destination
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    if strided_parent:
        assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional
        return
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert "pto.subview" in text and "pto.treshape" in text and "pto.set_validshape" in text
    _compile_dynamic_state_native(lowered, tmp_path, passes.MemoryPlanner.PYPTO)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    source = torch.arange(512, dtype=torch.bfloat16).reshape(16, 32)
    partial = torch.zeros_like(source)
    destination = torch.empty((1, 160), dtype=torch.bfloat16)
    namespace["kernel"](source, partial, destination)
    torch.testing.assert_close(destination, source[:5].reshape(1, 160))


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dynamic", ["offset", "valid"])
def test_dynamic_window_forms_select_buffer(planner, dynamic):
    offset = "rows" if dynamic == "offset" else "0"
    valid = ", valid_shape=[rows, 32]" if dynamic == "valid" else ""
    program = pl.parse_program(f"""
@pl.program
class DynamicWindow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], rows: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [{offset}, 0]{valid})
        return pl.store(window, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Buffer


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dynamic", ["static", "offset", "valid", "both"])
def test_default_window_aliases_preserve_storage(tmp_path, planner, dynamic):
    offsets = "[row, col * 8]" if dynamic in ("offset", "both") else "[0, 0]"
    valid = ", valid_shape=[16, cols]" if dynamic in ("valid", "both") else ""
    program = pl.parse_program(f"""
@pl.program
class WindowAliases:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], row: pl.Scalar[pl.INDEX],
               col: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], {offsets}{valid})
        alias = window
        chained = alias
        value = pl.add(alias, chained)
        out = pl.store(whole, [0, 0], out)
        return pl.store(value, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    for row, col, cols in [(0, 0, 0), (1, 1, 1), (32, 2, 32), (8, 1, 0)]:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, row, col, cols, out)
        start_row, start_col = (row, col * 8) if dynamic in ("offset", "both") else (0, 0)
        width = cols if dynamic in ("valid", "both") else 32
        expected = x.clone()
        expected[:16, :width] = 2 * x[start_row : start_row + 16, start_col : start_col + width]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "window_aliases.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "window_aliases.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["nested", "assemble", "nested_assemble"])
def test_default_runtime_offsets_preserve_source_and_overlap(tmp_path, planner, kind):
    body = {
        "nested": """outer = pl.tile.slice(whole, [32, 48], [row, 0])
        window = pl.tile.slice(outer, [16, 32], [8, col * 8])
        value = pl.add(window, window)
        out = pl.store(whole, [0, 0], out)
        out = pl.store(value, [0, 0], out)""",
        "assemble": """window = pl.tile.slice(whole, [16, 64], [row, 0])
        value = pl.tile.assemble(whole, window, [col, 0])
        out = pl.store(value, [0, 0], out)""",
        "nested_assemble": """outer = pl.tile.slice(whole, [32, 64], [row, 0])
        window = pl.tile.slice(outer, [16, 64], [8, 0])
        value = pl.tile.assemble(whole, window, [col, 0])
        out = pl.store(value, [0, 0], out)""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class RuntimeOffsets:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], row: pl.Scalar[pl.INDEX],
               col: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        {body}
        return out
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    # Reuse one compiled body with changing offsets; a frozen first offset
    # or wrong nested pitch must change both the data and untouched sentinel.
    samples = [(0, 0), (1, 1), (16, 2), (32, 0)]
    for row, col in samples:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, row, col, out)
        if kind == "nested":
            expected = x.clone()
            expected[:16, :32] = 2 * x[row + 8 : row + 24, col * 8 : col * 8 + 32]
        else:
            expected = x.clone()
            start = row + (8 if kind == "nested_assemble" else 0)
            expected[col : col + 16] = x[start : start + 16].clone()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert _kernel(_lower(program, planner, False)).ir_stage == ir.FunctionIRStage.Functional
    if find_ptoas_binary() is not None:
        source = tmp_path / "runtime_offsets.pto"
        source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "runtime_offsets.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "kind",
    [
        "load",
        "setter",
        "mixed",
        "zero",
        "mixed_zero",
        "chain",
        "two_axes",
        "different_capacity",
        "static_setter_capacity",
    ],
)
def test_default_runtime_valid_empty_and_nonempty_paths(tmp_path, planner, kind):
    valid = (
        "[rows, cols]"
        if kind == "two_axes"
        else "[16, 0]"
        if kind in ("zero", "mixed_zero")
        else "[16, cols]"
    )
    body = f"loaded = pl.load(x, [0, 0], [16, 32], valid_shape={valid})"
    operation = "pl.add(loaded, loaded)"
    if kind == "setter":
        body = """whole = pl.load(x, [0, 0], [16, 32])
        loaded = pl.tile.set_validshape(whole, 16, cols)"""
    elif kind in ("mixed", "mixed_zero"):
        body = f"""whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [0, 0], valid_shape={valid})
        {body}"""
        operation = "pl.add(window, loaded)"
    elif kind == "chain":
        body = """whole = pl.load(x, [0, 0], [16, 32])
        narrowed = pl.tile.set_validshape(whole, 16, cols)
        out = pl.store(narrowed, [32, 0], out)
        loaded = pl.tile.set_validshape(narrowed, 16, 32)"""
    elif kind in ("different_capacity", "static_setter_capacity"):
        # The larger allocation dies before the runtime-valid load. PYPTO must
        # not turn the smaller dynamic allocation into an interior byte view.
        body = f"""whole = pl.load(x, [0, 0], [64, 64])
        out = pl.store(whole, [0, 0], out)
        {body}"""
        if kind == "static_setter_capacity":
            body = """whole = pl.load(x, [0, 0], [64, 64])
        out = pl.store(whole, [0, 0], out)
        full = pl.load(x, [0, 0], [16, 32])
        loaded = pl.tile.set_validshape(full, 16, 7)"""
    program = pl.parse_program(f"""
@pl.program
class RuntimeValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], rows: pl.Scalar[pl.INDEX],
               cols: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        {body}
        computed = {operation}
        return pl.store(computed, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    for rows, cols in [(0, 0), (0, 32), (16, 0), (1, 1), (16, 32), (7, 13), (0, 0)]:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, rows, cols, out)
        expected = (
            x.clone()
            if kind in ("different_capacity", "static_setter_capacity")
            else torch.full_like(x, -777)
        )
        vr = rows if kind == "two_axes" else 16
        vc = 32 if kind == "chain" else 0 if kind in ("zero", "mixed_zero") else cols
        if kind == "static_setter_capacity":
            vc = 7
        expected[:vr, :vc] = 2 * x[:vr, :vc]
        if kind == "chain":
            expected[32:48, :cols] = x[:16, :cols]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if find_ptoas_binary() is not None:
        source = tmp_path / "runtime_valid.pto"
        text = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "runtime_valid.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("alias", ["direct", "alias", "window"])
def test_runtime_valid_old_metadata_aliases(tmp_path, planner, alias):
    saved = {
        "direct": "",
        "alias": "old = loaded",
        "window": "old = pl.tile.slice(loaded, [16, 32], [0, 0])",
    }[alias]
    old = "loaded" if alias == "direct" else "old"
    program = pl.parse_program(f"""
@pl.program
class OldValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        loaded = pl.load(x, [0, 0], [16, 32])
        {saved}
        narrowed = pl.tile.set_validshape(loaded, 16, cols)
        small = pl.add(narrowed, narrowed)
        out = pl.store(small, [0, 0], out)
        full = pl.add({old}, {old})
        return pl.store(full, [16, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    for cols in (0, 7, 32, 1, 0):
        out = torch.full((32, 32), -777.0)
        namespace["kernel"](x, cols, out)
        expected = torch.full_like(out, -777)
        expected[:16, :cols] = 2 * x[:, :cols]
        expected[16:] = 2 * x
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["unaligned_offset", "nested_valid", "different_valid"])
def test_dynamic_unsupported_forms_remain_tile(planner, kind):
    body = {
        "unaligned_offset": "value = pl.tile.slice(whole, [16, 32], [0, cols])",
        "nested_valid": """window = pl.tile.slice(whole, [32, 64], [0, 0], valid_shape=[32, cols])
        value = pl.tile.slice(window, [16, 32], [0, 0], valid_shape=[16, cols])""",
        "different_valid": """left = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        right = pl.load(x, [0, 0], [16, 32], valid_shape=[16, other])
        value = pl.add(left, right)""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class UnsupportedDynamic:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               other: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        {body}
        return pl.store(value, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


def _compile_dynamic_state_native(lowered, tmp_path, planner):
    if find_ptoas_binary() is None:
        return
    source = tmp_path / "dynamic_state.pto"
    source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(tmp_path / "dynamic_state.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("consumer", ["slice", "reshape", "reinterpret", "assemble"])
def test_static_window_setter_dominates_view_consumers(tmp_path, planner, consumer):
    body = {
        "slice": "value = pl.tile.slice(narrowed, [4, 32], [2, 16])",
        "reshape": "value = pl.reshape(narrowed, [8, 128])",
        "reinterpret": "value = pl.tile.reinterpret_view(narrowed, pl.INT32)",
        "assemble": """target = pl.tile.full([16, 64], pl.FP32, -1.0)
        value = pl.tile.assemble(target, narrowed, [0, 0])""",
    }[consumer]
    dtype = "INT32" if consumer == "reinterpret" else "FP32"
    program = pl.parse_program(f"""
@pl.program
class StaticWindowSetter:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[32, 128], pl.{dtype}]]
               ) -> pl.Tensor[[32, 128], pl.{dtype}]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 64], [8, 0])
        narrowed = pl.tile.set_validshape(window, 8, 64)
        {body}
        return pl.store(value, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, lowered) == []
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    out = torch.full((32, 128), -777, dtype=torch.int32 if consumer == "reinterpret" else torch.float32)
    expected = out.clone()
    if consumer == "slice":
        expected[:4, :32] = x[10:14, 16:48]
    elif consumer == "reshape":
        expected[:4, :128] = x[8:16].reshape(4, 128)
    elif consumer == "reinterpret":
        expected[:8, :64] = x[8:16].view(torch.int32)
    else:
        expected[:16, :64] = -1
        expected[:8, :64] = x[8:16]
    namespace["kernel"](x, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["window_setter", "if", "for", "while"])
def test_default_dynamic_valid_state_regions(tmp_path, planner, kind):
    body = {
        "window_setter": """whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [8, 16])
        narrowed = pl.tile.set_validshape(window, 16, cols)
        out = pl.store(narrowed, [0, 0], out)
        restored = pl.tile.set_validshape(narrowed, 16, 32)
        out = pl.store(restored, [16, 0], out)
        value = pl.add(window, window)""",
        "if": """loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        if count > 0:
            value = pl.add(loaded, loaded)
        else:
            value = pl.sub(loaded, loaded)""",
        "for": """loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        for index in pl.range(count):
            loaded = pl.add(loaded, loaded)
        value = loaded""",
        "while": """loaded = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        index = 0
        while index < count:
            loaded = pl.add(loaded, loaded)
            index = index + 1
        value = loaded""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class DynamicState:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               count: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]
               ) -> pl.Tensor[[64, 64], pl.FP32]:
        {body}
        return pl.store(value, [32, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    assert _kernel(_lower(program, planner, False)).ir_stage == ir.FunctionIRStage.Functional
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    for cols, count in [(0, 0), (7, 0), (32, 1), (13, 2), (1, 3), (0, 3)]:
        out = torch.full_like(x, -777)
        namespace["kernel"](x, cols, count, out)
        expected = torch.full_like(x, -777)
        if kind == "window_setter":
            expected[:16, :cols] = x[8:24, 16 : 16 + cols]
            expected[16:32, :32] = x[8:24, 16:48]
            expected[32:48, :32] = 2 * x[8:24, 16:48]
        else:
            factor = (2 if count else 0) if kind == "if" else 2**count
            expected[32:48, :cols] = factor * x[:16, :cols]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["if", "for", "while", "nested"])
def test_dynamic_valid_scalar_state_and_old_alias(tmp_path, planner, kind):
    # Valid state is ordinary scalar SSA. Each use reconstructs its own window;
    # the final old alias must still read the full original allocation.
    body = {
        "if": """if count > 0:
            width = pl.yield_(cols)
        else:
            width = pl.yield_(32)
        value = pl.tile.set_validshape(window, 16, width)
        out = pl.store(value, [0, 0], out)""",
        "for": """for index, (width, other) in pl.range(count, init_values=(cols, 32)):
            value = pl.tile.set_validshape(window, 16, width)
            out = pl.store(value, [index * 16, 0], out)
            final_width, final_other = pl.yield_(other, width)
        value = pl.tile.set_validshape(window, 16, final_width)
        out = pl.store(value, [48, 0], out)""",
        "while": """for index, width, other in pl.while_(init_values=(0, cols, 32)):
            pl.cond(index < count)
            value = pl.tile.set_validshape(window, 16, width)
            out = pl.store(value, [index * 16, 0], out)
            final_index, final_width, final_other = pl.yield_(index + 1, other, width)
        value = pl.tile.set_validshape(window, 16, final_width)
        out = pl.store(value, [48, 0], out)""",
        "nested": """for index in pl.range(count):
            for inner in pl.range(2):
                if inner == 0:
                    width = pl.yield_(cols)
                else:
                    width = pl.yield_(32)
                value = pl.tile.set_validshape(window, 16, width)
                out = pl.store(value, [index * 16, inner * 32], out)""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class ScalarValidState:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               count: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[128, 64], pl.FP32]]
               ) -> pl.Tensor[[128, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [8, 16])
        {body}
        out = pl.store(window, [64, 0], out)
        return pl.store(whole, [64, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    ir.assert_structural_equal(lowered, ir.deserialize(ir.serialize(lowered)))
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    for count in range(4):
        for cols in (0, 7, 32, 0):
            out = torch.full((128, 64), -777.0)
            namespace["kernel"](x, cols, count, out)
            expected = torch.full_like(out, -777)
            window = x[8:24, 16:48]
            if kind == "if":
                width = cols if count else 32
                expected[:16, :width] = window[:, :width]
            else:
                for index in range(count):
                    width = cols if index % 2 == 0 or kind == "nested" else 32
                    expected[index * 16 : index * 16 + 16, :width] = window[:, :width]
                    if kind == "nested":
                        expected[index * 16 : index * 16 + 16, 32:] = window
                if kind != "nested":
                    width = cols if count % 2 == 0 else 32
                    expected[48:64, :width] = window[:, :width]
            expected[64:] = x
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _compile_dynamic_state_native(lowered, tmp_path, planner)


@pytest.mark.parametrize("kind", ["if", "for", "while"])
def test_changing_valid_tile_carry_requires_scalar_state(kind):
    body = {
        "if": """if count > 0:
            value = pl.tile.set_validshape(whole, 16, cols)
        else:
            value = whole""",
        "for": """value = whole
        for index in pl.range(count):
            value = pl.tile.set_validshape(value, 16, cols)""",
        "while": """value = whole
        index = 0
        while index < count:
            value = pl.tile.set_validshape(value, 16, cols)
            index = index + 1""",
    }[kind]
    with pytest.raises(ParserTypeError, match="Cannot reassign.*with a different type"):
        pl.parse_program(f"""
@pl.program
class ChangingTileValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], cols: pl.Scalar[pl.INDEX],
               count: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [16, 32])
        {body}
        return pl.store(value, [0, 0], out)
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "kind", ["dynamic_relabel", "relabelled_window_setter", "mat_assemble", "acc_assemble"]
)
def test_remaining_view_forms_keep_their_selection_boundary(planner, kind):
    # These forms need distinct storage/native recipes. C's metadata support
    # must not accidentally enable byte-root relabels or matrix insert paths.
    body = {
        "dynamic_relabel": """whole = pl.load(x, [0, 0], [16, 32], valid_shape=[16, cols])
        bits = pl.tile.reinterpret_view(whole, pl.INT32)
        value = pl.tile.reinterpret_view(bits, pl.FP32)""",
        "relabelled_window_setter": """whole = pl.load(x, [0, 0], [32, 32])
        window = pl.tile.slice(whole, [16, 32], [0, 0])
        reshaped = pl.reshape(window, [8, 64])
        value = pl.tile.set_validshape(reshaped, 8, 32)""",
        "mat_assemble": """target = pl.tile.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
        source = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat)
        value = pl.tile.assemble(target, source, [16, 0])""",
        "acc_assemble": """target = pl.tile.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
        left = pl.tile.move(target, target_memory=pl.MemorySpace.Left)
        right = pl.tile.move(target, target_memory=pl.MemorySpace.Right)
        source = pl.tile.matmul(left, right)
        value = pl.tile.assemble(target, source, [0, 0])""",
    }[kind]
    program = pl.parse_program(f"""
@pl.program
class RemainingViews:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        {body}
        return pl.store(value, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype,scale", [("FP16", 16), ("FP32", 8)])
@pytest.mark.parametrize("kind", ["slice_valid", "slice_drop", "assemble"])
@pytest.mark.parametrize("offset_kind", ["raw", "aligned", "misaligned", "constant"])
def test_window_selection_proves_byte_alignment(planner, dtype, scale, kind, offset_kind):
    offset = {
        "raw": "col",
        "aligned": "scaled + step",
        "misaligned": "scaled + 1",
        "constant": "1",
    }[offset_kind]
    if kind == "assemble":
        body = f"""window = pl.tile.slice(whole, [16, 32], [0, 0])
        value = pl.tile.assemble(whole, window, [row, {offset}])"""
    else:
        drop = ", drop_dims=[]" if kind == "slice_drop" else ""
        body = f"value = pl.tile.slice(whole, [16, 32], [row, {offset}], valid_shape=[16, cols]{drop})"
    program = pl.parse_program(f"""
@pl.program
class WindowAlignment:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.{dtype}], row: pl.Scalar[pl.INDEX],
               col: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[64, 64], pl.{dtype}]]) -> pl.Tensor[[64, 64], pl.{dtype}]:
        step = {scale}
        scaled = col * step
        whole = pl.load(x, [0, 0], [64, 64])
        {body}
        return pl.store(value, [0, 0], out)
""")
    expected = ir.FunctionIRStage.Buffer if offset_kind == "aligned" else ir.FunctionIRStage.Functional
    assert _kernel(_lower(program, planner)).ir_stage == expected


@pytest.mark.parametrize("rows,col,selected", [(1, 0, True), (1, 8, True), (1, 1, False), (2, 0, False)])
def test_assemble_narrow_row_requires_alignment_and_no_inter_row_stride(rows, col, selected):
    program = pl.parse_program(f"""
@pl.program
class NarrowRowAssemble:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[{rows}, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[8, 16], pl.FP32]]) -> pl.Tensor[[8, 16], pl.FP32]:
        source = pl.load(x, [0, 0], [{rows}, 8])
        canvas = pl.tile.full([8, 16], dtype=pl.FP32, value=-3.402823e38)
        value = pl.tile.assemble(canvas, source, [0, {col}])
        return pl.store(value, [0, 0], out)
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    if not selected:
        assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional
        assert _kernel(lowered).attrs["buffer_admission_property"] == "window_geometry"
        return
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.copy" in lowered.as_python()
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        logical = passes.insert_sync()(lowered)
    with passes.PassContext([]):
        physical = passes.lower_sync()(logical)
    text = codegen.PTOCodegen().generate(physical, emit_tile_addr=True, emit_source_loc=False)
    assert "pto.tmov ins(" in text


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("valid_cols", ["16", "cols"])
def test_standalone_valid_shape_setter_remains_tile(tmp_path, planner, valid_cols):
    program = pl.parse_program(f"""
@pl.program
class StandaloneSetter:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [16, 32])
        pl.tile.set_validshape(whole, 16, {valid_cols})
        return pl.store(whole, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional
    if find_ptoas_binary() is not None:
        source = tmp_path / "standalone_setter.pto"
        source.write_text(
            codegen.PTOCodegen().generate(
                lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
            )
        )
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(source), str(tmp_path / "standalone_setter.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


def _softmax_program(shape, valid, dtype="FP32", out_layout=None):
    rows, cols = shape
    valid_rows, valid_cols = valid
    scalar_param = "cols: pl.Scalar[pl.INDEX], " if isinstance(valid_cols, str) else ""
    layout_kwarg = f", out_layout=pl.TileLayout.{out_layout}" if out_layout else ""
    return pl.parse_program(f"""
@pl.program
class Softmax:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, {scalar_param}a: pl.Tensor[[{rows}, {cols}], pl.{dtype}],
               out: pl.InOut[pl.Tensor[[{2 * rows}, {cols}], pl.{dtype}]]
               ) -> pl.Tensor[[{2 * rows}, {cols}], pl.{dtype}]:
        source = pl.load(a, [0, 0], [{rows}, {cols}], valid_shape=[{valid_rows}, {valid_cols}])
        scratch = pl.create_tile([{rows}, {cols}], dtype=pl.{dtype}, target_memory=pl.MemorySpace.Vec)
        maximum = pl.row_max(source, scratch{layout_kwarg})
        shifted = pl.row_expand_sub(source, maximum)
        exponent = pl.exp(shifted)
        total = pl.row_sum(exponent, scratch{layout_kwarg})
        result = pl.row_expand_div(exponent, total)
        out = pl.store(result, [0, 0], out)
        return pl.store(source, [{rows}, 0], out)
""")


@pytest.mark.parametrize("planner", _PLANNERS)
def test_explicit_default_out_layout_still_selects_buffer_row_recipe(planner):
    """An explicit, behaviorally-default out_layout=col_major must not lose
    the fast buffer-row recipe path (issue #3319): SupportsLogicalBufferRow
    used to reject any call with non-empty kwargs, even one that only
    restates the implicit default, so spelling out_layout=col_major on
    row_max/row_sum silently fell back to the general Buffer IR path (or
    Tile IR) instead of the dedicated recipe this otherwise-identical kernel
    already uses -- compare test_default_fp32_softmax_preserves_live_input_and_partial_output.
    """
    program = _softmax_program((64, 64), (64, 64), out_layout="col_major")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    text = codegen.PTOCodegen().generate(
        lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    for operation in ("trowmax", "trowsum"):
        assert text.count(f"pto.{operation} ins(") == 1


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("shape,valid", [((64, 64), (64, 64)), ((16, 128), (13, 117)), ((16, 64), (1, 1))])
def test_default_fp32_softmax_preserves_live_input_and_partial_output(tmp_path, planner, shape, valid):
    rows, cols = shape
    valid_rows, valid_cols = valid
    program = _softmax_program(shape, valid)
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    # The input remains live across both reductions; scratch is reused only
    # between instructions. Partial output and the second store expose clobbers.
    source = torch.randn((rows, cols), generator=torch.Generator().manual_seed(19)) * 4
    output = torch.full((2 * rows, cols), -777.0)
    expected = output.clone()
    expected[:valid_rows, :valid_cols] = torch.softmax(source[:valid_rows, :valid_cols], dim=-1)
    expected[rows : rows + valid_rows, :valid_cols] = source[:valid_rows, :valid_cols]
    namespace = {}
    exec(torch_codegen(restored), namespace)  # noqa: S102
    namespace["kernel"](source, output)
    torch.testing.assert_close(output, expected)
    text = codegen.PTOCodegen().generate(
        restored, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    for operation in ("trowmax", "trowsum", "trowexpandsub", "trowexpanddiv"):
        assert text.count(f"pto.{operation} ins(") == 1
    if find_ptoas_binary() is not None:
        path = tmp_path / "softmax.pto"
        path.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(path), str(tmp_path / "softmax.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["off", "fp16", "dynamic"])
def test_softmax_selection_preserves_unopened_forms(planner, kind):
    program = _softmax_program(
        (16, 64), (16, "cols" if kind == "dynamic" else 64), "FP16" if kind == "fp16" else "FP32"
    )
    lowered = _lower(program, planner, False if kind == "off" else None)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
def test_softmax_default_a5_boundary_is_unchanged(ascend_backend, planner):
    assert ascend_backend == BackendType.Ascend950
    lowered = _lower(_softmax_program((16, 64), (16, 64)), planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_reduction_column_store_remains_tile(planner):
    # A contiguous column cannot use the dense matrix GM transfer recipe: its
    # destination has a row stride of 64 rather than contiguous scalar rows.
    program = pl.parse_program("""
@pl.program
class ColumnStore:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        source = pl.load(a, [0, 0], [16, 64])
        scratch = pl.create_tile([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        maximum = pl.row_max(source, scratch)
        return pl.store(maximum, [0, 0], out)
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


def _conversion_program(scalar, mode=4, valid=(8, 64), cast_mode="rint", dtype="FP16"):
    rows, cols = valid
    compare = f"pl.tile.cmps(x, 0.0, cmp_type={mode})" if scalar else f"pl.tile.cmp(x, y, cmp_type={mode})"
    scratch = (
        "pl.create_tile([1, 64], dtype=pl.FP32)" if scalar else "pl.create_tile([1, 16], dtype=pl.UINT32)"
    )
    select = "pl.tile.sels(mask, x, scratch, -3.0)" if scalar else "pl.tile.sel(mask, x, y, scratch)"
    source = f"""
@pl.program
class ConversionChain:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[8, 64], pl.FP16], b: pl.Tensor[[8, 64], pl.FP16],
               out: pl.InOut[pl.Tensor[[24, 64], pl.FP16]]) -> pl.Tensor[[24, 64], pl.FP16]:
        a_tile = pl.load(a, [0, 0], [8, 64], valid_shape=[{rows}, {cols}])
        b_tile = pl.load(b, [0, 0], [8, 64], valid_shape=[{rows}, {cols}])
        x = pl.cast(a_tile, pl.FP32, mode="{cast_mode}")
        y = pl.cast(b_tile, pl.FP32, mode="{cast_mode}")
        mask = {compare}
        scratch = {scratch}
        selected = {select}
        result = pl.cast(selected, pl.FP16, mode="{cast_mode}")
        out = pl.store(result, [0, 0], out)
        out = pl.store(a_tile, [8, 0], out)
        return pl.store(b_tile, [16, 0], out)
"""
    return pl.parse_program(source.replace("pl.FP16", f"pl.{dtype}"))


@pytest.mark.parametrize("cast_mode", ["rint", "round"])
@pytest.mark.parametrize("valid", [(8, 64), (7, 61)])
def test_bfloat_conversion_pipeline(tmp_path, cast_mode, valid):
    program = _conversion_program(False, valid=valid, cast_mode=cast_mode, dtype="BF16")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    source = codegen.PTOCodegen().generate(lowered)
    assert "round_mode " + cast_mode.upper() in source
    assert "bf16" in source
    path = tmp_path / "bfloat.pto"
    path.write_text(source)
    _run_ptoas(str(path), str(tmp_path / "bfloat.cpp"), ["--pto-arch=a2", "--pto-level=level3"])


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("mode", range(6))
@pytest.mark.parametrize("valid", [(8, 64), (7, 61)])
@pytest.mark.parametrize("cast_mode", ["none", "rint"])
def test_default_cast_compare_select_chain(tmp_path, planner, scalar, mode, valid, cast_mode):
    lowered = _lower(_conversion_program(scalar, mode, valid, cast_mode), planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    # Repeated comparisons include equal elements. Partial masks end mid-byte;
    # untouched output and live source stores expose packing and alias mistakes.
    a = ((torch.arange(512).reshape(8, 64) % 17) - 8).half() / 4
    b = torch.flip(a, [1])
    out = torch.full((24, 64), -777.0, dtype=torch.float16)
    expected = out.clone()
    rows, cols = valid
    lhs, rhs = a[:rows, :cols].float(), b[:rows, :cols].float()
    comparison = (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge)[mode]
    mask = comparison(lhs, 0.0 if scalar else rhs)
    expected[:rows, :cols] = torch.where(mask, lhs, -3.0 if scalar else rhs).half()
    expected[8 : 8 + rows, :cols] = a[:rows, :cols]
    expected[16 : 16 + rows, :cols] = b[:rows, :cols]
    namespace = {}
    exec(torch_codegen(restored), namespace)  # noqa: S102
    namespace["kernel"](a, b, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    text = codegen.PTOCodegen().generate(
        restored, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    assert "pto.tcvt" in text
    assert ("pto.tcmps" if scalar else "pto.tcmp") in text
    assert ("pto.tsels" if scalar else "pto.tsel") in text
    if find_ptoas_binary() is not None:
        path = tmp_path / "conversion_chain.pto"
        path.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(
            str(path), str(tmp_path / "conversion_chain.cpp"), ["--pto-arch=a2", f"--pto-level={level}"]
        )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["off", "rounding", "empty"])
def test_conversion_rollout_preserves_unopened_forms(planner, kind):
    program = _conversion_program(
        True,
        valid=(0, 64) if kind == "empty" else (8, 64),
        cast_mode="ceil" if kind == "rounding" else "rint",
    )
    lowered = _lower(program, planner, False if kind == "off" else None)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
def test_conversion_rollout_keeps_a5_on_tile(ascend_backend, planner):
    assert ascend_backend == BackendType.Ascend950
    assert _kernel(_lower(_conversion_program(True), planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_symbolic_leading_gm_extent_preserves_buffer_geometry(tmp_path, planner):
    program = pl.parse_program("""
ROWS = pl.dynamic("ROWS")
@pl.program
class LeadingExtent:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[ROWS, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[ROWS, 64], pl.FP32]]) -> pl.Tensor[[ROWS, 64], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32])
        out = pl.store(value, [0, 0], out)
        return out
""")
    lowered = _lower(program, planner, True)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    assert isinstance(function.params[0].type, ir.TensorType)
    shape = function.params[0].type.shape
    assert isinstance(shape[0], ir.Var)
    assert isinstance(shape[1], ir.ConstInt) and shape[1].value == 64
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner == passes.MemoryPlanner.PYPTO)
    assert "pto.tload" in text and "pto.tstore" in text
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    # Selecting a Buffer descriptor is not a proof of allocation bounds for
    # autosync. Its strict importer must preserve the missing-range refusal.
    with pytest.raises(ValueError, match="Dynamic allocation extent has no established caller range"):
        testing.analyze_placed_sync(function, True)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("slots", [2, 3])
def test_nested_accumulator_seed_with_operand_slot_pools(planner, slots, tmp_path):
    """Admission consumes the existing seed/carry proof, before legalization.

    Both operand levels use declared pools. The inner selector resets on every
    K chunk; the pool identity must survive that reset and Buffer conversion.
    """
    program = pl.parse_program(f"""
@pl.program
class NestedSlots:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 512], pl.FP16], b: pl.Tensor[[512, 64], pl.FP16],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        acc = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        for k in pl.range(4):
            am: pl.Tile[
                [64, 128], pl.FP16, pl.MemRef("am", slots={slots})[k % {slots}], pl.Mem.Mat
            ] = pl.load(a, [0, k * 128], [64, 128], target_memory=pl.Mem.Mat)
            bm: pl.Tile[
                [128, 64], pl.FP16, pl.MemRef("bm", slots={slots})[k % {slots}], pl.Mem.Mat
            ] = pl.load(b, [k * 128, 0], [128, 64], target_memory=pl.Mem.Mat)
            for j in pl.range(4):
                left: pl.Tile[
                    [64, 32], pl.FP16, pl.MemRef("left", slots={slots})[j % {slots}], pl.Mem.Left
                ] = pl.tile.extract(am, 0, j * 32, [64, 32], target_memory=pl.Mem.Left)
                right: pl.Tile[
                    [32, 64], pl.FP16, pl.MemRef("right", slots={slots})[j % {slots}], pl.Mem.Right
                ] = pl.tile.extract(bm, j * 32, 0, [32, 64], target_memory=pl.Mem.Right)
                if k == 0 and j == 0:
                    acc = pl.matmul(left, right)
                else:
                    acc = pl.matmul_acc(acc, left, right)
        return pl.store(acc, [0, 0], out)
""")
    lowered = _lower(program, planner)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.matmul" in str(lowered)
    native = codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert native.count("pto.alloc_multi_tile") == 4
    # Both L1 operands share one dominating outer selector; L0 operands and
    # both accumulator branches share the inner selector. Re-emitting the
    # outer selector inside the inner loop breaks enabled PTOAS synchronization
    # when its entry wait references that inner definition.
    assert native.count("arith.remui") == 2
    source = tmp_path / "nested.pto"
    source.write_text(native)
    _run_ptoas(
        str(source),
        str(tmp_path / "nested.cpp"),
        [
            "--pto-level=" + ("level2" if planner == passes.MemoryPlanner.PTOAS else "level3"),
            "--pto-arch=a2",
            "--enable-insert-sync=true",
        ],
    )


def test_slot_selector_cache_does_not_escape_sibling_regions(tmp_path):
    # Direct Buffer IR isolates codegen dominance from the deliberately narrower
    # branch admission policy. Each lexical region contains two equivalent slot
    # uses, without introducing payload or synchronization semantics.
    span = ir.Span.unknown()

    def integer(value):
        return ir.ConstInt(value, DataType.INDEX, span)

    def internal(name, args, result):
        return core_ir._create_internal_op_call(name, args, {}, result, span)

    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    multi = ir.MultiBufferType(element, 2)
    group = ir.Var("ring", multi, span)
    index = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    allocation = ir.AssignStmt(
        group, internal("buffer.alloc_multi", [ir.MakeTuple([], span), integer(0)], multi), span
    )

    def selections(prefix):
        return ir.SeqStmts(
            [
                ir.AssignStmt(
                    ir.Var(f"{prefix}_{number}", element, span),
                    internal(
                        "buffer.slot", [group, ir.FloorMod(index, integer(2), DataType.INDEX, span)], element
                    ),
                    span,
                )
                for number in range(2)
            ],
            span,
        )

    branch = ir.IfStmt(
        ir.Eq(index, integer(0), DataType.BOOL, span), selections("then"), selections("else"), [], span
    )
    loop = ir.ForStmt(
        index,
        integer(0),
        integer(4),
        integer(1),
        [],
        ir.SeqStmts([branch, selections("after")], span),
        [],
        span,
    )
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.SeqStmts([allocation, loop], span),
        span,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    program = ir.Program([function], "BranchSlots", span)
    native = codegen.PTOCodegen().generate(program, emit_tile_addr=True)
    # Each branch needs its own first definition; neither dominates the sibling
    # or the continuation. Within each body both selections share one remainder.
    assert native.count("arith.remui") == 3
    source = tmp_path / "branch_slots.pto"
    source.write_text(native)
    _run_ptoas(
        str(source),
        str(tmp_path / "branch_slots.cpp"),
        ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"],
    )


def _run_autosync(program, planner=passes.MemoryPlanner.PYPTO, policy="conservative_prefix"):
    with passes.PassContext(
        [],
        memory_planner=planner,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy=policy,
    ):
        synced = PassManager(OptimizationStrategy.Default).run_passes(program)
    fn = _kernel(synced)
    assert fn.attrs.get("buffer_ir_selected") is True
    assert fn.attrs.get("pypto_autosync") is True
    buf_props = passes.IRPropertySet()
    buf_props.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(buf_props, synced) == []
    return synced


@pytest.mark.parametrize(
    "dtype_name,torch_dtype,fill_val",
    [
        ("BF16", torch.bfloat16, 1.5),
        ("FP16", torch.float16, 1.5),
        ("INT32", torch.int32, 7),
    ],
)
def test_non_fp32_tile_full_selects_executes_and_owns_autosync(dtype_name, torch_dtype, fill_val):
    program = pl.parse_program(f"""
@pl.program
class NonFp32Fill:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self, out: pl.Out[pl.Tensor[[16, 32], pl.{dtype_name}]]
    ) -> pl.Tensor[[16, 32], pl.{dtype_name}]:
        filled = pl.tile.full([16, 32], pl.{dtype_name}, {fill_val})
        return pl.store(filled, [0, 0], out)
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.full" in str(lowered)
    assert "pto.texpands" in codegen.PTOCodegen().generate(lowered, emit_tile_addr=True)
    ns = {}
    exec(torch_codegen(lowered), ns)  # noqa: S102
    out = torch.zeros((16, 32), dtype=torch_dtype)
    ns["kernel"](out)
    torch.testing.assert_close(out, torch.full((16, 32), fill_val, dtype=torch_dtype))
    _run_autosync(program)


def test_four_arg_store_matching_source_shape_admits_and_sub_shape_refuses():
    good = pl.parse_program("""
@pl.program
class StoreShapes:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[16, 32], pl.BF16], out: pl.Out[pl.Tensor[[32, 64], pl.BF16]]
               ) -> pl.Tensor[[32, 64], pl.BF16]:
        x = pl.load(a, [0, 0], [16, 32])
        return pl.store(x, [0, 0], out, [16, 32])
""")
    lowered = _lower(good, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    _run_autosync(good)

    narrow = pl.parse_program("""
@pl.program
class NarrowStoreShapes:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[16, 32], pl.BF16], out: pl.Out[pl.Tensor[[32, 64], pl.BF16]]
               ) -> pl.Tensor[[32, 64], pl.BF16]:
        x = pl.load(a, [0, 0], [16, 32])
        return pl.store(x, [0, 0], out, [8, 32])
""")
    rejected = _lower(narrow, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(rejected).attrs["buffer_ir_selected"] is False


def test_load_cache_bypass_single_row_stride_and_max_loop_step_own_autosync():
    program = pl.parse_program("""
@pl.program
class BypassAndMaxStep:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, n: pl.Scalar[pl.INDEX],
               a: pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)],
               out: pl.Out[pl.Tensor[[1, 64], pl.FP32]]
               ) -> pl.Tensor[[1, 64], pl.FP32]:
        acc = pl.tile.full([1, 64], pl.FP32, 0.0)
        final_acc = acc
        for i, (cur,) in pl.range(0, 1, pl.max(n, 1), init_values=(acc,)):
            loaded = pl.load(a, [0, 0], [1, 64], cache=pl.CachePolicy.BYPASS)
            nxt = pl.add(cur, loaded)
            final_acc = pl.yield_(nxt)
        return pl.store(final_acc, [0, 0], out)
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    mlir_text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=True)
    assert "%__pypto_l2_cache_offset: i64" in mlir_text
    assert "cache_policy = #pto.load_cache_policy<l2_bypass>" in mlir_text
    _run_autosync(program)


def test_vec_to_vec_tile_extract_selects_executes_and_validates_bounds():
    program = pl.parse_program("""
@pl.program
class VecExtract:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        src = pl.load(a, [0, 0], [32, 64])
        sub = pl.tile.extract(src, 8, 16, [16, 32], target_memory=pl.MemorySpace.Vec)
        return pl.store(sub, [0, 0], out)
""")
    lowered = _lower(program, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.extract" in str(lowered)
    assert "pto.textract ins(" in codegen.PTOCodegen().generate(lowered, emit_tile_addr=True)
    ns = {}
    exec(torch_codegen(lowered), ns)  # noqa: S102
    a = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    out = torch.empty((16, 32), dtype=torch.float32)
    ns["kernel"](a, out)
    torch.testing.assert_close(out, a[8:24, 16:48])
    _run_autosync(program)

    # Misaligned column offset (4 elements = 16 bytes < 32 bytes) must refuse Buffer IR.
    bad_prog = pl.parse_program("""
@pl.program
class BadVecExtract:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[32, 64], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
               ) -> pl.Tensor[[16, 32], pl.FP32]:
        src = pl.load(a, [0, 0], [32, 64])
        sub = pl.tile.extract(src, 0, 4, [16, 32], target_memory=pl.MemorySpace.Vec)
        return pl.store(sub, [0, 0], out)
""")
    rejected = _lower(bad_prog, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(rejected).attrs["buffer_ir_selected"] is False


def test_int32_to_fp32_dequant_cast_and_single_row_slice_reshape_own_autosync():
    dequant = pl.parse_program("""
@pl.program
class Int32Dequant:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 64], pl.INT32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
               ) -> pl.Tensor[[8, 64], pl.FP32]:
        loaded = pl.load(x, [0, 0], [8, 64])
        fp = pl.tile.cast(loaded, target_type=pl.FP32, mode=0)
        return pl.store(fp, [0, 0], out)
""")
    lowered_dq = _lower(dequant, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(lowered_dq).ir_stage == ir.FunctionIRStage.Buffer
    assert "pto.tcvt" in codegen.PTOCodegen().generate(lowered_dq, emit_tile_addr=True)
    ns = {}
    exec(torch_codegen(lowered_dq), ns)  # noqa: S102
    x = torch.arange(-256, 256, dtype=torch.int32).reshape(8, 64)
    out = torch.empty((8, 64), dtype=torch.float32)
    ns["kernel"](x, out)
    torch.testing.assert_close(out, x.float())
    _run_autosync(dequant)

    slice_reshape = pl.parse_program("""
@pl.program
class SingleRowSliceReshape:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, w: pl.Tensor[[8, 8], pl.FP32], x: pl.Tensor[[8, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]) -> pl.Tensor[[8, 64], pl.FP32]:
        weights = pl.load(w, [0, 0], [8, 8])
        row0 = pl.tile.slice(weights, [1, 8], [1, 0])
        col = pl.tile.reshape(row0, [8, 1])
        xs = pl.load(x, [0, 0], [8, 64])
        scaled = pl.tile.row_expand_mul(xs, col)
        return pl.store(scaled, [0, 0], out)
""")
    lowered_sr = _lower(slice_reshape, passes.MemoryPlanner.PYPTO, enabled=True)
    assert _kernel(lowered_sr).ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.subview" in str(lowered_sr) and "buffer.reshape" in str(lowered_sr)
    mlir_sr = codegen.PTOCodegen().generate(lowered_sr, emit_tile_addr=True)
    assert "pto.subview" in mlir_sr and "pto.treshape" in mlir_sr
    ns_sr = {}
    exec(torch_codegen(lowered_sr), ns_sr)  # noqa: S102
    w = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    xd = torch.ones((8, 64), dtype=torch.float32) * 2.0
    out_sr = torch.empty((8, 64), dtype=torch.float32)
    ns_sr["kernel"](w, xd, out_sr)
    torch.testing.assert_close(out_sr, xd * w[1:2, :].reshape(8, 1))
    _run_autosync(slice_reshape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
