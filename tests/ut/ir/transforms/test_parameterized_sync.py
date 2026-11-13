# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller-proved coordinates and unchanged PyPTO-lib frontend regressions."""

import json

import pypto.language as pl
import pytest
from pypto import ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .sync_prefill_cases import capture_prefill, load_prefill, prefill_arguments

pytestmark = pytest.mark.usefixtures("ascend_backend")


# Unchanged PyPTO-lib source at da5350d2; fixture hashes are pinned in the
# prefill qualification receipt. These are production frontend/placement tests,
# not manually manufactured Buffer inputs.
@pytest.mark.parametrize("name", ["matmul", "gemm"])
def test_pypto_lib_invocation_coordinates(name):
    program = capture_prefill(name)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    bounds = testing.sync_parameter_bounds(program)[original.name]
    assert bounds == {1: (0, 192), 3: (0, 192)}
    with pytest.raises(ValueError, match="established caller range"):
        testing.analyze_placed_sync(original, True)
    retained = testing.analyze_placed_sync(original, True, parameter_bounds=bounds)
    description = retained.describe()
    translated = 0
    for region in description["regions"]:
        for site in region["sites"]:
            for access, interval in zip(site["accesses"], site["physical_intervals"], strict=True):
                storage = description["storage"][access[0]]
                if storage["row_parameters"] or storage["column_parameters"]:
                    assert interval is None  # A relative offset is not a fixed physical interval.
                    translated += 1
    assert translated
    physical = retained.realize_physical("a2a3", list(range(6)))
    baseline_orders = {}
    for row, col in [(0, 0), (64, 128), (192, 192)]:
        parameters = {1: row, 3: col}
        validate(original, retained, parameters=parameters)
        baseline_orders[row, col] = run_physical(
            original, physical["function"], range(6), parameters=parameters
        )[2]
    assert retained.boundary(0, 0, 0) is None
    assert retained.boundary(0, 0, 0, {1: 256}) is None
    absent = retained.boundary(0, 0, 0, {1: 64})
    assert absent is not None and absent["first_writer"] is None
    assert not any(absent["first_readers"])
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        synchronized = passes.realize_sync()(program)
    selected = next(f for f in synchronized.functions.values() if f.name == original.name)
    assert dict(selected.attrs)["pypto_autosync"]
    # The actual-token emitter may flatten statement containers differently.
    # Compare complete payload prerequisite relations, not container identity.
    for (row, col), baseline in baseline_orders.items():
        assert run_physical(original, selected, range(6), parameters={1: row, 3: col})[2] == baseline
    with pytest.raises(ValueError, match="exceeds its proved allocation"):
        testing.analyze_placed_sync(original, True, parameter_bounds={1: (0, 193), 3: (0, 192)})


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("submit", [False, True])
def test_caller_range_joins_every_call(unknown, submit):
    # One unknown site must poison an otherwise valid range.
    source = """
@pl.program
class Callers:
    @pl.function(type=pl.FunctionType.AIC)
    def child(
        self, a: pl.Tensor[[256,256], pl.FP32], offset: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[256,256], pl.FP32]:
        return a
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, a: pl.Tensor[[256,256], pl.FP32], unknown: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[256,256], pl.FP32]:
        for i in pl.range(0, 192, 64):
            a = self.child(a, i)
        a = self.child(a, LAST)
        return a
""".replace("LAST", "unknown" if unknown else "192")
    if submit:
        source = source.replace("a = self.child(a, i)", "a, tid = pl.submit(self.child, a, i)")
        source = source.replace("a = self.child(a, ", "a, tid = pl.submit(self.child, a, ")
    program = pl.parse_program(source)
    facts = testing.sync_parameter_bounds(program)
    assert facts.get("child", {}) == ({} if unknown else {1: (0, 192)})


@pytest.mark.parametrize("expression", ["x * 9223372036854775807 * 0", "x - 1"])
def test_unproved_caller_arithmetic_does_not_supply_ranges(expression):
    program = pl.parse_program(
        """
@pl.program
class Callers:
    @pl.function(type=pl.FunctionType.AIC)
    def child(
        self, a: pl.Tensor[[256,256], pl.FP32], offset: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[256,256], pl.FP32]:
        return a
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[256,256], pl.FP32]) -> pl.Tensor[[256,256], pl.FP32]:
        for x in pl.range(3):
            a = self.child(a, EXPR)
        return a
""".replace("EXPR", expression)
    )
    assert testing.sync_parameter_bounds(program).get("child", {}) == {}


def test_parameterized_gemm_scaling_keeps_retained_size():
    signatures = []
    for count in [4, 8, 64, 1000000]:
        program = capture_prefill("gemm", count)
        function = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
        bounds = testing.sync_parameter_bounds(program)[function.name]
        retained = testing.analyze_placed_sync(function, True, parameter_bounds=bounds)
        d = retained.describe()
        physical = retained.realize_physical("a2a3", list(range(6)))
        signatures.append(
            (
                d["analyzed_occurrences"],
                d["cells"],
                [(len(r["sites"]), len(r["demands"]), r["distance_entries"]) for r in d["regions"]],
                physical["set_sites"],
                physical["wait_sites"],
                physical["reserved_lanes"],
            )
        )
        assert any(r["count"] == count for r in d["regions"])
    assert all(s == signatures[0] for s in signatures)


@pytest.mark.parametrize("name", ["matmul", "gemm"])
def test_pypto_lib_parameterized_native_compilation(name, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is required for native source generation")
    module = load_prefill(name)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
    ):
        getattr(module, name).compile(
            *prefill_arguments(module),
            config=RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path)),
        )
    report = json.loads((tmp_path / "report" / "autosync.json").read_text())
    assert report["kernels"] and all(k["selected_route"] == "pypto" for k in report["kernels"])
    assert list(tmp_path.rglob("*.cpp"))


@pytest.mark.parametrize("count", [1, 2, 3])
def test_short_parameterized_gemm(count):
    program = capture_prefill("gemm", count)
    function = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    bounds = testing.sync_parameter_bounds(program)[function.name]
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=bounds)
    physical = retained.realize_physical("a2a3", list(range(6)))
    for parameters in [{1: 0, 3: 192}, {1: 192, 3: 0}]:
        validate(function, retained, parameters=parameters)
        run_physical(function, physical["function"], range(6), parameters=parameters)


def test_refuse_inconsistent_domain_translation():
    """Two individually in-range accesses do not prove parameter-independent aliasing."""
    function = staged_fixture(2)
    span = function.span
    parameter = ir.Var("offset", ir.ScalarType(ir.DataType.INDEX), span)

    class TranslateFirst(ir.IRMutator):
        changed = False

        def visit_call(self, op):
            call = op
            if call.op.name != ir.get_op("buffer.load").name or self.changed:
                return call
            self.changed = True
            args = list(call.args)
            offsets = args[1]
            assert isinstance(offsets, ir.MakeTuple)
            args[1] = ir.MakeTuple(
                [offsets.elements[0], ir.Add(offsets.elements[1], parameter, ir.DataType.INDEX, span)], span
            )
            return ir.Call(call.op, args, call.type, span)

    body = TranslateFirst().visit_stmt(function.body)
    function = ir.Function(
        function.name,
        [*function.params, parameter],
        function.return_types,
        body,
        span,
        function.func_type,
        ir_stage=function.ir_stage,
    )
    with pytest.raises(ValueError, match="common invocation translation"):
        testing.analyze_placed_sync(function, True, parameter_bounds={3: (0, 1)})


@pytest.mark.parametrize("bounds", [{0: (0, 1)}, {50: (0, 1)}, {1: (-1, 192)}, {1: (4, 3)}])
def test_parameter_bounds_preserve_original_positions(bounds):
    program = capture_prefill("matmul")
    function = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    with pytest.raises(ValueError, match="original scalar parameter|Invalid invocation coordinate range"):
        testing.analyze_placed_sync(function, True, parameter_bounds=bounds)


@pytest.mark.parametrize("depth", [12, 24, 30])
def test_caller_range_inspects_shared_expression_dag_once(depth):
    """A shared arithmetic DAG is not an exponentially expanded syntax tree."""
    program = pl.parse_program(
        """
@pl.program
class Callers:
    @pl.function(type=pl.FunctionType.AIC)
    def child(self, offset: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        return offset
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self) -> pl.Scalar[pl.INDEX]:
        result = self.child(1)
        return result
"""
    )

    class SharedArgument(ir.IRMutator):
        def visit_call(self, op):
            call = op
            if call.op.name != "child":
                return call
            expression = ir.ConstInt(1, ir.DataType.INDEX, call.span)
            for _ in range(depth):
                expression = ir.Add(expression, expression, ir.DataType.INDEX, call.span)
            return ir.Call(call.op, [expression], call.type, call.span)

    program = SharedArgument().visit_program(program)
    profile = testing.sync_parameter_bounds_profile(program)
    assert profile["bounds"] == {"child": {0: (2**depth, 2**depth)}}
    # Count inspected work, not elapsed time or just the single retained bound.
    assert profile["expression_visits"] <= 8 * depth + 32
    assert profile["range_visits"] <= 8 * depth + 32


@pytest.mark.parametrize("unknown", [False, True])
def test_caller_range_cache_respects_loop_scopes_and_unknown_sites(unknown):
    program = pl.parse_program(
        """
@pl.program
class Callers:
    @pl.function(type=pl.FunctionType.AIC)
    def child(self, offset: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        return offset
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, unknown: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        for i in pl.range(0, 4):
            first = self.child(i + 1)
        for i in pl.range(20, 24):
            second = self.child(i + 1)
        last = self.child(LAST)
        return last
""".replace("LAST", "unknown + 1" if unknown else "41")
    )
    profile = testing.sync_parameter_bounds_profile(program)
    assert profile["bounds"].get("child", {}) == ({} if unknown else {0: (1, 41)})


@pytest.mark.parametrize("parameter_count", [8, 32, 128])
def test_coordinate_prefix_definitions_keep_linear_retained_work(parameter_count):
    """One DMA uses a sum built by SSA prefixes, without retaining every prefix map."""
    span = ir.Span.unknown()
    index, fp = ir.DataType.INDEX, ir.DataType.FP32

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(value) if isinstance(value, int) else value for value in values], span)

    source = ir.Var("source", ir.TensorType([16, parameter_count + 32], fp), span)
    coordinates = [ir.Var(f"coordinate_{i}", ir.ScalarType(index), span) for i in range(parameter_count)]
    buffer_type = ir.BufferType(
        [16, 32],
        fp,
        ir.MemorySpace.Mat,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=512,
    )
    destination = ir.Var("destination", buffer_type, span)
    statements: list[ir.Stmt] = [
        ir.AssignStmt(
            destination,
            ir.Call(ir.get_op("buffer.alloc"), [tup([]), ci(0)], buffer_type, span),
            span,
        )
    ]
    offset = coordinates[0]
    for i, coordinate in enumerate(coordinates[1:], 1):
        next_offset = ir.Var(f"prefix_{i}", ir.ScalarType(index), span)
        statements.append(ir.AssignStmt(next_offset, ir.Add(offset, coordinate, index, span), span))
        offset = next_offset
    statements.extend(
        [
            ir.EvalStmt(
                ir.Call(
                    ir.get_op("buffer.load"),
                    [source, tup([0, offset]), tup([16, 32]), destination],
                    ir.VoidType(),
                    span,
                ),
                span,
            ),
            ir.ReturnStmt([], span),
        ]
    )
    function = ir.Function(
        "parameter_prefix_dma",
        [source, *coordinates],
        [],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.AIC,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    bounds = {i: (0, 1) for i in range(1, parameter_count + 1)}
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=bounds)
    description = retained.describe()
    assert description["coordinate_nodes"] <= 4 * parameter_count + 32
    assert description["coordinate_flatten_visits"] <= 8 * parameter_count + 32
    assert description["coordinate_coefficient_entries"] <= 2 * parameter_count
    for value in [0, 1]:
        validate(function, retained, parameters={i: value for i in bounds})
