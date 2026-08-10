# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Whole-function cube AutoTile admission, planning, and emission tests."""

from __future__ import annotations

import re
from collections import Counter

import pypto.language as pl
import pytest
from pypto import LogLevel, backend, ir, passes, set_log_level
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture(autouse=True)
def _ascend_910b_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_auto_tile(program: ir.Program) -> ir.Program:
    program = passes.convert_to_ssa()(program)
    program = passes.simplify()(program)
    program = passes.normalize_stmt_structure()(program)
    program = passes.flatten_call_expr()(program)
    return passes.auto_tile()(program)


def _run_default(program: ir.Program) -> ir.Program:
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


class _Structure(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.ops: Counter[str] = Counter()
        self.spmd = 0
        self.work_units: list[int] = []
        self.pipeline_loops = 0
        self.pipeline_stages: list[int] = []
        self.atomic_stores = 0

    def visit_call(self, op: ir.Call) -> None:
        self.ops[op.op.name] += 1
        if op.op.name == "tensor.assemble" and op.kwargs.get("atomic", 0) != 0:
            self.atomic_stores += 1
        super().visit_call(op)

    def visit_spmd_scope_stmt(self, op: ir.SpmdScopeStmt) -> None:
        self.spmd += 1
        assert isinstance(op.core_num, ir.ConstInt)
        self.work_units.append(int(op.core_num.value))
        super().visit_spmd_scope_stmt(op)

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        if op.kind == ir.ForKind.Pipeline:
            self.pipeline_loops += 1
            self.pipeline_stages.append(int(op.attrs["pipeline_stages"]))
        super().visit_for_stmt(op)


def _structure(program: ir.Program) -> _Structure:
    result = _Structure()
    result.visit_program(program)
    return result


def _tile_memref_bases(program: ir.Program, memory_space: ir.MemorySpace) -> set[str]:
    """Collect physical buffer identities assigned in one tile memory space."""
    bases: set[str] = set()

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt: ir.AssignStmt) -> None:
            tile = stmt.var.type
            if (
                isinstance(tile, ir.TileType)
                and tile.memory_space == memory_space
                and tile.memref is not None
            ):
                bases.add(tile.memref.base_.name_hint)
            super().visit_assign_stmt(stmt)

    _Collector().visit_program(program)
    return bases


def _tile_memory_high_water(program: ir.Program, memory_space: ir.MemorySpace) -> int:
    """Return the highest addressed byte of one allocated tile space."""
    high_water = 0

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt: ir.AssignStmt) -> None:
            nonlocal high_water
            tile = stmt.var.type
            if (
                isinstance(tile, ir.TileType)
                and tile.memory_space == memory_space
                and tile.memref is not None
                and isinstance(tile.memref.byte_offset_, ir.ConstInt)
            ):
                high_water = max(high_water, int(tile.memref.byte_offset_.value) + tile.memref.size_)
            super().visit_assign_stmt(stmt)

    _Collector().visit_program(program)
    return high_water


@pl.program
class MatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[64, 64], pl.FP32],
        rhs: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(lhs, rhs)
        return out


@pl.program
class RaggedMatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[130, 64], pl.FP32],
        rhs: pl.Tensor[[64, 260], pl.FP32],
    ) -> pl.Tensor[[130, 260], pl.FP32]:
        out: pl.Tensor[[130, 260], pl.FP32] = pl.matmul(lhs, rhs)
        return out


@pl.program
class Fp16MatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[64, 64], pl.FP16],
        rhs: pl.Tensor[[64, 64], pl.FP16],
    ) -> pl.Tensor[[64, 64], pl.FP16]:
        return pl.matmul(lhs, rhs, out_dtype=pl.FP16)


@pl.program
class Bf16MatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[64, 64], pl.BF16],
        rhs: pl.Tensor[[64, 64], pl.BF16],
    ) -> pl.Tensor[[64, 64], pl.BF16]:
        return pl.matmul(lhs, rhs, out_dtype=pl.BF16)


@pl.program
class RaggedKMatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[128, 736], pl.FP32],
        rhs: pl.Tensor[[736, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return pl.matmul(lhs, rhs)


@pl.program
class SplitKMatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[16, 16384], pl.FP32],
        rhs: pl.Tensor[[16384, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.matmul(lhs, rhs)


@pl.program
class RetainedPanelMatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def matmul(
        self,
        lhs: pl.Tensor[[1024, 256], pl.FP32],
        rhs: pl.Tensor[[256, 2048], pl.FP32],
    ) -> pl.Tensor[[1024, 2048], pl.FP32]:
        return pl.matmul(lhs, rhs)


@pl.program
class ChainedMatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def chain(
        self,
        lhs: pl.Tensor[[128, 256], pl.BF16],
        middle: pl.Tensor[[256, 128], pl.BF16],
        rhs: pl.Tensor[[128, 256], pl.BF16],
    ) -> pl.Tensor[[128, 256], pl.BF16]:
        intermediate: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(lhs, middle)
        return pl.matmul(intermediate, rhs)


@pl.program
class ProducedTreeMatmulProgram:
    @pl.function(attrs={"auto_tile": True})
    def tree(
        self,
        a: pl.Tensor[[32, 48], pl.BF16],
        b: pl.Tensor[[48, 80], pl.BF16],
        c: pl.Tensor[[80, 64], pl.BF16],
        d: pl.Tensor[[64, 96], pl.BF16],
    ) -> pl.Tensor[[32, 96], pl.FP32]:
        lhs: pl.Tensor[[32, 80], pl.BF16] = pl.matmul(a, b)
        rhs: pl.Tensor[[80, 96], pl.BF16] = pl.matmul(c, d)
        return pl.matmul(lhs, rhs, out_dtype=pl.FP32)


@pl.program
class RepeatedRhsChainProgram:
    @pl.function(attrs={"auto_tile": True})
    def chain(
        self,
        lhs: pl.Tensor[[512, 16], pl.BF16],
        shared_rhs: pl.Tensor[[16, 16], pl.BF16],
    ) -> pl.Tensor[[512, 16], pl.BF16]:
        intermediate: pl.Tensor[[512, 16], pl.BF16] = pl.matmul(lhs, shared_rhs)
        return pl.matmul(intermediate, shared_rhs)


@pl.program
class MultiRoleBoundaryProgram:
    @pl.function(attrs={"auto_tile": True})
    def tree(
        self,
        shared: pl.Tensor[[32, 48], pl.BF16],
        lhs_rhs: pl.Tensor[[48, 64], pl.BF16],
        rhs_lhs: pl.Tensor[[64, 32], pl.BF16],
    ) -> pl.Tensor[[32, 48], pl.BF16]:
        lhs: pl.Tensor[[32, 64], pl.BF16] = pl.matmul(shared, lhs_rhs)
        rhs: pl.Tensor[[64, 48], pl.BF16] = pl.matmul(rhs_lhs, shared)
        return pl.matmul(lhs, rhs)


@pl.program
class ProducedMultiRoleProgram:
    @pl.function(attrs={"auto_tile": True})
    def chain(
        self,
        lhs: pl.Tensor[[16, 16], pl.BF16],
        rhs: pl.Tensor[[16, 16], pl.BF16],
    ) -> pl.Tensor[[16, 16], pl.BF16]:
        shared: pl.Tensor[[16, 16], pl.BF16] = pl.matmul(lhs, rhs)
        return pl.matmul(shared, shared)


def test_single_matmul_emits_one_cube_spmd_kernel():
    after = _run_auto_tile(MatmulProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.work_units[0] >= 1
    # One pair initializes the first K window; the second pair is the rolled
    # window body that later pipeline lowering replicates dynamically.
    assert structure.ops["tensor.slice"] == 4
    assert structure.ops["tensor.matmul"] == 1
    assert structure.ops["tensor.assemble"] == 1
    function = after.get_function("matmul")
    assert function is not None
    assert "auto_tile" not in function.attrs
    body = function.as_python()
    assert "pl.spmd" in body
    assert "pl.tensor.matmul" in body


def test_matmul_plan_log_is_deterministic(capfd: pytest.CaptureFixture[str]):
    lines: list[str] = []
    set_log_level(LogLevel.INFO)
    try:
        for _ in range(2):
            capfd.readouterr()
            _run_auto_tile(MatmulProgram)
            selected = [line for line in capfd.readouterr().err.splitlines() if "AutoTile[matmul]" in line]
            assert len(selected) == 1
            lines.append(selected[0].split(" | ", maxsplit=1)[1])
    finally:
        set_log_level(LogLevel.INFO)
    assert lines[0] == lines[1]
    assert "cube schedule=k_window_pipeline" in lines[0]
    assert "spatial_policy=" in lines[0]
    assert "gm_to_l1_bytes=" in lines[0]
    assert "chunk=" in lines[0]
    assert "stages=2" in lines[0]
    assert "l0_tile=" in lines[0]


def test_outer_k_window_pipeline_carries_one_accumulator_and_peels_tail():
    after = _run_auto_tile(RaggedKMatmulProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.ops["tensor.matmul"] == 1
    assert structure.ops["tensor.matmul_acc"] == 2
    assert structure.ops["tensor.create"] == 0
    assert structure.ops["tensor.assemble"] == 1
    body = after.get_function("matmul").as_python()
    assert body.count("pl.pipeline") == 1
    assert "matmul_tile_body_lhs_tail" in body
    assert body.index("matmul_tile_body_lhs_tail") < body.index("pl.tensor.assemble")


def test_outer_k_window_pipeline_uses_one_physical_accumulator():
    """The full lowering must not split one K reduction over two L0C buffers."""
    after = _run_default(RaggedKMatmulProgram)
    acc_bases = _tile_memref_bases(after, ir.MemorySpace.Acc)
    assert len(acc_bases) == 1, (
        "outer K-window init, rolled windows, tail, and drain must share one "
        f"physical accumulator, got {sorted(acc_bases)}\n{ir.python_print(after)}"
    )


def test_split_k_uses_first_partial_then_atomic_without_zero_seed(
    capfd: pytest.CaptureFixture[str],
):
    set_log_level(LogLevel.INFO)
    try:
        after = _run_auto_tile(SplitKMatmulProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[matmul]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    structure = _structure(after)
    assert structure.spmd == 2
    assert structure.work_units[0] == 1
    assert structure.work_units[1] > 0
    assert sum(structure.work_units) > 1
    assert structure.ops["tensor.assemble"] == 2
    assert structure.atomic_stores == 1
    assert structure.ops["tensor.full"] == 0
    body = after.get_function("matmul").as_python()
    assert body.index("matmul_first_partial_spmd") < body.index("matmul_atomic_rest_spmd")
    assert "atomic=pl.AtomicType.Add" in body
    assert "split_merge=first_partial_then_atomic" in line
    assert "split_sync_cycles=0" in line


def test_split_k_complete_pipeline_reuses_one_gm_output_across_both_aic_phases():
    lowered = _run_default(SplitKMatmulProgram)
    aic = [function for function in lowered.functions.values() if function.func_type == ir.FunctionType.AIC]
    assert len(aic) == 2
    text = lowered.as_python()
    assert "matmul_first_partial" in text
    assert "matmul_atomic_rest" in text
    assert text.count("atomic=pl.AtomicType.Add") == 1
    orchestration = " ".join(lowered.get_function("matmul").as_python().split())
    assert "matmul_first_partial_spmd( t__tmp_v0_out, lhs__ssa_v0, rhs__ssa_v0" in orchestration
    assert "matmul_atomic_rest_spmd( matmul_first_partial_out, lhs__ssa_v0, rhs__ssa_v0" in orchestration


def test_retained_panel_is_loaded_once_outside_serial_output_tile_loop(
    capfd: pytest.CaptureFixture[str],
):
    set_log_level(LogLevel.INFO)
    try:
        after = _run_auto_tile(RetainedPanelMatmulProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[matmul]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    assert "retained_panels=1" in line
    assert "retained_l1=0" not in line
    body = after.get_function("matmul").as_python()
    assert body.count("lhs_l1") > 1
    assert body.index("lhs_l1") < body.index("for matmul_tile")

    lowered = _run_default(RetainedPanelMatmulProgram)
    aic = [function for function in lowered.functions.values() if function.func_type == ir.FunctionType.AIC]
    assert len(aic) == 1
    kernel = aic[0].as_python()
    assert kernel.count("pl.tile.load(") == 5
    assert kernel.index("lhs_l1__tile") < kernel.index("for matmul_tile")
    assert "pl.tile.extract(matmul_lhs_l1__tile" in " ".join(kernel.split())
    assert "pl.tile.create" not in kernel


def test_serial_matmul_dag_keeps_internal_handoff_in_l1(
    capfd: pytest.CaptureFixture[str],
):
    set_log_level(LogLevel.INFO)
    try:
        after = _run_auto_tile(ChainedMatmulProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[chain]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.ops["tensor.matmul"] == 2
    assert structure.ops["tensor.create_l1"] == 1
    assert structure.ops["tensor.assemble"] == 2
    assert structure.atomic_stores == 0
    assert "schedule=serial_dag" in line
    assert "requests=2" in line
    body = after.get_function("chain").as_python()
    assert body.index("chain_request_0_l1") < body.index("chain_request_1_tile")

    lowered = _run_default(ChainedMatmulProgram)
    aic = [function for function in lowered.functions.values() if function.func_type == ir.FunctionType.AIC]
    assert len(aic) == 1
    kernel = aic[0].as_python()
    assert "target_memory=pl.Mem.Mat" in kernel
    assert kernel.count("pl.tile.store(") == 1
    assert "pl.tile.assemble(" in kernel


def test_serial_matmul_dag_retains_one_compatible_boundary_until_last_use(
    capfd: pytest.CaptureFixture[str],
):
    set_log_level(LogLevel.INFO)
    try:
        after = _run_auto_tile(RepeatedRhsChainProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[chain]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    assert "schedule=serial_dag" in line
    assert "requests=2" in line
    assert "retained_panels=1" in line
    body = after.get_function("chain").as_python()
    assert body.count("chain_resident_0") > 1
    assert body.index("chain_resident_0") < body.index("chain_request_0_tile")


def test_multi_role_boundary_keeps_lhs_and_rhs_representations_distinct():
    after = _run_auto_tile(MultiRoleBoundaryProgram)
    function = after.get_function("tree")
    assert function is not None
    shared = next(param for param in function.params if param.name_hint.startswith("shared"))

    class SharedSliceCounter(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def visit_call(self, op: ir.Call) -> None:
            source = op.args[0] if op.op.name == "tensor.slice" and op.args else None
            if isinstance(source, ir.Var) and source.same_as(shared):
                self.count += 1
            super().visit_call(op)

    counter = SharedSliceCounter()
    counter.visit_stmt(function.body)
    # ``shared`` is LHS of one producer and RHS of another.  Role expansion
    # deliberately prevents one Mat-layout residency value from serving both.
    assert counter.count >= 2
    assert "resident_" not in function.as_python()


def test_serial_dag_peak_l1_covers_emitted_physical_high_water(
    capfd: pytest.CaptureFixture[str],
):
    """The cube plan must not price below the buffers emitted for its request DAG."""
    set_log_level(LogLevel.INFO)
    try:
        lowered = _run_default(MultiRoleBoundaryProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[tree]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    match = re.search(r"\bpeak_l1=(\d+)\b", line)
    assert match is not None
    planned_peak = int(match.group(1))
    emitted_high_water = _tile_memory_high_water(lowered, ir.MemorySpace.Mat)
    assert planned_peak >= emitted_high_water, (
        f"serial cube plan prices {planned_peak} L1 bytes but lowering addresses {emitted_high_water} bytes"
    )


def test_produced_value_used_in_both_roles_expands_to_two_physical_requests(
    capfd: pytest.CaptureFixture[str],
):
    set_log_level(LogLevel.INFO)
    try:
        after = _run_auto_tile(ProducedMultiRoleProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[chain]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    # The logical producer is replayed once for its LHS representation and once
    # for its RHS representation before the sink consumes both.
    assert "requests=3" in line
    assert _structure(after).ops["tensor.create_l1"] == 2


@pytest.mark.parametrize("program", [ChainedMatmulProgram, ProducedTreeMatmulProgram])
def test_serial_matmul_dag_is_numerically_correct(program: ir.Program):
    torch = pytest.importorskip("torch")
    from pypto.debug import torch_codegen  # noqa: PLC0415

    after = _run_auto_tile(program)
    namespace: dict = {}
    exec(torch_codegen(after), namespace)  # noqa: S102
    torch.manual_seed(4)
    if program is ChainedMatmulProgram:
        lhs = torch.randn(128, 256, dtype=torch.bfloat16) * 0.05
        middle = torch.randn(256, 128, dtype=torch.bfloat16) * 0.05
        rhs = torch.randn(128, 256, dtype=torch.bfloat16) * 0.05
        output = torch.full((128, 256), torch.nan, dtype=torch.bfloat16)
        actual = namespace["chain"](lhs, middle, rhs, output)
        expected = (lhs @ middle) @ rhs
    else:
        a = torch.randn(32, 48, dtype=torch.bfloat16) * 0.05
        b = torch.randn(48, 80, dtype=torch.bfloat16) * 0.05
        c = torch.randn(80, 64, dtype=torch.bfloat16) * 0.05
        d = torch.randn(64, 96, dtype=torch.bfloat16) * 0.05
        output = torch.full((32, 96), torch.nan, dtype=torch.float32)
        actual = namespace["tree"](a, b, c, d, output)
        expected = ((a @ b) @ (c @ d)).float()
    written = ~torch.isnan(actual)
    assert written.any()
    assert not written.all()
    assert torch.allclose(actual[written], expected[written], rtol=1e-3, atol=1e-3)


def test_fp32_internal_matmul_handoff_is_rejected_during_admission():
    @pl.program
    class Fp32Chain:
        @pl.function(attrs={"auto_tile": True})
        def chain(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            middle: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            intermediate: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(lhs, middle)
            return pl.matmul(intermediate, rhs)

    with pytest.raises(ValueError, match="must use FP16 or BF16 storage"):
        _run_auto_tile(Fp32Chain)


def test_serial_matmul_dag_declines_when_one_internal_region_cannot_fit_l1():
    @pl.program
    class OversizedIntermediate:
        @pl.function(attrs={"auto_tile": True})
        def chain(
            self,
            lhs: pl.Tensor[[16, 16], pl.BF16],
            wide: pl.Tensor[[16, 32768], pl.BF16],
            rhs: pl.Tensor[[32768, 16], pl.BF16],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            intermediate: pl.Tensor[[16, 32768], pl.BF16] = pl.matmul(lhs, wide)
            return pl.matmul(intermediate, rhs, out_dtype=pl.FP32)

    with pytest.raises(ValueError, match="one capacity-safe Ascend910B cube kernel"):
        _run_auto_tile(OversizedIntermediate)


@pytest.mark.parametrize("program", [MatmulProgram, RaggedMatmulProgram])
def test_emitted_matmul_is_numerically_correct(program: ir.Program):
    torch = pytest.importorskip("torch")
    from pypto.debug import torch_codegen  # noqa: PLC0415

    after = _run_auto_tile(program)
    code = torch_codegen(after)
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102
    torch.manual_seed(0)
    if program is MatmulProgram:
        lhs = torch.randn(64, 64, dtype=torch.float32)
        rhs = torch.randn(64, 64, dtype=torch.float32)
    else:
        lhs = torch.randn(130, 64, dtype=torch.float32)
        rhs = torch.randn(64, 260, dtype=torch.float32)
    output = torch.full((lhs.shape[0], rhs.shape[1]), torch.nan, dtype=torch.float32)
    actual = namespace["matmul"](lhs, rhs, output)
    expected = lhs @ rhs
    # The Torch interpreter intentionally executes block zero of an SPMD
    # scope.  Compare exactly the region that block zero assembled; device
    # system tests cover concurrent execution of the complete grid.
    written = ~torch.isnan(actual)
    assert written.any()
    assert not written.all()
    assert torch.allclose(actual[written], expected[written], rtol=1e-4, atol=1e-4)


def test_ragged_matmul_uses_clamped_static_regions(capfd: pytest.CaptureFixture[str]):
    set_log_level(LogLevel.INFO)
    try:
        capfd.readouterr()
        after = _run_auto_tile(RaggedMatmulProgram)
        line = next(line for line in capfd.readouterr().err.splitlines() if "AutoTile[matmul]" in line)
    finally:
        set_log_level(LogLevel.INFO)
    assert "spatial_policy=clamped_overlap" in line
    body = after.get_function("matmul").as_python()
    assert "pl.min" in body


def test_explicit_output_buffer_is_preserved():
    @pl.program
    class ExplicitOut:
        @pl.function(attrs={"auto_tile": True})
        def matmul(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP16],
            rhs: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            out = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
            return out

    after = _run_auto_tile(ExplicitOut)
    function = after.get_function("matmul")
    assert function is not None
    assert len(function.params) == 3
    # The output buffer is reused and the first K window initializes the Acc
    # directly, so no tensor-level identity allocation is needed.
    assert _structure(after).ops["tensor.create"] == 0
    assert _structure(after).ops["tensor.assemble"] == 1


def test_bf16_storage_and_repeated_operand_are_admitted():
    @pl.program
    class SquareMatmul:
        @pl.function(attrs={"auto_tile": True})
        def matmul(self, value: pl.Tensor[[64, 64], pl.BF16]) -> pl.Tensor[[64, 64], pl.BF16]:
            return pl.matmul(value, value)

    after = _run_auto_tile(SquareMatmul)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.ops["tensor.matmul"] == 1
    assert structure.ops["tensor.slice"] == 4


def test_called_helper_keeps_its_declared_signature():
    @pl.program
    class CalledHelper:
        @pl.function(attrs={"auto_tile": True})
        def helper(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            return pl.matmul(lhs, rhs)

        @pl.function
        def main(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            return self.helper(lhs, rhs)

    after = _run_auto_tile(CalledHelper)
    helper = after.get_function("helper")
    assert helper is not None
    assert len(helper.params) == 2
    # The only allocation is the helper-local GM output; the first K window
    # initializes the Acc directly.
    assert _structure(after).ops["tensor.create"] == 1


@pytest.mark.parametrize(
    "program", [MatmulProgram, Fp16MatmulProgram, Bf16MatmulProgram, RaggedKMatmulProgram]
)
def test_complete_pipeline_outlines_one_aic_kernel(program: ir.Program):
    lowered = _run_default(program)
    incore_types = [
        function.func_type for function in lowered.functions.values() if ir.is_incore_type(function.func_type)
    ]
    assert incore_types.count(ir.FunctionType.AIC) == 1
    assert incore_types.count(ir.FunctionType.AIV) == 0


def test_mixed_cube_vector_function_is_rejected():
    @pl.program
    class Mixed:
        @pl.function(attrs={"auto_tile": True})
        def kernel(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            mm: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(lhs, rhs)
            out: pl.Tensor[[64, 64], pl.FP32] = pl.exp(mm)
            return out

    with pytest.raises(ValueError, match="one homogeneous cube schedule"):
        _run_auto_tile(Mixed)


def test_transposed_matmul_is_rejected_during_admission():
    @pl.program
    class Transposed:
        @pl.function(attrs={"auto_tile": True})
        def matmul(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            return pl.matmul(lhs, rhs, a_trans=True)

    with pytest.raises(ValueError, match="non-transposed ND tensor.matmul only"):
        _run_auto_tile(Transposed)


def test_nonfractal_k_rejects_and_prior_l1_overflow_streams():
    @pl.program
    class NonFractalK:
        @pl.function(attrs={"auto_tile": True})
        def matmul(
            self,
            lhs: pl.Tensor[[64, 18], pl.FP32],
            rhs: pl.Tensor[[18, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            return pl.matmul(lhs, rhs)

    @pl.program
    class L1Overflow:
        @pl.function(attrs={"auto_tile": True})
        def matmul(
            self,
            lhs: pl.Tensor[[16, 16384], pl.FP32],
            rhs: pl.Tensor[[16384, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return pl.matmul(lhs, rhs)

    message = "one capacity-safe Ascend910B cube kernel"
    with pytest.raises(ValueError, match=message):
        _run_auto_tile(NonFractalK)
    streamed = _structure(_run_auto_tile(L1Overflow))
    assert streamed.spmd == 2
    assert streamed.atomic_stores == 1
    assert streamed.ops["tensor.matmul_acc"] == 4


def test_cube_autotile_rejects_unsupported_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises(ValueError, match="cube scheduling currently supports Ascend910B only"):
        _run_auto_tile(MatmulProgram)
