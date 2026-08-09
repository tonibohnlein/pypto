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

    def visit_call(self, op: ir.Call) -> None:
        self.ops[op.op.name] += 1
        super().visit_call(op)

    def visit_spmd_scope_stmt(self, op: ir.SpmdScopeStmt) -> None:
        self.spmd += 1
        assert isinstance(op.core_num, ir.ConstInt)
        self.work_units.append(int(op.core_num.value))
        super().visit_spmd_scope_stmt(op)


def _structure(program: ir.Program) -> _Structure:
    result = _Structure()
    result.visit_program(program)
    return result


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


def test_single_matmul_emits_one_cube_spmd_kernel():
    after = _run_auto_tile(MatmulProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.work_units[0] >= 1
    assert structure.ops["tensor.slice"] == 2
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
    assert "cube schedule=serial_matmul" in lines[0]
    assert "spatial_policy=" in lines[0]
    assert "gm_to_l1_bytes=" in lines[0]
    assert "l0_tile=" in lines[0]


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
    assert structure.ops["tensor.slice"] == 2


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
    assert _structure(after).ops["tensor.create"] == 1


@pytest.mark.parametrize("program", [MatmulProgram, Fp16MatmulProgram, Bf16MatmulProgram])
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


def test_nonfractal_k_and_l1_overflow_fail_without_partial_emission():
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
    with pytest.raises(ValueError, match=message):
        _run_auto_tile(L1Overflow)


def test_cube_autotile_rejects_unsupported_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises(ValueError, match="cube scheduling currently supports Ascend910B only"):
        _run_auto_tile(MatmulProgram)
