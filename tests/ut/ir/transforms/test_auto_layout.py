# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit and codegen tests for layout instance readout, writeback, and cost calibration.

Verifies:
1. Pass placement after `AutoTileMatmulL0` (Pass 20) and `InferTileMemorySpace`
   (Pass 22) using L0-fitting shapes (`64x128 @ 128x64`), completely bypassing
   `AutoTileMatmulL0`.
2. Solver instance extraction (`extract_layout_instance`) reading out GM
   tensors (`DDR`), L1 tiles (`Mat`), and L0 tiles (`Left`, `Right`, `Acc`)
   plus all data-movement and compute operations.
3. Cost model evaluation and enumeration (`LayoutCostModel`,
   `evaluate_assignment`, `enumerate_assignments`), including shared-conversion
   accounting (Section 2 of Eisenhofer et al., arXiv:2608.21555).
4. Result writeback (`apply_layout_assignment`) into post-Pass-22 IR for both
   GM `TensorLayout` and on-chip L0/L1 `TileLayoutDesc`, followed by
   `run_post_layout_passes` and `.pto` MLIR emission.
5. Exact host `pack_nz` / `unpack_nz` round-trip and blocked-offset placement.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto import DataType, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir import auto_layout, auto_layout_bench
from pypto.ir.auto_layout import (
    COL_MAJOR_FLAT,
    FRACTAL_ACC_NZ,
    FRACTAL_NZ,
    FRACTAL_ZN,
    ROW_BCAST_FLAT,
    ROW_MAJOR_FLAT,
    CalibrationMeasurement,
    CostBreakdown,
    LayoutAssignment,
    LayoutCostModel,
    LayoutOperationNode,
    LayoutProblemInstance,
    LayoutValueNode,
    LayoutVariantArtifact,
    OperatorConfig,
    apply_layout_assignment,
    c0_elements,
    compile_layout_variants,
    enumerate_assignments,
    evaluate_assignment,
    extract_layout_instance,
    fit_cost_model_from_measurements,
    is_nz_compatible_shape,
    make_auto_layout_pass,
    pack_dn,
    pack_nz,
    run_post_layout_passes,
    run_pre_layout_passes,
    unpack_dn,
    unpack_nz,
)
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    reset_for_testing()
    yield
    reset_for_testing()


def _emit_pto_from_lowered(lowered: ir.Program) -> str:
    """Emit `.pto` MLIR for all InCore functions in an already-lowered `Program`."""
    parts: list[str] = []
    for func in lowered.functions.values():
        if func.func_type not in (pl.FunctionType.InCore, pl.FunctionType.AIC, pl.FunctionType.AIV):
            continue
        single = ir.Program([func], func.name, lowered.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


@pl.program
class MatmulL0Fitting:
    """L0-fitting 64x128 @ 128x64 FP16 -> FP32 matmul program.

    Footprints:
    - A (`[64, 128]` FP16) = 16 KiB <= 64 KiB L0A (`Left`)
    - B (`[128, 64]` FP16) = 16 KiB <= 64 KiB L0B (`Right`)
    - C (`[64, 64]` FP32)  = 16 KiB <= 128 KiB L0C (`Acc`)
    Because `(m, n, k) == (M, N, K)`, `AutoTileMatmulL0` is a complete no-op.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        x: pl.Tensor[[64, 128], pl.FP16],
        w: pl.Tensor[[64, 128], pl.FP16],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        xt: pl.Tile[[64, 128], pl.FP16, pl.Mem.Mat] = pl.load(x, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        wt: pl.Tile[[64, 128], pl.FP16, pl.Mem.Mat] = pl.load(w, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        wt_t: pl.Tile[[128, 64], pl.FP16, pl.Mem.Mat] = pl.tile.transpose_view(wt)
        acc: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.matmul(xt, wt_t)
        out_res: pl.Tensor[[64, 64], pl.FP32] = pl.store(acc, [0, 0], out)
        return out_res

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[64, 128], pl.FP16],
        w: pl.Tensor[[64, 128], pl.FP16],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.matmul_kernel(x, w, out)


class TestInstanceReadoutAndWriteback:
    """Test solver instance readout and writeback after `InferTileMemorySpace` (Pass 22)."""

    def test_readout_captures_gm_l1_and_l0_values_and_bypasses_autotile(self):
        """Verify that `run_pre_layout_passes` bypasses `AutoTileMatmulL0` and exposes GM, L1, and L0."""
        set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            pre_prog = run_pre_layout_passes(MatmulL0Fitting)

        # Verify AutoTileMatmulL0 was completely bypassed (no ForStmt K-loops emitted).
        printed = ir.python_print(pre_prog)
        assert "pl.range" not in printed
        assert any(f.name == "matmul_kernel" for f in pre_prog.functions.values())

        instance = extract_layout_instance(
            pre_prog,
            cost_model=LayoutCostModel(expected_uses=1),
            reusable_tensors={"w"},
        )

        # 1. GM Tensors (`DDR`)
        assert instance.values["x"].memory_space == ir.MemorySpace.DDR
        assert instance.values["w"].memory_space == ir.MemorySpace.DDR
        assert instance.values["out"].memory_space == ir.MemorySpace.DDR
        assert instance.values["w"].admissible_layouts == (ir.TensorLayout.ND, ir.TensorLayout.NZ)

        # 2. L1 Tiles (`Mat`)
        assert instance.values["xt"].memory_space == ir.MemorySpace.Mat
        assert instance.values["xt"].current_layout == FRACTAL_NZ
        assert instance.values["wt"].memory_space == ir.MemorySpace.Mat
        assert instance.values["wt"].current_layout == FRACTAL_NZ
        assert instance.values["wt_t"].memory_space == ir.MemorySpace.Mat
        assert instance.values["wt_t"].current_layout == FRACTAL_ZN

        # 3. L0 Tiles (`Left`, `Right`, `Acc`) materialized by InferTileMemorySpace
        l0_left = [v for v in instance.values.values() if v.memory_space == ir.MemorySpace.Left]
        l0_right = [v for v in instance.values.values() if v.memory_space == ir.MemorySpace.Right]
        l0_acc = [v for v in instance.values.values() if v.memory_space == ir.MemorySpace.Acc]
        assert len(l0_left) == 1
        assert len(l0_right) == 1
        assert len(l0_acc) == 1
        assert l0_left[0].current_layout == FRACTAL_NZ
        assert l0_right[0].current_layout == FRACTAL_ZN
        assert l0_acc[0].current_layout == FRACTAL_ACC_NZ

        # 4. Operations in dataflow order: 2 loads, 1 transpose_view, 2 L1->L0 moves, 1 matmul, 1 store
        op_names = [op.op_name for op in instance.operations]
        assert op_names == [
            "tile.load",
            "tile.load",
            "tile.transpose_view",
            "tile.move",
            "tile.move",
            "tile.matmul",
            "tile.store",
        ]

    def test_writeback_gm_nz_and_l0_tile_layouts_lowers_to_pto(self):
        """Verify writing back a candidate `LayoutAssignment` after Pass 22 lowers to `.pto` MLIR."""
        set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            pre_prog = run_pre_layout_passes(MatmulL0Fitting)
            instance = extract_layout_instance(
                pre_prog,
                cost_model=LayoutCostModel(expected_uses=100),
                reusable_tensors={"w"},
            )
            ranked = enumerate_assignments(instance)
            best = ranked[0]

            # At R=100, reusable weight `w` prefers NZ while streaming `x` stays ND.
            assert best.assignment.tensor_layouts["x"] == ir.TensorLayout.ND
            assert best.assignment.tensor_layouts["w"] == ir.TensorLayout.NZ
            assert best.assignment.tensor_layouts["out"] == ir.TensorLayout.ND

            rewritten = apply_layout_assignment(pre_prog, best.assignment)
            lowered = run_post_layout_passes(rewritten)
            pto_text = _emit_pto_from_lowered(lowered)

        assert "#pto.layout<nz>" in pto_text
        # [64, 128] FP16 (c0=16) blocks to rank-5 [1, 128/16, 64/16, 16, 16] = [1, 8, 4, 16, 16].
        assert "!pto.tensor_view<?x?x?x?x?xf16>" in pto_text
        assert "pto.tmatmul" in pto_text

    def test_writeback_explicit_l0_move_layout_updates_ir(self):
        """Verify writing back an explicit L0 `TileLayoutDesc` onto a `tile.move` variable."""
        set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            pre_prog = run_pre_layout_passes(MatmulL0Fitting)
            instance = extract_layout_instance(pre_prog)

            l0_left_name = next(
                v.name for v in instance.values.values() if v.memory_space == ir.MemorySpace.Left
            )
            custom_assignment = LayoutAssignment(
                value_layouts={
                    "w": ir.TensorLayout.NZ,
                    l0_left_name: FRACTAL_ZN,
                }
            )
            rewritten = apply_layout_assignment(pre_prog, custom_assignment)

        rewritten_text = ir.python_print(rewritten, explicit_layout=True)
        assert "blayout=pl.TileLayout.row_major" in rewritten_text
        assert "slayout=pl.TileLayout.col_major" in rewritten_text

    def test_valid_assignments_make_resolve_backend_op_layouts_a_noop_and_respect_backend(self):
        """Verify that backend-filtered assignments make `ResolveBackendOpLayouts` (Pass 24) a no-op."""
        set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            pre_prog = run_pre_layout_passes(MatmulL0Fitting)
            inst_910 = extract_layout_instance(
                pre_prog,
                cost_model=LayoutCostModel(backend_type=BackendType.Ascend910),
                reusable_tensors={"w"},
            )
            inst_950 = extract_layout_instance(
                pre_prog,
                cost_model=LayoutCostModel(backend_type=BackendType.Ascend950),
                reusable_tensors={"w"},
            )

            # Ascend910 ('a2a3') admits both ND and NZ output store from FP32 Acc,
            # whereas Ascend950 ('a5') rejects Acc -> NZ store.
            store_910 = next(op for op in inst_910.operations if op.op_name == "tile.store")
            store_950 = next(op for op in inst_950.operations if op.op_name == "tile.store")
            assert {c.output_layout for c in store_910.configs} == {
                ir.TensorLayout.ND,
                ir.TensorLayout.NZ,
            }
            assert {c.output_layout for c in store_950.configs} == {ir.TensorLayout.ND}

            # Every candidate assignment produced for Ascend910 must already satisfy
            # `ResolveBackendOpLayouts` so Pass 24 is a complete no-op.
            resolve_pass = passes.resolve_backend_op_layouts()
            for candidate in enumerate_assignments(inst_910):
                rewritten = apply_layout_assignment(pre_prog, candidate.assignment)
                after_resolve = resolve_pass(rewritten)
                ir.assert_structural_equal(after_resolve, rewritten)


class TestCostModelCalibrationAndEvaluation:
    """Test `LayoutCostModel`, `evaluate_assignment`, and `enumerate_assignments`."""

    @pytest.mark.parametrize(
        "dtype, backend, shape, expected_cycles",
        [
            (DataType.FP16, BackendType.Ascend910, (64, 128, 64), 149.0),
            (DataType.FP32, BackendType.Ascend910, (64, 128, 64), 533.0),
            (DataType.FP32, BackendType.Ascend950, (64, 128, 64), 2069.0),
            (DataType.FP16, BackendType.Ascend910, (17, 33, 31), 33.0),
            (DataType.FP32, BackendType.Ascend910, (17, 33, 31), 61.0),
        ],
    )
    def test_matmul_cost_counts_dtype_specific_hardware_fractals(
        self,
        dtype: DataType,
        backend: BackendType,
        shape: tuple[int, int, int],
        expected_cycles: float,
    ):
        set_backend_type(backend)
        m_dim, k_dim, n_dim = shape
        lhs = LayoutValueNode(
            name="lhs",
            shape=(m_dim, k_dim),
            dtype=dtype,
            memory_space=ir.MemorySpace.Left,
            current_layout=FRACTAL_NZ,
            admissible_layouts=(FRACTAL_NZ,),
        )
        rhs = replace(lhs, name="rhs", shape=(k_dim, n_dim), memory_space=ir.MemorySpace.Right)
        out = replace(lhs, name="out", shape=(m_dim, n_dim), memory_space=ir.MemorySpace.Acc)
        model = LayoutCostModel(backend_type=backend)
        assert model.estimate_matmul_cost(lhs, rhs, out) == pytest.approx(expected_cycles)
        faster = replace(model, cube_macs_per_cycle=2 * model.cube_macs_per_cycle)
        assert faster.estimate_matmul_cost(lhs, rhs, out) == pytest.approx(
            model.cube_issue_overhead_cycles + (expected_cycles - model.cube_issue_overhead_cycles) / 2
        )

    def test_cost_model_single_use_vs_reused_weight_ranking(self):
        """Compare candidate rankings at R=1 (keeps ND) vs. R=100 (selects NZ for reusable weight `w`)."""
        set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            pre_prog = run_pre_layout_passes(MatmulL0Fitting)

        # Single use (R=1): packing `w` costs more than a single ND2NZ -> NZ2NZ load saving.
        inst_r1 = extract_layout_instance(
            pre_prog,
            cost_model=LayoutCostModel(expected_uses=1),
            reusable_tensors={"w"},
        )
        ranked_r1 = enumerate_assignments(inst_r1)
        assert ranked_r1[0].assignment.tensor_layouts["x"] == ir.TensorLayout.ND
        assert ranked_r1[0].assignment.tensor_layouts["w"] == ir.TensorLayout.ND
        assert ranked_r1[0].conversion_cost_cycles == pytest.approx(0.0)

        # Reused weight (R=100): one-time packing of `w` is amortized across 100 runs.
        inst_r100 = extract_layout_instance(
            pre_prog,
            cost_model=LayoutCostModel(expected_uses=100),
            reusable_tensors={"w"},
        )
        ranked_r100 = enumerate_assignments(inst_r100)
        assert ranked_r100[0].assignment.tensor_layouts["x"] == ir.TensorLayout.ND
        assert ranked_r100[0].assignment.tensor_layouts["w"] == ir.TensorLayout.NZ
        assert ranked_r100[0].conversion_cost_cycles > 0.0
        assert ranked_r100[0].total_cost_cycles < ranked_r100[1].total_cost_cycles

    def test_shared_conversion_is_charged_once_across_consumers(self):
        """Verify $\\sum_{\\ell \\in \\text{cons}(t)} C^{\\text{tr}}_t(\\text{prod}(t), \\ell)$ on fan-out."""
        t_in = LayoutValueNode(
            name="x_in",
            shape=(64, 64),
            dtype=DataType.FP16,
            memory_space=ir.MemorySpace.DDR,
            current_layout=ir.TensorLayout.ND,
            admissible_layouts=(ir.TensorLayout.ND,),
        )
        t_mid = LayoutValueNode(
            name="t",
            shape=(64, 64),
            dtype=DataType.FP16,
            memory_space=ir.MemorySpace.DDR,
            current_layout=ir.TensorLayout.ND,
            admissible_layouts=(ir.TensorLayout.ND, ir.TensorLayout.NZ),
            producer_op_id="P",
        )
        t_out1 = LayoutValueNode(
            name="y",
            shape=(64, 64),
            dtype=DataType.FP16,
            memory_space=ir.MemorySpace.DDR,
            current_layout=ir.TensorLayout.ND,
            admissible_layouts=(ir.TensorLayout.ND,),
            producer_op_id="Q1",
        )
        t_out2 = LayoutValueNode(
            name="z",
            shape=(64, 64),
            dtype=DataType.FP16,
            memory_space=ir.MemorySpace.DDR,
            current_layout=ir.TensorLayout.ND,
            admissible_layouts=(ir.TensorLayout.ND,),
            producer_op_id="Q2",
        )

        # Conversion cost of `t`: gm_pack_fixed_cycles=4.0, gm_pack_bytes_per_cycle=inf -> C^tr = 4.0
        model = LayoutCostModel(
            gm_pack_fixed_cycles=4.0,
            gm_pack_bytes_per_cycle=1e18,
        )

        op_p = LayoutOperationNode(
            op_id="P",
            op_name="producer",
            inputs=("x_in",),
            output="t",
            invocations=1,
            configs=(
                OperatorConfig("P", "producer", (ir.TensorLayout.ND,), ir.TensorLayout.ND, 2.0),
                OperatorConfig("P", "producer", (ir.TensorLayout.ND,), ir.TensorLayout.NZ, 5.0),
            ),
        )
        op_q1 = LayoutOperationNode(
            op_id="Q1",
            op_name="consumer",
            inputs=("t",),
            output="y",
            invocations=1,
            configs=(
                OperatorConfig("Q1", "consumer", (ir.TensorLayout.ND,), ir.TensorLayout.ND, 6.0),
                OperatorConfig("Q1", "consumer", (ir.TensorLayout.NZ,), ir.TensorLayout.ND, 1.0),
            ),
        )
        op_q2 = LayoutOperationNode(
            op_id="Q2",
            op_name="consumer",
            inputs=("t",),
            output="z",
            invocations=1,
            configs=(
                OperatorConfig("Q2", "consumer", (ir.TensorLayout.ND,), ir.TensorLayout.ND, 6.0),
                OperatorConfig("Q2", "consumer", (ir.TensorLayout.NZ,), ir.TensorLayout.ND, 1.0),
            ),
        )

        instance = LayoutProblemInstance(
            values={"x_in": t_in, "t": t_mid, "y": t_out1, "z": t_out2},
            operations=[op_p, op_q1, op_q2],
            cost_model=model,
        )

        # Assignment 1: P produces ND (2), Q1 & Q2 both consume NZ (1 + 1) + 1 shared conversion (4) = 8.
        shared_conv_eval = evaluate_assignment(
            instance,
            LayoutAssignment(
                value_layouts={"t": ir.TensorLayout.ND},
                op_configs={
                    "P": op_p.configs[0],
                    "Q1": op_q1.configs[1],
                    "Q2": op_q2.configs[1],
                },
            ),
        )
        assert shared_conv_eval.operator_cost_cycles == pytest.approx(4.0)
        assert shared_conv_eval.conversion_cost_cycles == pytest.approx(4.0)
        assert shared_conv_eval.total_cost_cycles == pytest.approx(8.0)

        # Assignment 2 (best): P produces NZ directly (5), Q1 & Q2 consume NZ (1 + 1) + 0 conversions = 7.
        ranked = enumerate_assignments(instance)
        assert ranked[0].assignment.op_configs["P"].output_layout == ir.TensorLayout.NZ
        assert ranked[0].total_cost_cycles == pytest.approx(7.0)
        assert ranked[0].conversion_cost_cycles == pytest.approx(0.0)


class TestHostNzPacking:
    """Verify `pack_nz`, `unpack_nz`, and `is_nz_compatible_shape`."""

    @pytest.mark.parametrize(
        ("dtype", "expected_c0"),
        [(torch.int8, 32), (torch.float16, 16), (torch.float32, 8)],
    )
    def test_pack_and_unpack_nz_2d_and_3d(self, dtype: torch.dtype, expected_c0: int):
        assert c0_elements(dtype) == expected_c0
        rows, cols = 64, 128
        logical_2d = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols).to(dtype)
        packed_2d = pack_nz(logical_2d)
        assert packed_2d.shape == (rows, cols)

        flat = packed_2d.reshape(-1)
        for r, c in ((0, 0), (0, expected_c0), (1, 0), (17, 33), (rows - 1, cols - 1)):
            i, ii = divmod(r, 16)
            j, jj = divmod(c, expected_c0)
            offset = j * (rows * expected_c0) + i * (16 * expected_c0) + ii * expected_c0 + jj
            assert flat[offset] == logical_2d[r, c]

        assert torch.equal(unpack_nz(packed_2d), logical_2d)

        logical_3d = torch.stack([logical_2d, logical_2d + 1])
        packed_3d = pack_nz(logical_3d)
        assert packed_3d.shape == (2, rows, cols)
        assert torch.equal(unpack_nz(packed_3d), logical_3d)

    def test_nz_compatibility_checks(self):
        assert is_nz_compatible_shape((128, 256), DataType.FP16)
        assert is_nz_compatible_shape((4, 128, 256), DataType.INT8)
        assert not is_nz_compatible_shape((120, 256), DataType.FP16)
        assert not is_nz_compatible_shape((128, 24), DataType.FP16)
        assert not is_nz_compatible_shape((256,), DataType.FP16)


class TestPassPipelineIntegrationAndCalibration:
    """Verify `PassManager`/`ir.compile` insertion, variant export, and linear calibration."""

    def test_pass_manager_inserts_layout_pass_after_infer_tile_memory_space(self):
        default_pm = PassManager.get_strategy(OptimizationStrategy.Default)
        assert "AutoLayout" not in default_pm.pass_names

        custom_pm = PassManager.get_strategy(
            OptimizationStrategy.Default,
            layout_pass=lambda: make_auto_layout_pass(LayoutAssignment(value_layouts={})),
        )
        names = custom_pm.pass_names
        infer_idx = names.index("InferTileMemorySpace")
        resolve_idx = names.index("ResolveBackendOpLayouts")
        assert names[infer_idx + 1] == "AutoLayout"
        assert names[infer_idx + 2] == "BlockNzTensorViews"
        assert infer_idx < resolve_idx

    def test_ir_compile_with_layout_pass_and_variant_export(self, tmp_path: Path):
        set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            pre_prog = run_pre_layout_passes(MatmulL0Fitting)
            inst = extract_layout_instance(
                pre_prog,
                cost_model=LayoutCostModel(expected_uses=100),
                reusable_tensors={"w"},
            )
            best = enumerate_assignments(inst)[0]
            assert best.assignment.tensor_layouts["w"] == ir.TensorLayout.NZ

        compiled_dir = tmp_path / "single_compile"
        ir.compile(
            MatmulL0Fitting,
            output_dir=str(compiled_dir),
            backend_type=BackendType.Ascend910,
            skip_ptoas=True,
            layout_pass=lambda: make_auto_layout_pass(best.assignment),
        )
        pto_files = list((compiled_dir / "kernels").rglob("*.pto"))
        assert len(pto_files) == 1
        pto_text = pto_files[0].read_text()
        assert "nz" in pto_text.lower()

        variants_dir = tmp_path / "variants"
        _, variants = compile_layout_variants(
            MatmulL0Fitting,
            output_dir=variants_dir,
            cost_model=LayoutCostModel(expected_uses=100),
            reusable_tensors={"w"},
            backend_type=BackendType.Ascend910,
            skip_ptoas=True,
        )
        assert len(variants) == 8
        manifest = json.loads((variants_dir / "manifest.json").read_text())
        assert manifest["num_variants"] == 8
        assert len(manifest["variants"]) == 8
        assert any(v["gm_input_layouts"].get("w") == "NZ" for v in manifest["variants"])

    @pytest.mark.parametrize("reuse_horizons", [(100, 100, 100, 100), (1, 10, 100, 1000)])
    def test_fit_cost_model_from_synthetic_measurements(self, reuse_horizons: tuple[int, ...]):
        set_backend_type(BackendType.Ascend910)
        shapes = [
            (64, 128, 96, None, None),
            (64, 256, 64, None, None),
            (128, 1024, 1024, 128, 128),
            (128, 128, 128, None, None),
        ]
        true_overhead_us = 5.0
        true_us_per_cycle = 1.0 / 1500.0
        true_nd2nz_bpc = 16.0
        true_nz2nz_bpc = 64.0
        true_store_nd_bpc = 24.0
        true_store_nz_bpc = 56.0
        base_model = LayoutCostModel(
            expected_uses=100,
            backend_type=BackendType.Ascend910,
            mte2_nd2nz_row_penalty_cycles=0.3,
            fixpipe_drain_fixed_cycles=7.0,
        )
        nd_ratio = base_model.mte2_nd2nz_contiguous_bytes_per_cycle / base_model.mte2_nd2nz_bytes_per_cycle
        true_model = replace(
            base_model,
            mte2_nd2nz_bytes_per_cycle=true_nd2nz_bpc,
            mte2_nd2nz_contiguous_bytes_per_cycle=nd_ratio * true_nd2nz_bpc,
            mte2_nz2nz_bytes_per_cycle=true_nz2nz_bpc,
            fixpipe_acc_to_nd_bytes_per_cycle=true_store_nd_bpc,
            fixpipe_acc_to_nz_bytes_per_cycle=true_store_nz_bpc,
        )
        measurements: list[CalibrationMeasurement] = []
        for s_idx, (m_dim, k_dim, n_dim, k_tile, n_tile) in enumerate(shapes):
            expected_uses = reuse_horizons[s_idx]
            case = auto_layout_bench.build_single_matmul_case(
                m_dim,
                k_dim,
                n_dim,
                k_tile=k_tile,
                n_tile=n_tile,
                expected_uses=expected_uses,
                seed=s_idx,
            )
            pre_prog = run_pre_layout_passes(case.program)
            inst = extract_layout_instance(
                pre_prog,
                cost_model=replace(
                    base_model,
                    expected_uses=expected_uses,
                    mte1_l0a_bytes_per_cycle=2.0 * base_model.mte1_l0a_bytes_per_cycle,
                ),
                reusable_tensors={"w"},
            )
            true_inst = extract_layout_instance(
                pre_prog,
                cost_model=replace(true_model, expected_uses=expected_uses),
                reusable_tensors={"w"},
            )
            true_variants = {
                tuple(sorted(bd.assignment.value_layouts.items())): bd
                for bd in enumerate_assignments(true_inst, deduplicate=True)
            }
            for v_idx, bd in enumerate(enumerate_assignments(inst, deduplicate=True)):
                true_bd = true_variants[tuple(sorted(bd.assignment.value_layouts.items()))]
                measured_us = (
                    true_overhead_us + true_us_per_cycle * true_bd.operator_cost_cycles / expected_uses
                )
                measurements.append(
                    CalibrationMeasurement(
                        variant_id=f"{case.case_id}/v{v_idx}",
                        measured_us=measured_us,
                        instance=inst,
                        breakdown=bd,
                        label=case.case_id,
                    )
                )

        report = fit_cost_model_from_measurements(measurements, base_model)
        assert set(report.residuals_us) == {measurement.variant_id for measurement in measurements}
        assert report.r2 > 0.999
        assert report.spearman_rho == pytest.approx(1.0)
        assert report.effective_freq_fallback is False
        assert report.fallback_params == ()
        assert report.fixed_overhead_us == pytest.approx(true_overhead_us, abs=1e-4)
        assert report.effective_freq_ghz == pytest.approx(1.5, rel=1e-3)
        assert report.fitted_model.mte2_nd2nz_bytes_per_cycle == pytest.approx(true_nd2nz_bpc, rel=1e-3)
        assert report.fitted_model.mte2_nd2nz_contiguous_bytes_per_cycle == pytest.approx(
            nd_ratio * true_nd2nz_bpc, rel=1e-3
        )
        assert report.fitted_model.mte2_nz2nz_bytes_per_cycle == pytest.approx(true_nz2nz_bpc, rel=1e-3)
        assert report.fitted_model.fixpipe_acc_to_nd_bytes_per_cycle == pytest.approx(
            true_store_nd_bpc, rel=1e-3
        )
        assert report.fitted_model.fixpipe_acc_to_nz_bytes_per_cycle == pytest.approx(
            true_store_nz_bpc, rel=1e-3
        )
        assert report.fitted_model.mte2_nd2nz_row_penalty_cycles == 0.3
        assert report.fitted_model.fixpipe_drain_fixed_cycles == 7.0
        assert max(abs(residual) for residual in report.residuals_us.values()) < 1e-8

        serialized = auto_layout_bench._serialize_calibration_report(measurements, base_model)
        assert serialized is not None
        reloaded = json.loads(json.dumps(serialized))
        assert replace(base_model, **reloaded["fitted_params"]) == report.fitted_model

        one_shape = [m for m in measurements if m.label == measurements[0].label]
        unidentifiable = fit_cost_model_from_measurements(one_shape, base_model)
        assert unidentifiable.effective_freq_fallback is True
        assert set(unidentifiable.fallback_params) == set(unidentifiable.param_t_stats)
        assert all(t_stat == 0.0 for t_stat in unidentifiable.param_t_stats.values())

    def test_calibration_rank_diagnostics_handle_ties(self):
        assert auto_layout._rank_array([3.0, 1.0, 1.0, 2.0]).tolist() == [3.0, 0.5, 0.5, 2.0]
        assert auto_layout._spearman_rank_correlation([1.0, 1.0, 2.0], [2.0, 2.0, 4.0]) == pytest.approx(1.0)
        assert auto_layout._spearman_rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
        assert auto_layout._spearman_rank_correlation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 0.0

    def test_auto_layout_bench_compile_only_suite(self, tmp_path: Path):
        cases = [
            auto_layout_bench.build_single_matmul_case(64, 128, 64, expected_uses=100, seed=0),
            auto_layout_bench.build_two_matmul_chain_case(64, 128, 64, 64, expected_uses=100, seed=1),
        ]
        summary = auto_layout_bench.run_calibration_suite(
            cases,
            output_dir=tmp_path / "calib_suite",
            platform="a2a3",
            compile_only=True,
        )
        assert summary["compile_only"] is True
        assert summary["calibration"] is None
        assert len(summary["cases"]) == 2
        single_variants = summary["cases"][0]["variants"]
        chain_variants = summary["cases"][1]["variants"]
        assert len(single_variants) == 8
        assert len(chain_variants) == 32
        assert any(v["gm_layouts"].get("mid") == "NZ" for v in chain_variants)
        assert any(v["gm_layouts"].get("mid") == "ND" for v in chain_variants)

    def test_default_suite_all_single_matmuls_fit_l0_with_8_variants(self, tmp_path: Path):
        cases = auto_layout_bench.default_a2_calibration_suite(include_chain=False, expected_uses=100)
        summary = auto_layout_bench.run_calibration_suite(
            cases,
            output_dir=tmp_path / "single_suite",
            platform="a2a3",
            compile_only=True,
        )
        assert len(summary["cases"]) == 12
        for case_rec in summary["cases"]:
            variants = case_rec["variants"]
            if case_rec["case_id"].endswith("_outer_m"):
                # Outer-M tiling (`m_tile < M` on 2D output) legally restricts `out` to `ND` (4 variants).
                assert len(variants) == 4, (
                    f"{case_rec['case_id']} expected 4 ND-out variants: {len(variants)}"
                )
                assert all(v["gm_output_layouts"].get("out") == "ND" for v in variants)
            else:
                assert len(variants) == 8, f"{case_rec['case_id']} unexpectedly split L0: {len(variants)}"
                assert any(v["gm_output_layouts"].get("out") == "NZ" for v in variants)
                assert any(v["gm_output_layouts"].get("out") == "ND" for v in variants)

    def test_tiled_loop_multiplicity_and_large_chain_extraction(self):
        set_backend_type(BackendType.Ascend910)
        model = LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910)

        tiled = auto_layout_bench.build_single_matmul_case(
            128, 1024, 1024, k_tile=128, n_tile=128, expected_uses=100, seed=0
        )
        pre_tiled = run_pre_layout_passes(tiled.program)
        inst_tiled = extract_layout_instance(
            pre_tiled, cost_model=model, reusable_tensors=tiled.reusable_tensors
        )
        assert set(inst_tiled.values) >= {"x", "w", "out"}
        loads = [op for op in inst_tiled.operations if op.op_name == "tile.load"]
        stores = [op for op in inst_tiled.operations if op.op_name == "tile.store"]
        assert len(loads) == 2
        assert all(op.invocations == 6400 for op in loads)
        assert len(stores) == 1
        assert stores[0].invocations == 800
        assert len(enumerate_assignments(inst_tiled, deduplicate=True)) == 8

        large_chain = auto_layout_bench.build_two_matmul_chain_case(
            64,
            2048,
            512,
            1024,
            k_tile=256,
            h_tile_s1=64,
            h_tile_s2=128,
            n_tile=64,
            expected_uses=100,
            seed=1,
        )
        pre_chain = run_pre_layout_passes(large_chain.program)
        inst_chain = extract_layout_instance(
            pre_chain, cost_model=model, reusable_tensors=large_chain.reusable_tensors
        )
        gm_names = {n for n, v in inst_chain.values.items() if v.memory_space == ir.MemorySpace.DDR}
        assert gm_names == {"x", "w1", "mid", "w2", "out"}
        assert len(enumerate_assignments(inst_chain, deduplicate=True)) == 32

        # Verify k_tile granularity sensitivity: kt64 (128B) > kt128 (256B) > kt256 (512B) ND->NZ delta.
        kt_deltas: list[float] = []
        for kt in (64, 128, 256):
            c_kt = auto_layout_bench.build_single_matmul_case(64, 4096, 64, k_tile=kt, expected_uses=1)
            inst_kt = extract_layout_instance(
                run_pre_layout_passes(c_kt.program), cost_model=LayoutCostModel(expected_uses=1)
            )
            by_gm = {
                (
                    bd.assignment.tensor_layouts["x"],
                    bd.assignment.tensor_layouts["w"],
                    bd.assignment.tensor_layouts["out"],
                ): bd.operator_cost_cycles
                for bd in enumerate_assignments(inst_kt, deduplicate=True)
            }
            kt_deltas.append(
                by_gm[(ir.TensorLayout.ND, ir.TensorLayout.ND, ir.TensorLayout.ND)]
                - by_gm[(ir.TensorLayout.NZ, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
            )
        assert kt_deltas[0] > kt_deltas[1] > kt_deltas[2] > 0.0

        # Verify stationarity sensitivity:
        # 1. Outer-N 128x1024x1024 (`x` 256KiB re-read 8x in L2, `w` 2MiB streamed 1x) -> delta(w) > delta(x)
        by_gm_n = {
            (
                bd.assignment.tensor_layouts["x"],
                bd.assignment.tensor_layouts["w"],
                bd.assignment.tensor_layouts["out"],
            ): bd.operator_cost_cycles
            for bd in enumerate_assignments(inst_tiled, deduplicate=True)
        }
        dx_n = (
            by_gm_n[(ir.TensorLayout.ND, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
            - by_gm_n[(ir.TensorLayout.NZ, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
        )
        dw_n = (
            by_gm_n[(ir.TensorLayout.NZ, ir.TensorLayout.ND, ir.TensorLayout.ND)]
            - by_gm_n[(ir.TensorLayout.NZ, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
        )
        assert dw_n > dx_n > 0.0

        # 2. Outer-M 1024x1024x128 (`x` 2MiB streamed 1x, `w` 256KiB re-read 8x in L2) -> delta(x) > delta(w)
        outer_m = auto_layout_bench.build_single_matmul_case(
            1024, 1024, 128, m_tile=128, k_tile=128, expected_uses=100, seed=8
        )
        inst_m = extract_layout_instance(
            run_pre_layout_passes(outer_m.program),
            cost_model=model,
            reusable_tensors=outer_m.reusable_tensors,
        )
        by_gm_m = {
            (
                bd.assignment.tensor_layouts["x"],
                bd.assignment.tensor_layouts["w"],
                bd.assignment.tensor_layouts["out"],
            ): bd.operator_cost_cycles
            for bd in enumerate_assignments(inst_m, deduplicate=True)
        }
        assert len(by_gm_m) == 4
        dx_m = (
            by_gm_m[(ir.TensorLayout.ND, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
            - by_gm_m[(ir.TensorLayout.NZ, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
        )
        dw_m = (
            by_gm_m[(ir.TensorLayout.NZ, ir.TensorLayout.ND, ir.TensorLayout.ND)]
            - by_gm_m[(ir.TensorLayout.NZ, ir.TensorLayout.NZ, ir.TensorLayout.ND)]
        )
        assert dx_m > dw_m > 0.0

        # 3. Stationary-A 64x256x2048 (`x` hoisted in L1 Mat once, `w` loaded 32x)
        stat_a = auto_layout_bench.build_single_matmul_case(
            64, 256, 2048, k_tile=256, n_tile=64, stationary="a", expected_uses=100, seed=3
        )
        inst_sa = extract_layout_instance(
            run_pre_layout_passes(stat_a.program), cost_model=model, reusable_tensors=stat_a.reusable_tensors
        )
        sa_loads = [op for op in inst_sa.operations if op.op_name == "tile.load"]
        assert [op.invocations for op in sa_loads] == [100, 3200]

        for k_dim in (512, 192, 64):
            with pytest.raises(ValueError, match="stationary='a' supports only one or two K tiles"):
                auto_layout_bench.build_single_matmul_case(64, k_dim, 64, k_tile=128, stationary="a")

    def test_prepare_variant_tensors_uses_entry_param_order_for_chain(self, tmp_path: Path):
        chain = auto_layout_bench.build_two_matmul_chain_case(64, 128, 64, 64, seed=7)
        # Verify mixed in-layouts: only `w2` is NZ while `x` and `w1` are ND.
        # Entry signature is `(x, w1, w2, out)`; previously mm_stage1's params caused sorted ['w1','w2','x']!
        dummy_bd = CostBreakdown(
            assignment=LayoutAssignment(value_layouts={}),
            total_cost_cycles=1.0,
            operator_cost_cycles=1.0,
            conversion_cost_cycles=0.0,
            per_op_costs={},
            per_value_conversion_costs={},
        )
        variant_w2_nz = LayoutVariantArtifact(
            variant_id="variant_w2_nz",
            breakdown=dummy_bd,
            build_dir=tmp_path,
            pto_files=(),
            gm_input_layouts={"x": "ND", "w1": "ND", "w2": "NZ"},
            gm_output_layouts={"out": "ND"},
            gm_layouts={"x": "ND", "w1": "ND", "w2": "NZ", "mid": "ND", "out": "ND"},
        )
        tensors, out_buf = auto_layout_bench.prepare_variant_tensors(chain, variant_w2_nz)
        assert len(tensors) == 3
        assert torch.equal(tensors[0], chain.inputs[0])
        assert torch.equal(tensors[1], chain.inputs[1])
        assert torch.equal(unpack_nz(tensors[2]), chain.inputs[2])
        assert out_buf.shape == chain.output_shape

    def test_run_calibration_suite_isolates_benchmark_before_swimlane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cases = [
            auto_layout_bench.build_single_matmul_case(64, 128, 64, expected_uses=100, seed=0),
            auto_layout_bench.build_single_matmul_case(64, 256, 64, expected_uses=100, seed=1),
            auto_layout_bench.build_single_matmul_case(128, 128, 64, expected_uses=100, seed=2),
            auto_layout_bench.build_single_matmul_case(128, 128, 128, expected_uses=100, seed=3),
        ]
        # Skip ptoas during compilation while exercising the non-compile-only runtime orchestration loop.
        original_compile_variants = auto_layout_bench.compile_layout_variants

        def _compile_no_ptoas(*args, **kwargs):
            kwargs["skip_ptoas"] = True
            return original_compile_variants(*args, **kwargs)

        monkeypatch.setattr(auto_layout_bench, "compile_layout_variants", _compile_no_ptoas)

        expected_by_dir: dict[str, tuple[torch.Tensor, bool]] = {}
        call_events: list[str] = []

        def _make_fake_compiled(build_dir: Path, **kwargs):
            def _runner(*all_args, config=None):
                nonlocal expected_by_dir
                in_args = all_args[:-1]
                out_buf = all_args[-1]
                m_dim, k_dim = in_args[0].shape
                n_dim = in_args[1].shape[0]
                case_key = f"matmul_{m_dim}x{k_dim}x{n_dim}"
                # Match case_dir lookup from variant build_dir parent.
                case = next(c for c in cases if c.case_id == case_key)
                out_buf.copy_(
                    pack_nz(case.expected_output)
                    if expected_by_dir[str(build_dir)][1]
                    else case.expected_output
                )
                if getattr(config, "enable_chip_swimlane", 0):
                    call_events.append("swimlane")
                    dfx = Path(build_dir) / "dfx_outputs"
                    dfx.mkdir(parents=True, exist_ok=True)
                    (dfx / "chip_swimlane_records.json").write_text(
                        json.dumps([{"ph": "X", "dur": 4.25 + 0.01 * (m_dim + n_dim)}])
                    )
                else:
                    call_events.append("accuracy")

            return _runner

        def _fake_from_dir(build_dir: Path, **kwargs):
            manifest = json.loads((Path(build_dir).parent / "manifest.json").read_text())
            v_entry = next(v for v in manifest["variants"] if v["build_dir"] == str(build_dir))
            is_out_nz = v_entry["gm_output_layouts"].get("out") == "NZ"
            expected_by_dir[str(build_dir)] = (torch.empty(0), is_out_nz)
            return _make_fake_compiled(build_dir, **kwargs)

        def _fake_benchmark(compiled, all_args, *, rounds=100, warmup=3, **kwargs):
            # Must not pass platform= or device_id= alongside config= (benchmark() rejects both).
            assert "platform" not in kwargs
            assert "device_id" not in kwargs
            assert kwargs.get("config") is not None
            call_events.append("benchmark")
            base_us = 12.0 + 0.05 * all_args[-1].shape[0]
            return SimpleNamespace(
                device_us_median=base_us,
                device_us_min=base_us - 0.5,
                device_us_mean=base_us,
                device_us_max=base_us + 0.5,
                device_us_stdev=0.2,
                all_zero_device=False,
                invocations=[SimpleNamespace(effective_us=base_us - 1.0)],
            )

        monkeypatch.setattr(auto_layout_bench.CompiledProgram, "from_dir", staticmethod(_fake_from_dir))
        monkeypatch.setattr(auto_layout_bench, "benchmark", _fake_benchmark)

        summary = auto_layout_bench.run_calibration_suite(
            cases,
            output_dir=tmp_path / "full_mock_suite",
            platform="a2a3",
            rounds=10,
            warmup=2,
            compile_only=False,
            capture_swimlane=True,
        )

        # Strictly every benchmark() finishes before the first enable_chip_swimlane=1 run.
        first_swimlane = call_events.index("swimlane")
        last_benchmark = max(i for i, ev in enumerate(call_events) if ev == "benchmark")
        assert last_benchmark < first_swimlane
        assert call_events.count("benchmark") == 32
        assert call_events.count("swimlane") == 32
        assert summary["calibration"]["num_measurements"] == 32
        assert summary["calibration_effective"]["num_measurements"] == 32
        assert summary["calibration_swimlane"]["num_measurements"] == 32
        for c_rec in summary["cases"]:
            for v in c_rec["variants"]:
                assert v["accuracy_ok"] is True
                assert v["swimlane_aicore_us"] > 0.0

    def test_extract_swimlane_aicore_us_real_schemas_and_fallback_flag(self, tmp_path: Path):
        # 1. Real merged_swimlane_*.json with 2 AIC tasks (chain): must sum kernel-duration-us.
        d1 = tmp_path / "merged_case" / "dfx_outputs"
        d1.mkdir(parents=True)
        (d1 / "merged_swimlane_0.json").write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"ph": "X", "cat": "event", "args": {"kernel-duration-us": 3.20}},
                        {"ph": "X", "cat": "event", "args": {"kernel-duration-us": 3.26}},
                        {"ph": "X", "cat": "other", "args": {"kernel-duration-us": 99.0}},
                    ]
                }
            )
        )
        assert auto_layout_bench._extract_swimlane_aicore_us(tmp_path / "merged_case") == pytest.approx(6.46)

        # 2. Real chip_swimlane_records.json (`aicore_tasks` + `clock_freq_hz=50_000_000`).
        d2 = tmp_path / "records_case" / "dfx_outputs"
        d2.mkdir(parents=True)
        (d2 / "chip_swimlane_records.json").write_text(
            json.dumps(
                {
                    "chip_swimlane_level": 1,
                    "metadata": {"clock_freq_hz": 50_000_000},
                    "aicore_tasks": [
                        [0, 1, 1, 1000, 1169, 94, 112],
                    ],
                }
            )
        )
        assert auto_layout_bench._extract_swimlane_aicore_us(tmp_path / "records_case") == pytest.approx(
            169.0 * 1e6 / 50_000_000
        )

        # 3. Negative-slope measurements surface fallback_params instead of silent defaults.
        set_backend_type(BackendType.Ascend910)
        base_model = LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910)
        neg_meas: list[CalibrationMeasurement] = []
        for s_idx, (m_dim, k_dim, n_dim) in enumerate(
            [(64, 128, 64), (64, 256, 64), (128, 128, 64), (128, 128, 128)]
        ):
            case = auto_layout_bench.build_single_matmul_case(
                m_dim, k_dim, n_dim, expected_uses=100, seed=s_idx
            )
            inst = extract_layout_instance(run_pre_layout_passes(case.program), cost_model=base_model)
            for v_idx, bd in enumerate(enumerate_assignments(inst, deduplicate=True)):
                load_b = sum(
                    float(inst.values[op.output].byte_size)
                    for op in inst.operations
                    if op.op_name == "tile.load"
                )
                store_b = sum(
                    float(inst.values[op.inputs[0]].byte_size)
                    for op in inst.operations
                    if op.op_name == "tile.store"
                )
                # Deliberately invert sign on load/store bytes so all 4 bandwidth slopes <= 0.
                measured_us = 100.0 - 1e-4 * (load_b + store_b)
                neg_meas.append(
                    CalibrationMeasurement(
                        variant_id=f"neg_{s_idx}_{v_idx}",
                        measured_us=measured_us,
                        instance=inst,
                        breakdown=bd,
                    )
                )
        report = fit_cost_model_from_measurements(neg_meas, base_model)
        assert set(report.fallback_params) == {
            "mte2_nd2nz_bytes_per_cycle",
            "mte2_nz2nz_bytes_per_cycle",
            "fixpipe_acc_to_nd_bytes_per_cycle",
            "fixpipe_acc_to_nz_bytes_per_cycle",
        }
        assert report.r2 < 0.99

    def test_fit_cost_model_normalizes_expected_uses_and_flags_overlapped_freq(self):
        set_backend_type(BackendType.Ascend910)
        base_model = LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910)
        nominal_us_per_cyc = 1.0 / (1.8 * 1000.0)
        true_nd2nz_bpc = 75.0
        true_nz2nz_bpc = 120.0
        base_transfer_model = replace(
            base_model,
            mte2_nd2nz_row_penalty_cycles=0.0,
            mte2_nd2nz_sub512b_penalty_cycles=0.0,
            fixpipe_drain_fixed_cycles=0.0,
            fixpipe_odd_n1_penalty_cycles=0.0,
        )
        transfer_model = replace(
            base_transfer_model,
            mte2_nd2nz_bytes_per_cycle=true_nd2nz_bpc,
            mte2_nd2nz_contiguous_bytes_per_cycle=(
                true_nd2nz_bpc
                * base_model.mte2_nd2nz_contiguous_bytes_per_cycle
                / base_model.mte2_nd2nz_bytes_per_cycle
            ),
            mte2_nz2nz_bytes_per_cycle=true_nz2nz_bpc,
        )
        meas: list[CalibrationMeasurement] = []
        for s_idx, (m_dim, k_dim, n_dim, kt, nt) in enumerate(
            [
                (64, 2048, 64, 256, None),
                (64, 8192, 64, 256, None),
                (64, 256, 2048, 256, 64),
                (128, 1024, 1024, 128, 128),
            ]
        ):
            case = auto_layout_bench.build_single_matmul_case(
                m_dim, k_dim, n_dim, k_tile=kt, n_tile=nt, expected_uses=100, seed=s_idx
            )
            inst = extract_layout_instance(run_pre_layout_passes(case.program), cost_model=base_model)
            for v_idx, bd in enumerate(enumerate_assignments(inst, deduplicate=True)):
                run_scale = float(inst.cost_model.expected_uses)
                transfer_cyc = other_cyc = 0.0
                for op in inst.operations:
                    cfg = bd.assignment.op_configs[op.op_id]
                    inv = float(op.invocations) / run_scale
                    if op.op_name == "tile.load":
                        src = inst.values[op.inputs[0]]
                        dst = inst.values[op.output]
                        if dst.memory_space == ir.MemorySpace.Mat:
                            scale = base_model.effective_mte2_load_multiplier(src, dst, inv)
                            load_args = (src, dst, cfg.input_layouts[0], cfg.output_layout)
                            transfer_cyc += inv * scale * transfer_model.estimate_load_cost(*load_args)
                            other_cyc += (
                                inv
                                * scale
                                * (
                                    base_model.estimate_load_cost(*load_args)
                                    - base_transfer_model.estimate_load_cost(*load_args)
                                )
                            )
                            continue
                    elif op.op_name == "tile.store":
                        store_args = (inst.values[op.inputs[0]], inst.values[op.output], cfg.output_layout)
                        other_cyc += inv * (
                            base_model.estimate_store_cost(*store_args)
                            - base_transfer_model.estimate_store_cost(*store_args)
                        )
                        continue
                    other_cyc += inv * cfg.op_cost_cycles

                # Simulate 2-stage pipelined execution (`~3.2 GHz` raw slope = 0.5625x nominal
                # period, matching the 910B2 124-variant device fit) and ~0 FIXPIPE delta.
                measured_us = 16.0 + nominal_us_per_cyc * (transfer_cyc + 0.5625 * other_cyc)
                meas.append(
                    CalibrationMeasurement(
                        variant_id=f"case_{s_idx}/v_{v_idx}",
                        measured_us=measured_us,
                        instance=inst,
                        breakdown=bd,
                    )
                )

        report = fit_cost_model_from_measurements(meas, base_model)
        assert report.effective_freq_fallback is True
        assert report.raw_freq_ghz == pytest.approx(3.2, rel=1e-2)
        assert report.effective_freq_ghz == pytest.approx(1.8)
        assert report.within_case_r2 > 0.9
        assert report.within_case_op_spearman_rho > 0.9
        assert report.fitted_model.mte2_nd2nz_bytes_per_cycle == pytest.approx(true_nd2nz_bpc, rel=1e-2)
        assert report.fitted_model.mte2_nz2nz_bytes_per_cycle == pytest.approx(true_nz2nz_bpc, rel=1e-2)
        assert set(report.fallback_params) == {
            "fixpipe_acc_to_nd_bytes_per_cycle",
            "fixpipe_acc_to_nz_bytes_per_cycle",
        }


def test_differential_anchoring_and_short_row_strided_classification():
    from pypto.ir.auto_layout import ROW_MAJOR_FLAT, _calibration_features  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)
    nd = ir.TensorLayout.ND
    nz = ir.TensorLayout.NZ
    # Item 6: cols < parent_cols must classify as strided even when parent_row_bytes <= l2_cache_line_bytes.
    base_model = LayoutCostModel(
        backend_type=BackendType.Ascend910,
        mte2_nd2nz_bytes_per_cycle=77.0,
        mte2_nz2nz_bytes_per_cycle=102.0,
    )
    src = LayoutValueNode(
        name="x",
        memory_space=ir.MemorySpace.DDR,
        shape=(16, 128),  # 128 * 2B = 256B <= 512B L2 cache line
        dtype=DataType.FP16,
        current_layout=nd,
        admissible_layouts=(nd, nz),
        producer_op_id=None,
    )
    dst = LayoutValueNode(
        name="tx",
        memory_space=ir.MemorySpace.Mat,
        shape=(16, 64),  # cols < parent_cols
        dtype=DataType.FP16,
        current_layout=ROW_MAJOR_FLAT,
        admissible_layouts=(ROW_MAJOR_FLAT,),
        producer_op_id="load_0",
    )
    op = LayoutOperationNode(
        op_id="load_0",
        op_name="tile.load",
        inputs=("x",),
        output="tx",
        configs=(
            OperatorConfig(
                op_id="load_0",
                op_name="tile.load",
                input_layouts=(nd,),
                output_layout=ROW_MAJOR_FLAT,
                op_cost_cycles=1.0,
            ),
        ),
        invocations=100,
    )
    inst = LayoutProblemInstance(
        values={"x": src, "tx": dst},
        operations=[op],
        cost_model=base_model,
    )
    bd = evaluate_assignment(
        inst,
        LayoutAssignment(value_layouts={"x": nd, "tx": ROW_MAJOR_FLAT}, op_configs={"load_0": op.configs[0]}),
    )
    meas_item = CalibrationMeasurement(
        variant_id="short_row_strided",
        measured_us=1.0,
        instance=inst,
        breakdown=bd,
    )
    feats_strided = _calibration_features(base_model, meas_item, 100.0, 2.0)
    assert feats_strided[1] == pytest.approx(float(dst.byte_size))

    # Item 5: synthetic differential anchoring with r > 1 contiguous loads recovers exact bandwidths.
    base_model_cal = LayoutCostModel(
        expected_uses=100,
        backend_type=BackendType.Ascend910,
        mte2_nd2nz_bytes_per_cycle=50.0,
        mte2_nd2nz_contiguous_bytes_per_cycle=100.0,
    )
    base_transfer = replace(
        base_model_cal,
        mte2_nd2nz_row_penalty_cycles=0.0,
        mte2_nd2nz_sub512b_penalty_cycles=0.0,
        fixpipe_drain_fixed_cycles=0.0,
        fixpipe_odd_n1_penalty_cycles=0.0,
    )
    true_transfer = replace(
        base_transfer,
        mte2_nd2nz_bytes_per_cycle=77.0,
        mte2_nd2nz_contiguous_bytes_per_cycle=154.0,
        mte2_nz2nz_bytes_per_cycle=102.0,
    )
    nominal_us_per_cyc = 1.0 / (1.8 * 1000.0)
    meas: list[CalibrationMeasurement] = []
    for s_idx, (m_dim, k_dim, n_dim, kt, nt) in enumerate(
        [
            (64, 2048, 64, 256, None),
            (64, 8192, 64, 256, None),
            (64, 256, 2048, 256, 64),
            (128, 1024, 1024, 128, 128),
        ]
    ):
        case = auto_layout_bench.build_single_matmul_case(
            m_dim, k_dim, n_dim, k_tile=kt, n_tile=nt, expected_uses=100, seed=s_idx
        )
        c_inst = extract_layout_instance(run_pre_layout_passes(case.program), cost_model=base_model_cal)
        for v_idx, c_bd in enumerate(enumerate_assignments(c_inst, deduplicate=True)):
            run_scale = float(c_inst.cost_model.expected_uses)
            transfer_cyc = other_cyc = 0.0
            for c_op in c_inst.operations:
                cfg = c_bd.assignment.op_configs[c_op.op_id]
                inv = float(c_op.invocations) / run_scale
                if c_op.op_name == "tile.load":
                    c_src = c_inst.values[c_op.inputs[0]]
                    c_dst = c_inst.values[c_op.output]
                    if c_dst.memory_space == ir.MemorySpace.Mat:
                        sc = base_model_cal.effective_mte2_load_multiplier(c_src, c_dst, inv)
                        l_args = (c_src, c_dst, cfg.input_layouts[0], cfg.output_layout)
                        transfer_cyc += inv * sc * true_transfer.estimate_load_cost(*l_args)
                        other_cyc += (
                            inv
                            * sc
                            * (
                                base_model_cal.estimate_load_cost(*l_args)
                                - base_transfer.estimate_load_cost(*l_args)
                            )
                        )
                        continue
                elif c_op.op_name == "tile.store":
                    s_args = (c_inst.values[c_op.inputs[0]], c_inst.values[c_op.output], cfg.output_layout)
                    other_cyc += inv * (
                        base_model_cal.estimate_store_cost(*s_args)
                        - base_transfer.estimate_store_cost(*s_args)
                    )
                    continue
                other_cyc += inv * cfg.op_cost_cycles
            meas.append(
                CalibrationMeasurement(
                    variant_id=f"c_{s_idx}/v_{v_idx}",
                    measured_us=10.0 + nominal_us_per_cyc * (transfer_cyc + other_cyc),
                    instance=c_inst,
                    breakdown=c_bd,
                )
            )
    rep = fit_cost_model_from_measurements(meas, base_model_cal)
    assert rep.fitted_model.mte2_nd2nz_bytes_per_cycle == pytest.approx(77.0, rel=1e-2)
    assert rep.fitted_model.mte2_nz2nz_bytes_per_cycle == pytest.approx(102.0, rel=1e-2)


def test_bench_metrics_and_variant_matching_fixes():
    from unittest import mock  # noqa: PLC0415

    from pypto.ir.auto_layout import ROW_MAJOR_FLAT, _calibration_prediction_cycles  # noqa: PLC0415
    from pypto.ir.auto_layout_bench import _evaluate_solvers_for_case, _match_variant_record  # noqa: PLC0415
    from pypto.ir.auto_layout_production_bench import _extract_bench_metrics  # noqa: PLC0415
    from pypto.ir.auto_layout_solver import (  # noqa: PLC0415
        LayoutSolverResult,
        LayoutSolverStats,
    )

    nd = ir.TensorLayout.ND
    nz = ir.TensorLayout.NZ

    class _FakeStats:
        def per_round(self, metric: str) -> list[float]:
            return {
                "device": [10.0, 12.0, 14.0],
                "effective": [11.0, 13.0, 15.0],
                "host": [20.0, 22.0, 24.0],
            }[metric]

    assert _extract_bench_metrics(_FakeStats()) == (12.0, 13.0, 22.0)

    # Item 8: value_layouts disambiguates identical gm_layouts; writeback_issues blocks match.
    recs = [
        {
            "variant_id": "v_nd_tile",
            "gm_layouts": {"x": "ND"},
            "value_layouts": {"x": "ND", "t": "RM"},
            "device_us_median": 50.0,
        },
        {
            "variant_id": "v_nz_tile",
            "gm_layouts": {"x": "ND"},
            "value_layouts": {"x": "ND", "t": "NZ"},
            "device_us_median": 25.0,
        },
    ]
    matched = _match_variant_record(recs, {"x": "ND"}, {"x": "ND", "t": "NZ"})
    assert matched is not None
    assert matched["variant_id"] == "v_nz_tile"
    assert _match_variant_record(recs, {"x": "ND"}, {"x": "ND", "t": "CM"}) is None

    x_node = LayoutValueNode(
        name="x",
        memory_space=ir.MemorySpace.DDR,
        shape=(16, 16),
        dtype=DataType.FP16,
        current_layout=nd,
        admissible_layouts=(nd,),
        producer_op_id=None,
    )
    inst = LayoutProblemInstance(values={"x": x_node}, operations=[], cost_model=LayoutCostModel())
    bad_res = LayoutSolverResult(
        breakdown=evaluate_assignment(inst, LayoutAssignment(value_layouts={"x": nd})),
        stats=LayoutSolverStats(1, 0, 0, 1.0, "optimal", 0, 0),
        conversions=(),
        writeback_issues=("incompatible helper signature",),
    )

    with (
        mock.patch("pypto.ir.auto_layout_bench.solve_greedy", return_value=bad_res),
        mock.patch("pypto.ir.auto_layout_bench.solve_treewidth_dp", return_value=bad_res),
        mock.patch("pypto.ir.auto_layout_bench.solve_auto_layout", return_value=bad_res),
    ):
        summary = _evaluate_solvers_for_case(
            inst,
            [{"variant_id": "v_nd", "gm_layouts": {"x": "ND"}, "device_us_median": 10.0}],
        )
    assert summary["auto_layout"]["variant_id"] is None
    assert summary["auto_layout"]["device_us_median"] is None
    assert summary["auto_layout"]["writeback_issues"] == ["incompatible helper signature"]

    # Item 5: paired collinear within-case measurements (x_store = 2 * x_load) with estimable clock.
    col_model = LayoutCostModel(
        expected_uses=1,
        backend_type=BackendType.Ascend910,
        mte2_nd2nz_bytes_per_cycle=77.0,
        mte2_nd2nz_contiguous_bytes_per_cycle=77.0,
    )
    col_meas: list[CalibrationMeasurement] = []
    for c_idx, comp_cyc in enumerate([100.0, 200.0, 300.0]):
        x_val = LayoutValueNode(
            name="x",
            shape=(16, 16),
            dtype=DataType.FP16,
            memory_space=ir.MemorySpace.DDR,
            current_layout=nd,
            admissible_layouts=(nd, nz),
            producer_op_id=None,
        )
        tx_val = LayoutValueNode(
            name="tx",
            shape=(16, 16),
            dtype=DataType.FP16,
            memory_space=ir.MemorySpace.Mat,
            current_layout=ROW_MAJOR_FLAT,
            admissible_layouts=(ROW_MAJOR_FLAT,),
            producer_op_id="ld",
        )
        tacc_val = LayoutValueNode(
            name="ta",
            shape=(16, 16),
            dtype=DataType.FP32,
            memory_space=ir.MemorySpace.Acc,
            current_layout=ROW_MAJOR_FLAT,
            admissible_layouts=(ROW_MAJOR_FLAT,),
            producer_op_id="mm",
        )
        y_val = LayoutValueNode(
            name="y",
            shape=(16, 16),
            dtype=DataType.FP32,
            memory_space=ir.MemorySpace.DDR,
            current_layout=nd,
            admissible_layouts=(nd, nz),
            producer_op_id="st",
            is_external_output=True,
        )
        ld_op = LayoutOperationNode(
            op_id="ld",
            op_name="tile.load",
            inputs=("x",),
            output="tx",
            configs=(
                OperatorConfig(
                    "ld",
                    "tile.load",
                    (nd,),
                    ROW_MAJOR_FLAT,
                    col_model.estimate_load_cost(x_val, tx_val, nd, ROW_MAJOR_FLAT),
                ),
                OperatorConfig(
                    "ld",
                    "tile.load",
                    (nz,),
                    ROW_MAJOR_FLAT,
                    col_model.estimate_load_cost(x_val, tx_val, nz, ROW_MAJOR_FLAT),
                ),
            ),
        )
        mm_op = LayoutOperationNode(
            op_id="mm",
            op_name="compute",
            inputs=("tx",),
            output="ta",
            configs=(OperatorConfig("mm", "compute", (ROW_MAJOR_FLAT,), ROW_MAJOR_FLAT, comp_cyc),),
        )
        st_op = LayoutOperationNode(
            op_id="st",
            op_name="tile.store",
            inputs=("ta",),
            output="y",
            configs=(
                OperatorConfig(
                    "st",
                    "tile.store",
                    (ROW_MAJOR_FLAT,),
                    nd,
                    col_model.estimate_store_cost(tacc_val, y_val, nd),
                ),
                OperatorConfig(
                    "st",
                    "tile.store",
                    (ROW_MAJOR_FLAT,),
                    nz,
                    col_model.estimate_store_cost(tacc_val, y_val, nz),
                ),
            ),
        )
        c_inst = LayoutProblemInstance(
            values={"x": x_val, "tx": tx_val, "ta": tacc_val, "y": y_val},
            operations=[ld_op, mm_op, st_op],
            cost_model=col_model,
        )
        for v_idx, gm_l in enumerate([nd, nz]):
            bd = evaluate_assignment(
                c_inst,
                LayoutAssignment(
                    value_layouts={"x": gm_l, "tx": ROW_MAJOR_FLAT, "ta": ROW_MAJOR_FLAT, "y": gm_l},
                    op_configs={
                        "ld": ld_op.configs[v_idx],
                        "mm": mm_op.configs[0],
                        "st": st_op.configs[v_idx],
                    },
                ),
            )
            stub_m = CalibrationMeasurement(f"col_{c_idx}/v_{v_idx}", 0.0, c_inst, bd)
            cyc = _calibration_prediction_cycles(col_model, stub_m)
            col_meas.append(
                CalibrationMeasurement(
                    variant_id=f"col_{c_idx}/v_{v_idx}",
                    measured_us=10.0 + cyc / 1800.0,
                    instance=c_inst,
                    breakdown=bd,
                )
            )
    col_rep = fit_cost_model_from_measurements(col_meas, col_model)
    assert col_rep.effective_freq_fallback is False
    assert col_rep.param_t_stats["fixpipe_acc_to_nd_bytes_per_cycle"] == 0.0
    assert "fixpipe_acc_to_nd_bytes_per_cycle" in col_rep.fallback_params
    assert col_rep.fitted_model.fixpipe_acc_to_nd_bytes_per_cycle == pytest.approx(
        col_model.fixpipe_acc_to_nd_bytes_per_cycle
    )

    # Singleton cases must not inflate within-case residual degrees of freedom.
    sing_meas: list[CalibrationMeasurement] = []
    for c_idx, comp_cyc in enumerate([100.0, 200.0, 300.0, 400.0, 500.0]):
        ta_v = LayoutValueNode(
            name="ta",
            shape=(64, 16),
            dtype=DataType.FP32,
            memory_space=ir.MemorySpace.Acc,
            current_layout=ROW_MAJOR_FLAT,
            admissible_layouts=(ROW_MAJOR_FLAT,),
            producer_op_id="mm",
        )
        y_v = LayoutValueNode(
            name="y",
            shape=(64, 16),
            dtype=DataType.FP32,
            memory_space=ir.MemorySpace.DDR,
            current_layout=nd,
            admissible_layouts=(nd, nz),
            producer_op_id="st",
            is_external_output=True,
        )
        mm_s = LayoutOperationNode(
            op_id="mm",
            op_name="compute",
            inputs=(),
            output="ta",
            configs=(OperatorConfig("mm", "compute", (), ROW_MAJOR_FLAT, comp_cyc),),
        )
        st_s = LayoutOperationNode(
            op_id="st",
            op_name="tile.store",
            inputs=("ta",),
            output="y",
            configs=(
                OperatorConfig(
                    "st",
                    "tile.store",
                    (ROW_MAJOR_FLAT,),
                    nd,
                    col_model.estimate_store_cost(ta_v, y_v, nd),
                ),
                OperatorConfig(
                    "st",
                    "tile.store",
                    (ROW_MAJOR_FLAT,),
                    nz,
                    col_model.estimate_store_cost(ta_v, y_v, nz),
                ),
            ),
        )
        s_inst = LayoutProblemInstance(
            values={"ta": ta_v, "y": y_v}, operations=[mm_s, st_s], cost_model=col_model
        )
        variants = [(0, nd), (1, nz)] if c_idx == 0 else [(1, nz)]
        for cfg_i, gm_l in variants:
            bd = evaluate_assignment(
                s_inst,
                LayoutAssignment(
                    value_layouts={"ta": ROW_MAJOR_FLAT, "y": gm_l},
                    op_configs={"mm": mm_s.configs[0], "st": st_s.configs[cfg_i]},
                ),
            )
            stub_s = CalibrationMeasurement(f"s_{c_idx}/v_{cfg_i}", 0.0, s_inst, bd)
            cyc_s = _calibration_prediction_cycles(col_model, stub_s)
            perturb = 0.01 if (c_idx == 0 and gm_l == nd) else 0.0
            sing_meas.append(
                CalibrationMeasurement(
                    variant_id=f"s_{c_idx}/v_{cfg_i}",
                    measured_us=100.0 + cyc_s / 1800.0 + perturb,
                    instance=s_inst,
                    breakdown=bd,
                )
            )
    sing_rep = fit_cost_model_from_measurements(sing_meas, col_model)
    assert sing_rep.param_t_stats["fixpipe_acc_to_nd_bytes_per_cycle"] == 0.0
    assert "fixpipe_acc_to_nd_bytes_per_cycle" in sing_rep.fallback_params
    assert sing_rep.fitted_model.fixpipe_acc_to_nd_bytes_per_cycle == pytest.approx(
        col_model.fixpipe_acc_to_nd_bytes_per_cycle
    )


def test_device_sweep_l3_host_tuple_sync_and_rank5_reshape_fixes():
    from pypto.ir.auto_layout import _LayoutAssignmentWriter, _tensor_type_layout  # noqa: PLC0415

    nd = ir.TensorLayout.ND
    nz = ir.TensorLayout.NZ

    # 1) _sync_call_return_type preserves caller dynamic DimExpr shapes on TupleType while updating layout.
    span = ir.Span.unknown()
    c16 = ir.ConstInt(16, DataType.INDEX, span)
    caller_dim = ir.Var("caller_t", ir.ScalarType(DataType.INDEX), span)
    callee_dim = ir.Var("callee_t", ir.ScalarType(DataType.INDEX), span)
    nd_view = ir.TensorView([], ir.TensorLayout.ND)
    nz_view = ir.TensorView([], ir.TensorLayout.NZ)
    caller_t = ir.TensorType([caller_dim, c16], DataType.FP32, None, nd_view)
    callee_t = ir.TensorType([callee_dim, c16], DataType.FP32, None, nz_view)
    call_tuple = ir.TupleType([caller_t, ir.ScalarType(DataType.INT64)])
    synced_tuple = _LayoutAssignmentWriter._sync_call_return_type(call_tuple, [callee_t])
    assert isinstance(synced_tuple, ir.TupleType)
    synced_elem0 = synced_tuple.types[0]
    assert isinstance(synced_elem0, ir.TensorType)
    assert synced_elem0.tensor_view is not None and synced_elem0.tensor_view.layout == ir.TensorLayout.NZ
    assert synced_elem0.shape[0].same_as(caller_dim)

    # 2) Pre-blocked rank-5 NZ GM weight keeps admissible_layouts == (NZ,) even when preceded by a
    #    1D flat ND reshape alias passed to a prefetch helper (`prefetch_o_proj_w` before `proj_a_mm`).
    empty_body = ir.SeqStmts([], span)
    wo_a_orch = ir.Var(
        "wo_a__ssa_v0",
        ir.TensorType([1, 16, 8, 16, 16], DataType.FP16, None, nz_view),
        span,
    )
    wo_a_flat_var = ir.Var(
        "wo_a_flat__ssa_v0",
        ir.TensorType([32768], DataType.FP16, None, nd_view),
        span,
    )
    reshape_call = ir.create_op_call(
        "tensor.reshape",
        [wo_a_orch, ir.MakeTuple([ir.ConstInt(32768, DataType.INDEX, span)], span)],
        {},
        span,
    )
    prefetch_param = ir.Var(
        "wo_a_flat__ssa_v0",
        ir.TensorType([32768], DataType.FP16, None, nd_view),
        span,
    )
    fn_prefetch = ir.Function(
        "prefetch_o_proj_w",
        [prefetch_param],
        [],
        empty_body,
        span,
        type=ir.FunctionType.AIV,
        level=ir.Level.AIV,
    )
    wo_a_rank5_nz = ir.Var(
        "wo_a__ssa_v0",
        ir.TensorType([1, 16, 8, 16, 16], DataType.FP16, None, nz_view),
        span,
    )
    fn_rank5 = ir.Function(
        "proj_a_mm",
        [wo_a_rank5_nz],
        [],
        empty_body,
        span,
        type=ir.FunctionType.AIC,
        level=ir.Level.AIC,
    )
    orch_body = ir.SeqStmts(
        [
            ir.AssignStmt(wo_a_flat_var, reshape_call, span),
            ir.EvalStmt(
                ir.Call(
                    ir.GlobalVar("prefetch_o_proj_w"), [wo_a_flat_var], {}, None, ir.VoidType.get(), span
                ),
                span,
            ),
            ir.EvalStmt(
                ir.Call(ir.GlobalVar("proj_a_mm"), [wo_a_orch], {}, None, ir.VoidType.get(), span),
                span,
            ),
        ],
        span,
    )
    fn_orch = ir.Function(
        "decode_csa_orch",
        [wo_a_orch],
        [],
        orch_body,
        span,
        type=ir.FunctionType.Orchestration,
        level=ir.Level.CHIP,
    )
    mod_rank5 = ir.Program([fn_orch, fn_prefetch, fn_rank5], "mod_rank5", span)
    inst_rank5 = extract_layout_instance(mod_rank5)
    assert inst_rank5.values["wo_a"].current_layout == nz
    assert inst_rank5.values["wo_a"].admissible_layouts == (nz,)
    assert inst_rank5.values["wo_a_flat"].current_layout == nd
    assert inst_rank5.values["wo_a_flat"].admissible_layouts == (nd,)
    rewritten_rank5 = apply_layout_assignment(
        mod_rank5,
        LayoutAssignment(
            value_layouts={"wo_a": nz, "wo_a_flat": nd},
            op_configs={},
        ),
    )
    r5_fns = {fn.name: fn for fn in rewritten_rank5.functions.values()}
    proj_a_t = r5_fns["proj_a_mm"].params[0].type
    prefetch_t = r5_fns["prefetch_o_proj_w"].params[0].type
    assert isinstance(proj_a_t, ir.TensorType) and _tensor_type_layout(proj_a_t) == nz
    assert isinstance(prefetch_t, ir.TensorType) and _tensor_type_layout(prefetch_t) == nd

    # 3) L3 HOST orchestrator functions keep logical ND parameter layouts when AutoLayout writes back NZ.
    w_host = ir.Var("wq_a", ir.TensorType([2, 64, 64], DataType.FP16, None, nd_view), span)
    fn_host = ir.Function(
        "l3_host",
        [w_host],
        [],
        empty_body,
        span,
        level=ir.Level.HOST,
    )
    w_chip = ir.Var("wq_a", ir.TensorType([64, 64], DataType.FP16, None, nd_view), span)
    fn_chip = ir.Function(
        "l2_chip",
        [w_chip],
        [],
        empty_body,
        span,
        type=ir.FunctionType.Orchestration,
        level=ir.Level.CHIP,
    )
    mod_dist = ir.Program([fn_host, fn_chip], "mod_dist", span)
    assign_dist = LayoutAssignment(value_layouts={"wq_a": nz}, op_configs={})
    rewritten_dist = apply_layout_assignment(mod_dist, assign_dist)
    fns_by_name = {fn.name: fn for fn in rewritten_dist.functions.values()}
    host_param_t = fns_by_name["l3_host"].params[0].type
    chip_param_t = fns_by_name["l2_chip"].params[0].type
    assert isinstance(host_param_t, ir.TensorType) and _tensor_type_layout(host_param_t) == ir.TensorLayout.ND
    assert isinstance(chip_param_t, ir.TensorType) and _tensor_type_layout(chip_param_t) == ir.TensorLayout.NZ

    # 4) Full two-blocking-pass pipeline: Pass 10 BlockNzTensorViews -> AutoLayout tile rewrite ->
    #    Pass 22.5 BlockNzTensorViews must NOT double-block an already-blocked NZ parameter.
    from pypto.pypto_core import passes as core_passes  # noqa: PLC0415

    wo_a_3d = ir.Var(
        "wo_a__ssa_v0",
        ir.TensorType([1, 256, 128], DataType.FP16, None, nz_view),
        span,
    )
    t_src = ir.Var(
        "t_src",
        ir.TileType(
            [16, 16],
            DataType.FP16,
            None,
            ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major, fractal=512),
            ir.MemorySpace.Mat,
        ),
        span,
    )
    move_call = ir.create_op_call(
        "tile.move",
        [t_src],
        {"target_memory": ir.MemorySpace.Left},
        span,
    )
    t_dst = ir.Var("t_dst", move_call.type, span)
    fn_pre_nz = ir.Function(
        "proj_a_mm",
        [wo_a_3d, t_src],
        [],
        ir.AssignStmt(t_dst, move_call, span),
        span,
        type=ir.FunctionType.AIC,
        level=ir.Level.AIC,
    )
    wq_a_2d = ir.Var("wq_a__ssa_v0", ir.TensorType([64, 64], DataType.FP16, None, nd_view), span)
    fn_other = ir.Function(
        "other_mm",
        [wq_a_2d],
        [],
        empty_body,
        span,
        type=ir.FunctionType.AIC,
        level=ir.Level.AIC,
    )
    prog_two_pass = ir.Program([fn_pre_nz, fn_other], "prog_two_pass", span)
    set_backend_type(BackendType.Ascend910)
    block_pipe = core_passes.PassPipeline()
    block_pipe.add_pass(core_passes.block_nz_tensor_views())
    pass1_prog = block_pipe.run(prog_two_pass)
    p1_fns = {fn.name: fn for fn in pass1_prog.functions.values()}
    assert "nz_tensor_views_blocked" in p1_fns["proj_a_mm"].attrs
    p1_t0 = p1_fns["proj_a_mm"].params[0].type
    assert isinstance(p1_t0, ir.TensorType)
    assert [int(d.value) for d in p1_t0.shape if isinstance(d, ir.ConstInt)] == [1, 8, 16, 16, 16]

    rewritten_two_pass = apply_layout_assignment(
        pass1_prog,
        LayoutAssignment(
            value_layouts={
                "wo_a": nz,
                "wq_a": nz,
                "proj_a_mm.t_dst": FRACTAL_ZN,
            },
            op_configs={},
        ),
    )
    pass2_prog = block_pipe.run(rewritten_two_pass)
    p2_fns = {fn.name: fn for fn in pass2_prog.functions.values()}
    p2_t0 = p2_fns["proj_a_mm"].params[0].type
    p2_other_t0 = p2_fns["other_mm"].params[0].type
    assert isinstance(p2_t0, ir.TensorType) and isinstance(p2_other_t0, ir.TensorType)
    assert [int(d.value) for d in p2_t0.shape if isinstance(d, ir.ConstInt)] == [1, 8, 16, 16, 16]
    assert [int(d.value) for d in p2_other_t0.shape if isinstance(d, ir.ConstInt)] == [1, 4, 4, 16, 16]

    # 5) Vec tile.store uses MTE3 bandwidth (no Fixpipe odd_n1 penalty), and reductions price by max
    #    input/output elements.
    from pypto.ir.auto_layout import ROW_MAJOR_FLAT, _calibration_op_cost  # noqa: PLC0415

    cm = LayoutCostModel(mte3_vec_bytes_per_cycle=200.0, vec_elements_per_cycle=128.0)
    vec_src = LayoutValueNode(
        name="v_out",
        shape=(16, 48),  # odd_n1 = 3 if treated as Fixpipe
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.Vec,
        current_layout=ROW_MAJOR_FLAT,
        admissible_layouts=(ROW_MAJOR_FLAT,),
    )
    gm_dst = LayoutValueNode(
        name="y_gm",
        shape=(16, 48),
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.DDR,
        current_layout=nd,
        admissible_layouts=(nd,),
    )
    assert cm.estimate_store_cost(vec_src, gm_dst, nd) == pytest.approx(float(vec_src.byte_size) / 200.0)

    red_in = LayoutValueNode(
        name="x_ub",
        shape=(64, 128),
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.Vec,
        current_layout=ROW_MAJOR_FLAT,
        admissible_layouts=(ROW_MAJOR_FLAT,),
    )
    red_out = LayoutValueNode(
        name="s_ub",
        shape=(64, 1),
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.Vec,
        current_layout=ROW_MAJOR_FLAT,
        admissible_layouts=(ROW_MAJOR_FLAT,),
        producer_op_id="red",
    )
    red_cfg = OperatorConfig("red", "tile.row_sum", (ROW_MAJOR_FLAT,), ROW_MAJOR_FLAT, 0.0)
    red_op = LayoutOperationNode("red", "tile.row_sum", ("x_ub",), "s_ub", configs=(red_cfg,))
    red_inst = LayoutProblemInstance(
        values={"x_ub": red_in, "s_ub": red_out},
        operations=[red_op],
        cost_model=cm,
    )
    assert _calibration_op_cost(cm, red_inst, red_op, red_cfg) == pytest.approx(8192.0 / 128.0)


def _run_resolve_layouts_pass(prog: ir.Program) -> ir.Program:
    pipe = passes.PassPipeline()
    pipe.add_pass(passes.resolve_backend_op_layouts())
    return pipe.run(prog)


def test_pack_unpack_dn_roundtrip():
    """Verify pack_dn / unpack_dn preserve 2D and 3D logical tensors and reject invalid ranks."""
    t2 = torch.arange(24, dtype=torch.float16).reshape(4, 6)
    p2 = pack_dn(t2)
    assert p2.shape == (4, 6)
    assert p2.is_contiguous()
    assert torch.equal(unpack_dn(p2), t2)

    t3 = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)
    p3 = pack_dn(t3)
    assert p3.shape == (2, 4, 6)
    assert p3.is_contiguous()
    assert torch.equal(unpack_dn(p3), t3)

    with pytest.raises(ValueError, match="rank >= 2"):
        pack_dn(torch.zeros((4,)))
    with pytest.raises(ValueError, match="rank >= 2"):
        unpack_dn(torch.zeros((4,)))


def test_direction_a_reduction_elementwise_broadcast_writeback_and_pto():
    """Direction A: eliminate [M, 1] <-> [1, M] ping-pong and share ROW_BCAST_FLAT across consumers."""
    from pypto.ir.auto_layout_solver import solve_auto_layout  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)

    @pl.jit
    def vec_softmax_style(
        x: pl.Tensor[[32, 32], pl.FP32],
        q0: pl.Tensor[[32, 32], pl.FP32],
        q1: pl.Tensor[[32, 32], pl.FP32],
        q2: pl.Tensor[[32, 32], pl.FP32],
        out0: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        out1: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        out2: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tuple[
        pl.Tensor[[32, 32], pl.FP32],
        pl.Tensor[[32, 32], pl.FP32],
        pl.Tensor[[32, 32], pl.FP32],
    ]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_softmax"):
            tx = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq0 = pl.load(q0, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq1 = pl.load(q1, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq2 = pl.load(q2, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tmp = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            rsum = pl.tile.row_sum(tx, tmp)
            # Chain of row-major scalar ops on [32, 1]: would ping-pong without Direction A.
            scaled = pl.tile.muls(rsum, 0.5)
            shifted = pl.tile.adds(scaled, 1.0)
            y0 = pl.tile.row_expand_add(tq0, shifted)
            y1 = pl.tile.row_expand_add(tq1, shifted)
            y2 = pl.tile.row_expand_add(tq2, shifted)
            out0 = pl.store(y0, [0, 0], out0)
            out1 = pl.store(y1, [0, 0], out1)
            out2 = pl.store(y2, [0, 0], out2)
        return out0, out1, out2

    stub = torch.zeros((32, 32), dtype=torch.float32)
    prog = vec_softmax_style.specialize(stub, stub, stub, stub, stub, stub, stub)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        pre = run_pre_layout_passes(prog)
        inst = extract_layout_instance(pre, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910))

        # Both [32, 1] nodes should expose COL_MAJOR_FLAT, ROW_MAJOR_FLAT, and ROW_BCAST_FLAT.
        shifted_node = next(v for k, v in inst.values.items() if k.endswith("shifted"))
        assert COL_MAJOR_FLAT in shifted_node.admissible_layouts
        assert ROW_MAJOR_FLAT in shifted_node.admissible_layouts
        assert ROW_BCAST_FLAT in shifted_node.admissible_layouts

        res = solve_auto_layout(inst)
        assert res.writeback_issues == ()
        # All 3 row_expand_add consumers should share ROW_BCAST_FLAT to amortize vbrcb.
        re_adds = [op for op in inst.operations if op.op_name == "tile.row_expand_add"]
        assert len(re_adds) == 3
        for op in re_adds:
            assert res.assignment.op_configs[op.op_id].input_layouts[1] == ROW_BCAST_FLAT

        rewritten = apply_layout_assignment(pre, res.writeback_assignment())
        ir_text = str(rewritten)
        # Only 2 reshapes total: one entry ([32, 1] -> [1, 32]) + one exit ([1, 32] -> [32, 1]),
        # plus one shared tile.row_expand instead of 3 per-consumer vbrcb broadcasts.
        assert ir_text.count("tile.reshape") == 2
        assert ir_text.count("tile.row_expand(") == 1

        # ResolveBackendOpLayouts must be a no-op after AutoLayout writeback.
        after_p24 = _run_resolve_layouts_pass(rewritten)
        ir.assert_structural_equal(after_p24, rewritten)

        lowered = run_post_layout_passes(rewritten)
        pto_text = _emit_pto_from_lowered(lowered)
        assert "pto.trowexpand" in pto_text

    # Single consumer (K=1): stays COL_MAJOR_FLAT without materializing ROW_BCAST_FLAT.
    @pl.jit
    def vec_single_row_expand(
        x: pl.Tensor[[32, 32], pl.FP32],
        q0: pl.Tensor[[32, 32], pl.FP32],
        out0: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_k1"):
            tx = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq0 = pl.load(q0, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tmp = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            rsum = pl.tile.row_sum(tx, tmp)
            y0 = pl.tile.row_expand_add(tq0, rsum)
            out0 = pl.store(y0, [0, 0], out0)
        return out0

    k1_prog = vec_single_row_expand.specialize(stub, stub, stub)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        k1_pre = run_pre_layout_passes(k1_prog)
        k1_inst = extract_layout_instance(
            k1_pre, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910)
        )
        k1_res = solve_auto_layout(k1_inst)
        assert k1_res.writeback_issues == ()
        k1_re = next(op for op in k1_inst.operations if op.op_name == "tile.row_expand_add")
        assert k1_res.assignment.op_configs[k1_re.op_id].input_layouts[1] == COL_MAJOR_FLAT
        k1_rewritten = apply_layout_assignment(k1_pre, k1_res.writeback_assignment())
        assert "tile.row_expand(" not in str(k1_rewritten)
        ir.assert_structural_equal(_run_resolve_layouts_pass(k1_rewritten), k1_rewritten)

    # User-authored tile.reshape(rsum, [1, 32]) keeps exactly 1 reshape + col_sum/col_expand_add chain.
    @pl.jit
    def vec_user_reshape_and_col_chain(
        x: pl.Tensor[[32, 32], pl.FP32],
        q0: pl.Tensor[[32, 32], pl.FP32],
        out_vec: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
        out_col: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tuple[pl.Tensor[[32, 1], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_user_rm"):
            tx = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq0 = pl.load(q0, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tmp_r = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            rsum = pl.tile.row_sum(tx, tmp_r)
            r_rm = pl.tile.reshape(rsum, [1, 32])
            r_scaled = pl.tile.muls(r_rm, 0.5)
            r_back = pl.tile.reshape(r_scaled, [32, 1])
            out_vec = pl.store(r_back, [0, 0], out_vec)
            tmp_c = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            csum = pl.tile.col_sum(tx, tmp_c)
            y_col = pl.tile.col_expand_add(tq0, csum)
            out_col = pl.store(y_col, [0, 0], out_col)
        return out_vec, out_col

    vec_out_stub = torch.zeros((32, 1), dtype=torch.float32)
    ur_prog = vec_user_reshape_and_col_chain.specialize(stub, stub, vec_out_stub, stub)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        ur_pre = run_pre_layout_passes(ur_prog)
        ur_inst = extract_layout_instance(
            ur_pre, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910)
        )
        ur_res = solve_auto_layout(ur_inst)
        assert ur_res.writeback_issues == ()
        assert ur_res.breakdown.conversion_cost_cycles == pytest.approx(0.0)
        ur_rewritten = apply_layout_assignment(ur_pre, ur_res.writeback_assignment())
        assert str(ur_rewritten).count("tile.reshape") == 2
        ir.assert_structural_equal(_run_resolve_layouts_pass(ur_rewritten), ur_rewritten)
        _emit_pto_from_lowered(run_post_layout_passes(ur_rewritten))


def test_direction_b_dn_gm_and_col_major_vec_writeback_and_pto():
    """Direction B: DN GM + COL_MAJOR_FLAT Vec load/store beats strided ND MTE2/MTE3 and lowers cleanly."""
    from pypto.ir.auto_layout_solver import solve_auto_layout  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)

    @pl.program
    class _DnVecProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def vec_copy(
            self,
            x: pl.Tensor[[64, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            for j in pl.range(0, 8, 1):
                tx = pl.load(x, [0, j * 16], [64, 16], target_memory=pl.Mem.Vec)
                out = pl.store(tx, [0, j * 16], out)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def vec_consume(
            self,
            mid: pl.Tensor[[64, 128], pl.FP32],
            y: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            tm = pl.load(mid, [0, 0], [64, 16], target_memory=pl.Mem.Vec)
            ts = pl.tile.muls(tm, 2.0)
            y = pl.store(ts, [0, 0], y)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            x: pl.Tensor[[64, 128], pl.FP32],
            y: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            with pl.manual_scope():
                mid = pl.tensor.create([64, 128], dtype=pl.FP32)
                mid, t0 = pl.submit(self.vec_copy, x, mid)
                y, _t1 = pl.submit(self.vec_consume, mid, y, deps=[t0])
            return y

    cm = LayoutCostModel(
        expected_uses=100,
        backend_type=BackendType.Ascend910,
        allow_external_input_relayout=True,
    )
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        pre = run_pre_layout_passes(_DnVecProgram)
        inst = extract_layout_instance(pre, cost_model=cm, reusable_tensors={"x"})
        assert ir.TensorLayout.DN in inst.values["x"].admissible_layouts
        assert ir.TensorLayout.DN in inst.values["mid"].admissible_layouts

        res = solve_auto_layout(inst)
        assert res.writeback_issues == ()
        # Reading/writing [64, 16] columns out of [64, 128] is contiguous under DN+COL_MAJOR_FLAT.
        assert res.assignment.tensor_layouts["x"] == ir.TensorLayout.DN
        assert res.assignment.tensor_layouts["mid"] == ir.TensorLayout.DN
        assert res.assignment.tile_layouts["vec_copy.tx"] == COL_MAJOR_FLAT
        assert res.assignment.tile_layouts["vec_consume.tm"] == COL_MAJOR_FLAT
        assert res.assignment.tile_layouts["vec_consume.ts"] == ROW_MAJOR_FLAT

        rewritten = apply_layout_assignment(pre, res.writeback_assignment())
        ir_text = str(rewritten)
        assert "DN" in ir_text
        assert "col_major" in ir_text
        assert "tile.move" in ir_text

        after_p24 = _run_resolve_layouts_pass(rewritten)
        ir.assert_structural_equal(after_p24, rewritten)

        lowered = run_post_layout_passes(rewritten)
        pto_text = _emit_pto_from_lowered(lowered)
        assert "pto.tload" in pto_text
        assert "pto.tstore" in pto_text
        assert "pto.tmov" in pto_text

    # Wide row slice [8, 128] out of [64, 128] is contiguous under ND and chooses ND + ROW_MAJOR_FLAT.
    @pl.jit
    def wide_row_slice_vec(
        x: pl.Tensor[[64, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 128], pl.FP32]],
    ) -> pl.Tensor[[8, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="wide_row_slice"):
            tx = pl.load(x, [0, 0], [8, 128], target_memory=pl.Mem.Vec)
            ts = pl.tile.muls(tx, 2.0)
            out = pl.store(ts, [0, 0], out)
        return out

    wide_prog = wide_row_slice_vec.specialize(
        torch.zeros((64, 128), dtype=torch.float32),
        torch.zeros((8, 128), dtype=torch.float32),
    )
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        wide_pre = run_pre_layout_passes(wide_prog)
        wide_inst = extract_layout_instance(wide_pre, cost_model=cm, reusable_tensors={"x"})
        wide_res = solve_auto_layout(wide_inst)
        assert wide_res.writeback_issues == ()
        assert wide_res.assignment.tensor_layouts["x"] == ir.TensorLayout.ND
        assert wide_res.assignment.tile_layouts["tx"] == ROW_MAJOR_FLAT

    # (i) Implicit col_major on [32, 1] load + muls + store keeps ResolveBackendOpLayouts a no-op.
    @pl.jit
    def col_vec_muls_store(
        x: pl.Tensor[[32, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_vec_muls"):
            vx = pl.load(x, [0, 0], [32, 1], target_memory=pl.Mem.Vec)
            vs = pl.tile.muls(vx, 2.0)
            out = pl.store(vs, [0, 0], out)
        return out

    col_vec_stub = torch.zeros((32, 1), dtype=torch.float32)
    col_prog = col_vec_muls_store.specialize(col_vec_stub, col_vec_stub)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        col_pre = run_pre_layout_passes(col_prog)
        col_inst = extract_layout_instance(
            col_pre, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910)
        )
        assert col_inst.values["vx"].current_layout == COL_MAJOR_FLAT
        col_res = solve_auto_layout(col_inst)
        assert col_res.writeback_issues == ()
        col_rw = apply_layout_assignment(col_pre, col_res.writeback_assignment())
        ir.assert_structural_equal(_run_resolve_layouts_pass(col_rw), col_rw)

    # (ii) [1, 32] row-vector load + muls + store on [16, 32] never reshapes to col_major [32, 1].
    @pl.jit
    def row_vec_muls_store(
        x: pl.Tensor[[16, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="row_vec_muls"):
            tx = pl.load(x, [0, 0], [1, 32], target_memory=pl.Mem.Vec)
            ts = pl.tile.muls(tx, 2.0)
            out = pl.store(ts, [0, 0], out)
        return out

    row_mat_stub = torch.zeros((16, 32), dtype=torch.float32)
    row_prog = row_vec_muls_store.specialize(row_mat_stub, row_mat_stub)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        row_pre = run_pre_layout_passes(row_prog)
        row_inst = extract_layout_instance(
            row_pre,
            cost_model=LayoutCostModel(
                backend_type=BackendType.Ascend910, allow_external_input_relayout=True
            ),
            reusable_tensors={"x"},
        )
        assert ir.TensorLayout.DN not in row_inst.values["x"].admissible_layouts
        assert ir.TensorLayout.DN not in row_inst.values["out"].admissible_layouts
        variants = enumerate_assignments(row_inst)
        assert len(variants) >= 1
        for bd in variants:
            rw = apply_layout_assignment(row_pre, bd.assignment)
            ir.assert_structural_equal(_run_resolve_layouts_pass(rw), rw)
            run_post_layout_passes(rw)


def test_direction_c_fractal_nz_vec_pointwise_between_cube_matmuls_and_pto():
    """Direction C: Pointwise Mem.Vec ops stay FRACTAL_NZ between Cube Mat/Acc boundaries."""
    from pypto.ir.auto_layout_solver import solve_auto_layout  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)

    @pl.program
    class _CubeVecCubeProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def mm_prod(
            self,
            a: pl.Tensor[[64, 64], pl.FP16],
            b: pl.Tensor[[64, 64], pl.FP16],
            mid_fp32: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            for _i in pl.range(0, 32, 1):
                ta = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                mid_fp32 = pl.store(pl.matmul(ta, tb), [0, 0], mid_fp32)
            return mid_fp32

        @pl.function(type=pl.FunctionType.InCore)
        def vec_nz_fp16(
            self,
            x: pl.Tensor[[64, 128], pl.FP16],
            scale: pl.Scalar[pl.FP16],
            out_fp16: pl.Out[pl.Tensor[[64, 128], pl.FP16]],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            tx = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            tzero = pl.tile.full([64, 64], pl.FP16, 0.0)
            texp = pl.tile.adds(tzero, scale)
            tact = pl.tile.relu(tx)
            tout16 = pl.tile.add(tact, texp)
            out_fp16 = pl.store(tout16, [0, 0], out_fp16)
            return out_fp16

        @pl.function(type=pl.FunctionType.InCore)
        def vec_nz_fp32(
            self,
            mid_in: pl.Tensor[[64, 128], pl.FP32],
            out_fp32: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            for _i in pl.range(0, 32, 1):
                tm = pl.load(mid_in, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
                tm_cast = pl.tile.cast(tm, pl.INT32)
                tm_back = pl.tile.cast(tm_cast, pl.FP32)
                tm_scaled = pl.tile.muls(tm_back, 0.5)
                out_fp32 = pl.store(tm_scaled, [0, 0], out_fp32)
            return out_fp32

        @pl.function(type=pl.FunctionType.InCore)
        def mm_cons(
            self,
            a: pl.Tensor[[64, 128], pl.FP16],
            w: pl.Tensor[[64, 64], pl.FP16],
            y: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            ta = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            tw = pl.load(w, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            y = pl.store(pl.matmul(ta, tw), [0, 0], y)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            a: pl.Tensor[[64, 64], pl.FP16],
            b: pl.Tensor[[64, 64], pl.FP16],
            x: pl.Tensor[[64, 128], pl.FP16],
            w: pl.Tensor[[64, 64], pl.FP16],
            scale: pl.Scalar[pl.FP16],
            out_fp32: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            y: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tuple[pl.Tensor[[64, 128], pl.FP32], pl.Tensor[[64, 64], pl.FP32]]:
            with pl.manual_scope():
                mid = pl.tensor.create([64, 128], dtype=pl.FP32)
                mid2 = pl.tensor.create([64, 128], dtype=pl.FP16)
                mid, t0 = pl.submit(self.mm_prod, a, b, mid)
                out_fp32, t1 = pl.submit(self.vec_nz_fp32, mid, out_fp32, deps=[t0])
                mid2, t2 = pl.submit(self.vec_nz_fp16, x, scale, mid2)
                y, _t3 = pl.submit(self.mm_cons, mid2, w, y, deps=[t1, t2])
            return out_fp32, y

    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        pre = run_pre_layout_passes(_CubeVecCubeProgram)
        cm = LayoutCostModel(
            expected_uses=100,
            backend_type=BackendType.Ascend910,
            allow_external_input_relayout=True,
        )
        inst = extract_layout_instance(pre, cost_model=cm, reusable_tensors={"a", "b", "x", "w"})

        # All eligible Vec tiles inside vec_nz_fp16 / vec_nz_fp32 must admit FRACTAL_NZ.
        for key in ("tx", "tzero", "texp", "tact", "tout16"):
            assert FRACTAL_NZ in inst.values[f"vec_nz_fp16.{key}"].admissible_layouts
        for key in ("tm", "tm_cast", "tm_back", "tm_scaled"):
            assert FRACTAL_NZ in inst.values[f"vec_nz_fp32.{key}"].admissible_layouts

        res = solve_auto_layout(inst)
        assert res.writeback_issues == ()
        assert res.assignment.tensor_layouts["mid"] == ir.TensorLayout.NZ
        assert res.assignment.tensor_layouts["mid2"] == ir.TensorLayout.NZ
        assert res.assignment.tile_layouts["vec_nz_fp32.tm_scaled"] == FRACTAL_NZ
        assert res.assignment.tile_layouts["vec_nz_fp16.tout16"] == FRACTAL_NZ

        rewritten = apply_layout_assignment(pre, res.writeback_assignment())
        fns = {fn.name: fn for fn in rewritten.functions.values()}
        assert "tensor.view" in str(fns["vec_nz_fp16"])
        assert "tensor.view" in str(fns["vec_nz_fp32"])

        after_p24 = _run_resolve_layouts_pass(rewritten)
        ir.assert_structural_equal(after_p24, rewritten)

        lowered = run_post_layout_passes(rewritten)
        pto_text = _emit_pto_from_lowered(lowered)
        assert "pto.trelu" in pto_text
        assert "pto.tadd" in pto_text
        assert "pto.tmuls" in pto_text

    # Expected_uses=1 + allow_external_input_relayout=False: internal mid1, mid2 stay NZ across tiled Vec!
    @pl.program
    class _CubeTiledVecCubeProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def mm_prod(
            self,
            a: pl.Tensor[[64, 64], pl.FP16],
            b: pl.Tensor[[64, 64], pl.FP16],
            mid1: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            for c in pl.range(0, 128, 64):
                ta = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                mid1 = pl.store(pl.matmul(ta, tb), [0, c], mid1)
            return mid1

        @pl.function(type=pl.FunctionType.InCore)
        def vec_nz_multi(
            self,
            m1: pl.Tensor[[64, 128], pl.FP32],
            x: pl.Tensor[[64, 64], pl.FP16],
            out1: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            out2: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
        ) -> tuple[pl.Tensor[[64, 128], pl.FP32], pl.Tensor[[64, 64], pl.FP16]]:
            for c in pl.range(0, 128, 32):
                t1 = pl.load(m1, [0, c], [64, 32], target_memory=pl.Mem.Vec)
                t1_s = pl.tile.muls(t1, 0.5)
                out1 = pl.store(t1_s, [0, c], out1)
            tx = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            tx_r = pl.tile.relu(tx)
            out2 = pl.store(tx_r, [0, 0], out2)
            return out1, out2

        @pl.function(type=pl.FunctionType.InCore)
        def vec_relu_twice(
            self,
            m1: pl.Tensor[[64, 128], pl.FP32],
            m2: pl.Tensor[[64, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            for _it in pl.range(0, 32, 1):
                t1 = pl.load(m1, [0, 0], [64, 32], target_memory=pl.Mem.Vec)
                t2 = pl.load(m2, [0, 0], [64, 32], target_memory=pl.Mem.Vec)
                s = pl.tile.add(pl.tile.relu(t1), t2)
                out = pl.store(s, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            a: pl.Tensor[[64, 64], pl.FP16],
            b: pl.Tensor[[64, 64], pl.FP16],
            x: pl.Tensor[[64, 64], pl.FP16],
            y: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            with pl.manual_scope():
                mid1 = pl.tensor.create([64, 128], dtype=pl.FP32)
                mid2 = pl.tensor.create([64, 128], dtype=pl.FP32)
                mid3 = pl.tensor.create([64, 64], dtype=pl.FP16)
                mid1, t0 = pl.submit(self.mm_prod, a, b, mid1)
                (mid2, mid3), t1 = pl.submit(self.vec_nz_multi, mid1, x, mid2, mid3, deps=[t0])
                y, _t2 = pl.submit(self.vec_relu_twice, mid1, mid2, y, deps=[t0, t1])
            return y

    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        c_pre = run_pre_layout_passes(_CubeTiledVecCubeProgram)
        c_cm = LayoutCostModel(
            expected_uses=1,
            backend_type=BackendType.Ascend910,
            allow_external_input_relayout=False,
        )
        c_inst = extract_layout_instance(c_pre, cost_model=c_cm)
        c_res = solve_auto_layout(c_inst)
        assert c_res.writeback_issues == ()
        # External a, b, x stay ND; internal Cube->Vec buffers mid1 & mid2 promote to NZ!
        assert c_res.assignment.tensor_layouts["a"] == ir.TensorLayout.ND
        assert c_res.assignment.tensor_layouts["x"] == ir.TensorLayout.ND
        assert c_res.assignment.tensor_layouts["mid1"] == ir.TensorLayout.NZ
        assert c_res.assignment.tensor_layouts["mid2"] == ir.TensorLayout.NZ
        c_rewritten = apply_layout_assignment(c_pre, c_res.writeback_assignment())
        c_fns = {fn.name: fn for fn in c_rewritten.functions.values()}
        # Multi-output return_types sync: vec_nz_multi returns (mid2=NZ, mid3=ND).
        ret_types = c_fns["vec_nz_multi"].return_types
        assert len(ret_types) == 2
        assert isinstance(ret_types[0], ir.TensorType) and ret_types[0].tensor_view is not None
        assert ret_types[0].tensor_view.layout == ir.TensorLayout.NZ
        assert isinstance(ret_types[1], ir.TensorType) and ret_types[1].tensor_view is None
        ir.assert_structural_equal(_run_resolve_layouts_pass(c_rewritten), c_rewritten)
        run_post_layout_passes(c_rewritten)

    # Negative gates: row_sum or unaligned [16, 8] FP16 excludes FRACTAL_NZ from the Mem.Vec component.
    @pl.jit
    def vec_ineligible_ops(
        x: pl.Tensor[[16, 8], pl.FP16],
        y: pl.Tensor[[32, 32], pl.FP16],
        out_x: pl.Out[pl.Tensor[[16, 8], pl.FP16]],
        out_y: pl.Out[pl.Tensor[[32, 1], pl.FP16]],
    ) -> pl.Tuple[pl.Tensor[[16, 8], pl.FP16], pl.Tensor[[32, 1], pl.FP16]]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_neg"):
            tx = pl.load(x, [0, 0], [16, 8], target_memory=pl.Mem.Vec)
            tx1 = pl.tile.relu(tx)
            out_x = pl.store(tx1, [0, 0], out_x)
            ty = pl.load(y, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            ty_act = pl.tile.relu(ty)
            tmp = pl.tile.create([32, 32], dtype=pl.FP16, target_memory=pl.Mem.Vec)
            ty_sum = pl.tile.row_sum(ty_act, tmp)
            out_y = pl.store(ty_sum, [0, 0], out_y)
        return out_x, out_y

    neg_prog = vec_ineligible_ops.specialize(
        torch.zeros((16, 8), dtype=torch.float16),
        torch.zeros((32, 32), dtype=torch.float16),
        torch.zeros((16, 8), dtype=torch.float16),
        torch.zeros((32, 1), dtype=torch.float16),
    )
    neg_pre = run_pre_layout_passes(neg_prog)
    neg_inst = extract_layout_instance(
        neg_pre,
        cost_model=LayoutCostModel(backend_type=BackendType.Ascend910, allow_external_input_relayout=True),
    )
    assert FRACTAL_NZ not in neg_inst.values["tx"].admissible_layouts
    assert FRACTAL_NZ not in neg_inst.values["tx1"].admissible_layouts
    assert FRACTAL_NZ not in neg_inst.values["ty"].admissible_layouts
    assert FRACTAL_NZ not in neg_inst.values["ty_act"].admissible_layouts

    # Verify `tile.expands` disqualifies FRACTAL_NZ on Vec tiles and NZ/DN promotion on GM tensors.
    @pl.jit
    def vec_expands_nz(
        x: pl.Tensor[[64, 128], pl.FP16],
        s: pl.Scalar[pl.FP16],
        out: pl.Out[pl.Tensor[[64, 128], pl.FP16]],
    ) -> pl.Tensor[[64, 128], pl.FP16]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_exp_nz"):
            tx = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            te = pl.tile.expands(tx, s)
            out = pl.store(te, [0, 0], out)
        return out

    exp_prog = vec_expands_nz.specialize(
        torch.zeros((64, 128), dtype=torch.float16),
        0.0,
        torch.zeros((64, 128), dtype=torch.float16),
    )
    exp_pre = run_pre_layout_passes(exp_prog)
    exp_inst = extract_layout_instance(
        exp_pre,
        cost_model=LayoutCostModel(backend_type=BackendType.Ascend910, allow_external_input_relayout=True),
        reusable_tensors={"x"},
    )
    assert FRACTAL_NZ not in exp_inst.values["te"].admissible_layouts
    assert ir.TensorLayout.NZ not in exp_inst.values["x"].admissible_layouts
    assert ir.TensorLayout.DN not in exp_inst.values["x"].admissible_layouts

    # K2 probe: identity tensor.view folded into `xs = x` propagates `_uid_to_name` cleanly.
    @pl.jit
    def folded_view_add(
        x: pl.Tensor[[64, 16], pl.FP16],
        w: pl.Tensor[[64, 16], pl.FP16],
        y: pl.Out[pl.Tensor[[64, 16], pl.FP16]],
    ) -> pl.Tensor[[64, 16], pl.FP16]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="folded_view_add"):
            xs = pl.tensor.view(x, layout=pl.ND)
            t1 = pl.load(xs, [0, 0], [64, 16], target_memory=pl.Mem.Vec)
            tw = pl.load(w, [0, 0], [64, 16], target_memory=pl.Mem.Vec)
            ts = pl.tile.add(t1, tw)
            y = pl.store(ts, [0, 0], y)
        return y

    k2_stub = torch.zeros((64, 16), dtype=torch.float16)
    k2_prog = folded_view_add.specialize(k2_stub, k2_stub, k2_stub)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        k2_pre = run_pre_layout_passes(k2_prog)
        k2_inst = extract_layout_instance(
            k2_pre,
            cost_model=LayoutCostModel(
                backend_type=BackendType.Ascend910, allow_external_input_relayout=True
            ),
            reusable_tensors={"x", "w"},
        )
        assert "t1" in k2_inst.values
        k2_variants = enumerate_assignments(k2_inst)
        assert len(k2_variants) >= 1
        for bd in k2_variants:
            k2_rw = apply_layout_assignment(k2_pre, bd.assignment)
            ir.assert_structural_equal(_run_resolve_layouts_pass(k2_rw), k2_rw)
            run_post_layout_passes(k2_rw)


def test_nested_if_while_yield_scoping_and_return_var_symmetry():  # noqa: PLR0915
    set_backend_type(BackendType.Ascend910)
    span = ir.Span.unknown()
    nd = ir.TensorLayout.ND
    nz = ir.TensorLayout.NZ

    gm_type = ir.TensorType([64, 64], DataType.FP16)
    bool_type = ir.ScalarType(DataType.BOOL)
    idx0 = ir.ConstInt(0, DataType.INDEX, span)
    idx1 = ir.ConstInt(1, DataType.INDEX, span)
    idx2 = ir.ConstInt(2, DataType.INDEX, span)
    idx16 = ir.ConstInt(16, DataType.INDEX, span)

    def _make_tile_type(
        space: ir.MemorySpace,
        blayout: ir.TileLayout = ir.TileLayout.row_major,
        slayout: ir.TileLayout = ir.TileLayout.none_box,
        fractal: int = 512,
        dtype: DataType = DataType.FP16,
    ) -> ir.TileType:
        return ir.TileType(
            [16, 16],
            dtype,
            None,
            ir.TileView([idx16, idx16], [], None, blayout, slayout, fractal),
            space,
        )

    vec_type = _make_tile_type(ir.MemorySpace.Vec)
    mat_type = _make_tile_type(ir.MemorySpace.Mat, ir.TileLayout.col_major, ir.TileLayout.row_major)
    left_type = _make_tile_type(ir.MemorySpace.Left, ir.TileLayout.col_major, ir.TileLayout.row_major)
    right_type = _make_tile_type(ir.MemorySpace.Right, ir.TileLayout.row_major, ir.TileLayout.col_major)
    acc_type = _make_tile_type(
        ir.MemorySpace.Acc, ir.TileLayout.col_major, ir.TileLayout.row_major, 1024, DataType.FP32
    )
    out_fp32_type = ir.TensorType([64, 64], DataType.FP32)

    x = ir.Var("x", gm_type, span)
    w = ir.Var("w", gm_type, span)
    v_in = ir.Var("v_in", gm_type, span)
    z = ir.Var("z", gm_type, span)
    y_acc = ir.Var("y_acc", out_fp32_type, span)
    y_vec = ir.Var("y_vec", gm_type, span)
    cond = ir.Var("cond", bool_type, span)
    flag = ir.Var("flag", bool_type, span)

    offs = ir.MakeTuple([idx0, idx0], span)
    win = ir.MakeTuple([idx16, idx16], span)
    t_v_init = ir.Var("t_v_init", vec_type, span)
    t_v_init_stmt = ir.AssignStmt(
        t_v_init,
        ir.create_op_call("tile.load", [v_in, offs, win, win], {"target_memory": ir.MemorySpace.Vec}, span),
        span,
    )

    for_gm = ir.IterArg("for_gm", gm_type, x, span)
    for_tile = ir.IterArg("for_tile", vec_type, t_v_init, span)

    while_gm = ir.IterArg("while_gm", gm_type, for_gm, span)
    while_tile = ir.IterArg("while_tile", vec_type, for_tile, span)
    w_gm_ret = ir.Var("w_gm_ret", gm_type, span)
    w_tile_ret = ir.Var("w_tile_ret", vec_type, span)
    while_stmt = ir.WhileStmt(
        cond,
        [while_gm, while_tile],
        ir.YieldStmt([while_gm, while_tile], span),
        [w_gm_ret, w_tile_ret],
        span,
    )

    t_then = ir.Var("t_then", vec_type, span)
    t_then_stmt = ir.AssignStmt(
        t_then,
        ir.create_op_call("tile.load", [z, offs, win, win], {"target_memory": ir.MemorySpace.Vec}, span),
        span,
    )
    t_else_raw = ir.Var("t_else_raw", vec_type, span)
    t_else_raw_stmt = ir.AssignStmt(
        t_else_raw,
        ir.create_op_call("tile.load", [z, offs, win, win], {"target_memory": ir.MemorySpace.Vec}, span),
        span,
    )
    t_else = ir.Var("t_else", vec_type, span)
    t_else_stmt = ir.AssignStmt(t_else, ir.create_op_call("tile.neg", [t_else_raw], {}, span), span)

    if_gm_ret = ir.Var("if_gm_ret", gm_type, span)
    if_tile_ret = ir.Var("if_tile_ret", vec_type, span)
    if_stmt = ir.IfStmt(
        flag,
        ir.SeqStmts([t_then_stmt, ir.YieldStmt([for_gm, t_then], span)], span),
        ir.SeqStmts([t_else_raw_stmt, t_else_stmt, ir.YieldStmt([for_gm, t_else], span)], span),
        [if_gm_ret, if_tile_ret],
        span,
    )

    t_mat_a = ir.Var("t_mat_a", mat_type, span)
    t_mat_a_stmt = ir.AssignStmt(
        t_mat_a,
        ir.create_op_call(
            "tile.load", [if_gm_ret, offs, win, win], {"target_memory": ir.MemorySpace.Mat}, span
        ),
        span,
    )
    t_mat_b = ir.Var("t_mat_b", mat_type, span)
    t_mat_b_stmt = ir.AssignStmt(
        t_mat_b,
        ir.create_op_call("tile.load", [w, offs, win, win], {"target_memory": ir.MemorySpace.Mat}, span),
        span,
    )
    t_l0a_kwargs: dict[str, Any] = {
        "target_memory": ir.MemorySpace.Left,
        "blayout": ir.TileLayout.col_major,
        "slayout": ir.TileLayout.row_major,
    }
    t_l0a = ir.Var("t_l0a", left_type, span)
    t_l0a_stmt = ir.AssignStmt(
        t_l0a,
        ir.create_op_call("tile.move", [t_mat_a], t_l0a_kwargs, span),
        span,
    )
    t_l0b_kwargs: dict[str, Any] = {
        "target_memory": ir.MemorySpace.Right,
        "blayout": ir.TileLayout.row_major,
        "slayout": ir.TileLayout.col_major,
    }
    t_l0b = ir.Var("t_l0b", right_type, span)
    t_l0b_stmt = ir.AssignStmt(
        t_l0b,
        ir.create_op_call("tile.move", [t_mat_b], t_l0b_kwargs, span),
        span,
    )
    t_acc = ir.Var("t_acc", acc_type, span)
    t_acc_stmt = ir.AssignStmt(t_acc, ir.create_op_call("tile.matmul", [t_l0a, t_l0b], {}, span), span)
    y_acc_loop = ir.Var("y_acc_loop", out_fp32_type, span)
    store_acc_stmt = ir.AssignStmt(
        y_acc_loop, ir.create_op_call("tile.store", [t_acc, offs, y_acc, win], {}, span), span
    )

    t_sum = ir.Var("t_sum", vec_type, span)
    t_sum_stmt = ir.AssignStmt(
        t_sum, ir.create_op_call("tile.add", [w_tile_ret, if_tile_ret], {}, span), span
    )
    f_gm_ret = ir.Var("f_gm_ret", gm_type, span)
    f_tile_ret = ir.Var("f_tile_ret", vec_type, span)
    for_stmt = ir.ForStmt(
        ir.Var("loop_i", ir.ScalarType(DataType.INDEX), span),
        idx0,
        idx2,
        idx1,
        [for_gm, for_tile],
        ir.SeqStmts(
            [
                while_stmt,
                if_stmt,
                t_mat_a_stmt,
                t_mat_b_stmt,
                t_l0a_stmt,
                t_l0b_stmt,
                t_acc_stmt,
                store_acc_stmt,
                t_sum_stmt,
                ir.YieldStmt([w_gm_ret, t_sum], span),
            ],
            span,
        ),
        [f_gm_ret, f_tile_ret],
        span,
    )

    y_vec_out = ir.Var("y_vec_out", gm_type, span)
    store_vec_stmt = ir.AssignStmt(
        y_vec_out, ir.create_op_call("tile.store", [f_tile_ret, offs, y_vec, win], {}, span), span
    )
    fn = ir.Function(
        "nested_cf_kernel",
        [
            (x, ir.ParamDirection.In),
            (w, ir.ParamDirection.In),
            (v_in, ir.ParamDirection.In),
            (z, ir.ParamDirection.In),
            (y_acc, ir.ParamDirection.Out),
            (y_vec, ir.ParamDirection.Out),
            (cond, ir.ParamDirection.In),
            (flag, ir.ParamDirection.In),
        ],
        [gm_type],
        ir.SeqStmts([t_v_init_stmt, for_stmt, store_vec_stmt, ir.ReturnStmt([y_vec_out], span)], span),
        span,
        type=ir.FunctionType.InCore,
    )
    prog = ir.Program([fn], "nested_cf_prog", span)

    # 1) Extraction must isolate inner IfStmt yields from outer ForStmt iter_args (`x`/`w` vs `v_in`/`z`).
    inst = extract_layout_instance(
        prog,
        cost_model=LayoutCostModel(backend_type=BackendType.Ascend910, allow_external_input_relayout=True),
        reusable_tensors={"x", "w", "y_acc"},
    )
    assert nz in inst.values["x"].admissible_layouts
    assert nz in inst.values["w"].admissible_layouts
    assert nz not in inst.values["z"].admissible_layouts

    # 2) Writeback with both GM (ND -> NZ) and Mem.Vec (ROW_MAJOR <-> COL_MAJOR) rewrites
    #    must update ForStmt/WhileStmt/IfStmt return_vars_ symmetrically without leaking
    #    outer ForStmt iter_arg layouts into inner IfStmt YieldStmts.
    assignment = LayoutAssignment(
        value_layouts={
            "x": nz,
            "w": nz,
            "v_in": nd,
            "z": nd,
            "y_acc": nz,
            "y_vec": nd,
            "t_v_init": COL_MAJOR_FLAT,
            "t_then": ROW_MAJOR_FLAT,
            "t_else_raw": ROW_MAJOR_FLAT,
            "t_else": ROW_MAJOR_FLAT,
            "t_mat_a": FRACTAL_NZ,
            "t_mat_b": FRACTAL_NZ,
            "t_l0a": FRACTAL_NZ,
            "t_l0b": FRACTAL_ZN,
            "t_acc": FRACTAL_ACC_NZ,
            "t_sum": ROW_MAJOR_FLAT,
        },
        op_configs={
            "op_0_t_v_init_load": OperatorConfig(
                "op_0_t_v_init_load", "tile.load", (nd,), COL_MAJOR_FLAT, 1.0
            ),
            "op_1_t_then_load": OperatorConfig("op_1_t_then_load", "tile.load", (nd,), ROW_MAJOR_FLAT, 1.0),
            "op_2_t_else_raw_load": OperatorConfig(
                "op_2_t_else_raw_load", "tile.load", (nd,), ROW_MAJOR_FLAT, 1.0
            ),
            "op_3_t_else_generic": OperatorConfig(
                "op_3_t_else_generic", "tile.neg", (ROW_MAJOR_FLAT,), ROW_MAJOR_FLAT, 1.0
            ),
            "op_4_t_mat_a_load": OperatorConfig("op_4_t_mat_a_load", "tile.load", (nz,), FRACTAL_NZ, 1.0),
            "op_5_t_mat_b_load": OperatorConfig("op_5_t_mat_b_load", "tile.load", (nz,), FRACTAL_NZ, 1.0),
            "op_6_t_l0a_move": OperatorConfig("op_6_t_l0a_move", "tile.move", (FRACTAL_NZ,), FRACTAL_NZ, 1.0),
            "op_7_t_l0b_move": OperatorConfig("op_7_t_l0b_move", "tile.move", (FRACTAL_NZ,), FRACTAL_ZN, 1.0),
            "op_8_t_acc_matmul": OperatorConfig(
                "op_8_t_acc_matmul", "tile.matmul", (FRACTAL_NZ, FRACTAL_ZN), FRACTAL_ACC_NZ, 1.0
            ),
            "op_9_y_acc_loop_store": OperatorConfig(
                "op_9_y_acc_loop_store", "tile.store", (FRACTAL_ACC_NZ, nz), nz, 1.0
            ),
            "op_10_t_sum_generic": OperatorConfig(
                "op_10_t_sum_generic", "tile.add", (COL_MAJOR_FLAT, ROW_MAJOR_FLAT), ROW_MAJOR_FLAT, 1.0
            ),
            "op_11_y_vec_out_store": OperatorConfig(
                "op_11_y_vec_out_store", "tile.store", (ROW_MAJOR_FLAT, nd), nd, 1.0
            ),
        },
    )
    rewritten = apply_layout_assignment(prog, assignment)
    rw_fn = next(fn for gv, fn in rewritten.functions.items() if gv.name == "nested_cf_kernel")
    assert isinstance(rw_fn.params[0].type, ir.TensorType) and rw_fn.params[0].type.tensor_view is not None
    assert rw_fn.params[0].type.tensor_view.layout == nz
    assert isinstance(rw_fn.params[1].type, ir.TensorType) and rw_fn.params[1].type.tensor_view is not None
    assert rw_fn.params[1].type.tensor_view.layout == nz
    assert isinstance(rw_fn.params[4].type, ir.TensorType) and rw_fn.params[4].type.tensor_view is not None
    assert rw_fn.params[4].type.tensor_view.layout == nz

    assert isinstance(rw_fn.body, ir.SeqStmts)
    rw_stmts = rw_fn.body.stmts
    rw_for = next(s for s in rw_stmts if isinstance(s, ir.ForStmt))
    assert (
        isinstance(rw_for.iter_args[0].type, ir.TensorType)
        and rw_for.iter_args[0].type.tensor_view is not None
    )
    assert rw_for.iter_args[0].type.tensor_view.layout == nz
    assert (
        isinstance(rw_for.iter_args[1].type, ir.TileType) and rw_for.iter_args[1].type.tile_view is not None
    )
    assert rw_for.iter_args[1].type.tile_view.blayout == ir.TileLayout.col_major
    assert rw_for.return_vars[0].type == rw_for.iter_args[0].type
    assert rw_for.return_vars[1].type == rw_for.iter_args[1].type

    assert isinstance(rw_for.body, ir.SeqStmts)
    rw_for_stmts = rw_for.body.stmts
    rw_while = next(s for s in rw_for_stmts if isinstance(s, ir.WhileStmt))
    assert (
        isinstance(rw_while.iter_args[0].type, ir.TensorType)
        and rw_while.iter_args[0].type.tensor_view is not None
    )
    assert rw_while.iter_args[0].type.tensor_view.layout == nz
    assert (
        isinstance(rw_while.iter_args[1].type, ir.TileType)
        and rw_while.iter_args[1].type.tile_view is not None
    )
    assert rw_while.iter_args[1].type.tile_view.blayout == ir.TileLayout.col_major
    assert rw_while.return_vars[0].type == rw_while.iter_args[0].type
    assert rw_while.return_vars[1].type == rw_while.iter_args[1].type

    rw_if = next(s for s in rw_for_stmts if isinstance(s, ir.IfStmt))
    assert (
        isinstance(rw_if.return_vars[0].type, ir.TensorType)
        and rw_if.return_vars[0].type.tensor_view is not None
    )
    assert rw_if.return_vars[0].type.tensor_view.layout == nz
    # Inner IfStmt branches must yield ROW_MAJOR_FLAT (`t_then` / `t_else`), not COL_MAJOR_FLAT.
    assert isinstance(rw_if.then_body, ir.SeqStmts)
    then_yield = rw_if.then_body.stmts[-1]
    assert isinstance(then_yield, ir.YieldStmt)
    then_yield_t = then_yield.value[1].type
    assert isinstance(then_yield_t, ir.TileType)
    assert then_yield_t.tile_view is None or then_yield_t.tile_view.blayout == ir.TileLayout.row_major
    assert rw_if.return_vars[1].type == then_yield.value[1].type

    verify_props = passes.IRPropertySet()
    verify_props.insert(passes.IRProperty.SSAForm)
    verify_props.insert(passes.IRProperty.AssignTypeSymmetry)
    verify_props.insert(passes.IRProperty.NormalizedStmtStructure)
    passes.verify_properties(
        verify_props,
        rewritten,
        "test_nested_if_while_yield_scoping_and_return_var_symmetry",
    )


def test_if_join_distinct_producers_while_cond_and_distributed_tensor_preservation():
    """Verify IfStmt branch joins, WhileStmt condition iter_arg remap, and DistributedTensorType safety."""
    from pypto.ir.auto_layout_solver import solve_auto_layout  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)

    # 1) Conditional IfStmt join with two distinct reusable load producers forces consistent branch layouts.
    @pl.jit
    def if_join_kernel(
        x: pl.Tensor[[16, 16], pl.FP16],
        y: pl.Tensor[[16, 16], pl.FP16],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
    ) -> pl.Tensor[[16, 16], pl.FP16]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="if_join_vec"):
            if flag > 0:
                t = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            else:
                t = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            out = pl.store(t, [0, 0], out)
        return out

    stub = torch.zeros((16, 16), dtype=torch.float16)
    prog_if = if_join_kernel.specialize(stub, stub, 1, stub)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        pre_if = run_pre_layout_passes(prog_if)
        inst_if = extract_layout_instance(
            pre_if,
            cost_model=LayoutCostModel(backend_type=BackendType.Ascend910, expected_uses=100),
            reusable_tensors={"x", "y"},
        )
        assert any(op.op_name == "control.if_join" for op in inst_if.operations)
        res_if = solve_auto_layout(inst_if)
        assert res_if.writeback_issues == ()
        rw_if_prog = apply_layout_assignment(pre_if, res_if.writeback_assignment())
        verify_props = passes.IRPropertySet()
        verify_props.insert(passes.IRProperty.SSAForm)
        verify_props.insert(passes.IRProperty.AssignTypeSymmetry)
        passes.verify_properties(verify_props, rw_if_prog, "if_join_distinct_producers")

    # 2) WhileStmt condition referencing an IterArg whose type changes during writeback stays in SSA form.
    span = ir.Span.unknown()
    gm_type = ir.TensorType([16, 16], DataType.FP16, None)
    mat_type = ir.TileType(
        [16, 16],
        DataType.FP16,
        None,
        ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major, fractal=512),
        ir.MemorySpace.Mat,
    )
    in_buf = ir.Var("in_buf", gm_type, span)
    out_buf = ir.Var("out_buf", gm_type, span)
    ia = ir.IterArg("loop_gm", gm_type, in_buf, span)
    # Condition references `ia` inside a Call (`tile.load`) so WhileStmt must substitute `ia` into condition_.
    cond_load = ir.create_op_call(
        "tile.load",
        [
            ia,
            ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span), ir.ConstInt(0, DataType.INDEX, span)], span),
            ir.MakeTuple(
                [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)], span
            ),
            ir.MakeTuple(
                [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)], span
            ),
        ],
        {"target_memory": ir.MemorySpace.Mat},
        span,
    )
    cond_expr = ir.ConstBool(False, span)
    # Force an explicit Var reference to `ia` inside the WhileStmt condition expression.
    zero_off = ir.ConstInt(0, DataType.INDEX, span)
    cond_var_ref = ir.Ne(
        ir.TupleGetItemExpr(
            ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span), ia], span),
            0,
            span,
        ),
        zero_off,
        DataType.BOOL,
        span,
    )
    body_t = ir.Var("body_t", mat_type, span)
    body_load = ir.AssignStmt(body_t, cond_load, span)
    stored_v = ir.Var("stored_v", gm_type, span)
    body_store = ir.AssignStmt(
        stored_v,
        ir.create_op_call(
            "tile.store",
            [
                body_t,
                ir.MakeTuple([zero_off, zero_off], span),
                out_buf,
                ir.MakeTuple(
                    [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)], span
                ),
            ],
            {},
            span,
        ),
        span,
    )
    rv = ir.Var("loop_rv", gm_type, span)
    ws = ir.WhileStmt(
        ir.And(cond_expr, cond_var_ref, DataType.BOOL, span),
        [ia],
        ir.SeqStmts([body_load, body_store, ir.YieldStmt([ia], span)], span),
        [rv],
        span,
    )
    fn_w = ir.Function(
        "while_cond_fn",
        [in_buf, out_buf],
        [gm_type],
        ir.SeqStmts([ws, ir.ReturnStmt([out_buf], span)], span),
        span,
        type=ir.FunctionType.AIC,
        level=ir.Level.AIC,
    )
    prog_w = ir.Program([fn_w], "prog_w", span)
    rw_w = apply_layout_assignment(
        prog_w,
        LayoutAssignment(value_layouts={"in_buf": ir.TensorLayout.NZ}, op_configs={}),
    )
    rw_w_fn = next(iter(rw_w.functions.values()))
    assert isinstance(rw_w_fn.body, ir.SeqStmts)
    rw_ws = next(s for s in rw_w_fn.body.stmts if isinstance(s, ir.WhileStmt))
    assert rw_ws.iter_args[0].name_hint == "loop_gm"
    # Verify WhileStmt condition references the new IterArg pointer, not the stale original `ia`.
    assert str(ia.type) != str(rw_ws.iter_args[0].type)
    assert rw_ws.iter_args[0] is not ia
    assert isinstance(rw_ws.condition, ir.And)
    assert isinstance(rw_ws.condition.right, ir.Ne)
    assert isinstance(rw_ws.condition.right.left, ir.TupleGetItemExpr)
    assert isinstance(rw_ws.condition.right.left.tuple, ir.MakeTuple)
    assert rw_ws.condition.right.left.tuple.elements[1] is rw_ws.iter_args[0]
    assert rw_ws.condition.right.left.tuple.elements[1] is not ia

    # 3) DistributedTensorType parameters keep subtype & pinned layout across extraction and writeback.
    dist_t = ir.DistributedTensorType([16, 16], DataType.FP16, None)
    d_in = ir.Var("d_in", dist_t, span)
    d_out = ir.Var("d_out", gm_type, span)
    d_tile = ir.Var("d_tile", mat_type, span)
    d_load = ir.AssignStmt(
        d_tile,
        ir.create_op_call(
            "tile.load",
            [
                d_in,
                ir.MakeTuple([zero_off, zero_off], span),
                ir.MakeTuple(
                    [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)], span
                ),
                ir.MakeTuple(
                    [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)], span
                ),
            ],
            {"target_memory": ir.MemorySpace.Mat},
            span,
        ),
        span,
    )
    d_ret = ir.Var("d_ret", gm_type, span)
    d_store = ir.AssignStmt(
        d_ret,
        ir.create_op_call(
            "tile.store",
            [
                d_tile,
                ir.MakeTuple([zero_off, zero_off], span),
                d_out,
                ir.MakeTuple(
                    [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)], span
                ),
            ],
            {},
            span,
        ),
        span,
    )
    fn_dist = ir.Function(
        "dist_kernel",
        [d_in, d_out],
        [gm_type],
        ir.SeqStmts([d_load, d_store, ir.ReturnStmt([d_ret], span)], span),
        span,
        type=ir.FunctionType.AIC,
        level=ir.Level.AIC,
    )
    prog_dist = ir.Program([fn_dist], "prog_dist", span)
    inst_dist = extract_layout_instance(
        prog_dist, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910)
    )
    assert "d_in" in inst_dist.values
    assert inst_dist.values["d_in"].admissible_layouts == (ir.TensorLayout.ND,)
    rw_dist = apply_layout_assignment(
        prog_dist, LayoutAssignment(value_layouts={"d_in": ir.TensorLayout.ND}, op_configs={})
    )
    rw_dist_fn = next(iter(rw_dist.functions.values()))
    assert isinstance(rw_dist_fn.params[0].type, ir.DistributedTensorType)


def test_calibration_prediction_includes_internal_onchip_conversions():
    """Verify calibration predictor and features count internal conversions, excluding GM packing."""
    from pypto.ir.auto_layout import (  # noqa: PLC0415
        COL_MAJOR_FLAT,
        ROW_BCAST_FLAT,
        ROW_MAJOR_FLAT,
        CalibrationMeasurement,
        _calibration_features,
        _calibration_prediction_cycles,
    )

    cm = LayoutCostModel(expected_uses=2)
    gm_in = LayoutValueNode(
        name="x",
        shape=(32, 32),
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.DDR,
        current_layout=ir.TensorLayout.ND,
        admissible_layouts=(ir.TensorLayout.ND, ir.TensorLayout.NZ),
        is_reusable_weight=True,
    )
    vec_val = LayoutValueNode(
        name="rsum",
        shape=(32, 1),
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.Vec,
        current_layout=COL_MAJOR_FLAT,
        admissible_layouts=(COL_MAJOR_FLAT, ROW_BCAST_FLAT),
        producer_op_id="op_prod",
    )
    vec_out = LayoutValueNode(
        name="y",
        shape=(32, 32),
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.Vec,
        current_layout=ROW_MAJOR_FLAT,
        admissible_layouts=(ROW_MAJOR_FLAT,),
        producer_op_id="op_cons",
    )
    op_prod = LayoutOperationNode(
        op_id="op_prod",
        op_name="tile.row_sum",
        inputs=("x",),
        output="rsum",
        invocations=2,
        configs=(OperatorConfig("op_prod", "tile.row_sum", (ir.TensorLayout.NZ,), COL_MAJOR_FLAT, 10.0),),
    )
    op_cons = LayoutOperationNode(
        op_id="op_cons",
        op_name="tile.row_expand_add",
        inputs=("rsum",),
        output="y",
        invocations=2,
        configs=(OperatorConfig("op_cons", "tile.row_expand_add", (ROW_BCAST_FLAT,), ROW_MAJOR_FLAT, 5.0),),
    )
    inst = LayoutProblemInstance(
        values={"x": gm_in, "rsum": vec_val, "y": vec_out},
        operations=[op_prod, op_cons],
        cost_model=cm,
    )
    bd = evaluate_assignment(
        inst, LayoutAssignment(value_layouts={"x": ir.TensorLayout.NZ, "rsum": COL_MAJOR_FLAT})
    )
    meas = CalibrationMeasurement(variant_id="v", measured_us=1.0, instance=inst, breakdown=bd)
    expected_op = (32.0 / cm.vec_elements_per_cycle) + (1024.0 / cm.vec_elements_per_cycle)
    expected_internal_conv = cm.estimate_conversion_cost(vec_val, COL_MAJOR_FLAT, ROW_BCAST_FLAT)
    pred = _calibration_prediction_cycles(cm, meas)
    feats = _calibration_features(cm, meas, expected_uses=2.0, contiguous_nd_ratio=1.0)
    assert pred == pytest.approx(expected_op + expected_internal_conv)
    assert feats[5] == pytest.approx(expected_op + expected_internal_conv)


def test_callee_specialization_remaps_attrs_and_symbolic_types():
    """Callee specialization substitutes scalar params inside attrs, param types, and return_type."""
    from pypto.pypto_core import passes  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)
    span = ir.Span.unknown()
    c64 = ir.ConstInt(64, DataType.INDEX, span)
    nd_view = ir.TensorView([], ir.TensorLayout.ND)
    nz_view = ir.TensorView([], ir.TensorLayout.NZ)

    callee_k = ir.Var("k_callee", ir.ScalarType(DataType.INDEX), span)
    callee_sym_ty = ir.TensorType([callee_k, c64], DataType.FP16, None, nd_view)
    callee_x = ir.Var("x_callee", callee_sym_ty, span)
    callee_fn = ir.Function(
        "sym_callee",
        [callee_k, callee_x],
        [callee_sym_ty],
        ir.SeqStmts([ir.ReturnStmt([callee_x], span)], span),
        span,
        type=ir.FunctionType.InCore,
        level=ir.Level.CHIP_DIE,
        attrs={"sym_k": callee_k},
    )

    orch_k = ir.Var("k_orch", ir.ScalarType(DataType.INDEX), span)
    orch_ty_nd = ir.TensorType([orch_k, c64], DataType.FP16, None, nd_view)
    orch_x_nd = ir.Var("x_nd__ssa_v0", orch_ty_nd, span)
    orch_x_nz = ir.Var("x_nz__ssa_v0", orch_ty_nd, span)
    orch_out0 = ir.Var("out0", orch_ty_nd, span)
    orch_out1 = ir.Var("out1", orch_ty_nd, span)
    callee_gv = ir.GlobalVar("sym_callee")
    orch_body = ir.SeqStmts(
        [
            ir.AssignStmt(
                orch_out0,
                ir.Call(callee_gv, [orch_k, orch_x_nd], {}, None, orch_ty_nd, span),
                span,
            ),
            ir.AssignStmt(
                orch_out1,
                ir.Call(callee_gv, [orch_k, orch_x_nz], {}, None, orch_ty_nd, span),
                span,
            ),
            ir.ReturnStmt([orch_out0], span),
        ],
        span,
    )
    orch_fn = ir.Function(
        "entry",
        [orch_k, orch_x_nd, orch_x_nz],
        [orch_ty_nd],
        orch_body,
        span,
        type=ir.FunctionType.Orchestration,
        level=ir.Level.CHIP,
    )
    prog = ir.Program([orch_fn, callee_fn], "sym_spec_prog", span)
    inst = extract_layout_instance(prog)
    assignment = LayoutAssignment(
        value_layouts={
            "x_nd": ir.TensorLayout.ND,
            "x_nz": ir.TensorLayout.NZ,
            "x_callee": ir.TensorLayout.ND,
        }
    )
    rewritten = apply_layout_assignment(prog, assignment)
    spec_fns = [f for f in rewritten.functions.values() if f.name not in ("entry", "sym_callee")]
    assert len(spec_fns) == 1
    spec_fn = spec_fns[0]
    new_k = spec_fn.params[0]
    assert not new_k.same_as(callee_k)
    assert isinstance(spec_fn.attrs["sym_k"], ir.Expr) and spec_fn.attrs["sym_k"].same_as(new_k)
    spec_p1_ty = spec_fn.params[1].type
    assert isinstance(spec_p1_ty, ir.TensorType)
    assert spec_p1_ty.tensor_view is not None and spec_p1_ty.tensor_view.layout == nz_view.layout
    assert spec_p1_ty.shape[0].same_as(new_k)
    assert len(spec_fn.return_types) == 1
    assert isinstance(spec_fn.return_types[0], ir.TensorType)
    assert spec_fn.return_types[0].shape[0].same_as(new_k)
    ssa_props = passes.IRPropertySet()
    ssa_props.insert(passes.IRProperty.SSAForm)
    passes.verify_properties(ssa_props, rewritten, "callee_spec_ssa")
    assert inst is not None


def test_if_join_col_major_and_transpose_nd_preservation():
    """Verify IfStmt branch-yield conversions via control.if_join and tensor.transpose ND pinning."""
    from pypto.ir.auto_layout_solver import solve_auto_layout  # noqa: PLC0415
    from pypto.pypto_core import passes  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)

    # 1) tensor.transpose on an external ND input in Orchestration does not alias source/dest or emit NZ/DN.
    @pl.program
    class _TransposeOrchProg:
        @pl.function(type=pl.FunctionType.InCore)
        def vec_copy(
            self,
            inp: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            t = pl.load(inp, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            t2 = pl.tile.muls(t, 2.0)
            out = pl.store(t2, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            out0: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
            out1: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
        ) -> tuple[pl.Tensor[[64, 64], pl.FP16], pl.Tensor[[64, 64], pl.FP16]]:
            xt = pl.transpose(x, 0, 1)
            out0 = self.vec_copy(x, out0)
            out1 = self.vec_copy(xt, out1)
            return out0, out1

    pre_tr = run_pre_layout_passes(_TransposeOrchProg)
    inst_tr = extract_layout_instance(pre_tr, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910))
    assert inst_tr.values["x"].admissible_layouts == (ir.TensorLayout.ND,)
    assert inst_tr.values["xt"].admissible_layouts == (ir.TensorLayout.ND,)
    res_tr = solve_auto_layout(inst_tr)
    assert res_tr.writeback_issues == ()

    # 2) IfStmt returning a [32, 1] tile: COL_MAJOR_FLAT reduction in `then`, load+add in `else`,
    # joined via control.if_join and stored to ND GM without op_queue desync or writeback issues.
    @pl.jit
    def if_join_col_major_kernel(
        flag: pl.Scalar[pl.BOOL],
        x: pl.Tensor[[32, 32], pl.FP16],
        y: pl.Tensor[[32, 1], pl.FP16],
        z: pl.Tensor[[32, 1], pl.FP16],
        out: pl.Out[pl.Tensor[[32, 1], pl.FP16]],
    ) -> pl.Tensor[[32, 1], pl.FP16]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="if_join_vec"):
            tx = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            ty = pl.load(y, [0, 0], [32, 1], target_memory=pl.Mem.Vec)
            tz = pl.load(z, [0, 0], [32, 1], target_memory=pl.Mem.Vec)
            if flag:
                tmp = pl.tile.create([32, 32], dtype=pl.FP16, target_memory=pl.Mem.Vec)
                branch = pl.tile.row_max(tx, tmp)
            else:
                branch = pl.tile.add(ty, tz)
            out = pl.store(branch, [0, 0], out)
        return out

    stub32 = torch.zeros((32, 32), dtype=torch.float16)
    stub32x1 = torch.zeros((32, 1), dtype=torch.float16)
    prog_if = if_join_col_major_kernel.specialize(True, stub32, stub32x1, stub32x1, stub32x1)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        pre_if = run_pre_layout_passes(prog_if)
        inst_if = extract_layout_instance(
            pre_if, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910)
        )
        assert any(op.op_name == "control.if_join" for op in inst_if.operations)
        res_if = solve_auto_layout(inst_if)
        assert res_if.writeback_issues == ()
        rw_if = apply_layout_assignment(pre_if, res_if.writeback_assignment())
        ir.assert_structural_equal(_run_resolve_layouts_pass(rw_if), rw_if)
        run_post_layout_passes(rw_if)


def test_tile_reshape_512_and_row_expand_with_reshape_consumer():
    """Verify fractal=512 tile.reshape extraction and ROW_BCAST_FLAT exclusion on tile.reshape."""
    from pypto.ir.auto_layout_solver import solve_auto_layout  # noqa: PLC0415
    from pypto.pypto_core import passes  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910)

    # 1) Direct fractal=512 tile.reshape must preserve its layout in admissible_layouts without SOLVER-FAIL.
    span = ir.Span.unknown()
    tv512 = ir.TileView(blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.none_box, fractal=512)
    ty512 = ir.TileType([1, 16, 32], DataType.FP16, None, tv512, ir.MemorySpace.Vec)
    t_in = ir.Var("t_512_in", ty512, span)
    t_out = ir.Var("t_512_out", ty512, span)
    reshape_call = ir.create_op_call(
        "tile.reshape",
        [
            t_in,
            ir.MakeTuple([ir.ConstInt(d, DataType.INDEX, span) for d in (1, 16, 32)], span),
        ],
        {},
        span,
    )
    fn_512 = ir.Function(
        "reshape_512_fn",
        [t_in],
        [ty512],
        ir.SeqStmts([ir.AssignStmt(t_out, reshape_call, span), ir.ReturnStmt([t_out], span)], span),
        span,
        type=ir.FunctionType.AIV,
        level=ir.Level.AIV,
    )
    from pypto.ir.auto_layout import TileLayoutDesc  # noqa: PLC0415

    inst_512 = extract_layout_instance(ir.Program([fn_512], "prog_512", span))
    desc512 = TileLayoutDesc(blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.none_box, fractal=512)
    assert desc512 in inst_512.values["t_512_in"].admissible_layouts
    assert desc512 in inst_512.values["t_512_out"].admissible_layouts
    res_512 = solve_auto_layout(inst_512)
    assert res_512.writeback_issues == ()

    # 2) [32, 1] tile feeding both multiple row_expand_adds AND tile.reshape excludes ROW_BCAST_FLAT
    # from tile.reshape so element counts always agree.
    @pl.jit
    def row_expand_plus_reshape(
        x: pl.Tensor[[32, 32], pl.FP32],
        q0: pl.Tensor[[32, 32], pl.FP32],
        q1: pl.Tensor[[32, 32], pl.FP32],
        q2: pl.Tensor[[32, 32], pl.FP32],
        out0: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        out1: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        out2: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        out_r: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[32, 32], pl.FP32],
        pl.Tensor[[32, 32], pl.FP32],
        pl.Tensor[[32, 32], pl.FP32],
        pl.Tensor[[1, 32], pl.FP32],
    ]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="vec_re_reshape"):
            tx = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq0 = pl.load(q0, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq1 = pl.load(q1, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tq2 = pl.load(q2, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            tmp = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            rsum = pl.tile.row_sum(tx, tmp)
            y0 = pl.tile.row_expand_add(tq0, rsum)
            y1 = pl.tile.row_expand_add(tq1, rsum)
            y2 = pl.tile.row_expand_add(tq2, rsum)
            r_row = pl.tile.reshape(rsum, [1, 32])
            out0 = pl.store(y0, [0, 0], out0)
            out1 = pl.store(y1, [0, 0], out1)
            out2 = pl.store(y2, [0, 0], out2)
            out_r = pl.store(r_row, [0, 0], out_r)
        return out0, out1, out2, out_r

    stub = torch.zeros((32, 32), dtype=torch.float32)
    stub1x32 = torch.zeros((1, 32), dtype=torch.float32)
    prog_re = row_expand_plus_reshape.specialize(stub, stub, stub, stub, stub, stub, stub, stub1x32)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        memory_planner=passes.MemoryPlanner.PYPTO,
    ):
        pre_re = run_pre_layout_passes(prog_re)
        inst_re = extract_layout_instance(
            pre_re, cost_model=LayoutCostModel(backend_type=BackendType.Ascend910)
        )
        reshape_ops = [op for op in inst_re.operations if op.op_name == "tile.reshape"]
        assert len(reshape_ops) == 1
        assert all(ROW_BCAST_FLAT not in cfg.input_layouts for cfg in reshape_ops[0].configs)
        res_re = solve_auto_layout(inst_re)
        assert res_re.writeback_issues == ()
        rw_re = apply_layout_assignment(pre_re, res_re.writeback_assignment())
        ir.assert_structural_equal(_run_resolve_layouts_pass(rw_re), rw_re)
        run_post_layout_passes(rw_re)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
