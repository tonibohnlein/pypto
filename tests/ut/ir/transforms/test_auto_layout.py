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
from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import DataType, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir import auto_layout_bench
from pypto.ir.auto_layout import (
    FRACTAL_ACC_NZ,
    FRACTAL_NZ,
    FRACTAL_ZN,
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
    pack_nz,
    run_post_layout_passes,
    run_pre_layout_passes,
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

    def test_fit_cost_model_from_synthetic_measurements(self):
        set_backend_type(BackendType.Ascend910)
        shapes = [(64, 128, 64), (64, 256, 64), (128, 128, 64), (128, 128, 128)]
        true_overhead_us = 5.0
        true_us_per_cycle = 1.0 / 1500.0
        true_nd2nz_bpc = 16.0
        true_nz2nz_bpc = 64.0
        true_store_nd_bpc = 24.0
        true_store_nz_bpc = 56.0
        base_model = LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910)
        measurements: list[CalibrationMeasurement] = []
        for s_idx, (m_dim, k_dim, n_dim) in enumerate(shapes):
            case = auto_layout_bench.build_single_matmul_case(
                m_dim, k_dim, n_dim, expected_uses=100, seed=s_idx
            )
            pre_prog = run_pre_layout_passes(case.program)
            inst = extract_layout_instance(
                pre_prog,
                cost_model=base_model,
                reusable_tensors={"w"},
            )
            for v_idx, bd in enumerate(enumerate_assignments(inst, deduplicate=True)):
                nd2nz_b = 0.0
                nz2nz_b = 0.0
                store_nd_b = 0.0
                store_nz_b = 0.0
                other_cycles = 0.0
                for op in inst.operations:
                    cfg = bd.assignment.op_configs[op.op_id]
                    if op.op_name == "tile.load":
                        dst = inst.values[op.output]
                        if dst.memory_space == ir.MemorySpace.Mat:
                            if cfg.input_layouts[0] == ir.TensorLayout.NZ:
                                nz2nz_b += float(dst.byte_size)
                            else:
                                nd2nz_b += float(dst.byte_size)
                            continue
                    elif op.op_name == "tile.store":
                        src = inst.values[op.inputs[0]]
                        if cfg.output_layout == ir.TensorLayout.NZ:
                            store_nz_b += float(src.byte_size)
                        else:
                            store_nd_b += float(src.byte_size)
                        continue
                    other_cycles += cfg.op_cost_cycles

                measured_us = true_overhead_us + true_us_per_cycle * (
                    nd2nz_b / true_nd2nz_bpc
                    + nz2nz_b / true_nz2nz_bpc
                    + store_nd_b / true_store_nd_bpc
                    + store_nz_b / true_store_nz_bpc
                    + other_cycles
                )
                measurements.append(
                    CalibrationMeasurement(
                        variant_id=f"{case.case_id}/v{v_idx}",
                        measured_us=measured_us,
                        instance=inst,
                        breakdown=bd,
                    )
                )

        report = fit_cost_model_from_measurements(measurements, base_model)
        assert report.r2 > 0.999
        assert report.spearman_rho == pytest.approx(1.0)
        assert report.effective_freq_ghz == pytest.approx(1.5, rel=1e-3)
        assert report.effective_freq_fallback is False
        assert report.fallback_params == ()
        assert report.fixed_overhead_us == pytest.approx(true_overhead_us, abs=1e-4)
        assert report.fitted_model.mte2_nd2nz_bytes_per_cycle == pytest.approx(true_nd2nz_bpc, rel=1e-3)
        assert report.fitted_model.mte2_nz2nz_bytes_per_cycle == pytest.approx(true_nz2nz_bpc, rel=1e-3)
        assert report.fitted_model.fixpipe_acc_to_nd_bytes_per_cycle == pytest.approx(
            true_store_nd_bpc, rel=1e-3
        )
        assert report.fitted_model.fixpipe_acc_to_nz_bytes_per_cycle == pytest.approx(
            true_store_nz_bpc, rel=1e-3
        )

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

    def test_fit_cost_model_normalizes_expected_uses_and_flags_overlapped_freq(self):
        set_backend_type(BackendType.Ascend910)
        base_model = LayoutCostModel(expected_uses=100, backend_type=BackendType.Ascend910)
        nominal_us_per_cyc = 1.0 / (1.8 * 1000.0)
        true_nd2nz_bpc = 75.0
        true_nz2nz_bpc = 120.0
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
                nd2nz_b = nz2nz_b = other_cyc = 0.0
                for op in inst.operations:
                    cfg = bd.assignment.op_configs[op.op_id]
                    inv = float(op.invocations) / run_scale
                    if op.op_name == "tile.load":
                        dst = inst.values[op.output]
                        if dst.memory_space == ir.MemorySpace.Mat:
                            if cfg.input_layouts[0] == ir.TensorLayout.NZ:
                                nz2nz_b += inv * float(dst.byte_size)
                            else:
                                nd2nz_b += inv * float(dst.byte_size)
                            continue
                    elif op.op_name == "tile.store":
                        continue
                    other_cyc += inv * cfg.op_cost_cycles

                # Simulate pipelined execution: `other_cyc` contributes only 25% non-overlapped
                # (~7.2 GHz raw slope) and FIXPIPE stores add ~0 delta.
                measured_us = (
                    16.0
                    + nominal_us_per_cyc * (nd2nz_b / true_nd2nz_bpc + nz2nz_b / true_nz2nz_bpc)
                    + 0.25 * nominal_us_per_cyc * other_cyc
                )
                meas.append(
                    CalibrationMeasurement(
                        variant_id=f"pipe_{s_idx}_{v_idx}",
                        measured_us=measured_us,
                        instance=inst,
                        breakdown=bd,
                    )
                )

        report = fit_cost_model_from_measurements(meas, base_model)
        assert report.effective_freq_fallback is True
        assert report.effective_freq_ghz == pytest.approx(1.8)
        assert report.fitted_model.mte2_nd2nz_bytes_per_cycle == pytest.approx(true_nd2nz_bpc, rel=1e-2)
        assert report.fitted_model.mte2_nz2nz_bytes_per_cycle == pytest.approx(true_nz2nz_bpc, rel=1e-2)
        assert set(report.fallback_params) == {
            "fixpipe_acc_to_nd_bytes_per_cycle",
            "fixpipe_acc_to_nz_bytes_per_cycle",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
