# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Calibration & benchmarking harness for Ascend layout cost modeling.

This module defines `@pl.jit` L0-fitting calibration workloads, enumerates and
compiles every legal layout assignment via :func:`compile_layout_variants`
(with ``AutoLayout`` inserted after ``InferTileMemorySpace``), executes
correctness + multi-round device timing via :func:`pypto.runtime.benchmark`,
and fits :class:`LayoutCostModel` coefficients from the measured latencies.

Usage on a dev host (compile-only / `.pto` codegen check)::

    python -m pypto.ir.auto_layout_bench --compile-only --output-dir build_output/layout_calib

Usage on an Ascend A2 device host::

    python -m pypto.ir.auto_layout_bench --platform a2a3 --device 0 \\
        --rounds 100 --warmup 5 --output calibration_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

import pypto.language as pl
from pypto._target import _parse_platform
from pypto.ir.auto_layout import (
    CalibrationMeasurement,
    CalibrationReport,
    LayoutCostModel,
    LayoutProblemInstance,
    LayoutVariantArtifact,
    compile_layout_variants,
    fit_cost_model_from_measurements,
    pack_nz,
    unpack_nz,
)
from pypto.ir.compiled_program import CompiledProgram
from pypto.pypto_core import ir as core_ir
from pypto.pypto_core import passes
from pypto.runtime import RunConfig, benchmark


@dataclass(frozen=True)
class CalibrationCaseSpec:
    """Description of one `@pl.jit` calibration workload & reference inputs."""

    case_id: str
    program: core_ir.Program
    inputs: tuple[torch.Tensor, ...]
    expected_output: torch.Tensor
    output_shape: tuple[int, ...]
    output_dtype: torch.dtype
    reusable_tensors: set[str]
    expected_uses: int = 100


def build_single_matmul_case(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    expected_uses: int = 100,
    seed: int = 0,
) -> CalibrationCaseSpec:
    """Build an L0-fitting `C[M, N] = X[M, K] @ W[N, K].T` FP16->FP32 `@pl.jit` workload."""

    @pl.jit.incore
    def matmul_kernel(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        xt = pl.load(x, [0, 0], [m_dim, k_dim], target_memory=pl.MemorySpace.Mat)
        wt = pl.load(w, [0, 0], [n_dim, k_dim], target_memory=pl.MemorySpace.Mat)
        wt_t = pl.tile.transpose_view(wt)
        acc = pl.matmul(xt, wt_t)
        pl.store(acc, [0, 0], out)
        return out

    @pl.jit
    def matmul_calib_entry(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return matmul_kernel(x, w, out)

    gen = torch.Generator().manual_seed(seed)
    x_host = torch.randn((m_dim, k_dim), generator=gen, dtype=torch.float16) * 0.25
    w_host = torch.randn((n_dim, k_dim), generator=gen, dtype=torch.float16) * 0.25
    out_stub = torch.zeros((m_dim, n_dim), dtype=torch.float32)

    program = matmul_calib_entry.specialize(x_host, w_host, out_stub)
    expected = x_host.float() @ w_host.float().T

    return CalibrationCaseSpec(
        case_id=f"matmul_{m_dim}x{k_dim}x{n_dim}",
        program=program,
        inputs=(x_host, w_host),
        expected_output=expected,
        output_shape=(m_dim, n_dim),
        output_dtype=torch.float32,
        reusable_tensors={"w"},
        expected_uses=expected_uses,
    )


def build_two_matmul_chain_case(
    m_dim: int = 64,
    k_dim: int = 128,
    h_dim: int = 64,
    n_dim: int = 64,
    *,
    expected_uses: int = 100,
    seed: int = 1,
) -> CalibrationCaseSpec:
    """Build a two-matmul `@pl.jit` chain `Mid = X @ W1.T; Out = Mid @ W2.T`.

    `Mid` (`[M, H]` FP32) is produced in GM by `mm_stage1` and consumed in GM
    by `mm_stage2`, allowing calibration of intra-program `ND` vs. `NZ`
    intermediate GM storage without host packing.
    """

    @pl.jit.incore
    def mm_stage1(
        x: pl.Tensor,
        w1: pl.Tensor,
        mid: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        xt = pl.load(x, [0, 0], [m_dim, k_dim], target_memory=pl.MemorySpace.Mat)
        w1t = pl.load(w1, [0, 0], [h_dim, k_dim], target_memory=pl.MemorySpace.Mat)
        w1_t = pl.tile.transpose_view(w1t)
        acc1 = pl.matmul(xt, w1_t)
        pl.store(acc1, [0, 0], mid)
        return mid

    @pl.jit.incore
    def mm_stage2(
        mid: pl.Tensor,
        w2: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        mt = pl.load(mid, [0, 0], [m_dim, h_dim], target_memory=pl.MemorySpace.Mat)
        w2t = pl.load(w2, [0, 0], [n_dim, h_dim], target_memory=pl.MemorySpace.Mat)
        w2_t = pl.tile.transpose_view(w2t)
        acc2 = pl.matmul(mt, w2_t)
        pl.store(acc2, [0, 0], out)
        return out

    @pl.jit
    def two_matmul_chain_entry(
        x: pl.Tensor,
        w1: pl.Tensor,
        w2: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        mid = pl.create_tensor([m_dim, h_dim], dtype=pl.FP32)
        mid = mm_stage1(x, w1, mid)
        out = mm_stage2(mid, w2, out)
        return out

    gen = torch.Generator().manual_seed(seed)
    x_host = torch.randn((m_dim, k_dim), generator=gen, dtype=torch.float16) * 0.25
    w1_host = torch.randn((h_dim, k_dim), generator=gen, dtype=torch.float16) * 0.25
    w2_host = torch.randn((n_dim, h_dim), generator=gen, dtype=torch.float32) * 0.25
    out_stub = torch.zeros((m_dim, n_dim), dtype=torch.float32)

    program = two_matmul_chain_entry.specialize(x_host, w1_host, w2_host, out_stub)
    mid_ref = x_host.float() @ w1_host.float().T
    expected = mid_ref @ w2_host.float().T

    return CalibrationCaseSpec(
        case_id=f"chain_{m_dim}x{k_dim}x{h_dim}x{n_dim}",
        program=program,
        inputs=(x_host, w1_host, w2_host),
        expected_output=expected,
        output_shape=(m_dim, n_dim),
        output_dtype=torch.float32,
        reusable_tensors={"w1", "w2"},
        expected_uses=expected_uses,
    )


def _entry_function(program: core_ir.Program) -> core_ir.Function:
    """Return the `Orchestration` entry function of *program* (falling back to the last function)."""
    orch_funcs = [
        fn for fn in program.functions.values() if fn.func_type == core_ir.FunctionType.Orchestration
    ]
    if orch_funcs:
        return orch_funcs[-1]
    return next(reversed(list(program.functions.values())))


def prepare_variant_tensors(
    case: CalibrationCaseSpec,
    variant: LayoutVariantArtifact,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Pack any `NZ`-assigned GM inputs via `pack_nz` in entry-parameter order and allocate output."""
    entry = _entry_function(case.program)
    param_order = [p.name_hint for p in entry.params]
    input_param_names = [n for n in param_order if n in variant.gm_input_layouts]
    if len(input_param_names) != len(case.inputs):
        raise ValueError(
            f"{case.case_id}/{variant.variant_id}: entry params {param_order} matched "
            f"{input_param_names} in gm_input_layouts={sorted(variant.gm_input_layouts)}, "
            f"expected {len(case.inputs)} inputs."
        )

    call_inputs: list[torch.Tensor] = []
    for p_name, tensor in zip(input_param_names, case.inputs, strict=True):
        if variant.gm_input_layouts.get(p_name) == "NZ":
            call_inputs.append(pack_nz(tensor))
        else:
            call_inputs.append(tensor.contiguous())

    out_buf = torch.zeros(case.output_shape, dtype=case.output_dtype)
    return call_inputs, out_buf


def _extract_swimlane_aicore_us(work_dir: Path) -> float | None:
    """Read median AICore task duration (µs) from level-1 swimlane records if present."""
    dfx_dir = work_dir / "dfx_outputs"
    records_path = dfx_dir / "chip_swimlane_records.json"
    if not records_path.is_file():
        return None
    try:
        payload_obj: Any = json.loads(records_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload_obj, list):
        events = cast(list[object], payload_obj)
    elif isinstance(payload_obj, dict):
        payload_dict = cast(dict[str, object], payload_obj)
        raw_events: object = payload_dict.get("traceEvents") or payload_dict.get("events")
        events = cast(list[object], raw_events) if isinstance(raw_events, list) else []
    else:
        return None
    durations_us: list[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_dict = cast(dict[str, object], ev)
        dur = ev_dict.get("dur")
        ph = ev_dict.get("ph")
        if ph == "X" and isinstance(dur, (int, float)) and dur > 0:
            durations_us.append(float(dur))
    if not durations_us:
        return None
    return statistics.median(durations_us)


def default_a2_calibration_suite(
    *,
    include_chain: bool = True,
    expected_uses: int = 100,
) -> list[CalibrationCaseSpec]:
    """Return the default L0-fitting Ascend A2 (`Ascend910`) calibration workload suite."""
    # All single-matmul operand tiles stay <= 32 KiB in L0A/L0B so AutoTileMatmulL0 is bypassed
    # cleanly while preserving full rank across [1, load_bytes, store_bytes, macs].
    cases = [
        build_single_matmul_case(64, 128, 64, expected_uses=expected_uses, seed=0),
        build_single_matmul_case(64, 256, 64, expected_uses=expected_uses, seed=1),
        build_single_matmul_case(128, 128, 64, expected_uses=expected_uses, seed=2),
        build_single_matmul_case(128, 128, 128, expected_uses=expected_uses, seed=3),
    ]
    if include_chain:
        cases.append(build_two_matmul_chain_case(64, 128, 64, 64, expected_uses=expected_uses, seed=4))
    return cases


def _serialize_calibration_report(
    measurements: Sequence[CalibrationMeasurement],
    base_model: LayoutCostModel,
) -> dict[str, Any] | None:
    if not measurements:
        return None
    report: CalibrationReport = fit_cost_model_from_measurements(measurements, base_model)
    return {
        "num_measurements": len(measurements),
        "fixed_overhead_us": report.fixed_overhead_us,
        "effective_freq_ghz": report.effective_freq_ghz,
        "r2": report.r2,
        "spearman_rho": report.spearman_rho,
        "mape_percent": report.mape_percent,
        "fitted_params": {
            "mte2_nd2nz_bytes_per_cycle": report.fitted_model.mte2_nd2nz_bytes_per_cycle,
            "mte2_nz2nz_bytes_per_cycle": report.fitted_model.mte2_nz2nz_bytes_per_cycle,
            "fixpipe_acc_to_nd_bytes_per_cycle": report.fitted_model.fixpipe_acc_to_nd_bytes_per_cycle,
            "fixpipe_acc_to_nz_bytes_per_cycle": report.fitted_model.fixpipe_acc_to_nz_bytes_per_cycle,
        },
        "residuals_us": report.residuals_us,
    }


def run_calibration_suite(  # noqa: PLR0913
    cases: Sequence[CalibrationCaseSpec],
    *,
    output_dir: str | Path,
    platform: str = "a2a3",
    device_id: int = 0,
    rounds: int = 100,
    warmup: int = 5,
    compile_only: bool = False,
    capture_swimlane: bool = False,
    memory_planner: passes.MemoryPlanner = passes.MemoryPlanner.PYPTO,
    cost_model: LayoutCostModel | None = None,
) -> dict[str, Any]:
    """Compile all enumerated layout variants for `cases` and optionally benchmark on device.

    Returns a serializable dictionary with every variant's build directory,
    predicted cost breakdown, accuracy check, and measured device latency, plus
    the fitted `CalibrationReport` when device timings are collected.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    backend_type, execution_mode = _parse_platform(platform)

    base_model = cost_model or LayoutCostModel(backend_type=backend_type)
    measurements: list[CalibrationMeasurement] = []
    swimlane_measurements: list[CalibrationMeasurement] = []
    suite_records: list[dict[str, Any]] = []
    pending_swimlanes: list[
        tuple[
            CalibrationCaseSpec,
            LayoutProblemInstance,
            LayoutVariantArtifact,
            CompiledProgram,
            list[torch.Tensor],
            torch.Tensor,
            bool,
            dict[str, Any],
        ]
    ] = []

    # Pass A: compile all variants, check numerical accuracy, and run benchmark()
    # before any enable_chip_swimlane=1 launch redirects host stderr logs away from [STRACE].
    for case in cases:
        case_dir = out_root / case.case_id
        model_for_case = LayoutCostModel(
            expected_uses=case.expected_uses,
            backend_type=backend_type,
            mte2_nz2nz_bytes_per_cycle=base_model.mte2_nz2nz_bytes_per_cycle,
            mte2_nd2nz_bytes_per_cycle=base_model.mte2_nd2nz_bytes_per_cycle,
            mte2_nd2nz_row_penalty_cycles=base_model.mte2_nd2nz_row_penalty_cycles,
            mte2_vec_bytes_per_cycle=base_model.mte2_vec_bytes_per_cycle,
            mte1_l1_to_l0_bytes_per_cycle=base_model.mte1_l1_to_l0_bytes_per_cycle,
            mte1_transpose_penalty_cycles=base_model.mte1_transpose_penalty_cycles,
            cube_macs_per_cycle=base_model.cube_macs_per_cycle,
            cube_issue_overhead_cycles=base_model.cube_issue_overhead_cycles,
            vec_elements_per_cycle=base_model.vec_elements_per_cycle,
            fixpipe_acc_to_nd_bytes_per_cycle=base_model.fixpipe_acc_to_nd_bytes_per_cycle,
            fixpipe_acc_to_nz_bytes_per_cycle=base_model.fixpipe_acc_to_nz_bytes_per_cycle,
            gm_pack_bytes_per_cycle=base_model.gm_pack_bytes_per_cycle,
            gm_pack_fixed_cycles=base_model.gm_pack_fixed_cycles,
            onchip_relayout_bytes_per_cycle=base_model.onchip_relayout_bytes_per_cycle,
            allow_external_output_relayout=base_model.allow_external_output_relayout,
        )

        instance, variants = compile_layout_variants(
            case.program,
            output_dir=case_dir,
            cost_model=model_for_case,
            reusable_tensors=case.reusable_tensors,
            backend_type=backend_type,
            execution_mode=execution_mode,
            skip_ptoas=compile_only,
            memory_planner=memory_planner,
        )

        variant_records: list[dict[str, Any]] = []
        for variant in variants:
            rec: dict[str, Any] = {
                "case_id": case.case_id,
                "variant_id": variant.variant_id,
                "build_dir": str(variant.build_dir),
                "gm_input_layouts": variant.gm_input_layouts,
                "gm_output_layouts": variant.gm_output_layouts,
                "gm_layouts": variant.gm_layouts,
                "predicted_total_cycles": variant.breakdown.total_cost_cycles,
                "predicted_op_cycles": variant.breakdown.operator_cost_cycles,
                "predicted_conv_cycles": variant.breakdown.conversion_cost_cycles,
                "pto_count": len(variant.pto_files),
            }

            if not compile_only:
                compiled = CompiledProgram.from_dir(
                    variant.build_dir,
                    backend_type=backend_type,
                    execution_mode=execution_mode,
                )
                in_tensors, out_buf = prepare_variant_tensors(case, variant)
                all_args = [*in_tensors, out_buf]

                # 1. Correctness validation
                run_cfg = RunConfig(
                    backend_type=backend_type,
                    execution_mode=execution_mode,
                    device_id=device_id,
                )
                compiled(*all_args, config=run_cfg)
                actual = (
                    unpack_nz(out_buf)
                    if next(iter(variant.gm_output_layouts.values()), "ND") == "NZ"
                    else out_buf
                )
                max_err = float((actual - case.expected_output).abs().max().item())
                ok = bool(torch.allclose(actual, case.expected_output, rtol=1e-2, atol=1e-2))
                rec["accuracy_ok"] = ok
                rec["max_abs_err"] = max_err

                # 2. Device benchmark (do not mix config= with platform=/device_id=).
                stats = benchmark(
                    compiled,
                    all_args,
                    rounds=rounds,
                    warmup=warmup,
                    config=run_cfg,
                )
                eff_samples = [inv.effective_us for inv in stats.invocations if inv.effective_us > 0.0]
                eff_median = statistics.median(eff_samples) if eff_samples else 0.0

                rec["device_us_median"] = stats.device_us_median
                rec["device_us_min"] = stats.device_us_min
                rec["device_us_mean"] = stats.device_us_mean
                rec["device_us_max"] = stats.device_us_max
                rec["device_us_stdev"] = stats.device_us_stdev
                rec["effective_us_median"] = eff_median
                rec["all_zero_device"] = stats.all_zero_device

                measured_metric = stats.device_us_median if stats.device_us_median > 0.0 else eff_median
                if ok and measured_metric > 0.0:
                    measurements.append(
                        CalibrationMeasurement(
                            variant_id=f"{case.case_id}/{variant.variant_id}",
                            measured_us=measured_metric,
                            instance=instance,
                            breakdown=variant.breakdown,
                            label=f"{case.case_id}:{variant.gm_input_layouts}->{variant.gm_output_layouts}",
                        )
                    )
                if capture_swimlane:
                    pending_swimlanes.append((case, instance, variant, compiled, all_args, out_buf, ok, rec))

            variant_records.append(rec)

        suite_records.append({"case_id": case.case_id, "variants": variant_records})

    # Pass B: level-1 chip swimlane capture after all benchmark() invocations complete.
    if capture_swimlane and not compile_only:
        swimlane_cfg = RunConfig(
            backend_type=backend_type,
            execution_mode=execution_mode,
            device_id=device_id,
            enable_chip_swimlane=1,
        )
        for case, instance, variant, compiled, all_args, out_buf, ok, rec in pending_swimlanes:
            out_buf.zero_()
            compiled(*all_args, config=swimlane_cfg)
            aicore_us = _extract_swimlane_aicore_us(variant.build_dir)
            rec["swimlane_aicore_us"] = aicore_us
            if ok and isinstance(aicore_us, float) and aicore_us > 0.0:
                swimlane_measurements.append(
                    CalibrationMeasurement(
                        variant_id=f"{case.case_id}/{variant.variant_id}",
                        measured_us=aicore_us,
                        instance=instance,
                        breakdown=variant.breakdown,
                        label=f"{case.case_id}:{variant.gm_input_layouts}->{variant.gm_output_layouts}",
                    )
                )

    calibration = _serialize_calibration_report(measurements, base_model)
    calibration_swimlane = _serialize_calibration_report(swimlane_measurements, base_model)

    summary: dict[str, Any] = {
        "platform": platform,
        "device_id": device_id,
        "rounds": rounds,
        "warmup": warmup,
        "compile_only": compile_only,
        "cases": suite_records,
        "calibration": calibration,
        "calibration_swimlane": calibration_swimlane,
    }
    return summary


def _print_calibration_block(title: str, cal: dict[str, Any] | None) -> None:
    if cal is None:
        return
    print(f"\n=== {title} ===")
    print(
        f"  measurements={cal['num_measurements']} R2={cal['r2']:.4f} "
        f"Spearman={cal['spearman_rho']:.4f} MAPE={cal['mape_percent']:.2f}% "
        f"overhead={cal['fixed_overhead_us']:.2f}us eff_freq={cal['effective_freq_ghz']:.3f}GHz"
    )
    print(f"  fitted_params={cal['fitted_params']}")


def _print_summary(summary: dict[str, Any]) -> None:
    for case in summary["cases"]:
        print(f"\n=== Case: {case['case_id']} ({len(case['variants'])} variants) ===")
        for v in case["variants"]:
            dev = v.get("device_us_median")
            dev_str = f"dev_med={dev:.2f}us" if isinstance(dev, float) and not math.isnan(dev) else "compiled"
            sw = v.get("swimlane_aicore_us")
            sw_str = f" swim={sw:.2f}us" if isinstance(sw, float) and not math.isnan(sw) else ""
            acc = v.get("accuracy_ok")
            acc_str = f" acc={'PASS' if acc else 'FAIL'}" if acc is not None else ""
            print(
                f"  {v['variant_id']} gm={v['gm_layouts']} "
                f"pred_op={v['predicted_op_cycles']:.1f}cyc pred_total={v['predicted_total_cycles']:.1f}cyc "
                f"{dev_str}{sw_str}{acc_str}"
            )
    _print_calibration_block("Calibrated LayoutCostModel (device_us_median)", summary.get("calibration"))
    _print_calibration_block(
        "Calibrated LayoutCostModel (swimlane_aicore_us)", summary.get("calibration_swimlane")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-layout variant generator & A2 calibration harness")
    parser.add_argument("--platform", default="a2a3", help="Target platform (a2a3, a2a3sim, a5, a5sim)")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device ID for onboard benchmarking")
    parser.add_argument("--rounds", type=int, default=100, help="Measured rounds per variant")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup rounds per variant")
    parser.add_argument("--expected-uses", type=int, default=100, help="Reuse factor R for weights")
    parser.add_argument(
        "--output-dir",
        default="build_output/auto_layout_calib",
        help="Root directory for compiled variant artifacts and manifest.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file path to save benchmark + calibration results",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only enumerate and emit IR / .pto artifacts without ptoas or device execution",
    )
    parser.add_argument(
        "--swimlane",
        action="store_true",
        help="Also capture level-1 chip swimlane AICore task timings on device",
    )
    parser.add_argument(
        "--no-chain",
        action="store_true",
        help="Skip the two-matmul chained workload",
    )
    args = parser.parse_args(argv)

    cases = default_a2_calibration_suite(
        include_chain=not args.no_chain,
        expected_uses=args.expected_uses,
    )
    summary = run_calibration_suite(
        cases,
        output_dir=args.output_dir,
        platform=args.platform,
        device_id=args.device,
        rounds=args.rounds,
        warmup=args.warmup,
        compile_only=args.compile_only,
        capture_swimlane=args.swimlane,
    )
    _print_summary(summary)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
