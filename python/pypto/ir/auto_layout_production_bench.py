# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark PyPTO auto-layout pass on real `pypto-lib` production kernels.

Compiles and benchmarks each production kernel in two (or more) configurations:
1. `baseline_nd`: 2D weights and activations in default `ND` layout (no auto-layout pass).
2. `auto_layout`: Compiled with `make_auto_layout_pass` using the calibrated Ascend A2
   `LayoutCostModel` and `solve_auto_layout` (hybrid treewidth-DP / greedy dispatcher).

Validates every variant against the kernel's reference `golden_fn` and collects
`pypto.runtime.benchmark` (`device_us_median`, `effective_us_median`) plus optional
level-1 chip swimlane AICore durations (`swimlane_aicore_us`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

import pypto.language as pl
from pypto._target import ExecutionMode, _parse_platform
from pypto.backend import BackendType, set_backend_type
from pypto.ir.auto_layout import (
    LayoutAssignment,
    LayoutCostModel,
    evaluate_assignment,
    extract_layout_instance,
    make_auto_layout_pass,
    pack_nz,
    run_pre_layout_passes,
    unpack_nz,
)
from pypto.ir.auto_layout_bench import _extract_swimlane_aicore_us
from pypto.ir.auto_layout_solver import (
    LayoutSolverConfig,
    compute_treewidth,
    solve_auto_layout,
    solve_greedy,
    solve_treewidth_dp,
)
from pypto.ir.compile import compile as compile_program
from pypto.pypto_core import ir as core_ir

TensorLayout = core_ir.TensorLayout

PRODUCTION_WEIGHT_NAMES: frozenset[str] = frozenset(
    {
        "b",
        "w",
        "w1",
        "w2",
        "w3",
        "wo",
        "wq",
        "wk",
        "wv",
        "w_gate_up",
        "w_down",
        "weight",
        "lm_head_weight",
        "shared_w1",
        "shared_w2",
        "shared_w3",
        "e_proj_w",
        "h_proj_w",
        "wq_a",
        "wq_b",
        "wkv",
        "wgate",
        "idx_wq_b",
        "weights_proj",
        "wo_a",
        "wo_b",
    }
)


@dataclass(frozen=True)
class ProductionKernelCase:
    """Specification of a `pypto-lib` production kernel benchmark case."""

    case_id: str
    category: str
    description: str
    jit_fn: Any
    specs: list[Any]
    golden_fn: Callable[[dict[str, Any]], None]
    compare_fn: dict[str, Callable[..., Any]]
    rtol: float = 1e-3
    atol: float = 1e-3
    reusable_tensors: frozenset[str] = PRODUCTION_WEIGHT_NAMES
    expected_uses: int = 100
    include_shared_input_variant: bool = False


def _reset_model_modules(subdir: Path, module_names: Sequence[str]) -> None:
    """Cleanly switch `sys.path` to a model directory with local `config`/`utils` modules."""
    subdir_str = str(subdir)
    for p in list(sys.path):
        if "/models/" in p and p != subdir_str:
            sys.path.remove(p)
    if subdir_str not in sys.path:
        sys.path.insert(0, subdir_str)
    for mod in module_names:
        sys.modules.pop(mod, None)


def load_pypto_lib_production_suite(
    pypto_lib_dir: str | Path,
    *,
    expected_uses: int = 100,
    case_filter: Sequence[str] | None = None,
) -> list[ProductionKernelCase]:
    """Load the production kernel benchmark suite from `pypto-lib`."""
    root = Path(pypto_lib_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pypto-lib directory not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    golden_mod = importlib.import_module("golden")
    ratio_allclose = golden_mod.ratio_allclose
    ratio_reldiff = golden_mod.ratio_reldiff

    selected = set(case_filter) if case_filter else None
    cases: list[ProductionKernelCase] = []

    def _want(cid: str) -> bool:
        return selected is None or cid in selected

    # 1. Examples (`matmul`, `gemm`, `gemm_eltwise`, `qkv_proj`)
    if _want("examples.matmul"):
        ex_matmul = importlib.import_module("examples.beginner.matmul")

        cases.append(
            ProductionKernelCase(
                case_id="examples.matmul",
                category="examples",
                description="[256, 256] @ [256, 256] FP32 tiled matmul",
                jit_fn=ex_matmul.matmul,
                specs=ex_matmul.build_tensor_specs(),
                golden_fn=ex_matmul.golden_matmul,
                compare_fn={},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    if _want("examples.gemm"):
        ex_gemm = importlib.import_module("examples.intermediate.gemm")

        cases.append(
            ProductionKernelCase(
                case_id="examples.gemm",
                category="examples",
                description="[256, 256] @ [256, 256] FP32 K-accumulated GEMM",
                jit_fn=ex_gemm.gemm,
                specs=ex_gemm.build_tensor_specs(),
                golden_fn=ex_gemm.golden_gemm,
                compare_fn={},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    if _want("examples.gemm_eltwise"):
        ex_ge = importlib.import_module("examples.advanced.gemm_eltwise")

        cases.append(
            ProductionKernelCase(
                case_id="examples.gemm_eltwise",
                category="examples",
                description="[16, 8192] @ [8192, 8192] + [16, 8192] BF16->FP32 projection + residual",
                jit_fn=ex_ge.gemm_eltwise,
                specs=ex_ge.build_tensor_specs(),
                golden_fn=ex_ge.golden_gemm_eltwise,
                compare_fn={},
                rtol=1e-2,
                atol=1e-2,
                expected_uses=expected_uses,
            )
        )

    if _want("examples.qkv_proj"):
        ex_mp = importlib.import_module("examples.advanced.multi_proj")

        cases.append(
            ProductionKernelCase(
                case_id="examples.qkv_proj",
                category="examples",
                description="3-call shared-input Q/K/V projection [16, 8192] @ [8192, 8192] BF16->FP32",
                jit_fn=ex_mp.qkv_proj,
                specs=ex_mp.build_tensor_specs(),
                golden_fn=ex_mp.golden_qkv_proj,
                compare_fn={},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
                include_shared_input_variant=True,
            )
        )

    # 2. GLM-5.3-Flash production kernels (`dense_mlp_int8`, `expert_shared`)
    if _want("glm5.dense_mlp_int8"):
        glm5_dm = importlib.import_module("models.glm5_3_flash.dense_mlp")

        cases.append(
            ProductionKernelCase(
                case_id="glm5.dense_mlp_int8",
                category="glm5_3_flash",
                description="GLM-5.3-Flash W8A8 INT8 SwiGLU Dense MLP (T=64, D=4096, inter=768)",
                jit_fn=glm5_dm.dense_mlp_test,
                specs=glm5_dm.build_tensor_specs(64),
                golden_fn=glm5_dm.golden_dense_mlp_fn,
                compare_fn={"output": ratio_reldiff(diff_thd=0.003, pct_thd=0.01)},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    if _want("glm5.expert_shared"):
        glm5_es = importlib.import_module("models.glm5_3_flash.expert_shared")

        cases.append(
            ProductionKernelCase(
                case_id="glm5.expert_shared",
                category="glm5_3_flash",
                description="GLM-5.3-Flash W8A8 INT8 MoE Shared Expert (T=64, D=4096, inter=128)",
                jit_fn=glm5_es.expert_shared_test,
                specs=glm5_es.build_tensor_specs(64),
                golden_fn=glm5_es.golden_expert_shared_fn,
                compare_fn={"output": ratio_reldiff(diff_thd=0.003, pct_thd=0.01)},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    # 3. DeepSeek-V4-Flash MTP production kernels
    dsv4_dir = root / "models" / "deepseek_v4_flash_mtp"
    _reset_model_modules(
        dsv4_dir,
        (
            "config",
            "utils",
            "rmsnorm",
            "rope_interleave",
            "expert_shared",
            "mtp_projection",
            "qkv_proj_rope",
            "decode_compressor_ratio4",
            "decode_compressor_ratio128",
            "decode_sparse_attn_csa",
            "decode_indexer",
            "decode_csa",
        ),
    )

    if _want("dsv4.expert_shared"):
        dsv4_es = importlib.import_module("expert_shared")
        cases.append(
            ProductionKernelCase(
                case_id="dsv4.expert_shared",
                category="deepseek_v4_flash_mtp",
                description="DeepSeek-V4-Flash W8A8 INT8 MoE Shared Expert (T=16, D=4096, inter=2048)",
                jit_fn=dsv4_es.expert_shared_test,
                specs=dsv4_es.build_tensor_specs(),
                golden_fn=dsv4_es.golden_expert_shared,
                compare_fn={"sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    if _want("dsv4.mtp_projection"):
        dsv4_mtp = importlib.import_module("mtp_projection")
        cases.append(
            ProductionKernelCase(
                case_id="dsv4.mtp_projection",
                category="deepseek_v4_flash_mtp",
                description="DeepSeek-V4-Flash W8A8 INT8 MTP Projection (decode T=16, D=4096)",
                jit_fn=dsv4_mtp.mtp_projection_test,
                specs=dsv4_mtp.build_tensor_specs(dsv4_mtp.DECODE_BATCH, dsv4_mtp.DECODE_SEQ),
                golden_fn=dsv4_mtp.golden_mtp_projection,
                compare_fn={"hidden_states_out": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05)},
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    if _want("dsv4.qkv_proj_rope"):
        dsv4_qkv = importlib.import_module("qkv_proj_rope")
        cases.append(
            ProductionKernelCase(
                case_id="dsv4.qkv_proj_rope",
                category="deepseek_v4_flash_mtp",
                description="DeepSeek-V4-Flash MLA QKV Projection + RMSNorm + RoPE (decode T=16)",
                jit_fn=dsv4_qkv.qkv_proj_rope_test,
                specs=dsv4_qkv.build_tensor_specs(dsv4_qkv.DECODE_BATCH, dsv4_qkv.DECODE_SEQ),
                golden_fn=dsv4_qkv.golden_qkv_proj_rope,
                compare_fn={
                    "q": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=0.02),
                    "kv": ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.01),
                    "qr": ratio_allclose(atol=2, rtol=0.05, max_error_ratio=0.02),
                    "qr_scale": ratio_allclose(atol=1e-4, rtol=1e-2),
                },
                rtol=1e-2,
                atol=1e-2,
                expected_uses=expected_uses,
            )
        )

    if _want("dsv4.decode_compressor_ratio4"):
        dsv4_c4 = importlib.import_module("decode_compressor_ratio4")
        cases.append(
            ProductionKernelCase(
                case_id="dsv4.decode_compressor_ratio4",
                category="deepseek_v4_flash_mtp",
                description="DeepSeek-V4-Flash Decode KV/Gate Compressor Ratio-4 (wkv, wgate [512, 4096])",
                jit_fn=dsv4_c4.compressor_test,
                specs=dsv4_c4.build_tensor_specs(start_pos=64),
                golden_fn=dsv4_c4.golden_compressor,
                compare_fn={
                    "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0, ignore_nan=True),
                    "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
                    "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
                },
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    if _want("dsv4.decode_compressor_ratio128"):
        dsv4_c128 = importlib.import_module("decode_compressor_ratio128")
        cases.append(
            ProductionKernelCase(
                case_id="dsv4.decode_compressor_ratio128",
                category="deepseek_v4_flash_mtp",
                description="DeepSeek-V4-Flash Decode KV/Gate Compressor Ratio-128 (wkv, wgate [512, 4096])",
                jit_fn=dsv4_c128.compressor_test,
                specs=dsv4_c128.build_tensor_specs(start_pos=64),
                golden_fn=dsv4_c128.golden_compressor,
                compare_fn={
                    "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0, ignore_nan=True),
                    "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
                    "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
                },
                rtol=1e-3,
                atol=1e-3,
                expected_uses=expected_uses,
            )
        )

    return cases


def _prepare_nd_baseline_program(jit_fn: Any) -> tuple[core_ir.Program, set[str]]:
    """Specialize `jit_fn` with all 2D `pl.NZ` annotations stripped to `pl.ND`.

    Returns:
        `(raw_nd_program, originally_nz_2d_tensor_names)`
    """
    spec, _ = jit_fn._resolve_specialization((), {}, allow_signature_mode=True)
    orig_nz_2d: set[str] = set()
    nd_meta = {}
    for name, meta in spec.tensor_meta.items():
        if meta.layout == TensorLayout.NZ and len(meta.shape) == 2:
            orig_nz_2d.add(name)
            nd_meta[name] = replace(meta, layout=TensorLayout.ND)
        else:
            nd_meta[name] = meta
    prog, _ = jit_fn._compile_to_program_with_rename_map(
        nd_meta,
        spec.scalar_dtypes,
        spec.constexpr_values,
        spec.per_func_dyn,
        pl,
    )
    return prog, orig_nz_2d


def _materialize_canonical_tensors_and_golden(
    case: ProductionKernelCase,
    orig_nz_2d: set[str],
    raw_nd_prog: core_ir.Program,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Materialize logical `ND` input tensors and compute golden outputs once."""
    golden_spec_mod = importlib.import_module("golden.spec")
    ScalarSpec = golden_spec_mod.ScalarSpec
    TensorSpec = golden_spec_mod.TensorSpec

    # Stamp parameter directions from the entry Orchestration function.
    orch_funcs = [f for f in raw_nd_prog.functions.values() if str(f.func_type).endswith("Orchestration")]
    entry = orch_funcs[-1] if orch_funcs else next(reversed(list(raw_nd_prog.functions.values())))
    dir_map = {
        core_ir.ParamDirection.In: "in",
        core_ir.ParamDirection.Out: "out",
        core_ir.ParamDirection.InOut: "inout",
    }
    for spec_item, direction in zip(case.specs, entry.param_directions, strict=True):
        if isinstance(spec_item, TensorSpec):
            spec_item.direction = dir_map.get(direction, "in")

    logical_nd: dict[str, torch.Tensor] = {}
    golden_scratch: dict[str, Any] = {}
    for spec_item in case.specs:
        if isinstance(spec_item, ScalarSpec):
            golden_scratch[spec_item.name] = spec_item.to_python()
            continue
        t = spec_item.create_tensor().clone().contiguous()
        if spec_item.is_input and spec_item.name in orig_nz_2d:
            nd_t = unpack_nz(t).contiguous()
            logical_nd[spec_item.name] = nd_t
            golden_scratch[spec_item.name] = t.clone()
        else:
            logical_nd[spec_item.name] = t
            golden_scratch[spec_item.name] = t.clone()

    case.golden_fn(golden_scratch)
    golden_outputs = {
        s.name: golden_scratch[s.name].clone().contiguous()
        for s in case.specs
        if isinstance(s, TensorSpec) and s.is_output
    }
    return logical_nd, golden_outputs


def _build_variant_specs(
    base_specs: Sequence[Any],
    logical_nd: dict[str, torch.Tensor],
    nz_tensor_names: set[str],
) -> list[Any]:
    """Create `TensorSpec`s whose `init_value` returns `pack_nz(t)` for any tensor in `nz_tensor_names`."""
    golden_spec_mod = importlib.import_module("golden.spec")
    ScalarSpec = golden_spec_mod.ScalarSpec
    TensorSpec = golden_spec_mod.TensorSpec

    variant_specs: list[Any] = []
    for s in base_specs:
        if isinstance(s, ScalarSpec):
            variant_specs.append(s)
            continue
        base_t = logical_nd[s.name]
        if s.is_input and s.name in nz_tensor_names:
            packed_t = pack_nz(base_t).contiguous()
            ts = TensorSpec(
                s.name,
                list(s.shape),
                s.dtype,
                init_value=lambda t=packed_t: t.clone(),
            )
        else:
            ts = TensorSpec(
                s.name,
                list(s.shape),
                s.dtype,
                init_value=lambda t=base_t: t.clone(),
            )
        ts.direction = s.direction
        variant_specs.append(ts)
    return variant_specs


def _extract_bench_metrics(bench_stats: Any) -> tuple[float | None, float | None, float | None]:
    """Extract `(device_us_median, effective_us_median, host_us_median)` from `BenchmarkStats`."""
    if bench_stats is None:
        return None, None, None
    invocations: Sequence[Any] = bench_stats.invocations or ()
    eff_round: Sequence[Any] = bench_stats.per_round("effective") or ()
    host_round: Sequence[Any] = bench_stats.host_wall_us or ()
    dev_samples = [float(iv.device_us) for iv in invocations if float(iv.device_us) > 0.0]
    eff_samples = [float(e) for e in eff_round if float(e) > 0.0]
    host_samples = [float(h) for h in host_round if float(h) > 0.0]
    dev_med = float(statistics.median(dev_samples)) if dev_samples else None
    eff_med = float(statistics.median(eff_samples)) if eff_samples else None
    host_med = float(statistics.median(host_samples)) if host_samples else None
    return dev_med, eff_med, host_med


def _set_or_restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _execute_and_time_variant(  # noqa: PLR0913
    case: ProductionKernelCase,
    v_dir: Path,
    logical_nd: dict[str, torch.Tensor],
    nz_names: set[str],
    golden_fn: Callable[[dict[str, Any]], None],
    *,
    cli_backend: str,
    is_sim: bool,
    device_id: int,
    rounds: int,
    warmup: int,
    capture_swimlane: bool,
    rec: dict[str, Any],
) -> None:
    """Execute, validate against `golden_fn`, and benchmark one compiled variant."""
    golden_run = importlib.import_module("golden").run

    v_specs = _build_variant_specs(case.specs, logical_nd, nz_names)
    old_bench = os.environ.get("PYPTO_BENCH")
    old_rounds = os.environ.get("PYPTO_BENCH_ROUNDS")
    old_warmup = os.environ.get("PYPTO_BENCH_WARMUP")
    try:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = str(warmup)
        run_res = golden_run(
            fn=case.jit_fn,
            specs=v_specs,
            golden_fn=golden_fn,
            compare_fn=case.compare_fn,
            rtol=case.rtol,
            atol=case.atol,
            runtime_dir=str(v_dir),
            config={
                "backend_type": cli_backend,
                "execution_mode": is_sim,
                "device_id": device_id,
                "enable_chip_swimlane": 0,
            },
        )
    finally:
        _set_or_restore_env("PYPTO_BENCH", old_bench)
        _set_or_restore_env("PYPTO_BENCH_ROUNDS", old_rounds)
        _set_or_restore_env("PYPTO_BENCH_WARMUP", old_warmup)

    rec["accuracy_ok"] = bool(run_res.passed)
    if not run_res.passed:
        rec["error"] = run_res.error
    dev_med, eff_med, host_med = _extract_bench_metrics(run_res.bench)
    rec["device_us_median"] = dev_med
    rec["effective_us_median"] = eff_med
    rec["host_us_median"] = host_med

    if capture_swimlane and run_res.passed:
        old_bench_sw = os.environ.get("PYPTO_BENCH")
        try:
            os.environ["PYPTO_BENCH"] = "0"
            sw_specs = _build_variant_specs(case.specs, logical_nd, nz_names)
            golden_run(
                fn=case.jit_fn,
                specs=sw_specs,
                golden_fn=golden_fn,
                compare_fn=case.compare_fn,
                rtol=case.rtol,
                atol=case.atol,
                runtime_dir=str(v_dir),
                config={
                    "backend_type": cli_backend,
                    "execution_mode": is_sim,
                    "device_id": device_id,
                    "enable_chip_swimlane": 1,
                },
            )
            rec["swimlane_aicore_us"] = _extract_swimlane_aicore_us(v_dir)
        finally:
            _set_or_restore_env("PYPTO_BENCH", old_bench_sw)


def run_production_benchmark_suite(  # noqa: PLR0913
    cases: Sequence[ProductionKernelCase],
    *,
    output_dir: str | Path,
    platform: str = "a2a3",
    device_id: int = 0,
    rounds: int = 100,
    warmup: int = 5,
    compile_only: bool = False,
    capture_swimlane: bool = False,
    max_treewidth: int = 8,
) -> dict[str, Any]:
    """Compile, validate, and benchmark `cases` comparing Baseline (`ND`) vs Auto-Layout."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    backend_type, execution_mode = _parse_platform(platform)
    set_backend_type(backend_type)

    is_sim = execution_mode == ExecutionMode.SIM
    cli_backend = "a5" if backend_type == BackendType.Ascend950 else "a2a3"

    case_records: list[dict[str, Any]] = []

    for case in cases:
        print(f"\n=== Production Kernel: {case.case_id} ({case.description}) ===", flush=True)
        case_dir = out_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        raw_nd_prog, orig_nz_2d = _prepare_nd_baseline_program(case.jit_fn)
        logical_nd, golden_outputs = _materialize_canonical_tensors_and_golden(case, orig_nz_2d, raw_nd_prog)

        def _precomputed_golden(
            scratch: dict[str, Any],
            expected: dict[str, torch.Tensor] = golden_outputs,
        ) -> None:
            for k, v in expected.items():
                scratch[k].copy_(v)

        # Calibrated Ascend 910B2 cost model (weights-only GM relayout by default)
        cost_model = LayoutCostModel(
            backend_type=backend_type,
            expected_uses=case.expected_uses,
            mte2_nd2nz_bytes_per_cycle=76.0,
            mte2_nz2nz_bytes_per_cycle=102.0,
            allow_external_input_relayout=False,
        )
        pre_prog = run_pre_layout_passes(raw_nd_prog)
        instance = extract_layout_instance(
            pre_prog,
            cost_model=cost_model,
            reusable_tensors=set(case.reusable_tensors),
        )
        base_bd = evaluate_assignment(instance, LayoutAssignment(value_layouts={}))

        t0 = time.perf_counter()
        tw = compute_treewidth(instance)
        tw_us = (time.perf_counter() - t0) * 1e6

        t0 = time.perf_counter()
        greedy_res = solve_greedy(instance, config=LayoutSolverConfig(max_treewidth=max_treewidth))
        greedy_us = (time.perf_counter() - t0) * 1e6

        t0 = time.perf_counter()
        dp_res = solve_treewidth_dp(instance, config=LayoutSolverConfig(max_treewidth=max_treewidth))
        dp_us = (time.perf_counter() - t0) * 1e6

        t0 = time.perf_counter()
        auto_res = solve_auto_layout(instance, config=LayoutSolverConfig(max_treewidth=max_treewidth))
        auto_us = (time.perf_counter() - t0) * 1e6

        variants_to_run: list[tuple[str, LayoutAssignment | None, float, float]] = [
            ("baseline_nd", None, base_bd.total_cost_cycles, base_bd.operator_cost_cycles),
            (
                "auto_layout",
                auto_res.writeback_assignment(),
                auto_res.total_cost_cycles,
                auto_res.breakdown.operator_cost_cycles,
            ),
        ]

        if case.include_shared_input_variant:
            full_model = replace(cost_model, allow_external_input_relayout=True)
            full_inst = extract_layout_instance(
                pre_prog,
                cost_model=full_model,
                reusable_tensors=set(case.reusable_tensors),
            )
            full_res = solve_auto_layout(full_inst, config=LayoutSolverConfig(max_treewidth=max_treewidth))
            variants_to_run.append(
                (
                    "auto_layout_with_shared_inputs",
                    full_res.writeback_assignment(),
                    full_res.total_cost_cycles,
                    full_res.breakdown.operator_cost_cycles,
                )
            )

        variant_records: dict[str, dict[str, Any]] = {}
        for var_name, assignment, pred_total, pred_op in variants_to_run:
            v_dir = case_dir / var_name
            v_dir.mkdir(parents=True, exist_ok=True)
            nz_names: set[str] = (
                {k for k, v in assignment.tensor_layouts.items() if v == core_ir.TensorLayout.NZ}
                if assignment is not None
                else set()
            )
            zn_tiles: list[str] = (
                [k for k, v in assignment.tile_layouts.items() if v.short_name() == "ZN"]
                if assignment is not None
                else []
            )
            layout_pass_fn = (
                (lambda a=assignment: make_auto_layout_pass(a)) if assignment is not None else None
            )

            compile_program(
                raw_nd_prog,
                output_dir=str(v_dir),
                backend_type=backend_type,
                execution_mode=execution_mode,
                skip_ptoas=compile_only,
                layout_pass=layout_pass_fn,
            )

            rec: dict[str, Any] = {
                "variant": var_name,
                "build_dir": str(v_dir),
                "nz_tensors": sorted(nz_names),
                "zn_tiles": sorted(zn_tiles),
                "predicted_total_cycles": pred_total,
                "predicted_op_cycles": pred_op,
                "accuracy_ok": None,
                "device_us_median": None,
                "effective_us_median": None,
                "host_us_median": None,
                "swimlane_aicore_us": None,
            }

            if not compile_only:
                _execute_and_time_variant(
                    case,
                    v_dir,
                    logical_nd,
                    nz_names,
                    _precomputed_golden,
                    cli_backend=cli_backend,
                    is_sim=is_sim,
                    device_id=device_id,
                    rounds=rounds,
                    warmup=warmup,
                    capture_swimlane=capture_swimlane,
                    rec=rec,
                )

            variant_records[var_name] = rec

        base_rec = variant_records["baseline_nd"]
        auto_rec = variant_records["auto_layout"]
        pred_red_pct = (
            100.0 * (base_bd.total_cost_cycles - auto_res.total_cost_cycles) / base_bd.total_cost_cycles
            if base_bd.total_cost_cycles > 0
            else 0.0
        )

        # Compute measured speedups when device timings are available
        for metric_key in ("device_us_median", "effective_us_median", "swimlane_aicore_us"):
            b_val = base_rec.get(metric_key)
            a_val = auto_rec.get(metric_key)
            if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float)) and a_val > 0:
                auto_rec[f"{metric_key}_delta_vs_nd"] = float(a_val - b_val)
                auto_rec[f"{metric_key}_speedup_vs_nd"] = float(b_val / a_val)
                auto_rec[f"{metric_key}_reduction_pct"] = float(100.0 * (b_val - a_val) / b_val)

        case_records.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "originally_hand_tuned_nz": sorted(orig_nz_2d),
                "predicted_cost_reduction_pct": pred_red_pct,
                "treewidth": {
                    "num_values": tw.num_values,
                    "num_operations": tw.num_operations,
                    "num_movable_operations": tw.num_movable_operations,
                    "dataflow_treewidth": tw.dataflow_treewidth,
                    "primal_treewidth": tw.primal_treewidth,
                    "active_primal_treewidth": tw.active_primal_treewidth,
                    "max_bag_domain_product": tw.max_bag_domain_product,
                    "total_dp_states": tw.total_dp_states,
                    "compute_time_us": tw_us,
                },
                "solvers": {
                    "greedy": {
                        "nz_tensors": sorted(
                            k
                            for k, v in greedy_res.assignment.tensor_layouts.items()
                            if v == core_ir.TensorLayout.NZ
                        ),
                        "predicted_total_cycles": greedy_res.total_cost_cycles,
                        "evaluated_moves": greedy_res.stats.evaluated_moves,
                        "solve_time_us": greedy_us,
                    },
                    "treewidth_dp": {
                        "nz_tensors": sorted(
                            k
                            for k, v in dp_res.assignment.tensor_layouts.items()
                            if v == core_ir.TensorLayout.NZ
                        ),
                        "predicted_total_cycles": dp_res.total_cost_cycles,
                        "evaluated_moves": dp_res.stats.evaluated_moves,
                        "solve_time_us": dp_us,
                    },
                    "auto_layout": {
                        "nz_tensors": sorted(
                            k
                            for k, v in auto_res.assignment.tensor_layouts.items()
                            if v == core_ir.TensorLayout.NZ
                        ),
                        "predicted_total_cycles": auto_res.total_cost_cycles,
                        "termination_reason": auto_res.stats.termination_reason,
                        "solve_time_us": auto_us,
                    },
                },
                "variants": variant_records,
            }
        )

    return {
        "suite": "pypto_lib_production_kernels",
        "platform": platform,
        "backend_type": backend_type.name,
        "execution_mode": execution_mode.name,
        "compile_only": compile_only,
        "rounds": rounds,
        "warmup": warmup,
        "max_treewidth": max_treewidth,
        "num_cases": len(case_records),
        "cases": case_records,
    }


def _print_production_summary(summary: dict[str, Any]) -> None:
    header = (
        f"{'Kernel':31s} | {'tw(A)':>5s} | {'NZ Weights Chosen':34s} | "
        f"{'Pred Red%':>9s} | {'Base eff(us)':>12s} | {'Auto eff(us)':>12s} | "
        f"{'Speedup':>8s} | {'Acc':>4s}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for c in summary["cases"]:
        tw_act = c["treewidth"]["active_primal_treewidth"]
        variants = c["variants"]
        base_v = variants["baseline_nd"]
        auto_v = variants["auto_layout"]
        nz_str = ",".join(auto_v["nz_tensors"]) or "(none)"
        pred_red = c["predicted_cost_reduction_pct"]
        b_eff = base_v.get("effective_us_median") or base_v.get("device_us_median")
        a_eff = auto_v.get("effective_us_median") or auto_v.get("device_us_median")
        b_str = f"{b_eff:.2f}" if isinstance(b_eff, float) and not math.isnan(b_eff) else "compiled"
        a_str = f"{a_eff:.2f}" if isinstance(a_eff, float) and not math.isnan(a_eff) else "compiled"
        sp = auto_v.get("effective_us_median_speedup_vs_nd") or auto_v.get("device_us_median_speedup_vs_nd")
        sp_str = f"{sp:.3f}x" if isinstance(sp, float) else "-"
        acc = auto_v.get("accuracy_ok")
        acc_str = "PASS" if acc is True else ("FAIL" if acc is False else "-")
        print(
            f"{c['case_id']:31s} | {tw_act:5d} | {nz_str:34s} | "
            f"{pred_red:8.1f}% | {b_str:>12s} | {a_str:>12s} | "
            f"{sp_str:>8s} | {acc_str:>4s}"
        )
    print("=" * len(header))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark PyPTO Auto-Layout Pass on pypto-lib production kernels"
    )
    default_lib = Path(__file__).resolve().parents[4] / "pypto-lib"
    parser.add_argument(
        "--pypto-lib-dir",
        default=os.environ.get("PYPTO_LIB_DIR", str(default_lib)),
        help="Path to the pypto-lib repository checkout",
    )
    parser.add_argument("--platform", default="a2a3", help="Target platform (a2a3, a2a3sim, a5, a5sim)")
    parser.add_argument("-d", "--device", type=int, default=0, help="Ascend NPU device ID")
    parser.add_argument("--rounds", type=int, default=100, help="Measured benchmark rounds per variant")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup rounds per variant")
    parser.add_argument("--expected-uses", type=int, default=100, help="Weight reuse factor R")
    parser.add_argument("--max-treewidth", type=int, default=8, help="Treewidth threshold for exact DP")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=None,
        help="Optional case_id filter (may be passed multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        default="build_output/pypto_lib_auto_layout_bench",
        help="Root directory for compiled artifacts",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file path to save the production benchmark report",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile all Baseline and Auto-Layout variants without NPU execution",
    )
    parser.add_argument(
        "--swimlane",
        action="store_true",
        help="Also capture level-1 chip swimlane AICore task durations on device",
    )
    args = parser.parse_args(argv)

    cases = load_pypto_lib_production_suite(
        args.pypto_lib_dir,
        expected_uses=args.expected_uses,
        case_filter=args.cases,
    )
    summary = run_production_benchmark_suite(
        cases,
        output_dir=args.output_dir,
        platform=args.platform,
        device_id=args.device,
        rounds=args.rounds,
        warmup=args.warmup,
        compile_only=args.compile_only,
        capture_swimlane=args.swimlane,
        max_treewidth=args.max_treewidth,
    )
    _print_production_summary(summary)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
