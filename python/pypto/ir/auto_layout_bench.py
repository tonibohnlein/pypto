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
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
from pypto.ir.auto_layout_solver import (
    compute_treewidth,
    solve_auto_layout,
    solve_greedy,
    solve_treewidth_dp,
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


@pl.jit.incore
def _matmul_single_tile(
    a: pl.Tensor,
    b: pl.Tensor,
    c: pl.Out[pl.Tensor],
    m_dim: pl.constexpr,
    red_dim: pl.constexpr,
    out_cols: pl.constexpr,
) -> pl.Tensor:
    at = pl.load(a, [0, 0], [m_dim, red_dim], target_memory=pl.MemorySpace.Mat)
    bt = pl.load(b, [0, 0], [out_cols, red_dim], target_memory=pl.MemorySpace.Mat)
    bt_t = pl.tile.transpose_view(bt)
    acc = pl.matmul(at, bt_t)
    pl.store(acc, [0, 0], c)
    return c


@pl.jit.incore
def _matmul_tiled_nk(
    a: pl.Tensor,
    b: pl.Tensor,
    c: pl.Out[pl.Tensor],
    m_dim: pl.constexpr,
    red_dim: pl.constexpr,
    out_cols: pl.constexpr,
    red_tile: pl.constexpr,
    col_tile: pl.constexpr,
) -> pl.Tensor:
    for c0 in pl.range(0, out_cols, col_tile):
        initial = pl.tile.create([m_dim, col_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        for r0, (carry,) in pl.pipeline(0, red_dim, red_tile, init_values=(initial,), stage=2):
            at = pl.load(a, [0, r0], [m_dim, red_tile], target_memory=pl.MemorySpace.Mat)
            bt = pl.load(b, [c0, r0], [col_tile, red_tile], target_memory=pl.MemorySpace.Mat)
            bt_t = pl.tile.transpose_view(bt)
            cur = pl.tile.matmul_acc(carry, at, bt_t, r0 == 0)
            acc = pl.yield_(cur)
        pl.store(acc, [0, c0], c)  # pyright: ignore[reportPossiblyUnboundVariable]
    return c


@pl.jit.incore
def _matmul_tiled_mk(
    a: pl.Tensor,
    b: pl.Tensor,
    c: pl.Out[pl.Tensor],
    m_dim: pl.constexpr,
    red_dim: pl.constexpr,
    out_cols: pl.constexpr,
    m_tile: pl.constexpr,
    red_tile: pl.constexpr,
) -> pl.Tensor:
    for m0 in pl.range(0, m_dim, m_tile):
        initial = pl.tile.create([m_tile, out_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        for r0, (carry,) in pl.pipeline(0, red_dim, red_tile, init_values=(initial,), stage=2):
            at = pl.load(a, [m0, r0], [m_tile, red_tile], target_memory=pl.MemorySpace.Mat)
            bt = pl.load(b, [0, r0], [out_cols, red_tile], target_memory=pl.MemorySpace.Mat)
            bt_t = pl.tile.transpose_view(bt)
            cur = pl.tile.matmul_acc(carry, at, bt_t, r0 == 0)
            acc = pl.yield_(cur)
        pl.store(acc, [m0, 0], c)  # pyright: ignore[reportPossiblyUnboundVariable]
    return c


@pl.jit.incore
def _matmul_stationary_a(
    a: pl.Tensor,
    b: pl.Tensor,
    c: pl.Out[pl.Tensor],
    m_dim: pl.constexpr,
    red_dim: pl.constexpr,
    out_cols: pl.constexpr,
    col_tile: pl.constexpr,
) -> pl.Tensor:
    at = pl.load(a, [0, 0], [m_dim, red_dim], target_memory=pl.MemorySpace.Mat)
    for c0 in pl.range(0, out_cols, col_tile):
        bt = pl.load(b, [c0, 0], [col_tile, red_dim], target_memory=pl.MemorySpace.Mat)
        bt_t = pl.tile.transpose_view(bt)
        acc = pl.matmul(at, bt_t)
        pl.store(acc, [0, c0], c)
    return c


@pl.jit.incore
def _matmul_stationary_a_2step(
    a: pl.Tensor,
    b: pl.Tensor,
    c: pl.Out[pl.Tensor],
    m_dim: pl.constexpr,
    out_cols: pl.constexpr,
    red_tile: pl.constexpr,
    col_tile: pl.constexpr,
) -> pl.Tensor:
    at0 = pl.load(a, [0, 0], [m_dim, red_tile], target_memory=pl.MemorySpace.Mat)
    at1 = pl.load(a, [0, red_tile], [m_dim, red_tile], target_memory=pl.MemorySpace.Mat)
    for c0 in pl.range(0, out_cols, col_tile):
        bt0 = pl.load(b, [c0, 0], [col_tile, red_tile], target_memory=pl.MemorySpace.Mat)
        bt0_t = pl.tile.transpose_view(bt0)
        acc0 = pl.matmul(at0, bt0_t)
        bt1 = pl.load(b, [c0, red_tile], [col_tile, red_tile], target_memory=pl.MemorySpace.Mat)
        bt1_t = pl.tile.transpose_view(bt1)
        acc = pl.tile.matmul_acc(acc0, at1, bt1_t, False)
        pl.store(acc, [0, c0], c)
    return c


def build_single_matmul_case(  # noqa: PLR0913
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    m_tile: int | None = None,
    k_tile: int | None = None,
    n_tile: int | None = None,
    stationary: str = "output",
    case_id: str | None = None,
    expected_uses: int = 100,
    seed: int = 0,
) -> CalibrationCaseSpec:
    """Build an L0-fitting `C[M, N] = X[M, K] @ W[N, K].T` FP16->FP32 `@pl.jit` workload.

    The A-stationary templates support one or two complete K tiles only.
    """
    m_step = m_tile if m_tile is not None else m_dim
    k_step = k_tile if k_tile is not None else k_dim
    n_step = n_tile if n_tile is not None else n_dim
    if stationary == "a" and k_dim not in (k_step, 2 * k_step):
        raise ValueError(
            "stationary='a' supports only one or two K tiles: require k_dim == k_tile "
            f"or k_dim == 2 * k_tile, got k_dim={k_dim}, k_tile={k_step}. "
            "Use stationary='output' for other reduction tilings."
        )
    is_single_tile = m_step == m_dim and k_step == k_dim and n_step == n_dim

    @pl.jit
    def matmul_single_entry(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return _matmul_single_tile(x, w, out, m_dim, k_dim, n_dim)

    @pl.jit
    def matmul_tiled_entry(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return _matmul_tiled_nk(x, w, out, m_dim, k_dim, n_dim, k_step, n_step)

    @pl.jit
    def matmul_tiled_mk_entry(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return _matmul_tiled_mk(x, w, out, m_dim, k_dim, n_dim, m_step, k_step)

    @pl.jit
    def matmul_stationary_a_entry(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return _matmul_stationary_a(x, w, out, m_dim, k_dim, n_dim, n_step)

    @pl.jit
    def matmul_stationary_a_2step_entry(
        x: pl.Tensor,
        w: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return _matmul_stationary_a_2step(x, w, out, m_dim, n_dim, k_step, n_step)

    if stationary == "a":
        entry_fn = matmul_stationary_a_entry if k_step == k_dim else matmul_stationary_a_2step_entry
    elif m_step < m_dim:
        entry_fn = matmul_tiled_mk_entry
    elif is_single_tile:
        entry_fn = matmul_single_entry
    else:
        entry_fn = matmul_tiled_entry

    gen = torch.Generator().manual_seed(seed)
    scale = 0.05 if k_dim >= 1024 else 0.25
    x_host = torch.randn((m_dim, k_dim), generator=gen, dtype=torch.float16) * scale
    w_host = torch.randn((n_dim, k_dim), generator=gen, dtype=torch.float16) * scale
    out_stub = torch.zeros((m_dim, n_dim), dtype=torch.float32)

    program = entry_fn.specialize(x_host, w_host, out_stub)
    expected = x_host.float() @ w_host.float().T

    resolved_id = case_id or f"matmul_{m_dim}x{k_dim}x{n_dim}"
    return CalibrationCaseSpec(
        case_id=resolved_id,
        program=program,
        inputs=(x_host, w_host),
        expected_output=expected,
        output_shape=(m_dim, n_dim),
        output_dtype=torch.float32,
        reusable_tensors={"w"},
        expected_uses=expected_uses,
    )


def build_two_matmul_chain_case(  # noqa: PLR0913
    m_dim: int = 64,
    k_dim: int = 128,
    h_dim: int = 64,
    n_dim: int = 64,
    *,
    k_tile: int | None = None,
    h_tile_s1: int | None = None,
    h_tile_s2: int | None = None,
    n_tile: int | None = None,
    expected_uses: int = 100,
    seed: int = 1,
) -> CalibrationCaseSpec:
    """Build a two-matmul `@pl.jit` chain `Mid = X @ W1.T; Out = Mid @ W2.T`.

    `Mid` (`[M, H]` FP32) is produced in GM by `mm_stage1` and consumed in GM
    by `mm_stage2`, allowing calibration of intra-program `ND` vs. `NZ`
    intermediate GM storage without host packing.
    """
    k_step = k_tile if k_tile is not None else k_dim
    h1_step = h_tile_s1 if h_tile_s1 is not None else h_dim
    h2_step = h_tile_s2 if h_tile_s2 is not None else h_dim
    n_step = n_tile if n_tile is not None else n_dim
    s1_single = k_step == k_dim and h1_step == h_dim
    s2_single = h2_step == h_dim and n_step == n_dim

    if s1_single and s2_single:

        @pl.jit
        def two_matmul_chain_entry(
            x: pl.Tensor,
            w1: pl.Tensor,
            w2: pl.Tensor,
            out: pl.Out[pl.Tensor],
        ) -> pl.Tensor:
            mid = pl.create_tensor([m_dim, h_dim], dtype=pl.FP32)
            mid = _matmul_single_tile(x, w1, mid, m_dim, k_dim, h_dim)
            out = _matmul_single_tile(mid, w2, out, m_dim, h_dim, n_dim)
            return out
    else:

        @pl.jit
        def two_matmul_chain_entry(
            x: pl.Tensor,
            w1: pl.Tensor,
            w2: pl.Tensor,
            out: pl.Out[pl.Tensor],
        ) -> pl.Tensor:
            mid = pl.create_tensor([m_dim, h_dim], dtype=pl.FP32)
            mid = _matmul_tiled_nk(x, w1, mid, m_dim, k_dim, h_dim, k_step, h1_step)
            out = _matmul_tiled_nk(mid, w2, out, m_dim, h_dim, n_dim, h2_step, n_step)
            return out

    gen = torch.Generator().manual_seed(seed)
    s1_scale = 0.05 if k_dim >= 1024 else 0.25
    s2_scale = 0.05 if h_dim >= 256 else 0.25
    x_host = torch.randn((m_dim, k_dim), generator=gen, dtype=torch.float16) * s1_scale
    w1_host = torch.randn((h_dim, k_dim), generator=gen, dtype=torch.float16) * s1_scale
    w2_host = torch.randn((n_dim, h_dim), generator=gen, dtype=torch.float32) * s2_scale
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


@pl.jit.inline
def _qkv_shared_x_sum(  # noqa: PLR0913
    x: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    out: pl.Tensor,
    m_dim: pl.constexpr,
    k_dim: pl.constexpr,
    n_dim: pl.constexpr,
    k_tile: pl.constexpr,
) -> pl.Tensor:
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qkv_shared_x"):
        acc_q = pl.create_tensor([m_dim, n_dim], dtype=pl.FP32)
        for kb in pl.range(0, k_dim // k_tile, 1):
            k0 = kb * k_tile
            xq = pl.slice(x, [m_dim, k_tile], [0, k0])
            wqt = pl.slice(wq, [n_dim, k_tile], [0, k0])
            acc_q = pl.matmul_acc(acc_q, xq, wqt, b_trans=True, init_cond=kb == 0)

        acc_k = pl.create_tensor([m_dim, n_dim], dtype=pl.FP32)
        for kb in pl.range(0, k_dim // k_tile, 1):
            k0 = kb * k_tile
            xk = pl.slice(x, [m_dim, k_tile], [0, k0])
            wkt = pl.slice(wk, [n_dim, k_tile], [0, k0])
            acc_k = pl.matmul_acc(acc_k, xk, wkt, b_trans=True, init_cond=kb == 0)

        acc_v = pl.create_tensor([m_dim, n_dim], dtype=pl.FP32)
        for kb in pl.range(0, k_dim // k_tile, 1):
            k0 = kb * k_tile
            xv = pl.slice(x, [m_dim, k_tile], [0, k0])
            wvt = pl.slice(wv, [n_dim, k_tile], [0, k0])
            acc_v = pl.matmul_acc(acc_v, xv, wvt, b_trans=True, init_cond=kb == 0)

        qk = pl.add(acc_q, acc_k)
        qkv = pl.add(qk, acc_v)
        out = pl.assemble(out, qkv, [0, 0])
    return out


def build_shared_input_qkv_case(  # noqa: PLR0913
    m_dim: int = 64,
    k_dim: int = 1024,
    n_dim: int = 64,
    *,
    k_tile: int = 128,
    expected_uses: int = 100,
    seed: int = 9,
) -> CalibrationCaseSpec:
    """Build a 3-projection shared-input workload `Out = X@Wq.T + X@Wk.T + X@Wv.T`.

    `x` is a non-reusable activation consumed by three separate K-reduction
    matmuls, creating the shared-conversion barrier where single-consumer greedy
    flips leave `x` in `ND` while treewidth DP jointly flips all three loads to `NZ`.
    """

    @pl.jit
    def qkv_shared_x_entry(
        x: pl.Tensor,
        wq: pl.Tensor,
        wk: pl.Tensor,
        wv: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        out = _qkv_shared_x_sum(x, wq, wk, wv, out, m_dim, k_dim, n_dim, k_tile)
        return out

    gen = torch.Generator().manual_seed(seed)
    scale = 0.05 if k_dim >= 1024 else 0.25
    x_host = torch.randn((m_dim, k_dim), generator=gen, dtype=torch.float16) * scale
    wq_host = torch.randn((n_dim, k_dim), generator=gen, dtype=torch.float16) * scale
    wk_host = torch.randn((n_dim, k_dim), generator=gen, dtype=torch.float16) * scale
    wv_host = torch.randn((n_dim, k_dim), generator=gen, dtype=torch.float16) * scale
    out_stub = torch.zeros((m_dim, n_dim), dtype=torch.float32)

    program = qkv_shared_x_entry.specialize(x_host, wq_host, wk_host, wv_host, out_stub)
    expected = (
        (x_host.float() @ wq_host.float().T)
        + (x_host.float() @ wk_host.float().T)
        + (x_host.float() @ wv_host.float().T)
    )

    return CalibrationCaseSpec(
        case_id=f"qkv_shared_x_{m_dim}x{k_dim}x{n_dim}",
        program=program,
        inputs=(x_host, wq_host, wk_host, wv_host),
        expected_output=expected,
        output_shape=(m_dim, n_dim),
        output_dtype=torch.float32,
        reusable_tensors={"wq", "wk", "wv"},
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


def _sum_merged_swimlane_us(dfx_dir: Path) -> float | None:
    """Sum `kernel-duration-us` across AIC task events in `merged_swimlane_*.json`."""
    merged_files = sorted(dfx_dir.glob("merged_swimlane_*.json"))
    if not merged_files:
        return None
    try:
        merged_obj: Any = json.loads(merged_files[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(merged_obj, dict):
        return None
    raw_trace: object = cast(dict[str, object], merged_obj).get("traceEvents")
    if not isinstance(raw_trace, list):
        return None
    kern_us: list[float] = []
    for ev in cast(list[object], raw_trace):
        if not isinstance(ev, dict):
            continue
        ev_dict = cast(dict[str, object], ev)
        if ev_dict.get("ph") != "X" or ev_dict.get("cat") != "event":
            continue
        args_obj = ev_dict.get("args")
        k_dur = (
            cast(dict[str, object], args_obj).get("kernel-duration-us")
            if isinstance(args_obj, dict)
            else None
        )
        dur = k_dur if isinstance(k_dur, (int, float)) and k_dur > 0 else ev_dict.get("dur")
        if isinstance(dur, (int, float)) and dur > 0:
            kern_us.append(float(dur))
    return float(sum(kern_us)) if kern_us else None


def _sum_chip_records_us(dfx_dir: Path) -> float | None:
    """Sum AICore task durations (µs) from `chip_swimlane_records.json`."""
    records_path = dfx_dir / "chip_swimlane_records.json"
    if not records_path.is_file():
        return None
    try:
        payload_obj: Any = json.loads(records_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    events_obj: object
    if isinstance(payload_obj, dict):
        payload_dict = cast(dict[str, object], payload_obj)
        meta_obj = payload_dict.get("metadata")
        hz = cast(dict[str, object], meta_obj).get("clock_freq_hz") if isinstance(meta_obj, dict) else None
        tasks_obj = payload_dict.get("aicore_tasks")
        if isinstance(tasks_obj, list) and isinstance(hz, (int, float)) and float(hz) > 0:
            task_us: list[float] = []
            for raw_task in cast(list[object], tasks_obj):
                if not isinstance(raw_task, list):
                    continue
                row = cast(list[object], raw_task)
                if len(row) < 5:
                    continue
                start, end = row[3], row[4]
                if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                    task_us.append((float(end) - float(start)) * 1e6 / float(hz))
            if task_us:
                return float(sum(task_us))
        events_obj = payload_dict.get("traceEvents") or payload_dict.get("events")
    else:
        events_obj = payload_obj

    if not isinstance(events_obj, list):
        return None
    durations_us: list[float] = []
    for raw_ev in cast(list[object], events_obj):
        if not isinstance(raw_ev, dict):
            continue
        ev_dict = cast(dict[str, object], raw_ev)
        dur = ev_dict.get("dur")
        if ev_dict.get("ph") == "X" and isinstance(dur, (int, float)) and dur > 0:
            durations_us.append(float(dur))
    return float(sum(durations_us)) if durations_us else None


def _extract_swimlane_aicore_us(work_dir: Path) -> float | None:
    """Read total AICore kernel duration (µs) from level-1 swimlane dumps."""
    dfx_dir = work_dir / "dfx_outputs"
    merged_us = _sum_merged_swimlane_us(dfx_dir)
    if merged_us is not None:
        return merged_us
    return _sum_chip_records_us(dfx_dir)


def default_a2_calibration_suite(
    *,
    include_chain: bool = True,
    include_variations: bool = True,
    include_shared_qkv: bool = False,
    expected_uses: int = 100,
) -> list[CalibrationCaseSpec]:
    """Return a high-volume L0-fitting Ascend A2 (`Ascend910`) calibration workload suite.

    Each inner loop step keeps operand tiles <= 32 KiB in L0A/L0B (bypassing
    `AutoTileMatmulL0`) while outer N/K/M tiling amplifies GM/L1 MTE2 loads and
    L0C->GM FIXPIPE stores across:
    - Problem-size & aspect-ratio sweeps (`0.5-4.0 MiB` per launch),
    - Inner `k_tile` granularity sweeps (`64`, `128`, `256` -> `128 B`, `256 B`, `512 B` row width)
      at fixed `(M, K, N) = (64, 4096, 64)`, and
    - Stationarity / loop-order variations (L1-hoisted A-stationary `x` vs output-stationary
      outer-`N` vs outer-`M` B-reused `w`).
    """
    cases = [
        # MTE2 K-accumulation sweep (loads 0.5 MiB -> 2.0 MiB, stores 16-64 KiB)
        build_single_matmul_case(64, 2048, 64, k_tile=256, expected_uses=expected_uses, seed=0),
        build_single_matmul_case(64, 8192, 64, k_tile=256, expected_uses=expected_uses, seed=1),
        build_single_matmul_case(128, 4096, 128, k_tile=128, expected_uses=expected_uses, seed=2),
        # FIXPIPE store-heavy & balanced N/K-tiled matmuls (stores 512 KiB - 1.0 MiB)
        build_single_matmul_case(64, 256, 2048, k_tile=256, n_tile=64, expected_uses=expected_uses, seed=3),
        build_single_matmul_case(128, 256, 2048, k_tile=128, n_tile=128, expected_uses=expected_uses, seed=4),
        build_single_matmul_case(
            128, 1024, 1024, k_tile=128, n_tile=128, expected_uses=expected_uses, seed=5
        ),
    ]
    if include_variations:
        cases.extend(
            [
                # Tile inner-width sweep on identical 64x4096x64 tensors (128B vs 256B vs 512B L2 line)
                build_single_matmul_case(
                    64,
                    4096,
                    64,
                    k_tile=64,
                    case_id="matmul_64x4096x64_kt64",
                    expected_uses=expected_uses,
                    seed=7,
                ),
                build_single_matmul_case(
                    64,
                    4096,
                    64,
                    k_tile=128,
                    case_id="matmul_64x4096x64_kt128",
                    expected_uses=expected_uses,
                    seed=7,
                ),
                build_single_matmul_case(
                    64,
                    4096,
                    64,
                    k_tile=256,
                    case_id="matmul_64x4096x64_kt256",
                    expected_uses=expected_uses,
                    seed=7,
                ),
                # Stationarity variations: L1-hoisted A-stationary (`x` loaded once) & outer-M (`w` reused)
                build_single_matmul_case(
                    64,
                    256,
                    2048,
                    k_tile=256,
                    n_tile=64,
                    stationary="a",
                    case_id="matmul_64x256x2048_stationary_a",
                    expected_uses=expected_uses,
                    seed=3,
                ),
                build_single_matmul_case(
                    128,
                    256,
                    2048,
                    k_tile=128,
                    n_tile=128,
                    stationary="a",
                    case_id="matmul_128x256x2048_stationary_a",
                    expected_uses=expected_uses,
                    seed=4,
                ),
                build_single_matmul_case(
                    1024,
                    1024,
                    128,
                    m_tile=128,
                    k_tile=128,
                    case_id="matmul_1024x1024x128_outer_m",
                    expected_uses=expected_uses,
                    seed=8,
                ),
            ]
        )
    if include_chain:
        # Large chained matmul moving 128 KiB into GM `mid` and reading 2.0 MiB back out.
        cases.append(
            build_two_matmul_chain_case(
                64,
                2048,
                512,
                1024,
                k_tile=256,
                h_tile_s1=64,
                h_tile_s2=128,
                n_tile=64,
                expected_uses=expected_uses,
                seed=6,
            )
        )
    if include_shared_qkv:
        cases.append(
            build_shared_input_qkv_case(
                64,
                1024,
                64,
                k_tile=128,
                expected_uses=expected_uses,
                seed=9,
            )
        )
    return cases


def _match_variant_record(
    variant_records: Sequence[dict[str, Any]],
    gm_layouts: dict[str, str],
) -> dict[str, Any] | None:
    for rec in variant_records:
        if rec.get("gm_layouts") == gm_layouts:
            return rec
    return None


def _evaluate_solvers_for_case(
    instance: LayoutProblemInstance,
    variant_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Run treewidth and layout solvers on `instance` and match their selected variants."""
    t0 = time.perf_counter()
    tw = compute_treewidth(instance)
    tw_us = (time.perf_counter() - t0) * 1e6

    t0 = time.perf_counter()
    greedy_res = solve_greedy(instance)
    greedy_us = (time.perf_counter() - t0) * 1e6

    t0 = time.perf_counter()
    dp_res = solve_treewidth_dp(instance)
    dp_us = (time.perf_counter() - t0) * 1e6

    t0 = time.perf_counter()
    auto_res = solve_auto_layout(instance)
    auto_us = (time.perf_counter() - t0) * 1e6

    nd_layouts = {
        name: "ND" for name, node in instance.values.items() if node.memory_space == core_ir.MemorySpace.DDR
    }
    base_rec = _match_variant_record(variant_records, nd_layouts)
    base_dev_us = base_rec.get("device_us_median") if base_rec else None
    base_swim_us = base_rec.get("swimlane_aicore_us") if base_rec else None

    def _pack_solver_entry(res: Any, solve_us: float) -> dict[str, Any]:
        gm_lay = {k: v.name for k, v in res.assignment.tensor_layouts.items()}
        matched = _match_variant_record(variant_records, gm_lay)
        dev_us = matched.get("device_us_median") if matched else None
        eff_us = matched.get("effective_us_median") if matched else None
        swim_us = matched.get("swimlane_aicore_us") if matched else None
        delta_dev = (
            float(dev_us - base_dev_us)
            if isinstance(dev_us, (int, float)) and isinstance(base_dev_us, (int, float))
            else None
        )
        delta_swim = (
            float(swim_us - base_swim_us)
            if isinstance(swim_us, (int, float)) and isinstance(base_swim_us, (int, float))
            else None
        )
        return {
            "gm_layouts": gm_lay,
            "variant_id": matched.get("variant_id") if matched else None,
            "predicted_total_cycles": res.total_cost_cycles,
            "predicted_op_cycles": res.breakdown.operator_cost_cycles,
            "predicted_conv_cycles": res.breakdown.conversion_cost_cycles,
            "evaluated_moves": res.stats.evaluated_moves,
            "termination_reason": res.stats.termination_reason,
            "solve_time_us": solve_us,
            "device_us_median": dev_us,
            "effective_us_median": eff_us,
            "swimlane_aicore_us": swim_us,
            "device_delta_vs_nd_us": delta_dev,
            "swimlane_delta_vs_nd_us": delta_swim,
        }

    greedy_entry = _pack_solver_entry(greedy_res, greedy_us)
    dp_entry = _pack_solver_entry(dp_res, dp_us)
    auto_entry = _pack_solver_entry(auto_res, auto_us)
    g_dev = greedy_entry.get("device_us_median")
    dp_dev = dp_entry.get("device_us_median")
    if isinstance(g_dev, (int, float)) and isinstance(dp_dev, (int, float)):
        dp_entry["device_delta_vs_greedy_us"] = float(dp_dev - g_dev)

    return {
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
        "baseline_nd": {
            "gm_layouts": nd_layouts,
            "variant_id": base_rec.get("variant_id") if base_rec else None,
            "predicted_total_cycles": base_rec.get("predicted_total_cycles") if base_rec else None,
            "predicted_op_cycles": base_rec.get("predicted_op_cycles") if base_rec else None,
            "device_us_median": base_dev_us,
            "effective_us_median": base_rec.get("effective_us_median") if base_rec else None,
            "swimlane_aicore_us": base_swim_us,
        },
        "greedy": greedy_entry,
        "treewidth_dp": dp_entry,
        "auto_layout": auto_entry,
    }


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
        "effective_freq_fallback": report.effective_freq_fallback,
        "raw_freq_ghz": report.raw_freq_ghz,
        "r2": report.r2,
        "within_case_r2": report.within_case_r2,
        "spearman_rho": report.spearman_rho,
        "within_case_op_spearman_rho": report.within_case_op_spearman_rho,
        "pooled_op_spearman_rho": report.pooled_op_spearman_rho,
        "mape_percent": report.mape_percent,
        "within_case_mte2_delta_us_per_mb": report.within_case_mte2_delta_us_per_mb,
        "within_case_fixpipe_delta_us_per_mb": report.within_case_fixpipe_delta_us_per_mb,
        "fitted_params": {
            "mte2_nd2nz_bytes_per_cycle": report.fitted_model.mte2_nd2nz_bytes_per_cycle,
            "mte2_nd2nz_contiguous_bytes_per_cycle": (
                report.fitted_model.mte2_nd2nz_contiguous_bytes_per_cycle
            ),
            "mte2_nz2nz_bytes_per_cycle": report.fitted_model.mte2_nz2nz_bytes_per_cycle,
            "fixpipe_acc_to_nd_bytes_per_cycle": report.fitted_model.fixpipe_acc_to_nd_bytes_per_cycle,
            "fixpipe_acc_to_nz_bytes_per_cycle": report.fitted_model.fixpipe_acc_to_nz_bytes_per_cycle,
        },
        "fallback_params": list(report.fallback_params),
        "param_t_stats": report.param_t_stats,
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
    effective_measurements: list[CalibrationMeasurement] = []
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
        model_for_case = replace(
            base_model,
            expected_uses=case.expected_uses,
            backend_type=backend_type,
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
                if ok and eff_median > 0.0:
                    effective_measurements.append(
                        CalibrationMeasurement(
                            variant_id=f"{case.case_id}/{variant.variant_id}",
                            measured_us=eff_median,
                            instance=instance,
                            breakdown=variant.breakdown,
                            label=f"{case.case_id}:{variant.gm_input_layouts}->{variant.gm_output_layouts}",
                        )
                    )
                if capture_swimlane:
                    pending_swimlanes.append((case, instance, variant, compiled, all_args, out_buf, ok, rec))

            variant_records.append(rec)

        suite_records.append(
            {
                "case_id": case.case_id,
                "variants": variant_records,
                "_instance": instance,
            }
        )

    # Pass B: level-1 chip swimlane capture after all benchmark() invocations complete.
    if capture_swimlane and not compile_only:
        warm_cfg = RunConfig(
            backend_type=backend_type,
            execution_mode=execution_mode,
            device_id=device_id,
        )
        swimlane_cfg = RunConfig(
            backend_type=backend_type,
            execution_mode=execution_mode,
            device_id=device_id,
            enable_chip_swimlane=1,
        )
        swimlane_warmup = min(warmup, 2)
        for case, instance, variant, compiled, all_args, out_buf, ok, rec in pending_swimlanes:
            for _ in range(swimlane_warmup):
                out_buf.zero_()
                compiled(*all_args, config=warm_cfg)
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

    for case_rec in suite_records:
        inst = cast(LayoutProblemInstance, case_rec.pop("_instance"))
        case_rec["solver_evaluation"] = _evaluate_solvers_for_case(inst, case_rec["variants"])

    calibration = _serialize_calibration_report(measurements, base_model)
    calibration_effective = _serialize_calibration_report(effective_measurements, base_model)
    calibration_swimlane = _serialize_calibration_report(swimlane_measurements, base_model)

    summary: dict[str, Any] = {
        "platform": platform,
        "device_id": device_id,
        "rounds": rounds,
        "warmup": warmup,
        "compile_only": compile_only,
        "cases": suite_records,
        "calibration": calibration,
        "calibration_effective": calibration_effective,
        "calibration_swimlane": calibration_swimlane,
    }
    return summary


def _print_calibration_block(title: str, cal: dict[str, Any] | None) -> None:
    if cal is None:
        return
    print(f"\n=== {title} ===")
    raw_f = cal.get("raw_freq_ghz", 0.0)
    freq_note = f" (nominal fallback; raw={raw_f:.3f}GHz)" if cal.get("effective_freq_fallback") else ""
    print(
        f"  measurements={cal['num_measurements']} R2={cal['r2']:.4f} "
        f"within_case_R2={cal.get('within_case_r2', 1.0):.4f} "
        f"Spearman={cal['spearman_rho']:.4f} "
        f"within_case_op_Spearman={cal.get('within_case_op_spearman_rho', 1.0):.4f} "
        f"MAPE={cal['mape_percent']:.2f}% "
        f"overhead={cal['fixed_overhead_us']:.2f}us eff_freq={cal['effective_freq_ghz']:.3f}GHz{freq_note}"
    )
    print(
        f"  within_case_deltas_us_per_mb={{'mte2_nd_minus_nz': "
        f"{cal.get('within_case_mte2_delta_us_per_mb', 0.0):+.4f}, "
        f"'fixpipe_nd_minus_nz': {cal.get('within_case_fixpipe_delta_us_per_mb', 0.0):+.4f}}}"
    )
    print(f"  fitted_params={cal['fitted_params']}")
    if cal.get("param_t_stats"):
        print(f"  param_t_stats={cal['param_t_stats']}")
    if cal.get("fallback_params"):
        print(f"  WARNING unfitted fallback_params={cal['fallback_params']}")


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
        se = case.get("solver_evaluation")
        if isinstance(se, dict):
            se_dict = cast(dict[str, dict[str, Any]], se)
            tw = se_dict.get("treewidth", {})
            g = se_dict.get("greedy", {})
            dp = se_dict.get("treewidth_dp", {})
            al = se_dict.get("auto_layout", {})
            print(
                f"  [solvers] tw=(df={tw.get('dataflow_treewidth')}, pr={tw.get('primal_treewidth')}, "
                f"act={tw.get('active_primal_treewidth')}, states={tw.get('total_dp_states')}) "
                f"greedy={g.get('variant_id')}({g.get('gm_layouts')}, dev={g.get('device_us_median')}) "
                f"dp={dp.get('variant_id')}({dp.get('gm_layouts')}, dev={dp.get('device_us_median')}) "
                f"auto={al.get('variant_id')}({al.get('termination_reason')})"
            )
    _print_calibration_block("Calibrated LayoutCostModel (device_us_median)", summary.get("calibration"))
    _print_calibration_block(
        "Calibrated LayoutCostModel (effective_us_median)", summary.get("calibration_effective")
    )
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
    parser.add_argument(
        "--include-shared-qkv",
        action="store_true",
        help="Include the 3-projection shared-input QKV case (tests DP vs greedy barrier)",
    )
    args = parser.parse_args(argv)

    cases = default_a2_calibration_suite(
        include_chain=not args.no_chain,
        include_shared_qkv=args.include_shared_qkv,
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
