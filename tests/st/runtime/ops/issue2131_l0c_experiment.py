# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure forced L0C double buffering for PyPTO issue #2131.

The baseline uses one accumulator while ``dbc`` opts the inner pipeline into
the existing two-accumulator lowering. Run this script through pypto-lib so
compile artifacts, correctness data, and per-round device timings have the
same format:

    python tests/st/runtime/ops/issue2131_l0c_experiment.py \
        --variant baseline --planner pypto --compile-only

Set ``PYPTO_BENCH=1`` for the timing run. Use
``issue2131_validate_structure.py`` on compile artifacts and
``issue2131_analyze.py`` on replicated timing artifacts.
"""

import argparse
import json
import os
from pathlib import Path

import pypto.language as pl
import torch
from golden import TensorSpec, run_jit
from pypto.pypto_core.passes import MemoryPlanner

STACKS = 4
M = 16
K = 128
STACK_N = 512
L0_N = 128


@pl.jit
def two_level_pingpong_baseline(
    q: pl.Tensor[[M, K], pl.BF16],
    b: pl.Tensor[[STACKS * K, STACK_N], pl.BF16],
    out: pl.Out[pl.Tensor[[STACKS * M, STACK_N], pl.FP32]],
) -> pl.Tensor[[STACKS * M, STACK_N], pl.FP32]:
    for _ in pl.spmd(1, name_hint="two_level_pingpong_baseline"):
        q_l1: pl.Tile[[M, K], pl.BF16, pl.Mem.Mat] = pl.load(
            q,
            [0, 0],
            [M, K],
            target_memory=pl.MemorySpace.Mat,
        )
        q_l0: pl.Tile[[M, K], pl.BF16, pl.Mem.Left] = pl.tile.extract(
            q_l1,
            0,
            0,
            [M, K],
            target_memory=pl.MemorySpace.Left,
        )
        for stack, (out_outer,) in pl.pipeline(STACKS, stage=2, init_values=(out,)):
            b_l1: pl.Tile[[K, STACK_N], pl.BF16, pl.Mem.Mat] = pl.load(
                b,
                [stack * K, 0],
                [K, STACK_N],
                target_memory=pl.MemorySpace.Mat,
            )
            for col, (out_inner,) in pl.pipeline(
                0,
                STACK_N,
                L0_N,
                stage=2,
                init_values=(out_outer,),
            ):
                b_l0: pl.Tile[[K, L0_N], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    b_l1,
                    0,
                    col,
                    [K, L0_N],
                    target_memory=pl.MemorySpace.Right,
                )
                acc: pl.Tile[[M, L0_N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(q_l0, b_l0)
                out_next = pl.store(acc, [stack * M, col], out_inner)
                out_inner_yield = pl.yield_(out_next)
            out_outer_yield = pl.yield_(out_inner_yield)
    return out_outer_yield


@pl.jit
def two_level_pingpong_dbc(
    q: pl.Tensor[[M, K], pl.BF16],
    b: pl.Tensor[[STACKS * K, STACK_N], pl.BF16],
    out: pl.Out[pl.Tensor[[STACKS * M, STACK_N], pl.FP32]],
) -> pl.Tensor[[STACKS * M, STACK_N], pl.FP32]:
    for _ in pl.spmd(1, name_hint="two_level_pingpong_dbc"):
        q_l1: pl.Tile[[M, K], pl.BF16, pl.Mem.Mat] = pl.load(
            q,
            [0, 0],
            [M, K],
            target_memory=pl.MemorySpace.Mat,
        )
        q_l0: pl.Tile[[M, K], pl.BF16, pl.Mem.Left] = pl.tile.extract(
            q_l1,
            0,
            0,
            [M, K],
            target_memory=pl.MemorySpace.Left,
        )
        for stack, (out_outer,) in pl.pipeline(STACKS, stage=2, init_values=(out,)):
            b_l1: pl.Tile[[K, STACK_N], pl.BF16, pl.Mem.Mat] = pl.load(
                b,
                [stack * K, 0],
                [K, STACK_N],
                target_memory=pl.MemorySpace.Mat,
            )
            for col, (out_inner,) in pl.pipeline(
                0,
                STACK_N,
                L0_N,
                stage=2,
                init_values=(out_outer,),
                attrs={
                    "pipeline_overlap_stores": False,
                    "pipeline_double_buffer_c": True,
                },
            ):
                b_l0: pl.Tile[[K, L0_N], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    b_l1,
                    0,
                    col,
                    [K, L0_N],
                    target_memory=pl.MemorySpace.Right,
                )
                acc: pl.Tile[[M, L0_N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(q_l0, b_l0)
                out_next = pl.store(acc, [stack * M, col], out_inner)
                out_inner_yield = pl.yield_(out_next)
            out_outer_yield = pl.yield_(out_inner_yield)
    return out_outer_yield


def _golden(values: dict[str, torch.Tensor]) -> None:
    q = values["q"].float()
    b = values["b"].float()
    out = values["out"]
    for stack in range(STACKS):
        b_stack = b[stack * K : (stack + 1) * K]
        out[stack * M : (stack + 1) * M] = q @ b_stack


def _planner(name: str) -> MemoryPlanner:
    if name == "pypto":
        return MemoryPlanner.PYPTO
    return MemoryPlanner.PTOAS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "dbc"), required=True)
    parser.add_argument("--planner", choices=("pypto", "ptoas"), required=True)
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--output-root", default="build_output/issue2131")
    parser.add_argument("--enable-l2-swimlane", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    specs = [
        TensorSpec("q", [M, K], torch.bfloat16, init_value=torch.randn),
        TensorSpec("b", [STACKS * K, STACK_N], torch.bfloat16, init_value=torch.randn),
        TensorSpec("out", [STACKS * M, STACK_N], torch.float32, is_output=True),
    ]
    fn = two_level_pingpong_baseline if args.variant == "baseline" else two_level_pingpong_dbc
    output_dir = Path(args.output_root) / f"rep{args.replicate}_{args.variant}_{args.planner}_seed{args.seed}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    result = run_jit(
        fn,
        specs,
        golden_fn=_golden,
        golden_data=args.golden_data,
        compile_cfg={
            "dump_passes": True,
            "memory_planner": _planner(args.planner),
            "save_kernels": True,
            "save_kernels_dir": str(output_dir),
        },
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_l2_swimlane": args.enable_l2_swimlane,
        },
        rtol=2e-2,
        atol=2e-2,
        compile_only=args.compile_only,
        save_data=args.save_data,
    )
    summary = {
        "variant": args.variant,
        "planner": args.planner,
        "seed": args.seed,
        "replicate": args.replicate,
        "platform": args.platform,
        "device": args.device,
        "compile_only": args.compile_only,
        "bench_enabled": os.environ.get("PYPTO_BENCH", ""),
        "enable_l2_swimlane": args.enable_l2_swimlane,
        "golden_data": None if args.golden_data is None else str(Path(args.golden_data).resolve()),
        "passed": result.passed,
        "error": result.error,
        "work_dir": None if result.work_dir is None else str(result.work_dir),
    }
    if result.bench is not None:
        summary["benchmark"] = {
            "rounds": result.bench.rounds,
            "warmup": result.bench.warmup,
            "effective_us": result.bench.per_round("effective"),
            "device_wall_us": result.bench.per_round("device"),
            "host_wall_us": result.bench.per_round("host"),
        }
    if result.work_dir is not None:
        result_path = Path(result.work_dir) / "issue2131_result.json"
        result_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"ISSUE2131_RESULT={json.dumps(summary, sort_keys=True)}")
    if not result.passed:
        raise RuntimeError(result.error)


if __name__ == "__main__":
    main()
