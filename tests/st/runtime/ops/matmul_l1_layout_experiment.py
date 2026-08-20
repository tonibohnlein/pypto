# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure how an L1/Mat tile's fractal view affects AutoTile matmul cost.

A natural GM->Mat load has the NZ tile view. ``tile.transpose_view`` is a
zero-copy reinterpretation of the same L1 buffer as ZN. The four arms below
present all NZ/ZN combinations to otherwise identical BF16 matmuls:

* A=NZ, B=ZN: both L1 views match their cube operand roles;
* A=NZ, B=NZ: Right extraction performs the layout conversion;
* A=ZN, B=ZN: Left extraction performs the layout conversion;
* A=ZN, B=NZ: both extractions perform the layout conversion.

The script first compiles every arm and rejects the experiment unless all arms
have the same L0 tile signature and operation counts. On device it then checks
numerics and benchmarks the four already-compiled orchestration functions in a
position-balanced order. Raw per-launch timing is retained in ``results.json``.

Example::

    python tests/st/runtime/ops/matmul_l1_layout_experiment.py --list
    python tests/st/runtime/ops/matmul_l1_layout_experiment.py \
        --platform a2a3 --planner dsa_rp --run-device --device 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch
from pypto import ir
from pypto.ir.pass_manager import PassDumpLevel
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig, benchmark


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    k: int
    n: int


@dataclass(frozen=True)
class Arm:
    name: str
    orchestration: str
    a_layout: str
    b_layout: str


SHAPES = (
    Shape("skinny_m", 32, 512, 128),
    Shape("skinny_n", 128, 512, 32),
    Shape("k_heavy", 128, 512, 128),
    Shape("mn_tiled", 256, 128, 256),
)

# The first arm is the role-aligned reference on A2/A3 and A5: NZ->Left and
# ZN->Right avoid the transpose variant of their respective TEXTRACT path.
ARMS = (
    Arm("a_nz_b_zn", "orch_a_nz_b_zn", "NZ", "ZN"),
    Arm("a_nz_b_nz", "orch_a_nz_b_nz", "NZ", "NZ"),
    Arm("a_zn_b_zn", "orch_a_zn_b_zn", "ZN", "ZN"),
    Arm("a_zn_b_nz", "orch_a_zn_b_nz", "ZN", "NZ"),
)

_PLANNERS = {
    "pypto": MemoryPlanner.PYPTO,
    "dsa_rp": MemoryPlanner.DSA_RP,
    "ptoas": MemoryPlanner.PTOAS,
}
_LAYOUT_FIELDS = {
    "NZ": ("col_major", "row_major"),
    "ZN": ("row_major", "col_major"),
}
_TYPE_FIELD_RE = re.compile(r"\b(blayout|slayout)=([a-z_]+)")
_TILE_SHAPE_RE = re.compile(r"loc=(left|right|acc).*?rows=(\d+), cols=(\d+)")


def _build_program(shape: Shape) -> Any:
    """Build one multi-orchestration program containing all four layout arms."""
    m, k, n = shape.m, shape.k, shape.n

    @pl.program
    class L1LayoutMatmul:
        @pl.function(type=pl.FunctionType.InCore)
        def gemm_a_nz_b_zn(
            self,
            a: pl.Tensor[[m, k], pl.BF16],
            b: pl.Tensor[[n, k], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            a_nz = pl.load(a, [0, 0], [m, k], target_memory=pl.Mem.Mat)
            b_nz = pl.load(b, [0, 0], [n, k], target_memory=pl.Mem.Mat)
            b_zn = pl.tile.transpose_view(b_nz)
            return pl.store(pl.matmul(a_nz, b_zn, out_dtype=pl.FP32), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch_a_nz_b_zn(
            self,
            a: pl.Tensor[[m, k], pl.BF16],
            b: pl.Tensor[[n, k], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            return self.gemm_a_nz_b_zn(a, b, out)

        @pl.function(type=pl.FunctionType.InCore)
        def gemm_a_nz_b_nz(
            self,
            a: pl.Tensor[[m, k], pl.BF16],
            b: pl.Tensor[[k, n], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            a_nz = pl.load(a, [0, 0], [m, k], target_memory=pl.Mem.Mat)
            b_nz = pl.load(b, [0, 0], [k, n], target_memory=pl.Mem.Mat)
            return pl.store(pl.matmul(a_nz, b_nz, out_dtype=pl.FP32), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch_a_nz_b_nz(
            self,
            a: pl.Tensor[[m, k], pl.BF16],
            b: pl.Tensor[[k, n], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            return self.gemm_a_nz_b_nz(a, b, out)

        @pl.function(type=pl.FunctionType.InCore)
        def gemm_a_zn_b_zn(
            self,
            a: pl.Tensor[[k, m], pl.BF16],
            b: pl.Tensor[[n, k], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            a_nz = pl.load(a, [0, 0], [k, m], target_memory=pl.Mem.Mat)
            a_zn = pl.tile.transpose_view(a_nz)
            b_nz = pl.load(b, [0, 0], [n, k], target_memory=pl.Mem.Mat)
            b_zn = pl.tile.transpose_view(b_nz)
            return pl.store(pl.matmul(a_zn, b_zn, out_dtype=pl.FP32), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch_a_zn_b_zn(
            self,
            a: pl.Tensor[[k, m], pl.BF16],
            b: pl.Tensor[[n, k], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            return self.gemm_a_zn_b_zn(a, b, out)

        @pl.function(type=pl.FunctionType.InCore)
        def gemm_a_zn_b_nz(
            self,
            a: pl.Tensor[[k, m], pl.BF16],
            b: pl.Tensor[[k, n], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            a_nz = pl.load(a, [0, 0], [k, m], target_memory=pl.Mem.Mat)
            a_zn = pl.tile.transpose_view(a_nz)
            b_nz = pl.load(b, [0, 0], [k, n], target_memory=pl.Mem.Mat)
            return pl.store(pl.matmul(a_zn, b_nz, out_dtype=pl.FP32), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch_a_zn_b_nz(
            self,
            a: pl.Tensor[[k, m], pl.BF16],
            b: pl.Tensor[[k, n], pl.BF16],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            return self.gemm_a_zn_b_nz(a, b, out)

    return L1LayoutMatmul


def _layout_from_tile_type(tile_type: str) -> tuple[str, str] | None:
    fields = dict(_TYPE_FIELD_RE.findall(tile_type))
    if "blayout" not in fields or "slayout" not in fields:
        return None
    return fields["blayout"], fields["slayout"]


def _pto_structure(build_dir: Path) -> dict[str, Any]:
    pto_files = sorted(build_dir.rglob("*.pto"))
    if not pto_files:
        raise RuntimeError(f"no .pto files found below {build_dir}")
    text = "\n".join(path.read_text(encoding="utf-8") for path in pto_files)
    textract_lines = [line.strip() for line in text.splitlines() if "pto.textract " in line]
    source_layouts: dict[str, set[tuple[str, str]]] = {"left": set(), "right": set()}
    for line in textract_lines:
        if "loc=mat" not in line:
            continue
        before_outs, _, outs = line.partition(" outs(")
        source_type = before_outs[before_outs.find("!pto.tile_buf<loc=mat") :]
        layout = _layout_from_tile_type(source_type)
        for destination, layouts in source_layouts.items():
            if f"loc={destination}" in outs and layout is not None:
                layouts.add(layout)

    tile_signatures: set[tuple[str, int, int]] = set()
    for line in text.splitlines():
        if "pto.tmatmul " not in line and "pto.tmatmul.acc " not in line:
            continue
        for memory, rows, cols in _TILE_SHAPE_RE.findall(line):
            tile_signatures.add((memory, int(rows), int(cols)))

    return {
        "pto_files": [str(path) for path in pto_files],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_layouts": {
            memory: [list(layout) for layout in sorted(layouts)] for memory, layouts in source_layouts.items()
        },
        "tile_signatures": [list(signature) for signature in sorted(tile_signatures)],
        "counts": {
            "textract_left": sum("loc=left" in line.partition(" outs(")[2] for line in textract_lines),
            "textract_right": sum("loc=right" in line.partition(" outs(")[2] for line in textract_lines),
            "tmatmul": text.count("pto.tmatmul "),
            "tmatmul_acc": text.count("pto.tmatmul.acc "),
            "scf_for": text.count("scf.for "),
        },
    }


def _verify_structures(compiled: Any) -> dict[str, Any]:
    structures: dict[str, Any] = {}
    reference_signature: tuple[Any, Any] | None = None
    for arm in ARMS:
        sub_build = Path(compiled[arm.orchestration].output_dir)
        structure = _pto_structure(sub_build)
        structures[arm.name] = structure

        expected_left = _LAYOUT_FIELDS[arm.a_layout]
        expected_right = _LAYOUT_FIELDS[arm.b_layout]
        observed_left = {tuple(value) for value in structure["source_layouts"]["left"]}
        observed_right = {tuple(value) for value in structure["source_layouts"]["right"]}
        if observed_left != {expected_left}:
            raise AssertionError(f"{arm.name}: Left sources {observed_left}, expected {expected_left}")
        if observed_right != {expected_right}:
            raise AssertionError(f"{arm.name}: Right sources {observed_right}, expected {expected_right}")

        signature = (structure["tile_signatures"], structure["counts"])
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise AssertionError(
                f"{arm.name}: AutoTile structure differs from {ARMS[0].name}: "
                f"{signature} != {reference_signature}"
            )
    return structures


def _arm_inputs(
    arm: Arm,
    canonical_a: torch.Tensor,
    canonical_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = canonical_a if arm.a_layout == "NZ" else canonical_a.T.contiguous()
    b = canonical_b.T.contiguous() if arm.b_layout == "ZN" else canonical_b
    return a, b


def _balanced_orders(replicates: int) -> list[list[Arm]]:
    if replicates <= 0:
        raise ValueError(f"--replicates must be positive, got {replicates}")
    if replicates % (2 * len(ARMS)) != 0:
        raise ValueError(f"--replicates must be a multiple of {2 * len(ARMS)} for position balance")
    orders: list[list[Arm]] = []
    for rep in range(replicates):
        rotation = (rep // 2) % len(ARMS)
        order = list(ARMS[rotation:] + ARMS[:rotation])
        if rep % 2:
            order.reverse()
        orders.append(order)
    return orders


def _bootstrap_median_ci(values: list[float], *, samples: int = 20_000, seed: int = 0) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(samples))
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def _summarize_replicates(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    reference = ARMS[0].name
    for metric in ("effective_us", "device_wall_us"):
        reference_by_rep = {row["replicate"]: row["medians"][reference][metric] for row in replicates}
        metric_summary: dict[str, Any] = {}
        for arm in ARMS:
            medians = [row["medians"][arm.name][metric] for row in replicates]
            penalties = [
                100.0 * (row["medians"][arm.name][metric] / reference_by_rep[row["replicate"]] - 1.0)
                for row in replicates
            ]
            metric_summary[arm.name] = {
                "median_us": statistics.median(medians),
                "penalty_vs_role_aligned_percent": statistics.median(penalties),
                "penalty_95ci_percent": _bootstrap_median_ci(penalties),
                "replicate_penalties_percent": penalties,
            }
        summary[metric] = metric_summary
    return summary


def _run_shape(
    compiled: Any,
    shape: Shape,
    *,
    platform: str,
    device: int,
    rounds: int,
    warmup: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    canonical_a = torch.randn((shape.m, shape.k), dtype=torch.bfloat16)
    canonical_b = torch.randn((shape.k, shape.n), dtype=torch.bfloat16)
    expected = canonical_a.float() @ canonical_b.float()
    config = RunConfig(platform=platform, device_id=device)

    correctness: dict[str, Any] = {}
    arm_args: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for arm in ARMS:
        a, b = _arm_inputs(arm, canonical_a, canonical_b)
        out = torch.zeros((shape.m, shape.n), dtype=torch.float32)
        compiled[arm.orchestration](a, b, out, config=config)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
        correctness[arm.name] = {
            "max_abs": (out - expected).abs().max().item(),
            "max_rel": ((out - expected).abs() / expected.abs().clamp_min(1e-6)).max().item(),
        }
        arm_args[arm.name] = (a, b, out)

    replicate_rows: list[dict[str, Any]] = []
    for replicate, order in enumerate(_balanced_orders(replicates)):
        raw: dict[str, Any] = {}
        medians: dict[str, Any] = {}
        for arm in order:
            stats = benchmark(
                compiled[arm.orchestration],
                arm_args[arm.name],
                rounds=rounds,
                warmup=warmup,
                config=config,
            )
            effective = stats.per_round("effective")
            device_wall = stats.per_round("device")
            raw[arm.name] = {
                "effective_us": effective,
                "device_wall_us": device_wall,
            }
            medians[arm.name] = {
                "effective_us": statistics.median(effective),
                "device_wall_us": statistics.median(device_wall),
            }
        replicate_rows.append(
            {
                "replicate": replicate,
                "order": [arm.name for arm in order],
                "medians": medians,
                "raw": raw,
            }
        )

    return {
        "correctness": correctness,
        "replicates": replicate_rows,
        "summary": _summarize_replicates(replicate_rows),
    }


def _selected_shapes(names: list[str]) -> list[Shape]:
    if not names:
        return list(SHAPES)
    by_name = {shape.name: shape for shape in SHAPES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown shapes {unknown}; choose from {sorted(by_name)}")
    return [by_name[name] for name in names]


def _print_summary(shape: Shape, result: dict[str, Any]) -> None:
    print(f"\n{shape.name}: M={shape.m} K={shape.k} N={shape.n}")
    print(f"{'arm':16} {'effective_us':>13} {'penalty':>11} {'95% CI':>24}")
    for arm in ARMS:
        row = result["summary"]["effective_us"][arm.name]
        lo, hi = row["penalty_95ci_percent"]
        print(
            f"{arm.name:16} {row['median_us']:13.3f} "
            f"{row['penalty_vs_role_aligned_percent']:+10.2f}% "
            f"[{lo:+8.2f}%, {hi:+8.2f}%]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list shapes and arms")
    parser.add_argument("--shape", action="append", default=[], help="shape name; repeat to select several")
    parser.add_argument("--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--planner", choices=tuple(_PLANNERS), default="dsa_rp")
    parser.add_argument("--run-device", action="store_true", help="run correctness and timing after compile")
    parser.add_argument("--skip-ptoas", action="store_true", help="host-only raw PTO generation")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run root (default: build_output/autotile_l1_layout_<timestamp>)",
    )
    args = parser.parse_args()

    if args.list:
        print("Shapes:")
        for shape in SHAPES:
            print(f"  {shape.name:10} M={shape.m:4} K={shape.k:4} N={shape.n:4}")
        print("Arms:")
        for arm in ARMS:
            print(f"  {arm.name:16} A={arm.a_layout} B={arm.b_layout}")
        return
    if args.run_device and args.skip_ptoas:
        parser.error("--run-device cannot be combined with --skip-ptoas")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    try:
        _balanced_orders(args.replicates)
    except ValueError as exc:
        parser.error(str(exc))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or Path("build_output") / f"autotile_l1_layout_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "config": {
            "platform": args.platform,
            "device": args.device,
            "planner": args.planner,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "replicates": args.replicates,
            "seed": args.seed,
            "skip_ptoas": args.skip_ptoas,
        },
        "arms": [asdict(arm) for arm in ARMS],
        "shapes": {},
    }

    for shape in _selected_shapes(args.shape):
        shape_dir = output_root / shape.name
        compiled = ir.compile(
            _build_program(shape),
            output_dir=str(shape_dir),
            dump_passes=PassDumpLevel.EXPLICIT,
            skip_ptoas=args.skip_ptoas,
            memory_planner=_PLANNERS[args.planner],
            platform=args.platform,
        )
        shape_result: dict[str, Any] = {
            "shape": asdict(shape),
            "build_dir": str(shape_dir),
            "structure": _verify_structures(compiled),
        }
        print(f"{shape.name}: structural gate PASS")
        if args.run_device:
            shape_result.update(
                _run_shape(
                    compiled,
                    shape,
                    platform=args.platform,
                    device=args.device,
                    rounds=args.rounds,
                    warmup=args.warmup,
                    replicates=args.replicates,
                    seed=args.seed,
                )
            )
            _print_summary(shape, shape_result)
        results["shapes"][shape.name] = shape_result
        (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nResults: {output_root / 'results.json'}")


if __name__ == "__main__":
    main()
