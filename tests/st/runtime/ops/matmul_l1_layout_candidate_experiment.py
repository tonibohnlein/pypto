# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test whether the L1 Right layout changes the ranking of legal L0 tiles.

This is a research driver, not a system test. It fixes the Left source to NZ,
varies the Right source between ZN and NZ, and forces two legal L0 tile
candidates for each problem. Every cell is compiled independently, checked
structurally, checked against three seeded goldens, and benchmarked in
position-balanced blocks. Candidate contrasts are paired within each block;
the script never subtracts aggregate medians.

Example::

    python tests/st/runtime/ops/matmul_l1_layout_candidate_experiment.py --list
    python tests/st/runtime/ops/matmul_l1_layout_candidate_experiment.py \
        --platform a2a3 --planner dsa_rp --run-device --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from matmul_l1_layout_experiment import (  # noqa: PLC2701
    _LAYOUT_FIELDS,
    ARMS,
    Shape,
    _arm_inputs,
    _build_program,
    _pto_structure,
)
from pypto import ir
from pypto.ir.pass_manager import PassDumpLevel
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import ChipWorker, RunConfig, benchmark


@dataclass(frozen=True)
class Candidate:
    name: str
    m: int
    n: int
    k: int
    stationarity: str = "OS"
    double_buffer_c: bool = False

    @property
    def force(self) -> str:
        return f"{self.m},{self.n},{self.k},{self.stationarity},{int(self.double_buffer_c)}"


@dataclass(frozen=True)
class StudyCase:
    shape: Shape
    current: Candidate
    alternate: Candidate


# k_heavy compares two model-tied K granularities. The older suggestion
# (64,128,256) is intentionally absent: output-stationary operand double
# buffering makes its 256x128 BF16 Right panel exceed L0B.
CASES = (
    StudyCase(
        Shape("k_heavy", 128, 512, 128),
        Candidate("current", 128, 128, 128),
        Candidate("alternate", 128, 128, 64),
    ),
    StudyCase(
        Shape("mn_tiled", 256, 128, 256),
        Candidate("current", 64, 256, 128, stationarity="B", double_buffer_c=True),
        Candidate("alternate", 128, 128, 128),
    ),
)

LAYOUT_ARMS = tuple(arm for arm in ARMS if arm.name in {"a_nz_b_zn", "a_nz_b_nz"})
_PLANNERS = {
    "pypto": MemoryPlanner.PYPTO,
    "dsa_rp": MemoryPlanner.DSA_RP,
    "ptoas": MemoryPlanner.PTOAS,
}
_SSA_RE = re.compile(r"%[A-Za-z0-9_.$]+")
_ADDR_RE = re.compile(r"addr=\d+")
_LOC_RE = re.compile(r'loc\("[^"]*"[^)]*\)')


@contextmanager
def _forced(candidate: Candidate):
    previous = os.environ.get("PYPTO_FORCE_L0_TILE")
    os.environ["PYPTO_FORCE_L0_TILE"] = candidate.force
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PYPTO_FORCE_L0_TILE", None)
        else:
            os.environ["PYPTO_FORCE_L0_TILE"] = previous


def _candidates(case: StudyCase) -> tuple[Candidate, Candidate]:
    return case.current, case.alternate


def _cell_name(candidate: Candidate, arm: Any) -> str:
    return f"{candidate.name}__{arm.b_layout.lower()}"


def _normalised_tloads(build_dir: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(build_dir.rglob("*.pto")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "pto.tload " not in line:
                continue
            normal = _SSA_RE.sub("%v", line.strip())
            normal = _ADDR_RE.sub("addr=<address>", normal)
            normal = _LOC_RE.sub("loc(<source>)", normal)
            lines.append(normal)
    return sorted(lines)


def _normalised_core_structure(build_dir: Path) -> list[str]:
    """Return layout-independent scheduling lines for cross-arm comparison."""
    lines: list[str] = []
    markers = ("scf.for ", "pto.tmatmul ", "pto.tmatmul.acc ", "pto.tstore ")
    for path in sorted(build_dir.rglob("*.pto")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(marker in line for marker in markers) and "pipeline_double_buffer_c" not in line:
                continue
            normal = _SSA_RE.sub("%v", line.strip())
            normal = _ADDR_RE.sub("addr=<address>", normal)
            normal = _LOC_RE.sub("loc(<source>)", normal)
            lines.append(normal)
    return lines


def _verify_cell(candidate: Candidate, arm: Any, compiled: Any) -> dict[str, Any]:
    build_dir = Path(compiled[arm.orchestration].output_dir)
    structure = _pto_structure(build_dir)
    observed_left = {tuple(value) for value in structure["source_layouts"]["left"]}
    observed_right = {tuple(value) for value in structure["source_layouts"]["right"]}
    if observed_left != {_LAYOUT_FIELDS["NZ"]}:
        raise AssertionError(f"{_cell_name(candidate, arm)}: Left layouts {observed_left}")
    if observed_right != {_LAYOUT_FIELDS[arm.b_layout]}:
        raise AssertionError(f"{_cell_name(candidate, arm)}: Right layouts {observed_right}")

    signatures = {tuple(value) for value in structure["tile_signatures"]}
    expected = {
        ("left", candidate.m, candidate.k),
        ("right", candidate.k, candidate.n),
        ("acc", candidate.m, candidate.n),
    }
    if signatures != expected:
        raise AssertionError(
            f"{_cell_name(candidate, arm)}: forced tile signatures {sorted(signatures)}, "
            f"expected exactly {sorted(expected)}"
        )
    if len(structure["pto_files"]) != 1:
        raise AssertionError(
            f"{_cell_name(candidate, arm)}: expected one AIC codegen unit, got {structure['pto_files']}"
        )
    structure["normalised_tloads"] = _normalised_tloads(build_dir)
    structure["normalised_core_structure"] = _normalised_core_structure(build_dir)
    return structure


def _compile_case(
    case: StudyCase,
    *,
    output_root: Path,
    platform: str,
    planner: MemoryPlanner,
    skip_ptoas: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compiled: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    baseline_dir = output_root / case.shape.name / "unforced_baseline"
    baseline = ir.compile(
        _build_program(case.shape),
        output_dir=str(baseline_dir),
        dump_passes=PassDumpLevel.EXPLICIT,
        skip_ptoas=skip_ptoas,
        memory_planner=planner,
        platform=platform,
    )
    baseline_structures = {arm.name: _verify_cell(case.current, arm, baseline) for arm in LAYOUT_ARMS}
    for candidate in _candidates(case):
        candidate_dir = output_root / case.shape.name / candidate.name
        with _forced(candidate):
            program = ir.compile(
                _build_program(case.shape),
                output_dir=str(candidate_dir),
                dump_passes=PassDumpLevel.EXPLICIT,
                skip_ptoas=skip_ptoas,
                memory_planner=planner,
                platform=platform,
            )
        compiled[candidate.name] = program
        for arm in LAYOUT_ARMS:
            cell = _cell_name(candidate, arm)
            structures[cell] = _verify_cell(candidate, arm, program)

    # Never silently compare two arbitrary forced points. The arm labelled
    # current must reproduce the unforced production chooser structurally.
    for arm in LAYOUT_ARMS:
        forced = structures[_cell_name(case.current, arm)]
        unforced = baseline_structures[arm.name]
        for field in ("tile_signatures", "counts", "normalised_tloads", "normalised_core_structure"):
            if forced[field] != unforced[field]:
                raise AssertionError(
                    f"{case.shape.name}/{arm.b_layout}: forced current differs from "
                    f"unforced baseline in {field}"
                )
    structures["unforced_baseline"] = baseline_structures

    # Within one layout, changing the forced L0 tile must not change the
    # full-tensor GM->L1 load geometry. This catches accidental differences in
    # the experiment before they can be misattributed to TEXTRACT or matmul.
    for arm in LAYOUT_ARMS:
        current = structures[_cell_name(case.current, arm)]["normalised_tloads"]
        alternate = structures[_cell_name(case.alternate, arm)]["normalised_tloads"]
        if current != alternate:
            raise AssertionError(
                f"{case.shape.name}/{arm.b_layout}: GM->L1 TLOAD geometry differs between candidates"
            )
    # For a fixed candidate, layout is the sole intended variable. Require the
    # same exact tile set, op counts, and layout-independent loop/matmul/store
    # schedule; TEXTRACT source layout is checked separately by _verify_cell.
    for candidate in _candidates(case):
        left = structures[_cell_name(candidate, LAYOUT_ARMS[0])]
        right = structures[_cell_name(candidate, LAYOUT_ARMS[1])]
        for field in ("tile_signatures", "counts", "normalised_core_structure"):
            if left[field] != right[field]:
                raise AssertionError(f"{case.shape.name}/{candidate.name}: B layout changes {field}")
    return compiled, structures


def _balanced_orders(case: StudyCase, replicates: int) -> list[list[tuple[Candidate, Any]]]:
    cells = [(candidate, arm) for candidate in _candidates(case) for arm in LAYOUT_ARMS]
    if replicates <= 0 or replicates % (2 * len(cells)) != 0:
        raise ValueError(f"--replicates must be a positive multiple of {2 * len(cells)}")
    orders: list[list[tuple[Candidate, Any]]] = []
    for replicate in range(replicates):
        rotation = (replicate // 2) % len(cells)
        order = cells[rotation:] + cells[:rotation]
        if replicate % 2:
            order = list(reversed(order))
        orders.append(order)
    return orders


def _bootstrap_median_ci(values: list[float], *, samples: int = 50_000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(samples))
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def _classify(median: float, ci: list[float]) -> str:
    lo, hi = ci
    if lo > 2.0:
        return "CURRENT_WINS_GT_2_PERCENT"
    if hi < -2.0:
        return "ALTERNATE_WINS_GT_2_PERCENT"
    if lo >= -2.0 and hi <= 2.0:
        return "EQUIVALENT_WITHIN_2_PERCENT"
    return "UNRESOLVED"


def _summarize_metrics(
    replicates: list[dict[str, Any]], case: StudyCase, metrics: tuple[str, ...]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in metrics:
        by_layout: dict[str, Any] = {}
        for arm in LAYOUT_ARMS:
            current_cell = _cell_name(case.current, arm)
            alternate_cell = _cell_name(case.alternate, arm)
            contrasts = [
                100.0 * (row["medians"][alternate_cell][metric] / row["medians"][current_cell][metric] - 1.0)
                for row in replicates
            ]
            median = statistics.median(contrasts)
            ci = _bootstrap_median_ci(contrasts)
            by_layout[arm.b_layout] = {
                "alternate_penalty_vs_current_percent": median,
                "paired_95ci_percent": ci,
                "classification": _classify(median, ci),
                "replicate_contrasts_percent": contrasts,
            }
        summary[metric] = by_layout
    return summary


def _summarize(replicates: list[dict[str, Any]], case: StudyCase) -> dict[str, Any]:
    return _summarize_metrics(replicates, case, ("effective_us", "device_wall_us"))


def _run_case(
    case: StudyCase,
    compiled: dict[str, Any],
    *,
    platform: str,
    device: int,
    rounds: int,
    warmup: int,
    replicates: int,
) -> dict[str, Any]:
    config = RunConfig(platform=platform, device_id=device)
    correctness: dict[str, Any] = {}
    benchmark_args: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for seed in range(3):
        torch.manual_seed(seed)
        canonical_a = torch.randn((case.shape.m, case.shape.k), dtype=torch.bfloat16)
        canonical_b = torch.randn((case.shape.k, case.shape.n), dtype=torch.bfloat16)
        expected = canonical_a.float() @ canonical_b.float()
        for candidate in _candidates(case):
            for arm in LAYOUT_ARMS:
                cell = _cell_name(candidate, arm)
                a, b = _arm_inputs(arm, canonical_a, canonical_b)
                out = torch.zeros((case.shape.m, case.shape.n), dtype=torch.float32)
                compiled[candidate.name][arm.orchestration](a, b, out, config=config)
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                correctness[f"seed_{seed}/{cell}"] = {
                    "max_abs": (out - expected).abs().max().item(),
                    "max_rel": ((out - expected).abs() / expected.abs().clamp_min(1e-6)).max().item(),
                }
                if seed == 0:
                    benchmark_args[cell] = (a, b, out)

    replicate_rows: list[dict[str, Any]] = []
    for replicate, order in enumerate(_balanced_orders(case, replicates)):
        raw: dict[str, Any] = {}
        medians: dict[str, Any] = {}
        for candidate, arm in order:
            cell = _cell_name(candidate, arm)
            stats = benchmark(
                compiled[candidate.name][arm.orchestration],
                benchmark_args[cell],
                rounds=rounds,
                warmup=warmup,
                config=config,
            )
            effective = stats.per_round("effective")
            device_wall = stats.per_round("device")
            if len(effective) != rounds or len(device_wall) != rounds or min(effective + device_wall) <= 0:
                raise RuntimeError(f"{case.shape.name}/{cell}: incomplete or zero timing samples")
            raw[cell] = {"effective_us": effective, "device_wall_us": device_wall}
            medians[cell] = {
                "effective_us": statistics.median(effective),
                "device_wall_us": statistics.median(device_wall),
            }
        replicate_rows.append(
            {
                "replicate": replicate,
                "order": [_cell_name(candidate, arm) for candidate, arm in order],
                "medians": medians,
                "raw": raw,
            }
        )
    return {
        "correctness": correctness,
        "replicates": replicate_rows,
        "summary": _summarize(replicate_rows, case),
    }


def _find_case(name: str) -> StudyCase:
    case = next((item for item in CASES if item.shape.name == name), None)
    if case is None:
        raise ValueError(f"unknown case {name!r}")
    return case


def _find_candidate(case: StudyCase, name: str) -> Candidate:
    candidate = next((item for item in _candidates(case) if item.name == name), None)
    if candidate is None:
        raise ValueError(f"unknown candidate {name!r} for {case.shape.name}")
    return candidate


def _find_layout_arm(layout: str) -> Any:
    arm = next((item for item in LAYOUT_ARMS if item.b_layout == layout), None)
    if arm is None:
        raise ValueError(f"unknown B layout {layout!r}")
    return arm


def _read_sole_task_duration(dfx_dir: Path) -> tuple[float, int]:
    from simpler_setup.tools.swimlane_converter import read_perf_data  # noqa: PLC0415

    records = dfx_dir / "chip_swimlane_records.json"
    if not records.is_file():
        raise RuntimeError(f"chip swimlane records not found: {records}")
    tasks = read_perf_data(str(records)).get("tasks", [])
    if len(tasks) != 1:
        raise RuntimeError(f"expected exactly one AIC task in {records}, got {len(tasks)}")
    duration = float(tasks[0].get("duration_us", 0.0))
    if duration <= 0:
        raise RuntimeError(f"invalid task duration {duration} in {records}")
    return duration, int(tasks[0]["func_id"])


def _run_task_sample(
    *,
    case_name: str,
    candidate_name: str,
    layout: str,
    build_dir: Path,
    output: Path,
    platform: str,
    device: int,
    warmup: int,
) -> None:
    case = _find_case(case_name)
    candidate = _find_candidate(case, candidate_name)
    arm = _find_layout_arm(layout)
    compiled = ir.CompiledProgram.from_dir(build_dir, platform=platform)
    torch.manual_seed(0)
    canonical_a = torch.randn((case.shape.m, case.shape.k), dtype=torch.bfloat16)
    canonical_b = torch.randn((case.shape.k, case.shape.n), dtype=torch.bfloat16)
    expected = canonical_a.float() @ canonical_b.float()
    a, b = _arm_inputs(arm, canonical_a, canonical_b)
    out = torch.zeros((case.shape.m, case.shape.n), dtype=torch.float32)
    dfx_dir = build_dir / "dfx_outputs"
    config = RunConfig(platform=platform, device_id=device)
    with ChipWorker(
        config,
        runtime=compiled.runtime_name,
        enable_sdma=bool(compiled.runtime_config.get("enable_sdma", False)),
    ) as worker:
        handle = worker.register(compiled)
        for _ in range(warmup):
            handle(a, b, out, config=config)
        shutil.rmtree(dfx_dir, ignore_errors=True)
        out.zero_()
        handle(
            a,
            b,
            out,
            config=RunConfig(platform=platform, device_id=device, enable_chip_swimlane=1),
        )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    duration, func_id = _read_sole_task_duration(dfx_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    archived_dfx = output.parent / "dfx_outputs"
    shutil.rmtree(archived_dfx, ignore_errors=True)
    shutil.copytree(dfx_dir, archived_dfx)
    output.write_text(
        json.dumps(
            {
                "shape": case_name,
                "candidate": candidate.name,
                "layout": layout,
                "force": candidate.force,
                "task_duration_us": duration,
                "func_id": func_id,
                "max_abs": (out - expected).abs().max().item(),
                "max_rel": ((out - expected).abs() / expected.abs().clamp_min(1e-6)).max().item(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_task_timing(
    case: StudyCase,
    compiled: dict[str, Any],
    *,
    output_root: Path,
    platform: str,
    device: int,
    warmup: int,
    replicates: int,
    task_rounds: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for replicate, order in enumerate(_balanced_orders(case, replicates)):
        medians: dict[str, Any] = {}
        samples: dict[str, Any] = {}
        for candidate, arm in order:
            cell = _cell_name(candidate, arm)
            build_dir = Path(compiled[candidate.name][arm.orchestration].output_dir)
            cell_samples: list[dict[str, Any]] = []
            for launch in range(task_rounds):
                sample_dir = (
                    output_root
                    / "task_timing"
                    / case.shape.name
                    / f"replicate_{replicate:02d}"
                    / cell
                    / f"launch_{launch:02d}"
                )
                sample_json = sample_dir / "task_sample.json"
                command = [
                    sys.executable,
                    str(script),
                    "--task-sample",
                    "--task-shape-name",
                    case.shape.name,
                    "--task-candidate",
                    candidate.name,
                    "--task-layout",
                    arm.b_layout,
                    "--task-build-dir",
                    str(build_dir),
                    "--task-output",
                    str(sample_json),
                    "--platform",
                    platform,
                    "--device",
                    str(device),
                    "--warmup",
                    str(warmup),
                ]
                try:
                    result = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        f"task timing child timed out for {case.shape.name}/{cell}/"
                        f"replicate {replicate}/launch {launch}"
                    ) from error
                (sample_dir / "child.stdout.log").write_text(result.stdout, encoding="utf-8")
                (sample_dir / "child.stderr.log").write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(
                        f"task timing child failed for {case.shape.name}/{cell}/replicate {replicate}/"
                        f"launch {launch}: {result.stderr[-2000:]}"
                    )
                cell_samples.append(json.loads(sample_json.read_text(encoding="utf-8")))
            durations = [float(sample["task_duration_us"]) for sample in cell_samples]
            samples[cell] = cell_samples
            medians[cell] = {"task_duration_us": statistics.median(durations)}
        rows.append(
            {
                "replicate": replicate,
                "order": [_cell_name(candidate, arm) for candidate, arm in order],
                "medians": medians,
                "samples": samples,
            }
        )
    return {
        "replicates": rows,
        "summary": _summarize_metrics(rows, case, ("task_duration_us",)),
    }


def _print_summary(case: StudyCase, result: dict[str, Any]) -> None:
    print(f"\n{case.shape.name}: M={case.shape.m} K={case.shape.k} N={case.shape.n}")
    summaries = dict(result["summary"])
    if "task_timing" in result:
        summaries.update(result["task_timing"]["summary"])
    for metric, layouts in summaries.items():
        print(metric)
        for layout, row in layouts.items():
            lo, hi = row["paired_95ci_percent"]
            print(
                f"  B={layout}: alternate-current "
                f"{row['alternate_penalty_vs_current_percent']:+.2f}% "
                f"[{lo:+.2f}%, {hi:+.2f}%] {row['classification']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--planner", choices=tuple(_PLANNERS), default="dsa_rp")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument(
        "--task-timing",
        action="store_true",
        help="also collect 24 fresh-process chip-swimlane task-duration blocks",
    )
    parser.add_argument("--skip-ptoas", action="store_true")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--task-rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task-sample", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-shape-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--task-candidate", default="", help=argparse.SUPPRESS)
    parser.add_argument("--task-layout", default="", help=argparse.SUPPRESS)
    parser.add_argument("--task-build-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.task_sample:
        if args.task_build_dir is None or args.task_output is None:
            parser.error("--task-sample requires --task-build-dir and --task-output")
        _run_task_sample(
            case_name=args.task_shape_name,
            candidate_name=args.task_candidate,
            layout=args.task_layout,
            build_dir=args.task_build_dir,
            output=args.task_output,
            platform=args.platform,
            device=args.device,
            warmup=args.warmup,
        )
        return

    if args.list:
        for case in CASES:
            print(
                f"{case.shape.name}: M={case.shape.m} K={case.shape.k} N={case.shape.n}; "
                f"current={case.current.force}; alternate={case.alternate.force}"
            )
        return
    if args.run_device and args.skip_ptoas:
        parser.error("--run-device cannot be combined with --skip-ptoas")
    if args.task_timing and not args.run_device:
        parser.error("--task-timing requires --run-device")
    if args.rounds <= 0 or args.task_rounds <= 0 or args.warmup < 0:
        parser.error("--rounds/--task-rounds must be positive and --warmup non-negative")
    try:
        _balanced_orders(CASES[0], args.replicates)
    except ValueError as error:
        parser.error(str(error))

    selected = [case for case in CASES if not args.shape or case.shape.name in args.shape]
    unknown = sorted(set(args.shape) - {case.shape.name for case in CASES})
    if unknown:
        parser.error(f"unknown shapes {unknown}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or Path("build_output") / f"autotile_l1_layout_candidates_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "config": {
            "platform": args.platform,
            "planner": args.planner,
            "device": args.device,
            "rounds": args.rounds,
            "task_rounds": args.task_rounds,
            "warmup": args.warmup,
            "replicates": args.replicates,
        },
        "cases": {},
    }
    for case in selected:
        compiled, structures = _compile_case(
            case,
            output_root=output_root,
            platform=args.platform,
            planner=_PLANNERS[args.planner],
            skip_ptoas=args.skip_ptoas,
        )
        case_result: dict[str, Any] = {
            "shape": asdict(case.shape),
            "candidates": [asdict(candidate) | {"force": candidate.force} for candidate in _candidates(case)],
            "structure": structures,
        }
        print(f"{case.shape.name}: structural gate PASS")
        if args.run_device:
            case_result.update(
                _run_case(
                    case,
                    compiled,
                    platform=args.platform,
                    device=args.device,
                    rounds=args.rounds,
                    warmup=args.warmup,
                    replicates=args.replicates,
                )
            )
            if args.task_timing:
                case_result["task_timing"] = _run_task_timing(
                    case,
                    compiled,
                    output_root=output_root,
                    platform=args.platform,
                    device=args.device,
                    warmup=args.warmup,
                    replicates=args.replicates,
                    task_rounds=args.task_rounds,
                )
            _print_summary(case, case_result)
        results["cases"][case.shape.name] = case_result
        (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults: {output_root / 'results.json'}")


if __name__ == "__main__":
    main()
