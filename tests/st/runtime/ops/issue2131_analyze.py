# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Summarize and bootstrap issue #2131 device benchmark results."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

EXPECTED_REPLICATES = {1, 2, 3}
PLANNERS = ("pypto", "ptoas")
VARIANTS = ("baseline", "dbc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2131)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("issue2131_result.json"))
    if not paths:
        raise RuntimeError(f"No issue2131_result.json found under {root}")
    return [json.loads(path.read_text()) | {"_path": str(path)} for path in paths]


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _speedup(baseline: float, dbc: float) -> float:
    return 100.0 * (baseline - dbc) / baseline


def _run_bootstrap_ci(
    deltas: list[float],
    iterations: int,
    rng: random.Random,
) -> tuple[float, float]:
    draws: list[float] = []
    for _ in range(iterations):
        draws.append(_median(rng.choices(deltas, k=len(deltas))))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return lo, hi


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        path = row["_path"]
        assert row.get("passed"), f"Failed run in timing root: {path}"
        assert not row.get("compile_only"), f"Compile-only run in timing root: {path}"
        assert row.get("bench_enabled") == "1", f"PYPTO_BENCH=1 missing: {path}"
        assert row.get("benchmark"), f"Benchmark data missing: {path}"
        assert row.get("planner") in PLANNERS, f"Unexpected planner: {path}"
        assert row.get("variant") in VARIANTS, f"Unexpected variant: {path}"
        assert row.get("replicate") in EXPECTED_REPLICATES, f"Unexpected replicate: {path}"
        assert row.get("enable_l2_swimlane") is False, f"L2 swimlane must be disabled: {path}"
        assert row.get("golden_data"), f"Frozen golden-data provenance missing: {path}"
        key = (row["planner"], row["replicate"], row["variant"])
        assert key not in indexed, f"Duplicate timing row {key}: {path}"
        indexed[key] = row

    expected = {
        (planner, replicate, variant)
        for planner in PLANNERS
        for replicate in EXPECTED_REPLICATES
        for variant in VARIANTS
    }
    assert set(indexed) == expected, f"Incomplete timing matrix: missing={expected - set(indexed)}"

    metadata = {(row["device"], row["golden_data"]) for row in indexed.values()}
    assert len(metadata) == 1, f"Timing metadata mismatch: {metadata}"
    for row in indexed.values():
        assert row["seed"] == 2040, f"Timing seed must be 2040: {row['_path']}"
        assert row["platform"] == "a2a3", f"Timing platform must be a2a3: {row['_path']}"
        benchmark = row["benchmark"]
        assert benchmark["rounds"] == 100, f"Timing requires 100 rounds: {row['_path']}"
        assert benchmark["warmup"] == 5, f"Timing requires 5 warmups: {row['_path']}"
        lengths = {
            len(benchmark["effective_us"]),
            len(benchmark["device_wall_us"]),
            len(benchmark["host_wall_us"]),
            benchmark["rounds"],
        }
        assert len(lengths) == 1, f"Benchmark sample-count mismatch: {row['_path']}"
        samples = benchmark["effective_us"] + benchmark["device_wall_us"] + benchmark["host_wall_us"]
        assert all(math.isfinite(value) and value > 0.0 for value in samples), (
            f"Timing samples must be finite and positive: {row['_path']}"
        )
    return indexed


def _direction(deltas: list[float]) -> str:
    median = _median(deltas)
    if median > 2.0 and all(delta > 0.0 for delta in deltas):
        return "improves"
    if median < -2.0 and all(delta < 0.0 for delta in deltas):
        return "regresses"
    return "tied_or_mixed"


def _summarize(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    assert bootstrap > 0, "bootstrap iterations must be positive"
    indexed = _index_rows(rows)
    summary: dict[str, Any] = {"planners": {}}
    for planner in PLANNERS:
        planner_summary: dict[str, Any] = {"replicates": []}
        effective_deltas: list[float] = []
        device_deltas: list[float] = []
        baseline_effective_medians: list[float] = []
        dbc_effective_medians: list[float] = []
        baseline_device_medians: list[float] = []
        dbc_device_medians: list[float] = []
        for replicate in sorted(EXPECTED_REPLICATES):
            baseline = indexed[(planner, replicate, "baseline")]["benchmark"]
            dbc = indexed[(planner, replicate, "dbc")]["benchmark"]
            baseline_eff = _median(baseline["effective_us"])
            dbc_eff = _median(dbc["effective_us"])
            baseline_dev = _median(baseline["device_wall_us"])
            dbc_dev = _median(dbc["device_wall_us"])
            effective_delta = _speedup(baseline_eff, dbc_eff)
            device_delta = _speedup(baseline_dev, dbc_dev)
            baseline_effective_medians.append(baseline_eff)
            dbc_effective_medians.append(dbc_eff)
            baseline_device_medians.append(baseline_dev)
            dbc_device_medians.append(dbc_dev)
            effective_deltas.append(effective_delta)
            device_deltas.append(device_delta)
            planner_summary["replicates"].append(
                {
                    "replicate": replicate,
                    "effective_baseline_median_us": baseline_eff,
                    "effective_dbc_median_us": dbc_eff,
                    "effective_speedup_pct": effective_delta,
                    "device_baseline_median_us": baseline_dev,
                    "device_dbc_median_us": dbc_dev,
                    "device_speedup_pct": device_delta,
                }
            )
        rng = random.Random(seed + (0 if planner == "pypto" else 1))
        eff_ci = _run_bootstrap_ci(effective_deltas, bootstrap, rng)
        dev_ci = _run_bootstrap_ci(device_deltas, bootstrap, rng)
        eff_speedup = _median(effective_deltas)
        direction = _direction(effective_deltas)
        planner_summary["independent_runs"] = {
            "effective_baseline_median_of_run_medians_us": _median(baseline_effective_medians),
            "effective_dbc_median_of_run_medians_us": _median(dbc_effective_medians),
            "effective_speedup_median_pct": eff_speedup,
            "effective_speedup_run_bootstrap_95ci_pct": list(eff_ci),
            "device_baseline_median_of_run_medians_us": _median(baseline_device_medians),
            "device_dbc_median_of_run_medians_us": _median(dbc_device_medians),
            "device_speedup_median_pct": _median(device_deltas),
            "device_speedup_run_bootstrap_95ci_pct": list(dev_ci),
            "direction": direction,
            "clears_2pct_gate": direction == "improves" and eff_ci[0] > 0.0,
        }
        summary["planners"][planner] = planner_summary
    runs = [value["independent_runs"] for value in summary["planners"].values()]
    if all(run["clears_2pct_gate"] for run in runs):
        decision = "PROCEED_WITH_AUTOTILE_MODEL"
    elif (
        runs[0]["effective_speedup_median_pct"] * runs[1]["effective_speedup_median_pct"] < 0.0
        and max(abs(run["effective_speedup_median_pct"]) for run in runs) > 2.0
    ):
        decision = "INVESTIGATE_PLANNER_DIVERGENCE"
    else:
        decision = "DO_NOT_AUTO_ENABLE"
    summary["decision"] = decision
    return summary


def main() -> None:
    args = _parse_args()
    rows = _load(args.results_root)
    summary = _summarize(rows, args.bootstrap, args.seed)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
