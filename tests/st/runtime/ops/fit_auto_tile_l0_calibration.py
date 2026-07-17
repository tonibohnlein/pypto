# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fit #2079 L0 issue/dependency terms from the device calibration CSV.

The fit uses within-shape timing deltas, so fixed runtime and GM/L1 costs cancel.
It compares the baseline, extract-only, accumulator-only, and combined models.
By default the exact #2079 shape is held out and reported as a generalization
check rather than being used to choose the coefficients.
"""

import argparse
import csv
import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

_FREQ_MHZ = 1850.0


@dataclass(frozen=True)
class Point:
    case_id: str
    category: str
    label: str
    shape: tuple[int, int, int, str, str]
    force: str
    exec_us: float
    load: float
    mad: float
    drain: float
    a_extracts: int
    b_extracts: int
    acc_continuations: int


@dataclass(frozen=True)
class Coefficients:
    load_a_issue_cycles: float = 0.0
    load_b_issue_cycles: float = 0.0
    mad_acc_dependency_cycles: float = 0.0


def _read_points(path: Path) -> list[Point]:
    samples: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row.get("count", "0")) != 1:
                raise ValueError(
                    f"{row.get('case_id', '<unknown>')}: expected one L2 task, got count={row.get('count')}"
                )
            if row.get("timing_source") != "l2_task_duration_us":
                raise ValueError(
                    f"{row.get('case_id', '<unknown>')}: unsupported timing source "
                    f"{row.get('timing_source')!r}; rerun with structured L2 timing"
                )
            samples.setdefault(row["case_id"], []).append(row)
    points = []
    for case_id, rows in samples.items():
        first = rows[0]
        points.append(
            Point(
                case_id=case_id,
                category=first["category"],
                label=first["label"],
                shape=(int(first["M"]), int(first["N"]), int(first["K"]), first["dtype"], first["layout"]),
                force=first["force"],
                exec_us=statistics.median(float(row["exec_us"]) for row in rows),
                load=float(first["model_load"]),
                mad=float(first["model_mad"]),
                drain=float(first["model_drain"]),
                a_extracts=int(first["a_extracts"]),
                b_extracts=int(first["b_extracts"]),
                acc_continuations=int(first["acc_continuations"]),
            )
        )
    return points


def _groups(points: list[Point]) -> list[list[Point]]:
    grouped: dict[tuple[int, int, int, str, str], list[Point]] = {}
    for point in points:
        grouped.setdefault(point.shape, []).append(point)
    return [group for group in grouped.values() if len(group) >= 2]


def _predict(point: Point, coefficients: Coefficients) -> float:
    load = (
        point.load
        + point.a_extracts * coefficients.load_a_issue_cycles
        + point.b_extracts * coefficients.load_b_issue_cycles
    )
    mad = point.mad + point.acc_continuations * coefficients.mad_acc_dependency_cycles
    compute = max(load, mad)
    m, n, _k, _stat, dbc = point.force.split(",")
    if dbc == "1":
        M, N, *_ = point.shape
        output_tiles = ((M + int(m) - 1) // int(m)) * ((N + int(n) - 1) // int(n))
        return max(compute, point.drain) + min(compute, point.drain) / output_tiles
    return compute + point.drain


def _metrics(groups: list[list[Point]], coefficients: Coefficients) -> tuple[float, float, int]:
    errors: list[float] = []
    winner_hits = 0
    for group in groups:
        measured = [point.exec_us * _FREQ_MHZ for point in group]
        predicted = [_predict(point, coefficients) for point in group]
        measured_min = min(measured)
        predicted_min = min(predicted)
        scale = max(100.0, *(value - measured_min for value in measured))
        errors.extend(
            ((pred - predicted_min) - (meas - measured_min)) / scale
            for pred, meas in zip(predicted, measured)
        )
        predicted_winner = min(range(len(group)), key=predicted.__getitem__)
        # Device winners within 2% are treated as a noise-equivalent set.
        winner_hits += measured[predicted_winner] <= measured_min * 1.02
    rms = (sum(error * error for error in errors) / len(errors)) ** 0.5 if errors else float("inf")
    accuracy = winner_hits / len(groups) if groups else 0.0
    return rms, accuracy, len(groups)


def _grid(text: str) -> list[float]:
    try:
        start, stop, step = (float(value) for value in text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid must be START:STOP:STEP") from error
    if step <= 0 or stop < start:
        raise argparse.ArgumentTypeError("grid requires STEP>0 and STOP>=START")
    values = []
    current = start
    while current <= stop + step * 1e-9:
        values.append(current)
        current += step
    return values


def _fit(
    groups: list[list[Point]], a_grid: list[float], b_grid: list[float], acc_grid: list[float], variant: str
) -> tuple[Coefficients, tuple[float, float, int]]:
    a_values = a_grid if variant in {"extract", "combined"} else [0.0]
    b_values = b_grid if variant in {"extract", "combined"} else [0.0]
    acc_values = acc_grid if variant in {"acc", "combined"} else [0.0]
    best: tuple[tuple[float, float, float], Coefficients, tuple[float, float, int]] | None = None
    for a_issue, b_issue, acc_dependency in itertools.product(a_values, b_values, acc_values):
        coefficients = Coefficients(a_issue, b_issue, acc_dependency)
        metrics = _metrics(groups, coefficients)
        # First minimize normalized delta error, then maximize top-1 accuracy,
        # then prefer the smallest total correction within an exact tie.
        key = (metrics[0], -metrics[1], a_issue + b_issue + acc_dependency)
        if best is None or key < best[0]:
            best = (key, coefficients, metrics)
    assert best is not None
    return best[1], best[2]


def _describe(name: str, coefficients: Coefficients, train_groups, holdout_groups, all_groups) -> None:
    print(
        f"{name:10} A_issue={coefficients.load_a_issue_cycles:6.1f} "
        f"B_issue={coefficients.load_b_issue_cycles:6.1f} "
        f"ACC_dep={coefficients.mad_acc_dependency_cycles:6.1f}"
    )
    for label, groups in [("train", train_groups), ("holdout", holdout_groups), ("all", all_groups)]:
        rms, accuracy, count = _metrics(groups, coefficients)
        print(f"  {label:7} groups={count:2d} normalized_RMS={rms:7.4f} top1@2%={accuracy:6.1%}")


def _report_groups(groups: list[list[Point]], coefficients: Coefficients) -> None:
    print("\nPer-shape measured and predicted ranking:")
    for group in sorted(groups, key=lambda value: value[0].shape):
        measured_best = min(point.exec_us for point in group)
        predicted_values = [_predict(point, coefficients) for point in group]
        predicted_best = min(predicted_values)
        M, N, K, dtype, layout = group[0].shape
        print(f"  {M}x{N}x{K} {dtype}/{layout}")
        for point, prediction in sorted(zip(group, predicted_values), key=lambda item: item[0].exec_us):
            print(
                f"    {point.label:14} measured={point.exec_us:8.4f}us "
                f"delta={(point.exec_us / measured_best - 1) * 100:7.2f}% "
                f"pred_delta={(prediction / predicted_best - 1) * 100:7.2f}% {point.force}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--a-grid", type=_grid, default=_grid("0:400:25"))
    parser.add_argument("--b-grid", type=_grid, default=_grid("0:400:25"))
    parser.add_argument("--acc-grid", type=_grid, default=_grid("0:800:50"))
    parser.add_argument("--include-issue2079-in-fit", action="store_true")
    parser.add_argument("--json", type=Path, help="write the selected combined coefficients")
    args = parser.parse_args()

    all_groups = _groups(_read_points(args.csv))
    holdout_groups = [group for group in all_groups if any(point.category == "issue2079" for point in group)]
    train_groups = (
        all_groups
        if args.include_issue2079_in_fit
        else [group for group in all_groups if group not in holdout_groups]
    )
    if not train_groups:
        parser.error("no training groups; run the broad suite or pass --include-issue2079-in-fit")

    fits: dict[str, Coefficients] = {"baseline": Coefficients()}
    for variant in ("extract", "acc", "combined"):
        fits[variant], _ = _fit(train_groups, args.a_grid, args.b_grid, args.acc_grid, variant)
    for name, coefficients in fits.items():
        _describe(name, coefficients, train_groups, holdout_groups, all_groups)

    selected = fits["combined"]
    _report_groups(all_groups, selected)
    if args.json:
        args.json.write_text(json.dumps(selected.__dict__, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
