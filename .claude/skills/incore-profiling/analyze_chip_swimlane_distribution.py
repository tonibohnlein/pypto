# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
"""Summarize per-slice and per-core chip-swimlane duration distributions."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"--trace expects NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"--trace repeats name {name!r}")
        result[name] = Path(raw_path)
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    mean = statistics.fmean(values) if values else None
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "median": statistics.median(values) if values else None,
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": mean,
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0 if values else None
        ),
    }


def analyze_trace(records: dict[str, Any], task_id: int | None = None) -> dict[str, Any]:
    metadata = records.get("metadata") or {}
    frequency = int(metadata.get("clock_freq_hz") or 0)
    if frequency <= 0:
        raise ValueError("chip-swimlane metadata has no positive clock_freq_hz")
    core_types = list(metadata.get("core_types") or [])
    slices = []
    for raw in records.get("aicore_tasks") or []:
        if len(raw) < 5:
            raise ValueError(f"malformed aicore_tasks row: {raw!r}")
        core_id, token = int(raw[0]), int(raw[1])
        entry_task = token & 0xFFFFFFFF
        if task_id is not None and entry_task != task_id:
            continue
        start, end = int(raw[3]), int(raw[4])
        if end < start:
            raise ValueError(f"negative slice duration on core {core_id}: {start}..{end}")
        slices.append(
            {
                "core_id": core_id,
                "core_type": core_types[core_id] if core_id < len(core_types) else "",
                "round": token >> 32,
                "task_id": entry_task,
                "reg_task_id": int(raw[2]),
                "start_ticks": start,
                "end_ticks": end,
                "duration_ticks": end - start,
                "duration_us": (end - start) / frequency * 1e6,
            }
        )
    if not slices:
        raise ValueError("no matching AICore slices")
    task_tokens = {(entry["round"], entry["task_id"]) for entry in slices}
    if task_id is None and len(task_tokens) != 1:
        raise ValueError(f"trace contains {len(task_tokens)} task tokens; pass --task-id to select one")

    per_core: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in slices:
        per_core[entry["core_id"]].append(entry)
    core_rows = []
    wave_durations: dict[int, list[float]] = defaultdict(list)
    for core_id, entries in sorted(per_core.items()):
        entries.sort(key=lambda entry: (entry["start_ticks"], entry["end_ticks"]))
        busy_ticks = sum(entry["duration_ticks"] for entry in entries)
        span_ticks = entries[-1]["end_ticks"] - entries[0]["start_ticks"]
        gaps = [
            max(0, current["start_ticks"] - prior["end_ticks"])
            for prior, current in zip(entries, entries[1:])
        ]
        for wave, entry in enumerate(entries):
            entry["core_local_wave"] = wave
            wave_durations[wave].append(entry["duration_us"])
        core_rows.append(
            {
                "core_id": core_id,
                "core_type": entries[0]["core_type"],
                "slice_count": len(entries),
                "busy_us": busy_ticks / frequency * 1e6,
                "span_us": span_ticks / frequency * 1e6,
                "idle_gap_us": sum(gaps) / frequency * 1e6,
                "max_gap_us": max(gaps, default=0) / frequency * 1e6,
                "first_start_ticks": entries[0]["start_ticks"],
                "last_end_ticks": entries[-1]["end_ticks"],
            }
        )

    task_start = min(entry["start_ticks"] for entry in slices)
    task_end = max(entry["end_ticks"] for entry in slices)
    return {
        "clock_freq_hz": frequency,
        "task_tokens": [list(token) for token in sorted(task_tokens)],
        "task_window_us": (task_end - task_start) / frequency * 1e6,
        "slice_count": len(slices),
        "core_count": len(per_core),
        "core_type_counts": dict(sorted(Counter(entry["core_type"] for entry in slices).items())),
        "slice_duration_us": _distribution([entry["duration_us"] for entry in slices]),
        "core_busy_us": _distribution([row["busy_us"] for row in core_rows]),
        "core_span_us": _distribution([row["span_us"] for row in core_rows]),
        "core_idle_gap_us": _distribution([row["idle_gap_us"] for row in core_rows]),
        "wave_duration_us": {
            str(wave): _distribution(values) for wave, values in sorted(wave_durations.items())
        },
        "slices": sorted(slices, key=lambda entry: (entry["start_ticks"], entry["core_id"])),
        "cores": core_rows,
    }


def analyze_traces(traces: dict[str, dict[str, Any]], task_id: int | None = None) -> dict[str, Any]:
    captures = {name: analyze_trace(records, task_id) for name, records in traces.items()}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, capture in captures.items():
        group, separator, _capture_name = name.rpartition("/")
        grouped[group if separator else name].append(capture)
    groups = {}
    for name, members in sorted(grouped.items()):
        slices = [entry for member in members for entry in member["slices"]]
        cores = [entry for member in members for entry in member["cores"]]
        waves: dict[int, list[float]] = defaultdict(list)
        registration_waves: dict[int, list[float]] = defaultdict(list)
        for entry in slices:
            waves[int(entry["core_local_wave"])].append(float(entry["duration_us"]))
            registration_waves[int(entry["reg_task_id"])].append(float(entry["duration_us"]))
        groups[name] = {
            "capture_count": len(members),
            "task_window_us": _distribution([float(member["task_window_us"]) for member in members]),
            "slice_duration_us": _distribution([float(entry["duration_us"]) for entry in slices]),
            "core_busy_us": _distribution([float(entry["busy_us"]) for entry in cores]),
            "core_span_us": _distribution([float(entry["span_us"]) for entry in cores]),
            "core_idle_gap_us": _distribution([float(entry["idle_gap_us"]) for entry in cores]),
            "wave_duration_us": {str(wave): _distribution(values) for wave, values in sorted(waves.items())},
            "registration_wave_duration_us": {
                str(wave): _distribution(values) for wave, values in sorted(registration_waves.items())
            },
        }
    return {"schema_version": 1, "captures": captures, "groups": groups}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = _named_paths(args.trace)
        if not paths:
            raise ValueError("at least one --trace is required")
        report = analyze_traces({name: _read_object(path) for name, path in paths.items()}, args.task_id)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
