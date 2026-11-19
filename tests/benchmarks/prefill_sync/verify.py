# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay two hand-edited Qwen3-14B prefill synchronization plans from PyPTO 441b5c417.

Use Python 3.10+, g++, and PTOAS 0.74. From the repository root run
    python tests/benchmarks/prefill_sync/verify.py --ptoas /path/to/ptoas

The M2 controls use conservative_prefix and disjoint_gm with the PyPTO memory planner
and PyPTO-lib b4f395ec. Both variants retain the original payloads and return drain.
SiLU replaces trip-count cases with previous/next tile predicates. Gate/up extends
four MAT slot returns across the projection boundary and signals accumulator reuse
from the gate store to the first up multiply. The reference covers layer 0 on A2/A3
with disjoint GM arguments, across all 24 workers and four MLP bands.

results.json records generated-control counts and modeled completion-before-issue
order. Hardware timing, native instruction counts and numerical outputs are separate
measurements.
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from control_trace import PIPE, profile, trace
from storage_reference import reference, source_trace

HERE = Path(__file__).resolve().parent


def check_flag_lifetimes(events, cube, *, event_slots=6):
    """Check causal consumption before an event slot is republished, including across loop iterations."""
    latest_start, latest_completion, gate, commands = {}, {}, {}, {}
    published, consumed = {}, {}
    next_node = 0
    reuse_count = 0

    def node(ancestors):
        nonlocal next_node
        bits = ancestors | (1 << next_node)
        next_node += 1
        return bits

    for line in events:
        kind, *args = line.split()
        if kind == "if":
            continue
        if kind == "op":
            (op,) = args
            p = {
                "TLOAD": "MTE2",
                "TSTORE": "FIX" if cube else "MTE3",
                "TEXTRACT": "MTE1",
                "TMATMUL": "M",
                "TMATMUL_ACC": "M",
            }.get(op, "V")
            latest_start[p] = node(latest_start.get(p, 0) | gate.get(p, 0))
            latest_completion[p] = node(latest_start[p] | latest_completion.get(p, 0))
        elif kind == "bar":
            affected = list(PIPE) if args == ["ALL"] else args
            parents = 0
            for p in affected:
                parents |= latest_completion.get(p, 0) | commands.get(p, 0)
            barrier = node(parents)
            for p in affected:
                gate[p] = commands[p] = barrier
        else:
            src, dst, slot = args
            assert src != dst and 0 <= int(slot) < event_slots
            key = tuple(args)
            if kind == "set":
                assert key not in published
                parents = latest_completion.get(src, 0) | commands.get(src, 0) | gate.get(src, 0)
                if key in consumed:
                    assert parents & consumed[key], ("reuse before consumption", key)
                    reuse_count += 1
                published[key] = node(parents)
                commands[src] = published[key]
            else:
                assert kind == "wait" and key in published
                parents = latest_start.get(dst, 0) | gate.get(dst, 0) | published.pop(key)
                consumed[key] = 1 << next_node
                gate[dst] = node(parents)
                commands[dst] = commands.get(dst, 0) | gate[dst]
    assert not published and events[-1] == "bar ALL"
    return reuse_count


def order_delta(rows, reference_rows):
    diff = [x - y for a, b in zip(rows, reference_rows, strict=True) for x, y in zip(a, b, strict=True)]
    return {"extra": sum(max(n, 0) for n in diff), "missing": sum(max(-n, 0) for n in diff)}


def check_silu(temp):
    inputs = [temp / "silu-baseline.cpp", temp / "silu-guards.cpp"]
    (base_sites, base), (new_sites, new) = [trace(p, [], 24, temp) for p in inputs]
    trips = {}
    for worker, (a, b) in enumerate(zip(base, new, strict=True)):
        assert [x for x in a if x != "if"] == [x for x in b if x != "if"]
        ops, rows, counts = profile(a, False)
        new_ops, new_rows, new_counts = profile(b, False)
        assert ops == new_ops and rows == new_rows
        assert check_flag_lifetimes(a, False) == check_flag_lifetimes(b, False)
        old_guards, new_guards = counts["if"], new_counts["if"]
        assert old_guards == 2 * new_guards
        key = str(ops.count("TSTORE"))
        sample = {
            "baseline": old_guards,
            "candidate": new_guards,
            "sets": counts["set"],
            "waits": counts["wait"],
            "local_barriers": counts["bar V"],
        }
        group = trips.setdefault(key, {**sample, "workers": []})
        assert all(group[k] == v for k, v in sample.items())
        group["workers"].append(worker)
    return {"static_guard_sites": [base_sites["if"], new_sites["if"]], "trips": trips}


def check_gate(temp):
    inputs = [temp / "gate-up-baseline.cpp", temp / "gate-up-handoff.cpp"]
    summaries = {}
    for band in range(4):
        args, width = [6 * band, 17 * band, 0, 5120], 24
        (_, base), (_, new) = [trace(p, args, width, temp) for p in inputs]
        for worker, (a, b) in enumerate(zip(base, new, strict=True)):
            ops = source_trace(HERE / "gate-up-original.pto", args, width, worker)
            required = reference(ops)
            old_ops, old_rows, old_counts = profile(a, True)
            new_ops, new_rows, new_counts = profile(b, True)
            assert old_ops == new_ops == [op[0] for op in ops]
            old_delta, new_delta = order_delta(old_rows, required), order_delta(new_rows, required)
            role = (
                "both"
                if old_ops.count("TSTORE") == 2
                else ("gate" if (worker + 6 * band) % 24 < 17 else "up")
            )
            assert old_delta == {"extra": 78 if role == "both" else 0, "missing": 0}
            assert new_delta == {"extra": 0, "missing": 0}, (band, worker, new_delta)
            reuses = [check_flag_lifetimes(events, True) for events in (a, b)]
            sample = {
                "payloads": len(ops),
                "required_pairs": sum(map(sum, required)),
                "baseline": old_delta,
                "candidate": new_delta,
                "commands": [old_counts, new_counts],
                "event_reuses_checked": reuses,
            }
            group = summaries.setdefault(role, {**sample, "workers_by_band": {}})
            assert all(group[k] == v for k, v in sample.items()), (role, worker, band)
            group["workers_by_band"].setdefault(band, []).append(worker)
    return summaries


def payload_ir(stem: str) -> list[str]:
    """Compare all PTO payload, placement and view operations, normalizing temporary SSA names."""
    return [
        re.sub(r"%value_(\d+)\b", r"%\1", line)
        for line in (HERE / f"{stem}.pto").read_text().splitlines()
        if "pto." in line and not re.search(r"pto\.(?:set_flag|wait_flag|barrier)\b", line)
    ]


def main():
    if not __debug__:
        raise RuntimeError("Run the replay with Python assertions enabled")
    assert payload_ir("silu-baseline") == payload_ir("silu-guards")
    assert payload_ir("gate-up-baseline") == payload_ir("gate-up-handoff")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptoas", default="ptoas", help="PTOAS 0.74 executable")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="prefill-sync-") as tmp:
        temp = Path(tmp)
        for stem in ("silu-baseline", "silu-guards", "gate-up-baseline", "gate-up-handoff"):
            subprocess.run(
                [
                    options.ptoas,
                    "--pto-arch=a2",
                    "--pto-level=level3",
                    "--enable-insert-sync=false",
                    str(HERE / f"{stem}.pto"),
                    "-o",
                    str(temp / f"{stem}.cpp"),
                ],
                check=True,
            )
        results = {"silu": check_silu(temp), "gate_up_proj": check_gate(temp)}
    expected = json.loads((HERE / "results.json").read_text())
    # JSON object keys are strings, including the integer band indices.
    assert json.loads(json.dumps(results)) == expected
    print("Passed 24 SiLU workers and 96 gate/up worker-band combinations.")


if __name__ == "__main__":
    main()
