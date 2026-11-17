# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bound nested synchronization construction without executing an unfolded oracle.

Run with python/ and tests/ut/ir on PYTHONPATH. Each configuration runs in a
fresh, serial child process. The watchdog applies to each named stage, including
bootstrap; a timed-out native call is terminated by its parent, independently
of Python signal delivery or the extension's GIL behavior. Production and
additional diagnostic requests use separate workers and receipts.
"""

import argparse
import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Only native workers load compiler and fixture modules. The parent remains
# lightweight and enforces the bootstrap watchdog as well as native stages.
if "--worker" in sys.argv:
    from pypto import backend, ir, testing
    from pypto.backend import BackendType

    from transforms.test_nested_sync_repetition import BOUNDS, nested_fixture

WATCHDOG_SECONDS = 12


def _event(stage: str, status: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, "status": status, **details}), flush=True)


def _worker(depth: int, trips: int, diagnostic: bool) -> None:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    _event("analysis", "start")
    started = time.perf_counter()
    function = nested_fixture([trips] * depth)
    retained = testing.analyze_placed_sync(function, True, parameter_bounds=BOUNDS)
    description = retained.describe()
    _event(
        "analysis",
        "done",
        seconds=time.perf_counter() - started,
        source_coordinate_entries=description["source_coordinate_entries"],
        source_region_nodes=description["source_region_nodes"],
    )
    _event("logical", "start")
    started = time.perf_counter()
    logical = retained.realize()
    _event(
        "logical",
        "done",
        seconds=time.perf_counter() - started,
        record_sites=logical["record_sites"],
        wait_sites=logical["wait_sites"],
    )
    if not diagnostic:
        _event("physical", "start")
        started = time.perf_counter()
        physical = testing.lower_sync_schedule(
            logical["function"], parameter_bounds=BOUNDS, eligible_ids=list(range(6))
        )
        cases = []
        for case in physical["physical_cases"]:
            # Exclude IR objects and axes from JSON. Scalar counters describe
            # production work; palette counters retain their existing names.
            summary: dict[str, Any] = {
                key: value for key, value in case.items() if isinstance(value, (int, float, str, bool))
            }
            summary["palette_attempts"] = [
                {key: value for key, value in attempt.items() if key != "pieces"}
                | {"rank_pieces": len(attempt["pieces"])}
                for attempt in case["palette_attempts"]
            ]
            cases.append(summary)
        _event("physical", "done", seconds=time.perf_counter() - started, cases=cases)
        return

    tokens = logical["function"]
    # These explicit requests are audit work, not additive components of the
    # production LowerSync measurement above. Every request consumes the same
    # original logical function; no easier specialized function is generated.
    for prepare in (False, True):
        stage = "prepared_families" if prepare else "raw_families"
        _event(stage, "start")
        started = time.perf_counter()
        inventory = testing.inspect_scoped_token_families(tokens, parameter_bounds=BOUNDS, prepare=prepare)
        _event(stage, "done", seconds=time.perf_counter() - started, work=inventory["work"])
    _event("actual_token_order", "start")
    started = time.perf_counter()
    order = testing.analyze_scoped_token_order(tokens, parameter_bounds=BOUNDS)
    _event(
        "actual_token_order",
        "done",
        seconds=time.perf_counter() - started,
        summary_nodes=order.summary_nodes,
        interface_entries=order.interface_entries,
    )
    for source, target in (
        (ir.PipeType.MTE2, ir.PipeType.MTE3),
        (ir.PipeType.MTE3, ir.PipeType.MTE2),
        (ir.PipeType.MTE2, ir.PipeType.V),
        (ir.PipeType.V, ir.PipeType.MTE2),
    ):
        stage = f"palette_{source.name}_{target.name}"
        _event(stage, "start")
        started = time.perf_counter()
        result = order.palette(source, target, 6)
        counters = {key: value for key, value in result.items() if key != "pieces"}
        counters["rank_pieces"] = len(result["pieces"])
        _event(stage, "done", seconds=time.perf_counter() - started, work=counters)


def _sample(output: Path, depth: int, trips: int, diagnostic: bool) -> dict[str, Any]:
    kind = "diagnostic" if diagnostic else "production"
    stem = output.parent / f"{output.stem}-{kind}-d{depth}-k{trips}"
    row: dict[str, Any] = {"depth": depth, "trips_per_frame": trips, "kind": kind, "events": []}
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(depth), str(trips)]
    if diagnostic:
        command.append("--diagnostic")
    row["command"] = command
    environment = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (
        stem.with_suffix(".stderr.log").open("w") as errors,
        stem.with_suffix(".stdout.jsonl").open("w") as raw,
    ):
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, env=environment) as child:
            assert child.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + WATCHDOG_SECONDS
            current = "bootstrap"
            pending = b""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(max(0, remaining)):
                    child.kill()
                    child.wait()
                    row.update(status="timeout", timed_out_stage=current)
                    break
                chunk = os.read(child.stdout.fileno(), 65536)
                if not chunk:
                    row.update(status="passed" if child.wait() == 0 else "failed", exit_code=child.returncode)
                    break
                raw.write(chunk.decode("utf-8", errors="replace"))
                raw.flush()
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Native diagnostics remain available in the raw log.
                    row["events"].append(event)
                    if event["status"] == "start":
                        current = event["stage"]
                        deadline = time.monotonic() + WATCHDOG_SECONDS
            selector.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--trips", type=int, nargs="+", default=[2])
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--worker", type=int, nargs=2, metavar=("DEPTH", "TRIPS"))
    args = parser.parse_args()
    if args.worker:
        _worker(*args.worker, args.diagnostic)
        return
    if args.output is None:
        parser.error("output is required for a parent sweep")
    if any(value < 1 for value in [*args.depths, *args.trips]):
        parser.error("depths and trip counts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "watchdog_seconds_per_stage": WATCHDOG_SECONDS,
        "workers": 1,
        "device_validation": False,
        "execution_oracle": False,
        "timing_scope": "single descriptive runs; diagnostic requests are separate, non-additive audit work",
        "python": sys.version,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    for trips in args.trips:
        for depth in args.depths:
            receipt["rows"].append(_sample(args.output, depth, trips, args.diagnostic))
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
