# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproducible interval/atom baseline; compilation times are descriptive."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, testing
from pypto.backend import BackendType

from .sync_logical_oracle import CommandExecution
from .sync_matmul_oracle import interpret, validate
from .sync_physical_oracle import directional_widths, run_physical
from .test_collective_sync_interface import collective_fixture
from .test_collective_sync_realization import repeat_collective
from .test_interval_sync_analysis import interval_pipeline
from .test_slot_sync_import import slot_fixture


def qualify_collective(receipt, realization=False):
    """Measure certified witness frames; large rows are host-only derivatives."""
    cases = (
        [(5, 0), (5, 1), (5, 3), (5, 1000), (5, 1000000), (1000, 3), (1000000, 3)]
        if realization
        else [(8, 1), (1000, 1), (1000000, 1), (8, 2), (8, 8), (8, 32)]
    )
    for rows, phases in cases:
        function = (
            repeat_collective(rows, phases)
            if realization
            else collective_fixture(rows, [[(0, rows)]] * phases)
        )
        start = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - start
        description = result.describe()
        row: dict[str, Any] = {
            "rows": rows,
            "visits" if realization else "static_phases": phases,
            "analysis_seconds_inclusive": elapsed,
        }
        for key in [
            "collective_domains",
            "collective_witnesses",
            "collective_access_visits",
            "interface_cells",
            "interface_queries",
            "interface_ports",
            "merge_nodes",
            "source_coordinate_entries",
            "interface_entries",
            "analyzed_occurrences",
        ]:
            if key in description:
                row[key] = description[key]
        row["semantic"] = validate(function, result) if rows <= 8 and phases <= 8 else "scaling only"
        row["physical"] = "not requested in M3"
        if realization:
            start = time.perf_counter()
            physical = result.realize_physical("a2a3", list(range(6)))
            row["physical_seconds"] = time.perf_counter() - start
            row["physical"] = {k: v for k, v in physical.items() if k not in ["function", "families"]}
            row["physical_ir_sha256"] = hashlib.sha256(
                ir.python_print(physical["function"], format=False).encode()
            ).hexdigest()
            start = time.perf_counter()
            texts = [
                codegen.PTOCodegen().generate(
                    ir.Program([f], "collective", f.span), emit_tile_addr=True, emit_source_loc=False
                )
                for f in [function, physical["function"]]
            ]
            row["source_ir_seconds_original_and_physical"] = time.perf_counter() - start
            row["source_ir"] = {
                "added_arithmetic_operations": texts[1].count("arith.") - texts[0].count("arith."),
                "physical_bytes": len(texts[1].encode()),
                "physical_sha256": hashlib.sha256(texts[1].encode()).hexdigest(),
                "scope": "PTO IR before downstream folding, not device instructions",
            }
            if rows <= 8 and phases <= 8:
                state, order, projected = run_physical(function, physical["function"], range(6))
                _, calls = interpret(function)
                original = CommandExecution(calls, [])
                original.execute(function.body, {})
                row["emitted_checks"] = {
                    "executed_handoffs": len(state.records),
                    "executed_waits": len(state.waits),
                    "executed_barriers": len(state.barriers),
                    "executed_added_guard_tests": state.branches - original.branches,
                    "causal_reuses": len(state.reuses),
                    "payload_order_pairs": len(projected),
                    "directional_widths": {str(k): v for k, v in directional_widths(state, order).items()},
                    "ids": {str(k): sorted(v) for k, v in state.ids.items()},
                    "excess_pairs": 0,
                }
        receipt["cases"].append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="Measure exact nested interval views")
    parser.add_argument("--collective", action="store_true", help="Measure collective crossing witnesses")
    parser.add_argument(
        "--realization", action="store_true", help="Include repeated collective physical checks"
    )
    args = parser.parse_args()
    backend.set_backend_type(BackendType.Ascend910)
    root = Path.cwd()
    files = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    files += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    files = sorted(set(files))
    receipt: dict[str, Any] = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in files
            if p.startswith(("src/", "include/", "python/", "tests/"))
        },
        "extension_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "python/pypto").glob("pypto_core*.so")
        },
        "device": "not run",
        "timing": "single descriptive samples; no speedup claim",
        "cases": [],
    }
    if args.collective:
        qualify_collective(receipt, args.realization)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        return
    cases = (
        [(7, 3, n) for n in [4, 16, 64, 128]]
        if args.partial
        else [(count, slots, 0) for count, slots in [(7, 3), (1000, 3), (1000000, 3), (7, 101), (7, 1000003)]]
    )
    for count, slots, readers in cases:
        function = (
            interval_pipeline(count, readers, slots)
            if args.partial
            else slot_fixture(count, slots, alias=True)
        )
        start = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - start
        description = result.describe()
        region = description["regions"][0]
        row = {
            "count": count,
            "readers": readers,
            "slots": slots,
            "analysis_seconds_inclusive": elapsed,
            "interval_access_visits": description["interval_access_visits"],
            "qualified_intervals": description["qualified_intervals"],
        }
        for key in [
            "atom_tests",
            "interval_domains",
            "interval_fragments",
            "interval_endpoints",
            "interval_queries",
            "interval_updates",
            "interval_node_visits",
            "interval_tree_entries",
            "rotating_fragments",
            "sort_comparisons",
            "sweep_visits",
            "shortest_path_edge_visits",
            "retained_frontier_entries",
            "generators",
            "demands",
        ]:
            row[key] = region[key]
        start = time.perf_counter()
        try:
            physical = result.realize_physical("a2a3", list(range(6)))
            row["physical_ir_sha256"] = hashlib.sha256(
                ir.python_print(physical["function"], format=False).encode()
            ).hexdigest()
            row["physical"] = "accepted"
        except ValueError as error:
            row["physical"] = str(error).splitlines()[0]
        row["physical_seconds"] = time.perf_counter() - start
        receipt["cases"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
