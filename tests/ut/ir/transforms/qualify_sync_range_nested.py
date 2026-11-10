# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Record host checks and scaling for explicit ranges and invariant perfect nests.

Run as python -m tests.ut.ir.transforms.qualify_sync_range_nested --output DIR.
The synthetic fixtures do not establish general Qwen nested composition.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .qualify_parameterized_prefill import digest
from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .test_nested_sync_repetition import BOUNDS, nested_fixture
from .test_sync_range_proofs import range_fixture


def measure(function, bounds, parameters):
    start = time.perf_counter()
    analysis = testing.analyze_placed_sync(function, True, parameter_bounds=bounds)
    analyzed = time.perf_counter()
    physical = analysis.realize_physical("a2a3", list(range(6)))
    emitted = time.perf_counter()
    d = analysis.describe()
    row: dict[str, Any] = {
        "case": function.name,
        "bounds": bounds,
        "seconds": {
            "analysis_inclusive": analyzed - start,
            "physical_inclusive": emitted - analyzed,
            **{
                k: d[k]
                for k in ["import_seconds", "partition_seconds", "reduction_seconds", "composition_seconds"]
            },
        },
        "retained": {
            k: d[k]
            for k in [
                "analyzed_occurrences",
                "cells",
                "boundary_graph_edges",
                "coordinate_nodes",
                "coordinate_flatten_visits",
                "class_tests",
                "port_pair_tests",
            ]
        },
        "regions": [
            {
                "count": r["count"],
                "frames": len(r["nested_frames"]),
                "original_frames": [
                    {"count": count, "start": start, "step": step}
                    for _, count, start, step in r["nested_frames"]
                ],
                "sites": len(r["sites"]),
                "generators": len(r["generators"]),
                "demands": len(r["demands"]),
                "distance_entries": r["distance_entries"],
            }
            for r in d["regions"]
        ],
        "physical": {
            k: physical[k]
            for k in [
                "set_sites",
                "wait_sites",
                "barrier_sites",
                "guard_sites",
                "reuse_queries",
                "reserved_lanes",
            ]
        },
        "checks": [],
    }
    for p in parameters:
        check = validate(function, analysis, parameters=p)
        execution, _, _ = run_physical(function, physical["function"], range(6), parameters=p)
        row["checks"].append(
            {
                "parameters": p,
                **check,
                "executed_sets": len(execution.records),
                "executed_waits": len(execution.waits),
                "causal_reuse_checks": len(execution.reuses),
                "directional_ids": {
                    f"{int(a)}->{int(b)}": sorted(ids) for (a, b), ids in execution.ids.items()
                },
                "modeled_order_equality": True,
            }
        )
    return row, physical["function"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qwen-capture", type=Path, help="Reanalyze an existing Buffer capture, without recapture"
    )
    parser.add_argument("--draft-paper", type=Path)
    parser.add_argument("--native-source", action="store_true", help="PTOAS C++ generation only; no device")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    root = Path(__file__).resolve().parents[4]
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=root, text=True).splitlines()
    additions = [
        Path(__file__),
        Path(__file__).with_name("test_nested_sync_repetition.py"),
        Path(__file__).with_name("test_sync_range_proofs.py"),
    ]
    sources = {*changed, *(str(p.relative_to(root)) for p in additions)}
    receipt: dict[str, Any] = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_hashes": {p: digest(root / p) for p in sorted(sources) if not p.endswith(".json")},
        "extension_hashes": {
            str(p.relative_to(root)): digest(p) for p in (root / "python/pypto").glob("pypto_core*.so")
        },
        "python": platform.python_version(),
        "contract": {
            "target": "a2a3",
            "disjoint_gm": True,
            "directional_pool": list(range(6)),
            "scope": "explicit ranges; invariant perfect nests; no general nested child composition",
            "device": "not run; native entry/return remains a supplied premise",
        },
        "timings": "single descriptive runs; detail timers are included in analysis, not additive",
        "cases": [],
        "native_source": [],
    }
    cases = [
        (range_fixture(), {1: (0, 16), 3: (32, 128)}, [{1: x, 3: r} for x in [0, 16] for r in [32, 64, 128]])
    ]
    for counts in [
        [0, 3],
        [2, 0],
        [1, 1],
        [2, 3],
        [2, 2, 2],
        [2, 0, 3],
        [2, 1000],
        [1000, 3],
        [1000000, 1000000],
    ]:
        small = max(counts) < 10
        cases.append(
            (
                nested_fixture(counts),
                BOUNDS,
                [{1: 0, 3: 64}, {1: 16, 3: 128}] if small else [],
            )
        )
    cases.append((nested_fixture([2, 3], repeats=2), BOUNDS, [{1: 0, 3: 64}, {1: 16, 3: 128}]))
    for i, (function, bounds, parameters) in enumerate(cases):
        row, output = measure(function, bounds, parameters)
        receipt["cases"].append(row)
        if args.native_source and i in [0, 4]:
            source, cpp = args.output / f"case-{i}.pto", args.output / f"case-{i}.cpp"
            source.write_text(
                codegen.PTOCodegen().generate(
                    ir.Program([output], "Qualified", function.span),
                    emit_tile_addr=True,
                    emit_source_loc=False,
                )
            )
            _run_ptoas(str(source), str(cpp), ["--pto-level=level3", "--pto-arch=a2"])
            receipt["native_source"].append(
                {"case": i, "pto_sha256": digest(source), "cpp_sha256": digest(cpp)}
            )
    if args.native_source:
        tool = find_ptoas_binary()
        assert tool is not None
        path = Path(tool).resolve()
        receipt["ptoas"] = {
            "launcher_sha256": digest(path),
            "version": subprocess.check_output([tool, "--version"], text=True).strip(),
            "binaries": {
                str(p.relative_to(path.parent.parent)): digest(p)
                for p in (path.parent.parent / "ptoas").rglob("*.so")
            },
        }
    if args.qwen_capture:
        program = ir.deserialize(args.qwen_capture.read_bytes())
        assert isinstance(program, ir.Program)
        facts = testing.sync_parameter_bounds(program)
        rows = []
        for function in program.functions.values():
            if function.func_type != ir.FunctionType.AIC:
                continue
            try:
                result = testing.analyze_placed_sync(
                    function, True, parameter_bounds=facts.get(function.name, {})
                )
                rows.append(
                    {
                        "function": function.name,
                        "analysis": "accepted",
                        "regions": len(result.describe()["regions"]),
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        "function": function.name,
                        "analysis": "unavailable",
                        "reason": str(error).splitlines()[0],
                    }
                )
        receipt["qwen_frozen_capture"] = {
            "path": str(args.qwen_capture),
            "sha256": digest(args.qwen_capture),
            "caller_bounds": facts,
            "outcomes": rows,
        }
    if args.draft_paper:
        receipt["draft"] = {
            name: digest(args.draft_paper / "sections" / name)
            for name in ["minimum_demands.tex", "direct_realization.tex", "event_id_reuse.tex"]
        }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
