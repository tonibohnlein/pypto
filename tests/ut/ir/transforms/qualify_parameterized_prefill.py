# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproducible host-only receipt for the parameterized finite/periodic checkpoint.

Run as ``python -m tests.ut.ir.transforms.qualify_parameterized_prefill --output DIR``.
No device is used. K derivatives change source configuration, not retained IR.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from pypto import backend, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig

from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .sync_prefill_cases import REVISION, SOURCES, capture_prefill, load_prefill, prefill_arguments


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(name, count, check):
    started = time.perf_counter()
    program = capture_prefill(name, count)
    lowered = time.perf_counter()
    function = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    bounds = testing.sync_parameter_bounds(program)[function.name]
    ranged = time.perf_counter()
    result = testing.analyze_placed_sync(function, True, parameter_bounds=bounds)
    analyzed = time.perf_counter()
    description = result.describe()
    physical = result.realize_physical("a2a3", list(range(6)))
    emitted = time.perf_counter()
    row: dict[str, Any] = {
        "source": name,
        "k_tiles_derivative": count,
        "bounds": bounds,
        "timing_scope": "single descriptive measurement; analysis detail nested within analysis wall time",
        "seconds": {
            "frontend_placement": lowered - started,
            "caller_evidence": ranged - lowered,
            "analysis_inclusive": analyzed - ranged,
            "physical_realization_inclusive": emitted - analyzed,
            **{
                key: description[key]
                for key in ["import_seconds", "partition_seconds", "reduction_seconds", "composition_seconds"]
            },
        },
        "retained": {
            key: description[key]
            for key in [
                "analyzed_occurrences",
                "cells",
                "boundary_graph_edges",
                "class_tests",
                "class_lookup_tests",
                "port_pair_tests",
                "crossing_path_searches",
                "coordinate_nodes",
                "coordinate_flatten_visits",
                "coordinate_coefficient_entries",
            ]
        },
        "regions": [
            {
                "count": r["count"],
                "sites": len(r["sites"]),
                "generators": len(r["generators"]),
                "demands": len(r["demands"]),
                "distance_entries": r["distance_entries"],
            }
            for r in description["regions"]
        ],
        "crossings": len(description["crossing_demands"]),
        "physical": {
            key: physical[key]
            for key in [
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
    if check:
        for params in [{1: 0, 3: 0}, {1: 64, 3: 128}, {1: 192, 3: 192}]:
            semantic = validate(function, result, parameters=params)
            execution, _, _ = run_physical(function, physical["function"], range(6), parameters=params)
            row["checks"].append(
                {
                    "parameters": params,
                    **semantic,
                    "executed_sets": len(execution.records),
                    "executed_waits": len(execution.waits),
                    "executed_barriers": len(execution.barriers),
                    "causal_reuse_checks": len(execution.reuses),
                    "directional_ids": {
                        f"{int(a)}->{int(b)}": sorted(ids) for (a, b), ids in execution.ids.items()
                    },
                    "modeled_order_equality": True,
                }
            )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true", help="Also invoke pinned PTOAS for C++ generation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    root = Path(__file__).resolve().parents[4]
    tracked = subprocess.check_output(["git", "diff", "--name-only"], cwd=root, text=True).splitlines()
    additions = [
        Path(__file__).relative_to(root),
        Path(__file__).with_name("sync_prefill_cases.py").relative_to(root),
        Path(__file__).with_name("test_parameterized_sync.py").relative_to(root),
    ]
    binaries = list((root / "python/pypto").glob("pypto_core*.so"))
    receipt: dict[str, Any] = {
        "scope": "stage 1; nested Qwen output and gate/up composition remain unimplemented",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_source_hashes": {str(p): digest(root / p) for p in [*map(Path, tracked), *additions]},
        "binary_hashes": {str(p.relative_to(root)): digest(p) for p in binaries},
        "python": platform.python_version(),
        "lib_revision": REVISION,
        "lib_sources": SOURCES,
        "contract": {
            "target": "a2a3",
            "disjoint_gm": True,
            "ids": list(range(6)),
            "native_entry_return": "unchanged prerequisite, no new device qualification",
        },
        "rows": [],
        "native": [],
    }
    for name, count, check in [
        ("matmul", None, True),
        ("gemm", None, True),
        *[("gemm", k, k <= 4) for k in [1, 2, 3, 4, 8, 64, 1000000]],
    ]:
        receipt["rows"].append(measure(name, count, check))
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.native:
        tool = find_ptoas_binary()
        assert tool is not None
        tool_path = Path(tool).resolve()
        receipt["ptoas"] = {
            "path": str(tool_path),
            "launcher_sha256": digest(tool_path),
            "version": subprocess.check_output([tool, "--version"], text=True).strip(),
            "package_binary_hashes": {
                str(p.relative_to(tool_path.parent.parent)): digest(p)
                for p in (tool_path.parent.parent / "ptoas").rglob("*.so")
            },
        }
        for name in ["matmul", "gemm"]:
            module = load_prefill(name)
            output = args.output / name
            start = time.perf_counter()
            with passes.PassContext(
                [],
                enable_buffer_ir=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
                enable_pypto_autosync=True,
                autosync_disjoint_gm=True,
            ):
                getattr(module, name).compile(
                    *prefill_arguments(module),
                    config=RunConfig(
                        arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(output)
                    ),
                )
            report = json.loads((output / "report/autosync.json").read_text())
            assert report["kernels"] and all(k["selected_route"] == "pypto" for k in report["kernels"])
            cpp = list(output.rglob("*.cpp"))
            assert cpp
            receipt["native"].append(
                {
                    "source": name,
                    "compilation_seconds_inclusive": time.perf_counter() - start,
                    "report": report,
                    "cpp_hashes": {str(p.relative_to(args.output)): digest(p) for p in cpp},
                }
            )
    receipt["complete"] = True
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
