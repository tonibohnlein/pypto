# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pin Qwen M1 inputs, caller evidence and delegated full C++ compilation.

Run with --output on disk. No demand or physical claim is inferred from input
qualification; the M1 original whole kernels still delegate transactionally.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from pypto import backend, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig

from .qualify_parameterized_prefill import digest
from .sync_matmul_oracle import validate
from .sync_qwen_prefill_cases import FIXTURES, MANIFEST, capture_qwen_projection, load_qwen_projection
from .test_nested_sync_repetition import (
    expand_scoped_families,
    resident_pipeline_tree,
    scoped_startup_tree,
    structured_scaling_fixture,
)
from .test_sync_range_proofs import invocation_chain


def qualify_numerical(receipt):
    """Measure retained numerical constructions; no production route is forced."""
    receipt["stage"] = "M2 supported numerical resident/nested composition; full Qwen M3-M5 pending"
    receipt["static_scaling"] = []
    receipt["trip_scaling"] = []
    receipt["semantic_checks"] = []

    def summarize(function):
        started = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - started
        d = result.describe()
        metrics = {
            k: d[k]
            for k in [
                "analyzed_occurrences",
                "cells",
                "class_tests",
                "class_lookup_tests",
                "interface_cells",
                "interface_ports",
                "interface_entries",
                "interface_queries",
                "merge_nodes",
                "bridge_candidates",
                "source_region_nodes",
                "source_coordinate_entries",
                "import_seconds",
                "partition_seconds",
                "reduction_seconds",
                "composition_seconds",
            ]
        }
        metrics.update(
            {
                k: sum(r[k] for r in d["regions"])
                for k in [
                    "distance_entries",
                    "retained_frontier_entries",
                    "shortest_path_runs",
                    "shortest_path_edge_visits",
                ]
            }
        )
        metrics["static_sites"] = sum(len(r["sites"]) for r in d["regions"])
        metrics["leaf_generators"] = sum(len(r["generators"]) for r in d["regions"])
        metrics["leaf_demands"] = sum(len(r["demands"]) for r in d["regions"])
        metrics["seconds_inclusive"] = elapsed
        return result, metrics

    for axis in ["sites", "children", "classes"]:
        for size in [1, 2, 4, 8, 16]:
            _, row = summarize(structured_scaling_fixture(axis, size))
            receipt["static_scaling"].append({"axis": axis, "size": size, **row})
    for axis in ["inner", "outer", "replay"]:
        counts = [1, 2, 4] if axis == "inner" else [1, 10, 1000, 1000000]
        for count in counts:
            dims = dict(inner=2, outer=2, replay=2)
            dims[axis] = count
            _, row = summarize(resident_pipeline_tree(**dims))
            receipt["trip_scaling"].append({"axis": axis, "dimensions": dims, **row})
    for dims in [(0, 2, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (1, 2, 2), (2, 2, 2)]:
        for read in [False, True]:
            original = scoped_startup_tree(*dims, intermediate_read=read)
            result, row = summarize(original)
            checked = validate(original, result)
            try:
                result.realize_physical("a2a3", list(range(6)))
                physical = "accepted"
            except ValueError as error:
                physical = {"status": "unavailable", "reason": str(error)}
            receipt["semantic_checks"].append(
                {
                    "inner_k_replay": dims,
                    "intermediate_fix_read": read,
                    "checks": checked,
                    "metrics": row,
                    "physical_export": physical,
                }
            )
    for name in ["output", "gate_up"]:
        program = capture_qwen_projection(name, layers=2, band=3)
        evidence = testing.sync_invocation_evidence(program)
        worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
        try:
            testing.analyze_placed_sync(
                worker, True, parameter_bounds=evidence["parameters"].get(worker.name, {})
            )
            status = "accepted"
        except ValueError as error:
            status = {"status": "unavailable", "reason": str(error)}
        receipt["measurements"].append({"name": name, "analysis": status})
    receipt["timing_claim"] = "single descriptive samples, stages inclusive within total; no speedup claim"
    receipt["complete"] = True


def qualify_crossings(receipt, output, previous=None):
    """M3 source-scoped exports, exhaustive launch cases, independent bounded checks."""

    def progress(stage, **context):
        receipt["progress"] = {"stage": stage, **context}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(receipt["progress"], flush=True)

    receipt["stage"] = "M3 scoped demand/coordinate exports; physical transport remains M4"
    receipt["semantic_checks"] = [] if previous is None else previous["semantic_checks"]
    completed = {
        (row["name"], row["hidden"], row["band"], row["block"], row["layer"])
        for row in receipt["semantic_checks"]
    }
    receipt["trip_scaling"] = []

    def summarize(retained):
        d = retained.describe()
        export = retained.export_scoped_demands()
        return {
            **{
                k: d[k]
                for k in [
                    "analyzed_occurrences",
                    "cells",
                    "interface_entries",
                    "interface_ports",
                    "merge_nodes",
                    "source_region_nodes",
                    "source_coordinate_entries",
                ]
            },
            "scoped_families": len(export["families"]),
            "scoped_axis_entries": export["coordinate_entries"],
            "scoped_source_visits": export["visited_source_nodes"],
            "scoped_merge_visits": export["visited_merge_nodes"],
        }

    for name in ["output", "gate_up"]:
        for hidden in [512, 1024, 5120]:
            for band in [0] if name == "output" else range(4):
                progress("capture", name=name, hidden=hidden, band=band)
                program = capture_qwen_projection(name, layers=2, hidden=hidden, band=band)
                worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
                evidence = testing.sync_invocation_evidence(program)
                progress("analysis", name=name, hidden=hidden, band=band)
                start = time.perf_counter()
                owner = testing.analyze_sync_invocation(
                    worker,
                    True,
                    "a2a3",
                    evidence["parameters"][worker.name],
                    evidence["block_indices"][worker.name],
                )
                analysis_seconds = time.perf_counter() - start
                d = owner.describe()
                cases = []
                for case in d["cases"]:
                    lo, hi = case["block_bounds"]
                    progress("export", name=name, hidden=hidden, band=band, bounds=[lo, hi])
                    retained = owner.select_case(lo, {3: 0})
                    assert retained is not None
                    cases.append({"bounds": [lo, hi], **summarize(retained)})
                    if hidden == 512:
                        for block, layer in [(lo, 0), (hi, 1)]:
                            if (name, hidden, band, block, layer) in completed:
                                continue
                            progress("oracle", name=name, hidden=hidden, band=band, block=block, layer=layer)
                            oracle_start = time.perf_counter()
                            selected = owner.select_case(block, {3: layer})
                            assert selected is not None
                            checks = validate(worker, selected, parameters={3: layer}, block_index=block)
                            actual = expand_scoped_families(selected.export_scoped_demands())
                            assert actual == set(selected.expand_retained_demands(checks["payloads"]))
                            receipt["semantic_checks"].append(
                                {
                                    "name": name,
                                    "hidden": hidden,
                                    "band": band,
                                    "block": block,
                                    "layer": layer,
                                    "checks": checks,
                                    "scoped_records": len(actual),
                                    "oracle_seconds": time.perf_counter() - oracle_start,
                                }
                            )
                receipt["measurements"].append(
                    {
                        "name": name,
                        "hidden": hidden,
                        "band": band,
                        "cases": cases,
                        "case_count": len(cases),
                        "coordinate_nodes": d["coordinate_nodes"],
                        "coordinate_pieces": d["coordinate_pieces"],
                        "retained_sum": {
                            key: sum(case[key] for case in cases) for key in cases[0] if key != "bounds"
                        },
                        "analysis_seconds_inclusive": analysis_seconds,
                        "physical": "not requested; scoped transport remains M4",
                    }
                )
    for count in [1, 2, 10, 1000, 1000000]:
        progress("repeat_scaling", count=count)
        start = time.perf_counter()
        retained = testing.analyze_placed_sync(resident_pipeline_tree(2, count, count), True)
        receipt["trip_scaling"].append(
            {
                "outer_and_replay": count,
                **summarize(retained),
                "seconds_inclusive": time.perf_counter() - start,
            }
        )
    receipt["timing_claim"] = "Single descriptive measurements; per-case mathematics is charged in aggregate"
    receipt["complete"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-source", action="store_true")
    parser.add_argument(
        "--numerical", action="store_true", help="M2 numerical resident/source-tree qualification"
    )
    parser.add_argument("--crossings", action="store_true", help="M3 scoped/activation qualification")
    parser.add_argument("--resume", action="store_true", help="Resume pinned completed M3 oracle cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    root = Path(__file__).resolve().parents[4]
    changed = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    additions = [
        root / "src/ir/transforms/utils/invocation_sync_analysis.cpp",
        Path(__file__),
        Path(__file__).with_name("sync_qwen_prefill_cases.py"),
        Path(__file__).with_name("test_qwen_prefill_pipeline.py"),
        *FIXTURES.glob("*.source"),
        FIXTURES / "manifest.json",
    ]
    sources = {*changed, *(str(p.relative_to(root)) for p in additions)}
    receipt: dict[str, Any] = {
        "stage": "M1 input qualification; M2-M5 not implemented",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {p: digest(root / p) for p in sorted(sources)},
        "extension_hashes": {
            str(p.relative_to(root)): digest(p) for p in (root / "python/pypto").glob("pypto_core*.so")
        },
        "python": platform.python_version(),
        "input_manifest": MANIFEST,
        "contract": {
            "target": "a2a3",
            "disjoint_gm": True,
            "ids": list(range(6)),
            "entry": "standalone ready inputs, statically sized layer allocations",
            "device": "not run",
            "native_entry_return": "unqualified",
        },
        "measurements": [],
        "call_graph_scaling": [],
        "native_source": [],
    }
    if args.crossings:
        previous = None
        if args.resume:
            path = args.output / "receipt.json"
            previous = json.loads(path.read_text())
            assert previous["extension_hashes"] == receipt["extension_hashes"]
            assert previous["input_manifest"] == receipt["input_manifest"]
            driver = str(Path(__file__).relative_to(root))
            for name, value in previous["source_hashes"].items():
                if name != driver:  # Driver-only checkpoint support may change; semantics may not.
                    assert receipt["source_hashes"].get(name) == value, name
            receipt["resumed_checkpoint_sha256"] = digest(path)
            receipt["resume_scope"] = "Identical production, input, extension and independent-oracle hashes"
            path.with_name("resumed-partial.json").write_text(path.read_text())
        qualify_crossings(receipt, args.output, previous)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return
    if args.numerical:
        qualify_numerical(receipt)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return
    for depth in [8, 16, 32, 64]:
        receipt["call_graph_scaling"].append(
            {"depth": depth, **testing.sync_invocation_evidence(invocation_chain(depth))}
        )
    for name in ["output", "gate_up"]:
        for hidden in [512, 1024, 5120, 51200]:
            start = time.perf_counter()
            program = capture_qwen_projection(name, layers=2, band=3, hidden=hidden)
            lowered = time.perf_counter()
            evidence = testing.sync_invocation_evidence(program)
            ended = time.perf_counter()
            worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
            row = {
                "name": name,
                "hidden": hidden,
                "layers": 2,
                "band": 3,
                "seconds": {"frontend_placement": lowered - start, "caller_evidence": ended - lowered},
                "evidence": evidence,
                "timing_claim": "single descriptive measurement",
            }
            try:
                testing.analyze_placed_sync(
                    worker, True, parameter_bounds=evidence["parameters"].get(worker.name, {})
                )
                row["analysis"] = "accepted"
            except ValueError as error:
                row["analysis"] = {"status": "unavailable", "reason": str(error)}
            receipt["measurements"].append(row)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.native_source:
        tool = find_ptoas_binary()
        assert tool is not None
        tool_path = Path(tool).resolve()
        receipt["ptoas"] = {
            "launcher_hash": digest(tool_path),
            "version": subprocess.check_output([tool, "--version"], text=True).strip(),
            "binary_hashes": {
                str(p.relative_to(tool_path.parent.parent)): digest(p)
                for p in (tool_path.parent.parent / "ptoas").rglob("*.so")
            },
        }
        for name in ["output", "gate_up"]:
            module, entry = load_qwen_projection(name, layers=2, band=3)
            specs = (
                [
                    (module.TOK_TILE, module.HIDDEN),
                    (module.TOK_TILE, module.HIDDEN),
                    (2 * module.HIDDEN, module.HIDDEN),
                ]
                if name == "output"
                else [
                    (2 * module.MLP_M_TILE, module.MLP_BAND_WIDTH),
                    (module.MLP_M_TILE, module.HIDDEN),
                    (2 * module.HIDDEN, module.INTERMEDIATE),
                    (2 * module.HIDDEN, module.INTERMEDIATE),
                ]
            )
            inputs = [
                torch.empty(shape, dtype=torch.float32 if i == 0 else torch.bfloat16, device="meta")
                for i, shape in enumerate(specs)
            ]
            output = args.output / name
            start = time.perf_counter()
            with passes.PassContext(
                [],
                enable_buffer_ir=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
                enable_pypto_autosync=True,
                autosync_disjoint_gm=True,
            ):
                entry.compile(
                    *inputs,
                    config=RunConfig(
                        arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(output)
                    ),
                )
            report = json.loads((output / "report/autosync.json").read_text())
            cpp = list(output.rglob("*.cpp"))
            assert cpp and report["kernels"]
            receipt["native_source"].append(
                {
                    "name": name,
                    "seconds_inclusive": time.perf_counter() - start,
                    "report": report,
                    "cpp_hashes": {str(p.relative_to(args.output)): digest(p) for p in cpp},
                }
            )
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["complete"] = True
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
