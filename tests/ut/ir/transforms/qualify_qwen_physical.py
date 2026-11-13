# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure scoped physical transport; original production dispatch is separate."""

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
from .sync_qwen_prefill_cases import MANIFEST, capture_qwen_projection
from .test_nested_sync_repetition import scoped_startup_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    changed = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    sources = sorted(
        p for p in set(changed + added) if p.startswith(("src/", "include/", "python/", "tests/"))
    )
    tool_path = find_ptoas_binary()
    assert tool_path is not None, "PTOAS is required for qualification"
    tool = Path(tool_path).resolve()
    receipt: dict[str, Any] = {
        "stage": "M4 scoped physical transport; ordinary production is M5",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {p: digest(root / p) for p in sources},
        "extension_hashes": {
            str(p.relative_to(root)): digest(p) for p in (root / "python/pypto").glob("pypto_core*.so")
        },
        "python": platform.python_version(),
        "input_manifest": MANIFEST,
        "ptoas": {
            "launcher_hash": digest(tool),
            "version": subprocess.check_output([str(tool), "--version"], text=True).strip(),
        },
        "measurements": [],
        "synthetic_scaling": [],
        "cpp": [],
        "timing_claim": (
            "Single serial descriptive timings; no speedup claim. Analysis and physical times are disjoint."
        ),
        "native_invocation_completion": "Unqualified beyond existing runtime/manual-mode contract",
        "device": "Not run",
        "complete": False,
    }

    def save():
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    for name in ["output", "gate_up"]:
        for band in [3] if name == "output" else range(4):
            for hidden in [512, 1024, 5120]:
                program = capture_qwen_projection(name, layers=2, band=band, hidden=hidden)
                worker = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
                t = time.perf_counter()
                owner = testing.analyze_sync_invocation(
                    worker, True, "a2a3", {3: (0, 1)}, (0, 19 if name == "output" else 23)
                )
                analysis_seconds = time.perf_counter() - t
                t = time.perf_counter()
                physical = owner.realize_physical("a2a3", list(range(6)))
                realization_seconds = time.perf_counter() - t
                cases = []
                for case in owner.describe()["cases"]:
                    result = owner.select_case(case["block_bounds"][0], {3: 0})
                    assert result is not None
                    d = result.describe()
                    cases.append(
                        {
                            k: d[k]
                            for k in [
                                "analyzed_occurrences",
                                "source_region_nodes",
                                "merge_nodes",
                                "interface_entries",
                                "scoped_export_builds",
                                "scoped_reuse_queries",
                            ]
                        }
                    )
                receipt["measurements"].append(
                    {
                        "name": name,
                        "band": band,
                        "hidden": hidden,
                        "analysis_seconds": analysis_seconds,
                        "physical_seconds": realization_seconds,
                        "cases": cases,
                        **{
                            k: physical[k]
                            for k in ["set_sites", "wait_sites", "barrier_sites", "guard_sites"]
                        },
                    }
                )
                if hidden == 5120 and band == 3:
                    pto = args.output / f"{name}.pto"
                    pto.write_text(
                        codegen.PTOCodegen().generate(
                            ir.Program([physical["function"]], "Projection", worker.span),
                            emit_tile_addr=True,
                            emit_source_loc=False,
                        )
                    )
                    outputs = []
                    for enabled in [False, True]:
                        cpp = args.output / f"{name}-{enabled}.cpp"
                        t = time.perf_counter()
                        _run_ptoas(
                            str(pto),
                            str(cpp),
                            [
                                "--pto-level=level3",
                                "--pto-arch=a2",
                                f"--enable-insert-sync={str(enabled).lower()}",
                            ],
                        )
                        outputs.append(cpp.read_text())
                        receipt["cpp"].append(
                            {
                                "name": name,
                                "downstream_autosync": enabled,
                                "seconds": time.perf_counter() - t,
                                "pto_hash": digest(pto),
                                "cpp_hash": digest(cpp),
                            }
                        )
                    assert outputs[0] == outputs[1], "downstream modified an owned manual plan"
                save()
    for count in [3, 16, 1024, 1000000]:
        t = time.perf_counter()
        retained = testing.analyze_placed_sync(scoped_startup_tree(1, count, count), True)
        analysis_seconds = time.perf_counter() - t
        t = time.perf_counter()
        physical = retained.realize_physical("a2a3", list(range(6)))
        physical_seconds = time.perf_counter() - t
        d = retained.describe()
        receipt["synthetic_scaling"].append(
            {
                "K_and_replay": count,
                "analysis_seconds": analysis_seconds,
                "physical_seconds": physical_seconds,
                **{k: d[k] for k in ["analyzed_occurrences", "merge_nodes", "scoped_reuse_queries"]},
                **{
                    k: physical[k]
                    for k in ["set_sites", "wait_sites", "barrier_sites", "guard_sites", "reuse_queries"]
                },
            }
        )
    receipt["complete"] = True
    save()


if __name__ == "__main__":
    main()
