# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Full configured Qwen projections: ordinary production and downstream C++ only."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig

from .qualify_parameterized_prefill import digest
from .sync_qwen_prefill_cases import MANIFEST, load_qwen_projection


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
        "stage": "M5 ordinary opt-in production and C++",
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
        "complete": False,
        "timing_claim": "Single descriptive times; pass time is included in compile time, not additive",
        "device": "Not run",
        "native_invocation_completion": "Not established",
    }
    for name, band in [("output", 3), *(("gate_up", b) for b in range(4))]:
        for enabled in [False, True]:
            module, entry = load_qwen_projection(name, layers=2, band=band)
            shapes = (
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
            tensors = [
                torch.empty(shape, dtype=torch.float32 if i == 0 else torch.bfloat16, device="meta")
                for i, shape in enumerate(shapes)
            ]
            output = args.output / f"{name}-band{band}-{'pypto' if enabled else 'ptoas'}"
            timing = {}

            def before(pass_obj, program):
                if pass_obj.get_name() == "RealizeSync":
                    timing["start"] = time.perf_counter()

            def after(pass_obj, program):
                if pass_obj.get_name() == "RealizeSync":
                    timing["seconds"] = time.perf_counter() - timing.pop("start")

            start = time.perf_counter()
            with passes.PassContext(
                [passes.CallbackInstrument(before_pass=before, after_pass=after)],
                enable_buffer_ir=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
                enable_pypto_autosync=enabled,
                autosync_disjoint_gm=True,
            ):
                entry.compile(
                    *tensors,
                    config=RunConfig(
                        arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(output)
                    ),
                )
            elapsed = time.perf_counter() - start
            report_path = output / "report/autosync.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            cpp = list(output.rglob("*.cpp"))
            assert cpp
            if enabled:
                assert report and report["kernels"] and report["downstream_autosync_enabled"]
                assert all(k["selected_route"] == "pypto" for k in report["kernels"]), report
            receipt["measurements"].append(
                {
                    "kernel": name,
                    "band": band,
                    "pypto_enabled": enabled,
                    "compile_seconds_inclusive": elapsed,
                    "realize_sync_seconds_included": timing.get("seconds"),
                    "report": report,
                    "cpp_hashes": {str(p.relative_to(args.output)): digest(p) for p in cpp},
                }
            )
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["complete"] = True
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
