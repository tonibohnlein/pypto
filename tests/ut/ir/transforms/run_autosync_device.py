# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run original matmul examples under either synchronization owner.

Use --compile-only for host source generation. Device execution is explicit;
reported wall times include dispatch/transfers and are NOT kernel latency.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import pypto.pypto_core as core
import torch
from pypto import CacheConfig, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.runtime import ExecutionMode, RunConfig

from .run_sync_matmul_prototype import sha
from .sync_matmul_cases import CASES, ROOT, load_case


def run(args: argparse.Namespace) -> None:
    """Compile or execute one mode and persist partial results on failure."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = find_ptoas_binary()
    if binary is None:
        raise FileNotFoundError("Set PTOAS_ROOT to the pinned PTOAS v0.72 installation")
    check_ptoas_version(binary)
    receipt: dict[str, Any] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "python": platform.python_version(),
        "extension_sha256": sha(core.__file__),
        "ptoas": str(binary),
        "ptoas_version": subprocess.check_output([binary, "--version"], text=True).strip(),
        "ptoas_launcher_sha256": sha(binary),
        "runtime_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / "runtime", text=True
        ).strip(),
        "pto_isa_pin": (ROOT / "runtime/pto_isa.pin").read_text().strip(),
        "mode": args.mode,
        "device_execution_requested": not args.compile_only,
        "contract": "distinct contiguous GM allocations; manual a2a3 macros",
        "host_times_are_kernel_latency": False,
        "cases": [],
    }
    try:
        for name in args.cases:
            record: dict[str, Any] = {"name": name, "source_sha256": sha(ROOT / CASES[name][0])}
            receipt["cases"].append(record)
            m, k, n = CASES[name][1]
            torch.manual_seed(args.seed)
            a, b = torch.randn(m, k), torch.randn(k, n)
            c = torch.empty(m, n)
            config = RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                device_id=args.device,
                save_kernels=True,
                save_kernels_dir=str(output / name),
                dump_passes=True,
                dump_ptoas_passes=True,
                compile_profiling=True,
                cache_config=CacheConfig(enabled=False),
                enable_chip_swimlane=args.swimlane,
            )
            start = time.perf_counter()
            with passes.PassContext(
                [], enable_pypto_autosync=args.mode == "pypto", autosync_disjoint_gm=args.mode == "pypto"
            ):
                compiled = load_case(name).compile(a, b, c, config=config)
            record["source_generation_seconds"] = time.perf_counter() - start
            record["source_generation"] = "passed"
            record["artifact_dir"] = str(compiled.output_dir)
            sync_report = output / name / "report/autosync.json"
            record["sync_selection"] = json.loads(sync_report.read_text()) if sync_report.exists() else None
            if args.mode == "pypto":
                selection = record["sync_selection"]
                if (
                    not selection
                    or not selection["kernels"]
                    or any(kernel["selected_route"] != "pypto" for kernel in selection["kernels"])
                ):
                    raise RuntimeError(f"{name}: intended PyPTO device case delegated to PTOAS")
            if args.compile_only:
                continue
            # Separate device build/first invocation from warmed calls. Fresh
            # data and poisoned output catch stale data and incomplete stores.
            times = []
            for visit in range(args.warmup + args.trials):
                torch.manual_seed(args.seed + visit)
                a.copy_(torch.randn_like(a))
                b.copy_(torch.randn_like(b))
                c.fill_(float("nan"))
                golden = a.double() @ b.double()
                start = time.perf_counter()
                compiled(a, b, c, config=config)
                elapsed = time.perf_counter() - start
                torch.testing.assert_close(c.double(), golden, rtol=1e-3, atol=1e-3)
                if visit == 0:
                    record["first_dispatch_seconds_including_device_build"] = elapsed
                if visit >= args.warmup:
                    times.append(elapsed)
            record["numerical"] = "passed"
            record["checked_invocations"] = args.warmup + args.trials
            record["host_dispatch_seconds"] = times
    except Exception as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["generated_sha256"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file() and p.suffix in {".pto", ".cpp"}
        }
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pypto", "ptoas"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--swimlane", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.trials < 1:
        parser.error("warmup and trials must be positive")
    run(args)


if __name__ == "__main__":
    main()
