# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce the host-only analysis campaign; substantial outputs stay on disk."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import pypto.pypto_core as core
from pypto import ir, testing
from pypto.ir.printer import python_print

from .sync_matmul_cases import CASES, ROOT, capture, staged_fixture
from .sync_matmul_oracle import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(result):
    d = result.describe()
    out = {
        k: v
        for k, v in d.items()
        if k not in {"storage", "regions", "crossing_demands", "crossing_generators"}
    }
    out["storage_domains"] = len(d["storage"])
    out["crossing_demands"] = len(d["crossing_demands"])
    out["regions"] = [
        {
            "count": r["count"],
            "start": r["start"],
            "step": r["step"],
            "sites": len(r["sites"]),
            "families": len(r["demands"]),
            "generator_families": len(r["generators"]),
            "distance_entries": r["distance_entries"],
            "representation": "two-body periodic" if r["loop"] is not None else "finite numerical",
        }
        for r in d["regions"]
    ]
    return out


def run(output, draft):
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "binary_sha256": sha(core.__file__),
        "configuration": {
            "target": "Ascend910/A2-A3",
            "planner": "PYPTO",
            "buffer_ir": True,
            "alias_contract": "disjoint ND GM arguments",
            "mode": "analysis only",
        },
        "originals": [],
        "scaling": [],
        "bounded": [],
        "autotile_derivatives": [],
    }
    if draft:
        receipt["draft"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=draft, text=True).strip()
    source_files = (
        subprocess.check_output(["git", "ls-files", "-m", "-o", "--exclude-standard", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    receipt["source_sha256"] = {
        p: sha(ROOT / p)
        for p in source_files
        if p
        and not p.startswith("build-py314/")
        and (p.endswith((".cpp", ".h", ".py", ".pyi")) or p == "CMakeLists.txt")
    }
    for name, (path, shape) in CASES.items():
        begin = time.perf_counter()
        function = capture(name)
        lower_seconds = time.perf_counter() - begin
        ir.serialize_to_file(function, str(output / f"{name}.msgpack"))
        (output / f"{name}.py").write_text(python_print(function))
        result = testing.analyze_placed_sync(function, True)
        begin = time.perf_counter()
        checks = validate(function, result)
        receipt["originals"].append(
            {
                "name": name,
                "source": path,
                "source_sha256": sha(ROOT / path),
                "shape": shape,
                "input_sha256": sha(output / f"{name}.msgpack"),
                "lower_seconds": lower_seconds,
                "analysis": summary(result),
                "checks": checks,
                "oracle_seconds": time.perf_counter() - begin,
            }
        )
    for count in [0, 1, 2, 3, 5]:
        for visits in [1, 2, 3]:
            f = staged_fixture(count, visits=visits)
            r = testing.analyze_placed_sync(f, True)
            receipt["bounded"].append({"count": count, "visits": visits, **validate(f, r)})
    for axis, values in [
        ("count", [2, 4, 8, 16, 64, 1024, 1000000]),
        ("stages", [1, 2, 4, 8, 16]),
        ("visits", [1, 2, 4, 8]),
    ]:
        for value in values:
            args = {"count": 4, "stages": 2, "visits": 2, axis: value}
            r = testing.analyze_placed_sync(staged_fixture(**args), True)
            receipt["scaling"].append({"axis": axis, "value": value, **summary(r)})
    # Real AutoTile may change the schedule. Report that change rather than
    # using these captures as the fixed-schedule compactness proof.
    for k in [64, 192, 256]:
        begin = time.perf_counter()
        try:
            f = capture("ddr_split_k", (256, k, 256))
            r = testing.analyze_placed_sync(f, True)
            receipt["autotile_derivatives"].append(
                {
                    "k": k,
                    "status": "accepted",
                    "seconds": time.perf_counter() - begin,
                    "analysis": summary(r),
                    "checks": validate(f, r),
                }
            )
        except ValueError as error:
            receipt["autotile_derivatives"].append(
                {
                    "k": k,
                    "status": "unavailable",
                    "reason": str(error),
                    "seconds": time.perf_counter() - begin,
                }
            )
    receipt["scope"] = {
        "guarantee": "minimum demands for declared operation footprints and core native chains",
        "realization": "not requested",
        "allocation": "not requested",
        "device": "not available",
        "timings": "single-run descriptive, not speedup evidence",
        "oracle_dependency": (
            "independent IR traversal and physical addresses; shared declared native/layout model"
        ),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "originals": len(receipt["originals"]),
                "bounded": len(receipt["bounded"]),
                "output": str(output / "receipt.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    args = parser.parse_args()
    run(args.output, args.draft)
