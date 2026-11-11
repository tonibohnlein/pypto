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

from pypto import backend, ir, testing
from pypto.backend import BackendType

from .test_slot_sync_import import slot_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
    for count, slots in [(7, 3), (1000, 3), (1000000, 3), (7, 101), (7, 1000003)]:
        function = slot_fixture(count, slots, alias=True)
        start = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - start
        description = result.describe()
        region = description["regions"][0]
        row = {
            "count": count,
            "slots": slots,
            "analysis_seconds_inclusive": elapsed,
            "interval_access_visits": description["interval_access_visits"],
            "qualified_intervals": description["qualified_intervals"],
        }
        for key in [
            "atom_tests",
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
