# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce scoped-token size and host construction evidence (no flag claim)."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from pypto import backend, ir, passes, pypto_core, testing

from .test_collective_sync_realization import nested_collective
from .test_scoped_logical_sync import check
from .test_token_sync_lowering import _apply

SOURCES = [
    "src/ir/transforms/utils/scoped_logical_sync.cpp",
    "src/ir/transforms/utils/scoped_logical_sync.h",
    "src/ir/transforms/utils/sync_token_names.h",
    "src/ir/transforms/utils/sync_placement.h",
    "src/ir/transforms/utils/logical_sync_realization.cpp",
    "tests/ut/ir/transforms/test_scoped_logical_sync.py",
    "tests/ut/ir/transforms/qualify_scoped_token_insertion.py",
    "tests/ut/ir/transforms/test_collective_sync_realization.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(rows, counts):
    original = nested_collective(rows, counts)
    input_hash = hashlib.sha256(ir.serialize(ir.Program([original], "Input", original.span))).hexdigest()
    start = time.perf_counter()
    retained = testing.analyze_placed_sync(original, True)
    analysis_seconds = time.perf_counter() - start
    start = time.perf_counter()
    logical = retained.realize()
    logical_seconds = time.perf_counter() - start
    start = time.perf_counter()
    production = _apply(passes.insert_sync, original)
    production_seconds = time.perf_counter() - start
    ir.assert_structural_equal(production, logical["function"])
    description = retained.describe()
    result = {
        "rows": rows,
        "counts": counts,
        "input_sha256": input_hash,
        "diagnostic_analysis_seconds": analysis_seconds,
        "diagnostic_logical_export_seconds": logical_seconds,
        "production_insert_seconds": production_seconds,
        "timing_scope": "separate single runs; descriptive, no speedup claim; do not sum inclusive times",
        "logical_inventory": {key: value for key, value in logical.items() if key != "function"},
        "mathematical_inventory": {
            key: description[key]
            for key in [
                "merge_nodes",
                "interface_entries",
                "source_coordinate_entries",
                "collective_witnesses",
            ]
        },
        "production_same_tokens": True,
        "physical_lowering": "not requested by this insertion-only campaign; no allocation or C++ claim",
        "device": "not run",
    }
    if rows <= 5 and max(counts) <= 3:
        _, result["independent_semantics"] = check(original)
    else:
        result["independent_semantics"] = "large size check only; no unfolded execution"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    receipt = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: digest(path) for path in SOURCES},
        "extension_sha256": digest(pypto_core.__file__),
        "python": sys.version,
        "alias_contract": "disjoint distinct GM roots",
        "target": "a2a3",
        "scope": "M3 scoped logical insertion only; physical lowering not requested",
        "command": [
            sys.executable,
            "-m",
            "tests.ut.ir.transforms.qualify_scoped_token_insertion",
            "--output",
            str(args.output),
        ],
        "cases": [
            measure(rows, counts)
            for rows, counts in [
                (3, (2, 3)),
                (1000, (2, 3)),
                (1000000, (2, 3)),
                (3, (1000, 3)),
                (3, (2, 1000000)),
            ]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
