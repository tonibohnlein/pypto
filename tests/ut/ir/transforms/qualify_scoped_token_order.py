# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure actual-token nested order, physical lowering and compact code generation."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from pypto import backend, codegen, ir, passes, pypto_core, testing

from .test_collective_sync_realization import nested_collective
from .test_scoped_token_order import check_order
from .test_token_sync_lowering import _apply

SOURCES = [
    "src/ir/transforms/utils/scoped_token_order.cpp",
    "src/ir/transforms/utils/scoped_token_order.h",
    "src/ir/transforms/utils/scoped_token_families.cpp",
    "src/ir/transforms/utils/structured_token_sync.cpp",
    "tests/ut/ir/transforms/test_scoped_token_order.py",
    "tests/ut/ir/transforms/qualify_scoped_token_order.py",
    "tests/ut/ir/transforms/test_collective_sync_realization.py",
]


def measure(rows, counts):
    original = nested_collective(rows, counts)
    start = time.perf_counter()
    logical = _apply(passes.insert_sync, original)
    insertion = time.perf_counter() - start
    start = time.perf_counter()
    order = testing.analyze_scoped_token_order(logical)
    proof = time.perf_counter() - start
    start = time.perf_counter()
    order.realize()
    realization = time.perf_counter() - start
    start = time.perf_counter()
    physical = _apply(passes.lower_sync, logical)
    lowering = time.perf_counter() - start
    text = codegen.PTOCodegen().generate(ir.Program([physical], "Physical", original.span))
    families = testing.inspect_scoped_token_families(logical)["families"]
    if rows < 6 and max(counts) < 4:
        check_order(logical)
    return {
        "rows": rows,
        "counts": counts,
        "input_sha256": hashlib.sha256(
            ir.serialize(ir.Program([original], "Input", original.span))
        ).hexdigest(),
        "logical_sha256": hashlib.sha256(
            ir.serialize(ir.Program([logical], "Logical", logical.span))
        ).hexdigest(),
        "timings_seconds": {
            "production_insert": insertion,
            "diagnostic_import_parse_order": proof,
            "diagnostic_allocation_emission": realization,
            "production_lower_inclusive": lowering,
        },
        "timing_scope": "single descriptive runs; do not sum overlapping columns or infer speedup",
        "distinct_nonempty_summary_nodes": order.summary_nodes,
        "distinct_row_entries": order.interface_entries,
        "families": len(families),
        "axes": sum(len(f["axes"]) for f in families),
        "pto_set_sites": text.count("pto.set_flag"),
        "pto_wait_sites": text.count("pto.wait_flag"),
        "pto_arithmetic_sites": text.count("arith."),
        "physical": "shared default six-ID catalog; accepted",
        "independent_semantics": "all event queries, proof inclusion, reuse, emitted matching/order"
        if rows < 6 and max(counts) < 4
        else "large representation-size check; not unfolded",
        "device": "not run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    data = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES},
        "extension_sha256": hashlib.sha256(Path(pypto_core.__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "alias_contract": "disjoint distinct GM roots",
        "target": "a2a3",
        "command": [
            sys.executable,
            "-m",
            "tests.ut.ir.transforms.qualify_scoped_token_order",
            "--output",
            str(args.output),
        ],
        "cases": [
            measure(rows, counts)
            for rows, counts in [
                (3, (3, 3)),
                (1023, (3, 3)),
                (1048575, (3, 3)),
                (3, (1023, 3)),
                (3, (1048575, 3)),
                (3, (3, 1048575)),
            ]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
