# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce finite token-pipeline qualification; never execute a device."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, passes, testing
from pypto.backend import BackendType

from .sync_matmul_cases import capture, staged_fixture
from .test_token_sync_lowering import _check_actual_tokens
from .test_vec_sync_analysis import vec_fixture


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(function, tokens):
    program = ir.Program([function], "TokenQualification", function.span)
    result = {}
    with passes.PassContext([], autosync_disjoint_gm=True):
        begin = time.perf_counter()
        if tokens:
            logical = passes.insert_sync()(program)
            inserted = time.perf_counter()
            optimized = passes.optimize_sync()(logical)
            checked = time.perf_counter()
            physical = passes.lower_sync()(optimized)
            end = time.perf_counter()
            result.update(
                insert_seconds=inserted - begin,
                optimize_seconds=checked - inserted,
                lower_seconds=end - checked,
            )
            logical_function = next(iter(logical.functions.values()))
            emitted = next(iter(physical.functions.values()))
        else:
            retained = testing.analyze_placed_sync(function, True)
            analyzed = time.perf_counter()
            physical = retained.realize_physical("a2a3", list(range(6)))
            end = time.perf_counter()
            result.update(analysis_seconds=analyzed - begin, realization_seconds=end - analyzed)
            logical_function = None
            emitted = physical["function"]
        result["total_seconds"] = end - begin
    return result, logical_function, emitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structured", action="store_true")
    args = parser.parse_args()
    backend.set_backend_type(BackendType.Ascend910)
    root = Path(__file__).resolve().parents[4]
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=root, text=True).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src", "include", "python", "tests"],
        cwd=root,
        text=True,
    ).splitlines()
    sources = sorted(
        {
            p
            for p in changed + added
            if p == "CMakeLists.txt" or p.startswith(("src/", "include/", "python/", "tests/"))
        }
    )
    extension = next((root / "python/pypto").glob("pypto_core*.so"))
    receipt: dict[str, Any] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_source_sha256": {p: digest(root / p) for p in sources},
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=root)
        ).hexdigest(),
        "extension_sha256": digest(extension),
        "python": sys.version,
        "scope": "finite standalone RFC stages; ordinary production remains unchanged",
        "timing": "three interleaved pairs; access recovery is included, no speedup claim",
        "alias_contract": "disjoint distinct GM roots",
        "target": "a2a3",
        "device": "not run",
        "cases": [],
    }
    names = (
        ["ddr_split_k", "structured_k2", "structured_k1024", "structured_k1000000"]
        if args.structured
        else ["matmul_64", "matmul_acc_64", "finite_vec"]
    )
    if args.structured:
        receipt["scope"] = "canonical structured standalone token stages; production remains unchanged"
    for name in names:
        count = int(name.removeprefix("structured_k")) if name.startswith("structured_k") else None
        function = (
            staged_fixture(count, visits=2)
            if count is not None
            else (vec_fixture() if name == "finite_vec" else capture(name))
        )
        bounded = count is None or count <= 5
        row: dict[str, Any] = {
            "name": name,
            "input_sha256": hashlib.sha256(ir.python_print(function, format=False).encode()).hexdigest(),
            "pairs": [],
        }
        outputs = {}
        logical = None
        for pair in range(3):
            samples = {}
            for route in [False, True] if pair != 1 else [True, False]:
                sample, logical, emitted = measure(function, route)
                samples["tokens" if route else "direct"] = sample
                outputs[route] = emitted
                if route and bounded:
                    _, _, state = _check_actual_tokens(function, logical, emitted)
                    row["executed_commands"] = {
                        "set": len(state.records),
                        "wait": len(state.waits),
                        "barrier": len(state.barriers),
                    }
                    row["directional_ids"] = {
                        f"{a.name}->{b.name}": sorted(ids) for (a, b), ids in state.ids.items()
                    }
            row["pairs"].append(samples)
        assert logical is not None
        row["median_paired_total_ratio"] = statistics.median(
            p["tokens"]["total_seconds"] / p["direct"]["total_seconds"] for p in row["pairs"]
        )
        # Compare actual physical payload order for both constructions against
        # the same independently interpreted logical program, not just counts.
        if bounded:
            _check_actual_tokens(function, logical, outputs[False])
        texts = [
            codegen.PTOCodegen().generate(
                ir.Program([outputs[route]], "TokenQualification", function.span),
                emit_tile_addr=True,
                emit_source_loc=False,
            )
            for route in [False, True]
        ]
        row["physical_pto_byte_equal"] = texts[0] == texts[1]
        row["static_counts"] = [
            {
                "set": t.count("pto.set_flag"),
                "wait": t.count("pto.wait_flag"),
                "arithmetic": t.count("arith."),
            }
            for t in texts
        ]
        row["semantic"] = (
            "equal token/flag payload relations, access coverage, matching, causal reuse, width"
            if bounded
            else "size/timing only; bounded variants checked independently"
        )
        row["token_text_bytes"] = len(ir.python_print(logical, format=False))
        row["physical_pto_bytes"] = [len(t) for t in texts]
        receipt["cases"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
