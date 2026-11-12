# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pin analytic invocation cases and costs of actual-token physical lowering."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from pypto import backend, codegen, ir, passes, pypto_core, testing

from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_parameterized_token_sync import apply, worker


def direct_audit(program):
    """Old direct construction, explicitly requested outside production dispatch."""
    evidence = testing.sync_invocation_evidence(program)
    functions = []
    for function in program.functions.values():
        if function.func_type != ir.FunctionType.AIC:
            functions.append(function)
            continue
        analysis = testing.analyze_sync_invocation(
            function,
            True,
            "a2a3",
            evidence["parameters"][function.name],
            evidence["block_indices"][function.name],
        )
        physical = analysis.realize_physical("a2a3")["function"]
        functions.append(physical)
    return ir.Program(functions, program.name, program.span)


def measure(name, hidden):
    program = capture_qwen_projection(name, layers=2, band=3, hidden=hidden)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    evidence = testing.sync_invocation_evidence(program)
    parameters = evidence["parameters"][original.name]
    blocks = evidence["block_indices"][original.name]
    start = time.perf_counter()
    try:
        logical = apply(passes.insert_sync, program)
    except ValueError as error:
        # Enlarging hidden also changes the output grid, not only K. Preserve
        # the existing multi-visit activation refusal as a measured outcome.
        elapsed = time.perf_counter() - start
        try:
            direct_audit(program)
            baseline_refusal = None
        except ValueError as baseline_error:
            baseline_refusal = str(baseline_error).split("\n", 1)[0]
        return {
            "kernel": name,
            "hidden": hidden,
            "band": 3,
            "layers": 2,
            "input_sha256": hashlib.sha256(ir.serialize(program)).hexdigest(),
            "insertion_refusal": str(error).split("\n", 1)[0],
            "insertion_seconds": elapsed,
            "baseline_accepted": baseline_refusal is None,
            "baseline_refusal": baseline_refusal,
        }
    insertion = time.perf_counter() - start
    start = time.perf_counter()
    optimized = apply(passes.optimize_sync, logical)
    optimization = time.perf_counter() - start
    start = time.perf_counter()
    physical = apply(passes.lower_sync, optimized)
    lowering = time.perf_counter() - start
    start = time.perf_counter()
    direct = direct_audit(program)
    direct_time = time.perf_counter() - start
    pairs = []
    for pair in range(3):
        elapsed = {}
        for route in ("direct", "tokens") if pair != 1 else ("tokens", "direct"):
            start = time.perf_counter()
            if route == "direct":
                direct_audit(program)
            else:
                apply(passes.lower_sync, apply(passes.optimize_sync, apply(passes.insert_sync, program)))
            elapsed[route] = time.perf_counter() - start
        pairs.append(elapsed)
    # Audit request is separate from production timings and never selects a route.
    description = testing.analyze_sync_invocation(original, True, "a2a3", parameters, blocks).describe()
    inventory = []
    for case in description["cases"]:
        interval = tuple(case["block_bounds"])
        start = time.perf_counter()
        result = testing.inspect_scoped_token_families(
            worker(logical, original.name), parameter_bounds=parameters, block_bounds=interval
        )
        inventory.append(
            {
                "blocks": interval,
                "work": result["work"],
                "families": len(result["families"]),
                "axes": sum(len(f["axes"]) for f in result["families"]),
                "audit_import_parse_seconds": time.perf_counter() - start,
            }
        )
    code = {}
    for route, result in [("tokens", physical), ("direct", direct)]:
        selected = worker(result, original.name)
        text = codegen.PTOCodegen().generate(ir.Program([selected], "P", original.span))
        code[route] = {
            "set": text.count("pto.set_flag"),
            "wait": text.count("pto.wait_flag"),
            "barrier": text.count("pto.barrier"),
            "arithmetic": text.count("arith."),
            "physical_pto_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "physical_ownership_marker": bool(dict(selected.attrs).get("pypto_autosync")),
        }
    return {
        "kernel": name,
        "hidden": hidden,
        "band": 3,
        "layers": 2,
        "input_sha256": hashlib.sha256(ir.serialize(program)).hexdigest(),
        "logical_sha256": hashlib.sha256(ir.serialize(logical)).hexdigest(),
        "timings_seconds": {
            "insert_inclusive": insertion,
            "optimize": optimization,
            "lower_inclusive": lowering,
            "baseline_direct_inclusive": direct_time,
        },
        "timing_scope": "single descriptive samples; no speedup claim or sum of inclusive/audit work",
        "cases": inventory,
        "code": code,
        "interleaved_pairs_seconds": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    cases = [measure(name, hidden) for name in ("output", "gate_up") for hidden in (512, 2048, 8192)]
    sources = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    sources += [
        "src/ir/transforms/utils/sync_invocation_domains.h",
        "tests/ut/ir/transforms/qualify_invocation_tokens.py",
        "tests/ut/ir/transforms/test_invocation_token_sync.py",
    ]
    receipt = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "extension_sha256": hashlib.sha256(Path(pypto_core.__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "command": [
            sys.executable,
            "-m",
            "tests.ut.ir.transforms.qualify_invocation_tokens",
            "--output",
            str(args.output),
        ],
        "cases": cases,
        "alias_contract": "disjoint GM roots",
        "eligible_ids": list(range(6)),
        "device": "not run",
        "default_dispatch": "off; opt-in production uses tokens",
        "baseline": "explicit direct audit API, not production RealizeSync",
        "semantics": "separate bounded test gate",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
