# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproducible caller-evidence and parameterized token-stage measurements."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from pypto import backend, ir, passes, pypto_core, testing

from .sync_prefill_cases import capture_prefill
from .test_parameterized_token_sync import apply, check, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    cases = []
    for name, count in [("matmul", None), ("gemm", 4), ("gemm", 64), ("gemm", 1000000)]:
        program = capture_prefill(name, count)
        original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
        start = time.perf_counter()
        evidence = testing.sync_invocation_evidence(program)
        evidence_seconds = time.perf_counter() - start
        start = time.perf_counter()
        logical = apply(passes.insert_sync, program)
        insertion_seconds = time.perf_counter() - start
        start = time.perf_counter()
        physical = apply(passes.lower_sync, logical)
        lower_seconds = time.perf_counter() - start
        ops = {}

        class Count(ir.IRVisitor):
            def visit_call(self, op):
                ops[op.op.name] = ops.get(op.op.name, 0) + 1
                super().visit_call(op)

        Count().visit_function(worker(physical, original.name))
        if count is None or count == 4:
            for row, col in [(0, 0), (64, 128), (192, 192)]:
                check(
                    original,
                    worker(logical, original.name),
                    worker(physical, original.name),
                    {1: row, 3: col},
                )
        cases.append(
            {
                "kernel": name,
                "K_tiles": count,
                "input_sha256": hashlib.sha256(ir.serialize(program)).hexdigest(),
                "logical_sha256": hashlib.sha256(ir.serialize(logical)).hexdigest(),
                "caller_evidence": evidence,
                "timings_seconds": {
                    "diagnostic_caller_evidence": evidence_seconds,
                    "production_insert_inclusive": insertion_seconds,
                    "production_lower_inclusive": lower_seconds,
                },
                "timing_scope": "single descriptive runs; inclusive stages overlap; no speedup claim",
                "static_ops": ops,
                "independent_validation": "three coordinate valuations"
                if count is None or count == 4
                else "representation scaling only; no unfolding",
            }
        )
    sources = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    sources += [
        "tests/ut/ir/transforms/qualify_parameterized_tokens.py",
        "tests/ut/ir/transforms/test_parameterized_token_sync.py",
    ]
    result = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "extension_sha256": hashlib.sha256(Path(pypto_core.__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "command": [
            sys.executable,
            "-m",
            "tests.ut.ir.transforms.qualify_parameterized_tokens",
            "--output",
            str(args.output),
        ],
        "cases": cases,
        "alias_contract": "disjoint GM roots",
        "eligible_ids": list(range(6)),
        "limitations": [
            "uniform caller hull only; no activation partition",
            "actual-token lowering does not restore coverage after caller edits",
            "default production route unchanged",
            "no device run",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
