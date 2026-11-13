# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare actual PTOAS/PyPTO command graphs on bounded original-input executions.

Requires the matching qualify_sync_control output directory. Large fixtures
retain static/control receipts; graph checks use explicitly labeled small inputs.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, testing
from pypto.backend.pto_backend import _run_ptoas

from .qualify_sync_control import fixtures, program_of
from .sync_cpp_order import compare_cpp_order, cpp_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.set_backend_type(backend.BackendType.Ascend910B)
    rows = []
    for name, program, valuations, independent in fixtures():
        if not (independent or name.startswith("lib_") or name.endswith("_h512")):
            continue
        original = next(
            f for f in program.functions.values() if f.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV)
        )
        row: dict[str, Any] = {
            "name": name,
            "input_sha256": hashlib.sha256(ir.serialize(program)).hexdigest(),
        }
        phase = "original C++ export"
        try:
            pto = args.output / (name + ".pto")
            pto.write_text(codegen.PTOCodegen().generate(program_of(original)))
            source = args.output / (name + "-ptoas.cpp")
            _run_ptoas(
                str(pto), str(source), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"]
            )
            candidate = args.physical / (name + ".cpp")
            row["sources"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, candidate)}
            evidence = testing.sync_invocation_evidence(program)
            phase = "bounded command graph"
            row["checks"] = []
            # First active worker and one boundary/empty worker; both layers
            # are covered separately by the complete ordered-trace campaign.
            selected = valuations if len(valuations) <= 2 else [valuations[0], valuations[-1]]
            for index, (parameters, block) in enumerate(selected):
                values = [
                    parameters[i] for i, p in enumerate(original.params) if isinstance(p.type, ir.ScalarType)
                ]
                if block is not None:
                    values += [block, evidence["block_indices"][original.name][1] + 1]
                before = cpp_trace(source, args.output / f"{name}-{index}-ptoas-trace", values)
                after = cpp_trace(candidate, args.output / f"{name}-{index}-pypto-trace", values)
                result = compare_cpp_order(original, before, after, parameters, block)
                row["checks"].append({"parameters": parameters, "block": block, **result})
                assert result["before"]["coverage"] and result["after"]["coverage"], (
                    "missing modeled payload coverage"
                )
            row["status"] = "checked"
        except (AssertionError, ValueError, KeyError) as error:
            row.update({"status": "unavailable-or-failed", "phase": phase, "reason": str(error)})
        rows.append(row)
        print(name, row["status"], row.get("reason", ""), flush=True)
        (args.output / "ordering.json").write_text(
            json.dumps(
                {
                    "command": sys.argv,
                    "cases": rows,
                    "scope": (
                        "Host command model; original macro/layout contracts assumed; "
                        "no native or device qualification."
                    ),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
