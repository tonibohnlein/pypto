# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare the prior resource path and production on identical retained tokens.

Run with python and tests/ut/ir on PYTHONPATH. The legacy method disables only
the new palette provider; it does not reanalyze demands or change input effects.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from pypto import backend, ir, testing
from pypto.backend import BackendType

from transforms.sync_control_metrics import static_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    rows = []
    for path in args.inputs:
        program = ir.deserialize_from_file(str(path))
        assert isinstance(program, ir.Program)
        workers = [f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC]
        assert len(workers) == 1, "Select a capture with one AIC worker"
        function = workers[0]
        evidence = testing.sync_invocation_evidence(program)
        bounds = {k: (v[0], v[0]) for k, v in evidence["parameters"].get(function.name, {}).items()}
        for block in args.blocks:
            row = {
                "input": str(path),
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "block": block,
                "parameters": bounds,
            }
            try:
                order = testing.analyze_scoped_token_order(
                    function, parameter_bounds=bounds, block_bounds=(block, block)
                )
            except ValueError as error:
                row["unavailable"] = str(error)
                rows.append(row)
                continue
            texts = {}
            for label, run in (
                ("incumbent", lambda: order.realize_legacy(eligible_ids=list(range(6)))["function"]),
                ("production", lambda: order.realize(eligible_ids=list(range(6)))),
            ):
                start = time.perf_counter()
                try:
                    emitted = run()
                except ValueError as error:
                    row[label] = {"status": "refused", "reason": str(error)}
                else:
                    elapsed = time.perf_counter() - start
                    text = ir.python_print(emitted, format=False)
                    texts[label] = text
                    row[label] = {
                        "status": "allocated",
                        "seconds": elapsed,
                        "census": dict(static_control(emitted)),
                        "printed_ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
            if len(texts) == 2:
                row["identical_printed_ir"] = texts["incumbent"] == texts["production"]
            rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
