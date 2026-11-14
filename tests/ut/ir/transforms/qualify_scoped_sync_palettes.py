# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Record diagnostic palette proposals for retained actual-token input.

This tool never changes production dispatch or emits a physical plan. Repeated
budget attempts deliberately include audit work; their timings are not ordinary
compiler timings.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pypto import backend, ir, testing
from pypto.backend import BackendType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    program = ir.deserialize_from_file(str(args.input))
    assert isinstance(program, ir.Program)
    workers = [f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC]
    assert len(workers) == 1, "Select a capture with exactly one AIC worker"
    pipes = [getattr(ir.PipeType, name) for name in ("MTE2", "MTE1", "M", "FIX", "V", "MTE3", "S")]
    receipt: dict[str, Any] = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "kind": "diagnostic-not-production",
        "maximum_proposed_budget": args.budget,
        "cases": [],
    }
    for block in args.blocks:
        start = time.perf_counter()
        order = testing.analyze_scoped_token_order(workers[0], block_bounds=(block, block))
        row: dict[str, Any] = {"block": block, "order_seconds": time.perf_counter() - start, "directions": []}
        for source in pipes:
            for target in pipes:
                if source == target:
                    continue
                attempts = []
                for budget in range(1, args.budget + 1):
                    start = time.perf_counter()
                    result = order.palette(source, target, budget)
                    seconds = time.perf_counter() - start
                    if not result["publications"] and result["status"] == "certified":
                        break
                    result["rank_piece_count"] = len(result.pop("pieces"))
                    attempts.append({"budget": budget, "seconds": seconds, **result})
                    if result["status"] != "unproved-reuse":
                        break
                if attempts:
                    row["directions"].append(
                        {"source": source.name, "target": target.name, "attempts": attempts}
                    )
        receipt["cases"].append(row)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
