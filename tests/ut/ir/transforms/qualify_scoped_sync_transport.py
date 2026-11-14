# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualify public and restricted-ID lowering of an existing logical capture.

This is a host compiler receipt, not a device or independent semantic oracle.
Small emitted-program regressions supply the latter separately.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, passes, pypto_core, testing
from pypto.backend import BackendType
from pypto.backend.pto_backend import _run_ptoas


class CommandCounts(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.counts = Counter()
        self.ids = defaultdict(set)

    def visit_call(self, op):
        if op.op.name.startswith("system."):
            self.counts[op.op.name] += 1
            if op.op.name in (ir.get_op("system.set_flag").name, ir.get_op("system.wait_flag").name):
                key = f"{op.kwargs['src_pipe']}->{op.kwargs['dst_pipe']}"
                self.ids[key].add(op.kwargs["event_id"])
        return super().visit_call(op)


def emit(program, directory: Path) -> dict:
    start = time.perf_counter()
    text = codegen.PTOCodegen().generate(program)
    source = directory.with_suffix(".pto")
    cpp = directory.with_suffix(".cpp")
    source.write_text(text)
    pto_seconds = time.perf_counter() - start
    start = time.perf_counter()
    try:
        _run_ptoas(
            str(source), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"]
        )
    except RuntimeError as error:
        return {
            "status": "ptoas-refused",
            "reason": str(error),
            "pto_generation_seconds": pto_seconds,
            "ptoas_seconds": time.perf_counter() - start,
            "pto_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

    counts = CommandCounts()
    counts.visit_program(program)
    return {
        "status": "success",
        "pto_generation_seconds": pto_seconds,
        "ptoas_seconds": time.perf_counter() - start,
        "pto_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cpp_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
        "cpp_bytes": cpp.stat().st_size,
        "static_system_calls": dict(counts.counts),
        "directional_ids": {key: sorted(ids) for key, ids in counts.ids.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    parser.add_argument("--eligible", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    original = ir.deserialize_from_file(str(args.input))
    assert isinstance(original, ir.Program)
    workers = [f for f in original.functions.values() if f.func_type == ir.FunctionType.AIC]
    assert len(workers) == 1, "Select a capture with exactly one AIC worker"
    receipt: dict[str, Any] = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(Path(pypto_core.__file__).read_bytes()).hexdigest(),
        "device_validation": False,
        "python": sys.version,
        "public": {},
        "restricted_cases": [],
    }
    start = time.perf_counter()
    try:
        with passes.PassContext([]):
            physical = passes.lower_sync()(original)
    except ValueError as error:
        receipt["public"] = {
            "status": "refused",
            "reason": str(error),
            "lowering_seconds": time.perf_counter() - start,
        }
    else:
        public: dict[str, Any] = {"status": "allocated", "lowering_seconds": time.perf_counter() - start}
        receipt["public"] = public
        ir.serialize_to_file(physical, str(args.output / "public.msgpack"))
        core_functions = [
            f
            for f in physical.functions.values()
            if f.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV, ir.FunctionType.InCore)
        ]
        core_program = ir.Program(core_functions, "PublicCores", physical.span)
        public["output"] = emit(core_program, args.output / "public")
    for block in args.blocks:
        row = {"block": block, "eligible": args.eligible}
        start = time.perf_counter()
        try:
            order = testing.analyze_scoped_token_order(workers[0], block_bounds=(block, block))
        except ValueError as error:
            row.update(
                status="preparation-refused",
                reason=str(error),
                actual_token_preparation_seconds=time.perf_counter() - start,
            )
            receipt["restricted_cases"].append(row)
            continue
        row["actual_token_preparation_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        try:
            physical = order.realize(eligible_ids=args.eligible)
        except ValueError as error:
            row.update(status="refused", reason=str(error), lowering_seconds=time.perf_counter() - start)
        else:
            row.update(status="allocated", lowering_seconds=time.perf_counter() - start)
            program = ir.Program([physical], "ScopedTransport", physical.span)
            ir.serialize_to_file(program, str(args.output / f"block-{block}.msgpack"))
            row["output"] = emit(program, args.output / f"block-{block}")
        receipt["restricted_cases"].append(row)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
