# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Attempt an explicitly selected original captured child; no discovery or specialization."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, passes
from pypto.backend.pto_backend import _run_ptoas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_program = ir.deserialize_from_file(str(args.input))
    assert isinstance(original_program, ir.Program)
    function = original_program.get_function(args.function)
    assert function is not None
    scopes = []
    pending = [function.body]
    while pending:
        statement = pending.pop()
        if isinstance(statement, ir.SeqStmts):
            pending.extend(statement.stmts)
        elif isinstance(statement, ir.ForStmt):
            if statement.loop_var.name_hint == args.scope:
                scopes.append(statement)
            pending.append(statement.body)
        elif isinstance(statement, ir.IfStmt):
            pending.append(statement.then_body)
            if statement.else_body is not None:
                pending.append(statement.else_body)
    assert len(scopes) == 1, "request must identify exactly one original loop"
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    result: dict[str, Any] = {
        "command": sys.argv,
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "function": args.function,
        "scope": args.scope,
        "stages": {},
        "physical_shape_metadata": [
            {
                "name": symbol.name_hint,
                "original_formal_dimensions": [
                    [index, axis]
                    for index, parameter in enumerate(function.params)
                    if isinstance(parameter.type, ir.TensorType)
                    for axis, dimension in enumerate(parameter.type.shape)
                    if dimension is symbol
                ],
            }
            for symbol in codegen.collect_tensor_shape_dyn_vars(function)
        ],
        "device": "not run",
    }
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    result["dirty_sources_sha256"] = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(set(changed + added))
        if path.startswith(("src/", "include/", "python/", "tests/")) and Path(path).is_file()
    }
    result["python"] = sys.executable
    result["python_version"] = sys.version
    result["extension_sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path("python/pypto").glob("pypto_core*.so")
    }
    stage = "regional_preparation"
    start = time.perf_counter()
    try:
        with passes.PassContext([]):
            child = passes.prepare_eventful_sync_children(function, scopes)[0]
        result["stages"][stage] = {
            "status": "ready",
            "seconds": time.perf_counter() - start,
            "retained": child.describe(),
        }
        stage = "logical_composition"
        start = time.perf_counter()
        with passes.PassContext([], autosync_policy="conservative_prefix"):
            logical = passes.insert_sync(eventful_children=[child])(
                ir.Program([function], "RetainedChild", function.span)
            )
        result["stages"][stage] = {"status": "ready", "seconds": time.perf_counter() - start}
        ir.serialize_to_file(logical, str(args.output / "logical.msgpack"))
        stage = "physical_realization"
        start = time.perf_counter()
        with passes.PassContext([]):
            physical = passes.lower_sync()(logical)
        result["stages"][stage] = {"status": "ready", "seconds": time.perf_counter() - start}
        ir.serialize_to_file(physical, str(args.output / "physical.msgpack"))
        stage = "cpp_generation"
        start = time.perf_counter()
        pto = args.output / "child.pto"
        cpp = args.output / "child.cpp"
        pto.write_text(codegen.PTOCodegen().generate(physical, emit_tile_addr=True, emit_source_loc=False))
        _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
        result["stages"][stage] = {
            "status": "ready",
            "seconds": time.perf_counter() - start,
            "sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
        }
    except ValueError as error:
        result["stages"][stage] = {
            "status": "refused",
            "seconds": time.perf_counter() - start,
            "reason": str(error),
        }
    for name in ["regional_preparation", "logical_composition", "physical_realization", "cpp_generation"]:
        result["stages"].setdefault(name, {"status": "not_attempted"})
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
