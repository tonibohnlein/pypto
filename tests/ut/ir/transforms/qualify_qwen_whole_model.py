# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only current-library whole-Qwen compilation and fail-closed route inventory."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from pypto import backend, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig

from .sync_physical_oracle import run_physical


def digest(path: Path) -> str:
    """Hash a source or tool artifact without executing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Compile original model bodies with meta tensors, retaining refusal evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--mode", choices=("pypto", "ptoas"), default="pypto")
    parser.add_argument("--require-pypto", action="store_true")
    parser.add_argument("--tool-artifact", type=Path, action="append", default=[])
    parser.add_argument("--check-function", action="append", default=[])
    args = parser.parse_args()
    library, output = args.library.resolve(), args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory; stale artifacts cannot establish coverage")
    output.mkdir(parents=True)
    model = library / "models/qwen3_14b"
    root = Path.cwd()
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    sources = sorted(
        p for p in set(changed + added) if p.startswith(("src/", "include/", "python/", "tests/"))
    )
    tool = find_ptoas_binary()
    sys.path[:0] = [str(model), str(library)]
    receipt = {
        "compiler_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "library_revision": subprocess.check_output(
            ["git", "-C", str(library), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_sources": {p.name: digest(p) for p in sorted(model.glob("*.py"))},
        "configuration": {"layers": args.layers, "sequence": args.sequence, "batch": 1},
        "mode": args.mode,
        "command": sys.argv,
        "python": sys.version,
        "compiler_dirty_sources": {p: digest(root / p) if (root / p).is_file() else None for p in sources},
        "extension_hashes": {str(p): digest(p) for p in (root / "python/pypto").glob("pypto_core*.so")},
        "ptoas": {
            "path": tool,
            "hash": digest(Path(tool)) if tool else None,
            "version": subprocess.check_output([tool, "--version"], text=True).strip() if tool else None,
            "additional_artifacts": {str(p.resolve()): digest(p) for p in args.tool_artifact},
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("PYTHONPATH", "PTOAS_ROOT", "OMP_NUM_THREADS", "PYPTO_CODEGEN_MAX_WORKERS")
        },
        "access_contract": "Distinct input GM arguments assumed disjoint; no runtime values inferred",
        "device": "not run",
        "passes": [],
        "functions_before_sync": [],
        "compile_success": False,
        "all_generated_leaves_pypto": False,
    }
    started = time.perf_counter()

    def before(pass_obj, program):
        name = pass_obj.get_name()
        receipt["passes"].append(name)
        if name == "RealizeSync":
            (output / "before-sync.py").write_text(str(program))
            ir.serialize_to_file(program, str(output / "before-sync.msgpack"))
            receipt["functions_before_sync"] = [
                {
                    "name": function.name,
                    "type": str(function.func_type),
                    "stage": str(function.ir_stage),
                    "requires_runtime_binding": function.requires_runtime_binding,
                    "attrs": {str(k): str(v) for k, v in dict(function.attrs).items()},
                }
                for function in program.functions.values()
            ]

    def after(pass_obj, program):
        if pass_obj.get_name() == "RealizeSync":
            ir.serialize_to_file(program, str(output / "after-sync.msgpack"))

    try:
        module = importlib.import_module("prefill_fwd")
        specs = module.build_tensor_specs(
            batch=1, max_seq=args.sequence, num_layers=args.layers, use_max_seq=True
        )
        receipt["arguments"] = [
            {"name": spec.name, "shape": spec.shape, "dtype": str(spec.dtype)} for spec in specs
        ]
        tensors = [torch.empty(spec.shape, dtype=spec.dtype, device="meta") for spec in specs]
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        with passes.PassContext(
            [passes.CallbackInstrument(before_pass=before, after_pass=after)],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=args.mode == "pypto",
            autosync_disjoint_gm=True,
        ):
            module.prefill_fwd.compile(
                *tensors,
                config=RunConfig(
                    arch=BackendType.Ascend910,
                    codegen_only=True,
                    save_kernels_dir=str(output / "compiled"),
                ),
            )
        receipt["compile_success"] = True
    except Exception:
        receipt["error"] = traceback.format_exc()
    finally:
        receipt["compile_seconds_inclusive"] = time.perf_counter() - started
        report_path = output / "compiled/report/autosync.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        receipt["autosync"] = report
        cpp = sorted((output / "compiled/kernels").rglob("*.cpp"))
        receipt["cpp"] = {str(p.relative_to(output)): digest(p) for p in cpp}
        kernels = report.get("kernels", [])
        # Source census and route census must agree; unknown/external owners are
        # unresolved here, never silently counted as successful PyPTO leaves.
        leaves = {
            f["name"]
            for f in receipt["functions_before_sync"]
            if f["type"] in ("FunctionType.AIC", "FunctionType.AIV")
        }
        receipt["all_generated_leaves_pypto"] = (
            receipt["compile_success"]
            and bool(leaves)
            and (
                len(kernels) == len(cpp)
                and {k["function"] for k in kernels} == {p.stem for p in cpp} == leaves
                and all(k["selected_route"] == "pypto" for k in kernels)
            )
        )
        if args.check_function and receipt["compile_success"]:
            original = ir.deserialize_from_file(str(output / "before-sync.msgpack"))
            physical = ir.deserialize_from_file(str(output / "after-sync.msgpack"))
            evidence = testing.sync_invocation_evidence(original)
            originals = {f.name: f for f in original.functions.values()}
            emitted = {f.name: f for f in physical.functions.values()}
            receipt["independent_physical_checks"] = []
            try:
                for name in args.check_function:
                    assert any(k["function"] == name and k["selected_route"] == "pypto" for k in kernels)
                    assert not evidence["parameters"].get(name), (
                        "parameter valuation needs a separate contract"
                    )
                    first, last = evidence["block_indices"][name]
                    assert last - first < 64, "exhaustive launch oracle is intentionally bounded"
                    for block in range(first, last + 1):
                        run_physical(originals[name], emitted[name], range(6), block_index=block)
                    receipt["independent_physical_checks"].append(
                        {"function": name, "logical_blocks": [first, last], "passed": True}
                    )
            except Exception:
                receipt["oracle_error"] = traceback.format_exc()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if (
        not receipt["compile_success"]
        or "oracle_error" in receipt
        or (args.require_pypto and not receipt["all_generated_leaves_pypto"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
