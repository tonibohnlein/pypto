# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce bounded physical allocation and command checks without a device."""

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pypto.pypto_core as core
from pypto import codegen, ir, testing
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .run_sync_matmul_prototype import sha, summary
from .sync_logical_oracle import CommandExecution
from .sync_matmul_cases import CASES, ROOT, capture, staged_fixture
from .sync_matmul_oracle import interpret, validate
from .sync_physical_oracle import directional_widths, run_physical


def run(output: Path, native: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    files = (
        subprocess.check_output(["git", "ls-files", "-m", "-o", "--exclude-standard", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    receipt: dict[str, Any] = {
        "base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: sha(ROOT / p) for p in files if p and not p.startswith("build-py314/")},
        "binary_sha256": sha(core.__file__),
        "python": platform.python_version(),
        "runtime_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / "runtime", text=True
        ).strip(),
        "pto_isa_pin": (ROOT / "runtime/pto_isa.pin").read_text().strip(),
        "configuration": {
            "target": "a2a3",
            "eligible_ids": list(range(6)),
            "alias_contract": "disjoint ND GM arguments",
            "native_mode": "manual",
        },
        "originals": [],
        "bounded": [],
        "scaling": [],
        "limitations": [
            "Host model only; no device compilation/execution",
            "Fresh entry; closed notifications do not imply payload completion at return",
            "Compact lane reservation is sufficient, not a minimum-ID theorem",
            "Single-run timings are descriptive",
        ],
    }
    if native:
        binary = find_ptoas_binary()
        assert binary is not None
        check_ptoas_version(binary)
        receipt["ptoas"] = {
            "launcher": str(binary),
            "sha256": sha(binary),
            "version": subprocess.check_output([str(binary), "--version"], text=True),
            "native_libraries": {
                str(p.relative_to(Path(binary).parent.parent)): sha(p)
                for p in sorted(Path(binary).parent.parent.rglob("*.so"))
            },
        }
    for name, (path, shape) in CASES.items():
        begin = time.perf_counter()
        function = capture(name)
        lower_seconds = time.perf_counter() - begin
        ir.serialize_to_file(function, str(output / f"{name}.input.msgpack"))
        begin = time.perf_counter()
        retained = testing.analyze_placed_sync(function, True)
        analysis_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        result: dict[str, Any] = dict(retained.realize_physical("a2a3", list(range(6))))
        physical_seconds = time.perf_counter() - begin
        emitted = result.pop("function")
        ir.serialize_to_file(emitted, str(output / f"{name}.physical.msgpack"))
        state, order, _ = run_physical(function, emitted, range(6))
        widths = directional_widths(state, order)
        # Separate original initialization branches from introduced flag tests.
        _, original_calls = interpret(function)
        original_state = CommandExecution(original_calls, [])
        original_state.execute(function.body, {})
        result.update(
            {
                "analysis": summary(retained),
                "analysis_checks": validate(function, retained),
                "directional_width": {f"{a}->{b}": w for (a, b), w in widths.items()},
                "assigned_ids": {f"{a}->{b}": sorted(ids) for (a, b), ids in state.ids.items()},
                "executed_sets": len(state.handoffs),
                "executed_waits": len(state.waits),
                "executed_barriers": len(state.barriers),
                "executed_added_guard_tests": state.branches - original_state.branches,
                "checked_causal_reuses": len(state.reuses),
                "modeled_payload_order_equal": True,
                "matching_closed": True,
                "lower_seconds": lower_seconds,
                "analysis_seconds": analysis_seconds,
                "physical_seconds": physical_seconds,
                "name": name,
                "shape": shape,
                "source_sha256": sha(ROOT / path),
                "input_sha256": sha(output / f"{name}.input.msgpack"),
                "physical_sha256": sha(output / f"{name}.physical.msgpack"),
            }
        )
        text = codegen.PTOCodegen().generate(
            ir.Program([emitted], "Physical", function.span), emit_tile_addr=True, emit_source_loc=False
        )
        original_text = codegen.PTOCodegen().generate(
            ir.Program([function], "Original", function.span), emit_tile_addr=True, emit_source_loc=False
        )
        # PTO-level inventory before downstream folding; not device instruction counts.
        result["added_pto_arithmetic_operations"] = len(re.findall(r"\barith\.\w+", text)) - len(
            re.findall(r"\barith\.\w+", original_text)
        )
        source = output / f"{name}.pto"
        source.write_text(text)
        if native:
            begin = time.perf_counter()
            cpp = output / f"{name}.cpp"
            _run_ptoas(
                str(source), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=false"]
            )
            result["cpp"] = {"sha256": sha(cpp), "seconds": time.perf_counter() - begin}
        receipt["originals"].append(result)
    for visits in [1, 2, 3]:
        for count in [0, 1, 2, 3, 5]:
            function = staged_fixture(count, visits=visits)
            result = testing.analyze_placed_sync(function, True).realize_physical("a2a3", list(range(6)))
            state, _, _ = run_physical(function, result["function"], range(6))
            receipt["bounded"].append(
                {
                    "visits": visits,
                    "count": count,
                    "executed_sets": len(state.handoffs),
                    "matching_reuse_and_order_equal": True,
                }
            )
    for count in [4, 8, 16, 64, 1024, 1000000]:
        function = staged_fixture(count, visits=2)
        begin = time.perf_counter()
        retained = testing.analyze_placed_sync(function, True)
        analysis_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        result = dict(retained.realize_physical("a2a3", list(range(6))))
        result.pop("function")
        result["families"] = len(result["families"])
        receipt["scaling"].append(
            {
                "count": count,
                "analysis": summary(retained),
                "analysis_seconds": analysis_seconds,
                "physical_seconds": time.perf_counter() - begin,
                **result,
            }
        )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Checked 3 originals, 15 bounded cases, 6 compact scaling cases: {output / 'receipt.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true", help="Require PTOAS and generate C++")
    args = parser.parse_args()
    run(args.output, args.native)
