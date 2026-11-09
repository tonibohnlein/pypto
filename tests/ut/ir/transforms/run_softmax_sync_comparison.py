# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only softmax A/B: identical placed payloads, PTOAS autosync on in both modes."""

import argparse
import difflib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import pypto.pypto_core as core
from pypto import codegen, ir
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .run_sync_matmul_prototype import sha
from .sync_physical_oracle import run_physical
from .test_autosync_switch import apply_hybrid
from .test_row_sync_analysis import softmax_original_and_emitted


def kernel_body(text, name):
    """The selected fixture has one straight-line AIV body; exclude helper definitions."""
    start = text.index("AICORE void " + name + "(")
    return text[start:]


def inventory(body):
    # Follow the one actually called tail variant. The unused helper definition
    # contains both flags and an ALL barrier and must not inflate either count.
    tails = re.findall(r"ptoas_auto_sync_tail\(([^)]*)\);", body)
    assert all(t == "PTOAutoSyncTailMode::kBarrierAll" for t in tails), tails
    payloads = [
        line.strip()
        for line in body.splitlines()
        if re.match(r"\s*(?:TLOAD|TSTORE|TROWMAX|TROWSUM|TROWEXPANDSUB|TROWEXPANDDIV|TEXP)\(", line)
    ]
    assert len(payloads) == 7, "comparison is qualified for the original straight-line softmax only"
    return {
        "set_sites": len(re.findall(r"\bset_flag\(", body)),
        "wait_sites": len(re.findall(r"\bwait_flag\(", body)),
        "v_barriers": body.count("pipe_barrier(PIPE_V)"),
        "all_barriers_including_called_tail": body.count("pipe_barrier(PIPE_ALL)") + len(tails),
        "tail_calls": len(tails),
        "payload_calls": payloads,
        "body_lines": len(body.splitlines()),
        "body_bytes": len(body.encode()),
        "scope": "explicit kernel commands; identical native payload macro internals excluded",
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    binary = find_ptoas_binary()
    if binary is None:
        raise FileNotFoundError("Set PTOAS_ROOT to the pinned compiler")
    check_ptoas_version(binary)
    original, _ = softmax_original_and_emitted()
    start = time.perf_counter()
    emitted = apply_hybrid([original])[original.name]
    preparation = time.perf_counter() - start
    run_physical(original, emitted, range(6))
    flags = ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"]
    for mode, function in [("ptoas", original), ("pypto", emitted)]:
        ir.serialize_to_file(function, str(output / f"{mode}.msgpack"))
        text = codegen.PTOCodegen().generate(
            ir.Program([function], "softmax", function.span),
            emit_tile_addr=True,
            emit_source_loc=False,
        )
        (output / f"{mode}.pto").write_text(text)
        # Warm version discovery and executable/file caches before paired runs.
        _run_ptoas(str(output / f"{mode}.pto"), str(output / f"{mode}.cpp"), flags)
    times = {"ptoas": [], "pypto": []}
    pairs = []
    for pair in range(3):
        order = ["ptoas", "pypto"] if pair % 2 == 0 else ["pypto", "ptoas"]
        sample = {}
        for mode in order:
            start = time.perf_counter()
            _run_ptoas(str(output / f"{mode}.pto"), str(output / f"{mode}.cpp"), flags)
            sample[mode] = time.perf_counter() - start
            times[mode].append(sample[mode])
        pairs.append({"order": order, "seconds": sample})
    sources = {mode: (output / f"{mode}.cpp").read_text() for mode in times}
    counts = {mode: inventory(kernel_body(text, original.name)) for mode, text in sources.items()}
    assert counts["ptoas"]["payload_calls"] == counts["pypto"]["payload_calls"]
    (output / "kernel.diff").write_text(
        "".join(
            difflib.unified_diff(
                sources["ptoas"].splitlines(keepends=True),
                sources["pypto"].splitlines(keepends=True),
                fromfile="ptoas.cpp",
                tofile="pypto.cpp",
            )
        )
    )
    files = subprocess.check_output(
        ["git", "ls-files", "-m", "-o", "--exclude-standard", "-z"], text=True
    ).split("\0")
    receipt = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: sha(p) for p in files if p and not p.startswith("build-py314/")},
        "extension_sha256": sha(core.__file__),
        "ptoas": binary,
        "ptoas_sha256": sha(binary),
        "ptoas_version": subprocess.check_output([binary, "--version"], text=True).strip(),
        "ptoas_flags": flags,
        "input": "examples/intermediate/02_softmax.py::tile_softmax",
        "input_sha256": sha("examples/intermediate/02_softmax.py"),
        "configuration": "FP32 64x64, placed Buffer IR, PYPTO planner, disjoint GM arguments, a2a3",
        "inventory": counts,
        "pairs": pairs,
        "median_pto_to_cpp_seconds": {m: statistics.median(t) for m, t in times.items()},
        "median_paired_pypto_over_ptoas": statistics.median(
            s["seconds"]["pypto"] / s["seconds"]["ptoas"] for s in pairs
        ),
        "pypto_analysis_and_realization_seconds_single_run": preparation,
        "timing_scope": (
            "PTO-to-C++ subprocess, excludes common Python lowering and device compilation; no device timing"
        ),
        "device_validation": False,
        "checks": {
            "identical_payload_macro_calls": True,
            "pypto_modeled_order_matching_and_reuse": True,
            "ptoas_plan_independent_semantic_validation": False,
        },
        "artifact_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ["inventory", "median_pto_to_cpp_seconds", "median_paired_pypto_over_ptoas"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
