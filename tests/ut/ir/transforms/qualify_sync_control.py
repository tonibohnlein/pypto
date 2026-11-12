# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce the production guard census and compare saved physical command traces.

Run as a module with --output DIR and optionally --baseline DIR containing
previous .msgpack outputs. This is host validation, not native timing.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from pypto import backend, codegen, ir, passes, pypto_core, testing
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .qualify_token_production import program_of
from .sync_control_metrics import static_control, trace_control
from .sync_matmul_cases import CASES, capture
from .sync_matmul_oracle import hardware_reference
from .sync_order_metrics import compare_order
from .sync_physical_oracle import run_physical
from .sync_prefill_cases import capture_prefill
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_buffered_sync_realization import local_sequence
from .test_collective_sync_realization import repeat_collective
from .test_nested_sync_repetition import caller_program, nested_fixture
from .test_parameterized_token_sync import apply, worker
from .test_row_sync_analysis import softmax_original_and_emitted
from .test_streamed_slot_composition import streamed_fixture
from .test_vec_sync_analysis import vec_fixture


def fixtures():
    """Yield original inputs plus explicitly bounded executions and oracle scope."""
    for name in CASES:
        yield name, program_of(capture(name)), [({}, None)], True
    for name, function in [
        ("vec", vec_fixture()),
        ("rotating7", streamed_fixture(7)),
        ("crossings", local_sequence((6, 5))),
        ("collective", repeat_collective(5, 3)),
        ("softmax", softmax_original_and_emitted()[0]),
    ]:
        yield name, program_of(function), [({}, None)], True
    yield "nested", caller_program(nested_fixture((2, 3))), [({1: 0, 3: 128}, None)], True
    for name in ("matmul", "gemm"):
        yield "lib_" + name, capture_prefill(name), [({1: 0, 3: 0}, None)], False
    for hidden in (512, 5120):
        for name, band in [("output", 3), *(("gate_up", b) for b in range(4))]:
            program = capture_qwen_projection(name, layers=2, band=band, hidden=hidden)
            values = [({3: layer}, b) for layer in (0, 1) for b in range(20 if name == "output" else 24)]
            yield f"qwen_{name}_b{band}_h{hidden}", program, values, False
    for count in (1, 70, 7000, 1000000):
        yield (
            f"rotating{count}",
            program_of(streamed_fixture(count)),
            [({}, None)] if count <= 70 else [],
            False,
        )
    for counts in ((2, 1000), (1000, 3), (1000000, 1000000)):
        yield f"nested_{counts}", caller_program(nested_fixture(counts)), [], False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def examine(name, program, valuations, independent, output, baseline):
    original = next(
        f for f in program.functions.values() if f.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV)
    )
    started = time.perf_counter()
    physical_program = apply(passes.realize_sync, program)
    elapsed = time.perf_counter() - started
    physical = worker(physical_program, original.name)
    assert dict(physical.attrs)["pypto_autosync"], dict(physical.attrs)
    old = None
    old_path = baseline / f"{name}.msgpack" if baseline else None
    if old_path:
        saved = ir.deserialize_from_file(str(old_path))
        assert isinstance(saved, ir.Program)
        old = worker(saved, original.name)
    text = codegen.PTOCodegen().generate(program_of(physical))
    stats = static_control(physical)
    assert stats["if_sites"] == text.count("scf.if")
    rows = []
    for parameters, block in valuations:
        actual = trace_control(physical, parameters, block)
        previous = trace_control(old, parameters, block) if old else None
        if previous:
            assert previous.trace == actual.trace, (name, block, "ordered command trace differs")
        if independent:
            state, _, order = run_physical(original, physical, range(6), parameters, block_index=block)
            old_order = (
                run_physical(original, old, range(6), parameters, block_index=block)[2] if old else order
            )
            ordering = compare_order(
                [c[2] for c in state.calls], hardware_reference(original, state.calls)[2], old_order, order
            )
        else:
            ordering = None
        rows.append(
            {
                "parameters": parameters,
                "block": block,
                "after": dict(actual.counts),
                "before": dict(previous.counts) if previous else None,
                "trace_events": len(actual.trace),
                "trace_equal": previous is not None,
                "ordering": ordering,
            }
        )
    # Read-only analysis diagnostics: measured separately, never counted as
    # ordinary production construction time or needed by the optimizer.
    evidence = testing.sync_invocation_evidence(program)
    construction = None
    if original.name in evidence["block_indices"]:
        retained = testing.analyze_sync_invocation(
            original,
            True,
            "a2a3",
            evidence["parameters"].get(original.name, {}),
            tuple(evidence["block_indices"][original.name]),
        )
        diagnostic = retained.realize_physical("a2a3", list(range(6)))
        construction = {"case_sites": diagnostic.get("case_sites")}
        for field in (
            "constructed_guard_sites",
            "shared_guard_sites",
            "control_statement_visits",
            "simplified_guard_sites",
            "shared_control_values",
            "cut_domain_interval_visits",
            "cut_case_comparisons",
            "coalesced_cut_cases",
            "control_expression_visits",
        ):
            construction[field] = (
                sum(c[field] for c in diagnostic["cases"])
                if all(field in c for c in diagnostic["cases"])
                else None
            )
    (output / f"{name}.pto").write_text(text)
    ir.serialize_to_file(physical_program, str(output / f"{name}.msgpack"))
    return {
        "name": name,
        "input_sha256": hashlib.sha256(ir.serialize(program)).hexdigest(),
        "baseline_ir_sha256": digest(old_path) if old_path else None,
        "before": dict(static_control(old)) if old else None,
        "after": dict(stats),
        "pto_arithmetic": text.count("arith."),
        "pto_bytes": len(text),
        "production_seconds_single_run": elapsed,
        "independent_access_order_matching_reuse": independent,
        "construction_audit": construction,
        "executions": rows,
    }


def compile_cpp(source, output):
    """Compile with downstream autosync enabled and census the emitted body."""
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    text = output.read_text()
    # The generic unused tail helper precedes the kernel. Count only the body
    # and explicitly ensure it has not acquired a reachable tail invocation.
    body = text.split("AICORE void ", 1)[1]
    assert "ptoas_auto_sync_tail(" not in body
    return {
        "source_sha256": digest(output),
        "bytes": len(text),
        "body_if_sites": body.count("if ("),
        "body_set_sites": body.count("set_flag("),
        "body_wait_sites": body.count("wait_flag("),
        "body_barrier_sites": body.count("pipe_barrier("),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--emit-cpp", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.set_backend_type(backend.BackendType.Ascend910B)
    rows = []
    for name, program, values, independent in fixtures():
        row = examine(name, program, values, independent, args.output, args.baseline)
        if args.emit_cpp:
            row["cpp_after"] = compile_cpp(args.output / f"{name}.pto", args.output / f"{name}.cpp")
            if args.baseline:
                row["cpp_before"] = compile_cpp(
                    args.baseline / f"{name}.pto", args.output / f"{name}-before.cpp"
                )
        rows.append(row)
        print(name, row["after"], flush=True)
    tracked = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    paths = [Path(p) for p in tracked if Path(p).is_file()]
    paths += [
        Path(__file__).with_name(name)
        for name in (
            "qualify_sync_control.py",
            "qualify_sync_order.py",
            "sync_control_metrics.py",
            "sync_cpp_order.py",
            "sync_order_metrics.py",
            "test_sync_command_control.py",
            "test_sync_cut_control.py",
            "test_sync_cpp_order.py",
            "test_sync_order_metrics.py",
        )
    ]
    receipt = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_source_sha256": {str(p.resolve().relative_to(Path.cwd())): digest(p) for p in paths},
        "extension_sha256": digest(pypto_core.__file__),
        "python": sys.version,
        "ptoas_launcher_sha256": digest(find_ptoas_binary()) if args.emit_cpp else None,
        "command": sys.argv,
        "cases": rows,
        "scope": "Source control and modeled host semantics; no device or optimized instruction timing.",
    }
    (args.output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
