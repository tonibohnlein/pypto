# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualify the fixed schedule catalogue against one actual-token/map baseline.

Run serially as a module. Large trip counts are construction-only measurements;
host control counts are not device timings. Unavailable modes remain explicit.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pypto import backend, codegen, ir, passes, pypto_core, testing

from .qualify_sync_control import compile_cpp
from .qualify_token_production import program_of
from .sync_control_metrics import static_control
from .sync_matmul_cases import staged_fixture
from .sync_physical_oracle import checked_payload_calls
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_parameterized_token_sync import apply, check, worker
from .test_rotating_sync_analysis import rotating_fixture
from .test_streamed_slot_composition import streamed_fixture
from .test_sync_case_only_schedule import case_only_fixture
from .test_sync_schedule_ablation import (
    MODES,
    assert_receipt,
    census,
    emitted_work,
    fingerprint,
    invocation_scaling_fixture,
    lower,
    predicate_sharing_pair,
    selected,
)
from .test_token_sync_lowering import _apply, _check_actual_tokens


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fixtures():
    """Yield bounded semantics and independent static/trip scaling dimensions."""
    for count, start, step in ((1, 0, 1), (2, 0, 1), (4, 5, 2), (7, 0, 1), (11, 5, 2), (17, 0, 1)):
        original = streamed_fixture(count, start=start, step=step)
        name = "rotating7" if count == 7 else f"streamed_{count}_{start}_{step}"
        yield name, program_of(original), original, _apply(passes.insert_sync, original), {}, [({}, None)]
    original = staged_fixture(3)
    yield "staged3", program_of(original), original, _apply(passes.insert_sync, original), {}, [({}, None)]
    original, logical = case_only_fixture()
    yield (
        "case_only",
        program_of(original),
        original,
        logical,
        {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, 1)},
        [({4: 0}, 0), ({4: 0}, 1)],
    )
    program = capture_qwen_projection("output", layers=2, band=3, hidden=512)
    original = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIC)
    logical = worker(apply(passes.insert_sync, program), original.name)
    evidence = testing.sync_invocation_evidence(program)
    bounds = tuple(evidence["block_indices"][original.name])
    yield (
        "qwen_output_b3_h512",
        program,
        original,
        logical,
        {"parameter_bounds": evidence["parameters"][original.name], "block_bounds": bounds},
        [({3: 0}, bounds[0]), ({3: 1}, bounds[1])],
    )
    for count in (70, 7000, 1000000):
        original = streamed_fixture(count)
        yield (
            f"trip_scaling_{count}",
            program_of(original),
            original,
            _apply(passes.insert_sync, original),
            {},
            [],
        )
    for cases in (2, 4, 8, 16):
        original, logical = invocation_scaling_fixture(cases)
        yield (
            f"case_scaling_{cases}",
            program_of(original),
            original,
            logical,
            {"parameter_bounds": {4: (0, 0)}, "block_bounds": (0, cases - 1)},
            [],
        )
    for stages in (1, 2, 4, 8):
        original = rotating_fixture(
            1700,
            [(3, 1)],
            [("load", 0, 0), *[("rmw", 0, 0)] * stages, ("store", 0, 0)],
            stream_input=True,
            stream_output=True,
        )
        yield (
            f"source_scaling_{stages}",
            program_of(original),
            original,
            _apply(passes.insert_sync, original),
            {},
            [],
        )


def codegen_owner(function):
    """Apply the ownership marker used by the production LowerSync pass wrapper."""
    return ir.Function(
        function.name,
        list(zip(function.params, function.param_directions)),
        function.return_types,
        function.body,
        function.span,
        function.func_type,
        level=function.level,
        role=function.role,
        attrs={**dict(function.attrs), "pypto_autosync": True},
        ir_stage=function.ir_stage,
    )


def qualify_fixture(name, input_program, original, logical, evidence, valuations, output, prior, emit_cpp):
    input_hash = hashlib.sha256(ir.serialize(input_program)).hexdigest()
    ir.serialize_to_file(input_program, str(output / f"{name}-input.msgpack"))
    ir.serialize_to_file(logical, str(output / f"{name}-logical.msgpack"))
    baseline = lower(logical, "guarded", **evidence)
    baseline_fingerprint = fingerprint(baseline)
    baseline_digest = hashlib.sha256(json.dumps(baseline_fingerprint, sort_keys=True).encode()).hexdigest()
    reference = {}
    for parameters, block in valuations:
        if block is None:
            _check_actual_tokens(original, logical, baseline["function"], check_codegen=False)
        else:
            check(
                original,
                logical,
                baseline["function"],
                parameters,
                block=block,
                minimum_protocol=name != "case_only",
            )
        reference[(tuple(parameters.items()), block)] = emitted_work(baseline["function"], parameters, block)
    rows = []
    for mode in (*MODES, "auto", "production_default"):
        started = time.perf_counter()
        result = (
            testing.lower_sync_schedule(logical, **evidence)
            if mode == "production_default"
            else lower(logical, mode, **evidence)
        )
        elapsed = time.perf_counter() - started
        requested = "auto" if mode == "production_default" else mode
        assert_receipt(result, requested)
        assert fingerprint(result) == baseline_fingerprint
        map_digest = hashlib.sha256(json.dumps(fingerprint(result), sort_keys=True).encode()).hexdigest()
        executions = []
        for parameters, block in valuations:
            checked_payload_calls(original, result["function"], parameters, block_index=block)
            state, work = emitted_work(result["function"], parameters, block)
            old, old_work = reference[(tuple(parameters.items()), block)]
            assert state.trace == old.trace
            if requested == "auto":
                assert sum(work) <= sum(old_work)
            executions.append(
                {
                    "parameters": parameters,
                    "block": block,
                    "work": work,
                    "guarded_work": old_work,
                    "exact_trace_equal": True,
                    "payload_operands_equal": True,
                }
            )
        actual = census(result["function"])
        selections = selected(result)
        matching = sum(r["mode"] == mode for r in selections)
        coverage = (
            (
                "all_reported_loops"
                if matching == len(selections)
                else "some_reported_loops"
                if matching
                else "none"
            )
            if mode in MODES
            else None
        )
        row = {
            "requested": mode,
            "selected_modes": [r["mode"] for r in selected(result)],
            "requested_mode_coverage": coverage,
            "policy": result["schedule_policy"],
            "schedule": result["schedule"],
            "static_payload_sites": actual.payloads,
            "static_command_sites": actual.commands,
            "expanded_ir_nodes": actual.nodes,
            "static_control": dict(static_control(result["function"])),
            "construction_seconds_single_run": elapsed,
            "executions": executions,
            "allocation_fingerprint_equal": True,
            "allocation_fingerprint_sha256": map_digest,
            "retained_cases": len(result["physical_cases"]),
            "retained_families": sum(len(case["families"]) for case in result["physical_cases"]),
        }
        physical = codegen_owner(result["function"])
        ir.serialize_to_file(physical, str(output / f"{name}-{mode}.msgpack"))
        if emit_cpp and valuations:
            source = output / f"{name}-{mode}.pto"
            source.write_text(codegen.PTOCodegen().generate(program_of(physical)))
            row["pto_bytes"] = source.stat().st_size
            row["cpp"] = compile_cpp(source, output / f"{name}-{mode}.cpp")
        if mode == "production_default" and prior and (prior / f"{name}.msgpack").exists():
            previous_receipt = json.loads((prior / "qualification.json").read_text())
            previous_row = next(row for row in previous_receipt["cases"] if row["name"] == name)
            assert previous_row["input_sha256"] == input_hash, "Prior artifact input does not match"
            previous = ir.deserialize_from_file(str(prior / f"{name}.msgpack"))
            if isinstance(previous, ir.Program):
                previous = worker(previous, original.name)
            comparison = []
            for parameters, block in valuations:
                checked_payload_calls(original, previous, parameters, block_index=block)
                old, old_work = emitted_work(previous, parameters, block)
                new, new_work = emitted_work(result["function"], parameters, block)
                assert old.trace == new.trace
                comparison.append(
                    {
                        "parameters": parameters,
                        "block": block,
                        "prior_work": old_work,
                        "default_work": new_work,
                        "trace_equal": True,
                    }
                )
            row["prior_default_comparison"] = {
                "status": "compared",
                "input_sha256_equal": True,
                "artifact_sha256": digest(prior / f"{name}.msgpack"),
                "prior_static_control": dict(static_control(previous)),
                "executions": comparison,
            }
        elif mode == "production_default":
            row["prior_default_comparison"] = {"status": "missing_artifact" if prior else "not_requested"}
        rows.append(row)
    return {
        "name": name,
        "input_sha256": input_hash,
        "logical_sha256": hashlib.sha256(ir.serialize(logical)).hexdigest(),
        "invocation_evidence": evidence,
        "baseline_allocation_fingerprint_sha256": baseline_digest,
        "original_expanded_nodes": census(original).nodes,
        "scope": "independent baseline token/access/flag proof plus exact mode traces and operands"
        if valuations
        else "construction only; no dynamic enumeration",
        "modes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--emit-cpp", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.set_backend_type(backend.BackendType.Ascend910B)
    rows = []
    for fixture in fixtures():
        row = qualify_fixture(*fixture, args.output, args.prior, args.emit_cpp)
        rows.append(row)
        print(row["name"], "qualified", flush=True)
    baseline, output = predicate_sharing_pair()
    predicates = []
    for coordinate in (-1, 0, 3, 8, 10):
        before, old_work = emitted_work(baseline, {4: coordinate})
        after, new_work = emitted_work(output, {4: coordinate})
        assert before.trace == after.trace and sum(new_work) < sum(old_work)
        predicates.append({"coordinate": coordinate, "before": old_work, "after": new_work})
    counts = {mode: {"all_reported_loops": 0, "some_reported_loops": 0, "none": 0} for mode in MODES}
    for fixture in rows:
        for row in fixture["modes"]:
            if row["requested"] in counts:
                counts[row["requested"]][row["requested_mode_coverage"]] += 1
    names = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    names += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    paths = sorted(
        {
            Path(name)
            for name in names
            if name.startswith(("src/", "include/", "python/", "tests/", "docs/")) and Path(name).is_file()
        }
    )
    prior_counts = {status: 0 for status in ("compared", "missing_artifact", "not_requested")}
    for fixture in rows:
        default = next(row for row in fixture["modes"] if row["requested"] == "production_default")
        prior_counts[default["prior_default_comparison"]["status"]] += 1
    receipt = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": {str(p): digest(p) for p in paths},
        "extension_sha256": digest(Path(pypto_core.__file__)),
        "command": [
            sys.executable,
            "-m",
            "tests.ut.ir.transforms.qualify_sync_schedule_ablation",
            *sys.argv[1:],
        ],
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "PTOAS_ROOT",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYPTO_CODEGEN_MAX_WORKERS",
            )
        },
        "python": sys.version,
        "mode_counts": counts,
        "mode_count_unit": "Fixture requests; partial means selection by only some reported loops.",
        "prior_default_comparison_counts": prior_counts,
        "fixtures": rows,
        "predicate_sharing": {
            "scope": "emitter-only control measurement; not protocol certification or case versioning",
            "payload_sites_before": census(baseline).payloads,
            "payload_sites_after": census(output).payloads,
            "executions": predicates,
        },
        "limits": [
            "Production defaults prohibit extra payload versions; cloning rows explicitly opt in.",
            "Scoped token ID maps remain period one; unavailable modes are not successful ablations.",
            "No device timing or native invocation-return proof.",
        ],
    }
    (args.output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
