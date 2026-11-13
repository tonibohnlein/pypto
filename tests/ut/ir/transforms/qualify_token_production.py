# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualify the opt-in production token pipeline without changing dispatch."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from pypto import backend, codegen, ir, passes, pypto_core, testing
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_cases import CASES, capture
from .sync_physical_oracle import run_physical
from .sync_qwen_prefill_cases import capture_qwen_projection
from .test_buffered_sync_realization import local_sequence
from .test_nested_sync_repetition import BOUNDS, caller_program, nested_fixture
from .test_parameterized_token_sync import apply, block_resident, callers, worker
from .test_streamed_slot_composition import streamed_fixture
from .test_vec_sync_analysis import vec_fixture


def program_of(function):
    return ir.Program([function], "Qualification", function.span)


def inventory(function):
    class Count(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls = Counter()
            self.carries = 0

        def visit_call(self, op):
            self.calls[op.op.name] += 1
            super().visit_call(op)

        def visit_for_stmt(self, op):
            self.carries += sum(isinstance(arg.type, ir.SyncTokenType) for arg in op.iter_args)
            super().visit_for_stmt(op)

    visitor = Count()
    visitor.visit_function(function)
    return {"calls": dict(visitor.calls), "token_carries": visitor.carries}


def direct_audit(program):
    """Explicit pre-migration construction; includes Python binding overhead."""
    evidence = testing.sync_invocation_evidence(program)
    result = {}
    for function in program.functions.values():
        if function.func_type not in (ir.FunctionType.AIC, ir.FunctionType.AIV):
            continue
        parameters = evidence["parameters"].get(function.name, {})
        blocks = evidence["block_indices"].get(function.name)
        if blocks is not None:
            owner = testing.analyze_sync_invocation(function, True, "a2a3", parameters, blocks)
        else:
            owner = testing.analyze_placed_sync(function, True, parameter_bounds=parameters)
        allocated = owner.realize_physical("a2a3", list(range(6)))
        assert allocated["set_sites"] and allocated["wait_sites"]
        selected = allocated["function"]
        # Reproduce the former successful wrapper's ownership tag only after
        # its full physical certificate succeeds, including nonempty markers.
        # This lets slot codegen use the same qualified path as that wrapper.
        result[function.name] = ir.Function(
            selected.name,
            list(zip(selected.params, selected.param_directions)),
            selected.return_types,
            selected.body,
            selected.span,
            selected.func_type,
            level=selected.level,
            role=selected.role,
            attrs={**dict(selected.attrs), "pypto_autosync": True},
            ir_stage=selected.ir_stage,
        )
    return result


def qualify(name, original, directory, valuations):
    baseline = direct_audit(original)
    start = time.perf_counter()
    physical = apply(passes.realize_sync, original)
    production_seconds = time.perf_counter() - start
    stages = {}
    logical = original
    for stage, factory in [("insert", passes.insert_sync), ("optimize", passes.optimize_sync)]:
        start = time.perf_counter()
        logical = apply(factory, logical)
        stages[stage] = time.perf_counter() - start
    archive = directory / f"{name}.tokens.msgpack"
    ir.serialize_to_file(logical, str(archive))
    restored = ir.deserialize_from_file(str(archive))
    assert isinstance(restored, ir.Program)
    start = time.perf_counter()
    standalone = apply(passes.lower_sync, restored)
    stages["lower"] = time.perf_counter() - start
    ir.assert_structural_equal(physical, standalone)
    checks, outputs = [], []
    for function in original.functions.values():
        if function.name not in baseline:
            continue
        selected = worker(physical, function.name)
        assert dict(selected.attrs)["pypto_autosync"]
        for parameters, block in valuations:
            old_state, _, old_order = run_physical(
                function, baseline[function.name], range(6), parameters, block_index=block
            )
            new_state, _, new_order = run_physical(
                function, selected, range(6), parameters, block_index=block
            )
            # Both directions of containment, with independent effects,
            # matching, actual command causality and resource checks above.
            assert new_order == old_order
            checks.append(
                {
                    "parameters": parameters,
                    "block": block,
                    "added_pairs": len(new_order - old_order),
                    "removed_pairs": len(old_order - new_order),
                    "old_executed_sets": len(old_state.records),
                    "new_executed_sets": len(new_state.records),
                }
            )
        text = codegen.PTOCodegen().generate(program_of(selected))
        audit_text = codegen.PTOCodegen().generate(program_of(baseline[function.name]))
        source, cpp = directory / f"{name}.pto", directory / f"{name}.cpp"
        source.write_text(text)
        _run_ptoas(
            str(source), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"]
        )
        outputs.append(
            {
                "function": function.name,
                "logical": inventory(worker(logical, function.name)),
                "physical": inventory(selected),
                "audit_physical": inventory(baseline[function.name]),
                "arithmetic_sites": text.count("arith."),
                "audit_arithmetic_sites": audit_text.count("arith."),
                "audit_pto_sha256": hashlib.sha256(audit_text.encode()).hexdigest(),
                "pto_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "cpp_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
            }
        )
    pairs = []
    for pair in range(3):
        elapsed = {}
        for route in ("audit", "production") if pair != 1 else ("production", "audit"):
            start = time.perf_counter()
            direct_audit(original) if route == "audit" else apply(passes.realize_sync, original)
            elapsed[route] = time.perf_counter() - start
        pairs.append(elapsed)
    return {
        "name": name,
        "input_sha256": hashlib.sha256(ir.serialize(original)).hexdigest(),
        "production_seconds": production_seconds,
        "standalone_stage_seconds": stages,
        "paired_seconds": pairs,
        "outputs": outputs,
        "independent_checks": checks,
        "independent_reference": (
            "fresh checks above"
            if valuations
            else "token-pipeline-m4.json: full Qwen actual-token gate, same layers=2/hidden=512/bands"
        ),
    }


def scaling():
    rows = []
    for counts in [(2, 3), (2, 1000), (1000, 3), (1000000, 1000000)]:
        function = nested_fixture(counts)
        original = caller_program(function)
        start = time.perf_counter()
        physical = apply(passes.realize_sync, original)
        elapsed = time.perf_counter() - start
        assert dict(worker(physical, function.name).attrs)["pypto_autosync"]
        logical = worker(apply(passes.insert_sync, original), function.name)
        parsed = testing.inspect_scoped_token_families(logical, parameter_bounds=BOUNDS)
        # Diagnostic construction is outside the production timing.
        description = testing.analyze_placed_sync(
            function, True, parameter_bounds=BOUNDS, original_frames=True
        ).describe()
        rows.append(
            {
                "counts": counts,
                "production_seconds": elapsed,
                "families": len(parsed["families"]),
                "axes": sum(len(f["axes"]) for f in parsed["families"]),
                "parse_work": parsed["work"],
                "logical": inventory(logical),
                "physical": inventory(worker(physical, function.name)),
                "mathematical_interface": {
                    key: description[key]
                    for key in (
                        "source_region_nodes",
                        "interface_cells",
                        "interface_ports",
                        "interface_entries",
                        "merge_nodes",
                    )
                },
            }
        )
    assert all(r["families"] == 7 and r["axes"] <= 5 for r in rows)
    cases = []
    function, _ = block_resident()
    for blocks in (1, 8, 10):
        original = callers(function, blocks=blocks)
        logical = worker(apply(passes.insert_sync, original), function.name)
        evidence = testing.sync_invocation_evidence(original)
        params = evidence["parameters"][function.name]
        domains = testing.analyze_sync_invocation(function, True, "a2a3", params, (0, blocks - 1)).describe()
        inspections = [
            testing.inspect_scoped_token_families(
                logical, parameter_bounds=params, block_bounds=tuple(case["block_bounds"])
            )["work"]
            for case in domains["cases"]
        ]
        cases.append(
            {
                "blocks": blocks,
                "case_count": len(inspections),
                "parse_work": inspections,
                "logical": inventory(logical),
            }
        )
    return {"trip_counts": rows, "activation_hulls_same_payload_body": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.parent / "token-production-artifacts"
    directory.mkdir(exist_ok=True, parents=True)
    backend.set_backend_type(backend.BackendType.Ascend910B)
    cases = [(name, program_of(capture(name)), [({}, None)]) for name in CASES]
    cases += [
        ("vec", program_of(vec_fixture()), [({}, None)]),
        ("rotating", program_of(streamed_fixture(7)), [({}, None)]),
        ("crossings", program_of(local_sequence((6, 5))), [({}, None)]),
        ("nested", caller_program(nested_fixture((2, 3))), [({1: 0, 3: 128}, None), ({1: 16, 3: 64}, None)]),
    ]
    results = [qualify(name, program, directory, valuations) for name, program, valuations in cases]
    # Reuse the full bounded independent Qwen gate, then establish equality of
    # this production wrapper with the same serialized token lowering here.
    qwen = [
        qualify(
            f"qwen_{name}_{band}",
            capture_qwen_projection(name, layers=2, band=band, hidden=512),
            directory,
            [],
        )
        for name, band in [("output", 3), *(("gate_up", b) for b in range(4))]
    ]
    sources = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    sources += [str(Path(__file__).relative_to(Path.cwd()))]
    binary = find_ptoas_binary()
    assert binary is not None
    receipt = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "extension_sha256": hashlib.sha256(Path(pypto_core.__file__).read_bytes()).hexdigest(),
        "ptoas_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "python": sys.version,
        "command": [
            sys.executable,
            "-m",
            "tests.ut.ir.transforms.qualify_token_production",
            "--output",
            str(args.output),
        ],
        "cases": results,
        "qwen_production": qwen,
        "scaling": scaling(),
        "timing_scope": (
            "Three interleaved pairs; audit includes Python API boundaries; no device/speedup claim. "
            "Standalone stage timings include separate evidence collection; "
            "they are not parts of production total."
        ),
        "contracts": {"target": "a2a3", "eligible_ids": list(range(6)), "disjoint_gm": True},
        "device": "not run",
        "default_enabled": False,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
