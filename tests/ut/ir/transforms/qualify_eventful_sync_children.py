# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Serial host-only before/after evidence for the initial local child family."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

from pypto import backend, codegen, ir, passes
from pypto.backend.pto_backend import _run_ptoas

from .sync_closed_command_oracle import ClosedCommands, check_closed_commands
from .sync_matmul_oracle import protected_accumulator_pairs, sequence_stmts
from .test_autosync_switch import clone
from .test_eventful_sync_children import BASES, count_statements, eventful_fixture, readonly_external_fixture
from .test_finite_sync_analysis import closure
from .test_logical_sync_realization import verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--readonly-external", action="store_true")
arguments = parser.parse_args()
ROOT = pathlib.Path.cwd()
OUTPUT = arguments.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
backend.reset_for_testing()
backend.set_backend_type(backend.BackendType.Ascend910)


def fixture(count):
    if arguments.readonly_external:
        function, scope, _ = readonly_external_fixture(count)
        return function, scope
    return eventful_fixture(count)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_commands(function):
    counts = {"set": 0, "wait": 0, "barrier": 0}
    pending = [function.body]
    while pending:
        statement = pending.pop()
        if isinstance(statement, ir.SeqStmts):
            pending.extend(statement.stmts)
        elif isinstance(statement, ir.ForStmt):
            pending.append(statement.body)
        elif isinstance(statement, ir.IfStmt):
            pending.append(statement.then_body)
            if statement.else_body is not None:
                pending.append(statement.else_body)
        elif isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call):
            name = statement.expr.op.name
            if name == ir.get_op("system.set_flag").name:
                counts["set"] += 1
            elif name == ir.get_op("system.wait_flag").name:
                counts["wait"] += 1
            elif any(
                name == ir.get_op(f"system.bar_{pipe}").name
                for pipe in ["all", "mte1", "mte2", "mte3", "m", "fix", "v"]
            ):
                counts["barrier"] += 1
    return counts


def concrete_required(original, parameters):
    reference = ClosedCommands(original, parameters, range(6), BASES)
    reference.run()
    protected = protected_accumulator_pairs(original, reference.calls)
    edges = set(reference.edges)
    for later, (_, _, reads, writes) in enumerate(reference.payloads):
        for earlier in range(later):
            _, _, previous_reads, previous_writes = reference.payloads[earlier]
            if (previous_writes & (reads | writes) or previous_reads & writes) and (
                earlier,
                later,
            ) not in protected:
                edges.add((("payload", earlier, "C"), ("payload", later, "I")))
    reference.order = closure(reference.vertices, edges)
    return payload_order(reference)


def payload_order(state):
    return {(a[1], b[1]) for a, b in state.order if a[0] == b[0] == "payload" and a[2] == "C" and b[2] == "I"}


def describe_witness(pair, state):
    def payload(index):
        call, env, pipe, _, _ = state.calls[index]
        return {
            "index": index,
            "operation": call.op.name,
            "pipe": str(pipe),
            "coordinates": {
                var.name_hint: value
                for var, value in env.items()
                if isinstance(var, ir.Var)
                and isinstance(var.type, ir.ScalarType)
                and isinstance(value, (int, bool))
            },
        }

    source, target = ("payload", pair[0], "C"), ("payload", pair[1], "I")
    previous: dict[tuple[Any, ...], tuple[Any, ...] | None] = {source: None}
    pending = [source]
    outgoing = {}
    for a, b in state.edges:
        outgoing.setdefault(a, set()).add(b)
    for node in pending:
        if node == target:
            break
        for successor in sorted(outgoing.get(node, ()), key=repr):
            if successor not in previous:
                previous[successor] = node
                pending.append(successor)
    assert target in previous
    path = []
    node = target
    while node is not None:
        item = {"event": repr(node)}
        if node in state.record_origins:
            item["record"] = state.record_origins[node]
        path.append(item)
        node = previous[node]
    return {
        "source": payload(pair[0]),
        "target": payload(pair[1]),
        "path": path[::-1],
        "present_before_physical_allocation": True,
    }


def build(function, children):
    start = time.perf_counter()
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync(eventful_children=children)(
            ir.Program([function], "MeasuredChild", function.span)
        )
    inserted = time.perf_counter()
    verify(next(iter(logical_program.functions.values())))
    verified = time.perf_counter()
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    finished = time.perf_counter()
    return (
        next(iter(logical_program.functions.values())),
        next(iter(physical_program.functions.values())),
        {
            "logical_insertion_seconds": inserted - start,
            "logical_verification_seconds": verified - inserted,
            "physical_realization_seconds": finished - verified,
        },
    )


rows = []
scaling = []
for count in [0, 1, 3]:
    original, scope = fixture(count)
    start = time.perf_counter()
    with passes.PassContext([]):
        child = passes.prepare_eventful_sync_children(original, [scope])[0]
    analysis_seconds = time.perf_counter() - start
    baseline = build(original, [])
    candidate = build(original, [child])
    for visits in [0, 1, 3]:
        for execute in [False, True]:
            states = []
            logical_states = []
            for logical, physical, _ in [baseline, candidate]:
                checked_logical, checked = check_closed_commands(
                    logical,
                    physical,
                    {3: visits, 4: execute},
                    range(6),
                    tensor_bases=BASES,
                    compare_commands=False,
                    hardware_source=original,
                )
                states.append(checked)
                logical_states.append(checked_logical)
            old, new = map(payload_order, states)
            required = concrete_required(original, {3: visits, 4: execute})
            assert required <= old and required <= new
            rows.append(
                {
                    "K": count,
                    "visits": visits,
                    "execute": execute,
                    "old_prerequisites": len(old),
                    "new_prerequisites": len(new),
                    "removed": len(old - new),
                    "added": len(new - old),
                    "added_witnesses": [
                        describe_witness(pair, logical_states[1]) for pair in sorted(new - old)[:3]
                    ],
                    "removed_witnesses": [
                        describe_witness(pair, logical_states[0]) for pair in sorted(old - new)[:3]
                    ],
                    "required": len(required),
                    "old_excess": len(old - required),
                    "new_excess": len(new - required),
                    "old_executed_set": len(states[0].publications),
                    "new_executed_set": len(states[1].publications),
                    "old_executed_wait": len(states[0].acquisitions),
                    "new_executed_wait": len(states[1].acquisitions),
                    "old_executed_barrier": states[0].barriers,
                    "new_executed_barrier": states[1].barriers,
                    "new_directional_ids": {
                        str(direction): sorted(ids) for direction, ids in states[1].ids.items()
                    },
                }
            )
    scaling.append(
        {
            "K": count,
            **child.describe(),
            "analysis_seconds": analysis_seconds,
            "baseline_cost": baseline[2],
            "candidate_cost": candidate[2],
            "baseline_static_commands": static_commands(baseline[1]),
            "candidate_static_commands": static_commands(candidate[1]),
        }
    )
    if count == 3:
        start = time.perf_counter()
        text = codegen.PTOCodegen().generate(
            ir.Program([candidate[1]], "MeasuredChild", original.span),
            emit_tile_addr=True,
            emit_source_loc=False,
        )
        pto = OUTPUT / "candidate.pto"
        cpp = OUTPUT / "candidate.cpp"
        pto.write_text(text)
        _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
        scaling[-1]["source_generation_seconds"] = time.perf_counter() - start
        scaling[-1]["generated_pto_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        scaling[-1]["generated_cpp_sha256"] = hashlib.sha256(cpp.read_bytes()).hexdigest()
for count in [2, 17, 1023, 1000000]:
    original, scope = fixture(count)
    start = time.perf_counter()
    with passes.PassContext([]):
        child = passes.prepare_eventful_sync_children(original, [scope])[0]
    scaling.append({"K": count, **child.describe(), "analysis_seconds": time.perf_counter() - start})
static_children = []
for number in [1, 2, 4, 8]:
    original, first = fixture(2)
    scopes = [first]
    for index in range(1, number):
        coordinate = ir.Var(f"child_{index}", first.loop_var.type, first.span)
        cloned, _ = ir.deep_clone(first.body)
        statements = list(sequence_stmts(cloned))
        scopes.append(
            ir.ForStmt(
                coordinate,
                first.start,
                first.stop,
                first.step,
                [],
                ir.SeqStmts(statements, first.span),
                [],
                first.span,
            )
        )
    root = list(sequence_stmts(original.body))
    outer = root[-2]
    assert isinstance(outer, ir.ForStmt)
    scope_statements: list[ir.Stmt] = [scope for scope in scopes]
    body = [
        ir.IfStmt(member.condition, ir.SeqStmts(scope_statements, first.span), None, [], member.span)
        if isinstance(member, ir.IfStmt) and member.then_body is first
        else member
        for member in sequence_stmts(outer.body)
    ]
    wrapper = ir.ForStmt(
        outer.loop_var, outer.start, outer.stop, outer.step, [], ir.SeqStmts(body, outer.span), [], outer.span
    )
    original = clone(original, body=ir.SeqStmts([*root[:-2], wrapper, root[-1]], original.span))
    start = time.perf_counter()
    with passes.PassContext([]):
        children = passes.prepare_eventful_sync_children(original, scopes)
    static_children.append(
        {
            "children": number,
            "source_nodes": count_statements(original.body),
            "analysis_seconds": time.perf_counter() - start,
            "interfaces": [child.describe() for child in children],
        }
    )
changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
untracked = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard"], text=True
).splitlines()
source_paths = sorted(set(changed + untracked))
binaries = {}
for name, module in sys.modules.copy().items():
    filename = getattr(module, "__file__", None)
    if name.startswith("pypto") and isinstance(filename, str) and filename.endswith(".so"):
        binaries[str(pathlib.Path(filename).resolve())] = digest(pathlib.Path(filename))
result = {
    "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "working_tree_sha256": {path: digest(ROOT / path) for path in source_paths if (ROOT / path).is_file()},
    "python": sys.executable,
    "python_version": sys.version,
    "binaries": binaries,
    "static_child_scaling": static_children,
    "scope": ("readonly external" if arguments.readonly_external else "local")
    + " eventful child, host-only, identical conservative-prefix policy",
    "timing_qualification": (
        "Single serial measurements; inclusive stages are not summed or speedup evidence."
    ),
    "checks": (
        "Actual-token/physical equality, independently recovered physical-byte conflicts, "
        "acyclicity, matching, causal reuse"
    ),
    "executions": rows,
    "scaling": scaling,
}
(OUTPUT / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
