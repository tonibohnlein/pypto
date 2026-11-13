# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Attribute demand-to-command expansion without changing production decisions.

Diagnostics reconstruct the same admitted inputs separately and are timed as
audits. Exact schema comparisons retain original loop-variable identities;
cross-case grouping is only an opportunity census, never permission to merge.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from pypto import backend, ir, passes, pypto_core, testing

from .qualify_sync_control import fixtures
from .qualify_token_production import inventory
from .sync_control_metrics import ControlTrace, statements, static_control, trace_control
from .sync_logical_oracle import value
from .sync_order_metrics import compare_order
from .sync_physical_oracle import run_physical
from .test_parameterized_token_sync import apply, worker


def family_key(family):
    """Compare exact coordinates/axes within one imported original frame."""
    # A singleton Cartesian axis has only digit zero and contributes no
    # source/target displacement. The actual token parser may omit it.
    axes = tuple((axis[0].loop_var, *axis[1:]) for axis in family["axes"] if axis[1] != 1)
    return tuple(family["source"]), tuple(family["target"]), axes


def guard_accounting(original, physical, valuations) -> dict[str, Any]:
    """Classify dependencies, not estimated hardware latency or semantic necessity."""
    definitions = {s.var: s.value for s in statements(physical.body) if isinstance(s, ir.AssignStmt)}
    loop_vars = {s.loop_var for s in statements(physical.body) if isinstance(s, ir.ForStmt)}
    original_conditions = {
        ir.structural_hash(s.condition, False) for s in statements(original.body) if isinstance(s, ir.IfStmt)
    }
    memo = {}

    def dependencies(expression, seen):
        if expression in memo:
            return memo[expression]
        if expression in seen:
            return frozenset({"unknown"})
        seen = seen | {expression}
        if isinstance(expression, ir.Var):
            if expression in loop_vars:
                result = frozenset({"iteration"})
            elif expression in definitions:
                result = dependencies(definitions[expression], seen)
            elif expression in original.params:
                result = frozenset({"parameter"})
            else:
                result = frozenset({"unknown"})
        elif isinstance(expression, ir.BinaryExpr):
            result = dependencies(expression.left, seen) | dependencies(expression.right, seen)
            if isinstance(expression, ir.FloorMod):
                result |= {"modulo"}
        elif isinstance(expression, ir.UnaryExpr):
            result = dependencies(expression.operand, seen)
        elif isinstance(expression, ir.Call):
            result = frozenset(
                {"launch" if expression.op.name == ir.get_op("buffer.get_block_idx").name else "unknown"}
            )
        elif isinstance(expression, (ir.ConstInt, ir.ConstBool, ir.ConstFloat)):
            result = frozenset()
        else:
            result = frozenset({"unknown"})
        memo[expression] = result
        return result

    roles = {}
    for statement in statements(physical.body):
        if not isinstance(statement, ir.IfStmt):
            continue
        if ir.structural_hash(statement.condition, False) in original_conditions:
            role = "original_condition"
        else:
            deps = dependencies(statement.condition, frozenset())
            role = "+".join(sorted(deps)) or "constant"
        roles[statement] = role

    class AttributedTrace(ControlTrace):
        def __init__(self):
            super().__init__()
            self.roles = Counter()
            self.inactive = Counter()

        def execute(self, stmt, env):
            if stmt in roles:
                self.roles[roles[stmt]] += 1
                self.inactive[roles[stmt]] += not value(stmt.condition, env)
            return super().execute(stmt, env)

    executed = []
    selected = valuations if len(valuations) <= 2 else [valuations[0], valuations[-1]]
    for parameters, block in selected:
        state = AttributedTrace()
        env = {physical.params[i]: v for i, v in parameters.items()}
        if block is not None:
            env["__block_index__"] = block
        state.execute(physical.body, env)
        assert sum(state.roles.values()) == state.counts["conditions"]
        executed.append(
            {
                "parameters": parameters,
                "block": block,
                "evaluated": dict(state.roles),
                "false": dict(state.inactive),
            }
        )
    return {"static": dict(Counter(roles.values())), "executions": executed}


def case_inventory(original, mathematical, parsed, allocated):
    """Describe one case; preserve family multiplicity and actual ID assignments."""
    description = mathematical.describe()
    exported = mathematical.export_scoped_demands()
    token_families = parsed["families"] if parsed else None
    if exported is not None and token_families is not None:
        assert Counter(map(family_key, exported["families"])) == Counter(map(family_key, token_families)), (
            "math/token schemas differ"
        )
    if token_families is not None:
        assert Counter(map(family_key, token_families)) == Counter(map(family_key, allocated["families"])), (
            "token/physical schemas differ"
        )
    calls = {
        statement.expr: index
        for index, statement in enumerate(statements(original.body))
        if isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call)
    }
    loops = {
        statement.loop_var: index
        for index, statement in enumerate(statements(original.body))
        if isinstance(statement, ir.ForStmt)
    }
    rows = []
    for f in allocated["families"]:
        sr, sv, ss = f["source"]
        tr, tv, ts = f["target"]
        source = description["regions"][sr]["sites"][ss]
        target = description["regions"][tr]["sites"][ts]
        axes = [[loops[axis[0].loop_var], *axis[1:]] for axis in f["axes"]]
        rows.append(
            {
                "source": [calls[source["original"]], sv],
                "target": [calls[target["original"]], tv],
                "source_op": source["original"].op.name,
                "target_op": target["original"].op.name,
                "direction": [int(source["pipe"]), int(target["pipe"])],
                "cross_segment": sr != tr,
                "axes": axes,
                "count": f["count"],
                "ids": f["ids"],
            }
        )
    return {
        "hierarchical": description["hierarchical"],
        "segment_count": len(description["regions"]),
        "sites": sum(len(r["sites"]) for r in description["regions"]),
        "internal_generators": sum(len(r["generators"]) for r in description["regions"]),
        "internal_demands": sum(len(r["demands"]) for r in description["regions"]),
        "flat_crossings": description["crossing_demands"],
        "scoped_mathematical_families": len(exported["families"]) if exported is not None else None,
        "token_families": len(token_families) if token_families is not None else None,
        "physical_families": len(rows),
        "token_parse": parsed["work"]
        if parsed
        else {"unavailable": "Flat transport does not use the scoped-token inspection interface"},
        "physical_sites": {key: allocated[key] for key in ("set_sites", "wait_sites", "guard_sites")},
        "id_widths": dict(Counter(len(f["ids"]) for f in rows)),
        "cross_segment_families": sum(f["cross_segment"] for f in rows),
        "dynamic_handoffs": sum(f["count"] for f in rows if f["ids"]),
        "families": rows,
        "retained": {
            key: description[key]
            for key in ("interface_cells", "interface_ports", "interface_entries", "source_region_nodes")
        },
    }


def measure(name, program, valuations, independent):
    original = next(
        f for f in program.functions.values() if f.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV)
    )
    start = time.perf_counter()
    production = apply(passes.realize_sync, program)
    production_seconds = time.perf_counter() - start
    selected = worker(production, original.name)
    assert dict(selected.attrs)["pypto_autosync"]
    logical = program
    stages, inventories = {}, {}
    for label, factory in [("insert", passes.insert_sync), ("optimize", passes.optimize_sync)]:
        start = time.perf_counter()
        logical = apply(factory, logical)
        stages[label] = time.perf_counter() - start
        inventories[label] = inventory(worker(logical, original.name))
    start = time.perf_counter()
    lowered = apply(passes.lower_sync, logical)
    stages["lower"] = time.perf_counter() - start
    ir.assert_structural_equal(production, lowered)
    token_function = worker(logical, original.name)
    evidence = testing.sync_invocation_evidence(program)
    parameters = evidence["parameters"].get(original.name, {})
    blocks = evidence["block_indices"].get(original.name)
    start = time.perf_counter()
    if blocks is not None:
        owner = testing.analyze_sync_invocation(original, True, "a2a3", parameters, blocks)
        bounds = [tuple(c["block_bounds"]) for c in owner.describe()["cases"]]
        allocation = owner.realize_physical("a2a3", list(range(6)))
        allocations = allocation["cases"]
        mathematics = [owner.select_case(b[0], {i: v[0] for i, v in parameters.items()}) for b in bounds]
    else:
        owner = testing.analyze_placed_sync(original, True, parameter_bounds=parameters)
        bounds = [None]
        allocation = owner.realize_physical("a2a3", list(range(6)))
        allocations = [allocation]
        mathematics = [owner]
    cases = []
    for bound, allocated, mathematical in zip(bounds, allocations, mathematics, strict=True):
        assert mathematical is not None, "Admitted invocation case has no retained mathematics"
        parsed = None
        if mathematical.describe()["hierarchical"]:
            try:
                parsed = testing.inspect_scoped_token_families(
                    token_function, parameter_bounds=parameters, block_bounds=bound
                )
            except ValueError as error:
                if "requires an original region tree" not in str(error):
                    raise
        row = case_inventory(original, mathematical, parsed, allocated)
        row["block_bounds"] = bound
        cases.append(row)
    audit_seconds = time.perf_counter() - start
    # The coarse pair count ignores domains. Encoded endpoint descriptors keep
    # their coordinate axes and volume, but omit activation-case predicates and
    # their paired endpoint. Neither census is permission to fuse commands.
    endpoint_pairs = set()
    exact_endpoints, colored_endpoints = set(), set()
    all_families = []
    for case in cases:
        for f in case["families"]:
            endpoint_pairs.add((f["source"][0], f["target"][0], *f["direction"]))
            all_families.append(f)
            for side, stride in (("source", 2), ("target", 3)):
                key = json.dumps(
                    [side, f[side], f["direction"], f["count"], [[a[0], a[1], a[stride]] for a in f["axes"]]],
                    sort_keys=True,
                )
                exact_endpoints.add(key)
                colored_endpoints.add((key, tuple(f["ids"])))
    checks = []
    # Bounded independent checks only. Large executions retain compact audits;
    # they are not unfolded to construct or time a production representation.
    if valuations and (independent or name.startswith("lib_") or name.endswith("_h512")):
        selected_values = valuations if len(valuations) <= 2 else [valuations[0], valuations[-1]]
        for parameters_at, block in selected_values:
            state, _, order = run_physical(original, selected, range(6), parameters_at, block_index=block)
            reference = state.required_reference
            metric = compare_order([c[2] for c in state.original_calls], reference[2], reference[2], order)
            cross_covers = sum(
                state.original_calls[a][2] != state.original_calls[b][2] for a, b in reference[3]
            )
            assert cross_covers == len(state.records) == len(state.waits)
            trace = trace_control(selected, parameters_at, block)
            case = next(
                c
                for c in cases
                if c["block_bounds"] is None or c["block_bounds"][0] <= block <= c["block_bounds"][1]
            )
            assert case["dynamic_handoffs"] == cross_covers
            checks.append(
                {
                    "parameters": parameters_at,
                    "block": block,
                    "payloads": len(state.original_calls),
                    "nonnative_covers": len(reference[3]),
                    "cross_pipe_covers": cross_covers,
                    "executed_set_wait_pairs": len(state.records),
                    "excess_prerequisites": metric["after"]["excess_completion_start"],
                    "ordering": metric["relation"],
                    "executed_control": dict(trace.counts),
                }
            )
    return {
        "name": name,
        "input_sha256": hashlib.sha256(ir.serialize(program)).hexdigest(),
        "production_seconds_single_run": production_seconds,
        "standalone_stage_seconds_single_run": stages,
        "reconstruction_audit_seconds": audit_seconds,
        "logical_inventory": inventories,
        "original_static": dict(static_control(original)),
        "physical_static": dict(static_control(selected)),
        "cases": cases,
        "family_instances_across_cases": len(all_families),
        "distinct_original_endpoint_pairs_ignoring_domains": len(endpoint_pairs),
        "distinct_encoded_endpoint_descriptors": len(exact_endpoints),
        "distinct_colored_endpoint_descriptors": len(colored_endpoints),
        "guard_dependencies": guard_accounting(original, selected, valuations),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend.set_backend_type(backend.BackendType.Ascend910B)
    rows = []
    for name, program, valuations, independent in fixtures():
        try:
            row = measure(name, program, valuations, independent)
            row["status"] = "checked"
        except (ValueError, AssertionError) as error:
            row = {
                "name": name,
                "status": "unavailable-or-failed",
                "reason": str(error),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        print(name, row["status"], row.get("reason", ""), flush=True)
        (args.output / "accounting.json").write_text(
            json.dumps(
                {
                    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "extension_sha256": hashlib.sha256(Path(pypto_core.__file__).read_bytes()).hexdigest(),
                    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "command": sys.argv,
                    "cases": rows,
                    "scope": "Host audit; production unchanged; cross-case equality is not a merge proof.",
                },
                indent=2,
            )
            + "\n"
        )
    assert all(row["status"] == "checked" for row in rows), "Accounting failure; see receipt"


if __name__ == "__main__":
    main()
