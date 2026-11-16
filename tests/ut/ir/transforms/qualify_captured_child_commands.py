# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent finite validation of an original captured child and emitted commands."""

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from pypto import backend, codegen, ir, passes, testing

from .sync_closed_command_oracle import ClosedCommands
from .sync_matmul_oracle import (
    check_imported_accesses,
    protected_accumulator_pairs,
    translate_access_description,
    validate_boundaries,
)
from .test_eventful_sync_children import eventful_fixture
from .test_finite_sync_analysis import closure


class EventOrder:
    """Exact DAG reachability; no compiler relations or injected reuse edges."""

    def __init__(self, vertices, edges):
        nodes = sorted(vertices, key=repr)
        self.index = {node: index for index, node in enumerate(nodes)}
        successors = [[] for _ in nodes]
        incoming = [0] * len(nodes)
        for source, target in edges:
            a, b = self.index[source], self.index[target]
            successors[a].append(b)
            incoming[b] += 1
        ready = deque(index for index, count in enumerate(incoming) if not count)
        ordered = []
        while ready:
            node = ready.popleft()
            ordered.append(node)
            for target in successors[node]:
                incoming[target] -= 1
                if not incoming[target]:
                    ready.append(target)
        assert len(ordered) == len(nodes), "cyclic actual command graph"
        self.rows = [0] * len(nodes)
        for node in reversed(ordered):
            for target in successors[node]:
                self.rows[node] |= (1 << target) | self.rows[target]

    def __contains__(self, pair):
        source, target = pair
        return bool(self.rows[self.index[source]] & (1 << self.index[target]))


class CommandPrefix:
    """Persistent conjunction of the same command completions as the old list.

    A consumer receives one aggregate predecessor instead of all history edges.
    Appending an acquire adds no history predecessor to that acquire itself.
    Thus original event reachability is unchanged, including flag reuse paths.
    """

    def __init__(self, owner):
        self.owner = owner
        self.tail: tuple[str, int] | None = None

    def append(self, event):
        aggregate = ("command_prefix", self.owner.prefix_nodes)
        self.owner.prefix_nodes += 1
        self.owner.vertices.add(aggregate)
        self.owner.edge(self.tail, aggregate)
        self.owner.edge(event, aggregate)
        self.tail = aggregate

    def __iter__(self):
        return iter(()) if self.tail is None else iter((self.tail,))


class BoundedCommands(ClosedCommands):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_nodes = 0
        self.commands = defaultdict(lambda: CommandPrefix(self))
        self.byte_pool = {}

    def bytes(self, operand, rect=None):
        points = super().bytes(operand, rect)
        canonical = self.byte_pool.setdefault(points, points)
        _, _, height, width, _ = self.storage[operand]
        key = self.signature(operand), rect or (0, 0, height, width)
        self.footprints[key] = canonical
        return canonical

    def payload(self, call, env):
        super().payload(call, env)
        signature, pipe, reads, writes = self.payloads[-1]
        # Full immutable-set equality, including every physical byte, owns this
        # interning. Repeated accumulator unions need not allocate per visit.
        reads = self.byte_pool.setdefault(reads, reads)
        writes = self.byte_pool.setdefault(writes, writes)
        self.payloads[-1] = signature, pipe, reads, writes
        original_call, original_env, original_pipe, _, _ = self.calls[-1]
        self.calls[-1] = original_call, original_env, original_pipe, reads, writes

    def build_order(self):
        started = time.perf_counter()
        self.execute(self.function.body, dict(self.env))
        print(
            json.dumps(
                {
                    "audit_stage": "executed_ir",
                    "payloads": len(self.payloads),
                    "vertices": len(self.vertices),
                    "edges": len(self.edges),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        assert not self.live, "unconsumed notification at exit"
        started = time.perf_counter()
        self.reachability = EventOrder(self.vertices, self.edges)
        print(
            json.dumps({"audit_stage": "command_reachability", "seconds": time.perf_counter() - started}),
            flush=True,
        )
        assert all(pair in self.reachability for pair in self.reuses), (
            "consumption does not precede republication"
        )
        return self.reachability


def crosscheck_graph_storage():
    function, _ = eventful_fixture(1)
    state = ClosedCommands(function, {3: 1, 4: True}, range(6))
    state.execute(function.body, dict(state.env))
    old = closure(state.vertices, state.edges)
    new = EventOrder(state.vertices, state.edges)
    assert all(((a, b) in old) == ((a, b) in new) for a in state.vertices for b in state.vertices)


def logical_plan(function, *, children=None, automatic=False, disjoint_gm=False):
    """Select the producer explicitly: empty public requests now select children."""
    assert not (automatic and children is not None)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=disjoint_gm):
        program = ir.Program([function], "ActualChild", function.span)
        if automatic:
            return passes.insert_sync()(program)
        if children is not None:
            return passes.insert_sync(eventful_children=children)(program)
        primitive = testing.insert_conservative_prefixes(function)
        return ir.Program([primitive], "PrimitivePrefix", function.span)


def crosscheck_command_prefixes():
    """Compare every original event pair, not only payloads or reuse counts."""
    function, scope = eventful_fixture(1)
    with passes.PassContext([]):
        child = passes.prepare_eventful_sync_children(function, [scope])[0]
    for children in [None, [child]]:
        logical = logical_plan(function, children=children)
        with passes.PassContext([]):
            physical = passes.lower_sync()(logical)
        for program in [logical, physical]:
            candidate = next(iter(program.functions.values()))
            old = ClosedCommands(candidate, {3: 2, 4: True}, range(6))
            original_order = old.run()
            new = BoundedCommands(candidate, {3: 2, 4: True}, range(6))
            new.build_order()
            assert old.vertices <= new.vertices
            assert all(
                ((source, target) in original_order) == ((source, target) in new.reachability)
                for source in old.vertices
                for target in old.vertices
            ), "prefix compression changed original command-event reachability"
            assert old.reuses == new.reuses
            assert old.trace == new.trace
            del old, new, original_order
            gc.collect()


def exact_native_pairs(function, calls, overlap):
    """Same rule/excluded branches as the unchanged oracle; indexed intermediates.

    For a fixed accumulator footprint F, bad(i) is exactly the old intermediate
    predicate. A pair (a,b) has no bad intermediate iff prefix[b]-prefix[a+1]==0.
    Binary eligibility is obtained from that original oracle on two-entry traces,
    so this wrapper adds no native qualification or footprint assumption.
    """
    eligible_cache = {}

    def eligible(source, target):
        key = (source[0], target[0], source[4], target[4])
        if key not in eligible_cache:
            eligible_cache[key] = (0, 1) in protected_accumulator_pairs(function, [source, target])
        return eligible_cache[key]

    anchors = {}
    for entry in calls:
        if entry[2] == ir.PipeType.M and eligible(entry, entry):
            anchors.setdefault(entry[4], entry)
    prefixes = {}
    for footprint, anchor in anchors.items():
        prefix = [0]
        for entry in calls:
            if entry[2] == ir.PipeType.M:
                bad = not eligible(entry, anchor)
            else:
                bad = overlap(entry[3], footprint) or overlap(entry[4], footprint)
            prefix.append(prefix[-1] + int(bad))
        prefixes[footprint] = prefix
    protected = set()
    for source in range(len(calls)):
        if calls[source][2] != ir.PipeType.M:
            continue
        for target in range(source + 1, len(calls)):
            if calls[target][2] != ir.PipeType.M or not eligible(calls[source], calls[target]):
                continue
            prefix = prefixes[calls[target][4]]
            if prefix[target] == prefix[source + 1]:
                protected.add((source, target))
    return protected, {"footprints": len(anchors), "binary_qualifications": len(eligible_cache)}


def crosscheck_native_reference():
    function, _ = eventful_fixture(2)
    state = ClosedCommands(function, {3: 2, 4: True}, range(6))
    state.execute(function.body, dict(state.env))
    expected = protected_accumulator_pairs(function, state.calls)
    actual, _ = exact_native_pairs(function, state.calls, lambda a, b: not a.isdisjoint(b))
    assert expected == actual, "indexed native reference changed a hardware exemption"


def scope_named(statement, name):
    pending = [statement]
    found = []
    while pending:
        current = pending.pop()
        if isinstance(current, ir.SeqStmts):
            pending.extend(current.stmts)
        elif isinstance(current, ir.ForStmt):
            if current.loop_var.name_hint == name:
                found.append(current)
            pending.append(current.body)
        elif isinstance(current, ir.IfStmt):
            pending.append(current.then_body)
            if current.else_body is not None:
                pending.append(current.else_body)
    assert len(found) == 1
    return found[0]


def payload_rows(state):
    events = [("payload", index, kind) for index in range(len(state.payloads)) for kind in ["I", "C"]]
    return [
        sum(1 << index for index, target in enumerate(events) if (source, target) in state.reachability)
        for source in events
    ]


def completion_rows(order, count):
    """Exact completion-to-start projection, including every target ordinal."""
    return [
        sum(
            1 << target
            for target in range(count)
            if (("payload", source, "C"), ("payload", target, "I")) in order
        )
        for source in range(count)
    ]


def required_reference(original, source):
    # Bounds only reject disjoint spaces/intervals; every possible overlap still
    # uses the original exact byte sets, independently of compiler partitions.
    bounds_cache = {}

    def bounds(points):
        key = id(points)
        if key not in bounds_cache:
            groups = {}
            for space, address in points:
                if space not in groups:
                    groups[space] = [address, address]
                else:
                    groups[space][0] = min(groups[space][0], address)
                    groups[space][1] = max(groups[space][1], address)
            bounds_cache[key] = groups
        return bounds_cache[key]

    def overlap(a, b):
        left, right = bounds(a), bounds(b)
        if not any(
            space in right and lo <= right[space][1] and right[space][0] <= hi
            for space, (lo, hi) in left.items()
        ):
            return False
        return not a.isdisjoint(b)

    print(json.dumps({"audit_stage": "native_accumulator_reference"}), flush=True)
    native_started = time.perf_counter()
    protected, native_work = exact_native_pairs(original, source.calls, overlap)
    print(
        json.dumps(
            {
                "audit_stage": "native_accumulator_reference_complete",
                "seconds": time.perf_counter() - native_started,
                **native_work,
            }
        ),
        flush=True,
    )
    required_edges = set(source.edges)
    hazards = 0
    for later, (_, _, reads, writes) in enumerate(source.payloads):
        for earlier in range(later):
            _, _, old_reads, old_writes = source.payloads[earlier]
            if (overlap(old_writes, reads) or overlap(old_writes, writes) or overlap(old_reads, writes)) and (
                earlier,
                later,
            ) not in protected:
                required_edges.add((("payload", earlier, "C"), ("payload", later, "I")))
                hazards += 1
    required = EventOrder(source.vertices, required_edges)
    return completion_rows(required, len(source.payloads)), hazards


def reference_storage(source, description, calls):
    """Map claimed domains to unique original physical descriptors and byte units."""
    element_bytes = {}
    storage = {}
    for variable, (space, base, rows, columns, width) in source.storage.items():
        layout = source.layouts.get(variable)
        storage[variable] = storage[variable.name_hint] = (space, base, rows, columns, layout, width)
        if not isinstance(space, str) and space != ir.MemorySpace.Vec:
            assert element_bytes.setdefault(space, width) == width, (
                "mixed element widths need byte-model adapter"
            )
            assert base % width == 0, "unaligned matrix element model is unavailable"
    available_names = set()
    for domain in description["storage"]:
        if not domain["boundary_available"]:
            continue
        assert domain["name"] not in available_names, "ambiguous retained storage-domain name"
        available_names.add(domain["name"])
        candidates = [
            descriptor
            for variable, descriptor in storage.items()
            if not isinstance(variable, str)
            and variable.name_hint == domain["name"]
            and descriptor[0] == domain["space"]
            and descriptor[1] == domain["address"]
            and descriptor[2:4] == (domain["rows"], domain["columns"])
        ]
        assert len(candidates) == 1, (
            "ambiguous original allocation geometry",
            domain,
            [
                (variable.name_hint, value[:4])
                for variable, value in storage.items()
                if not isinstance(variable, str) and variable.name_hint == domain["name"]
            ],
        )
        storage[domain["name"]] = candidates[0]
    converted = {}

    def elements(points):
        key = id(points)
        if key not in converted:
            # Existing matrix selector diagnostics use element starts; uniform
            # aligned per-space widths make this an exact map of the same bytes.
            # Vec remains byte-addressed. Original full byte sets remain the
            # independent hazard reference below, including external reads.
            converted[key] = frozenset(
                (space, address)
                for space, address in points
                if isinstance(space, str)
                or space == ir.MemorySpace.Vec
                or address % element_bytes[space] == 0
            )
        return converted[key]

    translated = [
        (call, env, pipe, elements(reads), elements(writes)) for call, env, pipe, reads, writes in calls
    ]
    return storage, translated


def check_retained_reference(original, source, retained, scope):
    """Check the actual retained child, not a reanalyzed specialized program."""
    description = retained.describe()
    modeled = {
        site[key]
        for region in description["regions"]
        for site in region["sites"]
        for key in ["original", "continuation"]
        if site[key] is not None
    }

    class OriginalCalls(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls = set()

        def visit_call(self, op):
            self.calls.add(op)
            super().visit_call(op)

    original_calls = OriginalCalls()
    original_calls.visit_stmt(scope)
    calls = [entry for entry in source.calls if entry[0] in original_calls.calls]
    assert [entry for entry in source.calls if entry[0] in modeled] == calls, (
        "retained inventory must cover every executed original child payload"
    )
    storage, translated = reference_storage(source, description, calls)
    translate_access_description(description, {}, None)
    print(json.dumps({"audit_stage": "retained_accesses", "payloads": len(calls)}), flush=True)
    occurrences, effects = check_imported_accesses(description, storage, translated, {}, True)
    print(json.dumps({"audit_stage": "retained_covers"}), flush=True)
    assert len(set(occurrences)) == len(occurrences), "reference requires one complete child invocation"
    vertices = [("payload", index, kind) for index in range(len(calls)) for kind in ["I", "C"]]
    edges = {(("payload", index, "I"), ("payload", index, "C")) for index in range(len(calls))}
    previous = {}
    for index, (_, _, pipe, _, _) in enumerate(calls):
        if pipe in previous:
            for kind in ["I", "C"]:
                edges.add((("payload", previous[pipe], kind), ("payload", index, kind)))
        previous[pipe] = index
    protected, _ = exact_native_pairs(original, calls, lambda a, b: not a.isdisjoint(b))
    candidates = []
    for later, (_, _, _, reads, writes) in enumerate(calls):
        for earlier, (_, _, _, old_reads, old_writes) in enumerate(calls[:later]):
            if (
                not old_writes.isdisjoint(reads)
                or not old_writes.isdisjoint(writes)
                or not old_reads.isdisjoint(writes)
            ) and (earlier, later) not in protected:
                edge = (("payload", earlier, "C"), ("payload", later, "I"))
                edges.add(edge)
                candidates.append((earlier, later))
    order = EventOrder(vertices, edges)
    predecessors = {vertex: 0 for vertex in vertices}
    for a in vertices:
        for b in vertices:
            if (a, b) in order:
                predecessors[b] |= 1 << order.index[a]
    covers = set()
    for a, b in candidates:
        start, end = ("payload", a, "C"), ("payload", b, "I")
        if order.rows[order.index[start]] & predecessors[end] == 0:
            covers.add((occurrences[a], occurrences[b]))
    assert description["hierarchical"], "captured reference expects the retained regional interface"
    selected = set(retained.expand_retained_demands(len(calls)))
    assert selected == covers, ("actual retained covers", selected - covers, covers - selected)
    print(json.dumps({"audit_stage": "retained_queries", "covers": len(covers)}), flush=True)
    query_checks = 0
    for i, a in enumerate(occurrences):
        for j, b in enumerate(occurrences):
            for left, source_kind in [(False, "I"), (True, "C")]:
                for right, target_kind in [(False, "I"), (True, "C")]:
                    expected = (("payload", i, source_kind), ("payload", j, target_kind)) in order
                    assert retained.precedes(a, left, b, right) == expected, (a, left, b, right)
                    query_checks += 1
    unavailable = [i for i, domain in enumerate(description["storage"]) if not domain["boundary_available"]]
    print(json.dumps({"audit_stage": "retained_boundaries"}), flush=True)
    boundaries = validate_boundaries(
        description, storage, effects, occurrences, retained, unavailable_boundaries=unavailable
    )
    return {
        "access_checks": len(calls),
        "actual_covers": len(covers),
        "event_queries": query_checks,
        "boundary_samples": boundaries,
        "unavailable_gm_domains": len(unavailable),
    }


def execution_case(original, plans, parameters, shape_values, bases, block_index, retained=None, scope=None):
    """Keep the source and one command arm live; return compact exact projections.

    The source byte sets and native graph stay available for independent hazard
    reconstruction. An arm's matching/reuse checks run before it is released.
    Full I/C projections preserve the previous equality test without retaining
    several million-edge logical and physical graphs at once. No state or
    footprint cache escapes this invocation to the next block-index case.
    """
    cache = {}
    byte_pool = {}

    def interpret(function, arm):
        print(
            json.dumps({"audit_stage": "interpret_arm", "arm": arm, "block_index": block_index}), flush=True
        )
        metadata = {}
        for symbol in codegen.collect_tensor_shape_dyn_vars(function):
            assert symbol.name_hint in shape_values, "missing explicitly supplied physical ABI shape"
            metadata[symbol] = shape_values[symbol.name_hint]
        state = BoundedCommands(
            function, parameters, range(6), bases, shape_parameters=metadata, block_index=block_index
        )
        state.footprints = cache
        state.byte_pool = byte_pool
        state.build_order()
        return state

    source = interpret(original, 0)
    mathematical_checks = (
        check_retained_reference(original, source, retained, scope) if retained is not None else None
    )
    required, hazards = required_reference(original, source)
    projections = []
    physical_summaries = []
    arm = 1
    token_checks = []
    for logical, physical in plans:
        expressed = None
        if logical is not None:
            state = interpret(logical, arm)
            assert state.payloads == source.payloads, "changed original payload/bytes"
            assert state.descriptors == source.descriptors, "changed storage descriptors"
            expressed = payload_rows(state)
            del state
            gc.collect()
        token_checks.append(expressed is not None)
        state = interpret(physical, arm + 1)
        assert state.payloads == source.payloads, "changed original payload/bytes"
        assert state.descriptors == source.descriptors, "changed storage descriptors"
        if expressed is not None:
            assert expressed == payload_rows(state), "changed actual logical payload event order"
        selected = completion_rows(state.reachability, len(source.payloads))
        assert all(needed & ~actual == 0 for needed, actual in zip(required, selected, strict=True)), (
            "uncovered physical-byte prerequisite"
        )
        projections.append(selected)
        physical_summaries.append(
            {
                "commands": [len(state.publications), len(state.acquisitions), state.barriers],
                "directional_ids": {str(key): sorted(value) for key, value in state.ids.items()},
                "checked_reuses": len(state.reuses),
            }
        )
        del state, expressed
        gc.collect()
        arm += 2
    old_pairs, new_pairs = projections

    def count(rows):
        return sum(row.bit_count() for row in rows)

    def difference(left, right):
        return count(a & ~b for a, b in zip(left, right, strict=True))

    return {
        "retained_mathematical_checks": mathematical_checks,
        "block_index": block_index,
        "payloads": len(source.payloads),
        "hazards": hazards,
        "required_pairs": count(required),
        "old_excess": difference(old_pairs, required),
        "new_excess": difference(new_pairs, required),
        "removed_pairs": difference(old_pairs, new_pairs),
        "added_pairs": difference(new_pairs, old_pairs),
        "old_commands": physical_summaries[0]["commands"],
        "new_commands": physical_summaries[1]["commands"],
        "new_directional_ids": physical_summaries[1]["directional_ids"],
        "checked_reuses": physical_summaries[1]["checked_reuses"],
        "actual_token_all_event_order_preserved": True if all(token_checks) else None,
        "actual_token_order_checked_by_arm": token_checks,
    }


def start_memory_guard(output, limit_mib):
    """Refuse on observed excess RSS; polling is a best-effort Linux safeguard."""
    assert limit_mib > 0
    status = Path("/proc/self/status")
    assert status.is_file(), "The bounded audit memory guard currently requires Linux /proc"

    def watch():
        while True:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                    if rss_kib > limit_mib * 1024:
                        refusal = {"status": "audit_memory_limit", "rss_kib": rss_kib, "limit_mib": limit_mib}
                        (output / "memory-refusal.json").write_text(json.dumps(refusal) + "\n")
                        print(json.dumps(refusal), flush=True)
                        os._exit(125)
                    break
            time.sleep(0.1)

    threading.Thread(target=watch, daemon=True, name="audit-rss-limit").start()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--scope", help="Original loop name for explicit-child mode")
    parser.add_argument("--automatic", action="store_true", help="Audit the public automatic child producer")
    parser.add_argument(
        "--disjoint-gm",
        action="store_true",
        help="Supply the explicit disjoint-GM contract to both producers",
    )
    parser.add_argument("--parameters", required=True, help="JSON original scalar parameter index -> value")
    parser.add_argument(
        "--shape-values", required=True, help="JSON original physical ABI shape name -> value"
    )
    parser.add_argument("--block-index", action="append", type=int, required=True)
    parser.add_argument("--baseline-physical", type=Path, help="Previously emitted owned physical Program")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-mib", type=int, default=2048)
    parser.add_argument(
        "--check-retained",
        action="store_true",
        help="Also audit actual retained covers/all queries/selectors; expensive, use one execution",
    )
    args = parser.parse_args(argv)
    if args.automatic and (args.scope is not None or args.check_retained):
        parser.error("--automatic cannot be combined with --scope or --check-retained")
    if not args.automatic and args.scope is None:
        parser.error("explicit-child mode requires --scope")
    if args.output.exists():
        parser.error("output must be a new directory; preserve previous audit evidence")
    return args


def source_identity() -> dict[str, Any]:
    """Pin the compiler actually loaded, including dirty source and Python."""
    root = Path(__file__).resolve().parents[4]
    assert __spec__ is not None, "Run the qualification module with python -m"

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    changed = git("diff", "--name-only", "HEAD").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    sources = sorted(set(changed + untracked))
    binaries = {}
    for name, module in sys.modules.copy().items():
        filename = getattr(module, "__file__", None)
        if name.startswith("pypto") and isinstance(filename, str) and filename.endswith(".so"):
            path = Path(filename).resolve()
            binaries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_revision": git("rev-parse", "HEAD"),
        "dirty_source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources
            if name.startswith(("src/", "include/", "python/", "tests/")) and (root / name).is_file()
        },
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "python_version": sys.version,
        "binaries": binaries,
        "command": [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
        "native_tool_execution": "No PTOAS, CCEC or device execution in this actual-command audit",
    }


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True)
    start_memory_guard(args.output, args.memory_limit_mib)
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    print(json.dumps({"audit_stage": "small_graph_crosscheck"}), flush=True)
    crosscheck_graph_storage()
    crosscheck_command_prefixes()
    crosscheck_native_reference()
    original_program = ir.deserialize_from_file(str(args.input))
    assert isinstance(original_program, ir.Program)
    original = original_program.get_function(args.function)
    assert original is not None
    scope = scope_named(original.body, args.scope) if args.scope is not None else None
    parameters = {int(key): value for key, value in json.loads(args.parameters).items()}
    shape_values = json.loads(args.shape_values)
    started = time.perf_counter()
    child = None
    retained = None
    if not args.automatic:
        assert scope is not None
        print(json.dumps({"audit_stage": "prepare_retained_child"}), flush=True)
        with passes.PassContext([]):
            child = passes.prepare_eventful_sync_children(original, [scope])[0]
        retained = testing.inspect_eventful_sync_child(child)
    math_reference = retained if args.check_retained else None
    logical_programs = [
        logical_plan(original, disjoint_gm=args.disjoint_gm),
        logical_plan(
            original,
            children=None if args.automatic else [child],
            automatic=args.automatic,
            disjoint_gm=args.disjoint_gm,
        ),
    ]
    plans: list[tuple[ir.Function | None, ir.Function]] = []
    for logical in logical_programs:
        with passes.PassContext([]):
            physical = passes.lower_sync()(logical)
        plans.append((next(iter(logical.functions.values())), next(iter(physical.functions.values()))))
    baseline_identity = None
    if args.baseline_physical is not None:
        baseline_program = ir.deserialize_from_file(str(args.baseline_physical))
        assert isinstance(baseline_program, ir.Program)
        baseline = baseline_program.get_function(args.function)
        assert baseline is not None and dict(baseline.attrs).get("pypto_autosync") is True
        # The old actual token program is unavailable. Do not substitute its
        # physical commands as a fake logical arm or claim token-order equality.
        # Both emitted plans still undergo independent byte/order/reuse checks.
        plans[0] = (None, baseline)
        baseline_identity = {
            "path": str(args.baseline_physical.resolve()),
            "sha256": hashlib.sha256(args.baseline_physical.read_bytes()).hexdigest(),
            "logical_program": "unavailable",
        }
    candidate_logical = plans[1][0]
    assert candidate_logical is not None
    candidate_attributes = dict(candidate_logical.attrs)
    results = []
    bases = {
        index: index << 40
        for index, parameter in enumerate(original.params)
        if isinstance(parameter.type, ir.TensorType)
    }
    for block_index in args.block_index:
        results.append(
            execution_case(
                original, plans, parameters, shape_values, bases, block_index, math_reference, scope
            )
        )
        (args.output / "completed-executions.json").write_text(json.dumps(results, indent=2) + "\n")
        gc.collect()
    receipt: dict[str, Any] = {
        **source_identity(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input": str(args.input.resolve()),
        "function": args.function,
        "baseline_producer": "captured_physical_plan"
        if baseline_identity
        else "primitive_conservative_prefix",
        "baseline_identity": baseline_identity,
        "candidate_producer": "public_automatic_children" if args.automatic else "explicit_retained_child",
        "alias_contract": "explicit_disjoint_gm" if args.disjoint_gm else "may_alias",
        "tensor_bases": bases,
        "backend": "Ascend910/a2a3",
        "candidate_child_diagnostics": {
            name: value
            for name, value in candidate_attributes.items()
            if name.startswith("autosync_child_") or name == "autosync_retained_child_count"
        },
        "scope": args.scope,
        "parameters": parameters,
        "shape_values": shape_values,
        "caller_qualification": (
            "Finite explicitly supplied ABI executions; these values are not compiler assumptions."
        ),
        "retained": child.describe() if child is not None else None,
        "executions": results,
        "seconds_inclusive": time.perf_counter() - started,
        "device": "not run",
        "memory_limit_mib": args.memory_limit_mib,
        "observed_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "graph_lifetime": "One logical or physical arm at a time; no graph/cache retained across executions",
        "retained_mathematics": retained.describe()["guarantee"] if retained is not None else None,
        "retained_mathematics_checked": args.check_retained,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
