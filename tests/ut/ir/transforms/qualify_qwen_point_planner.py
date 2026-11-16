# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded independent checks of the captured Qwen scalar work-list planner.

Use actual emitted commands and original byte effects. Reachability comes only
from the existing test interpreter's command graph, with its independently
crosschecked indexed graph/prefix representation. Large transitive pair sets,
compiler demand graphs, and compiler effect summaries are not used.
"""

import argparse
import gc
import hashlib
import importlib
import json
import resource
import sys
import time
from pathlib import Path

from pypto import backend, ir, passes

from .qualify_captured_child_commands import (
    BoundedCommands,
    EventOrder,
    crosscheck_command_prefixes,
    crosscheck_graph_storage,
    start_memory_guard,
)
from .sync_closed_command_oracle import ClosedCommands, opname
from .test_gm_scalar_write_staging import lower, memory, write_program


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crosscheck_point_commands():
    """Expanded closure is used only for this small actual point-transfer case."""
    original = write_program(1, recurrence=True)
    program = lower(original, passes.MemoryPlanner.PYPTO)
    function = program.get_function("kernel")
    assert function is not None and function.attrs.get("pypto_autosync") is True
    arguments = {2: 0, 3: 3, 4: 1, 5: 700}
    old = ClosedCommands(function, arguments, range(6), scalar_memory=memory(1))
    expected = old.run()
    new = BoundedCommands(function, arguments, range(6), scalar_memory=memory(1))
    actual = new.build_order()
    assert old.vertices <= new.vertices
    assert all(
        ((source, target) in expected) == ((source, target) in actual)
        for source in old.vertices
        for target in old.vertices
    ), "point-transfer prefix/indexed graph changed actual-event reachability"
    assert old.trace == new.trace and old.reuses == new.reuses
    assert old.payloads == new.payloads and old.scalar_memory == new.scalar_memory
    assert old.scalar_memory is not None
    assert {old.scalar_memory["gm", byte] for byte in range(4)} == {102}


def source_contract(library, capture):
    model = library / "models/qwen3_14b"
    sys.path[:0] = [str(model), str(library)]
    module = importlib.import_module("prefill_fwd")
    names = ("QKPV_PLAN_SB_BATCH", "QKPV_PLAN_WORK_ITEMS", "ATTN_GI_BLOCKS", "ATTN_TOK_GROUP")
    constants = {name: getattr(module, name) for name in names}
    assert tuple(constants.values()) == (1, 128, 8, 8), "different source plan needs a new reference domain"
    hashes = {name: digest(model / name) for name in capture["model_sources"]}
    assert hashes == capture["model_sources"], "model source differs from the captured compiler input"
    return constants, hashes


def expected_work(constants, sb_chunk, context_blocks, finalized):
    """Independent original active/inactive partition; no generated predicates."""
    count = constants["QKPV_PLAN_WORK_ITEMS"]
    lanes = constants["ATTN_GI_BLOCKS"]
    token_group = constants["ATTN_TOK_GROUP"]
    active, inactive = [], []
    for work in range(count):
        present = sb_chunk < context_blocks and work // lanes * token_group < finalized
        (active if present else inactive).append(work)
    return active + inactive


def initial_memory():
    # Original wcur and order arrays are separated by a concrete permitted ABI
    # layout. Sentinels outside both declared arrays must remain unchanged.
    cells = {("gm", byte): -1001 for byte in range(-4, 8)}
    cells.update({("gm", 4096 + 4 * i + byte): -2002 for i in range(-1, 129) for byte in range(4)})
    return cells


def check_values(execution, expected):
    cells = execution.scalar_memory
    assert cells is not None
    assert {cells["gm", byte] for byte in range(4)} == {len(expected)}
    for index, work in enumerate(expected):
        assert {cells["gm", 4096 + 4 * index + byte] for byte in range(4)} == {work}
    assert {cells["gm", byte] for byte in range(-4, 0)} == {-1001}
    assert {cells["gm", byte] for byte in range(4, 8)} == {-1001}
    for index in (-1, 128):
        assert {cells["gm", 4096 + 4 * index + byte] for byte in range(4)} == {-2002}


def check_abi(function):
    assert len(function.params) == 5
    for index, length in ((0, 1), (4, 128)):
        kind = function.params[index].type
        assert isinstance(kind, ir.TensorType) and kind.dtype == ir.DataType.INT32
        assert len(kind.shape) == 1 and isinstance(kind.shape[0], ir.ConstInt)
        assert kind.shape[0].value == length
    for index, dtype in ((1, ir.DataType.INDEX), (2, ir.DataType.INDEX), (3, ir.DataType.INT32)):
        kind = function.params[index].type
        assert isinstance(kind, ir.ScalarType) and kind.dtype == dtype


def build_reference(before, parameters, bases, expected, invocations):
    original = ClosedCommands(before, parameters, range(6), bases, scalar_memory=initial_memory())
    baseline_payloads = baseline_descriptors = None
    previous_return = None
    for _ in range(invocations):
        first_payload, first_descriptor = len(original.payloads), len(original.descriptors)
        original.execute(before.body, dict(original.env))
        returned = original.returns[-1]
        for index in range(first_payload, len(original.payloads)):
            # These are required invocation-boundary edges in the REFERENCE,
            # not assumptions inserted into the actual physical command graph.
            original.edge(("payload", index, "C"), returned)
            original.edge(previous_return, ("payload", index, "I"))
        previous_return = returned
        check_values(original, expected)
        if baseline_payloads is None:
            baseline_payloads = original.payloads[first_payload:]
            baseline_descriptors = original.descriptors[first_descriptor:]
        else:
            assert original.payloads[first_payload:] == baseline_payloads
            assert original.descriptors[first_descriptor:] == baseline_descriptors
    return original, baseline_payloads, baseline_descriptors


def execute_case(before, physical, constants, finalized, invocations):
    started = time.perf_counter()
    parameters, bases = {1: 0, 2: 1, 3: finalized}, {0: 0, 4: 4096}
    expected = expected_work(constants, 0, 1, finalized)
    original, baseline_payloads, baseline_descriptors = build_reference(
        before, parameters, bases, expected, invocations
    )
    actual = BoundedCommands(physical, parameters, range(6), bases, scalar_memory=initial_memory())
    invocation_boundaries = []
    for _ in range(invocations):
        first_payload, first_descriptor = len(actual.payloads), len(actual.descriptors)
        actual.execute(physical.body, dict(actual.env))
        assert not actual.live, "notification not consumed at invocation exit"
        assert actual.last_all is not None and len(actual.returns) == len(invocation_boundaries) + 1
        assert actual.payloads[first_payload:] == baseline_payloads, "changed original payload bytes/values"
        assert actual.descriptors[first_descriptor:] == baseline_descriptors, (
            "changed original allocation windows"
        )
        check_values(actual, expected)
        invocation_boundaries.append(
            (first_payload, len(actual.payloads), actual.last_all, actual.returns[-1])
        )
    execution_seconds = time.perf_counter() - started
    started = time.perf_counter()
    order = EventOrder(actual.vertices, actual.edges)
    graph_seconds = time.perf_counter() - started
    assert all(pair in order for pair in actual.reuses), "consumption does not precede republication"
    assert all(ids <= set(range(6)) for ids in actual.ids.values()), "ineligible directional ID"
    assert len(actual.publications) == len(actual.acquisitions), "unmatched actual commands"
    assert original.payloads == actual.payloads, "changed payload identities across invocations"
    conflict_checks = 0
    for index, (_, _, reads, writes) in enumerate(actual.payloads):
        for earlier in range(index):
            _, _, old_reads, old_writes = actual.payloads[earlier]
            if old_writes & (reads | writes) or old_reads & writes:
                assert (("payload", earlier, "C"), ("payload", index, "I")) in order, (
                    "unordered original-byte conflict",
                    earlier,
                    index,
                )
                # Independently extracted original staged-byte conflicts own
                # the reference edge; compiler effects/demands are not used.
                original.edge(("payload", earlier, "C"), ("payload", index, "I"))
                conflict_checks += 1
    for first, stop, completion, returned in invocation_boundaries:
        assert (completion, returned) in order, "native ALL does not precede invocation return"
        for index in range(first, stop):
            assert (("payload", index, "C"), completion) in order, "pending payload at owned return"
    comparison_started = time.perf_counter()
    reference_order = EventOrder(original.vertices, original.edges)
    required_pairs = realized_pairs = missing_pairs = excess_pairs = 0
    for source in range(len(actual.payloads)):
        for target in range(len(actual.payloads)):
            pair = (("payload", source, "C"), ("payload", target, "I"))
            required, realized = pair in reference_order, pair in order
            required_pairs += required
            realized_pairs += realized
            missing_pairs += required and not realized
            excess_pairs += realized and not required
    assert missing_pairs == 0, "physical program omits staged-model reference order"
    comparison_seconds = time.perf_counter() - comparison_started
    point_loads = point_stores = scalar_reads = 0
    for call, _, _, reads, writes in actual.calls:
        name = call.op.name
        if name == opname("buffer.load"):
            assert len(reads) == 4 and len(writes) == 32, "point read/native carrier enclosure changed"
            point_loads += 1
        elif name == opname("buffer.store"):
            assert len(reads) == 32 and len(writes) == 4, "point store/native carrier enclosure changed"
            point_stores += 1
        elif name == opname("buffer.read"):
            scalar_reads += 1
    assert point_loads == scalar_reads == 128 * invocations
    assert point_stores == 257 * invocations
    result = {
        "finalize_tok": finalized,
        "domain": "admitted positive caller execution"
        if finalized
        else "zero structural stress; outside positive caller",
        "parameters": parameters,
        "invocations": invocations,
        "payloads": len(actual.payloads),
        "publications": len(actual.publications),
        "acquisitions": len(actual.acquisitions),
        "reuse_checks": len(actual.reuses),
        "byte_conflict_checks": conflict_checks,
        "staged_payload_order": {
            "reference_completion_to_start_pairs": required_pairs,
            "physical_completion_to_start_pairs": realized_pairs,
            "missing_pairs": missing_pairs,
            "excess_pairs": excess_pairs,
            "comparison_seconds": comparison_seconds,
            "reference_integer_bytes": sum((row.bit_length() + 7) // 8 for row in reference_order.rows),
            "domain": (
                "Every ordered staged payload C-to-I pair, including required invocation "
                "completion across replay; no pair set materialized"
            ),
        },
        "vertices": len(actual.vertices),
        "edges": len(actual.edges),
        "reachability_integer_bytes": sum((row.bit_length() + 7) // 8 for row in order.rows),
        "ids": {
            f"{source.name}->{destination.name}": sorted(ids)
            for (source, destination), ids in actual.ids.items()
        },
        "execution_seconds": execution_seconds,
        "reachability_seconds": graph_seconds,
        "seconds": time.perf_counter() - started + execution_seconds,
        "output_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory; preserve prior qualification evidence")
    args.output.mkdir(parents=True)
    start_memory_guard(args.output, args.memory_limit_mib)
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    crosscheck_graph_storage()
    crosscheck_command_prefixes()
    crosscheck_point_commands()
    capture = json.loads((args.capture / "receipt.json").read_text())
    constants, model_sources = source_contract(args.library, capture)
    before_program = ir.deserialize_from_file(str(args.capture / "before-sync.msgpack"))
    after_program = ir.deserialize_from_file(str(args.capture / "after-sync.msgpack"))
    assert isinstance(before_program, ir.Program) and isinstance(after_program, ir.Program)
    before = before_program.get_function("qkpv_plan")
    physical = after_program.get_function("qkpv_plan")
    assert before is not None and physical is not None and physical.attrs.get("pypto_autosync") is True
    check_abi(before)
    check_abi(physical)
    results = []
    for finalized in (1, 8, 9, 16, 17, 127, 128, 0):
        results.append(execute_case(before, physical, constants, finalized, 2 if finalized == 9 else 1))
        (args.output / "completed-cases.json").write_text(json.dumps(results, indent=2) + "\n")
        gc.collect()
    paths = [args.capture / name for name in ("receipt.json", "before-sync.msgpack", "after-sync.msgpack")]
    dependencies = (
        "sync_closed_command_oracle.py",
        "sync_logical_oracle.py",
        "sync_matmul_oracle.py",
        "test_finite_sync_analysis.py",
        "qualify_captured_child_commands.py",
        "test_gm_scalar_write_staging.py",
        "test_gm_scalar_staging.py",
        "test_eventful_sync_children.py",
        "buffer_test_utils.py",
    )
    paths += [Path(__file__), *(Path(__file__).with_name(name) for name in dependencies)]
    extension = getattr(sys.modules["pypto.pypto_core"], "__file__", None)
    assert isinstance(extension, str), "loaded compiler extension has no pinned path"
    paths += [Path(extension), Path(sys.executable)]
    receipt = {
        "status": "all bounded original planner executions passed",
        "input_hashes": {str(path): digest(path) for path in paths},
        "python": {"version": sys.version, "executable": sys.executable},
        "loaded_extension": extension,
        "model_sources": model_sources,
        "source_constants": constants,
        "cases": results,
        "observed_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_limit_mib": args.memory_limit_mib,
        "graph": (
            "Existing actual-command interpreter plus crosschecked indexed DAG reachability; "
            "no compiler relations or injected reuse edges"
        ),
        "value_limitation": (
            "With SB_BATCH1 active work is already a prefix; outputs are0..127. "
            "Cases exercise distinct branch/readiness paths, not a nonidentity permutation."
        ),
        "guarantee": (
            "Bounded modeled access/value preservation, byte-conflict coverage, matching, "
            "acyclicity, eligible IDs, causal reuse including replay, and terminal completion"
        ),
        "order_diagnostic": (
            "Reference is actual before-sync native-pipe/SSA/control order plus independently "
            "extracted staged-byte conflicts and explicit invocation completion. It models "
            "MTE2/S/MTE3 with a private32B UB enclosure; it is not original scalar-DCache "
            "order, an exact minimum-cover result, or a native theorem."
        ),
        "not_established": [
            "uniform symbolic caller theorem",
            "minimum demands",
            "minimum IDs",
            "native/device execution",
            "general model completion",
        ],
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
