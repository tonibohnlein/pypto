# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read and compare PTOAS InsertSync JSONL summaries by function.

One PTO file can contain several InCore functions. PTOAS writes one summary
object per function, so comparing JSONL rows by position silently attributes
one function's synchronization to another when traversal order changes. This
module keys every record by its explicit ``function`` field and rejects
duplicates or mismatched function sets.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_MAPPING_METRICS = ("pipe_pair_groups", "pipe_pair_event_ids")
_IGNORED_NUMERIC_FIELDS = {"schema_version"}
_FUNCTION_RE = re.compile(r"\bfunc\.func\s+(?:private\s+)?@([-A-Za-z0-9_.$]+)")
_SYNC_OP_RE = re.compile(r"\bpto\.(set_flag(?:_dyn|_d)?|wait_flag(?:_dyn|_d)?)\b")
_BARRIER_RE = re.compile(r"\bpto\.barrier\b")
_HIGH_LEVEL_SYNC_RE = re.compile(r"\bpto\.(record_event|wait_event)\b")
_PIPE_RE = re.compile(r"\bPIPE_[A-Z0-9_]+\b")
_EVENT_RE = re.compile(r"\bEVENT_ID[A-Z0-9_]+\b")
_SSA_RE = re.compile(r"%[-A-Za-z0-9_.$]+")
_LOOP_RE = re.compile(r"\bscf\.for\b")
_UNRESOLVED_STRUCTURED_CONTROL_RE = re.compile(r"\bscf\.(?:if|while|index_switch)\b")
_UNRESOLVED_CONTROL_CONTINUATION_RE = re.compile(r"^\s*(?:else|do|case\b|default\b)")
_UNSTRUCTURED_CONTROL_RE = re.compile(r"\bcf\.(?:br|cond_br|switch)\b")
_INTEGER_CONSTANT_RE = re.compile(
    r"^\s*(%[-A-Za-z0-9_.$]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(?:index|i\d+)\b"
)
_FOR_RE = re.compile(
    r"\bscf\.for\s+%\S+\s*=\s*(%[-A-Za-z0-9_.$]+|-?\d+)\s+to\s+"
    r"(%[-A-Za-z0-9_.$]+|-?\d+)\s+step\s+(%[-A-Za-z0-9_.$]+|-?\d+)\b"
)


def load_sync_summaries(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a PTOAS sync-summary JSONL file keyed by function name.

    Args:
        path: JSONL summary emitted by PTOAS.

    Returns:
        Summary objects keyed by their explicit function names.

    Raises:
        ValueError: A non-empty line is not a JSON object, has no valid
            ``function`` field, or repeats a function name.
    """
    source = Path(path)
    summaries: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        function = record.get("function")
        if not isinstance(function, str) or not function:
            raise ValueError(f"{source}:{line_number}: missing non-empty 'function'")
        if function in summaries:
            raise ValueError(f"{source}:{line_number}: duplicate function '{function}'")
        summaries[function] = record
    if not summaries:
        raise ValueError(f"{source}: no sync-summary records")
    return summaries


def _numeric_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, int | float]:
    deltas: dict[str, int | float] = {}
    for key in sorted(set(baseline) | set(candidate)):
        if key in _IGNORED_NUMERIC_FIELDS or key in _MAPPING_METRICS:
            continue
        before = baseline.get(key)
        after = candidate.get(key)
        if (
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
        ):
            deltas[key] = after - before
    return deltas


def _mapping_delta(baseline: Any, candidate: Any) -> dict[str, int | float]:
    before = baseline if isinstance(baseline, Mapping) else {}
    after = candidate if isinstance(candidate, Mapping) else {}
    delta: dict[str, int | float] = {}
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key, 0)
        after_value = after.get(key, 0)
        if (
            not isinstance(before_value, (int, float))
            or isinstance(before_value, bool)
            or not isinstance(after_value, (int, float))
            or isinstance(after_value, bool)
        ):
            raise ValueError(f"sync metric '{key}' is not numeric")
        delta[key] = after_value - before_value
    return delta


def diff_sync_summaries(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return candidate-minus-baseline deltas keyed by function.

    The two inputs must contain exactly the same function names. Numeric scalar
    metrics and the two pipe-pair maps emitted by PTOAS are compared
    independently for each function.

    Args:
        baseline: Baseline summary objects keyed by function.
        candidate: Candidate summary objects keyed by function.

    Returns:
        Numeric and pipe-pair deltas for each function.

    Raises:
        ValueError: Function sets differ or a pipe-pair metric is non-numeric.
    """
    baseline_functions = set(baseline)
    candidate_functions = set(candidate)
    if baseline_functions != candidate_functions:
        missing = sorted(baseline_functions - candidate_functions)
        added = sorted(candidate_functions - baseline_functions)
        raise ValueError(f"sync-summary function mismatch: missing={missing}, added={added}")

    result: dict[str, dict[str, Any]] = {}
    for function in sorted(baseline):
        before = baseline[function]
        after = candidate[function]
        function_delta: dict[str, Any] = {"metrics": _numeric_deltas(before, after)}
        for key in _MAPPING_METRICS:
            function_delta[key] = _mapping_delta(before.get(key), after.get(key))
        result[function] = function_delta
    return result


def _transition_kind(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    before_stack = tuple(before["loop_stack"])
    after_stack = tuple(after["loop_stack"])
    before_type = str(before["type"])
    after_type = str(after["type"])
    if before_stack == after_stack:
        if before_type.startswith("set_flag") and after_type.startswith("wait_flag"):
            return "within_iteration"
        return "same_scope_rearm"
    if before_stack == after_stack[: len(before_stack)]:
        return "loop_entry"
    if after_stack == before_stack[: len(after_stack)]:
        return "loop_exit"
    return "cross_loop_boundary"


def _static_execution_multiplier(
    operation: Mapping[str, Any],
    loop_trip_counts: Mapping[int, int | None],
) -> int | None:
    if operation["unresolved_control_flow_stack"]:
        return None
    multiplier = 1
    for loop_id in operation["loop_stack"]:
        trip_count = loop_trip_counts.get(loop_id)
        if trip_count is None:
            return None
        multiplier *= trip_count
    return multiplier


def _summarize_lowered_function(
    function: str,
    operations: list[dict[str, Any]],
    loop_trip_counts: Mapping[int, int | None],
    has_unstructured_control_flow: bool,
) -> dict[str, Any]:
    by_type = Counter(str(operation["type"]) for operation in operations)
    by_pipe_pair = Counter(
        f"{operation['src_pipe']}->{operation['dst_pipe']}"
        for operation in operations
        if operation["type"] != "barrier"
    )
    barriers_by_pipe = Counter(
        str(operation["src_pipe"]) for operation in operations if operation["type"] == "barrier"
    )
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        if operation["type"] == "barrier":
            continue
        by_key[(operation["src_pipe"], operation["dst_pipe"], operation["event"])].append(operation)
    transitions: list[dict[str, Any]] = []
    for key, keyed_operations in sorted(by_key.items()):
        for before, after in zip(keyed_operations, keyed_operations[1:]):
            transitions.append(
                {
                    "src_pipe": key[0],
                    "dst_pipe": key[1],
                    "event": key[2],
                    "from_line": before["line"],
                    "to_line": after["line"],
                    "from_type": before["type"],
                    "to_type": after["type"],
                    "kind": _transition_kind(before, after),
                    "basis": "lexical_successor",
                }
            )
        by_innermost_loop: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for operation in keyed_operations:
            if operation["loop_stack"]:
                by_innermost_loop[operation["loop_stack"][-1]].append(operation)
        for loop_id, loop_operations in sorted(by_innermost_loop.items()):
            first, last = loop_operations[0], loop_operations[-1]
            if not (str(last["type"]).startswith("set_flag") and str(first["type"]).startswith("wait_flag")):
                continue
            transitions.append(
                {
                    "src_pipe": key[0],
                    "dst_pipe": key[1],
                    "event": key[2],
                    "from_line": last["line"],
                    "to_line": first["line"],
                    "from_type": last["type"],
                    "to_type": first["type"],
                    "kind": "loop_carried",
                    "basis": "inferred_loop_backedge",
                    "loop": loop_id,
                }
            )
    transition_counts = Counter(transition["kind"] for transition in transitions)
    execution_multipliers = [
        _static_execution_multiplier(operation, loop_trip_counts) for operation in operations
    ]
    execution_complete = not has_unstructured_control_flow and all(
        multiplier is not None for multiplier in execution_multipliers
    )

    def weighted_counts(key) -> dict[str, int] | None:
        if not execution_complete:
            return None
        counts: Counter[str] = Counter()
        for operation, multiplier in zip(operations, execution_multipliers, strict=True):
            if multiplier is None:
                raise AssertionError("complete execution multipliers must be integers")
            label = key(operation)
            if label is not None:
                counts[str(label)] += multiplier
        return dict(sorted(counts.items()))

    estimated_instruction_executions = None
    if execution_complete:
        estimated_instruction_executions = 0
        for multiplier in execution_multipliers:
            if multiplier is None:
                raise AssertionError("complete execution multipliers must be integers")
            estimated_instruction_executions += multiplier

    return {
        "schema_version": 1,
        "summary_kind": "actual_post_insert_sync_lowered_ir_v1",
        "function": function,
        "instruction_site_count": len(operations),
        "instruction_sites_by_type": dict(sorted(by_type.items())),
        "instruction_sites_by_pipe_pair": dict(sorted(by_pipe_pair.items())),
        "barrier_sites_by_pipe": dict(sorted(barriers_by_pipe.items())),
        "inside_loop_instruction_sites": sum(bool(operation["loop_stack"]) for operation in operations),
        "outside_loop_instruction_sites": sum(not operation["loop_stack"] for operation in operations),
        "unresolved_control_flow_instruction_sites": sum(
            bool(operation["unresolved_control_flow_stack"]) for operation in operations
        ),
        "has_unstructured_control_flow": has_unstructured_control_flow,
        "static_loop_trip_counts": {
            str(loop_id): trip_count for loop_id, trip_count in sorted(loop_trip_counts.items())
        },
        "static_execution_estimate_status": (
            "COMPLETE" if execution_complete else "INCOMPLETE_DYNAMIC_OR_UNRESOLVED_CONTROL_FLOW"
        ),
        "static_estimated_instruction_executions": estimated_instruction_executions,
        "static_estimated_executions_by_type": weighted_counts(lambda operation: operation["type"]),
        "static_estimated_executions_by_pipe_pair": weighted_counts(
            lambda operation: (
                None
                if operation["type"] == "barrier"
                else f"{operation['src_pipe']}->{operation['dst_pipe']}"
            )
        ),
        "static_estimated_barrier_executions_by_pipe": weighted_counts(
            lambda operation: operation["src_pipe"] if operation["type"] == "barrier" else None
        ),
        "transition_inference_version": "event_key_lexical_and_innermost_backedge_v1",
        "inferred_transition_counts": dict(sorted(transition_counts.items())),
        "operations": operations,
        "inferred_event_transitions": transitions,
    }


def summarize_lowered_pto(  # noqa: PLR0912 - textual MLIR scanner handles function, loop, and sync states
    text: str,
) -> dict[str, dict[str, Any]]:
    """Summarize synchronization instructions present in lowered post-InsertSync PTO IR."""
    functions: dict[str, list[dict[str, Any]]] = {}
    loop_trip_counts_by_function: dict[str, dict[int, int | None]] = {}
    unstructured_control_by_function: dict[str, bool] = {}
    current_function: str | None = None
    function_opened = False
    depth = 0
    loop_stack: list[tuple[int, int]] = []
    unresolved_control_flow_stack: list[tuple[int, int]] = []
    pending_loop: tuple[int, int | None] | None = None
    pending_unresolved_control_flow: int | None = None
    integer_constants: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        code = line.split("//", maxsplit=1)[0]
        if current_function is None:
            function_match = _FUNCTION_RE.search(code)
            if function_match is None:
                continue
            function_name = function_match.group(1)
            if function_name in functions:
                raise ValueError(f"lowered PTO repeats function '{function_name}'")
            functions[function_name] = []
            loop_trip_counts_by_function[function_name] = {}
            unstructured_control_by_function[function_name] = False
            current_function = function_name
            function_opened = "{" in code
            depth = code.count("{") - code.count("}") if function_opened else 0
            loop_stack = []
            unresolved_control_flow_stack = []
            pending_loop = None
            pending_unresolved_control_flow = None
            integer_constants = {}
            continue

        leading = code.lstrip()
        leading_closes = len(leading) - len(leading.lstrip("}"))
        if leading_closes:
            depth -= leading_closes
            while loop_stack and loop_stack[-1][1] > depth:
                loop_stack.pop()
            while unresolved_control_flow_stack and unresolved_control_flow_stack[-1][1] > depth:
                unresolved_control_flow_stack.pop()

        if _UNSTRUCTURED_CONTROL_RE.search(code):
            unstructured_control_by_function[current_function] = True

        if constant_match := _INTEGER_CONSTANT_RE.match(code):
            integer_constants[constant_match.group(1)] = int(constant_match.group(2))

        if _HIGH_LEVEL_SYNC_RE.search(code):
            raise ValueError(
                f"function {current_function} line {line_number}: high-level event op remains; "
                "expected lowered post-InsertSync PTO"
            )
        operation_match = _SYNC_OP_RE.search(code)
        barrier_match = _BARRIER_RE.search(code)
        if operation_match is not None:
            pipes = _PIPE_RE.findall(code)
            if len(pipes) < 2:
                raise ValueError(
                    f"function {current_function} line {line_number}: sync op has fewer than two pipes"
                )
            event_match = _EVENT_RE.search(code)
            event = event_match.group(0) if event_match else None
            if event is None:
                ssa_values = _SSA_RE.findall(code)
                if not ssa_values:
                    raise ValueError(
                        f"function {current_function} line {line_number}: dynamic sync has no event SSA"
                    )
                event = ssa_values[-1]
            functions[current_function].append(
                {
                    "line": line_number,
                    "type": operation_match.group(1),
                    "src_pipe": pipes[0],
                    "dst_pipe": pipes[1],
                    "event": event,
                    "loop_stack": [loop_id for loop_id, _ in loop_stack],
                    "unresolved_control_flow_stack": [
                        control_id for control_id, _ in unresolved_control_flow_stack
                    ],
                }
            )
        elif barrier_match is not None:
            pipes = _PIPE_RE.findall(code)
            if len(pipes) != 1:
                raise ValueError(
                    f"function {current_function} line {line_number}: barrier must name exactly one pipe"
                )
            functions[current_function].append(
                {
                    "line": line_number,
                    "type": "barrier",
                    "src_pipe": pipes[0],
                    "dst_pipe": pipes[0],
                    "event": None,
                    "loop_stack": [loop_id for loop_id, _ in loop_stack],
                    "unresolved_control_flow_stack": [
                        control_id for control_id, _ in unresolved_control_flow_stack
                    ],
                }
            )

        remainder = leading[leading_closes:]
        opens = remainder.count("{")
        closes = remainder.count("}")
        if not function_opened and opens:
            function_opened = True
        if _LOOP_RE.search(code):
            trip_count = None
            if loop_match := _FOR_RE.search(code):
                values = [
                    integer_constants.get(value) if value.startswith("%") else int(value)
                    for value in loop_match.groups()
                ]
                lower, upper, step = values
                if lower is not None and upper is not None and step is not None and step > 0:
                    trip_count = max(0, (upper - lower + step - 1) // step)
            pending_loop = (line_number, trip_count)
        if _UNRESOLVED_STRUCTURED_CONTROL_RE.search(code) or (
            leading_closes and _UNRESOLVED_CONTROL_CONTINUATION_RE.match(remainder)
        ):
            pending_unresolved_control_flow = line_number
        if pending_loop is not None and opens > closes:
            loop_line, trip_count = pending_loop
            loop_stack.append((loop_line, depth + 1))
            loop_trip_counts_by_function[current_function][loop_line] = trip_count
            pending_loop = None
        if pending_unresolved_control_flow is not None and opens > closes:
            unresolved_control_flow_stack.append((pending_unresolved_control_flow, depth + 1))
            pending_unresolved_control_flow = None
        depth += opens - closes
        if function_opened and depth <= 0:
            current_function = None
            function_opened = False
            loop_stack = []
            unresolved_control_flow_stack = []
            pending_loop = None
            pending_unresolved_control_flow = None

    if current_function is not None:
        raise ValueError(f"unterminated function '{current_function}' in lowered PTO")
    if not functions:
        raise ValueError("lowered PTO contains no functions")
    return {
        function: _summarize_lowered_function(
            function,
            operations,
            loop_trip_counts_by_function[function],
            unstructured_control_by_function[function],
        )
        for function, operations in sorted(functions.items())
    }


def summarize_arm_manifest(path: str | Path) -> dict[str, Any]:
    """Collect actual post-InsertSync summaries for every declared placement arm."""
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{source}: expected arm manifest schema_version=1")
    expected_arms = payload.get("expected_arms")
    cells = payload.get("cells")
    if (
        not isinstance(expected_arms, list)
        or not expected_arms
        or not all(isinstance(arm, str) and arm for arm in expected_arms)
    ):
        raise ValueError(f"{source}: expected_arms must be a non-empty string array")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{source}: cells must be a non-empty array")

    rows: list[dict[str, Any]] = []
    arms_by_case: dict[tuple[str, str], set[str]] = defaultdict(set)
    functions_by_case_arm: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(dict)
    cell_keys: set[tuple[str, str, str]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(f"{source}: cell {index} must be an object")
        case, capacity, arm = cell.get("case"), cell.get("capacity"), cell.get("arm")
        raw_pto = cell.get("post_insert_sync_pto")
        if (
            not isinstance(case, str)
            or not case
            or not isinstance(capacity, str)
            or not capacity
            or not isinstance(arm, str)
            or not arm
            or not isinstance(raw_pto, str)
            or not raw_pto
        ):
            raise ValueError(f"{source}: cell {index} has incomplete identity or PTO path")
        key = (case, capacity, arm)
        if key in cell_keys:
            raise ValueError(f"{source}: duplicate arm cell {key}")
        cell_keys.add(key)
        arms_by_case[(case, capacity)].add(arm)
        pto_path = (source.parent / raw_pto).resolve()
        pto_bytes = pto_path.read_bytes()
        summaries = summarize_lowered_pto(pto_bytes.decode())
        requested_function = cell.get("function")
        if requested_function is not None:
            if not isinstance(requested_function, str) or not requested_function:
                raise ValueError(f"{source}: cell {index} has invalid function identity")
            if requested_function not in summaries:
                raise ValueError(f"{pto_path}: function '{requested_function}' is not present")
            summaries = {requested_function: summaries[requested_function]}
        functions_by_case_arm[(case, capacity)][arm] = set(summaries)
        for function, summary in sorted(summaries.items()):
            rows.append(
                {
                    "case": case,
                    "capacity": capacity,
                    "arm": arm,
                    "function": function,
                    "post_insert_sync_pto": raw_pto,
                    "post_insert_sync_pto_sha256": hashlib.sha256(pto_bytes).hexdigest(),
                    "summary": summary,
                }
            )
    required = set(expected_arms)
    for case_key, actual in sorted(arms_by_case.items()):
        if actual != required:
            raise ValueError(
                f"{source}: arm coverage mismatch for {case_key}: "
                f"missing={sorted(required - actual)}, added={sorted(actual - required)}"
            )
        function_sets = functions_by_case_arm[case_key]
        reference_arm = expected_arms[0]
        reference_functions = function_sets[reference_arm]
        for arm in expected_arms[1:]:
            if function_sets[arm] != reference_functions:
                raise ValueError(
                    f"{source}: function-set mismatch for {case_key}: "
                    f"{reference_arm}={sorted(reference_functions)}, "
                    f"{arm}={sorted(function_sets[arm])}"
                )
    return {
        "schema_version": 1,
        "summary_kind": "actual_post_insert_sync_per_arm_v1",
        "manifest": str(source),
        "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "expected_arms": expected_arms,
        "cell_count": len(cells),
        "function_summary_count": len(rows),
        "cells": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ptoas_sync_summary",
        description="Compare PTOAS InsertSync JSONL summaries by function.",
    )
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument(
        "--arm-manifest",
        type=Path,
        help="Collect actual lowered post-InsertSync summaries for every arm in a manifest.",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        if args.arm_manifest is not None:
            if args.baseline is not None or args.candidate is not None:
                raise ValueError("--arm-manifest cannot be combined with baseline/candidate paths")
            result = summarize_arm_manifest(args.arm_manifest)
        else:
            if args.baseline is None or args.candidate is None:
                raise ValueError("baseline and candidate paths are required without --arm-manifest")
            result = diff_sync_summaries(
                load_sync_summaries(args.baseline),
                load_sync_summaries(args.candidate),
            )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
