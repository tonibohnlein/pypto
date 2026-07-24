# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate compile-time structure for the issue #2131 experiment."""

import argparse
import json
import re
from pathlib import Path

EXPECTED_ORDER = {
    "baseline": ["extract", "extract", "matmul", "store", "matmul", "store"],
    "dbc": ["extract", "extract", "matmul", "matmul", "store", "store"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    return parser.parse_args()


def _inner_op_sequences(path: Path) -> list[list[str]]:
    lines = path.read_text().splitlines()
    sequences: list[list[str]] = []
    for index, line in enumerate(lines):
        if "for col" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        ops: list[str] = []
        for body_line in lines[index + 1 :]:
            if body_line.strip() and len(body_line) - len(body_line.lstrip()) <= indent:
                break
            for op in ("extract", "matmul", "store"):
                if f"pl.tile.{op}(" in body_line:
                    ops.append(op)
        sequences.append(ops)
    if not sequences:
        raise AssertionError(f"No inner col loop found in {path}")
    return sequences


def _memory_summary(path: Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    row = re.compile(r"^\s*(Mat|Left|Right|Acc)\s+\|\s+([0-9.]+\s+[KM]B)\s+\|.*\|\s+(\d+)\s*$")
    for line in path.read_text().splitlines():
        match = row.match(line)
        if match:
            result[match.group(1)] = (match.group(2), int(match.group(3)))
    return result


def _validate_case(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    case_dir = result_path.parent
    variant = result["variant"]
    planner = result["planner"]
    assert variant in EXPECTED_ORDER, result
    assert planner in {"pypto", "ptoas"}, result
    assert result["passed"], result
    assert result["compile_only"], result

    lower = next((case_dir / "passes_dump").glob("*_after_LowerPipelineLoops.py"))
    canonical = next((case_dir / "passes_dump").glob("*_after_CanonicalizeIOOrder.py"))
    lower_text = lower.read_text()
    orders = _inner_op_sequences(canonical)
    expected_orders = [EXPECTED_ORDER[variant], EXPECTED_ORDER[variant]]
    assert orders == expected_orders, (case_dir, orders)

    matmul_memberships = re.findall(
        r"pl\.tile\.matmul\([^\n]*attrs=\{\"pipeline_membership\": \"([^\"]+)\"\}",
        lower_text,
    )
    if variant == "baseline":
        assert not matmul_memberships, (case_dir, matmul_memberships)
    else:
        assert matmul_memberships == ["0:0", "0:1", "0:0", "0:1"], (
            case_dir,
            matmul_memberships,
        )

    summary: dict[str, object] = {
        "variant": variant,
        "planner": planner,
        "canonical_orders": orders,
        "acc_memberships": matmul_memberships,
    }
    if planner == "pypto":
        memory_path = case_dir / "report" / "memory_after_AllocateMemoryAddr.txt"
        memory = _memory_summary(memory_path)
        expected_acc = ("8.0 KB", 1) if variant == "baseline" else ("16.0 KB", 2)
        assert memory["Mat"] == ("256.0 KB", 2), memory
        assert memory["Left"] == ("4.0 KB", 1), memory
        assert memory["Right"] == ("64.0 KB", 2), memory
        assert memory["Acc"] == expected_acc, memory

        pto = next((case_dir / "kernels" / "aic").glob("*.pto"))
        acc_addresses = sorted(
            {
                int(value)
                for value in re.findall(
                    r"alloc_tile addr = %c(\d+)_i64[^\n]*loc=acc",
                    pto.read_text(),
                )
            }
        )
        expected_addresses = [0] if variant == "baseline" else [0, 8192]
        assert acc_addresses == expected_addresses, (case_dir, acc_addresses)
        summary["memory"] = memory
        summary["acc_addresses"] = acc_addresses
        summary["placement_status"] = "VERIFIED"
    else:
        summary["placement_status"] = "PENDING_PTOAS_FINAL_PLACEMENT_INSPECTION"
    return summary


def main() -> None:
    args = _parse_args()
    result_paths = sorted(args.results_root.rglob("issue2131_result.json"))
    if not result_paths:
        raise RuntimeError(f"No issue2131_result.json found under {args.results_root}")
    cases_by_key: dict[tuple[object, object], dict[str, object]] = {}
    for path in result_paths:
        case = _validate_case(path)
        key = (case["planner"], case["variant"])
        assert key not in cases_by_key, f"Duplicate structural case {key}: {path}"
        cases_by_key[key] = case
    expected = {
        ("pypto", "baseline"),
        ("pypto", "dbc"),
        ("ptoas", "baseline"),
        ("ptoas", "dbc"),
    }
    assert set(cases_by_key) == expected, (set(cases_by_key), expected)
    print(
        json.dumps(
            {
                "status": "PASS_WITH_PTOAS_PLACEMENT_PENDING",
                "cases": list(cases_by_key.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
