# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for timing-blind DSA evaluation-capacity selection."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / ".claude"
    / "skills"
    / "incore-profiling"
    / "select_dsa_evaluation_capacity.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_select_dsa_evaluation_capacity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    problems = tmp_path / "problems.tsv"
    _write_tsv(
        problems,
        [
            "tier",
            "driver_id",
            "instance",
            "problem_fingerprint",
            "operation_class",
            "pool",
            "pool_id",
            "buffers",
            "reuse_penalties",
        ],
        [
            {
                "tier": "expanded",
                "driver_id": "toy",
                "instance": "target",
                "problem_fingerprint": "0123456789abcdef",
                "operation_class": "reduction",
                "pool": "Vec",
                "pool_id": "1",
                "buffers": "12",
                "reuse_penalties": "7",
            }
        ],
    )
    screen = tmp_path / "screen.tsv"
    columns = [
        "tag",
        "pool_id",
        "pool",
        "capacity_label",
        "capacity",
        "arm",
        "status",
        "runtime_us",
        "pool_placement_sha256",
        "cypress_actual_alias_pairs",
    ]
    rows: list[dict[str, str]] = []
    for capacity_index, capacity in enumerate(selector.CAPACITIES):
        for arm_index, arm in enumerate(selector.ARMS):
            rows.append(
                {
                    "tag": "toy__target-0123456789abcdef",
                    "pool_id": "1",
                    "pool": "Vec",
                    "capacity_label": capacity,
                    "capacity": str(4096 - 512 * capacity_index),
                    "arm": arm,
                    "status": "feasible",
                    "runtime_us": str(arm_index + capacity_index),
                    "pool_placement_sha256": f"{capacity}-{arm}",
                    "cypress_actual_alias_pairs": "2" if capacity != "native" else "0",
                }
            )
    _write_tsv(screen, columns, rows)
    return problems, screen


def _read_selection(path: Path) -> dict[str, str]:
    with path.open(newline="") as source:
        return next(csv.DictReader(source, delimiter="\t"))


def test_selects_least_restrictive_reuse_pressured_capacity(tmp_path: Path):
    problems, screen = _fixture(tmp_path)
    summary = selector.select_evaluation_capacities(problems, screen, tmp_path / "out")
    row = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert summary["primary_count"] == 1
    assert summary["uses_device_latency"] is False
    assert row["evaluation_capacity"] == "half"
    assert row["cypress_actual_alias_pairs"] == "2"


def test_rejects_capacity_without_three_distinct_policy_placements(tmp_path: Path):
    problems, screen = _fixture(tmp_path)
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    for row in rows:
        if row["capacity_label"] != "native" and row["arm"] == "dsa_rp_cg":
            row["pool_placement_sha256"] = f"{row['capacity_label']}-cypress"
    _write_tsv(screen, list(rows[0]), rows)
    selector.select_evaluation_capacities(problems, screen, tmp_path / "out")
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert selected["selection_status"] == "NO_THREE_WAY_PLACEMENT_SEPARATION"
    assert selected["evaluation_capacity"] == ""


def test_selection_identity_ignores_solver_runtime(tmp_path: Path):
    problems, screen = _fixture(tmp_path)
    first = selector.select_evaluation_capacities(problems, screen, tmp_path / "first")
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    for index, row in enumerate(rows):
        row["runtime_us"] = str(10000 + index)
    _write_tsv(screen, list(rows[0]), rows)
    second = selector.select_evaluation_capacities(problems, screen, tmp_path / "second")

    assert second["selection_sha256"] == first["selection_sha256"]


def test_sidecar_hashes_written_tsv(tmp_path: Path):
    problems, screen = _fixture(tmp_path)
    selector.select_evaluation_capacities(problems, screen, tmp_path / "out")
    selection_path = tmp_path / "out" / "evaluation-instances.tsv"
    sidecar_hash = (tmp_path / "out" / "evaluation-instances.tsv.sha256").read_text().split()[0]

    assert sidecar_hash == hashlib.sha256(selection_path.read_bytes()).hexdigest()


def test_incomplete_arm_cell_is_not_primary(tmp_path: Path):
    problems, screen = _fixture(tmp_path)
    rows = [
        row
        for row in csv.DictReader(screen.open(), delimiter="\t")
        if not (row["capacity_label"] != "native" and row["arm"] == "geometry_cg")
    ]
    _write_tsv(screen, list(rows[0]), rows)
    selector.select_evaluation_capacities(problems, screen, tmp_path / "out")
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert selected["selection_status"] == "INCOMPLETE_FOUR_ARM_CELL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
