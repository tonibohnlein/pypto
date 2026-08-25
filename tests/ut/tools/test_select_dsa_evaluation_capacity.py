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
import json
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


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
    problem_documents = tmp_path / "problem-documents"
    problem_documents.mkdir()
    (problem_documents / "toy__target-0123456789abcdef.dsa.json").write_text(
        """{
  "problem": {
    "pools": [{"id": 1, "name": "Vec", "capacity": 4096, "reserved_ranges": []}],
    "buffers": [
      {"id": 0, "size": 1900, "alignment": 32, "allowed_pools": [1], "live_intervals": []},
      {"id": 1, "size": 1900, "alignment": 32, "allowed_pools": [1], "live_intervals": []}
    ]
  }
}\n"""
    )
    problem_status = tmp_path / "problem-status.tsv"
    _write_tsv(
        problem_status,
        ["driver_id", "instance", "problem_fingerprint", "status"],
        [
            {
                "driver_id": "toy",
                "instance": "target",
                "problem_fingerprint": "0123456789abcdef",
                "status": "MEASURED",
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
        "reuse_cost",
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
                    "reuse_cost": str(
                        {"geometry_ff": 7, "geometry_cg": 7, "cypress": 5, "dsa_rp_cg": 2}[arm]
                    ),
                    "pool_placement_sha256": f"{capacity}-{arm}",
                    "cypress_actual_alias_pairs": "2" if capacity != "native" else "0",
                }
            )
    _write_tsv(screen, columns, rows)
    return problems, screen, problem_documents, problem_status


def _read_selection(path: Path) -> dict[str, str]:
    with path.open(newline="") as source:
        return next(csv.DictReader(source, delimiter="\t"))


def test_selects_least_restrictive_capacity_meeting_uniform_pressure(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    summary = selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "out"
    )
    row = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert summary["primary_count"] == 1
    assert summary["uses_device_latency"] is False
    assert row["evaluation_capacity"] == "tight"
    assert row["mandatory_disjoint_bytes_lower_bound"] == "3800"
    assert row["forced_reuse_bytes"] == "1240"
    assert row["forced_reuse_percent"] == "32.631579"
    assert row["minimum_forced_reuse_percent"] == "25"
    assert row["cypress_actual_alias_pairs"] == "2"
    assert row["dsa_rp_minus_cypress_reuse_cost"] == "-3"
    assert summary["dsa_rp_vs_cypress_reuse_cost"] == {"lower": 1, "equal": 0, "higher": 0}
    assert summary["minimum_forced_reuse_percent"] == 25


def test_rejects_capacity_without_three_distinct_policy_placements(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    for row in rows:
        if row["capacity_label"] != "native" and row["arm"] == "dsa_rp_cg":
            row["pool_placement_sha256"] = f"{row['capacity_label']}-cypress"
    _write_tsv(screen, list(rows[0]), rows)
    selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "out"
    )
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert selected["selection_status"] == "NO_THREE_WAY_PLACEMENT_SEPARATION"
    assert selected["evaluation_capacity"] == ""


def test_selection_identity_ignores_solver_runtime(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    first = selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "first"
    )
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    for index, row in enumerate(rows):
        row["runtime_us"] = str(10000 + index)
    _write_tsv(screen, list(rows[0]), rows)
    second = selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "second"
    )

    assert second["selection_sha256"] == first["selection_sha256"]


def test_sidecar_hashes_written_tsv(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "out"
    )
    selection_path = tmp_path / "out" / "evaluation-instances.tsv"
    sidecar_hash = (tmp_path / "out" / "evaluation-instances.tsv.sha256").read_text().split()[0]

    assert sidecar_hash == hashlib.sha256(selection_path.read_bytes()).hexdigest()


def test_incomplete_arm_cell_is_not_primary(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    rows = [
        row
        for row in csv.DictReader(screen.open(), delimiter="\t")
        if not (row["capacity_label"] != "native" and row["arm"] == "geometry_cg")
    ]
    _write_tsv(screen, list(rows[0]), rows)
    selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "out"
    )
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert selected["selection_status"] == "INCOMPLETE_FOUR_ARM_CELL"


def test_voluntary_cypress_alias_does_not_prove_capacity_pressure(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    for row in rows:
        if row["capacity_label"] == "native" and row["arm"] == "cypress":
            row["cypress_actual_alias_pairs"] = "3"
    _write_tsv(screen, list(rows[0]), rows)
    selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "out"
    )
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert selected["evaluation_capacity"] == "tight"
    assert selected["capacity_attempts"].startswith(
        "native:CAPACITY_DOES_NOT_FORCE_REUSE;half:INSUFFICIENT_FORCED_REUSE_PRESSURE;"
    )


def test_pressure_threshold_is_configurable_and_timing_blind(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    summary = selector.select_evaluation_capacities(
        problems,
        screen,
        problem_documents,
        problem_status,
        tmp_path / "out",
        minimum_forced_reuse_percent=5,
    )
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert selected["evaluation_capacity"] == "half"
    assert selected["forced_reuse_percent"] == "5.684211"
    assert selected["minimum_forced_reuse_percent"] == "5"
    assert summary["minimum_forced_reuse_percent"] == 5


def test_pressure_policy_participates_in_selection_identity(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    zero = selector.select_evaluation_capacities(
        problems,
        screen,
        problem_documents,
        problem_status,
        tmp_path / "zero",
        minimum_forced_reuse_percent=0,
    )
    five = selector.select_evaluation_capacities(
        problems,
        screen,
        problem_documents,
        problem_status,
        tmp_path / "five",
        minimum_forced_reuse_percent=5,
    )

    assert _read_selection(tmp_path / "zero" / "evaluation-instances.tsv")["evaluation_capacity"] == "half"
    assert _read_selection(tmp_path / "five" / "evaluation-instances.tsv")["evaluation_capacity"] == "half"
    assert zero["selection_sha256"] != five["selection_sha256"]


def test_invalid_pressure_threshold_fails_closed(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)

    with pytest.raises(ValueError, match="between 0 and 100"):
        selector.select_evaluation_capacities(
            problems,
            screen,
            problem_documents,
            problem_status,
            tmp_path / "out",
            minimum_forced_reuse_percent=101,
        )


def test_device_blocked_problem_is_not_selected(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    rows = list(csv.DictReader(problem_status.open(), delimiter="\t"))
    rows[0]["status"] = "DRIVER_RUNTIME_BLOCKED"
    _write_tsv(problem_status, list(rows[0]), rows)
    summary = selector.select_evaluation_capacities(
        problems, screen, problem_documents, problem_status, tmp_path / "out"
    )
    selected = _read_selection(tmp_path / "out" / "evaluation-instances.tsv")

    assert summary["primary_count"] == 0
    assert selected["device_status"] == "DRIVER_RUNTIME_BLOCKED"
    assert selected["selection_status"] == "NOT_DEVICE_MEASURABLE"


def test_hard_colocation_fails_closed_in_disjoint_bound(tmp_path: Path):
    problems, screen, problem_documents, problem_status = _fixture(tmp_path)
    problem_path = next(problem_documents.glob("*.dsa.json"))
    document = json.loads(problem_path.read_text())
    document["problem"]["constraints"] = {"colocations": [{"first": 0, "second": 1}]}
    problem_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="has hard colocations"):
        selector.select_evaluation_capacities(
            problems, screen, problem_documents, problem_status, tmp_path / "out"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
