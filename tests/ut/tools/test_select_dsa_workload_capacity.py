# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "select_dsa_workload_capacity.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_select_dsa_workload_capacity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _problem() -> dict:
    return {
        "problem": {
            "pools": [{"id": 1, "name": "Vec", "capacity": 512, "reserved_ranges": []}],
            "buffers": [
                {"id": 0, "size": 160, "allowed_pools": [1]},
                {"id": 1, "size": 160, "allowed_pools": [1]},
                {"id": 2, "size": 160, "allowed_pools": [1]},
            ],
            "constraints": {"colocations": []},
            "cost_model": {"reuse_penalties": [{"first": 0, "second": 1, "cost": 1}]},
        }
    }


def _fixture(tmp_path: Path, *, cypress_equals_rp: bool = False) -> tuple[Path, ...]:
    problem_fingerprint = "0123456789abcdef"
    cohort = tmp_path / "cohort.tsv"
    _write_tsv(
        cohort,
        [
            {
                "script": "models/toy.py",
                "measurement_unit": "SINGLE_KERNEL_DRIVER",
                "dsa_instance_count": "1",
                "problem_fingerprints": f"kernel={problem_fingerprint}",
            }
        ],
    )
    corpus = tmp_path / "corpus"
    (corpus / "captures/toy").mkdir(parents=True)
    problem_path = corpus / "captures/toy/pypto_kernel.dsa.json"
    problem_path.write_text(json.dumps(_problem(), sort_keys=True))
    instances = tmp_path / "instances.tsv"
    _write_tsv(
        instances,
        [
            {
                "script": "models/toy.py",
                "instance": "kernel",
                "problem_fingerprint": problem_fingerprint,
                "document": "captures/toy/pypto_kernel.dsa.json",
                "document_sha256": hashlib.sha256(problem_path.read_bytes()).hexdigest(),
            }
        ],
    )
    status = tmp_path / "status.tsv"
    _write_tsv(
        status,
        [{"script": "models/toy.py", "terminal_status": "VERIFIED_ALL_CAPACITIES"}],
    )

    feasibility_rows: list[dict[str, str]] = []
    map_rows: list[dict[str, str]] = []
    replay = tmp_path / "replay"
    for capacity_index, capacity in enumerate(selector.CAPACITIES_TIGHTEST_FIRST):
        cap = 300 + capacity_index * 60
        for arm in selector.ARMS:
            digest = f"{capacity}-{arm}"
            if arm == "dsa_rp_cg" and cypress_equals_rp:
                digest = f"{capacity}-cypress"
            feasibility_rows.append(
                {
                    "script": "models/toy.py",
                    "instance": "kernel",
                    "capacity_label": capacity,
                    "arm": arm,
                    "status": "feasible",
                    "validation": "VALID",
                    "capacity_profile": f"1={cap}",
                    "reuse_cost": "1" if arm == "cypress" else "0",
                    "runtime_us": str(capacity_index * 100),
                }
            )
            map_dir = replay / f"maps/{capacity}/{arm}"
            map_dir.mkdir(parents=True)
            offsets = {
                "geometry_ff": (0, 160, 0),
                "geometry_cg": (0, 160, 0),
                "cypress": (0, 0, 160),
                "dsa_rp_cg": (0, 160, 160),
            }[arm]
            if arm == "dsa_rp_cg" and cypress_equals_rp:
                offsets = (0, 0, 160)
            (map_dir / "pypto_kernel.dsa.solution.json").write_text(
                json.dumps(
                    {
                        "placements": [
                            {"buffer": index, "pool": 1, "offset": offset}
                            for index, offset in enumerate(offsets)
                        ]
                    }
                )
            )
            map_rows.append(
                {
                    "script": "models/toy.py",
                    "capacity_label": capacity,
                    "arm": arm,
                    "map_dir": f"maps/{capacity}/{arm}",
                    "map_digest": digest,
                    "provenance_verified": "YES",
                }
            )
    feasibility = tmp_path / "feasibility.tsv"
    maps = tmp_path / "maps.tsv"
    _write_tsv(feasibility, feasibility_rows)
    _write_tsv(maps, map_rows)
    return cohort, instances, feasibility, maps, status, corpus, replay


def _select(tmp_path: Path, *, cypress_equals_rp: bool = False, name: str = "out") -> dict:
    paths = _fixture(tmp_path, cypress_equals_rp=cypress_equals_rp)
    return selector.select_workload_capacities(*paths, tmp_path / name)


def test_selects_tightest_capacity_with_forced_cypress_reuse(tmp_path: Path) -> None:
    summary = _select(tmp_path)
    row = summary["workloads"][0]

    assert row["selection_status"] == "PRIMARY_THREE_WAY"
    assert row["evaluation_capacity"] == "tight"
    assert row["forced_reuse_bytes"] == 180
    assert row["cypress_alias_pairs"] == 1
    assert row["cypress_penalized_alias_pairs"] == 1
    assert summary["uses_device_latency"] is False


def test_identical_cypress_and_dsa_rp_is_retained_as_null_control(tmp_path: Path) -> None:
    summary = _select(tmp_path, cypress_equals_rp=True)
    row = summary["workloads"][0]

    assert row["selection_status"] == "NULL_CONTROL_CYPRESS_DSA_RP_IDENTICAL"
    assert row["evaluation_capacity"] == "tight"
    assert summary["primary_count"] == 0
    assert summary["null_control_count"] == 1


def test_selection_identity_ignores_solver_runtime(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = selector.select_workload_capacities(*paths, tmp_path / "first")
    rows = list(csv.DictReader(paths[2].open(), delimiter="\t"))
    for index, row in enumerate(rows):
        row["runtime_us"] = str(100000 + index)
    _write_tsv(paths[2], rows)
    second = selector.select_workload_capacities(*paths, tmp_path / "second")

    assert second["selection_sha256"] == first["selection_sha256"]


def test_fresh_problem_fingerprint_drift_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = list(csv.DictReader(paths[1].open(), delimiter="\t"))
    rows[0]["problem_fingerprint"] = "fedcba9876543210"
    _write_tsv(paths[1], rows)

    with pytest.raises(ValueError, match="Fresh DSA problems drifted"):
        selector.select_workload_capacities(*paths, tmp_path / "out")


def test_colocation_components_count_once_in_disjoint_bound() -> None:
    problem = _problem()["problem"]
    problem["constraints"]["colocations"] = [{"first": 0, "second": 1}]

    assert selector.mandatory_disjoint_bytes_by_pool(problem) == {1: 320}


def test_checked_in_driver_first_freeze_is_timing_blind_and_complete() -> None:
    freeze_path = (
        Path(__file__).parents[3]
        / ".claude"
        / "skills"
        / "incore-profiling"
        / "dsa_driver_first_evaluation_v1.json"
    )
    freeze = json.loads(freeze_path.read_text())
    workloads = freeze["workloads"]

    assert freeze["uses_device_latency"] is False
    assert freeze["source_archive_sha256"] == (
        "6e41ba62e6018529d55000b3db1eb55e971662c0f2f6556e817e1ae020a105e7"
    )
    assert freeze["workload_count"] == len(workloads) == 20
    assert freeze["primary_count"] == 18
    assert freeze["null_control_count"] == 2
    assert {row["evaluation_capacity"] for row in workloads} == {"tight"}
    assert len({row["script"] for row in workloads}) == 20
    assert all(row["forced_reuse_bytes"] > 0 for row in workloads)
    assert all(row["cypress_penalized_alias_pairs"] > 0 for row in workloads)
    assert not any("latency" in key or "timing" in key for row in workloads for key in row)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
