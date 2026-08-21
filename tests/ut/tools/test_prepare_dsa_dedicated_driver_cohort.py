# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for dedicated functional-driver DSA cohort freezing."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / ".claude"
    / "skills"
    / "incore-profiling"
    / "prepare_dsa_dedicated_driver_cohort.py"
)
_CATALOG = _SCRIPT.with_name("dsa_dedicated_driver_cohort_v1.json")
_SPEC = importlib.util.spec_from_file_location("_test_prepare_dsa_dedicated_driver_cohort", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
cohort = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cohort)


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    lib = tmp_path / "lib"
    source = lib / "models" / "toy" / "driver.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """import pypto.language as pl

@pl.jit
def driver_test(x):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"target\"):
        return x

def golden_driver(tensors):
    return None

def build_tensor_specs():
    return []

if __name__ == \"__main__\":
    pass
"""
    )
    subprocess.run(["git", "init", "-q"], cwd=lib, check=True)
    subprocess.run(["git", "add", "models/toy/driver.py"], cwd=lib, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=lib,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=lib, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_policy": "test",
                "timing_blind_selection": True,
                "pypto_lib_revision": revision,
                "arms": list(cohort.ARMS),
                "capacities": list(cohort.CAPACITIES),
                "input_seed": 19,
                "measurement_contract": dict(cohort._MEASUREMENT_CONTRACT),
                "drivers": [
                    {
                        "tier": "canary_expanded",
                        "driver_id": "toy",
                        "script": "models/toy/driver.py",
                        "entry": "driver_test",
                        "golden": "golden_driver",
                        "specs": "build_tensor_specs",
                        "argv": [],
                        "targets": [
                            {
                                "instance": "target",
                                "problem_fingerprint": "0123456789abcdef",
                                "pool_id": "1",
                                "operation_class": "reduction",
                            }
                        ],
                    }
                ],
            }
        )
    )

    problems = tmp_path / "problems.tsv"
    problem_columns = [
        "script",
        "instance",
        "problem_fingerprint",
        "buffers",
        "pools",
        "pool_names",
        "reuse_penalties",
        "recognizer",
    ]
    problem_row = {
        "script": "models/toy/driver.py",
        "instance": "target",
        "problem_fingerprint": "0123456789abcdef",
        "buffers": "12",
        "pools": "1",
        "pool_names": "Vec",
        "reuse_penalties": "7",
        "recognizer": "test",
    }
    # Repeated invocations of one semantic problem are accepted.
    _write_tsv(problems, problem_columns, [problem_row, problem_row])

    screen = tmp_path / "screen.tsv"
    screen_columns = [
        "tag",
        "pool_id",
        "pool",
        "capacity_label",
        "capacity",
        "arm",
        "status",
        "placement_sha256",
    ]
    rows = []
    for capacity_index, capacity in enumerate(cohort.CAPACITIES):
        for arm_index, arm in enumerate(cohort.ARMS):
            rows.append(
                {
                    "tag": "toy__target-0123456789abcdef",
                    "pool_id": "1",
                    "pool": "Vec",
                    "capacity_label": capacity,
                    "capacity": str(4096 - capacity_index * 512),
                    "arm": arm,
                    "status": "feasible",
                    "placement_sha256": f"{capacity_index:02x}{arm_index:02x}",
                }
            )
    _write_tsv(screen, screen_columns, rows)
    return catalog_path, lib, problems, screen


def test_prepare_cohort_freezes_direct_golden_driver(tmp_path: Path):
    catalog, lib, problems, screen = _fixture(tmp_path)
    output = tmp_path / "out"
    summary = cohort.prepare_cohort(catalog, lib, problems, screen, output)
    assert summary == {
        "verdict": "DEDICATED_DRIVER_COHORT_FROZEN",
        "cohort_sha256": summary["cohort_sha256"],
        "driver_count": 1,
        "problem_count": 1,
        "canary_driver_count": 1,
        "expanded_problem_count": 1,
        "operation_class_count": 1,
        "cell_count": 16,
    }
    assert len((output / "preflight-cells.tsv").read_text().splitlines()) == 17
    frozen = json.loads((output / "cohort-frozen.json").read_text())
    assert frozen["schema_version"] == 2
    assert "screen_results_sha256" not in frozen["inputs"]
    assert frozen["drivers"][0]["definition_source_sha256"]["models/toy/driver.py"]


def test_cohort_identity_ignores_screen_runtime_and_row_order(tmp_path: Path):
    catalog, lib, problems, screen = _fixture(tmp_path)
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    columns = [*rows[0], "runtime_us"]
    for index, row in enumerate(rows):
        row["runtime_us"] = str(index + 1)
    _write_tsv(screen, columns, rows)
    first = cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "first")

    for index, row in enumerate(reversed(rows)):
        row["runtime_us"] = str(10000 + index)
    _write_tsv(screen, columns, list(reversed(rows)))
    second = cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "second")

    assert second["cohort_sha256"] == first["cohort_sha256"]


def test_cohort_identity_changes_with_selected_placement(tmp_path: Path):
    catalog, lib, problems, screen = _fixture(tmp_path)
    first = cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "first")
    rows = list(csv.DictReader(screen.open(), delimiter="\t"))
    rows[0]["placement_sha256"] = "changed-placement"
    _write_tsv(screen, list(rows[0]), rows)
    second = cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "second")

    assert second["cohort_sha256"] != first["cohort_sha256"]


def test_catalog_extension_excludes_and_adds_drivers(tmp_path: Path):
    catalog, _, _, _ = _fixture(tmp_path)
    base = json.loads(catalog.read_text())
    base["drivers"][0]["driver_id"] = "excluded"
    catalog.write_text(json.dumps(base))
    overlay = tmp_path / "overlay.json"
    replacement = dict(base["drivers"][0])
    replacement["driver_id"] = "replacement"
    overlay.write_text(
        json.dumps(
            {
                "extends": catalog.name,
                "selection_policy": "test_v2",
                "exclude_driver_ids": ["excluded"],
                "drivers": [replacement],
            }
        )
    )

    resolved, hashes = cohort._load_catalog(overlay)

    assert resolved["selection_policy"] == "test_v2"
    assert [driver["driver_id"] for driver in resolved["drivers"]] == ["replacement"]
    assert set(hashes) == {
        "catalog_source_sha256",
        "base_catalog_source_sha256",
        "catalog_semantics_sha256",
    }


def test_catalog_extension_rejects_unknown_exclusion(tmp_path: Path):
    catalog, _, _, _ = _fixture(tmp_path)
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"extends": catalog.name, "exclude_driver_ids": ["typo"]}))

    with pytest.raises(ValueError, match="excludes unknown drivers: typo"):
        cohort._load_catalog(overlay)


def test_prepare_cohort_rejects_infeasible_cell(tmp_path: Path):
    catalog, lib, problems, screen = _fixture(tmp_path)
    text = screen.read_text().replace("\tfeasible\t", "\tno_fit\t", 1)
    screen.write_text(text)
    with pytest.raises(ValueError, match="infeasible screen cell"):
        cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "out")


def test_prepare_cohort_rejects_missing_direct_golden(tmp_path: Path):
    catalog, lib, problems, screen = _fixture(tmp_path)
    data = json.loads(catalog.read_text())
    data["drivers"][0]["golden"] = "missing_golden"
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="golden function 'missing_golden' is absent"):
        cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "out")


def test_prepare_cohort_rejects_missing_original_source_symbol(tmp_path: Path):
    catalog, lib, problems, screen = _fixture(tmp_path)
    data = json.loads(catalog.read_text())
    data["drivers"][0]["targets"][0]["source_name_hint"] = "not_the_kernel"
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source symbol/name_hint 'not_the_kernel' is absent"):
        cohort.prepare_cohort(catalog, lib, problems, screen, tmp_path / "out")


def test_catalog_forbids_performance_driven_mode():
    with pytest.raises(ValueError, match="timing_blind_selection"):
        cohort._validate_catalog(
            {
                "schema_version": 1,
                "timing_blind_selection": False,
                "arms": list(cohort.ARMS),
                "capacities": list(cohort.CAPACITIES),
                "drivers": [{}],
            }
        )


def test_catalog_forbids_whole_driver_latency_as_kernel_timing(tmp_path: Path):
    catalog, _, _, _ = _fixture(tmp_path)
    data = json.loads(catalog.read_text())
    data["measurement_contract"]["allow_whole_driver_as_kernel_timing"] = True
    with pytest.raises(ValueError, match="measurement_contract"):
        cohort._validate_catalog(data)


def test_checked_in_catalog_has_expected_bounded_cohort():
    data = json.loads(_CATALOG.read_text())
    cohort._validate_catalog(data)
    drivers = data["drivers"]
    assert len(drivers) == 14
    assert sum(driver["tier"] in {"canary", "canary_expanded"} for driver in drivers) == 9
    assert (
        sum(len(driver["targets"]) for driver in drivers if driver["tier"] in {"expanded", "canary_expanded"})
        == 19
    )
    assert len({target["operation_class"] for driver in drivers for target in driver["targets"]}) == 14
