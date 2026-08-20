# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for timing-blind DSA measurement-cohort selection."""

import csv
import json

import pytest
from pypto.tools import dsa_measurement_cohort


def _case_rows(case_id: str, *, family: str = "family", parent: str = "parent") -> list[dict[str, str]]:
    return [
        {
            "case_id": case_id,
            "family": family,
            "parent": parent,
            "function": f"fn_{case_id}",
            "capacity": capacity,
            "arm": arm,
            "solve_status": "FEASIBLE",
            "schedule_status": "STATIC_SCHEDULE",
            "launch_status": "RUNNABLE",
            "correctness_status": "PASS",
            "endpoint_digest": f"{case_id}-{capacity}-{arm}",
        }
        for capacity in dsa_measurement_cohort.CAPACITIES
        for arm in dsa_measurement_cohort.ARMS
    ]


def _write_preflight(path, rows, extra_columns=()):
    analysis_columns = ("schedule_status",) if "schedule_status" in rows[0] else ()
    columns = [*dsa_measurement_cohort.REQUIRED_COLUMNS, *analysis_columns, *extra_columns]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_construct_cohort_requires_all_arms_and_capacities_but_not_analysis_support():
    good = _case_rows("good")
    missing = _case_rows("missing")[:-1]
    branch = _case_rows("branch")
    branch[0]["schedule_status"] = "CONTROL_FLOW_EXCLUDED"

    result = dsa_measurement_cohort.construct_cohort([*good, *missing, *branch], minimum=1, maximum=4)

    assert result["verdict"] == "COHORT_FROZEN"
    assert [row["case_id"] for row in result["selected"]] == ["branch", "good"]
    assert {row["case_id"]: row["status"] for row in result["audit"]} == {
        "branch": "ELIGIBLE",
        "good": "ELIGIBLE",
        "missing": "INCOMPLETE_FOUR_ARM_FOUR_CAPACITY_MATRIX",
    }


def test_schedule_status_is_optional_analysis_metadata(tmp_path):
    preflight = tmp_path / "preflight.tsv"
    rows = _case_rows("case")
    for row in rows:
        row.pop("schedule_status")
    columns = [column for column in dsa_measurement_cohort.REQUIRED_COLUMNS if column != "schedule_status"]
    with preflight.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    result = dsa_measurement_cohort.freeze_cohort(preflight, tmp_path / "out", minimum=1)

    assert result["verdict"] == "COHORT_FROZEN"


def test_selection_balances_families_without_performance_data():
    rows = []
    rows.extend(_case_rows("a1", family="a", parent="p1"))
    rows.extend(_case_rows("a2", family="a", parent="p1"))
    rows.extend(_case_rows("a3", family="a", parent="p1"))
    rows.extend(_case_rows("b1", family="b", parent="p2"))

    result = dsa_measurement_cohort.construct_cohort(rows, minimum=3, maximum=3)

    assert result["verdict"] == "COHORT_FROZEN"
    assert {row["case_id"] for row in result["selected"]} == {"a1", "a2", "b1"}


def test_freezer_rejects_performance_and_objective_columns(tmp_path):
    preflight = tmp_path / "preflight.tsv"
    rows = _case_rows("case")
    for row in rows:
        row["latency_us"] = "12.3"
    _write_preflight(preflight, rows, extra_columns=("latency_us",))

    with pytest.raises(ValueError, match="forbidden"):
        dsa_measurement_cohort.freeze_cohort(preflight, tmp_path / "out", minimum=1)


def test_freezer_writes_content_addressed_manifest(tmp_path):
    preflight = tmp_path / "preflight.tsv"
    output = tmp_path / "out"
    _write_preflight(preflight, _case_rows("case"))

    result = dsa_measurement_cohort.freeze_cohort(preflight, output, minimum=1)

    assert result["verdict"] == "COHORT_FROZEN"
    assert (output / "cohort-frozen.tsv").is_file()
    assert result["cohort_sha256"] in (output / "cohort-frozen.tsv.sha256").read_text()
    summary = json.loads((output / "cohort-summary.json").read_text())
    assert summary["timing_blind"] is True
    assert summary["selected_count"] == 1


def test_freezer_does_not_claim_a_too_small_corpus(tmp_path):
    preflight = tmp_path / "preflight.tsv"
    output = tmp_path / "out"
    _write_preflight(preflight, _case_rows("case"))

    dsa_measurement_cohort.freeze_cohort(preflight, output, minimum=1)
    assert (output / "cohort-frozen.tsv").exists()

    result = dsa_measurement_cohort.freeze_cohort(preflight, output, minimum=2)

    assert result["verdict"] == "CORPUS_TOO_SMALL"
    assert not (output / "cohort-frozen.tsv").exists()
    assert dsa_measurement_cohort.main([str(preflight), str(output), "--minimum", "2"]) == 2
