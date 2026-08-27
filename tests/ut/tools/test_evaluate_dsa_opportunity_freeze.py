# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling"
sys.path.insert(0, str(_TOOLS))
_SCRIPT = _TOOLS / "evaluate_dsa_opportunity_freeze.py"
_SPEC = importlib.util.spec_from_file_location("_test_evaluate_dsa_opportunity_freeze", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluator)


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_development_join_preserves_selection_and_classifies_direction(tmp_path: Path) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "selection_policy": "cypress_dsa_rp_penalty_opportunity_v1",
                "uses_device_latency": False,
                "workloads": [
                    {
                        "script": "models/win.py",
                        "measurement_unit": "SINGLE_KERNEL_DRIVER",
                        "evaluation_capacity": "half",
                        "selection_status": "OPPORTUNITY_PRIMARY",
                        "cypress_minus_dsa_rp_reuse_cost": 8,
                        "penalized_relation_disagreement": 4,
                        "reuse_relation_disagreement": 7,
                    },
                    {
                        "script": "models/control.py",
                        "measurement_unit": "SINGLE_KERNEL_DRIVER",
                        "evaluation_capacity": "tight",
                        "selection_status": "NULL_CONTROL_GEOMETRY_CYPRESS_DSA_RP_NOT_DISTINCT",
                        "cypress_minus_dsa_rp_reuse_cost": 0,
                        "penalized_relation_disagreement": 0,
                        "reuse_relation_disagreement": 0,
                    },
                ],
            }
        )
    )
    timing_path = tmp_path / "pairwise.tsv"
    _write_tsv(
        timing_path,
        [
            {
                "script": "models/win.py",
                "capacity": "half",
                "comparison": "cypress vs geometry_ff",
                "A_percent_change": "-8.0",
                "B_percent_change": "-7.0",
                "verdict": "CONFIRMED",
            },
            {
                "script": "models/win.py",
                "capacity": "half",
                "comparison": "dsa_rp_cg vs geometry_ff",
                "A_percent_change": "-12.0",
                "B_percent_change": "-11.0",
                "verdict": "CONFIRMED",
            },
            {
                "script": "models/win.py",
                "capacity": "half",
                "comparison": "dsa_rp_cg vs cypress",
                "A_percent_change": "-5.0",
                "B_percent_change": "-4.0",
                "verdict": "CONFIRMED",
            },
            {
                "script": "models/control.py",
                "capacity": "tight",
                "comparison": "cypress vs geometry_ff",
                "A_percent_change": "0.0",
                "B_percent_change": "0.0",
                "verdict": "PHYSICAL_NULL",
            },
            {
                "script": "models/control.py",
                "capacity": "tight",
                "comparison": "dsa_rp_cg vs geometry_ff",
                "A_percent_change": "0.0",
                "B_percent_change": "0.0",
                "verdict": "PHYSICAL_NULL",
            },
            {
                "script": "models/control.py",
                "capacity": "tight",
                "comparison": "dsa_rp_cg vs cypress",
                "A_percent_change": "0.0",
                "B_percent_change": "0.0",
                "verdict": "PHYSICAL_NULL",
            },
        ],
    )

    summary = evaluator.evaluate(freeze_path, timing_path, tmp_path / "out")
    rows = list(csv.DictReader((tmp_path / "out/development-evaluation.tsv").open(), delimiter="\t"))

    assert summary["prospective_evidence"] is False
    assert summary["objective_order_agrees"] == 1
    assert summary["objective_order_disagrees"] == 0
    assert rows[0]["capacity"] == "half"
    assert rows[0]["development_assessment"] == "OBJECTIVE_ORDER_AGREES"
    assert rows[0]["full_geometry_cypress_dsa_rp_order"] == "YES"
    assert rows[1]["development_assessment"] == "NULL_CONTROL"


def test_join_rejects_a_freeze_that_used_device_latency(tmp_path: Path) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "selection_policy": "cypress_dsa_rp_penalty_opportunity_v1",
                "uses_device_latency": True,
                "workloads": [],
            }
        )
    )

    with pytest.raises(ValueError, match="not timing-blind"):
        evaluator.evaluate(freeze_path, tmp_path / "unused.tsv", tmp_path / "out")
