# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling"
sys.path.insert(0, str(_TOOLS))
_SCRIPT = _TOOLS / "freeze_dsa_opportunity_holdout.py"
_SPEC = importlib.util.spec_from_file_location("_test_freeze_dsa_opportunity_holdout", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
freezer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(freezer)


def _row(script: str, fingerprint: str, gap: int = 4) -> dict:
    return {
        "script": script,
        "measurement_unit": "SINGLE_KERNEL_DRIVER",
        "dsa_instances": "1",
        "problem_fingerprints": f"kernel={fingerprint}",
        "selection_status": "OPPORTUNITY_PRIMARY",
        "evaluation_capacity": "half",
        "cypress_minus_dsa_rp_reuse_cost": gap,
        "penalized_relation_disagreement": gap + 1,
        "reuse_relation_disagreement": gap + 2,
    }


def _write_freeze(path: Path, rows: list[dict], **extra: object) -> None:
    path.write_text(
        json.dumps(
            {
                "selection_policy": "cypress_dsa_rp_penalty_opportunity_v1",
                "uses_device_latency": False,
                "workloads": rows,
                **extra,
            }
        )
    )


def test_freezes_diverse_new_workloads_without_development_overlap(tmp_path: Path) -> None:
    development = tmp_path / "development.json"
    opportunity = tmp_path / "opportunity.json"
    _write_freeze(development, [_row("models/family_a/old.py", "old")])
    rows = [_row("models/family_a/old.py", "new-script-problem")]
    rows.append(_row("models/family_b/renamed.py", "old"))
    rows.extend(
        _row(f"models/family_{index % 3}/class_{index}.py", f"new-{index}", gap=20 - index)
        for index in range(10)
    )
    _write_freeze(opportunity, rows)

    result = freezer.freeze_holdout(opportunity, development, tmp_path / "out", minimum=8, maximum=8)

    assert result["selected_workload_count"] == 8
    assert result["prospective_holdout"] is True
    assert result["device_timing_state"] == "SEALED_UNSEEN"
    assert not any(row["script"].endswith(("old.py", "renamed.py")) for row in result["workloads"])
    assert {row["selection_rank"] for row in result["workloads"]} == set(range(1, 9))
    assert len(result["source_classes"]) == 8


def test_rejects_performance_fields_before_selecting(tmp_path: Path) -> None:
    development = tmp_path / "development.json"
    opportunity = tmp_path / "opportunity.json"
    _write_freeze(development, [_row("models/family_a/old.py", "old")])
    _write_freeze(
        opportunity,
        [_row(f"models/family_a/new_{index}.py", f"new-{index}") for index in range(8)],
        median_us=12.0,
    )

    with pytest.raises(ValueError, match="performance field"):
        freezer.freeze_holdout(opportunity, development, tmp_path / "out")


def test_fails_closed_below_minimum(tmp_path: Path) -> None:
    development = tmp_path / "development.json"
    opportunity = tmp_path / "opportunity.json"
    _write_freeze(development, [_row("models/family_a/old.py", "old")])
    _write_freeze(
        opportunity,
        [_row(f"models/family_a/new_{index}.py", f"new-{index}") for index in range(7)],
    )

    with pytest.raises(ValueError, match="only 7 eligible workloads"):
        freezer.freeze_holdout(opportunity, development, tmp_path / "out")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
