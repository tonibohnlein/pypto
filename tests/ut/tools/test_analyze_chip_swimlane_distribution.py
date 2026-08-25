# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for chip-swimlane distribution analysis."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / ".claude"
    / "skills"
    / "incore-profiling"
    / "analyze_chip_swimlane_distribution.py"
)


@pytest.fixture(scope="module")
def analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_analyze_chip_swimlane_distribution", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records() -> dict:
    token = (1 << 32) | 7
    return {
        "metadata": {
            "clock_freq_hz": 1_000_000,
            "core_types": ["aiv", "aiv"],
        },
        "aicore_tasks": [
            [0, token, 1, 0, 10, 0],
            [1, token, 1, 2, 14, 0],
            [0, token, 2, 20, 26, 0],
            [1, token, 2, 18, 26, 0],
        ],
    }


def test_reports_slice_core_and_wave_distributions(analyzer: ModuleType) -> None:
    report = analyzer.analyze_trace(_records())

    assert report["task_tokens"] == [[1, 7]]
    assert report["task_window_us"] == pytest.approx(26.0)
    assert report["slice_count"] == 4
    assert report["core_count"] == 2
    assert report["slice_duration_us"]["median"] == pytest.approx(9.0)
    assert report["core_busy_us"]["min"] == pytest.approx(16.0)
    assert report["core_busy_us"]["max"] == pytest.approx(20.0)
    assert report["core_idle_gap_us"]["min"] == pytest.approx(4.0)
    assert report["core_idle_gap_us"]["max"] == pytest.approx(10.0)
    assert report["wave_duration_us"]["0"]["count"] == 2
    assert report["wave_duration_us"]["1"]["count"] == 2


def test_requires_task_selection_for_multiple_tokens(analyzer: ModuleType) -> None:
    records = _records()
    records["aicore_tasks"].append([0, (1 << 32) | 8, 3, 30, 31, 0])

    with pytest.raises(ValueError, match="task tokens"):
        analyzer.analyze_trace(records)
    selected = analyzer.analyze_trace(records, task_id=7)
    assert selected["slice_count"] == 4


def test_groups_slash_named_captures(analyzer: ModuleType) -> None:
    report = analyzer.analyze_traces(
        {"device6/cypress/capture1": _records(), "device6/cypress/capture2": _records()}
    )

    group = report["groups"]["device6/cypress"]
    assert group["capture_count"] == 2
    assert group["task_window_us"]["count"] == 2
    assert group["slice_duration_us"]["count"] == 8
    assert group["core_busy_us"]["count"] == 4
    assert group["registration_wave_duration_us"]["1"]["count"] == 4
    assert group["registration_wave_duration_us"]["2"]["count"] == 4
