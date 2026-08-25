# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for physical DSA layout analysis."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "analyze_dsa_physical_layout.py"
)


@pytest.fixture(scope="module")
def analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_analyze_dsa_physical_layout", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem() -> dict:
    return {
        "schema_version": 1,
        "instance": "sample",
        "problem": {
            "buffers": [{"id": buffer_id, "name": f"b{buffer_id}", "size": 64} for buffer_id in range(4)],
            "pools": [{"id": 1, "name": "Vec", "capacity": 256}],
            "constraints": {},
            "cost_model": {
                "reuse_penalties": [
                    {"first": 0, "second": 2, "cost": 3, "reason": "cross_pipe"},
                    {"first": 1, "second": 2, "cost": 5, "reason": "cross_pipe"},
                ]
            },
        },
    }


def _solution() -> dict:
    return {
        "placements": [
            {"buffer": 0, "pool": 1, "offset": 0},
            {"buffer": 1, "pool": 1, "offset": 0},
            {"buffer": 2, "pool": 1, "offset": 0},
            {"buffer": 3, "pool": 1, "offset": 64},
        ]
    }


def _schedule() -> dict:
    tile = {
        "known_physical_addresses": True,
        "scope": "VEC",
        "base_addresses": [0],
    }
    gm = {
        "known_physical_addresses": False,
        "scope": "GM",
        "base_addresses": [0],
    }
    return {
        "nodes": [
            {
                "id": 0,
                "kind": "operation",
                "op_name": "pto.tload",
                "pipe": "PIPE_MTE2",
                "operation": {"pypto_access_order": 1},
                "uses": [gm],
                "defs": [tile],
            },
            {
                "id": 1,
                "kind": "operation",
                "op_name": "pto.tadd",
                "pipe": "PIPE_V",
                "operation": {"pypto_access_order": 2},
                "uses": [tile],
                "defs": [],
            },
        ]
    }


def test_logical_edges_collapse_to_one_physical_group(analyzer: ModuleType) -> None:
    report = analyzer.analyze(_problem(), {"arm": _solution()}, {"arm": _schedule()}, [128])
    arm = report["arms"]["arm"]

    assert arm["logical_reuse_cost"] == 8
    assert len(arm["active_logical_reuse_edges"]) == 2
    assert arm["canonical_reuse_group_count"] == 1
    assert arm["canonical_physical_reuse_groups"][0]["logical_edge_count"] == 2
    assert arm["canonical_physical_reuse_groups"][0]["logical_cost"] == 8
    assert arm["physical_components"][0]["buffers"] == [0, 1, 2]
    assert len(arm["operation_accesses"]) == 2


def test_interleave_metrics_are_explicit_hypotheses(analyzer: ModuleType) -> None:
    report = analyzer.analyze(_problem(), {"arm": _solution()}, {"arm": _schedule()}, [128])
    sensitivity = report["arms"]["arm"]["interleave_sensitivity"][0]

    assert report["interleave_interpretation"] == "hypothesis_only_no_hardware_bank_mapping_assumed"
    assert sensitivity["period_bytes"] == 128
    assert sensitivity["unique_group_residues"] == 2
    assert sensitivity["group_collision_pairs"] == 0
    assert sensitivity["same_residue_access_transitions"] == 1
