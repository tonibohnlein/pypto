# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for function-aware PTOAS sync-summary comparison."""

import json

import pytest
from pypto.tools import ptoas_sync_summary


def _record(function: str, groups: int, pair_groups: dict[str, int]) -> dict:
    return {
        "schema_version": 1,
        "status": "analyzed",
        "function": function,
        "active_sync_groups": groups,
        "active_sync_operations": groups * 2,
        "pipe_pair_groups": pair_groups,
        "pipe_pair_event_ids": {},
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_load_and_diff_match_functions_instead_of_row_order(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _record("mixed_aic", 10, {"PIPE_M->PIPE_MTE1": 2}),
            _record("mixed_aiv", 20, {"PIPE_V->PIPE_MTE2": 3}),
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _record("mixed_aiv", 18, {"PIPE_V->PIPE_MTE2": 1}),
            _record("mixed_aic", 11, {"PIPE_M->PIPE_MTE1": 3}),
        ],
    )

    delta = ptoas_sync_summary.diff_sync_summaries(
        ptoas_sync_summary.load_sync_summaries(baseline_path),
        ptoas_sync_summary.load_sync_summaries(candidate_path),
    )

    assert delta["mixed_aic"]["metrics"]["active_sync_groups"] == 1
    assert delta["mixed_aic"]["pipe_pair_groups"]["PIPE_M->PIPE_MTE1"] == 1
    assert delta["mixed_aiv"]["metrics"]["active_sync_groups"] == -2
    assert delta["mixed_aiv"]["pipe_pair_groups"]["PIPE_V->PIPE_MTE2"] == -2


def test_diff_rejects_function_set_mismatch():
    with pytest.raises(ValueError, match="missing=\\['aic'\\].*added=\\['other'\\]"):
        ptoas_sync_summary.diff_sync_summaries(
            {"aic": _record("aic", 1, {})},
            {"other": _record("other", 1, {})},
        )


def test_load_rejects_duplicate_function(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(path, [_record("aiv", 1, {}), _record("aiv", 2, {})])
    with pytest.raises(ValueError, match="duplicate function 'aiv'"):
        ptoas_sync_summary.load_sync_summaries(path)


def test_load_rejects_invalid_records(tmp_path):
    bad_json = tmp_path / "bad-json.jsonl"
    bad_json.write_text("{broken\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        ptoas_sync_summary.load_sync_summaries(bad_json)

    missing_function = tmp_path / "missing-function.jsonl"
    missing_function.write_text('{"schema_version": 1}\n')
    with pytest.raises(ValueError, match="missing non-empty 'function'"):
        ptoas_sync_summary.load_sync_summaries(missing_function)


def test_main_writes_function_keyed_json(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "diff.json"
    _write_jsonl(baseline, [_record("aiv", 3, {})])
    _write_jsonl(candidate, [_record("aiv", 2, {})])

    assert ptoas_sync_summary.main([str(baseline), str(candidate), "-o", str(output)]) == 0
    assert json.loads(output.read_text())["aiv"]["metrics"]["active_sync_groups"] == -1


def _lowered_loop_pto() -> str:
    return """
module {
  func.func @kernel() {
    // pto.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID7>] must not be counted.
    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    scf.for %i = 0 to 4 step 1 {
      pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      pto.barrier <PIPE_V>
      pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    }
    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    return
  }
}
"""


def test_summarize_lowered_pto_models_loop_entry_carried_and_exit_transitions():
    summary = ptoas_sync_summary.summarize_lowered_pto(_lowered_loop_pto())["kernel"]

    assert summary["summary_kind"] == "actual_post_insert_sync_lowered_ir_v1"
    assert summary["instruction_site_count"] == 5
    assert summary["instruction_sites_by_type"] == {
        "barrier": 1,
        "set_flag": 2,
        "wait_flag": 2,
    }
    assert summary["inside_loop_instruction_sites"] == 3
    assert summary["outside_loop_instruction_sites"] == 2
    assert summary["static_loop_trip_counts"] == {"6": 4}
    assert summary["static_execution_estimate_status"] == "COMPLETE"
    assert summary["static_estimated_instruction_executions"] == 14
    assert summary["static_estimated_executions_by_type"] == {
        "barrier": 4,
        "set_flag": 5,
        "wait_flag": 5,
    }
    assert summary["static_estimated_executions_by_pipe_pair"] == {"PIPE_MTE2->PIPE_V": 10}
    assert summary["static_estimated_barrier_executions_by_pipe"] == {"PIPE_V": 4}
    assert summary["inferred_transition_counts"] == {
        "loop_carried": 1,
        "loop_entry": 1,
        "loop_exit": 1,
        "same_scope_rearm": 1,
    }
    loop_carried = [
        transition
        for transition in summary["inferred_event_transitions"]
        if transition["kind"] == "loop_carried"
    ]
    assert loop_carried == [
        {
            "src_pipe": "PIPE_MTE2",
            "dst_pipe": "PIPE_V",
            "event": "EVENT_ID0",
            "from_line": 9,
            "to_line": 7,
            "from_type": "set_flag",
            "to_type": "wait_flag",
            "kind": "loop_carried",
            "basis": "inferred_loop_backedge",
            "loop": 6,
        }
    ]


def test_summarize_lowered_pto_marks_dynamic_loop_execution_estimate_incomplete():
    pto = _lowered_loop_pto().replace("0 to 4 step 1", "0 to %arg0 step 1")

    summary = ptoas_sync_summary.summarize_lowered_pto(pto)["kernel"]

    assert summary["static_loop_trip_counts"] == {"6": None}
    assert summary["static_execution_estimate_status"] == "INCOMPLETE_DYNAMIC_OR_UNRESOLVED_LOOP"
    assert summary["static_estimated_instruction_executions"] is None
    assert summary["static_estimated_executions_by_type"] is None


def test_summarize_lowered_pto_rejects_unlowered_event_ops():
    pto = """
module {
  func.func @kernel() {
    pto.record_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    return
  }
}
"""
    with pytest.raises(ValueError, match="high-level event op remains"):
        ptoas_sync_summary.summarize_lowered_pto(pto)


def test_arm_manifest_requires_and_summarizes_every_declared_arm(tmp_path):
    pto = tmp_path / "kernel.pto"
    pto.write_text(_lowered_loop_pto())
    manifest = tmp_path / "arms.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_arms": ["geometry_ff", "dsa_rp_cg"],
                "cells": [
                    {
                        "case": "rms_norm",
                        "capacity": "tight",
                        "arm": arm,
                        "post_insert_sync_pto": pto.name,
                        "function": "kernel",
                    }
                    for arm in ("geometry_ff", "dsa_rp_cg")
                ],
            }
        )
    )

    result = ptoas_sync_summary.summarize_arm_manifest(manifest)

    assert result["summary_kind"] == "actual_post_insert_sync_per_arm_v1"
    assert result["cell_count"] == 2
    assert result["function_summary_count"] == 2
    assert {row["arm"] for row in result["cells"]} == {"geometry_ff", "dsa_rp_cg"}


def test_arm_manifest_rejects_incomplete_arm_coverage(tmp_path):
    pto = tmp_path / "kernel.pto"
    pto.write_text(_lowered_loop_pto())
    manifest = tmp_path / "arms.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_arms": ["geometry_ff", "dsa_rp_cg"],
                "cells": [
                    {
                        "case": "rms_norm",
                        "capacity": "tight",
                        "arm": "geometry_ff",
                        "post_insert_sync_pto": pto.name,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="arm coverage mismatch"):
        ptoas_sync_summary.summarize_arm_manifest(manifest)


def test_arm_manifest_rejects_different_function_sets(tmp_path):
    pto = tmp_path / "kernels.pto"
    pto.write_text(
        _lowered_loop_pto()
        + """
module {
  func.func @other() {
    pto.barrier <PIPE_ALL>
    return
  }
}
"""
    )
    manifest = tmp_path / "arms.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_arms": ["geometry_ff", "dsa_rp_cg"],
                "cells": [
                    {
                        "case": "rms_norm",
                        "capacity": "tight",
                        "arm": "geometry_ff",
                        "post_insert_sync_pto": pto.name,
                        "function": "kernel",
                    },
                    {
                        "case": "rms_norm",
                        "capacity": "tight",
                        "arm": "dsa_rp_cg",
                        "post_insert_sync_pto": pto.name,
                        "function": "other",
                    },
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="function-set mismatch"):
        ptoas_sync_summary.summarize_arm_manifest(manifest)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
