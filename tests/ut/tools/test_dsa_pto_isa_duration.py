# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the pinned PTO-ISA duration provider."""

import json
import math
from pathlib import Path

import pytest
from pypto.tools import dsa_pto_isa_duration, dsa_schedule_model

_REVISION = "a" * 40
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _provider(*, policy: str = "error") -> dsa_pto_isa_duration.PtoIsaDurationProvider:
    return dsa_pto_isa_duration.PtoIsaDurationProvider(
        revision=_REVISION,
        frequency_hz=1.85e9,
        bandwidth_gib_per_s={
            "GM_TO_UB": 100.9,
            "GM_TO_L1": 135.0,
            "UB_TO_GM": 188.46,
            "L1_TO_GM": 32.0,
            "UB_TO_UB": 1024.0,
            "L0C_TO_GM": 70.0,
            "L0C_TO_L1": 128.0,
            "L1_TO_L0A": 441.0,
            "L1_TO_L0B": 220.5,
            "L1_TO_BT": 32.0,
            "L1_TO_FB": 32.0,
        },
        formula_parameters=[
            dsa_pto_isa_duration.FormulaParameter("TMUL", "fp32", 128, 0.0325, 17.5),
            dsa_pto_isa_duration.FormulaParameter("TEXP", "any", None, 0.0314, 30.1),
        ],
        source_sha256={"formula_params.csv": "b" * 64},
        unsupported_policy=policy,
        fallback_cycles=7.0,
    )


def _node(op_name: str, pipe: str, *operand_types: str) -> dict:
    return {
        "id": 0,
        "kind": "operation",
        "op_name": op_name,
        "pipe": pipe,
        "loop_stack": [],
        "defs": [{"allocate_size_bytes": 4096}],
        "uses": [{"allocate_size_bytes": 4096}],
        "operation": {"operand_types": list(operand_types), "result_types": [], "attributes": {}},
    }


def test_formula_and_any_dtype_lookup_match_pto_isa_rounding():
    provider = _provider()
    mul = provider.estimate(
        _node("pto.tmul", "PIPE_V", "!pto.tile_buf<vec, 8x128xf32, valid=?x?>"),
        work_bytes=4096,
    )
    exp = provider.estimate(
        _node("pto.texp", "PIPE_V", "!pto.tile_buf<vec, 1x4096xf16, valid=?x?>"),
        work_bytes=8192,
    )

    assert mul.cycles == 51
    assert mul.source == "pto_isa_formula"
    assert exp.cycles == 159


def test_pipeline_estimate_uses_only_explicit_splits_and_fails_closed_on_fitted_totals():
    provider = _provider()
    vector = _node("pto.tmul", "PIPE_V", "!pto.tile_buf<vec, 8x128xf32, valid=?x?>")
    transfer = _node(
        "pto.tload",
        "PIPE_MTE2",
        "!pto.partition_tensor_view<1x1024xf32>",
        "!pto.tile_buf<vec, 1x1024xf32, valid=?x?>",
    )
    matmul = _node(
        "pto.tmatmul",
        "PIPE_M",
        "!pto.tile_buf<left, 16x512xf32, valid=?x?>",
        "!pto.tile_buf<right, 512x16xf32, valid=?x?>",
    )

    estimate = provider.estimate_pipeline(matmul)

    assert estimate is not None
    assert estimate.startup_cycles == 6.0
    assert estimate.pending_tail_cycles == 0.0
    assert estimate.source == "pto_isa_matmul_head"
    assert provider.estimate_pipeline(vector) is None
    assert provider.estimate_pipeline(transfer) is None


def test_transfer_converts_gib_per_second_to_cycles():
    provider = _provider()
    estimate = provider.estimate(
        _node(
            "pto.tload",
            "PIPE_MTE2",
            "!pto.partition_tensor_view<1x1024xf32>",
            "!pto.tile_buf<vec, 1x1024xf32, valid=?x?>",
        ),
        work_bytes=4096,
    )

    expected = math.floor((4096 / (1024**3)) / 100.9 * 1.85e9)
    assert estimate.cycles == expected
    assert estimate.cycles == 69
    assert estimate.source == "pto_isa_bandwidth"


def test_transfer_accepts_tile_type_in_result_metadata():
    provider = _provider()
    node = _node("pto.tload", "PIPE_MTE2", "!pto.partition_tensor_view<8x128xf32>")
    node["operation"]["result_types"] = [
        "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=?, v_col=?>"
    ]

    estimate = provider.estimate(node, work_bytes=0)

    assert estimate.cycles == 69
    assert estimate.source == "pto_isa_bandwidth"


def test_static_type_size_supports_keyed_tiles_and_partition_views():
    assert (
        dsa_pto_isa_duration.static_type_size_bytes(
            "!pto.tile_buf<loc=vec, dtype=bf16, rows=8, cols=128, v_row=?, v_col=?>"
        )
        == 2048
    )
    assert dsa_pto_isa_duration.static_type_size_bytes("!pto.partition_tensor_view<8x128xf32>") == 4096
    assert dsa_pto_isa_duration.static_type_size_bytes("f32") is None


def test_matmul_uses_pto_isa_tile_formula():
    estimate = _provider().estimate(
        _node(
            "pto.tmatmul",
            "PIPE_M",
            "!pto.tile_buf<left, 16x512xf32, valid=?x?>",
            "!pto.tile_buf<right, 512x16xf32, valid=?x?>",
            "!pto.tile_buf<acc, 16x16xf32, valid=?x?>",
        ),
        work_bytes=32768,
    )

    assert estimate.cycles == 134
    assert estimate.source == "pto_isa_matmul_formula"


def test_matmul_formula_accepts_a2a3_bfloat16_inputs():
    estimate = _provider().estimate(
        _node(
            "pto.tmatmul",
            "PIPE_M",
            "!pto.tile_buf<left, 16x256xbf16, valid=?x?>",
            "!pto.tile_buf<right, 256x64xbf16, valid=?x?>",
        ),
        work_bytes=8192,
    )

    assert estimate.cycles == 70
    assert estimate.source == "pto_isa_matmul_formula"


def test_perf_sim_default_is_pinned_nonfallback_and_uses_result_work_shape():
    node = _node(
        "pto.textract",
        "PIPE_MTE1",
        "!pto.tile_buf<mat, 16x512xbf16, valid=?x?>",
        "index",
        "index",
    )
    node["operation"]["result_types"] = ["!pto.tile_buf<left, 16x128xbf16, valid=?x?>"]

    estimate = _provider().estimate(node, work_bytes=4096)

    assert estimate.cycles == 66
    assert estimate.source == "pto_isa_perf_sim_default"
    assert estimate.fallback is False


def test_missing_formula_signature_uses_pinned_perf_sim_default():
    node = _node(
        "pto.trowsum",
        "PIPE_V",
        "!pto.tile_buf<vec, 8x512xf32, valid=?x?>",
        "!pto.tile_buf<vec, 8x512xf32, valid=?x?>",
    )
    node["operation"]["result_types"] = ["!pto.tile_buf<vec, 8x1xf32, valid=?x?>"]

    estimate = _provider().estimate(node, work_bytes=16384)

    assert estimate.cycles == 130
    assert estimate.source == "pto_isa_perf_sim_default"
    assert estimate.fallback is False


def test_ttrans_uses_pinned_mte1_default_rule():
    node = _node(
        "pto.ttrans",
        "PIPE_V",
        "!pto.tile_buf<vec, 16x8xf32, valid=?x?>",
        "!pto.tile_buf<vec, 16x8xf32, valid=?x?>",
    )
    node["operation"]["result_types"] = ["!pto.tile_buf<vec, 8x16xf32, valid=?x?>"]

    estimate = _provider().estimate(node, work_bytes=512)

    assert estimate.cycles == 3
    assert estimate.source == "pto_isa_perf_sim_default"


@pytest.mark.parametrize(
    ("node", "expected_source"),
    [
        (
            {
                **_node("pto.load_scalar", "PIPE_S", "!pto.ptr<f32>"),
                "operation": {
                    "operand_types": ["!pto.ptr<f32>"],
                    "result_types": ["f32"],
                    "attributes": {},
                },
            },
            "pto_isa_perf_sim_scalar_stage",
        ),
        (_node("pto.tpush", "PIPE_FIX"), "pto_isa_perf_sim_scalar_stage"),
        (_node("pto.tpop", "PIPE_MTE2"), "pto_isa_perf_sim_scalar_stage"),
    ],
)
def test_scalar_stage_operations_use_pinned_perf_sim_contract(node, expected_source):
    estimate = _provider().estimate(node, work_bytes=0)

    assert estimate.cycles == 1
    assert estimate.source == expected_source
    assert estimate.fallback is False


def test_scalar_load_fails_closed_without_pointer_and_scalar_result():
    with pytest.raises(ValueError, match="scalar load lacks one pointer and one scalar result"):
        _provider().estimate(_node("pto.load_scalar", "PIPE_S", "i32"), work_bytes=0)


def test_unsupported_operation_fails_closed_or_is_explicit_fallback():
    node = _node("pto.trsqrt", "PIPE_V", "!pto.tile_buf<vec, 1x8xf32>")
    with pytest.raises(ValueError, match="unsupported PTO-ISA duration for pto.trsqrt"):
        _provider().estimate(node, work_bytes=32)

    estimate = _provider(policy="fallback").estimate(node, work_bytes=32)
    assert estimate.cycles == 7
    assert estimate.source == "unsupported_fallback"
    assert estimate.fallback is True


def test_provider_snapshot_round_trip_is_portable():
    provider = _provider(policy="fallback")
    restored = dsa_pto_isa_duration.PtoIsaDurationProvider.from_json(provider.to_json())

    assert restored == provider
    assert dsa_pto_isa_duration.provider_snapshot_sha256(restored) == (
        dsa_pto_isa_duration.provider_snapshot_sha256(provider)
    )


def test_checked_in_integer_trowmax_calibration_is_exact_signature_only():
    calibration = _REPO_ROOT / ".claude/skills/incore-profiling/dsa_pto_isa_exact_durations_v1.json"
    payload = json.loads(calibration.read_text())
    record = payload["instructions"]["TROWMAX"][0]

    model = dsa_schedule_model.calibrate_from_metrics([calibration], dsa_schedule_model.DurationModel())
    signature_key = dsa_schedule_model._operation_signature_key(record["operation_signature"])

    assert record["cycles"] == 435.0
    assert model.operation_signature_cycles == {signature_key: 435.0}
    assert model.calibration_sources == [str(calibration)]


def test_snapshot_duration_command_writes_portable_model(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "snapshot.json"
    model = dsa_schedule_model.DurationModel(
        calibration_status="pto_isa_pinned",
        pto_isa_provider=_provider(policy="fallback"),
    )
    source.write_text(json.dumps(model.to_json()))

    assert dsa_schedule_model.main(["snapshot-duration", "--model", str(source), "-o", str(output)]) == 0
    restored = dsa_schedule_model.DurationModel.from_json(json.loads(output.read_text()))
    assert restored.to_json() == model.to_json()


def test_schedule_model_reports_exact_and_fallback_coverage():
    model = dsa_schedule_model.DurationModel(
        calibration_status="pto_isa_pinned",
        pto_isa_provider=_provider(policy="fallback"),
    )
    record = {
        "schema_version": 1,
        "function": "kernel",
        "status": "analyzed",
        "nodes": [
            _node("pto.tmul", "PIPE_V", "!pto.tile_buf<vec, 8x128xf32, valid=?x?>"),
            {**_node("pto.trsqrt", "PIPE_V", "!pto.tile_buf<vec, 1x8xf32>"), "id": 1},
        ],
        "stream_edges": [{"source": 0, "target": 1, "pipe": "PIPE_V"}],
        "sync_edges": [],
    }

    result = dsa_schedule_model.score_schedule(record, model)

    assert result["exact_duration_nodes"] == 1
    assert result["fallback_duration_nodes"] == 1
    assert result["exact_duration_coverage"] == 0.5
    assert result["duration_source_counts"] == {
        "pto_isa_formula": 1,
        "unsupported_fallback": 1,
    }
    assert result["pto_isa_provider"]["revision"] == _REVISION


def test_perf_sim_validation_reports_formula_error(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {"ph": "X", "name": "TMUL(8x128,fp32):0", "dur": 49},
                {"ph": "X", "name": "UNSUPPORTED(1x8,fp32):1", "dur": 2},
            ]
        )
    )

    result = dsa_schedule_model.validate_pto_isa_formulas_against_perf_sim([trace], _provider())

    assert result["event_count"] == 1
    assert result["operation_count"] == 1
    assert result["validation_scope"] == "lightweight_formula_vs_perf_sim_effective_events"
    assert result["events"][0]["predicted_cycles"] == 51
    assert result["events"][0]["error_cycles"] == 2
    assert result["by_operation"]["TMUL"] == {
        "event_count": 1,
        "mean_predicted_cycles": 51.0,
        "mean_perf_sim_cycles": 49.0,
        "mean_error_cycles": 2.0,
        "mean_absolute_error_cycles": 2.0,
        "median_absolute_error_cycles": 2.0,
        "mean_absolute_percentage_error": 2 / 49,
        "median_absolute_percentage_error": 2 / 49,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
