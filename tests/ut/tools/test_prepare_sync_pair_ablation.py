# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for sync-only PTO ablation preparation."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "prepare_sync_pair_ablation.py"
)


@pytest.fixture(scope="module")
def preparer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_prepare_sync_pair_ablation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = """func.func @sample() {
  pto.compute_a
  pto.compute_b
  return
}
"""
_REFERENCE = """func.func @sample() {
  pto.compute_a
  pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  pto.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  pto.compute_b
  return
}
"""


def test_restores_only_the_reference_proven_pair(preparer: ModuleType) -> None:
    output, report = preparer.prepare(
        _BASE,
        _REFERENCE,
        set_after="pto.compute_a",
        wait_before="pto.compute_b",
        source_pipe="PIPE_V",
        destination_pipe="PIPE_MTE2",
        event_id=3,
    )

    assert output == _REFERENCE
    assert report["reference_pair_proof"] is True
    assert report["non_sync_lines_identical_to_base"] is True
    assert len(report["added_lines"]) == 2


def test_rejects_reference_without_the_pair(preparer: ModuleType) -> None:
    with pytest.raises(ValueError, match="reference does not contain"):
        preparer.prepare(
            _BASE,
            _BASE,
            set_after="pto.compute_a",
            wait_before="pto.compute_b",
            source_pipe="PIPE_V",
            destination_pipe="PIPE_MTE2",
            event_id=3,
        )


def test_rejects_ambiguous_anchor(preparer: ModuleType) -> None:
    with pytest.raises(ValueError, match="matched 2 lines"):
        preparer.prepare(
            _BASE.replace("pto.compute_b", "pto.compute_a"),
            _REFERENCE,
            set_after="pto.compute_a",
            wait_before="return",
            source_pipe="PIPE_V",
            destination_pipe="PIPE_MTE2",
            event_id=3,
        )
