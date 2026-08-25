# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for topology-preserving PTO address translation."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / ".claude"
    / "skills"
    / "incore-profiling"
    / "prepare_dsa_address_translation.py"
)


@pytest.fixture(scope="module")
def preparer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_prepare_dsa_address_translation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem() -> dict:
    return {
        "problem": {
            "buffers": [
                {"id": 0, "size": 64},
                {"id": 1, "size": 64},
                {"id": 2, "size": 64},
            ]
        }
    }


def _solution(offsets: list[int]) -> dict:
    return {
        "placements": [
            {"buffer": buffer_id, "pool": 1, "offset": offset} for buffer_id, offset in enumerate(offsets)
        ]
    }


_PTO = """module {
  %address_zero = arith.constant 0 : i64
  %address_sixty_four = arith.constant 64 : i64
  %unrelated = arith.constant 7 : i64
  %0 = pto.alloc_tile addr = %address_zero
  pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  %1 = pto.alloc_tile addr = %address_sixty_four
  pto.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
}
"""


def test_translates_address_constants_and_keeps_sync(preparer: ModuleType) -> None:
    output, report = preparer.prepare(_problem(), _solution([0, 0, 64]), _solution([64, 64, 0]), _PTO)

    assert "%address_zero = arith.constant 64 : i64" in output
    assert "%address_sixty_four = arith.constant 0 : i64" in output
    assert "%unrelated = arith.constant 7 : i64" in output
    assert report["overlap_geometry_identical"] is True
    assert report["sync_topology_identical"] is True
    assert report["address_group_map"] == {"0": 64, "64": 0}


def test_rejects_changed_overlap_geometry(preparer: ModuleType) -> None:
    with pytest.raises(ValueError, match="overlap geometry"):
        preparer.prepare(_problem(), _solution([0, 0, 64]), _solution([0, 64, 128]), _PTO)


def test_rejects_address_constant_with_semantic_use(preparer: ModuleType) -> None:
    pto = _PTO.replace(
        "  %0 = pto.alloc_tile addr = %address_zero\n",
        "  %0 = pto.alloc_tile addr = %address_zero\n  %x = arith.addi %address_zero, %address_zero\n",
    )

    with pytest.raises(ValueError, match="non-address uses"):
        preparer.prepare(_problem(), _solution([0, 0, 64]), _solution([64, 64, 0]), pto)
