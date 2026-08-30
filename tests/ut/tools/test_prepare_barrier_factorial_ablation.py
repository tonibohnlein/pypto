# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the placement x barrier factorial source preparer."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / ".claude"
    / "skills"
    / "incore-profiling"
    / "prepare_barrier_factorial_ablation.py"
)


@pytest.fixture(scope="module")
def preparer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_prepare_barrier_factorial_ablation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DISJOINT = """void kernel() {
  if (predicate) {
    TADD(out, lhs, rhs);
  } else {
    pipe_barrier(PIPE_V);
    TMOV(dst, src);
  }
}
"""
_OVERLAPPING = """void kernel() {
  if (predicate) {
    TADD(out, lhs, rhs);
  } else {
    TMOV(dst, src);
  }
}
"""


def test_prepares_exact_reversible_factorial(preparer: ModuleType) -> None:
    cells, report = preparer.prepare(
        _DISJOINT,
        _OVERLAPPING,
        disjoint_target_anchor="TMOV(dst, src);",
        overlapping_target_anchor="TMOV(dst, src);",
        barrier_statement="pipe_barrier(PIPE_V);",
    )

    assert cells["disjoint_barrier_present"] == _DISJOINT
    assert cells["overlapping_barrier_absent"] == _OVERLAPPING
    assert "pipe_barrier(PIPE_V);\n    TMOV" not in cells["disjoint_barrier_absent"]
    assert "pipe_barrier(PIPE_V);\n    TMOV" in cells["overlapping_barrier_present"]
    assert report["within_geometry_non_barrier_lines_identical"] is True
    assert report["exactly_reversible"] is True
    assert report["code_layout_control_required_after_device_disassembly"] is True


def test_rejects_missing_reference_barrier(preparer: ModuleType) -> None:
    with pytest.raises(ValueError, match="disjoint reference must contain"):
        preparer.prepare(
            _OVERLAPPING,
            _OVERLAPPING,
            disjoint_target_anchor="TMOV(dst, src);",
            overlapping_target_anchor="TMOV(dst, src);",
            barrier_statement="pipe_barrier(PIPE_V);",
        )


def test_rejects_ambiguous_target(preparer: ModuleType) -> None:
    with pytest.raises(ValueError, match="matched 2 lines"):
        preparer.prepare(
            _DISJOINT.replace("TMOV(dst, src);", "TMOV(dst, src);\n    TMOV(dst, src);"),
            _OVERLAPPING,
            disjoint_target_anchor="TMOV(dst, src);",
            overlapping_target_anchor="TMOV(dst, src);",
            barrier_statement="pipe_barrier(PIPE_V);",
        )
