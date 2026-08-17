# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "build_dsa_standalone_census.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_build_dsa_standalone_census", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
census = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = census
_SPEC.loader.exec_module(census)


def _invocation(
    script: str, *, penalties: int = 4, document_sha256: str = "shared-document"
) -> dict[str, str]:
    return {
        "script": script,
        "instance": "kernel",
        "problem_fingerprint": "abc123",
        "document_sha256": document_sha256,
        "document": f"captures/{script}/problem.json",
        "buffers": "30",
        "pools": "1",
        "pool_names": "Vec",
        "reuse_penalties": str(penalties),
        "recognizer": "quadratic",
    }


def _screen_rows(tag: str, *, infeasible: str | None = None) -> list[dict[str, str]]:
    rows = []
    for arm in census.ARMS:
        status = "no_fit" if arm == infeasible else "feasible"
        rows.append(
            {
                "tag": tag,
                "pool_id": "1",
                "pool": "Vec",
                "capacity_label": "native",
                "arm": arm,
                "status": status,
                "placement_sha256": "shared" if arm == "geometry_cg" else f"placement-{arm}",
            }
        )
    return rows


def _solutions(screen_root: Path, tag: str, *, skip: str | None = None) -> None:
    for arm in census.ARMS:
        if arm == skip:
            continue
        path = screen_root / "raw" / tag / "pool-1-Vec" / "native" / arm / "solution.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")


def test_census_preserves_invocations_and_requires_four_arms(tmp_path: Path) -> None:
    tag = "program__kernel-abc123"
    _solutions(tmp_path, tag)
    invocations = [_invocation("a.py"), _invocation("b.py"), _invocation("control.py", penalties=0)]
    unique = [
        {
            **_invocation("representative.py"),
            "corpus_document": f"corpus/penalty-bearing/{tag}.dsa.json",
        }
    ]

    rows, problems = census.build_census_rows(
        invocations,
        unique,
        _screen_rows(tag),
        tmp_path,
    )

    assert len(rows) == 2
    assert {row["source_script"] for row in rows} == {"a.py", "b.py"}
    assert all(row["host_status"] == "FOUR_ARM_FEASIBLE" for row in rows)
    assert all(row["measurement_mode"] == "NEEDS_CURRENT_DISPATCH_CAPTURE" for row in rows)
    assert all(row["parent_fallback"] == "FORBIDDEN" for row in rows)
    assert len(problems) == 1
    assert problems[0]["invocations"] == 2


def test_census_uses_one_semantic_representative_for_different_documents(tmp_path: Path) -> None:
    first_tag = "program_a__kernel-abc123"
    second_tag = "program_b__kernel-abc123"
    _solutions(tmp_path, first_tag)
    _solutions(tmp_path, second_tag)
    invocations = [
        _invocation("a.py", document_sha256="document-a"),
        _invocation("b.py", document_sha256="document-b"),
    ]
    unique = [
        {
            **_invocation("a.py", document_sha256="document-a"),
            "corpus_document": f"corpus/penalty-bearing/{first_tag}.dsa.json",
        },
        {
            **_invocation("b.py", document_sha256="document-b"),
            "corpus_document": f"corpus/penalty-bearing/{second_tag}.dsa.json",
        },
    ]

    rows, problems = census.build_census_rows(
        invocations,
        unique,
        _screen_rows(first_tag) + _screen_rows(second_tag),
        tmp_path,
    )

    assert len(rows) == 2
    assert len(problems) == 1
    assert problems[0]["document_sha256"] == "document-a"
    assert problems[0]["invocations"] == 2
    assert {row["problem_tag"] for row in rows} == {first_tag}


def test_census_marks_infeasible_arm_without_requiring_a_solution(tmp_path: Path) -> None:
    tag = "program__kernel-abc123"
    _solutions(tmp_path, tag, skip="cypress")
    unique = [
        {
            **_invocation("representative.py"),
            "corpus_document": f"corpus/penalty-bearing/{tag}.dsa.json",
        }
    ]

    rows, unused_problems = census.build_census_rows(
        [_invocation("a.py")],
        unique,
        _screen_rows(tag, infeasible="cypress"),
        tmp_path,
    )
    del unused_problems

    assert rows[0]["host_status"] == "ARM_NOT_FEASIBLE"
    assert rows[0]["infeasible_arms"] == "cypress"
    assert rows[0]["measurement_mode"] == "NOT_MEASURABLE_HOST"


def test_census_rejects_missing_feasible_solution(tmp_path: Path) -> None:
    tag = "program__kernel-abc123"
    _solutions(tmp_path, tag, skip="geometry_cg")
    unique = [
        {
            **_invocation("representative.py"),
            "corpus_document": f"corpus/penalty-bearing/{tag}.dsa.json",
        }
    ]
    with pytest.raises(FileNotFoundError, match="native solution is missing"):
        census.build_census_rows([_invocation("a.py")], unique, _screen_rows(tag), tmp_path)
