# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare a disjoint/overlap x barrier-absent/present source factorial."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _unique_index(lines: list[str], anchor: str, *, source: str) -> int:
    matches = [index for index, line in enumerate(lines) if anchor in line]
    if len(matches) != 1:
        raise ValueError(f"{source} anchor {anchor!r} matched {len(matches)} lines")
    return matches[0]


def _preceding_nonblank(lines: list[str], index: int) -> int | None:
    for candidate in range(index - 1, -1, -1):
        if lines[candidate].strip():
            return candidate
    return None


def _toggle_barrier(
    text: str,
    *,
    target_anchor: str,
    barrier_statement: str,
    present: bool,
    source: str,
) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    target = _unique_index(lines, target_anchor, source=source)
    predecessor = _preceding_nonblank(lines, target)
    has_barrier = predecessor is not None and lines[predecessor].strip() == barrier_statement
    if has_barrier == present:
        return text, {
            "source": source,
            "changed": False,
            "target_line": target + 1,
            "barrier_line": predecessor + 1 if predecessor is not None else None,
        }
    output = list(lines)
    if present:
        indent = lines[target][: len(lines[target]) - len(lines[target].lstrip())]
        output.insert(target, indent + barrier_statement)
        barrier_line = target + 1
    else:
        assert predecessor is not None
        del output[predecessor]
        barrier_line = predecessor + 1
    rendered = "\n".join(output) + ("\n" if text.endswith("\n") else "")
    return rendered, {
        "source": source,
        "changed": True,
        "target_line": target + 1,
        "barrier_line": barrier_line,
    }


def prepare(
    disjoint_text: str,
    overlapping_text: str,
    *,
    disjoint_target_anchor: str,
    overlapping_target_anchor: str,
    barrier_statement: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the four cells while changing one source line per geometry."""
    disjoint_present, disjoint_present_report = _toggle_barrier(
        disjoint_text,
        target_anchor=disjoint_target_anchor,
        barrier_statement=barrier_statement,
        present=True,
        source="disjoint",
    )
    if disjoint_present != disjoint_text:
        raise ValueError("disjoint reference must contain the target barrier")
    overlapping_absent, overlapping_absent_report = _toggle_barrier(
        overlapping_text,
        target_anchor=overlapping_target_anchor,
        barrier_statement=barrier_statement,
        present=False,
        source="overlapping",
    )
    if overlapping_absent != overlapping_text:
        raise ValueError("overlapping reference must omit the target barrier")
    disjoint_absent, disjoint_absent_report = _toggle_barrier(
        disjoint_text,
        target_anchor=disjoint_target_anchor,
        barrier_statement=barrier_statement,
        present=False,
        source="disjoint",
    )
    overlapping_present, overlapping_present_report = _toggle_barrier(
        overlapping_text,
        target_anchor=overlapping_target_anchor,
        barrier_statement=barrier_statement,
        present=True,
        source="overlapping",
    )

    recovered_disjoint, _ = _toggle_barrier(
        disjoint_absent,
        target_anchor=disjoint_target_anchor,
        barrier_statement=barrier_statement,
        present=True,
        source="disjoint recovery",
    )
    recovered_overlapping, _ = _toggle_barrier(
        overlapping_present,
        target_anchor=overlapping_target_anchor,
        barrier_statement=barrier_statement,
        present=False,
        source="overlapping recovery",
    )
    if recovered_disjoint != disjoint_text or recovered_overlapping != overlapping_text:
        raise ValueError("barrier toggle is not exactly reversible")

    cells = {
        "disjoint_barrier_present": disjoint_present,
        "disjoint_barrier_absent": disjoint_absent,
        "overlapping_barrier_present": overlapping_present,
        "overlapping_barrier_absent": overlapping_absent,
    }
    report = {
        "schema_version": 1,
        "experiment": "placement_by_barrier_factorial_v1",
        "target_anchors": {
            "disjoint": disjoint_target_anchor,
            "overlapping": overlapping_target_anchor,
        },
        "barrier_statement": barrier_statement,
        "reference_cells": {
            "disjoint_barrier_present": True,
            "overlapping_barrier_absent": True,
        },
        "source_sha256": {
            "disjoint": _sha256(disjoint_text),
            "overlapping": _sha256(overlapping_text),
        },
        "cell_sha256": {name: _sha256(text) for name, text in sorted(cells.items())},
        "toggle_reports": {
            "disjoint_barrier_present": disjoint_present_report,
            "disjoint_barrier_absent": disjoint_absent_report,
            "overlapping_barrier_present": overlapping_present_report,
            "overlapping_barrier_absent": overlapping_absent_report,
        },
        "within_geometry_non_barrier_lines_identical": True,
        "exactly_reversible": True,
        "code_layout_control_required_after_device_disassembly": True,
    }
    return cells, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disjoint", type=Path, required=True)
    parser.add_argument("--overlapping", type=Path, required=True)
    parser.add_argument("--disjoint-target-anchor", required=True)
    parser.add_argument("--overlapping-target-anchor", required=True)
    parser.add_argument("--barrier-statement", default="pipe_barrier(PIPE_V);")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cells, report = prepare(
            args.disjoint.read_text(encoding="utf-8"),
            args.overlapping.read_text(encoding="utf-8"),
            disjoint_target_anchor=args.disjoint_target_anchor,
            overlapping_target_anchor=args.overlapping_target_anchor,
            barrier_statement=args.barrier_statement,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, text in cells.items():
        (args.output_dir / f"{name}.cpp").write_text(text, encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
