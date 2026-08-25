# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
"""Prepare a sync-only PTO ablation by restoring one proven event pair."""

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


def prepare(
    base_text: str,
    reference_text: str,
    *,
    set_after: str,
    wait_before: str,
    source_pipe: str,
    destination_pipe: str,
    event_id: int,
) -> tuple[str, dict[str, Any]]:
    if event_id < 0:
        raise ValueError("event_id must be non-negative")
    base = base_text.splitlines()
    reference = reference_text.splitlines()
    base_set_anchor = _unique_index(base, set_after, source="base set-after")
    base_wait_anchor = _unique_index(base, wait_before, source="base wait-before")
    reference_set_anchor = _unique_index(reference, set_after, source="reference set-after")
    reference_wait_anchor = _unique_index(reference, wait_before, source="reference wait-before")
    if base_set_anchor >= base_wait_anchor or reference_set_anchor >= reference_wait_anchor:
        raise ValueError("set-after anchor must precede wait-before anchor")

    set_line = f"pto.set_flag[<{source_pipe}>, <{destination_pipe}>, <EVENT_ID{event_id}>]"
    wait_line = f"pto.wait_flag[<{source_pipe}>, <{destination_pipe}>, <EVENT_ID{event_id}>]"
    if reference[reference_set_anchor + 1].strip() != set_line:
        raise ValueError("reference does not contain the requested set_flag after the set anchor")
    if reference[reference_wait_anchor - 1].strip() != wait_line:
        raise ValueError("reference does not contain the requested wait_flag before the wait anchor")
    if any(line.strip() in (set_line, wait_line) for line in base[base_set_anchor + 1 : base_wait_anchor]):
        raise ValueError("base already uses the requested pipe-pair event between the anchors")

    set_indent = base[base_set_anchor][: len(base[base_set_anchor]) - len(base[base_set_anchor].lstrip())]
    wait_indent = base[base_wait_anchor][: len(base[base_wait_anchor]) - len(base[base_wait_anchor].lstrip())]
    output = list(base)
    output.insert(base_wait_anchor, wait_indent + wait_line)
    output.insert(base_set_anchor + 1, set_indent + set_line)
    rendered = "\n".join(output) + ("\n" if base_text.endswith("\n") else "")
    recovered = list(output)
    del recovered[base_wait_anchor + 1]
    del recovered[base_set_anchor + 1]
    if recovered != base:
        raise ValueError("sync ablation changed a non-restored line")
    return rendered, {
        "schema_version": 1,
        "ablation": "restore_one_event_pair",
        "base_sha256": _sha256(base_text),
        "reference_sha256": _sha256(reference_text),
        "output_sha256": _sha256(rendered),
        "source_pipe": source_pipe,
        "destination_pipe": destination_pipe,
        "event_id": event_id,
        "set_after": set_after,
        "wait_before": wait_before,
        "base_set_anchor_line": base_set_anchor + 1,
        "base_wait_anchor_line": base_wait_anchor + 1,
        "reference_pair_proof": True,
        "added_lines": [set_indent + set_line, wait_indent + wait_line],
        "non_sync_lines_identical_to_base": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--set-after", required=True)
    parser.add_argument("--wait-before", required=True)
    parser.add_argument("--source-pipe", required=True)
    parser.add_argument("--destination-pipe", required=True)
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rendered, report = prepare(
            args.base.read_text(encoding="utf-8"),
            args.reference.read_text(encoding="utf-8"),
            set_after=args.set_after,
            wait_before=args.wait_before,
            source_pipe=args.source_pipe,
            destination_pipe=args.destination_pipe,
            event_id=args.event_id,
        )
    except (IndexError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
