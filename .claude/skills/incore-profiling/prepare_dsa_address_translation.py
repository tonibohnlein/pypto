# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
"""Apply a topology-preserving DSA placement translation to post-sync PTO."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_CONSTANT = re.compile(
    r"^(?P<prefix>\s*(?P<name>%[\w.$-]+)\s*=\s*arith\.constant\s+)"
    r"(?P<value>\d+)(?P<suffix>\s*:\s*i64\s*)$"
)
_ADDRESS = re.compile(r"\baddr\s*=\s*(?P<name>%[\w.$-]+)")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _placement(problem: dict[str, Any], solution: dict[str, Any]) -> dict[int, tuple[int, int, int]]:
    buffers = {int(buffer["id"]): buffer for buffer in problem["problem"]["buffers"]}
    result = {}
    for entry in solution["placements"]:
        buffer_id = int(entry["buffer"])
        if buffer_id in result or buffer_id not in buffers:
            raise ValueError(f"invalid or repeated solution buffer {buffer_id}")
        result[buffer_id] = (int(entry["pool"]), int(entry["offset"]), int(buffers[buffer_id]["size"]))
    if set(result) != set(buffers):
        raise ValueError("solution and problem buffer sets differ")
    return result


def _overlap_geometry(placement: dict[int, tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    result = {}
    ordered = sorted(placement)
    for index, first_id in enumerate(ordered):
        first_pool, first_offset, first_size = placement[first_id]
        for second_id in ordered[index + 1 :]:
            second_pool, second_offset, second_size = placement[second_id]
            if first_pool != second_pool:
                continue
            begin = max(first_offset, second_offset)
            end = min(first_offset + first_size, second_offset + second_size)
            if begin < end:
                result[(first_id, second_id)] = end - begin
    return result


def prepare(
    problem: dict[str, Any],
    base_solution: dict[str, Any],
    translated_solution: dict[str, Any],
    pto_text: str,
) -> tuple[str, dict[str, Any]]:
    base = _placement(problem, base_solution)
    translated = _placement(problem, translated_solution)
    if _overlap_geometry(base) != _overlap_geometry(translated):
        raise ValueError("translated solution does not preserve exact physical overlap geometry")

    address_map: dict[int, int] = {}
    for buffer_id, (base_pool, base_offset, _size) in base.items():
        translated_pool, translated_offset, _translated_size = translated[buffer_id]
        if base_pool != translated_pool:
            raise ValueError(f"buffer {buffer_id} changes pool")
        prior = address_map.setdefault(base_offset, translated_offset)
        if prior != translated_offset:
            raise ValueError(
                f"buffers sharing base address {base_offset} do not translate as one address group"
            )

    lines = pto_text.splitlines()
    definitions: dict[str, tuple[int, int]] = {}
    address_names: set[str] = set()
    for index, line in enumerate(lines):
        constant = _CONSTANT.match(line)
        if constant:
            definitions[constant.group("name")] = (index, int(constant.group("value")))
        address = _ADDRESS.search(line)
        if address:
            address_names.add(address.group("name"))
    missing = address_names - definitions.keys()
    if missing:
        raise ValueError(f"PTO uses non-constant tile addresses: {sorted(missing)}")
    for name in sorted(address_names):
        definition_index, _value = definitions[name]
        unexpected_uses = [
            index + 1
            for index, line in enumerate(lines)
            if index != definition_index
            and name in line
            and not re.search(rf"\baddr\s*=\s*{re.escape(name)}\b", line)
        ]
        if unexpected_uses:
            raise ValueError(
                f"PTO address constant {name} also has non-address uses on lines {unexpected_uses}"
            )

    changed = []
    output = list(lines)
    for name in sorted(address_names):
        index, old_value = definitions[name]
        if old_value not in address_map:
            raise ValueError(f"PTO address {old_value} has no base-solution address group")
        new_value = address_map[old_value]
        match = _CONSTANT.match(lines[index])
        assert match is not None
        output[index] = f"{match.group('prefix')}{new_value}{match.group('suffix')}"
        if old_value != new_value:
            changed.append({"ssa": name, "from": old_value, "to": new_value})
    rendered = "\n".join(output) + ("\n" if pto_text.endswith("\n") else "")

    def sync_lines(source: list[str]) -> list[str]:
        return [
            line.strip()
            for line in source
            if "pto.set_flag" in line or "pto.wait_flag" in line or "pto.barrier" in line
        ]

    sync_before = sync_lines(lines)
    sync_after = sync_lines(output)
    if sync_before != sync_after:
        raise ValueError("address translation changed synchronization topology")
    return rendered, {
        "schema_version": 1,
        "ablation": "topology_preserving_address_translation",
        "base_pto_sha256": _sha256(pto_text),
        "output_pto_sha256": _sha256(rendered),
        "address_group_map": {str(old): new for old, new in sorted(address_map.items())},
        "changed_address_constants": changed,
        "overlap_geometry_identical": True,
        "sync_topology_identical": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--base-solution", type=Path, required=True)
    parser.add_argument("--translated-solution", type=Path, required=True)
    parser.add_argument("--base-pto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rendered, report = prepare(
            _read_object(args.problem),
            _read_object(args.base_solution),
            _read_object(args.translated_solution),
            args.base_pto.read_text(encoding="utf-8"),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
