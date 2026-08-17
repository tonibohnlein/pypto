# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze every penalty-bearing invocation for a four-arm standalone census.

This is a host-side join.  It deliberately does not infer launchability from a
DSA problem: current dispatch capture and standalone correctness must establish
that on the device host.  In particular, it never falls back to timing one
changed kernel inside a parent program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty census table: {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tag_from_corpus_document(value: str) -> str:
    name = Path(value).name
    suffix = ".dsa.json"
    if not name.endswith(suffix):
        raise ValueError(f"corpus document is not a DSA problem: {value}")
    return name[: -len(suffix)]


def _native_arm_rows(screen_rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in screen_rows:
        if row.get("capacity_label") == "native" and row.get("arm") in ARMS:
            grouped[(row["tag"], row["arm"])].append(row)

    by_tag: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for (tag, arm), rows in grouped.items():
        statuses = {row.get("status", "") for row in rows}
        placement_hashes = {
            row.get("placement_sha256", "")
            for row in rows
            if row.get("status") == "feasible" and row.get("placement_sha256")
        }
        if statuses == {"feasible"} and len(placement_hashes) != 1:
            raise ValueError(
                f"native {tag}/{arm} has {len(placement_hashes)} placement identities across pools"
            )
        representative = min(rows, key=lambda row: (int(row.get("pool_id", -1)), row.get("pool", "")))
        by_tag[tag][arm] = {
            **representative,
            "placement_sha256": next(iter(placement_hashes)) if placement_hashes else "",
        }
    return by_tag


def _solution_path(screen_root: Path, row: dict[str, str]) -> Path:
    return (
        screen_root
        / "raw"
        / row["tag"]
        / f"pool-{row['pool_id']}-{row['pool']}"
        / "native"
        / row["arm"]
        / "solution.json"
    )


def _index_unique_problems(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    by_fingerprint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_fingerprint[row["problem_fingerprint"]].append(row)
    structural_fields = ("instance", "buffers", "pools", "pool_names", "reuse_penalties")
    for fingerprint, members in by_fingerprint.items():
        signatures = {tuple(member[field] for field in structural_fields) for member in members}
        if len(signatures) != 1:
            raise ValueError(f"semantic fingerprint has incompatible problem rows: {fingerprint}")
    return by_fingerprint


def _resolve_unique_problem(
    invocation: dict[str, str],
    by_fingerprint: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    fingerprint = invocation["problem_fingerprint"]
    candidates = by_fingerprint.get(fingerprint, [])
    if not candidates:
        raise ValueError(f"invocation has no unique-problem row: {fingerprint}")
    return min(candidates, key=lambda row: (row["corpus_document"], row["document_sha256"]))


def build_census_rows(
    invocations: list[dict[str, str]],
    unique_problems: list[dict[str, str]],
    screen_rows: list[dict[str, str]],
    screen_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join all penalty-bearing invocations to their four native placements."""
    unique_by_fingerprint = _index_unique_problems(unique_problems)
    native = _native_arm_rows(screen_rows)
    screen_terminals = {
        row["tag"]: row.get("status", "") for row in screen_rows if row.get("tag") and not row.get("arm")
    }
    invocation_rows: list[dict[str, Any]] = []
    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for invocation in invocations:
        if int(invocation["reuse_penalties"]) <= 0:
            continue
        fingerprint = invocation["problem_fingerprint"]
        unique = _resolve_unique_problem(invocation, unique_by_fingerprint)
        tag = _tag_from_corpus_document(unique["corpus_document"])
        representative_document_sha = unique["document_sha256"]
        arms = native.get(tag, {})
        missing = [arm for arm in ARMS if arm not in arms]
        infeasible = [arm for arm in ARMS if arm in arms and arms[arm].get("status") != "feasible"]
        host_status = "FOUR_ARM_FEASIBLE"
        if missing:
            terminal = screen_terminals.get(tag)
            host_status = f"HOST_{terminal.upper()}" if terminal else "SCREEN_MISSING"
        elif infeasible:
            host_status = "ARM_NOT_FEASIBLE"

        arm_fields: dict[str, Any] = {}
        placement_hashes: set[str] = set()
        for arm in ARMS:
            row = arms.get(arm)
            solution = _solution_path(screen_root, row) if row is not None else None
            if row is not None and row.get("status") == "feasible":
                if solution is None or not solution.is_file():
                    raise FileNotFoundError(f"native solution is missing: {solution}")
                placement_hashes.add(row["placement_sha256"])
            arm_fields[f"{arm}_status"] = "" if row is None else row.get("status", "")
            arm_fields[f"{arm}_placement_sha256"] = "" if row is None else row.get("placement_sha256", "")
            arm_fields[f"{arm}_solution"] = "" if solution is None else str(solution)

        identity = "\0".join((invocation["script"], invocation["instance"], invocation["document_sha256"]))
        census_row = {
            "invocation_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
            "source_script": invocation["script"],
            "instance": invocation["instance"],
            "problem_fingerprint": fingerprint,
            "problem_tag": tag,
            "document_sha256": invocation["document_sha256"],
            "representative_document_sha256": representative_document_sha,
            "buffers": int(invocation["buffers"]),
            "pools": int(invocation["pools"]),
            "pool_names": invocation["pool_names"],
            "reuse_penalties": int(invocation["reuse_penalties"]),
            "host_status": host_status,
            "missing_arms": ";".join(missing),
            "infeasible_arms": ";".join(infeasible),
            "distinct_placement_count": len(placement_hashes),
            "measurement_mode": (
                "NEEDS_CURRENT_DISPATCH_CAPTURE"
                if host_status == "FOUR_ARM_FEASIBLE"
                else "NOT_MEASURABLE_HOST"
            ),
            "parent_fallback": "FORBIDDEN",
            "terminal_status": (
                "PENDING_DEVICE_LAUNCHABILITY" if host_status == "FOUR_ARM_FEASIBLE" else host_status
            ),
            **arm_fields,
        }
        invocation_rows.append(census_row)
        by_fingerprint[fingerprint].append(census_row)

    if not invocation_rows:
        raise ValueError("invocation inventory contains no penalty-bearing rows")
    invocation_rows.sort(
        key=lambda row: (
            str(row["source_script"]),
            str(row["instance"]),
            str(row["invocation_id"]),
        )
    )
    representative_rows: list[dict[str, Any]] = []
    for fingerprint, members in sorted(by_fingerprint.items()):
        first = members[0]
        representative_rows.append(
            {
                "document_sha256": first["representative_document_sha256"],
                "problem_fingerprint": fingerprint,
                "problem_tag": first["problem_tag"],
                "invocations": len(members),
                "source_scripts": ";".join(sorted({str(member["source_script"]) for member in members})),
                "buffers": first["buffers"],
                "pools": first["pools"],
                "pool_names": first["pool_names"],
                "reuse_penalties": first["reuse_penalties"],
                "host_status": first["host_status"],
                "distinct_placement_count": first["distinct_placement_count"],
                "device_status": "PENDING_INVOCATION_CENSUS",
            }
        )
    return invocation_rows, representative_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--unique-problems", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        parser.error(f"output root already exists: {args.output_root}")
    screen_results = args.screen_root / "screen-results.tsv"
    invocation_rows, representative_rows = build_census_rows(
        _read_tsv(args.invocations),
        _read_tsv(args.unique_problems),
        _read_tsv(screen_results),
        args.screen_root,
    )
    args.output_root.mkdir(parents=True)
    _write_tsv(args.output_root / "invocation-census.tsv", invocation_rows)
    _write_tsv(args.output_root / "problem-census.tsv", representative_rows)
    (args.output_root / "census-frozen.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": "all_penalty_bearing_invocations_four_arm_native_v1",
                "arms": list(ARMS),
                "parent_fallback": "forbidden",
                "input_sha256": {
                    "invocations": _sha256(args.invocations),
                    "unique_problems": _sha256(args.unique_problems),
                    "screen_results": _sha256(screen_results),
                },
                "invocations": len(invocation_rows),
                "unique_problems": len(representative_rows),
                "four_arm_host_feasible_invocations": sum(
                    row["host_status"] == "FOUR_ARM_FEASIBLE" for row in invocation_rows
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"FROZEN invocations={len(invocation_rows)} unique_problems={len(representative_rows)} "
        f"four_arm_host_feasible={sum(row['host_status'] == 'FOUR_ARM_FEASIBLE' for row in invocation_rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
