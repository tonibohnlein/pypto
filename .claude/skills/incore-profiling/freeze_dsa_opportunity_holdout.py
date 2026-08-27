# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze a diverse prospective holdout from opportunity-capacity workloads."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from select_dsa_workload_capacity import OPPORTUNITY_POLICY

HOLDOUT_POLICY = "diverse_cypress_dsa_rp_opportunity_holdout_v1"
_PERFORMANCE_FIELD_FRAGMENTS = ("latency", "median_us", "runtime_us", "speedup", "timing", "winner")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _problem_fingerprints(row: Mapping[str, Any]) -> set[str]:
    fingerprints: set[str] = set()
    for item in str(row.get("problem_fingerprints", "")).split(";"):
        if not item:
            continue
        _name, separator, fingerprint = item.partition("=")
        if not separator or not fingerprint:
            raise ValueError(f"Malformed problem_fingerprints item {item!r}")
        fingerprints.add(fingerprint)
    if not fingerprints:
        raise ValueError(f"Workload {row.get('script', '<unknown>')} has no problem fingerprints")
    return fingerprints


def _assert_timing_blind(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "uses_device_latency":
                if child is not False:
                    raise ValueError(f"{child_path} must be false")
                continue
            if any(fragment in str(key).lower() for fragment in _PERFORMANCE_FIELD_FRAGMENTS):
                raise ValueError(f"Holdout selection rejects performance field {child_path}")
            _assert_timing_blind(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_timing_blind(child, f"{path}[{index}]")


def _model_family(script: str) -> str:
    parts = Path(script).parts
    return parts[1] if len(parts) > 2 and parts[0] == "models" else "unknown"


def _source_class(script: str) -> str:
    stem = Path(script).stem
    return stem.removeprefix("dsa_eval_")


def _selection_key(
    row: Mapping[str, Any], used_families: set[str], used_classes: set[str]
) -> tuple[Any, ...]:
    family = _model_family(str(row["script"]))
    source_class = _source_class(str(row["script"]))
    return (
        -int(source_class not in used_classes),
        -int(family not in used_families),
        -int(row["cypress_minus_dsa_rp_reuse_cost"]),
        -int(row["penalized_relation_disagreement"]),
        -int(row["reuse_relation_disagreement"]),
        str(row["script"]),
    )


def freeze_holdout(
    opportunity_freeze_path: str | Path,
    development_freeze_path: str | Path,
    output_root: str | Path,
    *,
    minimum: int = 8,
    maximum: int = 12,
) -> dict[str, Any]:
    """Freeze timing-blind opportunity workloads not present in development data."""
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"Invalid holdout bounds: minimum={minimum}, maximum={maximum}")

    opportunity_payload = Path(opportunity_freeze_path).read_bytes()
    development_payload = Path(development_freeze_path).read_bytes()
    opportunity = json.loads(opportunity_payload)
    development = json.loads(development_payload)
    for name, document in (("opportunity", opportunity), ("development", development)):
        if document.get("selection_policy") != OPPORTUNITY_POLICY:
            raise ValueError(f"{name} freeze uses unexpected policy {document.get('selection_policy')!r}")
        _assert_timing_blind(document, name)

    development_scripts = {str(row["script"]) for row in development["workloads"]}
    development_fingerprints = {
        fingerprint for row in development["workloads"] for fingerprint in _problem_fingerprints(row)
    }
    seen_scripts: set[str] = set()
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for source in opportunity["workloads"]:
        row = dict(source)
        script = str(row["script"])
        if script in seen_scripts:
            raise ValueError(f"Opportunity freeze repeats script {script}")
        seen_scripts.add(script)
        if row["selection_status"] != "OPPORTUNITY_PRIMARY":
            exclusions.append({"script": script, "reason": str(row["selection_status"])})
            continue
        overlap = sorted(_problem_fingerprints(row) & development_fingerprints)
        if script in development_scripts:
            exclusions.append({"script": script, "reason": "DEVELOPMENT_SCRIPT"})
        elif overlap:
            exclusions.append({"script": script, "reason": f"DEVELOPMENT_PROBLEM:{','.join(overlap)}"})
        else:
            eligible.append(row)

    if len(eligible) < minimum:
        raise ValueError(f"Prospective holdout has only {len(eligible)} eligible workloads; need {minimum}")

    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    used_classes: set[str] = set()
    remaining = eligible.copy()
    while remaining and len(selected) < maximum:
        row = min(remaining, key=lambda item: _selection_key(item, used_families, used_classes))
        remaining.remove(row)
        frozen_row = {
            **row,
            "selection_rank": len(selected) + 1,
            "model_family": _model_family(str(row["script"])),
            "source_class": _source_class(str(row["script"])),
        }
        selected.append(frozen_row)
        used_families.add(frozen_row["model_family"])
        used_classes.add(frozen_row["source_class"])

    freeze = {
        "schema_version": 1,
        "selection_policy": HOLDOUT_POLICY,
        "capacity_selection_policy": OPPORTUNITY_POLICY,
        "uses_device_latency": False,
        "prospective_holdout": True,
        "device_timing_state": "SEALED_UNSEEN",
        "minimum_workloads": minimum,
        "maximum_workloads": maximum,
        "eligible_workload_count": len(eligible),
        "selected_workload_count": len(selected),
        "excluded_workload_count": len(exclusions),
        "model_families": sorted(used_families),
        "source_classes": sorted(used_classes),
        "inputs": {
            "opportunity_freeze_sha256": _sha256(opportunity_payload),
            "development_freeze_sha256": _sha256(development_payload),
        },
        "excluded_workloads": exclusions,
        "workloads": selected,
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    freeze_path = output / "holdout-freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_sha = _sha256(freeze_path.read_bytes())
    (output / "holdout-freeze.json.sha256").write_text(
        f"{freeze_sha}  holdout-freeze.json\n", encoding="utf-8"
    )
    return {**freeze, "holdout_freeze_sha256": freeze_sha}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunity-freeze", required=True, type=Path)
    parser.add_argument("--development-freeze", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--minimum", type=int, default=8)
    parser.add_argument("--maximum", type=int, default=12)
    args = parser.parse_args(argv)
    result = freeze_holdout(
        args.opportunity_freeze,
        args.development_freeze,
        args.output_root,
        minimum=args.minimum,
        maximum=args.maximum,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
