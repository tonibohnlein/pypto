# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Consolidate archived driver timings without selecting on measured latency.

Run from the repository root with --analysis-root pointing to the completed
host reconstruction. Outputs are retrospective development evidence.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
WEIGHTS = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
CAMPAIGN = "dsa-rp-rebased-corpus-device-continuation-3eabcfd22"


def read_json(path: Path) -> dict:
    """Read one JSON document."""
    return json.loads(path.read_text())


def read_tsv(path: Path) -> list[dict]:
    """Read a tab-separated table."""
    with path.open(newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    """Write deterministic tables with the union of all row fields."""
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=fields, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONNUMERIC
        )
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    """Hash an input or output file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def placement_relations(document: dict, solution: dict) -> set[tuple[int, int]]:
    """Enumerate physical range intersections, independently of penalty weights."""
    buffers = {row["id"]: row for row in document["problem"]["buffers"]}
    placements = {row["buffer"]: row for row in solution["placements"]}
    if buffers.keys() != placements.keys():
        raise ValueError("incomplete placement")
    relations = set()
    for first, second in itertools.combinations(sorted(buffers), 2):
        a, b = placements[first], placements[second]
        if a["pool"] == b["pool"] and (
            a["offset"] < b["offset"] + buffers[second]["size"]
            and b["offset"] < a["offset"] + buffers[first]["size"]
        ):
            relations.add((first, second))
    return relations


def choose_primary(structural_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Select one complete-map configuration per driver/argv without timing fields.

    Maximize the complete-map objective gap where the three main arms differ
    and Cypress realizes penalized reuse. Keep non-opportunity rows as controls.
    Selection is restricted to archived configurations; it cannot invent a
    missing capacity measurement.
    """
    groups = defaultdict(list)
    for row in structural_rows:
        if any("latency" in key or "median" in key or "samples" in key for key in row):
            raise ValueError("capacity selector received latency-bearing fields")
        groups[(row["script"], row["argv_json"])].append(row)
    selected, audit = [], []
    for key in sorted(groups):
        rows = groups[key]
        ranked = sorted(
            rows,
            key=lambda row: (
                -int(row["opportunity"]),
                -row["unit_gap"] if row["opportunity"] else 0,
                -row["relation_disagreement"] if row["opportunity"] else 0,
                row["capacity_bytes"],
                row["artifact_id"],
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        for row in ranked:
            audit.append(
                {
                    **row,
                    "role": "PRIMARY" if row is chosen else "SENSITIVITY",
                    "selection": "OPPORTUNITY" if chosen["opportunity"] else "CONTROL",
                }
            )
    return selected, audit


def launch_samples(record: dict, arm: str, device: int) -> list[list[float]]:
    """Read only the requested physical endpoint, respecting map-identity sharing."""
    expected = record["maps"][arm]["map_digest"]
    launches = [
        row for row in record["launches"] if row["device"] == device and row["map_digest"] == expected
    ]
    if len(launches) < 3 or len({row["rep"] for row in launches}) != len(launches):
        raise ValueError(f"missing or duplicated launches for {arm}, device {device}")
    result = []
    for launch in sorted(launches, key=lambda row: row["rep"]):
        values = launch["samples_us"]
        if launch["status"] != "PASS" or launch["replay_provenance"] != "PASS":
            raise ValueError("unvalidated timing launch")
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("nonpositive or missing device timing")
        result.append(values)
    return result


def contrast(first: list[list[float]], second: list[list[float]], *, identical: bool) -> dict:
    """Estimate candidate/reference latency change with launches as resampling units."""
    a = [statistics.median(values) for values in first]
    b = [statistics.median(values) for values in second]
    change = 100 * (statistics.median(b) / statistics.median(a) - 1)
    if identical:
        if first != second:
            raise ValueError("identical endpoints have different shared measurements")
        return {"latency_change_pct": 0.0, "ci_low_pct": 0.0, "ci_high_pct": 0.0}
    rng = random.Random(0)
    distribution = sorted(
        100 * (statistics.median(rng.choices(b, k=len(b))) / statistics.median(rng.choices(a, k=len(a))) - 1)
        for _ in range(2000)
    )
    return {"latency_change_pct": change, "ci_low_pct": distribution[49], "ci_high_pct": distribution[1949]}


def classify_device(changes: list[dict]) -> str:
    """Require the effect threshold and a launch-bootstrap interval on both devices."""
    if len(changes) != 2:
        raise ValueError("expected two devices")
    if all(row["latency_change_pct"] <= -2 and row["ci_high_pct"] < 0 for row in changes):
        return "DSA_RP_FASTER"
    if all(row["latency_change_pct"] >= 2 and row["ci_low_pct"] > 0 for row in changes):
        return "CYPRESS_FASTER"
    if all(abs(row["latency_change_pct"]) < 2 for row in changes):
        return "SMALL_OR_NULL"
    return "UNCONFIRMED_OR_DEVICE_DISAGREEMENT"


def direction(cypress: float, dsa_rp: float) -> str:
    """Interpret lower predictor values as lower predicted latency."""
    return "DSA_RP_FASTER" if cypress > dsa_rp else "CYPRESS_FASTER" if cypress < dsa_rp else "TIE"


def assessment(predicted: str, observed: str) -> str:
    """Keep unavailable, wrong, tied, and unresolved outcomes distinct."""
    if predicted == "UNAVAILABLE":
        return "UNAVAILABLE"
    if observed in {"DSA_RP_FASTER", "CYPRESS_FASTER"}:
        return "CORRECT" if predicted == observed else "MISSED_DIRECTION" if predicted == "TIE" else "WRONG"
    if observed == "SMALL_OR_NULL":
        return "NULL_TIE" if predicted == "TIE" else "STRICT_ON_SMALL_OR_NULL"
    return "DEVICE_UNRESOLVED"


def load_structural_inputs(root: Path) -> tuple:
    """Validate archive identities and derive structural metrics for each map."""
    campaign = root / "input" / CAMPAIGN
    frozen = read_json(root / "development-corpus-current-v1.json")
    prediction_path = root / "complete-placement-predictions.json"
    if digest(prediction_path) != prediction_path.with_suffix(".json.sha256").read_text().split()[0]:
        raise ValueError("prediction seal mismatch")
    for name, expected in frozen["tables"].items():
        if digest(campaign / "results" / name) != expected:
            raise ValueError(f"archived table changed: {name}")
    predictions = read_json(prediction_path)["records"]
    by_map = defaultdict(list)
    for record in predictions:
        by_map[(record["script"], record["map_digest"])].append(record)
    documents = {}
    for row in read_tsv(root / "current-export-all/invocations.tsv"):
        document = read_json(root / "current-export-all" / row["document"])
        documents[(row["script"], row["instance"])] = document
    deterministic = {
        (row["script"], row["workload"], row["capacity"])
        for row in read_tsv(campaign / "results/determinism.tsv")
        if row["classification"] == "CROSS_ARM_BIT_IDENTICAL"
        and all(
            row[key] == "YES"
            for key in (
                "within_arm_device_deterministic",
                "cross_arm_bit_identity",
                "golden_all_pass",
                "identical_inputs",
            )
        )
    }
    export = {row["script"]: row for row in read_tsv(root / "current-export-all/export-status.tsv")}
    records, structural, metrics = {}, [], {}
    for path in sorted((campaign / "artifacts/device").glob("*/row.json")):
        record = read_json(path)
        row, maps = record["row"], record["maps"]
        if (row["script"], row["workload"], row["capacity"]) not in deterministic:
            continue
        if record["terminal_status"] != "DEVICE_MEASURED" or set(maps) != set(ARMS):
            raise ValueError(f"incomplete measured row {path}")
        relations = {}
        for arm, report in maps.items():
            if report["status"] != "MAP_COMPLETE":
                raise ValueError(f"incomplete map {path}, {arm}")
            union, cost = set(), 0.0
            for name, instance in report["instances"].items():
                solution = read_json(
                    root / "reconstructed-maps" / report["map_digest"] / f"pypto_{name}.dsa.solution.json"
                )
                pairs = placement_relations(documents[(row["script"], name)], solution)
                union.update((name, *pair) for pair in pairs)
                cost += float(instance["reuse_cost"])
            relations[arm] = union
            metrics[(path.parent.name, arm)] = {
                "unit_cost": cost,
                "range_intersection_pairs": len(union),
                "peak_sum_bytes": sum(int(x["total_peak"]) for x in report["instances"].values()),
                "map_digest": report["map_digest"],
            }
        costs = {arm: metrics[(path.parent.name, arm)]["unit_cost"] for arm in ARMS}
        gap = costs["cypress"] - costs["dsa_rp_cg"]
        distinct = len({maps[arm]["map_digest"] for arm in ("geometry_ff", "cypress", "dsa_rp_cg")})
        structural.append(
            {
                "artifact_id": path.parent.name,
                "script": row["script"],
                "target": row["target"],
                "argv_json": json.dumps(row.get("argv") or [], separators=(",", ":")),
                "capacity": row["capacity"],
                "capacity_bytes": row["capacity_bytes"],
                "tightened_pool_id": row["pool_id"],
                "child_instances": len(maps["cypress"]["instances"]),
                "unit_gap": gap,
                "distinct_primary_maps": distinct,
                "relation_disagreement": len(relations["cypress"] ^ relations["dsa_rp_cg"]),
                "opportunity": costs["cypress"] > 0 and gap > 0 and distinct == 3,
                **{f"{arm}_map": maps[arm]["map_digest"] for arm in ARMS},
            }
        )
        records[path.parent.name] = record
    return frozen, by_map, export, records, structural, metrics


def evaluate_cells(audit: list[dict], records: dict, export: dict, metrics: dict, by_map: dict) -> tuple:
    """Join selected configurations to measured driver windows and comparable models."""
    timings, comparisons, evaluation, case_study = [], [], [], []
    for row in audit:
        evaluate_cell(row, records, export, metrics, by_map, timings, comparisons, evaluation, case_study)
    return timings, comparisons, evaluation, case_study


def summarize_predictors(evaluation: list[dict], *, matched: bool) -> list[dict]:
    """Compare like-for-like coverage, without selecting on predictor correctness."""
    eligible = {
        (row["artifact_id"], row["sync_weight_cycles"])
        for row in evaluation
        if row["predictor"] == "dag" and row["predicted"] != "UNAVAILABLE"
    }
    groups = defaultdict(Counter)
    for row in evaluation:
        weights = WEIGHTS if matched and row["predictor"] != "dag" else (row["sync_weight_cycles"],)
        for weight in weights:
            if matched and (row["artifact_id"], weight) not in eligible:
                continue
            groups[(row["role"], row["predictor"], weight)][row["assessment"]] += 1
    return [
        {
            "coverage": "DAG_MATCHED" if matched else "ALL",
            "role": role,
            "predictor": predictor,
            "sync_weight_cycles": weight,
            **{
                key: counts[key]
                for key in (
                    "CORRECT",
                    "WRONG",
                    "MISSED_DIRECTION",
                    "NULL_TIE",
                    "STRICT_ON_SMALL_OR_NULL",
                    "DEVICE_UNRESOLVED",
                    "UNAVAILABLE",
                )
            },
        }
        for (role, predictor, weight), counts in sorted(groups.items())
    ]


def audit_reserved_candidates(candidates: list[dict], development_targets: set[str]) -> list[dict]:
    """Keep candidate rows, source families and independence claims separate."""
    families = Counter(row["family"] or row["instance"] for row in candidates)
    return [
        {
            **row,
            "source_family": row["family"] or row["instance"],
            "rows_in_family": families[row["family"] or row["instance"]],
            "development_family_overlap": (row["family"] or row["instance"]) in development_targets,
            "independent_workload_proven": False,
            "next_gate": "ISOLATE_ORIGINAL_HELPER_AND_PROVE_SHAPE_IDENTITY",
        }
        for row in candidates
    ]


def main() -> None:
    """Produce the primary paper table and scope-matched predictor comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.analysis_root, args.output_root
    frozen, by_map, export, records, structural, metrics = load_structural_inputs(root)
    prediction_path = root / "complete-placement-predictions.json"
    primary, audit = choose_primary(structural)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "capacity-selection.tsv", audit)
    write_tsv(
        out / "source-records.tsv",
        [
            {
                "artifact_id": row["artifact_id"],
                "sha256": digest(
                    root / "input" / CAMPAIGN / "artifacts/device" / row["artifact_id"] / "row.json"
                ),
            }
            for row in audit
        ],
    )
    # Raw timing-bearing record hashes are provenance, not structural corpus identity.
    manifest = {
        "schema_version": 1,
        "role": "RETROSPECTIVE_DEVELOPMENT",
        "selection_policy": "complete_map_gap_then_disagreement_then_capacity_v1",
        "capacity_universe": "existing measured configurations only",
        "uses_latency_for_selection": False,
        "pins": frozen["pins"],
        "rows": primary,
    }
    (out / "primary-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    primary_ids = {row["artifact_id"] for row in primary}
    timings, comparisons, evaluation, case_study = evaluate_cells(audit, records, export, metrics, by_map)
    write_tsv(out / "paper-primary.tsv", [row for row in timings if row["artifact_id"] in primary_ids])
    write_tsv(out / "sensitivity.tsv", [row for row in timings if row["artifact_id"] not in primary_ids])
    write_tsv(out / "pairwise.tsv", comparisons)
    write_tsv(out / "predictor-evaluation.tsv", evaluation)
    write_tsv(out / "hidden-norm-case-study.tsv", case_study)
    write_tsv(out / "predictor-summary.tsv", summarize_predictors(evaluation, matched=False))
    write_tsv(out / "predictor-matched-summary.tsv", summarize_predictors(evaluation, matched=True))
    reserved = audit_reserved_candidates(
        read_tsv(root / "evaluation/reserved-prospective-holdout.tsv"),
        {row["target"] for row in audit},
    )
    write_tsv(out / "reserved-candidate-audit.tsv", reserved)
    modes = {
        row["artifact_id"]: row["measurement_mode"] for row in timings if row["artifact_id"] in primary_ids
    }
    summary = {
        "primary_workloads": len(primary),
        "deterministic_configurations": len(audit),
        "primary_measurement_modes": dict(Counter(modes.values())),
        "reserved_candidate_rows": len(reserved),
        "reserved_source_families": len({row["source_family"] for row in reserved}),
        "reserved_families_overlapping_development": sorted(
            {row["source_family"] for row in reserved if row["development_family_overlap"]}
        ),
        "prediction_sha256": digest(prediction_path),
        "primary_manifest_sha256": digest(out / "primary-manifest.json"),
        "source_archive": frozen["source_archive"],
        "limitations": [
            "three independent launches per endpoint/device; fixed order per device",
            "capacity selection retrospective; measured configurations only",
            "alias count and peak are Cypress proxies; relaxed-edge counts unavailable",
            "range intersections count original allocation pairs, not contracted Cypress search nodes",
            "reported medians are medians of launch medians; bootstrap resamples launches",
            "strict score ordering on a measured small effect does not establish a magnitude error",
            "DAG scoring on multi-function parents requires composition",
            "all timing is the driver device window; no per-dispatch extraction",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


def evaluate_cell(
    row: dict,
    records: dict,
    export: dict,
    metrics: dict,
    by_map: dict,
    timings: list,
    comparisons: list,
    evaluation: list,
    case_study: list,
) -> None:
    """Evaluate one configuration without changing its selection status."""
    if not row:
        raise ValueError("missing configuration")
    aid = row["artifact_id"]
    record = records[aid]
    devices = sorted({launch["device"] for launch in record["launches"]})
    inventory = export[row["script"]]
    mode = (
        "SINGLE_FUNCTION_DRIVER_WINDOW"
        if row["child_instances"] == 1 and int(inventory["emitted_kernels"]) == 1
        else "PARENT_PROGRAM_WINDOW"
    )
    common = {
        "artifact_id": aid,
        "script": row["script"],
        "target": row["target"],
        "argv": row["argv_json"],
        "capacity": row["capacity"],
        "tightened_pool_id": row["tightened_pool_id"],
        "capacity_bytes": row["capacity_bytes"],
        "role": row["role"],
        "selection": row["selection"],
        "measurement_mode": mode,
        "metric": "driver_device_effective_us",
        "child_instances": row["child_instances"],
        "placement_scope": "COMPLETE_DRIVER_MAP",
        "capacity_scope": "SELECTED_TARGET_POOL_OTHER_POOLS_NATIVE",
        "estimator": "MEDIAN_OF_LAUNCH_MEDIANS",
    }
    changes = []
    for device in devices:
        samples = {arm: launch_samples(record, arm, device) for arm in ARMS}
        wide = {**common, "device": device}
        for arm in ARMS:
            values = samples[arm]
            wide[f"{arm}_us"] = statistics.median([statistics.median(value) for value in values])
            wide[f"{arm}_map"] = metrics[(aid, arm)]["map_digest"]
            wide[f"{arm}_launches"] = len(values)
        timings.append(wide)
        for first, second in itertools.combinations(ARMS, 2):
            same = metrics[(aid, first)]["map_digest"] == metrics[(aid, second)]["map_digest"]
            change = contrast(samples[first], samples[second], identical=same)
            comparisons.append(
                {
                    **common,
                    "device": device,
                    "reference": first,
                    "candidate": second,
                    "physical_null": same,
                    **change,
                }
            )
            if (first, second) == ("cypress", "dsa_rp_cg"):
                changes.append(change)
    observed = classify_device(changes)
    for predictor in ("unit_cost", "range_intersection_pairs", "peak_sum_bytes", "dag"):
        for weight in WEIGHTS if predictor == "dag" else (0,):
            scores, reason = {}, ""
            for arm in ("cypress", "dsa_rp_cg"):
                if predictor != "dag":
                    scores[arm] = metrics[(aid, arm)][predictor]
                    continue
                units = by_map[(row["script"], metrics[(aid, arm)]["map_digest"])]
                if mode != "SINGLE_FUNCTION_DRIVER_WINDOW" or len(units) != 1:
                    reason = "PARENT_GRAPH_COMPOSITION_REQUIRED"
                elif units[0]["model_status"] != "MODEL_ELIGIBLE":
                    reason = units[0].get("model_reason", "MODEL_INELIGIBLE")
                else:
                    scores[arm] = units[0]["score_grid"][str(weight)]["penalty_cycles"]
            predicted = (
                direction(scores["cypress"], scores["dsa_rp_cg"]) if len(scores) == 2 else "UNAVAILABLE"
            )
            evaluation.append(
                {
                    **common,
                    "predictor": predictor,
                    "sync_weight_cycles": weight,
                    "cypress_score": scores.get("cypress", ""),
                    "dsa_rp_score": scores.get("dsa_rp_cg", ""),
                    "predicted": predicted,
                    "observed": observed,
                    "assessment": assessment(predicted, observed),
                    "reason": reason,
                }
            )
    if row["target"] == "mtp_hidden_norm_quant":
        for arm in ARMS:
            unit = by_map[(row["script"], metrics[(aid, arm)]["map_digest"])]
            if len(unit) != 1:
                raise ValueError("hidden norm is not a single graph")
            unit = unit[0]
            case_study.append(
                {
                    **common,
                    "arm": arm,
                    **metrics[(aid, arm)],
                    "base_cycles": unit.get("base_longest_path_cycles"),
                    "reuse_edges": unit.get("placement_reuse_edge_count"),
                    "recurrences": unit.get("placement_reuse_recurrence_count"),
                    **{f"penalty_w{w}": unit["score_grid"][str(w)]["penalty_cycles"] for w in WEIGHTS},
                }
            )


if __name__ == "__main__":
    main()
