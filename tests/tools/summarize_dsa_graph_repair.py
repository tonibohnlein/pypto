"""Write compact, reproducible evidence from host-only graph-repair runs."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pypto.tools.dsa_graph_audit import parse_weighted_graph
from score_frozen_dsa_graph_manifest import digest, write_json


def base_invariance(root, rows):
    variants = defaultdict(set)
    for i, row in enumerate(rows):
        if row["status"] != "GRAPH_SCORE_RESOLVED":
            continue
        graph = root / f"endpoint-{i:03d}" / "baseline.txt"
        nodes, edges = parse_weighted_graph(graph.read_text())
        canonical = dict(
            nodes={
                i: {k: n[k] for k in ("op", "pipe", "access", "loop_depth", "cycles")}
                for i, n in nodes.items()
            },
            edges=edges,
        )
        key = (row.get("family", "gate"), row.get("cell", row["function"]), row["capacity"])
        variants[key].add(hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest())
    if any(len(v) != 1 for v in variants.values()):
        raise ValueError("non-reusing base graph varies across placements")
    return len(variants)


def write_tsv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--transfers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    historical = json.loads((args.historical / "predictions.json").read_text())
    seal = json.loads((args.historical / "prediction-seal.json").read_text())
    if digest(args.historical / "predictions.json") != seal["predictions_sha256"]:
        raise ValueError("historical prediction seal mismatch")
    gate = json.loads((args.gate / "predictions.json").read_text())
    comparison = json.loads((args.historical / "comparison.json").read_text())
    transfers = json.loads(args.transfers.read_text())
    if transfers["predictions_sha256"] != digest(args.gate / "predictions.json"):
        raise ValueError("mixed transfer audit belongs to a different score run")
    summary = dict(
        new_placements=0,
        new_device_runs=0,
        retrospective_development_analysis=True,
        invocation_model_complete=False,
        sources={
            "historical_predictions": digest(args.historical / "predictions.json"),
            "gate_predictions": digest(args.gate / "predictions.json"),
            "historical_seal": digest(args.historical / "prediction-seal.json"),
            "gate_provenance": digest(args.gate / "provenance.json"),
            "mixed_transfers": digest(args.transfers),
        },
        historical_status=dict(Counter(r["status"] for r in historical)),
        historical_ranges=dict(Counter(r.get("logical_range_precision") for r in historical)),
        gate_status=dict(Counter(r["status"] for r in gate)),
        gate_ranges=dict(Counter(r.get("logical_range_precision") for r in gate)),
        invariant_historical_bases=base_invariance(args.historical, historical),
        invariant_gate_child_bases=base_invariance(args.gate, gate),
        mixed_cells=len(transfers["cells"]),
        weight_grid=[],
    )
    decided = [r for r in comparison if r["confirmed_2pct_both_devices"] == "True"]
    for weight in (0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256):
        counts = Counter(
            "tie"
            if r.get(f"lp_prediction_w{weight}") == "TIE"
            else "correct"
            if r.get(f"lp_prediction_w{weight}") == r["observed"]
            else "wrong"
            for r in decided
        )
        summary["weight_grid"].append(dict(weight=weight, confirmed_cells=len(decided), **counts))
    gate_table = []
    for row in gate:
        item = {k: row[k] for k in ("function", "capacity", "arm", "status", "logical_range_precision")}
        item.update(row["baseline"])
        item.update({f"penalty_w{g['weight']}": g["longest_path_cycles_delta"] for g in row["grid"]})
        gate_table.append(item)
    write_json(args.output / "dsa_graph_repair_v7.json", summary)
    write_tsv(args.output / "dsa_graph_repair_historical_v7.tsv", comparison)
    write_tsv(args.output / "dsa_graph_repair_gate_v7.tsv", gate_table)


if __name__ == "__main__":
    main()
