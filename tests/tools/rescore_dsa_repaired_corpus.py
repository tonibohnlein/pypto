"""Rescore frozen historical placements with repaired base dependencies; no timing runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pypto.tools import dsa_schedule_model as model
from pypto.tools.dsa_pto_isa_duration import PtoIsaDurationProvider
from score_frozen_dsa_graph_manifest import digest, run_graph, write_json


def compare_existing(rows, observations):
    """Join fixed placements only; these scores are not new-placement measurements."""
    groups = defaultdict(dict)
    for row in rows:
        key = (
            f"fourcand|{row['cell']}"
            if row["family"] == "fourcand"
            else "|".join((row["family"], row["script"], row["function"], row["capacity"]))
        )
        if row["arm"] in groups[key]:
            raise ValueError(f"duplicate frozen arm {key}/{row['arm']}")
        groups[key][row["arm"]] = row
    result = []
    for observation in observations:
        arms = groups.get(observation["cell"], {})
        cypress, rp = arms.get("cypress"), arms.get("dsa_rp_cg")
        row = {
            key: observation[key]
            for key in ("cell", "observed", "device_deltas_pct", "confirmed_2pct_both_devices")
        }
        row.update(status="INCOMPLETE", invocation_model_complete=False)
        if cypress and rp and all(r["status"] == "GRAPH_SCORE_RESOLVED" for r in (cypress, rp)):
            if cypress["baseline"] != rp["baseline"]:
                raise ValueError(f"placement-independent base differs: {observation['cell']}")
            row["status"] = "GRAPH_SCORE_RESOLVED"
            for c, d in zip(cypress["grid"], rp["grid"], strict=True):
                if c["weight"] != d["weight"]:
                    raise ValueError("weight grid mismatch")
                delta = d["longest_path_cycles_delta"] - c["longest_path_cycles_delta"]
                row[f"lp_delta_w{c['weight']}"] = delta
                row[f"lp_prediction_w{c['weight']}"] = (
                    "DSA_RP_FASTER" if delta < 0 else "CYPRESS_FASTER" if delta > 0 else "TIE"
                )
        result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-opt", type=Path, required=True)
    parser.add_argument("--duration-model", type=Path, required=True)
    parser.add_argument("--pto-isa-root", type=Path, required=True)
    parser.add_argument("--pto-isa-revision", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--existing-cell-table", type=Path, required=True)
    parser.add_argument("--conservative-logical-ranges", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Rejoin already sealed predictions; never rerun or alter their scores",
    )
    args = parser.parse_args()
    if args.evaluate_only:
        seal = json.loads((args.output / "prediction-seal.json").read_text())
        if digest(args.output / "predictions.json") != seal["predictions_sha256"]:
            raise ValueError("prediction seal mismatch")
        predictions = json.loads((args.output / "predictions.json").read_text())
        with args.existing_cell_table.open() as stream:
            observations = list(csv.DictReader(stream, delimiter="\t"))
        write_json(args.output / "comparison.json", compare_existing(predictions, observations))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    duration = model.DurationModel.from_json(json.loads(args.duration_model.read_text()))
    duration.pto_isa_provider = PtoIsaDurationProvider.from_checkout(
        args.pto_isa_root, expected_revision=args.pto_isa_revision, unsupported_policy="error"
    )
    duration = model.calibrate_from_metrics([args.calibration], duration)
    rows = []
    frozen = json.loads((args.prior / "predictions.json").read_text())
    for i, old in enumerate(frozen):
        out = args.output / f"endpoint-{i:03d}"
        out.mkdir()
        row = {k: old[k] for k in ("family", "script", "cell", "function", "capacity", "arm")}
        row.update(status="INCOMPLETE", invocation_model_complete=False)
        try:
            endpoint = args.prior / old["family"] / f"{old['cell']}__{old['arm_dir']}"
            address = json.loads((endpoint / "address-evidence.json").read_text())
            pto = Path(address["source"]).parent.parent / "pto" / f"{old['function']}.pto"
            if digest(pto) != old["old_pto_sha256"]:
                raise ValueError("frozen placed PTO hash changed")
            solution = endpoint / "reconstructed.solution.json"
            if digest(solution) != old["reconstructed_solution_sha256"]:
                raise ValueError("frozen solution hash changed")
            problem = endpoint / "instrumented-problem.json"
            fresh = (
                args.prior
                / "fresh"
                / f"{old['family']}__{old['script'].replace('/', '_')}__{old['function']}"
            )
            schedule_path = fresh / "schedule.jsonl"
            record = json.loads(schedule_path.read_text())
            row["inputs"] = {
                k: {"path": str(p), "sha256": digest(p)}
                for k, p in (
                    ("pto", pto),
                    ("solution", solution),
                    ("problem", problem),
                    ("schedule", schedule_path),
                )
            }
            options = [f"function-name={old['function']}", "semantic-boundaries=true"]
            skeleton = out / "skeleton.txt"
            run_graph(args.graph_opt, pto, skeleton, *options)
            logical = model.emit_ptoas_logical_memory_topology(
                record,
                problem,
                skeleton,
                conservative_ranges=args.conservative_logical_ranges,
            )
            row["logical_range_precision"] = logical["range_precision"]
            logical_path = out / "logical-memory.json"
            write_json(logical_path, logical)
            options.append(f"logical-memory-edges={logical_path}")
            base = out / "base.txt"
            run_graph(args.graph_opt, pto, base, *options)
            topology = model.emit_ptoas_placement_reuse_topology(record, problem, solution, base)
            write_json(out / "topology.json", topology)
            durations = model.emit_ptoas_resolved_node_durations(record, duration, base)
            write_json(out / "durations.json", durations)
            options += [f"node-durations={out / 'durations.json'}", "require-exact-durations=true"]
            baseline = run_graph(args.graph_opt, pto, out / "baseline.txt", *options)
            grid = []
            for w in (0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256):
                score = run_graph(
                    args.graph_opt,
                    pto,
                    out / f"weight-{w}.txt",
                    *options,
                    f"placement-reuse-edges={out / 'topology.json'}",
                    f"reuse-sync-latency-cycles={w}",
                )
                grid.append(
                    dict(weight=w, **score, **{f"{k}_delta": v - baseline[k] for k, v in score.items()})
                )
            row.update(
                status="GRAPH_SCORE_RESOLVED",
                baseline=baseline,
                grid=grid,
                logical_edges=len(logical["edges"]),
                reuse_edges=len(topology["edges"]),
            )
        except (ValueError, KeyError, OSError) as error:
            row["error"] = str(error)
        rows.append(row)
        write_json(args.output / "predictions.json", rows)
        print(f"{i + 1}/{len(frozen)} {old['function']}/{old['arm']}: {row['status']}", flush=True)
    write_json(
        args.output / "prediction-seal.json",
        dict(
            predictions_sha256=digest(args.output / "predictions.json"),
            graph_opt_sha256=digest(args.graph_opt),
            prior_sha256=digest(args.prior / "predictions.json"),
            new_device_runs=0,
            retrospective_development_analysis=True,
        ),
    )
    # Read existing observations only after the structural run has been sealed.
    with args.existing_cell_table.open() as stream:
        observations = list(csv.DictReader(stream, delimiter="\t"))
    write_json(
        args.output / "existing-observations.json",
        dict(
            source=str(args.existing_cell_table),
            sha256=digest(args.existing_cell_table),
            rows=observations,
        ),
    )
    write_json(args.output / "comparison.json", compare_existing(rows, observations))
    return 0 if all(r["status"] == "GRAPH_SCORE_RESOLVED" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
