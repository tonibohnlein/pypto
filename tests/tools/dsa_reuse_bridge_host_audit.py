# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Audit the v3 bridge's archived model inputs, serially and without timing data.

Inputs are explicit reconstruction plans, schedules, PTO, native DSA problems,
and archived solutions. Each endpoint runs in a fresh output directory. No
compiler, solver, device runtime, or latency table is invoked or opened.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from pypto.tools import dsa_schedule_model as model
from pypto.tools.dsa_pto_isa_duration import PtoIsaDurationProvider

GRID = (0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
ISA_REVISION = "cd4a3d3f7a1a27fcfe536f617e9bca3008929664"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def graph(opt: Path, pto: Path, output: Path, *options: str) -> dict[str, int]:
    result = subprocess.run(
        [
            str(opt),
            str(pto),
            "-mlir-disable-threading",
            "-o",
            "/dev/null",
            "-pto-print-kernel-schedule-graph=" + " ".join(("format=text", *options)),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output.write_text(result.stdout)
    output.with_suffix(".err").write_text(result.stderr[-8000:])
    if result.returncode:
        errors = [line for line in result.stderr.splitlines() if "error:" in line]
        raise ValueError(f"PTOAS rc={result.returncode}: {(errors or [result.stderr[-1000:]])[0]}")
    fields = {}
    for name in ("longest_path_cycles", "recurrence_ii_cycles", "latency_lower_bound_cycles"):
        match = re.search(rf"\b{name}=(\d+)\b", result.stdout)
        if match is None:
            raise ValueError(f"PTOAS did not produce numeric {name}")
        fields[name] = int(match[1])
    return fields


def resolve_inputs(args: argparse.Namespace, row: dict, family: str) -> tuple[Path, Path, Path]:
    if family == "endpoints":
        base = args.four_candidate_root / "evidence"
        problem = base / "problems" / f"pypto_{row['function']}.dsa.json"
        solution = (
            base / "replay-maps" / row["cell"] / row["arm_dir"] / f"pypto_{row['function']}.dsa.solution.json"
        )
    else:
        parent = row["cell"].split("__", 1)[0]
        problem = args.captures / parent / "call-001" / f"pypto_{row['function']}.dsa.json"
        solution = (
            args.holdout_root
            / "evidence/replay-maps"
            / parent
            / row["capacity"]
            / row["arm_dir"]
            / f"pypto_{row['function']}.dsa.solution.json"
        )
    endpoint = args.archive_root / "artifacts" / family / f"{row['cell']}__{row['arm_dir']}"
    return problem, solution, endpoint


def audit_endpoint(args: argparse.Namespace, row: dict, family: str, durations: model.DurationModel) -> dict:
    problem, solution, endpoint = resolve_inputs(args, row, family)
    out = args.output / family / endpoint.name
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "cell": row["cell"],
        "function": row["function"],
        "script": row["script"],
        "arm": row["arm"],
        "family": family,
        "status": "INCOMPLETE",
    }
    pto = endpoint / "pto" / f"{row['function']}.pto"
    schedule_path = endpoint / "schedule.jsonl"
    try:
        paths = {"problem": problem, "solution": solution, "schedule": schedule_path, "pto": pto}
        result["inputs"] = {
            key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in paths.items()
        }
        record = json.loads(schedule_path.read_text().splitlines()[0])
        record = model.enrich_native_schedule_from_pto(record, pto.read_text(), pto_source=str(pto))
        write_json(out / "schedule.enriched.json", record)
        problem_doc = json.loads(problem.read_text())
        solution_doc = json.loads(solution.read_text())
        result["physical_pairs"] = model._realized_physical_reuse_pairs(problem_doc, solution_doc)
        base_graph = out / "graph.base.txt"
        if args.archive_graphs_only:
            archived_graph = endpoint / "graph.clean.txt"
            base_graph.write_bytes(archived_graph.read_bytes())
            result["graph_validation"] = "ARCHIVED_GRAPH_ONLY_CPP_CHANGES_NOT_VALIDATED"
        else:
            graph(args.ptoas, pto, base_graph)
        errors = {}
        for stage in ("topology", "durations"):
            try:
                document = (
                    model.emit_ptoas_placement_reuse_topology(
                        record, problem, solution, base_graph, function=row["function"]
                    )
                    if stage == "topology"
                    else model.emit_ptoas_resolved_node_durations(
                        record, durations, base_graph, function=row["function"]
                    )
                )
                write_json(out / f"{stage}.json", document)
                if stage == "topology":
                    result["reuse_edges"] = len(document["edges"])
                    result["reuse_recurrences"] = sum(
                        bool(e.get("iteration_distance")) for e in document["edges"]
                    )
                    result["non_edge_reuses"] = document["non_edge_reuses"]
                else:
                    result["duration_evidence"] = dict(
                        Counter(node["evidence_class"] for node in document["nodes"])
                    )
            except ValueError as error:
                errors[stage] = str(error)
        if errors:
            result["errors"] = errors
            return result
        if args.archive_graphs_only:
            result["status"] = "BRIDGE_INPUTS_COMPLETE"
            result["invocation_latency_complete"] = False
            return result
        duration_arg = f"node-durations={out / 'durations.json'}"
        baseline = graph(
            args.ptoas, pto, out / "graph.weighted-base.txt", duration_arg, "require-exact-durations=true"
        )
        result["baseline"] = baseline
        result["grid"] = []
        for weight in GRID:
            scored = graph(
                args.ptoas,
                pto,
                out / f"graph.w{weight}.txt",
                duration_arg,
                "require-exact-durations=true",
                f"placement-reuse-edges={out / 'topology.json'}",
                f"reuse-sync-latency-cycles={weight}",
            )
            result["grid"].append(
                {
                    "weight": weight,
                    "score": scored,
                    "delta": {key: value - baseline[key] for key, value in scored.items()},
                }
            )
        # These are structural iteration/recurrence bounds. A finite invocation
        # also requires branch selection, trip counts, and resource FIFO order.
        result["status"] = "STRUCTURAL_BOUNDS_COMPLETE"
        result["invocation_latency_complete"] = False
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive-root", "four-candidate-root", "holdout-root", "captures", "pto-isa-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ptoas", type=Path)
    parser.add_argument(
        "--archive-graphs-only",
        action="store_true",
        help="Validate Python joins against archived graphs; do not run any C++ tool.",
    )
    args = parser.parse_args()
    if not args.archive_graphs_only and args.ptoas is None:
        parser.error("--ptoas is required unless --archive-graphs-only is selected")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    provider = PtoIsaDurationProvider.from_checkout(
        args.pto_isa_root, expected_revision=ISA_REVISION, unsupported_policy="error"
    )
    durations = model.DurationModel(pto_isa_provider=provider)
    write_json(args.output / "provider.json", provider.to_json())
    rows = []
    for plan, family in (("fourcand-plan.json", "endpoints"), ("holdout-plan.json", "holdout")):
        for row in json.loads((args.archive_root / "artifacts" / plan).read_text()):
            result = audit_endpoint(args, row, family, durations)
            rows.append(result)
            write_json(args.output / "endpoints.json", rows)
            print(f"{result['cell']} / {result['arm']}: {result['status']}", flush=True)
    by_problem = defaultdict(list)
    for row in rows:
        by_problem[(row["script"], row["function"])].append(row)
    complete = [
        key
        for key, values in by_problem.items()
        if all(row["status"] == "STRUCTURAL_BOUNDS_COMPLETE" for row in values)
    ]
    summary = {
        "endpoint_statuses": dict(Counter(row["status"] for row in rows)),
        "complete_problem_bounds": complete,
        "complete_problem_bound_count": len(complete),
        "complete_invocation_score_count": 0,
        "verdict": "INVOCATION_MODEL_INCOMPLETE",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
