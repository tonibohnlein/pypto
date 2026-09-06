"""Score frozen DSA endpoints without reading latency or launching devices.

The input manifest names exact placed PTO, problem, replay solution and a
metadata-only re-export of the same problem for each endpoint. The result is
a per-function DAG/recurrence *lower bound*, not an invocation-time estimate.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pypto.tools import dsa_schedule_model as model
from pypto.tools.dsa_pto_isa_duration import PtoIsaDurationProvider

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".claude/skills/incore-profiling"))
from prepare_dsa_ablation import _fingerprint, _validate_envelope  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run_graph(binary: Path, source: Path, output: Path, *options: str) -> dict[str, int]:
    command = [
        str(binary),
        str(source),
        "-mlir-disable-threading",
        "-o",
        "/dev/null",
        "-pto-print-kernel-schedule-graph=" + " ".join(("format=text", *options)),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    output.write_text(completed.stdout)
    output.with_suffix(".stderr").write_text(completed.stderr)
    if completed.returncode:
        raise ValueError(f"graph export failed: {completed.stderr[-800:]}")
    if completed.stdout.count("KernelScheduleGraph @") != 1:
        raise ValueError("requires exactly one function graph; mixed-function composition is not implied")
    result = {}
    for key in ("longest_path_cycles", "recurrence_ii_cycles", "latency_lower_bound_cycles"):
        match = re.search(rf"\b{key}=(\d+)\b", completed.stdout)
        if match is None:
            raise ValueError(f"graph has no resolved numeric {key}")
        result[key] = int(match[1])
    return result


def product_evidence(args, index, paths, row, cache_rows, destination):
    """Keep product-codegen evidence separate from analysis-only graph lowering."""
    cpp = destination / "analysis.cpp"
    if cache_rows is not None:
        cached = cache_rows[index]
        if cached.get("input_sha256") != row["input_sha256"] or not cached.get("generated_cpp_sha256"):
            raise ValueError("product cache source mismatch or unsuccessful compile")
        text = (args.product_cache / f"endpoint-{index:03d}" / "insertsync.log").read_text()
        row["generated_cpp_sha256"] = cached["generated_cpp_sha256"]
        row["product_evidence"] = "cached_product_compile_reimported"
    else:
        completed = subprocess.run(
            [
                str(args.ptoas),
                str(paths["pto"]),
                "-o",
                str(cpp),
                "--enable-insert-sync",
                "--pto-level=level3",
                "--pto-arch",
                "a3",
                "--pto-insert-sync-debug=3",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (destination / "insertsync.log").write_text(completed.stderr)
        if completed.returncode:
            raise ValueError(f"product assembler rejected frozen input: {completed.stderr[-800:]}")
        row["generated_cpp_sha256"] = digest(cpp)
        cpp.unlink()
        text = completed.stderr
        row["product_evidence"] = "fresh_product_compile"
    (destination / "insertsync.log").write_text(text)
    row["product_debug_sha256"] = digest(destination / "insertsync.log")
    return text


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--instrumented-directory",
        type=Path,
        help="Metadata-only re-export; solver-visible fields must match frozen problems",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ptoas", type=Path, required=True)
    parser.add_argument("--graph-opt", type=Path, required=True)
    parser.add_argument("--pto-isa-root", type=Path, required=True)
    parser.add_argument("--pto-isa-revision", required=True)
    parser.add_argument("--duration-model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument(
        "--product-cache",
        type=Path,
        help="Reuse product logs after checking manifest, assembler and source hashes; no recompiles",
    )
    parser.add_argument("--weights", default="0,8,16,24,32,48,64,96,128,160,192,256")
    parser.add_argument(
        "--repair-base-graph",
        action="store_true",
        help="Apply logical-allocation dependencies and semantic structured boundaries",
    )
    parser.add_argument(
        "--schedule-records",
        type=Path,
        help="Rejoin existing capacity/arm/final/function.schedule.json records to raw PTO",
    )
    parser.add_argument(
        "--lower-frontend-pipes",
        action="store_true",
        help="Lower frontend pipe operations in an analysis copy",
    )
    parser.add_argument(
        "--conservative-logical-ranges",
        action="store_true",
        help="Explicitly overapproximate unresolved subranges by their logical allocation",
    )
    args = parser.parse_args()
    weights = [int(value) for value in args.weights.split(",")]
    if not weights or len(set(weights)) != len(weights) or min(weights) < 0:
        parser.error("weights must be unique nonnegative integers")
    args.output.mkdir(parents=True, exist_ok=False)
    return args, weights


def main() -> int:
    args, weights = parse_args()
    provider = PtoIsaDurationProvider.from_checkout(
        args.pto_isa_root, expected_revision=args.pto_isa_revision, unsupported_policy="error"
    )
    duration_data = json.loads(args.duration_model.read_text())
    # Override the provider explicitly; retain pinned signature observations.
    durations = model.DurationModel.from_json(duration_data)
    durations.pto_isa_provider = provider
    durations = model.calibrate_from_metrics(args.calibration, durations)
    write_json(args.output / "provider.json", provider.to_json())
    manifest = json.loads(args.manifest.read_text())
    cache_rows = None
    if args.product_cache:
        cache_provenance = json.loads((args.product_cache / "provenance.json").read_text())
        for key, expected in (
            ("manifest_sha256", digest(args.manifest)),
            ("ptoas_sha256", digest(args.ptoas)),
        ):
            if cache_provenance.get(key) != expected:
                raise ValueError(f"product cache {key} mismatch")
        cache_rows = json.loads((args.product_cache / "predictions.json").read_text())
        if len(cache_rows) != len(manifest):
            raise ValueError("product cache endpoint count mismatch")
    rows = []
    for index, item in enumerate(manifest):
        destination = args.output / f"endpoint-{index:03d}"
        destination.mkdir()
        row = {**item, "status": "INCOMPLETE", "output": str(destination)}
        try:
            paths = {key: Path(item[key]) for key in ("pto", "problem", "solution", "instrumented_problem")}
            row["input_sha256"] = {key: digest(path) for key, path in paths.items()}
            problem = json.loads(paths["problem"].read_text())
            solution = json.loads(paths["solution"].read_text())
            if solution.get("problem_fingerprint") != _fingerprint(problem):
                raise ValueError("frozen replay fingerprint does not match original problem")
            _validate_envelope(problem, solution)
            instrumented_path = (
                args.instrumented_directory / f"pypto_{item['function']}.dsa.json"
                if args.instrumented_directory
                else paths["instrumented_problem"]
            )
            instrumented = json.loads(instrumented_path.read_text())
            row["analysis_metadata_sha256"] = digest(instrumented_path)
            old_fields, new_fields = dict(problem["problem"]), dict(instrumented["problem"])
            old_fields.pop("pypto_structure", None)
            new_fields.pop("pypto_structure", None)
            if old_fields != new_fields:
                raise ValueError("metadata re-export changed solver-visible problem fields")
            problem.setdefault("metadata", {})["allocation_accesses_v1"] = instrumented["metadata"][
                "allocation_accesses_v1"
            ]
            bound_problem = destination / "instrumented-problem.json"
            write_json(bound_problem, problem)
            debug_text = product_evidence(args, index, paths, row, cache_rows, destination)
            if args.schedule_records:
                saved = (
                    args.schedule_records
                    / item["capacity"]
                    / item["arm"]
                    / "final"
                    / f"{item['function']}.schedule.json"
                )
                schedule = model.enrich_native_schedule_from_pto(
                    json.loads(saved.read_text()), paths["pto"].read_text(), pto_source=str(paths["pto"])
                )
                row["schedule_record_sha256"] = digest(saved)
                row["schedule_evidence"] = "existing_record_rejoined_to_unchanged_full_module"
            else:
                schedule = model.import_insert_sync_debug(
                    debug_text, function=item["function"], pto_text=paths["pto"].read_text()
                )
            write_json(destination / "schedule.json", schedule)
            graph_path = destination / "graph.base.txt"
            function_option = f"function-name={item['function']}"
            graph_source = paths["pto"]
            if args.lower_frontend_pipes:
                graph_source = destination / "lowered.pto"
                lower = subprocess.run(
                    [
                        str(args.graph_opt),
                        str(paths["pto"]),
                        "-mlir-disable-threading",
                        "-pto-lower-frontend-pipe-ops",
                        "-mlir-print-debuginfo",
                        "-mlir-print-local-scope",
                        "-o",
                        str(graph_source),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if lower.returncode:
                    raise ValueError(f"analysis frontend pipe lowering failed: {lower.stderr[-800:]}")
                row["analysis_lowered_pto_sha256"] = digest(graph_source)
            base_options = [function_option]
            if args.repair_base_graph:
                base_options.append("semantic-boundaries=true")
                skeleton = destination / "graph.skeleton.txt"
                run_graph(args.graph_opt, graph_source, skeleton, *base_options)
                logical = model.emit_ptoas_logical_memory_topology(
                    schedule,
                    bound_problem,
                    skeleton,
                    function=item["function"],
                    conservative_ranges=args.conservative_logical_ranges,
                )
                logical_path = destination / "logical-memory.json"
                write_json(logical_path, logical)
                base_options.append(f"logical-memory-edges={logical_path}")
                row["logical_memory_edge_count"] = len(logical["edges"])
                row["logical_range_precision"] = logical["range_precision"]
            run_graph(args.graph_opt, graph_source, graph_path, *base_options)
            topology = model.emit_ptoas_placement_reuse_topology(
                schedule, bound_problem, paths["solution"], graph_path, function=item["function"]
            )
            topology_path = destination / "topology.json"
            write_json(topology_path, topology)
            node_durations = model.emit_ptoas_resolved_node_durations(
                schedule, durations, graph_path, function=item["function"]
            )
            duration_path = destination / "durations.json"
            write_json(duration_path, node_durations)
            duration_options = (
                *base_options,
                f"node-durations={duration_path}",
                "require-exact-durations=true",
            )
            baseline = run_graph(
                args.graph_opt, graph_source, destination / "baseline.txt", *duration_options
            )
            grid = []
            for weight in weights:
                score = run_graph(
                    args.graph_opt,
                    graph_source,
                    destination / f"weight-{weight}.txt",
                    *duration_options,
                    f"placement-reuse-edges={topology_path}",
                    f"reuse-sync-latency-cycles={weight}",
                )
                grid.append(
                    {
                        "weight": weight,
                        **score,
                        **{key + "_delta": value - baseline[key] for key, value in score.items()},
                    }
                )
            row.update(
                status="GRAPH_SCORE_RESOLVED",
                baseline=baseline,
                grid=grid,
                duration_evidence=dict(Counter(node["evidence_class"] for node in node_durations["nodes"])),
                reuse_edge_count=len(topology["edges"]),
                physical_pair_count=topology["realized_physical_pair_count"],
                invocation_model_complete=False,
            )
        except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
            row["error"] = str(error)
        rows.append(row)
        write_json(args.output / "predictions.json", rows)
        print(f"{index + 1}/{len(manifest)} {item['function']}/{item['arm']}: {row['status']}", flush=True)
    write_json(
        args.output / "provenance.json",
        {
            "manifest_sha256": digest(args.manifest),
            "weights": weights,
            "duration_model_sha256": digest(args.duration_model),
            "calibration_sha256": {str(path): digest(path) for path in args.calibration},
            "ptoas_sha256": digest(args.ptoas),
            "graph_opt_sha256": digest(args.graph_opt),
            "timing_read": False,
            "new_device_runs": 0,
            "repair_base_graph": args.repair_base_graph,
            "lower_frontend_pipes": args.lower_frontend_pipes,
            "conservative_logical_ranges": args.conservative_logical_ranges,
            "interpretation": (
                "per-function DAG and single-cycle recurrence lower bounds; not invocation latency"
            ),
        },
    )
    return 0 if all(row["status"] == "GRAPH_SCORE_RESOLVED" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
