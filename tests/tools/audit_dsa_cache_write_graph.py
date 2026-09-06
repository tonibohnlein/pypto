"""Reproduce the cache-write graph miss using frozen artifacts, without timing.

The additional graph is a conformance diagnostic, not a device prediction.
No physical-address equality is used to infer semantic aliases.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypto.tools.dsa_graph_audit import dag_witness, parse_weighted_graph, single_writer_raw_dependencies


def reachable(source, target, edges):
    pending, visited = [source], set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(e["target"] for e in edges if e["source"] == node and not e.get("distance", 0))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--cypress", type=Path, required=True)
    parser.add_argument("--dsa-rp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nodes, base = parse_weighted_graph(args.base.read_text())
    schedule = json.loads(args.schedule.read_text())
    problem_path = args.cypress / "instrumented-problem.json"
    problem = json.loads(problem_path.read_text())
    catalog = json.loads(problem["metadata"]["allocation_accesses_v1"])
    required = single_writer_raw_dependencies(nodes, catalog, schedule)
    missing = [edge for edge in required if not reachable(edge["source"], edge["target"], base)]
    boundary = [
        e
        for e in base
        if e["kind"] == "control"
        and not nodes[e["source"]]["loop_depth"]
        and nodes[e["target"]]["loop_depth"]
    ]
    arms, scores, sources = {}, [], [args.base, args.schedule, problem_path]
    native_checks = []
    for arm, directory in (("cypress", args.cypress), ("dsa_rp_cg", args.dsa_rp)):
        topology_path = directory / "topology.json"
        sources.append(topology_path)
        topology = json.loads(topology_path.read_text())
        arms[arm] = topology["edges"]
        for path in sorted(directory.glob("graph.w*.txt")):
            weight = int(re.fullmatch(r"graph.w(\d+)\.txt", path.name)[1])
            exported_nodes, exported_edges = parse_weighted_graph(path.read_text())
            if exported_nodes != nodes:
                raise ValueError("arm changed base operation nodes or durations")
            witness = dag_witness(nodes, exported_edges)
            native = int(re.search(r"\blongest_path_cycles=(\d+)", path.read_text())[1])
            if witness["longest_path_cycles"] != native:
                raise ValueError("independent longest path differs from native exporter")
            native_checks.append(dict(arm=arm, weight=weight, cycles=native))
            selected = [
                dict(
                    source=e["source_node"],
                    target=e["target_node"],
                    distance=e.get("iteration_distance", 0),
                    latency=weight,
                )
                for e in topology["edges"]
            ]
            for label, edges in (("archived", base), ("required_raw_added_diagnostic", base + required)):
                baseline = dag_witness(nodes, edges)
                placed = dag_witness(nodes, edges + selected)
                scores.append(
                    dict(
                        arm=arm,
                        weight=weight,
                        graph=label,
                        baseline=baseline["longest_path_cycles"],
                        placed=placed["longest_path_cycles"],
                        penalty=placed["longest_path_cycles"] - baseline["longest_path_cycles"],
                        critical_path=placed["critical_path"],
                    )
                )
    output = dict(
        status="GRAPH_CONFORMANCE_FAIL_NOT_A_DEVICE_CAUSAL_EXPLANATION",
        scope="acyclic longest path only; dynamic invocation and recurrence composition not supplied",
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        missing_required_raw=missing,
        prelude_to_loop_control_edges=boundary,
        native_lp_crosschecks=native_checks,
        scores=scores,
        realized_edges=arms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            dict(
                missing_raw=len(missing),
                boundary_edges=len(boundary),
                native_crosschecks=len(native_checks),
                score_rows=len(scores),
            )
        )
    )


if __name__ == "__main__":
    main()
