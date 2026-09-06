"""Bind frozen mixed-core transfer sites to weighted child graphs, without timing."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from pypto.tools.dsa_graph_audit import frontend_c2v_dependencies, orchestration_tensor_dependencies
from score_frozen_dsa_graph_manifest import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--consumer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(dict)
    for row in json.loads((args.scores / "predictions.json").read_text()):
        if row["function"] in (args.producer, args.consumer):
            key = row["capacity"], row["arm"]
            if row["function"] in groups[key]:
                raise ValueError("duplicate mixed function")
            groups[key][row["function"]] = row
    result = []
    for (capacity, arm), functions in sorted(groups.items()):
        if set(functions) != {args.producer, args.consumer}:
            raise ValueError("mixed group is missing a function")
        schedules, graphs = {}, {}
        sources = set()
        for name, row in functions.items():
            if row["status"] != "GRAPH_SCORE_RESOLVED":
                raise ValueError(f"unresolved mixed child {name}/{capacity}/{arm}")
            path = Path(row["pto"])
            if digest(path) != row["input_sha256"]["pto"]:
                raise ValueError("frozen mixed PTO changed")
            sources.add(path.read_text())
            root = Path(row["output"])
            schedules[name] = json.loads((root / "schedule.json").read_text())
            graphs[name] = (root / "baseline.txt").read_text()
        if len(sources) != 1:
            raise ValueError("mixed children must originate in one unchanged module")
        evidence = frontend_c2v_dependencies(sources.pop(), args.producer, args.consumer, schedules, graphs)
        root = Path(functions[args.producer]["pto"]).parent.parent
        config = (root / "kernel_config.py").read_text()
        names = {int(i): name for i, name in re.findall(r'"func_id":\s*(\d+),\s*"name":\s*"([^"]+)"', config)}
        orchestrations = list((root / "orchestration").glob("*.cpp"))
        if len(orchestrations) != 1:
            raise ValueError("requires one frozen orchestration source")
        evidence["orchestration"] = orchestration_tensor_dependencies(orchestrations[0].read_text(), names)
        result.append(dict(capacity=capacity, arm=arm, **evidence))
    if not result:
        raise ValueError("empty mixed transfer audit")
    write_json(
        args.output,
        dict(predictions_sha256=digest(args.scores / "predictions.json"), timing_read=False, cells=result),
    )


if __name__ == "__main__":
    main()
