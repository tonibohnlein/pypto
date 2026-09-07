# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the latency search on existing local exports; never read timing tables."""

import argparse
import json
from pathlib import Path

from pypto.tools import dsa_latency_planner as planner
from pypto.tools.dsa_pto_isa_duration import PtoIsaDurationProvider
from pypto.tools.dsa_schedule_model import DurationModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--pto-isa-root", type=Path, required=True)
    parser.add_argument("--pto-isa-revision", required=True)
    parser.add_argument("--sync-cycles", type=float, required=True)
    parser.add_argument("--function", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=16)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    duration = DurationModel(
        sync_latency_cycles=args.sync_cycles,
        pto_isa_provider=PtoIsaDurationProvider.from_checkout(
            args.pto_isa_root, expected_revision=args.pto_isa_revision
        ),
    )
    model_path = args.output_root / "model.json"
    model_path.write_text(json.dumps(duration.to_json(), indent=2, sort_keys=True) + "\n")
    results = []
    for function in args.function:
        candidates = sorted(
            (args.analysis_root / "complete-placement-analysis").glob(f"*/{function}/result.json")
        )
        selected = None
        for candidate in candidates:
            record = json.loads(candidate.read_text())
            seed = (
                args.analysis_root
                / "reconstructed-maps"
                / record["map_digest"]
                / f"pypto_{function}.dsa.solution.json"
            )
            if not seed.is_file():
                continue
            solution = json.loads(seed.read_text())
            if solution.get("metadata", {}).get("solver") in {
                "geometry-first-fit",
                "first-fit",
                "geometry-canonical-greedy",
            }:
                selected = candidate, record, seed
                break
        if selected is None:
            results.append({"function": function, "status": "NO_GEOMETRY_SEED"})
            continue
        candidate, record, seed = selected
        common = [
            "--problem",
            str(args.analysis_root / record["problem"]),
            "--seed-solution",
            str(seed),
            "--model",
            str(model_path),
            "--schedule",
            str(candidate.parent / "official-v057-schedule.jsonl"),
            "--graph",
            str(candidate.parent / "research-graph.txt"),
            "--max-evaluations",
            str(args.max_evaluations),
        ]
        for objective in ("structural", "latency"):
            out = args.output_root / f"{function}-{objective}"
            try:
                planner.main([*common, "--objective", objective, "--output-root", str(out)])
                report = json.loads((out / "search.json").read_text())
                results.append({"function": function, "objective": objective, **report})
            except (ValueError, KeyError) as error:
                results.append(
                    {
                        "function": function,
                        "objective": objective,
                        "status": "MODEL_OR_INPUT_BLOCKED",
                        "reason": str(error),
                    }
                )
    (args.output_root / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            [
                {
                    key: row.get(key)
                    for key in ("function", "objective", "status", "initial_score", "final_score", "reason")
                }
                for row in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
