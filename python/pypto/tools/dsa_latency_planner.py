# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded complete-placement greedy refinement using the existing DAG oracle.

This research search starts from a complete legal map and moves colocation
classes to aligned address boundaries. It is NOT the C++ canonical-greedy
algorithm. Both latency and structural objectives use precisely this search.
Pools remain fixed: moving an allocation between execution resources would
invalidate the frozen schedule. No compiler or device is invoked.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pypto.tools import dsa_schedule_model as schedule

Placement = dict[int, tuple[int, int]]
PlacementKey = tuple[tuple[int, int, int], ...]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**64:
        raise ValueError(f"{name}: expected integer in [{minimum}, 2**64), got {value!r}")
    return value


def _overlap(a: int, size_a: int, b: int, size_b: int) -> bool:
    return a < b + size_b and b < a + size_a


def placement_key(placements: Placement) -> PlacementKey:
    """Return a stable complete placement identity, independent of JSON order."""
    return tuple((bid, pool, offset) for bid, (pool, offset) in sorted(placements.items()))


def problem_fingerprint(document: Mapping[str, Any]) -> str:
    """Match the frozen solver's alias-normalized structured FNV-1a identity."""
    canonical = copy.deepcopy(dict(document))
    canonical.pop("problem_fingerprint", None)
    for alias in canonical["problem"].get("pypto_structure", {}).get("alias_classes", []):
        alias["members"] = [f"member_{i}" for i in range(len(alias["members"]))]
    value = 14695981039346656037
    for byte in (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode():
        value = ((value ^ byte) * 1099511628211) & (2**64 - 1)
    return f"{value:016x}"


class PlacementProblem:
    """Validate supported schema-v1 geometry and enumerate legal relocations.

    Colocation classes use their maximum extent for inter-class conflicts,
    matching the solver's hard-constraint contract. Structured pipeline members
    are kept at their seed positions. Unknown hard fields fail closed instead
    of being silently ignored.
    """

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = copy.deepcopy(dict(document))
        if document.get("schema_version") != 1:
            raise ValueError("planner requires schema_version=1")
        problem = document["problem"]
        known = {"buffers", "pools", "constraints", "cost_model", "objective", "pypto_structure"}
        if set(problem) - known:
            raise ValueError(f"unknown problem fields: {sorted(set(problem) - known)}")
        self.buffers = self._index(problem["buffers"])
        self.pools = self._index(problem["pools"])
        if not self.buffers or not self.pools:
            raise ValueError("problem must contain buffers and pools")
        structure = problem.get("pypto_structure", {})
        if set(structure) - {"alias_classes", "pipeline_groups"}:
            raise ValueError("unknown structured geometry fields")
        self.fixed_pipeline_buffers = {
            member["buffer"] for group in structure.get("pipeline_groups", []) for member in group["members"]
        }
        if self.fixed_pipeline_buffers - set(self.buffers):
            raise ValueError("pipeline group references unknown buffers")
        self._validate_geometry()
        constraints = problem["constraints"]
        pairs = self._constraint_pairs(constraints)
        roots = {bid: bid for bid in self.buffers}
        for a, b in sorted(pairs["colocations"]):
            old, new = max(roots[a], roots[b]), min(roots[a], roots[b])
            roots = {bid: new if root == old else root for bid, root in roots.items()}
        self.roots = roots
        self.groups = tuple(
            tuple(b for b in sorted(roots) if roots[b] == r) for r in sorted(set(roots.values()))
        )
        self.extents = {
            b: max(self.buffers[x]["size"] for x in group) for group in self.groups for b in group
        }
        self.pins = {}
        for pin in constraints.get("pinned_allocations", []):
            bid = pin["buffer"]
            if bid not in self.buffers or bid in self.pins or pin["pool"] not in self.pools:
                raise ValueError("invalid pinned allocation")
            _integer(pin["offset"], "pinned offset")
            if type(pin["exclusive_for_all_time"]) is not bool:
                raise ValueError("exclusive_for_all_time must be boolean")
            self.pins[bid] = pin
        self.conflicts = set(pairs["separations"])
        for a, b in itertools.combinations(sorted(self.buffers), 2):
            live = any(
                x["lower"] < y["upper"] and y["lower"] < x["upper"]
                for x in self.buffers[a]["live_intervals"]
                for y in self.buffers[b]["live_intervals"]
            )
            exclusive = any(self.pins.get(bid, {}).get("exclusive_for_all_time") for bid in (a, b))
            if roots[a] != roots[b] and ((live and (a, b) not in pairs["temporal_exclusions"]) or exclusive):
                self.conflicts.add((a, b))
        self.no_partial = pairs["no_partial_overlaps"]
        self.penalties = problem.get("cost_model", {}).get("reuse_penalties", [])
        for penalty in self.penalties:
            if penalty["first"] not in self.buffers or penalty["second"] not in self.buffers:
                raise ValueError("penalty references unknown buffer")
            _integer(penalty["cost"], "reuse cost")

    def _validate_geometry(self) -> None:
        for pool in self.pools.values():
            _integer(pool["capacity"], "finite pool capacity", 1)
            if set(pool) - {"id", "name", "capacity", "reserved_ranges", "bank_geometry"}:
                raise ValueError("unknown pool field")
            for interval in pool.get("reserved_ranges", []):
                lo = _integer(interval["begin"], "reserved begin")
                hi = _integer(interval["end"], "reserved end", 1)
                if not lo < hi <= pool["capacity"]:
                    raise ValueError("invalid reserved range")
        for buffer in self.buffers.values():
            if set(buffer) - {"id", "name", "size", "alignment", "allowed_pools", "live_intervals"}:
                raise ValueError("unknown buffer field")
            _integer(buffer["size"], "buffer size", 1)
            _integer(buffer["alignment"], "buffer alignment", 1)
            allowed = buffer["allowed_pools"]
            if not allowed or any(pool not in self.pools for pool in allowed):
                raise ValueError("invalid allowed pools")
            if not buffer["live_intervals"]:
                raise ValueError("empty lifetime")
            for interval in buffer["live_intervals"]:
                if any(type(interval[k]) is not int for k in ("lower", "upper")):
                    raise ValueError("lifetime bounds must be integers")
                if interval["lower"] >= interval["upper"]:
                    raise ValueError("invalid lifetime")

    def _constraint_pairs(self, constraints: Mapping[str, Any]) -> dict[str, set[tuple[int, int]]]:
        names = {
            "colocations",
            "separations",
            "no_partial_overlaps",
            "temporal_exclusions",
            "pinned_allocations",
        }
        if set(constraints) - names:
            raise ValueError("unknown hard constraints")
        pairs = {}
        for name in names - {"pinned_allocations"}:
            pairs[name] = set()
            for pair in constraints.get(name, []):
                if set(pair) - {"first", "second", "reasons"}:
                    raise ValueError(f"unknown {name} pair fields")
                a, b = pair["first"], pair["second"]
                if a not in self.buffers or b not in self.buffers or a == b:
                    raise ValueError(f"invalid {name} pair")
                pairs[name].add(tuple(sorted((a, b))))
        return pairs

    @staticmethod
    def _index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        result = {}
        for row in rows:
            bid = _integer(row["id"], "id")
            if bid in result:
                raise ValueError(f"duplicate id {bid}")
            result[bid] = row
        return result

    def read_solution(self, solution: Mapping[str, Any]) -> Placement:
        """Decode a complete solution and reject any illegal seed."""
        if solution.get("instance") != self.document.get("instance"):
            raise ValueError("solution instance mismatch")
        if solution.get("profile") != self.document.get("profile"):
            raise ValueError("solution profile mismatch")
        fingerprint = solution.get("problem_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 16:
            raise ValueError("solution requires a 16-digit problem fingerprint")
        if any(char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError("malformed solution problem fingerprint")
        if self.document.get("problem_fingerprint", fingerprint) != fingerprint:
            raise ValueError("solution problem fingerprint mismatch")
        if fingerprint != problem_fingerprint(self.document):
            raise ValueError("solution fingerprint differs from the actual problem semantics")
        placements = {}
        for row in solution["placements"]:
            bid = _integer(row["buffer"], "placement buffer")
            if bid in placements:
                raise ValueError("duplicate placed buffer")
            placements[bid] = (
                _integer(row["pool"], "placement pool"),
                _integer(row["offset"], "placement offset"),
            )
        if not self.valid(placements):
            raise ValueError("seed is not a complete legal placement")
        return placements

    def valid(self, placements: Placement) -> bool:
        """Check capacities, reservations, hard pairs, lifetimes, aliases and pins."""
        if set(placements) != set(self.buffers):
            return False
        for bid, (pool, offset) in placements.items():
            buffer = self.buffers[bid]
            if (
                type(pool) is not int
                or type(offset) is not int
                or offset < 0
                or pool not in buffer["allowed_pools"]
                or offset % buffer["alignment"]
                or offset + buffer["size"] >= 2**64
            ):
                return False
            if offset + buffer["size"] > self.pools[pool]["capacity"]:
                return False
            if any(
                _overlap(offset, buffer["size"], r["begin"], r["end"] - r["begin"])
                for r in self.pools[pool].get("reserved_ranges", [])
            ):
                return False
            pin = self.pins.get(bid)
            if pin and (pool, offset) != (pin["pool"], pin["offset"]):
                return False
        for group in self.groups:
            if len({placements[b] for b in group}) != 1:
                return False
        for a, b in self.conflicts | self.no_partial:
            pa, oa = placements[a]
            pb, ob = placements[b]
            if pa != pb or not _overlap(oa, self.extents[a], ob, self.extents[b]):
                continue
            if (a, b) in self.conflicts or (oa, self.extents[a]) != (ob, self.extents[b]):
                return False
        return True

    def candidates(self, placements: Placement) -> Iterator[Placement]:
        """Yield deterministic aligned boundary moves, holding pools fixed."""
        for group in self.groups:
            if any(bid in self.pins or bid in self.fixed_pipeline_buffers for bid in group):
                continue
            pool = placements[group[0]][0]
            extent = self.extents[group[0]]
            alignment = math.lcm(*(self.buffers[b]["alignment"] for b in group))
            limit = self.pools[pool]["capacity"] - extent
            boundaries = {0, limit}
            for bid, (other_pool, offset) in placements.items():
                if bid not in group and other_pool == pool:
                    end = offset + self.extents[bid]
                    boundaries.update((offset, end, offset - extent, end - extent))
            for reserved in self.pools[pool].get("reserved_ranges", []):
                boundaries.update((reserved["begin"] - extent, reserved["end"]))
            offsets = {
                aligned
                for x in boundaries
                for aligned in (x // alignment * alignment, -(-x // alignment) * alignment)
                if 0 <= aligned <= limit
            }
            for offset in sorted(offsets):
                if offset == placements[group[0]][1]:
                    continue
                candidate = dict(placements)
                candidate.update({bid: (pool, offset) for bid in group})
                yield candidate

    def structural_cost(self, placements: Placement) -> float:
        """Compute the existing sum-of-realized-penalties objective."""
        result = 0
        for row in self.penalties:
            a, b = row["first"], row["second"]
            pa, oa = placements[a]
            pb, ob = placements[b]
            if (
                self.roots[a] != self.roots[b]
                and pa == pb
                and _overlap(oa, self.extents[a], ob, self.extents[b])
            ):
                result += row["cost"]
        return float(result)


def refine_placement(
    problem: PlacementProblem,
    seed: Placement,
    score: Callable[[Placement], float],
    *,
    max_evaluations: int = 128,
    max_candidates: int = 10000,
) -> dict[str, Any]:
    """Greedily choose strictly improving complete legal relocations.

    Scores are memoized by the complete map, never by independent pair cost.
    There are no sideways moves, perturbations or inferred global optima.
    A bounded prefix of a neighborhood may be used; truncation is reported.
    The final score is recomputed uncached as a consistency gate.
    """
    _integer(max_evaluations, "max_evaluations", 1)
    _integer(max_candidates, "max_candidates", 1)
    if not problem.valid(seed):
        raise ValueError("search requires a legal complete seed")
    cache: dict[PlacementKey, float] = {}
    hits = 0

    def evaluate(placements: Placement) -> float:
        nonlocal hits
        key = placement_key(placements)
        if key in cache:
            hits += 1
            return cache[key]
        value = float(score(placements))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid placement score {value}")
        cache[key] = value
        return value

    current = dict(seed)
    initial_score = current_score = evaluate(current)
    examined = 0
    trace = []
    status = "LOCAL_NEIGHBORHOOD_EXHAUSTED"
    while True:
        best, best_score = current, current_score
        exhausted = False
        for candidate in problem.candidates(current):
            if examined >= max_candidates or len(cache) >= max_evaluations:
                exhausted = True
                break
            examined += 1
            if not problem.valid(candidate):
                continue
            value = evaluate(candidate)
            if value < best_score:
                best, best_score = candidate, value
        if best_score < current_score:
            trace.append({"before": current_score, "after": best_score, "placements": placement_key(best)})
            current, current_score = best, best_score
        elif not exhausted:
            break
        if exhausted:
            status = "BUDGET_EXHAUSTED"
            break
    verified = float(score(current))
    if not problem.valid(current) or verified != current_score:
        raise ValueError("final uncached score/legality verification failed")
    return {
        "search": "complete_placement_greedy_relocation_v1",
        "status": status,
        "placements": current,
        "initial_score": initial_score,
        "final_score": current_score,
        "score_evaluations": len(cache),
        "verification_evaluations": 1,
        "cache_hits": hits,
        "candidates_examined": examined,
        "accepted_moves": trace,
    }


def solution_document(seed: Mapping[str, Any], placements: Placement, solver: str) -> dict[str, Any]:
    """Produce a replay-shaped solution without stale solver-cost metadata."""
    return {
        "schema_version": 1,
        "profile": seed["profile"],
        "instance": seed["instance"],
        "problem_fingerprint": seed["problem_fingerprint"],
        "metadata": {"solver": solver},
        "placements": [
            {"buffer": bid, "pool": pool, "offset": offset} for bid, pool, offset in placement_key(placements)
        ],
    }


class LatencyObjective:
    """Bind a frozen schedule/model to the complete physical-placement scorer."""

    def __init__(
        self,
        record: Mapping[str, Any],
        model: schedule.DurationModel,
        problem_path: Path,
        graph_path: Path,
        seed: Mapping[str, Any],
        workspace: Path,
    ) -> None:
        if record.get("runtime_branch_profile") or record.get("runtime_parallel_branch_profile"):
            raise ValueError("captured runtime profiles are analysis-only, not planner input")
        if not math.isfinite(model.sync_latency_cycles) or model.sync_latency_cycles <= 0:
            raise ValueError("planner requires one finite positive synchronization weight")
        _, evidence, _ = schedule.estimate_node_durations(record, model)
        if any(row.get("fallback") for row in evidence.values()):
            raise ValueError("planner duration coverage contains fallback nodes")
        self.record, self.model = copy.deepcopy(record), copy.deepcopy(model)
        self.problem_path, self.graph_path = problem_path, graph_path
        self.seed = copy.deepcopy(seed)
        self.workspace = workspace
        workspace.mkdir(parents=True, exist_ok=False)
        self.input_hashes = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (problem_path, graph_path)
        }
        self.last_result: dict[str, Any] = {}
        self.evidence_classes = sorted({row["evidence_class"] for row in evidence.values()})

    def __call__(self, placements: Placement) -> float:
        if any(
            hashlib.sha256(path.read_bytes()).hexdigest() != digest
            for path, digest in self.input_hashes.items()
        ):
            raise ValueError("frozen planner problem/graph changed during search")
        candidate_path = self.workspace / "candidate.solution.json"
        candidate_path.write_text(json.dumps(solution_document(self.seed, placements, "latency-greedy")))
        result = schedule.score_physical_placement_dag(
            self.record, self.model, self.problem_path, candidate_path, self.graph_path
        )
        if result.get("status") != "COMPLETE":
            raise ValueError(f"MODEL_INELIGIBLE: {result.get('status')}: {result.get('limitations')}")
        value = result.get("critical_path_extension_cycles")
        if value is None:
            raise ValueError("MODEL_INELIGIBLE: branch-dependent score has no single static objective")
        # The public scorer reports additive diagnostics for legacy callers;
        # this adapter deliberately does not compute a singleton-pair model.
        for key in ("pairwise_additive_cost_cycles", "nonadditive_interaction_cycles"):
            result.pop(key, None)
        self.last_result = result
        return float(value)


def main(argv: list[str] | None = None) -> int:
    """Run bounded host-only search and write replay solution plus audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--seed-solution", type=Path, required=True)
    parser.add_argument("--objective", choices=("latency", "structural"), required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=10000)
    args = parser.parse_args(argv)
    problem_doc = json.loads(args.problem.read_text())
    seed_doc = json.loads(args.seed_solution.read_text())
    problem = PlacementProblem(problem_doc)
    seed = problem.read_solution(seed_doc)
    args.output_root.mkdir(parents=True, exist_ok=False)
    inputs = {"problem": args.problem, "seed": args.seed_solution}
    objective: Callable[[Placement], float] = problem.structural_cost
    oracle = None
    if args.objective == "latency":
        if not all((args.schedule, args.graph, args.model)):
            parser.error("latency objective requires --schedule, --graph and --model")
        records = schedule.load_schedule_graphs(args.schedule)
        record = records[problem_doc["instance"]]
        model = schedule.DurationModel.from_json(json.loads(args.model.read_text()))
        oracle = LatencyObjective(
            record, model, args.problem, args.graph, seed_doc, args.output_root / "oracle"
        )
        objective = oracle
        inputs.update(schedule=args.schedule, graph=args.graph, model=args.model)
    input_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}
    result = refine_placement(
        problem, seed, objective, max_evaluations=args.max_evaluations, max_candidates=args.max_candidates
    )
    solution = solution_document(seed_doc, result.pop("placements"), f"dsa-rp-{args.objective}-relocation")
    if input_hashes != {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}:
        raise ValueError("frozen planner inputs changed during search")
    result.update(
        objective=args.objective,
        seed_policy="explicit_complete_seed_no_timing_selection",
        structural_cost=problem.structural_cost(problem.read_solution(solution)),
        input_sha256=input_hashes,
    )
    if oracle is not None:
        result["complete_placement_score"] = oracle.last_result
        result["duration_evidence_classes"] = oracle.evidence_classes
    (args.output_root / "solution.json").write_text(json.dumps(solution, indent=2, sort_keys=True) + "\n")
    (args.output_root / "search.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
