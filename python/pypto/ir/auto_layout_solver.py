# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Greedy heuristic and exact treewidth dynamic-programming backends for layout selection.

One variable selects a complete `OperatorConfig`. A tensor factor charges each
distinct requested representation once, including fixed external-output demands.
- `solve_greedy` follows Section 4.1 of arXiv:2608.21555: topological
  initialization followed by coordinate descent on the full objective.
- `compute_treewidth` computes min-fill tree decompositions and treewidth
  diagnostics (`dataflow_treewidth`, `primal_treewidth`, `active_primal_treewidth`,
  and DP state-space bounds) for a `LayoutProblemInstance`.
- `solve_treewidth_dp` follows Section 4.2 of arXiv:2608.21555: exact dynamic
  programming over a tree decomposition of the operation interaction graph.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .auto_layout import (
    CostBreakdown,
    LayoutAssignment,
    LayoutChoice,
    LayoutOperationNode,
    LayoutProblemInstance,
    MemorySpace,
    OperatorConfig,
    _conversion_multiplier,
    _operator_cost_multiplier,
    evaluate_assignment,
)


def _require_integer(value: object, name: str, minimum: int | None = None) -> None:
    """Validate runtime inputs, including values arriving from untyped callers."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")


@dataclass(frozen=True)
class LayoutSolverConfig:
    """Deterministic search limits, with costs expressed in modeled cycles.

    Attributes:
        max_evaluations: Candidate evaluations across initialization and search.
        max_greedy_sweeps: Bound on full coordinate-descent sweeps.
        max_treewidth: Maximum active primal treewidth for exact treewidth DP (default 8).
        max_bag_states: Maximum states allowed in a single treewidth-DP bag.
        tolerance: Absolute minimum improvement in cycles; ties retain the incumbent.
    """

    max_evaluations: int = 100_000
    max_greedy_sweeps: int = 100
    max_treewidth: int = 8
    max_bag_states: int = 100_000
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in ("max_evaluations", "max_greedy_sweeps", "max_bag_states"):
            _require_integer(getattr(self, name), name, 1)
        _require_integer(self.max_treewidth, "max_treewidth", 0)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError(f"tolerance must be finite and nonnegative, got {self.tolerance!r}")


@dataclass(frozen=True)
class LayoutConversion:
    """One shared conversion required by the scored assignment.

    Boundary conversions need caller-side packing/unpacking. Internal conversions
    need a future IR materializer; the current writer cannot emit them. The cost
    already includes the value's dynamic execution/reuse multiplier.
    """

    value_name: str
    source_layout: LayoutChoice
    target_layout: LayoutChoice
    cost_cycles: float
    consumer_op_ids: tuple[str, ...]
    is_boundary: bool


@dataclass(frozen=True)
class TreewidthStats:
    """Treewidth and DP state-space diagnostics for a `LayoutProblemInstance`.

    Attributes:
        num_values: Total number of `LayoutValueNode`s in the instance.
        num_operations: Total number of `LayoutOperationNode`s in the instance.
        num_movable_operations: Operations with more than one legal `OperatorConfig`.
        dataflow_treewidth: Treewidth of the producer-consumer graph (without co-consumer cliques).
        primal_treewidth: Treewidth of the full primal factor graph (cliques on shared tensors).
        active_primal_treewidth: Treewidth of the primal subgraph induced by movable operations.
        max_bag_size: Largest bag cardinality $|B_k|$ in the primal tree decomposition.
        max_active_bag_size: Largest number of movable operations in any primal bag.
        max_bag_domain_product: Maximum DP state count $\\prod_{u \\in B_k} |K_u|$ across bags.
        total_dp_states: Sum of $\\prod_{u \\in B_k} |K_u|$ across all bags in the decomposition.
        elimination_order: Operation IDs in min-fill elimination order.
        bags: Tuple of bags (each a tuple of operation IDs) in elimination order.
    """

    num_values: int
    num_operations: int
    num_movable_operations: int
    dataflow_treewidth: int
    primal_treewidth: int
    active_primal_treewidth: int
    max_bag_size: int
    max_active_bag_size: int
    max_bag_domain_product: int
    total_dp_states: int
    elimination_order: tuple[str, ...] = ()
    bags: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class LayoutSolverStats:
    """Search accounting and structural treewidth metadata."""

    evaluated_moves: int
    moves_applied: int
    greedy_sweeps: int
    initial_cost_cycles: float
    termination_reason: str
    treewidth: int = 0
    active_treewidth: int = 0


@dataclass(frozen=True)
class LayoutSolverResult:
    """Best modeled solution, search diagnostics, and its writeback requirements."""

    breakdown: CostBreakdown
    stats: LayoutSolverStats
    conversions: tuple[LayoutConversion, ...]
    writeback_issues: tuple[str, ...]
    treewidth_stats: TreewidthStats | None = None

    @property
    def assignment(self) -> LayoutAssignment:
        """Return the canonical assignment, which may require new conversion nodes."""
        return self.breakdown.assignment

    @property
    def total_cost_cycles(self) -> float:
        """Return the objective of the retained solution."""
        return self.breakdown.total_cost_cycles

    def writeback_assignment(self) -> LayoutAssignment:
        """Return an assignment representable by the existing annotation writer.

        Raises:
            ValueError: An internal conversion or multiple GM representations
                cannot be expressed by the current writer.
        """
        if self.writeback_issues:
            raise ValueError(
                f"Layout solution requires conversion materialization: {'; '.join(self.writeback_issues)}"
            )
        return self.assignment


class _EvaluationLimit(Exception):
    """Internal bounded-search exit; the current incumbent remains valid."""


@dataclass
class _Progress:
    config: LayoutSolverConfig
    evaluated_moves: int = 0
    moves_applied: int = 0
    greedy_sweeps: int = 0
    initial_cost_cycles: float = 0.0
    termination_reason: str = "local_optimum"

    def evaluate(self, count: int = 1) -> None:
        if self.evaluated_moves + count > self.config.max_evaluations:
            self.termination_reason = "evaluation_budget"
            raise _EvaluationLimit
        self.evaluated_moves += count

    def applied(self) -> None:
        self.moves_applied += 1


def _finite_cost(value: float, context: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context} must be finite and nonnegative, got {value!r}")
    return number


class _Problem:
    """Validate and index immutable operation domains and tensor factors once."""

    def __init__(self, instance: LayoutProblemInstance) -> None:
        self.instance = instance
        self.ops = tuple(instance.operations)
        self.by_id: dict[str, int] = {}
        self.producers: dict[str, int] = {}
        self.participants: dict[str, set[int]] = {name: set() for name in instance.values}
        self.uses: list[tuple[dict[str, Counter[LayoutChoice]], ...]] = []
        self.op_costs: list[tuple[float, ...]] = []
        self.conversion_costs: dict[str, dict[LayoutChoice, dict[LayoutChoice, float]]] = {}
        self.boundary: dict[str, Counter[LayoutChoice]] = {}
        _require_integer(instance.cost_model.expected_uses, "expected_uses", 1)
        self._index_operations()
        self.order = self._topological_order()
        self._index_factors()

    def _index_operations(self) -> None:
        for index, op in enumerate(self.ops):
            if op.op_id in self.by_id:
                raise ValueError(f"Duplicate operation ID {op.op_id!r}")
            self.by_id[op.op_id] = index
            if not op.configs:
                raise ValueError(f"Operation {op.op_id!r} has no legal configurations")
            _require_integer(op.invocations, f"Invocations of {op.op_id!r}", 0)
            for name in (*op.inputs, op.output):
                if name not in self.instance.values:
                    raise ValueError(f"Operation {op.op_id!r} references unknown value {name!r}")
                self.participants[name].add(index)
            if op.output in self.producers:
                raise ValueError(
                    f"Value {op.output!r} has multiple producers; distinct SSA values are required"
                )
            self.producers[op.output] = index
            costs: list[float] = []
            uses: list[dict[str, Counter[LayoutChoice]]] = []
            cost_multiplier = _finite_cost(_operator_cost_multiplier(self.instance, op), op.op_id)
            for cfg in op.configs:
                self._validate_config(op, cfg)
                costs.append(_finite_cost(cost_multiplier * cfg.op_cost_cycles, f"Cost of {op.op_id!r}"))
                per_value: dict[str, Counter[LayoutChoice]] = {}
                for name, layout in zip(op.inputs, cfg.input_layouts, strict=True):
                    per_value.setdefault(name, Counter())[layout] += 1
                uses.append(per_value)
            self.op_costs.append(tuple(costs))
            self.uses.append(tuple(uses))
        for name, node in self.instance.values.items():
            producer = self.producers.get(name)
            actual = self.ops[producer].op_id if producer is not None else None
            if node.producer_op_id != actual:
                raise ValueError(
                    f"Value {name!r} declares producer {node.producer_op_id!r}, "
                    f"but operations define {actual!r}"
                )

    def _validate_config(self, op: LayoutOperationNode, cfg: OperatorConfig) -> None:
        if cfg.op_id != op.op_id or cfg.op_name != op.op_name:
            raise ValueError(
                f"Configuration for {cfg.op_id!r}/{cfg.op_name!r} does not belong to {op.op_id!r}"
            )
        if len(cfg.input_layouts) != len(op.inputs):
            raise ValueError(
                f"Configuration of {op.op_id!r} has {len(cfg.input_layouts)} inputs, "
                f"expected {len(op.inputs)}"
            )
        _finite_cost(cfg.op_cost_cycles, f"Configuration cost of {op.op_id!r}")
        for name, layout in zip(
            (*op.inputs, op.output), (*cfg.input_layouts, cfg.output_layout), strict=True
        ):
            if layout not in self.instance.values[name].admissible_layouts:
                raise ValueError(
                    f"Configuration of {op.op_id!r} requests inadmissible layout {layout!r} for {name!r}"
                )

    def _topological_order(self) -> tuple[int, ...]:
        successors: list[set[int]] = [set() for _ in self.ops]
        indegrees = [0] * len(self.ops)
        for index, op in enumerate(self.ops):
            predecessors = {self.producers[name] for name in op.inputs if name in self.producers}
            indegrees[index] = len(predecessors)
            for predecessor in predecessors:
                successors[predecessor].add(index)
        ready = [index for index, degree in enumerate(indegrees) if degree == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            index = heapq.heappop(ready)
            order.append(index)
            for successor in sorted(successors[index]):
                indegrees[successor] -= 1
                if indegrees[successor] == 0:
                    heapq.heappush(ready, successor)
        if len(order) != len(self.ops):
            raise ValueError(
                "Layout solver requires an acyclic single-producer graph; found a dependency cycle"
            )
        return tuple(order)

    def _index_factors(self) -> None:
        op_by_id = {op.op_id: op for op in self.ops}
        for name, node in self.instance.values.items():
            producer = self.producers.get(name)
            sources = (
                dict.fromkeys(cfg.output_layout for cfg in self.ops[producer].configs)
                if producer is not None
                else {node.current_layout: None}
            )
            targets: dict[LayoutChoice, None] = {}
            conversion_invocations: set[int] = set()
            for index in sorted(self.participants[name]):
                for uses in self.uses[index]:
                    requested = uses.get(name, ())
                    targets.update(dict.fromkeys(requested))
                    if any(source != target for source in sources for target in requested):
                        conversion_invocations.add(self.ops[index].invocations)
            boundary: Counter[LayoutChoice] = Counter()
            if node.is_external_output and not self.instance.cost_model.allow_external_output_relayout:
                targets[node.current_layout] = None
                boundary[node.current_layout] = 1
                if any(source != node.current_layout for source in sources):
                    conversion_invocations.add(self.instance.cost_model.expected_uses)
            self.boundary[name] = boundary
            multiplier = _conversion_multiplier(self.instance, node, op_by_id, tuple(conversion_invocations))
            self.conversion_costs[name] = {
                src: {
                    dst: _finite_cost(
                        0.0
                        if src == dst
                        else multiplier * self.instance.cost_model.estimate_conversion_cost(node, src, dst),
                        f"Conversion cost for {name!r}: {src!r} -> {dst!r}",
                    )
                    for dst in targets
                }
                for src in sources
            }

    def initial_choices(self, initial: LayoutAssignment) -> list[int]:
        unknown = set(initial.op_configs) - self.by_id.keys()
        missing = self.by_id.keys() - initial.op_configs.keys()
        if unknown or missing:
            raise ValueError(
                "Initial assignment needs exactly one configuration per operation; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        unknown_values = set(initial.value_layouts) - self.instance.values.keys()
        if unknown_values:
            raise ValueError(f"Initial assignment references unknown values {sorted(unknown_values)}")
        choices: list[int] = []
        for op in self.ops:
            cfg = initial.op_configs[op.op_id]
            if cfg not in op.configs:
                raise ValueError(f"Initial configuration of {op.op_id!r} is not in its legal catalog")
            supplied = initial.value_layouts.get(op.output, cfg.output_layout)
            if supplied != cfg.output_layout:
                raise ValueError(
                    f"Initial value layout for {op.output!r} contradicts its producer configuration"
                )
            choices.append(op.configs.index(cfg))
        return choices


class _State:
    """Demand-count and source-cost caches for exact incremental move gains."""

    def __init__(self, problem: _Problem, choices: list[int]) -> None:
        self.problem = problem
        self.choices = choices.copy()
        self.counts = {name: counts.copy() for name, counts in problem.boundary.items()}
        self.sources = {name: node.current_layout for name, node in problem.instance.values.items()}
        for index, choice in enumerate(choices):
            self.sources[problem.ops[index].output] = problem.ops[index].configs[choice].output_layout
            for name, uses in problem.uses[index][choice].items():
                self.counts[name].update(uses)
        self.source_costs = {
            name: {src: math.fsum(costs[dst] for dst in self.counts[name]) for src, costs in table.items()}
            for name, table in problem.conversion_costs.items()
        }
        self.cost = math.fsum(problem.op_costs[i][q] for i, q in enumerate(choices)) + math.fsum(
            self.source_costs[name][source] for name, source in self.sources.items()
        )

    def _changes(self, index: int, choice: int) -> dict[str, Counter[LayoutChoice]]:
        old_uses = self.problem.uses[index][self.choices[index]]
        new_uses = self.problem.uses[index][choice]
        changes: dict[str, Counter[LayoutChoice]] = {}
        for name in dict.fromkeys((*old_uses, *new_uses)):
            delta = Counter(new_uses.get(name, {}))
            delta.subtract(old_uses.get(name, {}))
            changes[name] = Counter({layout: count for layout, count in delta.items() if count})
        return changes

    def gain(self, index: int, choice: int) -> float:
        op = self.problem.ops[index]
        changes = self._changes(index, choice)
        changes.setdefault(op.output, Counter())
        delta = self.problem.op_costs[index][choice] - self.problem.op_costs[index][self.choices[index]]
        for name, counts in changes.items():
            source = op.configs[choice].output_layout if name == op.output else self.sources[name]
            factor = self.source_costs[name][source]
            for target, count in counts.items():
                old = self.counts[name].get(target, 0)
                factor += (int(old + count > 0) - int(old > 0)) * self.problem.conversion_costs[name][source][
                    target
                ]
            delta += factor - self.source_costs[name][self.sources[name]]
        return -delta

    def apply(self, index: int, choice: int) -> set[str]:
        gain = self.gain(index, choice)
        changes = self._changes(index, choice)
        touched: set[str] = set()
        for name, counts in changes.items():
            if counts:
                touched.add(name)
            for target, count in counts.items():
                old = self.counts[name].get(target, 0)
                new = old + count
                support_delta = int(new > 0) - int(old > 0)
                if support_delta:
                    for source, costs in self.problem.conversion_costs[name].items():
                        self.source_costs[name][source] += support_delta * costs[target]
                if new:
                    self.counts[name][target] = new
                else:
                    self.counts[name].pop(target, None)
        op = self.problem.ops[index]
        output_layout = op.configs[choice].output_layout
        if self.sources[op.output] != output_layout:
            touched.add(op.output)
            self.sources[op.output] = output_layout
        self.choices[index] = choice
        self.cost -= gain
        return touched

    def assignment(self) -> LayoutAssignment:
        layouts = dict(self.sources)
        for name, node in self.problem.instance.values.items():
            if name not in self.problem.producers and node.memory_space == MemorySpace.DDR:
                if len(self.counts[name]) == 1:
                    layouts[name] = next(iter(self.counts[name]))
        return LayoutAssignment(
            value_layouts=layouts,
            op_configs={op.op_id: op.configs[self.choices[i]] for i, op in enumerate(self.problem.ops)},
        )


def _min_fill_decomposition(
    vertices: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> tuple[list[int], list[tuple[int, ...]], list[int | None], int]:
    """Compute a min-fill elimination ordering, bags, parent links, and treewidth."""
    if not vertices:
        return [], [], [], 0
    adj_bits: dict[int, int] = {v: 0 for v in vertices}
    for u, v in edges:
        if u != v and u in adj_bits and v in adj_bits:
            adj_bits[u] |= 1 << v
            adj_bits[v] |= 1 << u

    remaining = sorted(set(vertices))
    elim_order: list[int] = []
    bags: list[tuple[int, ...]] = []

    def _compute_key(v: int) -> tuple[int, int, int]:
        nb = adj_bits[v]
        deg = nb.bit_count()
        fill2 = 0
        cur = nb
        while cur:
            low = cur & -cur
            u1 = low.bit_length() - 1
            fill2 += (nb & ~adj_bits[u1] & ~low).bit_count()
            cur ^= low
        return (fill2 >> 1, deg, v)

    keys: dict[int, tuple[int, int, int]] = {v: _compute_key(v) for v in remaining}

    while remaining:
        best_v = min(remaining, key=lambda v: keys[v])
        nb = adj_bits[best_v]
        nbrs: list[int] = []
        dirty_mask = nb
        cur = nb
        while cur:
            low = cur & -cur
            u = low.bit_length() - 1
            nbrs.append(u)
            adj_bits[u] = (adj_bits[u] | nb) & ~(1 << u) & ~(1 << best_v)
            dirty_mask |= adj_bits[u]
            cur ^= low
        bags.append((best_v, *nbrs))
        elim_order.append(best_v)
        del adj_bits[best_v]
        del keys[best_v]
        remaining.remove(best_v)
        dirty_mask &= ~(1 << best_v)
        while dirty_mask:
            low = dirty_mask & -dirty_mask
            w = low.bit_length() - 1
            if w in keys:
                keys[w] = _compute_key(w)
            dirty_mask ^= low

    step_of = {v: step for step, v in enumerate(elim_order)}
    parents: list[int | None] = []
    for bag in bags:
        sep = bag[1:]
        if not sep:
            parents.append(None)
        else:
            parents.append(min(step_of[u] for u in sep))

    tw = max((len(b) - 1 for b in bags), default=0)
    return elim_order, bags, parents, max(0, tw)


def _compute_active_decomposition(
    problem: _Problem,
) -> tuple[TreewidthStats, list[int], list[tuple[int, ...]], list[int | None]]:
    """Decompose only the movable operations (`|K_u| > 1`) for fast treewidth DP."""
    movable = sorted(i for i, op in enumerate(problem.ops) if len(op.configs) > 1)
    movable_set = set(movable)
    active_edges: list[tuple[int, int]] = []
    for participants in problem.participants.values():
        pts = [u for u in sorted(participants) if u in movable_set]
        for i, u in enumerate(pts):
            for v in pts[i + 1 :]:
                active_edges.append((u, v))
    elim_order, bags, parents, active_tw = _min_fill_decomposition(movable, active_edges)
    if len(problem.ops) <= 64:
        tw_stats, _, _, _ = _compute_problem_treewidth(problem)
        return tw_stats, elim_order, bags, parents
    bag_products = [math.prod(len(problem.ops[u].configs) for u in b) for b in bags]
    max_prod = max(bag_products, default=1)
    total_states = sum(bag_products) if bag_products else (1 if problem.ops else 0)
    max_active_bag_size = max((len(b) for b in bags), default=0)
    tw_stats = TreewidthStats(
        num_values=len(problem.instance.values),
        num_operations=len(problem.ops),
        num_movable_operations=len(movable_set),
        dataflow_treewidth=active_tw,
        primal_treewidth=active_tw,
        active_primal_treewidth=active_tw,
        max_bag_size=max_active_bag_size,
        max_active_bag_size=max_active_bag_size,
        max_bag_domain_product=max_prod,
        total_dp_states=total_states,
        elimination_order=tuple(problem.ops[u].op_id for u in elim_order),
        bags=tuple(tuple(problem.ops[u].op_id for u in b) for b in bags),
    )
    return tw_stats, elim_order, bags, parents


def _compute_problem_treewidth(
    problem: _Problem,
) -> tuple[TreewidthStats, list[int], list[tuple[int, ...]], list[int | None]]:
    all_ops = list(range(len(problem.ops)))
    movable_set = {i for i, op in enumerate(problem.ops) if len(op.configs) > 1}

    dataflow_edges: list[tuple[int, int]] = []
    primal_edges: list[tuple[int, int]] = []
    active_edges: list[tuple[int, int]] = []

    for name, participants in problem.participants.items():
        pts = sorted(participants)
        prod = problem.producers.get(name)
        if prod is not None:
            for c in pts:
                if c != prod:
                    dataflow_edges.append((prod, c))
        for i, u in enumerate(pts):
            for v in pts[i + 1 :]:
                primal_edges.append((u, v))
                if u in movable_set and v in movable_set:
                    active_edges.append((u, v))

    _, _, _, df_tw = _min_fill_decomposition(all_ops, dataflow_edges)
    elim_order, bags, parents, primal_tw = _min_fill_decomposition(all_ops, primal_edges)
    _, active_bags, _, active_tw = _min_fill_decomposition(sorted(movable_set), active_edges)

    max_bag_size = max((len(b) for b in bags), default=0)
    max_active_bag_size = max((len(b) for b in active_bags), default=0)
    bag_products = [math.prod(len(problem.ops[u].configs) for u in b) for b in active_bags]
    max_prod = max(bag_products, default=1 if problem.ops else 0)
    total_states = sum(bag_products) if bag_products else (1 if problem.ops else 0)

    stats = TreewidthStats(
        num_values=len(problem.instance.values),
        num_operations=len(problem.ops),
        num_movable_operations=len(movable_set),
        dataflow_treewidth=df_tw,
        primal_treewidth=primal_tw,
        active_primal_treewidth=active_tw,
        max_bag_size=max_bag_size,
        max_active_bag_size=max_active_bag_size,
        max_bag_domain_product=max_prod,
        total_dp_states=total_states,
        elimination_order=tuple(problem.ops[u].op_id for u in elim_order),
        bags=tuple(tuple(problem.ops[u].op_id for u in b) for b in bags),
    )
    return stats, elim_order, bags, parents


def compute_treewidth(instance: LayoutProblemInstance) -> TreewidthStats:
    """Compute treewidth and DP state-space statistics for *instance*."""
    problem = _Problem(instance)
    stats, _, _, _ = _compute_problem_treewidth(problem)
    return stats


def _topological_seed(problem: _Problem, progress: _Progress) -> list[int]:
    """Minimize operator plus new input-conversion cost against the assigned prefix."""
    choices = [0] * len(problem.ops)
    produced = {name: node.current_layout for name, node in problem.instance.values.items()}
    requested: dict[str, set[LayoutChoice]] = {name: set() for name in problem.instance.values}
    for index in problem.order:
        best_choice = 0
        best_cost = math.inf
        try:
            for choice in range(len(problem.ops[index].configs)):
                progress.evaluate()
                cost = problem.op_costs[index][choice]
                for name, targets in problem.uses[index][choice].items():
                    cost += math.fsum(
                        problem.conversion_costs[name][produced[name]][target]
                        for target in targets
                        if target not in requested[name]
                    )
                if cost < best_cost - progress.config.tolerance:
                    best_choice, best_cost = choice, cost
        except _EvaluationLimit:
            choices[index] = best_choice
            return choices
        choices[index] = best_choice
        op = problem.ops[index]
        produced[op.output] = op.configs[best_choice].output_layout
        for name, targets in problem.uses[index][best_choice].items():
            requested[name].update(targets)
    return choices


def _best_move(state: _State, index: int, progress: _Progress) -> tuple[float, int] | None:
    best: tuple[float, int] | None = None
    for choice in range(len(state.problem.ops[index].configs)):
        if choice == state.choices[index]:
            continue
        progress.evaluate()
        gain = state.gain(index, choice)
        if best is None or gain > best[0] + progress.config.tolerance:
            best = gain, choice
    return best


def _greedy_refine(state: _State, progress: _Progress) -> None:
    try:
        for _ in range(progress.config.max_greedy_sweeps):
            progress.greedy_sweeps += 1
            improved = False
            for index in state.problem.order:
                move = _best_move(state, index, progress)
                if move is not None and move[0] > progress.config.tolerance:
                    state.apply(index, move[1])
                    progress.applied()
                    improved = True
            if not improved:
                progress.termination_reason = "local_optimum"
                return
        progress.termination_reason = "greedy_sweep_budget"
    except _EvaluationLimit:
        return


def _conversion_report(state: _State) -> tuple[tuple[LayoutConversion, ...], tuple[str, ...]]:
    conversions: list[LayoutConversion] = []
    issues: list[str] = []
    problem = state.problem
    for name, node in problem.instance.values.items():
        source = state.sources[name]
        external_input = name not in problem.producers and node.memory_space == MemorySpace.DDR
        if external_input and len(state.counts[name]) > 1:
            issues.append(f"GM input {name!r} needs multiple representations")
        for target in state.counts[name]:
            if target == source:
                continue
            consumers = tuple(
                problem.ops[i].op_id
                for i in sorted(problem.participants[name])
                if target in problem.uses[i][state.choices[i]].get(name, {})
            )
            boundary_output = not consumers and node.is_external_output and target == node.current_layout
            boundary = external_input or boundary_output
            conversions.append(
                LayoutConversion(
                    name, source, target, problem.conversion_costs[name][source][target], consumers, boundary
                )
            )
            if not boundary:
                issues.append(f"Value {name!r} needs an internal conversion {source!r} -> {target!r}")
    return tuple(conversions), tuple(issues)


def _result(
    state: _State,
    progress: _Progress,
    tw_stats: TreewidthStats | None = None,
    *,
    active_tw: int = 0,
) -> LayoutSolverResult:
    breakdown = evaluate_assignment(state.problem.instance, state.assignment())
    if not math.isclose(state.cost, breakdown.total_cost_cycles, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError(
            f"Layout gain cache disagrees with evaluator: {state.cost} vs {breakdown.total_cost_cycles}"
        )
    conversions, issues = _conversion_report(state)
    stats = LayoutSolverStats(
        evaluated_moves=progress.evaluated_moves,
        moves_applied=progress.moves_applied,
        greedy_sweeps=progress.greedy_sweeps,
        initial_cost_cycles=progress.initial_cost_cycles,
        termination_reason=progress.termination_reason,
        treewidth=tw_stats.primal_treewidth if tw_stats is not None else active_tw,
        active_treewidth=tw_stats.active_primal_treewidth if tw_stats is not None else active_tw,
    )
    return LayoutSolverResult(breakdown, stats, conversions, issues, treewidth_stats=tw_stats)


def solve_greedy(
    instance: LayoutProblemInstance,
    *,
    initial: LayoutAssignment | None = None,
    config: LayoutSolverConfig | None = None,
) -> LayoutSolverResult:
    """Run the paper's topological initialization and full-objective greedy refinement."""
    problem = _Problem(instance)
    progress = _Progress(config or LayoutSolverConfig())
    choices = (
        problem.initial_choices(initial) if initial is not None else _topological_seed(problem, progress)
    )
    state = _State(problem, choices)
    progress.initial_cost_cycles = state.cost
    if progress.termination_reason != "evaluation_budget":
        _greedy_refine(state, progress)
    return _result(state, progress)


def _eval_tensor_factor_at_bag(
    problem: _Problem,
    name: str,
    bag_pos: dict[int, int],
    combo: tuple[int, ...],
) -> float:
    node = problem.instance.values[name]
    prod = problem.producers.get(name)
    if prod is None:
        source = node.current_layout
    else:
        prod_q = combo[bag_pos[prod]] if prod in bag_pos else 0
        source = problem.ops[prod].configs[prod_q].output_layout
    requested: set[LayoutChoice] = set(problem.boundary[name].keys())
    for u in problem.participants[name]:
        q_u = combo[bag_pos[u]] if u in bag_pos else 0
        requested.update(problem.uses[u][q_u].get(name, ()))
    costs = problem.conversion_costs[name][source]
    return math.fsum(costs[dst] for dst in requested)


def _run_treewidth_dp_on_decomposition(
    problem: _Problem,
    cfg: LayoutSolverConfig,
    progress: _Progress,
    tw_stats: TreewidthStats,
    elim_order: Sequence[int],
    bags: Sequence[tuple[int, ...]],
    parents: Sequence[int | None],
) -> LayoutSolverResult:
    if not problem.ops:
        progress.termination_reason = "optimal"
        return _result(_State(problem, []), progress, tw_stats)

    if tw_stats.active_primal_treewidth > cfg.max_treewidth:
        raise ValueError(
            f"Treewidth DP active primal treewidth {tw_stats.active_primal_treewidth} exceeds "
            f"max_treewidth={cfg.max_treewidth}"
        )

    if tw_stats.max_bag_domain_product > cfg.max_bag_states:
        raise ValueError(
            f"Treewidth DP max bag states {tw_stats.max_bag_domain_product} exceeds "
            f"max_bag_states={cfg.max_bag_states} "
            f"(active_primal_treewidth={tw_stats.active_primal_treewidth})"
        )

    step_of = {u: step for step, u in enumerate(elim_order)}
    factors_at_bag: list[list[str]] = [[] for _ in bags]
    for name, pts in problem.participants.items():
        act_pts = [u for u in pts if u in step_of]
        if not act_pts:
            continue
        first_step = min(step_of[u] for u in act_pts)
        factors_at_bag[first_step].append(name)

    children_of: list[list[int]] = [[] for _ in bags]
    for step, parent in enumerate(parents):
        if parent is not None:
            children_of[parent].append(step)

    messages: list[dict[tuple[int, ...], float]] = [{} for _ in bags]
    backpointers: list[dict[tuple[int, ...], int]] = [{} for _ in bags]
    child_seps: list[tuple[int, ...]] = [bag[1:] for bag in bags]

    for step, bag in enumerate(bags):
        elim_u = bag[0]
        sep = bag[1:]
        bag_pos = {u: idx for idx, u in enumerate(bag)}
        child_pos = [(c, tuple(bag_pos[u] for u in child_seps[c])) for c in children_of[step]]
        sep_domains = [range(len(problem.ops[u].configs)) for u in sep]
        elim_domain = range(len(problem.ops[elim_u].configs))
        bag_factors = factors_at_bag[step]

        msg_table: dict[tuple[int, ...], float] = {}
        arg_table: dict[tuple[int, ...], int] = {}
        for sep_combo in itertools.product(*sep_domains):
            best_val = math.inf
            best_q = 0
            for q_elim in elim_domain:
                progress.evaluate()
                combo = (q_elim, *sep_combo)
                val = problem.op_costs[elim_u][q_elim]
                for t_name in bag_factors:
                    val += _eval_tensor_factor_at_bag(problem, t_name, bag_pos, combo)
                for c_idx, pos_tuple in child_pos:
                    c_key = tuple(combo[p] for p in pos_tuple)
                    val += messages[c_idx][c_key]
                if val < best_val - cfg.tolerance:
                    best_val = val
                    best_q = q_elim
            msg_table[sep_combo] = best_val
            arg_table[sep_combo] = best_q
        messages[step] = msg_table
        backpointers[step] = arg_table

    choices = [0] * len(problem.ops)
    for step in range(len(bags) - 1, -1, -1):
        elim_u = bags[step][0]
        sep = bags[step][1:]
        sep_key = tuple(choices[u] for u in sep)
        choices[elim_u] = backpointers[step][sep_key]

    state = _State(problem, choices)
    progress.initial_cost_cycles = state.cost
    progress.termination_reason = "optimal"
    return _result(state, progress, tw_stats)


def solve_treewidth_dp(
    instance: LayoutProblemInstance,
    *,
    config: LayoutSolverConfig | None = None,
) -> LayoutSolverResult:
    """Solve `instance` to proven global optimality via treewidth dynamic programming."""
    problem = _Problem(instance)
    cfg = config or LayoutSolverConfig()
    progress = _Progress(cfg)
    tw_stats, elim_order, bags, parents = _compute_active_decomposition(problem)
    return _run_treewidth_dp_on_decomposition(problem, cfg, progress, tw_stats, elim_order, bags, parents)


def solve_auto_layout(
    instance: LayoutProblemInstance,
    *,
    initial: LayoutAssignment | None = None,
    config: LayoutSolverConfig | None = None,
) -> LayoutSolverResult:
    """Solve `instance` via exact treewidth DP when `active_treewidth <= max_treewidth`, else greedy."""
    problem = _Problem(instance)
    cfg = config or LayoutSolverConfig()
    progress = _Progress(cfg)
    tw_stats, elim_order, bags, parents = _compute_active_decomposition(problem)
    if (
        tw_stats.active_primal_treewidth <= cfg.max_treewidth
        and tw_stats.max_bag_domain_product <= cfg.max_bag_states
    ):
        return _run_treewidth_dp_on_decomposition(problem, cfg, progress, tw_stats, elim_order, bags, parents)
    choices = (
        problem.initial_choices(initial) if initial is not None else _topological_seed(problem, progress)
    )
    state = _State(problem, choices)
    progress.initial_cost_cycles = state.cost
    if progress.termination_reason != "evaluation_budget":
        _greedy_refine(state, progress)
    return _result(state, progress, tw_stats)


__all__ = [
    "LayoutConversion",
    "LayoutSolverConfig",
    "LayoutSolverResult",
    "LayoutSolverStats",
    "TreewidthStats",
    "compute_treewidth",
    "solve_auto_layout",
    "solve_greedy",
    "solve_treewidth_dp",
]
