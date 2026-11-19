# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python wrapper delegating AutoLayout greedy/treewidth-DP solving to native C++."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from pypto.pypto_core import auto_layout as _cpp_al

from .auto_layout import (
    CostBreakdown,
    LayoutAssignment,
    LayoutChoice,
    LayoutProblemInstance,
    _from_cpp_breakdown,
    _from_cpp_layout_spec,
    _to_cpp_assignment,
    _to_cpp_instance,
)


def _require_integer(value: object, name: str, minimum: int | None = None) -> None:
    """Validate runtime inputs, including values arriving from untyped callers."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")


@dataclass(frozen=True)
class LayoutSolverConfig:
    """Deterministic search limits, with costs expressed in modeled cycles."""

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
    """One shared conversion required by the scored assignment."""

    value_name: str
    source_layout: LayoutChoice
    target_layout: LayoutChoice
    cost_cycles: float
    consumer_op_ids: tuple[str, ...]
    is_boundary: bool


@dataclass(frozen=True)
class TreewidthStats:
    """Treewidth and DP state-space diagnostics for a `LayoutProblemInstance`."""

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
        return self.breakdown.assignment

    @property
    def total_cost_cycles(self) -> float:
        return self.breakdown.total_cost_cycles

    def writeback_assignment(self) -> LayoutAssignment:
        if self.writeback_issues:
            raise ValueError(
                f"Layout solution requires conversion materialization: {'; '.join(self.writeback_issues)}"
            )
        return self.assignment


def _to_cpp_solver_config(cfg: LayoutSolverConfig | None) -> _cpp_al.LayoutSolverConfig:
    c = cfg or LayoutSolverConfig()
    out = _cpp_al.LayoutSolverConfig()
    out.max_evaluations = c.max_evaluations
    out.max_greedy_sweeps = c.max_greedy_sweeps
    out.max_treewidth = c.max_treewidth
    out.max_bag_states = c.max_bag_states
    out.tolerance = c.tolerance
    out.validate()
    return out


def _from_cpp_treewidth_stats(tw: _cpp_al.TreewidthStats) -> TreewidthStats:
    return TreewidthStats(
        num_values=tw.num_values,
        num_operations=tw.num_operations,
        num_movable_operations=tw.num_movable_operations,
        dataflow_treewidth=tw.dataflow_treewidth,
        primal_treewidth=tw.primal_treewidth,
        active_primal_treewidth=tw.active_primal_treewidth,
        max_bag_size=tw.max_bag_size,
        max_active_bag_size=tw.max_active_bag_size,
        max_bag_domain_product=tw.max_bag_domain_product,
        total_dp_states=tw.total_dp_states,
        elimination_order=tuple(tw.elimination_order),
        bags=tuple(tuple(b) for b in tw.bags),
    )


def _from_cpp_solver_result(res: _cpp_al.LayoutSolverResult) -> LayoutSolverResult:
    conversions = tuple(
        LayoutConversion(
            value_name=c.value_name,
            source_layout=_from_cpp_layout_spec(c.source_layout),
            target_layout=_from_cpp_layout_spec(c.target_layout),
            cost_cycles=c.cost_cycles,
            consumer_op_ids=tuple(c.consumer_op_ids),
            is_boundary=c.is_boundary,
        )
        for c in res.conversions
    )
    stats = LayoutSolverStats(
        evaluated_moves=res.stats.evaluated_moves,
        moves_applied=res.stats.moves_applied,
        greedy_sweeps=res.stats.greedy_sweeps,
        initial_cost_cycles=res.stats.initial_cost_cycles,
        termination_reason=res.stats.termination_reason,
        treewidth=res.stats.treewidth,
        active_treewidth=res.stats.active_treewidth,
    )
    tw_stats = _from_cpp_treewidth_stats(res.treewidth_stats) if res.treewidth_stats is not None else None
    return LayoutSolverResult(
        breakdown=_from_cpp_breakdown(res.breakdown),
        stats=stats,
        conversions=conversions,
        writeback_issues=tuple(res.writeback_issues),
        treewidth_stats=tw_stats,
    )


class _Problem:
    """Python compatibility shim backed by C++ `IndexedProblem`."""

    def __init__(self, instance: LayoutProblemInstance) -> None:
        self.instance = instance
        self._cpp_inst = _to_cpp_instance(instance)
        self._cpp = _cpp_al.IndexedProblem(self._cpp_inst)
        self.ops = tuple(instance.operations)
        self.by_id = {op.op_id: idx for idx, op in enumerate(instance.operations)}

    def initial_choices(self, initial: LayoutAssignment | None) -> list[int]:
        if initial is None:
            return [0] * self._cpp.num_ops
        return list(self._cpp.initial_choices(_to_cpp_assignment(initial)))


class _State:  # pyright: ignore[reportUnusedClass]
    """Python compatibility shim backed by C++ `IncrementalState`."""

    def __init__(self, problem: _Problem, choices: list[int]) -> None:
        self.problem = problem
        self._cpp = _cpp_al.IncrementalState(problem._cpp, choices)

    @property
    def cost(self) -> float:
        return self._cpp.cost

    @property
    def choices(self) -> list[int]:
        return list(self._cpp.choices)

    @property
    def counts(self) -> dict[str, Counter[LayoutChoice]]:
        out: dict[str, Counter[LayoutChoice]] = {}
        for v_info, cnts in zip(self.problem._cpp.values, self._cpp.counts, strict=True):
            c: Counter[LayoutChoice] = Counter()
            for lay, cnt in zip(v_info.layouts, cnts, strict=True):
                if cnt:
                    c[_from_cpp_layout_spec(lay)] = cnt
            out[v_info.name] = c
        return out

    @property
    def sources(self) -> dict[str, LayoutChoice]:
        return {
            v_info.name: _from_cpp_layout_spec(v_info.layouts[src_idx])
            for v_info, src_idx in zip(self.problem._cpp.values, self._cpp.sources, strict=True)
        }

    @property
    def source_costs(self) -> dict[str, dict[LayoutChoice, float]]:
        out: dict[str, dict[LayoutChoice, float]] = {}
        for v_info, scs in zip(self.problem._cpp.values, self._cpp.source_costs, strict=True):
            out[v_info.name] = {
                _from_cpp_layout_spec(lay): sc for lay, sc in zip(v_info.layouts, scs, strict=True)
            }
        return out

    def gain(self, index: int, choice: int) -> float:
        return self._cpp.gain(index, choice)

    def apply(self, index: int, choice: int) -> None:
        self._cpp.apply(index, choice)


def compute_treewidth(instance: LayoutProblemInstance) -> TreewidthStats:
    """Compute min-fill treewidth and DP state-space bounds for `instance`."""
    return _from_cpp_treewidth_stats(_cpp_al.compute_treewidth(_to_cpp_instance(instance)))


def solve_greedy(
    instance: LayoutProblemInstance,
    *,
    initial: LayoutAssignment | None = None,
    config: LayoutSolverConfig | None = None,
) -> LayoutSolverResult:
    """Solve `instance` via native C++ topological initialization + coordinate descent."""
    cpp_inst = _to_cpp_instance(instance)
    cpp_init = _to_cpp_assignment(initial) if initial is not None else None
    cpp_cfg = _to_cpp_solver_config(config)
    return _from_cpp_solver_result(_cpp_al.solve_greedy(cpp_inst, cpp_init, cpp_cfg))


def solve_treewidth_dp(
    instance: LayoutProblemInstance,
    *,
    config: LayoutSolverConfig | None = None,
) -> LayoutSolverResult:
    """Solve `instance` to global optimality via native C++ min-fill treewidth DP."""
    cpp_inst = _to_cpp_instance(instance)
    cpp_cfg = _to_cpp_solver_config(config)
    return _from_cpp_solver_result(_cpp_al.solve_treewidth_dp(cpp_inst, cpp_cfg))


def solve_auto_layout(
    instance: LayoutProblemInstance,
    *,
    initial: LayoutAssignment | None = None,
    config: LayoutSolverConfig | None = None,
) -> LayoutSolverResult:
    """Run exact treewidth DP when `active_treewidth <= max_treewidth`, else greedy."""
    cpp_inst = _to_cpp_instance(instance)
    cpp_init = _to_cpp_assignment(initial) if initial is not None else None
    cpp_cfg = _to_cpp_solver_config(config)
    return _from_cpp_solver_result(_cpp_al.solve_auto_layout(cpp_inst, cpp_init, cpp_cfg))


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
