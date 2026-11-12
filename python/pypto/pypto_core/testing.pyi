# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Type stubs for pypto.testing submodule

Internal testing utilities (do not use in production)
"""

from typing import Any, Literal, NoReturn, TypedDict

from .ir import (
    BufferAccess,
    BufferResultBehavior,
    Call,
    Expr,
    Function,
    OpIRStage,
    PipeType,
    Program,
    Stmt,
    Type,
)

def validate_buffer_op_contract(
    stage: OpIRStage,
    arity: int | None,
    args: list[Expr],
    result_type: Type,
    effects: list[tuple[int, bool, BufferAccess, BufferAccess]],
    results: list[tuple[int, BufferResultBehavior, int | None]],
    internal_only: bool = True,
) -> None:
    """Check a local registration and call without mutating the global registry.

    Effect entries give argument index, non-memory classification, data access,
    and metadata access. Result entries give result index, ownership behavior,
    and optional alias source argument. ``arity=None`` omits the declaration.
    """

def validate_op_type_registration(
    stage: OpIRStage, internal_only: bool, typing_modes: list[Literal["deduced", "explicit"]]
) -> None:
    """Check ordered typing-mode declarations without changing the global registry."""

def validate_buffer_call(call: Call) -> None:
    """Validate a stored Buffer call without changing its arguments or result type."""

class DsaReusePenaltyEdge(TypedDict):
    """One internal pre-solver DSA-RP recognizer result."""

    first_interval: int
    second_interval: int
    first_name: str
    second_name: str
    cost: int

def raise_value_error(message: str) -> NoReturn:
    """Raise a ValueError from C++ for testing error handling"""

def raise_type_error(message: str) -> NoReturn:
    """Raise a TypeError from C++ for testing error handling"""

def raise_runtime_error(message: str) -> NoReturn:
    """Raise a RuntimeError from C++ for testing error handling"""

def raise_not_implemented_error(message: str) -> NoReturn:
    """Raise a NotImplementedError from C++ for testing error handling"""

def raise_index_error(message: str) -> NoReturn:
    """Raise an IndexError from C++ for testing error handling"""

def raise_generic_error(message: str) -> NoReturn:
    """Raise a generic Error from C++ for testing error handling"""

def raise_assertion_error(message: str) -> NoReturn:
    """Raise an AssertionError from C++ for testing purposes"""

def raise_internal_error(message: str) -> NoReturn:
    """Raise an InternalError from C++ for testing error handling"""

def raise_internal_error_with_span(message: str, filename: str, line: int, col: int) -> NoReturn:
    """Raise an InternalError with IR source span for testing"""

def rethrow_with_message(kind: str, original: str, replacement: str) -> NoReturn:
    """Raise `kind` and rethrow it via Error::RethrowWithMessage for testing"""

def recognize_dsa_reuse_penalties(function: Function) -> list[DsaReusePenaltyEdge]:
    """Return recognized DSA-RP edges without running placement."""

def try_infer_pipe(call: Call) -> int | None:
    """Return the exact backend pipe for a Call, or None."""

def get_execution_memory_access_evidence(op_name: str) -> Literal["unknown", "functional", "no_access"]:
    """Return an operation's execution-memory-access evidence."""

def get_declared_core_affinity(op_name: str) -> Literal["cube", "vector", "shared", "mixed"] | None:
    """Return an operation's explicitly declared core affinity, or None.

    ``None`` means the op declares no affinity, so ``ClassifyCallAffinity``
    derives it from the call itself (memory spec, operand tiles, result tile).
    """

def is_no_duplicate_op(op_name: str) -> bool:
    """Return whether an operation must not run on a second core.

    A no-duplicate op changes what the program means when it is replicated onto
    the other lane of a mixed kernel — ``pld.system.notify`` can release a peer
    from the cube lane before the vector lane's TPUT has landed the data.
    Orthogonal to core affinity, which decides placement. Read by
    ``LowerAutoVectorSplit``'s ``pl.split_aiv`` region placement stamp.
    """

def classify_call_affinity(call: Call) -> Literal["cube", "vector", "shared", "mixed"]:
    """Return the core affinity ``ClassifyCallAffinity`` derives for a Call.

    Unlike :func:`get_declared_core_affinity` this is the *effective* placement:
    it runs the full classification chain (declared affinity, the dynamic
    special cases, output memory spec, first tile argument, result tile
    memory), so the answer depends on how far the call has been lowered.
    """

class FiniteSyncBoundary(TypedDict):
    """Original occurrence indices; None inside an available boundary proves absence."""

    first_writer: int | None
    last_writer: int | None
    first_readers: list[int | None]
    last_readers: list[int | None]

class FiniteSyncResult(TypedDict):
    """Numerical demand analysis; no executable or allocation guarantee."""

    exact_accesses: bool
    generators: list[tuple[int, int]]
    demands: list[tuple[int, int]]
    starts: list[list[int]]
    completions: list[list[int]]
    queries: list[bool]
    boundaries: list[FiniteSyncBoundary | None]
    work: dict[str, int]

def analyze_finite_sync(
    occurrences: list[tuple[PipeType, bool, list[int], list[int]]],
    covered_cells: list[int],
    exact_accesses: bool,
    prerequisites: list[tuple[int, int]],
    native_order: list[tuple[int, int]],
    queries: list[tuple[int, bool, int, bool]],
    boundary_cells: list[int],
) -> FiniteSyncResult:
    """Analyze resolved occurrences over provider-certified physical classes.

    Each occurrence is (pipe, present, reads, writes). Queries are
    (source, source_is_completion, target, target_is_completion), strictly
    ordered. An unavailable storage query returns None, never empty selectors.
    False exact_accesses denotes a sound conservative model; returned covers
    are minimum for that model. This helper does not import PyPTO IR effects.
    """

class LogicalSyncRealization(TypedDict):
    """Original payloads with logical tokens; no hardware lowering guarantee."""

    function: Function
    record_sites: int
    wait_sites: int
    guarded_wait_sites: int
    carried_tokens: int
    name_scan_visits: int
    name_scan_nodes: int
    guarantee: str

class PhysicalSyncFamily(TypedDict):
    """Original occurrence frame and cyclic ID list of one demand family."""

    source: tuple[int, int, int]
    target: tuple[int, int, int]
    count: int
    ids: list[int]

class PhysicalSyncRealization(TypedDict):
    """Qualified physical commands; native invocation premises remain explicit."""

    function: Function
    families: list[PhysicalSyncFamily]
    set_sites: int
    wait_sites: int
    barrier_sites: int
    guard_sites: int
    constructed_guard_sites: int
    shared_guard_sites: int
    control_statement_visits: int
    reuse_queries: int
    reserved_lanes: int
    native_reserved_ids: list[tuple[PipeType, PipeType, int]]
    finite_minimum: bool
    guarantee: str
    native_mode: str

class CompactSyncAnalysis:
    """Retained research analysis with opt-in logical and qualified physical realization."""

    def describe(self) -> dict[str, Any]:
        """Return retained geometry, occurrence mapping, families and work counts."""
        ...

    def expand_retained_demands(
        self, max_payloads: int
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """Audit actual retained records within an explicit payload budget; no new analysis."""
        ...

    def export_crossing_demands(
        self,
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None:
        """Return original-coordinate flat crossings, or unavailable for unsupported shared uses."""
        ...

    def scoped_consecutive_reuse(self, family: int) -> bool:
        """Uniform period-one reuse of an owned schema; False is unproved, not scarcity."""
        ...
    def export_scoped_demands(self) -> dict[str, Any] | None:
        """Return cached original-frame Cartesian demand families, or unavailable."""
        ...

    def realize(self) -> LogicalSyncRealization:
        """Insert logical record/wait commands from this retained result."""
        ...

    def realize_physical(self, target: str, eligible_ids: list[int] | None = None) -> PhysicalSyncRealization:
        """Realize retained demands using the target's eligible directional IDs.

        Optional IDs restrict the catalog. Refuse unsupported realization,
        unproved compact allocation or proved finite fixed-plan scarcity.
        This does not lower arbitrary frontend logical tokens.
        """
        ...

    def precedes(
        self,
        source: tuple[int, int, int],
        source_completion: bool,
        target: tuple[int, int, int],
        target_completion: bool,
    ) -> bool:
        """Query segment/visit/site event order."""
        ...

    def boundary(
        self,
        storage: int,
        row: int,
        column: int,
        parameters: dict[int, int] = {},
        block_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Return modeled point selectors; None is unavailable, not absence."""
        ...

    def boundary_slot(
        self,
        storage: int,
        slot: int,
        row: int,
        column: int,
        parameters: dict[int, int] = {},
        block_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Query an explicit physical slot; unavailable differs from proven absence."""
        ...

def sync_parameter_bounds_profile(program: Program) -> dict[str, Any]:
    """Inspect proved ranges and expression/range visits per binding environment."""
    ...

def sync_parameter_bounds(program: Program) -> dict[str, dict[int, tuple[int, int]]]:
    """Inspect inclusive ranges joined at all callers through an acyclic call graph."""
    ...

def sync_invocation_evidence(program: Program) -> dict[str, Any]:
    """Inspect formal bounds, launch-index hulls and static call-graph traversal work.

    Missing entries are unproved. Launch indices are not formal ABI parameters.
    Recursive components and their descendants remain unavailable.
    """
    ...

def analyze_placed_sync(
    function: Function,
    disjoint_gm: bool,
    target: str = "a2a3",
    parameter_bounds: dict[int, tuple[int, int]] = {},
    composition_order: str = "balanced",
    block_bounds: tuple[int, int] | None = None,
    original_frames: bool = False,
) -> CompactSyncAnalysis:
    """Analyze placed Buffer IR with explicit GM aliases and a named hardware contract.

    original_frames requests the existing structured representation for perfect
    multi-frame nests, matching logical token insertion's occurrence contract.
    """
    ...

def describe_placed_sync_input(
    function: Function,
    disjoint_gm: bool,
    target: str = "a2a3",
    parameter_bounds: dict[int, tuple[int, int]] = {},
    block_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Inspect accesses and optional exact byte intervals without constructing demands.

    A missing physical_intervals entry denotes unavailable interval geometry,
    not absence. interval_access_visits counts static qualification work.
    An analyzed region also reports interval tree work separately from atom tests.
    collective_witnesses counts certified crossing points, not a complete cell partition.
    """
    ...

class SyncInvocationAnalysis:
    def realize_physical(self, target: str, eligible_ids: list[int] | None = None) -> dict[str, Any]:
        """Qualify cases and emit cuts; case_sites precede coalescing, top-level counts follow it."""
        ...
    def select_case(self, block_index: int, parameters: dict[int, int]) -> CompactSyncAnalysis | None:
        """Return retained mathematics only for an admitted fully bound invocation."""
        ...
    def describe(self) -> dict[str, Any]:
        """Inspect exhaustive disjoint cases and construction work."""
        ...

def analyze_sync_invocation(
    function: Function,
    disjoint_gm: bool,
    target: str,
    parameter_bounds: dict[int, tuple[int, int]],
    block_bounds: tuple[int, int],
) -> SyncInvocationAnalysis:
    """Construct qualified cases over the entire supplied launch contract."""
    ...

def inspect_scoped_token_families(
    function: Function,
    target: str = "a2a3",
    parameter_bounds: dict[int, tuple[int, int]] = {},
    block_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Inspect actual token families with disjoint GM roots; no reduction or allocation."""
    ...

class ScopedTokenOrder:
    @property
    def summary_nodes(self) -> int: ...
    @property
    def interface_entries(self) -> int: ...
    def precedes(
        self,
        source: tuple[int, int, int],
        source_completion: bool,
        target: tuple[int, int, int],
        target_completion: bool,
    ) -> bool: ...
    def uniform(self, family: int) -> bool: ...
    def reuse(self, family: int) -> bool: ...
    def barrier(self, family: int) -> bool: ...
    def realize(self, target: str = "a2a3") -> Function: ...

def analyze_scoped_token_order(function: Function, target: str = "a2a3") -> ScopedTokenOrder: ...
def share_sync_command_guards(commands: Stmt) -> Stmt:
    """Test emitter-only pure coordinate-guarded physical commands, with no definitions or payloads."""
    ...

def materialize_sync_command_values(owner: Function, commands: Stmt) -> Stmt:
    """Test sharing of pure emitter-owned predicates inside their original command scope."""
    ...

def emit_sync_command_cases(
    block: Expr, possible: list[tuple[int, int]], cases: list[tuple[int, int, Stmt]]
) -> Stmt:
    """Test command cases against a supplied possible cut domain; unknown requires the full domain."""
    ...
