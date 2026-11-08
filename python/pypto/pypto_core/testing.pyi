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

from .ir import BufferAccess, BufferResultBehavior, Call, Expr, Function, OpIRStage, PipeType, Type

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

class CompactSyncAnalysis:
    """Retained research analysis with opt-in logical realization; no physical allocation."""

    def describe(self) -> dict[str, Any]:
        """Return retained geometry, occurrence mapping, families and work counts."""
        ...

    def realize(self) -> LogicalSyncRealization:
        """Insert logical record/wait commands from this retained result."""
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

    def boundary(self, storage: int, row: int, column: int) -> dict[str, Any] | None:
        """Return modeled point selectors; None is unavailable, not absence."""
        ...

def analyze_placed_sync(function: Function, disjoint_gm: bool) -> CompactSyncAnalysis:
    """Analyze qualified placed Buffer IR under an explicit GM alias contract."""
    ...
