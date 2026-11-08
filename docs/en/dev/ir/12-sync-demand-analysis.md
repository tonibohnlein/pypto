# Finite synchronization demand analysis

`ir::sync::AnalyzeFiniteSync` implements the concrete finite construction:
physical access classes → storage generators → nonnative covers. It is a
read-only C++ analysis service in `transforms/utils/finite_sync_analysis.h`.
The default compilation pipeline does not call it yet. This first component
supplies demand analysis; it does not insert synchronization commands.

## Provider contract

The input lists occurrences in reference order, concrete `PipeType`s, resolved
presence, and read/write cell memberships. Indices retain original occurrence
identity even when intervening occurrences are absent. An importer must retain
the mapping back to original statements and enclosing invocation coordinates.
Cells must model physical overlap, including different SSA roots reusing bytes.
The service does not recover footprints from operand effects or interpret
machine instructions, implicit scratch accesses, scalar visibility or macros.

`covered_cells` declares the complete storage domain of this result. Accesses
outside that domain are rejected, including in absent occurrence descriptions.
A query outside the domain returns `nullopt`; a query for a covered unused cell
returns an available boundary with absent selectors. Local coverage cannot
silently answer a missing global-memory query as empty.

`exact_accesses=true` is a provider certificate that the supplied classes
preserve physical conflicts exactly. With `false`, classes must still be sound
enclosures. Both cases compute minimum demands for the **supplied model**;
only the first may inherit an exact physical-demand guarantee. This boolean
does not itself validate access extraction or native semantics.

Core native order is start→completion per occurrence and separate start/start
and completion/completion chains per pipe. It does not serialize same-pipe
payloads. Additional native completion→start facts and required prerequisites
are separate forward edge lists. Native facts participate before reduction.
A caller can preserve qualified scalar/accumulator exemptions through that
interface; the analysis invents no exemptions from instruction or pipe names.
Edges apply only when both endpoints are present. Other native event-edge
forms and unresolved predicates are outside this finite contract.

## Construction and example

For each cell the scan retains its previous writer and read-only accesses
since that writer. A new access receives a previous-writer edge; a write also
receives edges from those readers before replacing the frontier. RMW accesses
query old state once, then become writers. Endpoint duplicates are merged.
Reduction visits each consumer's incoming sources in decreasing reference
order and joins completion-rank rows only for indispensable demands.

```cpp
using namespace pypto::ir;
sync::FiniteSyncInput input;
input.covered_cells = {0};
input.exact_accesses = true;
input.occurrences = {{PipeType::MTE2, true, {}, {0}},
                     {PipeType::V, true, {0}, {}},
                     {PipeType::MTE2, true, {}, {0}}};
auto result = sync::AnalyzeFiniteSync(input);
// generators: (0,1), (1,2), (0,2); demands: (0,1), (1,2).
// MTE2 completion order alone does not protect V's read from the overwrite.
```

The internal Python testing seam exercises this same C++ implementation:

```python
from pypto import ir, testing
result = testing.analyze_finite_sync(
    [(ir.PipeType.MTE2, True, [], [0]),
     (ir.PipeType.V, True, [0], []),
     (ir.PipeType.MTE2, True, [], [0])],
    [0], True, [], [], [(0, True, 1, False)], [0, 99],
)
assert result["demands"] == [(0, 1), (1, 2)]
assert result["queries"] == [True]
assert result["boundaries"][1] is None  # unavailable domain, not absence
```

## Queries, boundaries and cost

Numerical rows answer all four strict start/completion reachability queries.
They retain start ancestors as well as completion ancestors; a start must not
be confused with its own completion. Queries involving absent occurrences are
false; invalid indices are errors. No symbolic DAG or all-pairs matrix is built.

Storage boundaries return first/last writers and the first reader before the
first writer and last reader after the last writer on each pipe. RMW sites are
writers. For a writer-free region, reader boundaries cover the whole region.
These are ingredients for exact contiguous-region composition, not a claim
that the region has completed or that physical flags can be reset at its exit.

For h covered cells, n occurrences, c input incidences, e prerequisites,
a additional native edges, k pipes and f retained demands, expected work is
O(hk+n+c+e+k(n+a+f)); space is O(hk+nk+c+e+a). The hardware pipe set is fixed.
Source distribution is linear; there is no incoming-list sorting or read/read
pair scan. Counters distinguish input accesses, unique active cell visits,
reader-frontier visits, emitted witnesses, candidate tests and demand joins.
The Python test serializer copies rows for inspection and is not a production
regional-export adapter.

Tests construct all conflict pairs and explicit event graphs independently of
the storage scan, checking closure, actual covers, all event queries and
boundary selectors. They include native prerequisites, RMW, resolved presence,
physical byte overlap and every cut of a finite composition fixture. Geometric
read-heavy cases check inspected work up to 4096 readers. Enumerating guard
valuations in tests does not implement symbolic guarded analysis.

## Integration boundary

An IR provider must establish these input facts from admitted final IR,
including exact versus enclosed geometry and original occurrence identities.
Unresolved accesses cannot be omitted. Symbolic guarded circuits and compact
repeated-region analysis require separate constructions rather than unfolding
loops through this API. Logical completion tokens may express the resulting
dependencies, but executable guards, legal cuts, matching, invocation closure,
physical-ID reuse and device qualification remain separate obligations.

[The placed matmul prototype](13-matmul-sync-prototype.md) now supplies a qualified IR importer and compact repeated-region consumer. Its restrictions and independent evidence are documented separately.
