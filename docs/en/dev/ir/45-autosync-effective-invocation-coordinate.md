# Autosync effective invocation coordinates

This is a bounded M5 extension on top of accepted M2 commit `da5f6cd660`.
It changes neither launch scheduling nor storage layout. It recognizes an
existing original scalar coordinate and carries its proof through analysis,
logical construction and actual-token lowering. Broader M5 relational composition
and complete Qwen ownership remain unfinished. The
[receipt](45-autosync-effective-invocation-coordinate.json) pins the source
snapshot and coherent build. The fresh 72-test gate and source hooks pass;
final documentation review and accepted commit publication are pending.

## Provider-owned contract

The supported original expression is

```text
b = buffer.get_block_idx()
e = (b + r) % B
for i in range(e, stop, B):
    original payload(i)
```

The original INDEX addition, remainder and aliases must retain their types.
`B > 0` is constant and equals an independently established, singleton actual
launch cardinality. Every admitted caller must supply that count, and the full
original block domain is exactly `[0, B-1]`. The block-index hull alone does not
prove cardinality. A launch-count interval, a subset of the original workers or
a different count refuses. Existing all-caller invocation evidence supplies the
production count; an explicit diagnostic request must supply it separately.

The rotation `r` is an original invocation-invariant parameter with proved
nonnegative bounds. The original addition must not overflow its original INDEX
representation for any admitted block and parameter value. The modulo does not
repair an overflowing intermediate; narrowing casts are not erased. The named
coordinate and its supporting definitions must dominate the payload tree.
Matching names are not matching SSA identities, and a foreign loop variable or
parameter gains no evidence.

For each fixed admitted `r`, translation by `r` modulo `B` is a bijection on
`[0, B-1]`. Its inverse is subtraction modulo `B`. Therefore every original worker
maps to exactly one effective coordinate, and every effective coordinate is
covered without enumerating worker or parameter values. Nonnegativity and the
original arithmetic check establish the meaning of the actual remainder, not
merely that of an abstract mathematical permutation.

`SyncInvocationCoordinate` is immutable and constructible only through
`Qualify`. It retains the original function, expression, definition, parameter
contract, independent count and supporting identities. Imports reject a foreign
owner or parameter/case contract. This is one qualified coordinate family, not
a general symbolic inverse or address-swizzling construction.

## Analysis and executable consumers

The activation partition and placed-access coordinate graph use `e` as their
numerical analysis coordinate. They reject any unrelated use of the original
block lookup that would incorrectly be reinterpreted as `e`. Every interval must
still qualify; a successful representative is not a whole-invocation result.

Original payload expressions, loop IVs, starts, guards, reset identities and
parameter SSA bindings remain unchanged. Logical case guards and assigned-command
scheduling use the **original named `e` expression** at its legal cuts. The
structured parser retains that same coordinate owner. Lowering interprets the
actual supplied logical tokens, with the independently supplied cardinality;
it does not reconstruct a different token plan from the accesses.

Public case selection accepts the original worker and actual original parameter
valuation, then applies the checked mapping. Storage-boundary queries likewise
map and validate the original valuation. Missing values, out-of-contract values
or a query for a different selected case return unavailable. They do not guess a
representative worker or report a false empty selector. Saved/reparsed logical
programs recover the contract through the existing all-caller evidence and
original definitions, rather than a detached sidecar.

No synchronization, allocation coloring, resource budget or hardware exemption
is changed. The existing six-ID checks and command reuse proofs remain necessary
for every candidate. Recognizing a coordinate permutation does not by itself
establish any physical assignment.

## Construction work and readiness

The source index records assignment and loop occurrences; it does not recursively
visit unrelated scalar expression DAGs. One immutable alias memo compresses
original single-assignment chains across loop starts and candidate operands.
Cyclic aliases refuse. This memo is **not** lexical readiness evidence: a separate
bounded walk checks the candidate against the original dominating statement
prefix. A future or nested definition remains unavailable even if the global
index can resolve it.

The analytic interval service retains its existing node/piece counters. A compact
partition does not imply that all downstream interface or physical work has the
same bound. Tests vary launch cardinality independently from visits and retain
separate named counters rather than inferring complexity from one wall time.

## Independent qualification and status

The focused fixture uses one retained symbolic recipe and original block/rotation
valuations. The independent interpreter extracts actual byte accesses from the
original program, then compares actual retained covers, all four event-kind
queries and first/last storage boundaries. The physical interpreter checks the
actual emitted commands for required-order coverage, matching, acyclicity and
consumption before republication under the supplied eligible IDs.

The passed focused gate covers:

- Launch counts 8 and 24, active and empty worker visits, and rotations 0, 1 and
  `B-1`.
- Multi-trip stops 9, 16 and 19 with `B=8`, including repeated buffer reuse and
  final short orbits under one retained assignment.
- Public Insert/Optimize/Lower with serialization, original loop identities and
  independent caller-cardinality collection.
- Missing or mismatched counts, unknown callers, varying counts across callers,
  unrelated original block uses, foreign identities, narrowing and overflow.
- A 512-alias/512-loop discovery fixture with an unrelated depth-28 shared
  Boolean DAG; cyclic aliases and future definitions still refuse.
- Launch sizes 8, 24 and 1024 with equal analytic node/piece counts for the same
  two-case structure. This measures partition work, not every compiler stage.

Run serially with the source loader and pinned tool environment from the receipt:

```bash
timeout 120s python -m pytest -q --require-ptoas \
  tests/ut/ir/transforms/test_effective_invocation_coordinate.py
```

The earlier paused run had 12 passes and two boundary-selector failures. That
failed evidence is retained; it is not the final gate for the restored package.
The restored coherent build passes 72 tests in 6.37 s across this module and
`test_parameterized_token_sync.py`, under the existing 120-second watchdog.
The full applicable source hooks pass. The pinned source/build snapshot was
saved before subsequent GM-staging work; later dirty files are not substituted
for that tested snapshot.

A fresh unchanged two-layer Qwen conservative-prefix capture generates all 32
leaf C++ files in 6.637 s inclusive. Ownership remains 47 observed task-control
domains, 17 PyPTO leaves and 15 delegated leaves. This is unchanged coverage,
not a new root admission result. The time is one descriptive host measurement;
no device execution or performance improvement is established.

## Paper consequences

| Established construction | Remaining premise or limitation | Appropriate claim |
| ------------------------ | ------------------------------- | ----------------- |
| Provider-owned effective coordinate retains original identities and arithmetic proof | One explicit rotation family, ready parameter bounds and full fixed caller launch | Qualified invocation-coordinate adapter, not general symbolic composition |
| Existing regional algorithms consume analytic cases | Access/alias, export and physical premises remain independent | No whole-nest unfolding or new demand algorithm |
| Actual-token lowering retains original coordinate and launch evidence | Matching and physical reuse still require their existing certificates | Coordinate support does not prove resource feasibility |
| Broader M5 goals are preserved | Symbolic shape relations, carried state and exact GM interfaces remain incomplete | A partial reusable M5 slice, not complete Qwen autosynchronization |
