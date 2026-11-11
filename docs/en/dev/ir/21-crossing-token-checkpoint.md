# Numerical crossing-token checkpoint

This checkpoint extends `572d6e61b` from single-region histories to finite
numerical crossings between rotating regions. It preserves the existing
`ExportCrossingDemands` contract: unavailable is distinct from an empty inventory.
Nested source frames and production orchestration remain separate migration work.

## Construction and proof

A source occurrence inside a loop need not be its final visit. Each distinct
selected source occurrence gets one scalar capture, initialized to completed.
After the original source record, an `if` result chooses that record when the
original loop coordinate equals the selected coordinate; otherwise it preserves
the same incoming carry. The loop simultaneously yields the selected value.
Induction over visits establishes that the exported result denotes exactly the
selected completion prefix, or completed if that visit never executes.

The parser checks the equality in the original start/step frame, both pure token
yield arms, the same preserved carry, completed entry and an unconditional
same-invocation current record. It only admits capture use through the loop
result. Effectful arms, other carries, intermediate uses and noncanonical guards
refuse before output publication. No private history, new token operation or
opaque sidecar is involved.

Each actual consumer use becomes a distinct physical handoff. A consumer inside
a later loop uses an equality guard for its original occurrence. All supplied
crossing endpoints become ports before the shared boundary graph constructs
child-internal order links. Contiguity and reference-forward order ensure that
a path decomposes into those links and supplied crossing/native edges, without
leaving and reentering a child. The existing sufficient reuse and same-pipe
barrier checks remain authoritative. Allocation and physical emission are unchanged.

The implementation uses the existing logical realizer and structured parser,
plus the shared `SyncBoundaryGraph`'s optional supplied-port list. Mathematical
clients keep their original first/final ports by default.

## Independent evidence

Tests reuse original-access interpretation for covers, event queries and storage
selectors, and independently execute the actual token and flag programs. They
compare full payload-order relations, coverage, acyclicity, matching, physical
reuse and directional occupancy. Fixtures include interior-to-interior crossings,
resident prologue/stream/tail, empty children, counter resets, nonunit coordinates,
serialized captures, fanout, and absent source or target occurrences.

A manual edit selecting a later source prefix must produce the corresponding
stronger relation. A different test shortens a previously analyzed source loop:
its old captures become absent, and its old token program no longer covers the
changed access problem. Lowering faithfully preserves that supplied token order;
it does not silently rerun demand generation to repair the edited program.
This negative coverage result is explicit and is not counted as an exact
synchronization success. Other emitted-plan checks still pass for that token program.

Malformed-capture tests require immutable refusal. C++ generation retains PTOAS
autosync enabled and verifies that literal physical flags establish manual ownership.
No device execution or new native semantic claim is made.

The broad gate passed **1,117 tests** with 28 existing warnings. The final
rebuilt focused gate passed **121 tests**, including two additional shortened-
target cases and C++ generation. All three reviewers accepted the scoped change;
all configured repository checks passed.

## Representation and costs

Let H count retained shift-history state and B count distinct captured source
occurrences. Token state is `O(H+B)`; multiple uses share B while retaining their
physical identities. This avoids shifting `trip_count - source_visit` values to
recover one interior prefix. Explicit interfaces may still enumerate slots or
rows. Boundary construction remains `O(sum P_child^2)` plus child query costs,
where P includes the new ports; a crossing query uses `O(P+E)` graph work.

Three-slot derivatives use source counts 6, 9, 1,023 and 999,999, with a second
five-visit region. Each has eight carried facts, including two captures, four
record sites and seven logical wait sites. Counts divisible by three preserve the tested interior-crossing
shape. All measured physical PTO outputs are byte-identical to the unchanged
production route, with 14 SET and 14 WAIT sites. Token-stage medians are about
1.4–1.5 ms versus direct, with paired ratios of 2.4–2.5. This is host overhead,
not a device performance improvement. Large counts are size/timing checks;
small instances supply the execution evidence.

## Reproduction and limits

Use the qualified Linux environment from the [finite checkpoint](18-token-pipeline-checkpoint.md).
The [receipt](token-pipeline-m2b.json) pins commands, source/binary hashes, samples
and regression outcomes.

```bash
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m pytest -q tests/ut/ir/transforms/test_crossing_token_sync.py
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_token_pipeline \
  --crossings --output build-py314/token-m2b-qualification.json
```

The [pass contract](../passes/60-logical_sync_pipeline.md) remains the current
interface definition. Captures use supplied finite numerical occurrence choices;
this is not arbitrary symbolic selector transport. Shared-reset nested scopes,
caller range evidence through lowering, collective/nested token interfaces and
production migration remain pending. Existing production support stays available.

| Established evidence | Remaining premise | Appropriate paper claim |
| -------------------- | ----------------- | ----------------------- |
| Selected occurrence captures preserve actual token prefixes | Qualified frames, cuts and pure capture recurrence | Executable transport for finite numerical interfaces |
| Complete crossing ports preserve bounded token/flag relations | Supplied child order and all actual crossings | Qualified regional composition |
| Fixed-slot schemas stay compact as trips grow | Slot/row interface expansion charged separately | Trip compactness, not uniformly compact exports |
| Existing physical policy accepts the tested plans | Sufficient reuse evidence and native assumptions | Certified selected-plan assignments, not general token lowering |
