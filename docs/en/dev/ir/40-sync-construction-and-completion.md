# Compact construction and invocation completion

This checkpoint addresses the review of `bef68ac841798bc136b2956de9c8d3d505119268`.
The review's two compiler-scaling findings and resolver-documentation mismatch
were reproduced. Notification closure also did not establish final-payload
completion. These are implementation/qualification gaps, not counterexamples
to the admitted minimum-demand construction.

The companion [receipt](40-sync-construction-and-completion.json) pins the base,
dirty implementation sources, extension, inputs, commands and host results.
Existing untracked captures were preserved. No allocation algorithm, tiling,
protocol matching, scalar exemption or qualified accumulator exemption changed.

## Construction and validity

**Ordinary invariant loops use the existing periodic pipe frontiers.**
Invariant activity/class membership, original per-pipe order and nonnegative
wrap generators establish this query contract; a nested region tree is not a
premise. The obsolete event-by-event distance table and its query branch were
removed. Each participating pipe contributes two event-frontier rows, with
`O(km)` retained entries. Construction uses the existing weighted shortest-path
engine; this is not a claim that every compiler stage has linear cost.

**Rectangles use a complete row-sweep partition and one shared index.**
Only column boundaries of writers active in a row band split that band.
Reference counts preserve shared boundaries when another writer ends. Storage
bounds remain present, so holes and read-only domains retain coverage. A write
covers every class it intersects; reads select intersected classes. Thus the
partition preserves conflict pairs under the supplied layout/alias model,
following the draft's write-footprint conflict-class lemma.

The shared row/column index queries the full affine access envelope; the
existing exact all-visits membership proof still rejects changing membership.
An envelope is only a lookup filter. Original access descriptions continue to
answer point selectors. Missing exports never become empty answers.
For W writes and C produced classes, partition construction costs
`O(W log W + C)`. Indexed query work is charged to row bands and intersected
classes, including logarithmic searches. Arbitrary overlapping rectangles can
still produce quadratic C; no general rectangle-linear bound is asserted.

**Composition retains actual interface ports.** A larger static-site probe
revealed that the former boundary graph still exported every site's first and
last visit. It now retains native per-pipe boundaries and all crossing-generator
endpoints. A cross-region path enters/exits a child through one of these edges;
exact child queries replace its internal path. Arbitrary query endpoints bridge
through those same child queries. Both event kinds, empty children and original
coordinates are preserved. Crossings are generated before constructing the
required graph and then reduced. The actual-token client already supplies its
cross-family endpoints and shares the same graph service.

The general graph still costs `O(sum P_child^2)` to construct, plus child-query
and graph-search costs. P is now the actual interface size, not every payload
site. No implicit completion or resource reset is introduced at child boundaries.

## Invocation completion

Public `LowerSync` and `RealizeSync` now add the existing A2/A3
`system.bar_all`, with the owning core type, before returning an owned kernel.
This replaces the completion mechanism skipped when PTOAS sees explicit flags.
A canonical terminal return and evaluated return operands are required;
unsupported exits refuse transactionally. Every supported return receives the
completion. Already lowered public input remains unchanged, while optional
insertion still rejects reprocessing an owned kernel.

Internal `LowerLogicalSync` and child exports do not receive this wrapper.
Closed notifications, pending payloads, and physical reuse remain separate
properties. The drain establishes modeled payload-to-return completion under
the existing ALL-pipe contract; it does not consume unmatched flags. Host tests
remove precisely the drain and show that matching still holds while final-store
completion at return is no longer established. This models the prerequisite for
a dependent next task or immediate local-storage reuse, not a device execution
of those scenarios. Existing internal payload-order equality checks remain.

No CCEC machine-code or device validation was performed. Fresh-entry and native
instruction contracts still apply. Historical device comparisons omitted this
drain on the PyPTO side; their performance claims cannot be transferred to this
revision. Future comparisons must use identical invocation-boundary policies.
The previously issued device task remains pinned to its original revision.

## Measured construction

All probes keep trip count at three and use generic placed Buffer operations.
No kernel names or shape-specific admission rules were added.

| Probe | Reviewed revision | Repaired implementation |
| ----- | ----------------: | ----------------------: |
| 128 static loop sites: pairwise distance entries | 65,536 | 0 |
| 128 static loop sites: retained frontier entries | 0 | 512 |
| 128 diagonal stores: classes | 16,386 | 384 |
| 128 diagonal stores: membership tests | 2,097,282 | 258 |
| 1,024 static loop sites: boundary pair tests | not recorded initially | 24 |
| 1,024 diagonal stores: classes / membership tests | not run initially | 3,072 / 2,050 |

The loop's retained frontiers grow as `4m` on this one-pipe body. The diagonal
family has `3N` complete-coverage classes and `2N+2` membership tests. Its
logarithmic lookup work is recorded separately. Static sites, not merely trip
counts, vary geometrically. The initial repair exposed remaining quadratic
boundary construction; after port compaction the 1,024-site descriptive sample
spends about 41.8 ms importing, 1.62 ms reducing and 0.012 ms composing (43.5 ms
whole analysis). These are single host measurements, not paired speedup evidence.

Independent bounded checks interpret physical addresses directly from input IR
and compare actual retained covers, closure, all four event queries and first/
last storage selectors. Cases include zero/one/multiple visits, interior crossing
ports, overlapping/shared writer endpoints and unwritten/read-only holes.
Physical checks interpret actual tokens and emitted consuming commands; the
terminal drain is checked separately from unchanged internal order and reuse.

## Reproduction and remaining limits

Run from the compiler worktree, using its own built extension:

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 1
export PYTHONPATH="$PWD/python:$PWD/tests/ut/ir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PTOAS_ROOT="$PWD/build-py314/architecture-correction/ptoas-launcher"
python3 -m pytest -q tests/ut/ir/transforms/test_sync_construction_scaling.py \
  tests/ut/ir/transforms/test_token_sync_lowering.py --require-ptoas
```

The final gate reports **647 passed, 1 failed, 0 skipped**, with two existing
inline-helper deprecation warnings. The only failure is the documented small
Shenggan Buffer-admission case. The focused scaling/semantic module passes 34/34.
Code and report passed independent review. The receipt records the exact module
list and outcomes. Scaling
fixtures are `scaling_fixture(m)` and `scaling_fixture(N, diagonal=True)` in the
new regression module; call `testing.analyze_placed_sync(f, True).describe()`
after selecting the Ascend910 backend to obtain the recorded work counters.

The known small Shenggan GEMM remains Functional IR and is refused before
InsertSync; this earlier Buffer-admission failure was not repaired here. General
interface growth, consumer-domain-dependent token resolution and unsupported
native footprints remain separate limits. The English and Chinese pipeline
guides now state the resolver's actual per-use `O(ED)` pattern work, with E up
to `UT`, and separately charge potentially quadratic normalization candidates.
