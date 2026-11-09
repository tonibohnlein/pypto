# Placed matmul demand-analysis prototype

This opt-in **research analysis** imports placed A2/A3 Buffer IR and computes
minimum demands for its declared operation-footprint and native-chain model.
It does not insert tokens, barriers or flags, change default dispatch, or allocate
physical IDs. It is not a native instruction or device qualification.

## Baseline and dependency decisions

The 2026-11-08 checkpoint uses PyPTO `1f25c43de09fe935574ffd815843dca26ccd13e9`
and draft `ad6c7ac5f353e1661597dfdbd4115b9da0d187a5`. The existing uncommitted
finite service was preserved and reused. [The receipt](matmul-sync-prototype.json)
pins its starting dirty files, final implementation hashes, binary, inputs,
configuration, commands and measurements. A commit alone does not identify this build.

All three selected examples lower on this baseline without cherry-picking:

| Open work inspected | Decision |
| ------------------- | -------- |
| [#3291 logical tokens](https://github.com/hw-native-sys/pypto/pull/3291) | Future realization interface; no dependency for analysis |
| [#3290 nested Vec pipelines](https://github.com/hw-native-sys/pypto/pull/3290) | Different initial scope; not a matmul prerequisite |
| [#3021 AutoTile/multi-slot lowering](https://github.com/hw-native-sys/pypto/pull/3021) | Conditional dependency; no blocker in these captures |
| [#3193 accumulator windows](https://github.com/hw-native-sys/pypto/pull/3193), issue #3132 | Preserve layout/window identity; no needed patch here |
| [#3161 explicit aliases](https://github.com/hw-native-sys/pypto/pull/3161), issue #3066 | Use final storage identities; unsupported views remain unavailable |
| Issues #1908/#1980 placement and #3128 integer bounds | Separate concerns; no allocator changes or inferred range facts |

These are decisions for the pinned prototype, not assertions about later PR
status. Taskboard 3.1 is the natural algorithm-design home; no taskboard or PR
was changed.

## Provider contract and interfaces

`ImportPlacedSync` and `AnalyzeCompactSync` are C++ services beside the existing
finite reducer. The internal `testing.analyze_placed_sync(function, disjoint_gm)`
seam returns an owning result with `describe`, `precedes` and `boundary` queries.
The importer operates after `LowerTileToBuffer`, with the **PYPTO** memory planner.

The supported contract is intentionally explicit:

- One AIC leaf; fully valid, byte-addressable rank-two buffers at fixed addresses.
  ND GM arguments are disjoint by an explicitly supplied caller contract.
- Full local writes; aligned Mat-to-Left/Right extraction; full operand reads;
  accumulator initialization/RMW; rectangular GM transfers. Operation data modes
  are checked against the existing Buffer registry. Partial fractal boxes,
  dynamic validity, unsupported views and overlapping incompatible layouts refuse.
- Local aliases with identical physical extent and layout share a domain. Different
  SSA names do not establish local disjointness. GM-relative offsets are never
  compared as absolute addresses across arguments.
- Sequential constant-bound loops with entirely `INDEX` affine read offsets.
  Every intermediate and loop increment must remain nonnegative and fit signed 64-bit range.
  The first-visit `matmul`/later `matmul_acc` alternative must have identical
  operands/destination. Both original calls and the loop frame remain owned.
- Occurrences are `(segment, normalized visit, site)`, with original start/step
  retained. Separate segments retain distinct identities across counter resets.
  Runtime-symbolic bounds, nested loops and general data-dependent guards refuse.

The model uses native I→C per payload and separate per-pipe I/I and C/C chains.
It adds no inferred scalar or accumulator completion→start exemption. The finite
service's explicit native-edge input is unchanged. Native macro expansion,
unit-flag effects, instruction scratch and stronger hardware order still require
qualification before claiming minimum demands for actual hardware execution.

Exact rectangle descriptions are retained separately from conflict classes.
Static write cuts partition domains. A read can use a whole class only after
proving it intersects that class on every visit, and every write covers the
entire class. Thus these classes preserve modeled conflicts even when a moving
read covers only part of a class. Missing membership proofs cause refusal.
Point selectors use the original affine rectangles, not those class enclosures.
An out-of-domain query returns `None`; a covered untouched point returns an
available answer with absent writers/readers.

## Construction and validity argument

1. **Finite regions:** retain the numerical scan, covers and rank rows. A finite
   root does not construct a periodic index or an all-pairs boundary graph.
2. **Repeated regions:** class memberships and writer roles are invariant. Each
   mutable class has a writer every body, so generators span at most one visit.
   A one-body summary plus a two-body representative window suffices. Keep
   reduced records sourced in its first body and translate/filter their endpoints.
   Reference-forward paths between a record's endpoints stay inside that same
   two-body interval; neither earlier history nor a later visit changes cover status.
   This is the draft's representative-window construction with p=1 and Δ=1,
   not expansion proportional to K. Initialization's omitted accumulator read
   does not change conflicts because that occurrence still writes the accumulator.
3. **Queries:** a weighted periodic event graph has separate start/completion
   nodes and 0/1 visit distances. Native wraps can pad a reachable path by whole
   visits. Minimum distances answer all four event queries after coordinate checks.
   This prototype stores type-pair distances; it does not implement the draft's
   smaller per-pipe frontier index.
4. **Composition:** import first/last ports and exact child reachability. Storage
   frontiers create crossings; native start and completion chains cross boundaries
   separately. A crossing survives iff deleting it removes its path. Child covers
   remain valid because regions are contiguous. A region exit adds no completion.

The relevant draft material is Section 5.2, the representative periodic-window
and quotient arguments, and Section 5.5's contiguous-region composition. Supporting
appendices are still listed as unreviewed in the manuscript; reading their
arguments and testing this subclass is not a full proof audit of Sections 5–6.

For the actual split-K capture, each K body has four extracts and two matrix sites
(the first matrix site preserves its initialization/continuation choice). Each
loop retains six demand families. Seven reduced crossing records connect loads,
the two loops and their stores. In particular, the first store's completion
precedes the second loop's accumulator initialization. Operand-slot reuse also
crosses that counter reset independently; no whole-child completion is invented.

## Independent results and scaling

| Original example, FP32 | Executed modeled payloads | Actual covers | Event-query comparisons | Boundary points |
| ---------------------- | ------------------------: | ------------: | ----------------------: | --------------: |
| `matmul_64`, 64×64×64 | 6 | 4 | 144 | 16 |
| `matmul_acc_64`, two K=32 chunks | 11 | 10 | 484 | 20 |
| `ddr_split_k`, 256×128×256 | 28 | 25 | 3,136 | 42 |

An independent interpreter traverses the original IR, evaluates loop/branch
coordinates and maps ND/boxed layouts to physical element addresses. It compares
accesses, all raw conflicts, closure, expanded retained covers, all four queries,
and boundary selectors. It does not consume compiler partitions or generators.
It does share the declared instruction/layout/native model: agreement is not
independent evidence that this model covers every native instruction effect.
The bounded oracle currently requires unique variable names; this is a test
fixture restriction, not a production importer assumption.

Fifteen streamed-input derivatives cover 0/1/2/3/5 trips and 1/2/3 invocations,
including resets: 230,944 event-query comparisons and 576 boundary checks pass.
Zero trips leave a caller-initialized output untouched; no uninitialized store is
introduced. One through four static stages are checked separately. Real AutoTile
K=64/192/256 derivatives also pass; their schedules are allowed to differ.

For two static stages and two invocations, increasing count from 2 to 1,000,000
retains **62 inspected numerical occurrences, 52 class tests, 800 periodic distance
entries and 3,208 boundary-port pair tests**. The boundary graph has 1,235 edges
at count 2 and stabilizes at 1,383 by count 4. It is bounded by the fixed port
inventory, not required to be identical at every short count.

Static-size costs are not uniformly linear. At 1/2/4/8/16 stages, port pair tests
are 808/3,208/12,808/51,208/204,808. Each child uses O(m(m+g)) periodic-index work
and O(m²) distances; regional port construction uses O(ΣP_child²), and crossing
verification performs graph searches. Static rectangle partitioning can also
produce a Cartesian grid. These deliberate research costs require further
engineering/approval before becoming an ordinary production pass.

Exclusive single-run import/partition/reduction/composition times in microseconds:

| Original | Import | Partition | Reduction + periodic index | Composition |
| -------- | -----: | --------: | -------------------------: | ----------: |
| Single matmul | 156.2 | 3.4 | 12.1 | 0.0 |
| Two chunks | 145.6 | 3.7 | 16.9 | 0.0 |
| AutoTile split-K | 218.7 | 4.9 | 80.4 | 50.8 |

Lowering took approximately 10.7–12.6 ms in this run. Timings exclude oracle work and
are descriptive, not paired speedup evidence. No emitted synchronization,
allocation, device compilation or execution was requested.

## Reproduction and next boundary

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 1
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 60 python3 -m tests.ut.ir.transforms.run_sync_matmul_prototype \
  --output build-py314/matmul-sync-prototype --draft "$SYNC_DRAFT"
```

`SYNC_DRAFT` names the pinned paper checkout. The receipt records the focused
pytest/CTest commands. The fresh gate passed **300 Python tests and two C++ tests**.
Ruff, formatting and diff checks also pass. Captures, logs and serialized inputs
stay in the disk-backed build directory. No broad system/device suite was run.

| Established evidence | Remaining limitation | Appropriate claim |
| -------------------- | -------------------- | ----------------- |
| Three real lowered matmuls match an independent modeled-effect oracle | Native footprint/macro/order qualification | Exact demands for the declared model |
| Uniform representative-window argument and fixed-K-size measurements | Runtime-symbolic/nested/general guarded admission | Compact repeated analysis for the admitted subclass |
| Cross-reset covers, queries and point selectors agree | Quadratic static port/index construction | Correct prototype regional interface, not the strongest complexity bound |
| Original calls, guards and coordinates are retained | Token placement, matching and physical reuse unimplemented | Analysis foundation for a later realization milestone |

The reusable next improvement is the per-pipe threshold index and compact
regional query interface, followed by a separately qualified token realization.
Do not infer that a demand-only success already supplies an executable autosync pass.

Review accepted this research-only scope after fixing addressless-allocation admission
and narrowing/overflow in affine intermediates. New regressions cover identical local
aliases and original loop starts/steps. All configured hooks use their existing
Python 3.12 environment (the pinned Torch 2.8 wheel is unavailable on Python 3.14);
only the hook interpreter default changes, with no checks disabled.

## Incremental AIV extension

The same retained owner now also accepts the qualified static Vec fragment
specified in [RealizeSync](../passes/59-realize_sync.md#finite-vec-coverage).
The original matmul results above remain historical evidence. Vec byte-domain
coordinates are identified by `byte_coordinates` in `describe()`: row zero and
column equal to the physical Vec byte address. This avoids treating typed or
partially overlapping placed roots as independent storage. Point queries remain
exact for admitted footprints, including holes between strided window rows;
no descriptor-derived enclosure is promoted into an exact access.

The extension uses shared `DenseBufferBytes`, Buffer operation validation,
elementwise recipes and the Buffer IR alias verifier. It does not import pending
upstream GM-subview or rotating-pool designs. Metadata writers and unsupported
coordinate frames retain explicit refusals. The original logical and physical
realization entry points consume the owner without reconstructing analysis.
