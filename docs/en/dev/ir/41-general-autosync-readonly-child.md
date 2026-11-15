# Read-only external effects in retained compact children

Status: production adapter and independent source review accepted. Final checks,
source pins and commands are recorded in the companion JSON receipt. This is an
explicit extension of the [local child contract](41-general-autosync-eventful-child.md),
not automatic fallback selection or a replacement allocation algorithm.

## Contract and validity

A whole closed invocation may retain its original compact analysis and token
recipe when every explicit and implicit DDR effect is read-only. The provider
checks the selected implementation's effects, not operation spelling. DDR reads
cannot conflict with each other or with local-memory effects. Removing their
geometry from internal conflict extraction therefore preserves internal required
order. All payload events and local effects remain represented. An external
write, unsupported effect, or missing native contract refuses this projection.

The original DDR roots remain in the storage inventory with unavailable boundary
coverage. They have no fabricated cell geometry. Boundary queries for that
domain return unavailable before absence or bounds tests. Local selectors and
all event queries remain useful. The conservative parent independently retains
**all original DDR reads** as sound enclosures: an overlapping parent write still
requires protection. This is exact internal reduction for qualified modeled
local accesses and read-only external effects, not exact physical GM selectors.
The older resource-empty native-child export retains its narrower contract and
explicitly refuses projected storage.

Preparation and actual-token lowering both qualify the policy. The default
import policy remains ordinary geometry. Lowering does not trust a child marker
as proof, and it does not recover a different minimal token program.

Original binding expressions retain immutable enclosing-frame contexts. Lazy
numeric evaluation is needed only where local mathematics uses the coordinate.
Ready pure scalar expressions may serve opaque GM addresses without supplying
numerical ranges. Every captured induction variable must lexically dominate its
original initializer use. Later guards cannot justify earlier arithmetic, and
same-name variables cannot substitute for original identities.

Physical tensor-shape symbols use the compiler's existing ABI owners:
`CollectParamDimSymbols` intersected with `ClassifySignatureSymbols.bound`.
Only original bare dimensions of formal tensor shapes are entry-ready metadata;
valid-shape-only or foreign symbols remain unavailable. This supplies readiness,
not a range bound, allocation extent proof, GM visibility rule or scalar-load
readiness. Data-loaded coordinates still require their own contract.

The existing parent entry acknowledgments, protected child commands, private
palettes and directional certificates remain authoritative. Participating pipes
include record sources and wait destinations even without payloads. Closed
notifications do not complete pending payloads. Parent summaries do not kill
those effects. The original logical protocol, six-ID pool, allocation algorithm,
expert plans, zero-inner-body barriers and native same-pipe exemptions remain
unchanged. Refusal commits no partial program or resource ownership.

## Independent validation

The focused eventful-child module checks actual retained covers, all four event
query combinations, local selectors and explicit unavailable GM answers for
small zero/short/repeated/reset/skipped visits. Original concrete accesses are
expanded independently of compiler partitions and generators. Aliased GM inputs
and parent writes overlapping those inputs test the projected-domain contract.
Negative fixtures cover absent enclosing frames, pre-loop use of a future IV,
foreign shape identities, unsupported native exports and insufficient resources.

The generic before/after qualification uses identical input, access contracts and
explicit conservative policy. It preserves payload/control traces, coverage,
matching, acyclicity and causal consumption before republication. This family
**does not have a general prerequisite-dominance guarantee**: one checked
three-visit case removes 39 pairs and adds 6, lowering excess from 74 to 41.
Its SET/WAIT counts each fall from 252 to 165 and barriers from 64 to 28.
The added pairs are already present in the selected logical wrapper; allocation
preserves that logical order. Command counts alone are not a quality proof.

For K = 2, 17, 1023 and 1,000,000 the generic retained child has 15 analyzed
occurrences, five cells, one region, nine query-scope nodes and five binding
definitions. Two external roots and one unavailable storage space are retained.
This demonstrates trip-count compactness for this family, not a bound for every
regional export or arbitrary symbolic nesting.

## Original Qwen invocation

The unchanged pinned `down_proj` capture uses the whole K-loop invocation at
`cb_inline1641__idx_v0`. It now prepares, composes logically, allocates through
the existing six-ID interfaces and generates C++. Its retained description has
40 analyzed occurrences, 13 cells, three regions, 29 query-scope nodes and
14 bindings. The shared source index has 62 nodes. Two external roots remain
explicitly unavailable for exact storage-boundary queries.

The independent command audit executes original ABI blocks 0 and 19 with the
explicitly supplied shape `LAYER_INTER_ROWS_DYN=34816` and scalar parameters 3=0,
4=0. These finite values are **not compiler assumptions**. Each execution has
443 payloads and 95,614 required completion-to-start pairs. The imported plan
removes 2,188 prerequisites and adds none relative to the previous selected
conservative policy: excess falls from 2,259 to 71. Executed SET/WAIT counts each
fall from 1,772 to 408, barriers from 444 to 31. The audit checks 372 causal
reuses; heavy directions use IDs 0–5. For each checked execution, the baseline logical/physical pair and the candidate
logical/physical pair each agree on their complete payload-event order. This is finite host-model evidence, not uniform dominance,
minimum physical capacity, device correctness or speedup.

The audit initially exceeded memory while retaining several expanded command
graphs and duplicate byte sets. Its research interpreter now keeps one command
arm at a time, interns **equal immutable full byte sets**, and uses persistent
conjunction nodes for the same pipe-prefix dependencies. Appending an acquire
does not order the previous prefix before that acquire. Exact bitset DAG closure
and tiny old/new event-pair comparisons protect this representation change.
A best-effort Linux RSS monitor refuses above 2 GiB. The original 120-second
watchdog is retained; audit work is separate from production construction cost.
Prior timeout and memory-refusal evidence remains in the disk-backed workspace.

Reproduce from the current worktree and its pinned Python/PTOAS environment:

```bash
python -m pytest -q --require-ptoas tests/ut/ir/transforms/test_eventful_sync_children.py
python -m transforms.qualify_eventful_sync_children --readonly-external \
  --output build-py314/readonly-child-new
python -m transforms.qualify_retained_sync_child \
  --input build-py314/column-transfer-20261115/qwen-prefix/before-sync.msgpack \
  --function down_proj --scope cb_inline1641__idx_v0 \
  --output build-py314/qwen-child-new
timeout --signal=TERM 120s python -m transforms.qualify_captured_child_commands \
  --input build-py314/column-transfer-20261115/qwen-prefix/before-sync.msgpack \
  --function down_proj --scope cb_inline1641__idx_v0 --parameters '{"3":0,"4":0}' \
  --shape-values '{"LAYER_INTER_ROWS_DYN":34816}' --block-index 0 --block-index 19 \
  --memory-limit-mib 2048 --output build-py314/qwen-command-new
```

For exhaustive retained mathematics, add `--check-retained` and request one block
index per run, under the same 120-second watchdog. This independently checks
416 original child accesses, 284 actual retained covers, all 692,224 event queries
and 86 boundary samples per execution, with two unavailable GM domains. The
original two-block exhaustive run completed block 0 but timed out during block
19; this timeout is retained. These per-invocation exhaustive checks are a
separate audit configuration, not a faster rerun of the combined request.

Outputs must be new directories. See the receipt for full source/tool/input
hashes, the integrated dirty-tree binary and distinct construction/audit timings.

## Paper consequences and remaining obligations

| Established result | Remaining limitation | Appropriate claim |
| ------------------ | -------------------- | ----------------- |
| Read-only external projection preserves qualified internal conflicts | Exact external selectors unavailable | Compact precision provider with explicit storage coverage |
| Original frames and ABI metadata retained | No invented range or scalar visibility premises | Coordinates and readiness are separately qualified |
| Existing entry protocol and directional assignment accept the child | Closed matching does not complete payloads | Executable retained-child composition under its wrapper contract |
| Finite Qwen prerequisite containment and generic tradeoff measured | No uniform quality or device timing theorem | Report selected-order coverage and realization separately |

Original full-model coverage is still partial: mixed full/partial ADD value semantics,
scalar GM visibility and cross-core transport completion/release contracts remain
separate blockers. This adapter does not establish whole-Qwen ownership.
