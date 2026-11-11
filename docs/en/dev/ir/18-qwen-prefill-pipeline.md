# Qwen prefill pipeline milestones

Starting implementation: `62ab48ed54325ee3cddfdb8b8579410aaa87d108`.
This extends the [buffered pipeline checkpoint](17-buffered-pipeline-checkpoint.md).
The target is the complete output-projection and combined gate/up-projection
compute bodies, extracted from PyPTO-lib prefill with explicit caller contracts.
This is not qualification of the whole Qwen model or its attention body.

Each milestone requires separate architecture, correctness, and
algorithm/performance acceptance before the next implementation starts.
Review findings and exact source identities belong in the checkpoint receipt.

| Milestone | Implementation and acceptance |
| --------- | ----------------------------- |
| 1. Inputs | Pin original bodies/configuration; retain caller and launch evidence, physical accesses, guards and original frames. Unknown ranges remain unavailable. Validate output and gate/up active workers independently. |
| 2. Resident and nested phases | Generalize existing leaf/merge services to sequence/repeat bodies, including startup and accumulator reset. Whole-write conflict classes retain separate exact point selectors. Check actual covers and all four event queries. |
| 3. Crossing exports | Export original scoped occurrences and selectors without unfolding K. Prefer existing numerical interfaces for numerical counts; implement interval/modular algorithms only where their premises and benefit apply. |
| 4. Physical transport | Consume scoped families through existing allocation and emission; verify matching, original cuts, directional palettes and causal reuse. No allocation-policy changes. |
| 5. Qualification | Original output then combined gate/up through ordinary opt-in production and C++ generation. Independent bounded semantics, geometric scaling and a reproducible remote device matrix. |

For the output projection, the original body loads two A/B operand pairs,
then executes two inner extraction/accumulation loops. Each L1 fill supplies
readiness to its own first reader; its last MTE1 reader releases that storage.
The final fill of the second pair must not automatically gate the first pair.
The accumulator remains resident across both inner loops and all K pairs.

The output schedule launches 20 workers for 20 tiles. Gate/up launches 24
workers for 17 tiles per projection, with a rotated starting worker and an
additional shift for up. Its two loops are independently zero-or-one visits.
A skipped phase does not refresh storage or reset notification resources.

`placed_sync_import.cpp` previously joined bounds only at direct call sites.
Milestone 1 now propagates established bounds through an indexed acyclic call
graph, joining all callers before visiting a callee. Recursive or unknown
caller evidence remains unproved. Launch coordinates are separate from
formal-parameter indices. Passing a sample scalar to JIT is not a range proof.

The implementation preserves the scalar and qualified accumulator hardware
rules, six-ID budget, transactional PTOAS delegation and zero-inner behavior.
Exact modeled demands, original footprint qualification, order-preserving
realization, physical reuse and native invocation completion are separate
claims. No device result is established by these host milestones.

Current status: all five milestones accepted by architecture, correctness and algorithm/performance reviewers. The final fresh focused gate passes 550/550; device qualification remains outstanding.

## Milestone 1 evidence

The pinned PyPTO-lib source is `f2c80074c0b06ab5bf87fc2bfb06db417fcd35f1`.
The fixture manifest records all source hashes and wrapper adaptations. Both
extracted compute bodies are checked against the original AST. The original
symbolic ABI remains unqualified where caller/extent facts are absent.

| Premise | Source and scope |
| ------- | ---------------- |
| Ready input tensors | Explicit standalone invocation contract; this replaces the original external task dependency, not its payload computation. |
| Layer range and weight extent | The configured caller loops over two layers and declares `2*HIDDEN` weight rows. The retained worker formal at ABI position 3 is the layer index in `[0,1]`. This is not inferred from a sample scalar. |
| Logical block index | Original SPMD launch: 20 workers for output, 24 for gate/up. Call-site launch metadata overrides the function default. Unknown launches and unsupported direct-AIC callers invalidate the evidence. |
| Band | Each standalone configuration explicitly fixes one of the original four bands. Unconfigured runtime band remains unproved. |
| Aliasing | Distinct GM arguments are disjoint under the explicitly enabled contract. Physical local buffers retain their placed addresses. |
| Native footprints | The host oracle interprets the declared Buffer load/extract/matmul/store geometry. It does not independently qualify instruction microarchitecture or invocation completion. |

The independent control interpreter checks all 20 output workers and all 24
gate/up workers in each of four bands, at both layer extremes: 232 worker
executions. It observes initialization, accumulation and skipped phases from
original IR. Twelve smaller original-layout executions additionally enumerate
actual modeled access sets (hidden-size derivative 512, bands 0 and 3, layers
1 and 0). Their allocations and accesses are interpreted independently of
compiler partitions and demand generators. These are input checks, not cover,
matching, or allocation checks for the new nested kernels.

The fresh serial gate passes **250/250**, including parameter ranges, compact
nested repetition, buffered realization, regional slot and streamed composition
regressions. Eleven warnings concern retained upstream inline direction
annotations; no expectation or watchdog was relaxed. Full `.compile()` for both
configured projection entries produces C++ through **PTOAS delegation**. The
current PyPTO refusal is `Unavailable affine synchronization access`; launch
facts are exported separately and the flat importer does not yet consume the
nested body. This is the M2/M3 boundary, not an M1 synchronization success.

For hidden sizes 1024, 5120 and 51200, output caller proof retains 3 function
nodes, 2 edges, 156 graph-expression visits, 180 range-pass expression visits
and 45 range visits; gate/up retains 3, 2, 256, 316 and 82 respectively. The
512 cases contain less original IR after singleton-loop lowering. These are
static input-proof counts, not demand-analysis scaling evidence. Recorded wall
times are single descriptive runs, with no speedup claim.

Reproduce from this worktree using its Python 3.14 build and pinned PTOAS 0.72;
PTOAS's launcher needs its compatible Python 3.11 on `PATH`. Load the machine
limits and run serially when other local work is active:

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
cmake --build build-py314 --parallel 1
"${PYPTO_HOST_PYTHON:?set to the Python 3.14 executable}" -m pytest tests/ut/ir/transforms/test_qwen_prefill_pipeline.py \
  tests/ut/ir/transforms/test_sync_range_proofs.py \
  tests/ut/ir/transforms/test_parameterized_sync.py \
  tests/ut/ir/transforms/test_nested_sync_repetition.py \
  tests/ut/ir/transforms/test_buffered_sync_realization.py \
  tests/ut/ir/transforms/test_regional_slot_composition.py \
  tests/ut/ir/transforms/test_streamed_slot_composition.py -q
"$PYPTO_HOST_PYTHON" -m tests.ut.ir.transforms.qualify_qwen_prefill \
  --output build-py314/qwen-prefill-m1 --native-source
```

The receipt `18-qwen-prefill-m1.json` pins dirty implementation, input, extension
and PTOAS binary hashes. M1 acceptance does not establish demand covers, selectors,
physical reuse or PyPTO code generation for the full nested bodies; those remain
the next milestone gates. Native/device qualification remains separate.

## Milestone 2: supported numerical resident/nested composition

M2 adds an original Sequence/Repeat tree over the existing finite/periodic
leaves and numerical merge service. Source paths retain original loop identities,
starts, steps and reset coordinates; shared summary nodes carry no substituted
source identity. Moving reads are admitted only with invariant whole-write
conflict-class membership. Composition consumes class boundaries, while point
selectors independently solve the exact inner membership interval and lift its
first/last corners through enclosing repetition.

Structured periodic leaves reuse the existing per-pipe core-frontier builder.
They do not retain the older quadratic all-event distance table. Source path
construction uses push/pop traversal and copies only final leaf coordinates.
Numerical merge work is charged to supplied ports and storage interfaces; a
large physical partition is not claimed to be free or universally linear.

Startup is qualified within a leaf or its direct parent/leaf pair. The original
first-visit conjunction, operands and branch calls remain unchanged. First-touch
qualification covers the whole parent body and its physical aliases. A fresh,
immutable continuation-entry view shares internal mathematics for later visits;
the first body remains startup, including after enclosing resets. Empty bodies
supply no entry evidence. Existing accumulator footprint protection additionally
requires adjacent qualified M boundaries on the same accumulator and no
intervening reader (including FIX). It never becomes a fictitious completion-to-
start edge. Initialization spanning further intermediate repeats remains
unavailable because their continuation seams need additional evidence.

Review found and corrected an empty-descendant startup defect before acceptance.
It also identified missing intermediate-seam evidence; that case now refuses
explicitly. The independent original-IR oracle checks concrete modeled accesses,
expanded actual retained covers, closure, all four event queries and storage
selectors. It shares the declared native instruction/layout model, not the
compiler's storage partition or generator extraction. These are host checks,
not native qualification.

The serial focused gate passes **318/318** with eleven retained upstream inline
annotation warnings. Numerical fixtures include interleaved fills, nested sliced
readers, accumulator initialization/continuation, FIX interference, empty visits
and outer resets. New source-tree physical export still refuses explicitly;
previously supported nested physical/C++ paths remain covered by the gate.

| Scaling axis | Observed retained work |
| ------------ | ---------------------- |
| Static body copies 1→16, fixed classes | Frontier entries 60→960; edge visits 162→2592; zero dense distance entries; constant supplied boundary interface. |
| Static children 1→16, fixed classes | Interface entries 1,904→30,464; source paths and merge count track child definitions. |
| Independent written stage buffers 1→16 | Interface entries 1,904→24,584; increasing supplied class costs are reported separately. |
| Outer/replay count 1→1,000,000 | 14 analyzed occurrences remain retained; merge count grows 3→28 through shared repetition. |

Timings are single descriptive samples. No device or compiler-speedup claim is
made. Full configured output and gate/up still refuse affine import; M2 does not
claim full original Qwen compilation. M3 owns original-coordinate crossings and
M4 owns shared-repetition physical endpoint transport. Structured rotating-domain
import remains outside the new numerical source-tree contract.

Reproduce M2 measurements with the same environment above:

```bash
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_qwen_prefill \
  --output build-py314/qwen-prefill-m2 --numerical
```

`18-qwen-prefill-m2.json` records dirty source and extension hashes, semantic
counts, scaling measurements and precise export refusals. The focused gate log
and JUnit receipt are under `build-py314/qwen-prefill-m2-gate.*`.

## Milestone 3: scoped exports and invocation activation

M3 retains immutable demand schemas under their original `CompactSyncResult`.
Each schema carries original endpoint occurrences and named loop axes, with a
separate flattened stride for each endpoint. A sequence exports its retained
crossings; a repeat exports its body and one adjacent-body seam, repeated over
the original frame. Source traversal stops at retained child summaries, so it
visits definitions rather than every use of the binary repetition DAG. Empty
children remain transparent. Cached exports neither rerun reduction nor unfold K.
The finite-only physical adapter still refuses these schemas until M4.

The validity argument uses contiguous original children: internal covers stay
covers, and the retained merge supplies exactly the reduced crossings. Uniform
repetition has the same interior and exit rows in each visit; startup changes
entry qualification only. Consequently the first body-to-body seam represents
all admitted adjacent seams. Different endpoint strides preserve unequal child
sizes. Tests expand the resulting schemas and compare them with both the
retained DAG and an independently reconstructed original-access cover graph.

Ready launch coordinates are now distinct from scalar formals and original
loop ordinals. A singleton symbolic-start frame retains its actual start
expression; its normalized ordinal zero does not establish an original `iv==0`
initialization condition. Storage translations use the proved case's lower
block coordinate as their origin. Missing block/parameter values return an
unavailable selector; valid local selectors remain usable independently.

`SyncInvocationAnalysis` supplies a shallow partition for invocation-invariant
activation. It checks every arithmetic intermediate in its original type and
splits a proved single-orbit, unit-stride modulo domain at wrap and zero/one-trip
boundaries. Offsets must be constants or explicitly proved fixed parameters.
Every resulting nonempty interval must qualify before the owner is published.
Case selection validates all supplied caller bounds. Original IR and frame
identities remain shared; a case with no active payloads is exact absence.
Nonunit/multi-orbit modulo and nonfixed offsets remain unavailable.

This construction is case-sensitive: coordinate piece processing and all
per-case import/reduction/interface work are charged. It is not a uniform
linear-in-source-IR claim. Export size depends on supplied schemas and frame
depth, with no enumeration proportional to trip counts. Physical consumers
must retain the exact case owner because leaf IDs are owner-relative; case
exclusivity alone supplies no cross-invocation physical reuse proof.

M3 qualification uses `qualify_qwen_prefill --crossings`; final review and
results are recorded below when the campaign completes. Matching, physical
allocation and generated Qwen synchronization remain M4/M5 obligations.

The M3 focused gate passes **372/372**. The final campaign checks **34** small
original-layout executions at both ends of every activation interval, covering
2,226 payload/access instances, 1,436 retained covers, 696,632 event-query answers
and 3,240 storage selectors. All agree with the independent oracle. This includes
both-active and skipped gate/up phases, all four configured bands and both layer
extremes. These checks establish finite modeled-access agreement, not native
footprint qualification or a uniform realization proof.

Full configured output analysis uses one case; gate/up uses three or four.
Across hidden sizes 1,024→5,120, output keeps 41 analyzed occurrences and 17 scoped
families. Gate/up's per-band summed occurrences and family counts remain fixed
(164–246 occurrences, 73–112 families), while numerical merge interfaces grow
with shared repetition. Increasing independent outer/replay counts 2→1,000,000
keeps 14 analyzed occurrences, 12 families and 17 exported axis entries; merge
nodes grow 4→52. Case multiplication and interface costs are recorded explicitly.
Analysis samples are approximately 0.6–4 ms, descriptive only.

The independent oracle is much more expensive than production analysis: its
small cases take roughly 5–15 seconds. The aggregate 240-second campaign watchdog
interrupted validation after 32 completed cases. The incremental receipt was
resumed for the remaining cases under the same watchdog, after verifying identical
production, input, extension and oracle hashes. This was no compilation-timeout
repair. A test-only Boolean rewrite avoids constructing temporary read/write
unions without changing the protection predicate. The prior focused gate and
final campaign are reported separately in `18-qwen-prefill-m3.json`.

The ordinary physical path still reports its existing shared-repetition scope
refusal. No flags, barriers, allocator policy or native exemption changed in M3.

Final M3 rerun after the equivalent oracle optimization: **372/372**, 18 upstream
annotation warnings, 94.52 seconds. All three reviewers accepted M3 and verified
the receipt source hashes. M4 may now begin.

## Milestone 4: scoped physical transport

The physical adapter consumes the cached, provider-owned scoped families from
M3. It preserves logical occurrence tuples and original cuts; a constant
physical ID does not erase logical record coordinates. The existing allocator
and directional six-ID catalog remain unchanged. Only `PhysicalSyncInvocation.function`
is executable; its case entries retain allocation/cut evidence and deliberately
have unset function fields.

The initial certificate is sufficient period-one reuse. Each family must use
the same original enclosing-frame prefix at its endpoints, exact mixed-radix
strides and rectangular active intervals. Only its innermost axis may be
shifted or clipped. These facts come from the retained numerical source tree,
not detached affine maps. For each carrying digit, the producer checks
`C_consumer(last lower digits) -> I_source(next carrying digit, zero lower digits)`.
Every other member of that carry class translates the same contiguous interval
through invariant body and seam relations. A reference-forward witness cannot
leave and return to that interval, so incoming startup-only paths cannot justify
reuse. There are O(depth) carry queries and suffix arithmetic; each query's
retained-graph cost remains separate. Whole-span lane sorting uses original
reference order and costs O(F log F * depth) coordinate work. No claim of an
optimal coloring is made.

For same-pipe M/V demands, equal native-rank increments under the common axes
make predecessor equality invariant. The base must be the immediate active
native predecessor. A nonadjacent demand remains unavailable; DMA/FIX barrier
mechanisms remain outside this path. Scalar barriers remain forbidden and the
existing native scalar/accumulator exemptions are unchanged.

All invocation cases must retain the original owner and caller contract and
form an exhaustive contiguous launch partition. All allocations finish before
emission. The original dominating block-index value guards complete command
cut inventories, which the shared emitter installs in one traversal of the
original payload function. Initialization branches and payload sites are not
case-cloned. Case exclusivity permits palette sharing within this invocation;
it supplies no new native fresh-entry or return-completion guarantee.

The independent emitted-command tests cover mixed-radix carries, K=3/4/5,
zero and singleton dimensions, multiple resets, conditional activation and
both accumulator branches. They compare actual payload order with the modeled
required order, check matching, acyclicity, closed notifications and causal
consumption before republication. Generic nested Vec fixtures separately accept
adjacent barriers and refuse nonadjacent barriers. A non-dominating block-index
binding refuses without modifying original IR or invalidating mathematics.
The original gate/up combined program is tested at every activation boundary
for bands 0 and 3, including empty phases and both layer values.

| Measurement | Result |
| ----------- | ------ |
| Full output, hidden=5120 | 1 case; 17 SET and 17 WAIT sites; no barriers |
| Full gate/up, bands 0/1/2/3 | 3/4/4/4 exclusive cases; 73/90/112/90 SET sites and equally many WAIT sites; no barriers |
| Synthetic K and replay 3 through 1,000,000 | 31 analyzed occurrences; 17 SET, 17 WAIT, 22 guard sites; 29 carry queries; merges 6 through 58 |
| Repeated physical request | Zero new carry-certificate queries; fresh command inventory |
| Fifteen original-layout construction measurements | Analysis 0.55–4.67 ms; physical allocation/emission 0.18–1.14 ms; single serial descriptive runs |
| Full output and gate/up diagnostic C++ | PTOAS 0.72 succeeds; autosync enabled/disabled outputs are byte-identical |

Static gate/up counts include all mutually exclusive case cut inventories;
they are not per-worker executed counts. We do not claim optimal guard code or
a device speedup. Generic nested sharing fixtures still refuse when whole-span
reservation needs seven lanes; that sufficient bound is not an infeasibility
proof. The current admitted invariant producers supplied no counterexample reaching
the defensive period-one refusal; we did not fabricate a detached mutable
schema or call an earlier export refusal a reuse failure. Missing scoped
certificates remain unproved. Native entry/return
qualification and device execution remain separate.

Reproduce the physical measurements with the same environment above and
`python -m tests.ut.ir.transforms.qualify_qwen_physical --output build-py314/qwen-prefill-m4`.
The machine-readable receipt pins dirty sources, extension, input and tool.
M5 will separately test ordinary opt-in production selection; these diagnostic
exports do not establish that dispatch already selects the new path.

The expanded M4 gate initially passed 535/539. Four serialization/separate-compile
checks exposed an oracle Var/name fallback TypeError. The reviewed repair keeps
SSA identity primary and requires a unique active name when identity changes.
All three affected files then passed 128/128, resolving all four failures.
The aggregate is 539/539; this is not a fresh single all-green run. Both receipts
are retained. Architecture, correctness and algorithm reviewers accepted M4.

## Milestone 5: ordinary production qualification

`RealizeSync` now consumes the complete invocation owner when the closed
program supplies launch bounds. Functions without those bounds retain the
existing direct path. Both paths publish only a fully realized whole kernel;
no-marker candidates and qualification refusals retain the original body for
PTOAS. The pass remains opt-in, and PTOAS autosync remains enabled in both modes.
No kernel name participates in recognition or dispatch.

Dependency discovery walks the original scalar SSA DAG once per stage. A
block-independent loop start keeps the importer's existing full-domain range,
readiness and trip-count proof; the activation partition does not require an
unrelated bounded parameter to become a constant. Generic AIC/SPMD tests check
constant and parameter-only singleton loops under actual caller/launch facts.
Mixed block-plus-nonfixed-scalar starts still require the current fixed-offset
partition contract. Some such uniform singleton inputs could qualify through
the direct importer but conservatively delegate through this invocation path;
preserving that broader full-domain qualification is a remaining selection
improvement, not a demand-coverage or physical-reuse result.

The original full output projection and combined gate/up projection, for all
four bands, now select PyPTO through ordinary `.compile()` with the configured
caller. Ten full C++ runs compare the five configurations with PyPTO disabled
and enabled. All succeed. The enabled reports select PyPTO and retain downstream
autosync; the disabled runs use PTOAS. This is the entire extracted projection
body with the pinned caller adaptations, not the full Qwen prefill model.

Single-run full compilation was 88–148 ms. Included `RealizeSync` time was
1.09 ms for output and 3.67–5.28 ms for gate/up. These are descriptive host
measurements, not paired speedup evidence. The numerical/physical construction
sizes and disjoint stage measurements are recorded in M3/M4. The final focused
gate and review decisions are recorded in `18-qwen-prefill-m5.json`.

The device handoff is a local file in the top-level `pypto3_sync_more` directory,
not a branch document. It requires the exact source snapshot (the current
work remains uncommitted), original configured output/gate/up bodies, both
ownership modes, numerical checks and interleaved device kernel timings. No
device result is claimed here.

| Established result | Remaining premise or limitation | Appropriate manuscript claim |
| ------------------ | ------------------------------- | ---------------------------- |
| Shared numerical nested analysis with original reset coordinates and resident boundaries | Supplied numerical counts; one supported moving-access frame; qualified physical layout | A concrete Section 5.4/5.5 implementation for this admitted region class |
| Retained original-occurrence covers, all event-query kinds and point selectors agree with independent bounded input-effect graphs | Finite validation complements the construction argument; it does not qualify native instruction footprints | Exact minimum demands relative to the declared Buffer access model |
| Original-frame direct commands match and preserve modeled payload order on checked executions | Supported adjacent same-pipe placement, existing native scalar/accumulator contract; no arbitrary branch-history transport | Qualified direct realization; no universal arbitrary-kernel claim |
| Six-ID assignment and causal consuming-flag reuse for full configured projections | Sufficient period-one/whole-span certificate; some other fixtures refuse without infeasibility proof | Feasibility of this fixed protocol, not minimum physical budget |
| Ordinary PyPTO route and downstream C++ succeed | Native invocation completion and device behavior not proved | Host compiler qualification; device testing is a separate outcome |
| Unsupported kernels retain whole-function PTOAS delegation | Delegation is not the draft's conservative token construction | No claim of a general Section 6 implementation in this PyPTO milestone |

The general rotating interval sweep, symbolic crossing-selector theorem and
resident/guarded attention composition remain separate extensions. This batch
uses the existing numerical reduction/merge service for the supplied Qwen
counts; it does not claim to implement every pipeline theorem in the draft.

Final M5 gate: **550/550 passed**, 26 original upstream inline-direction warnings,
367.13 seconds serially. This fresh run includes the M4 oracle repair, all scoped
physical regressions, original production checks and uniform scalar-launch cases.
All three reviewers accepted the final milestone; no commit or push was made.

## Commit qualification after rebase

After rebasing onto upstream `7302cd7ff` and integrating the barrier API changes,
the native M5 implementation matched its accepted source hashes before commit
cleanup. That cleanup adds direct includes, standardizes header guards and adds
explicit braces and a Boolean cast without changing the algorithms. Later
Python changes narrow fixture types, preserve the oracle coordinate contract,
and repair documentation navigation. The expanded host gate passes **1,465/1,465**
with 49 warnings; no test expectation or watchdog was relaxed. The M1–M5 receipts
retain their historical source identities and review decisions. The M5 receipt
adds the post-rebase validation and current typing hashes. This qualification
does not add device evidence or change the stated native entry/return limitations.
