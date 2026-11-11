# Compact buffered-pipeline implementation

## Contract and staged acceptance

This work implements the numerical buffered-pipeline constructors in draft
Sections 5.4–5.5, starting from the qualified finite/periodic services. The receipt
pins the starting revision and source contents. Each stage requires independent
review acceptance before implementation of the next. Passing recognition is not
passing analysis, executable realization, allocation, or native qualification.

| Stage | Production construction | Required review evidence |
| ----- | ----------------------- | ------------------------ |
| 1 | Existing MultiBuffer allocation/selection operations, Tile-to-Buffer adapter, physical slot descriptors | Physical address agreement, aliases, serialization, verification, codegen and negative geometry/range tests |
| 2 | Common-stride modular extraction and periodic reduction | Independently extracted closure, actual covers, queries/selectors; no slot/visit expansion |
| 3 | Numerical chain interfaces and shared regional composition | Actual covers and queries across partitions/resets; indexed boundary work and retained-size bounds |
| 4 | Resident and streamed boundaries | Exact membership including padding/absence; reset crossings; justified visit ownership |
| 5 | Existing physical realization consuming the new interfaces | Matching, acyclicity, causal reuse, six-ID results and C++ generation; separate export/resource refusals |

The implementation keeps the current hardware exemptions and allocation
algorithms. Unsupported original Qwen caller ranges remain unproved. Whole-kernel
PTOAS delegation is not the paper's conservative token/dataflow construction.

## Stage 1: storage evidence

`MultiBufferType` already specifies uniform slots. The missing operations are
`buffer.alloc_multi(valid_extents, [effective_base])` and
`buffer.slot(group, index)`. Allocation owns the full group; selection aliases it
and has no payload read/write. A selected handle is not another allocation.
The minimal lowering uses existing PTOAS multi-tile declarations and selections.

For the admitted placed, whole-slot accesses, a family has base `a`, slot bytes
`w`, count `b`, and original selector `(s*iv+c) % b`. An original loop with
`iv=start+i*step` normalizes this to `(first+i*stride) % b`, where
`first=(s*start+c)%b` and `stride=(s*step)%b`. Original integer intermediates must
be proved nonnegative and nonoverflowing before this rewrite. Physical addresses
are `a+slot*w+within_slot_offset`. Identically placed compatible families share
identity; overlapping incompatible layouts remain unavailable. Different SSA
names never prove disjointness.

Native multi-tile lowering rounds slot strides to the target memory alignment.
The placed importer and native adapter therefore require the packed descriptor
stride and fixed base to satisfy that alignment. Padding is not silently ignored;
unqualified padded families retain a precise refusal. Singleton groups normalize
to ordinary native allocations. Native slot-count eligibility remains separate
from the numerical IR description.

One descriptor represents every slot. No loop visits, ring slots, or buffer bytes
are enumerated by recognition. Parameter-dependent phases, partial slot views,
and unknown geometry remain unavailable. A successful read-only
`testing.describe_placed_sync_input` returns access evidence only and explicitly
states that demand analysis was not requested. Stage 1 does not claim a periodic
reducer for the new families.

The complete-class recognition theorem starts from normalized exact descriptions;
access recovery and physical alias reconciliation are charged separately here.
Current alias reconciliation may scan `S` retained storage descriptions for each
of `V` static views/allocations, taking `O(V*S)` work. This stage does not establish
the normalized theorem's end-to-end recognition bound. Existing planner alignment
is a sufficient native-alignment condition and can be stricter than PTOAS's
minimum slot alignment.

## Validation status

Stage 1 accepted after independent architecture/performance and correctness
reviews and a serial **1,599-test passing gate** (88.37 seconds). The gate includes
ordinary Buffer selection, explicit lowering, serialization, physical access
agreement, negative alias/geometry cases and native PTOAS C++ generation.
Earlier failing gates are retained in the workspace receipt logs. Source and
binary hashes identify the accepted dirty worktree; no commit is implied.

The earlier five-stage baseline was implemented and independently accepted. See
[its historical receipt](buffered-pipeline-receipt.json). The interval and collective
follow-up below has a separate M1–M5 acceptance record.
No device validation.

## Stage 2: rotating generators and numerical queries

For each qualified physical family, normalization retains its common stride
`s`, slot count `b`, and exact static footprint fragments. Set `h=gcd(s,b)` and
`R=b/h`. Offsets in different residue classes modulo `h` cannot overlap. Within
one orbit, modular phases and body positions give a circular access order.
Strict preceding/next writer sweeps produce RAW, WAR and WAW generators;
read-modify-write occurrences participate as writers. Endpoint/distance records
are deduplicated before reduction. No slot, orbit period or loop visit is
expanded.

The quotient retains separate start and completion events, native chains with
wrap weight one, and generator weights equal to visit distances. For a pipe
with `h_p` sites, a multisource shortest-path pass seeded with `h_p-rank(a)`
computes `D_p(v)`. The reachable prefix offset is `beta_p(v)=h_p-D_p(v)`.
Thus a source of rank `r` reaches target type `v` after displacement `d` exactly
when `h_p*d+beta_p(v)>=r`. Separate start-origin and completion-origin indices
answer all four event combinations. Strict queries exclude identical events;
invalid occurrences are rejected. Wide checked arithmetic preserves thresholds
larger than a representable invocation without converting them to unreachable.

For a candidate generator, the largest incoming prefix value excluding that
edge identity is an exact alternative-path witness. Two maxima retain distinct
identities, including ties. A path using the tested edge before entering through
another edge cannot justify its deletion: that would require a nonpositive
cycle, excluded by reference-forward occurrence order. Restricting the quotient
to a finite invocation preserves the argument because every intermediate event
of a forward path lies between its present endpoints. Zero and incomplete final
periods therefore require endpoint clipping, not a new analysis.

`boundary_slot` solves each access's modular progression and affine point
membership, then selects the first/last writer and the per-pipe read-only
prefix/suffix endpoints. It preserves RMW identity, original visit coordinates
and absent or unreachable residues. A query that omits a multi-slot coordinate
is unavailable; an untouched covered slot has an available empty summary.
Moving read-only GM accesses retain point membership even though they add no
internal storage conflicts.

After supplied normalized fragments, extraction uses `O(A log(A+1))` comparisons
and linear sweeps. The two sets of per-pipe shortest-path indices retain
`O(km)` entries rather than a type-pair matrix. Counters distinguish fragment
normalization, sorting, sweeps and shortest-path edge visits. General static
rectangle normalization can inspect `O(A^3)` combinations of cuts and accesses;
this is separately charged by `atom_tests`, and is not the whole-slot bound.
Whole-slot families have one within-slot atom. Access/alias recovery is also
separate. Numerical descriptors larger than physical hardware capacity test
representation scaling only, not native storage feasibility.

This stage admits one invariant periodic region. Startup-dependent effects,
protected accumulator transformations, moving accesses sharing a written domain,
multi-region slot composition and new executable endpoint maps retain explicit
refusals. A missing executable export does not mutate the retained mathematics.

Stage 2 accepted: **1,649 tests passed** (87.57 seconds), independent architecture/algorithm and correctness reviews accepted.

## Stage 3: numerical composition and shared reset repetition

The new hierarchy references the retained leaf mathematics. Each leaf interface
contains exact storage/native ports, with separate ordered start and completion
chains. Monotone sweeps construct forward and reverse thresholds. A merge forms
writer/reader storage bridges and native links, deduplicates complete event pairs
with native priority, and reduces demand crossings using inclusive rank buckets
and two distinct witnesses. Parent rows combine same-child reachability with
one-crossing reachability; query indices retain suffix minima and prefix maxima.
All-event queries navigate the endpoint uses and their common ancestor. They do
not build an all-pairs matrix or search a reconstructed boundary graph.

Correctness depends on contiguous original children. A path crossing a boundary
uses exactly one crossing, so excluding a candidate crossing identifies precisely
its alternatives. Internal covers remain internal covers. Empty children pass
storage and native frontiers through; they introduce no payload completion,
notification consumption or ID reset.

Perfect numerical nests with invariant physical effects and an inner modular
slot selection now retain the inner invocation once. Binary squaring shares
immutable nodes and translates numerical identities per use. The public
`(segment,visit,site)` remains an original mixed-radix occurrence identity, while
access formulas use the local inner visit. Thus the slot phase resets at every
outer invocation. Leaf demand families apply only inside their declared scope;
node crossings represent the reset obligations. Different uses of the same DAG
node remain different occurrences. Outer-dependent geometry, startup predicates
and unsupported accumulator qualifications are not inferred from this mechanism.

A bounded research operation expands the **retained** demand descriptions for
comparison with the independent finite oracle. It never performs a new analysis.
Flat crossing inventories are marked unavailable (`None`) for hierarchical
results; their lack of materialization is not an empty answer. At Stage 3 acceptance, physical exporters
refused these results; Stage 5 now binds original endpoints for qualified flat hierarchies.

Complete interface construction enumerates visited physical slot support, bounded
per access by `min(inner_count, orbit_period)`, and static within-slot atoms. This
is charged interface output work, not the slot-count-independent Stage 2 query
bound. No joint bank period is formed. A globally read-only physical domain needs
no storage bridges, but retains point selectors. Disjointness remains scoped to
the alias-qualified input and the analyzed visits.

The indexed threshold/reduction kernel implements `O(kP+k²r)` work. The current
adapter additionally inspects `O(kS)` summary roles, normalizes ports and bridges
with ordered sets/maps, performs endpoint lookups, and stores `O(kS)` summaries.
These costs include `O(kS log(kS+1) + r log(P+1) + P log(P+1))` work per merge
beyond the indexed kernel; they are not presented as the full manuscript's word
bound. Counters expose access visits, slot visits, bridge candidates and port
normalization visits separately. The existing no-ring qualified accumulator
construction remains unchanged; an unused slot allocation does not redirect it.

The descriptive host campaign holds three body sites and three slots fixed:
inner counts 4, 1,000 and 1,000,000 retain **2,240 interface entries and four merge
nodes** for seven outer visits. Increasing outer visits from two to one million
retains the same three analyzed sites and grows shared merge definitions from
one to 25, with 11,648 entries at the largest case. This demonstrates logarithmic
shared-definition growth for these admitted reset bodies, not general nested
Qwen coverage or physical flag feasibility.

Stage 3 accepted: **1,712 tests passed** (87.70 seconds); independent architecture/algorithm and correctness reviews accepted.

## Stage 4: scoped streamed ownership

Both independent reviewers accepted the following sufficient construction before
implementation. For one alias-qualified physical domain, every modeled access
must have the same translation vector. The union of all relative rectangles,
including reads and overlapping stores, must fit between consecutive translations
in at least one physically injective axis. This establishes disjoint storage
between local visits. Non-wrapping writer sweeps retain every within-visit RAW,
WAR and WAW generator; only cross-visit storage conflicts for the certified domain
are absent. Native order and other domains' generators remain unchanged.

The certificate belongs to its original region and invocation frame. Exact point
membership and first/last writers/readers continue to use the original affine
accesses. Padding and holes are genuine absence answers only inside the covered
domain; unsupported domains remain unavailable.

A parent can omit materializing this domain's crossing ports only when **all**
its accesses belong to one unrepeated child. An outside reader invalidates this
shortcut just as an outside writer does. Shared repetition that resets a streamed
output address cannot inherit the inner ownership proof. The initial adapter
refuses that case; it does not omit reset conflicts or expand the nest. Stationary
resident and rotating local storage still use the accepted numerical merge.

This is a restricted realization of the draft's streamed-family premise, not a
claim of general nested output composition. Existing qualified layout and alias
contracts are prerequisites; different tensor names do not prove physical
separation. No new caller range, native footprint, allocation policy or invocation
completion semantics is inferred.

Stage 4 accepted: **1,758 tests passed** (91.22 seconds), with both independent reviews accepted. Streamed counts 7, 1,000 and 1,000,000 retain 2,128 interface entries; the output domain is queried by exact affine membership rather than enumerated per visit.

## Stage 5: qualified physical endpoint adapter

The existing allocator consumes the exact `Threshold(C_target,I_source)` query
instead of indexing a dense event-distance table. Its lane bound, coloring,
directional pools, native reservations and same-pipe placement checks remain
unchanged. For a demand displacement `d`, a reuse separation `q` is sufficient
when `q >= d + threshold`. Every witness edge is reference-forward, so a path
between two present endpoints stays within their finite invocation interval.
The adapter clips against the family count before wide addition; unavailable
reachability never becomes a zero threshold.

Flat numerical hierarchies export retained crossings in original coordinates,
without expanding periodic families. An interior source or consumer is guarded
by its exact occurrence range. Export checks all shared-use scopes first, then
validates endpoint bounds and order. Reset repetition requiring repeated scoped
families remains unavailable; refusal preserves the retained mathematics and
transactional delegation. This is separate from a physical capacity refusal.

The specialized physical route does not require implementing arbitrary frontend
token lowering. Logical-token transport for displacements beyond the existing
contract remains a separate unavailable capability. Matching, fresh-entry and
closed-notification-exit obligations are checked under the existing command
model. Closed notifications do not establish native payload completion at return.
Native slot eligibility and C++ generation are reported separately from analysis
and flag allocation; no device qualification is implied.

Stage 5 accepted: **1,796 tests passed** (94.44 seconds), including **38 focused
physical tests and four PTOAS C++ cases**. Both independent architecture/algorithm
and correctness reviewers accepted the source and focused evidence before the
broad gate. No device tests were run.

### Measured scope and remaining obligations

For 7, 1,000 and 1,000,000 visits, the streamed example retains three demand
families and emits **9 SET / 9 WAIT static sites, no barriers and 76 arithmetic
operations**. The resident-reader/prologue/stream/tail variant retains six
families and emits **12 SET / 12 WAIT sites, no barriers and 97 arithmetic
operations**. These counts are invariant across those trip counts; source byte
length changes with literal widths. Under the six-ID restriction the bounded
executions use at most **three IDs per direction**. Independent finite widths
agree for those executions; this is not a general minimum-allocation claim.

The seven-visit streamed execution has 21 payloads, 18 publications and 18
consumptions; its checked command graph has nine causal reuses and no excess
payload-order pairs. The resident-reader variant has 24 payloads, 21 publications
and 21 consumptions, ten causal reuses and no excess pairs. Actual original slot
selections, commands and endpoint identities are interpreted independently of the
compiler's demand families and allocator inventory.

Single host measurements put analysis around 0.32–0.38 ms for the streamed case
and 0.68–0.76 ms for the resident-reader variant. Physical preparation, allocation
and emission are measured together, around 0.05–0.14 ms. Ordinary production
measurements include analysis again and must not be added to those stage times.
These are descriptive runs, not a speedup experiment. Native C++ success is
separate from these source-IR timings.

The writer-only prologue remains a useful negative: overwriting its local buffer
requires a same-DMA-pipe demand, whose barrier mechanism is not qualified by the
current physical exporter. The separate resident-reader fixture includes an
actual intervening vector operation and qualifies through cross-pipe demands.
No payload was inserted into the original case and no hardware rule was changed.

| Draft construction | Established evidence | Remaining boundary |
| ------------------ | -------------------- | ------------------ |
| Common-stride rotating demands | Exact selected-model covers, four event queries, modular selectors; no trip or joint-period expansion | General alias/layout recovery and startup-dependent rotating effects |
| Numerical regional composition | Independent flat/composed covers, queries and selectors across contiguous regions and resets | Complete interface output may enumerate physical slot support |
| Shared reset repetition | Retained mathematics and logarithmic shared-node growth | Physical repeated-family transport is explicitly unavailable |
| Streamed ownership | All-access translation certificate and exact membership; single-owner regional export | Outer reset or outside reader needs a separate crossing construction |
| Direct physical realization | Actual matching, causal reuse, six-ID assignment and C++ for qualified periodic/flat regions | Unsupported same-DMA barriers, native entry/return and device qualification |

### Reproduction

Use the source, extension and tool hashes in [the receipt](buffered-pipeline-receipt.json).
The serial gate command and environment are recorded there, including the Python
interpreter required by the pinned PTOAS launcher. After a local serial build:

```bash
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python3 -m tests.ut.ir.transforms.qualify_buffered_sync \
  --include-hierarchy --include-streamed --include-physical \
  --output build-py314/buffered-final-scaling.json
```

The receipt retains each accepted stage and the earlier failed attempts. That receipt predates the baseline commit. Default dispatch, the resource
pool, allocation algorithm and hardware exemptions are unchanged.

## Interval and collective-interface follow-up

The next implementation starts from compiler `e10d882b2348` and draft
`3a7c65b`, on a separate branch from the pinned device campaign. Its five
review gates cover exact interval evidence, interval-sweep generation,
collective-row boundaries, numerical composition/realization and final
qualification. General symbolic crossing selectors and allocation changes
remain outside this work.

### M1: exact interval evidence

`SyncAccess::physical_interval` is an optional half-open byte range inside
its canonical slot. The existing Vec byte domain uses its canonical byte
coordinates. Modes, slot selection and occurrence frames remain separate.
The importer qualifies whole boxed tiles, dense row/column-major contiguous
rectangles and existing exact byte intervals. Partial boxed footprints,
noncontiguous rectangles, moving/parameter-translated accesses and enclosures
retain their original effects without an interval certificate. Missing evidence
is not absence. No demand, merge, endpoint or allocation algorithm changes.

Qualification visits each supplied static access once. Existing Vec access
recovery can already enumerate strided rows; these input intervals are retained,
not merged across gaps. Thus M1 does not claim row-independent access recovery
or the full draft interval-list recognizer. Native footprint qualification
remains the existing operation contract.

Run `python -m tests.ut.ir.transforms.qualify_buffered_intervals --output
build-py314/interval-m1-baseline.json` for the pre-sweep baseline, using the
worktree-built extension. It records source/binary hashes, inspected work,
retained demands, physical outcomes and physical IR hashes when available.
The five baseline ring cases retain their existing same-pipe realization refusal;
this is not a capacity witness. Timing samples are descriptive.
The first test attempt used an obsolete default PTOAS launcher; all four C++
failures were environment failures. Reruns use pinned PTOAS v0.72 with its
compatible Python on PATH. No source or expectation was changed to repair them.

M1 accepted by architecture, correctness and algorithm/performance reviewers.
The focused gate passed **206 tests** with 18 existing inline-annotation warnings.

### M2: interval-sweep extraction

The existing rotating service now queries latest-writer/latest-access range
assignment trees on compressed physical endpoints. It batches all fragments of
one canonical occurrence before updates, sweeps two circular copies and compares
candidates across families by original relative source identities. Only the
latest conflicting source on each pipe is retained for each target. Earlier
sources reach it through native completion order; finite clipping preserves the
selected source whenever an earlier source is present. The existing quotient
reducer and physical-point selectors consume these generators unchanged.

A whole domain takes this path only when every access has exact stationary
interval evidence. Other domains keep their existing geometry construction and
separate costs. The `km` generator bound applies to fully interval-qualified
storage generation, not the union with arbitrary fallback contributions.
Temporary range trees are extraction scratch; neither trips nor slots are
enumerated. Counters distinguish input fragments, compressed endpoints, tree
queries/updates/node visits and allocated tree entries. The entry counter is
total allocation across groups, not peak live memory.

Partial rotating Mat extracts now use the same contiguous-interval qualification
as retained access export. Partial boxed/noncontiguous extracts remain
unavailable. Independent original-address tests cover nested, disjoint and
partly overlapping views, equal-key RMW batching, different offsets at one
payload, and different-period families sharing a source pipe. No verifier or
alias contract was relaxed: dynamically different Vec slots remain unavailable
as potentially overlapping in-place operands unless the operation contract
establishes legality.

For 4/16/64/128 nested-prefix readers the new path retains 10/34/130/258
generators and inspects 126/654/3558/8178 tree nodes, with zero atom/access tests.
The old atom partition would inspect every prefix at every row partition.
Existing ring baseline demands and physical refusals are unchanged. The new
partial Mat fixtures also retain the explicit `Unavailable same-pipe barrier
mechanism` refusal; C++ generation is therefore unavailable for those fixtures.
This is an executable-contract boundary, not an ID-capacity result or a demand
failure. Existing accepted physical routes are checked separately.

Reproduce the interval campaign with the M1 runner plus `--partial`. Receipts
record descriptive single-run times, not paired speedup evidence. Access
recovery, native footprint qualification and interface enumeration retain their
previous scope. General symbolic crossings and allocation changes remain out
of scope.

M2 accepted by the architecture, correctness and algorithm/performance reviewers.
The focused gate passed **329 tests**, including existing physical/C++ and Qwen
regressions, with 18 existing inline-annotation warnings.

### M3 design: certified collective crossing witnesses

The next scoped provider consumes a complete sequence of full increasing row
fills and finite row-interval reader phases over one qualified Vec byte domain.
Every fill writes each row exactly once; readers on other pipes cover all rows
before the next fill. All other operation effects remain in ordinary exact
interfaces. Dynamic subview origins must retain their original loop frame,
inherited pitch and proved containment. This uses the existing runtime-offset
`buffer.subview` contract, not a new operation or buffer model.

The domain certificate retains first/last rows and every reader interval's first
and last row as crossing witnesses. Fill-to-reader requirements factor through
the last overlapping fill by native completion order. Reader-to-fill requirements
factor through the first protected fill by native start order. Full reader
coverage supplies omitted write/write requirements. These are generators and
are reduced jointly with other domains; they are not automatically covers.
Point selectors retain their original formulas, including unsampled rows.
Witnesses are explicitly tagged and never advertised as a complete cell partition.

The initial global witness directory can cost the product of phase count and
reader endpoints. It removes the physical-row factor but does not implement the
draft's stronger adjacency-local frame bound. Point queries still inspect the
original access list; no logarithmic selector claim is made. Full reader/native
ports and all other-domain endpoints must remain available. Unknown siblings,
partial fills, skipped writes and incomplete reader coverage cannot inherit this
certificate.

M3 evidence currently exercises MTE2 row fills and MTE3 interval readers with
real GM output effects. A vector reader writing a different Vec buffer does not
qualify merely because the resident part does: all placed Vec bytes share the
canonical domain and its additional writes must satisfy the complete-domain
contract. Multiple reader-pipe collective fixtures and general nested collective
composition are not established by this milestone. The existing generic numerical
native-boundary tests remain authoritative for pipes used only in middle children.
Other domains retain their explicit interface costs.

For 8, 1,000 and 1,000,000 rows, the simple one-phase case retains two collective
witnesses, three total interface points, 336 interface entries and 15 leaf query
construction steps. The large shapes are host-only scaling derivatives, not
native-memory feasibility claims. Eight static phases at eight rows pass 20,736
all-event queries plus actual-cover, access and boundary checks. Nested static
subviews and reshape aliases of the moving row retain the original frame and
pitch. The initial C++ failures used the wrong PTOAS launcher; the pinned rerun
restores the established environment without changing semantics.

M3 accepted by all three reviewers; **404 tests passed**, with 18 existing warnings.

### M4: complete-phase repetition and direct physical export

A structured repeat may use the collective certificate only when its touched
source interval starts and ends on complete, already coverage-checked phases.
The check visits every repeated ancestor, including zero/one-count constructs.
The row-fill loop itself is an original leaf, not an exemption for a fill-only
repeated ancestor. Empty complete phases can disappear; skipped partial phases
cannot inherit an intervening-reader argument.

The existing source tree owns repetition. Row-fill leaves analyze their local
row count; finite reader leaves analyze one local visit. Numerical merges share
repeated summaries and retain original mixed-radix coordinates. The source-layout
invariance premise used by scoped endpoint/reuse certification holds for these
complete phases. The physical exporter, matching policy and allocator are
unchanged. One initial integration failure exposed the absent original-access
index on this newly structured route; index construction is now shared with the
invariant constructor and requires a fresh result. Enclosing-motion read-only GM
selectors keep their original path. Written moving domains without a reset
certificate retain their explicit refusal.

Bounded validation includes zero/one/multiple invocations, two enclosing frames,
counter resets, original GM and local selectors, actual covers, every event-kind
query and emitted matching/causal ID reuse. Complete-phase commands are compared
to independently recovered actual-access order, not only to retained generators.
Partial-phase repeated ancestors are negative cases. Large row/visit counts test
representation growth, not native capacity. Static command constancy is distinct
from logarithmic growth of repeated numerical summaries.

This is the existing opt-in production route. Fresh notifications at entry,
closed matching at exit, payload completion, native instruction qualification and
device validation remain separate contracts. No return-boundary rule, barrier
exemption, allocation rule or expert protocol changes here.

M4's broader gate passed **289 tests** with 20 existing warnings. The
[receipt](buffered-intervals-m4.json) records independent bounded emitted checks
and descriptive timings. At three visits, 5/1,000/1,000,000 rows retain 1,008
interface entries. At five rows, 3/1,000/1,000,000 visits retain
1,008/5,040/8,736 entries and 3/15/26 merge nodes. Emitted sites remain two SETs,
two WAITs, four guards and no barriers for these repeated cases; zero/one visits
specialize separately. This establishes row independence and logarithmic repeat
summary growth for the tested contract, not constant total representation size.

M4 accepted by the architecture, correctness and algorithm/performance reviewers.

### M5: qualification and conformance

The final campaign uses the committed M4 implementation, with the qualification
driver and structural bounds pinned separately. It checks the existing finite,
logical, rotating, numerical, nested and physical paths as well as Buffer view,
slot, serialization and lowering regressions. The new regression bounds distinguish
constant command schemas from logarithmic retained repeat summaries.

| Draft construction | Production component and evidence | Remaining scope |
| ------------------ | --------------------------------- | --------------- |
| Qualified physical interval descriptions | `placed_sync_import.cpp`; exact contiguous byte spans, physical aliases and original frame checks | Native instruction footprints remain a separate premise; noncontiguous/partial boxed views cannot borrow interval evidence |
| Rotating interval generators | `rotating_sync_analysis.cpp`; per-pipe range sweeps, common-stride phase normalization and existing quotient reducer | The supplied-interval bound excludes access recovery and unsupported-domain atomization |
| Collective row and resident boundaries | `CollectiveDomain` and original point selectors; independently checked actual covers, all four event queries and unsampled rows | Complete-domain fill/read coverage required; global witnesses can cost phases times reader endpoints; no adjacency-local interface bound claimed |
| Numerical composition and repeats | Existing source-layout and numerical merge services; complete-phase boundaries, mixed-radix resets and read-only GM selectors | General symbolic crossing selectors, partial-phase repeats and unqualified moving written domains remain unavailable |
| Direct physical realization | Existing endpoint/reuse services; independently checked command matching, acyclicity, order equality and causal reuse on bounded qualified fixtures | Six-ID certification is separate from minimum demands; unsupported DMA same-pipe mechanisms retain refusal; native entry/return is not proved |

The five original ring cases retain exactly the baseline reduced demands and
physical refusal. The sweep retains nine generators rather than five in the
small ring example: its per-pipe witnesses need not be the smallest generating
set. Joint reduction still returns the same four covers. The demonstrated gain
is avoiding interval/atom incidence expansion, not fewer generators on every input.

For complete collective phases, static control remains small as rows/visits grow:
the measured repeated cases add 9–10 PTO arithmetic operations and retain two SET
and two WAIT sites. The 5-row, 3-visit execution has five SETs, five WAITs, three
causal reuses and 43 added `IfStmt` guard tests (not loop tests or scalar
operations within predicates). Its projected order equals the
independently recovered access order. This is finite validation under the command
model, not a uniform native proof, and compact static code does not imply constant
executed control work. Large shapes/counts are host scaling derivatives only.

Configured Qwen output projection and all four gate/up bands still select PyPTO
in ordinary opt-in production and produce C++. All ten PyPTO/PTOAS-mode output
sets match the earlier retained Qwen production receipt byte-for-byte. Its revision
and dirty-source hashes are recorded in the final receipt: this is an artifact
regression comparison, not a fresh baseline timing campaign. These kernels retain
their existing nested route; they do not establish that the new collective provider
covers arbitrary attention or partial-write phases.

Every timing here is a descriptive single run. Import/reduction/composition detail
timers are included in total analysis time and must not be summed with it.
Physical preparation, allocation and emission share one measured call; they are
not separately attributed. PTO arithmetic counts precede downstream folding.
No device work, return-boundary change, allocation change or automatic fallback
policy change belongs to this campaign.

M5 accepted by all three reviewers. The final autosync gate passed **889 tests**
with 28 existing deprecation warnings; the separate Buffer/slot companion gate
passed **398 tests**. All **1,287 tests passed**, and ten configured Qwen mode
cells generated C++. [The final receipt](buffered-intervals-m5.json) contains
source/tool/input hashes, exact commands, scaling samples and the historical
artifact comparison. This completes all five scoped follow-up milestones; the
limitations in the table remain explicit.
