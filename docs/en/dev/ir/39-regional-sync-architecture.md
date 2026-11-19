# Regional synchronization architecture and implementation gates

Status: all five implementation gates accepted within the scope below. The
original broad host gate was 525 passes and one unchanged pre-analysis admission
failure. The bounded architecture follow-ups below now pass 651 selected tests
with the same one admission failure; their individual gates are recorded below. This is not an all-green
repository or native/device qualification.

## Objective and RFC alignment

Preserve placed Buffer IR and immutable completion tokens; connect the existing
exact regional algorithms to reusable, certified physical palettes. Follow
[RFC #3375](https://github.com/hw-native-sys/pypto/issues/3375), inspected on
2026-11-14 (open, updated 2026-11-10, no comments).

| RFC responsibility | Decision |
| ------------------ | -------- |
| InsertSync after stable execution/storage decisions | Use placed effects and existing exact regional services |
| OptimizeSync preserves logical semantics | Transform tokens; supported unoptimized tokens must also lower |
| LowerSync consumes logical IR | Interpret actual token uses, never regenerate minimum demands |
| Compilation-scoped resource ownership | Transactional claims, target eligibility and native reservations |
| Entry/exit obligations | Require explicit boundary evidence; closed notifications are insufficient |
| Opt-in ownership | One owner per kernel; complete transactional delegation on capability refusal |

AIR is a reference for structured asynchronous dependencies and separate
lowering. PTOAS Frontier supplies regional order and reuse-certificate designs.
Neither its dispatcher nor AIR's dependency extractor is copied wholesale.
PyPTO tokens denote source-pipe-prefix completion, not operation-only completion.

```text
Placed Buffer IR + operation/native contracts
  -> InsertSync: modeled accesses and regional minimum demands
  -> logical completion-token IR -> OptimizeSync
  -> LowerSync: actual-token interpretation
  -> retained unassigned protocol -> certified palettes
  -> existing allocation -> immutable AssignedSyncPlan
  -> existing command descriptors and emission
```

No new public Buffer or communication dialect, backend, tiling change, payload
rescheduling, larger pool, admission-limit increase or whole-nest expansion.

## Ownership and capabilities

The placed-input owner retains the unchanged function, complete operation
inventory, storage identities and aliases, footprint precision, original frames,
slot selection, readiness and explicit caller bounds. Exact footprints,
conservative enclosures and unavailable effects are distinct.

The mathematical owner retains its finite/rotating/numerical result. Demands,
queries, selectors and executable endpoints are independent capabilities.
Selectors distinguish exact answers, proved absence in a covered domain and
unavailable coverage. A failed export preserves mathematics and unrelated local
exports. Child boundaries do not complete payloads or reset resources.

Share existing numerical merge/query kernels between two separate owners:

| Client | Edge producer | Meaning |
| ------ | ------------- | ------- |
| Mathematical analysis | Native edges and storage requirements | Required modeled payload order |
| Actual-token interpretation | Native edges and actual token uses | Order expressed by this token program |

Shared phase summaries never replace original occurrence identities. All four
start/completion query combinations retain their meaning. Reuse proofs may use
the existing stronger completion-to-start payload witness; failure means
unproved, not absence of command reachability. More permissive command-cut
proofs require admitted native command semantics, not textual order.

Retain one immutable unassigned protocol skeleton with every consuming
handoff, matching identity, source/consumer cut, intra-cut order, exact activity
domain (including holes), coordinate map, direction, invocation scope and
entry/exit notification obligations. Multiple token uses remain separate
handoffs unless a separate protocol proof permits sharing. Normalize the
multiset of uses, not merely endpoint sets.

Rank construction, reuse verification and emission consume the same skeleton.
Emission must not independently invent a different tie order. AssignedSyncPlan
binds the exact protocol, owner, frames, caller domain, target and claim snapshot.

Caches belong to immutable owners. Token edits, changed frames/guards/caller
bounds, changed protocol order or resource claims invalidate dependent evidence.
Coordinate adapters transform or invalidate exports. Consumer-specialized
capture results are not cached by token identity alone.

## Certified palettes

Current scoped lowering generally reserves one ID for an entire family span.
The extension allows several logical records to reference one certified palette;
it preserves their commands, matching identities and domains.

The initial class is deterministic finite numerical-phase schedules with
qualified prefix, repeated phases, tail, original nested coordinates and resets.
It uses established activity/arithmetic bounds and supported capture/carry
semantics. Arbitrary dynamic histories and general token fanout realization are
separate extensions.

For each directed resource namespace and execution scope:

1. Derive the publication stream from the retained protocol schedule.
2. Assign publication rank n using prefix/phase counts, enclosing visits and
   deterministic intra-cut order. Reset-local IVs alone are insufficient.
3. Try B from one through the existing eligible directional capacity.
4. Propose lane n mod B, retaining the logical records separately.
5. Prove WAIT(n) precedes SET(n+B) for every active successor in that lane.

Both endpoints compute the same publication rank through the checked matching
map. Prove every prefix/phase/tail, carry and reset transition using qualified
finite classes, never sampled executions as a uniform argument. No successor
means no reuse obligation, not a fabricated WAIT. Missing history, overflow or
inconsistent frames cause refusal.

For example, startup, even and odd records may share a palette through their
original publication ranks without merging demand predicates. The certificate
retains lane count, record maps, internal reuse and invocation premises.
First/last boundary witnesses are internal proof work, not a separate retained
export without a production consumer. This is a sufficient
certificate, not a minimum-budget theorem.

The existing allocator chooses physical IDs. Its coloring algorithm and
chain/suffix premises remain unchanged. Flat and periodic paths become producers
of the same resource interface. Emit certified maps through existing descriptors;
do not add synchronization to fit a palette.

The compilation context owns eligibility, native/explicit claims and atomic
whole-kernel transactions. Lifetimes remain scoped to actual resource namespaces;
unrelated invocations do not conflict merely because they share a compilation.
Failure rolls back IR and tentative claims.

## Failure and invocation contracts

Distinguish unsupported effects, missing alias/range/readiness premises,
unavailable mathematics, unavailable exports, unsupported token realization,
unproved reuse, proved fixed-plan scarcity and unqualified invocation boundaries.
Malformed compiler output is an internal error and must not delegate.

Executable qualification requires an explicit checked target/runtime boundary
guarantee or an established boundary protocol. Otherwise it remains unsupported
or wholly delegated. Device success and notification closure do not prove
payload completion at return. The original milestone introduced no drain policy. Follow-up `12e99a75c`
now adds an explicit A2/A3 terminal ALL-pipe drain before public ownership;
unsupported exit forms refuse. Internal child lowering still permits pending
payloads. Earlier drain-asymmetric device receipts remain separately qualified. No scarce-ID repair
may change the logical completion fact.

## Ordered implementation and review gates

Each gate requires compiler/correctness and architecture acceptance before the
next stage. Preserve existing dirty changes. At each gate migrate the selected
production callers and delete replaced paths; do not retain duplicate pipelines
under a wrapper. Temporary independent test oracles may remain clearly labeled.

1. **Baseline/contracts:** pin HEAD, dirty tracked/untracked sources, tools,
   inputs and failures; stabilize frame/coverage/native ownership and separate
   order clients. Existing answers and outputs remain unchanged. No new demand
   analysis during lowering.
2. **Actual-token protocol:** consolidate handoffs, matching maps, original
   cuts and exact domains; preserve zero-trip/carry semantics. Independent
   expansion agrees with token uses; edited tokens invalidate old evidence.
3. **Palette/resource certificates:** compact ranks, bounded palette proposals,
   uniform proof and existing allocation adapter with transactional claims.
   Generic phase/reset fixtures qualify; negatives retain precise refusals.
4. **Emission/cleanup:** consume certified maps from the immutable assigned plan,
   preserve existing guard sharing and schedule checks, and remove replaced
   period-one assumptions for the admitted contract. Decoded and emitted IDs
   agree; ordered commands remain unchanged apart from ID renaming.
5. **Whole-kernel qualification:** pinned unswizzled Shenggan original with only
   manual sync removed; existing matmul, Qwen, rotating/reset regressions;
   independently varied K, visits, slots, stages and launch partitions. Report
   analysis, logical preparation, physical certification, C++ and invocation
   qualification separately. A precise remaining obligation is a valid result;
   a kernel-specific workaround is not.

## Validation and costs

Independent checks reconstruct Buffer effects, closure, actual retained covers,
selectors and all event queries; validate matching, acyclicity, prerequisites and
physical consumption-before-republication on emitted commands. Cover zero, one
and multiple visits, skipped activity, startup/tails, nonunit coordinates, resets,
fanout, claims/collisions, arithmetic overflow and unavailable histories.

Stale-evidence negatives change tokens, frames, guards, target claims and
intra-cut order. Compare complete prerequisite relations, not only counts.
Retain scalar native order, the qualified accumulator exemption, compact K/replay
and zero-inner-body regressions.

Measure providers, records, ports, phase classes, proof attempts/cache hits,
mapping/control size and stage costs. Do not enumerate trip counts; explicitly
charge logarithmic repetition work and finite phase/interface expansion. Avoid
new global pairwise scans; document existing algorithm bounds honestly.

Use repository build/test instructions with the machine-wide aggregate maximum
of two intensive workers. Keep artifacts in disk-backed build directories.
Deliver focused receipts, source/tool hashes, deletion ledger and RFC/paper
conformance. Host results establish modeled guarantees; native/device claims
remain separate.

## Initial baseline

HEAD: `48c40512e6c90f444204ad94ef6c6c027d34d970`.
Dirty-source inventory SHA256:
`f6ef37f956e34ad42fd072b5f8c746fe2e19408aff18622379f3082ae58f253e`.
The inventory, binary hash and patch are in
`build-py314/regional-sync-architecture-20261114/baseline-source.json` and
`baseline.patch`. Gate results will be recorded separately; a Git hash alone
does not identify this starting checkpoint.

### Gate 1 result

Architecture and compiler reviewers accepted the baseline/contracts gate.
Existing shared numerical merge/query services already separate mathematical
and token edge owners; no new wrapper was needed. The initial focused run was
218 passed / 3 failed (188.62 seconds). Two C++ failures came from the ambient
PTOAS launcher missing a shared library; both pass with the existing pinned
0.72 launcher. The remaining inherited `M128 N256 K512 workers1` derivative
fails Buffer admission before InsertSync. Its positive test remains unchanged.
This is an accurately classified non-green baseline, not an all-green campaign.
The [receipt](39-regional-sync-architecture.json) records commands and hashes.

### Gate 2 result

The immutable `ScopedSyncProtocol` owns the imported input snapshot, original
token function, raw handoffs and frozen cut order. `ScopedTokenOrder` borrows
that one input; only the trusted preparation factory can construct a normalized
view. Allocation validates the exact owner and prepared inventory. The assigned
plan retains the protocol and raw-to-assigned mapping, and emission traverses
raw cuts/domains rather than deriving order from fused family positions.

Review caught and resolved mutable-input rebinding and a public prepared-vector
constructor that could have admitted self-justifying evidence. Architecture and
independent correctness review accepted the fixes. The final focused run was
221 passed / one unchanged pre-InsertSync Buffer-admission failure (191.76 s).
No palette certificate or new physical acceptance is claimed by this gate.

### Gate 3 construction supplement

The accepted implementation uses a publication-only sequence/repetition grammar
from actual raw cut order and exact AP endpoint domains. Original-loop domain
endpoints partition activity intervals; proved finite residue cycles retain
repetition counts and tails. This is coordinate/activity construction, not
reuse of mathematical demand edges. Counts, phase partition work and retained
rank pieces are charged explicitly.

For a candidate B, each node retains its publication count and first/last at
most B publications. Internal proofs recurse. A sequence contributes at most B
distance-B pairs crossing its seam. For a repeated body with P publications,
rank r's successor has body rank `(r+B)%P` and visit shift `floor((r+B)/P)`;
the starting-visit domain is clipped to `0 <= q < count-shift`. If P>B, only
boundary publications are needed. If P<=B, all P are already retained. P=0
has no publications. Thus B>P and reset seams require no trip enumeration.

These pairs are query obligations, never edges installed in the proof graph.
The initial implementation uses one palette per directed execution namespace;
it does not claim general inter-palette coloring. Last-consumption boundaries
follow the proved consumption/publication chain, not merely publication rank.
Both endpoint lane expressions substitute the same matched source coordinates;
lower rank modulo B with reduced coefficients to avoid unnecessary wide INDEX
values. Proof counts still use checked wide arithmetic.

The resource-owner review confirms that the existing stack-scoped Allocator and
whole-function RealizeSync transaction already own the relevant catalog, native
claims, directional eligibility and atomic replacement for the admitted exclusive
leaf cases. Retain them instead of adding another context wrapper. The palette
producer receives a candidate B from that resource owner; it does not choose a
second pool. This is the scoped implementation of the ownership contract, not a
claim that the RFC's general cross-core/multi-pass manager is implemented.

### Gate 3 diagnostic checkpoint (production unchanged)

The general `ScopedEventTransition` query proves the sufficient payload bridge
`C(previous consumer) -> I(next producer)` over a rectangle of original
coordinates. It never installs an edge in the actual-token graph. Explicit
step/arity/frame mismatches are errors; a well-formed question without a proof
remains unproved. Independent checks include first-visit-only waits: a true
initial query does not become a uniform proof across resets.

The diagnostic publication grammar preserves raw-use multiplicity, original
cut order and exact arithmetic-progression domains. It retains each source-rank
expression and substitutes the matching source coordinates at WAIT. Indexed
subtree routes, per-region cuts and sequence prefix counts avoid repeated global
searches. Certificates retain raw-use identity alongside sufficient payload
boundary witnesses. They are not arbitrary command-cut reachability exports.

On the pinned original Shenggan logical input, worker cases 0, 16 and 23 all
qualify within the proposed six-ID budget. These are certificate diagnostics,
not physical allocation or generated-code results:

| Direction | First proved candidate B | Retained rank pieces | Grammar nodes |
| --------- | -----------------------: | -------------------: | ------------: |
| MTE2 → MTE1 | 4 | 12 | 36 |
| MTE1 → MTE2 | 4 | 28 | 152 |
| MTE1 → M | 2 | 24 | 120 |
| M → MTE1 | 2 | 62 | 308 |
| M → FIX | 1 | 1 | 18 |
| FIX → M | 1 | 1 | 9 |

The same sizes hold for the checked ten- and eleven-visit cases. Failed smaller
B attempts are unproved, not certified lower bounds. Original-token order
construction took about 3.5 ms per checked case in one descriptive host run;
budget trials are additional audit work. These are not speedup measurements.

Regression scaling independently varies inner count and outer visits from 100
to 10,000. For its fixed two-phase input, grammar nodes, phase visits and rank
pieces remain unchanged. Finite residue expansion is charged separately: an
LCM larger than a loop can still cause work proportional to that finite phase
prefix. No arbitrary-period constant-size theorem is claimed.

Reproduce the original-input diagnostic with the current worktree extension:

```bash
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 tests/ut/ir/transforms/qualify_scoped_sync_palettes.py \
  build-py314/gemm-remaining-gates/unswizzled-original/stage2-logical.msgpack \
  build-py314/regional-sync-architecture-20261114/gate3-original-palette.json \
  --blocks 0 16 23 --budget 6
```

The input hash remains
`9c8dd963554977d418e355bb69e476b6eac33488f7de8666e578d4685834b7e4`.
Invocation-boundary/native qualification remains separate. Production use must
pass the remaining resource-owner, assigned-map and emitted-program gates.

### Gate 3 acceptance

The final focused gate passed 236 tests with the one unchanged pre-analysis
Buffer-admission failure (193.86 seconds). Compiler and architecture reviews
accepted the immutable schedule, uniform proof and independent checks.
`BuildScopedPublicationSchedule` constructs once; `CertifyScopedPalette` reuses
that owner across budgets and rejects foreign protocols. No production palette
selection was enabled by this gate.

Gate 4 assigns distinct eligible IDs to each certified lane through the existing
allocator. The complete palette owns one direction. Its per-lane proof does not
establish the older coloring routine's cross-lane theorem, so that routine is
not used to merge palette lanes. Unsupported or unproved palettes may retain
the existing allocation capability for the entire direction. The assigned plan
must retain a tagged global-rank map; family-local modulo is not equivalent.

### Gate 4 acceptance and selection correction

The allocator first attempts the existing complete directional assignment. Only
an unproved assignment requests the retained publication schedule and its
uniform palette certificate. Exact scarcity, invalid inputs and unavailable
eligible resources do not become speculative palette requests. Failed attempts
publish no IDs or partial ownership. A certified palette receives distinct
eligible IDs through the existing allocator; the emitter consumes its tagged
rank map through the existing command descriptors.

This ordering is a measured design requirement. An initial palette-first
implementation increased Qwen output from 13 SET/WAIT pairs and 11 branches to
25 pairs and 35 branches. It was rejected at review. With incumbent-first
selection, all four checked Qwen output/gate-up worker cases have byte-identical
printed IR to the previous assignment, on identical logical inputs and pools.
These four comparisons fix the inferred parameter to its lower endpoint; they
are not an all-parameter output-equivalence claim.
No demand, token, protocol, payload, cut or target pool was changed.

The final focused gate passed 78 tests. The known small-GEMM Buffer-admission
failure was explicitly excluded from this narrow gate and is included in Gate 5.
Tests independently execute actual tokens and emitted consuming commands and
compare their complete payload prerequisite relations, matching, causal reuse
and pool compliance. Intentionally weakened token fixtures check that altered
protocol, not coverage of the original storage requirements.

The original Shenggan configuration now completes public `LowerSync` and C++
generation. This public target exposes eight eligible IDs and the output uses
IDs 0–7 across its static cases. Separately, all 24 admitted worker cases lower
with an explicitly restricted six-ID pool and generate C++; those outputs use
only IDs 0–3. The latter is a scoped six-ID certificate, not a claim that public
lowering selects that same assignment or that four IDs are minimal. Full-input
compiler success is separate from bounded independent semantic validation.

Receipts are `gate4-incumbent-recheck.json` and
`gate4-original-final/receipt.json` under the pinned campaign directory. The
transport driver records input and extension hashes and retains generated PTO
and C++. It runs public lowering before specializing diagnostic worker bounds.
The input remains byte-identical to the baseline. Native return completion is
still unqualified; no return barrier was removed or added in this stage.

PTOAS autosync remains enabled for C++ generation. Its success alone does not
prove the downstream compiler inserted no commands; the independent semantic
checks apply to the emitted PyPTO physical IR.

## Paper and RFC consequences

| Established construction | Qualification | Appropriate claim |
| ------------------------ | ------------- | ----------------- |
| Existing regional minimum-demand producers feed token IR | Exactness is relative to admitted modeled accesses; unavailable exports stay unavailable | This task preserves the regional analysis boundary; it does not introduce new access classes |
| LowerSync retains actual token handoffs and original cuts | Fanout uses remain distinct; modified tokens are interpreted rather than replaced | Logical synchronization is a real input to resource lowering |
| Publication-rank palettes bridge qualified phase/reset seams | Every rank-distance reuse must have a uniform actual-token/native proof | Supported repeated protocols can share physical IDs across logical families |
| Existing allocator assigns a complete directional palette | Distinct eligible lanes, original target claims, whole-function transaction | Additional fixed protocols become physically certifiable without changing demands or adding synchronization |
| Frozen cut emission and bounded independent checks | Exact modeled payload relation compared on tested executions | Mapping preserves the tested logical protocol; no device speedup or native return proof follows |
| Fixed-period compact schedule and coordinate formulas | Residue expansion and checked integer capacity remain explicit limits | Trip-count compactness is qualified by the supplied activity representation, not universal for arbitrary periods |

The remaining limitations are independently actionable: the inherited tiny GEMM
Buffer-admission case precedes demand analysis; collapsed startup/continuation
frames can lack executable exports; unproved resource questions are not capacity
lower bounds; general dynamic-history and cross-core resource ownership need
additional contracts. Native instruction footprints and invocation entry/return
remain separate from the host model. None is repaired by changing tile sizes or
copying the manual kernel's synchronization.

### Reproduce the physical qualification

Source `.claude/skills/testing/load-env.sh` first and cap aggregate build/test
workers at two. Use the current worktree's extension, not a copied installation.
For this Linux checkpoint the configured build uses Python 3.14.7; PTOAS 0.72
runs through the recorded Python 3.11 launcher. `env.sh` is not the environment
source for this campaign.

```bash
export PYTHONPATH="$PWD/python:$PWD/tests/ut/ir"
export PTOAS_ROOT="$PWD/build-py314/architecture-correction/ptoas-launcher"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 tests/ut/ir/transforms/qualify_scoped_sync_transport.py \
  build-py314/gemm-remaining-gates/unswizzled-original/stage2-logical.msgpack \
  build-py314/regional-sync-architecture-20261114/transport-rerun \
  --blocks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
python3 tests/ut/ir/transforms/qualify_scoped_sync_scaling.py \
  build-py314/regional-sync-architecture-20261114/scaling-rerun.json
```

The full focused regression command, pinned initial dirty-source inventory,
intermediate review gates and final source/tool hashes are recorded in the
companion JSON. Raw build/pytest logs and generated files stay in the disk-backed
campaign directory. The source tree deliberately retains unrelated pre-existing
changes; HEAD alone is not this experiment's identity.

### Gate 5 acceptance and measured limits

The final broad host gate passed **525 tests** with the one unchanged
`test_shenggan_original_tiles_single_invocation_cpp` failure (406.67 seconds).
That M128/N256/K512, one-worker derivative remains Functional IR and is refused
before demand analysis. Its positive expectation was not weakened. Six warnings
are existing inline-helper direction-annotation deprecations. The separate new
emission-scaling regression passed; architecture review accepted this evidence.

The broad gate includes independent modeled access, closure, actual cover,
selector, all-event-query, token matching and physical-reuse checks; Qwen
activation boundaries, rotating slots, counter resets, zero trips and native
scalar/accumulator rules are retained. It is a focused 13-file campaign, not the
entire repository test suite. Full original-kernel C++ success and bounded
semantic checks remain different evidence.

The synthetic scaling campaign contains eleven ordinary-lowering runs followed
by separately timed explicit resource audits. All ordinary and six-ID runs
succeed. With fixed two-phase selection and four inner slices, varying K from
32 to 32,000 or outer visits from 4 to 4,000 leaves these quantities unchanged:

| Quantity | Retained or emitted value |
| -------- | ------------------------: |
| Prepared families | 36 |
| Palette rank pieces | 62 |
| Palette grammar nodes / phase visits | 308 / 81 |
| Domain inspections | 1,432 |
| Emitted statement occurrences | 1,877 |
| Static branches | 778 |
| Static SET / WAIT sites | 163 / 117 |

SET/WAIT static counts need not match: guards and repetition determine executed
matching. The independent bounded checks test executed matching and reuse.
Single ordinary-lowering measurements are 26.9–28.7 ms; these are descriptive,
not paired speedup measurements. Token analysis, logical construction and
explicit budget audits are recorded separately and are not added to overlapping
inclusive timers.

Inner-slice extents 2/4/8 produce 561/1,877/3,852 statement occurrences and
212/778/1,616 branches, with approximately 10.3/26.6/56.2 ms lowering. This is a
counted-loop-extent experiment, not a new static-source-stage scaling claim.
Existing numerical-interface static-scaling tests are included separately in
the broad gate. The two-slice ordinary path retains its existing assignment;
the new fixed-period compactness regression requires an actual palette and a
legacy six-ID refusal, so it cannot pass by testing only the incumbent.

This closes the planned certified-transport milestone, not code-size
optimization. The large absolute control inventory is a measured follow-up for
the existing descriptor/schedule lowering. Successful Qwen incumbents remain
byte-identical; the new palette is selected only where the old sufficient
assignment is unproved. No kernel-name special case, tile-size change, demand
change, additional synchronization or larger pool was used.

### Pre-commit architecture cleanup

The production-only call-site audit removed the diagnostic one-shot palette
wrapper and speculative first-publication/last-consumption exports. The actual
uniform proof still checks every required boundary transition; allocation and
emission consume the certified schedule and budget, as before. Tests inspect
those production services and independently check ranks and reuse.

The old test-only mathematical-demand-to-physical adapter is replaced by logical
insertion followed by actual-token lowering. Its private mathematical reuse and
adjacent-barrier certificates, caches and counters are removed where no longer
consumed. Test consumers are not a reason to retain a second implementation.
The capability-disabled incumbent comparison remains a test-owned invocation of
the same current allocator, not a separate algorithm.

Cleanup validation is an explicitly combined campaign: 113 independent checks;
389 initial consumer passes followed by all five corrected expectation checks;
132 scoped/nested tail passes; seven assignment checks; the remaining full Qwen
gate/up case passed in 84.36 seconds. An interrupted earlier run recorded 95
completed cases and is not described as a fresh all-green run. The final token
gate passed 162 tests with the unchanged pre-analysis admission failure.
The original public C++ output and all 24 restricted-six worker C++ outputs are
byte-identical to Gate 4 after the cleanup. All tests still use the original
access and causal command oracles. Native/device scope is unchanged.

The final mechanical-hook rebuild also passes 63 focused core tests and 365
additional type-narrowing checks. The five-file annotation gate records 234
passes plus two corrected fixture checks: increasing the prefix stride from two
to four preserves its admitted zero/one executions while reaching the intended
nonuniform-guard refusal. All 25 original GEMM C++ outputs remain byte-identical.

## Bounded architecture follow-ups (2026-11-14)

Baseline: `12e99a75c`. These three gates strengthen existing ownership and
interfaces; they do not add coverage, change demand algorithms, optimize token
programs, alter allocation, or change payload schedules. The earlier five gates
and their receipts remain historical evidence. Each new gate must receive
independent architecture/correctness acceptance before the next is implemented.

### Gate A — immutable published analysis

At the baseline, `CompactSyncResult` exposed mutable input, reduced segments and crossings
beside a lazy scoped-demand cache. Replace caller discipline with a mutable,
internal construction state and a read-only published owner. Move retained data
at publication; do not copy relation/segment inventories to freeze them. Separate
query-work counters from proof-bearing state. Internal construction may query
its current result, but must not publish cached exports before finalization.

For example, changing a construction's original frame or crossing inventory
must never leave a previously exported demand cache attached to that changed
state. Published owners may share immutable evidence; consumers cannot edit it.

Migrate invariant, rotating and numerical builders in
`compact_sync_analysis.cpp`, `rotating_sync_analysis.cpp` and
`numerical_sync_{hierarchy,phase,merge}.cpp`; update existing consumers/bindings.
Delete public mutable construction access. Preserve selectors' unavailable vs
covered-empty answers, and retain independent mathematical and actual-token
owners. Test publication/copy lifetime, repeated exports, changed-input isolation
and existing covers/queries/selectors. Inspect construction and retained sizes.

### Gate B — selected native implementation contracts

At the baseline, `physical_sync_allocation.cpp::NativeClaims` reconstructed native macro
selection. Move the admitted selection/claim contract to the existing backend
recipe/catalog boundary; have import and allocation consume it. Native ordering
and invocation completion must use that same qualified target/variant scope.
Retain operation-specific footprint recovery in the importer: selecting TLOAD
alone does not prove a rectangle or its alias policy.

For example, a Mat-to-Left TEXTRACT must not inherit the excluded Vec overload's
resource claims; an unsupported variant must refuse rather than appear
resource-free. A2/A3 scalar native order, the qualified Cube accumulator rule,
and public terminal completion remain explicit premises. Do not create a second
operation registry or claim executable support for A5/compound operations.

Migrate the existing importer and native resource consumer; use shared recipes
already used by code generation. Keep broad emission-name selection separate
from narrow synchronization qualification. An unknown hardware target may still
receive conservative mathematical analysis without the Cube exemption, but
physical realization must refuse. Resolve both initialization and continuation
variants and retain per-core/mode/lifetime resource qualification. Delete
allocator-local macro selection.
Acceptance includes variants, missing claims, reserved IDs, scalar and Cube
exemptions, target refusal and terminal drains. Generated plans and resource
assignments must remain unchanged for existing admitted fixtures.

### Gate C — shared consumer contracts

The identified duplicate is family ordinal decoding in
`physical_sync_allocation.cpp::Endpoint` and
`scoped_token_order.cpp::FamilyOccurrence`. Consolidate checked translation in
the existing physical endpoint service. Preserve two explicit conventions:
empty axes mean contiguous flat records for the flat allocator, but a singleton
for scoped-token families. Validate record bounds, axis extents, arithmetic and
original region/site identities without changing the admitted language. Share
mechanics, not the meanings of required order and expressed token order.
Do not force finite and periodic kernels into one mathematical representation.

For example, a missing original-coordinate binding must remain unavailable in
both query and endpoint consumers; neither may synthesize coordinate zero.
Covered-but-untouched storage remains an exact empty answer. Choose actual
production duplicates and list removed adapters in the gate receipt; do not
introduce a general interface without migrated production clients.

Acceptance requires actual-token order preservation, independent matching/reuse,
all event-query combinations, selector absence/refusal negatives, and unchanged
compact representation growth. Existing small-GEMM Buffer admission is a
separate known failure. OptimizeSync rewrites and new target coverage are deferred.

### Execution and evidence

For each gate: implement, serial build, focused independent regressions, inspect
removals and complexity, reviewer accepts or requests correction. Record exact
source/tool identities and results in the final receipt. The aggregate machine
limit is two intensive workers; this campaign uses serial builds/tests. No
native/device qualification or new performance claim follows from these refactors.

### Accepted implementation checkpoints

Gate A is accepted. `CompactSyncBuilder` owns mutable construction state;
`CompactSyncView` shares read-only query code and has no scoped-export cache.
`CompactSyncResult` publishes shared immutable state and owns the successful or
unavailable export memo. Base assignment is inaccessible, preventing replacement
of premises beneath a published cache. Diagnostics remain mutable counters,
never proof premises. Publication moves/shares inventories without copying them;
it adds constant-sized ownership allocations and makes no speedup claim.
The gate passed 354 Python tests and the existing C++ numerical-order target
(including publication/copy lifetime and 2,000 independent graph cases). One
previously known small-GEMM Buffer-admission failure was explicitly deselected.

Gate B is accepted. `backend/common/buffer_sync_contract` owns native operation
identity, current single-pipe variant selection, bound core/target resource
claims, Cube size eligibility and the supported scalar/terminal contract.
`PTOCodegen` shares broad operation-name selection without adopting autosync's
narrow admission policy. The importer retains physical geometry, precision,
readiness and same-accumulator/slot/interference proofs. Allocation no longer
selects macros or filters overload-specific claims itself. Both initialization
and continuation arms are qualified. Unsupported target mathematics remain
available without native protection; physical ownership remains refused.

The native gate passed 219 focused tests and 458 Buffer-emission/Vec tests.
Two old positive-step diagnostic regexes were corrected to match the baseline's
existing range-proof behavior; their nine negative cases still require refusal.
Six pinned emitted PTO plans (matmul, accumulating matmul, split-K, Vec, repeated
Vec and staged pipeline) remain byte-identical to `12e99a75c`.

Gate C is accepted. `DecodeSyncFamilyEndpoint` in the existing physical-allocation
interface is the sole shared ordinal arithmetic implementation for direct
allocation and scoped-token order checking. Callers explicitly select flat
contiguous or scoped singleton empty-axis conventions. It validates positive
axis extents, ordinal bounds, complete Cartesian volume, nonnegative strides
and checked visit arithmetic; unavailable maps do not become occurrence zero.
Original frame/guard/cut proofs remain with their existing owners. The old
`Endpoint` and `FamilyOccurrence` arithmetic bodies are deleted; thin wrappers
choose conventions and preserve contextual refusal messages. Work remains
O(axis depth), without unfolding trip counts.

The final serial integration gate is **651 passed, one unchanged failure**:
`test_scoped_token_order::test_shenggan_original_tiles_single_invocation_cpp`
still reaches Functional IR instead of placed Buffer IR before analysis. Two
existing deprecation warnings remain. The existing C++ target passes publication
lifetime, 150 independently enumerated endpoint tuples plus invalid/overflow
maps, and 2,000 numerical graph cases. All six pinned emitted plans remain
byte-identical to the baseline after Gate C. The established static-site and
trip-count scaling bounds pass unchanged.

Configured checks passed on the tracked tree plus precisely these changes.
The pinned hook dependencies use Python 3.11 (their torch version is unavailable
for the compiler's Python 3.14); unrelated untracked research files were excluded
from the check snapshot, not modified or deleted. A visitor-parameter naming
correction required by the type checker passed its two-test recheck.

See [the machine-readable receipt](39-regional-sync-architecture-followups.json)
for source/tool hashes, exact commands, per-gate results and artifact hashes.
This work does not add thread-safe concurrent diagnostic/cache mutation, new
kernel admission, OptimizeSync rewrites or native/device qualification. Required
order, actual-token order, consuming notification reuse and invocation completion
remain distinct contracts. Allocation algorithms, pool sizes, hardware exemptions,
logical protocols and payload schedules are unchanged.

## Minimum-demand audit and deferred control cleanup (2026-11-15)

The audit at `b39fc091453856ca6210c93a81394e21caa8d307` distinguishes three
properties: retained demands equal the nonnative covers of the qualified access
model; generated tokens express that required payload order; physical commands
preserve the actual token order. Compilation success alone establishes none of
these equalities. In particular, `conservative_prefix` deliberately selects a
stronger order, and downstream PTOAS delegation is not a PyPTO minimum-demand
result. The original whole-model capture and the separately qualified Qwen
projection captures have different caller evidence and ownership outcomes.

For exact sequential composition, `numerical_sync_merge.cpp` constructs storage
crossings from last writers/readers and first writers/readers, includes native
pipe crossings, and reduces the crossing set using child reachability. It does
not simply concatenate child handoffs. With forward sequential execution, a path
leaving a child cannot return to make one of that child's internal covers
redundant. The hierarchy applies this merge recursively to qualified repetitions,
preserving original coordinates and empty visits. This matches the draft's
numerical sequence/repetition construction. It is not a claim of exact treatment
of arbitrary interleaving or data-dependent branches.

With disjoint GM supplied, the `conservative_prefix` policy can retry an
unavailable whole-invocation exact analysis by projecting qualified read-only
input roots. The importer checks distinct original GM bases and direct effects
of the selected DMA operations. The exact model still includes writable GM
geometry and all local effects. The compiler preserves original DMA addresses
and runtime descriptor extents. The existing invocation partitioner and
directional allocator prove the combined order before the compiler publishes it.

For Qwen gate/up, the dynamic layer offset and weight extents belong to
read-only DMA operands. The common numerical owner exposes the last extract
and first reload of each reused MAT slot, and the final gate store and first up
multiply share the accumulator boundary. Independent children can still need
entry acknowledgments. Replacing those acknowledgments requires a shared
storage key, aligned activation predicates, and a proof that each lane's
previous WAIT causally precedes its next SET. A direction at the child boundary
does not identify those events.

For overlapping placed matrix allocations on A2/A3, exact import checks each
access in its original layout before mapping it into a shared byte domain.
A fully tiled descriptor maps a rectangular access to contiguous runs along
its inner box axis. A moving access retains its original affine frame and
advances the run coordinates by the corresponding physical byte stride.
This handles partial overlap and different pitches across same-space Mat,
Left, Right and Acc allocations. Unsupported layouts, slots, alignment and
effects retain the conservative route. A static effect can describe up to
128 discontiguous runs. Accumulator hazard protection still requires the
same qualified window and layout throughout the continuation. Overlapping Acc
components require whole-window accesses and pairwise-intersecting windows.

`sync_matmul_oracle.validate` independently interprets original Buffer addresses,
constructs concrete conflicts, computes closure and covers, and compares the
actual retained demand records after bounded expansion. It also checks all four
event-kind queries and storage selectors at sampled boundary/interior points.
Regional tests cover unequal siblings, alternative merge parenthesizations,
resident storage, nested rotations, counter resets, empty visits and qualified
initialization/accumulation branches. Separate logical/physical interpreters
check payload-order equality, matching and causal flag reuse where claimed.
These are bounded checks under declared layout, alias and native-operation
contracts; they do not independently qualify native footprints or prove every
parameter instance by enumeration. Guarded invocation partitions are qualified
cases, not a general minimum-demand construction for arbitrary runtime branches.

Two avoidable emitted-control patterns are recorded for later work, without
changing the compiler or the selected synchronization plan:

1. Qwen output: a final `SET(M,FIX,0)` guarded by `stage == 18` at the end of
   `for (stage=0; stage<20; stage+=2)` can move just beyond that nonempty loop,
   inside the same enclosing visit. No payload or command is crossed. The
   general rule needs nonemptiness, a terminal legal cut and compatible exit
   control; it cannot introduce a publication for a zero-trip loop.
2. Qwen gate/up: adjacent alternatives at one cut both test `k == 192`.
   Factor that test once; for `stage` in `{0,2,...,18}`, use `else` instead of a
   separate `stage == 18` test after `stage < 18`. Preserve the worker guard and
   the two different IDs. Five conditional statements become three without
   payload duplication. Commands separated by payloads or another publication
   must not be regrouped by this argument.

Generated-C++ rewrite experiments preserved ordered payload-macro/flag traces
for 40 output and 48 gate/up invocations. These checks are host-only, and do not
establish device speedups. The terminal invocation drain remains a separate
completion obligation; it adds no internal completion-to-start payload pair.
The fresh serial pytest gates passed 602 and 268 tests (870 total, no failures or
skips; 24 existing inline-helper deprecation warnings). The C++ numerical-order
gate also passed its 2,000 supplied-order graph seeds; that gate checks queries,
not storage extraction or minimality by itself. Nested compactness regressions
include million-trip counts without payload-visit expansion. No minimum-demand,
order-preservation or reuse counterexample was found in these checked cases.
See `39-regional-sync-minimum-audit.json` for the pinned validation receipt.
