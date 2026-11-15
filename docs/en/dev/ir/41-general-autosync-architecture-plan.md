# General autosync integration: architecture and Qwen acceptance plan

Current priority (2026-11-16): external review identifies four blocking defects.
Complete repair gates R1–R4 below before resuming the paused coverage packages.
Earlier accepted gates and test receipts are historical evidence, not a current
merge-readiness claim.

Status: the continued operation, ownership and retained-child gates are reviewed.
The original six milestones and whole-model ownership remain **partially complete**:
missing native and value-domain premises are not inferred from a passing capture.
The latest unchanged two-layer Qwen capture produces 32/32 C++ leaves under both
policies: exact owns 3, conservative prefix owns 16 and delegates 16. Another
47 existing task-control owners are now observed separately (see the M1 gate). The runtime package passes
256 tests; the broader 52-module campaign recorded 2,669 passes and six stale
fixture expectations, resolved by a reviewed 196-test module rerun. This is
aggregate evidence, not a fresh single all-green 2,675-test campaign. See the
continued ledger and [retained-child checkpoint](41-general-autosync-readonly-child.md).
Baseline: `b5597650a75b598645b1b1d3fe05bd3aab220f09`. This extends the
[regional architecture](39-regional-sync-architecture.md). Qwen end-to-end remains
the acceptance target, not a source of kernel-specific admission rules. Autosync
remains default-off; the producer policy defaults to `exact`. Choosing
`conservative_prefix` explicitly does not enable autosync or establish ownership.

## Revised execution order after external review (2026-11-16)

The original goal and M1–M6 remain in force: implement reusable synchronization
services and qualify the unchanged Qwen model, preserving exact analysis where
its premises hold. The repairs below are prerequisites, not a replacement goal
or evidence that the remaining milestones are complete. Cross-core and
inter-kernel synchronization construction remain excluded. Preserve their
existing manual/runtime/compiler owners and qualify only local boundaries.

The external review covers `bef68ac84` through `ef3572fd2`. Inspection of the
current sources supports its four reported defect mechanisms; this plan update
does not claim a fresh reproduction or adopt its aggregated test counts as our
own green run. HEAD is `3521d7f64`; the subsequent active-domain, scalar-payload
and effective-coordinate edits are **paused, uncommitted and unqualified**.
Keep those edits intact and separately owned while repairing the reviewed base.
Do not run the interrupted milestone build or resume coverage work implicitly.

| Order | Repair gate | Required result before advancing |
| ----- | ----------- | -------------------------------- |
| R1 | Closed-prefix recognition and complete token elimination | DAG-safe preflight; valid completed-token aliases either lower completely or receive a transactional capability refusal; no residual logical operations or token-typed IR in any successful physical result |
| R2 | Shared native placement legality | Exact, conservative and actual-token ownership paths agree on operand/workspace legality, including supplied placed Buffer IR |
| R3 | Static row descriptors on promoted allocations | Preserve proved static extents at each row-operation use, or refuse during selection before Buffer mutation; all three planners covered |
| R4 | Fresh regression and scaling gate | Correct stale expectations without weakening semantic negatives; record one fresh green focused run; separately attribute nested physical-realization growth |
| Then | Resume original M2/M3 and M5 work | Rebase the paused edits on accepted repairs; operation/readiness and symbolic-coordinate constructions retain their original proof obligations |
| Finally | Complete M4 and M6 acceptance | Recheck retained children, local manual-boundary refusals, whole-model local ownership and matching/reuse under the same completion policy |

Each repair follows **implementation → independent review → accepted commit**.
Dependent work waits for acceptance. A failed gate is repaired and reviewed
again. Keep default dispatch, allocation algorithms, six-ID eligibility, scalar
native order, qualified accumulator exemptions, expert protocols, compact K/
replay and zero-inner-body behavior unchanged. No larger limits or watchdogs.

### R1 — Closed-prefix lowering must preserve valid IR and bounded discovery

Owner: `closed_prefix_sync.cpp` discovery/emission and the existing public
lowering/ownership boundary in `token_sync_lowering.cpp` and `token_sync.cpp`.

Before: a depth-D shared expression `t = join(t, t)` is recursively revisited
through both references; an unrelated input can pay this cost before the prefix
recognizer declines it. An unused `alias = completed_token` can survive after
its definition is erased, leaving a free token variable in an owned function.

After: memoize context-independent expression inspection by original identity,
or decline unsupported syntax before traversing it. Keep statement occurrences,
ordered cuts, frames and uses separate: a shared expression is not permission to
deduplicate two executed commands. Resolve supported pure-token aliases as part
of lowering, including unused completed facts. Unsupported alias/control forms
must decline before mutation; no optimizer prepass is a hidden requirement.

Require a shared final postcondition on every successful physical lowering and
ownership publication: no logical operators, token-valued expressions/bindings,
token carries or token-bearing signatures remain. These are logical-sync
operations and `SyncTokenType` values; ordinary runtime TaskIds retain their
separate meaning and owner. Reuse the verifier/type walk;
memoize shared expressions/types without skipping occurrence checks. Unsupported
input remains a capability refusal; invalid output from an accepted lowering is
a compiler failure and cannot receive an ownership marker.

Acceptance: restore the existing 28-join public regression; vary distinct DAG
nodes independently of uses and count distinct traversal work, with timing only
descriptive. Cover command-only and payload-bearing services, completed alias
chains, unused tokens, branch/carry/capture boundaries, OptimizeSync on/off and
serialization. Check output SSA, token elimination, payload order, matching,
acyclicity, consuming reuse and PTOAS generation on accepted cases. Preserve
actual-token semantics, including deliberately stronger waits.

### R2 — Native workspace legality is independent of synchronization policy

Owner: the existing shared selected-native contract and placement/view evidence;
clients include `placed_sync_import.cpp`, `conservative_sync_dataflow.cpp` and
closed-prefix/native qualification. Factor the existing exact check rather than
maintaining a second policy-specific interpretation or a new registry.

Example: a row broadcast with an explicit operand at 184 KiB overlaps its
selected native staging workspace `[184 KiB,192 KiB)`. Adding barriers between
calls cannot make the instruction's internal overwrite-before-read safe.

Retain static instruction selection separately from placement qualification.
After local allocation/view facts are available, prove separation for every
explicit data operand against the selected variant's implicit workspace. Reuse
indexed local alias, stride/slot and enclosing-coordinate evidence; account for
all admitted positions with checked byte arithmetic. A hull intersecting scratch is
unproved unless stronger existing evidence establishes separation. An existing
allocator reservation may supply the proof only within its enforced scope.
This check must not require exact GM selectors or declare an uncertain GM
location empty. Implicit effects remain in cross-call hazard analysis as well.

Acceptance: extend `independent_broadcasts(..., operand_alias=True)` through exact
and conservative analysis, public InsertSync/LowerSync and enabled RealizeSync.
Include aliases/views, workspace edges, read and write operands, legal disjoint
controls, reserved-placement evidence and unavailable local bounds. Refusals
preserve original IR and publish no commands/claims/ownership. Direct actual-token
lowering cannot bypass native legality by skipping storage-demand extraction.

### R3 — Preserve static row-operation descriptors at their original uses

Owner: `buffer_valid_analysis.h`, shared row recipes and
`lower_tile_to_buffer_pass.cpp`, reusing the static-cast descriptor bridge.

Example: `source` has physical `[8,32]` and static valid `[8,16]`; another alias
sets valid columns to runtime `width`. A later row sum on `source` still requests
its original `[8,16]`, even though the shared allocation handle became dynamic.
Restore the logical alias's metadata and supply an independently owned static
descriptor window at that use. Preserve physical address/pitch, reduction axis,
scratch effects, output extent and ordering. Metadata restoration does not
restore bytes overwritten through another alias. Do not widen to 32 columns,
assume dynamic `width` equals 16 or remove runtime metadata checks.
Restore every affected operand/output/scratch metadata object as required by
its use, preserve zero-extent guards and qualify descriptor ownership under the
selected planner, including alias/borrowed-view restrictions.

If the bridge cannot establish the selected native contract, reject during
SelectBufferIRFunctions and preserve the original Functional body. Selection
success must not defer a known descriptor incompatibility into lowering failure.

Acceptance: partial/full static columns with a dynamic alias, alias use before
and after the row operation, reductions and broadcasts, changing rows/columns,
scratch reuse and invalid/unproved metadata. Run PYPTO, DSA_RP and PTOAS planners,
plus Buffer-disabled controls. Check native descriptors, physical access
agreement, original values and guards, and conservative/exact policies where
applicable. Keep genuinely dynamic full-column requirements intact.

### R4 — Restore a fresh gate and diagnose nested physical cost separately

Update the reported stale tests at their actual changed boundary: diagnostic
text, newly admitted packed columns and descriptor construction versus use-site
qualification. Preserve function-level negatives for unrelated runtime metadata;
construction success is not execution evidence. Do not xfail, skip or loosen a
semantic assertion merely to produce a passing count.

Run the affected modules and established focused gate serially, with pinned
dirty-source/tool hashes and retained failure logs. Report fresh runs separately
from the external review's aggregated 2,354 passes and 12 stale failures. Repeat
real PTOAS ownership/final-drain and debug-rebuild checks; no device claim.

Independently isolate nested physical realization using the same logical tokens
and assignment policy. Vary static nesting depth with payload sites/trip counts
controlled; record stage times, retained size, distinct expression/statement
visits, resolver steps, query/cache work and emitted command count. Avoid the
exhaustive execution oracle in the cost probe. The review did not establish this
slowdown as a regression, so do not attribute it to R1 without evidence. If
superlinear repeated work is confirmed, repair its owning service under a
separate reviewed change, preserving the protocol and allocation algorithm.
Record any watchdog refusal; increasing it cannot satisfy the scaling gate.

Passing R1–R4 makes the reviewed base suitable for continuing M2–M6; it does not
extend regional capabilities beyond their qualified interfaces, qualify scalar GM
visibility, prove missing caller relations or make delegated local leaves owned.
Any additional common-completion construction retains M3's separate design
subgate only if needed; these repairs do not add it as a mandatory new algorithm.
Resume and test the paused active-domain, blocking-scalar and effective-coordinate
packages against the repaired base before their individual reviewer gates.

## Evidence and scope

The current branch already has immutable published analyses, shared selected
native contracts, checked endpoint decoding, compact flat-loop frontiers,
indexed storage partitioning and explicit terminal completion for supported
owned invocations. These are completed work, not new milestones.

The historical whole-Qwen capture used unchanged Qwen3-14B prefill, batch one,
sequence 128, original hidden/intermediate dimensions 5120/17408, and one/two
layers. Its library pin was `b72c66d9df69ba5f11982b212530a255c050988d`.
The recorded two-layer census was 2/32 PyPTO-owned leaves, 24 Buffer-admission
refusals, five caller-coordinate refusals and one activation-partition refusal.
Those counts belong to the old captured source snapshot, not this HEAD.
The existing `qualify_qwen_whole_model.py` harness was refreshed at the first
implementation gate. Its newer policy and strict named-ownership capture is recorded separately
below; this is not a new survey.

Source inspection of those captures reveals scalar planner code, data-loaded
addresses, scratch/row operations, atomic output and paired AIC/AIV transport.
Thus neither exact GEMM support nor local ALL barriers alone can establish
whole-model synchronization. First refusals do not establish later feasibility.
Native footprints, runtime visibility and device behavior remain independently
qualified. General prefill/decode, all shapes and all targets are not implied
by this initial model/configuration scope.

## Ownership: use the compiler's existing information

Paths below are relative to the checkout. The audit found these existing owners:

| Responsibility | Existing service | Autosync integration |
| -------------- | ---------------- | -------------------- |
| Structured control | `include/pypto/ir/stmt.h`, existing visitors | Traverse `IfStmt`, `ForStmt`, `WhileStmt`, yields and returns; derive graph edges only for algorithms requiring them |
| Effects and result ownership | `include/pypto/ir/op_registry.h`: `BufferArgEffect`, `OpArgEffectSpec`, `ExecutionMemoryAccessEvidence`, `BufferResultBehavior` | Consume data/metadata effects and allocation/alias/borrow evidence; do not create a second effect registry |
| Placed storage and views | `buffer_root_collector.cpp`, `buffer_view_analysis.h`, `buffer_valid_analysis.h`, `buffer_indexed_analysis.h` | Preserve physical overlap, valid extents, slots and provenance; different SSA roots do not imply disjointness |
| Operation implementations | `backend/common/buffer_{row,indexed,conversion,elementwise}_recipes.*`, `buffer_sync_contract.*` | Bind the emitted implementation to its effects, scratch, pipes, resource claims and completion behavior |
| Arithmetic | `include/pypto/ir/arith/analyzer.h` | Reuse bounds, symbolic sets, modular facts and scoped constraints; no replacement general solver |
| Caller/shape correspondence | `call_shape_binding.*`, `CollectSyncInvocationEvidence` in `placed_sync_import.cpp` | Retain proved relations and their caller scope, including Call/Submit distinctions |
| Communication | `expand_mixed_kernel_pass.cpp`, `cross_core_pipe.*`, existing tpush/tpop | Preserve transport ownership; qualify local publication/acquisition/release boundaries |
| Logical and physical sync | `token_sync.cpp`, `scoped_token_families.cpp`, assigned plans and existing emission | Keep actual-token interpretation separate from required-order analysis |

The existing `StmtDependencyGraph` is an SSA def-use graph with an InOut-use
premise and compound nodes. It is useful for its stated purpose; it does not
replace physical conflict analysis after storage placement. No new public CFG,
Buffer dialect or model-specific synchronization IR is proposed.

## End-state and semantic boundaries

```text
Existing structured IR + placed storage + selected backend implementation
  -> qualified effects / control / caller context
  -> exact regional providers OR explicitly selected conservative construction
  -> logical token program expressing the selected order
  -> OptimizeSync (optional, correctness does not depend on it)
  -> LowerSync: actual token and protocol order
  -> existing resource interfaces, assignment and command emission
  -> qualified invocation exit and domain ownership
```

Input recovery must no longer require successful exact occurrence analysis just
to expose sound effects. Extend the existing input owners with independent
capabilities, constructed on demand and tied to immutable input/target/context:

| Capability | Required distinctions |
| ---------- | --------------------- |
| Effects | Exact footprint; sound enclosing footprint; unavailable operation semantics; explicit no-access |
| Storage | Canonical physical space/root/view; possible overlap; supplied or proved disjointness and its scope |
| Control/readiness | Original structured statement and legal cut; executable guard; value availability and visibility obligations |
| Parameters | SSA/shape identities, assumptions and proof provenance; call/submit mappings; checked machine arithmetic |
| Analysis | Required modeled order, demands and independent query/selector/endpoint availability |
| Child | Sound effects per pipe; original visits; closed matching; possibly pending payloads; protocol/resource ownership |
| Realization | Actual logical order; consuming-command protocol; physical assignment/reuse; invocation obligations |

Keep `SyncStorage`'s numerical rectangle schema and `SyncSite`'s one-pipe schema
as exact-provider views. They are not universal operation contracts. Shared
recovery should retain original IR/storage references and optional certified
enclosures; a compound communication operation needs its protocol adapter rather
than an invented single-pipe payload. Factor reusable recovery out of
`ImportPlacedSync` before numerical qualification, migrating its existing client.

An operation's `Write` effect is not a full-overwrite certificate. A possible
write never kills older obligations without a coverage proof. A descriptor
metadata read is not automatically a payload-data read or a ready scalar value.
Registry defaults are not evidence for missing execution semantics. Selected
compound operations require their real pipe phases and hidden resources.

An unresolved address may use its containing allocation, union of candidates,
memory space or all classes, as justified. Enclosures cannot repair omitted
effects, unready control or invalid program memory accesses. A selector must
distinguish exact absence from unavailable coverage. Valid local exports remain
available when another domain lacks an exact interface.

Keep finite, rotating and numerical algorithms as precision providers. Do not
replace them by one representation or force every consumer to build selectors.
Retain exact mathematics after endpoint failure. An explicit coverage policy may
make a separate conservative request; report that choice and its added order.

Required order, selected logical order and physical order remain separate.
For the existing equality-preserving LowerSync contract, a conservative
acknowledgment must be represented in the selected logical program first:

```python
# Schematic token IR at a qualified consumer cut; not Python frontend syntax.
forward = sync.record(p)
sync.wait(forward, q)
returned = sync.record(q)   # Includes the preceding imported wait and q prefix.
sync.wait(returned, p)
```

This is intentionally stronger than the one-way prefix service. LowerSync must
implement the supplied order, not invent reverse-prefix ordering in allocation.
Matching, progress and consumption-before-republication are additional proofs;
payload-order equality alone cannot justify removing command-only handshakes.
Unknown-trip guarded protocols need structural proofs, not finite sampling.

The general scoped parser retains its payload-cut contract. A separate qualified
closed-protocol path now accepts the four-command form above and local two-command
completion at original cuts, including empty prefixes and wait-to-record forwarding.
It does not fabricate payload sites or broaden the general parser to arbitrary
command programs. Its supported grammar and proof are recorded in the ledger.

Regional common completion and conservative prefix services remain distinct
policies. The six public token operators currently have no general ALL-pipe
operator. Before adding regional completion, review whether a qualified ordered
token protocol expresses it exactly, or whether a separate IR/RFC change is
needed. The public terminal drain is not a general in-loop completion export.
No implicit public IR extension is authorized by this plan.

## Execution domains and ownership

Buffer is the owner of tile storage geometry. Pure scalar programs must not need
dummy Buffer loads/stores just to pass the current automatic selection heuristic.
Their scalar memory effects, visibility, native order and invocation exits still
need qualification. A no-sync result is an established property, not failure to
find an effect. Barrier-only/empty-plan ownership requires an explicit supported
PyPTO/PTOAS agreement; do not manufacture SET/WAIT markers.

Mixed AIC/AIV programs keep their existing transport. Local tokens protect local
work around publication, acquisition and release; they do not become TaskIds or
cross-core completion tokens. Transport contracts must expose participants,
borrowed storage lifetime, visibility, legal local cuts and overlapping resource
claims. Paired domains are qualified together where the proof requires it; a
partial ownership change must not invalidate the peer protocol.

Atomic output needs its registered read/write modes, native implementation,
footprint/enclosure and completion contract. Local autosync does not establish
inter-worker atomic semantics or algorithmic race freedom. The existing eligible-ID policy,
native scalar rule and qualified accumulator exemption remain unchanged.

## Milestones and dependencies

Each implementation gate requires an architecture/correctness review and
independent bounded validation before dependent work proceeds. This is a coherent
end-state delivered incrementally, not a minimum-diff objective. Existing
services must be migrated when extended; delete replaced duplicate paths.

### M1 — Shared qualification and domain ownership

Extend `SelectBufferIRFunctions`' Boolean admission to retain a typed first
obligation (operation, property, stage, source location) without swallowing
internal errors. Connect registry effects, backend selection, storage and
readiness to an input contract usable independently of exact-loop admission.
Use existing result/diagnostic conventions rather than a universal new registry.
Refresh the existing whole-model capture and record every generated domain,
including externally owned communication and scalar work.

Before: an unsupported atomic store leaves a function in Tile IR and autosync
reports only "requires physically placed Buffer IR". After: the coverage report
identifies atomic Buffer lowering/implementation qualification as the first
obligation, distinct from any downstream mathematical or resource refusal.

Acceptance: identical accepted exact outputs, complete leaf accounting, explicit
unknown vs no-access tests, metadata vs data effects, overlapping views and
Call/Submit scope tests. Distinguish an unavailable synchronization footprint
from an invalid address calculation; a sound enclosure cannot discharge the
program's memory-safety preconditions. No missing information is converted to
an empty summary.

### M2 — Complete the operation and storage contracts

Use the existing Buffer recipes and analyses to connect selection, lowering,
effect recovery, footprint or enclosure, native claims and emission. Work in
generic families: row/scratch, partial-valid views, scalar/indexed access and
atomic transfer. Native meaning is a prerequisite, not inferred from op names.
Each family is independently gated; the first local conditional-buffer slice
does not depend on completing atomic or cross-core qualification. Update
bindings/docs only if a public contract changes.

Acceptance: row reductions with scratch reuse; overlapping typed views; changing
valid extents; scalar reads feeding addresses/control; atomic read/write effects;
negative unsupported native variants. Check non-Qwen reduction, indexed and
elementwise fixtures alongside the corresponding original Qwen leaves. Reject
invalid valid-extents/overflow distinctly from uncertain but soundly enclosed
locations; supplied memory-validity assumptions remain named caller contracts.

### M3 — General structured conservative construction and protocol lowering

Use the existing structured statements. For basic may-access facts, sequence
composes transfers, branches union, and loops take a bounded monotone fixed
point with zero-trip/backedge/final-false paths. Track class, mode and source
pipe; protect non-storage requirements separately. Begin without KILL.
Extend actual-token interpretation to genuine current-prefix command cuts,
including empty prefixes, forwarded waits and command-only paths.

Example: `if refresh: load(x); use(x)` must protect the use on both paths under
the initialized-input contract; a skipped load cannot erase incoming history.
The chosen conservative policy emits its selected protocol in logical IR,
including any acknowledgment strengthening, before physical lowering.

Acceptance: skipped/conditional writers, both service orientations, zero and
multiple visits, counter resets, fan-out and empty-prefix semantics, command-only
deadlocks, scalar-produced control and insufficient reverse directions/resources.
Interpret actual emitted commands independently. Missing progress or reuse proof
is a refusal, not permission to change allocation or add hidden synchronization.
Prove loop-local logical matching identities, branch-compatible command visits,
acknowledgment reset/rearming and progress for the admitted unknown-trip control
structure. A may-access fixed point proves none of those protocol properties.
Do not transfer direct-handoff allocation premises to command-only protocols;
any additional sufficient certificate must be justified through existing resource
interfaces, or remain unavailable.
Common-completion encoding receives a separate design subgate if needed.

### M4 — Retained children and existing communication boundaries

Export sound child effects without requiring exact GM selectors. Retain compact
mathematics, original coordinates and internal token code. Prove local entry
readiness, closed notification matching, pending payload summaries, joint palette
ownership and cross-visit rearming. A child boundary itself completes nothing.

Scope clarification (2026-11-15): cross-core synchronization construction is
**not** part of these milestones. Preserve explicit/native/compiler-owned
`tpush`/`tpop`/`tfree` handshakes, peer matching and epochs. Local autosync may
consume qualified boundary effects and resource reservations, but must neither
synthesize peer protocols nor claim a local ALL establishes remote completion.
Where those local boundary contracts are unavailable, preserve delegation and
report the missing interface. Adding a new cross-core algorithm is deferred.

Acceptance: exact repeated child in an irregular parent with shared storage,
zero/skipped/reset visits, palette collisions and missing boundary evidence.
For an existing communication operation, preserve the original peer command
order and ownership; test local readiness, borrowed-buffer lifetime and resource
coexistence only where its selected contract supplies them. Check added and
removed payload prerequisite pairs, not only counts. Preservation of child code
does not prove no added global order.

### M5 — Relational caller facts and parameterized exact composition

This is an independent precision/coverage track after M1, not a prerequisite
for all conservative synchronization. Extend invocation evidence using
`CallShapeBindingPlan`, `arith::Analyzer` and existing original-coordinate
services. Preserve established relations instead of reducing everything to
independent numerical intervals. Extend activation partition/export only for
declared symbolic forms with controlled size and executable predicates.

Example: `offset = layer * 5120` needs a justified relation to the allocation's
row extent and checked narrowing/arithmetic. A benchmark helper setting
`rows = layers * 5120` is not an enforced compiler premise. Likewise `6*band`
and `17*band` must retain their common SSA parameter rather than unrelated ranges.
No caller-value enumeration or whole-nest expansion is allowed.

Acceptance: original projection/gate-up source, invalid/missing caller relations,
multiple callers and submits, narrowing, zero/short trips, reset identities,
independent actual covers/all event queries/local and GM selectors. If a sound
enclosure resolves only synchronization geometry, report the remaining program
validity and readiness premises separately.

### M6 — Whole-model acceptance and deployment

Requires the applicable contracts from M1–M4 and any unresolved parameter
requirements from M5. Establish explicit empty/barrier-only ownership with
PTOAS before counting scalar leaves as PyPTO-owned. Preserve existing task and
transport owners while requiring PyPTO ownership for supported local autosync
domains. A delegated local leaf remains a coverage gap, not an external owner.

Compile the unchanged pinned Qwen model with strict expected-ownership checks.
Record input configurations, generated specializations/domains and executed
tests as separate denominators. Test task transitions, repeated invocations and
final stores. Compare PyPTO/PTOAS with identical terminal-completion policies.
Host code generation, logical correctness, physical feasibility and device
behavior are separate outcomes. Default dispatch stays unchanged until a
separate deployment decision.

Dependency order: M1 first; M2 by operation family feeds M3; M3 feeds M4; M5
can proceed independently once caller contexts are established; M6 joins them.
The compiler expert recommends a conditional buffered transform as the first
vertical slice. Its generic fixture covers dynamic counts/valid extents, skipped
stores, local reuse, zero visits and resets. The original Qwen candidate is
`mlp_out_cast`, whose captured shape is:

```text
for hb = block; hb < 40; hb += 20:
    load a [128,128] FP32 tile
    if hb == 0: store a [1,8] slice to the completion tensor
    cast the full tile to BF16
    expose a [valid_tt,128] view
    store that view at [m0,hb*128]
```

First diagnose its precise Buffer view/validity refusal and establish the scalar
parameters' contracts; this is a proposed witness, not a certified admission.
Do not specialize `valid_tt` or loop counts to captured values. The generic
family drives the interface; passing Qwen's two-visit launch is insufficient.
This exercises shared effects, structured conservative construction, actual-token
lowering and physical reuse without requiring atomic or cross-core semantics in
the first implementation. Those remain required later for the whole model.

## Validation, cost and review contract

All gates keep independent required-order extraction separate from compiler
generators. Check access agreement/enclosure coverage; actual retained covers
where exactness is claimed; all four event queries; selector absence vs
unavailability; logical command order; physical matching/progress/reuse; and
payload/control preservation. Recheck compact K, replay, zero-inner-body,
scalar/Cube exemptions and terminal drains. Bounded execution tests supplement,
not replace, parameter-uniform arguments.

For dataflow, let J be the qualified class/mode/pipe fact universe. Delta
propagation should charge O((V+E)J) fact propagation and O(VJ) retained sets,
plus access recovery, service lookup and protocol work. This is not automatically
O(N log N). Decide a sound class-coarsening policy or a documented complexity
exception before broad production admission; never cap work and publish an
incomplete result. Exact geometry/interface output sizes remain explicitly
charged. Repeat proofs, symbolic predicates and selected command circuits must
not unfold runtime trips.

Measure static sites, class incidence, control depth, symbolic dimensions and
trip counts independently. Record rejected attempts, conversions, retained
interfaces, analysis/export/endpoint/allocation time, static/executed commands,
guard work and excess payload order. Optional service pruning follows a joint
coverage/protocol certificate and is not required for the first correct route.

The previous PTOAS `PrefixInput`, `PrefixServices`, `PrefixRecipe` and child
qualification work are sources of tested algorithms and negative cases. Reuse
their reasoning where premises match; do not transplant their MLIR adapters,
native assumptions or physical-order strengthening into PyPTO unchanged.

Each gate's review must identify existing owners reused, duplicate code removed,
new proof obligations, independent tests, size bounds and rollback behavior.
Unsupported effects, geometry, readiness, mathematical/export capabilities,
resource evidence and ownership remain distinct refusals. A whole-model promise
is conditional on these contracts; no finite set of successful kernels proves
arbitrary-kernel synchronization.

## Design review disposition

Both reviewers accepted the design after making two obligations explicit:
unknown-trip protocol identity/progress/rearming needs its own proof, and a
conservative enclosure cannot establish program address validity. Acceptance is
for the plan, not the unimplemented algorithms, proposed native contracts or
Qwen admission. No compiler source, allocation algorithm, protocol or deployment
setting changed during that initial design assessment. The companion
[source-audit receipt](41-general-autosync-architecture-plan.json) pins the
inspected sources and historical evidence; it contains no new performance data.

## Implementation gate ledger

The [implementation receipt](41-general-autosync-implementation.json) preserves
historical M1/static-cast evidence and adds current source/tool pins, raw logs,
all 32 generated leaves and the 47 other domain names. Accepted slices do not
satisfy every original milestone. No device execution or all-function ownership
is claimed.

| Milestone | Delivered scope and evidence | Remaining contract and responsible owner |
| --------- | ---------------------------- | ---------------------------------------- |
| M1 | `71b83d34b`: shared static operation qualification and typed first obligation; original 984-test gate and 32 byte-identical C++ leaves. | Shared readiness/operation producers still owe scalar/indexed value visibility and unsupported execution semantics. Effects alone are insufficient. |
| M2 | `fb0727c53`, `76221dc46`, `eff99b7de`, `ed7590db9`, `c4b9c237e`: static cast, contiguous views, implicit scratch/MUL, per-use INDEX slot proof and default Rsqrt. `a111cd34a` adds the qualified atomic family. | Native implementation and geometry owners must qualify other layouts, dynamic casts, high-precision Rsqrt and broader atomic modes. Existing GM initialization, cross-worker visibility and reduction determinism are not supplied by local atomic completion. |
| M3 | `ce4d593b5`: explicit `conservative_prefix` production policy, no-KILL effects, actual closed command protocol and physical lowering. | Admission still requires qualified effects/readiness and local execution. This is not general scalar/transport recovery. |
| M4 | `ce4d593b5` consumer plus newly tested public `prepare_native_sync_children` batch producer (`362716444`). Handles retain one whole-original exact analysis; all-public producer → Insert → Optimize → Lower → PTOAS and independent command oracle pass in the 444-test gate. | The retained-child/resource owners still owe eventful matching, joint palettes, rearm and transport contracts. The implemented native-only resource-empty slice does not complete that broader milestone. |
| M5 | `fad11f7f4`: checked scoped numerical caller correlations; 68 tests and fresh Qwen C++ equality. | Invocation/relational providers still need symbolic activation/shape relations, carried-state and external ABI evidence. No benchmark bounds are injected. |
| M6 | Strict named-domain harness, final 32-leaf census, ten compactness rows and six bounded command/byte checks. The 2 exact and 11 prefix required-name sets pass. | The 2,437-test broad gate passes. Whole-model acceptance remains open: 21 prefix delegates, 47 other unverified domains, and empty/barrier-only downstream ownership agreement. |

The narrow atomic recipe is default A2/A3 FP32 Acc → packed ND FP32 GM atomic
add. It models the destination as read/write through the shared call-qualified
effect registry, preserves its existing transfer geometry and selected native
configuration/reset implementation, and claims no hidden flags or scratch.
The [pinned native audit](41-general-autosync-atomic-audit.json)
separates this local contract from initialization and cross-worker semantics.
An unsupported variant is not treated as an ordinary write or empty effect.

The public M4 factory requires an active context's target and explicit disjoint-GM
premise for **one whole original placed-function exact analysis**. Requested loop
scopes are original, unique and nonoverlapping; failure returns no partial list.
There is no detached child analysis or automatic backend search. Each handle owns
the immutable mathematics after the call returns. The exporter still rejects
external child effects, unsupplied caller/launch premises and event resources;
its conservative parent does not inherit the root's disjoint-GM premise. See the
[public API contract](../passes/60-logical_sync_pipeline.md#closed-current-prefix-transactions).

M5 proves original machine-width operands and intermediates before simplifying
correlations. Bindings are valid at their definition cut; branch facts and local
definitions are restored on exit, and unknown callers poison joined numerical
bounds. A later branch cannot legitimize an earlier overflow. This produces
numerical evidence for existing consumers, not a symbolic callee relation ABI.

## Earlier atomic checkpoint: qualification and reproduction

The earlier atomic-checkpoint receipts are under
`build-py314/general-autosync-20261115/m6-atomic/{exact,conservative_prefix}`.
Both use compiler base `c4b9c237ed1606fa29484504c93955c5c3002fec` **plus the recorded
dirty-source hashes and binary**, not an assumed clean commit. Library/model pin:
`b72c66d9df69ba5f11982b212530a255c050988d`; unchanged Qwen3-14B prefill, two layers,
batch 1, sequence 128, dimensions 5120/17408. The machine receipt records the
Python version, matching extension hash, PTOAS 0.72 launcher/native tool hashes,
exact argument vectors, environment and source locations. Full model input/hash
inventories remain in the pinned raw receipts rather than being copied here.

| Policy | C++ leaves | PyPTO-owned | Delegated | Required-name gate | Inclusive seconds | C++ bytes |
| ------ | ---------: | ----------: | --------: | -----------------: | ----------------: | --------: |
| exact | 32/32 | 2/32 | 30/32 | 2/2 | 6.5621 | 1,399,234 |
| conservative_prefix | 32/32 | 11/32 | 21/32 | 11/11 | 7.7847 | 1,440,504 |

The table describes the **fresh two-layer capture only**; historical one-/two-layer
observations retain their own denominators. C++ sizes compare the same 32 generated
artifacts, not runtime cost. The 47 other domains are unverified, not certified
external owners. These single inclusive timings establish no speedup.
C++ bytes are actual file lengths; per-file hashes were checked. No device test
was run. Earlier pre-Rsqrt census commands requested no strict names; they remain
historical census evidence only. The later Rsqrt capture passed explicit 2/10
name sets without ownership gain: `final_rmsnorm` progressed from Rsqrt refusal
to `tile.col_expand_mul`. Atomic qualification subsequently makes `down_proj`
prefix-owned; exact still refuses its unproved caller coordinate range.

Reproduce using the existing `tests/ut/ir/transforms/qualify_qwen_whole_model.py`
with `--layers 2 --policy exact` or `--policy conservative_prefix`, the recorded
library/tool paths and **every** recorded `--require-function` argument. Exact
requires `mlp_out_seed` and `silu`; the eleven prefix-owned names are precisely the
`pypto` rows below. The receipt pins the full command, required names and outcome;
omitting names changes the qualification into a census.

| Function | Exact owner | Prefix owner | Prefix first obligation | C++ bytes exact / prefix | Prefix SET / WAIT / barrier |
| -------- | ----------- | ------------ | ----------------------- | -----------------------: | --------------------------: |
| `attention_finalize_phase` | ptoas | ptoas | type_layout/Var | 15709 / 15709 | 0 / 0 / 0 |
| `down_proj` | ptoas | pypto | — | 23962 / 31408 | 104 / 104 / 27 |
| `final_rmsnorm` | ptoas | ptoas | operation_recipe/tile.col_expand_mul | 19185 / 19185 | 0 / 0 / 0 |
| `gate_up_proj` | ptoas | pypto | — | 30559 / 41172 | 144 / 144 / 37 |
| `kv_proj` | ptoas | pypto | — | 31191 / 42357 | 144 / 144 / 37 |
| `lm_head` | ptoas | ptoas | type_layout/Var | 22780 / 22780 | 0 / 0 / 0 |
| `mlp_out_cast` | ptoas | pypto | — | 12696 / 13466 | 16 / 16 / 5 |
| `mlp_out_cast_0` | ptoas | pypto | — | 12727 / 13497 | 16 / 16 / 5 |
| `mlp_out_seed` | pypto | pypto | — | 7371 / 7714 | 4 / 4 / 3 |
| `out_proj_aic` | ptoas | pypto | — | 19113 / 24565 | 72 / 72 / 19 |
| `out_proj_aiv` | ptoas | ptoas | operation_recipe/tile.cast | 13110 / 13110 | 0 / 0 / 0 |
| `out_proj_aiv_0` | ptoas | ptoas | operation_recipe/tile.cast | 13142 / 13142 | 0 / 0 / 0 |
| `post_rmsnorm` | ptoas | ptoas | type_layout/Var | 22611 / 22611 | 0 / 0 / 0 |
| `q_proj` | ptoas | pypto | — | 19104 / 24556 | 72 / 72 / 19 |
| `qk_pv_partial_merge` | ptoas | ptoas | type_layout/Var | 42305 / 42305 | 0 / 0 / 0 |
| `qk_pv_partial_plan_aic` | ptoas | ptoas | type_layout/Var | 208297 / 208297 | 0 / 0 / 0 |
| `qk_pv_partial_plan_aiv` | ptoas | ptoas | type_layout/Var | 208459 / 208459 | 0 / 0 / 0 |
| `qk_pv_single_block_0_aic` | ptoas | ptoas | type_layout/Var | 85114 / 85114 | 0 / 0 / 0 |
| `qk_pv_single_block_0_aiv` | ptoas | ptoas | type_layout/Var | 85276 / 85276 | 0 / 0 / 0 |
| `qk_pv_single_block_aic` | ptoas | ptoas | type_layout/Var | 85106 / 85106 | 0 / 0 / 0 |
| `qk_pv_single_block_aiv` | ptoas | ptoas | type_layout/Var | 85268 / 85268 | 0 / 0 / 0 |
| `qkpv_plan` | ptoas | ptoas | type_layout/Var | 139663 / 139663 | 0 / 0 / 0 |
| `rms_recip` | ptoas | ptoas | type_layout/Var | 15858 / 15858 | 0 / 0 / 0 |
| `rms_recip_0` | ptoas | ptoas | type_layout/Var | 15892 / 15892 | 0 / 0 / 0 |
| `rope_kv_cache` | ptoas | ptoas | type_layout/Var | 81374 / 81374 | 0 / 0 / 0 |
| `save_prefill_last_token_group` | ptoas | pypto | — | 8785 / 8910 | 4 / 4 / 3 |
| `save_prefill_last_token_single` | ptoas | pypto | — | 8396 / 8521 | 4 / 4 / 3 |
| `silu` | pypto | pypto | — | 14556 / 13564 | 40 / 40 / 11 |
| `token_embed_group` | ptoas | ptoas | type_layout/Var | 12043 / 12043 | 0 / 0 / 0 |
| `token_embed_single` | ptoas | ptoas | type_layout/Var | 10556 / 10556 | 0 / 0 / 0 |
| `x_gamma` | ptoas | ptoas | operation_recipe/tile.cast | 14497 / 14497 | 0 / 0 / 0 |
| `x_gamma_0` | ptoas | ptoas | operation_recipe/tile.cast | 14529 / 14529 | 0 / 0 / 0 |

**The command columns are not a whole-plan optimization comparison.** Exact and
prefix own different leaf sets (2 versus 11), and their published PyPTO-stage
metadata excludes downstream PTOAS insertions for 30 versus 21 delegated leaves.
A delegated row's zero therefore does **not** mean synchronization-free C++.
The respective 8/8/8 and 620/620/169 SET/WAIT/barrier totals describe only those
unequal owned subsets, not a benefit or regression of complete synchronization.
They are static sites, not executed command counts. Coarse enclosures can add
substantial order/control work. The machine table also retains original and
lowered For/If/While counts.

For every PyPTO-owned leaf in this capture, the qualified path is shared effect
qualification → selected exact/conservative policy → actual logical command
program → certified physical lowering within the eligible directional pool →
C++ emission. The current target catalog has eight hardware IDs before policy
and native-claim exclusions; the public capture does not request a six-ID cap.
Restricted-pool tests are separate evidence. This is a compiler certificate for
the admitted local protocol, not minimum-ID or device validation. Delegated leaves
report their first PyPTO obligation separately and proceed through PTOAS's own
allocation pool; their C++ outputs are **not** certified by the PyPTO path. Neither
local atomic completion nor this ownership flag proves GM initialization,
cross-worker visibility or deterministic reduction.

A separate inventory of actual owned `after-sync.msgpack` outputs observes only
literal event IDs 0 and 1 across the two exact-owned and eleven prefix-owned
leaves. The hashed `m6-atomic/literal-id-inventory.json` retains every direction.
It excludes native macro internals and delegated insertions; this observation
neither imposes a six-ID admission cap nor proves a minimal ID count.

Validation preserves failed checkpoints: the first Rsqrt gate had 21 failures
and 1,093 passes; a later fixture gate had 11 failures and 1,103 passes. Both were
repaired before the final 24-test Rsqrt gate passed. The atomic/public-provider
gate passes 444 tests in 8.56 seconds. The earlier slot gate passed 1,791 tests
with four inherited warnings. **Final broad gate: 2,437 passed, four warnings,
496.55 seconds.** Its exact test manifest and completed log are hashed in the
receipt. The final focused CTest gate also passes 2/2 (VF IR contract and numerical
sync order), 0.35 seconds total. `final-gate/toolchain.json` pins the Python
executable/version/hash, C++/CMake versions, build cache/Ninja hashes, matching
extension, test logs and explicit two-worker build command. Both files are hashed
in the implementation receipt. Failed logs are not replaced by passing receipts.

## Validity premises and construction cost

The proof uses qualified native semantics, not merely the presence of registered
operations. The [A2/A3 pipe-completion contract](https://asc.gitcode.com/api/SIMD-API/c_api/sync/intra_core_sync/asc_sync_pipe.html)
permits V, M, MTE1/2/3, FIX and ALL completion and excludes S. The
[2201 AIC/AIV synchronization tables](https://asc.gitcode.com/api/SIMD-API/c_api/sync/overview.html)
provide the diagonal barriers and directed notification/wait combinations.
Existing DMA/FIX completion operations consume these contracts; this work adds
no new native primitive and supplies no device validation.

1. Original operations contribute may-read/may-write facts by class and pipe.
   Sequence propagation and branch union retain prior obligations. Loop input is
   IN union GEN(body), covering zero visits/backedges without unfolding. There is
   no KILL; possible writes do not erase conflicts or establish value readiness.
2. Each cross-pipe service explicitly supplies `record(p), wait(q), record(q),
   wait(p)`. Same-pipe services use the qualified two-command completion form.
   Physical lowering interprets that order rather than inventing acknowledgment.
3. Forward consumption precedes return publication; return consumption precedes
   next forward publication. Opposite orientations use distinct roles. Sequence,
   branch and sequential-loop structural induction establish closed notification
   state and rearming, including skipped/zero visits. At most two eligible IDs
   per directed namespace suffice for this protocol; a failed pool request is
   not a hardware minimum-capacity theorem. ALL completes payloads but does not
   consume unmatched flags.
4. Native-only children retain original owner/target/scope, no flag lifetime and
   pending may-effects. Native Acc protection removes its qualified hazard only;
   it does not add an invented full completion-to-next-start edge. Eventful or
   peer-transport ownership requires a separate certificate.

Let N count original statement occurrences, C the finite space/pipe/read-write
fact universe, E actual readiness-query DAG visits, and K selected static services.
The implemented ordered-set no-KILL analysis has conservative accounting
O(N C log C + N C² + E) work and O(N C) fact state; C is target-fixed. E deduplicates
within each query, not globally. Requested control paths add their output depth
to one original parent index. Realization retains O(K) static command sites (two
local or four cross-pipe commands per service), with no trip-count expansion.
This excludes exact child-provider construction, requested diagnostics, map/name
lookup, verification and later compilation; it is not a whole-compiler linear bound.

The serial compactness diagnostic covers counts 0/1/3/31/1023 in primitive and
native-child forms. All ten rows complete; six bounded actual-command/physical-byte
checks at counts ≤3 pass. Larger counts are static-only. Primitive rows retain
six sites, 42 facts, 15 services and 17 statement/dataflow visits. Native-child rows
retain six primitive sites plus one child, 94 facts, 21 services, 20 statement and
17 dataflow visits; nonzero child exports visit five source nodes and one summary
node. These named counters exclude whole-root exact analysis cost and peak RSS.
Script and output hashes are in the implementation receipt.

The child command oracle explicitly checks the barrier difference
`outer_visits * (1 - inner_count)`: zero-trip imports can add entry barriers.
Payload order containment, matching and causal reuse remain required. No universal
compactness, fewer-command or speedup claim follows from this fixture.

## Remaining contracts and paper consequences

| Remaining contract | Missing evidence and owner |
| ------------------ | -------------------------- |
| Broader atomic modes/concurrency | Native implementation and execution-domain owners must supply variant effects, valid initial GM state, external ordering/visibility and any determinism guarantee. The admitted local atomic-add slice does not establish these. |
| Scalar/indexed readiness | Readiness owners must prove definition-to-use visibility and valid original cuts; storage enclosures cannot make asynchronous address/control values ready. |
| Layouts/dynamic casts | Selection/lowering/native-contract owners must agree on descriptor, valid extents, precision, scratch and resource effects. Missing recipes are not empty accesses. |
| Eventful children/transport | Child and transport exporters must provide matching confinement, pending effects, joint palettes, rearming and acquire/release ownership. Native-only imports cannot substitute for them. |
| Symbolic relations | Invocation and exact-interface owners must retain checked caller relations and their scope; benchmark values do not establish external ABI premises. |
| Empty/barrier-only ownership | PyPTO and downstream compilation must agree that all obligations are discharged; absence of flags is not a fake ownership marker. |

The mathematical kernels retain their qualified claims. Practical compilation
also pays for recovery, conversion, exports, selected protocol construction,
allocation and generated control. Compact dependence analysis alone proves none
of their whole-compiler bounds. Conservative prefix is an explicitly stronger
selected order, while LowerSync preserves the supplied actual-token order. Partial
coverage identifies specific unprovided contracts; it is not evidence that the
algorithms or hardware are inherently incapable. All six original milestones and
whole-model ownership remain incomplete at this checkpoint.

## Continued M2 gate: column broadcast (2026-11-15)

The existing row/broadcast recipe now represents broadcast orientation explicitly.
`col_expand_mul` repeats a row-major `[1,N]` carrier across an `[M,N]` matrix;
row expansion continues to use a column-major `[M,1]` carrier. Selection,
lowering, verification, native emission and synchronization effects consume that
same recipe. The admitted variant is full-valid aligned FP32 Vec, `M <= 255`,
`N <= 2040`, and `N % 8 == 0`. These bounds prevent narrowing the selected
native implementation's uint8 repeat and block-stride operands. Matrix/output
exact aliasing remains permitted; output/carrier overlap remains forbidden by
the existing storage verifier. No hidden scratch or events are claimed for this
variant. Partial-valid operands and other layouts remain unavailable.

The native audit uses pto-isa `5487a6b0` A2/A3 headers:
`TColExpandBinOp.hpp` SHA256
`00bf43def423bd0701b28ea78784b64ce988f2a7f2e00275f7e9f281c00855ab`, and
`TColExpandMul.hpp` SHA256
`76a3eb7be75560f5ef909ecfad9c61d9a8ac839b6cf80d8a309f74a7f5d386b8`.
Generic frontend cases exercise several shapes and repeated visits through
independent access/cover/query/selector checks, physical matching/reuse and
PTOAS C++ generation. The unchanged RMSNorm example now has an owned exact plan;
LayerNorm's separate missing recipe remains explicit. The refreshed two-layer
Qwen prefix capture generates 32/32 C++ leaves and owns 12/32, adding
`final_rmsnorm` while preserving all eleven prior required owners. The focused
gate passes 80 tests and the broader regression passes 198 tests. The historical
11/32 table above remains the pre-recipe baseline. See
`41-general-autosync-column-broadcast.json` for commands and hashes. This is host-model
qualification, not device validation or whole-Qwen acceptance.

The next type/layout gate requires a transfer construction, rather than a
relaxed descriptor check. Singleton-column GM `[M,1]` currently has neither an
admitted column-major transfer nor an exact partial-valid padded-row transfer.
Runtime-valid column vectors need coordinated descriptor/recipe/footprint
evidence. Rank-one index loads additionally require scalar-memory visibility
and readiness evidence. The existing refusals are retained until those
obligations are established. Eventful child composition and paired AIC/AIV
transport remain subsequent gates; this operation recipe does not qualify them.

## Continued accepted gates and remaining contracts (2026-11-15)

The earlier tables preserve their original source snapshots. The current
operation/native audits, reproduction commands and binary hashes are in the
runtime-valid, runtime-axis, module-ownership and read-only-child companion
receipts. The unchanged model pin remains `b72c66d9`, two layers, batch one,
sequence 128 and original hidden/intermediate dimensions 5120/17408.

| Milestone | Delivered through reviewed gates | First remaining obligation |
| --------- | -------------------------------- | -------------------------- |
| M1 effects and ownership | Typed first obligations, shared effects, original owner/frame index, explicit complete-unit ownership and complete 79-domain owner accounting | Owner observation does not supply native completion/visibility contracts; those remain separate M2/M4/M6 obligations |
| M2 native operation and storage contracts | Column broadcast/transfer, use-site runtime metadata, FP32 row reductions/axis aliases, BF16/FP32 casts | Mixed full/partial ADD value semantics, unqualified scalar GM visibility and broader native variants |
| M3 conservative control and actual protocols | Existing no-KILL transfer, explicit prefix/acknowledgment services, joint pruning and actual-token lowering | A general common-completion or arbitrary command protocol needs its own admitted grammar/progress proof |
| M4 retained children | Local and read-only-GM whole invocations, original frames, pending effects, existing entry rearming and directional palettes | Qualified local boundaries/resource coexistence for existing hand protocols; no new cross-core construction; exact GM boundaries remain unavailable |
| M5 caller and exact composition | Existing bounded caller evidence and compact K/replay retained; original physical-shape ABI metadata can be ready opaque coordinates | Additional symbolic relations/ranges and executable exact GM exports need actual caller evidence |
| M6 complete-model deployment | Opt-in empty/barrier/flag module ownership, final drain, debug/cache rebuild checks; all 32 leaves generate C++ | 16 prefix leaves still delegate; no whole-model device or universal ownership qualification |

The fresh prefix census improves from 14 to 16 owned leaves at the immediate
preceding packed-transfer checkpoint; `x_gamma` and `x_gamma_0` are new. Exact
owns `final_rmsnorm`, `mlp_out_seed` and `silu`. The older 11/32 and 2/32 figures
above are historical, not the current denominator. Neither route silently
converts exact export failure into stronger synchronization.

The read-only adapter is explicitly requested through the existing child API.
Ordinary model capture does not automatically select retained children. Original
Qwen K preparation, logical composition, six-ID physical allocation and C++ now
succeed. Each independent bounded invocation checks 284 actual retained covers,
all 692,224 event queries, 86 boundary samples and 416 original accesses. Original
GM readers remain pending in the parent and their exact selectors unavailable.
The two checked complete parent executions remove 2,188 prerequisite pairs and
add none. Generic fixtures also expose cases with added wrapper order, so no
uniform dominance claim follows. Matching, payload completion and physical reuse
remain separate obligations.

Completion of all original milestones cannot be declared from these results.
The next obligations are operation semantics or explicit backend/caller facts:

1. A full-valid carrier and partial-valid matrix cannot be silently narrowed.
   Runtime row-reduction output leaves inactive rows untouched, so no implicit
   zeros justify mixed full/partial ADD. Enclosing addresses do not fix this.
2. Native scalar same-pipe order does not prove visibility of externally loaded
   control. The runtime's task barrier alone does not establish the required
   cacheline/producer publication contract; benchmark shape facts supply none.
3. Cross-core pushes, pops and frees have distinct local readiness, peer release,
   and resource epochs. Local prefix/ALL completion cannot replace that protocol.

The initial broad test failures, combined exhaustive-audit timeout and audit
memory refusals remain recorded. Research interpreters can unfold bounded
instances; production retained representations still avoid trip-count expansion.
No allocator, ID pool, hardware exemption, expert protocol or zero-inner-body
barrier was changed to obtain the new acceptance.

## M1 completion gate: actual task-control owners

The existing orchestration code generator now publishes `task_domains` in its
read-only result. Its registration owner records only the actual emitted entry,
Graph fragments and dispatched Spmd/Group wrappers. Unreferenced wrappers do not
acquire an owner. Call and Submit continue through their existing shared ABI,
launch, TaskId and dependency validation paths; no second task analysis exists.

After successful whole-unit generation, the existing autosync report joins this
map with every original IR function. Task-control ownership is reported separately
from local PyPTO/PTOAS ownership. It certifies emission by the existing runtime
control owner, not payload visibility, invocation completion or a peer protocol.
An error publishes no qualified task-control report. Downstream per-PTO ownership
and automatic insertion options are unchanged. See the [task-domain receipt](41-general-autosync-task-domains.json) for
pinned sources, whole-model accounting and emitted-source equality checks.

Task-owner status is `task_control_owner_observed`. This records successful
selection of the existing emitter and matching original identity/type/artifact
under its declared wrapper IR contract. It is not an independent canonical-body
verifier or proof of arbitrary hand-assembled wrapper semantics. A strict
`--require-domain-owners` capture checks accounting separately from
`--require-function`/`--require-pypto` local synchronization qualification.

The pinned original Qwen capture now accounts for all 79 observed domains:
47 task-control owners observed, 16 PyPTO local leaves and 16 delegated local
leaves. This does not increase local synchronization coverage. All 32 kernel
C++ files and the orchestration C++ file are byte-identical to the immediate
baseline. The strict owner-accounting gate and existing strict local-name gates
both pass. The broader 147-test gate covers real Call/Submit/Graph/wrapper
emission, unused wrappers, publication failures and module ownership. Two stale
Graph test stubs were updated to the already committed compiler keyword API;
no native or dependency check was weakened.
