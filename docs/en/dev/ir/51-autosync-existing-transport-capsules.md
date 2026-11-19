# Local synchronization around existing transport

Status: **shared production layout and workspace-context implementation installed;
the coherent build and context/integration regressions pass; source hooks pass.
Transport capsule admission remains unchanged**. This extends
M4 consumption of existing communication boundaries. It preserves the selected
TPUSH/TPOP/TFREE protocol, peer pairing, ring epochs and progress rules. It does
not construct a new cross-core or inter-kernel synchronization protocol.

## The missing interface is local evidence

Six original captured leaves remain delegated: `qk_pv_partial_plan_{aic,aiv}`,
`qk_pv_single_block_{aic,aiv}` and `qk_pv_single_block_0_{aic,aiv}`. The pinned
`qwen-pure-write-qualified` capture first reports `type_layout` on a Var, before
local transport qualification. This is not proof that the hardware cannot
support local autosync. Existing owners already know workspace sizes, selected
pipe parameters, local reserves and consumer release structure; they do not yet
publish them to Buffer admission and local synchronization.

The source carries `__gm_pipe_buffer` as a one-element FP32 placeholder with a
four-byte MemRef. Codegen explicitly owns the real workspace size. Treating the
placeholder as a four-byte transport footprint would be unsound. A name match
also cannot certify a workspace. The new consumer must bind evidence to the
original injected parameter identity and its actual pipe setup/call owner.

## Reuse these existing owners

| Source owner | Recoverable fact | Required export |
| ------------ | ---------------- | --------------- |
| `cross_core_pipe.cpp`, `BuildAutomaticPipeSetup` | Directions, slot size/depth, local reserve and original paired setup | Immutable selected pipe identity and parameters; preserve explicit handwritten settings rather than replace them with defaults |
| `RequiredConsumerDepth` in the same file | FIFO pop/free order and supported scope-local queue closure | Retained local borrow/release evidence, including aliases handled by the existing finalizer |
| `tpop_tfree_finalizer.cpp` | View aliases and uses extend the original popped tile lifetime | Original pop, dependent local uses and logical release cut |
| `gm_pipe_layout.h`, `LayoutEvidence`, `PrepareGMSlotBufferLayout` | Effective slot count, independent rings, full workspace and per-pipe offsets | One checked shared layout result consumed by orchestration, codegen and local analysis |
| `pto_ops_crosscore.cpp` | Actual split, dtype, transport valid shape, initializer operands and native spelling | Selected variant identity and original local input/result binding |
| `isa_macro_resources_protocol.cpp` | Pipe-owned ready/free signal reservations, separate local event claims where applicable | Claims selected for the actual direction, target and macro variant, with owner lifetime |
| Existing native qualification and conservative effect services | Target/core admission, local payload effects, workspace separation and local SET/WAIT construction | Consume the transport evidence without a parallel operation-effects registry |

For the captured bidirectional setup, `slot_size=8192`, `slot_num=2` and two
rings imply **32,768 GM bytes**, not four. Each shown local consumer reserve is
16,384 bytes. These are facts of that setup, not universal sizes. General
construction must use checked arithmetic, consistent producer/consumer setup,
actual local depth and all independent pipe regions. Reuse the shared layout
rules rather than copying their formulas into an autosync-only size engine.

## A capsule belongs to the selected original transport

Add a factory-owned local transport capability alongside the existing selected
native implementation evidence. Its owner is the original Program/function pair
and target-specific pipe setup. A per-call contribution references that owner;
call names or externally supplied attributes cannot manufacture it.

The capability must retain:

- Original initializer, injected workspace parameter, reserve/import identities,
  direction, pipe ID, split, dtype, slot sizes/depths and native variant.
- Checked physical enclosures for every GM ring and local reserve, with actual
  overlap against ordinary local buffers. Conservative enclosures do not become
  exact storage selectors.
- Original local input readiness cuts, pending transfer effects and result-use
  readiness; a logical pop result is not automatically a completed local load.
- The borrowed local slot's use interval and its rearm boundary, including views,
  nested scopes and every later operation that can overwrite that slot.
- Owner-lifetime cross-core signal reservations and any selected local event
  claims, kept distinct from ordinary directed pipeline-event pools.
- The unchanged peer command sequence and the supported local progress premise.
  Neither a child boundary nor local ALL establishes peer completion.

The first implementation should qualify the existing A2/A3 GM-backed tile-data
variants with established geometry and lexical setup. Unknown control-payload,
external initialization, mixed variants, escaping borrow/carry and unsupported
split forms keep a precise refusal. This is a support boundary, not a claim that
all other variants lack hardware contracts.

## Native entry, release and progress need separate proofs

The inspected TPUSH, TPOP, FIFO and TFREE sources are pinned at revision
`5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f` in the disk-backed
`native-transport-5487` artifact. The companion JSON records each primary URL,
Git blob and SHA-256. This is source evidence, not proof of the compiler-selected
variant correspondence or runtime/device behavior.

In its A2/A3 implementation, TPUSH may wait for a free ring slot, then issues a
TSTORE on FIX or MTE3 and publishes readiness on that pipe. TPOP waits for peer
readiness, binds a local reserved slot and issues TLOAD on MTE2. Tile-data TPOP
may publish the GM-slot free notification on MTE2. The tile-data TFREE overload
is a no-op. Consequently, the following distinctions are mandatory:

1. **Push input readiness:** complete relevant local source producers before the
   actual FIX/MTE3 read; preserve that read as pending until its local completion.
2. **Pop result readiness:** peer readiness authorizes reading the GM ring; it
   does not by itself complete the subsequent MTE2 write to the local reserve.
   Local consumers need the existing MTE2-to-consumer prerequisite.
3. **Two lifetimes:** GM-ring release after the MTE2 read differs from keeping a
   local reserved slot alive for its consumers. Do not interpret a no-op TFREE as
   a new local completion instruction. The retained FIFO schedule and actual
   subsequent slot writes must establish local reuse safety.
4. **Resources:** ready/free signals live across calls until the selected owner
   drains them. An individual push/pop is not resource-empty. Bind actual signal
   bases and direction variants; do not reserve every alternative union as though
   all variants execute, or silently allocate those signals from local event IDs.
5. **Progress:** added local services must not move a peer wait, publication or
   release, or introduce a wait whose completion requires a later peer command.
   A whole-pipe drain around every transport call is not an automatic proof.
   The supported variant must expose which local transfer completion precedes its
   existing remote notification and which waits only consume remote progress.

The missing native premise, if unresolved, must name the selected operation and
cut: for example, whether a specific pop's free notification is ordered after its
GM read, or whether the next local-ring write can overtake a retained consumer.
Missing exported geometry/FIFO ownership is an implementation task, not an
unavailable hardware premise. Preserve an original accepted peer protocol;
module-level pairing and existing progress qualification remain its authority.

## Construction and rollback

Build the shared pipe/layout owner once from original setup, then export local
contributions at existing command cuts. Buffer selection consumes verified ABI
and reserved-slot bindings without relabeling the workspace as an ordinary tiny
Tensor. Lowering preserves original push/pop/free calls and their result aliases.
The conservative analyzer composes local reads/writes and prerequisites through
its existing services; hidden resources enter the current reservation mechanism.
No exact regional backend, visit enumeration or general protocol solver is added.

Publish a transformed owner only after all local effects, lifetime, target and
resource obligations succeed. Keep original source and peer protocol intact on
failure. Partial success must not suppress the external completion/transport
owner or claim full PyPTO ownership for an unqualified sibling.

## Required validation before implementation acceptance

- General one-way and two-way setups; several slot sizes and local/GM depths;
  actual full workspace versus the four-byte placeholder; multiple pipe offsets;
  mismatched setup, insufficient workspace, overlap and foreign-owner negatives.
- Push-to-local-write, pop-to-local-read and local-use-to-slot-reuse order, including
  views, skipped/zero/repeated visits, nested resets and live borrows. Check all
  original peer operations and their sequence independently of local commands.
- A small paired-core model for the existing protocol: credits, ready/free epochs,
  pending local DMA, blocking waits and final drain. Test that adding the local
  services preserves progress; mutate a release/readiness edge to make it fail.
- Resource namespace and lifetime tests, including collisions, unsupported selected
  variants, and target mismatch. Local ALL must not consume remote notifications.
- Public selection through actual PTOAS C++ with workspace allocation and offsets;
  compare native transport calls/settings before and after. Then recapture the six
  original leaves, reporting each new refusal rather than assuming six successes.

No native transport qualification runs have been performed for this proposal. The six remaining transport
exports are actionable development work; neither this design nor successful C++
generation establishes device behavior or complete model ownership.

## Uniform proof and inspected-work contract

Finite paired-core executions are regression oracles, not the acceptance proof.
The supported capsule must carry a structural invariant for every admitted
execution: setup dominates every use; each branch preserves the incoming borrow
and credit state; a loop body restores the same local ownership invariant for
zero or any number of visits; alias uses remain inside the borrow interval; and
new local prerequisites cannot create a dependency on a future peer action.
Original native credit/epoch logic remains authoritative. A failed branch, loop
or alias premise refuses that capsule without guessing a trip bound.

Charge setup/layout/FIFO recovery once per immutable original owner. Retain an
indexed per-call export so each requested operation resolves its evidence without
a whole-function rescan. Record distinct source statements/expressions, setup
entries, layout entries, borrow/use incidences, per-call lookups and rejected
attempt work separately. Caching only successful calls does not bound repeated
refusal work: immutable failed recovery must also retain its precise obligation.
No claim of global linear compiler time follows from this local contract.

The initial evidence component may recover shared original setup and layout while
returning explicit unmet native/lifetime/progress obligations. Admission stays
closed until all selected-variant proofs are consumed by the production path.
A diagnostic-only parallel inventory is not completion of this milestone; existing
layout and transport consumers must ultimately share the retained owner.

## Independent design gate

Architecture/correctness review accepted this design for implementation. This
acceptance is not an implementation, executable ownership or validation claim.

Bounded paired-core schedules must supplement a structural uniform lifetime and
progress argument for admitted repeated visits/resets; they cannot replace it.
The cost ledger must separate one-time setup/layout/FIFO recovery, indexed call
contributions, effect-incidence output and rejected attempts. Repeated whole-body
or alias-DAG scans per call and runtime slot/visit enumeration are excluded.
Selected 5487 TPOP/TPUSH/FIFO/TFREE sources are pinned in the disk-backed
`native-transport-5487` artifact. Tile-data TFREE overloads are no-ops; the
GlobalData free path is distinct. Complete selected-variant correspondence
remains an implementation admission gate.

## Production layout substrate

`gm_pipe::LayoutEvidence` now retains the exact immutable Program, original setup
statements and indexed function-call graph. Orchestration entry and Graph emission
share one owner through `KernelRegistrations`; the backend supplies contexts from
that same owner to PTO codegen. The two former layout scans are removed.
Existing `gm_pipe_layout.h` remains the sole ring/depth/footprint formula owner.

Geometry is decoded lazily once per requested setup. Reachable-root and whole-module
results are immutable; successes and layout refusals are memoized by original root
identity. Foreign roots and ambiguous repeated statement identities cannot export
an original setup cut. The unused standalone workspace-size wrapper has been deleted.
Selected malformed kwargs keep their existing exception behavior; unrequested
malformed geometry in unrelated functions is not decoded.

Statement-only discovery covers concrete scope bodies, including Runtime,
Hierarchy and SIMT, so layout consumers no longer disagree about hidden geometry.
This does not admit native transport in any of those contexts. No expression DAG,
peer schedule or runtime visit is expanded. The cost ledger separates source
statement occurrences, setup decodes/lookups, call edges, closure function/setup
incidences, cache hits and refusals. Distinct overlapping root queries may revisit
indexed incidences and retain separate output maps; this is not a claim of total
linear work across all roots.

New focused tests cover ownership, geometry, failure caching, recursive calls,
shared cuts, integer overflow, nested scopes and 8/32/128-leaf scaling. Existing
public orchestration/PTO codegen tests remain the production regression gate.
The earlier substrate gate passed 168 tests before the context repair; the final
context and integration gates below pass. This substrate supplies layout evidence only: native
readiness, FIFO lifetime, resource and uniform progress proofs are still required
before any transport capsule or new leaf ownership is accepted.

## Implemented layout-context consistency

A shared formula does not establish a shared allocation context. Orchestration
sizes `ReachableFrom(actual_launch_root)`, whereas the former PTO emitter assigned
`WholeProgram()` offsets. For unrelated roots with `(id,dir)=(3,3)` and `(0,3)`,
each using 2,048 bytes, the first root can receive a 2,048-byte allocation but an
emitted offset of 2,048. Ordinary backend generation already separates Groups,
so that example must be tested at both the direct codegen API and actual Group
boundary before claiming an ordinary compiler failure. Ungrouped peer inclusion
and independently owned workspaces are also relevant. Increasing every allocation
to the entire model's footprint would undo the per-callee workspace contract.

The accepted repair extends the existing layout owner with an immutable
`WorkspaceContext`, issued only by `LayoutEvidence::BindRoot(original_root)`.
The context has no public constructor or mutable fields. It retains the source
owner, original root, checked workspace formal/call forwarding, admitted setup
cuts and the exact `LayoutResult` used for both allocation bytes and offsets.
A geometry result alone is not a context certificate.

Binding must establish that every contributing original setup belongs to the
root's actual workspace: find the existing injected formal role, then validate
original call/Submit argument forwarding through the existing ABI mapping.
The argument identity, rather than the reserved parameter name, proves sharing.
A missing, ambiguous, separately allocated or unsupported alias/carry binding
refuses the context. Duplicate statement occurrences cannot manufacture a unique
setup cut. Reuse the original call index and existing ABI helpers; do not scan
whole bodies for each contribution. No physical size is inferred from the
four-byte placeholder. Contexts with no setup need no workspace proof.

Concrete API and consumer changes installed for this subgate:

1. Expose the existing production layout owner through the codegen binding and
   add its opaque `bind_root` result. Keep inspection counters on that same owner;
   do not add another layout analyzer or an attribute-based certificate.
2. `_generate_single_chip` creates one owner for its immutable original Program.
   It binds the actual Group root before reducing to the member PTO Program, or
   the actual ungrouped launch root before including peer/helper definitions.
   Merely adding a peer definition does not enlarge the root allocation.
3. `GenerateOrchestration` accepts an optional shared layout owner. Its existing
   create-use resolver selects the actual root; the corresponding bound context
   supplies allocation bytes. Entry and Graph emitters share that owner.
4. Both `PTOCodegen::Generate` and `GenerateBundle` accept an optional immutable
   workspace context, forwarded by `_generate_pto_bundle` in normal and diagnostic
   routes. Every emitted setup must match `SetupAt(original_function,original_cut)`
   and belong to that context, with unchanged geometry. Original function identity
   remains valid when a reduced Program reuses the original Function objects.
   Unknown or copied/modified function/cut identities refuse before publication.
5. Without an explicit context, direct PTO generation may infer one only when all
   GM-bearing entry roots have individually valid, identical complete layouts
   (same keys, offsets and extents). A single root is the ordinary case. Differing
   entry layouts refuse with a request for an explicit original-root context;
   the API must not silently combine them into a shared allocation. A Group whose
   members intentionally use different subsets supplies its explicit context.
6. Context handles are process-local capabilities, not serialized attributes.
   After deserialization or source rewriting, reconstruct them from the complete
   original Program and root. A reduced debug module lacking that owner must
   satisfy the direct default rule or refuse; it cannot recover authority from a
   matching function name, saved byte count or user-authored metadata.

Validation must pair actual orchestration allocation with actual emitted offsets:
unequal, reversed-ID independent Groups keep separate extents; a valid multi-pipe
Group keeps disjoint shared offsets; direct unrelated roots refuse; foreign owner,
changed setup, missing forwarding, separately bound workspace and ambiguous cut
cases refuse. Graph forwarding, repeated generator use, `generate_bundle`, debug
output and fresh serialized-original reconstruction must follow the same rule.
No native transport admission changes as part of this repair.

Charge once-owned source and forwarding indexes, requested context closure work,
setup-membership validation, cache hits and refused bindings separately. Distinct
contexts may have overlapping indexed closures; retained maps and validation cost
are charged to that output. The implementation is installed and frozen for build and regression review.
The final context and integration tests pass; results and limits are recorded below.

The installed implementation keeps immutable `LayoutSource` separate from query
memoization, so retained contexts do not form a cycle with their cache. It checks
every contributing Call/Submit occurrence, including repeated calls to the same
callee, using the existing caller-supplied ABI mapping. The narrow forwarding
contract requires the exact original formal; it does not infer alias/carry
relationships. Reduced-module validation uses immutable original Function identity
to seal its body and indexed setup cuts without rescanning those bodies. The
public `codegen.GMPipeLayoutEvidence` is the production owner; the testing name is
only an alias of that binding. The old standalone size adapter is removed.

The added context regression pairs actual orchestration allocations of 3,072 and
4,096 bytes with their separately emitted Group offsets. It also covers explicit
and default entry contexts, Call/Submit forwarding, memoized refusal, ambiguous
cuts, foreign/deserialized identities, fresh reconstruction, owner lifetime, and
repeated normal/bundle generation. Native capsule readiness, rearm, resource and
progress obligations remain closed and outside this batch.

A reviewed compatibility exception preserves the existing unmanaged direct-codegen
spelling path: no explicit context and no managed-workspace formal anywhere in
the emitted Program. That path retains the former whole-module numeric geometry
checks, emits no GM pointer or managed offsets, and publishes no workspace
certificate. It does not assert that an A2/A3 native pipeline can lower an omitted
GM operand. Local PTOAS source requires that operand for A2/A3; these historical
unit fixtures deliberately exercise spelling before the injection prerequisite.
Any managed-role formal (including malformed/ambiguous forms), mixed mode or
explicit context uses the strict context checks. Existing `BindRoot` refusal is
unchanged. The regression compares each unmanaged function body with its separate
emission and checks numeric errors, mixed-mode refusal and absence of GM offsets.

## Final worktree closure

The final coherent build passes. The passing module gates contain 681 verifier,
token-erasure and layout cases; 26 shared-DAG cases under the original 10 s
watchdog; 274 PTO-codegen cases; 151 native/orchestration cases; 96 ownership,
runtime and backend cases; and three C++ resource/order tests. These overlapping
runs are reported separately, not added into a new aggregate test count.
The companion JSON pins each source file, loaded binary, Python, PTOAS, model,
commands and logs under `final_validation`.

The original two-layer model still generates 32 C++ leaves: 21 local PyPTO
owners, 11 delegates and 47 existing task-control owners observed. All 32 C++
files are byte-identical to the accepted scalar-write checkpoint. The eight-case
actual-command audit passes, including replay without clearing physical history.
Its staged shared-carrier reference is already totally ordered: zero excess
establishes neither improved overlap nor minimum commands. No new kernel or
native transport admission, protocol, ID budget or allocation algorithm is added.

The earlier 5,014-pass/one-failure run is preserved. It exposed partial erasure of
a guarded token cut; the valid-fixture regression and pre-import refusal now pass.
Other preserved failures are the aggregate120s watchdog and stale API/metadata
expectations. The final smaller integration batches retain that120s watchdog.
No failed run is relabeled as an all-green full-suite result. Final hook and
receipt review status is recorded in the machine-readable checkpoint.

## A2/A3 consumer slot checks

Automatic setup and AutoSync fallback now share a FIFO walk over the original
pop and free markers. Automatic setup sizes its single generated queue.
Fallback counts outstanding entries separately for each declared pipe ID.
A supported loop or branch restores its incoming queues, so the analysis
does not enumerate visits. The fallback reports this marker depth against the
declaration reached through the consumer initializer's reserve operand and
keeps the original Buffer admission reason. Marker depth does not establish
alias-use lifetime, local DMA completion, or peer progress.

Before no-split AIV dispatch is duplicated, the compiler rejects a concrete
slot overwrite in the original active lane. The proof requires a single
dominating consumer setup, an unconditional sequence of fully valid dense FP32
Vec pops, and a direct registered tile data read of an older outstanding value
after its slot wraps. Unknown shapes, early exits and alias reads remain
outside this rejection. PTOAS retains grouped synchronization ownership and the
original native peer epochs, including the zero-valid handshake lane.
