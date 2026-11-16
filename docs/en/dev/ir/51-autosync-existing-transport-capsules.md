# Local synchronization around existing transport

Status: **design accepted for implementation with the structural proof and cost
requirements below; no implementation or validation claim**. This extends
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
| `gm_pipe_layout.h`, `ComputeGMPipeWorkspaceElements`, `PrepareGMSlotBufferLayout` | Effective slot count, independent rings, full workspace and per-pipe offsets | One checked shared layout result consumed by orchestration, codegen and local analysis |
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

The inspected local source snapshot is
`physical-sync-prototype/isa/hw-native-sys-pto-isa-f83d4b7`. It is useful source
evidence, but final admission must pin the implementation actually selected by
the compiler and verify the same obligations there; this snapshot is not itself
proof of the current runtime/toolchain identity.

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

No new runs have been performed for this proposal. The six remaining transport
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
Selected5487 TPOP/TPUSH/FIFO sources are now pinned in the disk-backed
`native-transport-5487` artifact; complete selected-variant correspondence,
including TFREE, remains an implementation admission gate.
