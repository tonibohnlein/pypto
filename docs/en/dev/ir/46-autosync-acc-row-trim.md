# Addressed accumulator row trims: operation-contract subgate

Status: implementation and independent source review accepted; the coherent
139-test gate, subsequent 151-test alias-policy gate and unchanged-model capture
pass. Final source hooks, coherent post-format rebuild, a fresh 151-test rerun
(106.14 s) and the three C++ resource/order tests also pass. This is a reusable M2 metadata/export repair
within the [architecture plan](41-general-autosync-architecture-plan.md). It
follows the separate [GM scalar staging gate](44-autosync-gm-scalar-staging.md).
No accumulator, tile-size, allocation or synchronization-policy change is proposed.

## Missing interface and supported family

An already qualified static Acc producer/carry names physical storage `[M,N]`.
A later alias restricts only the rows observed by a GM writeback:

```text
acc = qualified static matmul / matmul_acc carry  # original full descriptor
trimmed = tile.set_validshape(acc, rows, N)       # same complete byte window
if rows > 0:
    tile.store(trimmed, offsets, output)           # original execution cut
```

The static writer and its loop-carried descriptors remain unchanged. No runtime
metadata is propagated back into matmul, no tiling is changed, and no new payload
operation is inserted. The trim is storage aliasing plus separately owned native
metadata, not a copy and not proof that the writer has completed.

Admit only an A2/A3 AIC addressed Acc family with the existing supported dtype,
fractal layout, physical shape, allocation capacity and native writer contract.
The initial producer/carry has established full static valid `[M,N]`; the trim
retains full columns, zero offset, the identical dtype/layout/fractal and
`CompactMode::Null`. The ready row expression must satisfy `0 <= rows <= M`
under the existing defined-operation/range contract and be available at the
original use. A native store executes only with strictly positive rows. Preserve
native field bounds, destination bounds and the original GM alias policy.

A selected consumer cannot infer `rows = M` or overwrite definedness from an
allocation size. If existing arithmetic/metadata evidence cannot justify the
needed range or readiness, keep the precise admission refusal. The design does
not add shape samples or kernel-specific assumptions.

An Acc row extent can also be proved at its original call or submit. A host
loop over `range((extent + tile - 1) // tile)` with `tile > 0` proves
`extent - tile * i >= 1` in its body once the numerator, loop increment and
tile origin fit their original machine types. Clipping the remaining extent
with `min` adds an upper bound. Host shape symbols use the unsigned 32-bit
runtime tensor dimension field widened to INDEX. The compiler joins actual
ranges at every call and through forwarding wrappers. An unknown actual or
an aggregate range outside the Acc capacity leaves the kernel Functional.
When a row bound depends on the caller, lowering recomputes it and guards
the metadata update and store with both row limits. The emitted Buffer
function can then be verified independently of its callers.

Initially reject dynamic accumulator producers/carries, column trims, compact
packing, offset/pitch/layout changes, views or escapes of the runtime alias,
unsupported native consumers and addressless placement. Start with the existing
default two-operand TSTORE variant; atomic, quantization, activation and explicit
phase variants are not newly admitted by this metadata proof. A result that later
feeds matmul, another unknown operation, a carry, return or aggregate needs a
separate contract. Reject unqualified metadata observations anywhere in the
affected physical alias component, including the old static source, attributes
and aggregate references. Inspecting only direct users of the trim is insufficient
because the original Tile setter shares metadata. Preserve the original
Functional body on unsupported selection.

## Existing native evidence

At PTO-ISA pin `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`,
`include/pto/common/arch/memory/tstore_common.hpp::TStoreAccNz2nd` takes runtime
`validRow/validCol` and checks that they match the GM partition dimensions.
With compact mode disabled, its source stride remains `TileData::Rows`,
independent of the trimmed runtime row count. The native row/column field limits
and the writer's established physical packing still apply. With compact mode
normal, stride depends on rounded valid rows; this is why that case is excluded.
`TStore.hpp` requires positive source valid dimensions: empty stores therefore
need a guard, not an invalid native call with `validRow = 0`.

The existing native `pto.set_validshape` interface requires **both** valid
metadata dimensions dynamic. A descriptor `[-1,N]` is not an executable substitute
under that interface. Emit an independently owned alias descriptor `[-1,-1]`,
then set its actual metadata to `(rows,N)` at every use. Distinct runtime trims
with the same typed window can share that dynamic handle in the existing owner;
restore each original alias's actual extents through `RestoreDataOperands` before
every such use. Independence from the static writer is not global independence
between all runtime aliases. The runtime-valid proof
must retain that the second value is the original full width; two `-1` fields
alone do not establish equality, bounds or metadata availability.

This is qualification under the existing native instruction contract, not new
device evidence or a general runtime-layout theorem.

## Implementation owners and validity argument

| Owner | Bounded change |
| ----- | -------------- |
| Shared matrix/metadata recipe | Identify static full Acc writer plus row-only store alias; distinguish physical storage from valid metadata |
| `BufferValidAnalysis` and selection | Admit only the qualified addressed consumer family; preserve early refusal for escapes, dynamic carries and unsupported planners |
| `StorageIndex::FinalizeTypedRoot` | Reuse the existing placed typed-alias owner to retain distinct static writer and dynamic store metadata objects at the same certified address |
| `MatrixDescriptor` and per-use lowering | Produce dynamic-both-dimensions metadata only for qualified aliases; restore each runtime alias's `(rows,N)` before every use and preserve the static producer/carry descriptor |
| Runtime-valid verifier and codegen | Require original ready metadata, full columns, positive store guard and native encoding bounds before emission |
| Effects and synchronization | Preserve the actual store footprint/enclosure, pending M/FIX work, original occurrence identity and selected token protocol |

The storage proof establishes equal byte address, capacity and original physical
layout. It does not manufacture disjointness between the descriptors. Native
metadata objects separate the static writer from the runtime store handle.
Runtime aliases can share their handle, so per-use restoration is essential and
no independence between those aliases is presumed. Unqualified observers of the
original shared metadata are excluded throughout the affected component. The
store's row guard, full-column update and GM partition `[rows,N]` meet the selected
native contract, while its static physical source pitch
still addresses the established accumulator contents.

Memory effects remain real: source completion, store issue/completion, physical
reuse and invocation drain retain their existing owners. A metadata setter
completes no payload. Exact selectors remain available only for their declared
storage domain; conservative effects are not upgraded to exact physical exports.
The adapter changes no allocation algorithm, ID palette, hardware exemption,
expert protocol, default dispatch or zero-inner-body barrier.

## Acceptance gate

- Generic addressed GEMM/Acc fixtures, independent of Qwen names and tilings:
  zero, one, partial and full rows; several static physical shapes and repeats.
- Independently check original store addresses/values and its original guard;
  preserve static matmul initialization, carry and counter-reset identities.
- Use the old static alias before and after the trimmed store and alternate
  multiple runtime row aliases to expose accidental shared metadata mutation.
- Reject columns other than full `N`, invalid/unknown row range, unready bounds,
  compact packing, unknown consumers, dynamic carries/views, escapes and PTOAS
  addressless planning before Buffer lowering commits.
- Check both alias policies, required-order coverage, actual logical/physical
  order, matching, acyclicity and consumption before event republication.
- Produce real PTO/C++ for qualified addressed planners; verify the store
  partition uses `(rows,N)` and both native dynamic metadata fields. Keep the
  static writer descriptor unchanged and preserve the final invocation drain.
- Re-run compact-K/replay, local scalar, runtime-valid and zero-inner-body gates,
  then refresh the existing pinned Qwen census without changing its source.

Representation and construction work must follow static aliases/uses and their
shared metadata DAG, without unfolding runtime rows or visits. Success is a
qualified metadata bridge; it is not a new minimum-demand theorem or a device
speedup claim.

## Implemented checkpoint (2026-11-16)

`BufferAccTrimAnalysis` now owns one immutable, function-scoped capability used by
selection, `BufferValidAnalysis` and `MatrixDescriptor`. It recognizes the
qualified static full writer plus row-only setter, checks the identical complete
storage window again after placement, and excludes unsupported observers and
runtime carries before committing to Buffer lowering. The initial observer gate
is deliberately stronger than component-local exclusion: an unknown Acc metadata
observer anywhere in the function refuses this family.

The selected backend must be configured and its active handler must be A2/A3;
AIC alone is insufficient. Addressless selection remains Functional. Numerical
facts come from original ready scalar SSA, tensor-shape ABI nonnegativity,
constants, min/max and atomic guard comparisons. No arbitrary arithmetic or
benchmark bound is inferred. Counted-loop coordinates are ready only in their
original body, without an inferred sign, count or interval. Native capacity
checks require physical `M <= 8192` and fixed full `N <= 4095`, independently of
allocation capacity.

`FinalizeTypedRoot` continues to own physical storage and typed native handles.
The writer/carry stays static; the store alias receives two dynamic fields.
Existing `RestoreDataOperands` restores `(rows,N)` for each use, including A/B/A
alternation on a shared runtime handle and later uses of the old full source.
The actual Buffer metadata checker independently verifies address ownership,
positive bounded rows, full columns and exact equality with the GM partition.
Conservative analysis, actual-token lowering, ownership and code generation use
that same check. Invalid or unavailable evidence is a refusal, never an empty
access answer. This changes neither allocation nor the logical protocol.

### Bounded evidence and cost repair

A coherent build and the final combined run pass **139 tests in 55.46 s**:
Acc bridge, proof contexts, static row descriptors, runtime row operations,
runtime casts and GM scalar staging. The 17 proof-context cases include A5 and
unconfigured-target refusal. Generic matrix shapes are `(16,32)`, `(32,64)` and
`(64,128)`; rows include zero, one, `M-1` and `M`, with repeated visits/resets,
full static carries at zero/one/three updates and alternating runtime aliases.
Independent interpreters compare payload identities, declared transfer rectangles
in the original physical packing, guard behavior, matching, acyclicity and causal
consumption before republication. Actual noncontiguous IDs `{2,5}` are checked.
Real PTOAS 0.72 C++ is produced for addressed planners, including the default-off
PyPTO route that leaves downstream PTOAS synchronization enabled.

The subsequent combined two-policy gate passes **151 tests in 105.16 s** under
the same 120 s watchdog. Both policies use identical, explicitly disjoint concrete
GM bases. The may-not-alias caller contract excludes overlapping distinct GM roots. These tests do not claim
numerical matmul/device validation or exact burst-level native footprints.
Conservative whole-Acc/DDR summaries preserve the required enclosures.

A new deep shared-guard test exposed an existing receipt-cost traversal:
`ScheduleTreeSize` unfolded an expression DAG and hit the unchanged **30 s**
watchdog. It now memoizes expression-subtree **expanded occurrence weights**,
retains statement/command occurrences, and counts actual inspection work
separately. Small inputs agree with the independent expanded census; depth-28
DAGs complete while retaining their large expanded size. Overflow is explicit
`cost_unproved`: optional complete-candidate selection retains the original
guarded assigned-cut emission, never treats saturated counts as a fit, and does
not throw an internal baseline-size error. Bound addends are widened before
arithmetic. Demands, IDs, protocol and growth budgets are unchanged.

Shared guard proofs likewise extract distinct atomic conjuncts without passing
compound DAGs to the recursive arithmetic simplifier. Range caches are scoped to
lexical proof contexts; repeated contexts can repeat queries, so there is no
claim of globally linear work over arbitrary contexts. Runtime rows and visits
are not unfolded.

### Pins, commands and preserved failures

The companion JSON pins the design baseline, dirty compiler/test sources,
compiled extension, native source snapshots, PTOAS artifacts, environment and
capture command. Capture-stage hashes identify the measured pre-final-hooks
snapshot. The JSON separately records the final formatted source and rebuilt
extension pins, rather than relabelling the measured capture snapshot. Use the Linux environment loader with the pinned
Python/PTOAS launcher; the checkout's stale `env.sh` is not the toolchain source.
The exact combined test command is:

```bash
timeout 120s python -m pytest -q \
  tests/ut/ir/transforms/test_acc_row_trim_bridge.py \
  tests/ut/ir/transforms/test_acc_row_trim_proof_context.py \
  tests/ut/ir/transforms/test_static_row_descriptor_bridge.py \
  tests/ut/ir/transforms/test_runtime_valid_broadcast.py \
  tests/ut/ir/transforms/test_runtime_valid_cast.py \
  tests/ut/ir/transforms/test_gm_scalar_staging.py
```

The initial 29-pass/three-failure run concerned fixtures: an unsupported consumer
changed core classification, constant-zero loops were eliminated before Buffer
selection, and a guard mutator expected a comparison rather than its actual SSA
Boolean. A subsequent 136-pass/one-failure run used fresh deserialization IDs with
an old empty-execution parameter map. These were corrected without changing the
native qualification. Their logs remain, along with the original shared-DAG
watchdog. A separate **10 s** raw-DAG BufferIR verification watchdog remains an
unresolved generic verifier cost issue: SSA/use-after-def checks cannot be fixed
by blindly memoizing uses across scopes. The flag-bearing invocation-completion
rewriter's generic expression traversal is also a separate follow-up, not the
cause of the no-flag receipt timeout and not claimed fixed here.

The first capture command incorrectly invoked Python with timeout options. That
runner error reached no compiler and is retained separately. The corrected run
kept the **120 s** watchdog and generated all **32 C++ leaves**, with **20 PyPTO
owners, 12 delegated leaves and 47 observed task-control owners**, in **6.7544 s
inclusive**. `lm_head` is the added owned leaf; the previous GM-staging checkpoint
was 19/13. Expected-ownership failures are empty. This is one descriptive host
measurement, not a speedup or whole-model/device correctness result.

The original M1–M6 plan remains partially complete. The remaining twelve local
leaves comprise four mixed-valid ADD leaves, `qkpv_plan`'s `tensor.write`,
`rope_kv_cache`'s reshape family, and six existing mixed-peer domains. No peer or
inter-kernel synchronization is generated or replaced. Exact dynamic GM exports,
other arithmetic/readiness forms, broader caller configurations, and native
execution remain distinct obligations.

| Established result | Remaining premise or limitation | Appropriate claim |
| ------------------ | ------------------------------- | ----------------- |
| Static Acc writers and ready row-only store aliases share original bytes with restored metadata | Qualified A2/A3 native store, original access validity and full-column/packing premises | Reusable descriptor/export bridge |
| Independent bounded command checks and existing allocation accept the family | Conservative root ordering; no new minimum-demand, minimum-ID or burst-footprint theorem | Host-model synchronization feasibility |
| Expanded cost weights preserve small census values without DAG unfolding | Raw-DAG generic BufferIR/SSA verification remains unqualified; context-dependent proof costs remain explicit | Bounded diagnostic/candidate-count construction |
| Original Qwen capture gains `lm_head`, reaching 20/32 owned leaves | Twelve leaves and the original M1–M6 programme remain incomplete; no device validation | Incremental production coverage, not complete-model autosynchronization |
