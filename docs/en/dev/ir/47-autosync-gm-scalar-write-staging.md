# GM scalar writes through the existing private carrier

Status: **design accepted with the selected-TSTORE qualification below;
implementation and validation pending**.
This extends the [GM scalar-read recipe](44-autosync-gm-scalar-staging.md)
and [local scalar contract](43-autosync-value-and-scalar-contracts.md).
It does not add a backend, public pass, allocator or external visibility rule.
The [machine-readable checkpoint](47-autosync-gm-scalar-write-staging.json)
records the outstanding gates; there are no new test or device results.

## One original write, one executed transfer

Admit an ordinary A2/A3 AIV `tensor.write(destination, [index], value)` on a
packed, unpadded ND rank-one INT32 GM tensor. The original index and INT32 value
must be ready at that statement. Preserve the original in-bounds, nonwrapping
access precondition; neither benchmark bounds nor pointer names prove it.
Initially refuse non-default kwargs/attributes, peer/distributed tensors,
atomic or publication variants, other dtypes/cores and nested effectful calls
without an existing statement cut.

```text
before, either statement form:
  tensor.write(destination, [index], value)
  updated = tensor.write(destination, [index], value)

after, at that same executed cut:
  local = tile.write(private_carrier, [0,0], original_value)
  tile.store(local, [original_index], original_destination)  # one GM element
  updated = original_destination                           # assigned form only
```

This is semantic pseudocode, not new public syntax. The normal Tile/Buffer
lowerer supplies the transfer's explicit one-element extent. Reuse the existing
32-byte Vec carrier, physical INT32 shape `[1,8]`, valid shape `[1,1]`, ordinary
allocation/placement and constant local point access. Do not load the old GM
value or initialize eight GM elements. The local S write establishes the sole
valid scalar; the MTE3 transfer must write exactly its four destination bytes.
No result or effect may observe the carrier's other seven lanes.

The assigned result aliases the original destination, as required by
`ResultAliasedArgIndex`; it is neither a copy nor a new storage root. Preserve
its original SSA name/type and subsequent loop/branch alias use. The discarded
form must still execute both side effects exactly once. Preserve original
For/While frames, skips, resets and statement order. Carrier allocation may live
at the ordinary function allocation cut, but its write/store may not move out of
an executed branch or loop.

## Completion and memory premises

The existing local `buffer.write` contract supplies a blocking typed UB write
on S. Existing prefix services must order that write before the MTE3 read and
order the MTE3 read before the next write to the same carrier. They must also
order the GM write against subsequent overlapping local reads/writes and retain
the normal invocation completion barrier. Closed SET/WAIT matching alone is not
payload completion. Repeated visits reuse the same private carrier only under
these actual dependencies; no trip-count expansion or new flag pool is needed.

Two native facts need a write-specific source review before admission:

1. The selected A2/A3 INT32 Vec-to-GM TSTORE path supports valid `[1,1]`, a
   32-byte-aligned UB carrier and an element-aligned GM destination, including
   offsets 1 and 7 that are not 32-byte aligned. Its GM write length is four
   bytes, with no read-modify-write or overwrite of adjacent GM elements.
2. The existing SetValue/UB and MTE3 synchronization contract makes the written
   scalar available to that transfer. Any carrier read enclosure may cover 32
   bytes, but padding must not influence destination values or bytes. A sound
   overapproximation of local reads does not authorize wider GM writes.

The local pinned snapshots already identify the directional construction:
`tstore_common.hpp:366–394`, `TStoreUb2gmNd2nd`, gives `nBurst=1` and
`lenBurst=1*sizeof(INT32)=4` for the unit GM view; `TStore.hpp:19–28`
selects `copy_ubuf_to_gm_align_b32`. The JSON records both snapshot hashes.
This establishes selected byte-count construction under the normative TSTORE
contract. It does not independently reproduce the private CCE intrinsic's
correspondence with the public nonaligned-copy API.

Bind the remaining facts to that selected native implementation and the pinned
PTO-ISA source used in design44. The GM-to-UB TLOAD evidence is **not** evidence
for the reverse direction. Compilation or poisoned-memory host tests support
construction checks; they do not independently prove hardware cache behavior.
Missing a justified selected-TSTORE contract keeps the new recipe unavailable.
The unverified private CCE/public-API mapping remains an explicit native
qualification under the existing normative TSTORE premise, as in design44;
it is not by itself a requirement to reject that already qualified contract.

This construction does not establish peer initialization, atomicity, cross-core
visibility, publication ordering, inter-kernel readiness or deterministic races.
Do not convert a peer protocol into ordinary private GM traffic. The existing
pre-selection rejection of mixed scalar-D-cache and MTE3 writes to one GM root
remains: accepting a complete alternative for that case would require a separate
whole-root channel proof. A late-selection-only increment retains that rejection.
If the implementation admits an originally mixed root, the same immutable stager
must first construct a complete qualified candidate before
`UpgradeWrittenTensorParamDirections` classifies channels. Classify its actual
Tile stores and derive directions from their original GM destinations; never
merely label an original `tensor.write` as MTE3. An unstaged scalar access or
unproved cache/publication interaction retains ordinary rejection. The original
function remains the rollback owner through later whole-function admission.
The early candidate placement and its rollback lifetime require review before
such mixed-root support is claimed. This accepted increment uses late-selection
staging only; early candidate relocation is not approved here.

## Extend the existing owners in order

| Existing owner | Required bounded change |
| -------------- | ----------------------- |
| `buffer_scalar_recipes.{h,cpp}` | Share packed rank-one geometry; add directional store qualification for carrier/source, original GM destination and one-element extent. Never call the load-only proof with reordered operands. |
| `buffer_scalar_staging.{h,cpp}` | Extend the current read stager to direct assigned and Eval writes; reuse names, allocations and immutable body traversal. Preserve assigned destination aliases and leave unsupported candidates unchanged. |
| `SelectBufferIRFunctions` | Keep the existing enable-Buffer + enable-autosync + `conservative_prefix` + A2/A3 AIV gate. Publish only after ordinary whole-function admission; on any later failure return the original function. |
| Buffer memory deduction, Tile lowering and PTO Buffer codegen | Consume the same qualified store geometry, preserving rank-one ABI, original index, root, output direction and exact extent. Both addressed planners and delegated PTOAS placement need the normal path. |
| `buffer_sync_contract.cpp` and existing effect import | Retain S carrier-write and MTE3 carrier-read/GM-write effects, local readiness and reuse; use the existing operation registry, with no parallel effects table. |
| `gm_access.cpp` channel/direction analysis | Preserve ordinary scalar/MTE3 refusal. Any early staged candidate must be complete, retain original rollback ownership, and be classified from its actual payloads; no permissive channel override. |
| DCE/result-alias clients | Preserve observable Eval writes and assigned aliases using the current operation contracts; do not classify the rewritten side effects as an unused pure result. |

The current source has a read-only `StageBufferScalarReads`, skips Eval nodes,
and admits rank-one transfers only when `load` is true in native/codegen checks.
The original `tensor.write` registry declares scalar-channel writes and its
Functional codegen emits `pto.store`. Extend the existing staging owner (rename
it to reflect both directions if useful); leave the original operation contract
and default-off Functional path intact. No public configuration is added.

Construction should visit original statement occurrences and distinct expression/
type nodes, retaining one carrier and a constant number of operations per staged
site. Shared scalar-expression DAGs must remain shared. Runtime indices, visits
and reset counts do not enlarge the representation. These bounds concern the
stager, not the complete synchronization/placement pipeline.

## Acceptance gates before claiming the family supported

1. **Native contract first:** pin the selected TSTORE path, byte length,
   alignment, local source-read enclosure and S/MTE3 ordering. Record remaining
   qualifications separately from compiler tests.
2. **Independent values and bytes:** Eval and assigned writes at indices
   0/1/7/8/final, tensor lengths 1 and 9, INT32 extrema, symbolic lengths, both
   alias policies and permitted overlapping roots. Poison every adjacent GM
   byte and carrier padding; compare original destination bytes and untouched
   neighbors, not just printed transfer dimensions.
3. **Executed control:** zero/one/multiple visits, conditional skips,
   read-modify-write scalar recurrences, repeated writes, outer resets and
   final-false While behavior. Preserve original root/index/value identities and
   return aliases. Nested/unsupported effects must refuse transactionally.
4. **Actual commands:** independently check required memory order, matching,
   acyclicity, noncontiguous eligible IDs, consumption before republication and
   final completion. An edited missing handshake must fail the oracle.
5. **Public paths:** Select/LowerBuffer, Insert/Optimize/LowerSync and real PTOAS
   C++ for PYPTO/DSA/PTOAS planners; explicit policy only. Autosync-off,
   Buffer-disabled, exact-only and unsupported/native-metadata variants retain
   their original admission behavior. Keep mixed-channel refusal tests.
6. **Compactness and integration:** count sites/carriers/commands across runtime
   trip values without unrolling; rerun the scalar-read/local-scalar suites and
   only then the original model capture. A newly reached obligation is not
   complete model ownership, a minimum-demand claim or a device speedup.
