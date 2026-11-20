# GM scalar writes through the existing private carrier

Status: implementation source accepted; final coherent build, 4,259-test broad
gate, 128-test post-format gate, three C++ tests, configured hooks and eight-case
actual-planner audit pass. Independent final report review accepted this checkpoint. Native
qualification limits remain explicit; no device result is claimed.
This extends the [GM scalar-read recipe](44-autosync-gm-scalar-staging.md)
and [local scalar contract](43-autosync-value-and-scalar-contracts.md).
It does not add a backend, public pass, allocator or external visibility rule.
The [machine-readable checkpoint](47-autosync-gm-scalar-write-staging.json)
records the preserved failed runs and final source/tool identities. No device execution
has been performed.

## Packed rank-two point writes

The explicit DMA option also handles packed, fully valid, unpadded ND INT32
and FP32 rank-two tensors under the same A2/A3 AIV and native row-pitch
constraints as point reads. The store preserves the original GM allocation
and both coordinates. Its `[1, 1]` partition selects one four-byte burst in
the ND b32 TSTORE path even at a column that is unaligned to 32 bytes. Local
effects cover the complete private carrier. The GM write still touches just
the selected element. Scalar D-cache publication and peer readiness remain
separate contracts. [GM point-read geometry](44-autosync-gm-scalar-staging.md)

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

`StageBufferScalarAccesses` now handles original direct assigned reads and both
Eval/assigned writes in the same immutable traversal. Directional transfer
qualification is shared by Buffer type deduction, native qualification and
codegen; the existing lowerer emits a rank-one store extent `[1]` alongside the
rank-two `[1,8]` carrier. No exact rank-one GM selector capability is granted.
The original default-off Functional path still uses scalar `pto.store`.

The write stager admits only an original destination Var/IterArg and one index
Var/IterArg or integer literal. The stored INT32 value may be a scalar-typed DAG
of Var/IterArg, integer literals, Cast, BinaryExpr and UnaryExpr nodes, provided
every child qualifies. An iterative, stage-owned memo records positive and
negative answers once per distinct node. Calls, Submit, aggregate/projection and
unknown nodes refuse even beneath casts; Var/IterArg nodes are leaves, without
inspecting their definitions or replaying carry initializers.

The original AST is passed unchanged to the local scalar write at the same
executed cut. No cast is removed, arithmetic simplified, range assumed or machine
width changed. This qualification proves syntactic purity only; original
operation definedness, lexical readiness and native scalar support remain
separate obligations. It admits generic values such as
`Cast(Add(Cast(original_value, INDEX), 1), INT32)` without a workload-specific
rule. The original 122-test gate predates this extension. A fresh 125-test gate now
passes, including inline arithmetic, INT32 extrema and hidden effects beneath
Cast. The first pure-value build failed because its helper was placed in the
name collector instead of the Stage that owns the memo; the helper was moved
without semantic changes, and the failed build log is preserved.

The registry now declares `tensor.write`'s existing physical result reuse of
argument zero. Its `ArgEffect::Write` and scalar `WriteChannel` are unchanged.
`InitMemRef` consequently shares the existing destination window instead of
creating a fresh result root. Distributed window metadata is retained; this
change neither allocates a missing runtime window nor treats collective lineage
as physical reuse. The redundant `tensor.write` case in the general alias table
is removed, while collective cases remain intact.

Operation result types may precede placement: the stager permits a Call/result
with absent MemRef only through the registered destination alias, after exact
logical geometry, layout and dtype comparison. Every explicitly supplied
MemRef must preserve destination base identity and the complete window. Foreign
roots, offsets, strides or types still refuse; absence of placement is not a
license to overwrite contradictory provenance.

A qualified local scalar write can now satisfy the selector's existing
source-producer condition alongside load/fill. This admits pure point-write
leaves whose sole valid carrier lane is initialized. It does **not** prove that
the remaining physical carrier or an arbitrary output buffer is initialized.

A default-off assigned-write C++ regression also exposed a pre-existing view
binding error: scalar `pto.store` correctly used the raw GM pointer, but its
assigned Tensor result registered that pointer as a tensor view. The repair
binds the result to the destination's existing canonical tensor view, as normal
Tile stores do. Raw store pointer/index/value, base-pointer provenance and
communication context remain unchanged. No view is fabricated and no If merge
check is weakened; an unavailable canonical view keeps its normal refusal.

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

## Validation checkpoint and preserved failures

The first passing three-module gate recorded **122 tests in 11.26 s**, before
the pure-value extension. It includes the
write family, independent identity-reshape changes and existing scalar reads.
While final-false/repeated visits and default-off assigned branch/loop real C++
now pass. Earlier runs remain in the JSON receipt rather than being replaced:

| Run | Outcome | Issue exposed |
| --- | ------- | ------------- |
| Initial write-family run | 10 passed, 24 failed | Unplaced operation-result types were incorrectly compared with placed destination types. |
| After logical-type repair | 22 passed, 14 failed | Assigned `tensor.write` lacked its physical-reuse declaration; foreign-root rejection correctly prevented staging. |
| Joint write/identity run | 63 passed, 4 failed | Fixture/API expectations required correction; original negative provenance obligations remain. |
| Corrected combined run | 119 passed, 1 failed | Registered write-result placement fixture still needed correction. |
| While/default-off combined run | 121 passed, 1 failed | Actual Functional C++ path exposed raw-pointer versus tensor-view result binding. |

The original passing receipt is `write-identity-qualified-gate.log` (122 tests,
11.26 s). After the pure-value extension, `write-pure-value-qualified-gate.log`
passes **125 tests in 11.58 s** across write, identity-reshape and scalar-read
modules. Neither number is a write-family-only count. The JSON retains the
failed pure-value build `gm-write-pure-value-build.log` and the historical
pre-format source checkpoint. These tests precede the final formatted build;
source hooks, fresh review and accepted commit remain separate gates.

## Original-model capture: coverage with substantial command cost

`qwen-pure-write-qualified/receipt.json` records successful C++ generation for
all **32** original leaves in **9.973 s inclusive**, with **21 PyPTO owners,
11 delegates and 47 observed task-control domains**. Required ownership checks
report no failure. This is one descriptive host measurement, not a speedup or
analysis-only timing. The preceding `qwen-write-identity-qualified` capture had
20 owners: all remain, and the generic pure-value write recipe adds `qkpv_plan`.
The 47 task domains are observations, not newly proved external owners.

The added owner has a substantial static protocol:

| Published `qkpv_plan` operations | Static sites |
| -------------------------------- | -----------: |
| SET / WAIT | 8,450 / 8,450 |
| Local scalar write / GM store | 769 / 769 |
| GM load / local scalar read | 384 / 384 |
| MTE2 / MTE3 barriers | 383 / 768 |
| Final ALL | 1 |

These counts describe the emitted candidate after the source's existing
construction, not dynamic execution cost. They are a coverage result with
significant control growth, not evidence of minimum demands, minimum IDs or a
performance improvement. Its metadata reports whole-memory-space enclosures and
preservation of the selected token order. The bounded audit below checks actual planner output separately from fixture
correctness and C++ generation; it does not establish a symbolic whole-model
command theorem.

`qwen-point-plan-order-audit/receipt.json` records **eight passing bounded
cases** at `finalize_tok` 1, 8, 9, 16, 17, 127, 128 and 0. The zero case is
explicit structural stress outside the positive caller domain. The 9 case
replays two invocations; the others execute once. The actual-command interpreter
uses independently extracted staged byte accesses, native-pipe/SSA/control
order and required invocation completion, with the private 32-byte UB read
enclosure. Physical reuse edges are checked against existing causal order,
not inserted to make the proof succeed.

| Executed audit | One invocation | Two-invocation replay |
| -------------- | -------------: | --------------------: |
| Payloads | 770 | 1,540 |
| SET / WAIT | 2,818 / 2,818 | 5,636 / 5,636 |
| Causal reuse checks | 2,806 | 5,624 |
| Byte-conflict checks | 230,401 | 922,246 |
| Required / physical completion-to-start pairs | 296,065 / 296,065 | 1,185,030 / 1,185,030 |
| Missing / excess pairs | 0 / 0 | 0 / 0 |

Here 296,065 = 770 × 769 / 2 and 1,185,030 = 1,540 × 1,539 / 2:
the staged shared-carrier byte reference already requires lexical total payload
order. Zero excess therefore does not demonstrate retained overlap, minimum
command count or an improvement over the original scalar program. Reducing the
substantial command cost remains follow-up work, not an optimization established
by this increment.

The bounded runs also check matching, acyclicity, eligible IDs, modeled values
and terminal completion. The source's batch-one active work is already a
prefix, so outputs are 0–127: these cases exercise different branch/readiness
paths, not a nonidentity permutation. Order equality is relative to the staged
MTE2/S/MTE3 byte model, **not** the original scalar-D-cache order, a minimum-cover
result or a native theorem. The receipt pins its own runtime extension and
inputs; final formatted-source/tool pins remain a separate gate.

M1–M6 remain partial. This package provides no new peer protocol, allocator,
cutoff, exact GM selector, complete model ownership or device result. Its native
premise remains the existing normative selected TSTORE contract, with private
CCE/public-API correspondence explicitly unverified.

## Final coherent qualification (2026-11-16)

The final source is accepted by independent review. The fresh serial broad gate
passes **4,259 tests** with seven warnings and zero failures in 560.27 s.
After formatting/type-check corrections, the coherent build, **128 focused tests**
(11.32 s under the unchanged 120 s watchdog), all **three C++ tests**, and all
configured hooks pass. The command manifests and log hashes are recorded in the
companion JSON's `final_verification`; it pins dirty production sources, loaded
extension, Python and PTOAS separately from the parent commit.

The fresh original-model capture again generates 32 C++ leaves, with 21 local
PyPTO owners, 11 delegates and 47 existing task-control owners observed. Its
strict expected-name checks pass. The final eight-case actual-command audit
also passes against the declared staged-byte reference, including replay without
resetting physical-ID history. Shared-carrier conflicts already require total
payload order in those executions; zero excess gives no overlap improvement or
minimum-command claim. Static protocol cost remains explicit in report47.

The initial refresh runner used a file invocation despite package-relative
imports and failed before compilation. The corrected `-m` command, original
input and unchanged watchdog pass; both receipts are retained. Generic verifier
DAG scaling, local transport capsules, unresolved mixed-valid semantics and the
remaining original M1–M6 obligations are separate work. No device validation or
complete-model local ownership is claimed.
