# Contiguous vector identity reshape: bounded design gate

Status: implementation source/test review accepted. The final coherent gates
pass: 4,259 broad tests, 128 post-format tests, three C++ tests and all hooks.
The earlier 20/12 capture is historical; the combined write/identity capture is
21/11, with the additional owner attributed to write staging. Independent final report
review accepted this checkpoint.
This continues M2/M3 of the [original plan](41-general-autosync-architecture-plan.md)
after [checkpoint 46](46-autosync-acc-row-trim.md). It adds no new buffer model,
native geometry, allocation algorithm or synchronization protocol.

## Missing contract

The existing `SupportsLogicalContiguousVectorReshape` recognizes a qualified
FP32 Vec axis map `[L,1] <-> [1,L]` with corresponding active prefixes. It rejects
same-orientation reshapes even when physical shape, layout, pitch, offsets,
dtype and valid metadata are identical. `BufferValidAnalysis` then treats the
reshape as restricted in a function containing runtime metadata. The original
Qwen `rope_kv_cache` exposes this at the explicit `[L,1] -> [L,1]` row-result
reshape. This is an admission/metadata gap, not a new demand-analysis class.

Extend the existing shared predicate only within its already qualified FP32
contiguous-vector geometry. A same-orientation identity must preserve:

- The physical `[1,L]` or `[L,1]` shape, dtype, memory space and orientation.
- Effective layout, fractal, padding and compact mode; zero descriptor offset.
- The existing proved dense physical pitch and complete original byte window.
- Both actual logical valid dimensions, including zero or runtime prefixes.

Reuse `LogicalContiguousVectorLength` and its current geometry/stride checks.
The physical length remains a positive multiple of eight. Physical `[1,1]` is
unavailable under this existing recipe; active length one in a physical
length-eight-or-larger vector is different and remains useful. No arbitrary
2-D identity reshape, dtype reinterpretation, shifted window or new strided
layout is admitted by this extension.

## Owners and validity argument

Keep the existing boolean predicate and distinguish identity from axis exchange
at the domain-transfer use sites. A same-orientation identity preserves the
original prefix axis in both `DomainIndex::Forward` and `OperandDomain`; only
an actual orientation exchange uses the existing axis-map transfer. Inferring a
new axis from an extent equal to one is unnecessary and would be fragile if
physical degeneracies were admitted later.

The geometry predicate proves no view provenance. Selection retains its existing
`!IsWindow` and dense-pitch checks for this qualified route. `BufferValidAnalysis`
must apply the same provenance restriction when deciding that an identity is an
unrestricted runtime-metadata recipe. Its global dynamic/restricted rule must
not accidentally admit a dynamic window through selection's older static-view
fallback. Existing qualified static window forms retain their previous behavior.

Use `StorageIndex`/`FinalizeTypedRoot` and the current relabel lowering. Recheck
the original planned byte window; the identity transfers no payload and needs
no new alias allocation, native instruction, flags or completion cut. Native
metadata restoration remains per use, so an intervening setter cannot erase an
old SSA alias's declared active prefix. Preserve the existing planner-specific
metadata ownership restrictions, rather than claiming a new addressless
reinterpretation export.

The proof is descriptor equality over the admitted original domain: every
active element retains its same physical address and scalar value. The active
prefix axis is unchanged; views with unavailable pitch/provenance are not
classified as identities. This says nothing about payload completion, GM
selectors or physical event reuse; their current services remain responsible.

## Implementation and acceptance sequence

1. Extend the shared contiguous-vector predicate with the strict identity case.
   Audit every consumer, especially active-domain forward/backward transfers.
   Keep the existing true axis-map behavior and native geometry unchanged.
2. Apply the provenance-aware runtime admission consistently. Reuse existing
   placed relabel checks and per-use metadata services; do not emit a copy.
3. Add generic row/column identity fixtures with static/runtime prefixes,
   zero/one/partial/full active extents, skipped branches, old aliases after
   another setter, repeated visits and counter resets. Test multiple supported
   physical lengths and preserve the physical-`L=1` refusal.
4. Independently check declared accesses/active metadata, original byte/value
   correspondence, actual token/flag payload order, matching and reuse. Exercise
   both alias policies on layouts admitted by each contract. Test views,
   nontrivial stride/offset, changed shape/orientation metadata, dtype and
   unsupported consumers as precise refusals. Check actual planner outcomes;
   no blanket addressless permission is inferred.
5. Produce PTOAS C++ where qualification succeeds, run existing vector-axis-map,
   active-domain, scalar, Acc and completion regressions, and refresh the same
   original Qwen capture. Report the next first obligation if removing this
   reshape gap does not qualify the whole leaf.

Review the implementation before accepting and committing this gate. Retained
representation must follow static descriptors/uses and proof contexts, without
unfolding visits or active lengths. The separate raw-DAG verifier and invocation
rewriter costs remain outside this geometry change. M1–M6 and whole-model local
ownership are not complete merely because this identity becomes available.

## Implementation and first validation gate

The existing shared predicate now accepts the strict identity case; it retains
the FP32 vector length/alignment, canonical pitch/zero-offset checks and fixed
inactive dimension of one. True orientation exchange keeps its previous rules.
The normalizer's private `IsVectorReshape` and `ReshapeDomain` preserve the prefix
axis explicitly for identity in both forward and operand-demand transfer.
`BufferValidAnalysis` restricts identity on inherited windows or unknown pitch;
the existing SelectBuffer vector-relabel provenance check is unchanged. Static
view fallback is not promoted into permission for dynamic metadata. No selector
source hunk, new public operation, backend or CMake source is needed.

The first focused module passes **22 tests in 9.11 s**, before strengthening its
no-reshape assertion to inspect both Assign and Eval Call nodes. No-copy checks
already inspected actual payload calls. This is historical first-gate evidence,
not a claim that the strengthened final source has already been rerun.

The generic fixtures cover row and column identities at physical lengths 8/16,
runtime zero/one/partial/full prefixes, static prefixes 1/15/16, old aliases after
an intervening setter, changing prefixes across repeated visits and counter
resets, skipped branches and both alias policies on the same disjoint layouts.
PYPTO and DSA_RP addressed paths produce real PTOAS C++; matrix identities,
inherited windows and physical length one remain refused. Existing vector
geometry premises (dtype, stride and offset) are reused, not broadened.

Torch generation is compared with independently written arithmetic and preserved
output sentinels. Existing byte/control interpreters independently check modeled
required-order coverage, equality of actual token- and flag-induced payload
order, matching, acyclicity and causal event reuse. These are host-model checks;
they are not native/device numerical validation or a new exact minimum-cover
result. This root uses the existing conservative-prefix policy.

For visit counts 3, 257 and 65537, the test compares static Buffer statement and
payload inventories: active lengths and visits are not unfolded. No additional
copy or buffer-reshape Call may appear due to identity. This checks representation
size, not a device speedup or a general complexity theorem for every context.
The strengthened joint gate with checkpoint 47 and the existing GM scalar-read
regressions passes **122 tests in 11.26 s**; its log is
`build-py314/review-repairs-20261116/write-identity-qualified-gate.log`.

The unchanged two-layer Qwen capture `qwen-write-identity-qualified/receipt.json`
generates all 32 local C++ leaves, retaining **20 owned / 12 delegated** and 47
task-control domains. It does not qualify the entire Rope leaf. Its next recorded
refusal is `tile.mul` at `prefill_fwd.py:781`, multiplying a `[16,64]` slice of a
`[16,128]` allocation with a corresponding cos/sin slice. Their declared active
rows are full16 and `group_tok` respectively, so equality is not syntactically
established; source pitch is a subsequent window obligation. This is an
active-metadata/window contract, not the already-repaired column-vector scalar
multiplication. The subsequent scalar-write-only capture is a distinct snapshot:
its 21 owned / 11 delegated result includes checkpoint47's point-write changes,
not an additional whole-leaf ownership gain from identity admission. Source subset hashes
remain explicitly pre-final-format pins; the full capture receipt pins its
compiled shared source/tool context. The original M1–M6 plan and separate
verifier/realization gaps remain intact.

The shared statement-rewrite follow-up is tracked in
[checkpoint49](49-autosync-statement-rewrite.md); generic raw-DAG verification
costs remain separate. Post-format joint gate results are pending.

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
