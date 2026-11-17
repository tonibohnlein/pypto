# Runtime-valid row broadcast: use-site metadata evidence

Status: source/design accepted by the independent reviewer; 14 focused tests
pass, including 24 bounded command executions. Integrated build and final
commit gates are recorded in the companion receipt. This extends
M2 (shared operation/metadata qualification), feeding M3, of the general autosync architecture plan. It is a reusable operation and
metadata service, not a Qwen-name admission rule.

## Contract and construction

FP32 `col_expand_mul` admits a static physical dense matrix `[M,N]` with runtime
active rows `r` and fixed positive full columns `N`, a full-valid row carrier
`[1,N]`, and an equally shaped destination. Existing native repeat/stride,
placement and exact-or-disjoint alias restrictions remain enforced. A2/A3 is
the qualified target; the new runtime variant is explicitly unavailable on A5.

A descriptor sentinel is not a relational proof. `BufferRuntimeValidAnalysis`
reads the actual allocation and `buffer.set_validshape` operands, keeps indexed
facts for ordinary allocation metadata owners, and checks source/output row
equality, fixed full columns, carrier state and positive-row control at each
use. It uses existing arithmetic comparison evidence on atomic signed integer
SSA values/constants. Numeric substitution through unchecked arithmetic is not
allowed. An unsupported guard contributes no arithmetic fact in either arm;
its negation must not become a false context proving arbitrary metadata.

This service runs in Buffer verification and therefore in code generation.
Executable ownership additionally consumes it through the shared invocation
qualification service. It does not infer scalar-memory readiness: existing
readiness and visibility contracts remain separate admission obligations.

The logical lowering restores operands' actual valid metadata at each data use,
sets output state, and guards the native broadcast with `r > 0`. On an empty
broadcast visit only its metadata is updated; other payloads such as a carrier
load retain their original control. Branch joins and loop boundaries invalidate
facts by a generation increment. Unknown state stays unavailable until a fresh
owned setter establishes it. Slots, borrowed metadata and arbitrary alias
metadata propagation are outside this slice.

Runtime `0 <= r <= M` is an existing Buffer operation input-validity precondition,
not a newly compiler-proved caller bound. Sound effects cannot repair invalid
metadata or out-of-bounds execution.

## Native validity argument

At pinned pto-isa `5487a6b0`, `TColExpandBinOp.hpp` chooses CountMode when the
matrix descriptor has dynamic valid columns. The source-audited CountMode sets
a count mask to the supplied full `N`, issues VMUL on each active matrix row
using the declared physical row stride and row carrier, then restores the normal
vector mask. It has no hidden scratch or numbered event operation. Thus the
selected active matrix/carrier/result bytes match the stated operation model
under the unchanged layout/placement premises. Source qualification is separate
from native/device execution.

Source and output valid rows refer to the same actual value, or to values proved
equal in the original qualified guard. Therefore each native invocation uses
matching row domains. Positivity prevents an empty native call. Epoch invalidation
cannot create an access-absence claim. These facts establish native metadata
compatibility for the selected variant; they do not establish exact storage
selectors or payload completion at a child exit.

## Independent validation and cost

The focused fixture uses ordinary frontend lowering and explicit conservative
prefix insertion/lowering. It covers active rows 0,1,7,8, taken/skipped branches,
one/several inner visits and repeated outer visits with counter resets. Torch
replay checks active output values and untouched inactive output sentinels.
An independent interpreter reads original scalar values and actual metadata
setters, expands active physical bytes without compiler partitions/generators,
and checks declared conservative effect coverage and concrete hazards.

Actual-token and physical programs are compared for exact payload/control trace
and complete induced payload-event order. Unused immutable completion records
may disappear: they are pure facts, not consuming flags. Physical matching,
acyclicity, six-ID eligibility and causal consumption-before-republication are
checked separately on the emitted program.

Negative cases preserve refusal for unrelated row metadata, missing positive
control, unsupported arithmetic guards in either branch arm, state lost at a
join/backedge and unsupported targets. Additional cases check original guarded
row equality and zero-trip elimination. Literal counts 3,1000000,2000000 retain
the same conservative site/state counters; this checks trip-count compactness,
not an asymptotic bound on arbitrary metadata-expression proof work.

The service performs one statement traversal with indexed handle state and
constant-time generation invalidation. Existing arithmetic query work is
additional. No branch valuations or dynamic visits are expanded.

## Remaining obligations

The unchanged Qwen post-RMSNorm witness additionally needs dynamic reduction and
contiguous `[M,1]`/`[1,M]` valid-axis exports. Its captured IR also adds a full-valid
accumulator to a partial-valid reduction result and returns a full-valid result.
The current slice does not prove inactive reduction output is initialized to
zero and does not establish `r=M` for that caller. The mixed-valid add remains a
separate operation-contract obligation. No kernel, tiling or launch assumption
was inserted to remove it.

| Established evidence | Remaining premise | Appropriate claim |
| -------------------- | ----------------- | ----------------- |
| Actual runtime metadata relations and guarded FP32 broadcast | Valid-range input contract and selected native implementation | Qualified runtime-row operation support |
| Concrete byte coverage and exact logical/physical payload-order agreement on bounded executions | Unknown exact dynamic storage selectors | Conservative synchronization coverage, not minimum physical demands |
| Closed consuming flags and causal reuse within the existing six-ID construction | Separate invocation/runtime and device qualification | Host-model physical feasibility for checked executions |
| Retained size invariant under tested literal trip-count changes | General arithmetic proof and static-IR growth costs | No unfolding proportional to those trip counts |
