# Statement-only synchronization rewriting

Status: design and final source review accepted. The corrected build, unchanged
three-depth public regression, 4,259-test broad gate, 128-test post-format gate,
three C++ tests and all configured hooks pass. Independent final report review accepted this checkpoint. This is the bounded R4 traversal follow-up in the
[original M1–M6 plan](41-general-autosync-architecture-plan.md), separate from
[vector identity admission](48-autosync-vector-identity-reshape.md).

## Failure and owner

`CompleteOwnedInvocation` only inserts the already-qualified terminal completion
at return statements. The closed-prefix emitter only rewrites statement cuts.
Both previously inherited generic `IRMutator` container hooks. An override of
`VisitExpr` did not protect shared expression DAGs: typed hooks such as the
`IfStmt` mutator directly dispatched the condition. Function traversal also
revisited signatures and metadata unrelated to either transformation.

The public owned-return regression completed depths 0 and 4, then hit its
unchanged **30 s watchdog** at shared-conjunction depth 28. The earlier internal
isolation hit a 12 s watchdog. These are historical failures, not successful
scaling measurements; their logs remain under
`build-py314/review-repairs-20261116/`. No watchdog was increased to claim repair.

## Shared construction and validity

`sync_statement_rewrite.h` now owns `SyncStatementMutator` for already-qualified
synchronization owners. It explicitly handles the supported statement containers
and preserves original expressions by identity. For branches it rewrites bodies
while retaining the condition and return binders. For loops and scopes it copies
the concrete statement subtype only when its body changes, retaining all original
bounds, binders, attributes, control and scope evidence. Assign, Eval, Yield and
Return leaves remain unchanged unless the derived owner replaces that cut.
Unknown containers are preserved; this helper does not admit their semantics.
The existing preflight and owner-specific refusal boundaries remain responsible
for qualifying the input.

Both closed-prefix emission and owned-return completion reuse this helper.
They shallow-copy the function and rewrite only its body, preserving signatures,
parameter identities and types. `RewriteSyncSequence` retains its existing
single occurrence walk/flattening behavior. Statement occurrences are not
memoized: two executions represented by two statement occurrences must still be
rewritten separately. Unchanged scalar/type DAGs are not traversed or copied.

Correctness follows from the rewrite boundary: preserved payload statements,
guards, frame expressions and metadata have identical operands and control;
existing derived hooks insert the same assigned commands or completion at the
same cuts. The change introduces no new synchronization, allocation, hardware
exemption, native operation or caller premise. It does not alter the meaning of
`sync.record`, immutable tokens, physical matching or event-ID reuse.

## Qualification and remaining limits

`test_invocation_completion_rewrite.py` exercises public flag-bearing `LowerSync`
at shared-guard depths 0/4/28, checks original parameters and guard identity, and
checks actual consuming notifications plus terminal ALL before return. Its zero
row selects a false guard and skips the payload. This tests rewrite work and
completion, not initialized Acc bytes, numeric matmul or execution of all leaves
of an expanded Boolean tree.

The bounded target is statement occurrence work plus inserted statements for
these two owners; no general complexity theorem is claimed for every pass.
Generic raw-DAG BufferIR/SSA verification and the expression interpreter remain
separate cost boundaries. Independent actual-command tests still own matching,
acyclicity, consumption before republication and payload-order coverage.

The new point-planner checker uses the existing indexed test DAG representation;
it is not a compiler algorithm change. Its ordering reference is the staged
MTE2/S/MTE3 program, scalar SSA/control readiness and independently decoded byte
conflicts, with the private 32-byte UB enclosure and explicit invocation-completion
requirements. It is not equivalence to the original scalar-cache implementation,
a minimum-demand theorem, or native/device qualification.

Fresh build/test results and final source/tool hashes will be appended after the
review gate. The earlier successful eight-case point-planner receipt remains
historical; the added ordering diagnostic requires a new run. This checkpoint
does not complete M1–M6 or claim full-model PyPTO ownership.

## Accepted observations before final formatting

The corrected build passes. Under the original 30 s watchdog, depths 0/4/28
all pass: **3 tests in 0.07 s**, recorded in `shared-body-rewrite-gate.log`.
The independently reviewed extension to `qualify_qwen_point_planner.py` passes
seven positive caller cases and separately labeled zero stress, including a
second invocation at finalize9. In `qwen-point-plan-order-audit/receipt.json`,
the staged-model reference and physical program both contain **296065** ordered
completion-to-start pairs for a single invocation, and **1185030** for the replay.
All eight cases have **zero missing and zero excess pairs** relative to this
stated reference. These counts are `770*769/2` and `1540*1539/2`: reuse of the
shared staging carrier serializes all staged payload pairs in this model. This
equality therefore establishes no overlap benefit. These are host diagnostics for the selected staged model,
not minimum-demand or original-scalar/native claims. The checker source was
subsequently refactored for lint without changing its checks; final formatted
source hashes and a fresh run remain pending.

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
