# Verifier expression-DAG traversal: bounded design gate

Status: independent final source review accepted; build, bounded reproduction
and focused regression gates pass. Final source hooks pass. This continues
R4/M6 of the [retained M1–M6 plan](41-general-autosync-architecture-plan.md).
It is separate from the accepted statement-only synchronization rewrite in
[checkpoint49](49-autosync-statement-rewrite.md). That repair preserves unchanged
DAGs; this gate makes their structural verification practical.

## Reproduction and semantic boundary

The historical raw-guard isolation prints `constructed` and
`enter_BufferIR_verify`, then reaches its unchanged 10 s watchdog. The log is
`build-py314/review-repairs-20261116/acc-guard-buffer-verify-isolation.log`.
The input contains a depth28 shared scalar guard with only linearly many distinct
expression nodes. The identical post-repair probe is recorded below.

Before this repair, `BufferIRVisitor::VisitExpr` and generic structural verifiers
recursively walked
both references to every shared node. The final BufferIR property composes SSA,
strict lexical use-after-definition and assignment-type symmetry; selected
pipeline verification also runs type checking and manual-dependency checks.
`NoNestedCall` is a separate public property, not a direct BufferIR subcheck.

A global expression-pointer cache would be unsound. The same expression can be
valid after a definition but invalid before it, valid in one branch and invalid
in a sibling, or allowed as an operand but forbidden in attribute metadata.
Statement occurrences, defining occurrences, native calls, windows and their
placement-sensitive checks must not be skipped because an expression was seen
elsewhere.

## Construction and exact lifetime

The internal verifier utility handles one expression-use traversal. It does
not change `IRVisitor`, `IRMutator`, token semantics or production dispatch.
Its RAII entry increments traversal depth; depth0 starts a new cache, and the
final exit destroys that use's entries. A cache key contains expression identity
and the client's semantic inspection context. Shared children are visited once
within that context; every new top-level expression use starts afresh.

An expression traversal introduces no statement definitions or control scope.
Thus all nodes cached within one use/context observe the same bindings, native
window table and available metadata. Reentering another statement, branch,
loop header or type-use root cannot reuse that cache. Do not memoize statement
visits. Failure and successful inspection are both bounded within the same
use; acceptance and diagnostic category/span remain unchanged, while duplicate
identical diagnostics from repeated DAG edges may be coalesced.

The bounded client list is:

| Client / entry | Inspection context and retained occurrence checks |
| -------------- | ------------------------------------------------- |
| `BufferIRVisitor::VisitExpr` | Include ordinary versus actual attribute/type/window metadata context. Preserve direct Assign/Eval role and current statement-call identity. Allocation/window/constant state still updates on every original statement. |
| `SSAVerifier::VisitExpr` | Per-use cache, never across `DefineVar`, scope entry/exit, assignment or loop-result checks. Repeated definitions and statement cardinalities remain occurrence checked. |
| `UseAfterDefChecker::VisitExpr` | Per-use cache includes ordinary versus `visiting_type_` mode. Keep strict and legacy scope transitions unchanged. |
| `TypeChecker::VisitExpr` | Keep per-function callee/value provenance state and original valid-shape/type checks; memoize expression decomposition only for this use. |
| `ManualDepsOnSubmitOnlyChecker::VisitExpr` | Calls and Submit ownership are checked again at every new statement/use; no change to permitted dependency fields. |
| Assignment counter / symmetry visitors | Their work is on statement definitions and types. Stop traversing unrelated expression bodies; retain every Assign occurrence and exact existing type comparison. |
| `NoNestedCallVerifier::VisitExpr` | Separate bounded public-property client; preserve each parent-edge call prohibition and its existing SIMT while/declarative predicate exemptions. Its mode cannot leak into a checked body. |

`CollectTypeVars` must not expand shared scalar metadata while registering
bindings. Use a set-producing, context-independent collector local to that
request; it may cache distinct expression/type nodes because the result is a
set of referenced Var identities. Preserve its exact existing metadata-field
policy. For lexical metadata use checking, any type-node deduplication must be
local to that one use, not global across scopes. Do not change the general
`VarDefUseCollector`'s statement occurrence lists or unrelated codegen traversal.
Shared-tuple type validation and generic pretty-print performance receive no
broader complexity claim from this scalar-expression repair.

## Bounded failure diagnostics

SSA previously called `PythonPrint` on the entire function when recording the
first error. That could itself expand a malformed shared DAG. It now preserves
function name, error category, original source span and concise error text,
without automatically printing the entire function in this verifier. Explicit printing remains available
and unchanged. Do not add a new formatter, hide failures, or raise watchdogs.

## Independent acceptance gate

First reproduce valid and invalid depth0/4/28 inputs through the public BufferIR
property and individual SSA/use-after-def properties. Keep the 10 s reproduction
watchdog; root coordinates all serial runs. Check:

- One shared guard used before and after a definition, in both orders.
- A branch-local definition used inside its branch and through the same shared
  expression in a sibling or after the join; nested loop binder/reset scopes.
- Shared scalar expressions in actual type metadata and attributes, including
  forbidden hidden Buffer references; ordinary validity cannot suppress errors.
- Reused direct native-call Expr objects at distinct statements, allocation/view
  provenance changes, repeated assignment to the same Var and duplicate effects.
- Nested calls rejected where required even if the node was previously visited
  in an exempt declarative/SIMT condition; checked loop bodies still reject.
- Existing lexical window-size uses, call placement, use-before-definition,
  type mismatch, malformed nodes and token/physical-program regressions.

Compare valid acceptance and invalid diagnostic category/span with small ordinary
tree equivalents. Large valid DAGs must finish without unfolding; large invalid
DAGs must diagnose without an automatic expanded function dump. No semantic
caller bound, new alias premise, allocation change or hardware exemption is
introduced. Record per-use/context work rather than claiming global linear time
across arbitrarily many uses. Final source review, fresh gates and source/tool
pins precede acceptance and commit. M1–M6 remain tracked independently.

## Independent design gate

Architecture/correctness review accepted this design for implementation. This
acceptance is not an implementation, executable ownership or validation claim.

## Implementation

The verifier-only `expression_use_cache.h` now supplies the RAII per-use/context
cache to the listed clients. BufferIR includes attribute context and direct-call
result role; use-after-def separates ordinary/type inspection. Assignment and
local value collectors retain statement occurrences without walking scalar DAGs.
`VisitTypeExprFieldsOnce` is explicitly opt-in for one lexical type use; the
ordinary enumerator remains unchanged, while set-valued `CollectTypeVars` owns
its request-local distinct-node collector. SSA retains concise diagnostics and
no longer automatically expands the whole function.

`test_verifier_expression_dag.py` adds public valid/invalid shared-depth28,
selected-pipeline, lexical scope/reset, metadata, attributes, duplicate native
statement/definition and nested-call-mode regressions. Tests and probes were run
serially by the coordinating agent; no device execution was performed.

## Fresh qualification before final hooks

The coherent native build passes. The new public-expression module passes
**26 tests in 0.06 s** under a 10 s watchdog (`verifier-dag-first.log`). The
nine-module verifier gate, including those 26 cases, passes **588 tests in
1.45 s** (`verifier-existing-gate.log`). The identical original
`metric_fixture(28)`/public BufferIR probe changes from the unchanged 10 s
watchdog exit124 to **zero diagnostics in 0.0002418477 s**
(`generic-verifier-post-repair.log`). The probe checks representation/proof
admission, not initialized Acc bytes or numerical execution.

The broader continued gate completed **5014 passed / 1 failed**, 76 warnings,
in597.55s (`continued-combined-gate.log`). Its failure exposed a pre-existing
partial-token-erasure boundary; it was not a fresh all-green five-thousand-test
run. After the bounded fix below, the affected scoped-family module passes
**97 tests in5.30s**. These are separate runs, not a fabricated aggregate gate.

## Partial-erasure refusal and fixture correction

The unsupported-wait-cut fixture originally retained Assign leaves but let a
generic mutator remap surrounding capture/loop bindings. It could therefore fail
SSA before testing its intended unsupported cut; the old full-function diagnostic
accidentally satisfied its regex. The fixture now rewrites only statement bodies,
preserving original conditions, bounds, carry/result identities and leaves.
Both the original and edited token programs independently pass the existing
SSA/BufferIR/logical verifier before the unchanged capability-refusal check.

That stronger fixture exposed `structured_token_sync.cpp::Erase`: after stripping
tokens inside an ordinary If, a nonempty candidate caused it to return the
original branch, resurrecting WAITs whose outer carries were erased. A monotone
structural-erasure generation now records token Assigns, WAITs, pure captures,
For token binder/result pairs and Yield token values. A nonempty branch whose
generation changed refuses before payload import with the precise unsupported
guarded logical-token-cut obligation. Empty token branches still erase; unchanged
native branches remain intact. Two valid guarded Scalar/payload loops with only
completed-token carries protect the binder/Yield paths. This is a conservative
refusal-boundary repair, not a new planner, protocol or admitted synchronization
language. The original IR remains unchanged on refusal.

## Original-model checkpoint and pins

Fresh `qwen-verifier-layout/receipt.json` retains **21 owned / 11 delegated local
leaves**,47 task-control domains and all32 C++ outputs, byte-identical to the
previous capture. Its9.59558s capture time is descriptive, not a measured compile
speedup. This repair adds no kernel coverage or payload overlap claim.

The receipt pins the current source subset (including erasure/fixture repairs),
native extension and Python against parent
`41a0c33870f4f955b4a2e87c26e38c711785509f`. The binary also contains checkpoint51's
first shared-layout substrate; it does not qualify51transport contexts/capsules.
Final hooks may refresh pins. No production dispatch, demand construction,
allocation algorithm, hardware exemption or expert peer protocol changed.

The cost result remains per expression-use/context and statement occurrence.
Structural type comparison, shape validation, other dimension/type diagnostic
printers and arbitrary shared-Tuple paths are not globally repaired. M1–M6 and
remaining native/visibility/transport obligations remain open as recorded in the
original plan.

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
