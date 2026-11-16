# Verifier expression-DAG traversal: bounded design gate

Status: proposed design; no implementation or new test result. This continues
R4/M6 of the [retained M1–M6 plan](41-general-autosync-architecture-plan.md).
It is separate from the accepted statement-only synchronization rewrite in
[checkpoint49](49-autosync-statement-rewrite.md). That repair preserves unchanged
DAGs; this gate makes their structural verification practical.

## Reproduction and semantic boundary

The existing raw-guard isolation prints `constructed` and
`enter_BufferIR_verify`, then reaches its unchanged 10 s watchdog. The log is
`build-py314/review-repairs-20261116/acc-guard-buffer-verify-isolation.log`.
The input contains a depth28 shared scalar guard with only linearly many distinct
expression nodes. No new performance result is claimed here.

`BufferIRVisitor::VisitExpr` and generic structural verifiers recursively walk
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

Add a small internal verifier utility for one expression-use traversal. It does
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

SSA currently calls `PythonPrint` on the entire function when recording the
first error. That can itself expand a malformed shared DAG. Keep function name,
error category, original source span and concise error text; do not automatically
print the entire function in this verifier. Explicit printing remains available
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
