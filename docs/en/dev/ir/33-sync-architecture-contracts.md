# Autosync architecture contracts

This checkpoint hardens the existing token-to-command pipeline before extending
symbolic or attention-region coverage. The baseline is `0945950aa`. It does not
change demand generation, physical allocation, hardware exemptions, protocols or
production dispatch. Device qualification of the baseline is a separate campaign.

The boundaries are:

```text
qualified accesses → demand analysis → logical completion-token IR
→ shared logical optimization/verification → actual-token interpretation
→ existing allocator → immutable assigned plan
→ qualified command descriptors → selected schedule and scalar-binding plan
→ physical SET/WAIT/barrier IR
```

Logical tokens describe completion facts. Assigned plans describe consuming
resource use. Descriptor and scalar-binding plans describe how to emit that
already chosen protocol; they do not authorize different payload ordering.

## Logical stage ownership

Public `OptimizeSync` and production `RealizeSync` both use `OptimizeLogicalSync`.
The service currently validates without rewriting. Future logical transformations
belong in that service, with generated-output verification before returning a
rewritten function. Production insertion and optimization outputs are checked as
compiler-owned IR: malformed output is a compiler error, not a qualification
refusal that can delegate to PTOAS.

The IR boundary remains immutable, non-consuming completion tokens. `LowerSync`
interprets actual token uses, not a newly reconstructed minimum-demand inventory.
Valid logical IR can be outside the admitted lowering language; that is an
explicit capability refusal. Final consuming hardware IDs are separate resources.

## Qualified command descriptors

Descriptor construction validates coordinate-predicate syntax and physical
synchronization leaves. Production separately binds cached predicate-variable
requirements to the original cut's scope: scalar formals and the dominating
launch lookup are entry facts, and original induction variables are available
only inside their active loops. Entry facts are shared across cuts. Repeated
static anchors intersect their local scopes; equal variable names do not identify
the same coordinate. Arbitrary SSA definitions do not gain readiness merely by
having scalar type.

Diagnostic command fragments explicitly retain their caller-supplied readiness
contract. Their optional scope validation uses the production scope construction.
Unproved arithmetic disables optional sharing; it does not imply a constant or
empty guard. Production rejects such arithmetic when qualifying its commands.
Effectful conditions and payload leaves are not command descriptors.

Retained endpoint domains must have a nonnull frame and matching original owner,
invocation domain, region, loop coordinates and valid half-open ranges. Missing
coverage is not an empty range. Binding a descriptor to a different boundary
renews its predicate requirements. Interval and residue proofs remain separate;
this checkpoint introduces no new symbolic-domain construction.

## Immutable physical assignment

The existing allocator retains mutable families only while qualifying barriers,
reserving lanes and assigning colors. On success, it moves that vector into a
private-constructible `AssignedSyncPlan`. Scheduling and diagnostic readers obtain
const references; there is no second family inventory. `AssignedSyncCase` carries
that immutable assignment separately from mutable output/work counters.

The assignment owns its family records, original IR handles, effective directional
pools and native reservations. Emission validates the original input identity,
owner, target, caller domain and frame/site metadata. The input must remain alive
and unchanged throughout this synchronous allocation/emission transaction. A
retained diagnostic handle is not a serialized certificate authorizing another
input or a later transaction. This boundary does not rerun allocation or change
its sufficient reuse proof, capacity decisions or selected ID tables.

Physical IDs remain target operation attributes. A logical completion token is
still non-consuming and may have multiple uses; it is never an allocated flag.
Command-grouping diagnostics remain explicitly outside physical certification.

## Shared scalar-binding plan

Each command forest retains an immutable scalar-binding plan. The plan records
original predicates, scope-local expression recipes, first-use definitions and
scalar operation counts. Schedule costing weights those counts and uses the
original predicates to partition activity. Emission consumes the same recipes
and assigns fresh SSA names only after schedule selection. Neither consumer
independently rediscovers common expressions.

Branch arms have independent binding scopes. Definitions remain at their first
evaluation; they are not moved to loop entry or outside an enclosing guard.
Short-circuit and otherwise unqualified expressions remain unshared. Optional
structural-comparison failure preserves an unshared valid construction.

Reuse is local to this lowering transaction. A cache lookup compares the complete
ordered immutable root handles, charging those inspections. Changing a forest,
including parent-gap composition, rebuilds its plan. The frozen snapshot contains
plain root handles, avoiding a cycle through its cache owner. Reusing a syntactic
plan never substitutes for the original-cut readiness check. Original-coordinate
materialization still precedes the existing clone/binding operation.

## Production and test-support ownership

Production physical lowering starts from actual tokens through `RealizeDirectSync`
and `EmitAssignedSyncCases`. The old public `RealizePhysicalSync` and
`RealizePhysicalInvocation` mathematical adapters are removed. Supplied-demand
fixture setup now belongs to `python/bindings/modules/testing_sync_support.*`;
it uses the same allocator and emitter and is not an independent correctness
oracle or evidence that actual-token interpretation succeeded.

Raw diagnostic command import also lives in testing support. Diagnostics consume
the production frozen-command materializer; the second recursive statement emitter
has been deleted. Private schedule inspection accepts descriptors, preserving
construction counters rather than reparsing statements. The redundant
`CollectSyncParameterBounds` forwarding function is removed; clients consume the
existing invocation-evidence result.

The retained test adapters support targeted allocator/control tests. Production
coverage remains the actual-token suites plus independent access/order/matching
and physical-reuse oracles. Merely having a test caller does not justify retaining
a separate production algorithm.

The source-cleanup gate passes 363 focused tests plus two targeted counter checks.
All 16 production C++ outputs remain byte-identical, with 235 ordered traces and
nine independent semantic fixtures passing. These checks preserve the prior
protocol; they are not a device result.

## Review and validation

All four stages passed independent review before progression. The focused gates
passed 54, 186, 268 and 188 tests respectively; these overlapping suites are not
an aggregate count of distinct tests. Two final binding-plan smoke tests also
passed. Earlier diagnostic and binding-substitution failures were fixed before
the respective gates closed.

The final campaign compares 16 kernels with the baseline: all generated C++ is
byte-identical, all 235 checked ordered command traces agree, and nine fixtures
pass independent access/order/matching/reuse checks. This includes finite matmul,
vector, rotating, nested, library GEMM and Qwen projection cases. The eight
applicable schedule combinations remain covered by the focused ablation tests.

The [machine-readable receipt](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir/sync-architecture-contracts.json) records source
and tool hashes, commands, results and construction counters. Timing observations
are single runs, not evidence of a speedup. The baseline artifact was recorded
before commit with a dirty source snapshot; its implementation/test hashes were
checked against `0945950aa`. This task changes no allocation policy or emitted
protocol. Host checks do not establish native instruction footprints or a
uniform invocation-return contract. The independent device campaign remains
pinned to its original revision.
