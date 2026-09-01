# DSA Reuse-Penalty Modeling

## Status

PyPTO's reuse-penalty recognizer is experimental and disabled by default. This
document separates three concerns that must not be conflated:

1. the stable DSA-RP optimization problem;
2. the recognizer and promotion policy currently implemented in PyPTO; and
3. the evidence used to decide which recognized candidates deserve a positive
   weight.

The device campaigns, including blocked and superseded runs, are indexed in
[DSA-RP Device Experiment Ledger](dsa_device_experiment_ledger.md).

The evidence supports the hard-constraint part of the optimization model, but
the latest legal ablation shows that a relation's measured marginal can be
negative. The current non-negative solver objective can conservatively clip
such relations to zero, but cannot actively seek the beneficial overlap. The
evidence does not support enabling the current promotion policy in production.

## Stable optimization problem

Each physical memory space is an independent standard DSA arena. Capacity and
correctness constraints are hard. A sparse soft edge
`e = (buffer_i, buffer_j, weight_e)` is active when the placed byte ranges of
the two lifetime-compatible buffers overlap:

```text
reuse_cost(p) =
    sum(weight_e for active overlapping edges e)
```

The planner finds a valid placement within capacity, minimizes
`reuse_cost`, and may minimize peak as a final tie-break. Weights are
non-negative. An absent edge or weight zero means that the compiler has no
evidence of performance damage; it does not mean that overlap is guaranteed
free.

The model deliberately contains only buffer pairs and weights. Recognition,
PTOAS synchronization behavior, and weight estimation are producer-side
questions. Pipeline-intent hard constraints and their explicit soft fallback
are orthogonal to the access-hazard recognizer discussed here.

## Implemented recognizer

`DsaReusePenaltyRecognizer.QUADRATIC` is the only enabled research mode. It is
off by default and favors coverage over compile time.

### Candidate generation

The recognizer:

1. normalizes semantic aliases into physical allocation identities;
2. collects execution-time reads and writes from each operator's authoritative
   `ArgEffect` declarations and SSA results, including tuple results, mutating
   operations, base allocations, and known byte ranges;
3. maps each access to an abstract source/destination route and execution
   resource;
4. constructs terminal-access and initial-write frontiers per allocation;
5. scans every lifetime-compatible buffer pair in one address space; and
6. records distance-zero and distance-one WAR/WAW handoffs with route, range,
   loop, control-flow, and ordering provenance.

The legacy recognizer population remains exported in
`metadata.recognized_reuse_candidate_records_v4`. The expanded v5 population
also includes pipeline-serialization provenance and is counted separately by
`recognized_reuse_candidates_v5`. An SSA-reachable record has a deterministic
region/statement `dag_path`; an unordered record uses `dag_path=none`. The command

```bash
python -m pypto.tools.dsa_reuse_candidates PROBLEM.dsa.json
```

parses these pre-policy records.

### Current v5 construction and weight policy

The current `cross_resource_completion_pair_v5` policy constructs one pair edge when at
least one record for the pair is:

- cross-resource;
- full-allocation and completely observed;
- not dependent on a conservative initial anchor;
- not a same-operation alias-contract question.

SSA-ordered and loop-carried records remain eligible. Device experiments
refuted SSA reachability as a completion proof and identified a costly
distance-one handoff. Same-resource, partial-view, and uncertain records remain
report-only. Multiple qualifying records for one buffer pair produce one edge.
`unit_v1` then assigns cost `1` to every constructed edge.

This produces an additive, non-negative `cross_pipe` model of potential
synchronization obligations. It does not yet decide whether an obligation
extends the consumer's effective completion frontier or is exposed on the
critical path; metadata records that limitation as
`reuse_penalty_completion_exposure_model=unmodeled_v1`. The recognizer remains
experimental and disabled by default.

The implementation uses same-resource issue order and SSA reachability while
constructing per-allocation access frontiers. Same-resource ordering is an
abstract completion-chain assumption. SSA reachability is exported only as
provenance for cross-resource candidates, not as a suppression rule.

## Evidence and rejected rules

Controlled placements hold the DSA problem and generated operations fixed and
change only selected physical overlaps. The accumulated results establish:

- Most legal reuse is synchronization-neutral.
- Synchronization-group count does not predict latency.
- Route class alone is insufficient: the same route can be harmful, neutral,
  or covered by another release.
- Loop frequency amplifies an exposed handoff, but cannot make a covered
  handoff costly.
- Several predecessors of one consumer behave approximately like the latest
  active predecessor, not like an additive count.
- A controlled Gumbel ablation now justifies a negative *relation marginal*:
  restoring `(2,39)` improves latency by 2.1-2.35% on both devices. It also
  removes one static ELSE-arm barrier, but a later branch-profiled validation
  shows that most of the benefit occurs in THEN blocks where that barrier does
  not execute. The relation effect is causal; the barrier mechanism is not yet
  causal evidence.

The exact ordered-pair study added two important counterexamples to the
previous v4 policy:

| Pair class | Result |
| ---------- | ------ |
| SSA-ordered `V -> MTE2` WAR | overlap added one matching PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap removed a redundant handoff, with no confirmed latency effect |
| four other matched pairs | synchronization unchanged |

Therefore v5:

- retains SSA `dag_path` as provenance rather than a suppression predicate;
- includes distance-one candidates rather than suppressing all loop-carried
  handoffs;
- still treats every constructed edge as an uncalibrated unit obligation.

The experiments also show that promoting every cross-resource obligation with
the same positive performance weight is too broad:

- A positive weight for every synchronization-changing pair is unjustified.
- The non-negative pair model remains a conservative approximation: neutral or
  beneficial pairs can receive no positive edge, but that approximation cannot
  prefer a placement whose synchronization interaction is beneficial.
- Several whole-kernel RP placements are reproducibly faster, including UB and
  L1 cases, but the largest gains are not ranked by the number of inserted
  synchronization groups.
- Clean pair ablations can reproduce substantial latency changes while their
  changed synchronization summaries point in different directions. Candidate
  recognition is therefore ahead of mechanism attribution and weight
  calibration.
- A pair's effect can depend on the surrounding placement. Pair isolation must
  preserve capacity and account for every overlap relation changed by the
  construction.

## Current completion-frontier conjecture

Three forms of ordering must remain distinct:

| Layer | Question |
| ----- | -------- |
| Logical ordering | Can one PyPTO value-producing operation reach another? |
| Completion ordering | Has the prior asynchronous access released the reused bytes before the overwrite? |
| Exposed delay | Does extending that release point change initiation interval or kernel latency? |

For lifetime-compatible logical buffers `A != B`, let `u` be the last relevant
access to the reused subrange of `A`, and `v` the first overwrite of the same
subrange by `B`. Reuse creates a candidate physical WAR/WAW handoff `u -> v`.

The next recognizer policy should suppress the candidate only when a
**completion-carrying path** already orders `u` before `v`. Examples include an
explicit event/barrier/token or a target-guaranteed FIFO completion relation;
ordinary SSA reachability is insufficient. The v5 constructor therefore keeps
such candidates, while its unit weight remains an experimental upper-level
surrogate rather than a calibrated latency claim.

A positive edge is indicated only when `u` extends `v`'s existing
completion-release frontier. A qualitative cost conjecture is:

```text
dynamic frequency
    * max(0, completion(u) - latest unavoidable release of v)
```

This expression is a modeling guide, not a cycle estimator currently available
to PyPTO. The checked-in v5 unit model deliberately stops before this step:
it recognizes sparse pair obligations but does not label them production
performance costs. A future calibrated producer should use zero as the default
and assign a positive weight only to repeatedly demonstrated harmful
mechanisms. For several candidate predecessors of one consumer, retain the
dominant supported pair rather than summing duplicate evidence. OR groups,
hyperedges, the solver representation of negative marginals, and global
event-budget terms remain deferred.

## Critical-path model v0

The next modeling layer is a schedule graph rather than another recognizer
filter. A diagnostic PTOAS exporter records the post-InsertSync operation
nodes, per-pipe issue-order edges, synchronization edges, loop membership,
and known allocation sizes. `pypto.tools.dsa_schedule_model` assigns operation
durations and evaluates two directed acyclic graphs:

- a baseline containing only per-pipe stream edges; and
- the collapsed operation-only graph after adding synchronization edges whose
  endpoints are both represented operation nodes.

Version 0 reports the difference between their longest paths as
`synchronization_exposure_cycles`. It also evaluates every synchronization
edge against the baseline critical path. This is deliberately different from
counting synchronization groups: an edge receives zero exposure when it is
covered by work already on the critical path.

Static loops are aggregated by their trip count in the whole-function DAG.
Dynamic loops fail closed because the model has no defensible multiplier.
Loop-carried synchronization edges are excluded from that acyclic DAG and
handled separately with the recurrence lower bound described below. Valid
loop-marker endpoints remain in the structural graph as zero-duration nodes;
only endpoints that are genuinely absent from the imported graph are excluded
and reported. Legacy trace imports may also omit barrier dependency nodes. In that case
`latency_graph_complete` is false even if no excluded edge is visible in the
imported graph.

The exporter exposes active Final-SyncIR records before SyncCodegen lowering.
Those records are not synchronization instructions: codegen can coalesce
identical set/wait operations and neighboring barriers. The model therefore
reports them as `pre_codegen_sync_record_summary`. Candidate reuse uses a
separate, explicitly hypothetical `sync_endpoint_estimator_version`; its
source-plus-target execution count is an uncoalesced pressure feature, not an
observed instruction count. Actual post-InsertSync instruction summaries must
be collected from lowered IR for each placement arm.

Model version `loop_sync_ii_and_boundary_v1` retains zero-duration loop
markers in the structural graph. It reports loop-entry and loop-exit
synchronization separately and models every identified distance-one
loop-carried dependency as a recurrence lower bound on initiation interval.
This remains a lower bound: it does not claim a modulo schedule or account for
finite event-slot allocation.

The actual per-arm instruction collector consumes a manifest of lowered
post-InsertSync PTO files. It fails if high-level `record_event`/`wait_event`
operations remain, requires the declared arm and function sets for every
case/capacity, and counts the synchronization instruction sites actually
present in the lowered IR. It also infers candidate event-lifecycle transitions
under the versioned `event_key_lexical_and_innermost_backedge_v1` contract.
Those transitions are not directly emitted facts: event IDs can be recycled,
and lowered IR does not preserve Final-SyncIR group identity. The inference
therefore remains separate from the actual instruction-site counts.
For statically bounded loops, the collector also reports estimated dynamic
instruction executions by multiplying each actual site by its enclosing trip
counts. The estimate is labelled incomplete instead of guessing when a
synchronization site's enclosing loop bound is dynamic or unresolved, or when
the site is nested in unresolved conditional/control-flow regions. Unstructured
control flow also makes the function-level estimate incomplete.

The realized-placement scorer keeps six distinct levels of evidence instead
of treating every logical buffer pair as an independent hardware event:

1. `unit_realized_cost` counts the promoted logical buffer-pair weights;
2. `canonical_physical_reuse_group_count` collapses duplicate logical pairs
   that refer to the same unordered pair of full physical tile ranges;
3. `unique_induced_sync_edge_count` collapses those groups again by verified
   schedule-edge identity;
4. `estimated_sync_endpoint_executions` applies static loop trip counts to the
   unique source and target endpoints;
5. `critical_path_realized_cost_cycles` sums the singleton critical-path
   extension assigned to each realized logical relation; and
6. `complete_placement_critical_path_cycles` inserts the union of all unique
   realized dependency edges into one reference DAG and computes one
   longest-path extension.

The physical-range key includes the full two placed ranges, not only their
intersection: distinct tile layouts can share the same intersection. This
canonicalization removes duplicate alias evidence but does not infer an
undocumented hardware bank or interleave mapping. The penalty-model evaluator
reports all six metrics so a device ordering can reveal the first abstraction
level at which the arms actually separate.

The sixth metric is the InsertSync-independent complete-placement score in
model v2. It reconstructs the non-reusing base graph from two sources only:

- fixed issue order on every execution pipe; and
- logical-root RAW, WAR, and WAW dependencies derived from operation
  `uses`/`defs` metadata.

Cross-pipe dependencies already present in this base graph carry the same
calibrated synchronization edge cost as placement-induced dependencies;
same-pipe dependencies rely on FIFO pipe order and add no separate cost. This
preserves the baseline synchronization slack against which a reuse edge is
measured.

Existing `sync_edges`, synchronization groups, barrier records, and physical
addresses do not participate in the base graph. For each pair of buffers that
the placement physically overlaps, the scorer joins the exported pre-InsertSync
access provenance to a directed address-reuse hazard. Every unique hazard is
inserted once with the positive calibrated `sync_latency_cycles` edge weight.
The complete placement penalty is

```text
penalty(P) = LP(G_no_reuse + E_reuse(P)) - LP(G_no_reuse)
```

This is not the sum of pairwise penalties. One finite longest-path calculation
captures duplicate edges, transitive ordering, shared slack, and interactions
among all selected relations. Static loops are expanded to dynamic operation
occurrences, so a distance-one hazard connects iteration `i` to `i + 1` and
pays the synchronization weight on every exposed recurrence.

The positive edge weight models the synchronization mechanism in addition to
the overlap destroyed by the new precedence constraint. A zero value is only a
dependency lower bound and is rejected as uncalibrated by model v2. The weight
may initially be one architecture-level constant and later become pipe-pair or
signature specific, but it must be frozen independently of the placements
being evaluated.

Model v5 no longer rejects all structured control flow. The raw-PTO bridge
attaches a stable predicate identity and polarity to every `scf.if`. Re-tests
of the same materialized predicate share one scenario variable, and candidate
sites on mutually exclusive paths do not create a reuse edge. The scorer
enumerates the reachable structured paths instead of guessing branch
frequencies.

One scenario bit is shared across loop iterations only when raw PTO proves that
the predicate is defined outside every enclosing loop (or is a function
argument). For a loop-contained predicate, the bridge follows the scalar SSA
chain through integer casts and arithmetic. If it reduces to a comparison of a
static induction variable and constants, the scorer records the exact Boolean
sequence for every iteration. A runtime-loaded or otherwise unresolved
loop-contained branch is `INCOMPLETE`; the model never substitutes
all-then/all-else paths for an unknown mixed per-iteration profile.

Dynamic loops are parametric rather than expanded to an arbitrary finite trip
count. Loops proven by raw PTO to share the same lower bound, upper bound, and
step share one symbolic parameter `N`. The scorer evaluates four concrete
probes and accepts the *placement extension* only when those probes have the
affine form

```text
startup + (N - 1) * steady_state,  N >= 1
```

It may compare the resulting affine models by dominance for `N >= 1`, but the
score is labelled `PARAMETRIC_ASSUMPTION`, not `COMPLETE`. This is explicitly an
extrapolation from exact probes at `N = 1, 2, 3, 4`, not a proof that an
arbitrary max-plus graph remains affine for all trip counts. It fails closed if
the probes are not affine, if independent dynamic parameters would have to be
conflated, or if the raw-PTO identity is missing. Static loops still use finite
expansion.

The lowered-access join distinguishes a missing exporter record from an access
eliminated by lowering. Complete raw-PTO access provenance can prove that an
absent access order is non-materialized. It cannot prove that a loop-carried
recurrence disappeared merely because its surviving endpoints do not share a
lowered loop. Such a recurrence fails closed until the exporter preserves its
original-loop identity through peeling, unrolling, or loop splitting. The same
identity is required when endpoints share multiple nested lowered loops. A
distance-one handoff between contradictory branch arms also fails closed until
the recurrence scorer evaluates source iteration `i` and target iteration
`i + 1` separately; same-iteration mutual exclusion is not sufficient evidence
to remove that edge.

The remaining fail-closed conditions include missing branch nodes, unresolved
access joins, promoted `pipeline_serialization` relations with no operation
provenance, an uncalibrated edge weight, independent dynamic loop parameters,
or an oversized expansion. An old export that omitted branch nodes is
therefore `INCOMPLETE`; it is never treated as a branch-free graph with zero
placement cost.

An August 2026 host-only re-export initially reported zero complete-placement
extension because it used the geometry endpoint's post-InsertSync graph as the
reference. That graph already contained geometry's placement-induced
dependencies. Model v2 fixes the methodological error directly: it ignores all
InsertSync records and rebuilds the base graph from logical SSA/allocation
roots and fixed pipe order.

### Global synchronization-weight sensitivity

The current host-only sweep rescored the complete placement graph at one global
synchronization-edge weight over `16, 32, 64, 96, 128, 160` cycles. It did not
fit a per-kernel or per-edge constant. Device orderings are labels only when the
archived campaign established a reproducible two-device effect; below-threshold
and non-reproducing effects remain diagnostics.

The reanalysis covers the 19 previously measured non-Gate problem-capacity
cells plus the historical multi-function `mtp/gate` counterexample. The twelve
loop-aware RMSNorm/top-k cells remain complete at every weight. Exact
mixed-iteration profiles are derived statically for induction-variable
predicates with static bounds. Runtime-loaded predicates instead require an
explicit `exact_runtime_branch_profile_v1` input. That input is bound to hashes
of the schedule, problem, captured tensors/scalars, and loop-trip metadata; it
records each active branch occurrence and never promotes a captured value into
a compile-time fact. Nested branches additionally record their active flattened
occurrence indices.

Using the archived deterministic inputs,
`rmsnorm_rope_cache_write/half` is now complete. Its two runtime-loaded
predicates use the exact mixed branch profile of the measured dispatch. The
same contract can represent `softmax_pool_c128/native`, but that case has not
yet been rescored from a captured profile and is not new evidence here.

The KV gap is repaired. A fresh export records producer and consumer access
sites for every `pipeline_serialization` penalty. Of the 64 realized pairs in
each arm, Cypress has 34 pipeline-serialization pairs and DSA-RP has 40; 16 and
20 respectively materialize as lowered operation edges, while the remainder
are proven eliminated by a unique raw-PTO join. Despite equal unit cost
(`64 -> 64`), the complete-placement score assigns Cypress an extension of
`408, 560, 1200, 1840, 2480, 3120` cycles over the frozen weight grid and DSA-RP
zero throughout. This correctly predicts the measured DSA-RP win without
consulting InsertSync.

Gate was re-exported product-faithfully at its frozen compiler revision. The
new source-loop marker survives PTO codegen and lets `gate_aic` join multiple
lowered loops back to their original source loops; both `gate_aic` and
`x_norm_quant` are now complete. Pinned PTO-ISA estimates also cover Gate's
`tmul(fp32, 1x4096)`, `tfree`, and `tfillpad` operations without a local fitted
constant. The aggregate parent score still fails closed: `ffn_norm`,
`gate_aiv`, and `route_sort` require unsupported `trecip`/`tabs`, `tmaxs`, and
`trowexpanddiv` signatures respectively. A parent-wide Gate ordering must not
be inferred from the two complete children.

Global calibration is still not identifiable. The confirmed KV and RMS wins
are now explained over the entire frozen weight grid, but Gate's confirmed
Cypress ordering has no complete parent score. Moreover, the complete Gumbel
model predicts DSA-RP better in every symbolic branch scenario while both
devices measure a latency null at q1. Leave-one-workload-out calibration
therefore has too few confirmed, model-complete directional labels, and no
weight is selected.

The requested representative cases expose the missing model coverage and the
remaining mechanism gap:

| Case | Existing two-device evidence | Unit objective | Whole-function model status |
| ---- | ---------------------------- | -------------- | --------------------------- |
| `rmsnorm_rope_cache_write/half` | DSA-RP over Cypress `-7.19%/-7.43%` | `8 -> 0` | Complete with the exact digest-bound runtime profile. Cypress adds `62, 97, 161, 354, 610, 761` cycles and DSA-RP adds zero over weights `16--160`. |
| `kv_score_proj_c128/native` | DSA-RP over Cypress `-2.50%/-2.30%` | `64 -> 64` | Complete. Pipeline-serialization provenance and the repaired lowered-access join give Cypress a positive critical-path extension and DSA-RP zero at every tested weight. |
| `mtp/gate` | Cypress faster at all four capacities, approximately `1.6--4.1%` | DSA-RP ties or improves the count | Re-exported but parent-incomplete: source-loop identity is repaired and two children are complete, while three fail closed on unsupported duration signatures. |
| `gumbel_argmax/q1` | DSA-RP versus Cypress `+0.22%/+0.11%`, a latency null | `14 -> 0` | Complete and false-confident at every weight. Correlating the three re-tests of one predicate removes impossible paths but does not explain the null. |
| `hc_post/native` | weak, sign-changing, or non-reproducing effects across campaigns | `33 -> 24` in the original cell | `PARAMETRIC_ASSUMPTION` under one correlated dynamic-loop parameter. It predicts DSA-RP no worse, but neither the extrapolated score nor the device label supports validation. |

The durable device sources are
`dsa-rp-loop-aware-model-prospective-0820ab418-final.tar.gz`
(`1a7e5d5ffe93a43b260012d47af98321cb5a10156ecc8486dbc37f00767374d2`)
and `dsa-rp-four-candidate-physical-penalty-aeba32c70-final.tar.gz`
(`a05aad5829865d196bc7d7a415b40d8c06b3e6b566d1f80fce269352c78765a0`).
Every `score-realized-grid` result records the SHA-256 of its schedule, problem,
solution, duration model, and optional non-materialization evidence, plus the
selected function and fail-closed duration policy. A grid can be reproduced
without opening latency data while scoring each frozen placement:

```bash
python -m pypto.tools.dsa_schedule_model score-realized-grid \
  SCHEDULE.jsonl PROBLEM.dsa.json SOLUTION.dsa.solution.json \
  --function FUNCTION --model DURATION_MODEL.json \
  --sync-latency-grid 16,32,64,96,128,160 -o ARM_GRID.json
python -m pypto.tools.dsa_penalty_model_evaluation sync-weight-grid-input.tsv \
  --sync-weight-grid --split development --minimum-device-effect 0.02 \
  --required-device-count 2 -o sync-weight-grid-evaluation.json
```

This analysis fails the gate for an incremental critical-path planner. The
planner has not been changed and no new device task is justified from this
grid. Static and exact runtime mixed-iteration branches, source-loop identity,
KV pipeline provenance, and the KV access join are complete. The predeclared
scientific gate nevertheless fails: Gate is duration-incomplete and Gumbel is a
false-confident prediction at every tested weight. The incremental greedy
planner is therefore deliberately not implemented. The next local requirement
is to resolve those two failures and obtain enough independently measured,
model-complete directional cases for meaningful leave-one-workload-out
calibration.

Scoring each arm's actual post-InsertSync graph gives the intended analysis
oracle. On the same retrospective corpus, every one of 40 strict model
orderings agrees with the measured direction; all 24 strict comparisons whose
device effect is at least 2% agree. RMSNorm geometry scores 12,736 cycles versus
12,249 for both Cypress and DSA-RP, predicting the correct speedup direction
but only about 3.8% versus the measured 21.6--22.7%. A one-edge graph ablation
isolates the modeled difference: geometry adds a V-to-MTE2 dependency from the
first loop's end to the second loop's beginning. Removing that edge changes
12,736 to 12,249 cycles; adding it to Cypress changes 12,249 to 12,736. Thus
the post-InsertSync latency approach is directionally useful on this corpus,
while its magnitude model remains incomplete.

The same re-export recovered product-faithful graphs and timing-blind logical
and physical placement catalogs for the five-cell physical-penalty corpus.
Those targets contain structured branches, so candidate model v1 still fails
closed rather than inventing unconditional edges between mutually exclusive
paths. The KV graph additionally lacks lowered sites for 48 of 128 raw
candidate records. A branch-aware candidate-to-join mapping is therefore the
next modeling requirement; the recovered catalogs preserve every branch and
loop context needed to implement it.

### Signed post-InsertSync marginal cost

Legal pair ablations show that a reuse relation is not necessarily a positive
latency obligation. For a fixed surrounding placement `P` and one relation
`r`, the analysis oracle is therefore

```text
p(r | P) = L(InsertSync(P + r)) - L(InsertSync(P))
```

where `L` is the loop/resource-aware makespan estimate of the complete
post-InsertSync schedule. The value is deliberately signed: a negative value
means that adding `r` removed a more expensive synchronization dependency.
The `evaluate` command exports this value as
`signed_marginal_sync_cost_cycles` and fails closed unless both latency graphs
are complete. It remains an analysis oracle, not yet the sparse approximation
used by the DSA solver.

The evaluator also reconstructs the candidate dependency graph from the
baseline plus the multiset delta of final InsertSync edges. Reconstruction must
produce exactly the candidate's modeled makespan. It then scores each added or
removed final edge independently in the baseline context:

```text
q(P' | P) = sum(e in E(P') - E(P)) delta_add(e | P)
          + sum(e in E(P) - E(P')) delta_remove(e | P)
```

The report keeps `q` and the non-additive residual `p - q` separate. A
deterministic sequential attribution is also emitted and must telescope to the
exact marginal. This catches both duplicate synchronization edges and cases
where several individually exposed dependencies cover the same critical-path
segment.

On the existing RMSNorm/top-k development slice, the final-edge approximation
has 40 strict predictions and all 40 agree with device direction; all 24
comparisons with a device effect of at least 2% also agree. The interaction
residual is zero in all 72 arm/device comparisons. This is encouraging but is
not yet a solver penalty: the approximation consumes the final InsertSync edge
delta. The remaining compiler bridge must predict that delta from a logical
reuse relation and its surrounding partial placement, including loop-boundary
lifting and dependency implication.

The controlled Gumbel study isolates four relations while holding operation
order and address translation controls fixed. The synchronization column is a
correlate of each relation contrast, not by itself a causal attribution:

| Relation | Final synchronization change | Two-device latency result |
| -------- | ---------------------------- | ------------------------- |
| `(2,39)` | removes one V-pipe barrier | `-2.07%/-2.30%` and `-2.16%/-2.35%` |
| `(38,42)` | adds one barrier, set, and wait | about `+1.9%` on both devices |
| `(3,38)` | adds one final barrier | about `+0.4%` |
| `(38,79)` | adds set/wait sites but no barrier | latency null |

Structurally, for `(2,39)`, D0 contains a loop-carried V-to-V WAR from the
prior iteration's `trowargmax` read to the next iteration's else-arm `tmov`
write. InsertSync therefore emits a barrier before that `tmov`. Restoring the
overlap aliases the `trowargmax` scratch with the next iteration's MTE2 load
destination. That adds
a V-to-MTE2 recurrence, and the existing MTE2-to-V load-completion handoff then
makes the direct V-to-V dependency transitively implied. The barrier disappears
inside `InsertSyncAnalysis`; phase-by-phase dumps show it is already absent
before `MoveSyncState`, `RemoveRedundantSync`, and event-ID allocation. This is
dependency implication, not event coalescing.

The exact logical-to-lowered trace is:

| Relation | DSA access orders | Lowered operations | Post-InsertSync change | Dynamic location |
| -------- | ----------------- | ------------------ | ---------------------- | ---------------- |
| `(2,39)` | `139/140 -> 142` and distance-one `142 -> 99` | `tadd`/else `tmov` -> `trowargmax`, then `trowargmax` -> `tload` | removes V-pipe barrier `52 -> 49`; changes recurrence source `51 -> 52` for `-> 11` | inside the 63-trip outer loop |
| `(3,38)` | `114 -> 139` and distance-one `144 -> 101` | `tcolexpand` -> `tadd`, then scalar `tgetval` -> `texpands` | adds V-pipe recurrence barrier `52 -> 13` | inside the 63-trip outer loop |
| `(38,42)` | `144 -> 153` | scalar `tgetval` -> post-loop `texpands` | adds S-to-V handoff `59 -> 61` and V barrier `52 -> 61` | once after the loop |
| `(38,79)` | `144 -> 194/195` | scalar `tgetval` -> branch `tadd`/`tmov` | adds branch-lifted S-to-V handoff `59 -> 64` | once after the loop |

The other relations separate measured effects from structural counts.
`(38,42)` adds post-loop synchronization and has a reproducible positive
effect. `(3,38)` adds a V barrier but costs only about 0.4%. `(38,79)` adds an
event pair and is a latency null. Consequently neither logical reuse count nor
barrier count alone is a defensible penalty model.

The branch-aware schedule graph uses one zero-duration control point per
`(branch-or-loop marker, pipe)`. Then and else arms start from the same per-pipe
frontier and merge by taking their maximum; they are never serialized. Sync
operations attached to `IF_BEGIN`, `IF_END`, or loop markers bind to the
corresponding pipe-specific control point. Legacy PTOAS debug imports preserve
the printed branch skeleton and barrier dependency node when present. Missing
barrier dependencies or branch nodes continue to make
`latency_graph_complete=false`.

This support applies to scoring a complete post-InsertSync arm. The older
`candidate_v1` hypothetical-edge scorer still fails closed when a candidate
endpoint is conditional, because lifting a set from one arm to the branch join
is an InsertSync transformation rather than a plain graph-edge insertion.

The archived KV endpoint predates the pre-DSA Simplify placement now used by
the research pipeline. Its candidate access orders 98 and 103 are eliminated
before the lowered schedule. Current exports close the join directly; analysis
of the archived endpoint must instead supply digest-bound non-materialization
evidence and must not invent schedule sites for those orders.

A host-only retrospective reanalysis reconstructed all eight Gumbel endpoints
with the product PTOAS v0.57 InsertSync implementation. Every endpoint has 93
operation nodes, 100% exact/pinned duration coverage, and a complete structured
control-flow graph. The collapsed operation-only signed oracle nevertheless
does not explain the measured ordering:

| Relation | Predicted marginal | Existing two-device result | Interpretation |
| -------- | ------------------ | -------------------------- | -------------- |
| `(2,39)` | `0` cycles | about `-2.1%/-2.3%` | correctly shows no collapsed-DAG barrier exposure, but does not explain the placement effect |
| `(3,38)` | `0` cycles | about `+0.4%` | negligible effect, correctly treated as slack |
| `(38,42)` | `+189` cycles | about `+1.9%` | correct sign, underestimated magnitude |
| `(38,79)` | `+56` cycles | null | small structural false positive |

The queue/event model `static_unrolled_pipe_event_branch_extremes_v2`
explicitly unrolls statically bounded loops, preserves each pipe's FIFO issue
order, and maps loop-carried events from iteration `i` to `i + 1`. A prospective
device validation recovered the real branch profile independently as six THEN
and two ELSE blocks. On ten topology contrasts, the model removed both sign
errors made by unsigned reuse count, but it did not beat emitted barrier-site
count and it missed the central `(2,39)` result:

| Relation | Real-profile queue/event marginal | Existing result |
| -------- | --------------------------------- | --------------- |
| `(2,39)` | `0` cycles | beneficial, about `-2.1%/-2.3%` |
| `(3,38)` | `+63` cycles when active | small regression, about `+0.4%` |
| `(38,42)` | `+192` cycles | regression, about `+1.9%` |
| `(38,79)` | `+56` cycles | latency null |

The removed node-49 barrier is in the ELSE arm. It does not execute in the six
long THEN blocks, yet those blocks carry most of the measured improvement.
Therefore the experiment does **not** establish that removing the barrier
caused the `(2,39)` speedup. A placement-by-barrier 2x2 factorial, plus a
same-footprint code-layout control after device disassembly, is required for
that claim.

The same validation rejects a per-pipe constant: device-0 calibration from
`(3,38)` and `(38,42)` differs by 4.40x. PTO-ISA explains why. A barrier charges
the barrier instruction, drains queued predecessor tail work, clears the
stream, and makes the successor repay startup. The public evaluator therefore
also reports `queue_drain_restart_signed_marginal`, whose site cost is:

```text
barrier instruction + predecessor pending tail + successor stream restart
```

The startup/tail split is resolved by complete operation signature from pinned
PTO-ISA data or an explicit calibration. Transfers for which the split is not
available fail closed. This is intentional for node 49's `tmov`: the factorial
and disassembly must establish the device mechanism before a value is fitted.
Barrier dependency provenance is read from the public exported
`sync_groups.operations.dependency_node` field; the evaluator no longer needs
a campaign-private reconstruction path.

```bash
python -m pypto.tools.dsa_schedule_model evaluate arm-manifest.json \
    --model duration-model.json -o signed-marginals.json
```

```bash
python -m pypto.tools.ptoas_sync_summary --arm-manifest post-sync-arms.json \
    -o post-sync-summary.json
```

Operation durations come from
a provider snapshot loaded from the exact PTO-ISA revision in
`runtime/pto_isa.pin`. The snapshot includes the fitted A2/A3 formula rows,
transfer bandwidths, frequency, source hashes, and the full revision.
Unsupported operations fail closed by default. An exploratory run may opt into
`--unsupported-policy fallback`, but every such node is labelled and the score
reports exact/fallback coverage. Simulator instruction medians can override
the analytical provider without removing its provenance.

```bash
python -m pypto.tools.dsa_schedule_model snapshot-duration \
    --pto-isa-root <exact-pto-isa-checkout> -o duration-pinned.json
python -m pypto.tools.dsa_schedule_model score schedule.jsonl \
    --model duration-pinned.json -o score.json
python -m pypto.tools.dsa_schedule_model validate-perf-sim trace-*.json \
    --model duration-pinned.json -o perf-sim-validation.json
python -m pypto.tools.dsa_schedule_model calibrate instr_metrics.json \
    --base-model duration-pinned.json -o duration-calibrated.json
```

The pinned provider removes PyPTO's former silent, coarse pipe constants; it
does **not** make the structural schedule graph cycle-accurate by itself.
Perf-Sim uses the richer CCE mock's recorded cycles when available and only
falls back to PTO-ISA's lightweight formulas otherwise. On the pinned A2/A3
validation set (102 formula-supported events), the lightweight provider had an
82-cycle mean absolute error and 73.6% mean absolute percentage error against
effective Perf-Sim events. Elementwise `TMUL` was much closer (11.3% MAPE),
while reductions and exponentials were not.

The paired device check reaches the same conclusion. On the branch-free
RMSNorm case in the current four-capacity development dataset, the provider
scores 40% of nodes exactly and labels the rest as fallback. It predicts equal
full makespans for all three physical endpoints at all four capacities, while
the archived device measurements show about a 13% geometry-to-penalty-aware
improvement at native, half, and quarter capacity. Thus the provider is a
pinned, auditable starting point, but the current critical-path model does not
yet explain the observed placement effect. Per-kernel Perf-Sim instruction
traces are required for calibration before these cycle scores can be used as
DSA-RP weights.

The current critical-path calibration subset uses the
`structured_branch_static_loop_v2` eligibility policy. It accepts loops with
an exported non-negative static trip count and structured if/else branches;
only dynamically bounded loops remain excluded. This is a timing-blind
structural filter: it does not inspect solver objectives or prior device
results. Qualify exported schedules before freezing that analysis subset with:

```bash
python -m pypto.tools.dsa_schedule_model qualify schedule-*.jsonl \
    -o schedule-eligibility.json
```

Corpus discovery is driver-first. A source is considered only when static
inspection proves a local JIT entry, tensor specifications, a direct golden,
and an executable `run_jit` contract. DSA problems are joined only from a fresh
export of that exact PyPTO-Lib revision. Older inventories are hints for which
sources to re-export; they cannot make a current candidate. The discovery tool
also records the real measurement unit instead of counting every child DSA
problem as an independent kernel:

- one DSA problem in one submit is a single-kernel driver;
- multiple DSA problems in one submit form one complete mixed group; and
- multiple submits form one parent-wide policy workload.

```bash
python .claude/skills/incore-profiling/discover_dsa_direct_golden_corpus.py \
    --pypto-lib-root <pypto-lib> \
    --invocations <fresh-export>/invocations.tsv \
    --inventory-revision <exact-pypto-lib-sha> \
    --export-status <fresh-export>/export-status.tsv \
    --output-root <discovery>
```

The base problem identity is the semantic DSA fingerprint. Controlled tiling
variants must carry a separate explicit tiling identity and remain grouped by
base problem; they are not independent workload families. Discovery rejects
performance fields in the current export inventory. It may annotate a prior
terminal status, but that annotation never changes membership or ordering.

The device corpus is then frozen by measurability, not by observed performance
or solver objective. A case must be feasible, runnable, and correct under all
four logical policies (geometry first-fit, geometry canonical greedy, Cypress,
and DSA-RP canonical greedy) at native, half, q1, and tight capacity. Schedule
eligibility is recorded separately and gates only critical-path predictions;
branches or dynamic loops do not make a device-measurable case ineligible. The
cohort freezer rejects input tables containing timing, speedup, objective, or
predicted-critical-path columns:

```bash
python -m pypto.tools.dsa_measurement_cohort preflight.tsv results/ \
    --minimum 20 --maximum 40
```

After launchability is established, select one evaluation capacity per workload
without consulting device time. The purpose of this capacity is to expose the
difference between unweighted Cypress relaxation and DSA-RP's structured
objective, rather than merely to maximize memory pressure. A capacity is an
**opportunity capacity** only when:

- all four logical policies are feasible and independently valid;
- Cypress realizes at least one penalized reuse relation;
- geometry first-fit, Cypress, and DSA-RP have three distinct complete maps; and
- DSA-RP has a strictly lower realized reuse-penalty objective than Cypress.

Among opportunity capacities, the selector maximizes, in order, the
Cypress-minus-DSA-RP objective gap, the symmetric difference of realized
penalized relations, and the symmetric difference of all realized reuse
relations. Tighter capacity is only the final tie-breaker. This rule is wholly
structural: solver runtime and device latency are rejected as inputs. A workload
with no opportunity capacity remains an explicitly labelled null control rather
than being silently promoted from a two-map or objective-tied cell. Mandatory
disjoint-size shortage remains an audit field, not a selector input.

The currently measured workloads are a development corpus: it is valid to join
their structurally frozen opportunity capacities to existing timings to refine
the model, but not to call that join prospective evidence. New workloads form a
holdout only when their capacities are frozen with the same rule before any of
their timing is inspected:

```bash
python .claude/skills/incore-profiling/select_dsa_workload_capacity.py \
    --cohort inputs/results/corpus-frozen.tsv \
    --instances fresh-export/invocations.tsv \
    --feasibility inputs/results/full-policy-feasibility.tsv \
    --maps inputs/results/map-digests.tsv \
    --workload-status inputs/results/workload-status.tsv \
    --corpus-root fresh-export --replay-root inputs \
    --output-root evaluation-capacities

# Development-only, and only after evaluation-freeze.json exists:
python .claude/skills/incore-profiling/evaluate_dsa_opportunity_freeze.py \
    --freeze evaluation-capacities/evaluation-freeze.json \
    --pairwise-effects prior-timing/results/pairwise-effects.tsv \
    --output-root development-analysis

# Prospective holdout, before any timing:
python .claude/skills/incore-profiling/freeze_dsa_opportunity_holdout.py \
    --opportunity-freeze new-candidates/evaluation-freeze.json \
    --development-freeze \
      .claude/skills/incore-profiling/dsa_driver_first_opportunity_development_v1.json \
    --minimum 8 --maximum 12 --output-root prospective-holdout
```

The holdout freezer rejects performance-bearing input fields and excludes both
development scripts and semantic DSA problem fingerprints. This prevents a new
wrapper filename from relabelling an already observed problem as prospective.
It selects deterministically for source-class and model-family diversity, then
for structural opportunity, and seals the resulting capacity rows before the
device timing table may be opened.

The first application of this rule is frozen in
`dsa_driver_first_opportunity_development_v1.json`. After excluding one
stock-golden-blocked gate workload, it contains 19 workloads: 16 opportunity
cells and three structural null controls, at 11 tight, four quarter, two half,
and two native capacities. The post-freeze development join finds four
confirmed Cypress-versus-DSA-RP orderings: three agree with the structured
objective and one disagrees. Only one cell confirms the complete
geometry-first-fit > Cypress > DSA-RP latency ordering. Objective-gap magnitude
is negatively rank-correlated with the measured DSA-RP advantage in this small
development set, so a larger unit-cost gap must not be presented as a latency
prediction. These observations motivate better penalty weights; they are not
prospective validation of the selection rule.

When more than 40 cases qualify, selection is deterministic and round-robin
across model family and parent program. Endpoint-identical logical policies are
retained in the matrix but need only one physical measurement.

Planner comparisons use paired schedule graphs and the convention
`candidate / baseline - 1`, where a negative value predicts that the candidate
is faster. The evaluator first requires the same operation stream in both arms,
then reports the synchronization dependencies added and removed by the
placement change. A comparison manifest may omit both observed latencies for a
held-out cohort, but it may not provide a one-sided observation. The evaluator
content-addresses the manifest, schedule inputs, and predictions so that
held-out predictions can be frozen before device timing:

```bash
python -m pypto.tools.dsa_schedule_model calibrate \
    instr_metrics.json -o duration-v0.json
python -m pypto.tools.dsa_schedule_model evaluate \
    comparisons.json --model duration-v0.json -o predictions.json
```

This model is research infrastructure, not the current `unit_v1` weight
policy. Before it can assign DSA-RP weights, it must predict fresh arm-pair
latency directions and rankings. In particular, a single schedule graph can
validate graph construction but cannot explain a placement-induced latency
difference.

The raw candidate and schedule coordinates are joined explicitly. With
`PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1`, PTO codegen wraps each stamped lowered
operation location in a `pypto.access.N` NameLoc. `N` is attached to the source
Call when the DSA problem is constructed and preserved through later lowering;
it is not recomputed from lowered statement order. It therefore matches the
candidate record's `sites=prior->next` field. PTOAS copies that integer to
the schedule graph. A defensive `Simplify` now runs before `InitMemRef`: it
removes statically dead pipeline-slot branches before they can contribute DSA
lifetimes or candidates. The join fails closed when a site is absent or when
the candidate route has no verified PTOAS-pipe mapping; SSA node numbers and
source line numbers are never treated as interchangeable coordinates.

Historical problems created before that boundary may still contain candidate
records for operations that do not exist in their paired lowered schedule.
They may be scored only with an explicit `--nonmaterialized-access-evidence`
document whose SHA-256 fields bind it to that exact problem and schedule. Such
records retain their logical unit penalty for auditing but contribute zero to
the executable relation, physical-group, synchronization-execution, and
critical-path predictors. This exception is evidence-driven; without it the
join continues to fail closed.

Solver-promoted `pipeline_serialization` penalties describe relaxed
pipeline-stage separation. The v5 exporter records their producer and consumer
access sites alongside the penalty reason, so current problems can join them
to lowered operations like other reuse relations. Historical exports without
those v5 records remain incomplete whenever a placement realizes such a
relation; the missing provenance must never be interpreted as a
non-materialized operation or a modeled cost of zero.

The first candidate-weight prototype adds a hypothetical completion edge from
the terminal macro phase at the prior site to the initial macro phase at the
next site, retaining all synchronization already present in the reference
schedule. Its non-negative weight is the increase in longest-path cycles. It
also adds all candidate edges sharing one consumer together and reports the
difference between the combined cost and the sum of singleton costs. This
consumer grouping exposes release coalescence rather than double-counting it:

```bash
PYPTO_EMIT_DSA_ACCESS_PROVENANCE=1 python my_export.py
python -m pypto.tools.dsa_schedule_model score-candidates \
    schedule.jsonl problem.dsa.json --model duration-v0.json \
    -o candidate-weights.json
```

When a solution is supplied, the report also contains `edge_explanations`.
Each row traces one logical reuse relation through its canonical overlapping
physical range, lowered producer and consumer operations, inserted sync group,
loop execution multiplier, and critical-path or recurrence slack. Duration
calibration is keyed by the full pinned PTO-ISA signature; unsupported
signatures fail closed rather than falling back to an instruction-family
median.

Version 1 retains the acyclic longest-path score for distance-zero candidates
and adds a lower-bound score for distance-one candidates. It joins both PTOAS
sites, checks that they share a real loop, and selects their innermost common
loop. The base initiation-interval lower bound is the maximum of per-pipe work
in one iteration and every already-present single-recurrence cycle. For a
hypothetical edge `source(i) -> target(i+1)`, the model finds the longest
intra-iteration path `target -> source`; if it exists, that path plus the edge
latency is another recurrence bound. The non-negative weight is the increase
in the base lower bound. If no return path exists, the edge changes phase but
does not raise this throughput bound and receives zero.

This is deliberately a lower bound, not a complete modulo-scheduling model.
It does not yet search cycles containing several new recurrence edges.
Multiple candidate records that join to the same `(loop, source, target)` are
therefore collapsed in `loop_recurrence_edges` so downstream analysis does not
sum duplicate evidence. Distance-zero records are likewise collapsed in
`distance_zero_edges`; `candidate_weight_summary` reports counts, sums, and
maxima over these unique schedule edges rather than raw buffer-pair records.

For PTOAS revisions that predate the JSONL exporter, the legacy level-3 debug
importer can recover the same stable access coordinates from the raw PTO.  The
join requires exact executable-operation order and fails instead of guessing
when an operation or `pypto.access.N` location is missing:

```bash
python -m pypto.tools.dsa_schedule_model import-debug insert-sync.log \
    --function kernel --pto kernel.pto -o schedule.jsonl
```

PTOAS also attaches a loop-end identity to some function-wide event
lifecycles. The bridge classifies a group as a recurrence only when every
active set/wait endpoint is an operation inside the indicated loop. A
prologue-to-loop-end or prologue-to-epilogue lifecycle remains a boundary
dependency rather than becoming a spurious initiation-interval constraint.

The raw-PTO join also preserves operand/result types and scalar constant
operands, and derives `static_work_bytes` from statically shaped tile and
partition types. This is enough to price transfers without inventing DSA
allocation sizes: the latter
remain marked missing because legacy SyncIR buffer identifiers do not match raw
PTO SSA names. A trace-side `pto.tmatmul.acc` may join a raw `pto.tmatmul` only
when the raw operation has an accumulator input; an ordinary two-input matmul
remains distinct and fails the mismatch check.

Version 0 has verified pipe mappings only for inbound/outbound DMA, L1-to-L0,
L0-to-external, vector, matrix, and scalar resources. It rejects the remaining
transfer-route families until their PTOAS pipeline mapping is established.

## Remaining validation

The completion-frontier factorial has been run. It confirmed that several
active predecessors of one consumer may collapse to one release frontier, but
the isolated frontier extensions were latency-neutral because an existing drain
already quiesced the resource. It supports consumer-aware deduplication; it does
not justify a positive weight.

The next study should start from kernels with a replicated RP-versus-compact
latency difference and work backward:

1. identify candidate pairs changed by the endpoint placement;
2. construct exact-XOR single-pair and small factorial placements, preferably
   with a capacity-preserving address exchange;
3. freeze the predicted mechanism before compiling through PTOAS;
4. compare the complete synchronized instruction topology and predecessor
   identity, not only group counts;
5. validate all written outputs with real kernel inputs and scalars; and
6. measure kernel-only latency on two devices, escalating samples only when the
   initial confidence interval is informative.

The immediate targets are the confirmed UB and L1 kernels for which endpoint
speedups exist but pair-level mechanism attribution is incomplete. A mechanism
earns a positive weight only when it predicts the sign across fresh kernels and
placement backgrounds. The checked-in v5 unit model may be used to generate
algorithm-study instances, but production promotion remains unsupported until
that calibration exists.
