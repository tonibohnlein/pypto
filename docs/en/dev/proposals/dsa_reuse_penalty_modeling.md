# DSA Reuse-Penalty Modeling

## Status

PyPTO's reuse-penalty recognizer is experimental and disabled by default. This
document separates three concerns that must not be conflated:

1. the stable DSA-RP optimization problem;
2. the recognizer and promotion policy currently implemented in PyPTO; and
3. the evidence used to decide which recognized candidates deserve a positive
   weight.

The current evidence supports keeping the optimization problem simple and
non-negative. It does not support enabling the current promotion policy in
production.

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

The raw records are exported in
`metadata.recognized_reuse_candidate_records_v4`. An SSA-reachable record has a
deterministic region/statement `dag_path`; an unordered record uses
`dag_path=none`. The command

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
- No experiment has justified a negative optimization weight. Apparent
  synchronization removal has not produced a replicated latency benefit.

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
- The non-negative pair model itself remains viable: neutral or apparently
  beneficial pairs can simply receive no positive edge.
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
hyperedges, negative weights, and global event-budget terms remain deferred.

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

The first critical-path calibration subset uses the `static_loop_v1`
eligibility policy. It accepts loops with an exported non-negative static trip
count and excludes branches and dynamically bounded loops. This is a
timing-blind structural filter: it does not inspect solver objectives or prior
device results. Qualify exported schedules before freezing that analysis
subset with:

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

After launchability is established, select one evaluation capacity per problem
without consulting device time. `cypress_actual_alias_pairs > 0` is not enough:
Cypress may choose reuse even when a fully disjoint placement would fit. For the
selected pool, the selector sums the sizes of buffers fixed to that pool. This
is a hard lower bound on any physically disjoint placement, independent of
alignment. It selects the least restrictive capacity for which:

- capacity is strictly below that disjoint lower bound, so reuse is mandatory;
- the forced shortage is at least 25% of the disjoint lower bound, applying the
  same timing-blind pressure floor to every problem;
- all four logical policies are feasible;
- Cypress realizes at least one alias pair; and
- geometry first-fit, Cypress, and DSA-RP have distinct physical placements.

The raw-size bound is deliberately conservative: it may reject an instance
where alignment alone forces reuse, but it cannot falsely label voluntary reuse
as capacity-forced. The pressure floor avoids choosing a barely constrained
capacity merely because one byte no longer fits. The selector records the lower
bound, byte and percentage margins, and each arm's unit reuse cost. The costs
are audit outputs and do not participate in capacity selection:

```bash
python .claude/skills/incore-profiling/select_dsa_evaluation_capacity.py \
    --problems frozen/cohort/problems.tsv \
    --problems-dir fresh-export/corpus/penalty-bearing \
    --problem-status results/problem-status.tsv \
    --screen-results inputs/screen-results.tsv \
    --minimum-forced-reuse-percent 25 \
    --output-root evaluation-capacities
```

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
the schedule graph. The join fails closed when a site is absent or when the
candidate route has no verified PTOAS-pipe mapping; SSA node numbers and source
line numbers are never treated as interchangeable coordinates.

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
