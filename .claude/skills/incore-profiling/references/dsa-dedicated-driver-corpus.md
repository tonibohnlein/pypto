# Dedicated-driver DSA corpus

Prefer existing dedicated PyPTO-Lib functional drivers over runtime argument
capture. They call the original inline kernel definitions at real model shapes
and provide direct Torch goldens.

Freeze selected DSA functions against the host screen with:

```bash
python .claude/skills/incore-profiling/prepare_dsa_dedicated_driver_cohort.py \
  --catalog .claude/skills/incore-profiling/dsa_dedicated_driver_cohort_v1.json \
  --pypto-lib-root <pypto-lib> --problems <corpus>/invocations.tsv \
  --screen-results <screen>/screen-results.tsv --output-root <cohort>
```

After a four-capacity exploratory run, freeze one primary evaluation instance
per problem without consulting device latency:

```bash
python .claude/skills/incore-profiling/select_dsa_evaluation_capacity.py \
  --problems <cohort>/problems.tsv \
  --problems-dir <corpus>/penalty-bearing \
  --problem-status <results>/problem-status.tsv \
  --screen-results <screen>/screen-results.tsv \
  --minimum-forced-reuse-percent 25 \
  --output-root <evaluation>
```

The selector takes the least restrictive capacity where the forced shortage is
at least 25% of the disjoint-size lower bound, Cypress performs real reuse, and
geometry first-fit, Cypress, and DSA-RP have three distinct selected-pool
placements. This excludes both the disjoint-address trivial regime and barely
constrained cases. Cases without three-way separation remain controls; they are
not primary instances for testing policy ordering.

Do not choose a capacity because its measured ordering is favorable. The
expected paper hypothesis is `geometry > Cypress > DSA-RP` in latency, but the
capacity is fixed exclusively from the host-side structural rule above. Device
results that contradict the hypothesis are retained for penalty-model
development rather than replaced by a different capacity.

The first four-capacity development dataset is pinned by
`dsa_penalty_modeling_dataset_v1.json`. It remains immutable when the corpus is
expanded or reduced to one evaluation capacity per problem.

For the driver-first workload corpus, apply each capacity label to every memory
pool of every child DSA problem simultaneously. Use
`combined_capacity_profiles()` and `with_pool_capacities()` from
`screen_dsa_capacity_corpus.py`; the native profile must remain byte-identical
to the exported problem. After device correctness has verified all four
capacities, freeze one workload-level capacity with:

The Cypress baseline follows the published relaxation outline but does not
claim an undocumented edge-removal policy. It starts from all auxiliary
conflict edges, retries the Knight-style packer after each removal, and probes a
timing- and penalty-weight-blind six-variant portfolio: stable ID order, reverse
ID order, largest potential overlap first, and seeded random orders 0, 1, and
2. The selected variant minimizes, lexicographically, infeasibility, realized
alias pairs, relaxed auxiliary edges, total peak, order, and seed. This makes
the baseline reproducible and at least as strong as any one arbitrary removal
order while preserving Cypress's lack of a structured synchronization-cost
objective.

```bash
python .claude/skills/incore-profiling/select_dsa_workload_capacity.py \
  --cohort <verification>/results/corpus-frozen.tsv \
  --instances <fresh-export>/invocations.tsv \
  --feasibility <verification>/results/full-policy-feasibility.tsv \
  --maps <verification>/results/map-digests.tsv \
  --workload-status <verification>/results/workload-status.tsv \
  --corpus-root <fresh-export> --replay-root <verification> \
  --output-root <evaluation-freeze>
```

The opportunity selector requires Cypress to realize penalized reuse, requires
three distinct complete maps for geometry first-fit, Cypress, and DSA-RP, and
requires DSA-RP to have a strictly lower realized penalty objective. It then
maximizes the Cypress-minus-DSA-RP objective gap, penalized-relation
disagreement, and all-relation disagreement, using tighter capacity only as a
final tie-breaker. Workloads without such a capacity are retained as explicit
null controls. Selection never reads device latency. Use the resulting current
workloads only as development data; a prospective holdout requires new drivers
whose capacity is frozen before any timing is inspected.

After screening a fresh candidate set, seal 8--12 opportunity workloads with:

```bash
python .claude/skills/incore-profiling/freeze_dsa_opportunity_holdout.py \
  --opportunity-freeze <new-candidates>/evaluation-freeze.json \
  --development-freeze \
    .claude/skills/incore-profiling/dsa_driver_first_opportunity_development_v1.json \
  --minimum 8 --maximum 12 --output-root <holdout-freeze>
```

The freezer rejects latency-bearing inputs, previously used scripts, and any
DSA problem fingerprint already present in the development corpus. It then
selects deterministically for source-class and model-family diversity before
using the structural opportunity gap. Device timing starts only after the
resulting holdout hash is recorded.

## Structured penalty model

Use all four capacities of dataset v1 as development data. For each target,
export the schedule and synchronization graph with the PTOAS version pinned by
PyPTO, preserving `pypto.access.N` provenance. Model eligibility is separate
from device measurability: statically bounded loops are supported, while
branches and dynamically bounded loops are excluded only from duration-model
v0.

When the pinned PTOAS exposes only the final InsertSync debug stream, import it
with `dsa_schedule_model import-debug --pto <raw.pto>`. The bridge joins the
operation stream by exact order and derives each static `scf.for` trip count
from the raw-PTO lower/upper/step operands. A loop-count mismatch or a genuinely
dynamic bound fails closed; neither may be replaced by a guessed trip count.
The join preserves operand/result types and derives static operation work bytes
from tile shapes. It does not invent allocation sizes whose legacy SyncIR names
cannot be joined to raw PTO. `pto.tmatmul` matches trace-side
`pto.tmatmul.acc` only when its input types prove accumulator semantics.

Derive pairwise critical-path weights and score an actual placement with:

```bash
python -m pypto.tools.dsa_schedule_model qualify <schedule.jsonl> -o qualification.json
python -m pypto.tools.dsa_schedule_model score-candidates \
  <schedule.jsonl> <problem.dsa.json> --solution <solution.dsa.solution.json> \
  --pto-isa-root <exact-pto-isa-checkout> -o placement-score.json
```

The checkout must be at the full revision pinned by `runtime/pto_isa.pin` and
its cost-model sources must be clean. Unsupported instructions fail closed.
Use `--unsupported-policy fallback` only for coverage studies; fallback nodes
and coverage are reported and must not be presented as cycle-accurate. A
portable `--model` snapshot may replace `--pto-isa-root` after it has been
created and, optionally, calibrated from kernel-specific Perf-Sim metrics:

```bash
python -m pypto.tools.dsa_schedule_model snapshot-duration \
  --pto-isa-root <exact-pto-isa-checkout> -o duration-pinned.json
python -m pypto.tools.dsa_schedule_model calibrate instr_metrics.json \
  --base-model duration-pinned.json -o duration-calibrated.json
```

The candidate scorer collapses access-site records into the promoted buffer
pairs the DSA solver sees. Distance-zero records use the combined longest-path
extension; statically bounded loop recurrences use their initiation-interval
extension over the remaining iterations. The realized-placement summary
reports the original logical unit cost, canonical physical tile-range groups,
unique induced synchronization edges, statically estimated synchronization
endpoint executions, and the critical-path-weighted cost. A physical group is
keyed by the unordered pair of complete placed ranges, not only by their
intersection, so distinct tile layouts remain distinct while duplicate logical
aliases collapse.

Assemble one four-arm row per `(problem, capacity, device)` and compare both
penalty models with device ordering using:

```bash
python -m pypto.tools.dsa_penalty_model_evaluation scores.tsv \
  --split development -o development-evaluation.json
```

For a new holdout, omit every latency field and pass
`--freeze-before-timing`. The command rejects any leaked device latency and
content-addresses the predictions. Cypress auxiliary-edge, relaxed-edge,
alias-pair, and packing-attempt metrics are retained as structural explanatory
features; they are never used to choose a capacity or a holdout case.

The complete functional driver is the correctness unit and every DSA function
inside it must use the same arm. A selected runtime task, or the complete mixed
AIC/AIV group containing it, is the kernel timing unit. Whole-driver latency
must not be relabelled as target-kernel latency, and runtime argument capture is
not part of this path. Before timing, use `dsa_codegen_comparability.py` to
discover nested artifacts, join replay addresses per `func.func`, reject empty
identity captures, and normalize only proven placement facts.

The freezer verifies:

- the exact PyPTO-Lib revision and source hashes;
- the original source symbol or `name_hint` for every selected function;
- direct-golden and tensor-spec entry points;
- the deterministic input seed and measurement-scope contract;
- all four algorithms at native, half, quarter, and tight capacity; and
- host feasibility without consulting objectives or device time.

On device, solve every DSA instance in one driver with the same algorithm. For
the immutable development dataset, retain its original target-pool-only
capacity convention. For the driver-first evaluation corpus, use the frozen
combined capacity profile for every child and every pool. Require the complete
Torch golden and cross-arm correctness before collecting per-task timings.
Derive the task mapping from orchestration and `kernel_config.py`; never time an
AIC or AIV half independently.
