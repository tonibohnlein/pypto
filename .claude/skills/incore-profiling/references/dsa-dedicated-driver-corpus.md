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
reports both the original unit cost and the critical-path-weighted cost.

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
not part of this path.

The freezer verifies:

- the exact PyPTO-Lib revision and source hashes;
- the original source symbol or `name_hint` for every selected function;
- direct-golden and tensor-spec entry points;
- the deterministic input seed and measurement-scope contract;
- all four algorithms at native, half, quarter, and tight capacity; and
- host feasibility without consulting objectives or device time.

On device, solve every DSA instance in one driver with the same algorithm. At a
target's reduced capacity, only that target pool is tightened; all sibling
instances retain native capacity but are still solved by the same algorithm.
Require the complete Torch golden and cross-arm correctness before collecting
per-task timings. Derive the task mapping from orchestration and
`kernel_config.py`; never time an AIC or AIV half independently.
