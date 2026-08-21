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
  --screen-results <screen>/screen-results.tsv --output-root <evaluation>
```

The selector takes the least restrictive capacity where Cypress performs real
reuse and geometry first-fit, Cypress, and DSA-RP have three distinct selected-
pool placements. This excludes both the disjoint-address trivial regime and the
forced-placement regime. Cases without three-way separation remain controls;
they are not primary instances for testing policy ordering.

Do not choose a capacity because its measured ordering is favorable. The
expected paper hypothesis is `geometry > Cypress > DSA-RP` in latency, but the
capacity is fixed exclusively from the host-side structural rule above. Device
results that contradict the hypothesis are retained for penalty-model
development rather than replaced by a different capacity.

The first four-capacity development dataset is pinned by
`dsa_penalty_modeling_dataset_v1.json`. It remains immutable when the corpus is
expanded or reduced to one evaluation capacity per problem.

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
