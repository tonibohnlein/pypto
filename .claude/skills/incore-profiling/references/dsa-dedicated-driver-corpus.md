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
