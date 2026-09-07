# Device task: bounded expansion beyond the current development corpus

## Objective and scope

Prepare and validate dedicated drivers from the twelve reserved candidates,
then freeze one structural capacity profile per surviving new workload and
measure geometry first-fit, geometry canonical greedy, Cypress, and DSA-RP
canonical greedy on two devices. Aim for 8–12 new workloads. Never turn numbered
copies of one helper into independent workloads without a distinct shape or
tiling proof. No new planner or duration model is part of this task.

Model coverage, hidden norm and a workload-count minimum must not prevent
reporting valid device results. If fewer than eight survive the bounded search,
freeze and measure those survivors, report `EXPANSION_TIMED_PARTIAL`, and state
the shortfall. Do not claim that the eight-workload objective was achieved.

This task uses already-published endpoint revisions and verified campaign
archives, not uncommitted local consolidation tooling. Fetch exact SHAs; no
branch-tip substitution. Run the repository's in-core profiling skill where
applicable, with this task's identity, selection and storage rules taking priority.

## Frozen endpoint sources and inputs

| Component | Revision / identity |
| --- | --- |
| PyPTO | `3eabcfd22894151cbda6c752dcb300708a34f28d` |
| PyPTO-Lib | `83e19f6b06eb2125eb14c06232358f342941c8d5` |
| runtime gitlink | `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| PTO-ISA | `a8040450238f162985d8b596fbebeb54bfba2bf5` |
| dsa-solver | `60ac39b02b008dc3907b876ad47a1cc342aa01fa` |
| Product assembler | official PTOAS v0.57 wheel, SHA-256 `4858c837e12b1b588f281207c95916dc20968a7cdc656aadd549a87658a06692` |

Verify sidecars and internal manifests before reading either archive:

- `/opt/pypto/dsa-rp-rebased-corpus-expansion-3eabcfd22-final.tar.gz`,
  SHA-256 `5ffabae8ef1da34e0eceb80bfac3e0ceff1369bed3397d00271d25f7a6b631af`.
- `/opt/pypto/dsa-rp-rebased-corpus-device-continuation-3eabcfd22-final.tar.gz`,
  SHA-256 `a797b5b0d93fd963980197df8559c740e2ca05c4251e500dde39728e17e969e7`.

These archives supply discovery, complete-map and validation harness evidence.
Their existing timings are **development** data. Do not open their timing
tables to choose candidates, shapes, capacities, seeds or predictions.
Record toolchain binaries, build flags and interpreter imports. Reuse a proven
same-pin build; do not rebuild unrelated toolchains or all production parents.
Choose an explicit build worker count appropriate for the remote machine.

## 1. Bounded driver construction and novelty audit

Start from the archived `NEEDS_ISOLATED_DRIVER` rows:

| Source script | Reserved instances | Family / caveat |
| --- | --- | --- |
| `models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py` | `prefill_c4_cache_write` | Previously structurally null; retain as control if so |
| `models/deepseek_v4_flash_dspark/prefill_sparse_attn.py` | `build_bias`, `_0`, `_1`, `_2` | One helper family, already has development relatives |
| same | `merge_rope_pack`, `_0`, `_1`, `_2` | One helper family; avoid the unrelated `proj_a_mm` sibling |
| `models/deepseek_v4_pro/qkv_proj_rope.py` | `kv_proj_matmul` | Distinct maps but no previous unit-objective gap |
| same | `kv_rms_norm_rope` | Avoid the unrelated `qproj_matmul_aic` sibling |
| same | `q_rope_prepare` | Already has development relatives |

Before compiling or timing, write `candidate-registry.tsv` with source helper,
full source revision, input/output shapes, dtype, tiling, launch geometry,
golden contract, source-body digest and family. After exporting, add canonical
solver-field and semantic problem identities. Generated names alone are not
identities. Compare against the archived measured drivers and structural
manifests; matching driver/helper/shape configurations are development replays,
not new holdout members. Report family-overlap variants separately from
family-unseen workloads. Do not claim complete historical novelty when the
available structural records do not establish it.

Use the original inline helper when callable. Otherwise isolate its original
body with a documented source-to-driver correspondence. Preserve operations,
real model shapes and math; write a direct independent Torch golden and fixed
inputs. Do not use post-parent captured buffers as a kernel reference. Permit
only the wrapper, deterministic input construction and independent golden to
change. Put new driver source under the campaign, not in shared source trees.
Any semantic kernel edit is outside scope: classify that candidate blocked.

If the deduplicated pool cannot provide eight new workloads, predeclare at most
eight supplementary helper/shape candidates from these pinned sources before
reading any new timing. Prefer missing operation families; genuine real-model
shape/tiling variants are allowed but must be grouped by family. Record their
selection reasons using structural information only. This is the only allowed
replacement round. Do not inflate counts or continue an unbounded search.

Report progress after the first three driver qualifications, including blocked
reasons and live storage. Reuse exports and solutions between subsequent gates.

## 2. Host screen and device correctness

For each driver, export every child problem and solve all four policies at
native/half/q1/tight. Use the checked-in
`.claude/skills/incore-profiling/screen_dsa_capacity_corpus.py` capacity helpers
to apply the profile to all pools of all children simultaneously. Native must
retain the original semantics. Never silently use single-pool tightening as a
simultaneous profile. Preserve the full per-pool capacity vector.

Use the frozen screener's policy settings; verify seed/restart settings and
record the Cypress portfolio and chosen variant. Cypress selection must be
latency- and penalty-weight-blind. Save actual alias count, relaxed-edge count,
peak, order and seed so later analysis need not substitute proxies.

Independently validate every complete map against both the derived problem and
native replay problem. Fingerprint retargeting, if necessary, changes metadata
only and must be logged. Exercise capacity/alignment/temporal-overlap/hard-edge
negative controls. No invalid or missing sibling may pass by omission.

Run each driver's stock planner first. Stock golden or execution failure is
`DRIVER_RUNTIME_BLOCKED`; nondeterminism on identical inputs is separately
recorded. Exclude these from placement comparisons, without repairing math or
tolerances. For stock-runnable drivers:

- Prove replay consumption, all placement identities, late-elimination
  provenance, function/kernel/dispatch inventories, ABI, scalars and launch
  geometry; reject empty captures or zero-denominator proofs.
- Normalize only proven nonsemantic names and tile addresses. Changed op,
  dtype, shape, semantic constant, missing solution and wrong-arm controls
  must fail. Discover artifacts recursively.
- Group byte-identical complete maps into one physical endpoint, retaining all
  four logical policies. One algorithm selects **all child placements**; no
  target-only parent comparisons.
- Correctness-check every feasible distinct endpoint at every candidate
  capacity on one device. For the primary capacity later selected, require
  three deterministic launches on each of two devices, reversed arm order,
  original golden and full validated-buffer comparison on identical inputs.

An arm-specific placement correctness failure stops that workload and preserves
a compact reproducer; investigate whether it is shared infrastructure before
continuing others. Mere unsupported modeling never removes a device workload.

## 3. Structural freeze, before timing

Use only correctness/feasibility, complete-map and objective records. Prefer
capacities where Cypress realizes penalized reuse, geometry/Cypress/DSA-RP maps
are all distinct, and DSA-RP's complete-map unit objective is strictly lower.
Maximize the Cypress-minus-DSA-RP gap, then penalized-relation disagreement,
then all-relation disagreement; tighter capacity is the final tie-breaker.
Use the checked-in `select_dsa_workload_capacity.py` under the profiling skill
where its schema applies. Do not claim to run it if a campaign adapter is used;
test that adapter and preserve it.

Workloads lacking an opportunity remain explicit null/control rows, outside the
opportunity subgroup but inside the measured corpus. Select one capacity per
workload; other capacities are correctness/sensitivity records, not independent
workloads. If more than twelve new workloads survive, select by operation/source
family diversity, then deterministic structural keys, never performance.

Write `cohort-frozen.json` twice and require identical semantic bytes. Include
driver source hashes, all component pins, solver settings, seeds, complete
capacity vectors, map digests, measurement modes and every exclusion. Timing
and solver wall-clock fields must not participate in the identity.

Freeze predictions alongside it: unit objective, actual Cypress portfolio
metrics, and DAG scores **only where an already-available, pinned analyzer
produces a scope-matched complete-placement bound with product-faithfulness
proof**. Official v0.57 lacks the research graph CLI. Do not install/build a
new research toolchain or implement graph repairs for this task; mark missing
DAG predictions `MODEL_UNAVAILABLE` before timing. A child bound is not a
parent prediction. No new synchronization weight or duration fitting.

## 4. Balanced timing on two devices

At most two device lanes, one timing process per device. Each lane needs its
own build/run/output directory; shared `dfx_outputs` can corrupt evidence.
Use the driver's device `effective_us` for a single-function driver. Prefer
isolated single-kernel drivers. For an unavoidable multi-function driver,
report `PARENT_PROGRAM_WINDOW` and apply each policy parent-wide; never divide
by dispatch count or relabel it as kernel latency.

Assert nonzero device timing and no host fallback. Use eight balanced blocks
per distinct endpoint per device, with three warmups and twenty timed samples
per block. Precompute the order and reverse/counterbalance it on device two.
Include a same-binary repeated-label null control in a representative cell.
Do not add a broad second timing sweep: all primary rows already use both
devices. No instruction/capture profiling unless a timing-integrity gate fails.

Keep devices separate. Bootstrap at the balanced-block level, preserve raw
samples, and report all ratios, CIs, physical nulls, regressions and ambiguous
results. Report sign, magnitude and uncertainty separately. Use a 2% magnitude
and CI-excludes-zero rule on both devices for the predeclared decided subgroup;
also show the continuous results, not only winners. Do not keep launching to
move a result across the threshold.

Evaluate predictions on matched coverage and on all workloads, with explicit
unavailable/missed/wrong/null counts. Do not treat strict score order as a
calibrated percentage prediction. Present opportunity/control and familiar/new
source-family strata separately. No pooled kernel/parent geomean.

## 5. Storage, deliverables and finish

Keep campaign files under `/opt/dsa-rp-development-expansion/`. No `/tmp`
archives, build trees or captures. Prune each completed compile/run after
extracting verified evidence. Keep code, maps, small provenance records and raw
timing; passing tensors and bulk `args.bin` are unnecessary. Keep at most one
lean failing input/output reproducer, capped at 128 MiB.

Write `REPORT.md`, `HANDOFF.md`, `PINS.md`, registry/novelty/exclusion tables,
feasibility and complete-map tables, correctness/comparability/negative-control
tables, structural selection audit, frozen cohort/predictions and hashes,
raw timing, pairwise results and matched predictor evaluation. Include the new
driver source and campaign adapters required to reproduce them. Report exact
workload/shape/family counts; no renaming targets into independent kernels.

Stage by explicit allowlist into
`/opt/pypto/dsa-rp-development-expansion-final.tar.gz` plus SHA-256 sidecar and
internal manifest. Enforce ≤256 MiB staged data, ≤25 MiB per ordinary file,
zero symlinks/build trees/binaries/caches/venvs before compression. Extract
fresh and verify exact manifest coverage and hashes. Preserve input archives.

Use `PROSPECTIVE_EXPANSION_COMPLETE` only for 8–12 genuinely new timed
workloads; otherwise report the actual partial funnel and valid results.
Never substitute `COMPLETE` for a blocked build or empty table. Leave source
trees clean, processes stopped and devices idle. The hidden-norm case and
paper drafting remain independent of this task.
