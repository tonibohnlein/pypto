# Current paper development table

Host-only consolidation, 2026-09-07. No new placement or device timing.
This is retrospective development evidence, not a prospective test.

## Latency-guided placement search: implementation status

The historical tables below compare four algorithms. They contain **no device
measurements of latency-guided placements**. A separate research implementation
now creates such placements: `python -m pypto.tools.dsa_latency_planner`.
It starts from an explicit complete legal seed and greedily relocates allocation
classes to aligned address boundaries. Each candidate is checked against the
DSA hard constraints before scoring the **union of its physical reuse edges**:

```text
score(P) = LP(non-reusing SSA/pipe graph + E_reuse(P), w)
           - LP(non-reusing SSA/pipe graph, w)
```

The implementation uses the existing complete-placement oracle, including its
supported loop expansion. It does not invoke InsertSync, use device timings,
or rely on the penalty-candidate catalog to enumerate physical reuse. Complete
map scores are cached; the final score is verified uncached. This is bounded
greedy relocation, **not** C++ canonical greedy or incremental longest-path
maintenance. It guarantees neither a global optimum nor exhaustive address
search. Pools and structured pipeline members stay fixed; colocation classes
move together. Budget exhaustion is reported explicitly.

```bash
PYTHONPATH=python python -m pypto.tools.dsa_latency_planner \
  --problem problem.json --seed-solution geometry.solution.json \
  --objective latency --schedule schedule.jsonl --graph research-graph.txt \
  --model duration-model.json --max-evaluations 128 \
  --output-root build/latency-search
```

The fresh output directory contains `solution.json` and `search.json` with
input hashes, accepted moves, budgets, scores and duration evidence classes.
`--objective structural` uses the **same search** with the existing weighted
reuse sum. Use the same seed and budgets for that diagnostic control; otherwise
a comparison against structural canonical greedy changes both search and
objective. Do not describe it as isolating the cost model alone.

The latency objective requires complete non-fallback coverage and a single
static score. Captured runtime branch profiles, unresolved dynamic scores,
missing access provenance, input drift and unsupported geometry fail closed.
Pinned approximations remain labelled approximations; they are not promoted to
calibrated signatures. Each parent function must be handled explicitly before
a complete parent replay map can be published.

Three host integration canaries, with one global diagnostic weight of 16 cycles
and 32 score evaluations per search, produced:

| Function | Structural-search objective | Latency-search penalty (cycles) |
| --- | ---: | ---: |
| `build_bias` | 13 → 6 | 461 → 0 |
| `mtp_hidden_norm_quant` | 30 → 29 | 1017 → 168 |
| `split_pre_post` | 30 → 24 | 214 → 174 |

These use existing exports and geometry seeds, not a newly frozen 19-workload
five-arm panel. The searches exhausted their budgets. They demonstrate actual
placement selection, not device speedups or weight calibration. The host-only
reproducer is `tests/tools/run_dsa_latency_planner_audit.py`; it reads structural
artifacts only. Device correctness and balanced timing of the new maps are
still required.

### Five-arm release preparation

`tests/tools/prepare_dsa_five_arm_release.py` materializes all 95 fixed-panel
host slots without reading timings. It preserves the four baseline maps,
checks historical digests, and validates native and selected capacities with
an independent pair-scan checker. The global sync weight is 16 cycles, fixed
before search, not fitted. Each child uses a geometry-FF seed and at most 128
score evaluations. The same-search structural diagnostic is retained. A latency
parent map is published only if every child scores completely; no fallback.

At planner commit `ae16f3a93`, **5/19 workloads** produce full five-arm maps:
`dspark_o_lora_quant`, `mtp_dequant`, `mtp_hidden_norm_quant`, `build_bias`, and
`split_pre_post`. The other 14 have explicit input/model exclusions: missing
child graphs, unsupported branch/dynamic-loop scores, a `tpush` join, or a
missing `tcmp` duration. Scorable functions inside blocked parents do not make
the parent eligible. This is stricter than the older per-function bound
analysis; it is not a loss of device measurability.

The packet includes graph inputs and pins research PTOAS
`062d4b16f27f7a6baef91b5d6cfdcf6fe5f2f26f`. Scoring those graphs requires no
exporter build on the device host. An optional thin Git bundle carries that
commit on prerequisite `9d72b90ff49f67749ede982b21b30d98508028fb`; it contains no
uncommitted checkout changes. Product compilation remains official v0.57.
The external task must report partial coverage, not 19 successes.

Publication uses a file allowlist (128 MiB total, 25 MiB per file), an exact
manifest, a sidecar, and fresh extraction validation. With the published
tooling checkout, verify an extracted packet using:

```bash
PYTHONPATH=python python tests/tools/prepare_dsa_five_arm_release.py verify PACKET
```

This checks all 95 slots, every published child map against both capacity
profiles, native fingerprints, map identities and a capacity-overflow negative
control. It does not replace device replay/comparability/correctness checks.

## Tables and accounting

| Artifact | Content |
| --- | --- |
| [Primary table](data/current-paper-development/paper-primary.tsv) | All four logical algorithms, each workload's two actual device IDs, medians, complete-map identities and measurement modes |
| [Capacity manifest](data/current-paper-development/primary-manifest.json) | Structural selection and exact endpoint pins |
| [Source records](data/current-paper-development/source-records.tsv) | Raw-record integrity hashes, kept out of structural corpus identity |
| [Selection audit](data/current-paper-development/capacity-selection.tsv) | Every deterministic archived configuration, selected or sensitivity |
| [Pairwise comparisons](data/current-paper-development/pairwise.tsv) | All six algorithm pairs; nulls, adverse directions and launch-bootstrap intervals |
| [Sensitivity table](data/current-paper-development/sensitivity.tsv) | Unselected configurations, excluded from primary statistics |
| [Predictor rows](data/current-paper-development/predictor-evaluation.tsv) | Observations, predictions, coverage exclusions and mistakes |
| [Matched comparison](data/current-paper-development/predictor-matched-summary.tsv) | Predictors compared on the same DAG-score-eligible workloads |
| [All-coverage comparison](data/current-paper-development/predictor-summary.tsv) | Primary and sensitivity results, with unavailable rows explicit |
| [Hidden-norm case](data/current-paper-development/hidden-norm-case-study.tsv) | Every archived capacity/arm across the unchanged weight grid |
| [Expansion audit](data/current-paper-development/reserved-candidate-audit.tsv) | Twelve reserved rows, source families and development overlap |

The source archive has 32 measured configurations. Four are excluded for
intrinsic nondeterminism. The 28 remaining configurations contain 25 semantic
target identities but only **19 distinct driver/argument workloads**. Different
target labels within one parent do not turn its wall time into independent
kernel measurements. The primary table therefore has **19 workloads / 38 device
rows**, not 23 or 25 independently timed kernels. Nine other configurations are
sensitivity data. No measurement has been discarded because an arm is slow.

Of the primary workloads, 13 have one exported function and one DSA instance;
six are multi-function parents. The archived harness uses the **driver's device
window** in both cases. Its `effective_us` is not a per-dispatch latency. The
table labels these `SINGLE_FUNCTION_DRIVER_WINDOW` and `PARENT_PROGRAM_WINDOW`;
it does not divide parent time by submit count or pool the two strata.
Different workloads used different device pairs; actual IDs remain in the table.

Maps cover the complete driver. The chosen capacity profile tightens the
recorded target pool, with other instance/pool capacities native. A label such
as `half` is not a claim that every child pool was simultaneously halved.

## Selection and timing estimator

For each driver/argument workload, among its **already measured** configurations:

1. Prefer Cypress penalized reuse, three distinct geometry/Cypress/DSA-RP maps,
   and a positive complete-map unit-objective gap in DSA-RP's favor.
2. Maximize that gap, then realized-relation disagreement; use lower capacity
   bytes, then artifact ID, as deterministic tie-breakers.
3. Keep a control configuration if no opportunity exists. Never inspect
   latency inside the selector. Preserve every other configuration separately.

This is not a search over unmeasured capacities. Selecting on a favorable unit
objective conditions the subsequent predictor comparison: it cannot establish
unconditional accuracy on arbitrary workloads. Capacity selection is
retrospective even though its inputs are structural only.

The reported estimate is the **median of three launch medians**, with 20 device
samples in each launch. This differs deliberately from the source report's
pooled-sample median. Resampling uses whole launch medians, not 60 supposedly
independent samples. Equal complete maps share one measurement by identity.
No devices are pooled. A decided direction requires at least 2% and a 95%
launch-bootstrap interval excluding zero on **both** devices. With only three
launches and fixed arm order per device, these are exploratory decisions, not
strong confirmatory or multiple-comparison-adjusted claims.

Four selected workloads favor DSA-RP over Cypress under that rule:
`idx_qr_proj_dequant`, `mtp_hidden_norm_quant`, MTP `hc_post`, and
`split_pre_post`. Twelve are small/null and three are unresolved or
device-dependent. Full adverse and null contrasts remain in the pairwise
and sensitivity tables; no overall speedup is inferred from the win count.

## What the penalty comparison establishes

The scorer has comparable single-function bounds for **12/19** primary
workloads. Six parents need graph composition; one additional single-function
case is ineligible. A child's DAG score is not compared to a parent's latency.
Score availability is not a claim of a complete loop-expanded invocation model.
Durations include pinned approximations as well as calibrated signatures.

On those same 12 workloads, at global synchronization weights 8–64 cycles:

| Predictor | Correct decided order | Wrong order | Missed direction (tie) | Tie on small/null | Strict on small/null | Device unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unit reuse objective | 4 | 0 | 0 | 2 | 4 | 2 |
| Complete-placement DAG bound | 3 | 0 | 1 | 3 | 3 | 2 |
| Physical-range intersection count | 1 | 2 | 1 | 1 | 5 | 2 |
| Sum of pool peaks | 0 | 2 | 2 | 4 | 2 | 2 |

At weights 96–256 the DAG still misses hidden norm and adds a strict prediction
on one more small/null workload. No weight was fitted or selected here.
Strict ordering alone has no calibrated percentage magnitude, so a strict
prediction on a small effect is **not automatically a false-confident error**.

The last two predictors are **Cypress-inspired proxies**, not its complete
portfolio-selection objective. Original-allocation intersections differ from
contracted search-node alias counts, and relaxed-edge counts are unavailable.
We cannot claim to have evaluated the full Cypress objective from these proxies.

The supported conclusion is narrow: **the current DAG bound does not improve
on unit penalties in this selected development slice**. Four decided cases are
not enough for convincing leave-one-workload-out calibration, especially when
all four favor the algorithm preferred by the capacity-selection rule.

## Bounded hidden-norm study: stop here

For `mtp_hidden_norm_quant/half`, DSA-RP/Cypress changes are −8.44% and −7.82%
on devices 0 and 1 using this estimator. Unit cost falls 10→4. Nevertheless:

| Quantity | Cypress | DSA-RP |
| --- | ---: | ---: |
| Base DAG longest path | 3589 cycles | 3589 cycles |
| Placement-induced distance-zero edges | 38 | 42 |
| Placement-induced positive-distance recurrences | 12 | 4 |
| DAG penalty, every weight 8–256 | 0 | 0 |
| Emitted barrier / set / wait sites | 25 / 20 / 20 | 27 / 22 / 22 |

The common 44-node base contains 636 distance-zero edges, **604 labeled
control**. Its duration evidence is 1 calibrated signature, 14 analytical
models, 2 shape approximations and 27 pinned Perf-Sim approximations. A broad
control-ordering envelope and incomplete finite-loop composition are specific
modeling leads; neither has been shown to cause this miss. Fewer emitted sync
sites cannot explain the win either. Increasing the global reuse-edge weight
within the frozen grid does not repair it.

Do not fit a hidden-norm-specific constant, change its placement, or launch a
causal campaign as a prerequisite for the paper. Retain it as a documented
model miss and proceed with expansion and writing.

## Prospective expansion, independently of modeling

The twelve reserved rows contain six source families: cache write (1),
build bias (4), merge/rope pack (4), KV matmul (1), KV RMS/rope (1), and
Q-rope preparation (1). Build bias and Q-rope preparation already have measured
development relatives. Numbered helper copies are not proven new workloads.

Next, isolate the original helpers with deterministic inputs and direct Torch
references. Record source-body, shape, tiling, dtype and semantic problem
identity; collapse identical shapes/code despite different names. Group related
variants together and label development-family overlap. Supplement with new
families if needed to reach 8–12 genuinely new runnable workloads. Register
supplementary candidates before timing; never replace candidates after seeing
performance. Null controls are useful and must not become correctness failures.

Device correctness and a timing-blind capacity freeze precede prospective
timing. The existing four algorithms remain unchanged. Model-ineligible cases
remain device-measurable; frozen missing predictions are reported as such.
Neither hidden norm nor a minimum model-coverage count blocks this expansion.
The next remote task is a separate preparation/validation/freeze workflow,
not authorization to call these twelve rows an eight-workload holdout today.

## Reproduction

```bash
python tests/tools/consolidate_dsa_paper_development.py \
  --analysis-root build/dsa-paper-current-analysis \
  --output-root docs/en/dev/proposals/data/current-paper-development
```

The source device archive is
`dsa-rp-rebased-corpus-device-continuation-3eabcfd22-final.tar.gz`, SHA-256
`a797b5b0d93fd963980197df8559c740e2ca05c4251e500dde39728e17e969e7`.
The tool verifies frozen table hashes and the prediction seal, uses reconstructed
maps, and records source-row hashes separately from the structural capacity
manifest. The compact checked-in tables are derived evidence, not a replacement
for that archive or its raw launch records.
