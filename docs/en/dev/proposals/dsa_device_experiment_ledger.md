# DSA-RP Device Experiment Ledger

Last consolidated: 2026-08-31.

## Purpose

This ledger is the durable index for DSA-RP device evidence. It records failed
infrastructure campaigns as well as positive results so that a later analysis
does not silently reuse a superseded number. The archive named in each row is
the source of record; conclusions must be re-read from its tables rather than
from this summary. Percentages use the archive's own sign convention, normally
candidate/baseline minus one, so negative means faster.

The campaigns used different corpora and measurement strata. Standalone-kernel,
single-submit driver, per-task swimlane, and whole-parent measurements must not
be pooled. Capacity sensitivity rows are not independent workloads.

## Campaigns that reached device execution

| Campaign archive | Device evidence | Primary result and present interpretation |
| ---------------- | --------------- | ----------------------------------------- |
| `dsa-rp-final-pr-bounded-final.tar.gz` | PR system-test lanes | The original PR failed on `test_dyn_orch_paged_attention`: DSA-RP required a top-level `SeqStmts`. Performance was correctly not run. |
| `dsa-rp-final-pr-bounded-ef581ae7-final.tar.gz` | 39 tests, two lanes; bounded timing | The compile precondition was fixed; 39 passed and 2 skipped. Timing was inconclusive. |
| `dsa-rp-pr-large-kernel-overnight-ef581ae7-final.tar.gz` | 60 timed standalone endpoints, two-device confirmation | 13 confirmed wins across six programs and no confirmed regression; the largest was `q_lora_rmsnorm` at about 28.7%. Coverage was limited by argument reconstruction. This is exploratory, not the final paper corpus. |
| `dsa-rp-paper-three-arm-broad-screen-final.tar.gz` | 20 confirmed kernels on four devices | DSA-RP beat geometry first-fit by 16-19% on every device. DSA-RP versus Cypress did not replicate: three devices were non-significant and directions varied; one device showed about 1.8% for DSA-RP. |
| `dsa-rp-paper-81-four-capacity-screen-final.tar.gz` | 19 timed capacity cells, only four kernels at all four capacities | A pilot only. One tight Cypress cell diverged numerically, and capture/reconstruction failures dominated coverage. |
| `dsa-rp-paper-81-four-capacity-parent-partial.tar.gz` | 40 parent cells reached terminal status | Seven arm-specific golden failures exposed an unsafe placement class. Parent-wide timing was not suitable for paper claims until correctness was resolved. |
| `dsa-rp-paper-correctness-then-broad-native-final.tar.gz` | seeded failure reproducers and placement ablations | Isolated deterministic MTP and HCA failures; the initial missing-handoff hypothesis was later retracted. |
| `dsa-rp-paper-partial-overlap-sync-causal-v2.tar.gz` | exact overlap ladder on device | Proved staggered source/destination overlap inside one instruction corrupts data. Sync schedules were identical; the defect was in DSA lifetime/alias construction, not InsertSync. |
| `dsa-rp-current-corpus-device-screen-fced3c67-final.tar.gz` | 106 native target-only parent cells; priority half-capacity cells | One confirmed geometry-vs-DSA-RP regression and four correctness failures. The latter were a second lifetime-construction bug, later isolated as interior containment after an allocation ended too early. |
| `dsa-rp-softmax-pool-correctness-causal-final.tar.gz` | exact relation ladder and maximal-barrier controls | Proved non-zero-offset containment caused corruption while equal-base containment passed. Maximal synchronization did not repair it; the lifetime root mapping was incomplete. |
| `dsa-rp-paper-canary-v054-9b59f244-final.tar.gz` | 21 comparable endpoints, two devices | MTP failures closed after the first lifetime repair. `prefill_c4_softmax_pool` still failed under geometry and DSA-RP, motivating the chained loop-return fix. |
| `dsa-rp-loop-return-softmax-canary-48053d242-final.tar.gz` | 24 successful `prefill_c4` runs across four arms and two devices | The chained loop-return lifetime fix made every `prefill_c4` arm bit-identical. The second parent was stock-runtime blocked by heap-ring deadlock and was excluded from placement claims. |
| `dsa-rp-paper-broad-stock-gated-536fed244-final.tar.gz` | 315 correct target-only parent cells; 1,185 comparisons | Six comparisons confirmed, but changing one kernel in a parent diluted effects and did not satisfy the intended corpus design. Its principal contribution is the stock-first correctness funnel and compact archive discipline. |
| `dsa-rp-current-minimal-drivers-db5a6dcf-final.tar.gz` | three direct drivers plus one parent control, four arms, two devices | Direct RMSNorm and HC-post drivers showed roughly 8-21% penalty-aware wins over geometry; DSA-RP and Cypress were within noise. Established that minimal real-golden drivers are measurable. |
| `dsa-rp-full-corpus-parent-dispatch-screen-final.tar.gz` | 19 targets at all four capacities, parent/task strata | DSA-RP had nine confirmed wins over geometry; Cypress had fourteen. DSA-RP did not systematically beat Cypress and reduced capacity did not widen their separation. |
| `dsa-rp-dedicated-driver-corpus-886b52614-final.tar.gz` | 64 problem-capacity cells, four arms, two-device confirmation for large effects | 30 cells chose DSA-RP as fastest, 19 Cypress, 15 geometry variants. Every effect at least 10% reproduced; 2-5% effects were often device-sensitive. This is development dataset v1, not a holdout. |
| `dsa-rp-weighted-dag-device-validation-2e027d131-final.tar.gz` | 48 logical cells, 9,920 samples | The first complete weighted-DAG test did not beat unit reuse cost. Unit cost was 14/14 on device-decided orders; the DAG made two false-confident errors. |
| `dsa-rp-loop-aware-model-prospective-0820ab418-final.tar.gz` | 51,840 launches on two devices | Zero of 12 Cypress-vs-DSA-RP cells cleared the preregistered effect gates. Frozen sync features were identical on the useful near-misses, so this corpus was insufficient to validate them. |
| `dsa-rp-hc-post-exposed-wait-final.tar.gz` | native Cypress/DSA-RP HC-post on two devices | The MTP regression premise did not reproduce. DsPark, intended as a null, showed a repeatable DSA-RP slowdown, but chip swimlanes exposed no intra-kernel waits. |
| `dsa-rp-dspark-sync-address-ablation-final.tar.gz` | five legal DsPark endpoints, two devices | The DsPark slowdown again failed to reproduce. A restored PTO handshake compiled to a byte-identical device binary and therefore was only a same-binary noise control. No causal effect was resolved. |
| `dsa-rp-driver-first-corpus-verification-0fe01d2e-final.tar.gz` | 20 workloads, 320 logical cells, 419 correctness runs | All 20 workloads passed all four capacities and four logical policies with no placement-correctness failures. This established the first robust driver-first corpus; no timing was taken. |
| `dsa-rp-driver-first-timing-8d8e76df-final.tar.gz` | 19 of 20 primary workloads on two device domains | Tight-capacity selection produced many physical nulls. Only `dspark/rmsnorm.py` confirmed DSA-RP over Cypress, about 2.6-3.6%. Native-map injection controls proved the timing instrument could resolve 8-33% effects in the same cells. |
| `dsa-rp-replay-fixed-prospective-holdout-9b05800db-final.tar.gz` | eight frozen workloads, nine targets, 13,440 samples | DSA-RP beat Cypress on five targets and was never confirmed slower: about 3.2-6.1% on `rmsnorm_rope`, HC-post-prefill, split-pre-post, cache-write, and KV/cache-write. This is the strongest prospective policy result. |
| `dsa-rp-four-candidate-physical-penalty-aeba32c70-final.tar.gz` | four new workloads, 6,400 samples | `kv_score_proj_c128` confirmed DSA-RP over Cypress by about 2.3-2.5% despite more sync sites; Gumbel's large unit-cost gap was a latency null. Critical-path coverage was incomplete. |
| `dsa-rp-kv-gumbel-legal-ablations-2bdc441b0-final.tar.gz` | five relation families, address controls, 6,400 samples | Restoring Gumbel `(2,39)` both removed one static ELSE-arm barrier and improved latency by 2.1-2.35%; this established a causal placement-relation contrast, not that the barrier caused the speedup. `(38,42)` cost about 1.9%, `(3,38)` about 0.4%, and `(38,79)` was null. KV `(8,22)` was not causal; all translation controls were null. |
| `dsa-rp-queue-event-model-device-validation-9217dd575-final.tar.gz` | 16 endpoints, 6,400 samples, 32 branch-profile swimlanes | The real 6-THEN/2-ELSE profile makes the `(2,39)` barrier marginal exactly zero, while the measured improvement is carried mainly by the long THEN blocks. A per-pipe barrier constant was rejected by a 4.40x calibration spread. The barrier-causality question remains unresolved pending a placement x barrier factorial. |
| `dsa-rp-gumbel-barrier-factorial-ff5d8121e-final.tar.gz` | placement-by-barrier factorial, branch-profile controls, two devices | Runtime execution of the node-49 barrier is not causal: the approximately 2.1% effect persists when its branch never executes and disappears when it executes in every row. A source-level barrier toggle causes broad downstream binary differences, but no scheduling, register-allocation, layout, or encoding mechanism was isolated. The paper-safe result is a compiler-mediated non-local binary change with exact mechanism unresolved; this contrast is excluded from barrier-cost calibration. |

## Campaigns that deliberately stopped before useful timing

| Campaign archive | Stop reason | Lasting result |
| ---------------- | ----------- | -------------- |
| `dsa-rp-paper-canary-9b59f244-final.tar.gz` | frozen PyPTO and PTOAS revisions were incompatible at level 3 | Established the explicit-`tmp` toolchain mismatch; no placement conclusion. Re-run with the declared stable PTOAS produced the later v0.54 canary. |
| `dsa-rp-standalone-four-arm-pilot30-ec4151192-final.tar.gz` | captured parent completion state was not a per-dispatch golden | All distinct endpoints agreed with each other, proving a reference-reconstruction defect rather than a placement failure. No timing. |
| `dsa-rp-standalone-harness-canary-final.tar.gz` | runtime capture was not task-adjacent | Proved the selected `topk_select` task writes only one row while the archived completion snapshot includes later tasks. This led to the dedicated-driver strategy. |
| `dsa-rp-four-arm-capacity-corpus-f8ff87b04-final.tar.gz` | fewer than 20 model-eligible standalone cases | Showed that compile-only exports lack sufficient dispatch ABI and that static loops had been wrongly treated as control-flow exclusions. |
| `dsa-rp-prospective-opportunity-holdout-fd228dbc1-final.tar.gz` | only three of eight survivors had a structural opportunity capacity | Proved eleven proposed drivers were mathematically infeasible, not canonical-greedy failures. No latency was unsealed. |
| `dsa-rp-prospective-driver-validation-021354ee-final.tar.gz` | only five non-development workloads survived | Found replay-address checking treated late-eliminated allocations as missing. The provenance checker was repaired before the successful holdout. |
| `dsa-rp-critical-path-penalty-holdout-f9a7e00f-final.tar.gz` | PTOAS divergence and control-flow rejection | Extracted 7,712 realized reuse rows and showed unit cost's errors concentrated on `gate`; no predictions were frozen. |
| `dsa-rp-pto-isa-duration-calibration-39cfb942d-final.tar.gz` | only three model-v0 problems and no buildable Perf-Sim | Quantified exact duration coverage as at most 14.75% and exposed missing operation metadata. |
| `dsa-rp-pto-isa-calibration-canary-7cf5960ea-final.tar.gz` | pinned Perf-Sim source did not compile | Repaired types, transfer work sizes, and matmul joining; documented the remaining scalar-constant gap. |
| `dsa-rp-weighted-dag-calibration-5d30202c-final.tar.gz` | unsupported signatures prevented any complete score | Perf-Sim and the product-faithful exporter first ran end to end. Unit reuse cost remained a strong archived predictor; weighted-DAG was not computed. |
| `dsa-rp-weighted-dag-device-validation-final.tar.gz` | 34 of 75 nodes lacked exact/pinned durations | `topk_select_inactive` was the first complete existence proof; no device was touched. |

## Current evidence boundary

The prospective eight-workload holdout supports the claim that a structured
penalty-aware DSA-RP policy can beat Cypress on real kernels. It does not yet
validate the current learned/analytical penalty model. The legal ablations show
why: pair cost is contextual and signed, and exposed latency depends on the
complete post-InsertSync graph rather than relation or barrier counts alone.
The next model evaluation must freeze signed post-InsertSync marginals before
opening held-out timing.
