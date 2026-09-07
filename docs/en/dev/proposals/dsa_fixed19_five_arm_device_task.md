# Device task: fixed 19-workload, five-arm DSA comparison

## Status: DRAFT — NOT READY TO DISPATCH

This specifies the experiment now; it is not a claim that the fifth planner
already exists. The current checkout provides a complete-placement DAG scorer,
not a latency-guided placement search. Do not provision/build a campaign or
open a device until the release packet below is published and passes the
source gate. Missing implementation is SOURCE_GATE_BLOCKED, not a corpus or
duration-coverage result. Do not implement the planner on the device machine.

## Objective

Measure the same fixed workloads under five logical algorithms:

| Arm | Placement objective |
| --- | ------------------- |
| geometry_ff | Geometry first-fit |
| geometry_cg | Geometry canonical greedy |
| cypress | Frozen, timing- and penalty-weight-blind relaxation portfolio |
| dsa_rp_structural | Existing canonical greedy with structural reuse penalties |
| dsa_rp_latency | Experimental canonical greedy with complete-placement DAG marginal |

The latency objective is
`p_w(P) = LP(G0 + E_reuse(P), w) - LP(G0)`.
Start from a non-reusing SSA/pipe dependency graph. Add all directed,
placement-induced reuse dependencies together, including their synchronization
weight; do not sum independent pairwise longest-path deltas. The search marginal
is `p_w(P_candidate) - p_w(P_current)` under the frozen partial-placement
contract. Preserve legal lifetime, alias, capacity and alignment constraints.

This is development-set comparison, not an unseen-workload holdout. Repeated
measurements do not restore blindness to the previously observed results.
No corpus expansion, new capacities, instruction profiling, simulator campaign,
model fitting or causal ablations are authorized here.

## 0. Release packet: local prerequisites, before dispatch

The owner must publish a fetchable release with:

- Full committed/pushed PyPTO analysis-tooling, experimental planner/dsa-solver,
  and research PTOAS SHAs, plus tested invocation commands. Do not substitute
  an unpushed checkout, branch tip, undocumented flag or private-function bypass.
- A model JSON containing duration-provider source hashes, evidence classes,
  one global synchronization weight in cycles, loop/branch semantics,
  recurrence handling, partial-placement evaluation, tie-breaking and
  parent-composition policy. Freeze it before new timing. No per-kernel tuning.
  Generic pinned approximations must not be called calibrated signatures;
  missing durations or provenance must not silently become zero.
- The 19-row structural manifest, all child problem documents, complete
  per-pool capacity vectors, five-arm maps, independent validation reports,
  baseline-map identity proofs and model-eligibility reasons.
- Planner tests proving that the latency objective participates in candidate
  selection, that combined-edge interactions are scored, and that cached
  updates match full recomputation on legal placements. Merely scoring the
  structural planner's output is not the fifth algorithm.
- An input archive with SHA-256 sidecar and internal manifest; include all
  reproducibility sources and actual placements, not map reports alone.
- A release JSON pinning these files and the exact commands to reconstruct,
  compile, verify, freeze and time them. Publish the archive's name and full
  SHA-256 in the dispatch copy of this task.

These fields are intentionally not assigned speculative values in this draft.
The dispatcher must mark this task READY only after remote fetchability and a
small end-to-end host smoke test are established. The device agent reports
SOURCE_GATE_BLOCKED immediately if this packet is missing or the latency arm
is absent. Do not spend a multi-day run rediscovering that fact.

## 1. Fixed workload and endpoint identity

The authoritative panel is
`docs/en/dev/proposals/data/current-paper-development/primary-manifest.json`
at PyPTO tooling commit `11c70802312700e18476ae268d6b5569f7400a0b`.
It contains exactly 19 driver/argument workloads. The companion
`paper-primary.tsv` gives the existing measurement modes, not new selection
criteria. There are 13 single-function driver windows and six parent-program
windows. Target names and capacities are not additional corpus members.

Use the existing selected capacity for each row. Preserve its full vector:
the historical convention tightens the selected target pool and leaves other
pools/children native. Do not change this to simultaneous all-pool tightening.
Keep the other capacities and the latest four expansion workloads outside this
task. Do not replace a difficult, null or adverse row.

Frozen product endpoint pins:

| Component | Revision / identity |
| --------- | ------------------- |
| PyPTO | `3eabcfd22894151cbda6c752dcb300708a34f28d` |
| PyPTO-Lib | `83e19f6b06eb2125eb14c06232358f342941c8d5` |
| runtime | `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| PTO-ISA | `a8040450238f162985d8b596fbebeb54bfba2bf5` |
| Existing solver baseline | `60ac39b02b008dc3907b876ad47a1cc342aa01fa` |
| Product assembler | official PTOAS v0.57 wheel, SHA-256 `4858c837e12b1b588f281207c95916dc20968a7cdc656aadd549a87658a06692` |

The published experimental planner may run out of tree and emit replay maps;
all device arms must compile with the same product toolchain above. Any required
product compiler change requires a revised release and common-pin regeneration
of all arms, not a private fix to only one arm.

Use original drivers, deterministic inputs and their own Torch goldens.
For parents, each policy selects placements for every child; no target-only
placement substitution. The latency arm must cover every placement-relevant
child, not mix in structural or geometry fallback children. Trivial children
may have a proven zero objective. Report per-function scores separately from
any scope-matched parent prediction: summing child penalties is not proof of
a parent critical path. Freeze any separable parent search aggregation as such.

## 2. Host/source gate and stock preflight

Verify archives before using contents; record exact revisions, remote URLs,
clean state, submodule pins, binary hashes, model hash, CANN/compiler versions,
device type, commands and environment. Use a proven same-pin installation.
Follow the checked-in in-core profiling skill only where relevant; these five
arms and fixed workload rules override its older three/four-arm examples.

Confirm all 95 workload/arm slots have a terminal host status. Independently
validate maps, including zero-penalty siblings and native replay legality.
Reconstructed existing arms must match their frozen semantic problem and
complete-map digests. Canonicalize only proven nonsemantic SSA labels; raw JSON
byte drift alone is not a changed solver problem. Fingerprint retargeting must
be metadata-only, logged and checked against both capacity profiles.

Record Cypress's actual portfolio, chosen variant, aliases, relaxed edges and
packing attempts. Preserve the frozen baseline settings. Structural and latency
CG should use matched ordering, seeds, restarts and search budgets as declared
in the release; record any necessary algorithmic difference. Report solver
time/evaluations separately, never include them in device latency.

Run stock once per workload with deterministic repeated correctness launches.
Stock failure or intrinsic nondeterminism blocks that workload's comparison,
not the entire corpus. Retain its row and reason. No tolerance or seed tuning.

If the implementation exists but some workloads are MODEL_INELIGIBLE or the
fifth arm cannot place legally, retain these statuses explicitly. Do not claim
a complete five-arm experiment. Continue valid five-arm workloads; do not spend
device time on a new four-arm-only sweep. Preserve their existing baseline
results separately, not as measurements of the missing arm.

## 3. Replay, comparability and correctness

Compile one physical endpoint per identical complete map. Keep all five
logical labels. Different maps can share execution only with independently
verified equivalent executable/ABI identity; source counters alone are neither
a semantic difference nor a sufficient equivalence proof.

For every endpoint prove:

- Compiler replay consumption and per-function placement equality.
- Every expected allocation is emitted, a valid view, or late-eliminated with
  compiler provenance. Address containment alone cannot prove replay.
- Identical function/kernel inventories, submit counts, ABI, scalar arguments,
  block geometry and normalized pre-InsertSync operation streams across arms.
- Nonempty recursive artifact discovery; no zero-denominator or missing-child
  proof can pass vacuously.
- Product InsertSync is enabled; actual post-InsertSync summaries are retained.
  An analysis exporter must prove product-faithful operation/sync output before
  its graph is used. Official v0.57 lacks the research graph CLI.

Mutation controls must reject a changed op, dtype, shape, semantic scalar,
missing solution, wrong-arm map, wrong function/graph identity and an illegal
placement. Address-only normalization must leave non-address constants visible.

Run each distinct endpoint three times on each of two devices, restoring
identical input buffers/scalars. Require within-arm determinism, the unchanged
full parent golden and cross-arm full-output bit identity on this deterministic
panel. If the source contract permits numerical variation, document it rather
than silently relaxing the frozen panel's requirement.

A placement-specific correctness failure stops that workload and preserves a
lean reproducer; check for shared infrastructure failure before attribution.
Continue independent workloads only when the harness remains trustworthy.

## 4. Freeze, then balanced device wall-time measurement

Before the first timing launch, write the complete 19-row status manifest,
five-arm map/executable identities, capacities, eligibility, model configuration,
structural and DAG predictions, seeds, device IDs, measurement modes and planned
orders. Generate twice and require identical semantic bytes. Never consult
historical latency to change capacities, model parameters, maps or order.
Keep already-known historical results labelled development data.

Use two quiet, compatible devices, one timing lane per device. Every lane has
private build/run/output directories; never share writable `dfx_outputs`.
Select an explicit build worker count appropriate for the remote machine.
Reuse builds and prune sequentially; no unbounded parallelism.

Measure device elapsed wall time over the same frozen driver window across
all arms. The single-function and parent strata remain separate. Do not time
host compilation/golden work, sum per-core times, use one mixed-kernel half or
divide a parent window by dispatch count. Assert nonzero device-domain samples,
no host fallback and correct runtime-span names. Prove the timed binary is the
one that passed replay and correctness.

Use a validated Williams/counterbalanced design over distinct physical endpoints,
with at least eight complete blocks per device rounded up to a full design
cycle (five distinct endpoints require the ten-sequence odd-arm design).
Verify position and directed carryover counts mechanically. Counterbalance
device two. Each endpoint occurrence gets three warmups and twenty timed samples.
Preserve input reset semantics. Do not continue sampling until significance.

Include one same-binary repeated-label control and a small independent,
known nonzero instrument control when available from frozen inputs. If a timing
integrity check fails, quarantine and repeat only the affected blocks with
documented cause; do not discard slow but valid blocks.

No new broad capacity sweep or chip trace collection. Save concise actual
post-InsertSync summaries and executable hashes for every physical endpoint.

## 5. Analysis and verdicts

Report all ten logical pairwise contrasts per five-arm workload, separately
by device: medians, candidate/reference ratios, percent latency change
`100 * (candidate / reference - 1)`, and paired balanced-block bootstrap CIs.
Positive means slower. Share identical endpoints as PHYSICAL_NULL, not as
independent evidence. Do not pool devices or treat samples within a block as
independent launches.

Predeclare a decided effect as at least 2% with a 95% CI excluding zero and
the same direction on both devices. Also publish continuous estimates, smaller
effects and discordance. Keep nulls and regressions. Report aggregates separately
for single-function and parent strata on identical coverage sets; counts of
targets/capacities must not inflate workload sample size.

Primary contrasts: latency versus structural DSA-RP; each versus Cypress;
each versus geometry FF. Geometry CG is a search control. Compare predictor
orderings on matched model coverage and report misses, ties, wrong directions
and unavailable rows. A DAG tie cannot be called a calibrated latency null;
a strict cycle ordering is not automatically a percentage-speedup prediction.

- FIVE_ARM_TIMING_COMPLETE: 19/19 workloads pass all five arms and both devices.
- FIVE_ARM_TIMING_PARTIAL: at least one valid five-arm row measured, with every
  other fixed row explicitly classified.
- NO_FIVE_ARM_TIMING: no full comparison could be measured; give reasons.
- SOURCE_GATE_BLOCKED: unpublished/missing fifth planner or release packet.
- PROVISIONING_BLOCKED / HARNESS_INTEGRITY_BLOCKED: distinguish unavailable
  frozen inputs from an invalid measurement/provenance harness.

Scientific outcome is separate from completion: latency-guided DSA-RP may win,
tie or regress. Do not relabel structural DSA-RP as latency-guided because it
has a post-hoc DAG score. New maps identical to old maps are legitimate nulls.

## 6. Deliverables and storage

Keep the campaign under `/opt/dsa-rp-fixed19-five-arm/`. Preserve all input
archives. Checkpoint per workload and report completed/blocked counts and live
storage regularly. Start with two host-qualified workloads to validate the
five-arm harness, then finish the fixed panel without outcome-based selection.

Keep actual problems/maps, driver/model/release sources, commands, compact
provenance and sync summaries, frozen predictions, raw device samples and all
terminal tables. Remove transient per-endpoint builds and passing tensor dumps
after evidence is verified. No bulk argument capture or simulator output.

Required tables: workload-status, model-coverage, planner-settings/costs,
map-identities, placement-validation, replay-provenance, comparability,
negative-controls, correctness, post-insertsync-summary, frozen-predictions,
timing-raw, timing-blocks, instrument-controls, pairwise-results and
predictor-evaluation. Include REPORT.md, HANDOFF.md, PINS.md, the freeze and
semantic hashes, exact reproduction commands and all adapters.

Stage file-by-file from an allowlist to
`/opt/pypto/dsa-rp-fixed19-five-arm-final.tar.gz` with a sidecar and internal
manifest. Enforce <=256 MiB ordinary staged evidence, <=25 MiB per ordinary
file, <=128 MiB separately declared failure-reproducer exceptions and <=384 MiB
total. No symlinks, builds, binaries, caches, virtual environments or passing
tensor payloads. Stop with ARCHIVE_SIZE_BLOCKED rather than compressing an
oversized root. Re-extract and verify exact manifest coverage and every hash.
Leave devices idle, no campaign processes and source worktrees tracked-clean.
