# Retrospective reuse-weight validation, 2026-09-05

Follow-up: [v6 graph audit](dsa_cache_gate_graph_audit_v6.md) finds missing RAW
paths in cache-write. "Resolved graph inputs" below does not certify a correct,
complete latency graph. These numbers remain historical model output.

This is research-paper development evidence, not a new device campaign or a
PyPTO planner change. No latency was collected. Predictions were generated
twice identically over the unchanged global grid
`0,8,16,24,32,48,64,96,128,160,192,256` before this run joined the existing
timing tables. Those tables were already known development data; this is not
a prospective or blinded validation.

## Results

All 51 historical endpoints, covering 14 problem-capacity cells, now have
resolved graph inputs. The additions cover pinned fp32 `tmax`, scalar `tcmps`
(including NE's internal inversion), `tdiv`, and `tsel` lowering. They are
labelled `pinned_analytical_model`, not measured exact-signature durations.
They assume full valid tiles and isolated queues; select uses an isolated-row
upper-envelope approximation. The loop/DAG scores are bounds within this
assumed model, not hardware latency bounds or complete invocation predictions.

At weights 8 and 16, four of five same-direction, at-least-2%-on-both-devices
DSA-RP wins receive the correct strict ordering. The fifth,
`kv_and_cache_write/tight`, remains a tie at **every** grid weight despite
DSA-RP being 3.05%/3.93% faster. Including the recurrence bound does not resolve
that tie. Unit cost predicts its direction correctly. Both sub-1% null cells
remain ties at 8 and 16; weights of 24 or more introduce a false ordering on
Gumbel. The magnitude gate here is not an independent confidence-interval test.

Gate was rebuilt from `f9a7e00f` with diagnostic-only access/loop provenance.
All five solver-visible DSA problems match the frozen problems. Metadata changes
their fingerprints, so archived placements were left unchanged and used with
the original placed PTO. Of 60 child endpoints (five functions, four
capacities, three physical policies), 36 resolve. `gate_aic`/`gate_aiv` remain
blocked by mixed-function graph composition and final-trace joins. At native,
the resolved `ffn_norm` graph favours Cypress by 8/17 cycles at weights 8/16;
this is **not** a prediction of the measured whole-parent Gate latency. The
missing children cannot be discarded or their timings inferred from the parent.

**Decision:** the 8–16-cycle interval has not passed mixed-direction validation.
Do not integrate the incremental planner based on these results. The next
research checks are the cache-write tie and Gate's complete composition.

## Tables and reproduction

- [Per-cell scores and observed device ratios](data/dsa_host_cell_scores_v5.tsv)
- [Global weight sensitivity](data/dsa_host_weight_grid_v5.tsv)
- [Gate child scores, including recurrence deltas](data/dsa_gate_partial_scores_v5.tsv)
- [Provenance, evidence classes, blockers and source-table hashes](data/dsa_host_weight_validation_v5.json)

`tests/tools/score_frozen_dsa_graph_manifest.py --help` describes the serial,
host-only endpoint scorer. It consumes exact placed PTO, replay solutions,
original problems and metadata-only exports; mismatched solver-visible fields
or incomplete joins are rejected. Its output explicitly does not certify an
invocation model. Focused duration/graph tests: 234 passed. The frozen compiler
build used one worker with a 2500 MiB/no-swap cgroup cap; no build used `/tmp`.
