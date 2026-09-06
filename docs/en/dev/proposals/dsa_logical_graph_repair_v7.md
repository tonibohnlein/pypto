# Logical allocation and structured-boundary repair, v7

Host-only follow-up to [the v6 graph audit](dsa_cache_gate_graph_audit_v6.md).
No new device timing and no new evaluated placement. Existing observations
are retrospective development evidence, not prospective validation.

## Implementation and limits

1. `emit_ptoas_logical_memory_topology` exports zero-delay logical RAW/WAR/WAW
   dependencies from `allocation_accesses_v1`. Reshaped views retain the source
   allocation identity even when PTO contains distinct `alloc_tile` values.
   Neither physical address equality nor the penalty-candidate catalog creates
   these base dependencies. Function, access joins and graph hashes are checked.
   PTOAS loads them through `logical-memory-edges` before validating the duration
   and placement-edge sidecars against the resulting base graph.
2. Opt-in `semantic-boundaries` replaces the all-to-all completion edges around
   `scf.for`/`scf.if` with actual predicate/bound readiness, per-pipe order,
   cross-region memory dependencies and yielded-value SSA dependencies.
   Opposing branches do not acquire same-iteration order edges. Calls and
   unsupported boundaries retain the conservative behavior. Positive-distance
   memory recurrences remain in the graph.
3. The research recognizer now records an L0-to-UB source read for
   `tile.tpush_to_aiv`, despite its lack of a local result. At Gate's frozen
   compiler snapshot, access 75 now reads the 1,024-byte accumulator. The fresh
   metadata export preserves all solver-visible fields of all five child
   problems. It is not a replacement executable for the archived measurements.
4. The small analysis tool exposes the existing frontend pipe-lowering pass.
   This recovers the push in an analysis copy of the complete mixed module;
   original product inputs and product-codegen evidence are retained separately.

Unresolved subranges fail closed by default. An explicit
`conservative-logical-ranges` diagnostic treats an unresolved subrange as the
whole known logical allocation and records every affected access. This is safe
ordering overapproximation, **not exact range recovery**. Its scores must not be
described as exact model coverage. Duration evidence likewise distinguishes
calibrated signatures, analytical formulas and pinned Perf-Sim approximations.

The semantic-boundary mode and logical catalog pair enumeration are research
analysis, not enabled product optimizations. Dense pair construction can take
quadratic work/output within the analyzed graph/allocation; no production-pass
scaling claim is made. A sparse frontier implementation and review are required
before considering production use.

## Coverage and source invariance

| Analysis | Numeric child-graph scores | Source-range precision | Conservative envelopes |
| --- | ---: | ---: | ---: |
| Historical manifest | 51 endpoints / 14 measured cells | 43 endpoints | 8 endpoints |
| Gate, four capacities / three physical arms / five children | 60 endpoints | 48 endpoints | 12 endpoints |

All 14 historical bases and all 20 Gate child/capacity bases are identical
across arms when comparing operation/access/pipe/duration records and semantic
dependency edges, including recurrence depth. This is stronger than equality
of only their longest-path values. Frozen placed-PTO and solution hashes are
checked before historical rescoring.

These are acyclic longest-path and recurrence **bounds**, not completed
invocation-latency predictions. Finite loop counts, branch feasibility across
iterations and runtime lane/core assignment are not fully composed here.
`invocation_model_complete` therefore remains false, including for endpoints
whose individual node durations are all resolved.

## Gate mixed execution

The named reserve/import channel `gate_c2v_slot_buffer` has two 1,024-byte slots.
Its peer binding, configuration and all transfer sites are checked in each of
the twelve arm/capacity cells:

| Source | Target | Meaning |
| --- | --- | --- |
| AIC access 75 / node 17 | AIV access 57 / node 2 | C2V availability, then branch |
| AIC access 75 / node 17 | AIV access 92 / node 27 | C2V availability, else branch |
| AIV access 57 / node 2 | AIV access 59 / node 4 | Slot release after consumption |
| AIV access 92 / node 27 | AIV access 94 / node 29 | Alternative slot release |

The original orchestration has the mixed group `{gate_aic, gate_aiv, gate_aiv}`,
not a single interchangeable child. Source tensor dependencies recover
`ffn_norm → x_norm_quant`, `ffn_norm → mixed gate`, and
`mixed gate → route_sort`, with two tensor witnesses for each. A phase-fence
dummy and source-loop submit multiplicity are recorded separately, not erased.

The extractor retains branch contexts. It does **not** sum child bounds, pretend
that both branches execute on one AIV lane, or infer a whole-parent critical
path from this inventory. Runtime lane expansion, explicit fence semantics and
finite-loop/task execution still block the complete Gate timing prediction.

## Comparison with existing timings

The unchanged grid is 0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256 cycles.
The table reports DSA-RP/Cypress device deltas; negative is faster. The model
column is DSA-RP minus Cypress reuse-induced acyclic LP extension at weight 16.

| Fixed measured cell | Device deltas | Model delta, cycles | Reading |
| --- | --- | ---: | --- |
| `kv_score_proj_c128` | −2.62% / −2.47% | −86 | Correct direction |
| `rmsnorm_rope/tight` | −5.77% / −5.80% | −152 | Correct direction; conservative range |
| `rmsnorm_rope_cache_write/q1` | −4.05% / −4.12% | −186 | Correct direction |
| `kv_and_cache_write/tight` | −3.05% / −3.93% | 0 | Still misses the win |
| `hc_post_prefill/native` | −4.65% / −4.76% | +244 | Wrong direction |
| `gumbel_argmax` | +0.20% / +0.17% | −57 | Nonzero prediction on a measured near-null |

Do not conflate the cache-write tight cell or rope/cache-write q1 cell with the
separate approximately 7% half-capacity result. These numbers come from the
specific fixed cells named in the linked table.

Among the five cells exceeding 2% on both archived devices, weights 8–16 give
three correct directions, one tie and one wrong direction. At 64–256 the
cache-write tie separates correctly, but hc_post remains wrong: four of five.
This is not permission to choose a larger weight from these outcomes. All five
decided cases favor DSA-RP; Gate is still not a composed Cypress-winning
counterexample. The mixed-direction scientific gate **does not pass**. No
incremental planner or new placement experiment is warranted by this result.

## Artifacts and reproduction

- [Historical comparison, all weights](data/dsa_graph_repair_historical_v7.tsv)
- [Gate child bounds, all weights](data/dsa_graph_repair_gate_v7.tsv)
- [Coverage, source hashes and weight-grid accounting](data/dsa_graph_repair_v7.json)
- [Gate mixed transfer and orchestration evidence](data/dsa_gate_mixed_transfers_v7.json)

Detailed graph/sidecar/log artifacts are under `build/dsa_graph_repair_v7/`.
Checked-in entry points are `tests/tools/rescore_dsa_repaired_corpus.py`,
`score_frozen_dsa_graph_manifest.py`, `audit_dsa_mixed_transfers.py` and
`summarize_dsa_graph_repair.py`; each provides `--help`. The two score runners
run serial graph processes. Gate uses the frozen manifest, cached official
product evidence, rejoined source records, repaired metadata and analysis-only
frontend lowering. No archived executable is rebuilt for measurement.

Local validation: rebuilt current PyPTO and the small PTOAS analysis target;
280 focused Python tests and six PTOAS FileCheck runs passed. Builds used one
worker and a 2 GiB/no-swap cgroup cap. At most two serial scoring jobs ran
concurrently, each capped at 1.2 GiB with swap disabled. No large `/tmp` files
were created.
