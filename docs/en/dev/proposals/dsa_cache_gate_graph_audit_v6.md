# Cache-write and Gate graph audit, 2026-09-05

Host-only follow-up to [weight validation v5](dsa_host_weight_validation_v5.md).
No new timing, input tuning, solver changes or planner integration. Existing
measurements remain development data, not prospective evidence.

## Cache-write: two missing logical dependencies

`kv_and_cache_write/tight` has an archived DSA-RP advantage of 3.05%/3.93%
over Cypress. This is not the separate approximately 7% rope/cache-write case.
The audit independently reproduces all 24 native acyclic LP scores (two arms,
twelve weights), including their ties over the entire grid.

The non-reusing base graph lacks these required RAW paths:

| Logical allocation | Producer access / node | Consumer access / node |
| --- | --- | --- |
| 5 | 24 / 4, trowmax | 27 / 6, tmax |
| 9 | 29 / 8, tdiv | 33 / 10, trowexpandmul |

The evidence is the explicit `allocation_accesses_v1` catalog: one dominating
writer covers the read range. It is **not** inferred from equal addresses.
Reshaped views become distinct `alloc_tile` SSA values in placed PTO;
`KernelScheduleGraphBuilder::getAliasRoot` follows explicit alias operations
and cannot recover this logical relationship from those values alone.

Adding required RAW edges as a labelled diagnostic changes baseline LP from
**548 to 854 model cycles**. At weights 8 and 16 both arms still have zero
penalty. They first separate at 96 in the frozen grid: Cypress 59 versus
DSA-RP 30. This does not license selecting 96 to fit this case: mixed-direction
validation must still pass with one global weight.

There are also **84 conservative control edges** connecting every one of the
14 prelude operations to every one of the six loop-body operations. These
encode completion before loop entry, not merely instruction/control order.
The diagnostic leaves them intact; replacing them requires justified
control/pipe semantics. A single-iteration DAG also does not model this
invocation's dynamic trip count and per-iteration cache-row predicate.

**Interpretation:** demonstrated graph-conformance failure, but no demonstrated
hardware explanation. The tie is not clean evidence against the proposed
latency formulation. Fixing only two RAW edges does not rescue weights 8–16.

## Gate: safer extraction, not complete mixed execution

The compatibility importer now selects a unique final trace by the requested
raw-PTO function's strict operation/provenance join, not final-phase order.
It preserves virtual-else placeholders and rejects duplicate node identities,
incomplete declared counts, ambiguous matches and detectable interleaving.

The analysis PTOAS pass accepts `function-name=...`, selecting a defined
function in the **unchanged mixed module**. Missing functions fail explicitly;
peer functions need not be replaced with stubs. This does not compose peers.

All 60 frozen child endpoints were reanalyzed. `ffn_norm`, `route_sort` and
`x_norm_quant` resolve on all twelve arm/capacity combinations: **36/60**.
The other 24 remain incomplete:

1. `gate_aic` access 75 is frontend `tpush_to_aiv`, without `OpPipeInterface`;
   the graph builder skips it. Product lowering emits `tpush` on `PIPE_FIX`,
   leaving the source/export join incomplete.
2. Product v0.57 can interleave mixed-function debug records and exposes no
   `--mlir-disable-threading`. The first trial using that flag failed before
   compilation; its output was superseded, not scored. Subsequent damaged or
   ambiguous AIV phases are rejected.
3. Orchestration contains `ffn_norm` (8 blocks), `x_norm_quant`, an empty
   pre-route task, mixed `{AIC,AIV,AIV}` gate (16 blocks), and `route_sort`
   (1 block), plus a phase-fence dummy. Tensor dependencies, pipe transport,
   block multiplicity and runtime overlap must be represented before child
   scores can be compared with parent timing. Summing child LPs or taking
   `max(AIC,AIV)` alone does not establish that model.

## Evidence and next step

- [Audit, scores, blockers and input hashes](data/dsa_cache_gate_graph_audit_v6.json).
- `tests/tools/audit_dsa_cache_write_graph.py --help`: required-RAW witnesses,
  independent native-LP checks, labelled diagnostic scores.
- `tests/tools/score_frozen_dsa_graph_manifest.py --help`: per-function scoring.
  `--product-cache` checks manifest, assembler and source hashes before reusing
  product logs, avoiding repeated compilation.
- Working evidence: `build/dsa_cache_gate_audit_v6/`; no `/tmp` artifacts.
- **241 focused Python tests passed**; **3 PTOAS FileCheck commands passed**
  for selection, missing-function rejection and the existing control graph.
- Analysis executable: one build worker, 2 GiB/no-swap cap. Analysis and tests:
  serial per job, 1.2–1.5 GiB/no-swap caps.

Next: preserve logical view identity in the base graph, model control/pipe
boundaries explicitly, and export Gate after frontend pipe lowering with a
product-equivalence check. Rerun the unchanged grid on existing timings after
conformance passes. No incremental planner is justified by these results.
