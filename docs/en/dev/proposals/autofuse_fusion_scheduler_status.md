# AutoFuse fusion-scheduler — design, cost-model journey, status & handoff

**Purpose.** A self-contained handoff for the AutoFuse solver-driven fusion + tiling work
(current development branch `fusion-scheduler`).
It records the GOAL, the **cost-model ↔ emit fidelity contract**,
**how the cost model was designed** (the reasoning, not just the code), what is implemented and
validated, and what remains. Read this to pick up the work cold.

Companion documents:

- `docs/en/dev/proposals/autofuse_cost_model_emit_contract.md` — **the fidelity contract** (A1–A7,
  the P-ladder, §6 fidelity status). This status doc summarizes it; that doc is the authority.
- `docs/en/dev/proposals/autofuse_mixed_cost_model_emit_contract.md` — the 24-group mixed
  cube/vector stage, GM-FIFO, loop-axis, and cross-core overlap contract.
- Operational device verification tasks live outside the repository in `/home/toni/work/pypto3/`.

**Vector device checkpoint (2026-07-13).** PyPTO `2e81d8ff` pins solver `4ca1026`. The
vector-only model now derives a stack-local `VectorStreamPlan` for every candidate, costs the
emitted init/rolled/tail/finalize phases, and re-derives that plan only for a winning or forced
configuration. The emitter consumes the same materialized/pointwise strip geometry and reduction
phase schedule. The 910B2 closure passed 51/51 correctness and confirmed phase traffic/overlap, but
found two decision-fidelity gaps: logical free-grid work units (G5) and P4 online-algorithm compute
(G7). Both are now host-fixed: G5 makes logical regions authoritative while keeping DMA allocation
separate; G7 puts the emitted softmax/Welford primitive work in the solver-owned plan and prices it
per phase. G5 silicon revalidation is in flight; the new G7 costs and selected grids still require a
follow-up fused-versus-cut run. The current targeted rerun is the operational task
`/home/toni/work/pypto3/autofuse_device_followup_vector_fidelity.md` (outside the repository).

**Mixed host checkpoint (2026-07-13).** The solver now builds one immutable same-engine stage DAG
and cube/vector transfer graph per mixed candidate subgraph. A stack-local `MixedSchedulePlan`
derives grid, 24-group mapping, loop trips, skew mode, and separate model-granted versus
implementable-overlap bits without entering `CostResult`; winning/forced plans are reconstructed
into solver JSON and PyPTO's `SolverTile`. Runtime mixed admission exists as a default-off `Problem`
policy. The four canonical one-way/single-round-trip scalar costs are unchanged; newly represented
non-skewable topologies receive a serial sum. Tests expose the main roofline gap: two global tiles
assigned one-per-group do not create a two-item loop on either group. Direct QK-to-softmax can
reuse the exact P4 vector-stage descriptor, but full `C→V→C→V` attention remains serial/inadmissible
in compiler mode—and its key-chunk loop remains unrepresentable—until whole-FIFO multi-round-trip
skew and the second loop axis exist. Analytic mode retains the four-stage topology at a serial cost.

---

## 1. Goal

Build a pass that turns a function's **tensor-op DAG** into **fused, tiled SPMD kernels** — vector
kernels on the Ascend 910B AIV cores and cube (matmul) kernels on the AIC cores — by:

1. running the **MLSys solver** (`3rdparty/mlsys26/`, linked as `solver_lib`) to **partition** the DAG
   into convex groups (each group → one kernel) and choose each group's **tile / grid / split /
   materialize-vs-stream**, and
2. **emitting** each group as the tiled kernel the solver priced,

such that **every cost-model term prices the algorithm the emit actually builds**. This last clause is
the whole game (see §2). The pass is `src/ir/transforms/auto_fuse_pass.cpp`; it runs behind
`PYPTO_AUTOFUSE_GENERIC_EMIT=1` (the generic emit) with an additional `PYPTO_AUTOFUSE_P4=1` for the
fused online softmax/layernorm path.

---

## 2. The cost-model ↔ emit fidelity contract (the central principle)

> **A Solver Solution is not a number — it is a specification of a kernel algorithm.** Every term in
> the cost model implicitly assumes the emitted kernel implements a particular algorithm. If the emit
> implements a *different* (usually cheaper-looking) algorithm, the cost is fictional and the solver's
> ranking is wrong.

This is §0 of `autofuse_cost_model_emit_contract.md`. The contract enumerates, per cost term, **what the
model assumes** and **what the emit must therefore build** (obligations A1–A7):

| Ref | the model assumes… | the emit must… |
| --- | ------------------ | -------------- |
| A1 | compute spreads over `U = parts_m·parts_n` wave invocations | launch one per-tile body over the grid (`SpmdScopeStmt` + `get_block_idx`) |
| A2 | ephemeral intermediates cost **0 DDR** (the fusion win) | keep intermediates on-chip (UB), never round-trip them |
| A3 | each op runs at its **back-propagated role** shape | slice `FROM_NT*` operands to the tile; read `FIXED_1`/broadcast in full |
| A4 | UB feasibility = peak live bands over the **pebbling order** | replay ops in `dfs_order_`; MemoryReuse frees bands per liveness |
| A5 | roofline overlap is phase-local | emit only eligible rolled loops **software-pipelined**; keep init/tail/finalize serial |
| A6 | reduced-axis split `S` = S cores reduce + **atomic-add** merge | build seed + atomic-add, or don't price a split the emit declines |
| A7 | streamed input multiplicity follows phase use | stream with running stats and price stats/apply traffic per input |

**The rule of thumb** (contract §6): *before trusting a cost, ask — what algorithm does this number
assume, and does the emit build exactly that, with the same operand tile shapes (A3), the same band
liveness (A4), the same DDR traffic (A2/A7), and the same overlap (A5)?* When the answer is "no", the
fix is either to make the emit implement the assumed algorithm, or to make the model price what the
emit builds. The recent 4-review cycle (§6) was exactly this audit; it found P4 divergences and we
fixed them.

---

## 3. How the cost model was designed (the reasoning)

The Ascend 910B cost model (`3rdparty/mlsys26/src/core/ascend910b_cost.cpp`, `Ascend910BCost`) was
built bottom-up from the hardware, not fitted. The design steps:

**(a) Grounded, not regressed.** Every term is derived from the **pto-isa** machine model — grounded
per-op cycles (`VecOpCompute`: `slope·repeat + head/tail`; reductions as their tree: row ≈
`45·(W/epr−1)+51`, col ≈ `16·(H−1)+30·log2(H)`), grounded per-direction bandwidths (`bw_gm_ub`,
`bw_ub_gm`, `bw_gm_l1`, `bw_l0c_gm`), and grounded cube L1↔L0 hierarchy. pto-isa is the reference GEMM
oracle (see the ecosystem note). No fitted constants except two calibrations (`kCubeComputeCost`,
`kKernelFillCost`) and the device-grounded C3 `c_task`.

**(b) The roofline is `max(compute, DDR)` — but only when the tile double-buffers.** `db_roofline`/`rfl`:
`db = tile_bytes ≥ 2·vec_reg_bytes`; overlapping load(k+1) with compute(k) gives `max`, else the tile
is too small to ping-pong and it **serializes** to `compute + DDR`. Compute spreads over the wave
makespan (`WaveComputeCycles(total, U, C) = total·ceil(U/C)/U`); DDR counts **only boundary tensors**
(ephemeral = 0, A2) over full bytes, divided across the active cores' pipes with a sub-burst
`dma_pen`. “Wave” is only the queue makespan: the emitter launches `U` ready SPMD tasks and the
runtime assigns them without affinity controls. Each task owns one logical region and performs its
UB strips/reduced-axis chunks internally on that core; no wave identity is emitted.

**(c) Feasibility is a pebbling game, not a static sum.** `vector_peak_ub` computes the **peak
simultaneously-live UB bands** over `dfs_order_` (a post-order DFS from the sinks chosen to *minimize*
that peak — finish a branch and free its bands before starting a sibling). An intermediate that is free
in DDR (fused) still costs a pebble in UB; the order is what keeps that affordable. For the cube, the
analog is `derive_exec` (the red-blue pebble peak over L1).

**(d) The sink tile back-propagates to define the per-tile algorithm.** The tile `(w,h)` is a property
of the sink output; it propagates reverse-topo to give every tensor a per-axis **role**
(`FROM_NT*` = slice to the tile; `FIXED_1` = read full / broadcast / reduced-axis). Combining the
reverse walk (roles) with the forward walk (`dfs_order_`, liveness) *is* the algorithm.

**(e) The reduction problem = the flash-attention problem.** When a reduced axis doesn't fit UB the
row/column can't be materialized; you must **stream it in blocks carrying running statistics** — the
online/flash algorithm. This yielded the **P-ladder**: P1 (bare reduction, trivial online), P2
(reduction → spanning pointwise apply, 2 passes), P3 (naive multipass, retired), **P4 (online
multi-stat: softmax `(m,l)` flash rescale, layernorm running moments)**. A7 prices the read count
(`x` is normally read twice for a spanning output; apply-only inputs once); the correctness gate declines non-foldable reductions (order
statistics can't flash).

**(f) Device grounding (the model is a validated decision oracle).** The model's *ranking* was
validated on 910B2 via a FORCE_PLAN A/B experiment: the natural argmin's wall-time tracked the model's
cheapest plan (free-tile ρ≈0.9, zero regret). Two grounding refinements came from device data:

- **C3 — per-task launch overhead.** `kernel_fill` is per-WAVE (flat for `num_tiles ≤ cores`), so the
  model *tied* plans the device ranks by task count and its argmin landed on the most-tasks / slowest
  plan. Added `+ num_tiles·split·c_task` to the vector latency; `c_task = 64` model-cycles ≈ the
  device-measured 0.2 µs/task (calibrated by the ~6.5× model↔wall factor). Device-confirmed **no
  regret**. Gated on the generic streaming emit (only it can build the fewer/larger-tile plans C3
  prefers — pricing them for the legacy tiler would pick tiles it can't realize).
- **G3 — spanning streamed reductions re-read shared input.** Device-confirmed 2.00× MTE2 for `x`;
  phase masks avoid doubling apply-only operands.
- **R0 — couple the reduced axis to its full extent in `vector_peak_ub`**, else a bare reduction sink
  (thin `[·,1]` output) looks materialized and streaming is never detected.
- **Vector closure — correctness/traffic passed, decisions not closed.** All 51 device cases passed.
  The op simulator measured exact P2 `2×x + 1×bias` and P4 `2×x` traffic plus 1.3–1.5× phase-local
  overlap. Softmax P4 was 4.6% faster than its cut. Welford was correct at mean `+2000` but 33.6%
  slower than its zero-mean cut. Forced-plan ranking was coarsely positive, but the natural P4
  candidates had about 40% regret because the checkpoint priced 12 work units while emitting 8.
  The host fix now emits all 12 logical regions and ranks the uniform `h=8` plan lower; device
  confirmation is pending.

**(g) The contract emerged from systematically checking (b)–(f) against the emit.** Each ⚠️ in contract
§6 is a place the model priced an algorithm the emit didn't build; we closed them on host (G2
pipelining, G3, G4 broadcast, G5 logical-region identity, R0, granule padding) or documented them
(G6 declined splits). G5 still needs its silicon rerun.

---

## 4. Architecture (files, emit paths, flags, commands)

- **Emit** — `src/ir/transforms/auto_fuse_pass.cpp`:
  - `AnalyzeP4Patterns` + `ProblemBuilder::Build` — DAG → solver `Problem`; one exact semantic
    analysis recognizes canonical softmax / dual-sum layernorm, records each complete op set plus
    its apply substitutions in `Problem::p4_patterns`, and retains the same `P4Match` handles for
    emission. The analytic
    multi-reduction override stays false in AutoFuse, so other candidate groups cut. Registers only
    **top-level** `var = <call>` ops → a nested-arg call drops the inner op (a known landmine).
  - `EmitFusedGroupGeneric` (~:1296) — the **vector** emit. Sub-paths: solver-planned pointwise /
    materialized row+width strips, streamed reductions P1/P2 (`stream_p1/p2`,
    `emit_strip`/`strip_at`/`slice_input`), **P4** (consumes the shared exact `P4Match`; softmax
    `p4_chunk` custom `(m,l)` body; exact layernorm → Welford), multi-sink, S2 split-K, broadcast (G4).
    Folded P1 may finish with a thin pointwise cone once, without a spanning second input pass.
    P1/P2/P4 materialize-vs-stream, chunk/tail, serial phases, and loop stages are re-derived for the
    winning config
    with the same `vector_stream_plan` helper used during pricing; an internal check verifies the
    local loop construction matches it. Shared carried-loop and spanning-apply builders serve P2,
    softmax, and Welford. Emit descriptors are not retained in the local-search cache.
  - `TileMatmul` (~:813) / `BuildTileMatmul` (k-pipeline) / `EmitLoneMatmulGeneric` — the **cube** emit.
  - Flag helpers `GenericEmitEnabled()`, `P4Enabled()` (re-read env per call).
- **Cost model** — `3rdparty/mlsys26/src/core/ascend910b_cost.cpp` (+ `types.h`, `dag.h`), branch
  `ascend-910b-vector-stream-plan`, linked as `solver_lib`. `VectorStreamPlan` is stack-local while
  pricing candidates and re-derived only for final/forced configs consumed by AutoFuse.
- **Flags:** `PYPTO_AUTOFUSE_GENERIC_EMIT`, `PYPTO_AUTOFUSE_P4`, `PYPTO_AUTOFUSE_FORCE_PLAN`
  (`"[g<N>:]w,h,split[,pm,pn]"`, **static-cached per process** → one force per fresh subprocess),
  `PYPTO_AUTOFUSE_FORCE_MERGE=none|all`, `PYPTO_AUTOFUSE_DUMP_PLANS`, `PYPTO_AUTOFUSE_STRICT`.
- **Build (MAX 2 cores):** `cmake --build build --parallel 2`;
  `cmake --build 3rdparty/mlsys26/build --target solver_lib -j2`.
- **Test:** `PYTHONPATH=$(pwd)/python python -m pytest tests/ut/ir/transforms/test_auto_fuse.py -q -n 4`
  (27 passed / 1 xfail — the xfail is #1908 chained-matmul lowering). Solver suite
  `./3rdparty/mlsys26/build/tests/ascend_910b_test` (338 pass / 7 documented baseline failures). Numeric:
  `pypto.debug.torch_codegen(passes.auto_fuse()(Prog), run_all_spmd_blocks=True)` — write P4 DSL FULLY
  NAMED (nested args drop ops from the solver graph → miss P4).

---

## 5. What is implemented and validated

**Vector emit** (behind `GENERIC_EMIT`):

- Pointwise: fused chains, UB-streamed row+width strips sized by **real peak-liveness** (+1 prefetch
  band) in `VectorStreamPlan`; the emitter consumes that geometry. Tall / wide / reused-input all
  handled within UB.
- Reductions: P1 (bare) + P2 (spanning apply), streamed over the reduced axis, **both passes pipelined
  (A5/G2)** — accumulator persists, loads double-buffer; numerically exact.
- Broadcast operands (G4), multi-sink, S2 cross-core split-K.
- **P4 softmax** — fused online flash `(m,l)` with `exp(m_old−m_new)` rescale; the cone is verified
  EXACT by the shared descriptor (only `row_max→sub(x,m)→exp→row_sum→div`); stats and apply passes
  stage-2 pipelined when their rolled trip count is at least 2. It was numerically correct and 4.6%
  faster than its two-kernel cut on 910B2, but remains flagged until G5 and the new G7 ranking are
  revalidated.
- **P4 layernorm** — stable streaming **Welford** (running count/mean/M2, Chan's parallel merge),
  reached only after proving the exact `sum(x)` / `sum(x*x)` / mean / variance / rsqrt / centered-apply
  algebra, then substituting stable `mean`/`var` into that cone. Its three-carry stats and apply passes
  are stage-2 pipelined under the same trip-count guard. ≤ the cut's accuracy at input mean
  0/100/1000/2000, **no NaN at +2000** (the dual-sum form NaN'd there), but its device wall was
  33.6% slower than the zero-mean cut. Temperature softmax,
  weighted-second-moment graphs, and patterns with an escaping internal stat cut and preserve semantics.

**Cube emit:** G-A ceil+clamp grid (non-uniform lone matmul tiles across cores — **device-validated**:
`[272,272]`→spmd(8), torch-exact; forcing an untiled 1×1 crashes, so G-A is load-bearing), ragged-K
peel, deep-T chained (tensor-level; #1908-xfail at lowering). k-loop is `ForKind::Pipeline`.

**Cost model:** C3 per-task overhead (device-validated no-regret), per-input G3 phase traffic, R0
reduced-axis coupling, granule-padded feasibility, the reduction source/work two-band floor, and
candidate-local P4 feasibility. The stack-local
`VectorStreamPlan` records pebble/scratch peaks; owns materialized/pointwise strips and streamed-reduction
init/rolled/tail/finalize phases; and gates each phase's A5 overlap on its actual loop stage. Cost is
the sum of phase rooflines, so barriers never hide work across phases. AutoFuse re-derives the plan for
the winning config. It now also owns element-balanced M/N logical partitions and their exact
`work_units`; `free_tile_alloc` carries DMA padding separately, so alignment cannot change the SPMD
count. `CostResult`
stays at its pre-refactor 112-byte footprint (guarded at ≤128 bytes) for the local-search cache.

**Vector refactor preservation audit.** The plan extraction was separated from the subsequent
model corrections so structural movement could be checked independently. Solver `10f8f8b`
(pre-plan) and `1fc542d` (plan authoritative and absent from `CostResult`, before phase pricing)
produce the same fixed-cost anchors: fused/separate pointwise `11003.7/22007.4`, fused/cut softmax
`30073.8/88186.9`, few-row reduction `10563`, and long streamed reduction `101566`. Current
`4ca1026` deliberately changed the phase-sensitive anchors to softmax `22153.9/88309.8`, few-row
reduction `10093.3`, and long streamed reduction `59469.3`. The logical-region identity correction
then moves only the affected vector anchors to softmax `22208.7/88371.3`, few-row reduction
`10097.6`, and long streamed reduction `60614.3`; pointwise remains
`11003.7/22007.4`. These are intended consequences of phase-local compute/traffic, serial edge
phases, the reduction source/work floor, and making the cost replay the exact emitted task grid—not
drift from moving schedule derivation into a plan.

**Current host gates.** After the logical-region regression, AutoFuse UT is 33 passed / 1 expected
xfail; the solver suite retains its 7 documented baseline failures. The vector checkpoint's device file
collects 51 A2/A3 cases. Four persistent cases
cover exact P4 softmax `[128,8192]`, Welford layernorm at input mean `+2000`, a scaled-softmax near
miss that must cut, and a P2 apply-only bias input whose expected MTE2 payload is `2×x + 1×bias`.

**Device-validated on 910B2:** vector correctness 51/51; exact P2/P4 phase traffic; phase-local
overlap; softmax P4 correctness and a 4.6% win; high-mean Welford correctness but a 33.6% wall
regression; cube G-A + ragged-K correctness. Two earlier device-found crashes (reused-input
pointwise, split-K seed) are fixed. The latest completed ranking run does not validate zero regret
because it predates the G5 host fix and counted 12 cores for an emitted 8-core kernel.

**Current commit arc** (newest first): PyPTO `2e81d8ff` phase-traffic device case · `8e3d8731` wide
P4 device cases · `11ff2a36` consume phase-priced schedules · `e307e15c` rebuild plans only for
winners · `8922552e` consume solver-owned vector plans · `727c6310` shared P4 detection + pipelined
stats. Solver `4ca1026` phase rooflines · `1fc542d` compact search cache · `0e34918` authoritative
plans · `975570a` expose vector plans. Earlier P4/cost arc: `d8f650e4` Welford · `e566ecbe` P4
hardening · `807fb391` softmax · `10aa6fd0` G2 · `bbd46b4b` C3.

---

## 6. The 4-review cycle (what it found and how we responded)

Four independent adversarial reviews (contract / correctness / architecture / performance) audited the
P4 batch against the contract. The batch's *fixes* were sound (G2 is a real ping-pong, the band-count
and split-K-seed fixes hold); the *bugs* were all in P4 and masked by P4-off default. Confirmed + fixed:

- **C1 (silent-wrong):** `classify_p4` accepted any coupled max+sum, but the emit **hardcodes**
  `exp(sub(x,m))` and the flash rescale is math-specific to `exp(x−m)`. A temperature/scaled softmax
  streamed the wrong stat (~12× off, no error). → verify the EXACT cone; else cut. (`e566ecbe`)
- **A1 (crash):** the adapter cost gate (op counts) was looser than `classify_p4` (structural) → the
  solver fused a shape the emit declined → over-UB crash. → adapter mirrors `classify_p4`. (`e566ecbe`)
- **C2 (crash):** a wide high-reuse pointwise declined to the legacy tiler (no UB guard) → overflow. →
  width-chunk instead of declining. (`e566ecbe`)
- **A5 (perf):** P4 loops were serial while priced pipelined and the cut is pipelined → fused could be
  device-slower. → apply pass pipelined in `e566ecbe`; softmax `(m,l)` and Welford
  `(mean,M2,count)` stats loops now pipeline too when they have at least two rolled iterations.
- **C3 (accuracy):** dual-sum layernorm less accurate than the cut, NaN at high mean → **Welford**
  (`d8f650e4`).
- **C4 (silent-wrong):** any independent pair of row sums could enter the Welford path; a graph using
  `sum(x)` and `sum(2*x*x)` was silently reinterpreted as layernorm. → one exact `P4Match` analysis is
  shared by model and emit; only the canonical layernorm algebra reaches Welford, every near miss cuts.

**Architectural debt paid down:** the duplicated cost-gate ↔ emit-classifier predicate is removed;
P2/softmax/Welford share one planned carried-loop constructor and one spanning-apply builder. Their
statistics update math remains deliberately algorithm-specific.

---

## 7. What remains (ordered)

1. **Device-close G5 logical-region identity.** Re-run the forced `h=11` and `h=8` plans. Confirm
   `pl.spmd(12)` versus `pl.spmd(16)`, numeric correctness, per-plan traffic, and that the repaired
   natural argmin is the device-best `h=8` plan rather than refitting pto-isa constants.
2. **Device-close G7 algorithm work.** The host now represents softmax `(m,l)` correction and
   Welford chunk/Chan operations in exact init/rolled/tail/finalize work tallies, using grounded
   per-op costs. Re-run natural and forced fused/cut plans because the selected grids changed; do not
   add a fitted surcharge if the remaining wall ranking disagrees—first identify the missing phase or
   hardware serialization.
3. **Finish the current targeted vector follow-up** from the operational task outside the repository:
   `/home/toni/work/pypto3/autofuse_device_followup_vector_fidelity.md`. Repeat work-unit identity, forced-plan ranking,
   softmax fusion, and Welford fusion-versus-cut decisions. Keep P4 flagged until this closes.
4. **Resolve the Release-only cube assertions.** The vector checkpoint's clean Release build reported
   five exact-geometry cube test failures. Capture the actual winner/cost deltas and determine code
   versus test error; do not infer harmlessness from unrelated device cases or edit expectations
   without approval.
5. **Remaining vector fidelity:** gate or emit the G6 materialized max/row-reduction split families.
   Only after the G5 rerun and G7 may exact softmax be considered separately for default-on;
   layernorm must remain off while Welford is materially slower than the cut.
6. **Complete cube-only fidelity:** the role-aware `CubeSchedulePlan` and recursive uniform-grid
   emitter are implemented (§8). Next reconcile GM reload with the emitted L0-subtile loop, then
   introduce per-node cube phase rooflines, price split seed/tasks, and close non-uniform grids.
7. **Complete mixed fidelity:** make the plan choose a real pipeline-item axis and active-group
   count, replace global-tile overlap with serial-versus-realizable phase costs, then implement the
   one-way and single-round-trip emit through `ExpandMixedKernel` → `InjectGMPipeBuffer` →
   `SkewCrossCorePipeline`. Full flash attention follows only after whole-FIFO multi-round-trip skew.

**Deferred (all decline *gracefully* today — correctness intact, not fused):** the ProblemBuilder
nested-arg gap (hoist nested compute-call args to SSA temps); P4 col-softmax / scale-then-softmax /
chained layernorm; the cube
ragged-K peel lowering test + deep-T decline logging; mixed cube+vector (a separate charter).

---

## 8. Cube-only plan and current fidelity boundary

The cube path now follows the same solver-owned-plan architecture as the vector path, with a
separate `CubeSchedulePlan` for cube-specific decisions: requested tensor regions, per-instance K
streams, L1 residency, L0 subdivision, spatial ownership, and split-K. The candidate-invariant
request topology is built once in `Ascend910BCost::create()`; candidate evaluation performs only an
O(nodes) stack-local derivation. The complete descriptor is reconstructed for the winning/forced
configuration and is not stored in `CostResult` or the local-search cache.

**General request model.** For `O=A@B`, a consumer request `O[rows,cols]` recursively requests
`A[rows,K]` and `B[K,cols]`. Requests are memoized by tensor plus symbolic height/width bindings.
Identical requests share one producer; different fan-out roles become different instances and pay
recomputation. The resulting postorder supports produced RHS, both inputs produced, non-square
trees, deep chains, fan-out, and multiple roots. One root can bind its contraction to `ParallelK`;
multiple roots force `S=1` because one split coordinate cannot identify several atomic targets.

**Plan contents.** Every `CubeMatmulSchedule` records its request-instance ID, producer-instance
dependencies, exact output/LHS/RHS bindings and extents, contraction/share, L1 window, actual load
chunk, rolled trip count, serial tail, and loop stage. The group records the exact partitions,
split/work units, peak L1, L0 M/N tile dimensions, roots, seed requirement, and overlap bits.
Feasibility, recursive cube MAC/extract work, boundary traffic, and plan reconstruction consume this
same request DAG.

**Generic emitter.** For uniform multi-matmul grids, AutoFuse replays the plan
producer-before-consumer. Produced operands stay local; an intermediate spanning several L0 tiles is
assembled in an explicit L1 tensor. Each node uses its own planned K loop. Distinct fan-out roles
are recomputed as priced. Sink split-K emits a tiled zero seed and atomic stores; `S=1` stores
directly. Multiple roots at `S=1` are supported. Strict mode reports a contract rejection; normal
mode falls back to dependency-ordered standalone matmuls rather than silently emitting another
algorithm. FP16/BF16 inputs with FP32 accumulation are priced from operand precision. A
low-precision final output that would need a K carry declines until FIXPIPE narrowing is explicit.

**Serial-overlap fix.** The old scalar `K/S ≥ 32` gate is removed for pure cube. The cost now grants
its global `max(compute,DDR)` only if every request that loads a boundary operand reconstructs a
real stage-2 rolled loop. A one-trip loop therefore serializes and can change the natural argmin;
this is a deliberate fidelity correction, not a descriptor-only refactor.

**Host validation.** Solver tests cover exact role regions/peaks, non-square both-produced trees,
fan-out role switches, multi-sink, plan L0 dimensions, compact cache behavior, and the one-trip
no-overlap case. AutoFuse tests cover forced recursive plans, split/no-split, ragged K, fan-out,
multiple outputs, Torch numerics, strict fallback, and default lowering of a chain whose
intermediate spans multiple L0 tiles.

**Remaining gaps, in implementation order:**

1. Reconcile GM→L1 reload with L0 subdivision. The model charges each logical boundary request once
   per work unit, but the current emitter nests a full K stream inside each L0 output subtile. Either
   multiply traffic by that loop or retain input panels with explicitly priced L1/accumulator state.
2. Replace the remaining subgraph-wide cube roofline with per-node init/rolled/tail/store phases.
   Only each concrete rolled phase may receive overlap; serial tails cannot hide behind another
   node's work.
3. Price the split zero-seed/barrier and per-task launch overhead.
4. Close spatial fidelity. AutoFuse currently filters unequal multi-op grids. The lone-matmul
   ceil+clamp path is numerically idempotent but can execute more max-size work than balanced LPT
   prices; implement exact non-uniform shapes or price the actual overlap.
5. Represent a shared boundary panel and its lifetime before emitting a request that the model
   deduplicates. The current plan emitter declines that family.
6. Add a planned final FIXPIPE phase for streamed/split FP16/BF16 outputs; today they fall back.
7. Run forced-plan correctness, trace traffic/overlap, wall-versus-model, and regret validation on
   910B2. Keep mixed cube+vector outside this charter.

The authoritative obligation table and validation ladder are in
`docs/en/dev/proposals/autofuse_cube_cost_model_emit_contract.md`.

## 9. Operational gotchas (don't relearn these)

- **Build MAX 2 cores.** Use absolute build paths (a `cd 3rdparty/mlsys26` persists across shell calls
  and mis-targets the build).
- **FORCE_PLAN is static-cached per process** → one force per fresh subprocess; the `group[0]` solver
  log is misleading under force (prints the argmin, not the forced tile) — trust the emitted
  `pl.spmd(N)` count.
- **Nested-arg DSL** (`exp(sub(x,m))`) silently drops the inner op from the solver graph → misses P4 +
  mis-costs. Write NAMED temps.
- **Welford count column** must be derived from a reduction output (col-major), NOT `tensor.full`
  (row-major → trips `ResolveBackendOpLayouts`' col-vector reshape → lowering crash).
- **Remote access:** SSH is currently blocked; use HTTPS, submodule first when publishing code.
  `.gitmodules` still contains an SSH URL for `mlsys26`, so device checkouts must locally override
  `submodule.3rdparty/mlsys26.url` with `https://github.com/tonibohnlein/mlsys26.git`. NEVER push /
  open PRs without explicit order; NEVER add AI co-author lines; never hack test expectations.
- **Device runs:** a separate agent checks out `fusion-scheduler-vector-stream-plan`, builds, and runs
  on 910B2 with a working
  `device_wall effective_us` STRACE path; `benchmark()` hits error 507018 (avoid). Fingerprint-gate on
  HEAD + mlsys26 hash first.
