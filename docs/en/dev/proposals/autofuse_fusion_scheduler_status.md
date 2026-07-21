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

**Vector device checkpoint (2026-07-13).** The completed 910B2 run at PyPTO `d8ca8a8f` / solver
`e566674` passed 51/51 vector cases and silicon-closed G5/G7/G8/G9. Exact phase structure and traffic
remained intact (`2×x` for P4, `2×x+1×bias` for P2); the PTO row-reduction anchors reproduced
byte-for-byte; and `[768,8192]` naturally selected 48 tasks in the device-best cluster (4.2% regret,
versus the pre-G9 4.2× miss). Exact softmax was 25.5% faster than its cut and Welford was 24.7%
faster at zero mean while remaining correct at `+2000`. Exact softmax is therefore default-on;
Welford remains explicitly gated pending a clean extreme-shift sweep. The subsequent host closure
adds the separately ordered G6 zero seed, PTO A2/A3 column-reduction fits (G10), and exact source
descriptors for the remaining one-instruction vector operations (G11); none changes the validated
P4 phase algorithm.

The follow-up at PyPTO `95e24c32` / solver `f7bea24b` passed the same 51/51 device surface and
silicon-closed G10/G11 plus the aligned G6 seed protocol. It also established Welford's FP32
accuracy envelope (roughly `mean/std <= 5–6e4`) and found one real buildability hole: a row-major
FP32 split seed narrower than 32 bytes (`N<8`) cannot lower. The current host batch gates that case
to `S=1`, keeps aligned seeds split-capable, and adds exact returned-live-out, capability, UB,
ragged-traffic, fill-wave, and cache-publication contracts. These host refinements await their own
fingerprinted device follow-up.

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

**Cube device checkpoint (2026-07-16).** Pure-cube schedules that reached silicon passed 75/75
runs across clamped overlap, multi-window/ragged K, split-K, and a low-precision recursive chain.
The former oversized sub-fractal `1x1` candidate is now absent from both cost modes. On
`[272,272]@[272,272]`, the default analytic winner A8 (`144x80`, 8 tasks) was 7.3% faster than the
exact winner E12 (`80x96`, 12 tasks) under the scheduler/orchestration execution metric; forcing a
fixed grid produced a byte-identical executable in both modes. Analytic therefore remains the
default and exact remains opt-in. Pipe tracing closed the apparent inversion: E12 is cheaper per
task in the op simulator, the generated kernel already executes `PIPE_ALL` after its final TSTORE,
and a redundant second barrier costs zero. The entire 6.6–6.7 us device gap is the AICPU scheduler,
but the follow-up multi-count sweep falsified a scalar per-work-unit correction: scheduler time was
U-shaped for the 272 family and fell as the 512 family was divided over more cores. The prior
`1.6 us/task` slope was local to A8/E12, where task count and per-core work changed together. Add no
dispatch or pipe constant. Analytic remains default and exact remains opt-in; next hold tile/K work
constant while varying only the number of output regions.

---

## 1. Goal

Build a pass that turns a function's **tensor-op DAG** into **fused, tiled SPMD kernels** — vector
kernels on the Ascend 910B AIV cores and cube (matmul) kernels on the AIC cores — by:

1. running **PTO Fusebox** (`3rdparty/pto-fusebox/`, linked as `solver_lib`) to **partition** the DAG
   into convex groups (each group → one kernel) and choose each group's **tile / grid / split /
   materialize-vs-stream**, and
2. **emitting** each group as the tiled kernel the solver priced,

such that **every cost-model term prices the algorithm the emit actually builds**. This last clause is
the whole game (see §2). The pass is `src/ir/transforms/auto_fuse_pass.cpp`; it runs behind
`PYPTO_AUTOFUSE_GENERIC_EMIT=1` (the generic emit). Exact canonical softmax P4 is default-on;
`PYPTO_AUTOFUSE_P4=0` selects the cut fallback and `PYPTO_AUTOFUSE_P4=1` additionally enables the
still-gated Welford layernorm path.

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

The Ascend 910B cost model (`3rdparty/pto-fusebox/src/core/ascend910b_cost.cpp`, `Ascend910BCost`) was
built bottom-up from the hardware rather than fitted to AutoFuse wall times. The design steps:

**(a) Grounded, not AutoFuse-wall regressed.** Every term is derived from the **pto-isa** machine
model — grounded per-op cycles (`VecOpCompute`: `slope·repeat + head/tail`; FP32/FP16 row and column
reductions use PTO's profiled fit formulas, while unsupported dtypes retain the structural tree),
grounded per-direction bandwidths (`bw_gm_ub`,
`bw_ub_gm`, `bw_gm_l1`, `bw_l0c_gm`), and grounded cube L1↔L0 hierarchy. pto-isa is the reference GEMM
oracle (see the ecosystem note). No coefficient is fitted to AutoFuse wall measurements: PTO's own
instruction fit table is imported as grounding; the remaining calibrations are `kCubeComputeCost`,
`kKernelFillCost`, and the device-grounded C3 `c_task`.

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
- **Vector silicon closure — G5/G7/G8/G9 passed.** All 51 device cases passed. The op simulator
  measured exact P2 `2×x + 1×bias` and P4 `2×x` traffic; phase-local overlap and serial edge phases
  matched the plan. Grounded row work moved the tall natural grid to 48 tasks and cut argmin regret
  to 4.2%. Exact softmax beat its cut by 25.5%; Welford beat the zero-mean cut by 24.7% and was the
  only numerically sound path at mean `+2000`.

**(g) The contract emerged from systematically checking (b)–(f) against the emit.** Each ⚠️ in contract
§6 is a place the model priced an algorithm the emit didn't build; we closed them on host (G2
pipelining, G3, G4 broadcast, G5 logical-region identity, G6 split admission, R0, and granule
padding). G5/G7/G8/G9 are silicon-closed. G6 seed work, G10 column fits, and G11 exact primitive
coverage close the remaining representational gaps on host without fitting AutoFuse wall time.

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
    winning config with the same `vector_stream_plan` helper used during pricing; an internal check
    verifies the local loop construction matches it. Shared carried-loop and spanning-apply builders serve P2,
    softmax, and Welford. Emit descriptors are not retained in the local-search cache.
  - `TileMatmul` (~:813) / `BuildTileMatmul` (k-pipeline) / `EmitLoneMatmulGeneric` — the **cube** emit.
  - Flag helpers `GenericEmitEnabled()`, `P4Enabled(P4PatternKind)` (re-read env per call). With the
    P4 variable unset, exact softmax is enabled and Welford is not; `0` disables both and a nonzero
    value enables both.
- **Cost model** — `3rdparty/pto-fusebox/src/core/ascend910b_cost.cpp` (+ `types.h`, `dag.h`),
  published from `pto-fusebox/main` and linked as `solver_lib`. `VectorStreamPlan` is stack-local while
  pricing candidates and re-derived only for final/forced configs consumed by AutoFuse.
- **Flags:** `PYPTO_AUTOFUSE_GENERIC_EMIT`, `PYPTO_AUTOFUSE_P4` (unset = exact softmax only,
  `0` = neither, nonzero = exact softmax + Welford), `PYPTO_AUTOFUSE_FORCE_PLAN`
  (`"[g<N>:]w,h,split[,pm,pn]"`, **static-cached per process** → one force per fresh subprocess),
  `PYPTO_AUTOFUSE_FORCE_MERGE=none|all`, `PYPTO_AUTOFUSE_DUMP_PLANS`, `PYPTO_AUTOFUSE_STRICT`.
- **Visualization:** `PYPTO_AUTOFUSE_DUMP=<dir>` plus the partition and per-kernel algorithm views
  described in [autofuse_schedule_visualization.md](autofuse_schedule_visualization.md).
- **Build (MAX 2 cores):** `cmake --build build --parallel 2`;
  `cmake --build 3rdparty/pto-fusebox/build --target solver_lib -j2`.
- **Test:** `PYTHONPATH=$(pwd)/python python -m pytest tests/ut/ir/transforms/test_auto_fuse.py -q -n 4`
  (42 passed / 1 xfail — the xfail is #1908 chained-matmul lowering). Solver suite
  `./3rdparty/pto-fusebox/build/tests/ascend_910b_test` (450 pass / 7 documented baseline failures). Numeric:
  `pypto.debug.torch_codegen(passes.auto_fuse()(Prog), run_all_spmd_blocks=True)` — write P4 DSL FULLY
  NAMED (nested args drop ops from the solver graph → miss P4).

---

## 5. What is implemented and validated

**Vector emit** (behind `GENERIC_EMIT`):

- Pointwise: fused chains, UB-streamed row+width strips sized by **real peak-liveness** plus an
  explicit prefetch copy, with every tensor's actual dtype in `VectorStreamPlan`; the emitter
  consumes that geometry. Tall / wide / reused-input and mixed-width
  FP32-intermediate→INT8-output chains all fit UB.
- Reductions: P1 (bare) + P2 (spanning apply), streamed over the reduced axis, **both passes pipelined
  (A5/G2)** — accumulator persists, loads double-buffer; numerically exact.
- Broadcast operands (G4), multi-sink, and S2 terminal-`col_sum` cross-core split with tiled zero
  seed plus atomic-add merge. The seed is an explicit serial `TEXPANDS`-grounded fill/store phase
  with its own task and kernel-fill terms. A seed row must span one DMA block, so thin FP32 outputs
  (`N<8`) stay at `S=1`. Unsupported row/max/min/internal/ragged split families also stay at `S=1`.
- Every adapter op carries an explicit vector buildability capability. Elementwise replay and
  row/column sum/max reductions enter the generic scheduler; `prod`, arg reductions, min reductions,
  shape-generating `full`, and other unsupported transforms are partition barriers until their
  distinct algorithms exist.
- Function returns are explicit solver live-outs. A returned-and-consumed SSA value is both a UB
  lifetime and a DDR boundary output; P4 rejects an escaping statistic and multi-sink emission
  assembles every returned value. Partition/group-DAG and solution ephemeral-gap checks likewise
  treat that producing group as a slow-memory source for consumers in later groups.
- **P4 softmax** — fused online flash `(m,l)` with `exp(m_old−m_new)` rescale; the cone is verified
  EXACT by the shared descriptor (only `row_max→sub(x,m)→exp→row_sum→div`); stats and apply passes
  stage-2 pipelined when their rolled trip count is at least 2. The G7/G8/G9 device run found it
  numerically correct and 25.5% faster than its two-kernel cut, so exact softmax is default-on.
- **P4 layernorm** — stable streaming **Welford** (running count/mean/M2, Chan's parallel merge),
  reached only after proving the exact `sum(x)` / `sum(x*x)` / mean / variance / rsqrt / centered-apply
  algebra, then substituting stable `mean`/`var` into that cone. Its three-carry stats and apply passes
  are stage-2 pipelined under the same trip-count guard. ≤ the cut's accuracy at input mean
  0/100/1000/2000, **no NaN at +2000** (the dual-sum form NaN'd there). It was 24.7% faster than
  the zero-mean cut after G7/G8/G9 and remained correct at `+2000`; it stays opt-in until the
  extreme-shift envelope is measured cleanly. Temperature softmax,
  weighted-second-moment graphs, and patterns with an escaping internal stat cut and preserve semantics.

**Cube emit:** G-A ceil+clamp grid (non-uniform lone matmul tiles across cores — **device-validated**:
`[272,272]`→spmd(8), torch-exact; forcing an untiled 1×1 crashes, so G-A is load-bearing), ragged-K
peel, deep-T chained (tensor-level; #1908-xfail at lowering). k-loop is `ForKind::Pipeline`.

**Cost model:** C3 per-task overhead (device-validated no-regret), per-input G3 phase traffic, R0
reduced-axis coupling, granule-padded feasibility, the reduction source/work two-band floor, and
candidate-local P4 feasibility. Materialized cross-core reduction candidates carry an exact
`ColSumAtomicAdd` descriptor only after semantic, UB, and partition checks; upstream pointwise
ephemerals are validated and replayed at the emitted partial geometry. Their separate zero seed is
an ordered serial plan phase with grounded fill, store, task, and kernel-launch costs. The stack-local
`VectorStreamPlan` records pebble/scratch peaks; owns materialized/pointwise strips and streamed-reduction
init/rolled/tail/finalize phases; and gates each phase's A5 overlap on its actual loop stage. Cost is
the sum of phase rooflines, so barriers never hide work across phases. AutoFuse re-derives the plan for
the winning config. It now also owns element-balanced M/N logical partitions and their exact
`work_units`; `free_tile_alloc` carries DMA padding separately, so alignment cannot change the SPMD
count. Grounded source-DAG add/mul/div/exp/log/rsqrt/abs/sqrt/neg, scalar and broadcast variants,
part add/mul/max/min, and supported row/column reductions carry a compact primitive/geometry
descriptor. Costing replays that descriptor per planned strip/chunk/task, including count-mode,
reduction-layout row-expand barriers, PTO row/column fit formulas, and generated P1/P2 merges.
Composite research instances may retain the explicit `Generic` fallback; production PyPTO admission
is capability-gated. `CostResult`
stays at its pre-refactor 112-byte footprint (guarded at ≤128 bytes) for the local-search cache.
`create()` precomputes UB band intervals, flattened transient references, and phase-ordered op/input
lists once per subgraph. A materialized tile performs one linear replay; an overflowing strip/chunk
uses a logarithmic fit search over the same byte-weighted lifetime metadata. P1/P2/P4 add a constant
number of phases, not another combinatorial search. A
Release A/B microbenchmark of the 11-config tall-softmax sweep retained the identical aggregate cost
and reduced candidate evaluation by about 7.5%. Whole-suite profiling is dominated by partition
search and allocation, not this vector evaluator.

The candidate cache now release-publishes immutable entries through an explicit
`Empty→Writing→Ready` state, with a concurrent publication stress test. Its two default tables were
reduced from one million to 131072 slots, while the retention table is allocated lazily only when
retention-aware evaluation is used. Candidate evaluation no longer validates tilings
twice, and pointwise→matmul prologue constraints use one reverse-topological `O(N+E)` DP instead of
one downstream BFS per pointwise op.

**Vector refactor preservation audit.** The plan extraction was separated from the subsequent
model corrections so structural movement could be checked independently. Solver `10f8f8b`
(pre-plan) and `1fc542d` (plan authoritative and absent from `CostResult`, before phase pricing)
produce the same fixed-cost anchors: fused/separate pointwise `11003.7/22007.4`, fused/cut softmax
`30073.8/88186.9`, few-row reduction `10563`, and long streamed reduction `101566`. Current
`4ca1026` deliberately changed the phase-sensitive anchors. After logical-region identity,
per-source-tensor dtype/lifetime-exact UB planning (plus explicit conservative generated scratch),
and emitted-static-body ragged traffic, current descriptor-free
controls are softmax `22208.7/88373.3`, few-row
reduction `10097.6`, and pointwise `11003.7/22007.4`. These are intended consequences of phase-local
compute/traffic, serial edge phases, the reduction source/work floor, and exact task-grid replay—not
drift from moving schedule derivation into a plan.

The G8 descriptor path deliberately does not change those descriptor-free C++ benchmark anchors.
Real PyPTO problems now carry lowering semantics, so their source-op startup/count-mode work may
change; this separates an intended fidelity correction from the plan-refactor preservation control.

**Current host gates.** AutoFuse UT is 46 passed with no xfail; the solver suite is 461 passed
with the same 7 documented baseline failures. The vector checkpoint's device file
collects 51 A2/A3 cases. Four persistent cases
cover exact P4 softmax `[128,8192]`, Welford layernorm at input mean `+2000`, a scaled-softmax near
miss that must cut, and a P2 apply-only bias input whose expected MTE2 payload is `2×x + 1×bias`.

**Device-validated on 910B2:** vector correctness 51/51; exact P2/P4 phase traffic; phase-local
overlap; G5 logical-grid identity; G7/G8 algorithm/source replay; and G9 row-grounded task ranking.
The tall natural softmax is in the device-best cluster; exact softmax is 25.5% faster than its cut;
high-mean Welford is correct and the zero-mean path is 24.7% faster than its cut.
Cube G-A/ragged-K and recursive-plan correctness are also device-validated.

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

1. **Device-close the final host-only vector refinements.** Verify returned live-outs, capability
   declines, mixed-dtype UB, thin-seed `S=1`, aligned G6 fill waves, and the emitted ragged traffic
   equation. In the same run decide whether a stage-2 vector phase overlaps MTE2 and MTE3 as
   independent ports (`max(compute,in,out)`) or as the current summed DDR term
   (`max(compute,in+out)`). Do not change that roofline without silicon evidence.
2. **Keep Welford opt-in and document its numeric domain.** The device sweep places its FP32 ceiling
   near `mean/std = 5–6e4`; exact softmax remains default-on independently and `P4=0` preserves its
   fallback surface.
3. **Profile broader solver search before more vector micro-optimization.** Candidate-invariant UB
   topology and phase ordering are now precomputed, the 11-config P4 sweep is only a few microseconds,
   and `CostResult` still caches only scalar/config data. The end-to-end profile points to partition
   search/allocation; do not cache complete stream plans unless a new profile overturns that result.
4. **Close cube-only rollout policy:** the role-aware `CubeSchedulePlan`, heterogeneous recursive
   uniform-grid emit, exact phase-local K-window/seed cost, emitted reload multiplicity, lone
   clamped-overlap grids, and exact retained panels are implemented (§8). Nested pipe/FIXPIPE
   behavior, the 24-core wave shape, and retained-panel traffic/profitability are device-grounded.
   Before rollout, close pure-cube semantic/dtype admission, duplicate boundary-request feasibility,
   and the analytic global-roofline/seed/tile-multiplicity gaps. Then profile exact search cost and
   retention hit rate. Keep ragged recursive multi-op grids uniform-only unless workloads justify
   edge-specific request propagation.
5. **Complete mixed fidelity:** make the plan choose a real pipeline-item axis and active-group
   count, replace global-tile overlap with serial-versus-realizable phase costs, then implement the
   one-way and single-round-trip emit through `ExpandMixedKernel` → `InjectGMPipeBuffer` →
   `SkewCrossCorePipeline`. Full flash attention follows only after whole-FIFO multi-round-trip skew.

**Deferred (all decline *gracefully* today — correctness intact, not fused):** the ProblemBuilder
nested-arg gap (hoist nested compute-call args to SSA temps); P4 col-softmax / scale-then-softmax /
chained layernorm; the cube
ragged-K peel lowering test + deep-T decline logging; mixed cube+vector (a separate charter).

---

## 8. Cube-only plan and current fidelity boundary

The cube path now has the same solver-owned-plan discipline as the vector path, split across two
hardware levels.

**`CubeSchedulePlan` (cross-core and GM/L1).** The plan records exact spatial/split work units,
recursive producer requests, L1 pebble lifetimes, per-request GM K windows, output/L0C variants,
final drains, and the split seed. The request topology is built once in `create()`; the full plan is
reconstructed for a winning/forced configuration and is not stored in `CostResult`.

**`L0MatmulPlan` (L1/L0).** Cube costing now has two modes. The default analytic mode ranks outer
plans with the grounded fixed-base-tile cube/MTE1 surrogate and attaches only the semantic
Acc/L1/GM residency intent; `AutoTileMatmulL0` independently chooses detailed L0 geometry. Setting
`PYPTO_AUTOFUSE_EXACT_L0_COST=1` enables the prospective `-O3` mode: every candidate is priced with
the shared L0 chooser and the winning tensor phases carry a detailed record that AutoTile re-derives
and validates before creating Left/Right/Acc IR.

**Grounded nesting.** PTO's manual and automode A2/A3 GEMMs put the output tile outside K. One L0C
accumulator survives the complete GM→L1 and L1→L0 K stream, then drains once. A2/A3 has Acc→Mat/GM
but no Mat→Acc. AutoFuse now emits exactly this order. The outer loop is tagged GM→L1-only so its
stage depth does not multiply the child L0 ping/pong buffers.

**Exact-mode phase cost.** Uniform candidates put every full K window, including K=0, in one
stage-2 ring and price its fill/steady-state/drain, followed by serial K tails, exact ragged output
variants, and one final drain. Boundary requests are
charged once per emitted output tile, including the known LHS reload across N subtiles. Split seed
fill/store/tasks and its kernel-fill wave are explicit. The child L0 plan has the same phase
decomposition. Its already-grounded geometry ordering is retained until a per-iteration PTO event
term is measured; using phase granularity alone would falsely prefer baseK=16.

**Dtype contract.** BF16/FP16 operands accumulate in FP32 L0C. An internal producer narrows once to
BF16/FP16 Mat, matching PTO's fused-chain kernel; roots narrow/store to their declared output.
Same-type FP32 internal L1 handoff is not an A2/A3 instruction. Exact mode declines it; analytic
currently ranks then falls back, which is a TODO below. Direct Mat→GM store is legal.

**Host validation.** PTO Fusebox reports 496 passing checks with the same six documented baseline
failures; the full AutoFuse file reports 56 passing tests. Compiler coverage includes
natural/forced lone matmuls, BF16 recursive trees/fan-out/deep chains, FP32-chain decline, split seed,
ragged K, multi-window output residency, a 192 KiB internal region, descriptor consumption, Torch
numerics, and PTOAS-backed full lowering. The former strict chained-matmul xfail now passes.

The latest host closure fixes two compiler-side lifetime mismatches without changing cube costs.
Early simplification prevents a dead branch from allocating another persistent L0C accumulator;
serial child init/tail extracts keep order and reuse a rolled Left/Right bank instead of entering
an enclosing GM→L1 prefetch tier.
The formerly failing exact A4, B16/B24/B48, and split-K S16/S32 candidates now reach final allocation
with one accumulator and the planned two-bank operand ring.

**Silicon isolation.** The forced pure-cube `[192,64]@[64,256]` four-window schedule now passes on
910B2 with 48 logical AIC blocks. The same producer followed by a separate AIV bias epilogue still
fails at 12 blocks and one K window. DFX proves the allocation and covered AIC→AIV dependency, so
that residual is a mixed/orchestration FIXPIPE-visibility issue, not a cube-only schedule blocker.

**Silicon ranking.** For clamped A8/E12, analytic selects device-best A8 while exact selects E12,
which is 7.3% slower. A fixed grid emits a byte-identical binary in both modes. MTE2 is the critical
per-task pipe and the final TSTORE already has a complete `PIPE_ALL` drain; the difference appears
in the runtime scheduler span. Wider sweeps found no transferable per-task correction: one family
was U-shaped while another improved with more tasks. Exact remains incomplete at this system
boundary, but no scalar correction is supported.

**Remaining gaps:**

1. Add one candidate-invariant homogeneous cube capability descriptor. Reject unsupported
   transpose/layout, operand dtype, and Acc→Mat cases before solving and recheck at emit; use a typed
   integer zero for an INT8→INT32 split seed.
2. Make analytic cost follow emission: sum request-local phase roofs, add serial init/tail/drains and
   the split seed, and count GM→L1/MTE1 at emitted output-tile multiplicity. Do not deduplicate a
   boundary request across matmuls without a group-level shared-L1 lifetime; otherwise reject it
   before ranking.
3. Exact initial L1 feasibility uses whole-request M/N before output subtiles are known, so couple
   child-tile and K-window feasibility before pursuing broader exact-mode ranking.
4. Non-uniform buildable cost/emission: lone split=1 has `ClampedOverlap`; ragged split-K,
   sub-fractal valid M/N edges, and unequal multi-op grids decline consistently.
5. Exact retention compares none/LHS/RHS/both, serially preloads the panel, keeps it live across the
   output-tile loop, and emits local extracts. Device evidence confirms byte-exact traffic and a
   resolved reuse win, including bounded natural plans. Profile evaluation cost and hit rate before
   keeping it exact-only or adding a cheap analytic surrogate; analytic remains unchanged today.
6. The constant-tile, constant-K sweep found no scalar per-task term. Keep the existing
   `ceil(work_units/24)` wave shape; the remaining small-count/within-wave queue effects are too
   small and nonlinear to fit, and vector C3 remains separate.
7. Keep the flat GM->L1 bandwidth term as a ranking surrogate, not a request-accurate transfer law.
   The measured 256-byte class is real, but the tested compact replacements fail direct prediction;
   a faithful dynamic phase replay changes none of the tested fixed-K B16 decisions. Revisit only
   with a direct model that generalizes and improves within-problem ranking.
8. Different M/N/K shapes and recursive roles are supported. Ragged multi-matmul grids would need
   edge-specific valid/physical regions and lifetimes propagated through the request DAG.
9. Remove full plan construction from the exact candidate hot path only if full-solver profiling
   shows it matters; exact added about 1 ms to the measured full compiler pipeline. Promote
   retained, narrowing, split-atomic, and heterogeneous recursive cases into the persistent 910B2
   device surface before relaxing the generic cube-emitter guard.

The authoritative obligation table and validation ladder are in
`docs/en/dev/proposals/autofuse_cube_cost_model_emit_contract.md`.

## 9. Operational gotchas (don't relearn these)

- **Build MAX 2 cores.** Use absolute build paths (a `cd 3rdparty/pto-fusebox` persists across shell calls
  and mis-targets the build).
- **FORCE_PLAN is static-cached per process** → one force per fresh subprocess; the `group[0]` solver
  log is misleading under force (prints the argmin, not the forced tile) — trust the emitted
  `pl.spmd(N)` count.
- **Nested-arg DSL** (`exp(sub(x,m))`) silently drops the inner op from the solver graph → misses P4 +
  mis-costs. Write NAMED temps.
- **Welford count column** must be derived from a reduction output (col-major), NOT `tensor.full`
  (row-major → trips `ResolveBackendOpLayouts`' col-vector reshape → lowering crash).
- **Remote access:** SSH is currently blocked; use the public HTTPS PTO Fusebox submodule and publish
  it before the parent when both repositories change. NEVER push / open PRs without explicit order;
  NEVER add AI co-author lines; never hack test expectations.
- **Device runs:** a separate agent checks out `fusion-scheduler-vector-stream-plan`, builds, and runs
  on 910B2 with a working
  `device_wall effective_us` STRACE path; `benchmark()` hits error 507018 (avoid). Fingerprint-gate on
  HEAD + PTO Fusebox hash first.
