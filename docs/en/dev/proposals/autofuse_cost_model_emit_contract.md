# AutoFuse vector kernels — cost-model assumptions ↔ emit obligations

**Purpose.** In AutoFuse, a solver **Solution is not a number — it is a specification
of a kernel algorithm.** Every term in the cost model implicitly assumes the emitted
kernel implements a particular algorithm. If the emit implements a *different* (usually
cheaper-looking) algorithm, the cost is fictional and the solver's ranking is wrong.

This document enumerates, for the **vector** path (pointwise + reduction on the Ascend
910B AIV cores), **what the model assumes** and **what the emit must therefore produce**.
It is the fidelity contract. Scope: vector kernels only (no cube / no mixed cube+vector).

Code references are `3rdparty/mlsys26/src/core/ascend910b_cost.cpp` (cost) and
`src/ir/transforms/auto_fuse_pass.cpp` (emit) unless noted.

---

## 0. The core principle

```text
                         ┌─ group 1 ─▶ kernel 1 ─▶ launched over its grid (1 invocation / output tile)
tensor DAG ─▶ PARTITION ─┼─ group 2 ─▶ kernel 2 ─▶ launched over its grid ...
   (one global           └─ group N ─▶ kernel N ─▶ ...
    decision)
```

A Solution has **two levels**:

### Program level — the partition (ONE global decision)

The solver splits the whole tensor DAG into a set of convex groups. **Each group becomes
exactly one kernel** (outlined to an InCore function downstream). Partition *is* the fusion
decision — it is not a per-group property; it is the single choice that produces the groups.
The cost model scores a Solution as the sum over groups (each group costed independently by
the roofline of §1).

### Per group — the kernel algorithm (ONE spec per group)

For each group the Solution fixes **six** things that together define that kernel's algorithm:

1. **sink tile `(w,h)`** — the output-tile shape (a property of the group's *boundary output*).
2. **role back-propagation** — the sink tile propagates *against dataflow* to fix every inner
   tensor's per-axis role (see §3).
3. **pebbling order** (`dfs_order_`, `:924`) — the forward execution order; sets band liveness
   → UB feasibility (see §4).
4. **grid `parts_m × parts_n`** — how many tiles the output is cut into.
5. **split `S`** — cross-core reduced-axis split with atomic-add merge (reductions only).
6. **materialize vs stream** — whether the reduced axis fits UB or must be chunked.

### Instantiation — ONE kernel body, launched many times

The emitted kernel is a **single body** (the per-tile algorithm), wrapped in a `SpmdScopeStmt`
and **launched `parts_m·parts_n` times — one invocation per output tile**. Every invocation
runs the *same* body, differing ONLY in the block index it decodes (`get_block_idx`) into its
tile **offset** `(mi,ni)`; that invocation slices its operands at that offset, computes its one
tile, and assembles it into the output region. So: **"the kernel" = the per-tile algorithm; the
grid is how many (tile, offset) instances of it run across the cores.** (A streamed reduction
additionally runs its inner reduced-axis chunk loop *within* each invocation — §5.)

This is why the roofline in §1 is a *per-tile* algorithm scored across `U = parts_m·parts_n`
invocations (compute divided by occupancy; DDR divided across the launched cores' pipes).

Two distinct walks over the DAG matter (within a group):

| walk | direction | produces | governs |
| ---- | --------- | -------- | ------- |
| **role back-prop** | reverse-topo (sinks→inputs) | each tensor's tile shape / role | *what* each op computes per tile — the algorithm |
| **pebbling order** | forward-topo | execution sequence + band liveness | *when* + feasibility (peak UB) + emit replay + MemoryReuse |

---

## 1. The roofline (what the model computes)

For one fused vector group at a fixed tile, over `C` AIV cores (910B: 48), the emitted
barriers define the costing phases:

```text
latency = Σphase roofline(phase)
roofline(p) = max(compute_mk[p], ddr[p])   [only for a stage-2 rolled loop]
              compute_mk[p] + ddr[p]      [serial body / peeled init / tail / finalize]
```

For a streamed reduction the ordered phases are `stats_init` (serial), rolled `stats`,
`stats_tail` (serial), an optional serial finalize, rolled `apply`, and `apply_tail` (serial).
No phase may hide work across a barrier.

### Compute (`GroundedVectorOpCompute`, `WaveComputeCycles`)

- The adapter records the pto-isa primitive family and emitted geometry for grounded source ops:
  add/sub/max/min, mul, div, exp, log, rsqrt, scalar add/sub/mul, and their row/column
  broadcast forms, plus the supported row/column reductions. Their coefficients come from the same table as generated P4 work. Unsupported
  source ops retain the historical `vec_slope`/`vec_fixed` fallback rather than being assigned a
  guessed primitive.
- Candidate costing replays each grounded op at the exact valid frame of one emitted materialized
  strip or streamed-reduction chunk, then multiplies by that phase's loop iterations and logical
  tasks. `repeat = ceil(elems/epr)` for flat ops; row/column broadcasts use their strided
  `rows·ceil(cols/epr)` geometry. A row-expand binary in a reduction-layout group additionally pays
  the emitted `vbrcb` and `PIPE_V` barrier needed for its col-major statistic; the pure-pointwise
  row-major form does not. Count-mode floors are applied per invocation.
- Reduction work is replayed at the same frame. FP32/FP16 row sum/extrema use PTO-ISA's fit formula
  `round(slope(W)·valid_rows·W+bias(W))`; exact table widths reproduce PTO and other emitted widths
  interpolate adjacent anchors. Column reductions and unsupported dtypes retain the explicit
  structural fallback pending their own grounding audit. P1/P2 also price the thin add/max merge
  emitted after every non-initial statistics chunk; that merge is absent from the source DAG.
- Head+tail is paid **once per back-to-back pointwise run in every emitted invocation**. A
  barrier-bearing row expansion starts a new run. An op in both phase cones (softmax's
  `sub`/`exp`) is charged in both because the apply pass recomputes it.
- Spread over the grid: `compute_mk = WaveComputeCycles(total, U, C) = total·ceil(U/C)/U`,
  `U = num_tiles = parts_m·parts_n`. Filling toward `C` lowers it; past `C` costs extra waves.

### DDR (`:1744`, `dma_pen :1779`)

- Counts **only boundary tensors**: external reads (`io_in`) + boundary writes (`io_out`).
  Each input has a phase mask derived from the DAG: stats+apply inputs read twice, stats-only
  and apply-only inputs once. Size-1 broadcast axes are re-read per emitted chunk; uniform
  clamp-overlap strips are charged at the actual planned strip geometry.
- **Ephemeral intermediates contribute ZERO DDR** — they stay in UB. *This is the entire
  fusion win.*
- Tile enters only via the **DMA-shape penalty** `dma_pen = max(1, vec_reg_bytes/(w·dtype))`
  (a sub-burst-narrow `w` is charged more) and via **pipe-sharing**:
  `ddr = io_in·dma_pen/par(active, bw_gm→ub) + io_out·dma_pen/par(active, bw_ub→gm)`,
  `active = min(U, C)`, `par(active,peak) = min(active, HBM/peak)`.

### Worked example — two fused pointwise ops `x → t1=op1(x) → y=op2(t1)`, all `[H,W]` fp32

- `total_compute = 2·slope·ceil(H·W/64) + (head+tail)` (op1 pays head+tail, op2 continues).
- `compute_mk = total_compute · ceil(U/48) / U`, `U ≈ (H/h)·(W/w)`.
- `io_in = H·W·4·ub_in`, `io_out = H·W·4·ub_out`, `t1 = 0` (ephemeral, on-chip).
- `ddr = io_in·dma_pen/par + io_out·dma_pen/par`.
- This pointwise example has one planned body phase, so its latency is that phase's
  `max(compute_mk, ddr)` when the body loop is stage-2, otherwise their sum.

The tile size changes occupancy, DMA efficiency, and the exact emitted strip traffic
(ragged clamp overlap and repeated broadcasts); phase membership changes recomputation.

---

## 2. Feasibility is a PEBBLING problem (not a static sum)

The model does not check "does the whole group fit UB." It checks **peak simultaneously-live
bands over the pebbling order** (`vector_peak_ub :1241`):

- an ephemeral tensor `t` occupies a UB band across `[producer, last consumer]` in `dfs_order_`;
- at each step, peak = live bands + transient input/output tiles;
- a reduction materialization includes both its source tile and tensor-lowering work/layout tile;
- feasible ⟺ `max over steps ≤ UB`.
- `dfs_order_` (post-order DFS from sinks) is chosen to **minimize** that peak — finish a
  branch (and free its bands) before starting a sibling. This is the pebble game.

An intermediate that is **free in DDR (fused) still costs a pebble in UB.** The order is what
keeps that affordable, and it is part of the Solution.

---

## 3. The sink tile back-propagates to define the per-step algorithm (`:629–671`)

The tile `(w,h)` is a property of the **sink output**. It propagates reverse-topo to give
every tensor a per-axis **role**:

- **Pointwise** → inputs follow the output tiling `(out_h,out_v)`, EXCEPT an input with
  extent 1 on a tiled axis is a **broadcast** → `FIXED_1` (read whole, reused across tiles).
- **Reduction** → the input is `FIXED_1` on the **reduced** axis (read full) and follows the
  output on the free axis. `[H,W]→[1,W]` (col reduce): input role `{tiled-w, FIXED_1-on-H}`.
- **Matmul** → LHS/RHS take the contraction role (`FROM_NK`) on K; a non-sink matmul reads
  `FIXED_1` on its non-shared axis.

`FROM_NT*` = "slice to the tile"; `FIXED_1` = "read in full / broadcast / reduced-axis".
**Combining the two walks gives the algorithm:** for each op in `dfs_order_`, execute it at
its back-propagated operand tile shapes.

> This back-prop is the origin of the C2 subtlety: for a bare `col_sum` the sink output is
> `[1,w]`, so the *output* extent on the reduced axis is 1 — but the *input* role there is
> "read full `[H,w]`". A UB test that reads the output-derived `cfg.h=1` misses the full-axis
> read; the correct test couples the reduced axis to `reduced_extent_`.

---

## 4. Assumptions ↔ emit obligations (the contract)

| Ref | The model assumes… | So the emit MUST… | Status |
| --- | ------------------ | ----------------- | ------ |
| A1 | compute spreads over `U = parts_m·parts_n` invocations (wave makespan) | emit ONE per-logical-region kernel body and launch it over the exact solver grid (`SpmdScopeStmt` + `get_block_idx` → offset) | ✅ host-fixed, device follow-up pending. `VectorStreamPlan` owns element-balanced M/N partitions and `work_units`; cost, diagnostics, and emit consume that count. DMA padding is recorded separately and cannot change the launch grid. |
| A2 | ephemeral intermediates cost **0 DDR** (fusion win) | keep intermediates **on-chip** (UB scratch), never round-trip them through DDR | ✅ honored |
| A3 | each op runs at its **back-propagated role** shape | slice `FROM_NT*` operands to `[h,w]`; read `FIXED_1` operands in full (broadcast / reduced-axis) — `emit_strip` | ✅ honored (G4 done, 2026-07-09). `emit_strip` slices any 2D operand per-axis: a full axis follows the tile `[sh/sw]` at `[smi/sni]`, a size-1 (broadcast/`FIXED_1`) axis stays `[1]` at offset 0; the op replay re-infers the broadcast. Covers `[1,N]` bias-add, `[M,1]` scale / reduced-axis stat, `[1,1]`. Other 2D shapes still decline. |
| A4 | UB feasibility = peak live bands over the **pebbling order** | replay ops in `dfs_order_`; let MemoryReuse alloc/free UB per the emitted liveness | ✅ honored (order matched) |
| A5 | roofline `max(compute, DDR)` only within a loop that overlaps load k+1 with compute k | emit that exact loop **software-pipelined / double-buffered** (`ForKind::Pipeline` + `kPipelineStagesAttr`) and keep barriers serial | ✅ `VectorStreamPlan` owns materialized/pointwise row+width strips and P1/P2/P4 init/rolled/tail/finalize phases. Cost is `Σphase roofline`; peeled phases use `compute+DDR`, and sub-register or short rolled loops stay sequential. |
| A6 | reduced-axis split `S` = S cores reduce slices, **atomic-add** merge | build the cross-core split (seed + `SetAtomicAdd`) — realized ONLY for a bare **sum col-reduction** sink `:1630`; else fall back to serial | ⚠️ partial — model must NOT price `S` where the emit declines it (see C2) |
| A7 | streamed input reads are phase-specific: an operand may be stats-only, apply-only, or both | stream with running stats; derive each phase's dependency cone and read only its inputs | ✅ G3 uses per-input phase masks. `x` in P2/P4 is normally read in stats+apply; an apply-only scale/bias is charged and emitted once. |

“Wave” in A1 is an analytical makespan, not an emitted runtime construct. The runtime receives `U`
ready SPMD tasks and schedules them without affinity controls. A logical region therefore stays one
task; its UB-resident pointwise strips or reduced-axis chunks execute inside that task on the same
core. For `U > C`, `ceil(U/C)` describes the queue-completion bound and the model also charges every
task plus every kernel-fill round; the emitter does not try to assign explicit wave identities.

---

## 5. The reduction problem = the flash-attention problem

If the reduced axis does not fit UB, the row/column cannot be materialized — the one-pass
"load the whole row and reduce" algorithm is **not expressible** (the reduced axis is
`FIXED_1`, can't be spatially split without a cross-core merge). You are forced to **stream
the reduced axis in blocks carrying running statistics** — this *is* the flash-attention
algorithm (online softmax), applied here to a DDR-resident reduction rather than a fused
matmul's scores.

### The online (flash) algorithm

- tile the reduced axis into blocks; stream them single-core;
- maintain running stats (max `m`, sum `l`); per block rescale the accumulator by
  `exp(m_old − m_new)` and add the block's contribution;
- persist only the small `[·,1]`/`[1,·]` stats — never the full row;
- **reads the data once during the stats pass**; a spanning output still requires the second apply
  read described by A7.

### The P-ladder

| tier | shape | flash content | status |
| ---- | ----- | ------------- | ------ |
| **P1** | bare reduction (sum / max) | online but trivial — sum associative, running max; no rescale | built, device-confirmed (col reductions) |
| **P2** | reduction → pointwise | accumulate, then re-stream to apply (2 passes) | built |
| **P3** | multipass softmax/layernorm (>1 reduction) | one stream per statistic pass (2–3 reads) | retired |
| **P4** | online-stats | one stats pass with softmax `(m,l)` rescale or layernorm Welford, then apply | built for exact canonical row-softmax/layernorm behind `PYPTO_AUTOFUSE_P4` |

### Why P4, not P3

A7 prices **two reads** for a spanning result: one online-stats read and one apply read. P3 would
add extra statistic reads and run materially slower than priced, so streamed softmax/layernorm must
use **P4 (online/flash)**. By A5 each phase receives overlap credit only if its loop is actually
pipelined. Both P4 rolled loops pipeline only when their trip count and strip size satisfy the
double-buffer floor; peeled init/tail/finalize work remains serial and separately costed.

### Scope caveat

Real flash *attention* fuses `Q@Kᵀ → softmax → P@V` with the scores kept on-chip — a **mixed
cube+vector** kernel, deliberately out of scope for now. Here we apply the flash *algorithm*
to a **vector-only, DDR-resident** reduction (softmax / layernorm / rmsnorm over a large
reduced dim). Same online-streaming-with-running-stats structure; different block source.

---

## 5.1 The decision rule — when 1 read (flash) vs when we must reload

Given a group whose reduced axis **R** must stream (doesn't fit UB), two independent
yes/no tests decide the number of streaming passes (= how many times each input band is read):

**Test 1 — are ALL reductions over R FOLDABLE?**
Foldable = the reduction is a **fixed-size associative running state** (§2 abstraction):
the log-sum-exp family (sum, max, min, prod, softmax-`(m,l)`) and moments (mean/var via
Welford). NOT foldable = order statistics (median, quantile, top-k, sort) — no fixed-size
exact summary.

- **Not foldable** → the reduced axis *cannot* be streamed correctly. Either it must
  materialize (fit UB) or the group is **declined**. *(correctness gate — never fold an
  order-statistic.)*
- **Foldable** → the stats can be gathered in ONE online pass (flash), exactly. Continue to Test 2.

**Test 2 — does ANY group output SPAN R?** (a live-out whose extent along R is > 1)

- **No (output-folded)** → the reduction result *is* the output, or is consumed only by
  further R-reductions that fold into the running state. **`stream_passes = 1` — read once.**
  (P1; attention.)
- **Yes (output spans R)** → producing each R-position needs the *finalized* whole-axis
  stats, so an APPLY pass over R is mandatory *after* the stats are final.
  **`stream_passes = 2` — read twice** (1 online-stats pass + 1 apply pass). (P2 single-stat;
  softmax / layernorm / rmsnorm.)

**General (chained spanning reductions):** `stream_passes = 1 + depth`, where `depth` = the
number of *finalize-reduce → per-element-apply → feed-another-reduce* stages. Standard
norms/softmax have `depth = 1` → 2 passes. `depth ≥ 2` (a reduction consuming the applied
output of an earlier spanning stage) is where >2 reloads come from — not in our current op set.

```text
reduced axis fits UB?
├─ yes → materialize (one tile, no streaming)                         [non-streamed path]
└─ no  → all R-reductions foldable?
         ├─ no  → DECLINE (order statistics can't flash)              [correctness gate]
         └─ yes → any output spans R?
                  ├─ no  → FLASH: stream_passes = 1 (read once)       [P1; attention=1-pass]
                  └─ yes → stream_passes = 2: flash-stats + apply     [softmax/layernorm = P4]
                           (general: 1 + depth)
```

**Key point (what flash does and does not save):** flash never removes the *apply* pass for a
spanning output — that reload is fundamental (the output element needs the finalized stat).
Flash removes the *extra stats passes*: naive softmax is 3 passes (max → sum-exp → apply)
because sum-exp needs the max; flash folds max+sum into ONE online stats pass → **2 total**.
The saving is one DDR read, not single-pass.

## 5.2 Making the cost model aware

The model derives and costs the emitted phase graph:

- exact P4 descriptors establish `foldable`; `spans = any live-out extent along R > 1`;
- per-input phase masks price stats-only, apply-only, and shared traffic;
- shared ops in the apply cone are recomputed and charged again; online chunk merge/startup is priced;
- if `streams && !foldable` → **infeasible** (`cost = inf`) so the partitioner never picks a
  group it cannot emit correctly.

The resulting traffic stays between the old flat `×1` (too optimistic) and the retired
`#reductions+1` (too pessimistic); the truth for canonical spanning softmax/layernorm is 2.

## 5.3 Emit design (per case)

- **output-folded, 1 pass** (P1): as today — SPMD over the free axis, inner chunk-accumulate the
  single stat, assemble the `[·,1]`/`[1,·]` output.
- **spanning, single stat** (P2): pass 0 accumulate the stat, pass 1 apply over R chunks; both
  full-chunk loops use `ForKind::Pipeline` when their rolled trip count is at least 2 (A5).
- **spanning, multi-stat foldable** (softmax / layernorm — **P4, built for exact canonical cones**): pass 0 = ONE
  online streamed pass maintaining the running stats with the exact rescale (softmax `(m,l)`
  with `exp(m−M)`; layernorm Welford `(count,mean,M2)`); pass 1 = re-stream R, apply the
  finalized stats per element, assemble. Both full-chunk loops use `ForKind::Pipeline` when their
  rolled trip count is at least 2; multi-carry pipeline lowering is validated by the P4 end-to-end
  tests. This replaces the naive 3-pass P3.
- **non-foldable over a streamed axis**: decline (matches the `cost = inf` gate).

---

## 6. Fidelity status (audited 2026-07-08 by 4 independent reviewers against this doc)

Confirmed accurate against the code: the roofline math (§1), pebbling feasibility (§2), all
role back-prop rules (§3), ephemeral = 0 DDR (A2, both sides), the emit replaying in
`dfs_order_` (A4 — `flush()` stable-sorts by `execution_order()` before emit), multi-role
tensor tracking, the C2 split gate, and the S2 emit predicate. The gaps below are ranked;
several refined the ⚠️ rows above.

### Root streaming signal — fixed

**R0 — reduced-axis coupling. [FIXED.]** `vector_peak_ub` now sizes a reduction input from the
tensor's full reduced extent even when the sink output is thin. Feasibility, compute, and split
gates therefore agree on materialized versus streamed execution.

### 🔴 Correctness

**G1 — large softmax/layernorm silently overflow UB. [FIXED; exact P4 capability built.]**
The stream gate requires `p1_nreds == 1` (`auto_fuse_pass.cpp:1231`), so a 2-reduction group
cannot stream; it used to fall through to a full-reduced-axis materialized tile that overflowed
(hard `AllocateMemoryAddr` failure). **Fixed** (mlsys26 `603ec35`, pypto `69d7f508`): a new
`Problem::allow_model_ahead_multi_reduction_stream` flag — the AutoFuse adapter sets it false
(buildable), so a streamed >1-reduction group is **infeasible** and the partitioner **cuts** it
into single-reduction (streamable) + pointwise pieces. An **unfused softmax IS buildable**, so
large softmax/layernorm now **compile** (verified: `softmax[128,16384]` builds, was a crash).
Emit defense-in-depth remains. With P4 enabled, one shared exact semantic analysis records complete
canonical softmax/layernorm op sets; the model admits only an exactly equal candidate and the emitter
consumes the same descriptor. Temperature/scaled softmax, weighted moments, chained norms, and
multi-sink escapes cut rather than being reinterpreted.

### Roofline

**G2 — A5 — phase rooflines and solver-owned loops — FIXED (2026-07-12).** Both streamed passes emit
`ForKind::Pipeline` + stages=2 (mirroring the pointwise strip): the accumulate pass 0 (the
running accumulator or P4 `(m,l)` / Welford `(mean,M2,count)` IterArgs) and the apply pass 1
(assembles disjoint reduced-axis chunks, lowered to in-place stores by
`RewriteReturnedAssembleLoopToStore`). `LowerPipelineLoops` double-buffers only the per-chunk load
while keeping loop-carried state single-buffered/persistent. Pipelined only when the rolled chunk trip
is at least 2 (nothing to overlap otherwise). `VectorStreamPlan` now owns the reduction chunk and
trip counts; stage-2 chunk sizing duplicates source-DAG transient bands while leaving carried state
single-buffered. `compute_cost` uses the plan as a stack-local derivation and the emitter re-derives the same
plan for the winning config. Materialized/pointwise row+width strips use that same plan; the old
quadratic emitter-side liveness/scheduling scan is gone. Costing sums barrier-separated phase
rooflines: init/tail/finalize serialize, while each eligible rolled loop independently receives
`max(compute,DDR)`. The shared P2/softmax/Welford carried-loop and spanning-apply builders consume
the plan's stage count directly.

### ⚠️ Cost fidelity

**G3 — A7 — streamed input multiplicity — FIXED per input.** Candidate-invariant DAG cones assign
each boundary input to stats, apply, finalize, or body. Costing charges the exact emitted chunks for
those phases; a stats+apply `x` reads twice, while an apply-only scale/bias reads once. The emitter
replays the same dependency cone and treats substituted online statistics as leaves.

**G4 — A3 — broadcast priced-fusible but emit-declined — FIXED (2026-07-09).** The emit now
builds a broadcast operand (one axis full, the other 1): `emit_strip` slices per-axis (full axis
follows the tile, size-1 axis stays `[1]` at offset 0; the op replay re-infers the broadcast).
Covers `[1,N]` bias-add (FFN/attention), `[M,1]` scale / reduced-axis stat, `[1,1]`. This also
unblocked **G3** (its accurate pricing routes a cross-group `[M,1]` stat that the emit can now
take), so G3's 2× read is applied unconditionally (was gated on the model-ahead flag until G4).
Other (ragged) 2D operand shapes still decline.

**G5 — A1 — logical-region identity — DEVICE-CLOSED (2026-07-13).** Device work exposed
the old mismatch: forced `8192,11,1,12,1` was costed as 12 tasks but DMA alignment changed the
streamed free tile and emission to 8 blocks. The fix deliberately keeps the user's solver choice:
`work_units = parts_m·parts_n = 12`, with an element-balanced 11/10-row ownership partition.
`free_tile_alloc=16` records the FP32 UB/DMA allocation independently; reduced-axis `chunk` remains
the inner stream inside each logical task. Costing uses the same logical count for waves, active GM
pipes, per-task overhead, `CostResult`, and traffic replay; emission uses it for `pl.spmd(12)` and
reconstructs the balanced offsets. On 910B2 every forced grid matched `pl.spmd(N)`; the padded
`h=11` plan loaded only 11 valid rows per task, and the repaired natural `[128,8192]` argmin became
the device-best `h=8`, 16-task plan. Cost traffic now sums the exact logical 11/10 partition rather
than multiplying all tasks by the maximum region extent.

**G6 — A6 residual (C2 DONE).** Split still priced for materialized max/row reductions the emit
declines (bare-`col_sum`-only, `:1630`). Bounded, unmeasured; left as-is pending a probe.

**G7 — P4 algorithm-specific compute — HOST-FIXED; SILICON RANKING FOLLOW-UP REQUIRED.** Phase masks
price the source DAG, but P4 stats emission builds a different online algorithm. Welford emits chunk
mean, centered M2, and Chan merge operations; softmax emits `(m,l)` rescale/correction operations.
These are not represented by the original dual-sum/softmax cones. `VectorStreamPlan` now carries a
compact, fixed-size primitive tally for stats init, one repeated stats update (also used by a ragged
tail), and finalize. Candidate costing consumes that tally directly: wide `[free,chunk]` and thin
`[free,1]` work use the grounded pto-isa add/mul/div/exp/scalar coefficients and count-mode floors.
The wide broadcast subtraction is represented separately as the composite `TROWEXPANDSUB` cost
(`vbrcb + barrier + vsub`, including its row-strided repeat geometry), rather than as a plain add;
each logical task pays its own barrier-separated reduction trees before the normal wave makespan is
applied. The apply cone remains a source-DAG replay with the online stats substituted. The emitter
re-derives the shared descriptor for the winning P4 kind and rejects a mismatch. No fitted surcharge
was added, and `CostResult` remains unchanged.

The first 910B2 run found the correct Welford kernel 33.6% slower than its zero-mean cut; after G5
repaired the grid, the same comparison became 18.8% faster while retaining high-mean accuracy. That
second fingerprint predates exact G7/G8/G9 compute. Therefore G7 closes the representational
model↔emit gap on host, not the decision-oracle validation: re-run natural and forced fused/cut plans
on silicon before enabling layernorm P4 by default.

**G8 — source-DAG strip/chunk compute — HOST-FIXED; SILICON RANKING FOLLOW-UP REQUIRED.** The old
source-op path first cost a full tensor and then fractionally scaled it into a phase. That preserved
the slope term but paid a fixed startup only once (or even fractionally), despite the emitter replaying
the op once per materialized strip, P2/P4 chunk, and logical task. It also treated scalar `adds/muls`
as ordinary binary add/mul and priced a row broadcast as a flat binary instruction. The adapter now
attaches a compact `(primitive family, emitted geometry)` descriptor to each supported source op.
`VectorStreamPlan` costing consumes the descriptor at its strip/chunk frame, including per-invocation
startup, count-mode floors, row-expand `vbrcb+barrier`, exact scalar coefficients, source reduction
trees, and the generated P1/P2 accumulator merge. The row-expand composite overhead is gated by the
candidate's reduction layout, matching the emitter's current layout decision. The descriptor is candidate-invariant `Op`
metadata; it is not cached in `CostResult`, whose local-search footprint remains unchanged. Dumped
problems serialize the descriptor so offline replay takes the same path. Descriptor-free benchmark
problems retain their old scalar anchors intentionally; real PyPTO P2/P4 rankings can change and need
the next silicon rerun. Unsupported primitives remain explicit `Generic` fallbacks, so this closes the
canonical softmax/layernorm apply cones without claiming that every PyPTO vector op is grounded.
The explicit reduction descriptor also selects this path for a bare P1 kernel, so its per-chunk
reduction tree and generated add/max merge are not lost merely because the graph has no pointwise op.

**G9 — row-aware reduction work — HOST-FIXED; SILICON FORMULA FOLLOW-UP REQUIRED.** The original
`45·(ceil(cols/epr)-1)+51` row-reduction tree came from PTO-ISA's stub perf simulator. Its count-mode
implementation intentionally drops repeat work and is rows-independent; PTO-ISA's documentation
warns that real hardware is not. The A2/A3 fit backend instead grounds `TROWSUM` and `TROWMAX` as
`round(slope(cols)·valid_rows·valid_cols+bias(cols))`, with fit tests against cycle profiling. G9
ports those FP32/FP16 formula anchors, distinguishes sum from max/min in both source-op descriptors
and generated P4 work, and replays each reduction at the plan's valid `[free_tile,chunk]` shape.
Exact tabulated widths reproduce PTO-ISA exactly; legal AutoFuse-only widths interpolate adjacent
total-cycle anchors, and widths beyond the table continue proportionally from the last anchor.
Unsupported dtypes and legacy descriptor-free research inputs retain the old structural tree
explicitly. No scheduler constant was fitted, `CostResult` is unchanged, and the existing
`WaveComputeCycles(total,U,C)` equation remains intact: 96 eight-row tasks are two waves of half the
per-task row work, not twice the wall of 48 sixteen-row tasks. The host `[768,8192]` softmax sweep now
ranks U48 below U32 and chooses 48 tasks naturally (U12/U32/U48/U96 =
344722/163497/159365/173609 model-cycles). Device work must validate the interpolated emitted chunk
widths and the post-G7/G8/G9 natural/forced ranking. Column sum/extrema are now classified exactly,
but still use the legacy column-tree cost pending their own PTO fit audit.

### Minor / doc-completeness

- **Offline task-cost replay — FIXED (2026-07-13).** Problem JSON now serializes and reloads
  `per_task_overhead_cycles`; a dumped generic-emitter problem therefore retains C3's 64-cycle
  per-logical-task term instead of silently re-ranking grids offline.
- **Granule-padding feasibility fiction — FIXED (2026-07-09, BUG-G1THRESH).** The emit allocates
  `AlignUp(sh,g)×AlignUp(sw,g)` tiles (`g = vec_dma_align_bytes / group_min_dtype_bytes`); the cost
  model's `vector_peak_ub::tile_bytes` used unpadded `min(cfg,dim)`, so a thin free axis (M-tile
  3→8, ~2.7×) was under-counted → an over-UB group looked materializable → the emit overflowed
  `AllocateMemoryAddr` (softmax/layernorm N=4096/8192). Both `vector_peak_ub` AND the emit's own
  materialize-vs-stream trigger now count the padded footprint (via `Problem::vec_dma_align_bytes`,
  sourced from `BackendHandler::GetVectorDmaAlignmentBytes`), so model and emit agree: width always
  padded, height padded for reductions (col-major). This also makes the streamed-chunk sizing
  granule-faithful — closing most of R3.
- **Reduction source/work materialization floor — FIXED (2026-07-12).** Tensor reduction lowering
  allocates a work/layout tile alongside the source tile. `vector_peak_ub` now counts both bands, so
  a reduction that needs more than UB streams instead of reaching `AllocateMemoryAddr` and failing.
- `dfs_order_` is a greedy topo-tie-break heuristic, not a provably-minimal pebbling.

### The rule of thumb

> Before trusting a cost, ask: *what algorithm does this number assume, and does the emit
> build exactly that algorithm — with the same operand tile shapes (A3), the same band
> liveness (A4), the same DDR traffic (A2/A7), and the same overlap (A5)?* When the answer is
> "no," the fix is either to make the emit implement the assumed algorithm, or to make the
> model price the algorithm the emit actually builds.
