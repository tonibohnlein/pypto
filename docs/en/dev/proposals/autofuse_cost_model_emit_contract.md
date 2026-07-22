# AutoFuse vector kernels — cost-model assumptions ↔ emit obligations

**Purpose.** In AutoFuse, a solver **Solution is not a number — it is a specification
of a kernel algorithm.** Every term in the cost model implicitly assumes the emitted
kernel implements a particular algorithm. If the emit implements a *different* (usually
cheaper-looking) algorithm, the cost is fictional and the solver's ranking is wrong.

This document enumerates, for the **vector** path (pointwise + reduction on the Ascend
910B AIV cores), **what the model assumes** and **what the emit must therefore produce**.
It is the fidelity contract. Scope: vector kernels only (no cube / no mixed cube+vector).

Code references are `3rdparty/pto-fusebox/src/core/ascend910b_cost.cpp` (cost) and
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
  add/sub/max/min, mul, div, exp, log, rsqrt, abs, sqrt, neg, scalar add/sub/mul/max/min, exact
  part add/mul/max/min, their row/column broadcast forms, and supported row/column reductions. Their
  coefficients come from the same tables as generated P4 work. Composite or alias-sensitive source
  ops retain the explicit `Generic` fallback rather than being assigned a guessed primitive.
- Candidate costing replays each grounded op at the exact valid frame of one emitted materialized
  strip or streamed-reduction chunk, then multiplies by that phase's loop iterations and logical
  tasks. `repeat = ceil(elems/epr)` for flat ops; row/column broadcasts use their strided
  `rows·ceil(cols/epr)` geometry. A row-expand binary in a reduction-layout group additionally pays
  the emitted `vbrcb` and `PIPE_V` barrier needed for its col-major statistic; the pure-pointwise
  row-major form does not. Count-mode floors are applied per invocation.
- Reduction work is replayed at the same frame. FP32/FP16 row and column sum/extrema use PTO-ISA's
  A2/A3 fit tables; exact table shapes reproduce PTO and other emitted shapes interpolate adjacent
  anchors. Unsupported dtypes retain the explicit structural fallback. P1/P2 also price the thin
  add/max merge emitted after every non-initial statistics chunk; that merge is absent from the
  source DAG.
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

- an ephemeral tensor `t` occupies a UB band across `[producer, last consumer]` in the selected order;
- a boundary input occupies one UB band from first through last use in each replay phase. Vector
  identity is `(tensor, phase)`: operand position does not split the UB representation, while the
  barrier between stats and apply deliberately creates two lifetimes and two GM→UB reads;
- at each step, peak = live bands + transient input/output tiles;
- a reduction materialization includes both its source tile and tensor-lowering work/layout tile;
- feasible ⟺ `max over steps ≤ UB`.
- `dfs_order_` is the legacy name for the selected order. Post-order DFS remains the default and
  finishes one branch before a sibling; dependency-constrained Gorder is available for comparison.

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
- A vector tensor has no cube-like LHS/RHS representation split. Within one phase, all consumers
  reuse the same shape/dtype UB tile; only shape propagation and the replay phase affect identity.

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
| A1 | compute spreads over `U = parts_m·parts_n` invocations (wave makespan) | emit ONE per-logical-region kernel body and launch it over the exact solver grid (`SpmdScopeStmt` + `get_block_idx` → offset) | ✅ device-confirmed. `VectorStreamPlan` owns element-balanced M/N partitions and `work_units`; cost, diagnostics, and emit consume that count. DMA padding is recorded separately and cannot change the launch grid. |
| A2 | ephemeral intermediates cost **0 DDR** (fusion win), except an explicitly returned value is also a boundary output | keep ordinary intermediates **on-chip**; for a returned-and-consumed SSA value, retain its UB lifetime and also assemble its required DDR live-out | ✅ returned live-outs are explicit in `Problem`; P4 rejects escaped stats. |
| A3 | each op runs at its **back-propagated role** shape | slice `FROM_NT*` operands to `[h,w]`; read `FIXED_1` operands in full (broadcast / reduced-axis) — `emit_strip` | ✅ honored (G4 done, 2026-07-09). `emit_strip` slices any 2D operand per-axis: a full axis follows the tile `[sh/sw]` at `[smi/sni]`, a size-1 (broadcast/`FIXED_1`) axis stays `[1]` at offset 0; the op replay re-infers the broadcast. Covers `[1,N]` bias-add, `[M,1]` scale / reduced-axis stat, `[1,1]`. Other 2D shapes still decline. |
| A4 | UB feasibility = peak live bands over the **pebbling order**, actual tensor dtypes, generated scratch, and pipeline copies | replay the selected order; load a boundary tile once per phase/strip, keep it through its last use, and let MemoryReuse realize that liveness | ✅ produced and boundary-input lifetimes are byte-weighted; the winning emit validates the exact tensor/use-op descriptor before caching the slice. Planned prefetch/generated scratch remain explicit. |
| A5 | roofline `max(compute, DDR)` only within a loop that overlaps load k+1 with compute k | emit that exact loop **software-pipelined / double-buffered** (`ForKind::Pipeline` + `kPipelineStagesAttr`) and keep barriers serial | ✅ `VectorStreamPlan` owns materialized/pointwise row+width strips and P1/P2/P4 init/rolled/tail/finalize phases. Cost is `Σphase roofline`; peeled phases use `compute+DDR`, and sub-register or short rolled loops stay sequential. |
| A6 | reduced-axis split `S` = S cores reduce slices, **atomic-add** merge | build the cross-core split (seed + `SetAtomicAdd`) — realized ONLY for a terminal **sum col-reduction** sink; else stay serial | ✅ exact admission, seed cost, `work_units*S` fill waves, and 32-byte seed-row buildability gate. |
| A7 | streamed input reads are phase-specific: an operand may be stats-only, apply-only, or both | consume the shared `VectorInputLifetimeTopology`; read each tensor once per emitted phase/chunk and reuse it for all source-op consumers | ✅ `x` in P2/P4 has separate stats/apply lifetimes; an apply-only scale/bias is absent from stats. Repeated operands such as `mul(x,x)` remain one transfer/use-op. |

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
| **P4** | online-stats | one stats pass with softmax `(m,l)` rescale or layernorm Welford, then apply | exact row-softmax default-on; Welford opt-in with `PYPTO_AUTOFUSE_P4=1`; `0` disables both |

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

**G1 — large softmax/layernorm silently overflow UB. [FIXED; exact P4 capability built.]** A
multi-reduction group that lacks an implemented online algorithm is infeasible and is cut into
streamable pieces instead of reaching `AllocateMemoryAddr`. P4 shares one exact canonical
softmax/layernorm descriptor between admission and emission; temperature/scaled softmax, weighted
moments, chained norms, and multi-sink escapes still cut rather than being reinterpreted.

### Roofline

**G2 — A5 — phase rooflines and solver-owned loops — FIXED (2026-07-12).** Stats and apply rolled
loops emit `ForKind::Pipeline` with stage 2 only when at least two chunks can overlap; init, tail,
finalize, and loop-carried statistics stay serial/persistent. `VectorStreamPlan` owns the chunks,
trips, stages, and materialized/pointwise strips. Costing sums barrier-separated phase rooflines,
and the shared P2/softmax/Welford builders consume those fields directly.

### ⚠️ Cost fidelity

**G3 — A7 — streamed input multiplicity/lifetime — HOST-CLOSED.** Candidate-invariant phase cones
produce `VectorInputLifetimeTopology`: every boundary tensor records first/last step, distinct
source-op uses, and phase. UB feasibility retains it to last use, traffic charges once per
tensor/phase/chunk, and emission resolves the descriptor back to one SSA value and caches one slice.
Thus stats+apply `x` reads twice across the barrier, while apply-only bias reads once.

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
the inner stream inside each task. Cost and emit use the same count for waves, active GM pipes,
overhead, traffic, and `pl.spmd`. On 910B2 every forced grid matched `pl.spmd(N)` and the repaired
natural `[128,8192]` argmin became the device-best 16-task plan. Ragged blocks price the emitter's
clamped maximum-shape body rather than the logical union.

**G6 — A6 — materialized reduction split admission and seed — HOST-CLOSED.** The 910B backend lowers
only atomic add, and slicing a reduction cone preserves semantics only for one terminal `col_sum`
whose upstream cone is pointwise. The solver now derives that exact capability from the source
primitive, axis, sinks, and reductions; unsupported cases enumerate only `S=1`. An `S>1` candidate
proves materialized work, exact granule-aligned partitions, and replays its cone at emitted
`[M/S,free]` geometry. `VectorStreamPlan` records the atomic body and a separate ordered
`VectorReductionSeedPlan`; cost and emit include seed fill/store/tasks/wave. Admission also requires
a 32-byte-buildable seed row, so `[2048,4]` stays serial while aligned `[2048,8]` retains the split.
Missing or mismatched descriptors fail the contract rather than falling back.

**G7 — P4 algorithm-specific compute — SILICON-CLOSED (2026-07-13).** Phase masks
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

At the G7/G8/G9 fingerprint, exact softmax was correct and 25.5% faster than its two-kernel cut;
Welford was 24.7% faster at zero mean and remained correct at mean `+2000`, where the dual-sum cut
failed. The emitted stats/apply loops and serial init/tail/finalize phases matched the descriptor.
Exact softmax is therefore default-on. Welford remains opt-in only for the independent extreme-shift
accuracy gate, not because of a remaining cost/emit representation gap.

**G8 — source-DAG strip/chunk compute — SILICON-CLOSED (2026-07-13).** The old
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
problems retain their old scalar anchors intentionally. The silicon run confirmed the emitted P2/P4
phase descriptors, exact `3:1`/`2:1` GM traffic, and fused-versus-cut decisions. Unsupported
primitives remain explicit `Generic` fallbacks, so this closes the
canonical softmax/layernorm apply cones without claiming that every PyPTO vector op is grounded.
The explicit reduction descriptor also selects this path for a bare P1 kernel, so its per-chunk
reduction tree and generated add/max merge are not lost merely because the graph has no pointwise op.

**G9 — row-aware reduction work — SILICON-CLOSED (2026-07-13).** The original
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
344722/163497/159365/173609 model-cycles). The 910B2 sweep put natural U48 in the device-best cluster
with 4.2% regret and preserved the gross forced-plan order (Spearman 0.70). The short `[128,8192]`
case over-splits mildly (7.1% versus the clean 16-task grid), a bounded ranking refinement rather
than a model/emit mismatch.

**G10 — column-reduction grounding — SILICON-CLOSED (2026-07-13).** Exact FP32/FP16 `TCOLSUM` and `TCOLMAX` now use
the PTO A2/A3 fit tables rather than the legacy structural tree. Exact anchors include FP32
`[64,64]` sum/max = `1263/1137` cycles and FP16 `[64,128]` = `1263/1137`; intermediate shapes
interpolate adjacent table anchors and remain monotone in valid rows. Unsupported dtypes retain the
named structural fallback. The device sweep reproduced every anchor byte-exact and confirmed the
column decision ordering.

**G11 — source primitive coverage — SILICON-CLOSED (2026-07-13).** The adapter and plan now name every audited
one-instruction vector lowering used by the supported emit surface: abs, sqrt, neg, scalar max/min,
and exact part add/mul/max/min join the existing arithmetic, transcendental, broadcast, and reduction
families. The device run matched the emitted family/geometry descriptors and numeric output across
square, ragged, and multi-strip shapes. High-precision rsqrt remains `Generic` in research input;
production AutoFuse now separately capability-gates every op so an unknown or unsupported lowering
cannot inherit pointwise strip semantics merely because it lacks a descriptor.

**Post-review live-out/capability/UB closure — HOST-CLOSED.** `Problem::required_outputs` makes a
returned-and-consumed value both a boundary output and a live UB interval; the emitter assembles it,
the shared P4 consumer analysis treats `ReturnStmt` as an escaping use, and partition/solution gap
checks preserve that materialized value across group boundaries. `VectorOpCapability`
separates cost family from implemented scheduling algorithm: elementwise and sum/max reductions are
admitted, while prod/arg/min/full and position/shape-changing operations decline. Strip feasibility
replays byte-weighted tensor lifetimes and a dtype-exact prefetch copy; streamed reductions add their
explicit generated scratch. The FP32→INT8 forced-plan regression now lowers without UB overflow.

### Minor / doc-completeness

- **Concurrent cost cache and creation complexity — FIXED (2026-07-13).** Cache slots publish
  through `Empty→Writing→Ready`, so an acquiring reader cannot observe a partially copied
  `CostResult`; a concurrent stress test covers the protocol. The default base table uses 131072
  slots instead of one million, and the equally sized retention tier is allocated only if used.
  Duplicate validation is removed; prologue reachability is one reverse-topological `O(N+E)` DP.
- **Candidate-hot replay — FIXED (2026-07-13).** Creation stores candidate-invariant UB metadata,
  input lifetimes, and phase op lists; candidates derive plans/costs and `CostResult` stays plan-free.
- **Offline task-cost replay — FIXED (2026-07-13).** Problem JSON now serializes and reloads
  `per_task_overhead_cycles`, preserving C3's 64-cycle term in dumped generic-emitter problems.
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
- `dfs_order_` is the legacy storage name for the selected pebbling order. DFS remains default; dependency-constrained Gorder is implemented for controlled comparison, not claimed optimal.

### The rule of thumb

> Before trusting a cost, ask: *what algorithm does this number assume, and does the emit
> build exactly that algorithm — with the same operand tile shapes (A3), the same band
> liveness (A4), the same DDR traffic (A2/A7), and the same overlap (A5)?* When the answer is
> "no," the fix is either to make the emit implement the assumed algorithm, or to make the
> model price the algorithm the emit actually builds.
