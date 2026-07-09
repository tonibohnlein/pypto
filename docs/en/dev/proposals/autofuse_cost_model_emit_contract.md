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

```
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
|---|---|---|---|
| **role back-prop** | reverse-topo (sinks→inputs) | each tensor's tile shape / role | *what* each op computes per tile — the algorithm |
| **pebbling order** | forward-topo | execution sequence + band liveness | *when* + feasibility (peak UB) + emit replay + MemoryReuse |

---

## 1. The roofline (what the model computes)

For one fused vector group at a fixed tile, over `C` AIV cores (910B: 48):

```
latency = max(compute_mk, ddr)      [if the tile double-buffers; else compute_mk + ddr]
```

### Compute (`VecOpCompute :63`, `WaveComputeCycles :153`)

- Per op, grounded cycles: pointwise = `slope·repeat + (stream_start ? head+tail : 0)`,
  `repeat = ceil(elems / epr)`, `epr = vec_reg_bytes / dtype_bytes` (fp32: 256/4 = 64).
  Reduction = its **tree**: row-reduce ≈ `45·(W/epr − 1) + 51` (linear in W, rows-independent);
  col-reduce ≈ `16·(H−1) + 30·log2(H)` (log-depth).
- **Tiling-invariant**: summed over the FULL element count (each element touched once).
  The tile does NOT change the raw cycle count.
- head+tail is paid **once per back-to-back pointwise run** (only the stream-start op).
- Spread over the grid: `compute_mk = WaveComputeCycles(total, U, C) = total·ceil(U/C)/U`,
  `U = num_tiles = parts_m·parts_n`. Filling toward `C` lowers it; past `C` costs extra waves.

### DDR (`:1744`, `dma_pen :1779`)

- Counts **only boundary tensors**: external reads (`io_in`) + boundary writes (`io_out`),
  over the FULL tensor bytes (tiling-invariant: read each input once, write each output once).
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
- `latency = max(compute_mk, ddr)`.

The tile size changes **occupancy** (`U` → compute divisor) and **DMA efficiency / pipe
count** (`dma_pen`, `active` → DDR divisor) — never the raw cycles or byte counts.

---

## 2. Feasibility is a PEBBLING problem (not a static sum)

The model does not check "does the whole group fit UB." It checks **peak simultaneously-live
bands over the pebbling order** (`vector_peak_ub :1241`):

- an ephemeral tensor `t` occupies a UB band across `[producer, last consumer]` in `dfs_order_`;
- at each step, peak = live bands + transient input/output tiles;
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

| # | The model assumes… | So the emit MUST… | Status |
|---|---|---|---|
| A1 | compute spreads over `U = parts_m·parts_n` invocations (wave makespan) | emit ONE per-tile kernel body and launch it over the grid (`SpmdScopeStmt` + `get_block_idx` → offset), one invocation per output tile | ⚠️ mechanism honored, but the emit launches **`ceil(IM/h)·ceil(IN/w)`** invocations, NOT the priced `parts_m·parts_n` — equal for uniform grids, **divergent (fictional occupancy) for non-uniform/ragged grids** (logged not reconciled, `auto_fuse_pass.cpp:1758`) |
| A2 | ephemeral intermediates cost **0 DDR** (fusion win) | keep intermediates **on-chip** (UB scratch), never round-trip them through DDR | ✅ honored |
| A3 | each op runs at its **back-propagated role** shape | slice `FROM_NT*` operands to `[h,w]`; read `FIXED_1` operands in full (broadcast / reduced-axis) — `emit_strip :1343` | ⚠️ reduced-axis `FIXED_1` honored; **broadcast `FIXED_1` NOT** — model prices a pointwise-broadcast group (bias-add `[IM,IN]+[1,IN]`) as fusible (`cost :638`) but the emit **declines** any 2D broadcast operand (`auto_fuse_pass.cpp:1174`, Tier-A) → legacy tiler |
| A4 | UB feasibility = peak live bands over the **pebbling order** | replay ops in `dfs_order_`; let MemoryReuse alloc/free UB per the emitted liveness | ✅ honored (order matched) |
| A5 | roofline `max(compute, DDR)` (load k+1 overlaps compute k) | emit the per-core loop **software-pipelined / double-buffered** (`ForKind::Pipeline` + `kPipelineStagesAttr`) | ⚠️ **only for non-streamed loops** — pointwise strip `:1740` is Pipeline; **streamed P1/P2 loops are `Sequential` `:1489/:1533` → serial → real latency = compute+DDR** |
| A6 | reduced-axis split `S` = S cores reduce slices, **atomic-add** merge | build the cross-core split (seed + `SetAtomicAdd`) — realized ONLY for a bare **sum col-reduction** sink `:1630`; else fall back to serial | ⚠️ partial — model must NOT price `S` where the emit declines it (see C2) |
| A7 | a streamed reduction reads each input band **`stream_passes`** times (§5.1): **1** if every output folds into a reduction, **2** if any output spans the reduced axis | stream with running stats; **1 pass** when output-folded, **2 passes** (flash-stats + apply) when an output spans R; fold multi-stat reductions **online** (never naive multipass) | ⚠️ model prices a flat **read-once** (too optimistic for spanning outputs — softmax/layernorm are 2 reads); emit: P1 1-pass ✓, softmax/layernorm online (P4) NOT built; neither loop pipelined (A5) |

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
- **reads the data once** (matches A7).

### The P-ladder

| tier | shape | flash content | status |
|---|---|---|---|
| **P1** | bare reduction (sum / max) | online but trivial — sum associative, running max; no rescale | built, device-confirmed (col reductions) |
| **P2** | reduction → pointwise | accumulate, then re-stream to apply (2 passes) | built |
| **P3** | multipass softmax/layernorm (>1 reduction) | one stream per statistic pass (2–3 reads) | designed, not built |
| **P4** | online-stats | the **true flash**: single streaming pass, running (max,sum) + rescale | designed, not built |

### Why P4, not P3

A7 prices **read-once**. P3 (multipass) reads the input 2–3× from DDR — and streamed
reductions are DDR-bound, so a multipass emit runs materially slower than priced. To keep
the model honest for streamed softmax/layernorm, the emit must be **P4 (online/flash)**, not
P3. And by A5 it must also be **pipelined**. Both requirements are on the same streamed loop.

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

```
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

Today the model prices a flat "read once" for every streamed reduction (`:1759`), which is
correct only for the output-folded case. Add:

- derive `foldable` (all R-reductions in the log-sum-exp/moments family) and
  `spans = (any live-out extent along R > 1)`;
- `stream_passes = spans ? 2 : 1` (general: `1 + depth`);
- when streaming, **scale the streamed input read `io_in` by `stream_passes`** (the output
  write stays ×1) and scale the applied-cone **compute** by the apply pass;
- if `streams && !foldable` → **infeasible** (`cost = inf`) so the partitioner never picks a
  group it cannot emit correctly.

This sits exactly between today's flat `×1` (too optimistic — undercounts softmax/layernorm by
one full input read) and the retired `#reductions+1` (too pessimistic — charged 3 for softmax;
the truth with flash is 2).

## 5.3 Emit design (per case)

- **output-folded, 1 pass** (P1): as today — SPMD over the free axis, inner chunk-accumulate the
  single stat, assemble the `[·,1]`/`[1,·]` output.
- **spanning, single stat** (P2): as today — pass 0 accumulate the stat, pass 1 apply over R
  chunks; **both loops must become `ForKind::Pipeline`** (A5).
- **spanning, multi-stat foldable** (softmax / layernorm — **P4, to build**): pass 0 = ONE
  online streamed pass maintaining the running stats with the exact rescale (softmax `(m,l)`
  with `exp(m−M)`; layernorm Welford `(count,mean,M2)`); pass 1 = re-stream R, apply the
  finalized stats per element, assemble. Both loops `ForKind::Pipeline`. This replaces the
  naive 3-pass P3.
- **non-foldable over a streamed axis**: decline (matches the `cost = inf` gate).

---

## 6. Fidelity status (audited 2026-07-08 by 4 independent reviewers against this doc)

Confirmed accurate against the code: the roofline math (§1), pebbling feasibility (§2), all
role back-prop rules (§3), ephemeral = 0 DDR (A2, both sides), the emit replaying in
`dfs_order_` (A4 — `flush()` stable-sorts by `execution_order()` before emit), multi-role
tensor tracking, the C2 split gate, and the S2 emit predicate. The gaps below are ranked;
several refined the ⚠️ rows above.

### Root cost-model gap — fix FIRST (everything streaming depends on it)

**R0 — streaming is not detected for a bare reduction sink.** The C2 `reduced_extent_` coupling
lives ONLY in `reduction_materializes()` (the split gate, `ascend910b_cost.cpp:1826`). The
feasibility gate (`vector_stream`'s materialize pre-check, `:1319`) and the compute
streaming-detection (`:1761`) both call `vector_peak_ub(cfg)` with the RAW grid cfg — whose
reduced axis is collapsed to the thin output extent (`out_H_/out_W_ = 1`). So a bare `col_sum`/
`row_sum` sink is under-counted by its full reduced extent and priced as a MATERIALIZED tile
that fits UB, even though the emit streams it (P1). Contradicts the `vector_peak_ub` header
contract (`ascend910b_cost.h:124`). **Fix:** couple the reduced axis at the feasibility +
compute sites too — then A5/A7/P4 have a correct streaming signal. (Softmax works today only
because its wide pointwise SINK keeps `cfg.w` large, so `vector_peak_ub` sees the overflow;
bare reductions have a thin sink and slip through.)

### 🔴 Correctness

**G1 — large softmax/layernorm silently overflow UB (no decline). [CRASH FIXED — P4 capability still open.]**
The stream gate requires `p1_nreds == 1` (`auto_fuse_pass.cpp:1231`), so a 2-reduction group
cannot stream; it used to fall through to a full-reduced-axis materialized tile that overflowed
(hard `AllocateMemoryAddr` failure). **Fixed** (mlsys26 `603ec35`, pypto `69d7f508`): a new
`Problem::allow_model_ahead_multi_reduction_stream` flag — the AutoFuse adapter sets it false
(buildable), so a streamed >1-reduction group is **infeasible** and the partitioner **cuts** it
into single-reduction (streamable) + pointwise pieces. An **unfused softmax IS buildable**, so
large softmax/layernorm now **compile** (verified: `softmax[128,16384]` builds, was a crash).
Emit defense-in-depth: `GenericDeclineB` if such a group still reaches the materialized path and
its thinnest tile overflows. **Still open:** the FUSED single-kernel capability (P4 — online
multi-reduction) that would let these stream as one kernel instead of being cut.

### 🔴 Roofline

**G2 — A5 — streamed loops not pipelined.** P1/P2 emit `ForKind::Sequential`
(`auto_fuse_pass.cpp:1489,1533`); `LowerPipelineLoops` skips them (`lower_pipeline_loops_pass.cpp:208`
requires `ForKind::Pipeline`) → serial → real = `compute+DDR`, model prices `max` (`:1798`).
**Fix:** emit `ForKind::Pipeline` + stages=2 (mirror the pointwise strip `:1740`); reconcile the
in-place accumulator IterArg with MemoryReuse stage separation + the pipeline verifiers (only
the chunk buffers double-buffer; the accumulator stays single-buffered persistent).

### ⚠️ Cost fidelity

**G3 — A7 — `io_in` not scaled by `stream_passes`.** The streaming surcharge (`:1767`) adds only
a thin compute term; io stays ×1. Spanning-output reductions (P2 / softmax / layernorm) read the
input twice. Fix per §5.2 — but it needs R0 first (the model must detect streaming to price it).

**G4 — A3 — broadcast priced-fusible but emit-declined.** Model prices `[IM,IN]+[1,IN]` bias-add
as fusible (`cost :638`); emit declines any 2D broadcast operand (`auto_fuse_pass.cpp:1174`,
Tier-A) → legacy tiler. Ubiquitous in FFN/attention. Fix: implement broadcast-operand emit, or
make the model decline broadcast groups (align the two).

**G5 — A1 — grid-count divergence.** Emit launches `ceil(IM/h)·ceil(IN/w)`, model prices
`parts_m·parts_n` (`:1465`); fictional occupancy for non-uniform/ragged grids (logged not
reconciled, `:1758`).

**G6 — A6 residual (C2 DONE).** Split still priced for materialized max/row reductions the emit
declines (bare-`col_sum`-only, `:1630`). Bounded, unmeasured; left as-is pending a probe.

### Minor / doc-completeness

- **Granule-padding feasibility fiction — FIXED (2026-07-09, BUG-G1THRESH).** The emit allocates
  `AlignUp(sh,g)×AlignUp(sw,g)` tiles (`g = vec_dma_align_bytes / group_min_dtype_bytes`); the cost
  model's `vector_peak_ub::tile_bytes` used unpadded `min(cfg,dim)`, so a thin free axis (M-tile
  3→8, ~2.7×) was under-counted → an over-UB group looked materializable → the emit overflowed
  `AllocateMemoryAddr` (softmax/layernorm N=4096/8192). Both `vector_peak_ub` AND the emit's own
  materialize-vs-stream trigger now count the padded footprint (via `Problem::vec_dma_align_bytes`,
  sourced from `BackendHandler::GetVectorDmaAlignmentBytes`), so model and emit agree: width always
  padded, height padded for reductions (col-major). This also makes the streamed-chunk sizing
  granule-faithful — closing most of R3.
- `dfs_order_` is a greedy topo-tie-break heuristic, not a provably-minimal pebbling.
- §1's pointwise compute omits the `+16` count-mode floor charged when `width % epr != 0` (`:90`).

### The rule of thumb

> Before trusting a cost, ask: *what algorithm does this number assume, and does the emit
> build exactly that algorithm — with the same operand tile shapes (A3), the same band
> liveness (A4), the same DDR traffic (A2/A7), and the same overlap (A5)?* When the answer is
> "no," the fix is either to make the emit implement the assumed algorithm, or to make the
> model price the algorithm the emit actually builds.
