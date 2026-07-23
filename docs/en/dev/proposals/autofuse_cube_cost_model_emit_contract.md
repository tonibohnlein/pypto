# AutoFuse cube cost-model ↔ emit contract

## 1. Scope

This contract covers homogeneous AIC groups containing only `tensor.matmul` operations. Mixed
cube/vector kernels are a separate charter. A solver solution describes the complete GM→L1
algorithm; a shared backend plan describes the nested L1→L0 algorithm. Cost and emission must replay
both levels with the same shapes, lifetimes, traffic, overlap, and output ownership.

The candidate-invariant request DAG is built once in `Ascend910BCost::create()`. Candidate evaluation
stores only `CostResult` in the local-search cache. AutoFuse reconstructs the full
`CubeSchedulePlan` only for winning or explicitly forced configurations.

Cube evaluation has two deliberate optimization modes:

- **analytic (default):** use the grounded fixed-base-tile cube/MTE1 surrogate for outer-plan
  ranking, then let `AutoTileMatmulL0` choose detailed L0 geometry after the winning GM/L1 plan is
  known;
- **exact/co-optimized:** set `PYPTO_AUTOFUSE_EXACT_L0_COST=1` to minimize the hierarchical cost over
  the shared L0 design space for every outer candidate. This is the prospective `-O3` mode and
  remains opt-in pending compile-time and silicon-regret comparison.

## 2. Two-level ownership

`CubeSchedulePlan` owns cross-core and GM/L1 decisions:

- spatial regions and split-K work units;
- recursive producer requests and their L1 lifetimes;
- boundary inputs resident from their first request through their last compatible consumer;
- optional boundary panels retained in L1 across one request's output-tile loop;
- each matmul's GM→L1 K window, chunk, init, rolled loop, and tail;
- the L0C-resident output-tile grid;
- internal Acc→Mat and root Acc→GM drains;
- split-K first-partial and atomic-rest ownership.

The shared `L0MatmulPlan`, selected by the same PTO Fusebox chooser used by
`AutoTileMatmulL0`, owns:

- L0 M/N/K and stationarity;
- L0A/L0B/L0C buffer depths;
- its K-loop chunk, full-block count, tail, and pipeline stage;
- serial fill, rolled overlap, serial tail, and drain costs;
- the child output target (`Acc` for a surrounding GM/L1 K stream).

AutoFuse emits tensor-level calls and never creates Left/Right/Acc tiles. Both modes attach only the
semantic child output target (`Acc`, `L1`, or `GM`). Exact mode additionally attaches a versioned
child-plan record. `AutoTileMatmulL0` always runs the active backend chooser and alone emits L1→L0
extracts, cube operations, and low-level drains; in exact mode it also fails on descriptor drift.

## 3. Obligations

| Ref | The model assumes | The emitter must build |
| --- | ----------------- | ---------------------- |
| C1 | `parts_m × parts_n × split_k` independent work units | Launch exactly that SPMD grid; use disjoint ownership, except for the explicit split=1 idempotent clamped-overlap policy |
| C2 | Only boundary tensors touch GM | Keep every supported intermediate in L1; never use GM as an implicit fused-chain buffer |
| C3 | Consumers back-propagate exact row/column requests | Slice LHS as `[requested rows,K]` and RHS as `[K,requested cols]`, including produced operands on either side |
| C4 | Peak L1 is the request-order pebble peak for produced values and boundary inputs | Replay producer-before-consumer, preload repeated compatible inputs at first use, retain every value through its priced last use, and allocate no unpriced scratch |
| C5 | Every request owns its GM→L1 K loop | Use that request's contraction, chunk, rolled stage, and serial tail |
| C6 | One complete output tile remains in L0C across all K windows | Nest output tiles outside the K-window loop; never spill a partial to Mat and reload it into Acc |
| C7 | Exact mode prices a concrete shared-backend L0 plan; analytic mode prices the grounded surrogate | Always attach the output-residency intent; attach and validate detailed geometry only in exact mode; let `AutoTileMatmulL0` realize both |
| C8 | Overlap is local to one concrete full-window loop | Put K=0 and every rolled full window in the same eligible stage ring; add only its fill/drain, the ragged tail, and final output drain serially |
| C9 | GM traffic follows the emitted request and output-tile loops | Give a repeated compatible boundary value one serial group preload; otherwise charge repeated per-tile feeds, except for an explicitly selected request-local retained panel |
| C10 | Split-K has one initializing share and `S-1` merge shares | Emit an ordered normal-store AIC phase for share zero, then an atomic-add AIC phase for shares `1..S-1`, or select `S=1` |
| C11 | Cube accumulation dtype differs from storage dtype | Accumulate float inputs in FP32, then narrow once at the planned BF16/FP16 or GM drain |

## 4. Grounded loop structure

The PTO A2/A3 performance GEMM uses this hierarchy. An exact/co-optimized plan may first retain a
complete reusable boundary panel in L1:

```text
optional: load reusable A[rows,K] and/or B[K,cols] GM -> L1 once
for output tile (m, n):
    for K window loaded GM -> L1:       # L1 ping/pong
        # a retained operand is locally sliced instead of loaded again
        for base-K block L1 -> L0A/B:   # L0A/L0B ping/pong
            TMATMUL / TMATMUL_ACC       # one L0C accumulator
    TSTORE completed accumulator        # exactly one final drain
```

The manual kernel is
`pto-isa/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`; the automode kernel has
the same `i, j, K` nesting. `chain_fused_kernel.cpp` grounds the supported on-chip chain handoff:
the producer accumulates in FP32 L0C, drains once to a BF16/FP16 L1 Mat tile, and the consumer reads
that Mat tile. PTO A2/A3 supports Acc→Mat and Acc→GM, but not Mat→Acc. It also does not provide a
same-type FP32 Acc→Mat handoff. Consequently:

- all GM-window child plans target `Acc` and have no child drain;
- the completed tile has one explicit final drain;
- BF16/FP16 intermediates can remain in L1 through FP32→low-precision FIXPIPE conversion;
- an explicitly FP32 intermediate declines fused on-chip replay and is partitioned into standalone
  matmuls rather than changing precision or emitting an impossible reload.

The outer GM→L1 pipeline carries `pipeline_gm_to_l1_only`. `LowerPipelineLoops` therefore does not
multiply its stage count into an existing child L0A/L0B ping/pong allocation. When the child fits
one L0 tile and has no inner pipeline membership, the outer loop itself owns the necessary two-bank
L0 operand lifetime. This represents hierarchical double buffering, not four copies of every
lower-level operand, while still preventing adjacent outer stages from aliasing one live L0 address.

Child init and ragged-K tail extracts carry a transient `pipeline_serial_phase` marker. They remain
after the rolled L1→L0/Matrix phase and reuse one of its operand banks; an enclosing GM→L1 pipeline
must not hoist them into its prefetch tier. Pipeline lowering is simplified before memory
materialization so constant-dead peeled branches cannot create a second L0C accumulator. These are
compiler schedule invariants, not additional model feasibility terms.

### 4.1 Serial request composition is the current contract

A homogeneous cube DAG is currently replayed as a serial sequence of complete matmul requests. For
`T = A @ B; Y = T @ D`, one work unit executes:

```text
preload group-resident boundary values
run all output tiles and K windows of A @ B
drain the completed T region to L1/Mat
run all output tiles and K windows of T @ D
drain the completed Y region to GM
```

Each matmul may independently use its emitted stage-2 K-window ring. The dependency between the two
requests is nevertheless serial: the second request does not start consuming a panel while the
first request is still producing later panels. The cost must therefore be:

```text
T_group = T_resident_preloads
        + sum(T_request_init
              + T_request_rolled_roofline
              + T_request_tail
              + T_request_drain)
        + T_first_partial_phase
        + T_split_sync
        + T_atomic_rest_phase

T_request_rolled_roofline = max(MTE2, MTE1/Matrix)
                            only for that request's emitted stage-2 loop
```

It must not use `max(sum(request compute), sum(request traffic))`: that expression overlaps work
across a producer/consumer boundary that the emitter does not pipeline. The request-order pebbling
model is the matching memory contract. Produced regions and compatible repeated boundary inputs
remain live from first use through last use; only the currently executing request contributes its
transient GM→L1 window. No cross-request ping/pong buffers are assumed.

Vector fusion does not have the same mismatch. A vector phase emits one strip/chunk loop whose body
replays all pointwise operations in that phase. Summing those operations before applying the phase
roofline is valid because they are work in one concrete loop iteration, and iteration `i+1` is
actually prefetched while iteration `i` computes. Stats, apply, init, tail, and finalize are distinct
barrier-separated phases; their rooflines are added. `VectorStreamPlan` also charges the second copy
of pipelined source-DAG bands and the next iteration's boundary inputs in UB. The vector model never
uses one roofline to hide work across the stats/apply barrier.

### 4.2 Deferred extension: panelized back-to-back matmul

A later cube revision may add a different algorithm for compatible chains:

```text
Y_acc = 0
for panel j of the shared N axis:
    T_j = A @ B[:, j]
    Y_acc += T_j @ D[j, :]
store Y_acc
```

This can avoid materializing the full intermediate and permit GM/L1 work for a neighboring request
to overlap. It is not an alternative cost equation for the current plan; it is a new schedule and
must have a new solver-owned descriptor and emitter. A future `PanelizedB2BPlan` must record:

- the shared panel axis, panel extent, warmup, steady state, and tail;
- the intermediate Mat/L1 ring depth;
- the persistent consumer accumulator and temporary producer accumulator, which are simultaneously
  live in L0C;
- co-live producer and consumer GM→L1 windows and any retained panels;
- the L0A/L0B banks that can be reused between sequential Matrix instructions;
- exact per-memory-space peaks derived from stage lifetimes.

The steady-state Matrix term is `Matrix_MM1 + Matrix_MM2`, never their maximum, because both use the
same Matrix pipe. A combined phase roofline is legal only after emission replays this combined
schedule. Until then, serial request composition is the only admitted cube contract and the existing
pebbling model remains unchanged.

## 5. Implemented schedule and cost

For a requested `O[rows,cols] = A @ B`, reverse request propagation produces
`A[rows,K]` and `B[K,cols]`. The postorder supports deep chains and trees, produced RHS operands,
both inputs produced, fan-out role changes, and multiple roots. Distinct requested roles become
distinct producer instances and pay recomputation; identical instances are shared in the request
DAG.

Request ordering goes through the common `PebblingOrderStrategy` interface. Deterministic DFS
postorder remains the compile-time default for both the source-op DAG and cube's role-expanded
request DAG. A dependency-constrained implementation of the
[Gorder locality objective](https://dl.acm.org/doi/10.1145/2882903.2915220) is also available: it
greedily maximizes direct-neighbor plus common-predecessor/request-value score in a five-node sliding
window, but selects only ready nodes so producers still precede consumers. Boundary locality uses
the exact role-expanded value identity below. Its incremental ready-set update runs once in
`create()`, not for every tile candidate. Build with `-DPTO_FUSEBOX_GORDER=ON` for controlled
comparison; DFS stays default until model and device evidence justify changing it.

Gorder is not generally an `O(V+E)` algorithm. With a fixed-size window, direct-edge updates are
linear, but common-predecessor updates can cost `sum(out_degree(u)^2)`; the role-expanded extension
has the analogous `sum(users(value)^2)` term for shared boundary values, plus ready-set logarithmic
factors. It is close to linear on bounded-fan-out DAGs and avoids an `O(V^2)` materialized score
table, but high-fan-out graphs retain a superlinear worst case. This ordering is computed once per
subgraph, never once per tiling candidate.

Boundary-input identity is `(source tensor, requested region, dtype, memory pool, operand role)`.
The role is load-bearing: two matmuls requesting the same LHS slice may share it, while `A @ A`
produces distinct LHS and RHS values because they require different L1 representations. Repeated
compatible inputs receive an always-retained lifetime from first to last request. Produced values
use the same first-use/last-use pebbling principle. Both contribute to one request-order L1 peak.

For every request, the plan records:

- concrete input/output regions and symbolic axis bindings;
- producer-instance dependencies and storage/accumulator dtypes;
- group-resident boundary descriptors with role, first/last use, use count, region, and exact bytes;
- the GM→L1 K-loop descriptor;
- an optional retained-LHS/retained-RHS descriptor with exact L1 bytes;
- up to four L0 output variants: full/full, tail/full, full/tail, and tail/tail;
- init/rolled/tail child `L0MatmulPlan`s for each variant;
- one final L0C→L1 or L0C→GM drain with exact tile count, bytes, and atomic mode.

In **analytic mode**, the outer solver retains the earlier grounded surrogate: cube fractal work and
L1→L0 extraction use the PTO base geometry (normally 128×256), while GM traffic, L1 feasibility,
grid makespan, and split-K remain candidate-specific. Detailed L0 selection runs only after the
outer winner is known. This is cheap, but its fixed-base reload approximation must be improved and
silicon-compared with exact mode.

In **exact mode**, the buildable uniform-grid cost is the sum of the emitted phases. For one output
variant:

```text
group residents = one serial GM->L1 preload at first use, live through last use
retained panels = one serial full-panel GM->L1 preload per request-local selected side
full K windows  = first feed + max(first child, next feed)
                  + (Q-2) * max(rolled child, next feed) + last child
K tail          = GM feed + child L0 wall
final drain     = Acc->Mat or Acc->GM
```

Variant cost is multiplied by its exact count, then by ready-queue waves. Internal drains use the
grounded L0C→L1 pipe; roots use the grounded L0C→GM/FIXPIPE model. Split-K uses
`FirstPartialThenAtomic`: one serialized AIC launch computes share zero for every spatial region with
normal stores, then a second AIC launch computes the remaining shares with atomic-add stores. The
cost is the sum of both phase walls plus an explicit, zero-default synchronization term. Each launch
contributes its own kernel-fill waves. For each request the planner
compares no retention, LHS retention, RHS retention, and both. A choice is admitted only when the
complete panel lifetime plus the other operand's rolling window fits L1. Its preload is added
serially before the output-tile loop; subsequent K-window feeds omit exactly that retained side.

The child L0 wall similarly records serial first-block fill, rolled L1→L0/Cube overlap, serial
partial-K tail, and drain. The existing device-grounded chooser ordering is deliberately retained:
using the phase equation to re-select baseK without a grounded per-iteration event/synchronization
term falsely prefers the minimum 16-wide block. The chosen geometry is unchanged; the explicit
phase wall is used when composing the hierarchical cube candidate cost.

The emitter replays the same algorithm:

- one SPMD body per uniform spatial/split work unit;
- repeated compatible boundary values loaded once at their planned first use and reused by every
  request through their planned last use;
- request-local retained panels loaded once before that request's output-tile loop;
- output/L0C tile outer, GM K-window inner;
- one stage-2 full-window ring whose K=0 arm is `matmul` and later arms are
  `matmul_acc`, followed by a serial K tail;
- FP32/INT32 tensor-level accumulator values, followed by one narrowing/store drain;
- one L1 scratch per supported internal request and direct GM root assembly;
- one normal-store first-partial launch before the split-K atomic-rest launch.

Direct `Mat`→GM `tile.store` is legal PTO `TSTORE`; memory-space inference therefore does not insert
an unnecessary Mat→Vec move.

## 6. Current fidelity boundary

The admitted exact/co-optimized subset has one model/plan/emit algorithm. The default analytic
path is still a ranking surrogate and has several concrete fidelity gaps; it must not be described
as an exact replay of the reconstructed winner. Remaining work, in priority order, is:

1. **Operation and handoff capability.** Pure-cube admission must use one candidate-invariant
   capability descriptor shared by problem construction, both cost modes, and emit validation.
   Today the adapter records non-default `a_trans`/`b_trans`/`c_matrix_nz` semantics but declines
   them only for mixed groups, while cube replay hard-codes all three flags to false. A square
   transposed matmul can therefore pass shape checks and silently change meaning. Operand dtype,
   accumulator/storage dtype, and internal Acc->Mat conversion support are also checked too late:
   analytic mode can rank an FP32 internal handoff that emission rejects. Decline every unsupported
   case before solving, revalidate it at emission, and add pure-cube square transpose/layout,
   analytic FP32-chain, and unsupported-dtype regressions. Split-K roots must store the hardware
   accumulator dtype because both normal and atomic phases publish partial accumulators directly.

2. **Default analytic serial phase fidelity — host-closed.** The analytic path now composes a
   multi-request group from serial request-local phase roofs. Each request derives its own K-window
   init/rolled/tail structure from the existing L1 pebble headroom; only its emitted rolled loop may
   overlap GM->L1 feed with the analytic MTE1/Matrix surrogate. Internal Acc->Mat drains, root
   Acc->GM drains, repeated boundary-feed multiplicity across the surrogate L0 output grid, and
   group-resident preloads are additive at their emitted boundaries. A lone matmul retains the
   equivalent single-request equation. Fixed-grid two-request and fusion-decision regressions prevent
   the former `max(sum(compute), sum(DDR))` from returning. Split-K now serializes its
   first-partial and atomic-rest phase walls in both modes. `cube_split_sync_cycles` is a separate
   zero-default boundary term; it must remain zero until silicon isolates synchronization from the
   already charged launch fills.

3. **Requested-value pebbling policy.** Cross-request boundary reuse now has an explicit
   `CubeResidentBoundaryPlan`: compatible repeated inputs are preloaded at first use, retained until
   last use, included in peak L1, omitted from later GM feeds, and replayed by emission. Canonical
   identity includes operand role, so `A @ A` cannot alias its LHS and RHS representations. The
   initial policy always retains a repeated compatible value and declines the candidate if the
   combined produced/input lifetime peak does not fit. A later optimization may compare spill/reload
   alternatives. The shared interface now implements deterministic DFS and dependency-constrained
   Gorder; silicon-compare them before changing the DFS default. The common ordering/lifetime
   abstraction now also covers vector boundary inputs. Vector identity is `(tensor, replay phase)`
   rather than cube `(tensor, operand role)`: one UB slice is reused through its last consumer in a
   phase, while a stats/apply barrier creates two reads. A separate greedy reuse-distance strategy
   is intentionally deferred until a literature review of
   dependency-constrained locality ordering, register-pressure-aware DAG scheduling, and pebbling;
   do not add an ad hoc heuristic before its objective and complexity are compared with Gorder.

4. **Exact feasibility and objective scope.** Initial exact feasibility sizes boundary strips at
   the whole requested M/N region before the smaller emitted output/L0C tile is chosen, so it can
   conservatively reject legal low-task-count schedules. Solve output-tile and K-window feasibility
   to a fixed point or use a safe child-tile strip for initial admission. Exact mode faithfully
   replays the selected backend plan, but does not yet globally optimize every child L0 geometry,
   K-window, and retention combination; keep it opt-in and describe it as exact replay rather than
   a globally optimal emitted objective.

5. **Non-uniform grids.** A lone split=1 matmul has an explicit `ClampedOverlap` plan: every task
   computes the maximum static region, clamps a ragged edge backward, and charges its repeated
   reads, MADs, and drain. Ragged split-K is rejected because overlapping edge regions would have
   multiple atomic owners. A valid M/N region with a sub-fractal edge is also rejected consistently
   by analytic and exact compiler modes until the shared L0 plan separates physical padding from
   valid extents. Multi-matmul groups remain uniform-only.
6. **Retained boundary panels.** Exact/co-optimized mode compares the four bounded retention choices
   per request. A selected LHS or RHS is loaded once into L1, remains live through the output-tile
   loop, and is locally extracted for each child; cost and emit use the same lifetime and traffic.
   Descriptor-matched device validation confirmed the emitted request count and a 58.3% MTE2-byte
   reduction (`786432 -> 327680`) with unchanged Matrix/FIXPIPE work, producing an approximately
   17% execution-span win in the forced reuse-heavy case. A later bounded natural-plan study found
   exact-mode retention on narrow/wide shapes and confirmed that the selected retained plans beat
   their repeated-load controls. The remaining question is policy, not implementation correctness:
   profile the exact evaluator and candidate hit rate, then either keep retention as the opt-in exact
   mode or introduce a cheap analytic surrogate derived from the same child-tile reuse and L1
   capacity conditions. Do not add a blanket reuse bonus.
7. **Runtime scheduling boundary.** A8/E12 pipe tracing confirms the nested cube phase equation and
   the existing final `PIPE_ALL` barrier. A wider two-shape sweep then falsified a scalar
   per-work-unit correction: scheduler time was U-shaped with task count for `[272,272]`, but fell
   from 127 us to 33 us as fixed-shape `[512,512]` work was divided over more cores. A later
   constant-tile/constant-K sweep made per-task PTO byte-identical across 1, 2, 4, 8, 12, 16, 24,
   and 48 work units. Its linear per-task slope was approximately zero; silicon showed a small-count
   launch region, a flat 4–24 plateau, and a step at the second 24-core wave. Keep the current
   `ceil(work_units/24)` wave shape and add neither a scalar dispatch term nor vector C3.
8. **Low-precision and integer envelope.** BF16/FP16 on-chip handoff is represented. Exact mode
   declines same-type FP32 internal storage; the analytic pre-ranking gate is item 1. Other Acc→Mat
   conversion families require explicit PTO capability descriptors before admission.
9. **GM->L1 absolute pricing.** The flat 135 GiB/s term is not an accurate request-level transfer
   law. Device microbenchmarks confirm a real ND->NZ inner-width effect, but the tested two-bandwidth
   and shared-head descriptor models miss direct requests by 26--29% and fail their 10% transfer
   gate. A corrected replay through the actual per-output-tile/K-window phase equation, with dynamic
   request multiplicities, preserves the current B16 decision for every tested fixed-K problem.
   Keep the flat term as the current ranking surrogate; revisit it only with a direct-request model
   that passes held-out geometry/K tests and improves decisions among candidates for the *same*
   matmul problem. Never pool different K problems into one argmin.
10. **Evaluation cost and mode comparison.** An ephemeral L0-request memo reduced direct Release
   exact `best_cost()` measurements from roughly 13–54 ms to 2.2–9.4 ms on the sampled lone
   matmuls, and from roughly 22 ms to 2.3 ms on the sampled two-matmul chain. Analytic evaluation was
   roughly 0.1–3.8 ms. The memo lives only for one enumeration and never enters `CostResult` or the
   global search cache. Exact mode still constructs schedule descriptors in its candidate path;
   replace those with smaller scalar summaries if full-solver profiling shows they matter. Device
   A/B must measure whether lower outer-plan regret justifies the remaining prospective `-O3` cost.
   Profile candidate count, derivation time, L0-memo hit rate, and cube-cost share in a full solver
   run before optimizing isolated `best_cost()`. Emission also statically expands every output tile's
   K loop even though cost compresses them into four counted variants; add PTOAS code-size and
   compile-time gates, then use a runtime tile loop or explicit expansion limit if growth is material.

11. **Ragged recursive grids.** Different M/N/K shapes, deep chains, produced operands on either
   side, fan-out role changes, and multiple roots are already represented by the recursive request
   DAG; they are not a missing "variable-shape" feature. The narrower unsupported case is a
   non-uniform spatial grid for a multi-matmul group. Supporting it would require distinct valid and
   physical edge regions to be propagated through every producer request, with matching L1
   lifetimes and single-owner output semantics. Keep the current uniform-only admission unless
   workloads demonstrate enough value to justify that additional plan and emit complexity.

12. **Device/default closure.** Retained panels, clamped lone matmuls, recursive BF16/FP16 DAGs,
   the former zero-seed split protocol, serial K tails, and former allocation-overflow plans have targeted silicon
   evidence. Promote representative cases into the persistent device surface and broaden the dtype,
   shape, and multi-root matrix before enabling the generic cube emitter without its current guard.
   PTO Fusebox now reports `507 passed / 1 failed`. The stale `2MM`/`REUSE` assertions were replaced
   by descriptor-level `[CUBERES]` checks for one request, exact lifetime/bytes, role separation,
   and capacity decline. The remaining `FDM` failure is a real analytic ranking gap for a wide
   intermediate, not shared-boundary model/emit debt.

If exact replay is unavailable, strict mode fails with the rejected contract condition. Production
mode partitions or falls back to standalone matmuls rather than silently emitting another
algorithm.

The first clamped-grid silicon ranking exposed a system-boundary mismatch. For
`[272,272]@[272,272]`, analytic selects A8 (`144x80`, 8 tasks), exact selects E12 (`80x96`, 12
tasks), and a fixed grid lowers to the same binary in both modes. A8 is 7.3% faster under the
runtime's scheduler/orchestration execution span even though exact ranks E12 lower. Per-task traces
match the planned byte multiplicities and favor E12; the kernel already executes `PIPE_ALL` after
the final TSTORE, its barrier cycles equal FIXPIPE, and a redundant barrier adds no work. The full
device difference is in the scheduler phase, not a cube pipe. However, the follow-up sweep found no
shape-stable per-work-unit slope: the two tested shape families moved in opposite directions because
per-core kernel work still dominated the measured span. Add neither a pipe correction nor a scalar
dispatch correction. Analytic remains default and exact remains opt-in; the next gate is a
constant-tile, constant-K, variable-region-count sweep.

The same sweep exposed several finite exact candidates that failed memory allocation. They were not
true capacity misses. Pipeline peeling retained a constant-dead branch and duplicated the persistent
L0C accumulator; separately, a serial partial-K child was hoisted into the rolled phase and kept a
third Left/Right panel live. Early simplification, explicit serial-phase ordering, and preservation
of compiler call metadata now make the reported A4, B16/B24/B48, and split-K S16/S32 plans lower
with exactly the priced accumulator and operand-bank lifetimes. Device validation confirmed the
former overflow cases now build deterministically; A4 and B16/B24/B48 pass the existing cube
tolerance, while long-K S16/S32 retain the separately tracked FP32 reduction-order tolerance issue.

Mixed cube/vector fusion is outside this contract.

The latest silicon isolation preserves that boundary. At PyPTO `8a97865a`, the forced pure-cube
`[192,64]@[64,256]` kernel (`32,32,1,6,8`: 48 logical AIC blocks and four GM→L1 K windows) is
numerically correct on 910B2. Adding a separately launched AIV bias epilogue still corrupts the
result, including with 12 AIC blocks or a single K window. Runtime DFX proves that the two tasks use
the same allocation and that the covered AIC→AIV tensor dependency is present with the expected
`INOUT`/`INPUT` tags. That residual is tracked as a cross-engine final FIXPIPE/GM-visibility issue;
it does not block cube-only cost, schedule, or ranking validation.

## 7. Validation ladder

1. PTO Fusebox tests: recursive request geometry and L1 peak, dtype handoff capability, exact output
   variants, phase equations, first-partial/atomic-rest split ownership, compact cache, and no overlap for serial loops.
2. Host compiler tests: forced/natural plans, BF16 recursive DAGs, FP32-chain decline, ragged K,
   split/no-split, output-tile nesting, child descriptor consumption, Torch numerics, and full
   PTOAS-backed lowering.
3. Forced-plan device correctness: compare every output with Torch and inspect the emitted plan and
   instruction trace.
4. Device performance/model validation: measure MTE2/MTE1/Matrix/FIXPIPE utilization and bytes,
   task/grid behavior, split overhead, ranking correlation, and argmin regret. Do not fit a constant
   to one shape.
