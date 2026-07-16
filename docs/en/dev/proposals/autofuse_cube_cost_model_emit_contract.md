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
- optional boundary panels retained in L1 across one request's output-tile loop;
- each matmul's GM→L1 K window, chunk, init, rolled loop, and tail;
- the L0C-resident output-tile grid;
- internal Acc→Mat and root Acc→GM drains;
- split-K seed and atomic ownership.

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
| C4 | Peak L1 is the request-order pebble peak | Replay producer-before-consumer, retain through the priced last use, and allocate no unpriced scratch |
| C5 | Every request owns its GM→L1 K loop | Use that request's contraction, chunk, rolled stage, and serial tail |
| C6 | One complete output tile remains in L0C across all K windows | Nest output tiles outside the K-window loop; never spill a partial to Mat and reload it into Acc |
| C7 | Exact mode prices a concrete shared-backend L0 plan; analytic mode prices the grounded surrogate | Always attach the output-residency intent; attach and validate detailed geometry only in exact mode; let `AutoTileMatmulL0` realize both |
| C8 | Overlap is local to one concrete full-window loop | Put K=0 and every rolled full window in the same eligible stage ring; add only its fill/drain, the ragged tail, and final output drain serially |
| C9 | GM traffic follows the emitted output-tile loop | Charge repeated boundary-panel loads by default; when the plan selects retention, add one serial GM→L1 preload, keep that panel live, and remove only its represented per-tile feeds |
| C10 | Split-K writes `S` atomic partials | Emit one ordered, tiled zero seed and `S` disjoint K shares, or select `S=1` |
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

## 5. Implemented schedule and cost

For a requested `O[rows,cols] = A @ B`, reverse request propagation produces
`A[rows,K]` and `B[K,cols]`. The postorder supports deep chains and trees, produced RHS operands,
both inputs produced, fan-out role changes, and multiple roots. Distinct requested roles become
distinct producer instances and pay recomputation; identical instances are shared in the request
DAG.

For every request, the plan records:

- concrete input/output regions and symbolic axis bindings;
- producer-instance dependencies and storage/accumulator dtypes;
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
retained panels = one serial full-panel GM->L1 preload per selected side
full K windows  = first feed + max(first child, next feed)
                  + (Q-2) * max(rolled child, next feed) + last child
K tail          = GM feed + child L0 wall
final drain     = Acc->Mat or Acc->GM
```

Variant cost is multiplied by its exact count, then by ready-queue waves. Internal drains use the
grounded L0C→L1 pipe; roots use the grounded L0C→GM/FIXPIPE model. A split seed is an explicit serial
vector fill/store/task phase and contributes its own kernel-fill wave. For each request the planner
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
- selected boundary panels loaded once to L1 before the request's output-tile loop;
- output/L0C tile outer, GM K-window inner;
- one stage-2 full-window ring whose K=0 arm is `matmul` and later arms are
  `matmul_acc`, followed by a serial K tail;
- FP32/INT32 tensor-level accumulator values, followed by one narrowing/store drain;
- one L1 scratch per supported internal request and direct GM root assembly;
- one tiled vector seed before split-K atomic root stores.

Direct `Mat`→GM `tile.store` is legal PTO `TSTORE`; memory-space inference therefore does not insert
an unnecessary Mat→Vec move.

## 6. Current fidelity boundary

The buildable subset now has one model/plan/emit algorithm. Remaining work is:

1. **Non-uniform grids.** A lone split=1 matmul has an explicit `ClampedOverlap` plan: every task
   computes the maximum static region, clamps a ragged edge backward, and charges its repeated
   reads, MADs, and drain. Ragged split-K is rejected because overlapping edge regions would have
   multiple atomic owners. A valid M/N region with a sub-fractal edge is also rejected consistently
   by analytic and exact compiler modes until the shared L0 plan separates physical padding from
   valid extents. Multi-matmul groups remain uniform-only.
2. **Retained boundary panels.** Exact/co-optimized mode now compares the four bounded retention
   choices per request. A selected LHS or RHS is loaded once into L1, remains live through the
   output-tile loop, and is locally extracted for each child; cost and emit use the same lifetime and
   traffic. Analytic mode deliberately retains the prior repeated-load surrogate and emits no
   retained panel. Device validation must confirm the predicted MTE2 reduction and silicon win
   before retention is considered for the analytic default.
3. **Runtime scheduling boundary.** A8/E12 pipe tracing confirms the nested cube phase equation and
   the existing final `PIPE_ALL` barrier. A wider two-shape sweep then falsified a scalar
   per-work-unit correction: scheduler time was U-shaped with task count for `[272,272]`, but fell
   from 127 us to 33 us as fixed-shape `[512,512]` work was divided over more cores. A later
   constant-tile/constant-K sweep made per-task PTO byte-identical across 1, 2, 4, 8, 12, 16, 24,
   and 48 work units. Its linear per-task slope was approximately zero; silicon showed a small-count
   launch region, a flat 4–24 plateau, and a step at the second 24-core wave. Keep the current
   `ceil(work_units/24)` wave shape and add neither a scalar dispatch term nor vector C3.
4. **Low-precision and integer envelope.** BF16/FP16 on-chip handoff is represented. Same-type FP32
   internal storage declines. Other Acc→Mat conversion families require explicit PTO capability
   descriptors before admission.
5. **Device grounding.** A8/E12 validates descriptor consumption, exact per-task operand/FIXPIPE
   bytes, nested MTE2/MTE1/Matrix/FIXPIPE execution, final-drain completion, and forced-grid ranking.
   Extend the same evidence to internal narrowing, split seed/atomic behavior, retained panels, and
   variable-shape schedules.
6. **Evaluation cost and mode comparison.** An ephemeral L0-request memo reduced direct Release
   exact `best_cost()` measurements from roughly 13–54 ms to 2.2–9.4 ms on the sampled lone
   matmuls, and from roughly 22 ms to 2.3 ms on the sampled two-matmul chain. Analytic evaluation was
   roughly 0.1–3.8 ms. The memo lives only for one enumeration and never enters `CostResult` or the
   global search cache. Exact mode still constructs schedule descriptors in its candidate path;
   replace those with smaller scalar summaries if full-solver profiling shows they matter. Device
   A/B must measure whether lower outer-plan regret justifies the remaining prospective `-O3` cost.

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
   variants, phase equations, split seed/fill, compact cache, and no overlap for serial loops.
2. Host compiler tests: forced/natural plans, BF16 recursive DAGs, FP32-chain decline, ragged K,
   split/no-split, output-tile nesting, child descriptor consumption, Torch numerics, and full
   PTOAS-backed lowering.
3. Forced-plan device correctness: compare every output with Torch and inspect the emitted plan and
   instruction trace.
4. Device performance/model validation: measure MTE2/MTE1/Matrix/FIXPIPE utilization and bytes,
   task/grid behavior, split overhead, ranking correlation, and argmin regret. Do not fit a constant
   to one shape.
