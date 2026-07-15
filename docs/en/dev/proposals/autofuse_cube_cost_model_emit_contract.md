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
| C1 | `parts_m × parts_n × split_k` independent work units | Launch exactly that SPMD grid and preserve one owner per output element |
| C2 | Only boundary tensors touch GM | Keep every supported intermediate in L1; never use GM as an implicit fused-chain buffer |
| C3 | Consumers back-propagate exact row/column requests | Slice LHS as `[requested rows,K]` and RHS as `[K,requested cols]`, including produced operands on either side |
| C4 | Peak L1 is the request-order pebble peak | Replay producer-before-consumer, retain through the priced last use, and allocate no unpriced scratch |
| C5 | Every request owns its GM→L1 K loop | Use that request's contraction, chunk, rolled stage, and serial tail |
| C6 | One complete output tile remains in L0C across all K windows | Nest output tiles outside the K-window loop; never spill a partial to Mat and reload it into Acc |
| C7 | Exact mode prices a concrete shared-backend L0 plan; analytic mode prices the grounded surrogate | Always attach the output-residency intent; attach and validate detailed geometry only in exact mode; let `AutoTileMatmulL0` realize both |
| C8 | Overlap is local to one concrete full-window loop | Put K=0 and every rolled full window in the same eligible stage ring; add only its fill/drain, the ragged tail, and final output drain serially |
| C9 | GM traffic follows the emitted output-tile loop | Charge repeated boundary-panel loads when another output tile reloads them; credit reuse only when represented |
| C10 | Split-K writes `S` atomic partials | Emit one ordered, tiled zero seed and `S` disjoint K shares, or select `S=1` |
| C11 | Cube accumulation dtype differs from storage dtype | Accumulate float inputs in FP32, then narrow once at the planned BF16/FP16 or GM drain |

## 4. Grounded loop structure

The PTO A2/A3 performance GEMM uses this hierarchy:

```text
for output tile (m, n):
    for K window loaded GM -> L1:       # L1 ping/pong
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
full K windows  = first feed + max(first child, next feed)
                  + (Q-2) * max(rolled child, next feed) + last child
K tail          = GM feed + child L0 wall
final drain     = Acc->Mat or Acc->GM
```

Variant cost is multiplied by its exact count, then by ready-queue waves. Internal drains use the
grounded L0C→L1 pipe; roots use the grounded L0C→GM/FIXPIPE model. A split seed is an explicit serial
vector fill/store/task phase and contributes its own kernel-fill wave.

The child L0 wall similarly records serial first-block fill, rolled L1→L0/Cube overlap, serial
partial-K tail, and drain. The existing device-grounded chooser ordering is deliberately retained:
using the phase equation to re-select baseK without a grounded per-iteration event/synchronization
term falsely prefers the minimum 16-wide block. The chosen geometry is unchanged; the explicit
phase wall is used when composing the hierarchical cube candidate cost.

The emitter replays the same algorithm:

- one SPMD body per uniform spatial/split work unit;
- output/L0C tile outer, GM K-window inner;
- one stage-2 full-window ring whose K=0 arm is `matmul` and later arms are
  `matmul_acc`, followed by a serial K tail;
- FP32/INT32 tensor-level accumulator values, followed by one narrowing/store drain;
- one L1 scratch per supported internal request and direct GM root assembly;
- one tiled vector seed before split-K atomic root stores.

Direct `Mat`→GM `tile.store` is legal PTO `TSTORE`; memory-space inference therefore does not insert
an unnecessary Mat→Vec move.

## 6. Current fidelity boundary

The uniform buildable subset now has one model/plan/emit algorithm. Remaining work is:

1. **Non-uniform grids.** A lone split=1 matmul retains the legacy ceil-and-clamp path. Ragged
   split-K is rejected because overlapping edge regions would have multiple atomic owners.
   Multi-matmul groups remain uniform-only. The legacy lone path still uses the older analytic cost.
2. **Shared boundary panels.** The current output-tile-outer emitter may reload a shared LHS for each
   N output tile (or RHS for each M tile). The hierarchical cost charges that multiplicity. Explicit
   panel retention can be added later only with a matching lifetime and traffic change.
3. **L0 synchronization grounding.** Geometry selection remains on the validated aggregate PTO
   oracle. The phase composer needs device/op-simulator validation before a per-baseK event cost is
   introduced or geometry selection is changed.
4. **Low-precision and integer envelope.** BF16/FP16 on-chip handoff is represented. Same-type FP32
   internal storage declines. Other Acc→Mat conversion families require explicit PTO capability
   descriptors before admission.
5. **Device grounding.** Validate descriptor consumption, nested MTE2/MTE1/Matrix/FIXPIPE overlap,
   exact GM/L1 traffic, internal narrowing, split seed/atomic behavior, and forced-plan ranking on
   Ascend 910B2 with the latest PTOAS.
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

Mixed cube/vector fusion is outside this contract.

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
