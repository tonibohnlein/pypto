# AutoFuse cube cost-model ↔ emit contract

## 1. Scope and algorithm

This contract covers homogeneous AIC groups containing only `tensor.matmul` operations. Mixed
cube/vector kernels remain a separate charter. Unlike vector reductions, a cube group needs no
online softmax/Welford algorithm: every operation is a matrix contraction. Its fidelity problem is
nevertheless non-trivial—spatial ownership, recursive operand regions, L1-resident intermediates,
per-operation K streams, L0 subdivision, and split-K must describe the same kernel in the model and
emitter.

A solver-owned `CubeSchedulePlan` specifies one fixed `TileConfig` and selected parallel split. The
candidate-invariant request DAG is built once in `Ascend910BCost::create()`. Candidate evaluation
uses that compact topology plus an O(nodes) window/peak derivation; it does **not** put a complete
plan into `CostResult` or the local-search cache. AutoFuse re-derives the full plan only for a winning
or explicitly forced configuration.

Each sink request is discovered backwards because a consumer determines the exact region it needs
from each producer. Plan replay is producer-before-consumer. A producer is memoized by
`(tensor, height binding, width binding)` within one work unit. Identical requests are shared by the
model; distinct fan-out roles become distinct schedule instances and are explicitly recomputed.

## 2. Obligations

| Ref | The model assumes | The emitter must build |
| --- | ----------------- | ---------------------- |
| C1 | `parts_m × parts_n × split_k` independent work units | The same spatial regions and K shares, with one SPMD body per work unit |
| C2 | Only boundary operands and boundary outputs touch GM | Keep every ephemeral result in L1/Acc; never use GM as an implicit chain buffer |
| C3 | Consumers back-propagate exact row/column requests | Slice LHS as `[requested rows,K]` and RHS as `[K,requested cols]`, including produced operands on either side |
| C4 | Peak L1 is the live-region pebble peak in request order | Replay producer-before-consumer, retain regions through their priced last use, and allocate no unpriced live scratch |
| C5 | Every matmul owns an L1 window and K loop | Use that instance's contraction, chunk, rolled-loop stage, and serial tail; never reuse the sink K loop globally |
| C6 | A requested region is subdivided into grounded L0 accumulator tiles | Use the plan's L0 M/N sizes and explicitly assemble every L0 result into its planned L1 or boundary destination |
| C7 | GM traffic overlaps compute only behind a real stage-2 rolled loop | Grant `max(compute,DDR)` only when every contributing boundary-load phase can ping-pong; serialize init and tail work |
| C8 | Sink split-K writes `S` partials with atomic add | Emit a zero seed and exactly `S` disjoint K shares per spatial region, or choose/price `S=1` |
| C9 | Reload counts include the actual loop nest and reuse | Charge every GM→L1 load induced by L0 subdivision, while crediting reuse only when the emitter retains that panel |

## 3. General matmul-DAG request propagation

For a requested output region `O[rows, cols]` of `O = A @ B`:

- request `A[rows, K]` from the left input;
- request `B[K, cols]` from the right input;
- recursively produce an internal operand at exactly that requested orientation;
- stream boundary portions along this operation's own K-loop plan.

This rule supports deep chains and trees, a produced RHS, both inputs produced, non-square internal
shapes, fan-out, and multiple sinks. It replaces the historical shared-M/full-N shortcut: an
intermediate used as a left operand needs a row band, while one used as a right operand needs a
column band. A tensor used in both roles can therefore have two producer instances. Identical
requests may instead share one instance and remain live until their last consumer.

With one sink, its contraction can bind to `ParallelK`: each work unit requests only its K share
from either boundary or produced operands. All internal contractions remain independent sequential
K streams. Multiple sinks currently force `split_k=1`, because a single solver split coordinate
cannot identify multiple atomic targets.

## 4. Implemented state (2026-07-12)

The solver now provides:

- a candidate-invariant recursive request DAG, memoized by tensor and axis bindings;
- role-aware feasibility, live-region peak L1, boundary reload, cube MAC/extract work, and
  recomputation for arbitrary pure-matmul DAGs; MAC/extract precision follows operand dtype rather
  than the often-FP32 accumulator/output dtype;
- one `CubeMatmulSchedule` per request instance with producer dependencies, concrete regions,
  contraction/share, L1 window, pipelined K chunk, full trips, tail, and L0 M/N tile sizes;
- producer-before-consumer execution order, roots, split/work-unit counts, seed requirement, and
  overlap/buildability bits in the reconstructed `CubeSchedulePlan`;
- a conservative overlap gate: a global cube roofline receives `max(compute,DDR)` only when every
  boundary-loading request instance reconstructs a real stage-2 rolled K loop. The former scalar
  `K/S >= 32` test could grant overlap to a one-trip loop and is retired;
- a buildability gate used by AutoFuse that enumerates only uniform M/N grids for multi-matmul
  groups. Analytic solver use remains unrestricted, and lone matmuls retain their established
  balanced-grid search space.

The generic AutoFuse emitter now consumes that plan for buildable multi-matmul groups. It:

- replays schedule instances in producer-before-consumer order and uses exact plan bindings for
  spatial offsets, K-share offsets, produced LHS/RHS values, and deep trees;
- recomputes a shared producer for distinct fan-out roles exactly when the plan has distinct
  instances, while materializing an identical request once;
- keeps small intermediates local and assembles an intermediate spanning several L0 tiles into an
  explicit L1 tensor for its consumer;
- emits each planned K rolled loop as stage-2 pipeline plus a serial tail;
- emits one seed SPMD scope plus atomic root stores for split-K, and direct stores for `S=1`;
- handles multiple roots at `S=1` and lowers a large-L1-intermediate chain through the default pass
  stack.

If exact replay is unavailable, strict mode fails with the rejected contract condition. Production
mode falls back to dependency-ordered standalone matmuls rather than silently emitting a different
fused algorithm.

## 5. Remaining fidelity work

These gaps prevent calling the cube path complete or using it as a device-ranking oracle:

1. **L0-subtile reload multiplicity (C9).** `cube_request_reload()` charges each logical boundary
   request once per work unit. The current emitter nests a complete K stream inside every L0 output
   subtile. If an output region has three L0-N subtiles, for example, its LHS panel is loaded three
   times. `CubeExtractCycles` prices the corresponding L1→L0 reuse, but GM→L1 currently assumes a
   stronger retained-panel algorithm. Either the cost must multiply traffic by the emitted M/N
   subtile loop, or the emitter must use a K-outer retained-panel algorithm and price its partial
   accumulator/L1 traffic.
2. **Phase-local cube rooflines (C7).** The serial-loop fiction is closed conservatively, but the
   cost still applies one global `max` or `sum` to all request instances. It must become a sum of
   per-node init/rolled/tail/store phases so serial tails are never hidden by unrelated work.
3. **Non-uniform grids (C1).** A multi-op plan with unequal balanced region shapes is filtered by
   the AutoFuse buildability gate. The legacy lone-matmul ceil+clamp emitter is numerically
   idempotent but can execute more max-size work than the balanced-grid cost. Exact non-uniform
   static-shape emission or matching cost remains open.
4. **Shared boundary-panel lifetime (C4/C9).** The model deduplicates identical boundary requests.
   Until an explicit shared L1 panel and its lifetime are represented, the plan emitter declines
   such a group instead of emitting duplicate unpriced loads.
5. **Seed and launch cost (C8).** The vector zero-seed/barrier and per-task launch overhead are not
   yet included in the cube candidate cost.
6. **Low-precision final outputs.** Floating K carries accumulate in FP32. Until the plan represents
   a final FIXPIPE narrowing phase, a streamed/split cube plan whose root is FP16/BF16 declines to
   the original standalone matmul rather than building an ill-typed low-precision accumulator.
7. **Device grounding.** Host tests validate structure, numerics, and lowering; they do not validate
   wall time, MTE2/Matrix overlap, traffic multiplicity, atomic behavior, or plan ranking on 910B2.

Mixed cube/vector fusion is outside this contract.

## 6. Validation ladder

1. Solver unit tests: fixed lone-matmul anchors; exact regions and L1 peak for produced-LHS,
   produced-RHS, both-input-produced, fan-out role switches, deep chains, multi-sink; compact cache;
   and a one-trip loop that must not receive overlap.
2. Host numeric/codegen: lone and recursive DAGs, exact/ragged K, split/no-split, non-square produced
   operands, fan-out, multiple outputs, large L1 intermediates, strict rejection, and lowering
   through the default pass stack.
3. Forced-plan device correctness: compare every output with Torch and inspect traces to confirm
   that intermediates do not touch GM unexpectedly.
4. Device performance/model validation: measure per-op MTE2/Matrix overlap, L0-subtile input reloads,
   split seed/atomic overhead, ragged tails, grid occupancy, and forced-plan ranking/regret. Compare
   model cycles to wall time only through the established calibration; do not fit to one shape.
