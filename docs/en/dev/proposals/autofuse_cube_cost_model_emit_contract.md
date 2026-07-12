# AutoFuse cube cost-model ↔ emit contract

## 1. Scope and algorithm

This contract covers homogeneous AIC groups containing only `tensor.matmul` operations. Mixed
cube/vector kernels remain a separate charter. Unlike vector reductions, a cube group needs no
online softmax/Welford algorithm: every operation is a matrix contraction. The hard part is still
algorithmic fidelity—spatial ownership, recursive operand regions, L1-resident intermediates,
per-operation K streams, and split-K must describe the same kernel in the model and emitter.

A `CubeSchedulePlan` is the solver-owned specification for one fixed `TileConfig` and selected
parallel split. Candidate evaluation keeps only the compact cost result plus the lightweight
peak/per-op-K derivation it needs. The full plan is rediscovered once for a winning or forced
configuration; it is deliberately absent from `CostResult`.

The kernel is discovered from each sink backwards, because a consumer determines the exact region
it needs from each producer. Execution is nevertheless producer-before-consumer in the solver's
`execution_order()`. A producer is memoized by `(tensor, requested region)` within one work unit;
shared producers are not recomputed unless the cost explicitly includes that recomputation.

## 2. Obligations

| # | The model assumes | The emitter must build |
|---|---|---|
| C1 | `parts_m × parts_n × split_k` independent work units | The exact disjoint spatial partitions and K shares, with one SPMD body per work unit |
| C2 | Only boundary operands and boundary outputs touch DDR | Every ephemeral matmul result remains in L1/Acc and never round-trips GM |
| C3 | A consumer back-propagates an exact row/column region to each operand | LHS gets the consumer's output rows plus K; RHS gets K plus the consumer's output columns, including producers on either input |
| C4 | Peak L1 is the live-band pebble peak in `execution_order()` | Emit in that order, materialize each requested intermediate once, and release it at its priced last use |
| C5 | Each matmul has its own L1 window and GM→L1 K stream | Use that operation's planned load chunk, rolled-loop stage, and serial tail; never reuse the sink K for every operation |
| C6 | A large solver region is subdivided into pto-isa L0 accumulator tiles | Explicitly compute L0c-fitting subtiles when a tensor-level `matmul_acc` would otherwise materialize the full region |
| C7 | A stage-2 roofline hides GM traffic only behind a real rolled loop | Grant `max(compute, DDR)` only to a loop with two simultaneously fitting load buffers and enough rolled iterations; price serial init/tail separately |
| C8 | Sink split-K writes `S` partials with atomic add | Emit a zero seed/barrier and exactly `S` disjoint K shares per spatial region, or price/choose `S=1` |
| C9 | Ragged regions and K tails perform specified extra work | Use the planned balanced region geometry and an explicit serial, 16-aligned K tail; include both in cost |

## 3. General matmul-DAG region propagation

For a requested output region `O[rows, cols]` of `O = A @ B`:

- request `A[rows, K]` from the left input;
- request `B[K, cols]` from the right input;
- stream boundary portions along K using this operation's K-loop plan;
- recursively produce an internal operand at that requested orientation.

This rule naturally supports deep trees, a produced RHS, and a sink whose two operands are both
produced. It also exposes why the historical full-width M-band formula is not general: that formula
is correct for an intermediate used as a left operand, but a produced right operand needs a
full-K/N-band instead. Fan-out can yield multiple requested regions for one tensor. Identical
requests are shared; different requests are either separately priced computations or make the
candidate unsupported until a materialization strategy is modeled.

Parallel split-K applies only to the selected boundary sink. An internal producer feeding that
sink may be recomputed per K-share only if the compute term includes it. Otherwise the plan must
produce the exact K subregion needed by the sink. Multiple boundary sinks require one seed and
atomic target per sink and must not be represented by one implicit split coordinate.

## 4. Current implementation status

Implemented in the first `CubeSchedulePlan` increment:

- final/forced-plan reconstruction in `mlsys26`, including exact grid counts, split, execution
  order, peak L1, per-op resident window, actual load chunk, loop stage, and ragged tail;
- solution JSON serialization and direct AutoFuse handoff without enlarging `CostResult`;
- cost-path removal of the previous duplicate `derive_exec` call;
- lone-matmul emission consumes the planned K loop;
- solver regions larger than L0c are explicitly subdivided within the 128×256 bounds used by the
  grounded Phase-D model, with exact disjoint ragged edge tiles, so K accumulation no longer
  allocates one oversized Acc/UB tensor or requires divisor-shaped tiles;
- split-K seed/atomic emission remains active, with numeric tests for exact and ragged K streams;
- characterization marks a produced RHS as `emit_compatible=false`; strict validation fails loudly
  before the legacy pair matcher can consume it, while production uses dependency-ordered
  standalone matmuls as a correctness fallback.

The extraction is cost-preserving. In particular, it has not yet replaced the historical
`K/S ≥ 32` roofline gate. A characterized 64×64 matmul reaches per-core K=32: the model grants
overlap, while the concrete plan has only a serial matmul (`model_overlap_granted=true`,
`overlap_implementable=false`). Keeping both bits makes the gap explicit without changing solver
argmins in the descriptor commit.

Known gaps before general-DAG emission is trustworthy:

- the plan still records the historical shared-M/full-N internal-band algorithm; role-aware
  recursive regions are the next solver increment;
- `TileChainedMatmul` does not yet consume every per-op plan field and still has an emitter-owned
  deep-intermediate panel choice;
- balanced disjoint non-uniform regions are not yet emitted exactly for atomic split-K;
- the global cube roofline must be replaced with phase-local serial init/tail plus rolled-loop
  rooflines; the correction must be a separate, measured cost-model change;
- split seed/barrier cost is not represented;
- mixed cube/vector cost and emission are outside this contract.

## 5. Validation ladder

1. Solver characterization: fixed costs, L1 peak, execution order, per-op windows, produced-LHS,
   produced-RHS, both-input-produced, fan-out, deep chains, multi-sink, and compact cache size.
2. Host numeric/codegen: lone and deep DAGs, exact/ragged K, non-uniform M/N, split/no-split,
   produced operands on both sides, shared producers, and lowering through the default pass stack.
3. Forced-plan device correctness: output versus Torch and confirmation that no intermediate
   appears in GM traffic.
4. Device performance/model validation: grid occupancy, per-op MTE2/Matrix overlap, split seed
   overhead, ragged tails, and forced-plan ranking/regret. Compare model cycles to wall time only
   through the established calibration; do not fit the model to one shape.
