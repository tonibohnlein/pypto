# AutoFuse mixed cube/vector schedule contract

**Status:** first buildable `C->V` increment implemented behind
`PYPTO_AUTOFUSE_MIXED=1`; silicon validation pending.
The homogeneous vector and cube contracts remain authoritative for work inside each engine.
This document defines the additional contract at cube/vector boundaries.

## 1. Hardware and execution model

Ascend 910B has no direct UB-to-Mat/L1 path. A tensor crossing between an AIC cube core and
an AIV vector core is written to GM by the producer and read from GM by the consumer. Fusion
therefore does not make crossing traffic free. Its benefit is a single launch and overlap between
the two engines when successive pipeline items use distinct GM FIFO slots.

The logical scheduling resource is 24 groups:

```text
group 0  = AIC 0  + AIV 0,1
group 1  = AIC 1  + AIV 2,3
...
group 23 = AIC 23 + AIV 46,47
```

Each group owns one cube lane and two vector lanes. Spatial work is distributed among groups;
the two vector lanes split a vector stage's rows. Cross-engine overlap occurs inside a group.

The grounding sources are pto-isa's `mixed_tile_study` and manual flash-attention kernel. The
former measures a skewed pipeline approaching `max(cube, vector) + fill`, while an unskewed
dependency chain takes `cube + vector` or worse. The latter supplies the full-attention stage
sequence and running-statistic algorithm.

## 2. A solver solution specifies two loop axes

A mixed solution cannot be described by one output tile count:

1. The **group grid** partitions independent query/output regions among up to 24 groups.
2. The **pipeline-item loop** supplies successive cross-engine items within each group.

For a simple matmul epilogue, one item may be an output strip. For flash attention, one item is a
key-axis chunk for a fixed query tile. The flash-attention loop is:

```text
QK matmul (C)
  -> online softmax update (V: running m,l)
  -> PV matmul (C)
  -> output update (V: rescale running O)
```

Thus `num_spatial_tiles >= 2` does not prove that a group has a successor item to overlap. The
schedule must record the actual per-group trip count.

## 3. MixedSchedulePlan

The cost model owns one `MixedSchedulePlan` per evaluated configuration. Candidate-invariant
stage topology is discovered once when the subgraph is created; candidate derivation adds grid,
loop, traffic, split, and overlap facts. The plan is not stored in `CostResult`, keeping the
local-search cache compact. It is re-derived once for a winning or forced configuration.

The current plan contains:

- maximal same-engine stages and their op membership;
- every cube/vector tensor transfer, its direction, and its producer/consumer stages;
- balanced spatial partitions, active group count, and split-K factor;
- the derived cube GM-to-L1 K window and the materialized vector-stage kind/peak UB;
- pipeline axis, chunk extent, item count, per-group trips, stage count, and skew depth;
- GM FIFO tensor, direction, valid tile shape, slot bytes, slot count, and total reserved bytes;
- the explicit AIV row/column split and number of compute lanes;
- `model_overlap_granted` and independently derived `overlap_implementable` bits.

Stage-local `CubeSchedulePlan` and `VectorStreamPlan` views remain the next
model increment; they are a required part of the complete contract, not fields
that exist in the first implementation.

The last two fields intentionally fail loud during migration. A cost may use `max` only when the
emitter can construct the recorded loop and PyPTO's lowering passes can realize its skew.

## 4. Pipeline modes

| mode | topology | PyPTO lowering | cost rule |
| ---- | -------- | -------------- | --------- |
| serial | no realizable successor item | sequential loop | sum stage walls |
| one-way | `C->V` or `V->C` | sequential per-engine loops decoupled by a GM FIFO | cross-engine wavefront only with at least two equal items per active group |
| single-round-trip skew | `C->V->C` or `V->C->V` | `SkewCrossCorePipeline` | max only when its structural skew predicate succeeds |
| multi-round-trip | e.g. full `C->V->C->V` attention | future whole-FIFO wavefront | serial until that transform exists |

Today `SkewCrossCorePipeline` safely skews exactly one push and one pop on the producer role. It
demotes multi-round-trip loops to sequential to preserve FIFO order. The solver must mirror that
predicate; structural alternation depth alone is insufficient.

## 5. Fidelity obligations

| Ref | model assumption | emitter obligation |
| --- | ---------------- | ------------------ |
| M1 | resources are 24 groups of 1 cube plus 2 vector lanes | launch and index exactly those active groups and row shards |
| M2 | every crossing tensor pays a GM write and read | emit `tile.move` boundaries that expand to matching push/pop FIFO traffic |
| M3 | stage work uses role-propagated regions | replay stage-local cube and vector plans without reclassifying shapes |
| M4 | a pipeline item has a named axis and chunk | build that loop inside each group with the recorded trip distribution |
| M5 | cross-engine overlap requires a realizable successor-item wavefront | emit FIFO-decoupled per-engine loops and satisfy the selected PyPTO expansion/skew predicate |
| M6 | serial prologue, drain, and ragged tail are additive | keep them outside the overlapped steady phase and price them separately |
| M7 | FIFO slots separate live producer/consumer items | request the plan's ring depth and preserve transfer order |
| M8 | vector P2/P4 statistics persist across items | carry `(m,l)` or other planned state through the mixed loop |
| M9 | full attention carries a running output | rescale old `O`, add the current PV partial, and finalize by the running sum |
| M10 | stage-local double buffering is conditional | grant each cube/vector roofline only when its local plan implements it |

## 6. Current implementation and audit

`Ascend910BMixed` already models four GM port directions, the shared-HBM cap, the 1:2 resource
ratio, cube/vector stage balance, and single-round-trip fill behavior. Pure groups delegate to the
homogeneous models.

`Subgraph::create` builds immutable same-engine stages and explicit crossing transfers once; each
winning or forced configuration re-derives a lightweight `MixedSchedulePlan`, while `CostResult`
remains compact. The plan now records actual launch groups, per-group trips, two-lane split, and
FIFO slots. `model_overlap_granted` equals `overlap_implementable`: at least two equal trips must
exist on every active group. Two global tiles on two groups are correctly serial.

The compiler buildability mode admits only one default-orientation standard matmul followed by a
linear, same-shape, PTO-grounded elementwise epilogue (`C->V`). It requires a uniform grid,
split-K 1, no escaped intermediate, one output, and an exactly materialized per-AIV half tile.
The two matmul operands must have the same floating PTO cube dtype (`FP16`, `BF16`, or `FP32`),
the result must be FP32, and every tensor operand/result in the vector epilogue must remain FP32.
A lower-precision result is a clean partition boundary until the plan and emitter represent a
separate accumulator carry and one final FIXPIPE narrow before the FIFO push. `INT8->INT32` waits
for an integer vector-family capability table; implicit tensor promotion is never assumed.
Feasibility includes the half-tile UB lifetime plus all eight full crossing FIFO slots. AutoFuse
emits:

```text
spmd(active_groups)
  split UP_DOWN
    sequential(per_group_trips)
      tensor.matmul tile
      elementwise epilogue tile
      assemble output tile
```

`LowerAutoVectorSplit` converts the UP_DOWN contract into real half-row AIV work;
`ExpandMixedKernel` constructs the GM-backed push/pop FIFO; `InjectGMPipeBuffer` supplies its
workspace. The outer mixed loop deliberately is not `ForKind::Pipeline`: a generic pipeline tag
would multiply nested AutoTileL0 buffers. The independently running AIC/AIV functions and FIFO
backpressure form the cross-engine wavefront. A complete host structural test verifies 48 logical
regions -> 24 group launches x 2 trips, one push/pop/free in each physical loop, 4096-byte slots,
and both AIV row shards. Tensor-level numeric replay matches the unfused matmul epilogue, while
transposed/NZ matmuls cleanly decline before solving. Current candidate enumeration exposes at most
two trips per group; deeper FIFO backpressure is therefore not part of the buildable surface yet.

The buildable cost now prices the exact grounded primitive chain on each valid AIV half tile,
including one stream startup per item. It applies role-aware boundary-input multiplicities
(`[M,N]`: 1, `[M,1]`: `parts_n`, `[1,N]`: `2*parts_m`, scalar:
`2*spatial_tiles`). TPUSH/TPOP traffic is blocking inside each stage: ordinary GM-to-L1 feed may
overlap cube work, then the crossing write adds; the vector crossing read, pointwise chain, and
final store add. Only complete successor items receive the two-stage cross-engine wavefront.
Cube GM-to-L1 feed overlaps MAD only when the derived K window produces the emitted three-or-more
chunk loop. All full chunks, including K=0, share that ring; its cost is `first feed + steady-state
roofline + last child + serial ragged tail + blocking crossing push`. One- and two-window schedules
therefore serialize load and compute exactly as the emitter does, even when K contains several
fractals.

The remaining model/emit gaps are:

- active groups are still a deterministic mapping (`min(spatial tiles, 24)`), not a separately
  enumerated choice between more serial groups and fewer pipelined groups;
- the current materialized pointwise subset is exact, but a stage-local `VectorStreamPlan` is still
  required before mixed pointwise strip streaming or P2/P4 can be admitted;
- low-precision floating matmul outputs require an explicit FP32 K-window carry plus final FIXPIPE
  narrow; compiler mode declines them instead of silently rebuilding a full-K matmul;
- promoted/mixed-dtype vector operands and `INT8->INT32` epilogues remain cut until their cast or
  integer primitive semantics are represented and priced;
- a direct QK matmul plus an exact softmax cone can now reuse the P4 vector-stage descriptor, but
  mixed costing does not yet replay its phase-local compute and traffic;
- stage-local cube request topology is constructed only for homogeneous cube groups;
- FIFO depth eight is explicit and faithful, but smaller depths have not been device-compared and
  are not yet a scheduling dimension;
- the cross-engine wavefront and mixed launch overhead still need latest-PTOAS silicon grounding;
  host lowering proves ordering and capacity, not AIC item `k+1` overlap with AIV item `k`;
- analytic `C->V->C->V` topology is retained and receives a serial stage sum, while compiler mode
  cuts it; the current unified spatial grid also cannot express its key-chunk loop.

These are explicit migration gaps, not permission for the emitter to approximate the plan.

## 7. Implementation sequence

1. **Done:** add candidate-invariant stage/transfer topology and a lightweight
   `MixedSchedulePlan`; consume it without changing canonical cost anchors.
2. **Done:** remove the optimistic global-tile overlap grant; record launch groups, equal
   per-group trips, AIV split/lane count, and FIFO slots.
3. **Done (host-ready, device pending):** emit and fully lower the exact materialized `C->V`
   matmul epilogue; include FIFO reservation, blocking crossing traffic, exact per-lane vector work,
   broadcast multiplicity, live-out, and matmul-semantic gates. Unsupported mixed topologies remain
   partition boundaries.
4. Enumerate or analytically choose between more serial groups and fewer pipelined groups. Price
   dependent init, steady, tail, and drain phases separately.
5. Build stage-local `CubeSchedulePlan` and `VectorStreamPlan` views for mixed components and use
   their exact compute, traffic, liveness, and double-buffer decisions.
6. Add the symmetric `V->C` and exact single-round-trip emitters. Mirror the complete skew
   capability predicate before granting overlap.
7. Reuse the implemented embedded P4 stage descriptor in stage-local mixed compute and traffic;
   continue rejecting any extra vector prefix or tail outside that exact cone.
8. Add whole-FIFO multi-round-trip skew, then implement full flash attention with the key-chunk
   loop and running `(m,l,O)` state.

Default mixed fusion remains off until plan/emit structural tests and 910B correctness and
wall-time validation close M1-M10.
