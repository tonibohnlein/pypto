# AutoFuse mixed cube/vector schedule contract

**Status:** design and staged implementation plan for Ascend 910B mixed AutoFuse kernels.
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

The plan contains:

- maximal same-engine stages and their op membership;
- every cube/vector tensor transfer, its direction, and its producer/consumer stages;
- balanced spatial partitions, active group count, and split-K factor;
- pipeline axis, chunk extent, item count, per-group trips, stage count, and skew depth;
- GM FIFO direction and slot requirements;
- stage-local cube/vector schedules;
- `model_overlap_granted` and independently derived `overlap_implementable` bits.

The last two fields intentionally fail loud during migration. A cost may use `max` only when the
emitter can construct the recorded loop and PyPTO's lowering passes can realize its skew.

## 4. Pipeline modes

| mode | topology | PyPTO lowering | cost rule |
| ---- | -------- | -------------- | --------- |
| serial | no realizable successor item | sequential loop | sum stage walls |
| one-way | `C->V` or `V->C` | pipelined loop plus GM FIFO | max only with at least two items per active group |
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
| M5 | `max(stage walls)` requires realizable overlap | use `ForKind::Pipeline` and satisfy the selected PyPTO skew/lowering predicate |
| M6 | serial prologue, drain, and ragged tail are additive | keep them outside the overlapped steady phase and price them separately |
| M7 | FIFO slots separate live producer/consumer items | request the plan's ring depth and preserve transfer order |
| M8 | vector P2/P4 statistics persist across items | carry `(m,l)` or other planned state through the mixed loop |
| M9 | full attention carries a running output | rescale old `O`, add the current PV partial, and finalize by the running sum |
| M10 | stage-local double buffering is conditional | grant each cube/vector roofline only when its local plan implements it |

## 6. Current model audit

`Ascend910BMixed` already models four GM port directions, the shared-HBM cap, the 1:2 resource
ratio, cube/vector stage balance, and single-round-trip fill behavior. Pure groups delegate to the
homogeneous models.

The first solver-side contract increment is implemented. `Subgraph::create` builds immutable
same-engine stages and explicit crossing transfers once; each winning or forced configuration
re-derives a lightweight `MixedSchedulePlan`, while `CostResult` remains compact. Tests cover
one-way, single-round-trip, multi-transfer, per-group trip counts, and embedded exact P4 detection.

The remaining model/emit gaps are:

- the canonical cost still grants some cross-engine overlap from global tile count rather than
  selecting a concrete active-group count whose inner loops have enough trips;
- vector regional work is an output-area fraction rather than a `VectorStreamPlan` replay;
- a direct QK matmul plus an exact softmax cone can now reuse the P4 vector-stage descriptor, but
  mixed costing does not yet replay its phase-local compute and traffic;
- stage-local cube request topology is constructed only for homogeneous cube groups;
- crossing traffic is deduplicated per tensor even when a fan-out may require several reads;
- analytic `C->V->C->V` topology is retained and receives a serial stage sum, while compiler mode
  cuts it; the current unified spatial grid also cannot express its key-chunk loop.

These are explicit migration gaps, not permission for the emitter to approximate the plan.

## 7. Implementation sequence

1. **Done:** add candidate-invariant stage/transfer topology and a lightweight
   `MixedSchedulePlan`; consume it without changing canonical cost anchors.
2. **Done:** record model-versus-implementable overlap and test one-way,
   single-round-trip, multi-transfer, per-group trip count, and the current no-loop gap.
3. Replace global overlap eligibility with a choice between more serial groups and fewer
   pipelined groups. Price init, steady, tail, and drain separately.
4. Build stage-local `CubeSchedulePlan` and `VectorStreamPlan` views for mixed components and use
   their exact compute, traffic, liveness, and double-buffer decisions.
5. Add a generic mixed emitter that constructs one plan-defined inner pipeline loop and relies on
   `ExpandMixedKernel`, `InjectGMPipeBuffer`, and `SkewCrossCorePipeline` for lowering.
6. Reuse the implemented embedded P4 stage descriptor in stage-local mixed compute and traffic;
   continue rejecting any extra vector prefix or tail outside that exact cone.
7. Implement and ground one-way epilogues and single-round-trip `C->V->C` first.
8. Add whole-FIFO multi-round-trip skew, then implement full flash attention with the key-chunk
   loop and running `(m,l,O)` state.

Default mixed fusion remains off until plan/emit structural tests and 910B correctness and
wall-time validation close M1-M10.
