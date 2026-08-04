# AutoFuse mixed cube/vector schedule contract

**Status:** the buildable `C->V` increment and first exact `C,C->V->C`
dense-SwiGLU increment are implemented behind `PYPTO_AUTOFUSE_MIXED=1`.
The explicit A2A3 dual-AIV FIFO lane bridge is function-scoped,
pipe-descriptor-aware, and byte-exact: it rewrites only the selected AIV
function, gives pipe-only AIV functions one private runtime lane parameter,
and derives each pipe endpoint's entry offset from that pipe's own tile shape
and dtype. Frontend logical IDs validate IR wiring; generated PTOAS `TPipe`
template IDs may be renumbered and are bound through preserved direction and
slot geometry rather than numeric equality.

Silicon closes the one-way C2V epilogue after generic `[M,N] + [1,N]` was fixed
to materialize `tile.col_expand_add` (50/50 sequential launches, no drift).
Dense SwiGLU is also silicon-closed at PyPTO `67df6fb6` with PTOAS v0.55.
Earlier lowering collapsed its two C2V projection channels and one V2C
activation reply into one bidirectional FIFO; the repaired protocol instead
preserves three independent logical pipes with exact byte widths, slot counts,
workspace ranges, and runtime lane offsets. PTOAS renumbers their C++ template
IDs (`0/1/2` becomes `0/2/4`), so the backend binds physical declarations by
direction and slot geometry rather than numeric identity. The production
kernel passed more than 200 launches across two 910B2 devices, including 50/50
sequential launches per device at the forced F=128 descriptor and the formerly
intermittent F=112/F=144 natural plans. An independent tagged three-pipe
primitive verifies descriptor binding structurally but has not yet run on
device. Mixed mode remains default-off pending traffic, overlap, and ranking
grounding rather than correctness repair.

Host tests also close tensor-level equivalence and the full
`ExpandMixedKernel -> SkewCrossCorePipeline -> AutoTileMatmulL0` structure.
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

- an algorithm kind; the generic stage DAG remains available, while exact
  algorithms may add topology-specific loop and carry facts;
- maximal same-engine stages and their op membership;
- every cube/vector tensor transfer, its direction, and its producer/consumer stages;
- balanced spatial partitions, active group count, and split-K factor;
- stage-local homogeneous views: cube GM-to-L1 K windows and vector
  `VectorStreamPlan` descriptors;
- pipeline axis, chunk extent, item count, per-group trips, stage count, and skew depth;
- GM FIFO tensor, direction, pipe ID, bundle ID, valid tile shape, slot bytes,
  slot count, and total reserved bytes;
- the explicit AIV row/column split and number of compute lanes;
- `model_overlap_granted` and independently derived `overlap_implementable` bits.

The dense-SwiGLU algorithm additionally records the input, intermediate, and
output extents; the feature-chunk loop; the two projection K windows; the down
feed window; and the persistent FP32 down accumulator. These are mixed
composition facts, not a second implementation of homogeneous tiling.

The winning plan is carried across the tensor-to-tile boundary as a private,
versioned FIFO descriptor. Each logical crossing is one record and one physical
unidirectional queue. `ExpandMixedKernel` validates direction, shape, bytes,
and order before assigning the record's ID to `tpush`/`tpop`/`tfree`; repeated
uses of the same activation in the mutually exclusive first/accumulate branches
share the reply ID. Distinct same-shaped sources such as gate and up never
collapse. The descriptor is stripped after expansion and cannot leak into
generated AIC/AIV/Group functions.

The last two fields intentionally fail loud during migration. A cost may use `max` only when the
emitter can construct the recorded loop and PyPTO's lowering passes can realize its skew.

## 4. Pipeline modes

| mode | topology | PyPTO lowering | cost rule |
| ---- | -------- | -------------- | --------- |
| serial | no realizable successor item | sequential loop | sum stage walls |
| one-way | `C->V` or `V->C` | sequential per-engine loops decoupled by a GM FIFO | cross-engine wavefront only with at least two equal items per active group |
| single-round-trip skew | `C->V->C` or `V->C->V` | `SkewCrossCorePipeline` | max only when its structural skew predicate succeeds |
| multi-round-trip | e.g. full `C->V->C->V` attention | future whole-FIFO wavefront | serial until that transform exists |

Today `SkewCrossCorePipeline` safely skews one ordered producer push bundle
followed by one reply pop. The bundle may contain multiple pushes, such as the
gate and up projections of SwiGLU, but every push must precede the first pop.
When the reply is branch-local, both arms must carry the identical FIFO
protocol; the first-matmul/`matmul_acc` choice is therefore one logical pop.
Any path-dependent conditional protocol is demoted to sequential.
Any push after a pop is a second round trip and is demoted to sequential to
preserve FIFO order. The solver mirrors that predicate; structural alternation
depth alone is insufficient.

`V` and `C` name maximal homogeneous stages, not individual operations or
physical cores. Consequently, a connected `C->V->V->C` source graph is a
three-stage `C->V->C` protocol: both vector operations remain in one vector
stage. The two physical AIV cores in a 910B group are represented separately
by `MixedVectorSplit` and execute spatial shards of that one logical stage.
Conversely, two independent vector branches that return two distinct tensors
to a matmul are two logical vector stages. They require a two-reply bundle,
which the current skew pass does not admit, so production AutoFuse partitions
that graph.

The candidate-invariant topology carries an explicit
`MixedCrossCoreProtocol`: `OneWay`, `SingleRoundTripBundle`, or `Unsupported`.
The bundled protocol records producer stages, the peer stage, sink stage, and
the exact transfer indices in the producer and reply bundles. The model grants
single-round-trip overlap only when this descriptor is compatible with
`SkewCrossCorePipeline`; the emitter rechecks the same descriptor before
building the mixed scope. Protocol recognition alone is not cost admission:
generic costing currently accepts one producer and one reply, while a larger
bundle requires an exact algorithm such as dense SwiGLU to provide every
stage-local cost and cross-stage lifetime.

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
homogeneous models. Mixed algorithms use the same grounded homogeneous
primitives for their stage work: cube MAC/extract/GM-to-L1 terms and vector
primitive/traffic terms are not refitted inside the mixed model. The mixed
wrapper adds only crossing traffic, FIFO capacity, the cross-engine wavefront,
and state whose lifetime crosses a stage boundary.

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
workspace. A2A3 codegen explicitly separates each AIV lane's FIFO entry using
the runtime subblock parameter because the native hardware subblock register is
not programmed by simpler MIX dispatch. The bridge is derived from split FIFO
operations, not from tensor indexing: it reuses the subblock parameter when
one already exists and otherwise adds a wrapper-private parameter to the
selected AIV PTOAS function. Frontend IDs validate endpoint wiring, while each
renumbered physical `TPipe` is matched by `(direction, slot_size, slot_num)` and
receives the descriptor's independent consumer/producer offset. Duplicate
descriptors are accepted only when their offsets are identical; ambiguous
bindings fail closed. The sibling AIC function is outside that function-scoped rewrite. The outer mixed loop deliberately is not
`ForKind::Pipeline`: a generic pipeline tag
would multiply nested AutoTileL0 buffers. The independently running AIC/AIV functions and FIFO
backpressure form the cross-engine wavefront. A complete host structural test verifies 48 logical
regions -> 24 group launches x 2 trips, one push/pop/free in each physical loop, 4096-byte slots,
and both AIV row shards. Tensor-level numeric replay matches the unfused matmul epilogue, while
transposed/NZ matmuls cleanly decline before solving. Current candidate enumeration exposes at most
two trips per group; deeper FIFO backpressure is therefore not part of the buildable surface yet.
On 910B, MemoryReuse must also keep the vector epilogue output distinct from
any loaded broadcast tile when the writer consumes a MemRef-less
`tpop_from_aic`. This backend hazard is part of the physical emit contract, and
its decision is invariant to whether a `PassContext`/IR-dump instrument is
active.

The exact dense-SwiGLU surface is:

```text
gate = matmul(x, w_gate)       C
up   = matmul(x, w_up)         C
act  = swiglu(gate, up)        V
out  = matmul(act, w_down)     C
```

The admitted activation is the exact source chain
`neg -> exp -> scalar_add(1) -> recip -> mul -> mul -> cast`, with BF16/FP16
cube inputs, FP32 gate/up results, a low-precision activation tile, and an FP32
down result. Intermediate projection values may not escape. The spatial grid
partitions final output M/N tiles among at most 24 groups. Inside each group an
intermediate-feature loop:

1. computes ordinary tensor-matmul gate and up tiles with the planned
   homogeneous cube K windows;
2. sends both tiles as one ordered two-push bundle;
3. evaluates the materialized homogeneous vector plan on the two AIV lanes;
4. returns one activation tile; and
5. initializes or accumulates the down matmul into one FP32 tile that remains
   live until the last feature chunk.

The emitter produces only tensor-level matmul/vector operations inside an
`UP_DOWN` mixed scope. `ExpandMixedKernel` creates `tpush`/`tpop`/`tfree`,
`SkewCrossCorePipeline` realizes the one-round-trip wavefront, and
`AutoTileMatmulL0` chooses all L0 M/N/K tiles and buffers. AutoFuse neither
emits raw cross-core instructions nor attaches an L0 plan. The persistent down
accumulator is the one topology-specific cube wrapper: replaying an independent
homogeneous `CubeSchedulePlan` per feature chunk would incorrectly drain it to
GM after every chunk.

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
- the current materialized pointwise subset is exact; mixed pointwise strip
  streaming and P2/P4 still require algorithm-specific state and loop admission;
- low-precision floating matmul outputs require an explicit FP32 K-window carry plus final FIXPIPE
  narrow; compiler mode declines them instead of silently rebuilding a full-K matmul;
- promoted/mixed-dtype vector operands and `INT8->INT32` epilogues remain cut until their cast or
  integer primitive semantics are represented and priced;
- the explicit A2A3 FIFO lane-offset bridge supports multiple planned IDs and
  different static tile sizes/dtypes, but dynamic or ragged transfer shapes
  still fail closed until the launch path supplies an equivalent native
  subblock ID and dynamic endpoint contract;
- a direct QK matmul plus an exact softmax cone can now reuse the P4 vector-stage descriptor, but
  mixed costing does not yet replay its phase-local compute and traffic;
- the dense stage views reuse homogeneous primitive equations but do not yet
  embed an independently enumerable full `CubeSchedulePlan` for each
  projection; doing so must preserve the feature-loop accumulator contract;
- FIFO depth eight is explicit and faithful, but smaller depths have not been device-compared and
  are not yet a scheduling dimension;
- the cross-engine wavefront and mixed launch overhead still need latest-PTOAS silicon grounding;
  host lowering proves ordering and capacity, not AIC item `k+1` overlap with AIV item `k`;
- analytic `C->V->C->V` topology is retained and receives a serial stage sum, while compiler mode
  cuts it; the current unified spatial grid also cannot express its key-chunk loop.
- the dense surface still needs latest-PTOAS 910B traffic, overlap, and ranking
  validation; its three-pipe numerical contract is silicon-closed.

These are explicit migration gaps, not permission for the emitter to approximate the plan.

Latest 910B2 isolation closes the C2V sentinel: the final AIV uses
`tile.col_expand_add`, preserves disjoint load/result allocations, and passes
50/50 sequential launches. The three-pipe dense-SwiGLU sentinel initially
stopped at code generation because its IR logical IDs `0/1/2` became PTOAS
physical template IDs `0/2/4`. Descriptor-based binding removes that downstream
ABI assumption without changing the model or expanded IR. The repaired
production kernel now passes more than 200 launches across two devices with no
NaN, hang, AICPU exception, or numerical mismatch. This closes correctness;
the flag remains off until mixed performance and overlap are grounded.

## 7. Implementation sequence

1. **Done:** add candidate-invariant stage/transfer topology and a lightweight
   `MixedSchedulePlan`; consume it without changing canonical cost anchors.
2. **Done:** remove the optimistic global-tile overlap grant; record launch groups, equal
   per-group trips, AIV split/lane count, and FIFO slots.
3. **Done (silicon-closed):** emit and fully lower the exact materialized `C->V`
   matmul epilogue; include FIFO reservation, blocking crossing traffic, exact per-lane vector work,
   broadcast multiplicity, live-out, matmul-semantic gates, and the 910B
   MemRef-less load/TPOP no-alias guard. Generic `[M,N] + [1,N]` is now required
   to materialize as `tile.col_expand_add`. Unsupported mixed topologies remain partition boundaries.
4. **Done (silicon-closed):** add the exact
   `C,C->V->C` dense-SwiGLU algorithm. Reuse homogeneous stage costs, carry the
   down accumulator across feature chunks, and extend skew to one ordered
   multi-push bundle followed by one path-invariant reply. Preserve its two C2V
   projections and V2C reply as three independently sized logical FIFOs;
   descriptor-bind their PTOAS-renumbered physical declarations before adding
   lane offsets. The production protocol passes more than 200 launches across
   two 910B2 devices; the independent tagged primitive remains structural-only.
5. Enumerate or analytically choose between more serial groups and fewer pipelined groups. Price
   dependent init, steady, tail, and drain phases separately.
6. Generalize the current stage-local homogeneous views without duplicating
   the homogeneous search. Preserve mixed-only lifetime facts such as the
   feature-loop accumulator.
7. Add the symmetric `V->C` emitter. Mirror the complete skew
   capability predicate before granting overlap.
8. Reuse the implemented embedded P4 stage descriptor in stage-local mixed compute and traffic;
   continue rejecting any extra vector prefix or tail outside that exact cone.
9. Add whole-FIFO multi-round-trip skew, then implement full flash attention with the key-chunk
   loop and running `(m,l,O)` state.

Default mixed fusion remains off until plan/emit structural tests and 910B correctness and
wall-time validation close M1-M10.
