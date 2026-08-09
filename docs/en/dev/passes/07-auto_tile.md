# AutoTile

`AutoTile` turns one explicitly marked tensor function into one complete
Ascend 910B homogeneous schedule. The user fixes the operation graph; the pass
chooses a Vector schedule or a Cube schedule, including the core grid, tile or
stream shape, on-chip lifetimes, implementation phases, and stores.

This is deliberately different from graph fusion. `AutoTile` never partitions
the marked operation DAG and never falls back to independently planned
subgraphs. A schedule may contain multiple ordered implementation phases, such
as the two AIC phases of split-K, but they still implement one admitted graph
under one plan. A marked function either has one exact, capacity-safe schedule
or compilation fails. Unmarked functions are unchanged.

## Placement and API

The default strategy runs `AutoTile` immediately after
[`FlattenCallExpr`](06-flatten_call_expr.md) and before hierarchy or InCore
outlining. At that point calls are in three-address form and the complete tensor
DAG is still visible.

Mark a function with `auto_tile`:

```python
import pypto.language as pl


@pl.program
class Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self, x: pl.Tensor[[128, 8192], pl.FP32]
    ) -> pl.Tensor[[128, 8192], pl.FP32]:
        shifted: pl.Tensor[[128, 8192], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(shifted, 2.0)
        return out
```

The pass is also available directly for custom pipelines. Direct use requires
the same tensor-level preparation prefix as the default strategy:

```python
from pypto import passes

prepared = passes.convert_to_ssa()(program)
prepared = passes.simplify()(prepared)
prepared = passes.normalize_stmt_structure()(prepared)
prepared = passes.flatten_call_expr()(prepared)
result = passes.auto_tile()(prepared)
```

An absent or false marker is a no-op. A successful rewrite consumes the marker,
so running the pass again is also a no-op.

## Admission contract

The initial implementation supports the following closed surface:

- an explicitly configured Ascend 910B backend and a marked tensor-level
  `FunctionType::Opaque` function before scope outlining;
- a straight-line, topologically ordered SSA tensor DAG with one top-level
  return and positive static rank-2 shapes;
- FP32 and FP16 vector computation;
- BF16 tensor storage and native cast-chain endpoints;
- a terminal, unconsumed FP32-to-INT8 cast;
- elementwise arithmetic, scalar forms, same-shape `part_*`, row/column
  broadcasts, `exp`, `log`, `abs`, `sqrt`, `rsqrt`, `recip`, FP32
  same-shape `fmod`, and FP32 `fmods`;
- `row_sum`, `row_max`, `col_sum`, and `col_max`;
- one reduction plus an elementwise producer or consumer DAG;
- multiple same-axis reductions when the complete per-core DAG fits a
  materialized schedule;
- the canonical five-operation row softmax graph;
- multiple returned pointwise values and capacity-fitting materialized
  reduction live-outs; and
- row and column reductions that fit one non-atomic kernel schedule.

The Cube surface additionally supports a straight-line DAG of non-transposed,
static rank-2 `tensor.matmul` operations. Each matmul has equal FP16, BF16, or
FP32 operand dtypes and FP16, BF16, or FP32 result storage. A single matmul may
use a uniform spatial grid, a backward-clamped ragged grid, or split-K. A
serial multi-matmul DAG uses one uniform AIC SPMD kernel and keeps its internal
handoffs in Mat/L1. Internal results must use FP16 or BF16 storage because the
Ascend 910B A2/A3 handoff narrows the FP32 accumulator before a later matmul
consumes it.

Unified binary operations are normalized to explicit row- or column-expand
operations before emission. Ambiguous `[1,1]` tensor broadcasts and a
broadcasted left operand of non-commutative subtraction or division are rejected.
Broadcast division supports FP16 and FP32; its high-precision form is not part
of this contract.

The Ascend 910B A2/A3 `TFMOD` instruction accepts FP32 equal-shape operands,
and `TFMODS` accepts FP32. AutoTile rejects FP16 fmod and tensor-fmod
broadcasts during admission. The `part_*` instruction family likewise has no
row/column-expand form, so its two tensor operands must have identical shapes.

The validated Ascend 910B A2/A3 AutoTile arithmetic surface is FP16/FP32;
PTOAS rejects direct BF16 `TADD` on this target. AutoTile therefore rejects
BF16 arithmetic during admission, before planning or PTOAS compilation. BF16
remains supported as a stored tensor and as the source or destination of a
native cast chain. AutoTile does not implicitly promote BF16 arithmetic through
FP32 because that would be a different algorithm with additional UB storage,
transfers, and modeled cost.

A native cast chain may consume a boundary or full-frame elementwise value. A
cast rooted directly in a reduction result is declined: reduction emission owns
a separately padded result box that the current emitter cannot yet widen to the
cast chain's common physical granule. Applying or broadcasting the reduction
result back into the full iteration frame before casting remains supported.

The pass rejects dynamic or non-rank-2 shapes, control flow, side-effecting
statements, `full`/shape construction, minimum/product/argument reductions,
unsupported Cube operations, mixed kernels, Welford, unsupported dtypes, and
any graph that cannot be represented by one homogeneous schedule. Cube
transpose flags, non-fractal K extents, or a plan whose streamed and persistent
L1 lifetimes exceed capacity are outside the current surface. Split-K is
currently a single-matmul schedule. Multi-matmul schedules currently require a
uniform outer grid and `split_k = 1`. These are user-facing admission or
planning errors, not requests to another planner.

## Vector planning model

The planner first constructs a typed vector graph. Tensor nodes record static
shape, dtype, boundary status, and whether the value must survive to a return.
Operation nodes record their primitive and geometry. The original SSA statement
order is the topological order; the pass does not reorder user operations.

Same-shaped values connected by elementwise operations form a physical-shape
class. For a native cast chain, the planner takes the least common multiple of
the per-dtype DMA element granularities and assigns that one element-count
granule to the complete class. Thus every native `TCVT` hop has an identical
physical shape. This does **not** allocate every value as the widest dtype:
each SSA result is charged and later allocated as
`physical_elements * sizeof(its_own_dtype)`. The emitter carries the original
logical extent as `valid_shape`, so padding never changes the program's
logical bounds.

It then enumerates balanced two-dimensional core grids. Every emitted task uses
one static maximum-size tile. Ragged partitions clamp or overlap their final
tile, so repeated edge work is idempotent and is included in modeled traffic.
Candidate task counts are selected around the hardware's 48-vector-core wave
shape.

For each candidate, the planner builds an explicit `VectorSchedulePlan`. It
contains:

- row and column partitions and the exact work-unit count;
- full-region and streamed strip or chunk extents;
- phase operation lists and boundary-input first/last-use records;
- an explicit generated-algorithm tag for phases such as the online-softmax
  update, whose work is synthesized rather than replayed from source ops;
- pipeline trip counts and depths;
- logical and DMA-padded reduction extents;
- full and emitted UB peaks plus modeled compute and transfer cycles.

The emitter validates this descriptor against the source graph before creating
IR. It does not rediscover tiling, lifetime, split, or phase choices. This
one-way plan-to-emission contract is important: a cost is meaningful only when
the emitted algorithm performs the work and owns the memory that the plan
priced.

At `INFO` log level, every successful rewrite prints one `AutoTile[name]` line
containing the selected schedule, grid, work units, tile/strip/chunk extents,
pipeline depths, UB peaks, phase traffic, modeled cycles, whether the
reduction estimate is grounded or a fallback, and whether pointwise estimates
use a generic or cast proxy. The Ascend 910B coefficients are
ported without refitting from the silicon-grounded scheduler model; AutoTile
changes ownership of planning and emission, not those measurements.

## Cube planning model

Cube admission constructs a typed `CubeGraph` from the complete marked
function. Every matmul records its two producer edges, operand role, static
`M/N/K`, accumulator dtype, and storage dtype. The returned matmul is the one
sink, and every admitted matmul must be a transitive producer of that sink.
Source statement order is already topological; request expansion recursively
visits producers and emits them before consumers.

The outer planner enumerates bounded, 16-element-aligned static M/N regions and
at most two 24-core waves. A single-matmul candidate may use backward-clamped
edge regions. Such regions recompute the overlapping edge deterministically;
they do not use atomics. A multi-matmul candidate currently requires exact
uniform regions so every internal request has one statically reusable shape.

For a multi-matmul graph, the sink region is expanded backward into a
role-aware request DAG. Requesting an output region `[H,W]` from
`[M,K] @ [K,N]` requests `[H,K]` from the LHS producer and `[K,W]` from the RHS
producer. Identical producer-region requests are memoized. LHS and RHS roles
remain distinct even when they refer to the same logical tensor: `A @ A`, for
example, requires different physical requests and is not treated as one
reusable panel. The same rule applies when an internal producer feeds both
roles; its request is replayed separately for each role.

The planner applies a black-pebbling-style L1 lifetime simulation to this
request order. An internal result is allocated before its producing request,
remains live through its last consumer, and is then released. A boundary panel
with the same tensor, role, requested region, and axis binding at multiple
request sites is always loaded once and kept from first through last use. The
alternative of evicting and reloading a compatible resident boundary is not
enumerated yet. Separately, one matmul request may retain its LHS or RHS panel
across several serial output-child tiles when that removes repeated GM-to-L1
loads and the complete lifetime fits.

For every feasible outer candidate, `ChooseL0Tile` evaluates the existing
Ascend 910B L1-to-L0/Matrix/FIXPIPE model. Cube AutoTile does not copy or
replace that lower-level planner. It converges on child dimensions that the L0
chooser accepts unchanged for the initializing K window, accumulated windows,
and any serial tail. This fixed-point check is the model-to-emitter contract:
the descriptor priced by the outer planner is the descriptor replayed later by
`AutoTileMatmulL0`.

One matmul request is modeled as:

```text
request = retained_preload
        + output_children * (
              first_GM_to_L1 + first_L0_work
            + rolled_K_window_pipeline
            + serial_K_tail
            + final_FIXPIPE_drain)

serial_DAG_task = sum(requests in producer-before-consumer order)
serial_DAG_wall = ceil(spatial_work_units / 24) * serial_DAG_task
```

The first K window initializes one persistent L0C accumulator. When at least
two rolled full windows remain, their GM-to-L1 operand feed and L0/Matrix work
use a two-stage pipeline; the first window and a ragged fractal-aligned tail
remain serial. The model grants overlap only for that explicitly emitted
rolled phase. Every output child drains exactly once after its final K window.
The GM-to-L1 term uses the PTO-ISA-grounded 910B request bandwidth.

For a single matmul, split-K uses `FirstPartialThenAtomic`. One ordered AIC
phase writes K share zero non-atomically; a second AIC phase computes shares
one through `split_k - 1` and atomically adds them into the same output. This
avoids a separate AIV zero-fill kernel. The model prices the two phase wave
counts separately and exposes an explicit synchronization term, currently
zero because device evidence did not support a transferable nonzero
coefficient.

The selected `CubeSchedulePlan` records the spatial policy, static regions,
request DAG and execution order, persistent and transient L1 lifetimes,
retained panels, exact GM-to-L1 bytes, K-window phases, child L0 descriptors,
split merge policy, drain work, and component/model cycles. The emitter
validates coverage and identities before replaying that descriptor. A serial
DAG allocates each internal FP16/BF16 result with `tensor.create_l1`, assembles
the producer's accumulator into it, and passes it directly to later requests;
only the sink is stored to GM.

After tensor-to-tile conversion, `AutoTileMatmulL0` remains the sole owner of
L1-to-L0 tiling and its local software pipelines. This separation prevents the
outer model and emitter from silently choosing conflicting L0 algorithms.

Mixed AIC/AIV graphs remain outside this contract. Multi-matmul split-K,
non-uniform internal request grids, transpose variants, and an explicit
resident-versus-reload search are later Cube extensions. AutoTile rejects these
cases rather than pricing an algorithm it cannot emit.

## Schedule reports

Every successful Vector `ir.compile()` of an AutoTile function also writes two small,
deterministic artifacts:

```text
<output_dir>/report/auto_tile/<function>.json
<output_dir>/report/auto_tile/<function>.txt
```

The JSON file is a versioned compiler-artifact schema. It records the selected
grid, balanced partitions, representative region, strip/chunk loops, serial
tails, phase operation order, boundary-input lifetimes, logical and physical
tile extents, dtype-specific element sizes, UB peaks, traffic, and modeled
cycles. It is intended for tooling and is not yet a stable public API.

The text file renders the same structured descriptor as tile-centric
pseudocode. It describes one representative SPMD work unit rather than drawing
all cores. For example, an online-softmax report separates the serial first
chunk, the two-stage statistics loop, a serial ragged tail when present, and the
two-stage apply/store loop. Ping/pong slots and persistent running statistics
are explicit. A line such as `lifetime ends: x(t0)` describes the logical
last-use point; it does not imply that the IR contains an explicit free
instruction.

Cube schedules currently use the deterministic `AutoTile[name]` INFO line;
the versioned Cube JSON/pseudocode report is part of the next descriptor
extension.

Bare calls to `passes.auto_tile()` have no compilation artifact directory and
therefore keep the concise `INFO` log only. Use the schedule report together
with [IR Lowering Trace](../07-ir-lower-trace.md) to inspect the transformation
and [Memory Map](../07-memory-map.md) to inspect final UB addresses and physical
reuse.

## Schedule families

### Materialized

The complete per-core region fits in UB. Boundary tensors are sliced once per
phase, reused through their last topological use, and every returned value stays
live through its distinct `tensor.assemble` store.

General DAGs with multiple reductions, such as LayerNorm's mean followed by
variance, are materialized-only. Their reductions execute in source topological
order and the ordinary lifetime model accounts for every intervening full-frame
and thin value. If no spatial partition makes that complete live set fit in UB,
AutoTile rejects the function rather than applying the single-reduction streaming
schedule to an inexpressible dependency chain.

### Pointwise stream

An oversized pointwise region is split along one axis. The emitted strip loop is
a two-stage `ForKind::Pipeline`, allowing load/store work for one strip to
overlap vector work for another after [`LowerPipelineLoops`](29-lower_pipeline_loops.md).
All returned values are loop-carried and stored; no intermediate GM tensor is
introduced. DMA-aligned physical tiles retain their exact logical valid shape
through every generated operation, so a ragged store cannot write padded rows
or columns.

### Folded and spanning reduction

The stats phase reduces fixed-size chunks and carries a thin accumulator.
A folded schedule runs the remaining thin operations once after the reduction.
A spanning schedule makes a second chunked pass over the wide input and applies
the reduced statistic. Full chunks may use a two-stage pipeline; initialization
and the ragged tail are serial.

### Softmax

The canonical softmax schedule carries a running maximum and corrected running
sum across chunks, then makes one chunked output pass. It is numerically stable
without materializing exponentials in GM.

When the complete per-work-unit softmax live set fits in UB, the planner also
enumerates a one-pass materialized candidate. That candidate replays the source
DAG once, keeping the exponential and both reduction results on chip through
the final divide. The ordinary lifetime model prices its complete UB footprint,
one input read, one output write, and one execution of every source operation.
The online candidate remains independently costed with its statistics and apply
passes. The lower modeled cost wins; fitting in UB is a feasibility condition,
not an unconditional preference for materialization. Wide rows whose complete
live set does not fit retain the online schedule.

### Column reduction

Column reductions stream chunks of the reduced row axis while carrying one thin
column accumulator. A consumer over the original wide tensor uses a second
streaming apply phase. AutoTile does not emit a seed kernel or atomic partial
stores: if a column-reduction graph cannot be realized by one capacity-safe
kernel, the marked function is rejected.

## UB and transfer accounting

UB planning is dtype- and lifetime-aware. It accounts for boundary loads,
intermediates, all returned live-outs, DMA padding, the second bank of a
two-stage pipeline, padded row-reduction scratch tiles inserted by tensor-to-tile
lowering, high-precision `rsqrt` scratch, and thin accumulators. Metadata-only
`set_validshape` aliases do not allocate a second buffer. Column reductions
lower without a scratch tile and are priced that way.

Ordinary casts are conversions with distinct source and destination storage;
MemoryReuse must not turn `tile.cast` into an in-place operation. An explicitly
requested equal-byte `tile.reinterpret_view` remains the zero-copy opt-in and
lowers through the bitcast/view path instead of `TCVT`.

The cost model combines:

1. primitive cycle estimates and grounded Ascend 910B FP32/FP16 row/column
   reduction tables, interpolated at the emitted extent;
2. the exact primitive sequence generated for online-softmax initialization and
   updates;
3. summed logical GM-to-UB input and UB-to-GM output traffic for each phase;
4. `max(compute, transfer)` only for an emitted two-stage phase, otherwise the
   serial sum; and
5. task and wave fill terms.

Every supported candidate in the bounded reduction search is evaluated and the
minimum modeled cost wins; the largest fitting chunk is not assumed fastest.
The search considers the complete reduced extent and 16-element-aligned chunks
up to 4096 elements, subject to UB capacity. Both admitted compute dtypes use
grounded reduction tables. The implementation retains an explicit conservative
fallback for future backend expansion rather than presenting an ungrounded
estimate as measured data.

Most pointwise primitives use the transferred 910B grounding. Operations
classified as generic and native cast hops use explicit conservative proxy
coefficients: the generic proxy is not an operation-specific measurement, and
the cast proxy does not distinguish individual source/destination dtype pairs.
The plan log and schedule report expose this provenance as `pointwise_model`.

The model intentionally keeps the conservative summed directional GM term. It
does not assume independent MTE2/MTE3 overlap, fit a new bandwidth coefficient,
or infer an implicit pipeline that is absent from the IR.

## Outputs and calls

For an entry function, returned tensors become explicit `Out` parameters while
the original return tuple is retained for later normalization. For a marked
helper called elsewhere in the program, output storage is created inside the
helper so its call signature remains valid. Multiple live-outs always receive
distinct stores. Existing explicit `Out` parameters are reused positionally;
direct calls and `Submit` sites keep that declared signature.

Successful emission contains one `pl.spmd` scope and one non-split InCore body.
Later hierarchy outlining therefore produces one AIV kernel for a Vector DAG
or one AIC kernel for the supported Cube matmul.

## Relationship to other tilers

This pass owns tensor-level Vector scheduling from GM through UB and the
supported Cube matmul's outer GM-to-L1 spatial schedule. It does not replace
[`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md), which works later on Cube L0
geometry. Mixed-kernel and graph-partitioning support remain outside the
AutoTile contract.
