# RealizeSync (experimental)

This optional pass tries PyPTO synchronization per whole kernel after storage
lowering. **PTOAS autosync remains enabled in both modes.** Default compilation
continues to delegate synchronization to PTOAS.

```python
from pypto import passes

# Explicit caller promise: distinct GM arguments denote disjoint allocations.
with passes.PassContext([], enable_pypto_autosync=True,
                        autosync_disjoint_gm=True):
    compiled = kernel.compile(a, b, c, config=config)

# Turn PyPTO's pass off; PTOAS still handles synchronization.
with passes.PassContext([], enable_pypto_autosync=False):
    baseline = kernel.compile(a, b, c, config=config)
```

PyPTO runs the shared `InsertSync` producer, logical verification (the current
`OptimizeSync` stage), and actual-token `LowerSync` consumer for physically placed Buffer IR on the
qualified AIC matmul and AIV Vec fragments, using manual a2a3 macro variants. Missing placement,
GM disjointness, target, access, control, endpoint or resource qualifications
leave the **original kernel body** for PTOAS. This delegation does not promise
that PTOAS supports every input; its failures still propagate. Internal compiler
errors are not converted into fallback. Verification of insertion-owned logical output preserves this boundary:
malformed generated tokens raise an internal compiler error. Standalone malformed
user tokens continue to report validation errors. The retained-analysis research APIs
remain strict and report qualification failures directly. The former direct
mathematics-to-physical APIs remain explicit audit tools; production never uses
them to bypass a token-export refusal. Preexisting logical synchronization and
already-owned physical plans are rejected before any eligibility/delegation
exit. Use standalone `LowerSync` for existing token IR.

The transaction collects caller evidence once. Logical insertion preserves the
original call graph, signatures, scalar definitions and payload control, so its
own lowering can reuse that evidence. Standalone `LowerSync` re-establishes it
from its current Program, including serialized or edited tokens. A later failure
publishes no intermediate token IR. Empty or barrier-only candidates retain the
existing whole-kernel delegation policy.

A successful attempt publishes its complete body and the boolean
`pypto_autosync=True` attribute. PTOAS skips insertion independently for each
function containing explicit SET/WAIT or record/wait events. Thus a module can
contain both PyPTO-synchronized and delegated functions. **Barriers alone do not
trigger this skip.** Empty or barrier-only PyPTO candidates are discarded and the
original body is delegated; the pass never manufactures a dummy flag. The pass
never publishes partial synchronization on a failed attempt. Already present
user synchronization remains governed by PTOAS's existing skip rule, rather
than being completed or independently validated by this switch.

For an opted-in compilation, `report/autosync.json` lists each attempted kernel's
selected route (`pypto` or `ptoas`) and the fallback reason. This is selection
metadata, not a certificate that PTOAS inserted commands or that compilation
succeeded. External-source, builtin-template and runtime-bound functions retain
their external ownership and are not included in this report. Do not edit owned
bodies or forge the attribute: like physical addresses, it is trusted compiler
output, not a proof checker for arbitrary user edits. Reapplying the pass to an
owned plan is refused.

The toggle and disjointness contract are part of JIT cache identity. Dumping and
profiling preserve them. A `PassManager` cannot change either after construction.
Allocation algorithms are unchanged. This interface does not expose an unsafe
synchronization-free execution mode.

The qualified [physical construction](../ir/15-physical-demand-realization.md)
establishes modeled payload order, matching and legal directional reuse. It
assumes fresh invocation entry and closes notifications; it does **not** prove
that returning from the function has completed its last asynchronous payload.
Native macro semantics and runtime invocation completion require separate
device qualification.

## Finite Vec coverage

The AIV path reuses the same demand analysis, placement and flag allocator. It
admits full-valid static rank-two Vec loads/stores and elementwise operations
from the shared Buffer recipe catalogue. Constant scalar operands are ready;
arbitrary scalar-producing computations are not admitted. The shared Buffer
verifier enforces each instruction's in-place alias policy.

The same path admits the shared `row_max`, `row_sum`, `row_expand_sub` and
`row_expand_div` recipes, including full-valid contiguous column vectors.
Reductions read the source and write the destination plus an enclosure covering
the whole declared scratch buffer; broadcasts read both inputs. Minimum demands
and selectors are relative to these modeled effects, not a claim of exact native
scratch-byte recovery. The existing verifier rejects illegal scratch overlap,
and the resource catalogue reserves native V/S event IDs before allocation.
General column-major matrices and partial-valid descriptors remain unavailable.

The original `examples/intermediate/02_softmax.py` now selects PyPTO autosync and
generates C++ through PTOAS. Host checks cover its modeled accesses, covers,
queries, selectors, matching and physical reuse, plus repeated small pipelines
with scratch reuse and counter resets. Retained sizes remain constant as the
tested loop count grows from 4 to 1,000,000. The original RMSNorm and LayerNorm
examples still delegate: their complete bodies are not selected for Buffer IR.
This is an earlier lowering boundary, not an autosync allocation failure. These
new cases have no device qualification.

Placed Vec allocations and static subviews share a byte-address domain. Nested
subviews preserve the original row pitch; dense reshapes preserve their byte
extent. Overlapping allocations remain overlapping even with different SSA
names or element widths. Dense accesses require one interval; strided windows
require one interval per physical row. This cost depends on static window size,
not loop trip count. For stationary column intervals in one row band, an ordered
index skips unrelated storage atoms. `class_tests` counts membership checks;
`class_lookup_tests` separately counts binary-search comparisons. The shared
analysis still enforces its existing per-visit membership and refresh premises
before using a compact repeated result.

Dynamic valid-state updates, runtime view offsets, rotating slot selections and
noncontiguous reshapes lack the required importer/readiness certificates and
remain delegated. GM view/stride support remains subject to the existing
importer and emitter contracts: this extension does not turn a pitched GM view
into a packed one or infer alias guarantees from SSA identity. Static full-valid
local views do not establish general dynamic metadata support.

A same-MTE2 demand can have valid minimum-demand analysis but no qualified local
barrier mechanism. That is an executable-export refusal, not an empty demand.
Native fixed event owners are reported separately as `native_reserved_ids` in
the research physical result and excluded from the appropriate AIC/AIV pools.
The allocator and eligible pool sizes are unchanged.

Host regressions cover original add/multiply examples, overlapping allocations,
nested pitches, in-place operations, zero and repeated visits, byte access
agreement, actual covers, four-event queries, selectors and causal flag reuse.
PTOAS C++ generation is a separate check; no new device qualification is implied.

## Native accumulator protection and scalar barriers

For the named manual `a2a3` target contract, consecutive Mmad operations on the
same physical L0C accumulator may omit their storage synchronization when
`floor(M/16) * floor(N/16) >= 10`. The importer checks effective full-valid shapes,
compatible Left/Right/Acc operands and dimensions in `[1, 4095]`. This implements
the [Mmad synchronization optimization](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html).
The rule is not a blanket matrix-pipe or FIX-pipe exemption.

Physical accesses and boundary selectors remain unchanged. A provider certificate
filters only protected storage witnesses before finite reduction and crossing
reduction; explicit prerequisites are never filtered. Sparse counters establish
that every intervening matrix operation qualifies on the same accumulator and no
other access touches it. Counts include original loop multiplicities and counter
resets without unfolding visits. A FIX read breaks protection. Reinitialization
never consumes incoming protection; identical placed aliases share the same
physical identity. The rule adds no completion-before-start event edge. The ordinary frontier scan
remains valid: native start order carries incoming obligations forward along a
protected writer chain, and native completion order carries its outgoing
obligations to later consumers. Read-only WAR hazards remain explicit.

First-matrix initialization followed by accumulation is supported. A conditional
initializer after another matrix site keeps conservative treatment: its
startup-only internal demands need a richer family activity contract. Unknown
targets receive no exemption, and physical lowering rejects a target different
from the retained analysis contract. Minimum-demand and order-equality statements
refer to this selected hardware/effect contract, not every possible native
optimization.

`PIPE_S` barriers are forbidden. Physical qualification and emission admit local
barriers only on M or V. The numerical service can consume explicitly supplied
scalar native-order edges; these do not remove cross-pipe readiness obligations.
Scalar payload import remains unavailable, so unsupported scalar programs retain
the existing delegation path instead of gaining a fabricated scalar barrier.
The [PipeBarrier specification](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0271.html)
states that scalar synchronization is automatic and a scalar barrier causes a
hardware error. Host regressions validate the graph consequences under these
contracts; they are not new device qualification.

## Indexed Buffer contracts

The opt-in importer consumes the shared `buffer.gather` and `buffer.scatter`
recipes for full-valid, placed, dense AIV buffers. Indices are valid flattened
physical element offsets into the supplied local buffer; this is the operation's
caller contract, not a statically proved bound or support for arbitrary pointers.
Gather reads use a whole-source enclosure and scratch writes use a whole-scratch
enclosure. Scatter clears its whole destination, so that write is exact under
the shared contract. Index reads and ordinary destination writes remain explicit.
The retained diagnostic labels each access `exact` or `enclosure`; covers and
boundary selectors refer to this selected model. Conservative covers need not
be the concrete execution's minimum covers, but their closure covers its conflicts.

The existing native-resource catalogue reserves manual-mode V/S helper flags;
allocation and hybrid PTOAS delegation are unchanged. Missing placement, partial
valid extents or illegal writable aliases still refuse transactionally. Row-major
padded row-reduction output from PR #3333 remains outside this full-valid Buffer
contract; integration of that API does not silently admit its padding as data.

## Ready invocation coordinates

The pass gathers bounded integer coordinates from the compiled program's direct
Call/Submit sites before analyzing kernels. It joins all sites, preserves original
formal positions, and leaves unknown bounds unproved. The importer admits a common
parameter translation of each GM domain with checked arithmetic and allocation
bounds; numerical analysis is retained without specializing to one tile offset.
Storage queries require the corresponding original coordinate bindings.
See the [parameterized prefill contract and checkpoint](../ir/16-parameterized-prefill-checkpoint.md).
Explicit bounded INDEX formals may name a dynamic leading GM extent; accesses
must fit its smallest admitted size and selectors require its actual value.
Original-width arithmetic and casts need nonnegative, value-preserving proofs.

Perfect constant loop nests with invariant effects retain their original frames
and use a checked mixed-radix occurrence ordinal for physical commands. This
specialization does not cover moving inner slices or accumulator-reset branches.
The qualified nested projection construction below extends this baseline when
caller, access and scoped endpoint evidence is available. Arbitrary symbolic
nesting remains unavailable; qualified nested token transport is described in
the [logical pipeline](60-logical_sync_pipeline.md).

## Parameterized nested projections

The configured Qwen output and combined gate/up projections use the same
opt-in pass. Closed-program caller and launch evidence qualify ready invariant
coordinates; numerical regional composition preserves resident operands,
accumulator initialization and counter resets. Every activation case must
qualify before the original payload body receives guarded command cuts.
Six-ID allocation uses the existing sufficient certificate. Unsupported
coordinates, same-pipe mechanisms or resource evidence still delegate.
See the [pipeline checkpoint](../ir/18-qwen-prefill-pipeline.md) for exact
premises, independent bounded checks and remaining selection limitations.
This does not qualify arbitrary symbolic nesting, the full prefill model,
native invocation completion or device behavior.
