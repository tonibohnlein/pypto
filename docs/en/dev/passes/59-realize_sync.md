# RealizeSync (experimental)

This optional pass tries PyPTO synchronization after storage lowering. Default
compilation continues to delegate synchronization to PTOAS.

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
publishes no intermediate token IR.

Caller-range recovery retains qualified SSA definitions and scoped integer
comparisons through the existing arithmetic analyzer. Correlated expressions
can therefore establish a numeric bound without losing their shared operands.
Every original intermediate must first pass its machine-width checks; algebraic
cancellation cannot repair an earlier overflow. Branch-local evidence does not
escape its arm, and an unqualified caller invalidates a joined parameter bound.
This exports numerical intervals, not a general relational shape contract.

A successful attempt publishes its complete body and the boolean
`pypto_autosync=True` attribute. With per-function ownership, the compiler runs
PTOAS with insertion enabled. PTOAS skips each function containing explicit
SET/WAIT or record/wait events and can still insert synchronization in other
functions in the same module. A candidate with only barriers or no commands
delegates the original body because PTOAS requires a flag event for this skip.
Existing user synchronization on this route follows the same PTOAS rule.

With `autosync_policy="conservative_prefix"` and PyPTO-placed storage on A2/A3,
PyPTO can also plan a qualified AIC/AIV peer unit together. The managed GM FIFO
route retains the native transport and verifies local readiness and release
ordering in Buffer IR or through qualified Functional IR borrows. The authored
GM route requires both endpoints in Buffer IR. It preserves the original
communication and qualifies local synchronization and shared-GM visibility.
A refusal delegates both endpoints to the shared PTOAS invocation. The exact
policy uses the per-function route. Standalone passes also expose
`require_whole_module=True` to require ownership of all local Buffer domains.

For whole-unit ownership, the compiler checks that every emitted domain,
including linked peers, has complete ownership and that there are no generated
helpers. Qualified empty and barrier-only plans can participate. The backend
validates `ptoas/<unit>.sync.json` against the emitted PTO before compiling the
unit with `--enable-insert-sync=false`. Keep the PTO and its manifest together
for an ownership-aware rebuild. Editing the emitted PTO requires a fresh IR
compile to establish ownership again.

For an opted-in compilation, `report/autosync.json` records each attempted
kernel's selected planning route (`pypto` or `ptoas`) and reason. The unit
manifest records the downstream insertion policy. External-source,
builtin-template and runtime-bound functions retain external ownership and
are omitted from the planning report. Preserve owned IR bodies and their
compiler-generated attributes. Reapplying the pass to an owned plan is refused.

The toggle and disjointness contract are part of JIT cache identity. Dumping and
profiling preserve them. A `PassManager` cannot change either after construction.
Allocation algorithms are unchanged. This interface does not expose an unsafe
synchronization-free execution mode.

The qualified [physical construction](../ir/15-physical-demand-realization.md)
establishes modeled payload order, matching and legal directional reuse. It
assumes fresh invocation entry and closes notifications. Public `RealizeSync`
and `LowerSync` additionally emit an explicit A2/A3 `system.bar_all` before
return when taking whole-kernel ownership. This replaces the terminal completion
that would otherwise be supplied by PTOAS; notification closure alone is not
sufficient. Unsupported exits refuse ownership transactionally. Internal child
lowering still permits pending payloads and does not insert a child-exit drain.
Native instruction and runtime behavior remain separate device qualifications;
performance comparisons must use the same invocation-boundary policy.

## Explicit conservative-prefix policy

`autosync_policy="exact"` remains the default. A separate explicit request can
use qualified operation effects without exact occurrence counts or GM selectors:

```python
with passes.PassContext([], enable_pypto_autosync=True,
                        autosync_policy="conservative_prefix"):
    compiled = kernel.compile(a, b, c, config=config)
```

This route uses whole-memory-space read/write enclosures: distinct GM roots may
overlap, and local slots/views share their physical memory-space class. Original
access-validity preconditions still apply. Branches union may-access facts;
loops retain zero-trip and backedge facts without unfolding trip counts. No
possible write kills older obligations. Unsupported effects, scalar-memory
readiness and compound transport still refuse transactionally.

A consumer normally waits for the current prefix of the source pipe. This can
delay storage reuse if an unrelated operation on that pipe follows the last
access to the storage. In a single sequential loop, the conservative planner
can capture completion immediately after that last access and issue the wait
there on an idle destination pipe. Placement uses the physical read/write
domains and control paths of the operations. The planner reruns its fixed-point
check over loop entry, branches and backedges, and checks that the handoff
replaces the original wait without adding another prerequisite service. When
these checks fail it keeps the original placement. Physical lowering allocates
events and supplies a return acknowledgment when reuse requires one.

For qualified imported peer endpoints, a record can precede a contiguous
suffix of independent source-pipe operations while its wait stays with the
consuming operations. The debt check restores any later service that still
needs the displaced work. A qualified first loop visit can acquire a preheader
prefix under a guard built from already available values. The replay excludes
the displaced effects from that first acquire and keeps the original service
on later visits. Authored peer commands retain their original cuts, and
operations with native event claims end the candidate suffix. The original
schedule remains available if event allocation cannot realize the split.

After allocating an imported endpoint, the planner can place a local SET
on each arm of the preceding branch or at the preceding loop latch. A
branch-arm SET uses the receiving branch's predicate. A unit-stride latch
publishes only when the next iteration will execute its guarded receive.
Both forms preserve the WAIT cut and physical event ID. The planner checks
the crossed commands for event reuse and rechecks the fixed services with
the displaced payload effects excluded from the earlier prefix. If these
checks or Buffer validity fail, the endpoint uses its original allocation.

A storage overwrite can also use an earlier release from the pipe that last
read its slots. When a single wait protects several slots, the planner places
the record after the last reader contributing to that shared wait. It checks
the completion facts available at the record against subsequent requirements
on the original control flow, including event reuse. It keeps the candidate
only when it can realize the release with the existing flag budget and wait
placement.

A shared wait can order independent operands. For example, the left matrix
move needs the earlier query load while the right move needs the later page
load. The planner can keep the original page publication, deliver the earlier
prefix to the first move on a free directional event, and postpone the page
wait until the second move. It checks physical byte overlap, native
reservations, intervening commands and the reverse handoff that orders
consumption before the event's next publication. In a longer run of consumers
the conservative dataflow can also nominate an earlier capture and restore
the remaining prerequisites when a later consumer first needs them.

For a verified work loop on one core, the compiler can simplify synchronization
conditions in a small child loop with constant bounds. It evaluates each
eligible condition at its original iteration coordinates and can split the loop
into two successive ranges or group two successive iterations. A candidate
must save control work and fit the same per-function growth limits used for
local scheduling. The compiler charges the extra payload sites and the whole
replacement control tree. Payloads,
local commands and authored FIFO operations keep their execution order.

At a single cut in a structured endpoint, several generated SET/WAIT pairs
can deliver completion to the same local pipe. For example, a third pipe
that has already waited for two producers can deliver both completions in
one pair. The planner compares the original batch with a smaller subset
of its pairs and with a single relay. It accepts a saving only if every
queue sees the same payload completions at subsequent issues and at fences
that observe completed commands. It also checks that earlier event
consumptions remain ordered before reuse. A new relay needs an available
directional ID. Inside a loop, retained commands must carry its consumption
back to the publishing pipe before the next visit. These checks use the
original control flow and introduce no guards. If the relay would also
deliver unrelated work, the original batch remains eligible.

Two successive retained child loops can transfer completion through an
existing event. For a bounded invocation with known control flow, the
planner replays the generated pairs and local barriers at the second
child's entry against the complete physical schedule. It can drop a pair
or barrier when every payload still sees the same completed prefix on each
pipe at issue, every event is consumed before reuse, and all events are
consumed at exit. An unknown branch, an opaque native effect, or a replay
that exceeds the work budget keeps the original handoff.

A loop head can reuse storage first touched in the prologue. The first
overwrite may need a pre-loop reader to finish, while later overwrites need
the previous iteration’s reader to finish. The planner separates the first
visit from the backedge fixed point for the unconditional first payload of a
qualified innermost sequential counted loop. This includes loops nested in
branches or other loops. For example, a reduction can acquire the previous
row’s store completion at entry and carry it across iterations as long as
no iteration needs another completion from the store pipe. For nested loops
and functions with multiple loops, the planner hoists only entry-only
services from pipes that issue no payload in the loop. It keeps every other
service at its original cut so it can still establish event reuse. A single
top-level loop can additionally guard backedge-only services to execute after
the first iteration. If a backedge source pipe is provably empty at that
loop's entry, its wait can execute on every visit.
The first wait then captures an empty prefix and the recurring event pair can
acknowledge reuse without a separate return event. Both analyses use
physical storage overlaps, including aliases. A conditional access can
therefore renew a debt on a later visit. The generated loop keeps its original
bounds and payload body.

The same invocation drain and whole-kernel ownership rules apply. This policy
does not silently replace successful exact mathematics after an export failure.
Its identity is retained by context cloning, JIT caches and route reports.

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
not loop trip count. Partitioning sweeps write-rectangle row events, splitting
columns only at active writer boundaries, while retaining holes and read-only
domains. A shared row/column index finds classes intersecting the full affine
access envelope; the exact membership proof still checks each candidate.
For W writes and C output cells, partition construction is `O(W log W + C)`;
general overlapping rectangles can still produce quadratic C. `class_tests` counts membership checks;
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
See the [parameterized prefill contract and checkpoint](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir/16-parameterized-prefill-checkpoint.md).
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
When launch cases differ only in the trip count of an original symbolic-start
loop, lowering can use first, last, previous-visit or next-visit predicates
if they select the same physical commands in the same order in every case.
Six-ID allocation uses the existing sufficient certificate. Unsupported
coordinates, same-pipe mechanisms or resource evidence still delegate.

Physical lowering can split a qualified innermost loop into up to three
constant-bound phases at assigned endpoint boundaries. Loops with fewer than
four trips are eligible only when every phase is a single visit. Each phase
emits the original visits and the same allocated commands at their original
payload cuts. With the residual enclosing-loop guards accounted for, the upper bound
on the specialized loop's control work, including its loop headers, must be
below the lower bound for guarded lowering in every launch case. The compiler
charges each duplicated payload site and the full expanded replacement across
the leaf, and preserves guarded emission when the growth budget does not fit.
This selection does not expand invocation cases or period residues.

Between two adjacent payloads, the scheduler can evaluate a shared slot or loop condition once
for both synchronization decisions. It may copy one intervening local synchronization command
into the two mutually exclusive arms. Each path executes that command in its original position
and preserves every payload and event ID.

See the [pipeline checkpoint](https://github.com/tonibohnlein/pypto/blob/365b803a45072f1c00e8eca5da9942bd9828685f/docs/en/dev/ir/18-qwen-prefill-pipeline.md) for exact
premises, independent bounded checks and remaining selection limitations.
This does not qualify arbitrary symbolic nesting, the full prefill model,
native invocation completion or device behavior.
