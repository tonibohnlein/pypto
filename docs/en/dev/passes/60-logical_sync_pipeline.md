# Logical synchronization stages (experimental)

The standalone `insert_sync`, `optimize_sync` and `lower_sync` passes separate
minimum-demand analysis from logical completion IR and physical event ownership.
They run after physical placement and Buffer lowering. They do not change the
default pipeline or enable experimental autosync automatically.

```python
from pypto import passes

# This is an explicit caller promise about distinct GM argument allocations.
with passes.PassContext([], autosync_disjoint_gm=True):
    logical = passes.insert_sync()(placed_program)
    logical = passes.optimize_sync()(logical)
    physical = passes.lower_sync()(logical)
```

## InsertSync

Insertion uses the existing placed-access importer, retained minimum-demand
analysis and original-coordinate logical exporter. It creates non-consuming
completion tokens, with records after producers and waits before consumers.
Its exporter supports qualified repeated bodies and single-region rotating
slot streams using scalar token histories. Finite numerical crossing exports
use occurrence captures as described below. Qualified constant nested frames
also support scoped logical insertion and qualified physical lowering as described below.
Unsupported inputs raise; standalone insertion does not delegate.
A new insertion request must not contain existing synchronization.

Exactness is relative to the admitted access model. Scratch enclosures, native
hardware rules and the explicit GM alias contract retain their existing scope.
No physical event IDs are assigned at this stage.

## OptimizeSync

The initial implementation verifies logical token schemas, ownership and lexical
SSA uses and returns the same program. Lowering cannot depend on an optimization
having run. Well-formedness does not establish dependence coverage or physical
feasibility.

## LowerSync

Lowering supports straight-line finite token programs and the canonical
constant-lag loop form and qualified constant nested frames described below, over admitted placed Buffer payloads. Lowering uses a separate payload qualification entry point sharing
descriptor, coordinate, control, pipe and native-resource checks with insertion.
It does not construct dependence footprints, a GM alias partition, physical
interval exports or storage startup summaries. Unused scalar formals need no
range proof, and unused tensor formals need no access-model geometry. Native
layout, placement, extraction and arithmetic qualifications remain required;
coordinates actually used by those checks still require current caller evidence.
Insertion and lowering each collect caller parameter
and block-bound evidence from their current Program, through the existing closed
call/submit graph collector. Lowering never retains insertion's old proof domain. It interprets actual `sync.record`, `sync.completed`,
`sync.join` and `sync.wait` definitions and uses. A record captures the source
pipe prefix, including prior imported waits. Same-pipe joins denote conjunction;
for finite ordered records, the latest prefix subsumes earlier prefixes.

A non-consuming token may have several uses. Each physical cross-pipe handoff
has a distinct publication/consumption identity, even when publications share a
record cut. Physical IDs may be reused only with a causal consumption-before-
republication proof. The implementation shares the existing directional catalog,
native reservations and lane-coloring policy with the direct physical path.

Unsupported control flow and unproved same-pipe placement report an unavailable
export. A capacity refusal is separate. Lowering does not reanalyze storage to
replace manually supplied dependencies. Hand-authored supported programs and
serialized programs use the same token interpretation.

## Finite validity argument and costs

Within a straight-line invocation, each pipe's waits form an ordered chain.
For each modeled event, a fixed-size row stores the largest wait rank on every
pipe that can reach that event. Start and completion rows remain separate;
records join the completion row with prior imported waits. A wait joins the
recorded row with its destination's preceding start and wait state, then adds
its own destination rank. Induction over the statement list establishes these
rows without an all-pairs graph. Records on one pipe have monotone prefixes, so
a join of such records is exactly their latest prefix.

Each nonempty cross-pipe use receives its own SET/WAIT pair at the actual record
and wait cuts. Sharing a physical ID is admitted only when the previous wait's
rank appears in the later publication's row. For a fixed direction these legal
reuse positions form a suffix of the ordered publications. This supplies the
premises of the existing lifetime-chain coloring sweep. Its finite minimum
claim concerns this selected per-use command plan, not all possible realizations
of a non-consuming token. A same-pipe barrier additionally requires that its
whole completed prefix equals the token prefix; older prefixes refuse.

New token interpretation costs are proportional to distinct expression nodes,
statement occurrences and token uses, with a fixed number of pipes. Indexed
lookups and coloring add logarithmic factors. Shared joins are memoized and
nested statement sequences are gathered once. Shared descriptor/control and
native-variant qualification costs are charged separately. Dependence access
recovery belongs to insertion, not token lowering.

## Canonical structured transport

The structured adapter accepts records immediately after an original payload
(or the whole qualified initialization/accumulation branch), and waits in the
contiguous block before their destination payload. It retains every actual use,
including repeated waits. It accepts completed-initialized loop carries that
unconditionally shift facts from that invocation's current record. For lag `d`,
the exact original `iv > start + (d - 1) * step` guard excludes the initially
empty history. Yield updates are simultaneous. A three-visit history yields
`[current, previous_1, previous_2]`, starting from three completed facts. The
resolver follows variable identities, not the ordering of carries, and rejects
cycles or histories rooted in another invocation. Only the first history output selects the final source visit; nonempty older
outputs are not accepted as final-source crossings. A crossing
into another loop must use an exact original-coordinate equality guard. Zero-trip outputs
remain completed. Starts, positive steps, types and normalized counts come from
the existing checked importer; no representative execution supplies a premise.

These checks justify translating each use to a direct completion-to-start
family. The order owner constructs periodic frontiers from those supplied
families without extracting covers. The original physical emitter consumes the
same families; same-pipe barriers still require the established placement proof.
For reuse, a proved `C_consumer(h) -> I_source(g)` yields
`WAIT(h) -> C_consumer(h) -> I_source(g) -> SET(g)` at the certified cuts.
This sufficient test can refuse feasible plans; structured lowering does not
claim a minimum physical ID count. Accumulator storage forwarding is not added
as a completion-to-start edge.

The periodic core takes `O(k(m+u) log(m+u))` work for `m` sites, `u` token-use
families and `k` pipes. It does not enumerate visits. Single-segment programs
need no boundary graph. Multiple segments share the existing explicit boundary graph (first/final ports
plus actual crossing endpoints): construction is quadratic in child port counts, and each
cross-region query takes `O(P+E)` graph work plus child queries. These costs are
separate from import, allocation and emission; compactness in trip count is not
a claim of linear total compilation cost.

### Numerical occurrence captures

A crossing may refer to an interior occurrence. Insertion retains one scalar
capture per distinct selected source occurrence. For source visit `v`, a
completed-initialized loop carry chooses the current source record under
`iv == start + v*step`, and otherwise preserves that same carry. Its final loop
result is exactly the selected completion fact. This differs from a shift
history: state is constant per capture, independent of the distance to loop exit.

Lowering verifies both pure token-yield arms, the same-carry recurrence, the
original invocation, record pipe and coordinate calculation. A capture may be
used only through its loop result; using its intermediate state remains
unavailable. A provably absent selected occurrence exports completed, not a
fabricated access-absence answer. Missing mathematical crossing export is still
unavailable. Multiple consumer uses share capture state but retain separate
physical publications and consumptions.

A point consumer uses `iv == start + w*step`. Each actual crossing endpoint is
registered as a port before the boundary graph constructs child-internal links.
Reference-forward paths across contiguous children decompose into these links
and supplied crossings/native boundaries; no path leaves and reenters a child.
Thus interior prefixes are preserved in token-order and physical-reuse queries.
The existing quadratic port-link construction is charged on the enlarged port
inventory. With B distinct captures, scalar state is `O(H+B)`; supplied exports
may themselves enumerate slots or rows. This is not a universal compact-interface
bound. The crossing adapter does not require or introduce new token operations.

Scalar history state is `H = sum(maximum active lag per source)`. Insertion
constructs `O(H)` carry/result/yield nodes and does not allocate history for a
family with no admitted uses. Indexed traversal costs
`O((N+F+H) log(N+F+H))`; interpretation is iterative and memoized. This retains
fixed-lag compactness as trip counts grow, not compactness in a binary-encoded
period. Mathematical demand/frontier storage keeps its existing representation.

After complete lowering with literal SET and WAIT sites, the pass publishes
the existing physical-ownership attribute used by Buffer codegen. This marks
ownership of the supplied logical protocol, not independent storage coverage
for handwritten tokens. Empty plans do not receive a fabricated marker. Existing
delegated slot restrictions and native multi-buffer size limits still apply.

Nested token frames without the scoped certificate below, arbitrary structured joins, non-completed carry entry,
cyclic histories, live older-result crossings, noncanonical guards and unavailable caller range evidence
remain explicit refusals. The opt-in production wrapper uses these same stages.

## Scoped nested insertion

For a retained original-region tree, insertion consumes its existing scoped
Cartesian demand families. Each endpoint retains its original frame identity and
mixed-radix coordinates. Axes must be a common enclosing prefix with exact
strides and bounds. The supported seam advances the last varying common axis by
one visit; longer lags and differing fixed common-frame coordinates without such
a seam remain unavailable.

A source fact exits each source-only inner frame through a completed-initialized
carry. A final-visit source uses an ordinary yield; an interior source uses a pure
token selection at the selected coordinate. Results are lifted outward one frame
at a time. For an adjacent enclosing seam, the outer loop carries that selected
fact into the next visit. Inner source and target counters may then reset to
different fixed coordinates. Initialization stays at the owning invocation entry.

The consumer guard is exactly its original Cartesian occurrence domain, excluding
empty entry facts. Records follow the entire qualified payload macro. Source
records may be shared as immutable completion facts; uses keep distinct identities.
The logical verifier checks SSA dominance in addition to the coordinate contract.

The construction adds `O(FD)` state and guard nodes for F supplied families and
frame depth D, plus indexed lookups and original-DAG name reservation. It does not
unfold row or trip counts. Upstream mathematical export construction is separate.
Constant original frame bounds or proved dynamic zero/singleton starts are required; other dynamic starts remain unavailable
without invalidating retained mathematics. These are logical insertion guarantees,
not by themselves a consuming-flag assignment. The scoped order certificate below
supplies a separate sufficient physical qualification.

The [scoped insertion receipt](../ir/token-pipeline-m3b.json) records 29 focused
semantic/refusal cases, a 1,148-test broad gate, and a 150-test rebuilt token gate.
A nested collective derivative retains two record sites, three wait sites and
four carried facts as either row count or visit count grows to one million.
Numerical summary work still grows with logarithmic repetition composition.
Single-run production insertion measurements are descriptive, not speedup evidence.

```bash
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_scoped_token_insertion \
  --output build-py314/token-m3b-qualification.json
```

### Interpreting scoped token uses

The diagnostic `testing.inspect_scoped_token_families` interprets the actual
scoped SSA program without a GM alias assumption. It verifies
logical SSA, strips only qualified pure token constructs, and uses current
payload qualification to identify original payloads and frames. No storage-demand reducer is
called. The returned direct-cut families have no assigned physical IDs.

A token pattern retains its source and original mixed-radix coordinates. Its
validity domain is exactly where it is non-completed. Completed initialization,
simultaneous carries and pure own-carry captures establish this invariant.
Consequently, intersecting an actual wait guard with that domain removes only
waits on completed facts. Guard conditions must describe exact rectangular
intervals in active original frames. Duplicated waits remain duplicated handoffs.
Source fanout that cannot be expressed by an injective common-prefix family is
unavailable, as are intermediate capture uses and displaced direct cuts.

The unary equation resolver is iterative and memoized. With T equations, U uses
and depth D, pattern storage/copying is `O(TD)` plus indexed lookups, and family
construction is `O(UD log(D+2))`. Payload scope checks visit D frames. Guard work
is charged per condition, including repeated use of a shared guard and copying
nested domain maps; there is no global guard-DAG-linear claim. No trip count is
unfolded. Separate importer and verifier costs still apply.

This interpretation can describe clipped intervals and longer histories than the
current insertion subset. It does **not** certify uniform repeated order or
physical reuse. Those require separate proofs before nested physical lowering
is admitted. The independent tests compare every expanded family use with
actual SSA execution, including changed captures, narrowed or removed guards,
completed entry, duplicate waits, serialization and resets. The
[interpretation receipt](../ir/token-pipeline-m3c.json) pins the tested increment.

## Scoped numerical order and reuse

For an original region tree, `LowerSync` now interprets its actual scoped tokens
and constructs an order owner. It shares `BuildNumericalMergeOrder`,
`QueryNumericalOrder` and the existing direct physical allocator/emitter. It does
not call access conflict extraction, crossing generation or minimum-demand reduction.

The owner normalizes the supplied original schedule into static leaves,
contiguous joins and named repeats. It admits a token edge to its **proof graph**
only when that edge is active throughout all repetitions of its owning region.
A body-local edge requires full common ancestor domains. An adjacent seam requires
the original last-left/first-right body coordinates and the complete seam domain;
fixed suffix coordinates remain fixed after reset. First-only, last-only and other
clipped edges remain actual handoff obligations but are omitted from this proof graph.
Thus a positive query proves actual order; a negative query means unproved, not
absence from the complete token program.

Every join retains native start/start and completion/completion links plus its
supplied token crossings. Every crossing endpoint is exported before composing.
Binary repetition translates a seam between adjacent **original bodies**, even
when the children summarize many bodies. Caches belong to immutable body/seam
owners and include count and requested relative ports. Equal-sized distinct reset
bodies cannot share incompatible summaries. No accumulator footprint exemption
is converted into a completion-before-start edge.

A repeated handoff uses one reuse query for each mixed-radix successor carry
class: lower digits move last-to-first and one digit advances. All proof edges
are uniform at their named frame; a reference-forward path inside a contiguous
repeat cannot leave it and return. Translating that representative path therefore
proves every successor in the class. A clipped handoff can use such a uniform
witness even when its own edge was excluded from the proof graph. At direct cuts,
`C_consumer(h) -> I_source(g)` establishes `WAIT(h) -> SET(g)`. Failure reports
insufficient evidence and does not establish scarcity. Same-pipe barriers require
an independent immediate native-predecessor check in the complete payload schedule.

The actual family inventory, including duplicates and exceptional uses, reaches
the unchanged allocator. No commands are removed to satisfy a proof or pool limit.
The owner survives through allocation and emission; its query callbacks borrow
that owner. Refusal preserves the immutable input. The opt-in `RealizeSync` wrapper now uses this actual-token route. Explicit
direct-realization APIs remain audit tools only.

Summary depth is logarithmic in repeat counts. B concrete external endpoints
propagate through `O(B log K)` paths; symbolic families do not expand into K ports.
The indexed merge kernel, port-set copying/comparison, source frame paths,
leaf-query construction and interface sizes remain separate costs. Counters report
distinct nonempty summary nodes and retained row entries, excluding cache aliases;
they are not a byte count for all owner/cache storage.

The bounded oracle expands the selected proof edges independently, checks all four
event queries and inclusion in the actual SSA order, and checks every accepted
reuse/barrier certificate. The physical oracle checks actual matching, causal flag
reuse, six-ID eligibility and equality to the selected token order. Serialization
and C++ generation are covered separately. The [receipt](../ir/token-pipeline-m3d.json)
records scaling to 1,048,575 rows or visits: three SET sites, three WAIT sites and
24–26 arithmetic sites, with logarithmic summary growth. Single-run timings are
descriptive. Native/device and invocation-return qualification remain separate.

Unsupported DMA same-pipe mechanisms, nonuniform dynamic original-frame bounds, general
nonrectangular control and missing caller/block evidence remain explicit limits.
Resident replay/zero-inner cases retain their required logical ordering even when
the existing physical mechanism refuses.

```bash
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_scoped_token_order \
  --output build-py314/token-m3d-qualification.json
```

## Migration and guarantees

`realize_sync()` is a transactional orchestrator of the same insertion,
verification and actual-token lowering services. The opt-in switch remains off
by default, and PTOAS remains enabled for delegated original kernels. A token
export or allocation refusal returns the original unsynchronized body; there is
no hidden direct-mathematics realization fallback. Preexisting logical tokens
and already-owned physical plans refuse before all delegation exits.

The wrapper collects caller evidence once because its own insertion preserves
callers, scalar definitions, payloads and control. Public `LowerSync` continues
to collect fresh evidence from its current input. `OptimizeSync` currently
verifies without rewriting tokens. No additional token simplification or
hardware exemption is introduced by this orchestration change.

The historical direct construction is available through explicit testing APIs.
The invocation qualification driver uses those APIs for its baseline, so it does
not accidentally compare the new production path with itself.

No successful logical or physical result proves native invocation completion.
The existing fresh-entry and native macro assumptions still apply. Device
qualification and the runtime return-boundary contract remain separate.

## Caller evidence and dynamic singleton frames

Both passes consume explicit caller ranges, including the complete logical
block-index hull. A uniform scope is admitted when the whole hull supports the same imported
representation. Block-dependent hulls straddling activation boundaries use the
analytic partition described below; other varying domains remain unavailable.

For an invocation-invariant dynamic start, the existing importer can prove
uniform zero or one visits. Scoped token insertion and lowering preserve the
original start expression and ordinal zero. The numerical placeholder start
is never used as an executable coordinate. Zero-trip bodies need no descendant
frame export; any reachable missing frame still refuses. Arbitrary guards on
a dynamic frame remain unavailable, even if a literal guard happens to hold
for one sampled start. Flat loop transport continues to require constant bounds.

Fresh evidence proves coordinate validity, not fresh dependence coverage after
arbitrary program edits. For example, changing a caller to activate a body
that was absent during insertion can invalidate its supplied token protocol.
LowerSync still interprets that protocol; it does not insert missing demands.
Transformations changing payload/control/access semantics must preserve or
re-establish synchronization before lowering. The physical ownership marker
continues to certify only the supplied protocol, including manually edited uses.

Evidence collection is charged once per pass, so the complete pipeline runs it
twice. Existing expression-cache invalidations and call-graph work are included;
there is no global one-visit-per-expression cost claim. No loop visits or caller
values are enumerated. The unchanged PyPTO-lib matmul/GEMM, dynamic zero/singleton
frames, changed caller refusals, edited uses and original coordinates have bounded
independent token/flag checks. The activation transport described below extends this uniform-hull support.

## Invocation activation cases

The existing analytic activation partition is now shared independently of demand
analysis. It provides exhaustive, disjoint block-index intervals with checked
original-type arithmetic. Insertion qualifies each interval, then merges its
source-keyed token cuts and loop transports onto one original payload tree.
Records at the same original payload macro share a completion fact; each case's
carry/capture variables remain distinct. Every wait, including same-pipe waits,
has its exact invocation guard. Records and state updates are non-consuming.

Unused case state must still be defined over the full invocation domain. Records
remain after dominating original payload macros, carries start completed, and
zero-trip results stay completed. Captures compare original constant coordinates;
a proved dynamic singleton exports its original start expression. No record is
moved inside one initialization arm. These restrictions make inactive state total
without evaluating skipped payload work or manufacturing branch history.

Lowering strips tokens only to import the original payload/native contract and
re-establish activation domains. For each domain it interprets actual SSA uses.
A comparison of the original dominating block value is accepted only when proved
true or false over the complete interval. Nonuniform edited guards refuse; no
case label, token name or insertion sidecar supplies semantics. Inactive waits
produce no physical families, while actual duplicate uses remain distinct.

Each case uses the existing allocator with its own immutable coordinate owner.
Shared cut emission guards both commands and endpoint arithmetic. Exhaustive
exclusive invocation guards permit palettes to be shared between cases, under
the unchanged fresh-entry/closed-notification native contract. No allocation
algorithm or hardware exception changes. A failing case refuses the whole result.

For C cases, N original statements and T total overlaid token state, repeated
case traversal has an `O(C(N+T))` input cost; T can itself grow with C. Frame-depth,
guard-domain processing, numerical order summaries, payload qualification and allocation
retain their separate costs. This is not a constant-cost claim for arbitrary
activation partitions. The diagnostic token inventory reports parser Read visits, waits
inspected, invocation-case-disabled waits and emitted families. Read visits exclude
the nested wait-only traversal, whose wait leaves are counted separately; skipped
zero-trip bodies and empty loop-coordinate domains are not case-disabled waits. No visit count is unfolded.

Opt-in production orchestration uses the same activation transport. Neither
route establishes native return completion or device correctness.

## Perfect nests and original frames

Token insertion and payload reimport request `SyncOccurrenceLayout::OriginalFrames`.
This selects the existing structured importer only when the original source has
multiple loop frames that the automatic path would flatten. Finite and single-loop
selection is unchanged. The public mathematical audit API keeps its automatic
default; `original_frames=True` explicitly requests the token producer's layout.
Invocation-case import propagates the same request. No second analysis is tried.

Original loop objects, nonzero starts, nonunit steps, zero visits and counter resets
remain owned by that tree. Existing access, control and startup qualifications
still apply; this does not admit arbitrary nested rotating slots or symbolic trips.
The existing numerical merge supplies covers and boundaries before logical export,
and actual-token order supplies physical reuse afterward. Numerical interface
construction can cost more than the flattened periodic form. Count sweeps retain
seven families and at most five axes in the nested-DMA regression; a singleton
carry interval may omit an axis. This is a fixture bound, not a universal linear
bound for numerical interfaces.
