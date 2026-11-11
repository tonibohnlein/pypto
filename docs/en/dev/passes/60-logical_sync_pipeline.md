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
use occurrence captures as described below. Nested exports remain unavailable. Unsupported inputs raise; standalone insertion does not delegate.
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
constant-lag loop form described below, over admitted placed Buffer payloads. The operation/footprint importer is reused for
pipe and native-resource qualification; its admission restrictions and access-
recovery costs still apply. Standalone lowering does not yet transport caller
parameter or block-bound evidence. It interprets actual `sync.record`, `sync.completed`,
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
nested statement sequences are gathered once. Access recovery and native-variant
qualification still run through the existing importer and are charged separately.

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

Nested token frames, arbitrary structured joins, non-completed carry entry,
cyclic histories, live older-result crossings, noncanonical guards and unavailable caller range evidence
remain explicit refusals. Existing production coverage is unchanged.

## Migration and guarantees

`realize_sync()` retains the existing production implementation until all
currently accepted compact cases have migrated and passed review. The intended
final role is a transactional orchestrator for these three stages, with the
same opt-in switch and whole-kernel PTOAS delegation.

No successful logical or physical result proves native invocation completion.
The existing fresh-entry and native macro assumptions still apply. Device
qualification and the runtime return-boundary contract remain separate.
