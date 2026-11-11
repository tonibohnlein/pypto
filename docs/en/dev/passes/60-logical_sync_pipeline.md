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
Its current exporter supports the previously qualified zero/one-distance
repeated bodies. Rotating and nested exports remain unavailable during the
migration. Unsupported inputs raise; standalone insertion does not delegate.
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

The first lowering subset is straight-line finite token programs over admitted
static placed Buffer payloads. The operation/footprint importer is reused for
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

## Migration and guarantees

`realize_sync()` retains the existing production implementation until all
currently accepted compact cases have migrated and passed review. The intended
final role is a transactional orchestrator for these three stages, with the
same opt-in switch and whole-kernel PTOAS delegation.

No successful logical or physical result proves native invocation completion.
The existing fresh-entry and native macro assumptions still apply. Device
qualification and the runtime return-boundary contract remain separate.
