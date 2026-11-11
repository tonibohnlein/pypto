# Supplied numerical-order service checkpoint

This is preparation for nested token transport, following `f0ad5a36a`. It does
not enable nested `LowerSync`, change production dispatch, or add an allocator.

## Ownership and validity

`BuildNumericalMergeOrder` receives exact child event-order rows, selected original
parent ports, and explicit crossings. It does not inspect accesses, propose storage
conflicts, or reduce demands. Native links between child chains are supplied by
the caller. All later join endpoints must remain in the exported interface.

For a reference-forward path across contiguous children, its last left-child
event and first right-child event identify a supplied crossing. Child forward
and reverse rows summarize the parts on each side. Propagating these rows through
the crossings therefore computes the same order as the supplied graph. Separate
start and completion chains preserve their different meanings.

`QueryNumericalOrder` uses a synchronous leaf-query callback and an inspected-work
counter. The callback must agree with the leaf rows and be reflexive and monotone
on the supplied chains. It does not borrow a mathematical analysis through an
invented result wrapper. Original interior events remain queryable after parent
port projection through the retained hierarchy.

The existing storage-analysis caller still proposes and reduces crossings. It
passes the reduced inventory to the shared service, preserving its compact query
index and mathematical result. Future token callers must instead supply actual
token edges and keep every handoff identity separately from the order graph.

## Cost and validation

The indexed merge kernel costs `O(kP + k²E)`, where P includes inspected child and
parent ports, including discarded ports. Ordered map/set indexing adds
`O(P log P)`; constructing the supplied interface is a separate cost. Leaf query
cost and hierarchy traversal remain explicit. This is not an end-to-end linear
compiler claim.

The native regression independently flattens supplied graphs and computes their
transitive closure. It checks 2,000 generated cases and 1,396,352 original-event
queries, all four event-kind combinations, projected rows, and multilevel merges.
No compiler storage partition or demand generator supplies the oracle edges.
The production regression gate separately exercises storage reduction, selectors,
scoped physical exports, Qwen fixtures and existing token paths.

The broad gate passed 1,119 tests (28 existing warnings). After formatting and
rebuilding, the focused gate passed 96 tests (two existing warnings), and the
native oracle passed again. All configured repository hooks passed.
Architecture, correctness and algorithm reviewers accepted this preparation.
The [receipt](token-pipeline-m3a.json) records exact build/test outcomes and hashes.

```bash
cmake --build build-py314 --parallel 1
build-py314/tests/ut/cpp/pypto_numerical_sync_order_test
```

## Remaining milestone work

Nested insertion needs frame-owned completed initialization, simultaneous carries,
source captures and exact target guards. Nested lowering must derive its families
and reuse proofs from those actual tokens. Affine endpoint coordinates alone do
not prove uniform order: a startup-only path cannot justify every later reuse.
Caller range/block evidence and production token orchestration also remain pending.
Existing direct production coverage remains available; no device claim is made.
