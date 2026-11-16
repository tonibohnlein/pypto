# Addressed accumulator row trims: operation-contract subgate

Status: design accepted in principle; implementation, independent source review
and fresh tests are still required. This is a reusable M2 metadata/export repair
within the [architecture plan](41-general-autosync-architecture-plan.md). It
follows the separate [GM scalar staging gate](44-autosync-gm-scalar-staging.md).
No accumulator, tile-size, allocation or synchronization-policy change is proposed.

## Missing interface and supported family

An already qualified static Acc producer/carry names physical storage `[M,N]`.
A later alias restricts only the rows observed by a GM writeback:

```text
acc = qualified static matmul / matmul_acc carry  # original full descriptor
trimmed = tile.set_validshape(acc, rows, N)       # same complete byte window
if rows > 0:
    tile.store(trimmed, offsets, output)           # original execution cut
```

The static writer and its loop-carried descriptors remain unchanged. No runtime
metadata is propagated back into matmul, no tiling is changed, and no new payload
operation is inserted. The trim is storage aliasing plus separately owned native
metadata, not a copy and not proof that the writer has completed.

Admit only an A2/A3 AIC addressed Acc family with the existing supported dtype,
fractal layout, physical shape, allocation capacity and native writer contract.
The initial producer/carry has established full static valid `[M,N]`; the trim
retains full columns, zero offset, the identical dtype/layout/fractal and
`CompactMode::Null`. The ready row expression must satisfy `0 <= rows <= M`
under the existing defined-operation/range contract and be available at the
original use. A native store executes only with strictly positive rows. Preserve
native field bounds, destination bounds and the original GM alias policy.

A selected consumer cannot infer `rows = M` or overwrite definedness from an
allocation size. If existing arithmetic/metadata evidence cannot justify the
needed range or readiness, keep the precise admission refusal. The design does
not add shape samples or kernel-specific assumptions.

Initially reject dynamic accumulator producers/carries, column trims, compact
packing, offset/pitch/layout changes, views or escapes of the runtime alias,
unsupported native consumers and addressless placement. Start with the existing
default two-operand TSTORE variant; atomic, quantization, activation and explicit
phase variants are not newly admitted by this metadata proof. A result that later
feeds matmul, another unknown operation, a carry, return or aggregate needs a
separate contract. Reject unqualified metadata observations anywhere in the
affected physical alias component, including the old static source, attributes
and aggregate references. Inspecting only direct users of the trim is insufficient
because the original Tile setter shares metadata. Preserve the original
Functional body on unsupported selection.

## Existing native evidence

At PTO-ISA pin `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`,
`include/pto/npu/a2a3/tstore_common.hpp::TStoreAccNz2nd` takes runtime
`validRow/validCol` and checks that they match the GM partition dimensions.
With compact mode disabled, its source stride remains `TileData::Rows`,
independent of the trimmed runtime row count. The native row/column field limits
and the writer's established physical packing still apply. With compact mode
normal, stride depends on rounded valid rows; this is why that case is excluded.
`TStore.hpp` requires positive source valid dimensions: empty stores therefore
need a guard, not an invalid native call with `validRow = 0`.

The existing native `pto.set_validshape` interface requires **both** valid
metadata dimensions dynamic. A descriptor `[-1,N]` is not an executable substitute
under that interface. Emit an independently owned alias descriptor `[-1,-1]`,
then set its actual metadata to `(rows,N)` at every use. Distinct runtime trims
with the same typed window can share that dynamic handle in the existing owner;
restore each original alias's actual extents through `RestoreDataOperands` before
every such use. Independence from the static writer is not global independence
between all runtime aliases. The runtime-valid proof
must retain that the second value is the original full width; two `-1` fields
alone do not establish equality, bounds or metadata availability.

This is qualification under the existing native instruction contract, not new
device evidence or a general runtime-layout theorem.

## Implementation owners and validity argument

| Owner | Bounded change |
| ----- | -------------- |
| Shared matrix/metadata recipe | Identify static full Acc writer plus row-only store alias; distinguish physical storage from valid metadata |
| `BufferValidAnalysis` and selection | Admit only the qualified addressed consumer family; preserve early refusal for escapes, dynamic carries and unsupported planners |
| `StorageIndex::FinalizeTypedRoot` | Reuse the existing placed typed-alias owner to retain distinct static writer and dynamic store metadata objects at the same certified address |
| `MatrixDescriptor` and per-use lowering | Produce dynamic-both-dimensions metadata only for qualified aliases; restore each runtime alias's `(rows,N)` before every use and preserve the static producer/carry descriptor |
| Runtime-valid verifier and codegen | Require original ready metadata, full columns, positive store guard and native encoding bounds before emission |
| Effects and synchronization | Preserve the actual store footprint/enclosure, pending M/FIX work, original occurrence identity and selected token protocol |

The storage proof establishes equal byte address, capacity and original physical
layout. It does not manufacture disjointness between the descriptors. Native
metadata objects separate the static writer from the runtime store handle.
Runtime aliases can share their handle, so per-use restoration is essential and
no independence between those aliases is presumed. Unqualified observers of the
original shared metadata are excluded throughout the affected component. The
store's row guard, full-column update and GM partition `[rows,N]` meet the selected
native contract, while its static physical source pitch
still addresses the established accumulator contents.

Memory effects remain real: source completion, store issue/completion, physical
reuse and invocation drain retain their existing owners. A metadata setter
completes no payload. Exact selectors remain available only for their declared
storage domain; conservative effects are not upgraded to exact physical exports.
The adapter changes no allocation algorithm, ID palette, hardware exemption,
expert protocol, default dispatch or zero-inner-body barrier.

## Acceptance gate

- Generic addressed GEMM/Acc fixtures, independent of Qwen names and tilings:
  zero, one, partial and full rows; several static physical shapes and repeats.
- Independently check original store addresses/values and its original guard;
  preserve static matmul initialization, carry and counter-reset identities.
- Use the old static alias before and after the trimmed store and alternate
  multiple runtime row aliases to expose accidental shared metadata mutation.
- Reject columns other than full `N`, invalid/unknown row range, unready bounds,
  compact packing, unknown consumers, dynamic carries/views, escapes and PTOAS
  addressless planning before Buffer lowering commits.
- Check both alias policies, required-order coverage, actual logical/physical
  order, matching, acyclicity and consumption before event republication.
- Produce real PTO/C++ for qualified addressed planners; verify the store
  partition uses `(rows,N)` and both native dynamic metadata fields. Keep the
  static writer descriptor unchanged and preserve the final invocation drain.
- Re-run compact-K/replay, local scalar, runtime-valid and zero-inner-body gates,
  then refresh the existing pinned Qwen census without changing its source.

Representation and construction work must follow static aliases/uses and their
shared metadata DAG, without unfolding runtime rows or visits. Success is a
qualified metadata bridge; it is not a new minimum-demand theorem or a device
speedup claim.
