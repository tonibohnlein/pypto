# LegalizeTileCastFragments Pass

Materializes target-specific native-cast width restrictions after physical tile
layout and storage planning are complete.

## Why this is a late pass

`LegalizeTileCast` decides whether a dtype conversion is native. A separate
backend capability, `BackendHandler::GetTcvtSafeFragmentWidth(src, dst)`, may
restrict the physical column width of that native instruction. The restriction
cannot be implemented safely before the source and destination parent pitches
are known.

The default pipeline therefore runs `LegalizeTileCastFragments` after
`InsertCommFence` and before `MaterializeValidShapeSymbols`. At that point every
affected tile has a final layout, memory space, and allocation identity.

## Rewrite

For an unsafe `tile.cast`, the pass:

1. keeps the original destination allocation and `valid_shape`;
2. computes the valid width of each fragment once, outside the row loop;
3. visits only valid rows and guards empty runtime fragments;
4. creates zero-copy `tile.slice` views of the source and destination parents;
5. writes each pair with the internal destination-passing
   `tile.cast_fragment` operation.

`tile.slice` lowers directly to `pto.subview`, so both parent row pitches are
preserved. The fragment views own no MemRef and allocate no dense temporary.
`tile.cast_fragment` lowers mechanically to exactly one `pto.tcvt`; PTO codegen
does not choose fragment widths, insert loops, or reassemble results.

On Ascend910B the restricted pairs are `INT32→FP16` and `FP16→INT8`, with a
128-element repair granule. For example, a physical width of 224 is issued as
`128 + 96` per valid row. A padded physical frame such as 256 with valid width
129 remains pitch-preserving and issues valid widths `128 + 1`.
Complete unpadded frames no wider than 128, or composed entirely of aligned
128-element fragments, retain the original single `tile.cast`.

Storage outside `valid_shape` is unspecified. Runtime tests compare only the
logical rectangle and exercise residual widths on both sides of the 128-element
boundary.

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeTileCastFragments()` | `passes.legalize_tile_cast_fragments()` |

`tile.cast_fragment` is compiler-internal. User programs continue to write
`pl.cast` / `pl.tile.cast`.

## See Also

- [17-legalize_tile_cast.md](17-legalize_tile_cast.md) — dtype-pair legalization
- [54-materialize_valid_shape_symbols.md](54-materialize_valid_shape_symbols.md) — the next default pass
- [00-pass_manager.md](00-pass_manager.md) — default ordering and pass properties
