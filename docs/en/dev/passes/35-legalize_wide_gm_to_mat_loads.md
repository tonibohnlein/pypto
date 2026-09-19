# LegalizeWideGmToMatLoads Pass

Rewrites GM-to-Mat loads whose source row stride exceeds the target's directly
encodable leading dimension.

## Why this is a late pass

The restriction belongs to the target, not to the DSL. The backend reports it
through `BackendHandler::GetMaxGmToMatRowStrideElements(dtype)`. The pass runs
immediately after [`InitMemRef`](34-init_memref.md), when tensor strides, tile
memory spaces, layouts, and allocation identity are known.

On Ascend910B the GM-to-L1 ND-to-NZ path accepts at most 65,535 elements in its
source row-stride field. At 65,536 the field wraps silently: each row of the Mat
operand aliases its first row. Ordinary GM loads are unaffected.

## Rewrite

For an unsafe rank-2 ND `tile.load` into Mat, the pass:

1. preserves the original destination allocation and logical `valid_shape`;
2. iterates over the logical source rows;
3. computes each row's flat source offset with unbounded IR `index` arithmetic;
4. rebases the raw GM pointer to that row;
5. presents the target with a compact `[1, C]` ND view and loads it directly
   into the corresponding destination subview.

The compiler-internal `tile.load_rebased_row` operation records this decision.
PTO codegen lowers it mechanically to `pto.addptr`, a compact
`pto.make_tensor_view`, `pto.partition_view`, and `pto.tload`. The original
parent stride is therefore used only in address arithmetic and never reaches
the bounded target field. No GM intermediate or dense Mat reassembly is added.

Targets that return `std::nullopt` from the capability query keep the original
load. Loads at or below the reported limit are also unchanged.

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeWideGmToMatLoads()` | `passes.legalize_wide_gm_to_mat_loads()` |

`tile.load_rebased_row` is compiler-internal. User programs continue to write
`pl.load` / `pl.tile.load`.

## See Also

- [34-init_memref.md](34-init_memref.md) — establishes the allocation identity this pass preserves
- [36-materialize_semantic_aliases.md](36-materialize_semantic_aliases.md) — the next default pass
- [00-pass_manager.md](00-pass_manager.md) — default ordering and pass properties
