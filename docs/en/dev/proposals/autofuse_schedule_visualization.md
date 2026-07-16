# AutoFuse schedule visualization

## Purpose

AutoFuse exposes two complementary Graphviz views:

1. the tensor/operator DAG with the chosen fusion partition; and
2. one algorithm timeline for each selected homogeneous vector or cube kernel.

The partition view answers **which operations fuse**. The algorithm view answers **what one logical
work unit actually does**. It does not draw all SPMD blocks: the title records the grid and the body
shows one solver-owned region/tile, including its internal streaming loops.

## Generate the views

Dump the extracted problem and the selected solution while compiling:

```bash
mkdir -p build/autofuse-dump
PYPTO_AUTOFUSE_DUMP=build/autofuse-dump \
PYPTO_AUTOFUSE_GENERIC_EMIT=1 \
python -m pytest tests/ut/ir/transforms/test_auto_fuse.py -k case_name -q
```

For a function named `kernel`, render the partition and one selected kernel:

```bash
python 3rdparty/pto-fusebox/scripts/visualize.py solution \
  build/autofuse-dump/kernel.dag.json \
  build/autofuse-dump/kernel.sol.json \
  build/autofuse-dump/kernel.dot

python 3rdparty/pto-fusebox/scripts/visualize.py algorithm \
  build/autofuse-dump/kernel.dag.json \
  build/autofuse-dump/kernel.sol.json 0 \
  build/autofuse-dump/kernel-kernel-0.dot

dot -Tpng build/autofuse-dump/kernel.dot \
  -o build/autofuse-dump/kernel.png
dot -Tpng build/autofuse-dump/kernel-kernel-0.dot \
  -o build/autofuse-dump/kernel-kernel-0.png
```

Use `algorithms` to emit one DOT file for every homogeneous vector/cube step:

```bash
python 3rdparty/pto-fusebox/scripts/visualize.py algorithms \
  build/autofuse-dump/kernel.dag.json \
  build/autofuse-dump/kernel.sol.json \
  build/autofuse-dump/kernel
```

The Fusebox helper `scripts/render_fusion_dag.sh` generates the partition PNG and all per-kernel
algorithm PNGs in one command.

## Partition view

Operations use the selected kernel's color; no cluster box changes the DAG layout. A compact legend
maps each color to its kernel kind, maximum region, grid, split, core count, and modeled latency.
Tensors distinguish GM-visible values from within-kernel on-chip values. Orthogonal, port-anchored
edges are drawn before nodes so arrows do not cover operation labels.

## Per-kernel algorithm view

Each row is one ordered algorithm event. The left column is the action; the right column is the
on-chip state immediately after that action. The action colors are stable: loads are blue, compute
is green, pipelines/loops are purple, carries are yellow, drains/stores are orange, and releases are
gray.

Vector timelines are derived from `VectorStreamPlan` and show:

- one logical region and its physical UB allocation;
- the strip driver or streamed statistics/apply phases;
- serial init, ragged tail, and finalize phases outside stage-2 loops;
- P4-generated online statistics work;
- source operations in the solver's pebbling/topological order;
- boundary loads, intermediate last-use releases, and GM stores.

Cube timelines are derived from `CubeSchedulePlan` plus its shared `L0MatmulPlan` children and show:

- one spatial/split-K work unit and optional zero-seed prologue;
- every recursive matmul request in execution order;
- output/L0C tile variants;
- GM→L1 K-window init, rolled overlap, and ragged tail;
- nested L1→L0A/L0B loads and `TMATMUL`/`TMATMUL_ACC` work;
- the single final FIXPIPE drain to L1 or GM;
- L1 intermediate retention through its priced last consumer and its release.

The solution serializer reconstructs these descriptors only for final selected steps. They remain
absent from the local-search `CostResult` cache. Old `.sol.json` files without `vector_stream` or
`cube_schedule` must be regenerated before rendering an algorithm timeline.
