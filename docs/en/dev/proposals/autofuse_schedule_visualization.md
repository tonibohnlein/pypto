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

The vector and cube views use different layouts because they describe different algorithm
structures. DOT remains the interchange format. The cube renderer uses Graphviz for a tile-centric
flow rather than treating pipeline phases as an operation-dependency DAG.

Vector timelines remain ordered event/liveness views. Each row contains an action and the on-chip
state immediately after it. They are derived from `VectorStreamPlan` and show:

- one logical region and its physical UB allocation;
- the strip driver or streamed statistics/apply phases;
- serial init, ragged tail, and finalize phases outside stage-2 loops;
- P4-generated online statistics work;
- source operations in the solver's pebbling/topological order;
- boundary loads, intermediate last-use releases, and GM stores.

Cube schedules are tile-centric flows derived from `CubeSchedulePlan` plus its shared
`L0MatmulPlan` children. Each matmul request shows one representative iteration of the output-tile
loop:

- the outer output-tile loop and all full/tail tile variants;
- an upper flow of K-slice operand tiles through fill, first overlap, repeated steady state, and
  pipeline drain;
- a lower flow representing the **same output tile** remaining in L0C while Matrix operations
  accumulate successive K slices into it;
- dashed Matrix-update arrows only where a K slice changes the output tile; a feed-only fill stage
  deliberately has no such arrow;
- the child L0 schedule summarized in the request header and executed by each Matrix update;
- a serial ragged-K tail outside the stage-2 ring when present;
- the single final FIXPIPE drain after the tile becomes complete;
- recursive L1 result retention and last-use release between matmul requests.

The optional split-K merge is shown as two ordered AIC phases. Phase one assigns K share zero to one
task per spatial region and uses normal stores. The dependency boundary then releases the remaining
shares, which atomic-add into the initialized GM output. The diagram reports both launch sizes and
the explicit synchronization-cost hook; there is no AIV zero-fill prologue.

The solution serializer reconstructs these descriptors only for final selected steps. They remain
absent from the local-search `CostResult` cache. Old `.sol.json` files without `vector_stream` or
`cube_schedule` must be regenerated before rendering an algorithm timeline.
