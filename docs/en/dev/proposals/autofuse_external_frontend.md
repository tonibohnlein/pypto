# External AutoFuse frontend and PyPTO source generation

## Status

This document records a future research direction. It does not describe a currently supported
PyPTO API. The existing compiler-integrated AutoFuse implementation remains the reference for the
cost model, schedule-plan contracts, and emitted-kernel behavior.

## Goal

Run AutoFuse and AutoTile as an external source-to-source scheduling system:

```text
PyTorch/Hugging Face module + configuration + shape constraints
        |
        v
torch.export / FX graph capture
        |
        v
Torch-to-scheduler-DAG adapter
        |
        v
Fusebox AutoFuse + AutoTile planning
        |
        v
PyPTO DSL source generator
        |
        v
ordinary PyPTO compiler and runtime
```

The generated PyPTO source already contains the selected fusion boundaries, grid, regions,
topological order, physical tiles, loops, pipelines, cross-core FIFOs, and valid-shape handling.
PyPTO verifies and lowers that explicit schedule; it does not rerun AutoFuse or AutoTile.

## Component boundaries

### Torch/Hugging Face capture adapter

Use `torch.export` or FX rather than translating arbitrary Python source. The adapter imports:

- tensor operators, dtypes, layouts, aliases, mutations, and outputs;
- symbolic-shape constraints and representative input metadata;
- module configuration needed to resolve model constants;
- explicit opaque nodes for unsupported control flow, custom operators, and data-dependent access.

Hugging Face model code is normally a PyTorch `nn.Module` plus configuration and weights. It
describes model semantics, not necessarily deployment choices such as paged KV-cache layout,
quantization, or runtime task dependencies. Those choices must be present in the captured graph or
supplied as explicit frontend configuration; the adapter must not guess them.

### Fusebox scheduler core

Fusebox should remain independent of Torch and PyPTO implementation types. Its input is the existing
tensor DAG extended only with frontend-neutral shape constraints and opaque-boundary metadata.

For each connected candidate group, the scheduler retains the unified design:

1. choose fusion or cut boundaries;
2. assign one common grid to the selected group;
3. propagate output regions backwards through the tensor DAG;
4. choose a legal topological/pebbling order;
5. account for boundary and produced-value lifetimes;
6. check UB/L1/L0 and crossing-FIFO feasibility;
7. price compute, transfers, drains, and implementable overlap;
8. return a complete, code-generatable solution descriptor.

The scheduler does not create a full-program hardware schedule. It emits good kernels and preserves
their dependency graph; the PyPTO runtime scheduler launches ready kernels and overlaps independent
AIC/AIV work.

### PyPTO source backend

The backend consumes only the selected solution descriptor and emits ordinary, readable PyPTO DSL.
Depending on the plan, that includes `pl.spmd`, static physical tile shapes, runtime `valid_shape`,
`pl.range`, `pl.pipeline`, tensor views, explicit GM boundaries, and supported `tpush`/`tpop`/`tfree`
cross-core transport.

The backend must not redo planning. Every emitted loop, lifetime, transfer, and FIFO must be traceable
to a field in the solution descriptor. Alongside the source, it should publish the existing schedule
report and pseudocode so a user can inspect why the implementation was selected.

## Dynamic shapes

PyPTO already supports one extent-polymorphic artifact: runtime dimensions drive tensor views, loop
bounds, offsets, and valid shapes while physical hardware tiles stay static. The external scheduler
should preserve that baseline.

### Type 1: dynamic extent, static physical chunk

The first supported dynamic-shape class keeps every schedule-defining quantity static:

```text
runtime:       M, loop trip count, offsets, final valid extent
compile time:  CHUNK, physical tile shapes, grid policy, pipeline depth, memory allocation
```

For a fixed `CHUNK`, Fusebox plans the body exactly like a static problem. The dynamic axis only
changes how many copies of that static region exist and the logical size of the last copy:

```python
m = pl.tensor.dim(x, 0)
for m0 in pl.range(0, m, CHUNK):
    valid_m = pl.min(CHUNK, m - m0)
    x_tile = pl.slice(x, [CHUNK, D], [m0, 0], valid_shape=[valid_m, D])
    # Statically planned DAG over physical [CHUNK, D].
```

Admission initially requires:

- the dynamic dimension is an outer/free axis whose chunks are independent;
- all physical tile extents and memory footprints are compile-time constants;
- region propagation through the group is affine and preserves the chunk boundary;
- a tail is representable as a clamped logical region inside the same physical frame;
- no data-dependent address or control decision changes the per-chunk DAG.

Capacity is checked for a full physical chunk. For a runtime extent `M`, the planner derives
`ceildiv(M, CHUNK)` logical regions and prices their waves using the existing static-region model;
the final region uses the normal clamped/tail price. It must not multiply a per-chunk latency when
those regions are actually concurrent hardware work units.

Two modes use the same contract:

1. **Programmer-fixed chunk.** Import `CHUNK` as a schedule constraint and optimize only the group
   inside it.
2. **Scheduler-selected chunk.** Enumerate a small legal static set, reject capacity-infeasible
   values, and compare total cost for a representative `M`, a bounded range, or a supplied shape
   distribution.

The generated artifact remains extent-polymorphic in both modes. Multiple variants are warranted
only when the best physical chunk, grid, or pipeline changes materially between shape regimes.

Dynamic reduction axes are not part of this first class: they require an explicit loop-carried
reduction state such as a cube accumulator, online-softmax state, or Welford tuple. Each chunk is
still static, but the recurrence needs its own model/emit contract.

Examples already present in `pypto-lib`:

- `models/deepseek_v4_pro/rmsnorm.py`: dynamic token extent `T_DYN`, static `T_TILE=8` and
  `D_TILE=128`. Its supported decode/prefill configurations assert divisibility, so it has no tail.
- `models/deepseek_v4_pro/hc_head.py`: dynamic token extent, static `LINEAR_T_TILE`, and
  `t_rows = min(LINEAR_T_TILE, t_dim - t0)` carried as the input slice's `valid_shape`. This is the
  canonical full-chunk-plus-runtime-tail form.
- `models/qwen3_14b/rms_lm_head.py`: dynamic batch rows, static `BATCH_TILE`, and
  `lm_valid_rows = min(BATCH_TILE, batch - b0)` applied to the cube result before the output store.
- `models/deepseek_v4_pro/hc_post.py`: dynamic token count selects the SPMD work-unit count while
  each work item uses static token/data tiles; the prefill variant guards the partial final tile.

Generate multiple plan variants only when a different physical tile, grid, or pipeline is materially
better for another bounded shape regime. The external tool then emits the variants plus a small host
dispatcher. It does not compile one variant per request, and it leaves data-dependent operations such
as TopK or paged gathers as opaque fusion boundaries until their semantics and cost are represented.

## Conceptual API

The following illustrates ownership; it is not a proposed final Python API:

```python
exported = torch.export.export(module, example_args, dynamic_shapes=shape_constraints)
problem = torch_frontend.to_fusebox(exported, model_config=config)
solution = fusebox.plan(problem, target="ascend910b")

source, report = pypto_backend.generate(solution)
compiled = pypto_compile(source)
```

`fusebox.plan` owns grouping and tiling. `pypto_backend.generate` is a deterministic serialization of
the solution, and `pypto_compile` is the ordinary compiler path.

## Initial validation ladder

1. Round-trip a static vector RMSNorm graph and compare generated PyPTO with the current AutoTile
   solution and device result.
2. Round-trip a static cube matmul and compare plan fields, emitted PTO, and numerics.
3. Round-trip a generic `QK -> vector softmax DAG -> PV` graph without an attention recognizer.
4. Preserve unsupported nodes as explicit cuts and verify values crossing each boundary.
5. Add one bounded dynamic outer dimension using a static physical tile and runtime valid shape.
6. Add plan variants only after device evidence shows one dynamic plan is materially suboptimal.
7. Port target parameters and capability admission before evaluating DeepSeek V4 Pro on Ascend A5.

Each level requires graph-semantic comparison against PyTorch, solution-to-source contract tests,
PyPTO parse/compile tests, and device correctness before performance ranking.

## Non-goals

- translating arbitrary Python control flow directly;
- recognizing model names or hard-coding FlashAttention/SwiGLU algorithms;
- choosing quantization precision on behalf of the model author;
- replacing the PyPTO compiler, verifier, PTOAS, or runtime scheduler;
- silently approximating unsupported aliases, mutations, views, or data-dependent accesses.
