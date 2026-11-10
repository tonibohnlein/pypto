# Parameterized prefill checkpoint

This checkpoint implements ready invocation-invariant coordinates for the existing
finite/periodic synchronization provider. **Nested Qwen output projection and
gate/up composition are not implemented by this change.** It neither expands
whole nests nor introduces another analysis or allocation backend.

[Machine-readable receipt](parameterized-prefill-checkpoint.json) records the final build and commands.

## Admitted coordinate contract

`RealizeSync` is a program pass so it can inspect original direct callers before
attempting each kernel. `CollectSyncParameterBounds` uses PyPTO's
`ConstIntBoundAnalyzer` environment and the existing call/Submit argument mapping.
It joins inclusive bounds across all direct sites. One unknown site invalidates
that formal's evidence. Submit arguments, dependencies, launch operands and
reference attributes use the normal IR visitor; runtime-allocated outputs do not
shift scalar parameter identities. Function handles are Ops, not first-class
scalar expressions. No call graph fixed point or inference from parameter names
is used.

The admitted expressions are INDEX constants, bounded loop indices, pure scalar
assignments, addition and multiplication with proved nonnegative, nonoverflowing
intermediates. The access importer additionally requires affine coordinates.
Bounds are established before normalization: an overflowing multiplication does
not become qualified because a later operation multiplies it by zero. Unknown
values, narrow casts, subtraction, symbolic loop limits and interprocedural
parameter forwarding remain unproved. No branch sample establishes a range.

Callee formals are ready at invocation entry and immutable within the invocation.
Scalar payload computation and cross-pipe scalar readiness are not newly admitted.
The production evidence belongs to the closed program being compiled. The strict
research API accepts explicit `parameter_bounds` keyed by **original parameter
position**; these are supplied proof premises, not runtime samples. An isolated
function with missing evidence still refuses.

## Why the common translation preserves exact demands

For each GM domain `s`, every imported access must have the same parameter-dependent
translation `T_s(p)` and an otherwise supported relative footprint. Thus each
physical footprint is `T_s(p) + F_i`. For a fixed invocation, translation is
injective and preserves intersections between all reads and writes in that domain.
Different GM domains remain disjoint only under the explicit existing
`autosync_disjoint_gm` caller contract. Local placement and hardware qualifications
are unchanged. Checked upper bounds establish that translated footprints and all
intermediate arithmetic remain within the supplied static allocation geometry.

The existing reducer therefore computes the same modeled required closure and
covers for every admitted parameter assignment. Numerical rank/distance results
are retained once. Addresses in the emitted program are the original expressions;
there is no recompilation for an arbitrarily chosen coordinate value.

Storage exports retain the translation coefficients. `boundary(..., parameters)`
checks the supplied bindings against the retained bounds and translates the queried
physical point back into the relative frame. A missing or invalid binding returns
unavailable (`None`). A covered point outside every translated footprint returns
an available answer with absent readers/writers. Local domains require no GM
parameter binding. Query results over original occurrences need no parameter
specialization because the selected order is invariant under the common translation.

This argument does **not** establish disjointness across invocations, notification
rearming between separate calls, or native return completion. The existing fresh
entry/closed-notification and runtime invocation assumptions still apply.

## Construction work

Scalar definitions retain a checked expression DAG. Only requested access roots
are flattened into affine coefficients, using aggregate weights in topological
order. Caller range and expression traversal caches are local to each binding
environment and are invalidated when bounds change. This avoids repeated path
traversal of shared expressions and quadratic storage of every SSA prefix's
coefficient map. It does not promise globally linear work across all environments
or overlapping requested roots.

Structural tests cover shared DAG depths 12, 24 and 30 and scalar-prefix chains
with 8, 32 and 128 parameters. They bound inspected expression/range visits,
retained coordinate nodes, flatten visits and emitted coefficient entries. The
prefix cases also compare accesses at both parameter extremes independently.

## Source and host qualification

The fixtures are byte-identical copies of PyPTO-lib
`da5350d2bfc40ce12edeaafd44b698050f6dc85b`:

| Source | SHA-256 |
| ------ | ------- |
| `examples/beginner/matmul.py` | `8316cc2f039f818f08dd5e0c2bb8c3a90e9abff19501e253de4a0b787bd95585` |
| `examples/intermediate/gemm.py` | `565faaef5ad54962c122c8a5e9f7819475832b63cc70cd74e1f270f7068be065` |

Their original 256×256×256 configuration establishes tile coordinates `[0,192]`
from caller loops. Both select PyPTO autosync through the real frontend, Buffer
lowering and placement, allocate within six eligible IDs per direction, and
produce C++ through pinned PTOAS v0.72. No device run is included.

| Per-kernel invocation | Matmul | GEMM |
| --------------------- | -----: | ---: |
| Retained analyzed occurrences | 9 | 16 |
| Retained cells | 13 | 11 |
| Static SET / WAIT / local-barrier sites | 5 / 5 / 0 | 7 / 7 / 0 |
| Executed SET / WAIT sites in the checked original invocation | 5 / 5 | 22 / 22 |
| Highest directional ID used | 1 | 1 |

The seven reserved lanes in GEMM are summed **across directions**, not seven IDs
in one direction. These counts are for the emitted kernel invocation, not all
M/N tile invocations of the orchestration program.

The independent interpreter evaluates original accesses using concrete bindings,
without compiler cells or generators. It checks imported footprint agreement,
constructs conflicts and event closure independently, compares **actual retained
covers**, all four event-query combinations, and boundary points including absent
answers. The command checker verifies matching, acyclicity, directional capacity,
causal consumption before republication, and equality of modeled payload order on
the checked executions. Native macro/access semantics remain supplied model
premises; these checks do not prove them on silicon.

The original two kernels are checked at three offset pairs. Positive K derivatives
1, 2, 3 and 4 cover short and steady-state behavior. K=0 is not a valid derivative
of this GEMM source: its final store would read an uninitialized accumulator.
Existing zero-body/reset fixtures remain separate regressions rather than
inventing initialization for that input.

K derivatives 4, 8, 64 and 1,000,000 retain exactly 16 analyzed occurrences,
11 cells, a five-site repeated body, six repeated demand families, 100 distance
entries, and 7/7 static SET/WAIT sites. The K trip count is retained numerically.
This is compactness evidence for the existing single-level repeated class,
**not** validation of nested Qwen composition. Measurements are descriptive
single runs. Analysis detail timers are nested within analysis wall time; physical
realization includes allocation and emission. Do not sum inclusive timers.

Reproduce from this worktree (with the pinned PTOAS installation on `PTOAS_ROOT`
and its required Python interpreter on `PATH`):

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 1
export PYTHONPATH="$PWD/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
/usr/bin/python3 -m pytest tests/ut/ir/transforms/test_parameterized_sync.py -q --require-ptoas
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_parameterized_prefill \
  --output build-py314/prefill-coverage-20261110/qualified --native
```

Absolute paths in the receipt are recorded host provenance; the commands above
use worktree-relative paths for reproduction.

The generated JSON pins dirty source and extension hashes, inputs, individual
checks, retained sizes, inclusive stage measurements and generated C++ hashes.
The local campaign also retains failed build logs and the initial wrong-PTOAS
launcher failure; neither is counted as a successful final verification.

## Remaining nested composition work

The pinned Qwen `_out_proj_aic_phase` body has been lowered as an extracted region,
not as the complete model. Its first production boundary is Buffer selection:
`SelectBufferIRFunctions::CheckType` requires static tensor extents, while the
weight has dynamic leading extent `LAYER_HIDDEN_ROWS_DYN`. Local allocation shapes
and the ND row stride are known. Admitting that geometry must preserve the dynamic
allocation extent and prove the layer-offset/cast/access bounds; substituting the
benchmark's layer count would not establish that contract.

After that boundary, the retained synchronization service still represents a flat
sequence of finite or singly repeated regions. Qwen has output-tile, K-pipeline
and operand-fill loops. It requires the following extension of the existing owner:

1. Retain a child result and its original multi-level occurrence coordinates.
   Export exact native and storage boundaries and all-event queries; do not rebuild
   the child as an expanded list of visits.
2. Compose K stages using those ports. Keep Mat operands resident through all
   their inner fills; distinguish the first accumulator initialization from later
   accumulation and from the next output-tile reset.
3. Reduce crossings jointly using child reachability. Carry output membership and
   selectors outward, even where output tiles are disjoint for the current visit.
   Reuse of those tiles after a reset needs a new crossing check at that scope.
4. Extend endpoint coordinate translation and prove physical reuse through the
   same mixed-event queries. A closed child notification state alone does not
   certify palette reuse at the next visit.
5. Check actual covers, queries/selectors and emitted commands independently on
   bounded nested instances, then establish representation scaling in outer K
   and inner fill counts separately. Only then apply the construction to gate/up.

The draft snapshot inspected here is `ad6c7ac5` plus working changes. During the
checkpoint, `minimum_demands.tex` changed from SHA-256 `ed071499…` to `38dc658b…`;
the latter adds qualified interface bounds and symbolic-inner-length repetition.
A subsequent snapshot (`696d41dc…`) adds constructive numerical buffered pipelines,
row-indexed collective composition and streamed external-family interfaces.
Those are relevant next-stage constructions, not implemented features of this checkpoint.
Its child-query, selector, coincidence and construction-cost premises remain
obligations of the planned implementation. The present code does not implement
that general symbolic repetition theorem.

## Checkpoint gate

The focused synchronization gate passed 289/289 tests after restoring the existing
rotating-view refusal diagnostic. It includes finite, compact, logical, physical,
Vec, row and indexed regressions as well as the new parameterized cases. The
change preserves the opt-in switch, allocator, hardware exemptions and original
fallback bodies. The larger nested-prefill milestone remains open.
