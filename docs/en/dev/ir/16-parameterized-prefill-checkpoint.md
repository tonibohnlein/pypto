# Parameterized prefill checkpoint

The original checkpoint below implements ready invocation-invariant coordinates for the existing
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

## Range and numerical-repetition follow-up (2026-11-10)

The reviewed Buffer ABI increment is committed as `b97b6fd0`. Its separate
928-test gate and three specialist reviews passed. Buffer selection and direct
code generation now preserve symbolic leading GM extents; that infrastructure
does not itself establish an access range. The earlier checkpoint and receipt
above retain their historical scope. This follow-up extends the same importer,
owner and physical emitter, without changing reduction or allocation algorithms.

### Original-width range evidence

The caller collector and coordinate DAG now admit nonnegative bounded integer
arithmetic and explicit value-preserving casts, checking every intermediate in
its original type. A widening cast cannot hide an earlier overflow. Narrowing
requires the entire source interval to fit; a multiply-by-zero does not suppress
qualification of its operands. Binary operands and results must have matching
types unless an explicit cast intervenes. Loop updates must also fit the original
induction type. Unknown, negative, overflowing and unsupported expressions still
provide no evidence. This reuses the existing bound environment and call mapping.

A dynamic leading GM extent is admitted only when its original shape variable is
also an explicit INDEX formal with established bounds. This is the existing typed
Tensor ABI relation, not a correlation inferred from parameter names or samples.
The largest admitted extent bounds retained geometry; every access must fit the
**smallest** admitted extent. This sufficient test deliberately does not prove
relational bounds such as `offset + size <= rows` when independent interval hulls
are insufficient. A storage query needs the actual, in-range extent and must lie
inside that invocation's allocation. Outside the allocation is unavailable;
inside it, untouched storage has an exact absence answer. Local queries remain
available without unrelated GM bindings.

The new independent cases cover original INT32-to-INDEX coordinates, safe and
lossy narrowing, signed/unsigned loop increments, hidden overflow, invalid or
missing extent bindings, and actual caller-derived production selection. Original
address expressions and physical commands are checked together. Dynamic loop
trip counts, implicit shape symbols and forwarded relational caller facts remain
unproved. In particular, Qwen's implicit weight extent and INT32 layer offset are
not qualified merely because the new generic fixture is.

### Invariant perfect nests

A perfect nest of constant sequential loops may reuse one finite body when its
modeled access descriptions are invariant under every enclosing counter and it
has no conditional initialization/reset. No loop has carried results. All
original frames, starts, steps and counts are retained. A visit with normalized
coordinates `(i_0, ..., i_d)` maps to the checked mixed-radix ordinal
`(...((i_0 * T_1 + i_1) * T_2 + i_2)... * T_d + i_d)`.
This is a bijection preserving reference order. Identical effects, pipes and
prerequisites therefore give the same native and storage graph as the existing
repeated-body representation, including crossings at inner-counter resets.
Every product and original induction update is checked for signed-64-bit safety.
Zero counts produce no occurrences and preserve the original empty control flow.

The physical emitter computes this ordinal from the original in-scope variables;
it keeps the original loops and payload cuts. Global first/last guards and reuse
residues consequently do not restart at each inner loop. Existing allocation
checks remain authoritative. This does not assert that a child exit completes
payloads. The separate logical-token export refuses nested transport until tokens
can be carried through every enclosing frame; it does not discard mathematics
or block the independently qualified physical export.

This is an invariant perfect-nest specialization, **not general nested regional
composition**. Moving inner slices, outer-coordinate accesses, reset branches,
non-perfect bodies and unavailable same-DMA-pipe barriers remain explicit
obligations. A negative fixture retains correct mathematics after physical export
refusal. Neither hardware exemptions nor barrier placement are weakened.

At fixed depth the tested site, cell, demand, distance-table and static-command
cardinalities stay constant as either trip count grows, through `10^6 * 10^6`
leaf visits. Original frames and emitted ordinal arithmetic grow with depth;
numeric coordinates need logarithmic bits and are bounded here by signed 64-bit
arithmetic. Bounded tests include zero, singleton, repeated and three-level
visits, nonzero starts, nonunit steps, parent accesses and repeated nests.
They independently check accesses, actual retained covers, all event queries,
selectors, matching, acyclicity and causal flag reuse. Source generation goes
through PTOAS v0.72; there is no new device qualification.

### Relationship to the current draft

The read snapshot is draft revision `ad6c7ac5` plus working changes;
`minimum_demands.tex` SHA-256 is
`5e890a89ae29279a2578ec3b6ae7b05de9184ed0063404ce00e30602c6b8cb6e`.

| Draft construction | Implementation at this checkpoint | Remaining obligation |
| ------------------ | --------------------------------- | -------------------- |
| Section 5.2 finite demands | Existing numerical reducer remains shared | Broader guarded input contracts |
| Section 5.4 practical compact classes | Caller-range premises and existing bounded representative-body reduction | Resident slices, phase-dependent effects and richer access certificates |
| Section 5.5 composition/repetition | Existing sequential crossings plus invariant perfect-nest occurrence normalization | General retained-child summaries, moving selectors and accumulator resets |
| Direct realization and finite-ID reuse | Existing physical construction with original-frame ordinal translation | Nested logical-token transport; native invocation completion |

The general shared numerical interface theorem and streamed/resident-child
constructions are **not** implemented by multiplying trip counts. Those remain
the next step for Qwen output projection, followed by gate/up. Section 6's token
research is outside this change.

The [follow-up receipt](range-nested-followup.json) records source and tool hashes,
synthetic cases, bounded independent checks and a reanalysis of the existing Qwen
Buffer capture. It does not replace that capture with a specialized input.
Reproduce the focused checks with the same build and environment above:

```bash
/usr/bin/python3 -m pytest tests/ut/ir/transforms/test_sync_range_proofs.py \
  tests/ut/ir/transforms/test_nested_sync_repetition.py -q --require-ptoas
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_range_nested \
  --output build-py314/range-nested-followup --native-source \
  --qwen-capture build-py314/qwen-buffer-next/program.msgpack
```
