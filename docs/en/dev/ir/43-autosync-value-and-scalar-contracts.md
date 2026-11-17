# Autosync value domains and local scalar contracts

This checkpoint contains two separately qualified M2 packages on top of
`005ebfd35`. It preserves the three public synchronization passes, existing
allocation algorithms and explicit conservative policy. The source remains
dirty; the [receipt](43-autosync-value-and-scalar-contracts.json) identifies the
package files, logs and reproduction environment. The fresh isolated tests and
source hooks pass; the accepted commit pin is a separate publication step.

## A. Active-prefix value normalization

`NormalizeBufferActiveDomains` is an internal candidate producer called by
`SelectBufferIRFunctions`, not a new pass or analysis backend. It separates
guaranteed-defined value lanes, downstream required lanes and metadata observers.
An overwide result annotation alone does not establish defined values.

The admitted family contains FP32 lane-wise recipes, qualified row reductions,
contiguous vector axis maps and direct counted-loop ADD accumulation. A full
initialized seed can be restricted to the same runtime prefix as its backedge
when the bound is ready, lexically visible and invariant inside that loop.
One named prefix per component is retained; unrelated bounds, general unions,
inner-counter-dependent bounds and unknown producers remain unavailable.
Existing operation-validity premises bound the original active extent by its
physical capacity. The producer does not invent a caller restriction such as
`r = 8`.

```text
full initialized seed -> explicit prefix alias -> counted carry
partial rows -> row reduction -> contiguous row-vector prefix -> ADD
carry result -> unchanged lane-wise arithmetic -> column-vector prefix
partial matrix consumer -> original partial output store
```

The seed-producing payload is preserved. The loop initializer becomes an explicit
prefix alias of those same initialized bytes; a zero-trip loop retains its
qualified initialized values. Zero active rows use the existing empty-operation guards.
Physical shape, storage pitch, payload arithmetic, original guards and trip
counts remain unchanged. A full store observing a prefix-defined result refuses
normalization; the store is not narrowed. This distinguishes the positive
post-normalization shape from the full-output `rms_recip` obligation.

`tile.set_validshape` shares storage and metadata. Old full users qualify only
through the existing dynamic-owner per-use restoration and storage/liveness
contracts. Unknown metadata observers, including nested calls, call/scope/function
attributes and scalar carry initializers, retain the original function. Tile-
bearing aggregate tuples also refuse. Scalar initializers are inspected at their
For/While declaration, including zero-trip paths; body references do not replay
their initializer.

The immutable candidate is published only after **all** ordinary Buffer admission
checks succeed. A later recipe, planner or descriptor refusal restores the original
body, not a partially normalized Functional program. Lowering consumes explicit
normalized aliases and types; it performs no second permissive domain analysis.

### Construction work

Definition/domain indexing visits statement occurrences without following scalar
expression paths. Metadata discovery uses separate bounded caches for data and
observer contexts and a shared type inventory. Backward requirements have bounded
height: unused, one prefix, then full. Unsupported intersections are unavailable,
not empty answers.

A single immutable stability check covers scalar calls, arithmetic, tuples,
projections and metadata. Qualified expressions remain verbatim. Unchanged SSA
bindings, function ABI and metadata remain unchanged; only proved local Tile
declarations are rebound. In particular, preserving a scalar call expression does
not eliminate any of its statement executions. This avoids per-binding-generation
rescans and the base mutator's recursive walks of shared TupleType or IterArg
initializer graphs. Specific statement/expression handlers record actual dispatch
work; generic dispatcher counters would miss internal functor calls.

The read-only `testing.normalize_buffer_active_domains` adapter returns this same
candidate and optional per-request counters. `changed` means normalization only,
not whole Buffer admission or executable ownership. The sink is reset per request
and is not proof state or a cache.

## B. Local UB scalar payloads and Tile-to-Buffer bridge

The selected A2/A3 AIV contract admits `buffer.read` and `buffer.write` on dense
rank-2 Vec FP32/INT32 storage at constant coordinates inside static valid extents.
The flattened native index must fit uint32. Other cores, dtypes, dynamic points,
unproved metadata and missing layout provenance remain unavailable.

These are ordinary PIPE_S memory payloads. `buffer.read` has a read effect and a
scalar Value result; `buffer.write` writes its destination. The selected
`pto.tgetval`/`pto.tsetval` implementation uses typed UB GetValue/SetValue accesses,
without GM DCache, hidden asynchronous workspace or flags. The selected read
establishes readiness of its original blocking scalar result. Result behavior
alone does not supply that native guarantee.

The shared Tile bridge retains the original read SSA definition and cut. Later
branches, addresses and loop conditions consume that actual value. A discarded
or nested Tile read without the required direct SSA binding delegates before
lowering; no dummy value is manufactured. Assigned and discarded Tile writes both
mutate the original storage. A returned write result is an alias, not a fresh
allocation or copy; the complete MemRef/window, including slot coordinates, must
agree. Old aliases continue to observe the mutation.

### Static physical pitch and metadata

Native scalar flattening uses `row * physical_columns + column`. A narrow window
can retain a wider parent's row pitch, so descriptor shape alone is insufficient.
The existing placement/window inventory must establish the dense physical pitch.
Full-width overlapping windows can qualify; narrow inherited-pitch windows and
borrowed handles lacking that evidence refuse. Validation, analysis, insertion,
actual-token lowering and code generation enforce the same premise. An ALL
barrier cannot repair an incorrect native address calculation.

An original static point can use the existing static descriptor bridge after
another alias promotes its owner to runtime metadata. Its original extent and
physical pitch are restored; this does not admit a genuinely dynamic point or
infer exact access from an unavailable descriptor.

### Independent command-model correction

The bounded oracle now gives native scalar completion order to subsequent S
instructions and to actual scalar-value/control consumers. It does **not** turn
every scalar access into a fence for unrelated pipes. Reads/writes still need
the applicable cross-pipe service. Tests expose a deliberately missing S-to-V
write service, independent-pipe cases, branch skips, empty loops and final-false
while tests. Matching, acyclicity and consumption-before-republication are checked
from actual commands, separately from modeled byte-conflict coverage.

## Host evidence and reproduction

Recorded focused runs and the fresh isolated package gate use the unchanged
120-second watchdog:

| Gate | Result | Scope |
| ---- | ------ | ----- |
| Active-domain module | 31 passed, 15.19 s | Prefix values, zero/short trips, resets, observers, rollback, shared DAG/type work and SSA identities |
| Scalar family, eight modules | 124 passed, 3.59 s | Tile bridge, physical pitch, command oracle, scalar payloads and existing coordinate/row/cast/broadcast regressions |
| Isolated M2 plus retained-child API, nine modules | 152 passed, 17.68 s | Fresh coherent M2 build with M5 code parked; normalization, scalar contracts and existing M4 import boundary |

The first two rows are separate runs, not one combined 155-test receipt. The
isolated gate excludes the pending M5 coordinate package. Earlier compilation,
fixture, counter and timeout evidence is retained in the JSON; it is not erased
by the final focused results. The compiler expert accepted the source contracts
with the integrated tests and source hooks now passing. The three selected
resource/numerical CTests also pass.

From this worktree, source `.claude/skills/testing/load-env.sh`, set the build/test,
OpenMP and code-generation worker counts explicitly to one, and use the pinned
Python and PTOAS launcher recorded in the receipt. The normalizer gate is:

```bash
timeout 120s python -m pytest -q --require-ptoas \
  tests/ut/ir/transforms/test_runtime_valid_reduction_axis.py
```

The receipt lists the exact eight scalar modules and environment. Source/dirty
hashes identify this checkpoint; the final isolated commit/build pin remains a
separate publication step.

### Unchanged whole-Qwen host capture

At the pinned PyPTO-lib source, the original two-layer, sequence-128, batch-1
capture generates all 32 leaf C++ files. The explicit conservative-prefix policy
reports 79 execution domains: 47 observed task-control owners, 17 verified PyPTO
leaves and 15 delegated leaves. `post_rmsnorm` is newly owned; the previous count
was 16 owned and 16 delegated. Its 6.690 s inclusive compilation time is one
descriptive measurement, not an analysis-only time or speedup.

The remaining Buffer admission obligations are reported without reinterpretation:

| First reported obligation | Delegated leaves |
| ------------------------- | ---------------- |
| `operation_recipe / tile.add` | `out_proj_aiv`, `out_proj_aiv_0`, `rms_recip`, `rms_recip_0` |
| `type_layout / Var` | `lm_head`, `qk_pv_partial_plan_aic`, `qk_pv_partial_plan_aiv`, `qk_pv_single_block_0_aic`, `qk_pv_single_block_0_aiv`, `qk_pv_single_block_aic`, `qk_pv_single_block_aiv`, `qkpv_plan`, `rope_kv_cache`, `token_embed_group`, `token_embed_single` |

These are the first production admission diagnostics, not a claim that each
underlying problem has the same cause. In particular, the full-observation
`rms_recip` case remains a refusal. No hand-written or compiler peer protocol is
replaced by new cross-core synchronization.

A separate exact-policy capture also generates all 32 C++ files and retains the
three existing exact owners (`final_rmsnorm`, `mlp_out_seed`, `silu`); 29 leaves
delegate. That command incorrectly required three prefix-only projection owners
and exited with an ownership-expectation failure. The receipt preserves that
failure: it is not a strict green exact qualification and is not a compiler
failure. The 6.668 s measurement is likewise inclusive and descriptive.

## Claims and remaining obligations

| Established result | Remaining premise or limitation | Appropriate claim |
| ------------------ | ------------------------------- | ----------------- |
| Qualified value-prefix normalization preserves observed lanes and original control | Unsupported producers, observations and aggregate escapes refuse | A reusable input/native-metadata repair, not a new minimum-demand theorem |
| Local scalar effects and original SSA readiness reach existing synchronization services | A2/A3 AIV, static FP32/INT32 UB points and proved pitch only | Qualified local scalar realization with bounded host checks |
| Independent actual-command checks preserve matching and causal reuse | Existing finite oracle/native model and six eligible IDs | Feasibility for checked plans, not minimum capacity or device correctness |
| Unchanged two-layer Qwen capture generates 32 C++ leaves; prefix policy owns 17 and delegates 15 | Exact policy retains three owners; incomplete exports, effects and control remain | Improved bounded host coverage, not complete Qwen ownership or device qualification |

GM `tensor.read` visibility and producer publication remain separate obligations.
DMA-staged GM scalar reads, INT64 staging, new peer synchronization and general
token/protocol realization are not implemented by this package. Allocation,
hardware exemptions, expert protocols and zero-inner-body barriers are unchanged.
No device validation or device speedup is claimed.
