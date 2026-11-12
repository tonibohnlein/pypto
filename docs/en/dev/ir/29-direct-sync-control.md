# Direct construction of shared synchronization control

This checkpoint continues the reviewed case-sequence sharing implementation.
The starting source is `4f5fdfeada1085841de6cb5bfd7637d391a9663e` plus the
working changes pinned in `build-py314/direct-control/baseline/source.json`.
A commit hash alone does not identify that baseline. The companion
[receipt](direct-sync-control.json) pins the source, tools, commands and results.

## Contract and ownership

`LowerSync` consumes the actual SSA token uses. The scoped parser now retains
immutable endpoint intervals before flattening them into allocation coordinates.
A domain owns a shared original-frame context: function, invocation case,
region and original loops. Its half-open intervals retain source substitutions
and target guards. An unavailable capability is not an empty interval.

Allocation retains those handles and changes only its existing ID assignment.
Lowering checks owner, case, region and frame identities. Mathematical diagnostic
clients decode into the same endpoint representation at their compatibility
boundary. This does not authorize a mathematical sidecar to overwrite actual
logical token uses.

At each original payload cut, physical lowering constructs ordered command
**descriptors**, containing the assigned action/direction/ID and guarded domain.
The cut inventory owns placement and before/after order. Adjacent guard sharing,
case-prefix/suffix sharing, schedule selection and size accounting operate on
these descriptors. Only the selected forest is materialized into statements.
The former production statement-forest scheduler is removed. The testing helper
imports command-only statements into the same scheduler.

Endpoint predicates are cached per retained domain; immutable original-loop
bounds are simplified once per canonical domain, with no branch-local facts.
The existing final scope-local scalar-value materializer remains separate and
never hoists a value from a skipped branch. No allocation algorithm, logical
protocol, native exemption, synchronization cut or payload instruction changes.

## Exact unions and the cost certificate

Two identical complete cut sequences can share one body under the union of
their disjoint launch intervals. For example:

```text
if block <= 4: S
if block >= 19: S
```

becomes:

```text
if (block <= 4) or (block >= 19): S
```

Blocks 5 through 18 still execute no command. The whole ordered sequence must
match, including IDs, directions, guard SSA identities and duplicate commands.
This does not combine partially equal symbolic families or different protocols.

Sharing must improve static command/guard/expression size without worsening
any component. It also passes a per-cut worst-case control certificate. The
compiler counts comparisons, eager Boolean operators and conditional decisions
on the exact interval truth partition, separately for the active-body and empty
classes. Both operands of scalar `And`/`Or` contribute; this is not a host-language
short-circuit estimate.

Unchanged inner work cancels only when its admissible domain is independent of
the launch interval. The initial sufficient check permits original literal-bound,
positive-step IVs and constants, without intervening conditional control.
A nonliteral nested loop clears earlier IV evidence; alias visits intersect their
evidence. Unknown SSA values or correlated control retain the incumbent.
This is a per-cut bound, not an exact whole-kernel runtime bound. Individual case
costs can change even when the worst case does not increase. The initial union
subclass contains exactly two nonempty residual sequences. For three exhaustive
`A/B/A` cases, the scheduler may first strip a common ordered prefix and suffix,
provided the middle residual is empty. Their complete inner work must satisfy
the same uniformity certificate. Dispatch cost is compared against the actual
selected incumbent, and the final size check includes the reattached prefix and
suffix. Other unions keep the existing tree.

## Original-coordinate boundary normalization

For a retained domain on the original loop `i = 0 .. 256 step 64`, the last
visit is `i == 192`. Lowering can emit that directly instead of `(i/64) == 3`;
the nonfirst interval similarly uses `i >= 64` instead of `(i/64) >= 1`.

The proof requires the same original loop/frame, literal start and positive step,
a valid admitted digit, representable thresholds and last IV, and a normalized
subtraction span `(count-1)*step <= INT64_MAX`, checked in `__int128`.
Nondivisible stop values use the actual lattice point, never `stop-step`.
Empty frames produce no endpoint; singleton domains do not lose duplicate
commands. Unknown or symbolic starts retain the previous normalized expression.
Negative loop starts remain outside the existing importer contract. The emitter
check does not broaden admission or justify wrapping arithmetic.

Symbolic family unification, phase restructuring and loop splitting remain
separate work. This checkpoint neither clones payloads nor infers new periods.

## Measurement scope

The receipt separates static commands, branches, guard-expression occurrences,
C++ bytes, bounded executed commands, scalar comparisons, Boolean operations and
arithmetic operations. Expression occurrences are not native instruction counts.
The independent ordering campaign checks emitted C++ and reports unsupported
oracle cases explicitly. Exact trace equality includes payload cuts, direction,
ID, multiplicity and order, so it preserves the checked prerequisite relation;
smaller code does not trade communication or payload concurrency for size.

Construction counters distinguish endpoint interval entries visited on the first
guard construction request, constructed predicate roots, descriptor allocations (including
rejected candidates), and actual selected-forest statement constructors.
`materialized_control_statements` includes sequence wrappers but excludes the
existing final CSE reconstruction and original payload-tree reconstruction.
Parser map-key copies and the arithmetic simplifier's internal work are not
measured by these counters. Production timings and diagnostic construction
inventories are reported separately. Single runs do not establish a speedup.

## Results

All 477 bounded ordered payload/command traces across 28 fixtures match the
starting checkpoint exactly. Both versions generate C++ for all 28 fixtures.
The full synchronization gate passes **1,577 tests** (37 warnings). The
independent emitted-C++ campaign completes 15 checks across ten fixtures; six
comparisons retain their existing unsupported status. The receipt distinguishes
this functional build from the final formatting-only rebuild and focused recheck.

| Fixture | SET/WAIT sites, before → after | IR branches | Guard-expression visits | C++ bytes |
| ------- | ------------------------------ | ----------- | ----------------------- | --------- |
| Gate/up H=5120, band 2 | 68/68 → 63/64 | 168 → 147 | 640 → 485 | 42,186 → 39,004 |
| Gate/up H=512, band 2 | 52/52 → 47/48 | 96 → 83 | 352 → 285 | 31,975 → 30,453 |
| Output H=5120 | 17/17 → 17/17 | 27 → 27 | 113 → 85 | 17,599 → 17,307 |
| Nested fixture | 7/7 → 7/7 | 10 → 10 | 66 → 30 | 6,943 → 6,695 |

Unequal static SET/WAIT site counts do not mean unmatched notifications: every
checked execution retains its exact previous command sequence. The original
nested Qwen duplication is now a shared residual body behind its exact union.

For large gate/up, the totals over both layers and all 24 blocks are:

| Modeled scalar work | Before | After |
| ------------------- | ------ | ----- |
| Conditional decisions | 122,692 | 117,858 |
| Comparisons | 125,412 | 127,412 |
| Boolean operators | 2,720 | 9,554 |
| Arithmetic operators | 39,392 | 2,368 |

Union decisions increase some comparison/Boolean counts; fewer decisions alone
would hide that tradeoff. The combined modeled primitive work decreases from
290,216 to 257,192 (11.4%). None of the 477 measured executions increases this
combined count. This finite measurement is not a uniform runtime theorem or a
device speedup. The native instruction cost of a comparison and division is not
equated by a hardware model here.

Rotating trip counts through 1,000,000 retain 9/9 sites and 20 branches. Nested
counts through (1,000,000,1,000,000) retain 7/7 sites, ten branches and 30 guard
expression visits. Parameter magnitude does not cause this scheduler to unfold
visits. Other case partitions can still require different genuine families.

| Established result | Remaining limitation | Paper consequence |
| ------------------ | -------------------- | ----------------- |
| Exact ordered traces survive shared control construction | Qualified original cuts, frame domains and physical plan | Control representation can shrink independently of minimum-demand analysis |
| Bounded access/order, matching and reuse checks pass | Host model and existing native premises | Code size improvements need not add payload order |
| Original-IV lattice comparisons remove quotient work | Proven static lattice; symbolic starts keep the old form | Compact matching formulas also need qualified economical emission |
| No additional analysis/allocator or production AST fallback | Symbolic family unification and phase restructuring remain separate | Retained mathematical compactness alone does not bound all synchronization code |

## Reproduction

Use the repository testing environment and the worktree's Python 3.14 extension.
The pinned PTOAS v0.72 launcher needs Python 3.11 on PATH. See the receipt for
resolved binary hashes and complete commands. All local intensive jobs run
serially; no device experiment is performed.

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
export PTOAS_ROOT="$PWD/build-py314/physical-sync-prototype/ptoas-v072"
PYPTO_TOKEN_PY311=$(dirname "$(readlink -f /home/toni/.local/bin/python3.11)")
export PATH="$PYPTO_TOKEN_PY311:$PATH"
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_control \
  --baseline build-py314/case-sharing/final \
  --output build-py314/direct-control/final --emit-cpp
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_order \
  --physical build-py314/direct-control/final \
  --output build-py314/direct-control/ordering
```

The unchanged native footprint and invocation-entry/return premises remain
separate from host validation. No claim of optimal physical IDs, fewer executed
communications, general symbolic-family unification or device speedup follows.
