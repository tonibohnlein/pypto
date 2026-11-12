# Compact synchronization control

This checkpoint reduces repeated control in the existing physical emitter. It
preserves the ordered command trace, including physical IDs, rather than
recomputing demands or dropping token uses. It applies to both direct diagnostic
realization and production lowering of actual SSA tokens.

## What other frameworks contribute

[AIR's concurrency pipeline](https://github.com/Xilinx/mlir-air/blob/main/docs/AIRAsyncConcurrency.md)
separates dependency extraction and transitive reduction from operation
canonicalization. The former changes the dependency representation; it is not
a justification for deleting a consuming SET/WAIT pair from an allocated plan.

[MLIR SCF's `CombineIfs`](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/IR/SCF.cpp)
merges consecutive conditionals with the same condition, retaining the ordered
bodies. This is the closest precedent for our local guard-prefix sharing.
[MLIR CSE](https://mlir.llvm.org/docs/Passes/#-cse) relies on memory-effect evidence;
[canonicalization](https://mlir.llvm.org/docs/Canonicalization/) is best effort
and must not be required for correctness. We reuse PyPTO's structural comparison
and structured IR. No AIR/MLIR dependency or new synchronization backend is added.

General expression CSE, loop-invariant code motion, and reduction of actual token
dependencies remain separate transformations. This change does not speculate
control, hoist calculations, or change the allocator.

## Diagnosis and construction

The Qwen gate/up band-2 fixture at H=5120 had 362 physical branch sites but only
29 distinct guard expressions in the earlier census. The problem was repeated
activation and endpoint tests across already qualified case plans. Each guard
was small (at most seven expression nodes). Flat finite matmul, Vec and softmax
already emitted no synchronization guards. Rotating examples additionally tested
every possible ID residue independently.

The repair is in `physical_sync_realization.cpp`:

1. Each original before/after payload cut retains the complete command sequence
   and its closed block interval. Consecutive intervals merge only when complete
   sequences compare equal, with free SSA variables compared by identity.
   Missing cases are numeric gaps. Activation guards are built after this merge.
2. An ordered mutable tree shares only consecutive equal guard prefixes within
   emitter-created command forests. Leaves retain their order and multiplicity.
   Else-bearing ID decisions are opaque. Before/after cuts are processed
   separately; no command crosses a payload or an original control boundary.
3. Literal-ID selections use `if`/`else if` rather than independent tests. The
   final equality remains, so the change needs no new exhaustive-range premise.
   Every selection remains beneath its original endpoint-presence guards.

Structural comparison failure means **do not merge**, not a new refusal. In
particular, its depth/representation limits do not invalidate an existing plan.
No kernel name, new admission limit, loop expansion or alternative coloring is
involved. The test-only `share_sync_command_guards` binding exercises the same
internal helper; it is explicitly not an arbitrary-IR optimization pass.

## Validity and cost

For an admitted invocation, one original case is active. Replacing identical
complete sequences on adjacent domains by the same sequence on their union
preserves the ordered commands at that cut. Distinct domains with no commands
between them are not merged through that gap.

For prefix sharing, induction on the ordered tree preserves the sequence of
active leaves. Generated predicates read immutable, ready coordinates; command
leaves neither redefine those values nor have their own results. Equal printed
variable names are insufficient. Coactive duplicate commands remain duplicates.
Residue equality predicates are mutually exclusive, so skipping later tests
following a match cannot skip another active command. The final test is retained.
Thus modeled payload order, matching and causal ID reuse are unchanged whenever
the original plan established them. Native instruction and return-boundary
qualification remain the original, separate premises.

Construction scans input control occurrences and freezes each output vector
once. Whole-case comparisons are between neighboring sequences, never all
pairs, and never compare a growing concatenated body. Charge structural
comparison traversal separately; no bound purely in distinct expression DAG
nodes is claimed. `control_statement_visits` counts input-forest inspections,
excluding comparison and freezing work. `constructed_guard_sites` and
`shared_guard_sites` describe per-case construction; `case_sites` precedes
cross-case coalescing. Top-level invocation counters now census final emitted
sites. `if_sites` includes original payload control; `guard_sites` excludes it.

## Results

Baseline: `3f437376e4188a31e5ddf5dc5e284dc99a9d742b`. The machine-readable
[receipt](compact-sync-control.json) pins the dirty source, extension, inputs,
commands and bounded results. All 28 input hashes agree with the baseline.

| Case | Physical IR branches before → after | PTO arithmetic before → after |
| ---- | ----------------------------------- | ----------------------------- |
| Matmul / accumulation / Vec / softmax | 0 → 0 | Unchanged |
| DDR split-K | 19 → 18 | 50 → 48 |
| Rotating, 7 visits | 20 → 20 | 76 → 76 |
| Two rotating regions | 36 → 36 | 111 → 111 |
| Nested / collective / library GEMM | Unchanged | Unchanged |
| Qwen output H=5120 | 27 → 27 | 93 → 93 |
| Qwen gate/up band 0 | 237 → 187 | 547 → 419 |
| Qwen gate/up band 1 | 292 → 242 | 691 → 563 |
| Qwen gate/up band 2 | 362 → 312 | 742 → 614 |
| Qwen gate/up band 3 | 292 → 242 | 693 → 565 |

All **245** before/after bounded command traces agree exactly, including ordered
payload positions, directions and IDs. Nine executions additionally run the
independent original-access, closure, cover, matching and reuse checker. The
broader gate also exercises small Qwen invocation boundaries and both layers
through the existing independent token/physical command-graph checks. Large
Qwen control traces alone are not independent access proofs.

For gate/up band 2, a two-tile execution decreases branch evaluations from
7,364 to 6,562, with false tests falling from 4,568 to 3,766. Both versions execute
719 SETs and 719 WAITs. Across all 24 checked blocks, conditions fall from
123,368 to 109,734 (11.1%). Rotating 7 visits falls from 122 to 80 conditions;
70 visits falls from 1,382 to 962. `condition_nodes` counts syntactic expression
occurrences in reached conditions, including potentially short-circuited
operands; it is not an instruction count.

The final C++ comparison succeeds for 28 old/new pairs, with downstream
InsertSync enabled. Gate/up band 2 retains the 362 → 312 branch reduction and
source size falls from 57,869 to 53,460 bytes. Its static SET/WAIT sites become
99/94 from 112/112: cases share publication and consumption sites differently.
Executed matching is unchanged; static site counts need not be equal.

These are source/control improvements, not a device-speed result. Single-run
production timings are descriptive: gate/up band 2 measured about 12 ms before
and 16 ms after. The normalization adds host work. No compilation speedup is
claimed. All comparisons use the same physical protocol and eligible pool.

Retained static sizes remain stable as rotating counts increase through
70, 7,000 and 1,000,000 and nested counts reach (1,000,000, 1,000,000). The
zero-trip rotating fixture retains its existing transactional delegation; the
repair does not claim new empty-ownership support.

The complete gate passed **1,284 tests** (36 deprecation warnings). After the
formatted rebuild, the focused guard and structural-comparison tests passed
**168/168**. Architecture, correctness and algorithm reviewers accepted the
implementation. Hooks pass after fixing test type narrowing and translated
Markdown alignment; the existing Python 3.11 hook environment supports the
pinned torch dependency, while runtime tests use the Python 3.14 extension.

## Reproduction and remaining work

Use the Linux testing environment, one build/test worker, the pinned PTOAS v0.72
wheel, and Python 3.11 on PATH for its launcher. The extension campaign uses
Python 3.14. Do not source the stale macOS `env.sh`.

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 1
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYPTO_CODEGEN_MAX_WORKERS=1 /usr/bin/python3 -m pytest -q \
  tests/ut/ir/transforms/test_sync_command_control.py
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYPTO_CODEGEN_MAX_WORKERS=1 /usr/bin/python3 -m \
  tests.ut.ir.transforms.qualify_sync_control \
  --baseline build-py314/guard-audit --output build-py314/guard-control/final --emit-cpp
```

The baseline directory contains the pinned pre-edit `.msgpack` captures. To
recreate it, run the same qualification driver with the baseline extension and
without `--baseline`, into a separate output directory. Driver and metric helper
are test infrastructure and can be used against the old Python/extension tree.
The receipt distinguishes ordinary production time from explicitly requested
read-only mathematical/physical diagnostics.

Remaining repeated case guards and scalar arithmetic are visible, especially
where complete sequences differ. Sharing partial sequences across cases or
hoisting invariant predicates needs its own order/availability proof and cost
check. The current work establishes a general, local representation improvement;
it neither completes all guard optimization nor changes the paper's demand,
realization or allocation guarantees. Device timing and the existing native
entry/return premise remain unqualified by this host checkpoint.
