# Cut-aware synchronization control and ordering quality

This follows [compact synchronization control](25-compact-sync-control.md).
The baseline is that reviewed, uncommitted checkpoint on
`3f437376e4188a31e5ddf5dc5e284dc99a9d742b`, not the bare commit. The
[receipt](cut-aware-sync-control.json) pins both dirty snapshots, the extension,
inputs, commands and results. No allocation, demand, token, hardware exemption
or synchronization protocol changes are made.

## The representation problem

Minimum demands, small command code, and low executed control cost are separate
objectives. A minimum demand can have a complicated executable selector. Several
records can also repeat the same small guard. Counting commands or distinct
expressions alone misses both problems.

Here the dominant remaining duplication was in physical case lowering. A case
had a block interval and a complete command sequence, but its cut did not carry
the execution domain already established by the retained region tree. The
emitter consequently repeated case tests even across workers where the original
cut could not execute. Normalized iteration quotients were also reconstructed
in several conditions at the same cut.

The repair is shared by production token lowering and diagnostic realization
in `src/ir/transforms/utils/physical_sync_realization.cpp`. It does not recognize
Qwen or analyze another candidate representation.

## Contract and validity

Each original before/after payload cut has a **possible execution domain**,
represented by ordered, disjoint closed block intervals. This is an upper bound:
unknown means the full admitted invocation domain. Only the existing retained
zero-trip repetition proof subtracts an interval. A missing payload site or an
empty command inventory is not absence evidence.

An original structured-statement traversal carries these facts to cuts. It
avoids traversing payload expression DAGs. Case coalescing compares neighboring
complete command sequences, preserving SSA identity. It may bridge a numeric
gap only if the possible execution domain does not intersect that gap. An
executing cut with no commands still prevents the merge. Before and after cuts
remain separate; no command crosses a payload or original control boundary.

For each execution that reaches a cut, exactly the original case's command
sequence is selected. Changes on proved-absent executions cannot run. Equality
of complete ordered sequences preserves commands, multiplicity, directions and
physical IDs. Structural comparison failure means retain the original cases.

Two subsequent local steps reduce scalar work:

1. PyPTO's existing arithmetic analyzer simplifies emitted conditions using
   original constant, positive-step loop ranges. It changes no payload control
   and assumes no new dynamic bounds. Unsupported comparison retains the guard.
2. Scope-local value numbering shares repeated INDEX comparisons and normalized
   variable/positive-constant quotients. It admits only strict, pure INDEX/BOOL
   trees; calls, short-circuit expressions and unsafe divisors are excluded.
   A value is defined at its first actual evaluation, inside the same guard and
   visit. Branches have separate tables. No calculation moves to a loop preheader
   or escapes a branch. Names are allocated in original source order, including
   after serialization, rather than pointer-map order.

The implementation uses existing structured IR, `arith::Analyzer`, structural
hash/equality and the synchronization name allocator. It is deliberately not
general CSE or speculative code motion. Hash collisions and unavailable
comparison lose sharing without invalidating an accepted synchronization plan.

Consequently the transformation preserves the exact ordered command trace.
Its correctness does not rely on a lower excess-prerequisite count or a new
physical allocation proof. Existing native instruction and invocation-boundary
premises remain separate.

## Measure order before comparing code size

`sync_order_metrics.py` consumes independently constructed payload closures.
For each target payload start and source pipe, it computes the draft's greatest
required completion rank, `K(b,p)`. It verifies the completion-prefix premise
before equating the sum of rank differences with the number of excess
completion-to-start pairs. Full event-pair coverage is checked separately.

Comparisons retain actual added and removed prerequisite pairs. Equal excess
counts can describe incomparable plans; a lower count alone proves no dominance.
Tests exercise both cases. A trailing ALL drain is counted separately: it may
matter for invocation completion without adding an internal payload prerequisite.

`sync_cpp_order.py` extracts the control of supported generated C++ into a host
trace, preserving scalar ABI types and signed/unsigned promotions. Its command
graph checks actual SET/WAIT/barrier placement, matching, acyclicity and causal
consumption before republication. It does not insert reuse edges to make a plan
pass. Priming SETs are supported; ALL never consumes outstanding notifications.

Payload effects come from the original-IR access oracle, not compiler demand
records. The macro sequence must match the original IR. This is independent of
the demand reducer, but assumes the qualified macro lowering, layout, native
pipe and scalar command-issue contracts. It does not prove arbitrary C++ payload
equivalence or native correctness. Unsupported exports and trace syntax remain
explicitly unavailable.

## Results and cost accounting

The full synchronization gate passed **1,551 tests** (36 deprecation warnings);
the final focused gate passed **30 tests**. The source reviewer accepted the
implementation, command-graph checker and report. Type, format and documentation
checks pass. All 28 original input hashes match the prior checkpoint. All **477**
exact ordered traces agree over both
layers and every block of the Qwen fixtures, plus the earlier bounded kernels.
Nine small cases also compare independently constructed before/after payload
orders. C++ generation is exercised on both versions of all 28 cases.

| Fixture | Physical IR branches, previous → current | C++ bytes, previous → current |
| ------- | ---------------------------------------- | ----------------------------- |
| Qwen output H=5120 | 27 → 27 | 17,499 → 17,599 |
| Gate/up band 0 H=5120 | 187 → 159 | See receipt |
| Gate/up band 1 H=5120 | 242 → 171 | See receipt |
| Gate/up band 2 H=5120 | 312 → 234 | 53,460 → 46,104 |
| Gate/up band 3 H=5120 | 242 → 207 | See receipt |
| DDR split-K | 18 → 18 | 11,440 → 11,361 |

For gate/up band 2, layer 0 across all 24 blocks, evaluated conditions decrease
from 109,734 to 82,466. Condition-expression visits decrease from 464,146 to
348,102, while scalar-value expression visits increase from 3,154 to 12,514.
Their combined structural count falls 22.8%; the 12,188 executed SETs are
unchanged. For Qwen output, branches are unchanged and C++ grows by 100 bytes,
but the same combined structural count falls from 94,080 to 79,680. Sharing is
therefore not a claim of uniformly smaller source or fewer temporaries. Nested
fixtures also grow slightly in source bytes (for example, 6,898 → 6,943), while
keeping the same commands and branch sites.

These counters count source expression occurrences, including syntactically
present short-circuit operands. They are not hardware instruction counts.
Scalar definitions and evaluations are reported alongside predicates so moving
work into a temporary cannot masquerade as eliminating it.

The independent PTOAS comparison checks ten supported cases. Basic matmul,
accumulation and library GEMM have equal prerequisite profiles. On the checked
DDR split-K, library matmul and Qwen executions, PyPTO removes prerequisites
without adding others. The receipt retains individual profiles and witnesses,
including empty boundary workers. Six other attempted cases have explicit
export/trace-adapter limitations; these are not six compiler failures. PTOAS's
eligible pool is eight and PyPTO's is six, so this is not a minimum-ID comparison.

Rotating counts through 1,000,000 retain 20 branch sites and 9 SET/WAIT sites;
nested counts through (1,000,000, 1,000,000) retain 10 branches and 7 SET/WAIT
sites. The zero-trip rotating case keeps its original transactional delegation.
Generic domain tests also vary the launch bound through 7,000,000 without
enumerating blocks.

Construction counters separate propagated interval pieces, adjacent case
comparisons, coalesced cases, predicate simplifications, value sharing and
expression inspections. Charge structural comparison/hash traversal and existing
arithmetic analysis separately; `control_expression_visits` does not count those
libraries' internals. Nested domain transport depends on propagated interval
pieces, not only distinct source nodes. No uniform linear bound in source size
or distinct DAG nodes is claimed. Production time excludes the separately
requested diagnostic analysis. Single-run timings are descriptive, not speedup
evidence.

## Reproduction

Use the pinned Linux extension and PTOAS v0.72; do not source the stale macOS
`env.sh`. The launcher needs Python 3.11 on PATH; the extension uses Python 3.14.
Run serially when another intensive job occupies the other machine worker.

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
export PTOAS_ROOT="$PWD/build-py314/physical-sync-prototype/ptoas-v072"
PYPTO_TOKEN_PY311=$(dirname "$(readlink -f /home/toni/.local/bin/python3.11)")
export PATH="$PYPTO_TOKEN_PY311:$PATH"
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_control \
  --baseline build-py314/guard-control/final \
  --output build-py314/cut-control/final --emit-cpp
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_order \
  --physical build-py314/cut-control/final \
  --output build-py314/cut-control/ordering
```

The baseline artifacts and source snapshot are disk-backed campaign inputs.
Recreate them from the prior checkpoint using its driver and extension before
building this change. The receipt pins their hashes. Its full-gate command lists
all files explicitly; no timeout or expectation is relaxed. An earlier gate
attempt using the wrong PTOAS launcher was stopped and is recorded separately.

## Paper consequences and remaining work

| Established result | Limitation | Appropriate claim |
| ------------------ | ---------- | ----------------- |
| Bounded profiles and pair inclusion are checked independently | Qualified access/native model; finite executions | Ordering quality is measured, not inferred from command counts |
| Case and local-value optimization preserves complete ordered traces | Existing realization and allocation premises | Control can shrink without changing accepted demands or physical protocol |
| Retained loop counts do not enlarge these tested emitted schemas | Static stages, interval pieces and interfaces can grow | Trip-count compactness is not a universal code-size bound |
| Source and evaluated control costs are reported separately | No new device experiment | No device speedup or invocation-return proof follows |

Changing repeated command families, reducing tokens, fusing different protocols,
or choosing a different ID assignment would need separate matching, order and
reuse arguments. This checkpoint leaves those decisions unchanged. The remaining
guards distinguish different commands or domains, or fall outside the present
proof rules; they are not all proved necessary. Broader control minimization is
a secondary optimization objective with its own representation and cost bounds.
