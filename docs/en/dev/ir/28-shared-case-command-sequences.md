# Shared case decisions and command sequences

This implements the first target from
[synchronization expansion accounting](27-sync-expansion-accounting.md): reduce
case-selection duplication while retaining the selected protocol and assigned
physical IDs. The baseline is `4f5fdfeada1085841de6cb5bfd7637d391a9663e` plus the
uncommitted accounting diagnostic. The [receipt](shared-case-command-sequences.json)
pins both starting and final source contents, tools and inputs.

## Placement and contract

The change lives in `physical_sync_realization.cpp`, inside the existing
`EmitSyncCommandCases` used by production token lowering and diagnostic physical
realization. It consumes already assigned command sequences at a single original
before/after payload cut. It does not rerun demand analysis or allocation, change
logical token semantics, or introduce a new synchronization pass.

Inputs retain the existing emitter-only contract: physical command leaves,
sequences and guards over ready immutable original coordinates. They contain no
payloads or definitions. The original cut's possible execution domain is a
conservative upper bound; unavailable absence evidence means the full domain.
An executing case with no commands is distinct from a proved-absent case.

The existing whole-sequence coalescer runs first. The new construction then:

1. Flattens only unconditional sequence containers, retaining guarded commands
   as opaque trees with original SSA identities.
2. Creates ordered case slices, including empty slices for uncovered executing
   intervals. It omits intervals only when they cannot reach the original cut.
3. Builds a balanced binary decision tree over these slices. A shared prefix
   executes before a decision, and a shared suffix after it. Their internal order
   and duplicate commands are preserved. Children process the remaining slices.
4. Compares the candidate with the previous emitter output. It accepts only if
   command sites, condition sites and expression occurrences do not increase,
   and at least one decreases. Otherwise it retains the previous output.

For example, `[A,B,D]` versus `[A,C,D]` becomes `A; if case: B else: C; D`.
`[A,B]` versus `[B,A]` remains two ordered alternatives. Equal labels alone do not
permit exchanging commands, merging publications or renaming IDs.

## Validity argument

For any admitted execution reaching the cut, exactly one interval slice applies.
The binary comparison selects its subtree. At each node the factored prefix and
suffix are structurally identical in every descendant slice; removing one copy
from each and executing one shared copy preserves the selected ordered sequence.
The induction terminates at the original remaining leaf sequence.

Empty executing slices prevent factoring a command into a path that originally
issued none. Missing intervals may be skipped only outside the supplied possible
domain. Original before and after cuts are separate, and no command crosses a
payload or original control boundary. Inner conditions are not moved out of their
own guard trees; a skipped coordinate expression remains skipped. Structural
comparison preserves free-variable identity and unsupported equality loses
sharing rather than invalidating the accepted plan.

Thus every execution retains the same commands, directions, IDs, multiplicities
and positions relative to payloads. The established matching and causal reuse
proofs apply to the same protocol. This does not establish an absent native
invocation-return premise or permit arbitrary effectful control code as input.

## Work and limits

Let C be the number of interval slices, S the supplied top-level command-sequence
entries and T their expanded statement/expression size. Balanced recursion has
O(log(C+1)) depth. Each level inspects at most O(S+C) sequence entries; structural
comparison of opaque trees is charged separately, conservatively O(T log(C+1)).
Interval intersection uses indexed lookup in the normalized possible domain.
No dynamic loop occurrence or launch index is enumerated. Construction retains
O(S+C) sequence/tree storage plus borrowed original expression trees.

Counters include candidate attempts even when rejected: `case_sequence_visits`,
`case_sequence_comparisons`, and `factored_case_cuts`. Size-accounting expression
traversals contribute to `control_expression_visits`; structural-comparison
library internals remain separate. Static Pareto acceptance is a code metric,
not a proof of lower runtime cost for every path or a device timing model.

This shares prefixes and suffixes at decision-tree nodes. It does not solve a
longest-common-subsequence problem, harmonize case-local ID assignments, share
across different payload cuts, or phase-split loops. `OptimizeSync` continues to
verify the logical tokens; this change optimizes their existing physical lowering.

## Results

All 28 fixture inputs retain their baseline hashes. The campaign compares 477
ordered payload/command traces, including both layers and all blocks of each
Qwen fixture. All agree exactly, including IDs and command multiplicity.
The production and diagnostic routes use the same implementation.

The full synchronization gate passes **1,562 tests** (37 warnings); the final
focused run passes **39 tests** (five warnings). Production size bounds were
tightened after full-gate collection, and the focused run validates those final
assertions. Ruff, focused type checks, clang-format21, cpplint, Markdown and
documentation parity/navigation checks pass.

| Gate/up H=5120 band | Static SET / WAIT, before → after | Branches, before → after | C++ bytes, before → after |
| ------------------- | --------------------------------- | ------------------------ | ------------------------- |
| 0 | 60/55 → 51/51 | 159 → 111 | 38,150 → 34,639 |
| 1 | 66/55 → 51/51 | 171 → 115 | 39,723 → 35,260 |
| 2 | 86/76 → 68/68 | 234 → 168 | 46,104 → 42,186 |
| 3 | 75/71 → 63/63 | 207 → 149 | 42,916 → 40,003 |

Across both layers and all 24 blocks, band 2 evaluates 164,932 → 122,692
conditions. Combined condition/scalar-value expression occurrences fall from
721,232 to 464,752 (35.6%). Actual commands do not change. At block 0/layer 0,
719 SET/WAIT pairs remain, while conditions decrease from 4,958 to 3,038; at
block 23/layer 1, they decrease from 4,958 to 3,998. The path asymmetry is retained
in the receipt rather than hidden behind the aggregate.

Dependency attribution confirms the intended mechanism: iteration-dependent
tests stay at 1,996 in both executions. Launch-only tests fall from 2,882 to
962 at block 0, and to 1,922 at block 23. Across all 477 measured executions,
neither condition evaluations nor condition-expression visits increase.

Qwen output, earlier finite kernels, rotating-ID and nested-loop examples retain
their previous sites; their control problem is outside launch-case sharing.
Rotating counts through 1,000,000 and nested counts through
(1,000,000,1,000,000) retain their previous representation size. Synthetic tests
vary 8/32/128 cases with million-wide intervals: shared prefixes/suffixes yield
C+2 command leaves and C−1 decisions, without enumerating the launch domain.

Generated C++ is checked for both before and after versions of all 28 fixtures.
The independent access/order checks in the control campaign and the C++ ordering
campaign distinguish unchanged prerequisite relations from static code reduction.
The draft's excess-prerequisite metric is unchanged; no additional payload
ordering is traded for smaller code. Native footprints, scalar issue and
entry/return assumptions remain separately qualified. No device experiment ran.

Single-run compiler times are descriptive. Diagnostic reconstruction and semantic
oracles are outside the production timer. The band-2 diagnostic processes 1,444
case-sequence/size visits and 114 opaque sequence comparisons, accepting 30 cuts.
The construction introduces no new mathematical analysis or physical assignment.

## Reproduction

Use the existing Linux build, Python 3.14 extension and pinned PTOAS v0.72.
The PTOAS launcher needs its qualified Python 3.11 on PATH. Build serially when
other machine work may occupy the second worker.

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
export PTOAS_ROOT="$PWD/build-py314/physical-sync-prototype/ptoas-v072"
PYPTO_TOKEN_PY311=$(dirname "$(readlink -f /home/toni/.local/bin/python3.11)")
export PATH="$PYPTO_TOKEN_PY311:$PATH"
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_control \
  --baseline build-py314/cut-control/final \
  --output build-py314/case-sharing/final --emit-cpp
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_order \
  --physical build-py314/case-sharing/final \
  --output build-py314/case-sharing/ordering
```

The receipt records the exact regression-gate command and results. Generic tests
cover empty/missing cases, domain holes, original SSA identities, unavailable
skipped values, duplicate commands, opposite command orders and repeated visits.
A deterministic generated family adds 100 sequence/domain combinations.
Baseline serialized IR and source hashes remain in the disk-backed campaign.

## Paper consequences

| Established result | Limitation | Appropriate claim |
| ------------------ | ---------- | ----------------- |
| Per-cut case factoring preserves each ordered command trace | Ready immutable coordinates and command-only bodies | Control representation can shrink without changing demands or protocol |
| Bounded command graphs preserve coverage, matching and reuse | Qualified host access/native model | Smaller static code does not require added payload order |
| Static and evaluated control decrease on the Qwen case family | No device run; other guard sources unchanged | Secondary control-code optimization complements compact demand analysis |
