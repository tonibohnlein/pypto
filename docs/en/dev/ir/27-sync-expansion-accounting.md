# Synchronization expansion accounting

The remaining overhead has several sources. On the checked small executions,
PyPTO executes one handoff for each cross-pipe minimum cover. That does not imply
minimal static command code or inexpensive control. This audit follows demand
families through logical tokens, allocation and emitted commands to separate
these questions. Production algorithms and outputs are unchanged.

The baseline is `4f5fdfeada1085841de6cb5bfd7637d391a9663e`.
The [receipt](sync-expansion-accounting.json) pins the extension, diagnostic
source, inputs, measurements and reused comparison evidence. This follows
[cut-aware synchronization control](26-cut-aware-sync-control.md).

## Where the representation grows

For Qwen gate/up, band 2, H=5120, the invocation partition has four cases:

| Block range | Segments / payload sites | Internal demands | Crossing families | Exported families | Executed handoffs per selected invocation |
| ----------- | ------------------------ | ---------------- | ----------------- | ----------------- | ----------------------------------------- |
| 0–4 | 8 / 34 | 16 | 23 | 39 | 719 |
| 5–11 | 4 / 17 | 8 | 9 | 17 | 357 |
| 12–18 | 4 / 17 | 8 | 9 | 17 | 357 |
| 19–23 | 8 / 34 | 16 | 23 | 39 | 719 |

The audit compares the complete multiset of exported mathematical families with
families parsed from the actual logical token program and those supplied to the
physical allocator. Original endpoint coordinates and repetition axes agree.
A singleton Cartesian axis contributes zero displacement and is normalized away;
other axes and family multiplicity remain. There is no family multiplication
between these layers. Every family in these Qwen cases has one assigned ID.

Across the mutually exclusive cases there are 112 family instances. Logical IR
has 29 `sync.record`, 112 `sync.wait`, 110 `sync.completed` calls and ten token
carries. Records can share a source cut. Completed tokens are SSA placeholders,
not hardware commands. `OptimizeSync` leaves this inventory unchanged.

Physical IR contains 86 SET sites, 76 WAIT sites and 234 branches: 232 inserted
guards and two original conditions. The largest guard has seven expression
nodes; there are 61 distinct guard expressions. The problem is repeated small
controls and case-specific command sequences, not one enormous Boolean formula.

The source trace is:

- `invocation_sync_analysis.cpp`, `AnalyzeSyncInvocation`: partition the
  invocation and retain each case's analysis.
- `scoped_logical_sync.cpp`, `Append`: transport scoped families into token IR.
- `token_sync.cpp`, `OptimizeSync`: verify logical synchronization; currently
  perform no optimization.
- `structured_token_sync.cpp`, `LowerStructuredLogicalSync`: parse actual tokens
  and allocate each admitted case independently.
- `physical_sync_realization.cpp`, `ScopedGuards`, `Commands`, and
  `EmitAssignedSyncCases`: construct endpoint guards, literal-ID alternatives
  and complete ordered command sequences at original cuts.
- `EmitSyncCommandCases`: share identical complete sequences over qualified
  cut domains. Common subsequences of different sequences are not shared.

Different legal case-local assignments can inhibit sharing. For example, the
same encoded source endpoint at original statement ordinal 65 uses ID 1 in
blocks 0–4 and 19–23, but ID 0 in blocks 5–11. Its direction is 0→3 and its
encoded volume is 40. The census finds 78 endpoint descriptors before including
IDs and 102 after including IDs. These counts omit case predicates and paired
endpoints: **78 is neither a proved lower bound nor a safe merged plan**.

## Executed control and the other kernels

For gate/up band 2 H=5120, both checked endpoints of the launch domain
(block 0/layer 0 and block 23/layer 1) execute:

| Condition dependency | Evaluations | False outcomes |
| -------------------- | ----------- | -------------- |
| Launch case only | 2,882 | 1,921 |
| Iteration coordinates | 1,996 | 965 |
| Original conditions | 80 | 78 |

Launch-only tests are 58.1% of all 4,958 evaluated conditions. An invariant
predicate can still be tested repeatedly even when its scalar value is shared.
Dependency classification is descriptive; it is not permission to move a
command, evaluate a skipped expression or hoist across an invocation reset.

| Fixture | Family instances across cases | Static SET / WAIT | Branches including original control |
| ------- | ----------------------------- | ----------------- | ----------------------------------- |
| Matmul / accumulation | 4 / 10 | 4/4; 10/10 | 0; 0 |
| DDR split-K | 15 | 15 / 15 | 18 |
| Vec / softmax | 3 / 6 | 2/2; 2/2 | 0; 0 |
| Rotating, 7 visits | 3 | 9 / 9 | 20 |
| Crossings | 7 | 14 / 14 | 36 |
| Collective / nested | 2 / 7 | 2/2; 7/7 | 4; 10 |
| Library matmul / GEMM | 5 / 7 | 5/5; 7/7 | 0; 8 |
| Qwen output H=5120 | 17 | 17 / 17 | 27 |
| Gate/up bands 0,1,2,3 H=5120 | 73,90,112,90 | 60/55; 66/55; 86/76; 75/71 | 159,171,234,207 |

Vec and softmax also retain one and four same-pipe barriers respectively;
family counts therefore need not equal flag pairs. Static counts are not
executed counts. In particular, the earlier device report's 73–112 versus 46
comparison counted static sites, not per-invocation communications.

Qwen output has 26 iteration-dependent guards and one original condition. Its
checked active execution evaluates 868 iteration tests and 40 original tests.
Case sharing alone will not address this overhead. The rotating fixture assigns
three IDs per family, explaining its expansion from three families to nine
SET and nine WAIT sites. Its selected execution evaluates 14 iteration-only
and 66 iteration/modulo tests. This is a distinct selector/phase problem.

Rotating trip counts 7, 70, 7,000 and 1,000,000 retain three families, nine
SET/WAIT sites and 20 branches. Nested counts (2,3), (2,1000), (1000,3) and
(1000000,1000000) retain seven families, seven SET/WAIT sites and ten branches.
Compactness in trip count holds for these fixtures; it does not bound growth
in static stages, partitions or independently varying interfaces.

## Ordering and necessary communications

All 28 accounting cases complete. Twenty-one small executions independently
reconstruct accesses, native order, required closure and minimum covers from
original IR, then check the emitted commands. Matching closes, the graph is
acyclic, causal flag reuse holds and payload order equals the selected required
order. Executed SET/WAIT pairs equal cross-pipe covers. This supports the current
direct protocol; it is not a lower bound over every possible protocol.

The previous generated-C++ ordering campaign remains applicable: production
sources are unchanged and input/source hashes match. Fifteen existing host trace
binaries were freshly reexecuted for counts. The ordering proofs below are reused
from that campaign, not new device measurements. Representative executions:

| Fixture | Executed pairs, PTOAS → PyPTO | PTOAS internal barriers | Excess completion-to-start pairs, PTOAS → PyPTO |
| ------- | ----------------------------- | ----------------------- | ----------------------------------------------- |
| Matmul / accumulation | 4→4; 10→10 | 0 | 0→0 |
| DDR split-K | 25→19 | 12 | 21→0 |
| Library matmul | 5→5 | 1 | 1→0 |
| Library GEMM | 28→22 | 0 | 0→0 |
| Qwen output H=512, active worker | 45→33 | 21 | 32→0 |
| Gate/up H=512, band 2 | 93→71 | 42 | 202→0 |
| Gate/up H=512, other bands | 51→33 | 21 | 56→0 |

Pair-set comparisons establish equality or removal without added prerequisites
on these executions, not merely smaller totals. The metric is the draft's
completion-rank profile K(b,p), checked against explicit excess pairs under its
prefix premise. PTOAS additionally executes one return ALL drain in each checked
trace; that drain is separate from internal payload order. Its absence in PyPTO
remains an invocation-boundary qualification question. PTOAS uses an eight-ID
pool and PyPTO six; no minimum physical-ID result follows.

The access oracle is independent of compiler demand generation but assumes the
qualified operation footprints, layout and native issue model. Large H=5120
cases receive schema and control checks, not a full expanded independent
command-graph proof here. Six earlier C++ comparison cases remain explicitly
unavailable in the prior receipt. No new device or C++ generation campaign ran.

## Cost, reproducibility and next implementation

The receipt separates ordinary `RealizeSync`, standalone Insert/Optimize/Lower
calls and diagnostic reconstruction. These are different runs; do not add them
or count audit reconstruction as production work. Timings are single-run host
observations, not speedup evidence. Independent oracle expansion is outside the
production timers. The diagnostic recreates analysis from the pinned input; it
does not inspect the actual production owner's private storage.

| Fixture | Ordinary production, ms | Standalone Insert / Optimize / Lower, ms | Separate audit, ms |
| ------- | ----------------------- | ---------------------------------------- | ------------------ |
| Rotating, 7 visits | 1.42 | 0.44 / 0.05 / 0.85 | 0.84 |
| Qwen output H=5120 | 5.06 | 1.52 / 0.31 / 3.60 | 4.02 |
| Gate/up band 2 H=5120 | 36.35 | 6.75 / 1.27 / 30.62 | 36.22 |

Lowering includes token interpretation, allocation and emission; these timings
do not isolate the allocator or prove which internal operation dominates.

Reproduce with the current worktree's built Python 3.14 extension:

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_accounting \
  --output build-py314/sync-accounting
/usr/bin/python3 -m pytest tests/ut/ir/transforms/test_sync_cut_control.py -q
```

The 11 focused tests pass, including singleton-axis normalization and executed
condition attribution. Ruff and focused type checks pass. The earlier 1,551-test
gate is prior evidence, not a fresh full-suite run for this diagnostic change.
Full family inventories remain in the disk-backed accounting artifact; the
checked-in receipt retains its hash and per-case summaries.

The next milestone should optimize the existing token-to-command pipeline:

1. Retain case alternatives and endpoint identity long enough to share common
   decisions and ordered command subsequences at a legal cut. Start with
   identical assigned commands; do not require a new allocation strategy.
   If logical waits are factored in `OptimizeSync`, preserve token capture,
   guarded availability and occurrence identity in the actual token program.
2. Evaluate case-local ID renaming only as a separate certified opportunity.
   A consistent directional bijection must preserve all uses and reservations;
   independently renaming one endpoint is invalid. The census is not that proof.
3. Address repeated first/last and residue tests with qualified phase decisions,
   including zero, short and tail cases. Avoid cloning whole payload bodies to
   eliminate an invariant test. Keep unsupported cases on the existing path.
4. Gate each change on ordered command traces (or a separate justified protocol
   equivalence), full prerequisite inclusion, matching and causal reuse. Measure
   static sites, value definitions, evaluated false tests and construction work.

This targets duplication around an already compact, exact demand description.
It does not justify removing required handoffs, adding barriers for code size,
or claiming that all remaining guards are necessary.
