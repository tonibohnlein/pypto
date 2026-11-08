# Physical realization of retained demands

This opt-in research milestone consumes `CompactSyncAnalysis`; it is not a
general `LowerSync` pass for arbitrary frontend token programs. Default PyPTO
and PTOAS synchronization selection remains unchanged.

## Accepted plan

An independent review on 2026-11-09 accepted the following bounded construction
after reading the draft, repository, current GitHub work and authenticated
taskboard item 3. [Task 3](https://taskboard.pypto.cloud/?task=5276dfa6-98d8-45c9-ad11-5df3af2f78c8)
contains 3.1, still TODO and assigned to Feng Siyuan. Tasks 3.2
and 3.2.1 are done and provide the existing flag and barrier APIs. [PR 3291](https://github.com/hw-native-sys/pypto/pull/3291)
provides logical tokens; hardware lowering is follow-up work. Resource catalogs
([PR 3124](https://github.com/hw-native-sys/pypto/pull/3124)) and macro claims ([PR 3147](https://github.com/hw-native-sys/pypto/pull/3147)) are inputs, not allocation certificates.

1. `retained.realize_physical(target, eligible_ids=None)` consumes original
   occurrence coordinates and retained demands. One cross-pipe demand family
   becomes one matched handoff family. Fanout receives separate handoffs.
   Publications execute only when their selected consumer exists. Final-source
   crossings publish only on the final visit; no terminal cleanup is invented.
2. Same-pipe barriers require an available mechanism and proof that the immediate
   preceding payload's completion is already required. Otherwise this strict
   order-preserving route refuses. It never substitutes an ALL barrier.
3. Finite handoffs use first-reuse thresholds and interval coloring. A reusable
   ID requires an existing consumption-to-publication path, not textual order.
4. Repeated families use certified cyclic lanes. Each lane reserves its full
   first-publication/last-consumption span; other lanes may reuse that ID only
   through an established boundary path. This sufficient compact construction
   does not claim minimum allocation across repeated families.
5. Native target, executor, reserved IDs and macro variants are qualified before
   assigning IDs. Unknown ownership remains unavailable. A caller's optional
   eligible-ID list can restrict the catalog, never enlarge it.
6. Emit literal flag IDs, including guarded residue cases. This avoids the
   documented dynamic-only autosync ownership ambiguity without a dummy flag.
   Check the actual downstream tool separately with automatic insertion disabled.
7. Independently compare emitted commands against original access requirements
   and the existing token realization, including matching, reuse, acyclicity,
   full payload order, zero trips, resets, tails and sparse/reserved pools.

For example, a family `a[i] -> b[i+1]` publishes only for `i+1 < trips` and
waits only for `i > 0`. Both endpoints select the ID for the source ordinal.
With a proved two-lane assignment, the SET uses `i % 2` and the WAIT uses
`(i-1) % 2`. Segment boundaries do not reset physical notification state.

## Guarantees and boundaries

Keep certified success, proved fixed-plan scarcity, and unproved allocation
separate. Failure of a sufficient lane construction does not prove scarcity.
Retained mathematics and original IR survive every refusal. No extra commands
are added to repair scarce resources; no trip-count expansion is permitted.

Logical closure, payload completion at return, physical resource reuse and
native device correctness remain separate obligations. Fresh entry and closed
notification exit are the allocation contract; the surrounding invocation must
provide its original completion/visibility interface. Host command-graph checks
do not establish that native interface or device execution.

Implementation uses the existing registry, mutator, original-cut machinery,
retained numerical queries, resource catalog and flag/barrier code generation.
The finite sweep costs linear-many reachability queries plus sorting. Compact
reservation size depends on static families and eligible hardware capacity,
not on the number of dynamic iterations. Existing query construction costs
remain separately charged.

## Interface and proof obligations

```python
from pypto import testing

retained = testing.analyze_placed_sync(function, True)  # Explicit disjoint-GM contract.
physical = retained.realize_physical("a2a3", [0, 1, 2, 3, 4, 5])
function_with_flags = physical["function"]
```

The optional list restricts each directional catalog pool. `None` uses its legal,
non-reserved IDs after macro claims. Six is a test configuration, not an invented
hardware-wide capacity. Results include original family coordinates, cyclic IDs,
static command/guard counts, reserved lanes, query work and `finite_minimum`.
The C++ entry is `sync::RealizePhysicalSync(const CompactSyncResult&, target, ids)`.

`physical_sync_realization.cpp` consumes immutable retained demands. Both logical
and physical emitters use `sync_placement.h` to locate original cuts, preserve
initialization branches and leave zero-trip bodies untouched. The existing
`system.set_flag`, `system.wait_flag`, `system.bar_m` and `system.bar_v` operators
and backend generate code. Buffer codegen now admits those existing operators
(and the existing explicit `bar_all`) without reconstructing Tile IR.

For a canonical cross-pipe handoff, a consumption-to-republication path is justified by
`C_consumer(h) -> I_source(g)`. The WAIT gates its consumer; the next source's
completion gates its SET. Conversely, with direct cuts, outgoing SETs before
barriers/WAITs, no resource-repair edges, and the native start/completion chains,
a path between those command endpoints must cross the corresponding required
payload order. Same-pipe barriers are admitted only when their immediate prior
payload completion is already required at the target. Thus allocation queries
use the retained required order without adding ordering to manufacture reuse.

Within one direction, publications and consumptions lie on ordered native
chains. Legal reuse targets form a suffix of publication order, whose first
index is monotone in consumption order. Sorting both chains and sweeping obtains
these thresholds with linear-many retained queries. Interval coloring is optimal
for these finite intervals. The diagnostic `Fixed-plan scarcity` concerns only
this fixed, qualified protocol and eligible pool, never all possible protocols.

For repeated families, a retained periodic completion-to-start distance proves
reuse after a fixed lane period. Endpoint guards restrict the proof to existing
occurrences. Reserving each lane's entire first-SET/last-WAIT span is sufficient
for cross-family sharing; it can reject a physically feasible alternative.
`Allocation unproved` preserves that distinction. A segment or counter reset
provides no rearming evidence by itself. Cross-segment sharing uses the same
retained order query, including storage crossings and native seams.

## Native and downstream qualification

The host campaign uses PTOAS **v0.72**, and audits the runtime-pinned PTO-ISA
revision `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7` in manual a2a3 mode.
The admitted AIC variants are TLOAD(Mat), TSTORE(Acc), TMOV/TEXTRACT(Mat to
Left/Right), TMATMUL and TMATMUL_ACC. Their implementation headers have no hidden
AIC flag claims in this mode. Public wrappers use an empty `WaitAllEvents` pack
and return a trivial `RecordEvent`; no Event conversion is emitted. TEXTRACT's
V/S flags belong to excluded Vec overloads. This is a narrow inspected variant
contract, not a rule that missing catalog entries mean arbitrary macros are safe.
Unresolved owners or non-fixed resource bindings remain unavailable.

Compile emitted PTO with `--pto-level=level3 --pto-arch=a2
--enable-insert-sync=false`. Literal flags also make auto-on and auto-off output
identical for the three checked kernels. Tests compare exact direction/ID
sequences and command counts in the generated kernel body. PTOAS emits an unused
`ptoas_auto_sync_tail` helper; these kernels do not call it. Its definition is
excluded from executed-code counts, not assumed to discharge return completion.
Do not enable `__PTO_AUTO__` or `PTO_FLAG_TEST` in a downstream build covered by
this manual-mode audit. No device compilation or execution was performed.

## Reproduction and evidence

Use Python from this worktree's build, with `PTOAS_ROOT` pointing to v0.72 and a
compatible Python 3.11 on PATH for its launcher. The PyPTO build used Python 3.14;
invoke that interpreter explicitly when the launcher's PATH differs.

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 2
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m tests.ut.ir.transforms.run_physical_sync_prototype \
  --native --output build-py314/physical-sync-prototype/campaign
PYTHONPATH=$PWD/python python3 -m pytest \
  tests/ut/ir/transforms/test_physical_sync_realization.py \
  --require-ptoas -n 1
```

The machine-readable receipt pins dirty sources, input/output serialization,
example sources, compiler extension, Python and PTOAS. It records analysis and
physical construction time separately; single runs do not establish speedup.
The independent interpreter derives physical accesses from original IR, constructs
reference conflict/native graphs, checks exact covers and full payload order,
then verifies emitted matching, acyclicity and causal reuse without adding reuse
edges. A separate bipartite-matching oracle computes fixed-plan directional width;
tiny cases also enumerate antichains. These share the declared access/native
model, not the production generator or allocator.

Focused tests cover original matmuls, zero/short trips, 1–3 sequential visits,
initialization/accumulation, final stores, nonzero starts/nonunit steps, 1–4 static
stages, sparse pools, fanout, serialization and repeated requests. Corruptions
with a missing WAIT, wrong ID or textually alternating but causally unsafe reuse
must fail. A finite width-two example refuses one ID as scarcity; a compact
one-ID refusal remains unproved. K scaling up to one million constructs only
static families and bounded lanes; the independent oracle unfolds small cases
only.

| Established result | Remaining premise | Claim |
| ------------------ | ----------------- | ----- |
| Physical commands match logical demands and selected order in bounded checks | Qualified original effects and native event semantics | Direct physical realization of the admitted subclass |
| Finite directional width and sufficient compact lanes checked independently | Fresh entry, explicit eligible pools, original invocation interface | Certified fixed-protocol reuse; qualified finite minimum only |
| PTOAS emits the same explicit flag plan | Caller completion and native implementation contract | Host lowering compatibility, not device qualification |
| Static representation stays bounded as K grows | Static families, available original endpoints | Compact repeated realization, not arbitrary token lowering |

Independent source review accepted the implementation and its generality after
adding causal-reuse and scarcity negatives. No default pass, upstream task or PR
was modified. General frontend token lowering, guarded histories, arbitrary
nested regions, scarcity repair and invocation-completion synthesis remain
outside this milestone.

## Recorded host results

The fresh focused gate passed **639 Python tests and six C++ tests**. All three
original examples generate C++ with PTOAS v0.72; their checked executions have
exact modeled payload order and closed matching, using IDs 0–1 at most in each
direction within the supplied six-ID pool.

| Original example | Static SET / WAIT / barrier | Executed SET / WAIT / barrier |
| ---------------- | --------------------------- | ----------------------------- |
| `matmul_64` | 4 / 4 / 0 | 4 / 4 / 0 |
| `matmul_acc_64` | 10 / 10 / 0 | 10 / 10 / 0 |
| `ddr_split_k` | 15 / 15 / 4 | 19 / 19 / 6 |

The controlled two-visit derivative retains 37 families, 33 reserved lane spans,
60 reuse queries, 33 static SETs and WAITs, four barriers and 41 guards for each
K in 4, 8, 16, 64, 1024 and 1,000,000. Fifteen independent bounded derivative
executions cover zero through five iterations and one through three visits.
These are fixed-stage derivatives, not alternative AutoTile configurations.

The pre-commit rerun also separates synchronization control from original
initialization branches. Both finite matmuls introduce zero scalar operations
and zero executed guard tests. Split-K adds 38 PTO `arith.*` operations before
downstream optimization and executes 38 added guard tests for the pinned input
(19 static guards). These counts are not device instruction or cycle estimates.
Its independent command graph has zero excess payload prerequisites; the
allocator uses two IDs per direction at most, matching the bounded graph widths.
This establishes good modeled ordering and small resource use, not minimum
static command count or globally optimal control code. Fanout coalescing and
control simplification remain possible follow-ups. Single-run retained analysis
and physical construction each take less than one millisecond for these cases;
frontend capture and downstream compilation are recorded separately.

A fresh pre-commit reviewer accepted this scoped conclusion and implementation.
The 639-test and six-C++-test gate was repeated successfully. No device timing,
numerical execution or payload-completion guarantee at return is inferred.
