# Regional synchronization schedule implementation

The architecture correction and its sequential acceptance gates are tracked in
[the correction receipt](32-sync-schedule-architecture-correction.md). Stages 1–4
are independently accepted within their recorded scopes.
The earlier campaign results later on this page describe the
pinned earlier implementation, not the corrected default policy.

The fixed catalogue independently combines boundary splitting B, periodic
expansion P, and case versioning C. Identity and C retain the original loop
bounds. B contributes interval facts only; P contributes optional congruence
facts, preserves endpoint guards when B is absent, and owns its required
remainder. Unsupported period proofs exclude P without excluding identity or B.
Boundary-command movement uses one policy in every mode and is proved anew for
each complete candidate. The materializer consumes selected descriptors and
makes no transformation choices.

Production defaults to no additional payload-operation sites. Explicit diagnostic
policies may permit at most 4096 extra payload sites and 65536 extra expanded IR
nodes per function. These finite compiler-policy caps have no device-performance
interpretation. The IR bound counts operand occurrences and cloned scalar
assignments as well as control and commands; shared descriptor objects do not
reduce that bound. Payload growth is debited in original source loop order.
For control nodes, admission charges the full replacement upper bound, using
zero as the guarded baseline lower bound. This is stricter than charging actual
additional growth and remains safe when source sequence wrappers flatten.
Forced modes bypass modeled profitability only, and report an ineligible or
over-budget request separately from its guarded fallback.

Automatic selection compares all admitted complete candidates against the same
guarded incumbent. The six modeled components are arithmetic, comparisons,
Boolean operations, decisions, loop headers, and generated SSA bindings. Case
costs use interval range additions, and predicate costs reuse the emitter's
scope-local sharing eligibility. Unsupported mixed predicates make a saving
unproved. Accepted automatic choices must improve at least one case without
increasing another; equal costs prefer smaller code and fewer transformations.
`model_exact` describes this generated-control model, not device cycles or all
payload work. Independent executed-control measurements remain a separate gate.

C normally duplicates loop bodies. Pure predicate sharing retains one payload
body and is measured separately; it must not be reported as no-clone case
versioning. The diagnostic receipts distinguish inspected candidate/proof work
from selected descriptors and measured emitted output. Actual token domains,
assigned ID maps, and allocation authority remain unchanged.

The starting point is `868fdd471` plus the reviewed boundary-scheduling changes.
`build-py314/regional-schedule/baseline/pins.json` pins those source contents and
the loaded extension. Do not identify this baseline by commit alone.

## Contract

Actual SSA tokens supply the communication requirements. Existing allocation
supplies the certified ID maps. Lowering retains those maps and endpoint domains
until a region-wide schedule is selected. No demand analysis, protocol, native
exemption, eligible pool or storage mapping is changed by schedule construction.

For a qualified repeated body, construct a prologue, steady state, remainder and
tail. A supplied common ID period P permits P specialized visits in each steady
iteration. Every visit still uses its original coordinate and physical storage:

```text
prologue
for q in complete steady groups:
    body(original coordinate q*P + 0, assigned map specialized at residue 0)
    ...
    body(original coordinate q*P + P-1, assigned map specialized at residue P-1)
remainder
tail
```

A phase boundary neither completes payloads nor resets flags. Every matching
identity, publication cut and consumption-before-republication path is preserved.
Unsupported scope, arithmetic or availability keeps the guarded schedule.

## Representation and cost

One internal schedule owner retains original cuts, family indices, domains and
certified ID tables. Cases, phases and residues stay factored; no global LCM or
unbounded Cartesian state enumeration is allowed. Current guarded emission is its baseline
choice. No new public IR, analysis backend or allocator is introduced.

A supplied common period, qualified boundary distances and fixed body give code
size independent of trip count. Payload versions, commands, scalar control and
proof work are charged separately. Compiler DAG sharing is not emitted-code
sharing. Immediate-ID instructions can require several literal sites for one
family even when steady-state dispatch disappears.

An invocation choice can be lifted only to a legal ready scope. Case ID renaming
requires an injective directional permutation over an entire qualified closed
resource scope, preserving eligibility, reservations and all owners. Incompatible
color-sharing patterns cannot be reconciled by renaming.

## Stages and reviewer gates

1. **Shared owner.** Collect symbolic assigned commands before lowering any
   endpoint. Preserve the baseline output, ownership and errors. Review before
   selecting a new schedule.
2. **Regular construction.** Support qualified period-one Qwen boundaries and
   a genuine periodic-ID fixture. Steady regions have no generated boundary or
   ID selectors. Preserve traces and count all payload copies. Review before
   integrating invocation choices.
3. **Invocation integration.** Share qualified common schedules; retain separate
   incompatible cases. Apply optional whole-scope ID permutations only with the
   complete resource proof. Review code and concrete production outcomes.
4. **Consolidation.** Remove superseded per-family expansion and independent
   boundary/case ownership. Keep their useful algorithms as rules of the one
   schedule compiler. Review caller/deletion inventory.
5. **Qualification.** Check actual command/payload order, matching and physical
   reuse independently. Measure construction and emitted source as trips, static
   stages, phases and cases vary separately. Record limits and final acceptance.

## Initial implementation locations

- `physical_sync_realization.cpp`: replace eager family-to-guard construction
  with assigned-command collection and one schedule compiler/materializer.
- `structured_token_sync.cpp`: keep actual-token parsing authoritative and send
  qualified cases to the shared compiler; do not rerun mathematical analysis.
- `physical_sync_allocation.h`: keep allocation and borrowed-call lifetime
  contracts explicit; no coloring change.
- `compact_sync_analysis.h` and testing bindings: expose construction counters,
  including rejected schedules and copied payloads.
- `test_sync_boundary_schedule.py` and schedule qualification drivers: preserve
  boundary regressions and add phase/residue/case structure and semantic checks.

Reviewer acceptance is required before each subsequent implementation stage.

## Implemented ownership and validity

`AssignedRegionSchedule` collects all cases before constructing scalar control.
Its entries reference the actual retained token domains and assigned family ID
maps. Mathematical diagnostic clients decode their endpoint domains once at the
same boundary. The original function and immutable domain handles are owned;
input and allocation inventories are borrowed for this synchronous compilation.

`PlanRegularSchedules` accepts only literal positive-step INDEX loops with no
carried results, allocations, arbitrary scalar definitions or nested control.
Slot-view definitions and an already-qualified shared initialization/accumulation
cut are allowed. All cuts and endpoints of the replaced body must be covered in
every relevant case. Widened checks establish representable original coordinates,
normalized extents, group strides and the final loop increment.

For period one, the endpoint interval boundaries partition the loop. For a
nonunit common physical ID period P, only the longest uniform interval is
residue-expanded; prefix, remainder and tail remain guarded. Different nonunit
periods and scoped nonunit ID maps remain unsupported by this rule. There is no
LCM construction. Literal ID selection uses the original ordinal minus the
endpoint's first ordinal; a new phase never restarts a physical ID map.

Each emitted body is cloned once with a fresh loop/slot SSA frame and captured
allocation roots preserved. Exact per-visit payload and command order is the
validity invariant. It preserves matching, causal reuse and payload ordering
under the same native command model; a phase boundary supplies no completion.

`PreparedSchedulePhase` retains command descriptors before scalar materialization.
The invocation rule groups complete schedules by exact ordered descriptor
comparison; hashes only locate comparison candidates. Domains are exact unions
of case intervals. Empty-command cases still execute their payloads, and holes
are never filled. A dominating original launch value selects the group around
the complete loop schedule. No ID renaming is implemented or required. Selected
leaves retain all their original predicates, including launch predicates, and coordinates; neither
representative input bounds nor allocation evidence are rewritten.

The cost gate charges loop control, coordinate arithmetic and executed decisions
in an explicit unit-work model. Invocation savings are measured against the
actual factored per-cut incumbent, subtracting predicates retained in each leaf.
Unknown predicates contribute no guaranteed savings. A case-interval traversal
uses range additions rather than rerunning the complete tree for every case;
fragmented unions pay for their represented interval pieces. These are host IR
costs, not device latencies. All payload versions and output bytes are reported:
less executed control can increase source size.

This still materializes G distinct schedule groups times their bounded bodies.
For B period-one phases the output bound is O(G·B·(payload+commands)); for the
supported nonunit-period rule it is O(G·(P+3)·(payload+commands)). This charged
product is not eliminated by descriptor sharing. No unbounded joint-period
state enumeration is claimed or used.

## Deletion and caller ledger

| Previous ownership or route | Current replacement | Remaining callers |
| --------------------------- | ------------------- | ----------------- |
| `Emitter` constructs per-family physical statements | `AssignedRegionSchedule` then `AssignedCommandLowering` | Deleted |
| Separate direct emitter and invocation cut preparation | `CompileAssignedSchedule` for both | Both production entry points use it |
| Rebuilt site-to-cut directory in emission | Direct indexed references to owned `AssignedCut` records | Deleted |
| Independently wrapped boundary output | Boundary extraction rule in `ScheduleMaterializer` | One production materializer |
| Per-cut case factoring | Rule used when preparing the same schedule descriptors | Kept as baseline/unsupported-case rule |
| Standalone guard/case/boundary helpers | Testing adapters to those same rules | Testing bindings only; no production dispatcher |

`structured_token_sync.cpp` still parses actual supplied tokens. The allocator
still supplies the same maps. Neither requests new demand analysis to obtain a
more convenient schedule. Unsupported scheduling keeps the accepted guarded
lowering, rather than weakening the selected requirements.

## Qualification checkpoint

All five stages received independent reviewer acceptance. Each earlier stage was
accepted before implementation of the next stage began.
The final qualification receipt is [regional-sync-schedules.json](regional-sync-schedules.json).
Its baseline is the pinned starting dirty tree, not the last device-tested commit.
M1 preserved that baseline's generated files; the differential campaign uses M1
serialized functions to avoid depending on a rebuilt older extension.

The final campaign covers 28 fixtures and 477 executions with identical ordered
payload and physical-command traces. Nine campaign executions additionally run
the independent access/command-graph checker. The focused tests cover periodic
slots, original coordinates, skipped and empty cases, reset visits, same-pipe
hardware exemptions and physical consumption before republication. Actual-token
mutation/refusal tests remain part of the broad gate.

Phase cloning creates fresh SSA names. The independent physical checker compares
each executed original/candidate operand's allocation, base, shape, pitch, layout,
valid type and scalar value, plus operation and attributes. It then builds byte
conflicts from the original input once. It does not accept a compiler generator
or a hash match as evidence of access agreement. Static-site assertions were
revised where short and steady schedules legitimately differ; retained analysis
sizes remain checked, and the matrix exemption test checks executed barriers.

The initial broad gate passed **1614 of 1635** tests. Its 21 failures were two
missing floating-literal cases in the extended operand oracle and nineteen
static-layout expectations from before phase specialization. After the fixes,
the complete ten affected files passed **373/373**, including two new reference
ownership regressions. Every original failing test ID appears in that passing
recheck. This is an initial full run plus an affected-file recheck, not a fresh
all-green full run. Protocol, demand and allocation code were not changed to
address those failures.

Reviewer testing also found that two audit reporters rebuilt reference order
from effect-free cloned-call records. The execution checker itself still used
original effects. Both reporters now use explicitly retained original calls and
native-scope identities; the refreshed campaign supersedes those earlier
ordering rows. The broader accounting audit checks all 28 fixtures and records
its own reconstruction costs separately from production compilation.

The final style-only C++ rebuild additionally passes **104 focused tests**.
All 56 generated C++ outputs, 28 inventories/audits and 477 trace measurements
are identical to the preceding semantic build. The reviewer accepted M5 after
checking the corrected receipts against the current source. Device/cache
performance, native return behavior and unsupported scheduling classes remain
outside that acceptance.

### Evaluated work and generated source

These are representative layer-zero/block-zero host executions. Work is the sum
of evaluated arithmetic, comparisons, Boolean operators, branches, scalar
bindings and loop-header work (two operations per iteration plus two per entry).
It includes existing payload control and is not a hardware latency estimate.

| Fixture | Evaluated conditions, before → after | Total modeled work | C++ bytes |
| ------- | ------------------------------------ | ------------------ | --------- |
| Qwen output, band 3, H=5120 | 908 → 130 | 2175 → 699 | 17307 → 19812 |
| Qwen gate/up, band 3, H=5120 | 1674 → 230 | 3716 → 948 | 38675 → 50013 |
| Three-slot rotating loop, 70 visits | 962 → 66 | 3021 → 353 | 7654 → 21964 |

No checked execution increases total modeled work. Executed communications and
their order are unchanged. Static command sites can increase: they belong to
bounded specialized bodies, not extra executed communications. The source growth
is 14.5%, 29.3% and 187.0% respectively. The current cost rule favors provable
executed-control savings and accounts for body copies; it has no target-calibrated
instruction-cache model. This tradeoff requires device qualification.

The rotating cases at 70, 7000 and 1000000 visits retain the same 33 SET sites,
33 WAIT sites and 48 IR branches, with at most six payload-body versions. Their
single production-lowering measurements are around 4 ms and are descriptive,
not paired speedup evidence. Separate tests vary 1/2/4/8 static stages and
2/4/8/16/32 invocation cases; interval widths reach one million without domain
enumeration. Nested unsupported schedules keep the existing compact guarded
route. Compactness does not mean identical static size at every short trip count.

The emitted-C++ comparison with PTOAS checks 15 executions across ten fixtures.
Six existing checker/export limitations are recorded separately (Vec, rotating7,
crossings, collective, softmax and nested), rather than counted as passes. On the
two checked small gate/up block executions, PyPTO adds no completion-to-start
pairs relative to PTOAS and removes 56; its measured excess relative to the
reference is zero. This is bounded evidence, not a claim about every input.
The existing trailing ALL-barrier/native return premise remains unresolved.

### Reproduction

Run from this worktree with the Linux environment, the pinned PTOAS wheel and
its supported Python on PATH. Build and test serially when another worker is
active. All receipts and generated files are disk-backed.

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
export PTOAS_ROOT="$PWD/build-py314/physical-sync-prototype/ptoas-v072"
PYPTO_TOKEN_PY311=$(dirname "$(readlink -f /home/toni/.local/bin/python3.11)")
export PATH="$PYPTO_TOKEN_PY311:$PATH"
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m pytest tests/ut/ir/transforms/test_regional_sync_schedule.py -q
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_control \
  --baseline build-py314/regional-schedule/m1 \
  --output build-py314/regional-schedule/final --emit-cpp
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_sync_order \
  --physical build-py314/regional-schedule/final \
  --output build-py314/regional-schedule/ordering
```

The receipt includes the full gate command, source hashes, extension hash,
input hashes and result-log hashes. No device experiment was performed here.

### Paper consequences and limits

| Established implementation result | Limitation | Appropriate claim |
| --------------------------------- | ---------- | ----------------- |
| Schedule uses actual token endpoint domains and certified ID maps | No new exact-analysis or allocation construction | Executable representation improvement after demand generation |
| Original ordered traces retained in the admitted phase/case construction | Native pipe and invocation semantics inherited | Same modeled payload order and physical protocol |
| Common-period residue specialization and interval phase splitting | Literal positive INDEX loops; no general mixed-period or scoped nonunit rule | Trip-count-independent control for this qualified subclass |
| Exact interval-union case dispatch | No ID renaming; unequal coordinates/IDs may prevent sharing | Shared complete schedules when identities and availability agree |
| Lower evaluated scalar work in all 477 checked executions | Larger code; no target latency/cache model | Host control-work reduction, not measured device speedup |
| Existing independent cover/order/reuse checks retained | Not a universal proof of native entry/return or arbitrary token lowering | Scoped model-level qualification |
