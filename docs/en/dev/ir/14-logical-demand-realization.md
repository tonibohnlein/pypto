# Logical realization of retained demands

This opt-in research extension consumes `CompactSyncResult` without rerunning
access recovery or reduction. It uses the logical synchronization IR from
PR #3291, pinned at `5cbc3c2ac3c7a0c2be5403ec8adc028b6c1f9573`.
The dependency is adapted to the newer SIMT/property registry on this branch;
it is not a second token interface. Default autosynchronization dispatch stays
unchanged. The user's explicitly requested research insertion is separate from
the repository policy assigning ordinary production synchronization to PTOAS.

## Supported construction

The importer supplies a fixed, placed, single AIC invocation, contiguous finite
regions and nonnested constant-count loops. Every admitted site executes once
per body. Initialization/accumulation chooses between two original calls at the
same site; both have the same pipe. The existing access and alias premises apply.

For each source site needed by a retained demand, emit one `sync.record` after
the payload (after the whole initialization branch, when applicable). Each
consumer receives `sync.wait` before its payload. Several demands can share a
record: tokens are immutable completion facts, not consuming notifications.
All earlier outgoing records remain before a later payload's incoming waits.

Within one iteration, a wait uses the current source token. A distance-one
wait uses a loop-carried token and executes only after the first visit. Carries
start with `sync.completed`, which makes no claim that payload work has completed;
it represents only an empty initial obligation. Cross-region waits use the earlier region's
final exported token and execute on the target's first visit. No crossing is
manufactured for an empty region. Original loop variables, start/step, branch
conditions, payload calls and allocation identities are preserved.

The current boundary adapter accepts final-source/first-target crossings only.
Other occurrence selections must remain unavailable until their transport is
implemented. The mathematical owner is immutable and remains usable on refusal.

## Validity argument and limits

Each record captures its source completion and preceding completions on that
pipe. A wait orders subsequent destination commands after this captured prefix.
Native completion and start chains already imply those surrounding edges from
the selected source-to-consumer demand. Earlier waits included in a record's
prefix are also predecessors of that source, so forwarding them adds only
transitive requirements. In the admitted reference-forward program, the token
definition dominates every current-visit use; a carry supplies exactly the
previous visit, and a loop result supplies exactly the final visit. These facts
establish logical coverage, progress and order equality for the declared token
and payload model. Range-checked loop coordinates make the added guards ready
without private branch history or payload reads.

Same-pipe waits retain the captured prefix, including when it precedes another
same-pipe payload. Lowering such a wait to a full local barrier can strengthen
order and needs its own qualification. No physical flags, allocation, cleanup,
hardware barrier placement or invocation-completion protocol is constructed.
The original invocation's completion obligations remain outside this helper.
PTO code generation must refuse these unlowered tokens.

Construction indexes source/consumer sites and traverses original IR once;
name reservation memoizes shared expression DAGs and reports inspected visits
and distinct nodes separately.
Work is O((N+F) log(N+F)) in input nodes and retained demand records. A source
needs at most one current token and one carry/export. Token sites and guards
depend on static sites and crossings, not iteration count. This does not change
the preceding analysis's documented static quadratic costs.

## Validation contract

The bounded checker interprets the emitted structured SSA program, resolves
token identities through actual branch/loop executions, and constructs a command
graph from operation semantics. It must not insert retained demands into that
graph. It compares its projected payload order with the independently recovered
access/native requirements, and checks that each active retained demand has a
wait on the intended source occurrence. It checks payload/control preservation,
acyclicity, source-pipe agreement and empty initial-token use separately.

Tests cover the three placed matmul examples, empty/single/multiple visits,
initialization and accumulation, original nonunit coordinates, reset boundaries,
aliases, malformed token uses, serialization and refusal before hardware codegen.
Mutation checks remove or misplace commands to establish that the checker detects
missing coverage and extra order. No numerical matmul or device test is implied.

## Host qualification checkpoint

The starting branch revision is `0d0b6517ab5bd3909e567263073ac642ea9829ee`.
[The receipt](logical-demand-realization.json) pins the dirty source files,
logical-IR dependency patch, compiled extension, original inputs and commands.
The measurements identify the pre-commit working-tree snapshot. This combines
the prerequisite IR patch and the new retained-demand consumer; the latter is `logical_sync_realization.cpp` plus the
research binding, checker and campaign extension.

| Original placed input | Payloads | Static record / wait | Executed record / wait | Extra payload event pairs |
| --------------------- | -------: | -------------------: | ---------------------: | ------------------------: |
| `matmul_64` | 6 | 4 / 4 | 4 / 4 | 0 |
| `matmul_acc_64` | 11 | 9 / 10 | 9 / 10 | 0 |
| `ddr_split_k` | 28 | 11 / 19 | 19 / 25 | 0 |

The independent campaign also checks 15 staged executions (zero, one, two, three
and five iterations, each with one to three reset visits) and real AutoTile
K=64/192/256 derivatives. It checks all retained covers, access agreement, all
four event queries and boundaries before interpreting the emitted commands.
The new unit tests additionally cover nonzero starts, nonunit steps, identical
physical local allocations, name collisions, missing original yields and three
command mutations. The checker requires unique original variable names in these
fixtures; it compares names and structure across serialization.

For fixed two-stage/two-visit input, K=2 through 1,000,000 retains **21 record
sites, 37 wait sites and 12 carries**. Only bounded instances are unfolded by
the oracle. Increasing static stages from 1 to 16 gives 11 to 161 records and
18 to 275 waits. Increasing static visits from 1 to 8 gives 10 to 87 records
and 15 to 169 waits. This measures construction size, not device latency.
Single-run realization costs for the three originals were below one millisecond;
the receipt retains exact descriptive timings and separates independent checks.

Use the explicit research API after Buffer IR placement:

```python
retained = pypto.testing.analyze_placed_sync(function, True)
logical = retained.realize()
function_with_tokens = logical["function"]
```

`True` supplies the existing **disjoint GM argument contract**, not an inferred
alias fact. `.realize()` verifies logical schemas and ownership. Qualification
also runs SSA, type, lexical lifetime and Buffer IR verifiers. Reproduction:

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 2
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 60 python3 -m tests.ut.ir.transforms.run_sync_matmul_prototype \
  --logical --output build-py314/logical-sync-prototype/campaign
```

| Established result | Remaining boundary | Appropriate claim |
| ------------------ | ------------------ | ----------------- |
| Original accesses, covers and command graph agree on bounded inputs | Declared footprints and native chains need hardware qualification | Host validation under the admitted access/order model |
| Correct current, previous and final occurrence token selection | General nested, symbolic and data-dependent control is not admitted | Direct logical realization of the retained compact subclass |
| Full projected order equality in checks and the stated prefix argument | Physical waits/barriers may strengthen order | Logical selected-order equality, not a hardware concurrency guarantee |
| Command representation stays fixed with K | Event assignment, token fanout lowering and invocation completion | Compact logical commands; no physical feasibility claim |

The next functional boundary is qualified `LowerSync`: materializing these
immutable completion facts using hardware mechanisms, with separate matching,
resource-reuse, placement and realized-order contracts.

The final focused gate passes **1,163 Python tests and two C++ tests**. It includes
logical IR ownership, Buffer IR, serialization, hashing, SIMT registry compatibility
and codegen rejection. No broad device/system campaign was run.

## Pre-commit review

Independent reviews accepted the logical-IR foundation and the retained-demand
consumer after three fixes: shared-DAG name collection now preserves sharing;
SIMT launches and unresolved split-AIV scopes cannot transport completion tokens;
and a diagnostic export refusal keeps its successful analysis in one outcome
record, while independent-check failures propagate as failures.

Regressions cover each fix and a shared-call anchor refusal that leaves retained
mathematics usable. The reviewers found no kernel-name or configuration-specific
implementation branches. Physical realization remains outside this acceptance.
