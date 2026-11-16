# Autosync construction repair

This extends the original M1–M6 plan after bd8000fca. Coverage alone does not establish a practical synchronization plan. Preserve exact mathematics and the actual-token lowering boundary. Each stage requires independent review and an accepted commit before dependent work.

| Stage | Construction and acceptance |
| ----- | --------------------------- |
| 1 | Use statement-only token stripping/child replacement, context-sensitive direct-call verification and original-root workspace ownership. Test shared signature DAGs, valid token input/output and unowned peer setups. |
| 2 | Separate native scalar legalization from synchronization policy. Qualify original scalar GM load/store readiness, cache visibility and invocation behavior; preserve delegation where unproved. DMA staging must be explicit and default-off. |
| 3 | Preserve existing storage roots, physical views, exact ranges and sound enclosures through conservative control flow. Apply caller disjointness only in its admitted scope; distinct roots alone prove no independence. |
| 4 | Construct only outstanding obligations using path-correct established completion facts. Protect control, invocation and physical-rearming obligations independently. Validate complete protocols, matching, progress and causal reuse. |
| 5 | Requalify existing exact/regional regressions and original model with identical payload/alias/exit contracts. Report legalization expansion, added order, commands, construction cost and remaining refusals separately. |

## Stage 1 accepted subset

`Erase` and retained-child `Replace` now inherit `SyncStatementMutator`. Before: generic `VisitFunction` traversed shared expression/type metadata. After: copy the immutable function and rewrite only its body; preserve original guards, signatures and binders. BufferIR rejects a call in metadata even when its pointer matches the assignment RHS.

The fresh gate passes 130 tests, including depth28 shared signature metadata with valid logical input and physical output. Three changed C++ translation units build serially. No protocol, assignment or hardware exemption changed. An ungrouped peer with its own setup and no established common allocation remains a precise refusal; including a definition cannot establish workspace ownership. Native peer lifecycle qualification remains separate.

## Why scalar staging needs a separate decision

Original `tensor.read/write` uses `pto.load/store`. At the starting checkpoint, the bridge selected DMA load/local scalar read and local scalar write/DMA store when `conservative_prefix` was requested. Stage2 removes that coupling. S-to-S native order does not establish GM cache visibility, and changing implementation can introduce cross-pipe obligations.

For qkpv_plan, source unrolling supplies 384 possible item bodies and128 executed bodies. Staging creates six payloads per item. Before service selection, conservative services cost22 flag pairs per item: static 2+384*22=8450; executed2+128*22=2818. Eight earlier trace-specific diagnostics reduced2818 to1282. That diagnostic was neither the implemented uniform algorithm nor a minimum-command claim. Its evidence remains under `build-py314/qkpv-cost-investigation`; stage5 below measures the actual production repair.

Do not optimize rotating DMA carriers before deciding whether original scalar execution can be retained. Do not erase GM dependencies indiscriminately: different qualified roots may be disjoint, repeated access to the same root remains relevant. No device qualification follows from host execution.

## Stage 2: explicit scalar implementation

`PassContext(enable_scalar_gm_dma=False)` now preserves original scalar GM operations regardless of `autosync_policy`. Explicit `True` selects the qualified AIV/A2A3 packed rank-one INT32 DMA legalization independently of autosync enablement and producer policy. Existing exact/native qualification still decides ownership; staging grants no ownership. Context reconstruction, manager construction contracts and JIT cache identity retain this choice.

The existing pinned ordinary PTOAS qkpv_plan artifact (`qwen-m2-qualified/compiled/kernels/aiv/qkpv_plan.cpp`) emits scalar pointer reads and writes with no TLOAD/TSTORE staging. Its exits contain PIPE_ALL, whole-cache dcci and dsb. This is evidence about emitted implementation, not a proved cache/native contract. The original scalar route therefore remains delegated until selected-native readiness, visibility and exit obligations are established. The default model owner count may decrease: ownership obtained solely by implicit DMA staging is no longer a default coverage claim.

The staged regressions now request the implementation explicitly. A flag/autosync/policy matrix protects this separation; a cache test distinguishes both implementations and a manager test refuses a changed option after construction. These tests supplement the existing point geometry/value/matching/reuse tests.

The model qualifier records `--scalar-gm-dma` explicitly in its receipt. Compare default/native and explicit-staging counts separately; historical21-owned receipts used staging.

## Stage 3: storage precision and its cost

The conservative provider now retains source-owned storage-domain IDs. The existing native placement index supplies local byte enclosures. A sorted sweep merges overlapping half-open enclosures into conservative connected components, including implicit native workspaces. Touching intervals remain distinct. Unknown geometry uses domain0, a wildcard in its memory space; it never becomes a disjoint unknown root. Under the supplied disjoint-GM contract, original canonical roots and registered aliases can retain separate domains. Shared original MemRef bases remain together. May-alias and unresolved carried/borrowed roots remain conservative.

Primitive and child effects use the same immutable catalog. Generated subtree summaries are rebuilt after domain finalization; no-KILL and acknowledgment services are unchanged. Indexed read/write source-pipe summaries replace incoming-fact × consumer-fact scans. Diagnostic precision is `storage_domain_no_kill`; emitted quality is `qualified_storage_enclosures`, with exact selectors still unavailable.

For L enclosure occurrences, the sweep costs O(L log L). Production moves one running fact/index state along straight-line sequences and adds only new facts. Full incoming snapshots are research-only (`retain_incoming_diagnostics=True`); unrequested snapshots are marked unavailable rather than empty. Production retains source effects, selected source pipes and exit facts. Branch forks/merges and loop/subtree summaries still depend on J, the class/mode/pipe universe; counters charge their copies and merges separately. Generic nested CFG scratch/diagnostic inventories may require O(VJ); this is not a whole-pass O(N log N) claim. Flat disjoint-domain production tests separately require linear retained state and indexed/delta source work. Trip-count compactness and geometry/input-size scalability remain separate measurements; no admission limit or incomplete result is introduced.

## Stage 4: construct only outstanding storage services

The producer retains may-access history separately from must-established source-payload completions. These are compiler facts, not runtime counters or private branch history. At each primitive cut, initial destination knowledge and each possible source advertisement define coverage of requested source-payload epochs. The producer chooses an inclusion-minimal cover using sequential coverage-count updates, then emits those services in the original order. No tokens or physical commands exist during this selection. A later S acquisition can therefore cover an earlier MTE2 request at the same cut without constructing both services.

The selected four-command packets are replayed asymmetrically: forward WAIT imports source completions into destination; return WAIT imports destination completions into source. Merely publishing an acknowledgment does not complete the destination's own pending payload. Only selected packets establish facts. Branches, loops and retained children reset this knowledge; child entry/rearming services remain protected. This is source-payload completion knowledge, not interchangeable full token-prefix identity. Arbitrary supplied tokens, forwarding and invocation protocols stay outside the selector.

A separate whole-candidate AST traversal checks the ordered subset, all required completions at each final payload cut, protected child services and every omission. Existing physical lowering must still certify matching, resource eligibility and causal reuse. Failure is never repaired by deleting or adding emitted flags. The cover count kernel is shared by production and the existing C++ numerical test, including exhaustive small matrices; no test-only production API is added.

For k participating pipes, coverage matrices/count updates use O(k²) cells/work; establishing and querying completion maps additionally pays their ordered-map factors. Retained per-site request/selection vectors use O(k) entries. The target pipe universe is fixed. This is cut-local inclusion minimality, not a globally minimum protocol or an optimum over alternative handshakes. Unknown-trip correctness follows the block invariant starting with no knowledge, including epoch0, and the existing closed-packet reuse premise; bounded tests supplement that argument.

## Stage 5: matched production qualification

The five repair stages are complete within the qualified subset; original M1–M6 whole-model ownership is still incomplete. The [machine receipt](52-autosync-construction-repair.json) pins commits, binary/tool hashes, model sources, commands and durable local evidence. Baseline source is `7fd98e54a`, candidate source `89a481d46`, both clean at capture. Both use library `b72c66d9`, unchanged two-layer Qwen3-14B, batch1/sequence128, A2/A3, may-alias, PYPTO placement and conservative-prefix policy. Each kernel capture/audit keeps its 120-second watchdog. No device was used.

| Scalar implementation | PyPTO local owners before → after | Delegated local leaves | C++ leaves | Inclusive compile seconds before → after |
| --------------------- | --------------------------------- | ---------------------- | ---------- | ---------------------------------------- |
| Original native, default | 18 → 18 | 14 | 32/32 | 6.661 → 6.659 |
| Explicit DMA staging | 21 → 21 | 11 | 32/32 | 9.972 → 12.175 |

All47 existing task-control owners are separately observed, not newly synthesized synchronization. Pre-sync serialized IR is byte-identical within each comparison. Every post-sync payload/control inventory is unchanged. C++ is byte-identical for19 native leaves and16 staged leaves; the other leaves have changed synchronization inventories, with unchanged payload/control inventories. The staged compile increase is descriptive single-run evidence, not a speedup or a controlled attribution experiment.

The11 common delegates remain four mixed-valid ADD leaves, one RoPE MUL leaf requiring qualified partial-column descriptors, and six peer-transport leaves with unavailable local Buffer contracts. Default/native additionally delegates qkpv_plan and two token-embedding leaves at scalar-operation admission. Those three do not regain implicit staging. Native scalar cache visibility, readiness, exit qualification and admission/export support remain separate unfinished obligations. Cross-core and inter-kernel owners are preserved.

### Actual commands and ordering

| Staged qkpv_plan measurement | Before | After |
| ---------------------------- | ------ | ----- |
| Static SET sites, equally many WAIT sites | 8450 | 4610 |
| Static MTE3 same-pipe barriers | 768 | 0 |
| Static MTE2 barriers / terminal ALL | 383 / 1 | 383 / 1 |
| Executed pairs per invocation | 2818 | 1538 |
| Executed payloads per invocation | 770 | 770 |
| Payload completion-to-start prerequisites | 296065 | 296065 |

Eight actual-command audits cover finalized token counts1,8,9,16,17,127,128 and0; count9 executes two invocations. Zero is structural stress outside the positive caller domain. Both emitted programs equal the same independent staged-byte/native-pipe reference: no added **or** removed payload prerequisites, not merely equal counts. Replay checks1,185,030 prerequisites on1540 payloads. Values, original allocation windows, guards, matching, acyclicity, notification closure, terminal completion and causal consumption-before-republication pass; directional IDs are subsets of `{0,1}` within the unchanged six-ID pool.

The reference derives conflicts from executed byte effects, not compiler storage classes or generators. Its command graph/indexed reachability implementation is crosschecked against an expanded small graph. This validates the supplied staged model, not native scalar-DCache equivalence. The shared carrier already makes this reference totally ordered. With SB_BATCH1, outputs are0..127: branch/readiness paths vary, but these cases do not test a nonidentity permutation. Peak host audit RSS was806/657MiB before/after under the same1024MiB guard. Neither this nor command reduction establishes device speedup or minimum protocol size.

The construction also reduces commands in13 native-mode and16 staged-mode leaves, including down/gate/up projection, normalization and attention-finalize. Their static inventories and ownership are checked; the eight-case full relation claim above applies specifically to qkpv_plan. Generic bounded fixtures separately compare complete baseline/candidate relations, original bytes/control, skipped writers, for/while loops, zero/multiple visits, reset counters, scalar control and physical reuse. Arbitrary supplied forwarding tokens remain unchanged.

### Cost and compactness

For flat independent groups N=8,32,128, fresh candidate production retained facts are57,225,897; incremental conflict-summary additions25,97,385 and indexed requests32,128,512. Requested diagnostic snapshots were191,3071,49151. Production does not retain these snapshots. The source and tests expose the remaining control-join/subtree O(VJ) dependence; no whole-pass linearity claim follows.

A candidate-only qkpv_plan shared-service probe records2306 payload sites, one known local domain plus wildcard GM,3465 retained facts and no incoming diagnostic snapshots. Of6528 baseline service requests,3072 remain and3456 are suppressed; service counts include same-pipe requests and are not physical flag-pair counts. Analysis with Python diagnostic export takes0.042s, analysis/selection/certification/logical construction0.085s, and physical qualification/allocation/emission0.265s. These are separate descriptive calls: the analysis work repeats inside logical construction, and the physical call includes native checks/export. Do not add them or infer isolated allocation cost. Whole-model source generation is included only in the capture's inclusive compile time.

The final gate retains exact covers, selectors, four-kind event queries, compact repetition, resident/rotating storage, resets, actual-token order and physical reuse. The final fresh gate passes692 distinct Python tests in eight bounded subgroups and3 C++ targets. Results and the preserved combined-batch timeout are recorded in the receipt. All selected checks complete in bounded subgroups; the watchdog is never increased. The previously reported small-GEMM test also passes freshly; that observation does not establish universal Buffer admission.

### Reproduction

From this worktree, load `.claude/skills/testing/load-env.sh`, set `PYPTO_BUILD_JOBS=1`, `PYPTO_TEST_JOBS=1`, `PYPTO_CODEGEN_MAX_WORKERS=1` and BLAS/OpenMP threads to1. Use the pinned Linux Python/compiler extension and PTOAS0.72 launcher from the receipt; do not source stale macOS paths. `PYTHONPATH` includes this checkout's `python`, `tests/ut/ir`, `tests/ut/codegen` and root. Durable outputs are below `build-py314/production-repair`.

```bash
timeout 120s python -m transforms.qualify_qwen_whole_model \
  --library "$PYPTO_LIB" --output "$NEW_CAPTURE" --layers 2 \
  --policy conservative_prefix --require-domain-owners
# A separate capture adds --scalar-gm-dma; never overwrite old receipts.
timeout 120s python -m transforms.qualify_qwen_point_planner \
  --capture "$STAGED_CAPTURE" --library "$PYPTO_LIB" --output "$NEW_AUDIT"
```

The receipt lists the exact pytest subgroup arguments, hashes and source identities. Accepted stages1–4 have separate commits; this final evidence commit retains the original architecture plan and records its current limitations.

| Established result | Remaining premise or limitation | Appropriate paper claim |
| ------------------ | ------------------------------- | ----------------------- |
| Shared-DAG-safe body rewrites; source-owned enclosure domains and explicit alias assumptions | Unknown accesses remain wildcard; selectors are not exact physical selectors | Sound conservative modeled effects with separately stated precision |
| Cut-local service selection before tokens; complete candidate certificate | Reset at control/child boundaries; no global minimum-command theorem | Certified reduction of conservative storage services under the existing protocol |
| Matched staged qkpv_plan has fewer commands and identical bounded payload order | Staged reference is already total; original scalar qualification incomplete | Bounded realization validation, not a stronger exact-demand or native theorem |
| Exact compact provider regressions and supplied-token lowering retained | General §8 completion and full model ownership remain incomplete | Exact and conservative providers share interfaces without conflating guarantees |
| Existing six-ID realization and C++ generation pass | Host only; compilation increase measured, no device performance result | Physical feasibility for qualified plans, separately from native execution |

## Next repair plan: regional construction and production contracts

Planning baseline: `a19c1ead6`. The five accepted stages above remain historical, scoped results; they did not complete original M1–M6 or eliminate conservative service redundancy. This continuation preserves those goals and the architecture in [39](39-regional-sync-architecture.md). Status of every milestone below is **planned**, not implemented or accepted.

The latest source audit used draft `7f2f244` and PTOAS Frontier `fe32dacf3`. The draft distinguishes exact selector/query composition, conservative prefix entry wrappers, and regional common completion. The latter two preserve a child's internal implementation but can strengthen crossings. Basic prefix summaries intentionally retain pending effects; exploiting internal delivery requires additional evidence. A child boundary never implies completion or physical rearming.

Fresh diagnostics on the current staged qkpv_plan remove one whole service per executed item: 1538→1282 pairs on eight bounded executions, preserving values, total staged payload order and causal reuse. This is a trace-specific diagnostic, not an implemented uniform optimization. The remaining protocol/staging cost is not proved necessary. The same investigation reproduces the active-prefix plain-alias type mismatch and the seven stale Buffer-codegen test expectations. The supported reciprocal-peer Group integration still needs a fresh end-to-end reproduction; the existing refusal-only test is not its repair.

### P1. Repair default-compilation integration

Owners: `buffer_active_domain_analysis.cpp`, `pto_backend.py`, `gm_pipe_layout_evidence.cpp`, and existing Group ABI normalization services.

For `alias = scaled`, normalization must retype/rebind every affected Tile alias while preserving allocation identity, then propagate the binding to later uses. Candidate publication must satisfy assignment symmetry and all required IR properties. If a use cannot be represented, decline the entire normalization without publishing partially rewritten IR.

For reused Group adapters, derive the emission inventory and workspace context from the same actual owner. Reproduce reciprocal peer transport through the Default pipeline and real backend generation with autosync disabled. Determine whether originals remain legitimate public entries before considering their exclusion; neither blindly union independent allocations nor bypass workspace validation.

Gate: plain aliases and chains, negative allocation/metadata cases, both Group adapters and legitimate original entrypoints, independent workspace layouts, all three planners where applicable. Run `test_buffer_codegen.py` including the seven corrected expectations: retain the six semantic negatives with current diagnostics and replace the obsolete singleton-column rejection with a positive qualified emission check. Ordinary compilation with autosync disabled must pass.

### P2. Separate policy, alias contract and native implementation

Owners: existing `PassContext`, selected-native contracts and `qualify_qwen_whole_model.py`. Reuse the existing knobs; add no independent memory model.

The qualifier currently derives `autosync_disjoint_gm` from `policy == exact`. Replace this coupling with an explicit alias-policy input and receipt field. The primary PTOAS-comparable campaign supplies the established distinct-GM-root disjointness contract to both producers; a separate may-alias campaign exercises conservative geometry. Pin the caller/ABI source and enforcement or explicit caller promise for each admitted case. PTOAS comparability itself establishes no disjointness: without that premise retain may-alias treatment or delegation. This is an explicit caller premise, never a proof from SSA names. Same-root overlaps, aliased views and local physical reuse remain dependencies. Unknown roots remain unavailable/conservative unless the contract actually covers them.

Keep `enable_scalar_gm_dma=False` for the main acceptance matrix. Audit the selected original scalar load/store implementation for source readiness, cache visibility and owned return behavior using the existing backend contract. Scalar S→S native ordering stays active; it does not establish cross-pipe visibility. Where original scalar ownership cannot be justified, preserve delegation and name the missing premise. DMA staging remains an explicit diagnostic alternative, not a coverage requirement or automatic fallback.

Gate: exact/prefix × supplied-disjoint/may-alias × native/explicit-DMA matrix on small generic views and the unchanged original kernels. Compare PyPTO/PTOAS only under matched payload, alias and exit contracts. Do not remove all GM effects to imitate disjoint-root treatment.

### P3. Make retained regions the unit of composition

Owners: existing immutable regional analysis, `retained_sync_child.*`, `conservative_sync_realization.cpp` and `token_sync.cpp`. First inventory the current automatic child selection and refusal reasons; reuse accepted providers rather than introducing a parallel dispatcher.

For each original structured region, request only the capabilities required by its parent:

- Exact composition consumes occurrence identities, activity domains, all four event queries and covered writer/reader selectors. Construct and jointly reduce crossings using existing numerical services.
- Conservative prefix composition consumes closed child commands, sound per-pipe effects, legal entry cuts and the existing palette/rearming certificate. Keep pending effects visible after exit.
- Common-completion composition requires its distinct `(A,E,S,t)` transfer and qualified completion commands. Do not reinterpret prefix summaries as this interface or silently select this policy.

Example: an irregular outer branch containing a qualified K loop should retain one whole K invocation, with services at its qualified entry and crossings outside it. It must not receive primitive parent services throughout an already synchronized child. Open iterations with cross-iteration notifications do not qualify independently.

Use PTOAS `ChainComposition`, `CompletionTransfer`, `CompletionPrefixChild` and `PrefixChildren` as algorithm/interface references. Preserve PyPTO's public InsertSync → OptimizeSync → LowerSync ownership: exact results survive failed exports; explicit conservative requests may choose a separately labeled plan. LowerSync continues interpreting the actual token program.

Gate: first a generic nested/guarded fixture, then an existing Qwen projection and one vector region. Validate actual covers for exact results, coverage for conservative results, selectors including unavailable GM domains, child internal order, crossing order, guards, zero/skipped/repeated visits, reset coordinates, matching and reuse. Measure retained state versus trip count and nesting; include failed attempts and interface construction. A missing provider capability is a precise checkpoint result, not authorization for another recognizer.

### P4. Carry justified delivery facts through control flow

Owner: existing conservative producer and whole-candidate certificate. This is a targeted precision extension for residual conservative regions after P3, not a replacement for regional composition.

Keep two distinct analyses: may-access obligations propagate by union; guaranteed delivery facts describe work known complete on a destination pipe. Preserve incoming delivery through branches when neither path invalidates it. At a join retain only semantic facts valid on every reachable incoming edge, including the skipped edge of an if without else. Do not compare branch-local epoch integers as if they identified the same work.

Before implementation, specify finite fact identities, transfer/kill rules, lattice height and worklist cost. Use original occurrences/frames or a proved finite class-based delivery relation; no private runtime history, branch-valuation enumeration or unknown-trip unfolding. Loops require entry/backedge/zero-trip reasoning; retained children contribute only facts justified by an explicit command-transfer capability. Without that capability, retain conservative facts and protected entry/rearming services.

Example: after `complete(A); if(c) unrelated(); use(A)`, completion of A survives. If one arm issues a new write to A, the old completion does not establish completion of that new write. Likewise, a new read of A creates a reader-completion obligation before a later overwrite: old delivery cannot discharge that WAR obligation. Track access mode and newly issued work, rather than invalidating facts only on writes. If both arms establish the same semantic delivery, the join may retain it even when their token names differ, but only with a proof relating those facts.

Select service descriptors before emitting tokens. Extend the independent whole-candidate verifier to the same admitted control-flow domain, and validate it against an independent execution oracle with adversarial joins/loops. A failed certificate retains the baseline. Protect scalar-control, invocation and physical-reuse obligations separately. Arbitrary supplied tokens remain outside this producer transformation.

Gate: the generic examples above, skipped writes, newly issued reads before overwrites, nested branches, loop backedges, empty visits, child pending effects and mutually dependent omissions. Compare complete prerequisite relations, not counts alone. The staged qkpv_plan diagnostic is an additional regression; 1282 is neither a target constant nor an optimality claim. Require a documented uniform argument and bounded independent matching/reuse checks before acceptance.

### P5. Qualify quality and finish with an explicit remaining-obligation list

Re-run the unchanged supported kernel corpus and two-layer model under matched policies. Record native scalar and staged derivatives separately. For each leaf report selected provider/child boundaries, effect precision, ownership/refusal, logical construction, physical allocation and C++ outcome. Preserve the original end-to-end target and all delegated leaves in the denominator.

Account separately for: minimum modeled demands; excess payload prerequisites; forward handoffs; acknowledgment/rearming pairs; same-pipe and invocation barriers; guards and static alternatives. Compare full relation inclusion on bounded original executions. Measure analysis, failed provider attempts, exports, certificates, physical preparation and codegen where instrumentation permits; do not sum overlapping timers. Retain watchdog evidence and static-size/trip-count scaling. No further protocol rewrite follows merely from a large count: first establish which cost remains and which existing realization premises permit improvement.

Gate: P1 public default-off regressions, compact K/replay/zero-body tests, scalar and accumulator native exemptions, actual-token preservation, six-ID eligibility, terminal completion and production C++ generation. Publish reproducible commands/hashes and a machine receipt. Native/device qualification remains separate; no speedup is inferred from host counts. No allocator, tiling, cross-core protocol, ID-pool or admission-limit changes are included.

### Review and commit loop

Execute P1→P2→P3→P4→P5. For each milestone: pin input/source/tools; present the invariant and owned interface; implement; run focused independent checks; obtain independent architecture/correctness/cost review; repair findings; commit only the accepted milestone. Dependent implementation waits for acceptance. A verified missing native/mathematical premise is reported explicitly and may close a bounded investigation, but is never reported as implemented coverage. Review gates must distinguish a correctness pass from an ordering-quality or complexity limitation. Respect the machine-wide two-worker limit and disk-backed artifacts.

### P1 implementation checkpoint

The alias bridge rebinds plain Tile assignments and known-operation results through one helper, preserving existing physical storage and later-use identities. An incompatible allocation causes the complete normalization candidate to be declined. Public Default-pipeline regressions cover alias chains and all three planners with autosync disabled.

Ungrouped codegen now binds the exact emitted peer inventory using existing independent-entry context inference, while checking agreement with the original root layout. Equivalent layouts permit definition bundling; they establish no shared launch allocation or transport completion. Inference also checks native slot geometry, so equal total bytes cannot conceal incompatible rings. Explicit original-root contexts still reject unrelated setup definitions. The reused reciprocal-Group regression retains legitimate originals and adapters and runs real PTOAS under all three planners.

The Buffer-codegen gate retains unsupported-geometry negatives with current diagnostics and positively emits the supported packed singleton-column parameter. Detailed commands, source/tool identities and final review outcomes are recorded in the continuation receipt as milestones finish.

P1 independent source review accepted the ownership, correctness and cost boundaries. The final focused runs pass 187 distinct cases (186 in the combined gate, plus the added original-root offset regression in the 12-case workspace rerun), including real PTOAS generation. The [continuation receipt](52-autosync-regional-repair.json) records commands and source hashes.

### P2 contract checkpoint

The production comparison must independently select analysis policy, alias promise and scalar implementation. The qualifier's default is may-alias with original scalar operations. A disjoint campaign requires a nonempty caller-contract artifact; its exact bytes/hash and provenance are recorded. This supplies an assumption to analysis, without proving it from pointer names. Same-root views, physical local overlap and unsupported roots keep their existing treatment.

The source audit confirms `SelectBufferSyncImplementation` qualifies `buffer.read/write` only as A2/A3 AIV local Vec TGETVAL/TSETVAL. Direct Buffer codegen uses that same registered recipe and payload cut. This does not qualify cacheable GM scalar accesses. The earlier original-scalar qkpv_plan C++ contains scalar pointer accesses and cache/exit operations, but that observation establishes neither cross-pipe visibility nor data-loaded readiness. Native scalar ownership remains refused where those contracts or Buffer admission are unavailable. Explicit DMA remains a separate implementation, with its carrier effects and costs. Existing S→S and accumulator native-order rules are preserved.

For the P5 comparison, supply `--gm-alias-policy disjoint --gm-contract <caller-contract>` to either analysis policy, or use the default may-alias campaign. The former is conditional on the recorded caller promise; comparing to PTOAS does not establish that promise. No broad GM-effect deletion or default ABI change is included.

### P3 regional-construction checkpoint

Explicit conservative production requests now discover pairwise-disjoint whole sequential leaf-For invocations and reuse the existing eventful-child provider. Public InsertSync and RealizeSync share this selection; explicit supplied children retain their original request semantics. The exact policy is unchanged. Discovery/indexing is once per owner; parameter/physical-dimension membership is indexed once, while referenced bindings, effect inspection, analysis and executable-export work are charged per attempt. Numerical trip counts are not unfolded.

A typed attempt separates effect/binding/analysis qualification from executable export. Successful immutable mathematics survives a later export refusal in the published analysis. Read-only external projection leaves GM boundaries explicitly unavailable and all original GM effects visible to the parent. Qualified child token recipes, original coordinates and guards are retained; parent entry acknowledgments and physical certification remain the existing services. Loop/child exits retain pending effects.

Logical-export readiness is not a physical certificate. A later whole-plan matching/palette/reuse refusal delegates the entire original kernel with its reason. There is no automatic retry that drops selected children. This optional import can therefore lose ownership relative to primitive prefix construction; P5 must report such changes. GM-writing candidates, unsupported control and missing caller evidence remain explicit refusals. The selector retains whole leaf loops; it makes no claim to retain every enclosing nested K pipeline. Common-completion `(A,E,S,t)` composition remains a separate unavailable capability, never inferred from a prefix summary.

P3 independent review accepted the implementation after 12 new and 75 existing child cases, serial build and all hooks passed. At counts 2/16/1024 the generic child retains 9 analyzed occurrences, 5 cells and two demands. The original full-model `down_proj` has three candidate leaf loops: two export logically and the first precisely refuses unavailable enclosing coordinate `cb_inline1641__idx_v0`. Its combined selected plan allocates and generates real PTOAS C++; this does not establish retention of every enclosing K region. Exact command/source/tool evidence is in the continuation receipt.
