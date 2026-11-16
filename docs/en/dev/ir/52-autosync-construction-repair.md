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
