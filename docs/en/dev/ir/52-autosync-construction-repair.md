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

Original `tensor.read/write` uses `pto.load/store`. The recently introduced bridge selects DMA load/local scalar read and local scalar write/DMA store instead. It currently depends on `conservative_prefix`; that coupling is the stage2 repair target. S-to-S native order does not establish GM cache visibility, and changing implementation can introduce cross-pipe obligations.

For qkpv_plan, source unrolling supplies 384 possible item bodies and128 executed bodies. Staging creates six payloads per item. Current conservative services cost22 flag pairs per item: static 2+384*22=8450; executed2+128*22=2818. Eight bounded trace-specific diagnostics reduce2818 to1282 while preserving staged total payload order, values, matching and reuse. This is neither a uniform optimizer nor a minimum-command claim. Evidence is under `build-py314/qkpv-cost-investigation`.

Do not optimize rotating DMA carriers before deciding whether original scalar execution can be retained. Do not erase GM dependencies indiscriminately: different qualified roots may be disjoint, repeated access to the same root remains relevant. No device qualification follows from host execution.
