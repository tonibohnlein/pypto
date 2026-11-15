# Explicit module ownership checkpoint

This completes a bounded deployment interface: an explicit whole-Program request can own empty, barrier-only and flag-bearing local kernels without fabricating flags. It does not complete whole-Qwen acceptance or the original plan's M6.

## Contract and implementation

`LowerSync(require_whole_module=True)` and `RealizeSync(require_whole_module=True)` qualify every local placed AIC/AIV body atomically. Unknown helpers, external domains, unplaced storage and an already-owned physical program refuse before publication. Ordinary calls keep their existing per-leaf deployment behavior.

Operation/native-resource checks run even when no logical operations exist. The shared runtime-valid metadata checker proves the selected native call's use-site relations; dynamic descriptor sentinels are not evidence. `LowerSync` implements the actual token program and does not assert that manually supplied tokens cover memory hazards. `RealizeSync` retains its separate insertion guarantee.

A qualified payload/command plan retains the existing terminal ALL completion. A truly empty plan is identified by both absence of native payloads and zero actual SET/WAIT/barrier sites across all physical receipts; a nonempty diagnostic receipt alone creates no completion debt.

The backend inventories every actual function given to code generation and every generated helper. A reserved `autosync_whole_module` claim can suppress PTOAS insertion only for a fully qualified unit on the existing A2/A3 target with preserved level3 placement. A mixed unit refuses instead of silently altering the explicit plan. Default flag-bearing deployment keeps PTOAS enabled.

Owned PTO receives a first-line marker and a sibling `.sync.json` with the exact source SHA256, complete domain inventory and bounded compiler options. Group/raw exports put it beside the actual source. Ownership-aware debug rebuild verifies bytes and restores those options. Edited sources, missing or downgraded claims, unsupported target/options and replanning refuse. The format detects accidental mismatch; it is not authentication. Standalone PTOAS CLI compilation of owned raw PTO without this driver is outside the contract.

## Validation

The focused ownership file checks empty Lower/Realize, barrier-only actual-token order, final-store drain, identical default flag bodies, serialization, mixed SIMT/opaque/helper refusal, actual unit inventories, per-unit compiler options, grouped raw artifacts and unchanged/edited/missing/downgraded debug manifests. Runtime metadata without equality/positive-row evidence and target/placement transplants refuse without mutating input IR.

All ownership cases passed in the combined focused gate. Its three runtime active-access fixture failures were separate from ownership; the corrected runtime gate subsequently passed 14/14. The final integrated gate passed 183 tests in 111.50 seconds, and all repository hooks, including full Pyright, passed. These are host checks; the companion JSON pins the dirty source snapshot, integrated binary and logs. No device was run.

## Remaining dispositions

| Obligation | Disposition |
| ---------- | ----------- |
| Scalar GM values/control | Native scalar ordering is preserved, but data-cache visibility/readiness is not established merely by no local writes. Keep refusal without an explicit supported contract. |
| Local transfer and AIC/AIV transport | Packed column transfer is separately qualified. Cross-core communication still needs an explicit owner/readiness/resource boundary; a broad address enclosure cannot supply it. |
| Runtime post-normalization | Fixed-column runtime-row broadcast is a separate qualified slice. Dynamic reduction, axis metadata and mixed full/partial add require their own native footprint/padding/value contracts. No benchmark r=8 restriction is assumed. |
| Retained children | Initial eventful local family is separately certified. Read-only GM projection and broader original-frame bindings remain independent gates. |
| Whole model | Re-run the named-leaf census after all accepted packages; report exact, prefix and delegated domains separately. Do not infer full coverage from empty-plan ownership. |

## Paper consequences

| Established result | Remaining premise or limitation | Appropriate claim |
| ------------------ | ------------------------------- | ----------------- |
| Actual-token lowering is separate from insertion. | Manual tokens can omit hazards. | Lowering preserves the admitted expressed order; insertion separately establishes demand coverage. |
| A qualified explicit local unit owns its exit obligation and downstream insertion mode. | Native selected operations and runtime task semantics remain bounded to the existing contract. | Certified local deployment for the admitted target/operation family. |
| Empty/barrier-only plans no longer require fake literal flags. | Mixed/helper domains refuse; no automatic fallback change. | Explicit whole-module ownership interface. |
| Source/manifest/options survive aware debug rebuild. | Raw standalone CLI does not read the manifest. | Artifact-consistent ownership transport, not universal raw-PTO ownership. |
