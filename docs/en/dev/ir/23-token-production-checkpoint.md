# Production token pipeline checkpoint

The five-stage migration is complete for its qualified finite, rotating,
captured-boundary and nested projection cases. Opt-in `RealizeSync` now uses
`InsertSync → logical token IR → OptimizeSync → LowerSync`; the old direct
mathematical realization has only explicit diagnostic callers. `OptimizeSync`
currently verifies without rewriting. Default compilation still delegates to
PTOAS, and unsupported whole kernels retain their original bodies.

This is a host checkpoint at implementation revision
`421b91fa703ce429429529d3aedef2e5742078d4`, following the activation transport in
`c58f5db15630644db4096ca10328c5486ba185f8`. The
[receipt](token-pipeline-m5.json) pins source, extension, Python and PTOAS hashes.
Its companion receipts retain [scaling](token-pipeline-m5-scaling.json) and
[Qwen comparisons](token-pipeline-m5-qwen.json). No device result is claimed.

## Validation and ownership

A fresh [M4 gate](token-pipeline-m4.json) passed **1,269/1,269** synchronization
checks, with 32 existing deprecation warnings. It includes actual retained covers,
all four event queries, selectors, zero trips, original coordinates, resets,
serialization, command matching, causal physical reuse, hardware exemptions and
transactional refusal. This is the selected synchronization gate, not every
repository test.

The final qualification freshly compared seven representative fixtures against
the historical direct construction on eight bounded executions. Both plans were
checked independently against original modeled accesses; their complete payload
prerequisite relations agree in both directions, with no added or removed pairs.
Executed SET counts agree: 4, 10, 19, 2, 18, 19 and 15 for the listed fixture order.
Every publication has its corresponding consumption and reuse is causal under
the existing six-ID contract. Native footprints and invocation return remain
separate premises.

Production output equals the independently serialized standalone token pipeline
for all seven fixtures and five Qwen configurations. **12/12 candidate C++ outputs**
were generated through PTOAS with downstream autosync enabled. Qwen's independent
bounded checks reuse the fresh M4 gate for the identical layers=2, hidden=512
configurations: output band 3 and gate/up bands 0–3. Audit baselines receive PTO
and code-size comparison, not a separate device campaign.

The audit API is strict. After its complete physical certificate and nonempty
SET/WAIT inventories succeed, the driver reconstructs the former production
wrapper's ownership tag for slot code generation. This is not a certificate
bypass or a new allocation policy. Production rejects preexisting tokens before
all delegation exits; use standalone `LowerSync` for such input.

## Construction cost and generated control

Three interleaved pairs per fixture compare the current production C++ wrapper
with the historical direct construction requested through Python audit bindings.
The latter includes those binding boundaries and ownership wrapping; it is not
an old-binary production benchmark. Values below are median milliseconds.
Standalone insertion/verification/lowering timings in the receipt collect
caller evidence separately and must not be added to the production total.
All timing is host compiler time, not device latency.

| Fixture | Direct audit ms | Token production ms | Static SET/WAIT pairs, old → new | Arithmetic sites, old → new |
| ------- | --------------: | ------------------: | -------------------------------: | --------------------------: |
| matmul_64 | 0.127 | 0.366 | 4 → 4 | 5 → 5 |
| matmul_acc_64 | 0.171 | 0.558 | 10 → 10 | 6 → 6 |
| ddr_split_k | 0.348 | 1.014 | 15 → 15 | 50 → 50 |
| vec | 0.284 | 0.643 | 2 → 2 | 7 → 7 |
| rotating | 0.376 | 0.846 | 9 → 9 | 76 → 76 |
| crossings | 0.648 | 1.377 | 14 → 14 | 111 → 111 |
| nested | 0.322 | 0.909 | 6 → 7 | 50 → 44 |
| qwen_output_3 | 0.731 | 1.951 | 13 → 13 | 83 → 83 |
| qwen_gate_up_0 | 2.073 | 5.170 | 57 → 57 | 380 → 380 |
| qwen_gate_up_1 | 2.466 | 6.473 | 70 → 70 | 480 → 480 |
| qwen_gate_up_2 | 2.791 | 7.275 | 88 → 88 | 487 → 487 |
| qwen_gate_up_3 | 2.514 | 6.221 | 70 → 70 | 482 → 482 |

The token pipeline costs more compiler work in these measurements. It is an
architectural separation, not a compile-time speedup. Commands and arithmetic
remain equal in these fixtures except the perfect-nest adapter: static pairs
increase from six to seven while arithmetic falls from 50 to 44. Both versions
execute 15 pairs and induce identical bounded payload order. Thus static site
count, executed communication and scheduling quality remain distinct metrics.

## Compactness and limits

The perfect-nest count sweep `(2,3)`, `(2,1000)`, `(1000,3)`, `(10^6,10^6)` retains
seven families, at most five coordinate axes and 32 parser statement visits.
A singleton carry interval may elide one axis. Fresh mathematical interface
entries grow from 1,288 to 6,664; these are separately requested audit summaries,
not the actual-token order owner's inventory. The original source tree retains
five nodes; no production nest is unfolded.

A fixed payload body with launch hulls 1, 8 and 10 uses one, one and two analytic
activation cases. The empty case remains explicit. This small sweep does not
prove a general case-count bound: the overlaid-token traversal still costs
`O(C(N+T))`, with interface, query, arithmetic and allocation work charged
separately.

The Qwen [activation receipt](token-pipeline-m4.json) keeps output projection at
hidden=8192 explicitly unavailable: it needs multiple visits per activation
case. Gate/up at 2048 and 8192 retains 90 families, 200 axes, 90 SET/WAIT sites
and 692 arithmetic operations. Growing hidden changes the output grid as well
as K, so it is not a pure trip-count experiment.

General data-dependent token control, nonrectangular guards, unrestricted dynamic
nesting, some DMA same-pipe mechanisms and insufficient resource proofs remain
refusals. The scalar-barrier prohibition and qualified accumulator exemption
remain unchanged. No limit or ID pool was increased.

## Reproduction

Use the worktree's own build, the pinned PTOAS v0.72 environment, and the Python
3.11 launcher environment recorded in earlier checkpoints. The extension in this
receipt was built for Python 3.14.7. Do not source the stale macOS `env.sh`.
The local campaign used serial builds/tests and isolated timing runs.

```bash
source .claude/skills/testing/load-env.sh
cmake --build build-py314 --parallel 1
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYPTO_CODEGEN_MAX_WORKERS=1 /usr/bin/python3 \
  -m tests.ut.ir.transforms.qualify_token_production \
  --output build-py314/token-m5-qualification.json
```

Set `PTOAS_ROOT` to the qualified installation and put its required Python 3.11
on `PATH` before that command. The exact broad gate command is in the M4 receipt.
Substantial generated inputs, PTO/C++ and archives remain under `build-py314/`.
Previous device results remain evidence for their pinned revision. This host
work does not transfer that qualification to the new implementation; a new
campaign must explicitly name the new commit.

## Paper consequences

| Established implementation evidence | Remaining premise or limitation | Appropriate claim |
| ----------------------------------- | ------------------------------- | ----------------- |
| Qualified finite/rotating demand generation and numerical composition feed tokens | Access/layout, alias and recognized-frame contracts | Exact demands for the admitted selected model |
| Actual token definitions/uses survive serialization and drive lowering | Canonical histories, captures and scoped guards only | Logical completion is a separate, usable IR boundary |
| Checked token and physical payload orders agree | Direct placement and admitted native rules | Qualified order-preserving realization, not arbitrary token lowering |
| Matching and causal reuse use the existing six-ID allocator | General scarcity/minimum budget not established | Certified feasibility for accepted plans |
| Opt-in whole-kernel rollback delegates unsupported originals | This is not the paper's general conservative construction | Safe deployment boundary with explicit refusals |
| Host checks and C++ generation pass | Native entry/return and device execution separate | No new device correctness or speedup claim |
