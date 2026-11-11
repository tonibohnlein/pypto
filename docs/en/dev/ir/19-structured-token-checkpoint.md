# Canonical loop-token checkpoint

This checkpoint extends the finite foundation at `0611a9e07` to the original
three matmul examples, including `ddr_split_k`. It completes the first migration
milestone's target cases. It does not complete rotating, nested Qwen or production
migration; ordinary `RealizeSync` remains unchanged.

## Construction and contract

The structured parser consumes actual records, waits, loop arguments, yields and
results. It certifies direct payload cuts and exact first/later-visit guards.
It accepts completed-initialized carries with unconditional current-record
yields, retains per-use identities and preserves original starts, steps and
counter resets. Zero-trip loop results remain completed facts.

The retained token families supply the existing periodic frontier kernel without
cover extraction. Multi-region order uses the extracted legacy boundary graph;
minimum-demand generation remains with its existing mathematical owners.
`DirectSyncOrder` supplies sufficient completion-to-start reuse proofs to the
existing allocator, while input-only placement reuses its physical emitter.
There is no private sidecar, new coloring policy, storage-demand reconstruction
or changed pool. Structured allocation does not claim a minimum ID count.

The [pass contract](../passes/60-logical_sync_pipeline.md) states the validity
argument and cost bounds. Single-segment queries need no boundary graph.
Multiple segments retain first/final ports; their construction may be quadratic
in static port count. Cross-region queries now traverse the graph once from all
source ports. These bounds are independent of trip count but are not a uniform
linear end-to-end compilation claim.

## Validation scope

Independent token and physical interpreters compare the full projected payload
relation in both directions, modeled access coverage, matching, command
acyclicity and causal ID reuse. The original physical access interpreter supplies
concrete requirements independently of compiler demand records. Native access
qualification and invocation completion remain separate assumptions.

The fixtures cover zero, one and repeated visits, initialization/accumulation,
original nonunit loop coordinates, reset crossings, repeated waits and serialized
IR. Negative tests retain refusal for noncanonical guards, non-completed carry
entry, old-history yields and records displaced after pending waits. A zero-trip
source exports a completed token to a live region. Shared sequence rewriting
avoids copying nested prefixes repeatedly.

The large-count derivatives use identical static schedules at K=2, 1024 and
1,000,000. They measure schema size and host construction time; only bounded
instances receive concrete execution checks. The machine-readable receipt records
exact final commands, hashes, test results and paired samples in
[token-pipeline-m1b.json](token-pipeline-m1b.json).

The broad gate passed **947 tests** (28 existing warnings) before final formatting
and an earlier axis-bound validation check. The rebuilt focused gate then passed
**218 tests**, including scoped allocation, collective realization and C++ output.
All three reviewers accepted the qualified implementation.

The four measured cases produce byte-identical physical PTO to the direct route.
`ddr_split_k` retains 15 SET and 15 WAIT sites. The K=2/1024/1,000,000 derivatives
retain 33 SET and 33 WAIT sites each. Total physical arithmetic is 50 for DDR and
70/71/71 for those derivatives; this includes original payload-control arithmetic.
Token-stage medians are about 1.4–2.2 ms, with paired ratios of 3.1–4.6 versus the
direct route. This is host overhead, not a performance improvement.

## Reproduction

Use the Linux environment described in the [finite checkpoint](18-token-pipeline-checkpoint.md).
Load the repository testing environment and limit intensive work to one worker
per command, respecting the machine-wide aggregate limit.

```bash
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m pytest -q tests/ut/ir/transforms/test_structured_token_sync.py
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_token_pipeline \
  --structured --output build-py314/token-m1b-qualification.json
```

The qualifier compares the same original input and target through the direct
research route and the standalone token stages in three interleaved pairs.
Timing is descriptive host evidence. No device run or speedup is established.

## Remaining obligations

| Capability | Status |
| ---------- | ------ |
| Original three matmuls through standalone token stages | Supported by the qualified finite/canonical-loop contracts |
| Rotating token histories beyond one previous visit | Separate next milestone |
| Interior crossing selectors and nested token frames | Unavailable in this adapter |
| Caller parameter/block evidence through standalone lowering | Still unavailable |
| Production token orchestration | Pending; direct production route retained |
| Device qualification of this token migration | Not run |

## Paper consequences

| Established result | Limitation | Appropriate claim |
| ------------------ | ---------- | ----------------- |
| Actual loop token uses lower to compact direct families | Canonical cuts, carries and guards | Executable transport for the stated original-coordinate subset |
| Numerical order supports physical reuse | Sufficient certificate, not minimum allocation | Certified assignment where accepted |
| Bounded token/flag relations agree | Concrete admitted executions and access model | Independent finite semantic validation |
| Representation is independent of K in derivatives | Static-port graph can be quadratic | Trip-count compactness with separately charged interface work |
| Existing production routes remain available | Migration is incomplete | No whole-branch token-pipeline or new native/device claim |
