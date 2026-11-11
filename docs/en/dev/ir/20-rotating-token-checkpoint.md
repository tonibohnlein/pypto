# Rotating token-history checkpoint

This checkpoint extends `a49405511` with single-region rotating-slot token
transport. Existing finite and canonical reset sequences remain supported.
Rotating crossings, nested tokens and production orchestration are still pending;
this is the single-region portion of the second migration milestone.

## Construction and validity

The existing mathematical owner supplies distance-labelled demands. Insertion
creates scalar RFC token histories only for active families. A distance-three
source carries `[previous_1, previous_2, previous_3]`, initially completed, and
yields `[current, previous_1, previous_2]` simultaneously. Its consumer waits on
`previous_3` under `iv > start + 2*step`. The last three source records have no
such consumer and therefore generate no corresponding physical publications.

Lowering follows the actual carry/yield equations by variable identity. It
iteratively memoizes their acyclic paths to current records, preserving lag,
pipe and invocation. The exact original-coordinate guard must match that lag.
No storage reduction is rerun. Uses become the existing direct-cut families;
the unchanged order, physical allocation and emission services consume them.
Only the first history output is an admitted final-source crossing. An older
output is completed when the invocation is shorter than its lag; otherwise its
crossing export remains unavailable.

Successful literal SET/WAIT lowering publishes the existing `SyncOutcome`
ownership marker, shared with production. This allows Buffer codegen to use
already-qualified slot expressions rather than its narrower PTOAS delegation
rules. Ownership concerns the supplied logical protocol: hand-written tokens
still need their own dependence-coverage argument. Empty plans receive no dummy
marker. No allocator, default dispatch, hardware rule or eligible pool changed.

The [pass contract](../passes/60-logical_sync_pipeline.md) gives the full argument.

## Independent validation

The original-access interpreter computes concrete conflicts independently of the
compiler's partitions and generators. Existing validation checks actual covers,
all event queries and storage boundaries. Separate token and flag interpreters
compare their full payload relations in both directions, modeled access coverage,
matching, graph acyclicity, notification closure and causal ID reuse. Bounded
cases also compute directional width for the selected fixed command graph.

Fixtures cover zero, short, steady and tail visits; nonzero starts and nonunit
steps; noncoprime slot stride; coprime slot families; simultaneous carry
permutation; serialization; and shortening an existing long-history recipe.
Cycle, wrong-guard and live older-result tests require immutable refusal. Existing
matmul, reset and native-rule tests remain part of the gate. C++ generation uses
PTOAS with downstream autosync still enabled and verifies manual ownership.

Two limits are deliberately visible: the existing importer can reject an empty
loop whose representative footprint exceeds its declared allocation; and a
zero-trip rotating plan without a flag marker can retain delegated codegen's
slot-expression refusal. A synthetic million-slot mathematical fixture checks
inactive-history omission but retains codegen's existing 16-slot limit.

The broad gate passed **981 tests** with 28 existing warnings. The final rebuilt
focused gate passed **93 tests**, including the added coprime-history regression
and C++ generation. All three reviewers accepted the scoped checkpoint, and all
repository checks passed (Markdown formatting was corrected and rechecked).

## Size and cost

Scalar token state is proportional to `H`, the sum of maximum live lag per
source. Fixed-lag state is independent of K; it is not compact in the binary
encoding of arbitrary periods. With two independent periods two and three, the
fixture keeps five carries, not a joint-period table. Large-count checks measure
construction only; they do not execute a million visits in the oracle.

Three interleaved host timing pairs compare the unchanged production route with
standalone token stages on identical streamed inputs. All four measured PTO
outputs are byte-identical. Counts include original control arithmetic.

| Trips | Static SET / WAIT | Arithmetic | Token-stage median | Paired ratio to direct |
| ----- | ----------------- | ---------- | ------------------ | ---------------------- |
| 4 | 7 / 7 | 51 | 0.958 ms | 2.287 |
| 7 | 9 / 9 | 76 | 0.972 ms | 2.425 |
| 1,024 | 9 / 9 | 76 | 0.908 ms | 2.557 |
| 1,000,000 | 9 / 9 | 76 | 0.942 ms | 2.414 |

K=4 executes nine SET/WAIT pairs; K=7 executes eighteen. The latter uses IDs
0–2 per participating direction. Those bounded assignments fit six IDs, but the
pass retains the target catalog's existing eligibility rather than introducing
a six-ID override. These are host overhead measurements, not a speedup claim.

## Reproduction and remaining work

Use the Linux environment in the [finite checkpoint](18-token-pipeline-checkpoint.md).
Commands, source and binary hashes, results and individual timing samples are
recorded in [token-pipeline-m2.json](token-pipeline-m2.json).

```bash
cmake --build build-py314 --parallel 1
/usr/bin/python3 -m pytest -q tests/ut/ir/transforms/test_rotating_token_sync.py
/usr/bin/python3 -m tests.ut.ir.transforms.qualify_token_pipeline \
  --rotating --output build-py314/token-m2-qualification.json
```

| Evidence | Limitation | Paper consequence |
| -------- | ---------- | ----------------- |
| Actual RFC histories lower to direct families | Canonical constant lag and cuts | Executable logical transport for this subset |
| Bounded access/covers/selectors and token/flag order agree | Declared access and native model | Independent finite validation, not device qualification |
| Fixed-lag schemas stay constant in K | Scalar history grows with live lag | Trip compactness, not uniform binary-period compactness |
| Existing allocator certifies selected families | Sufficient certificate | No universal minimum-ID or alternative-protocol claim |
| Single-region rotating stream reaches C++ | Crossings, nested frames and caller bounds pending | Migration is incomplete; production still uses the direct route |
