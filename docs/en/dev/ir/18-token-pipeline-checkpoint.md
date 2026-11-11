# RFC token pipeline: finite checkpoint

This checkpoint implements the finite part of the token migration on top of
`40f6bca81a169722cc84e4b683bb37cb517d23a0`. It does **not** complete the five-stage
plan or its first full milestone: `ddr_split_k` still needs structured token
lowering. Production remains on the existing `RealizeSync` implementation.

## Implemented and reviewed

Public `InsertSync`, identity `OptimizeSync`, and finite `LowerSync` factories
are available in C++ and Python. Lowering consumes actual token definitions,
uses and original cuts. It does not rerun storage demand reduction, erase
manually added prerequisites, or require a private retained-analysis sidecar.

The existing native-resource policy and monotone lane-coloring algorithm are
shared with the direct physical path. A publication with several non-consuming
token uses receives separately matched consuming handoffs. Actual wait-rank
reachability certifies reuse. The [pass contract](../passes/60-logical_sync_pipeline.md)
gives the validity argument and explicit limits.

Architecture, correctness and algorithm/performance reviewers accepted this
finite subcheckpoint. Review and tests exposed and repaired whole-program
property checking, statement-mutator dispatch, shared-expression traversal and
nested-sequence copying issues before acceptance.

## Host evidence

| Case | SET / WAIT / barrier | Comparison with direct physical path |
| ---- | -------------------- | ------------------------------------ |
| `matmul_64` | 4 / 4 / 0 | Byte-identical physical PTO |
| `matmul_acc_64` | 10 / 10 / 0 | Byte-identical physical PTO |
| Finite Vec fixture | 2 / 2 / 1 | Byte-identical physical PTO |

Independent command interpreters compare full payload-order relations in both
directions, modeled access coverage, matching, acyclicity and causal physical
reuse. For the admitted acyclic matched finite graphs, directional width is
computed independently and agrees with assigned ID counts. This concerns the
selected per-use handoff graph, not the optimal realization of arbitrary token
fanout. All three cases also generate C++ through the pinned PTOAS compiler with
downstream autosync enabled and no executed automatic tail call.

The new tests include stronger hand-written token dependencies, joins of distinct
prefixes, repeated waits, imported waits without an intervening source payload,
serialization, empty records, stale same-pipe prefixes, invalid owners and
unreachable cuts. A shared-join DAG and nested sequences exercise construction
without expanding expression paths or repeatedly copying flattened prefixes.

Validation receipts distinguish:

- Initial baseline: 89 logical/physical tests passed; three C++ checks deselected.
- Broad gate: 913 tests passed, 28 existing warnings. This preceded only the final
  sequence-gathering repair and additional focused cases.
- Final rebuilt focused gate: 122 tests passed, including 30 token-pipeline cases
  and the existing logical/physical suites. C++ generation is included.

Three interleaved pairs per finite case compare explicit standalone token stages
with the direct research analysis/realization route. The token route adds
verification, program wrapping and operation-import work. The observed overhead
is approximately 2.4–3.8 times on these tiny cases, with token totals below one
millisecond. These are descriptive host timings, not a production or device
speedup. Individual samples and stage times are in the receipt.

## Reproduction and pins

See `token-pipeline-m1a.json` for the starting revision, dirty source/build-file
hashes, extension hash, input hashes, commands, timing samples and scope.
The Linux host interpreter is `/usr/bin/python3`; the PTOAS launcher requires the
separately pinned Python 3.11 on `PATH`. Do not source stale platform paths from
`env.sh`. Load `.claude/skills/testing/load-env.sh` and use one intensive worker.

```bash
export PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYPTO_CODEGEN_MAX_WORKERS=1
cmake --build build-py314 --parallel 1
python -m pytest -q tests/ut/ir/transforms/test_token_sync_lowering.py
python -m tests.ut.ir.transforms.qualify_token_pipeline --output receipt.json
```

The qualifier uses the current worktree extension, not another checkout's build.
It measures strict diagnostic routes; ordinary production selection is unchanged.

## Remaining migration

| Part | Status |
| ---- | ------ |
| Finite token interpretation and physical lowering | Implemented and reviewed |
| Loop-carried tokens and `ddr_split_k` | Not implemented in LowerSync |
| Rotating and nested Qwen token exports | Not implemented |
| Caller parameter/block evidence through standalone lowering | Unavailable |
| Production token orchestration and removal of direct bypass | Pending |
| Complete migration scaling and device campaign | Pending |

The next construction derives families from actual structured token SSA and uses
the existing periodic frontier service for order queries. It must preserve loop
frames, absent predecessors and reset crossings without regenerating storage
minimum demands. Missing transport remains a refusal, not a capacity claim.

## Paper consequences

| Established result | Limitation | Appropriate claim |
| ------------------ | ---------- | ----------------- |
| Finite selected demands export to logical tokens | Admitted access model and readiness contracts | Implementation representation of the qualified direct construction |
| Finite tokens lower to matched consuming handoffs | Restricted control flow and native macro contract | Semantic refinement for the supported token subset |
| Six-ID assignments checked independently | Fixed emitted graph; fresh entry assumption | Physical feasibility for checked finite cases |
| Byte-identical finite physical output | Added host compilation overhead | Representation migration without a new synchronization-quality claim |
| No device execution in this checkpoint | Native invocation completion remains separate | Host-model qualification only |
