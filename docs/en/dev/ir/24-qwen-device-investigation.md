# Qwen device campaign investigation

The archived campaign qualifies selected projection configurations on Atlas 910B2
at `e10d882b23480d170377dd6fa227831354db8d49`: 50 mode-specific cells and 1,734
checked invocations passed. It does not qualify the newer production token
pipeline. The archive's 4,176 manifest entries were verified locally; its SHA256
is `eb8a434f0be74a7521d74ecc3aa015d2dcfa690959c6351c8a0434827bae6d81`.
See the [receipt](qwen-device-investigation.json) for the fresh host checks.
No new device execution was performed in this investigation.

## The failed preflight is a portable-fingerprint defect

The structural identity check passed on device. The digest disagreed because
Python 3.14's default `ast.dump` omits empty fields that the older interpreters
include. On the unchanged source, the local default produces the committed
`344ebf0f…`; `show_empty=True` produces the device's `d12d89fd…` exactly.
Agreement between Python 3.10 and 3.12 did not test the authoring interpreter.

The fixture helper now explicitly includes empty fields on Python >=3.13 and
uses the older equivalent default otherwise. The manifest records that format
and the corresponding digest. Original/adapted AST equality and source-byte
hashes remain enforced; payload and control edits remain detectable. Fresh
checks passed on Python 3.11, 3.12 and 3.14. This is not a promise that future AST
schema changes cannot require an explicit format revision.

## Static sites are not executed commands

The archive's `tools/sync_inventory.py` counts sites in reachable functions. It
excludes dead helper definitions but does not evaluate loops, guards or switch
arms. Consequently its `executed_set` field and the report's "executed" tables
must be interpreted as static reachable-site counts.

For example, the generated gate/up band-0 body repeats identical WAITs under
`block < 7` and `7 <= block < 17`. Only one of those sites executes for a given
block. The reachable PTOAS tail contains two switch arms; the actual call
selects `kBarrierAll`, so its alternative MTE3-to-S SET/WAIT is not executed.

A new host census preserves the archived C++ integer calculations, loops and
conditions, replacing supported payload/flag calls with counters. It explicitly
selects the cube body, rejects unknown statements and checks equal per-operation
payload counts for each original block across modes. Host compilation fails if
an erased storage value is needed by scalar control. It is a source-control
measurement, not a dependency oracle, device simulator or optimized assembly
measurement. It does not establish payload address or sequence equivalence.

For H=5120, L=1, all four gate/up bands give:

| Per original worker | PyPTO SET/WAIT pairs | PTOAS SET/WAIT pairs | PyPTO barriers | PTOAS barriers |
| ------------------- | -------------------- | -------------------- | -------------- | -------------- |
| Output projection | 357 | 369 | 0 | 202 |
| Gate/up, one output tile | 357 | 411 | 0 | 202 |
| Gate/up, two output tiles | 719 | 813 | 0 | 403 |

Gate/up has 14 one-tile and 10 two-tile workers. Each tile executes 160 cube
payload calls. Larger static PyPTO code therefore does **not** establish more
executed communications; this census shows fewer pairs for the pinned cases.

There is nevertheless avoidable control construction. Output evaluates 908
source `if` conditions versus 40. Gate/up band 2 evaluates 3,551–3,553 for
one-tile workers and 7,364 for two-tile workers, versus 40 and 80. These counts
exclude loop tests and do not count individual Boolean operators. The device
compiler can simplify these conditions; their runtime instruction cost remains
unmeasured. Whole generated C++ byte counts also include payload code.

The construction source is `EmitAssignedSyncCases` in
`src/ir/transforms/utils/physical_sync_realization.cpp`: independently qualified
activation cases append guarded commands at shared original cuts. The same
mechanism remains in the current token pipeline. Identical endpoints and
physical assignments can appear under disjoint invocation guards. A reusable
next optimization is to combine only equivalent command sites under exact guard
unions, preserving matching, cuts, IDs and actual communication multiplicity.
This investigation does not change that emitter or the allocation algorithm.

## Why the gains are smaller, and what the regressions mean

The archived Qwen worker-duration medians improve by 0.8–4.4%; the small matmul
regressions improve by 53–62%. A similarly sized absolute saving is a much
smaller fraction of a 50–100 microsecond projection than of a four-microsecond
matmul. Removing synchronization matters only when it delays the critical path;
minimum demands alone do not establish a device-latency improvement. Already
satisfied barriers, compute/memory overlap and extra control work can limit gains.
The archive lacks the instruction/stall evidence to apportion these effects.

All 30 Qwen **caller-span** batch ratios favor PyPTO. Two pooled worker-duration
medians regress slightly (L=1 band 1 batch 1, band 2 batch 2). Per-core inspection
does not find a core regressing in all three batches. Task IDs are identical
within L=1, so these are not matched logical-block comparisons. Exploratory
short/medium/long duration groups show the band-1 regression reverses in later
batches; band-2's middle/long groups do not regress in its slightly worse pooled
batch. A repeatable caller or logical-worker regression is not established.

Two report claims need correction. Reversed batch 2 does not always fall between
batches 1 and 3 (output L=1 ratios are 0.9533/0.9684/0.9465), so that observation
cannot exclude order drift. Also, caller spans use full trace `ts`/`dur` and
include setup, whereas `kernel-duration-us` excludes setup. Do not conflate them.

The trailing ALL drain is a plausible contributor, not an isolated cause: there
is no same-plan, same-boundary ablation. Dividing elapsed savings by static
barrier sites cannot estimate per-executed-barrier cost. Successful runs with
runtime `stream_wait` exercise completion; they do not prove the native return
contract or that this wait alone caused correctness.

A larger speedup is therefore possible but not implied by these results. Before
claiming one, compare modes under the same established invocation-boundary
contract, record logical block/phase IDs, and inspect scalar instructions and
pipe stalls. Compact identical command guards first, then repeat interleaved
measurements with the same workload and boundary policy.

## Reproduction and scope

Extract the verified device archive into a disk-backed workspace. Run serially:

```bash
python3 tests/ut/ir/transforms/qualify_qwen_device_commands.py \
  build-py314/device-20261111/qwen-prefill-device-20261111 \
  build-py314/device-command-census-final
PYTHONPATH=python python3 -m pytest -q \
  tests/ut/ir/transforms/test_qwen_prefill_pipeline.py \
  -k 'body_is_unchanged or fingerprint'
```

The census requires a host C++17 compiler. Its generated files and full per-block
receipt remain outside source control. No kernels, protocols, hardware exemptions
or allocation algorithms changed. The device-pinned branch remains intact;
newer host commits need their own device qualification.
