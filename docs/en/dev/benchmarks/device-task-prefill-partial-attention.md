# Run L · Qwen prefill reader release and merge on 910B2

Measure two changes independently. Does releasing the AIC receive slot earlier reduce partial-attention duration? Does removing redundant entry synchronization reduce AIV merge duration? Does the complete candidate close the first256 prefill gap to stock?

## Build from the published source

Use [tonibohnlein/pypto at 5d13044ee](https://github.com/tonibohnlein/pypto/commit/5d13044eebb185507f0b674c69fd7e1aa963e74e) on `codex/pypto-autosync-clean`. Fetch the branch into a separate checkout and check out that exact SHA. Use upstream hw-native-sys/pypto-lib at `182990263ee87ef29c835ead20a716e3f4aa3d19`, Simpler at `ec5dd4fc0fa29e977b6b71a533c191439fb96ff9`, PTO-ISA at `45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b` and PTOAS **0.78**. Build that checkout’s extension and runtime using the device machine’s normal setup. Use the same CANN compiler, flags and device throughout. Record all revisions, loaded extension paths, versions and device settings.

Run from the compiler checkout in Bash. Set LIBRARY to the absolute upstream model checkout, OUT to a fresh absolute disk-backed directory and CARD to the assigned 910B2. Activate the project Python first. Set PTOAS_ROOT to the pinned toolchain. Allow sufficient queue time for stock compilation.

```bash
set -eu
test "$(git rev-parse HEAD)" = 5d13044eebb185507f0b674c69fd7e1aa963e74e
export LIBRARY OUT CARD PYPTO_PTOAS_TIMEOUT_SECONDS=3600 PYPTO_REBUILD_FROM_PTO=0
export PYTHONPATH="$PWD/python:$PWD/runtime/python:$PWD/tests/ut/ir:$PWD:$LIBRARY"
python -c 'from pypto import pypto_core; print(pypto_core.__file__)'
mkdir -p "$OUT"
for size in 128 256; do
  for mode in pypto ptoas; do
    flags=()
    if [[ $mode == pypto ]]; then flags=(--require-pypto --require-domain-owners --audit-ptoas-insertion); fi
    python -m transforms.qualify_qwen_whole_model --library "$LIBRARY" --output "$OUT/$size-$mode" \
      --entry prefill --batch 1 --layers 2 --sequence "$size" --mode "$mode" \
      --policy conservative_prefix --gm-alias-policy may-alias --scalar-gm-dma "${flags[@]}"
  done
  python tests/benchmarks/prefill_sync/partial_attention/prepare.py --sequence "$size" \
    --candidate "$OUT/$size-pypto" --stock "$OUT/$size-ptoas" --output "$OUT/arms/$size"
done
```

Save the four compilation receipts. Candidate acceptance is 32/32 owned leaves and 29/29 identical insertion audits for each shape. Preparation verifies all ten whole-program fingerprints. Record any failure and resolve the provenance discrepancy before timing.

## Programs and launch checks

| Label | Prepared directory | Runner `--arm` | Meaning |
| --- | --- | --- | --- |
| A | A | B | Earlier reader and merge schedules |
| B | B | B | Reader change only |
| M | M | B | Merge change only |
| D | D | B | Complete published candidate |
| Dp | D | Bp | Reuse D’s exact native objects |
| R | R | R | Fresh stock PTOAS synchronization |

A/B/M are experimental controls built from D with checked-in, fingerprinted schedule substitutions. Both wrappers of the mixed unit carry the cube source. All other sources, orchestration and ABI are fixed across A/B/M/D for a given shape. The directory selects the actual program. The runner’s policy flag and measurement label are separate.

Example checked and attributed smoke launch:

```bash
runner=tests/benchmarks/prefill_sync/matched/qwen_harness.py
python "$runner" --entry prefill --case first256 --arm B --label D --device "$CARD" --seed 4107 \
  --runtime-dir "$OUT/arms/256/D" --capture --out "$OUT/smoke/first256-D"
```

- On first128 and first256, check A/B/M/D/R on seeds 4107 and 4111 with three repeats per seed. Validate Dp once per shape. Require finite logits, model golden at rtol=atol=0.005 and matching winning token on every row. Compare input digests across arms. Retain failing tensors and stop timing the affected arm/shape.
- Verify ONBOARD execution. Record hashes of the **loaded native text** per shape, arm and launch. Dp must load D’s objects. Check both active AIC and AIV text; report if native compilation erases a source difference. Reconcile any unexpected difference outside the target units.
- Capture a named dependency trace for each arm and shape. Verify the actual routes: first256 should exercise partial attention and merge; first128 is the single-block routing control. Report counts and both endpoints by invocation token. Distinguish observed merge dispatch from an observed accumulation branch.

## Timing and delivery

Warm each arm/shape, then run one serial timing stream on the same card. Collect **24 complete six-arm blocks at first256**, and **12 at first128**. Start with A B R M Dp D. Generate the other five balanced orders by cyclic relabeling in the alphabet A B M D Dp R. Use each row once per cycle, shuffle the rows with a recorded seed, and keep the seed fixed within each block. Use `--timing` once the smoke trace confirms attribution. Give every launch a unique output directory and move its traces out of the replay directory before reuse. Rerun incomplete blocks.

| Primary contrast at first256 | Device endpoint per launch |
| --- | --- |
| B−A | Sum of partial-plan invocation spans, earliest AIC/AIV worker start to latest finish per invocation |
| M−A | Sum of partial-merge invocation spans |
| D−R | Whole-entry makespan, first device task start to last finish |

Report within-block median differences in µs and ratios, with endpoint-wise 95% paired block-bootstrap intervals (at least 10,000 draws). For each endpoint also compute Dp−D. Classify a result as resolved when the change interval and its paired change-minus-control interval both exclude zero in the same direction. Include D−A, D−B, D−M, the interaction D−B−M+A, first128 makespan, both endpoint worker distributions, and missing attribution. Run K’s first256 regression was 3.28% on an earlier revision; use the fresh R measurement here. Overlapping spans cannot be added to attribute device latency to a particular wait.

Return a short report and a checksummed archive with correctness and routing receipts, raw named traces and dependencies, actual loaded hashes, native target objects, launch orders, block data, build commands and analysis scripts. Give separate verdicts for the reader, merge, integrated gap and correctness. Exclude weights and caches. Monitor capture staging, remove campaign-owned dumps after failed or skipped captures, and leave the device queue clean.
