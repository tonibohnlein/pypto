# 910B2 device task · matched Qwen prefill projections

Determine where the historical persistent GEMM advantage survives in unchanged Qwen prefill. Run the matrix correctness check independently. This file is the complete handoff. All tools and expected fingerprints are in the PyPTO repository.

## Checkout and environment

Fetch `codex/pypto-autosync-clean` from https://github.com/tonibohnlein/pypto.git. Use the handoff revision specified with this file, or locate the commit adding this path with `git log -1 --diff-filter=A --format=%H -- docs/en/dev/benchmarks/device-task-matched-prefill.md`. Set CURRENT to a clean checkout at that commit and BEFORE to a separate checkout at `03f912b1ab737a83255b6cf32eecfd66d5f7d765`. Build each checkout's own Python extension. CURRENT has the compiler behavior of 7178c3bc2. Keep toolchains, runtime and ISA identical. Use upstream hw-native-sys/pypto-lib at `182990263ee87ef29c835ead20a716e3f4aa3d19`, Simpler at `a5349f42c71666bb8cacfb3759f34eeda1572beb`, PTO-ISA at `45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b`, and PTOAS 0.77 as in Run J. Record CANN, driver, card, repository revisions and loaded extension paths.

Set LIBRARY to the absolute upstream model checkout, OUT to a new absolute disk-backed results directory and CARD to the assigned device. Use the working 910B2 runtime environment. Allow a 3600-second PTOAS timeout and enough queue time for stock prefill compilation. Run shell commands with error checking. Use fresh build/cache directories per revision and arm.

```bash
set -eu
export LIBRARY PYPTO_PTOAS_TIMEOUT_SECONDS=3600
TOOLS="$CURRENT/tests/benchmarks/prefill_sync/matched"
mkdir -p "$OUT"
# Select the Python environment built for the argument before switching checkouts
# if the two extensions were built with different Python interpreters.
use_tree() {
    cd "$1"
    export PYTHONPATH="$1/python:$1/runtime/python:$1/tests/ut/ir:$1:$LIBRARY"
    python -c 'from pypto import pypto_core; print(pypto_core.__file__)'
}
for arm in current before stock; do
    mode=pypto
    flags=(--audit-ptoas-insertion --require-pypto --require-domain-owners)
    if [[ $arm == current ]]; then use_tree "$CURRENT"; else use_tree "$BEFORE"; fi
    if [[ $arm == stock ]]; then mode=ptoas; flags=(); fi
    python -m transforms.qualify_qwen_whole_model --library "$LIBRARY" --output "$OUT/$arm" \
        --entry prefill --layers 40 --batch 1 --sequence 128 --mode "$mode" \
        --policy conservative_prefix --gm-alias-policy may-alias --scalar-gm-dma "${flags[@]}"
done
use_tree "$CURRENT"
for arm in stock pypto disjoint; do
    python "$TOOLS/compile_control.py" "$arm" "$OUT/prepared/persistent/$arm"
done
python "$TOOLS/prepare.py" "$OUT/prepared" --current "$OUT/current" --before "$OUT/before" --stock "$OUT/stock"
bash "$TOOLS/run-device.sh" "$OUT/matrix" "$CARD"
```

Preparation must match the published manifest. Verify only gate_up_proj.cpp changes between the two PyPTO model builds. Matrix acceptance is 100 exact launches per arm at each of 48 and 80 rows with alternating inputs, NaN poisoning and matching C++ fingerprints. Save all four receipts.

## Compare the three workloads

1. **Persistent GEMM.** Run current, stock and disjoint on identical inputs. Current matches Qwen's may-alias policy. Disjoint matches the historical positive control. Report each speedup separately.
2. **Projection workers.** Run gate_up_proj and down_proj, all four bands, current/before/stock. Gate/up launches 24 logical workers, down launches 20. Gate and up each cover 17 output tiles. For logical block b in band j, gate is active if (b+6j) mod 24 < 17 and up if (b+6j+17) mod 24 < 17. Separate the ten double-projection workers from the fourteen single-projection workers when logical block IDs are observable. Physical core ID alone cannot establish that classification. Before/current down is a same-code check.
3. **Integrated prefill.** Compare first128 and first256 at two layers, identical seeds and model inputs. Use current, stock, and a same-object current control. For before, use the BEFORE compiler with the same harness. Collect full named swimlane records, name map and dependency graph; check record completeness and kernel attribution. Report per-band gate/up and down spans and whole-entry makespan. Reconstruct the observed critical path for each shape. Record cube activity and transfer stalls if supported.

Example invocations after preparing the benchmark:

```bash
python "$TOOLS/run_leaf.py" persistent current --root "$OUT/prepared" --device "$CARD" --seed 4107
python "$TOOLS/run_leaf.py" gate_up_proj before --root "$OUT/prepared" --band 0 --device "$CARD" --capture
python "$TOOLS/run_leaf.py" down_proj stock --root "$OUT/prepared" --band 3 --device "$CARD" --launches 5
python "$TOOLS/qwen_harness.py" --entry prefill --case first128 --arm B --label current \
    --device "$CARD" --seed 4107 --capture --out "$OUT/first128-current-smoke"
```

Use B for PyPTO and R for stock in the integrated runner. Use the selected compiler checkout for every fresh build. Use seeds 4107 and 4111 for checked launches, exact comparisons for standalone kernels and the published 5e-3 tolerances for two-layer prefill. Stop timing a failed case and save inputs, outputs and receipts. Full-depth numerical testing is optional and requires a criterion recorded before execution.

## Timing and delivery

Smoke-test one attributed block first. Verify the loaded native text hashes for each label and same-object control. Use the same device within a block, balance the arm orders, discard warm-ups, and retain one directory per capture. Time device task spans from swimlanes; exclude compilation, golden computation, staging and process startup. Aim for 24 complete paired blocks for the persistent control and gate/up band zero, then cover the other bands, down and both integrated shapes with at least 12 balanced blocks. Before/current gate/up and current/stock are separate contrasts. Report paired median differences and ratios with 95% block-bootstrap intervals and the same-object control. Mark control-sized effects unresolved.

Return one report and a checksummed archive with fingerprints, correctness receipts, raw block measurements, full traces and commands. Omit weights and caches, clean staged inputs after failed or skipped captures, and record any unmeasured endpoint.
