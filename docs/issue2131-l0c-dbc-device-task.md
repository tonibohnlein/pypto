# Device Task: Validate the Issue 2131 L0C Double-Buffer Gate

## Objective

Determine whether PyPTO should automatically reserve two L0C accumulator
buffers for a matmul inside a software pipeline.

This is a measurement and validation task. Do **not** change AutoTile,
fit cost-model coefficients, open a pull request, or modify the experiment
kernel. Report evidence first.

Compare `baseline` (one accumulator, immediate drain) against forced `dbc`
(two co-live accumulators, allowing the next matmul to overlap the previous
FIXPIPE drain) under both PyPTO and PTOAS memory planners.

The production hypothesis is worth pursuing only if the forced `dbc` variant
is correct and gives a stable improvement greater than 2% under **both**
planners.

## Context

This narrow current-main experiment is motivated by
[#2131](https://github.com/hw-native-sys/pypto/issues/2131). It tests whether an
automatic pipeline heuristic is worthwhile. It does not reproduce the
historical issue environment, validate the requested public buffer-slot API,
or address issue #2130.

The working branch contains experiment infrastructure only. It does not change
AutoTile behavior:

- `tests/st/runtime/ops/issue2131_l0c_experiment.py`
- `tests/st/runtime/ops/issue2131_validate_structure.py`
- `tests/st/runtime/ops/issue2131_analyze.py`

The isolated workload has:

- `q`: BF16 `[16, 128]`;
- four BF16 `b` stacks, each `[128, 512]`;
- an outer stage-2 pipeline over stacks;
- an inner stage-2 pipeline over four 128-column output tiles;
- FP32 L0C accumulators of shape `[16, 128]`, 8 KiB each.

The only intentional difference between the variants is that the `dbc` inner
pipeline has:

```python
attrs={
    "pipeline_overlap_stores": False,
    "pipeline_double_buffer_c": True,
}
```

## Non-Negotiable Safety Rules

- Never use more than two workers in total, including nested build jobs.
- Run benchmark processes serially on one device.
- Never use bare `--parallel`, bare `-j`, `-j0`, `-j$(nproc)`, `-n auto`, or an
  equivalent automatic worker count.
- Use HTTPS for every clone and fetch. SSH is unavailable.
- Work in isolated checkouts. Do not modify a shared PyPTO installation.
- Do not install into or alter the global Python environment.
- Keep the PyPTO and pypto-lib tracked worktrees clean.
- Do not reuse binaries from another PyPTO checkout.
- Do not enable L2 swimlane, dump-args, PMU, dependency generation, or in-core
  simulation during authoritative timing.

Export these limits before any build:

```bash
export MAX_JOBS=2
export MAKEFLAGS=-j2
export PYTHONNOUSERSITE=1
```

The pinned runtime currently contains internal thread pools that can exceed the
two-worker limit even when these variables are set. Before invoking its package
build, inspect:

```text
runtime/simpler_setup/build_runtimes.py
runtime/simpler_setup/runtime_builder.py
runtime/simpler_setup/runtime_compiler.py
```

In an isolated runtime checkout, temporarily set the outer runtime-build pool
to one worker, the inner target-build pool to one worker, and the underlying
CMake build to `--parallel 2`. This avoids nested parallelism while allowing
the active build two jobs. Keep this patch applied until all device work is
finished in case a lazy rebuild occurs, then restore it and prove the submodule
is tracked-clean. Do not run the unmodified installer: its runtime compiler
hardcodes CMake parallelism up to 32.

## Required Revisions

- PyPTO branch: `issue-2131-l0c-dbc-experiment`
- Initial experiment commit: `788f58c51df5af06b73dfa2262520a58fba7fc3b`
- PyPTO base main: `efb78378071d6f6568e72402ae0f529c83ec81dc`
- pypto-lib: `72ee6a1e2e34ac2f9cad9e89fb6a70f6ee5302eb`
- runtime submodule: `8cdb306cb9a81ad1a0561325021105c676a69c1e`
- PTOAS: `v0.50`
- PTOAS AArch64 SHA256:
  `acf7b316bedccf0689971d2dc92f9f80621ab5eba131d805317e01c766c1dc2c`
- PTO ISA: `83d01313d9bfc247c4b7c8bcf969d1019f0d106f`

Record the branch HEAD; setup verifies `788f58c5` ancestry and exact current
script hashes.

## 1. Isolated Setup

Use a dedicated root such as:

```bash
ROOT=/opt/pypto/issue2131-l0c-dbc
mkdir -p "$ROOT"/{environment,logs,results}
exec > >(tee -a "$ROOT/logs/task.log") 2>&1
set -euo pipefail
set -x
```

Clone through HTTPS:

```bash
git clone --recurse-submodules --jobs 2 \
  --branch issue-2131-l0c-dbc-experiment \
  https://github.com/tonibohnlein/pypto.git \
  "$ROOT/pypto"

git clone https://github.com/hw-native-sys/pypto-lib.git \
  "$ROOT/pypto-lib"

git -C "$ROOT/pypto-lib" checkout \
  72ee6a1e2e34ac2f9cad9e89fb6a70f6ee5302eb
```

Verify revisions before building:

```bash
git -C "$ROOT/pypto" rev-parse HEAD
git -C "$ROOT/pypto" merge-base --is-ancestor \
  788f58c51df5af06b73dfa2262520a58fba7fc3b HEAD
cd "$ROOT/pypto"
echo "c390e1c86f9575183bc3d68fa3614a0fd883171951a13f4fa902ec466a8b685c  tests/st/runtime/ops/issue2131_l0c_experiment.py" | sha256sum -c -
echo "f27cd67f7aa71c46e751d9f3d26cd3ec7b9b8a3fa4181b63ec774dc04d8619e9  tests/st/runtime/ops/issue2131_validate_structure.py" | sha256sum -c -
echo "261d8cfb29e6c9a60ef10d3a65f04e66f34f5a273ca1f32d0c0f98692229b0e9  tests/st/runtime/ops/issue2131_analyze.py" | sha256sum -c -
git -C "$ROOT/pypto" submodule status
git -C "$ROOT/pypto-lib" rev-parse HEAD
cat "$ROOT/pypto/runtime/pto_isa.pin"
git -C "$ROOT/pypto" status --short > "$ROOT/environment/pypto-initial-status.txt"
git -C "$ROOT/pypto-lib" status --short > "$ROOT/environment/pypto-lib-initial-status.txt"
```

The `merge-base` command must return zero. The runtime and PTO ISA revisions
must match the table above.

Create a venv that reuses the device host's working Torch/NumPy installation,
then install the declared build backends locally. Build the pinned runtime only
after applying the worker-cap patch:

```bash
python3 -m venv --system-site-packages "$ROOT/venv"
source "$ROOT/venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install "scikit-build-core>=0.10.0" "nanobind>=2.0.0" \
  "cmake>=3.15" "ninja>=1.11.0" "cloudpickle>=2.2"
CMAKE_BUILD_PARALLEL_LEVEL=1 \
  python -m pip install --no-build-isolation -e "$ROOT/pypto/runtime"
python -m pip freeze > "$ROOT/environment/pip-freeze.txt"
```

If a stale global editable finder still redirects `simpler` or
`_task_interface` to another checkout, fix the **venv only**. Do not modify the
global installation. A venv-local `sitecustomize.py` may remove the stale
editable finder if `PYTHONNOUSERSITE=1` is insufficient. Preserve that file as
environment evidence.

Configure and build PyPTO with an explicit nested-build limit:

```bash
cmake -S "$ROOT/pypto" -B "$ROOT/pypto/build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_BUILD_PARALLEL_LEVEL=2

CMAKE_BUILD_PARALLEL_LEVEL=1 \
  cmake --build "$ROOT/pypto/build" --parallel 1
```

Install the exact AArch64 PTOAS v0.50 archive:

```bash
curl --fail --location --retry 3 --retry-all-errors \
  https://github.com/hw-native-sys/PTOAS/releases/download/v0.50/ptoas-bin-aarch64.tar.gz \
  -o "$ROOT/ptoas-bin-aarch64.tar.gz"
echo "acf7b316bedccf0689971d2dc92f9f80621ab5eba131d805317e01c766c1dc2c  $ROOT/ptoas-bin-aarch64.tar.gz" \
  | sha256sum -c -
mkdir -p "$ROOT/ptoas-bin"
tar -xzf "$ROOT/ptoas-bin-aarch64.tar.gz" -C "$ROOT/ptoas-bin"
chmod +x "$ROOT/ptoas-bin/ptoas" "$ROOT/ptoas-bin/bin/ptoas"
export PTOAS_ROOT="$ROOT/ptoas-bin"
export PATH="$PTOAS_ROOT/bin:$PTOAS_ROOT:$PATH"
ptoas --version
```

Set the experiment imports:

```bash
export PYTHONPATH="$ROOT/pypto-lib:$ROOT/pypto/python"
```

Prove import provenance. The printed paths must point into the isolated
checkouts and venv:

```bash
python -c "import golden, pypto, simpler, _task_interface; \
print('golden', golden.__file__); \
print('pypto', pypto.__file__); \
print('simpler', simpler.__file__); \
print('_task_interface', _task_interface.__file__)"
```

Also instantiate `_task_interface.CallConfig` and verify that the fields
expected by current PyPTO exist, including `enable_dump_args` and
`enable_l2_swimlane`.

Record:

- `npu-smi info`;
- selected device ID;
- CANN/toolchain environment;
- Python version;
- compiler versions;
- PyPTO, pypto-lib, runtime, PTO ISA, and PTOAS revisions;
- all import paths;
- the final worker-cap configuration used during builds.

Capture these in `environment/`, including:

```bash
git -C "$ROOT/pypto/runtime" diff > "$ROOT/environment/runtime-worker-cap.patch"
git -C "$ROOT/pypto" status --short > "$ROOT/environment/pypto-worker-cap-status.txt"
env | sort | grep -E \
  '^(ASCEND_HOME_PATH|ASCEND_OPP_PATH|ASCEND_TOOLKIT_HOME|CANN_HOME|CANN_SET_ENV|LD_LIBRARY_PATH|MAKEFLAGS|MAX_JOBS|PATH|PTOAS_ROOT|PTO_ISA_ROOT|PYTHONNOUSERSITE|PYTHONPATH)=' \
  > "$ROOT/environment/environment.txt"
npu-smi info > "$ROOT/environment/npu-smi.txt"
```

## 2. No-Device and Structural Gate

Set:

```bash
PYPTO="$ROOT/pypto"
EXPERIMENT="$PYPTO/tests/st/runtime/ops/issue2131_l0c_experiment.py"
VALIDATE="$PYPTO/tests/st/runtime/ops/issue2131_validate_structure.py"
ANALYZE="$PYPTO/tests/st/runtime/ops/issue2131_analyze.py"
COMPILE_ROOT="$ROOT/results/compile"
DEVICE=0
```

Choose one healthy A2/A3 device and replace `DEVICE=0` if needed.

Keep `PYPTO_BENCH` unset with `unset PYPTO_BENCH`.

Compile all four cases:

```bash
python "$EXPERIMENT" --variant baseline --planner pypto \
  --compile-only --seed 2040 --output-root "$COMPILE_ROOT"

python "$EXPERIMENT" --variant dbc --planner pypto \
  --compile-only --seed 2040 --output-root "$COMPILE_ROOT"

python "$EXPERIMENT" --variant baseline --planner ptoas \
  --compile-only --seed 2040 --output-root "$COMPILE_ROOT"

python "$EXPERIMENT" --variant dbc --planner ptoas \
  --compile-only --seed 2040 --output-root "$COMPILE_ROOT"
```

Run the structural checker:

```bash
python "$VALIDATE" "$COMPILE_ROOT" \
  > "$ROOT/results/structure-summary.json"
```

Expected status is `PASS`.

The checker proves:

- both lowered inner-loop copies have the expected operation order;
- baseline accumulators have no double-buffer membership;
- `dbc` has exactly `0:0, 0:1, 0:0, 0:1`;
- PyPTO uses one 8 KiB Acc allocation for baseline;
- PyPTO uses two non-overlapping 8 KiB Acc allocations for `dbc`;
- PTOAS's final C++ has the required compute/store order, one unique Acc range
  for baseline, and two non-overlapping 8 KiB Acc ranges for `dbc`;
- Mat, Left, and Right allocations are unchanged.

`PASS_WITH_PTOAS_PLACEMENT_PENDING` is acceptable only on a host where PTOAS
was deliberately skipped. On the device host it means
`PTOAS_PLACEMENT_UNPROVEN` and blocks timing.

Stop with `PLUMBING_BLOCKED` if:

- any compile fails;
- any expected structure differs;
- `dbc` does not produce exactly two L0C accumulator slots;
- the outer pipeline multiplies this into four accumulator slots;
- any planner reports allocation overflow or fallback.

## 3. Correctness Gate

Run Sections 3 and 4 under one exclusive device lease. Put their commands in
`$ROOT/environment/run-device-phase.sh`. The driver must start with
`set -euo pipefail`, source the venv, restore the exports above, and set
`DEVICE="${TASK_DEVICE:?}"`. Invoke it with the host lock service:

```bash
task-submit --device auto --device-num 1 \
  --run "bash $ROOT/environment/run-device-phase.sh"
```

Preserve the driver and lock-service log. Do not run any device command outside
that lease.

Use seeds 2040, 2041, and 2042. For each seed, save the PyPTO-baseline
inputs/golden and replay exactly that data for the other three cases. Keep
`PYPTO_BENCH` unset. This loop is deliberately serial:

```bash
for SEED in 2040 2041 2042; do
  CORRECT_ROOT="$ROOT/results/correctness/seed$SEED"

  python "$EXPERIMENT" --variant baseline --planner pypto \
    --seed "$SEED" --device "$DEVICE" --save-data \
    --output-root "$CORRECT_ROOT"

  GOLDEN_DATA="$CORRECT_ROOT/rep0_baseline_pypto_seed$SEED/data"

  for CASE in "dbc pypto" "baseline ptoas" "dbc ptoas"; do
    set -- $CASE
    python "$EXPERIMENT" --variant "$1" --planner "$2" \
      --seed "$SEED" --device "$DEVICE" --golden-data "$GOLDEN_DATA" \
      --output-root "$CORRECT_ROOT"
  done
done

find "$ROOT/results/correctness" -type f -path '*/data/*' -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$ROOT/environment/golden-data-sha256.txt"

python "$ANALYZE" --correctness-root "$ROOT/results/correctness" \
  --correctness-only > "$ROOT/results/correctness-summary.json"
```

The validator requires exactly 12 passing rows before timing. All use:

```text
rtol = 2e-2
atol = 2e-2
```

On a failure, stop with `CORRECTNESS_BLOCKED` and report:

- planner, variant, and seed;
- first incorrect index and values;
- mismatch count;
- maximum absolute and relative error;
- generated PTO/C++;
- synchronization and memory-placement differences.

Do not use timing from a correctness-failing kernel.

## 4. Authoritative Device Timing

Use the frozen seed-2040 data:

```bash
GOLDEN_DATA="$ROOT/results/correctness/seed2040/rep0_baseline_pypto_seed2040/data"
TIMING_ROOT="$ROOT/results/timing"
export PYPTO_BENCH=1
```

The pinned pypto-lib runner registers the compiled program once, performs five
warmup launches, then records 100 launches. The authoritative metric is
`effective_us`, the runtime's post-graph-build execution window. Report
`device_wall_us` as corroborating context. Do not substitute host wall time,
the op simulator, a compile-time estimate, or a swimlane-instrumented run.

Run ten paired replicates serially under the same exclusive lease. The order
below is deterministic, balanced by planner and baseline/dbC precedence:

```bash
bench_case() {
  python "$EXPERIMENT" --variant "$1" --planner "$2" \
    --seed 2040 --replicate "$3" --device "$DEVICE" \
    --golden-data "$GOLDEN_DATA" --output-root "$TIMING_ROOT"
}

ORDERS=(
  "baseline:pypto dbc:pypto baseline:ptoas dbc:ptoas"
  "dbc:ptoas baseline:ptoas dbc:pypto baseline:pypto"
  "baseline:ptoas dbc:ptoas baseline:pypto dbc:pypto"
  "dbc:pypto baseline:pypto dbc:ptoas baseline:ptoas"
  "baseline:pypto dbc:pypto baseline:ptoas dbc:ptoas"
  "dbc:ptoas baseline:ptoas dbc:pypto baseline:pypto"
  "baseline:ptoas dbc:ptoas baseline:pypto dbc:pypto"
  "dbc:pypto baseline:pypto dbc:ptoas baseline:ptoas"
  "baseline:pypto baseline:ptoas dbc:pypto dbc:ptoas"
  "dbc:ptoas dbc:pypto baseline:ptoas baseline:pypto"
)

REP=0
for ORDER in "${ORDERS[@]}"; do
  REP=$((REP + 1))
  for ITEM in $ORDER; do
    IFS=: read -r VARIANT PLANNER <<< "$ITEM"
    bench_case "$VARIANT" "$PLANNER" "$REP"
  done
done
```

Run the strict analyzer:

```bash
python "$ANALYZE" "$TIMING_ROOT" \
  --correctness-root "$ROOT/results/correctness" \
  --output "$TIMING_ROOT/SUMMARY.json"
```

The analyzer rejects:

- missing or duplicate rows;
- anything other than replicates 1 through 10;
- failed or compile-only rows;
- mismatched device, seed, platform, golden data, warmup, or round metadata;
- enabled L2 swimlane;
- non-finite, non-positive, or incomplete timing samples.

Its confidence interval resamples the ten independent run-level effects,
not the 100 serial rounds as though they were independent experiments.

Optional post-timing simulator/swimlane diagnostics must follow the repository
`incore-profiling` skill, show non-zero CUBE work, and be labelled
non-authoritative.

## 5. Decision Rules

Use the analyzer's verdict without manually relaxing it:

- `PROCEED_WITH_AUTOTILE_MODEL`: structure, correctness, and PTOAS placement
  pass; every replicate improves under both planners; each median improvement
  exceeds 2%; both lower run-level 95% bounds exceed zero. This authorizes only
  a follow-up design proposal.
- `DO_NOT_AUTO_ENABLE`: both planners are tied, slower, noisy, or below gate.
- `INVESTIGATE_PLANNER_DIVERGENCE`: planner effects have materially opposite
  signs; compare placement, schedule, synchronization, and generated code.
- Blocking: `ENVIRONMENT_BLOCKED`, `PLUMBING_BLOCKED`,
  `PTOAS_PLACEMENT_UNPROVEN`, `CORRECTNESS_BLOCKED`, or
  `TIMING_EVIDENCE_INCOMPLETE`.

Never turn blocked or incomplete evidence into a performance conclusion.

## 6. Required Report

Begin `REPORT.md` with exactly one verdict. Include the recommendation,
revision/toolchain/device/import provenance, two-worker proof, initial/final
clean status, four-case structural and PTOAS-placement evidence, 12-run
correctness matrix, per-replicate/run-level timing tables, `SUMMARY.json`
intervals, and all anomalies or limitations.

Preserve exact command logs, every `issue2131_result.json`, pass dumps, PTO and
generated C++, memory/PTOAS reports, frozen golden data, correctness/timing
logs, `SUMMARY.json`, environment proof, the temporary runtime worker-cap
patch, and any optional diagnostics labelled non-authoritative.

After all runs, restore the temporary runtime patch and capture final status:

```bash
git -C "$ROOT/pypto/runtime" restore -- \
  simpler_setup/build_runtimes.py simpler_setup/runtime_builder.py \
  simpler_setup/runtime_compiler.py
git -C "$ROOT/pypto" status --short > "$ROOT/environment/pypto-final-status.txt"
git -C "$ROOT/pypto-lib" status --short > "$ROOT/environment/pypto-lib-final-status.txt"
```

Archive everything:

```bash
tar -czf /opt/pypto/issue2131-l0c-dbc-gate.tar.gz \
  -C "$ROOT" results REPORT.md environment logs

tar -tzf /opt/pypto/issue2131-l0c-dbc-gate.tar.gz
sha256sum /opt/pypto/issue2131-l0c-dbc-gate.tar.gz
```

If directory names differ, adjust the archive command without dropping
evidence. Return the verdict, key results, archive path/size/SHA256, final
PyPTO and pypto-lib clean status, and confirmation that no process remains.
