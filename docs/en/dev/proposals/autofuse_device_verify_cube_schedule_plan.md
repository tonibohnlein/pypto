# Device task: AutoFuse CubeSchedulePlan fidelity

This task validates the solver-owned `CubeSchedulePlan` and the recursive generic cube emitter on an
Ascend 910B2. It is a correctness and implementation-fidelity gate, not yet a cost-ranking gate. The
known GM-to-L1 panel-reload and global-roofline gaps must be closed before natural-plan regret can be
interpreted as a model-grounding result.

Use the `fusion-scheduler` branch of PyPTO and the submodule commit recorded by that revision. The
host agent must provide the exact parent and submodule hashes when assigning this task. Abort if either
fingerprint differs.

## 0. Checkout and build

SSH port 22 is unavailable. Fetch both repositories over HTTPS and build with at most two jobs.

```bash
cd <pypto-checkout>
git fetch https://github.com/tonibohnlein/pypto.git fusion-scheduler
git switch --detach FETCH_HEAD
git rev-parse HEAD

git submodule sync --recursive
git config submodule.3rdparty/mlsys26.url https://github.com/tonibohnlein/mlsys26.git
git submodule update --init --recursive
git -C 3rdparty/mlsys26 rev-parse HEAD

cmake -S "$PWD/3rdparty/mlsys26" -B "$PWD/3rdparty/mlsys26/build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$PWD/3rdparty/mlsys26/build" --target solver_lib -j2
cmake -S "$PWD" -B "$PWD/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$PWD/build" --parallel 2

export PYTHONPATH="$PWD/python:$PYTHONPATH"
export PTOAS_ROOT=<ptoas-directory>
export PYPTO_AUTOFUSE_GENERIC_EMIT=1
export PYPTO_LOG_LEVEL=info
```

Report the full hashes, device SKU, CANN/ptoas versions, and build type. Do not use `benchmark()` if
it still fails with device error 507018; use the established `device_wall effective_us` STRACE path.

`PYPTO_AUTOFUSE_FORCE_PLAN` is cached per process. Every forced case below must run in a fresh pytest
or Python subprocess. Require the exact five-field `FORCED` key in the solver log; the ordinary
`group[0]` line continues to show the natural argmin under force.

## T0. Host gates on the device checkout

```bash
PYTHONPATH="$PWD/python" python -m pytest \
  tests/ut/ir/transforms/test_auto_fuse.py -q -n 4
"$PWD/3rdparty/mlsys26/build/tests/ascend_910b_test"
PYTHONPATH="$PWD/python" python -m pytest \
  tests/st/runtime/ops/test_auto_fuse_device.py --collect-only -q
```

Expected at the handoff revision:

- AutoFuse unit tests: `32 passed, 1 xfailed`; the xfail is the documented #1908 chained-matmul
  lowering limit.
- Solver tests: `367 pass, 7 fail`; the seven failures are the historical baseline. Capture their
  names and abort if the count or failing set changes.
- Device tests collect without import errors.

Also retain the `CUBEPLAN`, `CUBEFAN`, `CUBEDTYPE`, and full-K-window lines from the solver suite.
They establish that the extracted plan and the cost path agree before device execution.

## T1. Lone matmul controls

Run these as ordinary AutoFuse functions and compare every result with `torch.matmul`.

| case | shape | required evidence |
| ---- | ----- | ----------------- |
| uniform | `[512,512] @ [512,512]` FP32 | AIC kernel, multi-core SPMD grid, correct output |
| non-uniform grid | `[272,272] @ [272,272]` FP32 | ceil-and-clamp SPMD grid with more than one work unit; exact ragged coverage |
| ragged K | `[128,720] @ [720,128]` FP32 | stage-2 full-K loop plus serial `matmul_acc` peel; correct tail contribution |
| L0-N bound | `[32,64] @ [64,512]` FP32 | force `512,32,1,1,1`; two `[32,256]` accumulator subtiles, not one unplanned 512-column tile |
| low-precision fallback | `[64,64] @ [64,64]` FP16 | no invalid FP16 `matmul_acc`; graceful one-matmul fallback; correct FP16 output |

For the L0-N case use a fresh process:

```bash
PYPTO_AUTOFUSE_FORCE_PLAN='512,32,1,1,1' \
PYPTO_AUTOFUSE_STRICT=1 \
PYPTO_AUTOFUSE_GENERIC_EMIT=1 \
python <case-runner>
```

Report maximum absolute and relative error, emitted SPMD count, L0 matmul count, whether lowering and
assembly succeeded, and `device_wall` samples after three warmups.

## T2. Recursive matmul DAG correctness

Each forced case must execute in its own fresh process with
`PYPTO_AUTOFUSE_FORCE_MERGE=all`, `PYPTO_AUTOFUSE_STRICT=1`, and generic emission enabled.
Use the fully named DSL forms from the matching unit tests in
`tests/ut/ir/transforms/test_auto_fuse.py`; do not nest matmul calls in arguments.

| case | tensors and expression | forced plan | required plan/emit evidence |
| ---- | ---------------------- | ----------- | --------------------------- |
| natural chain | `([128,256]@[256,128])@[128,256]` | natural | one fused cube group; recursive producer before consumer; correct output |
| both-produced root | `([32,48]@[48,80]) @ ([80,64]@[64,96])` | `48,16,5,2,2` | 20 split work units plus four seed units; three plan nodes; both produced operands remain on chip |
| role-switch fanout | `shared=[64,64]@[64,64]`; return `shared@c`, `d@shared` | `32,32,1,2,2` | four work units; two request-specific instances of `shared`; both roots correct |
| deep chain | four consecutive `[64,64]` matmuls | `32,32,1,2,2` | four recursive plan nodes in one SPMD cube kernel; correct output |
| large L1 intermediate | `([128,256]@[256,768])@[768,64]` | `64,64,1,2,1` | explicit `[64,768]` L1 scratch, three producer L0-N subtiles, then the consumer |

For FP32 chains, use scaled random inputs as in the unit tests and require at least
`rtol=1e-3, atol=1e-3`. Report the actual maximum error rather than only pass/fail. Verify that no
ephemeral matmul result is assembled to GM: only roots may perform boundary stores.

The existing #1908 allocator limitation may reject some fused chains during final lowering. If so,
do not weaken the test or silently run a cut. Report the first failing pass, the complete allocation
diagnostic, the planned L0A/L0B/L0C buffers, and whether tensor-level `torch_codegen` remained correct.
This distinguishes an AutoFuse algorithm error from the known backend packing limit.

## T3. Split-K protocol

Validate both the lone natural split-K matmul and the forced both-produced root from T2.

Required evidence:

1. One disjoint zero seed per spatial root region executes before the cube kernel.
2. Exactly `S` cube work units contribute to each root region.
3. Root writes use atomic add; intermediate L1 writes do not.
4. Every K share covers a disjoint interval and the union is the full contraction axis, including a
   ragged final share where applicable.
5. Device output matches the unsplit torch reference across at least three random seeds.

Capture the transformed IR and, if available, profiler evidence for seed, cube work, and atomic MTE3
traffic. A correct numerical result with a missing seed is not sufficient because it can depend on
allocator contents.

## T4. Characterize the known GM-to-L1 reload gap

This measurement informs the next host-side fidelity patch. It is not a pass/fail model-ranking test.

Use the L0-N-bound lone case and the large-L1-intermediate chain. For every plan node, report:

- requested L1 region and its L0-M/L0-N subdivision count;
- GM-to-L1 payload for each boundary LHS and RHS;
- the number of times each LHS panel is loaded as L0-N advances;
- the number of times each RHS panel is loaded as L0-M advances;
- MTE2, MTE1, and Matrix busy cycles and any observed overlap.

The current cost model charges one logical boundary request per work unit, while the emitter's
L0-subtile-outer replay may load LHS once per L0-N subtile and RHS once per L0-M subtile. State the
measured multiplicity explicitly. A multiplicity above one confirms the documented gap and supplies
the factor needed to decide whether the model should charge it or the emitter should retain panels.

Also inspect first/rolled/tail K stages. Only a real stage-2 rolled K loop may receive a roofline
`max(compute, transfer)`; initialization, ragged peel, finalize, and store phases are serial and must
remain additive in the next phase-model revision.

## T5. Smoke the existing vector surface

The cube change must not regress the already validated vector path:

```bash
PYPTO_AUTOFUSE_GENERIC_EMIT=1 PYPTO_AUTOFUSE_P4=1 \
python -m pytest tests/st/runtime/ops/test_auto_fuse_device.py \
  --platform a2a3 -q --forked \
  -k 'ragged_pointwise or pw_tall or p4_softmax_wide or p4_layernorm_wide'
```

Report the selected count and all results. Do not change vector tolerances.

## Report

Return:

- parent/submodule fingerprints and toolchain/device details;
- T0 counts and the exact seven solver baseline failures;
- per-case numerical error, lowering/assembly/run status, SPMD and kernel counts;
- transformed IR for the forced recursive cases and split-K protocol evidence;
- GM-to-L1 panel reload multiplicities and pipe counters from T4;
- raw wall samples for characterization only;
- a one-line decision: is the supported CubeSchedulePlan subset device-correct, and is the next
  blocker AutoFuse emission, backend allocation (#1908), or the documented cost-fidelity gaps?

Do not tune constants, alter expected values, or push fixes from the device checkout. Return evidence
to the host agent first. Do not claim cost-model ranking validation from this run.
