# AutoFuse Reference-Kernel Parity

## Scope

This document records homogeneous vector/cube algorithms found in public PTO repositories, which
ones AutoFuse can realize today, and what a reference comparison does and does not prove. Mixed
cube/vector programs are catalogued only to prevent false equivalence.

Audited revisions:

- PTO-ISA `0c112d61f41342bd0867ce1080c29f1590d72484`;
- PTO-Kernels `a8675c5a30bb4792ccc5e5f096737d12e0dfb0cc`;
- MegaGDN-PTO `8b4ae6f9413976c598d7149b545ad003efc72164`;
- PTO-DSL `b10afbea191dcce6f718d1f1240d5fdc4fca990a`;
- PTOAS `8296984f3e89913ce07fd4696542c60c2737f053`.

PTOAS is primarily an assembler, optimizer, and instruction-conformance repository. A sample is
not automatically a tuned performance reference.

## PyPTO ownership boundary

AutoFuse emits tensor-level `spmd`/`pipeline` scopes, slices/assembles, and
`tensor.matmul`/`tensor.matmul_acc`. The normal Default pipeline owns outlining, tensor-to-tile
conversion, composite lowering, `AutoTileMatmulL0`, memory-space/layout inference, pipeline
lowering, reuse/allocation, dependency construction, and PTO generation.

`tests/ut/ir/transforms/test_auto_fuse_pto_isa_reference.py` exercises that complete pipeline. It
checks algorithm and pipe-family parity, not source or binary identity.

## Realizable references

| Reference | Status | Current comparison |
| --------- | ------ | ------------------ |
| PTO-ISA add→ReLU→mul | ready | One software-pipelined AIV kernel. PyPTO uses equivalent `TMAXS(x,0)` instead of `TRELU`. |
| PTO-Kernels `abs` | ready | One `TLOAD→TABS→TSTORE` AIV dataflow. |
| PTO-Kernels FP16 SiLU/SwiGLU | ready | One two-stage AIV kernel, no intermediate GM. PyPTO uses `TNEG`; the reference uses `TMULS(-1)`. |
| PTO-Kernels affine FP16 LayerNorm | semantic match, performance gap | AutoFuse emits two AIV kernels with a GM boundary. The reference overlays phases in one kernel and applies two Newton refinements after `TRSQRT`. |
| PTO-DSL GEGLU | ready | The tanh-via-exp expression lowers to one AIV kernel. |
| PTOAS `FFN/ffn_act.pto` | ready | The clipped cubic gate and second-input multiply lower to one AIV kernel without intermediate GM. |
| PTO-ISA 1536³ FP16→FP32 GEMM | ready | Same GM→L1→L0A/L0B→Matrix→FIXPIPE hierarchy and `TMATMUL_ACC`. |
| PTO-ISA BF16 chained GEMM | ready | Producer survives through L0C→L1; only the root reaches GM. |

The focused host suite passes 9/9. The persistent a2a3 file collects 62/124 cases; the new
reference-labelled cases still require their first two-device silicon comparison.

## Represented but not new references

- PTOAS FFN FC1/FC2, GQA QK/SV, and FlashAttention QK/SV are isolated matmul/split-K stages already
  covered by GEMM, ragged-K, and `FirstPartialThenAtomic` tests.
- PTO-DSL add/ReLU and simple matmul duplicate the vector-fusion and GEMM comparisons.
- PTOAS LReLU/PReLU and primitive samples mostly validate dedicated instructions. Their semantics
  are composable, but they are not useful kernel-performance baselines without dedicated lowering.
- PTOAS GQA/FlashAttention “softmax” is a clipped polynomial without the exact row-max/row-sum
  recurrence. It is not equivalent to P4 exact softmax.

## Capability and schedule gaps

| Family | Gap |
| ------ | --- |
| PTOAS dynamic-tail matmul | Fixed ragged M/N/K is covered; runtime valid sizes and dynamic work mapping are not. |
| Transposed GEMM | Cube replay currently admits only default operand orientation. |
| GEMV/MX | Dedicated low-precision/scaling instructions lack tensor capability, role, cost, and emit descriptors. |
| PTO-DSL matmul swizzle | Base GEMM is supported; custom L2/output-grid swizzle is not a work-unit policy. |
| PTO-DSL Sinkhorn K=4 | Requires one matrix resident across alternating row/column loop phases; cut static groups are not equivalent. |
| Batch matrix square | Requires batched/3-D matmul and batch-index work mapping. |
| Triangular inverse | Requires recursive/control-flow matmul series, triangular structure, padding, and in-place updates. |
| Scan/CSR/Hadamard/quantization | Needs scan state, gather/shuffle, reshape/pad, or packed INT4/INT8 formats. |
| Causal Conv1D/GDN/KDA | Needs stencil/recurrent state, scan, dynamic chunk loops, masking, triangular solve, or mixed engines. |

The highest-value missing families are:

1. **TopK:** no sort/gather capability, streaming state, two-output value/index descriptor, or
   merge plan.
2. **Conv2D:** no convolution/stencil tensor op, halo/stride/dilation propagation,
   im2col-versus-direct plan, or emitter.
3. **MoE routing/dispatch/combine:** needs TopK, prefix/scan or histogram state, gather/scatter,
   variable expert batches, expert GEMMs, combine, and possibly communication.
4. **FlashAttention:** intentionally mixed. QK→online-softmax→PV needs cube/vector FIFO/pipeline
   machinery; vector P4 is only one component.

No system test should claim parity for these families before an explicit capability/plan/emit
contract exists.

## Interpretation and device obligations

A structural pass means equivalent algorithmic work and pipe family. Remaining differences:

1. Hand-written references choose core count, tile geometry, ping/pong layout, barriers, and
   sometimes swizzle. AutoFuse delegates several of those decisions to standard PyPTO passes.
2. Equivalent opcodes can have different performance (`TNEG` versus `TMULS(-1)`, `TMAXS` versus
   `TRELU`).
3. Affine LayerNorm has a real one-kernel-versus-two-kernel boundary.
4. GEMM parity stops at the memory hierarchy; dynamic tails, transpose, GEMV/MX, and reference
   swizzles remain separate work.
5. Mixed programs and different approximations are excluded from homogeneous performance claims.

Device comparison must therefore report both descriptors, normalized traffic, instruction/pipe
counts, numerical tolerance, and isolated wall distributions. It must not attribute every
difference to the AutoFuse cost model.
