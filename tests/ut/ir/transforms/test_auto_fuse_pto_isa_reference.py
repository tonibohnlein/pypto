# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural comparisons with public pure-vector and pure-cube PTO reference kernels.

These are algorithm-contract tests, not textual golden tests.  AutoFuse owns the
tensor partition and GM/L1 schedule, while the normal PyPTO pipeline owns
outlining, tensor-to-tile conversion, L0 tiling, memory-space inference,
pipeline lowering, memory reuse/allocation, and PTO code generation.  Different
problem sizes or legal L0 choices may therefore change instruction counts
without changing the reference algorithm.

The PTO-ISA references were audited at
``0c112d61f41342bd0867ce1080c29f1590d72484`` and are pinned by their
repository-relative paths:

* ``kernels/custom/fused_add_relu_mul/fused_add_relu_mul_kernel.cpp``
* ``kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp``
* ``kernels/manual/a2a3/gemm_performance/chain_fused_kernel.cpp``

Two additional pure-vector references were audited in the public ecosystem:

* PTO-Kernels ``a8675c5a30bb4792ccc5e5f096737d12e0dfb0cc``:
  ``csrc/kernel/kernel_abs.cpp``,
  ``csrc/kernel/kernel_swiglu.cpp``,
  ``examples/jit_cpp/silu_dynamic/silu_dynamic.cpp``, and
  ``examples/jit_cpp/layernorm/kernel_layernorm.cpp``
* PTO-DSL ``b10afbea191dcce6f718d1f1240d5fdc4fca990a``:
  ``examples/aot/activations/geglu_dynamic_multicore/geglu_builder.py``
* PTOAS ``8296984f3e89913ce07fd4696542c60c2737f053``:
  ``test/samples/FFN/ffn_act.pto``

The tests intentionally do not read a sibling PTO-ISA checkout.  CI verifies
the stable instruction-level contract described by those sources, and the
comments record intentional semantic-lowering differences such as TMAXS versus
the dedicated TRELU instruction.
"""

import pypto.language as pl
import pytest
from pypto import codegen, ir
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _lowered_pto(program) -> dict[str, list[str]]:
    """Run the complete default pipeline and return PTO text by engine."""
    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    result: dict[str, list[str]] = {"aic": [], "aiv": []}
    for _, func in lowered.functions.items():
        if not ir.is_incore_type(func.func_type):
            continue
        engine = "aic" if str(func.func_type) == "FunctionType.AIC" else "aiv"
        single = ir.Program([func], func.name, func.span)
        result[engine].append(codegen.PTOCodegen().generate(single))
    return result


def _ordered(text: str, *needles: str) -> None:
    """Assert that every instruction occurs after its predecessor."""
    position = -1
    for needle in needles:
        next_position = text.find(needle, position + 1)
        assert next_position >= 0, f"missing {needle!r} after byte {position}\n{text}"
        position = next_position


class TestPtoIsaVectorReferences:
    """Compare AutoFuse vector lowering with PTO-ISA fused vector algorithms."""

    def test_abs_matches_pto_kernels_reference_dataflow(self, ascend_backend, monkeypatch):
        """One AIV kernel realizes the PTO-Kernels TLOAD->TABS->TSTORE path."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def absolute(self, x: pl.Tensor[[512, 1024], pl.FP32]) -> pl.Tensor[[512, 1024], pl.FP32]:
                out: pl.Tensor[[512, 1024], pl.FP32] = pl.abs(x)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 1
        pto = kernels["aiv"][0]
        _ordered(pto, "pto.tload", "pto.tabs", "pto.tstore")

    def test_fused_add_relu_mul_matches_reference_dataflow(self, ascend_backend, monkeypatch):
        """One AIV kernel realizes the PTO-ISA add->ReLU->mul dataflow.

        The PTO-ISA reference uses the dedicated ``TRELU`` instruction.  Tensor
        AutoFuse currently expresses ReLU as ``maximum(x, 0)`` and therefore
        lowers the semantically equivalent middle step to ``TMAXS``.  The
        comparison requires the same one-load/one-store fused dataflow and
        permits that intentional opcode difference.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def fused(self, x: pl.Tensor[[512, 1024], pl.FP32]) -> pl.Tensor[[512, 1024], pl.FP32]:
                biased: pl.Tensor[[512, 1024], pl.FP32] = pl.add(x, 1.0)
                relu: pl.Tensor[[512, 1024], pl.FP32] = pl.maximum(biased, 0.0)
                out: pl.Tensor[[512, 1024], pl.FP32] = pl.mul(relu, 0.5)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 1
        pto = kernels["aiv"][0]
        _ordered(pto, "pto.tload", "pto.tadds", "pto.tmaxs", "pto.tmuls", "pto.tstore")
        assert pto.count("pto.tstore") == 2  # two software-pipeline stages, no intermediate spill
        assert pto.count("pto.tadds") == pto.count("pto.tmaxs") == pto.count("pto.tmuls") == 2

    def test_geglu_matches_pto_dsl_reference_dataflow(self, ascend_backend, monkeypatch):
        """The PTO-DSL tanh-based GEGLU recipe remains one fused AIV kernel.

        Like PTO-DSL, the tensor graph derives one as ``exp(gate-gate)`` and
        derives two by adding that tile to itself.  This keeps all arithmetic in
        FP16 and permits an instruction-level comparison of the complete
        no-intermediate-GM dataflow.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def geglu(
                self,
                gate: pl.Tensor[[256, 1024], pl.FP16],
                up: pl.Tensor[[256, 1024], pl.FP16],
            ) -> pl.Tensor[[256, 1024], pl.FP16]:
                zero: pl.Tensor[[256, 1024], pl.FP16] = pl.sub(gate, gate)
                ones: pl.Tensor[[256, 1024], pl.FP16] = pl.exp(zero)
                twice: pl.Tensor[[256, 1024], pl.FP16] = pl.add(gate, gate)
                exp_twice: pl.Tensor[[256, 1024], pl.FP16] = pl.exp(twice)
                numerator: pl.Tensor[[256, 1024], pl.FP16] = pl.sub(exp_twice, ones)
                denominator: pl.Tensor[[256, 1024], pl.FP16] = pl.add(exp_twice, ones)
                tanh_gate: pl.Tensor[[256, 1024], pl.FP16] = pl.div(numerator, denominator)
                one_plus_tanh: pl.Tensor[[256, 1024], pl.FP16] = pl.add(tanh_gate, ones)
                gated: pl.Tensor[[256, 1024], pl.FP16] = pl.mul(gate, one_plus_tanh)
                twos: pl.Tensor[[256, 1024], pl.FP16] = pl.add(ones, ones)
                gelu: pl.Tensor[[256, 1024], pl.FP16] = pl.div(gated, twos)
                out: pl.Tensor[[256, 1024], pl.FP16] = pl.mul(gelu, up)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 1
        pto = kernels["aiv"][0]
        assert "pto.tload" in pto and "pto.tstore" in pto
        instructions = ("pto.tadd", "pto.texp", "pto.tsub", "pto.tdiv", "pto.tmul")
        for instruction in instructions:
            assert instruction in pto
        # Generic vector streaming uses two software stages.  A larger count
        # would expose a cut or an intermediate spill.
        assert pto.count("pto.tstore") == 2

    def test_silu_matches_pto_kernels_reference_dataflow(self, ascend_backend, monkeypatch):
        """FP16 SiLU remains the reference's load->negate->exp->add->div->store chain.

        PTO-Kernels writes negation as ``TMULS(x, -1)``.  The PyPTO tensor
        operation lowers to the dedicated, semantically equivalent ``TNEG``.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def silu(self, x: pl.Tensor[[256, 1024], pl.FP16]) -> pl.Tensor[[256, 1024], pl.FP16]:
                neg: pl.Tensor[[256, 1024], pl.FP16] = pl.neg(x)
                exp: pl.Tensor[[256, 1024], pl.FP16] = pl.exp(neg)
                one: pl.Scalar[pl.FP16] = pl.cast(1.0, pl.FP16)
                denom: pl.Tensor[[256, 1024], pl.FP16] = pl.add(exp, one)
                out: pl.Tensor[[256, 1024], pl.FP16] = pl.div(x, denom)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 1
        pto = kernels["aiv"][0]
        _ordered(pto, "pto.tload", "pto.tneg", "pto.texp", "pto.tadds", "pto.tdiv", "pto.tstore")
        assert pto.count("pto.tstore") == 2

    def test_swiglu_matches_pto_kernels_reference_dataflow(self, ascend_backend, monkeypatch):
        """FP16 SwiGLU adds the reference's second input and final TMUL without a GM spill."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def swiglu(
                self,
                gate: pl.Tensor[[256, 1024], pl.FP16],
                up: pl.Tensor[[256, 1024], pl.FP16],
            ) -> pl.Tensor[[256, 1024], pl.FP16]:
                neg: pl.Tensor[[256, 1024], pl.FP16] = pl.neg(gate)
                exp: pl.Tensor[[256, 1024], pl.FP16] = pl.exp(neg)
                one: pl.Scalar[pl.FP16] = pl.cast(1.0, pl.FP16)
                denom: pl.Tensor[[256, 1024], pl.FP16] = pl.add(exp, one)
                silu: pl.Tensor[[256, 1024], pl.FP16] = pl.div(gate, denom)
                out: pl.Tensor[[256, 1024], pl.FP16] = pl.mul(silu, up)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 1
        pto = kernels["aiv"][0]
        _ordered(pto, "pto.tload", "pto.tneg", "pto.texp", "pto.tadds", "pto.tdiv", "pto.tmul", "pto.tstore")
        assert pto.count("pto.tstore") == 2

    def test_affine_layernorm_records_current_two_kernel_gap(self, ascend_backend, monkeypatch):
        """The FP16 affine LayerNorm graph uses the reference primitives, but currently cuts once.

        This comparison deliberately checks the algorithm and memory path rather
        than claiming textual identity.  PTO-Kernels uses a hand-scheduled
        one-kernel phase overlay and two Newton refinements after ``TRSQRT``.
        AutoFuse leaves the refinement to normal PyPTO lowering and currently
        cuts between the mean/center operation and the variance/apply operation.
        Keep that difference explicit: semantic support is device-testable, but
        this is not yet a performance-equivalent reference implementation.
        """
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def layernorm(
                self,
                x: pl.Tensor[[32, 1024], pl.FP16],
                gamma: pl.Tensor[[1, 1024], pl.FP16],
                beta: pl.Tensor[[1, 1024], pl.FP16],
            ) -> pl.Tensor[[32, 1024], pl.FP16]:
                x32: pl.Tensor[[32, 1024], pl.FP32] = pl.cast(x, pl.FP32)
                sum_x: pl.Tensor[[32, 1], pl.FP32] = pl.row_sum(x32)
                mean: pl.Tensor[[32, 1], pl.FP32] = pl.mul(sum_x, 1.0 / 1024)
                centered: pl.Tensor[[32, 1024], pl.FP32] = pl.row_expand_sub(x32, mean)
                square: pl.Tensor[[32, 1024], pl.FP32] = pl.mul(centered, centered)
                sum_square: pl.Tensor[[32, 1], pl.FP32] = pl.row_sum(square)
                variance: pl.Tensor[[32, 1], pl.FP32] = pl.mul(sum_square, 1.0 / 1024)
                variance_eps: pl.Tensor[[32, 1], pl.FP32] = pl.add(variance, 1.0e-5)
                inv_std: pl.Tensor[[32, 1], pl.FP32] = pl.rsqrt(variance_eps)
                normalized: pl.Tensor[[32, 1024], pl.FP32] = pl.row_expand_mul(centered, inv_std)
                gamma32: pl.Tensor[[1, 1024], pl.FP32] = pl.cast(gamma, pl.FP32)
                beta32: pl.Tensor[[1, 1024], pl.FP32] = pl.cast(beta, pl.FP32)
                scaled: pl.Tensor[[32, 1024], pl.FP32] = pl.mul(normalized, gamma32)
                shifted: pl.Tensor[[32, 1024], pl.FP32] = pl.add(scaled, beta32)
                out: pl.Tensor[[32, 1024], pl.FP16] = pl.cast(shifted, pl.FP16)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 2
        pto = "\n".join(kernels["aiv"])
        for instruction in (
            "pto.tcvt",
            "pto.trowsum",
            "pto.trowexpandsub",
            "pto.tmul",
            "pto.trsqrt",
            "pto.trowexpandmul",
            "pto.tadd",
            "pto.tstore",
        ):
            assert instruction in pto
        assert all(kernel.count("pto.tstore") >= 1 for kernel in kernels["aiv"])

    def test_ptoas_ffn_activation_matches_reference_dataflow(self, ascend_backend, monkeypatch):
        """PTOAS's standalone clipped-cubic FFN activation remains one AIV kernel."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def activate(
                self,
                h1: pl.Tensor[[32, 32], pl.FP32],
                h2: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                low: pl.Tensor[[32, 32], pl.FP32] = pl.maximum(h1, -4.0001)
                clipped: pl.Tensor[[32, 32], pl.FP32] = pl.minimum(low, 4.0001)
                square: pl.Tensor[[32, 32], pl.FP32] = pl.mul(clipped, clipped)
                cube: pl.Tensor[[32, 32], pl.FP32] = pl.mul(square, clipped)
                cubic: pl.Tensor[[32, 32], pl.FP32] = pl.mul(cube, -1.0 / 48.0)
                linear: pl.Tensor[[32, 32], pl.FP32] = pl.mul(clipped, 0.25)
                shifted: pl.Tensor[[32, 32], pl.FP32] = pl.add(linear, 0.5)
                gate: pl.Tensor[[32, 32], pl.FP32] = pl.add(shifted, cubic)
                out: pl.Tensor[[32, 32], pl.FP32] = pl.mul(gate, h2)
                return out

        kernels = _lowered_pto(Prog)
        assert not kernels["aic"]
        assert len(kernels["aiv"]) == 1
        pto = kernels["aiv"][0]
        instructions = (
            "pto.tload",
            "pto.tmaxs",
            "pto.tmins",
            "pto.tmul",
            "pto.tmuls",
            "pto.tadds",
            "pto.tadd",
        )
        for instruction in instructions:
            assert instruction in pto
        assert pto.count("pto.tstore") == 2


class TestPtoIsaCubeReferences:
    """Compare AutoFuse cube lowering with PTO-ISA GEMM algorithms."""

    def test_fp16_fp32_gemm_matches_reference_memory_path(self, ascend_backend, monkeypatch):
        """The 1536^3 table point produces the reference GM/L1/L0/FIXPIPE path."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[1536, 1536], pl.FP16],
                b: pl.Tensor[[1536, 1536], pl.FP16],
            ) -> pl.Tensor[[1536, 1536], pl.FP32]:
                out: pl.Tensor[[1536, 1536], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                return out

        kernels = _lowered_pto(Prog)
        assert len(kernels["aic"]) == 1
        pto = kernels["aic"][0]
        assert "pto.tload" in pto  # GM -> L1
        assert "pto.textract" in pto or "pto.tmov" in pto  # L1 -> L0A/L0B
        assert "pto.tmatmul" in pto
        assert "pto.tmatmul.acc" in pto
        assert "pto.tstore" in pto  # persistent L0C -> GM through FIXPIPE
        assert "loc=acc" in pto

    def test_bf16_chain_keeps_intermediate_off_gm(self, ascend_backend, monkeypatch):
        """The recursive cube DAG matches PTO-ISA's L0C->L1 producer handoff."""
        monkeypatch.setenv("PYPTO_AUTOFUSE_GENERIC_EMIT", "1")
        monkeypatch.setenv("PYPTO_AUTOFUSE_EXACT_L0_COST", "1")

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def chain(
                self,
                a: pl.Tensor[[128, 256], pl.BF16],
                b: pl.Tensor[[256, 128], pl.BF16],
                d: pl.Tensor[[128, 256], pl.BF16],
            ) -> pl.Tensor[[128, 256], pl.BF16]:
                intermediate: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(a, b)
                out: pl.Tensor[[128, 256], pl.BF16] = pl.matmul(intermediate, d)
                return out

        kernels = _lowered_pto(Prog)
        assert len(kernels["aic"]) == 1
        pto = kernels["aic"][0]
        assert pto.count("pto.tmatmul") >= 2
        assert "pto.tmov" in pto  # first FP32 accumulator narrows into an L1 Mat tile
        assert pto.count("pto.tstore") == 1  # only the root result reaches GM
        assert "loc=mat" in pto and "loc=acc" in pto


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
