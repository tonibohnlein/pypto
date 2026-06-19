# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AutoFuse pass (MLSys-solver-driven fusion + IR emit).

AutoFuse intercepts the raw tensor-op DAG of a function marked
``attrs={"auto_fuse": True}``, runs the MLSys solver to choose a fusion
partition + tile, and rewrites the body so each fused group is wrapped in an
``InCoreScopeStmt`` for the Outline/Convert/Tile pipeline to lower into a kernel.

v0 emits one InCore scope per group and ignores the chosen spatial tile
(``AutoTileMatmulL0`` picks the L0 tile downstream), so a matmul whose output
fits L0 lowers end-to-end; applying the solver's ``[w,h]`` as cross-core chunk
loops (for larger outputs) is a later increment.
"""

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


class TestAutoFuse:
    """AutoFuse solver-driven fusion + emit."""

    def test_single_matmul_emits_one_incore_scope(self):
        """A lone matmul becomes a single group wrapped in one InCore scope."""

        @pl.program
        class Before:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        @pl.program
        class Expected:
            @pl.function
            def mm(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_0"):
                    c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        After = passes.auto_fuse()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_single_matmul_lowers_to_cube_kernel(self):
        """The emitted scope lowers through the full pipeline to a cube PTO kernel."""
        set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(attrs={"auto_fuse": True})
            def mm(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        out = pm.run_passes(Prog)

        # The matmul group was outlined into exactly one kernel (cube/AIC) and the
        # host became an orchestration function calling it.
        incores = [f for _, f in out.functions.items() if ir.is_incore_type(f.func_type)]
        assert len(incores) == 1, [f.name for _, f in out.functions.items()]
        kernel = incores[0]

        mlir = codegen.PTOCodegen().generate(ir.Program([kernel], kernel.name, kernel.span))
        assert "pto.kernel_kind" in mlir
        assert "cube" in mlir  # a pure matmul lowers to a cube kernel
        assert "pto.tload" in mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
