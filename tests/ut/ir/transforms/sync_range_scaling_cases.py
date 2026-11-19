# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Default-pipeline input with one independent moving subview per static site."""

from collections.abc import Iterable

from pypto import ir, passes
from pypto import language as pl
from pypto.ir.pass_manager import PassManager


def moving_subviews(sites: int) -> ir.Program:
    """Give each slice a different scalar definition in the same two-trip loop."""
    rows = sites + 1
    body = "\n".join(
        f"            row{site} = i + {site}\n"
        f"            view{site} = pl.tile.slice(whole, [1, 16], [row{site}, 0])\n"
        f"            out = pl.store(view{site}, [row{site}, 0], out)"
        for site in range(sites)
    )
    return pl.parse_program(f"""
@pl.program
class RangeScaling:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[{rows}, 16], pl.FP32],
               out: pl.Out[pl.Tensor[[{rows}, 16], pl.FP32]]) -> pl.Tensor[[{rows}, 16], pl.FP32]:
        whole = pl.tile.load(x, [0, 0], [{rows}, 16], target_memory=pl.MemorySpace.Vec)
        for i in pl.range(2):
{body}
        return out
""")


def lower_moving_subviews(
    program: ir.Program, *, instruments: Iterable[passes.PassInstrument] = ()
) -> ir.Program:
    """Run the ordinary placed-buffer pipeline with its conservative producer."""
    with passes.PassContext(
        list(instruments),
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
        autosync_policy="conservative_prefix",
    ):
        return PassManager(ir.OptimizationStrategy.Default).run_passes(program)
