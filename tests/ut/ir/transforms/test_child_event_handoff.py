# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Directional event reuse across retained AutoSync children."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.ir.pass_manager import PassManager

from .sync_closed_command_oracle import check_closed_commands
from .test_closed_prefix_sync import PIPES, SPAN, Services, ci, seq
from .test_logical_sync_realization import verify
from .test_tilelang_gemm_sync import MultiSlotClosedCommands

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("execute", [False, True], ids=["skipped", "taken"])
def test_retained_vector_children_order_directional_flag_reuse(execute: bool) -> None:
    """The production route must order a consumed flag before it is republished."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            source: pl.Tensor[[2, 32], pl.FP32],
            execute: pl.Scalar[pl.BOOL],
            vector_out: pl.Out[pl.Tensor[[2, 32], pl.FP32]],
            copy_out: pl.Out[pl.Tensor[[2, 32], pl.FP32]],
            out: pl.Out[pl.Tensor[[6, 32], pl.FP32]],
        ) -> pl.Tensor[[6, 32], pl.FP32]:
            if execute:
                for i in pl.range(2):
                    a: pl.Tile[[1, 32], pl.FP32, pl.MemRef("A", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                        source, [i, 0], [1, 32]
                    )
                    abs_a: pl.Tile[[1, 32], pl.FP32, pl.MemRef("AA", slots=2)[i % 2], pl.Mem.Vec] = pl.abs(a)
                    out = pl.store(abs_a, [i, 0], out)
                for j in pl.range(2):
                    b: pl.Tile[[1, 32], pl.FP32, pl.MemRef("B", slots=2)[j % 2], pl.Mem.Vec] = pl.tile.full(
                        [1, 32], pl.FP32, 7.0
                    )
                    abs_b: pl.Tile[[1, 32], pl.FP32, pl.MemRef("BB", slots=2)[j % 2], pl.Mem.Vec] = pl.abs(b)
                    vector_out = pl.store(abs_b, [j, 0], vector_out)
                    # The load overwrites the slot read by the vector pipe.
                    b_reload: pl.Tile[[1, 32], pl.FP32, pl.MemRef("B", slots=2)[j % 2], pl.Mem.Vec] = pl.load(
                        source, [j, 0], [1, 32]
                    )
                    copy_out = pl.store(b_reload, [j, 0], copy_out)
            return out

    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        function = PassManager(ir.OptimizationStrategy.Default).run_passes(Before).get_function("kernel")

    assert function is not None
    assert function.attrs.get(PYPTO_AUTOSYNC_ATTR) is True
    assert function.attrs.get("autosync_retained_child_count") == 2
    commands = MultiSlotClosedCommands(function, {1: execute}, range(8))
    commands.run()
    if execute:
        assert len(commands.payloads) == 16
        assert commands.reuses
    else:
        assert not commands.payloads
        assert not commands.publications
        assert not commands.acquisitions


@pytest.mark.parametrize("early_record", [False, True], ids=["at-boundary", "before-first-child"])
def test_child_entry_wait_requires_record_after_previous_child(early_record: bool) -> None:
    """LowerSync must recheck the publication cut of an authored child handoff."""
    dma, vector = PIPES[:2]
    services = Services()

    def child(name: str, *, first_visit: bool = False) -> ir.ForStmt:
        record, wait, *_ = services.closed(vector, dma)
        attrs: dict[str, object] = {"pypto.closed_sync_child": 1}
        if first_visit:
            attrs["pypto.closed_sync_child_first_visit"] = True
        return ir.ForStmt(
            ir.Var(name, ir.ScalarType(ir.DataType.INDEX), SPAN),
            ci(0),
            ci(2),
            ci(1),
            [],
            seq(record, wait, ir.YieldStmt([], SPAN)),
            [],
            SPAN,
            attrs=attrs,
        )

    first = child("first", first_visit=True)
    second = child("second")
    reverse_record, reverse_wait, forward_record, forward_wait = services.closed(dma, vector)
    # Only the reverse record moves. The wait still reads the same token at the boundary.
    prefix = [reverse_record, first] if early_record else [first, reverse_record]
    logical = ir.Function(
        "child_entry_record_cut",
        [],
        [],
        seq(*prefix, reverse_wait, forward_record, forward_wait, second, ir.ReturnStmt([], SPAN)),
        SPAN,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    restored = ir.deserialize(ir.serialize(verify(logical)))
    assert isinstance(restored, ir.Program)

    with passes.PassContext([]):
        if early_record:
            with pytest.raises(ValueError, match="cross-visit rearm unproved"):
                passes.lower_sync()(restored)
            return
        physical = passes.lower_sync()(restored).get_function(logical.name)

    check_closed_commands(logical, physical, {}, compare_commands=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
