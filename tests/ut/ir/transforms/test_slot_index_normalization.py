# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Use-scoped canonical slots survive simplification without changing coordinates."""

import pypto.language as pl
import pytest
from pypto import DataType, codegen, ir, passes
from pypto.backend.pto_backend import _run_ptoas
from pypto.pypto_core import ir as core_ir

from .test_select_buffer_ir_functions import _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def direct_slots(start=0, stop=2, step=1, *, dtype=DataType.INDEX, mode="direct", counts=(2,)):
    def integer(value):
        return ir.ConstInt(value, dtype, SPAN)

    def internal(name, args, result):
        return core_ir._create_internal_op_call(name, args, {}, result, SPAN)

    element = ir.BufferType([1, 32], DataType.FP32, ir.MemorySpace.Vec)
    index = ir.Var("i", ir.ScalarType(dtype), SPAN)
    foreign = ir.Var("i", ir.ScalarType(dtype), SPAN)
    groups, allocations = [], []
    for number, count in enumerate(counts):
        multi = ir.MultiBufferType(element, count)
        group = ir.Var(f"ring{number}", multi, SPAN)
        groups.append(group)
        allocations.append(
            ir.AssignStmt(
                group,
                internal(
                    "buffer.alloc_multi",
                    [ir.MakeTuple([], SPAN), ir.ConstInt(number * 512, DataType.INDEX, SPAN)],
                    multi,
                ),
                SPAN,
            )
        )

    def selections(prefix):
        selected = foreign if mode == "foreign" else index
        statements = []
        if mode == "alias":
            for number in range(4):
                alias = ir.Var(f"{prefix}_alias{number}", ir.ScalarType(dtype), SPAN)
                statements.append(ir.AssignStmt(alias, selected, SPAN))
                selected = alias
        if mode == "narrow":
            selected = ir.Cast(ir.Cast(selected, DataType.INT32, SPAN), DataType.INDEX, SPAN)
        for number, group in enumerate(groups):
            for duplicate in range(2):
                result = ir.Var(f"{prefix}_slot{number}_{duplicate}", element, SPAN)
                statements.append(
                    ir.AssignStmt(result, internal("buffer.slot", [group, selected], element), SPAN)
                )
        return ir.SeqStmts(statements, SPAN)

    body = selections("body")
    if mode == "branches":
        body = ir.SeqStmts(
            [
                ir.IfStmt(
                    ir.Eq(index, integer(0), DataType.BOOL, SPAN),
                    selections("then"),
                    selections("else"),
                    [],
                    SPAN,
                ),
                selections("after"),
            ],
            SPAN,
        )
    loop = ir.ForStmt(index, integer(start), integer(stop), integer(step), [], body, [], SPAN)
    if mode == "reset":
        outer = ir.Var("visit", ir.ScalarType(DataType.INDEX), SPAN)
        loop = ir.ForStmt(outer, integer(0), integer(3), integer(1), [], loop, [], SPAN)
    function = ir.Function(
        "kernel",
        [foreign] if mode == "foreign" else [],
        [],
        ir.SeqStmts([*allocations, loop], SPAN),
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "DirectSlots", SPAN)


@pytest.mark.parametrize(
    "start,stop,step", [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 2, 1), (0, 2, 2), (1, 2, (1 << 63) - 2)]
)
def test_direct_index_range_and_safe_increment(start, stop, step):
    program = direct_slots(start, stop, step)
    before = ir.serialize(program)
    native = codegen.PTOCodegen().generate(program, emit_tile_addr=True)
    assert native.count("arith.remui") == 1
    assert native.count("pto.multi_tile_get") == 2
    assert ir.serialize(program) == before


@pytest.mark.parametrize(
    "options",
    [
        {"stop": 3},
        {"start": -1},
        {"start": 1, "step": (1 << 63) - 1},
        {"dtype": DataType.INT32},
        {"mode": "narrow"},
        {"mode": "foreign"},
    ],
)
def test_unproved_or_narrow_direct_indices_still_refuse(options):
    with pytest.raises(ValueError):
        codegen.PTOCodegen().generate(direct_slots(**options), emit_tile_addr=True)


@pytest.mark.parametrize(
    "mode,counts,remainders",
    [
        ("alias", (2,), 1),
        ("reset", (2,), 1),
        ("branches", (2,), 3),
        ("direct", (2, 3), 2),
    ],
)
def test_normalization_is_per_slot_use_and_lexical_scope(mode, counts, remainders, tmp_path):
    program = direct_slots(mode=mode, counts=counts)
    native = codegen.PTOCodegen().generate(program, emit_tile_addr=True)
    assert native.count("arith.remui") == remainders
    source, output = tmp_path / "slots.pto", tmp_path / "slots.cpp"
    source.write_text(native)
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    assert output.stat().st_size > 0


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_normalized_slot_does_not_replace_original_gm_coordinate(planner, tmp_path):
    original = pl.parse_program("""
@pl.program
class ShortSlotLoop:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, source: pl.Tensor[[2, 32], pl.FP32],
               target: pl.Out[pl.Tensor[[2, 32], pl.FP32]]) -> pl.Tensor[[2, 32], pl.FP32]:
        for i in pl.range(2):
            slot = i
            value: pl.Tile[[1, 32], pl.FP32, pl.MemRef("ring", slots=2)[slot], pl.Mem.Vec] = pl.load(
                source, [i, 0], [1, 32])
            target = pl.store(value, [i, 0], target)
        return target
""")
    before = ir.serialize(original)
    lowered = _lower(original, planner)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    pending, slots, loads = [function.body], [], []
    while pending:
        statement = pending.pop()
        if isinstance(statement, ir.SeqStmts):
            pending.extend(statement.stmts)
        elif isinstance(statement, ir.ForStmt):
            pending.append(statement.body)
        elif isinstance(statement, ir.AssignStmt) and isinstance(statement.value, ir.Call):
            if statement.value.op.name == ir.get_op("buffer.slot").name:
                slots.append(statement.value.args[1])
        elif isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call):
            if statement.expr.op.name == ir.get_op("buffer.load").name:
                loads.append(statement.expr)
    assert slots and loads
    assert all(isinstance(slot, ir.FloorMod) for slot in slots)
    row = loads[0].args[1].elements[0]
    assert not isinstance(row, ir.FloorMod)
    assert slots[0].left is row
    assert ir.serialize(original) == before
    source, output = tmp_path / "payload.pto", tmp_path / "payload.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    )
    _run_ptoas(
        str(source),
        str(output),
        [
            "--pto-level=" + ("level2" if planner == passes.MemoryPlanner.PTOAS else "level3"),
            "--pto-arch=a2",
            "--enable-insert-sync=true",
        ],
    )
    assert output.stat().st_size > 0
