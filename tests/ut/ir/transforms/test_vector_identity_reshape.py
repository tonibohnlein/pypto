# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Identity vector relabels preserve the original active axis and storage."""

import re

import pypto.language as pl
import pytest
import torch
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import PassManager

from .buffer_test_utils import statements
from .test_runtime_valid_broadcast import check_runtime_commands

pytestmark = pytest.mark.usefixtures("ascend_backend")
PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]


def parse_length_program(source, length):
    # Keep the fixture self-contained: parse_program need not execute arbitrary
    # module-level constant assignments to resolve physical shape annotations.
    source = source.replace(f"L = {length}\n", "")
    return pl.parse_program(re.sub(r"\bL\b", str(length), source))


def identity_program(orientation="column", length=8, *, visits=3, replay=2, varying=False):
    column = "column = pl.reshape(column, [L,1])" if orientation == "column" else ""
    row = "row = pl.reshape(row, [1,L])" if orientation == "row" else ""
    active = "pl.min(bounded, visit % L)" if varying else "bounded"
    return parse_length_program(
        f"""
L = {length}
@pl.program
class IdentityVector:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[L,16], pl.FP32], rows: pl.Scalar[pl.INDEX],
               choice: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[L,16], pl.FP32]]) -> pl.Tensor[[L,16], pl.FP32]:
        for outer in pl.range({replay}):
            for visit in pl.range({visits}):
                if visit % 2 == choice:
                    bounded = pl.min(pl.max(rows,0),L)
                    active = {active}
                    source = pl.load(x, [0,0], [L,16], valid_shape=[active,16])
                    workspace = pl.create_tile([L,16], dtype=pl.FP32)
                    column = pl.row_sum(pl.mul(source,source), workspace)
                    {column}
                    row = pl.reshape(column, [1,L])
                    {row}
                    mixed = pl.add(pl.tile.full([1,L], value=0.0, dtype=pl.FP32), row)
                    shifted = pl.add(mixed,1.0)
                    carrier = pl.reshape(shifted,[L,1])
                    result = pl.row_expand_mul(source,carrier)
                    out = pl.store(result,[0,0],out)
        return out
""",
        length,
    )


def lower(program, planner=passes.MemoryPlanner.PYPTO):
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        result = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
    function = result.get_function("kernel")
    assert function is not None
    return result, function


def synchronize(program, disjoint=False):
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=disjoint):
        logical = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical = passes.lower_sync()(logical)
    before, after = [candidate.get_function("kernel") for candidate in (logical, physical)]
    assert before is not None and after is not None
    return before, after, physical


def statement_calls(program):
    result = []
    for statement in statements(program):
        value = (
            statement.value
            if isinstance(statement, ir.AssignStmt)
            else (statement.expr if isinstance(statement, ir.EvalStmt) else None)
        )
        if isinstance(value, ir.Call):
            result.append(value)
    return result


def payload_calls(program):
    return [
        statement.expr
        for statement in statements(program)
        if isinstance(statement, ir.EvalStmt)
        and isinstance(statement.expr, ir.Call)
        and statement.expr.op.name not in (ir.get_op("buffer.set_validshape").name,)
    ]


@pytest.mark.parametrize("orientation", ["row", "column"])
@pytest.mark.parametrize("length", [8, 16])
@pytest.mark.parametrize("disjoint", [False, True])
def test_identity_preserves_prefix_values_addresses_and_actual_commands(orientation, length, disjoint):
    program, source = lower(identity_program(orientation, length))
    assert source.ir_stage == ir.FunctionIRStage.Buffer
    native = payload_calls(program)
    assert all(call.op.name != ir.get_op("buffer.copy").name for call in native)
    assert all(call.op.name != ir.get_op("buffer.reshape").name for call in statement_calls(program))
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    logical, physical, _ = synchronize(program, disjoint)
    x = torch.arange(length * 16, dtype=torch.float32).reshape(length, 16) / 31
    for rows in (0, 1, length - 1, length):
        for choice in (0, 1, 99):
            output = torch.full_like(x, -777)
            namespace["kernel"](x, rows, choice, output)
            expected = torch.full_like(output, -777)
            if choice in (0, 1):
                expected[:rows] = x[:rows] * ((x[:rows] ** 2).sum(dim=1, keepdim=True) + 1)
            torch.testing.assert_close(output, expected)
            before, after = check_runtime_commands(logical, physical, {1: rows, 2: choice})
            assert before.payloads == after.payloads
            assert not after.live


@pytest.mark.parametrize("orientation", ["row", "column"])
def test_identity_prefix_changes_across_counter_resets(orientation):
    program, source = lower(identity_program(orientation, 16, varying=True))
    assert source.ir_stage == ir.FunctionIRStage.Buffer
    logical, physical, _ = synchronize(program)
    for rows in (0, 1, 15, 16):
        for choice in (0, 1):
            check_runtime_commands(logical, physical, {1: rows, 2: choice})


def alias_program(length=16, fixed=None):
    active = "pl.min(pl.max(rows,0),L)" if fixed is None else str(fixed)
    return parse_length_program(
        f"""
L = {length}
@pl.program
class IdentityAliases:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[L,16], pl.FP32], rows: pl.Scalar[pl.INDEX],
               other: pl.Scalar[pl.INDEX], a: pl.InOut[pl.Tensor[[1,L], pl.FP32]],
               b: pl.InOut[pl.Tensor[[1,L], pl.FP32]]) -> pl.Tensor[[1,L], pl.FP32]:
        active = {active}
        second = pl.min(pl.max(other,0),active)
        source = pl.load(x,[0,0],[L,16], valid_shape=[active,16])
        workspace = pl.create_tile([L,16],dtype=pl.FP32)
        column = pl.row_sum(source,workspace)
        original = pl.reshape(column,[L,1])
        changed = pl.tile.set_validshape(column,second,1)
        row_a = pl.reshape(original,[1,L])
        a = pl.store(row_a,[0,0],a)
        row_b = pl.reshape(changed,[1,L])
        b = pl.store(row_b,[0,0],b)
        row_again = pl.reshape(original,[1,L])
        a = pl.store(row_again,[0,0],a)
        return a
""",
        length,
    )


@pytest.mark.parametrize("fixed", [None, 1, 15, 16])
def test_identity_old_alias_survives_intervening_setter(fixed):
    program, source = lower(alias_program(fixed=fixed))
    assert source.ir_stage == ir.FunctionIRStage.Buffer
    logical, physical, _ = synchronize(program)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    x = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16) / 13
    for rows in (0, 1, 15, 16):
        for other in (0, 1, 16):
            a, b = [torch.full((1, 16), -777.0) for _ in range(2)]
            namespace["kernel"](x, rows, other, a, b)
            expected_a, expected_b = [torch.full((1, 16), -777.0) for _ in range(2)]
            active = rows if fixed is None else fixed
            second = min(other, active)
            expected_a[:, :active] = x[:active].sum(dim=1).reshape(1, -1)
            expected_b[:, :second] = x[:second].sum(dim=1).reshape(1, -1)
            torch.testing.assert_close(a, expected_a)
            torch.testing.assert_close(b, expected_b)
            check_runtime_commands(logical, physical, {1: rows, 2: other})


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("orientation", ["row", "column"])
def test_identity_uses_existing_addressed_planners_and_native_cpp(planner, orientation, tmp_path):
    program, source = lower(identity_program(orientation, 16, visits=1, replay=1), planner)
    assert source.ir_stage == ir.FunctionIRStage.Buffer
    _, _, physical = synchronize(program)
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS compiler unavailable")
    pto, cpp = tmp_path / "identity.pto", tmp_path / "identity.cpp"
    pto.write_text(codegen.PTOCodegen().generate(physical))
    _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2"])
    assert cpp.is_file()


def test_identity_stays_compact_as_visits_increase():
    counts = []
    for visits in (3, 257, 65537):
        program, source = lower(identity_program("column", 16, visits=visits))
        assert source.ir_stage == ir.FunctionIRStage.Buffer
        counts.append((len(list(statements(program))), len(payload_calls(program))))
    assert len(set(counts)) == 1


@pytest.mark.parametrize("variant", ["window", "matrix", "physical_l1"])
def test_identity_does_not_admit_new_geometry_or_window_provenance(variant):
    length = 1 if variant == "physical_l1" else 16
    source_shape = [2, length] if variant == "window" else [length, length] if variant == "matrix" else [1, 1]
    target_shape = [1, length] if variant == "window" else source_shape
    view = f"pl.slice(source,[1,{length}],[0,0])" if variant == "window" else "source"
    program = pl.parse_program(f"""
@pl.program
class UnsupportedIdentity:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[{source_shape}, pl.FP32], rows: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[{source_shape}, pl.FP32]]) -> pl.Tensor[{source_shape}, pl.FP32]:
        source = pl.load(x,[0,0],{source_shape})
        dynamic = pl.tile.set_validshape(source,rows,{source_shape[1]})
        out = pl.store(dynamic,[0,0],out)
        view = {view}
        identity = pl.reshape(view,{target_shape})
        out = pl.store(identity,[0,0],out)
        return out
""")
    lowered, function = lower(program)
    assert function.ir_stage == ir.FunctionIRStage.Functional
    assert "buffer.reshape" not in str(lowered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
