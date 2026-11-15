# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static operation evidence does not require exact loop/coordinate admission."""

import pypto.language as pl
import pytest
from pypto import ir, passes, testing

from .test_select_buffer_ir_functions import _kernel, _lower
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


class Calls(ir.IRVisitor):
    def __init__(self):
        super().__init__()
        self.calls = []

    def visit_call(self, op):
        self.calls.append(op)
        super().visit_call(op)


def _calls(function, names):
    visitor = Calls()
    visitor.visit_function(function)
    registered = {ir.get_op(name).name for name in names}
    return [call for call in visitor.calls if call.op.name in registered]


@pytest.mark.parametrize("control", ["if", "while"])
def test_static_effects_survive_unknown_control(control):
    body = {
        "if": """if count > 0:
            value = pl.add(value, value)
        else:
            value = pl.sub(value, value)""",
        "while": """index = 0
        while index < count:
            value = pl.add(value, value)
            index = index + 1""",
    }[control]
    program = pl.parse_program(f"""
@pl.program
class StaticContracts:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], count: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32])
        {body}
        return pl.store(value, [0, 0], out)
""")
    function = _kernel(_lower(program, passes.MemoryPlanner.PYPTO))
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    before = ir.serialize(function)
    calls = _calls(function, ["buffer.load", "buffer.store", "buffer.add", "buffer.sub"])
    assert len(calls) == (4 if control == "if" else 3)
    for call in calls:
        description = testing.describe_buffer_sync_operation(call, function.func_type)
        ir.assert_structural_equal(description["operation"], call)
        assert len(description["arguments"]) == len(call.args)
        assert not description["footprint_available"]
        assert not description["readiness_available"]
        assert not description["resource_assignment_available"]
    # Static contracts neither specialize the condition nor manufacture a count.
    # Existing exact admission still refuses these unknown-control regions.
    with pytest.raises(ValueError):
        testing.analyze_placed_sync(function, True)
    assert ir.serialize(function) == before


def test_gm_metadata_and_non_memory_effects_are_distinct():
    function = vec_fixture()
    calls = _calls(function, ["buffer.load", "buffer.store"])
    assert calls
    for call in calls:
        info = testing.describe_buffer_sync_operation(call, function.func_type)
        effects = info["arguments"]
        assert effects[0]["data"] == "read"
        assert effects[3]["data"] == "write"
        assert effects[0]["metadata"] == effects[3]["metadata"] == "read"
        assert not effects[0]["non_memory"] and not effects[3]["non_memory"]
        for index in (1, 2):
            assert effects[index]["data"] == effects[index]["metadata"] == "none"
            assert effects[index]["non_memory"]
        for index, effect in enumerate(effects):
            ir.assert_structural_equal(effect["operand"], call.args[index])
    # The production exact client consumes this same static contract and still
    # constructs its existing physical access model.
    assert testing.analyze_placed_sync(function, True).describe()["regions"]


def test_in_place_operands_keep_separate_modes_without_disjointness():
    function = vec_fixture(inplace=True)
    call = _calls(function, ["buffer.add"])[0]
    info = testing.describe_buffer_sync_operation(call, function.func_type)
    assert [effect["data"] for effect in info["arguments"]] == ["read", "read", "write"]
    ir.assert_structural_equal(info["arguments"][0]["operand"], info["arguments"][2]["operand"])
    assert not info["footprint_available"]


def test_scalar_effect_does_not_prove_ready_value():
    function = vec_fixture(operation="buffer.adds")
    call = _calls(function, ["buffer.adds"])[0]
    scalar = ir.Var("runtime_factor", call.args[1].type, call.span)
    dynamic = ir.Call(call.op, [call.args[0], scalar, call.args[2]], call.type, call.span)
    info = testing.describe_buffer_sync_operation(dynamic, function.func_type)
    assert info["arguments"][1]["non_memory"]
    assert info["arguments"][1]["data"] == info["arguments"][1]["metadata"] == "none"
    assert not info["readiness_available"]


def test_unsupported_native_variant_is_not_empty_effects():
    function = vec_fixture()
    call = _calls(function, ["buffer.add"])[0]
    # Vec->Vec copy is valid Buffer IR but not in the selected native sync subset.
    copy = ir.Call(ir.get_op("buffer.copy"), [call.args[0], call.args[2]], call.type, call.span)
    with pytest.raises(ValueError, match="Only Mat to operand-buffer movement"):
        testing.describe_buffer_sync_operation(copy, function.func_type)


def test_malformed_call_is_not_a_qualification_refusal():
    call = ir.Call(ir.get_op("buffer.load"), [], ir.VoidType(), ir.Span.unknown())
    with pytest.raises(ValueError, match="argument|requires|operand"):
        testing.describe_buffer_sync_operation(call, ir.FunctionType.AIV)
