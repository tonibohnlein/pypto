# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""First unmet Buffer obligation survives nested traversal and production cloning."""

import pypto.language as pl
import pytest
from pypto import InternalError, ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

from .qualify_qwen_whole_model import function_inventory
from .test_select_buffer_ir_functions import _control_flow_program, _kernel, _program

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _select(program, enabled=True, include_selection=True):
    with passes.PassContext([], enable_buffer_ir=True):
        manager = PassManager(OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        for transform in manager.passes[:boundary]:
            program = transform(program)
    if include_selection:
        with passes.PassContext([], enable_buffer_ir=enabled):
            program = passes.select_buffer_ir_functions()(program)
    return program


def _obligation(program):
    attrs = dict(_kernel(program).attrs)
    assert attrs["buffer_ir_selected"] is False
    assert attrs["buffer_admission_version"] == 1
    assert attrs["buffer_admission_stage"] == "SelectBufferIRFunctions"
    return {
        k.removeprefix("buffer_admission_"): v for k, v in attrs.items() if k.startswith("buffer_admission_")
    }


def _atomic_program():
    return pl.parse_program("""
@pl.program
class NestedObligation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], count: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        value = pl.load(x, [0, 0], [16, 32])
        for i in pl.range(count):
            if i == 0:
                out = pl.store(value, [0, 0], out, atomic=pl.AtomicType.Add)
            else:
                out = pl.store(value, [0, 0], out)
        return out
""")


def test_nested_atomic_first_obligation_and_roundtrip():
    selected = _select(_atomic_program())
    obligation = _obligation(selected)
    assert obligation["property"] == "atomic_store_recipe"
    assert obligation["operation"] == ir.get_op("tile.store").name
    # A nested operation must not be blamed on its enclosing loop/branch.
    assert obligation["location"]
    restored = ir.deserialize(ir.serialize(selected))
    assert isinstance(restored, ir.Program)
    assert _obligation(restored) == obligation
    again = passes.select_buffer_ir_functions()(restored)
    ir.assert_structural_equal(restored, again, enable_auto_mapping=True)


@pytest.mark.parametrize("enabled,expected", [(False, "buffer_ir_disabled"), (True, "positive_loop_step")])
def test_broad_precondition_precedes_nested_failure(enabled, expected):
    selected = _select(_control_flow_program("for", "step"), enabled)
    obligation = _obligation(selected)
    assert obligation["property"] == expected
    assert obligation["operation"] == ("Function" if not enabled else "ForStmt")


def test_success_has_no_refusal_and_absence_is_not_success():
    original = _program("add")
    assert function_inventory(original)[0]["buffer_admission"]["status"] == "not_evaluated"
    selected = _select(original)
    assert _kernel(selected).attrs["buffer_ir_selected"] is True
    assert not any(k.startswith("buffer_admission_") for k in _kernel(selected).attrs)
    assert function_inventory(selected)[0]["buffer_admission"] == {"status": "selected"}


def test_production_refusal_keeps_first_obligation():
    # Run the actual default pipeline and RealizeSync, not an isolated formatter.
    with passes.PassContext([], enable_buffer_ir=True, enable_pypto_autosync=True, autosync_disjoint_gm=True):
        result = PassManager(OptimizationStrategy.Default).run_passes(_atomic_program())
    obligation = _obligation(result)
    assert obligation["property"] == "atomic_store_recipe"
    assert "SelectBufferIRFunctions/atomic_store_recipe" in _kernel(result).attrs["autosync_fallback_reason"]
    inventory = function_inventory(result)
    assert len(inventory) == len(result.functions)
    assert (
        next(x for x in inventory if x["name"] == "kernel")["buffer_admission"]["property"]
        == obligation["property"]
    )


def test_dynamic_cast_names_first_recipe_obligation():
    program = pl.parse_program("""
@pl.program
class DynamicCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 32], pl.FP32], rows: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP16]]) -> pl.Tensor[[16, 32], pl.FP16]:
        value = pl.load(x, [0, 0], [16, 32], valid_shape=[rows, 32])
        converted = pl.cast(value, pl.FP16)
        return pl.store(converted, [0, 0], out)
""")
    obligation = _obligation(_select(program))
    # The per-operation cast recipe rejects runtime valid extents before the
    # later whole-function dynamic-metadata compatibility check is reached.
    assert obligation["property"] == "operation_recipe"
    assert obligation["operation"] == ir.get_op("tile.cast").name


def test_malformed_kwarg_is_not_a_capability_refusal():
    class MalformedCache(ir.IRMutator):
        def visit_call(self, op):
            call = super().visit_call(op)
            assert isinstance(call, ir.Call)
            if call.op.name != ir.get_op("tile.load").name:
                return call
            kwargs = dict(call.kwargs)
            kwargs["cache"] = "malformed"
            return ir.Call(call.op, list(call.args), kwargs, dict(call.attrs), call.type, call.span)

    original = _select(_program("add"), include_selection=False)
    malformed = MalformedCache().visit_program(original)
    with pytest.raises((InternalError, TypeError, ValueError), match="cache|cast|type|integer"):
        passes.select_buffer_ir_functions()(malformed)


def test_scalar_only_domain_is_not_claimed_empty_or_selected():
    span = ir.Span("scalar-admission.py", 3, 1, 3, 6)
    function = ir.Function("kernel", [], [], ir.ReturnStmt([], span), span, type=ir.FunctionType.InCore)
    program = ir.Program([function], "ScalarDomain", span)
    selected = passes.select_buffer_ir_functions()(program)
    obligation = _obligation(selected)
    assert obligation["property"] == "load_or_fill_and_store"
    assert obligation["operation"] == "Function"
    assert obligation["location"] == "scalar-admission.py:3:1"
    inventory = function_inventory(selected)
    assert len(inventory) == 1
    assert inventory[0]["buffer_admission"]["status"] == "refused"
