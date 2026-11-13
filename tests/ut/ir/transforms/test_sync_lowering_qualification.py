# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Token lowering qualifies current payloads without rebuilding access demands."""

import pytest
from pypto import InternalError, ir, passes, testing
from pypto.ir.op import sync

from .sync_matmul_cases import staged_fixture
from .test_autosync_switch import clone
from .test_parameterized_token_sync import apply, callers, check, worker
from .test_token_sync_lowering import _apply, _two_streams

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("unused", ["index", "float", "tensor"])
def test_unused_formal_does_not_require_dependence_admission(structured, unused):
    original, logical = _two_streams()
    if structured:
        original = staged_fixture(4)
        logical = _apply(passes.insert_sync, original)
    extra_type = {
        "index": ir.ScalarType(ir.DataType.INDEX),
        "float": ir.ScalarType(ir.DataType.FP32),
        "tensor": ir.TensorType([2, 3, 4], ir.DataType.FP32),
    }[unused]
    extra = ir.Var("unused", extra_type, logical.span)

    def append_parameter(function):
        return ir.Function(
            function.name,
            [*zip(function.params, function.param_directions), extra],
            function.return_types,
            function.body,
            function.span,
            type=function.func_type,
            ir_stage=function.ir_stage,
        )

    logical = append_parameter(logical)
    physical = _apply(passes.lower_sync, logical)
    assert physical.params[-1].same_as(extra)
    assert "sync.record" not in str(physical)
    assert "system.set_flag" in str(physical)
    # This deliberately does not expand insertion's access-model admission.
    with pytest.raises(ValueError, match="coordinate|tensor arguments"):
        _apply(passes.insert_sync, append_parameter(original))


def test_generated_verification_is_internal_but_user_validation_is_not():
    _, logical = _two_streams()
    testing.check_generated_logical_sync(logical)
    missing = ir.Var("undefined_completion", ir.SyncTokenType(ir.PipeType.MTE2), logical.span)
    invalid = clone(
        logical,
        body=ir.SeqStmts(
            [ir.EvalStmt(sync.wait(missing, ir.PipeType.V), logical.span), logical.body], logical.span
        ),
    )
    before = str(invalid)
    with pytest.raises(InternalError, match="insertion produced invalid logical IR"):
        testing.check_generated_logical_sync(invalid)
    with pytest.raises(ValueError):
        _apply(passes.lower_sync, invalid)
    assert str(invalid) == before


def test_token_lowering_does_not_require_a_common_gm_translation():
    original = staged_fixture(2)
    logical = _apply(passes.insert_sync, original)
    span = original.span
    offset = ir.Var("offset", ir.ScalarType(ir.DataType.INDEX), span)

    def translate(function):
        class TranslateFirst(ir.IRMutator):
            changed = False

            def visit_call(self, op):
                if op.op.name != ir.get_op("buffer.load").name or self.changed:
                    return op
                self.changed = True
                args = list(op.args)
                rows, columns = args[1].elements
                args[1] = ir.MakeTuple([rows, ir.Add(columns, offset, ir.DataType.INDEX, span)], span)
                return ir.Call(op.op, args, op.type, span)

        return ir.Function(
            function.name,
            [*zip(function.params, function.param_directions), offset],
            function.return_types,
            TranslateFirst().visit_stmt(function.body),
            span,
            type=function.func_type,
            ir_stage=function.ir_stage,
        )

    original, logical = translate(original), translate(logical)
    physical = worker(apply(passes.lower_sync, callers(logical, starts=(0, 1))), logical.name)
    for value in (0, 1):
        check(original, logical, physical, {3: value}, minimum_protocol=False)
    with pytest.raises(ValueError, match="common invocation translation"):
        apply(passes.insert_sync, callers(original, starts=(0, 1)))


def test_lowering_keeps_placed_descriptor_qualification():
    _, logical = _two_streams()

    class RemovePlacement(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("buffer.alloc").name:
                return ir.Call(op.op, list(op.args[:1]), op.type, op.span)
            return super().visit_call(op)

    unplaced = RemovePlacement().visit_function(logical)
    with pytest.raises(ValueError, match="placed|address"):
        _apply(passes.lower_sync, unplaced)
