# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Selected native scratch shares physical storage with explicit Buffer effects.

The 8KiB envelope is independently declared from pinned A2/A3 helper source.
These checks prove selected-model behavior, not native instruction footprints.
"""

import pytest
import torch
from pypto import ir, passes, testing
from pypto.debug import torch_codegen

from .sync_matmul_oracle import interpret, validate
from .test_autosync_switch import clone
from .test_contiguous_vector_views import reduction_vector_program
from .test_physical_sync_realization import check
from .test_select_buffer_ir_functions import _PLANNERS, _compile_dynamic_state_native, _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")
SCRATCH = (184 * 1024, 192 * 1024)


def independent_broadcasts(names=("sub", "div"), count=None, explicit_alias=False, operand_alias=False):
    span = ir.Span.unknown()
    fp, index = ir.DataType.FP32, ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index, span)

    def tup(values):
        return ir.MakeTuple([ci(value) for value in values], span)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    definitions: list[ir.Stmt] = []

    def alloc(name, address, shape=(8, 8), column=False):
        ty = ir.BufferType(
            list(shape),
            fp,
            ir.MemorySpace.Vec,
            blayout=ir.TileLayout.col_major if column else ir.TileLayout.row_major,
        )
        value = ir.Var(name, ty, span)
        definitions.append(ir.AssignStmt(value, call("buffer.alloc", [tup([]), ci(address)], ty), span))
        return value

    a = ir.Var("a", ir.TensorType([8, 8], fp), span)
    out = ir.Var("out", ir.TensorType([8 * len(names), 8], fp), span)
    loads: list[ir.Stmt] = []
    computations: list[ir.EvalStmt] = []
    stores: list[ir.Stmt] = []
    for i, name in enumerate(names):
        x = alloc(f"x{i}", SCRATCH[0] if operand_alias and i == 0 else i * 1024)
        row = alloc(f"row{i}", i * 1024 + 256, (1, 8))
        column = alloc(f"column{i}", i * 1024 + 256, (8, 1), True)
        z = alloc(f"z{i}", i * 1024 + 512)
        loads.extend(
            [
                payload("buffer.load", [a, tup([0, 0]), tup([8, 8]), x]),
                payload("buffer.load", [a, tup([0, 0]), tup([1, 8]), row]),
                # The zero-trip case still stores initialized payload data.
                payload("buffer.load", [a, tup([0, 0]), tup([8, 8]), z]),
            ]
        )
        computations.append(payload(f"buffer.row_expand_{name}", [x, column, z]))
        stores.append(payload("buffer.store", [z, tup([8 * i, 0]), tup([8, 8]), out]))
    if explicit_alias:
        alias = alloc("explicit_scratch_alias", SCRATCH[0])
        loads.append(payload("buffer.load", [a, tup([0, 0]), tup([8, 8]), alias]))
    scope = None
    if count is not None:
        scope = ir.ForStmt(
            ir.Var("iteration", ir.ScalarType(index), span),
            ci(0),
            ci(count),
            ci(1),
            [],
            ir.SeqStmts([*computations, ir.YieldStmt([], span)], span),
            [],
            span,
        )
    body: list[ir.Stmt] = [
        *definitions,
        *loads,
        *([scope] if scope is not None else computations),
        *stores,
        ir.ReturnStmt([out], span),
    ]
    function = ir.Function(
        "broadcasts",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(body, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    call_nodes: list[ir.Call] = []
    for item in computations:
        assert isinstance(item.expr, ir.Call)
        call_nodes.append(item.expr)
    return function, call_nodes, scope


@pytest.mark.parametrize("name", ["sub", "div", "mul"])
def test_selected_native_broadcast_retains_implicit_enclosure(name):
    function, calls, _ = independent_broadcasts((name,))
    info = testing.describe_buffer_sync_operation(calls[0], function.func_type)
    assert [item["data"] for item in info["arguments"]] == ["read", "read", "write"]
    assert info["implicit_effects"] == [
        {
            "space": ir.MemorySpace.Vec,
            "begin": SCRATCH[0],
            "end": SCRATCH[1],
            "data": "read_write",
            "provenance": "a2a3-row-broadcast-mode1-implicit-scratch",
            "exact_footprint": False,
        }
    ]
    with pytest.raises(ValueError, match="implicit effects require"):
        testing.describe_buffer_sync_operation(calls[0], function.func_type, "unknown")


@pytest.mark.parametrize("names", [("sub", "div"), ("mul", "sub"), ("div", "mul")])
@pytest.mark.parametrize("count", [None, 0, 1, 3])
def test_disjoint_broadcasts_still_order_shared_scratch(names, count):
    function, _, _ = independent_broadcasts(names, count)
    retained = testing.analyze_placed_sync(function, True)
    validate(function, retained)
    description = retained.describe()
    assert len([s for s in description["storage"] if s["space"] == ir.MemorySpace.Vec]) == 1
    for region in description["regions"]:
        for site in region["sites"]:
            if "row_expand" not in site["original"].op.name:
                continue
            accesses = [
                (a, p)
                for a, p in zip(site["accesses"], site["access_precision"], strict=True)
                if a[3] == SCRATCH[0]
            ]
            assert len(accesses) == 2
            assert {a[1] for a, _ in accesses} == {False, True}
            assert all(a[7] == SCRATCH[1] - SCRATCH[0] and p == "enclosure" for a, p in accesses)
    physical, _, _, _ = check(function, list(range(6)))
    if count != 0:
        assert physical["barrier_sites"] > 0
    # Concrete operand buffers are pairwise disjoint; the independent declared
    # scratch model alone connects these two different broadcast invocations.
    calls = [c for c in interpret(function)[1] if "row_expand" in c[0].op.name]
    if len(calls) >= 2:
        common = calls[0][4] & calls[1][3]
        assert common == {(ir.MemorySpace.Vec, b) for b in range(*SCRATCH)}


def test_implicit_scratch_aliases_explicit_placed_bytes():
    function, _, _ = independent_broadcasts(explicit_alias=True)
    retained = testing.analyze_placed_sync(function, True)
    validate(function, retained)
    assert len([s for s in retained.describe()["storage"] if s["space"] == ir.MemorySpace.Vec]) == 1


def test_native_operand_cannot_overlap_its_own_implicit_workspace():
    function, _, _ = independent_broadcasts(operand_alias=True)
    before = ir.serialize(function)
    with pytest.raises(ValueError, match="Native implicit workspace separation is unproved"):
        testing.analyze_placed_sync(function, True)
    assert ir.serialize(function) == before


def test_implicit_scratch_does_not_invent_disjointness_from_a_slot_domain():
    function, _, _ = independent_broadcasts()
    span = function.span
    element = ir.BufferType([8, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    group_type = ir.MultiBufferType(element, 2)
    group = ir.Var("reserved_overlap", group_type, span)
    declaration = ir.AssignStmt(
        group,
        ir.Call(
            ir.get_op("buffer.alloc_multi"),
            [ir.MakeTuple([], span), ir.ConstInt(SCRATCH[0], ir.DataType.INDEX, span)],
            group_type,
            span,
        ),
        span,
    )
    assert isinstance(function.body, ir.SeqStmts)
    function = clone(function, body=ir.SeqStmts([declaration, *function.body.stmts], span))
    before = ir.serialize(function)
    with pytest.raises(ValueError, match="common physical-layout partition"):
        testing.analyze_placed_sync(function, True)
    assert ir.serialize(function) == before


def test_implicit_native_interface_stays_compact_in_trip_count():
    sizes = []
    for count in (3, 31, 1023):
        function, _, _ = independent_broadcasts(count=count)
        d = testing.analyze_placed_sync(function, True).describe()
        sizes.append((d["cells"], len(d["storage"]), tuple(len(r["demands"]) for r in d["regions"])))
    assert len(set(sizes)) == 1


@pytest.mark.parametrize("planner", _PLANNERS)
def test_row_multiply_reuses_conversion_view_and_row_family(planner, tmp_path):
    # The same general reduction/view pipeline now consumes the existing MUL
    # recipe. No Qwen names, dimensions, caller restrictions or dispatch cases.
    class Multiply(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("tile.row_expand_sub").name:
                return ir.Call(ir.get_op("tile.row_expand_mul"), op.args, op.type, op.span)
            return super().visit_call(op)

    program = Multiply().visit_program(reduction_vector_program())
    lowered = _lower(program, planner)
    function = _kernel(lowered)
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    x = torch.arange(2048).reshape(16, 128).float() / 2048
    output = torch.empty_like(x)
    namespace["kernel"](x, output)
    torch.testing.assert_close(output, x * (x.sum(dim=1, keepdim=True) + 2))
    _compile_dynamic_state_native(lowered, tmp_path, planner)
    if planner != passes.MemoryPlanner.PTOAS:
        validate(function, testing.analyze_placed_sync(function, True))
