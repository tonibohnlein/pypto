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
from pypto import Error, ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR
from pypto.debug import torch_codegen

from .sync_matmul_oracle import interpret, validate
from .test_autosync_switch import clone
from .test_closed_prefix_sync import Services
from .test_contiguous_vector_views import reduction_vector_program
from .test_physical_sync_realization import check
from .test_select_buffer_ir_functions import _PLANNERS, _compile_dynamic_state_native, _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")
SCRATCH = (184 * 1024, 192 * 1024)


def independent_broadcasts(
    names=("sub", "div"), count=None, explicit_alias=False, operand_alias=False, addresses=None
):
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
        address = (addresses or {}).get(name, address)
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


@pytest.mark.parametrize("name", ["sub", "div", "mul"])
@pytest.mark.parametrize("operand", ["x0", "column0", "z0"])
@pytest.mark.parametrize("route", ["exact", "conservative", "insert", "lower", "realize"])
def test_native_workspace_legality_is_shared_by_all_ownership_routes(name, operand, route):
    function, _, _ = independent_broadcasts((name,), addresses={operand: SCRATCH[0]})
    if route == "lower":
        assert isinstance(function.body, ir.SeqStmts)
        function = clone(
            function,
            body=ir.SeqStmts(
                [*Services().closed(ir.PipeType.MTE2, ir.PipeType.V), *function.body.stmts], function.span
            ),
        )
    before = ir.serialize(function)
    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", autosync_disjoint_gm=True
    ):
        if route == "realize":
            result = passes.realize_sync()(ir.Program([function], "IllegalWorkspace", function.span))
            preserved = next(iter(result.functions.values()))
            ir.assert_structural_equal(preserved.body, function.body)
            assert not dict(preserved.attrs).get("pypto_autosync", False)
            assert (
                "Native implicit workspace separation is unproved"
                in dict(preserved.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
            )
        else:
            with pytest.raises(ValueError, match="Native implicit workspace separation is unproved"):
                if route == "exact":
                    testing.analyze_placed_sync(function, True)
                elif route == "conservative":
                    testing.analyze_conservative_sync(function)
                else:
                    transform = passes.insert_sync() if route == "insert" else passes.lower_sync()
                    transform(ir.Program([function], "IllegalWorkspace", function.span))
    assert ir.serialize(function) == before


@pytest.mark.parametrize("address", [SCRATCH[0] - 256, SCRATCH[1]])
def test_workspace_half_open_edges_remain_qualified(address):
    function, _, _ = independent_broadcasts(("mul",), addresses={"x0": address})
    # A matrix touching, but not intersecting, either edge is not illegal.
    testing.analyze_placed_sync(function, True)
    testing.analyze_conservative_sync(function)


@pytest.mark.parametrize("address", [None, SCRATCH[0], 0])
@pytest.mark.parametrize("alias", [False, True])
def test_conservative_workspace_requires_available_alias_placement(address, alias):
    function, _, _ = independent_broadcasts(("mul",), addresses={"x0": address or 0})
    assert isinstance(function.body, ir.SeqStmts)
    statements = list(function.body.stmts)
    declaration = statements[0]
    assert isinstance(declaration, ir.AssignStmt) and declaration.var.name_hint == "x0"
    parameters = list(zip(function.params, function.param_directions))
    if address is None:
        statements.pop(0)
        parameters.append((declaration.var, ir.ParamDirection.InOut))
    if alias:
        renamed = ir.Var("matrix_alias", declaration.var.type, function.span)
        statements.insert(
            1 if address is not None else 0,
            ir.AssignStmt(
                renamed,
                ir.Call(ir.get_op("buffer.reshape"), [declaration.var], declaration.var.type, function.span),
                function.span,
            ),
        )
        rewritten: list[ir.Stmt] = []
        for statement in statements:
            if isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call):
                call = statement.expr
                if call.op.name == ir.get_op("buffer.row_expand_mul").name:
                    statement = ir.EvalStmt(
                        ir.Call(call.op, [renamed, *call.args[1:]], call.type, call.span), function.span
                    )
            rewritten.append(statement)
        statements = rewritten
    function = ir.Function(
        function.name,
        parameters,
        function.return_types,
        ir.SeqStmts(statements, function.span),
        function.span,
        function.func_type,
        ir_stage=function.ir_stage,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    program = ir.Program([function], "WorkspaceAliasInput", function.span)
    if address is None:
        # Borrowed operands also lack the within-operation allocation provenance
        # required by BufferIR. Preserve that earlier refusal; this is not a
        # verified native program whose missing placement may be assumed away.
        with pytest.raises(Error, match="allocation provenance"):
            passes.verify_properties(properties, program, "input")
    else:
        passes.verify_properties(properties, program, "input")
    if address == 0:
        testing.analyze_conservative_sync(function)
    else:
        with pytest.raises(ValueError, match="Native implicit workspace separation is unproved"):
            testing.analyze_conservative_sync(function)


@pytest.mark.parametrize("kind", ["subview", "slot"])
@pytest.mark.parametrize("address", [0, SCRATCH[0]])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("route", ["conservative", "lower", "realize"])
def test_workspace_view_containment_requires_proved_offset_range(kind, address, dynamic, route):
    function, _, _ = independent_broadcasts(("mul",), addresses={"x0": address})
    assert isinstance(function.body, ir.SeqStmts)
    statements = list(function.body.stmts)
    declaration = statements[0]
    assert isinstance(declaration, ir.AssignStmt)
    original = declaration.var
    span = function.span
    offset = ir.Var("unbounded_offset", ir.ScalarType(ir.DataType.INDEX), span)
    zero = ir.ConstInt(0, ir.DataType.INDEX, span)
    selector = offset if dynamic else zero
    window = ir.Var("window", original.type, span)
    if kind == "subview":
        view = ir.Call(
            ir.get_op("buffer.subview"),
            [original, ir.MakeTuple([selector, zero], span)],
            original.type,
            span,
        )
        statements.insert(1, ir.AssignStmt(window, view, span))
    else:
        assert isinstance(original.type, ir.BufferType)
        multi_type = ir.MultiBufferType(original.type, 2)
        multi = ir.Var("slots", multi_type, span)
        allocation = ir.Call(
            ir.get_op("buffer.alloc_multi"),
            [ir.MakeTuple([], span), ir.ConstInt(address, ir.DataType.INDEX, span)],
            multi_type,
            span,
        )
        view = ir.Call(ir.get_op("buffer.slot"), [multi, selector], original.type, span)
        statements[0:1] = [ir.AssignStmt(multi, allocation, span), ir.AssignStmt(window, view, span)]
    rewritten: list[ir.Stmt] = []
    for statement in statements:
        if isinstance(statement, ir.EvalStmt) and isinstance(statement.expr, ir.Call):
            call = statement.expr
            arguments = [window if argument.same_as(original) else argument for argument in call.args]
            statement = ir.EvalStmt(ir.Call(call.op, arguments, call.type, span), span)
        rewritten.append(statement)
    if route == "lower":
        rewritten = [*Services().closed(ir.PipeType.MTE2, ir.PipeType.V), *rewritten]
    function = ir.Function(
        function.name,
        [*zip(function.params, function.param_directions), (offset, ir.ParamDirection.In)],
        function.return_types,
        ir.SeqStmts(rewritten, span),
        span,
        function.func_type,
        ir_stage=function.ir_stage,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    passes.verify_properties(properties, ir.Program([function], "WorkspaceViewInput", span), "input")
    before = ir.serialize(function)
    refused = address == SCRATCH[0] or (dynamic and kind == "slot")

    def invoke():
        if route == "conservative":
            return testing.analyze_conservative_sync(function)
        transform = passes.lower_sync() if route == "lower" else passes.realize_sync()
        return transform(ir.Program([function], "WorkspaceViewRange", span))

    with passes.PassContext(
        [], enable_pypto_autosync=True, autosync_policy="conservative_prefix", autosync_disjoint_gm=True
    ):
        if route == "realize":
            physical = invoke()
            assert isinstance(physical, ir.Program)
            result = next(iter(physical.functions.values()))
            if refused:
                ir.assert_structural_equal(result.body, function.body)
                assert not dict(result.attrs).get("pypto_autosync", False)
                assert (
                    "Native implicit workspace separation is unproved"
                    in dict(result.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
                )
            else:
                assert dict(result.attrs).get("pypto_autosync", False)
        elif refused:
            with pytest.raises(ValueError, match="Native implicit workspace separation is unproved"):
                invoke()
        else:
            invoke()
    assert ir.serialize(function) == before
