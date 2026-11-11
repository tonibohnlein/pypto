# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Finite Vec import, physical aliases and guarded unsupported interfaces."""

import importlib.util
from pathlib import Path

import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.runtime import RunConfig

from .sync_matmul_cases import staged_fixture
from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .test_autosync_switch import apply_hybrid
from .test_physical_sync_realization import check

pytestmark = pytest.mark.usefixtures("ascend_backend")


def vec_fixture(count=None, visits=1, view="dense", operation="buffer.add", inplace=False):
    """Placed derivative; no fixture name is used by the production importer.

    Overlap denotes separate allocations sharing physical bytes. The nested
    view has eight columns but retains the 32-column pitch of its original root.
    """
    span = ir.Span.unknown()
    fp, index = ir.DataType.FP32, ir.DataType.INDEX

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    definitions = []

    def alloc(name, shape, address, dtype=fp):
        ty = ir.BufferType(shape, dtype, ir.MemorySpace.Vec)
        var = ir.Var(name, ty, span)
        definitions.append(ir.AssignStmt(var, call("buffer.alloc", [tup([]), ci(address)], ty), span))
        return var

    root_shape = [4, 32] if view == "nested" else [2, 8]
    a = ir.Var("a", ir.TensorType(root_shape, fp), span)
    b = ir.Var("b", ir.TensorType([2, 8], fp), span)
    out = ir.Var("out", ir.TensorType([2, 8], fp), span)
    if view == "byte_root":
        raw = alloc("raw", [2, 32], 0, ir.DataType.UINT8)
        ty = ir.BufferType(root_shape, fp, ir.MemorySpace.Vec)
        root = ir.Var("root", ty, span)
        definitions.append(ir.AssignStmt(root, call("buffer.reshape", [raw], ty), span))
    else:
        root = alloc("root", root_shape, 0)
    x = root
    if view == "nested":
        for name, shape, offsets in [("outer", [3, 16], [1, 8]), ("inner", [2, 8], [1, 0])]:
            ty = ir.BufferType(shape, fp, ir.MemorySpace.Vec)
            var = ir.Var(name, ty, span)
            definitions.append(ir.AssignStmt(var, call("buffer.subview", [x, tup(offsets)], ty), span))
            x = var
    elif view == "reshape":
        for name, shape in [("flat", [1, 16]), ("restored", [2, 8])]:
            ty = ir.BufferType(shape, fp, ir.MemorySpace.Vec)
            var = ir.Var(name, ty, span)
            definitions.append(ir.AssignStmt(var, call("buffer.reshape", [x], ty), span))
            x = var
    y = alloc("y", [2, 8], 32 if view == "overlap" else 1024)
    z = x if inplace else alloc("z", [2, 8], 2048)
    body = list(definitions)
    for visit in range(visits):
        args = [x, y, z]
        if operation in ("buffer.abs", "buffer.recip"):
            args = [x, z]
        elif operation == "buffer.adds":
            args = [x, ir.ConstFloat(2.0, fp, span), z]
        statements = [
            payload("buffer.load", [a, tup([0, 0]), tup(root_shape), root]),
            payload("buffer.load", [b, tup([0, 0]), tup([2, 8]), y]),
            payload(operation, args),
            payload("buffer.abs", [z, z]),
            payload("buffer.store", [z, tup([0, 0]), tup([2, 8]), out]),
        ]
        if view == "overlap":
            # The first contents are consumed before the overlapping DMA write.
            # This tests cross-pipe WAR reuse, without requesting an unsupported
            # same-MTE2 completion barrier between two adjacent conflicting loads.
            statements.insert(1, payload("buffer.abs", [x, z]))
        if count is None:
            body.extend(statements)
        else:
            iv = ir.Var(f"i{visit}", ir.ScalarType(index), span)
            body.append(
                ir.ForStmt(
                    iv,
                    ci(0),
                    ci(count),
                    ci(1),
                    [],
                    ir.SeqStmts([*statements, ir.YieldStmt([], span)], span),
                    [],
                    span,
                )
            )
    body.append(ir.ReturnStmt([out], span))
    return ir.Function(
        "finite_vec",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(body, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("view", ["dense", "overlap", "nested", "reshape", "byte_root"])
@pytest.mark.parametrize("count", [None, 0, 1, 3])
def test_vec_end_to_end(view, count):
    original = vec_fixture(count, visits=2 if count is not None else 1, view=view)
    analysis = testing.analyze_placed_sync(original, True)
    validate(original, analysis)
    result, _, _, text = check(original, list(range(6)))
    assert "tload" in text
    transformed = apply_hybrid([original])[original.name]
    assert dict(transformed.attrs)[PYPTO_AUTOSYNC_ATTR] == (count != 0)
    run_physical(original, transformed, range(6))
    if count == 0:
        ir.assert_structural_equal(original.body, transformed.body)
    else:
        assert result["barrier_sites"] >= 1
        assert "pto.barrier" in text or "pto.bar" in text


@pytest.mark.parametrize("operation", ["buffer.add", "buffer.mul", "buffer.sub", "buffer.abs", "buffer.adds"])
@pytest.mark.parametrize("inplace", [False, True])
def test_shared_recipe_modes_and_inplace(operation, inplace):
    original = vec_fixture(operation=operation, inplace=inplace)
    validate(original, testing.analyze_placed_sync(original, True))
    check(original, list(range(6)))


def test_nested_window_keeps_parent_pitch_and_absence():
    original = vec_fixture(view="nested")
    analysis = testing.analyze_placed_sync(original, True)
    validate(original, analysis)
    domain = next(i for i, s in enumerate(analysis.describe()["storage"]) if s["byte_coordinates"])
    # inner begins at row 2, col 8 of a 32-column FP32 root, not an 8-column root.
    inside = analysis.boundary(domain, 0, (2 * 32 + 8) * 4)
    gap = analysis.boundary(domain, 0, (2 * 32 + 16) * 4)
    assert inside is not None and gap is not None
    assert inside["last_readers"][int(ir.PipeType.V)] is not None
    assert gap["last_readers"][int(ir.PipeType.V)] is None
    absent = analysis.boundary(domain, 0, 768)
    assert absent is not None and absent["last_writer"] is None
    assert analysis.boundary(domain, 0, 100000) is None


def test_vec_compact_counts_do_not_expand():
    sizes = []
    for count in [2, 4, 16, 1024, 1000000]:
        analysis = testing.analyze_placed_sync(vec_fixture(count, visits=2, view="nested"), True)
        d = analysis.describe()
        physical = analysis.realize_physical("a2a3", list(range(6)))
        # At two visits some boundary ports coincide; after startup the graph
        # saturates. Both sizes are bounded by the same retained port inventory.
        ports = sum(4 * len(r["sites"]) for r in d["regions"])
        assert d["boundary_graph_edges"] <= ports**2
        sizes.append(
            (
                d["analyzed_occurrences"],
                d["cells"],
                [len(r["demands"]) for r in d["regions"]],
                physical["set_sites"],
                physical["wait_sites"],
            )
        )
    assert all(s == sizes[0] for s in sizes)


def test_partial_inplace_is_an_instruction_refusal():
    class Partial(ir.IRMutator):
        def visit_call(self, op):
            call = op
            if (
                call.op.name == ir.get_op("buffer.alloc").name
                and isinstance(call.args[1], ir.ConstInt)
                and call.args[1].value == 2048
            ):
                return ir.Call(
                    call.op,
                    [call.args[0], ir.ConstInt(32, ir.DataType.INDEX, call.span)],
                    call.type,
                    call.span,
                )
            return call

    original = Partial().visit_function(vec_fixture())
    with pytest.raises(ValueError, match="partially overlapping"):
        testing.analyze_placed_sync(original, True)
    transformed = apply_hybrid([original])[original.name]
    assert not dict(transformed.attrs)[PYPTO_AUTOSYNC_ATTR]
    ir.assert_structural_equal(original.body, transformed.body)


def test_native_helper_fixed_claims_are_reserved():
    original = vec_fixture(operation="buffer.recip")
    analysis = testing.analyze_placed_sync(original, True)
    validate(original, analysis)
    physical = analysis.realize_physical("a2a3", list(range(6)))
    # Manual TRECIP uses fixed ID0 in V->S and S->V. These are separate
    # directional pools from the explicit DMA/V handoffs, but must be reserved.
    assert set(physical["native_reserved_ids"]) == {
        (ir.PipeType.V, ir.PipeType.S, 0),
        (ir.PipeType.S, ir.PipeType.V, 0),
    }
    run_physical(original, physical["function"], range(6))


@pytest.mark.parametrize("name", ["tile_add_64", "tile_mul_64"])
def test_original_vec_examples_use_production_pass(name):
    path = Path(__file__).resolve().parents[4] / "examples/beginner/02_elementwise.py"
    spec = importlib.util.spec_from_file_location("sync_vec_examples", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, name)
    args = [torch.empty(64, 64) for _ in range(3)]
    config = RunConfig(arch=BackendType.Ascend910)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        baseline = kernel.lower(*args, config=config)
    original = next(f for f in baseline.functions.values() if f.func_type == ir.FunctionType.AIV)
    validate(original, testing.analyze_placed_sync(original, True))
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
    ):
        program = kernel.lower(*args, config=config)
    emitted = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    run_physical(original, emitted, range(6))


def test_same_dma_pipe_overlap_remains_unrealized():
    class RemoveConsumer(ir.IRMutator):
        def visit_seq_stmts(self, op):
            stmts = list(op.stmts)
            # Remove the deliberately inserted first V consumer only.
            first = next(
                i
                for i, s in enumerate(stmts)
                if isinstance(s, ir.EvalStmt)
                and isinstance(s.expr, ir.Call)
                and s.expr.op.name == ir.get_op("buffer.abs").name
            )
            del stmts[first]
            return ir.SeqStmts(stmts, op.span)

    original = RemoveConsumer().visit_function(vec_fixture(view="overlap"))
    analysis = testing.analyze_placed_sync(original, True)
    validate(original, analysis)
    with pytest.raises(ValueError, match="same-pipe barrier mechanism"):
        analysis.realize_physical("a2a3", list(range(6)))
    emitted = apply_hybrid([original])[original.name]
    ir.assert_structural_equal(original.body, emitted.body)
    assert not dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]


@pytest.mark.parametrize("view", ["dense", "nested", "reshape", "byte_root"])
def test_vec_downstream_manual_flags_and_cpp(view, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    original = vec_fixture(view=view)
    result, _, _, _ = check(original, list(range(6)))
    source = tmp_path / "vec.pto"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([result["function"]], "vec", original.span), emit_tile_addr=True, emit_source_loc=False
        )
    )
    outputs = []
    for enabled in [False, True]:
        target = tmp_path / f"vec-{enabled}.cpp"
        _run_ptoas(
            str(source),
            str(target),
            ["--pto-level=level3", "--pto-arch=a2", f"--enable-insert-sync={'true' if enabled else 'false'}"],
        )
        outputs.append(target.read_text())
    assert outputs[0] == outputs[1]
    body = outputs[0].split("AICORE void ", 1)[1]
    assert body.count("set_flag(") == result["set_sites"]
    assert body.count("wait_flag(") == result["wait_sites"]


@pytest.mark.parametrize("kind", ["valid_update", "rotating_view", "noncontiguous_reshape"])
def test_unavailable_metadata_and_coordinate_contracts_delegate(kind):
    original = vec_fixture(view="nested")
    span = original.span
    assert isinstance(original.body, ir.SeqStmts)
    if kind == "noncontiguous_reshape":
        # Build a structurally coherent view with the unsupported parent pitch.
        stmts = list(original.body.stmts)
        index = next(
            i for i, s in enumerate(stmts) if isinstance(s, ir.AssignStmt) and s.var.name_hint == "inner"
        )
        definition = stmts[index]
        assert isinstance(definition, ir.AssignStmt) and isinstance(definition.value, ir.Call)
        parent = definition.value.args[0]
        ty = ir.BufferType([3, 16], ir.DataType.FP32, ir.MemorySpace.Vec)
        var = ir.Var("bad_reshape", ty, span)
        stmts.insert(
            index, ir.AssignStmt(var, ir.Call(ir.get_op("buffer.reshape"), [parent], ty, span), span)
        )
        expected = "noncontiguous window"
    else:
        stmts = list(original.body.stmts)
        definition = stmts[0]
        assert isinstance(definition, ir.AssignStmt)
        root = definition.var
        if kind == "valid_update":
            valid = ir.MakeTuple([ir.ConstInt(x, ir.DataType.INDEX, span) for x in [4, 32]], span)
            call = ir.Call(ir.get_op("buffer.set_validshape"), [root, valid], ir.VoidType(), span)
            stmts.insert(1, ir.EvalStmt(call, span))
            expected = "instruction contract"
        else:
            # Ready affine views are now supported, but every visit must fit
            # the parent. A two-row window at its last row exceeds capacity.
            iv = ir.Var("slot", ir.ScalarType(ir.DataType.INDEX), span)
            ty = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
            var = ir.Var("slot_view", ty, span)
            offsets = ir.MakeTuple([iv, ir.ConstInt(0, ir.DataType.INDEX, span)], span)
            call = ir.Call(ir.get_op("buffer.subview"), [root, offsets], ty, span)
            inner = ir.SeqStmts([ir.AssignStmt(var, call, span), ir.YieldStmt([], span)], span)
            assert isinstance(root.type, ir.BufferType)
            values = [ir.ConstInt(x, ir.DataType.INDEX, span) for x in [0, root.type.shape[0], 1]]
            stmts.insert(1, ir.ForStmt(iv, *values, [], inner, [], span))
            expected = "proved parent window"
    function = ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts(stmts, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with pytest.raises(ValueError, match=expected):
        testing.analyze_placed_sync(function, True)
    result = apply_hybrid([function])[function.name]
    assert not dict(result.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert expected in dict(result.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(function.body, result.body)


def test_gm_pitch_is_not_silently_treated_as_packed():
    original = vec_fixture()
    span = original.span
    stride = [ir.ConstInt(x, ir.DataType.INDEX, span) for x in [16, 1]]
    view = ir.TensorView(stride, ir.TensorLayout.ND)
    parameter = ir.Var("a", ir.TensorType([2, 8], ir.DataType.FP32, tensor_view=view), span)

    class Pitched(ir.IRMutator):
        def visit_var(self, op):
            return parameter if op.name_hint == "a" else op

    original = Pitched().visit_function(original)
    with pytest.raises(ValueError, match="unviewed ND tensor arguments"):
        testing.analyze_placed_sync(original, True)
    emitted = apply_hybrid([original])[original.name]
    assert "unviewed ND tensor arguments" in dict(emitted.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(original.body, emitted.body)


@pytest.mark.parametrize("vector", [False, True])
def test_flag_core_ownership_survives_serialization(vector):
    original = vec_fixture() if vector else staged_fixture(2)
    result = testing.analyze_placed_sync(original, True).realize_physical("a2a3", list(range(6)))
    restored = ir.deserialize(ir.serialize(result["function"]))
    assert isinstance(restored, ir.Function)
    run_physical(original, restored, range(6))

    class WrongCore(ir.IRMutator):
        def visit_call(self, op):
            call = op
            if call.op.name in {ir.get_op("system.set_flag").name, ir.get_op("system.wait_flag").name}:
                kwargs = dict(call.kwargs)
                kwargs["core_type"] = "aic" if vector else "aiv"
                return ir.Call(call.op, call.args, kwargs, None, call.type, call.span)
            return call

    with pytest.raises(AssertionError, match="wrong owning core"):
        run_physical(original, WrongCore().visit_function(restored), range(6))


def independent_vec_stages(stages):
    """Different placed buffers share a byte domain, not an all-accesses scan."""
    span = ir.Span.unknown()
    ty = ir.BufferType([1, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    gm = ir.Var("input", ir.TensorType([1, 8], ir.DataType.FP32), span)

    def tup(values):
        return ir.MakeTuple([ir.ConstInt(v, ir.DataType.INDEX, span) for v in values], span)

    definitions, payloads = [], []
    for stage in range(stages):
        src, dst = [ir.Var(f"b{stage}_{i}", ty, span) for i in range(2)]
        for i, var in enumerate([src, dst]):
            address = ir.ConstInt((2 * stage + i) * 32, ir.DataType.INDEX, span)
            call = ir.Call(ir.get_op("buffer.alloc"), [tup([]), address], ty, span)
            definitions.append(ir.AssignStmt(var, call, span))
        for name, args in [("buffer.load", [gm, tup([0, 0]), tup([1, 8]), src]), ("buffer.abs", [src, dst])]:
            payloads.append(ir.EvalStmt(ir.Call(ir.get_op(name), args, ir.VoidType(), span), span))
    return ir.Function(
        "independent_vec_stages",
        [gm],
        [],
        ir.SeqStmts([*definitions, *payloads, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("stages", [1, 2, 4, 8, 16, 32, 64, 128])
def test_static_stage_membership_work_is_indexed(stages):
    function = independent_vec_stages(stages)
    analysis = testing.analyze_placed_sync(function, True)
    description = analysis.describe()
    assert description["class_tests"] == 4 * stages
    # Membership work is linear; index comparisons are charged separately.
    assert description["class_lookup_tests"] <= 8 * stages * (2 * stages).bit_length()
    assert description["cells"] == 2 * stages + 1
    assert len(description["regions"][0]["demands"]) == stages
    if stages <= 8:
        validate(function, analysis)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
