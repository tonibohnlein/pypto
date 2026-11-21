# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Retain A3 programmer-authored GM protocol at its native cuts."""

import pytest
from pypto import codegen, ir, passes
from pypto import language as pl
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import testing

from .sync_closed_command_oracle import ClosedCommands, opname

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _program(read_index="position", workspace_use="pl.system.set_ffts(workspace)"):
    return pl.parse_program(f"""
columns = pl.dynamic("columns")

@pl.program
class AuthoredGM:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, workspace: pl.Tensor[[256], pl.INT64],
               table: pl.Tensor[[4, columns], pl.INT32], src: pl.Tensor[[1, 8], pl.INT32],
               out: pl.Out[pl.Tensor[[1, 8], pl.INT32]],
               position: pl.Scalar[pl.INDEX]) -> pl.Tensor[[1, 8], pl.INT32]:
        {workspace_use}
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        value = pl.load(src, [0, 0], [1, 8], target_memory=pl.Mem.Vec)
        place = pl.tensor.read(table, [1, {read_index}])
        pl.tile.write(value, [0, 0], place)
        result = pl.store(value, [0, 0], out)
        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
        return result
""")


def _lower(program, *, autosync=False):
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=autosync,
        autosync_policy="conservative_prefix",
    ):
        return PassManager(ir.OptimizationStrategy.Default).run_passes(program)


def test_authored_gm_protocol_and_dynamic_point_keep_native_order():
    lowered = _lower(_program(), autosync=True)
    leaf = lowered.get_function("kernel")
    assert leaf is not None
    assert leaf.ir_stage == ir.FunctionIRStage.Buffer
    # A raw A3 flag has no local set/wait contract or proof of a matching peer.
    assert not leaf.attrs["pypto_autosync"]
    native = codegen.PTOCodegen().generate(lowered)
    cursor = 0
    for step in (
        "pto.set_ffts",
        "pto.cmo.cacheinvalid",
        "pto.fence.barrier_all",
        "pto.syncall",
        "pto.cmo.cacheinvalid",
        "pto.sync.wait",
        "pto.tload",
        "pto.load ",
        "pto.tsetval",
        "pto.tstore",
        "pto.sync.set",
    ):
        cursor = native.index(step, cursor) + len(step)
    assert native.count("pto.set_ffts") == 1
    assert native.count("pto.sync.set") == 1
    assert native.count("pto.sync.wait") == 1


def test_i64_tensor_type_exception_does_not_admit_gm_scalar_payload():
    lowered = _lower(_program(workspace_use="pl.system.set_ffts(workspace)\n        pl.read(workspace, [0])"))
    leaf = lowered.get_function("kernel")
    assert leaf is not None
    assert leaf.ir_stage == ir.FunctionIRStage.Functional
    assert leaf.attrs["buffer_admission_property"] in {"type_layout", "operation_recipe"}


def _mixed_program(
    *,
    writer="fresh",
    publish_table=False,
    acquire="pl.system.cacheinvalid()",
    rendezvous="pl.system.syncall(core_type=pl.KernelType.MIX)",
    prefetch=False,
    consume_latest=False,
):
    vec_actual = {"fresh": "vec_scratch", "different_formal": "vec_out", "same_formal": "table"}[writer]
    publication = (
        "early = pl.load(data, [0, 0], [1, 8], target_memory=pl.Mem.Vec)\n"
        "        pl.store(early, [0, 0], table)"
        if publish_table
        else "pass"
    )
    lookahead = (
        "next_lhs = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)\n"
        "        next_rhs = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)"
        if prefetch
        else "pass"
    )
    next_rhs = "rhs" if consume_latest else "next_rhs"
    later_work = (
        "next_a = pl.move(next_lhs, target_memory=pl.Mem.Left)\n"
        f"        next_b = pl.move({next_rhs}, target_memory=pl.Mem.Right)\n"
        "        next_acc = pl.matmul(next_a, next_b)\n"
        "        result = pl.store(next_acc, [0, 0], result)"
        if prefetch
        else "pass"
    )
    first_rhs = "next_rhs" if consume_latest else "rhs"
    return pl.parse_program(f"""
@pl.program
class AuthoredMixedGM:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, workspace: pl.Tensor[[256], pl.INT64], x: pl.Tensor[[16, 16], pl.BF16],
             out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        pl.system.set_ffts(workspace)
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        lhs = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        rhs = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        {lookahead}
        a = pl.move(lhs, target_memory=pl.Mem.Left)
        b = pl.move({first_rhs}, target_memory=pl.Mem.Right)
        acc = pl.matmul(a, b)
        result = pl.store(acc, [0, 0], out)
        {later_work}
        pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
        return result

    @pl.function(type=pl.FunctionType.AIV)
    def vec(self, workspace: pl.Tensor[[256], pl.INT64], table: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            data: pl.Tensor[[1, 8], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 8], pl.INT32]]) -> pl.Tensor[[1, 8], pl.INT32]:
        pl.system.set_ffts(workspace)
        {publication}
        pl.system.cacheinvalid()
        pl.system.fence()
        {rendezvous}
        {acquire}
        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        item = pl.tensor.read(table, [0, 0])
        value = pl.load(data, [0, 0], [1, 8], target_memory=pl.Mem.Vec)
        pl.tile.write(value, [0, 0], item)
        return pl.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Group)
    def unit(self, workspace: pl.Tensor[[256], pl.INT64], x: pl.Tensor[[16, 16], pl.BF16],
             table: pl.InOut[pl.Tensor[[1, 8], pl.INT32]], data: pl.Tensor[[1, 8], pl.INT32],
             cube_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]], vec_out: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
        self.cube(workspace, x, cube_out)
        self.vec(workspace, table, data, vec_out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16, 16], pl.BF16], table: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
             data: pl.Tensor[[1, 8], pl.INT32], vec_out: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
        workspace = pl.create_tensor([256], dtype=pl.INT64)
        cube_scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
        vec_scratch = pl.create_tensor([1, 8], dtype=pl.INT32)
        self.unit(workspace, x, table, data, cube_scratch, {vec_actual})
""")


@pytest.mark.parametrize(
    "options,reason",
    [
        ({}, None),
        ({"publish_table": True}, None),
        ({"writer": "same_formal"}, "aliases a GM write"),
        ({"writer": "different_formal"}, "may alias a GM write"),
        ({"acquire": "pass"}, "publication seam"),
        ({"rendezvous": "pass"}, "publication seam"),
    ],
)
def test_authored_gm_pair_requires_original_publication_and_readonly_storage(options, reason):
    lowered = _lower(_mixed_program(**options), autosync=True)
    leaves: list[ir.Function] = []
    for name in ("cube", "vec"):
        leaf = lowered.get_function(name)
        assert leaf is not None, name
        leaves.append(leaf)
        assert leaf.ir_stage == ir.FunctionIRStage.Buffer, leaf.attrs
        assert leaf.attrs["pypto_autosync"] == (reason is None), leaf.attrs
        if reason is not None:
            assert reason in leaf.attrs["autosync_fallback_reason"]
    if reason is None:
        native = codegen.PTOCodegen().generate(ir.Program(leaves, "unit", ir.Span.unknown()))
        assert native.count("pto.set_ffts") == 2
        assert native.count("pto.fence.barrier_all") == 2
        assert native.count("pto.syncall") == 2
        assert native.count("pto.cmo.cacheinvalid") == 4
        assert native.count("pto.sync.set") == 1
        assert native.count("pto.sync.wait") == 1


class _AuthoredCommands(ClosedCommands):
    """Replay local storage and event lifetimes while recording authored cuts."""

    def __init__(self, function, *, scalar_memory=None):
        super().__init__(function, {}, range(6), scalar_memory=scalar_memory)
        self.authored = []

    def assign(self, statement, env):
        call = statement.value
        if isinstance(call, ir.Call) and call.op.name == opname("buffer.gm_read"):
            # Both scalar reads use the same independent byte map and S
            # issue/completion model for this in-bounds rank-two coordinate.
            call = ir.Call(ir.get_op("buffer.read"), call.args, call.type, call.span)
            statement = ir.AssignStmt(statement.var, call, statement.span)
        super().assign(statement, env)

    def evaluate(self, statement, env):
        call = statement.expr
        if call.op.name in {
            opname("system.set_ffts"),
            opname("system.cacheinvalid"),
            opname("system.fence"),
            opname("system.syncall"),
            opname("system.sync_set"),
            opname("system.sync_wait"),
        }:
            self.authored.append((len(self.payloads), call.op.name, call.kwargs))
            return
        super().evaluate(statement, env)


@pytest.mark.parametrize("consume_latest", [False, True])
def test_authored_gm_prefix_capture_skips_only_independent_prefetches(consume_latest):
    program = _mixed_program(prefetch=True, consume_latest=consume_latest)
    baseline = _AuthoredCommands(_lower(program).get_function("cube"))
    actual_leaf = _lower(program, autosync=True).get_function("cube")
    assert actual_leaf is not None
    assert actual_leaf.attrs["pypto_autosync"], actual_leaf.attrs
    actual = _AuthoredCommands(actual_leaf)
    baseline.run()
    order = actual.run()
    assert baseline.authored == actual.authored
    assert baseline.payloads == actual.payloads
    # Both lookahead loads are issued before the first multiply. Its first
    # pair of moves needs the latest load only in the dependence control.
    cube = ("payload", 6, "I")
    assert actual.payloads[6][1] == ir.PipeType.M
    assert (("payload", 0, "C"), cube) in order
    for prefetch in (2, 3):
        assert ((("payload", prefetch, "C"), cube) in order) == consume_latest
    # A later consumer still needs both lookahead prefixes. Check all
    # overlapping physical bytes independently of the compiler's domains.
    for index, (_, _, reads, writes) in enumerate(actual.payloads):
        for earlier, (_, pipe, old_reads, old_writes) in enumerate(actual.payloads[:index]):
            if pipe == ir.PipeType.M and actual.payloads[index][1] == ir.PipeType.M:
                continue
            if old_writes & (reads | writes) or old_reads & writes:
                assert (("payload", earlier, "C"), ("payload", index, "I")) in order


def _guarded_vec_prefix_program(alias_tail):
    program = _lower(_mixed_program())
    vec = program.get_function("vec")
    assert vec is not None and isinstance(vec.body, ir.SeqStmts)
    span = vec.span
    statements = vec.body.stmts
    load_index, load = next(
        (i, statement)
        for i, statement in enumerate(statements)
        if isinstance(statement, ir.EvalStmt)
        and isinstance(statement.expr, ir.Call)
        and statement.expr.op.name == opname("buffer.load")
    )
    read_index, read = next(
        (i, statement)
        for i, statement in enumerate(statements)
        if isinstance(statement, ir.AssignStmt)
        and isinstance(statement.value, ir.Call)
        and statement.value.op.name == opname("buffer.gm_read")
    )
    assert isinstance(load.expr, ir.Call)
    carrier = load.expr.args[3]
    scratch = ir.Var("scratch", carrier.type, span)
    index_type = ir.DataType.INDEX

    def ci(value):
        return ir.ConstInt(value, index_type, span)

    def fill(buffer):
        call = ir.Call(
            ir.get_op("buffer.full"), [ir.ConstInt(0, ir.DataType.INT32, span), buffer], ir.VoidType(), span
        )
        return ir.EvalStmt(call, span)

    carrier_alloc = next(
        statement
        for statement in statements[:load_index]
        if isinstance(statement, ir.AssignStmt) and statement.var.same_as(carrier)
    )
    assert isinstance(carrier_alloc.value, ir.Call)
    address = carrier_alloc.value.args[1] if alias_tail else ci(256)
    alloc_call = ir.Call(ir.get_op("buffer.alloc"), [ir.MakeTuple([], span), address], carrier.type, span)
    scratch_alloc = ir.AssignStmt(scratch, alloc_call, span)
    visits = ir.Var("visits", ir.ScalarType(index_type), span)
    define_visits = ir.AssignStmt(visits, ir.Cast(read.var, index_type, span), span)
    induction = ir.Var("visit", ir.ScalarType(index_type), span)
    step = ir.IfStmt(
        ir.Lt(induction, visits, ir.DataType.BOOL, span),
        ir.SeqStmts([load, fill(carrier)], span),
        None,
        [],
        span,
    )
    loop = ir.ForStmt(
        induction,
        ci(0),
        ir.Add(visits, ci(1), index_type, span),
        ci(1),
        [],
        step,
        [],
        span,
    )
    store = next(
        statement
        for statement in statements[load_index:]
        if isinstance(statement, ir.EvalStmt)
        and isinstance(statement.expr, ir.Call)
        and statement.expr.op.name == opname("buffer.store")
    )
    patched = ir.Function(
        vec.name,
        list(zip(vec.params, vec.param_directions, strict=True)),
        vec.return_types,
        ir.SeqStmts(
            [
                *statements[:read_index],
                scratch_alloc,
                read,
                define_visits,
                fill(carrier),
                fill(scratch),
                loop,
                store,
                statements[-1],
            ],
            span,
        ),
        span,
        vec.func_type,
        vec.level,
        vec.role,
        vec.attrs,
        vec.requires_runtime_binding,
        vec.ir_stage,
    )
    return ir.Program(
        [patched if f.name == vec.name else f for f in program.functions.values()], program.name, program.span
    )


@pytest.mark.parametrize("alias_tail", [False, True])
def test_authored_gm_first_loop_wait_keeps_only_the_needed_prefix(alias_tail):
    program = _guarded_vec_prefix_program(alias_tail)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        actual_leaf = passes.realize_sync()(program).get_function("vec")
    assert actual_leaf is not None
    assert actual_leaf.attrs["pypto_autosync"], actual_leaf.attrs
    table = next(i for i, var in enumerate(actual_leaf.params) if var.name_hint.startswith("table"))
    for trips in (-1, 0, 1, 3):
        memory = {("gm", table * 4096 + byte): trips for byte in range(4)}
        before = _AuthoredCommands(program.get_function("vec"), scalar_memory=memory)
        actual = _AuthoredCommands(actual_leaf, scalar_memory=memory)
        before.run()
        order = actual.run()
        assert before.authored == actual.authored
        assert before.payloads == actual.payloads
        loads = [i for i, (_, pipe, _, _) in enumerate(actual.payloads) if pipe == ir.PipeType.MTE2]
        assert len(loads) == max(0, trips)
        if loads:
            assert (("payload", 1, "C"), ("payload", loads[0], "I")) in order
            assert ((("payload", 2, "C"), ("payload", loads[0], "I")) in order) == alias_tail
            for previous, current in zip(loads, loads[1:], strict=False):
                assert (("payload", previous + 1, "C"), ("payload", current, "I")) in order
        else:
            assert ("acquire", ir.PipeType.V, ir.PipeType.MTE2) not in actual.trace


@pytest.mark.parametrize("requested_extent,upper", [("pl.min(length, 16)", 16), ("signed_extent", 2**32 - 1)])
def test_host_tensor_dim_uses_only_representable_requested_slice_bounds(requested_extent, upper):
    program = pl.parse_program(f"""
length = pl.dynamic("length")

@pl.program
class HostViewExtent:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, extent: pl.Scalar[pl.INDEX]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, source: pl.Tensor[[length, 16], pl.FP32], signed_extent: pl.Scalar[pl.INDEX]):
        view = pl.tensor.slice(source, [{requested_extent}, 16], [0, 0])
        observed = pl.tensor.dim(view, 0)
        self.kernel(observed)
""")
    assert testing.sync_invocation_evidence(program)["parameters"]["kernel"][0] == (0, upper)
