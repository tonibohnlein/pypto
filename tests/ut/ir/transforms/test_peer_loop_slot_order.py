# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay original FIFO ordinals against the single-core command graph."""

import json

import pytest
from pypto import ir, passes
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager
from pypto.runtime import RunConfig

from tests.ut.codegen.test_pto_codegen_cross_core import (
    _mat_receive_epilogue,
    _paired_consumer,
    _SharePopExpression,
)

from .sync_closed_command_oracle import ClosedCommands, opname
from .test_finite_sync_analysis import closure


class MatSlotCommands(ClosedCommands):
    """The native receive cursor advances once per executed pop, including loop visits."""

    def __init__(self, fn, depth, *, drop_releases=False):
        super().__init__(fn, {}, range(8))
        self.env.update({param: 0 for param in fn.params})
        self.depth = depth
        self.drop_releases = drop_releases
        self.pops = []
        self.readers = []
        self.order_checks = []

    def assign(self, statement, env):
        call = statement.value
        if not isinstance(call, ir.Call):
            return super().assign(statement, env)
        name = call.op.name
        if name in (
            ir.get_op("tile.alloc").name,
            ir.get_op("system.reserve_buffer").name,
            ir.get_op("system.import_peer_buffer").name,
        ):
            env[statement.var] = None
            return
        assert self.last_all is None, "unexpected drain before a payload"
        pipe = {
            ir.get_op("tile.tpop_from_aiv").name: ir.PipeType.MTE2,
            ir.get_op("tile.move").name: ir.PipeType.MTE1,
            ir.get_op("tile.matmul").name: ir.PipeType.M,
            ir.get_op("tile.store").name: ir.PipeType.FIX,
            ir.get_op("tile.load").name: ir.PipeType.MTE2,
        }[name]
        issue = ("issue", len(self.calls))
        done = ("done", len(self.calls))
        self.calls.append(call)
        self.issue(pipe, issue)
        self.vertices.add(done)
        self.edge(issue, done)
        self.edge(self.last_payload.get(pipe), done)
        self.last_payload[pipe] = done
        if name == ir.get_op("tile.tpop_from_aiv").name:
            visit = len(self.pops)
            if visit >= self.depth:
                self.order_checks.append((self.readers[visit - self.depth], issue))
            self.pops.append(done)
            self.readers.append(None)
            env[statement.var] = visit
        elif name == ir.get_op("tile.move").name and call.args[0] in env and env[call.args[0]] is not None:
            visit = env[call.args[0]]
            self.order_checks.append((self.pops[visit], issue))
            self.readers[visit] = done
            env[statement.var] = None
        else:
            env[statement.var] = None

    def evaluate(self, statement, env):
        call = statement.expr
        if call.op.name in (
            ir.get_op("system.tfree_to_aiv").name,
            ir.get_op("system.aic_initialize_pipe").name,
        ):
            return
        if self.drop_releases and call.op.name in (
            ir.get_op("system.set_flag").name,
            ir.get_op("system.wait_flag").name,
        ):
            if (ir.PipeType(call.kwargs["src_pipe"]), ir.PipeType(call.kwargs["dst_pipe"])) == (
                ir.PipeType.MTE1,
                ir.PipeType.MTE2,
            ):
                return
        return super().evaluate(statement, env)


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    ("trips", "per_trip", "two_readers"),
    [(3, 2, False), (3, 1, False), (1, 1, False), (0, 2, False), (3, 1, True)],
)
def test_last_reader_reaches_epilogue(tmp_path, trips, per_trip, two_readers):
    snapshots = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            snapshots.append(program.get_function("cube"))

    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [passes.CallbackInstrument(after_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        program = _mat_receive_epilogue(
            16 if two_readers else 128, "BF16", 2, loop=(trips, per_trip), two_readers=two_readers
        )
        if per_trip == 2:
            program = _SharePopExpression().visit_program(program)
        ir.compile(program, **cfg.compile_kwargs())
    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert manifest["owned"] and not manifest["ptoas_insert_sync"], manifest
    state = MatSlotCommands(snapshots[0], 2)
    order = state.run()
    count = trips * per_trip + 2
    assert len(state.readers) == len(state.pops) == count
    assert all(reader is not None for reader in state.readers)
    assert len(state.order_checks) == count * (1 + two_readers) + max(0, count - 2)
    assert all(edge in order for edge in state.order_checks)
    if trips == 3 and per_trip == 2:
        mutant = MatSlotCommands(snapshots[0], 2, drop_releases=True)
        # Dropping the reverse channel may also invalidate causal event-ID reuse.
        mutant.execute(mutant.function.body, dict(mutant.env))
        broken = closure(mutant.vertices, mutant.edges)
        for reader, pop in zip(mutant.readers[-4:-2], mutant.pops[-2:], strict=True):
            assert (reader, ("issue", pop[1])) not in broken


class PeerSenderCommands(ClosedCommands):
    # This fixture models each original push as a native command, including
    # the zero-row lane's handshake, and tracks its completion at the exit.
    push_pipes = {
        opname("tile.tpush_to_aiv"): ir.PipeType.FIX,
        opname("tile.tpush_to_aic"): ir.PipeType.MTE3,
    }
    payload_pipes = {
        **push_pipes,
        opname("tile.load"): ir.PipeType.MTE2,
        opname("tile.move"): ir.PipeType.MTE1,
        opname("tile.matmul"): ir.PipeType.M,
    }

    def __init__(self, fn, subblock):
        super().__init__(fn, {}, range(8))
        self.subblock = subblock

    def assign(self, statement, env):
        call = statement.value
        if isinstance(call, ir.Call):
            name = call.op.name
            if name == opname("tile.get_subblock_idx"):
                env[statement.var] = self.subblock
                return
            if name in (opname("tile.alloc"), opname("tile.create"), opname("system.import_peer_buffer")):
                return
            if name in self.payload_pipes:
                self.payload(call, env)
                return
        super().assign(statement, env)

    def evaluate(self, statement, env):
        if statement.expr.op.name in (
            opname("system.aic_initialize_pipe"),
            opname("system.aiv_initialize_pipe"),
        ):
            return
        super().evaluate(statement, env)

    def payload(self, call, env):
        assert self.last_all is None, "unexpected drain before a payload or handshake"
        pipe = self.payload_pipes[call.op.name]
        issue = ("issue", len(self.calls))
        done = ("done", len(self.calls))
        self.calls.append(call)
        self.issue(pipe, issue)
        self.vertices.add(done)
        self.edge(issue, done)
        self.edge(self.last_payload.get(pipe), done)
        self.last_payload[pipe] = done


def _rewrite_void_exits(fn, implicit_arms):
    def rewrite(body, subblock):
        assert isinstance(body, ir.SeqStmts)
        last = body.stmts[-1]
        assert isinstance(last, ir.ReturnStmt) and not last.value
        statements = body.stmts[:-1]
        if subblock not in implicit_arms:
            # Leave the explicit return under transparent sequence wrappers.
            # Keep it nested even if materialization flattens two levels.
            for _ in range(3):
                last = ir.SeqStmts([last], body.span)
            statements.append(last)
        return ir.SeqStmts(statements, body.span)

    assert not fn.return_types
    body = fn.body
    assert isinstance(body, ir.SeqStmts)
    terminal = body.stmts[-1]
    if isinstance(terminal, ir.IfStmt):
        assert terminal.else_body is not None
        branch = ir.IfStmt(
            terminal.condition,
            rewrite(terminal.then_body, 0),
            rewrite(terminal.else_body, 1),
            terminal.return_vars,
            terminal.span,
        )
        body = ir.SeqStmts([*body.stmts[:-1], branch], body.span)
    else:
        body = rewrite(body, None)
    return ir.Function(
        fn.name,
        list(zip(fn.params, fn.param_directions, strict=True)),
        fn.return_types,
        body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
        fn.ir_stage,
    )


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("endpoint", ["cube", "vector"])
def test_peer_void_fallthrough_matches_explicit_completion(endpoint):
    checkpoints = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            checkpoints.append(program)

    if endpoint == "cube":
        source = _paired_consumer(2, read_after_wrap=False)
        variants, subblocks, pushes = [(None,), ()], (None,), 2
    else:
        source = _mat_receive_epilogue(128, "BF16", 2)
        variants, subblocks, pushes = [(0,), (1,), (0, 1)], (0, 1), 4

    with passes.PassContext(
        [passes.CallbackInstrument(before_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        explicit = PassManager(ir.OptimizationStrategy.Default).run_passes(source)
        (placed,) = checkpoints
        original = placed[endpoint]
        explicit_endpoint = explicit[endpoint]
        assert explicit_endpoint is not None and explicit_endpoint.attrs[PYPTO_AUTOSYNC_ATTR]

        # Tensor-to-tile normally synthesizes an empty return for void source
        # functions. Rewrite them at the input to AutoSync so both
        # fallthrough and transparent nested returns reach the planner.
        for implicit_arms in variants:
            edited = _rewrite_void_exits(original, implicit_arms)
            fallthrough = ir.Program(
                [edited if fn.name == endpoint else fn for fn in placed.functions.values()],
                placed.name,
                placed.span,
            )
            implicit = passes.realize_sync()(fallthrough)
            implicit_endpoint = implicit[endpoint]
            assert implicit_endpoint is not None and implicit_endpoint.attrs[PYPTO_AUTOSYNC_ATTR]
            for subblock in subblocks:
                expected = PeerSenderCommands(explicit_endpoint, subblock)
                actual = PeerSenderCommands(implicit_endpoint, subblock)
                expected_order, actual_order = expected.run(), actual.run()
                assert expected.calls == actual.calls
                assert sum(call.op.name in actual.push_pipes for call in actual.calls) == pushes
                assert len(expected.returns) == 1
                assert len(actual.returns) == (subblock not in implicit_arms)
                assert sum(node[0] == "all_barrier" for node in actual.vertices) == 1
                assert all((("done", i), actual.last_all) in actual_order for i in range(len(actual.calls)))
                # Compare the command graph through completion. Only the
                # explicit path has an additional source-level return node.
                assert {edge for edge in actual_order if edge[1][0] != "return"} == {
                    edge for edge in expected_order if edge[1][0] != "return"
                }
