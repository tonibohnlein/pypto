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
from collections import defaultdict

import pytest
from pypto import ir, passes
from pypto._function_attrs import (
    AUTOSYNC_ANALYSIS_POLICY_ATTR,
    AUTOSYNC_WHOLE_MODULE_ATTR,
    PYPTO_AUTOSYNC_ATTR,
)
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager
from pypto.runtime import RunConfig

from tests.ut.codegen.test_pto_codegen_cross_core import (
    _mat_receive_epilogue,
    _paired_consumer,
    _paired_receive_program,
    _PeerWorkLoop,
    _SharePopExpression,
)

from .sync_closed_command_oracle import ClosedCommands, opname
from .test_finite_sync_analysis import closure


class MatSlotCommands(ClosedCommands):
    """The native receive cursor advances once per executed pop, including loop visits."""

    def __init__(self, fn, depth, *, drop_releases=False, core=0, workers=1):
        super().__init__(fn, {}, range(8))
        self.env.update({param: 0 for param in fn.params})
        self.coordinates = {
            opname(f"{stage}.{op}"): result
            for stage in ("tile", "buffer")
            for op, result in (("get_block_idx", core), ("get_block_num", workers))
        }
        self.depth = depth
        self.drop_releases = drop_releases
        self.pops = []
        self.readers = []
        self.all_readers = []
        self.order_checks = []
        self.local_accesses = defaultdict(list)
        self.local_checks = []
        self.gm_writers = {}
        self.gm_checks = []
        self.buffer_accesses = []
        self.rings = {}
        self.borrow_results = set()
        self.rendezvous = []

    def issue_payload(self, call, pipe):
        issue = ("issue", len(self.calls))
        done = ("done", len(self.calls))
        self.calls.append(call)
        self.issue(pipe, issue)
        self.vertices.add(done)
        self.edge(issue, done)
        self.edge(self.last_payload.get(pipe), done)
        self.last_payload[pipe] = done
        return issue, done

    def record_pop(self, result, issue, done, env):
        visit = len(self.pops)
        if visit >= self.depth:
            self.order_checks.extend((reader, issue) for reader in self.all_readers[visit - self.depth])
        self.pops.append(done)
        self.readers.append(None)
        self.all_readers.append([])
        env[result] = visit
        self.borrow_results.add(result)

    def record_read(self, visit, issue, done):
        self.order_checks.append((self.pops[visit], issue))
        self.readers[visit] = done
        self.all_readers[visit].append(done)

    def assign(self, statement, env):
        call = statement.value
        if not isinstance(call, ir.Call):
            return super().assign(statement, env)
        name = call.op.name
        if name in self.coordinates:
            env[statement.var] = self.coordinates[name]
            return
        if name == opname("buffer.peer_reserve"):
            env[statement.var] = (call.kwargs["base"], call.kwargs["size"])
            return
        if name == opname("buffer.peer_reserve_addr"):
            env[statement.var] = env[call.args[0]]
            return
        if name == opname("buffer.peer_pop"):
            base, capacity = env[call.args[0]]
            stride, local_depth = self.rings[call.kwargs.get("id", 0)]
            slot = len(self.pops) % local_depth
            kind = statement.var.type
            byte_count = kind.shape[0] * kind.shape[1] * kind.dtype.get_byte()
            assert local_depth == self.depth and byte_count <= stride and local_depth * stride <= capacity
            self.storage[statement.var] = (
                kind.memory_space,
                base + slot * stride,
                *kind.shape,
                kind.dtype.get_byte(),
            )
            self.layouts[statement.var] = kind
            issue, done = self.issue_payload(call, ir.PipeType.MTE2)
            self.record_pop(statement.var, issue, done, env)
            return
        if name == opname("buffer.alloc"):
            return super().assign(statement, env)
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
        issue, done = self.issue_payload(call, pipe)
        # All Left, Right and Acc accesses in this fixture use complete tiles.
        # Compare physical addresses even if different SSA names reuse a tile.
        for operand, write in [(arg, False) for arg in call.args] + [(statement.var, True)]:
            kind = operand.type
            if not isinstance(kind, ir.TileType) or kind.memory_space not in (
                ir.MemorySpace.Left,
                ir.MemorySpace.Right,
                ir.MemorySpace.Acc,
            ):
                continue
            assert kind.memref is not None and isinstance(kind.memref.byte_offset_, ir.ConstInt)
            accesses = self.local_accesses[kind.memory_space, kind.memref.byte_offset_.value]
            self.local_checks.extend(
                (older_done, issue)
                for older_pipe, older_write, older_done in accesses
                if pipe != older_pipe and (write or older_write)
            )
            accesses.append((pipe, write, done))
        if name == opname("tile.store"):
            # All stores write a 16-by-16 window of the same output. Distinct
            # row offsets in this fixture are multiples of 16.
            indices = call.args[1]
            assert isinstance(indices, ir.MakeTuple)
            offsets = tuple(index.value for index in indices.elements if isinstance(index, ir.ConstInt))
            assert len(offsets) == 2
            if offsets in self.gm_writers:
                self.gm_checks.append((self.gm_writers[offsets], issue))
            self.gm_writers[offsets] = done
        if name == ir.get_op("tile.tpop_from_aiv").name:
            self.record_pop(statement.var, issue, done, env)
        elif name == ir.get_op("tile.move").name and call.args[0] in env and env[call.args[0]] is not None:
            self.record_read(env[call.args[0]], issue, done)
            env[statement.var] = None
        else:
            env[statement.var] = None

    def evaluate(self, statement, env):
        call = statement.expr
        if call.op.name == opname("system.syncall"):
            # The original hard MIX call drains the local queues but leaves
            # the receive cursor and flag inventory untouched.
            assert call.kwargs.get("mode", "hard") == "hard"
            assert call.kwargs.get("core_type", "mix") == "mix"
            assert not self.live
            self.complete_all()
            self.last_all = None
            self.rendezvous.append(len(self.pops))
            return
        if call.op.name == opname("system.aic_initialize_pipe"):
            depth = call.kwargs["slot_num"]
            self.rings[call.kwargs.get("id", 0)] = (
                call.kwargs["slot_size"],
                call.kwargs.get("local_slot_num", depth),
            )
            return
        if call.op.name in (
            ir.get_op("system.tfree_to_aiv").name,
            opname("buffer.peer_free"),
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

    def payload(self, call, env):
        pipe, reads, writes, _, _ = self.memory_footprint(call, env) or self.compute_footprint(call, env)
        issue, done = self.issue_payload(call, pipe)
        for argument in set(call.args):
            if argument in self.borrow_results:
                self.record_read(env[argument], issue, done)
        for previous_pipe, previous_reads, previous_writes, previous_done in self.buffer_accesses:
            cells = (previous_writes & (reads | writes)) | (previous_reads & writes)
            if any(space == "gm" for space, _ in cells):
                self.gm_checks.append((previous_done, issue))
            if previous_pipe != pipe and any(space != "gm" for space, _ in cells):
                self.local_checks.append((previous_done, issue))
        self.buffer_accesses.append((pipe, reads, writes, done))


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
    assert len(state.order_checks) == (count + max(0, count - 2)) * (1 + two_readers)
    assert all(edge in order for edge in state.order_checks)
    if trips == 3 and per_trip == 2:
        mutant = MatSlotCommands(snapshots[0], 2, drop_releases=True)
        # Dropping the reverse channel may also invalidate causal event-ID reuse.
        mutant.execute(mutant.function.body, dict(mutant.env))
        broken = closure(mutant.vertices, mutant.edges)
        for reader, pop in zip(mutant.readers[-4:-2], mutant.pops[-2:], strict=True):
            assert (reader, ("issue", pop[1])) not in broken


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("inner_trips", [1, 2])
def test_runtime_work_loop_composes_slot_releases(tmp_path, inner_trips):
    snapshots = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            snapshots.append(program)

    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [passes.CallbackInstrument(after_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        # Three pops per item rotate the entry slot. Four return to the same slot.
        program = _mat_receive_epilogue(128, "BF16", 2, loop=(inner_trips, 1), work=_PeerWorkLoop(5))
        ir.compile(program, **cfg.compile_kwargs())
    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    cube = snapshots[0].get_function("cube")
    vector = snapshots[0].get_function("vector")
    assert manifest["owned"] and not manifest["ptoas_insert_sync"], (cube.attrs, vector.attrs)
    for workers in (1, 2, 5, 6):
        for core in range(workers):
            state = MatSlotCommands(cube, 2, core=core, workers=workers)
            order = state.run()
            pops = 2 + len(range(core, 5, workers)) * (inner_trips + 2) + 1
            assert len(state.pops) == len(state.readers) == pops
            assert len(state.order_checks) == pops + max(0, pops - 2)
            assert all(edge in order for edge in state.order_checks)
            assert state.local_checks and all(edge in order for edge in state.local_checks)
            assert all(edge in order for edge in state.gm_checks)
            if workers == 1:
                assert state.gm_checks
            for subblock in (0, 1):
                sender = PeerSenderCommands(vector, subblock, core=core, workers=workers)
                sender.run()
                assert sum(call.op.name in sender.push_pipes for call in sender.calls) == pops

    # Delete the entry handoff and take the zero-trip path. The suffix now
    # overwrites the first prefix slot without waiting for its reader.
    stmts = cube.body.stmts
    cut = next(i for i, stmt in enumerate(stmts) if isinstance(stmt, ir.ForStmt))
    start = cut
    while start and isinstance(stmts[start - 1], ir.EvalStmt):
        if stmts[start - 1].expr.span.begin_line != stmts[cut].span.begin_line:
            break
        start -= 1
    assert start < cut
    mutant = MatSlotCommands(cube, 2, core=5, workers=6)
    mutant.execute(ir.SeqStmts([*stmts[:start], *stmts[cut:]], cube.body.span), dict(mutant.env))
    broken = closure(mutant.vertices, mutant.edges)
    assert (mutant.readers[0], ("issue", mutant.pops[-1][1])) not in broken


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
def test_guarded_work_preserves_readers_through_join(tmp_path):
    snapshots = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            snapshots.append(program)

    source = _mat_receive_epilogue(128, "BF16", 2, loop=(1, 1), work=_PeerWorkLoop(5, guarded_until=3))
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [passes.CallbackInstrument(after_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        ir.compile(source, **cfg.compile_kwargs())

    cube = snapshots[0].get_function("cube")
    vector = snapshots[0].get_function("vector")
    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert manifest["owned"] and not manifest["ptoas_insert_sync"], (cube.attrs, vector.attrs)
    for core, workers in [(5, 6), (4, 6), (2, 2), (0, 2), (0, 1)]:
        # These paths include a zero-trip loop, a skipped branch, and odd/even
        # guard counts. The unguarded epilogue writes the actual rotated slot.
        state = MatSlotCommands(cube, 2, core=core, workers=workers)
        order = state.run()
        pops = 3 + 3 * len(range(core, 3, workers))
        assert len(state.pops) == pops
        assert all(state.readers)
        assert len(state.order_checks) == 2 * pops - 2
        assert all(edge in order for edge in state.order_checks + state.local_checks + state.gm_checks)
        for subblock in (0, 1):
            sender = PeerSenderCommands(vector, subblock, core=core, workers=workers)
            sender.run()
            assert sum(call.op.name in sender.push_pipes for call in sender.calls) == pops

    # On the zero-trip path the epilogue overwrites the first receive slot.
    mutant = MatSlotCommands(cube, 2, core=5, workers=6, drop_releases=True)
    mutant.execute(cube.body, dict(mutant.env))
    broken = closure(mutant.vertices, mutant.edges)
    assert (mutant.readers[0], ("issue", mutant.pops[-1][1])) not in broken


@pytest.mark.usefixtures("ascend_backend")
def test_guarded_work_rejects_receive_borrow_crossing_a_branch():
    source = _paired_receive_program(
        2,
        """values = pl.load(x, [0, 0], [16, 16])
result = pl.matmul(values, values)
pl.tpush_to_aiv(result, split=0)""",
        """received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
core = pl.tile.get_block_idx()
if core < 1:
    result = pl.add(received, received)
    out = pl.store(result, [0, 0], out)
pl.tfree_to_aic(received)
return out""",
    )
    _check_peer_region_fallback(source, "A live FIFO handle crosses structured control")


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("guarded_until", [None, 3])
def test_hard_mix_syncall_separates_runtime_work_regions(tmp_path, guarded_until):
    snapshots = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            snapshots.append(program)

    mixed_cut = "pl.system.syncall(core_type=pl.KernelType.MIX)"
    source = _mat_receive_epilogue(
        128,
        "BF16",
        2,
        loop=(1, 1),
        work=_PeerWorkLoop(5, guarded_until=guarded_until),
        region_boundaries=(mixed_cut,) * 2,
    )
    cfg = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [passes.CallbackInstrument(after_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        ir.compile(source, **cfg.compile_kwargs())

    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert manifest["owned"] and not manifest["ptoas_insert_sync"], manifest
    cube = snapshots[0].get_function("cube")
    for core, workers in [(5, 6), (4, 6), (0, 2)]:
        state = MatSlotCommands(cube, 2, core=core, workers=workers)
        order = state.run()
        # Replay zero, one and many visits. Odd receive counts make some
        # regions start with a rotated slot, including the zero-trip path.
        active_items = 5 if guarded_until is None else guarded_until
        per_region = 3 + len(range(core, active_items, workers)) * 3
        assert state.rendezvous == [per_region, 2 * per_region]
        assert len(state.pops) == len(state.readers) == 3 * per_region
        assert all(state.readers)
        assert state.reuses
        assert all(edge in order for edge in state.order_checks + state.local_checks + state.gm_checks)


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    ("boundary", "reason"),
    [
        ("", "at most one outer runtime work loop per region"),
        ("pl.system.syncall(core_type=pl.KernelType.AIC)", "authored hard MIX SYNCALL"),
    ],
)
def test_runtime_work_regions_require_hard_mix_syncall(boundary, reason):
    source = _mat_receive_epilogue(
        128,
        "BF16",
        2,
        loop=(1, 1),
        work=_PeerWorkLoop(5),
        region_boundaries=(boundary,),
    )
    _check_peer_region_fallback(source, reason)


@pytest.mark.usefixtures("ascend_backend")
def test_hard_mix_syncall_rejects_live_receive_borrow():
    source = _paired_receive_program(
        2,
        """values = pl.load(x, [0, 0], [16, 16])
result = pl.matmul(values, values)
pl.tpush_to_aiv(result, split=0)
pl.system.syncall(core_type=pl.KernelType.MIX)""",
        """received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
pl.system.syncall(core_type=pl.KernelType.MIX)
result = pl.add(received, received)
pl.tfree_to_aic(received)
return pl.store(result, [0, 0], out)""",
    )
    _check_peer_region_fallback(source, "A live FIFO handle crosses a hard MIX SYNCALL")


@pytest.mark.usefixtures("ascend_backend")
def test_hard_mix_region_without_receive_work_preserves_ring_cursor():
    source = _paired_receive_program(
        2,
        """values = pl.load(x, [0, 0], [16, 16])
result = pl.matmul(values, values)
core = pl.tile.get_block_idx()
workers = pl.tile.get_block_num()
for work in pl.range(core, 3, workers):
    pl.tpush_to_aiv(result, split=0)
pl.system.syncall(core_type=pl.KernelType.MIX)""",
        """core = pl.tile.get_block_idx()
workers = pl.tile.get_block_num()
for work in pl.range(core, 3, workers):
    received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
    copied = pl.add(received, received)
    pl.tfree_to_aic(received)
    out = pl.store(copied, [0, 0], out)
pl.system.syncall(core_type=pl.KernelType.MIX)
for work in pl.range(core, 3, workers):
    loaded = pl.load(out, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
    out = pl.store(pl.add(loaded, loaded), [0, 0], out)
return out""",
    )
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        result = PassManager(ir.OptimizationStrategy.Default).run_passes(source)
    for name in ("cube", "vector"):
        endpoint = result.get_function(name)
        assert endpoint is not None
        assert endpoint.attrs.get(PYPTO_AUTOSYNC_ATTR), endpoint.attrs

    # The second region reuses one Vec tile and one GM window. Its load,
    # vector add and store form a chain, so the next load needs only the store.
    vector = result.get_function("vector")
    assert vector is not None and isinstance(vector.body, ir.SeqStmts)
    branch = vector.body.stmts[-1]
    assert isinstance(branch, ir.IfStmt) and isinstance(branch.then_body, ir.SeqStmts)
    loop = [stmt for stmt in branch.then_body.stmts if isinstance(stmt, ir.ForStmt)][-1]
    assert isinstance(loop.body, ir.SeqStmts)
    calls: list[ir.Call] = []
    for stmt in loop.body.stmts:
        if isinstance(stmt, (ir.AssignStmt, ir.EvalStmt)):
            call = stmt.value if isinstance(stmt, ir.AssignStmt) else stmt.expr
            assert isinstance(call, ir.Call)
            calls.append(call)
    payload_pipes = {
        opname("tile.load"): ir.PipeType.MTE2,
        opname("tile.add"): ir.PipeType.V,
        opname("tile.store"): ir.PipeType.MTE3,
    }
    store = next(i for i, call in enumerate(calls) if call.op.name == opname("tile.store"))
    boundary = calls[store + 1 :]
    assert [call.op.name for call in boundary] == [opname("system.set_flag"), opname("system.wait_flag")]
    assert all(
        (call.kwargs["src_pipe"], call.kwargs["dst_pipe"]) == (ir.PipeType.MTE3.value, ir.PipeType.MTE2.value)
        for call in boundary
    )

    for keep_boundary in (True, False):
        state = MatSlotCommands(vector, 2)
        for _ in range(3):
            for call in calls if keep_boundary else calls[: store + 1]:
                if call.op.name in payload_pipes:
                    state.issue_payload(call, payload_pipes[call.op.name])
                else:
                    state.physical(call)
        order = closure(state.vertices, state.edges)
        assert len(state.calls) == 9 and state.reuses and not state.live
        for previous in (2, 5):
            assert ((("done", previous), ("issue", previous + 1)) in order) == keep_boundary
        assert all(edge in order for edge in state.reuses) == keep_boundary


def _check_peer_region_fallback(source, reason):
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        result = PassManager(ir.OptimizationStrategy.Default).run_passes(source)
    cube = result.get_function("cube")
    vector = result.get_function("vector")
    assert cube is not None and vector is not None
    endpoints = [cube, vector]
    assert all(not fn.attrs.get(PYPTO_AUTOSYNC_ATTR) for fn in endpoints)
    assert any(reason in fn.attrs.get("autosync_fallback_reason", "") for fn in endpoints), [
        fn.attrs for fn in endpoints
    ]


@pytest.mark.usefixtures("ascend_backend")
def test_runtime_work_loop_requires_positive_stride():
    # A core index may be zero. The same ready scalar is safe as a start but
    # cannot serve as an unconditionally positive loop step.
    source = _mat_receive_epilogue(128, "BF16", 2, loop=(1, 1), work=_PeerWorkLoop(5, stride="core"))
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        result = PassManager(ir.OptimizationStrategy.Default).run_passes(source)
    for name in ("cube", "vector"):
        endpoint = result.get_function(name)
        assert endpoint is not None
        attrs = endpoint.attrs
        assert not attrs.get(PYPTO_AUTOSYNC_ATTR)
        assert "proved positive" in attrs["autosync_fallback_reason"]


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

    def __init__(self, fn, subblock, *, core=0, workers=1):
        super().__init__(fn, {}, range(8))
        pipe = ir.PipeType.FIX if fn.func_type == ir.FunctionType.AIC else ir.PipeType.MTE3
        self.push_pipes = {**type(self).push_pipes, opname("buffer.peer_push"): pipe}
        self.payload_pipes = {
            **type(self).payload_pipes,
            **self.push_pipes,
            opname("buffer.load"): ir.PipeType.MTE2,
            opname("buffer.copy"): ir.PipeType.MTE1,
            opname("buffer.matmul"): ir.PipeType.M,
        }
        self.coordinates = {
            opname(f"{stage}.{op}"): result
            for stage in ("tile", "buffer")
            for op, result in (
                ("get_subblock_idx", subblock),
                ("get_block_idx", core),
                ("get_block_num", workers),
            )
        }

    def assign(self, statement, env):
        call = statement.value
        if isinstance(call, ir.Call):
            name = call.op.name
            if name in self.coordinates:
                env[statement.var] = self.coordinates[name]
                return
            if name in (
                opname("tile.alloc"),
                opname("tile.create"),
                opname("system.import_peer_buffer"),
                opname("buffer.alloc"),
                opname("buffer.peer_reserve"),
                opname("buffer.peer_reserve_addr"),
            ):
                return
            if name in self.payload_pipes:
                self.payload(call, env)
                return
        super().assign(statement, env)

    def evaluate(self, statement, env):
        if statement.expr.op.name in (
            opname("system.aic_initialize_pipe"),
            opname("system.aiv_initialize_pipe"),
            opname("buffer.set_validshape"),
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


class PeerMatrixCommands(ClosedCommands):
    """Interpret matrix bytes on the sending endpoint of a qualified FIFO pair."""

    def assign(self, statement, env):
        call = statement.value
        if isinstance(call, ir.Call) and call.op.name == opname("system.import_peer_buffer"):
            env[statement.var] = None
        else:
            super().assign(statement, env)

    def evaluate(self, statement, env):
        if statement.expr.op.name != opname("system.aic_initialize_pipe"):
            super().evaluate(statement, env)

    def memory_footprint(self, call, env):
        if call.op.name == opname("buffer.peer_push"):
            source, valid = call.args
            assert isinstance(valid, ir.MakeTuple)
            row, col = valid.elements
            assert isinstance(row, ir.ConstInt) and isinstance(col, ir.ConstInt)
            rows, columns = row.value, col.value
            rect = 0, 0, rows, columns
            signature = call.op.name, self.signature(source), rect
            return ir.PipeType.FIX, self.bytes(source, rect), frozenset(), signature, rect
        return super().memory_footprint(call, env)


def _with_matrix_slot_work(placed):
    cube = placed["cube"]
    span = cube.span
    index = ir.DataType.INDEX

    def ci(number):
        return ir.ConstInt(number, index, span)

    def tup(*numbers):
        return ir.MakeTuple([ci(n) for n in numbers], span)

    def call(name, args, kind=None):
        return ir.Call(ir.get_op(name), args, kind or ir.VoidType(), span)

    def define(var, name, *args):
        return ir.AssignStmt(var, call(name, list(args), var.type), span)

    def payload(name, *args):
        return ir.EvalStmt(call(name, list(args)), span)

    def matrix(shape, space):
        return ir.BufferType(
            shape,
            ir.DataType.FP16,
            space,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=512,
        )

    assert cube.ir_stage == ir.FunctionIRStage.Buffer and isinstance(cube.body, ir.SeqStmts)
    tensor = cube.params[0]
    mat = matrix([32, 16], ir.MemorySpace.Mat)
    ring = ir.Var("matrix_ring", ir.MultiBufferType(mat, 2), span)
    left = ir.Var("subtile", matrix([16, 16], ir.MemorySpace.Left), span)
    slot = ir.Var("matrix_slot", mat, span)
    wrap = ir.Var("wrap_slot", mat, span)
    iv = ir.Var("visit", ir.ScalarType(index), span)
    block = ir.Var("block", ir.ScalarType(index), span)
    loop = ir.ForStmt(
        iv,
        ci(0),
        ci(2),
        ci(1),
        [],
        ir.SeqStmts(
            [
                define(slot, "buffer.slot", ring, ir.FloorMod(iv, ci(2), index, span)),
                payload("buffer.load", tensor, tup(0, 0), tup(32, 16), slot),
                payload("buffer.extract", slot, ir.Mul(iv, ci(16), index, span), ci(0), left),
                ir.YieldStmt([], span),
            ],
            span,
        ),
        [],
        span,
    )
    *prefix, ret = cube.body.stmts
    assert isinstance(ret, ir.ReturnStmt)
    prefix = [
        *prefix,
        define(ring, "buffer.alloc_multi", tup(), ci(16384)),
        define(left, "buffer.alloc", tup(), ci(8192)),
        define(block, "buffer.get_block_idx"),
    ]
    suffix = [
        define(wrap, "buffer.slot", ring, ci(0)),
        payload("buffer.load", tensor, tup(0, 0), tup(32, 16), wrap),
        ret,
    ]

    def replace_cube(work):
        endpoint = ir.Function(
            cube.name,
            list(zip(cube.params, cube.param_directions, strict=True)),
            cube.return_types,
            ir.SeqStmts([*prefix, work, *suffix], span),
            span,
            cube.func_type,
            cube.level,
            cube.role,
            cube.attrs,
            cube.requires_runtime_binding,
            cube.ir_stage,
        )
        return ir.Program(
            [endpoint if fn.name == cube.name else fn for fn in placed.functions.values()],
            placed.name,
            placed.span,
        )

    # This authored, nonterminal conditional selects the structured peer
    # importer. Block 0 executes the exact same original loop in both versions.
    branch = ir.IfStmt(ir.Eq(block, ci(0), ir.DataType.BOOL, span), loop, None, [], span)
    return replace_cube(loop), replace_cube(branch)


@pytest.mark.usefixtures("ascend_backend")
def test_peer_finite_and_structured_routes_order_the_same_moving_matrix_bytes():
    checkpoints = []

    def capture(transform, program):
        if transform.get_name() == "RealizeSync":
            checkpoints.append(program)

    with passes.PassContext(
        [passes.CallbackInstrument(before_pass=capture)],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        PassManager(ir.OptimizationStrategy.Default).run_passes(_paired_consumer(2, read_after_wrap=False))
        (placed,) = checkpoints
        originals = _with_matrix_slot_work(placed)
        outputs = [passes.realize_sync()(program) for program in originals]

    for output in outputs:
        for name in ("cube", "vector"):
            endpoint = output[name]
            assert endpoint is not None
            assert endpoint.attrs.get(PYPTO_AUTOSYNC_ATTR), endpoint.attrs
            assert endpoint.attrs[AUTOSYNC_WHOLE_MODULE_ATTR]  # Published by the mixed peer planner.
    # Only the structured peer route runs conservative analysis and attaches
    # its policy outcome, even though both runs have the same PassContext.
    direct, structured = (output["cube"] for output in outputs)
    assert direct is not None and structured is not None
    assert AUTOSYNC_ANALYSIS_POLICY_ATTR not in direct.attrs
    assert structured.attrs[AUTOSYNC_ANALYSIS_POLICY_ATTR] == "conservative_prefix"

    common_payloads = None
    for original, output in zip(originals, outputs, strict=True):
        endpoint = output["cube"]
        before = PeerMatrixCommands(original["cube"], {}, range(8), block_index=0)
        after = PeerMatrixCommands(endpoint, {}, range(8), block_index=0)
        before.run()
        order = after.run()  # Also checks consuming flag reuse and empty inventory at exit.
        assert after.payloads == before.payloads and after.descriptors == before.descriptors
        if common_payloads is None:
            common_payloads = before.payloads
        assert after.payloads == common_payloads
        extract = [i for i, (call, *_) in enumerate(after.calls) if call.op.name == opname("buffer.extract")]
        assert len(extract) == 2
        for visit, i in enumerate(extract):
            _, pipe, reads, writes = after.payloads[i]
            # Every NZ box in this FP16 fixture contains 512 bytes. Slot 1
            # begins 1024 bytes later, and its extraction starts at box 1.
            start = 16384 + visit * 1024 + visit * 512
            assert reads == {(ir.MemorySpace.Mat, address) for address in range(start, start + 512)}
            assert writes == {(ir.MemorySpace.Left, address) for address in range(8192, 8192 + 512)}
            assert pipe == ir.PipeType.MTE1
            assert (("payload", i - 1, "C"), ("payload", i, "I")) in order
        wrap = len(after.payloads) - 1
        assert (("payload", extract[0], "C"), ("payload", wrap, "I")) in order
        for i, (_, pipe, reads, writes) in enumerate(after.payloads):
            for j, (_, earlier_pipe, earlier_reads, earlier_writes) in enumerate(after.payloads[:i]):
                if pipe != earlier_pipe and ((earlier_writes & (reads | writes)) or (earlier_reads & writes)):
                    assert (("payload", j, "C"), ("payload", i, "I")) in order, (j, i)
        assert after.acquisitions and after.reuses
        assert after.last_all is not None
        assert all((("payload", i, "C"), after.last_all) in order for i in range(len(after.payloads)))

    # The untaken structured path still performs the slot-0 overwrite. Any
    # prefix event for that overwrite must be closed even without an extract.
    skipped_before = PeerMatrixCommands(originals[1]["cube"], {}, range(8), block_index=1)
    skipped_after = PeerMatrixCommands(structured, {}, range(8), block_index=1)
    skipped_before.run()
    skipped_after.run()
    assert skipped_after.payloads == skipped_before.payloads
    assert skipped_after.descriptors == skipped_before.descriptors
    assert all(call.op.name != opname("buffer.extract") for call, *_ in skipped_after.calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
