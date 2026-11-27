# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Root GM argument provenance for paired synchronization ownership."""

from textwrap import indent

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager
from pypto.runtime import RunConfig

from .test_peer_loop_slot_order import PeerSenderCommands

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _peer_program(
    root_call: str,
    *,
    rhs_rows: int = 16,
    spmd: bool = False,
    uses_block_index: bool = True,
    runtime_loop: bool = False,
    projects_result: bool = False,
    guarded_transport: bool = False,
) -> ir.Program:
    out_rows = 384 if runtime_loop else (64 if spmd else 16)
    row = "pl.get_block_idx() * 16" if spmd and uses_block_index else "0"
    width_formal = ", width: pl.Scalar[pl.INT32]" if runtime_loop else ""
    width_actual = ", width" if runtime_loop else ""
    cube_work = f"""received: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
left = pl.move(received, target_memory=pl.Mem.Left)
pl.tfree_to_aiv(received)
product = pl.matmul(left, right)
out = pl.store(product, [{row}, 0], out)"""
    vector_work = "pl.tpush_to_aic(data, split=0)"
    if runtime_loop:
        # A 3-item grid-stride scan has zero trips on lanes past item 2.
        # Without the exact width relation the subtraction admits negative values.
        header = """core = pl.tile.get_block_idx()
wide = pl.cast(width, pl.INDEX)
trips = (wide - core + 2) // wide
for _ in pl.range(trips):
"""
        if guarded_transport:
            # The qualified dispatch width also controls whether a work item
            # issues its FIFO commands. The loop may carry a rotated cursor.
            cube_work = "if width < 8:\n" + indent(cube_work, "    ")
            vector_work = "if width < 8:\n" + indent(vector_work, "    ")
        cube_work = header + indent(cube_work, "    ")
        vector_work = header + indent(vector_work, "    ")
    output = f"pl.Tensor[[{out_rows}, 16], pl.FP32]"
    group_output = f"tuple[{output}, {output}]" if projects_result else output
    group_return = "result, out" if projects_result else "result"
    call = f"self.paired(rhs, x, out{width_actual})"
    wrapper_return = (
        f"pair = {call}\n        ignored = pair[1]\n        return pair[0]"
        if projects_result
        else f"return {call}"
    )
    wrapper = (
        f"""
    @pl.function(type=pl.FunctionType.Spmd)
    def grid(self, rhs: pl.Tensor[[16, 16], pl.FP16], x: pl.Tensor[[16, 16], pl.FP16],
             out: pl.Out[{output}]{width_formal}) -> {output}:
        {wrapper_return}
"""
        if spmd
        else ""
    )
    return pl.parse_program(f"""
@pl.program
class PeerInvocations:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, rhs: pl.Tensor[[16, 16], pl.FP16],
             out: pl.Out[{output}]{width_formal}) -> {output}:
        fifo = pl.reserve_buffer(name="inbox", size=2048, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size=1024, slot_num=2)
        matrix = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        right = pl.move(matrix, target_memory=pl.Mem.Right)
{indent(cube_work, "        ")}
        return out

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[16, 16], pl.FP16]{width_formal}):
        peer = pl.import_peer_buffer(name="inbox", peer_func="cube")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=1024, slot_num=2)
        data = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
{indent(vector_work, "        ")}

    @pl.function(type=pl.FunctionType.Group)
    def paired(self, rhs: pl.Tensor[[16, 16], pl.FP16], x: pl.Tensor[[16, 16], pl.FP16],
               out: pl.Out[pl.Tensor[[{out_rows}, 16], pl.FP32]]{width_formal}) -> {group_output}:
        result = self.cube(rhs, out{width_actual})
        self.vector(x{width_actual})
        return {group_return}
{wrapper}
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, rhs: pl.Tensor[[{rhs_rows}, 16], pl.FP16], x: pl.Tensor[[16, 16], pl.FP16],
             out: pl.Out[pl.Tensor[[{out_rows}, 16], pl.FP32]]) -> pl.Tensor[[{out_rows}, 16], pl.FP32]:
{indent(root_call, "        ")}
""")


@pytest.mark.parametrize(
    "root_call,rhs_rows,refusal",
    [
        pytest.param("return self.paired(rhs, x, out)", 16, None, id="distinct-actuals"),
        pytest.param(
            "return self.paired(x, x, out)",
            16,
            None,
            id="shared-ordinary-input",
        ),
        pytest.param(
            "first = self.paired(rhs, x, out)\nreturn self.paired(x, x, first)",
            16,
            None,
            id="explicit-writeback-in-second-invocation",
        ),
        pytest.param(
            "carried = out\nfor _ in pl.range(3):\n"
            "    carried = self.paired(rhs, x, carried)\nreturn carried",
            16,
            None,
            id="same-root-loop-backedge",
        ),
        pytest.param(
            "carried = out\nfor _ in pl.range(3):\n"
            "    self.paired(rhs, x, carried)\n"
            "    carried = pl.create_tensor([16, 16], dtype=pl.FP32)\n"
            "return carried",
            16,
            "GM tensor must forward an original root formal",
            id="conflicting-loop-backedge",
        ),
        pytest.param(
            "sub = pl.slice(rhs, [16, 16], [0, 0])\nreturn self.paired(sub, x, out)",
            32,
            "GM tensor must forward an original root formal",
            id="unknown-subview",
        ),
    ],
)
def test_peer_group_checks_root_invocation(root_call: str, rhs_rows: int, refusal: str | None) -> None:
    original = _peer_program(root_call, rhs_rows=rhs_rows)
    with _autosync_context([]):
        lowered = PassManager(ir.OptimizationStrategy.Default).run_passes(original)
    _assert_outcome(lowered, refusal)


@pytest.mark.parametrize(
    "count,uses_block_index,refusal",
    [
        pytest.param(4, True, None, id="sharded"),
        pytest.param(4, False, "no bounded, nonoverlapping native partition", id="unsharded"),
        pytest.param(2**30, True, "no bounded, nonoverlapping native partition", id="shape-overflow"),
    ],
)
def test_peer_group_workspace_follows_spmd_launch(
    count: int, uses_block_index: bool, refusal: str | None
) -> None:
    # The root allocation is sized for the whole dispatch. The managed
    # workspace formal reaches the paired endpoints through the SPMD wrapper.
    original = _peer_program(
        "with pl.manual_scope():\n"
        f"    result, _ = pl.spmd_submit(self.grid, rhs, x, out, core_num={count})\n"
        "return result",
        spmd=True,
        uses_block_index=uses_block_index,
    )
    with _autosync_context([]):
        lowered = PassManager(ir.OptimizationStrategy.Default).run_passes(original)
    _assert_outcome(lowered, refusal)


@pytest.mark.parametrize(
    "actuals, projects_result, guarded_transport, refusal",
    [
        pytest.param(("width",), False, False, None, id="same-launch-width"),
        pytest.param(("width",), True, False, None, id="tuple-result-projection"),
        pytest.param(
            ("pl.const(1, pl.INT32)",), False, False, "ready nonnegative bounds", id="unrelated-scalar"
        ),
        pytest.param(
            ("width", "pl.const(1, pl.INT32)"),
            False,
            False,
            "ready nonnegative bounds",
            id="one-of-two-launches-disagrees",
        ),
        pytest.param(("width",), False, True, None, id="guarded-transport"),
        pytest.param(
            ("pl.const(1, pl.INT32)",),
            False,
            True,
            "ready nonnegative bounds",
            id="guarded-unrelated-scalar",
        ),
    ],
)
def test_peer_work_loop_uses_original_dispatch_width(
    actuals, projects_result, guarded_transport, refusal
) -> None:
    calls = ["width = pl.system.available_cluster_count()", "with pl.manual_scope():"]
    source = "out"
    for idx, actual in enumerate(actuals):
        target = f"out_{idx}"
        calls.append(
            f"    {target}, _ = pl.spmd_submit(self.grid, rhs, x, {source}, {actual}, core_num=width)"
        )
        source = target
    calls.append(f"return {source}")
    original = _peer_program(
        "\n".join(calls),
        spmd=True,
        runtime_loop=True,
        projects_result=projects_result,
        guarded_transport=guarded_transport,
    )
    with _autosync_context([]):
        lowered = PassManager(ir.OptimizationStrategy.Default).run_passes(original)
    for name in ("cube", "vector"):
        endpoint = lowered.get_function(name)
        assert endpoint is not None and endpoint.attrs["buffer_ir_selected"]
    _assert_outcome(lowered, refusal)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("uses_block_index", [True, False])
def test_peer_group_workspace_native_partition(tmp_path, uses_block_index: bool) -> None:
    original = _peer_program(
        "with pl.manual_scope():\n"
        "    result, _ = pl.spmd_submit(self.grid, rhs, x, out, core_num=4)\n"
        "return result",
        spmd=True,
        uses_block_index=uses_block_index,
    )
    config = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with _autosync_context([]):
        ir.compile(original, **config.compile_kwargs())
    (orchestration,) = (tmp_path / "orchestration").glob("*.cpp")
    # Two 1024-byte slots / four bytes per element, replicated four times.
    assert "static_cast<uint32_t>((512) * (4))" in orchestration.read_text()
    for side, kernel in (("aic", "cube"), ("aiv", "vector")):
        native = (tmp_path / "kernels" / side / f"{kernel}.cpp").read_text()
        assert ("__pypto_gm_block_offset" in native) == uses_block_index


def _autosync_context(instruments: list[passes.PassInstrument]) -> passes.PassContext:
    return passes.PassContext(
        instruments,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
    )


def _assert_outcome(lowered: ir.Program, refusal: str | None) -> None:
    for name in ("cube", "vector"):
        endpoint = lowered.get_function(name)
        assert endpoint is not None
        assert endpoint.attrs[PYPTO_AUTOSYNC_ATTR] == (refusal is None), endpoint.attrs
        if refusal is not None:
            assert refusal in endpoint.attrs[AUTOSYNC_FALLBACK_REASON_ATTR]


class _RewireWorkspace(ir.IRMutator):
    def __init__(self, aliases_output: bool) -> None:
        super().__init__()
        self.aliases_output = aliases_output
        self.replacement: ir.Var | None = None

    def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
        if not self.aliases_output and op.var.name_hint.startswith("gm_pipe_buffer_"):
            self.replacement = ir.Var("tiny", op.var.type, op.var.span)
            return ir.AssignStmt(self.replacement, op.value, op.span)
        return super().visit_assign_stmt(op)

    def visit_call(self, op: ir.Call) -> ir.Expr:
        if isinstance(op.op, ir.GlobalVar) and op.op.name == "paired":
            args = list(op.args)
            assert len(args) == 4
            workspace = args[2] if self.aliases_output else self.replacement
            assert workspace is not None
            args[-1] = workspace
            return ir.Call(op.op, args, op.kwargs, op.attrs, op.type, op.span)
        return super().visit_call(op)


@pytest.mark.parametrize(
    "aliases_output,refusal",
    [
        pytest.param(
            False,
            "managed workspace requires a unique injected root tensor.create",
            id="tiny-ordinary-create",
        ),
        pytest.param(
            True, "managed workspace aliases an ordinary root tensor", id="workspace-aliases-output"
        ),
    ],
)
def test_peer_group_requires_a_distinct_runtime_sized_workspace(aliases_output: bool, refusal: str) -> None:
    before_realize: list[ir.Program] = []

    def capture(transform: passes.Pass, program: ir.Program) -> None:
        if transform.get_name() == "RealizeSync":
            before_realize.append(program)

    # Let the default pipeline inject its private workspace ABI. Change only
    # the root binding before re-running the production ownership pass.
    with _autosync_context([passes.CallbackInstrument(before_pass=capture)]):
        PassManager(ir.OptimizationStrategy.Default).run_passes(
            _peer_program("return self.paired(rhs, x, out)")
        )
    (placed,) = before_realize
    main = placed.get_function("main")
    assert main is not None
    mutated = _RewireWorkspace(aliases_output).visit_function(main)
    forged = ir.Program(
        [mutated if fn.name == main.name else fn for fn in placed.functions.values()],
        placed.name,
        placed.span,
    )
    with _autosync_context([]):
        lowered = passes.realize_sync()(forged)
    for name in ("cube", "vector"):
        endpoint = lowered.get_function(name)
        assert endpoint is not None and not endpoint.attrs[PYPTO_AUTOSYNC_ATTR]
        # The cube accesses out. The vector only reads x, so it rejects the
        # substituted workspace on allocation provenance alone.
        reason = "unique injected root tensor.create" if aliases_output and name == "vector" else refusal
        assert reason in endpoint.attrs[AUTOSYNC_FALLBACK_REASON_ATTR]


@pytest.mark.parametrize(
    "root_calls,may_alias",
    [
        ("self.paired(x, tmp)", False),
        ("self.paired(x, out)", True),
        ("self.paired(tmp, tmp)", True),
        (
            "self.paired(x, tmp)\nother = pl.create_tensor([16, 16], dtype=pl.FP16)\nself.paired(other, other)",
            True,
        ),
    ],
)
@pytest.mark.parametrize("push_latest", [False, True])
def test_fifo_local_completions_use_transfer_operand_and_all_callers(root_calls, may_alias, push_latest):
    # A push consumes one of two results on V. A subsequent GM load can
    # overlap the previous store only when every call separates their roots.
    pushed = "late" if push_latest else "early"
    program = pl.parse_program(f"""
@pl.program
class PeerCopies:
    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[16, 16], pl.FP16], out: pl.Out[pl.Tensor[[16, 16], pl.FP16]]):
        peer = pl.import_peer_buffer(name="inbox", peer_func="cube")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=1024, slot_num=2)
        core = pl.tile.get_block_idx()
        if core < 1:
            data = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            early = pl.add(data, data)
            late = pl.add(early, early)
            pl.tpush_to_aic({pushed}, split=0)
            out = pl.store(late, [0, 0], out)
            following = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            combined = pl.add(following, early)
            pl.tpush_to_aic(pl.add(combined, late), split=0)

    @pl.function(type=pl.FunctionType.AIC)
    def cube(self):
        fifo = pl.reserve_buffer(name="inbox", size=2048, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size=1024, slot_num=2)
        for i in pl.range(2):
            pkt: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(pkt)

    @pl.function(type=pl.FunctionType.Group)
    def paired(self, x: pl.Tensor[[16, 16], pl.FP16], out: pl.Out[pl.Tensor[[16, 16], pl.FP16]]):
        self.cube()
        self.vector(x, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16, 16], pl.FP16], out: pl.Out[pl.Tensor[[16, 16], pl.FP16]]):
        tmp = pl.create_tensor([16, 16], dtype=pl.FP16)
{indent(root_calls, "        ")}
""")
    with _autosync_context([]):
        lowered = PassManager(ir.OptimizationStrategy.Default).run_passes(program)
    leaf = lowered.get_function("vector")
    assert leaf is not None
    assert leaf.attrs.get(PYPTO_AUTOSYNC_ATTR), str(leaf.attrs)
    commands = PeerSenderCommands(leaf, 0)
    for op, pipe in (
        ("buffer.add", ir.PipeType.V),
        ("buffer.store", ir.PipeType.MTE3),
    ):
        commands.payload_pipes[ir.get_op(op).name] = pipe
    order = commands.run()
    assert [c.op.name for c in commands.calls] == [
        ir.get_op("buffer." + name).name
        for name in ("load", "add", "add", "peer_push", "store", "load", "add", "add", "peer_push")
    ]
    assert (("done", 1), ("issue", 3)) in order
    assert ((("done", 2), ("issue", 3)) in order) == push_latest
    assert (("done", 2), ("issue", 4)) in order
    assert ((("done", 4), ("issue", 5)) in order) == may_alias
