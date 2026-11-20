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
from pypto.ir.pass_manager import PassManager

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _peer_program(root_call: str, *, rhs_rows: int = 16) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class PeerInvocations:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, rhs: pl.Tensor[[16, 16], pl.FP16],
             out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        fifo = pl.reserve_buffer(name="inbox", size=2048, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size=1024, slot_num=2)
        matrix = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        right = pl.move(matrix, target_memory=pl.Mem.Right)
        received: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
        left = pl.move(received, target_memory=pl.Mem.Left)
        pl.tfree_to_aiv(received)
        product = pl.matmul(left, right)
        return pl.store(product, [0, 0], out)

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[16, 16], pl.FP16]):
        peer = pl.import_peer_buffer(name="inbox", peer_func="cube")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=1024, slot_num=2)
        data = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        pl.tpush_to_aic(data, split=0)

    @pl.function(type=pl.FunctionType.Group)
    def paired(self, rhs: pl.Tensor[[16, 16], pl.FP16], x: pl.Tensor[[16, 16], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        result = self.cube(rhs, out)
        self.vector(x)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, rhs: pl.Tensor[[{rhs_rows}, 16], pl.FP16], x: pl.Tensor[[16, 16], pl.FP16],
             out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
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
            "GM tensor must forward an original root formal",
            id="opaque-return-in-second-invocation",
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
