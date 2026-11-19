# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise shared completion expressions through the standalone optimizer."""

import subprocess
import sys
from pathlib import Path

import pytest
from pypto import ir, passes
from pypto.ir.op import sync


def make_shared_join_program(use: str, depth: int) -> ir.Program:
    """Build three join nodes per level without parsing or printing the DAG."""
    span = ir.Span.unknown()
    source = ir.PipeType.MTE2
    token_type = ir.SyncTokenType(source)
    earlier, later, ready, unused, joined = (
        ir.Var(name, token_type, span) for name in ("earlier", "later", "ready", "unused", "joined")
    )
    # Reusing the record expression still captures two position-sensitive
    # occurrences. Neither record binding may be treated as the other's alias.
    record = sync.record(source)
    base = sync.join(sync.join(earlier, ready), later)
    shared = base
    for _ in range(depth):
        shared = sync.join(sync.join(shared, earlier), sync.join(shared, later))

    body: list[ir.Stmt] = [
        ir.AssignStmt(earlier, record, span),
        ir.EvalStmt(  # Separate the two record cuts on their source pipe.
            sync.wait(earlier, source), span
        ),
        ir.AssignStmt(later, record, span),
        ir.AssignStmt(ready, sync.completed(source), span),
    ]
    if use == "binding":
        # The first use is a dead assignment inside a branch. It must neither
        # keep the DAG live nor stop the later live assignment from reaching it.
        inactive = ir.AssignStmt(unused, shared, span)
        body.append(ir.IfStmt(ir.ConstBool(False, span), ir.SeqStmts([inactive], span), None, [], span))
        body.append(ir.AssignStmt(joined, shared, span))
        body.append(ir.EvalStmt(sync.wait(joined, ir.PipeType.V), span))
    else:
        body.append(ir.EvalStmt(sync.wait(shared, ir.PipeType.V), span))
    body.append(ir.ReturnStmt([], span))
    function = ir.Function("kernel", [], [], ir.SeqStmts(body, span), span, type=ir.FunctionType.AIV)
    return ir.Program([function], "SharedJoinDag", span)


def _check_optimization(use: str, depth: int) -> None:
    result = passes.optimize_sync()(make_shared_join_program(use, depth)).get_function("kernel")
    assert result is not None and isinstance(result.body, ir.SeqStmts)
    statements = result.body.stmts
    definitions = {stmt.var.name_hint: stmt for stmt in statements if isinstance(stmt, ir.AssignStmt)}
    waits = [stmt.expr for stmt in statements if isinstance(stmt, ir.EvalStmt)]
    assert len(waits) == 2
    assert isinstance(waits[-1], ir.Call)
    if use == "binding":
        assert set(definitions) == {"earlier", "later", "joined"}
        branch = next(stmt for stmt in statements if isinstance(stmt, ir.IfStmt))
        assert isinstance(branch.then_body, ir.SeqStmts) and not branch.then_body.stmts
        assert waits[-1].args[0] is definitions["joined"].var
        shared = definitions["joined"].value
    else:
        assert set(definitions) == {"earlier", "later", "ready"}
        shared = waits[-1].args[0]

    for _ in range(depth):
        assert isinstance(shared, ir.Call)
        lhs, rhs = shared.args
        assert isinstance(lhs, ir.Call) and isinstance(rhs, ir.Call)
        assert lhs.args[0] is rhs.args[0]
        assert lhs.args[1] is definitions["earlier"].var
        assert rhs.args[1] is definitions["later"].var
        shared = lhs.args[0]
    assert isinstance(shared, ir.Call)
    assert shared.args[1] is definitions["later"].var
    if use == "binding":
        assert shared.args[0] is definitions["earlier"].var


@pytest.mark.parametrize("use", ["binding", "wait"])
def test_optimize_sync_shared_join_dag(use: str) -> None:
    """A tree traversal of the 60-level DAG would explore over 2**60 paths."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), use],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_optimization(sys.argv[1], 60)
