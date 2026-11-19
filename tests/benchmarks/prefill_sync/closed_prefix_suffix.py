# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure standalone LowerSync on staggered records in one loop body.

From a built checkout run
    PYTHONPATH=python python tests/benchmarks/prefill_sync/closed_prefix_suffix.py

Each group adds four statements and has at most two outstanding tokens.
The measured interval excludes construction and printing of the IR.
"""

import argparse
from statistics import median
from time import perf_counter

from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import sync

SPAN = ir.Span.unknown()


def make_program(groups: int) -> ir.Program:
    index_type = ir.ScalarType(ir.DataType.INDEX)
    body: list[ir.Stmt] = []
    load, vector = ir.PipeType.MTE2, ir.PipeType.V
    for i in range(groups):
        a = ir.Var(f"a{i}", ir.SyncTokenType(load), SPAN)
        b = ir.Var(f"b{i}", ir.SyncTokenType(vector), SPAN)
        body.extend(
            [
                ir.AssignStmt(a, sync.record(load), SPAN),
                ir.AssignStmt(b, sync.record(vector), SPAN),
                ir.EvalStmt(sync.wait(a, vector), SPAN),
                ir.EvalStmt(sync.wait(b, load), SPAN),
            ]
        )
    body.append(ir.YieldStmt([], SPAN))
    loop = ir.ForStmt(
        ir.Var("i", index_type, SPAN),
        ir.ConstInt(0, ir.DataType.INDEX, SPAN),
        ir.ConstInt(3, ir.DataType.INDEX, SPAN),
        ir.ConstInt(1, ir.DataType.INDEX, SPAN),
        [],
        ir.SeqStmts(body, SPAN),
        [],
        SPAN,
    )
    function = ir.Function(
        "staggered_prefixes",
        [],
        [],
        ir.SeqStmts([loop, ir.ReturnStmt([], SPAN)], SPAN),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "StaggeredPrefixes", SPAN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.groups) < 1 or args.repeats < 1:
        parser.error("groups and repeats must be positive")

    backend.set_backend_type(BackendType.Ascend910)
    print("groups,loop_statements,median_seconds")
    with passes.PassContext([], enable_pypto_autosync=True):
        lower = passes.lower_sync()
        lower(make_program(1))
        for groups in args.groups:
            program = make_program(groups)
            durations = []
            for _ in range(args.repeats):
                start = perf_counter()
                physical = lower(program)
                durations.append(perf_counter() - start)
                del physical
            print(f"{groups},{4 * groups + 1},{median(durations):.6f}")


if __name__ == "__main__":
    main()
