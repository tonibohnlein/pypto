# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime row metadata is proved at use, independently of descriptor sentinels."""

import pypto.language as pl
import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import PassManager

from .sync_closed_command_oracle import ClosedCommands, opname
from .sync_logical_oracle import value

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def runtime_program(visits=1, replay=1):
    return pl.parse_program(f"""
@pl.program
class RuntimeBroadcast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,16], pl.FP32], carrier: pl.Tensor[[1,16], pl.FP32],
               rows: pl.Scalar[pl.INDEX], choice: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[8,16], pl.FP32]]) -> pl.Tensor[[8,16], pl.FP32]:
        for outer in pl.range({replay}):
            for visit in pl.range({visits}):
                if visit % 2 == choice:
                    source = pl.load(x, [0,0], [8,16], valid_shape=[rows,16])
                    scale = pl.load(carrier, [0,0], [1,16])
                    result = pl.col_expand_mul(source, scale)
                    out = pl.store(result, [0,0], out)
        return out
""")


def lower(program):
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        result = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer
    return result, function


class ActiveRows(ClosedCommands):
    """Decode explicit runtime metadata and concrete active rectangles from IR.

    This uses input scalar values and setter operands. It consumes no compiler
    effect summaries, demand records or storage partitions. CountMode has fixed
    full columns; aligned physical rows avoid partial local-row padding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = {}

    @staticmethod
    def extents(expression, env):
        assert isinstance(expression, ir.MakeTuple)
        result: list[int] = []
        for item in expression.elements:
            extent = value(item, env)
            assert isinstance(extent, int) and not isinstance(extent, bool)
            result.append(extent)
        return tuple(result)

    def assign(self, statement, env):
        super().assign(statement, env)
        call = statement.value
        if isinstance(call, ir.Call) and call.op.name == opname("buffer.alloc"):
            kind = statement.var.type
            assert isinstance(kind, ir.BufferType)
            supplied = iter(self.extents(call.args[0], env))
            self.valid[statement.var] = tuple(next(supplied) if n < 0 else n for n in kind.valid_shape)

    def evaluate(self, statement, env):
        call = statement.expr
        if isinstance(call, ir.Call) and call.op.name == opname("buffer.set_validshape"):
            extent = self.extents(call.args[1], env)
            kind = call.args[0].type
            assert isinstance(kind, ir.BufferType)
            assert all(0 <= n <= capacity for n, capacity in zip(extent, kind.shape, strict=True))
            self.valid[call.args[0]] = extent
        else:
            super().evaluate(statement, env)

    def bytes(self, operand, rect=None):
        if rect is None and operand in self.valid:
            rect = (0, 0, *self.valid[operand])
        return super().bytes(operand, rect)


def check_runtime_commands(logical, physical, parameters):
    before, after = [ActiveRows(function, parameters, range(6)) for function in (logical, physical)]
    a, b = before.run(), after.run()
    assert before.payloads == after.payloads
    assert before.descriptors == after.descriptors
    assert before.while_tests == after.while_tests
    # Unused immutable completion facts may disappear during lowering. Compare
    # their complete induced payload order, rather than assigning observable
    # hardware command identities to pure facts. Each interpreter independently
    # checks physical matching, acyclicity and actual consumption-before-reuse.
    common = {n for n in before.vertices if n[0] == "payload"}
    assert {(x, y) for x, y in a if x in common and y in common} == {
        (x, y) for x, y in b if x in common and y in common
    }
    # Independently reconstruct RAW/WAR/WAW from actual active physical bytes.
    for i, (_, _, reads, writes) in enumerate(before.payloads):
        for j, (_, _, old_reads, old_writes) in enumerate(before.payloads[:i]):
            if old_writes & (reads | writes) or old_reads & writes:
                assert (("payload", j, "C"), ("payload", i, "I")) in b
    return before, after


@pytest.mark.parametrize("visits,replay", [(1, 1), (3, 1), (3, 2)])
def test_runtime_row_broadcast_preserves_payloads_and_closed_prefixes(visits, replay, tmp_path):
    program, original = lower(runtime_program(visits, replay))
    summary = testing.analyze_conservative_sync(original)
    assert summary["precision"] == "storage_domain_no_kill"
    assert not summary["exact_storage_selectors"]
    declared = {}
    for site in summary["sites"]:
        declared.setdefault(site["operation"].op.name, set()).update(
            (fact["space"], fact["write"]) for fact in site["effects"]
        )
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 8
    scale = torch.linspace(0.5, 2, 16).reshape(1, 16)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [p.get_function("kernel") for p in (logical_program, physical_program)]
    assert logical is not None and physical is not None
    for rows in [0, 1, 7, 8]:
        for choice in [0, 1]:
            out = torch.full_like(x, -777)
            namespace["kernel"](x, scale, rows, choice, out)
            expected = torch.full_like(x, -777)
            if visits > choice:
                expected[:rows] = x[:rows] * scale
            torch.testing.assert_close(out, expected)
            before, after = check_runtime_commands(logical, physical, {2: rows, 3: choice})
            assert after.payloads == before.payloads
            # Check soundness of declared whole-space enclosures separately
            # from exact concrete active bytes and command causality.
            for call, _, _, reads, writes in before.calls:
                for points, write in [(reads, False), (writes, True)]:
                    for memory, _ in points:
                        space = "Vec" if memory == ir.MemorySpace.Vec else "DDR"
                        assert (space, write) in declared[call.op.name]
            if rows == 0:
                # Carrier loads can execute even when the matrix has no rows;
                # the empty broadcast itself must not issue a native operation.
                assert not any(p[0][0] == opname("buffer.col_expand_mul") for p in after.payloads)
    if find_ptoas_binary() is not None:
        source, target = tmp_path / "runtime.pto", tmp_path / "runtime.cpp"
        source.write_text(codegen.PTOCodegen().generate(physical_program))
        _run_ptoas(str(source), str(target), ["--pto-level=level3", "--pto-arch=a2"])
        assert target.is_file()


def metadata_fixture(kind="matched"):
    """Explicit Buffer calls retain the existing 0<=valid<=capacity input contract.

    This verifier checks use-site metadata relations/positivity, and does not
    claim to prove the caller-supplied upper bound for arbitrary scalar inputs.
    """
    span = SPAN

    def ci(n):
        return ir.ConstInt(n, ir.DataType.INDEX, span)

    def tup(items):
        return ir.MakeTuple([ci(n) if isinstance(n, int) else n for n in items], span)

    def call(op, args, ty=None):
        return ir.Call(ir.get_op(op), args, ty or ir.VoidType(), span)

    def evaluate(op, args):
        return ir.EvalStmt(call(op, args), span)

    r1, r2 = [ir.Var(n, ir.ScalarType(ir.DataType.INDEX), span) for n in ["r1", "r2"]]
    matrix = ir.BufferType([8, 16], ir.DataType.FP32, ir.MemorySpace.Vec, valid_shape=[-1, -1])
    row = ir.BufferType([1, 16], ir.DataType.FP32, ir.MemorySpace.Vec)
    source, target, carrier = [
        ir.Var(n, t, span) for n, t in [("source", matrix), ("target", matrix), ("carrier", row)]
    ]
    allocations = [
        ir.AssignStmt(
            v, call("buffer.alloc", [tup([r1, 16]) if t is matrix else tup([]), ci(address)], t), span
        )
        for v, t, address in [(source, matrix, 0), (target, matrix, 1024), (carrier, row, 2048)]
    ]
    native = evaluate("buffer.col_expand_mul", [source, carrier, target])
    setters: list[ir.Stmt] = [
        evaluate("buffer.set_validshape", [source, tup([r1, 16])]),
        evaluate(
            "buffer.set_validshape", [target, tup([r2 if kind in {"mismatch", "guarded_equal"} else r1, 16])]
        ),
    ]
    positive = ir.Gt(r1, ci(0), ir.DataType.BOOL, span)
    if kind in {"unchecked_arithmetic", "unchecked_arithmetic_else"}:
        # r1 + 1 > 1 does not establish r1 > 0 under unspecified signed
        # overflow. An algebraic analyzer must not add that missing premise.
        positive = ir.Gt(ir.Add(r1, ci(1), ir.DataType.INDEX, span), ci(1), ir.DataType.BOOL, span)
    if kind == "guarded_equal":
        positive = ir.And(positive, ir.Eq(r1, r2, ir.DataType.BOOL, span), ir.DataType.BOOL, span)
    body = ir.IfStmt(positive, native, None, [], span)
    if kind == "unchecked_arithmetic_else":
        body = ir.IfStmt(positive, ir.SeqStmts([], span), ir.SeqStmts([*setters, native], span), [], span)
    if kind == "unguarded":
        body = native
    if kind == "join":
        setters = [ir.IfStmt(positive, ir.SeqStmts(setters, span), None, [], span)]
    if kind == "reset":
        iv = ir.Var("i", ir.ScalarType(ir.DataType.INDEX), span)
        body = ir.ForStmt(
            iv, ci(0), ci(2), ci(1), [], ir.SeqStmts([body, ir.YieldStmt([], span)], span), [], span
        )
    return ir.Function(
        "metadata",
        [r1, r2],
        [],
        ir.SeqStmts([*allocations, *setters, body, ir.ReturnStmt([], span)], span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def diagnostics(function):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.BufferIR)
    return passes.PropertyVerifierRegistry.verify(props, ir.Program([function], "Metadata", function.span))


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("mismatch", "matching runtime rows"),
        ("unguarded", "positive-row guard"),
        ("unchecked_arithmetic", "positive-row guard"),
        ("unchecked_arithmetic_else", "positive-row guard"),
        ("join", "unavailable"),
        ("reset", "unavailable"),
    ],
)
def test_unproved_runtime_metadata_remains_a_refusal(kind, reason):
    function = metadata_fixture(kind)
    errors = diagnostics(function)
    assert errors and any(reason in e.message for e in errors)
    with pytest.raises(ValueError, match=reason):
        codegen.PTOCodegen().generate(ir.Program([function], "Metadata", function.span))


def test_equal_descriptor_sentinels_do_not_replace_actual_metadata():
    assert not diagnostics(metadata_fixture())
    assert diagnostics(metadata_fixture("mismatch"))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_runtime_row_countmode_retains_target_qualification(ascend_backend):
    assert ascend_backend == BackendType.Ascend950
    function = metadata_fixture()
    with pytest.raises(ValueError, match="A2/A3"):
        codegen.PTOCodegen().generate(ir.Program([function], "Metadata", function.span))


def test_zero_trip_runtime_broadcast_has_no_native_payload():
    program = runtime_program(0, 2)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        result = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)

    class Calls(ir.IRVisitor):
        count = 0

        def visit_call(self, op):
            self.count += op.op.name in {
                ir.get_op("buffer.col_expand_mul").name,
                ir.get_op("tile.col_expand_mul").name,
            }
            super().visit_call(op)

    calls = Calls()
    calls.visit_program(result)
    assert calls.count == 0


def test_runtime_row_retained_prefix_size_is_independent_of_trip_count():
    summaries = []
    for count in [3, 1000000, 2000000]:
        _, function = lower(runtime_program(count, 2))
        summaries.append(testing.analyze_conservative_sync(function))
    assert all(s["cost"] == summaries[0]["cost"] for s in summaries)
    assert all(len(s["sites"]) == len(summaries[0]["sites"]) for s in summaries)


def test_actual_metadata_relation_can_be_established_by_original_guard():
    function = metadata_fixture("guarded_equal")
    assert not diagnostics(function)
    assert "pto.tcolexpandmul" in codegen.PTOCodegen().generate(ir.Program([function], "Metadata", SPAN))
