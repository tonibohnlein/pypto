# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Acc row trims share bytes, but restore independently owned native metadata."""

from textwrap import indent

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import PassManager

from .buffer_test_utils import statements
from .sync_closed_command_oracle import opname
from .sync_logical_oracle import value
from .sync_matmul_oracle import address_set, protected_accumulator_pairs
from .test_runtime_valid_broadcast import ActiveRows

pytestmark = pytest.mark.usefixtures("ascend_backend")
PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]
BASES = {i: i * (1 << 24) for i in (0, 1, 4, 5, 6)}


def trim_program(m=32, n=64, *, visits=1, replay=1, variant="ordinary"):
    rows = "rows_a" if variant == "unbounded" else f"pl.min(pl.max(rows_a, 0), {m})"
    columns = n - 1 if variant == "columns" else n
    store = "seen_a = pl.store(trim_a, [0, 0], seen_a)"
    guarded_store = store if variant == "unguarded" else f"if r_a > 0:\n                    {store}"
    source = f"""
@pl.program
class AccRowTrim:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[{m},32], pl.FP16], b: pl.Tensor[[32,{n}], pl.FP16],
               rows_a: pl.Scalar[pl.INDEX], rows_b: pl.Scalar[pl.INDEX],
               seen_a: pl.InOut[pl.Tensor[[{m},{n}], pl.FP32]],
               seen_b: pl.InOut[pl.Tensor[[{m},{n}], pl.FP32]],
               full: pl.InOut[pl.Tensor[[{m},{n}], pl.FP32]]) -> pl.Tensor[[{m},{n}], pl.FP32]:
        for outer in pl.range({replay}):
            for visit in pl.range({visits}):
                a_mat = pl.load(a, [0, 0], [{m}, 32], target_memory=pl.Mem.Mat)
                b_mat = pl.load(b, [0, 0], [32, {n}], target_memory=pl.Mem.Mat)
                lhs = pl.move(a_mat, target_memory=pl.Mem.Left)
                rhs = pl.move(b_mat, target_memory=pl.Mem.Right)
                seed = pl.matmul(lhs, rhs)
                acc = pl.matmul_acc(seed, lhs, rhs)
                full = pl.store(acc, [0, 0], full)
                r_a = {rows}
                r_b = pl.min(pl.max(rows_b, 0), {m})
                trim_a = pl.tile.set_validshape(acc, r_a, {columns})
                trim_b = pl.tile.set_validshape(acc, r_b, {n})
                {guarded_store}
                if r_b > 0:
                    seen_b = pl.store(trim_b, [0, 0], seen_b)
                if r_a > 0:
                    seen_a = pl.store(trim_a, [0, 0], seen_a)
                full = pl.store(acc, [0, 0], full)
        return full
"""
    if variant == "carry":
        source = source.replace(f"for visit in pl.range({visits}):", "for visit in pl.range(1):")
        source = source.replace(
            "acc = pl.matmul_acc(seed, lhs, rhs)",
            f"acc = seed\n                for k in pl.range({visits}):\n"
            "                    acc = pl.matmul_acc(acc, lhs, rhs)",
        )
    return pl.parse_program(source)


def lower(program, planner):
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        return PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)


def kernel(program):
    result = program.get_function("kernel")
    assert result is not None
    return result


class AccRows(ActiveRows):
    """Concrete native row metadata plus physical byte footprints, no effect summaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stores = []
        self.matrix_calls = []
        self.writers = {}

    def payload(self, call, env):
        if call.op.name in (opname("buffer.matmul"), opname("buffer.matmul_acc")):
            destination = call.args[2]
            assert isinstance(destination.type, ir.BufferType)
            assert self.valid[destination] == tuple(destination.type.shape)
            self.matrix_calls.append(call)
            self.writers[self.storage[destination]] = destination.type
        super().payload(call, env)
        if call.op.name == opname("buffer.store"):
            source, offsets, extents, destination = call.args
            assert isinstance(source.type, ir.BufferType)
            rows, columns = self.extents(extents, env)
            assert rows > 0 and columns == source.type.shape[1]
            assert self.valid[source] == (rows, columns)
            # Independently expand the original full Acc packing. Valid rows
            # restrict membership, never the physical stride or layout.
            space, base, m, n, item_bytes = self.storage[source]
            writer = self.writers[self.storage[source]]
            # Same packing and byte window preserve each initialized cell's
            # value, not merely the union of bytes read by the native store.
            for field in ("shape", "dtype", "blayout", "slayout", "fractal", "compact"):
                assert getattr(source.type, field) == getattr(writer, field)
            assert source.type.compact == ir.CompactMode.null
            points = address_set((space, base, m, n, source.type, item_bytes), 0, 0, rows, n)
            expected_reads = frozenset(
                (memory, address + byte) for memory, address in points for byte in range(item_bytes)
            )
            row, column = (value(x, env) for x in offsets.elements)
            assert isinstance(row, int) and isinstance(column, int)
            gm_space, gm_base, _, gm_width, gm_bytes = self.storage[destination]
            expected_writes = frozenset(
                (gm_space, gm_base + ((row + r) * gm_width + column + c) * gm_bytes + byte)
                for r in range(rows)
                for c in range(columns)
                for byte in range(gm_bytes)
            )
            _, pipe, reads, writes = self.payloads[-1]
            assert pipe == ir.PipeType.FIX and reads == expected_reads and writes == expected_writes
            self.stores.append((gm_base, rows, columns, base, reads))


def check_commands(original, logical, physical, parameters, *, output_rows=None):
    source, before, after = [
        AccRows(
            function,
            parameters,
            range(8),
            BASES,
            shape_parameters={} if output_rows is None else {function.params[2].type.shape[0]: output_rows},
        )
        for function in (original, logical, physical)
    ]
    for state in (source, before, after):
        state.run()
    assert source.payloads == before.payloads == after.payloads
    assert source.stores == before.stores == after.stores
    assert source.descriptors == before.descriptors == after.descriptors
    common = {node for node in before.vertices if node[0] == "payload"}
    assert {(a, b) for a, b in before.order if a in common and b in common} == {
        (a, b) for a, b in after.order if a in common and b in common
    }
    # A2/A3 protects qualified accumulator hazards in the matrix hardware.
    # Check this exemption against the original calls before requiring an event edge.
    protected = protected_accumulator_pairs(original, source.calls)
    for i, (_, _, reads, writes) in enumerate(before.payloads):
        for j, (_, _, old_reads, old_writes) in enumerate(before.payloads[:i]):
            if (old_writes & (reads | writes) or old_reads & writes) and (j, i) not in protected:
                assert (("payload", j, "C"), ("payload", i, "I")) in after.order
    assert all(pair in after.order for pair in after.reuses)
    assert not after.live
    assert all(ids <= set(range(8)) for ids in after.ids.values())
    return after


def sync_pipeline(program, *, disjoint=False):
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=disjoint):
        logical = passes.insert_sync()(program)
        optimized = passes.optimize_sync()(logical)
    # Exercise the production palette used for qualified accumulator trims.
    physical = kernel(passes.lower_sync()(optimized))
    return kernel(logical), physical


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("m,n", [(16, 32), (32, 64), (64, 128)])
@pytest.mark.parametrize("visits,replay", [(1, 1), (2, 2)])
@pytest.mark.parametrize("disjoint", [False, True])
def test_addressed_trim_aliases_restore_rows_and_old_full_source(planner, m, n, visits, replay, disjoint):
    program = lower(trim_program(m, n, visits=visits, replay=replay), planner)
    original = kernel(program)
    assert original.ir_stage == ir.FunctionIRStage.Buffer
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored)
    logical, physical = sync_pipeline(restored, disjoint=disjoint)
    for rows_a in (0, 1, m - 1, m):
        rows_b = m - rows_a
        state = check_commands(original, logical, physical, {2: rows_a, 3: rows_b})
        one_visit = [(BASES[6], m)]
        if rows_a:
            one_visit.append((BASES[4], rows_a))
        if rows_b:
            one_visit.append((BASES[5], rows_b))
        if rows_a:
            one_visit.append((BASES[4], rows_a))
        one_visit.append((BASES[6], m))
        assert [(base, rows) for base, rows, _, _, _ in state.stores] == one_visit * (visits * replay)
        assert len(state.matrix_calls) == 2 * visits * replay
        assert all(tuple(call.args[2].type.valid_shape) == (m, n) for call in state.matrix_calls)
    allocations = [
        statement.var.type
        for statement in statements(program)
        if isinstance(statement, ir.AssignStmt)
        and isinstance(statement.value, ir.Call)
        and statement.value.op.name == opname("buffer.alloc")
        and isinstance(statement.var.type, ir.BufferType)
        and statement.var.type.memory_space == ir.MemorySpace.Acc
    ]
    assert any(tuple(kind.valid_shape) == (m, n) for kind in allocations)
    assert any(tuple(kind.valid_shape) == (-1, -1) for kind in allocations)
    assert all(tuple(kind.valid_shape) != (-1, n) for kind in allocations)


def selection_input(program):
    manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
    boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
    for transform in manager.passes[:boundary]:
        program = transform(program)
    return program


def tiled_caller_program(variant):
    tile = (1 << 63) - 8 if variant == "overflow" else 16
    # Codegen resolves the free shape symbol to the scalar ABI argument when
    # their emitted names coincide. That argument has the full signed range.
    dimension = "launch_count" if variant == "shadowed-shape" else "ROWS"
    scalar = ", launch_count: pl.Scalar[pl.INDEX]" if variant == "shadowed-shape" else ""
    calls = f"""
for chunk in pl.range((ROWS + {tile - 1}) // {tile}):
    row0 = pl.cast(chunk * {tile}, pl.INDEX)
    rows = pl.min(ROWS - row0, 16)
    out, _ = pl.spmd_submit(self.grid, a, b, out, row0, rows, core_num=1)
"""
    if variant == "non-dominating":
        # The same formula outside the loop is not protected by its IV domain.
        calls += """
row0 = 16
rows = pl.min(ROWS - row0, 16)
out, _ = pl.spmd_submit(self.grid, a, b, out, row0, rows, core_num=1)
"""
    if variant == "invalid-actual":
        calls += "out, _ = pl.spmd_submit(self.grid, a, b, out, 0, 17, core_num=1)\n"
    return pl.parse_program(f"""
ROWS = pl.dynamic("{dimension}")

@pl.program
class TiledCaller:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP16], b: pl.Tensor[[32, 32], pl.FP16],
               out: pl.Out[pl.Tensor[[ROWS, 32], pl.FP32]], row0: pl.Scalar[pl.INDEX],
               rows: pl.Scalar[pl.INDEX]) -> pl.Tensor[[ROWS, 32], pl.FP32]:
        a_mat = pl.load(a, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        lhs = pl.move(a_mat, target_memory=pl.Mem.Left)
        rhs = pl.move(b_mat, target_memory=pl.Mem.Right)
        acc = pl.matmul(lhs, rhs)
        r = rows
        trim = pl.tile.set_validshape(acc, r, 32)
        return pl.store(trim, [row0, 0], out)

    @pl.function(type=pl.FunctionType.Spmd)
    def grid(self, a: pl.Tensor[[16, 32], pl.FP16], b: pl.Tensor[[32, 32], pl.FP16],
             out: pl.Out[pl.Tensor[[ROWS, 32], pl.FP32]], row0: pl.Scalar[pl.INDEX],
             rows: pl.Scalar[pl.INDEX]) -> pl.Tensor[[ROWS, 32], pl.FP32]:
        return self.kernel(a, b, out, row0, rows)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[16, 32], pl.FP16], b: pl.Tensor[[32, 32], pl.FP16],
             out: pl.InOut[pl.Tensor[[ROWS, 32], pl.FP32]]{scalar}) -> pl.Tensor[[ROWS, 32], pl.FP32]:
        with pl.manual_scope():
{indent(calls.strip(), "            ")}
        return out
""")


def test_tiled_spmd_caller_proves_acc_trim_and_preserves_store_order():
    program = lower(tiled_caller_program("bounded"), passes.MemoryPlanner.PYPTO)
    original = kernel(program)
    assert original.ir_stage == ir.FunctionIRStage.Buffer
    logical, physical = sync_pipeline(program)
    # Each host iteration launches one tile or the final partial tile. The
    # source-order interpreter receives concrete actuals and reads native
    # metadata and command operands directly.
    for batch in (1, 16, 17, 31, 32, 33):
        for start in range(0, batch, 16):
            rows = min(batch - start, 16)
            state = check_commands(original, logical, physical, {3: start, 4: rows}, output_rows=batch)
            assert [store[1:3] for store in state.stores] == [(rows, 32)]


@pytest.mark.parametrize("variant", ["non-dominating", "overflow", "invalid-actual", "shadowed-shape"])
def test_tiled_spmd_caller_requires_a_checked_window_at_every_call(variant):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        program = selection_input(tiled_caller_program(variant))
        original = kernel(program)
        selected = kernel(passes.select_buffer_ir_functions()(program))
    assert selected.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(original.body, selected.body, enable_auto_mapping=False)


def add_runtime_matrix_consumer(program):
    """Keep the original AIC function and add a native consumer of its alias."""

    class Consume(ir.IRMutator):
        matrix = None
        changed = False

        def visit_assign_stmt(self, op):
            call = op.value
            if isinstance(call, ir.Call) and call.op.name == opname("tile.matmul_acc"):
                self.matrix = call
            if (
                not self.changed
                and isinstance(call, ir.Call)
                and call.op.name == opname("tile.set_validshape")
            ):
                matrix = self.matrix
                assert matrix is not None
                consumer = ir.Call(matrix.op, [op.var, *matrix.args[1:]], matrix.type, matrix.span)
                output = ir.Var("runtime_matrix_consumer", matrix.type, matrix.span)
                self.changed = True
                return ir.SeqStmts([op, ir.AssignStmt(output, consumer, op.span)], op.span)
            return super().visit_assign_stmt(op)

    mutation = Consume()
    result = mutation.visit_program(program)
    assert mutation.changed
    return result


@pytest.mark.parametrize("variant", ["columns", "unbounded", "consumer"])
def test_unqualified_acc_consumers_keep_original_functional_body(variant):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        program = selection_input(trim_program(variant=variant))
        if variant == "consumer":
            program = add_runtime_matrix_consumer(program)
        original = kernel(program)
        before = ir.serialize(program)
        selected = kernel(passes.select_buffer_ir_functions()(program))
    assert ir.serialize(program) == before
    assert selected.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(original.body, selected.body, enable_auto_mapping=False)


def test_addressless_planner_keeps_row_trim_functional():
    program = lower(trim_program(), passes.MemoryPlanner.PTOAS)
    function = kernel(program)
    assert function.ir_stage == ir.FunctionIRStage.Functional
    assert "buffer.matmul" not in str(program)


@pytest.mark.parametrize(
    "kind,observe",
    [
        ("attribute", "original"),
        ("attribute", "trim"),
        ("aggregate", "original"),
        ("aggregate", "trim"),
        ("carry", "trim"),
    ],
)
def test_metadata_observers_and_runtime_alias_carries_refuse_transactionally(kind, observe):
    class Observe(ir.IRMutator):
        changed = False

        def visit_assign_stmt(self, op):
            rewritten = super().visit_assign_stmt(op)
            assert isinstance(rewritten, ir.AssignStmt)
            call = rewritten.value
            if self.changed or not isinstance(call, ir.Call) or call.op.name != opname("tile.set_validshape"):
                return rewritten
            operand = call.args[0] if observe == "original" else rewritten.var
            span = rewritten.span
            if kind == "attribute":
                query = ir.create_op_call("tile.get_block_idx", [], {}, span)
                value = ir.Call(query.op, [], {}, {"observed_tile": operand}, query.type, span)
                extra = ir.AssignStmt(ir.Var("observer", query.type, span), value, span)
            elif kind == "aggregate":
                captured = ir.MakeTuple([operand, ir.ConstInt(0, ir.DataType.INDEX, span)], span)
                extra = ir.AssignStmt(ir.Var("escape", captured.type, span), captured, span)
            else:
                carried = ir.IterArg("alias_carry", operand.type, operand, span)
                result = ir.Var("alias_result", operand.type, span)
                zero, one = [ir.ConstInt(x, ir.DataType.INDEX, span) for x in (0, 1)]
                extra = ir.ForStmt(
                    ir.Var("alias_visit", ir.ScalarType(ir.DataType.INDEX), span),
                    zero,
                    one,
                    one,
                    [carried],
                    ir.YieldStmt([carried], span),
                    [result],
                    span,
                )
            self.changed = True
            return ir.SeqStmts([rewritten, extra], span)

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        program = selection_input(trim_program())
        mutation = Observe()
        changed = mutation.visit_program(program)
        assert mutation.changed
        original = kernel(changed)
        selected = kernel(passes.select_buffer_ir_functions()(changed))
    assert selected.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(original.body, selected.body, enable_auto_mapping=False)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", PLANNERS)
def test_real_cpp_uses_dynamic_both_fields_without_changing_static_writer(tmp_path, planner):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    program = lower(trim_program(), planner)
    assert kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical = passes.insert_sync()(program)
        optimized = passes.optimize_sync()(logical)
    with passes.PassContext([]):
        physical = passes.lower_sync()(optimized)
    final = kernel(physical).body
    assert isinstance(final, ir.SeqStmts)
    assert isinstance(final.stmts[-1], ir.ReturnStmt)
    drain = final.stmts[-2]
    assert isinstance(drain, ir.EvalStmt) and isinstance(drain.expr, ir.Call)
    assert drain.expr.op.name == opname("system.bar_all")
    pto, cpp = tmp_path / "acc_rows.pto", tmp_path / "acc_rows.cpp"
    text = codegen.PTOCodegen().generate(physical, emit_tile_addr=True, emit_source_loc=False)
    pto.write_text(text)
    _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    generated = cpp.read_text()
    assert "pto.set_validshape" in text and "pto.tmatmul" in text and "pto.tstore" in text
    assert "set_flag(" in generated and "wait_flag(" in generated
    assert "ptoas_auto_sync_tail(" not in generated.split("AICORE void ", 1)[1]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_static_accumulator_carry_and_outer_reset_keep_full_writer(count):
    program = lower(trim_program(visits=count, replay=2, variant="carry"), passes.MemoryPlanner.PYPTO)
    original = kernel(program)
    assert original.ir_stage == ir.FunctionIRStage.Buffer
    logical, physical = sync_pipeline(program)
    for rows in (0, 1, 31, 32):
        state = check_commands(original, logical, physical, {2: rows, 3: 32 - rows})
        assert len(state.matrix_calls) == 2 * (1 + count)
        assert all(tuple(call.args[2].type.valid_shape) == (32, 64) for call in state.matrix_calls)
        assert sum(base == BASES[6] for base, *_ in state.stores) == 4


def test_empty_outer_invocation_has_no_payload_or_flag_obligation():
    program = lower(trim_program(replay=0), passes.MemoryPlanner.PYPTO)
    original = kernel(program)
    assert original.ir_stage == ir.FunctionIRStage.Functional
    calls = []
    for op in statements(program):
        expression = None
        if isinstance(op, ir.AssignStmt):
            expression = op.value
        elif isinstance(op, ir.EvalStmt):
            expression = op.expr
        if isinstance(expression, ir.Call):
            calls.append(expression.op.name)
    payloads = {
        opname(f"{namespace}.{operation}")
        for namespace in ("tile", "buffer")
        for operation in ("load", "store", "matmul", "matmul_acc")
    } | {opname("tile.move"), opname("buffer.copy")}
    commands = {opname(name) for name in ("sync.record", "sync.wait", "system.set_flag", "system.wait_flag")}
    assert not set(calls) & (payloads | commands)
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    assert kernel(restored).ir_stage == ir.FunctionIRStage.Functional
    # Deserialization creates fresh identities for harmless free storage
    # carriers retained in dead declarations; compare their consistent mapping.
    ir.assert_structural_equal(original.body, kernel(restored).body, enable_auto_mapping=True)


def test_original_shape_dimension_bounds_scalar_before_tensor_formal():
    program = trim_program()
    original = kernel(program)
    rows = original.params[2]

    class ShapeNonnegative(ir.IRMutator):
        changed = False

        def visit_max(self, op):
            if op.left.same_as(rows) and isinstance(op.right, ir.ConstInt) and op.right.value == 0:
                self.changed = True
                return op.left
            return super().visit_max(op)

    mutation = ShapeNonnegative()
    body = mutation.visit_stmt(original.body)
    assert mutation.changed
    witness = ir.Var(
        "shape_witness",
        ir.TensorType([rows, ir.ConstInt(1, ir.DataType.INDEX, original.span)], ir.DataType.FP16),
        original.span,
    )
    parameters = list(zip(original.params, original.param_directions))
    # The nonnegative shape premise belongs to the same original Var identity,
    # even when its scalar formal precedes the tensor that supplies that fact.
    changed = ir.Function(
        original.name,
        [parameters[2], witness, *parameters[:2], *parameters[3:]],
        original.return_types,
        body,
        original.span,
        original.func_type,
    )
    lowered = lower(ir.Program([changed], program.name, program.span), passes.MemoryPlanner.PYPTO)
    assert kernel(lowered).ir_stage == ir.FunctionIRStage.Buffer


def contains_runtime_acc_store(body):
    pending = [body]
    while pending:
        op = pending.pop()
        call = op.value if isinstance(op, ir.AssignStmt) else op.expr if isinstance(op, ir.EvalStmt) else None
        if (
            isinstance(call, ir.Call)
            and call.op.name == opname("buffer.store")
            and isinstance(call.args[0].type, ir.BufferType)
            and call.args[0].type.memory_space == ir.MemorySpace.Acc
            and any(n < 0 for n in call.args[0].type.valid_shape)
        ):
            return True
        if isinstance(op, ir.SeqStmts):
            pending.extend(op.stmts)
        elif isinstance(op, ir.IfStmt):
            pending.append(op.then_body)
            if op.else_body is not None:
                pending.append(op.else_body)
        elif isinstance(op, (ir.ForStmt, ir.WhileStmt)):
            pending.append(op.body)
    return False


@pytest.mark.parametrize("kind", ["addressless", "columns", "unguarded"])
def test_direct_buffer_runtime_alias_requires_complete_store_contract(kind):
    program = lower(trim_program(), passes.MemoryPlanner.PYPTO)
    original = kernel(program)
    assert original.ir_stage == ir.FunctionIRStage.Buffer

    class BreakContract(ir.IRMutator):
        changed = False

        def visit_assign_stmt(self, op):
            call = op.value
            if (
                kind == "addressless"
                and isinstance(call, ir.Call)
                and call.op.name == opname("buffer.alloc")
                and isinstance(op.var.type, ir.BufferType)
                and op.var.type.memory_space == ir.MemorySpace.Acc
                and any(n < 0 for n in op.var.type.valid_shape)
            ):
                self.changed = True
                assert len(call.args) == 2
                return ir.AssignStmt(op.var, ir.Call(call.op, call.args[:1], call.type, call.span), op.span)
            return super().visit_assign_stmt(op)

        def visit_call(self, op):
            if (
                kind == "columns"
                and op.op.name == opname("buffer.store")
                and isinstance(op.args[0].type, ir.BufferType)
                and any(n < 0 for n in op.args[0].type.valid_shape)
            ):
                extents = op.args[2]
                assert isinstance(extents, ir.MakeTuple)
                columns = ir.ConstInt(op.args[0].type.shape[1] - 1, ir.DataType.INDEX, op.span)
                args = [*op.args[:2], ir.MakeTuple([extents.elements[0], columns], op.span), *op.args[3:]]
                self.changed = True
                return ir.Call(op.op, args, op.type, op.span)
            return super().visit_call(op)

        def visit_if_stmt(self, op):
            if kind == "unguarded" and contains_runtime_acc_store(op.then_body):
                self.changed = True
                return self.visit_stmt(op.then_body)
            return super().visit_if_stmt(op)

    mutation = BreakContract()
    changed = mutation.visit_function(original)
    assert mutation.changed
    with pytest.raises(ValueError):
        testing.analyze_conservative_sync(changed)


def test_frontend_zero_row_store_gets_existing_native_nonempty_guard():
    # Tile stores with empty valid domains are no-ops. The existing lowering
    # must supply a native positive-row guard; the source need not repeat it.
    program = lower(trim_program(variant="unguarded"), passes.MemoryPlanner.PYPTO)
    original = kernel(program)
    assert original.ir_stage == ir.FunctionIRStage.Buffer
    logical, physical = sync_pipeline(program)
    for rows in (0, 1, 31, 32):
        state = check_commands(original, logical, physical, {2: rows, 3: 32 - rows})
        assert sum(base == BASES[4] for base, *_ in state.stores) == (2 if rows else 0)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", PLANNERS)
def test_default_off_delegates_same_address_acc_aliases_to_ptoas(tmp_path, planner):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    program = lower(trim_program(), planner)
    assert kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    # lower() used the default-off public pipeline. No PyPTO logical or
    # physical ownership is introduced; ordinary PTOAS insertion owns sync.
    text = codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False)
    pto, cpp = tmp_path / "delegated_acc_rows.pto", tmp_path / "delegated_acc_rows.cpp"
    pto.write_text(text)
    _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    assert "pto.set_validshape" in text and "pto.tmatmul" in text and "pto.tstore" in text
    assert cpp.stat().st_size > 0
