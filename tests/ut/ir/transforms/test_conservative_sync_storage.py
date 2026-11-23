# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Storage-domain refinement is independent of structured may-dataflow."""

from itertools import product

import pypto.language as pl
import pytest
from pypto import ir, passes, testing
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

from .sync_closed_command_oracle import ClosedCommands, check_closed_commands, opname
from .sync_logical_oracle import value
from .sync_matmul_oracle import hardware_reference
from .sync_physical_oracle import checked_payload_calls
from .test_closed_prefix_sync import lower
from .test_conservative_sync_dataflow import conservative_fixture
from .test_implicit_sync_effects import independent_broadcasts
from .test_logical_sync_realization import verify
from .test_vec_sync_analysis import vec_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def fixture(addresses=(0, 128, 64), *, unknown=False, same_gm_root=False, shared_gm_allocation=False):
    def ci(number):
        return ir.ConstInt(number, ir.DataType.INDEX, SPAN)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], SPAN)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), SPAN)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), SPAN)

    shared = (
        ir.MemRef(ir.Var("shared_gm_base", ir.PtrType(), SPAN), 0, 64, SPAN) if shared_gm_allocation else None
    )
    gm = ir.TensorType([2, 8], ir.DataType.FP32, shared)
    a, b, out = [ir.Var(name, gm, SPAN) for name in ("a", "b", "out")]
    local = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    x, z, y = [ir.Var(name, local, SPAN) for name in ("x", "z", "y")]
    definitions = [
        ir.AssignStmt(v, call("buffer.alloc", [tup([]), ci(address)], local), SPAN)
        for v, address in zip((x, z, y), addresses, strict=True)
    ]
    if unknown:
        definitions[-1] = ir.AssignStmt(y, call("buffer.alloc", [tup([])], local), SPAN)
    return ir.Function(
        "storage_domains",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                *definitions,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.abs", [x, z]),
                payload("buffer.load", [b, tup([0, 0]), tup([2, 8]), y]),
                payload("buffer.store", [z, tup([0, 0]), tup([2, 8]), a if same_gm_root else out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


def analyze(function, disjoint=False):
    with passes.PassContext([], autosync_disjoint_gm=disjoint):
        return testing.analyze_conservative_sync(function)


def domains_for(summary, space):
    return [domain for domain in summary["storage_domains"] if domain["space"] == space]


@pytest.mark.parametrize("address,conflict", [(64, False), (96, True), (128, True), (192, False)])
def test_half_open_local_enclosures_preserve_only_possible_overlap(address, conflict):
    original = fixture((0, 128, address))
    verify(original)
    summary = analyze(original)
    # The earlier V write occupies bytes [128,192); a new MTE2 destination
    # overlaps exactly when its independent 64-byte enclosure intersects it.
    assert (ir.PipeType.V in summary["sites"][2]["source_pipes"]) == conflict
    assert all(not domain["exact_selector"] for domain in summary["storage_domains"])


def test_transitive_enclosure_overlap_is_coarsened_soundly():
    summary = analyze(fixture((0, 64, 32)))
    local = domains_for(summary, "Vec")
    assert len(local) == 1 and tuple(local[0]["enclosure"]) == (0, 128)
    assert ir.PipeType.V in summary["sites"][2]["source_pipes"]


def test_unknown_local_geometry_remains_a_space_wildcard():
    summary = analyze(fixture(unknown=True))
    assert any(
        effect["space"] == "Vec" and effect["domain"] == 0 for effect in summary["sites"][2]["effects"]
    )
    assert ir.PipeType.V in summary["sites"][2]["source_pipes"]


@pytest.mark.parametrize("disjoint", [False, True])
@pytest.mark.parametrize("same_root", [False, True])
def test_gm_root_independence_requires_the_supplied_contract(disjoint, same_root):
    summary = analyze(fixture(same_gm_root=same_root), disjoint)
    # Local source z is separate from both MTE2 destinations. Thus the MTE2
    # prerequisite at the final store is solely the GM read/write possibility.
    # When disjoint=True and same_root=True, Site 1 (MTE2->V) already completed
    # Site 0's read of GM `a` and Site 2 only touches disjoint GM `b`, so V->MTE3
    # transitively covers Site 0 without re-emitting MTE2->MTE3.
    assert (ir.PipeType.MTE2 in summary["sites"][3]["required_source_pipes"]) == (not disjoint or same_root)
    assert (ir.PipeType.MTE2 in summary["sites"][3]["source_pipes"]) == (not disjoint)
    assert bool(domains_for(summary, "DDR")) == disjoint


def test_static_storage_classification_does_not_change_original_input():
    original = fixture()
    before = ir.serialize(original)
    first, second = analyze(original), analyze(original)
    assert ir.serialize(original) == before
    assert first["storage_domains"] == second["storage_domains"]
    assert first["cost"]["conflict_queries"] > 0


def test_distinct_gm_parameters_with_shared_allocation_do_not_become_disjoint():
    summary = analyze(fixture(shared_gm_allocation=True), disjoint=True)
    assert len(domains_for(summary, "DDR")) == 1
    assert ir.PipeType.MTE2 in summary["sites"][3]["source_pipes"]


def test_selected_native_workspace_effects_share_the_ordinary_class_universe():
    function, _, _ = independent_broadcasts(("sub", "div"))
    summary = analyze(function)
    scratch = next(
        domain
        for domain in domains_for(summary, "Vec")
        if tuple(domain["enclosure"]) == (184 * 1024, 192 * 1024)
    )
    computations = [
        site
        for site in summary["sites"]
        if site["operation"].op.name
        in {
            ir.get_op("buffer.row_expand_sub").name,
            ir.get_op("buffer.row_expand_div").name,
        }
    ]
    assert len(computations) == 2
    for site in computations:
        assert any(effect["domain"] == scratch["id"] for effect in site["effects"])
    assert ir.PipeType.V in computations[1]["source_pipes"]


def many_independent_windows(count):
    def ci(number):
        return ir.ConstInt(number, ir.DataType.INDEX, SPAN)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], SPAN)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), SPAN)

    gm = ir.Var("source", ir.TensorType([2, 8], ir.DataType.FP32), SPAN)
    local = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    statements = []
    for index in range(count):
        source, target = [ir.Var(f"{name}_{index}", local, SPAN) for name in ("x", "z")]
        for variable, address in ((source, 128 * index), (target, 128 * index + 64)):
            statements.append(
                ir.AssignStmt(variable, call("buffer.alloc", [tup([]), ci(address)], local), SPAN)
            )
        statements.append(ir.EvalStmt(call("buffer.load", [gm, tup([0, 0]), tup([2, 8]), source]), SPAN))
        statements.append(ir.EvalStmt(call("buffer.abs", [source, target]), SPAN))
    statements.append(ir.ReturnStmt([], SPAN))
    return ir.Function(
        "independent_windows",
        [gm],
        [],
        ir.SeqStmts(statements, SPAN),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("count", [0, 1, 8, 32])
def test_class_recovery_and_queries_charge_static_effects_without_pairwise_geometry(count):
    summary = analyze(many_independent_windows(count))
    assert len(summary["sites"]) == 2 * count
    assert len(domains_for(summary, "Vec")) == 2 * count
    assert summary["cost"]["enclosure_lookups"] == 3 * count
    assert summary["cost"]["conflict_queries"] == 4 * count
    for index, site in enumerate(summary["sites"]):
        assert set(site["source_pipes"]) == ({ir.PipeType.MTE2} if index % 2 else set())
    # No-KILL history is explicitly allowed to depend on J growing with the
    # class universe. These assertions concern recovery/query work, not a
    # false global-linear retained-state bound.


def clone_with_body(function, body):
    return ir.Function(
        function.name,
        list(zip(function.params, function.param_directions, strict=True)),
        function.return_types,
        body,
        function.span,
        type=function.func_type,
        ir_stage=function.ir_stage,
    )


def test_plain_tensor_alias_preserves_its_established_original_root():
    original = fixture()
    out = original.params[2]
    alias = ir.Var("output_alias", out.type, SPAN)
    assert isinstance(original.body, ir.SeqStmts)
    statements = list(original.body.stmts)
    store = statements[-2]
    assert isinstance(store, ir.EvalStmt) and isinstance(store.expr, ir.Call)
    rewritten = ir.Call(store.expr.op, [*store.expr.args[:3], alias], store.expr.type, SPAN)
    body = ir.SeqStmts(
        [
            *statements[:-2],
            ir.AssignStmt(alias, out, SPAN),
            ir.EvalStmt(rewritten, SPAN),
            statements[-1],
        ],
        SPAN,
    )
    summary = analyze(clone_with_body(original, body), disjoint=True)
    assert ir.PipeType.MTE2 not in summary["sites"][-1]["source_pipes"]
    assert all(effect["domain"] != 0 for effect in summary["sites"][-1]["effects"])


@pytest.mark.parametrize("kind", ["carried", "branch"])
def test_unqualified_tensor_result_origins_remain_gm_wildcards(kind):
    original = fixture()
    out = original.params[2]
    assert isinstance(original.body, ir.SeqStmts)
    statements = list(original.body.stmts)
    store = statements[-2]
    assert isinstance(store, ir.EvalStmt) and isinstance(store.expr, ir.Call)
    if kind == "carried":
        target = ir.IterArg("carried_output", out.type, out, SPAN)
        final = ir.Var("final_output", out.type, SPAN)
        rewritten = ir.Call(store.expr.op, [*store.expr.args[:3], target], store.expr.type, SPAN)
        loop_body = ir.SeqStmts(
            [
                ir.EvalStmt(rewritten, SPAN),
                ir.YieldStmt([target], SPAN),
            ],
            SPAN,
        )
        zero, one = [ir.ConstInt(i, ir.DataType.INDEX, SPAN) for i in (0, 1)]
        wrapper = ir.ForStmt(
            ir.Var("visit", ir.ScalarType(ir.DataType.INDEX), SPAN),
            zero,
            one,
            one,
            [target],
            loop_body,
            [final],
            SPAN,
        )
        ending = [wrapper, statements[-1]]
    else:
        target = ir.Var("selected_output", out.type, SPAN)
        branch = ir.IfStmt(
            ir.ConstBool(True, SPAN),
            ir.YieldStmt([out], SPAN),
            ir.YieldStmt([out], SPAN),
            [target],
            SPAN,
        )
        rewritten = ir.Call(store.expr.op, [*store.expr.args[:3], target], store.expr.type, SPAN)
        ending = [branch, ir.EvalStmt(rewritten, SPAN), statements[-1]]
    candidate = clone_with_body(original, ir.SeqStmts([*statements[:-2], *ending], SPAN))
    verify(candidate)
    summary = analyze(candidate, disjoint=True)
    final_store = summary["sites"][-1]
    assert any(effect["space"] == "DDR" and effect["domain"] == 0 for effect in final_store["effects"])
    assert ir.PipeType.MTE2 in final_store["source_pipes"]


def test_explicit_scratch_alias_and_implicit_effect_use_one_physical_domain():
    function, _, _ = independent_broadcasts(("sub", "div"), explicit_alias=True)
    summary = analyze(function)
    scratch = next(
        domain
        for domain in domains_for(summary, "Vec")
        if tuple(domain["enclosure"]) == (184 * 1024, 192 * 1024)
    )
    effects = [
        (site["pipe"], effect["write"])
        for site in summary["sites"]
        for effect in site["effects"]
        if effect["domain"] == scratch["id"]
    ]
    assert (ir.PipeType.MTE2, True) in effects
    assert (ir.PipeType.V, True) in effects


@pytest.mark.parametrize("disjoint", [False, True])
def test_refined_plan_preserves_actual_payload_order_coverage_and_physical_reuse(disjoint):
    original = conservative_fixture()
    with passes.PassContext([], autosync_disjoint_gm=disjoint):
        logical = testing.insert_conservative_prefixes(original)
        physical = lower(logical, eligible=list(range(6)))
    for visits, count, choice in product((0, 1, 2), (0, 1, 3), (0, 1)):
        parameters = {3: count, 4: visits, 5: choice}
        baseline = ClosedCommands(original, parameters, range(6))
        baseline.run()
        actual, _ = check_closed_commands(logical, physical, parameters, range(6), compare_commands=False)
        assert baseline.payloads == actual.payloads


def test_registered_local_alias_retains_the_parent_physical_domain():
    original = fixture()
    assert isinstance(original.body, ir.SeqStmts)
    statements = list(original.body.stmts)
    first_allocation = statements[0]
    assert isinstance(first_allocation, ir.AssignStmt)
    source = first_allocation.var
    alias = ir.Var("physical_alias", source.type, SPAN)
    offsets = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, SPAN)] * 2, SPAN)
    alias_call = ir.Call(ir.get_op("buffer.subview"), [source, offsets], source.type, SPAN)
    load = statements[-3]
    assert isinstance(load, ir.EvalStmt) and isinstance(load.expr, ir.Call)
    replacement = ir.EvalStmt(
        ir.Call(load.expr.op, [*load.expr.args[:3], alias], load.expr.type, SPAN),
        SPAN,
    )
    body = ir.SeqStmts(
        [
            *statements[:-3],
            ir.AssignStmt(alias, alias_call, SPAN),
            replacement,
            *statements[-2:],
        ],
        SPAN,
    )
    candidate = clone_with_body(original, body)
    verify(candidate)
    summary = analyze(candidate)
    assert ir.PipeType.V in summary["sites"][2]["source_pipes"]
    first_local = {effect["domain"] for effect in summary["sites"][0]["effects"] if effect["space"] == "Vec"}
    alias_local = {effect["domain"] for effect in summary["sites"][2]["effects"] if effect["space"] == "Vec"}
    assert first_local == alias_local and 0 not in alias_local


class _RowSubviewCommands(ClosedCommands):
    def assign(self, statement, env):
        call = statement.value
        if not isinstance(call, ir.Call) or call.op.name != opname("buffer.subview"):
            super().assign(statement, env)
            return

        parent, offsets = call.args[:2]
        space, base, parent_rows, parent_cols, element_bytes = self.storage[parent]
        assert isinstance(offsets, ir.MakeTuple)
        row, col = (value(offset, env) for offset in offsets.elements)
        rows, cols = statement.var.type.shape
        assert isinstance(row, int)
        assert col == 0 and cols == parent_cols
        assert 0 <= row <= parent_rows - rows
        self.storage[statement.var] = (
            space,
            base + row * parent_cols * element_bytes,
            rows,
            cols,
            element_bytes,
        )
        self.layouts[statement.var] = statement.var.type


@pytest.mark.parametrize("view", ["direct", "nested_store", "nested_write", "disjoint"])
def test_affine_row_slice_uses_physical_origin_for_store_before_scalar_write(view: str) -> None:
    # Store odd rows. The nested view reaches those same physical rows through
    # a one-row shift, while the disjoint variant writes even rows.
    direct = "pl.tile.slice(root, [1, 8], [2 * i + 1, 0])"
    nested = "pl.tile.slice(window, [1, 8], [2 * i, 0])"
    stored = nested if view == "nested_store" else direct
    overwritten = nested if view == "nested_write" else direct
    if view == "disjoint":
        overwritten = "pl.tile.slice(root, [1, 8], [2 * i, 0])"
    program = pl.parse_program(f"""
@pl.program
class StoreBeforeScalarWrite:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[16, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
        root = pl.load(x, [0, 0], [16, 8])
        replacement = pl.const(7.0, pl.FP32)
        for i in pl.range(8):
            window = pl.tile.slice(root, [15, 8], [1, 0])
            stored = {stored}
            out = pl.store(stored, [i, 0], out)
            overwritten = {overwritten}
            pl.tile.write(overwritten, [0, 0], replacement)
        return out
""")
    placed = []

    def capture(transform, before):
        if transform.get_name() == "RealizeSync":
            placed.append(before.get_function("kernel"))

    with passes.PassContext(
        [passes.CallbackInstrument(before_pass=capture, name="capture_placed_fibers")],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
    ):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        (original,) = placed
        assert original.ir_stage == ir.FunctionIRStage.Buffer, dict(original.attrs)
        summary = testing.analyze_conservative_sync(original)
    function = lowered.get_function("kernel")
    assert function is not None and function.attrs[PYPTO_AUTOSYNC_ATTR]
    execution = _RowSubviewCommands(function, {}, range(8))
    order = execution.run()
    conflicts = set()
    for store_index, (_, pipe, reads, _) in enumerate(execution.payloads):
        if pipe != ir.PipeType.MTE3:
            continue
        for write_index, (_, later_pipe, _, writes) in enumerate(
            execution.payloads[store_index + 1 :], start=store_index + 1
        ):
            if later_pipe == ir.PipeType.S and reads & writes:
                conflicts.add((("payload", store_index, "C"), ("payload", write_index, "I")))
    assert len(conflicts) == (0 if view == "disjoint" else 8)
    unordered = conflicts - order
    assert not unordered, f"store completion must precede scalar overwrite issue: {unordered}"
    (write,) = [site for site in summary["sites"] if site["pipe"] == ir.PipeType.S]
    assert (ir.PipeType.MTE3 in write["required_source_pipes"]) == (view != "disjoint")


@pytest.mark.parametrize(
    ("rows", "trips", "reverse", "shared_elements"),
    [
        (32, 2, False, 512),
        (48, 2, False, 0),
        (64, 2, False, 256),
        (80, 2, False, 0),
        (96, 2, False, 0),
        (48, 2, True, 0),
        (192, 8, False, 0),
        (208, 9, False, 0),
    ],
)
def test_affine_matrix_slices_depend_on_the_native_layout(
    rows: int, trips: int, reverse: bool, shared_elements: int
) -> None:
    # Memory reuse places source and replacement at the same Mat address. A
    # 64-row NZ tile's bottom rows share a fractal with the 32-row tile's top
    # rows because their column-block strides differ. Row indices alone would
    # declare those accesses disjoint and let FIX overwrite an active MTE1 read.
    height = trips * 16
    # Reverse the direction and use a nonzero loop origin in the control.
    start = 3 if reverse else 0
    row = f"16 * ({start + trips - 1} - i)" if reverse else "16 * i"
    program = pl.parse_program(f"""
@pl.program
class MatrixFiberReuse:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[{rows}, 32], pl.FP16],
               seed_a: pl.Tensor[[16, 32], pl.FP16],
               seed_b: pl.Tensor[[32, 32], pl.FP16],
               out: pl.Out[pl.Tensor[[{height}, 16], pl.FP32]]) -> pl.Tensor[[{height}, 16], pl.FP32]:
        seed_l1 = pl.tile.load(seed_a, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        seed_r1 = pl.tile.load(seed_b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        seed_l0 = pl.tile.move(seed_l1, target_memory=pl.Mem.Left)
        seed_r0 = pl.tile.move(seed_r1, target_memory=pl.Mem.Right)
        seed = pl.tile.matmul(seed_l0, seed_r0)
        for i in pl.range({start}, {start + trips}):
            source = pl.tile.load(a, [0, 0], [{rows}, 32], target_memory=pl.Mem.Mat)
            left = pl.tile.extract(source, {rows - height} + {row}, 0, [16, 32], target_memory=pl.Mem.Left)
            replacement = pl.tile.create([{height}, 32], pl.FP16, target_memory=pl.Mem.Mat)
            replacement = pl.tile.assemble(replacement, seed, [{row}, 0])
            transposed = pl.tile.transpose_view(replacement)
            right = pl.tile.extract(transposed, 0, {row}, [32, 16], target_memory=pl.Mem.Right)
            result = pl.tile.matmul(left, right)
            out = pl.tile.store(result, [{row}, 0], out)
        return out
""")
    placed = []

    def capture(transform, before):
        if transform.get_name() == "RealizeSync":
            placed.append(before.get_function("kernel"))

    with passes.PassContext(
        [passes.CallbackInstrument(before_pass=capture, name="capture_matrix_fibers")],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
    ):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    (original,) = placed
    function = lowered.get_function("kernel")
    assert function is not None and function.attrs[PYPTO_AUTOSYNC_ATTR]
    calls, emitted_calls = checked_payload_calls(original, function)

    class MatrixCommands(ClosedCommands):
        def assign(self, statement, env):
            if isinstance(statement.var.type, ir.ScalarType):
                super().assign(statement, env)

        def memory_footprint(self, call, env):
            # The geometry interpreter above has already compared the source
            # and emitted operands, including the NZ/ZN descriptor alias.
            index = len(self.payloads)
            assert emitted_calls[index][0].op.name == call.op.name
            _, _, pipe, reads, writes = calls[index]
            return pipe, reads, writes, call.op.name, None

    execution = MatrixCommands(function, {}, range(8))
    order = execution.run()
    assert len(execution.payloads) == len(calls)
    checked = 0
    for index, (read, write) in enumerate(zip(calls[:-1], calls[1:], strict=True)):
        if read[0].op.name != opname("buffer.extract") or write[0].op.name != opname("buffer.insert"):
            continue
        assert read[2] == ir.PipeType.MTE1 and write[2] == ir.PipeType.FIX
        shared = {address for space, address in read[3] & write[4] if space == ir.MemorySpace.Mat}
        assert len(shared) == shared_elements
        edge = (("payload", index, "C"), ("payload", index + 1, "I"))
        if shared or trips <= 8:
            assert (edge in order) == bool(shared)
        checked += 1
    assert checked == trips
    # Check every required completion-to-issue edge from the original payload
    # geometry, including prior visits and accesses outside Mat. The command
    # interpreter separately checks flag closure and consumption before reuse.
    required = hardware_reference(original, calls)[2]
    for (producer, completed), (consumer, issued_as_completion) in required:
        if completed and not issued_as_completion:
            assert (("payload", producer, "C"), ("payload", consumer, "I")) in order

    if rows == 96:
        summary = testing.analyze_conservative_sync(original)
        sites = summary["sites"]
        insert_op = opname("buffer.insert")
        insert = next(i for i, site in enumerate(sites) if site["operation"].op.name == insert_op)

        def mat_domains(site):
            return {effect["domain"] for effect in site["effects"] if effect["space"] == "Mat"}

        cover, read, write = (mat_domains(site) for site in sites[insert - 2 : insert + 1])
        assert read and write and read.isdisjoint(write)
        assert cover >= read | write
        assert sites[insert]["refreshed_domains"] == []


@pytest.mark.parametrize("count", [8, 32, 128])
def test_flat_production_state_avoids_copied_prefix_histories(count):
    function = many_independent_windows(count)
    production = testing.analyze_conservative_sync(function, retain_incoming_diagnostics=False)
    diagnostic = testing.analyze_conservative_sync(function, retain_incoming_diagnostics=True)
    assert not production["incoming_diagnostics_available"]
    assert all(site["incoming"] is None for site in production["sites"])
    assert diagnostic["incoming_diagnostics_available"]
    assert production["cost"]["incoming_diagnostic_facts"] == 0
    assert diagnostic["cost"]["incoming_diagnostic_facts"] > count * count
    assert production["cost"]["retained_facts"] <= 8 * count
    assert production["cost"]["conflict_summary_facts"] <= 4 * count
    assert production["cost"]["control_snapshot_facts"] == 0
    assert production["cost"]["control_merge_facts"] == 0
    assert [site["source_pipes"] for site in production["sites"]] == [
        site["source_pipes"] for site in diagnostic["sites"]
    ]
    assert production["exit_effects"] is None


@pytest.mark.parametrize("disjoint", [False, True])
def test_alias_premise_is_retained_even_when_no_domain_needs_it(disjoint):
    function = many_independent_windows(0)
    summary = analyze(function, disjoint)
    assert not summary["storage_domains"]
    assert summary["disjoint_gm"] == disjoint
    assert summary["gm_alias_contract"] == ("supplied_disjoint_roots" if disjoint else "may_alias")
    # Leaving/replacing PassContext cannot change the published premise.
    with passes.PassContext([], autosync_disjoint_gm=not disjoint):
        assert summary["disjoint_gm"] == disjoint


def test_repeated_native_fixed_claims_remain_reserved_after_storage_refinement():
    original = vec_fixture(count=3, visits=2, operation="buffer.recip")
    with passes.PassContext([], autosync_disjoint_gm=True):
        logical = testing.insert_conservative_prefixes(original)
        result = testing.lower_sync_schedule(logical, eligible_ids=list(range(6)))
    assert result["physical_cases"]
    for case in result["physical_cases"]:
        assert set(case["native_reserved_ids"]) == {
            (ir.PipeType.V, ir.PipeType.S, 0),
            (ir.PipeType.S, ir.PipeType.V, 0),
        }
    check_closed_commands(logical, result["function"], {}, range(6), compare_commands=False)


def test_certified_full_overwrite_kill_and_unsafe_refusals():
    def ci(number):
        return ir.ConstInt(number, ir.DataType.INDEX, SPAN)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], SPAN)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), SPAN)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), SPAN)

    gm = ir.TensorType([2, 8], ir.DataType.FP32)
    a, out = ir.Var("a", gm, SPAN), ir.Var("out", gm, SPAN)
    ub2d = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    x = ir.Var("x", ub2d, SPAN)
    y = ir.Var("y", ub2d, SPAN)
    alloc_x = ir.AssignStmt(x, call("buffer.alloc", [tup([]), ci(0)], ub2d), SPAN)
    alloc_y = ir.AssignStmt(y, call("buffer.alloc", [tup([]), ci(128)], ub2d), SPAN)

    # Full overwrite of x on V (`buffer.abs(y, x)`) after MTE2 load kills MTE2 obligation for later store.
    full_fn = ir.Function(
        "full_kill",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                alloc_x,
                alloc_y,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.abs", [y, x]),
                payload("buffer.store", [x, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        with_kill = testing.analyze_conservative_sync(full_fn, enable_refresh_kill=True)
        no_kill = testing.analyze_conservative_sync(full_fn, enable_refresh_kill=False)
    assert with_kill["precision"] == "storage_domain_refresh_kill"
    assert no_kill["precision"] == "storage_domain_no_kill"
    assert with_kill["cost"]["killed_facts"] > 0
    assert no_kill["cost"]["killed_facts"] == 0
    assert len(with_kill["sites"][1]["refreshed_domains"]) == 1
    assert ir.PipeType.MTE2 in no_kill["sites"][2]["required_source_pipes"]
    assert ir.PipeType.MTE2 not in with_kill["sites"][2]["required_source_pipes"]

    # Conditional full overwrite without else must NOT kill MTE2 before post-branch store.
    cond_fn = ir.Function(
        "cond_no_kill",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                alloc_x,
                alloc_y,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                ir.IfStmt(ir.ConstBool(True, SPAN), payload("buffer.abs", [y, x]), None, [], SPAN),
                payload("buffer.store", [x, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        cond_summary = testing.analyze_conservative_sync(cond_fn, enable_refresh_kill=True)
    assert ir.PipeType.MTE2 in cond_summary["sites"][2]["required_source_pipes"]

    # Subview partial overwrite must refuse KILL.
    sub_type = ir.BufferType([1, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    sx = ir.Var("sx", sub_type, SPAN)
    sub_src = ir.Var("sub_src", sub_type, SPAN)
    sub_fn = ir.Function(
        "subview_no_kill",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                alloc_x,
                ir.AssignStmt(sx, call("buffer.subview", [x, tup([0, 0])], sub_type), SPAN),
                ir.AssignStmt(sub_src, call("buffer.alloc", [tup([]), ci(256)], sub_type), SPAN),
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.abs", [sub_src, sx]),
                payload("buffer.store", [x, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        sub_summary = testing.analyze_conservative_sync(sub_fn, enable_refresh_kill=True)
    assert not sub_summary["sites"][1]["refreshed_domains"]
    assert ir.PipeType.MTE2 in sub_summary["sites"][2]["required_source_pipes"]

    # set_validshape / dynamic valid_shape view must refuse KILL.
    dyn_ub = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec, valid_shape=[-1, -1])
    vx = ir.Var("vx", dyn_ub, SPAN)
    vy = ir.Var("vy", dyn_ub, SPAN)
    alloc_vx = ir.AssignStmt(vx, call("buffer.alloc", [tup([1, 8]), ci(0)], dyn_ub), SPAN)
    alloc_vy = ir.AssignStmt(vy, call("buffer.alloc", [tup([1, 8]), ci(128)], dyn_ub), SPAN)
    valid_fn = ir.Function(
        "set_validshape_no_kill",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                alloc_vx,
                alloc_vy,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), vx]),
                payload("buffer.set_validshape", [vx, tup([2, 8])]),
                payload("buffer.abs", [vy, vx]),
                payload("buffer.store", [vx, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        valid_summary = testing.analyze_conservative_sync(valid_fn, enable_refresh_kill=True)
    assert not valid_summary["sites"][1]["refreshed_domains"]
    assert ir.PipeType.MTE2 in valid_summary["sites"][2]["required_source_pipes"]

    # Scalar point store (`buffer.write`) must refuse KILL.
    scalar_fn = ir.Function(
        "scalar_write_no_kill",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                alloc_x,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.write", [x, tup([0, 0]), ir.ConstFloat(1.0, ir.DataType.FP32, SPAN)]),
                payload("buffer.store", [x, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        scalar_summary = testing.analyze_conservative_sync(scalar_fn, enable_refresh_kill=True)
    assert not scalar_summary["sites"][1]["refreshed_domains"]
    assert ir.PipeType.MTE2 in scalar_summary["sites"][2]["required_source_pipes"]


def test_disjoint_domain_delivery_and_loop_invariant_pruning():
    def ci(number):
        return ir.ConstInt(number, ir.DataType.INDEX, SPAN)

    def tup(values):
        return ir.MakeTuple([ci(v) for v in values], SPAN)

    def call(name, args, result=None):
        return ir.Call(ir.get_op(name), args, result or ir.VoidType(), SPAN)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), SPAN)

    gm = ir.TensorType([2, 8], ir.DataType.FP32)
    a, b, out = ir.Var("a", gm, SPAN), ir.Var("b", gm, SPAN), ir.Var("out", gm, SPAN)
    ub = ir.BufferType([2, 8], ir.DataType.FP32, ir.MemorySpace.Vec)
    x, y, z1, z2 = [ir.Var(n, ub, SPAN) for n in ("x", "y", "z1", "z2")]
    allocs = [
        ir.AssignStmt(v, call("buffer.alloc", [tup([]), ci(addr)], ub), SPAN)
        for v, addr in ((x, 0), (y, 128), (z1, 256), (z2, 384))
    ]

    # Intervening MTE2 load to disjoint `y`/`b` must not wipe already-delivered MTE2->V completion on `x`.
    disjoint_fn = ir.Function(
        "disjoint_delivery",
        [a, b, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                *allocs,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.abs", [x, z1]),
                payload("buffer.load", [b, tup([0, 0]), tup([2, 8]), y]),
                payload("buffer.abs", [x, z2]),
                payload("buffer.store", [z2, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        summary = testing.analyze_conservative_sync(disjoint_fn)
    assert ir.PipeType.MTE2 in summary["sites"][3]["required_source_pipes"]
    assert ir.PipeType.MTE2 not in summary["sites"][3]["source_pipes"]
    assert summary["cost"]["suppressed_services"] >= 1
    assert summary["cost"]["certificate_omissions"] == summary["cost"]["suppressed_services"]

    # Loop-invariant delivery: x is synced MTE2->V before the loop and only read inside the loop.
    idx = ir.Var("i", ir.ScalarType(ir.DataType.INDEX), SPAN)
    loop_clean = ir.Function(
        "loop_invariant_delivery",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                *allocs,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.abs", [x, z1]),
                ir.ForStmt(
                    idx,
                    ci(0),
                    ci(4),
                    ci(1),
                    [],
                    ir.SeqStmts([payload("buffer.abs", [x, z2]), ir.YieldStmt([], SPAN)], SPAN),
                    [],
                    SPAN,
                ),
                payload("buffer.store", [z2, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        clean_summary = testing.analyze_conservative_sync(loop_clean)
    assert ir.PipeType.MTE2 in clean_summary["sites"][2]["required_source_pipes"]
    assert ir.PipeType.MTE2 not in clean_summary["sites"][2]["source_pipes"]
    assert clean_summary["cost"]["suppressed_services"] >= 1
    assert clean_summary["cost"]["certificate_omissions"] == clean_summary["cost"]["suppressed_services"]

    # Back-edge re-dirtied loop: x is overwritten on MTE2 after loop consumer, forcing MTE2->V to remain.
    loop_dirty = ir.Function(
        "loop_dirty_delivery",
        [a, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(
            [
                *allocs,
                payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                payload("buffer.abs", [x, z1]),
                ir.ForStmt(
                    idx,
                    ci(0),
                    ci(4),
                    ci(1),
                    [],
                    ir.SeqStmts(
                        [
                            payload("buffer.abs", [x, z2]),
                            payload("buffer.load", [a, tup([0, 0]), tup([2, 8]), x]),
                            ir.YieldStmt([], SPAN),
                        ],
                        SPAN,
                    ),
                    [],
                    SPAN,
                ),
                payload("buffer.store", [z2, tup([0, 0]), tup([2, 8]), out]),
                ir.ReturnStmt([out], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        type=ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    with passes.PassContext([], autosync_disjoint_gm=True):
        dirty_summary = testing.analyze_conservative_sync(loop_dirty)
    assert ir.PipeType.MTE2 in dirty_summary["sites"][2]["required_source_pipes"]
    assert ir.PipeType.MTE2 in dirty_summary["sites"][2]["source_pipes"]
    assert dirty_summary["cost"]["certificate_omissions"] == dirty_summary["cost"]["suppressed_services"]
