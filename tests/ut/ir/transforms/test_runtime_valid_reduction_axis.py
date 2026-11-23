# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original valid-axis identities survive placed contiguous reinterpretations."""

import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto import language as pl
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import PassManager

from .test_runtime_valid_broadcast import check_runtime_commands, lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


def axis_program(reduction="row_sum", visits=3, replay=2, mixed=False, varying=False):
    operation = "pl.add(pl.tile.full([1,8], value=0.0, dtype=pl.FP32), row)" if mixed else "pl.add(row, 1.0)"
    active = "pl.min(rows, visit % 8)" if varying else "rows"
    return pl.parse_program(f"""
@pl.program
class RuntimeAxis:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,16], pl.FP32],
               rows: pl.Scalar[pl.INDEX], choice: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[8,16], pl.FP32]]) -> pl.Tensor[[8,16], pl.FP32]:
        for outer in pl.range({replay}):
            for visit in pl.range({visits}):
                if visit % 2 == choice:
                    active = {active}
                    source = pl.load(x, [0,0], [8,16], valid_shape=[active,16])
                    workspace = pl.create_tile([8,16], dtype=pl.FP32)
                    column = pl.{reduction}(pl.mul(source, source), workspace)
                    row = pl.reshape(column, [1,8])
                    shifted = {operation}
                    carrier = pl.reshape(shifted, [8,1])
                    result = pl.row_expand_mul(source, carrier)
                    out = pl.store(result, [0,0], out)
        return out
""")


@pytest.mark.parametrize("reduction", ["row_sum", "row_max"])
@pytest.mark.parametrize("visits,replay", [(1, 1), (3, 2)])
def test_runtime_axis_metadata_bytes_matching_and_reuse(reduction, visits, replay, tmp_path):
    program, original = lower(axis_program(reduction, visits, replay))
    summary = testing.analyze_conservative_sync(original)
    assert not summary["exact_storage_selectors"]
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 8
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [p.get_function("kernel") for p in [logical_program, physical_program]]
    assert logical is not None and physical is not None
    for rows in [0, 1, 7, 8]:
        for choice in [0, 1]:
            output = torch.full_like(x, -777)
            namespace["kernel"](x, rows, choice, output)
            expected = torch.full_like(x, -777)
            if visits > choice:
                squares = x[:rows] * x[:rows]
                reduced = (
                    squares.sum(dim=1, keepdim=True)
                    if reduction == "row_sum"
                    else squares.amax(dim=1, keepdim=True)
                )
                expected[:rows] = x[:rows] * (reduced + 1)
            torch.testing.assert_close(output, expected)
            check_runtime_commands(logical, physical, {1: rows, 2: choice})
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "axis.pto", tmp_path / "axis.cpp"
        pto.write_text(codegen.PTOCodegen().generate(physical_program))
        _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2"])
        assert cpp.is_file()


def test_mixed_full_and_partial_add_qualifies_only_observed_prefix():
    # The full zero operand has values everywhere, but the reduction only
    # defines its prefix. The downstream matrix observes precisely that prefix.
    program, function = lower(axis_program(mixed=True))
    assert function.ir_stage == ir.FunctionIRStage.Buffer
    assert "buffer.set_validshape" in str(program)


def test_runtime_axis_retained_size_does_not_unfold_trips():
    summaries = []
    for count in [3, 1023, 1000000]:
        _, function = lower(axis_program(visits=count))
        summaries.append(testing.analyze_conservative_sync(function))
    assert all(summary["cost"] == summaries[0]["cost"] for summary in summaries)
    assert all(len(summary["sites"]) == len(summaries[0]["sites"]) for summary in summaries)


def test_runtime_axis_requires_placed_native_metadata_objects():
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PTOAS):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        program = manager.run_passes(axis_program())
    function = program.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional
    assert function.attrs["buffer_ir_selected"] is False


def test_reshape_requested_shape_must_match_the_declared_axis_map():
    class ForgeShape(ir.IRMutator):
        changed = False

        def visit_call(self, op):
            result = super().visit_call(op)
            assert isinstance(result, ir.Call)
            if self.changed or result.op.name != ir.get_op("tile.reshape").name:
                return result
            self.changed = True
            shape = ir.MakeTuple(
                [ir.ConstInt(n, ir.DataType.INDEX, result.span) for n in [2, 4]], result.span
            )
            return ir.Call(
                result.op,
                [result.args[0], shape],
                dict(result.kwargs),
                dict(result.attrs),
                result.type,
                result.span,
            )

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = axis_program()
        for transform in manager.passes[:boundary]:
            program = transform(program)
        mutation = ForgeShape()
        forged = mutation.visit_program(program)
        assert mutation.changed
        original = ir.serialize(forged)
        selected = passes.select_buffer_ir_functions()(forged)
    assert ir.serialize(forged) == original
    function = selected.get_function("kernel")
    assert function is not None and function.attrs["buffer_ir_selected"] is False


def test_runtime_axis_actual_metadata_changes_across_reset_visits():
    program, _ = lower(axis_program(varying=True))
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [p.get_function("kernel") for p in [logical_program, physical_program]]
    assert logical is not None and physical is not None
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 8
    for rows in [0, 1, 7, 8]:
        for choice in [0, 1]:
            output = torch.full_like(x, -777)
            namespace["kernel"](x, rows, choice, output)
            expected = torch.full_like(x, -777)
            for _ in range(2):
                for visit in range(3):
                    if visit % 2 == choice:
                        active = min(rows, visit % 8)
                        expected[:active] = x[:active] * ((x[:active] ** 2).sum(dim=1, keepdim=True) + 1)
            torch.testing.assert_close(output, expected)
            check_runtime_commands(logical, physical, {1: rows, 2: choice})


def recurrence_program(
    *, full_observation=False, varying=False, old_seed=False, inner_bound=False, aliases=0
):
    """The generic full-seed/prefix recurrence used by post-normalization.

    The full-observation variant matches rms_recip: a partial recurrence
    contribution followed by a full column store.
    """
    active = "pl.min(rows, visit % 8)" if varying else "rows"
    load_active = "pl.min(active, k % 8)" if inner_bound else "active"
    # The parser keeps source-level bindings type-stable before SSA conversion.
    # Reduction and final carriers intentionally have different active metadata;
    # distinct names let the test reach the normalization/admission boundary.
    final = f"scaled_alias_{aliases - 1}" if aliases else "scaled"
    alias_lines = "\n                    ".join(
        f"scaled_alias_{i} = {'scaled' if i == 0 else f'scaled_alias_{i - 1}'}" for i in range(aliases)
    )
    observe = (
        f"final_column = pl.reshape({final}, [8,1])\n"
        "                    seed_output = pl.store(final_column, [0,0], seed_output)"
        if full_observation
        else f"final_column = pl.reshape({final}, [8,1])\n"
        "                    output_source = pl.load(x, [0,0], [8,16], valid_shape=[active,16])\n"
        "                    result = pl.row_expand_mul(output_source, final_column)\n"
        "                    out = pl.store(result, [0,0], out)"
    )
    seed_use = (
        "seed_alias = seed\n"
        "                    original = pl.reshape(seed_alias, [8,1])\n"
        "                    seed_output = pl.store(original, [0,0], seed_output)"
        if old_seed
        else ""
    )
    return pl.parse_program(f"""
@pl.program
class PrefixRecurrence:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,16], pl.FP32],
               rows: pl.Scalar[pl.INDEX], trips: pl.Scalar[pl.INDEX], choice: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[8,16], pl.FP32]],
               seed_output: pl.InOut[pl.Tensor[[8,1], pl.FP32]]) -> pl.Tensor[[8,16], pl.FP32]:
        for outer in pl.range(2):
            for visit in pl.range(3):
                if visit % 2 == choice:
                    active = {active}
                    seed = pl.tile.full([1,8], value=2.0, dtype=pl.FP32)
                    total = seed
                    for k in pl.range(trips):
                        reduction_source = pl.load(x, [0,0], [8,16], valid_shape=[{load_active},16])
                        workspace = pl.create_tile([8,16], dtype=pl.FP32)
                        reduced_column = pl.row_sum(pl.mul(reduction_source, reduction_source), workspace)
                        part = pl.reshape(reduced_column, [1,8])
                        total = pl.add(total, part)
                    scaled = pl.add(pl.mul(total, 0.5), 1.0)
                    {alias_lines}
                    {observe}
                    {seed_use}
        return out
""")


@pytest.mark.parametrize("varying", [False, True])
@pytest.mark.parametrize("aliases", [0, 3])
def test_prefix_recurrence_values_matching_and_reset_visits(varying, aliases, tmp_path):
    # Values and commands are checked from emitted Buffer IR, independently of
    # the normalizer's domain graph. Zero trips must retain the initialized seed.
    program, _ = lower(recurrence_program(varying=varying, aliases=aliases))
    namespace = {}
    exec(torch_codegen(program), namespace)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [p.get_function("kernel") for p in [logical_program, physical_program]]
    assert logical is not None and physical is not None
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 32
    for rows in [0, 1, 7, 8]:
        for trips in [0, 1, 3]:
            for choice in [0, 1]:
                out, seed = torch.full_like(x, -777), torch.full((8, 1), -777.0)
                namespace["kernel"](x, rows, trips, choice, out, seed)
                expected = torch.full_like(x, -777)
                for _ in range(2):
                    for visit in range(3):
                        if visit % 2 == choice:
                            active = min(rows, visit % 8) if varying else rows
                            scale = (2 + trips * (x[:active] ** 2).sum(dim=1, keepdim=True)) * 0.5 + 1
                            expected[:active] = x[:active] * scale
                torch.testing.assert_close(out, expected)
                assert torch.equal(seed, torch.full_like(seed, -777))
                check_runtime_commands(logical, physical, {1: rows, 2: trips, 3: choice})
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "recurrence.pto", tmp_path / "recurrence.cpp"
        pto.write_text(codegen.PTOCodegen().generate(physical_program))
        _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2"])
        assert cpp.is_file()


def test_prefix_recurrence_unproved_inner_bound_preserves_original_body():
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = recurrence_program(inner_bound=True)
        for transform in manager.passes[:boundary]:
            program = transform(program)
        original = program.get_function("kernel")
        assert original is not None
        saved = ir.serialize(program)
        selected = passes.select_buffer_ir_functions()(program)
    assert ir.serialize(program) == saved
    function = selected.get_function("kernel")
    assert function is not None and function.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(original.body, function.body, enable_auto_mapping=False)


def test_prefix_recurrence_retains_old_full_seed_observation():
    # A full use of the ORIGINAL initialized value differs from a full use of
    # the partial recurrence result. Its values and metadata must be retained.
    program, _ = lower(recurrence_program(old_seed=True))
    namespace = {}
    exec(torch_codegen(program), namespace)
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 32
    for rows in [0, 1, 7, 8]:
        out, seed = torch.full_like(x, -777), torch.full((8, 1), -777.0)
        namespace["kernel"](x, rows, 3, 0, out, seed)
        torch.testing.assert_close(seed, torch.full_like(seed, 2))


def test_prefix_recurrence_failed_later_admission_keeps_original():
    # Addressless planning lacks the separate native axis-metadata objects.
    # A successfully normalized prefix must not escape when that later gate fails.
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PTOAS):
        manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = recurrence_program()
        for transform in manager.passes[:boundary]:
            program = transform(program)
        original = program.get_function("kernel")
        selected = passes.select_buffer_ir_functions()(program).get_function("kernel")
    assert original is not None and selected is not None
    assert selected.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(original.body, selected.body, enable_auto_mapping=False)


def _selection_input(program):
    manager = PassManager.get_strategy(ir.OptimizationStrategy.Default)
    boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
    for transform in manager.passes[:boundary]:
        program = transform(program)
    return program


@pytest.mark.parametrize("aliases", [1, 3])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_plain_prefix_aliases_default_compilation(aliases, planner):
    # Ordinary compilation (autosync disabled) must not select Buffer IR and
    # then fail assignment symmetry just because an inferred prefix has aliases.
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=planner):
        program = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
            recurrence_program(aliases=aliases)
        )
    function = program.get_function("kernel")
    assert function is not None
    expected = (
        ir.FunctionIRStage.Functional if planner == passes.MemoryPlanner.PTOAS else ir.FunctionIRStage.Buffer
    )
    assert function.ir_stage == expected


@pytest.mark.parametrize("aliases", [1, 3])
@pytest.mark.parametrize("allocation_case", ["preserved", "different_base", "different_offset"])
def test_plain_prefix_aliases_rebind_types_and_preserve_allocation(aliases, allocation_case):
    class AllocatedAliasSource(ir.IRMutator):
        memory = None

        def __init__(self):
            super().__init__()
            self.replacements = {}

        def visit_var(self, op):
            if op in self.replacements:
                return self.replacements[op]
            return super().visit_var(op)

        def visit_assign_stmt(self, op):
            result = super().visit_assign_stmt(op)
            assert isinstance(result, ir.AssignStmt)
            if not result.var.name_hint.startswith("scaled"):
                return result
            kind = result.var.type
            assert isinstance(kind, ir.TileType)
            if self.memory is None:
                self.memory = ir.MemRef("prefix_alias_storage", 0, 32, result.span)
            memory = self.memory
            if result.var.name_hint.startswith("scaled_alias_0"):
                if allocation_case == "different_base":
                    memory = ir.MemRef("incompatible_alias_storage", 0, 32, result.span)
                elif allocation_case == "different_offset":
                    memory = ir.MemRef(self.memory.base_, 4, 32, result.span)
            allocated = ir.TileType(kind.shape, kind.dtype, memory, kind.tile_view, kind.memory_space)
            replacement = ir.Var(result.var.name_hint, allocated, result.var.span)
            self.replacements[result.var] = replacement
            return ir.AssignStmt(replacement, result.value, result.span)

    class Assignments(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.by_name = {}

        def visit_assign_stmt(self, op):
            self.by_name[op.var.name_hint] = op

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        original = _selection_input(recurrence_program(aliases=aliases))
        allocation = AllocatedAliasSource()
        original = allocation.visit_program(original)
        source = original.get_function("kernel")
        assert source is not None and allocation.memory is not None
        saved = ir.serialize(original)
        report = testing.normalize_buffer_active_domains(source)
        candidate = report["candidate"]
        if allocation_case != "preserved":
            # Supplied placement contradicts a plain alias's existing source
            # allocation. Descriptor narrowing cannot silently repair or
            # redirect that storage, even if all value-domain facts are known.
            assert not report["changed"] and candidate.body.same_as(source.body)
            assert ir.serialize(original) == saved
            return
        properties = passes.IRPropertySet()
        for prop in [
            passes.IRProperty.AssignTypeSymmetry,
            passes.IRProperty.SSAForm,
            passes.IRProperty.UseAfterDef,
        ]:
            properties.insert(prop)
        assert passes.PropertyVerifierRegistry.verify(properties, original) == []
        assert (
            passes.PropertyVerifierRegistry.verify(
                properties, ir.Program([candidate], original.name, original.span)
            )
            == []
        )
    assert report["changed"] and ir.serialize(original) == saved
    before, after = Assignments(), Assignments()
    before.visit_stmt(source.body)
    after.visit_stmt(candidate.body)
    observed = 0
    for name, assignment in after.by_name.items():
        if not name.startswith("scaled_alias_"):
            continue
        old = before.by_name[name]
        assert isinstance(assignment.value, ir.Var)
        ir.assert_structural_equal(assignment.var.type, assignment.value.type)
        assert not assignment.var.same_as(old.var)
        kind = assignment.var.type
        assert isinstance(kind, ir.TileType) and kind.memref is not None
        assert kind.memref.same_as(allocation.memory)
        assert isinstance(old.var.type, ir.TileType) and old.var.type.memref is not None
        assert old.var.type.memref.same_as(allocation.memory)
        observed += 1
    assert observed == aliases


@pytest.mark.parametrize("aliases", [1, 3])
def test_plain_prefix_alias_metadata_observation_retains_original(aliases):
    class ObserveAlias(ir.IRMutator):
        changed = False

        def visit_assign_stmt(self, op):
            result = super().visit_assign_stmt(op)
            assert isinstance(result, ir.AssignStmt)
            if not result.var.name_hint.startswith(f"scaled_alias_{aliases - 1}"):
                return result
            zero = ir.ConstInt(0, ir.DataType.INDEX, result.span)
            coordinate = ir.MakeTuple([zero, zero], result.span)
            read = ir.create_op_call("tile.read", [result.var, coordinate], {}, result.span)
            observed = ir.Var("observed_alias", read.type, result.span)
            self.changed = True
            return ir.SeqStmts([result, ir.AssignStmt(observed, read, result.span)], result.span)

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        mutation = ObserveAlias()
        original = mutation.visit_program(_selection_input(recurrence_program(aliases=aliases)))
        source = original.get_function("kernel")
        assert source is not None and mutation.changed
        saved = ir.serialize(original)
        report = testing.normalize_buffer_active_domains(source)
        selected = passes.select_buffer_ir_functions()(original).get_function("kernel")
    assert not report["changed"] and ir.serialize(original) == saved
    assert selected is not None and selected.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(source.body, selected.body, enable_auto_mapping=False)


@pytest.mark.parametrize("aliases", [0, 1, 3])
def test_full_observation_preserves_recurrence_seed_through_aliases(aliases):
    # A zero-extended contribution leaves the defined seed in the inactive
    # lanes. The store still observes their values through each alias.
    input_program = recurrence_program(full_observation=True, aliases=aliases)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        original = _selection_input(input_program)
        saved = ir.serialize(original)
        selected = passes.select_buffer_ir_functions()(original).get_function("kernel")
    assert ir.serialize(original) == saved
    assert selected is not None and selected.attrs["buffer_ir_selected"] is True
    program, _ = lower(input_program)
    namespace = {}
    exec(torch_codegen(program), namespace)
    x = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 32
    for rows in [0, 1, 7, 8]:
        out = torch.full_like(x, -777)
        seed = torch.full((8, 1), -777.0)
        namespace["kernel"](x, rows, 3, 0, out, seed)
        expected = torch.full_like(seed, 2.0)
        expected[:rows] += 1.5 * x[:rows].square().sum(dim=1, keepdim=True)
        torch.testing.assert_close(seed, expected)


def _is_static_row_vector(tile: ir.Type) -> bool:
    if not isinstance(tile, ir.TileType):
        return False
    shape = []
    for extent in tile.shape:
        assert isinstance(extent, ir.ConstInt)
        shape.append(extent.value)
    return shape == [1, 8]


def test_active_domain_normalization_preserves_shared_scalar_guard_dag():
    class SharedGuard(ir.IRMutator):
        condition = None

        def visit_if_stmt(self, op: ir.IfStmt) -> ir.Stmt:
            rewritten = super().visit_if_stmt(op)
            assert isinstance(rewritten, ir.IfStmt)
            if self.condition is not None:
                return rewritten
            condition = rewritten.condition
            # N distinct nodes, exponentially many tree paths. The normalizer's
            # definition index must not traverse uses, and its Tile-only rewrite
            # must preserve this unchanged scalar DAG without rebuilding paths.
            for _ in range(18):
                condition = ir.And(condition, condition, ir.DataType.BOOL, condition.span)
            self.condition = condition
            return ir.IfStmt(
                condition,
                rewritten.then_body,
                rewritten.else_body,
                rewritten.return_vars,
                rewritten.span,
            )

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        original = _selection_input(recurrence_program())
        before = original.get_function("kernel")
        assert before is not None
        baseline = testing.normalize_buffer_active_domains(before)
        mutation = SharedGuard()
        program = mutation.visit_program(original)
        source = program.get_function("kernel")
        assert source is not None
        normalized = testing.normalize_buffer_active_domains(source)
        selected = normalized["candidate"]
    assert baseline["changed"] and normalized["changed"]
    for field in ["definition_statements", "domain_statements", "rewritten_statements"]:
        assert normalized["work"][field] == baseline["work"][field]
    assert (
        normalized["work"]["metadata_expression_states"]
        <= baseline["work"]["metadata_expression_states"] + 36
    )
    assert normalized["work"]["scalar_stability_nodes"] <= baseline["work"]["scalar_stability_nodes"] + 18
    condition = mutation.condition
    assert condition is not None

    class Conditions(ir.IRVisitor):
        found = False

        def visit_if_stmt(self, op: ir.IfStmt) -> None:
            if op.condition.same_as(condition):
                self.found = True
            # Visit statement occurrences, not the very DAG whose sharing this
            # regression checks. Using the ordinary recursive visitor here
            # would make the checker itself perform exponential work.
            self.visit_stmt(op.then_body)
            if op.else_body is not None:
                self.visit_stmt(op.else_body)

        def visit_expr(self, expr: ir.Expr) -> None:
            pass

    observed = Conditions()
    observed.visit_stmt(selected.body)
    assert observed.found
    node = mutation.condition
    for _ in range(18):
        assert isinstance(node, ir.And) and node.left.same_as(node.right)
        node = node.left


@pytest.mark.parametrize("kind", ["call", "call_attribute", "tuple_projection"])
@pytest.mark.parametrize("site", ["assignment", "for_initializer", "while_initializer"])
def test_nested_scalar_tile_observation_prevents_active_domain_normalization(kind, site):
    class NestedObserver(ir.IRMutator):
        changed = False

        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            # An assignment callback has no recursive statement children. A
            # same-kind SeqStmts trampoline can suppress nested callbacks while
            # its base implementation is active, missing the inner recurrence.
            rewritten = super().visit_assign_stmt(op)
            assert isinstance(rewritten, ir.AssignStmt)
            if self.changed:
                return rewritten
            statement = rewritten
            statements: list[ir.Stmt] = [statement]
            call = statement.value
            if not isinstance(call, ir.Call) or call.op.name != ir.get_op("tile.add").name:
                return rewritten
            tile = statement.var.type
            if not _is_static_row_vector(tile):
                return rewritten
            span = statement.span
            if kind == "call":
                coordinates = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INDEX, span)] * 2, span)
                read = ir.create_op_call("tile.read", [statement.var, coordinates], {}, span)
            elif kind == "call_attribute":
                query = ir.create_op_call("tile.get_block_idx", [], {}, span)
                read = ir.Call(query.op, [], {}, {"observed_tile": statement.var}, query.type, span)
            else:
                captured = ir.MakeTuple([statement.var, ir.ConstFloat(1.0, ir.DataType.FP32, span)], span)
                aggregate = ir.Var("tile_capture", captured.type, span)
                statements.append(ir.AssignStmt(aggregate, captured, span))
                read = ir.TupleGetItemExpr(aggregate, 1, span)
            # No direct Call assignment: the observer is nested under
            # scalar arithmetic and observes the recurrence result whose
            # overwide declaration the candidate would normalize.
            scalar_type = read.type
            assert isinstance(scalar_type, ir.ScalarType)
            dtype = scalar_type.dtype
            one = (
                ir.ConstFloat(1.0, dtype, span) if dtype == ir.DataType.FP32 else ir.ConstInt(1, dtype, span)
            )
            nested = ir.Add(read, one, dtype, span)
            value = ir.Var("nested_observation", ir.ScalarType(dtype), span)
            if site == "assignment":
                statements.append(ir.AssignStmt(value, nested, span))
            else:
                # Even an empty loop evaluates its carried initializer.
                # Its body only references the carry; the checker must
                # inspect the declaration without following every use.
                carried = ir.IterArg("observed_carry", value.type, nested, span)
                body = ir.YieldStmt([carried], span)
                if site == "for_initializer":
                    index = ir.Var("observation_index", ir.ScalarType(ir.DataType.INDEX), span)
                    zero = ir.ConstInt(0, ir.DataType.INDEX, span)
                    unit = ir.ConstInt(1, ir.DataType.INDEX, span)
                    statements.append(ir.ForStmt(index, zero, zero, unit, [carried], body, [value], span))
                else:
                    statements.append(ir.WhileStmt(ir.ConstBool(False, span), [carried], body, [value], span))
            self.changed = True
            return ir.SeqStmts(statements, rewritten.span)

    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        mutation = NestedObserver()
        original = mutation.visit_program(_selection_input(recurrence_program()))
        assert mutation.changed
        before = original.get_function("kernel")
        assert before is not None
        normalized = testing.normalize_buffer_active_domains(before)
        selected = passes.select_buffer_ir_functions()(original).get_function("kernel")
    assert before is not None and selected is not None
    assert not normalized["changed"] and normalized["candidate"].body.same_as(before.body)
    assert selected.attrs["buffer_ir_selected"] is False
    ir.assert_structural_equal(before.body, selected.body, enable_auto_mapping=False)


@pytest.mark.parametrize(
    "site", ["assignment", "for_initializer", "while_initializer", "projection", "tuple", "carry_chain"]
)
def test_shared_scalar_call_dag_is_classified_once_across_many_prefix_bindings(site):
    class ManyConsumers(ir.IRMutator):
        def __init__(self, consumers, depth):
            super().__init__()
            self.consumers = consumers
            self.changed = False
            self.bindings = []
            self.carried = []
            self.chain_results = []
            span = ir.Span.unknown()
            self.expression = ir.create_op_call("tile.get_block_idx", [], {}, span)
            if site == "carry_chain":
                for index in range(depth):
                    carry = ir.IterArg(f"chain_{index}", self.expression.type, self.expression, span)
                    self.carried.append(carry)
                    self.chain_results.append(ir.Var(f"chain_result_{index}", carry.type, span))
                    self.expression = carry
            elif site in {"projection", "tuple"}:
                self.expression = ir.MakeTuple([self.expression], span)
                for _ in range(depth):
                    self.expression = ir.MakeTuple([self.expression, self.expression], span)
                if site == "projection":
                    for _ in range(depth + 1):
                        self.expression = ir.TupleGetItemExpr(self.expression, 0, span)
            else:
                for _ in range(depth):
                    self.expression = ir.Add(self.expression, self.expression, ir.DataType.INDEX, span)

        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            # An assignment callback has no recursive statement children. A
            # same-kind SeqStmts trampoline can suppress nested callbacks while
            # its base implementation is active, missing the inner recurrence.
            rewritten = super().visit_assign_stmt(op)
            assert isinstance(rewritten, ir.AssignStmt)
            if self.changed:
                return rewritten
            statement = rewritten
            statements: list[ir.Stmt] = [statement]
            call = statement.value
            if not isinstance(call, ir.Call) or call.op.name != ir.get_op("tile.add").name:
                return rewritten
            tile = statement.var.type
            if not _is_static_row_vector(tile):
                return rewritten
            source = statement.var
            span = statement.span
            additions_begin = len(statements)
            # Each new Tile definition is affected by the proved prefix.
            # Interleave a fresh scalar binding of the SAME expression DAG.
            # A global-binding-generation cache revisited that DAG U times.
            for index in range(self.consumers):
                output_type = ir.TileType([1, 8], ir.DataType.FP32, memory_space=ir.MemorySpace.Vec)
                value = ir.Var(f"extra_prefix_{index}", output_type, span)
                arithmetic = ir.Call(
                    ir.get_op("tile.adds"),
                    [source, ir.ConstFloat(0.25, ir.DataType.FP32, span)],
                    output_type,
                    span,
                )
                statements.append(ir.AssignStmt(value, arithmetic, span))
                scalar = ir.Var(f"shared_scalar_{index}", self.expression.type, span)
                if site in {"assignment", "projection", "tuple", "carry_chain"}:
                    statements.append(ir.AssignStmt(scalar, self.expression, span))
                    self.bindings.append(scalar)
                else:
                    carried = ir.IterArg(f"shared_carry_{index}", scalar.type, self.expression, span)
                    body = ir.YieldStmt([carried], span)
                    if site == "for_initializer":
                        clock = ir.Var(f"scalar_clock_{index}", ir.ScalarType(ir.DataType.INDEX), span)
                        zero = ir.ConstInt(0, ir.DataType.INDEX, span)
                        unit = ir.ConstInt(1, ir.DataType.INDEX, span)
                        statements.append(
                            ir.ForStmt(clock, zero, zero, unit, [carried], body, [scalar], span)
                        )
                    else:
                        statements.append(
                            ir.WhileStmt(ir.ConstBool(False, span), [carried], body, [scalar], span)
                        )
                    self.bindings.append(carried)
                source = value
            if site == "carry_chain":
                # A single lexical chain, rather than separate simplified
                # inputs per use: the innermost scalar identity is referenced
                # U times, and its initializer reaches D enclosing carries.
                body = ir.SeqStmts(
                    [*statements[additions_begin:], ir.YieldStmt([self.carried[-1]], span)], span
                )
                del statements[additions_begin:]
                zero, unit = [ir.ConstInt(n, ir.DataType.INDEX, span) for n in [0, 1]]
                for index in reversed(range(len(self.carried))):
                    clock = ir.Var(f"chain_clock_{index}", ir.ScalarType(ir.DataType.INDEX), span)
                    loop = ir.ForStmt(
                        clock,
                        zero,
                        unit,
                        unit,
                        [self.carried[index]],
                        body,
                        [self.chain_results[index]],
                        span,
                    )
                    if index:
                        body = ir.SeqStmts([loop, ir.YieldStmt([self.carried[index - 1]], span)], span)
                    else:
                        statements.append(loop)
            self.changed = True
            return ir.SeqStmts(statements, rewritten.span)

    reports = {}
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        original = _selection_input(recurrence_program())
        for consumers, depth in [(8, 8), (8, 32), (32, 8), (32, 32)]:
            mutation = ManyConsumers(consumers, depth)
            source = mutation.visit_program(original).get_function("kernel")
            assert mutation.changed and source is not None
            report = testing.normalize_buffer_active_domains(source)
            assert report["changed"]

            # Walk statement structure explicitly. A recursively overridden
            # Python visitor can suppress same-kind callbacks while its base
            # trampoline is active, then accidentally walk initializer DAGs.
            bindings, carried, results = [], [], []
            pending = [report["candidate"].body]
            while pending:
                statement = pending.pop()
                if isinstance(statement, ir.SeqStmts):
                    pending.extend(reversed(statement.stmts))
                elif isinstance(statement, ir.AssignStmt):
                    if statement.value.same_as(mutation.expression):
                        bindings.append(statement.var)
                elif isinstance(statement, (ir.ForStmt, ir.WhileStmt)):
                    carried.extend(statement.iter_args)
                    results.extend(statement.return_vars)
                    for carry in statement.iter_args:
                        if carry.initValue.same_as(mutation.expression):
                            bindings.append(carry)
                    pending.append(statement.body)
                elif isinstance(statement, ir.IfStmt):
                    pending.append(statement.then_body)
                    if statement.else_body is not None:
                        pending.append(statement.else_body)
            assert len(bindings) == consumers
            assert all(any(actual.same_as(expected) for actual in bindings) for expected in mutation.bindings)
            assert all(any(actual.same_as(expected) for actual in carried) for expected in mutation.carried)
            assert all(
                any(actual.same_as(expected) for actual in results) for expected in mutation.chain_results
            )
            reports[consumers, depth] = report["work"]
    # The depth contribution depends on the distinct DAG, independently of U.
    # Static statement occurrences still grow with U and were not CSE'd away.
    field = "scalar_stability_nodes"
    depth_cost = 72 if site == "carry_chain" else 48 if site == "projection" else 24
    assert reports[8, 32][field] - reports[8, 8][field] == depth_cost
    assert reports[32, 32][field] - reports[32, 8][field] == depth_cost
    rewrite_growth = [
        reports[consumers, 32]["rewrite_expression_dispatches"]
        - reports[consumers, 8]["rewrite_expression_dispatches"]
        for consumers in [8, 32]
    ]
    assert rewrite_growth[0] == rewrite_growth[1]
    assert 0 <= rewrite_growth[0] <= (96 if site == "carry_chain" else 0)
    assert reports[32, 8]["rewritten_statements"] > reports[8, 8]["rewritten_statements"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
