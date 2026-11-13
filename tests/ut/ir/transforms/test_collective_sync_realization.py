# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Complete collective phase repetition through the shared physical exporter."""

import pytest
from pypto import codegen, ir, testing
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_oracle import hardware_reference, interpret, validate
from .sync_physical_oracle import directional_widths, run_physical
from .test_autosync_switch import apply_hybrid
from .test_collective_sync_interface import collective_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def repeat_collective(rows=5, visits=3, *, start=1, step=2, split=False):
    original = collective_fixture(rows, [[(0, rows)]])
    span = original.span
    assert isinstance(original.body, ir.SeqStmts)
    definition, *statements, ret = original.body.stmts
    iv = ir.Var("visit", ir.ScalarType(ir.DataType.INDEX), span)

    def ci(x):
        return ir.ConstInt(x, ir.DataType.INDEX, span)

    selected = statements[:1] if split else statements
    # Include a harmless scalar definition so split=True exercises the actual
    # structured repeated-subtree certificate, not the perfect-nest refusal.
    ready = ir.Var("ready", ir.ScalarType(ir.DataType.INDEX), span)
    body = ir.SeqStmts([ir.AssignStmt(ready, ci(0), span), *selected, ir.YieldStmt([], span)], span)
    loop = ir.ForStmt(iv, ci(start), ci(start + visits * step), ci(step), [], body, [], span)
    remaining = statements[1:] if split else []
    return ir.Function(
        original.name,
        list(zip(original.params, original.param_directions)),
        original.return_types,
        ir.SeqStmts([definition, loop, *remaining, ret], span),
        span,
        original.func_type,
        ir_stage=original.ir_stage,
    )


@pytest.mark.parametrize("rows", [1, 2, 5, 9])
@pytest.mark.parametrize("visits", [0, 1, 2, 3])
def test_complete_phase_resets_keep_covers_queries_and_matching(rows, visits):
    function = repeat_collective(rows, visits)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    physical = result.realize_physical("a2a3", list(range(6)))
    state, order, projected = run_physical(function, physical["function"], range(6))
    _, calls = interpret(function)
    required = hardware_reference(function, calls)[2]
    assert projected == required
    assert all(len(ids) <= 6 for ids in state.ids.values())
    assert all(width <= 6 for width in directional_widths(state, order).values())


@pytest.mark.parametrize("visits", [0, 1, 2, 5])
def test_partial_phase_repeat_cannot_inherit_full_phase_certificate(visits):
    with pytest.raises(ValueError, match="(Visit-owned|selector partition)"):
        testing.analyze_placed_sync(repeat_collective(5, visits, split=True), True)


def test_collective_repetition_and_row_growth_preserve_command_size():
    signatures = []
    row_interfaces = []
    for rows, visits in [(5, 3), (1000, 3), (1000000, 3), (5, 1000), (5, 1000000)]:
        retained = testing.analyze_placed_sync(repeat_collective(rows, visits), True)
        physical = retained.realize_physical("a2a3", list(range(6)))
        d = retained.describe()
        assert d["collective_witnesses"] == 2
        assert d["source_coordinate_entries"] == 6
        # Repetition uses a binary summary, not one retained node per visit.
        # Keep this separate from the constant physical command schema below.
        assert d["merge_nodes"] <= 2 * visits.bit_length()
        assert d["interface_entries"] <= 672 * visits.bit_length()
        if visits == 3:
            row_interfaces.append(
                tuple(d[k] for k in ["interface_entries", "interface_queries", "merge_nodes"])
            )
        signatures.append(
            tuple(
                physical[k]
                for k in ["set_sites", "wait_sites", "barrier_sites", "guard_sites", "reserved_lanes"]
            )
        )
    assert len(set(signatures)) == 1
    assert len(set(row_interfaces)) == 1


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("repeated", [False, True])
def test_collective_production_and_cpp(repeated, tmp_path):
    function = repeat_collective() if repeated else collective_fixture()
    emitted = apply_hybrid([function])[function.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR], dict(emitted.attrs)
    run_physical(function, emitted, range(6))
    text = codegen.PTOCodegen().generate(
        ir.Program([emitted], "collective", emitted.span), emit_tile_addr=True, emit_source_loc=False
    )
    source = tmp_path / "collective.pto"
    source.write_text(text)
    output = tmp_path / "collective.cpp"
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    assert "set_flag" in output.read_text()


def nested_collective(rows, counts):
    function = repeat_collective(rows, counts[-1])
    for depth, count in enumerate(reversed(counts[:-1])):
        assert isinstance(function.body, ir.SeqStmts)
        definition, loop, ret = function.body.stmts
        span = function.span
        iv = ir.Var(f"outer_{depth}", ir.ScalarType(ir.DataType.INDEX), span)
        bounds = [ir.ConstInt(x, ir.DataType.INDEX, span) for x in [0, count, 1]]
        wrapper = ir.ForStmt(iv, *bounds, [], ir.SeqStmts([loop, ir.YieldStmt([], span)], span), [], span)
        function = ir.Function(
            function.name,
            list(zip(function.params, function.param_directions)),
            function.return_types,
            ir.SeqStmts([definition, wrapper, ret], span),
            span,
            function.func_type,
            ir_stage=function.ir_stage,
        )
    return function


@pytest.mark.parametrize("counts", [(0, 2), (2, 0), (1, 1), (2, 3)])
def test_nested_complete_phase_repetition_preserves_mixed_radix_resets(counts):
    function = nested_collective(3, counts)
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    physical = result.realize_physical("a2a3", list(range(6)))
    run_physical(function, physical["function"], range(6))


def outer_access_fixture(write=False):
    original = repeat_collective(5, 2, start=0, step=1)

    class OuterAccess(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.outer = None
            self.output = ir.Var("outer_output", ir.TensorType([10, 8], ir.DataType.FP32), original.span)

        def visit_var(self, op):
            return self.output if write and op.same_as(original.params[1]) else op

        def visit_for_stmt(self, op):
            if op.loop_var.name_hint == "visit":
                self.outer = op.loop_var
            return super().visit_for_stmt(op)

        def visit_call(self, op):
            op = super().visit_call(op)
            assert isinstance(op, ir.Call)
            target = "buffer.store" if write else "buffer.load"
            if op.op.name == ir.get_op(target).name:
                assert self.outer is not None
                span = op.span
                value = (
                    ir.Mul(self.outer, ir.ConstInt(5, ir.DataType.INDEX, span), ir.DataType.INDEX, span)
                    if write
                    else self.outer
                )
                offsets = ir.MakeTuple([value, ir.ConstInt(0, ir.DataType.INDEX, span)], span)
                return ir.Call(op.op, [op.args[0], offsets, *op.args[2:]], op.type, span)
            return op

    return OuterAccess().visit_function(original)


def test_enclosing_readonly_gm_selectors_preserve_actual_frame():
    function = outer_access_fixture()
    result = testing.analyze_placed_sync(function, True)
    validate(function, result)
    run_physical(function, result.realize_physical("a2a3", list(range(6)))["function"], range(6))


def test_enclosing_written_domain_keeps_missing_reset_export_explicit():
    with pytest.raises(ValueError, match="(Visit-owned|selector partition)"):
        testing.analyze_placed_sync(outer_access_fixture(write=True), True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
