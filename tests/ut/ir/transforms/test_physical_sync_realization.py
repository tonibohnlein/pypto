# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualified direct physical commands, compact reuse and refusal contracts."""

import re

import pytest
from pypto import codegen, ir, testing
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .sync_logical_oracle import CommandExecution
from .sync_matmul_cases import CASES, capture, staged_fixture
from .sync_matmul_oracle import interpret
from .sync_physical_oracle import directional_widths, run_physical
from .test_finite_sync_analysis import closure, reference
from .test_logical_sync_realization import verify

pytestmark = pytest.mark.usefixtures("ascend_backend")


def check(function, eligible=None):
    retained = testing.analyze_placed_sync(function, True)
    before = retained.describe()
    result = retained.realize_physical("a2a3", eligible)
    verify(result["function"])
    state, order, projected = run_physical(
        function, result["function"], range(8) if eligible is None else eligible
    )
    # Compare the actual two emitted representations, not only their counters.
    _, calls = interpret(function)
    ref = reference([(p, True, reads, writes) for _, _, p, reads, writes in calls])
    logical = CommandExecution(calls, ref[0])
    logical.execute(retained.realize()["function"].body, {})
    logical_order = closure(logical.vertices, logical.edges)
    assert projected == {(a, b) for a, b in logical_order if a in set(ref[0]) and b in set(ref[0])}
    widths = directional_widths(state, order)
    if result["finite_minimum"]:
        assert widths == {d: len(ids) for d, ids in state.ids.items()}
    assert retained.describe() == before
    ir.assert_structural_equal(result["function"], retained.realize_physical("a2a3", eligible)["function"])
    program = ir.Program([result["function"]], "PhysicalMatmul", function.span)
    text = codegen.PTOCodegen().generate(program)
    assert "sync.record" not in text and "!pto.sync_token" not in text
    return result, state, widths, text


@pytest.mark.parametrize("name", CASES)
def test_original_matmuls(name):
    result, state, _, text = check(capture(name), list(range(6)))
    assert state.records and result["set_sites"] == result["wait_sites"]
    assert "pto.set_flag[" in text


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("name", CASES)
def test_downstream_manual_sync_preserves_flags(name, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    function = capture(name)
    result, _, _, _ = check(function, list(range(6)))
    program = ir.Program([result["function"]], "PhysicalMatmul", function.span)
    source = tmp_path / "physical.pto"
    source.write_text(codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False))
    outputs = []
    for auto in [False, True]:
        output = tmp_path / f"physical-{auto}.cpp"
        _run_ptoas(
            str(source),
            str(output),
            ["--pto-level=level3", "--pto-arch=a2", f"--enable-insert-sync={'true' if auto else 'false'}"],
        )
        outputs.append(output.read_text())
    # Literal flags establish manual ownership in the pinned downstream compiler.
    assert outputs[0] == outputs[1]
    # PTOAS emits an unused generic tail helper containing flags/barriers.
    # Count the kernel body and also ensure it does not call that helper.
    body = outputs[0].split("AICORE void ", 1)[1]
    assert "ptoas_auto_sync_tail(" not in body
    assert body.count("set_flag(") == result["set_sites"]
    assert body.count("wait_flag(") == result["wait_sites"]
    assert body.count("pipe_barrier(") == result["barrier_sites"]
    pto_flags = re.findall(
        r'pto\.(set_flag|wait_flag)\["(PIPE_\w+)", "(PIPE_\w+)", "(EVENT_ID\d+)"\]', source.read_text()
    )
    cpp_flags = re.findall(r"(set_flag|wait_flag)\((PIPE_\w+), (PIPE_\w+), (EVENT_ID\d+)\)", body)
    assert pto_flags == cpp_flags
    assert "__PTO_AUTO__" not in outputs[0] and "PTO_FLAG_TEST" not in outputs[0]


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("visits", [1, 2, 3])
def test_zero_tail_repeated_visits_and_resets(count, visits):
    result, state, _, _ = check(staged_fixture(count, visits=visits), list(range(6)))
    assert bool(state.records) == bool(count)
    assert result["set_sites"] == result["wait_sites"]


@pytest.mark.parametrize("start,step", [(0, 2), (3, 1), (3, 2)])
def test_original_occurrence_coordinates(start, step):
    check(staged_fixture(3, visits=2, start=start, step=step))


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_fanout_and_same_pipe_barriers(stages):
    check(staged_fixture(3, stages=stages))


def test_large_trip_counts_do_not_unfold():
    signatures = []
    for count in [4, 8, 16, 64, 1024, 1000000]:
        r = testing.analyze_placed_sync(staged_fixture(count, visits=2), True).realize_physical("a2a3")
        signatures.append(
            tuple(
                r[key]
                for key in [
                    "set_sites",
                    "wait_sites",
                    "barrier_sites",
                    "guard_sites",
                    "reserved_lanes",
                    "reuse_queries",
                ]
            )
        )
    assert all(s == signatures[0] for s in signatures)


def test_sparse_directional_pool():
    _, state, _, _ = check(staged_fixture(3), [2, 4, 6, 7])
    assert all(ids <= {2, 4, 6, 7} for ids in state.ids.values())


def test_refusal_keeps_original_mathematics():
    f = capture("matmul_64")
    retained = testing.analyze_placed_sync(f, True)
    before = retained.describe()
    with pytest.raises(ValueError, match="Fixed-plan scarcity"):
        retained.realize_physical("a2a3", [])
    with pytest.raises(ValueError, match="Unsupported physical target"):
        retained.realize_physical("a5")
    with pytest.raises(ValueError, match="Invalid/duplicate"):
        retained.realize_physical("a2a3", [0, 0])
    with pytest.raises(ValueError, match="outside the legal"):
        retained.realize_physical("a2a3", [8])
    assert before == retained.describe()
    verify(retained.realize()["function"])


def test_oracle_rejects_missing_consumer():
    f = staged_fixture(2)
    r = testing.analyze_placed_sync(f, True).realize_physical("a2a3")

    class Drop(ir.IRMutator):
        dropped = False

        def visit_eval_stmt(self, op: ir.EvalStmt) -> ir.Stmt:
            stmt = op
            if (
                not self.dropped
                and isinstance(stmt.expr, ir.Call)
                and stmt.expr.op.name == ir.get_op("system.wait_flag").name
            ):
                self.dropped = True
                return ir.SeqStmts([], stmt.span)
            return stmt

    with pytest.raises(AssertionError):
        run_physical(f, Drop().visit_function(r["function"]))


def test_nonempty_finite_scarcity_has_independent_width():
    function = capture("matmul_64")
    _, _, widths, _ = check(function)
    assert max(widths.values()) == 2
    retained = testing.analyze_placed_sync(function, True)
    with pytest.raises(ValueError, match="Fixed-plan scarcity.*required_at_least=2.*eligible=1"):
        retained.realize_physical("a2a3", [0])


def test_compact_insufficient_construction_is_not_infeasibility():
    retained = testing.analyze_placed_sync(staged_fixture(4), True)
    with pytest.raises(ValueError, match="Allocation unproved"):
        retained.realize_physical("a2a3", [0])
    # Refusal has not replaced or invalidated the retained mathematical owner.
    verify(retained.realize()["function"])


def test_oracle_rejects_wrong_wait_id():
    function = staged_fixture(2)
    emitted = testing.analyze_placed_sync(function, True).realize_physical("a2a3")["function"]

    class WrongId(ir.IRMutator):
        changed = False

        def visit_call(self, op: ir.Call) -> ir.Expr:
            call = op
            if not self.changed and call.op.name == ir.get_op("system.wait_flag").name:
                self.changed = True
                attrs = dict(call.kwargs)
                attrs["event_id"] = 7
                return ir.Call(call.op, call.args, attrs, None, call.type, call.span)
            return call

    with pytest.raises(AssertionError, match="wait without a matching publication"):
        run_physical(function, WrongId().visit_function(emitted))


def test_textually_matched_reuse_still_needs_causal_order():
    class Interleave(ir.IRMutator):
        def visit_for_stmt(self, op):
            assert isinstance(op.body, ir.SeqStmts)
            stmts = list(op.body.stmts)
            # load A, copy A, load B, copy B: flag uses alternate textually,
            # but MTE1's first WAIT does not reach MTE2's second publication.
            stmts[:4] = [stmts[0], stmts[2], stmts[1], stmts[3]]
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts(stmts, op.span),
                op.return_vars,
                op.span,
            )

    function = Interleave().visit_function(staged_fixture(1, stages=1))
    result, _, _, _ = check(function)

    class Collapse(ir.IRMutator):
        def visit_call(self, op: ir.Call) -> ir.Expr:
            call = op
            if call.op.name in {ir.get_op(n).name for n in ["system.set_flag", "system.wait_flag"]}:
                attrs = dict(call.kwargs)
                if attrs["src_pipe"] == int(ir.PipeType.MTE2) and attrs["dst_pipe"] == int(ir.PipeType.MTE1):
                    attrs["event_id"] = 0
                    return ir.Call(call.op, call.args, attrs, None, call.type, call.span)
            return call

    with pytest.raises(AssertionError, match="consumption does not causally precede republication"):
        run_physical(function, Collapse().visit_function(result["function"]))


def test_physical_round_trip_preserves_ids_and_guards(tmp_path):
    function = staged_fixture(3, visits=2, start=3, step=2)
    result, _, _, _ = check(function)
    path = tmp_path / "physical.msgpack"
    ir.serialize_to_file(result["function"], str(path))
    decoded = ir.deserialize_from_file(str(path))
    assert isinstance(decoded, ir.Function)
    ir.assert_structural_equal(result["function"], decoded)
    run_physical(function, decoded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
