# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native child retains payload order without inventing full completion edges."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .sync_closed_command_oracle import ClosedCommands, check_closed_commands
from .sync_matmul_oracle import protected_accumulator_pairs
from .test_conservative_sync_children import exported, resident_child_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")
BASES = {0: 0, 1: 1 << 20, 2: 2 << 20}


def payload_order(state):
    return {(a, b) for a, b in state.order if a[0] == b[0] == "payload"}


def run_plan(original, logical):
    physical = testing.lower_sync_schedule(logical, eligible_ids=[2, 5])["function"]
    return check_closed_commands(
        logical, physical, {}, [2, 5], tensor_bases=BASES, compare_commands=False, hardware_source=original
    )


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("visits", [0, 1, 2])
@pytest.mark.parametrize("initialize_in_child", [False, True])
def test_preserved_native_child_removes_only_protected_internal_completion_order(
    count, visits, initialize_in_child
):
    original, child_scope = resident_child_fixture(count, visits, initialize_in_child=initialize_in_child)
    before = ir.serialize(original)
    export, mathematics = exported(original, child_scope)
    primitive = testing.insert_conservative_prefixes(original)
    preserved = testing.insert_conservative_prefixes(original, children=[export["child"]])
    assert ir.serialize(original) == before
    # The retained mathematical owner survives both realization requests.
    assert testing.export_native_sync_child(mathematics, child_scope)["child"] is not None
    _, baseline = run_plan(original, primitive)
    _, candidate = run_plan(original, preserved)
    assert baseline.payloads == candidate.payloads
    baseline_order, candidate_order = payload_order(baseline), payload_order(candidate)
    assert not candidate_order - baseline_order, "retained child added payload order"

    source = ClosedCommands(original, {}, [2, 5], BASES)
    source.run()
    protected = protected_accumulator_pairs(original, source.calls)
    assert isinstance(original.body, ir.SeqStmts)
    outer = original.body.stmts[-2]
    assert isinstance(outer, ir.ForStmt)
    expected_removed = set()
    for a, b in protected:
        source_env, target_env = source.calls[a][1], source.calls[b][1]
        if (
            child_scope.loop_var in source_env
            and child_scope.loop_var in target_env
            and (source_env[outer.loop_var] == target_env[outer.loop_var])
        ):
            expected_removed.add((("payload", a, "C"), ("payload", b, "I")))
    removed = baseline_order - candidate_order
    assert removed == expected_removed, ("unexpected removed event pairs", removed, expected_removed)
    assert len(removed) == visits * count * (count - 1) // 2
    # The atomic child has one conservative entry barrier per outer visit,
    # even if its body executes zero times. Primitive execution instead has
    # one barrier per actual inner visit. Count this tradeoff explicitly;
    # payload-order containment above remains exact in every case.
    assert candidate.barriers - baseline.barriers == visits * (1 - count)
    if count > 1 and visits:
        assert candidate.barriers < baseline.barriers
        # The native accumulator rule exempts hazards; it does not claim full
        # instruction completion before the next protected instruction starts.
        assert all(pair not in candidate_order for pair in expected_removed)


@pytest.mark.parametrize("m,n,protected", [(64, 64, True), (16, 16, False)])
def test_accumulator_exemption_changes_required_hazards_not_native_event_edges(m, n, protected):
    original, child = resident_child_fixture(3, 1, m=m, n=n)
    source = ClosedCommands(original, {}, [2, 5], BASES)
    source.run()
    inner = [index for index, entry in enumerate(source.calls) if child.loop_var in entry[1]]
    assert len(inner) == 3
    pairs = protected_accumulator_pairs(original, source.calls)
    assert ((inner[0], inner[1]) in pairs) is protected
    assert (("payload", inner[0], "C"), ("payload", inner[1], "I")) not in source.order
    assert (("payload", inner[0], "I"), ("payload", inner[1], "I")) in source.order
    assert (("payload", inner[0], "C"), ("payload", inner[1], "C")) in source.order


@pytest.mark.ptoas_compile
def test_public_retained_child_pipeline_and_cpp_preserve_actual_commands(tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    original, scope = resident_child_fixture(3, 2, initialize_in_child=True)
    export, mathematics = exported(original, scope)
    program = ir.Program([original], "PublicNativeChild", original.span)
    before = ir.serialize(program)
    with passes.PassContext([], autosync_policy="conservative_prefix", autosync_disjoint_gm=True):
        logical_program = passes.insert_sync(accepted_children=[export["child"]])(program)
        optimized = passes.optimize_sync()(logical_program)
    # Actual-token lowering does not rely on the producer policy or its owner.
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(optimized)
    logical = next(iter(logical_program.functions.values()))
    physical = next(iter(physical_program.functions.values()))
    assert ir.serialize(program) == before
    assert testing.export_native_sync_child(mathematics, scope)["child"] is not None
    _, checked = check_closed_commands(
        logical, physical, {}, range(6), tensor_bases=BASES, compare_commands=False, hardware_source=original
    )
    assert dict(physical.attrs)["pypto_autosync"] is True
    assert all(
        (("payload", index, "C"), checked.returns[-1]) in checked.order
        for index in range(len(checked.payloads))
    )
    source, output = tmp_path / "native_child.pto", tmp_path / "native_child.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(physical_program, emit_tile_addr=True, emit_source_loc=False)
    )
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    cpp = output.read_text()
    assert "set_flag(" in cpp and "wait_flag(" in cpp and original.name in cpp
