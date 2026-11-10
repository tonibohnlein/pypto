# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Actual emitted slot-pipeline commands checked independently of allocation."""

import pytest
from pypto import codegen, ir, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_oracle import validate
from .sync_physical_oracle import directional_widths, run_physical
from .test_autosync_switch import apply_hybrid
from .test_logical_sync_realization import verify
from .test_regional_slot_composition import regional_fixture, repeated_fixture
from .test_rotating_sync_analysis import rotating_fixture
from .test_streamed_slot_composition import streamed_fixture, streamed_sequence

pytestmark = pytest.mark.usefixtures("ascend_backend")


def resident_streamed_sequence(count):
    return streamed_sequence(count, resident_reader=True)


def distinct_selector_fixture(count):
    return rotating_fixture(count, [(3, 1)], [("load", 0, 0), ("load", 0, 1), ("rmw", 0, 0), ("rmw", 0, 1)])


def local_sequence(counts=(6, 5)):
    return regional_fixture(counts, schedules=[[("load", 0, 0), ("rmw", 0, 0)]] * len(counts))


def check_buffered(function):
    retained = testing.analyze_placed_sync(function, True)
    semantic = validate(function, retained)
    before = ir.serialize(ir.Program([function], "original", function.span))
    result = retained.realize_physical("a2a3", list(range(6)))
    verify(result["function"])
    state, order, projected = run_physical(function, result["function"], range(6))
    assert ir.serialize(ir.Program([function], "original", function.span)) == before
    assert all(len(ids) <= 6 for ids in state.ids.values())
    assert sum(f["count"] for f in result["families"] if f["ids"]) == len(state.records)
    return retained, result, state, directional_widths(state, order), semantic, projected


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 10])
@pytest.mark.parametrize("step", [1, 2])
def test_periodic_distance_and_short_prefix_matching(count, step):
    function = streamed_fixture(count, start=1 if count else 0, step=step)
    retained, result, state, _, _, _ = check_buffered(function)
    if count >= 7:
        assert any(f["target"][1] - f["source"][1] > 1 for f in result["families"])
        assert state.reuses
    assert retained.describe()["regions"][0]["rotating"]


@pytest.mark.parametrize("count", [0, 1, 2, 5, 8])
def test_streamed_region_native_entry_and_tail_commands(count):
    check_buffered(resident_streamed_sequence(count))


@pytest.mark.parametrize("counts", [(0, 0), (1, 1), (2, 0, 3), (6, 5), (4, 5, 2)])
def test_flat_region_resets_bind_actual_boundary_occurrences(counts):
    retained, result, _, _, _, _ = check_buffered(local_sequence(counts))
    assert retained.describe()["hierarchical"]
    if counts == (6, 5):
        assert any(
            f["source"][0] != f["target"][0]
            and f["source"][1] != counts[f["source"][0]] - 1
            and f["target"][1] != 0
            for f in result["families"]
        ), "fixture must exercise interior-to-interior crossings"


def test_shared_reset_export_refuses_without_losing_mathematics():
    function = repeated_fixture(4, 3)
    retained = testing.analyze_placed_sync(function, True)
    expected = retained.expand_retained_demands(64)
    with pytest.raises(ValueError, match="(scope|repet|crossing|reset)"):
        retained.realize_physical("a2a3", list(range(6)))
    assert retained.expand_retained_demands(64) == expected
    validate(function, retained)
    delegated = apply_hybrid([function])[function.name]
    assert not dict(delegated.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert dict(delegated.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]
    ir.assert_structural_equal(function.body, delegated.body)


def test_scarcity_keeps_retained_answer_and_original_commands():
    function = streamed_fixture(7)
    retained = testing.analyze_placed_sync(function, True)
    with pytest.raises(ValueError, match="(capacity|scarcity|Allocation|eligible)"):
        retained.realize_physical("a2a3", [])
    validate(function, retained)
    result = retained.realize_physical("a2a3", list(range(6)))
    run_physical(function, result["function"], range(6))


@pytest.mark.parametrize("factory", [streamed_fixture, resident_streamed_sequence])
def test_large_trip_counts_do_not_expand_command_sites(factory):
    records = []
    for count in [7, 1000, 1000000]:
        retained = testing.analyze_placed_sync(factory(count), True)
        physical = retained.realize_physical("a2a3", list(range(6)))
        records.append(
            tuple(
                physical[key]
                for key in [
                    "set_sites",
                    "wait_sites",
                    "barrier_sites",
                    "guard_sites",
                    "reserved_lanes",
                ]
            )
        )
    assert len(set(records)) == 1


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "factory",
    [
        streamed_fixture,
        resident_streamed_sequence,
        distinct_selector_fixture,
        lambda count: local_sequence((count, 4)),
    ],
)
def test_ordinary_production_and_native_cpp(factory, tmp_path):
    function = factory(7)
    emitted = apply_hybrid([function])[function.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR], dict(emitted.attrs)
    run_physical(function, emitted, range(6))
    text = codegen.PTOCodegen().generate(
        ir.Program([emitted], "buffered", emitted.span), emit_tile_addr=True, emit_source_loc=False
    )
    assert "pto.set_flag" in text and "pto.wait_flag" in text
    assert "pto.alloc_multi_tile" in text
    source = tmp_path / "buffered.pto"
    source.write_text(text)
    output = tmp_path / "buffered.cpp"
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2"])
    cpp = output.read_text()
    assert "set_flag" in cpp and "wait_flag" in cpp
    assert "pipe_barrier(PIPE_S)" not in cpp


def test_writer_only_prologue_keeps_same_dma_pipe_refusal():
    function = streamed_sequence(5)
    retained = testing.analyze_placed_sync(function, True)
    validate(function, retained)
    with pytest.raises(ValueError, match="same-pipe barrier mechanism"):
        retained.realize_physical("a2a3", list(range(6)))
    delegated = apply_hybrid([function])[function.name]
    assert not dict(delegated.attrs)[PYPTO_AUTOSYNC_ATTR]
    ir.assert_structural_equal(function.body, delegated.body)


@pytest.mark.parametrize("count", [0, 1, 2, 5, 8])
def test_distinct_slot_selections_preserve_matching_and_reuse(count):
    check_buffered(distinct_selector_fixture(count))
