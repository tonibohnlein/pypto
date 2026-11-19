# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared production GM-pipe layout ownership and inspected-work regressions."""

import gc

import pytest
from pypto import DataType, ir, testing


def setup(pipe_id=0, direction=3, slot_bytes=8192, slots=2):
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INT32, span)
    return ir.EvalStmt(
        ir.create_op_call(
            "system.aic_initialize_pipe",
            [zero, zero],
            {"id": pipe_id, "dir_mask": direction, "slot_size": slot_bytes, "slot_num": slots},
            span,
        ),
        span,
    )


def function(name, statements):
    span = ir.Span.unknown()
    return ir.Function(name, [], [], ir.SeqStmts([*statements, ir.ReturnStmt(span)], span), span)


def call(name):
    span = ir.Span.unknown()
    return ir.EvalStmt(ir.Call(ir.GlobalVar(name), [], ir.VoidType(), span), span)


def program(functions):
    return ir.Program(functions, "pipe_layout", ir.Span.unknown())


@pytest.mark.parametrize("direction,slots,rings", [(1, 0, 1), (2, 3, 1), (3, 0, 2), (3, 2, 2)])
def test_original_setup_extent_and_identity(direction, slots, rings):
    cut = setup(direction=direction, slots=slots)
    leaf = function("leaf", [cut])
    owner = program([leaf])
    evidence = testing.GMPipeLayoutEvidence(owner)
    expected_slots = slots or (4 if direction == 3 else 8)
    result = evidence.layout(leaf)
    assert result["error"] == ""
    assert result["bytes"] == expected_slots * rings * 8192
    assert result["elements"] == result["bytes"] // 4
    assert result["offsets"] == {(0, direction): 0}
    assert evidence.has_setup(leaf, cut)
    foreign = function("leaf", [cut])
    assert not evidence.has_setup(foreign, cut)
    assert "original Program" in evidence.layout(foreign)["error"]
    del owner
    gc.collect()
    assert evidence.layout(leaf) == result
    assert evidence.work()["setup_decodes"] == 1


def test_reachable_failure_is_cached_without_poisoning_other_roots():
    small = function("small", [setup(slot_bytes=512)])
    large = function("large", [setup(slot_bytes=1024)])
    good = function("good", [call("small"), call("small")])
    bad = function("bad", [call("small"), call("large")])
    evidence = testing.GMPipeLayoutEvidence(program([small, large, good, bad]))
    assert evidence.layout(good)["bytes"] == 2048
    failure = evidence.layout(bad)
    assert "inconsistent slot_size" in failure["error"]
    assert failure["bytes"] == 0 and failure["offsets"] == {}
    work = evidence.work()
    for _ in range(32):
        assert evidence.layout(bad) == failure
        assert evidence.layout(good)["bytes"] == 2048
    after = evidence.work()
    assert after["functions_visited"] == work["functions_visited"]
    assert after["setup_incidences"] == work["setup_incidences"]
    assert after["statements"] == work["statements"]
    assert after["layouts_rejected"] == work["layouts_rejected"]
    assert after["layout_cache_hits"] - work["layout_cache_hits"] == 64


def test_multiple_regions_are_sorted_by_original_pipe_key():
    leaf = function("leaf", [setup(3, 2, 256, 3), setup(0, 3, 512, 2), setup(1, 1, 128, 8)])
    evidence = testing.GMPipeLayoutEvidence(program([leaf]))
    result = evidence.layout()
    assert result["error"] == ""
    assert result["offsets"] == {(0, 3): 0, (1, 1): 2048, (3, 2): 3072}
    assert result["bytes"] == 3840


@pytest.mark.parametrize("count", [8, 32, 128])
def test_once_discovery_and_indexed_root_queries(count):
    leaves = [function(f"leaf_{i}", [setup(i, 1, 128, 2)]) for i in range(count)]
    root = function("root", [call(leaf.name) for leaf in leaves])
    evidence = testing.GMPipeLayoutEvidence(program([root, *leaves]))
    before = evidence.work()
    assert before["setup_entries"] == count
    assert before["call_edges"] == count
    assert before["setup_decodes"] == 0
    assert evidence.layout(root)["bytes"] == count * 256
    recovered = evidence.work()
    assert recovered["functions_visited"] == count + 1
    assert recovered["setup_incidences"] == count
    assert recovered["setup_decodes"] == count
    for leaf in leaves:
        assert evidence.layout(leaf)["bytes"] == 256
    after = evidence.work()
    assert after["statements"] == before["statements"]
    assert after["setup_decodes"] == count
    assert after["functions_visited"] == 2 * count + 1
    assert after["setup_incidences"] == 2 * count


def test_recursive_call_graph_reuses_original_function_inventory():
    first = function("first", [setup(), call("second")])
    second = function("second", [call("first")])
    evidence = testing.GMPipeLayoutEvidence(program([first, second]))
    assert evidence.layout(first)["bytes"] == 32768
    assert evidence.work()["functions_visited"] == 2
    assert evidence.work()["setup_incidences"] == 1


def test_shared_setup_statement_does_not_claim_unique_original_cut():
    cut = setup()
    leaf = function("leaf", [cut, cut])
    evidence = testing.GMPipeLayoutEvidence(program([leaf]))
    assert evidence.layout(leaf)["bytes"] == 32768
    assert not evidence.has_setup(leaf, cut)


def test_layout_overflow_refuses_without_partial_offsets():
    maximum = (1 << 31) - 1
    first = setup(0, 3, maximum, maximum)
    second = setup(1, 3, maximum, maximum)
    leaf = function("leaf", [first, second])
    evidence = testing.GMPipeLayoutEvidence(program([leaf]))
    result = evidence.layout(leaf)
    assert "overflow" in result["error"]
    assert result["bytes"] == 0 and result["offsets"] == {}
    work = evidence.work()
    assert evidence.layout(leaf) == result
    assert evidence.work()["setup_incidences"] == work["setup_incidences"]


def test_element_ceiling_does_not_overflow_at_large_valid_layout():
    maximum = (1 << 31) - 1
    leaf = function("leaf", [setup(0, 3, maximum, maximum)])
    evidence = testing.GMPipeLayoutEvidence(program([leaf]))
    result = evidence.layout(leaf)
    expected = 2 * maximum * maximum
    assert result["error"] == ""
    assert result["bytes"] == expected
    assert result["elements"] == (expected + 3) // 4


def test_invalid_direction_does_not_become_a_zero_sized_native_certificate():
    leaf = function("leaf", [setup(direction=7)])
    result = testing.GMPipeLayoutEvidence(program([leaf])).layout(leaf)
    assert "invalid dir_mask" in result["error"]
    assert result["bytes"] == 0 and result["offsets"] == {}


@pytest.mark.parametrize("scope", ["runtime", "hierarchy", "simt"])
def test_scope_inventory_is_geometry_only(scope):
    span = ir.Span.unknown()
    cut = setup()
    if scope == "runtime":
        body = ir.RuntimeScopeStmt(body=cut, span=span)
    elif scope == "hierarchy":
        body = ir.HierarchyScopeStmt(ir.Level.CHIP, body=cut, span=span)
    else:
        body = ir.SimtScopeStmt([ir.ConstInt(1, DataType.INT32, span)] * 3, cut, span)
    leaf = function("leaf", [body])
    evidence = testing.GMPipeLayoutEvidence(program([leaf]))
    assert evidence.layout(leaf)["bytes"] == 32768
    assert evidence.has_setup(leaf, cut)
    assert evidence.work()["setup_entries"] == 1
    # Layout discovery does not certify this execution context or native call.
    assert set(evidence.layout(leaf)) == {"bytes", "elements", "offsets", "error"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
