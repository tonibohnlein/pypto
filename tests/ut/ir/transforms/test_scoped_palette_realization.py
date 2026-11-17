# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assigned publication-rank maps checked on actual consuming command traces."""

import hashlib
import time
from typing import Any

import pytest
from pypto import codegen, ir, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_control_metrics import statements, static_control
from .sync_matmul_oracle import hardware_reference, interpret
from .sync_physical_oracle import PhysicalExecution, checked_payload_calls
from .test_collective_sync_realization import nested_collective
from .test_nested_phase_sync import nested_matrix_fixture
from .test_scoped_token_families import Uses
from .test_token_sync_lowering import _project

pytestmark = pytest.mark.usefixtures("ascend_backend")


def check_physical(logical, eligible, *, require_coverage=True, emitted=None):
    inventory = testing.inspect_scoped_token_families(logical)
    original = inventory["function"]
    _, calls = interpret(original)
    reference = hardware_reference(original, calls)
    tokens = Uses(calls, reference[0])
    tokens.execute(logical.body, {})
    _, expected = _project(tokens.vertices, tokens.edges, reference[0])
    order = testing.analyze_scoped_token_order(logical)
    lowered = emitted if emitted is not None else order.realize(eligible_ids=list(eligible))
    _, payloads = checked_payload_calls(original, lowered)
    flags = PhysicalExecution(payloads, reference[0], eligible)
    flags.execute(lowered.body, {})
    graph, actual = _project(flags.vertices, flags.edges, reference[0])
    assert flags.payload_count == len(calls)
    assert actual == expected
    if require_coverage:
        assert reference[2] <= actual
    assert not flags.live and not any(flags.pending.values())
    assert len(flags.records) == len(flags.waits)
    assert all(pair in graph for pair in flags.reuses)
    assert all(ids <= set(eligible) for ids in flags.ids.values())
    return lowered, flags


@pytest.mark.parametrize("count,visits", [(0, 2), (2, 0), (1, 1), (2, 1), (3, 2), (5, 3)])
def test_rank_palette_realization_matches_actual_tokens(count, visits):
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(count, visits=visits, slices=4, tile_columns=2), True
    ).realize()["function"]
    check_physical(logical, range(6))


def test_rank_palette_nondefault_directional_ids():
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(5, visits=2, slices=4, tile_columns=2), True
    ).realize()["function"]
    _, flags = check_physical(logical, [1, 2, 4, 5, 6, 7])
    assert flags.records
    assert all(0 not in ids and 3 not in ids for ids in flags.ids.values())


@pytest.mark.parametrize("count,visits,accepted", [(1, 1, True), (2, 2, False)])
def test_rank_palette_duplicate_uses_preserve_command_multiplicity(count, visits, accepted):
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(count, visits=visits, slices=2), True
    ).realize()["function"]

    class Duplicate(ir.IRMutator):
        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("sync.wait").name:
                return ir.SeqStmts([op, op], op.span)
            return op

    duplicated = Duplicate().visit_function(logical)
    if accepted:
        check_physical(duplicated, range(6))
    else:
        # Doubling this larger fixed protocol remains unproved by the current
        # sufficient resource interfaces. This is not a capacity lower bound.
        before = ir.serialize(duplicated)
        order = testing.analyze_scoped_token_order(duplicated)
        with pytest.raises(ValueError, match="Allocation unproved"):
            order.realize(eligible_ids=list(range(6)))
        assert ir.serialize(duplicated) == before


def test_rank_palette_refusal_is_transactional():
    logical = testing.analyze_placed_sync(
        nested_matrix_fixture(5, visits=2, slices=4, tile_columns=2), True
    ).realize()["function"]
    before = ir.serialize(logical)
    order = testing.analyze_scoped_token_order(logical)
    with pytest.raises(ValueError, match="eligible|capacity|allocation|Allocation|reservation"):
        order.realize(eligible_ids=[])
    assert ir.serialize(logical) == before
    with pytest.raises(ValueError, match="duplicate eligible"):
        order.realize(eligible_ids=[0, 0])
    assert ir.serialize(logical) == before
    check_physical(logical, range(6))


def test_rank_palette_cpp_generation(tmp_path):
    logical = testing.analyze_placed_sync(nested_collective(3, (2, 3)), True).realize()["function"]
    lowered, _ = check_physical(logical, range(6))
    source, output = tmp_path / "palette.pto", tmp_path / "palette.cpp"
    source.write_text(codegen.PTOCodegen().generate(ir.Program([lowered], "Palette", lowered.span)))
    _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync=true"])
    text = output.read_text()
    assert "set_flag(" in text and "wait_flag(" in text


def test_existing_constant_assignment_is_retained_before_palette():
    logical = testing.analyze_placed_sync(nested_matrix_fixture(3, visits=2, slices=2), True).realize()[
        "function"
    ]
    order = testing.analyze_scoped_token_order(logical)
    legacy = order.realize_legacy(eligible_ids=list(range(6)))
    candidate = order.realize(eligible_ids=list(range(6)))
    ir.assert_structural_equal(legacy["function"], candidate)
    public = testing.lower_sync_schedule(logical)
    assert all(not case["palette_attempts"] for case in public["physical_cases"])


def _collect_case_metrics(physical_cases):
    return [
        {
            "families": len(case["families"]),
            "reserved_lanes": case["reserved_lanes"],
            "legacy_attempts": case["allocation_attempts"],
            "palettes": case["palettes"],
            "palette_work": [
                {
                    key: attempt[key]
                    for key in (
                        "status",
                        "reason",
                        "attempted_budgets",
                        "publications",
                        "grammar_nodes",
                        "phase_visits",
                        "domain_visits",
                        "routing_entries",
                        "boundary_node_visits",
                        "proof_axis_entries",
                        "transition_queries",
                    )
                }
                for attempt in case["palette_attempts"]
            ],
            "endpoint_domains": case["endpoint_domain_entries"],
            "endpoint_predicates": case["endpoint_predicates_constructed"],
            "control_descriptors": case["control_descriptors"],
            "control_expression_visits": case["control_expression_visits"],
        }
        for case in physical_cases
    ]


def palette_scaling_sample(count, visits, slices) -> dict[str, Any]:
    """Inspect compact construction and emitted IR; never expand large visits.

    Slices is an inner-loop extent, not the number of static payload sites.
    The same modeled geometry/initialization used by the bounded oracle is
    retained. These synthetic inputs do not qualify native instruction layouts.
    """
    original = nested_matrix_fixture(count, visits=visits, slices=slices, m=128, n=256, tile_columns=2)
    row: dict[str, Any] = {
        "k_count": count,
        "outer_visits": visits,
        "inner_slice_extent": slices,
        "original_ir_sha256": hashlib.sha256(ir.serialize(original)).hexdigest(),
        "original_static_control": dict(static_control(original)),
        "original_statement_occurrences": sum(1 for _ in statements(original.body)),
        "layout_scope": "synthetic modeled accesses; no native layout or device qualification",
        "pool": "unchanged native catalog for ordinary LowerSync; six IDs checked separately",
    }
    row["stage"] = "analysis"
    start = time.perf_counter()
    try:
        analysis = testing.analyze_placed_sync(original, True)
        row["analysis_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        row["stage"] = "logical insertion"
        logical = analysis.realize()["function"]
        row["logical_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        row["stage"] = "physical lowering"
        production = testing.lower_sync_schedule(logical)
        row["production_lowering_seconds"] = time.perf_counter() - start
    except ValueError as error:
        row["status"] = "refused"
        row["reason"] = str(error)
        return row
    row["status"] = "allocated"
    emitted = production["function"]
    row["emitted_static_control"] = dict(static_control(emitted))
    row["emitted_statement_occurrences"] = sum(1 for _ in statements(emitted.body))
    row["cases"] = _collect_case_metrics(production["physical_cases"])
    # Explicitly labeled audit requests; these are not included in production
    # timing and do not change its ordinary incumbent-first route selection.
    order = testing.analyze_scoped_token_order(logical)
    start = time.perf_counter()
    try:
        order.realize_legacy(eligible_ids=list(range(6)))
        row["legacy_six_status"] = "allocated"
    except ValueError as error:
        row["legacy_six_status"] = "refused"
        row["legacy_six_reason"] = str(error)
    row["legacy_six_audit_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    try:
        six = testing.lower_sync_schedule(logical, eligible_ids=list(range(6)))
        row["six_status"] = "allocated"
        row["six_static_control"] = dict(static_control(six["function"]))
        row["six_cases"] = _collect_case_metrics(six["physical_cases"])
    except ValueError as error:
        row["six_status"] = "refused"
        row["six_reason"] = str(error)
    row["six_audit_seconds"] = time.perf_counter() - start
    return row


def test_emitted_palette_representation_is_compact_in_fixed_period_trip_counts():
    rows = [palette_scaling_sample(*parameters) for parameters in ((32, 4, 4), (32000, 4, 4), (32, 4000, 4))]
    for row in rows:
        assert row["status"] == "allocated", row
        assert row["six_status"] == "allocated", row
        assert row["legacy_six_status"] == "refused", row
        assert "unproved" in row["legacy_six_reason"].lower(), row
        assert not any(case["palettes"] for case in row["cases"]), row
        assert any(case["palettes"] for case in row["six_cases"]), row
    baseline = rows[0]
    for row in rows[1:]:
        assert row["emitted_static_control"] == baseline["emitted_static_control"]
        assert row["emitted_statement_occurrences"] == baseline["emitted_statement_occurrences"]
        assert row["six_static_control"] == baseline["six_static_control"]
        for key in ("cases", "six_cases"):
            assert len(row[key]) == len(baseline[key])
            for actual, expected in zip(row[key], baseline[key], strict=True):
                assert actual["families"] == expected["families"]
                assert actual["endpoint_domains"] == expected["endpoint_domains"]
                assert actual["endpoint_predicates"] == expected["endpoint_predicates"]
                assert actual["control_descriptors"] == expected["control_descriptors"]
                assert [(p["rank_pieces"], len(p["ids"])) for p in actual["palettes"]] == [
                    (p["rank_pieces"], len(p["ids"])) for p in expected["palettes"]
                ]
                assert len(actual["palette_work"]) == len(expected["palette_work"])
                for a, b in zip(actual["palette_work"], expected["palette_work"], strict=True):
                    for metric in ("grammar_nodes", "phase_visits", "domain_visits", "routing_entries"):
                        assert a[metric] == b[metric], (metric, a, b)
