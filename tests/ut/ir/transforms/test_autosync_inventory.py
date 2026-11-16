# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Whole-model receipts preserve unresolved domains and named qualification."""

import hashlib
import json

import pytest
from pypto import ir, passes
from pypto.backend.pto_backend import _publish_sync_execution_domains

from .qualify_qwen_whole_model import (
    domain_ownership,
    expected_ownership_failures,
    function_inventory,
    qualification_context,
)
from .test_conservative_sync_dataflow import conservative_fixture

pytestmark = pytest.mark.usefixtures("ascend_backend")


def receipt(owner, *, complete=True, report=True, generated=True):
    original = {"name": "leaf", "synchronization_owner": None, "requires_runtime_binding": False}
    external = {"name": "external", "synchronization_owner": None, "requires_runtime_binding": True}
    source = {
        "functions_before_sync": [original, external],
        "functions_after_buffer_selection": [original, external],
        "functions_after_sync": [{**original, "synchronization_owner": owner}, external],
        "compile_success": complete,
        "policy": "conservative_prefix",
    }
    reports = [{"function": "leaf", "selected_route": "pypto"}] if report else []
    return domain_ownership(source, reports, {"leaf"} if generated else set())


def test_named_ownership_does_not_hide_other_domains():
    rows = receipt(True)
    assert {row["function"] for row in rows} == {"leaf", "external"}
    assert expected_ownership_failures(rows, ["leaf"]) == []
    assert expected_ownership_failures(rows, ["external", "absent", "external"]) == [
        {"function": "absent", "status": "not_observed"},
        {"function": "external", "status": "unverified_domain"},
    ]
    assert all(row["requested_policy"] == "conservative_prefix" for row in rows)


@pytest.mark.parametrize(
    "owner,options",
    [
        (False, {}),
        (None, {}),
        ("True", {}),
        (True, {"complete": False}),
        (True, {"report": False}),
        (True, {"generated": False}),
    ],
)
def test_expected_ownership_needs_actual_owner_report_output_and_success(owner, options):
    failures = expected_ownership_failures(receipt(owner, **options), ["leaf"])
    assert len(failures) == 1 and failures[0]["function"] == "leaf"
    assert failures[0]["status"] == ("delegated" if owner is False else "unverified_domain")


@pytest.mark.parametrize("kind", ["for", "while"])
def test_inventory_counts_control_without_executing_parameterized_domains(kind):
    function = conservative_fixture(kind)
    original = ir.serialize(function)
    rows = function_inventory(ir.Program([function], "Inventory", function.span))
    assert len(rows) == 1
    row = rows[0]
    assert row["synchronization_owner"] is None
    assert row["buffer_admission"]["status"] == "not_evaluated"
    counts = row["static_control"]
    assert counts.get("while", 0) == (kind == "while")
    assert sum(count for name, count in counts.items() if name.startswith("for:")) == (
        1 if kind == "while" else 2
    )
    assert counts["if"] == 2
    assert counts["call:buffer.load"] == 2
    assert counts["call:buffer.abs"] == 2
    assert counts["call:buffer.store"] == 2
    assert any(name.startswith("call:") for name in counts)
    assert ir.serialize(function) == original


@pytest.mark.parametrize(
    "kind",
    [ir.FunctionType.Orchestration, ir.FunctionType.Graph, ir.FunctionType.Spmd, ir.FunctionType.Group],
)
def test_task_control_report_requires_actual_emission_evidence(kind):
    span = ir.Span.unknown()
    control = ir.Function("control", [], [], ir.ReturnStmt([], span), span, kind)
    unused = ir.Function("unused", [], [], ir.ReturnStmt([], span), span, ir.FunctionType.Spmd)
    program = ir.Program([control, unused], "Owners", span)
    artifact = "orchestration/main.cpp"
    files = {"report/autosync.json": json.dumps({"kernels": []}), artifact: "original emitted code"}
    _publish_sync_execution_domains(program, files, {"control": kind}, artifact)
    rows = {row["function"]: row for row in json.loads(files["report/autosync.json"])["execution_domains"]}
    assert rows["control"]["status"] == "task_control_owner_observed"
    assert rows["control"]["owner"] == "runtime_task_codegen"
    assert "no independent control, visibility or peer completion" in rows["control"]["guarantee"]
    assert rows["unused"]["owner"] is None
    source = {"name": "control", "type": str(kind), "synchronization_owner": None}
    receipt = {
        "functions_before_sync": [source],
        "functions_after_buffer_selection": [source],
        "functions_after_sync": [source],
        "compile_success": True,
        "policy": "conservative_prefix",
    }
    joined = domain_ownership(receipt, [], set(), rows.values())
    assert joined[0]["status"] == "task_control_owner_observed"
    assert not joined[0]["verified_pypto_leaf"]
    assert expected_ownership_failures(joined, ["control"]), "task owner cannot satisfy a local-leaf gate"
    assert files[artifact] == "original emitted code"


@pytest.mark.parametrize("fault", ["missing_artifact", "unknown_name", "wrong_type", "local_type"])
def test_task_control_evidence_cannot_hide_missing_or_local_domains(fault):
    span = ir.Span.unknown()
    kind = ir.FunctionType.AIV if fault == "local_type" else ir.FunctionType.Orchestration
    function = ir.Function("control", [], [], ir.ReturnStmt([], span), span, kind)
    program = ir.Program([function], "Owners", span)
    artifact = "orchestration/main.cpp"
    files = {"report/autosync.json": json.dumps({"kernels": []})}
    if fault != "missing_artifact":
        files[artifact] = "emitted"
    evidence = {
        "other" if fault == "unknown_name" else "control": ir.FunctionType.Group
        if fault == "wrong_type"
        else kind
    }
    before = dict(files)
    with pytest.raises(ValueError, match="Task-control"):
        _publish_sync_execution_domains(program, files, evidence, artifact)
    assert files == before


def test_default_off_does_not_publish_task_ownership_or_change_artifacts():
    function = conservative_fixture()
    program = ir.Program([function], "NoReport", function.span)
    files = {"kernel.cpp": "unchanged"}
    _publish_sync_execution_domains(program, files, {}, None)
    assert files == {"kernel.cpp": "unchanged"}


@pytest.mark.parametrize("policy", ["exact", "conservative_prefix"])
@pytest.mark.parametrize("alias_policy", ["may-alias", "disjoint"])
@pytest.mark.parametrize("scalar_gm_dma", [False, True])
def test_qualification_alias_policy_and_scalar_implementation_are_independent(
    policy, alias_policy, scalar_gm_dma, tmp_path
):
    promise = "Caller guarantees distinct original external GM allocations do not overlap.\n"
    contract = tmp_path / "caller-contract.md"
    contract.write_text(promise)
    options, record = qualification_context(
        policy,
        gm_alias_policy=alias_policy,
        gm_contract=contract if alias_policy == "disjoint" else None,
        scalar_gm_dma=scalar_gm_dma,
    )
    # Exercise the same options dictionary passed by the actual qualifier.
    # Exact policy must not grant alias assumptions, and prefix policy must
    # not discard a caller promise explicitly supplied to it.
    with passes.PassContext([], **options) as context:
        assert context.get_autosync_policy() == policy
        assert context.get_autosync_disjoint_gm() == (alias_policy == "disjoint")
        assert context.get_enable_scalar_gm_dma() is scalar_gm_dma
    assert record["gm_alias_policy"] == alias_policy
    assert record["autosync_disjoint_gm"] == options["autosync_disjoint_gm"]
    assert "never inferred from SSA names" in record["qualification"]
    if alias_policy == "disjoint":
        assert record["provenance"] == {
            "kind": "explicit_caller_promise",
            "path": str(contract.resolve()),
            "sha256": hashlib.sha256(promise.encode()).hexdigest(),
            "promise": promise,
        }
        assert "same-root overlaps" in record["scope"]
        assert "Unresolved roots" in record["scope"]
        contract.write_text("changed after request")
        assert record["provenance"]["promise"] == promise
    else:
        assert record["provenance"] is None


@pytest.mark.parametrize("policy", ["exact", "conservative_prefix"])
def test_qualification_defaults_to_may_alias_and_native_scalar(policy):
    options, record = qualification_context(policy)
    assert options == {
        "autosync_policy": policy,
        "autosync_disjoint_gm": False,
        "enable_scalar_gm_dma": False,
    }
    assert record["provenance"] is None


@pytest.mark.parametrize("fault", ["absent", "empty", "whitespace", "missing", "directory", "non_utf8"])
@pytest.mark.parametrize("policy", ["exact", "conservative_prefix"])
def test_disjoint_qualification_requires_pinned_nonempty_caller_provenance(policy, fault, tmp_path):
    contract = tmp_path / "contract.md"
    if fault == "empty":
        contract.write_text("")
    elif fault == "whitespace":
        contract.write_text(" \n\t")
    elif fault == "directory":
        contract.mkdir()
    elif fault == "non_utf8":
        contract.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="[Cc]ontract"):
        qualification_context(
            policy, gm_alias_policy="disjoint", gm_contract=None if fault == "absent" else contract
        )


def test_qualification_does_not_silently_ignore_contract_or_unknown_policy(tmp_path):
    contract = tmp_path / "caller-contract.md"
    contract.write_text("Distinct external GM roots are disjoint.")
    with pytest.raises(ValueError, match="requires --gm-alias-policy=disjoint"):
        qualification_context("exact", gm_contract=contract)
    with pytest.raises(ValueError, match="Unknown autosync policy"):
        qualification_context("unknown")
    with pytest.raises(ValueError, match="Unknown GM alias policy"):
        qualification_context("exact", gm_alias_policy="unknown")
