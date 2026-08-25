# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for driver-first, timing-blind DSA corpus discovery."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / ".claude"
    / "skills"
    / "incore-profiling"
    / "discover_dsa_direct_golden_corpus.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_discover_dsa_direct_golden_corpus", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
discovery = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = discovery
_SPEC.loader.exec_module(discovery)


def _write_driver(path: Path, *, second_contract: bool = False, decorator: str = "pl.jit") -> None:
    source = f"""import pypto.language as pl

@{decorator}
def program():
    pass

def golden(tensors):
    return tensors

def build_tensor_specs():
    return []
"""
    if second_contract:
        source += """
@pl.jit
def other_program():
    pass

def other_golden(tensors):
    return tensors

def other_specs():
    return []
"""
    source += """
if __name__ == "__main__":
    from golden import run_jit
    run_jit(fn=program, specs=build_tensor_specs(), golden_fn=golden)
"""
    if second_contract:
        source += "    run_jit(fn=other_program, specs=other_specs(), golden_fn=other_golden)\n"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def _write_invocations(path: Path, *, performance_column: bool = False) -> None:
    columns = [
        "script",
        "instance",
        "problem_fingerprint",
        "buffers",
        "pools",
        "pool_names",
        "reuse_penalties",
        "recognizer",
    ]
    if performance_column:
        columns.append("runtime_us")
    row = {
        "script": "models/toy/direct.py",
        "instance": "kernel",
        "problem_fingerprint": "0123456789abcdef",
        "buffers": "12",
        "pools": "1",
        "pool_names": "Vec",
        "reuse_penalties": "7",
        "recognizer": "quadratic_route_frontier_v3",
        "runtime_us": "42",
    }
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({column: row[column] for column in columns})


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def _write_export_status(path: Path, *, mode: str = "SINGLE_SUBMIT_SITE_CANDIDATE") -> None:
    path.write_text(
        "script\tstatus\tproblems\tsubmit_sites\temitted_kernels\tdriver_mode\n"
        f"models/toy/direct.py\tEXPORTED\t1\t1\t1\t{mode}\n",
        encoding="utf-8",
    )


def _write_screen_results(path: Path) -> None:
    columns = ["tag", "pool_id", "capacity_label", "arm", "status", "runtime_us"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for capacity in ("native", "half", "q1", "tight"):
            for arm in ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg"):
                writer.writerow(
                    {
                        "tag": "toy__kernel-0123456789abcdef",
                        "pool_id": "1",
                        "capacity_label": capacity,
                        "arm": arm,
                        "status": "feasible",
                        "runtime_us": "1",
                    }
                )


def test_current_inventory_yields_one_four_arm_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "pypto-lib"
    _write_driver(library / "models" / "toy" / "direct.py")
    invocations = tmp_path / "invocations.tsv"
    _write_invocations(invocations)
    export_status = tmp_path / "export-status.tsv"
    _write_export_status(export_status)
    screen_results = tmp_path / "screen-results.tsv"
    _write_screen_results(screen_results)
    monkeypatch.setattr(discovery, "_git_head", lambda _root: "current-revision")

    summary = discovery.discover(
        library,
        [invocations],
        "current-revision",
        [],
        [],
        tmp_path / "out",
        [export_status],
        [screen_results],
    )
    candidates = _read_rows(tmp_path / "out" / "candidate-problems.tsv")

    assert summary["unique_candidate_problems"] == 1
    assert summary["host_ready_workloads"] == 1
    assert candidates[0]["contract_binding"] == "UNIQUE"
    assert candidates[0]["base_problem_id"] == "0123456789abcdef"
    assert candidates[0]["tiling_id"] == "canonical_unverified"
    assert candidates[0]["measurement_unit"] == "SINGLE_KERNEL_DRIVER"
    assert candidates[0]["host_screen_status"] == "FOUR_ARM_FOUR_CAPACITY_FEASIBLE"
    assert candidates[0]["next_gate"] == "FOUR_ARM_HOST_SCREEN"
    workloads = _read_rows(tmp_path / "out" / "workload-status.tsv")
    assert workloads[0]["status"] == "FOUR_CAPACITY_HOST_READY"


def test_stale_inventory_is_only_a_reexport_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "pypto-lib"
    _write_driver(library / "models" / "toy" / "direct.py")
    invocations = tmp_path / "invocations.tsv"
    _write_invocations(invocations)
    monkeypatch.setattr(discovery, "_git_head", lambda _root: "current-revision")

    discovery.discover(library, [invocations], "old-revision", [], [], tmp_path / "out")
    candidate = _read_rows(tmp_path / "out" / "candidate-problems.tsv")[0]

    assert candidate["inventory_state"] == "STALE"
    assert candidate["next_gate"] == "CURRENT_EXPORT"


def test_multiple_launch_contracts_fail_closed_without_cross_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "pypto-lib"
    _write_driver(library / "models" / "toy" / "direct.py", second_contract=True)
    invocations = tmp_path / "invocations.tsv"
    _write_invocations(invocations)
    monkeypatch.setattr(discovery, "_git_head", lambda _root: "current-revision")

    summary = discovery.discover(
        library,
        [invocations],
        "current-revision",
        [],
        [],
        tmp_path / "out",
    )
    candidates = _read_rows(tmp_path / "out" / "candidate-problems.tsv")

    assert summary["candidate_rows"] == 1
    assert candidates[0]["contract_binding"] == "AMBIGUOUS"
    assert candidates[0]["entry"] == ""
    assert candidates[0]["next_gate"] == "MANUAL_CONTRACT_BINDING"


def test_rejects_performance_fields_in_discovery_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "pypto-lib"
    _write_driver(library / "models" / "toy" / "direct.py")
    invocations = tmp_path / "invocations.tsv"
    _write_invocations(invocations, performance_column=True)
    monkeypatch.setattr(discovery, "_git_head", lambda _root: "current-revision")

    with pytest.raises(ValueError, match="timing-blind discovery rejects performance fields"):
        discovery.discover(
            library,
            [invocations],
            "current-revision",
            [],
            [],
            tmp_path / "out",
        )


def test_historical_timing_metadata_is_annotation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "pypto-lib"
    _write_driver(library / "models" / "toy" / "direct.py")
    invocations = tmp_path / "invocations.tsv"
    _write_invocations(invocations)
    status = tmp_path / "problem-status.tsv"
    status.write_text(
        "problem_fingerprint\tstatus\ttiming_scope\n0123456789abcdef\tMEASURED\tselected_task\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(discovery, "_git_head", lambda _root: "current-revision")

    discovery.discover(
        library,
        [invocations],
        "current-revision",
        [],
        [status],
        tmp_path / "out",
    )
    candidate = _read_rows(tmp_path / "out" / "candidate-problems.tsv")[0]

    assert candidate["prior_device_status"] == "MEASURED"


def test_measurement_unit_distinguishes_parent_and_mixed_group() -> None:
    assert discovery._measurement_unit(True, 1, "SINGLE_SUBMIT_SITE_CANDIDATE") == "SINGLE_KERNEL_DRIVER"
    assert discovery._measurement_unit(True, 3, "SINGLE_SUBMIT_SITE_CANDIDATE") == "COMPLETE_MIXED_GROUP"
    assert discovery._measurement_unit(True, 5, "MULTI_SUBMIT_PARENT") == "PARENT_WIDE_POLICY"
    assert discovery._next_gate(True, True, "UNKNOWN") == "EXPORT_STATUS_REQUIRED"


def test_host_jit_entry_is_a_direct_driver_contract(tmp_path: Path) -> None:
    library = tmp_path / "pypto-lib"
    source = library / "models" / "toy" / "direct.py"
    _write_driver(source, decorator="pl.jit.host")

    contracts, rejections = discovery._inspect_source(library, source)

    assert len(contracts) == 1
    assert contracts[0].entry == "program"
    assert rejections == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
