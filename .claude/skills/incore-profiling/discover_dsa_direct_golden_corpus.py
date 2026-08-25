# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Discover DSA candidates from direct-golden PyPTO-Lib drivers.

The broad DSA exporter discovers compiler problems, not measurable kernels.
This tool reverses that funnel: it first proves that a model source has an
executable ``run_jit`` contract with a local JIT entry, tensor specifications,
and a direct golden. Exported DSA problems are joined only after that source
contract exists. An inventory from another PyPTO-Lib revision remains a stale
hint and never becomes a current candidate.
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

_CONTRACTS_PATH = Path(__file__).with_name("dsa_driver_contracts.py")
_CONTRACTS_SPEC = importlib.util.spec_from_file_location("_dsa_driver_contracts", _CONTRACTS_PATH)
if _CONTRACTS_SPEC is None or _CONTRACTS_SPEC.loader is None:
    raise ImportError(f"cannot load source-contract scanner: {_CONTRACTS_PATH}")
_CONTRACTS = importlib.util.module_from_spec(_CONTRACTS_SPEC)
sys.modules[_CONTRACTS_SPEC.name] = _CONTRACTS
_CONTRACTS_SPEC.loader.exec_module(_CONTRACTS)
DirectGoldenContract = _CONTRACTS.DirectGoldenContract
_inspect_source = _CONTRACTS.inspect_source

_PERFORMANCE_FIELD_FRAGMENTS = (
    "latency",
    "median_us",
    "runtime_us",
    "speedup",
    "timing",
    "winner",
)
_REQUIRED_CAPACITIES = frozenset({"native", "half", "q1", "tight"})
_REQUIRED_ARMS = frozenset({"geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg"})


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _read_tsv(path: Path, *, reject_performance_fields: bool = True) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if reject_performance_fields:
            leaked = [
                field
                for field in fieldnames
                if any(fragment in field.lower() for fragment in _PERFORMANCE_FIELD_FRAGMENTS)
            ]
            if leaked:
                raise ValueError(
                    f"timing-blind discovery rejects performance fields in {path}: {', '.join(leaked)}"
                )
        return [dict(row) for row in reader]


def _write_tsv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_catalog_fingerprints(path: Path) -> set[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    drivers: list[dict[str, Any]] = []
    base_name = document.get("extends")
    if base_name is not None:
        base = json.loads((path.parent / str(base_name)).read_text(encoding="utf-8"))
        excluded = set(document.get("exclude_driver_ids", []))
        drivers.extend(
            driver for driver in base.get("drivers", []) if driver.get("driver_id") not in excluded
        )
    drivers.extend(document.get("drivers", []))
    return {str(target["problem_fingerprint"]) for driver in drivers for target in driver.get("targets", [])}


def _load_prior_status(paths: Iterable[Path]) -> dict[str, str]:
    statuses: defaultdict[str, set[str]] = defaultdict(set)
    for path in paths:
        # Historical status is annotation only and never participates in
        # candidate membership or ordering.  It may therefore carry timing
        # metadata columns that the current export inventory forbids.
        for row in _read_tsv(path, reject_performance_fields=False):
            fingerprint = row.get("problem_fingerprint", "")
            status = row.get("status", "")
            if fingerprint and status:
                statuses[fingerprint].add(status)
    return {fingerprint: ";".join(sorted(values)) for fingerprint, values in statuses.items()}


def _load_export_status(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in _read_tsv(path, reject_performance_fields=False):
            script = row.get("script", "")
            if not script:
                continue
            prior = statuses.get(script)
            facts = {
                field: row.get(field, "")
                for field in ("status", "problems", "submit_sites", "emitted_kernels", "driver_mode")
            }
            if prior is not None and facts != prior:
                raise ValueError(f"conflicting export status for {script}")
            statuses[script] = facts
    return statuses


def _load_host_screen_status(paths: Iterable[Path]) -> dict[str, str]:
    cells: defaultdict[str, dict[tuple[str, str], dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    terminals: defaultdict[str, set[str]] = defaultdict(set)
    for path in paths:
        for row in _read_tsv(path, reject_performance_fields=False):
            tag = row.get("tag", "")
            if "-" not in tag:
                continue
            fingerprint = tag.rsplit("-", 1)[-1]
            arm = row.get("arm", "")
            if not arm:
                terminals[fingerprint].add(row.get("status", "UNKNOWN"))
                continue
            key = (row.get("pool_id", ""), row.get("capacity_label", ""))
            prior = cells[fingerprint][key].get(arm)
            status = row.get("status", "")
            if prior is not None and prior != status:
                raise ValueError(f"conflicting screen status for {fingerprint}, {key}, {arm}")
            cells[fingerprint][key][arm] = status

    statuses: dict[str, str] = {}
    for fingerprint in sorted(set(cells) | set(terminals)):
        if terminals[fingerprint]:
            statuses[fingerprint] = ";".join(sorted(terminals[fingerprint]))
            continue
        fingerprint_cells = cells[fingerprint]
        pools = {pool for pool, _capacity in fingerprint_cells}
        capacities_complete = all(
            {capacity for cell_pool, capacity in fingerprint_cells if cell_pool == pool}
            == _REQUIRED_CAPACITIES
            for pool in pools
        )
        arms_complete = all(
            set(arms) == _REQUIRED_ARMS and all(status == "feasible" for status in arms.values())
            for arms in fingerprint_cells.values()
        )
        statuses[fingerprint] = (
            "FOUR_ARM_FOUR_CAPACITY_FEASIBLE"
            if pools and capacities_complete and arms_complete
            else "INCOMPLETE_OR_INFEASIBLE_SCREEN"
        )
    return statuses


def _load_invocations(paths: Iterable[Path]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in _read_tsv(path):
            required = {"script", "instance", "problem_fingerprint", "buffers", "reuse_penalties"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path} is missing invocation columns: {', '.join(missing)}")
            key = (row["script"], row["instance"], row["problem_fingerprint"])
            prior = unique.get(key)
            if prior is not None:
                facts = ("buffers", "pools", "pool_names", "reuse_penalties", "recognizer")
                if any(row.get(field, "") != prior.get(field, "") for field in facts):
                    raise ValueError(f"conflicting invocation facts for {key}")
                continue
            unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _contract_state(
    has_inventory: bool,
    inventory_current: bool,
    script_invocations: Sequence[Mapping[str, str]],
    penalty_invocations: Sequence[Mapping[str, str]],
) -> str:
    if not has_inventory or not script_invocations:
        return "DIRECT_GOLDEN_NEEDS_CURRENT_EXPORT"
    if not inventory_current:
        return "DIRECT_GOLDEN_STALE_EXPORT"
    if not penalty_invocations:
        return "DIRECT_GOLDEN_ZERO_PENALTY_CONTROL"
    return "DIRECT_GOLDEN_CURRENT_EXPORT"


def _measurement_unit(unique_binding: bool, script_problem_count: int, driver_mode: str) -> str:
    if not unique_binding:
        return "AMBIGUOUS_CONTRACT"
    if driver_mode == "SINGLE_SUBMIT_SITE_CANDIDATE":
        return "SINGLE_KERNEL_DRIVER" if script_problem_count == 1 else "COMPLETE_MIXED_GROUP"
    if driver_mode == "MULTI_SUBMIT_PARENT":
        return "PARENT_WIDE_POLICY"
    if driver_mode in {"NO_EMITTED_SUBMIT", "NO_SOURCE"}:
        return "NOT_MEASURABLE"
    return "UNKNOWN"


def _next_gate(unique_binding: bool, inventory_current: bool, measurement_unit: str) -> str:
    if not unique_binding:
        return "MANUAL_CONTRACT_BINDING"
    if not inventory_current:
        return "CURRENT_EXPORT"
    if measurement_unit == "NOT_MEASURABLE":
        return "SOURCE_EXECUTION_REPAIR"
    if measurement_unit == "UNKNOWN":
        return "EXPORT_STATUS_REQUIRED"
    return "FOUR_ARM_HOST_SCREEN"


def discover(
    pypto_lib_root: str | Path,
    invocation_paths: Sequence[str | Path],
    inventory_revision: str | None,
    catalog_paths: Sequence[str | Path],
    prior_status_paths: Sequence[str | Path],
    output_root: str | Path,
    export_status_paths: Sequence[str | Path] = (),
    screen_result_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Create a timing-blind driver-first candidate inventory."""
    lib_root = Path(pypto_lib_root).resolve()
    output = Path(output_root)
    source_revision = _git_head(lib_root)
    if invocation_paths and inventory_revision is None:
        raise ValueError("--inventory-revision is required when --invocations is provided")
    inventory_current = inventory_revision == source_revision if invocation_paths else False
    invocations = _load_invocations(Path(path) for path in invocation_paths)
    invocations_by_script: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for invocation in invocations:
        invocations_by_script[invocation["script"]].append(invocation)

    catalog_fingerprints: set[str] = set()
    for path in catalog_paths:
        catalog_fingerprints.update(_load_catalog_fingerprints(Path(path)))
    prior_status = _load_prior_status(Path(path) for path in prior_status_paths)
    export_status = _load_export_status(Path(path) for path in export_status_paths)
    host_screen_status = _load_host_screen_status(Path(path) for path in screen_result_paths)

    contracts: list[DirectGoldenContract] = []
    rejection_rows: list[dict[str, Any]] = []
    for path in sorted((lib_root / "models").rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        discovered, rejections = _inspect_source(lib_root, path)
        contracts.extend(discovered)
        if rejections:
            rejection_rows.append(
                {
                    "script": path.relative_to(lib_root).as_posix(),
                    "reasons": ";".join(rejections),
                }
            )

    contracts = list(
        {
            (contract.script, contract.entry, contract.specs_call, contract.golden): contract
            for contract in contracts
        }.values()
    )
    contracts_by_script: defaultdict[str, list[DirectGoldenContract]] = defaultdict(list)
    for contract in contracts:
        contracts_by_script[contract.script].append(contract)

    contract_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for contract in sorted(contracts, key=lambda item: (item.script, item.line, item.entry)):
        script_invocations = invocations_by_script.get(contract.script, [])
        penalty_invocations = [row for row in script_invocations if int(row["reuse_penalties"]) > 0]
        state = _contract_state(
            bool(invocation_paths), inventory_current, script_invocations, penalty_invocations
        )
        contract_rows.append(
            {
                **asdict(contract),
                "dsa_problem_count": len(script_invocations),
                "penalty_problem_count": len(penalty_invocations),
                "state": state,
            }
        )

    for script, script_invocations in sorted(invocations_by_script.items()):
        script_contracts = sorted(
            contracts_by_script.get(script, []),
            key=lambda item: (item.line, item.entry, item.specs_call, item.golden),
        )
        if not script_contracts:
            continue
        unique_binding = len(script_contracts) == 1
        contract = script_contracts[0] if unique_binding else None
        status = export_status.get(script, {})
        measurement_unit = _measurement_unit(
            unique_binding,
            len(script_invocations),
            status.get("driver_mode", ""),
        )
        for invocation in script_invocations:
            if int(invocation["reuse_penalties"]) <= 0:
                continue
            fingerprint = invocation["problem_fingerprint"]
            candidate_rows.append(
                {
                    "script": script,
                    "contract_binding": "UNIQUE" if unique_binding else "AMBIGUOUS",
                    "contract_line": contract.line if contract is not None else "",
                    "entry": contract.entry if contract is not None else "",
                    "specs_call": contract.specs_call if contract is not None else "",
                    "golden": contract.golden if contract is not None else "",
                    "export_status": status.get("status", ""),
                    "submit_sites": status.get("submit_sites", ""),
                    "emitted_kernels": status.get("emitted_kernels", ""),
                    "measurement_unit": measurement_unit,
                    "instance": invocation["instance"],
                    "problem_fingerprint": fingerprint,
                    "base_problem_id": fingerprint,
                    "tiling_id": "canonical_unverified",
                    "buffers": invocation["buffers"],
                    "pools": invocation.get("pools", ""),
                    "pool_names": invocation.get("pool_names", ""),
                    "reuse_penalties": invocation["reuse_penalties"],
                    "recognizer": invocation.get("recognizer", ""),
                    "inventory_state": "CURRENT" if inventory_current else "STALE",
                    "existing_catalog": "YES" if fingerprint in catalog_fingerprints else "NO",
                    "prior_device_status": prior_status.get(fingerprint, "UNTESTED"),
                    "host_screen_status": host_screen_status.get(fingerprint, "NOT_SCREENED"),
                    "next_gate": _next_gate(unique_binding, inventory_current, measurement_unit),
                }
            )

    output.mkdir(parents=True, exist_ok=False)
    contract_columns = (
        [*asdict(contracts[0]), "dsa_problem_count", "penalty_problem_count", "state"]
        if contracts
        else [
            "script",
            "line",
            "entry",
            "specs",
            "specs_call",
            "golden",
            "source_sha256",
            "dsa_problem_count",
            "penalty_problem_count",
            "state",
        ]
    )
    candidate_columns = [
        "script",
        "contract_binding",
        "contract_line",
        "entry",
        "specs_call",
        "golden",
        "export_status",
        "submit_sites",
        "emitted_kernels",
        "measurement_unit",
        "instance",
        "problem_fingerprint",
        "base_problem_id",
        "tiling_id",
        "buffers",
        "pools",
        "pool_names",
        "reuse_penalties",
        "recognizer",
        "inventory_state",
        "existing_catalog",
        "prior_device_status",
        "host_screen_status",
        "next_gate",
    ]
    _write_tsv(output / "driver-contracts.tsv", contract_columns, contract_rows)
    _write_tsv(output / "candidate-problems.tsv", candidate_columns, candidate_rows)
    _write_tsv(output / "source-rejections.tsv", ("script", "reasons"), rejection_rows)

    workloads: list[dict[str, Any]] = []
    candidates_by_script: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        candidates_by_script[str(candidate["script"])].append(candidate)
    for script, script_candidates in sorted(candidates_by_script.items()):
        measurement_units = {str(row["measurement_unit"]) for row in script_candidates}
        if len(measurement_units) != 1:
            raise ValueError(f"conflicting measurement units for {script}")
        measurement_unit = measurement_units.pop()
        blockers = [
            f"{row['instance']}:{row['host_screen_status']}"
            for row in script_candidates
            if row["host_screen_status"] != "FOUR_ARM_FOUR_CAPACITY_FEASIBLE"
        ]
        status = (
            "FOUR_CAPACITY_HOST_READY"
            if not blockers and measurement_unit != "NOT_MEASURABLE"
            else "NOT_READY"
        )
        workloads.append(
            {
                "script": script,
                "measurement_unit": measurement_unit,
                "penalty_problem_count": len(script_candidates),
                "unique_base_problem_count": len({row["base_problem_id"] for row in script_candidates}),
                "host_ready_problem_count": sum(
                    row["host_screen_status"] == "FOUR_ARM_FOUR_CAPACITY_FEASIBLE"
                    for row in script_candidates
                ),
                "status": status,
                "blockers": ";".join(blockers),
            }
        )
    workload_columns = [
        "script",
        "measurement_unit",
        "penalty_problem_count",
        "unique_base_problem_count",
        "host_ready_problem_count",
        "status",
        "blockers",
    ]
    _write_tsv(output / "workload-status.tsv", workload_columns, workloads)

    summary = {
        "schema_version": 1,
        "verdict": "DRIVER_FIRST_DISCOVERY_COMPLETE",
        "pypto_lib_revision": source_revision,
        "inventory_revision": inventory_revision,
        "inventory_current": inventory_current,
        "source_files_with_direct_golden_contract": len({row["script"] for row in contract_rows}),
        "direct_golden_contracts": len(contract_rows),
        "candidate_rows": len(candidate_rows),
        "unique_candidate_problems": len({row["problem_fingerprint"] for row in candidate_rows}),
        "new_candidate_problems": len(
            {row["problem_fingerprint"] for row in candidate_rows if row["existing_catalog"] == "NO"}
        ),
        "contract_states": dict(sorted(Counter(row["state"] for row in contract_rows).items())),
        "candidate_next_gates": dict(sorted(Counter(row["next_gate"] for row in candidate_rows).items())),
        "measurement_units": dict(sorted(Counter(row["measurement_unit"] for row in candidate_rows).items())),
        "host_ready_workloads": sum(row["status"] == "FOUR_CAPACITY_HOST_READY" for row in workloads),
        "timing_blind": True,
        "tiling_policy": "explicit_variants_required; canonical_unverified is not a frozen tiling identity",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    """Run the driver-first discovery CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-lib-root", required=True)
    parser.add_argument("--invocations", action="append", default=[])
    parser.add_argument("--inventory-revision")
    parser.add_argument("--catalog", action="append", default=[])
    parser.add_argument("--prior-status", action="append", default=[])
    parser.add_argument("--export-status", action="append", default=[])
    parser.add_argument("--screen-results", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    summary = discover(
        arguments.pypto_lib_root,
        arguments.invocations,
        arguments.inventory_revision,
        arguments.catalog,
        arguments.prior_status,
        arguments.output_root,
        arguments.export_status,
        arguments.screen_results,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
