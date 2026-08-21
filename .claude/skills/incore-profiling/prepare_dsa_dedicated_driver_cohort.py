# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Freeze DSA cases hosted by deterministic, golden-backed PyPTO-Lib drivers.

This tool deliberately separates a *functional driver* from the DSA functions
outlined inside it.  The driver supplies real model shapes, deterministic input
construction, and an end-to-end Torch golden.  A selected DSA function is the
per-task timing unit.  A device campaign must still apply one arm to every DSA
function in the driver and pass the driver's complete golden before using that
timing.

Selection is timing-blind.  Only source contracts, problem identity, and
four-arm/four-capacity host feasibility are consulted.
"""

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("geometry_ff", "geometry_cg", "cypress", "dsa_rp_cg")
CAPACITIES = ("native", "half", "q1", "tight")
_SCREEN_SEMANTIC_FIELDS = (
    "tier",
    "driver_id",
    "instance",
    "problem_fingerprint",
    "pool_id",
    "pool",
    "capacity",
    "capacity_bytes",
    "arm",
    "status",
    "placement_sha256",
)
_REQUIRED_DRIVER_FIELDS = ("driver_id", "script", "entry", "golden", "specs", "argv", "targets")
_REQUIRED_TARGET_FIELDS = ("instance", "problem_fingerprint", "pool_id", "operation_class")
_MEASUREMENT_CONTRACT = {
    "placement_scope": "all_dsa_functions_in_driver",
    "correctness_scope": "complete_driver_torch_golden",
    "timing_scope": "selected_runtime_task",
    "allow_whole_driver_as_kernel_timing": False,
    "requires_runtime_capture": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(value: Any) -> str:
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_catalog(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    catalog = json.loads(path.read_text())
    source_hashes = {"catalog_source_sha256": _sha256(path)}
    base_name = catalog.pop("extends", None)
    if base_name is None:
        source_hashes["catalog_semantics_sha256"] = _semantic_sha256(catalog)
        return catalog, source_hashes

    base_path = path.parent / str(base_name)
    base = json.loads(base_path.read_text())
    if "extends" in base:
        raise ValueError("nested catalog extensions are not supported")
    excluded = set(catalog.pop("exclude_driver_ids", []))
    base_driver_ids = {driver["driver_id"] for driver in base["drivers"]}
    unknown_exclusions = excluded - base_driver_ids
    if unknown_exclusions:
        raise ValueError(f"catalog excludes unknown drivers: {', '.join(sorted(unknown_exclusions))}")
    additions = catalog.pop("drivers", [])
    merged = dict(base)
    merged.update(catalog)
    merged["drivers"] = [driver for driver in base["drivers"] if driver["driver_id"] not in excluded]
    merged["drivers"].extend(additions)
    source_hashes.update(
        {
            "base_catalog_source_sha256": _sha256(base_path),
            "catalog_semantics_sha256": _semantic_sha256(merged),
        }
    )
    return merged, source_hashes


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return [dict(row) for row in csv.DictReader(source, delimiter="\t")]


def _write_tsv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _screen_semantics_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    semantic_rows = [{field: str(row[field]) for field in _SCREEN_SEMANTIC_FIELDS} for row in rows]
    semantic_rows.sort(key=lambda row: tuple(row[field] for field in _SCREEN_SEMANTIC_FIELDS))
    canonical = json.dumps(semantic_rows, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _decorator_names(function: ast.FunctionDef) -> set[str]:
    return {ast.unparse(decorator) for decorator in function.decorator_list}


def _source_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hints = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "name_hint"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    }
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    return hints | functions


def _source_contract(lib_root: Path, path: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{driver['driver_id']}: driver source does not exist: {path}")
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for field in ("entry", "golden", "specs"):
        name = str(driver[field])
        if name not in functions:
            raise ValueError(f"{driver['driver_id']}: {field} function {name!r} is absent from {path}")
    entry = functions[str(driver["entry"])]
    if not any(name.startswith("pl.jit") for name in _decorator_names(entry)):
        raise ValueError(f"{driver['driver_id']}: entry {entry.name!r} is not decorated with pl.jit")
    if not any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and "__name__" in ast.unparse(node.test)
        and "__main__" in ast.unparse(node.test)
        for node in tree.body
    ):
        raise ValueError(f"{driver['driver_id']}: {path} has no executable __main__ driver")

    definition_hashes: dict[str, str] = {}
    for target in driver["targets"]:
        definition_script = str(target.get("definition_script", driver["script"]))
        definition_path = lib_root / definition_script
        if not definition_path.is_file():
            raise ValueError(
                f"{driver['driver_id']}/{target['instance']}: definition source does not exist: "
                f"{definition_path}"
            )
        expected_hint = str(target.get("source_name_hint", target["instance"]))
        if expected_hint not in _source_symbols(definition_path):
            raise ValueError(
                f"{driver['driver_id']}/{target['instance']}: source symbol/name_hint {expected_hint!r} "
                f"is absent from {definition_script}"
            )
        definition_hashes[definition_script] = _sha256(definition_path)
    return {
        "source_sha256": _sha256(path),
        "definition_source_sha256": dict(sorted(definition_hashes.items())),
    }


def _validate_driver(driver: Mapping[str, Any], ids: set[str], cases: set[tuple[str, str]]) -> None:
    missing = sorted(set(_REQUIRED_DRIVER_FIELDS) - set(driver))
    if missing:
        raise ValueError(f"driver is missing fields: {', '.join(missing)}")
    driver_id = str(driver["driver_id"])
    if driver_id in ids:
        raise ValueError(f"duplicate driver_id: {driver_id}")
    ids.add(driver_id)
    if driver.get("tier") not in {"canary", "expanded", "canary_expanded"}:
        raise ValueError(f"{driver_id}: tier must be canary, expanded, or canary_expanded")
    if not isinstance(driver["argv"], list) or not all(isinstance(item, str) for item in driver["argv"]):
        raise ValueError(f"{driver_id}: argv must be a list of strings")
    if not driver["targets"]:
        raise ValueError(f"{driver_id}: targets must not be empty")
    for target in driver["targets"]:
        missing_target = sorted(set(_REQUIRED_TARGET_FIELDS) - set(target))
        if missing_target:
            raise ValueError(f"{driver_id}: target is missing fields: {', '.join(missing_target)}")
        key = (str(driver["script"]), str(target["problem_fingerprint"]))
        if key in cases:
            raise ValueError(f"duplicate selected problem: {key[0]} {key[1]}")
        cases.add(key)


def _validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema_version") != 1:
        raise ValueError("catalog schema_version must be 1")
    if catalog.get("timing_blind_selection") is not True:
        raise ValueError("catalog must assert timing_blind_selection=true")
    if tuple(catalog.get("arms", ())) != ARMS:
        raise ValueError(f"catalog arms must be {ARMS}")
    if tuple(catalog.get("capacities", ())) != CAPACITIES:
        raise ValueError(f"catalog capacities must be {CAPACITIES}")
    if catalog.get("measurement_contract") != _MEASUREMENT_CONTRACT:
        raise ValueError(f"catalog measurement_contract must be {_MEASUREMENT_CONTRACT}")
    if not isinstance(catalog.get("input_seed"), int) or catalog["input_seed"] < 0:
        raise ValueError("catalog input_seed must be a non-negative integer")
    drivers = catalog.get("drivers")
    if not isinstance(drivers, list) or not drivers:
        raise ValueError("catalog drivers must be a non-empty list")
    ids: set[str] = set()
    cases: set[tuple[str, str]] = set()
    for driver in drivers:
        _validate_driver(driver, ids, cases)


def prepare_cohort(
    catalog_path: str | Path,
    pypto_lib_root: str | Path,
    problem_inventory_path: str | Path,
    screen_results_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate and freeze a dedicated-driver DSA cohort."""
    catalog_file = Path(catalog_path)
    lib_root = Path(pypto_lib_root).resolve()
    inventory_file = Path(problem_inventory_path)
    screen_file = Path(screen_results_path)
    output = Path(output_directory)
    catalog, catalog_hashes = _load_catalog(catalog_file)
    _validate_catalog(catalog)

    actual_revision = _git_head(lib_root)
    expected_revision = str(catalog["pypto_lib_revision"])
    if actual_revision != expected_revision:
        raise ValueError(
            f"PyPTO-Lib revision mismatch: catalog={expected_revision}, checkout={actual_revision}"
        )

    inventory_rows = _read_tsv(inventory_file)
    problems_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in inventory_rows:
        key = (row["script"], row["instance"], row["problem_fingerprint"])
        prior = problems_by_key.get(key)
        if prior is not None:
            factual_columns = ("buffers", "pools", "pool_names", "reuse_penalties", "recognizer")
            if any(row[column] != prior[column] for column in factual_columns):
                raise ValueError(f"inconsistent repeated problem-inventory row: {key}")
            continue
        problems_by_key[key] = row

    screen_rows = _read_tsv(screen_file)
    screen_by_cell: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in screen_rows:
        fingerprint = row["tag"].rsplit("-", 1)[-1]
        screen_by_cell[(fingerprint, row["pool_id"], row["capacity_label"], row["arm"])].append(row)

    output.mkdir(parents=True, exist_ok=True)
    driver_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    frozen_drivers: list[dict[str, Any]] = []

    for driver in catalog["drivers"]:
        source_path = lib_root / driver["script"]
        source = _source_contract(lib_root, source_path, driver)
        frozen_driver = dict(driver)
        frozen_driver["source_sha256"] = source["source_sha256"]
        frozen_driver["definition_source_sha256"] = source["definition_source_sha256"]
        frozen_drivers.append(frozen_driver)
        driver_rows.append(
            {
                "tier": driver["tier"],
                "driver_id": driver["driver_id"],
                "script": driver["script"],
                "entry": driver["entry"],
                "golden": driver["golden"],
                "specs": driver["specs"],
                "argv_json": json.dumps(driver["argv"], separators=(",", ":")),
                "target_count": len(driver["targets"]),
                "source_sha256": source["source_sha256"],
            }
        )
        for target in driver["targets"]:
            fingerprint = str(target["problem_fingerprint"])
            key = (str(driver["script"]), str(target["instance"]), fingerprint)
            if key not in problems_by_key:
                raise ValueError(
                    f"{driver['driver_id']}: selected problem is absent from the problem inventory: {key}"
                )
            problem = problems_by_key[key]
            if int(problem["reuse_penalties"]) <= 0:
                raise ValueError(
                    f"{driver['driver_id']}/{target['instance']}: selected problem has no penalties"
                )

            target_cells: list[dict[str, str]] = []
            for capacity in CAPACITIES:
                for arm in ARMS:
                    cell_key = (fingerprint[:16], str(target["pool_id"]), capacity, arm)
                    matches = screen_by_cell.get(cell_key, [])
                    if len(matches) != 1:
                        raise ValueError(
                            f"{driver['driver_id']}/{target['instance']}: expected one screen row for "
                            f"pool={target['pool_id']} capacity={capacity} arm={arm}, got {len(matches)}"
                        )
                    cell = matches[0]
                    if cell["status"].upper() != "FEASIBLE":
                        raise ValueError(
                            f"{driver['driver_id']}/{target['instance']}: infeasible screen cell "
                            f"pool={target['pool_id']} capacity={capacity} arm={arm}: {cell['status']}"
                        )
                    target_cells.append(cell)
                    cell_rows.append(
                        {
                            "tier": driver["tier"],
                            "driver_id": driver["driver_id"],
                            "instance": target["instance"],
                            "problem_fingerprint": fingerprint,
                            "pool_id": target["pool_id"],
                            "pool": cell["pool"],
                            "capacity": capacity,
                            "capacity_bytes": cell["capacity"],
                            "arm": arm,
                            "status": cell["status"].upper(),
                            "placement_sha256": cell["placement_sha256"],
                        }
                    )

            capacities = {
                cell["capacity_label"]: cell["capacity"]
                for cell in target_cells
                if cell["arm"] == "geometry_ff"
            }
            pool_names = {cell["pool"] for cell in target_cells}
            if len(pool_names) != 1:
                raise ValueError(
                    f"{driver['driver_id']}/{target['instance']}: pool name changed across cells"
                )
            problem_rows.append(
                {
                    "tier": driver["tier"],
                    "driver_id": driver["driver_id"],
                    "script": driver["script"],
                    "instance": target["instance"],
                    "problem_fingerprint": fingerprint,
                    "buffers": problem["buffers"],
                    "pools": problem["pools"],
                    "reuse_penalties": problem["reuse_penalties"],
                    "operation_class": target["operation_class"],
                    "pool_id": target["pool_id"],
                    "pool": next(iter(pool_names)),
                    "native_capacity": capacities["native"],
                    "half_capacity": capacities["half"],
                    "q1_capacity": capacities["q1"],
                    "tight_capacity": capacities["tight"],
                }
            )

    frozen = {
        "schema_version": 2,
        "selection_policy": catalog["selection_policy"],
        "timing_blind_selection": True,
        "pypto_lib_revision": actual_revision,
        "arms": list(ARMS),
        "capacities": list(CAPACITIES),
        "input_seed": catalog["input_seed"],
        "measurement_contract": dict(_MEASUREMENT_CONTRACT),
        "driver_count": len(driver_rows),
        "problem_count": len(problem_rows),
        "canary_driver_count": sum(row["tier"] in {"canary", "canary_expanded"} for row in driver_rows),
        "expanded_problem_count": sum(row["tier"] in {"expanded", "canary_expanded"} for row in problem_rows),
        "operation_classes": sorted({row["operation_class"] for row in problem_rows}),
        "inputs": {
            **catalog_hashes,
            "problem_inventory_sha256": _sha256(inventory_file),
            "screen_semantics_sha256": _screen_semantics_sha256(cell_rows),
            "screen_semantics_fields": list(_SCREEN_SEMANTIC_FIELDS),
        },
        "drivers": frozen_drivers,
    }
    frozen_path = output / "cohort-frozen.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    frozen_sha = _sha256(frozen_path)
    (output / "cohort-frozen.json.sha256").write_text(f"{frozen_sha}  cohort-frozen.json\n")

    _write_tsv(
        output / "drivers.tsv",
        (
            "tier",
            "driver_id",
            "script",
            "entry",
            "golden",
            "specs",
            "argv_json",
            "target_count",
            "source_sha256",
        ),
        driver_rows,
    )
    _write_tsv(
        output / "problems.tsv",
        (
            "tier",
            "driver_id",
            "script",
            "instance",
            "problem_fingerprint",
            "buffers",
            "pools",
            "reuse_penalties",
            "operation_class",
            "pool_id",
            "pool",
            "native_capacity",
            "half_capacity",
            "q1_capacity",
            "tight_capacity",
        ),
        problem_rows,
    )
    _write_tsv(
        output / "preflight-cells.tsv",
        (
            "tier",
            "driver_id",
            "instance",
            "problem_fingerprint",
            "pool_id",
            "pool",
            "capacity",
            "capacity_bytes",
            "arm",
            "status",
            "placement_sha256",
        ),
        cell_rows,
    )
    summary = {
        "verdict": "DEDICATED_DRIVER_COHORT_FROZEN",
        "cohort_sha256": frozen_sha,
        "driver_count": len(driver_rows),
        "problem_count": len(problem_rows),
        "canary_driver_count": frozen["canary_driver_count"],
        "expanded_problem_count": frozen["expanded_problem_count"],
        "operation_class_count": len(frozen["operation_classes"]),
        "cell_count": len(cell_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--pypto-lib-root", required=True, type=Path)
    parser.add_argument(
        "--problems",
        required=True,
        type=Path,
        help="Problem inventory with script/instance/fingerprint rows; invocations.tsv preserves aliases.",
    )
    parser.add_argument("--screen-results", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    prepare_cohort(
        args.catalog,
        args.pypto_lib_root,
        args.problems,
        args.screen_results,
        args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
