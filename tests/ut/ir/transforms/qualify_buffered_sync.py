# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce host-only buffered-class checks and descriptive scaling measurements."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from pypto import backend, codegen, ir, testing
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType

from .qualify_parameterized_prefill import digest
from .sync_matmul_oracle import validate
from .sync_physical_oracle import directional_widths, run_physical
from .test_autosync_switch import apply_hybrid
from .test_buffered_sync_realization import resident_streamed_sequence
from .test_regional_slot_composition import regional_fixture, repeated_fixture
from .test_rotating_sync_analysis import rotating_fixture
from .test_slot_sync_import import slot_fixture
from .test_streamed_slot_composition import streamed_fixture, streamed_sequence


def hierarchy_measurements():
    records = []
    cases = [("inner", inner, 7, 3) for inner in [4, 1000, 1000000]]
    cases += [("outer", 4, outer, 3) for outer in [2, 7, 64, 1000000]]
    cases += [("interface_slots", 7, 3, slots) for slots in [3, 11, 101]]
    cases += [("regions", 4, regions, 3) for regions in [2, 4, 8, 16]]
    for axis, inner, outer, slots in cases:
        function = (
            regional_fixture([inner] * outer, slots)
            if axis == "regions"
            else repeated_fixture(inner, outer, slots)
        )
        started = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - started
        description = result.describe()
        record = {
            "axis": axis,
            "inner": inner,
            "outer": outer,
            "slots": slots,
            "analysis_inclusive_seconds": elapsed,
            "work": {
                key: description[key]
                for key in [
                    "analyzed_occurrences",
                    "interface_cells",
                    "interface_slot_visits",
                    "interface_ports",
                    "interface_entries",
                    "interface_queries",
                    "merge_nodes",
                    "crossing_tests",
                    "interface_access_visits",
                    "bridge_candidates",
                    "port_normalization_visits",
                ]
            },
            "timers": {
                key: description[key]
                for key in ["import_seconds", "partition_seconds", "reduction_seconds", "composition_seconds"]
            },
        }
        if inner * outer <= 8 and slots == 3:
            record["semantic_checks"] = validate(function, result)
        records.append(record)
    return records


def streamed_measurements():
    records = []
    for count in [7, 1000, 1000000]:
        function = streamed_sequence(count)
        started = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - started
        description = result.describe()
        record = {
            "axis": "streamed_trips",
            "trips": count,
            "analysis_inclusive_seconds": elapsed,
            "work": {
                key: description[key]
                for key in [
                    "analyzed_occurrences",
                    "interface_cells",
                    "interface_slot_visits",
                    "interface_entries",
                    "merge_nodes",
                    "omitted_bridge_domains",
                ]
            },
            "owned_domains": sum(len(region["visit_owned"]) for region in description["regions"]),
        }
        if count == 7:
            record["semantic_checks"] = validate(function, result)
        records.append(record)
    return records


def physical_measurements():
    records = []
    for name, factory in [
        ("streamed", streamed_fixture),
        ("resident_streamed_sequence", resident_streamed_sequence),
    ]:
        for count in [7, 1000, 1000000]:
            function = factory(count)
            started = time.perf_counter()
            retained = testing.analyze_placed_sync(function, True)
            analysis_time = time.perf_counter() - started
            started = time.perf_counter()
            result = retained.realize_physical("a2a3", list(range(6)))
            preparation_time = time.perf_counter() - started
            started = time.perf_counter()
            production = apply_hybrid([function])[function.name]
            production_time = time.perf_counter() - started
            assert dict(production.attrs)[PYPTO_AUTOSYNC_ATTR], dict(production.attrs)
            started = time.perf_counter()
            text = codegen.PTOCodegen().generate(
                ir.Program([production], "buffered", production.span),
                emit_tile_addr=True,
                emit_source_loc=False,
            )
            source_time = time.perf_counter() - started
            record = {
                "axis": "physical_trips",
                "form": name,
                "trips": count,
                "analysis_seconds": analysis_time,
                "physical_preparation_allocation_emission_seconds": preparation_time,
                "ordinary_production_inclusive_seconds": production_time,
                "source_ir_seconds": source_time,
                "static": {
                    key: result[key]
                    for key in [
                        "set_sites",
                        "wait_sites",
                        "barrier_sites",
                        "guard_sites",
                        "reserved_lanes",
                        "reuse_queries",
                    ]
                },
                "families": len(result["families"]),
                "source_arithmetic_operations": text.count("arith."),
                "source_bytes": len(text.encode()),
            }
            if count == 7:
                state, order, _ = run_physical(function, result["function"], range(6))
                record["bounded"] = {
                    "payloads": state.payload_count,
                    "sets": len(state.records),
                    "waits": len(state.waits),
                    "barriers": len(state.barriers),
                    "causal_reuses": len(state.reuses),
                    "directional_ids": {
                        f"{int(a)}->{int(b)}": sorted(ids) for (a, b), ids in state.ids.items()
                    },
                    "directional_width": {
                        f"{int(a)}->{int(b)}": width
                        for (a, b), width in directional_widths(state, order).items()
                    },
                    "extra_payload_pairs": 0,
                }
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-hierarchy", action="store_true")
    parser.add_argument("--include-streamed", action="store_true")
    parser.add_argument("--include-physical", action="store_true")
    args = parser.parse_args()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    records = []
    cases = [("trips", count, 3, 1) for count in [7, 1000, 1000000]]
    cases += [("slots", 7, slots, 1) for slots in [3, 101, 1000003]]
    cases += [("stages", 1000, 5, stages) for stages in [1, 4, 16, 64]]
    for axis, count, slots, stages in cases:
        function = (
            slot_fixture(count, slots)
            if axis != "stages"
            else rotating_fixture(
                count,
                [(slots, 1)],
                [(op, 0, stage % slots) for stage in range(stages) for op in ["load", "rmw", "store"]],
            )
        )
        started = time.perf_counter()
        result = testing.analyze_placed_sync(function, True)
        elapsed = time.perf_counter() - started
        description = result.describe()
        region = description["regions"][0]
        record = {
            "axis": axis,
            "trips": count,
            "slots": slots,
            "stages": stages,
            "analysis_inclusive_seconds": elapsed,
            "timers": {
                key: description[key]
                for key in ["import_seconds", "partition_seconds", "reduction_seconds", "composition_seconds"]
            },
            "work": {
                key: region[key]
                for key in [
                    "rotating_fragments",
                    "atom_tests",
                    "sort_comparisons",
                    "sweep_visits",
                    "shortest_path_runs",
                    "shortest_path_edge_visits",
                    "retained_frontier_entries",
                    "quotient_edges",
                    "distance_entries",
                ]
            },
            "sites": len(region["sites"]),
            "generators": len(region["generators"]),
            "demands": len(region["demands"]),
        }
        if count == 7 and slots == 3:
            record["semantic_checks"] = validate(function, result)
        records.append(record)
    if args.include_hierarchy:
        records.extend(hierarchy_measurements())
    if args.include_streamed:
        records.extend(streamed_measurements())
    if args.include_physical:
        records.extend(physical_measurements())
    paths = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    paths += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "include", "src", "tests", "python"], text=True
    ).splitlines()
    receipt = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_source_hashes": {name: digest(Path(name)) for name in paths},
        "extensions": {str(path): digest(path) for path in Path("python/pypto").glob("pypto_core*.so")},
        "python": platform.python_version(),
        "target": "a2a3",
        "disjoint_gm": True,
        "timing_scope": "Single descriptive host runs; detail timers included in analysis, not additive",
        "physical_scope": "Numerical geometry; large rings need not fit native local storage",
        "device": "not run",
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
