# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five-arm publication must validate bytes, real geometry and complete identities."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools/prepare_dsa_five_arm_release.py"
_SPEC = importlib.util.spec_from_file_location("five_arm_release", _PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


def fixture():
    problem = {
        "schema_version": 1,
        "profile": "test",
        "instance": "kernel",
        "problem": {
            "pools": [{"id": 1, "capacity": 32, "reserved_ranges": []}],
            "buffers": [
                {
                    "id": i,
                    "size": 8,
                    "alignment": 8,
                    "allowed_pools": [1],
                    "live_intervals": [{"lower": i * 4, "upper": i * 4 + 3}],
                }
                for i in (0, 1)
            ],
            "constraints": {"separations": [], "no_partial_overlaps": []},
        },
    }
    solution = {
        "schema_version": 1,
        "profile": "test",
        "instance": "kernel",
        "placements": [{"buffer": i, "pool": 1, "offset": 0} for i in (0, 1)],
    }
    return problem, solution


@pytest.mark.parametrize(
    "mutation",
    [
        "capacity",
        "alignment",
        "pool",
        "duplicate",
        "lifetime",
        "separation",
        "partial",
        "reserved",
        "unknown",
        "envelope",
    ],
)
def test_independent_validator_rejects_defects(mutation):
    problem, solution = fixture()
    release.validate(problem, solution)
    release.negative_control(problem, solution)
    if mutation == "capacity":
        solution["placements"][0]["offset"] = 32
    elif mutation == "alignment":
        solution["placements"][0]["offset"] = 1
    elif mutation == "pool":
        solution["placements"][0]["pool"] = 99
    elif mutation == "duplicate":
        solution["placements"].append(copy.deepcopy(solution["placements"][0]))
    elif mutation == "lifetime":
        problem["problem"]["buffers"][1]["live_intervals"] = [{"lower": 0, "upper": 8}]
    elif mutation == "separation":
        problem["problem"]["constraints"]["separations"] = [{"first": 0, "second": 1}]
    elif mutation == "partial":
        problem["problem"]["buffers"][1]["size"] = 16
        problem["problem"]["constraints"]["no_partial_overlaps"] = [{"first": 0, "second": 1}]
    elif mutation == "reserved":
        problem["problem"]["pools"][0]["reserved_ranges"] = [{"begin": 0, "end": 8}]
    elif mutation == "unknown":
        problem["problem"]["constraints"]["unknown"] = [1]
    else:
        solution["profile"] = "different"
    with pytest.raises(ValueError):
        release.validate(problem, solution)


def test_map_identity_and_empty_map(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        release.map_digest(tmp_path)
    _, solution = fixture()
    file = tmp_path / "kernel.solution.json"
    release.write(file, solution)
    original = release.map_digest(tmp_path)
    solution["metadata"] = {"solver": "different", "runtime_us": 123}
    solution["placements"].reverse()
    release.write(file, solution)
    assert release.map_digest(tmp_path) == original
    solution["placements"][0]["offset"] = 8
    release.write(file, solution)
    assert release.map_digest(tmp_path) != original


def test_packet_manifest_rejects_unlisted_files_and_changed_bytes(tmp_path):
    release.write(tmp_path / "a.json", {"value": 1})
    release.write(tmp_path / "MANIFEST.json", {"a.json": release.digest(tmp_path / "a.json")})
    release.write(tmp_path / "unexpected.json", {})
    with pytest.raises(ValueError, match="coverage"):
        release.verify_packet(tmp_path)
    (tmp_path / "unexpected.json").unlink()
    release.write(tmp_path / "a.json", {"value": 2})
    with pytest.raises(ValueError, match="hash"):
        release.verify_packet(tmp_path)


def test_portable_search_command():
    command = release.search_command("inputs/w00/kernel", "latency", 128)
    assert command[1:4] == ["-m", "pypto.tools.dsa_latency_planner", "--problem"]
    assert "inputs/w00/kernel/search-problem.json" in command
    assert "128" in command
    assert not any(arg.startswith("/home/") for arg in command[1:])
    assert json.loads(json.dumps(command)) == command


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
