# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling" / "dsa_codegen_comparability.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_dsa_codegen_comparability", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
comparability = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = comparability
_SPEC.loader.exec_module(comparability)


def _pto(address: int = 64, op: str = "pto.tadds", scalar: str = "3.0 : f32") -> str:
    return f"""module {{
  %c1 = arith.constant {address} : i64
  %cst_7 = arith.constant {scalar}
  func.func @kernel() {{
    %tile_inline17 = pto.alloc_tile addr = %c1 rows=8, cols=64, dtype=f32
    {op} %tile_inline17, %cst_7
    return
  }}
}}
"""


def _write_build(root: Path, *, nested: bool = False) -> Path:
    build = root / "next_levels" / "rank0" if nested else root
    (build / "ptoas").mkdir(parents=True)
    (build / "ptoas" / "kernel.pto").write_text(_pto())
    (build / "kernel_config.py").write_text('KERNELS = [{"name": "kernel"}]\n')
    (build / "orchestration").mkdir()
    (build / "orchestration" / "main.cpp").write_text("// orchestration\n")
    return build


def test_recursive_discovery_finds_nested_distributed_build(tmp_path: Path) -> None:
    _write_build(tmp_path, nested=True)
    artifacts = comparability.discover_codegen_artifacts(tmp_path)

    assert len(artifacts.pto_files) == 1
    assert len(artifacts.kernel_configs) == 1
    assert len(artifacts.orchestration_sources) == 1


def test_discovery_rejects_vacuous_capture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vacuous"):
        comparability.discover_codegen_artifacts(tmp_path)


def test_per_function_address_join_accepts_contained_views(tmp_path: Path) -> None:
    build = _write_build(tmp_path)
    pto_path = build / "ptoas/kernel.pto"
    source = pto_path.read_text()
    source = source.replace(
        "  %c1 = arith.constant 64 : i64",
        "  %c0 = arith.constant 32 : i64\n  %c1 = arith.constant 64 : i64",
    )
    source = source.replace(
        "    %tile_inline17 = pto.alloc_tile addr = %c1",
        "    %base_inline16 = pto.alloc_tile addr = %c0 rows=8, cols=64, dtype=f32\n"
        "    %tile_inline17 = pto.alloc_tile addr = %c1",
    )
    pto_path.write_text(source)
    map_root = tmp_path / "map"
    map_root.mkdir()
    (map_root / "pypto_kernel.dsa.solution.json").write_text(
        json.dumps({"placements": [{"buffer": 0, "pool": 1, "offset": 32}]})
    )
    artifacts = comparability.discover_codegen_artifacts(tmp_path)
    checks = comparability.check_build_placements(artifacts, map_root, {"kernel": {0: 64}})

    assert len(checks) == 1
    assert checks[0].matches
    assert checks[0].interior_addresses == 1


def test_per_function_join_rejects_missing_solution_and_outside_address(tmp_path: Path) -> None:
    _write_build(tmp_path)
    map_root = tmp_path / "map"
    map_root.mkdir()
    artifacts = comparability.discover_codegen_artifacts(tmp_path)
    with pytest.raises(ValueError, match="omits solution"):
        comparability.check_build_placements(artifacts, map_root, {"kernel": {0: 64}})

    (map_root / "pypto_kernel.dsa.solution.json").write_text(
        json.dumps({"placements": [{"buffer": 0, "pool": 1, "offset": 128}]})
    )
    with pytest.raises(ValueError, match="do not match"):
        comparability.check_build_placements(artifacts, map_root, {"kernel": {0: 64}})


def test_normalizer_masks_addresses_and_unstable_names_only() -> None:
    baseline = comparability.normalize_pre_insert_sync_pto(_pto(address=64))
    moved = comparability.normalize_pre_insert_sync_pto(
        _pto(address=4096).replace("_inline17", "_inline93").replace("%cst_7", "%cst_81")
    )

    assert moved == baseline
    assert comparability.normalize_pre_insert_sync_pto(_pto(op="pto.tmuls")) != baseline
    assert comparability.normalize_pre_insert_sync_pto(_pto().replace("rows=8", "rows=16")) != baseline
    assert comparability.normalize_pre_insert_sync_pto(_pto(scalar="4.0 : f32")) != baseline
    assert comparability.normalize_pre_insert_sync_pto(_pto(scalar="3.0 : f16")) != baseline


def test_normalizer_rejects_mixed_address_and_semantic_use() -> None:
    source = _pto().replace("return", "pto.tadds %tile_inline17, %c1\n    return")
    with pytest.raises(ValueError, match="semantic use"):
        comparability.normalize_pre_insert_sync_pto(source)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
