# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail-closed codegen comparability checks for DSA replay campaigns."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOC_ADDRESS = re.compile(r"pto\.alloc_tile\s+addr\s*=\s*%(\S+)")
_CONSTANT = re.compile(r"(?m)^\s*%(c[-\w]+)\s*=\s*arith\.constant\s+(-?\d+)\b")
_FUNCTION = re.compile(r"(?m)^\s*func\.func @([A-Za-z0-9_]+)")
_INDEXED_CONSTANT = re.compile(r"(?m)^\s*%(cst_\d+)\s*=\s*arith\.constant\s+(.+)$")


@dataclass(frozen=True)
class CodegenArtifacts:
    """Recursively discovered artifacts needed for a non-vacuous comparison."""

    pto_files: tuple[Path, ...]
    kernel_configs: tuple[Path, ...]
    orchestration_sources: tuple[Path, ...]
    compiled_metadata: tuple[Path, ...]
    kernel_sources: tuple[Path, ...]


@dataclass(frozen=True)
class PlacementCheck:
    """Address-coverage result for one emitted function."""

    function: str
    emitted_addresses: int
    placement_offsets: int
    interior_addresses: int
    unresolved_addresses: int
    outside_addresses: tuple[int, ...]
    uncovered_offsets: tuple[int, ...]

    @property
    def matches(self) -> bool:
        return not (self.unresolved_addresses or self.outside_addresses or self.uncovered_offsets)


def discover_codegen_artifacts(build_root: str | Path) -> CodegenArtifacts:
    """Discover nested codegen artifacts and reject an empty identity capture."""
    root = Path(build_root)
    artifacts = CodegenArtifacts(
        pto_files=tuple(sorted(root.rglob("ptoas/*.pto"))),
        kernel_configs=tuple(sorted(root.rglob("kernel_config.py"))),
        orchestration_sources=tuple(sorted(root.rglob("orchestration/*.cpp"))),
        compiled_metadata=tuple(
            sorted((*root.rglob("compiled_meta.json"), *root.rglob("distributed_meta.json")))
        ),
        kernel_sources=tuple(sorted(root.rglob("kernels/**/*.cpp"))),
    )
    missing = [
        name
        for name, paths in (
            ("PTO", artifacts.pto_files),
            ("kernel_config.py", artifacts.kernel_configs),
            ("orchestration C++", artifacts.orchestration_sources),
        )
        if not paths
    ]
    if missing:
        raise ValueError(f"Codegen identity under {root} is vacuous; missing {', '.join(missing)}")
    return artifacts


def split_pto_functions(pto_text: str) -> dict[str, str]:
    """Split one PTO module into function regions with module context."""
    parts = _FUNCTION.split(pto_text)
    functions = {parts[index]: parts[0] + parts[index + 1] for index in range(1, len(parts), 2)}
    if not functions:
        raise ValueError("PTO module contains no func.func regions")
    if len(functions) != (len(parts) - 1) // 2:
        raise ValueError("PTO module repeats a function name")
    return functions


def emitted_tile_addresses(pto_text: str) -> tuple[tuple[int, ...], int]:
    """Resolve integer constants used as physical ``alloc_tile`` addresses."""
    constants = dict(_CONSTANT.findall(pto_text))
    values: list[int] = []
    unresolved = 0
    for name in _ALLOC_ADDRESS.findall(pto_text):
        if name in constants:
            values.append(int(constants[name]))
            continue
        literal = re.fullmatch(r"c(-?\d+)_i64", name)
        if literal is None:
            unresolved += 1
        else:
            values.append(int(literal.group(1)))
    return tuple(values), unresolved


def check_function_placement(
    function: str,
    pto_text: str,
    solution: Mapping[str, Any],
    buffer_sizes: Mapping[int, int],
) -> PlacementCheck:
    """Require emitted address coverage by one function's replay solution."""
    values, unresolved = emitted_tile_addresses(pto_text)
    ranges: list[tuple[int, int]] = []
    offsets: set[int] = set()
    seen_buffers: set[int] = set()
    for placement in solution.get("placements", []):
        buffer = int(placement["buffer"])
        if buffer in seen_buffers:
            raise ValueError(f"Solution for {function} repeats buffer {buffer}")
        seen_buffers.add(buffer)
        if buffer not in buffer_sizes:
            raise ValueError(f"Solution for {function} references unknown buffer {buffer}")
        offset = int(placement["offset"])
        offsets.add(offset)
        ranges.append((offset, offset + int(buffer_sizes[buffer])))
    if not ranges:
        raise ValueError(f"Solution for {function} has no placements")

    emitted = set(values)
    outside = tuple(
        sorted(address for address in emitted if not any(lo <= address < hi for lo, hi in ranges))
    )
    uncovered = tuple(sorted(offsets - emitted))
    return PlacementCheck(
        function=function,
        emitted_addresses=len(emitted),
        placement_offsets=len(offsets),
        interior_addresses=len(emitted - offsets),
        unresolved_addresses=unresolved,
        outside_addresses=outside,
        uncovered_offsets=uncovered,
    )


def check_build_placements(
    artifacts: CodegenArtifacts,
    solution_directory: str | Path,
    buffer_sizes: Mapping[str, Mapping[int, int]],
) -> tuple[PlacementCheck, ...]:
    """Join replay solutions to emitted functions, never to PTO filenames."""
    functions: dict[str, str] = {}
    for path in artifacts.pto_files:
        for name, body in split_pto_functions(path.read_text(encoding="utf-8")).items():
            if name in functions:
                raise ValueError(f"Codegen emits function {name} more than once")
            functions[name] = body

    expected = set(buffer_sizes)
    missing = sorted(expected - set(functions))
    if missing:
        raise ValueError(f"Codegen does not emit expected DSA functions {missing}")
    solution_root = Path(solution_directory)
    checks: list[PlacementCheck] = []
    for name in sorted(expected):
        solution_path = solution_root / f"pypto_{name}.dsa.solution.json"
        if not solution_path.is_file():
            raise ValueError(f"Replay map omits solution for emitted function {name}")
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        check = check_function_placement(name, functions[name], solution, buffer_sizes[name])
        if not check.matches:
            raise ValueError(f"Emitted addresses do not match the replay solution for {name}: {check}")
        checks.append(check)

    orphans = sorted(
        name for name, body in functions.items() if name not in expected and emitted_tile_addresses(body)[0]
    )
    if orphans:
        raise ValueError(f"Emitted functions with alloc_tile sites have no DSA problem: {orphans}")
    if not checks:
        raise ValueError("No emitted function was joined to a DSA replay solution")
    return tuple(checks)


def _validate_address_constant_uses(pto_text: str, address_names: set[str]) -> None:
    for name in address_names:
        token = f"%{name}"
        for line in pto_text.splitlines():
            if token not in line:
                continue
            is_definition = re.match(rf"^\s*%{re.escape(name)}\s*=", line) is not None
            is_address = re.search(rf"pto\.alloc_tile\s+addr\s*=\s*%{re.escape(name)}\b", line) is not None
            if not (is_definition or is_address):
                raise ValueError(
                    f"Address constant %{name} also has a semantic use and cannot be normalized: "
                    f"{line.strip()}"
                )


def normalize_pre_insert_sync_pto(pto_text: str) -> str:
    """Erase placement-only facts while preserving semantic constants and types."""
    address_names = set(_ALLOC_ADDRESS.findall(pto_text))
    _validate_address_constant_uses(pto_text, address_names)
    lines = []
    for line in pto_text.splitlines():
        if any(re.match(rf"^\s*%{re.escape(name)}\s*=", line) for name in address_names):
            continue
        lines.append(line)
    text = "\n".join(lines)
    for name in sorted(address_names, key=len, reverse=True):
        text = re.sub(rf"%{re.escape(name)}\b", "%ADDR", text)
    text = re.sub(r"_inline\d+", "_inlineN", text)

    indexed_constants = dict(_INDEXED_CONSTANT.findall(text))
    for name in sorted(indexed_constants, key=len, reverse=True):
        signature = hashlib.sha256(indexed_constants[name].strip().encode()).hexdigest()[:12]
        text = re.sub(rf"%{re.escape(name)}\b", f"%cstV_{signature}", text)
    return text


def normalized_pto_digest(pto_files: Sequence[Path]) -> str:
    """Hash normalized functions independently of build-directory layout."""
    regions: list[str] = []
    for path in pto_files:
        for name, body in sorted(split_pto_functions(path.read_text(encoding="utf-8")).items()):
            regions.append(f"func.func @{name}\n{normalize_pre_insert_sync_pto(body)}")
    if not regions:
        raise ValueError("Cannot hash an empty PTO artifact set")
    return hashlib.sha256("\n".join(sorted(regions)).encode()).hexdigest()
