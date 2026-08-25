# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static source-contract inspection for driver-first DSA discovery."""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectGoldenContract:
    """One statically proven direct-golden launch contract."""

    script: str
    line: int
    entry: str
    specs: str
    specs_call: str
    golden: str
    source_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_leaf(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _simple_name(expression: ast.expr | None) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    return ""


def _main_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If):
        return False
    test = statement.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    sides = (test.left, test.comparators[0])
    return any(isinstance(side, ast.Name) and side.id == "__name__" for side in sides) and any(
        isinstance(side, ast.Constant) and side.value == "__main__" for side in sides
    )


def _jit_decorated(function: ast.FunctionDef) -> bool:
    decorators = {ast.unparse(decorator) for decorator in function.decorator_list}
    return any(name == "pl.jit" or name.startswith("pl.jit.") or name.endswith(".jit") for name in decorators)


def inspect_source(lib_root: Path, path: Path) -> tuple[list[DirectGoldenContract], list[str]]:
    """Return direct-golden contracts and terminal source rejections."""
    relative = path.relative_to(lib_root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return [], [f"SYNTAX_ERROR:{error.lineno}"]

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    main_blocks = [node for node in tree.body if _main_guard(node)]
    if not main_blocks:
        return [], ["NO_MAIN_GUARD"]

    contracts: list[DirectGoldenContract] = []
    rejections: list[str] = []
    for main in main_blocks:
        for call in (node for node in ast.walk(main) if isinstance(node, ast.Call)):
            if _call_leaf(call) not in {"run", "run_jit"}:
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}
            entry_expression = keywords.get("fn", keywords.get("program"))
            entry = _simple_name(entry_expression)
            golden = _simple_name(keywords.get("golden_fn"))
            specs_expression = keywords.get("specs")
            specs = _simple_name(specs_expression.func) if isinstance(specs_expression, ast.Call) else ""
            missing = [
                name for name, value in (("entry", entry), ("golden", golden), ("specs", specs)) if not value
            ]
            if missing:
                rejections.append(f"LINE_{call.lineno}_NONSTATIC_{'_'.join(missing).upper()}")
                continue
            if entry not in functions or not _jit_decorated(functions[entry]):
                rejections.append(f"LINE_{call.lineno}_ENTRY_NOT_LOCAL_JIT:{entry}")
                continue
            if golden not in functions:
                rejections.append(f"LINE_{call.lineno}_GOLDEN_NOT_LOCAL:{golden}")
                continue
            if specs not in functions:
                rejections.append(f"LINE_{call.lineno}_SPECS_NOT_LOCAL:{specs}")
                continue
            contracts.append(
                DirectGoldenContract(
                    script=relative,
                    line=call.lineno,
                    entry=entry,
                    specs=specs,
                    specs_call=ast.unparse(specs_expression),
                    golden=golden,
                    source_sha256=_sha256(path),
                )
            )
    if not contracts and not rejections:
        rejections.append("NO_DIRECT_GOLDEN_RUN_CALL")
    return contracts, sorted(set(rejections))
