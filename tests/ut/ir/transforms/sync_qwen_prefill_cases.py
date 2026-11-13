# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Pinned full Qwen projection bodies, with explicit extraction boundaries."""

import ast
import hashlib
import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any

from pypto import ir, passes
from pypto.backend import BackendType
from pypto.jit.decorator import JITFunction
from pypto.runtime import RunConfig

from .sync_matmul_oracle import integer

FIXTURES = Path(__file__).parent / "fixtures" / "qwen_prefill"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


def source_body_ast(tree: ast.AST) -> str:
    """Pin empty fields explicitly instead of inheriting ast.dump defaults.

    Python 3.13 added show_empty and later defaults omit empty fields. Older
    interpreters always include them. Keep the fixture fingerprint identical
    across the host and device Python versions without removing AST content.
    """
    if sys.version_info >= (3, 13):
        return ast.dump(tree, include_attributes=False, show_empty=True)
    return ast.dump(tree, include_attributes=False)


def _load(name: str, path: Path, namespace: dict[str, Any] | None = None) -> ModuleType:
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if namespace:
        module.__dict__.update({k: v for k, v in namespace.items() if not k.startswith("__")})
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_qwen_projection(
    name: str, *, layers: int | None = None, band: int = 0, hidden: int | None = None
) -> tuple[ModuleType, JITFunction]:
    """Load an unchanged compute body; the manifest names all wrapper adaptations."""
    if name not in {"output", "gate_up"}:
        raise ValueError(f"Unknown Qwen projection {name!r}")
    names = [
        "constants",
        "config",
        "rms_lm_head",
        "qwen_prefill_source",
        "qwen_gate_up_source",
        "qwen_output_source",
        "qwen_configured_caller",
    ]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        modules = {}
        for filename, module_name in zip(
            ["constants.py", "config.py", "rms_lm_head.py", "prefill_fwd.py"], names[:4], strict=True
        ):
            path = FIXTURES / f"{filename}.source"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["sources"][filename]
            modules[module_name] = _load(module_name, path)
        module = modules["qwen_prefill_source"]
        namespace = dict(vars(module))
        if layers is not None:
            if layers <= 0 or not 0 <= band < module.MLP_PROJ_BANDS:
                raise ValueError(f"Invalid explicit Qwen caller configuration: layers={layers}, band={band}")
            width = module.HIDDEN if hidden is None else hidden
            if width <= 0 or width % (2 * module.K_CHUNK):
                raise ValueError(
                    f"Qwen hidden dimension must be a positive two-stage K multiple, got {width}"
                )
            namespace.update(
                LAYERS=layers,
                BAND=band,
                HIDDEN=width,
                HIDDEN_BLOCKS=width // module.K_CHUNK,
                Q_OUT_BLOCKS=width // module.Q_OUT_CHUNK,
                LAYER_HIDDEN_ROWS_DYN=layers * width,
            )
        elif hidden is not None:
            raise ValueError("A hidden-size derivative requires an explicit caller configuration")
        if name == "output":
            path = FIXTURES / "output_projection.py.source"
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == MANIFEST["output_extraction"]["adapter_sha256"]
            )
            adapter = _load("qwen_output_source", path, namespace)
            phase = adapter._out_proj_aic_phase
        else:
            path = FIXTURES / "gate_up_projection.py.source"
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == MANIFEST["gate_up_extraction"]["adapter_sha256"]
            )
            adapter = _load("qwen_gate_up_source", path, namespace)
            phase = adapter.gate_up_projection
        if layers is None:
            entry = JITFunction(phase._func, func_type="orchestration", auto_scope=False)
        else:
            namespace[f"{name}_phase"] = JITFunction(phase._func, func_type="inline", auto_scope=False)
            caller_path = FIXTURES / "configured_callers.py.source"
            assert (
                hashlib.sha256(caller_path.read_bytes()).hexdigest() == MANIFEST["configured_caller_sha256"]
            )
            caller = _load("qwen_configured_caller", caller_path, namespace)
            entry = getattr(caller, f"{name}_projection_caller")
        return module, entry
    finally:
        # Local source imports must not replace unrelated caller modules.
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous


def assert_gate_up_body_identity() -> None:
    """Compare the entire combined payload/control AST, excluding only its task wrapper."""
    original = ast.parse((FIXTURES / "prefill_fwd.py.source").read_text())
    adapted = ast.parse((FIXTURES / "gate_up_projection.py.source").read_text())

    def body(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and any(
                    kw.arg == "name_hint"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "gate_up_proj_spmd"
                    for kw in item.context_expr.keywords
                )
                for item in node.items
            ):
                return source_body_ast(ast.Module(body=node.body, type_ignores=[]))
        raise AssertionError("Missing original combined gate/up launch")

    assert body(original) == body(adapted)
    assert (
        hashlib.sha256(body(original).encode()).hexdigest()
        == MANIFEST["gate_up_extraction"]["body_ast_sha256"]
    )


def capture_qwen_projection(name: str, **configuration: Any) -> ir.Program:
    """Lower the original ABI or the named explicit caller; samples add no proof."""
    _, entry = load_qwen_projection(name, **configuration)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        return entry.lower(config=RunConfig(arch=BackendType.Ascend910))


def original_control_trace(function: ir.Function, layer: int, block: int) -> list[tuple[str, dict[str, int]]]:
    """Execute original scalar/control IR without allocating an address set.

    This is a control-only oracle for the complete worker/band domain. The
    independent element-address interpreter separately checks bounded workers.
    Neither oracle consumes compiler loop summaries or range certificates.
    """
    events = []

    def walk(stmt, env):
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                walk(child, env)
        elif isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.var.type, ir.ScalarType):
                env[stmt.var.name_hint] = integer(stmt.value, env)
        elif isinstance(stmt, ir.ForStmt):
            for value in range(integer(stmt.start, env), integer(stmt.stop, env), integer(stmt.step, env)):
                walk(stmt.body, {**env, stmt.loop_var.name_hint: value})
        elif isinstance(stmt, ir.IfStmt):
            arm = stmt.then_body if integer(stmt.condition, env) else stmt.else_body
            if arm is not None:
                walk(arm, env)
        elif isinstance(stmt, ir.EvalStmt):
            assert isinstance(stmt.expr, ir.Call)
            events.append((stmt.expr.op.name, env.copy()))
        else:
            assert isinstance(stmt, (ir.YieldStmt, ir.ReturnStmt))

    walk(function.body, {function.params[3].name_hint: layer, "__block_index__": block})
    return events


def assert_output_body_identity() -> None:
    """Compare the entire extracted output definition, including its original ABI."""

    def definition(filename):
        tree = ast.parse((FIXTURES / filename).read_text())
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_out_proj_aic_phase"
        )
        return ast.dump(node, include_attributes=False)

    assert definition("prefill_fwd.py.source") == definition("output_projection.py.source")
