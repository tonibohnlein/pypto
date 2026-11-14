# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pinned original Shenggan payload, with explicitly configured host derivatives."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime import RunConfig

FIXTURES = Path(__file__).parent / "fixtures" / "shenggan_gemm"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


def payload_source(*, m=2048, n=4096, k=4096, workers=24, swizzle=False):
    """Remove only manual synchronization and the unrelated benchmark driver."""
    assert m > 0 and m % 128 == 0
    assert n > 0 and n % 256 == 0
    assert k > 0 and k % 256 == 0
    assert workers > 0
    path = FIXTURES / (MANIFEST["source"] + ".source")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == MANIFEST["sha256"]
    lines = data.decode().splitlines(keepends=True)
    for first, last in MANIFEST["removed_sync_lines"]:
        for index in range(first - 1, last):
            lines[index] = "\n"
    functions = ["".join(lines[first - 1 : last]) for first, last in MANIFEST["functions"].values()]
    config = dict(
        MANIFEST["original_configuration"], M=m, N=n, K=k, SPMD_CORES=workers, ENABLE_TILE_SWIZZLE=swizzle
    )
    constants = "\n".join(f"{name} = {value!r}" for name, value in config.items())
    constants += "\nM_BLOCKS = M // M_TILE\nN_BLOCKS = N // N_TILE\nTOTAL_TILES = M_BLOCKS * N_BLOCKS\n"
    return "import pypto.language as pl\n" + constants + "\n\n" + "\n\n".join(functions)


def capture_shenggan(directory, *, m=2048, n=4096, k=4096, workers=24, swizzle=False):
    """Ordinary Buffer lowering, no device or PTOAS, with original tile shapes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = payload_source(m=m, n=n, k=k, workers=workers, swizzle=swizzle)
    path = directory / "payload.py"
    path.write_text(source)
    module_name = "shenggan_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
            program = module.hpgemm.lower(
                torch.empty((m, k), dtype=torch.float16, device="meta"),
                torch.empty((k, n), dtype=torch.float16, device="meta"),
                torch.empty((m, n), dtype=torch.float32, device="meta"),
                config=RunConfig(arch=BackendType.Ascend910),
            )
    finally:
        sys.modules.pop(module_name, None)
    ir.serialize_to_file(program, str(directory / "before-sync.msgpack"))
    (directory / "before-sync.py").write_text(str(program))
    return program


def payload_compute_ast(text):
    """Check payload preservation independently of compiler output or comments."""
    tree = ast.parse(text)

    class RemoveSync(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in {
                "pl.system.sync_src",
                "pl.system.sync_dst",
                "pl.system.bar_all",
            }:
                return None
            return node

    functions = [
        RemoveSync().visit(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in MANIFEST["functions"]
    ]
    options = {"include_attributes": False}
    if sys.version_info >= (3, 13):
        options["show_empty"] = True
    return ast.dump(ast.Module(body=functions, type_ignores=[]), **options)
