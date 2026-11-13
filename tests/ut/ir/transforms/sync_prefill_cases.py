# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unchanged PyPTO-lib sources; only explicitly named K derivatives vary configuration."""

import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch
from pypto import passes
from pypto.backend import BackendType
from pypto.runtime import RunConfig

REVISION = "da5350d2bfc40ce12edeaafd44b698050f6dc85b"
SOURCES = {
    "matmul": (
        "examples/beginner/matmul.py",
        "8316cc2f039f818f08dd5e0c2bb8c3a90e9abff19501e253de4a0b787bd95585",
    ),
    "gemm": (
        "examples/intermediate/gemm.py",
        "565faaef5ad54962c122c8a5e9f7819475832b63cc70cd74e1f270f7068be065",
    ),
}


def load_prefill(name: str, k_tiles: int | None = None):
    """Load a hash-checked source without executing its device main block."""
    path = Path(__file__).parent / "fixtures" / "prefill_lib" / f"{name}.py.source"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCES[name][1]
    # Source snapshots are data, not locally reformatted Python modules.
    module_name = f"prefill_{name}_{k_tiles}"
    spec = importlib.util.spec_from_loader(module_name, SourceFileLoader(module_name, str(path)))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if k_tiles is not None:
        assert name == "gemm" and k_tiles > 0  # K=0 would leave this source's output uninitialized.
        setattr(module, "K", k_tiles * 64)
    return module


def prefill_arguments(module):
    """Meta tensors pin shape/dtype without allocating a million-tile input."""
    return [
        torch.empty(shape, device="meta")
        for shape in [(module.M, module.K), (module.K, module.N), (module.M, module.N)]
    ]


def capture_prefill(name: str, k_tiles: int | None = None):
    """Run the unchanged frontend and placement; no device or PTOAS invocation."""
    module = load_prefill(name, k_tiles)
    with passes.PassContext([], enable_buffer_ir=True, memory_planner=passes.MemoryPlanner.PYPTO):
        return getattr(module, name).lower(
            *prefill_arguments(module), config=RunConfig(arch=BackendType.Ascend910)
        )
