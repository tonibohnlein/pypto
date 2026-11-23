# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile one arm in a fresh interpreter from the matching PyPTO worktree."""

import argparse
from pathlib import Path

import torch
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.runtime import RunConfig
from transforms.test_tilelang_gemm_sync import jit_tilelang_persistent_gemm

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["pypto", "stock", "disjoint"])
p.add_argument("out", type=Path)
args = p.parse_args()
backend.set_backend_type(BackendType.Ascend910)
inputs = [
    torch.empty(shape, dtype=torch.float16, device="meta")
    for shape in ((8192, 8192), (8192, 1024), (8192, 1024))
]
with passes.PassContext(
    [],
    enable_buffer_ir=True,
    memory_planner=passes.MemoryPlanner.PYPTO,
    enable_pypto_autosync=args.arm != "stock",
    autosync_policy="conservative_prefix",
    autosync_disjoint_gm=args.arm == "disjoint",
):
    jit_tilelang_persistent_gemm.compile(
        *inputs,
        config=RunConfig(
            arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(args.out.resolve())
        ),
    )
assert (args.out / "ptoas/tl_persistent_gemm.cpp").is_file()
