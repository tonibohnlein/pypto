# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run a production-shaped standalone leaf on 910B2 with a seeded exact reference."""

import argparse
import json
from pathlib import Path

import torch
from pypto.backend import BackendType, set_backend_type
from pypto.ir import CompiledProgram
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig, ensure_pto_isa_root


def inputs(leaf, band, seed):
    rng = torch.Generator().manual_seed(seed)

    def ternary(shape, dtype):
        return torch.randint(-1, 2, shape, dtype=torch.int8, generator=rng).to(dtype)

    cols = slice(4352 * band, 4352 * (band + 1))
    if leaf == "persistent":
        a, b = [ternary(shape, torch.float16) for shape in ((8192, 8192), (8192, 1024))]
        want = (a.float() @ b.float()).half()
        out = torch.full_like(want, torch.nan)
        return [a, b, out], out, want, None
    if leaf == "gate_up_proj":
        x = ternary((128, 5120), torch.bfloat16)
        weights = [ternary((5120, 17408), torch.bfloat16) for _ in range(2)]
        want = torch.cat([x.float() @ w[:, cols].float() for w in weights])
        out = torch.full_like(want, torch.nan)
        return [out, x, *weights], out, want, None
    x, w = [ternary(shape, torch.bfloat16) for shape in ((128, 4352), (17408, 5120))]
    init = ternary((128, 5120), torch.float32)
    out = init.clone()
    return [out, x, w], out, init + (x.float() @ w[cols, :].float()), init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("leaf", choices=["gate_up_proj", "down_proj", "persistent"])
    p.add_argument("arm", choices=["before", "current", "stock", "disjoint"])
    p.add_argument("--band", type=int, choices=range(4), default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=4107)
    p.add_argument("--launches", type=int, default=1)
    p.add_argument("--capture", action="store_true", help="capture per-task swimlane for one checked launch")
    p.add_argument("--host-check", action="store_true")
    p.add_argument("--root", type=Path, required=True, help="Prepared benchmark directory")
    args = p.parse_args()
    root = args.root.resolve()
    if args.capture and args.launches != 1:
        p.error("one capture per process to retain the trace")
    if args.launches < 1:
        p.error("launches must be positive")
    if args.leaf == "persistent":
        if args.arm == "before" or args.band:
            p.error("persistent supports current, stock or disjoint and has no band")
        folder = root / "persistent" / ("pypto" if args.arm == "current" else args.arm)
    else:
        if args.arm == "disjoint":
            p.error("projection arms use may-alias GM")
        folder = root / "isolated" / args.arm / args.leaf / f"band{args.band}"
    set_backend_type(BackendType.Ascend910)
    program = CompiledProgram.from_dir(
        folder, backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD
    )
    if args.host_check:
        print(folder, program.platform)
        return
    ensure_pto_isa_root()
    tensors, out, want, init = inputs(args.leaf, args.band, args.seed)
    config = RunConfig(
        backend_type=BackendType.Ascend910,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device,
        enable_chip_swimlane=int(args.capture),
    )
    with ChipWorker(config=config, runtime="tensormap_and_ringbuffer") as worker:
        handle = worker.register(program)
        for _ in range(args.launches):
            if init is None:
                out.fill_(torch.nan)
            else:
                out.copy_(init)
            handle(*tensors, config=config)
            torch.testing.assert_close(out, want, rtol=0, atol=0)
    print(
        json.dumps(dict(case=args.leaf, arm=args.arm, band=args.band, seed=args.seed, checked=args.launches))
    )


if __name__ == "__main__":
    main()
