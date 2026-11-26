# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device half of the matrix storage reuse check: build, dispatch, compare exactly.

One arm and one row count per invocation. The program is the pinned kernel with
only an Orchestration wrapper added (see `matrix_program.py`). Each launch gets
one of two alternating seeded FP16 packs with entries from {-1, 0, 1}, the FP32
output is poisoned with NaN first so an unwritten element cannot pass, and all
32 x 16 values are compared for exact equality against a CPU golden computed
with the handoff's formula.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix_program as mp  # noqa: E402
from pypto import backend, ir, passes  # noqa: E402
from pypto.backend import BackendType  # noqa: E402
from pypto.ir.compiled_program import CompiledProgram  # noqa: E402
from pypto.runtime import ChipWorker, ExecutionMode, ensure_pto_isa_root  # noqa: E402
from pypto.runtime.runner import RunConfig  # noqa: E402

RUNTIME = "tensormap_and_ringbuffer"
TERNARY = torch.tensor([-1.0, 0.0, 1.0])


def pack(rows: int, seed: int) -> dict[str, torch.Tensor]:
    """One seeded, nonsymmetric FP16 input pack with entries from {-1, 0, 1}."""
    generator = torch.Generator().manual_seed(seed)

    def ternary(*shape: int) -> torch.Tensor:
        index = torch.randint(0, 3, shape, generator=generator)
        return TERNARY[index].to(torch.float16)

    a, seed_a, seed_b = ternary(rows, 32), ternary(16, 32), ternary(32, 32)
    # Nonsymmetric: a square pack that happened to be symmetric would hide a
    # transposed-view defect, which is exactly what this kernel exercises.
    for _ in range(64):
        if not torch.equal(seed_b, seed_b.T):
            break
        seed_b = ternary(32, 32)
    assert not torch.equal(seed_b, seed_b.T), "seed_b must not be symmetric"
    return {"a": a, "seed_a": seed_a, "seed_b": seed_b}


def golden(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """The handoff's CPU reference, verbatim."""
    seed = (values["seed_a"].float() @ values["seed_b"].float()).half().float()
    return values["a"][-32:, :].float() @ seed.T


def text_section(path: Path) -> tuple[int, str] | None:
    """(size, sha256) of an ELF's .text, so debug sections cannot mask a match."""
    blob = path.read_bytes()
    if blob[:4] != b"\x7fELF" or blob[4] != 2:
        return None
    end = "<" if blob[5] == 1 else ">"
    (e_shoff,) = struct.unpack_from(end + "Q", blob, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(end + "HHH", blob, 0x3A)

    def section(i: int):
        base = e_shoff + i * e_shentsize
        (name,) = struct.unpack_from(end + "I", blob, base)
        offset, size = struct.unpack_from(end + "QQ", blob, base + 0x18)
        return name, offset, size

    _, str_off, _ = section(e_shstrndx)
    for i in range(e_shnum):
        name_off, offset, size = section(i)
        stop = blob.index(b"\0", str_off + name_off)
        if blob[str_off + name_off : stop] == b".text":
            return size, hashlib.sha256(blob[offset : offset + size]).hexdigest()
    return None


def artefacts(build: Path) -> dict:
    """Emitted sources, their hashes, the layout descriptors and the native text."""
    out: dict = {"sources": {}, "natives": {}, "layout_descriptors": {}}
    for name in ("ptoas/kernel.pto", "ptoas/kernel.cpp", "kernels/aic/kernel.cpp", "orchestration/main.cpp"):
        path = build / name
        if path.exists():
            out["sources"][name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    manifest = build / "ptoas" / "kernel.sync.json"
    if manifest.exists():
        body = json.loads(manifest.read_text())
        out["sync_manifest"] = {
            "owned": body.get("owned"),
            "ptoas_insert_sync": body.get("ptoas_insert_sync"),
            "domains": body.get("domains"),
        }
    for obj in sorted((build / "kernels").rglob("*.o")) if (build / "kernels").is_dir() else []:
        info = text_section(obj)
        out["natives"][obj.name] = {
            "text_bytes": None if info is None else info[0],
            "text_sha256": None if info is None else info[1],
        }
    for binary in sorted(build.rglob("*.bin")):
        out["natives"].setdefault(binary.name, {})["binary_sha256"] = hashlib.sha256(
            binary.read_bytes()
        ).hexdigest()
    # Byte-order evidence: the fractal layout the emitted code declares for each
    # Mat operand. A transposed view that lost its descriptor would read the
    # same rows in a different byte order and the exact comparison would fail,
    # so these are recorded beside the result rather than instead of it.
    pto = build / "ptoas" / "kernel.pto"
    if pto.exists():
        text = pto.read_text()
        # Byte-order evidence: the fractal layout each tile buffer declares. A
        # transposed view that lost its descriptor would read the same rows in a
        # different byte order; the exact comparison would then fail, so these
        # are recorded beside the device result rather than instead of it.
        out["layout_descriptors"] = {
            "blayout": dict(Counter(re.findall(r"blayout=(\w+)", text))),
            "slayout": dict(Counter(re.findall(r"slayout=(\w+)", text))),
            "mat_tiles": sorted(
                set(
                    re.findall(
                        r"!pto\.tile_buf<loc=mat, dtype=(\w+), rows=(\d+), cols=(\d+), "
                        r"v_row=(\d+), v_col=(\d+), blayout=(\w+)",
                        text,
                    )
                )
            ),
            "transpose_views": len(re.findall(r"transpose", text)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("B", "R", "Bp"))
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs=2, default=(4107, 4111))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    autosync = args.arm in ("B", "Bp")
    args.out.mkdir(parents=True, exist_ok=True)
    backend.set_backend_type(BackendType.Ascend910)
    ensure_pto_isa_root()

    build = args.out / "build"
    program = mp.program(args.rows)
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=autosync,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=False,
        enable_scalar_gm_dma=True,
    ):
        compiled = ir.compile(
            program,
            output_dir=str(build),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=False,
        )

    record: dict = {
        "arm": args.arm,
        "rows": args.rows,
        "autosync": autosync,
        "device_id": args.device,
        "launches_requested": args.launches,
        "build": str(build),
        "artefacts": artefacts(build),
        "source_sha256": hashlib.sha256(mp.source(args.rows).encode()).hexdigest(),
    }

    packs = [pack(args.rows, s) for s in args.seeds]
    goldens = [golden(p) for p in packs]
    record["input_packs"] = [
        {
            "seed": s,
            "tensors": {
                k: {
                    "shape": list(v.shape),
                    "sha256": hashlib.sha256(v.view(torch.int16).numpy().tobytes()).hexdigest(),
                    "unique_values": sorted({float(x) for x in v.flatten().tolist()}),
                }
                for k, v in p.items()
            },
            "golden_sha256": hashlib.sha256(g.numpy().tobytes()).hexdigest(),
        }
        for s, p, g in zip(args.seeds, packs, goldens, strict=True)
    ]

    config = RunConfig(
        backend_type=BackendType.Ascend910,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device,
    )
    worker = ChipWorker(config=config, runtime=RUNTIME)
    failures: list[dict] = []
    exact = 0
    try:
        assert isinstance(compiled, CompiledProgram)
        handle = worker.register(compiled)
        for launch in range(args.launches):
            which = launch % len(packs)
            values, want = packs[which], goldens[which]
            # Poison: every element must be written by this launch.
            out = torch.full((32, 16), float("nan"), dtype=torch.float32)
            handle(values["a"], values["seed_a"], values["seed_b"], out)
            if torch.equal(out, want):
                exact += 1
                continue
            bad = (out != want) | torch.isnan(out)
            coords = torch.nonzero(bad)
            failures.append(
                {
                    "launch": launch,
                    "pack_seed": args.seeds[which],
                    "mismatched": int(bad.sum()),
                    "nan_count": int(torch.isnan(out).sum()),
                    "first_coords": coords[:20].tolist(),
                    "first_observed": [float(out[tuple(c)]) for c in coords[:20]],
                    "first_expected": [float(want[tuple(c)]) for c in coords[:20]],
                    "max_abs_err": float((out - want).abs().max()) if torch.isfinite(out).all() else None,
                }
            )
            if len(failures) == 1:
                torch.save(
                    {"observed": out.clone(), "expected": want.clone(), **values},
                    args.out / "first_failure.pt",
                )
    finally:
        worker.close()

    record["launches_run"] = exact + len(failures)
    record["exact_matches"] = exact
    record["failures"] = len(failures)
    record["failure_detail"] = failures[:10]
    (args.out / "result.json").write_text(json.dumps(record, indent=2))
    print(
        f"MATRIX arm={args.arm} rows={args.rows} launches={record['launches_run']} "
        f"exact={exact} failures={len(failures)}"
    )
    raise SystemExit(0 if len(failures) == 0 and exact == args.launches else 1)


if __name__ == "__main__":
    main()
