# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run unchanged upstream Qwen prefill with an explicit AutoSync policy on 910B2."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The model modules are imported by their published names, from the pinned
# checkout, unmodified; the library root supplies the `golden` harness.
_LIBRARY = Path(__import__("os").environ["LIBRARY"]).resolve()
sys.path[:0] = [str(_LIBRARY / "models/qwen3_14b"), str(_LIBRARY)]
model = importlib.import_module("prefill_fwd")
run = importlib.import_module("golden").run
from pypto import passes  # noqa: E402
from pypto.backend import BackendType  # noqa: E402
from pypto.runtime import ExecutionMode  # noqa: E402

RECORD: dict = {"outputs": {}}
OUT_DIR: list = [Path(".")]
SAVE_ALWAYS: list = [False]


def ok_preview(af, ef, rtol, atol, a, e, override) -> bool:
    """The verdict, computed before the record is written, for the save rule."""
    if override is not None and override[0] == 0.0 and override[1] == 0.0:
        return bool(torch.equal(a, e))
    return bool(torch.allclose(af, ef, rtol=rtol, atol=atol))


def digest(t: torch.Tensor) -> str:
    c = t.contiguous()
    raw = c.view(torch.int16) if c.dtype is torch.bfloat16 else c
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


# The model's own stated criteria, per entry and output. Recorded in the result
# so the verdict can be read against the criterion that produced it.
CRITERIA = {
    "prefill": {"out": (5e-3, 5e-3, "model entry prefill_fwd.py: rtol=atol=5e-3, torch.allclose")},
}


def recorder(name: str, override: tuple[float, float, str] | None = None):
    """Model-criterion comparator that also records the full diagnostics.

    The verdict is the harness's own ``torch.allclose`` at the supplied
    tolerance, so the pass/fail decision is unchanged; everything else is
    evidence the default comparator prints only on failure.
    """

    def compare(actual, expected, *, rtol, atol, **_):
        if override is not None:
            rtol, atol = override[0], override[1]
        a, e = actual.cpu(), expected.cpu()
        af, ef = a.float(), e.float()
        close = torch.isclose(af, ef, rtol=rtol, atol=atol)
        diff = (af - ef).abs()
        bad = torch.nonzero(~close)
        entry = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "rtol": rtol,
            "atol": atol,
            "elements": int(a.numel()),
            "outliers": int((~close).sum()),
            "max_abs_err": float(diff.max()) if a.numel() else 0.0,
            "mean_abs_err": float(diff.mean()) if a.numel() else 0.0,
            "actual_all_finite": bool(torch.isfinite(af).all()),
            "expected_all_finite": bool(torch.isfinite(ef).all()),
            "actual_sha256": digest(a),
            "expected_sha256": digest(e),
            "first_outlier_coords": bad[:20].tolist(),
            "first_outlier_actual": [float(af[tuple(c)]) for c in bad[:20]],
            "first_outlier_expected": [float(ef[tuple(c)]) for c in bad[:20]],
        }
        entry["criterion"] = override[2] if override else "harness default allclose"
        if a.dtype in (torch.int32, torch.int64):
            entry["exact_equal"] = bool(torch.equal(a, e))
            entry["mismatched_ids"] = int((a != e).sum())
        if name == "out" and a.dim() == 2 and a.shape[1] > 1000:
            # The generation signal, and the margin behind any disagreement.
            win_a, win_e = af.argmax(-1), ef.argmax(-1)
            agree = win_a == win_e
            entry["winning_token_actual"] = win_a.tolist()
            entry["winning_token_expected"] = win_e.tolist()
            entry["winning_token_agreements"] = int(agree.sum())
            entry["winning_token_rows"] = int(win_a.numel())
            margins = []
            for row in torch.nonzero(~agree).flatten().tolist():
                ka, ke = int(win_a[row]), int(win_e[row])
                margins.append(
                    {
                        "row": row,
                        "actual_token": ka,
                        "expected_token": ke,
                        "actual_margin": float(af[row, ka] - af[row, ke]),
                        "expected_margin": float(ef[row, ke] - ef[row, ka]),
                        "row_max_abs_err": float(diff[row].max()),
                    }
                )
            entry["winning_token_margins"] = margins
        RECORD["outputs"].setdefault(name, []).append(entry)
        # Keep the tensors behind any disagreement, and the tensors of a
        # full-depth run either way, so the arm-to-arm comparison the handoff
        # asks for separately can be made from saved bytes rather than from a
        # digest that only says "different".
        if (not ok_preview(af, ef, rtol, atol, a, e, override)) or SAVE_ALWAYS[0]:
            index = len(RECORD["outputs"][name]) - 1
            torch.save({"actual": a, "expected": e}, OUT_DIR[0] / f"tensors_{name}_r{index}.pt")
        if override is not None and override[0] == 0.0 and override[1] == 0.0:
            ok = bool(torch.equal(a, e))
        else:
            ok = bool(torch.allclose(af, ef, rtol=rtol, atol=atol))
        return ok, f"    outliers {entry['outliers']}/{entry['elements']} max_abs_err {entry['max_abs_err']}"

    compare.__name__ = f"recorded_allclose[{name}]"
    return compare


# --------------------------------------------------------------------------- prefill

PREFILL_CASES = {
    #  name          batch max_seq chunk_start chunk_size layers
    "first127": (1, 127, 0, 127, 2),
    "first128": (1, 128, 0, 128, 2),
    "first256": (1, 256, 0, 256, 2),
    "append128": (1, 256, 128, 128, 2),
    "b2first128": (2, 128, 0, 128, 2),
    "full40first128": (1, 128, 0, 128, 40),
}


def prefill_case(model, case: str, layers_override: int | None):
    batch, max_seq, start, size, layers = PREFILL_CASES[case]
    if layers_override:
        layers = layers_override
    specs = model.build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        num_layers=layers,
        use_max_seq=True,
        chunk_start=start,
        chunk_size=size,
    )
    geometry = {
        "case": case,
        "batch": batch,
        "max_seq": max_seq,
        "chunk_start": start,
        "chunk_size": size,
        "layers": layers,
    }
    return specs, geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("B", "Bp", "R"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--entry", required=True, choices=("prefill",))
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--capture", action="store_true", help="dependency capture + swimlane")
    parser.add_argument(
        "--timing", action="store_true", help="swimlane only, no golden: a timed launch, not a checked one"
    )
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    autosync = args.arm in ("B", "Bp")
    args.out.mkdir(parents=True, exist_ok=True)
    OUT_DIR[0] = args.out
    # A full-depth run is the one geometry with no second opinion available, so
    # its outputs are retained unconditionally for the arm-to-arm comparison.
    SAVE_ALWAYS[0] = "full40" in args.case
    torch.manual_seed(args.seed)

    specs, geometry = prefill_case(model, args.case, args.layers)
    entry, golden_fn = model.prefill_fwd, model.golden_qwen3_14b_prefill
    criterion = "model published golden_qwen3_14b_prefill, torch.allclose"

    config = dict(
        backend_type=BackendType.Ascend910,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device,
        dump_passes=False,
        save_kernels=True,
        save_kernels_dir=str(args.out / "build"),
    )
    if args.capture or args.timing:
        config |= {"enable_chip_swimlane": 1}
    if args.capture:
        config |= {"enable_dep_gen": True}

    RECORD.update(
        {
            "arm": args.arm,
            "label": args.label,
            "entry": args.entry,
            "case": args.case,
            "seed": args.seed,
            "repeats": args.repeats,
            "device_id": args.device,
            "geometry": geometry,
            "criterion": criterion,
            "rtol": args.rtol,
            "atol": args.atol,
            "capture": bool(args.capture),
            "timing": bool(args.timing),
            "enable_pypto_autosync": autosync,
            "autosync_policy": "conservative_prefix",
            "criteria": {k: list(v) for k, v in CRITERIA[args.entry].items()},
            "spec_order": [s.name for s in specs],
            "input_digests": {s.name: digest(s.create_tensor()) for s in specs if s.init_value is not None},
        }
    )

    verdicts = []
    for repeat in range(args.repeats):
        print(f"[HARNESS] {args.label} repeat {repeat + 1}/{args.repeats}", flush=True)
        # Fresh specs each repeat: every InOut buffer (the KV cache above all)
        # is rebuilt from its initial value, so repeats start from equivalent
        # state rather than from the previous repeat's updated cache.
        specs, _ = prefill_case(model, args.case, args.layers)
        # The arm lives here. `enable_pypto_autosync` defaults to False when no
        # PassContext is active, so a compile outside one is the stock arm
        # whatever the label says; the context must wrap the compile, which
        # `golden.run` performs internally.
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=autosync,
            autosync_policy="conservative_prefix",
            autosync_disjoint_gm=False,
            enable_scalar_gm_dma=True,
        ):
            result = run(
                fn=entry,
                specs=specs,
                golden_fn=None if args.timing else golden_fn,
                config=config,
                rtol=args.rtol,
                atol=args.atol,
                compare_fn=None
                if args.timing
                else {s.name: recorder(s.name, CRITERIA[args.entry].get(s.name)) for s in specs},
            )
        verdicts.append(
            {
                "repeat": repeat,
                "passed": bool(result.passed),
                "error": result.error,
                "work_dir": str(result.work_dir) if result.work_dir else None,
            }
        )
        if not result.passed:
            print(result.error, flush=True)

    RECORD["verdicts"] = verdicts
    RECORD["passed"] = all(v["passed"] for v in verdicts)
    # Repeat variation: the same arm on the same inputs should land on the same
    # bytes; if it does not, that bound is reported beside the arm comparison.
    for name, entries in RECORD["outputs"].items():
        shas = {e["actual_sha256"] for e in entries}
        RECORD.setdefault("repeat_variation", {})[name] = {
            "launches": len(entries),
            "distinct_outputs": len(shas),
            "bitwise_stable": len(shas) <= 1,
            "max_abs_err_range": [
                min(e["max_abs_err"] for e in entries),
                max(e["max_abs_err"] for e in entries),
            ],
        }
    (args.out / "result.json").write_text(json.dumps(RECORD, indent=2, default=str))
    print(f"HARNESS {args.label} passed={RECORD['passed']} repeats={args.repeats}")
    raise SystemExit(0 if RECORD["passed"] else 1)


if __name__ == "__main__":
    main()
