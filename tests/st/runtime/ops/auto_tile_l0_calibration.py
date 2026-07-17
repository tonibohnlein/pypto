# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device calibration sweep for AutoTileMatmulL0 issue/setup costs.

This is an opt-in device tool, not a pytest test. It requires the calibration
hook carried by the issue-2079 working branch. Each measured sample runs in a
fresh child process because the swimlane collector is not safely reusable.

Examples:
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite primary --list
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite primary --samples 5 -d 0
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite broad --samples 3 -d 0 --resume
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_ST_ROOT))

import pypto.language as pl  # noqa: E402
import torch  # noqa: E402
from harness.core.harness import DataType, PTOTestCase, TensorSpec  # noqa: E402
from harness.core.test_runner import TestRunner  # noqa: E402
from pypto.pypto_core import passes  # noqa: E402
from pypto.runtime import RunConfig  # noqa: E402


@dataclass(frozen=True)
class CalibrationCase:
    category: str
    label: str
    M: int
    N: int
    K: int
    dtype: str
    force: str
    transpose_b: bool = True

    @property
    def case_id(self) -> str:
        layout = "bt" if self.transpose_b else "plain"
        return f"{self.category}_{self.M}x{self.N}x{self.K}_{self.dtype}_{layout}_{self.label}"


def _case(
    category: str,
    label: str,
    shape: tuple[int, int, int],
    force: str,
    *,
    dtype: str = "bf16",
    transpose_b: bool = True,
) -> CalibrationCase:
    return CalibrationCase(category, label, *shape, dtype, force, transpose_b)


def _primary_cases() -> list[CalibrationCase]:
    shape = (16, 512, 128)
    return [
        _case("issue2079", "current", shape, "16,512,32,OS,0"),
        _case("issue2079", "intermediate", shape, "16,256,64,OS,0"),
        _case("issue2079", "single_k_os", shape, "16,128,128,OS,0"),
        _case("issue2079", "single_k_a", shape, "16,128,128,A,0"),
        _case("issue2079", "single_k_b", shape, "16,256,128,B,0"),
    ]


def _broad_cases() -> list[CalibrationCase]:
    cases = _primary_cases()

    # Wide-N threshold: hold M=16,K=128 and move through the point where the
    # chooser starts shortening K. Include the single-K OS and B-stationary routes.
    n_sweep = {
        256: [("wide", "16,256,64,OS,0"), ("single_k", "16,128,128,OS,0"), ("single_k_b", "16,256,128,B,0")],
        384: [
            ("wide", "16,384,32,OS,0"),
            ("mid", "16,192,64,OS,0"),
            ("single_k", "16,128,128,OS,0"),
            ("single_k_b", "16,256,128,B,0"),
        ],
        768: [
            ("wide", "16,768,16,OS,0"),
            ("mid32", "16,384,32,OS,0"),
            ("mid64", "16,256,64,OS,0"),
            ("single_k", "16,128,128,OS,0"),
            ("single_k_b", "16,256,128,B,0"),
        ],
        1024: [
            ("wide", "16,1024,16,OS,0"),
            ("mid32", "16,512,32,OS,0"),
            ("mid64", "16,256,64,OS,0"),
            ("single_k", "16,128,128,OS,0"),
            ("single_k_b", "16,256,128,B,0"),
        ],
    }
    for n, points in n_sweep.items():
        cases.extend(_case("n_sweep", label, (16, n, 128), force) for label, force in points)

    # K depth: distinguish extract setup from serial-accumulator dependency.
    k_sweep = {
        64: [("blocked", "16,512,32,OS,0"), ("single_k", "16,256,64,OS,0"), ("single_k_b", "16,512,64,B,0")],
        256: [
            ("k32", "16,512,32,OS,0"),
            ("k64", "16,256,64,OS,0"),
            ("k128", "16,128,128,OS,0"),
            ("single_k", "16,64,256,OS,0"),
            ("single_k_b", "16,128,256,B,0"),
        ],
    }
    for k, points in k_sweep.items():
        cases.extend(_case("k_sweep", label, (16, 512, k), force) for label, force in points)

    # Existing device-calibrated counterexample: K blocking must remain available.
    counter_shape = (16, 256, 512)
    for label, force in [
        ("k64", "16,256,64,OS,0"),
        ("k128", "16,128,128,OS,0"),
        ("k256", "16,64,256,OS,0"),
        ("single_k", "16,32,512,OS,0"),
        ("single_k_b", "16,64,512,B,0"),
    ]:
        cases.append(_case("counterexample", label, counter_shape, force))

    # M dependence: check that a coefficient fitted at M=16 transfers to fuller cube rows.
    for m, points in {
        32: [
            ("k32", "32,512,32,OS,0"),
            ("k64", "32,256,64,OS,0"),
            ("single_k", "32,128,128,OS,0"),
            ("single_k_dbc", "16,128,128,OS,1"),
            ("single_k_b", "32,256,128,B,0"),
        ],
        64: [
            ("k32", "64,512,32,OS,0"),
            ("k64", "64,256,64,OS,0"),
            ("single_k", "64,128,128,OS,0"),
            ("single_k_dbc", "32,128,128,OS,1"),
            ("single_k_b", "64,256,128,B,0"),
        ],
        128: [
            ("k32", "64,512,32,OS,0"),
            ("k64", "128,256,64,OS,0"),
            ("single_k", "128,128,128,OS,0"),
            ("single_k_dbc", "64,128,128,OS,1"),
            ("single_k_b", "128,256,128,B,0"),
        ],
    }.items():
        cases.extend(_case("m_sweep", label, (m, 512, 128), force) for label, force in points)

    # Dtype and source-layout controls.
    fp32_shape = (16, 512, 128)
    for label, force in [
        ("k16", "16,512,16,OS,0"),
        ("k32", "16,256,32,OS,0"),
        ("k64", "16,128,64,OS,0"),
        ("single_k", "16,64,128,OS,0"),
        ("single_k_b", "16,128,128,B,0"),
    ]:
        cases.append(_case("dtype", label, fp32_shape, force, dtype="fp32"))
    for base in _primary_cases():
        if base.label != "single_k_a":
            cases.append(_case("layout", base.label, (base.M, base.N, base.K), base.force, transpose_b=False))

    # Fuller square control for stationarity and aspect effects.
    square = (256, 256, 128)
    for label, force in [
        ("wide_n", "128,256,64,OS,0"),
        ("wide_m", "256,128,64,OS,0"),
        ("single_k", "128,128,128,OS,0"),
        ("single_k_dbc", "128,128,128,OS,1"),
        ("single_k_a", "256,128,128,A,0"),
        ("single_k_b", "128,256,128,B,0"),
    ]:
        cases.append(_case("square", label, square, force, transpose_b=False))

    seen: set[str] = set()
    unique: list[CalibrationCase] = []
    for item in cases:
        if item.case_id not in seen:
            seen.add(item.case_id)
            unique.append(item)
    return unique


def _suite(name: str) -> list[CalibrationCase]:
    return _primary_cases() if name == "primary" else _broad_cases()


_PL_DTYPE = {"bf16": pl.BF16, "fp32": pl.FP32}
_DATA_TYPE = {"bf16": DataType.BF16, "fp32": DataType.FP32}


class GemmCase(PTOTestCase):
    __test__ = False

    def __init__(self, case: CalibrationCase, *, config: RunConfig | None = None):
        super().__init__(config, enable_pypto_l0c_double_buffer=case.force.endswith(",1"))
        self.case = case

    def get_name(self) -> str:
        return self.case.case_id

    def define_tensors(self):
        c = self.case
        b_shape = [c.N, c.K] if c.transpose_b else [c.K, c.N]
        return [
            TensorSpec("a", [c.M, c.K], _DATA_TYPE[c.dtype], init_value=torch.randn),
            TensorSpec("b", b_shape, _DATA_TYPE[c.dtype], init_value=torch.randn),
            TensorSpec("out", [c.M, c.N], DataType.FP32, is_output=True),
        ]

    def get_program(self):
        c = self.case
        M, N, K, dtype = c.M, c.N, c.K, _PL_DTYPE[c.dtype]
        b_shape = [N, K] if c.transpose_b else [K, N]
        transpose_b = c.transpose_b

        @pl.program
        class GemmProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm(
                self,
                a: pl.Tensor[[M, K], dtype],
                b: pl.Tensor[b_shape, dtype],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b_raw = pl.load(b, offsets=[0, 0], shapes=b_shape, target_memory=pl.MemorySpace.Mat)
                tile_b = pl.tile.transpose_view(tile_b_raw) if transpose_b else tile_b_raw
                return pl.store(pl.matmul(tile_a, tile_b), offsets=[0, 0], output_tensor=out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[M, K], dtype],
                b: pl.Tensor[b_shape, dtype],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                return self.gemm(a, b, out)

        return GemmProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["out"][:] = a @ (b.T if self.case.transpose_b else b)


def _capture_stdout(fn) -> str:
    saved = os.dup(1)
    temp = tempfile.TemporaryFile(mode="w+")
    try:
        os.dup2(temp.fileno(), 1)
        try:
            fn()
        finally:
            os.dup2(saved, 1)
    finally:
        os.close(saved)
    temp.seek(0)
    return temp.read()


_TOTAL_RE = re.compile(r"^TOTAL\s+(\d+)\s+([\d.]+)\s+([\d.]+)", re.MULTILINE)
_TERM_RE = re.compile(
    r"load=(?P<load>\d+) mad=(?P<mad>\d+) drain=(?P<drain>\d+) wall=(?P<wall>\d+).*"
    r"a_bytes=(?P<a_bytes>\d+) b_bytes=(?P<b_bytes>\d+) a_extracts=(?P<a_extracts>\d+) "
    r"b_extracts=(?P<b_extracts>\d+) acc_continuations=(?P<acc_continuations>\d+)"
)


def _model_terms(case: CalibrationCase) -> dict[str, int]:
    cfg = passes.l0_tile_chooser.L0TileConfig()
    cfg.M, cfg.N, cfg.K = case.M, case.N, case.K
    cfg.l0a_bytes = cfg.l0b_bytes = 64 * 1024
    cfg.l0c_bytes = 128 * 1024
    cfg.bytes_a = cfg.bytes_b = 2 if case.dtype == "bf16" else 4
    cfg.bytes_c = 4
    cfg.min_m = cfg.min_n = cfg.min_k = 16
    cfg.align_m = cfg.align_n = cfg.align_k = 16
    cfg.allow_a_stationary = True
    cfg.allow_b_stationary = True
    cfg.allow_double_buffer_c = True
    cfg.allow_k_boundary = True
    os.environ["PYPTO_FORCE_L0_TILE"] = case.force
    result = passes.l0_tile_chooser.choose_l0_tile(cfg)
    match = _TERM_RE.search(result.perf_hint)
    if not match:
        raise RuntimeError(f"Cannot parse calibration model terms: {result.perf_hint}")
    return {name: int(value) for name, value in match.groupdict().items()}


_FIELDS = [
    "case_id",
    "sample",
    "category",
    "label",
    "M",
    "N",
    "K",
    "dtype",
    "layout",
    "force",
    "exec_us",
    "count",
    "model_load",
    "model_mad",
    "model_drain",
    "model_wall",
    "a_bytes",
    "b_bytes",
    "a_extracts",
    "b_extracts",
    "acc_continuations",
]


def _run_one(case: CalibrationCase, sample: int, args) -> dict[str, str | int | float]:
    os.environ["PYPTO_FORCE_L0_TILE"] = case.force
    torch.manual_seed(0)
    terms = _model_terms(case)
    correctness = RunConfig(platform=args.platform, device_id=args.device, rtol=2e-2, atol=2e-2)
    warm_runner = TestRunner(RunConfig(platform=args.platform, device_id=args.device))
    measured_runner = TestRunner(
        RunConfig(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=True,
            save_kernels=True,
            save_kernels_dir=str(Path(args.artifacts) / case.case_id / f"sample_{sample}"),
        )
    )
    for _ in range(args.warmup):
        _capture_stdout(lambda: warm_runner.run(GemmCase(case, config=correctness)))
    output = _capture_stdout(lambda: measured_runner.run(GemmCase(case, config=correctness)))
    total = _TOTAL_RE.search(output)
    if not total:
        raise RuntimeError(f"No TOTAL timing row for {case.case_id}; output tail:\n{output[-2000:]}")
    return {
        "case_id": case.case_id,
        "sample": sample,
        "category": case.category,
        "label": case.label,
        "M": case.M,
        "N": case.N,
        "K": case.K,
        "dtype": case.dtype,
        "layout": "bt" if case.transpose_b else "plain",
        "force": case.force,
        "exec_us": float(total.group(2)),
        "count": int(total.group(1)),
        **{
            f"model_{key}" if key in {"load", "mad", "drain", "wall"} else key: value
            for key, value in terms.items()
        },
    }


def _completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {(row["case_id"], int(row["sample"])) for row in csv.DictReader(stream)}


def _run_parent(cases: list[CalibrationCase], args) -> None:
    output_path = Path(args.output)
    done = _completed(output_path) if args.resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or not args.resume
    mode = "a" if args.resume and output_path.exists() else "w"
    by_id = {case.case_id: case for case in cases}
    with output_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        for case in cases:
            for sample in range(args.samples):
                if (case.case_id, sample) in done:
                    continue
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--suite",
                    args.suite,
                    "--run-one",
                    case.case_id,
                    "--sample",
                    str(sample),
                    "--device",
                    str(args.device),
                    "--platform",
                    args.platform,
                    "--warmup",
                    str(args.warmup),
                    "--artifacts",
                    args.artifacts,
                ]
                result = subprocess.run(command, check=False, text=True, capture_output=True)
                if result.returncode != 0:
                    sys.stderr.write(result.stderr)
                    raise RuntimeError(f"Calibration child failed for {case.case_id} sample {sample}")
                rows = list(csv.DictReader(result.stdout.splitlines(), fieldnames=_FIELDS))
                if not rows:
                    raise RuntimeError(f"Calibration child returned no CSV row: {result.stdout}")
                writer.writerow(rows[-1])
                stream.flush()
                print(f"[{case.case_id} sample={sample}] {rows[-1]['exec_us']} us", flush=True)
    if set(by_id) - {case_id for case_id, _ in _completed(output_path)}:
        raise RuntimeError("Calibration output is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=["primary", "broad"], default="primary")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("--output", default="autotile_l0_calibration.csv")
    parser.add_argument("--artifacts", default="autotile_l0_calibration_artifacts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--validate", action="store_true", help="check every forced point without a device run"
    )
    parser.add_argument("--run-one", default="", help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = _suite(args.suite)
    if args.list:
        for case in cases:
            print(f"{case.case_id:70} {case.force}")
        print(f"# {len(cases)} configurations")
        return
    if args.validate:
        for case in cases:
            _model_terms(case)
        print(f"validated {len(cases)} forced configurations")
        return
    if args.samples < 1 or args.warmup < 0:
        parser.error("--samples must be positive and --warmup non-negative")
    if args.run_one:
        case = next((item for item in cases if item.case_id == args.run_one), None)
        if case is None:
            parser.error(f"unknown --run-one case: {args.run_one}")
        writer = csv.DictWriter(sys.stdout, fieldnames=_FIELDS)
        writer.writerow(_run_one(case, args.sample, args))
        return
    _run_parent(cases, args)


if __name__ == "__main__":
    main()
