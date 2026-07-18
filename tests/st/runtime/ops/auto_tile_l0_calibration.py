# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L2 device calibration and L0 trace setup for AutoTileMatmulL0 costs.

This is an opt-in device tool, not a pytest test. It requires the calibration
hook carried by the issue-2079 working branch. Each measured sample runs in a
fresh child process because the L2 swimlane collector is not safely reusable.
The measured value is the sole task's structured ``duration_us`` from
``l2_swimlane_records.json``; converter console summaries are diagnostic only.

``--build-one`` creates a persistent PTOAS build for the repository's
``incore-profiling`` skill. Use those cycle-accurate L0 traces to explain the
MTE1/CUBE/FIXPIPE terms, while the L2 sweep remains the real-device wall-time
calibration target.

Examples:
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite primary --list
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --check-runtime
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite broad --validate
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite primary --samples 5 -d 0
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite broad --samples 3 -d 0 --resume
    python tests/st/runtime/ops/auto_tile_l0_calibration.py --suite primary \
        --build-one issue2079_16x512x128_bf16_bt_current
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
        _case("issue2079", "single_k_os_dbc", shape, "16,128,128,OS,1"),
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
_RUNTIME_SETUP = (
    "python3 -m venv --system-site-packages .venv && source .venv/bin/activate && "
    "export CMAKE_BUILD_PARALLEL_LEVEL=2 && "
    "pip install scikit-build-core nanobind cmake ninja && "
    "pip install --no-build-isolation .[dev] && "
    "pip install --no-build-isolation ./runtime"
)


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

        # The DSL parser intentionally does not constant-fold captured Python
        # booleans and does not support IfExp. Select one complete DSL body at
        # construction time so each parsed program has a single static B layout.
        if c.transpose_b:

            @pl.program
            class TransposedBGemmProgram:
                @pl.function(type=pl.FunctionType.InCore)
                def gemm(
                    self,
                    a: pl.Tensor[[M, K], dtype],
                    b: pl.Tensor[[N, K], dtype],
                    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                ) -> pl.Tensor[[M, N], pl.FP32]:
                    tile_a = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                    tile_b_raw = pl.load(b, offsets=[0, 0], shapes=[N, K], target_memory=pl.MemorySpace.Mat)
                    tile_b = pl.tile.transpose_view(tile_b_raw)
                    return pl.store(pl.matmul(tile_a, tile_b), offsets=[0, 0], output_tensor=out)

                @pl.function(type=pl.FunctionType.Orchestration)
                def orch(
                    self,
                    a: pl.Tensor[[M, K], dtype],
                    b: pl.Tensor[[N, K], dtype],
                    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                ) -> pl.Tensor[[M, N], pl.FP32]:
                    return self.gemm(a, b, out)

            return TransposedBGemmProgram

        @pl.program
        class PlainBGemmProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm(
                self,
                a: pl.Tensor[[M, K], dtype],
                b: pl.Tensor[[K, N], dtype],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                return pl.store(pl.matmul(tile_a, tile_b), offsets=[0, 0], output_tensor=out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[M, K], dtype],
                b: pl.Tensor[[K, N], dtype],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                return self.gemm(a, b, out)

        return PlainBGemmProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["out"][:] = a @ (b.T if self.case.transpose_b else b)


def _capture_stdout(fn):
    saved = os.dup(1)
    try:
        with tempfile.TemporaryFile(mode="w+") as temp:
            try:
                os.dup2(temp.fileno(), 1)
                result = fn()
            finally:
                os.dup2(saved, 1)
            temp.seek(0)
            return result, temp.read()
    finally:
        os.close(saved)


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


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _runtime_preflight() -> tuple[Path, Path]:
    """Require the active venv to own a runtime compatible with this checkout."""
    expected_pin = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-tree", "HEAD", "runtime"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[2]
    runtime_head = subprocess.run(
        ["git", "-C", str(_REPO_ROOT / "runtime"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if runtime_head != expected_pin:
        raise RuntimeError(
            f"runtime submodule is at {runtime_head}, but this PyPTO commit pins {expected_pin}; "
            "run `git submodule update --init --recursive` before calibration"
        )

    try:
        import _task_interface  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        import simpler  # noqa: PLC0415
        from simpler.task_interface import CallConfig  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            f"the pinned Simpler runtime is not importable ({error}); create the worktree-local "
            f"environment with `{_RUNTIME_SETUP}`"
        ) from error

    config = CallConfig()
    required_fields = (
        "enable_l2_swimlane",
        "enable_dump_args",
        "enable_pmu",
        "enable_dep_gen",
        "enable_scope_stats",
        "output_prefix",
    )
    missing = [field for field in required_fields if not hasattr(config, field)]
    simpler_path = Path(simpler.__file__).resolve()
    extension_path = Path(_task_interface.__file__).resolve()
    allowed_python_roots = (Path(sys.prefix).resolve(), (_REPO_ROOT / "runtime").resolve())
    if missing or not any(_path_within(simpler_path, root) for root in allowed_python_roots):
        detail = (
            f"missing CallConfig fields {missing}" if missing else f"simpler imported from {simpler_path}"
        )
        raise RuntimeError(
            f"incompatible Simpler runtime ({detail}). Use a worktree-local venv and install this checkout: "
            f"`{_RUNTIME_SETUP}`"
        )
    if not _path_within(extension_path, Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"_task_interface imported from {extension_path}, outside the active environment {sys.prefix}; "
            "install `./runtime` in the worktree-local venv before calibration"
        )
    return simpler_path, extension_path


def _single_task_duration_us(work_dir: Path) -> tuple[float, int]:
    """Read the sole L2 task's real-device execution duration."""
    from simpler_setup.tools.swimlane_converter import read_perf_data  # noqa: PLC0415

    records_path = work_dir / "dfx_outputs" / "l2_swimlane_records.json"
    if not records_path.is_file():
        raise RuntimeError(f"L2 timing records not found: {records_path}")
    data = read_perf_data(str(records_path))
    tasks = data.get("tasks", [])
    if len(tasks) != 1:
        shapes = [
            {
                "task_id": task.get("task_id"),
                "func_id": task.get("func_id"),
                "duration_us": task.get("duration_us"),
            }
            for task in tasks
        ]
        raise RuntimeError(
            f"expected exactly one dispatched L2 task in {records_path}, got {len(tasks)}: {shapes}"
        )
    duration_us = float(tasks[0].get("duration_us", 0.0))
    if duration_us <= 0.0:
        raise RuntimeError(f"invalid L2 task duration {duration_us} in {records_path}")
    return duration_us, int(tasks[0]["func_id"])


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
    "timing_source",
    "func_id",
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
    _runtime_preflight()
    os.environ["PYPTO_FORCE_L0_TILE"] = case.force
    torch.manual_seed(0)
    terms = _model_terms(case)
    correctness = RunConfig(platform=args.platform, device_id=args.device, rtol=2e-2, atol=2e-2)
    warm_runner = TestRunner(RunConfig(platform=args.platform, device_id=args.device))
    sample_root = Path(args.artifacts).resolve() / case.case_id / f"sample_{sample}"
    measured_runner = TestRunner(
        RunConfig(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=True,
            save_kernels=True,
            save_kernels_dir=str(sample_root),
        )
    )
    for warmup_index in range(args.warmup):
        result, output = _capture_stdout(lambda: warm_runner.run(GemmCase(case, config=correctness)))
        if not result.passed:
            raise RuntimeError(
                f"warmup {warmup_index} failed for {case.case_id}: {result.error}\n{output[-2000:]}"
            )
    result, output = _capture_stdout(lambda: measured_runner.run(GemmCase(case, config=correctness)))
    if not result.passed:
        raise RuntimeError(f"measured run failed for {case.case_id}: {result.error}\n{output[-2000:]}")
    work_dir = sample_root / case.case_id
    exec_us, func_id = _single_task_duration_us(work_dir)
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
        "exec_us": exec_us,
        "count": 1,
        "timing_source": "l2_task_duration_us",
        "func_id": func_id,
        **{
            f"model_{key}" if key in {"load", "mad", "drain", "wall"} else key: value
            for key, value in terms.items()
        },
    }


def _completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _FIELDS:
            raise RuntimeError(
                f"cannot resume {path}: CSV schema is {reader.fieldnames}, expected {_FIELDS}; "
                "write the structured-L2 sweep to a new output file"
            )
        return {(row["case_id"], int(row["sample"])) for row in reader}


def _find_case(cases: list[CalibrationCase], case_id: str) -> CalibrationCase:
    case = next((item for item in cases if item.case_id == case_id), None)
    if case is None:
        available = ", ".join(item.case_id for item in cases)
        raise ValueError(f"unknown case {case_id!r}; available cases: {available}")
    return case


def _build_one(case: CalibrationCase, args) -> Path:
    """Create a persistent PTOAS artifact for the incore-profiling skill."""
    os.environ["PYPTO_FORCE_L0_TILE"] = case.force
    output_root = Path(args.artifacts).resolve() / "l0_inputs"
    runner = TestRunner(
        RunConfig(
            platform=args.platform,
            codegen_only=True,
            save_kernels=True,
            save_kernels_dir=str(output_root),
        )
    )
    config = RunConfig(platform=args.platform, rtol=2e-2, atol=2e-2)
    result, output = _capture_stdout(lambda: runner.run(GemmCase(case, config=config)))
    if not result.passed:
        raise RuntimeError(f"L0 input build failed for {case.case_id}: {result.error}\n{output[-2000:]}")
    work_dir = output_root / case.case_id
    pto_files = list((work_dir / "ptoas").glob("*.pto"))
    if not pto_files:
        raise RuntimeError(f"L0 input build produced no PTOAS units under {work_dir / 'ptoas'}")
    return work_dir


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
        # Interleave configurations by sample so thermal or frequency drift does
        # not systematically favor one design point. Each child still owns a
        # fresh L2 collector because the collector is not safely reusable.
        for sample in range(args.samples):
            for case in cases:
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
    parser.add_argument("--output", default="build_output/auto_tile_l0_calibration.csv")
    parser.add_argument("--artifacts", default="build_output/auto_tile_l0_calibration")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check every forced point and construct both DSL layout variants without a device run",
    )
    parser.add_argument(
        "--build-one",
        metavar="CASE_ID",
        help="build one persistent PTOAS artifact for the incore-profiling skill",
    )
    parser.add_argument("--run-one", default="", help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = _suite(args.suite)
    if args.check_runtime:
        simpler_path, extension_path = _runtime_preflight()
        print(f"runtime preflight OK: simpler={simpler_path}")
        print(f"runtime preflight OK: _task_interface={extension_path}")
        return
    if args.list:
        for case in cases:
            print(f"{case.case_id:70} {case.force}")
        print(f"# {len(cases)} configurations")
        return
    if args.validate:
        for case in cases:
            _model_terms(case)
            GemmCase(case).get_program()
        print(f"validated {len(cases)} forced configurations and DSL programs")
        return
    if args.build_one:
        try:
            case = _find_case(cases, args.build_one)
        except ValueError as error:
            parser.error(str(error))
        work_dir = _build_one(case, args)
        target = "a5" if args.platform.startswith("a5") else "a2a3"
        profiler = _REPO_ROOT / ".claude" / "skills" / "incore-profiling" / "incore_profile.py"
        print(f"BUILD_DIR={work_dir}")
        print(f"python {profiler} --build-dir {work_dir} --target {target} --list-funcs")
        print(f"python {profiler} --build-dir {work_dir} --target {target}")
        return
    if args.samples < 1 or args.warmup < 0:
        parser.error("--samples must be positive and --warmup non-negative")
    if args.run_one:
        try:
            case = _find_case(cases, args.run_one)
        except ValueError as error:
            parser.error(str(error))
        writer = csv.DictWriter(sys.stdout, fieldnames=_FIELDS)
        writer.writerow(_run_one(case, args.sample, args))
        return
    _run_parent(cases, args)


if __name__ == "__main__":
    main()
