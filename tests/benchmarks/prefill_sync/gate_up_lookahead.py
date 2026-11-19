# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Build and run matched Qwen prefill K-loop scheduling experiments."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager
from pypto.runtime import ExecutionMode, RunConfig

from tests.ut.ir.transforms.sync_qwen_prefill_cases import (
    assert_gate_up_body_identity,
    load_qwen_projection,
)


def statements(body: ir.Stmt) -> list[ir.Stmt]:
    return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]


def operation(stmt: ir.Stmt) -> str:
    if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
        return stmt.expr.op.name
    return ""


def loop_body(loop: ir.ForStmt, body: list[ir.Stmt]) -> ir.ForStmt:
    assert not loop.iter_args and not loop.return_vars
    return ir.ForStmt(
        loop.loop_var,
        loop.start,
        loop.stop,
        loop.step,
        [],
        ir.SeqStmts(body, loop.span),
        [],
        loop.span,
        loop.kind,
        loop.attrs,
        loop.hardware_loop_policy,
    )


class Substitute(ir.IRMutator):
    def __init__(self, replacements: dict[ir.Var, ir.Expr]) -> None:
        super().__init__()
        self.replacements = replacements

    def visit_var(self, op: ir.Var) -> ir.Expr:
        return self.replacements.get(op, op)

    def visit_eval_stmt(self, op: ir.EvalStmt) -> ir.Stmt:
        # A physical instruction may occur at several new sites even when its
        # operands are unchanged. Each site needs an independent sync token.
        assert isinstance(op.expr, ir.Call)
        call = op.expr
        args = [self.visit_expr(arg) for arg in call.args]
        return ir.EvalStmt(ir.Call(call.op, args, call.kwargs, call.attrs, call.type, call.span), op.span)


def schedule_projection(projection: ir.ForStmt, *, lookahead: bool) -> ir.ForStmt:
    projection_body = statements(projection.body)
    loops = [i for i, stmt in enumerate(projection_body) if isinstance(stmt, ir.ForStmt)]
    assert len(loops) == 1
    loop_index = loops[0]
    k_loop = projection_body[loop_index]
    assert isinstance(k_loop, ir.ForStmt)
    span = k_loop.span
    start, stop, step = k_loop.start, k_loop.stop, k_loop.step
    assert isinstance(start, ir.ConstInt) and isinstance(stop, ir.ConstInt) and isinstance(step, ir.ConstInt)
    assert (start.value, stop.value, step.value) == (0, 20, 2)
    k_body = statements(k_loop.body)
    definitions = {stmt.var: stmt.value for stmt in k_body if isinstance(stmt, ir.AssignStmt)}
    loads = [i for i, stmt in enumerate(k_body) if operation(stmt) == "buffer.load"]
    inner = [i for i, stmt in enumerate(k_body) if isinstance(stmt, ir.ForStmt)]
    assert len(loads) == 4 and len(inner) == 2 and loads[-1] < inner[0]
    extract_loop, second = (k_body[i] for i in inner)
    assert isinstance(extract_loop, ir.ForStmt) and isinstance(second, ir.ForStmt)
    ext_body = statements(extract_loop.body)
    assert isinstance(extract_loop.start, ir.ConstInt) and extract_loop.start.value == 0
    assert [operation(stmt) for stmt in ext_body[:4]] == ["buffer.extract"] * 4

    def setup(k_value: ir.Expr) -> list[ir.Stmt]:
        addresses = Substitute({k_loop.loop_var: k_value})
        substitute = Substitute(
            {
                extract_loop.loop_var: extract_loop.start,
                **{var: addresses.visit_expr(value) for var, value in definitions.items()},
            }
        )
        return [substitute.visit_stmt(k_body[i]) for i in loads[:2]] + [
            substitute.visit_stmt(stmt) for stmt in ext_body[:4]
        ]

    # Statically split the first and last K groups. This leaves no speculative
    # first or last visit and does not add payload predicates to the steady loop.
    assert isinstance(ext_body[4], ir.IfStmt)
    initial = ext_body[4]
    accumulate = initial.else_body
    assert accumulate is not None and not initial.return_vars
    assert isinstance(extract_loop.step, ir.ConstInt) and extract_loop.step.value == 64
    assert isinstance(extract_loop.stop, ir.ConstInt) and extract_loop.stop.value == 256

    def phase(group: str, start: int, stop: int, *, first: bool, prefetch: bool) -> ir.ForStmt:
        def index(value: int) -> ir.ConstInt:
            return ir.ConstInt(value, DataType.INDEX, span)

        kvar = ir.Var(k_loop.loop_var.name_hint + group, k_loop.loop_var.type, span)
        ko1 = ir.Var(extract_loop.loop_var.name_hint + group, extract_loop.loop_var.type, span)
        ko2 = ir.Var(second.loop_var.name_hint + group, second.loop_var.type, span)
        addresses = Substitute({k_loop.loop_var: kvar})
        change = Substitute(
            {
                k_loop.loop_var: kvar,
                extract_loop.loop_var: ko1,
                second.loop_var: ko2,
                **{var: addresses.visit_expr(value) for var, value in definitions.items()},
            }
        )

        def replace_stmt(stmt: ir.Stmt) -> ir.Stmt:
            return change.visit_stmt(stmt)

        phase_body = [replace_stmt(k_body[i]) for i in (loads[2:] if lookahead and not first else loads)]
        if lookahead and not first:
            phase_body.extend(replace_stmt(s) for s in statements(accumulate))
            phase_body.append(replace_stmt(ext_body[5]))
        extract_body = [*ext_body[:4], *statements(accumulate), *ext_body[5:]]
        if first or not lookahead:
            # The first MMA initializes once. The rest of this inner loop is
            # uniform and performs only accumulator updates.
            zero = Substitute({extract_loop.loop_var: extract_loop.start})
            phase_body.extend(replace_stmt(zero.visit_stmt(s)) for s in ext_body[:4])
            first_mma = initial.then_body if first else accumulate
            phase_body.extend(replace_stmt(s) for s in statements(first_mma))
            phase_body.append(replace_stmt(ext_body[5]))
        phase_body.append(
            ir.ForStmt(
                ko1,
                extract_loop.step,
                extract_loop.stop,
                extract_loop.step,
                [],
                ir.SeqStmts([replace_stmt(s) for s in extract_body], span),
                [],
                span,
            )
        )
        if lookahead and prefetch:
            phase_body.extend(setup(ir.Add(kvar, k_loop.step, DataType.INDEX, span)))
        second_body = ir.SeqStmts([replace_stmt(s) for s in statements(second.body)], span)
        phase_body.append(ir.ForStmt(ko2, second.start, second.stop, second.step, [], second_body, [], span))
        phase_body.append(ir.YieldStmt(span))
        return ir.ForStmt(
            kvar, index(start), index(stop), k_loop.step, [], ir.SeqStmts(phase_body, span), [], span
        )

    projection_body[loop_index : loop_index + 1] = [
        phase("_first", 0, 2, first=True, prefetch=True),
        phase("_steady", 2, 18, first=False, prefetch=True),
        phase("_last", 18, 20, first=False, prefetch=False),
    ]
    return loop_body(projection, projection_body)


def gate_up_schedule(function: ir.Function, *, lookahead: bool) -> ir.Function:
    if function.name != "gate_up_proj":
        return function
    assert function.func_type == ir.FunctionType.AIC
    body = statements(function.body)
    assert sum(isinstance(stmt, ir.ForStmt) for stmt in body) == 2
    rewritten = [
        schedule_projection(stmt, lookahead=lookahead) if isinstance(stmt, ir.ForStmt) else stmt
        for stmt in body
    ]
    return ir.Function(
        function.name,
        list(zip(function.params, function.param_directions, strict=True)),
        function.return_types,
        ir.SeqStmts(rewritten, function.span),
        function.span,
        function.func_type,
        function.level,
        function.role,
        function.attrs,
        function.requires_runtime_binding,
        function.ir_stage,
    )


@contextmanager
def install_experiment(variant):
    """Insert a benchmark-only pass after physical allocation and before RealizeSync."""
    original_init = PassManager.__init__

    def initialize(manager, *args, **kwargs):
        original_init(manager, *args, **kwargs)
        original_passes = manager.passes
        names = [p.get_name() for p in original_passes]
        assert names[-2:] == ["LowerTileToBuffer", "RealizeSync"], names
        if variant != "baseline":
            pipeline = passes.PassPipeline()
            for existing in original_passes[:-1]:
                pipeline.add_pass(existing)
            pipeline.add_pass(
                passes.create_function_pass(
                    lambda function: gate_up_schedule(function, lookahead=variant == "lookahead"),
                    "QwenGateUpLookahead",
                )
            )
            pipeline.add_pass(original_passes[-1])
            manager._pipeline = pipeline

    with patch.object(PassManager, "__init__", initialize):
        yield


def build(variant, band, args, config):
    """Build one arm with the ordinary Qwen orchestration and runtime ABI."""
    assert variant in ("baseline", "peeled", "lookahead")
    if config.save_kernels_dir is None or Path(config.save_kernels_dir).exists():
        raise ValueError("Use a fresh save_kernels_dir to compile each benchmark arm")
    # An explicit output directory bypasses the normal JIT cache. The local
    # experiment pass is intentionally outside the production cache identity.
    assert_gate_up_body_identity()
    _, entry = load_qwen_projection("gate_up", layers=1, band=band, hidden=5120)
    with (
        passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=True,
            autosync_policy="exact",
            autosync_disjoint_gm=True,
        ),
        install_experiment(variant),
    ):
        return entry.compile(*args, config=config)


def tensor_contract():
    return (
        ((256, 4352), torch.float32),
        ((128, 5120), torch.bfloat16),
        ((5120, 17408), torch.bfloat16),
        ((5120, 17408), torch.bfloat16),
    )


def device_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    out_spec, act_spec, gate_spec, up_spec = tensor_contract()
    output = torch.full(out_spec[0], float("nan"), dtype=out_spec[1])
    activations = tuple(torch.randn(act_spec[0], dtype=act_spec[1], generator=generator) for _ in range(2))
    weights = tuple(
        torch.randn(shape, dtype=dtype, generator=generator).mul_(0.0625)
        for shape, dtype in (gate_spec, up_spec)
    )
    return output, activations, weights


def expected_output(activation, weights, *, band):
    """Independent FP32 products from the same BF16 inputs."""
    first_column = band * 4352
    columns = slice(first_column, first_column + 4352)
    activation = activation.float()
    return torch.cat(tuple(activation @ weight[:, columns].float() for weight in weights), dim=0)


def run_on_device(compiled, options):
    output, activations, weights = device_inputs(options.seed)
    run = RunConfig(
        arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=options.device_id
    )
    maximum = [0.0, 0.0]
    if options.correctness_runs:
        expected = [expected_output(a, weights, band=options.band) for a in activations]
        for visit in range(options.correctness_runs):
            case = visit % 2
            output.fill_(float("nan"))
            compiled(output, activations[case], *weights, config=run)
            torch.testing.assert_close(output, expected[case], rtol=0.02, atol=0.02)
            differences = (output - expected[case]).abs()
            for i, half in enumerate(differences.split(128)):
                maximum[i] = max(maximum[i], half.max().item())
    if options.profile_once:
        profile_run = RunConfig(
            arch=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            device_id=options.device_id,
            enable_chip_swimlane=1,
        )
        output.fill_(float("nan"))
        compiled(output, activations[0], *weights, config=profile_run)
    print(
        json.dumps(
            {
                "arm": options.variant,
                "band": options.band,
                "correctness_runs": options.correctness_runs,
                "profiled": options.profile_once,
                "max_abs_by_projection": maximum,
            }
        )
    )


def main():
    if not __debug__:
        raise RuntimeError("Run the benchmark with Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("baseline", "peeled", "lookahead"))
    parser.add_argument("--band", type=int, required=True, choices=range(4))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--seed", type=int, default=5765)
    parser.add_argument("--correctness-runs", type=int, default=0)
    parser.add_argument("--profile-once", action="store_true")
    options = parser.parse_args()
    assert options.correctness_runs >= 0
    if options.correctness_runs or options.profile_once:
        assert options.device_id is not None, "A device ID is required for execution"
    args = [torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in tensor_contract()]
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    compiled = build(
        options.variant,
        options.band,
        args,
        RunConfig(
            arch=BackendType.Ascend910,
            execution_mode=ExecutionMode.SIM if options.device_id is None else ExecutionMode.ONBOARD,
            codegen_only=True,
            save_kernels_dir=str(options.output),
        ),
    )
    report = json.loads((options.output / "report/autosync.json").read_text())
    assert any(
        row["function"] == "gate_up_proj" and row["selected_route"] == "pypto" for row in report["kernels"]
    ), report
    if options.device_id is not None and (options.correctness_runs or options.profile_once):
        run_on_device(compiled, options)


if __name__ == "__main__":
    main()
