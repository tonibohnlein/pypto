# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical debugging of the default Buffer pipeline and its pass dumps."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.decorator import jit
from pypto.runtime.runner import RunConfig

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


@jit.incore
def _arithmetic(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    rows, cols = a.shape
    x = pl.load(a, [0, 0], [rows, cols])
    y = pl.load(b, [0, 0], [rows, cols])
    pl.store(pl.mul(pl.add(x, y), y), [0, 0], out)
    return out


@jit
def buffer_debug(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _arithmetic(a, b, out)


@jit
def matrix_debug(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:, :] = pl.matmul(a[:, :], b[:, :], b_trans=True, out_dtype=pl.FP32)
    return out


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("planner", _PLANNERS)
def test_public_lower_and_compiled_pass_dumps_remain_numerically_debuggable(
    tmp_path, monkeypatch, planner, pass_verification_instruments, matrix
):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32) / 32
    b = torch.linspace(-1, 1, 512).reshape(16, 32)
    out = torch.empty((16, 16)) if matrix else torch.empty_like(a)
    expected = a @ b.T if matrix else (a + b) * b
    kernel = matrix_debug if matrix else buffer_debug
    config = RunConfig(
        platform="a2a3sim",
        save_kernels_dir=str(tmp_path / "compiled"),
        dump_passes=True,
    )
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        program = kernel.lower(a, b, out, config=config)
    assert any(function.ir_stage == ir.FunctionIRStage.Buffer for function in program.functions.values())
    entry = next(
        function for function in program.functions.values() if ir.is_orchestration_like(function.func_type)
    )
    namespace = {}
    exec(torch_codegen(program, check_shapes=True), namespace)  # noqa: S102
    namespace[entry.name](a, b, out)
    torch.testing.assert_close(out, expected)

    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        compiled = kernel.compile(a, b, out, config=config)
    final_dump = next((compiled.output_dir / "passes_dump").glob("*LowerTileToBuffer.py"))
    final_program = ir.deserialize(final_dump.with_suffix(".msgpack").read_bytes())
    assert isinstance(final_program, ir.Program)
    assert any(
        function.ir_stage == ir.FunctionIRStage.Buffer for function in final_program.functions.values()
    )
    compiled.validate_ir({"a": a, "b": b, "out": torch.empty_like(out)}, {"out": expected})


@jit
def hf32_matrix_debug(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor], mode: pl.constexpr):
    with pl.at(level=pl.Level.CORE_GROUP):
        product = pl.matmul(a[:, :], b[:, :], tf32_mode=mode)
        accumulated = pl.matmul_acc(product, a[:, :], b[:, :], tf32_mode=mode)
        out[:, :] = pl.matmul_acc(accumulated, a[:, :], b[:, :])
    return out


@pytest.mark.parametrize("mode", ["round_even", "round_away"])
def test_hf32_compiled_pass_dumps_preserve_rounding(tmp_path, monkeypatch, mode):
    """validate_ir must accept HF32 results throughout the default A5 pipeline."""
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    a = torch.zeros((16, 16))
    b = torch.zeros((16, 16))
    a[:, 0], a[:, 1] = 1 + 2**-11, -1
    b[0, :], b[1, :] = 1024, 1024
    # Each HF32 product is 0 (ties-even) or 1 (ties-away). The final FP32
    # product is 0.5 and must not inherit rounding from either preceding op.
    expected = torch.full((16, 16), 0.5 if mode == "round_even" else 2.5)
    config = RunConfig(platform="a5sim", save_kernels_dir=str(tmp_path / "compiled"), dump_passes=True)
    with passes.PassContext([], enable_buffer_ir=True):
        compiled = hf32_matrix_debug.compile(a, b, torch.empty_like(expected), mode, config=config)
    compiled.validate_ir(
        {"a": a, "b": b, "out": torch.empty_like(expected)}, {"out": expected}, rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", ["round_even", "round_away"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_hf32_buffer_matmul_reference(mode, accumulate):
    """Exercise explicit Buffer IR; A5's default pipeline currently selects Tile IR."""
    span = ir.Span.unknown()
    lhs = ir.Var("lhs", ir.BufferType([16, 16], pl.FP32, ir.MemorySpace.Left), span)
    rhs = ir.Var("rhs", ir.BufferType([16, 16], pl.FP32, ir.MemorySpace.Right), span)
    out = ir.Var("out", ir.BufferType([16, 16], pl.FP32, ir.MemorySpace.Acc), span)
    op = "buffer.matmul_acc" if accumulate else "buffer.matmul"
    call = ir.Call(ir.get_op(op), [lhs, rhs, out], {"tf32_mode": mode}, ir.VoidType(), span)
    function = ir.Function(
        "kernel", [lhs, rhs, out], [], ir.EvalStmt(call, span), span, ir_stage=ir.FunctionIRStage.Buffer
    )
    namespace = {}
    exec(torch_codegen(function), namespace)  # noqa: S102
    a, b = torch.zeros((16, 16)), torch.zeros((16, 16))
    # Put a halfway value in each operand; rounding only one side is insufficient.
    a[:, 0], a[:, 1], a[:, 2] = 1 + 2**-11, -1, 1024
    b[0, :], b[1, :], b[2, :] = 1024, 1024, 1 + 2**-11
    result = torch.full((16, 16), 3 + 2**-11)
    for value in (a, b, result):
        namespace["_buffer_set_valid"](value, (16, 16))
    namespace["kernel"](a, b, result)
    product = 1024 if mode == "round_even" else 1026
    expected = product + (3 + 2**-11 if accumulate else 0)
    torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)


def test_buffer_storage_views_alias_bytes_and_destination_writes():
    span = ir.Span.unknown()
    byte_type = ir.BufferType([4, 32], pl.UINT8, ir.MemorySpace.Vec)
    window_type = ir.BufferType([2, 32], pl.UINT8, ir.MemorySpace.Vec)
    value_type = ir.BufferType([2, 8], pl.FP32, ir.MemorySpace.Vec)
    storage = ir.Var("storage", byte_type, span)
    window = ir.Var("window", window_type, span)
    value = ir.Var("value", value_type, span)
    alias = ir.Var("alias", value_type, span)
    src = ir.Var("src", ir.TensorType([2, 8], pl.FP32), span)
    out = ir.Var("out", ir.TensorType([2, 8], pl.FP32), span)

    def dims(*values):
        return ir.MakeTuple([ir.ConstInt(value, pl.INDEX, span) for value in values], span)

    def call(name, args, type_):
        return ir.Call(ir.get_op(name), args, {}, type_, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(storage, call("buffer.alloc", [dims()], byte_type), span),
            ir.AssignStmt(window, call("buffer.subview", [storage, dims(2, 0)], window_type), span),
            ir.AssignStmt(value, call("buffer.reshape", [window], value_type), span),
            ir.AssignStmt(alias, call("buffer.reshape", [window], value_type), span),
            ir.EvalStmt(call("buffer.load", [src, dims(0, 0), dims(2, 8), value], ir.VoidType()), span),
            ir.EvalStmt(call("buffer.add", [value, value, value], ir.VoidType()), span),
            ir.EvalStmt(call("buffer.store", [alias, dims(0, 0), dims(2, 8), out], ir.VoidType()), span),
        ],
        span,
    )
    function = ir.Function("kernel", [src, out], [], body, span, ir_stage=ir.FunctionIRStage.Buffer)
    namespace = {}
    exec(torch_codegen(function), namespace)  # noqa: S102
    source = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    output = torch.empty_like(source)
    namespace["kernel"](source, output)
    torch.testing.assert_close(output, source * 2)


def test_buffer_scalar_read_keeps_its_value_after_an_overwrite():
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([2, 8], pl.FP32)
    buffer_type = ir.BufferType([2, 8], pl.FP32, ir.MemorySpace.Vec)
    source = ir.Var("source", tensor_type, span)
    output = ir.Var("output", tensor_type, span)
    buffer = ir.Var("buffer", buffer_type, span)
    saved = ir.Var("saved", ir.ScalarType(pl.FP32), span)

    def dims(*values):
        return ir.MakeTuple([ir.ConstInt(value, pl.INDEX, span) for value in values], span)

    def effect(name, *args):
        return ir.EvalStmt(ir.Call(ir.get_op(name), list(args), {}, ir.VoidType(), span), span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(buffer, ir.Call(ir.get_op("buffer.alloc"), [dims()], {}, buffer_type, span), span),
            effect("buffer.load", source, dims(0, 0), dims(2, 8), buffer),
            ir.AssignStmt(
                saved, ir.Call(ir.get_op("buffer.read"), [buffer, dims(0, 1)], {}, saved.type, span), span
            ),
            effect("buffer.write", buffer, dims(0, 1), ir.ConstFloat(43.0, pl.FP32, span)),
            effect("buffer.write", buffer, dims(1, 3), saved),
            effect("buffer.store", buffer, dims(0, 0), dims(2, 8), output),
        ],
        span,
    )
    function = ir.Function("kernel", [source, output], [], body, span, ir_stage=ir.FunctionIRStage.Buffer)
    namespace = {}
    exec(torch_codegen(function), namespace)  # noqa: S102
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    actual = torch.empty_like(inputs)
    namespace["kernel"](inputs, actual)
    expected = inputs.clone()
    expected[0, 1] = 43.0
    expected[1, 3] = inputs[0, 1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    "valid, count, load_offset, store_offset",
    [("32", 32, 0, 0), ("13", 13, 8, 16), ("n", 0, 8, 16), ("n", 1, 8, 16), ("n", 24, 8, 16)],
    ids=["full", "static-partial", "runtime-empty", "runtime-single", "runtime-partial"],
)
def test_default_rank_one_buffer_transfer(valid, count, load_offset, store_offset):
    program = pl.parse_program(f"""
@pl.program
class RankOneTransfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, source: pl.Tensor[[64], pl.FP32], output: pl.Out[pl.Tensor[[64], pl.FP32]],
        n: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[64], pl.FP32]:
        tile = pl.load(source, [{load_offset}], [32], valid_shape=[{valid}])
        return pl.store(tile, [{store_offset}], output)
""")
    lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    kernel = lowered.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Buffer
    namespace = {}
    exec(torch_codegen(lowered, check_shapes=True), namespace)  # noqa: S102
    source = torch.arange(64, dtype=torch.float32)
    output = torch.full_like(source, -1)
    # The local row has capacity 32. Only the valid GM window may change.
    expected = output.clone()
    expected[store_offset : store_offset + count] = source[load_offset : load_offset + count]
    namespace["kernel"](source, output, count)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("buffer_ir", [False, True], ids=["tile", "buffer"])
@pytest.mark.parametrize(
    "dtype, torch_dtype, large_integer, half_step",
    [("FP16", torch.float16, 1025, 2**-12), ("FP32", torch.float32, 8388609, 2**-25)],
    ids=["fp16", "fp32"],
)
def test_default_cast_round_away_from_zero(buffer_ir, dtype, torch_dtype, large_integer, half_step):
    program = pl.parse_program(f"""
@pl.program
class RoundAway:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, source: pl.Tensor[[2, 32], pl.{dtype}], output: pl.Out[pl.Tensor[[2, 32], pl.INT32]],
    ) -> pl.Tensor[[2, 32], pl.INT32]:
        tile = pl.load(source, [0, 0], [2, 32])
        return pl.store(pl.cast(tile, pl.INT32, mode="round"), [0, 0], output)
""")
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    expected_stage = ir.FunctionIRStage.Buffer if buffer_ir else ir.FunctionIRStage.Functional
    kernel = lowered.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == expected_stage
    namespace = {}
    exec(torch_codegen(lowered, check_shapes=True), namespace)  # noqa: S102
    # Adding 0.5 in the source dtype can change an odd integer or turn a
    # fraction immediately below 0.5 into a value that rounds up.
    below_half = 0.5 - half_step
    source = torch.tensor(
        [large_integer, -large_integer, 0.5, -0.5, 2.5, -2.5, below_half, -below_half] * 8,
        dtype=torch_dtype,
    ).reshape(2, 32)
    expected = torch.tensor([large_integer, -large_integer, 1, -1, 3, -3, 0, 0] * 8, dtype=torch.int32)
    output = torch.empty_like(source, dtype=torch.int32)
    namespace["kernel"](source, output)
    torch.testing.assert_close(output, expected.reshape(2, 32), rtol=0, atol=0)


@pytest.mark.parametrize("storage", ["allocated", "addressed", "reshaped"])
def test_uint32_scratch_replay_without_torch_uint32(monkeypatch, storage):
    """Scratch keeps four-byte elements even when Torch has no unsigned dtype."""
    span = ir.Span.unknown()
    scratch_type = ir.BufferType([1, 16], pl.UINT32, ir.MemorySpace.Vec)
    value_type = ir.BufferType([1, 16], pl.FP32, ir.MemorySpace.Vec)
    scratch = ir.Var("scratch", scratch_type, span)
    value = ir.Var("value", value_type, span)
    out = ir.Var("out", ir.TensorType([1, 16], pl.FP32), span)

    def dims(*values):
        return ir.MakeTuple([ir.ConstInt(value, pl.INDEX, span) for value in values], span)

    def call(name, args, type_):
        return ir.Call(ir.get_op(name), args, {}, type_, span)

    statements = []
    if storage == "reshaped":
        byte_type = ir.BufferType([2, 32], pl.UINT8, ir.MemorySpace.Vec)
        root = ir.Var("root", byte_type, span)
        statements.append(ir.AssignStmt(root, call("buffer.alloc", [dims()], byte_type), span))
        allocation = call("buffer.reshape", [root], scratch_type)
    else:
        args = [dims(), ir.ConstInt(0, pl.INDEX, span)] if storage == "addressed" else [dims()]
        allocation = call("buffer.alloc", args, scratch_type)
    statements.extend(
        [
            ir.AssignStmt(scratch, allocation, span),
            # This equal-width view also rejects an accidental INT64 fallback.
            ir.AssignStmt(value, call("buffer.reshape", [scratch], value_type), span),
            ir.EvalStmt(
                call("buffer.full", [ir.ConstFloat(1.25, pl.FP32, span), value], ir.VoidType()), span
            ),
            ir.EvalStmt(call("buffer.store", [value, dims(0, 0), dims(1, 16), out], ir.VoidType()), span),
        ]
    )
    function = ir.Function(
        "kernel", [out], [], ir.SeqStmts(statements, span), span, ir_stage=ir.FunctionIRStage.Buffer
    )
    result = torch.empty((1, 16))
    with monkeypatch.context() as legacy_torch:
        legacy_torch.delattr(torch, "uint32", raising=False)
        namespace = {}
        exec(torch_codegen(function), namespace)  # noqa: S102
        namespace["kernel"](result)
    torch.testing.assert_close(result, torch.full_like(result, 1.25), rtol=0, atol=0)


@pytest.mark.parametrize("address", [0, 64])
@pytest.mark.parametrize("valid_rows", [1, 2])
def test_addressed_vec_allocations_share_partial_storage(address, valid_rows):
    span = ir.Span.unknown()
    whole_type = ir.BufferType([4, 8], pl.FP32, ir.MemorySpace.Vec)
    window_type = ir.BufferType([2, 8], pl.FP32, ir.MemorySpace.Vec, valid_shape=[valid_rows, 8])
    whole = ir.Var("whole", whole_type, span)
    window = ir.Var("window", window_type, span)
    source = ir.Var("source", ir.TensorType([4, 8], pl.FP32), span)
    output = ir.Var("output", source.type, span)

    def dims(*values):
        return ir.MakeTuple([ir.ConstInt(value, pl.INDEX, span) for value in values], span)

    def call(name, args, type_=None):
        return ir.Call(ir.get_op(name), args, {}, type_ or ir.VoidType(), span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(
                whole, call("buffer.alloc", [dims(), ir.ConstInt(address, pl.INDEX, span)], whole_type), span
            ),
            ir.AssignStmt(
                window,
                call("buffer.alloc", [dims(), ir.ConstInt(address + 32, pl.INDEX, span)], window_type),
                span,
            ),
            ir.EvalStmt(call("buffer.load", [source, dims(0, 0), dims(4, 8), whole]), span),
            ir.EvalStmt(call("buffer.add", [window, window, window]), span),
            ir.EvalStmt(call("buffer.store", [whole, dims(0, 0), dims(4, 8), output]), span),
        ],
        span,
    )
    function = ir.Function("kernel", [source, output], [], body, span, ir_stage=ir.FunctionIRStage.Buffer)
    namespace = {}
    exec(torch_codegen(function), namespace)  # noqa: S102
    data = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    result = torch.empty_like(data)
    namespace["kernel"](data, result)
    expected = data.clone()
    expected[1 : 1 + valid_rows] *= 2
    torch.testing.assert_close(result, expected)


@jit.incore
def _overlapping_window(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    whole = pl.load(source, [0, 0], [32, 64])
    window = pl.tile.slice(whole, [32, 32], [0, 0])
    placed = pl.tile.assemble(whole, window, [0, 16])
    return pl.store(placed, [0, 0], out)


@jit
def window_debug(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _overlapping_window(source, out)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_public_compiled_window_snapshot_replays_buffer_dump(tmp_path, monkeypatch, planner):
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))
    source = torch.arange(2048, dtype=torch.float32).reshape(32, 64)
    expected = source.clone()
    expected[:, 16:48] = source[:, :32]
    config = RunConfig(platform="a2a3sim", save_kernels_dir=str(tmp_path / "compiled"), dump_passes=True)
    with passes.PassContext([], memory_planner=planner):
        compiled = window_debug.compile(source, torch.empty_like(source), config=config)
    dump = next((compiled.output_dir / "passes_dump").glob("*LowerTileToBuffer.msgpack"))
    program = ir.deserialize(dump.read_bytes())
    assert isinstance(program, ir.Program)
    assert any(function.ir_stage == ir.FunctionIRStage.Buffer for function in program.functions.values())
    compiled.validate_ir({"source": source, "out": torch.empty_like(source)}, {"out": expected})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
