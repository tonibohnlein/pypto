# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Row recipes through production autosync, under declared scratch enclosures."""

import importlib.util
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import PassManager
from pypto.runtime import RunConfig

from .sync_matmul_oracle import interpret, validate
from .sync_physical_oracle import run_physical
from .test_autosync_switch import apply_hybrid
from .test_physical_sync_realization import check

pytestmark = pytest.mark.usefixtures("ascend_backend")


def row_fixture(count=None, visits=1, scratch_io=False, scratch_alias=False):
    """Small softmax structure with shared scratch, repeated visits and resets.

    Optional DMA consumers make scratch ownership observable outside V. The
    enclosing workspace is the declared model, not an exact native footprint.
    """
    span = ir.Span.unknown()
    fp, index = ir.DataType.FP32, ir.DataType.INDEX

    def ci(x):
        return ir.ConstInt(x, index, span)

    def tup(xs):
        return ir.MakeTuple([ci(x) for x in xs], span)

    def call(name, args, ty=None):
        return ir.Call(ir.get_op(name), args, ty or ir.VoidType(), span)

    def payload(name, args):
        return ir.EvalStmt(call(name, args), span)

    definitions = []

    def alloc(name, address, column=False):
        ty = ir.BufferType(
            [8, 1] if column else [8, 8],
            fp,
            ir.MemorySpace.Vec,
            blayout=ir.TileLayout.col_major if column else ir.TileLayout.row_major,
        )
        var = ir.Var(name, ty, span)
        definitions.append(ir.AssignStmt(var, call("buffer.alloc", [tup([]), ci(address)], ty), span))
        return var

    a, out, workspace_out = [
        ir.Var(n, ir.TensorType([8, 8], fp), span) for n in ["a", "out", "workspace_out"]
    ]
    x = alloc("x", 0)
    scratch = alloc("scratch", 0 if scratch_alias else 512)
    column = alloc("column", 1024, column=True)
    z = alloc("z", 1536)
    body = list(definitions)
    for visit in range(visits):
        statements = [payload("buffer.load", [a, tup([0, 0]), tup([8, 8]), x])]
        if scratch_io:
            statements.append(payload("buffer.load", [a, tup([0, 0]), tup([8, 8]), scratch]))
        statements.extend(
            [
                payload("buffer.row_max", [x, scratch, column]),
                payload("buffer.row_expand_sub", [x, column, z]),
                payload("buffer.exp", [z, z]),
                payload("buffer.row_sum", [z, scratch, column]),
                payload("buffer.row_expand_div", [z, column, z]),
                payload("buffer.store", [z, tup([0, 0]), tup([8, 8]), out]),
            ]
        )
        if scratch_io:
            statements.append(payload("buffer.store", [scratch, tup([0, 0]), tup([8, 8]), workspace_out]))
        if count is None:
            body.extend(statements)
        else:
            iv = ir.Var(f"i{visit}", ir.ScalarType(index), span)
            body.append(
                ir.ForStmt(
                    iv,
                    ci(0),
                    ci(count),
                    ci(1),
                    [],
                    ir.SeqStmts([*statements, ir.YieldStmt([], span)], span),
                    [],
                    span,
                )
            )
    body.append(ir.ReturnStmt([out, workspace_out], span))
    return ir.Function(
        "row_pipeline",
        [a, (out, ir.ParamDirection.InOut), (workspace_out, ir.ParamDirection.InOut)],
        [out.type, workspace_out.type],
        ir.SeqStmts(body, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("count", [None, 0, 1, 3])
@pytest.mark.parametrize("scratch_io", [False, True])
def test_row_access_covers_selectors_and_commands(count, scratch_io):
    original = row_fixture(count, visits=2, scratch_io=scratch_io)
    retained = testing.analyze_placed_sync(original, True)
    # Expands original effects independently of the compiler's storage atoms,
    # and checks actual retained covers, four event queries and selectors.
    validate(original, retained)
    physical, _, _, _ = check(original, list(range(6)))
    if count != 0:
        assert set(physical["native_reserved_ids"]) == {
            (ir.PipeType.V, ir.PipeType.S, 0),
            (ir.PipeType.S, ir.PipeType.V, 0),
        }
    emitted = apply_hybrid([original])[original.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR] == (count != 0)
    run_physical(original, emitted, range(6))


def test_scratch_alias_refuses_without_mutating_input():
    original = row_fixture(scratch_alias=True)
    with pytest.raises(ValueError, match="requires disjoint placed"):
        testing.analyze_placed_sync(original, True)
    emitted = apply_hybrid([original])[original.name]
    assert not dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    ir.assert_structural_equal(original.body, emitted.body)


def test_repeated_row_pipeline_is_compact():
    sizes = []
    for count in [4, 16, 1024, 1000000]:
        retained = testing.analyze_placed_sync(row_fixture(count, visits=2), True)
        d = retained.describe()
        physical = retained.realize_physical("a2a3", list(range(6)))
        sizes.append(
            (
                d["cells"],
                d["analyzed_occurrences"],
                tuple(len(r["demands"]) for r in d["regions"]),
                physical["set_sites"],
                physical["wait_sites"],
                physical["barrier_sites"],
            )
        )
    assert all(size == sizes[0] for size in sizes)


def load_example(file, name):
    path = Path(__file__).resolve().parents[4] / "examples/intermediate" / file
    spec = importlib.util.spec_from_file_location("row_sync_" + name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def softmax_original_and_emitted():
    kernel = load_example("02_softmax.py", "tile_softmax")
    args = [torch.empty(64, 64), torch.empty(64, 64)]
    config = RunConfig(arch=BackendType.Ascend910)
    functions = []
    for enabled in [False, True]:
        with passes.PassContext(
            [],
            enable_buffer_ir=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_pypto_autosync=enabled,
            autosync_disjoint_gm=True,
        ):
            program = kernel.lower(*args, config=config)
        functions.append(next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIV))
    return functions


def test_original_softmax_selects_pypto_and_preserves_payloads():
    original, emitted = softmax_original_and_emitted()
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    validate(original, testing.analyze_placed_sync(original, True))
    run_physical(original, emitted, range(6))
    # Payload data/control are unchanged; modeled scratch writes remain visible.
    assert len(interpret(original)[1]) == 7


@pytest.mark.parametrize("original_example", [False, True])
def test_row_cpp_generation_and_ptoas_delegation(original_example, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    if original_example:
        original, emitted = softmax_original_and_emitted()
    else:
        original = row_fixture(3, visits=2, scratch_io=True)
        emitted = apply_hybrid([original])[original.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    source = tmp_path / "row.pto"
    source.write_text(
        codegen.PTOCodegen().generate(
            ir.Program([emitted], "row", original.span),
            emit_tile_addr=True,
            emit_source_loc=False,
        )
    )
    outputs = []
    for enabled in [False, True]:
        target = tmp_path / f"row-{enabled}.cpp"
        _run_ptoas(
            str(source),
            str(target),
            [
                "--pto-level=level3",
                "--pto-arch=a2",
                f"--enable-insert-sync={'true' if enabled else 'false'}",
            ],
        )
        outputs.append(target.read_text())
    assert outputs[0] == outputs[1]
    assert "TROWSUM" in outputs[0] and "TROWMAX" in outputs[0]


@pytest.mark.parametrize("name,inputs", [("rms_norm", 3), ("layer_norm", 4)])
def test_normalization_frontend_boundary_remains_explicit(name, inputs):
    kernel = load_example("03_normalization.py", name)
    args = [torch.empty(32, 64), *[torch.empty(1, 64) for _ in range(inputs - 2)], torch.empty(32, 64)]
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=True,
        autosync_disjoint_gm=True,
    ):
        program = kernel.lower(*args, config=RunConfig(arch=BackendType.Ascend910))
    function = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert not dict(function.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert "physically placed Buffer IR" in dict(function.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]


@pytest.mark.parametrize("op", ["row_sum", "row_max"])
def test_explicit_col_major_preserves_default_autosync_plan(op):
    """Spelling the default layout must neither delegate nor change physical synchronization.

    Consume the column locally in the supported broadcast recipe; a direct
    column-vector GM store has a separate, unavailable Buffer transfer contract.
    """
    emitted = []
    for layout in ["", ", out_layout=pl.TileLayout.col_major"]:
        program = pl.parse_program(f"""
@pl.program
class ColumnReduction:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[8, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[8, 64], pl.FP32]]) -> pl.Tensor[[8, 64], pl.FP32]:
        x = pl.load(a, [0, 0], [8, 64])
        scratch = pl.create_tile([8, 64], dtype=pl.FP32)
        reduced = pl.tile.{op}(x, scratch{layout})
        result = pl.tile.row_expand_sub(x, reduced)
        return pl.store(result, [0, 0], out)
""")
        functions = []
        for enabled in [False, True]:
            with passes.PassContext([], enable_pypto_autosync=enabled, autosync_disjoint_gm=True):
                lowered = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
            function = lowered.get_function("kernel")
            assert function is not None
            functions.append(function)
        original, synchronized = functions
        assert dict(synchronized.attrs)[PYPTO_AUTOSYNC_ATTR], dict(synchronized.attrs)
        retained = testing.analyze_placed_sync(original, True)
        validate(original, retained)
        run_physical(original, synchronized, range(6))
        emitted.append(synchronized)
    ir.assert_structural_equal(emitted[0], emitted[1])


def test_padded_row_major_reduction_remains_delegated():
    program = pl.parse_program("""
@pl.program
class RowMajor:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[4, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        x = pl.load(a, [0, 0], [4, 64])
        scratch = pl.create_tile([4, 64], dtype=pl.FP32)
        reduced = pl.tile.row_sum(x, scratch, out_layout=pl.TileLayout.row_major)
        return pl.store(reduced, [0, 0], out)
""")
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        lowered = PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(program)
    function = lowered.get_function("kernel")
    assert function is not None
    assert not dict(function.attrs)[PYPTO_AUTOSYNC_ATTR]
    assert "physically placed Buffer IR" in dict(function.attrs)[AUTOSYNC_FALLBACK_REASON_ATTR]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
