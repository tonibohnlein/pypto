# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Indexed local accesses: selected-model covers and concrete requirement coverage."""

import pytest
from pypto import codegen, ir, passes, testing
from pypto._function_attrs import PYPTO_AUTOSYNC_ATTR
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_oracle import interpret, validate
from .sync_physical_oracle import run_physical
from .test_autosync_switch import apply_hybrid
from .test_buffer_indexed_lowering import _gather_program, _kernel, _lower
from .test_finite_sync_analysis import reference
from .test_physical_sync_realization import check

pytestmark = pytest.mark.usefixtures("ascend_backend")


def indexed_fixture(scatter=False, count=None, visits=1, alias=False, partial=False):
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

    def alloc(name, address, dtype=fp):
        ty = ir.BufferType([2, 16], dtype, ir.MemorySpace.Vec, valid_shape=[1, 16] if partial else [2, 16])
        var = ir.Var(name, ty, span)
        definitions.append(ir.AssignStmt(var, call("buffer.alloc", [tup([]), ci(address)], ty), span))
        return var

    a, out = [ir.Var(n, ir.TensorType([2, 16], fp), span) for n in ["a", "out"]]
    idx = ir.Var("idx", ir.TensorType([2, 16], ir.DataType.INT32), span)
    x, indices = alloc("x", 0), alloc("indices", 256, ir.DataType.INT32)
    scratch = alloc("scratch", 512, ir.DataType.INT32)
    z = alloc("z", 0 if alias else 768)
    body = list(definitions)
    for visit in range(visits):
        statements = [
            payload("buffer.load", [a, tup([0, 0]), tup([2, 16]), x]),
            payload("buffer.load", [idx, tup([0, 0]), tup([2, 16]), indices]),
        ]
        statements.append(
            payload(
                "buffer.scatter" if scatter else "buffer.gather",
                [x, indices, z] if scatter else [x, indices, scratch, z],
            )
        )
        statements.append(payload("buffer.store", [z, tup([0, 0]), tup([2, 16]), out]))
        # The next visit overwrites the table, exercising the enclosed-read WAR.
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
    body.append(ir.ReturnStmt([out], span))
    return ir.Function(
        "indexed_pipeline",
        [a, idx, (out, ir.ParamDirection.InOut)],
        [out.type],
        ir.SeqStmts(body, span),
        span,
        ir.FunctionType.AIV,
        ir_stage=ir.FunctionIRStage.Buffer,
    )


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("count", [None, 0, 1, 3])
def test_indexed_models_and_actual_commands(scatter, count):
    original = indexed_fixture(scatter, count, visits=2)
    retained = testing.analyze_placed_sync(original, True)
    validate(original, retained)
    physical, _, _, _ = check(original, list(range(6)))
    emitted = apply_hybrid([original])[original.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR] == (count != 0)
    _, _, projected = run_physical(original, emitted, range(6))
    if count != 0:
        assert set(physical["native_reserved_ids"]) == {
            (ir.PipeType.V, ir.PipeType.S, 0),
            (ir.PipeType.S, ir.PipeType.V, 0),
        }
    # Concrete indexed effects, independent of compiler partitions/generators.
    # Scratch may touch any subset of its declared window; test both extremes.
    _, calls = interpret(original)
    for offsets in [list(range(32)), [0] * 32, [31] * 32, [(i * 13) % 32 for i in range(32)]]:
        for scratch_written in [False, True]:
            effects = []
            for call, _, pipe, reads, writes in calls:
                if call.op.name == ir.get_op("buffer.gather").name:
                    table = {(ir.MemorySpace.Vec, 4 * k + b) for k in offsets for b in range(4)}
                    index_reads = {(ir.MemorySpace.Vec, k) for k in range(256, 384)}
                    assert table | index_reads <= reads
                    reads = table | index_reads
                    if not scratch_written:
                        writes = {p for p in writes if not 512 <= p[1] < 640}
                effects.append((pipe, True, reads, writes))
            actual = reference(effects)
            assert actual[2] <= projected
    sites = [s for r in retained.describe()["regions"] for s in r["sites"]]
    indexed = [
        s
        for s in sites
        if s["original"].op.name == ir.get_op("buffer.scatter" if scatter else "buffer.gather").name
    ]
    assert all(("enclosure" in s["access_precision"]) != scatter for s in indexed)


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("kind", ["alias", "partial"])
def test_indexed_unsupported_contracts_delegate(scatter, kind):
    original = indexed_fixture(scatter, **{kind: True})
    with pytest.raises(ValueError):
        testing.analyze_placed_sync(original, True)
    emitted = apply_hybrid([original])[original.name]
    assert not dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    ir.assert_structural_equal(original.body, emitted.body)


@pytest.mark.parametrize("scatter", [False, True])
def test_indexed_trip_count_does_not_expand(scatter):
    sizes = []
    for count in [4, 16, 1024, 1000000]:
        retained = testing.analyze_placed_sync(indexed_fixture(scatter, count, visits=2), True)
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


@pytest.mark.parametrize("scatter", [False, True])
def test_indexed_cpp_with_ptoas_enabled(scatter, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("Pinned PTOAS unavailable")
    original = indexed_fixture(scatter, 3, visits=2)
    emitted = apply_hybrid([original])[original.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    program = ir.Program([emitted], "indexed", ir.Span.unknown())
    program = ir.deserialize(ir.serialize(program))
    assert isinstance(program, ir.Program)
    text = codegen.PTOCodegen().generate(program, emit_tile_addr=True, emit_source_loc=False)
    source, output = tmp_path / "indexed.pto", tmp_path / "indexed.cpp"
    source.write_text(text)
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level3", "--enable-insert-sync"])
    assert ("TSCATTER" if scatter else "TGATHER") in output.read_text()


@pytest.mark.parametrize("dtype", ["FP16", "FP32", "INT32"])
def test_default_buffer_gather_feeds_autosync(dtype):
    original = _kernel(_lower(_gather_program(dtype), passes.MemoryPlanner.PYPTO))
    emitted = apply_hybrid([original])[original.name]
    assert dict(emitted.attrs)[PYPTO_AUTOSYNC_ATTR]
    validate(original, testing.analyze_placed_sync(original, True))
    run_physical(original, emitted, range(6))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
