# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualify original cacheable scalar reads only from dispatch-wide read-only GM."""

import json
import re

import pytest
from pypto import ir, passes
from pypto import language as pl
from pypto._function_attrs import AUTOSYNC_FALLBACK_REASON_ATTR, PYPTO_AUTOSYNC_ATTR
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager
from pypto.runtime import RunConfig

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _program(
    *,
    rank_two: bool = False,
    position_type: str = "INT32",
    coordinate: str = "position",
    writer: bool = False,
    alias: str = "",
    columns: str = "16",
) -> ir.Program:
    source = f"pl.Tensor[[16, {columns}], pl.FP32]" if rank_two else "pl.Tensor[[length], pl.INT32]"
    coordinates = f"[{coordinate}, 0]" if rank_two else f"[{coordinate}]"
    update = (
        "\n        tile = pl.load(out, [0, 0], [16, 16], target_memory=pl.Mem.Vec)"
        "\n        pl.store(tile, [0, 0], out)"
        if writer
        else ""
    )
    vector_actual = "table" if alias == "group" else "out"
    group_actual = "table" if alias == "root" else "out"
    return pl.parse_program(f"""
length = pl.dynamic("length")

@pl.program
class PeerScalarRead:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, table: {source}, rhs: pl.Tensor[[32, 16], pl.FP16],
             position: pl.Scalar[pl.{position_type}]):
        fifo = pl.reserve_buffer(name="inbox", size=2048, base=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size=1024, slot_num=2)
        address = pl.tensor.read(table, {coordinates})
        row = pl.cast(pl.cast(address, pl.INT32), pl.INDEX)
        matrix = pl.load(rhs, [row, 0], [16, 16], target_memory=pl.Mem.Mat)
        right = pl.move(matrix, target_memory=pl.Mem.Right)
        received: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
        left = pl.move(received, target_memory=pl.Mem.Left)
        pl.tfree_to_aiv(received)
        product = pl.matmul(left, right)

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[16, 16], pl.FP16], out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]):
        peer = pl.import_peer_buffer(name="inbox", peer_func="cube")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=1024, slot_num=2)
        data = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        pl.tpush_to_aic(data, split=0){update}

    @pl.function(type=pl.FunctionType.Group)
    def paired(self, table: {source}, rhs: pl.Tensor[[32, 16], pl.FP16],
               x: pl.Tensor[[16, 16], pl.FP16], out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
               position: pl.Scalar[pl.{position_type}]):
        self.cube(table, rhs, position)
        self.vector(x, {vector_actual})

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, table: {source}, rhs: pl.Tensor[[32, 16], pl.FP16],
             x: pl.Tensor[[16, 16], pl.FP16], out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
             position: pl.Scalar[pl.{position_type}]):
        self.paired(table, rhs, x, {group_actual}, position)
""")


def _context(disjoint: bool) -> passes.PassContext:
    return passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=disjoint,
    )


@pytest.mark.parametrize(
    "options,disjoint,refusal",
    [
        pytest.param({"writer": True}, False, "may alias a GM write", id="may-alias-writer"),
        pytest.param(
            {"writer": True, "rank_two": True, "alias": "group"},
            True,
            "aliases a GM write",
            id="same-group-actual",
        ),
        pytest.param(
            {"writer": True, "rank_two": True, "alias": "root"},
            True,
            "aliases a GM write",
            id="same-root-actual",
        ),
        pytest.param({"position_type": "INDEX"}, False, "scalar GM address", id="unbounded-byte-offset"),
        pytest.param({"rank_two": True, "coordinate": "16"}, False, "scalar GM address", id="out-of-bounds"),
        pytest.param(
            {"rank_two": True, "columns": "length"}, False, "static inner extent", id="dynamic-pitch"
        ),
    ],
)
def test_original_peer_scalar_load_delegates_without_qualification(options, disjoint, refusal):
    with _context(disjoint):
        lowered = PassManager(ir.OptimizationStrategy.Default).run_passes(_program(**options))
    for name in ("cube", "vector"):
        endpoint = lowered.get_function(name)
        assert endpoint is not None
        assert not endpoint.attrs[PYPTO_AUTOSYNC_ATTR], endpoint.attrs
        assert refusal in endpoint.attrs[AUTOSYNC_FALLBACK_REASON_ATTR]


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "options,disjoint",
    [
        pytest.param({}, False, id="read-only-dispatch"),
        pytest.param({"writer": True}, True, id="rank-one-read-with-disjoint-writer"),
        pytest.param({"writer": True, "rank_two": True}, True, id="rank-two-read-with-disjoint-writer"),
    ],
)
def test_owned_peer_scalar_read_emits_original_native_load(tmp_path, options, disjoint):
    config = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with _context(disjoint):
        ir.compile(_program(**options), **config.compile_kwargs())

    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    assert manifest["owned"] and not manifest["ptoas_insert_sync"], manifest
    assert {domain["function"] for domain in manifest["domains"] if domain["owned"]} == {"cube", "vector"}
    pto_text = (tmp_path / "ptoas" / "paired.pto").read_text()
    assert pto_text.count("pto.load ") == 1
    assert "pto.cmo.cacheinvalid" not in pto_text

    native = (tmp_path / "kernels" / "aic" / "cube.cpp").read_text()
    cube = re.search(r"\bvoid cube\(__gm__ (?:int32_t|float)\*\s*(\w+),", native)
    assert cube is not None, native
    body = native[cube.end() :].split("\n}", 1)[0]
    assert re.search(rf"= \(?{re.escape(cube[1])}\)?\[[^\]\n]+\];", body), body
    assert "dcci(" not in body and "PTOAS__DCCI" not in body
