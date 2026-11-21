# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Retain A3 programmer-authored GM protocol at its native cuts."""

import pytest
from pypto import codegen, ir, passes
from pypto import language as pl
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import testing

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _program(read_index="position", workspace_use="pl.system.set_ffts(workspace)"):
    return pl.parse_program(f"""
columns = pl.dynamic("columns")

@pl.program
class AuthoredGM:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, workspace: pl.Tensor[[256], pl.INT64],
               table: pl.Tensor[[4, columns], pl.INT32], src: pl.Tensor[[1, 8], pl.INT32],
               out: pl.Out[pl.Tensor[[1, 8], pl.INT32]],
               position: pl.Scalar[pl.INDEX]) -> pl.Tensor[[1, 8], pl.INT32]:
        {workspace_use}
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        value = pl.load(src, [0, 0], [1, 8], target_memory=pl.Mem.Vec)
        place = pl.tensor.read(table, [1, {read_index}])
        pl.tile.write(value, [0, 0], place)
        result = pl.store(value, [0, 0], out)
        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
        return result
""")


def _lower(program, *, autosync=False):
    with passes.PassContext(
        [],
        enable_buffer_ir=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_pypto_autosync=autosync,
        autosync_policy="conservative_prefix",
    ):
        return PassManager(ir.OptimizationStrategy.Default).run_passes(program)


def test_authored_gm_protocol_and_dynamic_point_keep_native_order():
    lowered = _lower(_program(), autosync=True)
    leaf = lowered.get_function("kernel")
    assert leaf is not None
    assert leaf.ir_stage == ir.FunctionIRStage.Buffer
    # A raw A3 flag has no local set/wait contract or proof of a matching peer.
    assert not leaf.attrs["pypto_autosync"]
    native = codegen.PTOCodegen().generate(lowered)
    cursor = 0
    for step in (
        "pto.set_ffts",
        "pto.cmo.cacheinvalid",
        "pto.fence.barrier_all",
        "pto.syncall",
        "pto.cmo.cacheinvalid",
        "pto.sync.wait",
        "pto.tload",
        "pto.load ",
        "pto.tsetval",
        "pto.tstore",
        "pto.sync.set",
    ):
        cursor = native.index(step, cursor) + len(step)
    assert native.count("pto.set_ffts") == 1
    assert native.count("pto.sync.set") == 1
    assert native.count("pto.sync.wait") == 1


def test_i64_tensor_type_exception_does_not_admit_gm_scalar_payload():
    lowered = _lower(_program(workspace_use="pl.system.set_ffts(workspace)\n        pl.read(workspace, [0])"))
    leaf = lowered.get_function("kernel")
    assert leaf is not None
    assert leaf.ir_stage == ir.FunctionIRStage.Functional
    assert leaf.attrs["buffer_admission_property"] in {"type_layout", "operation_recipe"}


def _mixed_program(
    *,
    writer="fresh",
    publish_table=False,
    acquire="pl.system.cacheinvalid()",
    rendezvous="pl.system.syncall(core_type=pl.KernelType.MIX)",
):
    vec_actual = {"fresh": "vec_scratch", "different_formal": "vec_out", "same_formal": "table"}[writer]
    publication = (
        "early = pl.load(data, [0, 0], [1, 8], target_memory=pl.Mem.Vec)\n"
        "        pl.store(early, [0, 0], table)"
        if publish_table
        else "pass"
    )
    return pl.parse_program(f"""
@pl.program
class AuthoredMixedGM:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, workspace: pl.Tensor[[256], pl.INT64], x: pl.Tensor[[16, 16], pl.BF16],
             out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        pl.system.set_ffts(workspace)
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        lhs = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        rhs = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        a = pl.move(lhs, target_memory=pl.Mem.Left)
        b = pl.move(rhs, target_memory=pl.Mem.Right)
        acc = pl.matmul(a, b)
        result = pl.store(acc, [0, 0], out)
        pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
        return result

    @pl.function(type=pl.FunctionType.AIV)
    def vec(self, workspace: pl.Tensor[[256], pl.INT64], table: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            data: pl.Tensor[[1, 8], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 8], pl.INT32]]) -> pl.Tensor[[1, 8], pl.INT32]:
        pl.system.set_ffts(workspace)
        {publication}
        pl.system.cacheinvalid()
        pl.system.fence()
        {rendezvous}
        {acquire}
        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        item = pl.tensor.read(table, [0, 0])
        value = pl.load(data, [0, 0], [1, 8], target_memory=pl.Mem.Vec)
        pl.tile.write(value, [0, 0], item)
        return pl.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Group)
    def unit(self, workspace: pl.Tensor[[256], pl.INT64], x: pl.Tensor[[16, 16], pl.BF16],
             table: pl.InOut[pl.Tensor[[1, 8], pl.INT32]], data: pl.Tensor[[1, 8], pl.INT32],
             cube_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]], vec_out: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
        self.cube(workspace, x, cube_out)
        self.vec(workspace, table, data, vec_out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16, 16], pl.BF16], table: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
             data: pl.Tensor[[1, 8], pl.INT32], vec_out: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
        workspace = pl.create_tensor([256], dtype=pl.INT64)
        cube_scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
        vec_scratch = pl.create_tensor([1, 8], dtype=pl.INT32)
        self.unit(workspace, x, table, data, cube_scratch, {vec_actual})
""")


@pytest.mark.parametrize(
    "options,reason",
    [
        ({}, None),
        ({"publish_table": True}, None),
        ({"writer": "same_formal"}, "aliases a GM write"),
        ({"writer": "different_formal"}, "may alias a GM write"),
        ({"acquire": "pass"}, "publication seam"),
        ({"rendezvous": "pass"}, "publication seam"),
    ],
)
def test_authored_gm_pair_requires_original_publication_and_readonly_storage(options, reason):
    lowered = _lower(_mixed_program(**options), autosync=True)
    leaves: list[ir.Function] = []
    for name in ("cube", "vec"):
        leaf = lowered.get_function(name)
        assert leaf is not None, name
        leaves.append(leaf)
        assert leaf.ir_stage == ir.FunctionIRStage.Buffer, leaf.attrs
        assert leaf.attrs["pypto_autosync"] == (reason is None), leaf.attrs
        if reason is not None:
            assert reason in leaf.attrs["autosync_fallback_reason"]
    if reason is None:
        native = codegen.PTOCodegen().generate(ir.Program(leaves, "unit", ir.Span.unknown()))
        assert native.count("pto.set_ffts") == 2
        assert native.count("pto.fence.barrier_all") == 2
        assert native.count("pto.syncall") == 2
        assert native.count("pto.cmo.cacheinvalid") == 4
        assert native.count("pto.sync.set") == 1
        assert native.count("pto.sync.wait") == 1


@pytest.mark.parametrize("requested_extent,upper", [("pl.min(length, 16)", 16), ("signed_extent", 2**32 - 1)])
def test_host_tensor_dim_uses_only_representable_requested_slice_bounds(requested_extent, upper):
    program = pl.parse_program(f"""
length = pl.dynamic("length")

@pl.program
class HostViewExtent:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, extent: pl.Scalar[pl.INDEX]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, source: pl.Tensor[[length, 16], pl.FP32], signed_extent: pl.Scalar[pl.INDEX]):
        view = pl.tensor.slice(source, [{requested_extent}, 16], [0, 0])
        observed = pl.tensor.dim(view, 0)
        self.kernel(observed)
""")
    assert testing.sync_invocation_evidence(program)["parameters"]["kernel"][0] == (0, upper)
