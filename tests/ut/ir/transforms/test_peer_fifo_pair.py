# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Optional joint diagnostics for programmer-authored FIFO protocols."""

from textwrap import indent

import pypto.language as pl
import pytest
from pypto import passes, testing

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _pair(
    depth=2,
    *,
    vector_depth=None,
    import_name="c2v",
    sends=2,
    manual=False,
    group_calls=None,
    symbolic=False,
    same_bounds=True,
    collective=None,
    inactive_lane=False,
):
    vector_depth = depth if vector_depth is None else vector_depth
    push = "\n".join(["pl.tpush_to_aiv(result, split=0)"] * sends)
    barrier_stmt = "pl.system.syncall(core_type=pl.KernelType.MIX)"
    cube_body = (
        push
        + """
for i in pl.range(2):
    received: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
    pl.tfree_to_aiv(received)"""
    )
    vector_body = """for i in pl.range(2):
    pl.tpush_to_aic(value, split=0)
for j in pl.range(2):
    received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
    pl.tfree_to_aic(received)"""
    if inactive_lane:
        empty = "padding = pl.load(x, [0, 0], [16, 16], valid_shape=[0, 0], target_memory=pl.Mem.Vec)"
        shadow = empty + "\n" + vector_body.replace("tpush_to_aic(value", "tpush_to_aic(padding")
        vector_body = (
            "if pl.tile.get_subblock_idx() == 0:\n"
            + indent(vector_body, "    ")
            + "\nelse:\n"
            + indent(shadow, "    ")
        )
    if collective == "loop":
        cube_body += "\n" + barrier_stmt
        vector_body += "\n" + barrier_stmt
    if symbolic:
        cube_body = "for outer in pl.range(n):\n" + indent(cube_body, "    ")
        bound = "n" if same_bounds else "m"
        vector_body = f"for outer in pl.range({bound}):\n" + indent(vector_body, "    ")
    if manual:
        vector_body = "pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)\n" + vector_body
    if collective == "mismatched":
        cube_body += "\n" + barrier_stmt
        vector_body = barrier_stmt + "\n" + vector_body
    params = ", n: pl.Scalar[pl.INDEX], m: pl.Scalar[pl.INDEX]" if symbolic else ""
    args = "x, n, m" if symbolic else "x"
    calls = f"self.cube({args})\nself.vector({args})" if group_calls is None else group_calls
    return pl.parse_program(f"""
@pl.program
class PeerExample:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[16, 16], pl.FP16]{params}):
        incoming = pl.reserve_buffer(name="v2c", size={depth * 1024}, base=0)
        outgoing = pl.import_peer_buffer(name="{import_name}", peer_func="vector")
        pl.aic_initialize_pipe(outgoing, incoming, dir_mask=3, slot_size=1024, slot_num={depth})
        mat = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        left = pl.move(mat, target_memory=pl.Mem.Left)
        right = pl.move(mat, target_memory=pl.Mem.Right)
        result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.matmul(left, right)
{indent(cube_body, "        ")}

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[16, 16], pl.FP16]{params}):
        incoming = pl.reserve_buffer(name="c2v", size={vector_depth * 1024}, base=0)
        outgoing = pl.import_peer_buffer(name="v2c", peer_func="cube")
        pl.aiv_initialize_pipe(incoming, outgoing, dir_mask=3, slot_size=1024, slot_num={vector_depth})
        value = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
{indent(vector_body, "        ")}

    @pl.function(type=pl.FunctionType.Opaque)
    def extra(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP16]:
        return x

    @pl.function(type=pl.FunctionType.Group)
    def mixed(self, x: pl.Tensor[[16, 16], pl.FP16]{params}):
{indent(calls, "        ")}
""")


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, "FIFO ready/credit progress proved"),
        ({"group_calls": "self.cube(x)\nself.extra(x)\nself.vector(x)"}, "one unconditional call"),
        ({"group_calls": "self.cube(self.extra(x))\nself.vector(x)"}, "one unconditional call"),
        ({"depth": 1}, "FIFO ready/credit or rendezvous cycle"),
        ({"vector_depth": 1}, "transport depth or slot stride differs"),
        ({"import_name": "other"}, "import does not reference the receiving endpoint"),
        ({"sends": 1}, "stream counts, loop domains or message types differ"),
        ({"manual": True}, "authored event constraints"),
        ({"symbolic": True}, "FIFO ready/credit progress proved"),
        ({"symbolic": True, "same_bounds": False}, "stream counts, loop domains or message types differ"),
        ({"symbolic": True, "collective": "loop"}, "MIX rendezvous in a symbolic loop"),
        ({"collective": "mismatched"}, "FIFO ready/credit or rendezvous cycle"),
        (
            {"group_calls": "self.cube(x)\nself.vector(x)\nself.vector(x)"},
            "one unconditional call to each peer",
        ),
    ],
)
def test_authored_protocol_diagnostic(options, expected):
    assert expected in testing.inspect_peer_fifo_protocols(_pair(**options))["mixed"]


@pytest.mark.parametrize(
    "inactive_lane,expected",
    [(True, "single nonempty sender"), (False, "single-writer evidence unavailable")],
)
def test_managed_workspace_and_shared_slot_writers(inactive_lane, expected):
    program = passes.inject_gm_pipe_buffer()(passes.convert_to_ssa()(_pair(inactive_lane=inactive_lane)))
    report = testing.inspect_peer_fifo_protocols(program)["mixed"]
    assert "physical FIFO workspace matched" in report
    assert "FIFO ready/credit progress proved" in report
    assert expected in report
