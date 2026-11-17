# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent active-byte and command checks for runtime-row conversion."""

import pypto.language as pl
import pytest
import torch
from pypto import codegen, ir, passes, testing
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen

from .sync_closed_command_oracle import opname
from .test_runtime_valid_broadcast import check_runtime_commands, lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


def cast_program(source="BF16", target="FP32", visits=3, replay=2):
    return pl.parse_program(f"""
@pl.program
class RuntimeCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8,32], pl.{source}],
               rows: pl.Scalar[pl.INDEX], choice: pl.Scalar[pl.INDEX],
               out: pl.InOut[pl.Tensor[[8,32], pl.{target}]]) -> pl.Tensor[[8,32], pl.{target}]:
        for outer in pl.range({replay}):
            for visit in pl.range({visits}):
                if visit % 2 == choice:
                    source = pl.load(x, [0,0], [8,32], valid_shape=[rows,32])
                    result = pl.cast(source, target_type=pl.{target})
                    out = pl.store(result, [0,0], out)
        return out
""")


@pytest.mark.parametrize("source,target", [("BF16", "FP32"), ("FP32", "BF16")])
@pytest.mark.parametrize("visits,replay", [(1, 1), (3, 2)])
def test_runtime_cast_actual_bytes_order_matching_and_reuse(source, target, visits, replay, tmp_path):
    program, original = lower(cast_program(source, target, visits, replay))
    summary = testing.analyze_conservative_sync(original)
    assert not summary["exact_storage_selectors"]
    declared = {}
    for site in summary["sites"]:
        declared.setdefault(site["operation"].op.name, set()).update(
            (effect["space"], effect["write"]) for effect in site["effects"]
        )
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    source_type, target_type = [
        torch.bfloat16 if name == "BF16" else torch.float32 for name in (source, target)
    ]
    x = (torch.arange(256, dtype=torch.float32).reshape(8, 32) / 31).to(source_type)
    with passes.PassContext([], autosync_policy="conservative_prefix"):
        logical_program = passes.insert_sync()(program)
    with passes.PassContext([]):
        physical_program = passes.lower_sync()(logical_program)
    logical, physical = [
        candidate.get_function("kernel") for candidate in (logical_program, physical_program)
    ]
    assert logical is not None and physical is not None
    for rows in [0, 1, 7, 8]:
        for choice in [0, 1]:
            out = torch.full((8, 32), -99, dtype=target_type)
            namespace["kernel"](x, rows, choice, out)
            expected = torch.full_like(out, -99)
            if visits > choice:
                expected[:rows] = x[:rows].to(target_type)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            before, after = check_runtime_commands(logical, physical, {1: rows, 2: choice})
            for call, _, _, reads, writes in before.calls:
                for points, write in [(reads, False), (writes, True)]:
                    for memory, _ in points:
                        space = "Vec" if memory == ir.MemorySpace.Vec else "DDR"
                        assert (space, write) in declared[call.op.name]
            if not rows:
                assert not any(payload[0][0] == opname("buffer.cast") for payload in after.payloads)
    if find_ptoas_binary() is not None:
        pto, cpp = tmp_path / "cast.pto", tmp_path / "cast.cpp"
        pto.write_text(codegen.PTOCodegen().generate(physical_program))
        _run_ptoas(str(pto), str(cpp), ["--pto-level=level3", "--pto-arch=a2"])
        assert cpp.is_file()


def test_runtime_cast_retained_state_does_not_enumerate_visits():
    summaries = []
    for count in [3, 1023, 1000000]:
        _, function = lower(cast_program(visits=count))
        summaries.append(testing.analyze_conservative_sync(function))
    assert all(summary["cost"] == summaries[0]["cost"] for summary in summaries)
    assert all(len(summary["sites"]) == len(summaries[0]["sites"]) for summary in summaries)
