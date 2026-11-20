# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate actual descriptor state and bounds in placed peer endpoints."""

import json
import re

import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime import RunConfig

from tests.ut.codegen.test_pto_codegen_cross_core import _paired_receive_program


@pytest.mark.ptoas_compile
@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    "width_base, variant, reason",
    [
        pytest.param(8, "", None, id="bounded"),
        pytest.param(16, "", "no positive in-capacity bound", id="exceeds-capacity"),
        pytest.param(0, "", "no positive in-capacity bound", id="can-be-empty"),
        pytest.param(8, "stale", None, id="restored-old-alias"),
        pytest.param(8, "alias-after", None, id="restored-full-alias"),
    ],
)
def test_runtime_valid_vec_effects(tmp_path, width_base, variant, reason):
    payload = "result = pl.exp(trim)"
    if variant == "stale":
        payload = """out = pl.store(pl.exp(copied), [0, 0], out)
    result = pl.exp(trim)"""
    elif variant == "alias-after":
        # Restoring the alias and using the original name exercise two SSA
        # versions of the same Buffer allocation's mutable native descriptor.
        payload = """alias = trim
    restored = pl.tile.set_validshape(alias, 16, 16)
    wide = pl.exp(copied)
    result = pl.add(wide, restored)"""
    program = _paired_receive_program(
        2,
        """values = pl.load(x, [0, 0], [16, 16])
for i in pl.range(2):
    produced = pl.matmul(values, values)
    pl.tpush_to_aiv(produced, split=0)""",
        f"""for i in pl.range(2):
    incoming: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
    copied = pl.add(incoming, incoming)
    pl.tfree_to_aic(incoming)
    trim = pl.tile.set_validshape(copied, 16, {width_base} + i)
    {payload}
    out = pl.store(result, [0, 0], out)
return out""",
    )
    config = RunConfig(arch=BackendType.Ascend910, codegen_only=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=True,
        enable_pypto_autosync=True,
        autosync_policy="conservative_prefix",
        autosync_disjoint_gm=True,
    ):
        ir.compile(program, **config.compile_kwargs())
    manifest = json.loads((tmp_path / "ptoas" / "paired.sync.json").read_text())
    if reason:
        assert manifest["ptoas_insert_sync"], manifest
        report = json.loads((tmp_path / "report" / "autosync.json").read_text())
        assert any(
            reason in kernel["reason"] for kernel in report["kernels"] if kernel["function"] == "vector"
        ), report
    else:
        assert manifest["owned"] and not manifest["ptoas_insert_sync"], manifest
        source = (tmp_path / "ptoas" / "paired.pto").read_text()
        vector = source.split("func.func @vector(", 1)[1]
        dimensions = dict(re.findall(r"(%\w+) = arith\.constant (\d+) : index", vector))
        additions = re.findall(r"(%\w+) = arith\.addi (%\w+), (%\w+) : index", source)
        dynamic_columns = {name for name, _, rhs in additions if dimensions.get(rhs) == "8"}

        def extent(name):
            return "dynamic" if name in dynamic_columns else dimensions[name]

        def restored(tile, cut):
            setters = re.findall(rf"pto\.set_validshape {tile}, (%\w+), (%\w+) :", vector[:cut])
            return tuple(extent(x) for x in setters[-1])

        exp = list(re.finditer(r"pto\.texp ins\((%\w+) : [^\n]+ outs\((%\w+) :", vector))
        expected = [("16", "dynamic")] if not variant else [("16", "16")]
        if variant == "stale":
            expected.append(("16", "dynamic"))
        assert [restored(call[1], call.start()) for call in exp] == expected
        assert [restored(call[2], call.start()) for call in exp] == expected
        partitions = re.findall(r"pto\.partition_view [^\n]+ sizes = \[(%\w+), (%\w+)\]", vector)
        assert [tuple(extent(x) for x in region) for region in partitions] == expected
        stores = list(re.finditer(r"pto\.tstore ins\((%\w+) :", vector))
        assert [restored(call[1], call.start()) for call in stores] == expected
        for producer, store in zip(exp, stores, strict=True):
            assert 'pto.wait_flag["PIPE_V", "PIPE_MTE3"' in vector[producer.end() : store.start()]
        assert {"cube", "vector"} == {domain["function"] for domain in manifest["domains"] if domain["owned"]}
