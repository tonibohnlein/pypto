# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared conversion effects reach actual token and flag lowering."""

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes, testing
from pypto.backend.pto_backend import _run_ptoas

from .sync_matmul_oracle import validate
from .sync_physical_oracle import run_physical
from .test_select_buffer_ir_functions import _kernel, _lower

pytestmark = pytest.mark.usefixtures("ascend_backend")


def cast_fixture(dtype="BF16", mode="round", valid=(8, 64), source_dtype="FP32"):
    program = pl.parse_program(f"""
@pl.program
class CastSync:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8, 64], pl.{source_dtype}],
               out: pl.Out[pl.Tensor[[8, 64], pl.{dtype}]]) -> pl.Tensor[[8, 64], pl.{dtype}]:
        value = pl.load(x, [0, 0], [8, 64], valid_shape=[{valid[0]}, {valid[1]}])
        result = pl.cast(value, pl.{dtype}, mode="{mode}")
        return pl.store(result, [0, 0], out)
""")
    return _lower(program, passes.MemoryPlanner.PYPTO)


@pytest.mark.parametrize(
    "source_dtype,dtype,mode",
    [
        ("FP32", "FP16", "rint"),
        ("FP32", "BF16", "rint"),
        ("FP32", "BF16", "round"),
        ("BF16", "FP32", "round"),
    ],
)
def test_shared_cast_effects_tokens_and_flags(source_dtype, dtype, mode, tmp_path):
    original = cast_fixture(dtype, mode, source_dtype=source_dtype)
    function = _kernel(original)
    retained = testing.analyze_placed_sync(function, True)
    validate(function, retained)
    with passes.PassContext([], enable_pypto_autosync=True, autosync_disjoint_gm=True):
        logical = passes.insert_sync()(original)
        restored = ir.deserialize(ir.serialize(logical))
        physical = passes.lower_sync()(passes.optimize_sync()(restored))
    run_physical(function, _kernel(physical), range(6))
    source, output = tmp_path / "cast.pto", tmp_path / "cast.cpp"
    source.write_text(codegen.PTOCodegen().generate(physical))
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level3"])
    assert "set_flag(" in output.read_text()


def test_partial_cast_footprint_stays_unavailable():
    function = _kernel(cast_fixture(valid=(7, 61)))
    with pytest.raises(ValueError, match="fully valid|full-valid footprint"):
        testing.analyze_placed_sync(function, True)
