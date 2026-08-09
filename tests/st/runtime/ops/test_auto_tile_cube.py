# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device coverage for whole-function Ascend910B cube AutoTile.

The cases close the plan-to-emitter contracts that ordinary tensor-level
matmul tests cannot see: backward-clamped spatial regions, streamed outer K
windows, FirstPartialThenAtomic split-K, a retained GM-to-L1 panel, and serial
multi-matmul DAGs whose FP16/BF16 intermediates never materialize in GM.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_CUBE_TOL = RunConfig(rtol=1e-4, atol=1e-4)
_LONG_K_TOL = RunConfig(rtol=2e-3, atol=2e-3)
_BF16_CHAIN_TOL = RunConfig(rtol=2e-2, atol=2e-2)


class _AutoTileCubeCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)


@pl.program
class RaggedSpatialProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        lhs: pl.Tensor[[130, 64], pl.FP32],
        rhs: pl.Tensor[[64, 260], pl.FP32],
    ) -> pl.Tensor[[130, 260], pl.FP32]:
        return pl.matmul(lhs, rhs)


class RaggedSpatialCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_ragged_spatial"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [130, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("rhs", [64, 260], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [130, 260], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RaggedSpatialProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["lhs"] @ tensors["rhs"]


@pl.program
class OuterKStreamProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        lhs: pl.Tensor[[128, 736], pl.FP32],
        rhs: pl.Tensor[[736, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return pl.matmul(lhs, rhs)


class OuterKStreamCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_outer_k_stream"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [128, 736], DataType.FP32, init_value=torch.randn),
            TensorSpec("rhs", [736, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return OuterKStreamProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["lhs"] @ tensors["rhs"]


@pl.program
class SplitKProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        lhs: pl.Tensor[[16, 16384], pl.FP32],
        rhs: pl.Tensor[[16384, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.matmul(lhs, rhs)


class SplitKCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_first_partial_then_atomic"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [16, 16384], DataType.FP32, init_value=torch.randn),
            TensorSpec("rhs", [16384, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [16, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SplitKProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["lhs"] @ tensors["rhs"]


@pl.program
class RetainedPanelProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        lhs: pl.Tensor[[1024, 256], pl.FP32],
        rhs: pl.Tensor[[256, 2048], pl.FP32],
    ) -> pl.Tensor[[1024, 2048], pl.FP32]:
        return pl.matmul(lhs, rhs)


class RetainedPanelCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_retained_panel"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [1024, 256], DataType.FP32, init_value=torch.randn),
            TensorSpec("rhs", [256, 2048], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1024, 2048], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RetainedPanelProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["lhs"] @ tensors["rhs"]


@pl.program
class SerialChainProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        lhs: pl.Tensor[[128, 256], pl.BF16],
        middle: pl.Tensor[[256, 128], pl.BF16],
        rhs: pl.Tensor[[128, 256], pl.BF16],
    ) -> pl.Tensor[[128, 256], pl.BF16]:
        intermediate: pl.Tensor[[128, 128], pl.BF16] = pl.matmul(lhs, middle)
        return pl.matmul(intermediate, rhs)


class SerialChainCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_serial_chain"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [128, 256], DataType.BF16, init_value=torch.randn),
            TensorSpec("middle", [256, 128], DataType.BF16, init_value=torch.randn),
            TensorSpec("rhs", [128, 256], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [128, 256], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return SerialChainProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        intermediate = (tensors["lhs"].float() @ tensors["middle"].float()).to(torch.bfloat16)
        tensors["out"][:] = (intermediate.float() @ tensors["rhs"].float()).to(torch.bfloat16)


@pl.program
class ProducedTreeProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        a: pl.Tensor[[32, 48], pl.BF16],
        b: pl.Tensor[[48, 80], pl.BF16],
        c: pl.Tensor[[80, 64], pl.BF16],
        d: pl.Tensor[[64, 96], pl.BF16],
    ) -> pl.Tensor[[32, 96], pl.FP32]:
        lhs: pl.Tensor[[32, 80], pl.BF16] = pl.matmul(a, b)
        rhs: pl.Tensor[[80, 96], pl.BF16] = pl.matmul(c, d)
        return pl.matmul(lhs, rhs, out_dtype=pl.FP32)


class ProducedTreeCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_produced_tree"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 48], DataType.BF16, init_value=torch.randn),
            TensorSpec("b", [48, 80], DataType.BF16, init_value=torch.randn),
            TensorSpec("c", [80, 64], DataType.BF16, init_value=torch.randn),
            TensorSpec("d", [64, 96], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [32, 96], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ProducedTreeProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        lhs = (tensors["a"].float() @ tensors["b"].float()).to(torch.bfloat16)
        rhs = (tensors["c"].float() @ tensors["d"].float()).to(torch.bfloat16)
        tensors["out"][:] = lhs.float() @ rhs.float()


@pl.program
class ProducedMultiRoleProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        lhs: pl.Tensor[[16, 16], pl.BF16],
        rhs: pl.Tensor[[16, 16], pl.BF16],
    ) -> pl.Tensor[[16, 16], pl.BF16]:
        shared: pl.Tensor[[16, 16], pl.BF16] = pl.matmul(lhs, rhs)
        return pl.matmul(shared, shared)


class ProducedMultiRoleCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_produced_multi_role"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [16, 16], DataType.BF16, init_value=torch.randn),
            TensorSpec("rhs", [16, 16], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [16, 16], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ProducedMultiRoleProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        shared = (tensors["lhs"].float() @ tensors["rhs"].float()).to(torch.bfloat16)
        tensors["out"][:] = (shared.float() @ shared.float()).to(torch.bfloat16)


@pl.program
class MultiRoleBoundaryProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        shared: pl.Tensor[[32, 48], pl.BF16],
        lhs_rhs: pl.Tensor[[48, 64], pl.BF16],
        rhs_lhs: pl.Tensor[[64, 32], pl.BF16],
    ) -> pl.Tensor[[32, 48], pl.BF16]:
        lhs: pl.Tensor[[32, 64], pl.BF16] = pl.matmul(shared, lhs_rhs)
        rhs: pl.Tensor[[64, 48], pl.BF16] = pl.matmul(rhs_lhs, shared)
        return pl.matmul(lhs, rhs)


class MultiRoleBoundaryCase(_AutoTileCubeCase):
    def get_name(self) -> str:
        return "auto_tile_cube_multi_role_boundary"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("shared", [32, 48], DataType.BF16, init_value=torch.randn),
            TensorSpec("lhs_rhs", [48, 64], DataType.BF16, init_value=torch.randn),
            TensorSpec("rhs_lhs", [64, 32], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [32, 48], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiRoleBoundaryProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        lhs = (tensors["shared"].float() @ tensors["lhs_rhs"].float()).to(torch.bfloat16)
        rhs = (tensors["rhs_lhs"].float() @ tensors["shared"].float()).to(torch.bfloat16)
        tensors["out"][:] = (lhs.float() @ rhs.float()).to(torch.bfloat16)


class TestAutoTileCube:
    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        ("case_type", "config"),
        [
            (RaggedSpatialCase, _CUBE_TOL),
            (OuterKStreamCase, _CUBE_TOL),
            (SplitKCase, _LONG_K_TOL),
            (RetainedPanelCase, _CUBE_TOL),
            (SerialChainCase, _BF16_CHAIN_TOL),
            (ProducedTreeCase, _BF16_CHAIN_TOL),
            (ProducedMultiRoleCase, _BF16_CHAIN_TOL),
            (MultiRoleBoundaryCase, _BF16_CHAIN_TOL),
        ],
    )
    @pytest.mark.platforms("a2a3")
    def test_auto_tile_cube(self, test_runner, platform, case_type, config):
        result = test_runner.run(case_type(platform=platform, config=config))
        assert result.passed, f"{case_type.__name__} failed: {result.error}"
