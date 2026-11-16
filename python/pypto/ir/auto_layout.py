# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prototype infrastructure for layout instance extraction, writeback, and cost calibration.

This module implements the modeling and IR readout/writeback layer for
automatic layout selection in PyPTO, following the formalization in:

    Eisenhofer et al., "Tensor Seeks Layout: Formalizing Layout Selection for
    ML Compilers", arXiv:2608.21555.

Pipeline placement
------------------
To observe and select layouts for **both** global-memory tensors (`DDR`) and
on-chip L1/UB/L0 tiles (`Mat`, `Vec`, `Left`, `Right`, `Acc`), layout analysis
and writeback run **after** `AutoTileMatmulL0` (Pass 20) and
`InferTileMemorySpace` (Pass 22), and **before** `ResolveBackendOpLayouts`
(Pass 24):

1. `AutoTileMatmulL0` (Pass 20) is completely bypassed (a no-op) when matmul
   tile shapes fit directly in L0A / L0B / L0C (`(m, n, k) == (M, N, K)`).
2. `InferTileMemorySpace` (Pass 22) assigns concrete memory spaces (`Mat`,
   `Vec`, `Left`, `Right`, `Acc`) to all tile variables and materializes
   explicit `tile.move(Mat -> Left)` and `tile.move(Mat -> Right)` operations.
3. **Layout Readout & Writeback (this module)** extracts a
   `LayoutProblemInstance` covering GM tensors, L1/UB tiles, and L0 tiles,
   evaluates candidate `LayoutAssignment`s under `LayoutCostModel`, and writes
   a selected assignment back into the IR (running `BlockNzTensorViews` when a
   GM tensor is assigned `pl.NZ`).
4. Downstream passes (`ResolveBackendOpLayouts`, `MaterializeTensorStrides`,
   `InitMemRef`, `LowerTileToBuffer`, `PTOCodegen`) lower the rewritten IR to
   `.pto` MLIR.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pypto._target import ExecutionMode
from pypto.ir.compile import compile as ir_compile
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import DataType, passes
from pypto.pypto_core import backend as core_backend
from pypto.pypto_core import ir as core_ir

TensorLayout = core_ir.TensorLayout
TileLayout = core_ir.TileLayout
MemorySpace = core_ir.MemorySpace
TensorType = core_ir.TensorType
TensorView = core_ir.TensorView
TileType = core_ir.TileType
TileView = core_ir.TileView


def _dtype_byte_width(dtype: DataType) -> int:
    """Return the storage byte width for whole-byte data types."""
    bit_map = {
        DataType.BOOL: 1,
        DataType.INT8: 1,
        DataType.UINT8: 1,
        DataType.FP8E4M3FN: 1,
        DataType.FP8E5M2: 1,
        DataType.FP8E8M0: 1,
        DataType.INT16: 2,
        DataType.UINT16: 2,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.INT32: 4,
        DataType.UINT32: 4,
        DataType.FP32: 4,
        DataType.INT64: 8,
        DataType.UINT64: 8,
    }
    if dtype not in bit_map:
        raise ValueError(f"Unsupported or sub-byte dtype for NZ layout: {dtype}")
    return bit_map[dtype]


def c0_elements(dtype: DataType | torch.dtype) -> int:
    """Return the number of elements per 32-byte fractal C0 line (`32 // sizeof(dtype)`)."""
    if isinstance(dtype, torch.dtype):
        elem_bytes: int = int(torch.empty((), dtype=dtype).element_size())
    else:
        elem_bytes = _dtype_byte_width(dtype)
    if 32 % elem_bytes != 0:
        raise ValueError(f"32-byte C0 line is not divisible by element byte width {elem_bytes}")
    return 32 // elem_bytes


def is_nz_compatible_shape(shape: Sequence[int], dtype: DataType) -> bool:
    """Check whether a logical 2D `[R, C]` or 3D `[B, R, C]` shape can be blocked as `pl.NZ`."""
    if len(shape) not in (2, 3):
        return False
    rows, cols = shape[-2], shape[-1]
    if rows <= 0 or cols <= 0:
        return False
    try:
        c0 = c0_elements(dtype)
    except ValueError:
        return False
    return rows % 16 == 0 and cols % c0 == 0


def pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Pack a logical 2D `[R, C]` or 3D `[B, R, C]` tensor into Ascend NZ byte order.

    Permutes `[..., R, C]` into `[..., C // c0, R // 16, 16, c0]` contiguous
    storage and reshapes back to `logical.shape` so it can be bound to a
    `pl.Tensor[..., pl.NZ]` kernel parameter.
    """
    if logical.ndim not in (2, 3):
        raise ValueError(f"pack_nz expects a rank-2 or rank-3 tensor, got shape {tuple(logical.shape)}")
    rows: int = int(logical.shape[-2])
    cols: int = int(logical.shape[-1])
    c0 = c0_elements(logical.dtype)
    if rows % 16 != 0 or cols % c0 != 0:
        raise ValueError(
            f"pack_nz requires rows % 16 == 0 and cols % {c0} == 0, got shape {tuple(logical.shape)}"
        )
    lead: tuple[int, ...] = tuple(int(d) for d in logical.shape[:-2])
    reshaped = logical.reshape(*lead, rows // 16, 16, cols // c0, c0)
    if logical.ndim == 2:
        permuted = reshaped.permute(2, 0, 1, 3)
    else:
        permuted = reshaped.permute(0, 3, 1, 2, 4)
    return permuted.contiguous().reshape(logical.shape)


def unpack_nz(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nz`: convert an NZ-packed tensor back to logical ND order."""
    if packed.ndim not in (2, 3):
        raise ValueError(f"unpack_nz expects a rank-2 or rank-3 tensor, got shape {tuple(packed.shape)}")
    rows: int = int(packed.shape[-2])
    cols: int = int(packed.shape[-1])
    c0 = c0_elements(packed.dtype)
    if rows % 16 != 0 or cols % c0 != 0:
        raise ValueError(
            f"unpack_nz requires rows % 16 == 0 and cols % {c0} == 0, got shape {tuple(packed.shape)}"
        )
    lead: tuple[int, ...] = tuple(int(d) for d in packed.shape[:-2])
    reshaped = packed.reshape(*lead, cols // c0, rows // 16, 16, c0)
    if packed.ndim == 2:
        permuted = reshaped.permute(1, 2, 0, 3)
    else:
        permuted = reshaped.permute(0, 2, 3, 1, 4)
    return permuted.contiguous().reshape(packed.shape)


@dataclass(frozen=True)
class TileLayoutDesc:
    """On-chip tile layout descriptor `(blayout, slayout, fractal)`.

    Common Ascend on-chip layouts:
    - `ROW_MAJOR_FLAT`: `(row_major, none_box, 512)` — UB (`Vec`) row-major
    - `COL_MAJOR_FLAT`: `(col_major, none_box, 512)` — UB (`Vec`) col-major
    - `FRACTAL_NZ`: `(col_major, row_major, 512)` — L1 (`Mat`) / L0A (`Left`) NZ
    - `FRACTAL_ZN`: `(row_major, col_major, 512)` — L1 (`Mat`) transposed / L0B (`Right`) ZN
    - `FRACTAL_ACC_NZ`: `(col_major, row_major, 1024)` — L0C (`Acc`) 16x16 accumulator NZ
    """

    blayout: core_ir.TileLayout
    slayout: core_ir.TileLayout
    fractal: int = 512

    def short_name(self) -> str:
        """Return a concise human-readable layout label."""
        if self.slayout == core_ir.TileLayout.none_box:
            return f"flat_{self.blayout.name}"
        if self.fractal == 1024 and self.blayout == core_ir.TileLayout.col_major:
            return "acc_nz_1024"
        if (
            self.fractal == 512
            and self.blayout == core_ir.TileLayout.col_major
            and self.slayout == core_ir.TileLayout.row_major
        ):
            return "fractal_nz"
        if (
            self.fractal == 512
            and self.blayout == core_ir.TileLayout.row_major
            and self.slayout == core_ir.TileLayout.col_major
        ):
            return "fractal_zn"
        return f"{self.blayout.name}/{self.slayout.name}/{self.fractal}"


ROW_MAJOR_FLAT = TileLayoutDesc(core_ir.TileLayout.row_major, core_ir.TileLayout.none_box, 512)
COL_MAJOR_FLAT = TileLayoutDesc(core_ir.TileLayout.col_major, core_ir.TileLayout.none_box, 512)
FRACTAL_NZ = TileLayoutDesc(core_ir.TileLayout.col_major, core_ir.TileLayout.row_major, 512)
FRACTAL_ZN = TileLayoutDesc(core_ir.TileLayout.row_major, core_ir.TileLayout.col_major, 512)
FRACTAL_ACC_NZ = TileLayoutDesc(core_ir.TileLayout.col_major, core_ir.TileLayout.row_major, 1024)


LayoutChoice = core_ir.TensorLayout | TileLayoutDesc


@dataclass(frozen=True)
class LayoutValueNode:
    """A tensor (GM) or tile (L1/UB/L0) value in a `LayoutProblemInstance`.

    Attributes:
        name: SSA variable or parameter name inside the InCore function.
        shape: Static integer dimensions `(M, K)`, `(K, N)`, etc.
        dtype: Element `DataType`.
        memory_space: `DDR` for GM `TensorType` parameters, or `Mat`, `Vec`,
            `Left`, `Right`, `Acc` for on-chip `TileType` variables.
        current_layout: Layout currently stamped in the IR (`TensorLayout` for
            GM tensors, `TileLayoutDesc` for on-chip tiles).
        admissible_layouts: Candidate layouts that the solver/evaluator may
            assign to this value.
        producer_op_id: ID of the operation that defines this value (`None` for
            function input parameters).
        is_reusable_weight: Whether a GM conversion/packing of this tensor can
            be amortized across `expected_uses` invocations.
    """

    name: str
    shape: tuple[int, ...]
    dtype: DataType
    memory_space: core_ir.MemorySpace
    current_layout: LayoutChoice
    admissible_layouts: tuple[LayoutChoice, ...]
    producer_op_id: str | None = None
    is_reusable_weight: bool = False
    is_external_output: bool = False

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_size(self) -> int:
        return self.num_elements * _dtype_byte_width(self.dtype)


@dataclass(frozen=True)
class OperatorConfig:
    """One legal input/output layout configuration $q_o \\in Q_o$ for an IR operation.

    Attributes:
        op_id: Identifier of the operation in `LayoutProblemInstance`.
        op_name: PyPTO op name (`tile.load`, `tile.move`, `tile.matmul`, `tile.store`, ...).
        input_layouts: Required layout for each input value in `inputs`.
        output_layout: Produced layout for `output`.
        op_cost_cycles: Estimated execution cost $C^{\\text{op}}_o(q_o)$ in cycles.
        notes: Short description of the hardware path (e.g. `GM->L1 NZ2NZ`,
            `L1->L0A MTE1`, `L0 Cube MAD`).
    """

    op_id: str
    op_name: str
    input_layouts: tuple[LayoutChoice, ...]
    output_layout: LayoutChoice
    op_cost_cycles: float
    notes: str = ""


@dataclass(frozen=True)
class LayoutOperationNode:
    """An operation node in `LayoutProblemInstance` extracted after `InferTileMemorySpace`.

    Attributes:
        op_id: Unique operation identifier in program order.
        op_name: Registered PyPTO operator name (`tile.load`, `tile.move`,
            `tile.transpose_view`, `tile.matmul`, `tile.store`).
        inputs: Input value names (referencing `LayoutProblemInstance.values`).
        output: Output value name (referencing `LayoutProblemInstance.values`).
        invocations: Number of invocations $R_o$ per kernel run.
        configs: Candidate `OperatorConfig`s for this operation.
    """

    op_id: str
    op_name: str
    inputs: tuple[str, ...]
    output: str
    invocations: int = 1
    configs: tuple[OperatorConfig, ...] = ()


@dataclass
class LayoutCostModel:
    """Calibratable analytical cost model for GM, L1 (`Mat`), UB (`Vec`), and L0 (`Left`/`Right`/`Acc`) ops.

    All bandwidths and overheads are exposed as dataclass fields so they can be
    calibrated directly against hardware or simulator traces without touching
    any L0 auto-tiler internals.

    Attributes:
        expected_uses: Number of kernel runs $R$ over which reusable GM weight
            packing is amortized.
        mte2_nz2nz_bytes_per_cycle: Effective GM $\\to$ L1 bandwidth for
            contiguous `NZ2NZ` fractal transfers.
        mte2_nd2nz_bytes_per_cycle: Effective GM $\\to$ L1 bandwidth for
            on-the-fly `ND2NZ` scatter-gather transfers.
        mte2_nd2nz_row_penalty_cycles: Additional per-row stride penalty when
            `ND2NZ` gathers non-contiguous 32B C0 segments.
        mte2_vec_bytes_per_cycle: Effective GM $\\to$ UB (`Vec`) bandwidth.
        mte1_l1_to_l0_bytes_per_cycle: Effective L1 (`Mat`) $\\to$ L0A/L0B
            (`Left`/`Right`) MTE1 transfer bandwidth.
        mte1_transpose_penalty_cycles: Extra MTE1 overhead when moving a tile
            with a layout conversion vs. a direct fractal copy.
        cube_macs_per_cycle: Effective Cube MAC throughput ($16 \\times 16 \\times 16 = 4096$
            ops/cycle peak on Ascend 910B/910C).
        cube_issue_overhead_cycles: Fixed per-matmul issue overhead in cycles.
        fixpipe_acc_to_nd_bytes_per_cycle: Effective L0C (`Acc`) $\\to$ GM `ND`
            store bandwidth via FIXPIPE.
        fixpipe_acc_to_nz_bytes_per_cycle: Effective L0C (`Acc`) $\\to$ GM `NZ`
            store bandwidth via FIXPIPE (`TStoreAccNz2nz`).
        gm_pack_bytes_per_cycle: Effective bandwidth for offline/host or AIV
            GM `ND <-> NZ` relayout when conversion is required.
        gm_pack_fixed_cycles: Fixed launch/setup cost for a GM layout conversion.
        onchip_relayout_bytes_per_cycle: Effective on-chip `tile.move` layout
            coercion bandwidth when producer and consumer tile layouts differ.
    """

    expected_uses: int = 1
    backend_type: core_backend.BackendType | None = None
    mte2_nz2nz_bytes_per_cycle: float = 192.0
    mte2_nd2nz_bytes_per_cycle: float = 96.0
    mte2_nd2nz_row_penalty_cycles: float = 0.5
    mte2_vec_bytes_per_cycle: float = 192.0
    mte1_l1_to_l0_bytes_per_cycle: float = 512.0
    mte1_transpose_penalty_cycles: float = 8.0
    cube_macs_per_cycle: float = 4096.0
    cube_issue_overhead_cycles: float = 28.0
    vec_elements_per_cycle: float = 128.0
    fixpipe_acc_to_nd_bytes_per_cycle: float = 128.0
    fixpipe_acc_to_nz_bytes_per_cycle: float = 192.0
    gm_pack_bytes_per_cycle: float = 64.0
    gm_pack_fixed_cycles: float = 120.0
    onchip_relayout_bytes_per_cycle: float = 256.0
    allow_external_output_relayout: bool = False

    @property
    def resolved_backend_type(self) -> core_backend.BackendType:
        """Return the configured or explicit `BackendType`."""
        if self.backend_type is not None:
            return self.backend_type
        if core_backend.is_backend_configured():
            return core_backend.get_backend_type()
        return core_backend.BackendType.Ascend910

    def target_arch(self) -> str:
        """Return the PTO target architecture string (`'a2a3'` or `'a5'`)."""
        inst = core_backend.get_backend_instance(self.resolved_backend_type)
        return inst.get_handler().get_pto_target_arch()

    def supports_acc_nz_store(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
    ) -> bool:
        """Check whether the active backend supports direct `Acc -> NZ` GM store (`TStoreAccNz2nz`).

        Per `BlockNzTensorViews`, direct `Acc -> NZ` stores require A2/A3
        (`Ascend910`, `'a2a3'`), FP32 `Acc` and FP32 GM destination, and both
        rows and cols divisible by 16 (and rank 2/3 NZ compatibility).
        """
        if self.target_arch() == "a5":
            return False
        if len(dst_node.shape) not in (2, 3):
            return False
        rows, cols = dst_node.shape[-2], dst_node.shape[-1]
        return (
            src_node.memory_space == core_ir.MemorySpace.Acc
            and src_node.dtype == DataType.FP32
            and dst_node.dtype == DataType.FP32
            and rows % 16 == 0
            and cols % 16 == 0
            and is_nz_compatible_shape(dst_node.shape, dst_node.dtype)
        )

    def is_valid_load_config(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> bool:
        """Check whether `tile.load` with `(src_layout, dst_layout)` is backend-legal."""
        if dst_node.memory_space == core_ir.MemorySpace.Mat:
            return src_layout in (core_ir.TensorLayout.ND, core_ir.TensorLayout.NZ) and (
                dst_layout == dst_node.current_layout
            )
        # Vec/other loads cannot read NZ tensors directly (`BlockNzTensorViews` requires Mat).
        return src_layout == core_ir.TensorLayout.ND and dst_layout == dst_node.current_layout

    def is_valid_move_config(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> bool:
        """Check whether `tile.move(src -> dst)` with `(src_layout, dst_layout)` is backend-legal."""
        if not isinstance(src_layout, TileLayoutDesc) or not isinstance(dst_layout, TileLayoutDesc):
            return False
        # MTE1 Mat -> Left requires FRACTAL_NZ; Mat -> Right requires FRACTAL_ZN.
        if (
            src_node.memory_space == core_ir.MemorySpace.Mat
            and dst_node.memory_space == core_ir.MemorySpace.Left
        ):
            return src_layout == FRACTAL_NZ and dst_layout == FRACTAL_NZ
        if (
            src_node.memory_space == core_ir.MemorySpace.Mat
            and dst_node.memory_space == core_ir.MemorySpace.Right
        ):
            return src_layout == FRACTAL_ZN and dst_layout == FRACTAL_ZN
        return True

    def required_input_tile_layout(self, op_name: str, input_index: int) -> core_ir.TileLayout | None:
        """Query the active C++ `Backend`'s `BackendTileLayoutSpec` for `op_name` input `input_index`.

        By filtering candidate `OperatorConfig`s against this constraint, our
        layout selector only emits configurations that already satisfy
        `ResolveBackendOpLayouts` (making Pass 24 a guaranteed no-op).
        """
        if core_backend.is_backend_configured():
            return core_backend.get_input_tile_layout(op_name, input_index)
        return None

    def estimate_load_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate `tile.load` cost from GM (`src_node`) to `Mat` or `Vec` (`dst_node`)."""
        total_bytes = float(dst_node.byte_size)
        if dst_node.memory_space == core_ir.MemorySpace.Mat:
            if src_layout == core_ir.TensorLayout.NZ:
                return total_bytes / self.mte2_nz2nz_bytes_per_cycle
            rows = dst_node.shape[-2] if len(dst_node.shape) >= 2 else 1
            cols = dst_node.shape[-1] if len(dst_node.shape) >= 1 else 1
            c0 = c0_elements(dst_node.dtype)
            col_blocks = max(1, (cols + c0 - 1) // c0)
            return (
                total_bytes / self.mte2_nd2nz_bytes_per_cycle
                + rows * col_blocks * self.mte2_nd2nz_row_penalty_cycles
            )
        return total_bytes / self.mte2_vec_bytes_per_cycle

    def estimate_move_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate on-chip `tile.move` cost (e.g. `Mat -> Left` or `Mat -> Right`)."""
        total_bytes = float(dst_node.byte_size)
        base = total_bytes / self.mte1_l1_to_l0_bytes_per_cycle
        if src_layout != dst_layout:
            return base + self.mte1_transpose_penalty_cycles
        return base

    def estimate_matmul_cost(
        self,
        lhs_node: LayoutValueNode,
        rhs_node: LayoutValueNode,
        out_node: LayoutValueNode,
    ) -> float:
        """Estimate L0 `tile.matmul` (`Left` x `Right` -> `Acc`) compute cost."""
        m_dim = int(lhs_node.shape[-2])
        k_dim = int(lhs_node.shape[-1])
        n_dim = int(rhs_node.shape[-1])
        macs = float(m_dim * n_dim * k_dim)
        return self.cube_issue_overhead_cycles + (macs / self.cube_macs_per_cycle)

    def estimate_store_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate `tile.store` (`Acc -> GM` or `Vec -> GM`) cost."""
        total_bytes = float(src_node.byte_size)
        if dst_layout == core_ir.TensorLayout.NZ:
            return total_bytes / self.fixpipe_acc_to_nz_bytes_per_cycle
        return total_bytes / self.fixpipe_acc_to_nd_bytes_per_cycle

    def estimate_conversion_cost(
        self,
        val_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate conversion cost $C^{\\text{tr}}_t(\\ell_1, \\ell_2)$ for a value."""
        if src_layout == dst_layout:
            return 0.0
        total_bytes = float(val_node.byte_size)
        if val_node.memory_space == core_ir.MemorySpace.DDR:
            return self.gm_pack_fixed_cycles + (total_bytes / self.gm_pack_bytes_per_cycle)
        return total_bytes / self.onchip_relayout_bytes_per_cycle


def _empty_op_configs() -> dict[str, OperatorConfig]:
    return {}


@dataclass(frozen=True)
class LayoutAssignment:
    """A candidate assignment of layouts to values and configurations to operations.

    Attributes:
        value_layouts: Mapping from value name (`x`, `w`, `xt`, `xt_Left`, ...)
            to its assigned `TensorLayout` (for GM tensors) or `TileLayoutDesc`
            (for on-chip L1/UB/L0 tiles).
        op_configs: Optional mapping from `op_id` to the chosen `OperatorConfig`.
    """

    value_layouts: Mapping[str, LayoutChoice]
    op_configs: Mapping[str, OperatorConfig] = field(default_factory=_empty_op_configs)

    @property
    def tensor_layouts(self) -> dict[str, core_ir.TensorLayout]:
        """Return only the GM `TensorLayout` choices keyed by parameter name."""
        return {
            name: layout
            for name, layout in self.value_layouts.items()
            if isinstance(layout, core_ir.TensorLayout)
        }

    @property
    def tile_layouts(self) -> dict[str, TileLayoutDesc]:
        """Return only the on-chip `TileLayoutDesc` choices keyed by tile variable name."""
        return {
            name: layout for name, layout in self.value_layouts.items() if isinstance(layout, TileLayoutDesc)
        }


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost evaluation of one `LayoutAssignment` on a `LayoutProblemInstance`.

    Follows the paper's objective (Section 2 of arXiv:2608.21555):
    operator cost plus conversion cost, where conversion to each distinct
    consumer layout $\\ell \\in \\text{cons}(t)$ is charged once.
    """

    assignment: LayoutAssignment
    total_cost_cycles: float
    operator_cost_cycles: float
    conversion_cost_cycles: float
    per_op_costs: dict[str, float]
    per_value_conversion_costs: dict[str, float]


def _empty_values() -> dict[str, LayoutValueNode]:
    return {}


def _empty_operations() -> list[LayoutOperationNode]:
    return []


@dataclass
class LayoutProblemInstance:
    """Solver & cost-calibration instance extracted from post-`InferTileMemorySpace` IR.

    Contains all GM tensors (`DDR`), L1/UB tiles (`Mat`, `Vec`), and L0 tiles
    (`Left`, `Right`, `Acc`), together with all data-movement and compute
    operations (`tile.load`, `tile.transpose_view`, `tile.move`, `tile.matmul`,
    `tile.store`) and their candidate `OperatorConfig`s.
    """

    values: dict[str, LayoutValueNode] = field(default_factory=_empty_values)
    operations: list[LayoutOperationNode] = field(default_factory=_empty_operations)
    cost_model: LayoutCostModel = field(default_factory=LayoutCostModel)


_TENSOR_CREATE_OP = core_ir.get_op("tensor.create").name
_TILE_CREATE_OP = core_ir.get_op("tile.create").name
_TILE_LOAD_OP = core_ir.get_op("tile.load").name
_TILE_STORE_OP = core_ir.get_op("tile.store").name
_TILE_MOVE_OP = core_ir.get_op("tile.move").name
_TILE_TRANSPOSE_VIEW_OP = core_ir.get_op("tile.transpose_view").name
_TILE_MATMUL_OPS = frozenset(
    {
        core_ir.get_op("tile.matmul").name,
        core_ir.get_op("tile.matmul_acc").name,
        core_ir.get_op("tile.matmul_bias").name,
    }
)


_SSA_SUFFIX_RE = re.compile(r"__ssa_v\d+")


def _clean_ssa_name(name_hint: str) -> str:
    """Strip compiler SSA suffixes (`__ssa_v0`) for human-readable value names."""
    return _SSA_SUFFIX_RE.sub("", name_hint)


def _static_shape(shape_exprs: Sequence[core_ir.Expr]) -> tuple[int, ...] | None:
    dims: list[int] = []
    for expr in shape_exprs:
        if not isinstance(expr, core_ir.ConstInt):
            return None
        dims.append(int(expr.value))
    return tuple(dims)


def _effective_tile_layout(tile_type: core_ir.TileType) -> TileLayoutDesc:
    """Return the effective `TileLayoutDesc` for `tile_type` after `InferTileMemorySpace`."""
    space = tile_type.memory_space
    view = tile_type.tile_view
    if view is not None:
        return TileLayoutDesc(view.blayout, view.slayout, int(view.fractal))
    if space in (core_ir.MemorySpace.Mat, core_ir.MemorySpace.Left):
        return FRACTAL_NZ
    if space == core_ir.MemorySpace.Right:
        return FRACTAL_ZN
    if space == core_ir.MemorySpace.Acc:
        return FRACTAL_ACC_NZ
    return ROW_MAJOR_FLAT


def _admissible_tile_layouts(tile_type: core_ir.TileType) -> tuple[TileLayoutDesc, ...]:
    """Return candidate `TileLayoutDesc`s for an on-chip tile by its `MemorySpace`."""
    space = tile_type.memory_space
    if space in (core_ir.MemorySpace.Mat, core_ir.MemorySpace.Left, core_ir.MemorySpace.Right):
        # L1 (`Mat`) and L0A/L0B (`Left`/`Right`) support both 512B fractal orientations:
        # FRACTAL_NZ (col_major/row_major/512) and FRACTAL_ZN (row_major/col_major/512).
        current = _effective_tile_layout(tile_type)
        other = FRACTAL_ZN if current == FRACTAL_NZ else FRACTAL_NZ
        return (current, other)
    if space == core_ir.MemorySpace.Acc:
        return (FRACTAL_ACC_NZ,)
    if space == core_ir.MemorySpace.Vec:
        return (ROW_MAJOR_FLAT, COL_MAJOR_FLAT)
    return (_effective_tile_layout(tile_type),)


def run_pre_layout_passes(program: core_ir.Program) -> core_ir.Program:
    """Run the default compilation pipeline up through `InferTileMemorySpace` (Pass 22).

    At the returned stage:
    - All tensor/tile shapes are 2D (`FlattenTileNdTo2D`).
    - When tensors fit in L0, `AutoTileMatmulL0` (Pass 20) is a complete no-op.
    - `InferTileMemorySpace` (Pass 22) has assigned `Mat`, `Vec`, `Left`, `Right`,
      and `Acc` memory spaces to every tile and inserted `tile.move` into L0A/L0B.
    """
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    pipe = passes.PassPipeline()
    for p in pm.passes:
        pipe.add_pass(p)
        if p.get_name() == "InferTileMemorySpace":
            break
    return pipe.run(program)


def run_post_layout_passes(program: core_ir.Program) -> core_ir.Program:
    """Run `BlockNzTensorViews` (for any newly-assigned `pl.NZ` tensors) and Passes 23..end."""
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    pipe = passes.PassPipeline()
    # Ensure any GM TensorType rewritten to `pl.NZ` during layout writeback is
    # blocked into its rank-5 `[1, C/c0, R/16, 16, c0]` view before
    # `MaterializeTensorStrides` runs downstream.
    pipe.add_pass(passes.block_nz_tensor_views())
    after_infer = False
    for p in pm.passes:
        if p.get_name() == "InferTileMemorySpace":
            after_infer = True
            continue
        if after_infer:
            pipe.add_pass(p)
    return pipe.run(program)


def _for_trip_count(op: core_ir.ForStmt) -> int:
    """Return constant trip count of *op* when static, otherwise 1."""
    if (
        isinstance(op.start, core_ir.ConstInt)
        and isinstance(op.stop, core_ir.ConstInt)
        and isinstance(op.step, core_ir.ConstInt)
        and int(op.step.value) > 0
    ):
        span = int(op.stop.value) - int(op.start.value)
        step = int(op.step.value)
        return max(1, (span + step - 1) // step)
    return 1


def _iter_child_stmts(stmt: core_ir.Stmt) -> Sequence[core_ir.Stmt]:
    """Return immediate child body statements for any compound IR statement."""
    if isinstance(stmt, core_ir.SeqStmts):
        return stmt.stmts
    if isinstance(stmt, core_ir.IfStmt):
        branches = [stmt.then_body]
        if stmt.else_body is not None:
            branches.append(stmt.else_body)
        return branches
    body = getattr(stmt, "body", None)
    return (body,) if isinstance(body, core_ir.Stmt) else ()


class _OrchCallMultiplierVisitor:
    """Count static invocation multiplicity and resolve caller GM aliases per InCore callee."""

    def __init__(self, func_by_name: dict[str, core_ir.Function] | None = None) -> None:
        self._func_by_name = func_by_name or {}
        self._mult = 1
        self._alias_by_uid: dict[int, str] = {}
        self.call_counts: dict[str, int] = {}
        self.call_arg_names: dict[str, list[str | None]] = {}

    def walk(self, func: core_ir.Function) -> None:
        for param in func.params:
            self._alias_by_uid[param.unique_id] = _clean_ssa_name(param.name_hint)
        self._walk_stmt(func.body)

    def _resolve_arg_name(self, expr: core_ir.Expr) -> str | None:
        if not isinstance(expr, core_ir.Var):
            return None
        return self._alias_by_uid.get(expr.unique_id, _clean_ssa_name(expr.name_hint))

    def _record_calls(self, expr: core_ir.Expr, assign_var: core_ir.Var | None = None) -> None:
        if isinstance(expr, core_ir.Call):
            if isinstance(expr.op, core_ir.GlobalVar):
                name = expr.op.name
                self.call_counts[name] = self.call_counts.get(name, 0) + self._mult
                resolved = [self._resolve_arg_name(arg) for arg in expr.args]
                self.call_arg_names.setdefault(name, resolved)
                if assign_var is not None:
                    callee = self._func_by_name.get(name)
                    if callee is not None:
                        for arg_name, direction in zip(resolved, callee.param_directions, strict=True):
                            if (
                                direction in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut)
                                and arg_name is not None
                            ):
                                self._alias_by_uid[assign_var.unique_id] = arg_name
                                break
            elif assign_var is not None and expr.op.name == _TENSOR_CREATE_OP:
                self._alias_by_uid[assign_var.unique_id] = _clean_ssa_name(assign_var.name_hint)
            for arg in expr.args:
                self._record_calls(arg)

    def _walk_stmt(self, stmt: core_ir.Stmt) -> None:
        if isinstance(stmt, core_ir.ForStmt):
            prev = self._mult
            self._mult = prev * _for_trip_count(stmt)
            try:
                self._walk_stmt(stmt.body)
            finally:
                self._mult = prev
            return
        if isinstance(stmt, core_ir.AssignStmt):
            self._record_calls(stmt.value, assign_var=stmt.var)
        elif isinstance(stmt, core_ir.EvalStmt):
            self._record_calls(stmt.expr)
        elif isinstance(stmt, core_ir.ReturnStmt):
            for val in stmt.value:
                self._record_calls(val)
        for child in _iter_child_stmts(stmt):
            self._walk_stmt(child)


class _PostInferInstanceExtractor:
    """Extract `LayoutValueNode`s and `LayoutOperationNode`s from post-Pass-22 IR."""

    def __init__(
        self,
        cost_model: LayoutCostModel,
        reusable_tensors: set[str],
        *,
        external_out_names: set[str] | None = None,
        shared_values: dict[str, LayoutValueNode] | None = None,
        op_offset: int = 0,
        tile_prefix: str = "",
        call_multiplier: int = 1,
        param_name_overrides: Sequence[str | None] | None = None,
    ) -> None:
        self._model = cost_model
        self._reusable = reusable_tensors
        self._external_out_names = external_out_names
        self._tile_prefix = tile_prefix
        self.values: dict[str, LayoutValueNode] = dict(shared_values) if shared_values else {}
        self.operations: list[LayoutOperationNode] = []
        self._uid_to_name: dict[int, str] = {}
        self._op_counter = op_offset
        self._loop_multiplier = max(1, call_multiplier)
        self._yield_stack: list[list[int]] = []
        self._param_overrides = list(param_name_overrides) if param_name_overrides is not None else []

    @property
    def op_counter(self) -> int:
        return self._op_counter

    def extract(self, func: core_ir.Function) -> None:
        for idx, (param, direction) in enumerate(zip(func.params, func.param_directions, strict=True)):
            if not isinstance(param.type, core_ir.TensorType):
                continue
            shape = _static_shape(param.type.shape)
            if shape is None:
                continue
            override = self._param_overrides[idx] if idx < len(self._param_overrides) else None
            name = override or _clean_ssa_name(param.name_hint)
            self._uid_to_name[param.unique_id] = name
            if name in self.values:
                continue
            view = param.type.tensor_view
            cur_layout = view.layout if view is not None else core_ir.TensorLayout.ND
            if self._external_out_names is not None:
                is_ext_out = name in self._external_out_names
            else:
                is_ext_out = direction in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut)
            admissible: tuple[LayoutChoice, ...]
            if is_nz_compatible_shape(shape, param.type.dtype):
                admissible = (core_ir.TensorLayout.ND, core_ir.TensorLayout.NZ)
            else:
                admissible = (cur_layout,)
            self.values[name] = LayoutValueNode(
                name=name,
                shape=shape,
                dtype=param.type.dtype,
                memory_space=core_ir.MemorySpace.DDR,
                current_layout=cur_layout,
                admissible_layouts=admissible,
                producer_op_id=None,
                is_reusable_weight=(name in self._reusable or param.name_hint in self._reusable),
                is_external_output=is_ext_out,
            )
        self._walk_stmt(func.body)

    def _register_tile_var(self, var: core_ir.Var, producer_op_id: str | None) -> LayoutValueNode | None:
        if not isinstance(var.type, core_ir.TileType):
            return None
        shape = _static_shape(var.type.shape)
        if shape is None:
            return None
        base_name = _clean_ssa_name(var.name_hint)
        name = f"{self._tile_prefix}{base_name}" if self._tile_prefix else base_name
        self._uid_to_name[var.unique_id] = name
        space = var.type.memory_space or core_ir.MemorySpace.Vec
        cur_layout = _effective_tile_layout(var.type)
        admissible = _admissible_tile_layouts(var.type)
        node = LayoutValueNode(
            name=name,
            shape=shape,
            dtype=var.type.dtype,
            memory_space=space,
            current_layout=cur_layout,
            admissible_layouts=admissible,
            producer_op_id=producer_op_id,
        )
        self.values[name] = node
        return node

    def _extract_load(self, op: core_ir.AssignStmt, rhs: core_ir.Call) -> None:
        if len(rhs.args) < 1:
            return
        src_arg = rhs.args[0]
        if not isinstance(src_arg, core_ir.Var) or src_arg.unique_id not in self._uid_to_name:
            return
        src_name = self._uid_to_name[src_arg.unique_id]
        op_id = f"op_{self._op_counter}_{op.var.name_hint}_load"
        self._op_counter += 1
        dst_node = self._register_tile_var(op.var, op_id)
        src_node = self.values.get(src_name)
        if src_node is None or dst_node is None:
            return
        configs: list[OperatorConfig] = []
        for s_lay in src_node.admissible_layouts:
            for d_lay in dst_node.admissible_layouts:
                if not self._model.is_valid_load_config(src_node, dst_node, s_lay, d_lay):
                    continue
                cost = self._model.estimate_load_cost(src_node, dst_node, s_lay, d_lay)
                kind = "NZ2NZ" if s_lay == core_ir.TensorLayout.NZ else "ND2NZ"
                src_label = s_lay.name if isinstance(s_lay, core_ir.TensorLayout) else str(s_lay)
                configs.append(
                    OperatorConfig(
                        op_id=op_id,
                        op_name=rhs.op.name,
                        input_layouts=(s_lay,),
                        output_layout=d_lay,
                        op_cost_cycles=cost,
                        notes=f"GM({src_label})->{dst_node.memory_space.name} ({kind})",
                    )
                )
        self.operations.append(
            LayoutOperationNode(
                op_id=op_id,
                op_name=rhs.op.name,
                inputs=(src_name,),
                output=dst_node.name,
                invocations=self._model.expected_uses * self._loop_multiplier,
                configs=tuple(configs),
            )
        )

    def _extract_transpose_view(self, op: core_ir.AssignStmt, rhs: core_ir.Call) -> None:
        if len(rhs.args) < 1:
            return
        src_arg = rhs.args[0]
        if not isinstance(src_arg, core_ir.Var) or src_arg.unique_id not in self._uid_to_name:
            return
        src_name = self._uid_to_name[src_arg.unique_id]
        op_id = f"op_{self._op_counter}_{op.var.name_hint}_tview"
        self._op_counter += 1
        dst_node = self._register_tile_var(op.var, op_id)
        src_node = self.values.get(src_name)
        if src_node is None or dst_node is None:
            return
        configs = (
            OperatorConfig(
                op_id=op_id,
                op_name=rhs.op.name,
                input_layouts=(FRACTAL_NZ,),
                output_layout=FRACTAL_ZN,
                op_cost_cycles=0.0,
                notes="Zero-copy metadata view (NZ -> ZN)",
            ),
            OperatorConfig(
                op_id=op_id,
                op_name=rhs.op.name,
                input_layouts=(FRACTAL_ZN,),
                output_layout=FRACTAL_NZ,
                op_cost_cycles=0.0,
                notes="Zero-copy metadata view (ZN -> NZ)",
            ),
        )
        self.operations.append(
            LayoutOperationNode(
                op_id=op_id,
                op_name=rhs.op.name,
                inputs=(src_name,),
                output=dst_node.name,
                invocations=self._model.expected_uses * self._loop_multiplier,
                configs=configs,
            )
        )

    def _extract_move(self, op: core_ir.AssignStmt, rhs: core_ir.Call) -> None:
        if len(rhs.args) < 1:
            return
        src_arg = rhs.args[0]
        if not isinstance(src_arg, core_ir.Var) or src_arg.unique_id not in self._uid_to_name:
            return
        src_name = self._uid_to_name[src_arg.unique_id]
        op_id = f"op_{self._op_counter}_{op.var.name_hint}_move"
        self._op_counter += 1
        dst_node = self._register_tile_var(op.var, op_id)
        src_node = self.values.get(src_name)
        if src_node is None or dst_node is None:
            return
        configs: list[OperatorConfig] = []
        for s_lay in src_node.admissible_layouts:
            for d_lay in dst_node.admissible_layouts:
                if not self._model.is_valid_move_config(src_node, dst_node, s_lay, d_lay):
                    continue
                cost = self._model.estimate_move_cost(src_node, dst_node, s_lay, d_lay)
                configs.append(
                    OperatorConfig(
                        op_id=op_id,
                        op_name=rhs.op.name,
                        input_layouts=(s_lay,),
                        output_layout=d_lay,
                        op_cost_cycles=cost,
                        notes=f"MTE1 {src_node.memory_space.name}->{dst_node.memory_space.name}",
                    )
                )
        self.operations.append(
            LayoutOperationNode(
                op_id=op_id,
                op_name=rhs.op.name,
                inputs=(src_name,),
                output=dst_node.name,
                invocations=self._model.expected_uses * self._loop_multiplier,
                configs=tuple(configs),
            )
        )

    def _extract_matmul(self, op: core_ir.AssignStmt, rhs: core_ir.Call) -> None:
        if len(rhs.args) < 2:
            return
        lhs_idx = 1 if rhs.op.name == core_ir.get_op("tile.matmul_acc").name else 0
        rhs_idx = lhs_idx + 1
        if len(rhs.args) <= rhs_idx:
            return
        lhs_arg = rhs.args[lhs_idx]
        rhs_arg = rhs.args[rhs_idx]
        if (
            not isinstance(lhs_arg, core_ir.Var)
            or not isinstance(rhs_arg, core_ir.Var)
            or lhs_arg.unique_id not in self._uid_to_name
            or rhs_arg.unique_id not in self._uid_to_name
        ):
            return
        lhs_name = self._uid_to_name[lhs_arg.unique_id]
        rhs_name = self._uid_to_name[rhs_arg.unique_id]
        op_id = f"op_{self._op_counter}_{op.var.name_hint}_matmul"
        self._op_counter += 1
        dst_node = self._register_tile_var(op.var, op_id)
        lhs_node = self.values.get(lhs_name)
        rhs_node = self.values.get(rhs_name)
        if lhs_node is None or rhs_node is None or dst_node is None:
            return
        cost = self._model.estimate_matmul_cost(lhs_node, rhs_node, dst_node)
        configs = (
            OperatorConfig(
                op_id=op_id,
                op_name=rhs.op.name,
                input_layouts=(FRACTAL_NZ, FRACTAL_ZN),
                output_layout=FRACTAL_ACC_NZ,
                op_cost_cycles=cost,
                notes="L0 Cube MAD (Left[NZ] x Right[ZN] -> Acc[NZ1024])",
            ),
        )
        self.operations.append(
            LayoutOperationNode(
                op_id=op_id,
                op_name=rhs.op.name,
                inputs=(lhs_name, rhs_name),
                output=dst_node.name,
                invocations=self._model.expected_uses * self._loop_multiplier,
                configs=configs,
            )
        )

    def _extract_store(self, rhs: core_ir.Call) -> None:
        if len(rhs.args) < 3:
            return
        tile_arg = rhs.args[0]
        dst_arg = rhs.args[2]
        if (
            not isinstance(tile_arg, core_ir.Var)
            or not isinstance(dst_arg, core_ir.Var)
            or tile_arg.unique_id not in self._uid_to_name
            or dst_arg.unique_id not in self._uid_to_name
        ):
            return
        tile_name = self._uid_to_name[tile_arg.unique_id]
        dst_name = self._uid_to_name[dst_arg.unique_id]
        src_node = self.values.get(tile_name)
        dst_node = self.values.get(dst_name)
        if src_node is None or dst_node is None:
            return
        op_id = f"op_{self._op_counter}_{dst_name}_store"
        self._op_counter += 1
        # Mark `dst_node` as produced by this store so downstream consumers see it
        # as an intermediate GM result rather than an unconstrained input.
        self.values[dst_name] = LayoutValueNode(
            name=dst_node.name,
            shape=dst_node.shape,
            dtype=dst_node.dtype,
            memory_space=dst_node.memory_space,
            current_layout=dst_node.current_layout,
            admissible_layouts=dst_node.admissible_layouts,
            producer_op_id=op_id,
            is_reusable_weight=dst_node.is_reusable_weight,
            is_external_output=dst_node.is_external_output,
        )
        configs: list[OperatorConfig] = []
        for s_lay in src_node.admissible_layouts:
            for d_lay in dst_node.admissible_layouts:
                if d_lay == core_ir.TensorLayout.NZ and not self._model.supports_acc_nz_store(
                    src_node, dst_node
                ):
                    continue
                cost = self._model.estimate_store_cost(src_node, dst_node, d_lay)
                dst_label = d_lay.name if isinstance(d_lay, core_ir.TensorLayout) else str(d_lay)
                configs.append(
                    OperatorConfig(
                        op_id=op_id,
                        op_name=rhs.op.name,
                        input_layouts=(s_lay,),
                        output_layout=d_lay,
                        op_cost_cycles=cost,
                        notes=f"FIXPIPE {src_node.memory_space.name}->GM({dst_label})",
                    )
                )
        self.operations.append(
            LayoutOperationNode(
                op_id=op_id,
                op_name=rhs.op.name,
                inputs=(tile_name,),
                output=dst_name,
                invocations=self._model.expected_uses * self._loop_multiplier,
                configs=tuple(configs),
            )
        )

    def _extract_generic_tile_op(self, op: core_ir.AssignStmt, rhs: core_ir.Call) -> None:
        """Extract a general `tile.*` op, filtering configs by `Backend::GetTileLayoutSpec`."""
        if not isinstance(op.var.type, core_ir.TileType):
            return
        tile_inputs: list[tuple[int, str, LayoutValueNode]] = []
        for idx, arg in enumerate(rhs.args):
            if isinstance(arg, core_ir.Var) and arg.unique_id in self._uid_to_name:
                in_name = self._uid_to_name[arg.unique_id]
                in_node = self.values.get(in_name)
                if in_node is not None and in_node.memory_space != core_ir.MemorySpace.DDR:
                    tile_inputs.append((idx, in_name, in_node))
        if not tile_inputs:
            return

        op_id = f"op_{self._op_counter}_{op.var.name_hint}_{rhs.op.name.replace('.', '_')}"
        self._op_counter += 1
        dst_node = self._register_tile_var(op.var, op_id)
        if dst_node is None:
            return

        allowed_per_input: list[tuple[LayoutChoice, ...]] = []
        any_row_major_req = False
        for idx, _, in_node in tile_inputs:
            req = self._model.required_input_tile_layout(rhs.op.name, idx)
            if req is not None:
                any_row_major_req = any_row_major_req or (req == core_ir.TileLayout.row_major)
                filtered = tuple(
                    lay
                    for lay in in_node.admissible_layouts
                    if isinstance(lay, TileLayoutDesc) and lay.blayout == req
                )
                allowed_per_input.append(filtered if filtered else (in_node.current_layout,))
            else:
                allowed_per_input.append(in_node.admissible_layouts)

        if any_row_major_req:
            allowed_outputs: tuple[LayoutChoice, ...] = tuple(
                lay
                for lay in dst_node.admissible_layouts
                if isinstance(lay, TileLayoutDesc) and lay.blayout == core_ir.TileLayout.row_major
            ) or (dst_node.current_layout,)
        else:
            allowed_outputs = dst_node.admissible_layouts

        cost = float(dst_node.num_elements) / self._model.vec_elements_per_cycle
        configs: list[OperatorConfig] = []
        for in_combo in itertools.product(*allowed_per_input):
            for out_lay in allowed_outputs:
                if len(set(in_combo)) == 1 and out_lay == in_combo[0]:
                    configs.append(
                        OperatorConfig(
                            op_id=op_id,
                            op_name=rhs.op.name,
                            input_layouts=tuple(in_combo),
                            output_layout=out_lay,
                            op_cost_cycles=cost,
                            notes=f"Backend-validated {rhs.op.name}",
                        )
                    )
        if not configs:
            configs.append(
                OperatorConfig(
                    op_id=op_id,
                    op_name=rhs.op.name,
                    input_layouts=tuple(in_node.current_layout for _, _, in_node in tile_inputs),
                    output_layout=dst_node.current_layout,
                    op_cost_cycles=cost,
                    notes=f"Backend-validated {rhs.op.name}",
                )
            )
        self.operations.append(
            LayoutOperationNode(
                op_id=op_id,
                op_name=rhs.op.name,
                inputs=tuple(name for _, name, _ in tile_inputs),
                output=dst_node.name,
                invocations=self._model.expected_uses * self._loop_multiplier,
                configs=tuple(configs),
            )
        )

    def _walk_assign_stmt(self, op: core_ir.AssignStmt) -> None:
        rhs = op.value
        if isinstance(rhs, core_ir.Call):
            op_name = rhs.op.name
            if op_name == _TILE_CREATE_OP:
                self._register_tile_var(op.var, producer_op_id=None)
            elif op_name == _TILE_LOAD_OP:
                self._extract_load(op, rhs)
            elif op_name == _TILE_TRANSPOSE_VIEW_OP:
                self._extract_transpose_view(op, rhs)
            elif op_name == _TILE_MOVE_OP:
                self._extract_move(op, rhs)
            elif op_name in _TILE_MATMUL_OPS:
                self._extract_matmul(op, rhs)
            elif op_name == _TILE_STORE_OP:
                self._extract_store(rhs)
            elif op_name.startswith("tile."):
                self._extract_generic_tile_op(op, rhs)

    def _walk_for_stmt(self, op: core_ir.ForStmt) -> None:
        for iter_arg in op.iter_args:
            init_val = getattr(iter_arg, "initValue", None)
            if isinstance(init_val, core_ir.Var) and init_val.unique_id in self._uid_to_name:
                self._uid_to_name[iter_arg.unique_id] = self._uid_to_name[init_val.unique_id]
        self._yield_stack.append([])
        prev_mult = self._loop_multiplier
        self._loop_multiplier = prev_mult * _for_trip_count(op)
        try:
            self._walk_stmt(op.body)
        finally:
            self._loop_multiplier = prev_mult
            yielded_uids = self._yield_stack.pop()

        for idx, rv in enumerate(op.return_vars):
            mapped_name: str | None = None
            if idx < len(yielded_uids):
                mapped_name = self._uid_to_name.get(yielded_uids[idx])
            if mapped_name is None and idx < len(op.iter_args):
                mapped_name = self._uid_to_name.get(op.iter_args[idx].unique_id)
            if mapped_name is not None:
                self._uid_to_name[rv.unique_id] = mapped_name

    def _walk_stmt(self, stmt: core_ir.Stmt) -> None:
        if isinstance(stmt, core_ir.AssignStmt):
            self._walk_assign_stmt(stmt)
        elif isinstance(stmt, core_ir.ForStmt):
            self._walk_for_stmt(stmt)
            return
        elif isinstance(stmt, core_ir.YieldStmt):
            if self._yield_stack:
                self._yield_stack[-1] = [
                    val.unique_id if isinstance(val, core_ir.Var) else -1 for val in stmt.value
                ]
        elif isinstance(stmt, core_ir.EvalStmt):
            if isinstance(stmt.expr, core_ir.Call) and stmt.expr.op.name == _TILE_STORE_OP:
                self._extract_store(stmt.expr)
        elif isinstance(stmt, core_ir.ReturnStmt):
            for val in stmt.value:
                if isinstance(val, core_ir.Call) and val.op.name == _TILE_STORE_OP:
                    self._extract_store(val)

        for child in _iter_child_stmts(stmt):
            self._walk_stmt(child)


def extract_layout_instance(
    program: core_ir.Program,
    *,
    cost_model: LayoutCostModel | None = None,
    reusable_tensors: set[str] | None = None,
) -> LayoutProblemInstance:
    """Extract a `LayoutProblemInstance` from a post-`InferTileMemorySpace` `Program`.

    Reads out all GM tensors (`DDR`), L1/UB tiles (`Mat`, `Vec`), and L0 tiles
    (`Left`, `Right`, `Acc`) along with their operations and candidate
    `OperatorConfig`s.
    """
    model = cost_model or LayoutCostModel()
    reusable = reusable_tensors or set()
    values: dict[str, LayoutValueNode] = {}
    operations: list[LayoutOperationNode] = []

    incore_funcs = [f for f in program.functions.values() if core_ir.is_incore_type(f.func_type)]
    orch_funcs = [f for f in program.functions.values() if not core_ir.is_incore_type(f.func_type)]
    ext_out_names: set[str] | None = None
    call_counts: dict[str, int] = {}
    call_arg_names: dict[str, list[str | None]] = {}
    if orch_funcs:
        ext_out_names = {
            _clean_ssa_name(p.name_hint)
            for fn in orch_funcs
            for p, d in zip(fn.params, fn.param_directions, strict=True)
            if d in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut)
        }
        func_by_name = {f.name: f for f in program.functions.values()}
        call_counter = _OrchCallMultiplierVisitor(func_by_name)
        for fn in orch_funcs:
            call_counter.walk(fn)
        call_counts = call_counter.call_counts
        call_arg_names = call_counter.call_arg_names

    multi_incore = len(incore_funcs) > 1
    op_offset = 0
    for func in incore_funcs:
        prefix = f"{func.name}." if multi_incore else ""
        extractor = _PostInferInstanceExtractor(
            model,
            reusable,
            external_out_names=ext_out_names,
            shared_values=values,
            op_offset=op_offset,
            tile_prefix=prefix,
            call_multiplier=call_counts.get(func.name, 1),
            param_name_overrides=call_arg_names.get(func.name),
        )
        extractor.extract(func)
        values = extractor.values
        operations.extend(extractor.operations)
        op_offset = extractor.op_counter

    return LayoutProblemInstance(values=values, operations=operations, cost_model=model)


def evaluate_assignment(
    instance: LayoutProblemInstance,
    assignment: LayoutAssignment,
) -> CostBreakdown:
    """Evaluate a `LayoutAssignment` under the paper's objective (Section 2 of arXiv:2608.21555).

    Computes:
    - Operator execution cost $\\sum_{o \\in V} R_o \\cdot C^{\\text{op}}_o(q_o)$,
    - Conversion cost across distinct consumer layouts $\\ell \\in \\text{cons}(t)$,
      where conversion of a value $t$ to each distinct consumer layout
      $\\ell \\in \\text{cons}(t)$ is charged **once** (shared across consumers
      requesting the same $\\ell$).
    """
    model = instance.cost_model
    per_op_costs: dict[str, float] = {}
    op_cost_total = 0.0

    # Track the produced layout of each value and the set of distinct layouts
    # demanded by its consumers.
    produced_layout: dict[str, LayoutChoice] = {
        name: node.current_layout for name, node in instance.values.items() if node.producer_op_id is None
    }
    consumer_layouts: dict[str, set[LayoutChoice]] = {name: set() for name in instance.values}
    resolved_configs: dict[str, OperatorConfig] = {}

    for op in instance.operations:
        chosen_cfg = assignment.op_configs.get(op.op_id)
        if chosen_cfg is None:
            out_target = assignment.value_layouts.get(op.output, instance.values[op.output].current_layout)
            in_targets = tuple(
                assignment.value_layouts.get(inp, instance.values[inp].current_layout) for inp in op.inputs
            )
            # Match exact config first; otherwise pick the cheapest config producing `out_target`.
            matching = [
                c for c in op.configs if c.input_layouts == in_targets and c.output_layout == out_target
            ]
            if not matching:
                matching = [c for c in op.configs if c.output_layout == out_target]
            if not matching:
                matching = list(op.configs)
            chosen_cfg = min(matching, key=lambda c: c.op_cost_cycles)

        resolved_configs[op.op_id] = chosen_cfg
        cost = float(op.invocations) * chosen_cfg.op_cost_cycles
        per_op_costs[op.op_id] = cost
        op_cost_total += cost
        produced_layout[op.output] = chosen_cfg.output_layout

        for inp_name, req_layout in zip(op.inputs, chosen_cfg.input_layouts, strict=True):
            consumer_layouts.setdefault(inp_name, set()).add(req_layout)

    # External output tensors must be delivered in their caller-expected layout
    # unless `allow_external_output_relayout` is enabled.
    if not model.allow_external_output_relayout:
        for val_name, val_node in instance.values.items():
            if val_node.is_external_output:
                consumer_layouts.setdefault(val_name, set()).add(val_node.current_layout)

    per_value_conv: dict[str, float] = {}
    conv_cost_total = 0.0
    for val_name, req_set in consumer_layouts.items():
        if not req_set or val_name not in instance.values:
            continue
        val_node = instance.values[val_name]
        src_lay = produced_layout.get(val_name, val_node.current_layout)
        # Reusable GM weights pay conversion once; streaming values pay per invocation.
        multiplier = (
            1.0
            if (val_node.memory_space == core_ir.MemorySpace.DDR and val_node.is_reusable_weight)
            else float(model.expected_uses)
        )
        val_conv = 0.0
        for dst_lay in req_set:
            if dst_lay != src_lay:
                val_conv += multiplier * model.estimate_conversion_cost(val_node, src_lay, dst_lay)
        per_value_conv[val_name] = val_conv
        conv_cost_total += val_conv

    resolved_value_layouts = dict(produced_layout)
    resolved_value_layouts.update(assignment.value_layouts)
    full_assignment = LayoutAssignment(value_layouts=resolved_value_layouts, op_configs=resolved_configs)
    return CostBreakdown(
        assignment=full_assignment,
        total_cost_cycles=op_cost_total + conv_cost_total,
        operator_cost_cycles=op_cost_total,
        conversion_cost_cycles=conv_cost_total,
        per_op_costs=per_op_costs,
        per_value_conversion_costs=per_value_conv,
    )


def enumerate_assignments(
    instance: LayoutProblemInstance,
    *,
    deduplicate: bool = True,
) -> list[CostBreakdown]:
    """Enumerate all dataflow-consistent, backend-legal layout assignments.

    Walks `instance.operations` in order, enforcing that intermediate non-Vec
    edges (`Mat`, `Left`, `Right`, `Acc`, and intra-program GM buffers) match
    the producer's output layout, and that shared GM input parameters pick one
    consistent layout across consumers.

    Args:
        instance: Problem instance extracted by `extract_layout_instance`.
        deduplicate: When True (default), keep only the lowest-cost breakdown
            per distinct `value_layouts` mapping so each enumerated variant maps
            1:1 to a unique compiled program.

    Returns:
        List of `CostBreakdown` objects sorted by ascending `total_cost_cycles`.
    """
    if not instance.operations:
        empty = evaluate_assignment(instance, LayoutAssignment(value_layouts={}))
        return [empty]

    ops = instance.operations
    results: list[CostBreakdown] = []

    def _search(
        idx: int,
        val_map: dict[str, LayoutChoice],
        op_map: dict[str, OperatorConfig],
    ) -> None:
        if idx == len(ops):
            candidate = LayoutAssignment(value_layouts=dict(val_map), op_configs=dict(op_map))
            results.append(evaluate_assignment(instance, candidate))
            return

        op = ops[idx]
        for cfg in op.configs:
            consistent = True
            new_vals: dict[str, LayoutChoice] = {}
            for inp_name, req_lay in zip(op.inputs, cfg.input_layouts, strict=True):
                inp_node = instance.values[inp_name]
                existing = val_map.get(inp_name)
                if existing is not None:
                    # Only Vec -> tile.move can change on-chip layout on an existing edge.
                    can_relayout = (
                        inp_node.memory_space == core_ir.MemorySpace.Vec and op.op_name == _TILE_MOVE_OP
                    )
                    if not can_relayout and existing != req_lay:
                        consistent = False
                        break
                else:
                    new_vals[inp_name] = req_lay
            if not consistent:
                continue

            existing_out = val_map.get(op.output)
            if existing_out is not None and existing_out != cfg.output_layout:
                continue
            new_vals[op.output] = cfg.output_layout

            val_map.update(new_vals)
            op_map[op.op_id] = cfg
            _search(idx + 1, val_map, op_map)
            op_map.pop(op.op_id)
            for k in new_vals:
                val_map.pop(k, None)

    _search(0, {}, {})

    if deduplicate:
        best_by_layout: dict[tuple[tuple[str, str], ...], CostBreakdown] = {}
        for bd in results:
            sig = tuple(sorted((k, str(v)) for k, v in bd.assignment.value_layouts.items()))
            prev = best_by_layout.get(sig)
            if prev is None or bd.total_cost_cycles < prev.total_cost_cycles:
                best_by_layout[sig] = bd
        results = list(best_by_layout.values())

    results.sort(key=lambda b: b.total_cost_cycles)
    return results


def _tensor_type_layout(tensor_type: core_ir.TensorType) -> core_ir.TensorLayout:
    return tensor_type.tensor_view.layout if tensor_type.tensor_view is not None else core_ir.TensorLayout.ND


def _with_tensor_layout(tensor_type: core_ir.TensorType, layout: core_ir.TensorLayout) -> core_ir.TensorType:
    if _tensor_type_layout(tensor_type) == layout:
        return tensor_type
    old_view = tensor_type.tensor_view
    if layout == core_ir.TensorLayout.ND and (
        old_view is None
        or (not old_view.stride and not old_view.valid_shape and old_view.pad == core_ir.PadValue.null)
    ):
        return TensorType(tensor_type.shape, tensor_type.dtype, tensor_type.memref, None)

    stride = list(old_view.stride) if old_view is not None else []
    valid_shape = list(old_view.valid_shape) if old_view is not None else []
    pad = old_view.pad if old_view is not None else core_ir.PadValue.null
    new_view = TensorView(stride=stride, layout=layout, valid_shape=valid_shape, pad=pad)
    return TensorType(tensor_type.shape, tensor_type.dtype, tensor_type.memref, new_view)


def _with_tile_layout(tile_type: core_ir.TileType, desc: TileLayoutDesc) -> core_ir.TileType:
    if _effective_tile_layout(tile_type) == desc:
        return tile_type
    old_view = tile_type.tile_view
    valid_shape = list(old_view.valid_shape) if (old_view and old_view.valid_shape) else list(tile_type.shape)
    stride = list(old_view.stride) if old_view else []
    start_offset = old_view.start_offset if old_view else None
    pad = old_view.pad if old_view else core_ir.PadValue.null
    compact = old_view.compact if old_view else core_ir.CompactMode.null
    new_view = TileView(
        valid_shape=valid_shape,
        stride=stride,
        start_offset=start_offset,
        blayout=desc.blayout,
        slayout=desc.slayout,
        fractal=desc.fractal,
        pad=pad,
        compact=compact,
    )
    return TileType(tile_type.shape, tile_type.dtype, tile_type.memref, new_view, tile_type.memory_space)


class _LayoutAssignmentWriter(core_ir.IRMutator):
    """Write a `LayoutAssignment` (both GM `TensorLayout` and on-chip `TileLayoutDesc`) back to IR."""

    def __init__(self, assignment: LayoutAssignment) -> None:
        super().__init__()
        self._tensor_layouts = assignment.tensor_layouts
        self._tile_layouts = assignment.tile_layouts
        self._scoped_tensor_layouts: dict[tuple[str, str], core_ir.TensorLayout] = {}
        self._var_remap: dict[int, core_ir.Var] = {}
        self._func_return_types: dict[str, list[core_ir.Type]] = {}
        self._current_func_name = ""

    def _lookup_tensor_layout(self, name_hint: str) -> core_ir.TensorLayout | None:
        clean_name = _clean_ssa_name(name_hint)
        if self._current_func_name:
            scoped = self._scoped_tensor_layouts.get((self._current_func_name, clean_name))
            if scoped is not None:
                return scoped
        return self._tensor_layouts.get(clean_name, self._tensor_layouts.get(name_hint))

    def rewrite(self, program: core_ir.Program) -> core_ir.Program:
        funcs = list(program.functions.values())
        incore_funcs = [f for f in funcs if core_ir.is_incore_type(f.func_type)]
        other_funcs = [f for f in funcs if not core_ir.is_incore_type(f.func_type)]
        func_by_name = {f.name: f for f in funcs}
        if other_funcs:
            visitor = _OrchCallMultiplierVisitor(func_by_name)
            for fn in other_funcs:
                visitor.walk(fn)
            for callee_name, arg_names in visitor.call_arg_names.items():
                callee = func_by_name.get(callee_name)
                if callee is None:
                    continue
                for param, arg_name in zip(callee.params, arg_names, strict=True):
                    if arg_name is None:
                        continue
                    lay = self._tensor_layouts.get(arg_name)
                    if lay is not None:
                        self._scoped_tensor_layouts[(callee_name, _clean_ssa_name(param.name_hint))] = lay

        rewritten_by_name: dict[str, core_ir.Function] = {}
        for func in incore_funcs + other_funcs:
            rewritten_by_name[func.name] = self.visit_function(func)
        new_funcs = [rewritten_by_name[f.name] for f in funcs]
        if all(nf.same_as(of) for nf, of in zip(new_funcs, funcs, strict=True)):
            return program
        return core_ir.Program(new_funcs, program.name, program.span)

    def _lookup_tile_desc(self, name_hint: str) -> TileLayoutDesc | None:
        clean_name = _clean_ssa_name(name_hint)
        if self._current_func_name:
            scoped = self._tile_layouts.get(f"{self._current_func_name}.{clean_name}")
            if scoped is not None:
                return scoped
        return self._tile_layouts.get(clean_name, self._tile_layouts.get(name_hint))

    def visit_function(self, func: core_ir.Function) -> core_ir.Function:
        self._var_remap.clear()
        self._current_func_name = func.name
        new_params: list[tuple[core_ir.Var, core_ir.ParamDirection]] = []
        changed = False
        for param, direction in zip(func.params, func.param_directions, strict=True):
            target_layout = self._lookup_tensor_layout(param.name_hint)
            if (
                isinstance(param.type, core_ir.TensorType)
                and target_layout is not None
                and _tensor_type_layout(param.type) != target_layout
            ):
                new_type = _with_tensor_layout(param.type, target_layout)
                new_param = core_ir.Var(param.name_hint, new_type, param.span)
                self._var_remap[param.unique_id] = new_param
                new_params.append((new_param, direction))
                changed = True
            else:
                new_params.append((param, direction))

        new_body = self.visit_stmt(func.body)
        new_return_types: list[core_ir.Type] = list(func.return_types)
        if len(new_return_types) == 1 and isinstance(new_return_types[0], core_ir.TensorType):
            for param, direction in new_params:
                if direction in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut):
                    target_layout = self._lookup_tensor_layout(param.name_hint)
                    if (
                        target_layout is not None
                        and isinstance(param.type, core_ir.TensorType)
                        and _tensor_type_layout(new_return_types[0]) != target_layout
                    ):
                        new_return_types[0] = _with_tensor_layout(new_return_types[0], target_layout)
                        changed = True
                        break

        self._func_return_types[func.name] = new_return_types
        if not changed and new_body.same_as(func.body):
            return func

        # If any parameter was newly set to `pl.NZ`, strip any stale `nz_tensor_views_blocked`
        # stamp so `BlockNzTensorViews` can block the updated function.
        new_attrs = dict(func.attrs) if func.attrs else {}
        if any(lay == core_ir.TensorLayout.NZ for lay in self._tensor_layouts.values()):
            new_attrs.pop("nz_tensor_views_blocked", None)

        return core_ir.Function(
            func.name,
            new_params,
            new_return_types,
            new_body,
            func.span,
            type=func.func_type,
            level=func.level,
            role=func.role,
            attrs=new_attrs if new_attrs else None,
            requires_runtime_binding=func.requires_runtime_binding,
            ir_stage=func.ir_stage,
        )

    def visit_var(self, op: core_ir.Var) -> core_ir.Expr:
        return self._var_remap.get(op.unique_id, op)

    def visit_assign_stmt(self, op: core_ir.AssignStmt) -> core_ir.Stmt:
        rhs = op.value
        clean_name = _clean_ssa_name(op.var.name_hint)
        target_tile_desc = self._lookup_tile_desc(op.var.name_hint)
        target_tensor_layout = self._tensor_layouts.get(
            clean_name, self._tensor_layouts.get(op.var.name_hint)
        )

        # Rewrite `tensor.create` (e.g., Orchestration-allocated intermediate GM buffers).
        if (
            isinstance(rhs, core_ir.Call)
            and rhs.op.name == _TENSOR_CREATE_OP
            and target_tensor_layout is not None
            and isinstance(op.var.type, core_ir.TensorType)
            and _tensor_type_layout(op.var.type) != target_tensor_layout
        ):
            new_args = [self.visit_expr(arg) for arg in rhs.args]
            new_kwargs = dict(rhs.kwargs)
            new_kwargs["layout"] = target_tensor_layout
            new_value = core_ir.create_op_call(rhs.op.name, new_args, new_kwargs, rhs.span)
            if rhs.attrs:
                new_value = core_ir.set_call_attrs(new_value, dict(rhs.attrs))
            new_var = core_ir.Var(op.var.name_hint, new_value.type, op.var.span)
            self._var_remap[op.var.unique_id] = new_var
            return core_ir.AssignStmt(new_var, new_value, op.span)

        # If this is a `tile.move` and a specific L0/L1/UB `TileLayoutDesc` was chosen
        # for its destination variable, pass `blayout` and `slayout` explicitly on `tile.move`.
        if (
            isinstance(rhs, core_ir.Call)
            and rhs.op.name == _TILE_MOVE_OP
            and target_tile_desc is not None
            and isinstance(op.var.type, core_ir.TileType)
            and _effective_tile_layout(op.var.type) != target_tile_desc
        ):
            new_args = [self.visit_expr(arg) for arg in rhs.args]
            new_kwargs = dict(rhs.kwargs)
            new_kwargs["blayout"] = target_tile_desc.blayout
            new_kwargs["slayout"] = target_tile_desc.slayout
            new_value = core_ir.create_op_call(rhs.op.name, new_args, new_kwargs, rhs.span)
            if rhs.attrs:
                new_value = core_ir.set_call_attrs(new_value, dict(rhs.attrs))
            new_var = core_ir.Var(op.var.name_hint, new_value.type, op.var.span)
            self._var_remap[op.var.unique_id] = new_var
            return core_ir.AssignStmt(new_var, new_value, op.span)

        new_value = self.visit_expr(op.value)
        new_var = op.var
        if not new_value.same_as(op.value) and isinstance(
            op.var.type, (core_ir.TensorType, core_ir.TileType)
        ):
            if isinstance(new_value.type, (core_ir.TensorType, core_ir.TileType)):
                new_var = core_ir.Var(op.var.name_hint, new_value.type, op.var.span)
                self._var_remap[op.var.unique_id] = new_var
        elif target_tile_desc is not None and isinstance(op.var.type, core_ir.TileType):
            updated_type = _with_tile_layout(op.var.type, target_tile_desc)
            if updated_type != op.var.type:
                new_var = core_ir.Var(op.var.name_hint, updated_type, op.var.span)
                self._var_remap[op.var.unique_id] = new_var

        if new_var.same_as(op.var) and new_value.same_as(op.value):
            return op
        return core_ir.AssignStmt(new_var, new_value, op.span)

    def visit_call(self, op: core_ir.Call) -> core_ir.Expr:
        new_args = [self.visit_expr(arg) for arg in op.args]
        args_changed = any(not a.same_as(b) for a, b in zip(new_args, op.args, strict=True))
        if not args_changed:
            return op
        if isinstance(op.op, core_ir.GlobalVar):
            ret_types = self._func_return_types.get(op.op.name)
            ret_type = ret_types[0] if ret_types and len(ret_types) == 1 else op.type
            return core_ir.Call(
                op.op,
                new_args,
                dict(op.kwargs),
                dict(op.attrs) if op.attrs else None,
                ret_type,
                op.span,
            )
        rebuilt = core_ir.create_op_call(op.op.name, new_args, dict(op.kwargs), op.span)
        if op.attrs:
            rebuilt = core_ir.set_call_attrs(rebuilt, dict(op.attrs))
        return rebuilt

    def visit_submit(self, op: core_ir.Submit) -> core_ir.Expr:
        new_args = [self.visit_expr(arg) for arg in op.args]
        new_deps = [self.visit_expr(dep) for dep in op.deps]
        args_changed = any(not a.same_as(b) for a, b in zip(new_args, op.args, strict=True))
        deps_changed = any(not a.same_as(b) for a, b in zip(new_deps, op.deps, strict=True))
        if not args_changed and not deps_changed:
            return op
        ret_type = op.type
        if isinstance(op.op, core_ir.GlobalVar):
            ret_types = self._func_return_types.get(op.op.name)
            if ret_types is not None:
                ret_type = core_ir.make_call_result_type(ret_types, is_submit=True)
        return core_ir.Submit(
            op.op,
            new_args,
            new_deps,
            dict(op.kwargs),
            dict(op.attrs) if op.attrs else None,
            ret_type,
            op.span,
            op.core_num,
            op.sync_start,
            op.allow_early_resolve,
            op.predicate,
        )


def apply_layout_assignment(
    program: core_ir.Program,
    assignment: LayoutAssignment,
) -> core_ir.Program:
    """Write a `LayoutAssignment` back into a post-`InferTileMemorySpace` `Program`."""
    writer = _LayoutAssignmentWriter(assignment)
    return writer.rewrite(program)


def make_auto_layout_pass(
    assignment: LayoutAssignment,
    *,
    name: str = "AutoLayout",
) -> passes.Pass:
    """Wrap `apply_layout_assignment` as a `passes.Pass` insertable after `InferTileMemorySpace`."""
    return passes.create_program_pass(lambda prog: apply_layout_assignment(prog, assignment), name=name)


def _format_layout_choice(choice: LayoutChoice) -> str:
    if isinstance(choice, core_ir.TensorLayout):
        return choice.name
    return choice.short_name()


def _serialize_assignment(assignment: LayoutAssignment) -> dict[str, Any]:
    return {
        "value_layouts": {k: _format_layout_choice(v) for k, v in assignment.value_layouts.items()},
        "gm_layouts": {k: v.name for k, v in assignment.tensor_layouts.items()},
        "tile_layouts": {k: v.short_name() for k, v in assignment.tile_layouts.items()},
    }


@dataclass(frozen=True)
class LayoutVariantArtifact:
    """One enumerated and compiled layout solution ready for hardware benchmarking."""

    variant_id: str
    breakdown: CostBreakdown
    build_dir: Path
    pto_files: tuple[Path, ...]
    gm_input_layouts: dict[str, str]
    gm_output_layouts: dict[str, str]
    gm_layouts: dict[str, str]


def _extract_gm_io_layouts(
    instance: LayoutProblemInstance,
    assignment: LayoutAssignment,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    in_layouts: dict[str, str] = {}
    out_layouts: dict[str, str] = {}
    all_gm_layouts: dict[str, str] = {}
    for name, node in instance.values.items():
        if node.memory_space != core_ir.MemorySpace.DDR:
            continue
        lay = assignment.tensor_layouts.get(name, node.current_layout)
        lay_str = lay.name if isinstance(lay, core_ir.TensorLayout) else str(lay)
        all_gm_layouts[name] = lay_str
        if node.is_external_output:
            out_layouts[name] = lay_str
        elif node.producer_op_id is None:
            in_layouts[name] = lay_str
    return in_layouts, out_layouts, all_gm_layouts


def compile_layout_variants(  # noqa: PLR0913
    program: core_ir.Program,
    *,
    output_dir: str | Path,
    cost_model: LayoutCostModel | None = None,
    reusable_tensors: set[str] | None = None,
    backend_type: core_backend.BackendType = core_backend.BackendType.Ascend910,
    execution_mode: ExecutionMode = ExecutionMode.SIM,
    skip_ptoas: bool = False,
    memory_planner: passes.MemoryPlanner = passes.MemoryPlanner.PYPTO,
    max_variants: int | None = None,
    dump_passes: bool = False,
) -> tuple[LayoutProblemInstance, list[LayoutVariantArtifact]]:
    """Enumerate all legal layout solutions for `program` and compile each one.

    Runs the pass pipeline up through `InferTileMemorySpace` to extract the
    `LayoutProblemInstance`, enumerates every dataflow-consistent, backend-legal
    `LayoutAssignment`, and compiles each variant through the full `ir.compile`
    pipeline with `make_auto_layout_pass(assignment)` inserted after
    `InferTileMemorySpace`. Also writes `<output_dir>/manifest.json`.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    model = cost_model or LayoutCostModel(backend_type=backend_type)

    core_backend.set_backend_type(backend_type)
    outer_ctx = passes.PassContext.current()
    if outer_ctx is None:
        with passes.PassContext([], memory_planner=memory_planner):
            pre_prog = run_pre_layout_passes(program)
            instance = extract_layout_instance(
                pre_prog,
                cost_model=model,
                reusable_tensors=reusable_tensors,
            )
            candidates = enumerate_assignments(instance, deduplicate=True)
    else:
        pre_prog = run_pre_layout_passes(program)
        instance = extract_layout_instance(
            pre_prog,
            cost_model=model,
            reusable_tensors=reusable_tensors,
        )
        candidates = enumerate_assignments(instance, deduplicate=True)

    if max_variants is not None:
        candidates = candidates[:max_variants]

    variants: list[LayoutVariantArtifact] = []
    manifest_entries: list[dict[str, Any]] = []

    for idx, bd in enumerate(candidates):
        variant_id = f"variant_{idx:02d}"
        v_dir = out_root / variant_id
        v_dir.mkdir(parents=True, exist_ok=True)

        def _make_pass(assign: LayoutAssignment = bd.assignment) -> passes.Pass:
            return make_auto_layout_pass(assign)

        ir_compile(
            program,
            output_dir=str(v_dir),
            dump_passes=dump_passes,
            backend_type=backend_type,
            execution_mode=execution_mode,
            skip_ptoas=skip_ptoas,
            memory_planner=None if outer_ctx is not None else memory_planner,
            layout_pass=_make_pass,
        )

        pto_files = tuple(sorted((v_dir / "kernels").rglob("*.pto"))) if (v_dir / "kernels").is_dir() else ()
        gm_in, gm_out, gm_all = _extract_gm_io_layouts(instance, bd.assignment)
        artifact = LayoutVariantArtifact(
            variant_id=variant_id,
            breakdown=bd,
            build_dir=v_dir,
            pto_files=pto_files,
            gm_input_layouts=gm_in,
            gm_output_layouts=gm_out,
            gm_layouts=gm_all,
        )
        variants.append(artifact)
        manifest_entries.append(
            {
                "variant_id": variant_id,
                "build_dir": str(v_dir),
                "pto_files": [str(p) for p in pto_files],
                "gm_input_layouts": gm_in,
                "gm_output_layouts": gm_out,
                "gm_layouts": gm_all,
                "assignment": _serialize_assignment(bd.assignment),
                "predicted": {
                    "total_cost_cycles": bd.total_cost_cycles,
                    "operator_cost_cycles": bd.operator_cost_cycles,
                    "conversion_cost_cycles": bd.conversion_cost_cycles,
                    "per_op_costs": bd.per_op_costs,
                    "per_value_conversion_costs": bd.per_value_conversion_costs,
                },
            }
        )

    manifest = {
        "program": program.name,
        "backend_type": backend_type.name,
        "execution_mode": execution_mode.name,
        "skip_ptoas": skip_ptoas,
        "num_variants": len(manifest_entries),
        "reusable_tensors": sorted(reusable_tensors or ()),
        "values": {
            name: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "memory_space": v.memory_space.name,
                "byte_size": v.byte_size,
                "is_reusable_weight": v.is_reusable_weight,
                "is_external_output": v.is_external_output,
            }
            for name, v in instance.values.items()
        },
        "variants": manifest_entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return instance, variants


@dataclass(frozen=True)
class CalibrationMeasurement:
    """One measured hardware/simulator latency observation for cost-model fitting."""

    variant_id: str
    measured_us: float
    instance: LayoutProblemInstance
    breakdown: CostBreakdown
    label: str = ""


def _empty_float_dict() -> dict[str, float]:
    return {}


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of fitting `LayoutCostModel` parameters to device measurements."""

    fitted_model: LayoutCostModel
    fixed_overhead_us: float
    effective_freq_ghz: float
    r2: float
    spearman_rho: float
    mape_percent: float
    residuals_us: dict[str, float]
    fallback_params: tuple[str, ...] = ()
    effective_freq_fallback: bool = False
    param_t_stats: dict[str, float] = field(default_factory=_empty_float_dict)


def _rank_array(vals: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(vals)), key=vals.__getitem__)
    ranks = np.empty(len(vals), dtype=np.float64)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 1.0
    rx = _rank_array(xs)
    ry = _rank_array(ys)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx_c * rx_c) * np.sum(ry_c * ry_c)))
    if denom == 0.0:
        return 1.0
    return float(np.sum(rx_c * ry_c) / denom)


def fit_cost_model_from_measurements(
    measurements: Sequence[CalibrationMeasurement],
    base_model: LayoutCostModel | None = None,
    *,
    nominal_freq_ghz: float = 1.8,
    min_t_stat: float = 2.0,
    max_physical_bpc: float = 1024.0,
) -> CalibrationReport:
    """Fit MTE2/FIXPIPE bandwidth coefficients & dispatch offset to measured latencies.

    Decomposes each measured single-run device execution time $T_{\\text{meas}}$ into:
    - Fixed dispatch/launch overhead $\\beta_0$ ($\\mu$s),
    - `ND -> Mat` load bytes / `nd2nz_bw` (per-run $\\text{bytes} \\cdot \\text{loop\\_trips}$),
    - `NZ -> Mat` load bytes / `nz2nz_bw`,
    - `Acc -> ND` store bytes / `store_nd_bw`,
    - `Acc -> NZ` store bytes / `store_nz_bw`, and
    - Remaining modeled cycles (`MTE1` + `Cube` + other ops).

    Returns a calibrated `LayoutCostModel` plus goodness-of-fit diagnostics.
    """
    model = base_model or LayoutCostModel()
    if not measurements:
        return CalibrationReport(
            fitted_model=model,
            fixed_overhead_us=0.0,
            effective_freq_ghz=nominal_freq_ghz,
            r2=1.0,
            spearman_rho=1.0,
            mape_percent=0.0,
            residuals_us={},
        )

    # Extract per-run byte volumes and compute cycles by operation category.
    # Solver `op.invocations` scales runtime costs by `expected_uses`; normalize
    # back to a single kernel invocation to match single-run `measured_us`.
    # Features: [1.0, nd2nz_bytes, nz2nz_bytes, store_nd_bytes, store_nz_bytes, compute_cycles]
    rows_x: list[list[float]] = []
    rows_y: list[float] = []
    labels: list[str] = []

    for m in measurements:
        nd2nz_b = 0.0
        nz2nz_b = 0.0
        store_nd_b = 0.0
        store_nz_b = 0.0
        other_cycles = 0.0
        run_scale = float(max(1, m.instance.cost_model.expected_uses))

        for op in m.instance.operations:
            cfg = m.breakdown.assignment.op_configs.get(op.op_id)
            if cfg is None:
                continue
            inv = float(op.invocations) / run_scale
            if op.op_name == _TILE_LOAD_OP and op.inputs:
                dst_node = m.instance.values[op.output]
                if dst_node.memory_space == core_ir.MemorySpace.Mat:
                    if cfg.input_layouts[0] == core_ir.TensorLayout.NZ:
                        nz2nz_b += inv * float(dst_node.byte_size)
                    else:
                        nd2nz_b += inv * float(dst_node.byte_size)
                    continue
            elif op.op_name == _TILE_STORE_OP and op.inputs:
                src_node = m.instance.values[op.inputs[0]]
                if cfg.output_layout == core_ir.TensorLayout.NZ:
                    store_nz_b += inv * float(src_node.byte_size)
                else:
                    store_nd_b += inv * float(src_node.byte_size)
                continue
            other_cycles += inv * cfg.op_cost_cycles

        rows_x.append([1.0, nd2nz_b, nz2nz_b, store_nd_b, store_nz_b, other_cycles])
        rows_y.append(float(m.measured_us))
        labels.append(m.label or m.variant_id)

    x_mat = np.asarray(rows_x, dtype=np.float64)
    y_vec = np.asarray(rows_y, dtype=np.float64)
    col_norms = np.sqrt(np.sum(x_mat * x_mat, axis=0))
    safe_scales = np.asarray([s if s > 0.0 else 1.0 for s in col_norms.tolist()], dtype=np.float64)
    x_scaled = np.asarray(x_mat / safe_scales, dtype=np.float64)
    scaled_coef, _, _, _ = np.linalg.lstsq(x_scaled, y_vec, rcond=None)
    coef = scaled_coef / safe_scales
    y_pred = x_mat @ coef

    # OLS parameter standard errors (scaled back from normalized columns for stability).
    n_cols = len(rows_x[0])
    dof = max(1, len(rows_y) - n_cols)
    rss = float(np.sum((y_vec - y_pred) ** 2))
    sigma2 = rss / float(dof)
    scales_list = [float(s) for s in safe_scales.tolist()]
    gram: list[list[float]] = [
        [sum((row[i] / scales_list[i]) * (row[j] / scales_list[j]) for row in rows_x) for j in range(n_cols)]
        for i in range(n_cols)
    ]
    xtx_scaled_inv: np.ndarray = np.linalg.pinv(np.asarray(gram, dtype=np.float64))
    se_list = [
        math.sqrt(max(0.0, float(xtx_scaled_inv[j, j]) * sigma2)) / scales_list[j] for j in range(n_cols)
    ]

    # Reject non-physical clock estimates (negative or pipeline-overlapped > 3.5 GHz)
    # so overlap in `other_cycles` cannot deflate fitted bandwidths.
    raw_freq_ghz = (1.0 / (float(coef[5]) * 1000.0)) if float(coef[5]) > 1e-6 else 0.0
    if 0.5 <= raw_freq_ghz <= 3.5:
        us_per_cycle = float(coef[5])
        eff_freq_ghz = raw_freq_ghz
        freq_fallback = False
    else:
        us_per_cycle = 1.0 / (nominal_freq_ghz * 1000.0)
        eff_freq_ghz = nominal_freq_ghz
        freq_fallback = True

    fallback_params: list[str] = []
    param_t_stats: dict[str, float] = {}

    def _slope_to_bpc(name: str, col_idx: int, fallback: float) -> float:
        slope = float(coef[col_idx])
        err = se_list[col_idx]
        t_val = (slope / err) if err > 1e-18 else (float("inf") if slope > 1e-9 else 0.0)
        param_t_stats[name] = float(t_val)
        if slope <= 1e-9 or t_val < min_t_stat:
            fallback_params.append(name)
            return fallback
        bpc = us_per_cycle / slope
        if bpc > max_physical_bpc:
            fallback_params.append(name)
            return fallback
        return max(1.0, bpc)

    fitted = LayoutCostModel(
        expected_uses=model.expected_uses,
        backend_type=model.backend_type,
        mte2_nd2nz_bytes_per_cycle=_slope_to_bpc(
            "mte2_nd2nz_bytes_per_cycle", 1, model.mte2_nd2nz_bytes_per_cycle
        ),
        mte2_nz2nz_bytes_per_cycle=_slope_to_bpc(
            "mte2_nz2nz_bytes_per_cycle", 2, model.mte2_nz2nz_bytes_per_cycle
        ),
        mte2_nd2nz_row_penalty_cycles=0.0,
        mte2_vec_bytes_per_cycle=model.mte2_vec_bytes_per_cycle,
        mte1_l1_to_l0_bytes_per_cycle=model.mte1_l1_to_l0_bytes_per_cycle,
        mte1_transpose_penalty_cycles=model.mte1_transpose_penalty_cycles,
        cube_macs_per_cycle=model.cube_macs_per_cycle,
        cube_issue_overhead_cycles=model.cube_issue_overhead_cycles,
        vec_elements_per_cycle=model.vec_elements_per_cycle,
        fixpipe_acc_to_nd_bytes_per_cycle=_slope_to_bpc(
            "fixpipe_acc_to_nd_bytes_per_cycle", 3, model.fixpipe_acc_to_nd_bytes_per_cycle
        ),
        fixpipe_acc_to_nz_bytes_per_cycle=_slope_to_bpc(
            "fixpipe_acc_to_nz_bytes_per_cycle", 4, model.fixpipe_acc_to_nz_bytes_per_cycle
        ),
        gm_pack_bytes_per_cycle=model.gm_pack_bytes_per_cycle,
        gm_pack_fixed_cycles=model.gm_pack_fixed_cycles,
        onchip_relayout_bytes_per_cycle=model.onchip_relayout_bytes_per_cycle,
        allow_external_output_relayout=model.allow_external_output_relayout,
    )

    ss_res = float(np.sum((y_vec - y_pred) ** 2))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    mape = float(np.mean(np.abs(y_vec - y_pred) / np.maximum(np.abs(y_vec), 1e-9)) * 100.0)
    rho = _spearman_rank_correlation([float(yp) for yp in y_pred], rows_y)
    residuals = {lbl: float(yp - yt) for lbl, yt, yp in zip(labels, y_vec, y_pred, strict=True)}
    return CalibrationReport(
        fitted_model=fitted,
        fixed_overhead_us=float(coef[0]),
        effective_freq_ghz=eff_freq_ghz,
        r2=r2,
        spearman_rho=rho,
        mape_percent=mape,
        residuals_us=residuals,
        fallback_params=tuple(fallback_params),
        effective_freq_fallback=freq_fallback,
        param_t_stats=param_t_stats,
    )


__all__ = [
    "COL_MAJOR_FLAT",
    "CalibrationMeasurement",
    "CalibrationReport",
    "CostBreakdown",
    "FRACTAL_ACC_NZ",
    "FRACTAL_NZ",
    "FRACTAL_ZN",
    "LayoutAssignment",
    "LayoutChoice",
    "LayoutCostModel",
    "LayoutOperationNode",
    "LayoutProblemInstance",
    "LayoutValueNode",
    "LayoutVariantArtifact",
    "OperatorConfig",
    "ROW_MAJOR_FLAT",
    "TileLayoutDesc",
    "apply_layout_assignment",
    "c0_elements",
    "compile_layout_variants",
    "enumerate_assignments",
    "evaluate_assignment",
    "extract_layout_instance",
    "fit_cost_model_from_measurements",
    "is_nz_compatible_shape",
    "make_auto_layout_pass",
    "pack_nz",
    "run_post_layout_passes",
    "run_pre_layout_passes",
    "unpack_nz",
]
