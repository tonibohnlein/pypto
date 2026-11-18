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
from dataclasses import dataclass, field, replace
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
        invocations: Total executions $R_o$ over the optimization horizon:
            `expected_uses` times nested loop/call multiplicity. Calibration
            divides by `expected_uses` to compare with one-launch measurements.
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
    mte2_nz2nz_bytes_per_cycle: float = 123.0
    mte2_nd2nz_bytes_per_cycle: float = 77.0
    mte2_nd2nz_contiguous_bytes_per_cycle: float = 102.0
    mte2_nd2nz_row_penalty_cycles: float = 0.0
    mte2_nd2nz_sub512b_penalty_cycles: float = 2.5
    mte2_l2_reread_discount: float = 0.30
    l2_cache_line_bytes: int = 512
    mte2_vec_bytes_per_cycle: float = 192.0
    mte1_l1_to_l0_bytes_per_cycle: float = 129.7
    mte1_l0a_bytes_per_cycle: float = 129.7
    mte1_l0b_bytes_per_cycle: float = 85.4
    mte1_transpose_penalty_cycles: float = 8.0
    cube_macs_per_cycle: float = 4096.0
    cube_issue_overhead_cycles: float = 21.0
    mad_fp32_passes: int | None = None
    vec_elements_per_cycle: float = 128.0
    fixpipe_acc_to_nd_bytes_per_cycle: float = 118.0
    fixpipe_acc_to_nz_bytes_per_cycle: float = 122.0
    fixpipe_drain_fixed_cycles: float = 0.0
    fixpipe_odd_n1_penalty_cycles: float = 2.6
    gm_pack_bytes_per_cycle: float = 64.0
    gm_pack_fixed_cycles: float = 120.0
    onchip_relayout_bytes_per_cycle: float = 256.0
    allow_external_input_relayout: bool = True
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

    def gm_access_granularity_bytes(self) -> int:
        """Return the hardware GM access granularity in bytes (`512` on `a2a3`, `128` on `a5`)."""
        return 128 if self.target_arch() == "a5" else self.l2_cache_line_bytes

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
            and len(src_node.shape) >= 2
            and src_node.shape[-2] == rows
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
        if src_layout == src_node.current_layout and len(src_node.admissible_layouts) == 1:
            return dst_layout == dst_node.current_layout
        if dst_node.memory_space == core_ir.MemorySpace.Mat:
            if src_layout == core_ir.TensorLayout.NZ:
                if not is_nz_compatible_shape(dst_node.shape, dst_node.dtype):
                    return False
                c0 = c0_elements(dst_node.dtype)
                if c0 > 0 and len(dst_node.shape) >= 2 and len(src_node.shape) >= 2:
                    col_blocks = dst_node.shape[-1] // c0
                    if col_blocks > 1 and (src_node.shape[-2] - dst_node.shape[-2]) > 65535:
                        return False
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
            return src_layout == src_node.current_layout and dst_layout == FRACTAL_ZN
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

    def effective_mte2_load_multiplier(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        per_run_inv: float,
    ) -> float:
        """Discount outer-loop MTE2 re-reads that hit L2 (`loaded_bytes > src_node.byte_size`)."""
        if per_run_inv <= 1.0 or self.mte2_l2_reread_discount >= 1.0:
            return 1.0
        tile_b = float(dst_node.byte_size)
        uniq_b = float(src_node.byte_size)
        load_b = per_run_inv * tile_b
        if load_b <= uniq_b or load_b <= 0.0:
            return 1.0
        return (uniq_b + self.mte2_l2_reread_discount * (load_b - uniq_b)) / load_b

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
            parent_cols = src_node.shape[-1] if len(src_node.shape) >= 1 else cols
            bpe = _dtype_byte_width(dst_node.dtype)
            row_bytes = cols * bpe
            gran_b = float(max(1, self.gm_access_granularity_bytes()))
            strided = cols < parent_cols
            nd_bw = (
                self.mte2_nd2nz_bytes_per_cycle
                if strided
                else max(self.mte2_nd2nz_bytes_per_cycle, self.mte2_nd2nz_contiguous_bytes_per_cycle)
            )
            c0 = c0_elements(dst_node.dtype)
            col_blocks = max(1, (cols + c0 - 1) // c0)
            sub512_penalty = (
                rows * max(0.0, (gran_b - float(row_bytes)) / gran_b) * self.mte2_nd2nz_sub512b_penalty_cycles
                if strided
                else 0.0
            )
            return (
                total_bytes / nd_bw + rows * col_blocks * self.mte2_nd2nz_row_penalty_cycles + sub512_penalty
            )
        return total_bytes / self.mte2_vec_bytes_per_cycle

    def estimate_move_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate on-chip `tile.move` cost (e.g. `Mat -> Left` or `Mat -> Right`).

        Matches `L0CostModel` in `backend_handler.h`: `L1 -> L0A` (`Left`) and
        `L1 -> L0B` (`Right`) have distinct MTE1 port bandwidths (`129.7` vs
        `85.4 B/cyc` on `a2a3`).
        """
        total_bytes = float(dst_node.byte_size)
        if dst_node.memory_space == core_ir.MemorySpace.Left:
            bw = self.mte1_l0a_bytes_per_cycle
        elif dst_node.memory_space == core_ir.MemorySpace.Right:
            bw = self.mte1_l0b_bytes_per_cycle
        else:
            bw = self.mte1_l1_to_l0_bytes_per_cycle
        base = total_bytes / bw
        if src_layout != dst_layout:
            return base + self.mte1_transpose_penalty_cycles
        return base

    def estimate_matmul_cost(
        self,
        lhs_node: LayoutValueNode,
        rhs_node: LayoutValueNode,
        out_node: LayoutValueNode,
    ) -> float:
        """Estimate L0 `tile.matmul` (`Left` x `Right` -> `Acc`) compute cost.

        Follows `MadCycles` in `l0_tile_chooser.cpp`: full FP32 MMAD requires
        `mad_fp32_passes` (`2` on `a2a3`, `8` on `a5`) per K-fractal due to
        mantissa decomposition, while `<= 2B` dtypes (`FP16`/`BF16`) use `1` pass.
        A K-fractal contains 32 bytes, so FP32 also has twice as many K-fractals
        as FP16 for the same K. `cube_macs_per_cycle` scales the nominal 4096
        FP16-equivalent throughput; M/N/K tails pay for whole hardware fractals.
        """
        del out_node
        m_dim = int(lhs_node.shape[-2])
        k_dim = int(lhs_node.shape[-1])
        n_dim = int(rhs_node.shape[-1])
        bpe = _dtype_byte_width(lhs_node.dtype)
        fp32_passes = (
            self.mad_fp32_passes
            if self.mad_fp32_passes is not None
            else (8 if self.target_arch() == "a5" else 2)
        )
        cpr = max(1, fp32_passes) if bpe == 4 else max(1, bpe // 2)
        k_fractal = max(1, 32 // bpe)
        fractals = ((m_dim + 15) // 16) * ((n_dim + 15) // 16) * ((k_dim + k_fractal - 1) // k_fractal)
        return self.cube_issue_overhead_cycles + cpr * fractals * (4096.0 / self.cube_macs_per_cycle)

    def estimate_store_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate `tile.store` (`Acc -> GM` or `Vec -> GM`) cost.

        Follows `DrainCycles` in `l0_tile_chooser.cpp` (`pto-isa fixpipe-model.md`):
        includes optional per-drain fixed overhead and non-power-of-2 N-fractal
        (`odd(N1) - 1`) burst serialization penalty per M-row.
        """
        del dst_node
        total_bytes = float(src_node.byte_size)
        bw = (
            self.fixpipe_acc_to_nz_bytes_per_cycle
            if dst_layout == core_ir.TensorLayout.NZ
            else self.fixpipe_acc_to_nd_bytes_per_cycle
        )
        m_rows = int(src_node.shape[-2]) if len(src_node.shape) >= 2 else 1
        n_cols = int(src_node.shape[-1]) if len(src_node.shape) >= 1 else 1
        bpe_c = max(1, _dtype_byte_width(src_node.dtype))
        n0 = max(1, 32 // bpe_c)
        n1 = max(1, (n_cols + n0 - 1) // n0)
        odd_n1 = n1
        while (odd_n1 & 1) == 0:
            odd_n1 >>= 1
        odd_penalty = float(m_rows) * self.fixpipe_odd_n1_penalty_cycles * float(odd_n1 - 1)
        return self.fixpipe_drain_fixed_cycles + (total_bytes / bw) + odd_penalty

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
_TENSOR_RESHAPE_OP = core_ir.get_op("tensor.reshape").name
_TENSOR_VIEW_OP = core_ir.get_op("tensor.view").name
_TILE_CREATE_OP = core_ir.get_op("tile.create").name
_TILE_LOAD_OP = core_ir.get_op("tile.load").name
_TILE_STORE_OP = core_ir.get_op("tile.store").name
_TILE_MOVE_OP = core_ir.get_op("tile.move").name
_TILE_EXTRACT_OP = core_ir.get_op("tile.extract").name
_TILE_TRANSPOSE_VIEW_OP = core_ir.get_op("tile.transpose_view").name
_TILE_MATMUL_OPS = frozenset(
    {
        core_ir.get_op("tile.matmul").name,
        core_ir.get_op("tile.matmul_acc").name,
        core_ir.get_op("tile.matmul_bias").name,
    }
)


_SSA_SUFFIX_RE = re.compile(r"__(?:ssa|rv|iter|idx)_v\d+")


def _clean_ssa_name(name_hint: str) -> str:
    """Strip compiler SSA suffixes (`__ssa_v0`, `__rv_v2`, `__iter_v1`) for canonical value names."""
    return _SSA_SUFFIX_RE.sub("", name_hint)


def _static_shape(shape_exprs: Sequence[core_ir.Expr]) -> tuple[int, ...] | None:
    dims: list[int] = []
    for expr in shape_exprs:
        if not isinstance(expr, core_ir.ConstInt):
            return None
        dims.append(int(expr.value))
    return tuple(dims)


def _tensor_shape_with_dynamic_fallback(
    shape_exprs: Sequence[core_ir.Expr], default_dyn: int = 128
) -> tuple[tuple[int, ...], bool]:
    """Return `(dims, is_fully_static)` substituting `default_dyn` for symbolic dimensions."""
    dims: list[int] = []
    is_static = True
    for expr in shape_exprs:
        if isinstance(expr, core_ir.ConstInt):
            dims.append(int(expr.value))
        else:
            dims.append(default_dyn)
            is_static = False
    return tuple(dims), is_static


def _effective_tile_layout(tile_type: core_ir.TileType) -> TileLayoutDesc:
    """Return the effective `TileLayoutDesc` for `tile_type` after `InferTileMemorySpace`."""
    space = tile_type.memory_space
    view = tile_type.tile_view
    if view is not None:
        if space in (core_ir.MemorySpace.Mat, core_ir.MemorySpace.Left) and (
            view.slayout == core_ir.TileLayout.none_box
        ):
            return FRACTAL_NZ
        if space == core_ir.MemorySpace.Right and view.slayout == core_ir.TileLayout.none_box:
            return FRACTAL_ZN
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


def _eval_scalar_expr(
    expr: object,
    scalar_defs: Mapping[int, core_ir.Expr] | None = None,
    default_dyn: int | None = 128,
) -> int | None:
    """Evaluate a scalar IR expression to an integer when possible (`default_dyn=None` for exact)."""
    if isinstance(expr, int):
        return expr if default_dyn is None else max(1, expr)
    if isinstance(expr, core_ir.ConstInt):
        return int(expr.value)
    if isinstance(expr, core_ir.Var):
        if scalar_defs is not None and expr.unique_id in scalar_defs:
            return _eval_scalar_expr(scalar_defs[expr.unique_id], scalar_defs, default_dyn)
        if default_dyn is not None:
            hint_upper = expr.name_hint.upper()
            if any(tok in hint_upper for tok in ("DYN", "TOKEN", "ROW", "BATCH", "SEQ")):
                return default_dyn
        return None
    if isinstance(expr, core_ir.Add):
        lhs = _eval_scalar_expr(expr.left, scalar_defs, default_dyn)
        rhs = _eval_scalar_expr(expr.right, scalar_defs, default_dyn)
        return (lhs + rhs) if (lhs is not None and rhs is not None) else None
    if isinstance(expr, core_ir.Sub):
        lhs = _eval_scalar_expr(expr.left, scalar_defs, default_dyn)
        rhs = _eval_scalar_expr(expr.right, scalar_defs, default_dyn)
        if lhs is None or rhs is None:
            return None
        return (lhs - rhs) if default_dyn is None else max(0, lhs - rhs)
    if isinstance(expr, core_ir.Mul):
        lhs = _eval_scalar_expr(expr.left, scalar_defs, default_dyn)
        rhs = _eval_scalar_expr(expr.right, scalar_defs, default_dyn)
        return (lhs * rhs) if (lhs is not None and rhs is not None) else None
    if isinstance(expr, core_ir.FloorDiv):
        lhs = _eval_scalar_expr(expr.left, scalar_defs, default_dyn)
        rhs = _eval_scalar_expr(expr.right, scalar_defs, default_dyn)
        return (lhs // rhs) if (lhs is not None and rhs is not None and rhs > 0) else None
    if isinstance(expr, core_ir.Min):
        lhs = _eval_scalar_expr(expr.left, scalar_defs, default_dyn)
        rhs = _eval_scalar_expr(expr.right, scalar_defs, default_dyn)
        return min(lhs, rhs) if (lhs is not None and rhs is not None) else None
    if isinstance(expr, core_ir.Max):
        lhs = _eval_scalar_expr(expr.left, scalar_defs, default_dyn)
        rhs = _eval_scalar_expr(expr.right, scalar_defs, default_dyn)
        return max(lhs, rhs) if (lhs is not None and rhs is not None) else None
    return None


def _for_trip_count(
    op: core_ir.ForStmt,
    scalar_defs: Mapping[int, core_ir.Expr] | None = None,
) -> int:
    """Return estimated trip count of *op* (supporting static and grid-stride bounds)."""
    step_val = _eval_scalar_expr(op.step, scalar_defs)
    if step_val is None or step_val <= 0:
        return 1
    stop_val = _eval_scalar_expr(op.stop, scalar_defs)
    if stop_val is None or stop_val <= 0:
        return 1
    start_val = _eval_scalar_expr(op.start, scalar_defs)
    if start_val is not None:
        span = max(0, stop_val - start_val)
        return max(1, (span + step_val - 1) // step_val)
    # Grid-stride loop `for i in range(worker_id, STOP, STEP)` where `0 <= worker_id < STEP`.
    if isinstance(op.start, core_ir.Var):
        return max(1, (stop_val + step_val - 1) // step_val)
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


def _is_whole_tensor_flatten(call: core_ir.Call) -> bool:
    """Mirror `IsWholeTensorFlatten` from `block_nz_tensor_views_pass.cpp:198`."""
    if call.op.name != _TENSOR_RESHAPE_OP or len(call.args) < 2:
        return False
    src_type = call.args[0].type
    if not isinstance(src_type, core_ir.TensorType):
        return False
    src_shape = _static_shape(src_type.shape)
    if src_shape is None:
        return False
    shape_tuple = call.args[1]
    if not isinstance(shape_tuple, core_ir.MakeTuple) or len(shape_tuple.elements) != 1:
        return False
    flat_ext = shape_tuple.elements[0]
    if not isinstance(flat_ext, core_ir.ConstInt):
        return False
    return math.prod(src_shape) == int(flat_ext.value)


def _is_provable_multiple_of(  # noqa: PLR0913
    expr: core_ir.Expr,
    divisor: int,
    scalar_defs: Mapping[int, core_ir.Expr],
    loop_ranges: Mapping[int, tuple[core_ir.Expr, core_ir.Expr]],
    param_steps_by_uid: Mapping[int, int],
    budget: list[int],
) -> bool:
    """Mirror `IsProvableMultipleOf` from `tensor_view_semantics.h:356`."""
    if divisor <= 0 or (divisor & (divisor - 1)) != 0 or budget[0] <= 0:
        return False
    budget[0] -= 1
    if isinstance(expr, core_ir.ConstInt):
        return int(expr.value) % divisor == 0
    if isinstance(expr, core_ir.Mul):
        return _is_provable_multiple_of(
            expr.left, divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
        ) or _is_provable_multiple_of(
            expr.right, divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
        )
    if isinstance(expr, (core_ir.Add, core_ir.Sub, core_ir.Min, core_ir.Max)):
        return _is_provable_multiple_of(
            expr.left, divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
        ) and _is_provable_multiple_of(
            expr.right, divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
        )
    if isinstance(expr, core_ir.Var) and not isinstance(expr, core_ir.IterArg):
        uid = expr.unique_id
        step = param_steps_by_uid.get(uid)
        if step is not None and step > 1 and step % 8 == 0 and step % divisor == 0:
            return True
        if uid in loop_ranges:
            start_expr, step_expr = loop_ranges[uid]
            if _is_provable_multiple_of(
                start_expr, divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
            ) and _is_provable_multiple_of(
                step_expr, divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
            ):
                return True
        if uid in scalar_defs:
            return _is_provable_multiple_of(
                scalar_defs[uid], divisor, scalar_defs, loop_ranges, param_steps_by_uid, budget
            )
    return False


def _is_provable_non_negative(
    expr: core_ir.Expr,
    scalar_defs: Mapping[int, core_ir.Expr],
    loop_ranges: Mapping[int, tuple[core_ir.Expr, core_ir.Expr]],
    non_neg_uids: set[int],
    budget: list[int],
) -> bool:
    """Mirror `IsProvableNonNegative` from `tensor_view_semantics.h:428` (`Sub` is always refused)."""
    if budget[0] <= 0:
        return False
    budget[0] -= 1
    if isinstance(expr, core_ir.ConstInt):
        return int(expr.value) >= 0
    if isinstance(expr, (core_ir.Mul, core_ir.Add, core_ir.Min)):
        return _is_provable_non_negative(
            expr.left, scalar_defs, loop_ranges, non_neg_uids, budget
        ) and _is_provable_non_negative(expr.right, scalar_defs, loop_ranges, non_neg_uids, budget)
    if isinstance(expr, core_ir.Max):
        return _is_provable_non_negative(
            expr.left, scalar_defs, loop_ranges, non_neg_uids, budget
        ) or _is_provable_non_negative(expr.right, scalar_defs, loop_ranges, non_neg_uids, budget)
    if isinstance(expr, (core_ir.FloorMod, core_ir.FloorDiv)):
        return (
            isinstance(expr.right, core_ir.ConstInt)
            and int(expr.right.value) > 0
            and _is_provable_non_negative(expr.left, scalar_defs, loop_ranges, non_neg_uids, budget)
        )
    if isinstance(expr, core_ir.Var) and not isinstance(expr, core_ir.IterArg):
        uid = expr.unique_id
        if uid in non_neg_uids:
            return True
        if uid in loop_ranges:
            start_expr, step_expr = loop_ranges[uid]
            if _is_provable_non_negative(
                start_expr, scalar_defs, loop_ranges, non_neg_uids, budget
            ) and _is_provable_non_negative(step_expr, scalar_defs, loop_ranges, non_neg_uids, budget):
                return True
        if uid in scalar_defs:
            return _is_provable_non_negative(scalar_defs[uid], scalar_defs, loop_ranges, non_neg_uids, budget)
    return False


def _is_nz_compatible_offset_axis(  # noqa: PLR0913
    offset_expr: core_ir.Expr,
    divisor: int,
    scalar_defs: Mapping[int, core_ir.Expr],
    loop_ranges: Mapping[int, tuple[core_ir.Expr, core_ir.Expr]],
    param_steps_by_uid: Mapping[int, int],
    non_neg_uids: set[int],
) -> bool:
    """Mirror `block_axis` in `BlockNzOffsets` (`tensor_view_semantics.h:564`)."""
    if isinstance(offset_expr, core_ir.ConstInt):
        val = int(offset_expr.value)
        return val >= 0 and val % divisor == 0
    return _is_provable_multiple_of(
        offset_expr, divisor, scalar_defs, loop_ranges, param_steps_by_uid, [256]
    ) and _is_provable_non_negative(offset_expr, scalar_defs, loop_ranges, non_neg_uids, [256])


def _is_nz_compatible_tensor_type(tensor_type: core_ir.Type) -> tuple[int, ...] | None:
    """Check `BlockNzType` preconditions (`block_nz_tensor_views_pass.cpp:167-185`)."""
    if not isinstance(tensor_type, core_ir.TensorType):
        return None
    view = tensor_type.tensor_view
    if view is not None and (view.valid_shape or view.stride):
        return None
    shape = _static_shape(tensor_type.shape)
    if shape is None or len(shape) not in (2, 3):
        return None
    if not is_nz_compatible_shape(shape, tensor_type.dtype):
        return None
    return shape


def _is_nz_compatible_tensor_slice_call(call: core_ir.Call, tensor_type: core_ir.Type) -> bool:
    """Mirror `BlockTensorSliceArgs` from `block_nz_tensor_views_pass.cpp:654-767`."""
    shape = _is_nz_compatible_tensor_type(tensor_type)
    if shape is None or len(call.args) < 3 or not isinstance(tensor_type, core_ir.TensorType):
        return False
    rank = len(shape)
    shapes_arg, offsets_arg = call.args[1], call.args[2]
    if not isinstance(shapes_arg, core_ir.MakeTuple) or not isinstance(offsets_arg, core_ir.MakeTuple):
        return False
    if len(shapes_arg.elements) != rank or len(offsets_arg.elements) != rank:
        return False
    shapes_static = _static_shape(shapes_arg.elements)
    if shapes_static is None or not is_nz_compatible_shape(shapes_static, tensor_type.dtype):
        return False
    row_off, col_off = offsets_arg.elements[-2], offsets_arg.elements[-1]
    if shapes_static[-2] != shape[-2] or shapes_static[-1] != shape[-1]:
        return False
    if not isinstance(row_off, core_ir.ConstInt) or int(row_off.value) != 0:
        return False
    if not isinstance(col_off, core_ir.ConstInt) or int(col_off.value) != 0:
        return False
    if len(call.args) >= 4:
        valid_arg = call.args[3]
        if isinstance(valid_arg, core_ir.MakeTuple) and valid_arg.elements:
            if len(valid_arg.elements) != rank or _static_shape(valid_arg.elements) != shapes_static:
                return False
    if len(call.args) >= 5:
        drop_arg = call.args[4]
        if not isinstance(drop_arg, core_ir.MakeTuple):
            return False
        for d_expr in drop_arg.elements:
            if (
                not isinstance(d_expr, core_ir.ConstInt)
                or int(d_expr.value) < 0
                or int(d_expr.value) + 2 >= rank
            ):
                return False
    return True


def _is_nz_compatible_valid_shape(  # noqa: PLR0913
    valid_arg: core_ir.Expr,
    rank: int,
    c0: int,
    scalar_defs: Mapping[int, core_ir.Expr],
    loop_ranges: Mapping[int, tuple[core_ir.Expr, core_ir.Expr]],
    param_steps_by_uid: Mapping[int, int],
) -> bool:
    """Mirror `BlockNzValidShape` from `tensor_view_semantics.h:506-538`."""
    if not isinstance(valid_arg, core_ir.MakeTuple) or len(valid_arg.elements) != rank:
        return False
    if any(not isinstance(el, core_ir.ConstInt) or int(el.value) <= 0 for el in valid_arg.elements[:-2]):
        return False
    v_cols = valid_arg.elements[-1]
    if not isinstance(v_cols, core_ir.ConstInt) or int(v_cols.value) <= 0 or int(v_cols.value) % c0 != 0:
        return False
    v_rows = valid_arg.elements[-2]
    if isinstance(v_rows, core_ir.ConstInt):
        return int(v_rows.value) > 0 and int(v_rows.value) % 16 == 0
    return _is_provable_multiple_of(v_rows, 16, scalar_defs, loop_ranges, param_steps_by_uid, [256])


def _is_nz_compatible_tile_load_call(  # noqa: PLR0913
    call: core_ir.Call,
    tensor_type: core_ir.Type,
    scalar_defs: Mapping[int, core_ir.Expr],
    loop_ranges: Mapping[int, tuple[core_ir.Expr, core_ir.Expr]],
    param_steps_by_uid: Mapping[int, int],
    non_neg_uids: set[int],
) -> bool:
    """Mirror `BlockTileLoadArgs` from `block_nz_tensor_views_pass.cpp:771-813`."""
    shape = _is_nz_compatible_tensor_type(tensor_type)
    if (
        shape is None
        or len(call.args) < 3
        or not isinstance(tensor_type, core_ir.TensorType)
        or call.kwargs.get("target_memory") != core_ir.MemorySpace.Mat
    ):
        return False
    rank = len(shape)
    offsets_arg, shapes_arg = call.args[1], call.args[2]
    if (
        not isinstance(offsets_arg, core_ir.MakeTuple)
        or not isinstance(shapes_arg, core_ir.MakeTuple)
        or len(offsets_arg.elements) != rank
        or len(shapes_arg.elements) != rank
    ):
        return False
    shapes_static = _static_shape(shapes_arg.elements)
    c0 = c0_elements(tensor_type.dtype)
    if shapes_static is None or c0 <= 0 or not is_nz_compatible_shape(shapes_static, tensor_type.dtype):
        return False
    if not (
        _is_nz_compatible_offset_axis(
            offsets_arg.elements[-2], 16, scalar_defs, loop_ranges, param_steps_by_uid, non_neg_uids
        )
        and _is_nz_compatible_offset_axis(
            offsets_arg.elements[-1], c0, scalar_defs, loop_ranges, param_steps_by_uid, non_neg_uids
        )
    ):
        return False
    part_tuple = shapes_arg
    if len(call.args) >= 4:
        if not _is_nz_compatible_valid_shape(
            call.args[3], rank, c0, scalar_defs, loop_ranges, param_steps_by_uid
        ):
            return False
        part_tuple = call.args[3]
        assert isinstance(part_tuple, core_ir.MakeTuple)
    col_blocks = (
        int(part_tuple.elements[-1].value) // c0
        if isinstance(part_tuple.elements[-1], core_ir.ConstInt)
        else 2
    )
    loaded_rows = (
        int(part_tuple.elements[-2].value) if isinstance(part_tuple.elements[-2], core_ir.ConstInt) else 0
    )
    return not (col_blocks > 1 and (shape[-2] - loaded_rows) > 65535)


def _is_nz_compatible_tile_store_call(  # noqa: PLR0913
    call: core_ir.Call,
    tensor_type: core_ir.Type,
    scalar_defs: Mapping[int, core_ir.Expr],
    loop_ranges: Mapping[int, tuple[core_ir.Expr, core_ir.Expr]],
    param_steps_by_uid: Mapping[int, int],
    non_neg_uids: set[int],
    model: LayoutCostModel,
) -> bool:
    """Mirror `BlockTileStoreArgs` from `block_nz_tensor_views_pass.cpp:528-638`."""
    shape = _is_nz_compatible_tensor_type(tensor_type)
    if shape is None or len(call.args) not in (3, 4) or not isinstance(tensor_type, core_ir.TensorType):
        return False
    if model.target_arch() == "a5":
        return False
    tile_type = call.args[0].type
    if not isinstance(tile_type, core_ir.TileType):
        return False
    if (
        tile_type.dtype != DataType.FP32
        or tensor_type.dtype != DataType.FP32
        or tile_type.memory_space != core_ir.MemorySpace.Acc
        or _effective_tile_layout(tile_type) != FRACTAL_ACC_NZ
    ):
        return False
    tile_shape = _static_shape(tile_type.shape)
    if tile_shape is None or len(tile_shape) != 2:
        return False
    view = tile_type.tile_view
    if view is not None and view.valid_shape and _static_shape(view.valid_shape) != tile_shape:
        return False
    if (
        tile_shape[0] <= 0
        or tile_shape[0] % 16 != 0
        or tile_shape[0] > 65535
        or tile_shape[1] <= 0
        or tile_shape[1] % 16 != 0
        or tile_shape[0] != shape[-2]
    ):
        return False
    if (
        int(call.kwargs.get("atomic", 0)) != 0
        or "pre_quant" in call.kwargs
        or bool(call.kwargs.get("pre_relu", False))
    ):
        return False
    rank = len(shape)
    offsets_arg = call.args[1]
    if not isinstance(offsets_arg, core_ir.MakeTuple) or len(offsets_arg.elements) != rank:
        return False
    if len(call.args) == 4:
        win_arg = call.args[3]
        expected_win = (*(1 for _ in shape[:-2]), tile_shape[0], tile_shape[1])
        if not isinstance(win_arg, core_ir.MakeTuple) or _static_shape(win_arg.elements) != expected_win:
            return False
    for lead_off in offsets_arg.elements[:-2]:
        if not _is_provable_non_negative(lead_off, scalar_defs, loop_ranges, non_neg_uids, [256]):
            return False
    row_off = offsets_arg.elements[-2]
    if not isinstance(row_off, core_ir.ConstInt) or int(row_off.value) != 0:
        return False
    return _is_nz_compatible_offset_axis(
        offsets_arg.elements[-1], 16, scalar_defs, loop_ranges, param_steps_by_uid, non_neg_uids
    )


def _collect_func_nz_ineligible(  # noqa: PLR0913
    func: core_ir.Function,
    model: LayoutCostModel,
    param_overrides: Sequence[str | None] | None,
    param_steps: Mapping[tuple[str, int], int],
) -> set[str]:
    """Return canonical GM tensor names used in `func` in a way `BlockNzTensorViews` rejects."""
    ineligible: set[str] = set()
    tensor_aliases: dict[int, str] = {}
    scalar_defs: dict[int, core_ir.Expr] = {}
    loop_ranges: dict[int, tuple[core_ir.Expr, core_ir.Expr]] = {}
    non_neg_uids: set[int] = set()
    param_steps_by_uid: dict[int, int] = {}

    overrides = list(param_overrides) if param_overrides is not None else []
    already_blocked = bool(func.attrs and "nz_tensor_views_blocked" in func.attrs)
    for idx, param in enumerate(func.params):
        name = (overrides[idx] if idx < len(overrides) and overrides[idx] else None) or _clean_ssa_name(
            param.name_hint
        )
        if isinstance(param.type, core_ir.TensorType):
            tensor_aliases[param.unique_id] = name
            if already_blocked or _is_nz_compatible_tensor_type(param.type) is None:
                ineligible.add(name)
        elif isinstance(param.type, core_ir.ScalarType):
            step = param_steps.get((func.name, idx))
            if step is not None and step > 1 and step % 8 == 0:
                param_steps_by_uid[param.unique_id] = step
                non_neg_uids.add(param.unique_id)

    def _collect_facts(stmt: core_ir.Stmt) -> None:
        if isinstance(stmt, core_ir.AssignStmt):
            if isinstance(stmt.var.type, core_ir.ScalarType):
                scalar_defs[stmt.var.unique_id] = stmt.value
                if isinstance(stmt.value, core_ir.Call) and stmt.value.op.name in (
                    "tile.get_block_idx",
                    "tile.get_block_num",
                ):
                    non_neg_uids.add(stmt.var.unique_id)
            elif isinstance(stmt.var.type, core_ir.TensorType):
                if isinstance(stmt.value, core_ir.Var) and stmt.value.unique_id in tensor_aliases:
                    tensor_aliases[stmt.var.unique_id] = tensor_aliases[stmt.value.unique_id]
                elif isinstance(stmt.value, core_ir.Call):
                    if stmt.value.op.name in (_TENSOR_CREATE_OP, _TENSOR_RESHAPE_OP):
                        b_name = _clean_ssa_name(stmt.var.name_hint)
                        tensor_aliases[stmt.var.unique_id] = b_name
                        if (
                            stmt.value.op.name == _TENSOR_RESHAPE_OP
                            or _is_nz_compatible_tensor_type(stmt.var.type) is None
                        ):
                            ineligible.add(b_name)
                    elif (
                        stmt.value.args
                        and isinstance(stmt.value.args[0], core_ir.Var)
                        and isinstance(stmt.value.args[0].type, core_ir.TensorType)
                    ):
                        src_v = stmt.value.args[0]
                        src_name = tensor_aliases.get(src_v.unique_id, _clean_ssa_name(src_v.name_hint))
                        tensor_aliases[stmt.var.unique_id] = src_name
        elif isinstance(stmt, core_ir.ForStmt):
            loop_ranges[stmt.loop_var.unique_id] = (stmt.start, stmt.step)
            for iter_arg in stmt.iter_args:
                init_val = getattr(iter_arg, "initValue", None)
                if isinstance(init_val, core_ir.Var) and init_val.unique_id in tensor_aliases:
                    tensor_aliases[iter_arg.unique_id] = tensor_aliases[init_val.unique_id]
            for rv, iter_arg in zip(stmt.return_vars, stmt.iter_args, strict=False):
                if iter_arg.unique_id in tensor_aliases:
                    tensor_aliases[rv.unique_id] = tensor_aliases[iter_arg.unique_id]
        for child in _iter_child_stmts(stmt):
            _collect_facts(child)

    _collect_facts(func.body)

    def _check_expr(expr: core_ir.Expr) -> None:
        if isinstance(expr, core_ir.Submit):
            for arg in expr.args:
                _check_expr(arg)
            return
        if not isinstance(expr, core_ir.Call):
            return
        for arg in expr.args:
            _check_expr(arg)
        if isinstance(expr.op, core_ir.GlobalVar):
            return
        op_name = expr.op.name
        for i, arg in enumerate(expr.args):
            if not isinstance(arg, core_ir.Var) or not isinstance(arg.type, core_ir.TensorType):
                continue
            t_name = tensor_aliases.get(arg.unique_id, _clean_ssa_name(arg.name_hint))
            ok = False
            if op_name == _TENSOR_CREATE_OP:
                ok = _is_nz_compatible_tensor_type(expr.type) is not None
            elif op_name == _TENSOR_RESHAPE_OP and i == 0:
                ok = _is_whole_tensor_flatten(expr)
            elif op_name == "tensor.slice" and i == 0:
                ok = _is_nz_compatible_tensor_slice_call(expr, arg.type)
            elif op_name == _TILE_LOAD_OP and i == 0:
                ok = _is_nz_compatible_tile_load_call(
                    expr, arg.type, scalar_defs, loop_ranges, param_steps_by_uid, non_neg_uids
                )
            elif op_name == _TILE_STORE_OP and i == 2:
                ok = _is_nz_compatible_tile_store_call(
                    expr, arg.type, scalar_defs, loop_ranges, param_steps_by_uid, non_neg_uids, model
                )
            if not ok:
                ineligible.add(t_name)
                ineligible.add(_clean_ssa_name(arg.name_hint))

    def _check_stmt(stmt: core_ir.Stmt) -> None:
        if isinstance(stmt, core_ir.AssignStmt):
            _check_expr(stmt.value)
        elif isinstance(stmt, core_ir.EvalStmt):
            _check_expr(stmt.expr)
        elif isinstance(stmt, core_ir.ReturnStmt):
            for val in stmt.value:
                _check_expr(val)
        for child in _iter_child_stmts(stmt):
            _check_stmt(child)

    _check_stmt(func.body)
    return ineligible


class _OrchCallMultiplierVisitor:
    """Count static invocation multiplicity and resolve caller GM aliases per InCore callee."""

    def __init__(self, func_by_name: dict[str, core_ir.Function] | None = None) -> None:
        self._func_by_name = func_by_name or {}
        self._mult = 1
        self._alias_by_uid: dict[int, str] = {}
        self._tuple_aliases: dict[int, list[str | None]] = {}
        self._scalar_defs: dict[int, core_ir.Expr] = {}
        self._var_steps: dict[int, int] = {}
        self._invalid_param_steps: set[tuple[str, int]] = set()
        self.call_counts: dict[str, int] = {}
        self.call_arg_names: dict[str, list[str | None]] = {}
        self.call_sites: dict[str, list[tuple[list[str | None], int]]] = {}
        self.call_order: dict[str, int] = {}
        self.param_steps: dict[tuple[str, int], int] = {}
        self.reshaped_names: set[str] = set()
        self.internal_buffers: set[str] = set()

    def walk(self, func: core_ir.Function) -> None:
        caller_args = self.call_arg_names.get(func.name, [])
        for idx, param in enumerate(func.params):
            override = caller_args[idx] if idx < len(caller_args) else None
            self._alias_by_uid[param.unique_id] = override or _clean_ssa_name(param.name_hint)
            step = self.param_steps.get((func.name, idx))
            if step is not None and step > 1:
                self._var_steps[param.unique_id] = step
        prev_mult = self._mult
        self._mult = self.call_counts.get(func.name, 1)
        try:
            self._walk_stmt(func.body)
        finally:
            self._mult = prev_mult

    def _arg_alignment_step(self, arg: core_ir.Expr) -> int | None:
        if isinstance(arg, core_ir.Var) and arg.unique_id in self._var_steps:
            return self._var_steps[arg.unique_id]
        c_val = _eval_scalar_expr(arg, self._scalar_defs, default_dyn=None)
        if c_val is not None and c_val >= 0:
            if c_val == 0:
                return 0
            if c_val % 8 == 0:
                return c_val
        return None

    def _resolve_arg_name(self, expr: core_ir.Expr) -> str | None:
        if not isinstance(expr, core_ir.Var):
            return None
        return self._alias_by_uid.get(expr.unique_id, _clean_ssa_name(expr.name_hint))

    def _call_core_multiplier(self, expr: core_ir.Call | core_ir.Submit) -> int:
        if isinstance(expr, core_ir.Submit):
            val = _eval_scalar_expr(expr.core_num, self._scalar_defs)
            return max(1, val) if val is not None else 1
        if expr.attrs and "core_num" in expr.attrs:
            val = _eval_scalar_expr(expr.attrs["core_num"], self._scalar_defs)
            return max(1, val) if val is not None else 1
        return 1

    def _record_global_call(
        self,
        expr: core_ir.Call | core_ir.Submit,
        name: str,
        assign_var: core_ir.Var | None,
    ) -> None:
        call_mult = self._mult * self._call_core_multiplier(expr)
        self.call_order.setdefault(name, len(self.call_order))
        self.call_counts[name] = self.call_counts.get(name, 0) + call_mult
        resolved = [self._resolve_arg_name(arg) for arg in expr.args]
        callee = self._func_by_name.get(name)
        if callee is not None and len(resolved) < len(callee.params):
            if assign_var is not None:
                base_hint = f"{_clean_ssa_name(assign_var.name_hint)}_u{assign_var.unique_id}"
            else:
                site_idx = len(self.call_sites.get(name, ()))
                base_hint = f"{name}_site{site_idx}"
            for p_idx in range(len(resolved), len(callee.params)):
                omitted_name = f"{base_hint}_out{p_idx}"
                resolved.append(omitted_name)
                self.reshaped_names.add(omitted_name)
        self.call_arg_names.setdefault(name, resolved)
        for p_idx, arg in enumerate(expr.args):
            key = (name, p_idx)
            if key in self._invalid_param_steps:
                continue
            step = self._arg_alignment_step(arg)
            if step is None:
                self._invalid_param_steps.add(key)
                self.param_steps.pop(key, None)
            else:
                prev_step = self.param_steps.get(key)
                self.param_steps[key] = math.gcd(prev_step, step) if prev_step is not None else step
        sites = self.call_sites.setdefault(name, [])
        for idx, (prev_args, prev_mult) in enumerate(sites):
            if prev_args == resolved:
                sites[idx] = (prev_args, prev_mult + call_mult)
                break
        else:
            sites.append((resolved, call_mult))
        if assign_var is not None and callee is not None:
            out_names = [
                arg_name
                for arg_name, direction in zip(resolved, callee.param_directions, strict=False)
                if direction in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut)
            ]
            self._tuple_aliases[assign_var.unique_id] = out_names
            if len(out_names) == 1 and out_names[0] is not None:
                self._alias_by_uid[assign_var.unique_id] = out_names[0]

    def _record_local_tensor_call(
        self,
        expr: core_ir.Call,
        assign_var: core_ir.Var | None,
    ) -> None:
        if expr.op.name == _TENSOR_CREATE_OP and assign_var is not None:
            buf_name = _clean_ssa_name(assign_var.name_hint)
            self._alias_by_uid[assign_var.unique_id] = buf_name
            target_set = (
                self.internal_buffers
                if _is_nz_compatible_tensor_type(assign_var.type) is not None
                else self.reshaped_names
            )
            target_set.add(buf_name)
            return
        for i, arg in enumerate(expr.args):
            if not isinstance(arg, core_ir.Var) or not isinstance(arg.type, core_ir.TensorType):
                continue
            src_alias = self._resolve_arg_name(arg) or _clean_ssa_name(arg.name_hint)
            if (
                i == 0
                and assign_var is not None
                and isinstance(assign_var.type, core_ir.TensorType)
                and expr.op.name != _TENSOR_RESHAPE_OP
            ):
                self._alias_by_uid[assign_var.unique_id] = src_alias
            ok = (expr.op.name == _TENSOR_RESHAPE_OP and i == 0 and _is_whole_tensor_flatten(expr)) or (
                expr.op.name == "tensor.slice"
                and i == 0
                and _is_nz_compatible_tensor_slice_call(expr, arg.type)
            )
            if expr.op.name == _TENSOR_RESHAPE_OP and assign_var is not None:
                self.reshaped_names.add(_clean_ssa_name(assign_var.name_hint))
            if not ok:
                self.reshaped_names.add(src_alias)
                if assign_var is not None:
                    self.reshaped_names.add(_clean_ssa_name(assign_var.name_hint))

    def _record_calls(self, expr: core_ir.Expr, assign_var: core_ir.Var | None = None) -> None:
        if assign_var is not None and isinstance(assign_var.type, core_ir.ScalarType):
            self._scalar_defs[assign_var.unique_id] = expr

        if isinstance(expr, (core_ir.Call, core_ir.Submit)):
            if isinstance(expr.op, core_ir.GlobalVar):
                self._record_global_call(expr, expr.op.name, assign_var)
            elif isinstance(expr, core_ir.Call):
                self._record_local_tensor_call(expr, assign_var)
            for arg in expr.args:
                self._record_calls(arg)
        elif assign_var is not None and isinstance(expr, core_ir.TupleGetItemExpr):
            if isinstance(expr.tuple, core_ir.Var):
                out_list = self._tuple_aliases.get(expr.tuple.unique_id)
                idx = int(expr.index)
                if out_list is not None and 0 <= idx < len(out_list) and out_list[idx] is not None:
                    self._alias_by_uid[assign_var.unique_id] = out_list[idx] or ""
        elif assign_var is not None and isinstance(expr, core_ir.Var):
            alias = self._resolve_arg_name(expr)
            if alias is not None:
                self._alias_by_uid[assign_var.unique_id] = alias

    def _walk_stmt(self, stmt: core_ir.Stmt) -> None:
        if isinstance(stmt, core_ir.ForStmt):
            start_val = _eval_scalar_expr(stmt.start, self._scalar_defs, default_dyn=None)
            step_val = _eval_scalar_expr(stmt.step, self._scalar_defs, default_dyn=None)
            if (
                start_val is not None
                and start_val >= 0
                and step_val is not None
                and step_val > 0
                and step_val % 8 == 0
                and start_val % step_val == 0
            ):
                self._var_steps[stmt.loop_var.unique_id] = step_val
            for iter_arg in stmt.iter_args:
                init_val = getattr(iter_arg, "initValue", None)
                if isinstance(init_val, core_ir.Expr):
                    alias = self._resolve_arg_name(init_val)
                    if alias is not None:
                        self._alias_by_uid[iter_arg.unique_id] = alias
            prev = self._mult
            self._mult = prev * _for_trip_count(stmt, self._scalar_defs)
            try:
                self._walk_stmt(stmt.body)
            finally:
                self._mult = prev
            for rv, iter_arg in zip(stmt.return_vars, stmt.iter_args, strict=False):
                if iter_arg.unique_id in self._alias_by_uid:
                    self._alias_by_uid[rv.unique_id] = self._alias_by_uid[iter_arg.unique_id]
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

    def __init__(  # noqa: PLR0913
        self,
        cost_model: LayoutCostModel,
        reusable_tensors: set[str],
        *,
        external_out_names: set[str] | None = None,
        reshaped_names: set[str] | None = None,
        internal_buffers: set[str] | None = None,
        shared_values: dict[str, LayoutValueNode] | None = None,
        consumed_values: set[str] | None = None,
        op_offset: int = 0,
        tile_prefix: str = "",
        call_multiplier: int = 1,
        param_name_overrides: Sequence[str | None] | None = None,
    ) -> None:
        self._model = cost_model
        self._reusable = reusable_tensors
        self._external_out_names = external_out_names
        self._reshaped_names = reshaped_names or set()
        self._internal_buffers = internal_buffers or set()
        self._tile_prefix = tile_prefix
        self.values: dict[str, LayoutValueNode] = dict(shared_values) if shared_values else {}
        self.consumed_values: set[str] = set(consumed_values) if consumed_values else set()
        self.operations: list[LayoutOperationNode] = []
        self._uid_to_name: dict[int, str] = {}
        self._scalar_defs: dict[int, core_ir.Expr] = {}
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
            shape, is_static = _tensor_shape_with_dynamic_fallback(param.type.shape)
            if not shape:
                continue
            override = self._param_overrides[idx] if idx < len(self._param_overrides) else None
            name = override or _clean_ssa_name(param.name_hint)
            self._uid_to_name[param.unique_id] = name
            view = param.type.tensor_view
            cur_layout = view.layout if view is not None else core_ir.TensorLayout.ND
            if name in self.values:
                prev_node = self.values[name]
                if (
                    cur_layout == core_ir.TensorLayout.NZ
                    and prev_node.current_layout != core_ir.TensorLayout.NZ
                ):
                    self.values[name] = replace(
                        prev_node,
                        shape=shape,
                        current_layout=cur_layout,
                        admissible_layouts=(cur_layout,),
                    )
                continue
            if self._external_out_names is not None:
                is_ext_out = name in self._external_out_names
            else:
                is_ext_out = direction in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut)
            is_reusable = name in self._reusable or param.name_hint in self._reusable
            allow_nz = (
                is_static
                and cur_layout == core_ir.TensorLayout.ND
                and name not in self._reshaped_names
                and is_nz_compatible_shape(shape, param.type.dtype)
                and (
                    self._model.allow_external_input_relayout
                    or is_reusable
                    or is_ext_out
                    or name in self._internal_buffers
                )
            )
            admissible: tuple[LayoutChoice, ...]
            if allow_nz:
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
                is_reusable_weight=is_reusable,
                is_external_output=is_ext_out,
            )
        self._walk_stmt(func.body)

    def _register_tile_var(
        self,
        var: core_ir.Var,
        producer_op_id: str | None,
        *,
        op_inputs: Sequence[str] = (),
    ) -> LayoutValueNode | None:
        if not isinstance(var.type, core_ir.TileType):
            return None
        shape = _static_shape(var.type.shape)
        if shape is None:
            return None
        for inp in op_inputs:
            self.consumed_values.add(inp)
        base_name = _clean_ssa_name(var.name_hint)
        name = f"{self._tile_prefix}{base_name}" if self._tile_prefix else base_name
        if name in self.values and (
            self.values[name].producer_op_id is not None
            or producer_op_id is None
            or name in self.consumed_values
        ):
            raw_name = f"{self._tile_prefix}{var.name_hint}" if self._tile_prefix else var.name_hint
            name = (
                raw_name
                if (raw_name not in self.values and raw_name not in self.consumed_values)
                else f"{raw_name}_{self._op_counter}"
            )
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
        dst_node = self._register_tile_var(op.var, op_id, op_inputs=(src_name,))
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
        dst_node = self._register_tile_var(op.var, op_id, op_inputs=(src_name,))
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
        dst_node = self._register_tile_var(op.var, op_id, op_inputs=(src_name,))
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
        dst_node = self._register_tile_var(op.var, op_id, op_inputs=(lhs_name, rhs_name))
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
        self.consumed_values.add(tile_name)
        op_id = f"op_{self._op_counter}_{dst_name}_store"
        self._op_counter += 1
        is_additional_store = dst_node.producer_op_id is not None or dst_name in self.consumed_values
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
                        input_layouts=(s_lay, d_lay) if is_additional_store else (s_lay,),
                        output_layout=s_lay if is_additional_store else d_lay,
                        op_cost_cycles=cost,
                        notes=f"FIXPIPE {src_node.memory_space.name}->GM({dst_label})",
                    )
                )
        supported_out_layouts = tuple(
            d_lay
            for d_lay in dst_node.admissible_layouts
            if any((c.input_layouts[1] if is_additional_store else c.output_layout) == d_lay for c in configs)
        )
        if is_additional_store:
            self.consumed_values.add(dst_name)
            sink_name = f"{tile_name}_stored_{self._op_counter}"
            self.values[sink_name] = LayoutValueNode(
                name=sink_name,
                shape=src_node.shape,
                dtype=src_node.dtype,
                memory_space=src_node.memory_space,
                current_layout=src_node.current_layout,
                admissible_layouts=src_node.admissible_layouts,
                producer_op_id=op_id,
            )
            if supported_out_layouts and not self._model.allow_external_output_relayout:
                self.values[dst_name] = replace(dst_node, admissible_layouts=supported_out_layouts)
            self.operations.append(
                LayoutOperationNode(
                    op_id=op_id,
                    op_name=rhs.op.name,
                    inputs=(tile_name, dst_name),
                    output=sink_name,
                    invocations=self._model.expected_uses * self._loop_multiplier,
                    configs=tuple(configs),
                )
            )
            return
        # Mark `dst_node` as produced by this store and restrict its admissible layouts
        # to those actually supported by the store when external output relayout is off.
        self.values[dst_name] = LayoutValueNode(
            name=dst_node.name,
            shape=dst_node.shape,
            dtype=dst_node.dtype,
            memory_space=dst_node.memory_space,
            current_layout=dst_node.current_layout,
            admissible_layouts=(
                supported_out_layouts
                if (supported_out_layouts and not self._model.allow_external_output_relayout)
                else dst_node.admissible_layouts
            ),
            producer_op_id=op_id,
            is_reusable_weight=dst_node.is_reusable_weight,
            is_external_output=dst_node.is_external_output,
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
        dst_node = self._register_tile_var(op.var, op_id, op_inputs=tuple(name for _, name, _ in tile_inputs))
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
        if isinstance(op.var.type, core_ir.ScalarType):
            self._scalar_defs[op.var.unique_id] = op.value
        rhs = op.value
        if isinstance(rhs, core_ir.Call):
            op_name = rhs.op.name
            if op_name == _TILE_CREATE_OP:
                self._register_tile_var(op.var, producer_op_id=None)
            elif op_name == _TILE_LOAD_OP:
                self._extract_load(op, rhs)
            elif op_name == _TILE_TRANSPOSE_VIEW_OP:
                self._extract_transpose_view(op, rhs)
            elif op_name == _TILE_MOVE_OP or (
                op_name == _TILE_EXTRACT_OP
                and isinstance(op.var.type, core_ir.TileType)
                and op.var.type.memory_space in (core_ir.MemorySpace.Left, core_ir.MemorySpace.Right)
            ):
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
        self._loop_multiplier = prev_mult * _for_trip_count(op, self._scalar_defs)
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
    consumed_values: set[str] = set()
    operations: list[LayoutOperationNode] = []

    incore_funcs = [f for f in program.functions.values() if core_ir.is_incore_type(f.func_type)]
    non_incore_funcs = [f for f in program.functions.values() if not core_ir.is_incore_type(f.func_type)]
    ext_out_names: set[str] | None = None
    reshaped_names: set[str] = set()
    internal_buffers: set[str] = set()
    call_counts: dict[str, int] = {}
    call_arg_names: dict[str, list[str | None]] = {}
    call_sites: dict[str, list[tuple[list[str | None], int]]] = {}
    param_steps: dict[tuple[str, int], int] = {}
    if non_incore_funcs:
        top_orch = [f for f in non_incore_funcs if str(f.func_type).endswith("Orchestration")]
        sub_orch = [f for f in non_incore_funcs if not str(f.func_type).endswith("Orchestration")]
        entry_orch = top_orch if top_orch else non_incore_funcs
        ext_out_names = {
            _clean_ssa_name(p.name_hint)
            for fn in entry_orch
            for p, d in zip(fn.params, fn.param_directions, strict=True)
            if d in (core_ir.ParamDirection.Out, core_ir.ParamDirection.InOut)
        }
        func_by_name = {f.name: f for f in program.functions.values()}
        call_counter = _OrchCallMultiplierVisitor(func_by_name)
        for fn in top_orch:
            call_counter.walk(fn)
        sub_orch.sort(key=lambda fn: call_counter.call_order.get(fn.name, 10**9))
        for fn in sub_orch:
            call_counter.walk(fn)
        call_counts = call_counter.call_counts
        call_arg_names = call_counter.call_arg_names
        call_sites = call_counter.call_sites
        param_steps = call_counter.param_steps
        reshaped_names = call_counter.reshaped_names
        internal_buffers = call_counter.internal_buffers
        incore_funcs.sort(key=lambda fn: call_counter.call_order.get(fn.name, 10**9))

    for func in incore_funcs:
        sites = call_sites.get(func.name)
        if sites:
            for site_args, _ in sites:
                reshaped_names.update(_collect_func_nz_ineligible(func, model, site_args, param_steps))
        else:
            reshaped_names.update(
                _collect_func_nz_ineligible(func, model, call_arg_names.get(func.name), param_steps)
            )

    multi_incore = len(incore_funcs) > 1
    op_offset = 0
    for func in incore_funcs:
        sites = call_sites.get(func.name)
        if sites and len(sites) > 1:
            for site_idx, (site_args, site_mult) in enumerate(sites):
                prefix = f"{func.name}_{site_idx}."
                extractor = _PostInferInstanceExtractor(
                    model,
                    reusable,
                    external_out_names=ext_out_names,
                    reshaped_names=reshaped_names,
                    internal_buffers=internal_buffers,
                    shared_values=values,
                    consumed_values=consumed_values,
                    op_offset=op_offset,
                    tile_prefix=prefix,
                    call_multiplier=site_mult,
                    param_name_overrides=site_args,
                )
                extractor.extract(func)
                values = extractor.values
                consumed_values = extractor.consumed_values
                operations.extend(extractor.operations)
                op_offset = extractor.op_counter
            continue

        prefix = f"{func.name}." if multi_incore else ""
        extractor = _PostInferInstanceExtractor(
            model,
            reusable,
            external_out_names=ext_out_names,
            reshaped_names=reshaped_names,
            internal_buffers=internal_buffers,
            shared_values=values,
            consumed_values=consumed_values,
            op_offset=op_offset,
            tile_prefix=prefix,
            call_multiplier=call_counts.get(func.name, 1),
            param_name_overrides=call_arg_names.get(func.name),
        )
        extractor.extract(func)
        values = extractor.values
        consumed_values = extractor.consumed_values
        operations.extend(extractor.operations)
        op_offset = extractor.op_counter

    _prune_incompatible_gm_nz(values, operations, reshaped_names)

    filtered_operations: list[LayoutOperationNode] = []
    for op in operations:
        valid_cfgs = tuple(
            cfg
            for cfg in op.configs
            if cfg.output_layout in values[op.output].admissible_layouts
            and all(
                lay in values[inp].admissible_layouts
                for inp, lay in zip(op.inputs, cfg.input_layouts, strict=True)
            )
        )
        filtered_operations.append(replace(op, configs=valid_cfgs if valid_cfgs else op.configs))

    return LayoutProblemInstance(values=values, operations=filtered_operations, cost_model=model)


def _prune_incompatible_gm_nz(
    values: dict[str, LayoutValueNode],
    operations: Sequence[LayoutOperationNode],
    reshaped_names: set[str],
) -> None:
    """Prune `TensorLayout.NZ` from any GM tensor lacking an `NZ`-compatible config on all ops."""
    for val_name, val_node in list(values.items()):
        if (
            val_node.memory_space != core_ir.MemorySpace.DDR
            or len(val_node.admissible_layouts) <= 1
            or core_ir.TensorLayout.NZ not in val_node.admissible_layouts
        ):
            continue
        nz_supported = val_name not in reshaped_names
        for op in operations:
            if not nz_supported:
                break
            for in_idx, inp_name in enumerate(op.inputs):
                if inp_name == val_name and not any(
                    cfg.input_layouts[in_idx] == core_ir.TensorLayout.NZ for cfg in op.configs
                ):
                    nz_supported = False
                    break
            if (
                nz_supported
                and op.output == val_name
                and not any(cfg.output_layout == core_ir.TensorLayout.NZ for cfg in op.configs)
            ):
                nz_supported = False
        if not nz_supported:
            values[val_name] = replace(val_node, admissible_layouts=(val_node.current_layout,))


def _operator_cost_multiplier(instance: LayoutProblemInstance, op: LayoutOperationNode) -> float:
    """Return the execution weight, including the model's MTE2 L2 reread discount."""
    load_scale = 1.0
    if op.op_name == _TILE_LOAD_OP and op.inputs:
        src_val = instance.values[op.inputs[0]]
        dst_val = instance.values[op.output]
        if dst_val.memory_space == core_ir.MemorySpace.Mat:
            model = instance.cost_model
            per_run_inv = float(op.invocations) / float(max(1, model.expected_uses))
            load_scale = model.effective_mte2_load_multiplier(src_val, dst_val, per_run_inv)
    return float(op.invocations) * load_scale


def _conversion_multiplier(
    instance: LayoutProblemInstance,
    value: LayoutValueNode,
    operations_by_id: Mapping[str, LayoutOperationNode],
    conversion_invocations: Sequence[int],
) -> float:
    """Count dynamic values, while sharing conversions only within one value."""
    if value.memory_space == core_ir.MemorySpace.DDR and value.is_reusable_weight:
        return 1.0
    if value.memory_space != core_ir.MemorySpace.DDR and value.producer_op_id is not None:
        producer_invocations = operations_by_id[value.producer_op_id].invocations
        if any(
            count < producer_invocations or count % producer_invocations != 0
            for count in conversion_invocations
        ):
            raise ValueError(
                f"On-chip conversion for {value.name!r} has differing producer/consumer execution counts; "
                f"producer={producer_invocations}, consumers={sorted(set(conversion_invocations))}. "
                "Execution-domain-aware conversion costing is required"
            )
        return float(producer_invocations)
    return float(instance.cost_model.expected_uses)


def _resolved_value_layouts(
    instance: LayoutProblemInstance,
    assignment: LayoutAssignment,
    produced_layout: Mapping[str, LayoutChoice],
) -> dict[str, LayoutChoice]:
    """Keep caller packing choices, but never overwrite a scored producer layout."""
    resolved = dict(assignment.value_layouts)
    for name, layout in produced_layout.items():
        if instance.values[name].producer_op_id is not None:
            supplied = assignment.value_layouts.get(name, layout)
            if supplied != layout:
                raise ValueError(f"Value layout for {name!r} contradicts its producer configuration")
            resolved[name] = layout
        else:
            resolved.setdefault(name, layout)
    return resolved


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
    consumer_invocations: dict[str, dict[LayoutChoice, set[int]]] = {name: {} for name in instance.values}
    resolved_configs: dict[str, OperatorConfig] = {}
    operations_by_id = {op.op_id: op for op in instance.operations}

    for op in instance.operations:
        chosen_cfg = assignment.op_configs.get(op.op_id)
        if chosen_cfg is not None and chosen_cfg not in op.configs:
            raise ValueError(f"Configuration for {op.op_id!r} is not in its legal catalog")
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
        cost = _operator_cost_multiplier(instance, op) * chosen_cfg.op_cost_cycles
        per_op_costs[op.op_id] = cost
        op_cost_total += cost
        produced_layout[op.output] = chosen_cfg.output_layout

        for inp_name, req_layout in zip(op.inputs, chosen_cfg.input_layouts, strict=True):
            consumer_layouts.setdefault(inp_name, set()).add(req_layout)
            consumer_invocations[inp_name].setdefault(req_layout, set()).add(op.invocations)

    # External output tensors must be delivered in their caller-expected layout
    # unless `allow_external_output_relayout` is enabled.
    if not model.allow_external_output_relayout:
        for val_name, val_node in instance.values.items():
            if val_node.is_external_output:
                consumer_layouts.setdefault(val_name, set()).add(val_node.current_layout)
                consumer_invocations[val_name].setdefault(val_node.current_layout, set()).add(
                    model.expected_uses
                )

    per_value_conv: dict[str, float] = {}
    conv_cost_total = 0.0
    for val_name, req_set in consumer_layouts.items():
        if not req_set or val_name not in instance.values:
            continue
        val_node = instance.values[val_name]
        src_lay = produced_layout.get(val_name, val_node.current_layout)
        conversion_invocations = {
            count
            for target in req_set
            if target != src_lay
            for count in consumer_invocations[val_name][target]
        }
        multiplier = _conversion_multiplier(
            instance, val_node, operations_by_id, tuple(conversion_invocations)
        )
        val_conv = 0.0
        for dst_lay in req_set:
            if dst_lay != src_lay:
                val_conv += multiplier * model.estimate_conversion_cost(val_node, src_lay, dst_lay)
        per_value_conv[val_name] = val_conv
        conv_cost_total += val_conv

    resolved_value_layouts = _resolved_value_layouts(instance, assignment, produced_layout)
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


def _collect_expr_var_uids(expr: core_ir.Expr, out: set[int]) -> None:
    if isinstance(expr, core_ir.Var):
        out.add(expr.unique_id)
        return
    if isinstance(expr, core_ir.MakeTuple):
        for el in expr.elements:
            _collect_expr_var_uids(el, out)
        return
    if isinstance(expr, core_ir.Cast):
        _collect_expr_var_uids(expr.operand, out)
        return
    for attr in ("left", "right"):
        child = getattr(expr, attr, None)
        if isinstance(child, core_ir.Expr):
            _collect_expr_var_uids(child, out)


def _nz_offset_scalar_uids(
    body: core_ir.Stmt,
    new_params: Sequence[tuple[core_ir.Var, core_ir.ParamDirection]],
    old_params: Sequence[core_ir.Var],
) -> set[int]:
    """Return scalar parameter `unique_id`s transitively used in NZ load/store offsets in `body`."""
    nz_tensor_uids = {
        old_p.unique_id
        for (new_p, _), old_p in zip(new_params, old_params, strict=True)
        if isinstance(new_p.type, core_ir.TensorType)
        and _tensor_type_layout(new_p.type) == core_ir.TensorLayout.NZ
    }
    if not nz_tensor_uids:
        return set()
    scalar_defs: dict[int, core_ir.Expr] = {}
    loop_bounds: dict[int, tuple[core_ir.Expr, core_ir.Expr]] = {}
    direct_uids: set[int] = set()

    def _walk_expr(expr: core_ir.Expr) -> None:
        if not isinstance(expr, core_ir.Call):
            return
        for arg in expr.args:
            _walk_expr(arg)
        if (
            expr.op.name == _TILE_LOAD_OP
            and len(expr.args) >= 2
            and isinstance(expr.args[0], core_ir.Var)
            and expr.args[0].unique_id in nz_tensor_uids
        ):
            _collect_expr_var_uids(expr.args[1], direct_uids)
            if len(expr.args) >= 4:
                _collect_expr_var_uids(expr.args[3], direct_uids)
        elif (
            expr.op.name == _TILE_STORE_OP
            and len(expr.args) >= 3
            and isinstance(expr.args[2], core_ir.Var)
            and expr.args[2].unique_id in nz_tensor_uids
        ):
            _collect_expr_var_uids(expr.args[1], direct_uids)

    def _walk_stmt(stmt: core_ir.Stmt) -> None:
        if isinstance(stmt, core_ir.AssignStmt):
            if isinstance(stmt.var.type, core_ir.ScalarType):
                scalar_defs[stmt.var.unique_id] = stmt.value
            elif (
                isinstance(stmt.var.type, core_ir.TensorType)
                and isinstance(stmt.value, core_ir.Var)
                and stmt.value.unique_id in nz_tensor_uids
            ):
                nz_tensor_uids.add(stmt.var.unique_id)
            _walk_expr(stmt.value)
        elif isinstance(stmt, core_ir.ForStmt):
            loop_bounds[stmt.loop_var.unique_id] = (stmt.start, stmt.step)
        elif isinstance(stmt, core_ir.EvalStmt):
            _walk_expr(stmt.expr)
        elif isinstance(stmt, core_ir.ReturnStmt):
            for val in stmt.value:
                _walk_expr(val)
        for child in _iter_child_stmts(stmt):
            _walk_stmt(child)

    _walk_stmt(body)
    expanded = set(direct_uids)
    worklist = list(direct_uids)
    while worklist:
        uid = worklist.pop()
        if uid in scalar_defs:
            sub: set[int] = set()
            _collect_expr_var_uids(scalar_defs[uid], sub)
            for s_uid in sub - expanded:
                expanded.add(s_uid)
                worklist.append(s_uid)
        if uid in loop_bounds:
            start_e, step_e = loop_bounds[uid]
            sub_l: set[int] = set()
            _collect_expr_var_uids(start_e, sub_l)
            _collect_expr_var_uids(step_e, sub_l)
            for s_uid in sub_l - expanded:
                expanded.add(s_uid)
                worklist.append(s_uid)
    return expanded


class _LayoutAssignmentWriter(core_ir.IRMutator):
    """Write a `LayoutAssignment` (both GM `TensorLayout` and on-chip `TileLayoutDesc`) back to IR."""

    def __init__(self, assignment: LayoutAssignment) -> None:
        super().__init__()
        self._tensor_layouts = assignment.tensor_layouts
        self._tile_layouts = assignment.tile_layouts
        self._scoped_tensor_layouts: dict[tuple[str, str], core_ir.TensorLayout] = {}
        self._tile_prefix_override: dict[str, str] = {}
        self._specialized_callees: dict[tuple[str, tuple[core_ir.TensorLayout | None, ...]], str] = {}
        self._callee_formal_params: dict[str, tuple[core_ir.Var, ...]] = {}
        self._param_steps: dict[tuple[str, int], int] = {}
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

    def _site_param_signature(
        self,
        callee: core_ir.Function,
        site_args: Sequence[str | None],
    ) -> tuple[core_ir.TensorLayout | None, ...]:
        sig: list[core_ir.TensorLayout | None] = []
        for p_idx, param in enumerate(callee.params):
            if not isinstance(param.type, core_ir.TensorType):
                sig.append(None)
                continue
            arg_name = site_args[p_idx] if p_idx < len(site_args) else None
            default_lay = _tensor_type_layout(param.type)
            sig.append(
                self._tensor_layouts.get(arg_name, default_lay) if arg_name is not None else default_lay
            )
        return tuple(sig)

    def _register_callee_specializations(
        self,
        func_by_name: dict[str, core_ir.Function],
        visitor: _OrchCallMultiplierVisitor,
    ) -> list[core_ir.Function]:
        extra_incore: list[core_ir.Function] = []
        used_names: set[str] = set(func_by_name.keys())
        for callee_name, sites in visitor.call_sites.items():
            callee = func_by_name.get(callee_name)
            if callee is None or not sites:
                continue
            self._callee_formal_params[callee_name] = tuple(callee.params)
            if len(sites) == 1:
                site_args, _ = sites[0]
                for p_idx, param in enumerate(callee.params):
                    arg_name = site_args[p_idx] if p_idx < len(site_args) else None
                    lay = self._tensor_layouts.get(arg_name or "")
                    if lay is not None:
                        self._scoped_tensor_layouts[(callee_name, _clean_ssa_name(param.name_hint))] = lay
                continue
            for site_idx, (site_args, _) in enumerate(sites):
                sig_t = self._site_param_signature(callee, site_args)
                if site_idx == 0:
                    spec_name = callee_name
                    self._tile_prefix_override[spec_name] = f"{callee_name}_0"
                    self._specialized_callees[(callee_name, sig_t)] = spec_name
                elif (callee_name, sig_t) in self._specialized_callees:
                    continue
                else:
                    suffix = site_idx
                    spec_name = f"{callee_name}_site{suffix}"
                    while spec_name in used_names:
                        suffix += 1
                        spec_name = f"{callee_name}_site{suffix}"
                    used_names.add(spec_name)
                    self._tile_prefix_override[spec_name] = f"{callee_name}_{site_idx}"
                    self._specialized_callees[(callee_name, sig_t)] = spec_name
                    for p_idx in range(len(callee.params)):
                        if (callee_name, p_idx) in self._param_steps:
                            self._param_steps[(spec_name, p_idx)] = self._param_steps[(callee_name, p_idx)]
                    extra_incore.append(
                        core_ir.Function(
                            spec_name,
                            list(zip(callee.params, callee.param_directions, strict=True)),
                            list(callee.return_types),
                            callee.body,
                            callee.span,
                            type=callee.func_type,
                            level=callee.level,
                            role=callee.role,
                            attrs=dict(callee.attrs) if callee.attrs else None,
                            requires_runtime_binding=callee.requires_runtime_binding,
                            ir_stage=callee.ir_stage,
                        )
                    )
                for p_idx, param in enumerate(callee.params):
                    lay = sig_t[p_idx]
                    if lay is not None:
                        self._scoped_tensor_layouts[(spec_name, _clean_ssa_name(param.name_hint))] = lay
        return extra_incore

    def rewrite(self, program: core_ir.Program) -> core_ir.Program:
        funcs = list(program.functions.values())
        incore_funcs = [f for f in funcs if core_ir.is_incore_type(f.func_type)]
        other_funcs = [f for f in funcs if not core_ir.is_incore_type(f.func_type)]
        top_orch = [f for f in other_funcs if str(f.func_type).endswith("Orchestration")]
        sub_orch = [f for f in other_funcs if not str(f.func_type).endswith("Orchestration")]
        func_by_name = {f.name: f for f in funcs}
        extra_incore: list[core_ir.Function] = []
        if other_funcs:
            visitor = _OrchCallMultiplierVisitor(func_by_name)
            for fn in top_orch:
                visitor.walk(fn)
            sub_orch.sort(key=lambda fn: visitor.call_order.get(fn.name, 10**9))
            for fn in sub_orch:
                visitor.walk(fn)
            self._param_steps = visitor.param_steps
            extra_incore = self._register_callee_specializations(func_by_name, visitor)

        rewritten_by_name: dict[str, core_ir.Function] = {}
        for func in incore_funcs + extra_incore + sub_orch + top_orch:
            rewritten_by_name[func.name] = self.visit_function(func)
        used_extra = [
            rewritten_by_name[ef.name] for ef in extra_incore if ef.name in self._specialized_callees.values()
        ]
        new_funcs = (
            [rewritten_by_name[f.name] for f in incore_funcs]
            + used_extra
            + [rewritten_by_name[f.name] for f in other_funcs]
        )
        if len(new_funcs) == len(funcs) and all(
            nf.same_as(of) for nf, of in zip(new_funcs, funcs, strict=True)
        ):
            return program
        return core_ir.Program(new_funcs, program.name, program.span)

    def _lookup_tile_desc(self, name_hint: str) -> TileLayoutDesc | None:
        clean_name = _clean_ssa_name(name_hint)
        if self._current_func_name:
            ov = self._tile_prefix_override.get(self._current_func_name)
            if ov is not None:
                scoped_ov = self._tile_layouts.get(f"{ov}.{clean_name}")
                if scoped_ov is not None:
                    return scoped_ov
            scoped = self._tile_layouts.get(f"{self._current_func_name}.{clean_name}")
            if scoped is not None:
                return scoped
        return self._tile_layouts.get(clean_name, self._tile_layouts.get(name_hint))

    def _build_aligned_param_prologue(
        self,
        func_name: str,
        new_params: Sequence[tuple[core_ir.Var, core_ir.ParamDirection]],
        old_params: Sequence[core_ir.Var],
        body: core_ir.Stmt,
    ) -> list[core_ir.Stmt]:
        nz_uids = _nz_offset_scalar_uids(body, new_params, old_params)
        prologue_stmts: list[core_ir.Stmt] = []
        for p_idx, (param, _) in enumerate(new_params):
            if param.unique_id not in nz_uids:
                continue
            step = self._param_steps.get((func_name, p_idx))
            if step is not None and step > 1 and step % 8 == 0 and isinstance(param.type, core_ir.ScalarType):
                dt = param.type.dtype
                c0 = core_ir.ConstInt(0, dt, param.span)
                c_step = core_ir.ConstInt(step, dt, param.span)
                max_expr = core_ir.Max(param, c0, dt, param.span)
                div_expr = core_ir.FloorDiv(max_expr, c_step, dt, param.span)
                aligned_expr = core_ir.Mul(div_expr, c_step, dt, param.span)
                aligned_var = core_ir.Var(
                    f"{_clean_ssa_name(param.name_hint)}_aligned", param.type, param.span
                )
                self._var_remap[param.unique_id] = aligned_var
                prologue_stmts.append(core_ir.AssignStmt(aligned_var, aligned_expr, param.span))
        return prologue_stmts

    def visit_function(self, func: core_ir.Function) -> core_ir.Function:
        if func.level is not None and core_ir.level_to_linqu_level(func.level) >= 3:
            return func
        self._var_remap.clear()
        self._current_func_name = func.name
        new_params: list[tuple[core_ir.Var, core_ir.ParamDirection]] = []
        changed = False
        has_nz_param = False
        for param, direction in zip(func.params, func.param_directions, strict=True):
            target_layout = self._lookup_tensor_layout(param.name_hint)
            if (
                isinstance(param.type, core_ir.TensorType)
                and target_layout is not None
                and _tensor_type_layout(param.type) != target_layout
                and not (
                    _tensor_type_layout(param.type) == core_ir.TensorLayout.NZ and len(param.type.shape) >= 4
                )
                and not (target_layout == core_ir.TensorLayout.NZ and len(param.type.shape) < 2)
            ):
                new_type = _with_tensor_layout(param.type, target_layout)
                new_param = core_ir.Var(param.name_hint, new_type, param.span)
                self._var_remap[param.unique_id] = new_param
                new_params.append((new_param, direction))
                changed = True
                if target_layout == core_ir.TensorLayout.NZ:
                    has_nz_param = True
            else:
                new_params.append((param, direction))
                if (
                    isinstance(param.type, core_ir.TensorType)
                    and _tensor_type_layout(param.type) == core_ir.TensorLayout.NZ
                ):
                    has_nz_param = True

        prologue_stmts: list[core_ir.Stmt] = []
        if has_nz_param and core_ir.is_incore_type(func.func_type):
            prologue_stmts = self._build_aligned_param_prologue(func.name, new_params, func.params, func.body)
            if prologue_stmts:
                changed = True

        new_body = self.visit_stmt(func.body)
        if prologue_stmts:
            stmts = (
                [*prologue_stmts, *new_body.stmts]
                if isinstance(new_body, core_ir.SeqStmts)
                else [*prologue_stmts, new_body]
            )
            new_body = core_ir.SeqStmts(stmts, new_body.span)
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

    def _resolve_specialized_callee(
        self,
        callee_name: str,
        new_args: Sequence[core_ir.Expr],
    ) -> str:
        sig_list: list[core_ir.TensorLayout | None] = [
            _tensor_type_layout(a.type) if isinstance(a.type, core_ir.TensorType) else None for a in new_args
        ]
        callee_params = self._callee_formal_params.get(callee_name)
        if callee_params is not None and len(sig_list) < len(callee_params):
            for param in callee_params[len(sig_list) :]:
                sig_list.append(
                    _tensor_type_layout(param.type) if isinstance(param.type, core_ir.TensorType) else None
                )
        return self._specialized_callees.get((callee_name, tuple(sig_list)), callee_name)

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
            op.var.type, (core_ir.TensorType, core_ir.TileType, core_ir.TupleType)
        ):
            if isinstance(new_value.type, (core_ir.TensorType, core_ir.TileType, core_ir.TupleType)):
                new_var = core_ir.Var(op.var.name_hint, new_value.type, op.var.span)
                self._var_remap[op.var.unique_id] = new_var

        if new_var.same_as(op.var) and new_value.same_as(op.value):
            return op
        return core_ir.AssignStmt(new_var, new_value, op.span)

    @staticmethod
    def _sync_call_return_type(
        call_type: core_ir.Type,
        ret_types: Sequence[core_ir.Type] | None,
    ) -> core_ir.Type:
        if not ret_types:
            return call_type
        if isinstance(call_type, core_ir.TensorType) and isinstance(ret_types[0], core_ir.TensorType):
            return _with_tensor_layout(call_type, _tensor_type_layout(ret_types[0]))
        if isinstance(call_type, core_ir.TupleType):
            elems = list(call_type.types)
            changed = False
            for idx, r_type in enumerate(ret_types):
                if (
                    idx < len(elems)
                    and isinstance(elems[idx], core_ir.TensorType)
                    and isinstance(r_type, core_ir.TensorType)
                ):
                    updated = _with_tensor_layout(elems[idx], _tensor_type_layout(r_type))
                    if updated is not elems[idx]:
                        elems[idx] = updated
                        changed = True
            if changed:
                return core_ir.TupleType(elems)
        return call_type

    def visit_call(self, op: core_ir.Call) -> core_ir.Expr:
        new_args = [self.visit_expr(arg) for arg in op.args]
        args_changed = any(not a.same_as(b) for a, b in zip(new_args, op.args, strict=True))
        if isinstance(op.op, core_ir.GlobalVar):
            target_name = self._resolve_specialized_callee(op.op.name, new_args)
            if not args_changed and target_name == op.op.name:
                return op
            target_gv = core_ir.GlobalVar(target_name) if target_name != op.op.name else op.op
            ret_type = self._sync_call_return_type(op.type, self._func_return_types.get(target_name))
            return core_ir.Call(
                target_gv,
                new_args,
                dict(op.kwargs),
                dict(op.attrs) if op.attrs else None,
                ret_type,
                op.span,
            )
        if not args_changed:
            return op
        if op.op.name == _TILE_LOAD_OP:
            return core_ir.Call(
                op.op,
                new_args,
                dict(op.kwargs),
                dict(op.attrs) if op.attrs else None,
                op.type,
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
        target_name = (
            self._resolve_specialized_callee(op.op.name, new_args)
            if isinstance(op.op, core_ir.GlobalVar)
            else ""
        )
        callee_changed = isinstance(op.op, core_ir.GlobalVar) and target_name != op.op.name
        if not args_changed and not deps_changed and not callee_changed:
            return op
        ret_type = op.type
        target_op = op.op
        if isinstance(op.op, core_ir.GlobalVar):
            if callee_changed:
                target_op = core_ir.GlobalVar(target_name)
            ret_type = self._sync_call_return_type(op.type, self._func_return_types.get(target_name))
        return core_ir.Submit(
            target_op,
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


def make_solver_auto_layout_pass(
    cost_model: LayoutCostModel | None = None,
    *,
    reusable_tensors: set[str] | None = None,
    solver: str = "auto",
    max_treewidth: int = 8,
    name: str = "AutoLayout",
) -> passes.Pass:
    """Create a solver-driven `passes.Pass` that extracts, solves, and writes back layouts."""
    if solver not in ("auto", "greedy", "treewidth_dp"):
        raise ValueError(f"Unknown solver={solver!r}; expected 'auto', 'greedy', or 'treewidth_dp'")

    def _run_pass(prog: core_ir.Program) -> core_ir.Program:
        from pypto.ir.auto_layout_solver import (  # noqa: PLC0415
            LayoutSolverConfig,
            solve_auto_layout,
            solve_greedy,
            solve_treewidth_dp,
        )

        inst = extract_layout_instance(
            prog,
            cost_model=cost_model,
            reusable_tensors=reusable_tensors,
        )
        cfg = LayoutSolverConfig(max_treewidth=max_treewidth)
        if solver == "greedy":
            res = solve_greedy(inst, config=cfg)
        elif solver == "treewidth_dp":
            res = solve_treewidth_dp(inst, config=cfg)
        else:
            res = solve_auto_layout(inst, config=cfg)
        if res.writeback_issues:
            return prog
        return apply_layout_assignment(prog, res.writeback_assignment())

    return passes.create_program_pass(_run_pass, name=name)


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
    raw_freq_ghz: float = 0.0
    param_t_stats: dict[str, float] = field(default_factory=_empty_float_dict)
    within_case_r2: float = 1.0
    within_case_mte2_delta_us_per_mb: float = 0.0
    within_case_fixpipe_delta_us_per_mb: float = 0.0
    within_case_op_spearman_rho: float = 1.0
    pooled_op_spearman_rho: float = 1.0


def _rank_array(vals: Sequence[float]) -> np.ndarray:
    """Return zero-based average ranks, giving equal values equal ranks."""
    n = len(vals)
    order = sorted(range(n), key=vals.__getitem__)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = 0.5 * float(i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return rank correlation, or zero when there is no rank variation."""
    if len(xs) < 2:
        return 0.0
    rx = _rank_array(xs)
    ry = _rank_array(ys)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx_c * rx_c) * np.sum(ry_c * ry_c)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx_c * ry_c) / denom)


def _calibration_op_cost(
    model: LayoutCostModel,
    instance: LayoutProblemInstance,
    op: LayoutOperationNode,
    config: OperatorConfig,
) -> float:
    """Reprice one operation under the supplied calibration model."""
    output = instance.values[op.output]
    if op.op_name == _TILE_TRANSPOSE_VIEW_OP:
        return 0.0
    if op.op_name == _TILE_LOAD_OP:
        return model.estimate_load_cost(
            instance.values[op.inputs[0]], output, config.input_layouts[0], config.output_layout
        )
    if op.op_name == _TILE_MOVE_OP:
        return model.estimate_move_cost(
            instance.values[op.inputs[0]], output, config.input_layouts[0], config.output_layout
        )
    if op.op_name == _TILE_STORE_OP and op.inputs:
        in_names = list(op.inputs)
        in_lays = list(config.input_layouts)
        store_out_lay = in_lays[1] if len(in_lays) > 1 else config.output_layout
        store_dst_node = instance.values[in_names[1]] if len(in_names) > 1 else output
        return model.estimate_store_cost(instance.values[in_names[0]], store_dst_node, store_out_lay)
    if op.op_name in _TILE_MATMUL_OPS:
        return model.estimate_matmul_cost(
            instance.values[op.inputs[0]], instance.values[op.inputs[1]], output
        )
    if op.op_name.startswith("tile."):
        return float(output.num_elements) / model.vec_elements_per_cycle
    return config.op_cost_cycles


def _calibration_load_scale(
    model: LayoutCostModel,
    instance: LayoutProblemInstance,
    op: LayoutOperationNode,
    per_run_inv: float,
) -> float:
    """Return the evaluator's per-run MTE2 re-read discount for a load."""
    if op.op_name == _TILE_LOAD_OP and op.inputs:
        src = instance.values[op.inputs[0]]
        dst = instance.values[op.output]
        if dst.memory_space == core_ir.MemorySpace.Mat:
            return model.effective_mte2_load_multiplier(src, dst, per_run_inv)
    return 1.0


def _calibration_prediction_cycles(model: LayoutCostModel, measurement: CalibrationMeasurement) -> float:
    """Return single-launch operator cycles; boundary packing is outside timing."""
    expected_uses = float(measurement.instance.cost_model.expected_uses)
    total = 0.0
    for op in measurement.instance.operations:
        config = measurement.breakdown.assignment.op_configs.get(op.op_id)
        if config is None:
            continue
        inv = float(op.invocations) / expected_uses
        scale = _calibration_load_scale(model, measurement.instance, op, inv)
        total += inv * scale * _calibration_op_cost(model, measurement.instance, op, config)
    return total


def _calibration_features(
    model: LayoutCostModel,
    measurement: CalibrationMeasurement,
    expected_uses: float,
    contiguous_nd_ratio: float,
) -> list[float]:
    """Build four transfer-byte features and one retained-cycle feature."""
    instance = measurement.instance
    nd2nz_b = nz2nz_b = store_nd_b = store_nz_b = other_cycles = 0.0
    for op in instance.operations:
        cfg = measurement.breakdown.assignment.op_configs.get(op.op_id)
        if cfg is None:
            continue
        inv = float(op.invocations) / expected_uses
        if op.op_name == _TILE_LOAD_OP and op.inputs:
            src_node = instance.values[op.inputs[0]]
            dst_node = instance.values[op.output]
            if dst_node.memory_space == core_ir.MemorySpace.Mat:
                scale = _calibration_load_scale(model, instance, op, inv)
                tile_bytes = float(dst_node.byte_size)
                if cfg.input_layouts[0] == core_ir.TensorLayout.NZ:
                    nz2nz_b += inv * scale * tile_bytes
                    transfer_cycles = tile_bytes / model.mte2_nz2nz_bytes_per_cycle
                else:
                    cols = dst_node.shape[-1] if dst_node.shape else 1
                    parent_cols = src_node.shape[-1] if src_node.shape else cols
                    strided = cols < parent_cols
                    ratio = 1.0 if strided else max(1.0, contiguous_nd_ratio)
                    nd2nz_b += inv * scale * tile_bytes / ratio
                    transfer_cycles = tile_bytes / (model.mte2_nd2nz_bytes_per_cycle * ratio)
                other_cycles += (
                    inv * scale * (_calibration_op_cost(model, instance, op, cfg) - transfer_cycles)
                )
                continue
        elif op.op_name == _TILE_STORE_OP and op.inputs:
            src_node = instance.values[op.inputs[0]]
            store_out_lay = cfg.input_layouts[1] if len(cfg.input_layouts) > 1 else cfg.output_layout
            if store_out_lay == core_ir.TensorLayout.NZ:
                store_nz_b += inv * float(src_node.byte_size)
                store_bw = model.fixpipe_acc_to_nz_bytes_per_cycle
            else:
                store_nd_b += inv * float(src_node.byte_size)
                store_bw = model.fixpipe_acc_to_nd_bytes_per_cycle
            other_cycles += inv * (
                _calibration_op_cost(model, instance, op, cfg) - float(src_node.byte_size) / store_bw
            )
            continue
        other_cycles += inv * _calibration_op_cost(model, instance, op, cfg)
    return [1.0, nd2nz_b, nz2nz_b, store_nd_b, store_nz_b, other_cycles]


def _fit_within_case_differentials(  # noqa: PLR0913
    rows_x: Sequence[Sequence[float]],
    rows_y: Sequence[float],
    op_cycles_list: Sequence[float],
    case_groups: dict[str, list[int]],
    beta_mte2: float,
    beta_fixpipe: float,
    us_per_cycle: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Fit case-demeaned ND->NZ differential slopes for MTE2 loads and FIXPIPE stores."""
    adj_y = [
        rows_y[i]
        - beta_mte2 * (rows_x[i][1] + rows_x[i][2])
        - beta_fixpipe * (rows_x[i][3] + rows_x[i][4])
        - us_per_cycle * rows_x[i][5]
        for i in range(len(rows_y))
    ]
    d_rows_x: list[list[float]] = [[0.0, 0.0] for _ in rows_y]
    d_rows_y: list[float] = [0.0 for _ in rows_y]
    within_case_rhos: list[float] = []
    for idxs in case_groups.values():
        if len(idxs) < 2:
            continue
        mean_nd = sum(rows_x[i][1] for i in idxs) / float(len(idxs))
        mean_snd = sum(rows_x[i][3] for i in idxs) / float(len(idxs))
        mean_y = sum(adj_y[i] for i in idxs) / float(len(idxs))
        for i in idxs:
            d_rows_x[i] = [rows_x[i][1] - mean_nd, rows_x[i][3] - mean_snd]
            d_rows_y[i] = adj_y[i] - mean_y
        grp_ops = [op_cycles_list[i] for i in idxs]
        if max(grp_ops) > min(grp_ops):
            within_case_rhos.append(_spearman_rank_correlation(grp_ops, [rows_y[i] for i in idxs]))

    xd_mat = np.asarray(d_rows_x, dtype=np.float64)
    yd_vec = np.asarray(d_rows_y, dtype=np.float64)
    xd_norms = np.sqrt(np.sum(xd_mat * xd_mat, axis=0))
    xd_scales = [float(s) if float(s) > 0.0 else 1.0 for s in xd_norms.tolist()]
    xd_scaled = np.asarray(xd_mat / np.asarray(xd_scales, dtype=np.float64), dtype=np.float64)
    yd_scaled_coef, _, xd_rank, _ = np.linalg.lstsq(xd_scaled, yd_vec, rcond=None)
    delta_coef = [float(yd_scaled_coef[j]) / xd_scales[j] for j in range(2)]
    yd_pred = xd_mat @ np.asarray(delta_coef, dtype=np.float64)
    ss_res = float(np.sum((yd_vec - yd_pred) ** 2))
    ss_tot = float(np.sum(yd_vec * yd_vec))
    within_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    raw_dof_within = sum(len(idxs) - 1 for idxs in case_groups.values() if len(idxs) >= 2) - int(xd_rank)
    sigma2_within = ss_res / float(max(1, raw_dof_within))
    gram_within: list[list[float]] = [
        [
            sum((d_rows_x[r][i] / xd_scales[i]) * (d_rows_x[r][j] / xd_scales[j]) for r in range(len(rows_y)))
            for j in range(2)
        ]
        for i in range(2)
    ]
    gram_within_arr = np.asarray(gram_within, dtype=np.float64)
    xd_inv: np.ndarray = np.linalg.pinv(gram_within_arr)
    proj_within: np.ndarray = xd_inv @ gram_within_arr
    estimable_within = [
        raw_dof_within > 0 and float(xd_norms[j]) > 0.0 and abs(1.0 - float(proj_within[j, j])) <= 1e-7
        for j in range(2)
    ]
    se_within = [math.sqrt(max(0.0, float(xd_inv[j, j]) * sigma2_within)) / xd_scales[j] for j in range(2)]
    norm_mte2 = float(xd_norms[0])
    norm_fixpipe = float(xd_norms[1])
    delta_mte2 = delta_coef[0] if estimable_within[0] else 0.0
    delta_fixpipe = delta_coef[1] if estimable_within[1] else 0.0
    t_mte2 = (
        (delta_mte2 / se_within[0])
        if estimable_within[0] and se_within[0] > 1e-18
        else (float("inf") if estimable_within[0] and delta_mte2 > 1e-9 else 0.0)
    )
    t_fixpipe = (
        (delta_fixpipe / se_within[1])
        if estimable_within[1] and se_within[1] > 1e-18
        else (float("inf") if estimable_within[1] and abs(delta_fixpipe) > 1e-9 else 0.0)
    )
    within_op_rho = float(np.median(within_case_rhos)) if within_case_rhos else 1.0
    return (
        delta_mte2,
        t_mte2,
        norm_mte2,
        delta_fixpipe,
        t_fixpipe,
        norm_fixpipe,
        within_r2,
        within_op_rho,
    )


def _anchor_pair_from_differential(  # noqa: PLR0913
    nd_name: str,
    nz_name: str,
    fit_nd: float,
    fit_nz: float,
    default_nd: float,
    default_nz: float,
    delta_slope: float,
    t_delta: float,
    norm_delta: float,
    us_per_cycle: float,
    min_t_stat: float,
    max_physical_bpc: float,
    require_both_cols: bool,
    fallback_params: list[str],
    param_t_stats: dict[str, float],
) -> tuple[float, float]:
    trigger = (
        (nd_name in fallback_params or nz_name in fallback_params)
        if require_both_cols
        else (nd_name in fallback_params and nz_name not in fallback_params)
    )
    if not trigger or norm_delta <= 0.0:
        return fit_nd, fit_nz
    if require_both_cols:
        fit_nz = default_nz
        if nz_name not in fallback_params:
            fallback_params.append(nz_name)
    anchored_nd_slope = (us_per_cycle / fit_nz) + delta_slope
    param_t_stats[nd_name] = float(t_delta)
    t_ok = (abs(t_delta) >= min_t_stat) if require_both_cols else (t_delta >= min_t_stat)
    if t_ok and anchored_nd_slope > 1e-9 and (us_per_cycle / anchored_nd_slope) <= max_physical_bpc:
        fit_nd = max(1.0, us_per_cycle / anchored_nd_slope)
        if nd_name in fallback_params:
            fallback_params.remove(nd_name)
    else:
        fit_nd = default_nd
        if nd_name not in fallback_params:
            fallback_params.append(nd_name)
    return fit_nd, fit_nz


def fit_cost_model_from_measurements(
    measurements: Sequence[CalibrationMeasurement],
    base_model: LayoutCostModel | None = None,
    *,
    nominal_freq_ghz: float = 1.8,
    min_t_stat: float = 2.0,
    max_physical_bpc: float = 1024.0,
) -> CalibrationReport:
    """Fit MTE2/FIXPIPE bandwidth coefficients & dispatch offset to measured latencies."""
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

    rows_x: list[list[float]] = []
    rows_y: list[float] = []
    op_cycles_list: list[float] = []
    case_groups: dict[str, list[int]] = {}
    variant_ids: list[str] = []
    if (
        not math.isfinite(model.mte2_nd2nz_bytes_per_cycle)
        or model.mte2_nd2nz_bytes_per_cycle <= 0.0
        or not math.isfinite(model.mte2_nd2nz_contiguous_bytes_per_cycle)
        or model.mte2_nd2nz_contiguous_bytes_per_cycle <= 0.0
    ):
        raise ValueError("ND load calibration requires positive finite base bandwidths")
    contiguous_nd_ratio = model.mte2_nd2nz_contiguous_bytes_per_cycle / model.mte2_nd2nz_bytes_per_cycle

    for idx, m in enumerate(measurements):
        expected_uses = float(m.instance.cost_model.expected_uses)
        if not math.isfinite(expected_uses) or expected_uses <= 0.0:
            raise ValueError(
                f"Measurement {m.variant_id!r} requires positive finite expected_uses, "
                f"got {m.instance.cost_model.expected_uses}"
            )
        if not math.isfinite(m.measured_us) or m.measured_us <= 0.0:
            raise ValueError(
                f"Measurement {m.variant_id!r} requires positive finite measured_us, got {m.measured_us}"
            )
        rows_x.append(_calibration_features(model, m, expected_uses, contiguous_nd_ratio))
        rows_y.append(float(m.measured_us))
        op_cycles_list.append(float(m.breakdown.operator_cost_cycles) / expected_uses)
        case_key = m.variant_id.split("/", 1)[0] if "/" in m.variant_id else str(id(m.instance))
        case_groups.setdefault(case_key, []).append(idx)
        variant_ids.append(m.variant_id)

    x_mat = np.asarray(rows_x, dtype=np.float64)
    y_vec = np.asarray(rows_y, dtype=np.float64)
    col_norms = np.sqrt(np.sum(x_mat * x_mat, axis=0))
    safe_scales = np.asarray([s if s > 0.0 else 1.0 for s in col_norms.tolist()], dtype=np.float64)
    x_scaled = np.asarray(x_mat / safe_scales, dtype=np.float64)
    scaled_coef, _, design_rank, _ = np.linalg.lstsq(x_scaled, y_vec, rcond=None)
    coef = scaled_coef / safe_scales
    raw_y_pred = x_mat @ coef

    n_cols = len(rows_x[0])
    dof = len(rows_y) - int(design_rank)
    rss = float(np.sum((y_vec - raw_y_pred) ** 2))
    sigma2 = rss / float(max(1, dof))
    scales_list = [float(s) for s in safe_scales.tolist()]
    gram: list[list[float]] = [
        [sum((row[i] / scales_list[i]) * (row[j] / scales_list[j]) for row in rows_x) for j in range(n_cols)]
        for i in range(n_cols)
    ]
    gram_arr = np.asarray(gram, dtype=np.float64)
    xtx_scaled_inv: np.ndarray = np.linalg.pinv(gram_arr)
    proj_row: np.ndarray = xtx_scaled_inv @ gram_arr
    estimable = [dof > 0 and abs(1.0 - float(proj_row[j, j])) <= 1e-7 for j in range(n_cols)]
    se_list = [
        math.sqrt(max(0.0, float(xtx_scaled_inv[j, j]) * sigma2)) / scales_list[j] for j in range(n_cols)
    ]

    raw_freq_ghz = (1.0 / (float(coef[5]) * 1000.0)) if float(coef[5]) > 1e-6 else 0.0
    if estimable[5] and 0.8 * nominal_freq_ghz <= raw_freq_ghz <= 1.2 * nominal_freq_ghz:
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
        t_val = (
            ((slope / err) if err > 1e-18 else (float("inf") if slope > 1e-9 else 0.0))
            if estimable[col_idx]
            else 0.0
        )
        param_t_stats[name] = float(t_val)
        if not estimable[col_idx] or slope <= 1e-9 or t_val < min_t_stat:
            fallback_params.append(name)
            return fallback
        bpc = us_per_cycle / slope
        if bpc > max_physical_bpc:
            fallback_params.append(name)
            return fallback
        return max(1.0, bpc)

    fit_nd2nz = _slope_to_bpc("mte2_nd2nz_bytes_per_cycle", 1, model.mte2_nd2nz_bytes_per_cycle)
    fit_nz2nz = _slope_to_bpc("mte2_nz2nz_bytes_per_cycle", 2, model.mte2_nz2nz_bytes_per_cycle)
    fit_acc_to_nd = _slope_to_bpc(
        "fixpipe_acc_to_nd_bytes_per_cycle", 3, model.fixpipe_acc_to_nd_bytes_per_cycle
    )
    fit_acc_to_nz = _slope_to_bpc(
        "fixpipe_acc_to_nz_bytes_per_cycle", 4, model.fixpipe_acc_to_nz_bytes_per_cycle
    )

    anchor_acc_nz = (
        model.fixpipe_acc_to_nz_bytes_per_cycle
        if (
            "fixpipe_acc_to_nd_bytes_per_cycle" in fallback_params
            or "fixpipe_acc_to_nz_bytes_per_cycle" in fallback_params
        )
        else fit_acc_to_nz
    )
    raw_op_slope = float(coef[5]) if estimable[5] and float(coef[5]) > 0.0 else 0.0
    (
        delta_mte2_slope,
        t_delta_mte2,
        norm_mte2,
        delta_fixpipe_slope,
        t_delta_fixpipe,
        norm_fixpipe,
        within_r2,
        within_op_rho,
    ) = _fit_within_case_differentials(
        rows_x,
        rows_y,
        op_cycles_list,
        case_groups,
        us_per_cycle / fit_nz2nz,
        us_per_cycle / anchor_acc_nz,
        raw_op_slope,
    )

    if estimable[5] and len(case_groups) > 1:
        fit_nd2nz, fit_nz2nz = _anchor_pair_from_differential(
            "mte2_nd2nz_bytes_per_cycle",
            "mte2_nz2nz_bytes_per_cycle",
            fit_nd2nz,
            fit_nz2nz,
            model.mte2_nd2nz_bytes_per_cycle,
            model.mte2_nz2nz_bytes_per_cycle,
            delta_mte2_slope,
            t_delta_mte2,
            norm_mte2,
            us_per_cycle,
            min_t_stat,
            max_physical_bpc,
            False,
            fallback_params,
            param_t_stats,
        )
        fit_acc_to_nd, fit_acc_to_nz = _anchor_pair_from_differential(
            "fixpipe_acc_to_nd_bytes_per_cycle",
            "fixpipe_acc_to_nz_bytes_per_cycle",
            fit_acc_to_nd,
            fit_acc_to_nz,
            model.fixpipe_acc_to_nd_bytes_per_cycle,
            model.fixpipe_acc_to_nz_bytes_per_cycle,
            delta_fixpipe_slope,
            t_delta_fixpipe,
            norm_fixpipe,
            us_per_cycle,
            min_t_stat,
            max_physical_bpc,
            True,
            fallback_params,
            param_t_stats,
        )

    fitted = replace(
        model,
        mte2_nd2nz_bytes_per_cycle=fit_nd2nz,
        mte2_nd2nz_contiguous_bytes_per_cycle=contiguous_nd_ratio * fit_nd2nz,
        mte2_nz2nz_bytes_per_cycle=fit_nz2nz,
        fixpipe_acc_to_nd_bytes_per_cycle=fit_acc_to_nd,
        fixpipe_acc_to_nz_bytes_per_cycle=fit_acc_to_nz,
    )

    y_pred = np.asarray(
        [
            float(coef[0]) + us_per_cycle * _calibration_prediction_cycles(fitted, measurement)
            for measurement in measurements
        ],
        dtype=np.float64,
    )
    ss_res = float(np.sum((y_vec - y_pred) ** 2))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float(ss_res <= 1e-12)
    mape = 100.0 * float(np.mean(np.abs(y_vec - y_pred) / np.maximum(np.abs(y_vec), 1e-9)))
    rho = _spearman_rank_correlation([float(yp) for yp in y_pred], rows_y)
    pooled_op_rho = _spearman_rank_correlation(op_cycles_list, rows_y)
    residuals = {vid: float(yp - yt) for vid, yt, yp in zip(variant_ids, y_vec, y_pred, strict=True)}
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
        raw_freq_ghz=raw_freq_ghz,
        param_t_stats=param_t_stats,
        within_case_r2=within_r2,
        within_case_mte2_delta_us_per_mb=delta_mte2_slope * 1e6,
        within_case_fixpipe_delta_us_per_mb=delta_fixpipe_slope * 1e6,
        within_case_op_spearman_rho=within_op_rho,
        pooled_op_spearman_rho=pooled_op_rho,
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
    "make_solver_auto_layout_pass",
    "pack_nz",
    "run_post_layout_passes",
    "run_pre_layout_passes",
    "unpack_nz",
]
