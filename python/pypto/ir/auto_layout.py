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

import json
import math
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
from pypto.pypto_core import auto_layout as _cpp_al
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


def pack_dn(logical: torch.Tensor) -> torch.Tensor:
    """Pack a logical tensor (`ndim >= 2`) into Ascend DN (column-major trailing plane) byte order."""
    if logical.ndim < 2:
        raise ValueError(f"pack_dn expects rank >= 2, got shape {tuple(logical.shape)}")
    return logical.transpose(-2, -1).contiguous().reshape(logical.shape)


def unpack_dn(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_dn`: convert a DN-packed tensor back to logical ND order."""
    if packed.ndim < 2:
        raise ValueError(f"unpack_dn expects rank >= 2, got shape {tuple(packed.shape)}")
    lead: tuple[int, ...] = tuple(int(d) for d in packed.shape[:-2])
    rows: int = int(packed.shape[-2])
    cols: int = int(packed.shape[-1])
    return packed.reshape(*lead, cols, rows).transpose(-2, -1).contiguous()


@dataclass(frozen=True)
class TileLayoutDesc:
    """On-chip tile layout descriptor `(blayout, slayout, fractal[, tag])`.

    Common Ascend on-chip layouts:
    - `ROW_MAJOR_FLAT`: `(row_major, none_box, 512)` — UB (`Vec`) row-major
    - `COL_MAJOR_FLAT`: `(col_major, none_box, 512)` — UB (`Vec`) col-major
    - `ROW_BCAST_FLAT`: `(row_major, none_box, 512, "row_bcast")` — UB (`Vec`) pre-expanded `[M, c0]` carrier
    - `FRACTAL_NZ`: `(col_major, row_major, 512)` — L1 (`Mat`) / L0A (`Left`) NZ
    - `FRACTAL_ZN`: `(row_major, col_major, 512)` — L1 (`Mat`) transposed / L0B (`Right`) ZN
    - `FRACTAL_ACC_NZ`: `(col_major, row_major, 1024)` — L0C (`Acc`) 16x16 accumulator NZ
    """

    blayout: core_ir.TileLayout
    slayout: core_ir.TileLayout
    fractal: int = 512
    tag: str = ""

    def short_name(self) -> str:
        """Return a concise human-readable layout label."""
        if self.tag == "row_bcast":
            return "flat_row_bcast"
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
ROW_BCAST_FLAT = TileLayoutDesc(core_ir.TileLayout.row_major, core_ir.TileLayout.none_box, 512, "row_bcast")
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
    mte2_vec_strided_bytes_per_cycle: float = 96.0
    mte3_vec_bytes_per_cycle: float = 192.0
    mte3_vec_strided_bytes_per_cycle: float = 96.0
    vec_strided_burst_penalty_cycles: float = 3.0
    mte1_l1_to_l0_bytes_per_cycle: float = 129.7
    mte1_l0a_bytes_per_cycle: float = 129.7
    mte1_l0b_bytes_per_cycle: float = 85.4
    mte1_transpose_penalty_cycles: float = 8.0
    cube_macs_per_cycle: float = 4096.0
    cube_issue_overhead_cycles: float = 21.0
    mad_fp32_passes: int | None = None
    vec_elements_per_cycle: float = 128.0
    vec_reshape_metadata_cycles: float = 1.0
    vec_transpose_fixed_cycles: float = 0.0
    vec_vbrcb_fixed_cycles: float = 4.0
    vec_vbrcb_cycles_per_row: float = 0.5
    vec_row_expand_issue_cycles: float = 4.0
    vec_unsupported_conversion_cycles: float = 1e12
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

    def _to_cpp(self) -> _cpp_al.LayoutCostModel:
        return _to_cpp_cost_model(self)

    def supports_acc_nz_store(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
    ) -> bool:
        """Return whether `Acc -> GM` FIXPIPE supports direct NZ store for `(src_node, dst_node)`."""
        return self._to_cpp().supports_acc_nz_store(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
        )

    def estimate_vbrcb_cost(self, rows: int) -> float:
        """Estimate VBRCB broadcast cost across `rows` rows."""
        return self._to_cpp().estimate_vbrcb_cost(rows)

    def is_valid_load_config(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> bool:
        """Check whether `tile.load` with `(src_layout, dst_layout)` is backend-legal."""
        return self._to_cpp().is_valid_load_config(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )

    def is_valid_vec_store_config(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> bool:
        """Check whether `Mem.Vec` `tile.store` with `(src_layout, dst_layout)` is backend-legal."""
        return self._to_cpp().is_valid_vec_store_config(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )

    def is_valid_move_config(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> bool:
        """Check whether `tile.move(src -> dst)` with `(src_layout, dst_layout)` is backend-legal."""
        return self._to_cpp().is_valid_move_config(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )

    def required_input_tile_layout(self, op_name: str, input_index: int) -> core_ir.TileLayout | None:
        """Query the active C++ `Backend`'s `BackendTileLayoutSpec` for `op_name` input `input_index`."""
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
        return self._to_cpp().effective_mte2_load_multiplier(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            per_run_inv,
        )

    def estimate_load_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate `tile.load` cost from GM (`src_node`) to `Mat` or `Vec` (`dst_node`)."""
        return self._to_cpp().estimate_load_cost(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )

    def estimate_move_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        """Estimate on-chip `tile.move` cost (e.g. `Mat -> Left` or `Mat -> Right`)."""
        return self._to_cpp().estimate_move_cost(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )

    def estimate_matmul_cost(
        self,
        lhs_node: LayoutValueNode,
        rhs_node: LayoutValueNode,
        out_node: LayoutValueNode,
    ) -> float:
        """Estimate L0 `tile.matmul` (`Left` x `Right` -> `Acc`) compute cost."""
        return self._to_cpp().estimate_matmul_cost(
            _to_cpp_value_node(lhs_node),
            _to_cpp_value_node(rhs_node),
            _to_cpp_value_node(out_node),
        )

    def estimate_store_cost(
        self,
        src_node: LayoutValueNode,
        dst_node: LayoutValueNode,
        dst_layout: LayoutChoice,
        src_layout: LayoutChoice | None = None,
    ) -> float:
        """Estimate `tile.store` (`Acc -> GM` via FIXPIPE or `Vec -> GM` via MTE3) cost."""
        return self._to_cpp().estimate_store_cost(
            _to_cpp_value_node(src_node),
            _to_cpp_value_node(dst_node),
            _to_cpp_layout_spec(dst_layout),
            _to_cpp_layout_spec(src_layout) if src_layout is not None else None,
        )

    def supports_vec_tile_conversion(
        self,
        val_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> bool:
        """Check whether an on-chip `Mem.Vec` conversion `src_layout -> dst_layout` is realizable."""
        return self._to_cpp().supports_vec_tile_conversion(
            _to_cpp_value_node(val_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )

    def estimate_conversion_cost(
        self,
        val_node: LayoutValueNode,
        src_layout: LayoutChoice,
        dst_layout: LayoutChoice,
    ) -> float:
        r"""Estimate conversion cost $C^{\text{tr}}_t(\ell_1, \ell_2)$ for a value."""
        return self._to_cpp().estimate_conversion_cost(
            _to_cpp_value_node(val_node),
            _to_cpp_layout_spec(src_layout),
            _to_cpp_layout_spec(dst_layout),
        )


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
_OTHER_ROW_EXPAND_OPS = frozenset(
    {
        core_ir.get_op("tile.row_expand").name,
        core_ir.get_op("tile.row_expand_sub").name,
        core_ir.get_op("tile.row_expand_mul").name,
        core_ir.get_op("tile.row_expand_div").name,
        core_ir.get_op("tile.row_expand_max").name,
        core_ir.get_op("tile.row_expand_min").name,
        core_ir.get_op("tile.row_expand_expdif").name,
    }
)

_COST_MODEL_FIELDS = (
    "expected_uses",
    "backend_type",
    "mte2_nz2nz_bytes_per_cycle",
    "mte2_nd2nz_bytes_per_cycle",
    "mte2_nd2nz_contiguous_bytes_per_cycle",
    "mte2_nd2nz_row_penalty_cycles",
    "mte2_nd2nz_sub512b_penalty_cycles",
    "mte2_l2_reread_discount",
    "l2_cache_line_bytes",
    "mte2_vec_bytes_per_cycle",
    "mte2_vec_strided_bytes_per_cycle",
    "mte3_vec_bytes_per_cycle",
    "mte3_vec_strided_bytes_per_cycle",
    "vec_strided_burst_penalty_cycles",
    "mte1_l1_to_l0_bytes_per_cycle",
    "mte1_l0a_bytes_per_cycle",
    "mte1_l0b_bytes_per_cycle",
    "mte1_transpose_penalty_cycles",
    "cube_macs_per_cycle",
    "cube_issue_overhead_cycles",
    "mad_fp32_passes",
    "vec_elements_per_cycle",
    "vec_reshape_metadata_cycles",
    "vec_transpose_fixed_cycles",
    "vec_vbrcb_fixed_cycles",
    "vec_vbrcb_cycles_per_row",
    "vec_row_expand_issue_cycles",
    "vec_unsupported_conversion_cycles",
    "fixpipe_acc_to_nd_bytes_per_cycle",
    "fixpipe_acc_to_nz_bytes_per_cycle",
    "fixpipe_drain_fixed_cycles",
    "fixpipe_odd_n1_penalty_cycles",
    "gm_pack_bytes_per_cycle",
    "gm_pack_fixed_cycles",
    "onchip_relayout_bytes_per_cycle",
    "allow_external_input_relayout",
    "allow_external_output_relayout",
)


def _tensor_type_layout(tensor_type: core_ir.TensorType) -> core_ir.TensorLayout:  # pyright: ignore[reportUnusedFunction]
    return tensor_type.tensor_view.layout if tensor_type.tensor_view is not None else core_ir.TensorLayout.ND


def _to_cpp_layout_spec(choice: LayoutChoice) -> _cpp_al.LayoutSpec:
    if isinstance(choice, core_ir.TensorLayout):
        return _cpp_al.LayoutSpec.tensor(choice)
    return _cpp_al.LayoutSpec.tile(choice.blayout, choice.slayout, choice.fractal, choice.tag)


def _from_cpp_layout_spec(spec: _cpp_al.LayoutSpec) -> LayoutChoice:
    if spec.is_tensor():
        return spec.tensor_layout
    return TileLayoutDesc(spec.blayout, spec.slayout, spec.fractal, spec.tag)


def _to_cpp_value_node(node: LayoutValueNode) -> _cpp_al.ValueNode:
    out = _cpp_al.ValueNode()
    out.name = node.name
    out.shape = list(node.shape)
    out.dtype = node.dtype
    out.memory_space = node.memory_space
    out.current_layout = _to_cpp_layout_spec(node.current_layout)
    out.admissible_layouts = [_to_cpp_layout_spec(lay) for lay in node.admissible_layouts]
    out.producer_op_id = node.producer_op_id
    out.is_reusable_weight = node.is_reusable_weight
    out.is_external_output = node.is_external_output
    return out


def _from_cpp_value_node(node: _cpp_al.ValueNode) -> LayoutValueNode:
    return LayoutValueNode(
        name=node.name,
        shape=tuple(node.shape),
        dtype=node.dtype,
        memory_space=node.memory_space,
        current_layout=_from_cpp_layout_spec(node.current_layout),
        admissible_layouts=tuple(_from_cpp_layout_spec(lay) for lay in node.admissible_layouts),
        producer_op_id=node.producer_op_id,
        is_reusable_weight=node.is_reusable_weight,
        is_external_output=node.is_external_output,
    )


def _to_cpp_op_config(cfg: OperatorConfig) -> _cpp_al.OperatorConfig:
    out = _cpp_al.OperatorConfig()
    out.op_id = cfg.op_id
    out.op_name = cfg.op_name
    out.input_layouts = [_to_cpp_layout_spec(lay) for lay in cfg.input_layouts]
    out.output_layout = _to_cpp_layout_spec(cfg.output_layout)
    out.op_cost_cycles = cfg.op_cost_cycles
    out.notes = cfg.notes
    return out


def _from_cpp_op_config(cfg: _cpp_al.OperatorConfig) -> OperatorConfig:
    return OperatorConfig(
        op_id=cfg.op_id,
        op_name=cfg.op_name,
        input_layouts=tuple(_from_cpp_layout_spec(lay) for lay in cfg.input_layouts),
        output_layout=_from_cpp_layout_spec(cfg.output_layout),
        op_cost_cycles=cfg.op_cost_cycles,
        notes=cfg.notes,
    )


def _to_cpp_operation_node(op: LayoutOperationNode) -> _cpp_al.OperationNode:
    out = _cpp_al.OperationNode()
    out.op_id = op.op_id
    out.op_name = op.op_name
    out.inputs = list(op.inputs)
    out.output = op.output
    out.invocations = op.invocations
    out.configs = [_to_cpp_op_config(cfg) for cfg in op.configs]
    return out


def _from_cpp_operation_node(op: _cpp_al.OperationNode) -> LayoutOperationNode:
    return LayoutOperationNode(
        op_id=op.op_id,
        op_name=op.op_name,
        inputs=tuple(op.inputs),
        output=op.output,
        invocations=op.invocations,
        configs=tuple(_from_cpp_op_config(cfg) for cfg in op.configs),
    )


def _to_cpp_cost_model(model: LayoutCostModel) -> _cpp_al.LayoutCostModel:
    out = _cpp_al.LayoutCostModel()
    for name in _COST_MODEL_FIELDS:
        setattr(out, name, getattr(model, name))
    out.resolved_arch_hint = model.target_arch()
    return out


def _from_cpp_cost_model(model: _cpp_al.LayoutCostModel) -> LayoutCostModel:
    kwargs: dict[str, Any] = {name: getattr(model, name) for name in _COST_MODEL_FIELDS}
    return LayoutCostModel(**kwargs)


def _to_cpp_instance(instance: LayoutProblemInstance) -> _cpp_al.LayoutProblemInstance:
    out = _cpp_al.LayoutProblemInstance()
    out.cost_model = _to_cpp_cost_model(instance.cost_model)
    for node in instance.values.values():
        out.add_value(_to_cpp_value_node(node))
    out.operations = [_to_cpp_operation_node(op) for op in instance.operations]
    # Forward Python LayoutCostModel subclass overrides of `estimate_conversion_cost` into C++ table.
    if type(instance.cost_model).estimate_conversion_cost is not LayoutCostModel.estimate_conversion_cost:
        for node in instance.values.values():
            candidates = list(dict.fromkeys([node.current_layout, *node.admissible_layouts]))
            for src in candidates:
                for dst in candidates:
                    cost = float(instance.cost_model.estimate_conversion_cost(node, src, dst))
                    out.set_custom_conversion_cost(
                        node.name,
                        _to_cpp_layout_spec(src),
                        _to_cpp_layout_spec(dst),
                        cost,
                    )
    return out


def _from_cpp_instance(
    cpp_inst: _cpp_al.LayoutProblemInstance,
    cost_model: LayoutCostModel | None = None,
) -> LayoutProblemInstance:
    values: dict[str, LayoutValueNode] = {}
    for name in cpp_inst.value_order:
        if name in cpp_inst.values:
            values[name] = _from_cpp_value_node(cpp_inst.values[name])
    for name, node in cpp_inst.values.items():
        if name not in values:
            values[name] = _from_cpp_value_node(node)
    operations = [_from_cpp_operation_node(op) for op in cpp_inst.operations]
    model = cost_model if cost_model is not None else _from_cpp_cost_model(cpp_inst.cost_model)
    return LayoutProblemInstance(values=values, operations=operations, cost_model=model)


def _to_cpp_assignment(assignment: LayoutAssignment) -> _cpp_al.LayoutAssignment:
    out = _cpp_al.LayoutAssignment()
    out.value_layouts = {k: _to_cpp_layout_spec(v) for k, v in assignment.value_layouts.items()}
    out.op_configs = {k: _to_cpp_op_config(v) for k, v in assignment.op_configs.items()}
    return out


def _from_cpp_assignment(assignment: _cpp_al.LayoutAssignment) -> LayoutAssignment:
    return LayoutAssignment(
        value_layouts={k: _from_cpp_layout_spec(v) for k, v in assignment.value_layouts.items()},
        op_configs={k: _from_cpp_op_config(v) for k, v in assignment.op_configs.items()},
    )


def _from_cpp_breakdown(bd: _cpp_al.CostBreakdown) -> CostBreakdown:
    return CostBreakdown(
        assignment=_from_cpp_assignment(bd.assignment),
        total_cost_cycles=bd.total_cost_cycles,
        operator_cost_cycles=bd.operator_cost_cycles,
        conversion_cost_cycles=bd.conversion_cost_cycles,
        per_op_costs=dict(bd.per_op_costs),
        per_value_conversion_costs=dict(bd.per_value_conversion_costs),
    )


def run_pre_layout_passes(program: core_ir.Program) -> core_ir.Program:
    """Run the default compilation pipeline up through `InferTileMemorySpace` (Pass 22)."""
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
    pipe.add_pass(passes.block_nz_tensor_views())
    after_infer = False
    for p in pm.passes:
        if p.get_name() == "InferTileMemorySpace":
            after_infer = True
            continue
        if after_infer:
            pipe.add_pass(p)
    return pipe.run(program)


class _OrchCallMultiplierVisitor:  # pyright: ignore[reportUnusedClass]
    """White-box test compatibility shim delegating orchestration analysis to C++."""

    def __init__(self, func_by_name: Mapping[str, core_ir.Function]) -> None:
        self._func_by_name = dict(func_by_name)
        self._analyzed = False
        self.call_counts: dict[str, int] = {}
        self.call_arg_names: dict[str, list[str | None]] = {}
        self.call_sites: dict[str, list[tuple[tuple[str | None, ...], int]]] = {}
        self.call_order: dict[str, int] = {}
        self.param_steps: dict[tuple[str, int], int] = {}
        self.reshaped_names: set[str] = set()
        self.nz_ineligible_names: set[str] = set()
        self.internal_buffers: set[str] = set()

    def walk(self, func: core_ir.Function) -> None:
        if self._analyzed:
            return
        self._analyzed = True
        prog = core_ir.Program(list(self._func_by_name.values()), func.name, func.span)
        res = _cpp_al.analyze_orchestration(prog)
        self.call_counts = dict(res.call_counts)
        self.call_arg_names = dict(res.call_arg_names)
        self.call_sites = {
            k: [(tuple(site.arg_names), site.multiplicity) for site in sites]
            for k, sites in res.call_sites.items()
        }
        self.call_order = dict(res.call_order)
        self.param_steps = dict(res.param_steps)
        self.reshaped_names = set(res.reshaped_names)
        self.nz_ineligible_names = set(res.nz_ineligible_names)
        self.internal_buffers = set(res.internal_buffers)


class _LayoutAssignmentWriter:  # pyright: ignore[reportUnusedClass]
    """Compatibility helper exposing static `_sync_call_return_type` via C++."""

    @staticmethod
    def _sync_call_return_type(
        call_type: core_ir.Type,
        ret_types: Sequence[core_ir.Type] | None,
    ) -> core_ir.Type:
        return _cpp_al.sync_call_return_type(call_type, list(ret_types) if ret_types else [])


def extract_layout_instance(
    program: core_ir.Program,
    *,
    cost_model: LayoutCostModel | None = None,
    reusable_tensors: set[str] | None = None,
) -> LayoutProblemInstance:
    """Extract a `LayoutProblemInstance` from a post-`InferTileMemorySpace` `Program`."""
    model = cost_model or LayoutCostModel()
    cpp_inst = _cpp_al.extract_layout_instance(
        program,
        _to_cpp_cost_model(model),
        set(reusable_tensors) if reusable_tensors else set(),
    )
    return _from_cpp_instance(cpp_inst, cost_model=model)


def evaluate_assignment(
    instance: LayoutProblemInstance,
    assignment: LayoutAssignment,
) -> CostBreakdown:
    """Evaluate a `LayoutAssignment` under the paper's objective (Section 2 of arXiv:2608.21555)."""
    return _from_cpp_breakdown(
        _cpp_al.evaluate_assignment(_to_cpp_instance(instance), _to_cpp_assignment(assignment))
    )


def enumerate_assignments(
    instance: LayoutProblemInstance,
    *,
    deduplicate: bool = True,
) -> list[CostBreakdown]:
    """Enumerate all dataflow-consistent, backend-legal layout assignments, sorted by total cost."""
    return [
        _from_cpp_breakdown(bd)
        for bd in _cpp_al.enumerate_assignments(_to_cpp_instance(instance), deduplicate=deduplicate)
    ]


def apply_layout_assignment(
    program: core_ir.Program,
    assignment: LayoutAssignment,
) -> core_ir.Program:
    """Write a `LayoutAssignment` back into a post-`InferTileMemorySpace` `Program`."""
    return _cpp_al.apply_layout_assignment(program, _to_cpp_assignment(assignment))


def make_auto_layout_pass(
    assignment: LayoutAssignment,
    *,
    name: str = "AutoLayout",
) -> passes.Pass:
    """Wrap `apply_layout_assignment` as a `passes.Pass` insertable after `InferTileMemorySpace`."""
    return passes.auto_layout_from_assignment(_to_cpp_assignment(assignment), name=name)


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
    if (
        cost_model is None
        or type(cost_model).estimate_conversion_cost is LayoutCostModel.estimate_conversion_cost
    ):
        return passes.auto_layout(
            cost_model=_to_cpp_cost_model(cost_model) if cost_model is not None else None,
            reusable_tensors=set(reusable_tensors) if reusable_tensors else None,
            solver=solver,
            max_treewidth=max_treewidth,
            name=name,
        )

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
        return model.estimate_store_cost(
            instance.values[in_names[0]],
            store_dst_node,
            store_out_lay,
            src_layout=in_lays[0] if in_lays else None,
        )
    if op.op_name in _TILE_MATMUL_OPS:
        return model.estimate_matmul_cost(
            instance.values[op.inputs[0]], instance.values[op.inputs[1]], output
        )
    if op.op_name.startswith("tile."):
        work_elements = max(
            [
                output.num_elements,
                *(
                    instance.values[in_name].num_elements
                    for in_name in op.inputs
                    if in_name in instance.values
                    and instance.values[in_name].memory_space != core_ir.MemorySpace.DDR
                ),
            ]
        )
        base = float(work_elements) / model.vec_elements_per_cycle
        if (
            (op.op_name == "tile.row_expand_add" or op.op_name in _OTHER_ROW_EXPAND_OPS)
            and len(config.input_layouts) >= 2
            and config.input_layouts[1] == COL_MAJOR_FLAT
        ):
            ref_shape = instance.values[op.inputs[0]].shape if op.inputs else output.shape
            m_dim = int(ref_shape[-2]) if len(ref_shape) >= 2 else 1
            base += model.estimate_vbrcb_cost(m_dim)
        return base
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
            if src_node.memory_space == core_ir.MemorySpace.Acc:
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

    fitted_op_cycles = [_calibration_prediction_cycles(fitted, measurement) for measurement in measurements]
    y_pred = np.asarray(
        [float(coef[0]) + us_per_cycle * cyc for cyc in fitted_op_cycles],
        dtype=np.float64,
    )
    ss_res = float(np.sum((y_vec - y_pred) ** 2))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float(ss_res <= 1e-12)
    mape = 100.0 * float(np.mean(np.abs(y_vec - y_pred) / np.maximum(np.abs(y_vec), 1e-9)))
    rho = _spearman_rank_correlation([float(yp) for yp in y_pred], rows_y)
    post_fit_within_rhos: list[float] = []
    for idxs in case_groups.values():
        if len(idxs) < 2:
            continue
        grp_ops = [fitted_op_cycles[i] for i in idxs]
        if max(grp_ops) > min(grp_ops):
            post_fit_within_rhos.append(_spearman_rank_correlation(grp_ops, [rows_y[i] for i in idxs]))
    within_op_rho = float(np.median(post_fit_within_rhos)) if post_fit_within_rhos else within_op_rho
    pooled_op_rho = _spearman_rank_correlation(fitted_op_cycles, rows_y)
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
    "ROW_BCAST_FLAT",
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
    "pack_dn",
    "pack_nz",
    "run_post_layout_passes",
    "run_pre_layout_passes",
    "unpack_dn",
    "unpack_nz",
]
