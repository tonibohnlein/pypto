/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_BACKEND_COMMON_BUFFER_VIEW_SEMANTICS_H_
#define PYPTO_BACKEND_COMMON_BUFFER_VIEW_SEMANTICS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

/// Dense physical window size. Unsupported layouts and overflow are unproven.
std::optional<uint64_t> DenseBufferBytes(const ir::BufferTypePtr& type);

/// Physical window size of any byte-addressable rank-1/2 descriptor, including
/// fractal matrix layouts, whose physical extents are already whole boxes.
std::optional<uint64_t> PhysicalBufferBytes(const ir::BufferTypePtr& type);

/// A2/A3 Vec TMOV transfers aligned full columns per valid row in one burst.
/// Runtime rows of a minimum-padded descriptor require a separate use proof.
bool SupportsA2A3VecCopyDescriptor(const ir::BufferTypePtr& type);
bool SupportsLogicalBufferVecCopy(const ir::CallPtr& call);

/// Static native ND Vec insertion. Both widths are physically block aligned;
/// valid columns may end within a block and use the native masked tail.
bool SupportsA2A3VecInsert(const ir::BufferTypePtr& source, const ir::BufferTypePtr& destination,
                           const ir::ExprPtr& row, const ir::ExprPtr& column);

/// Address-independent geometry at a native TPUSH/TPOP site. The caller binds
/// a receive to its reserved ring; a pop does not allocate another local tile.
/// capacity_bytes is the full box, including inactive rows/columns. It bounds
/// one no-split slot; split/converted transfers need their own transport map.
struct PeerTransferBuffer {
  ir::BufferTypePtr descriptor;
  int64_t capacity_bytes;
};

std::optional<PeerTransferBuffer> DescribePeerTransferBuffer(const ir::TileTypePtr& tile);

/// Whole contiguous FP32 vector relabel: column-major [L,1] and row-major
/// [1,L], with full static valid extents and 32-byte alignment. This proves
/// descriptor equivalence only; callers must preserve the original storage
/// window/provenance and exclude inherited noncontiguous slices.
bool IsContiguousVectorReshape(const ir::BufferTypePtr& source, const ir::BufferTypePtr& destination);

/// Physical contiguous FP32 vector geometry, independent of valid metadata.
/// No storage identity, readiness or native metadata ownership is implied.
std::optional<int64_t> LogicalContiguousVectorLength(const ir::TileTypePtr& type);

/// A pure whole-vector axis map [r,1] <-> [1,r], or a same-orientation
/// identity retaining physical shape/layout and both actual valid dimensions.
/// This remains the existing FP32 contiguous-vector geometry (physical L is a
/// positive multiple of eight). Callers must prove original storage provenance
/// and preserve the prefix axis for an identity. No dynamic addressless native
/// reinterpretation or arbitrary two-dimensional reshape is implied.
bool SupportsLogicalContiguousVectorReshape(const ir::CallPtr& call);

/// Static native view forms, with every intermediate descriptor already in IR.
/// Subview slices full-width UINT8[N,32] rows; reshape preserves physical bytes.
/// A matrix-space reshape relabels one fractal window within its memory space.
void ValidateBufferSubview(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result);
void ValidateBufferReshape(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_VIEW_SEMANTICS_H_
