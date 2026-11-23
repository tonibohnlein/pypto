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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_MATRIX_BYTE_GEOMETRY_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_MATRIX_BYTE_GEOMETRY_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
struct MatrixBoxFootprint;
struct MatrixSubtileFootprint;
struct SyncCanonicalRoot;

bool IsSyncMatrixSpace(MemorySpace space);

// Absolute possible byte runs of a ranged rectangle in a single native slot.
// An unsupported layout, extent, or run count has no qualified projection.
std::optional<std::vector<backend::BufferSyncByteWindow>> TryMatrixByteWindows(const SyncCanonicalRoot& root,
                                                                               const MatrixBoxFootprint& box);

// Physical may-touch windows indexed by normalized loop visit. Each bit in a
// byte atom's mask denotes a visit on which the operation may touch that atom.
struct MatrixVisitFootprint {
  using Mask = uint8_t;
  const ForStmt* frame;
  std::vector<std::vector<backend::BufferSyncByteWindow>> visits;
  Mask VisitsOverlapping(const backend::BufferSyncByteWindow& atom) const;
};
std::optional<MatrixVisitFootprint> TryMatrixVisitFootprint(const SyncCanonicalRoot& root,
                                                            const MatrixBoxFootprint& box,
                                                            const MatrixSubtileFootprint& subtile);

// Change only overlapping placed matrix domains. Original rectangles must have
// been qualified against their own layouts; an unsupported component refuses
// the entire import before any consumer sees the rewritten inventory.
void CanonicalizeMatrixSyncStorage(CompactSyncInput& input);
}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_MATRIX_BYTE_GEOMETRY_H_
