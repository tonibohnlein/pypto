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

#ifndef SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_HIERARCHY_H_
#define SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_HIERARCHY_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
namespace pypto::ir::sync {
// Construct once on a fresh result: original-access directory for selectors
// across enclosing frames. Does not rebuild mathematics or mutate source facts.
void IndexSyncStorageAccesses(CompactSyncResult& result);
// Exact first/last flattened ordinals for one qualified original affine access.
// The interior need not be contiguous across resets; callers must not use the
// pair as a membership interval. Absence means no access in this qualified domain.
std::optional<std::pair<int64_t, int64_t>> OriginalAccessPointVisits(const SyncAccess& access,
                                                                     const SyncSegment& segment, int64_t row,
                                                                     int64_t column);
CompactSyncResult AnalyzeInvariantSyncLeaves(CompactSyncInput input);
CompactSyncResult AnalyzeNumericalSync(CompactSyncInput input, const std::string& order);
bool NumericalReferenceBefore(const CompactSyncResult& result, const Occurrence& a, const Occurrence& b);
bool NumericalPrecedes(const CompactSyncResult& result, const Occurrence& a, bool sc, const Occurrence& b,
                       bool tc);
std::optional<PointBoundary> QueryExternalNumericalBoundary(const CompactSyncResult& result, size_t storage,
                                                            int64_t slot, int64_t row, int64_t column,
                                                            const std::map<size_t, int64_t>& parameters,
                                                            std::optional<int64_t> block_index);
PointBoundary QueryNumericalBoundary(const CompactSyncResult& result, size_t storage, int64_t slot,
                                     int64_t row, int64_t column);
std::optional<std::vector<Crossing>> ExportNumericalCrossings(const CompactSyncResult& result);
std::shared_ptr<const ScopedSyncDemands> ExportNumericalScopedDemands(const CompactSyncResult& result);
std::vector<Crossing> ExpandNumericalDemands(const CompactSyncResult& result, size_t maximum);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_HIERARCHY_H_
