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
#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
namespace pypto::ir::sync {
CompactSyncResult AnalyzeInvariantSyncLeaves(CompactSyncInput input);
CompactSyncResult AnalyzeNumericalSync(CompactSyncInput input, const std::string& order);
bool NumericalReferenceBefore(const CompactSyncResult& result, const Occurrence& a, const Occurrence& b);
bool CertifyNumericalAdjacentBarrier(const CompactSyncResult& result, size_t family);
bool CertifyNumericalConsecutiveReuse(const CompactSyncResult& result, size_t family);
bool NumericalPrecedes(const CompactSyncResult& result, const Occurrence& a, bool sc, const Occurrence& b,
                       bool tc);
PointBoundary QueryNumericalBoundary(const CompactSyncResult& result, size_t storage, int64_t slot,
                                     int64_t row, int64_t column);
std::optional<std::vector<Crossing>> ExportNumericalCrossings(const CompactSyncResult& result);
std::shared_ptr<const ScopedSyncDemands> ExportNumericalScopedDemands(const CompactSyncResult& result);
std::vector<Crossing> ExpandNumericalDemands(const CompactSyncResult& result, size_t maximum);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_HIERARCHY_H_
