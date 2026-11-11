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

#ifndef SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_ALLOCATION_H_
#define SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_ALLOCATION_H_

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
using SyncDirection = std::pair<PipeType, PipeType>;
using SyncNativeReservations = std::map<SyncDirection, std::set<int>>;

// Shared native-owner and target eligibility policy. Neither helper computes
// demands or assumes that a logical token owns a consuming hardware event.
SyncNativeReservations SyncNativeClaims(const CompactSyncInput& input, const std::string& target);
std::vector<int> SyncEligibleIds(const CompactSyncInput& input, const std::string& target,
                                 SyncDirection direction, const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed);

// Existing monotone lifetime-chain sweep. Lanes must already be sorted by
// publication. For one direction, consumptions form a chain and each legal
// reuse set is a publication suffix. The callbacks retain their proof owner.
// O(L log L) ordering plus O(L) reuse queries; query cost is charged separately.
std::vector<int> ColorOrderedSyncLanes(size_t count, SyncDirection direction,
                                       const std::function<bool(size_t, size_t)>& wait_before,
                                       const std::function<bool(size_t, size_t)>& reusable,
                                       const std::vector<int>& eligible, bool finite_minimum);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_ALLOCATION_H_
