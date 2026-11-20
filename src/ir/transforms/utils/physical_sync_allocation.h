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
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
using SyncDirection = std::pair<PipeType, PipeType>;
using SyncNativeReservations = std::map<SyncDirection, std::set<int>>;

// Record emitted occurrence counts and explicit guarded fallback when no loop
// candidate exists. This census neither changes IR nor reconstructs demands.
void FinishSyncScheduleReceipt(const FunctionPtr& function, PhysicalSyncRealization& result,
                               const SyncSchedulePolicy& policy);

// Qualified direct-cut order provider. The command inventory is independent
// of minimum-demand analysis: token clients retain every actual use. Callbacks
// are consumed synchronously while their immutable owner is alive.
struct DirectSyncOrder {
  // Sufficient C(source) -> I(target) proof, not generic event precedence.
  std::function<bool(const Occurrence&, const Occurrence&)> precedes;
  std::function<bool(const Occurrence&, const Occurrence&)> reference_before;
  // Minimum uniform C(source)->I(target) displacement within one region.
  std::function<std::optional<__int128>(size_t, size_t, size_t)> threshold;
  std::function<bool(size_t)> scoped_reuse, scoped_barrier;
  bool finite_minimum = false;
  std::shared_ptr<const ScopedSyncProtocol> protocol;
  std::function<void(const std::vector<PhysicalSyncFamily>&)> validate_families;
  // One complete directional protocol with typed refusal and construction work.
  // The caller supplies existing eligible capacity, never a new resource pool.
  std::function<ScopedPaletteResult(PipeType, PipeType, int64_t)> scoped_palette;
  // Some native endpoint cuts are Tile operations. Their importer has already
  // collected every local claim using the selected target implementation.
  const SyncNativeReservations* imported_native_claims = nullptr;
  // A conservative plan may complete intervening issues on the same pipe.
  bool allow_stronger_barriers = false;
};
PhysicalSyncRealization RealizeDirectSync(const CompactSyncInput& input,
                                          std::vector<PhysicalSyncFamily> families,
                                          const DirectSyncOrder& order, const std::string& target,
                                          const std::optional<std::vector<int>>& restricted,
                                          const SyncSchedulePolicy& policy = {},
                                          const SyncNativeReservations& extra_reservations = {});

// Separate allocation from shared cut application for mutually exclusive cases.
PhysicalSyncRealization AllocateDirectSync(const CompactSyncInput& input,
                                           std::vector<PhysicalSyncFamily> families,
                                           const DirectSyncOrder& order, const std::string& target,
                                           const std::optional<std::vector<int>>& restricted,
                                           const SyncNativeReservations& extra_reservations = {});
struct AssignedSyncCase {
  const CompactSyncInput* input;
  std::shared_ptr<const AssignedSyncPlan> assignment;
  PhysicalSyncRealization* work;
};
// Borrowed owners live through this synchronous call. All inputs must refer to
// the same original owner and exhaust its supplied invocation domain.
FunctionPtr EmitAssignedSyncCases(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                  std::pair<int64_t, int64_t> blocks,
                                  const std::vector<AssignedSyncCase>& cases,
                                  const SyncSchedulePolicy& policy = {});

// Publish only an established complete physical owner (literal SET and WAIT),
// or an explicit delegation outcome. This does not qualify or allocate a plan.
FunctionPtr SyncOutcome(const FunctionPtr& function, bool pypto, const std::string& reason);

enum class SyncEndpointKind { Source, Target };
enum class SyncEmptyAxes { ContiguousVisits, Singleton };

// Decode a logical record, never a physical ID, into the original occurrence
// namespace. Flat families with no axes advance consecutive visits; a scoped
// Cartesian family with no axes has one record. Callers must name that convention
// rather than silently interpreting missing scoped coordinates as flat visits.
// This is only checked arithmetic: existing providers still qualify the frames,
// guards, endpoint bounds and cuts. Nullopt is unavailable, never occurrence zero.
// Work is O(axis count), independent of trip counts and record ordinal.
inline std::optional<Occurrence> DecodeSyncFamilyEndpoint(const PhysicalSyncFamily& family, int64_t ordinal,
                                                          SyncEndpointKind endpoint,
                                                          SyncEmptyAxes empty_axes) {
  if (family.count <= 0 || ordinal < 0 || ordinal >= family.count ||
      (endpoint != SyncEndpointKind::Source && endpoint != SyncEndpointKind::Target) ||
      (empty_axes != SyncEmptyAxes::ContiguousVisits && empty_axes != SyncEmptyAxes::Singleton))
    return std::nullopt;
  const bool source = endpoint == SyncEndpointKind::Source;
  __int128 visit = source ? family.source_first : family.target_first;
  if (visit < 0) return std::nullopt;
  constexpr auto maximum = std::numeric_limits<int64_t>::max();
  if (family.axes.empty()) {
    if (empty_axes == SyncEmptyAxes::Singleton && family.count != 1) return std::nullopt;
    visit += ordinal;
    if (visit > maximum) return std::nullopt;
  } else {
    __int128 volume = 1;
    for (const auto& axis : family.axes) {
      if (axis.count <= 0 || axis.source_stride < 0 || axis.target_stride < 0) return std::nullopt;
      // Cap every intermediate product: even a very deep malformed map cannot
      // overflow int128 before the final family-count comparison.
      volume *= axis.count;
      if (volume > maximum) return std::nullopt;
    }
    if (volume != family.count) return std::nullopt;
    for (auto axis = family.axes.rbegin(); axis != family.axes.rend(); ++axis) {
      visit += __int128(ordinal % axis->count) * (source ? axis->source_stride : axis->target_stride);
      if (visit > maximum) return std::nullopt;
      ordinal /= axis->count;
    }
    if (ordinal) return std::nullopt;
  }
  return Occurrence{source ? family.source_region : family.target_region, static_cast<int64_t>(visit),
                    source ? family.source_site : family.target_site};
}

// Read the already-qualified endpoint pipe pair; no access recovery or policy selection.
SyncDirection SyncFamilyPipes(const CompactSyncInput& input, const PhysicalSyncFamily& family);

// Shared native-owner and target eligibility policy. Neither helper computes
// demands or assumes that a logical token owns a consuming hardware event.
SyncNativeReservations SyncNativeClaims(const CompactSyncInput& input, const std::string& target);
// Protocol clients have original operations but need no numerical occurrence
// inventory. Keep native reservations and pool eligibility owned by allocation.
SyncNativeReservations SyncNativeClaims(const FunctionPtr& owner, const std::vector<CallPtr>& calls,
                                        const std::string& target);
std::vector<int> SyncEligibleIds(const FunctionPtr& owner, const std::string& target, SyncDirection direction,
                                 const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed);
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
