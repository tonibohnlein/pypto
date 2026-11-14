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

#ifndef SRC_IR_TRANSFORMS_UTILS_COMPACT_SYNC_NATIVE_H_
#define SRC_IR_TRANSFORMS_UTILS_COMPACT_SYNC_NATIVE_H_

#include <algorithm>
#include <map>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {

// Sparse occurrence counters retain whole-loop multiplicities, never visits.
// Each access/site is indexed once; pair qualification costs logarithmic lookup.
// 128-bit counts cover size_t static sites times signed-64-bit trip counts.
class OccurrenceCounter {
  struct Slice {
    size_t region;
    __int128 before;
    int64_t count;
    std::vector<size_t> sites;
  };
  std::vector<Slice> slices_;

 public:
  void Add(size_t region, int64_t count, size_t site) {
    if (slices_.empty() || slices_.back().region != region) {
      __int128 before = 0;
      if (!slices_.empty()) {
        const auto& last = slices_.back();
        before = last.before + __int128(last.count) * last.sites.size();
      }
      slices_.push_back({region, before, count, {}});
    }
    auto& sites = slices_.back().sites;
    if (sites.empty() || sites.back() != site) sites.push_back(site);
  }
  __int128 Before(const Occurrence& occurrence) const {
    const auto [region, visit, site] = occurrence;
    auto it = std::upper_bound(slices_.begin(), slices_.end(), region,
                               [](size_t r, const Slice& slice) { return r < slice.region; });
    if (it == slices_.begin()) return 0;
    --it;
    if (it->region != region) return it->before + __int128(it->count) * it->sites.size();
    return it->before + __int128(visit) * it->sites.size() +
           (std::lower_bound(it->sites.begin(), it->sites.end(), site) - it->sites.begin());
  }
};
class AccumulatorProtection {
  const CompactSyncInput& input_;
  OccurrenceCounter matrix_;
  std::map<size_t, OccurrenceCounter> qualified_, interference_;

 public:
  explicit AccumulatorProtection(const CompactSyncInput& input) : input_(input) {
    for (size_t r = 0; r < input.segments.size(); ++r) {
      const auto& segment = input.segments[r];
      for (size_t s = 0; s < segment.sites.size(); ++s) {
        const auto& site = segment.sites[s];
        if (site.pipe == PipeType::M) matrix_.Add(r, segment.count, s);
        if (site.protected_accumulator) qualified_[*site.protected_accumulator].Add(r, segment.count, s);
        for (const auto& access : site.accesses)
          if (input.storage[access.storage].space == MemorySpace::Acc &&
              site.protected_accumulator != access.storage)
            interference_[access.storage].Add(r, segment.count, s);
      }
    }
  }
  bool Protects(const Occurrence& a, const Occurrence& b, size_t storage) const {
    const auto [ar, av, as] = a;
    const auto [br, bv, bs] = b;
    const auto& source = input_.segments[ar].sites[as];
    const auto& target = input_.segments[br].sites[bs];
    if (source.protected_accumulator != storage || target.protected_accumulator != storage ||
        (target.initializes_accumulator && (!target.continuation || bv == 0)))
      return false;
    // Every intervening M operation must be qualified on this same physical
    // accumulator. Any other access (including a FIX read) breaks protection.
    // This is NOT a C->I edge: consumers on other pipes still need synchronization.
    const auto& qualified = qualified_.at(storage);
    if (matrix_.Before(b) - matrix_.Before(a) != qualified.Before(b) - qualified.Before(a)) return false;
    auto found = interference_.find(storage);
    return found == interference_.end() || found->second.Before(a) == found->second.Before(b);
  }
};
}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_COMPACT_SYNC_NATIVE_H_
