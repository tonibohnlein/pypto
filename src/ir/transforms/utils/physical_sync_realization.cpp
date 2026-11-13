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

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/physical_sync_schedule.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"

namespace pypto::ir::sync {
namespace {
using Family = PhysicalSyncFamily;
std::vector<Family> DirectFamilies(const CompactSyncResult& analysis) {
  PhysicalSyncRealization result_;
  if (analysis.input.region_tree) {
    result_.finite_minimum = false;  // Sufficient whole-span assignment, not a minimum-ID proof.
    const auto owned = analysis.ExportScopedDemands();
    CHECK(owned) << "Scoped endpoint export is unavailable";
    for (const auto& schema : owned->families) {
      const auto [sr, sv, ss] = schema.source;
      const auto [tr, tv, ts] = schema.target;
      __int128 count = 1;
      for (const auto& axis : schema.axes) {
        count *= axis.count;
        CHECK(count <= std::numeric_limits<int64_t>::max()) << "Scoped physical record domain overflow";
      }
      result_.families.push_back({sr, ss, tr, ts, sv, tv, static_cast<int64_t>(count), {}, schema.axes});
    }
  } else {
    const auto crossings = analysis.ExportCrossingDemands();
    CHECK(crossings.has_value())
        << "Minimum demands are retained; shared-repetition physical endpoint scope is unavailable";
    for (size_t r = 0; r < analysis.segments.size(); ++r) {
      const auto& segment = analysis.segments[r];
      for (const auto& f : segment.demands) {
        CHECK(f.distance >= 0 && f.source < segment.pipes.size() && f.target < segment.pipes.size())
            << "Invalid physical demand family";
        if (segment.count <= f.distance) continue;
        result_.families.push_back({r, f.source, r, f.target, 0, f.distance, segment.count - f.distance, {}});
      }
    }
    for (const auto& [source, target] : *crossings) {
      auto [sr, sv, ss] = source;
      auto [tr, tv, ts] = target;
      CHECK(sr < analysis.segments.size() && tr < analysis.segments.size() && sr < tr)
          << "Unavailable physical crossing frame";
      CHECK(sv >= 0 && sv < analysis.segments[sr].count && tv >= 0 && tv < analysis.segments[tr].count &&
            ss < analysis.segments[sr].pipes.size() && ts < analysis.segments[tr].pipes.size())
          << "Invalid physical crossing occurrence";
      result_.families.push_back({sr, ss, tr, ts, sv, tv, 1, {}});
    }
  }
  return std::move(result_.families);
}
DirectSyncOrder MathematicalOrder(const CompactSyncResult& analysis) {
  return {[&](const Occurrence& a, const Occurrence& b) { return analysis.Precedes(a, true, b, false); },
          [&](const Occurrence& a, const Occurrence& b) { return analysis.ReferenceBefore(a, b); },
          [&](size_t r, size_t a, size_t b) { return analysis.segments[r].Threshold(a, true, b, false); },
          [&](size_t i) { return analysis.ScopedConsecutiveReuse(i); },
          [&](size_t i) { return analysis.ScopedAdjacentBarrier(i); },
          true};
}

}  // namespace
PhysicalSyncRealization RealizeDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                          const DirectSyncOrder& order, const std::string& target,
                                          const std::optional<std::vector<int>>& restricted,
                                          const SyncSchedulePolicy& policy) {
  auto result = AllocateDirectSync(input, std::move(families), order, target, restricted);
  result.function = physical::CompileAssignedSyncSchedule(
      input.owner, {{&input, &result}},
      input.block_bounds.value_or(std::make_pair(static_cast<int64_t>(0), static_cast<int64_t>(0))), policy);
  return result;
}
PhysicalSyncRealization RealizePhysicalSync(const CompactSyncResult& analysis, const std::string& target,
                                            const std::optional<std::vector<int>>& eligible_ids,
                                            const SyncSchedulePolicy& policy) {
  return RealizeDirectSync(analysis.input, DirectFamilies(analysis), MathematicalOrder(analysis), target,
                           eligible_ids, policy);
}
FunctionPtr EmitAssignedSyncCases(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                  std::pair<int64_t, int64_t> blocks,
                                  const std::vector<AssignedSyncCase>& cases,
                                  const SyncSchedulePolicy& policy) {
  CHECK(owner && !cases.empty()) << "Missing complete invocation case owner";
  auto next = blocks.first;
  for (const auto& item : cases) {
    CHECK(item.input && item.plan && item.input->owner == owner &&
          item.input->parameter_bounds == parameters && item.input->block_bounds &&
          item.input->block_bounds->first == next && item.input->block_bounds->second >= next &&
          item.input->block_bounds->second < std::numeric_limits<int64_t>::max())
        << "Incomplete invocation case contract";
    next = item.input->block_bounds->second + 1;
  }
  CHECK(next == blocks.second + 1) << "Incomplete invocation activation domain";
  return physical::CompileAssignedSyncSchedule(owner, cases, blocks, policy);
}

PhysicalSyncInvocation RealizePhysicalInvocation(const SyncInvocationAnalysis& analysis,
                                                 const std::string& target,
                                                 const std::optional<std::vector<int>>& eligible_ids,
                                                 const SyncSchedulePolicy& policy) {
  // Qualify executable case selection before spending work on allocation.
  CHECK(analysis.cases.size() <= 1 || DominatingSyncBlock(analysis.owner))
      << "Original dominating invocation block binding is unavailable";
  PhysicalSyncInvocation result{analysis, nullptr, {}};
  result.cases.reserve(analysis.cases.size());
  std::vector<AssignedSyncCase> cases;
  for (const auto& item : analysis.cases) {
    CHECK(item.mathematics &&
          item.mathematics->input.block_bounds == std::make_pair(item.block_first, item.block_last))
        << "Missing original invocation analysis";
    result.cases.push_back(AllocateDirectSync(item.mathematics->input, DirectFamilies(*item.mathematics),
                                              MathematicalOrder(*item.mathematics), target, eligible_ids));
    cases.push_back({&item.mathematics->input, &result.cases.back()});
  }
  result.function =
      EmitAssignedSyncCases(analysis.owner, analysis.parameter_bounds, analysis.block_bounds, cases, policy);
  return result;
}

FunctionPtr SyncOutcome(const FunctionPtr& function, bool pypto, const std::string& reason) {
  auto attrs = function->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& attr) {
                               return attr.first == kAttrPyptoAutosync ||
                                      attr.first == kAttrAutosyncFallbackReason;
                             }),
              attrs.end());
  attrs.emplace_back(kAttrPyptoAutosync, pypto);
  if (!pypto) attrs.emplace_back(kAttrAutosyncFallbackReason, reason);
  return std::make_shared<Function>(function->name_, function->params_, function->param_directions_,
                                    function->return_types_, function->body_, function->span_,
                                    function->func_type_, function->level_, function->role_, attrs,
                                    function->requires_runtime_binding_, function->ir_stage_);
}

}  // namespace pypto::ir::sync
