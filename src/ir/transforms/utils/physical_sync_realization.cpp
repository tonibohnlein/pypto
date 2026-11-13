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

namespace pypto::ir::sync {
using Family = PhysicalSyncFamily;
PhysicalSyncRealization RealizeDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                          const DirectSyncOrder& order, const std::string& target,
                                          const std::optional<std::vector<int>>& restricted,
                                          const SyncSchedulePolicy& policy) {
  auto result = AllocateDirectSync(input, std::move(families), order, target, restricted);
  result.function = physical::CompileAssignedSyncSchedule(
      input.owner, {{&input, result.assignment, &result}},
      input.block_bounds.value_or(std::make_pair(static_cast<int64_t>(0), static_cast<int64_t>(0))), policy);
  return result;
}
FunctionPtr EmitAssignedSyncCases(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                  std::pair<int64_t, int64_t> blocks,
                                  const std::vector<AssignedSyncCase>& cases,
                                  const SyncSchedulePolicy& policy) {
  CHECK(owner && !cases.empty()) << "Missing complete invocation case owner";
  auto next = blocks.first;
  for (const auto& item : cases) {
    CHECK(item.input && item.assignment && item.work && item.input->owner == owner &&
          item.input->parameter_bounds == parameters && item.input->block_bounds &&
          item.input->block_bounds->first == next && item.input->block_bounds->second >= next &&
          item.input->block_bounds->second < std::numeric_limits<int64_t>::max())
        << "Incomplete invocation case contract";
    next = item.input->block_bounds->second + 1;
  }
  CHECK(next == blocks.second + 1) << "Incomplete invocation activation domain";
  return physical::CompileAssignedSyncSchedule(owner, cases, blocks, policy);
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
