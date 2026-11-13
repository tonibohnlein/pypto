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

#ifndef SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_SCHEDULE_H_
#define SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_SCHEDULE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/physical_sync_commands.h"

namespace pypto::ir::sync::physical {
// Read-only handoff from complete candidate selection to the single materializer.
struct RegularLoopSchedule {
  ForStmtPtr loop;
  int64_t count, start, step, period;
  std::vector<std::pair<int64_t, int64_t>> phases;
  std::vector<std::vector<size_t>> case_cuts;
  std::set<const Call*> anchors;
  size_t payloads = 0;
};
using RegularSchedules = std::map<const ForStmt*, RegularLoopSchedule>;

struct BoundarySchedule {
  std::map<const PhysicalCuts*, std::pair<ControlForest, ControlForest>> cuts;
  ControlForest entry, exit;
};
// A prepared phase contains descriptors, never generated scalar statements.
// Keeping its per-case leaves lets the region select one invocation schedule
// before cloning payloads, without rebuilding endpoint predicates or ID maps.
struct PreparedSchedulePhase {
  int64_t first, end, period;
  bool steady;
  std::vector<CutInventory> visits;
  ControlForest entry, exit;
};
struct InvocationScheduleGroup {
  size_t representative;
  Intervals intervals;
};
struct PreparedLoopSchedule {
  RegularLoopSchedule loop;
  std::string receipt_label;  // Stable original-loop label, not a generated SSA name.
  // Every version already owns its projected cuts and proved boundary motion.
  // Descriptor sharing does not reduce the emitted multiplicity charged here.
  std::vector<std::vector<PreparedSchedulePhase>> versions;
  std::vector<InvocationScheduleGroup> groups;
  bool identity = false;
  unsigned mode = 0;
};
using PreparedLoops = std::map<const ForStmt*, PreparedLoopSchedule>;
using BoundarySchedules = std::map<const ForStmt*, BoundarySchedule>;

// Synchronous compile transaction: borrowed inputs/maps outlive planning and emission.
FunctionPtr CompileAssignedSyncSchedule(const FunctionPtr& owner, const std::vector<AssignedSyncCase>& cases,
                                        std::pair<int64_t, int64_t> blocks, const SyncSchedulePolicy& policy);
// Consume only selected descriptors and the already cached launch coordinate.
// The caller owns one function-wide name reservation through all nested copies.
FunctionPtr MaterializeSelectedSchedule(const FunctionPtr& owner, const ExprPtr& block,
                                        const CutInventory& cuts, const PreparedLoops& loops,
                                        const BoundarySchedules& boundaries,
                                        std::pair<int64_t, int64_t> blocks, PhysicalSyncRealization& work,
                                        SyncTokenNames& names);
ExprPtr InvocationGroupGuard(const ExprPtr& block, std::pair<int64_t, int64_t> domain,
                             const Intervals& intervals, const Span& span);

}  // namespace pypto::ir::sync::physical

#endif  // SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_SCHEDULE_H_
