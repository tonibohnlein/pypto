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

#ifndef SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_COMMANDS_H_
#define SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_COMMANDS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
class SyncTokenNames;
}  // namespace pypto::ir::sync

namespace pypto::ir::sync::physical {
// Internal command descriptors retain original coordinates and assigned IDs.
// All implementation visitors and sharing tables remain in the command translation unit.
using Intervals = std::vector<std::pair<int64_t, int64_t>>;
struct CommandControl;
using ControlPtr = std::shared_ptr<const CommandControl>;
using ControlForest = std::vector<ControlPtr>;
struct CommandControl {
  ExprPtr condition, command;
  ControlForest body, alternative;
  Span span;
};
struct CaseCommands {
  std::pair<int64_t, int64_t> blocks;
  ControlPtr body;
};

struct PhysicalCuts {
  ControlForest before, after;
  std::vector<CaseCommands> before_cases, after_cases;
  // Upper bound on the invocation coordinates reaching this ORIGINAL cut.
  // Only a retained zero-visit proof removes coordinates. Missing evidence
  // leaves the full domain; a present cut with no commands is never absence.
  std::vector<std::pair<int64_t, int64_t>> domain;
  std::set<const Var*> uniform_variables;
  bool uniform_checked = false;
};
using CutInventory = std::map<const Call*, std::shared_ptr<PhysicalCuts>>;

bool SameControl(const IRNodePtr& left, const IRNodePtr& right);
bool SameControl(const ControlPtr& left, const ControlPtr& right);
bool CommandValueCandidate(const ExprPtr& expression);
bool PureCommandExpression(const ExprPtr& expression, size_t* inspections);
ControlPtr ControlSequence(ControlForest body, const Span& span, PhysicalSyncRealization& work);
ControlPtr ControlBranch(ExprPtr condition, ControlPtr body, std::optional<ControlPtr> alternative,
                         const Span& span, PhysicalSyncRealization& work);
ControlPtr ControlCommand(ExprPtr command, const Span& span, PhysicalSyncRealization& work);
void FlattenControlSequence(const ControlPtr& node, ControlForest& commands, PhysicalSyncRealization& work);
ControlPtr ImportControl(const StmtPtr& stmt, PhysicalSyncRealization& work);
std::vector<StmtPtr> MaterializeControl(const ControlForest& forest, PhysicalSyncRealization& work);
std::vector<StmtPtr> MaterializeCommandValues(const std::vector<StmtPtr>& commands, SyncTokenNames& names,
                                              PhysicalSyncRealization& work);
// Endpoint accounting applies only before invocation-case composition.
ControlForest ShareCommandGuards(const ControlForest& commands, PhysicalSyncRealization& work,
                                 bool endpoint = true);
void AppendCase(std::vector<CaseCommands>& cases, std::pair<int64_t, int64_t> blocks,
                const ControlForest& commands, const Span& span, PhysicalSyncRealization& work);
Intervals SubtractIntervals(const Intervals& domain, const Intervals& absent);
bool DomainIntersects(const Intervals& domain, int64_t lo, int64_t hi);
ControlForest ScheduleCommandCases(const ExprPtr& block, const Intervals& possible,
                                   const std::vector<CaseCommands>& source, PhysicalSyncRealization& work,
                                   const Span& span, const std::set<const Var*>& uniform_variables);
struct CaseCodeSize {
  size_t commands = 0, guards = 0, expressions = 0, statements = 0;
  explicit CaseCodeSize(const ControlForest& body);
  bool Improves(const CaseCodeSize& old) const;

 private:
  void Visit(const ControlPtr& node);
};

}  // namespace pypto::ir::sync::physical

#endif  // SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_COMMANDS_H_
