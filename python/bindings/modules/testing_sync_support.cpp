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

#include "python/bindings/modules/testing_sync_support.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/scoped_logical_sync.h"
#include "src/ir/transforms/utils/sync_token_names.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::python::sync_testing {
using ir::As;
using ir::EvalStmt;
using ir::Function;
using ir::IfStmt;
using ir::ParamDirection;
using ir::SeqStmts;
using ir::TypePtr;
using ir::VarPtr;
using ir::sync::LowerLogicalSync;
using ir::sync::OptimizeLogicalSync;
using ir::sync::RealizeCompactSync;
using ir::sync::RealizeLogicalInvocation;
using ir::sync::SyncTokenNames;
using ir::sync::physical::CaseCommands;
using ir::sync::physical::ControlBranch;
using ir::sync::physical::ControlCommand;
using ir::sync::physical::ControlForest;
using ir::sync::physical::ControlPtr;
using ir::sync::physical::ControlSequence;
using ir::sync::physical::Intervals;
using ir::sync::physical::MaterializeFrozenCommands;
using ir::sync::physical::ScheduleCommandCases;
using ir::sync::physical::ShareCommandGuards;
PhysicalSyncRealization RealizeDemandPlan(const CompactSyncResult& analysis, const std::string& target,
                                          const std::optional<std::vector<int>>& eligible_ids,
                                          const SyncSchedulePolicy& policy) {
  CHECK(analysis.State().input.hardware_target == target)
      << "Synchronization hardware contract differs from lowering target";
  auto logical = RealizeCompactSync(analysis);
  auto lowered =
      LowerLogicalSync(OptimizeLogicalSync(logical.function), target, eligible_ids,
                       analysis.State().input.parameter_bounds, analysis.State().input.block_bounds, policy);
  CHECK(lowered.physical_cases.size() == 1)
      << "Single-region test realization needs one physical case; use the invocation adapter";
  auto result = std::move(lowered.physical_cases.front());
  result.function = std::move(lowered.function);
  return result;
}
DemandInvocation RealizeDemandInvocation(const SyncInvocationAnalysis& analysis, const std::string& target,
                                         const std::optional<std::vector<int>>& eligible_ids,
                                         const SyncSchedulePolicy& policy) {
  for (const auto& item : analysis.cases)
    CHECK(item.mathematics && item.mathematics->State().input.hardware_target == target)
        << "Synchronization hardware contract differs from lowering target";
  auto logical = RealizeLogicalInvocation(analysis);
  auto lowered =
      LowerLogicalSync(OptimizeLogicalSync(logical.function), target, eligible_ids, analysis.parameter_bounds,
                       analysis.block_bounds, policy, ir::sync::SyncExternalPolicy::Geometry,
                       analysis.coordinate ? std::make_optional(analysis.coordinate->Count()) : std::nullopt);
  return {std::move(lowered.function), std::move(lowered.physical_cases)};
}

ControlPtr ImportCommands(const StmtPtr& stmt, PhysicalSyncRealization& work) {
  if (auto sequence = As<SeqStmts>(stmt)) {
    ControlForest children;
    for (const auto& child : sequence->stmts_) children.push_back(ImportCommands(child, work));
    return ControlSequence(std::move(children), stmt->span_, work);
  }
  if (auto branch = As<IfStmt>(stmt)) {
    CHECK(branch->return_vars_.empty()) << "Command schedule cannot move definitions";
    return ControlBranch(branch->condition_, ImportCommands(branch->then_body_, work),
                         branch->else_body_
                             ? std::optional<ControlPtr>(ImportCommands(*branch->else_body_, work))
                             : std::nullopt,
                         stmt->span_, work);
  }
  auto command = As<EvalStmt>(stmt);
  CHECK(command) << "Command schedule requires command-only input";
  return ControlCommand(command->expr_, stmt->span_, work);
}

namespace {
// Seed names from all supplied fragments. This owner is only a lexical naming
// scope, never evidence of readiness, matching, physical reuse or payload order.
FunctionPtr CommandOwner(const std::vector<StmtPtr>& commands) {
  return std::make_shared<Function>("test_commands", std::vector<VarPtr>{}, std::vector<ParamDirection>{},
                                    std::vector<TypePtr>{},
                                    std::make_shared<SeqStmts>(commands, Span::unknown()), Span::unknown());
}
}  // namespace
StmtPtr MaterializeValues(const FunctionPtr& owner, const StmtPtr& commands) {
  SyncTokenNames names(owner);
  PhysicalSyncRealization work;
  return std::make_shared<SeqStmts>(MaterializeFrozenCommands({ImportCommands(commands, work)}, names, work),
                                    commands->span_);
}
std::vector<StmtPtr> ShareGuards(const std::vector<StmtPtr>& commands, PhysicalSyncRealization& work) {
  ControlForest forest;
  for (const auto& command : commands) forest.push_back(ImportCommands(command, work));
  SyncTokenNames names(CommandOwner(commands));
  return MaterializeFrozenCommands(ShareCommandGuards(forest, work), names, work);
}
std::vector<StmtPtr> EmitCases(const ExprPtr& block, const Intervals& possible,
                               const std::vector<CommandCase>& cases, PhysicalSyncRealization& work,
                               const Span& span) {
  std::vector<CaseCommands> descriptors;
  std::vector<StmtPtr> bodies;
  for (const auto& item : cases) {
    descriptors.push_back({item.blocks, ImportCommands(item.body, work)});
    bodies.push_back(item.body);
  }
  // Include the dispatch coordinate when reserving generated names.
  bodies.push_back(std::make_shared<EvalStmt>(block, span));
  SyncTokenNames names(CommandOwner(bodies));
  return MaterializeFrozenCommands(ScheduleCommandCases(block, possible, descriptors, work, span, {}), names,
                                   work);
}
std::pair<std::vector<Intervals>, PhysicalSyncRealization> InspectGroups(
    const FunctionPtr& owner, const std::vector<CommandCase>& supplied) {
  PhysicalSyncRealization work;
  std::vector<CaseCommands> cases;
  for (const auto& c : supplied) cases.push_back({c.blocks, ImportCommands(c.body, work)});
  return InspectCommandGroups(owner, cases, std::move(work));
}
PhysicalSyncRealization EmitBoundaries(const FunctionPtr& owner,
                                       const std::vector<std::tuple<CallPtr, StmtPtr, StmtPtr>>& supplied,
                                       bool enabled, bool validate_scope) {
  PhysicalSyncRealization work;
  std::vector<std::tuple<CallPtr, ControlForest, ControlForest>> cuts;
  for (const auto& [call, before, after] : supplied)
    cuts.push_back({call, {ImportCommands(before, work)}, {ImportCommands(after, work)}});
  return EmitCommandBoundaries(owner, cuts, enabled, validate_scope, std::move(work));
}
}  // namespace pypto::python::sync_testing
