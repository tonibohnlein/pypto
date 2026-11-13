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

#include "testing_sync_support.h"

#include <limits>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::python::sync_testing {
using namespace ir::sync::physical;
namespace {
using Family = PhysicalSyncFamily;
std::vector<Family> DirectFamilies(const CompactSyncResult& analysis) {
  std::vector<Family> families;
  if (analysis.input.region_tree) {
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
      families.push_back({sr, ss, tr, ts, sv, tv, static_cast<int64_t>(count), {}, schema.axes});
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
        families.push_back({r, f.source, r, f.target, 0, f.distance, segment.count - f.distance, {}});
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
      families.push_back({sr, ss, tr, ts, sv, tv, 1, {}});
    }
  }
  return families;
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
PhysicalSyncRealization RealizeDemandPlan(const CompactSyncResult& analysis, const std::string& target,
                                          const std::optional<std::vector<int>>& eligible_ids,
                                          const SyncSchedulePolicy& policy) {
  return RealizeDirectSync(analysis.input, DirectFamilies(analysis), MathematicalOrder(analysis), target,
                           eligible_ids, policy);
}
DemandInvocation RealizeDemandInvocation(const SyncInvocationAnalysis& analysis, const std::string& target,
                                         const std::optional<std::vector<int>>& eligible_ids,
                                         const SyncSchedulePolicy& policy) {
  // Qualify executable case selection before spending work on allocation.
  CHECK(analysis.cases.size() <= 1 || DominatingSyncBlock(analysis.owner))
      << "Original dominating invocation block binding is unavailable";
  DemandInvocation result{nullptr, {}};
  result.cases.reserve(analysis.cases.size());
  std::vector<AssignedSyncCase> cases;
  for (const auto& item : analysis.cases) {
    CHECK(item.mathematics &&
          item.mathematics->input.block_bounds == std::make_pair(item.block_first, item.block_last))
        << "Missing original invocation analysis";
    result.cases.push_back(AllocateDirectSync(item.mathematics->input, DirectFamilies(*item.mathematics),
                                              MathematicalOrder(*item.mathematics), target, eligible_ids));
    cases.push_back({&item.mathematics->input, result.cases.back().assignment, &result.cases.back()});
  }
  result.function =
      EmitAssignedSyncCases(analysis.owner, analysis.parameter_bounds, analysis.block_bounds, cases, policy);
  return result;
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
