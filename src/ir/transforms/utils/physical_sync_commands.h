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

#include <array>
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
struct FrozenCommandForest;
struct ControlForest : std::vector<ControlPtr> {
  using std::vector<ControlPtr>::vector;
  mutable std::shared_ptr<const FrozenCommandForest> frozen;
};
struct CommandControl {
  ExprPtr condition, command;
  ControlForest body, alternative;
  Span span;
  // Cached requirements of this node's validated predicate, not readiness
  // evidence. Production must bind them to an original-cut scope below.
  std::vector<const Var*> predicate_inputs;
  // False only for diagnostic pure syntax whose arithmetic premise is absent.
  // Such a predicate stays opaque; production scope qualification rejects it.
  bool predicate_qualified = true;
};
// No generated SSA values are allocated while planning. A slot belongs only to
// one effective sequence scope; definitions appear at their first evaluation.
struct ScalarRecipe {
  ExprPtr original;
  std::shared_ptr<const ScalarRecipe> left, right;
  std::optional<size_t> slot;
};
using ScalarRecipePtr = std::shared_ptr<const ScalarRecipe>;
struct ScalarBinding {
  size_t slot;
  ScalarRecipePtr value;
};
struct FrozenCommand {
  ControlPtr original;
  ScalarRecipePtr condition;
  std::vector<ScalarBinding> definitions;
  std::array<size_t, 6> scalar_cost{};
  std::shared_ptr<const FrozenCommandForest> body, alternative;
};
struct FrozenCommandForest {
  std::vector<ControlPtr> roots;
  std::vector<FrozenCommand> commands;
};
std::shared_ptr<const FrozenCommandForest> FreezeCommandForest(const ControlForest& forest,
                                                               PhysicalSyncRealization& work);
std::vector<StmtPtr> MaterializeFrozenCommands(const ControlForest& forest, SyncTokenNames& names,
                                               PhysicalSyncRealization& work);
struct CommandPredicateScope {
  FunctionPtr owner;
  // Share entry facts once per owner, rather than copying every formal at
  // every cut. Local facts contain only active original loop IVs.
  std::shared_ptr<const std::set<const Var*>> entry_ready;
  std::set<const Var*> ready;
  // Arbitrary SSA definitions and branch-private values are not admitted.
};
void RequireCommandScope(const ControlForest& forest, const CommandPredicateScope& scope,
                         PhysicalSyncRealization& work);
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
  // Present for production inventories. Diagnostic-only imports explicitly
  // supply readiness externally and do not manufacture production evidence.
  std::optional<CommandPredicateScope> predicate_scope;
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
// Rebuilding preserves tree shape. Changed predicates renew their requirements;
// unchanged predicates retain the original qualification, including refusal.
ControlPtr RebuildControl(const ControlPtr& original, ExprPtr condition, ExprPtr command, ControlForest body,
                          ControlForest alternative, PhysicalSyncRealization& work);
void FlattenControlSequence(const ControlPtr& node, ControlForest& commands, PhysicalSyncRealization& work);
// Diagnostic trust boundary: caller supplies ready immutable free variables;
// this still rejects effectful predicates and non-synchronization leaves.
ControlPtr ImportDiagnosticControl(const StmtPtr& stmt, PhysicalSyncRealization& work);
std::vector<StmtPtr> MaterializeControl(const ControlForest& forest, PhysicalSyncRealization& work);
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
