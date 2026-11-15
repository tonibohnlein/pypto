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

#include "src/ir/transforms/utils/retained_sync_child.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"

namespace pypto::ir::sync {
namespace {
class Operations : public IRVisitor {
 public:
  Operations(const CompactSyncResult& mathematics, const std::string& target, NativeSyncChildExport& result)
      : mathematics_(mathematics), target_(target), result_(result) {
    for (const auto& segment : mathematics.State().input.segments)
      for (const auto& site : segment.sites) {
        modeled_.insert(site.original.get());
        if (site.continuation) modeled_.insert(site.continuation.get());
      }
  }
  std::vector<backend::BufferSyncOperation> payloads;
  bool available = true;

  void VisitStmt(const StmtPtr& statement) override {
    if (!available) return;
    if (!(As<SeqStmts>(statement) || As<AssignStmt>(statement) || As<EvalStmt>(statement) ||
          As<IfStmt>(statement) || As<ForStmt>(statement) || As<YieldStmt>(statement))) {
      Refuse("Child contains an unsupported scope or invocation exit");
      return;
    }
    IRVisitor::VisitStmt(statement);
  }
  void VisitExpr(const ExprPtr& expression) override {
    if (!available || !expression || !seen_.insert(expression.get()).second) return;
    if (!(AsVarLike(expression) || As<ConstInt>(expression) || As<ConstFloat>(expression) ||
          As<ConstBool>(expression) || As<BinaryExpr>(expression) || As<UnaryExpr>(expression) ||
          As<MakeTuple>(expression) || As<TupleGetItemExpr>(expression) || As<Call>(expression))) {
      Refuse("Child expression has no qualified execution/effect contract");
      return;
    }
    IRVisitor::VisitExpr(expression);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loop->kind_ != ForKind::Sequential) {
      Refuse("Child needs a sequential execution-domain contract after scheduling");
      return;
    }
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitExpr_(const CallPtr& call) override {
    ++result_.operations;
    if (const auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
      INTERNAL_CHECK(descriptor->SignalFree()) << "Qualified descriptor owns a signal";
    } else {
      if (!modeled_.count(call.get()) || !call->attrs_.empty()) {
        Refuse("Child call is outside retained payload mathematics or has unqualified attributes");
        return;
      }
      auto operation =
          backend::QualifyBufferSyncOperation(call, mathematics_.State().input.owner->func_type_, target_);
      for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
        if (operation.ArgumentEffects()[index].data == BufferAccess::None) continue;
        const auto type = call->args_[index]->GetType();
        const auto buffer = As<BufferType>(type);
        if (!buffer || buffer->memory_space_ == MemorySpace::DDR) {
          result_.status = NativeSyncChildStatus::AccessPremises;
          result_.reason = "Native child needs an external-storage alias premise not supplied to its parent";
          available = false;
          return;
        }
      }
      if (!operation.Implementation().Claims().empty()) {
        result_.status = NativeSyncChildStatus::NativeResources;
        result_.reason = "Native child implementation owns flags; resource-empty import is unavailable";
        available = false;
        return;
      }
      for (const auto& effect : operation.Implementation().ImplicitEffects()) {
        if (effect.space != MemorySpace::Vec || effect.begin < 0 || effect.end <= effect.begin) {
          result_.status = NativeSyncChildStatus::AccessPremises;
          result_.reason = "Native child implicit effects lack a qualified local storage domain";
          available = false;
          return;
        }
      }
      // Retain the selected implementation's implicit inventory as well as
      // operand effects. The parent recovers the same whole-space summary;
      // notification closure does not discard pending native scratch effects.
      payloads.push_back(std::move(operation));
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  void Refuse(const char* reason) {
    result_.status = NativeSyncChildStatus::UnqualifiedOperation;
    result_.reason = reason;
    available = false;
  }
  const CompactSyncResult& mathematics_;
  const std::string& target_;
  NativeSyncChildExport& result_;
  std::unordered_set<const Call*> modeled_;
  std::unordered_set<const Expr*> seen_;
};
}  // namespace

const char* NativeSyncChildStatusName(NativeSyncChildStatus status) {
  switch (status) {
    case NativeSyncChildStatus::Ready:
      return "ready";
    case NativeSyncChildStatus::ForeignContext:
      return "foreign_context";
    case NativeSyncChildStatus::MissingRetainedScope:
      return "missing_retained_scope";
    case NativeSyncChildStatus::UnavailableSummary:
      return "unavailable_summary";
    case NativeSyncChildStatus::NonNativeDemands:
      return "nonnative_internal_demands";
    case NativeSyncChildStatus::UnqualifiedOperation:
      return "unqualified_operation";
    case NativeSyncChildStatus::NativeResources:
      return "native_resources";
    case NativeSyncChildStatus::AccessPremises:
      return "access_premises";
  }
  INTERNAL_CHECK(false) << "Internal error: invalid native child status";
  return "";
}

void NativeSyncChild::ValidateFor(const FunctionPtr& owner, const std::string& target) const {
  CHECK(owner == Owner()) << "Retained native child belongs to a different original function/context";
  CHECK(target == target_ && mathematics_.State().input.hardware_target == target)
      << "Retained native child has a different native target";
}

NativeSyncChildExport ExportNativeSyncChild(const CompactSyncResult& mathematics, const ForStmtPtr& scope,
                                            const std::string& target) {
  NativeSyncChildExport result;
  const auto& state = mathematics.State();
  if (!state.input.owner || target != state.input.hardware_target) {
    result.status = NativeSyncChildStatus::ForeignContext;
    result.reason = "Child target/context differs from the retained mathematics";
    return result;
  }
  if (!state.input.parameter_bounds.empty() || state.input.block_bounds) {
    result.status = NativeSyncChildStatus::AccessPremises;
    result.reason = "Child caller/launch range premises are not supplied to the conservative parent";
    return result;
  }
  if (!scope || !state.hierarchy || !state.hierarchy->source_layout) {
    result.reason = "Provider has no original numerical source scope for this request";
    return result;
  }
  const NumericalSourceLayout* selected = nullptr;
  std::vector<const NumericalSourceLayout*> pending{&*state.hierarchy->source_layout};
  while (!pending.empty()) {
    const auto* node = pending.back();
    pending.pop_back();
    ++result.source_nodes;
    if (node->frame == scope || (node->leaf && state.input.segments.at(*node->leaf).original_loop == scope))
      selected = node;
    for (const auto& child : node->children) pending.push_back(&child);
  }
  if (!selected) {
    result.reason = "Requested loop has no retained provider scope in this original function";
    return result;
  }
  if (!selected->summary) {
    result.status = NativeSyncChildStatus::UnavailableSummary;
    result.reason = "The phase provider did not retain an independent summary for this inner scope";
    return result;
  }
  // Inspect immutable retained reduction records, not expanded occurrences or
  // a newly constructed hazard graph. Sharing in repeated summaries is retained.
  std::vector<std::shared_ptr<const NumericalNode>> summaries{selected->summary};
  std::unordered_set<const NumericalNode*> visited;
  while (!summaries.empty()) {
    const auto node = summaries.back();
    summaries.pop_back();
    if (!node || !node->length || !visited.insert(node.get()).second) continue;
    ++result.summary_nodes;
    bool nonnative = false;
    for (const auto& crossing : node->crossings) {
      ++result.demands;
      nonnative |= !crossing.native;
    }
    if (node->left) {
      summaries.push_back(node->left);
      summaries.push_back(node->right);
    } else if (const auto provider = state.hierarchy->finite_providers.find(node->leaf);
               provider != state.hierarchy->finite_providers.end()) {
      result.demands += provider->second.order.demands.size();
      nonnative |= !provider->second.order.demands.empty();
    } else {
      const auto& leaf = state.segments.at(node->leaf);
      for (const auto& demand : leaf.demands) {
        ++result.demands;
        nonnative |= demand.distance < leaf.internal_scope_count;
      }
    }
    if (nonnative) {
      result.status = NativeSyncChildStatus::NonNativeDemands;
      result.reason = "Retained child has nonnative internal demands; closure/palette export is required";
      return result;
    }
  }
  Operations operations(mathematics, target, result);
  operations.VisitStmt(scope);
  if (!operations.available) return result;
  result.child = std::shared_ptr<const NativeSyncChild>(
      new NativeSyncChild(mathematics, scope, target, std::move(operations.payloads)));
  result.status = NativeSyncChildStatus::Ready;
  result.reason = "Retained internal order is native-only, with no notification resources";
  return result;
}
}  // namespace pypto::ir::sync
