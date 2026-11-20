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

#include "src/ir/transforms/utils/peer_gm_invocation.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/peer_local_effects.h"

namespace pypto::ir::sync {
namespace {

bool IsPlainTensor(const VarPtr& value) {
  const auto tensor = value ? As<TensorType>(value->GetType()) : nullptr;
  return tensor && !tensor->tensor_view_;
}

bool IsManagedPlaceholder(const TypePtr& type) {
  const auto tensor = As<TensorType>(type);
  return tensor && !tensor->tensor_view_ && tensor->dtype_ == DataType::FP32 && tensor->shape_.size() == 1 &&
         transform_utils::EvalConstInt(tensor->shape_.front()) == 1;
}

bool IsManagedCreate(const AssignStmtPtr& statement) {
  const auto call = As<Call>(statement->value_);
  if (!alloc_batching::IsInjectedGMPipeCreateVar(statement->var_) || !IsOp(call, "tensor.create") ||
      call->args_.size() != 1 || !IsManagedPlaceholder(statement->var_->GetType()) ||
      !IsManagedPlaceholder(call->GetType())) {
    return false;
  }
  const auto shape = As<MakeTuple>(call->args_.front());
  return shape && shape->elements_.size() == 1 &&
         transform_utils::EvalConstInt(shape->elements_.front()) == 1 &&
         call->GetKwarg<DataType>("dtype", DataType::FP32) == DataType::FP32 &&
         call->GetKwarg<TensorLayout>("layout", TensorLayout::ND) == TensorLayout::ND &&
         call->GetKwarg<bool>("manual_dep", false);
}

enum class OriginKind { Unknown, Formal, Fresh };
struct Origin {
  const Var* value = nullptr;
  OriginKind kind = OriginKind::Unknown;
  AssignStmtPtr create;
};
struct Site {
  Span span;
  // A direct statement inside a sequential, optionally scoped root path.
  bool qualified;
  StmtPtr statement;
  std::vector<Origin> arguments;
};

const Expr* GmIdentity(const Origin& origin) {
  const auto tensor = As<TensorType>(origin.value->GetType());
  const auto memref = tensor->memref_.value_or(nullptr);
  return memref && memref->base_ ? memref->base_.get() : origin.value;
}

// A root inventory is cached across the prospective Groups it invokes. Allow
// explicitly scoped tensor.create bindings to dominate later calls in the same
// scope. Control outside this admitted form is still walked to find calls, so
// an unmodeled invocation cannot silently inherit the proof from another site.
class RootInventory final : public IRVisitor {
 public:
  RootInventory(const FunctionPtr& root, const std::set<std::string>& groups) : groups_(groups) {
    for (const auto& formal : root->params_) {
      formal_names_.insert(formal->name_hint_);
      if (IsPlainTensor(formal)) {
        formals_.insert(formal.get());
      }
    }
    VisitStmt(root->body_);
    visited_.clear();
    visited_.rehash(0);
  }

  void Check(const ProgramPtr& program, const FunctionPtr& group, const std::set<size_t>& ordinary_formals,
             size_t workspace_formal) const {
    const auto found = calls_.find(group->name_);
    CHECK(found != calls_.end()) << "Peer Group invocation lacks a visible root call";
    for (const auto& site : found->second) {
      CHECK_SPAN(site.qualified && site.arguments.size() == group->params_.size(), site.span)
          << "Peer Group GM proof requires a direct root invocation with explicit tensor actuals";
      CheckSite(program, group, site, ordinary_formals, workspace_formal);
    }
  }

  // Check after both endpoint imports and their ordinary GM provenance checks.
  // These sets are projected through the actual bindings of every launch.
  void CheckScalarReads(const FunctionPtr& group, const PeerGmEffects& effects, bool disjoint_gm) const {
    for (const auto& site : calls_.at(group->name_)) {
      std::unordered_set<const Expr*> write_roots;
      for (const auto formal : effects.writes) {
        write_roots.insert(GmIdentity(site.arguments[formal]));
      }
      for (const auto formal : effects.scalar_reads) {
        CHECK_SPAN(!write_roots.count(GmIdentity(site.arguments[formal])), site.span)
            << "Peer scalar GM read aliases a GM write in the mixed dispatch";
      }
      CHECK_SPAN(disjoint_gm || write_roots.empty(), site.span)
          << "Peer scalar GM read may alias a GM write without a disjoint GM contract";
    }
  }

  void VisitStmt(const StmtPtr& statement) override {
    const bool previous = qualified_;
    const auto previous_statement = statement_;
    statement_ = statement;
    qualified_ &= As<SeqStmts>(statement) || As<RuntimeScopeStmt>(statement) || As<AssignStmt>(statement) ||
                  As<EvalStmt>(statement);
    IRVisitor::VisitStmt(statement);
    statement_ = previous_statement;
    qualified_ = previous;
  }

  void VisitExpr(const ExprPtr& expression) override {
    if (const auto var = As<Var>(expression)) {
      ++uses_[var.get()];
    } else if (expression && (expression.get() == direct_ || visited_.insert(expression.get()).second)) {
      // Read a direct RHS at every statement cut. A nested expression is
      // traversed once, which suffices to refuse every nested Group use.
      IRVisitor::VisitExpr(expression);
    }
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& assignment) override {
    VisitDirect(assignment->value_);
    const auto& variable = assignment->var_;
    ++definitions_[variable.get()];
    if (qualified_ && sequence_ && !formal_names_.count(variable->name_hint_) &&
        IsManagedCreate(assignment) &&
        active_creates_.try_emplace(variable.get(), sequence_, assignment).second) {
      scoped_creates_.push_back(variable.get());
    }
  }

  void VisitStmt_(const EvalStmtPtr& evaluation) override { VisitDirect(evaluation->expr_); }

  void VisitStmt_(const SeqStmtsPtr& sequence) override {
    const auto* parent = sequence_;
    sequence_ = sequence.get();
    IRVisitor::VisitStmt_(sequence);
    sequence_ = parent;
  }

  void VisitStmt_(const RuntimeScopeStmtPtr& scope) override {
    const size_t entry = scoped_creates_.size();
    const auto* parent = sequence_;
    sequence_ = nullptr;
    IRVisitor::VisitStmt_(scope);
    sequence_ = parent;
    while (scoped_creates_.size() > entry) {
      active_creates_.erase(scoped_creates_.back());
      scoped_creates_.pop_back();
    }
  }

  void VisitExpr_(const CallPtr& call) override {
    Record(call, call->op_, call->args_);
    IRVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    Record(submit, submit->op_, submit->args_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  const std::set<std::string>& groups_;
  bool qualified_ = true;
  const Expr* direct_ = nullptr;
  const SeqStmts* sequence_ = nullptr;
  StmtPtr statement_;
  std::set<std::string> formal_names_;
  std::unordered_set<const Expr*> visited_;
  std::unordered_set<const Var*> formals_;
  std::unordered_map<const Var*, std::pair<const SeqStmts*, AssignStmtPtr>> active_creates_;
  std::vector<const Var*> scoped_creates_;
  std::unordered_map<const Var*, size_t> uses_, definitions_;
  std::map<std::string, std::vector<Site>> calls_;

  void VisitDirect(const ExprPtr& expression) {
    direct_ = expression.get();
    VisitExpr(expression);
    direct_ = nullptr;
  }

  Origin Bind(const ExprPtr& expression) const {
    const auto value = As<Var>(expression);
    if (formals_.count(value.get())) {
      return {value.get(), OriginKind::Formal, nullptr};
    }
    const auto create = active_creates_.find(value.get());
    if (create != active_creates_.end() && create->second.first == sequence_) {
      return {value.get(), OriginKind::Fresh, create->second.second};
    }
    return {};
  }

  void Record(const ExprPtr& expression, const OpPtr& op, const std::vector<ExprPtr>& args) {
    const auto global = As<GlobalVar>(op);
    if (!global || !groups_.count(global->name_)) {
      return;
    }
    Site site{expression->span_, qualified_ && expression.get() == direct_, statement_, {}};
    if (site.qualified) {
      for (const auto& arg : args) {
        site.arguments.push_back(Bind(arg));
      }
    }
    calls_[global->name_].push_back(std::move(site));
  }

  void CheckSite(const ProgramPtr& program, const FunctionPtr& group, const Site& site,
                 const std::set<size_t>& ordinary_formals, size_t workspace_formal) const {
    std::unordered_set<const Var*> origins;
    for (const auto formal : ordinary_formals) {
      const auto& actual = site.arguments[formal];
      CHECK_SPAN(actual.kind == OriginKind::Formal && !definitions_.count(actual.value), site.span)
          << "Peer Group GM tensor must forward an original root formal";
      origins.insert(actual.value);  // Ordinary GM aliases already share a local storage domain.
    }
    const auto& workspace = site.arguments[workspace_formal];
    CHECK_SPAN(!origins.count(workspace.value), site.span)
        << "Peer Group managed workspace aliases an ordinary root tensor";
    const auto uses = uses_.find(workspace.value);
    const auto definitions = definitions_.find(workspace.value);
    // The emitter sizes only its recognized placeholder and searches for its
    // concrete launch among following statements in the same sequence.
    CHECK_SPAN(workspace.kind == OriginKind::Fresh && uses != uses_.end() && uses->second == 1 &&
                   definitions != definitions_.end() && definitions->second == 1,
               site.span)
        << "Peer Group managed workspace requires a unique injected root tensor.create";
    // Both statements are direct siblings and the allocation has one use.
    // The omitted siblings cannot redefine or consume it, so the allocator
    // observes the same first consumer on this two-statement subsequence.
    const auto use = alloc_batching::ResolveGMPipeCreateUse({workspace.create, site.statement}, 0,
                                                            workspace.create->var_, program, {});
    CHECK_SPAN(use && use->callee == group, site.span)
        << "Peer Group is not the workspace allocation's concrete sizing consumer";
    CHECK_SPAN(!use->core_num || transform_utils::EvalConstInt(use->core_num) == 1, site.span)
        << "Peer Group managed workspace requires a single root launch";
  }
};

}  // namespace

struct PeerInvocationGmEvidence::State {
  ProgramPtr program;
  std::set<std::string> groups;
  std::map<std::string, std::set<std::string>> callers;
  std::map<std::string, std::unique_ptr<RootInventory>> roots;

  State(const ProgramPtr& owner, const SyncInvocationEvidence& invocation) : program(owner) {
    for (const auto& [global, function] : program->functions_) {
      if (function->func_type_ == FunctionType::Group) {
        groups.insert(global->name_);
      }
    }
    for (const auto& [caller, children] : invocation.callees) {
      for (const auto& callee : children) {
        callers[callee].insert(caller);
      }
    }
  }
};

PeerInvocationGmEvidence::PeerInvocationGmEvidence(const ProgramPtr& program,
                                                   const SyncInvocationEvidence& invocation)
    : state_(std::make_unique<State>(program, invocation)) {}

PeerInvocationGmEvidence::~PeerInvocationGmEvidence() = default;

const std::set<std::string>& PeerInvocationGmEvidence::CallersOf(const std::string& callee) const {
  static const std::set<std::string> empty;
  const auto found = state_->callers.find(callee);
  return found == state_->callers.end() ? empty : found->second;
}

void PeerInvocationGmEvidence::Check(const FunctionPtr& group, const std::set<size_t>& ordinary_formals,
                                     size_t workspace_formal) {
  CHECK(group && workspace_formal < group->params_.size() && IsPlainTensor(group->params_[workspace_formal]))
      << "Peer Group has no plain managed workspace formal";
  for (const auto formal : ordinary_formals) {
    CHECK(formal < group->params_.size() && formal != workspace_formal &&
          IsPlainTensor(group->params_[formal]))
        << "Peer Group has no distinct ordinary GM tensor formal";
  }
  const auto& callers = CallersOf(group->name_);
  CHECK(!callers.empty()) << "Peer Group has no root managed workspace allocation";
  for (const auto& caller : callers) {
    const auto root = state_->program->GetFunction(caller);
    CHECK(root && root->func_type_ == FunctionType::Orchestration && CallersOf(caller).empty())
        << "Peer Group GM forwarding requires a direct root orchestration";
    auto& inventory = state_->roots[caller];
    if (!inventory) {
      inventory = std::make_unique<RootInventory>(root, state_->groups);
    }
    inventory->Check(state_->program, group, ordinary_formals, workspace_formal);
  }
}

void PeerInvocationGmEvidence::CheckScalarReads(const FunctionPtr& group, const PeerGmEffects& effects,
                                                bool disjoint_gm) const {
  if (effects.scalar_reads.empty() || effects.writes.empty()) {
    return;
  }
  for (const auto& caller : CallersOf(group->name_)) {
    state_->roots.at(caller)->CheckScalarReads(group, effects, disjoint_gm);
  }
}

}  // namespace pypto::ir::sync
