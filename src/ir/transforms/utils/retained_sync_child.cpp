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

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
#include "pypto/ir/transforms/utils/signature_symbols.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"

namespace pypto::ir::sync {
namespace {
class Operations : public IRVisitor {
 public:
  Operations(const CompactSyncResult& mathematics, const std::string& target, NativeSyncChildExport& result,
             bool allow_disjoint_gm = false)
      : mathematics_(mathematics),
        target_(target),
        result_(result),
        allow_disjoint_gm_(allow_disjoint_gm) {
    for (const auto& segment : mathematics.State().input.segments)
      for (const auto& site : segment.sites) {
        modeled_.insert(site.original.get());
        if (site.continuation) modeled_.insert(site.continuation.get());
      }
  }
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
  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitExpr(branch->condition_);
    const auto& static_branches = mathematics_.State().input.static_branches;
    if (const auto resolved = static_branches.find(branch.get()); resolved != static_branches.end()) {
      if (resolved->second)
        VisitStmt(branch->then_body_);
      else if (branch->else_body_)
        VisitStmt(*branch->else_body_);
      return;
    }
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
  }
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
        const bool projected = mathematics_.State().input.unavailable_storage_spaces.count(MemorySpace::DDR);
        const bool external = As<TensorType>(type) || (buffer && buffer->memory_space_ == MemorySpace::DDR);
        if (external && (allow_disjoint_gm_ ||
                         (projected && operation.ArgumentEffects()[index].data == BufferAccess::Read)))
          continue;
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
        if (effect.space == MemorySpace::DDR &&
            (allow_disjoint_gm_ || (mathematics_.State().input.unavailable_storage_spaces.count(MemorySpace::DDR) &&
                                    effect.data == BufferAccess::Read)))
          continue;
        if (effect.space != MemorySpace::Vec || effect.begin < 0 || effect.end <= effect.begin) {
          result_.status = NativeSyncChildStatus::AccessPremises;
          result_.reason = "Native child implicit effects lack a qualified local storage domain";
          available = false;
          return;
        }
      }
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
  bool allow_disjoint_gm_ = false;
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
  if (!state.input.unavailable_storage_spaces.empty()) {
    result.status = NativeSyncChildStatus::AccessPremises;
    result.reason = "Native-only child export lacks an external-domain contract for projected storage";
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
  result.child = std::shared_ptr<const NativeSyncChild>(new NativeSyncChild(mathematics, scope, target));
  result.status = NativeSyncChildStatus::Ready;
  result.reason = "Retained internal order is native-only, with no notification resources";
  return result;
}
bool HasReadonlyExternalSyncEffects(const ForStmtPtr& scope, FunctionType core, const std::string& target,
                                    SyncChildEffectWork* work, bool allow_disjoint_gm_writes) {
  class Effects : public IRVisitor {
   public:
    Effects(FunctionType core, const std::string& target, SyncChildEffectWork* work,
            bool allow_disjoint_gm_writes)
        : core_(core),
          target_(target),
          work_(work),
          allow_disjoint_gm_writes_(allow_disjoint_gm_writes) {}
    bool external = false;
    bool has_write = false;
    void VisitStmt(const StmtPtr& statement) override {
      if (work_) ++work_->statements;
      IRVisitor::VisitStmt(statement);
    }
    void VisitExpr(const ExprPtr& expression) override {
      if (expression && seen_.insert(expression.get()).second) {
        if (work_) ++work_->expressions;
        IRVisitor::VisitExpr(expression);
      }
    }
    void VisitExpr_(const CallPtr& call) override {
      if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None ||
          backend::QualifyBufferSyncNonPayload(call, target_))
        return;
      if (work_) ++work_->native_qualifications;
      const auto operation = backend::QualifyBufferSyncOperation(call, core_, target_);
      for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
        const auto effect = operation.ArgumentEffects()[index].data;
        if (effect == BufferAccess::None) continue;
        const auto type = call->args_[index]->GetType();
        const auto buffer = As<BufferType>(type);
        if (As<TensorType>(type) || (buffer && buffer->memory_space_ == MemorySpace::DDR)) {
          if (effect != BufferAccess::Read) {
            CHECK(allow_disjoint_gm_writes_) << "Closed child external effects are not uniformly read-only";
            has_write = true;
          }
          external = true;
        }
      }
      for (const auto& effect : operation.Implementation().ImplicitEffects()) {
        if (effect.space != MemorySpace::DDR) continue;
        if (effect.data != BufferAccess::Read) {
          CHECK(allow_disjoint_gm_writes_) << "Closed child implicit external effect is not read-only";
          has_write = true;
        }
        external = true;
      }
    }

   private:
    FunctionType core_;
    const std::string& target_;
    std::unordered_set<const Expr*> seen_;
    SyncChildEffectWork* work_;
    bool allow_disjoint_gm_writes_;
  } effects(core, target, work, allow_disjoint_gm_writes);
  effects.VisitStmt(scope);
  return effects.external && !effects.has_write;
}

SyncParameterBounds ProjectSyncParameterBounds(const FunctionPtr& owner, const FunctionPtr& binding_owner,
                                               const SyncParameterBounds& owner_bounds) {
  SyncParameterBounds projected;
  if (!owner || !binding_owner || owner_bounds.empty()) return projected;
  std::unordered_map<const Var*, std::pair<int64_t, int64_t>> by_var;
  for (const auto& [idx, range] : owner_bounds) {
    if (idx < owner->params_.size()) {
      by_var.emplace(owner->params_[idx].get(), range);
    }
  }
  if (by_var.empty()) return projected;
  for (size_t c_idx = 0; c_idx < binding_owner->params_.size(); ++c_idx) {
    const auto found = by_var.find(binding_owner->params_[c_idx].get());
    if (found != by_var.end()) projected.emplace(c_idx, found->second);
  }
  return projected;
}

class SyncRegionBindings::State : public ParentStmtAnalysis {
 public:
  FunctionPtr owner;
  std::string target;
  size_t clock = 0;
  std::unordered_map<const Stmt*, size_t> occurrences, begin, end;
  std::unordered_map<const Var*, std::vector<AssignStmtPtr>> definitions;
  std::unordered_map<const Var*, std::vector<std::pair<StmtPtr, VarPtr>>> return_var_definitions;
  // Ephemeral request-local qualification cache. Published mathematics and
  // capabilities do not depend on later mutations of this index.
  std::map<std::pair<const AssignStmt*, SyncExternalPolicy>, bool> qualified;
  std::shared_ptr<const SyncRegionIndexWork> work;
  std::map<const Var*, std::pair<VarPtr, ParamDirection>> ready_parameters;

  State(FunctionPtr function, std::string native) : owner(std::move(function)), target(std::move(native)) {
    CHECK(owner) << "Regional synchronization requires an original owner";
    BuildMap(owner);
    // The existing signature service identifies physical tensor extents supplied
    // by the invocation ABI. This is readiness, never a numerical bound or a
    // valid-shape-only/tensor.read visibility premise.
    const auto physical_symbols = CollectParamDimSymbols(owner->params_);
    for (size_t index = 0; index < owner->params_.size(); ++index)
      ready_parameters.emplace(owner->params_[index].get(),
                               std::make_pair(owner->params_[index], owner->param_directions_[index]));
    for (const auto& symbol : ClassifySignatureSymbols(owner).bound)
      if (physical_symbols.count(symbol.get()))
        ready_parameters.emplace(symbol.get(), std::make_pair(symbol, ParamDirection::In));
    work = std::make_shared<const SyncRegionIndexWork>(SyncRegionIndexWork{clock, ready_parameters.size()});
  }
  void VisitStmt(const StmtPtr& statement) override {
    ++occurrences[statement.get()];
    begin.emplace(statement.get(), clock++);
    if (const auto definition = As<AssignStmt>(statement)) {
      definitions[definition->var_.get()].push_back(definition);
    } else {
      auto record_return_vars = [&](const std::vector<VarPtr>& return_vars) {
        for (const auto& rv : return_vars) {
          return_var_definitions[rv.get()].emplace_back(statement, rv);
        }
      };
      if (const auto branch = As<IfStmt>(statement)) {
        record_return_vars(branch->return_vars_);
      } else if (const auto loop = As<ForStmt>(statement)) {
        record_return_vars(loop->return_vars_);
      } else if (const auto loop = As<WhileStmt>(statement)) {
        record_return_vars(loop->return_vars_);
      }
    }
    ParentStmtAnalysis::VisitStmt(statement);
    end[statement.get()] = clock;
  }
  void VisitExpr(const ExprPtr&) override {}
  bool Dominates(const StmtPtr& definition, const StmtPtr& use) const {
    const auto parent = GetParent(definition);
    // A binding in a sequence dominates precisely the later sibling subtrees
    // in that sequence. Its interval ends at the original branch/loop scope;
    // a definition from another arm or a prior iteration cannot leak out.
    return parent && As<SeqStmts>(parent) && occurrences.at(definition.get()) == 1 &&
           occurrences.at(parent.get()) == 1 && end.at(definition.get()) <= begin.at(use.get()) &&
           begin.at(use.get()) < end.at(parent.get());
  }
  bool Qualify(const AssignStmtPtr& assignment, SyncExternalPolicy policy) {
    const auto key = std::make_pair(assignment.get(), policy);
    if (const auto found = qualified.find(key); found != qualified.end()) {
      return found->second;
    }
    const auto call = As<Call>(assignment->value_);
    bool ready;
    if (call) {
      // Completed facts have no physical notification; keep actual initializers
      // for the token parser without promoting them to payload completion.
      const auto descriptor = GetLogicalSyncKind(call->op_) == LogicalSyncKind::Completed
                                  ? std::optional<backend::BufferSyncOperation>{}
                                  : backend::QualifyBufferSyncNonPayload(call, target);
      ready = GetLogicalSyncKind(call->op_) == LogicalSyncKind::Completed ||
              (descriptor && descriptor->SignalFree());
    } else {
      ready = As<ConstInt>(assignment->value_) || As<ConstBool>(assignment->value_) ||
              As<ConstFloat>(assignment->value_) || AsVarLike(assignment->value_);
      if (policy == SyncExternalPolicy::ReadOnlyProjection) {
        CheckSyncScalarReadiness(assignment->value_, [](const VarPtr&) { return true; });
        ready = true;  // Every operand is separately bound at this original cut below.
      }
    }
    qualified.emplace(key, ready);
    return ready;
  }
};

SyncRegionBindings::SyncRegionBindings(FunctionPtr owner, std::string target)
    : state_(std::make_shared<State>(std::move(owner), std::move(target))) {}

size_t SyncRegionBindings::SourceNodes() const { return state_->work->source_nodes; }
size_t SyncRegionBindings::ParameterNodes() const { return state_->work->parameter_nodes; }

SyncRegionContext SyncRegionBindings::Bind(const ForStmtPtr& scope, SyncExternalPolicy policy,
                                           SyncRegionBindWork* work) const {
  CHECK(scope && state_->occurrences.count(scope.get()) && state_->occurrences.at(scope.get()) == 1)
      << "Regional synchronization scope must occur exactly once in its original owner";
  std::vector<StmtPtr> ancestors;
  for (auto parent = state_->GetParent(scope); parent; parent = state_->GetParent(parent))
    ancestors.push_back(parent);
  std::reverse(ancestors.begin(), ancestors.end());
  if (work) work->ancestor_links += ancestors.size();
  class Uses : public var_collectors::VarDefUseCollector {
   public:
    explicit Uses(SyncRegionBindWork* work) : work_(work) {}
    size_t scope_nodes = 0;
    void VisitStmt(const StmtPtr& statement) override {
      ++scope_nodes;
      if (work_) ++work_->scope_nodes;
      IRVisitor::VisitStmt(statement);
    }
    void VisitExpr(const ExprPtr& expression) override {
      if (expression && seen_.insert(expression.get()).second) {
        if (work_) ++work_->expression_nodes;
        IRVisitor::VisitExpr(expression);
      }
    }

   private:
    std::unordered_set<const Expr*> seen_;
    SyncRegionBindWork* work_;
  } uses(work);
  uses.VisitStmt(scope);
  std::vector<std::pair<const Var*, StmtPtr>> pending;
  for (const auto* variable : uses.var_uses)
    if (!uses.var_defs.count(variable)) pending.emplace_back(variable, scope);
  for (const auto* variable : uses.GetAllVarRefs())
    for (const auto* metadata : var_collectors::CollectTypeVars(variable->GetType()))
      if (!uses.var_defs.count(metadata)) pending.emplace_back(metadata, scope);
  std::set<const Var*> required;
  std::vector<AssignStmtPtr> ordered;
  std::map<const Var*, std::pair<VarPtr, ParamDirection>> captures;
  std::map<const Var*, std::pair<VarPtr, ParamDirection>> available;
  std::map<const Var*, ForStmtPtr> invocation_frames;
  if (policy == SyncExternalPolicy::ReadOnlyProjection) {
    for (const auto& ancestor : ancestors) {
      if (const auto loop = As<ForStmt>(ancestor)) {
        available.emplace(loop->loop_var_.get(), std::make_pair(loop->loop_var_, ParamDirection::In));
        invocation_frames.emplace(loop->loop_var_.get(), loop);
      }
    }
  }
  while (!pending.empty()) {
    const auto [variable, use] = pending.back();
    pending.pop_back();
    // Invocation parameters are indexed once, not copied into every candidate.
    // Only actual referenced captures contribute to a binding's output size.
    const auto parameter = state_->ready_parameters.find(variable);
    if (parameter != state_->ready_parameters.end()) {
      captures.emplace(variable, parameter->second);
      continue;
    }
    if (const auto captured = available.find(variable); captured != available.end()) {
      if (const auto frame = invocation_frames.find(variable); frame != invocation_frames.end()) {
        const auto& loop = frame->second;
        CHECK(state_->occurrences.at(loop.get()) == 1 &&
              state_->begin.at(loop.get()) < state_->begin.at(use.get()) &&
              state_->begin.at(use.get()) < state_->end.at(loop.get()))
            << "Opaque enclosing coordinate is unavailable at its original definition/use cut";
      }
      captures.emplace(variable, captured->second);
      continue;
    }
    const auto found = state_->definitions.find(variable);
    if (found == state_->definitions.end() && policy == SyncExternalPolicy::ReadOnlyProjection) {
      const auto rv_found = state_->return_var_definitions.find(variable);
      if (rv_found != state_->return_var_definitions.end() && rv_found->second.size() == 1) {
        const auto& [def_stmt, rv_var] = rv_found->second.front();
        if (As<ScalarType>(rv_var->GetType())) {
          CHECK(state_->Dominates(def_stmt, use))
              << "Regional synchronization descriptor binding does not dominate its original use";
          captures.emplace(variable, std::make_pair(rv_var, ParamDirection::In));
          continue;
        }
      }
    }
    CHECK(found != state_->definitions.end() && found->second.size() == 1)
        << "Regional synchronization child has an unavailable enclosing coordinate, caller value or "
           "descriptor binding: "
        << variable->name_hint_;
    const auto& definition = found->second.front();
    // Check every source-use cut, including repeated uses of a shared binding.
    // Dominance of the child alone cannot justify an earlier initializer use.
    CHECK(state_->Dominates(definition, use))
        << "Regional synchronization descriptor binding does not dominate its original use";
    CHECK(state_->Qualify(definition, policy))
        << "Regional synchronization requires a signal-free original binding";
    if (!required.insert(variable).second) continue;
    ordered.push_back(definition);
    if (work) ++work->binding_definitions;
    Uses dependencies(work);
    dependencies.VisitExpr(definition->value_);
    for (const auto* dependency : dependencies.var_uses) pending.emplace_back(dependency, definition);
    for (const auto* metadata : var_collectors::CollectTypeVars(variable->GetType()))
      pending.emplace_back(metadata, definition);
  }
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
    return state_->begin.at(a.get()) < state_->begin.at(b.get());
  });
  std::vector<StmtPtr> body(ordered.begin(), ordered.end());
  body.push_back(scope);
  body.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, scope->span_));
  auto binding = std::make_shared<Function>(*state_->owner);
  binding->params_.clear();
  binding->param_directions_.clear();
  for (const auto& [identity, capture] : captures) {
    binding->params_.push_back(capture.first);
    binding->param_directions_.push_back(capture.second);
  }
  binding->return_types_.clear();
  binding->body_ = std::make_shared<SeqStmts>(body, scope->span_);
  return {state_->owner,    scope,         std::move(ancestors), std::move(binding), state_->work,
          uses.scope_nodes, ordered.size()};
}

EventfulSyncChildAttempt TryPrepareEventfulSyncChild(
    const FunctionPtr& owner, const ForStmtPtr& scope, const std::string& target,
    const SyncRegionBindings& bindings, const SyncParameterBounds& parameters,
    const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  EventfulSyncChildAttempt attempt;
  attempt.scope = scope;
  auto started = std::chrono::steady_clock::now();
  auto finish_phase = [&] {
    const auto now = std::chrono::steady_clock::now();
    attempt.seconds[attempt.stage] += std::chrono::duration<double>(now - started).count();
    started = now;
  };
  auto enter_phase = [&](EventfulSyncChildStage next) {
    finish_phase();
    attempt.stage = next;
  };
  try {
    const auto* ctx = PassContext::Current();
    const bool disjoint_gm = ctx && ctx->GetAutoSyncDisjointGM();
    const auto policy = HasReadonlyExternalSyncEffects(scope, owner->func_type_, target,
                                                       &attempt.effect_work, disjoint_gm)
                            ? SyncExternalPolicy::ReadOnlyProjection
                            : SyncExternalPolicy::Geometry;
    enter_phase(EventfulSyncChildStage::Binding);
    auto context = bindings.Bind(scope, policy, &attempt.binding_work);
    CHECK(context.original_owner == owner)
        << "Regional synchronization index belongs to another original owner";
    enter_phase(EventfulSyncChildStage::Analysis);
    const auto child_bounds = ProjectSyncParameterBounds(owner, context.binding_owner, parameters);
    attempt.mathematics = std::make_shared<const CompactSyncResult>(
        AnalyzeCompactSync(policy == SyncExternalPolicy::ReadOnlyProjection
                               ? ImportReadonlyExternalSync(context.binding_owner, target, child_bounds, blocks)
                               : ImportPlacedSync(context.binding_owner, true, target, child_bounds, blocks)));
    const auto& mathematics = *attempt.mathematics;
    enter_phase(EventfulSyncChildStage::Qualification);
    NativeSyncChildExport qualification;
    Operations operations(mathematics, target, qualification, disjoint_gm);
    operations.VisitStmt(scope);
    CHECK(operations.available) << "Eventful child native/effect qualification unavailable: "
                                << qualification.reason;
    enter_phase(EventfulSyncChildStage::ExecutableExport);
    const auto realized = RealizeCompactSync(mathematics);
    attempt.logical_record_sites = realized.record_sites;
    attempt.logical_wait_sites = realized.wait_sites;
    auto logical = realized.function;
    class FindLogical : public IRVisitor {
     public:
      explicit FindLogical(const VarPtr& coordinate) : coordinate_(coordinate) {}
      ForStmtPtr found;

     protected:
      void VisitStmt_(const ForStmtPtr& loop) override {
        if (loop->loop_var_ == coordinate_) {
          CHECK(!found) << "Eventful child has ambiguous original loop coordinates";
          found = loop;
        }
        IRVisitor::VisitStmt_(loop);
      }
      void VisitExpr(const ExprPtr&) override {}

     private:
      VarPtr coordinate_;
    } selected(scope->loop_var_);
    selected.VisitFunction(logical);
    CHECK(selected.found) << "Eventful child logical recipe lost its original invocation frame";
    CheckLogicalSync(logical);
    const SyncRegionBindings logical_bindings(logical, target);
    attempt.binding_work.logical_index_nodes = logical_bindings.SourceNodes();
    const auto logical_context = logical_bindings.Bind(selected.found, policy, &attempt.binding_work);
    std::vector<StmtPtr> recipe;
    const auto bound = As<SeqStmts>(logical_context.binding_owner->body_);
    INTERNAL_CHECK(bound) << "Internal error: region bindings are not a sequence";
    for (const auto& statement : bound->stmts_) {
      const auto assignment = As<AssignStmt>(statement);
      if (assignment && As<SyncTokenType>(assignment->var_->GetType())) recipe.push_back(statement);
    }
    auto marked = std::make_shared<ForStmt>(*selected.found);
    CHECK(!marked->HasAttr(kClosedSyncChildAttr))
        << "Original child already contains synchronization routing metadata";
    marked->attrs_.emplace_back(kClosedSyncChildAttr, 1);
    marked->attrs_.emplace_back(
        kClosedSyncChildExternalPolicyAttr,
        std::string(policy == SyncExternalPolicy::ReadOnlyProjection ? "readonly_projection" : "geometry"));
    recipe.push_back(marked);
    auto logical_recipe = std::make_shared<SeqStmts>(recipe, scope->span_);
    std::set<SyncDirection> rearm_dirs;
    class CollectRearms : public IRVisitor {
     public:
      explicit CollectRearms(std::set<SyncDirection>& out) : out_(out) {}

     protected:
      void VisitExpr(const ExprPtr& expression) override {
        if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
      }
      void VisitExpr_(const CallPtr& call) override {
        if (IsOp(call, "sync.wait") && !call->args_.empty()) {
          if (const auto tok = As<SyncTokenType>(call->args_[0]->GetType())) {
            for (const auto& [name, value] : call->kwargs_) {
              if (name == "dst_pipe") {
                const auto dst = static_cast<PipeType>(AnyCast<int>(value, name));
                if (tok->pipe_ != dst) out_.emplace(dst, tok->pipe_);
                break;
              }
            }
          }
        }
        IRVisitor::VisitExpr_(call);
      }

     private:
      std::set<SyncDirection>& out_;
      std::unordered_set<const Expr*> seen_;
    } collector(rearm_dirs);
    collector.VisitStmt(logical_recipe);
    for (const auto& [dir, ids] : SyncNativeClaims(mathematics.State().input, target)) {
      (void)ids;
      if (dir.first != dir.second) rearm_dirs.emplace(dir.second, dir.first);
    }
    attempt.child = std::shared_ptr<const EventfulSyncChild>(
        new EventfulSyncChild(std::move(context), mathematics, logical_recipe, target,
                              std::vector<SyncDirection>(rearm_dirs.begin(), rearm_dirs.end())));
    enter_phase(EventfulSyncChildStage::Ready);
  } catch (const pypto::ValueError& error) {
    finish_phase();
    attempt.reason = error.what();
  }
  return attempt;
}

std::shared_ptr<const EventfulSyncChild> PrepareEventfulSyncChild(const FunctionPtr& owner,
                                                                  const ForStmtPtr& scope,
                                                                  const std::string& target,
                                                                  const SyncRegionBindings& bindings) {
  auto attempt = TryPrepareEventfulSyncChild(owner, scope, target, bindings);
  CHECK(attempt.child) << attempt.reason;
  return std::move(attempt.child);
}

const char* EventfulSyncChildStageName(EventfulSyncChildStage stage) {
  switch (stage) {
    case EventfulSyncChildStage::Effects:
      return "effects";
    case EventfulSyncChildStage::Binding:
      return "binding";
    case EventfulSyncChildStage::Analysis:
      return "analysis";
    case EventfulSyncChildStage::Qualification:
      return "qualification";
    case EventfulSyncChildStage::ExecutableExport:
      return "executable_export";
    case EventfulSyncChildStage::Ready:
      return "logical_export_ready";
  }
  INTERNAL_CHECK(false) << "Internal error: unknown eventful child attempt stage";
  return "";
}

AutomaticSyncChildren PrepareAutomaticSyncChildren(
    const FunctionPtr& owner, const std::string& target, const SyncParameterBounds& parameters,
    const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  // O(N) bottom-up syntactic pre-screen filters outer loops containing
  // While/Parallel/Return/Submit or empty bodies without building Bind/Analysis.
  struct ScopeSummary {
    bool has_payload = false;
    bool has_disqualifying_control = false;
    bool has_ddr_write = false;
    bool has_unqualified_call = false;
    void MergeFrom(const ScopeSummary& other) {
      has_payload |= other.has_payload;
      has_disqualifying_control |= other.has_disqualifying_control;
      has_ddr_write |= other.has_ddr_write;
      has_unqualified_call |= other.has_unqualified_call;
    }
  };
  std::unordered_map<const ForStmt*, ScopeSummary> loop_summaries;
  std::function<ScopeSummary(const StmtPtr&)> prescreen = [&](const StmtPtr& stmt) -> ScopeSummary {
    ScopeSummary sum;
    if (!stmt) return sum;
    if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) sum.MergeFrom(prescreen(child));
      return sum;
    }
    if (const auto branch = As<IfStmt>(stmt)) {
      sum.MergeFrom(prescreen(branch->then_body_));
      if (branch->else_body_) sum.MergeFrom(prescreen(*branch->else_body_));
      return sum;
    }
    if (const auto loop = As<ForStmt>(stmt)) {
      const auto body_sum = prescreen(loop->body_);
      loop_summaries.emplace(loop.get(), body_sum);
      sum = body_sum;
      if (loop->kind_ != ForKind::Sequential) sum.has_disqualifying_control = true;
      return sum;
    }
    if (const auto loop = As<WhileStmt>(stmt)) {
      prescreen(loop->body_);
      sum.has_disqualifying_control = true;
      return sum;
    }
    if (As<ReturnStmt>(stmt)) {
      sum.has_disqualifying_control = true;
      return sum;
    }
    auto scan_call = [&](const CallPtr& call) {
      if (!call || GetLogicalSyncKind(call->op_) != LogicalSyncKind::None) return;
      try {
        if (backend::QualifyBufferSyncNonPayload(call, target)) return;
        sum.has_payload = true;
        const auto operation = backend::QualifyBufferSyncOperation(call, owner->func_type_, target);
        for (size_t i = 0; i < operation.ArgumentEffects().size(); ++i) {
          const auto eff = operation.ArgumentEffects()[i].data;
          if (eff == BufferAccess::None || eff == BufferAccess::Read) continue;
          const auto type = call->args_[i]->GetType();
          const auto buf = As<BufferType>(type);
          if (As<TensorType>(type) || (buf && buf->memory_space_ == MemorySpace::DDR)) {
            sum.has_ddr_write = true;
          }
        }
        for (const auto& eff : operation.Implementation().ImplicitEffects()) {
          if (eff.space == MemorySpace::DDR && eff.data != BufferAccess::Read) sum.has_ddr_write = true;
        }
      } catch (const pypto::ValueError&) {
        sum.has_unqualified_call = true;
      }
    };
    if (const auto eval = As<EvalStmt>(stmt)) {
      if (const auto call = As<Call>(eval->expr_))
        scan_call(call);
      else if (As<Submit>(eval->expr_))
        sum.has_disqualifying_control = true;
    } else if (const auto assign = As<AssignStmt>(stmt)) {
      if (const auto call = As<Call>(assign->value_))
        scan_call(call);
      else if (As<Submit>(assign->value_))
        sum.has_disqualifying_control = true;
    }
    return sum;
  };
  prescreen(owner->body_);

  const auto* ctx = PassContext::Current();
  const bool disjoint_gm = ctx && ctx->GetAutoSyncDisjointGM();

  // Top-down maximal exact subregion discovery (Draft Section 8): attempt to
  // promote each enclosing sequential ForStmt subtree as a single Exact child
  // before descending into inner loops only when the enclosing scope fails Exact.
  AutomaticSyncChildren result;
  std::optional<SyncRegionBindings> bindings;
  auto ensure_bindings = [&]() -> const SyncRegionBindings& {
    if (!bindings) {
      bindings.emplace(owner, target);
      result.index_nodes = bindings->SourceNodes();
      result.parameter_nodes = bindings->ParameterNodes();
    }
    return *bindings;
  };
  std::function<void(const StmtPtr&)> discover = [&](const StmtPtr& statement) {
    if (!statement) return;
    ++result.discovery_nodes;
    if (const auto seq = As<SeqStmts>(statement)) {
      for (const auto& child : seq->stmts_) discover(child);
    } else if (const auto branch = As<IfStmt>(statement)) {
      discover(branch->then_body_);
      if (branch->else_body_) discover(*branch->else_body_);
    } else if (const auto loop = As<ForStmt>(statement)) {
      if (loop->kind_ == ForKind::Sequential) {
        const auto sit = loop_summaries.find(loop.get());
        const ScopeSummary sub = sit != loop_summaries.end() ? sit->second : ScopeSummary{};
        if (!sub.has_disqualifying_control && sub.has_payload) {
          if (sub.has_ddr_write && !sub.has_unqualified_call && !disjoint_gm) {
            EventfulSyncChildAttempt failed;
            failed.scope = loop;
            failed.stage = EventfulSyncChildStage::Effects;
            failed.reason = "Closed child external effects are not uniformly read-only";
            result.attempts.push_back(std::move(failed));
          } else {
            auto attempt = TryPrepareEventfulSyncChild(owner, loop, target, ensure_bindings(), parameters, blocks);
            const bool promoted = static_cast<bool>(attempt.child);
            result.attempts.push_back(std::move(attempt));
            if (promoted) return;
          }
        }
      }
      discover(loop->body_);
    } else if (const auto loop = As<WhileStmt>(statement)) {
      discover(loop->body_);
    }
  };
  discover(owner->body_);
  return result;
}

std::map<std::string, size_t> EventfulSyncChild::Describe() const {
  return {{"shared_index_source_nodes", context_.index_work->source_nodes},
          {"query_scope_nodes", context_.query_scope_nodes},
          {"binding_definitions", context_.binding_definitions},
          {"analyzed_occurrences", mathematics_.Diagnostics().analyzed_occurrences},
          {"retained_cells", mathematics_.Diagnostics().cells},
          {"retained_regions", mathematics_.State().segments.size()},
          {"unavailable_storage_domains", mathematics_.State().input.unavailable_storage_spaces.size()},
          {"projected_external_roots", mathematics_.State().input.projected_readonly_external_roots.size()}};
}

void EventfulSyncChild::ValidateFor(const FunctionPtr& owner, const std::string& target) const {
  CHECK(owner == Owner() && target == target_)
      << "Eventful child belongs to a different original owner or native context";
  // The original owner is immutable. Identity and target bind the retained
  // original scope/ancestor proof without rescanning the entire owner per use.
}

namespace {
bool IsProvablyNonNegativeExpr(const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& ssa_defs,
                               std::unordered_set<const Var*>& visiting) {
  if (!expr) return false;
  if (const auto c = As<ConstInt>(expr)) return c->value_ >= 0;
  if (const auto v = As<Var>(expr)) {
    const auto it = ssa_defs.find(v.get());
    if (it == ssa_defs.end() || !visiting.insert(v.get()).second) return false;
    const bool ok = IsProvablyNonNegativeExpr(it->second, ssa_defs, visiting);
    visiting.erase(v.get());
    return ok;
  }
  if (const auto call = As<Call>(expr)) {
    return IsOp(call, "buffer.get_block_idx") || IsOp(call, "buffer.get_block_num") ||
           IsOp(call, "buffer.get_subblock_idx");
  }
  if (const auto add = As<Add>(expr)) {
    return IsProvablyNonNegativeExpr(add->left_, ssa_defs, visiting) &&
           IsProvablyNonNegativeExpr(add->right_, ssa_defs, visiting);
  }
  if (const auto mul = As<Mul>(expr)) {
    return IsProvablyNonNegativeExpr(mul->left_, ssa_defs, visiting) &&
           IsProvablyNonNegativeExpr(mul->right_, ssa_defs, visiting);
  }
  if (const auto fdiv = As<FloorDiv>(expr)) {
    const auto rhs = As<ConstInt>(fdiv->right_);
    return rhs && rhs->value_ > 0 && IsProvablyNonNegativeExpr(fdiv->left_, ssa_defs, visiting);
  }
  if (const auto fmod = As<FloorMod>(expr)) {
    const auto rhs = As<ConstInt>(fmod->right_);
    return rhs && rhs->value_ > 0 && IsProvablyNonNegativeExpr(fmod->left_, ssa_defs, visiting);
  }
  if (const auto mn = As<Min>(expr)) {
    return IsProvablyNonNegativeExpr(mn->left_, ssa_defs, visiting) &&
           IsProvablyNonNegativeExpr(mn->right_, ssa_defs, visiting);
  }
  if (const auto mx = As<Max>(expr)) {
    return IsProvablyNonNegativeExpr(mx->left_, ssa_defs, visiting) ||
           IsProvablyNonNegativeExpr(mx->right_, ssa_defs, visiting);
  }
  return false;
}
}  // namespace

bool IsSpmdAtMostOneTripFor(const ForStmtPtr& loop,
                             const std::unordered_map<const Var*, ExprPtr>& ssa_defs) {
  if (!loop || loop->kind_ != ForKind::Sequential || As<ConstInt>(loop->start_)) return false;
  const auto stop = As<ConstInt>(loop->stop_);
  const auto step = As<ConstInt>(loop->step_);
  if (!stop || !step || step->value_ <= 1 || stop->value_ <= 0 || stop->value_ > step->value_) {
    return false;
  }
  std::unordered_set<const Var*> visiting;
  return IsProvablyNonNegativeExpr(loop->start_, ssa_defs, visiting);
}
}  // namespace pypto::ir::sync
