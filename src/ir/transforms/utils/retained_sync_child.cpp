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
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
#include "pypto/ir/transforms/utils/signature_symbols.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"
#include "src/ir/transforms/utils/sync_transform_common.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
namespace {
class Operations : public IRVisitor {
 public:
  Operations(const CompactSyncResult& mathematics, const std::string& target, NativeSyncChildExport& result,
             bool allow_disjoint_gm = false)
      : mathematics_(mathematics), target_(target), result_(result), allow_disjoint_gm_(allow_disjoint_gm) {
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
            (allow_disjoint_gm_ ||
             (mathematics_.State().input.unavailable_storage_spaces.count(MemorySpace::DDR) &&
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
namespace {
bool CheckReadonlyExternalEffects(const StmtPtr* stmts, size_t count, FunctionType core,
                                  const std::string& target, SyncChildEffectWork* work,
                                  bool allow_disjoint_gm_writes) {
  class Effects : public IRVisitor {
   public:
    Effects(FunctionType core, const std::string& target, SyncChildEffectWork* work,
            bool allow_disjoint_gm_writes)
        : core_(core), target_(target), work_(work), allow_disjoint_gm_writes_(allow_disjoint_gm_writes) {}
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
  for (size_t i = 0; i < count; ++i) effects.VisitStmt(stmts[i]);
  return effects.external && !effects.has_write;
}
}  // namespace

bool HasReadonlyExternalSyncEffects(const ForStmtPtr& scope, FunctionType core, const std::string& target,
                                    SyncChildEffectWork* work, bool allow_disjoint_gm_writes) {
  StmtPtr stmt = scope;
  return CheckReadonlyExternalEffects(&stmt, 1, core, target, work, allow_disjoint_gm_writes);
}

bool HasReadonlyExternalSyncEffectsInSlice(const SeqStmtsPtr& seq, size_t begin, size_t end,
                                           FunctionType core, const std::string& target,
                                           SyncChildEffectWork* work, bool allow_disjoint_gm_writes) {
  CHECK(seq && begin <= end && end <= seq->stmts_.size()) << "Invalid statement slice for effect check";
  return CheckReadonlyExternalEffects(seq->stmts_.data() + begin, end - begin, core, target, work,
                                      allow_disjoint_gm_writes);
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
  SyncControlRangeProver prover;
  size_t clock = 0;
  std::unordered_map<const Stmt*, size_t> occurrences, begin, end;
  std::unordered_map<const Var*, std::vector<AssignStmtPtr>> definitions;
  std::unordered_map<const Var*, std::vector<std::pair<StmtPtr, VarPtr>>> return_var_definitions;
  // Ephemeral request-local qualification cache. Published mathematics and
  // capabilities do not depend on later mutations of this index.
  std::map<std::pair<const AssignStmt*, SyncExternalPolicy>, bool> qualified;
  std::shared_ptr<const SyncRegionIndexWork> work;
  std::map<const Var*, std::pair<VarPtr, ParamDirection>> ready_parameters;

  State(FunctionPtr function, std::string native, SyncParameterBounds parameters,
        std::optional<std::pair<int64_t, int64_t>> blocks)
      : owner(std::move(function)), target(std::move(native)), prover(owner, parameters, blocks) {
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
      if (!ready && As<ScalarType>(assignment->var_->GetType())) {
        CheckSyncScalarReadiness(assignment->value_, [](const VarPtr&) { return true; });
        ready = true;  // Every operand is separately bound at this original cut below.
      }
    }
    qualified.emplace(key, ready);
    return ready;
  }

  static bool IsFiniteRange(const std::optional<SymbolicIntRange>& range) {
    return range && range->lo > arith::ConstIntBound::kNegInf &&
           range->hi < arith::ConstIntBound::kPosInf - 1;
  }

  template <typename UsesT>
  SyncRegionContext BindWithUses(std::vector<StmtPtr> ancestors, const ForStmtPtr& target_scope,
                                 const StmtPtr& use_cut, const UsesT& uses, SyncExternalPolicy policy,
                                 SyncRegionBindWork* work) const {
    std::vector<std::pair<const Var*, StmtPtr>> pending;
    for (const auto* variable : uses.var_uses)
      if (!uses.var_defs.count(variable)) pending.emplace_back(variable, use_cut);
    for (const auto* variable : uses.GetAllVarRefs())
      for (const auto* metadata : var_collectors::CollectTypeVars(variable->GetType()))
        if (!uses.var_defs.count(metadata)) pending.emplace_back(metadata, use_cut);
    std::set<const Var*> required;
    std::vector<AssignStmtPtr> ordered;
    std::map<const Var*, std::pair<VarPtr, ParamDirection>> captures;
    std::map<const Var*, std::pair<VarPtr, ParamDirection>> available;
    std::map<const Var*, ForStmtPtr> invocation_frames;
    for (const auto& ancestor : ancestors) {
      if (const auto loop = As<ForStmt>(ancestor)) {
        const bool admit = policy == SyncExternalPolicy::ReadOnlyProjection ||
                           IsFiniteRange(prover.LookupAtCut(use_cut.get(), loop->loop_var_));
        if (admit) {
          available.emplace(loop->loop_var_.get(), std::make_pair(loop->loop_var_, ParamDirection::In));
          invocation_frames.emplace(loop->loop_var_.get(), loop);
        }
      }
    }
    while (!pending.empty()) {
      const auto [variable, use] = pending.back();
      pending.pop_back();
      const auto parameter = ready_parameters.find(variable);
      if (parameter != ready_parameters.end()) {
        captures.emplace(variable, parameter->second);
        continue;
      }
      if (const auto captured = available.find(variable); captured != available.end()) {
        if (const auto frame = invocation_frames.find(variable); frame != invocation_frames.end()) {
          const auto& loop = frame->second;
          CHECK(occurrences.at(loop.get()) == 1 && begin.at(loop.get()) < begin.at(use.get()) &&
                begin.at(use.get()) < end.at(loop.get()))
              << "Opaque enclosing coordinate is unavailable at its original definition/use cut";
        }
        captures.emplace(variable, captured->second);
        continue;
      }
      const auto found = definitions.find(variable);
      if (found == definitions.end()) {
        const auto rv_found = return_var_definitions.find(variable);
        if (rv_found != return_var_definitions.end() && rv_found->second.size() == 1) {
          const auto& [def_stmt, rv_var] = rv_found->second.front();
          if (As<ScalarType>(rv_var->GetType())) {
            CHECK(Dominates(def_stmt, use))
                << "Regional synchronization descriptor binding does not dominate its original use";
            captures.emplace(variable, std::make_pair(rv_var, ParamDirection::In));
            continue;
          }
        }
      }
      CHECK(found != definitions.end() && found->second.size() == 1)
          << "Regional synchronization child has an unavailable enclosing coordinate, caller value or "
             "descriptor binding: "
          << variable->name_hint_;
      const auto& definition = found->second.front();
      CHECK(Dominates(definition, use))
          << "Regional synchronization descriptor binding does not dominate its original use";
      CHECK(const_cast<State*>(this)->Qualify(definition, policy))
          << "Regional synchronization requires a signal-free original binding";
      if (!required.insert(variable).second) continue;
      ordered.push_back(definition);
      if (work) ++work->binding_definitions;
      UsesT dependencies(work);
      dependencies.VisitExpr(definition->value_);
      for (const auto* dependency : dependencies.var_uses) pending.emplace_back(dependency, definition);
      for (const auto* metadata : var_collectors::CollectTypeVars(variable->GetType()))
        pending.emplace_back(metadata, definition);
    }
    std::sort(ordered.begin(), ordered.end(),
              [&](const auto& a, const auto& b) { return begin.at(a.get()) < begin.at(b.get()); });
    std::vector<StmtPtr> body(ordered.begin(), ordered.end());
    body.push_back(target_scope);
    body.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, target_scope->span_));
    auto binding = std::make_shared<Function>(*owner);
    binding->params_.clear();
    binding->param_directions_.clear();
    for (const auto& [identity, capture] : captures) {
      binding->params_.push_back(capture.first);
      binding->param_directions_.push_back(capture.second);
    }
    binding->return_types_.clear();
    binding->body_ = std::make_shared<SeqStmts>(body, target_scope->span_);
    return {owner,      target_scope,     std::move(ancestors), std::move(binding),
            work_ptr(), uses.scope_nodes, ordered.size()};
  }
  const std::shared_ptr<const SyncRegionIndexWork>& work_ptr() const { return work; }
};

SyncRegionBindings::SyncRegionBindings(FunctionPtr owner, std::string target, SyncParameterBounds parameters,
                                       std::optional<std::pair<int64_t, int64_t>> blocks)
    : state_(std::make_shared<State>(std::move(owner), std::move(target), std::move(parameters),
                                     std::move(blocks))) {}

const SyncControlRangeProver& SyncRegionBindings::Prover() const { return state_->prover; }
size_t SyncRegionBindings::SourceNodes() const { return state_->work->source_nodes; }
size_t SyncRegionBindings::ParameterNodes() const { return state_->work->parameter_nodes; }

namespace {
class SyncUsesCollector : public var_collectors::VarDefUseCollector {
 public:
  explicit SyncUsesCollector(SyncRegionBindWork* work) : work_(work) {}
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
};
}  // namespace

SyncRegionContext SyncRegionBindings::Bind(const ForStmtPtr& scope, SyncExternalPolicy policy,
                                           SyncRegionBindWork* work) const {
  CHECK(scope && state_->occurrences.count(scope.get()) && state_->occurrences.at(scope.get()) == 1)
      << "Regional synchronization scope must occur exactly once in its original owner";
  std::vector<StmtPtr> ancestors;
  for (auto parent = state_->GetParent(scope); parent; parent = state_->GetParent(parent))
    ancestors.push_back(parent);
  std::reverse(ancestors.begin(), ancestors.end());
  if (work) work->ancestor_links += ancestors.size();
  SyncUsesCollector uses(work);
  uses.VisitStmt(scope);
  return state_->BindWithUses(std::move(ancestors), scope, scope, uses, policy, work);
}

SyncRegionContext SyncRegionBindings::BindSlice(const SeqStmtsPtr& seq, size_t begin, size_t end,
                                                const ForStmtPtr& wrapped_scope, SyncExternalPolicy policy,
                                                SyncRegionBindWork* work) const {
  CHECK(seq && begin < end && end <= seq->stmts_.size() && wrapped_scope)
      << "Regional synchronization slice must be non-empty";
  for (size_t i = begin; i < end; ++i) {
    CHECK(state_->occurrences.count(seq->stmts_[i].get()) &&
          state_->occurrences.at(seq->stmts_[i].get()) == 1)
        << "Regional synchronization slice statement must occur exactly once in its original owner";
  }
  std::vector<StmtPtr> ancestors;
  for (auto parent = state_->GetParent(seq); parent; parent = state_->GetParent(parent))
    ancestors.push_back(parent);
  std::reverse(ancestors.begin(), ancestors.end());
  if (work) work->ancestor_links += ancestors.size();
  SyncUsesCollector uses(work);
  for (size_t i = begin; i < end; ++i) uses.VisitStmt(seq->stmts_[i]);
  return state_->BindWithUses(std::move(ancestors), wrapped_scope, seq->stmts_[begin], uses, policy, work);
}

SyncParameterBounds ProjectEnclosingLoopBounds(const SyncRegionContext& context,
                                               const SyncControlRangeProver& prover, const StmtPtr& cut,
                                               const SyncParameterBounds& child_bounds) {
  SyncParameterBounds projected = child_bounds;
  if (!context.binding_owner || !cut) return projected;
  std::unordered_set<const Var*> enclosing_loop_vars;
  for (const auto& ancestor : context.ancestors) {
    if (const auto loop = As<ForStmt>(ancestor)) {
      enclosing_loop_vars.insert(loop->loop_var_.get());
    }
  }
  if (enclosing_loop_vars.empty()) return projected;
  for (size_t c_idx = 0; c_idx < context.binding_owner->params_.size(); ++c_idx) {
    if (projected.count(c_idx)) continue;
    const auto& param = context.binding_owner->params_[c_idx];
    if (!enclosing_loop_vars.count(param.get())) continue;
    const auto range = prover.LookupAtCut(cut.get(), param);
    if (range && range->lo > arith::ConstIntBound::kNegInf && range->hi < arith::ConstIntBound::kPosInf - 1) {
      projected.emplace(c_idx, std::make_pair(range->lo, range->hi));
    }
  }
  return projected;
}

struct RetainedSyncChildDetail {
  static std::shared_ptr<const EventfulSyncChild> Make(SyncRegionContext context,
                                                       CompactSyncResult mathematics, StmtPtr logical_recipe,
                                                       std::string target,
                                                       std::vector<SyncDirection> resource_dirs,
                                                       SeqStmtsPtr window_seq, size_t window_begin,
                                                       size_t window_end) {
    return std::shared_ptr<const EventfulSyncChild>(new EventfulSyncChild(
        std::move(context), std::move(mathematics), std::move(logical_recipe), std::move(target),
        std::move(resource_dirs), std::move(window_seq), window_begin, window_end));
  }
};
namespace {
FunctionPtr StripSubblockAttrForImport(const FunctionPtr& binding_owner) {
  const auto seq = As<SeqStmts>(binding_owner ? binding_owner->body_ : nullptr);
  if (!seq || seq->stmts_.size() < 2) return binding_owner;
  const size_t scope_idx = seq->stmts_.size() - 2;
  const auto loop = As<ForStmt>(seq->stmts_[scope_idx]);
  if (!loop || !loop->HasAttr(kClosedSyncChildSubblockAttr)) return binding_owner;
  auto clean_loop = std::make_shared<ForStmt>(*loop);
  clean_loop->attrs_.erase(
      std::remove_if(clean_loop->attrs_.begin(), clean_loop->attrs_.end(),
                     [](const auto& attr) { return attr.first == kClosedSyncChildSubblockAttr; }),
      clean_loop->attrs_.end());
  auto next_stmts = seq->stmts_;
  next_stmts[scope_idx] = std::move(clean_loop);
  auto clean_fn = std::make_shared<Function>(*binding_owner);
  clean_fn->body_ = std::make_shared<SeqStmts>(std::move(next_stmts), seq->span_);
  return clean_fn;
}

void FinalizeEventfulSyncChild(const FunctionPtr& owner, const ForStmtPtr& scope, const StmtPtr& bound_cut,
                               const std::string& target, SyncRegionContext context,
                               SyncExternalPolicy policy, const SyncRegionBindings& bindings,
                               const SyncParameterBounds& parameters,
                               const std::optional<std::pair<int64_t, int64_t>>& blocks, bool disjoint_gm,
                               SeqStmtsPtr window_seq, size_t window_begin, size_t window_end,
                               EventfulSyncChildAttempt& attempt,
                               const std::function<void(EventfulSyncChildStage)>& enter_phase,
                               const std::function<void()>& finish_phase) {
  try {
    enter_phase(EventfulSyncChildStage::Analysis);
    auto child_bounds = ProjectSyncParameterBounds(owner, context.binding_owner, parameters);
    child_bounds = ProjectEnclosingLoopBounds(context, bindings.Prover(), bound_cut, child_bounds);
    const auto import_owner = StripSubblockAttrForImport(context.binding_owner);
    attempt.mathematics = std::make_shared<const CompactSyncResult>(
        AnalyzeCompactSync(policy == SyncExternalPolicy::ReadOnlyProjection
                               ? ImportReadonlyExternalSync(import_owner, target, child_bounds, blocks)
                               : ImportPlacedSync(import_owner, true, target, child_bounds, blocks)));
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
    if (window_seq) {
      marked->attrs_.emplace_back(kClosedSyncChildSubblockAttr, 1);
    }
    recipe.push_back(marked);
    auto logical_recipe = std::make_shared<SeqStmts>(recipe, scope->span_);
    // Parent insertion always uses the conservative-prefix policy. Plan the
    // standalone, unmarked recipe after the same token optimization that runs
    // before physical lowering, since removing a relay can change loop closure.
    const auto optimized = OptimizeLogicalSync(logical_context.binding_owner);
    const auto logical_bounds = ProjectSyncParameterBounds(context.binding_owner, optimized, child_bounds);
    auto prefix_dirs = PlanClosedPrefixEventDirections(optimized, logical_bounds, blocks);
    // Structured lowering assigns flags to the token waits' source and
    // destination pipes. Its lane allocator only changes IDs within those pools.
    auto resource_dirs = prefix_dirs ? std::move(*prefix_dirs) : CollectSyncWaitDirections(optimized->body_);
    for (const auto& [dir, ids] : SyncNativeClaims(mathematics.State().input, target)) {
      (void)ids;
      if (dir.first != dir.second) {
        resource_dirs.insert(dir);
      }
    }
    attempt.child =
        RetainedSyncChildDetail::Make(std::move(context), mathematics, logical_recipe, target,
                                      std::vector<SyncDirection>(resource_dirs.begin(), resource_dirs.end()),
                                      std::move(window_seq), window_begin, window_end);
    enter_phase(EventfulSyncChildStage::Ready);
  } catch (const pypto::ValueError& error) {
    finish_phase();
    attempt.reason = error.what();
  }
}
}  // namespace

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
    const auto policy =
        HasReadonlyExternalSyncEffects(scope, owner->func_type_, target, &attempt.effect_work, disjoint_gm)
            ? SyncExternalPolicy::ReadOnlyProjection
            : SyncExternalPolicy::Geometry;
    enter_phase(EventfulSyncChildStage::Binding);
    auto context = bindings.Bind(scope, policy, &attempt.binding_work);
    CHECK(context.original_owner == owner)
        << "Regional synchronization index belongs to another original owner";
    FinalizeEventfulSyncChild(owner, scope, scope, target, std::move(context), policy, bindings, parameters,
                              blocks, disjoint_gm, nullptr, 0, 0, attempt, enter_phase, finish_phase);
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

namespace {
struct SlicePrescreenSummary {
  bool has_payload = false;
  bool has_disqualifying_control = false;
  bool has_ddr_write = false;
  bool has_unqualified_call = false;
  bool is_alloc = false;
  bool is_pure_def = false;
  bool is_payload_eval = false;
  bool is_exact_for = false;
  void MergeFrom(const SlicePrescreenSummary& other) {
    has_payload |= other.has_payload;
    has_disqualifying_control |= other.has_disqualifying_control;
    has_ddr_write |= other.has_ddr_write;
    has_unqualified_call |= other.has_unqualified_call;
  }
};

std::optional<InlineExactSubBlock> TryPrepareInlineExactSubBlock(
    const FunctionPtr& owner, const SeqStmtsPtr& seq, size_t win_a, size_t win_b, const std::string& target,
    const SyncRegionBindings& bindings, const SyncParameterBounds& parameters,
    const std::optional<std::pair<int64_t, int64_t>>& blocks, bool disjoint_gm,
    const std::function<const SyncFootprintResolver&()>& ensure_resolver,
    std::map<SyncDirection, size_t>* scope_dir_peak) {
  try {
    const auto policy = HasReadonlyExternalSyncEffectsInSlice(seq, win_a, win_b, owner->func_type_, target,
                                                              nullptr, disjoint_gm)
                            ? SyncExternalPolicy::ReadOnlyProjection
                            : SyncExternalPolicy::Geometry;
    const auto span = seq->span_;
    auto win_iv = std::make_shared<Var>("_sync_win_iv", std::make_shared<ScalarType>(DataType::INDEX), span);
    std::vector<StmtPtr> win_stmts(seq->stmts_.begin() + win_a, seq->stmts_.begin() + win_b);
    win_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, span));
    auto win_body = std::make_shared<SeqStmts>(std::move(win_stmts), span);
    auto unit_loop = std::make_shared<ForStmt>(win_iv, std::make_shared<ConstInt>(0, DataType::INDEX, span),
                                               std::make_shared<ConstInt>(1, DataType::INDEX, span),
                                               std::make_shared<ConstInt>(1, DataType::INDEX, span),
                                               std::vector<IterArgPtr>{}, win_body, std::vector<VarPtr>{},
                                               span, ForKind::Sequential);

    auto context = bindings.BindSlice(seq, win_a, win_b, unit_loop, policy);
    auto child_bounds = ProjectSyncParameterBounds(owner, context.binding_owner, parameters);
    child_bounds = ProjectEnclosingLoopBounds(context, bindings.Prover(), seq->stmts_[win_a], child_bounds);
    auto mathematics = std::make_shared<const CompactSyncResult>(AnalyzeCompactSync(
        policy == SyncExternalPolicy::ReadOnlyProjection
            ? ImportReadonlyExternalSync(context.binding_owner, target, child_bounds, blocks)
            : ImportPlacedSync(context.binding_owner, true, target, child_bounds, blocks)));
    const auto& input = mathematics->State().input;
    const auto& result_segments = mathematics->State().segments;
    if (input.segments.size() != 1 || result_segments.size() != 1) return std::nullopt;
    const auto& seg = input.segments.front();
    const auto& cseg = result_segments.front();
    if (seg.sites.size() < 2) return std::nullopt;

    // Map every segment site original Call* back to its statement index in [win_a, win_b).
    std::unordered_map<const Call*, size_t> call_to_stmt;
    for (size_t k = win_a; k < win_b; ++k) {
      if (const auto eval = As<EvalStmt>(seq->stmts_[k])) {
        if (const auto call = As<Call>(eval->expr_)) {
          call_to_stmt.emplace(call.get(), k - win_a);
        }
      }
    }
    std::vector<size_t> site_to_stmt(seg.sites.size());
    for (size_t s = 0; s < seg.sites.size(); ++s) {
      if (seg.sites[s].continuation) return std::nullopt;
      const auto it = call_to_stmt.find(seg.sites[s].original.get());
      if (it == call_to_stmt.end()) return std::nullopt;
      site_to_stmt[s] = it->second;
    }

    // Keep intra-body (distance == 0) demands; unit-trip segment (`count == 1`) skips `distance >= 1`.
    // Compute Exact's latest producer statement `u_exact_max` per `(v_stmt, pipe_u)`.
    struct ExactFrontier {
      size_t u_stmt = 0;
      PipeType pipe_v = PipeType::ALL;
    };
    std::map<std::pair<size_t, PipeType>, ExactFrontier> exact_max;
    for (const auto& demand : cseg.demands) {
      if (demand.distance != 0) continue;
      if (demand.source >= demand.target || demand.target >= seg.sites.size()) return std::nullopt;
      const size_t u_stmt = site_to_stmt[demand.source];
      const size_t v_stmt = site_to_stmt[demand.target];
      if (u_stmt >= v_stmt) return std::nullopt;
      const PipeType pipe_u = seg.sites[demand.source].pipe;
      const PipeType pipe_v = seg.sites[demand.target].pipe;
      auto [it, ins] = exact_max.emplace(std::make_pair(v_stmt, pipe_u), ExactFrontier{u_stmt, pipe_v});
      if (!ins && u_stmt > it->second.u_stmt) {
        it->second.u_stmt = u_stmt;
        it->second.pipe_v = pipe_v;
      }
    }

    // Sharpening profitability gate [R3]: compare Exact against parent `SyncFootprintResolver`.
    // Accept iff there is at least one `(v_stmt, pipe_u)` where Conservative has an un-killed hazard
    // at a strictly later `u_stmt` than `exact_max(v_stmt, pipe_u)` (or where `exact_max` has none).
    const SyncFootprintResolver& resolver = ensure_resolver();
    struct SiteAccess {
      ResolvedOperandFootprint fp;
      bool implicit = false;
    };
    std::vector<PipeType> site_pipes(seg.sites.size());
    std::vector<std::vector<SiteAccess>> site_accesses(seg.sites.size());
    for (size_t s = 0; s < seg.sites.size(); ++s) {
      const CallPtr& call = seg.sites[s].original;
      const Stmt* cut = seq->stmts_[win_a + site_to_stmt[s]].get();
      const auto op = backend::QualifyBufferSyncOperation(call, owner->func_type_, target);
      site_pipes[s] = op.Implementation().Pipe();
      for (size_t arg_idx = 0; arg_idx < op.ArgumentEffects().size(); ++arg_idx) {
        const auto mode = op.ArgumentEffects()[arg_idx].data;
        if (mode == BufferAccess::None) continue;
        const auto arg_type = call->args_[arg_idx]->GetType();
        MemorySpace space = MemorySpace::DDR;
        if (const auto buf = As<BufferType>(arg_type)) space = buf->memory_space_;
        site_accesses[s].push_back({resolver.ResolveOperand(cut, call, arg_idx, mode, space,
                                                            /*enable_refresh_kill=*/true, disjoint_gm),
                                    false});
      }
      for (const auto& eff : op.Implementation().ImplicitEffects()) {
        ResolvedOperandFootprint fp;
        fp.space = eff.space;
        fp.read = (eff.data == BufferAccess::Read || eff.data == BufferAccess::ReadWrite);
        fp.write = (eff.data == BufferAccess::Write || eff.data == BufferAccess::ReadWrite);
        site_accesses[s].push_back({std::move(fp), true});
      }
    }

    auto access_conflicts = [&](const SiteAccess& sa, const SiteAccess& sb) -> bool {
      const auto& a = sa.fp;
      const auto& b = sb.fp;
      if (!a.write && !b.write) return false;
      if (a.space != b.space) return false;
      if (a.space == MemorySpace::DDR) {
        if (disjoint_gm && a.gm_root && b.gm_root) return a.gm_root.get() == b.gm_root.get();
        return true;
      }
      if (sa.implicit || sb.implicit || !a.byte_hull || !b.byte_hull) return true;
      if (__int128(a.byte_hull->begin) >= __int128(b.byte_hull->end) ||
          __int128(b.byte_hull->begin) >= __int128(a.byte_hull->end)) {
        return false;
      }
      if (a.space == MemorySpace::Vec && a.vec_lattice && b.vec_lattice &&
          !StridedByteFootprint::MayOverlap(*a.vec_lattice, *b.vec_lattice)) {
        return false;
      }
      if (a.space != MemorySpace::Vec && a.matrix_box && b.matrix_box &&
          !MatrixBoxFootprint::MayOverlap(*a.matrix_box, *b.matrix_box, resolver.Roots())) {
        return false;
      }
      if (a.fiber.kind == IterationFiber::Kind::ModularSlot &&
          b.fiber.kind == IterationFiber::Kind::ModularSlot && a.fiber.frame == b.fiber.frame &&
          a.fiber.modulus > 0 && a.fiber.modulus == b.fiber.modulus && a.fiber.coeff == b.fiber.coeff &&
          a.fiber.offset != b.fiber.offset) {
        return false;
      }
      if (a.fiber.kind == IterationFiber::Kind::AffineSlice &&
          b.fiber.kind == IterationFiber::Kind::AffineSlice && a.fiber.frame == b.fiber.frame &&
          a.fiber.slice_axis == b.fiber.slice_axis && a.fiber.coeff == b.fiber.coeff) {
        bool same_sym = (!a.fiber.symbolic_base && !b.fiber.symbolic_base);
        if (!same_sym && a.fiber.symbolic_base && b.fiber.symbolic_base) {
          if (a.fiber.symbolic_base.get() == b.fiber.symbolic_base.get()) {
            same_sym = true;
          } else {
            try {
              same_sym = structural_equal(a.fiber.symbolic_base, b.fiber.symbolic_base, false);
            } catch (const StructuralComparisonUnsupported&) {
              same_sym = false;
            }
          }
        }
        if (same_sym) {
          const __int128 a_end = __int128(a.fiber.offset) + a.fiber.extent;
          const __int128 b_end = __int128(b.fiber.offset) + b.fiber.extent;
          if (a_end <= b.fiber.offset || b_end <= a.fiber.offset) return false;
        }
      }
      return true;
    };

    auto killed_between = [&](size_t i, size_t j, const SiteAccess& sa) -> bool {
      if (sa.implicit || sa.fp.space == MemorySpace::DDR || sa.fp.root_id == 0 || !sa.fp.byte_hull) {
        return false;
      }
      for (size_t k = i + 1; k < j; ++k) {
        for (const auto& sk : site_accesses[k]) {
          if (sk.implicit || !sk.fp.candidate_full_overwrite || !sk.fp.write || sk.fp.read) continue;
          if (sk.fp.space != sa.fp.space || sk.fp.root_id != sa.fp.root_id || !sk.fp.byte_hull) continue;
          if (sk.fp.byte_hull->begin <= sa.fp.byte_hull->begin &&
              sk.fp.byte_hull->end >= sa.fp.byte_hull->end && access_conflicts(sa, sk)) {
            return true;
          }
        }
      }
      return false;
    };

    bool sharpens = false;
    for (size_t j = 1; j < seg.sites.size() && !sharpens; ++j) {
      const size_t vj = site_to_stmt[j];
      for (size_t i = 0; i < j && !sharpens; ++i) {
        const size_t ui = site_to_stmt[i];
        if (ui >= vj) continue;
        const auto ex_it = exact_max.find({vj, site_pipes[i]});
        if (ex_it != exact_max.end() && ui <= ex_it->second.u_stmt) continue;
        for (const auto& sa : site_accesses[i]) {
          bool pair_conflicts = false;
          for (const auto& sb : site_accesses[j]) {
            if (access_conflicts(sa, sb)) {
              pair_conflicts = true;
              break;
            }
          }
          if (pair_conflicts && !killed_between(i, j, sa)) {
            sharpens = true;
            break;
          }
        }
      }
    }
    if (!sharpens) return std::nullopt;

    // Coalesce per `(v_stmt, pipe_u)` to `u_max`, then drop transitively dominated `(u, v)`
    // along the same `(pipe_u, pipe_v)` lane pair using hardware FIFO ordering on both pipes.
    std::map<SyncDirection, std::vector<std::pair<size_t, size_t>>> by_dir;
    for (const auto& [vp, front] : exact_max) {
      by_dir[{vp.second, front.pipe_v}].emplace_back(vp.first, front.u_stmt);
    }
    struct CoalescedEdge {
      PipeType pipe_u;
      size_t u_stmt;
      PipeType pipe_v;
      size_t v_stmt;
    };
    std::vector<CoalescedEdge> kept_edges;
    for (auto& [dir, edges] : by_dir) {
      std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second > b.second;
      });
      ssize_t max_u_waited = -1;
      for (const auto& [v_stmt, u_stmt] : edges) {
        if (max_u_waited >= 0 && static_cast<size_t>(max_u_waited) >= u_stmt) continue;
        kept_edges.push_back({dir.first, u_stmt, dir.second, v_stmt});
        if (dir.first == dir.second) {
          max_u_waited = std::max<ssize_t>(max_u_waited, static_cast<ssize_t>(v_stmt) - 1);
        } else {
          max_u_waited = std::max<ssize_t>(max_u_waited, static_cast<ssize_t>(u_stmt));
        }
      }
    }
    std::sort(kept_edges.begin(), kept_edges.end(), [](const CoalescedEdge& a, const CoalescedEdge& b) {
      return std::tie(a.v_stmt, a.u_stmt, a.pipe_u, a.pipe_v) <
             std::tie(b.v_stmt, b.u_stmt, b.pipe_u, b.pipe_v);
    });

    // Splice reduced 1-to-1 zero-fanout logical tokens and check peak cross-pipe budget (<= 4 / dir).
    const size_t len = win_b - win_a;
    std::vector<std::vector<StmtPtr>> pre_stmts(len), post_stmts(len);
    std::vector<std::vector<SyncDirection>> pre_cross_dirs(len), post_cross_dirs(len);
    size_t tok_idx = 0;
    for (const auto& edge : kept_edges) {
      auto rec_call = OpRegistry::GetInstance().Create("sync.record", {},
                                                       {{"pipe", static_cast<int>(edge.pipe_u)}}, span);
      auto tok =
          std::make_shared<Var>("_sync_tok_sub_" + std::to_string(tok_idx++), rec_call->GetType(), span);
      auto wait_call = OpRegistry::GetInstance().Create("sync.wait", {tok},
                                                        {{"dst_pipe", static_cast<int>(edge.pipe_v)}}, span);
      auto rec_stmt = std::make_shared<AssignStmt>(tok, rec_call, span);
      auto wait_stmt = std::make_shared<EvalStmt>(wait_call, span);
      if (edge.pipe_u == edge.pipe_v) {
        pre_stmts[edge.v_stmt].push_back(std::move(rec_stmt));
        pre_stmts[edge.v_stmt].push_back(std::move(wait_stmt));
      } else {
        post_stmts[edge.u_stmt].push_back(std::move(rec_stmt));
        pre_stmts[edge.v_stmt].push_back(std::move(wait_stmt));
        post_cross_dirs[edge.u_stmt].emplace_back(edge.pipe_u, edge.pipe_v);
        pre_cross_dirs[edge.v_stmt].emplace_back(edge.pipe_u, edge.pipe_v);
      }
    }

    constexpr size_t kMaxInlineCrossPipeTokensPerDir = 4;
    std::map<SyncDirection, size_t> open_dir, subblock_peak;
    for (size_t i = 0; i < len; ++i) {
      for (const auto& dir : pre_cross_dirs[i]) --open_dir[dir];
      for (const auto& dir : post_cross_dirs[i]) {
        const size_t cur = ++open_dir[dir];
        subblock_peak[dir] = std::max(subblock_peak[dir], cur);
      }
    }
    for (const auto& [dir, pk] : subblock_peak) {
      const size_t used = (scope_dir_peak && scope_dir_peak->count(dir)) ? scope_dir_peak->at(dir) : 0;
      if (used + pk > kMaxInlineCrossPipeTokensPerDir) return std::nullopt;
    }

    std::vector<StmtPtr> spliced;
    spliced.reserve(len + 2 * tok_idx);
    for (size_t i = 0; i < len; ++i) {
      for (auto& st : pre_stmts[i]) spliced.push_back(std::move(st));
      spliced.push_back(seq->stmts_[win_a + i]);
      for (auto& st : post_stmts[i]) spliced.push_back(std::move(st));
    }
    auto spliced_seq = std::make_shared<SeqStmts>(std::move(spliced), span);
    const CompletionSummary completion =
        SummarizeStraightLineLogicalCompletion(spliced_seq, owner->func_type_, target);
    if (scope_dir_peak) {
      for (const auto& [dir, pk] : subblock_peak) (*scope_dir_peak)[dir] += pk;
    }
    return InlineExactSubBlock{seq, win_a, win_b, mathematics, spliced_seq, tok_idx, tok_idx, completion};
  } catch (const pypto::ValueError&) {
    return std::nullopt;
  }
}

EventfulSyncChildAttempt TryPrepareSubblockWindow(const FunctionPtr& owner, const SeqStmtsPtr& seq,
                                                  size_t win_a, size_t win_b, const std::string& target,
                                                  const SyncRegionBindings& bindings,
                                                  const SyncParameterBounds& parameters,
                                                  const std::optional<std::pair<int64_t, int64_t>>& blocks,
                                                  bool disjoint_gm) {
  EventfulSyncChildAttempt attempt;
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
    const auto span = seq->span_;
    auto win_iv = std::make_shared<Var>("_sync_win_iv", std::make_shared<ScalarType>(DataType::INDEX), span);
    std::vector<StmtPtr> win_stmts(seq->stmts_.begin() + win_a, seq->stmts_.begin() + win_b);
    win_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, span));
    auto win_body = std::make_shared<SeqStmts>(std::move(win_stmts), span);
    auto win_loop = std::make_shared<ForStmt>(win_iv, std::make_shared<ConstInt>(0, DataType::INDEX, span),
                                              std::make_shared<ConstInt>(1, DataType::INDEX, span),
                                              std::make_shared<ConstInt>(1, DataType::INDEX, span),
                                              std::vector<IterArgPtr>{}, win_body, std::vector<VarPtr>{},
                                              span, ForKind::Sequential,
                                              decltype(ForStmt::attrs_){{kClosedSyncChildSubblockAttr, 1}});
    attempt.scope = win_loop;
    const auto policy = HasReadonlyExternalSyncEffectsInSlice(seq, win_a, win_b, owner->func_type_, target,
                                                              &attempt.effect_work, disjoint_gm)
                            ? SyncExternalPolicy::ReadOnlyProjection
                            : SyncExternalPolicy::Geometry;
    enter_phase(EventfulSyncChildStage::Binding);
    auto context = bindings.BindSlice(seq, win_a, win_b, win_loop, policy, &attempt.binding_work);
    FinalizeEventfulSyncChild(owner, win_loop, seq->stmts_[win_a], target, std::move(context), policy,
                              bindings, parameters, blocks, disjoint_gm, seq, win_a, win_b, attempt,
                              enter_phase, finish_phase);
    return attempt;
  } catch (const pypto::ValueError& error) {
    finish_phase();
    attempt.reason = error.what();
  }
  return attempt;
}
}  // namespace

AutomaticSyncChildren PrepareAutomaticSyncChildren(const FunctionPtr& owner, const std::string& target,
                                                   const SyncParameterBounds& parameters,
                                                   const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  const auto* ctx = PassContext::Current();
  const bool disjoint_gm = ctx && ctx->GetAutoSyncDisjointGM();

  // O(N) bottom-up syntactic pre-screen filters loops/statements containing
  // While/Parallel/Return/Submit/allocs or unqualified calls without building Bind/Analysis.
  std::unordered_map<const Stmt*, SlicePrescreenSummary> stmt_summaries;
  std::function<SlicePrescreenSummary(const StmtPtr&)> prescreen =
      [&](const StmtPtr& stmt) -> SlicePrescreenSummary {
    SlicePrescreenSummary sum;
    if (!stmt) return sum;
    if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) sum.MergeFrom(prescreen(child));
      return sum;
    }
    if (const auto branch = As<IfStmt>(stmt)) {
      sum.MergeFrom(prescreen(branch->then_body_));
      if (branch->else_body_) sum.MergeFrom(prescreen(*branch->else_body_));
      stmt_summaries.emplace(stmt.get(), sum);
      return sum;
    }
    if (const auto loop = As<ForStmt>(stmt)) {
      const auto body_sum = prescreen(loop->body_);
      sum = body_sum;
      if (loop->kind_ != ForKind::Sequential) sum.has_disqualifying_control = true;
      sum.is_exact_for = loop->kind_ == ForKind::Sequential && !sum.has_disqualifying_control &&
                         sum.has_payload && (!sum.has_ddr_write || sum.has_unqualified_call || disjoint_gm);
      stmt_summaries.emplace(stmt.get(), sum);
      return sum;
    }
    if (const auto loop = As<WhileStmt>(stmt)) {
      prescreen(loop->body_);
      sum.has_disqualifying_control = true;
      stmt_summaries.emplace(stmt.get(), sum);
      return sum;
    }
    if (As<ReturnStmt>(stmt)) {
      sum.has_disqualifying_control = true;
      stmt_summaries.emplace(stmt.get(), sum);
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
      if (const auto call = As<Call>(eval->expr_)) {
        if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None ||
            backend::QualifyBufferSyncNonPayload(call, target)) {
          sum.has_disqualifying_control = true;
        } else {
          scan_call(call);
          if (sum.has_payload && !sum.has_unqualified_call) sum.is_payload_eval = true;
        }
      } else if (As<Submit>(eval->expr_)) {
        sum.has_disqualifying_control = true;
      }
    } else if (const auto assign = As<AssignStmt>(stmt)) {
      if (const auto call = As<Call>(assign->value_)) {
        if (IsOp(call, "buffer.alloc") || IsOp(call, "buffer.alloc_multi")) {
          sum.is_alloc = true;
        } else if (backend::QualifyBufferSyncNonPayload(call, target)) {
          sum.is_pure_def = true;
        } else {
          sum.has_disqualifying_control = true;
        }
      } else if (As<Submit>(assign->value_)) {
        sum.has_disqualifying_control = true;
      } else {
        sum.is_pure_def = true;
      }
    }
    stmt_summaries.emplace(stmt.get(), sum);
    return sum;
  };
  prescreen(owner->body_);

  AutomaticSyncChildren result;
  std::optional<SyncRegionBindings> bindings;
  auto ensure_bindings = [&]() -> const SyncRegionBindings& {
    if (!bindings) {
      bindings.emplace(owner, target, parameters, blocks);
      result.index_nodes = bindings->SourceNodes();
      result.parameter_nodes = bindings->ParameterNodes();
    }
    return *bindings;
  };

  std::optional<SyncFootprintResolver> resolver;
  auto ensure_resolver = [&]() -> const SyncFootprintResolver& {
    if (!resolver) {
      resolver.emplace(owner, &ensure_bindings().Prover(), /*run_scalar_layout_check=*/false, target);
    }
    return *resolver;
  };
  std::unordered_map<const Stmt*, std::map<SyncDirection, size_t>> scope_dir_peaks;

  auto has_var_use_after = [](const SeqStmtsPtr& seq, size_t win_b, const VarPtr& var) {
    var_collectors::VarDefUseCollector collector;
    for (size_t i = win_b; i < seq->stmts_.size(); ++i) collector.VisitStmt(seq->stmts_[i]);
    return collector.var_uses.count(var.get()) != 0;
  };

  std::function<void(const StmtPtr&, bool, const Stmt*)> discover = [&](const StmtPtr& statement,
                                                                        bool inside_unpromoted,
                                                                        const Stmt* scope_key) {
    if (!statement) return;
    ++result.discovery_nodes;
    if (const auto seq = As<SeqStmts>(statement)) {
      auto is_win_eligible = [&](size_t idx) -> bool {
        const auto& s = seq->stmts_[idx];
        const auto it = stmt_summaries.find(s.get());
        if (it == stmt_summaries.end()) return false;
        const auto& sum = it->second;
        if (sum.is_pure_def) return true;
        if (sum.is_payload_eval) return !sum.has_ddr_write || disjoint_gm;
        if (const auto loop = As<ForStmt>(s)) {
          return loop->kind_ == ForKind::Sequential && !sum.has_disqualifying_control && sum.has_payload;
        }
        return false;
      };

      auto try_tier1_for = [&](const ForStmtPtr& loop) {
        const auto it = stmt_summaries.find(loop.get());
        const SlicePrescreenSummary sub = it != stmt_summaries.end() ? it->second : SlicePrescreenSummary{};
        if (!sub.has_disqualifying_control && sub.has_payload) {
          if (sub.has_ddr_write && !sub.has_unqualified_call && !disjoint_gm) {
            EventfulSyncChildAttempt failed;
            failed.scope = loop;
            failed.stage = EventfulSyncChildStage::Effects;
            failed.reason = "Closed child external effects are not uniformly read-only";
            result.attempts.push_back(std::move(failed));
          } else {
            auto attempt =
                TryPrepareEventfulSyncChild(owner, loop, target, ensure_bindings(), parameters, blocks);
            const bool promoted = static_cast<bool>(attempt.child);
            result.attempts.push_back(std::move(attempt));
            if (promoted) return;
          }
        }
        discover(loop->body_, true, loop.get());
      };

      auto process_clean_slice = [&](auto& self, size_t a, size_t b) -> void {
        while (a < b) {
          const auto a_sum = stmt_summaries.find(seq->stmts_[a].get());
          if (a_sum != stmt_summaries.end() && a_sum->second.is_pure_def) {
            ++a;
            continue;
          }
          break;
        }
        while (b > a) {
          const auto b_sum = stmt_summaries.find(seq->stmts_[b - 1].get());
          if (b_sum != stmt_summaries.end() && b_sum->second.is_pure_def) {
            --b;
            continue;
          }
          break;
        }
        if (a >= b) return;

        // Split on any interior pure def whose SSA var is used after `b`.
        for (size_t k = a; k < b; ++k) {
          if (const auto assign = As<AssignStmt>(seq->stmts_[k])) {
            if (has_var_use_after(seq, b, assign->var_)) {
              self(self, a, k);
              self(self, k + 1, b);
              return;
            }
          }
        }

        size_t num_fors = 0, num_payloads = 0;
        for (size_t k = a; k < b; ++k) {
          if (As<ForStmt>(seq->stmts_[k]))
            ++num_fors;
          else if (As<EvalStmt>(seq->stmts_[k]))
            ++num_payloads;
        }

        if (num_fors == 1 && b - a == 1) {
          try_tier1_for(As<ForStmt>(seq->stmts_[a]));
          return;
        }

        if (num_fors >= 1) {
          if (inside_unpromoted && b - a >= 2) {
            // Check whether every ForStmt in [a, b) lacks DDR writes unless disjoint_gm.
            bool all_ok = true;
            for (size_t k = a; k < b; ++k) {
              if (const auto loop = As<ForStmt>(seq->stmts_[k])) {
                const auto& lsum = stmt_summaries.at(loop.get());
                if (lsum.has_ddr_write && !lsum.has_unqualified_call && !disjoint_gm) {
                  all_ok = false;
                  break;
                }
              }
            }
            if (all_ok) {
              auto attempt = TryPrepareSubblockWindow(owner, seq, a, b, target, ensure_bindings(), parameters,
                                                      blocks, disjoint_gm);
              if (attempt.child) {
                result.attempts.push_back(std::move(attempt));
                return;
              }
            }
          }
          // Split on inner ForStmt boundaries: promote each ForStmt via Tier 1 and try
          // straight-line gaps between them via Class B.
          size_t seg_start = a;
          for (size_t k = a; k < b; ++k) {
            if (const auto loop = As<ForStmt>(seq->stmts_[k])) {
              if (seg_start < k) self(self, seg_start, k);
              try_tier1_for(loop);
              seg_start = k + 1;
            }
          }
          if (seg_start < b) self(self, seg_start, b);
          return;
        }

        // Pure straight-line slice: Class B (`InlineExactSubBlock`).
        if (num_payloads >= 2) {
          if (auto inline_block = TryPrepareInlineExactSubBlock(
                  owner, seq, a, b, target, ensure_bindings(), parameters, blocks, disjoint_gm,
                  ensure_resolver, &scope_dir_peaks[scope_key])) {
            result.inline_subblocks.push_back(std::move(*inline_block));
          }
        }
      };

      const size_t n = seq->stmts_.size();
      size_t idx = 0;
      while (idx < n) {
        if (is_win_eligible(idx)) {
          size_t end = idx + 1;
          while (end < n && is_win_eligible(end)) ++end;
          process_clean_slice(process_clean_slice, idx, end);
          idx = end;
        } else {
          discover(seq->stmts_[idx], inside_unpromoted, scope_key);
          ++idx;
        }
      }
    } else if (const auto branch = As<IfStmt>(statement)) {
      discover(branch->then_body_, true, scope_key);
      if (branch->else_body_) discover(*branch->else_body_, true, scope_key);
    } else if (const auto loop = As<ForStmt>(statement)) {
      if (loop->kind_ == ForKind::Sequential) {
        const auto it = stmt_summaries.find(loop.get());
        const SlicePrescreenSummary sub = it != stmt_summaries.end() ? it->second : SlicePrescreenSummary{};
        if (!sub.has_disqualifying_control && sub.has_payload) {
          if (sub.has_ddr_write && !sub.has_unqualified_call && !disjoint_gm) {
            EventfulSyncChildAttempt failed;
            failed.scope = loop;
            failed.stage = EventfulSyncChildStage::Effects;
            failed.reason = "Closed child external effects are not uniformly read-only";
            result.attempts.push_back(std::move(failed));
          } else {
            auto attempt =
                TryPrepareEventfulSyncChild(owner, loop, target, ensure_bindings(), parameters, blocks);
            const bool promoted = static_cast<bool>(attempt.child);
            result.attempts.push_back(std::move(attempt));
            if (promoted) return;
          }
        }
      }
      discover(loop->body_, true, loop.get());
    } else if (const auto loop = As<WhileStmt>(statement)) {
      discover(loop->body_, true, loop.get());
    }
  };
  discover(owner->body_, false, owner->body_.get());
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
    return IsPureBlockQueryCall(call);
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

bool IsSpmdAtMostOneTripFor(const ForStmtPtr& loop, const std::unordered_map<const Var*, ExprPtr>& ssa_defs) {
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
