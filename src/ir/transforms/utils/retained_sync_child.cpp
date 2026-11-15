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
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/visitor.h"
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
        const bool projected = mathematics_.State().input.unavailable_storage_spaces.count(MemorySpace::DDR);
        const bool external = As<TensorType>(type) || (buffer && buffer->memory_space_ == MemorySpace::DDR);
        if (projected && external && operation.ArgumentEffects()[index].data == BufferAccess::Read) continue;
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
        if (mathematics_.State().input.unavailable_storage_spaces.count(MemorySpace::DDR) &&
            effect.space == MemorySpace::DDR && effect.data == BufferAccess::Read)
          continue;
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
  result.child = std::shared_ptr<const NativeSyncChild>(
      new NativeSyncChild(mathematics, scope, target, std::move(operations.payloads)));
  result.status = NativeSyncChildStatus::Ready;
  result.reason = "Retained internal order is native-only, with no notification resources";
  return result;
}
bool HasReadonlyExternalSyncEffects(const ForStmtPtr& scope, FunctionType core, const std::string& target) {
  class Effects : public IRVisitor {
   public:
    Effects(FunctionType core, const std::string& target) : core_(core), target_(target) {}
    bool external = false;
    void VisitExpr_(const CallPtr& call) override {
      if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None ||
          backend::QualifyBufferSyncNonPayload(call, target_))
        return;
      const auto operation = backend::QualifyBufferSyncOperation(call, core_, target_);
      for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
        const auto effect = operation.ArgumentEffects()[index].data;
        if (effect == BufferAccess::None) continue;
        const auto type = call->args_[index]->GetType();
        const auto buffer = As<BufferType>(type);
        if (As<TensorType>(type) || (buffer && buffer->memory_space_ == MemorySpace::DDR)) {
          CHECK(effect == BufferAccess::Read) << "Closed child external effects are not uniformly read-only";
          external = true;
        }
      }
      for (const auto& effect : operation.Implementation().ImplicitEffects()) {
        if (effect.space != MemorySpace::DDR) continue;
        CHECK(effect.data == BufferAccess::Read) << "Closed child implicit external effect is not read-only";
        external = true;
      }
    }

   private:
    FunctionType core_;
    const std::string& target_;
  } effects(core, target);
  effects.VisitStmt(scope);
  return effects.external;
}

class SyncRegionBindings::State : public ParentStmtAnalysis {
 public:
  FunctionPtr owner;
  std::string target;
  size_t clock = 0;
  std::unordered_map<const Stmt*, size_t> occurrences, begin, end;
  std::unordered_map<const Var*, std::vector<AssignStmtPtr>> definitions;
  // Ephemeral request-local qualification cache. Published mathematics and
  // capabilities do not depend on later mutations of this index.
  std::map<std::pair<const AssignStmt*, SyncExternalPolicy>, bool> qualified;
  std::shared_ptr<const SyncRegionIndexWork> work;
  std::vector<VarPtr> ready_shape_symbols;

  State(FunctionPtr function, std::string native) : owner(std::move(function)), target(std::move(native)) {
    CHECK(owner) << "Regional synchronization requires an original owner";
    BuildMap(owner);
    // The existing signature service identifies physical tensor extents supplied
    // by the invocation ABI. This is readiness, never a numerical bound or a
    // valid-shape-only/tensor.read visibility premise.
    const auto physical_symbols = CollectParamDimSymbols(owner->params_);
    for (const auto& symbol : ClassifySignatureSymbols(owner).bound)
      if (physical_symbols.count(symbol.get())) ready_shape_symbols.push_back(symbol);
    work = std::make_shared<const SyncRegionIndexWork>(SyncRegionIndexWork{clock});
  }
  void VisitStmt(const StmtPtr& statement) override {
    ++occurrences[statement.get()];
    begin.emplace(statement.get(), clock++);
    if (const auto definition = As<AssignStmt>(statement))
      definitions[definition->var_.get()].push_back(definition);
    ParentStmtAnalysis::VisitStmt(statement);
    end[statement.get()] = clock;
  }
  void VisitExpr(const ExprPtr&) override {}
  bool Dominates(const AssignStmtPtr& definition, const StmtPtr& use) const {
    const auto parent = GetParent(definition);
    // A binding in a sequence dominates precisely the later sibling subtrees
    // in that sequence. Its interval ends at the original branch/loop scope;
    // a definition from another arm or a prior iteration cannot leak out.
    return parent && As<SeqStmts>(parent) && occurrences.at(definition.get()) == 1 &&
           occurrences.at(parent.get()) == 1 && begin.at(definition.get()) < begin.at(use.get()) &&
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
              As<ConstFloat>(assignment->value_);
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

SyncRegionContext SyncRegionBindings::Bind(const ForStmtPtr& scope, SyncExternalPolicy policy) const {
  CHECK(scope && state_->occurrences.count(scope.get()) && state_->occurrences.at(scope.get()) == 1)
      << "Regional synchronization scope must occur exactly once in its original owner";
  std::vector<StmtPtr> ancestors;
  for (auto parent = state_->GetParent(scope); parent; parent = state_->GetParent(parent))
    ancestors.push_back(parent);
  std::reverse(ancestors.begin(), ancestors.end());
  class Uses : public var_collectors::VarDefUseCollector {
   public:
    size_t scope_nodes = 0;
    void VisitStmt(const StmtPtr& statement) override {
      ++scope_nodes;
      IRVisitor::VisitStmt(statement);
    }
    void VisitExpr(const ExprPtr& expression) override {
      if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
    }

   private:
    std::unordered_set<const Expr*> seen_;
  } uses;
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
    for (size_t index = 0; index < state_->owner->params_.size(); ++index)
      available.emplace(
          state_->owner->params_[index].get(),
          std::make_pair(state_->owner->params_[index], state_->owner->param_directions_[index]));
    for (const auto& symbol : state_->ready_shape_symbols)
      available.emplace(symbol.get(), std::make_pair(symbol, ParamDirection::In));
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
    Uses dependencies;
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

std::shared_ptr<const EventfulSyncChild> PrepareEventfulSyncChild(const FunctionPtr& owner,
                                                                  const ForStmtPtr& scope,
                                                                  const std::string& target,
                                                                  const SyncRegionBindings& bindings) {
  const auto policy = HasReadonlyExternalSyncEffects(scope, owner->func_type_, target)
                          ? SyncExternalPolicy::ReadOnlyProjection
                          : SyncExternalPolicy::Geometry;
  auto context = bindings.Bind(scope, policy);
  CHECK(context.original_owner == owner)
      << "Regional synchronization index belongs to another original owner";
  auto mathematics = AnalyzeCompactSync(policy == SyncExternalPolicy::ReadOnlyProjection
                                            ? ImportReadonlyExternalSync(context.binding_owner, target)
                                            : ImportPlacedSync(context.binding_owner, true, target));
  NativeSyncChildExport qualification;
  Operations operations(mathematics, target, qualification);
  operations.VisitStmt(scope);
  CHECK(operations.available) << "Eventful child native/effect qualification unavailable: "
                              << qualification.reason;
  std::set<PipeType> participating;
  for (const auto& operation : operations.payloads) participating.insert(operation.Implementation().Pipe());
  auto logical = RealizeCompactSync(mathematics).function;
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
  const auto logical_context = SyncRegionBindings(logical, target).Bind(selected.found, policy);
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
  recipe.push_back(marked);
  auto logical_recipe = std::make_shared<SeqStmts>(recipe, scope->span_);
  return std::shared_ptr<const EventfulSyncChild>(
      new EventfulSyncChild(std::move(context), std::move(mathematics), logical_recipe, target,
                            std::vector<PipeType>(participating.begin(), participating.end())));
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
}  // namespace pypto::ir::sync
