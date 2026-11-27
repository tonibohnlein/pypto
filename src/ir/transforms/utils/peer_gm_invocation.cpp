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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/gm_pipe_layout_evidence.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "src/ir/transforms/utils/peer_local_effects.h"

namespace pypto::ir::sync {
namespace {

std::map<size_t, ExprPtr> MakeGmDomains(const FunctionPtr& group, const std::set<size_t>& ordinary,
                                        const std::set<size_t>& isolated) {
  std::map<size_t, ExprPtr> domains;
  if (!isolated.empty()) {
    ExprPtr other;
    for (const auto formal : ordinary) {
      const auto& parameter = group->params_.at(formal);
      if (!isolated.count(formal) && !other) other = parameter;
      domains.emplace(formal, isolated.count(formal) ? parameter : other);
    }
  }
  return domains;
}

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

enum class OriginKind { Pending, Formal, Fresh, Unknown };
struct Origin {
  const Var* root = nullptr;
  OriginKind kind = OriginKind::Pending;

  bool Known() const { return kind == OriginKind::Formal || kind == OriginKind::Fresh; }
};
struct Site {
  struct Actual {
    const Var* value;
    AssignStmtPtr managed_create;
  };

  Span span;
  bool qualified;
  StmtPtr statement;
  std::vector<Actual> arguments;
  bool runtime_cluster_count;
  const Var* cluster_count_binding;
  ExprPtr effective_count;
};

// A formal carries the dispatch width only if all incoming sites agree.
void JoinCoordinates(std::optional<SyncDispatchCoordinates>& coordinates, int64_t minimum, int64_t maximum,
                     std::set<size_t> formals) {
  if (!coordinates) {
    coordinates = SyncDispatchCoordinates{{minimum, maximum}, std::move(formals)};
    return;
  }
  coordinates->block_count.first = std::min(coordinates->block_count.first, minimum);
  coordinates->block_count.second = std::max(coordinates->block_count.second, maximum);
  auto& common = coordinates->width_parameters;
  for (auto it = common.begin(); it != common.end();) {
    if (!formals.count(*it)) {
      it = common.erase(it);
    } else {
      ++it;
    }
  }
}

// An identical physical base can expose an already-known external alias.
// Unequal bases do not imply separation of externally supplied tensors.
const Expr* KnownAliasIdentity(const Origin& origin) {
  if (origin.kind != OriginKind::Formal) return origin.root;
  const auto tensor = As<TensorType>(origin.root->GetType());
  const auto memref = tensor->memref_.value_or(nullptr);
  return memref && memref->base_ ? memref->base_.get() : origin.root;
}

// A root inventory records exact storage-origin equations at lexical cuts.
// Tensor creates name fresh allocations. Formal names and memrefs cannot prove
// separation of externally supplied buffers. Each equation propagates at
// most two transitions (pending -> known -> unknown), including loop cycles.
class RootInventory final : public IRVisitor {
 public:
  RootInventory(const FunctionPtr& root, const std::set<std::string>& launches, const ProgramPtr& program,
                bool allow_slice_origins = false)
      : launches_(launches), program_(program), allow_slice_origins_(allow_slice_origins) {
    for (const auto& formal : root->params_) {
      formal_names_.insert(formal->name_hint_);
      live_.insert(formal.get());
      if (IsPlainTensor(formal)) {
        flows_.try_emplace(formal.get());
        Update(formal.get(), {formal.get(), OriginKind::Formal});
      }
    }
    VisitStmt(root->body_);
    Propagate(0);
    const size_t settled = pending_.size();
    for (const auto& [value, flow] : flows_) {
      if (flow.origin.kind == OriginKind::Pending) Update(value, kUnknown);
    }
    Propagate(settled);
  }

  void Check(const FunctionPtr& launch, const std::set<size_t>& ordinary_formals, size_t workspace_formal,
             int64_t elements, const SyncInvocationEvidence& invocation, bool sharded) const {
    const auto found = calls_.find(launch->name_);
    CHECK(found != calls_.end()) << "Peer Group invocation lacks a visible root call";
    for (const auto& site : found->second) {
      CHECK_SPAN(site.qualified, site.span)
          << "Peer Group GM proof requires a direct root invocation with explicit tensor actuals";
      CheckArguments(site, ordinary_formals, workspace_formal);
      CheckSite(launch, site, workspace_formal, elements, invocation, sharded);
    }
  }

  std::optional<SyncDispatchCoordinates> LaunchCoordinates(const FunctionPtr& launch,
                                                           const SyncInvocationEvidence& invocation) const {
    const auto found = calls_.find(launch->name_);
    if (found == calls_.end() || launch->func_type_ != FunctionType::Spmd) return std::nullopt;
    std::optional<SyncDispatchCoordinates> coordinates;
    for (const auto& site : found->second) {
      const auto& count = site.effective_count;
      if (!site.qualified || !count ||
          (site.cluster_count_binding && definitions_.at(site.cluster_count_binding) != 1)) {
        return std::nullopt;
      }
      const auto maximum = MaxLaunchCount(launch, count, site, invocation);
      if (maximum <= 0 || maximum > std::numeric_limits<int16_t>::max()) return std::nullopt;
      // The dispatch accepts a nonempty positive width. Only a representable
      // width can also denote the original scalar after LaunchSpec conversion.
      const int64_t minimum = transform_utils::EvalConstInt(count).value_or(1);
      std::set<size_t> formals;
      if (const auto variable = AsVarLike(count)) {
        for (size_t i = 0; i < site.arguments.size(); ++i) {
          if (site.arguments[i].value == variable.get() && As<ScalarType>(launch->params_[i]->GetType()) &&
              GetScalarDtype(launch->params_[i]) == GetScalarDtype(variable)) {
            formals.insert(i);
          }
        }
      }
      JoinCoordinates(coordinates, minimum, maximum, std::move(formals));
    }
    return coordinates;
  }

  void CheckHostRankOneExtent(const FunctionPtr& launch, size_t formal, const TensorType& expected) const {
    const auto matches = [&](const Var* value) {
      const auto type = value ? As<TensorType>(value->GetType()) : nullptr;
      return type && !type->tensor_view_ && type->dtype_ == expected.dtype_ &&
             tile_view_semantics::ShapeExprListsEquivalent(type->shape_, expected.shape_);
    };
    CHECK(matches(launch->params_.at(formal).get()))
        << "Peer scalar GM address requires its original packed rank-one launch shape";
    for (const auto& site : calls_.at(launch->name_)) {
      const auto* argument = site.arguments.at(formal).value;
      const auto origin = StorageOf(argument);
      // The actual root formal is described by the runtime Tensor ABI,
      // whose shapes[0] is uint32_t. Do not use this evidence for constructed
      // views or for a value that merely shares the original backing storage.
      CHECK_SPAN(origin.kind == OriginKind::Formal && origin.root == argument && matches(argument), site.span)
          << "Peer scalar GM address requires a direct bounded host Tensor extent";
    }
  }

  void CheckScalarReads(const FunctionPtr& launch, const PeerGmEffects& effects, bool disjoint_gm) const {
    for (const auto& site : calls_.at(launch->name_)) {
      std::unordered_set<const Expr*> writes;
      bool writes_formal = false;
      for (const auto write : effects.writes) {
        const auto origin = StorageOf(site.arguments.at(write).value);
        writes.insert(KnownAliasIdentity(origin));
        writes_formal |= origin.kind == OriginKind::Formal;
      }
      for (const auto read : effects.scalar_reads) {
        const auto read_origin = StorageOf(site.arguments.at(read).value);
        CHECK_SPAN(
            read_origin.Known() && !writes.count(nullptr) && !writes.count(KnownAliasIdentity(read_origin)),
            site.span)
            << "Peer scalar GM read aliases a GM write in the mixed dispatch";
        CHECK_SPAN(disjoint_gm || read_origin.kind == OriginKind::Fresh || !writes_formal, site.span)
            << "Peer scalar GM read may alias a GM write without a disjoint GM contract";
      }
    }
  }

  void CheckAuthoredGM(const FunctionPtr& launch, size_t workspace_formal,
                       const std::vector<PeerGmEffects>& phases, const std::set<size_t>& ordinary,
                       bool disjoint_gm) const {
    const auto found = calls_.find(launch->name_);
    CHECK(found != calls_.end() && !found->second.empty())
        << "Authored GM synchronization requires an original root launch";
    for (const auto& site : found->second) {
      CHECK_SPAN(site.qualified, site.span) << "Authored GM synchronization requires a direct root launch";
      const auto workspace = StorageOf(site.arguments.at(workspace_formal).value);
      CHECK_SPAN(workspace.kind == OriginKind::Fresh, site.span)
          << "Authored FFTS workspace requires a fresh root allocation";
      for (const auto formal : ordinary) {
        const auto actual = StorageOf(site.arguments.at(formal).value);
        CHECK_SPAN(actual.Known() && actual.root != workspace.root, site.span)
            << "Authored FFTS workspace lacks separation from an ordinary GM access";
      }
    }
    for (const auto& phase : phases) {
      CheckScalarReads(launch, phase, disjoint_gm);
    }
  }

  void RetainIsolatedFresh(const FunctionPtr& launch, const std::set<size_t>& ordinary,
                           std::set<size_t>& isolated) const {
    // Both invocation checkers have established known origins for every
    // ordinary actual. Intersect the fresh, singly bound roots at every call.
    for (const auto& site : calls_.at(launch->name_)) {
      std::unordered_map<const Var*, size_t> first_fresh;
      for (const auto formal : ordinary) {
        const auto actual = StorageOf(site.arguments.at(formal).value);
        if (actual.kind != OriginKind::Fresh) {
          isolated.erase(formal);
          continue;
        }
        const auto [prior, unique] = first_fresh.emplace(actual.root, formal);
        if (!unique) {
          isolated.erase(prior->second);
          isolated.erase(formal);
        }
      }
    }
  }

  void CheckArguments(const FunctionPtr& launch, const std::set<size_t>& ordinary_formals,
                      size_t workspace_formal) const {
    for (const auto& site : calls_.at(launch->name_)) {
      CheckArguments(site, ordinary_formals, workspace_formal);
    }
  }

  void VisitStmt(const StmtPtr& statement) override {
    const bool previous = qualified_;
    const auto previous_statement = statement_;
    const auto depth = live_order_.size();
    statement_ = statement;
    qualified_ &= As<SeqStmts>(statement) || As<RuntimeScopeStmt>(statement) || As<AssignStmt>(statement) ||
                  As<EvalStmt>(statement) || As<IfStmt>(statement) || As<ForStmt>(statement) ||
                  As<WhileStmt>(statement);
    IRVisitor::VisitStmt(statement);
    if (!qualified_) Restore(depth);
    statement_ = previous_statement;
    qualified_ = previous;
  }

  void VisitExpr(const ExprPtr& expression) override {
    if (const auto var = AsVarLike(expression)) {
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
    const auto value = assignment->value_;
    const auto* source = SourceOf(value);
    const auto call = transform_utils::AsCallOrSubmitView(value);
    Define(variable);
    assignments_.emplace(variable.get(), assignment);
    if (As<TensorType>(variable->GetType())) {
      if (IsOp(call, "tensor.create") && IsPlainTensor(variable)) {
        Update(variable.get(), {variable.get(), OriginKind::Fresh});
      } else {
        DependOn(variable, source);
      }
    } else if (As<TupleType>(variable->GetType())) {
      if (call || As<MakeTuple>(value)) {
        tuples_[variable.get()] = value;
      } else if (const auto tuple = tuples_.find(source); tuple != tuples_.end()) {
        tuples_[variable.get()] = tuple->second;
      }
    }
    if (qualified_ && sequence_ && !formal_names_.count(variable->name_hint_) &&
        IsManagedCreate(assignment)) {
      active_creates_.try_emplace(variable.get(), sequence_, assignment);
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
    const auto* parent = sequence_;
    sequence_ = nullptr;
    // Runtime scopes are transparent to SSA; sizing consumers still have to
    // share the exact statement list used by the allocation emitter.
    IRVisitor::VisitStmt_(scope);
    sequence_ = parent;
  }

  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitExpr(branch->condition_);
    const auto depth = live_order_.size();
    const auto walk_arm = [&](const StmtPtr& body) {
      if (body) VisitStmt(body);
      const auto yield = transform_utils::GetLastYieldStmt(body);
      for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
        DependOn(branch->return_vars_[i], YieldSource(yield, i));
      }
      Restore(depth);
    };
    walk_arm(branch->then_body_);
    walk_arm(branch->else_body_.value_or(nullptr));
    for (const auto& result : branch->return_vars_) Define(result);
  }

  void VisitStmt_(const ForStmtPtr& loop) override {
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    VisitLoop(loop->iter_args_, loop->body_, loop->return_vars_);
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    VisitExpr(loop->condition_);
    VisitLoop(loop->iter_args_, loop->body_, loop->return_vars_);
  }

  void VisitExpr_(const CallPtr& call) override {
    Record(call.get(), call);
    IRVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    Record(submit.get(), transform_utils::AsCallOrSubmitView(submit));
    IRVisitor::VisitExpr_(submit);
  }

 private:
  struct Flow {
    Origin origin;
    std::vector<const Var*> users;
  };

  const std::set<std::string>& launches_;
  ProgramPtr program_;
  bool allow_slice_origins_;
  bool qualified_ = true;
  const Expr* direct_ = nullptr;
  const SeqStmts* sequence_ = nullptr;
  StmtPtr statement_;
  std::set<std::string> formal_names_;
  std::unordered_set<const Expr*> visited_;
  std::unordered_set<const Var*> live_;
  std::vector<const Var*> live_order_, pending_;
  std::unordered_map<const Var*, Flow> flows_;
  std::unordered_map<const Var*, AssignStmtPtr> assignments_;
  std::unordered_map<const Var*, ExprPtr> tuples_;
  std::unordered_map<const Function*, std::vector<std::optional<size_t>>> returns_;
  std::unordered_map<const Var*, std::pair<const SeqStmts*, AssignStmtPtr>> active_creates_;
  std::unordered_map<const Var*, size_t> uses_, definitions_;
  std::map<std::string, std::vector<Site>> calls_;

  static constexpr Origin kUnknown{nullptr, OriginKind::Unknown};

  void Propagate(size_t from) {
    for (size_t i = from; i < pending_.size(); ++i) {
      const auto& flow = flows_.at(pending_[i]);
      const auto origin = flow.origin;
      for (const auto* user : flow.users) Update(user, origin);
    }
  }

  void Update(const Var* value, Origin incoming) {
    if (incoming.kind == OriginKind::Pending) return;
    auto& current = flows_.at(value).origin;
    if (current.kind == OriginKind::Unknown) return;
    if (current.kind != OriginKind::Pending) {
      if (current.root == incoming.root && current.kind == incoming.kind) return;
      incoming = kUnknown;
    }
    current = incoming;
    pending_.push_back(value);
  }

  void Define(const VarPtr& variable) {
    if (live_.insert(variable.get()).second) live_order_.push_back(variable.get());
    auto [entry, first] = definitions_.try_emplace(variable.get(), 1);
    if (!first) ++entry->second;
    if (!As<TensorType>(variable->GetType())) return;
    flows_.try_emplace(variable.get());
    if (!first || StorageOf(variable.get()).kind == OriginKind::Formal) Update(variable.get(), kUnknown);
  }

  void Restore(size_t depth) {
    while (live_order_.size() > depth) {
      live_.erase(live_order_.back());
      live_order_.pop_back();
    }
  }

  Origin StorageOf(const Var* value) const {
    const auto found = flows_.find(value);
    return found == flows_.end() ? kUnknown : found->second.origin;
  }

  const Var* LiveVar(const ExprPtr& expression) const {
    const auto value = AsVarLike(expression);
    return live_.count(value.get()) ? value.get() : nullptr;
  }

  void DependOn(const VarPtr& value, const Var* source) {
    if (!As<TensorType>(value->GetType())) return;
    flows_.try_emplace(value.get());
    if (source) {
      flows_[source].users.push_back(value.get());
    } else {
      Update(value.get(), kUnknown);
    }
  }

  const Var* YieldSource(const YieldStmtPtr& yield, size_t i) {
    return yield && i < yield->value_.size() ? SourceOf(yield->value_[i]) : nullptr;
  }

  void VisitLoop(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                 const std::vector<VarPtr>& results) {
    const auto depth = live_order_.size();
    for (const auto& arg : iter_args) {
      VisitExpr(arg->initValue_);
      DependOn(arg, SourceOf(arg->initValue_));
    }
    for (const auto& arg : iter_args) Define(arg);
    VisitStmt(body);
    const auto yield = transform_utils::GetLastYieldStmt(body);
    for (size_t i = 0; i < iter_args.size(); ++i) DependOn(iter_args[i], YieldSource(yield, i));
    Restore(depth);
    for (size_t i = 0; i < results.size(); ++i) {
      // The carry equation contains both the zero-trip initial value and all
      // loop backedges. Taking its origin is safe even when a loop never runs.
      DependOn(results[i], i < iter_args.size() ? iter_args[i].get() : nullptr);
      Define(results[i]);
    }
  }

  const Var* ReturnOrigin(const CallPtr& call, size_t index) {
    const auto global = call ? As<GlobalVar>(call->op_) : nullptr;
    const auto callee = global ? program_->GetFunction(global->name_) : nullptr;
    if (!callee || callee->func_type_ == FunctionType::Opaque) return nullptr;
    auto [entry, first] = returns_.try_emplace(callee.get());
    if (first) entry->second = return_lineage::ExplicitReturnedParamIndices(callee);
    if (index >= entry->second.size() || !entry->second[index]) return nullptr;
    return LiveVar(transform_utils::CallerSuppliedArg(call, callee, *entry->second[index]));
  }

  const Var* SourceOf(const ExprPtr& expression) {
    if (const auto call = transform_utils::AsCallOrSubmitView(expression)) {
      // The authored proof uses only storage identity from a host view,
      // never its projected box. Keep the existing managed-workspace proof's
      // stricter original-argument scope for the separate FIFO route.
      const bool authored_slice = allow_slice_origins_ && IsOp(call, "tensor.slice");
      if ((IsOp(call, "tensor.reshape") || IsOp(call, "tensor.assemble") || authored_slice) &&
          !call->args_.empty()) {
        return LiveVar(call->args_.front());
      }
      return ReturnOrigin(call, 0);
    }
    if (const auto project = As<TupleGetItemExpr>(expression)) {
      auto tuple = project->tuple_;
      if (const auto found = tuples_.find(LiveVar(tuple)); found != tuples_.end()) {
        tuple = found->second;
      }
      const auto index = static_cast<size_t>(project->index_);
      if (const auto values = As<MakeTuple>(tuple)) {
        return index < values->elements_.size() ? LiveVar(values->elements_[index]) : nullptr;
      }
      return ReturnOrigin(transform_utils::AsCallOrSubmitView(tuple), index);
    }
    return LiveVar(expression);
  }

  void VisitDirect(const ExprPtr& expression) {
    direct_ = expression.get();
    VisitExpr(expression);
    direct_ = nullptr;
  }

  void Record(const Expr* original, const CallPtr& call) {
    const auto global = As<GlobalVar>(call->op_);
    if (!global || !launches_.count(global->name_)) return;
    const auto callee = program_->GetFunction(global->name_);
    Site site{original->span_, qualified_ && original == direct_, statement_, {}, false, nullptr, nullptr};
    if (site.qualified) {
      for (size_t i = 0; i < callee->params_.size(); ++i) {
        const auto* value = LiveVar(transform_utils::CallerSuppliedArg(call, callee, i));
        AssignStmtPtr managed_create;
        // The injected allocator searches only later statements in this same
        // sequence. Retain the matching create for every operand independently.
        if (const auto create = active_creates_.find(value);
            create != active_creates_.end() && create->second.first == sequence_) {
          managed_create = create->second.second;
        }
        site.arguments.push_back({value, std::move(managed_create)});
      }
      auto count = alloc_batching::EffectiveCoreNum(call, callee);
      site.effective_count = count;
      if (auto variable = LiveVar(count)) {
        const auto definition = assignments_.find(variable);
        if (definition != assignments_.end()) {
          count = definition->second->value_;
          site.cluster_count_binding = variable;
        }
      }
      const auto query = As<Call>(count);
      site.runtime_cluster_count = IsOp(query, "system.available_cluster_count") && query->args_.empty();
    }
    calls_[global->name_].push_back(std::move(site));
  }

  int64_t MaxLaunchCount(const FunctionPtr& launch, const ExprPtr& count, const Site& site,
                         const SyncInvocationEvidence& invocation) const {
    if (const auto constant = transform_utils::EvalConstInt(count)) return *constant;
    const auto bound = invocation.block_counts.find(launch->name_);
    if (bound == invocation.block_counts.end() || bound->second.first <= 0) return 0;
    auto maximum = bound->second.second;
    if (site.runtime_cluster_count &&
        (!site.cluster_count_binding || definitions_.at(site.cluster_count_binding) == 1)) {
      // This proof is used only by the A2/A3 local plan. Its runtime provides
      // the scheduler's physical cluster count (PLATFORM_MAX_BLOCKDIM = 24).
      constexpr int64_t kMaxA2A3Clusters = 24;
      maximum = std::min(maximum, kMaxA2A3Clusters);
    }
    return maximum;
  }

  void CheckArguments(const Site& site, const std::set<size_t>& ordinary_formals,
                      size_t workspace_formal) const {
    for (const auto formal : ordinary_formals) {
      const auto actual = StorageOf(site.arguments[formal].value);
      CHECK_SPAN(actual.Known(), site.span)
          << "Peer Group GM tensor must forward an original root formal or fresh allocation";
      CHECK_SPAN(actual.root != site.arguments[workspace_formal].value, site.span)
          << "Peer Group managed workspace aliases an ordinary root tensor";
    }
  }

  void CheckSite(const FunctionPtr& launch, const Site& site, size_t workspace_formal, int64_t elements,
                 const SyncInvocationEvidence& invocation, bool sharded) const {
    const auto& workspace_arg = site.arguments[workspace_formal];
    const auto* workspace = workspace_arg.value;
    const auto uses = uses_.find(workspace);
    const auto definitions = definitions_.find(workspace);
    CHECK_SPAN(workspace_arg.managed_create && uses != uses_.end() && uses->second == 1 &&
                   definitions != definitions_.end() && definitions->second == 1,
               site.span)
        << "Peer Group managed workspace requires a unique injected root tensor.create";
    // Direct siblings with one workspace use have the same sizing consumer
    // when irrelevant siblings are omitted.
    const auto use = alloc_batching::ResolveGMPipeCreateUse({workspace_arg.managed_create, site.statement}, 0,
                                                            workspace_arg.managed_create->var_, program_, {});
    CHECK_SPAN(use && use->callee == launch, site.span)
        << "Peer Group is not the workspace allocation's concrete sizing consumer";
    const auto blocks = use->core_num ? MaxLaunchCount(launch, use->core_num, site, invocation) : 1;
    CHECK_SPAN(blocks > 0 && (blocks == 1 || sharded) && elements > 0 &&
                   elements <= std::numeric_limits<int32_t>::max() / blocks,
               site.span)
        << "Peer Group workspace launch has no bounded, nonoverlapping native partition";
  }
};

// This is the simple SPMD wrapper unwrapped by mixed-launch codegen. Every
// Group operand must keep its ABI position, in particular the managed pipe
// tensor. A conditional or another call could change the concrete consumer.
bool DirectForwarding(const FunctionPtr& wrapper, const FunctionPtr& group) {
  const auto body = transform_utils::FlattenToStmts(wrapper->body_);
  if (body.size() < 2 || !As<ReturnStmt>(body.back())) return false;
  ExprPtr expression;
  VarPtr result;
  if (const auto assignment = As<AssignStmt>(body.front())) {
    expression = assignment->value_;
    result = assignment->var_;
  } else if (const auto evaluation = As<EvalStmt>(body.front())) {
    expression = evaluation->expr_;
  }
  const auto call = As<Call>(expression);
  const auto callee = call ? As<GlobalVar>(call->op_) : nullptr;
  if (!callee || callee->name_ != group->name_ || call->HasAttr(kAttrCoreNum) ||
      call->HasAttr(kAttrPredicate) || call->HasAttr(kAttrSyncStart) ||
      wrapper->params_.size() != group->params_.size() || call->args_.size() != group->params_.size()) {
    return false;
  }
  // Outlined multi-result wrappers can retain pure output projections, even
  // when returning the existing output formals instead of these locals.
  const auto tuple = result ? As<TupleType>(result->GetType()) : nullptr;
  for (size_t i = 1; i + 1 < body.size(); ++i) {
    const auto assignment = As<AssignStmt>(body[i]);
    const auto item = assignment ? As<TupleGetItemExpr>(assignment->value_) : nullptr;
    if (!item || !tuple || item->tuple_ != result || item->index_ < 0 ||
        static_cast<size_t>(item->index_) >= tuple->types_.size()) {
      return false;
    }
  }
  for (size_t i = 0; i < call->args_.size(); ++i) {
    if (call->args_[i] != wrapper->params_[i]) return false;
    if (const auto scalar = As<ScalarType>(wrapper->params_[i]->GetType());
        scalar && (!As<ScalarType>(group->params_[i]->GetType()) ||
                   scalar->dtype_ != GetScalarDtype(group->params_[i]))) {
      return false;
    }
  }
  return true;
}

bool HasNativeSharding(const FunctionPtr& group, const ProgramPtr& program) {
  class BlockUse final : public IRVisitor {
   public:
    bool found = false;

   protected:
    void VisitExpr_(const CallPtr& call) override {
      // Match _SPMD_BLOCK_OPS in the mixed kernel wrapper generator.
      found |= IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num") ||
               IsOp(call, "buffer.get_block_idx") || IsOp(call, "buffer.get_block_num");
      if (!found) IRVisitor::VisitExpr_(call);
    }
  };
  for (const auto& child : CollectInnerCalls(group, program)) {
    BlockUse visitor;
    visitor.VisitStmt(child.inner_callee->body_);
    if (visitor.found) return true;
  }
  return false;
}

}  // namespace

struct PeerInvocationGmEvidence::State {
  ProgramPtr program;
  const SyncInvocationEvidence& invocation;
  codegen::gm_pipe::LayoutEvidence layout;
  std::set<std::string> launches;
  std::map<std::string, std::set<std::string>> callers;
  std::map<std::string, std::unique_ptr<RootInventory>> roots;
  std::map<std::string, std::unique_ptr<RootInventory>> authored_roots;
  std::map<std::string, std::vector<std::pair<RootInventory*, FunctionPtr>>> qualified;
  std::map<std::string, bool> sharding;
  std::map<std::string, std::optional<SyncDispatchCoordinates>> coordinates;

  State(const ProgramPtr& owner, const SyncInvocationEvidence& proof)
      : program(owner), invocation(proof), layout(owner) {
    for (const auto& [global, function] : program->functions_) {
      if (function->func_type_ == FunctionType::Group || function->func_type_ == FunctionType::Spmd) {
        launches.insert(global->name_);
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
  if (const auto checked = state_->qualified.find(group->name_); checked != state_->qualified.end()) {
    for (const auto& [inventory, launch] : checked->second) {
      inventory->CheckArguments(launch, ordinary_formals, workspace_formal);
    }
    return;
  }
  const auto& callers = CallersOf(group->name_);
  CHECK(!callers.empty()) << "Peer Group has no root managed workspace allocation";
  const auto group_layout = state_->layout.BindRoot(group);
  std::vector<std::pair<RootInventory*, FunctionPtr>> qualified;
  std::optional<SyncDispatchCoordinates> coordinates;
  bool complete_coordinates = true;
  for (const auto& caller : callers) {
    const auto enclosing = state_->program->GetFunction(caller);
    CHECK(enclosing) << "Peer Group GM forwarding has no enclosing function";
    const bool spmd = enclosing->func_type_ == FunctionType::Spmd;
    const auto launch = spmd ? enclosing : group;
    auto launch_layout = group_layout;
    if (spmd) {
      CHECK(DirectForwarding(launch, group))
          << "Peer Group SPMD wrapper must directly forward the dispatch ABI";
      launch_layout = state_->layout.BindRoot(launch);
      CHECK(launch_layout->Layout().offsets == group_layout->Layout().offsets &&
            launch_layout->Layout().bytes == group_layout->Layout().bytes)
          << "Peer Group SPMD wrapper uses another workspace layout";
    }
    const auto roots = spmd ? CallersOf(caller) : std::set<std::string>{caller};
    CHECK(!roots.empty()) << "Peer Group SPMD launch has no root managed workspace allocation";
    bool sharded = false;
    if (spmd) {
      const auto [sharding, first] = state_->sharding.try_emplace(group->name_, false);
      if (first) sharding->second = HasNativeSharding(group, state_->program);
      sharded = sharding->second;
    }
    for (const auto& name : roots) {
      const auto root = state_->program->GetFunction(name);
      CHECK(root && root->func_type_ == FunctionType::Orchestration && CallersOf(name).empty())
          << "Peer Group GM forwarding requires a direct root orchestration";
      auto& inventory = state_->roots[name];
      if (!inventory) inventory = std::make_unique<RootInventory>(root, state_->launches, state_->program);
      inventory->Check(launch, ordinary_formals, workspace_formal, launch_layout->Layout().Elements(),
                       state_->invocation, sharded);
      if (complete_coordinates) {
        if (auto evidence = inventory->LaunchCoordinates(launch, state_->invocation)) {
          JoinCoordinates(coordinates, evidence->block_count.first, evidence->block_count.second,
                          std::move(evidence->width_parameters));
        } else {
          complete_coordinates = false;
        }
      }
      qualified.emplace_back(inventory.get(), launch);
    }
  }
  state_->qualified.emplace(group->name_, std::move(qualified));
  if (complete_coordinates) {
    state_->coordinates.emplace(group->name_, std::move(coordinates));
  }
}

std::optional<SyncDispatchCoordinates> PeerInvocationGmEvidence::TryLaunchCoordinates(
    const FunctionPtr& group, size_t workspace_formal) {
  const auto& callers = CallersOf(group->name_);
  if (callers.empty() || !std::all_of(callers.begin(), callers.end(), [&](const auto& caller) {
        const auto function = state_->program->GetFunction(caller);
        return function && function->func_type_ == FunctionType::Spmd;
      })) {
    return std::nullopt;
  }
  try {
    // Optional loop evidence cannot report a workspace refusal before the
    // definitive verification knows all of this endpoint's ordinary operands.
    Check(group, {}, workspace_formal);
  } catch (const ValueError&) {
    return std::nullopt;
  }
  const auto found = state_->coordinates.find(group->name_);
  return found == state_->coordinates.end() ? std::nullopt : found->second;
}

void PeerInvocationGmEvidence::CheckHostRankOneExtent(const FunctionPtr& group, size_t formal) const {
  const auto expected = As<TensorType>(group->params_.at(formal)->GetType());
  const auto extent = expected && expected->shape_.size() == 1 ? As<Var>(expected->shape_.front()) : nullptr;
  CHECK(expected && !expected->tensor_view_ && extent && GetScalarDtype(extent) == DataType::INDEX)
      << "Peer scalar GM address requires a plain symbolic rank-one Group formal";
  for (const auto& [inventory, launch] : state_->qualified.at(group->name_)) {
    inventory->CheckHostRankOneExtent(launch, formal, *expected);
  }
}

std::map<size_t, ExprPtr> PeerInvocationGmEvidence::CheckAuthoredGM(const FunctionPtr& group,
                                                                    size_t workspace_formal,
                                                                    const std::vector<PeerGmEffects>& phases,
                                                                    bool disjoint_gm) {
  std::set<size_t> ordinary;
  for (const auto& phase : phases) ordinary.insert(phase.formals.begin(), phase.formals.end());
  // A fresh allocation at one call cannot establish separation at a different
  // call. A slice retains its origin, even when the two formals have disjoint names.
  auto isolated = ordinary;
  const auto& callers = CallersOf(group->name_);
  CHECK(!callers.empty()) << "Authored GM group has no original root launch";
  for (const auto& caller : callers) {
    const auto enclosing = state_->program->GetFunction(caller);
    CHECK(enclosing) << "Authored GM group has no enclosing function";
    const bool spmd = enclosing->func_type_ == FunctionType::Spmd;
    const auto launch = spmd ? enclosing : group;
    CHECK(!spmd || DirectForwarding(launch, group))
        << "Authored GM SPMD wrapper must directly forward the Group ABI";
    const auto roots = spmd ? CallersOf(caller) : std::set<std::string>{caller};
    CHECK(!roots.empty()) << "Authored GM group has no original root launch";
    for (const auto& name : roots) {
      const auto root = state_->program->GetFunction(name);
      CHECK(root && root->func_type_ == FunctionType::Orchestration && CallersOf(name).empty())
          << "Authored GM synchronization requires direct root orchestration";
      auto& inventory = state_->authored_roots[name];
      if (!inventory) {
        inventory = std::make_unique<RootInventory>(root, state_->launches, state_->program, true);
      }
      inventory->CheckAuthoredGM(launch, workspace_formal, phases, ordinary, disjoint_gm);
      inventory->RetainIsolatedFresh(launch, ordinary, isolated);
    }
  }
  return MakeGmDomains(group, ordinary, isolated);
}

std::map<size_t, ExprPtr> PeerInvocationGmEvidence::FifoGmDomains(const FunctionPtr& group,
                                                                  const std::set<size_t>& ordinary) const {
  auto isolated = ordinary;
  for (const auto& [inventory, launch] : state_->qualified.at(group->name_)) {
    inventory->RetainIsolatedFresh(launch, ordinary, isolated);
  }
  return MakeGmDomains(group, ordinary, isolated);
}

void PeerInvocationGmEvidence::CheckScalarReads(const FunctionPtr& group, const PeerGmEffects& effects,
                                                bool disjoint_gm) const {
  if (effects.scalar_reads.empty() || effects.writes.empty()) {
    return;
  }
  for (const auto& [inventory, launch] : state_->qualified.at(group->name_)) {
    inventory->CheckScalarReads(launch, effects, disjoint_gm);
  }
}

}  // namespace pypto::ir::sync
