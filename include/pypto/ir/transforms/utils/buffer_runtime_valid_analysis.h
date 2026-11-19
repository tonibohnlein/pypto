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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_acc_trim_analysis.h"

namespace pypto::ir {

// A dynamic descriptor's -1 fields state availability, never equality of the
// actual native metadata. This input-owned, use-site checker retains expressions
// supplied by dominating allocation/setter commands and proves relations under
// the original guard. It is intentionally scoped to ordinary alloc metadata:
// slot/borrowed handles need a native-metadata ownership export first.
//
// One walk and indexed per-handle state; control boundaries advance an epoch
// rather than scanning every handle. Unknown joins/backedges lose evidence,
// never become an empty active extent. Arithmetic proof work is additional and
// uses the existing analyzer. No branch outcomes or dynamic visits are unfolded.
class BufferRuntimeValidAnalysis : public IRVisitor {
 public:
  struct Issue {
    CallPtr call;
    std::string message;
  };
  explicit BufferRuntimeValidAnalysis(const FunctionPtr& function, bool collect_guards = false)
      : analyzer_(std::make_shared<arith::Analyzer>()), collect_guards_(collect_guards) {
    if (function) {
      core_ = function->func_type_;
      BindBufferAccTrimParameters(function, *analyzer_, ready_);
      if (function->body_) VisitStmt(function->body_);
    }
  }
  const std::optional<Issue>& issue() const { return issue_; }
  bool HasElidableGuards() const {
    return std::any_of(redundant_guards_.begin(), redundant_guards_.end(),
                       [](const auto& entry) { return entry.second; });
  }
  bool Elides(const IfStmtPtr& branch) const {
    const auto found = redundant_guards_.find(branch.get());
    return found != redundant_guards_.end() && found->second;
  }
  std::unordered_set<const Call*> NonEmptyTransfers() const {
    std::unordered_set<const Call*> proved;
    for (const auto& [call, positive] : transfer_guards_)
      if (positive) proved.insert(call);
    return proved;
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto call = As<Call>(statement->value_);
    const auto type = As<BufferType>(statement->var_->GetType());
    if (call && type && IsOp(call, "buffer.alloc")) {
      owners_.insert(statement->var_.get());
      if (call->args_.size() == 2) addressed_.insert(statement->var_.get());
      const auto values = call->args_.empty() ? nullptr : As<MakeTuple>(call->args_[0]);
      if (values) {
        State state{epoch_, {}};
        size_t next = 0;
        for (size_t axis = 0; axis < 2 && type->valid_shape_.size() == 2; ++axis) {
          if (type->valid_shape_[axis] < 0) {
            if (next < values->elements_.size()) state.extents[axis] = values->elements_[next++];
          } else {
            state.extents[axis] = Constant(type->valid_shape_[axis], statement->span_);
          }
        }
        states_[statement->var_.get()] = std::move(state);
      }
    } else if (const auto scalar = As<ScalarType>(statement->var_->GetType());
               scalar && scalar->dtype_ == DataType::BOOL && !call) {
      // Boolean aliases preserve comparisons of actual SSA values. Do not bind
      // numerical arithmetic: cancellation could require an unproved machine-
      // overflow premise. Scalar-memory readiness remains a separate contract.
      guard_aliases_.emplace(statement->var_.get(), statement->value_);
      if (ready_.insert(statement->var_.get()).second) scalar_definitions_.push_back(statement->var_);
    }
    if (const auto scalar = As<ScalarType>(statement->var_->GetType());
        scalar && (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt())) {
      const auto range = Bounds(statement->value_);
      if (range) {
        if (ready_.insert(statement->var_.get()).second) scalar_definitions_.push_back(statement->var_);
        analyzer_->const_int_bound.Update(statement->var_, *range);
      }
    }
    if (call) CheckCall(call);
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    if (const auto call = As<Call>(statement->expr_)) CheckCall(call);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    const auto checkpoint = scalar_definitions_.size();
    const auto range = BoundBufferLoopCoordinate(
        loop, *analyzer_, [this](const VarPtr& var) { return ready_.count(var.get()) != 0; }, bound_cache_);
    if (ready_.insert(loop->loop_var_.get()).second) {
      scalar_definitions_.push_back(loop->loop_var_);
      if (range) analyzer_->const_int_bound.Update(loop->loop_var_, *range);
    }
    ++epoch_;
    bound_cache_.clear();
    VisitStmt(loop->body_);
    RestoreScalars(checkpoint);
    ++epoch_;
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    const auto checkpoint = scalar_definitions_.size();
    bound_cache_.clear();
    ++epoch_;
    {
      BufferAccTrimGuardScope constraint(loop->condition_, true, analyzer_, guard_aliases_);
      VisitStmt(loop->body_);
      RestoreScalars(checkpoint);
    }
    ++epoch_;
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    const auto checkpoint = scalar_definitions_.size();
    if (collect_guards_ && !branch->else_body_ && branch->return_vars_.empty() &&
        branch->leading_comments_.empty()) {
      auto [entry, inserted] = redundant_guards_.emplace(branch.get(), true);
      if (entry->second) entry->second = ProvesValidityGuard(branch->condition_);
    }
    bound_cache_.clear();
    {
      BufferAccTrimGuardScope constraint(branch->condition_, true, analyzer_, guard_aliases_);
      VisitStmt(branch->then_body_);
      RestoreScalars(checkpoint);
    }
    // Neither arm's mutations prove the other's initial metadata. The initial
    // slice conservatively requires fresh setters in that arm when necessary.
    ++epoch_;
    if (branch->else_body_) {
      bound_cache_.clear();
      BufferAccTrimGuardScope constraint(branch->condition_, false, analyzer_, guard_aliases_);
      VisitStmt(*branch->else_body_);
      RestoreScalars(checkpoint);
    }
    ++epoch_;
  }

 private:
  struct State {
    uint64_t epoch;
    std::array<ExprPtr, 2> extents;
  };
  static ExprPtr Constant(int64_t value, const Span& span) {
    return std::make_shared<ConstInt>(value, DataType::INDEX, span);
  }
  static bool AtomicInteger(const ExprPtr& expression) {
    const auto type = expression ? As<ScalarType>(expression->GetType()) : nullptr;
    return type && (type->dtype_.IsSignedInt() || type->dtype_ == DataType::INDEX) &&
           (AsVarLike(expression) || As<ConstInt>(expression));
  }
  std::optional<arith::ConstIntBound> Bounds(const ExprPtr& value) {
    return BoundBufferAccTrimScalar(
        value, *analyzer_, [this](const VarPtr& var) { return ready_.count(var.get()) != 0; }, &bound_cache_);
  }
  bool Positive(const ExprPtr& extent) {
    if (AtomicInteger(extent) && analyzer_->const_int_bound(extent).min_value >= 1) return true;
    const auto bound = Bounds(extent);
    return bound && bound->min_value >= 1;
  }
  bool ProvesValidityGuard(const ExprPtr& predicate) {
    // Follow only the conjunction of positive-extent comparisons emitted by
    // Buffer lowering. The proof is about original SSA values at this cut.
    std::vector<ExprPtr> pending{predicate};
    std::unordered_set<const Expr*> seen;
    while (!pending.empty()) {
      const auto expr = std::move(pending.back());
      pending.pop_back();
      if (!seen.insert(expr.get()).second) continue;
      if (const auto both = As<And>(expr)) {
        pending.push_back(both->left_);
        pending.push_back(both->right_);
        continue;
      }
      const auto cmp = As<Gt>(expr);
      const auto zero = cmp ? As<ConstInt>(cmp->right_) : nullptr;
      if (!zero || zero->value_ != 0 || !Positive(cmp->left_)) return false;
    }
    return true;
  }
  bool Equal(const ExprPtr& a, const ExprPtr& b) {
    // The predicate query includes the analyzer's guarded comparison facts;
    // CanProveEqual's subtraction-interval query alone loses SSA equalities.
    return AtomicInteger(a) && AtomicInteger(b) && analyzer_->CanProve(MakeEq(a, b));
  }
  std::optional<std::array<ExprPtr, 2>> Extents(const ExprPtr& expression) const {
    const auto var = AsVarLike(expression);
    const auto type = var ? As<BufferType>(var->GetType()) : nullptr;
    if (!type || type->valid_shape_.size() != 2) return std::nullopt;
    std::array<ExprPtr, 2> result;
    const auto found = states_.find(var.get());
    for (size_t axis = 0; axis < 2; ++axis) {
      if (type->valid_shape_[axis] >= 0) {
        result[axis] = Constant(type->valid_shape_[axis], var->span_);
      } else {
        if (!owners_.count(var.get()) || found == states_.end() || found->second.epoch != epoch_ ||
            !found->second.extents[axis])
          return std::nullopt;
        result[axis] = found->second.extents[axis];
      }
    }
    return result;
  }
  void Refuse(const CallPtr& call, const std::string& message) {
    if (!issue_) issue_ = Issue{call, message};
  }
  void CheckCall(const CallPtr& call) {
    if (collect_guards_ && (IsOp(call, "buffer.load") || IsOp(call, "buffer.store")) &&
        call->args_.size() == 4) {
      const auto extents = As<MakeTuple>(call->args_[2]);
      if (extents && std::any_of(extents->elements_.begin(), extents->elements_.end(),
                                 [](const ExprPtr& extent) { return !As<ConstInt>(extent); })) {
        auto [entry, inserted] = transfer_guards_.emplace(call.get(), true);
        if (entry->second) {
          entry->second = std::all_of(extents->elements_.begin(), extents->elements_.end(),
                                      [this](const ExprPtr& extent) { return Positive(extent); });
        }
      }
    }
    // A dynamic Acc is a store-only metadata alias. No producer/carry/window
    // or unknown consumer inherits permission from its -1 descriptor fields.
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto type = As<BufferType>(call->args_[i]->GetType());
      if (!type || type->memory_space_ != MemorySpace::Acc ||
          std::none_of(type->valid_shape_.begin(), type->valid_shape_.end(),
                       [](int64_t extent) { return extent < 0; }))
        continue;
      if (i != 0 || (!IsOp(call, "buffer.store") && !IsOp(call, "buffer.set_validshape")) ||
          !backend::IsBufferAccRowTrimDescriptor(type)) {
        Refuse(call, "Runtime Acc aliases require the qualified row-only default store metadata contract");
        return;
      }
    }
    if (IsOp(call, "buffer.store") && call->args_.size() == 4) {
      const auto type = As<BufferType>(call->args_[0]->GetType());
      if (type && type->memory_space_ == MemorySpace::Acc &&
          std::any_of(type->valid_shape_.begin(), type->valid_shape_.end(),
                      [](int64_t extent) { return extent < 0; })) {
        if (!IsBufferAccTrimTargetQualified()) {
          Refuse(call, "Runtime Acc row stores require a qualified selected A2/A3 native target");
          return;
        }
        const auto owner = AsVarLike(call->args_[0]);
        const auto extents = Extents(call->args_[0]);
        const auto transfer = As<MakeTuple>(call->args_[2]);
        if (core_ != FunctionType::AIC || !owner || !addressed_.count(owner.get()) ||
            !backend::IsBufferAccRowTrimDescriptor(type) || !extents || !transfer ||
            transfer->elements_.size() != 2 || GetIntKwarg(call->kwargs_, "atomic", 0) != 0) {
          Refuse(call, "Runtime Acc row stores require owned addressed AIC metadata and default TSTORE");
        } else if (!Equal((*extents)[1], Constant(type->shape_[1], call->span_)) ||
                   !Equal(transfer->elements_[0], (*extents)[0]) ||
                   !Equal(transfer->elements_[1], (*extents)[1])) {
          Refuse(call,
                 "Runtime Acc store requires restored rows and fixed full columns matching the GM partition");
        } else {
          const auto range = Bounds((*extents)[0]);
          if (!range || range->min_value < 1 || range->max_value > type->shape_[0])
            Refuse(call,
                   "Runtime Acc store requires proved positive rows within physical capacity at this use");
        }
        return;
      }
    }

    if (IsOp(call, "buffer.set_validshape") && call->args_.size() == 2) {
      const auto handle = AsVarLike(call->args_[0]);
      const auto values = As<MakeTuple>(call->args_[1]);
      if (handle && owners_.count(handle.get()) && values && values->elements_.size() == 2) {
        states_[handle.get()] = {epoch_, {values->elements_[0], values->elements_[1]}};
      } else {
        // An unsupported native metadata alias cannot establish or preserve
        // ordinary-handle facts merely because its payload bytes overlap.
        ++epoch_;
      }
      return;
    }
    if (const auto* conversion = backend::FindBufferConversionRecipe(call->op_->name_);
        conversion && conversion->kind == backend::BufferConversionKind::Cast && call->args_.size() == 2) {
      bool dynamic = false;
      for (const auto& arg : call->args_) {
        if (const auto type = As<BufferType>(arg->GetType()))
          for (const auto extent : type->valid_shape_) dynamic |= extent < 0;
      }
      if (!dynamic) return;
      const auto source = Extents(call->args_[0]);
      const auto target = Extents(call->args_[1]);
      const auto type = As<BufferType>(call->args_[0]->GetType());
      if (!source || !target || !type) {
        Refuse(call,
               "Runtime cast valid metadata is unavailable at this use; restore owned allocation metadata");
      } else if (!Equal((*source)[0], (*target)[0]) ||
                 !Equal((*source)[1], Constant(type->shape_[1], call->span_)) ||
                 !Equal((*target)[1], Constant(type->shape_[1], call->span_))) {
        Refuse(call, "Runtime cast requires proved matching runtime rows and fixed full columns at this use");
      } else if (!AtomicInteger((*source)[0]) || !analyzer_->CanProveGreaterEqual((*source)[0], 1)) {
        Refuse(call, "Runtime cast requires a proved positive-row guard at this use");
      }
      return;
    }
    const auto* recipe = backend::FindBufferAxisRecipe(call->op_->name_);
    if (!recipe || call->args_.size() != 3 || (recipe->axis == 0 && recipe->reduction)) return;
    bool dynamic = false;
    for (const auto& arg : call->args_) {
      if (const auto type = As<BufferType>(arg->GetType())) {
        for (const auto extent : type->valid_shape_) dynamic |= extent < 0;
      }
    }
    // Existing static axis recipes retain their established active domains.
    if (!dynamic) return;
    if (recipe->axis == 0 && std::string_view(recipe->buffer_op) != "buffer.col_expand_mul") return;
    const auto source = Extents(call->args_[0]);
    const auto carrier = Extents(call->args_[1]);
    const auto target = Extents(call->args_[2]);
    if (!source || !target || (!recipe->reduction && !carrier)) {
      Refuse(call,
             "Row operation runtime valid metadata is unavailable at this use; restore owned "
             "allocation metadata after control-flow boundaries");
      return;
    }
    const auto type = As<BufferType>(call->args_[0]->GetType());
    if (!type || type->shape_.size() != 2) return;
    const auto columns = Constant(type->shape_[1], call->span_);
    const auto one = Constant(1, call->span_);
    const bool carrier_valid =
        recipe->reduction ||
        (recipe->axis == 0 ? Equal((*carrier)[0], one) && Equal((*carrier)[1], columns)
                           : Equal((*carrier)[0], (*source)[0]) && Equal((*carrier)[1], one));
    if (!Equal((*source)[0], (*target)[0]) || !Equal((*source)[1], columns) ||
        !Equal((*target)[1], recipe->reduction ? one : columns) || !carrier_valid) {
      Refuse(call,
             "Row operation requires proved matching runtime rows and fixed full columns at "
             "this use; unrelated dynamic descriptor fields are not equal metadata");
    } else if (!AtomicInteger((*source)[0]) || !analyzer_->CanProveGreaterEqual((*source)[0], 1)) {
      Refuse(call, "Row operation requires a proved positive-row guard at this use");
    }
  }
  void RestoreScalars(size_t checkpoint) {
    bound_cache_.clear();
    while (scalar_definitions_.size() > checkpoint) {
      analyzer_->Unbind(scalar_definitions_.back());
      ready_.erase(scalar_definitions_.back().get());
      guard_aliases_.erase(scalar_definitions_.back().get());
      scalar_definitions_.pop_back();
    }
  }
  FunctionType core_ = FunctionType::Opaque;
  std::unordered_set<const Var*> ready_, addressed_;
  BufferAccTrimBoundCache bound_cache_;
  std::unordered_map<const Var*, ExprPtr> guard_aliases_;
  std::vector<VarPtr> scalar_definitions_;
  uint64_t epoch_ = 0;
  arith::AnalyzerPtr analyzer_;
  std::unordered_set<const Var*> owners_;
  std::unordered_map<const Var*, State> states_;
  std::optional<Issue> issue_;
  bool collect_guards_;
  // The same immutable statement/call can be reused. Every occurrence must
  // have the witness before a pointer-keyed rewrite or emission can use it.
  std::unordered_map<const IfStmt*, bool> redundant_guards_;
  std::unordered_map<const Call*, bool> transfer_guards_;
};

// Remove only validity guards whose exact predicates follow from original
// lexical facts. An already placed Buffer input can share the same rewrite.
FunctionPtr ElideDominatedValidityGuards(const FunctionPtr& function);

// A selected descriptor's native runtime obligations apply equally to access
// analysis, actual-token lowering, ownership and code generation.
inline void RequireBufferRuntimeValid(const FunctionPtr& function) {
  const BufferRuntimeValidAnalysis metadata(function);
  if (const auto& issue = metadata.issue()) CHECK_SPAN(false, issue->call->span_) << issue->message;
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_
