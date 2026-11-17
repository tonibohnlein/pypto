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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"

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
  explicit BufferRuntimeValidAnalysis(const FunctionPtr& function)
      : analyzer_(std::make_shared<arith::Analyzer>()) {
    if (function && function->body_) VisitStmt(function->body_);
  }
  const std::optional<Issue>& issue() const { return issue_; }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto call = As<Call>(statement->value_);
    const auto type = As<BufferType>(statement->var_->GetType());
    if (call && type && IsOp(call, "buffer.alloc")) {
      owners_.insert(statement->var_.get());
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
               scalar && scalar->dtype_ == DataType::BOOL && !call && SafeGuard(statement->value_)) {
      // Boolean aliases preserve comparisons of actual SSA values. Do not bind
      // numerical arithmetic: cancellation could require an unproved machine-
      // overflow premise. Scalar-memory readiness remains a separate contract.
      analyzer_->Bind(statement->var_, statement->value_);
    }
    if (call) CheckCall(call);
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    if (const auto call = As<Call>(statement->expr_)) CheckCall(call);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    ++epoch_;
    VisitStmt(loop->body_);
    ++epoch_;
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    ++epoch_;
    {
      const auto condition = QualifiedGuard(loop->condition_);
      auto constraint = analyzer_->GetConstraintContext(
          condition ? condition : std::make_shared<ConstBool>(true, loop->span_));
      VisitStmt(loop->body_);
    }
    ++epoch_;
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    const auto condition = QualifiedGuard(branch->condition_);
    {
      auto constraint = analyzer_->GetConstraintContext(
          condition ? condition : std::make_shared<ConstBool>(true, branch->span_));
      VisitStmt(branch->then_body_);
    }
    // Neither arm's mutations prove the other's initial metadata. The initial
    // slice conservatively requires fresh setters in that arm when necessary.
    ++epoch_;
    if (branch->else_body_) {
      auto constraint = analyzer_->GetConstraintContext(
          condition ? MakeNot(condition, branch->span_) : std::make_shared<ConstBool>(true, branch->span_));
      VisitStmt(*branch->else_body_);
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
  static bool SafeGuard(const ExprPtr& expression) {
    if (As<ConstBool>(expression)) return true;
    if (const auto var = AsVarLike(expression)) {
      const auto type = As<ScalarType>(var->GetType());
      return type && type->dtype_ == DataType::BOOL;
    }
    if (const auto negation = As<Not>(expression)) return SafeGuard(negation->operand_);
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return false;
    if (As<And>(expression) || As<Or>(expression)) {
      return SafeGuard(binary->left_) && SafeGuard(binary->right_);
    }
    return (As<Eq>(expression) || As<Ne>(expression) || As<Lt>(expression) || As<Le>(expression) ||
            As<Gt>(expression) || As<Ge>(expression)) &&
           AtomicInteger(binary->left_) && AtomicInteger(binary->right_);
  }
  ExprPtr QualifiedGuard(const ExprPtr& condition) {
    // Check before simplifying: an unsupported arithmetic condition must not
    // become a usable fact through algebra requiring absent overflow evidence.
    if (!SafeGuard(condition)) return nullptr;
    return analyzer_->Simplify(condition);
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
  uint64_t epoch_ = 0;
  arith::AnalyzerPtr analyzer_;
  std::unordered_set<const Var*> owners_;
  std::unordered_map<const Var*, State> states_;
  std::optional<Issue> issue_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_
