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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_MINIMUM_FILLPAD_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_MINIMUM_FILLPAD_ANALYSIS_H_

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// A minimum-fill-only prefix is metadata until its consumers run. Deferring
// the setter lets each consumer handle an empty prefix without sending zero
// to native code. Only full producers qualify: native fillpad copies whole
// physical rows before replacing their column tails.
class BufferMinimumFillPadAnalysis : private IRVisitor {
 public:
  explicit BufferMinimumFillPadAnalysis(const StmtPtr& body) { VisitStmt(body); }

  bool DefersSetter(const VarPtr& prefix) const {
    const auto found = prefixes_.find(prefix.get());
    return found != prefixes_.end() && found->second.min_fills != 0 &&
           found->second.min_fills == found->second.uses;
  }

  bool Handles(const CallPtr& call) const {
    return minimum_fills_.count(call.get()) && DefersSetter(AsVarLike(call->args_[0]));
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    const auto prefix = prefixes_.find(var.get());
    if (prefix != prefixes_.end()) {
      ++prefix->second.uses;
    }
  }

  void VisitExpr_(const IterArgPtr& argument) override { VisitVarLike_(argument); }

  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) {
      VisitExpr(argument->initValue_);
    }
    IRVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) {
      VisitExpr(argument->initValue_);
    }
    IRVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const AssignStmtPtr& statement) override {
    VisitExpr(statement->value_);
    const auto call = As<Call>(statement->value_);
    if (IsFullProducer(call, statement->var_)) {
      full_sources_.insert(statement->var_.get());
    } else if (IsPrefixSetter(call)) {
      prefixes_.try_emplace(statement->var_.get());
    }
  }

  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "tile.fillpad") && call->args_.size() == 1 &&
        call->GetKwarg<PadValue>("pad_value", PadValue::zero) == PadValue::min) {
      const auto prefix = prefixes_.find(AsVarLike(call->args_[0]).get());
      const auto* recipe = backend::FindLogicalBufferConversionRecipe(call->op_->name_);
      if (prefix != prefixes_.end() && recipe && backend::SupportsLogicalBufferConversion(*recipe, call)) {
        ++prefix->second.min_fills;
        minimum_fills_.insert(call.get());
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  static bool IsFullProducer(const CallPtr& call, const VarPtr& result) {
    if (!call) {
      return false;
    }
    const auto type = backend::StaticLogicalBufferDescriptor(result->GetType());
    if (!type || type->dtype_ != DataType::FP32 || type->shape_.size() != 2 ||
        type->valid_shape_ != type->shape_) {
      return false;
    }
    const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call);
    return IsOp(call, "tile.load") || (recipe && backend::SupportsLogicalBufferElementwise(*recipe, call));
  }

  bool IsPrefixSetter(const CallPtr& call) const {
    if (!IsOp(call, "tile.set_validshape") || call->args_.size() != 3 ||
        !full_sources_.count(AsVarLike(call->args_[0]).get())) {
      return false;
    }
    const auto rows = As<ConstInt>(call->args_[1]);
    const auto width = AsVarLike(call->args_[2]);
    const auto width_type = width ? As<ScalarType>(width->GetType()) : nullptr;
    return rows && rows->value_ > 0 && width_type &&
           (width_type->dtype_ == DataType::INDEX || width_type->dtype_ == DataType::INT64);
  }

  struct PrefixUses {
    size_t uses = 0;
    size_t min_fills = 0;
  };

  std::unordered_set<const Var*> full_sources_;
  std::unordered_map<const Var*, PrefixUses> prefixes_;
  std::unordered_set<const Call*> minimum_fills_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_MINIMUM_FILLPAD_ANALYSIS_H_
