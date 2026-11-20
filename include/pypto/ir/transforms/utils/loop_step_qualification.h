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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_LOOP_STEP_QUALIFICATION_H_
#define PYPTO_IR_TRANSFORMS_UTILS_LOOP_STEP_QUALIFICATION_H_

#include <optional>
#include <unordered_map>
#include <vector>

#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto::ir {

inline constexpr auto kAttrBufferPositiveStepParameters = "buffer_positive_step_parameters";

// Shared selection/emission proof for the positive-step requirement of scf.for.
// Mathematical interval bounds alone are insufficient: a positive Add/Mul
// bound can wrap in the original scalar width before conversion to INDEX.
// Admit machine-qualified literals/negated literals, the native INDEX launch-count contract,
// caller-proved positive formals, and same-type SSA aliases or widening signed
// casts. Arithmetic and narrowing casts require a separate checked range proof.
class PositiveLoopStepProof {
 public:
  void BindCallerParameters(const FunctionPtr& function, const std::vector<int>& positive_formals) {
    for (const auto index : positive_formals) {
      CHECK_SPAN(index >= 0 && static_cast<size_t>(index) < function->params_.size(), function->span_)
          << "Positive Buffer loop step must refer to an original scalar formal";
      const auto& param = function->params_[index];
      const auto dtype = GetScalarDtype(param);
      CHECK_SPAN(dtype == DataType::INDEX || dtype.IsSignedInt(), param->span_)
          << "Positive Buffer loop step requires a signed integer argument";
      positive_.emplace(param.get(), Fact{true, index});
    }
  }

  bool IsPositive(const ExprPtr& expression) const { return Prove(expression).positive; }

  std::optional<int> CallerParameter(const ExprPtr& expression) const { return Prove(expression).formal; }

  // Call in definition order on SSA IR. Store facts so a loop does not walk
  // back through long alias chains. Repeated definitions invalidate the proof.
  void Define(const VarPtr& variable, const ExprPtr& value) {
    const auto fact = GetScalarDtype(variable) == GetScalarDtype(value) ? Prove(value) : Fact{};
    auto [found, inserted] = positive_.emplace(variable.get(), fact);
    if (!inserted) {
      found->second = {};
    }
  }

 private:
  struct Fact {
    bool positive = false;
    std::optional<int> formal = std::nullopt;
  };

  Fact Prove(const ExprPtr& expression) const {
    const auto dtype = GetScalarDtype(expression);
    if (dtype != DataType::INDEX && !dtype.IsSignedInt()) {
      return {};
    }
    // Simplify owns folding larger constant-expression DAGs before selection.
    const auto negation = As<Neg>(expression);
    if (As<ConstInt>(expression) || (negation && As<ConstInt>(negation->operand_))) {
      const auto constant = arith::TryEvaluateIntConstant(expression);
      return Fact{constant && *constant > 0};
    }
    if (const auto max_expr = As<Max>(expression)) {
      const auto left = Prove(max_expr->left_);
      if (left.positive && !left.formal) {
        return left;
      }
      const auto right = Prove(max_expr->right_);
      // Prefer a lexical witness if available, even when the other
      // operand also has a positive caller-derived range.
      return right.positive && (!left.positive || !right.formal) ? right : left;
    }
    if (const auto variable = AsVarLike(expression)) {
      const auto found = positive_.find(variable.get());
      return found == positive_.end() ? Fact{} : found->second;
    }
    if (const auto cast = As<Cast>(expression)) {
      const auto source = GetScalarDtype(cast->operand_);
      return (source == DataType::INDEX || source.IsSignedInt()) && dtype.GetBit() >= source.GetBit()
                 ? Prove(cast->operand_)
                 : Fact{};
    }
    const auto call = As<Call>(expression);
    return Fact{call && call->args_.empty() && dtype == DataType::INDEX &&
                (IsOp(call, "tensor.get_block_num") || IsOp(call, "tile.get_block_num") ||
                 IsOp(call, "buffer.get_block_num"))};
  }

  std::unordered_map<const Var*, Fact> positive_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LOOP_STEP_QUALIFICATION_H_
