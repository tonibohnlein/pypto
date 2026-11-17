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

#include <unordered_map>

#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto::ir {

// Shared selection/emission proof for the positive-step requirement of scf.for.
// Mathematical interval bounds alone are insufficient: a positive Add/Mul
// bound can wrap in the original scalar width before conversion to INDEX.
// Admit machine-qualified literals/negated literals, the native INDEX launch-count contract,
// and same-type SSA aliases. Dynamic arithmetic/casts need a separate checked
// range proof and remain unavailable here, even if their final bound is positive.
class PositiveLoopStepProof {
 public:
  bool IsPositive(const ExprPtr& expression) const {
    const auto dtype = GetScalarDtype(expression);
    if (dtype != DataType::INDEX && !dtype.IsSignedInt()) return false;
    // Preserve the previous literal syntax without recursively traversing a
    // potentially shared constant-expression DAG at every assignment. Ordinary
    // Simplify owns folding larger constant expressions before selection.
    const auto negation = As<Neg>(expression);
    if (As<ConstInt>(expression) || (negation && As<ConstInt>(negation->operand_))) {
      const auto constant = arith::TryEvaluateIntConstant(expression);
      return constant && *constant > 0;
    }
    if (const auto max_expr = As<Max>(expression)) {
      return IsPositive(max_expr->left_) || IsPositive(max_expr->right_);
    }
    if (const auto variable = AsVarLike(expression)) {
      const auto found = positive_.find(variable.get());
      return found != positive_.end() && found->second;
    }
    const auto call = As<Call>(expression);
    return call && call->args_.empty() && dtype == DataType::INDEX &&
           (IsOp(call, "tensor.get_block_num") || IsOp(call, "tile.get_block_num") ||
            IsOp(call, "buffer.get_block_num"));
  }

  // Call in definition order on SSA IR. Store facts, not expression aliases,
  // so long alias chains need constant work per definition. Repeated definitions
  // invalidate the fact rather than inheriting evidence from one assignment.
  void Define(const VarPtr& variable, const ExprPtr& value) {
    const bool positive = GetScalarDtype(variable) == GetScalarDtype(value) && IsPositive(value);
    auto [found, inserted] = positive_.emplace(variable.get(), positive);
    if (!inserted) found->second = false;
  }

 private:
  std::unordered_map<const Var*, bool> positive_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LOOP_STEP_QUALIFICATION_H_
