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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_COORDINATE_EVAL_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_COORDINATE_EVAL_H_

#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/scalar_expr.h"
#include "src/ir/transforms/utils/physical_sync_commands.h"

namespace pypto::ir::sync {

inline bool IsCoordinateScalar(const ExprPtr& expr) {
  const auto scalar = As<ScalarType>(expr->GetType());
  return scalar && (scalar->dtype_ == DataType::INDEX || scalar->dtype_ == DataType::BOOL);
}

// Admit only the emitter's checked INDEX and BOOL recipes. The folder
// detects signed arithmetic overflow and receives two concrete operands,
// so identities over unknown runtime values cannot select a path.
template <typename Evaluate>
ExprPtr FoldCoordinate(const ExprPtr& expr, const Evaluate& evaluate) {
  if (!IsCoordinateScalar(expr)) return nullptr;
  if (As<ConstBool>(expr)) return expr;
  if (const auto literal = As<ConstInt>(expr)) return literal->dtype() == DataType::INDEX ? expr : nullptr;
  if (As<Var>(expr)) return evaluate(expr);
  const auto bin = As<BinaryExpr>(expr);
  if (!bin || !physical::IsCommandCoordinateBinary(expr)) return nullptr;
  const auto lhs = evaluate(bin->left_), rhs = evaluate(bin->right_);
  if (!lhs || !rhs) return nullptr;
  if (As<FloorDiv>(expr) || As<FloorMod>(expr)) {
    const auto divisor = As<ConstInt>(rhs);
    if (!divisor || divisor->value_ <= 0) return nullptr;
  }
  const auto dtype = GetScalarDtype(expr);
  auto folded = arith::TryConstFoldBinary(expr->GetKind(), lhs, rhs, dtype);
  return folded && GetScalarDtype(folded) == dtype ? folded : nullptr;
}

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_COORDINATE_EVAL_H_
