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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_SCALAR_READINESS_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_SCALAR_READINESS_H_

#include <functional>
#include <unordered_set>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto::ir::sync {
// Readiness is distinct from a numerical range. Pure original machine
// expressions keep their original semantics; scalar memory reads and nested
// calls need a separate visibility contract and are never admitted here.
inline size_t CheckSyncScalarReadiness(const ExprPtr& expression,
                                       const std::function<bool(const VarPtr&)>& ready) {
  std::unordered_set<const Expr*> visited;
  std::vector<ExprPtr> pending{expression};
  while (!pending.empty()) {
    const auto current = pending.back();
    pending.pop_back();
    CHECK(current) << "Synchronization: missing expression";
    if (!visited.insert(current.get()).second) continue;
    if (const auto variable = AsVarLike(current)) {
      CHECK_SPAN(ready(variable), current->span_)
          << "Synchronization: scalar or descriptor use lacks a ready definition";
    } else if (As<ConstInt>(current) || As<ConstFloat>(current) || As<ConstBool>(current)) {
      continue;
    } else if (const auto binary = As<BinaryExpr>(current)) {
      pending.push_back(binary->left_);
      pending.push_back(binary->right_);
    } else if (const auto unary = As<UnaryExpr>(current)) {
      pending.push_back(unary->operand_);
    } else if (const auto tuple = As<MakeTuple>(current)) {
      pending.insert(pending.end(), tuple->elements_.begin(), tuple->elements_.end());
    } else if (const auto item = As<TupleGetItemExpr>(current)) {
      pending.push_back(item->tuple_);
    } else {
      CHECK_SPAN(false, current->span_)
          << "Synchronization: control/metadata expression requires an explicit readiness and visibility "
             "contract (nested calls and scalar memory reads are unavailable)";
    }
  }
  return visited.size();
}
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_SCALAR_READINESS_H_
