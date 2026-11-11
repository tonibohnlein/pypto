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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_TOKEN_NAMES_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_TOKEN_NAMES_H_

#include <set>
#include <string>

#include "pypto/ir/transforms/utils/var_collectors.h"

namespace pypto::ir::sync {
// Shared name reservation visits each expression DAG node once. A monotone
// suffix counter avoids rescanning previously rejected names.
class SyncTokenNames : public var_collectors::VarDefUseCollector {
 public:
  explicit SyncTokenNames(const FunctionPtr& function) {
    VisitFunction(function);
    for (const auto* var : GetAllVarRefs()) names_.insert(var->name_hint_);
  }
  void VisitExpr(const ExprPtr& expression) override {
    ++visits;
    if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
  }
  std::string Next() {
    std::string name;
    do {
      name = "sync_completion_" + std::to_string(next_++);
    } while (!names_.insert(name).second);
    return name;
  }
  size_t Nodes() const { return seen_.size(); }
  size_t visits = 0;

 private:
  std::set<const Expr*> seen_;
  std::set<std::string> names_;
  size_t next_ = 0;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_TOKEN_NAMES_H_
