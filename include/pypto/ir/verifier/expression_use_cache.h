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

#ifndef PYPTO_IR_VERIFIER_EXPRESSION_USE_CACHE_H_
#define PYPTO_IR_VERIFIER_EXPRESSION_USE_CACHE_H_

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_set>
#include <utility>

#include "pypto/ir/expr.h"

namespace pypto::ir::verifier_detail {
// Structural inspection, never execution/CSE: one expression-use root has
// stable bindings. New roots (including every direct call statement) restart
// the cache. Client context keeps attribute/type/result roles distinct.
class ExpressionUseCache {
  struct Key {
    const Expr* expression;
    std::string context;
    const void* role;
    bool assigns;
    bool operator==(const Key& other) const {
      return expression == other.expression && context == other.context && role == other.role &&
             assigns == other.assigns;
    }
  };
  struct Hash {
    size_t operator()(const Key& key) const {
      return std::hash<const Expr*>{}(key.expression) ^ (std::hash<std::string>{}(key.context) << 1) ^
             (std::hash<const void*>{}(key.role) << 2) ^ static_cast<size_t>(key.assigns);
    }
  };

 public:
  class Scope {
   public:
    Scope(ExpressionUseCache& owner, bool inspect) : owner_(owner), inspect_(inspect) {}
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    ~Scope() {
      if (--owner_.depth_ == 0) owner_.seen_.clear();
    }
    bool inspect() const { return inspect_; }

   private:
    ExpressionUseCache& owner_;
    bool inspect_;
  };

  Scope Enter(const ExprPtr& expression, std::string context = {}, const void* role = nullptr,
              bool assigns = false) {
    if (depth_++ == 0) seen_.clear();
    const bool inspect = seen_.insert({expression.get(), std::move(context), role, assigns}).second;
    return Scope(*this, inspect);
  }

 private:
  size_t depth_ = 0;
  std::unordered_set<Key, Hash> seen_;
};
}  // namespace pypto::ir::verifier_detail
#endif  // PYPTO_IR_VERIFIER_EXPRESSION_USE_CACHE_H_
