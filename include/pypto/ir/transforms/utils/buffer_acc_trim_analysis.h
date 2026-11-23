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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ACC_TRIM_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ACC_TRIM_ANALYSIS_H_

#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/function.h"

namespace pypto::ir {

// Non-overflowing scalar subset shared by Tile qualification and actual Buffer
// metadata validation. Readiness remains caller-owned; unavailable variables
// are not interpreted as free mathematical parameters.
using BufferAccTrimBoundCache = std::unordered_map<const Expr*, std::optional<arith::ConstIntBound>>;
std::optional<arith::ConstIntBound> BoundBufferAccTrimScalar(const ExprPtr& expr, arith::Analyzer& analyzer,
                                                             const std::function<bool(const VarPtr&)>& ready,
                                                             BufferAccTrimBoundCache* cache = nullptr);
std::optional<arith::ConstIntBound> BoundBufferLoopCoordinate(const ForStmtPtr& loop,
                                                              arith::Analyzer& analyzer,
                                                              const std::function<bool(const VarPtr&)>& ready,
                                                              BufferAccTrimBoundCache& cache);
// A supplied or already-evaluated narrow signed value fits its machine type,
// independently of the expression or memory contents that produced it.
std::optional<arith::ConstIntBound> NarrowSignedRepresentation(const VarPtr& var);
void BindBufferAccTrimParameters(const FunctionPtr& function, arith::Analyzer& analyzer,
                                 std::unordered_set<const Var*>& ready,
                                 const std::vector<int>& positive_loop_parameters = {},
                                 const std::map<size_t, std::pair<int64_t, int64_t>>& parameter_bounds = {});

// Enter only distinct atomic integer comparison facts, with De Morgan handling
// for false disjunctions. General Boolean DAGs are never sent to the recursive
// arithmetic simplifier. Unsupported facts remain unavailable. Aliases are
// original dominating Boolean SSA definitions, not numerical substitutions.
class BufferAccTrimGuardScope {
 public:
  BufferAccTrimGuardScope(const ExprPtr& condition, bool truth, const arith::AnalyzerPtr& analyzer,
                          const std::unordered_map<const Var*, ExprPtr>& aliases);
  ~BufferAccTrimGuardScope();
  BufferAccTrimGuardScope(const BufferAccTrimGuardScope&) = delete;
  BufferAccTrimGuardScope& operator=(const BufferAccTrimGuardScope&) = delete;

 private:
  std::vector<arith::ConstraintContext> scopes_;
};

// Qualification belongs to the selected native implementation. A core kind
// alone does not establish an A2/A3 runtime Acc store contract.
bool IsBufferAccTrimTargetQualified();

// One input-owned capability for addressed, row-only Acc store aliases. It is
// reconstructed after placement; it never changes the writer/carry descriptor,
// proves payload completion, or supplies exact dynamic GM boundary selectors.
class BufferAccTrimAnalysis {
 public:
  BufferAccTrimAnalysis(const FunctionPtr& function, bool addressed,
                        const std::map<size_t, std::pair<int64_t, int64_t>>& parameter_bounds = {});
  bool Qualifies(const TypePtr& type) const { return types_.count(type.get()) != 0; }
  bool Qualifies(const CallPtr& call) const { return setters_.count(call.get()) != 0; }
  bool NeedsCallerBounds() const { return needs_caller_bounds_; }

 private:
  std::unordered_set<const Type*> types_;
  std::unordered_set<const Call*> setters_;
  bool needs_caller_bounds_ = false;
};
}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ACC_TRIM_ANALYSIS_H_
