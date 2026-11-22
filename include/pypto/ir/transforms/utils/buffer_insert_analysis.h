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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_INSERT_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_INSERT_ANALYSIS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// The pinned native implementation checks template valid extents during
// debug builds. Specialize only a small exhaustive set of scalar SSA values
// so each native command sees static valid extents. Unknown values, including
// loop carries, deliberately have no recipe.
class BufferInsertAnalysis : public IRVisitor {
 public:
  explicit BufferInsertAnalysis(const StmtPtr& body, const BufferViewAnalysis* views = nullptr)
      : body_(body), views_(views) {}

  using ExtentPair = std::array<int64_t, 2>;

  std::vector<ExtentPair> Cases(const CallPtr& call) {
    if (!IsOp(call, "tile.assemble") || call->args_.size() != 3 || !call->kwargs_.empty()) return {};
    const auto& target = call->args_[0];
    const auto& source = call->args_[1];
    const auto offsets = As<MakeTuple>(call->args_[2]);
    const auto type = As<TileType>(source->GetType());
    const auto physical = backend::PhysicalLogicalVecDescriptor(source->GetType());
    const auto destination = backend::StaticLogicalBufferDescriptor(target->GetType());
    if (!offsets || offsets->elements_.size() != 2 || !type || type->shape_.size() != 2 ||
        !DisjointRoots(source, target) ||
        !backend::SupportsA2A3VecInsert(physical, destination, offsets->elements_[0],
                                        offsets->elements_[1])) {
      return {};
    }
    const auto valid = tile_view_semantics::GetEffectiveTileView(*type).valid_shape;
    if (valid.size() != 2 || (As<ConstInt>(valid[0]) && As<ConstInt>(valid[1]))) return {};
    if (!indexed_) {
      // Ordinary static assembles and functions with no such operation do not
      // need a scalar proof or a second window-provenance walk.
      if (!views_) {
        owned_views_.emplace(body_);
        views_ = &*owned_views_;
      }
      VisitStmt(body_);
      indexed_ = true;
    }
    if (views_->IsWindow(source) || views_->IsWindow(target)) return {};
    const auto rows = Values(valid[0]);
    const auto columns = Values(valid[1]);
    if (rows.empty() || columns.empty() || rows.size() * columns.size() > kMaxCases || *rows.begin() < 0 ||
        *columns.begin() < 0 || *rows.rbegin() > physical->shape_[0] ||
        *columns.rbegin() > physical->shape_[1]) {
      return {};
    }
    std::vector<ExtentPair> cases;
    for (const auto r : rows) {
      for (const auto c : columns) {
        // A one-element TINSERT selects the distinct scalar/event overload.
        if (r == 1 && c == 1) return {};
        cases.push_back({r, c});
      }
    }
    std::stable_partition(cases.begin(), cases.end(),
                          [](const ExtentPair& extents) { return extents[0] != 0 && extents[1] != 0; });
    return cases;
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (As<ScalarType>(assign->var_->GetType())) {
      RecordValues(assign->var_, Values(assign->value_));
    }
  }

  void VisitStmt_(const IfStmtPtr& branch) override {
    IRVisitor::VisitStmt_(branch);
    const auto lhs = transform_utils::GetLastYieldStmt(branch->then_body_);
    const auto rhs = branch->else_body_ ? transform_utils::GetLastYieldStmt(*branch->else_body_) : nullptr;
    if (!lhs || !rhs || lhs->value_.size() != branch->return_vars_.size() ||
        rhs->value_.size() != branch->return_vars_.size()) {
      return;
    }
    for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
      const auto a = Values(lhs->value_[i]);
      const auto b = Values(rhs->value_[i]);
      if (a.empty() || b.empty()) continue;
      std::set<int64_t> joined;
      std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::inserter(joined, joined.begin()));
      RecordValues(branch->return_vars_[i], std::move(joined));
    }
  }

 private:
  static constexpr size_t kMaxCases = 4;

  static bool DisjointRoots(const ExprPtr& source, const ExprPtr& target) {
    const auto a = GetTypeMemRef(source->GetType()).value_or(nullptr);
    const auto b = GetTypeMemRef(target->GetType()).value_or(nullptr);
    return a && b && a->base_ && b->base_ && a->base_ != b->base_ && a->slot_count_ == 1 &&
           b->slot_count_ == 1 && !a->slot_index_ && !b->slot_index_;
  }

  std::set<int64_t> Values(const ExprPtr& value) const {
    if (const auto constant = As<ConstInt>(value)) {
      if (arith::detail::ConstIntFitsDtype(constant->value_, constant->dtype())) return {constant->value_};
      return {};
    }
    const auto found = values_.find(AsVarLike(value).get());
    return found == values_.end() ? std::set<int64_t>{} : found->second;
  }

  void RecordValues(const VarPtr& variable, std::set<int64_t> values) {
    const auto scalar = As<ScalarType>(variable->GetType());
    if (!scalar || values.empty() || values.size() > kMaxCases) return;
    for (const auto value : values) {
      if (!arith::detail::ConstIntFitsDtype(value, scalar->dtype_)) return;
    }
    values_.emplace(variable.get(), std::move(values));
  }

  StmtPtr body_;
  const BufferViewAnalysis* views_;
  std::optional<BufferViewAnalysis> owned_views_;
  bool indexed_ = false;
  std::unordered_map<const Var*, std::set<int64_t>> values_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_INSERT_ANALYSIS_H_
