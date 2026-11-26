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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SLOT_INDEX_QUALIFICATION_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SLOT_INDEX_QUALIFICATION_H_

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// PTOAS recognizes remui(iv + nonnegative_constant, slot_count). Restrict the
// unsigned spelling to slot metadata whose loop bounds prove that equivalence;
// ordinary FloorMod expressions retain their existing lowering.
class CanonicalSlotCollector : public ir::IRVisitor {
 public:
  std::set<const ir::Expr*> remainders;
  // Only the destination of this qualified slot use is normalized. The original
  // IV may also be a payload coordinate or select another slot-count family.
  // These owned expressions keep synthetic remainder identities alive.
  std::map<const ir::Var*, ir::ExprPtr> normalized_indices;
  std::set<const ir::Var*> unsupported_bases;
  // Original first failing assignment, retained only for admission diagnostics.
  ir::StmtPtr first_unsupported;
  std::set<const ir::Var*> verified_bases;

  void VisitStmt_(const ir::ForStmtPtr& op) override {
    loops_.emplace(op->loop_var_.get(), op);
    const bool verified = op->HasAttr(ir::kSoftwarePipelineSlotsAttr);
    if (verified) ++verified_depth_;
    ir::IRVisitor::VisitStmt_(op);
    if (verified) --verified_depth_;
    loops_.erase(op->loop_var_.get());
  }

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (ir::As<ir::ScalarType>(op->var_->GetType())) {
      definitions_.emplace(op->var_.get(), op->value_);
      analyzer_.const_int_bound.Update(op->var_, analyzer_.const_int_bound(op->value_));
    }
    auto tile = ir::GetTileTypeWithMemRef(op->var_->GetType());
    if (tile) {
      const auto memref = ir::GetDefinedMemRef(tile);
      if (memref->slot_index_.has_value()) {
        if (verified_depth_) verified_bases.insert(memref->base_.get());
        if (!QualifyIndex(*memref->slot_index_, memref->slot_count_, op->var_)) {
          unsupported_bases.insert(memref->base_.get());
          if (!first_unsupported) {
            first_unsupported = op;
          }
        }
      }
    }
    if (auto call = ir::As<ir::Call>(op->value_); ir::IsOp(call, "buffer.slot") && call->args_.size() == 2) {
      const auto group = ir::AsVarLike(call->args_[0]);
      const auto multi = ir::As<ir::MultiBufferType>(call->args_[0]->GetType());
      if (group && multi) {
        if (!QualifyIndex(call->args_[1], static_cast<uint64_t>(multi->slot_count_), op->var_)) {
          unsupported_bases.insert(group.get());
          if (!first_unsupported) {
            first_unsupported = op;
          }
        }
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

 private:
  static bool IsIndex(const ir::ExprPtr& expr) {
    const auto scalar = expr ? ir::As<ir::ScalarType>(expr->GetType()) : nullptr;
    return scalar && scalar->dtype_ == DataType::INDEX;
  }

  ir::ExprPtr ResolveScalarAlias(ir::ExprPtr expr) const {
    // A fixed bound keeps canonical-form recognition linear in program size.
    // Shared affine ordinals are emitted once and referenced by several slots.
    for (int i = 0; i < 8; ++i) {
      if (!IsIndex(expr)) return nullptr;
      auto var = ir::AsVarLike(expr);
      if (!var) break;
      auto it = definitions_.find(var.get());
      if (it == definitions_.end()) break;
      expr = it->second;
    }
    return IsIndex(expr) ? expr : nullptr;
  }

  bool QualifyIndex(const ir::ExprPtr& original, uint64_t count, const ir::VarPtr& result) {
    auto index = ir::AsVarLike(original) ? ResolveScalarAlias(original) : original;
    if (IsCanonical(index, count)) {
      if (ir::As<ir::FloorMod>(index)) remainders.insert(index.get());
      return true;
    }
    const auto induction = ir::AsVarLike(index);
    const auto found = loops_.find(induction.get());
    if (!IsIndex(index) || found == loops_.end() || count == 0 ||
        count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    const auto& loop = found->second;
    if (loop->kind_ != ir::ForKind::Sequential || !IsIndex(loop->start_) || !IsIndex(loop->stop_) ||
        !IsIndex(loop->step_)) {
      return false;
    }
    const auto start = ir::As<ir::ConstInt>(loop->start_);
    if (!ir::As<ir::ConstInt>(loop->stop_) && !loop->HasAttr(ir::kSoftwarePipelineSlotsAttr)) return false;
    const auto stop_max = analyzer_.const_int_bound(loop->stop_).max_value;
    const auto step = ir::As<ir::ConstInt>(loop->step_);
    if (!start || !step || start->value_ < 0 || start->value_ > stop_max ||
        static_cast<uint64_t>(stop_max) > count || step->value_ <= 0) {
      return false;
    }
    // Every executed IV is <= stop-1. Qualify the original post-final increment
    // too; the new remainder must not conceal an overflowing source loop.
    if (start->value_ < stop_max && step->value_ > std::numeric_limits<int64_t>::max() - (stop_max - 1)) {
      return false;
    }
    auto divisor =
        std::make_shared<ir::ConstInt>(static_cast<int64_t>(count), DataType::INDEX, original->span_);
    auto normalized = std::make_shared<ir::FloorMod>(induction, divisor, DataType::INDEX, original->span_);
    const auto entry = normalized_indices.emplace(result.get(), std::move(normalized)).first;
    remainders.insert(entry->second.get());
    return true;
  }

  bool IsCanonical(const ir::ExprPtr& index, uint64_t count) {
    if (auto constant = ir::As<ir::ConstInt>(index)) {
      return constant->value_ >= 0 && static_cast<uint64_t>(constant->value_) < count;
    }
    auto mod = ir::As<ir::FloorMod>(index);
    // An INDEX result can still hide narrowing in an operand or alias. Only
    // an entirely INDEX affine chain supports the unsigned slot proof.
    if (!mod || !IsIndex(mod) || !IsIndex(mod->right_)) return false;
    auto divisor = ir::As<ir::ConstInt>(mod->right_);
    if (!divisor || divisor->value_ <= 0 || static_cast<uint64_t>(divisor->value_) != count) return false;
    ir::ExprPtr induction = ResolveScalarAlias(mod->left_);
    if (!induction) return false;
    int64_t offset = 0;
    if (auto add = ir::As<ir::Add>(induction)) {
      auto constant = ir::As<ir::ConstInt>(add->right_);
      if (!constant || !IsIndex(constant) || constant->value_ < 0) return false;
      induction = ResolveScalarAlias(add->left_);
      offset = constant->value_;
    }
    auto found = loops_.find(ir::AsVarLike(induction).get());
    if (found == loops_.end()) return false;
    const auto& loop = found->second;
    if (!IsIndex(loop->start_) || !IsIndex(loop->stop_) || !IsIndex(loop->step_)) return false;
    if (!ir::As<ir::ConstInt>(loop->stop_) && !loop->HasAttr(ir::kSoftwarePipelineSlotsAttr)) return false;
    auto start = ir::As<ir::ConstInt>(loop->start_);
    const auto stop_bound = analyzer_.const_int_bound(loop->stop_);
    auto step = ir::As<ir::ConstInt>(loop->step_);
    return start && step && start->value_ >= 0 && step->value_ > 0 &&
           (offset == 0 || stop_bound.max_value <= std::numeric_limits<int64_t>::max() - offset);
  }

  std::map<const ir::Var*, ir::ForStmtPtr> loops_;
  std::map<const ir::Var*, ir::ExprPtr> definitions_;
  ir::arith::Analyzer analyzer_;
  unsigned verified_depth_ = 0;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SLOT_INDEX_QUALIFICATION_H_
