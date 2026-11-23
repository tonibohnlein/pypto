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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_scalar_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_acc_trim_analysis.h"

namespace pypto::ir {

// A dynamic descriptor's -1 fields state availability, never equality of the
// actual native metadata. This input-owned, use-site checker retains expressions
// supplied by dominating allocation/setter commands and proves relations under
// the original guard. Mutable facts require ordinary alloc ownership.
// Rank-one transfers can also use immutable allocation or subview extents.
//
// One walk and indexed per-handle state; control boundaries advance an epoch
// rather than scanning every handle. Unknown joins/backedges lose evidence,
// never become an empty active extent. Arithmetic proof work is additional and
// uses the existing analyzer. No branch outcomes or dynamic visits are unfolded.
class BufferRuntimeValidAnalysis : public IRVisitor {
 public:
  struct Issue {
    CallPtr call;
    std::string message;
  };
  explicit BufferRuntimeValidAnalysis(
      const FunctionPtr& function, bool collect_guards = false, bool check_payload_capacity = false,
      const std::vector<int>& positive_loop_parameters = {},
      const std::map<size_t, std::pair<int64_t, int64_t>>& parameter_bounds = {})
      : analyzer_(std::make_shared<arith::Analyzer>()),
        collect_tail_bounds_(!parameter_bounds.empty()),
        collect_guards_(collect_guards),
        check_payload_capacity_(check_payload_capacity) {
    if (function) {
      core_ = function->func_type_;
      dual_aiv_ = core_ == FunctionType::AIV && function->GetAttr<bool>(kAttrDualAivDispatch);
      BindBufferAccTrimParameters(function, *analyzer_, ready_, positive_loop_parameters, parameter_bounds);
      if (function->body_) VisitStmt(function->body_);
    }
  }
  const std::optional<Issue>& issue() const { return issue_; }
  bool ProvesLogicalScalarPoint(const CallPtr& call) const {
    const auto found = logical_scalar_points_.find(call.get());
    return found != logical_scalar_points_.end() && found->second;
  }
  bool ProvesLogicalMinimumFillPad(const CallPtr& call) const {
    const auto found = logical_minimum_padding_.find(call.get());
    return found != logical_minimum_padding_.end() && found->second;
  }
  // Only definitions proved positive on every occurrence and actually used as
  // minimum-fill extents need a witness when the leaf is outlined.
  bool HasFillTailWitness(const AssignStmtPtr& statement) const {
    const auto found = checked_tails_.find(statement.get());
    return found != checked_tails_.end() && found->second && fill_extents_.count(statement->var_.get());
  }
  bool HasCheckedTails() const { return !checked_tails_.empty() && !fill_extents_.empty(); }
  bool HasElidableGuards() const {
    return std::any_of(redundant_guards_.begin(), redundant_guards_.end(),
                       [](const auto& entry) { return entry.second; });
  }
  bool Elides(const IfStmtPtr& branch) const {
    const auto found = redundant_guards_.find(branch.get());
    return found != redundant_guards_.end() && found->second;
  }
  std::unordered_set<const Call*> NonEmptyTransfers() const {
    std::unordered_set<const Call*> proved;
    for (const auto& [call, positive] : transfer_guards_)
      if (positive) proved.insert(call);
    return proved;
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto call = As<Call>(statement->value_);
    const auto type = As<BufferType>(statement->var_->GetType());
    if (call && type && IsOp(call, "buffer.alloc")) {
      owners_.insert(statement->var_.get());
      if (call->args_.size() == 2) addressed_.insert(statement->var_.get());
      const auto values = call->args_.empty() ? nullptr : As<MakeTuple>(call->args_[0]);
      if (values) {
        State state{epoch_, {}};
        size_t next = 0;
        for (size_t axis = 0; axis < 2 && type->valid_shape_.size() == 2; ++axis) {
          if (type->valid_shape_[axis] < 0) {
            if (next < values->elements_.size()) state.extents[axis] = values->elements_[next++];
          } else {
            state.extents[axis] = Constant(type->valid_shape_[axis], statement->span_);
          }
        }
        // PTOAS cannot update a descriptor with a fixed valid row, so its
        // dynamic column still names the allocation operand after a join.
        if (type->shape_.size() == 2 && type->shape_[0] == 1 && type->valid_shape_[0] == 1 &&
            type->valid_shape_[1] == -1 && state.extents[1]) {
          immutable_rank_one_extents_[statement->var_.get()] = state.extents;
        }
        states_[statement->var_.get()] = std::move(state);
      }
    } else if (call && type && IsOp(call, "buffer.peer_pop") && call->args_.size() == 2) {
      const auto valid = As<MakeTuple>(call->args_[1]);
      if (valid && valid->elements_.size() == 2) {
        states_[statement->var_.get()] = State{epoch_, {valid->elements_[0], valid->elements_[1]}};
      }
    } else if (call && type && type->shape_.size() == 2 && type->shape_[0] == 1 &&
               type->valid_shape_[1] == -1 && IsOp(call, "buffer.subview") && call->args_.size() == 3) {
      // A view's valid tuple belongs to the view itself, even if the parent
      // allocation's active extents are changed later.
      const auto valid = As<MakeTuple>(call->args_[2]);
      if (valid && valid->elements_.size() == 2 && valid->elements_[0] && valid->elements_[1]) {
        immutable_rank_one_extents_[statement->var_.get()] = {valid->elements_[0], valid->elements_[1]};
      }
    } else if (const auto scalar = As<ScalarType>(statement->var_->GetType());
               scalar && scalar->dtype_ == DataType::BOOL && !call) {
      // Boolean aliases preserve comparisons of actual SSA values. Do not bind
      // numerical arithmetic: cancellation could require an unproved machine-
      // overflow premise. Scalar-memory readiness remains a separate contract.
      guard_aliases_.emplace(statement->var_.get(), statement->value_);
      if (ready_.insert(statement->var_.get()).second) scalar_definitions_.push_back(statement->var_);
    }
    if (const auto scalar = As<ScalarType>(statement->var_->GetType());
        scalar && (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt())) {
      auto range = GuardedTileTail(statement->value_);
      if (collect_tail_bounds_ && As<Min>(statement->value_)) {
        const auto [entry, inserted] = checked_tails_.emplace(statement.get(), range.has_value());
        if (!inserted) entry->second &= range.has_value();
      }
      if (!range) range = Bounds(statement->value_);
      if (call && scalar->dtype_.IsSignedInt() && scalar->dtype_.GetBit() < 64 &&
          (IsOp(call, "tensor.read") || IsOp(call, "tile.read") || IsOp(call, "buffer.gm_read") ||
           IsOp(call, "buffer.read"))) {
        // Only the representation range is known for a scalar memory value.
        range = NarrowSignedRepresentation(statement->var_);
      }
      if (dual_aiv_ && call && call->args_.empty() &&
          (IsOp(call, "tile.get_subblock_idx") || IsOp(call, "buffer.get_subblock_idx"))) {
        // The dual-AIV dispatch ABI supplies the two hardware lane IDs.
        range = arith::ConstIntBound{0, 1};
      }
      if (!range) {
        range = ObservedMinimum(statement, scalar->dtype_);
      }
      if (range) {
        if (ready_.insert(statement->var_.get()).second) {
          scalar_definitions_.push_back(statement->var_);
          scalar_values_.emplace(statement->var_.get(), ScalarValue(statement->value_));
        }
        analyzer_->const_int_bound.Update(statement->var_, *range);
      }
    }
    if (call) {
      if (check_payload_capacity_ && type && IsOp(call, "buffer.alloc") && call->args_.size() >= 1) {
        CheckCapacity(call, type, As<MakeTuple>(call->args_[0]), /*dynamic_only=*/true);
      }
      CheckCall(call);
    }
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    if (const auto call = As<Call>(statement->expr_)) CheckCall(call);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    const auto checkpoint = scalar_definitions_.size();
    const auto range = BoundBufferLoopCoordinate(
        loop, *analyzer_, [this](const VarPtr& var) { return ready_.count(var.get()) != 0; }, bound_cache_);
    if (ready_.insert(loop->loop_var_.get()).second) {
      scalar_definitions_.push_back(loop->loop_var_);
      if (range) analyzer_->const_int_bound.Update(loop->loop_var_, *range);
    }
    ++epoch_;
    bound_cache_.clear();
    VisitStmt(loop->body_);
    RestoreScalars(checkpoint);
    ++epoch_;
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    const auto checkpoint = scalar_definitions_.size();
    bound_cache_.clear();
    ++epoch_;
    {
      BufferAccTrimGuardScope constraint(loop->condition_, true, analyzer_, guard_aliases_);
      VisitStmt(loop->body_);
      RestoreScalars(checkpoint);
    }
    ++epoch_;
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    const auto checkpoint = scalar_definitions_.size();
    if (collect_guards_ && !branch->else_body_ && branch->return_vars_.empty() &&
        branch->leading_comments_.empty()) {
      auto [entry, inserted] = redundant_guards_.emplace(branch.get(), true);
      if (entry->second) entry->second = ProvesValidityGuard(branch->condition_);
    }
    bound_cache_.clear();
    {
      BufferAccTrimGuardScope constraint(branch->condition_, true, analyzer_, guard_aliases_);
      const auto tail = GetTileTailGuard(branch->condition_);
      // A nested guard may prove the same ordinal and tile size using another
      // extent. Keep the nearest such witness only for its own lexical body.
      ExprPtr previous;
      if (tail) {
        const auto [entry, inserted] = tail_guards_.try_emplace(tail->first, tail->second);
        if (!inserted) {
          previous = std::exchange(entry->second, tail->second);
        }
      }
      VisitStmt(branch->then_body_);
      if (tail) {
        if (previous) {
          tail_guards_.at(tail->first) = std::move(previous);
        } else {
          tail_guards_.erase(tail->first);
        }
      }
      RestoreScalars(checkpoint);
    }
    // Neither arm's mutations prove the other's initial metadata. The initial
    // slice conservatively requires fresh setters in that arm when necessary.
    ++epoch_;
    if (branch->else_body_) {
      bound_cache_.clear();
      BufferAccTrimGuardScope constraint(branch->condition_, false, analyzer_, guard_aliases_);
      VisitStmt(*branch->else_body_);
      RestoreScalars(checkpoint);
    }
    ++epoch_;
  }

 private:
  struct State {
    uint64_t epoch;
    std::array<ExprPtr, 2> extents;
  };
  static ExprPtr Constant(int64_t value, const Span& span) {
    return std::make_shared<ConstInt>(value, DataType::INDEX, span);
  }
  static bool AtomicInteger(const ExprPtr& expression) {
    const auto type = expression ? As<ScalarType>(expression->GetType()) : nullptr;
    return type && (type->dtype_.IsSignedInt() || type->dtype_ == DataType::INDEX) &&
           (AsVarLike(expression) || As<ConstInt>(expression));
  }

  using TileTailKey = std::pair<VarPtr, int64_t>;
  ExprPtr ScalarValue(const ExprPtr& value) const {
    const auto var = AsVarLike(value);
    const auto found = scalar_values_.find(var.get());
    return found == scalar_values_.end() ? value : found->second;
  }

  VarPtr TailOrdinal(const ExprPtr& value) const {
    // Preserve the identity of a computed ordinal. Its machine value is
    // already fixed at this cut, and need not be a loop induction variable.
    const auto alias = AsVarLike(ScalarValue(value));
    return alias ? alias : AsVarLike(value);
  }

  std::optional<std::pair<TileTailKey, ExprPtr>> GetTileTailGuard(const ExprPtr& condition) {
    const auto less = As<Lt>(condition);
    if (!less) return std::nullopt;
    const auto ordinal = TailOrdinal(less->left_);
    const auto quotient = As<FloorDiv>(ScalarValue(less->right_));
    const auto cap = quotient ? As<ConstInt>(quotient->right_) : nullptr;
    const auto padded = quotient ? As<Add>(quotient->left_) : nullptr;
    const auto adjustment = padded ? As<ConstInt>(padded->right_) : nullptr;
    if (!ordinal || !cap || cap->value_ <= 0 || !adjustment || adjustment->value_ != cap->value_ - 1 ||
        !Bounds(quotient->left_)) {
      return std::nullopt;
    }
    if (!Bounds(less->left_)) return std::nullopt;
    return std::make_pair(TileTailKey{ordinal, cap->value_}, padded->left_);
  }

  std::optional<arith::ConstIntBound> GuardedTileTail(const ExprPtr& value) {
    const auto minimum = As<Min>(value);
    if (!minimum) return std::nullopt;
    const bool cap_on_right = As<ConstInt>(minimum->right_) != nullptr;
    const auto cap = As<ConstInt>(cap_on_right ? minimum->right_ : minimum->left_);
    const auto remainder = As<Sub>(ScalarValue(cap_on_right ? minimum->left_ : minimum->right_));
    const auto product = remainder ? As<Mul>(ScalarValue(remainder->right_)) : nullptr;
    const auto step = product ? As<ConstInt>(product->right_) : nullptr;
    const auto ordinal = product ? TailOrdinal(product->left_) : nullptr;
    if (!cap || !step || cap->value_ != step->value_ || !ordinal) return std::nullopt;
    const auto guard = tail_guards_.find(TileTailKey{ordinal, cap->value_});
    // Only compare the same typed extent, and first prove every subtraction
    // operand and result representable. Signed floor division gives
    // n+C-1 >= (i+1)*C whenever i < floor((n+C-1)/C), including negative i.
    // This bounds the width only. It supplies no tensor-index proof.
    if (guard == tail_guards_.end() || !Bounds(remainder) ||
        !SameTailExtent(guard->second, remainder->left_)) {
      return std::nullopt;
    }
    return arith::ConstIntBound{1, cap->value_};
  }

  static bool SameTailExtent(const ExprPtr& a, const ExprPtr& b) {
    if (a == b) return true;
    const auto ca = As<Cast>(a), cb = As<Cast>(b);
    return ca && cb && GetScalarDtype(ca) == GetScalarDtype(cb) && ca->operand_ == cb->operand_;
  }
  std::optional<arith::ConstIntBound> Bounds(const ExprPtr& value) {
    return BoundBufferAccTrimScalar(
        value, *analyzer_, [this](const VarPtr& var) { return ready_.count(var.get()) != 0; }, &bound_cache_);
  }
  static std::optional<arith::ConstIntBound> ObservedMinimum(const AssignStmtPtr& statement, DataType dtype) {
    // A clamp of an observed machine value proves both endpoints even when
    // the arithmetic producing the inner value has no replayable range.
    auto operand = statement->value_;
    ConstIntPtr lower;
    if (const auto maximum = As<Max>(operand)) {
      lower = As<ConstInt>(maximum->right_);
      if (!lower || lower->dtype() != dtype || GetScalarDtype(maximum->left_) != dtype) return std::nullopt;
      operand = maximum->left_;
    }
    const auto minimum = As<Min>(operand);
    if (!minimum || GetScalarDtype(minimum) != dtype || GetScalarDtype(minimum->left_) != dtype ||
        GetScalarDtype(minimum->right_) != dtype) {
      return std::nullopt;
    }
    auto cap = As<ConstInt>(minimum->right_);
    if (!cap) {
      cap = As<ConstInt>(minimum->left_);
    }
    const auto bits = dtype == DataType::INDEX ? 64 : dtype.GetBit();
    const auto maximum = (uint64_t{1} << (bits - 1)) - 1;
    if (!cap || cap->value_ < 0 || static_cast<uint64_t>(cap->value_) > maximum) {
      return std::nullopt;
    }
    // The original scalar assignment and its operands still execute. An
    // observed minimum cannot exceed its literal cap, even if the analysis
    // cannot justify replaying that assignment's numerical expression.
    const auto lo = bits == 64 ? arith::ConstIntBound::kNegInf : -static_cast<int64_t>(maximum) - 1;
    if (lower && (lower->value_ < lo || lower->value_ > cap->value_)) return std::nullopt;
    return arith::ConstIntBound{lower ? lower->value_ : lo, cap->value_};
  }
  bool Positive(const ExprPtr& extent) {
    if (AtomicInteger(extent) && analyzer_->const_int_bound(extent).min_value >= 1) return true;
    const auto bound = Bounds(extent);
    return bound && bound->min_value >= 1;
  }
  bool ProvesValidityGuard(const ExprPtr& predicate) {
    // Follow only the conjunction of positive-extent comparisons emitted by
    // Buffer lowering. The proof is about original SSA values at this cut.
    std::vector<ExprPtr> pending{predicate};
    std::unordered_set<const Expr*> seen;
    while (!pending.empty()) {
      const auto expr = std::move(pending.back());
      pending.pop_back();
      if (!seen.insert(expr.get()).second) continue;
      if (const auto both = As<And>(expr)) {
        pending.push_back(both->left_);
        pending.push_back(both->right_);
        continue;
      }
      const auto cmp = As<Gt>(expr);
      const auto zero = cmp ? As<ConstInt>(cmp->right_) : nullptr;
      if (!zero || zero->value_ != 0 || !Positive(cmp->left_)) return false;
    }
    return true;
  }
  bool Equal(const ExprPtr& a, const ExprPtr& b) {
    // The predicate query includes the analyzer's guarded comparison facts;
    // CanProveEqual's subtraction-interval query alone loses SSA equalities.
    return AtomicInteger(a) && AtomicInteger(b) && analyzer_->CanProve(MakeEq(a, b));
  }
  std::optional<std::array<ExprPtr, 2>> Extents(const ExprPtr& expression) const {
    const auto var = AsVarLike(expression);
    const auto type = var ? As<BufferType>(var->GetType()) : nullptr;
    if (!type || type->valid_shape_.size() != 2) return std::nullopt;
    std::array<ExprPtr, 2> result;
    const auto found = states_.find(var.get());
    for (size_t axis = 0; axis < 2; ++axis) {
      if (type->valid_shape_[axis] >= 0) {
        result[axis] = Constant(type->valid_shape_[axis], var->span_);
      } else {
        if (!owners_.count(var.get()) || found == states_.end() || found->second.epoch != epoch_ ||
            !found->second.extents[axis])
          return std::nullopt;
        result[axis] = found->second.extents[axis];
      }
    }
    return result;
  }
  std::optional<std::array<ExprPtr, 2>> RankOneTransferExtents(const ExprPtr& expression) const {
    const auto var = AsVarLike(expression);
    const auto immutable = immutable_rank_one_extents_.find(var.get());
    if (immutable != immutable_rank_one_extents_.end()) return immutable->second;
    return Extents(expression);
  }
  void Refuse(const CallPtr& call, const std::string& message) {
    if (!issue_) issue_ = Issue{call, message};
  }
  void CheckCapacity(const CallPtr& call, const BufferTypePtr& type, const MakeTuplePtr& extents,
                     bool dynamic_only = false) {
    if (!type || !extents || issue_) {
      return;
    }
    size_t operand = 0;
    for (size_t axis = 0; axis < type->shape_.size(); ++axis) {
      if (dynamic_only && type->valid_shape_[axis] >= 0) {
        continue;
      }
      if (operand >= extents->elements_.size()) {
        return;  // Arity belongs to the Buffer operation's schema.
      }
      const auto& extent = extents->elements_[operand++];
      if (const auto literal = As<ConstInt>(extent)) {
        if (literal->value_ < 0 || literal->value_ > type->shape_[axis]) {
          Refuse(call, "Buffer valid extent exceeds its physical capacity");
        }
        continue;
      }
      const auto bound = Bounds(extent);
      if (!bound || bound->min_value <= 0 || bound->max_value > type->shape_[axis]) {
        Refuse(call, "Buffer runtime valid extent has no positive in-capacity bound");
      }
    }
  }

  void CheckMatchingFullColumnRows(const CallPtr& call, const ExprPtr& source, const ExprPtr& destination,
                                   const BufferTypePtr& type) {
    const auto a = Extents(source);
    const auto b = Extents(destination);
    const auto columns = Constant(type->shape_[1], call->span_);
    if (!a || !b || !Equal((*a)[0], (*b)[0]) || !Equal((*a)[1], columns) || !Equal((*b)[1], columns)) {
      Refuse(call, call->op_->name_ + " requires matching runtime rows and fixed full columns");
      return;
    }
    const auto rows = Bounds((*a)[0]);
    if (!rows || rows->min_value < 1 || rows->max_value > type->shape_[0]) {
      Refuse(call, call->op_->name_ + " requires positive runtime rows within physical capacity");
    }
  }
  void CheckCall(const CallPtr& call) {
    const bool logical_point = IsOp(call, "tile.read") || IsOp(call, "tile.write");
    const bool physical_point = IsOp(call, "buffer.read") || IsOp(call, "buffer.write");
    if ((logical_point || physical_point) && call->args_.size() >= 2) {
      const auto type = logical_point ? backend::StaticLogicalBufferDescriptor(call->args_[0]->GetType(),
                                                                               /*allow_full_pad=*/true)
                                      : As<BufferType>(call->args_[0]->GetType());
      const auto& point = call->args_[1];
      bool proved = backend::GetBufferScalarPointLayout(type, point).has_value();
      if (proved) {
        const auto tuple = As<MakeTuple>(point);
        for (size_t axis = 0; axis < 2 && proved; ++axis) {
          const auto range = Bounds(tuple->elements_[axis]);
          proved = range && range->min_value >= 0 && range->max_value < type->valid_shape_[axis];
        }
      }
      if (logical_point) {
        auto [entry, inserted] = logical_scalar_points_.emplace(call.get(), proved);
        if (!inserted) entry->second &= proved;
      } else if (!proved) {
        Refuse(call, "Buffer scalar coordinates require a proved in-range point at this use");
      }
      return;
    }
    if (IsOp(call, "tile.fillpad") && call->args_.size() == 1 &&
        call->GetKwarg<PadValue>("pad_value", PadValue::zero) == PadValue::min) {
      auto [entry, inserted] = logical_minimum_padding_.emplace(call.get(), true);
      if (entry->second) {
        const auto type = As<TileType>(call->args_[0]->GetType());
        entry->second = type && type->shape_.size() == 2;
        if (entry->second) {
          const auto valid = tile_view_semantics::GetEffectiveTileView(*type).valid_shape;
          entry->second = valid.size() == 2;
          for (size_t axis = 0; axis < 2 && entry->second; ++axis) {
            const auto capacity = As<ConstInt>(type->shape_[axis]);
            if (collect_tail_bounds_) {
              if (const auto variable = AsVarLike(valid[axis])) fill_extents_.insert(variable.get());
            }
            const auto range = Bounds(valid[axis]);
            entry->second =
                capacity && range && range->min_value >= 1 && range->max_value <= capacity->value_;
          }
        }
      }
      return;
    }
    const bool load = IsOp(call, "buffer.load");
    if ((load || IsOp(call, "buffer.store")) && call->args_.size() == 4) {
      const auto extents = As<MakeTuple>(call->args_[2]);
      if (collect_guards_ && extents &&
          std::any_of(extents->elements_.begin(), extents->elements_.end(),
                      [](const ExprPtr& extent) { return !As<ConstInt>(extent); })) {
        auto [entry, inserted] = transfer_guards_.emplace(call.get(), true);
        if (entry->second) {
          entry->second = std::all_of(extents->elements_.begin(), extents->elements_.end(),
                                      [this](const ExprPtr& extent) { return Positive(extent); });
        }
      }
      const auto& local = call->args_[load ? 3 : 0];
      const auto type = As<BufferType>(local->GetType());
      if (type && type->valid_shape_.size() == 2 && type->valid_shape_[1] == -1 && extents &&
          extents->elements_.size() == 1 && backend::SupportsBufferVector1DTransfer(call->args_, load)) {
        const auto active = RankOneTransferExtents(local);
        if (!active || !Equal((*active)[0], Constant(1, call->span_)) ||
            !(structural_equal((*active)[1], extents->elements_[0]) ||
              Equal((*active)[1], extents->elements_[0]))) {
          Refuse(call,
                 "Runtime rank-one transfer requires restored valid shape [1, extent] "
                 "matching the GM partition");
        }
      }
    }
    // A dynamic Acc is a store-only metadata alias. No producer/carry/window
    // or unknown consumer inherits permission from its -1 descriptor fields.
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto type = As<BufferType>(call->args_[i]->GetType());
      if (!type || type->memory_space_ != MemorySpace::Acc ||
          std::none_of(type->valid_shape_.begin(), type->valid_shape_.end(),
                       [](int64_t extent) { return extent < 0; }))
        continue;
      if (i != 0 || (!IsOp(call, "buffer.store") && !IsOp(call, "buffer.set_validshape")) ||
          !backend::IsBufferAccRowTrimDescriptor(type)) {
        Refuse(call, "Runtime Acc aliases require the qualified row-only default store metadata contract");
        return;
      }
    }
    if (IsOp(call, "buffer.store") && call->args_.size() == 4) {
      const auto type = As<BufferType>(call->args_[0]->GetType());
      if (type && type->memory_space_ == MemorySpace::Acc &&
          std::any_of(type->valid_shape_.begin(), type->valid_shape_.end(),
                      [](int64_t extent) { return extent < 0; })) {
        if (!IsBufferAccTrimTargetQualified()) {
          Refuse(call, "Runtime Acc row stores require a qualified selected A2/A3 native target");
          return;
        }
        const auto owner = AsVarLike(call->args_[0]);
        const auto extents = Extents(call->args_[0]);
        const auto transfer = As<MakeTuple>(call->args_[2]);
        if (core_ != FunctionType::AIC || !owner || !addressed_.count(owner.get()) ||
            !backend::IsBufferAccRowTrimDescriptor(type) || !extents || !transfer ||
            transfer->elements_.size() != 2 || GetIntKwarg(call->kwargs_, "atomic", 0) != 0) {
          Refuse(call, "Runtime Acc row stores require owned addressed AIC metadata and default TSTORE");
        } else if (!Equal((*extents)[1], Constant(type->shape_[1], call->span_)) ||
                   !Equal(transfer->elements_[0], (*extents)[0]) ||
                   !Equal(transfer->elements_[1], (*extents)[1])) {
          Refuse(call,
                 "Runtime Acc store requires restored rows and fixed full columns matching the GM partition");
        } else {
          const auto range = Bounds((*extents)[0]);
          if (!range || range->min_value < 1 || range->max_value > type->shape_[0])
            Refuse(call,
                   "Runtime Acc store requires proved positive rows within physical capacity at this use");
        }
        return;
      }
    }

    if (IsOp(call, "buffer.set_validshape") && call->args_.size() == 2) {
      const auto handle = AsVarLike(call->args_[0]);
      immutable_rank_one_extents_.erase(handle.get());
      const auto values = As<MakeTuple>(call->args_[1]);
      if (check_payload_capacity_) {
        CheckCapacity(call, As<BufferType>(call->args_[0]->GetType()), values);
      }
      if (handle && owners_.count(handle.get()) && values && values->elements_.size() == 2) {
        states_[handle.get()] = {epoch_, {values->elements_[0], values->elements_[1]}};
      } else {
        // An unsupported native metadata alias cannot establish or preserve
        // ordinary-handle facts merely because its payload bytes overlap.
        ++epoch_;
      }
      return;
    }
    if (IsOp(call, "buffer.fillpad") && call->args_.size() == 2) {
      const auto type = As<BufferType>(call->args_[0]->GetType());
      const auto target = As<BufferType>(call->args_[1]->GetType());
      const auto extents = Extents(call->args_[0]);
      if (type && target && target->pad_ == PadValue::min && extents) {
        for (size_t axis = 0; axis < 2; ++axis) {
          const auto bounded = Bounds((*extents)[axis]);
          if (!bounded || bounded->min_value < 1 || bounded->max_value > type->shape_[axis]) {
            Refuse(call, "Minimum FillPad requires positive source extents within physical capacity");
          }
        }
      } else if (type && extents && backend::SupportsZeroFillPadColumnPrefix(type) &&
                 Equal((*extents)[0], Constant(1, call->span_))) {
        const auto cols = Bounds((*extents)[1]);
        if (!cols || cols->min_value < 1 || cols->max_value > type->shape_[1]) {
          Refuse(call, "Single-row FillPad requires positive source columns within physical capacity");
        }
      } else if (!type || !extents || !Equal((*extents)[1], Constant(type->shape_[1], call->span_)) ||
                 !AtomicInteger((*extents)[0]) || !analyzer_->CanProveGreaterEqual((*extents)[0], 1)) {
        Refuse(call, "FillPad requires a positive source row guard and fixed full columns");
      }
      return;
    }
    if ((IsOp(call, "buffer.copy") || IsOp(call, "buffer.exp")) && call->args_.size() == 2) {
      const auto type = As<BufferType>(call->args_[0]->GetType());
      const bool column_copy = IsOp(call, "buffer.copy") && backend::IsColumnVectorBufferGeometry(type);
      if ((backend::IsMinimumPaddedVecRowGeometry(type) || column_copy) &&
          (type->valid_shape_[0] < 0 || type->valid_shape_[1] < 0)) {
        CheckMatchingFullColumnRows(call, call->args_[0], call->args_[1], type);
      }
      return;
    }
    if (const auto* conversion = backend::FindBufferConversionRecipe(call->op_->name_);
        conversion && conversion->kind == backend::BufferConversionKind::Cast && call->args_.size() == 2) {
      bool dynamic = false;
      for (const auto& arg : call->args_) {
        if (const auto type = As<BufferType>(arg->GetType()))
          for (const auto extent : type->valid_shape_) dynamic |= extent < 0;
      }
      if (!dynamic) return;
      const auto source = Extents(call->args_[0]);
      const auto target = Extents(call->args_[1]);
      const auto type = As<BufferType>(call->args_[0]->GetType());
      const auto destination = As<BufferType>(call->args_[1]->GetType());
      if (backend::IsMinimumPaddedVecRowGeometry(type) ||
          backend::IsMinimumPaddedVecRowGeometry(destination)) {
        CheckMatchingFullColumnRows(call, call->args_[0], call->args_[1],
                                    backend::IsMinimumPaddedVecRowGeometry(type) ? type : destination);
        return;
      }
      if (!source || !target || !type) {
        Refuse(call,
               "Runtime cast valid metadata is unavailable at this use; restore owned allocation metadata");
      } else if (!Equal((*source)[0], (*target)[0]) ||
                 !Equal((*source)[1], Constant(type->shape_[1], call->span_)) ||
                 !Equal((*target)[1], Constant(type->shape_[1], call->span_))) {
        Refuse(call, "Runtime cast requires proved matching runtime rows and fixed full columns at this use");
      } else if (!AtomicInteger((*source)[0]) || !analyzer_->CanProveGreaterEqual((*source)[0], 1)) {
        Refuse(call, "Runtime cast requires a proved positive-row guard at this use");
      }
      return;
    }
    const auto* recipe = backend::FindBufferAxisRecipe(call->op_->name_);
    if (!recipe || call->args_.size() != 3 || (recipe->axis == 0 && recipe->reduction)) return;
    bool dynamic = false;
    for (const auto& arg : call->args_) {
      if (const auto type = As<BufferType>(arg->GetType())) {
        for (const auto extent : type->valid_shape_) dynamic |= extent < 0;
      }
    }
    // Existing static axis recipes retain their established active domains.
    if (!dynamic) return;
    if (recipe->axis == 0 && std::string_view(recipe->buffer_op) != "buffer.col_expand_mul") return;
    const auto source = Extents(call->args_[0]);
    const auto carrier = Extents(call->args_[1]);
    const auto target = Extents(call->args_[2]);
    if (!source || !target || (!recipe->reduction && !carrier)) {
      Refuse(call,
             "Row operation runtime valid metadata is unavailable at this use; restore owned "
             "allocation metadata after control-flow boundaries");
      return;
    }
    const auto type = As<BufferType>(call->args_[0]->GetType());
    if (!type || type->shape_.size() != 2) return;
    const auto columns = Constant(type->shape_[1], call->span_);
    const auto one = Constant(1, call->span_);
    const bool carrier_valid =
        recipe->reduction ||
        (recipe->axis == 0 ? Equal((*carrier)[0], one) && Equal((*carrier)[1], columns)
                           : Equal((*carrier)[0], (*source)[0]) && Equal((*carrier)[1], one));
    if (!Equal((*source)[0], (*target)[0]) || !Equal((*source)[1], columns) ||
        !Equal((*target)[1], recipe->reduction ? one : columns) || !carrier_valid) {
      Refuse(call,
             "Row operation requires proved matching runtime rows and fixed full columns at "
             "this use; unrelated dynamic descriptor fields are not equal metadata");
    } else if (!AtomicInteger((*source)[0]) || !analyzer_->CanProveGreaterEqual((*source)[0], 1)) {
      Refuse(call, "Row operation requires a proved positive-row guard at this use");
    } else if (backend::IsMinimumPaddedVecRowGeometry(type) &&
               (!Bounds((*source)[0]) || Bounds((*source)[0])->max_value > type->shape_[0])) {
      Refuse(call, "Minimum-padded row operation requires runtime rows within physical capacity");
    }
  }
  void RestoreScalars(size_t checkpoint) {
    bound_cache_.clear();
    while (scalar_definitions_.size() > checkpoint) {
      analyzer_->Unbind(scalar_definitions_.back());
      ready_.erase(scalar_definitions_.back().get());
      guard_aliases_.erase(scalar_definitions_.back().get());
      scalar_values_.erase(scalar_definitions_.back().get());
      scalar_definitions_.pop_back();
    }
  }
  FunctionType core_ = FunctionType::Opaque;
  bool dual_aiv_ = false;
  std::unordered_set<const Var*> ready_, addressed_;
  BufferAccTrimBoundCache bound_cache_;
  std::unordered_map<const Var*, ExprPtr> guard_aliases_;
  std::unordered_map<const Var*, ExprPtr> scalar_values_;
  std::map<TileTailKey, ExprPtr> tail_guards_;
  std::vector<VarPtr> scalar_definitions_;
  uint64_t epoch_ = 0;
  arith::AnalyzerPtr analyzer_;
  std::unordered_set<const Var*> owners_;
  std::unordered_map<const Var*, State> states_;
  std::unordered_map<const Var*, std::array<ExprPtr, 2>> immutable_rank_one_extents_;
  std::optional<Issue> issue_;
  std::unordered_map<const Call*, bool> logical_scalar_points_;
  std::unordered_map<const Call*, bool> logical_minimum_padding_;
  bool collect_tail_bounds_;
  std::unordered_map<const AssignStmt*, bool> checked_tails_;
  std::unordered_set<const Var*> fill_extents_;
  bool collect_guards_;
  bool check_payload_capacity_;
  // The same immutable statement/call can be reused. Every occurrence must
  // have the witness before a pointer-keyed rewrite or emission can use it.
  std::unordered_map<const IfStmt*, bool> redundant_guards_;
  std::unordered_map<const Call*, bool> transfer_guards_;
};

// Remove only validity guards whose exact predicates follow from original
// lexical facts. An already placed Buffer input can share the same rewrite.
FunctionPtr ElideDominatedValidityGuards(const FunctionPtr& function);

// A selected descriptor's native runtime obligations apply equally to access
// analysis, actual-token lowering, ownership and code generation.
inline void RequireBufferRuntimeValid(const FunctionPtr& function) {
  const BufferRuntimeValidAnalysis metadata(function);
  if (const auto& issue = metadata.issue()) CHECK_SPAN(false, issue->call->span_) << issue->message;
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_RUNTIME_VALID_ANALYSIS_H_
