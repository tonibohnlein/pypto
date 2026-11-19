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
#include "pypto/ir/transforms/utils/buffer_acc_trim_analysis.h"

#include <algorithm>
#include <any>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/type_metadata.h"

namespace pypto::ir {
namespace {
bool Acc(const TypePtr& type) {
  const auto tile = As<TileType>(type);
  return tile && tile->GetMemorySpace() == MemorySpace::Acc;
}
bool Integer(const ExprPtr& expr) {
  const auto type = expr ? As<ScalarType>(expr->GetType()) : nullptr;
  return type && (type->dtype_ == DataType::INDEX || type->dtype_.IsSignedInt());
}
bool Constant(const ExprPtr& expr, int64_t value) {
  const auto number = As<ConstInt>(expr);
  return number && number->value_ == value;
}

using Wide = __int128;

std::pair<Wide, Wide> SignedLimits(const ExprPtr& expr) {
  const auto dtype = GetScalarDtype(expr);
  const auto bits = dtype == DataType::INDEX ? 64 : dtype.GetBit();
  const Wide maximum = (Wide{1} << (bits - 1)) - 1;
  return {-maximum - 1, maximum};
}

std::pair<Wide, Wide> MachineBounds(const ExprPtr& expr, const arith::ConstIntBound& bound) {
  const auto [minimum, maximum] = SignedLimits(expr);
  return {bound.min_value == arith::ConstIntBound::kNegInf ? minimum : Wide{bound.min_value},
          bound.max_value == arith::ConstIntBound::kPosInf ? maximum : Wide{bound.max_value}};
}

std::optional<arith::ConstIntBound> BoundArithmetic(const ExprPtr& expr, const BinaryExprPtr& binary,
                                                    const arith::ConstIntBound& a,
                                                    const arith::ConstIntBound& b,
                                                    arith::Analyzer& analyzer) {
  const auto [lo_a, hi_a] = MachineBounds(binary->left_, a);
  const auto [lo_b, hi_b] = MachineBounds(binary->right_, b);
  Wide lo, hi;
  if (As<Sub>(expr)) {
    lo = lo_a - hi_b;
    hi = hi_a - lo_b;
    // Only a lexical comparison of the original SSA operands can refine
    // this difference. A subtraction-based solver would reason about the
    // very arithmetic whose machine range is not yet known.
    if (AsVarLike(binary->left_) && AsVarLike(binary->right_)) {
      const auto order = analyzer.transitive_cmp.TryCompare(binary->left_, binary->right_, false);
      if (order == arith::CompareResult::kGT) lo = std::max(lo, Wide{1});
      if (order == arith::CompareResult::kLT) hi = std::min(hi, Wide{-1});
    }
  } else if (As<Mul>(expr)) {
    lo = std::min({lo_a * lo_b, lo_a * hi_b, hi_a * lo_b, hi_a * hi_b});
    hi = std::max({lo_a * lo_b, lo_a * hi_b, hi_a * lo_b, hi_a * hi_b});
  } else {
    lo = lo_a + lo_b;
    hi = hi_a + hi_b;
  }
  const auto [minimum, maximum] = SignedLimits(expr);
  if (lo < minimum || hi > maximum || lo > hi || hi < arith::ConstIntBound::kNegInf) return std::nullopt;
  return arith::ConstIntBound{static_cast<int64_t>(std::max(lo, Wide{arith::ConstIntBound::kNegInf})),
                              static_cast<int64_t>(hi)};
}
bool StaticFull(const TileTypePtr& tile) {
  if (!tile || !Acc(tile) || tile->shape_.size() != 2) return false;
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  const auto valid = view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  if (valid.size() != 2) return false;
  for (size_t axis = 0; axis != 2; ++axis) {
    const auto extent = As<ConstInt>(tile->shape_[axis]);
    if (!extent || extent->value_ <= 0 || !Constant(valid[axis], extent->value_)) return false;
  }
  return true;
}
// Before InitMemRef the setter registry supplies the shared-storage identity.
// Once storage exists, require the same complete window. Final typed-root
// construction also checks allocation capacity and native address alignment.
bool SameStorage(const TileTypePtr& source, const TileTypePtr& result) {
  const auto a = source->memref_.value_or(nullptr), b = result->memref_.value_or(nullptr);
  if (!a && !b) return true;
  return a && b && a->base_ == b->base_ && a->size_ == b->size_ && a->slot_count_ == 1 &&
         b->slot_count_ == 1 && !a->slot_index_ && !b->slot_index_ &&
         structural_equal(a->byte_offset_, b->byte_offset_);
}

// Scalar facts are established at their original definition. Machine
// arithmetic must have a representable range even under dominating comparisons.
class CandidateIndex : public IRVisitor {
 public:
  explicit CandidateIndex(const FunctionPtr& function) : analyzer_(std::make_shared<arith::Analyzer>()) {
    BindBufferAccTrimParameters(function, *analyzer_, ready_);
    VisitStmt(function->body_);
  }
  std::unordered_set<const Type*> types;
  std::unordered_set<const Call*> setters;
  std::unordered_set<const Expr*> values;
  void VisitExpr(const ExprPtr&) override {}

 protected:
  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto call = As<Call>(statement->value_);
    const auto scalar = As<ScalarType>(statement->var_->GetType());
    if (scalar && scalar->dtype_ == DataType::BOOL && !call) {
      guard_aliases_.emplace(statement->var_.get(), statement->value_);
      if (ready_.insert(statement->var_.get()).second) bound_.push_back(statement->var_);
    }
    if (Integer(statement->var_)) {
      const auto range = Bounds(statement->value_);
      if (range) Define(statement->var_, *range);
    }
    if (!IsOp(call, "tile.set_validshape") || call->args_.size() != 3 || !call->kwargs_.empty()) return;
    const auto source = As<TileType>(call->args_[0]->GetType());
    const auto result = As<TileType>(statement->var_->GetType());
    if (!StaticFull(source) || !result || !SameStorage(source, result)) return;
    const auto a = tile_view_semantics::GetEffectiveTileView(*source);
    const auto b = tile_view_semantics::GetEffectiveTileView(*result);
    if (source->dtype_ != result->dtype_ || !structural_equal(source->shape_[0], result->shape_[0]) ||
        !structural_equal(source->shape_[1], result->shape_[1]) || !Acc(result) || a.blayout != b.blayout ||
        a.slayout != b.slayout || a.fractal != b.fractal || a.pad != PadValue::null ||
        b.pad != PadValue::null || a.compact != CompactMode::null || b.compact != CompactMode::null ||
        !a.stride.empty() || !b.stride.empty() || (a.start_offset && !Constant(a.start_offset, 0)) ||
        (b.start_offset && !Constant(b.start_offset, 0)))
      return;
    const auto rows = As<ConstInt>(source->shape_[0]), cols = As<ConstInt>(source->shape_[1]);
    if (!rows || !cols || !backend::SupportsBufferAccRowTrimCapacity(rows->value_, cols->value_) ||
        !Constant(call->args_[2], cols->value_))
      return;
    const auto range = Bounds(call->args_[1]);
    if (!range || range->min_value < 0 || range->max_value > rows->value_) return;
    const auto valid = b.valid_shape.empty() ? result->shape_ : b.valid_shape;
    if (valid.size() != 2 || !structural_equal(valid[0], call->args_[1]) || !Constant(valid[1], cols->value_))
      return;
    types.insert(result.get());
    types.insert(call->GetType().get());
    setters.insert(call.get());
    values.insert(statement->var_.get());
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    const auto checkpoint = bound_.size();
    {
      bound_cache_.emplace_back();
      BufferAccTrimGuardScope constraint(branch->condition_, true, analyzer_, guard_aliases_);
      VisitStmt(branch->then_body_);
      Restore(checkpoint);
    }
    if (branch->else_body_) {
      bound_cache_.emplace_back();
      BufferAccTrimGuardScope constraint(branch->condition_, false, analyzer_, guard_aliases_);
      VisitStmt(*branch->else_body_);
      Restore(checkpoint);
    }
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    const auto checkpoint = bound_.size();
    bound_cache_.emplace_back();
    Define(loop->loop_var_, {arith::ConstIntBound::kNegInf, arith::ConstIntBound::kPosInf});
    VisitStmt(loop->body_);
    Restore(checkpoint);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    const auto checkpoint = bound_.size();
    bound_cache_.emplace_back();
    VisitStmt(loop->body_);
    Restore(checkpoint);
  }

 private:
  arith::AnalyzerPtr analyzer_;
  std::unordered_map<const Var*, ExprPtr> guard_aliases_;
  std::vector<BufferAccTrimBoundCache> bound_cache_{1};
  std::unordered_set<const Var*> ready_;
  std::vector<VarPtr> bound_;
  void Define(const VarPtr& var, arith::ConstIntBound range) {
    if (!ready_.insert(var.get()).second) return;
    bound_.push_back(var);
    analyzer_->const_int_bound.Update(var, range);
  }
  void Restore(size_t checkpoint) {
    bound_cache_.pop_back();
    while (bound_.size() > checkpoint) {
      analyzer_->Unbind(bound_.back());
      ready_.erase(bound_.back().get());
      guard_aliases_.erase(bound_.back().get());
      bound_.pop_back();
    }
  }
  std::optional<arith::ConstIntBound> Bounds(const ExprPtr& expr) {
    return BoundBufferAccTrimScalar(
        expr, *analyzer_, [this](const VarPtr& variable) { return ready_.count(variable.get()) != 0; },
        &bound_cache_.back());
  }
};

// A conservative whole-Acc-component observer gate. This first family refuses
// all unknown Acc metadata observations rather than attempting to repair them.
// Distinct expression identities are visited once in each of data/observer
// contexts, so sharing a data use cannot hide an attribute/aggregate observer.
class Uses : public IRVisitor {
 public:
  explicit Uses(const CandidateIndex& candidates) : candidates_(candidates) {}
  bool supported = true;
  void VisitFunction(const FunctionPtr& function) override {
    Attributes(function->attrs_);
    for (const auto& parameter : function->params_) VisitExpr(parameter);
    for (const auto& type : function->return_types_) ObserveType(type);
    VisitStmt(function->body_);
  }
  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    auto& seen = observing_ ? observed_ : data_;
    if (!seen.insert(expr.get()).second) return;
    if (observing_ && Acc(expr->GetType())) supported = false;
    ObserveType(expr->GetType());
    IRVisitor::VisitExpr(expr);
  }
  void VisitStmt(const StmtPtr& statement) override {
    if (const auto loop = As<ForStmt>(statement)) Attributes(loop->attrs_);
    if (const auto scope = As<ScopeStmt>(statement)) Attributes(scope->attrs_);
    IRVisitor::VisitStmt(statement);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitExpr_(const CallPtr& call) override {
    const bool known = IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc") ||
                       IsOp(call, "tile.store") || IsOp(call, "tile.set_validshape") ||
                       IsOp(call, "tile.create") || IsOp(call, "tile.alloc");
    const bool previous = observing_;
    observing_ |= !known;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto& arg = call->args_[i];
      const auto tile = As<TileType>(arg->GetType());
      if (tile && Acc(tile) && !StaticFull(tile)) {
        // Runtime aliases are accepted only directly as default TSTORE source.
        if (!IsOp(call, "tile.store") || i != 0 || !candidates_.values.count(arg.get()) ||
            call->args_.size() != 3 || GetIntKwarg(call->kwargs_, "atomic", 0) != 0 ||
            GetIntKwarg(call->kwargs_, "st_phase", 0) != 0 ||
            GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") ||
            GetKwargOr<bool>(call->kwargs_, "pre_relu", false))
          supported = false;
      }
      VisitExpr(arg);
    }
    observing_ = true;
    Attributes(call->attrs_);
    for (const auto& [key, value] : call->kwargs_)
      ForEachAttrExpr(value, [this](const ExprPtr& expression) { VisitExpr(expression); });
    observing_ = previous;
  }
  void VisitExpr_(const MakeTuplePtr& tuple) override {
    const bool previous = observing_;
    observing_ = true;
    IRVisitor::VisitExpr_(tuple);
    observing_ = previous;
  }
  void VisitStmt_(const AssignStmtPtr& statement) override {
    if (Acc(statement->var_->GetType()) && !StaticFull(As<TileType>(statement->var_->GetType())) &&
        !candidates_.values.count(statement->var_.get()))
      supported = false;
    VisitExpr(statement->value_);
    ObserveType(statement->var_->GetType());
  }
  void VisitStmt_(const ReturnStmtPtr& statement) override {
    const bool previous = observing_;
    observing_ = true;
    IRVisitor::VisitStmt_(statement);
    observing_ = previous;
  }
  void VisitStmt_(const YieldStmtPtr& statement) override {
    // Static Acc loop carries remain admitted; runtime aliases may not escape.
    for (const auto& value : statement->value_) {
      if (Acc(value->GetType()) && !StaticFull(As<TileType>(value->GetType()))) supported = false;
      VisitExpr(value);
    }
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& arg : loop->iter_args_) {
      if (Acc(arg->GetType()) && !StaticFull(As<TileType>(arg->GetType()))) supported = false;
      VisitExpr(arg->initValue_);
    }
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& arg : loop->iter_args_) {
      if (Acc(arg->GetType()) && !StaticFull(As<TileType>(arg->GetType()))) supported = false;
      VisitExpr(arg->initValue_);
    }
    IRVisitor::VisitStmt_(loop);
  }

 private:
  const CandidateIndex& candidates_;
  bool observing_ = false;
  std::unordered_set<const Expr*> data_, observed_;
  std::unordered_set<const Type*> types_;
  void Attributes(const std::vector<std::pair<std::string, std::any>>& attrs) {
    const bool previous = observing_;
    observing_ = true;
    for (const auto& [key, value] : attrs)
      ForEachAttrExpr(value, [this](const ExprPtr& expression) { VisitExpr(expression); });
    observing_ = previous;
  }
  void ObserveType(const TypePtr& type) {
    if (!type || !types_.insert(type.get()).second) return;
    const bool previous = observing_;
    observing_ = true;
    type_metadata::ForEachTypeMetadataChild(
        type, [this](const TypePtr& child) { ObserveType(child); },
        [this](const ExprPtr& expr) { VisitExpr(expr); }, [this](const ExprPtr& expr) { VisitExpr(expr); });
    observing_ = previous;
  }
};
}  // namespace

BufferAccTrimGuardScope::BufferAccTrimGuardScope(const ExprPtr& condition, bool truth,
                                                 const arith::AnalyzerPtr& analyzer,
                                                 const std::unordered_map<const Var*, ExprPtr>& aliases) {
  std::vector<std::pair<ExprPtr, bool>> pending{{condition, truth}};
  std::unordered_set<const Expr*> positive, negative;
  while (!pending.empty()) {
    const auto [expr, value] = pending.back();
    pending.pop_back();
    if (!expr || !(value ? positive : negative).insert(expr.get()).second) continue;
    if (const auto variable = AsVarLike(expr)) {
      if (const auto found = aliases.find(variable.get()); found != aliases.end())
        pending.emplace_back(found->second, value);
      continue;
    }
    if (const auto negation = As<Not>(expr)) {
      pending.emplace_back(negation->operand_, !value);
      continue;
    }
    const auto binary = As<BinaryExpr>(expr);
    if (!binary) continue;
    if ((value && As<And>(expr)) || (!value && As<Or>(expr))) {
      pending.emplace_back(binary->left_, value);
      pending.emplace_back(binary->right_, value);
      continue;
    }
    const auto atomic = [](const ExprPtr& arg) {
      return Integer(arg) && (AsVarLike(arg) || As<ConstInt>(arg));
    };
    if ((As<Eq>(expr) || As<Ne>(expr) || As<Lt>(expr) || As<Le>(expr) || As<Gt>(expr) || As<Ge>(expr)) &&
        atomic(binary->left_) && atomic(binary->right_))
      scopes_.push_back(analyzer->GetConstraintContext(value ? expr : MakeNot(expr, expr->span_)));
  }
}
BufferAccTrimGuardScope::~BufferAccTrimGuardScope() {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) it->ExitScope();
}

std::optional<arith::ConstIntBound> BoundBufferAccTrimScalar(const ExprPtr& expression,
                                                             arith::Analyzer& analyzer,
                                                             const std::function<bool(const VarPtr&)>& ready,
                                                             BufferAccTrimBoundCache* cache) {
  BufferAccTrimBoundCache local;
  auto& memo = cache ? *cache : local;
  auto visit = [&](auto&& self, const ExprPtr& expr) -> std::optional<arith::ConstIntBound> {
    if (!expr || !Integer(expr)) return std::nullopt;
    if (const auto found = memo.find(expr.get()); found != memo.end()) {
      return found->second;
    }
    std::optional<arith::ConstIntBound> result;
    if (const auto constant = As<ConstInt>(expr)) {
      result = arith::ConstIntBound{constant->value_, constant->value_};
    } else if (const auto variable = AsVarLike(expr); variable && ready(variable)) {
      result = analyzer.const_int_bound(variable);
    } else if (const auto call = As<Call>(expr);
               call && IsOp(call, "buffer.get_block_idx") && call->args_.empty()) {
      result = arith::ConstIntBound{0, arith::ConstIntBound::kPosInf};
    } else if (const auto binary = As<BinaryExpr>(expr);
               binary &&
               (As<Min>(expr) || As<Max>(expr) || As<Add>(expr) || As<Sub>(expr) || As<Mul>(expr))) {
      const auto a = self(self, binary->left_), b = self(self, binary->right_);
      if (a && b) {
        if (As<Min>(expr)) {
          result = arith::ConstIntBound{std::min(a->min_value, b->min_value),
                                        std::min(a->max_value, b->max_value)};
        } else if (As<Max>(expr)) {
          result = arith::ConstIntBound{std::max(a->min_value, b->min_value),
                                        std::max(a->max_value, b->max_value)};
        } else {
          result = BoundArithmetic(expr, binary, *a, *b, analyzer);
        }
      }
    }
    // A narrow machine scalar must not inherit mathematical bounds through
    // an implicit narrowing. INDEX keeps its existing shape/arithmetic ABI.
    const auto scalar = As<ScalarType>(expr->GetType());
    if (result && scalar && scalar->dtype_ != DataType::INDEX && scalar->dtype_.GetBit() < 64) {
      const auto maximum = (int64_t{1} << (scalar->dtype_.GetBit() - 1)) - 1;
      if (result->min_value < -maximum - 1 || result->max_value > maximum) result = std::nullopt;
    }
    memo.emplace(expr.get(), result);
    return result;
  };
  return visit(visit, expression);
}

std::optional<arith::ConstIntBound> BoundBufferLoopCoordinate(const ForStmtPtr& loop,
                                                              arith::Analyzer& analyzer,
                                                              const std::function<bool(const VarPtr&)>& ready,
                                                              BufferAccTrimBoundCache& cache) {
  if (loop->kind_ != ForKind::Sequential || !Integer(loop->loop_var_)) return std::nullopt;
  const auto start = BoundBufferAccTrimScalar(loop->start_, analyzer, ready, &cache);
  const auto stop = BoundBufferAccTrimScalar(loop->stop_, analyzer, ready, &cache);
  const auto step = BoundBufferAccTrimScalar(loop->step_, analyzer, ready, &cache);
  if (!start || !stop || !step || step->min_value <= 0) return std::nullopt;
  const Wide last = static_cast<Wide>(stop->max_value) - 1;
  const auto [minimum, maximum] = SignedLimits(loop->loop_var_);
  // Every visited IV lies in [start, stop). Also ensure the exit update
  // could not wrap in the original IV type and start a second sequence.
  if (start->min_value <= arith::ConstIntBound::kNegInf || start->min_value < minimum ||
      start->min_value > last || last + step->max_value > maximum)
    return std::nullopt;
  return arith::ConstIntBound{start->min_value, static_cast<int64_t>(last)};
}

void BindBufferAccTrimParameters(const FunctionPtr& function, arith::Analyzer& analyzer,
                                 std::unordered_set<const Var*>& ready) {
  std::unordered_set<const Var*> shape_variables;
  for (const auto& parameter : function->params_) {
    if (Integer(parameter)) ready.insert(parameter.get());
    const auto tensor = As<TensorType>(parameter->GetType());
    if (!tensor) continue;
    for (const auto& dimension : tensor->shape_) {
      const auto variable = AsVarLike(dimension);
      // This is the existing input tensor-shape ABI (nonnegative extent), not
      // a bound inferred from allocation capacity or a benchmark configuration.
      if (variable && Integer(variable) && shape_variables.insert(variable.get()).second) {
        ready.insert(variable.get());
        analyzer.const_int_bound.Update(variable, {0, arith::ConstIntBound::kPosInf});
      }
    }
  }
}

bool IsBufferAccTrimTargetQualified() {
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* context = PassContext::Current();
  const auto* handler =
      context ? context->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  return handler && handler->GetPtoTargetArch() == "a2a3";
}

BufferAccTrimAnalysis::BufferAccTrimAnalysis(const FunctionPtr& function, bool addressed) {
  if (!function || !addressed || function->func_type_ != FunctionType::AIC ||
      !IsBufferAccTrimTargetQualified())
    return;
  class HasSetter : public IRVisitor {
   public:
    bool found = false;
    void VisitExpr(const ExprPtr&) override {}
    void VisitStmt_(const AssignStmtPtr& statement) override {
      const auto call = As<Call>(statement->value_);
      found |= IsOp(call, "tile.set_validshape") && Acc(statement->var_->GetType());
    }
  } candidates;
  candidates.VisitStmt(function->body_);
  if (!candidates.found) return;
  CandidateIndex index(function);
  if (index.setters.empty()) return;
  Uses uses(index);
  uses.VisitFunction(function);
  if (!uses.supported) return;
  types_ = std::move(index.types);
  setters_ = std::move(index.setters);
}
}  // namespace pypto::ir
