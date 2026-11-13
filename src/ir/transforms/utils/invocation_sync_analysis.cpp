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

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"

namespace pypto::ir::sync {
namespace {
using Wide = __int128;
Wide AddWide(Wide a, Wide b) {
  Wide result;
  CHECK(!__builtin_add_overflow(a, b, &result)) << "Invocation affine proof exceeds supported precision";
  return result;
}
Wide MultiplyWide(Wide a, Wide b) {
  Wide result;
  CHECK(!__builtin_mul_overflow(a, b, &result)) << "Invocation affine proof exceeds supported precision";
  return result;
}
struct Piece {
  int64_t lo, hi;
  Wide coefficient, constant;
  Wide At(int64_t block) const { return AddWide(MultiplyWide(coefficient, block), constant); }
};
using Pieces = std::vector<Piece>;
class SourceIndex : public IRVisitor {
 public:
  std::map<const Expr*, ExprPtr> definitions;
  std::vector<ForStmtPtr> loops;
  void VisitStmt_(const AssignStmtPtr& assignment) override {
    if (As<ScalarType>(assignment->var_->GetType()))
      CHECK(definitions.emplace(assignment->var_.get(), assignment->value_).second)
          << "Invocation coordinate requires original single-assignment identity";
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    loops.push_back(loop);
    VisitStmt(loop->body_);
  }
};
// Block-independent starts keep the existing full-domain range qualification.
// Merely supplying a launch bound must not require unrelated scalar parameters
// to be constants. Dependency discovery is indexed over the original SSA DAG;
// it does not prove arithmetic, activity or readiness (the importer does that).
class BlockDependency : public IRVisitor {
  const SourceIndex& source_;
  std::map<const Expr*, bool> memo_;
  std::set<const Expr*> active_;
  bool found_ = false;

 public:
  explicit BlockDependency(const SourceIndex& source) : source_(source) {}
  void VisitExpr(const ExprPtr& expression) override {
    if (!expression) return;
    if (auto cached = memo_.find(expression.get()); cached != memo_.end()) {
      found_ |= cached->second;
      return;
    }
    CHECK(active_.insert(expression.get()).second) << "Cyclic invocation coordinate is unavailable";
    const bool previous = found_;
    found_ = false;
    if (auto definition = source_.definitions.find(expression.get()); definition != source_.definitions.end())
      VisitExpr(definition->second);
    else if (auto call = As<Call>(expression); call && IsOp(call, "buffer.get_block_idx"))
      found_ = true;
    else
      IRVisitor::VisitExpr(expression);
    memo_.emplace(expression.get(), found_);
    active_.erase(expression.get());
    found_ |= previous;
  }
  bool Uses(const ExprPtr& expression) {
    found_ = false;
    VisitExpr(expression);
    return found_;
  }
  size_t Nodes() const { return memo_.size(); }
};
// Analytic piecewise-affine evaluation over one proved launch interval. Every
// intermediate is checked in its ORIGINAL type. It never samples workers or
// assumes an input band parameter has its benchmark value.
class ActivationPartition {
  const FunctionPtr& owner_;
  const SyncParameterBounds& parameters_;
  std::pair<int64_t, int64_t> blocks_;
  SourceIndex source_;
  BlockDependency block_dependency_{source_};
  std::map<const Expr*, size_t> parameter_indices_;
  std::map<const Expr*, Pieces> memo_;
  std::set<const Expr*> active_;
  size_t pieces_ = 0;

  void Check(const ExprPtr& expression, const Pieces& pieces) const {
    const auto maximum = SyncCoordinateMaximum(expression);
    CHECK(maximum) << "Invocation activation requires integer coordinates";
    for (const auto& piece : pieces)
      CHECK(piece.coefficient >= 0 && piece.coefficient <= std::numeric_limits<int64_t>::max() &&
            piece.At(piece.lo) >= 0 && piece.At(piece.hi) <= *maximum)
          << "Invocation activation may overflow or become negative in its original type";
  }
  Pieces Combine(const Pieces& left, const Pieces& right, bool multiply) const {
    Pieces out;
    size_t a = 0, b = 0;
    while (a < left.size() && b < right.size()) {
      const auto& x = left[a];
      const auto& y = right[b];
      const auto lo = std::max(x.lo, y.lo), hi = std::min(x.hi, y.hi);
      if (lo <= hi) {
        if (multiply) {
          CHECK(!x.coefficient || !y.coefficient)
              << "Nonaffine invocation activation requires another certificate";
          out.push_back(
              {lo, hi,
               AddWide(MultiplyWide(x.coefficient, y.constant), MultiplyWide(y.coefficient, x.constant)),
               MultiplyWide(x.constant, y.constant)});
        } else {
          out.push_back({lo, hi, AddWide(x.coefficient, y.coefficient), AddWide(x.constant, y.constant)});
        }
      }
      if (x.hi <= y.hi) ++a;
      if (y.hi <= x.hi) ++b;
    }
    return out;
  }
  const Pieces& Evaluate(const ExprPtr& expression) {
    if (auto found = memo_.find(expression.get()); found != memo_.end()) {
      return found->second;
    }
    CHECK(active_.insert(expression.get()).second) << "Cyclic invocation coordinate is unavailable";
    Pieces out;
    if (auto value = As<ConstInt>(expression)) {
      out.push_back({blocks_.first, blocks_.second, 0, value->value_});
    } else if (auto found = source_.definitions.find(expression.get()); found != source_.definitions.end()) {
      out = Evaluate(found->second);
    } else if (auto call = As<Call>(expression);
               call && IsOp(call, "buffer.get_block_idx") && call->args_.empty()) {
      out.push_back({blocks_.first, blocks_.second, 1, 0});
    } else if (auto cast = As<Cast>(expression)) {
      out = Evaluate(cast->operand_);
    } else if (auto add = As<Add>(expression)) {
      out = Combine(Evaluate(add->left_), Evaluate(add->right_), false);
    } else if (auto mul = As<Mul>(expression)) {
      out = Combine(Evaluate(mul->left_), Evaluate(mul->right_), true);
    } else if (auto mod = As<FloorMod>(expression)) {
      const auto modulus = As<ConstInt>(mod->right_);
      CHECK(modulus && modulus->value_ > 0) << "Activation modulo requires a positive fixed modulus";
      Evaluate(mod->right_);
      const Wide divisor = modulus->value_;
      for (const auto& piece : Evaluate(mod->left_)) {
        // This is a supplied single-orbit launch domain, not arbitrary period
        // discovery or expansion proportional to launch/trip counts.
        CHECK(piece.coefficient <= 1 && Wide(piece.hi) - piece.lo < divisor)
            << "Activation modulo needs a unit-stride single-orbit input certificate";
        const auto first = piece.At(piece.lo) / divisor, last = piece.At(piece.hi) / divisor;
        if (first == last) {
          out.push_back({piece.lo, piece.hi, piece.coefficient, piece.constant - first * divisor});
        } else {
          INTERNAL_CHECK(last == first + 1 && piece.coefficient == 1)
              << "Internal error: single-orbit activation has multiple wraps";
          const auto split = static_cast<int64_t>(last * divisor - piece.constant);
          out.push_back({piece.lo, split - 1, 1, piece.constant - first * divisor});
          out.push_back({split, piece.hi, 1, piece.constant - last * divisor});
        }
      }
    } else {
      auto parameter = parameter_indices_.find(expression.get());
      CHECK(parameter != parameter_indices_.end())
          << "Invocation activation is not a ready invariant expression";
      const auto index = parameter->second;
      auto bound = parameters_.find(index);
      CHECK(bound != parameters_.end() && bound->second.first == bound->second.second)
          << "Parameterized activation offset is not a proved fixed value";
      out.push_back({blocks_.first, blocks_.second, 0, bound->second.first});
    }
    Check(expression, out);
    active_.erase(expression.get());
    pieces_ += out.size();
    return memo_.emplace(expression.get(), std::move(out)).first->second;
  }

 public:
  ActivationPartition(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                      std::pair<int64_t, int64_t> blocks)
      : owner_(owner), parameters_(parameters), blocks_(blocks) {
    for (size_t i = 0; i < owner->params_.size(); ++i) parameter_indices_.emplace(owner->params_[i].get(), i);
    source_.VisitStmt(owner->body_);
  }
  std::vector<std::pair<int64_t, int64_t>> Build() {
    std::set<int64_t> cuts{blocks_.first, blocks_.second + 1};
    for (const auto& loop : source_.loops) {
      if (As<ConstInt>(loop->start_) || !block_dependency_.Uses(loop->start_)) continue;
      const auto stop = As<ConstInt>(loop->stop_), step = As<ConstInt>(loop->step_);
      CHECK(stop && step && step->value_ > 0 && stop->value_ >= 0)
          << "Activation partition requires a fixed stop and positive step";
      Evaluate(loop->stop_);
      Evaluate(loop->step_);
      for (const auto& piece : Evaluate(loop->start_)) {
        cuts.insert(piece.lo);
        cuts.insert(piece.hi + 1);
        const auto lo = piece.At(piece.lo), hi = piece.At(piece.hi);
        CHECK(lo >= stop->value_ || Wide(stop->value_) - lo <= step->value_)
            << "Invocation activation has more than one visit per case";
        if (lo < stop->value_ && hi >= stop->value_) {
          CHECK(piece.coefficient == 1) << "Nonunit activation threshold needs an exact quotient certificate";
          cuts.insert(static_cast<int64_t>(Wide(stop->value_) - piece.constant));
        }
      }
    }
    // The union of all analytic cuts is exactly the intersection partition of
    // all consumed normal forms. Adjacent intervals cover the caller domain.
    std::vector<int64_t> ordered(cuts.begin(), cuts.end());
    std::vector<std::pair<int64_t, int64_t>> out;
    for (size_t i = 1; i < ordered.size(); ++i) out.emplace_back(ordered[i - 1], ordered[i] - 1);
    return out;
  }
  // Counts work in both memoized stages, including nodes visited by each.
  size_t Nodes() const { return memo_.size() + block_dependency_.Nodes(); }
  size_t PieceCount() const { return pieces_; }
};
}  // namespace
SyncInvocationDomains PartitionSyncInvocation(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                              std::pair<int64_t, int64_t> blocks) {
  CHECK(owner && blocks.first >= 0 && blocks.second >= blocks.first &&
        blocks.second < std::numeric_limits<int64_t>::max())
      << "Invalid invocation launch contract";
  ActivationPartition partition(owner, parameters, blocks);
  auto intervals = partition.Build();
  return {std::move(intervals), partition.Nodes(), partition.PieceCount()};
}
ExprPtr DominatingSyncBlock(const FunctionPtr& owner) {
  const auto sequence = As<SeqStmts>(owner->body_);
  if (!sequence) return nullptr;
  for (const auto& statement : sequence->stmts_) {
    const auto assign = As<AssignStmt>(statement);
    if (!assign) break;
    const auto call = As<Call>(assign->value_);
    if (call && IsOp(call, "buffer.get_block_idx") && call->args_.empty()) {
      const auto type = As<ScalarType>(assign->var_->GetType());
      CHECK(type && type->dtype_ == DataType::INDEX) << "Invocation guard needs the original index binding";
      return assign->var_;
    }
  }
  return nullptr;
}
ExprPtr SyncInvocationGuard(const ExprPtr& block, std::pair<int64_t, int64_t> all,
                            std::pair<int64_t, int64_t> selected, const Span& span) {
  CHECK(all.first >= 0 && selected.first >= all.first && selected.second >= selected.first &&
        selected.second <= all.second && all.second < std::numeric_limits<int64_t>::max())
      << "Invalid invocation guard domain";
  if (selected == all) return nullptr;
  CHECK(block) << "Original dominating invocation block binding is unavailable";
  ExprPtr guard;
  if (selected.first > all.first)
    guard = std::make_shared<Ge>(block, std::make_shared<ConstInt>(selected.first, DataType::INDEX, span),
                                 DataType::BOOL, span);
  if (selected.second < all.second) {
    ExprPtr upper = std::make_shared<Lt>(
        block, std::make_shared<ConstInt>(selected.second + 1, DataType::INDEX, span), DataType::BOOL, span);
    guard = guard ? std::make_shared<And>(guard, upper, DataType::BOOL, span) : upper;
  }
  return guard;
}
SyncInvocationAnalysis AnalyzeSyncInvocation(const FunctionPtr& function, bool disjoint_gm,
                                             const std::string& target, const SyncParameterBounds& parameters,
                                             std::pair<int64_t, int64_t> blocks,
                                             SyncOccurrenceLayout layout) {
  CHECK(function && blocks.first >= 0 && blocks.second >= blocks.first &&
        blocks.second < std::numeric_limits<int64_t>::max())
      << "Invalid invocation launch contract";
  const auto domains = PartitionSyncInvocation(function, parameters, blocks);
  SyncInvocationAnalysis result{function, parameters, blocks, {}};
  result.coordinate_nodes = domains.coordinate_nodes;
  result.coordinate_pieces = domains.coordinate_pieces;
  for (const auto& domain : domains.intervals) {
    // Publish the whole owner only when EVERY nonempty case is qualified.
    // No successful subcase is a substitute for the original caller contract.
    auto input = ImportPlacedSync(function, disjoint_gm, target, parameters, domain, layout);
    auto mathematics = std::make_shared<CompactSyncResult>(AnalyzeCompactSync(std::move(input)));
    result.cases.push_back({domain.first, domain.second, std::move(mathematics)});
  }
  return result;
}
std::shared_ptr<const CompactSyncResult> SyncInvocationAnalysis::SelectCase(
    int64_t block, const std::map<size_t, int64_t>& parameters) const {
  if (block < block_bounds.first || block > block_bounds.second) return nullptr;
  for (const auto& [parameter, bound] : parameter_bounds) {
    auto value = parameters.find(parameter);
    if (value == parameters.end() || value->second < bound.first || value->second > bound.second)
      return nullptr;
  }
  auto found = std::upper_bound(cases.begin(), cases.end(), block, [](int64_t value, const auto& candidate) {
    return value < candidate.block_first;
  });
  if (found == cases.begin()) return nullptr;
  --found;
  if (block > found->block_last) return nullptr;
  return found->mathematics;
}
}  // namespace pypto::ir::sync
