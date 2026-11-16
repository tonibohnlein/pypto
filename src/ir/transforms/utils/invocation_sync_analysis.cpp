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
  // The index owns statement definitions and loop identities, not scalar
  // evaluation. Expression DAGs are inspected by their memoized consumers.
  void VisitExpr(const ExprPtr&) override {}
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
  std::shared_ptr<const SyncInvocationCoordinate> coordinate_;

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
    if (coordinate_ && (expression == coordinate_->Expression() || expression == coordinate_->Definition())) {
      out.push_back({blocks_.first, blocks_.second, 1, 0});
    } else if (auto value = As<ConstInt>(expression)) {
      out.push_back({blocks_.first, blocks_.second, 0, value->value_});
    } else if (auto found = source_.definitions.find(expression.get()); found != source_.definitions.end()) {
      out = Evaluate(found->second);
    } else if (auto call = As<Call>(expression);
               call && IsOp(call, "buffer.get_block_idx") && call->args_.empty()) {
      CHECK(!coordinate_) << "Invocation activation uses an unrelated original block coordinate";
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
                      std::pair<int64_t, int64_t> blocks,
                      std::shared_ptr<const SyncInvocationCoordinate> coordinate)
      : owner_(owner), parameters_(parameters), blocks_(blocks), coordinate_(std::move(coordinate)) {
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
        // A launch piece narrower than one stride crosses at most one
        // quotient threshold. Construct that threshold analytically, without
        // enumerating workers or iterations.
        const auto count = [&](Wide start) -> Wide {
          return start >= stop->value_ ? 0 : 1 + (Wide(stop->value_) - start - 1) / step->value_;
        };
        if (count(lo) != count(hi)) {
          CHECK(piece.coefficient == 1 && hi - lo < step->value_)
              << "Varying activation needs a unit-stride start range narrower than one loop stride";
          const auto threshold = Wide(stop->value_) - (count(lo) - 1) * step->value_;
          const auto cut = threshold - piece.constant;
          CHECK(cut > piece.lo && cut <= piece.hi) << "Invalid invocation quotient threshold";
          cuts.insert(static_cast<int64_t>(cut));
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
// Follow only original single-assignment aliases. One immutable source index
// supports path compression across all loop starts and candidate operands.
// Arithmetic is not simplified, and this cache proves no lexical dominance:
// the admitted candidate's original expressions are checked separately below.
class InvocationAliases {
  const SourceIndex& source_;
  std::map<const Expr*, ExprPtr> memo_;

 public:
  explicit InvocationAliases(const SourceIndex& source) : source_(source) {}
  ExprPtr Resolve(ExprPtr expression) {
    std::vector<ExprPtr> path;
    std::set<const Expr*> active;
    while (const auto variable = As<Var>(expression)) {
      if (const auto cached = memo_.find(variable.get()); cached != memo_.end()) {
        expression = cached->second;
        break;
      }
      CHECK(active.insert(variable.get()).second) << "Cyclic invocation coordinate is unavailable";
      path.push_back(expression);
      const auto definition = source_.definitions.find(variable.get());
      if (definition == source_.definitions.end()) break;
      expression = definition->second;
    }
    for (const auto& alias : path) memo_.emplace(alias.get(), expression);
    return expression;
  }
};
bool IsIndexCoordinate(const ExprPtr& expression) {
  const auto type = As<ScalarType>(expression->GetType());
  return type && type->dtype_ == DataType::INDEX;
}
struct RotationCandidate {
  ExprPtr expression, definition;
  size_t parameter;
  int64_t modulus;
  std::set<const Expr*> supporting;
};
std::optional<RotationCandidate> FindRotation(const FunctionPtr& owner, const SourceIndex& source,
                                              const SyncParameterBounds& parameters) {
  const auto sequence = As<SeqStmts>(owner->body_);
  if (!sequence) return std::nullopt;
  InvocationAliases aliases(source);
  std::set<const Expr*> starts;
  for (const auto& loop : source.loops) starts.insert(aliases.Resolve(loop->start_).get());
  std::map<const Expr*, size_t> parameter_indices;
  for (size_t i = 0; i < owner->params_.size(); ++i) parameter_indices.emplace(owner->params_[i].get(), i);
  SourceIndex dominating;
  for (const auto& statement : sequence->stmts_) {
    const auto assignment = As<AssignStmt>(statement);
    if (!assignment) break;  // The named coordinate must dominate the payload tree.
    dominating.VisitStmt(statement);
    const auto mod = As<FloorMod>(assignment->value_);
    if (!mod || !starts.count(mod.get())) continue;
    const auto add = As<Add>(aliases.Resolve(mod->left_));
    const auto divisor = As<ConstInt>(mod->right_);
    if (!add || !divisor || divisor->value_ <= 0) continue;
    auto left = aliases.Resolve(add->left_);
    auto right = aliases.Resolve(add->right_);
    auto block = As<Call>(left);
    if (!block || !IsOp(block, "buffer.get_block_idx")) {
      std::swap(left, right);
      block = As<Call>(left);
    }
    if (!block || !IsOp(block, "buffer.get_block_idx") || !block->args_.empty()) continue;
    const auto parameter = parameter_indices.find(right.get());
    if (parameter == parameter_indices.end()) continue;
    const size_t index = parameter->second;
    auto bound = parameters.find(index);
    if (bound == parameters.end() || bound->second.first == bound->second.second) continue;
    CHECK(IsIndexCoordinate(assignment->var_) && IsIndexCoordinate(mod) && IsIndexCoordinate(mod->left_) &&
          IsIndexCoordinate(mod->right_) && IsIndexCoordinate(add) && IsIndexCoordinate(add->left_) &&
          IsIndexCoordinate(add->right_) && IsIndexCoordinate(left) && IsIndexCoordinate(right))
        << "Effective invocation coordinate requires matching original index types";
    CHECK(bound->second.first >= 0 && bound->second.second >= bound->second.first &&
          Wide(bound->second.second) + divisor->value_ - 1 <= *SyncCoordinateMaximum(add))
        << "Effective invocation coordinate may overflow its original addition";
    std::set<const Expr*> supporting, seen;
    std::vector<ExprPtr> pending{mod};
    while (!pending.empty()) {
      auto expression = pending.back();
      pending.pop_back();
      if (!seen.insert(expression.get()).second) continue;
      CHECK(IsIndexCoordinate(expression))
          << "Effective coordinate aliases must preserve original index types";
      if (const auto definition = dominating.definitions.find(expression.get());
          definition != dominating.definitions.end()) {
        supporting.insert(expression.get());
        pending.push_back(definition->second);
      } else if (AsVarLike(expression)) {
        // Global alias discovery must not turn a future or nested definition
        // into an available invocation parameter. Only the original prefix
        // above dominates this candidate and its payload tree.
        CHECK(parameter_indices.count(expression.get()))
            << "Effective invocation coordinate requires dominating original definitions";
      } else if (const auto binary = As<BinaryExpr>(expression)) {
        pending.push_back(binary->left_);
        pending.push_back(binary->right_);
      }
    }
    return RotationCandidate{assignment->var_, mod, index, divisor->value_, std::move(supporting)};
  }
  return std::nullopt;
}
}  // namespace
std::shared_ptr<const SyncInvocationCoordinate> SyncInvocationCoordinate::Qualify(
    const FunctionPtr& owner, const SyncParameterBounds& parameters, std::pair<int64_t, int64_t> blocks,
    std::optional<int64_t> launch_count) {
  CHECK(owner) << "Missing effective invocation owner";
  SourceIndex source;
  source.VisitStmt(owner->body_);
  const auto candidate = FindRotation(owner, source, parameters);
  if (!candidate) return nullptr;
  CHECK(launch_count && *launch_count == candidate->modulus && blocks.first == 0 &&
        blocks.second == candidate->modulus - 1)
      << "Effective invocation coordinate requires an independent full launch cardinality";
  return std::shared_ptr<const SyncInvocationCoordinate>(
      new SyncInvocationCoordinate(owner, candidate->expression, candidate->definition, parameters,
                                   candidate->parameter, candidate->modulus, candidate->supporting));
}
std::optional<int64_t> SyncInvocationCoordinate::Map(int64_t original_block,
                                                     const std::map<size_t, int64_t>& parameters) const {
  if (original_block < 0 || original_block >= count_) return std::nullopt;
  for (const auto& [index, bound] : parameters_) {
    const auto value = parameters.find(index);
    if (value == parameters.end() || value->second < bound.first || value->second > bound.second)
      return std::nullopt;
  }
  return static_cast<int64_t>((Wide(original_block) + parameters.at(rotation_)) % count_);
}
SyncInvocationDomains PartitionSyncInvocation(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                              std::pair<int64_t, int64_t> blocks,
                                              std::optional<int64_t> launch_count) {
  CHECK(owner && blocks.first >= 0 && blocks.second >= blocks.first &&
        blocks.second < std::numeric_limits<int64_t>::max())
      << "Invalid invocation launch contract";
  auto coordinate = SyncInvocationCoordinate::Qualify(owner, parameters, blocks, launch_count);
  ActivationPartition partition(owner, parameters, blocks, coordinate);
  auto intervals = partition.Build();
  return {std::move(intervals), partition.Nodes(), partition.PieceCount(), std::move(coordinate)};
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
                                             std::optional<int64_t> launch_count) {
  CHECK(function && blocks.first >= 0 && blocks.second >= blocks.first &&
        blocks.second < std::numeric_limits<int64_t>::max())
      << "Invalid invocation launch contract";
  const auto domains = PartitionSyncInvocation(function, parameters, blocks, launch_count);
  SyncInvocationAnalysis result{function, parameters, blocks, {}};
  result.coordinate = domains.coordinate;
  result.coordinate_nodes = domains.coordinate_nodes;
  result.coordinate_pieces = domains.coordinate_pieces;
  for (const auto& domain : domains.intervals) {
    // Publish the whole owner only when EVERY nonempty case is qualified.
    // No successful subcase is a substitute for the original caller contract.
    auto input = ImportPlacedSync(function, disjoint_gm, target, parameters, domain, domains.coordinate);
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
  if (coordinate) {
    const auto mapped = coordinate->Map(block, parameters);
    if (!mapped) return nullptr;
    block = *mapped;
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
