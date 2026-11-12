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

// Interpret original-frame token equations, never storage demands. Each token
// equation is resolved once, iteratively. Pattern transport costs O(T D) for T
// equations and frame depth D, plus indexed lookups; no trip count is unfolded.
// U use families cost O(UD log(D+2)); payload scope checks visit their D frames.
// Guard DAG visits are counted per supplied condition, not globally: reused
// conditions can be revisited, and nested guard scopes copy their domain maps.
#include "src/ir/transforms/utils/scoped_token_families.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto::ir::sync {
namespace {
using Site = std::pair<size_t, size_t>;
using Interval = std::pair<int64_t, int64_t>;  // Half-open normalized visits.
using Domain = std::map<const ForStmt*, Interval>;
struct Digit {
  const ForStmt* variable = nullptr;
  int64_t offset = 0;
};
struct Pattern {
  std::optional<Site> source;
  std::vector<Digit> digits;
  // Exactly the domain where this fact is non-completed, not merely where a
  // proof succeeded. Completed entry and pure own-carry captures establish
  // this invariant, allowing waits on completed facts to vanish on other visits.
  Domain valid;
};
enum class EquationKind { Alias, Carry, Exit, Capture };
struct Equation {
  EquationKind kind;
  ExprPtr value;
  const ForStmt* frame = nullptr;
  const Var* carry = nullptr;
  int64_t selected = 0;
};
struct Wait {
  ExprPtr token;
  PipeType pipe;
  Domain domain;
};
struct Use {
  Wait wait;
  Site target;
};
struct Cursor {
  std::optional<Site> last;
  std::vector<Wait> pending;
  YieldStmtPtr yield;
  bool returned = false;
};
PipeType Pipe(const CallPtr& call, const char* key) {
  for (const auto& [name, value] : call->kwargs_)
    if (name == key) return static_cast<PipeType>(AnyCast<int>(value, name));
  CHECK(false) << "Missing logical synchronization pipe";
  return PipeType::ALL;
}
ExprPtr PureYield(StmtPtr body) {
  while (auto sequence = As<SeqStmts>(body)) {
    if (sequence->stmts_.size() != 1) return nullptr;
    body = sequence->stmts_.front();
  }
  const auto yield = As<YieldStmt>(body);
  return yield && yield->value_.size() == 1 ? yield->value_.front() : nullptr;
}
CallPtr Payload(const StmtPtr& statement) {
  if (auto eval = As<EvalStmt>(statement)) return As<Call>(eval->expr_);
  if (auto branch = As<IfStmt>(statement)) {
    auto first = branch->then_body_;
    if (auto sequence = As<SeqStmts>(first)) {
      if (sequence->stmts_.size() != 1) return nullptr;
      first = sequence->stmts_.front();
    }
    if (auto eval = As<EvalStmt>(first)) return As<Call>(eval->expr_);
  }
  return nullptr;
}
class Parser {
 public:
  Parser(const FunctionPtr& function, const CompactSyncInput& input,
         const std::map<const ForStmt*, ForStmtPtr>& stripped)
      : function_(function), input_(input) {
    CHECK(input.region_tree) << "Scoped token interpretation requires an original region tree";
    for (size_t r = 0; r < input.segments.size(); ++r) {
      const auto& segment = input.segments[r];
      for (const auto& frame : segment.nested_frames) Register(frame);
      for (size_t s = 0; s < segment.sites.size(); ++s)
        CHECK(sites_.emplace(segment.sites[s].original.get(), Site{r, s}).second)
            << "Scoped token payload anchors must be distinct";
    }
    Register(*input.region_tree);
    for (const auto& [original, erased] : stripped) {
      CHECK(frames_.count(erased.get())) << "Missing scoped token original frame";
      loops_.emplace(original, erased.get());
    }
  }
  std::vector<PhysicalSyncFamily> Run() {
    Cursor cursor;
    Read(function_->body_, cursor);
    CHECK(cursor.pending.empty()) << "Scoped wait has no direct consumer cut";
    for (const auto& initial : initials_)
      CHECK(!Resolve(initial).source) << "Scoped token carries require completed entry";
    std::vector<PhysicalSyncFamily> output;
    for (const auto& use : uses_) {
      auto pattern = Resolve(use.wait.token);
      if (!pattern.source) continue;
      auto family = Family(pattern, use);
      if (family) output.push_back(std::move(*family));
    }
    return output;
  }

 private:
  FunctionPtr function_;
  const CompactSyncInput& input_;
  std::map<const Call*, Site> sites_;
  std::map<const ForStmt*, SyncLoopFrame> frames_;
  std::map<const Var*, const ForStmt*> variables_;
  std::map<const ForStmt*, const ForStmt*> loops_;
  std::vector<const ForStmt*> scope_;
  std::set<const ForStmt*> active_frames_;
  std::map<const Var*, Pattern> resolved_;
  std::map<const Var*, Equation> equations_;
  std::vector<ExprPtr> initials_;
  std::vector<Use> uses_;

  void Register(const SyncLoopFrame& frame) {
    CHECK(!frame.start_expression || As<ConstInt>(frame.start_expression))
        << "Scoped token lowering requires constant original frame bounds";
    CHECK(frame.count >= 0 && frame.start >= 0 && frame.step > 0) << "Invalid scoped token frame";
    frames_.emplace(frame.loop.get(), frame);
    variables_.emplace(frame.loop->loop_var_.get(), frame.loop.get());
  }
  void Register(const SyncRegion& region) {
    if (region.repetition) Register(*region.repetition);
    for (const auto& child : region.children) Register(child);
  }
  void CheckScope(Site site) const {
    const auto& frames = input_.segments[site.first].nested_frames;
    CHECK(frames.size() == scope_.size()) << "Scoped token payload has a different enclosing frame";
    for (size_t i = 0; i < frames.size(); ++i)
      CHECK(frames[i].loop.get() == scope_[i]) << "Scoped token original frame identity mismatch";
  }
  Pattern Current(Site site) const {
    Pattern pattern;
    pattern.source = site;
    for (const auto& frame : input_.segments[site.first].nested_frames)
      pattern.digits.push_back({frame.loop.get(), 0});
    return pattern;
  }
  Interval Full(const ForStmt* frame) const { return {0, frames_.at(frame).count}; }
  void Intersect(Domain& domain, const ForStmt* frame, Interval interval) const {
    auto [at, inserted] = domain.try_emplace(frame, Full(frame));
    at->second.first = std::max(at->second.first, interval.first);
    at->second.second = std::min(at->second.second, interval.second);
  }
  // Guards denote exact rectangular domains, not an enclosure of arbitrary
  // Boolean control. Every variable must be an active original frame.
  std::pair<const ForStmt*, int64_t> Coordinate(const ExprPtr& condition) const {
    const auto binary = As<BinaryExpr>(condition);
    const auto var = binary ? AsVarLike(binary->left_) : nullptr;
    const auto constant = binary ? As<ConstInt>(binary->right_) : nullptr;
    const auto found = var ? variables_.find(var.get()) : variables_.end();
    CHECK(binary && constant && found != variables_.end() && active_frames_.count(found->second))
        << "Scoped token guard requires an active original coordinate";
    const auto& frame = frames_.at(found->second);
    const __int128 delta = __int128(constant->value_) - frame.start;
    CHECK(delta >= 0 && delta % frame.step == 0 && delta / frame.step < std::numeric_limits<int64_t>::max())
        << "Scoped token guard has an unavailable normalized coordinate";
    return {found->second, static_cast<int64_t>(delta / frame.step)};
  }
  void Guard(const ExprPtr& condition, Domain& domain) const {
    std::vector<ExprPtr> pending{condition};
    std::set<const Expr*> seen;
    while (!pending.empty()) {
      auto expression = pending.back();
      pending.pop_back();
      if (!seen.insert(expression.get()).second) continue;
      if (auto both = As<And>(expression)) {
        pending.push_back(both->left_);
        pending.push_back(both->right_);
        continue;
      }
      const auto [frame, digit] = Coordinate(expression);
      if (As<Eq>(expression))
        Intersect(domain, frame, {digit, digit + 1});
      else if (As<Gt>(expression))
        Intersect(domain, frame, {digit + 1, frames_.at(frame).count});
      else if (As<Lt>(expression))
        Intersect(domain, frame, {0, digit});
      else
        CHECK(false) << "Scoped token guard is not an exact supported interval";
    }
  }
  void Waits(const StmtPtr& statement, const Domain& domain, Cursor& cursor) {
    if (auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Waits(child, domain, cursor);
      return;
    }
    if (auto branch = As<IfStmt>(statement)) {
      CHECK(branch->return_vars_.empty() && !branch->else_body_) << "Unsupported scoped wait branch";
      auto selected = domain;
      Guard(branch->condition_, selected);
      Waits(branch->then_body_, selected, cursor);
      return;
    }
    const auto eval = As<EvalStmt>(statement);
    const auto call = eval ? As<Call>(eval->expr_) : nullptr;
    CHECK(IsOp(call, "sync.wait")) << "Conditional scoped token statement is not a wait";
    cursor.pending.push_back({call->args_.at(0), Pipe(call, "dst_pipe"), domain});
    cursor.last.reset();
  }
  void Read(const StmtPtr& statement, Cursor& cursor) {
    if (auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Read(child, cursor);
      return;
    }
    CHECK(!cursor.returned && !cursor.yield) << "Unreachable scoped token cut";
    const auto call = Payload(statement);
    const auto payload = sites_.find(call.get());
    if (payload != sites_.end()) {
      CheckScope(payload->second);
      for (const auto& wait : cursor.pending) {
        const auto [r, s] = payload->second;
        CHECK(wait.pipe == input_.segments[r].sites[s].pipe) << "Scoped wait is not at its destination cut";
        uses_.push_back({wait, payload->second});
      }
      cursor.pending.clear();
      cursor.last = payload->second;
      return;
    }
    if (auto eval = As<EvalStmt>(statement)) {
      Waits(eval, {}, cursor);
      return;
    }
    if (auto branch = As<IfStmt>(statement)) {
      if (branch->return_vars_.empty()) {
        Waits(branch, {}, cursor);
        return;
      }
      CHECK(cursor.pending.empty() && !scope_.empty() && branch->return_vars_.size() == 1 &&
            branch->else_body_ && As<Eq>(branch->condition_))
          << "Unsupported scoped token capture";
      const auto [frame, selected] = Coordinate(branch->condition_);
      CHECK(frame == scope_.back()) << "Scoped capture requires its own exact visit coordinate";
      const auto value = PureYield(branch->then_body_);
      const auto previous = PureYield(*branch->else_body_);
      CHECK(value && previous && AsVarLike(previous)) << "Scoped capture must select pure token facts";
      equations_.emplace(branch->return_vars_.front().get(),
                         Equation{EquationKind::Capture, value, frame, AsVarLike(previous).get(), selected});
      cursor.last.reset();
      return;
    }
    if (auto assign = As<AssignStmt>(statement)) {
      if (!As<SyncTokenType>(assign->var_->GetType())) {
        CHECK(cursor.pending.empty()) << "Scalar operation separates scoped wait and consumer";
        cursor.last.reset();
        return;
      }
      const auto token = As<Call>(assign->value_);
      if (IsOp(token, "sync.record")) {
        CHECK(cursor.last && cursor.pending.empty()) << "Scoped token record is not at a direct payload cut";
        const auto [r, s] = *cursor.last;
        CHECK(Pipe(token, "pipe") == input_.segments[r].sites[s].pipe) << "Scoped record pipe mismatch";
        resolved_.emplace(assign->var_.get(), Current(*cursor.last));
      } else if (IsOp(token, "sync.completed")) {
        resolved_.emplace(assign->var_.get(), Pattern{});
      } else {
        CHECK(AsVarLike(assign->value_)) << "Scoped token joins require a separate prefix qualification";
        equations_.emplace(assign->var_.get(), Equation{EquationKind::Alias, assign->value_});
      }
      return;
    }
    CHECK(cursor.pending.empty()) << "Scoped wait crosses an unsupported control cut";
    cursor.last.reset();
    if (auto loop = As<ForStmt>(statement)) {
      const auto frame = loops_.at(loop.get());
      const auto count = frames_.at(frame).count;
      CHECK(loop->iter_args_.size() == loop->return_vars_.size()) << "Scoped token loop arity mismatch";
      for (const auto& arg : loop->iter_args_) initials_.push_back(arg->initValue_);
      Cursor body;
      scope_.push_back(frame);
      active_frames_.insert(frame);
      if (count) Read(loop->body_, body);
      active_frames_.erase(frame);
      scope_.pop_back();
      CHECK(body.pending.empty()) << "Scoped wait has no consumer before loop exit";
      CHECK(!count || (loop->iter_args_.empty() && (!body.yield || body.yield->value_.empty())) ||
            (body.yield && body.yield->value_.size() == loop->iter_args_.size()))
          << "Scoped loop requires unconditional simultaneous token yields";
      for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
        if (!count) {
          resolved_.emplace(loop->return_vars_[i].get(), Pattern{});
          continue;
        }
        const auto value = body.yield->value_[i];
        const auto carry = loop->iter_args_[i].get();
        equations_.emplace(carry, Equation{EquationKind::Carry, value, frame, carry});
        equations_.emplace(loop->return_vars_[i].get(), Equation{EquationKind::Exit, value, frame, carry});
      }
      return;
    }
    if (auto yield = As<YieldStmt>(statement)) {
      CHECK(!scope_.empty()) << "Scoped token yield is outside a loop";
      cursor.yield = yield;
      return;
    }
    if (As<ReturnStmt>(statement)) {
      CHECK(scope_.empty()) << "Scoped token return inside loop is unavailable";
      cursor.returned = true;
      return;
    }
    CHECK(false) << "Unsupported scoped token statement";
  }

  void Transform(Pattern& pattern, const Equation& equation) const {
    if (!pattern.source || equation.kind == EquationKind::Alias) return;
    const auto* frame = equation.frame;
    if (equation.kind == EquationKind::Carry) {
      for (auto& digit : pattern.digits)
        if (digit.variable == frame) {
          CHECK(digit.offset > std::numeric_limits<int64_t>::min()) << "Scoped token history overflows";
          --digit.offset;
        }
      const auto found = pattern.valid.find(frame);
      const auto previous = found == pattern.valid.end() ? Full(frame) : found->second;
      pattern.valid[frame] = {
          static_cast<int64_t>(std::min<__int128>(frames_.at(frame).count, __int128(previous.first) + 1)),
          static_cast<int64_t>(std::min<__int128>(frames_.at(frame).count, __int128(previous.second) + 1))};
      return;
    }
    const auto selected = equation.selected;
    const auto valid = pattern.valid.find(frame);
    if (selected < 0 || selected >= frames_.at(frame).count ||
        (valid != pattern.valid.end() &&
         (selected < valid->second.first || selected >= valid->second.second))) {
      pattern = {};
      return;
    }
    pattern.valid.erase(frame);
    for (auto& digit : pattern.digits)
      if (digit.variable == frame) {
        CHECK(!__builtin_add_overflow(digit.offset, selected, &digit.offset))
            << "Scoped token substituted coordinate overflows";
        digit.variable = nullptr;
      }
  }
  Pattern Resolve(const ExprPtr& expression) {
    // Unary equations after checking capture's own-carry recurrence. Resolve
    // iteratively so long alias/history chains cannot exhaust the C++ stack.
    std::vector<std::pair<const Var*, Equation>> path;
    std::set<const Var*> seen;
    ExprPtr next = expression;
    Pattern result;
    while (true) {
      const auto var = AsVarLike(next);
      CHECK(var) << "Scoped token transport requires named completion facts";
      const auto memo = resolved_.find(var.get());
      if (memo != resolved_.end()) {
        result = memo->second;
        break;
      }
      CHECK(seen.insert(var.get()).second) << "Cyclic scoped token equation";
      const auto found = equations_.find(var.get());
      CHECK(found != equations_.end()) << "Unavailable scoped token binding";
      auto equation = found->second;
      CHECK(equation.kind != EquationKind::Capture) << "Scoped capture may only be used through loop result";
      if (equation.kind == EquationKind::Exit || equation.kind == EquationKind::Carry) {
        const auto yielded = AsVarLike(equation.value);
        const auto capture = yielded ? equations_.find(yielded.get()) : equations_.end();
        if (capture != equations_.end() && capture->second.kind == EquationKind::Capture) {
          CHECK(equation.kind == EquationKind::Exit && capture->second.frame == equation.frame &&
                capture->second.carry == equation.carry)
              << "Scoped capture must retain its own invocation carry";
          equation.selected = capture->second.selected;
          equation.value = capture->second.value;
        } else if (equation.kind == EquationKind::Exit) {
          equation.selected = frames_.at(equation.frame).count - 1;
        }
      }
      path.emplace_back(var.get(), equation);
      next = equation.value;
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      Transform(result, it->second);
      resolved_.emplace(it->first, result);
    }
    return result;
  }
  std::optional<PhysicalSyncFamily> Family(const Pattern& pattern, const Use& use) const {
    const auto [sr, ss] = *pattern.source;
    const auto [tr, ts] = use.target;
    const auto& source = input_.segments[sr].nested_frames;
    const auto& target = input_.segments[tr].nested_frames;
    auto domain = use.wait.domain;
    std::set<const ForStmt*> target_frames;
    for (const auto& frame : target) target_frames.insert(frame.loop.get());
    for (const auto& [frame, interval] : pattern.valid) {
      CHECK(target_frames.count(frame)) << "Unsubstituted scoped token history frame";
      Intersect(domain, frame, interval);
    }
    std::map<const ForStmt*, Interval> ranges;
    std::optional<size_t> last_axis;
    for (size_t i = 0; i < target.size(); ++i) {
      const auto key = target[i].loop.get();
      const auto found = domain.find(key);
      const auto interval = found == domain.end() ? Full(key) : found->second;
      if (interval.first >= interval.second) return std::nullopt;
      ranges.emplace(key, interval);
      if (interval.second - interval.first > 1) last_axis = i;
    }
    CHECK(pattern.digits.size() == source.size()) << "Scoped token source coordinate arity mismatch";
    std::vector<int64_t> source_strides(source.size()), target_strides(target.size());
    auto strides = [](const std::vector<SyncLoopFrame>& frames, std::vector<int64_t>& output) {
      __int128 stride = 1;
      for (size_t i = frames.size(); i-- > 0;) {
        CHECK(stride <= std::numeric_limits<int64_t>::max()) << "Scoped token stride overflows";
        output[i] = static_cast<int64_t>(stride);
        stride *= frames[i].count;
        CHECK(stride <= std::numeric_limits<int64_t>::max()) << "Scoped token frame product overflows";
      }
    };
    strides(source, source_strides);
    strides(target, target_strides);
    __int128 source_base = 0, target_base = 0, count = 1;
    for (size_t i = 0; i < source.size(); ++i) {
      const auto& digit = pattern.digits[i];
      __int128 value = digit.offset;
      if (digit.variable) {
        const auto found = ranges.find(digit.variable);
        CHECK(digit.variable == source[i].loop.get() && found != ranges.end())
            << "Scoped token source refers to an unavailable original frame";
        value += found->second.first;
        CHECK(value + found->second.second - found->second.first <= source[i].count)
            << "Scoped token source interval exceeds its original frame";
      }
      CHECK(value >= 0 && value < source[i].count) << "Scoped token source has no active occurrence";
      source_base += value * source_strides[i];
    }
    PhysicalSyncFamily out{sr, ss, tr, ts, 0, 0, 1, {}};
    for (size_t i = 0; i < target.size(); ++i) {
      const auto interval = ranges.at(target[i].loop.get());
      const auto axis_count = interval.second - interval.first;
      target_base += __int128(interval.first) * target_strides[i];
      if (last_axis && i <= *last_axis) {
        CHECK(i < source.size() && source[i].loop == target[i].loop &&
              (axis_count == 1 || pattern.digits[i].variable == target[i].loop.get()))
            << "Scoped token fanout is not an injective common-prefix handoff family";
        out.axes.push_back({target[i].loop, axis_count, source_strides[i], target_strides[i]});
        count *= axis_count;
        CHECK(count <= std::numeric_limits<int64_t>::max()) << "Scoped token family count overflows";
      }
    }
    CHECK(source_base < input_.segments[sr].count && target_base < input_.segments[tr].count &&
          count <= std::numeric_limits<int64_t>::max())
        << "Scoped token family exceeds its admitted occurrence domain";
    out.source_first = static_cast<int64_t>(source_base);
    out.target_first = static_cast<int64_t>(target_base);
    out.count = static_cast<int64_t>(count);
    return out;
  }
};
}  // namespace
std::vector<PhysicalSyncFamily> ParseScopedTokenFamilies(
    const FunctionPtr& function, const CompactSyncInput& input,
    const std::map<const ForStmt*, ForStmtPtr>& stripped_loops) {
  return Parser(function, input, stripped_loops).Run();
}
}  // namespace pypto::ir::sync
