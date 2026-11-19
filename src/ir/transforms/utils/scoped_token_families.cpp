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

// Interpret original-frame token equations, never storage demands. Resolution
// is iterative and per use: capture-and-hold is specialized to that consumer's
// exact domain and cannot share a global resolved-pattern cache. For E inspected
// equation steps and frame depth D, pattern work is O(E D), plus indexed/domain
// lookups; E can reach U T for U uses and T equations. Counters expose E rather
// than claiming one resolution per equation. No trip count is unfolded.
// Guard DAG visits are per supplied condition, not global: shared conditions
// can be revisited, and nested guard scopes copy their domain maps. Optional
// family normalization uses counted indexed passes and candidate comparisons;
// candidates within a group may be quadratic, and reuse proof cost is separate.
#include "src/ir/transforms/utils/scoped_token_families.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace {
using Site = std::pair<size_t, size_t>;
using Interval = std::pair<int64_t, int64_t>;  // Half-open normalized visits.
struct Progression {
  int64_t first = 0, second = 0, step = 1;
  int64_t Count() const { return first < second ? 1 + (second - 1 - first) / step : 0; }
  int64_t Last() const { return first + (Count() - 1) * step; }
};
using Domain = std::map<const ForStmt*, Progression>;
struct Digit {
  const ForStmt* variable = nullptr;
  int64_t offset = 0;
};
struct Pattern {
  std::optional<Site> source;
  std::vector<Digit> digits;
  // Exactly the domain where this fact is non-completed, relative to the
  // current consumer when capture-and-hold is specialized. No per-use pattern
  // is stored in the global binding cache. This is not merely where a proof succeeded. Completed entry and
  // pure own-carry captures establish this invariant, allowing waits on completed facts to vanish on other
  // visits.
  Domain valid;
};
enum class EquationKind { Alias, Carry, Exit, Capture };
struct Equation {
  EquationKind kind;
  ExprPtr value;
  const ForStmt* frame = nullptr;
  const Var* carry = nullptr;
  int64_t selected = 0;
  std::optional<Progression> capture;
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
ExprPtr PureYield(StmtPtr body) {
  while (auto sequence = As<SeqStmts>(body)) {
    if (sequence->stmts_.size() != 1) return nullptr;
    body = sequence->stmts_.front();
  }
  const auto yield = As<YieldStmt>(body);
  return yield && yield->value_.size() == 1 ? yield->value_.front() : nullptr;
}
class Parser {
 public:
  Parser(const FunctionPtr& function, const CompactSyncInput& input,
         const std::map<const ForStmt*, ForStmtPtr>& stripped, ScopedTokenWork& work)
      : function_(function), input_(input), work_(work) {
    CHECK(input.region_tree) << "Scoped token interpretation requires an original region tree";
    for (size_t r = 0; r < input.segments.size(); ++r) {
      const auto& segment = input.segments[r];
      for (const auto& frame : segment.nested_frames) Register(frame);
      for (size_t s = 0; s < segment.sites.size(); ++s)
        CHECK(sites_.emplace(segment.sites[s].original.get(), Site{r, s}).second)
            << "Scoped token payload anchors must be distinct";
    }
    Register(*input.region_tree);
    block_ = input.invocation_coordinate ? input.invocation_coordinate->Expression()
                                         : DominatingSyncBlock(function);
    for (const auto& [original, erased] : stripped) {
      // Import deliberately omits bodies of proved zero-trip frames. A
      // reachable missing frame is still rejected by Read, before its body.
      if (frames_.count(erased.get())) loops_.emplace(original, erased.get());
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
      auto pattern = Resolve(use.wait.token, &use.wait.domain);
      if (!pattern.source) continue;
      auto family = Family(pattern, use);
      if (family) {
        output.push_back(std::move(*family));
        ++work_.families;
      }
    }
    return output;
  }

 private:
  FunctionPtr function_;
  ExprPtr block_;
  const CompactSyncInput& input_;
  ScopedTokenWork& work_;
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
    CHECK(frame.count >= 0 && frame.start >= 0 && frame.step > 0) << "Invalid scoped token frame";
    // Fresh import proves a uniform count and original-width arithmetic over
    // this complete invocation case. Retain its symbolic origin; the numeric
    // start is not a representative worker that can replace that expression.
    CHECK(!frame.start_expression || structural_equal(frame.start_expression, frame.loop->start_, false))
        << "Scoped token frame has a different original start expression";
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
  Progression Full(const ForStmt* frame) const { return {0, frames_.at(frame).count}; }
  static void Restrict(Progression& value, Interval interval) {
    const auto low = std::max(value.first, interval.first);
    value.second = std::min(value.second, interval.second);
    if (low >= value.second) {
      value.first = value.second;
      return;
    }
    const auto skip = (__int128(low) - value.first + value.step - 1) / value.step;
    const auto first = __int128(value.first) + skip * value.step;
    value.first = static_cast<int64_t>(std::min<__int128>(first, value.second));
    if (value.Count()) value.second = value.Last() + 1;
  }
  void Intersect(Domain& domain, const ForStmt* frame, Progression other) const {
    auto [at, inserted] = domain.try_emplace(frame, Full(frame));
    auto& value = at->second;
    if (other.step != 1) {
      CHECK(value.step == 1 || value.step == other.step)
          << "Scoped token domain combines unsupported residue periods";
      if (value.step != 1 && (value.first - other.first) % other.step != 0) {
        value.second = value.first;
        return;
      }
      if (value.step == 1) {
        auto remainder = (value.first - other.first) % other.step;
        if (remainder < 0) remainder += other.step;
        const auto first = __int128(value.first) + (remainder ? other.step - remainder : 0);
        value.first = static_cast<int64_t>(std::min<__int128>(first, value.second));
        value.step = other.step;
      }
    }
    Restrict(value, {other.first, other.second});
  }
  // Guards denote exact rectangular domains, not an enclosure of arbitrary
  // Boolean control. Every variable must be an active original frame.
  std::pair<const ForStmt*, int64_t> Coordinate(const ExprPtr& condition) const {
    const auto binary = As<BinaryExpr>(condition);
    const auto var = binary ? AsVarLike(binary->left_) : nullptr;
    const auto found = var ? variables_.find(var.get()) : variables_.end();
    CHECK(binary && found != variables_.end() && active_frames_.count(found->second))
        << "Scoped token guard requires an active original coordinate";
    const auto& frame = frames_.at(found->second);
    if (frame.start_expression && !As<ConstInt>(frame.start_expression)) {
      const auto& threshold = binary->right_;
      CHECK(structural_equal(threshold->GetType(), frame.loop->loop_var_->GetType(), false))
          << "Scoped token symbolic threshold changes its original coordinate type";
      int64_t offset = 0;
      if (!structural_equal(threshold, frame.start_expression, false)) {
        const auto sum = As<Add>(threshold);
        const auto constant = sum ? As<ConstInt>(sum->right_) : nullptr;
        // Free variables preserve identity. This admits a serialized copy of
        // the same pure start expression, never a same-named foreign frame or
        // an algebraically guessed parameter binding.
        CHECK(sum && constant && structural_equal(sum->left_, frame.start_expression, false) &&
              structural_equal(sum->left_->GetType(), threshold->GetType(), false) &&
              structural_equal(constant->GetType(), threshold->GetType(), false))
            << "Scoped token symbolic guard requires its original start plus a constant offset";
        offset = constant->value_;
      }
      // Include the one-past threshold used by half-open guards. Fresh import
      // proves the post-loop increment, not merely the last payload coordinate,
      // in the original type for every caller. Guard also forms digit+1 below.
      CHECK(offset >= 0 && offset % frame.step == 0 && offset / frame.step <= frame.count &&
            offset / frame.step < std::numeric_limits<int64_t>::max())
          << "Scoped token symbolic guard is outside its original visit lattice";
      return {found->second, offset / frame.step};
    }
    const auto constant = As<ConstInt>(binary->right_);
    CHECK(constant) << "Scoped token guard requires a constant original coordinate";
    const __int128 delta = __int128(constant->value_) - frame.start;
    CHECK(delta >= 0 && delta % frame.step == 0 && delta / frame.step < std::numeric_limits<int64_t>::max())
        << "Scoped token guard has an unavailable normalized coordinate";
    return {found->second, static_cast<int64_t>(delta / frame.step)};
  }
  // Only the original dominating launch value may qualify invocation guards.
  // A guard varying inside the complete case interval remains unproved.
  std::optional<bool> InvocationPredicate(const ExprPtr& expression) const {
    const auto binary = As<BinaryExpr>(expression);
    const auto constant = binary ? As<ConstInt>(binary->right_) : nullptr;
    if (!binary || !constant || !block_ || binary->left_ != block_) return std::nullopt;
    CHECK(input_.block_bounds) << "Scoped invocation guard has no caller domain";
    const auto [lo, hi] = *input_.block_bounds;
    const auto n = constant->value_;
    std::optional<bool> result;
    if (As<Ge>(expression)) {
      if (lo >= n)
        result = true;
      else if (hi < n)
        result = false;
    } else if (As<Lt>(expression)) {
      if (hi < n)
        result = true;
      else if (lo >= n)
        result = false;
    } else if (As<Gt>(expression)) {
      if (lo > n)
        result = true;
      else if (hi <= n)
        result = false;
    } else if (As<Eq>(expression)) {
      if (lo == hi && lo == n)
        result = true;
      else if (n < lo || n > hi)
        result = false;
    }
    CHECK(result.has_value()) << "Scoped invocation guard is not uniform over the complete case";
    return result;
  }
  bool Guard(const ExprPtr& condition, Domain& domain) const {
    bool enabled = true;
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
      if (auto selected = InvocationPredicate(expression)) {
        enabled &= *selected;
        continue;
      }
      if (const auto equal = As<Eq>(expression)) {
        if (const auto modulo = As<FloorMod>(equal->left_)) {
          const auto normalized = As<Sub>(modulo->left_);
          const auto variable = normalized ? AsVarLike(normalized->left_) : nullptr;
          const auto at = variable ? variables_.find(variable.get()) : variables_.end();
          const auto modulus = As<ConstInt>(modulo->right_);
          const auto residue = As<ConstInt>(equal->right_);
          CHECK(at != variables_.end() && active_frames_.count(at->second) && modulus && residue)
              << "Scoped token residue requires an active original frame and literal modulus";
          const auto& frame = frames_.at(at->second);
          const auto start = frame.start_expression ? frame.start_expression : frame.loop->start_;
          CHECK(structural_equal(normalized->right_, start, false) && modulus->value_ > 0 &&
                modulus->value_ % frame.step == 0 && residue->value_ >= 0 &&
                residue->value_ < modulus->value_ && residue->value_ % frame.step == 0 &&
                structural_equal(normalized->GetType(), variable->GetType(), false) &&
                structural_equal(modulo->GetType(), variable->GetType(), false) &&
                structural_equal(modulus->GetType(), variable->GetType(), false) &&
                structural_equal(residue->GetType(), variable->GetType(), false))
              << "Scoped token residue changes its original nonnegative coordinate lattice";
          Intersect(domain, at->second,
                    {residue->value_ / frame.step, frame.count, modulus->value_ / frame.step});
          continue;
        }
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
    return enabled;
  }
  void Waits(const StmtPtr& statement, const Domain& domain, Cursor& cursor, bool enabled = true) {
    if (auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Waits(child, domain, cursor, enabled);
      return;
    }
    if (auto branch = As<IfStmt>(statement)) {
      CHECK(branch->return_vars_.empty() && !branch->else_body_) << "Unsupported scoped wait branch";
      auto selected = domain;
      const bool selected_case = Guard(branch->condition_, selected);
      Waits(branch->then_body_, selected, cursor, enabled && selected_case);
      return;
    }
    const auto eval = As<EvalStmt>(statement);
    const auto call = eval ? As<Call>(eval->expr_) : nullptr;
    CHECK(IsOp(call, "sync.wait")) << "Conditional scoped token statement is not a wait";
    ++work_.waits_inspected;
    if (!enabled) ++work_.inactive_waits;
    if (enabled)
      cursor.pending.push_back({call->args_.at(0), RequireSyncPipeKwarg(call, "dst_pipe"), domain});
    cursor.last.reset();
  }
  void Read(const StmtPtr& statement, Cursor& cursor) {
    ++work_.statements;
    if (auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Read(child, cursor);
      return;
    }
    CHECK(!cursor.returned && !cursor.yield) << "Unreachable scoped token cut";
    if (const auto branch = As<IfStmt>(statement); branch && branch->return_vars_.empty()) {
      if (const auto resolved = input_.static_branches.find(branch.get());
          resolved != input_.static_branches.end()) {
        if (resolved->second) {
          Read(branch->then_body_, cursor);
        } else if (branch->else_body_) {
          Read(*branch->else_body_, cursor);
        }
        return;
      }
    }
    CallPtr call = nullptr;
    if (const auto eval = As<EvalStmt>(statement)) {
      call = As<Call>(eval->expr_);
    } else if (const auto branch = As<IfStmt>(statement); branch && branch->return_vars_.empty()) {
      call = FindSiteCallInStmt(branch->then_body_, sites_);
      if (!call && branch->else_body_) call = FindSiteCallInStmt(*branch->else_body_, sites_);
    }
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
      if (std::none_of(branch->return_vars_.begin(), branch->return_vars_.end(),
                       [](const VarPtr& v) { return static_cast<bool>(As<SyncTokenType>(v->GetType())); })) {
        CHECK(cursor.pending.empty()) << "Scalar coordinate selection cannot sit inside a pending token wait";
        cursor.last.reset();
        return;
      }
      CHECK(cursor.pending.empty() && !scope_.empty() && branch->return_vars_.size() == 1 &&
            branch->else_body_)
          << "Unsupported scoped token capture";
      Domain capture;
      CHECK(Guard(branch->condition_, capture) && capture.size() == 1 &&
            capture.begin()->first == scope_.back())
          << "Scoped capture requires its own exact active visit progression";
      const auto frame = capture.begin()->first;
      const auto selected = capture.begin()->second;
      const auto value = PureYield(branch->then_body_);
      const auto previous = PureYield(*branch->else_body_);
      CHECK(value && previous && AsVarLike(previous)) << "Scoped capture must select pure token facts";
      equations_.emplace(
          branch->return_vars_.front().get(),
          Equation{EquationKind::Capture, value, frame, AsVarLike(previous).get(), 0, selected});
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
        CHECK(RequireSyncPipeKwarg(token, "pipe") == input_.segments[r].sites[s].pipe)
            << "Scoped record pipe mismatch";
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
      CHECK(loops_.count(loop.get())) << "Missing scoped token original frame";
      const auto frame = loops_.at(loop.get());
      const auto count = frames_.at(frame).count;
      CHECK(loop->iter_args_.size() == loop->return_vars_.size()) << "Scoped token loop arity mismatch";
      for (const auto& arg : loop->iter_args_)
        if (As<SyncTokenType>(arg->GetType())) initials_.push_back(arg->initValue_);
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
        if (!As<SyncTokenType>(loop->iter_args_[i]->GetType())) continue;
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

  void Transform(Pattern& pattern, const Equation& equation, const Domain* consumer) const {
    if (!pattern.source || equation.kind == EquationKind::Alias) return;
    const auto* frame = equation.frame;
    if (equation.kind == EquationKind::Carry && equation.capture) {
      CHECK(consumer) << "Captured history requires an actual consumer domain";
      ++work_.capture_specializations;
      const auto capture = *equation.capture;
      const auto available = pattern.valid.find(frame);
      if (available != pattern.valid.end()) {
        const auto& valid = available->second;
        CHECK(capture.first >= valid.first && capture.Last() < valid.second &&
              (capture.first - valid.first) % valid.step == 0 && capture.step % valid.step == 0)
            << "Capture can select a completed or incompatible source fact";
      }
      const auto found = consumer->find(frame);
      auto uses = found == consumer->end() ? Full(frame) : found->second;
      Restrict(uses, {capture.first + 1, frames_.at(frame).count});
      if (!uses.Count()) {
        pattern = {};
        return;
      }
      if (capture.Count() == 1 || uses.Count() == 1) {
        const auto selected = std::min(
            capture.Last(), capture.first + ((uses.first - 1 - capture.first) / capture.step) * capture.step);
        for (auto& digit : pattern.digits)
          if (digit.variable == frame) {
            CHECK(!__builtin_add_overflow(digit.offset, selected, &digit.offset))
                << "Scoped captured source coordinate overflows";
            digit.variable = nullptr;
          }
      } else {
        const auto delta = 1 + (uses.first - 1 - capture.first) % capture.step;
        CHECK(uses.step % capture.step == 0 && uses.Last() - delta <= capture.Last())
            << "Scoped consumer does not prove one latest-source residue without fanout";
        for (auto& digit : pattern.digits)
          if (digit.variable == frame) {
            CHECK(!__builtin_sub_overflow(digit.offset, delta, &digit.offset))
                << "Scoped captured source history overflows";
          }
      }
      // This is a per-use resolution, never a cacheable global fact. Keeping
      // the exact consumer progression preserves every hole in its activity.
      pattern.valid[frame] = uses;
      return;
    }
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
          static_cast<int64_t>(std::min<__int128>(frames_.at(frame).count, __int128(previous.second) + 1)),
          previous.step};
      return;
    }
    const auto selected = equation.selected;
    const auto valid = pattern.valid.find(frame);
    if (selected < 0 || selected >= frames_.at(frame).count ||
        (valid != pattern.valid.end() &&
         (selected < valid->second.first || selected >= valid->second.second ||
          (selected - valid->second.first) % valid->second.step != 0))) {
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
  Pattern Resolve(const ExprPtr& expression, const Domain* consumer = nullptr) {
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
      ++work_.equation_visits;
      CHECK(equation.kind != EquationKind::Capture) << "Scoped capture may only be used through loop result";
      if (equation.kind == EquationKind::Exit || equation.kind == EquationKind::Carry) {
        const auto yielded = AsVarLike(equation.value);
        const auto capture = yielded ? equations_.find(yielded.get()) : equations_.end();
        if (capture != equations_.end() && capture->second.kind == EquationKind::Capture) {
          CHECK(capture->second.frame == equation.frame && capture->second.carry == equation.carry)
              << "Scoped capture must retain its own invocation carry";
          CHECK(capture->second.capture) << "Missing exact capture progression";
          equation.capture = capture->second.capture;
          if (!equation.capture->Count()) {
            result = {};
            break;
          }
          equation.selected = equation.capture->Last();
          equation.value = capture->second.value;
        } else if (equation.kind == EquationKind::Exit) {
          equation.selected = frames_.at(equation.frame).count - 1;
        }
      }
      path.emplace_back(var.get(), equation);
      next = equation.value;
    }
    // A captured carry is specialized to this consumer domain. Composing a
    // second transformation of that same frame would need to pull the domain
    // backwards through it first. Preserve refusal rather than reuse the
    // unchanged consumer domain as a certificate for another history.
    std::map<const ForStmt*, size_t> transforms;
    std::set<const ForStmt*> captured_frames;
    for (const auto& [variable, equation] : path) {
      if (equation.kind == EquationKind::Alias) continue;
      ++transforms[equation.frame];
      if (equation.capture) captured_frames.insert(equation.frame);
    }
    for (const auto* frame : captured_frames)
      CHECK(transforms.at(frame) == 1) << "Composed captured history requires additional coordinate evidence";
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      Transform(result, it->second, consumer);
    }
    return result;
  }
  std::map<size_t, std::shared_ptr<const SyncEndpointFrame>> endpoint_frames_;
  using EndpointKey = std::tuple<size_t, std::vector<Interval>, std::vector<int64_t>>;
  std::map<EndpointKey, std::shared_ptr<const SyncEndpointDomain>> endpoint_domains_;
  std::shared_ptr<const SyncEndpointDomain> Endpoint(size_t region, std::vector<Interval> ranges,
                                                     std::vector<int64_t> steps) {
    EndpointKey key{region, std::move(ranges), std::move(steps)};
    if (auto found = endpoint_domains_.find(key); found != endpoint_domains_.end()) {
      return found->second;
    }
    auto& frame = endpoint_frames_[region];
    if (!frame)
      frame = std::make_shared<SyncEndpointFrame>(SyncEndpointFrame{input_.owner, input_.block_bounds, region,
                                                                    input_.segments[region].nested_frames});
    auto domain =
        std::make_shared<SyncEndpointDomain>(SyncEndpointDomain{frame, std::get<1>(key), std::get<2>(key)});
    endpoint_domains_.emplace(std::move(key), domain);
    return domain;
  }
  std::optional<PhysicalSyncFamily> Family(const Pattern& pattern, const Use& use) {
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
    Domain ranges;
    std::optional<size_t> last_axis;
    for (size_t i = 0; i < target.size(); ++i) {
      const auto key = target[i].loop.get();
      const auto found = domain.find(key);
      const auto interval = found == domain.end() ? Full(key) : found->second;
      if (interval.first >= interval.second) return std::nullopt;
      ranges.emplace(key, interval);
      if (interval.Count() > 1) last_axis = i;
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
    std::vector<Interval> source_ranges, target_ranges;
    std::vector<int64_t> source_steps, target_steps;
    for (size_t i = 0; i < source.size(); ++i) {
      const auto& digit = pattern.digits[i];
      __int128 value = digit.offset;
      if (digit.variable) {
        const auto found = ranges.find(digit.variable);
        CHECK(digit.variable == source[i].loop.get() && found != ranges.end())
            << "Scoped token source refers to an unavailable original frame";
        value += found->second.first;
        CHECK(value + found->second.Last() - found->second.first < source[i].count)
            << "Scoped token source interval exceeds its original frame";
      }
      CHECK(value >= 0 && value < source[i].count) << "Scoped token source has no active occurrence";
      source_base += value * source_strides[i];
      const auto width =
          digit.variable ? ranges.at(digit.variable).Last() - ranges.at(digit.variable).first + 1 : 1;
      source_ranges.emplace_back(static_cast<int64_t>(value), static_cast<int64_t>(value) + width);
      source_steps.push_back(digit.variable ? ranges.at(digit.variable).step : 1);
    }
    PhysicalSyncFamily out{sr, ss, tr, ts, 0, 0, 1, {}};
    for (size_t i = 0; i < target.size(); ++i) {
      const auto interval = ranges.at(target[i].loop.get());
      target_ranges.emplace_back(interval.first, interval.Last() + 1);
      target_steps.push_back(interval.step);
      const auto axis_count = interval.Count();
      target_base += __int128(interval.first) * target_strides[i];
      if (last_axis && i <= *last_axis) {
        CHECK(i < source.size() && source[i].loop == target[i].loop &&
              (axis_count == 1 || pattern.digits[i].variable == target[i].loop.get()))
            << "Scoped token fanout is not an injective common-prefix handoff family";
        const auto source_stride = __int128(source_strides[i]) * interval.step;
        const auto target_stride = __int128(target_strides[i]) * interval.step;
        CHECK(source_stride <= std::numeric_limits<int64_t>::max() &&
              target_stride <= std::numeric_limits<int64_t>::max())
            << "Scoped progression stride overflows";
        out.axes.push_back({target[i].loop, axis_count, static_cast<int64_t>(source_stride),
                            static_cast<int64_t>(target_stride), interval.step, interval.step});
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
    out.source_domain = Endpoint(sr, std::move(source_ranges), std::move(source_steps));
    out.target_domain = Endpoint(tr, std::move(target_ranges), std::move(target_steps));
    return out;
  }
};
// Optional physical preparation. Parsing actual tokens remains independent of
// order/allocation services; the caller supplies its existing reuse proof.
class FamilyNormalizer {
 public:
  FamilyNormalizer(const CompactSyncInput& input,
                   const std::function<bool(const PhysicalSyncFamily&)>& can_reuse, ScopedTokenWork& work)
      : input_(input), can_reuse_(can_reuse), work_(work) {}
  void Run(std::vector<PhysicalSyncFamily>& families) {
    for (const auto& family : families)
      CHECK(family.ids.empty() && family.source_domain && family.target_domain)
          << "Scoped family normalization requires unassigned exact endpoint inventories";
    Coalesce(families);
  }

 private:
  const CompactSyncInput& input_;
  const std::function<bool(const PhysicalSyncFamily&)>& can_reuse_;
  ScopedTokenWork& work_;
  static Progression EndpointProgression(const SyncEndpointDomain& domain, size_t axis) {
    const auto [first, end] = domain.ranges.at(axis);
    return {first, end, domain.steps.empty() ? 1 : domain.steps.at(axis)};
  }
  // Finite CRT intersection, not an interval-overlap approximation. Wide
  // arithmetic covers the lcm of two admitted int64 periods without expansion.
  static bool Disjoint(const Progression& a, const Progression& b) {
    const auto low = std::max(a.first, b.first), high = std::min(a.Last(), b.Last());
    if (low > high) return true;
    const auto divisor = std::gcd(a.step, b.step);
    const auto difference = b.first - a.first;
    if (difference % divisor) return true;
    const auto modulus = b.step / divisor;
    __int128 x = 0;
    if (modulus != 1) {
      __int128 r = a.step / divisor, next_r = modulus, coefficient = 1, next_coefficient = 0;
      while (next_r) {
        const auto quotient = r / next_r;
        const auto remainder = r - quotient * next_r;
        r = next_r;
        next_r = remainder;
        const auto replacement = coefficient - quotient * next_coefficient;
        coefficient = next_coefficient;
        next_coefficient = replacement;
      }
      x = (__int128(difference / divisor) * coefficient) % modulus;
      if (x < 0) x += modulus;
    }
    auto first = __int128(a.first) + __int128(a.step) * x;
    const auto period = __int128(a.step) * modulus;
    if (first < low) first += ((__int128(low) - first + period - 1) / period) * period;
    return first > high;
  }
  static std::optional<Progression> Union(const Progression& a, const Progression& b) {
    CHECK(a.Count() && b.Count()) << "Empty scoped family reached exact union";
    const auto first = std::min(a.first, b.first), last = std::max(a.Last(), b.Last());
    auto step = std::gcd(a.Count() > 1 ? a.step : 0, b.Count() > 1 ? b.step : 0);
    step = std::gcd(step, std::max(a.first, b.first) - first);
    if (!step) return std::nullopt;  // Equal singleton records must not be deduplicated.
    const auto count = (__int128(last) - first) / step + 1;
    if (count != __int128(a.Count()) + b.Count() || !Disjoint(a, b)) return std::nullopt;
    // Both supplied APs are subsets of this gcd lattice. Disjoint cardinality
    // equality therefore proves the complete union, including every hole.
    return Progression{first, last + 1, step};
  }
  void Rebuild(PhysicalSyncFamily& family, std::vector<Interval> source_ranges,
               std::vector<int64_t> source_steps, std::vector<Interval> target_ranges,
               std::vector<int64_t> target_steps) {
    const auto& source = input_.segments[family.source_region].nested_frames;
    const auto& target = input_.segments[family.target_region].nested_frames;
    for (size_t i = 0; i < source_ranges.size(); ++i)
      if (source_ranges[i].second - source_ranges[i].first == 1) source_steps[i] = 1;
    for (size_t i = 0; i < target_ranges.size(); ++i)
      if (target_ranges[i].second - target_ranges[i].first == 1) target_steps[i] = 1;
    auto base = [](const std::vector<SyncLoopFrame>& frames, const std::vector<Interval>& ranges) {
      __int128 value = 0;
      for (size_t i = 0; i < frames.size(); ++i) value = value * frames[i].count + ranges[i].first;
      CHECK(value <= std::numeric_limits<int64_t>::max()) << "Unified scoped endpoint overflows";
      return static_cast<int64_t>(value);
    };
    family.source_first = base(source, source_ranges);
    family.target_first = base(target, target_ranges);
    std::optional<size_t> last_axis;
    for (size_t i = 0; i < target.size(); ++i)
      if (target_ranges[i].second - target_ranges[i].first > 1) last_axis = i;
    family.axes.clear();
    __int128 count = 1;
    auto source_stride = input_.segments[family.source_region].count;
    auto target_stride = input_.segments[family.target_region].count;
    if (last_axis) {
      for (size_t i = 0; i <= *last_axis; ++i) {
        CHECK(i < source.size() && source[i].loop == target[i].loop)
            << "Unified scoped family loses its original common frame";
        source_stride /= source[i].count;
        target_stride /= target[i].count;
        const Progression a{source_ranges[i].first, source_ranges[i].second, source_steps[i]};
        const Progression b{target_ranges[i].first, target_ranges[i].second, target_steps[i]};
        CHECK(a.Count() == b.Count() && (a.Count() == 1 || a.step == b.step))
            << "Unified scoped family changes its source/consumer coordinate map";
        const auto ss = __int128(source_stride) * a.step, ts = __int128(target_stride) * b.step;
        CHECK(ss <= std::numeric_limits<int64_t>::max() && ts <= std::numeric_limits<int64_t>::max())
            << "Unified scoped family stride overflows";
        family.axes.push_back(
            {target[i].loop, b.Count(), static_cast<int64_t>(ss), static_cast<int64_t>(ts), a.step, b.step});
        count *= b.Count();
      }
    }
    CHECK(count <= std::numeric_limits<int64_t>::max()) << "Unified scoped family count overflows";
    family.count = static_cast<int64_t>(count);
    family.source_domain = std::make_shared<SyncEndpointDomain>(
        SyncEndpointDomain{family.source_domain->frame, std::move(source_ranges), std::move(source_steps)});
    family.target_domain = std::make_shared<SyncEndpointDomain>(
        SyncEndpointDomain{family.target_domain->frame, std::move(target_ranges), std::move(target_steps)});
  }
  void Coalesce(std::vector<PhysicalSyncFamily>& families) {
    // Match only one corresponding original axis at a time. All other endpoint
    // coordinates, cuts and translations are part of the index key. Rebuild the
    // index after a successful pass; no quadratic whole-inventory search is
    // hidden inside a callback, and all candidate comparisons are counted.
    bool changed;
    bool allow_singleton_pairs = false;
    do {
      changed = false;
      ++work_.union_passes;
      std::vector<bool> removed(families.size());
      std::map<std::vector<int64_t>, std::vector<size_t>> groups;
      for (size_t index = 0; index < families.size(); ++index) {
        const auto& f = families[index];
        if (input_.segments[f.source_region].sites[f.source_site].pipe ==
            input_.segments[f.target_region].sites[f.target_site].pipe)
          continue;
        const auto& source = input_.segments[f.source_region].nested_frames;
        const auto& target = input_.segments[f.target_region].nested_frames;
        for (size_t axis = 0;
             axis < std::min(source.size(), target.size()) && source[axis].loop == target[axis].loop;
             ++axis) {
          const auto a = EndpointProgression(*f.source_domain, axis);
          const auto b = EndpointProgression(*f.target_domain, axis);
          if (a.Count() != b.Count() || (a.Count() > 1 && a.step != b.step)) continue;
          std::vector<int64_t> key{static_cast<int64_t>(f.source_region),
                                   static_cast<int64_t>(f.source_site),
                                   static_cast<int64_t>(f.target_region),
                                   static_cast<int64_t>(f.target_site),
                                   static_cast<int64_t>(axis),
                                   b.first - a.first};
          for (const auto* domain : {f.source_domain.get(), f.target_domain.get()}) {
            key.push_back(static_cast<int64_t>(domain->ranges.size()));
            for (size_t i = 0; i < domain->ranges.size(); ++i) {
              if (i == axis) continue;
              const auto p = EndpointProgression(*domain, i);
              key.insert(key.end(), {p.first, p.second, p.Count() == 1 ? 1 : p.step});
            }
          }
          groups[std::move(key)].push_back(index);
        }
      }
      for (const auto& [key, raw_members] : groups) {
        if (raw_members.size() < 2) continue;
        const auto axis = static_cast<size_t>(key[4]);
        auto members = raw_members;
        std::sort(members.begin(), members.end(), [&](size_t a, size_t b) {
          const auto pa = EndpointProgression(*families[a].source_domain, axis);
          const auto pb = EndpointProgression(*families[b].source_domain, axis);
          return std::tie(pa.first, pa.second, a) < std::tie(pb.first, pb.second, b);
        });
        for (size_t x = 0; x < members.size(); ++x) {
          const auto i = members[x];
          if (removed[i]) continue;
          for (size_t y = x + 1; y < members.size(); ++y) {
            const auto j = members[y];
            if (removed[j]) continue;
            // Extend multi-record progressions before forming arbitrary new
            // two-singleton strides so prefix/tail pieces snap onto steady-state
            // stride families first.
            const auto pi = EndpointProgression(*families[i].source_domain, axis);
            const auto pj = EndpointProgression(*families[j].source_domain, axis);
            if (!allow_singleton_pairs && pi.Count() == 1 && pj.Count() == 1) continue;
            // A previous axis group can change an indexed coordinate. Only
            // merge one group per pass so no stale key can justify a union.
            ++work_.union_candidates;
            const auto joined = Union(pi, pj);
            if (!joined) continue;
            auto& f = families[i];
            auto sr = f.source_domain->ranges, tr = f.target_domain->ranges;
            auto ss = f.source_domain->steps, ts = f.target_domain->steps;
            if (ss.empty()) ss.assign(sr.size(), 1);
            if (ts.empty()) ts.assign(tr.size(), 1);
            const auto delta = key[5];
            sr[axis] = {joined->first, joined->second};
            tr[axis] = {joined->first + delta, joined->second + delta};
            ss[axis] = ts[axis] = joined->step;
            auto candidate = f;
            Rebuild(candidate, std::move(sr), std::move(ss), std::move(tr), std::move(ts));
            ++work_.union_reuse_checks;
            if (!can_reuse_(candidate)) {
              ++work_.union_reuse_refusals;
              continue;
            }
            candidate.protocol_uses.insert(candidate.protocol_uses.end(), families[j].protocol_uses.begin(),
                                           families[j].protocol_uses.end());
            f = std::move(candidate);
            removed[j] = true;
            changed = true;
            ++work_.union_merges;
          }
        }
        if (changed) break;
      }
      if (changed) {
        size_t retained = 0;
        for (size_t i = 0; i < families.size(); ++i)
          if (!removed[i]) {
            if (retained != i) families[retained] = std::move(families[i]);
            ++retained;
          }
        families.resize(retained);
        allow_singleton_pairs = false;
      } else if (!allow_singleton_pairs) {
        allow_singleton_pairs = true;
        changed = true;
      }
    } while (changed);
  }
};
}  // namespace
ScopedSyncProtocol::ScopedSyncProtocol(FunctionPtr tokens, CompactSyncInput input,
                                       std::vector<PhysicalSyncFamily> families)
    : tokens_(std::move(tokens)), input_(std::move(input)), families_(std::move(families)) {
  const auto& source_input = input_;
  std::map<std::pair<size_t, size_t>, ScopedProtocolCut> cuts;
  for (size_t i = 0; i < families_.size(); ++i) {
    auto& f = families_[i];
    CHECK(f.ids.empty() && f.protocol_uses.empty()) << "Protocol requires fresh unassigned token uses";
    f.protocol_uses.push_back(i);
    const auto source = std::make_pair(f.source_region, f.source_site);
    const auto target = std::make_pair(f.target_region, f.target_site);
    auto& consume =
        cuts.try_emplace(target, ScopedProtocolCut{target.first, target.second, {}, {}}).first->second;
    consume.before.push_back(i);
    if (source_input.segments[f.source_region].sites[f.source_site].pipe !=
        source_input.segments[f.target_region].sites[f.target_site].pipe)
      cuts.try_emplace(source, ScopedProtocolCut{source.first, source.second, {}, {}})
          .first->second.after.push_back(i);
  }
  for (auto& [key, cut] : cuts) {
    // Existing direct policy: same-pipe barriers precede incoming waits.
    std::stable_partition(cut.before.begin(), cut.before.end(), [&](size_t i) {
      const auto& f = families_[i];
      return source_input.segments[f.source_region].sites[f.source_site].pipe ==
             source_input.segments[f.target_region].sites[f.target_site].pipe;
    });
    cuts_.push_back(std::move(cut));
  }
}
void ScopedSyncProtocol::ValidateFor(const CompactSyncInput& input) const {
  CHECK(&input == &input_) << "Unassigned synchronization protocol belongs to a different input case";
}
std::shared_ptr<const ScopedSyncProtocol> ParseScopedTokenProtocol(
    const FunctionPtr& function, CompactSyncInput input,
    const std::map<const ForStmt*, ForStmtPtr>& stripped_loops, ScopedTokenWork* work) {
  ScopedTokenWork local;
  auto families = Parser(function, input, stripped_loops, work ? *work : local).Run();
  return std::shared_ptr<const ScopedSyncProtocol>(
      new ScopedSyncProtocol(function, std::move(input), std::move(families)));
}
std::vector<PhysicalSyncFamily> NormalizeScopedTokenFamilies(
    const CompactSyncInput& input, std::vector<PhysicalSyncFamily> families,
    const std::function<bool(const PhysicalSyncFamily&)>& can_reuse, ScopedTokenWork* work) {
  ScopedTokenWork local;
  FamilyNormalizer(input, can_reuse, work ? *work : local).Run(families);
  return families;
}
}  // namespace pypto::ir::sync
