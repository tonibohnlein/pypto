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

#include "src/ir/transforms/utils/physical_sync_commands.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_token_names.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace physical {

namespace {
ExprPtr MakeInt(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

std::optional<int64_t> OriginalBoundary(const AxisInfo& axis, int64_t digit) {
  if (axis.kind == AxisKind::LaunchBlock) {
    const __int128 value = __int128(axis.start) + digit;
    if (value < std::numeric_limits<int64_t>::min() || value > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    return static_cast<int64_t>(value);
  }
  return OriginalLoopBoundary(axis.loop, axis.start, axis.step, axis.count, axis.start_expression, digit);
}

ExprPtr FrameDigitExpr(const AxisInfo& axis, const Span& span) {
  if (axis.count == 1) return MakeInt(0, span);
  ExprPtr value = axis.var;
  if (axis.start_expression || axis.start != 0) {
    value = std::make_shared<Sub>(value,
                                  axis.start_expression ? axis.start_expression : MakeInt(axis.start, span),
                                  DataType::INDEX, span);
  }
  if (axis.step != 1) {
    value = std::make_shared<FloorDiv>(value, MakeInt(axis.step, span), DataType::INDEX, span);
  }
  return value;
}

__int128 ExtendedGcd(__int128 a, __int128 b, __int128& x, __int128& y) {
  if (b == 0) {
    x = 1;
    y = 0;
    return a;
  }
  __int128 x1 = 0, y1 = 0;
  __int128 g = ExtendedGcd(b, a % b, x1, y1);
  x = y1;
  y = x1 - (a / b) * y1;
  return g;
}

std::vector<const Var*> PredicateInputs(const ExprPtr& predicate, PhysicalSyncRealization& work,
                                        bool& qualified) {
  CHECK(predicate) << "Missing command predicate";
  const auto type = As<ScalarType>(predicate->GetType());
  CHECK(type && type->dtype_ == DataType::BOOL) << "Command predicate must be Boolean";
  std::set<const Var*> variables;
  std::set<const Expr*> visited;
  const auto visit = [&](const auto& self, const ExprPtr& value) -> void {
    if (!visited.insert(value.get()).second) return;
    ++work.control_expression_visits;
    const auto scalar = As<ScalarType>(value->GetType());
    CHECK(scalar && (scalar->dtype_ == DataType::INDEX || scalar->dtype_ == DataType::BOOL))
        << "Command predicate requires INDEX/BOOL coordinate expressions";
    if (const auto variable = As<Var>(value)) {
      variables.insert(variable.get());
      return;
    }
    if (As<ConstInt>(value) || As<ConstBool>(value)) return;
    const auto binary = As<BinaryExpr>(value);
    CHECK(binary && (As<Add>(value) || As<Sub>(value) || As<Mul>(value) || As<FloorDiv>(value) ||
                     As<FloorMod>(value) || As<Eq>(value) || As<Ne>(value) || As<Lt>(value) ||
                     As<Le>(value) || As<Gt>(value) || As<Ge>(value) || As<And>(value) || As<Or>(value)))
        << "Command predicate is not a pure supported coordinate expression";
    if (As<FloorDiv>(value) || As<FloorMod>(value)) {
      const auto divisor = As<ConstInt>(binary->right_);
      qualified &= divisor && divisor->value_ > 0;
    }
    self(self, binary->left_);
    self(self, binary->right_);
  };
  visit(visit, predicate);
  return {variables.begin(), variables.end()};
}

void CollectExprVars(const ExprPtr& expr, std::set<const Var*>& out, PhysicalSyncRealization& work) {
  if (!expr) return;
  class VarCollector : public IRVisitor {
    std::set<const Var*>& out_;
    PhysicalSyncRealization& work_;

   public:
    VarCollector(std::set<const Var*>& out, PhysicalSyncRealization& work) : out_(out), work_(work) {}
    void VisitExpr(const ExprPtr& value) override {
      ++work_.control_expression_visits;
      IRVisitor::VisitExpr(value);
    }
    void VisitExpr_(const VarPtr& var) override { out_.insert(var.get()); }
  };
  VarCollector(out, work).VisitExpr(expr);
}

bool SameAxisInfo(const AxisInfo& a, const AxisInfo& b) {
  if (a.kind != b.kind || a.count != b.count || a.start != b.start || a.step != b.step) return false;
  if (a.var != b.var && (!a.var || !b.var || !SameControl(a.var, b.var))) return false;
  if (static_cast<bool>(a.start_expression) != static_cast<bool>(b.start_expression)) return false;
  if (a.start_expression && !SameControl(a.start_expression, b.start_expression)) return false;
  return true;
}

size_t HashAxisInfo(const AxisInfo& info) {
  size_t h = static_cast<size_t>(info.kind);
  auto mix = [&](size_t v) { h ^= v + size_t(0x9e3779b9) + (h << 6) + (h >> 2); };
  mix(static_cast<size_t>(info.count));
  mix(static_cast<size_t>(info.start));
  mix(static_cast<size_t>(info.step));
  if (info.var) mix(structural_hash(info.var, false));
  if (info.start_expression) mix(structural_hash(info.start_expression, false));
  return h;
}

}  // namespace

std::optional<int64_t> OriginalLoopBoundary(const ForStmtPtr& loop, int64_t start_val, int64_t step_val,
                                            int64_t count, const ExprPtr& start_expr, int64_t digit) {
  if (!loop || start_expr || step_val <= 0 || count <= 0 || digit < 0 || digit >= count) return std::nullopt;
  const auto start = As<ConstInt>(loop->start_), step = As<ConstInt>(loop->step_);
  if (!start || !step || start->value_ != start_val || step->value_ != step_val) return std::nullopt;
  const auto last = __int128(start_val) + __int128(count - 1) * step_val;
  const auto bound = __int128(start_val) + __int128(digit) * step_val;
  if (last > std::numeric_limits<int64_t>::max() || last < std::numeric_limits<int64_t>::min() ||
      bound > std::numeric_limits<int64_t>::max() || bound < std::numeric_limits<int64_t>::min())
    return std::nullopt;
  return static_cast<int64_t>(bound);
}

// ============================================================================
// GuardDomainArena & Core RadixBoxGuard Algebra
// ============================================================================

AxisId GuardDomainArena::RegisterLaunchBlock(const VarPtr& block_var, std::pair<int64_t, int64_t> bounds) {
  CHECK(block_var) << "LaunchBlock axis requires a coordinate Var";
  const __int128 wide_count = __int128(bounds.second) - __int128(bounds.first) + 1;
  CHECK(wide_count >= 1 && wide_count <= std::numeric_limits<int64_t>::max())
      << "Invalid LaunchBlock domain count";
  const int64_t count = static_cast<int64_t>(wide_count);
  const auto key = std::make_tuple(block_var.get(), static_cast<uint8_t>(AxisKind::LaunchBlock), count,
                                   bounds.first, int64_t{1}, static_cast<const Expr*>(nullptr));
  if (auto it = index_.find(key); it != index_.end()) return it->second;
  const AxisId id = static_cast<AxisId>(axes_.size());
  axes_.push_back(AxisInfo{id, AxisKind::LaunchBlock, block_var, nullptr, count, bounds.first, 1, nullptr});
  index_.emplace(key, id);
  return id;
}

AxisId GuardDomainArena::RegisterLoopFrame(const SyncLoopFrame& frame) {
  CHECK(frame.loop && frame.loop->loop_var_ && frame.count >= 1 && frame.step >= 1)
      << "Invalid LoopFrame for GuardDomainArena";
  const auto key = std::make_tuple(frame.loop->loop_var_.get(), static_cast<uint8_t>(AxisKind::LoopFrame),
                                   frame.count, frame.start, frame.step, frame.start_expression.get());
  if (auto it = index_.find(key); it != index_.end()) return it->second;
  const AxisId id = static_cast<AxisId>(axes_.size());
  axes_.push_back(AxisInfo{id, AxisKind::LoopFrame, frame.loop->loop_var_, frame.loop, frame.count,
                           frame.start, frame.step, frame.start_expression});
  index_.emplace(key, id);
  return id;
}

std::optional<AxisAP> AxisAP::MakeCanonical(int64_t raw_first, int64_t raw_end, int64_t raw_stride,
                                            int64_t axis_count) {
  if (axis_count < 1 || raw_stride < 1) return std::nullopt;
  int64_t lo = std::max<int64_t>(0, raw_first);
  const int64_t hi = std::min<int64_t>(axis_count, raw_end);
  if (lo >= hi) return std::nullopt;
  if (raw_first < lo && raw_stride > 1) {
    const __int128 rem = (__int128(lo) - raw_first) % raw_stride;
    const __int128 shift = rem == 0 ? 0 : (raw_stride - rem);
    if (__int128(lo) + shift >= hi) return std::nullopt;
    lo = static_cast<int64_t>(__int128(lo) + shift);
  }
  const int64_t span = (hi - 1) - lo;
  const int64_t cnt = span / raw_stride + 1;
  if (cnt <= 1) return AxisAP{lo, lo + 1, 1};
  const int64_t last = lo + (cnt - 1) * raw_stride;
  return AxisAP{lo, last + 1, raw_stride};
}

ResidueMask::ResidueMask(int64_t modulus, uint64_t bits) : modulus_(modulus) {
  CHECK(modulus >= 1) << "ResidueMask modulus must be positive";
  if (modulus_ <= 64) {
    inline_bits_ = bits;
  } else {
    ext_bits_.resize(static_cast<size_t>((modulus_ + 63) / 64), 0);
    ext_bits_[0] = bits;
  }
  MaskUnusedBits();
}

void ResidueMask::MaskUnusedBits() {
  if (modulus_ <= 64) {
    if (modulus_ < 64) {
      inline_bits_ &= (uint64_t{1} << modulus_) - 1;
    }
  } else {
    const int64_t rem = modulus_ % 64;
    if (rem != 0 && !ext_bits_.empty()) {
      ext_bits_.back() &= (uint64_t{1} << rem) - 1;
    }
  }
}

bool ResidueMask::Empty() const {
  if (modulus_ <= 64) return inline_bits_ == 0;
  return std::all_of(ext_bits_.begin(), ext_bits_.end(), [](uint64_t w) { return w == 0; });
}

bool ResidueMask::Full() const { return PopCount() == static_cast<size_t>(modulus_); }

size_t ResidueMask::PopCount() const {
  if (modulus_ <= 64) return static_cast<size_t>(__builtin_popcountll(inline_bits_));
  size_t total = 0;
  for (uint64_t w : ext_bits_) total += static_cast<size_t>(__builtin_popcountll(w));
  return total;
}

bool ResidueMask::Test(int64_t r) const {
  if (r < 0 || r >= modulus_) return false;
  if (modulus_ <= 64) return (inline_bits_ >> r) & 1ULL;
  return (ext_bits_[static_cast<size_t>(r / 64)] >> (r % 64)) & 1ULL;
}

void ResidueMask::Set(int64_t r) {
  if (r < 0 || r >= modulus_) return;
  if (modulus_ <= 64) {
    inline_bits_ |= (uint64_t{1} << r);
  } else {
    ext_bits_[static_cast<size_t>(r / 64)] |= (uint64_t{1} << (r % 64));
  }
}

ResidueMask ResidueMask::operator&(const ResidueMask& o) const {
  CHECK(modulus_ == o.modulus_) << "ResidueMask modulus mismatch";
  ResidueMask out(modulus_);
  if (modulus_ <= 64) {
    out.inline_bits_ = inline_bits_ & o.inline_bits_;
  } else {
    for (size_t i = 0; i < ext_bits_.size(); ++i) out.ext_bits_[i] = ext_bits_[i] & o.ext_bits_[i];
  }
  return out;
}

ResidueMask ResidueMask::operator|(const ResidueMask& o) const {
  CHECK(modulus_ == o.modulus_) << "ResidueMask modulus mismatch";
  ResidueMask out(modulus_);
  if (modulus_ <= 64) {
    out.inline_bits_ = inline_bits_ | o.inline_bits_;
  } else {
    for (size_t i = 0; i < ext_bits_.size(); ++i) out.ext_bits_[i] = ext_bits_[i] | o.ext_bits_[i];
  }
  return out;
}

ResidueMask ResidueMask::Complement() const {
  ResidueMask out(modulus_);
  if (modulus_ <= 64) {
    out.inline_bits_ = ~inline_bits_;
  } else {
    for (size_t i = 0; i < ext_bits_.size(); ++i) out.ext_bits_[i] = ~ext_bits_[i];
  }
  out.MaskUnusedBits();
  return out;
}

ResidueMask ResidueMask::Shift(int64_t delta) const {
  const int64_t d = ((delta % modulus_) + modulus_) % modulus_;
  if (d == 0) return *this;
  ResidueMask out(modulus_);
  for (int64_t r = 0; r < modulus_; ++r) {
    if (Test(r)) out.Set((r + d) % modulus_);
  }
  return out;
}

std::optional<int64_t> ResidueMask::SmallestPeriod() const {
  if (modulus_ <= 1 || Empty() || Full()) return std::nullopt;
  for (int64_t d = 1; d < modulus_; ++d) {
    if (modulus_ % d != 0) continue;
    bool periodic = true;
    for (int64_t r = 0; r < modulus_; ++r) {
      if (Test(r) != Test(r % d)) {
        periodic = false;
        break;
      }
    }
    if (periodic) return d;
  }
  return std::nullopt;
}

bool ResidueMask::operator==(const ResidueMask& o) const {
  return modulus_ == o.modulus_ && inline_bits_ == o.inline_bits_ && ext_bits_ == o.ext_bits_;
}

bool ResidueMask::operator<(const ResidueMask& o) const {
  return std::tie(modulus_, inline_bits_, ext_bits_) < std::tie(o.modulus_, o.inline_bits_, o.ext_bits_);
}

size_t ResidueMask::Hash() const {
  size_t h = static_cast<size_t>(modulus_);
  auto mix = [&](size_t v) { h ^= v + size_t(0x9e3779b9) + (h << 6) + (h >> 2); };
  if (modulus_ <= 64) {
    mix(static_cast<size_t>(inline_bits_));
  } else {
    mix(ext_bits_.size());
    for (uint64_t w : ext_bits_) mix(static_cast<size_t>(w));
  }
  return h;
}

AxisAP RadixBox::AxisRange(AxisId axis, const GuardDomainArena& arena) const {
  for (const auto& c : constraints) {
    if (c.axis == axis) return c.ap;
    if (c.axis > axis) break;
  }
  return AxisAP{0, arena.Axis(axis).count, 1};
}

namespace {
bool CanonicalizeBox(RadixBox& box, const GuardDomainArena& arena);
}  // namespace

RadixBoxGuard RadixBoxGuard::True() {
  RadixBoxGuard g;
  g.kind_ = Kind::kBoxes;
  g.boxes_.push_back(RadixBox{});
  return g;
}

RadixBoxGuard RadixBoxGuard::FromBox(std::shared_ptr<const GuardDomainArena> arena,
                                     std::optional<RadixBox> box) {
  if (!box) return False();
  if (arena && !CanonicalizeBox(*box, *arena)) return False();
  if (box->IsUnconstrained()) return True();
  RadixBoxGuard g;
  g.kind_ = Kind::kBoxes;
  g.arena_ = std::move(arena);
  g.boxes_.push_back(std::move(*box));
  return g;
}

RadixBoxGuard RadixBoxGuard::FromBoxes(std::shared_ptr<const GuardDomainArena> arena,
                                       std::vector<RadixBox> boxes) {
  if (boxes.empty()) return False();
  bool disjoint = true;
  if (arena) disjoint = CoalesceBoxes(boxes, *arena);
  if (boxes.empty()) return False();
  if (boxes.size() == 1 && boxes[0].IsUnconstrained()) return True();
  RadixBoxGuard g;
  g.kind_ = Kind::kBoxes;
  g.arena_ = std::move(arena);
  g.boxes_ = std::move(boxes);
  g.disjoint_ = disjoint;
  return g;
}

RadixBoxGuard RadixBoxGuard::OpaqueDiagnostic(ExprPtr expr, std::vector<const Var*> inputs, bool qualified) {
  if (auto known = As<ConstBool>(expr)) {
    return known->value_ ? True() : False();
  }
  RadixBoxGuard g;
  g.kind_ = Kind::kOpaqueExpr;
  g.opaque_ = std::move(expr);
  g.opaque_inputs_ = std::move(inputs);
  g.opaque_qualified_ = qualified;
  return g;
}

bool RadixBoxGuard::IsFirstIterationOf(const VarPtr& loop_var, int64_t lr_start, int64_t lr_step) const {
  if (kind_ != Kind::kBoxes || !arena_ || boxes_.size() != 1) return false;
  const auto& box = boxes_[0];
  if (!box.ranks.empty() || box.constraints.size() != 1) return false;
  const auto& c = box.constraints[0];
  if (!c.ap.IsSingleton()) return false;
  const auto& axis = arena_->Axis(c.axis);
  if (axis.kind != AxisKind::LoopFrame || axis.var != loop_var || axis.start_expression ||
      axis.step != lr_step || axis.step <= 0)
    return false;
  const __int128 offset = __int128(lr_start) - __int128(axis.start);
  if (offset < 0 || offset % axis.step != 0) return false;
  const __int128 d_first = offset / axis.step;
  return d_first == c.ap.first;
}

std::optional<AxisAP> CRTIntersect(const AxisAP& a, const AxisAP& b, int64_t axis_count) {
  const __int128 L = std::max(a.first, b.first);
  const __int128 H = std::min(a.Last(), b.Last());
  if (L > H) return std::nullopt;
  const int64_t g = std::gcd(a.stride, b.stride);
  const __int128 diff = __int128(b.first) - __int128(a.first);
  if (((diff % g) + g) % g != 0) return std::nullopt;
  const __int128 m = b.stride / g;
  __int128 d_base = a.first;
  if (m > 1) {
    const __int128 u = ((a.stride / g) % m + m) % m;
    __int128 inv = 0, dummy = 0;
    ExtendedGcd(u, m, inv, dummy);
    inv = ((inv % m) + m) % m;
    const __int128 rhs = (((diff / g) % m) + m) % m;
    const __int128 x0 = (rhs * inv) % m;
    d_base = __int128(a.first) + x0 * __int128(a.stride);
  }
  const __int128 P = __int128(a.stride) * m;
  __int128 d0 = d_base;
  if (d0 < L) {
    const __int128 k = (L - d0 + P - 1) / P;
    d0 += k * P;
  }
  if (d0 > H) return std::nullopt;
  if (d0 + P > H) {
    return AxisAP{static_cast<int64_t>(d0), static_cast<int64_t>(d0) + 1, 1};
  }
  const __int128 last = d0 + ((H - d0) / P) * P;
  return AxisAP::MakeCanonical(static_cast<int64_t>(d0), static_cast<int64_t>(last) + 1,
                               static_cast<int64_t>(P), axis_count);
}

namespace {

bool FoldSingleConstraintIntoBox(RadixBox& box, AxisId axis, const AxisAP& ap,
                                 const GuardDomainArena& arena) {
  const int64_t count = arena.Axis(axis).count;
  for (auto it = box.constraints.begin(); it != box.constraints.end(); ++it) {
    if (it->axis == axis) {
      auto merged = CRTIntersect(it->ap, ap, count);
      if (!merged) return false;
      if (merged->IsFull(count)) {
        box.constraints.erase(it);
      } else {
        it->ap = *merged;
      }
      return true;
    }
    if (it->axis > axis) {
      if (!ap.IsFull(count)) box.constraints.insert(it, AxisConstraint{axis, ap});
      return true;
    }
  }
  if (!ap.IsFull(count)) box.constraints.push_back(AxisConstraint{axis, ap});
  return true;
}

bool CanonicalizeBox(RadixBox& box, const GuardDomainArena& arena) {
  // Remove full constraints and ensure sorted by axis.
  std::sort(box.constraints.begin(), box.constraints.end(),
            [](const AxisConstraint& a, const AxisConstraint& b) { return a.axis < b.axis; });
  std::vector<AxisConstraint> merged_constraints;
  for (const auto& c : box.constraints) {
    const int64_t cnt = arena.Axis(c.axis).count;
    if (!merged_constraints.empty() && merged_constraints.back().axis == c.axis) {
      auto crt = CRTIntersect(merged_constraints.back().ap, c.ap, cnt);
      if (!crt) return false;
      if (crt->IsFull(cnt))
        merged_constraints.pop_back();
      else
        merged_constraints.back().ap = *crt;
    } else if (!c.ap.IsFull(cnt)) {
      merged_constraints.push_back(c);
    }
  }
  box.constraints = std::move(merged_constraints);

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t ri = 0; ri < box.ranks.size();) {
      auto& rc = box.ranks[ri];
      CHECK(rc.modulus >= 1) << "Invalid RankCongruence modulus";
      rc.base = PositiveMod64(rc.base, rc.modulus);
      if (rc.residues.Empty()) return false;
      if (rc.residues.Full()) {
        box.ranks.erase(box.ranks.begin() + static_cast<std::ptrdiff_t>(ri));
        changed = true;
        continue;
      }
      if (auto period = rc.residues.SmallestPeriod()) {
        ResidueMask reduced(*period);
        for (int64_t r = 0; r < *period; ++r) {
          if (rc.residues.Test(r)) reduced.Set(r);
        }
        rc.modulus = *period;
        rc.base %= rc.modulus;
        rc.residues = std::move(reduced);
        changed = true;
      }
      if (rc.modulus == 1) {
        if (!rc.residues.Test(0)) return false;
        box.ranks.erase(box.ranks.begin() + static_cast<std::ptrdiff_t>(ri));
        changed = true;
        continue;
      }
      std::sort(rc.terms.begin(), rc.terms.end(),
                [](const RankTerm& a, const RankTerm& b) { return a.axis < b.axis; });
      std::vector<RankTerm> kept_terms;
      for (auto term : rc.terms) {
        const AxisAP ap = box.AxisRange(term.axis, arena);
        CHECK((ap.first - term.first) % term.step == 0) << "RankTerm step does not align with AxisAP origin";
        const int64_t p0 = (ap.first - term.first) / term.step;
        term.coeff = PositiveMod64(term.coeff, rc.modulus);
        if (ap.Count() == 1) {
          rc.base = PositiveMod128(__int128(rc.base) + __int128(term.coeff) * p0, rc.modulus);
          changed = true;
          continue;
        }
        CHECK(ap.stride % term.step == 0) << "RankTerm step does not divide AxisAP stride";
        const int64_t delta = ap.stride / term.step;
        if ((__int128(term.coeff) * delta) % rc.modulus == 0) {
          rc.base = PositiveMod128(__int128(rc.base) + __int128(term.coeff) * p0, rc.modulus);
          changed = true;
          continue;
        }
        kept_terms.push_back(term);
      }
      rc.terms = std::move(kept_terms);
      if (rc.terms.empty()) {
        if (!rc.residues.Test(rc.base)) return false;
        box.ranks.erase(box.ranks.begin() + static_cast<std::ptrdiff_t>(ri));
        changed = true;
        continue;
      }
      if (rc.terms.size() == 1) {
        const RankTerm t = rc.terms[0];
        const AxisAP ap = box.AxisRange(t.axis, arena);
        const int64_t n = ap.Count();
        const int64_t p0 = (ap.first - t.first) / t.step;
        const int64_t delta = ap.stride / t.step;
        const int64_t c = static_cast<int64_t>((__int128(t.coeff) * (delta % rc.modulus)) % rc.modulus);
        const int64_t g = std::gcd(c, rc.modulus);
        const int64_t P = rc.modulus / g;
        std::vector<int64_t> hits;
        for (int64_t k0 = 0; k0 < P && k0 < n; ++k0) {
          const __int128 val = __int128(rc.base) + __int128(t.coeff) * (p0 + __int128(k0) * delta);
          const int64_t r = PositiveMod128(val, rc.modulus);
          if (rc.residues.Test(r)) hits.push_back(k0);
        }
        if (hits.empty()) return false;
        std::optional<int64_t> folded_stride;
        int64_t folded_end_k = n;
        if (hits.size() == 1) {
          folded_stride = P;
          const int64_t max_k = hits[0] + ((n - 1 - hits[0]) / P) * P;
          folded_end_k = max_k + 1;
        } else {
          const int64_t dk = hits[1] - hits[0];
          bool uniform = true;
          for (size_t hi = 2; hi < hits.size(); ++hi) {
            if (hits[hi] - hits[hi - 1] != dk) {
              uniform = false;
              break;
            }
          }
          if (uniform && n <= P) {
            folded_stride = dk;
            folded_end_k = hits.back() + 1;
          } else if (uniform && P % dk == 0 && hits.size() == static_cast<size_t>(P / dk)) {
            folded_stride = dk;
            folded_end_k = n;
          }
        }
        if (folded_stride) {
          const int64_t new_first = ap.first + hits[0] * ap.stride;
          const int64_t new_last = ap.first + (folded_end_k - 1) * ap.stride;
          const int64_t axis_cnt = arena.Axis(t.axis).count;
          std::optional<AxisAP> new_ap;
          if (new_first == new_last) {
            new_ap = AxisAP{new_first, new_first + 1, 1};
          } else {
            const __int128 wide_stride = __int128(ap.stride) * (*folded_stride);
            new_ap =
                AxisAP::MakeCanonical(new_first, new_last + 1, static_cast<int64_t>(wide_stride), axis_cnt);
          }
          if (!new_ap || !FoldSingleConstraintIntoBox(box, t.axis, *new_ap, arena)) return false;
          box.ranks.erase(box.ranks.begin() + static_cast<std::ptrdiff_t>(ri));
          changed = true;
          continue;
        } else {
          // Tighten t.axis to the AxisAP hull of active hits even when non-uniform.
          int64_t max_k = hits.front();
          int64_t step_k = 0;
          for (int64_t k0 : hits) {
            const int64_t top_k = k0 + ((n - 1 - k0) / P) * P;
            max_k = std::max(max_k, top_k);
            step_k = std::gcd(step_k, k0 - hits.front());
            if (top_k > k0) step_k = std::gcd(step_k, P);
          }
          if (step_k > 0) {
            const int64_t new_first = ap.first + hits.front() * ap.stride;
            const int64_t new_last = ap.first + max_k * ap.stride;
            const int64_t axis_cnt = arena.Axis(t.axis).count;
            std::optional<AxisAP> hull_ap;
            if (new_first == new_last) {
              hull_ap = AxisAP{new_first, new_first + 1, 1};
            } else {
              const __int128 wide_stride = __int128(ap.stride) * step_k;
              if (wide_stride <= std::numeric_limits<int64_t>::max()) {
                hull_ap = AxisAP::MakeCanonical(new_first, new_last + 1, static_cast<int64_t>(wide_stride),
                                                axis_cnt);
              }
            }
            if (hull_ap && *hull_ap != ap) {
              if (!FoldSingleConstraintIntoBox(box, t.axis, *hull_ap, arena)) return false;
              changed = true;
            }
          }
        }
      }
      ++ri;
    }

    // Merge identical (modulus, terms) ranks in place.
    for (size_t i = 0; i < box.ranks.size(); ++i) {
      for (size_t j = i + 1; j < box.ranks.size();) {
        if (box.ranks[i].modulus == box.ranks[j].modulus && box.ranks[i].terms == box.ranks[j].terms) {
          const int64_t M = box.ranks[i].modulus;
          const int64_t delta = ((box.ranks[j].base - box.ranks[i].base) % M + M) % M;
          box.ranks[i].residues = box.ranks[i].residues & box.ranks[j].residues.Shift(-delta);
          if (box.ranks[i].residues.Empty()) return false;
          box.ranks.erase(box.ranks.begin() + static_cast<std::ptrdiff_t>(j));
          changed = true;
        } else {
          ++j;
        }
      }
    }
  }
  std::sort(box.ranks.begin(), box.ranks.end());
  if (!box.ranks.empty()) {
    std::vector<AxisId> axes;
    for (const auto& c : box.constraints) axes.push_back(c.axis);
    for (const auto& r : box.ranks)
      for (const auto& t : r.terms) axes.push_back(t.axis);
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    const auto card = BoxCardinality(box, axes, arena);
    if (card && *card == 0) return false;
  }
  return true;
}

bool SafeMulU128(unsigned __int128 a, unsigned __int128 b, unsigned __int128& out) {
  return !__builtin_mul_overflow(a, b, &out);
}

bool SafeAddU128(unsigned __int128 a, unsigned __int128 b, unsigned __int128& out) {
  if (b > std::numeric_limits<unsigned __int128>::max() - a) return false;
  out = a + b;
  return true;
}

}  // namespace

std::optional<RadixBox> IntersectBox(const RadixBox& a, const RadixBox& b, const GuardDomainArena& arena) {
  RadixBox out;
  size_t ia = 0, ib = 0;
  while (ia < a.constraints.size() && ib < b.constraints.size()) {
    if (a.constraints[ia].axis < b.constraints[ib].axis) {
      out.constraints.push_back(a.constraints[ia++]);
    } else if (b.constraints[ib].axis < a.constraints[ia].axis) {
      out.constraints.push_back(b.constraints[ib++]);
    } else {
      const AxisId ax = a.constraints[ia].axis;
      auto merged = CRTIntersect(a.constraints[ia].ap, b.constraints[ib].ap, arena.Axis(ax).count);
      if (!merged) return std::nullopt;
      if (!merged->IsFull(arena.Axis(ax).count)) out.constraints.push_back(AxisConstraint{ax, *merged});
      ++ia;
      ++ib;
    }
  }
  while (ia < a.constraints.size()) out.constraints.push_back(a.constraints[ia++]);
  while (ib < b.constraints.size()) out.constraints.push_back(b.constraints[ib++]);
  out.ranks = a.ranks;
  out.ranks.insert(out.ranks.end(), b.ranks.begin(), b.ranks.end());
  if (!CanonicalizeBox(out, arena)) return std::nullopt;
  return out;
}

std::optional<unsigned __int128> BoxCardinality(const RadixBox& box, const std::vector<AxisId>& universe,
                                                const GuardDomainArena& arena) {
  if (box.ranks.size() > 2) return std::nullopt;
  if (box.ranks.empty()) {
    unsigned __int128 total = 1;
    for (AxisId u : universe) {
      const unsigned __int128 c = static_cast<unsigned __int128>(box.AxisRange(u, arena).Count());
      if (!SafeMulU128(total, c, total)) return std::nullopt;
    }
    return total;
  }

  if (box.ranks.size() == 1) {
    const auto& rc = box.ranks[0];
    if (rc.modulus > 64) return std::nullopt;
    const size_t M = static_cast<size_t>(rc.modulus);
    std::vector<unsigned __int128> dp(M, 0), next_dp(M, 0);
    dp[static_cast<size_t>(PositiveMod64(rc.base, rc.modulus))] = 1;
    std::set<AxisId> rank_axes;
    for (const auto& t : rc.terms) {
      rank_axes.insert(t.axis);
      const AxisAP ap = box.AxisRange(t.axis, arena);
      const uint64_t n = static_cast<uint64_t>(ap.Count());
      const int64_t p0 = (ap.first - t.first) / t.step;
      const int64_t delta = ap.stride / t.step;
      const int64_t step_res = static_cast<int64_t>((__int128(t.coeff) * (delta % rc.modulus)) % rc.modulus);
      const int64_t P = rc.modulus / std::gcd(step_res, rc.modulus);
      std::vector<unsigned __int128> freq(M, 0);
      const uint64_t q = n / static_cast<uint64_t>(P);
      const uint64_t rem = n % static_cast<uint64_t>(P);
      for (int64_t k = 0; k < P; ++k) {
        const __int128 val = __int128(t.coeff) * (p0 + __int128(k) * delta);
        const size_t d = static_cast<size_t>(((val % rc.modulus) + rc.modulus) % rc.modulus);
        const unsigned __int128 cnt = q + (static_cast<uint64_t>(k) < rem ? 1 : 0);
        if (!SafeAddU128(freq[d], cnt, freq[d])) return std::nullopt;
      }
      std::fill(next_dp.begin(), next_dp.end(), 0);
      for (size_t r = 0; r < M; ++r) {
        if (dp[r] == 0) continue;
        for (size_t d = 0; d < M; ++d) {
          if (freq[d] == 0) continue;
          unsigned __int128 prod = 0;
          if (!SafeMulU128(dp[r], freq[d], prod)) return std::nullopt;
          const size_t nr = (r + d) % M;
          if (!SafeAddU128(next_dp[nr], prod, next_dp[nr])) return std::nullopt;
        }
      }
      dp.swap(next_dp);
    }
    unsigned __int128 rank_ways = 0;
    for (size_t r = 0; r < M; ++r) {
      if (rc.residues.Test(static_cast<int64_t>(r))) {
        if (!SafeAddU128(rank_ways, dp[r], rank_ways)) return std::nullopt;
      }
    }
    unsigned __int128 total = rank_ways;
    for (AxisId u : universe) {
      if (rank_axes.count(u)) continue;
      const unsigned __int128 c = static_cast<unsigned __int128>(box.AxisRange(u, arena).Count());
      if (!SafeMulU128(total, c, total)) return std::nullopt;
    }
    return total;
  }

  // Exact 2-rank convolution DP over joint residue pairs (r0, r1) when M0 * M1 <= 64.
  const auto& rc0 = box.ranks[0];
  const auto& rc1 = box.ranks[1];
  if (__int128(rc0.modulus) * rc1.modulus > 64) return std::nullopt;
  const size_t M0 = static_cast<size_t>(rc0.modulus);
  const size_t M1 = static_cast<size_t>(rc1.modulus);
  const size_t S = M0 * M1;
  std::vector<unsigned __int128> dp(S, 0), next_dp(S, 0);
  const size_t b0 = static_cast<size_t>(PositiveMod64(rc0.base, rc0.modulus));
  const size_t b1 = static_cast<size_t>(PositiveMod64(rc1.base, rc1.modulus));
  dp[b0 * M1 + b1] = 1;
  std::set<AxisId> rank_axes;
  for (const auto& t : rc0.terms) rank_axes.insert(t.axis);
  for (const auto& t : rc1.terms) rank_axes.insert(t.axis);

  auto find_term = [](const RankCongruence& rc, AxisId ax) -> const RankTerm* {
    for (const auto& t : rc.terms)
      if (t.axis == ax) return &t;
    return nullptr;
  };
  for (AxisId ax : rank_axes) {
    const AxisAP ap = box.AxisRange(ax, arena);
    const uint64_t n = static_cast<uint64_t>(ap.Count());
    const RankTerm* t0 = find_term(rc0, ax);
    const RankTerm* t1 = find_term(rc1, ax);
    int64_t P0 = 1, P1 = 1;
    if (t0) {
      const int64_t d0 = ap.stride / t0->step;
      const int64_t sr0 = static_cast<int64_t>((__int128(t0->coeff) * (d0 % rc0.modulus)) % rc0.modulus);
      P0 = rc0.modulus / std::gcd(sr0, rc0.modulus);
    }
    if (t1) {
      const int64_t d1 = ap.stride / t1->step;
      const int64_t sr1 = static_cast<int64_t>((__int128(t1->coeff) * (d1 % rc1.modulus)) % rc1.modulus);
      P1 = rc1.modulus / std::gcd(sr1, rc1.modulus);
    }
    const int64_t P = std::lcm(P0, P1);
    std::vector<unsigned __int128> freq(S, 0);
    const uint64_t q = n / static_cast<uint64_t>(P);
    const uint64_t rem = n % static_cast<uint64_t>(P);
    for (int64_t k = 0; k < P; ++k) {
      size_t d0 = 0, d1 = 0;
      if (t0) {
        const int64_t p0 = (ap.first - t0->first) / t0->step;
        const int64_t delta = ap.stride / t0->step;
        const __int128 v0 = __int128(t0->coeff) * (p0 + __int128(k) * delta);
        d0 = static_cast<size_t>(((v0 % rc0.modulus) + rc0.modulus) % rc0.modulus);
      }
      if (t1) {
        const int64_t p1 = (ap.first - t1->first) / t1->step;
        const int64_t delta = ap.stride / t1->step;
        const __int128 v1 = __int128(t1->coeff) * (p1 + __int128(k) * delta);
        d1 = static_cast<size_t>(((v1 % rc1.modulus) + rc1.modulus) % rc1.modulus);
      }
      const unsigned __int128 cnt = q + (static_cast<uint64_t>(k) < rem ? 1 : 0);
      if (!SafeAddU128(freq[d0 * M1 + d1], cnt, freq[d0 * M1 + d1])) return std::nullopt;
    }
    std::fill(next_dp.begin(), next_dp.end(), 0);
    for (size_t r0 = 0; r0 < M0; ++r0) {
      for (size_t r1 = 0; r1 < M1; ++r1) {
        const unsigned __int128 cur = dp[r0 * M1 + r1];
        if (cur == 0) continue;
        for (size_t d0 = 0; d0 < M0; ++d0) {
          for (size_t d1 = 0; d1 < M1; ++d1) {
            const unsigned __int128 f = freq[d0 * M1 + d1];
            if (f == 0) continue;
            unsigned __int128 prod = 0;
            if (!SafeMulU128(cur, f, prod)) return std::nullopt;
            const size_t idx = ((r0 + d0) % M0) * M1 + ((r1 + d1) % M1);
            if (!SafeAddU128(next_dp[idx], prod, next_dp[idx])) return std::nullopt;
          }
        }
      }
    }
    dp.swap(next_dp);
  }
  unsigned __int128 total = 0;
  for (size_t r0 = 0; r0 < M0; ++r0) {
    if (!rc0.residues.Test(static_cast<int64_t>(r0))) continue;
    for (size_t r1 = 0; r1 < M1; ++r1) {
      if (!rc1.residues.Test(static_cast<int64_t>(r1))) continue;
      if (!SafeAddU128(total, dp[r0 * M1 + r1], total)) return std::nullopt;
    }
  }
  for (AxisId u : universe) {
    if (rank_axes.count(u)) continue;
    const unsigned __int128 c = static_cast<unsigned __int128>(box.AxisRange(u, arena).Count());
    if (!SafeMulU128(total, c, total)) return std::nullopt;
  }
  return total;
}

namespace {

bool MakeDisjointBoxes(std::vector<RadixBox>& boxes, const GuardDomainArena& arena) {
  std::sort(boxes.begin(), boxes.end());
  boxes.erase(std::unique(boxes.begin(), boxes.end()), boxes.end());
  if (boxes.size() <= 1) return true;
  bool any_overlap = false;
  for (size_t i = 0; i < boxes.size() && !any_overlap; ++i) {
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (IntersectBox(boxes[i], boxes[j], arena)) {
        any_overlap = true;
        break;
      }
    }
  }
  if (!any_overlap) return true;
  std::vector<RadixBox> disjoint;
  for (const auto& b : boxes) {
    std::vector<RadixBox> rem{b};
    for (const auto& d : disjoint) {
      std::vector<RadixBox> next;
      for (const auto& r : rem) {
        auto diff = SubtractBox(r, d, arena);
        if (!diff) return false;
        if (disjoint.size() + next.size() + diff->size() > RadixBoxGuard::kMaxGuardBoxes) return false;
        next.insert(next.end(), std::make_move_iterator(diff->begin()), std::make_move_iterator(diff->end()));
      }
      rem = std::move(next);
      if (rem.empty()) break;
    }
    disjoint.insert(disjoint.end(), std::make_move_iterator(rem.begin()), std::make_move_iterator(rem.end()));
  }
  boxes = std::move(disjoint);
  return true;
}

}  // namespace

bool CoalesceBoxes(std::vector<RadixBox>& boxes, const GuardDomainArena& arena) {
  for (size_t i = 0; i < boxes.size();) {
    if (!CanonicalizeBox(boxes[i], arena))
      boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(i));
    else
      ++i;
  }
  if (boxes.empty()) return true;
  if (std::any_of(boxes.begin(), boxes.end(), [](const RadixBox& b) { return b.IsUnconstrained(); })) {
    boxes = {RadixBox{}};
    return true;
  }
  const bool disjoint_ok = MakeDisjointBoxes(boxes, arena);
  if (!disjoint_ok) {
    std::sort(boxes.begin(), boxes.end());
    boxes.erase(std::unique(boxes.begin(), boxes.end()), boxes.end());
    return false;
  }

  bool changed = true;
  while (changed) {
    changed = false;
    // 1. Pairwise rank-mask or single-axis AP merge.
    for (size_t i = 0; i < boxes.size() && !changed; ++i) {
      for (size_t j = i + 1; j < boxes.size() && !changed; ++j) {
        if (boxes[i].constraints == boxes[j].constraints && boxes[i].ranks.size() == boxes[j].ranks.size() &&
            !boxes[i].ranks.empty()) {
          std::optional<size_t> diff_idx;
          bool compatible = true;
          for (size_t r = 0; r < boxes[i].ranks.size(); ++r) {
            if (boxes[i].ranks[r] == boxes[j].ranks[r]) continue;
            if (diff_idx || boxes[i].ranks[r].modulus != boxes[j].ranks[r].modulus ||
                boxes[i].ranks[r].terms != boxes[j].ranks[r].terms) {
              compatible = false;
              break;
            }
            diff_idx = r;
          }
          if (compatible && diff_idx) {
            auto& ri = boxes[i].ranks[*diff_idx];
            const auto& rj = boxes[j].ranks[*diff_idx];
            const int64_t M = ri.modulus;
            const int64_t delta = ((rj.base - ri.base) % M + M) % M;
            ri.residues = ri.residues | rj.residues.Shift(-delta);
            boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
            if (!CanonicalizeBox(boxes[i], arena))
              boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(i));
            changed = true;
            break;
          }
        }
        if (boxes[i].ranks == boxes[j].ranks) {
          std::optional<AxisId> diff_axis;
          AxisAP ap_i, ap_j;
          size_t ia = 0, ib = 0;
          bool ok = true;
          const auto& ca = boxes[i].constraints;
          const auto& cb = boxes[j].constraints;
          while (ia < ca.size() || ib < cb.size()) {
            if (ia < ca.size() && (ib == cb.size() || ca[ia].axis < cb[ib].axis)) {
              ok = false;
              break;
            }
            if (ib < cb.size() && (ia == ca.size() || cb[ib].axis < ca[ia].axis)) {
              ok = false;
              break;
            }
            if (ca[ia].ap != cb[ib].ap) {
              if (diff_axis) {
                ok = false;
                break;
              }
              diff_axis = ca[ia].axis;
              ap_i = ca[ia].ap;
              ap_j = cb[ib].ap;
            }
            ++ia;
            ++ib;
          }
          if (ok && diff_axis) {
            const int64_t f1 = ap_i.first, l1 = ap_i.Last(), c1 = ap_i.Count(), s1 = ap_i.stride;
            const int64_t f2 = ap_j.first, l2 = ap_j.Last(), c2 = ap_j.Count(), s2 = ap_j.stride;
            const int64_t first = std::min(f1, f2);
            const int64_t last = std::max(l1, l2);
            const int64_t diff = std::abs(f1 - f2);
            const int64_t s = std::gcd(c1 > 1 ? s1 : 0, std::gcd(c2 > 1 ? s2 : 0, diff));
            const int64_t axis_cnt = arena.Axis(*diff_axis).count;
            if (!CRTIntersect(ap_i, ap_j, axis_cnt).has_value() && s > 0 &&
                __int128(last - first) / s + 1 == __int128(c1) + c2) {
              auto merged = AxisAP::MakeCanonical(first, last + 1, s, axis_cnt);
              if (merged) {
                for (auto it = boxes[i].constraints.begin(); it != boxes[i].constraints.end(); ++it) {
                  if (it->axis == *diff_axis) {
                    if (merged->IsFull(axis_cnt))
                      boxes[i].constraints.erase(it);
                    else
                      it->ap = *merged;
                    break;
                  }
                }
                boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
                CanonicalizeBox(boxes[i], arena);
                changed = true;
                break;
              }
            }
          }
        }
      }
    }
    if (changed) continue;

    // 2. Multi-way periodic hull collapse (>= 3 boxes differing on one axis).
    if (boxes.size() >= 3) {
      std::set<AxisId> candidate_axes;
      for (const auto& b : boxes)
        for (const auto& c : b.constraints) candidate_axes.insert(c.axis);
      for (AxisId ax : candidate_axes) {
        std::vector<size_t> members;
        std::vector<AxisConstraint> sig_constraints;
        std::vector<RankCongruence> sig_ranks;
        for (size_t i = 0; i < boxes.size() && !changed; ++i) {
          bool has_ax = false;
          for (const auto& c : boxes[i].constraints) {
            if (c.axis == ax) {
              has_ax = true;
              break;
            }
          }
          if (!has_ax) continue;
          sig_constraints.clear();
          for (const auto& c : boxes[i].constraints)
            if (c.axis != ax) sig_constraints.push_back(c);
          sig_ranks = boxes[i].ranks;
          members = {i};
          for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (boxes[j].ranks != sig_ranks) continue;
            bool j_has_ax = false;
            std::vector<AxisConstraint> j_other;
            for (const auto& c : boxes[j].constraints) {
              if (c.axis == ax)
                j_has_ax = true;
              else
                j_other.push_back(c);
            }
            if (j_has_ax && j_other == sig_constraints) members.push_back(j);
          }
          if (members.size() < 3) continue;
          int64_t f = std::numeric_limits<int64_t>::max();
          int64_t l = std::numeric_limits<int64_t>::min();
          __int128 sum_cnt = 0;
          std::vector<AxisAP> aps;
          aps.reserve(members.size());
          for (size_t idx : members) {
            const AxisAP ap = boxes[idx].AxisRange(ax, arena);
            aps.push_back(ap);
            f = std::min(f, ap.first);
            l = std::max(l, ap.Last());
            sum_cnt += ap.Count();
          }
          int64_t g = 0;
          for (const auto& ap : aps) {
            g = std::gcd(g, ap.Count() > 1 ? ap.stride : 0);
            g = std::gcd(g, ap.first - f);
          }
          const int64_t axis_cnt = arena.Axis(ax).count;
          bool aps_disjoint = true;
          for (size_t u = 0; u < aps.size() && aps_disjoint; ++u) {
            for (size_t v = u + 1; v < aps.size(); ++v) {
              if (CRTIntersect(aps[u], aps[v], axis_cnt).has_value()) {
                aps_disjoint = false;
                break;
              }
            }
          }
          if (aps_disjoint && g > 0 && __int128(l - f) / g + 1 == sum_cnt) {
            auto merged = AxisAP::MakeCanonical(f, l + 1, g, axis_cnt);
            if (merged) {
              for (auto it = boxes[members[0]].constraints.begin(); it != boxes[members[0]].constraints.end();
                   ++it) {
                if (it->axis == ax) {
                  if (merged->IsFull(axis_cnt))
                    boxes[members[0]].constraints.erase(it);
                  else
                    it->ap = *merged;
                  break;
                }
              }
              for (size_t k = members.size(); k-- > 1;) {
                boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(members[k]));
              }
              CanonicalizeBox(boxes[members[0]], arena);
              changed = true;
              break;
            }
          }
        }
        if (changed) break;
      }
    }
  }

  // 3. Global tautology collapse.
  if (!(boxes.size() == 1 && boxes[0].IsUnconstrained())) {
    std::vector<AxisId> universe;
    for (const auto& b : boxes) {
      for (const auto& c : b.constraints) universe.push_back(c.axis);
      for (const auto& r : b.ranks)
        for (const auto& t : r.terms) universe.push_back(t.axis);
    }
    std::sort(universe.begin(), universe.end());
    universe.erase(std::unique(universe.begin(), universe.end()), universe.end());
    unsigned __int128 total_domain = 1;
    bool overflow = false;
    for (AxisId u : universe) {
      if (!SafeMulU128(total_domain, static_cast<unsigned __int128>(arena.Axis(u).count), total_domain)) {
        overflow = true;
        break;
      }
    }
    if (!overflow) {
      unsigned __int128 sum = 0;
      bool exact = true;
      for (const auto& b : boxes) {
        const auto c = BoxCardinality(b, universe, arena);
        if (!c || !SafeAddU128(sum, *c, sum)) {
          exact = false;
          break;
        }
      }
      if (exact && sum == total_domain) {
        boxes = {RadixBox{}};
        return true;
      }
    }
  }
  std::sort(boxes.begin(), boxes.end());
  return true;
}

std::optional<std::vector<RadixBox>> SubtractBox(const RadixBox& s, const RadixBox& g,
                                                 const GuardDomainArena& arena) {
  auto g_eff = IntersectBox(s, g, arena);
  if (!g_eff) return std::vector<RadixBox>{s};
  std::vector<RadixBox> out;
  RadixBox cur = s;
  std::vector<AxisId> axes;
  for (const auto& c : cur.constraints) axes.push_back(c.axis);
  for (const auto& c : g_eff->constraints) axes.push_back(c.axis);
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

  for (AxisId ax : axes) {
    const AxisAP s_ap = cur.AxisRange(ax, arena);
    const AxisAP g_ap = g_eff->AxisRange(ax, arena);
    if (s_ap == g_ap) continue;
    const int64_t axis_cnt = arena.Axis(ax).count;
    const auto eff_ap = CRTIntersect(s_ap, g_ap, axis_cnt);
    if (!eff_ap.has_value()) {
      out.push_back(std::move(cur));
      return out;
    }
    if (s_ap == *eff_ap) continue;
    std::vector<AxisAP> diff_aps;
    if (auto pre = AxisAP::MakeCanonical(s_ap.first, eff_ap->first, s_ap.stride, axis_cnt))
      diff_aps.push_back(*pre);
    if (eff_ap->Last() < s_ap.Last()) {
      const __int128 suf_first = __int128(eff_ap->Last()) + s_ap.stride;
      if (suf_first < s_ap.end) {
        if (auto suf =
                AxisAP::MakeCanonical(static_cast<int64_t>(suf_first), s_ap.end, s_ap.stride, axis_cnt))
          diff_aps.push_back(*suf);
      }
    }
    if (eff_ap->Count() >= 2 && eff_ap->stride > s_ap.stride) {
      const int64_t m = eff_ap->stride / s_ap.stride;
      if (out.size() + diff_aps.size() + static_cast<size_t>(m - 1) > RadixBoxGuard::kMaxGuardBoxes)
        return std::nullopt;
      for (int64_t r = 1; r < m; ++r) {
        const __int128 mid_first = __int128(eff_ap->first) + __int128(r) * s_ap.stride;
        if (mid_first < eff_ap->Last()) {
          if (auto mid = AxisAP::MakeCanonical(static_cast<int64_t>(mid_first), eff_ap->Last(),
                                               eff_ap->stride, axis_cnt))
            diff_aps.push_back(*mid);
        }
      }
    }
    if (out.size() + diff_aps.size() > RadixBoxGuard::kMaxGuardBoxes) return std::nullopt;
    for (const auto& p_ap : diff_aps) {
      RadixBox piece = cur;
      if (FoldSingleConstraintIntoBox(piece, ax, p_ap, arena) && CanonicalizeBox(piece, arena))
        out.push_back(std::move(piece));
    }
    if (!FoldSingleConstraintIntoBox(cur, ax, *eff_ap, arena) || !CanonicalizeBox(cur, arena)) {
      return out;
    }
  }

  for (const auto& g_rc : g_eff->ranks) {
    bool matched = false;
    for (auto& c_rc : cur.ranks) {
      if (c_rc.modulus == g_rc.modulus && c_rc.terms == g_rc.terms) {
        const int64_t M = c_rc.modulus;
        const int64_t delta = ((g_rc.base - c_rc.base) % M + M) % M;
        const ResidueMask g_shifted = g_rc.residues.Shift(-delta);
        const ResidueMask rem_mask = c_rc.residues & g_shifted.Complement();
        if (!rem_mask.Empty()) {
          if (out.size() + 1 > RadixBoxGuard::kMaxGuardBoxes) return std::nullopt;
          RadixBox piece = cur;
          for (auto& pr : piece.ranks) {
            if (pr.modulus == c_rc.modulus && pr.terms == c_rc.terms) {
              pr.residues = rem_mask;
              break;
            }
          }
          if (CanonicalizeBox(piece, arena)) out.push_back(std::move(piece));
        }
        c_rc.residues = c_rc.residues & g_shifted;
        matched = true;
        break;
      }
    }
    if (!matched) {
      const ResidueMask comp_mask = g_rc.residues.Complement();
      if (!comp_mask.Empty()) {
        if (out.size() + 1 > RadixBoxGuard::kMaxGuardBoxes) return std::nullopt;
        RadixBox piece = cur;
        piece.ranks.push_back(RankCongruence{g_rc.modulus, g_rc.base, g_rc.terms, comp_mask});
        if (CanonicalizeBox(piece, arena)) out.push_back(std::move(piece));
      }
      cur.ranks.push_back(g_rc);
      if (!CanonicalizeBox(cur, arena)) return out;
    }
  }
  return out;
}

RadixBoxGuard IntersectGuards(const RadixBoxGuard& a, const RadixBoxGuard& b) {
  if (a.IsEmpty() || b.IsEmpty()) return RadixBoxGuard::False();
  if (a.IsTautology()) return b;
  if (b.IsTautology()) return a;
  if (a.IsOpaque() || b.IsOpaque()) {
    if (a.IsOpaque() && b.IsOpaque() && SameControl(a.OpaqueExpr(), b.OpaqueExpr())) return a;
    return RadixBoxGuard::False();
  }
  const auto arena = a.Arena() ? a.Arena() : b.Arena();
  CHECK(arena) << "Missing GuardDomainArena for non-trivial RadixBoxGuard";
  CHECK(!a.Arena() || !b.Arena() || a.Arena() == b.Arena())
      << "Cannot intersect RadixBoxGuards from distinct arenas";
  std::vector<RadixBox> boxes;
  for (const auto& ba : a.Boxes()) {
    for (const auto& bb : b.Boxes()) {
      if (auto inter = IntersectBox(ba, bb, *arena)) boxes.push_back(std::move(*inter));
    }
  }
  return RadixBoxGuard::FromBoxes(arena, std::move(boxes));
}

std::optional<RadixBoxGuard> UnionGuards(const RadixBoxGuard& a, const RadixBoxGuard& b) {
  if (a.IsEmpty()) return b;
  if (b.IsEmpty()) return a;
  if (a.IsTautology() || b.IsTautology()) return RadixBoxGuard::True();
  if (a.IsOpaque() || b.IsOpaque()) return std::nullopt;
  const auto arena = a.Arena() ? a.Arena() : b.Arena();
  CHECK(!a.Arena() || !b.Arena() || a.Arena() == b.Arena())
      << "Cannot union RadixBoxGuards from distinct arenas";
  // Ensure disjointness by subtracting A from B before concatenating.
  auto b_minus_a = SubtractGuards(b, a);
  if (!b_minus_a) return std::nullopt;
  std::vector<RadixBox> boxes = a.Boxes();
  if (!b_minus_a->IsEmpty()) {
    boxes.insert(boxes.end(), b_minus_a->Boxes().begin(), b_minus_a->Boxes().end());
  }
  if (boxes.size() > RadixBoxGuard::kMaxGuardBoxes) return std::nullopt;
  return RadixBoxGuard::FromBoxes(arena, std::move(boxes));
}

std::optional<RadixBoxGuard> SubtractGuards(const RadixBoxGuard& s, const RadixBoxGuard& g) {
  if (s.IsEmpty()) return RadixBoxGuard::False();
  if (g.IsEmpty()) return s;
  if (g.IsTautology()) return RadixBoxGuard::False();
  if (s.IsOpaque() || g.IsOpaque()) return std::nullopt;
  const auto arena = s.Arena() ? s.Arena() : g.Arena();
  CHECK(arena) << "Missing GuardDomainArena in SubtractGuards";
  std::vector<RadixBox> rem = s.Boxes();
  for (const auto& bg : g.Boxes()) {
    std::vector<RadixBox> next;
    for (const auto& r : rem) {
      auto diff = SubtractBox(r, bg, *arena);
      if (!diff) return std::nullopt;
      if (next.size() + diff->size() > RadixBoxGuard::kMaxGuardBoxes) return std::nullopt;
      next.insert(next.end(), std::make_move_iterator(diff->begin()), std::make_move_iterator(diff->end()));
    }
    rem = std::move(next);
    if (rem.empty()) break;
  }
  return RadixBoxGuard::FromBoxes(arena, std::move(rem));
}

std::vector<AxisId> ReferencedAxes(const RadixBoxGuard& a, const RadixBoxGuard& b, const RadixBoxGuard& c) {
  std::vector<AxisId> axes;
  auto collect = [&](const RadixBoxGuard& g) {
    if (g.IsEmpty() || g.IsOpaque()) return;
    for (const auto& box : g.Boxes()) {
      for (const auto& con : box.constraints) axes.push_back(con.axis);
      for (const auto& r : box.ranks)
        for (const auto& t : r.terms) axes.push_back(t.axis);
    }
  };
  collect(a);
  collect(b);
  collect(c);
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
  return axes;
}

std::optional<unsigned __int128> GuardCardinality(const RadixBoxGuard& g,
                                                  const std::vector<AxisId>& universe) {
  if (g.IsEmpty()) return unsigned{0};
  if (g.IsOpaque() || !g.IsDisjoint()) return std::nullopt;
  if (g.IsTautology()) {
    if (!g.Arena()) {
      if (universe.empty()) return unsigned{1};
      return std::nullopt;
    }
  }
  const auto& arena = g.Arena();
  if (!arena) return universe.empty() ? std::optional<unsigned __int128>(1) : std::nullopt;
  unsigned __int128 total = 0;
  for (const auto& box : g.Boxes()) {
    const auto card = BoxCardinality(box, universe, *arena);
    if (!card || !SafeAddU128(total, *card, total)) return std::nullopt;
  }
  return total;
}

namespace {
std::shared_ptr<const GuardDomainArena> SharedArena(const RadixBoxGuard& a, const RadixBoxGuard& b,
                                                    const RadixBoxGuard& c = {}) {
  if (a.Arena()) return a.Arena();
  if (b.Arena()) return b.Arena();
  return c.Arena();
}

std::optional<unsigned __int128> GuardCardinalityWithArena(const RadixBoxGuard& g,
                                                           const std::vector<AxisId>& universe,
                                                           const GuardDomainArena* arena) {
  if (g.IsEmpty()) return unsigned{0};
  if (g.IsOpaque() || !g.IsDisjoint()) return std::nullopt;
  if (!arena) return universe.empty() ? std::optional<unsigned __int128>(1) : std::nullopt;
  unsigned __int128 total = 0;
  for (const auto& box : g.Boxes()) {
    const auto card = BoxCardinality(box, universe, *arena);
    if (!card || !SafeAddU128(total, *card, total)) return std::nullopt;
  }
  return total;
}
}  // namespace

bool AreDisjointGuardsInScope(const RadixBoxGuard& a, const RadixBoxGuard& b, const RadixBoxGuard& scope) {
  if (a.IsOpaque() || b.IsOpaque() || scope.IsOpaque()) return false;
  const auto a_eff = IntersectGuards(scope, a);
  const auto b_eff = IntersectGuards(scope, b);
  return IntersectGuards(a_eff, b_eff).IsEmpty();
}

bool AreComplementaryGuardsInScope(const RadixBoxGuard& a, const RadixBoxGuard& b,
                                   const RadixBoxGuard& scope) {
  if (a.IsOpaque() || b.IsOpaque() || scope.IsOpaque()) return false;
  const auto arena = SharedArena(a, b, scope);
  const auto a_eff = IntersectGuards(scope, a);
  const auto b_eff = IntersectGuards(scope, b);
  if (!IntersectGuards(a_eff, b_eff).IsEmpty()) return false;
  const auto universe = ReferencedAxes(scope, a_eff, b_eff);
  const auto ca = GuardCardinalityWithArena(a_eff, universe, arena.get());
  const auto cb = GuardCardinalityWithArena(b_eff, universe, arena.get());
  const auto cs = GuardCardinalityWithArena(scope, universe, arena.get());
  if (!ca || !cb || !cs) return false;
  unsigned __int128 sum = 0;
  return SafeAddU128(*ca, *cb, sum) && sum == *cs;
}

bool EquivalentGuardsInScope(const RadixBoxGuard& a, const RadixBoxGuard& b, const RadixBoxGuard& scope) {
  if (a.IsOpaque() || b.IsOpaque() || scope.IsOpaque()) {
    return a.IsOpaque() && b.IsOpaque() && SameControl(a.OpaqueExpr(), b.OpaqueExpr());
  }
  const auto arena = SharedArena(a, b, scope);
  const auto a_eff = IntersectGuards(scope, a);
  const auto b_eff = IntersectGuards(scope, b);
  if (a_eff.IsEmpty() && b_eff.IsEmpty()) return true;
  if (a_eff.IsEmpty() != b_eff.IsEmpty()) return false;
  if (a_eff.Boxes() == b_eff.Boxes()) return true;
  const auto ab = IntersectGuards(a_eff, b_eff);
  const auto universe = ReferencedAxes(scope, a_eff, b_eff);
  const auto ca = GuardCardinalityWithArena(a_eff, universe, arena.get());
  const auto cb = GuardCardinalityWithArena(b_eff, universe, arena.get());
  const auto cab = GuardCardinalityWithArena(ab, universe, arena.get());
  return ca && cb && cab && *ca == *cb && *ca == *cab;
}

bool SameGuard(const RadixBoxGuard& a, const RadixBoxGuard& b) {
  if (a.IsEmpty() != b.IsEmpty() || a.IsTautology() != b.IsTautology() || a.IsOpaque() != b.IsOpaque())
    return false;
  if (a.IsEmpty() || a.IsTautology()) return true;
  if (a.IsOpaque()) return SameControl(a.OpaqueExpr(), b.OpaqueExpr());
  if (a.Arena() == b.Arena()) return a.Boxes() == b.Boxes();
  if (!a.Arena() || !b.Arena() || a.Boxes().size() != b.Boxes().size()) return false;
  for (size_t i = 0; i < a.Boxes().size(); ++i) {
    const auto& ba = a.Boxes()[i];
    const auto& bb = b.Boxes()[i];
    if (ba.constraints.size() != bb.constraints.size() || ba.ranks.size() != bb.ranks.size()) return false;
    for (size_t c = 0; c < ba.constraints.size(); ++c) {
      if (ba.constraints[c].ap != bb.constraints[c].ap) return false;
      if (!SameAxisInfo(a.Arena()->Axis(ba.constraints[c].axis), b.Arena()->Axis(bb.constraints[c].axis)))
        return false;
    }
    for (size_t r = 0; r < ba.ranks.size(); ++r) {
      const auto& ra = ba.ranks[r];
      const auto& rb = bb.ranks[r];
      if (ra.modulus != rb.modulus || ra.base != rb.base || ra.residues != rb.residues ||
          ra.terms.size() != rb.terms.size())
        return false;
      for (size_t t = 0; t < ra.terms.size(); ++t) {
        if (ra.terms[t].first != rb.terms[t].first || ra.terms[t].step != rb.terms[t].step ||
            ra.terms[t].coeff != rb.terms[t].coeff)
          return false;
        if (!SameAxisInfo(a.Arena()->Axis(ra.terms[t].axis), b.Arena()->Axis(rb.terms[t].axis))) return false;
      }
    }
  }
  return true;
}

namespace {

std::vector<RadixBox> RelaxBoxesForExpr(const RadixBoxGuard& g_eff, const RadixBoxGuard& scope,
                                        const GuardDomainArena& arena, const std::vector<AxisId>& universe) {
  const std::shared_ptr<const GuardDomainArena> arena_ref(&arena, [](const void*) {});
  std::vector<RadixBox> out_boxes = g_eff.Boxes();
  for (auto& box : out_boxes) {
    // 1. Drop implied RankCongruences.
    for (size_t ri = box.ranks.size(); ri-- > 0;) {
      RadixBox candidate = box;
      candidate.ranks.erase(candidate.ranks.begin() + static_cast<std::ptrdiff_t>(ri));
      const auto i_guard = IntersectGuards(scope, RadixBoxGuard::FromBox(arena_ref, candidate));
      const auto ci = GuardCardinalityWithArena(i_guard, universe, &arena);
      const auto cig = GuardCardinalityWithArena(IntersectGuards(i_guard, g_eff), universe, &arena);
      if (ci && cig && *ci == *cig) box = std::move(candidate);
    }
    // 2. Drop implied AxisConstraints innermost-to-outermost.
    for (size_t ci_idx = box.constraints.size(); ci_idx-- > 0;) {
      const AxisId ax = box.constraints[ci_idx].axis;
      bool used_by_rank = false;
      for (const auto& r : box.ranks) {
        for (const auto& t : r.terms) {
          if (t.axis == ax) {
            used_by_rank = true;
            break;
          }
        }
      }
      if (used_by_rank) continue;
      RadixBox candidate = box;
      candidate.constraints.erase(candidate.constraints.begin() + static_cast<std::ptrdiff_t>(ci_idx));
      const auto i_guard = IntersectGuards(scope, RadixBoxGuard::FromBox(arena_ref, candidate));
      const auto ci = GuardCardinalityWithArena(i_guard, universe, &arena);
      const auto cig = GuardCardinalityWithArena(IntersectGuards(i_guard, g_eff), universe, &arena);
      if (ci && cig && *ci == *cig) box = std::move(candidate);
    }
  }
  std::sort(out_boxes.begin(), out_boxes.end());
  out_boxes.erase(std::unique(out_boxes.begin(), out_boxes.end()), out_boxes.end());
  return out_boxes;
}

}  // namespace

RadixBoxGuard SimplifyGuardGiven(const RadixBoxGuard& g, const RadixBoxGuard& scope) {
  if (g.IsEmpty() || g.IsOpaque() || scope.IsOpaque()) return g;
  const auto g_eff = IntersectGuards(scope, g);
  if (g_eff.IsEmpty() || g_eff.IsTautology()) return g_eff;
  const auto arena = SharedArena(g_eff, scope);
  if (!arena) return g_eff;
  const auto universe = ReferencedAxes(scope, g_eff);
  const auto cs = GuardCardinalityWithArena(scope, universe, arena.get());
  const auto cg = GuardCardinalityWithArena(g_eff, universe, arena.get());
  if (!cs || !cg) return g_eff;
  if (*cg == *cs) return RadixBoxGuard::True();

  std::vector<RadixBox> out_boxes = RelaxBoxesForExpr(g_eff, scope, *arena, universe);
  if (MakeDisjointBoxes(out_boxes, *arena)) {
    return RadixBoxGuard::FromBoxes(arena, std::move(out_boxes));
  }
  return g_eff;
}

RadixBoxGuard RestrictGuardAxis(const RadixBoxGuard& g, AxisId axis, const AxisAP& range,
                                const std::shared_ptr<const GuardDomainArena>& arena) {
  if (g.IsEmpty() || g.IsTautology() || g.IsOpaque()) return g;
  RadixBox scope_box;
  if (!range.IsFull(arena->Axis(axis).count)) {
    scope_box.constraints.push_back(AxisConstraint{axis, range});
  }
  const auto scope = RadixBoxGuard::FromBox(arena, scope_box);
  return SimplifyGuardGiven(g, scope);
}

namespace {

struct OuterAxisHull {
  int64_t first = 0;
  int64_t end = 1;
  int64_t stride = 0;
};

OuterAxisHull EffectiveOuterRange(const RadixBoxGuard& scope, const RadixBox& known_box, AxisId axis,
                                  const GuardDomainArena& arena) {
  const int64_t axis_cnt = arena.Axis(axis).count;
  const auto scoped = IntersectGuards(
      scope,
      RadixBoxGuard::FromBox(std::shared_ptr<const GuardDomainArena>(&arena, [](const void*) {}), known_box));
  if (scoped.IsEmpty()) return OuterAxisHull{0, axis_cnt, 1};
  int64_t s_first = std::numeric_limits<int64_t>::max();
  int64_t s_end = std::numeric_limits<int64_t>::min();
  int64_t s_stride = 0;
  for (const auto& box : scoped.Boxes()) {
    const AxisAP ap = box.AxisRange(axis, arena);
    s_first = std::min(s_first, ap.first);
    s_end = std::max(s_end, ap.end);
    s_stride = std::gcd(s_stride, ap.Count() > 1 ? ap.stride : 0);
  }
  for (const auto& box : scoped.Boxes()) {
    const AxisAP ap = box.AxisRange(axis, arena);
    s_stride = std::gcd(s_stride, ap.first - s_first);
  }
  return OuterAxisHull{s_first, s_end, s_stride};
}

std::vector<ExprPtr> LowerAxisConstraint(const AxisConstraint& c, const OuterAxisHull& outer,
                                         const GuardDomainArena& arena, const Span& span) {
  const auto& axis = arena.Axis(c.axis);
  const AxisAP& ap = c.ap;
  if (axis.kind == AxisKind::LaunchBlock) {
    if (ap.IsSingleton()) {
      if (outer.first == ap.first && outer.end == ap.end) return {};
      return {std::make_shared<Eq>(axis.var, MakeInt(axis.start + ap.first, span), DataType::BOOL, span)};
    }
    const bool need_mod = (ap.stride > 1) && !(outer.stride > 0 && outer.stride % ap.stride == 0 &&
                                               (outer.first - ap.first) % ap.stride == 0);
    const bool need_ge = (ap.first - ap.stride >= outer.first);
    const bool need_lt = (ap.Last() + ap.stride < outer.end);
    ExprPtr pred;
    auto push_and = [&](const ExprPtr& e) {
      pred = pred ? std::make_shared<And>(pred, e, DataType::BOOL, span) : e;
    };
    if (need_ge) {
      push_and(std::make_shared<Ge>(axis.var, MakeInt(axis.start + ap.first, span), DataType::BOOL, span));
    }
    if (need_lt) {
      push_and(
          std::make_shared<Lt>(axis.var, MakeInt(axis.start + ap.Last() + 1, span), DataType::BOOL, span));
    }
    if (need_mod) {
      auto mod = std::make_shared<FloorMod>(FrameDigitExpr(axis, span), MakeInt(ap.stride, span),
                                            DataType::INDEX, span);
      push_and(std::make_shared<Eq>(mod, MakeInt(ap.first % ap.stride, span), DataType::BOOL, span));
    }
    if (!pred) return {};
    return {pred};
  }

  // LoopFrame lowering.
  if (ap.IsSingleton()) {
    if (outer.first == ap.first && outer.end == ap.end) return {};
    const auto ob = OriginalBoundary(axis, ap.first);
    const ExprPtr val = ob ? ExprPtr(axis.var) : FrameDigitExpr(axis, span);
    const ExprPtr rhs = MakeInt(ob ? *ob : ap.first, span);
    return {std::make_shared<Eq>(val, rhs, DataType::BOOL, span)};
  }

  const bool need_mod = (ap.stride > 1) && !(outer.stride > 0 && outer.stride % ap.stride == 0 &&
                                             (outer.first - ap.first) % ap.stride == 0);
  const bool need_ge = (ap.first - ap.stride >= outer.first);
  const bool need_lt = (ap.Last() + ap.stride < outer.end);

  std::vector<ExprPtr> result;
  if (need_mod) {
    auto mod = std::make_shared<FloorMod>(FrameDigitExpr(axis, span), MakeInt(ap.stride, span),
                                          DataType::INDEX, span);
    result.push_back(std::make_shared<Eq>(mod, MakeInt(ap.first % ap.stride, span), DataType::BOOL, span));
  }
  const auto ge_orig = need_ge ? OriginalBoundary(axis, ap.first) : std::nullopt;
  const auto lt_orig = need_lt ? OriginalBoundary(axis, ap.Last() + 1) : std::nullopt;
  const bool use_orig = (!need_ge || ge_orig.has_value()) && (!need_lt || lt_orig.has_value());
  if (need_ge || need_lt) {
    const ExprPtr val = use_orig ? ExprPtr(axis.var) : FrameDigitExpr(axis, span);
    if (need_ge) {
      result.push_back(
          std::make_shared<Ge>(val, MakeInt(use_orig ? *ge_orig : ap.first, span), DataType::BOOL, span));
    }
    if (need_lt) {
      result.push_back(std::make_shared<Lt>(val, MakeInt(use_orig ? *lt_orig : ap.Last() + 1, span),
                                            DataType::BOOL, span));
    }
  }
  return result;
}

ExprPtr LowerRankCongruenceExpr(const RankCongruence& rc, const GuardDomainArena& arena, const Span& span) {
  ExprPtr rank_expr;
  for (const auto& t : rc.terms) {
    const auto& ax = arena.Axis(t.axis);
    ExprPtr digit = FrameDigitExpr(ax, span);
    if (t.first != 0) digit = std::make_shared<Sub>(digit, MakeInt(t.first, span), DataType::INDEX, span);
    if (t.step != 1) digit = std::make_shared<FloorDiv>(digit, MakeInt(t.step, span), DataType::INDEX, span);
    digit = std::make_shared<FloorMod>(digit, MakeInt(rc.modulus, span), DataType::INDEX, span);
    if (t.coeff != 1) {
      digit = std::make_shared<FloorMod>(
          std::make_shared<Mul>(digit, MakeInt(t.coeff, span), DataType::INDEX, span),
          MakeInt(rc.modulus, span), DataType::INDEX, span);
    }
    if (rank_expr) {
      rank_expr = std::make_shared<FloorMod>(std::make_shared<Add>(rank_expr, digit, DataType::INDEX, span),
                                             MakeInt(rc.modulus, span), DataType::INDEX, span);
    } else {
      rank_expr = digit;
    }
  }
  if (rc.base != 0) {
    rank_expr = std::make_shared<FloorMod>(
        std::make_shared<Add>(MakeInt(rc.base, span), rank_expr, DataType::INDEX, span),
        MakeInt(rc.modulus, span), DataType::INDEX, span);
  }

  const size_t pop = rc.residues.PopCount();
  if (pop == 1) {
    for (int64_t r = 0; r < rc.modulus; ++r) {
      if (rc.residues.Test(r)) return std::make_shared<Eq>(rank_expr, MakeInt(r, span), DataType::BOOL, span);
    }
  }
  if (pop == static_cast<size_t>(rc.modulus - 1)) {
    for (int64_t r = 0; r < rc.modulus; ++r) {
      if (!rc.residues.Test(r))
        return std::make_shared<Ne>(rank_expr, MakeInt(r, span), DataType::BOOL, span);
    }
  }
  bool prefix = true;
  for (size_t r = 0; r < pop; ++r) {
    if (!rc.residues.Test(static_cast<int64_t>(r))) {
      prefix = false;
      break;
    }
  }
  if (prefix) {
    return std::make_shared<Lt>(rank_expr, MakeInt(static_cast<int64_t>(pop), span), DataType::BOOL, span);
  }
  bool suffix = true;
  const int64_t r_lo = rc.modulus - static_cast<int64_t>(pop);
  for (int64_t r = r_lo; r < rc.modulus; ++r) {
    if (!rc.residues.Test(r)) {
      suffix = false;
      break;
    }
  }
  if (suffix) {
    return std::make_shared<Ge>(rank_expr, MakeInt(r_lo, span), DataType::BOOL, span);
  }

  ExprPtr disj;
  for (int64_t r = 0; r < rc.modulus; ++r) {
    if (!rc.residues.Test(r)) continue;
    ExprPtr eq = std::make_shared<Eq>(rank_expr, MakeInt(r, span), DataType::BOOL, span);
    disj = disj ? ExprPtr(std::make_shared<Or>(disj, eq, DataType::BOOL, span)) : eq;
  }
  return disj;
}

std::vector<AxisConstraint> OrderedConstraintsForLowering(const RadixBox& box,
                                                          const GuardDomainArena& arena) {
  std::vector<AxisConstraint> ordered;
  ordered.reserve(box.constraints.size());
  for (const auto& c : box.constraints) {
    if (arena.Axis(c.axis).kind == AxisKind::LaunchBlock) ordered.push_back(c);
  }
  for (const auto& c : box.constraints) {
    if (arena.Axis(c.axis).kind != AxisKind::LaunchBlock) ordered.push_back(c);
  }
  return ordered;
}

}  // namespace

std::vector<ExprPtr> LowerBoxToSplitPredicates(const RadixBox& box, const RadixBoxGuard& scope,
                                               const GuardDomainArena& arena, const Span& span) {
  std::vector<ExprPtr> preds;
  const auto ordered = OrderedConstraintsForLowering(box, arena);
  RadixBox known_box;
  for (const auto& c : ordered) {
    const auto hull = EffectiveOuterRange(scope, known_box, c.axis, arena);
    auto axis_preds = LowerAxisConstraint(c, hull, arena, span);
    preds.insert(preds.end(), std::make_move_iterator(axis_preds.begin()),
                 std::make_move_iterator(axis_preds.end()));
    FoldSingleConstraintIntoBox(known_box, c.axis, c.ap, arena);
  }
  for (const auto& rc : box.ranks) {
    preds.push_back(LowerRankCongruenceExpr(rc, arena, span));
  }
  return preds;
}

namespace {
std::optional<std::vector<ExprPtr>> TryLowerSingleBoxSplitPredicates(const RadixBoxGuard& g_eff,
                                                                     const RadixBoxGuard& scope,
                                                                     const Span& span) {
  if (g_eff.Boxes().size() != 1 || !g_eff.Arena()) return std::nullopt;
  RadixBox box = g_eff.Boxes()[0];
  if (!scope.IsOpaque() && g_eff.IsDisjoint() && scope.IsDisjoint()) {
    const auto universe = ReferencedAxes(scope, g_eff);
    auto relaxed = RelaxBoxesForExpr(g_eff, scope, *g_eff.Arena(), universe);
    if (relaxed.size() != 1) return std::nullopt;
    box = std::move(relaxed[0]);
  }
  return LowerBoxToSplitPredicates(box, scope, *g_eff.Arena(), span);
}
}  // namespace

ExprPtr LowerRadixGuardToExpr(const RadixBoxGuard& g, const RadixBoxGuard& scope, const Span& span) {
  if (g.IsEmpty()) return std::make_shared<ConstBool>(false, span);
  if (g.IsTautology()) return std::make_shared<ConstBool>(true, span);
  if (g.IsOpaque()) return g.OpaqueExpr();
  const auto& arena = g.Arena();
  CHECK(arena) << "Missing GuardDomainArena in LowerRadixGuardToExpr";

  const RadixBoxGuard g_eff = scope.IsOpaque() ? g : IntersectGuards(scope, g);
  if (g_eff.IsEmpty()) return std::make_shared<ConstBool>(false, span);
  if (g_eff.IsTautology()) return std::make_shared<ConstBool>(true, span);

  // Complement shortcut: single Eq complement -> Ne(lhs, rhs).
  if (g_eff.IsDisjoint() && g_eff.Boxes().size() >= 2 && !scope.IsOpaque() && scope.IsDisjoint()) {
    if (const auto comp = SubtractGuards(scope, g_eff);
        comp && !comp->IsEmpty() && comp->Boxes().size() == 1) {
      const auto& cb = comp->Boxes()[0];
      if (cb.constraints.size() + cb.ranks.size() == 1) {
        const auto preds = LowerBoxToSplitPredicates(cb, scope, *arena, span);
        if (preds.size() == 1) {
          if (const auto eq = As<Eq>(preds[0])) {
            return std::make_shared<Ne>(eq->left_, eq->right_, DataType::BOOL, span);
          }
        }
      }
    }
  }

  std::vector<RadixBox> emit_boxes = g_eff.Boxes();
  if (!scope.IsOpaque() && g_eff.IsDisjoint() && scope.IsDisjoint()) {
    const auto universe = ReferencedAxes(scope, g_eff);
    emit_boxes = RelaxBoxesForExpr(g_eff, scope, *arena, universe);
  }

  ExprPtr disj;
  for (const auto& box : emit_boxes) {
    const auto preds = LowerBoxToSplitPredicates(box, scope, *arena, span);
    ExprPtr conj;
    for (const auto& p : preds) {
      conj = conj ? std::make_shared<And>(conj, p, DataType::BOOL, span) : p;
    }
    if (!conj) return std::make_shared<ConstBool>(true, span);
    disj = disj ? std::make_shared<Or>(disj, conj, DataType::BOOL, span) : conj;
  }
  return disj ? disj : std::make_shared<ConstBool>(true, span);
}

// ============================================================================
// Control Helpers & ScalarBindingPlanner
// ============================================================================

bool SameControl(const IRNodePtr& left, const IRNodePtr& right) {
  try {
    return structural_equal(left, right, false);
  } catch (const StructuralComparisonUnsupported&) {
    return false;
  }
}

namespace {

bool CommandValueCandidate(const ExprPtr& expression) {
  const auto binary = As<BinaryExpr>(expression);
  if (!binary) return false;
  if ((As<FloorDiv>(expression) && As<Var>(binary->left_) && As<ConstInt>(binary->right_)) ||
      (As<FloorMod>(expression) && As<ConstInt>(binary->right_)))
    return true;
  const auto type = As<ScalarType>(expression->GetType());
  const auto left = As<ScalarType>(binary->left_->GetType());
  const auto right = As<ScalarType>(binary->right_->GetType());
  return type && left && right && type->dtype_ == DataType::BOOL && left->dtype_ == DataType::INDEX &&
         right->dtype_ == DataType::INDEX;
}

bool PureCommandExpression(const ExprPtr& expression, size_t* inspections) {
  ++*inspections;
  const auto type = As<ScalarType>(expression->GetType());
  if (!type || (type->dtype_ != DataType::INDEX && type->dtype_ != DataType::BOOL)) return false;
  if (As<Var>(expression) || As<ConstInt>(expression) || As<ConstBool>(expression)) return true;
  const auto binary = As<BinaryExpr>(expression);
  if (!binary || As<And>(expression) || As<Or>(expression)) return false;
  if (As<FloorDiv>(expression) || As<FloorMod>(expression)) {
    auto divisor = As<ConstInt>(binary->right_);
    if (!divisor || divisor->value_ <= 0) return false;
  }
  return PureCommandExpression(binary->left_, inspections) &&
         PureCommandExpression(binary->right_, inspections);
}

size_t ExprNodeCount(const ExprPtr& expr) {
  if (!expr) return 0;
  size_t count = 1;
  if (const auto binary = As<BinaryExpr>(expr)) {
    count += ExprNodeCount(binary->left_) + ExprNodeCount(binary->right_);
  } else if (const auto unary = As<UnaryExpr>(expr)) {
    count += ExprNodeCount(unary->operand_);
  }
  return count;
}

class ScalarBindingPlanner {
  PhysicalSyncRealization& work_;
  bool share_;
  struct Value {
    ExprPtr expression;
    size_t uses = 0;
    std::optional<size_t> slot;
  };
  using Values = std::map<uint64_t, Value>;

  void Count(const ExprPtr& expression, Values& values) {
    ++work_.control_expression_visits;
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return;
    if (CommandValueCandidate(expression)) {
      auto& entry = values[structural_hash(expression, false)];
      if (!entry.expression) entry.expression = expression;
      if (SameControl(entry.expression, expression)) ++entry.uses;
    }
    Count(binary->left_, values);
    Count(binary->right_, values);
  }

  ScalarRecipePtr Expression(const ExprPtr& expression, Values& values, size_t& slots,
                             std::vector<ScalarBinding>& definitions) {
    ++work_.control_expression_visits;
    Value* entry = nullptr;
    if (share_ && CommandValueCandidate(expression)) {
      const auto found = values.find(structural_hash(expression, false));
      if (found != values.end() && found->second.uses > 1 &&
          SameControl(found->second.expression, expression))
        entry = &found->second;
    }
    if (entry && entry->slot)
      return std::make_shared<ScalarRecipe>(ScalarRecipe{expression, {}, {}, entry->slot});
    auto recipe = std::make_shared<ScalarRecipe>();
    recipe->original = expression;
    if (auto binary = As<BinaryExpr>(expression)) {
      recipe->left = Expression(binary->left_, values, slots, definitions);
      recipe->right = Expression(binary->right_, values, slots, definitions);
    }
    if (!entry) return recipe;
    entry->slot = slots++;
    definitions.push_back({*entry->slot, recipe});
    return std::make_shared<ScalarRecipe>(ScalarRecipe{expression, {}, {}, entry->slot});
  }

  static void Cost(const ScalarRecipePtr& expression, std::array<size_t, 6>& cost) {
    if (!expression || expression->slot || !As<BinaryExpr>(expression->original)) return;
    const auto& e = expression->original;
    const size_t kind = (As<And>(e) || As<Or>(e))                                                      ? 2
                        : (As<Lt>(e) || As<Le>(e) || As<Gt>(e) || As<Ge>(e) || As<Eq>(e) || As<Ne>(e)) ? 1
                                                                                                       : 0;
    ++cost[kind];
    Cost(expression->left, cost);
    Cost(expression->right, cost);
  }

  static const ControlForest& Arm(const ControlForest& forest) {
    if (forest.size() == 1 && !forest.front()->command && forest.front()->guard.IsTautology())
      return Arm(forest.front()->body);
    return forest;
  }

  struct LoweredNode {
    ControlPtr original;
    std::vector<ExprPtr> split_preds;
    ExprPtr combined_expr;
    RadixBoxGuard then_scope;
    RadixBoxGuard else_scope;
  };

  std::vector<LoweredNode> PrepareNodes(const ControlForest& forest, const RadixBoxGuard& scope) {
    std::vector<LoweredNode> lowered;
    lowered.reserve(forest.size());
    for (const auto& node : forest) {
      LoweredNode item;
      item.original = node;
      item.then_scope = scope;
      item.else_scope = scope;
      if (!node->command && !node->guard.IsTautology()) {
        if (node->guard.IsEmpty()) {
          item.combined_expr = std::make_shared<ConstBool>(false, node->span);
        } else if (node->guard.IsOpaque()) {
          item.combined_expr = node->guard.OpaqueExpr();
        } else {
          const auto g_eff = scope.IsOpaque() ? node->guard : IntersectGuards(scope, node->guard);
          item.then_scope = scope.IsOpaque() ? scope : g_eff;
          if (!node->alternative.empty() && !scope.IsOpaque()) {
            item.else_scope = SubtractGuards(scope, g_eff).value_or(scope);
          }
          if (!scope.IsOpaque() && g_eff.IsEmpty()) {
            item.combined_expr = std::make_shared<ConstBool>(false, node->span);
          } else if (!scope.IsOpaque() && EquivalentGuardsInScope(g_eff, RadixBoxGuard::True(), scope)) {
            item.combined_expr = std::make_shared<ConstBool>(true, node->span);
          } else if (node->alternative.empty()) {
            if (auto preds = TryLowerSingleBoxSplitPredicates(g_eff, scope, node->span)) {
              if (preds->empty()) {
                item.combined_expr = std::make_shared<ConstBool>(true, node->span);
              } else if (preds->size() == 1) {
                item.combined_expr = (*preds)[0];
              } else {
                item.split_preds = std::move(*preds);
                item.combined_expr = item.split_preds.front();
              }
            } else {
              item.combined_expr = LowerRadixGuardToExpr(node->guard, scope, node->span);
            }
          } else {
            item.combined_expr = LowerRadixGuardToExpr(node->guard, scope, node->span);
          }
        }
      }
      lowered.push_back(std::move(item));
    }
    return lowered;
  }

  std::shared_ptr<const FrozenCommandForest> BuildSplitTail(const LoweredNode& item, size_t pred_idx,
                                                            const RadixBoxGuard& scope) {
    auto result = std::make_shared<FrozenCommandForest>();
    result->scope = scope;
    FrozenCommand cmd;
    cmd.original = item.original;
    cmd.condition_expr = item.split_preds[pred_idx];
    Values unshared;
    size_t slots = 0;
    const bool pure = share_ && PureCommandExpression(cmd.condition_expr, &work_.control_expression_visits);
    if (pure) Count(cmd.condition_expr, unshared);
    cmd.condition = Expression(cmd.condition_expr, unshared, slots, cmd.definitions);
    for (const auto& binding : cmd.definitions) {
      Cost(binding.value, cmd.scalar_cost);
      ++cmd.scalar_cost[5];
    }
    Cost(cmd.condition, cmd.scalar_cost);
    ++cmd.scalar_cost[3];
    if (pred_idx + 1 < item.split_preds.size()) {
      cmd.body = BuildSplitTail(item, pred_idx + 1, scope);
    } else {
      cmd.body = BuildImpl(Arm(item.original->body), item.then_scope);
    }
    auto empty_alt = std::make_shared<FrozenCommandForest>();
    empty_alt->scope = scope;
    cmd.alternative = std::move(empty_alt);
    result->commands.push_back(std::move(cmd));
    return result;
  }

  std::shared_ptr<const FrozenCommandForest> BuildImpl(const ControlForest& forest,
                                                       const RadixBoxGuard& scope) {
    auto result = std::make_shared<FrozenCommandForest>();
    result->roots.assign(forest.begin(), forest.end());
    result->scope = scope;
    const auto lowered = PrepareNodes(forest, scope);
    Values values;
    size_t slots = 0;
    if (share_) {
      for (const auto& item : lowered) {
        if (item.combined_expr && PureCommandExpression(item.combined_expr, &work_.control_expression_visits))
          Count(item.combined_expr, values);
      }
    }
    for (const auto& item : lowered) {
      ++work_.control_statement_visits;
      const auto& node = item.original;
      if (const auto known = item.combined_expr ? As<ConstBool>(item.combined_expr) : nullptr) {
        if (known->value_) {
          auto sub = BuildImpl(Arm(node->body), item.then_scope);
          result->commands.insert(result->commands.end(), sub->commands.begin(), sub->commands.end());
        } else if (!node->alternative.empty()) {
          auto sub = BuildImpl(Arm(node->alternative), item.else_scope);
          result->commands.insert(result->commands.end(), sub->commands.begin(), sub->commands.end());
        }
        continue;
      }
      FrozenCommand command;
      command.original = node;
      command.condition_expr = item.combined_expr;
      if (command.condition_expr) {
        Values unshared;
        const bool pure =
            share_ && PureCommandExpression(command.condition_expr, &work_.control_expression_visits);
        command.condition =
            Expression(command.condition_expr, pure ? values : unshared, slots, command.definitions);
        for (const auto& binding : command.definitions) {
          Cost(binding.value, command.scalar_cost);
          ++command.scalar_cost[5];
        }
        Cost(command.condition, command.scalar_cost);
        ++command.scalar_cost[3];
      }
      if (!node->command) {
        if (item.split_preds.size() > 1) {
          command.body = BuildSplitTail(item, 1, scope);
          auto empty_alt = std::make_shared<FrozenCommandForest>();
          empty_alt->scope = scope;
          command.alternative = std::move(empty_alt);
        } else {
          command.body = BuildImpl(Arm(node->body), item.then_scope);
          command.alternative = BuildImpl(Arm(node->alternative), item.else_scope);
        }
      }
      result->commands.push_back(std::move(command));
    }
    return result;
  }

 public:
  ScalarBindingPlanner(PhysicalSyncRealization& work, bool share) : work_(work), share_(share) {}
  std::shared_ptr<const FrozenCommandForest> Build(const ControlForest& forest, const RadixBoxGuard& scope) {
    return BuildImpl(forest, scope);
  }
};

class ScalarRecipeEmitter : public IRMutator {
  ExprPtr left_, right_;
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& expression) override {
    class Children : public IRMutator {
      const BinaryExprPtr& root_;
      ExprPtr left_, right_;

     public:
      Children(const BinaryExprPtr& root, ExprPtr left, ExprPtr right)
          : root_(root), left_(std::move(left)), right_(std::move(right)) {}
      ExprPtr VisitBinaryExpr_(const BinaryExprPtr& expression) override {
        if (expression == root_->left_) return left_;
        if (expression == root_->right_) return right_;
        return IRMutator::VisitBinaryExpr_(expression);
      }
      ExprPtr Build() { return IRMutator::VisitBinaryExpr_(root_); }
    };
    return Children(expression, left_, right_).Build();
  }

 public:
  ScalarRecipeEmitter(ExprPtr left, ExprPtr right) : left_(std::move(left)), right_(std::move(right)) {}
};

ExprPtr EmitScalarRecipe(const ScalarRecipePtr& recipe, const std::map<size_t, VarPtr>& slots) {
  if (recipe->slot) return slots.at(*recipe->slot);
  if (!recipe->left) return recipe->original;
  return ScalarRecipeEmitter(EmitScalarRecipe(recipe->left, slots), EmitScalarRecipe(recipe->right, slots))
      .VisitExpr(recipe->original);
}

std::vector<StmtPtr> EmitFrozen(const FrozenCommandForest& plan, SyncTokenNames& names,
                                PhysicalSyncRealization& work) {
  std::vector<StmtPtr> output;
  std::map<size_t, VarPtr> slots;
  for (const auto& command : plan.commands) {
    const auto& node = command.original;
    for (const auto& binding : command.definitions) {
      auto expression = EmitScalarRecipe(binding.value, slots);
      auto variable =
          std::make_shared<Var>(names.Next("sync_control_"), expression->GetType(), expression->span_);
      slots.emplace(binding.slot, variable);
      output.push_back(std::make_shared<AssignStmt>(variable, expression, expression->span_));
      ++work.shared_control_values;
    }
    if (node->command) {
      output.push_back(std::make_shared<EvalStmt>(node->command, node->span));
      ++work.materialized_control_statements;
      continue;
    }
    auto body = std::make_shared<SeqStmts>(EmitFrozen(*command.body, names, work), node->span);
    ++work.materialized_control_statements;
    if (!command.condition_expr) {
      output.push_back(body);
      continue;
    }
    std::optional<StmtPtr> alternative;
    if (command.alternative && !command.alternative->commands.empty()) {
      alternative = std::make_shared<SeqStmts>(EmitFrozen(*command.alternative, names, work), node->span);
      ++work.materialized_control_statements;
    }
    output.push_back(std::make_shared<IfStmt>(EmitScalarRecipe(command.condition, slots), body, alternative,
                                              std::vector<VarPtr>{}, node->span));
    ++work.materialized_control_statements;
  }
  return output;
}

}  // namespace

std::shared_ptr<const FrozenCommandForest> FreezeCommandForest(const ControlForest& forest,
                                                               PhysicalSyncRealization& work,
                                                               const RadixBoxGuard& scope) {
  work.control_statement_visits += forest.size();
  if (forest.frozen && SameGuard(forest.frozen->scope, scope) &&
      forest.frozen->roots.size() == forest.size() &&
      std::equal(forest.begin(), forest.end(), forest.frozen->roots.begin()))
    return forest.frozen;
  try {
    forest.frozen = ScalarBindingPlanner(work, true).Build(forest, scope);
  } catch (const StructuralComparisonUnsupported&) {
    forest.frozen = ScalarBindingPlanner(work, false).Build(forest, scope);
  }
  return forest.frozen;
}

std::vector<StmtPtr> MaterializeFrozenCommands(const ControlForest& forest, SyncTokenNames& names,
                                               PhysicalSyncRealization& work, const RadixBoxGuard& scope) {
  return EmitFrozen(*FreezeCommandForest(forest, work, scope), names, work);
}

namespace {

std::vector<const Var*> GuardPredicateInputs(const RadixBoxGuard& guard, PhysicalSyncRealization& work,
                                             bool& qualified) {
  if (guard.IsEmpty() || guard.IsTautology()) {
    qualified = true;
    return {};
  }
  if (guard.IsOpaque()) {
    return PredicateInputs(guard.OpaqueExpr(), work, qualified);
  }
  qualified = true;
  std::set<const Var*> vars;
  const auto& arena = guard.Arena();
  if (arena) {
    for (const AxisId ax : ReferencedAxes(guard)) {
      const auto& info = arena->Axis(ax);
      if (info.var) vars.insert(info.var.get());
      if (info.start_expression) CollectExprVars(info.start_expression, vars, work);
    }
  }
  return {vars.begin(), vars.end()};
}

void VisitControlGuardExpressions(const RadixBoxGuard& guard, IRVisitor& visitor) {
  if (guard.IsEmpty() || guard.IsTautology()) return;
  if (guard.IsOpaque()) {
    if (guard.OpaqueExpr()) visitor.VisitExpr(guard.OpaqueExpr());
    return;
  }
  if (const auto& arena = guard.Arena()) {
    for (const AxisId ax : ReferencedAxes(guard)) {
      const auto& info = arena->Axis(ax);
      if (info.var) visitor.VisitExpr(info.var);
      if (info.start_expression) visitor.VisitExpr(info.start_expression);
    }
  }
}

}  // namespace

size_t HashRadixBoxGuard(const RadixBoxGuard& guard) {
  if (guard.IsEmpty()) return 0x101;
  if (guard.IsTautology()) return 0x202;
  if (guard.IsOpaque()) return structural_hash(guard.OpaqueExpr(), false);
  size_t h = 0x303;
  auto mix = [&](size_t v) { h ^= v + size_t(0x9e3779b9) + (h << 6) + (h >> 2); };
  const auto& arena = guard.Arena();
  mix(guard.Boxes().size());
  for (const auto& box : guard.Boxes()) {
    mix(box.constraints.size());
    for (const auto& c : box.constraints) {
      if (arena) mix(HashAxisInfo(arena->Axis(c.axis)));
      mix(static_cast<size_t>(c.ap.first));
      mix(static_cast<size_t>(c.ap.end));
      mix(static_cast<size_t>(c.ap.stride));
    }
    mix(box.ranks.size());
    for (const auto& r : box.ranks) {
      mix(static_cast<size_t>(r.modulus));
      mix(static_cast<size_t>(r.base));
      mix(r.terms.size());
      for (const auto& t : r.terms) {
        if (arena) mix(HashAxisInfo(arena->Axis(t.axis)));
        mix(static_cast<size_t>(t.first));
        mix(static_cast<size_t>(t.step));
        mix(static_cast<size_t>(t.coeff));
      }
      mix(r.residues.Hash());
    }
  }
  return h;
}

void RequireCommandScope(const ControlForest& forest, const CommandPredicateScope& scope,
                         PhysicalSyncRealization& work) {
  CHECK(scope.owner && scope.entry_ready) << "Command predicate scope has no original owner";
  std::set<const CommandControl*> visited;
  const auto visit = [&](const auto& self, const ControlPtr& node) -> void {
    if (!visited.insert(node.get()).second) return;
    ++work.schedule_inspected_descriptors;
    CHECK(node->predicate_qualified) << "Unproved command predicate arithmetic";
    for (const auto* variable : node->predicate_inputs) {
      ++work.schedule_inspected_expressions;
      CHECK(scope.ready.count(variable) || scope.entry_ready->count(variable))
          << "Command predicate uses an unavailable original-cut coordinate";
    }
    for (const auto& child : node->body) self(self, child);
    for (const auto& child : node->alternative) self(self, child);
  };
  for (const auto& node : forest) visit(visit, node);
}

ControlPtr ControlSequence(ControlForest body, const Span& span, PhysicalSyncRealization& work) {
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(
      CommandControl{RadixBoxGuard::True(), nullptr, std::move(body), {}, span});
}

ControlPtr ControlBranch(RadixBoxGuard guard, ControlPtr body, std::optional<ControlPtr> alternative,
                         const Span& span, PhysicalSyncRealization& work) {
  if (guard.IsTautology()) return body;
  if (guard.IsEmpty()) return alternative ? *alternative : ControlSequence({}, span, work);
  bool qualified = guard.PredicateQualified();
  auto inputs = GuardPredicateInputs(guard, work, qualified);
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(
      CommandControl{std::move(guard),
                     nullptr,
                     {std::move(body)},
                     alternative ? ControlForest{*alternative} : ControlForest{},
                     span,
                     std::move(inputs),
                     qualified});
}

ControlPtr ControlBranch(ExprPtr condition, ControlPtr body, std::optional<ControlPtr> alternative,
                         const Span& span, PhysicalSyncRealization& work) {
  bool qualified = true;
  auto inputs = PredicateInputs(condition, work, qualified);
  if (auto known = As<ConstBool>(condition))
    return known->value_ ? body : alternative ? *alternative : ControlSequence({}, span, work);
  return ControlBranch(RadixBoxGuard::OpaqueDiagnostic(std::move(condition), inputs, qualified),
                       std::move(body), std::move(alternative), span, work);
}

ControlPtr ControlCommand(ExprPtr command, const Span& span, PhysicalSyncRealization& work) {
  const auto call = As<Call>(command);
  CHECK(call && (IsOp(call, "system.set_flag") || IsOp(call, "system.wait_flag") ||
                 IsOp(call, "system.bar_m") || IsOp(call, "system.bar_v")))
      << "Command descriptor requires a qualified physical synchronization leaf";
  CHECK(call->args_.empty()) << "Command descriptor requires literal physical assignments";
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(
      CommandControl{RadixBoxGuard::True(), std::move(command), {}, {}, span});
}

ControlPtr RebuildControl(const ControlPtr& original, RadixBoxGuard guard, ExprPtr command,
                          ControlForest body, ControlForest alternative, PhysicalSyncRealization& work) {
  auto inputs = original->predicate_inputs;
  bool qualified = original->predicate_qualified;
  if (!SameGuard(guard, original->guard)) {
    qualified = guard.PredicateQualified();
    inputs = GuardPredicateInputs(guard, work, qualified);
  }
  INTERNAL_CHECK(command == original->command || SameControl(command, original->command))
      << "Descriptor rebuild cannot replace physical assignments";
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(CommandControl{std::move(guard), std::move(command),
                                                         std::move(body), std::move(alternative),
                                                         original->span, std::move(inputs), qualified});
}

bool SameControl(const ControlPtr& a, const ControlPtr& b) {
  if (a == b) return true;
  if (!SameGuard(a->guard, b->guard) || static_cast<bool>(a->command) != static_cast<bool>(b->command) ||
      a->body.size() != b->body.size() || a->alternative.size() != b->alternative.size())
    return false;
  if (a->command && !SameControl(a->command, b->command)) return false;
  for (size_t i = 0; i < a->body.size(); ++i)
    if (!SameControl(a->body[i], b->body[i])) return false;
  for (size_t i = 0; i < a->alternative.size(); ++i)
    if (!SameControl(a->alternative[i], b->alternative[i])) return false;
  return true;
}

namespace {

void VisitControlExpressions(const ControlPtr& node, IRVisitor& visitor) {
  VisitControlGuardExpressions(node->guard, visitor);
  if (node->command) visitor.VisitExpr(node->command);
  for (const auto& child : node->body) VisitControlExpressions(child, visitor);
  for (const auto& child : node->alternative) VisitControlExpressions(child, visitor);
}

// ============================================================================
// Domain-Aware CommandGuards
// ============================================================================

struct GuardMetrics {
  size_t guards = 0;
  size_t expressions = 0;
  size_t statements = 0;
};

GuardMetrics AnalyticGuardMetrics(const RadixBoxGuard& guard, const RadixBoxGuard& scope,
                                  bool has_alternative, const Span& span) {
  if (guard.IsEmpty() || guard.IsTautology()) return {};
  if (guard.IsOpaque()) {
    return {1, ExprNodeCount(guard.OpaqueExpr()), 1};
  }
  const RadixBoxGuard g_eff = scope.IsOpaque() ? guard : IntersectGuards(scope, guard);
  if (!scope.IsOpaque() &&
      (g_eff.IsEmpty() || EquivalentGuardsInScope(g_eff, RadixBoxGuard::True(), scope))) {
    return {};
  }
  if (!has_alternative) {
    if (const auto preds = TryLowerSingleBoxSplitPredicates(g_eff, scope, span)) {
      if (preds->empty()) return {};
      size_t exprs = 0;
      for (const auto& p : *preds) exprs += ExprNodeCount(p);
      return {preds->size(), exprs, 2 * preds->size() - 1};
    }
  }
  const ExprPtr expr = LowerRadixGuardToExpr(guard, scope, span);
  if (As<ConstBool>(expr)) return {};
  return {1, ExprNodeCount(expr), 1};
}

class CommandGuards {
  struct Node {
    RadixBoxGuard guard = RadixBoxGuard::True();
    ControlPtr leaf;
    std::shared_ptr<const Span> span;
    std::vector<Node> children, alternative;
    const Span& GetSpan() const {
      static const Span kUnknown = Span::unknown();
      return span ? *span : kUnknown;
    }
  };
  std::vector<Node> roots_;
  size_t visits_ = 0, expressions_ = 0;

  static bool SameNode(const Node& a, const Node& b) {
    if (static_cast<bool>(a.leaf) != static_cast<bool>(b.leaf) || !SameGuard(a.guard, b.guard) ||
        a.children.size() != b.children.size() || a.alternative.size() != b.alternative.size())
      return false;
    if (a.leaf && !SameControl(a.leaf, b.leaf)) return false;
    for (size_t i = 0; i < a.children.size(); ++i)
      if (!SameNode(a.children[i], b.children[i])) return false;
    for (size_t i = 0; i < a.alternative.size(); ++i)
      if (!SameNode(a.alternative[i], b.alternative[i])) return false;
    return true;
  }

  static bool SameNodeList(const std::vector<Node>& a, const std::vector<Node>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (!SameNode(a[i], b[i])) return false;
    return true;
  }

  static size_t NodeGuardCount(const Node& node, const RadixBoxGuard& scope) {
    if (node.leaf || node.guard.IsTautology() || node.guard.IsEmpty()) return 0;
    return AnalyticGuardMetrics(node.guard, scope, !node.alternative.empty(), node.GetSpan()).guards;
  }

  static size_t CountNodeGuards(const std::vector<Node>& nodes, const RadixBoxGuard& scope) {
    size_t count = 0;
    for (const auto& node : nodes) {
      if (node.leaf || node.guard.IsTautology()) {
        count += CountNodeGuards(node.children, scope);
        count += CountNodeGuards(node.alternative, scope);
        continue;
      }
      count += NodeGuardCount(node, scope);
      const RadixBoxGuard s_then =
          (node.guard.IsOpaque() || scope.IsOpaque()) ? scope : IntersectGuards(scope, node.guard);
      const RadixBoxGuard s_else = (node.guard.IsOpaque() || scope.IsOpaque() || node.alternative.empty())
                                       ? scope
                                       : SubtractGuards(scope, s_then).value_or(scope);
      count += CountNodeGuards(node.children, s_then);
      count += CountNodeGuards(node.alternative, s_else);
    }
    return count;
  }

  static void NodeSubtreeMetrics(const Node& node, const RadixBoxGuard& scope, size_t& commands,
                                 size_t& guards, size_t& expressions) {
    if (node.leaf) {
      const CaseCodeSize leaf_size({node.leaf}, scope);
      commands += leaf_size.commands;
      guards += leaf_size.guards;
      expressions += leaf_size.expressions;
      return;
    }
    const auto gm = AnalyticGuardMetrics(node.guard, scope, !node.alternative.empty(), node.GetSpan());
    guards += gm.guards;
    expressions += gm.expressions;
    const RadixBoxGuard s_then =
        (node.guard.IsOpaque() || scope.IsOpaque()) ? scope : IntersectGuards(scope, node.guard);
    const RadixBoxGuard s_else = (node.guard.IsOpaque() || scope.IsOpaque() || node.alternative.empty())
                                     ? scope
                                     : SubtractGuards(scope, s_then).value_or(scope);
    for (const auto& c : node.children) NodeSubtreeMetrics(c, s_then, commands, guards, expressions);
    for (const auto& c : node.alternative) NodeSubtreeMetrics(c, s_else, commands, guards, expressions);
  }

  static bool HasOpaqueDescendant(const Node& node) {
    if (node.leaf) return node.leaf->guard.IsOpaque();
    if (node.guard.IsOpaque()) return true;
    for (const auto& c : node.children)
      if (HasOpaqueDescendant(c)) return true;
    for (const auto& c : node.alternative)
      if (HasOpaqueDescendant(c)) return true;
    return false;
  }

  static bool CollectOps(const std::vector<Node>& nodes, std::string& kind, std::vector<CallPtr>& calls) {
    for (const auto& node : nodes) {
      if (node.leaf) {
        if (!node.leaf->command) return false;
        const auto call = As<Call>(node.leaf->command);
        if (!call) return false;
        const std::string& name = call->op_->name_;
        if (name != "system.set_flag" && name != "system.wait_flag") return false;
        if (kind.empty())
          kind = name;
        else if (kind != name)
          return false;
        calls.push_back(call);
      } else {
        if (!CollectOps(node.children, kind, calls) || !CollectOps(node.alternative, kind, calls))
          return false;
      }
    }
    return true;
  }

  static bool SameKindDisjointCommands(const std::vector<Node>& a, const Node& b) {
    std::string kind_a, kind_b;
    std::vector<CallPtr> calls_a, calls_b;
    if (!CollectOps(a, kind_a, calls_a) || !CollectOps({b}, kind_b, calls_b)) return false;
    if (kind_a.empty() || kind_a != kind_b) return false;
    for (const auto& ca : calls_a)
      for (const auto& cb : calls_b)
        if (ca == cb || SameControl(ca, cb)) return false;
    return true;
  }

  static bool GuardHasLaunchBlock(const RadixBoxGuard& g) {
    const auto arena = g.Arena();
    if (!arena || g.IsOpaque()) return false;
    for (const auto& box : g.Boxes()) {
      for (const auto& c : box.constraints) {
        if (arena->Axis(c.axis).kind == AxisKind::LaunchBlock) return true;
      }
      for (const auto& r : box.ranks) {
        for (const auto& t : r.terms) {
          if (arena->Axis(t.axis).kind == AxisKind::LaunchBlock) return true;
        }
      }
    }
    return false;
  }

  static bool CommutesAcross(const RadixBoxGuard& moving_guard, const std::vector<Node>& moving,
                             const std::vector<Node>& nodes, size_t lo, size_t hi,
                             const RadixBoxGuard& scope) {
    const bool check_launch_disjoint = GuardHasLaunchBlock(moving_guard);
    for (size_t m = lo; m < hi; ++m) {
      if (nodes[m].guard.IsOpaque() || !SameKindDisjointCommands(moving, nodes[m])) return false;
      if ((check_launch_disjoint || GuardHasLaunchBlock(nodes[m].guard)) &&
          !AreDisjointGuardsInScope(moving_guard, nodes[m].guard, scope)) {
        return false;
      }
    }
    return true;
  }

  struct OrderedCmp {
    bool is_le;
    ExprPtr lhs, rhs;
  };
  static std::optional<OrderedCmp> CanonicalComparison(const ExprPtr& expr) {
    if (const auto lt = As<Lt>(expr)) return OrderedCmp{false, lt->left_, lt->right_};
    if (const auto gt = As<Gt>(expr)) return OrderedCmp{false, gt->right_, gt->left_};
    if (const auto le = As<Le>(expr)) return OrderedCmp{true, le->left_, le->right_};
    if (const auto ge = As<Ge>(expr)) return OrderedCmp{true, ge->right_, ge->left_};
    return std::nullopt;
  }

  bool AreOpaqueComplementary(const ExprPtr& a, const ExprPtr& b) {
    if (!PureCommandExpression(a, &expressions_) || !PureCommandExpression(b, &expressions_)) return false;
    if (const auto na = As<Not>(a); na && SameControl(na->operand_, b)) return true;
    if (const auto nb = As<Not>(b); nb && SameControl(a, nb->operand_)) return true;
    {
      const auto ea = As<Eq>(a);
      const auto nb = As<Ne>(b);
      if (ea && nb &&
          ((SameControl(ea->left_, nb->left_) && SameControl(ea->right_, nb->right_)) ||
           (SameControl(ea->left_, nb->right_) && SameControl(ea->right_, nb->left_))))
        return true;
    }
    {
      const auto eb = As<Eq>(b);
      const auto na = As<Ne>(a);
      if (eb && na &&
          ((SameControl(eb->left_, na->left_) && SameControl(eb->right_, na->right_)) ||
           (SameControl(eb->left_, na->right_) && SameControl(eb->right_, na->left_))))
        return true;
    }
    const auto ca = CanonicalComparison(a);
    const auto cb = CanonicalComparison(b);
    return ca && cb && ca->is_le != cb->is_le && SameControl(ca->lhs, cb->rhs) &&
           SameControl(ca->rhs, cb->lhs);
  }

  bool AreComplementary(const Node& a, const Node& b, const RadixBoxGuard& scope) {
    if (a.leaf || b.leaf) return false;
    if (a.guard.IsOpaque() && b.guard.IsOpaque()) {
      return AreOpaqueComplementary(a.guard.OpaqueExpr(), b.guard.OpaqueExpr());
    }
    return AreComplementaryGuardsInScope(a.guard, b.guard, scope);
  }

  bool AreEquivalent(const Node& a, const Node& b, const RadixBoxGuard& scope) {
    if (a.leaf || b.leaf) return false;
    if (a.guard.IsOpaque() || b.guard.IsOpaque()) {
      return a.guard.IsOpaque() && b.guard.IsOpaque() &&
             SameControl(a.guard.OpaqueExpr(), b.guard.OpaqueExpr());
    }
    return EquivalentGuardsInScope(a.guard, b.guard, scope);
  }

  static bool IsComplementShortcutNe(const RadixBoxGuard& g, const RadixBoxGuard& scope, const Span& span) {
    if (g.IsOpaque() || scope.IsOpaque() || g.Boxes().size() < 2) return false;
    return As<Ne>(LowerRadixGuardToExpr(g, scope, span)) != nullptr;
  }

  static bool RankUsesAxis(const RadixBox& box, AxisId axis) {
    for (const auto& r : box.ranks) {
      for (const auto& t : r.terms) {
        if (t.axis == axis) return true;
      }
    }
    return false;
  }

  static bool MatchesOuterAxis(const Node& node, const AxisConstraint& outer_c) {
    return !node.leaf && !node.guard.IsOpaque() && node.alternative.empty() &&
           node.guard.Boxes().size() == 1 && !node.guard.Boxes()[0].constraints.empty() &&
           node.guard.Boxes()[0].constraints[0] == outer_c &&
           !RankUsesAxis(node.guard.Boxes()[0], outer_c.axis);
  }

  bool TryFactorNA4At(std::vector<Node>& list, size_t start, const RadixBoxGuard& scope,
                      bool pair_else_branches) {
    if (start + 1 >= list.size() || list[start].leaf || list[start].guard.IsOpaque() ||
        !list[start].alternative.empty() || list[start].guard.Boxes().size() != 1 ||
        list[start].guard.Boxes()[0].constraints.empty())
      return false;
    const auto& arena = list[start].guard.Arena();
    if (!arena) return false;
    const auto& box0 = list[start].guard.Boxes()[0];
    const AxisConstraint outer_c = box0.constraints[0];
    if (box0.constraints.size() == 1 && box0.ranks.empty()) return false;
    if (RankUsesAxis(box0, outer_c.axis)) return false;
    if (!MatchesOuterAxis(list[start + 1], outer_c)) return false;
    size_t k = start + 1;
    while (k + 1 < list.size() && MatchesOuterAxis(list[k + 1], outer_c)) ++k;

    RadixBox outer_box;
    outer_box.constraints.push_back(outer_c);
    const RadixBoxGuard outer_guard = RadixBoxGuard::FromBox(arena, outer_box);
    const auto span_i = list[start].span;
    const size_t outer_guards = AnalyticGuardMetrics(outer_guard, scope, false, list[start].GetSpan()).guards;
    size_t before_guards = 0;
    for (size_t idx = start; idx <= k; ++idx) before_guards += CountNodeGuards({list[idx]}, scope);
    std::vector<Node> inner(list.begin() + static_cast<std::ptrdiff_t>(start),
                            list.begin() + static_cast<std::ptrdiff_t>(k) + 1);
    const RadixBoxGuard inner_scope = IntersectGuards(scope, outer_guard);
    if (!inner_scope.IsDisjoint() || inner_scope.IsEmpty()) return false;
    for (const auto& child : inner) {
      const auto narrowed = SimplifyGuardGiven(child.guard, inner_scope);
      if (narrowed.Boxes().size() == 1 && !narrowed.Boxes()[0].constraints.empty() &&
          narrowed.Boxes()[0].constraints[0] == outer_c) {
        return false;
      }
    }
    NormalizeNodes(inner, inner_scope, pair_else_branches);
    const size_t after_guards = outer_guards + CountNodeGuards(inner, inner_scope);
    if (after_guards >= before_guards) return false;
    Node wrapped{outer_guard, nullptr, span_i, std::move(inner), {}};
    list.erase(list.begin() + static_cast<std::ptrdiff_t>(start),
               list.begin() + static_cast<std::ptrdiff_t>(k) + 1);
    list.insert(list.begin() + static_cast<std::ptrdiff_t>(start), std::move(wrapped));
    return true;
  }

  bool TryFuseAdjacent7BAt(std::vector<Node>& list, size_t idx, const RadixBoxGuard& scope,
                           bool pair_else_branches) {
    const size_t j = idx + 1;
    const auto min_len = std::min(list[idx].children.size(), list[j].children.size());
    size_t prefix = 0;
    while (prefix < min_len && SameNode(list[idx].children[prefix], list[j].children[prefix])) ++prefix;
    size_t suffix = 0;
    while (prefix + suffix < min_len && SameNode(list[idx].children[list[idx].children.size() - 1 - suffix],
                                                 list[j].children[list[j].children.size() - 1 - suffix]))
      ++suffix;
    if (!(prefix > 0 || suffix > 0 ||
          (pair_else_branches && !list[idx].children.empty() && !list[j].children.empty()))) {
      return false;
    }
    std::vector<Node> replacement;
    for (size_t p = 0; p < prefix; ++p) replacement.push_back(list[idx].children[p]);
    std::vector<Node> rem_i(list[idx].children.begin() + static_cast<std::ptrdiff_t>(prefix),
                            list[idx].children.end() - static_cast<std::ptrdiff_t>(suffix));
    std::vector<Node> rem_j(list[j].children.begin() + static_cast<std::ptrdiff_t>(prefix),
                            list[j].children.end() - static_cast<std::ptrdiff_t>(suffix));
    std::vector<Node> suf(list[idx].children.end() - static_cast<std::ptrdiff_t>(suffix),
                          list[idx].children.end());
    const auto& residual = !rem_i.empty() ? rem_i : rem_j;
    const bool suf_before_residual = suffix > 0 && (!rem_i.empty() != !rem_j.empty()) && suf.size() == 1 &&
                                     SameKindDisjointCommands(residual, suf.front());
    if (suf_before_residual) {
      replacement.insert(replacement.end(), std::make_move_iterator(suf.begin()),
                         std::make_move_iterator(suf.end()));
    }
    if (pair_else_branches && !rem_i.empty() && !rem_j.empty()) {
      replacement.push_back(
          Node{list[idx].guard, nullptr, list[idx].span, std::move(rem_i), std::move(rem_j)});
    } else {
      if (!rem_i.empty())
        replacement.push_back(Node{list[idx].guard, nullptr, list[idx].span, std::move(rem_i), {}});
      if (!rem_j.empty())
        replacement.push_back(Node{list[j].guard, nullptr, list[j].span, std::move(rem_j), {}});
    }
    if (!suf_before_residual) {
      replacement.insert(replacement.end(), std::make_move_iterator(suf.begin()),
                         std::make_move_iterator(suf.end()));
    }
    list.erase(list.begin() + static_cast<std::ptrdiff_t>(idx),
               list.begin() + static_cast<std::ptrdiff_t>(j) + 1);
    list.insert(list.begin() + static_cast<std::ptrdiff_t>(idx), std::make_move_iterator(replacement.begin()),
                std::make_move_iterator(replacement.end()));
    NormalizeNodes(list, scope, pair_else_branches);
    return true;
  }

  void NormalizeNodes(std::vector<Node>& nodes, const RadixBoxGuard& scope, bool pair_else_branches) {
    // 1. Top-down domain narrowing & recursive normalization.
    for (size_t i = 0; i < nodes.size();) {
      auto& node = nodes[i];
      if (node.leaf) {
        ++i;
        continue;
      }
      if (node.guard.IsOpaque() || scope.IsOpaque()) {
        NormalizeNodes(node.children, scope, pair_else_branches);
        NormalizeNodes(node.alternative, scope, pair_else_branches);
        ++i;
        continue;
      }
      const RadixBoxGuard g_eff = IntersectGuards(scope, node.guard);
      if (g_eff.IsEmpty()) {
        auto alt = std::move(node.alternative);
        nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
        NormalizeNodes(alt, scope, pair_else_branches);
        nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(i), std::make_move_iterator(alt.begin()),
                     std::make_move_iterator(alt.end()));
        continue;
      }
      if (EquivalentGuardsInScope(g_eff, RadixBoxGuard::True(), scope)) {
        auto lifted = std::move(node.children);
        nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
        NormalizeNodes(lifted, scope, pair_else_branches);
        nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(i), std::make_move_iterator(lifted.begin()),
                     std::make_move_iterator(lifted.end()));
        continue;
      }
      node.guard = SimplifyGuardGiven(g_eff, scope);
      NormalizeNodes(node.children, g_eff, pair_else_branches);
      const RadixBoxGuard s_else = SubtractGuards(scope, g_eff).value_or(scope);
      NormalizeNodes(node.alternative, s_else, pair_else_branches);
      ++i;
    }

    bool changed = true;
    while (changed) {
      changed = false;

      // 2. Empty / identical-arm cleanup.
      for (size_t i = 0; i < nodes.size();) {
        if (!nodes[i].leaf && !nodes[i].guard.IsTautology() && nodes[i].children.empty() &&
            nodes[i].alternative.empty()) {
          nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
          changed = true;
          continue;
        }
        if (!nodes[i].leaf && !nodes[i].guard.IsTautology() && !nodes[i].alternative.empty() &&
            SameNodeList(nodes[i].children, nodes[i].alternative)) {
          auto lifted = std::move(nodes[i].children);
          nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
          nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(i),
                       std::make_move_iterator(lifted.begin()), std::make_move_iterator(lifted.end()));
          changed = true;
          continue;
        }
        if (!nodes[i].leaf && !nodes[i].guard.IsOpaque() && !scope.IsOpaque() && nodes[i].children.empty() &&
            !nodes[i].alternative.empty()) {
          const auto g_eff = IntersectGuards(scope, nodes[i].guard);
          if (const auto comp = SubtractGuards(scope, g_eff)) {
            const auto comp_simp = SimplifyGuardGiven(*comp, scope);
            const size_t old_g = NodeGuardCount(nodes[i], scope);
            const size_t new_g = AnalyticGuardMetrics(comp_simp, scope, false, nodes[i].GetSpan()).guards;
            if (new_g <= old_g) {
              nodes[i].guard = comp_simp;
              nodes[i].children = std::move(nodes[i].alternative);
              nodes[i].alternative.clear();
              changed = true;
              continue;
            }
          }
        }
        ++i;
      }

      // 3. Adjacent equivalent guards.
      for (size_t i = 0; i + 1 < nodes.size();) {
        if (!nodes[i].leaf && !nodes[i + 1].leaf && AreEquivalent(nodes[i], nodes[i + 1], scope)) {
          const bool need_renorm_else = !nodes[i].alternative.empty() || !nodes[i + 1].alternative.empty();
          nodes[i].children.insert(nodes[i].children.end(),
                                   std::make_move_iterator(nodes[i + 1].children.begin()),
                                   std::make_move_iterator(nodes[i + 1].children.end()));
          nodes[i].alternative.insert(nodes[i].alternative.end(),
                                      std::make_move_iterator(nodes[i + 1].alternative.begin()),
                                      std::make_move_iterator(nodes[i + 1].alternative.end()));
          nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i) + 1);
          const RadixBoxGuard s_then = (nodes[i].guard.IsOpaque() || scope.IsOpaque())
                                           ? scope
                                           : IntersectGuards(scope, nodes[i].guard);
          NormalizeNodes(nodes[i].children, s_then, pair_else_branches);
          if (need_renorm_else) {
            const RadixBoxGuard s_else = (nodes[i].guard.IsOpaque() || scope.IsOpaque())
                                             ? scope
                                             : SubtractGuards(scope, s_then).value_or(scope);
            NormalizeNodes(nodes[i].alternative, s_else, pair_else_branches);
          }
          changed = true;
          continue;
        }
        ++i;
      }
      if (changed) continue;

      // 4. NA-1: Non-adjacent equivalent guards (fixes Latent Bug L1 via CommutesAcross).
      for (size_t i = 0; i < nodes.size() && !changed; ++i) {
        if (nodes[i].leaf || nodes[i].guard.IsOpaque() || !nodes[i].alternative.empty() || scope.IsOpaque())
          continue;
        for (size_t j = i + 2; j < nodes.size(); ++j) {
          if (nodes[j].leaf || nodes[j].guard.IsOpaque() || !nodes[j].alternative.empty()) continue;
          if (!AreEquivalent(nodes[i], nodes[j], scope)) continue;
          const RadixBoxGuard s_then = (nodes[i].guard.IsOpaque() || scope.IsOpaque())
                                           ? scope
                                           : IntersectGuards(scope, nodes[i].guard);
          if (CommutesAcross(nodes[i].guard, nodes[i].children, nodes, i + 1, j, scope)) {
            nodes[j].children.insert(nodes[j].children.begin(),
                                     std::make_move_iterator(nodes[i].children.begin()),
                                     std::make_move_iterator(nodes[i].children.end()));
            nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
            NormalizeNodes(nodes[j - 1].children, s_then, pair_else_branches);
            changed = true;
            break;
          }
          if (CommutesAcross(nodes[j].guard, nodes[j].children, nodes, i + 1, j, scope)) {
            nodes[i].children.insert(nodes[i].children.end(),
                                     std::make_move_iterator(nodes[j].children.begin()),
                                     std::make_move_iterator(nodes[j].children.end()));
            nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(j));
            NormalizeNodes(nodes[i].children, s_then, pair_else_branches);
            changed = true;
            break;
          }
        }
      }
      if (changed) continue;

      // 5. Rule U2 & NA-2: Disjoint union of identical then-only bodies.
      for (size_t i = 0; i < nodes.size() && !changed; ++i) {
        if (nodes[i].leaf || nodes[i].guard.IsOpaque() || !nodes[i].alternative.empty() || scope.IsOpaque())
          continue;
        for (size_t j = i + 1; j < nodes.size(); ++j) {
          if (nodes[j].leaf || nodes[j].guard.IsOpaque() || !nodes[j].alternative.empty()) continue;
          if (!SameNodeList(nodes[i].children, nodes[j].children)) continue;
          if (j > i + 1 && !CommutesAcross(nodes[i].guard, nodes[i].children, nodes, i + 1, j, scope))
            continue;
          if (!AreDisjointGuardsInScope(nodes[i].guard, nodes[j].guard, scope)) continue;
          const auto u_opt =
              UnionGuards(IntersectGuards(scope, nodes[i].guard), IntersectGuards(scope, nodes[j].guard));
          if (!u_opt) continue;
          const RadixBoxGuard u_simp = SimplifyGuardGiven(*u_opt, scope);
          if (EquivalentGuardsInScope(u_simp, RadixBoxGuard::True(), scope)) {
            auto lifted = std::move(nodes[j].children);
            nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(j));
            nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(j),
                         std::make_move_iterator(lifted.begin()), std::make_move_iterator(lifted.end()));
            nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
            changed = true;
            break;
          }
          const bool single_box_or_ne =
              u_simp.Boxes().size() == 1 ||
              (!HasOpaqueDescendant(nodes[i]) && IsComplementShortcutNe(u_simp, scope, nodes[i].GetSpan()));
          if (!single_box_or_ne) continue;
          size_t old_cmd = 0, old_g = 0, old_e = 0;
          NodeSubtreeMetrics(nodes[i], scope, old_cmd, old_g, old_e);
          NodeSubtreeMetrics(nodes[j], scope, old_cmd, old_g, old_e);
          Node merged{u_simp, nullptr, nodes[j].span, nodes[j].children, {}};
          size_t new_cmd = 0, new_g = 0, new_e = 0;
          NodeSubtreeMetrics(merged, scope, new_cmd, new_g, new_e);
          if (new_cmd <= old_cmd && new_g < old_g && new_e <= old_e) {
            nodes[j] = std::move(merged);
            nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
            changed = true;
            break;
          }
        }
      }
      if (changed) continue;

      // 6. NA-4: Outer-axis factoring across 1-box then-only siblings.
      for (size_t i = 0; i < nodes.size() && !changed; ++i) {
        if (TryFactorNA4At(nodes, i, scope, pair_else_branches)) {
          changed = true;
          break;
        }
        if (nodes[i].leaf || nodes[i].guard.IsOpaque() || !nodes[i].alternative.empty() ||
            nodes[i].guard.Boxes().size() != 1 || nodes[i].guard.Boxes()[0].constraints.empty() ||
            scope.IsOpaque() || !nodes[i].guard.Arena())
          continue;
        const auto& box0 = nodes[i].guard.Boxes()[0];
        const AxisConstraint outer_c = box0.constraints[0];
        if ((box0.constraints.size() == 1 && box0.ranks.empty()) || RankUsesAxis(box0, outer_c.axis))
          continue;
        // Non-adjacent slide for NA-4: commit only if factoring actually fires and strictly improves.
        for (size_t j = i + 2; j < nodes.size(); ++j) {
          if (!MatchesOuterAxis(nodes[j], outer_c)) continue;
          if (CommutesAcross(nodes[i].guard, nodes[i].children, nodes, i + 1, j, scope)) {
            std::vector<Node> trial = nodes;
            Node moved = std::move(trial[i]);
            trial.erase(trial.begin() + static_cast<std::ptrdiff_t>(i));
            trial.insert(trial.begin() + static_cast<std::ptrdiff_t>(j) - 1, std::move(moved));
            size_t factor_start = j - 1;
            while (factor_start > 0 && MatchesOuterAxis(trial[factor_start - 1], outer_c)) --factor_start;
            if (TryFactorNA4At(trial, factor_start, scope, pair_else_branches)) {
              nodes = std::move(trial);
              changed = true;
              break;
            }
          }
          if (CommutesAcross(nodes[j].guard, nodes[j].children, nodes, i + 1, j, scope)) {
            std::vector<Node> trial = nodes;
            Node moved = std::move(trial[j]);
            trial.erase(trial.begin() + static_cast<std::ptrdiff_t>(j));
            trial.insert(trial.begin() + static_cast<std::ptrdiff_t>(i) + 1, std::move(moved));
            if (TryFactorNA4At(trial, i, scope, pair_else_branches)) {
              nodes = std::move(trial);
              changed = true;
              break;
            }
          }
        }
      }
      if (changed) continue;

      // 7. Complementary & m-way exhaustive partitions: adjacent prefix/suffix hoist,
      //    NA-3 non-adjacent partial prefix/suffix hoist & sliding, and two-stage else fusion.
      for (size_t i = 0; i < nodes.size() && !changed; ++i) {
        if (nodes[i].leaf || !nodes[i].alternative.empty()) continue;

        // 7A. Check adjacent m-way partition starting at i (m >= 2).
        if (!nodes[i].guard.IsOpaque() && !scope.IsOpaque() && i + 1 < nodes.size() && !nodes[i + 1].leaf &&
            !nodes[i + 1].guard.IsOpaque() && nodes[i + 1].alternative.empty()) {
          const auto arena = SharedArena(nodes[i].guard, scope);
          std::vector<RadixBoxGuard> effs{IntersectGuards(scope, nodes[i].guard)};
          RadixBoxGuard union_acc = effs.back();
          size_t end_idx = i;
          bool disjoint_ok = !effs.back().IsEmpty();
          for (size_t k = i + 1; k < nodes.size() && disjoint_ok; ++k) {
            if (nodes[k].leaf || nodes[k].guard.IsOpaque() || !nodes[k].alternative.empty()) break;
            const auto ek = IntersectGuards(scope, nodes[k].guard);
            if (!IntersectGuards(union_acc, ek).IsEmpty()) break;
            const auto u_next = UnionGuards(union_acc, ek);
            if (!u_next) break;
            union_acc = *u_next;
            effs.push_back(ek);
            end_idx = k;
            if (EquivalentGuardsInScope(union_acc, RadixBoxGuard::True(), scope)) {
              const size_t m = end_idx - i + 1;
              // Compute common prefix & suffix across all m arms.
              size_t min_len = nodes[i].children.size();
              for (size_t r = i + 1; r <= end_idx; ++r) min_len = std::min(min_len, nodes[r].children.size());
              size_t prefix = 0;
              while (prefix < min_len) {
                bool match = true;
                for (size_t r = i + 1; r <= end_idx; ++r) {
                  if (!SameNode(nodes[i].children[prefix], nodes[r].children[prefix])) {
                    match = false;
                    break;
                  }
                }
                if (!match) break;
                ++prefix;
              }
              size_t suffix = 0;
              while (prefix + suffix < min_len) {
                bool match = true;
                for (size_t r = i + 1; r <= end_idx; ++r) {
                  if (!SameNode(nodes[i].children[nodes[i].children.size() - 1 - suffix],
                                nodes[r].children[nodes[r].children.size() - 1 - suffix])) {
                    match = false;
                    break;
                  }
                }
                if (!match) break;
                ++suffix;
              }
              if (prefix > 0 || suffix > 0 || pair_else_branches) {
                std::vector<Node> replacement;
                for (size_t p = 0; p < prefix; ++p) replacement.push_back(nodes[i].children[p]);
                std::vector<Node> suf(nodes[i].children.end() - static_cast<std::ptrdiff_t>(suffix),
                                      nodes[i].children.end());
                std::vector<std::pair<RadixBoxGuard, std::vector<Node>>> non_empty_arms;
                for (size_t r = i; r <= end_idx; ++r) {
                  std::vector<Node> rem(nodes[r].children.begin() + static_cast<std::ptrdiff_t>(prefix),
                                        nodes[r].children.end() - static_cast<std::ptrdiff_t>(suffix));
                  if (!rem.empty()) non_empty_arms.emplace_back(nodes[r].guard, std::move(rem));
                }
                const bool suf_before_residual =
                    m == 2 && suffix > 0 && non_empty_arms.size() == 1 && suf.size() == 1 &&
                    SameKindDisjointCommands(non_empty_arms[0].second, suf.front());
                if (suf_before_residual) {
                  replacement.insert(replacement.end(), std::make_move_iterator(suf.begin()),
                                     std::make_move_iterator(suf.end()));
                }
                if (pair_else_branches && non_empty_arms.size() == m) {
                  // Build nested if / else if / ... / else chain, ensuring GuardCount decreases.
                  Node chain = Node{RadixBoxGuard::True(),
                                    nullptr,
                                    nodes[i].span,
                                    std::move(non_empty_arms.back().second),
                                    {}};
                  std::vector<Node> chain_nodes = std::move(chain.children);
                  for (size_t r = non_empty_arms.size() - 1; r-- > 0;) {
                    Node branch{non_empty_arms[r].first, nullptr, nodes[i].span,
                                std::move(non_empty_arms[r].second), std::move(chain_nodes)};
                    chain_nodes = {std::move(branch)};
                  }
                  replacement.insert(replacement.end(), std::make_move_iterator(chain_nodes.begin()),
                                     std::make_move_iterator(chain_nodes.end()));
                } else {
                  for (auto& [arm_g, arm_rem] : non_empty_arms) {
                    replacement.push_back(
                        Node{std::move(arm_g), nullptr, nodes[i].span, std::move(arm_rem), {}});
                  }
                }
                if (!suf_before_residual) {
                  replacement.insert(replacement.end(), std::make_move_iterator(suf.begin()),
                                     std::make_move_iterator(suf.end()));
                }
                nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i),
                            nodes.begin() + static_cast<std::ptrdiff_t>(end_idx) + 1);
                nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(i),
                             std::make_move_iterator(replacement.begin()),
                             std::make_move_iterator(replacement.end()));
                NormalizeNodes(nodes, scope, pair_else_branches);
                changed = true;
                break;
              }
            }
          }
          if (changed) break;
        }

        // 7B. Adjacent opaque complementary pair & NA-3 non-adjacent complementary pair.
        for (size_t j = i + 1; j < nodes.size(); ++j) {
          if (nodes[j].leaf || !nodes[j].alternative.empty()) continue;
          if (!AreComplementary(nodes[i], nodes[j], scope)) continue;
          if (j > i + 1 && (nodes[i].guard.IsOpaque() || nodes[j].guard.IsOpaque() || scope.IsOpaque()))
            continue;
          if (j == i + 1) {
            if (TryFuseAdjacent7BAt(nodes, i, scope, pair_else_branches)) {
              changed = true;
              break;
            }
          } else {
            // Non-adjacent complementary (NA-3):
            // 1) Whole-branch slide only if adjacent 7B hoist/fusion actually fires.
            if (CommutesAcross(nodes[i].guard, nodes[i].children, nodes, i + 1, j, scope)) {
              std::vector<Node> trial = nodes;
              Node moved = std::move(trial[i]);
              trial.erase(trial.begin() + static_cast<std::ptrdiff_t>(i));
              trial.insert(trial.begin() + static_cast<std::ptrdiff_t>(j) - 1, std::move(moved));
              if (TryFuseAdjacent7BAt(trial, j - 1, scope, pair_else_branches)) {
                nodes = std::move(trial);
                changed = true;
                break;
              }
            }
            if (CommutesAcross(nodes[j].guard, nodes[j].children, nodes, i + 1, j, scope)) {
              std::vector<Node> trial = nodes;
              Node moved = std::move(trial[j]);
              trial.erase(trial.begin() + static_cast<std::ptrdiff_t>(j));
              trial.insert(trial.begin() + static_cast<std::ptrdiff_t>(i) + 1, std::move(moved));
              if (TryFuseAdjacent7BAt(trial, i, scope, pair_else_branches)) {
                nodes = std::move(trial);
                changed = true;
                break;
              }
            }
            // 2) Partial prefix/suffix hoist across commuting gap even if residuals don't commute.
            const auto min_len = std::min(nodes[i].children.size(), nodes[j].children.size());
            size_t prefix = 0;
            while (prefix < min_len && SameNode(nodes[i].children[prefix], nodes[j].children[prefix]))
              ++prefix;
            size_t suffix = 0;
            while (prefix + suffix < min_len &&
                   SameNode(nodes[i].children[nodes[i].children.size() - 1 - suffix],
                            nodes[j].children[nodes[j].children.size() - 1 - suffix]))
              ++suffix;
            if (prefix > 0) {
              std::vector<Node> pref(nodes[i].children.begin(),
                                     nodes[i].children.begin() + static_cast<std::ptrdiff_t>(prefix));
              if (CommutesAcross(nodes[j].guard, pref, nodes, i + 1, j, scope)) {
                nodes[i].children.erase(nodes[i].children.begin(),
                                        nodes[i].children.begin() + static_cast<std::ptrdiff_t>(prefix));
                nodes[j].children.erase(nodes[j].children.begin(),
                                        nodes[j].children.begin() + static_cast<std::ptrdiff_t>(prefix));
                nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(i),
                             std::make_move_iterator(pref.begin()), std::make_move_iterator(pref.end()));
                changed = true;
                break;
              }
            }
            if (suffix > 0) {
              std::vector<Node> suf(nodes[i].children.end() - static_cast<std::ptrdiff_t>(suffix),
                                    nodes[i].children.end());
              if (CommutesAcross(nodes[i].guard, suf, nodes, i + 1, j, scope)) {
                nodes[i].children.erase(nodes[i].children.end() - static_cast<std::ptrdiff_t>(suffix),
                                        nodes[i].children.end());
                nodes[j].children.erase(nodes[j].children.end() - static_cast<std::ptrdiff_t>(suffix),
                                        nodes[j].children.end());
                nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(j) + 1,
                             std::make_move_iterator(suf.begin()), std::make_move_iterator(suf.end()));
                changed = true;
                break;
              }
            }
          }
        }
      }
    }
  }

  void Append(std::vector<Node>& nodes, const ControlPtr& statement) {
    ++visits_;
    if (!statement->command && statement->guard.IsTautology()) {
      for (const auto& child : statement->body) Append(nodes, child);
      return;
    }
    auto span_ptr = std::make_shared<const Span>(statement->span);
    if (statement->command) {
      nodes.push_back(Node{RadixBoxGuard::True(), statement, span_ptr, {}, {}});
      return;
    }
    if (statement->guard.IsOpaque()) {
      const auto& expr = statement->guard.OpaqueExpr();
      if (!statement->predicate_qualified ||
          (!statement->alternative.empty() && !PureCommandExpression(expr, &expressions_))) {
        nodes.push_back(Node{RadixBoxGuard::True(), statement, span_ptr, {}, {}});
        return;
      }
    }
    nodes.push_back(Node{statement->guard, nullptr, span_ptr, {}, {}});
    for (const auto& child : statement->body) Append(nodes.back().children, child);
    for (const auto& child : statement->alternative) Append(nodes.back().alternative, child);
  }

  static ControlForest Freeze(const std::vector<Node>& nodes, PhysicalSyncRealization& work) {
    ControlForest result;
    for (const auto& node : nodes) {
      if (node.leaf) {
        result.push_back(node.leaf);
      } else if (node.guard.IsTautology()) {
        auto flat = Freeze(node.children, work);
        result.insert(result.end(), std::make_move_iterator(flat.begin()),
                      std::make_move_iterator(flat.end()));
      } else {
        const auto& sp = node.GetSpan();
        std::optional<ControlPtr> alternative;
        if (!node.alternative.empty())
          alternative = ControlSequence(Freeze(node.alternative, work), sp, work);
        result.push_back(ControlBranch(node.guard, ControlSequence(Freeze(node.children, work), sp, work),
                                       alternative, sp, work));
      }
    }
    return result;
  }

 public:
  ControlForest Build(const ControlForest& commands, PhysicalSyncRealization& result, bool endpoint,
                      bool pair_else_branches, const RadixBoxGuard& scope) {
    for (const auto& command : commands) Append(roots_, command);
    const auto guards_before = CountNodeGuards(roots_, scope);
    NormalizeNodes(roots_, scope, pair_else_branches);
    const auto guards_after = CountNodeGuards(roots_, scope);
    const size_t saved = guards_before > guards_after ? guards_before - guards_after : 0;
    if (endpoint) {
      INTERNAL_CHECK(result.guard_sites >= saved) << "Internal error: command guard accounting";
      result.guard_sites -= saved;
      result.shared_guard_sites += saved;
    } else {
      result.shared_composed_guard_sites += saved;
    }
    result.control_expression_visits += expressions_;
    result.control_statement_visits += visits_;
    if (!endpoint && saved == 0) {
      // Check if tree is already identical to input commands; if normalized changed structure (e.g.
      // dead/tautology or constraint drop or pairing), rebuild; otherwise reuse descriptors.
      auto rebuilt = Freeze(roots_, result);
      if (rebuilt.size() == commands.size() &&
          std::equal(rebuilt.begin(), rebuilt.end(), commands.begin(),
                     [](const ControlPtr& a, const ControlPtr& b) { return SameControl(a, b); }))
        return commands;
      return rebuilt;
    }
    return Freeze(roots_, result);
  }
};

}  // namespace

void AppendCase(std::vector<CaseCommands>& cases, std::pair<int64_t, int64_t> blocks,
                const ControlForest& commands, const Span& span, PhysicalSyncRealization& work) {
  if (commands.empty()) return;
  auto body = ControlSequence(commands, span, work);
  if (!cases.empty() && cases.back().blocks.second < std::numeric_limits<int64_t>::max() &&
      cases.back().blocks.second + 1 == blocks.first && SameControl(cases.back().body, body)) {
    cases.back().blocks.second = blocks.second;
  } else {
    cases.push_back({blocks, body});
  }
}

Intervals SubtractIntervals(const Intervals& domain, const Intervals& absent) {
  Intervals result;
  size_t j = 0;
  for (auto [lo, hi] : domain) {
    while (j < absent.size() && absent[j].second < lo) ++j;
    for (size_t k = j; k < absent.size() && absent[k].first <= hi; ++k) {
      if (lo < absent[k].first) result.emplace_back(lo, absent[k].first - 1);
      lo = std::max(lo, absent[k].second + 1);
      if (lo > hi) break;
    }
    if (lo <= hi) result.emplace_back(lo, hi);
  }
  return result;
}

namespace {

bool DomainIntersects(const Intervals& domain, int64_t lo, int64_t hi) {
  if (lo > hi) return false;
  const auto next =
      std::lower_bound(domain.begin(), domain.end(), lo,
                       [](const auto& interval, int64_t value) { return interval.second < value; });
  return next != domain.end() && next->first <= hi;
}

std::vector<CaseCommands> CoalesceCutCases(const std::vector<CaseCommands>& source, const Intervals& domain,
                                           PhysicalSyncRealization& work) {
  std::vector<CaseCommands> result;
  for (const auto& item : source) {
    ++work.cut_case_comparisons;
    if (!result.empty() &&
        !DomainIntersects(domain, result.back().blocks.second + 1, item.blocks.first - 1) &&
        SameControl(result.back().body, item.body)) {
      result.back().blocks.second = item.blocks.second;
      ++work.coalesced_cut_cases;
    } else {
      result.push_back(item);
    }
  }
  return result;
}

}  // namespace

void FlattenControlSequence(const ControlPtr& node, ControlForest& commands, PhysicalSyncRealization& work) {
  ++work.case_sequence_visits;
  if (!node->command && node->guard.IsTautology()) {
    for (const auto& child : node->body) FlattenControlSequence(child, commands, work);
  } else {
    commands.push_back(node);
  }
}

namespace {

RadixBoxGuard LaunchIntervalGuard(const std::shared_ptr<GuardDomainArena>& arena, AxisId block_axis,
                                  int64_t ctx_lo, int64_t lo, int64_t hi) {
  const int64_t count = arena->Axis(block_axis).count;
  const auto ap = AxisAP::MakeCanonical(lo - ctx_lo, hi - ctx_lo + 1, 1, count);
  if (!ap) return RadixBoxGuard::False();
  if (ap->IsFull(count)) return RadixBoxGuard::True();
  RadixBox box;
  box.constraints.push_back(AxisConstraint{block_axis, *ap});
  return RadixBoxGuard::FromBox(arena, std::move(box));
}

RadixBoxGuard LaunchIntervalsGuard(const std::shared_ptr<GuardDomainArena>& arena, AxisId block_axis,
                                   int64_t ctx_lo, const Intervals& intervals) {
  const int64_t count = arena->Axis(block_axis).count;
  std::vector<RadixBox> boxes;
  for (const auto& [lo, hi] : intervals) {
    if (const auto ap = AxisAP::MakeCanonical(lo - ctx_lo, hi - ctx_lo + 1, 1, count)) {
      if (ap->IsFull(count)) return RadixBoxGuard::True();
      RadixBox box;
      box.constraints.push_back(AxisConstraint{block_axis, *ap});
      boxes.push_back(std::move(box));
    }
  }
  return RadixBoxGuard::FromBoxes(arena, std::move(boxes));
}

class CaseSequenceTree {
  struct Slice {
    int64_t last;
    ControlForest commands;
    size_t begin = 0, end = 0;
  };
  std::vector<Slice> slices_;
  std::shared_ptr<GuardDomainArena> arena_;
  AxisId block_axis_;
  int64_t ctx_lo_, ctx_hi_;
  Span span_;
  PhysicalSyncRealization& work_;

  void Append(int64_t first, int64_t last, const ControlPtr& body, const Intervals& possible) {
    if (!DomainIntersects(possible, first, last)) return;
    Slice slice{last, {}};
    if (body) FlattenControlSequence(body, slice.commands, work_);
    slice.end = slice.commands.size();
    slices_.push_back(std::move(slice));
  }

  bool Common(size_t first, size_t last, bool suffix) {
    const auto& base = slices_[first];
    if (base.begin == base.end) return false;
    const auto& command = base.commands[suffix ? base.end - 1 : base.begin];
    for (size_t i = first + 1; i < last; ++i) {
      ++work_.case_sequence_visits;
      const auto& slice = slices_[i];
      if (slice.begin == slice.end) return false;
      ++work_.case_sequence_comparisons;
      if (!SameControl(command, slice.commands[suffix ? slice.end - 1 : slice.begin])) return false;
    }
    return true;
  }

  ControlPtr Body(ControlForest commands) const { return ControlSequence(std::move(commands), span_, work_); }

  ControlForest Build(size_t first, size_t last) {
    if (first == last) return {};
    if (last - first == 1) {
      const auto& slice = slices_[first];
      return {slice.commands.begin() + slice.begin, slice.commands.begin() + slice.end};
    }
    ControlForest result, suffix;
    while (Common(first, last, false)) {
      result.push_back(slices_[first].commands[slices_[first].begin]);
      for (size_t i = first; i < last; ++i) ++slices_[i].begin;
    }
    while (Common(first, last, true)) {
      suffix.push_back(slices_[first].commands[slices_[first].end - 1]);
      for (size_t i = first; i < last; ++i) --slices_[i].end;
    }
    const auto middle = first + (last - first) / 2;
    auto left = Build(first, middle);
    auto right = Build(middle, last);
    const int64_t split_last = slices_[middle - 1].last;
    if (!left.empty()) {
      auto guard = LaunchIntervalGuard(arena_, block_axis_, ctx_lo_, ctx_lo_, split_last);
      std::optional<ControlPtr> alternative;
      if (!right.empty()) alternative = Body(std::move(right));
      result.push_back(ControlBranch(std::move(guard), Body(std::move(left)), alternative, span_, work_));
    } else if (!right.empty()) {
      auto guard = LaunchIntervalGuard(arena_, block_axis_, ctx_lo_, split_last + 1, ctx_hi_);
      result.push_back(ControlBranch(std::move(guard), Body(std::move(right)), std::nullopt, span_, work_));
    }
    result.insert(result.end(), suffix.rbegin(), suffix.rend());
    return result;
  }

 public:
  CaseSequenceTree(std::shared_ptr<GuardDomainArena> arena, AxisId block_axis, int64_t ctx_lo, int64_t ctx_hi,
                   const Span& span, PhysicalSyncRealization& work)
      : arena_(std::move(arena)),
        block_axis_(block_axis),
        ctx_lo_(ctx_lo),
        ctx_hi_(ctx_hi),
        span_(span),
        work_(work) {}

  ControlForest Build(const std::vector<CaseCommands>& cases, const Intervals& possible) {
    const auto end = possible.back().second;
    auto next = possible.front().first;
    for (const auto& item : cases) {
      ++work_.case_sequence_visits;
      const auto first = std::max(next, item.blocks.first);
      const auto last = std::min(end, item.blocks.second);
      if (first > last) continue;
      Append(next, first - 1, nullptr, possible);
      Append(first, last, item.body, possible);
      next = last + 1;
    }
    Append(next, end, nullptr, possible);
    return Build(0, slices_.size());
  }
};

}  // namespace

void CaseCodeSize::Visit(const ControlPtr& node, const RadixBoxGuard& scope) {
  if (node->command) {
    ++statements;
    ++commands;
    expressions += ExprNodeCount(node->command);
    return;
  }
  if (node->guard.IsTautology()) {
    ++statements;
    for (const auto& child : node->body) Visit(child, scope);
    return;
  }
  if (node->guard.IsEmpty()) {
    ++statements;
    for (const auto& child : node->alternative) Visit(child, scope);
    return;
  }
  const auto gm = AnalyticGuardMetrics(node->guard, scope, !node->alternative.empty(), node->span);
  guards += gm.guards;
  expressions += gm.expressions;
  statements += gm.statements;
  const RadixBoxGuard s_then =
      (node->guard.IsOpaque() || scope.IsOpaque()) ? scope : IntersectGuards(scope, node->guard);
  const RadixBoxGuard s_else = (node->guard.IsOpaque() || scope.IsOpaque() || node->alternative.empty())
                                   ? scope
                                   : SubtractGuards(scope, s_then).value_or(scope);
  for (const auto& child : node->body) Visit(child, s_then);
  for (const auto& child : node->alternative) Visit(child, s_else);
}

CaseCodeSize::CaseCodeSize(const ControlForest& body, const RadixBoxGuard& scope) {
  for (const auto& node : body) Visit(node, scope);
}

bool CaseCodeSize::Improves(const CaseCodeSize& old) const {
  return commands <= old.commands && guards <= old.guards && expressions <= old.expressions &&
         (commands < old.commands || guards < old.guards || expressions < old.expressions);
}

namespace {

class UniformCommandBody : public IRVisitor {
  const std::set<const Var*>& variables_;

 public:
  bool valid = true;
  explicit UniformCommandBody(const std::set<const Var*>& variables) : variables_(variables) {}
  void VisitExpr_(const VarPtr& variable) override { valid &= variables_.count(variable.get()) != 0; }
  void VisitStmt_(const AssignStmtPtr&) override { valid = false; }
};

class CaseDispatchCost {
  const ExprPtr& block_;
  int64_t value_;
  size_t cost_ = 0;
  bool valid_ = true;

  std::optional<bool> Condition(const ExprPtr& expression) {
    if (auto known = As<ConstBool>(expression)) return known->value_;
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return std::nullopt;
    if (As<And>(expression) || As<Or>(expression)) {
      auto left = Condition(binary->left_), right = Condition(binary->right_);
      if (!left || !right) return std::nullopt;
      ++cost_;
      return As<And>(expression) ? *left && *right : *left || *right;
    }
    const auto bound = As<ConstInt>(binary->right_);
    if (binary->left_ != block_ || !bound) return std::nullopt;
    ++cost_;
    if (As<Le>(expression)) return value_ <= bound->value_;
    if (As<Lt>(expression)) return value_ < bound->value_;
    if (As<Ge>(expression)) return value_ >= bound->value_;
    if (As<Gt>(expression)) return value_ > bound->value_;
    if (As<Eq>(expression)) return value_ == bound->value_;
    if (As<Ne>(expression)) return value_ != bound->value_;
    valid_ = false;
    return std::nullopt;
  }

  void Visit(const FrozenCommand& cmd) {
    if (!cmd.condition_expr) {
      if (cmd.body)
        for (const auto& child : cmd.body->commands) Visit(child);
      return;
    }
    const auto saved = cost_;
    auto selected = Condition(cmd.condition_expr);
    if (!selected) {
      cost_ = saved;
      return;
    }
    ++cost_;
    const auto& branch = *selected ? cmd.body : cmd.alternative;
    if (branch)
      for (const auto& child : branch->commands) Visit(child);
  }

 public:
  CaseDispatchCost(const ExprPtr& block, int64_t value) : block_(block), value_(value) {}
  std::optional<size_t> Measure(const ControlForest& body, const RadixBoxGuard& scope,
                                PhysicalSyncRealization& work) {
    const auto frozen = FreezeCommandForest(body, work, scope);
    for (const auto& cmd : frozen->commands) Visit(cmd);
    return valid_ ? std::optional<size_t>(cost_) : std::nullopt;
  }
};

ControlForest ShareDisjointSequence(const std::shared_ptr<GuardDomainArena>& arena, AxisId block_axis,
                                    const ExprPtr& block, const Intervals& possible,
                                    const std::vector<CaseCommands>& cases, const ControlForest& incumbent,
                                    const std::set<const Var*>& uniform_variables,
                                    PhysicalSyncRealization& work, const Span& span) {
  if (cases.size() != 2 || !block || !SameControl(cases[0].body, cases[1].body)) return incumbent;
  UniformCommandBody uniform(uniform_variables);
  VisitControlExpressions(cases[0].body, uniform);
  if (!uniform.valid) return incumbent;
  const auto context = std::make_pair(possible.front().first, possible.back().second);
  Intervals selected_intervals;
  std::set<int64_t> boundaries{context.first, context.second + 1};
  for (const auto& item : cases) {
    const auto lo = std::max(context.first, item.blocks.first);
    const auto hi = std::min(context.second, item.blocks.second);
    if (lo > hi) return incumbent;
    selected_intervals.emplace_back(lo, hi);
    boundaries.insert(lo);
    boundaries.insert(hi + 1);
  }
  const auto excluded = SubtractIntervals(possible, selected_intervals);
  if (excluded.empty()) return incumbent;
  RadixBoxGuard predicate_guard = RadixBoxGuard::False();
  if (excluded.size() == 1) {
    int64_t ex_lo =
        DomainIntersects(possible, context.first, excluded[0].first - 1) ? excluded[0].first : context.first;
    int64_t ex_hi = DomainIntersects(possible, excluded[0].second + 1, context.second) ? excluded[0].second
                                                                                       : context.second;
    if (ex_lo > context.first && ex_hi < context.second && ex_lo < ex_hi) {
      ex_lo = std::min(ex_lo, selected_intervals[0].second + 1);
      ex_hi = std::max(ex_hi, selected_intervals[1].first - 1);
    }
    const auto hole = LaunchIntervalGuard(arena, block_axis, context.first, ex_lo, ex_hi);
    predicate_guard =
        SubtractGuards(RadixBoxGuard::True(), hole)
            .value_or(LaunchIntervalsGuard(arena, block_axis, context.first, selected_intervals));
  } else {
    if (!DomainIntersects(possible, context.first, selected_intervals.front().first - 1)) {
      selected_intervals.front().first = context.first;
    }
    if (!DomainIntersects(possible, selected_intervals.back().second + 1, context.second)) {
      selected_intervals.back().second = context.second;
    }
    predicate_guard = LaunchIntervalsGuard(arena, block_axis, context.first, selected_intervals);
  }
  if (predicate_guard.IsEmpty() || predicate_guard.IsTautology()) return incumbent;
  ControlForest candidate{ControlBranch(predicate_guard, cases[0].body, std::nullopt, span, work)};
  const CaseCodeSize old_size(incumbent), new_size(candidate);
  work.case_sequence_visits += old_size.statements + new_size.statements;
  work.control_expression_visits += old_size.expressions + new_size.expressions;
  if (!new_size.Improves(old_size)) return incumbent;
  std::array<size_t, 2> old_max{}, new_max{};
  for (auto it = boundaries.begin(), next = std::next(it); next != boundaries.end(); ++it, ++next) {
    if (!DomainIntersects(possible, *it, *next - 1)) continue;
    const bool active = (*it >= cases[0].blocks.first && *it <= cases[0].blocks.second) ||
                        (*it >= cases[1].blocks.first && *it <= cases[1].blocks.second);
    const auto before = CaseDispatchCost(block, *it).Measure(incumbent, RadixBoxGuard::True(), work);
    const auto after = CaseDispatchCost(block, *it).Measure(candidate, RadixBoxGuard::True(), work);
    if (!before || !after) return incumbent;
    old_max[active] = std::max(old_max[active], *before);
    new_max[active] = std::max(new_max[active], *after);
  }
  if (new_max[0] > old_max[0] || new_max[1] > old_max[1]) return incumbent;
  ++work.shared_disjoint_case_cuts;
  return candidate;
}

ControlForest ShareResidualSequence(const std::shared_ptr<GuardDomainArena>& arena, AxisId block_axis,
                                    const ExprPtr& block, const Intervals& possible,
                                    const std::vector<CaseCommands>& cases, const ControlForest& incumbent,
                                    const std::set<const Var*>& uniform_variables,
                                    PhysicalSyncRealization& work, const Span& span) {
  if (cases.size() != 3 || !block) return incumbent;
  Intervals covered;
  std::array<ControlForest, 3> bodies;
  UniformCommandBody uniform(uniform_variables);
  for (size_t i = 0; i < 3; ++i) {
    covered.push_back(cases[i].blocks);
    VisitControlExpressions(cases[i].body, uniform);
    FlattenControlSequence(cases[i].body, bodies[i], work);
  }
  if (!uniform.valid || !SubtractIntervals(possible, covered).empty()) return incumbent;
  size_t prefix = 0, suffix = 0;
  auto same = [&](size_t i, bool back) {
    if (prefix + suffix >= bodies[i].size()) return false;
    ++work.case_sequence_comparisons;
    return SameControl(bodies[0][back ? bodies[0].size() - 1 - suffix : prefix],
                       bodies[i][back ? bodies[i].size() - 1 - suffix : prefix]);
  };
  while (prefix + suffix < bodies[0].size() && same(1, false) && same(2, false)) ++prefix;
  while (prefix + suffix < bodies[0].size() && same(1, true) && same(2, true)) ++suffix;
  if (prefix + suffix != bodies[1].size() || prefix + suffix >= bodies[0].size() ||
      prefix + suffix >= bodies[2].size())
    return incumbent;
  std::vector<CaseCommands> residual;
  for (size_t i : {size_t{0}, size_t{2}})
    residual.push_back({cases[i].blocks,
                        ControlSequence({bodies[i].begin() + prefix, bodies[i].end() - suffix}, span, work)});
  const auto previous = work.shared_disjoint_case_cuts;
  auto middle = ShareDisjointSequence(arena, block_axis, block, possible, residual, incumbent,
                                      uniform_variables, work, span);
  if (work.shared_disjoint_case_cuts == previous) return incumbent;
  ControlForest candidate(bodies[0].begin(), bodies[0].begin() + prefix);
  candidate.insert(candidate.end(), middle.begin(), middle.end());
  candidate.insert(candidate.end(), bodies[0].end() - suffix, bodies[0].end());
  const CaseCodeSize old_size(incumbent), new_size(candidate);
  work.case_sequence_visits += old_size.statements + new_size.statements;
  work.control_expression_visits += old_size.expressions + new_size.expressions;
  if (!new_size.Improves(old_size)) {
    work.shared_disjoint_case_cuts = previous;
    return incumbent;
  }
  return candidate;
}

}  // namespace

ControlForest ScheduleCommandCases(const ExprPtr& block, const Intervals& possible,
                                   const std::vector<CaseCommands>& cases, PhysicalSyncRealization& work,
                                   const Span& span, const std::set<const Var*>& uniform_variables,
                                   const std::shared_ptr<GuardDomainArena>& supplied_arena) {
  ControlForest target;
  if (possible.empty()) return target;
  int64_t previous = -1;
  for (const auto& [lo, hi] : possible) {
    CHECK(lo > previous && hi >= lo && hi < std::numeric_limits<int64_t>::max())
        << "Invalid cut execution domain";
    previous = hi;
  }
  previous = -1;
  for (const auto& item : cases) {
    CHECK(item.blocks.first > previous && item.blocks.second >= item.blocks.first &&
          item.blocks.second < std::numeric_limits<int64_t>::max())
        << "Invalid ordered command cases";
    previous = item.blocks.second;
  }
  const auto context = std::make_pair(possible.front().first, possible.back().second);
  const auto coalesced = CoalesceCutCases(cases, possible, work);
  auto block_var = As<Var>(block);
  if (block && !block_var) {
    // Diagnostic non-Var block fallback (cannot occur in production).
    for (const auto& item : coalesced) {
      const auto selected = std::make_pair(std::max(context.first, item.blocks.first),
                                           std::min(context.second, item.blocks.second));
      if (selected.first > selected.second) continue;
      const auto guard = SyncInvocationGuard(block, context, selected, span);
      target.push_back(guard ? ControlBranch(guard, item.body, std::nullopt, span, work) : item.body);
    }
    return CommandGuards().Build(target, work, false, false, RadixBoxGuard::True());
  }

  if (!block_var) {
    for (const auto& item : coalesced) {
      const auto selected = std::make_pair(std::max(context.first, item.blocks.first),
                                           std::min(context.second, item.blocks.second));
      if (selected.first > selected.second) continue;
      CHECK(selected == context) << "Original dominating invocation block binding is unavailable";
      target.push_back(item.body);
    }
    return CommandGuards().Build(target, work, false, false, RadixBoxGuard::True());
  }

  auto arena = supplied_arena ? supplied_arena : std::make_shared<GuardDomainArena>();
  const AxisId block_axis = arena->RegisterLaunchBlock(block_var, context);

  std::optional<int64_t> prev_hi;
  for (size_t idx = 0; idx < coalesced.size(); ++idx) {
    const auto& item = coalesced[idx];
    int64_t lo = std::max(context.first, item.blocks.first);
    int64_t hi = std::min(context.second, item.blocks.second);
    if (lo > hi) continue;
    if (idx == 0 && !DomainIntersects(possible, context.first, lo - 1)) {
      lo = context.first;
    } else if (prev_hi && !DomainIntersects(possible, *prev_hi + 1, lo - 1)) {
      lo = *prev_hi + 1;
    }
    if (idx + 1 == coalesced.size() && !DomainIntersects(possible, hi + 1, context.second)) {
      hi = context.second;
    }
    prev_hi = hi;
    auto guard = LaunchIntervalGuard(arena, block_axis, context.first, lo, hi);
    target.push_back(!guard.IsTautology()
                         ? ControlBranch(std::move(guard), item.body, std::nullopt, span, work)
                         : item.body);
  }
  if (coalesced.size() > 1) {
    auto candidate = CaseSequenceTree(arena, block_axis, context.first, context.second, span, work)
                         .Build(coalesced, possible);
    const CaseCodeSize old_size(target), new_size(candidate);
    work.case_sequence_visits += old_size.statements + new_size.statements;
    work.control_expression_visits += old_size.expressions + new_size.expressions;
    if (new_size.Improves(old_size)) {
      ++work.factored_case_cuts;
      target = std::move(candidate);
    }
  }
  target = ShareDisjointSequence(arena, block_axis, block, possible, coalesced, target, uniform_variables,
                                 work, span);
  target = ShareResidualSequence(arena, block_axis, block, possible, coalesced, target, uniform_variables,
                                 work, span);
  return CommandGuards().Build(target, work, false, false, RadixBoxGuard::True());
}

ControlForest ShareCommandGuards(const ControlForest& commands, PhysicalSyncRealization& work, bool endpoint,
                                 bool pair_else_branches, const RadixBoxGuard& scope) {
  return CommandGuards().Build(commands, work, endpoint, pair_else_branches, scope);
}

}  // namespace physical
}  // namespace pypto::ir::sync
