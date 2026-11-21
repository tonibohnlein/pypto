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

#include "src/ir/transforms/utils/sync_loop_trip_proof.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_acc_trim_analysis.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace {

constexpr int kMaxBoolAliasDepth = 8;
constexpr __int128 kBoundNegInf = arith::ConstIntBound::kNegInf;
constexpr __int128 kBoundPosInf = arith::ConstIntBound::kPosInf;

struct Bound128 {
  __int128 lo = 0;
  __int128 hi = 0;
  bool is_const() const { return lo == hi; }
};

bool IsSignedIntOrIndex(const ExprPtr& expr) {
  const auto type = expr ? As<ScalarType>(expr->GetType()) : nullptr;
  return type && (type->dtype_ == DataType::INDEX || type->dtype_.IsSignedInt());
}

bool Is64BitSignedOrIndex(const ExprPtr& expr) {
  const auto type = expr ? As<ScalarType>(expr->GetType()) : nullptr;
  return type && (type->dtype_ == DataType::INDEX || type->dtype_.GetBit() >= 64);
}

std::pair<__int128, __int128> ScalarSignedLimits(const ExprPtr& expr) {
  const auto type = expr ? As<ScalarType>(expr->GetType()) : nullptr;
  if (!type || type->dtype_ == DataType::INDEX || type->dtype_.GetBit() >= 64) {
    return {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
  }
  const int bits = type->dtype_.GetBit();
  if (bits <= 1) return {0, 0};
  const __int128 hi = (__int128{1} << (bits - 1)) - 1;
  const __int128 lo = -hi - 1;
  return {lo, hi};
}

std::optional<Bound128> ClampToType(const ExprPtr& expr, __int128 lo, __int128 hi) {
  const auto [tmin, tmax] = ScalarSignedLimits(expr);
  if (lo < tmin || hi > tmax || lo > hi) return std::nullopt;
  return Bound128{lo, hi};
}

arith::ConstIntBound ToConstIntBound(const Bound128& b) {
  const __int128 lo = std::max(b.lo, kBoundNegInf);
  const __int128 hi = std::min(b.hi, kBoundPosInf);
  return arith::ConstIntBound{static_cast<int64_t>(std::min(lo, hi)), static_cast<int64_t>(hi)};
}

ExprPtr MakeConstInt(int64_t val, DataType dtype, const Span& span = Span::unknown()) {
  return std::make_shared<ConstInt>(val, dtype, span);
}

}  // namespace

class SyncControlRangeProver::Impl {
 public:
  Impl(const FunctionPtr& function, const SyncParameterBounds& parameters,
       const std::optional<std::pair<int64_t, int64_t>>& blocks, ScalarBoundOperands scalar_bound_operands,
       const std::optional<SyncDispatchCoordinates>& dispatch)
      : function_(function),
        parameters_(parameters),
        blocks_(blocks),
        dispatch_(dispatch),
        scalar_bound_operands_(scalar_bound_operands) {
    if (!function || !function->body_) return;
    analyzer_ = std::make_shared<arith::Analyzer>();
    InitParametersAndScopes();
  }

  SyncControlRangeWork Work() const {
    auto work = work_;
    work.fallback_definitions = fallback_definitions_.load(std::memory_order_relaxed);
    return work;
  }

  std::optional<std::pair<int64_t, int64_t>> LookupScalarBounds(const Stmt* cut, const Expr* expr) const {
    const auto site = scalar_bounds_.find(cut);
    if (site == scalar_bounds_.end() || ambiguous_cuts_.count(cut)) {
      return std::nullopt;
    }
    const auto found = site->second.find(expr);
    if (found == site->second.end() || !found->second) {
      return std::nullopt;
    }
    return std::pair{static_cast<int64_t>(found->second->lo), static_cast<int64_t>(found->second->hi)};
  }

  bool IsPositiveTrip(const ForStmt* loop) const {
    const auto it = loop_proofs_.find(loop);
    return it != loop_proofs_.end() && it->second.is_positive_trip;
  }

  bool IsAtMostOneTrip(const ForStmt* loop) const {
    const auto it = loop_proofs_.find(loop);
    return it != loop_proofs_.end() && it->second.is_at_most_one_trip;
  }

  bool HasSafeNonnegativeLoopBounds(const ForStmt* loop) const {
    const auto it = loop_proofs_.find(loop);
    return it != loop_proofs_.end() && !ambiguous_cuts_.count(loop) && it->second.has_safe_bounds;
  }

  bool IsPositiveTripCompatible(const ForStmt* loop) const {
    if (!loop) return false;
    if (const auto it = loop_proofs_.find(loop); it != loop_proofs_.end()) {
      return it->second.is_positive_trip;
    }
    const auto vit = loops_by_var_.find(loop->loop_var_.get());
    if (vit != loops_by_var_.end()) {
      bool matched_any = false;
      bool all_positive = true;
      for (const ForStmt* orig : vit->second) {
        if (structural_equal(orig->start_, loop->start_) && structural_equal(orig->stop_, loop->stop_) &&
            structural_equal(orig->step_, loop->step_)) {
          matched_any = true;
          if (!IsPositiveTrip(orig)) {
            all_positive = false;
            break;
          }
        }
      }
      if (matched_any) return all_positive;
    }
    if (loop->kind_ == ForKind::Sequential) {
      const auto s = As<ConstInt>(loop->start_);
      const auto e = As<ConstInt>(loop->stop_);
      const auto st = As<ConstInt>(loop->step_);
      if (s && e && st && st->value_ > 0 && e->value_ > s->value_) return true;
    }
    return false;
  }

  std::optional<SymbolicIntRange> LookupAtCut(const Stmt* cut, const ExprPtr& expr) const {
    if (!expr || !IsSignedIntOrIndex(expr) || !analyzer_) return std::nullopt;
    Cut target{0, clock_};
    if (cut) {
      const auto it = cuts_.find(cut);
      if (it == cuts_.end() || ambiguous_cuts_.count(cut)) {
        return std::nullopt;
      }
      target = it->second;
      if (const auto stored = footprint_ranges_.find(cut); stored != footprint_ranges_.end()) {
        for (const auto& result : stored->second) {
          if (result.expression == expr.get()) {
            return result.range;
          }
        }
      }
    }
    const size_t target_scope = target.scope;
    std::vector<size_t> chain;
    for (size_t scope = target_scope; scope != 0; scope = scopes_[scope].parent) {
      chain.push_back(scope);
    }

    // Non-footprint queries (e.g. a retained child's enclosing coordinate) can
    // arrive at any cut. Recreate only the scalar dependencies of the expression
    // and its dominating control facts, in their original order.
    auto scratch = std::make_shared<arith::Analyzer>();
    std::unordered_set<const Var*> scratch_ready;
    std::unordered_map<const Var*, ExprPtr> scratch_aliases;
    std::vector<const ForStmt*> scratch_loops;
    std::vector<arith::ConstraintContext> scratch_contexts;
    auto dependencies = CollectDependencies(expr, target, chain);

    for (const auto& var : dependencies.parameters) {
      const auto b = root_bindings_.at(var.get());
      scratch_ready.insert(var.get());
      if (b.is_const() && b.min_value > kBoundNegInf && b.max_value < kBoundPosInf) {
        scratch->Bind(var, MakeConstInt(b.min_value, GetScalarDtype(var)));
      }
      scratch->const_int_bound.Update(var, b);
    }

    auto next = dependencies.definitions.begin();
    auto bind_before = [&](size_t before) {
      for (; next != dependencies.definitions.end() && (*next)->position < before; ++next) {
        const auto& def = **next;
        scratch_ready.insert(def.var.get());
        if (def.is_bool_alias) {
          scratch_aliases.emplace(def.var.get(), def.value);
          continue;
        }
        if (def.non_overflow_value) {
          scratch->Bind(def.var, def.non_overflow_value);
        }
        scratch->const_int_bound.Update(def.var, ToConstIntBound(*def.proved_bound));
      }
    };

    for (auto scope = chain.rbegin(); scope != chain.rend(); ++scope) {
      const auto& step = scopes_[*scope];
      bind_before(step.entry);
      if (step.kind == ScopeNode::Kind::IfBranch) {
        PushGuardConstraints(step.condition, step.truth, scratch, scratch_aliases, scratch_ready,
                             scratch_contexts);
      } else if (step.kind == ScopeNode::Kind::ForLoop && step.loop) {
        scratch_loops.push_back(step.loop);
        EnterLoopConstraints(step.loop, step.loop_var_bound, step.start_non_overflow, step.stop_non_overflow,
                             scratch, scratch_ready, scratch_contexts);
      }
    }
    bind_before(target.position);
    fallback_definitions_.fetch_add(dependencies.definitions.size(), std::memory_order_relaxed);
    auto res = EvaluateWithAnalyzer(expr, *scratch, scratch_ready, scratch_loops);
    for (auto it = scratch_contexts.rbegin(); it != scratch_contexts.rend(); ++it) {
      it->ExitScope();
    }
    return res;
  }

 private:
  struct LoopProof {
    bool is_positive_trip = false;
    bool is_at_most_one_trip = false;
    bool has_safe_bounds = false;
  };

  struct Cut {
    size_t scope;
    size_t position;
  };

  struct RangeResult {
    const Expr* expression = nullptr;
    std::optional<SymbolicIntRange> range;
  };

  struct DefRecord {
    VarPtr var;
    ExprPtr value;
    bool is_bool_alias = false;
    ExprPtr non_overflow_value;
    std::optional<Bound128> proved_bound;
    size_t position = 0;
  };

  struct DefinitionIndex {
    size_t scope;
    size_t index;
  };

  struct Dependencies {
    std::vector<VarPtr> parameters;
    std::vector<const DefRecord*> definitions;
  };

  struct ScopeNode {
    enum class Kind { Root, IfBranch, ForLoop, WhileLoop };
    size_t parent = 0;
    Kind kind = Kind::Root;
    ExprPtr condition;
    bool truth = true;
    const ForStmt* loop = nullptr;
    arith::ConstIntBound loop_var_bound{arith::ConstIntBound::kNegInf, arith::ConstIntBound::kPosInf};
    bool start_non_overflow = false;
    bool stop_non_overflow = false;
    std::vector<DefRecord> defs;
    // The parent control cut and the exclusive end of this lexical subtree.
    size_t entry = 0;
    size_t exit = 0;
  };

  FunctionPtr function_;
  SyncParameterBounds parameters_;
  std::optional<std::pair<int64_t, int64_t>> blocks_;
  std::optional<SyncDispatchCoordinates> dispatch_;
  ScalarBoundOperands scalar_bound_operands_;

  enum class DispatchValue { Unknown, Index, Width };
  std::unordered_map<const Var*, DispatchValue> dispatch_values_;

  arith::AnalyzerPtr analyzer_;
  std::unordered_set<const Var*> ready_;
  std::unordered_map<const Var*, ExprPtr> guard_aliases_;
  std::unordered_map<const Var*, ExprPtr> ssa_defs_;
  std::unordered_set<const Var*> non_overflow_ssa_;
  std::vector<const ForStmt*> active_loops_;
  std::vector<ScopeNode> scopes_;
  size_t clock_ = 0;
  SyncControlRangeWork work_;
  mutable std::atomic<size_t> fallback_definitions_{0};
  std::unordered_map<const Var*, arith::ConstIntBound> root_bindings_;
  std::unordered_map<const Var*, std::optional<DefinitionIndex>> definitions_;
  std::unordered_map<const Stmt*, Cut> cuts_;
  std::unordered_set<const Stmt*> ambiguous_cuts_;
  std::unordered_map<const Stmt*, std::array<RangeResult, 2>> footprint_ranges_;
  std::unordered_map<const Stmt*, std::unordered_map<const Expr*, std::optional<Bound128>>> scalar_bounds_;
  std::unordered_map<const ForStmt*, LoopProof> loop_proofs_;
  std::unordered_map<const Var*, std::vector<const ForStmt*>> loops_by_var_;
  std::unordered_map<const Var*, size_t> var_def_scope_;
  std::unordered_map<const ForStmt*, size_t> loop_body_scope_;

  const DefRecord* FindDefinition(const Var* variable, Cut target) const {
    const auto found = definitions_.find(variable);
    if (found == definitions_.end() || !found->second) {
      return nullptr;
    }
    const auto [scope_id, index] = *found->second;
    const auto& scope = scopes_[scope_id];
    const auto& def = scope.defs[index];
    if (def.position >= target.position || target.position >= scope.exit) {
      return nullptr;
    }
    return &def;
  }

  Dependencies CollectDependencies(const ExprPtr& expr, Cut target, const std::vector<size_t>& chain) const {
    std::vector<std::pair<ExprPtr, Cut>> pending;
    for (const auto id : chain) {
      const auto& scope = scopes_[id];
      const Cut before{scope.parent, scope.entry};
      if (scope.condition) {
        pending.emplace_back(scope.condition, before);
      }
      if (scope.start_non_overflow) {
        pending.emplace_back(scope.loop->start_, before);
      }
      if (scope.stop_non_overflow) {
        pending.emplace_back(scope.loop->stop_, before);
      }
    }
    pending.emplace_back(expr, target);

    Dependencies result;
    std::unordered_map<const Expr*, size_t> inspected;
    std::unordered_set<const DefRecord*> needed;
    std::unordered_set<const Var*> parameters;
    while (!pending.empty()) {
      auto [current, use] = std::move(pending.back());
      pending.pop_back();
      auto [entry, inserted] = inspected.emplace(current.get(), use.position);
      if (!inserted && entry->second >= use.position) {
        continue;
      }
      entry->second = use.position;
      if (const auto var = AsVarLike(current)) {
        if (const auto* def = FindDefinition(var.get(), use);
            def && (def->is_bool_alias || def->proved_bound)) {
          if (needed.insert(def).second) {
            result.definitions.push_back(def);
            const auto& value = def->is_bool_alias ? def->value : def->non_overflow_value;
            if (value) {
              pending.emplace_back(value, Cut{use.scope, def->position});
            }
          }
        } else if (root_bindings_.count(var.get()) && parameters.insert(var.get()).second) {
          result.parameters.push_back(var);
        }
      } else if (const auto binary = As<BinaryExpr>(current)) {
        pending.emplace_back(binary->left_, use);
        pending.emplace_back(binary->right_, use);
      } else if (const auto unary = As<UnaryExpr>(current)) {
        pending.emplace_back(unary->operand_, use);
      }
    }
    std::sort(result.definitions.begin(), result.definitions.end(),
              [](const DefRecord* a, const DefRecord* b) { return a->position < b->position; });
    return result;
  }

  static bool AddOverflow128(__int128 a, __int128 b, __int128& out) {
    return __builtin_add_overflow(a, b, &out);
  }
  static bool SubOverflow128(__int128 a, __int128 b, __int128& out) {
    return __builtin_sub_overflow(a, b, &out);
  }
  static bool MulOverflow128(__int128 a, __int128 b, __int128& out) {
    return __builtin_mul_overflow(a, b, &out);
  }
  static bool NegOverflow128(__int128 a, __int128& out) {
    return __builtin_sub_overflow(__int128{0}, a, &out);
  }

  bool IsInScopeOrDescendant(size_t scope_id, size_t ancestor_scope) const {
    if (ancestor_scope == 0) return true;
    for (size_t cur = scope_id; cur != 0; cur = scopes_[cur].parent) {
      if (cur == ancestor_scope) return true;
    }
    return false;
  }

  bool IsVarStrictlyOutsideOutermostLoop(const Var* var, const std::vector<const ForStmt*>& loops) const {
    if (!var) return false;
    const auto vit = var_def_scope_.find(var);
    if (vit == var_def_scope_.end()) return false;
    if (loops.empty()) return true;
    const auto lit = loop_body_scope_.find(loops.front());
    if (lit == loop_body_scope_.end()) return false;
    return !IsInScopeOrDescendant(vit->second, lit->second);
  }

  void BindRoot(arith::Analyzer& az, std::unordered_set<const Var*>& ready_set) const {
    BindBufferAccTrimParameters(function_, az, ready_set);
    for (const auto& [param_idx, bounds] : parameters_) {
      if (param_idx >= function_->params_.size()) continue;
      const auto& param = function_->params_[param_idx];
      if (!IsSignedIntOrIndex(param)) continue;
      ready_set.insert(param.get());
      const auto clamped = ClampToType(param, bounds.first, bounds.second);
      if (clamped) {
        az.const_int_bound.Update(param, ToConstIntBound(*clamped));
        if (clamped->is_const() && clamped->lo > kBoundNegInf && clamped->hi < kBoundPosInf) {
          az.Bind(param, MakeConstInt(static_cast<int64_t>(clamped->lo), GetScalarDtype(param)));
        }
      }
    }
  }

  DispatchValue DispatchOrigin(const ExprPtr& expression) const {
    auto value = expression;
    while (const auto cast = As<Cast>(value)) {
      value = cast->operand_;
    }
    if (const auto var = AsVarLike(value)) {
      const auto found = dispatch_values_.find(var.get());
      return found == dispatch_values_.end() ? DispatchValue::Unknown : found->second;
    }
    const auto call = As<Call>(value);
    if (call && call->args_.empty() && call->kwargs_.empty() && call->attrs_.empty()) {
      if (IsOp(call, "buffer.get_block_idx") || IsOp(call, "tile.get_block_idx")) {
        return DispatchValue::Index;
      }
      if (IsOp(call, "buffer.get_block_num") || IsOp(call, "tile.get_block_num")) {
        return DispatchValue::Width;
      }
    }
    return DispatchValue::Unknown;
  }

  std::optional<Bound128> BoundNonOverflowingScalar(
      const ExprPtr& expr, arith::Analyzer& az, const std::unordered_set<const Var*>& ready_set,
      std::unordered_map<const Expr*, std::optional<Bound128>>& cache) const {
    if (!expr || !IsSignedIntOrIndex(expr)) return std::nullopt;
    if (const auto it = cache.find(expr.get()); it != cache.end()) {
      return it->second;
    }

    std::optional<Bound128> result;
    if (const auto ci = As<ConstInt>(expr)) {
      result = ClampToType(expr, ci->value_, ci->value_);
    } else if (const auto var = AsVarLike(expr)) {
      if (ready_set.count(var.get()) != 0) {
        const auto b = az.const_int_bound(var);
        if (Is64BitSignedOrIndex(expr)) {
          const __int128 lo =
              b.min_value <= kBoundNegInf ? std::numeric_limits<int64_t>::min() : __int128(b.min_value);
          // Never treat analyzer sentinels {kNegInf, kNegInf} or {kPosInf, kPosInf} as exact constants.
          if (!(b.min_value >= kBoundPosInf && b.max_value >= kBoundPosInf) &&
              !(b.min_value <= kBoundNegInf && b.max_value <= kBoundNegInf)) {
            result = ClampToType(expr, lo, b.max_value);
          }
        } else {
          result = ClampToType(expr, b.min_value, b.max_value);
        }
      }
    } else if (const auto call = As<Call>(expr)) {
      const auto origin = dispatch_ ? DispatchOrigin(expr) : DispatchValue::Unknown;
      if (origin == DispatchValue::Index) {
        result = ClampToType(expr, 0, dispatch_->block_count.second - 1);
      } else if (origin == DispatchValue::Width) {
        result = ClampToType(expr, dispatch_->block_count.first, dispatch_->block_count.second);
      } else if (IsOp(call, "buffer.get_block_idx") && call->args_.empty()) {
        if (blocks_) {
          result = ClampToType(expr, blocks_->first, blocks_->second);
        } else {
          result = ClampToType(expr, 0, kBoundPosInf - 1);
        }
      } else if (IsOp(call, "buffer.get_block_num") && call->args_.empty()) {
        result = ClampToType(expr, 1, kBoundPosInf);
      } else if (IsOp(call, "buffer.get_subblock_idx") && call->args_.empty()) {
        result = ClampToType(expr, 0, kBoundPosInf);
      }
    } else if (const auto cast = As<Cast>(expr)) {
      if (IsSignedIntOrIndex(cast->operand_)) {
        const auto inner = BoundNonOverflowingScalar(cast->operand_, az, ready_set, cache);
        if (inner) {
          result = ClampToType(expr, inner->lo, inner->hi);
        }
      }
    } else if (const auto neg = As<Neg>(expr)) {
      const auto inner = BoundNonOverflowingScalar(neg->operand_, az, ready_set, cache);
      if (inner) {
        result = ClampToType(expr, -inner->hi, -inner->lo);
      }
    } else if (const auto binary = As<BinaryExpr>(expr)) {
      const auto a = BoundNonOverflowingScalar(binary->left_, az, ready_set, cache);
      const auto b = BoundNonOverflowingScalar(binary->right_, az, ready_set, cache);
      if (a && b) {
        const __int128 alo = a->lo, ahi = a->hi;
        const __int128 blo = b->lo, bhi = b->hi;
        if (As<Add>(expr)) {
          result = ClampToType(expr, alo + blo, ahi + bhi);
        } else if (As<Sub>(expr)) {
          __int128 lo128 = alo - bhi;
          __int128 hi128 = ahi - blo;
          // These operands have already passed the original machine-width
          // checks, including any intervening casts. Intersect the independent
          // bounds with the logical dispatch relation, without sampling lanes.
          if (dispatch_ && DispatchOrigin(binary->left_) == DispatchValue::Width &&
              DispatchOrigin(binary->right_) == DispatchValue::Index) {
            lo128 = std::max<__int128>(lo128, 1);
          } else if (az.CanProve(MakeGt(binary->left_, binary->right_))) {
            lo128 = std::max<__int128>(lo128, 1);
          } else if (az.CanProve(MakeGe(binary->left_, binary->right_))) {
            lo128 = std::max<__int128>(lo128, 0);
          }
          if (az.CanProve(MakeLt(binary->left_, binary->right_))) {
            hi128 = std::min<__int128>(hi128, -1);
          } else if (az.CanProve(MakeLe(binary->left_, binary->right_))) {
            hi128 = std::min<__int128>(hi128, 0);
          }
          result = ClampToType(expr, lo128, hi128);
        } else if (As<Mul>(expr)) {
          const __int128 p1 = alo * blo;
          const __int128 p2 = alo * bhi;
          const __int128 p3 = ahi * blo;
          const __int128 p4 = ahi * bhi;
          const __int128 lo128 = std::min({p1, p2, p3, p4});
          const __int128 hi128 = std::max({p1, p2, p3, p4});
          result = ClampToType(expr, lo128, hi128);
        } else if (As<FloorDiv>(expr)) {
          if (blo > 0 || bhi < 0) {
            const __int128 d1 = FloorDiv128(alo, blo);
            const __int128 d2 = FloorDiv128(alo, bhi);
            const __int128 d3 = FloorDiv128(ahi, blo);
            const __int128 d4 = FloorDiv128(ahi, bhi);
            result = ClampToType(expr, std::min({d1, d2, d3, d4}), std::max({d1, d2, d3, d4}));
          }
        } else if (As<FloorMod>(expr)) {
          if (blo > 0) {
            if (a->is_const() && b->is_const() && alo > kBoundNegInf && ahi < kBoundPosInf &&
                blo > kBoundNegInf && bhi < kBoundPosInf) {
              const __int128 val = FloorMod128(alo, blo);
              result = ClampToType(expr, val, val);
            } else {
              // Note: FloorMod(x, m) with m > 0 always yields [0, bhi - 1], even when x is unbounded above.
              result = ClampToType(expr, 0, bhi - 1);
            }
          }
        } else if (As<Min>(expr)) {
          result = ClampToType(expr, std::min(alo, blo), std::min(ahi, bhi));
        } else if (As<Max>(expr)) {
          result = ClampToType(expr, std::max(alo, blo), std::max(ahi, bhi));
        }
      }
    }

    if (result) {
      // Intersect with analyzer's const_int_bound (sound tightening, never widens).
      const auto az_b = az.const_int_bound(expr);
      const __int128 az_lo = Is64BitSignedOrIndex(expr) && az_b.min_value <= kBoundNegInf
                                 ? std::numeric_limits<int64_t>::min()
                                 : __int128(az_b.min_value);
      const __int128 lo = std::max(result->lo, az_lo);
      const __int128 hi = std::min(result->hi, __int128(az_b.max_value));
      if (lo <= hi && !(az_b.min_value >= kBoundPosInf && result->lo < kBoundPosInf)) {
        result = ClampToType(expr, lo, hi);
      }
    }
    cache.emplace(expr.get(), result);
    return result;
  }

  void PushGuardConstraints(const ExprPtr& condition, bool truth, const arith::AnalyzerPtr& az,
                            const std::unordered_map<const Var*, ExprPtr>& aliases,
                            const std::unordered_set<const Var*>& ready_set,
                            std::vector<arith::ConstraintContext>& out) const {
    struct Pending {
      ExprPtr expr;
      bool value;
      int alias_depth;
    };
    std::vector<Pending> pending{{condition, truth, 0}};
    std::unordered_set<const Expr*> positive, negative;
    std::unordered_map<const Expr*, std::optional<Bound128>> bound_cache;

    auto is_ready_expr = [&](const ExprPtr& arg) -> std::optional<Bound128> {
      return BoundNonOverflowingScalar(arg, *az, ready_set, bound_cache);
    };

    while (!pending.empty()) {
      const auto [expr, value, depth] = pending.back();
      pending.pop_back();
      if (!expr || !(value ? positive : negative).insert(expr.get()).second) continue;
      if (const auto variable = AsVarLike(expr)) {
        if (depth < kMaxBoolAliasDepth) {
          if (const auto found = aliases.find(variable.get()); found != aliases.end()) {
            pending.push_back({found->second, value, depth + 1});
          }
        }
        continue;
      }
      if (const auto negation = As<Not>(expr)) {
        pending.push_back({negation->operand_, !value, depth});
        continue;
      }
      const auto binary = As<BinaryExpr>(expr);
      if (!binary) continue;
      if ((value && As<And>(expr)) || (!value && As<Or>(expr))) {
        pending.push_back({binary->left_, value, depth});
        pending.push_back({binary->right_, value, depth});
        continue;
      }
      const bool is_cmp =
          As<Eq>(expr) || As<Ne>(expr) || As<Lt>(expr) || As<Le>(expr) || As<Gt>(expr) || As<Ge>(expr);
      if (!is_cmp) continue;

      const auto lb = is_ready_expr(binary->left_);
      const auto rb = is_ready_expr(binary->right_);
      if (!lb || !rb) continue;

      const ExprPtr effective = value ? expr : MakeNot(expr, expr->span_);
      out.push_back(az->GetConstraintContext(effective));

      // Also materialize constant-RHS tightening for Var comparisons against bounded non-constant
      // expressions (e.g. block_idx < total_tiles with block_idx >= 0 => total_tiles >= 1), since
      // ConstIntBoundAnalyzer::EnterConstraint only tightens when the peer is constant.
      auto normalize_cmp = [&](const ExprPtr& c) {
        if (const auto lt = As<Lt>(c)) {
          if (const auto rvar = As<Var>(lt->right_); rvar && ready_set.count(rvar.get())) {
            if (const auto b = is_ready_expr(lt->left_); b && b->lo > kBoundNegInf) {
              const __int128 tight_lo = b->lo + 1;
              if (tight_lo <= kBoundPosInf) {
                out.push_back(az->GetConstraintContext(
                    MakeGe(rvar, MakeConstInt(static_cast<int64_t>(tight_lo), GetScalarDtype(rvar)))));
              }
            }
          }
          if (const auto lvar = As<Var>(lt->left_); lvar && ready_set.count(lvar.get())) {
            if (const auto b = is_ready_expr(lt->right_); b && b->hi < kBoundPosInf) {
              const __int128 tight_hi = b->hi - 1;
              if (tight_hi >= kBoundNegInf) {
                out.push_back(az->GetConstraintContext(
                    MakeLe(lvar, MakeConstInt(static_cast<int64_t>(tight_hi), GetScalarDtype(lvar)))));
              }
            }
          }
        } else if (const auto le = As<Le>(c)) {
          if (const auto rvar = As<Var>(le->right_); rvar && ready_set.count(rvar.get())) {
            if (const auto b = is_ready_expr(le->left_); b && b->lo > kBoundNegInf) {
              out.push_back(az->GetConstraintContext(
                  MakeGe(rvar, MakeConstInt(static_cast<int64_t>(b->lo), GetScalarDtype(rvar)))));
            }
          }
          if (const auto lvar = As<Var>(le->left_); lvar && ready_set.count(lvar.get())) {
            if (const auto b = is_ready_expr(le->right_); b && b->hi < kBoundPosInf) {
              out.push_back(az->GetConstraintContext(
                  MakeLe(lvar, MakeConstInt(static_cast<int64_t>(b->hi), GetScalarDtype(lvar)))));
            }
          }
        } else if (const auto gt = As<Gt>(c)) {
          if (const auto lvar = As<Var>(gt->left_); lvar && ready_set.count(lvar.get())) {
            if (const auto b = is_ready_expr(gt->right_); b && b->lo > kBoundNegInf) {
              const __int128 tight_lo = b->lo + 1;
              if (tight_lo <= kBoundPosInf) {
                out.push_back(az->GetConstraintContext(
                    MakeGe(lvar, MakeConstInt(static_cast<int64_t>(tight_lo), GetScalarDtype(lvar)))));
              }
            }
          }
          if (const auto rvar = As<Var>(gt->right_); rvar && ready_set.count(rvar.get())) {
            if (const auto b = is_ready_expr(gt->left_); b && b->hi < kBoundPosInf) {
              const __int128 tight_hi = b->hi - 1;
              if (tight_hi >= kBoundNegInf) {
                out.push_back(az->GetConstraintContext(
                    MakeLe(rvar, MakeConstInt(static_cast<int64_t>(tight_hi), GetScalarDtype(rvar)))));
              }
            }
          }
        } else if (const auto ge = As<Ge>(c)) {
          if (const auto lvar = As<Var>(ge->left_); lvar && ready_set.count(lvar.get())) {
            if (const auto b = is_ready_expr(ge->right_); b && b->lo > kBoundNegInf) {
              out.push_back(az->GetConstraintContext(
                  MakeGe(lvar, MakeConstInt(static_cast<int64_t>(b->lo), GetScalarDtype(lvar)))));
            }
          }
          if (const auto rvar = As<Var>(ge->right_); rvar && ready_set.count(rvar.get())) {
            if (const auto b = is_ready_expr(ge->left_); b && b->hi < kBoundPosInf) {
              out.push_back(az->GetConstraintContext(
                  MakeLe(rvar, MakeConstInt(static_cast<int64_t>(b->hi), GetScalarDtype(rvar)))));
            }
          }
        }
      };

      if (value) {
        normalize_cmp(expr);
      } else {
        if (const auto lt = As<Lt>(expr))
          normalize_cmp(MakeGe(lt->left_, lt->right_, expr->span_));
        else if (const auto le = As<Le>(expr))
          normalize_cmp(MakeGt(le->left_, le->right_, expr->span_));
        else if (const auto gt = As<Gt>(expr))
          normalize_cmp(MakeLe(gt->left_, gt->right_, expr->span_));
        else if (const auto ge = As<Ge>(expr))
          normalize_cmp(MakeLt(ge->left_, ge->right_, expr->span_));
      }
    }
  }

  void EnterLoopConstraints(const ForStmt* loop, const arith::ConstIntBound& iv_bound,
                            bool start_non_overflow, bool stop_non_overflow, const arith::AnalyzerPtr& az,
                            std::unordered_set<const Var*>& ready_set,
                            std::vector<arith::ConstraintContext>& out) const {
    ready_set.insert(loop->loop_var_.get());
    az->const_int_bound.Update(loop->loop_var_, iv_bound);
    if (iv_bound.is_const() && iv_bound.min_value > kBoundNegInf && iv_bound.max_value < kBoundPosInf) {
      az->Bind(loop->loop_var_, MakeConstInt(iv_bound.min_value, GetScalarDtype(loop->loop_var_)));
    }
    if (start_non_overflow) {
      out.push_back(az->GetConstraintContext(MakeGe(loop->loop_var_, loop->start_)));
    }
    if (stop_non_overflow) {
      out.push_back(az->GetConstraintContext(MakeLt(loop->loop_var_, loop->stop_)));
      if (const auto stop_var = As<Var>(loop->stop_); stop_var &&
                                                      iv_bound.min_value > arith::ConstIntBound::kNegInf &&
                                                      __int128(iv_bound.min_value) + 1 <= kBoundPosInf) {
        out.push_back(az->GetConstraintContext(MakeGe(
            stop_var,
            MakeConstInt(static_cast<int64_t>(__int128(iv_bound.min_value) + 1), GetScalarDtype(stop_var)))));
      }
    }
  }

  struct AffineForm {
    bool valid = false;
    std::map<const ForStmt*, __int128> coeffs;
    __int128 const_offset = 0;
    ExprPtr symbolic_invariant;  // Invariant wrt ALL active_loops
  };

  AffineForm ExtractAffine(const ExprPtr& expr, const std::vector<const ForStmt*>& loops,
                           const std::unordered_map<const Var*, const ForStmt*>& var_to_loop,
                           std::unordered_map<const Expr*, AffineForm>& memo, int depth = 0) const {
    if (!expr || depth > 64) return {};
    if (const auto it = memo.find(expr.get()); it != memo.end()) {
      return it->second;
    }

    auto add_sym = [&](const ExprPtr& a, const ExprPtr& b) -> ExprPtr {
      if (!a) return b;
      if (!b) return a;
      return MakeAdd(a, b, expr->span_);
    };

    AffineForm out;
    if (const auto ci = As<ConstInt>(expr)) {
      out.valid = true;
      out.const_offset = ci->value_;
    } else if (const auto var = AsVarLike(expr)) {
      if (const auto lit = var_to_loop.find(var.get()); lit != var_to_loop.end()) {
        out.valid = true;
        out.coeffs[lit->second] = 1;
      } else if (const auto dit = ssa_defs_.find(var.get());
                 dit != ssa_defs_.end() && non_overflow_ssa_.count(var.get()) &&
                 (!As<Call>(dit->second) || IsPureBlockQueryCall(As<Call>(dit->second)))) {
        out = ExtractAffine(dit->second, loops, var_to_loop, memo, depth + 1);
      } else if (IsVarStrictlyOutsideOutermostLoop(var.get(), loops)) {
        out.valid = true;
        out.symbolic_invariant = expr;
      }
    } else if (const auto neg = As<Neg>(expr)) {
      auto inner = ExtractAffine(neg->operand_, loops, var_to_loop, memo, depth + 1);
      __int128 neg_off = 0;
      if (inner.valid && !NegOverflow128(inner.const_offset, neg_off) &&
          (!inner.symbolic_invariant || inner.coeffs.empty())) {
        bool ok = true;
        std::map<const ForStmt*, __int128> neg_coeffs;
        for (const auto& [lp, c] : inner.coeffs) {
          __int128 nc = 0;
          if (NegOverflow128(c, nc)) {
            ok = false;
            break;
          }
          neg_coeffs[lp] = nc;
        }
        if (ok) {
          out.valid = true;
          out.const_offset = neg_off;
          out.coeffs = std::move(neg_coeffs);
          if (inner.symbolic_invariant)
            out.symbolic_invariant = MakeNeg(inner.symbolic_invariant, expr->span_);
        }
      }
    } else if (const auto add = As<Add>(expr)) {
      auto a = ExtractAffine(add->left_, loops, var_to_loop, memo, depth + 1);
      auto b = ExtractAffine(add->right_, loops, var_to_loop, memo, depth + 1);
      __int128 sum_off = 0;
      if (a.valid && b.valid && !AddOverflow128(a.const_offset, b.const_offset, sum_off)) {
        bool ok = true;
        std::map<const ForStmt*, __int128> sum_coeffs = a.coeffs;
        for (const auto& [lp, c] : b.coeffs) {
          __int128 sc = 0;
          if (AddOverflow128(sum_coeffs[lp], c, sc)) {
            ok = false;
            break;
          }
          if (sc == 0) {
            sum_coeffs.erase(lp);
          } else {
            sum_coeffs[lp] = sc;
          }
        }
        if (ok) {
          out.valid = true;
          out.const_offset = sum_off;
          out.coeffs = std::move(sum_coeffs);
          out.symbolic_invariant = add_sym(a.symbolic_invariant, b.symbolic_invariant);
        }
      }
    } else if (const auto sub = As<Sub>(expr)) {
      auto a = ExtractAffine(sub->left_, loops, var_to_loop, memo, depth + 1);
      auto b = ExtractAffine(sub->right_, loops, var_to_loop, memo, depth + 1);
      __int128 diff_off = 0;
      if (a.valid && b.valid && !SubOverflow128(a.const_offset, b.const_offset, diff_off)) {
        bool ok = true;
        std::map<const ForStmt*, __int128> diff_coeffs = a.coeffs;
        for (const auto& [lp, c] : b.coeffs) {
          __int128 dc = 0;
          if (SubOverflow128(diff_coeffs[lp], c, dc)) {
            ok = false;
            break;
          }
          if (dc == 0) {
            diff_coeffs.erase(lp);
          } else {
            diff_coeffs[lp] = dc;
          }
        }
        if (ok) {
          out.valid = true;
          out.const_offset = diff_off;
          out.coeffs = std::move(diff_coeffs);
          if (b.symbolic_invariant) {
            out.symbolic_invariant = a.symbolic_invariant
                                         ? MakeSub(a.symbolic_invariant, b.symbolic_invariant, expr->span_)
                                         : MakeNeg(b.symbolic_invariant, expr->span_);
          } else {
            out.symbolic_invariant = a.symbolic_invariant;
          }
        }
      }
    } else if (const auto mul = As<Mul>(expr)) {
      auto a = ExtractAffine(mul->left_, loops, var_to_loop, memo, depth + 1);
      auto b = ExtractAffine(mul->right_, loops, var_to_loop, memo, depth + 1);
      auto scale_by = [&](const AffineForm& f, __int128 k) -> AffineForm {
        if (!f.valid || k == 0) {
          AffineForm zero;
          zero.valid = true;
          return zero;
        }
        AffineForm r;
        if (MulOverflow128(f.const_offset, k, r.const_offset)) return {};
        for (const auto& [lp, c] : f.coeffs) {
          __int128 sc = 0;
          if (MulOverflow128(c, k, sc)) return {};
          r.coeffs[lp] = sc;
        }
        if (f.symbolic_invariant) {
          if (k < std::numeric_limits<int64_t>::min() || k > std::numeric_limits<int64_t>::max()) return {};
          r.symbolic_invariant = MakeMul(
              f.symbolic_invariant, MakeConstInt(static_cast<int64_t>(k), GetScalarDtype(expr)), expr->span_);
        }
        r.valid = true;
        return r;
      };
      if (a.valid && b.valid) {
        if (a.coeffs.empty() && !a.symbolic_invariant) {
          out = scale_by(b, a.const_offset);
        } else if (b.coeffs.empty() && !b.symbolic_invariant) {
          out = scale_by(a, b.const_offset);
        } else if (a.coeffs.empty() && b.coeffs.empty()) {
          // Product of two loop-invariant symbolic terms is still loop-invariant!
          auto materialize_sym = [&](const AffineForm& f) -> ExprPtr {
            const DataType dtype = GetScalarDtype(expr);
            if (f.const_offset == 0) return f.symbolic_invariant;
            auto c_expr = MakeConstInt(static_cast<int64_t>(f.const_offset), dtype, expr->span_);
            return f.symbolic_invariant ? MakeAdd(f.symbolic_invariant, c_expr, expr->span_) : c_expr;
          };
          if (a.const_offset >= std::numeric_limits<int64_t>::min() &&
              a.const_offset <= std::numeric_limits<int64_t>::max() &&
              b.const_offset >= std::numeric_limits<int64_t>::min() &&
              b.const_offset <= std::numeric_limits<int64_t>::max()) {
            const ExprPtr lhs_sym = materialize_sym(a);
            const ExprPtr rhs_sym = materialize_sym(b);
            if (lhs_sym && rhs_sym) {
              out.valid = true;
              out.symbolic_invariant = MakeMul(lhs_sym, rhs_sym, expr->span_);
            }
          }
        }
      }
    } else {
      // Any other expression containing no non-query Call and only Vars defined
      // strictly outside loops.front()'s body scope is loop-invariant!
      bool invariant_ok = true;
      std::function<void(const ExprPtr&, int)> scan = [&](const ExprPtr& e, int d) {
        if (!e || !invariant_ok) return;
        if (d > 64) {
          invariant_ok = false;
          return;
        }
        if (As<ConstInt>(e) || As<ConstFloat>(e) || As<ConstBool>(e)) {
          return;
        }
        if (const auto v = AsVarLike(e)) {
          if (var_to_loop.count(v.get()) || !IsVarStrictlyOutsideOutermostLoop(v.get(), loops)) {
            invariant_ok = false;
          }
          return;
        }
        if (const auto b = As<BinaryExpr>(e)) {
          scan(b->left_, d + 1);
          scan(b->right_, d + 1);
        } else if (const auto u = As<UnaryExpr>(e)) {
          scan(u->operand_, d + 1);
        } else if (const auto c = As<Call>(e)) {
          if (!IsPureBlockQueryCall(c)) {
            invariant_ok = false;
          }
        } else {
          invariant_ok = false;
        }
      };
      scan(expr, 0);
      if (invariant_ok) {
        out.valid = true;
        out.symbolic_invariant = expr;
      }
    }
    memo.emplace(expr.get(), out);
    return out;
  }

  std::optional<SymbolicIntRange> EvaluateWithAnalyzer(const ExprPtr& expr, arith::Analyzer& az,
                                                       const std::unordered_set<const Var*>& ready_set,
                                                       const std::vector<const ForStmt*>& loops) const {
    std::unordered_map<const Expr*, std::optional<Bound128>> bound_cache;
    const auto bound = BoundNonOverflowingScalar(expr, az, ready_set, bound_cache);
    if (!bound) return std::nullopt;

    SymbolicIntRange range;
    range.lo = static_cast<int64_t>(bound->lo);
    range.hi = static_cast<int64_t>(bound->hi);
    range.expr = expr;
    if (range.is_const() && bound->lo > kBoundNegInf && bound->hi < kBoundPosInf) {
      range.affine_base = range.lo;
      return range;
    }

    std::unordered_map<const Var*, const ForStmt*> var_to_loop;
    for (const ForStmt* lp : loops) {
      if (lp && lp->loop_var_) var_to_loop[lp->loop_var_.get()] = lp;
    }

    // Unwrap ANF SSA aliases in non_overflow_ssa_ before matching FloorMod(lhs, ConstInt(M)).
    ExprPtr unwrapped = expr;
    for (int hops = 0; hops < 8; ++hops) {
      const auto v = AsVarLike(unwrapped);
      if (!v || !non_overflow_ssa_.count(v.get())) break;
      const auto dit = ssa_defs_.find(v.get());
      if (dit == ssa_defs_.end() || !dit->second) break;
      unwrapped = dit->second;
    }
    if (const auto mod = As<FloorMod>(unwrapped)) {
      if (const auto m_ci = As<ConstInt>(mod->right_); m_ci && m_ci->value_ > 0) {
        const int64_t M = m_ci->value_;
        std::unordered_map<const Expr*, AffineForm> memo;
        const auto inner = ExtractAffine(mod->left_, loops, var_to_loop, memo);
        if (inner.valid && !inner.symbolic_invariant && inner.coeffs.size() == 1) {
          const auto [lp, c_iv] = *inner.coeffs.begin();
          const auto start_ci = As<ConstInt>(lp->start_);
          const auto step_ci = As<ConstInt>(lp->step_);
          if (start_ci && step_ci && step_ci->value_ > 0) {
            __int128 coeff = 0;
            __int128 start_scaled = 0;
            __int128 base = 0;
            if (!MulOverflow128(c_iv, step_ci->value_, coeff) &&
                !MulOverflow128(c_iv, start_ci->value_, start_scaled) &&
                !AddOverflow128(inner.const_offset, start_scaled, base)) {
              range.affine_frame = lp;
              range.modulus = M;
              range.affine_coeff = static_cast<int64_t>(FloorMod128(coeff, M));
              range.affine_base = static_cast<int64_t>(FloorMod128(base, M));
              return range;
            }
          }
        }
      }
    }

    // Otherwise check single-loop affine decomposition wrt the innermost active loop
    // referenced by `expr` (all outer loops are invariant wrt the innermost loop!).
    std::unordered_map<const Expr*, AffineForm> memo;
    const auto af = ExtractAffine(expr, loops, var_to_loop, memo);
    if (af.valid && !af.coeffs.empty()) {
      const ForStmt* innermost = nullptr;
      for (auto it = loops.rbegin(); it != loops.rend(); ++it) {
        if (af.coeffs.count(*it)) {
          innermost = *it;
          break;
        }
      }
      if (innermost) {
        const auto step_ci = As<ConstInt>(innermost->step_);
        if (step_ci && step_ci->value_ > 0) {
          const __int128 c_iv = af.coeffs.at(innermost);
          __int128 step_coeff = 0;
          if (!MulOverflow128(c_iv, step_ci->value_, step_coeff) &&
              step_coeff >= std::numeric_limits<int64_t>::min() &&
              step_coeff <= std::numeric_limits<int64_t>::max()) {
            const auto start_ci = As<ConstInt>(innermost->start_);
            __int128 const_base = af.const_offset;
            ExprPtr sym = af.symbolic_invariant;
            bool ok = true;
            // Fold any outer loop terms into `sym` in deterministic lexical `loops` order.
            for (const ForStmt* olp : loops) {
              if (olp == innermost) continue;
              const auto oit = af.coeffs.find(olp);
              if (oit == af.coeffs.end()) continue;
              const __int128 oc = oit->second;
              if (oc < std::numeric_limits<int64_t>::min() || oc > std::numeric_limits<int64_t>::max()) {
                ok = false;
                break;
              }
              ExprPtr term = olp->loop_var_;
              if (oc != 1) {
                term =
                    MakeMul(term, MakeConstInt(static_cast<int64_t>(oc), GetScalarDtype(expr)), expr->span_);
              }
              sym = sym ? MakeAdd(sym, term, expr->span_) : term;
            }
            if (ok && start_ci) {
              __int128 start_scaled = 0;
              if (MulOverflow128(c_iv, start_ci->value_, start_scaled) ||
                  AddOverflow128(const_base, start_scaled, const_base)) {
                ok = false;
              }
            } else if (ok) {
              if (c_iv < std::numeric_limits<int64_t>::min() || c_iv > std::numeric_limits<int64_t>::max()) {
                ok = false;
              } else {
                ExprPtr term = innermost->start_;
                if (c_iv != 1) {
                  term = MakeMul(term, MakeConstInt(static_cast<int64_t>(c_iv), GetScalarDtype(expr)),
                                 expr->span_);
                }
                sym = sym ? MakeAdd(sym, term, expr->span_) : term;
              }
            }
            if (ok && const_base >= std::numeric_limits<int64_t>::min() &&
                const_base <= std::numeric_limits<int64_t>::max()) {
              range.affine_frame = innermost;
              range.affine_coeff = static_cast<int64_t>(step_coeff);
              range.affine_base = static_cast<int64_t>(const_base);
              range.symbolic_base = sym;
              return range;
            }
          }
        }
      }
    }

    return range;
  }

  // Footprint clients can ask from multiple passes and in arbitrary order. Prove
  // their static operands once at the original use cut, while lexical bindings
  // and guards are already active. Store unknown results as well as proved ones.
  void RecordFootprintRanges(const Stmt* cut, const CallPtr& call) {
    if (!call) {
      return;
    }
    const auto operands = FootprintRangeOperands(call);
    for (size_t i = 0; i < operands.size(); ++i) {
      if (!IsSignedIntOrIndex(operands[i])) {
        continue;
      }
      auto& result = footprint_ranges_[cut][i];
      result = {operands[i].get(), EvaluateWithAnalyzer(operands[i], *analyzer_, ready_, active_loops_)};
      ++work_.footprint_queries;
    }
  }

  void RecordScalarBounds(const StmtPtr& stmt, const CallPtr& call) {
    if (!scalar_bound_operands_ || !call) {
      return;
    }
    const auto operands = scalar_bound_operands_(stmt, call);
    if (operands.empty()) {
      return;
    }
    std::unordered_map<const Expr*, std::optional<Bound128>> cache;
    auto& bounds = scalar_bounds_[stmt.get()];
    for (const auto& operand : operands) {
      auto [entry, fresh] = bounds.try_emplace(operand.get());
      if (fresh) {
        entry->second = BoundNonOverflowingScalar(operand, *analyzer_, ready_, cache);
        ++work_.scalar_bound_queries;
      }
    }
  }

  void RecordDef(size_t scope_id, DefRecord def) {
    def.position = clock_++;
    auto& definitions = scopes_[scope_id].defs;
    auto [entry, inserted] =
        definitions_.emplace(def.var.get(), DefinitionIndex{scope_id, definitions.size()});
    if (!inserted) {
      // A raw Var reused as a declaration has no unique SSA definition. Keep
      // the ad-hoc cut query conservative. Normal lexical shadowing uses a new Var.
      entry->second.reset();
    }
    definitions.push_back(std::move(def));
    ++work_.indexed_definitions;
  }

  void ExitScope(size_t scope_id) {
    scopes_[scope_id].exit = clock_;
    // A local SSA equality can stay in the private construction analyzer.
    // Its identity is no longer ready outside the scope, and a Boolean
    // alias has to be removed so a later guard cannot expand it.
    for (const auto& def : scopes_[scope_id].defs) {
      ready_.erase(def.var.get());
      if (def.is_bool_alias) {
        guard_aliases_.erase(def.var.get());
      }
    }
  }

  void InitParametersAndScopes() {
    scopes_.push_back(ScopeNode{0, ScopeNode::Kind::Root, nullptr, true, nullptr});
    for (const auto& param : function_->params_) {
      if (param) var_def_scope_[param.get()] = 0;
    }
    BindRoot(*analyzer_, ready_);
    if (dispatch_) {
      for (const auto index : dispatch_->width_parameters) {
        const auto& parameter = function_->params_.at(index);
        analyzer_->const_int_bound.Update(parameter,
                                          {dispatch_->block_count.first, dispatch_->block_count.second});
        dispatch_values_.emplace(parameter.get(), DispatchValue::Width);
      }
    }
    const auto remember_root = [&](const VarPtr& var) {
      if (var && ready_.count(var.get())) {
        root_bindings_.try_emplace(var.get(), analyzer_->const_int_bound(var));
      }
    };
    for (const auto& param : function_->params_) {
      remember_root(param);
      if (const auto tensor = As<TensorType>(param->GetType())) {
        for (const auto& dim : tensor->shape_) {
          remember_root(AsVarLike(dim));
        }
      }
    }
    VisitBody(function_->body_, 0);
    scopes_[0].exit = clock_ + 1;
  }

  void VisitBody(const StmtPtr& stmt, size_t scope_id) {
    if (!stmt) return;
    const Cut cut{scope_id, clock_++};
    if (!cuts_.emplace(stmt.get(), cut).second) {
      ambiguous_cuts_.insert(stmt.get());
    }

    if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) VisitBody(child, scope_id);
      return;
    }

    if (const auto assign = As<AssignStmt>(stmt)) {
      var_def_scope_[assign->var_.get()] = scope_id;
      const auto call = As<Call>(assign->value_);
      RecordFootprintRanges(stmt.get(), call);
      RecordScalarBounds(stmt, call);
      const auto scalar = As<ScalarType>(assign->var_->GetType());
      if (scalar && scalar->dtype_ == DataType::BOOL && !call) {
        guard_aliases_.emplace(assign->var_.get(), assign->value_);
        ready_.insert(assign->var_.get());
        RecordDef(scope_id, {assign->var_, assign->value_, true, nullptr, std::nullopt});
      } else if (IsSignedIntOrIndex(assign->var_)) {
        ssa_defs_[assign->var_.get()] = assign->value_;
        std::unordered_map<const Expr*, std::optional<Bound128>> cache;
        auto b = BoundNonOverflowingScalar(assign->value_, *analyzer_, ready_, cache);
        ExprPtr no_ov_expr;
        if (b) {
          ready_.insert(assign->var_.get());
          no_ov_expr = assign->value_;
          non_overflow_ssa_.insert(assign->var_.get());
          if (dispatch_ && scalar && scalar->dtype_ == GetScalarDtype(assign->value_)) {
            dispatch_values_.emplace(assign->var_.get(), DispatchOrigin(assign->value_));
          }
          analyzer_->Bind(assign->var_, assign->value_);
          analyzer_->const_int_bound.Update(assign->var_, ToConstIntBound(*b));
        }
        RecordDef(scope_id, {assign->var_, assign->value_, false, no_ov_expr, b});
      }
      return;
    }

    if (const auto evaluate = As<EvalStmt>(stmt)) {
      const auto call = As<Call>(evaluate->expr_);
      RecordFootprintRanges(stmt.get(), call);
      RecordScalarBounds(stmt, call);
      return;
    }

    if (const auto branch = As<IfStmt>(stmt)) {
      auto trailing_yield = [](const StmtPtr& arm) -> YieldStmtPtr {
        if (const auto y = As<YieldStmt>(arm)) return y;
        if (const auto seq = As<SeqStmts>(arm); seq && !seq->stmts_.empty()) {
          return As<YieldStmt>(seq->stmts_.back());
        }
        return nullptr;
      };
      std::vector<std::optional<Bound128>> then_bounds, else_bounds;
      const size_t then_scope = scopes_.size();
      scopes_.push_back(ScopeNode{scope_id, ScopeNode::Kind::IfBranch, branch->condition_, true, nullptr});
      scopes_.back().entry = cut.position;
      {
        std::vector<arith::ConstraintContext> ctxs;
        PushGuardConstraints(branch->condition_, true, analyzer_, guard_aliases_, ready_, ctxs);
        VisitBody(branch->then_body_, then_scope);
        if (!branch->return_vars_.empty() && branch->else_body_) {
          if (const auto ty = trailing_yield(branch->then_body_);
              ty && ty->value_.size() == branch->return_vars_.size()) {
            then_bounds.resize(branch->return_vars_.size());
            std::unordered_map<const Expr*, std::optional<Bound128>> cache;
            for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
              if (IsSignedIntOrIndex(branch->return_vars_[i]) && IsSignedIntOrIndex(ty->value_[i])) {
                then_bounds[i] = BoundNonOverflowingScalar(ty->value_[i], *analyzer_, ready_, cache);
              }
            }
          }
        }
        ExitScope(then_scope);
        for (auto it = ctxs.rbegin(); it != ctxs.rend(); ++it) it->ExitScope();
      }
      if (branch->else_body_) {
        const size_t else_scope = scopes_.size();
        scopes_.push_back(ScopeNode{scope_id, ScopeNode::Kind::IfBranch, branch->condition_, false, nullptr});
        scopes_.back().entry = cut.position;
        std::vector<arith::ConstraintContext> ctxs;
        PushGuardConstraints(branch->condition_, false, analyzer_, guard_aliases_, ready_, ctxs);
        VisitBody(*branch->else_body_, else_scope);
        if (!branch->return_vars_.empty() && !then_bounds.empty()) {
          if (const auto ey = trailing_yield(*branch->else_body_);
              ey && ey->value_.size() == branch->return_vars_.size()) {
            else_bounds.resize(branch->return_vars_.size());
            std::unordered_map<const Expr*, std::optional<Bound128>> cache;
            for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
              if (then_bounds[i] && IsSignedIntOrIndex(ey->value_[i])) {
                else_bounds[i] = BoundNonOverflowingScalar(ey->value_[i], *analyzer_, ready_, cache);
              }
            }
          }
        }
        ExitScope(else_scope);
        for (auto it = ctxs.rbegin(); it != ctxs.rend(); ++it) it->ExitScope();
      }
      for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
        const auto& rv = branch->return_vars_[i];
        var_def_scope_[rv.get()] = scope_id;
        if (IsSignedIntOrIndex(rv)) {
          std::optional<Bound128> joined;
          if (i < then_bounds.size() && i < else_bounds.size() && then_bounds[i] && else_bounds[i]) {
            joined = Bound128{std::min(then_bounds[i]->lo, else_bounds[i]->lo),
                              std::max(then_bounds[i]->hi, else_bounds[i]->hi)};
            ready_.insert(rv.get());
            analyzer_->const_int_bound.Update(rv, ToConstIntBound(*joined));
          }
          RecordDef(scope_id, {rv, nullptr, false, nullptr, joined});
        }
      }
      return;
    }

    if (const auto loop = As<ForStmt>(stmt)) {
      loops_by_var_[loop->loop_var_.get()].push_back(loop.get());
      LoopProof proof;
      std::unordered_map<const Expr*, std::optional<Bound128>> cache;
      std::optional<Bound128> start_b, stop_b, step_b;

      if (loop->kind_ == ForKind::Sequential && IsSignedIntOrIndex(loop->start_) &&
          IsSignedIntOrIndex(loop->stop_) && IsSignedIntOrIndex(loop->step_)) {
        step_b = BoundNonOverflowingScalar(loop->step_, *analyzer_, ready_, cache);
        start_b = BoundNonOverflowingScalar(loop->start_, *analyzer_, ready_, cache);
        stop_b = BoundNonOverflowingScalar(loop->stop_, *analyzer_, ready_, cache);

        if (step_b && step_b->lo >= 1 && start_b && stop_b) {
          // Check the increment after the last executed iteration as well as
          // the values initially assigned to the loop's native coordinate.
          const auto maximum = SyncCoordinateMaximum(loop->loop_var_);
          proof.has_safe_bounds = maximum && start_b->lo >= 0 && stop_b->lo >= 0 && start_b->hi <= *maximum &&
                                  stop_b->hi <= *maximum && step_b->hi <= *maximum &&
                                  stop_b->hi - 1 + step_b->hi <= *maximum;
          // 1. Positive-trip proof (>= 1 trip).
          if (start_b->hi < stop_b->lo) {
            proof.is_positive_trip = true;
          } else if (analyzer_->CanProve(MakeGt(loop->stop_, loop->start_))) {
            proof.is_positive_trip = true;
          }

          // 2. At-most-one-trip proof (<= 1 trip).
          if (stop_b->hi <= start_b->lo) {
            proof.is_at_most_one_trip = true;
          } else if (start_b->lo >= 0 && stop_b->hi <= step_b->lo) {
            proof.is_at_most_one_trip = true;
          } else if (start_b->hi + step_b->hi <= kBoundPosInf &&
                     analyzer_->CanProve(MakeLe(loop->stop_, MakeAdd(loop->start_, loop->step_)))) {
            proof.is_at_most_one_trip = true;
          }
        }
      }
      loop_proofs_[loop.get()] = proof;

      // Enter loop body scope.
      const size_t body_scope = scopes_.size();
      loop_body_scope_[loop.get()] = body_scope;
      if (loop->loop_var_) var_def_scope_[loop->loop_var_.get()] = body_scope;
      arith::ConstIntBound iv_bound{arith::ConstIntBound::kNegInf, arith::ConstIntBound::kPosInf};
      bool use_loop_constraints = false;
      if (loop->kind_ == ForKind::Sequential && step_b && step_b->lo >= 1) {
        use_loop_constraints = true;
        const __int128 lo = start_b ? start_b->lo : kBoundNegInf;
        __int128 hi = kBoundPosInf;
        if (stop_b) {
          // Even when stop_b->hi == kPosInf, iv < stop => iv <= kPosInf - 1!
          hi = std::min<__int128>(kBoundPosInf - 1, stop_b->hi - 1);
        }
        if (lo <= hi) {
          iv_bound = ToConstIntBound(Bound128{lo, hi});
        } else {
          // Loop body is unreachable (e.g. 0-trip constant loop); keep wide bound to avoid invalid analyzer
          // state.
          iv_bound = {arith::ConstIntBound::kNegInf, arith::ConstIntBound::kPosInf};
          use_loop_constraints = false;
        }
      }
      scopes_.push_back(ScopeNode{scope_id, ScopeNode::Kind::ForLoop, nullptr, true, loop.get(), iv_bound,
                                  use_loop_constraints && start_b.has_value(),
                                  use_loop_constraints && stop_b.has_value()});
      scopes_.back().entry = cut.position;

      std::vector<arith::ConstraintContext> ctxs;
      const bool inserted_iv = ready_.insert(loop->loop_var_.get()).second;
      if (use_loop_constraints) {
        EnterLoopConstraints(loop.get(), iv_bound, start_b.has_value(), stop_b.has_value(), analyzer_, ready_,
                             ctxs);
      } else {
        analyzer_->const_int_bound.Update(loop->loop_var_, iv_bound);
      }
      for (const auto& arg : loop->iter_args_) {
        var_def_scope_[arg.get()] = body_scope;
        if (IsSignedIntOrIndex(arg)) {
          ready_.insert(arg.get());
          RecordDef(body_scope, {arg, nullptr, false, nullptr, std::nullopt});
        }
      }
      active_loops_.push_back(loop.get());
      VisitBody(loop->body_, body_scope);
      active_loops_.pop_back();
      ExitScope(body_scope);
      for (auto it = ctxs.rbegin(); it != ctxs.rend(); ++it) it->ExitScope();
      if (inserted_iv) {
        ready_.erase(loop->loop_var_.get());
        analyzer_->Unbind(loop->loop_var_);
      }
      for (const auto& arg : loop->iter_args_) {
        if (IsSignedIntOrIndex(arg)) {
          ready_.erase(arg.get());
          analyzer_->Unbind(arg);
        }
      }
      for (const auto& rv : loop->return_vars_) {
        var_def_scope_[rv.get()] = scope_id;
        if (IsSignedIntOrIndex(rv)) {
          ready_.insert(rv.get());
          RecordDef(scope_id, {rv, nullptr, false, nullptr, std::nullopt});
        }
      }
      return;
    }

    if (const auto loop = As<WhileStmt>(stmt)) {
      const size_t body_scope = scopes_.size();
      scopes_.push_back(ScopeNode{scope_id, ScopeNode::Kind::WhileLoop, nullptr, true, nullptr});
      scopes_.back().entry = cut.position;
      for (const auto& arg : loop->iter_args_) {
        var_def_scope_[arg.get()] = body_scope;
        if (IsSignedIntOrIndex(arg)) {
          ready_.insert(arg.get());
          RecordDef(body_scope, {arg, nullptr, false, nullptr, std::nullopt});
        }
      }
      VisitBody(loop->body_, body_scope);
      ExitScope(body_scope);
      for (const auto& arg : loop->iter_args_) {
        if (IsSignedIntOrIndex(arg)) {
          ready_.erase(arg.get());
          analyzer_->Unbind(arg);
        }
      }
      for (const auto& rv : loop->return_vars_) {
        var_def_scope_[rv.get()] = scope_id;
        if (IsSignedIntOrIndex(rv)) {
          ready_.insert(rv.get());
          RecordDef(scope_id, {rv, nullptr, false, nullptr, std::nullopt});
        }
      }
      return;
    }
  }
};

SyncControlRangeProver::SyncControlRangeProver(const FunctionPtr& function,
                                               const SyncParameterBounds& parameters,
                                               const std::optional<std::pair<int64_t, int64_t>>& blocks,
                                               ScalarBoundOperands scalar_bound_operands,
                                               const std::optional<SyncDispatchCoordinates>& dispatch)
    : impl_(std::make_unique<Impl>(function, parameters, blocks, scalar_bound_operands, dispatch)) {}

SyncControlRangeProver::~SyncControlRangeProver() = default;
SyncControlRangeProver::SyncControlRangeProver(SyncControlRangeProver&&) noexcept = default;
SyncControlRangeProver& SyncControlRangeProver::operator=(SyncControlRangeProver&&) noexcept = default;

SyncControlRangeWork SyncControlRangeProver::Work() const {
  return impl_ ? impl_->Work() : SyncControlRangeWork{};
}

bool SyncControlRangeProver::IsPositiveTrip(const ForStmt* loop) const {
  return impl_ && impl_->IsPositiveTrip(loop);
}

bool SyncControlRangeProver::IsAtMostOneTrip(const ForStmt* loop) const {
  return impl_ && impl_->IsAtMostOneTrip(loop);
}

bool SyncControlRangeProver::HasSafeNonnegativeLoopBounds(const ForStmt* loop) const {
  return impl_ && impl_->HasSafeNonnegativeLoopBounds(loop);
}

bool SyncControlRangeProver::IsPositiveTripCompatible(const ForStmt* loop) const {
  return impl_ && impl_->IsPositiveTripCompatible(loop);
}

std::optional<SymbolicIntRange> SyncControlRangeProver::LookupAtCut(const Stmt* cut,
                                                                    const ExprPtr& expr) const {
  return impl_ ? impl_->LookupAtCut(cut, expr) : std::nullopt;
}

std::optional<std::pair<int64_t, int64_t>> SyncControlRangeProver::LookupScalarBounds(
    const Stmt* cut, const Expr* expr) const {
  return impl_ ? impl_->LookupScalarBounds(cut, expr) : std::nullopt;
}

}  // namespace pypto::ir::sync
