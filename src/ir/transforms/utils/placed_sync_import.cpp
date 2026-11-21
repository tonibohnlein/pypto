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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_scalar_recipes.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/verifier.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_matrix_byte_geometry.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
namespace {
int64_t Checked(__int128 value) {
  CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max())
      << "Synchronization import requires nonnegative nonoverflowing index arithmetic";
  return static_cast<int64_t>(value);
}
// Exact contiguous byte evidence for both admission and retained exports.
// No enclosure or partial boxed layout is flattened into an exact interval.
std::optional<std::pair<int64_t, int64_t>> PhysicalInterval(const SyncStorage& storage,
                                                            const SyncAccess& access) {
  if (access.external || !access.exact_footprint || access.row_step || access.column_step ||
      access.after_first || !storage.row_parameters.empty() || !storage.column_parameters.empty() ||
      storage.row_block || storage.column_block || storage.row_extent_parameter)
    return std::nullopt;
  const auto row_end = __int128(access.row) + access.rows;
  const auto column_end = __int128(access.column) + access.columns;
  if (access.row < 0 || access.column < 0 || access.rows <= 0 || access.columns <= 0 ||
      row_end > storage.rows || column_end > storage.columns)
    return std::nullopt;
  __int128 begin = 0, end = 0;
  if (storage.byte_coordinates) {
    if (access.row != 0 || access.rows != 1) return std::nullopt;
    begin = access.column;
    end = column_end;
  } else {
    const auto elements = __int128(storage.rows) * storage.columns;
    const auto slot_bytes = storage.slot_count == 1 ? storage.bytes : storage.slot_bytes;
    if (elements <= 0 || slot_bytes <= 0 || slot_bytes % elements) return std::nullopt;
    const auto bytes = slot_bytes / elements;
    if (storage.layout && storage.layout->slayout_ != TileLayout::none_box) {
      // Whole qualified fractal tiles are contiguous; partial logical
      // rectangles need a layout-specific interval proof, not flattening.
      if (access.row || access.column || access.rows != storage.rows || access.columns != storage.columns)
        return std::nullopt;
      end = slot_bytes;
    } else if (!storage.layout || storage.layout->blayout_ == TileLayout::row_major) {
      if (access.rows != 1 && (access.column || access.columns != storage.columns)) return std::nullopt;
      begin = (__int128(access.row) * storage.columns + access.column) * bytes;
      end = begin + __int128(access.rows) * access.columns * bytes;
    } else if (storage.layout->blayout_ == TileLayout::col_major) {
      if (access.columns != 1 && (access.row || access.rows != storage.rows)) return std::nullopt;
      begin = (__int128(access.column) * storage.rows + access.row) * bytes;
      end = begin + __int128(access.rows) * access.columns * bytes;
    } else {
      return std::nullopt;
    }
  }
  return std::make_pair(Checked(begin), Checked(end));
}
int64_t Constant(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  CHECK(c) << "Synchronization import requires a constant extent/address";
  return Checked(c->value_);
}
void RequireIndex(const ExprPtr& e) {
  auto type = As<ScalarType>(e->GetType());
  CHECK(type && type->dtype_ == DataType::INDEX)
      << "Synchronization address roots and loop coordinates require INDEX type";
}
// Prove value-preserving machine arithmetic, not merely unbounded integer
// algebra. INDEX is signed 64-bit in the qualified backend contract.

bool SameCoordinateType(const ExprPtr& a, const ExprPtr& b) {
  const auto x = As<ScalarType>(a->GetType()), y = As<ScalarType>(b->GetType());
  return x && y && x->dtype_ == y->dtype_;
}
struct AffineCoordinate {
  int64_t first = 0, second = 0;
  SyncTranslation parameters;
  int64_t block = 0;
  std::map<const ForStmt*, int64_t> frames;
};
using ParameterIndices = std::unordered_map<const Expr*, size_t>;
// Retain checked scalar definitions as a DAG. Coefficients are only materialized
// for requested access roots, never for every SSA prefix definition.
class CoordinateGraph {
  using Node = SyncCoordinateNode;
  using Kind = Node::Kind;
  std::shared_ptr<SyncCoordinateProgram> program_ = std::make_shared<SyncCoordinateProgram>();
  std::vector<Node>& nodes_ = program_->nodes;
  std::unordered_map<const Expr*, size_t> definitions_;
  std::unordered_map<const Expr*, ExprPtr> remainders_;
  std::vector<bool> affine_;
  struct CoordinateContext {
    ForStmtPtr original_loop;
    int64_t count, start, step;
    std::vector<SyncLoopFrame> nested_frames;
  };
  struct Deferred {
    ExprPtr value;
    std::shared_ptr<const CoordinateContext> context;
  };
  struct DeferredSelect {
    ExprPtr condition, then_value, else_value;
    std::shared_ptr<const CoordinateContext> context;
  };
  using FrameKey = std::tuple<const ForStmt*, int64_t, int64_t, int64_t, const Expr*>;
  using ContextKey = std::tuple<const ForStmt*, int64_t, int64_t, int64_t, std::vector<FrameKey>>;
  std::map<std::pair<const Expr*, const CoordinateContext*>, size_t> memo_;
  std::map<ContextKey, std::shared_ptr<const CoordinateContext>> contexts_;
  std::unordered_map<const Expr*, Deferred> deferred_;
  std::unordered_map<const Expr*, DeferredSelect> deferred_selects_;

  std::shared_ptr<const CoordinateContext> ContextFor(const SyncSegment& segment) {
    std::vector<FrameKey> frames;
    frames.reserve(segment.nested_frames.size());
    for (const auto& frame : segment.nested_frames)
      frames.emplace_back(frame.loop.get(), frame.count, frame.start, frame.step,
                          frame.start_expression.get());
    const ContextKey key{segment.original_loop.get(), segment.count, segment.start, segment.step,
                         std::move(frames)};
    auto found = contexts_.find(key);
    if (found == contexts_.end()) {
      found = contexts_
                  .emplace(key, std::make_shared<const CoordinateContext>(
                                    CoordinateContext{segment.original_loop, segment.count, segment.start,
                                                      segment.step, segment.nested_frames}))
                  .first;
    }
    return found->second;
  }

 public:
  bool Defined(const ExprPtr& e) const {
    return definitions_.count(e.get()) || remainders_.count(e.get()) || deferred_.count(e.get()) ||
           deferred_selects_.count(e.get());
  }
  const Node& NodeAt(size_t id) const { return nodes_.at(id); }
  ExprPtr SlotExpression(const ExprPtr& value) const {
    if (const auto found = deferred_.find(value.get()); found != deferred_.end()) {
      if (As<FloorMod>(found->second.value)) return found->second.value;
    }
    const auto found = remainders_.find(value.get());
    return found == remainders_.end() ? value : found->second;
  }
  size_t Build(const ExprPtr& expression, const SyncSegment& segment, const ParameterIndices& parameters,
               CompactSyncInput& input) {
    return BuildInContext(expression, *ContextFor(segment), parameters, input);
  }
  size_t BuildInContext(const ExprPtr& e, const CoordinateContext& segment,
                        const ParameterIndices& parameters, CompactSyncInput& input) {
    const auto maximum = SyncCoordinateMaximum(e);
    const auto st = As<ScalarType>(e->GetType());
    const bool is_bool = !maximum && st && st->dtype_ == DataType::BOOL;
    CHECK(maximum || is_bool) << "Synchronization coordinates require integer arithmetic";
    if (input.invocation_coordinate &&
        (e == input.invocation_coordinate->Expression() || e == input.invocation_coordinate->Definition())) {
      const auto key = std::make_pair(e.get(), &segment);
      if (auto found = memo_.find(key); found != memo_.end()) {
        return found->second;
      }
      Node node{Kind::Block};
      node.lower = input.block_bounds->first;
      node.upper = input.block_bounds->second;
      const auto id = nodes_.size();
      nodes_.push_back(node);
      affine_.push_back(true);
      memo_.emplace(key, id);
      ++input.coordinate_nodes;
      return id;
    }
    if (const auto found = deferred_.find(e.get()); found != deferred_.end()) {
      if (const auto materialized = definitions_.find(e.get()); materialized != definitions_.end())
        return materialized->second;
      // Own only original coordinate context, never a copy of accumulated sites.
      const auto& original = found->second;
      const auto root = BuildInContext(original.value, *original.context, parameters, input);
      definitions_.emplace(e.get(), root);
      return root;
    }
    if (const auto found = deferred_selects_.find(e.get()); found != deferred_selects_.end()) {
      if (const auto materialized = definitions_.find(e.get()); materialized != definitions_.end())
        return materialized->second;
      const auto& original = found->second;
      const auto cond_id = BuildInContext(original.condition, *original.context, parameters, input);
      const auto then_id = BuildInContext(original.then_value, *original.context, parameters, input);
      const auto else_id = BuildInContext(original.else_value, *original.context, parameters, input);
      const auto root = MakeSelect(cond_id, then_id, else_id, input);
      definitions_.emplace(e.get(), root);
      return root;
    }
    // A named remainder need not itself be affine: buffer.slot consumes the
    // affine numerator and modulus as a rotating family. Ordinary coordinate
    // consumers still go through the original range/wrap proof below.
    if (auto found = remainders_.find(e.get()); found != remainders_.end())
      return BuildInContext(found->second, segment, parameters, input);
    if (auto found = definitions_.find(e.get()); found != definitions_.end()) {
      return found->second;
    }
    const auto key = std::make_pair(e.get(), &segment);
    if (auto found = memo_.find(key); found != memo_.end()) {
      return found->second;
    }
    if (auto cast = As<Cast>(e)) {
      const auto child = BuildInContext(cast->operand_, segment, parameters, input);
      if (maximum)
        CHECK(nodes_[child].upper <= *maximum) << "Synchronization coordinate cast may change its value";
      memo_.emplace(key, child);
      return child;
    }
    Node n{Kind::Constant};
    if (auto c = As<ConstInt>(e)) {
      n.lower = n.upper = n.factor = Constant(c);
      n.constant = true;
    } else if (auto b = As<ConstBool>(e)) {
      n.lower = n.upper = n.factor = b->value_ ? 1 : 0;
      n.constant = true;
    } else if (auto call = As<Call>(e); call && IsOp(call, "buffer.get_block_idx")) {
      CHECK(!input.invocation_coordinate) << "Access uses an unrelated original block coordinate";
      CHECK(input.block_bounds && call->args_.empty()) << "Launch coordinate has no established caller range";
      n.kind = Kind::Block;
      n.lower = input.block_bounds->first;
      n.upper = input.block_bounds->second;
    } else if (auto loop = std::find_if(segment.nested_frames.begin(), segment.nested_frames.end(),
                                        [&](const auto& f) { return e == f.loop->loop_var_; });
               loop != segment.nested_frames.end()) {
      if (loop->start_expression && loop->count <= 1) {
        // Uniform singleton, not a sampled IV. Preserve the source frame in
        // the region tree while substituting its proved invariant start.
        const auto id = BuildInContext(loop->start_expression, segment, parameters, input);
        memo_.emplace(key, id);
        return id;
      }
      n.kind = Kind::Loop;
      n.frame = loop->loop;
      n.start = loop->start;
      n.lower = n.start;
      n.step = loop->step;
      if (loop->start_expression) {
        n.left = BuildInContext(loop->start_expression, segment, parameters, input);
        n.dynamic_start = true;
        n.lower = nodes_[n.left].lower;
        n.upper = Checked(__int128(nodes_[n.left].upper) +
                          __int128(n.step) * std::max<int64_t>(0, loop->count - 1));
      } else {
        n.upper = Checked(__int128(n.start) + __int128(n.step) * std::max<int64_t>(0, loop->count - 1));
      }
    } else if (segment.original_loop && e == segment.original_loop->loop_var_) {
      n.kind = Kind::Loop;
      n.frame = segment.original_loop;
      n.start = segment.start;
      n.lower = n.start;
      n.step = segment.step;
      n.upper = Checked(__int128(n.start) + __int128(n.step) * std::max<int64_t>(0, segment.count - 1));
    } else if (auto it = parameters.find(e.get()); it != parameters.end()) {
      CHECK(input.parameter_bounds.count(it->second))
          << "Invocation coordinate has no established caller range";
      n.kind = Kind::Parameter;
      n.parameter = it->second;
      n.lower = input.parameter_bounds.at(it->second).first;
      n.upper = input.parameter_bounds.at(it->second).second;
    } else if (As<FloorMod>(e) || As<FloorDiv>(e)) {
      const auto mod = As<FloorMod>(e);
      const auto div = As<FloorDiv>(e);
      const auto lhs = mod ? mod->left_ : div->left_;
      const auto rhs = mod ? mod->right_ : div->right_;
      CHECK(SameCoordinateType(e, lhs) && SameCoordinateType(e, rhs))
          << "Coordinate quotient/remainder requires matching original types";
      n.left = BuildInContext(lhs, segment, parameters, input);
      n.right = BuildInContext(rhs, segment, parameters, input);
      const auto x = nodes_[n.left], y = nodes_[n.right];
      CHECK(y.constant && y.lower > 0)
          << "Coordinate quotient/remainder requires a positive constant divisor";
      n.kind = mod ? Kind::Remainder : Kind::FloorDivide;
      n.lower = mod ? 0 : x.lower / y.upper;
      n.upper = mod ? std::min(x.upper, y.upper - 1) : x.upper / y.lower;
      n.constant = x.constant && y.constant;
      if (n.constant) n.lower = n.upper = mod ? x.upper % y.upper : x.upper / y.upper;
      if (mod && y.constant && x.lower / y.upper == x.upper / y.upper) {
        n.kind = Kind::ModularShift;
        n.factor = (x.lower / y.upper) * y.upper;
        n.lower = x.lower - n.factor;
        n.upper = x.upper - n.factor;
      }
    } else if (is_bool) {
      if (const auto neg = As<Not>(e)) {
        n.left = BuildInContext(neg->operand_, segment, parameters, input);
        const auto x = nodes_[n.left];
        if (x.constant) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = x.upper == 0 ? 1 : 0;
          n.constant = true;
        } else {
          n.kind = Kind::Not;
          n.lower = 0;
          n.upper = 1;
        }
      } else if (const auto logic_and = As<And>(e)) {
        CHECK(SameCoordinateType(e, logic_and->left_) && SameCoordinateType(e, logic_and->right_))
            << "Coordinate logical operand requires BOOL type";
        n.left = BuildInContext(logic_and->left_, segment, parameters, input);
        n.right = BuildInContext(logic_and->right_, segment, parameters, input);
        const auto x = nodes_[n.left], y = nodes_[n.right];
        if ((x.constant && x.upper == 0) || (y.constant && y.upper == 0)) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = 0;
          n.constant = true;
        } else if (x.constant && y.constant) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = 1;
          n.constant = true;
        } else {
          n.kind = Kind::And;
          n.lower = 0;
          n.upper = 1;
        }
      } else if (const auto logic_or = As<Or>(e)) {
        CHECK(SameCoordinateType(e, logic_or->left_) && SameCoordinateType(e, logic_or->right_))
            << "Coordinate logical operand requires BOOL type";
        n.left = BuildInContext(logic_or->left_, segment, parameters, input);
        n.right = BuildInContext(logic_or->right_, segment, parameters, input);
        const auto x = nodes_[n.left], y = nodes_[n.right];
        if ((x.constant && x.upper != 0) || (y.constant && y.upper != 0)) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = 1;
          n.constant = true;
        } else if (x.constant && y.constant) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = 0;
          n.constant = true;
        } else {
          n.kind = Kind::Or;
          n.lower = 0;
          n.upper = 1;
        }
      } else {
        Kind cmp_kind = Kind::Constant;
        ExprPtr left, right;
        if (auto eq = As<Eq>(e)) {
          cmp_kind = Kind::Equal;
          left = eq->left_;
          right = eq->right_;
        } else if (auto ne = As<Ne>(e)) {
          cmp_kind = Kind::NotEqual;
          left = ne->left_;
          right = ne->right_;
        } else if (auto lt = As<Lt>(e)) {
          cmp_kind = Kind::LessThan;
          left = lt->left_;
          right = lt->right_;
        } else if (auto le = As<Le>(e)) {
          cmp_kind = Kind::LessEqual;
          left = le->left_;
          right = le->right_;
        } else if (auto gt = As<Gt>(e)) {
          cmp_kind = Kind::GreaterThan;
          left = gt->left_;
          right = gt->right_;
        } else if (auto ge = As<Ge>(e)) {
          cmp_kind = Kind::GreaterEqual;
          left = ge->left_;
          right = ge->right_;
        } else {
          CHECK(false) << "Unavailable affine synchronization access";
        }
        CHECK(SameCoordinateType(left, right)) << "Coordinate comparison requires matching operand types";
        n.left = BuildInContext(left, segment, parameters, input);
        n.right = BuildInContext(right, segment, parameters, input);
        const auto x = nodes_[n.left], y = nodes_[n.right];
        std::optional<bool> folded;
        switch (cmp_kind) {
          case Kind::Equal:
            if (x.constant && y.constant)
              folded = (x.upper == y.upper);
            else if (x.upper < y.lower || y.upper < x.lower)
              folded = false;
            break;
          case Kind::NotEqual:
            if (x.constant && y.constant)
              folded = (x.upper != y.upper);
            else if (x.upper < y.lower || y.upper < x.lower)
              folded = true;
            break;
          case Kind::LessThan:
            if (x.upper < y.lower)
              folded = true;
            else if (x.lower >= y.upper)
              folded = false;
            break;
          case Kind::LessEqual:
            if (x.upper <= y.lower)
              folded = true;
            else if (x.lower > y.upper)
              folded = false;
            break;
          case Kind::GreaterThan:
            if (x.lower > y.upper)
              folded = true;
            else if (x.upper <= y.lower)
              folded = false;
            break;
          case Kind::GreaterEqual:
            if (x.lower >= y.upper)
              folded = true;
            else if (x.upper < y.lower)
              folded = false;
            break;
          default:
            break;
        }
        if (folded) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = *folded ? 1 : 0;
          n.constant = true;
        } else {
          n.kind = cmp_kind;
          n.lower = 0;
          n.upper = 1;
        }
      }
    } else {
      ExprPtr left, right;
      const bool multiply = static_cast<bool>(As<Mul>(e));
      const bool subtract = static_cast<bool>(As<Sub>(e));
      if (auto add = As<Add>(e)) {
        left = add->left_;
        right = add->right_;
      } else if (auto sub = As<Sub>(e)) {
        left = sub->left_;
        right = sub->right_;
      } else if (auto mul = As<Mul>(e)) {
        left = mul->left_;
        right = mul->right_;
      } else {
        CHECK(false) << "Unavailable affine synchronization access";
      }
      CHECK(SameCoordinateType(e, left) && SameCoordinateType(e, right))
          << "Coordinate arithmetic requires matching operand types and explicit casts";
      n.left = BuildInContext(left, segment, parameters, input);
      n.right = BuildInContext(right, segment, parameters, input);
      // Copy scalar node metadata: recursive construction can reallocate nodes_.
      auto x = nodes_[n.left], y = nodes_[n.right];
      if (multiply) {
        CHECK(x.constant || y.constant) << "Nonaffine synchronization access";
        if (!y.constant) {
          std::swap(n.left, n.right);
          std::swap(x, y);
        }
        n.kind = Kind::Scale;
        n.factor = y.upper;
        n.upper = Checked(__int128(x.upper) * n.factor);
        n.lower = Checked(__int128(x.lower) * n.factor);
        n.constant = x.constant || !n.factor;
      } else if (subtract) {
        if (x.constant && y.constant) {
          n.kind = Kind::Constant;
          n.lower = n.upper = n.factor = Checked(__int128(x.upper) - y.upper);
          n.constant = true;
        } else {
          n.kind = Kind::Sub;
          n.lower = Checked(__int128(x.lower) - y.upper);
          n.upper = Checked(__int128(x.upper) - y.lower);
          n.constant = false;
        }
      } else {
        n.kind = Kind::Add;
        n.upper = Checked(__int128(x.upper) + y.upper);
        n.lower = Checked(__int128(x.lower) + y.lower);
        n.constant = x.constant && y.constant;
      }
    }
    // Each intermediate has been range-checked in its original machine type,
    // including operands of zero and values before a widening INDEX cast.
    if (n.kind == Kind::Loop) {
      // The execution domain, unlike a rectangular caller/start hull, excludes
      // padded iterations beyond the original stop.
      if (auto stop = As<ConstInt>(n.frame->stop_); stop && stop->value_ > n.lower)
        n.upper = std::min(n.upper, stop->value_ - 1);
      // Empty static bodies retain the prior representative admission bounds;
      // their zero visit count, not a contradictory negative range, records absence.
    }
    CHECK(n.lower >= 0 && n.upper <= (maximum ? *maximum : 1))
        << "Synchronization coordinate arithmetic may overflow its source type";
    const auto id = nodes_.size();
    nodes_.push_back(n);
    // This capability is distinct from bounded exact evaluation. Nonlinear
    // quotients/remainders never silently flatten into representative offsets.
    const bool affine = n.constant || n.kind == Kind::Parameter || n.kind == Kind::Block ||
                        (n.kind == Kind::Loop && (!n.dynamic_start || affine_[n.left])) ||
                        ((n.kind == Kind::Scale || n.kind == Kind::ModularShift) && affine_[n.left]) ||
                        ((n.kind == Kind::Add || n.kind == Kind::Sub) && affine_[n.left] && affine_[n.right]);
    affine_.push_back(affine);
    memo_.emplace(key, id);
    ++input.coordinate_nodes;
    return id;
  }
  size_t MakeSelect(size_t cond_id, size_t then_id, size_t else_id, CompactSyncInput& input) {
    const auto cond = nodes_[cond_id];
    if (cond.constant) return cond.upper != 0 ? then_id : else_id;
    if (then_id == else_id) return then_id;
    const auto x = nodes_[then_id], y = nodes_[else_id];
    Node n{Kind::Select};
    n.condition = cond_id;
    n.left = then_id;
    n.right = else_id;
    n.lower = std::min(x.lower, y.lower);
    n.upper = std::max(x.upper, y.upper);
    if (x.constant && y.constant && x.upper == y.upper) {
      n.kind = Kind::Constant;
      n.factor = x.upper;
      n.constant = true;
    }
    const auto id = nodes_.size();
    nodes_.push_back(n);
    affine_.push_back(n.constant);
    ++input.coordinate_nodes;
    return id;
  }
  void DefineSelect(const ExprPtr& var, const ExprPtr& cond, const ExprPtr& then_value,
                    const ExprPtr& else_value, const SyncSegment& segment, const ParameterIndices& parameters,
                    CompactSyncInput& input, bool lazy = false) {
    CHECK(SameCoordinateType(var, then_value) && SameCoordinateType(var, else_value))
        << "Coordinate selection requires matching operand types";
    if (lazy || (input.invocation_coordinate && input.invocation_coordinate->SupportsDefinition(var))) {
      deferred_selects_.emplace(var.get(), DeferredSelect{cond, then_value, else_value, ContextFor(segment)});
      return;
    }
    const auto cond_id = Build(cond, segment, parameters, input);
    const auto then_id = Build(then_value, segment, parameters, input);
    const auto else_id = Build(else_value, segment, parameters, input);
    definitions_.emplace(var.get(), MakeSelect(cond_id, then_id, else_id, input));
  }
  void Define(const ExprPtr& var, const ExprPtr& value, const SyncSegment& segment,
              const ParameterIndices& parameters, CompactSyncInput& input, bool lazy = false) {
    CHECK(SameCoordinateType(var, value)) << "Coordinate definition requires an explicit type conversion";
    if (lazy || (input.invocation_coordinate && input.invocation_coordinate->SupportsDefinition(var))) {
      deferred_.emplace(var.get(), Deferred{value, ContextFor(segment)});
      return;
    }
    if (input.invocation_coordinate && var == input.invocation_coordinate->Expression()) {
      definitions_.emplace(var.get(), Build(value, segment, parameters, input));
      return;
    }
    if (auto mod = As<FloorMod>(value); mod && As<ConstInt>(mod->right_)) {
      CHECK(SameCoordinateType(value, mod->left_) && SameCoordinateType(value, mod->right_) &&
            Constant(mod->right_) > 0)
          << "Named slot remainder requires matching types and a positive fixed modulus";
      // Validate operands at the definition's original scope now. This keeps
      // forward references, cycles, narrowing and unready values unavailable;
      // only the affine flattening of the remainder itself is deferred.
      Build(mod->left_, segment, parameters, input);
      Build(mod->right_, segment, parameters, input);
      remainders_.emplace(var.get(), value);
      return;
    }
    definitions_.emplace(var.get(), Build(value, segment, parameters, input));
  }
  bool IsAffine(size_t root) const { return affine_.at(root); }
  int64_t Upper(size_t root) const { return nodes_.at(root).upper; }
  void CheckScope(size_t root, const SyncSegment& segment) const {
    std::set<const ForStmt*> active;
    if (segment.original_loop) active.insert(segment.original_loop.get());
    for (const auto& frame : segment.nested_frames) active.insert(frame.loop.get());
    std::set<size_t> seen;
    std::vector<size_t> pending{root};
    while (!pending.empty()) {
      const auto id = pending.back();
      pending.pop_back();
      if (!seen.insert(id).second) continue;
      const auto& node = nodes_.at(id);
      if (node.kind == Kind::Loop) {
        CHECK(active.count(node.frame.get())) << "External coordinate belongs to another occurrence frame";
        if (node.dynamic_start) pending.push_back(node.left);
      } else if (node.kind == Kind::Scale || node.kind == Kind::ModularShift || node.kind == Kind::Not) {
        pending.push_back(node.left);
      } else if (node.kind == Kind::Add || node.kind == Kind::Sub || node.kind == Kind::FloorDivide ||
                 node.kind == Kind::Remainder || node.kind == Kind::Equal || node.kind == Kind::NotEqual ||
                 node.kind == Kind::LessThan || node.kind == Kind::LessEqual ||
                 node.kind == Kind::GreaterThan || node.kind == Kind::GreaterEqual ||
                 node.kind == Kind::And || node.kind == Kind::Or) {
        pending.push_back(node.left);
        pending.push_back(node.right);
      } else if (node.kind == Kind::Select) {
        pending.push_back(node.condition);
        pending.push_back(node.left);
        pending.push_back(node.right);
      }
    }
  }
  bool SameNodeExpr(size_t a, size_t b) const {
    if (a == b) return true;
    const auto& x = nodes_.at(a);
    const auto& y = nodes_.at(b);
    if (x.kind != y.kind || x.constant != y.constant || x.dynamic_start != y.dynamic_start ||
        x.factor != y.factor || x.start != y.start || x.step != y.step || x.parameter != y.parameter ||
        x.frame != y.frame)
      return false;
    if (x.constant) return x.upper == y.upper;
    if (x.kind == Kind::Loop) return !x.dynamic_start || SameNodeExpr(x.left, y.left);
    if (x.kind == Kind::Parameter || x.kind == Kind::Block) return true;
    if (x.kind == Kind::Scale || x.kind == Kind::ModularShift || x.kind == Kind::Not)
      return SameNodeExpr(x.left, y.left);
    if (x.kind == Kind::Select)
      return SameNodeExpr(x.condition, y.condition) && SameNodeExpr(x.left, y.left) &&
             SameNodeExpr(x.right, y.right);
    return SameNodeExpr(x.left, y.left) && SameNodeExpr(x.right, y.right);
  }
  struct LinearMatch {
    int64_t coeff = 0, offset = 0;
  };
  std::optional<LinearMatch> MatchLinearIn(size_t id, size_t target) const {
    if (SameNodeExpr(id, target)) return LinearMatch{1, 0};
    const auto& n = nodes_.at(id);
    if (n.constant) return LinearMatch{0, n.upper};
    if (n.kind == Kind::Scale) {
      const auto sub = MatchLinearIn(n.left, target);
      if (!sub) return std::nullopt;
      return LinearMatch{sub->coeff * n.factor, sub->offset * n.factor};
    }
    if (n.kind == Kind::ModularShift) {
      const auto sub = MatchLinearIn(n.left, target);
      if (!sub) return std::nullopt;
      return LinearMatch{sub->coeff, sub->offset - n.factor};
    }
    if (n.kind == Kind::Add || n.kind == Kind::Sub) {
      const auto l = MatchLinearIn(n.left, target);
      const auto r = MatchLinearIn(n.right, target);
      if (!l || !r) return std::nullopt;
      const int64_t sign = n.kind == Kind::Add ? 1 : -1;
      return LinearMatch{l->coeff + sign * r->coeff, l->offset + sign * r->offset};
    }
    return std::nullopt;
  }
  std::optional<SyncSwizzledTileMap> MatchSwizzled(size_t primary_id, size_t secondary_id,
                                                   int64_t primary_tiles, int64_t secondary_tiles,
                                                   int64_t row_tiles, int64_t col_tiles, int64_t rows,
                                                   int64_t columns, bool col_major_groups) const {
    const auto& sec = nodes_.at(secondary_id);
    if (sec.kind != Kind::Add) return std::nullopt;
    for (int swap = 0; swap < 2; ++swap) {
      const size_t scale_id = swap ? sec.right : sec.left;
      const size_t rem_id = swap ? sec.left : sec.right;
      const auto& scaled = nodes_.at(scale_id);
      const auto& rem = nodes_.at(rem_id);
      if (scaled.kind != Kind::Scale || scaled.factor <= 0) continue;
      const int64_t group_tiles = scaled.factor;
      if (rem.kind != Kind::Remainder && rem.kind != Kind::ModularShift) continue;
      if (nodes_.at(rem.right).upper != group_tiles || !nodes_.at(rem.right).constant) continue;
      if (rem.kind == Kind::ModularShift &&
          (nodes_[rem.left].lower / group_tiles != nodes_[rem.left].upper / group_tiles ||
           rem.factor != (nodes_[rem.left].lower / group_tiles) * group_tiles))
        continue;
      const size_t block_id = scaled.left;
      const size_t in_group_id = rem.left;
      const auto& block_node = nodes_.at(block_id);
      const auto& in_group_node = nodes_.at(in_group_id);
      if (block_node.kind != Kind::FloorDivide ||
          (in_group_node.kind != Kind::Remainder && in_group_node.kind != Kind::ModularShift))
        continue;
      if (!SameNodeExpr(block_node.left, in_group_node.left)) continue;
      if (!nodes_.at(block_node.right).constant || !nodes_.at(in_group_node.right).constant) continue;
      const int64_t group_size = nodes_[block_node.right].upper;
      if (group_size <= 0 || nodes_[in_group_node.right].upper != group_size) continue;
      if (in_group_node.kind == Kind::ModularShift &&
          (nodes_[in_group_node.left].lower / group_size != nodes_[in_group_node.left].upper / group_size ||
           in_group_node.factor != (nodes_[in_group_node.left].lower / group_size) * group_size))
        continue;
      const size_t tile_id = block_node.left;
      const auto& tile = nodes_.at(tile_id);
      if (tile.kind != Kind::Loop || group_size != __int128(primary_tiles) * group_tiles) continue;
      const int64_t full_groups = secondary_tiles / group_tiles;
      const int64_t tail_cols = secondary_tiles % group_tiles;
      const __int128 max_tiles =
          tail_cols == 0 ? __int128(primary_tiles) * secondary_tiles
                         : __int128(full_groups) * group_size + __int128(primary_tiles) * tail_cols;
      if (tile.upper >= max_tiles) continue;
      if (tail_cols != 0) continue;  // Exact group_size requires full tail divisor unless generalized.
      bool snake = false;
      const auto& pri = nodes_.at(primary_id);
      auto is_base_primary = [&](size_t id) {
        const auto& div = nodes_.at(id);
        return div.kind == Kind::FloorDivide && nodes_.at(div.right).constant &&
               nodes_[div.right].upper == group_tiles && SameNodeExpr(div.left, in_group_id);
      };
      if (is_base_primary(primary_id)) {
        snake = false;
      } else if (pri.kind == Kind::Select && is_base_primary(pri.right)) {
        const auto linear = MatchLinearIn(pri.left, pri.right);
        if (!linear || linear->coeff != -1 || linear->offset != primary_tiles - 1) continue;
        const auto& cond = nodes_.at(pri.condition);
        size_t parity_node = 0;
        if (cond.kind == Kind::Equal) {
          if (nodes_[cond.left].constant && nodes_[cond.left].upper == 1)
            parity_node = cond.right;
          else if (nodes_[cond.right].constant && nodes_[cond.right].upper == 1)
            parity_node = cond.left;
          else
            continue;
        } else {
          continue;
        }
        const auto& pnode = nodes_.at(parity_node);
        if (pnode.kind != Kind::Remainder || !nodes_[pnode.right].constant ||
            nodes_[pnode.right].upper != 2 || !SameNodeExpr(pnode.left, block_id))
          continue;
        snake = true;
      } else {
        continue;
      }
      return SyncSwizzledTileMap{
          tile.frame,       tile_id,     tile.dynamic_start ? std::optional<size_t>(tile.left) : std::nullopt,
          row_tiles,        col_tiles,   rows,
          columns,          group_tiles, group_size,
          col_major_groups, snake};
    }
    return std::nullopt;
  }
  std::shared_ptr<const SyncExternalFootprint> External(size_t row, size_t column, const SyncStorage& storage,
                                                        int64_t rows, int64_t columns) const {
    auto out = std::make_shared<SyncExternalFootprint>();
    // The importer alone owns the mutable builder. This alias is published only
    // after complete import, when the builder dies; consumers see immutable DAGs.
    out->program = program_;
    out->row = row;
    out->column = column;
    auto unscale = [&](size_t id, int64_t scale) -> std::optional<size_t> {
      if (scale == 1) return id;
      const auto& node = nodes_[id];
      if (node.kind == Kind::Scale && node.factor == scale) return node.left;
      return std::nullopt;
    };
    const auto r = unscale(row, rows), c = unscale(column, columns);
    if (!r || !c || storage.row_extent_parameter || storage.rows % rows || storage.columns % columns)
      return out;
    const auto row_tiles = storage.rows / rows, col_tiles = storage.columns / columns;
    const auto& quotient = nodes_[*r];
    const auto& remainder = nodes_[*c];
    if (quotient.kind == Kind::FloorDivide &&
        (remainder.kind == Kind::Remainder || remainder.kind == Kind::ModularShift) &&
        quotient.left == remainder.left && nodes_[quotient.right].upper == nodes_[remainder.right].upper) {
      const auto& tile = nodes_[quotient.left];
      const auto divisor = nodes_[quotient.right].upper;
      const bool band_ok =
          remainder.kind != Kind::ModularShift || (tile.lower / divisor == tile.upper / divisor &&
                                                   remainder.factor == (tile.lower / divisor) * divisor);
      if (band_ok && tile.kind == Kind::Loop && divisor == col_tiles &&
          tile.upper < __int128(row_tiles) * divisor) {
        // Paired quotient/remainder is an injective tile map: t = rowTile*C+colTile.
        out->row_major_tiles = SyncRowMajorTileMap{
            tile.frame, quotient.left, tile.dynamic_start ? std::optional<size_t>(tile.left) : std::nullopt,
            row_tiles,  divisor,       rows,
            columns};
        return out;
      }
    }
    if (auto swizzled = MatchSwizzled(*r, *c, row_tiles, col_tiles, row_tiles, col_tiles, rows, columns,
                                      /*col_major_groups=*/true)) {
      out->swizzled_tiles = std::move(*swizzled);
      return out;
    }
    if (auto swizzled = MatchSwizzled(*c, *r, col_tiles, row_tiles, row_tiles, col_tiles, rows, columns,
                                      /*col_major_groups=*/false)) {
      out->swizzled_tiles = std::move(*swizzled);
    }
    return out;
  }
  AffineCoordinate Flatten(size_t root, CompactSyncInput& input) const {
    // Construction IDs are topological. Propagate aggregate weights once per
    // reachable node, avoiding path enumeration through shared subexpressions.
    std::map<size_t, int64_t, std::greater<size_t>> pending{{root, 1}};
    AffineCoordinate result;
    __int128 origin = 0;
    auto shift_origin = [&](__int128 delta) {
      CHECK(!__builtin_add_overflow(origin, delta, &origin))
          << "Coordinate affine proof exceeds supported precision";
    };
    while (!pending.empty()) {
      const auto [id, weight] = *pending.begin();
      pending.erase(pending.begin());
      ++input.coordinate_flatten_visits;
      const auto& n = nodes_[id];
      auto add = [&](size_t child, int64_t w) {
        if (w) pending[child] = Checked(__int128(pending[child]) + w);
      };
      if (n.constant) {
        shift_origin(__int128(weight) * n.upper);
      } else if (n.kind == Kind::Parameter) {
        auto& coefficient = result.parameters[n.parameter];
        coefficient = Checked(__int128(coefficient) + weight);
      } else if (n.kind == Kind::Block) {
        // Anchor each proved case at its lower launch coordinate. Modular
        // rewrites may subtract a constant, but the final origin stays >=0.
        shift_origin(__int128(weight) * input.block_bounds->first);
        result.block = Checked(__int128(result.block) + weight);
      } else if (n.kind == Kind::Loop) {
        if (n.dynamic_start)
          add(n.left, weight);
        else
          shift_origin(__int128(weight) * n.start);
        auto& coefficient = result.frames[n.frame.get()];
        coefficient = Checked(__int128(coefficient) + __int128(weight) * n.step);
      } else if (n.kind == Kind::Scale) {
        add(n.left, Checked(__int128(weight) * n.factor));
      } else if (n.kind == Kind::ModularShift) {
        shift_origin(-__int128(weight) * n.factor);
        add(n.left, weight);
      } else if (n.kind == Kind::Add) {
        add(n.left, weight);
        add(n.right, weight);
      } else if (n.kind == Kind::Sub) {
        add(n.left, weight);
        add(n.right, -weight);
      } else {
        CHECK(false) << "Nonlinear coordinate requires an exact external footprint export";
      }
    }
    result.first = Checked(origin);
    input.coordinate_coefficient_entries +=
        result.parameters.size() + result.frames.size() + (result.block != 0);
    return result;
  }
};
std::vector<StmtPtr> Statements(const StmtPtr& s) {
  if (auto seq = As<SeqStmts>(s)) return seq->stmts_;
  return {s};
}
class Importer {
 public:
  // Native instruction, descriptor and control qualification is shared. Only
  // dependence clients construct storage effects and their selector evidence.
  Importer(bool recover_accesses, SyncExternalPolicy policy)
      : recover_accesses(recover_accesses),
        readonly_external(policy == SyncExternalPolicy::ReadOnlyProjection),
        disjoint_readonly(policy == SyncExternalPolicy::DisjointReadOnlyProjection) {}
  const bool recover_accesses, readonly_external, disjoint_readonly;

  bool DeferredCoordinates() const { return readonly_external || disjoint_readonly; }

  // Only original, nonaliasing formal roots are admitted for selective
  // projection. Actual operand effects are checked separately at every site.
  void RegisterExternalFormals(const FunctionPtr& owner) {
    std::set<const Expr*> bases;
    for (size_t i = 0; i < owner->params_.size(); ++i) {
      const auto& var = owner->params_[i];
      const auto tensor = As<TensorType>(var->GetType());
      if (!tensor) {
        continue;
      }
      const auto memref = tensor->memref_.value_or(nullptr);
      const auto* base = memref ? memref->base_.get() : var.get();
      CHECK(base && bases.insert(base).second)
          << "Disjoint read-only projection requires distinct original GM bases";
      external_formals.insert(var.get());
      if (owner->param_directions_[i] == ParamDirection::In) {
        projected_formals.insert(var.get());
      }
    }
    CHECK(!projected_formals.empty()) << "Disjoint read-only projection has no original input roots";
  }

  bool ProjectsExternal(const ExprPtr& expr) const {
    return readonly_external || projected_formals.count(expr.get()) != 0;
  }
  std::set<const Expr*> external_formals, projected_formals;
  std::set<const Var*> ready_scalars;
  std::set<const Expr*> external_roots;
  CompactSyncInput result;
  std::unordered_map<const Expr*, size_t> objects;
  // Vec descriptors retain their own geometry; all placed Vec bytes share one
  // physical domain. SSA roots, typed aliases and gaps do not create disjointness.
  struct VecView {
    int64_t address, pitch;
    BufferTypePtr type;
    // Byte translation in one original counted-loop frame. The maximum
    // displacement is proved against the parent view, not sampled at visit 0.
    int64_t step = 0, last_displacement = 0;
    ForStmtPtr frame;
  };
  std::unordered_map<const Expr*, VecView> vec_views;
  std::optional<size_t> vec_storage;
  struct SlotSelection {
    int64_t first, step;
    ForStmtPtr frame;
    // The defining invocation must enclose every use, even for a constant
    // selector. Arithmetic invariance is not permission to escape SSA scope.
    ForStmtPtr scope;
  };
  std::unordered_map<const Expr*, SlotSelection> slots;
  std::unordered_set<const Expr*> transposed_mat;
  std::unordered_map<size_t, size_t> placed_matrix_roots;
  std::unique_ptr<SyncControlRangeProver> prover;
  std::unique_ptr<SyncFootprintResolver> resolver;
  ParameterIndices parameters;
  CoordinateGraph coordinates;
  std::set<size_t> translated_storage;
  std::vector<std::pair<VarPtr, ExprPtr>> scalar_defs;

  static bool SameControlExpr(const ExprPtr& a, const ExprPtr& b) {
    try {
      return structural_equal(a, b, false);
    } catch (const StructuralComparisonUnsupported&) {
      return false;
    }
  }
  static ExprPtr ConstFromBound(const ExprPtr& like, int64_t value) {
    const auto st = As<ScalarType>(like->GetType());
    if (st && st->dtype_ == DataType::BOOL) return std::make_shared<ConstBool>(value != 0, like->span_);
    return std::make_shared<ConstInt>(value, st ? st->dtype_ : DataType::INDEX, like->span_);
  }
  void PopulateAnalyzerFrames(arith::Analyzer& analyzer, const SyncSegment& segment) {
    if (result.owner) {
      for (const auto& [param_idx, bound] : result.parameter_bounds) {
        if (param_idx < result.owner->params_.size()) {
          analyzer.const_int_bound.Update(result.owner->params_[param_idx], {bound.first, bound.second});
        }
      }
    }
    auto bind_frame = [&](const SyncLoopFrame& frame) {
      if (!frame.loop || frame.count <= 0) return;
      int64_t lo = frame.start;
      __int128 hi = __int128(frame.start) + __int128(frame.count - 1) * frame.step;
      if (frame.start_expression) {
        if (readonly_external) {
          if (const auto call = As<Call>(frame.start_expression);
              call && IsOp(call, "buffer.get_block_idx") && result.block_bounds) {
            lo = result.block_bounds->first;
            hi = __int128(result.block_bounds->second) + __int128(frame.count - 1) * frame.step;
          } else {
            return;
          }
        } else {
          SyncSegment empty_ctx;
          const auto start_id = coordinates.Build(frame.start_expression, empty_ctx, parameters, result);
          const auto& sn = coordinates.NodeAt(start_id);
          lo = sn.lower;
          hi = __int128(sn.upper) + __int128(frame.count - 1) * frame.step;
        }
      }
      if (const auto stop = As<ConstInt>(frame.loop->stop_); stop && stop->value_ > lo)
        hi = std::min<__int128>(hi, stop->value_ - 1);
      if (hi >= std::numeric_limits<int64_t>::max()) return;
      analyzer.const_int_bound.Update(frame.loop->loop_var_, {lo, static_cast<int64_t>(hi)});
    };
    bool has_original_frame = false;
    for (const auto& frame : segment.nested_frames) {
      bind_frame(frame);
      has_original_frame |= frame.loop == segment.original_loop;
    }
    // A structured leaf's original loop is already in nested_frames. Do not
    // overwrite its proved symbolic start with the leaf's numerical origin.
    if (segment.original_loop && !has_original_frame) {
      bind_frame({segment.original_loop, segment.count, segment.start, segment.step});
    }
  }
  void QualifyBranchCondition(const ExprPtr& condition, const SyncSegment& segment) {
    if (disjoint_readonly) {
      const auto root = coordinates.Build(condition, segment, parameters, result);
      coordinates.CheckScope(root, segment);
    }
  }

  std::optional<bool> EvaluateStaticBranchCondition(const ExprPtr& condition, const SyncSegment& segment) {
    if (!condition) return std::nullopt;
    arith::Analyzer analyzer;
    PopulateAnalyzerFrames(analyzer, segment);
    for (const auto& [var, value] : scalar_defs) {
      if (const auto st = As<ScalarType>(var->GetType());
          st && (st->dtype_ == DataType::INDEX || st->dtype_.IsInt() || st->dtype_ == DataType::BOOL)) {
        analyzer.rewrite_simplify.Update(var, analyzer.Simplify(value));
      }
    }

    VarPtr block_var;
    class SubstituteBlockIdx : public IRMutator {
     public:
      SubstituteBlockIdx(VarPtr bv, std::optional<std::pair<int64_t, int64_t>> bounds)
          : block_var(std::move(bv)), bounds_(bounds) {}
      VarPtr block_var;
      bool used = false;
      bool missing_bounds = false;

     protected:
      ExprPtr VisitExpr_(const CallPtr& call) override {
        if (IsOp(call, "buffer.get_block_idx") && call->args_.empty()) {
          if (!bounds_) {
            missing_bounds = true;
            return call;
          }
          used = true;
          return block_var;
        }
        return IRMutator::VisitExpr_(call);
      }
      std::optional<std::pair<int64_t, int64_t>> bounds_;
    } subst(std::make_shared<Var>("_sync_block_idx", std::make_shared<ScalarType>(DataType::INDEX),
                                  condition->span_),
            result.block_bounds);
    ExprPtr normalized = subst.VisitExpr(condition);
    if (subst.missing_bounds) return std::nullopt;
    if (subst.used && result.block_bounds) {
      analyzer.const_int_bound.Update(subst.block_var,
                                      {result.block_bounds->first, result.block_bounds->second});
    }

    std::function<std::optional<bool>(const ExprPtr&)> eval =
        [&](const ExprPtr& expr) -> std::optional<bool> {
      const ExprPtr s = analyzer.Simplify(expr);
      if (const auto c = As<ConstBool>(s)) return c->value_;
      if (const auto c = As<ConstInt>(s)) return c->value_ != 0;
      if (const auto n = As<Not>(s)) {
        const auto v = eval(n->operand_);
        return v ? std::optional<bool>(!*v) : std::nullopt;
      }
      if (const auto a = As<And>(s)) {
        const auto lv = eval(a->left_);
        if (lv == false) return false;
        const auto rv = eval(a->right_);
        if (rv == false) return false;
        if (lv == true && rv == true) return true;
        return std::nullopt;
      }
      if (const auto o = As<Or>(s)) {
        const auto lv = eval(o->left_);
        if (lv == true) return true;
        const auto rv = eval(o->right_);
        if (rv == true) return true;
        if (lv == false && rv == false) return false;
        return std::nullopt;
      }
      auto eval_cmp = [&](const BinaryExprPtr& cmp, int kind) -> std::optional<bool> {
        const auto lt = As<ScalarType>(cmp->left_->GetType());
        const auto rt = As<ScalarType>(cmp->right_->GetType());
        if (!lt || !rt || lt->dtype_ != rt->dtype_ || (!lt->dtype_.IsInt() && lt->dtype_ != DataType::INDEX))
          return std::nullopt;
        const ExprPtr diff =
            analyzer.Simplify(std::make_shared<Sub>(cmp->left_, cmp->right_, lt->dtype_, cmp->span_));
        const auto b = analyzer.const_int_bound(diff);
        if (b.min_value > b.max_value) return std::nullopt;
        const bool non_neg = b.min_value >= 0;
        const bool pos = b.min_value > 0;
        const bool non_pos = b.max_value <= 0;
        const bool neg = b.max_value < 0;
        switch (kind) {
          case 0:  // Eq
            if (non_neg && non_pos) return true;
            if (pos || neg) return false;
            return std::nullopt;
          case 1:  // Ne
            if (pos || neg) return true;
            if (non_neg && non_pos) return false;
            return std::nullopt;
          case 2:  // Lt
            if (neg) return true;
            if (non_neg) return false;
            return std::nullopt;
          case 3:  // Le
            if (non_pos) return true;
            if (pos) return false;
            return std::nullopt;
          case 4:  // Gt
            if (pos) return true;
            if (non_pos) return false;
            return std::nullopt;
          case 5:  // Ge
            if (non_neg) return true;
            if (neg) return false;
            return std::nullopt;
          default:
            return std::nullopt;
        }
      };
      if (const auto c = As<Eq>(s)) return eval_cmp(c, 0);
      if (const auto c = As<Ne>(s)) return eval_cmp(c, 1);
      if (const auto c = As<Lt>(s)) return eval_cmp(c, 2);
      if (const auto c = As<Le>(s)) return eval_cmp(c, 3);
      if (const auto c = As<Gt>(s)) return eval_cmp(c, 4);
      if (const auto c = As<Ge>(s)) return eval_cmp(c, 5);
      return std::nullopt;
    };
    return eval(normalized);
  }
  AffineCoordinate Coordinate(const ExprPtr& e, const SyncSegment& segment) {
    RequireIndex(e);
    auto out = coordinates.Flatten(coordinates.Build(e, segment, parameters, result), result);
    for (auto [frame, coefficient] : out.frames)
      CHECK(frame == segment.original_loop.get() ||
            std::any_of(segment.nested_frames.begin(), segment.nested_frames.end(),
                        [&](const auto& f) { return f.loop.get() == frame; }))
          << "Coordinate definition belongs to another occurrence frame";
    if (auto found = out.frames.find(segment.original_loop.get()); found != out.frames.end()) {
      out.second = found->second;
      out.frames.erase(found);
    }
    return out;
  }
  void ReadyScalar(const ExprPtr& expression, const SyncSegment& segment) const {
    CheckSyncScalarReadiness(expression, [&](const VarPtr& variable) {
      return parameters.count(variable.get()) || ready_scalars.count(variable.get()) ||
             (segment.original_loop && variable == segment.original_loop->loop_var_) ||
             std::any_of(segment.nested_frames.begin(), segment.nested_frames.end(),
                         [&](const auto& frame) { return variable == frame.loop->loop_var_; });
    });
  }
  void DefineCoordinate(const AssignStmtPtr& assign, const SyncSegment& segment) {
    const auto scalar_type = As<ScalarType>(assign->var_->GetType());
    CHECK(scalar_type)
        << "Nested or data-dependent storage/view definitions are outside this compact provider";
    CHECK(readonly_external || SyncCoordinateMaximum(assign->var_) || scalar_type->dtype_ == DataType::BOOL)
        << "Unavailable scalar coordinate type";
    CHECK(!parameters.count(assign->var_.get()) && !coordinates.Defined(assign->var_) &&
          (!segment.original_loop || assign->var_ != segment.original_loop->loop_var_))
        << "Invocation and loop coordinates must retain their original single-assignment identities";
    if (DeferredCoordinates()) {
      if (const auto call = As<Call>(assign->value_)) {
        const auto descriptor = backend::QualifyBufferSyncNonPayload(call, result.hardware_target);
        CHECK(descriptor && descriptor->SignalFree())
            << "Opaque scalar needs a ready native metadata contract";
        for (const auto& argument : call->args_) ReadyScalar(argument, segment);
      } else {
        ReadyScalar(assign->value_, segment);
      }
      ready_scalars.insert(assign->var_.get());
    }
    scalar_defs.emplace_back(assign->var_, assign->value_);
    coordinates.Define(assign->var_, assign->value_, segment, parameters, result, DeferredCoordinates());
  }
  static bool ValidatePureScalarArm(const StmtPtr& stmt, size_t expected_yields) {
    auto stmts = Statements(stmt);
    if (stmts.empty()) return false;
    const auto yield = As<YieldStmt>(stmts.back());
    if (!yield || yield->value_.size() != expected_yields) return false;
    for (size_t i = 0; i + 1 < stmts.size(); ++i) {
      if (const auto assign = As<AssignStmt>(stmts[i])) {
        if (!As<ScalarType>(assign->var_->GetType())) return false;
        if (const auto call = As<Call>(assign->value_); call && !IsOp(call, "buffer.get_block_idx"))
          return false;
      } else if (const auto nested = As<IfStmt>(stmts[i])) {
        if (nested->return_vars_.empty() || !nested->else_body_) return false;
        for (const auto& rv : nested->return_vars_)
          if (!As<ScalarType>(rv->GetType())) return false;
        if (!ValidatePureScalarArm(nested->then_body_, nested->return_vars_.size()) ||
            !ValidatePureScalarArm(*nested->else_body_, nested->return_vars_.size()))
          return false;
      } else {
        return false;
      }
    }
    return true;
  }
  static ExprPtr SubstituteScalarMap(const ExprPtr& expr, const std::map<const Var*, ExprPtr>& map) {
    if (map.empty()) return expr;
    class Substituter : public IRMutator {
      const std::map<const Var*, ExprPtr>& map_;
      ExprPtr VisitExpr_(const VarPtr& var) override {
        if (const auto found = map_.find(var.get()); found != map_.end()) {
          return found->second;
        }
        return var;
      }

     public:
      explicit Substituter(const std::map<const Var*, ExprPtr>& map) : map_(map) {}
    };
    return Substituter(map).VisitExpr(expr);
  }
  static std::set<const Var*> CollectVars(const ExprPtr& expr) {
    std::set<const Var*> vars;
    class Collector : public IRVisitor {
      std::set<const Var*>& vars_;
      void VisitExpr_(const VarPtr& var) override { vars_.insert(var.get()); }

     public:
      explicit Collector(std::set<const Var*>& vars) : vars_(vars) {}
    };
    Collector(vars).VisitExpr(expr);
    return vars;
  }
  std::vector<ExprPtr> ResolvePureScalarArm(const StmtPtr& arm, const SyncSegment& segment,
                                            const arith::AnalyzerPtr& analyzer,
                                            const std::map<const Var*, ExprPtr>& bool_defs,
                                            const std::map<const Var*, ExprPtr>& unconstrained_defs) {
    std::map<const Var*, ExprPtr> local_defs;
    auto eval_expr = [&](ExprPtr expr) -> ExprPtr {
      expr = SubstituteScalarMap(expr, local_defs);
      ExprPtr bound_probe = expr;
      for (size_t step = 0; step < unconstrained_defs.size(); ++step) {
        const auto next = SubstituteScalarMap(bound_probe, unconstrained_defs);
        if (next == bound_probe) break;
        bound_probe = next;
      }
      const auto b = analyzer->const_int_bound(bound_probe);
      if (b.is_const() && b.min_value != arith::ConstIntBound::kNegInf &&
          b.max_value != arith::ConstIntBound::kPosInf) {
        return ConstFromBound(expr, b.min_value);
      }
      return analyzer->Simplify(expr);
    };
    auto stmts = Statements(arm);
    for (size_t i = 0; i + 1 < stmts.size(); ++i) {
      if (const auto assign = As<AssignStmt>(stmts[i])) {
        const auto resolved = eval_expr(assign->value_);
        local_defs[assign->var_.get()] = resolved;
        if (const auto st = As<ScalarType>(assign->var_->GetType());
            st && (st->dtype_ == DataType::INDEX || st->dtype_.IsInt() || st->dtype_ == DataType::BOOL)) {
          const auto b = analyzer->const_int_bound(resolved);
          if (b.min_value <= b.max_value) analyzer->const_int_bound.Update(assign->var_, b);
        }
      } else if (const auto nested = As<IfStmt>(stmts[i])) {
        const auto cond = SubstituteScalarMap(nested->condition_, local_defs);
        const auto expanded_cond = SubstituteScalarMap(cond, bool_defs);
        std::vector<ExprPtr> then_res, else_res;
        {
          auto ctx = analyzer->GetConstraintContext(expanded_cond);
          then_res =
              ResolvePureScalarArm(nested->then_body_, segment, analyzer, bool_defs, unconstrained_defs);
        }
        {
          auto ctx = analyzer->GetConstraintContext(MakeNot(expanded_cond));
          else_res =
              ResolvePureScalarArm(*nested->else_body_, segment, analyzer, bool_defs, unconstrained_defs);
        }
        for (size_t k = 0; k < nested->return_vars_.size(); ++k) {
          const auto& rv = nested->return_vars_[k];
          if (SameControlExpr(then_res[k], else_res[k])) {
            local_defs[rv.get()] = then_res[k];
            const auto b = analyzer->const_int_bound(then_res[k]);
            if (b.min_value <= b.max_value) analyzer->const_int_bound.Update(rv, b);
          } else {
            return {};
          }
        }
      }
    }
    const auto yield = As<YieldStmt>(stmts.back());
    std::vector<ExprPtr> out;
    out.reserve(yield->value_.size());
    for (const auto& value : yield->value_) out.push_back(eval_expr(value));
    return out;
  }
  bool TryPureScalarBranch(const IfStmtPtr& branch, const SyncSegment& segment) {
    if (branch->return_vars_.empty() || !branch->else_body_) return false;
    for (const auto& rv : branch->return_vars_) {
      if (!As<ScalarType>(rv->GetType())) return false;
      if (parameters.count(rv.get()) || coordinates.Defined(rv) ||
          (segment.original_loop && rv == segment.original_loop->loop_var_))
        return false;
    }
    if (!ValidatePureScalarArm(branch->then_body_, branch->return_vars_.size()) ||
        !ValidatePureScalarArm(*branch->else_body_, branch->return_vars_.size()))
      return false;

    auto analyzer = std::make_shared<arith::Analyzer>();
    PopulateAnalyzerFrames(*analyzer, segment);
    std::map<const Var*, ExprPtr> bool_defs, int_defs;
    for (const auto& [var, value] : scalar_defs) {
      const auto st = As<ScalarType>(var->GetType());
      if (!st || (!st->dtype_.IsInt() && st->dtype_ != DataType::INDEX && st->dtype_ != DataType::BOOL))
        continue;
      const auto expanded = SubstituteScalarMap(value, bool_defs);
      if (st->dtype_ == DataType::BOOL) {
        bool_defs[var.get()] = expanded;
      } else {
        int_defs[var.get()] = value;
        const auto b = analyzer->const_int_bound(value);
        if (b.min_value <= b.max_value) analyzer->const_int_bound.Update(var, b);
      }
    }

    auto define_expr_result = [&](const VarPtr& rv, const ExprPtr& value) {
      const auto scalar_type = As<ScalarType>(rv->GetType());
      CHECK(readonly_external || SyncCoordinateMaximum(rv) || scalar_type->dtype_ == DataType::BOOL)
          << "Unavailable scalar coordinate type";
      if (DeferredCoordinates()) {
        ReadyScalar(value, segment);
        ready_scalars.insert(rv.get());
      }
      scalar_defs.emplace_back(rv, value);
      coordinates.Define(rv, value, segment, parameters, result, DeferredCoordinates());
    };
    auto define_select_result = [&](const VarPtr& rv, const ExprPtr& cond, const ExprPtr& then_val,
                                    const ExprPtr& else_val) {
      const auto scalar_type = As<ScalarType>(rv->GetType());
      CHECK(readonly_external || SyncCoordinateMaximum(rv) || scalar_type->dtype_ == DataType::BOOL)
          << "Unavailable scalar coordinate type";
      if (DeferredCoordinates()) {
        ReadyScalar(cond, segment);
        ReadyScalar(then_val, segment);
        ReadyScalar(else_val, segment);
        ready_scalars.insert(rv.get());
      }
      coordinates.DefineSelect(rv, cond, then_val, else_val, segment, parameters, result,
                               DeferredCoordinates());
    };

    const auto verdict = EvaluateStaticBranchCondition(branch->condition_, segment);
    const auto cond_expanded = SubstituteScalarMap(branch->condition_, bool_defs);
    const auto cond_vars = CollectVars(cond_expanded);
    std::map<const Var*, ExprPtr> unconstrained_defs;
    for (const auto& [k, v] : int_defs)
      if (!cond_vars.count(k)) unconstrained_defs[k] = v;
    if (verdict) {
      std::vector<ExprPtr> selected;
      {
        auto ctx = analyzer->GetConstraintContext(*verdict ? cond_expanded : MakeNot(cond_expanded));
        selected = ResolvePureScalarArm(*verdict ? branch->then_body_ : *branch->else_body_, segment,
                                        analyzer, bool_defs, unconstrained_defs);
      }
      if (selected.size() != branch->return_vars_.size()) return false;
      for (size_t k = 0; k < branch->return_vars_.size(); ++k)
        define_expr_result(branch->return_vars_[k], selected[k]);
      return true;
    }

    std::vector<ExprPtr> then_res, else_res;
    {
      auto ctx = analyzer->GetConstraintContext(cond_expanded);
      then_res = ResolvePureScalarArm(branch->then_body_, segment, analyzer, bool_defs, unconstrained_defs);
    }
    {
      auto ctx = analyzer->GetConstraintContext(MakeNot(cond_expanded));
      else_res = ResolvePureScalarArm(*branch->else_body_, segment, analyzer, bool_defs, unconstrained_defs);
    }
    if (then_res.size() != branch->return_vars_.size() || else_res.size() != branch->return_vars_.size())
      return false;
    for (size_t k = 0; k < branch->return_vars_.size(); ++k) {
      const auto& rv = branch->return_vars_[k];
      if (SameControlExpr(then_res[k], else_res[k])) {
        define_expr_result(rv, then_res[k]);
      } else {
        define_select_result(rv, branch->condition_, then_res[k], else_res[k]);
      }
    }
    return true;
  }
  void SelectSlot(const AssignStmtPtr& assign, const SyncSegment& segment) {
    auto call = As<Call>(assign->value_);
    OpRegistry::GetInstance().ValidateBufferCall(call);
    const auto id = Storage(call->args_[0]);
    const auto count = result.storage[id].slot_count;
    RequireIndex(call->args_[1]);
    ExprPtr index = coordinates.SlotExpression(call->args_[1]);
    bool modulo = false;
    if (auto mod = As<FloorMod>(index)) {
      RequireIndex(mod->left_);
      RequireIndex(mod->right_);
      CHECK(Constant(mod->right_) == count) << "Slot modulus must equal the allocation slot count";
      index = mod->left_;
      modulo = true;
    }
    const auto coordinate = Coordinate(index, segment);
    CHECK(coordinate.parameters.empty() && !coordinate.block)
        << "Parameterized slot phase requires an explicit phase certificate";
    auto frames = coordinate.frames;
    if (coordinate.second) frames.emplace(segment.original_loop.get(), coordinate.second);
    CHECK(frames.size() <= 1) << "Coupled slot coordinates require a multidimensional phase certificate";
    ForStmtPtr frame;
    int64_t stride = 0;
    if (!frames.empty()) {
      const auto [selected, coefficient] = *frames.begin();
      stride = coefficient;
      if (segment.original_loop.get() == selected) frame = segment.original_loop;
      for (const auto& parent : segment.nested_frames)
        if (parent.loop.get() == selected) frame = parent.loop;
      CHECK(frame) << "Slot selection has no original occurrence frame";
    }
    CHECK(modulo || (!stride && coordinate.first < count))
        << "Slot selection must be a constant in range or a proved nonnegative affine rotation";
    auto scope = segment.original_loop;
    if (!scope && !segment.nested_frames.empty()) scope = segment.nested_frames.back().loop;
    objects[assign->var_.get()] = id;
    if (transposed_mat.count(call->args_[0].get())) transposed_mat.insert(assign->var_.get());
    slots[assign->var_.get()] = {coordinate.first % count, stride % count, frame, scope};
  }
  void QualifySlot(const ExprPtr& handle, const SyncSegment& segment) const {
    if (auto slot = slots.find(handle.get()); slot != slots.end()) {
      const auto contains = [&](const ForStmtPtr& frame) {
        return !frame || frame == segment.original_loop ||
               std::any_of(segment.nested_frames.begin(), segment.nested_frames.end(),
                           [&](const auto& parent) { return parent.loop == frame; });
      };
      CHECK(contains(slot->second.frame) && contains(slot->second.scope))
          << "Slot selection belongs to another occurrence frame";
    }
  }
  void ApplySlot(SyncAccess& access, const ExprPtr& handle, const SyncSegment& segment) const {
    QualifySlot(handle, segment);
    if (auto slot = slots.find(handle.get()); slot != slots.end()) {
      access.slot_first = slot->second.first;
      access.slot_step = slot->second.step;
      access.slot_coordinate_frame = slot->second.frame;
    }
  }
  // All source organizations import the same supported definition forms. The
  // caller still controls whether this cut may declare placed storage; sharing
  // dispatch does not admit an allocation inside a previously qualified leaf.
  void Definition(const AssignStmtPtr& assign, const SyncSegment& segment, bool allocations = false) {
    const auto call = As<Call>(assign->value_);
    if (IsOp(call, "buffer.slot"))
      SelectSlot(assign, segment);
    else if (IsOp(call, "buffer.subview") || IsOp(call, "buffer.reshape"))
      View(assign, call, segment);
    else if (As<ScalarType>(assign->var_->GetType()) || !allocations)
      DefineCoordinate(assign, segment);
    else
      Allocate(assign);
  }
  size_t VecByteDomain(int64_t begin, int64_t end) {
    CHECK(begin >= 0 && end > begin) << "Synchronization requires a nonempty physical byte interval";
    for (const auto& old : result.storage) {
      CHECK(old.space != MemorySpace::Vec || old.byte_coordinates || begin >= old.address + old.bytes ||
            old.address >= end)
          << "Overlapping slot and ordinary Vec views require a common physical-layout partition";
    }
    if (!vec_storage) {
      vec_storage = result.storage.size();
      result.storage.push_back({"Vec physical bytes", MemorySpace::Vec, 0, 1, end, end, nullptr, true});
    } else {
      auto& domain = result.storage[*vec_storage];
      domain.bytes = domain.columns = std::max(domain.bytes, end);
    }
    return *vec_storage;
  }
  void RegisterVec(const VarPtr& var, VecView view) {
    const auto type = view.type;
    // A full-valid column vector [R,1] occupies consecutive elements in either
    // major order. This does not admit general column-major matrices or views.
    CHECK(type->shape_ == type->valid_shape_ &&
          (backend::DenseBufferBytes(type) || backend::IsColumnVectorBuffer(type) ||
           backend::IsFullPaddedVec(type)))
        << "Synchronization Vec views require static full-valid dense descriptors";
    const auto end =
        Checked(__int128(view.address) + view.last_displacement + __int128(type->shape_[0] - 1) * view.pitch +
                __int128(type->shape_[1]) * type->dtype_.GetByte());
    objects[var.get()] = VecByteDomain(view.address, end);
    vec_views.emplace(var.get(), std::move(view));
  }
  void View(const AssignStmtPtr& assign, const CallPtr& call, const SyncSegment& segment = {}) {
    OpRegistry::GetInstance().ValidateBufferCall(call);
    if (IsOp(call, "buffer.reshape")) {
      const auto src_type = As<BufferType>(call->args_[0]->GetType());
      const auto dst_type = As<BufferType>(assign->var_->GetType());
      if (src_type && dst_type && src_type->memory_space_ != MemorySpace::Vec) {
        const auto* resolved = resolver ? resolver->ViewOf(assign->var_) : nullptr;
        CHECK(resolved && resolved->valid_matrix_coords)
            << "Exact synchronization only supports layout-compatible or transposed Matrix reshapes";
        const auto sid = Storage(call->args_[0]);
        objects[assign->var_.get()] = sid;
        if (resolved->axis_swapped) transposed_mat.insert(assign->var_.get());
        return;
      }
    }
    auto source = vec_views.find(call->args_[0].get());
    CHECK(source != vec_views.end()) << "Synchronization storage/view needs a qualified Vec source";
    auto view = source->second;
    auto type = As<BufferType>(assign->var_->GetType());
    CHECK(type && type->shape_.size() == 2) << "Synchronization view requires rank two";
    if (IsOp(call, "buffer.subview")) {
      const auto offsets = As<MakeTuple>(call->args_[1]);
      const auto row = As<ConstInt>(offsets->elements_[0])
                           ? AffineCoordinate{Constant(offsets->elements_[0]), 0}
                           : Coordinate(offsets->elements_[0], segment);
      const auto column = As<ConstInt>(offsets->elements_[1])
                              ? AffineCoordinate{Constant(offsets->elements_[1]), 0}
                              : Coordinate(offsets->elements_[1], segment);
      CHECK(row.parameters.empty() && column.parameters.empty() && row.frames.empty() &&
            column.frames.empty() && !row.block && !column.block)
          << "Vec subview translation needs one original loop frame";
      CHECK(!view.frame || view.frame == segment.original_loop)
          << "Vec subview belongs to another occurrence frame";
      const auto last = std::max<int64_t>(0, segment.count - 1);
      CHECK(__int128(row.first) + __int128(last) * row.second + type->shape_[0] <= view.type->shape_[0] &&
            __int128(column.first) + __int128(last) * column.second + type->shape_[1] <= view.type->shape_[1])
          << "Vec subview exceeds its proved parent window";
      view.address = Checked(__int128(view.address) + __int128(row.first) * view.pitch +
                             __int128(column.first) * view.type->dtype_.GetByte());
      view.step = Checked(__int128(view.step) + __int128(row.second) * view.pitch +
                          __int128(column.second) * view.type->dtype_.GetByte());
      if (view.step) {
        CHECK(segment.original_loop) << "Moving Vec view has no original counted frame";
        view.frame = segment.original_loop;
        view.last_displacement = Checked(__int128(last) * view.step);
      }
      // Inherit the parent's pitch, including through nested subviews.
    } else {
      // A single-row Vec subview has no inter-row stride gap even when it
      // inherits a wider parent pitch (e.g., buffer.gather_compare's mask prefix).
      CHECK(view.type->shape_[0] == 1 ||
            view.pitch == Checked(__int128(view.type->shape_[1]) * view.type->dtype_.GetByte()))
          << "Synchronization cannot reshape a noncontiguous window";
      view.pitch = Checked(__int128(type->shape_[1]) * type->dtype_.GetByte());
    }
    view.type = type;
    RegisterVec(assign->var_, view);
  }
  size_t Storage(const ExprPtr& e) const {
    auto found = objects.find(e.get());
    CHECK(found != objects.end()) << "Synchronization access has no qualified storage identity";
    return found->second;
  }
  void Allocate(const AssignStmtPtr& a) {
    auto c = As<Call>(a->value_);
    if (c && (IsOp(c, "buffer.subview") || IsOp(c, "buffer.reshape"))) {
      View(a, c);
      return;
    }
    CHECK(c && (IsOp(c, "buffer.alloc") || IsOp(c, "buffer.alloc_multi")))
        << "Unavailable storage/view or scalar definition in synchronization import";
    OpRegistry::GetInstance().ValidateBufferCall(c);
    CHECK(c->args_.size() == 2) << "Synchronization import requires placed buffer allocation addresses";
    auto multi = As<MultiBufferType>(a->var_->GetType());
    auto t = multi ? multi->element_type_ : As<BufferType>(a->var_->GetType());
    const auto slot_count = multi ? multi->slot_count_ : 1;
    CHECK(t && t->shape_.size() == 2 && t->shape_ == t->valid_shape_ && t->dtype_.GetBit() % 8 == 0)
        << "Synchronization import needs fully valid byte-addressable rank-two buffers";
    CHECK(t->compact_ == CompactMode::null &&
          (t->pad_ == PadValue::null ||
           (!multi && (backend::IsFullPaddedVec(t) ||
                       backend::SupportsMatrixBufferPadding(t->memory_space_, t->dtype_, t->pad_, true)))))
        << "Packed/padded buffer interpretation is not qualified";
    auto valid = As<MakeTuple>(c->args_[0]);
    CHECK(valid && valid->elements_.empty()) << "Dynamic valid extents are not qualified";
    const auto address = Constant(c->args_[1]);
    const auto physical_bytes = backend::PhysicalBufferBytes(t);
    CHECK(physical_bytes) << "Synchronization import requires representable physical buffer bytes";
    const auto slot_bytes = Checked(*physical_bytes);
    if (multi && slot_count > 1) {
      // Native multi-tile lowering rounds each slot to the target alignment.
      // Until padded slot geometry is represented, accept only equal strides.
      const auto context = PassContext::Current();
      const auto* backend = context ? context->GetBackend() : backend::GetBackend();
      const auto alignment = backend->GetMemAlignment(t->memory_space_);
      CHECK(alignment && slot_bytes % alignment == 0 && address % alignment == 0)
          << "Multi-buffer placement requires aligned whole-slot geometry";
    }
    auto bytes = Checked(__int128(slot_bytes) * slot_count);
    Checked(__int128(address) + bytes);
    if (t->memory_space_ == MemorySpace::Vec && !multi) {
      RegisterVec(a->var_, {address, Checked(__int128(t->shape_[1]) * t->dtype_.GetByte()), t});
      return;
    }
    // A whole boxed buffer is exact only when no partial box is present.
    if (t->slayout_ != TileLayout::none_box) {
      CHECK(t->fractal_ && t->fractal_ % (16 * t->dtype_.GetByte()) == 0) << "Unavailable fractal geometry";
      auto columns = t->fractal_ / (16 * t->dtype_.GetByte());
      CHECK(t->shape_[0] % 16 == 0 && t->shape_[1] % columns == 0)
          << "Partial fractal-box footprints require separate qualification";
    }
    auto layout = t;
    if (IsSyncMatrixSpace(t->memory_space_)) {
      // The resolver already recognizes identical windows and Mat NZ/ZN
      // transposes. Keep incompatible windows distinct until all accesses
      // have been checked in their original view coordinates.
      const auto* view = resolver->ViewOf(a->var_);
      const auto* root = view ? resolver->Root(view->root_id) : nullptr;
      CHECK(root && root->canonical_layout) << "Matrix allocation has no placed storage identity";
      if (view->axis_swapped) {
        transposed_mat.insert(a->var_.get());
      }
      const auto [entry, inserted] = placed_matrix_roots.emplace(root->id, result.storage.size());
      if (!inserted) {
        objects[a->var_.get()] = entry->second;
        return;
      }
      // The resolver also indexes inactive original arms. Its canonical axes
      // still govern when the first imported handle is a transposed alias.
      layout = root->canonical_layout;
    } else {
      for (size_t i = 0; i < result.storage.size(); ++i) {
        const auto& old = result.storage[i];
        if (old.space != t->memory_space_ || address >= old.address + old.bytes ||
            old.address >= address + bytes) {
          continue;
        }
        CHECK(old.address == address && old.bytes == bytes && old.slot_count == slot_count && old.layout &&
              old.rows == t->shape_[0] && old.columns == t->shape_[1] && old.layout->dtype_ == t->dtype_ &&
              old.layout->fractal_ == t->fractal_ && old.layout->blayout_ == t->blayout_ &&
              old.layout->slayout_ == t->slayout_)
            << "Overlapping placed buffers need a common physical-layout partition";
        objects[a->var_.get()] = i;
        return;
      }
    }
    objects[a->var_.get()] = result.storage.size();
    result.storage.push_back({a->var_->name_hint_, layout->memory_space_, address, layout->shape_[0],
                              layout->shape_[1], bytes, layout});
    result.storage.back().slot_count = slot_count;
    result.storage.back().slot_bytes = slot_bytes;
  }
  SyncAccess Access(size_t storage, bool write, AffineCoordinate row, AffineCoordinate column, int64_t rows,
                    int64_t columns, const SyncSegment& segment) {
    const auto& s = result.storage[storage];
    std::set<const ForStmt*> moving;
    for (const auto& [frame, coefficient] : row.frames)
      if (coefficient) moving.insert(frame);
    for (const auto& [frame, coefficient] : column.frames)
      if (coefficient) moving.insert(frame);
    if (row.second || column.second) moving.insert(segment.original_loop.get());
    CHECK(moving.size() <= 1) << "Coupled moving coordinates need a multidimensional selector certificate";
    ForStmtPtr coordinate_frame;
    int64_t count = segment.count;
    if (!moving.empty()) {
      const auto* selected = *moving.begin();
      if (selected == segment.original_loop.get()) coordinate_frame = segment.original_loop;
      for (const auto& frame : segment.nested_frames)
        if (frame.loop.get() == selected) {
          coordinate_frame = frame.loop;
          count = frame.count;
        }
      CHECK(coordinate_frame) << "Moving access has no original occurrence frame";
      if (auto found = row.frames.find(selected); found != row.frames.end()) {
        row.second = found->second;
      }
      if (auto found = column.frames.find(selected); found != column.frames.end()) {
        column.second = found->second;
      }
    }
    const auto last = std::max<int64_t>(0, count - 1);
    auto translation_bound = [&](const SyncTranslation& translation) {
      int64_t upper = 0;
      for (auto [p, coefficient] : translation)
        upper = Checked(__int128(upper) + __int128(coefficient) * result.parameter_bounds.at(p).second);
      return upper;
    };
    CHECK(rows > 0 && columns > 0 &&
          Checked(__int128(row.first) + __int128(last) * row.second + rows +
                  translation_bound(row.parameters) +
                  __int128(row.block) *
                      (result.block_bounds ? result.block_bounds->second - result.block_bounds->first : 0)) <=
              (s.row_extent_parameter ? result.parameter_bounds.at(*s.row_extent_parameter).first : s.rows) &&
          Checked(__int128(column.first) + __int128(last) * column.second + columns +
                  translation_bound(column.parameters) +
                  __int128(column.block) *
                      (result.block_bounds ? result.block_bounds->second - result.block_bounds->first : 0)) <=
              s.columns)
        << "Synchronization access exceeds its proved allocation";
    SyncAccess access{storage,       write, row.first, column.first, row.second,
                      column.second, rows,  columns,   false};
    access.coordinate_frame = coordinate_frame;
    return access;
  }
  void CheckExternalEffects(const backend::BufferSyncOperation& operation, const CallPtr& call) {
    if (!DeferredCoordinates()) {
      return;
    }
    for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
      const auto effect = operation.ArgumentEffects()[index].data;
      if (effect == BufferAccess::None) continue;
      const auto type = call->args_[index]->GetType();
      const auto buffer = As<BufferType>(type);
      if (!As<TensorType>(type) && (!buffer || buffer->memory_space_ != MemorySpace::DDR)) continue;
      const auto& operand = call->args_[index];
      if (disjoint_readonly) {
        CHECK(external_formals.count(operand.get()) &&
              (IsOp(call, "buffer.load") || IsOp(call, "buffer.store")))
            << "Disjoint read-only projection requires direct original GM operands on qualified DMA";
      }
      if (ProjectsExternal(operand)) {
        CHECK(effect == BufferAccess::Read) << "Read-only external projection cannot omit an external write";
        CHECK(!disjoint_readonly || IsOp(call, "buffer.load"))
            << "Disjoint read-only projection requires a qualified DMA load";
        if (external_roots.insert(operand.get()).second) {
          result.projected_readonly_external_roots.push_back(operand);
        }
      }
    }
    for (const auto& effect : operation.Implementation().ImplicitEffects()) {
      if (effect.space == MemorySpace::DDR) {
        CHECK(!disjoint_readonly) << "Disjoint read-only projection requires explicit GM effects";
        CHECK(effect.data == BufferAccess::Read)
            << "Read-only external projection cannot omit an implicit external write";
      }
    }
  }

  SyncSite Site(const CallPtr& call, const SyncSegment& segment) {
    const auto operation =
        backend::QualifyBufferSyncOperation(call, result.owner->func_type_, result.hardware_target);
    const auto& implementation = operation.Implementation();
    CheckExternalEffects(operation, call);
    SyncSite site{call, nullptr, implementation.Pipe(), {}};
    // Track modes by operand, not storage: in-place operands can share bytes
    // while retaining different registered read/write modes.
    std::vector<std::pair<bool, bool>> modes(call->args_.size());
    auto whole = [&](size_t arg, bool write) {
      (write ? modes[arg].second : modes[arg].first) = true;
      auto id = Storage(call->args_[arg]);
      if (auto found = vec_views.find(call->args_[arg].get()); found != vec_views.end()) {
        const auto& view = found->second;
        CHECK(!view.frame || view.frame == segment.original_loop)
            << "Vec view use belongs to another occurrence frame";
        if (!recover_accesses) return;
        const auto width = Checked(__int128(view.type->shape_[1]) * view.type->dtype_.GetByte());
        // Dense footprints use one interval. Strided views retain one interval
        // per physical row, never an enclosing rectangle that fills the gaps.
        if (view.pitch == width) {
          site.accesses.push_back(Access(id, write, {0, 0}, {view.address, view.step}, 1,
                                         Checked(__int128(width) * view.type->shape_[0]), segment));
        } else {
          for (int64_t row = 0; row < view.type->shape_[0]; ++row)
            site.accesses.push_back(Access(
                id, write, {0, 0}, {Checked(__int128(view.address) + __int128(row) * view.pitch), view.step},
                1, width, segment));
        }
        return;
      }
      if (!recover_accesses) {
        QualifySlot(call->args_[arg], segment);
        return;
      }
      const auto& s = result.storage[id];
      site.accesses.push_back(Access(id, write, {0, 0}, {0, 0}, s.rows, s.columns, segment));
      ApplySlot(site.accesses.back(), call->args_[arg], segment);
    };
    auto enclosure = [&](size_t arg, bool write) {
      const auto first = site.accesses.size();
      whole(arg, write);
      for (size_t i = first; i < site.accesses.size(); ++i) site.accesses[i].exact_footprint = false;
    };
    if (IsOp(call, "buffer.load") || IsOp(call, "buffer.store")) {
      const bool load = IsOp(call, "buffer.load");
      const auto gm_arg = load ? 0 : 3;
      const bool projected = ProjectsExternal(call->args_[gm_arg]);
      const auto gm_mode = operation.ArgumentEffects()[gm_arg].data;
      const bool gm_read = gm_mode == BufferAccess::Read || gm_mode == BufferAccess::ReadWrite;
      const bool gm_write = gm_mode == BufferAccess::Write || gm_mode == BufferAccess::ReadWrite;
      auto offsets = As<MakeTuple>(call->args_[1]), shape = As<MakeTuple>(call->args_[2]);
      const auto gm_tensor = As<TensorType>(call->args_[gm_arg]->GetType());
      const bool rank1_gm = gm_tensor && gm_tensor->shape_.size() == 1 && offsets && shape &&
                            offsets->elements_.size() == 1 && shape->elements_.size() == 1 &&
                            !recover_accesses;
      const bool scalar_point = backend::SupportsBufferScalarTransfer(call->args_, load);
      CHECK(!scalar_point || !recover_accesses)
          << "Synchronization point DMA requires conservative GM and local carrier enclosures";
      CHECK(rank1_gm || (offsets && shape && offsets->elements_.size() == 2 && shape->elements_.size() == 2))
          << "Synchronization DMA requires a rank-two rectangle or conservative rank-one transfer";
      if (scalar_point || rank1_gm) {
        if (!scalar_point) {
          RequireIndex(offsets->elements_[0]);
        }
        if (projected) {
          CHECK(load && gm_read && !gm_write) << "Read-only external projection requires a read-only DMA";
        }
        for (const auto& offset : offsets->elements_) {
          if (projected) {
            ReadyScalar(offset, segment);
          } else {
            const auto off_root = coordinates.Build(offset, segment, parameters, result);
            coordinates.CheckScope(off_root, segment);
          }
        }
        modes[gm_arg] = {gm_read, gm_write};
        if (scalar_point) {
          // The alignment block is the local footprint enclosure. The
          // selected native transfer still uses the original unit GM view.
          enclosure(load ? 3 : 0, load);
        } else {
          const auto local = As<BufferType>(call->args_[load ? 3 : 0]->GetType());
          CHECK(local && local->shape_[0] == 1 &&
                (local->valid_shape_[1] == -1 || local->shape_[1] == Constant(shape->elements_[0])))
              << "Partial DMA local footprint is not qualified";
          CHECK(local->memory_space_ == MemorySpace::Vec) << "Rank-one DMA requires Vec storage";
          whole(load ? 3 : 0, load);
        }
      } else {
        // Coordinate readiness and checked arithmetic remain native admission
        // evidence even when no dependence footprint is requested.
        RequireIndex(offsets->elements_[0]);
        RequireIndex(offsets->elements_[1]);
        if (projected) {
          CHECK(load && gm_read && !gm_write) << "Read-only external projection requires a read-only DMA";
          ReadyScalar(offsets->elements_[0], segment);
          ReadyScalar(offsets->elements_[1], segment);
        } else {
          const auto row_root = coordinates.Build(offsets->elements_[0], segment, parameters, result);
          const auto column_root = coordinates.Build(offsets->elements_[1], segment, parameters, result);
          coordinates.CheckScope(row_root, segment);
          coordinates.CheckScope(column_root, segment);
          if (recover_accesses) {
            const auto id = Storage(call->args_[load ? 0 : 3]);
            auto& storage = result.storage[id];
            const auto rows = Constant(shape->elements_[0]), columns = Constant(shape->elements_[1]);
            CHECK(rows > 0 && columns > 0 &&
                  __int128(coordinates.Upper(row_root)) + rows <=
                      (storage.row_extent_parameter
                           ? result.parameter_bounds.at(*storage.row_extent_parameter).first
                           : storage.rows) &&
                  __int128(coordinates.Upper(column_root)) + columns <= storage.columns)
                << "Synchronization access exceeds its proved allocation";
            std::optional<AffineCoordinate> row, column;
            if (coordinates.IsAffine(row_root) && coordinates.IsAffine(column_root)) {
              row = Coordinate(offsets->elements_[0], segment);
              column = Coordinate(offsets->elements_[1], segment);
            }
            std::set<const ForStmt*> moving;
            if (row && column) {
              for (const auto& coordinate : {*row, *column}) {
                for (const auto& [frame, coefficient] : coordinate.frames)
                  if (coefficient) moving.insert(frame);
                if (coordinate.second) moving.insert(segment.original_loop.get());
              }
            }
            if (!row || !column || moving.size() > 1) {
              CHECK(storage.space == MemorySpace::DDR) << "External coordinate needs a global storage domain";
              SyncAccess access{id, gm_write, 0, 0, 0, 0, rows, columns};
              access.external = coordinates.External(row_root, column_root, storage, rows, columns);
              site.accesses.push_back(std::move(access));
            } else {
              if (translated_storage.insert(id).second) {
                storage.row_parameters = row->parameters;
                storage.column_parameters = column->parameters;
                storage.row_block = row->block;
                storage.column_block = column->block;
              } else {
                CHECK(storage.row_parameters == row->parameters &&
                      storage.column_parameters == column->parameters && storage.row_block == row->block &&
                      storage.column_block == column->block)
                    << "GM accesses need a common invocation translation; parameter-dependent overlap is "
                       "unproved";
              }
              site.accesses.push_back(Access(id, gm_write, *row, *column, rows, columns, segment));
            }
            if (gm_read && gm_write) {
              // Atomic accumulation reads and writes exactly the same GM window.
              // Both effects belong to this occurrence; no new storage root or
              // cross-worker ordering is inferred from the atomic mode.
              auto write = site.accesses.back();
              site.accesses.back().write = false;
              site.accesses.push_back(std::move(write));
            }
          }
        }
        modes[gm_arg] = {gm_read, gm_write};
        const auto local = As<BufferType>(call->args_[load ? 3 : 0]->GetType());
        CHECK(local && local->shape_[0] == Constant(shape->elements_[0]) &&
              local->shape_[1] == Constant(shape->elements_[1]))
            << "Partial DMA local footprint is not qualified";
        CHECK(local->memory_space_ == MemorySpace::Vec ||
              (load ? local->memory_space_ == MemorySpace::Mat : local->memory_space_ == MemorySpace::Acc))
            << "Only GM/Mat, Acc/GM and GM/Vec DMA contracts are qualified";
        whole(load ? 3 : 0, load);
      }
    } else if (IsOp(call, "buffer.copy") || IsOp(call, "buffer.extract")) {
      const bool extract = IsOp(call, "buffer.extract");
      const auto target = extract ? 3 : 1;
      auto sid = Storage(call->args_[0]), tid = Storage(call->args_[target]);
      const auto& s = result.storage[sid];
      const auto& t = result.storage[tid];
      if (!extract && implementation.Pipe() == PipeType::V) {
        resolver->RequireVecCopyPlacement(operation);
        whole(0, false);
        whole(target, true);
      } else if (extract && s.space == MemorySpace::Vec && t.space == MemorySpace::Vec) {
        auto row = Coordinate(call->args_[1], segment);
        auto column = Coordinate(call->args_[2], segment);
        CHECK(row.parameters.empty() && column.parameters.empty() && !row.block && !column.block)
            << "Parameterized local extraction needs a separate membership and alignment certificate";
        const auto& src_view = vec_views.at(call->args_[0].get());
        const auto& dst_view = vec_views.at(call->args_[target].get());
        const auto bound_in_source = [&](const AffineCoordinate& coord, int64_t window, int64_t extent) {
          if (coord.first < 0 || window <= 0) return false;
          __int128 max_offset = coord.first;
          const auto add_step = [&](int64_t step, int64_t count) {
            const auto last = std::max<int64_t>(0, count - 1);
            if (step < 0) return coord.first + __int128(last) * step >= 0;
            max_offset += __int128(last) * step;
            return true;
          };
          if (coord.second && !add_step(coord.second, segment.count)) return false;
          for (const auto& [frame, coeff] : coord.frames) {
            if (!coeff) continue;
            int64_t count = 1;
            for (const auto& nested : segment.nested_frames)
              if (nested.loop.get() == frame) count = nested.count;
            if (!add_step(coeff, count)) return false;
          }
          return max_offset + window <= extent;
        };
        CHECK(bound_in_source(row, dst_view.type->shape_[0], src_view.type->shape_[0]) &&
              bound_in_source(column, dst_view.type->shape_[1], src_view.type->shape_[1]))
            << "Synchronization access exceeds its proved allocation";
        // Vec->Vec extraction may span non-contiguous physical rows; mark a proper
        // subwindow read as an enclosure so precision certificates distinguish it
        // from an exact full-buffer footprint.
        const bool full_source = row.first == 0 && row.second == 0 && row.frames.empty() &&
                                 column.first == 0 && column.second == 0 && column.frames.empty() &&
                                 dst_view.type->shape_[0] == src_view.type->shape_[0] &&
                                 dst_view.type->shape_[1] == src_view.type->shape_[1];
        if (full_source) {
          whole(0, false);
        } else {
          enclosure(0, false);
        }
        whole(target, true);
      } else {
        CHECK(s.space == MemorySpace::Mat && (t.space == MemorySpace::Left || t.space == MemorySpace::Right))
            << "Only Mat to operand-buffer or Vec to Vec movement is qualified";
        auto row = extract ? Coordinate(call->args_[1], segment) : AffineCoordinate{};
        auto column = extract ? Coordinate(call->args_[2], segment) : AffineCoordinate{};
        CHECK(row.parameters.empty() && column.parameters.empty() && !row.block && !column.block)
            << "Parameterized local extraction needs a separate membership and alignment certificate";
        const bool transposed = transposed_mat.count(call->args_[0].get()) != 0;
        const int64_t ext_rows = transposed ? t.columns : t.rows;
        const int64_t ext_cols = transposed ? t.rows : t.columns;
        if (transposed) std::swap(row, column);
        if (!extract) CHECK(s.rows == ext_rows && s.columns == ext_cols) << "Partial copy is not qualified";
        if (s.layout->slayout_ != TileLayout::none_box) {
          const auto box = s.layout->fractal_ / (16 * s.layout->dtype_.GetByte());
          const auto aligned_frames = [](const auto& frames, int64_t alignment) {
            return std::all_of(frames.begin(), frames.end(),
                               [&](const auto& entry) { return entry.second % alignment == 0; });
          };
          CHECK(row.first % 16 == 0 && row.second % 16 == 0 && ext_rows % 16 == 0 &&
                column.first % box == 0 && column.second % box == 0 && ext_cols % box == 0 &&
                aligned_frames(row.frames, 16) && aligned_frames(column.frames, box))
              << "Unaligned boxed extraction requires a separate footprint contract";
        }
        auto source_access = Access(sid, false, row, column, ext_rows, ext_cols, segment);
        // The retained rectangle and physical layout are exact even when a
        // moving or boxed slice has no stationary contiguous interval export.
        // Consumers qualify their own representation: the periodic extractor
        // still refuses unsupported moving shared geometry; numerical phase
        // snapshots may specialize original coordinates before partitioning it.
        ApplySlot(source_access, call->args_[0], segment);
        if (recover_accesses) {
          site.accesses.push_back(std::move(source_access));
        }
        modes[0].first = true;
        whole(target, true);
      }
    } else if (IsOp(call, "buffer.insert")) {
      const auto sid = Storage(call->args_[0]), tid = Storage(call->args_[3]);
      const auto& s = result.storage[sid];
      const auto& t = result.storage[tid];
      CHECK(s.space == MemorySpace::Acc && t.space == MemorySpace::Mat)
          << "Only Acc to Mat FIXPIPE insertion is qualified";
      auto row = Coordinate(call->args_[1], segment);
      auto column = Coordinate(call->args_[2], segment);
      CHECK(row.parameters.empty() && column.parameters.empty() && !row.block && !column.block)
          << "Parameterized local insertion needs a separate membership and alignment certificate";
      const bool transposed = transposed_mat.count(call->args_[3].get()) != 0;
      const int64_t ins_rows = transposed ? s.columns : s.rows;
      const int64_t ins_cols = transposed ? s.rows : s.columns;
      if (transposed) std::swap(row, column);
      const auto aligned_frames = [](const auto& frames, int64_t alignment) {
        return std::all_of(frames.begin(), frames.end(),
                           [&](const auto& entry) { return entry.second % alignment == 0; });
      };
      const auto box = t.layout->slayout_ != TileLayout::none_box
                           ? t.layout->fractal_ / (16 * t.layout->dtype_.GetByte())
                           : 16;
      CHECK(row.first % 16 == 0 && row.second % 16 == 0 && ins_rows % 16 == 0 && column.first % 16 == 0 &&
            column.second % 16 == 0 && ins_cols % 16 == 0 && column.first % box == 0 &&
            column.second % box == 0 && ins_cols % box == 0 && aligned_frames(row.frames, 16) &&
            aligned_frames(column.frames, 16) && aligned_frames(column.frames, box))
          << "Unaligned boxed insertion requires a separate footprint contract";
      auto target_access = Access(tid, true, row, column, ins_rows, ins_cols, segment);
      ApplySlot(target_access, call->args_[3], segment);
      whole(0, false);
      if (recover_accesses) {
        site.accesses.push_back(std::move(target_access));
      }
      modes[3].second = true;
    } else if (IsOp(call, "buffer.matmul") || IsOp(call, "buffer.matmul_acc")) {
      whole(0, false);
      whole(1, false);
      if (IsOp(call, "buffer.matmul_acc")) whole(2, false);
      whole(2, true);
      // Initialization is an operation effect, independent of native completion
      // protection. Small valid tiles still overwrite rather than read L0C.
      site.initializes_accumulator = IsOp(call, "buffer.matmul");
      // Backend size eligibility does not certify a common physical slot or
      // absence of interference; those remain analysis obligations.
      if (implementation.AccumulatorSizeEligible()) {
        CHECK(result.storage[Storage(call->args_[2])].slot_count == 1)
            << "Rotating accumulator protection requires same-physical-slot qualification";
        const auto storage = Storage(call->args_[2]);
        site.protected_accumulator = SyncAccumulatorBinding{storage, storage};
      }
    } else if (IsOp(call, "buffer.transpose")) {
      CHECK(result.owner->func_type_ == FunctionType::AIV)
          << "Transpose synchronization requires an AIV leaf";
      whole(0, false);
      enclosure(1, true);
      whole(2, true);
    } else if (auto recipe = backend::FindBufferConversionRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV &&
            (recipe->kind == backend::BufferConversionKind::Cast ||
             recipe->kind == backend::BufferConversionKind::FillPad))
          << "Synchronization conversion import requires a qualified AIV cast or fillpad";
      if (recipe->kind == backend::BufferConversionKind::FillPad) {
        // TFillPad copies the live source rows, then initializes the
        // trailing rows. The source allocation encloses the native reads.
        enclosure(0, false);
      } else {
        // Shape-wide accesses are exact only for full-valid casts.
        for (const auto& argument : call->args_) {
          const auto type = As<BufferType>(argument->GetType());
          CHECK(type && type->valid_shape_ == type->shape_)
              << "Synchronization cast requires a full-valid footprint";
        }
        whole(0, false);
      }
      whole(1, true);
    } else if (IsOp(call, "buffer.ci_workspace")) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Sequence synchronization requires an AIV leaf";
      ReadyScalar(call->args_[0], segment);
      enclosure(1, true);
      whole(2, true);
    } else if (auto recipe = backend::FindBufferIndexedRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Indexed synchronization requires an AIV leaf";
      // The shared contract uses in-range physical element offsets into a
      // known local buffer, not arbitrary pointers. Data-dependent gather
      // reads and native scratch use are enclosed without inventing selectors.
      if (recipe->compare) {
        whole(0, false);
        ReadyScalar(call->args_[1], segment);
        enclosure(2, true);
        enclosure(3, true);
        whole(4, true);
        whole(5, true);
      } else {
        if (recipe->scatter) {
          whole(0, false);
        } else {
          enclosure(0, false);
          enclosure(2, true);
        }
        whole(1, false);
        // TSCATTER clears the entire destination before its indexed writes.
        whole(recipe->input_count, true);
      }
    } else if (auto recipe = backend::FindBufferAxisRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Axis synchronization requires an AIV leaf";
      whole(0, false);
      // The shared contract distinguishes writable reduction scratch from a
      // read-only broadcast operand, and single-input column reductions have no scratch.
      if (recipe->HasScratch()) {
        enclosure(1, true);
      } else if (recipe->InputCount() > 1) {
        whole(1, false);
      }
      whole(recipe->InputCount(), true);
    } else if (auto recipe = backend::FindBufferElementwiseRecipe(call->op_->name_)) {
      CHECK(result.owner->func_type_ == FunctionType::AIV) << "Vec synchronization requires an AIV leaf";
      for (size_t arg = 0; arg < recipe->inputs.size(); ++arg) {
        if (recipe->inputs[arg].kind == backend::BufferElementwiseOperandKind::Buffer) {
          whole(arg, false);
        } else if (recipe->inputs[arg].kind == backend::BufferElementwiseOperandKind::Workspace) {
          enclosure(arg, true);
        } else {
          CHECK(As<ConstFloat>(call->args_[arg]) || As<ConstInt>(call->args_[arg]))
              << "Synchronization scalar operand readiness is unavailable";
        }
      }
      whole(recipe->inputs.size(), true);
    } else {
      CHECK(false) << "Unavailable instruction contract in synchronization import: " << call->op_->name_;
    }
    // Geometry above qualifies one explicit payload contract. Validate every
    // operand's data modes against the shared static contract, including GM
    // Tensor operands. Metadata reads are retained there but do not establish
    // readiness or create payload-data accesses here.
    for (size_t arg = 0; arg < call->args_.size(); ++arg) {
      const auto effect = operation.ArgumentEffects()[arg].data;
      const auto [reads, writes] = modes[arg];
      CHECK(reads == (effect == BufferAccess::Read || effect == BufferAccess::ReadWrite) &&
            writes == (effect == BufferAccess::Write || effect == BufferAccess::ReadWrite))
          << "Imported data modes disagree with registered operation contract";
    }
    backend::RequireBufferSyncWorkspaceSeparation(
        operation, [&](const ExprPtr& operand) -> std::optional<backend::BufferSyncByteWindow> {
          if (const auto view = vec_views.find(operand.get()); view != vec_views.end()) {
            const auto& value = view->second;
            const int64_t begin =
                Checked(__int128(value.address) + std::min<int64_t>(0, value.last_displacement));
            const int64_t end =
                Checked(__int128(value.address) + std::max<int64_t>(0, value.last_displacement) +
                        __int128(value.type->shape_[0] - 1) * value.pitch +
                        __int128(value.type->shape_[1]) * value.type->dtype_.GetByte());
            return backend::BufferSyncByteWindow{begin, end};
          }
          if (resolver) {
            if (const auto hull = resolver->ByteHull(operand)) return hull;
          }
          const auto& storage = result.storage[Storage(operand)];
          return backend::BufferSyncByteWindow{storage.address,
                                               Checked(__int128(storage.address) + storage.bytes)};
        });
    for (const auto& effect : implementation.ImplicitEffects()) {
      if (readonly_external && effect.space == MemorySpace::DDR) continue;
      CHECK(effect.space == MemorySpace::Vec)
          << "Implicit native effects require a qualified common physical storage domain";
      // Reuse ordinary placed-byte identity: implicit scratch is not a private
      // disjoint root. Any explicit access to these bytes shares its conflicts.
      const auto id = VecByteDomain(effect.begin, effect.end);
      if (!recover_accesses) continue;
      for (bool write : {false, true}) {
        if (effect.data != BufferAccess::ReadWrite &&
            effect.data != (write ? BufferAccess::Write : BufferAccess::Read))
          continue;
        auto access = Access(id, write, {0, 0}, {effect.begin, 0}, 1, effect.end - effect.begin, segment);
        access.exact_footprint = effect.exact_footprint;
        site.accesses.push_back(std::move(access));
      }
    }
    return site;
  }
  CallPtr InitializationCall(const StmtPtr& statement, const SyncSegment& segment) {
    auto statements = Statements(statement);
    // A void original branch may terminate with the explicit empty yield
    // introduced by Buffer lowering. It carries no payload/result and is not
    // an additional event. Never skip a value-bearing or nonterminal yield.
    if (!statements.empty()) {
      if (const auto yield = As<YieldStmt>(statements.back()); yield && yield->value_.empty()) {
        statements.pop_back();
      }
    }
    CHECK(!statements.empty()) << "Conditional synchronization body must contain one payload";
    // Buffer lowering materializes slot handles at each original use. Such
    // definitions describe storage, not extra payload events. Qualify them in
    // their actual arm; do not move them or admit arbitrary conditional work.
    for (size_t i = 0; i + 1 < statements.size(); ++i) {
      const auto assign = As<AssignStmt>(statements[i]);
      CHECK(assign &&
            (As<ScalarType>(assign->var_->GetType()) || IsOp(As<Call>(assign->value_), "buffer.slot")))
          << "Initialization alternatives require only qualified scalar/slot definitions before one payload";
      Definition(assign, segment);
    }
    const auto eval = As<EvalStmt>(statements.back());
    CHECK(eval) << "Conditional synchronization body must end in one payload evaluation";
    const auto call = As<Call>(eval->expr_);
    CHECK(call) << "Submit/indirect execution is outside the synchronization prototype";
    return call;
  }
  bool SameInitializationOperand(const ExprPtr& a, const ExprPtr& b) const {
    // Matrix operands currently admit allocations and explicit slots, not
    // arbitrary subviews (View is Vec-only). If matrix views are introduced,
    // their exact offset/stride map must join this comparison before admission.
    if (a == b) return true;
    if (!structural_equal(a->GetType(), b->GetType()) || Storage(a) != Storage(b)) return false;
    const auto x = slots.find(a.get()), y = slots.find(b.get());
    if (x == slots.end() || y == slots.end()) return x == slots.end() && y == slots.end();
    return x->second.first == y->second.first && x->second.step == y->second.step &&
           x->second.frame == y->second.frame && x->second.scope == y->second.scope;
  }
  static bool IsInitializationContinuationBranch(const IfStmtPtr& branch) {
    if (!branch || !branch->else_body_ || !branch->return_vars_.empty()) return false;
    auto single_call = [](const StmtPtr& stmt) -> CallPtr {
      std::vector<StmtPtr> stmts;
      if (const auto seq = As<SeqStmts>(stmt)) {
        stmts = seq->stmts_;
      } else {
        stmts.push_back(stmt);
      }
      if (!stmts.empty()) {
        if (const auto yield = As<YieldStmt>(stmts.back()); yield && yield->value_.empty()) {
          stmts.pop_back();
        }
      }
      if (stmts.empty()) return nullptr;
      for (size_t i = 0; i + 1 < stmts.size(); ++i) {
        const auto assign = As<AssignStmt>(stmts[i]);
        if (!assign ||
            (!As<ScalarType>(assign->var_->GetType()) && !IsOp(As<Call>(assign->value_), "buffer.slot"))) {
          return nullptr;
        }
      }
      const auto eval = As<EvalStmt>(stmts.back());
      return eval ? As<Call>(eval->expr_) : nullptr;
    };
    const auto first = single_call(branch->then_body_);
    const auto later = single_call(*branch->else_body_);
    return first && later && IsOp(first, "buffer.matmul") && IsOp(later, "buffer.matmul_acc");
  }
  void Body(const StmtPtr& stmt, SyncSegment& segment) {
    bool preceding_matrix = false;
    for (const auto& s : Statements(stmt)) {
      if (auto assign = As<AssignStmt>(s)) {
        Definition(assign, segment);
      } else if (auto eval = As<EvalStmt>(s)) {
        auto call = As<Call>(eval->expr_);
        CHECK(call) << "Submit or non-call payload is not qualified";
        segment.sites.push_back(Site(call, segment));
        preceding_matrix |= segment.sites.back().pipe == PipeType::M;
      } else if (auto branch = As<IfStmt>(s)) {
        QualifyBranchCondition(branch->condition_, segment);
        if (!branch->return_vars_.empty() && TryPureScalarBranch(branch, segment)) continue;
        if (branch->return_vars_.empty() && !IsInitializationContinuationBranch(branch)) {
          if (const auto verdict = EvaluateStaticBranchCondition(branch->condition_, segment)) {
            result.static_branches[branch.get()] = *verdict;
            if (*verdict) {
              Body(branch->then_body_, segment);
            } else if (branch->else_body_) {
              Body(*branch->else_body_, segment);
            }
            continue;
          }
        }
        auto frames = segment.nested_frames;
        if (frames.empty() && segment.original_loop)
          frames.push_back({segment.original_loop, segment.count, segment.start, segment.step});
        auto is_frame_var = [&](const Expr* expr) {
          return std::any_of(frames.begin(), frames.end(),
                             [&](const auto& f) { return f.loop && f.loop->loop_var_.get() == expr; });
        };
        bool has_parameter_guard = false;
        std::map<const Expr*, int64_t> first_visits;
        std::function<bool(const ExprPtr&)> collect = [&](const ExprPtr& condition) {
          if (auto both = As<And>(condition)) return collect(both->left_) && collect(both->right_);
          auto eq = As<Eq>(condition);
          auto value = eq ? As<ConstInt>(eq->right_) : nullptr;
          if (!eq || !value) return false;
          const auto lhs_var = As<Var>(eq->left_);
          if (lhs_var && parameters.count(lhs_var.get())) {
            has_parameter_guard = true;
            return true;
          }
          if (lhs_var && is_frame_var(lhs_var.get())) {
            return first_visits.emplace(lhs_var.get(), value->value_).second;
          }
          if (!lhs_var || coordinates.Defined(lhs_var)) {
            if (const auto st = As<ScalarType>(eq->left_->GetType()); st && st->dtype_ == DataType::INDEX) {
              const auto coord = Coordinate(eq->left_, segment);
              std::map<const ForStmt*, int64_t> coord_frames = coord.frames;
              if (coord.second != 0 && segment.original_loop) {
                coord_frames[segment.original_loop.get()] = coord.second;
              }
              if (coord_frames.empty()) {
                has_parameter_guard = true;
                return true;
              }
              if (coord.parameters.empty() && coord.block == 0 && coord_frames.size() == 1) {
                const auto [frame, coeff] = *coord_frames.begin();
                if (coeff != 0 && (value->value_ - coord.first) % coeff == 0) {
                  return first_visits.emplace(frame->loop_var_.get(), (value->value_ - coord.first) / coeff)
                      .second;
                }
              }
              return false;
            }
          }
          return first_visits.emplace(eq->left_.get(), value->value_).second;
        };
        CHECK(segment.original_loop && collect(branch->condition_) && branch->else_body_ &&
              branch->return_vars_.empty())
            << "Only first-visit initialization/continuation selection is qualified";
        CHECK((!first_visits.empty() || has_parameter_guard) && first_visits.size() <= frames.size())
            << "Initialization guard has an unavailable occurrence frame";
        // The current shared boundary view changes one enclosing repeat seam.
        // More frames need continuation views at every intermediate seam, not
        // just at the outermost K reset; retain explicit unavailability.
        CHECK(first_visits.size() <= 2)
            << "Initialization across intermediate repetitions lacks continuation seam evidence";
        const size_t first_frame = frames.size() - first_visits.size();
        for (size_t f = first_frame; f < frames.size(); ++f) {
          CHECK(!frames[f].start_expression)
              << "Symbolic first-visit initialization needs its exact activation predicate";
          auto found = first_visits.find(frames[f].loop->loop_var_.get());
          CHECK(found != first_visits.end() && found->second == frames[f].start)
              << "Initialization guard must test consecutive first-visit frames";
        }
        auto first = InitializationCall(branch->then_body_, segment);
        auto later = InitializationCall(*branch->else_body_, segment);
        CHECK(IsOp(first, "buffer.matmul") && IsOp(later, "buffer.matmul_acc") &&
              first->args_.size() == later->args_.size() &&
              std::equal(first->args_.begin(), first->args_.end(), later->args_.begin(),
                         [&](const auto& a, const auto& b) { return SameInitializationOperand(a, b); }))
            << "Conditional payloads must preserve operands and destination";
        if (has_parameter_guard) {
          auto site = Site(later, segment);
          site.continuation = first;
          preceding_matrix = true;
          segment.sites.push_back(std::move(site));
        } else {
          auto site = Site(first, segment);
          auto continuation = Site(later, segment);
          site.continuation = later;
          if (frames[first_frame].loop != segment.original_loop)
            site.initialization_scope = frames[first_frame].loop;
          // A nonleading initialization has startup-only internal demands. Keep
          // the existing uniform family contract conservative for that site.
          if (preceding_matrix) site.protected_accumulator.reset();
          preceding_matrix = true;
          for (auto a : continuation.accesses)
            if (!a.write && result.storage[a.storage].space == MemorySpace::Acc) {
              a.after_first = true;
              site.accesses.push_back(a);
            }
          segment.sites.push_back(std::move(site));
        }
      } else {
        CHECK(As<YieldStmt>(s)) << "Nested or data-dependent control is outside this compact provider";
        CHECK(As<YieldStmt>(s)->value_.empty()) << "Loop-carried scalar readiness is not qualified";
      }
    }
  }
};
// Discover source shape in one postorder traversal. Nested loops retain their
// original sequence/repetition tree; flat constant loops use the leaf path.
std::pair<bool, bool> NestedShape(const StmtPtr& statement, std::map<const ForStmt*, bool>& nested) {
  if (!statement) return {false, false};
  if (auto loop = As<ForStmt>(statement)) {
    const auto child = NestedShape(loop->body_, nested);
    nested.emplace(loop.get(), child.first);
    // Invocation-varying starts need the source-frame-aware numerical path,
    // including a nonnested zero/one loop after analytic case qualification.
    return {true, !As<ConstInt>(loop->start_) || child.second || child.first};
  }
  std::pair<bool, bool> result{};
  auto join = [&](const StmtPtr& child) {
    const auto shape = NestedShape(child, nested);
    result.first |= shape.first;
    result.second |= shape.second;
  };
  if (auto sequence = As<SeqStmts>(statement)) {
    for (const auto& child : sequence->stmts_) join(child);
  } else if (auto branch = As<IfStmt>(statement)) {
    join(branch->then_body_);
    if (branch->else_body_) join(*branch->else_body_);
  }
  return result;
}
SyncLoopFrame NumericalFrame(Importer& importer, const ForStmtPtr& loop,
                             const std::vector<SyncLoopFrame>& parents) {
  RequireIndex(loop->loop_var_);
  RequireIndex(loop->start_);
  RequireIndex(loop->stop_);
  RequireIndex(loop->step_);
  CHECK(loop->kind_ == ForKind::Sequential && loop->iter_args_.empty() && loop->return_vars_.empty())
      << "Structured synchronization requires coordinate-only sequential loops";
  const auto stop = Constant(loop->stop_), step = Constant(loop->step_);
  CHECK(step > 0) << "Nonpositive structured loop step";
  if (!As<ConstInt>(loop->start_)) {
    SyncSegment context;
    context.nested_frames = parents;
    const auto start = importer.Coordinate(loop->start_, context);
    CHECK(start.frames.empty() && !start.second) << "Structured loop start is not invocation invariant";
    __int128 lo = start.first, hi = start.first;
    for (const auto& [p, coefficient] : start.parameters) {
      lo += __int128(coefficient) * importer.result.parameter_bounds.at(p).first;
      hi += __int128(coefficient) * importer.result.parameter_bounds.at(p).second;
    }
    if (start.block) {
      INTERNAL_CHECK(importer.result.block_bounds) << "Internal error: missing launch proof";
      hi += __int128(start.block) *
            (importer.result.block_bounds->second - importer.result.block_bounds->first);
    }
    // The invocation partition must establish one count over this entire
    // domain. Keep the original start, not a representative worker's IV.
    if (lo >= stop) return {loop, 0, 0, step, loop->start_};
    CHECK(lo >= 0 && hi < stop)
        << "Optional or varying outer visits need a guarded regional activation interface";
    const auto count = 1 + (__int128(stop) - lo - 1) / step;
    CHECK(count == 1 + (__int128(stop) - hi - 1) / step)
        << "Optional or varying outer visits need a guarded regional activation interface";
    Checked(hi + count * step);
    return {loop, Checked(count), 0, step, loop->start_};
  }
  const auto start = Constant(loop->start_);
  const auto count = stop <= start ? 0 : 1 + (stop - start - 1) / step;
  Checked(__int128(start) + __int128(count) * step);
  return {loop, count, start, step};
}
SyncRegion ImportStructuredBody(Importer& importer, const StmtPtr& statement,
                                std::vector<SyncLoopFrame>& parents,
                                const std::map<const ForStmt*, bool>& nested) {
  SyncRegion sequence;
  SyncSegment finite;
  finite.nested_frames = parents;
  auto add = [&](SyncSegment segment) {
    const auto local_count = segment.count;
    segment.count = 1;
    for (const auto& frame : parents) segment.count = Checked(__int128(segment.count) * frame.count);
    segment.nested_frames = parents;
    if (segment.original_loop) segment.reset_period = local_count;
    SyncRegion leaf;
    leaf.leaf = importer.result.segments.size();
    importer.result.segments.push_back(std::move(segment));
    sequence.children.push_back(std::move(leaf));
  };
  auto flush = [&]() {
    if (!finite.sites.empty()) {
      add(std::move(finite));
      finite = SyncSegment{};
      finite.nested_frames = parents;
    }
  };
  for (const auto& child : Statements(statement)) {
    if (auto loop = As<ForStmt>(child)) {
      flush();
      const auto frame = NumericalFrame(importer, loop, parents);
      if (!frame.count && frame.start_expression) {
        // This complete case proves the original loop absent. Keep its frame,
        // but do not fabricate payload effects for unexecuted operations.
        SyncRegion absent;
        absent.repetition = frame;
        sequence.children.push_back(std::move(absent));
        continue;
      }
      parents.push_back(frame);
      if (nested.at(loop.get())) {
        auto region = ImportStructuredBody(importer, loop->body_, parents, nested);
        region.repetition = frame;
        sequence.children.push_back(std::move(region));
      } else {
        SyncSegment segment;
        segment.original_loop = loop;
        segment.start = frame.start;
        segment.step = frame.step;
        segment.count = frame.count;
        segment.nested_frames = parents;
        importer.Body(loop->body_, segment);
        add(std::move(segment));
      }
      parents.pop_back();
    } else if (auto branch = As<IfStmt>(child)) {
      importer.QualifyBranchCondition(branch->condition_, finite);
      if (branch->return_vars_.empty() && !Importer::IsInitializationContinuationBranch(branch)) {
        if (const auto verdict = importer.EvaluateStaticBranchCondition(branch->condition_, finite)) {
          importer.result.static_branches[branch.get()] = *verdict;
          const std::optional<StmtPtr> live_arm =
              *verdict ? std::optional<StmtPtr>(branch->then_body_) : branch->else_body_;
          if (live_arm) {
            std::function<bool(const StmtPtr&)> has_for = [&](const StmtPtr& st) {
              if (As<ForStmt>(st) || As<WhileStmt>(st)) return true;
              if (const auto s = As<SeqStmts>(st))
                return std::any_of(s->stmts_.begin(), s->stmts_.end(), has_for);
              if (const auto b = As<IfStmt>(st))
                return has_for(b->then_body_) || (b->else_body_ && has_for(*b->else_body_));
              return false;
            };
            if (has_for(*live_arm)) {
              flush();
              auto sub = ImportStructuredBody(importer, *live_arm, parents, nested);
              for (auto& c : sub.children) sequence.children.push_back(std::move(c));
            } else {
              importer.Body(*live_arm, finite);
            }
          }
          continue;
        }
      }
      importer.Body(child, finite);
    } else if (auto assign = As<AssignStmt>(child)) {
      importer.Definition(assign, finite, true);
    } else if (!As<YieldStmt>(child) && !As<ReturnStmt>(child)) {
      importer.Body(child, finite);
    }
  }
  flush();
  return sequence;
}
// Qualify startup at its actual source scope, not at a sampled leaf visit.
// The only changed effect between startup and continuation is the first
// accumulator read. No earlier M or access through a physical alias is allowed.
// Summary maps are proportional to supplied source interfaces, not trip counts.
struct StartupSummary {
  using Site = std::pair<size_t, size_t>;
  std::optional<Site> first_matrix;
  std::map<size_t, Site> first_access;
  std::map<const ForStmt*, Site> pending;
};
StartupSummary QualifyStartupScopes(SyncRegion& region, const CompactSyncInput& input,
                                    const std::set<const ForStmt*>& phase_scopes) {
  StartupSummary out;
  if (region.leaf) {
    const auto& segment = input.segments[*region.leaf];
    const bool live = !segment.original_loop || segment.nested_frames.back().count != 0;
    for (size_t s = 0; s < segment.sites.size(); ++s) {
      const auto& site = segment.sites[s];
      const auto position = std::make_pair(*region.leaf, s);
      if (live && site.pipe == PipeType::M && !out.first_matrix) out.first_matrix = position;
      if (live)
        for (const auto& access : site.accesses) out.first_access.emplace(access.storage, position);
      if (site.initialization_scope)
        CHECK(out.pending.emplace(site.initialization_scope.get(), position).second)
            << "Multiple initializers in one named scope are unavailable";
    }
  } else {
    for (auto& child : region.children) {
      auto next = QualifyStartupScopes(child, input, phase_scopes);
      if (!out.first_matrix) out.first_matrix = next.first_matrix;
      out.first_access.insert(next.first_access.begin(), next.first_access.end());
      for (const auto& [scope, site] : next.pending)
        CHECK(out.pending.emplace(scope, site).second)
            << "Multiple initializers in one named scope are unavailable";
    }
  }
  if (region.repetition) {
    auto found = out.pending.find(region.repetition->loop.get());
    if (found != out.pending.end()) {
      const auto [leaf, index] = found->second;
      const auto& site = input.segments[leaf].sites[index];
      if (out.first_matrix) {
        std::optional<size_t> initialized_storage;
        for (const auto& access : site.accesses)
          if (access.write && input.storage[access.storage].space == MemorySpace::Acc) {
            CHECK(!initialized_storage || *initialized_storage == access.storage)
                << "Scoped initialization requires one accumulator footprint";
            initialized_storage = access.storage;
          }
        // The phase provider reduces the actual finite startup separately.
        // That construction needs first-touch initialization, not the hardware
        // >=10-tile exemption. Keep the older invariant provider's stronger
        // entry contract until its unprotected seam is qualified independently.
        CHECK(initialized_storage && site.initializes_accumulator &&
              (site.protected_accumulator || phase_scopes.count(region.repetition->loop.get())) &&
              out.first_matrix == found->second && out.first_access.at(*initialized_storage) == found->second)
            << "Scoped initialization requires first-touch qualified accumulator evidence";
        region.startup = found->second;
      }
      out.pending.erase(found);
    }
    // An empty descendant cannot supply an ancestor's entry evidence. Keep
    // pending scope names for validation, but discard its unexecuted effects.
    if (!region.repetition->count) {
      out.first_matrix.reset();
      out.first_access.clear();
    }
  }
  return out;
}
// Reuse PyPTO's bound environment, but qualify integer operations before
// accepting mathematical bounds as machine-index evidence. Saturation, unknown
// parameters and data-loaded values remain unproved. No branch is sampled.
class InvocationBounds : public IRVisitor {
  using Bound = std::pair<int64_t, int64_t>;
  struct TiledLoop {
    ExprPtr extent;
    int64_t tile;
  };
  std::shared_ptr<arith::Analyzer> analyzer_ = std::make_shared<arith::Analyzer>();
  std::vector<VarPtr> definitions_;
  std::set<const Expr*> unproved_definitions_;
  std::unordered_map<const Var*, ExprPtr> scalar_definitions_;
  std::unordered_map<const Var*, TiledLoop> tiled_loops_;
  const std::map<std::string, FunctionPtr>& functions_;
  std::map<std::string, std::map<size_t, std::optional<Bound>>>& evidence_;
  std::map<std::string, std::optional<Bound>>& launches_;
  std::map<std::string, std::optional<Bound>>& counts_;
  FunctionPtr caller_;
  std::optional<Bound> block_index_, block_count_;
  SyncParameterWork& work_;
  std::unordered_map<const Expr*, std::optional<Bound>> ranges_;
  std::set<const Expr*> visited_;
  void BindHostShapeSymbols() {
    if (caller_->func_type_ != FunctionType::Orchestration) return;
    // Orchestration emits an INDEX symbol from the uint32_t Tensor::shapes
    // field only if its C++ name is not reserved by a scalar parameter or
    // body definition. Mirror that precedence, including SSA name suffixes.
    class LocalDefinitions : public var_collectors::VarDefUseCollector {
      void VisitExpr(const ExprPtr&) override {}
    } locals;
    locals.VisitStmt(caller_->body_);
    std::unordered_set<std::string> reserved;
    for (const auto* definition : locals.var_defs)
      reserved.insert(auto_name::GetCompatibleBaseName(definition->name_hint_));
    for (const auto& parameter : caller_->params_) {
      if (!AsTensorTypeLike(parameter->GetType()))
        reserved.insert(auto_name::GetCompatibleBaseName(parameter->name_hint_));
    }
    for (const auto& parameter : caller_->params_) {
      const auto tensor = AsTensorTypeLike(parameter->GetType());
      if (!tensor) continue;
      for (const auto& dimension : tensor->shape_) {
        const auto symbol = As<Var>(dimension);
        const auto scalar = symbol ? As<ScalarType>(symbol->GetType()) : nullptr;
        if (scalar && scalar->dtype_ == DataType::INDEX &&
            !reserved.count(auto_name::GetCompatibleBaseName(symbol->name_hint_))) {
          analyzer_->const_int_bound.Update(symbol, {0, std::numeric_limits<uint32_t>::max()});
        }
      }
    }
  }
  // Expressions may be shared across scopes; cached unknowns and successes
  // both belong to the current bound environment, never to the entire program.
  void Invalidate() {
    ranges_.clear();
    visited_.clear();
  }
  std::optional<Bound> Range(const ExprPtr& expression) {
    if (auto found = ranges_.find(expression.get()); found != ranges_.end()) {
      return found->second;
    }
    ++work_.range_visits;
    auto result = ComputeRange(expression);
    ranges_.emplace(expression.get(), result);
    return result;
  }
  // Only definitions proved at their original cuts are expanded. Collapse
  // aliases on insertion, so lookup does not replay earlier assignments.
  ExprPtr Definition(const ExprPtr& expression) const {
    if (const auto variable = As<Var>(expression)) {
      if (const auto it = scalar_definitions_.find(variable.get()); it != scalar_definitions_.end())
        return it->second;
    }
    return expression;
  }
  std::optional<TiledLoop> QualifyTiledLoop(const ForStmtPtr& loop) const {
    if (!IsOrchestrationLike(caller_->func_type_) || loop->kind_ != ForKind::Sequential ||
        GetScalarDtype(loop->loop_var_) != DataType::INDEX)
      return std::nullopt;
    const auto start = As<ConstInt>(Definition(loop->start_));
    const auto step = As<ConstInt>(Definition(loop->step_));
    const auto quotient = As<FloorDiv>(Definition(loop->stop_));
    if (!start || start->value_ != 0 || !step || step->value_ != 1 || !quotient ||
        !SameCoordinateType(loop->loop_var_, quotient))
      return std::nullopt;
    const auto tile = As<ConstInt>(Definition(quotient->right_));
    const auto numerator = As<Add>(Definition(quotient->left_));
    if (!tile || tile->value_ <= 0 || !numerator) return std::nullopt;
    auto extent = numerator->left_;
    auto bias = As<ConstInt>(Definition(numerator->right_));
    if (!bias || bias->value_ != tile->value_ - 1) {
      extent = numerator->right_;
      bias = As<ConstInt>(Definition(numerator->left_));
    }
    if (!bias || bias->value_ != tile->value_ - 1) return std::nullopt;
    return TiledLoop{Definition(extent), tile->value_};
  }
  bool IsRemainingTiledExtent(const ExprPtr& extent, const ExprPtr& origin) const {
    if (tiled_loops_.empty()) return false;
    const auto product = As<Mul>(Definition(origin));
    if (!product) return false;
    auto iv = As<Var>(Definition(product->left_));
    auto tile = As<ConstInt>(Definition(product->right_));
    if (!iv || !tile) {
      iv = As<Var>(Definition(product->right_));
      tile = As<ConstInt>(Definition(product->left_));
    }
    if (!iv || !tile) return false;
    const auto active = tiled_loops_.find(iv.get());
    return active != tiled_loops_.end() && tile->value_ == active->second.tile &&
           Definition(extent) == active->second.extent;
  }
  static bool IsLaunchCapacity(const ExprPtr& expression) {
    const auto call = As<Call>(expression);
    return call && call->args_.empty() &&
           (IsOp(call, "system.available_cluster_count") || IsOp(call, "system.available_aiv_count"));
  }
  std::optional<Bound> ComputeRange(const ExprPtr& expression) {
    const auto maximum = SyncCoordinateMaximum(expression);
    if (!maximum) return std::nullopt;
    // Orchestration reads the runtime's nonnegative INT32 launch capacity.
    // A dominated nonempty launch can narrow this to a positive worker stride.
    if (IsLaunchCapacity(expression)) {
      return Bound{0, *maximum};
    }
    if (auto call = As<Call>(expression);
        call && (IsOp(call, "buffer.get_block_idx") || IsOp(call, "tile.get_block_idx"))) {
      return block_index_;
    }
    if (auto call = As<Call>(expression);
        call && (IsOp(call, "buffer.get_block_num") || IsOp(call, "tile.get_block_num")))
      return block_count_;
    if (auto cast = As<Cast>(expression)) {
      auto source = Range(cast->operand_);
      return source && source->second <= *maximum ? source : std::nullopt;
    }
    if (As<Var>(expression) || As<ConstInt>(expression)) {
      // A later branch restriction must not legitimize an earlier overflowing
      // definition. Only definitions qualified at their original cut are bound.
      if (unproved_definitions_.count(expression.get())) return std::nullopt;
      auto b = analyzer_->const_int_bound(expression);
      const auto correlated = analyzer_->const_int_bound(analyzer_->Simplify(expression));
      b.min_value = std::max(b.min_value, correlated.min_value);
      b.max_value = std::min(b.max_value, correlated.max_value);
      if (b.min_value < 0 || b.min_value > b.max_value || b.max_value >= arith::ConstIntBound::kPosInf ||
          b.max_value > *maximum)
        return std::nullopt;
      return Bound{b.min_value, b.max_value};
    }
    ExprPtr left, right;
    bool multiply = false, modulo = false, subtract = false;
    if (auto add = As<Add>(expression)) {
      left = add->left_;
      right = add->right_;
    } else if (auto sub = As<Sub>(expression)) {
      left = sub->left_;
      right = sub->right_;
      subtract = true;
    } else if (auto mul = As<Mul>(expression)) {
      left = mul->left_;
      right = mul->right_;
      multiply = true;
    } else if (auto mod = As<FloorMod>(expression)) {
      left = mod->left_;
      right = mod->right_;
      modulo = true;
    } else if (auto div = As<FloorDiv>(expression)) {
      left = div->left_;
      right = div->right_;
    } else if (auto min = As<Min>(expression)) {
      left = min->left_;
      right = min->right_;
    } else {
      return std::nullopt;
    }
    if (!SameCoordinateType(expression, left) || !SameCoordinateType(expression, right)) return std::nullopt;
    auto a = Range(left), b = Range(right);
    if (!a || !b) return std::nullopt;
    // Simplify only after both original operands have passed machine-width
    // qualification. Correlation can prove a nonnegative subtraction even when
    // independent operand intervals overlap; no intermediate overflow is erased.
    if (subtract) {
      // In an active [0, ceildiv(n, tile)) loop, tile*i < n. The
      // original numerator, IV update and product have all passed range
      // checks in their machine types before this relation is used.
      if (GetScalarDtype(expression) == DataType::INDEX && IsRemainingTiledExtent(left, right))
        return Bound{1, a->second};
      const auto result = analyzer_->const_int_bound(analyzer_->Simplify(expression));
      if (result.min_value < 0 || result.min_value > result.max_value ||
          result.max_value >= arith::ConstIntBound::kPosInf || result.max_value > *maximum)
        return std::nullopt;
      return Bound{result.min_value, result.max_value};
    }
    // Only a proved positive constant divisor is admitted. Both operands were
    // checked in their original machine types before reducing the range.
    if (modulo) {
      if (b->first <= 0 || b->first != b->second) return std::nullopt;
      return a->second < b->first ? a : std::optional<Bound>{{0, b->first - 1}};
    }
    if (As<FloorDiv>(expression)) {
      if (b->first <= 0 || b->first != b->second) return std::nullopt;
      return Bound{a->first / b->first, a->second / b->first};
    }
    if (As<Min>(expression)) {
      return Bound{std::min(a->first, b->first), std::min(a->second, b->second)};
    }
    const __int128 lo = multiply ? __int128(a->first) * b->first : __int128(a->first) + b->first;
    const __int128 hi = multiply ? __int128(a->second) * b->second : __int128(a->second) + b->second;
    if (hi >= arith::ConstIntBound::kPosInf || hi > *maximum) return std::nullopt;
    return Bound{static_cast<int64_t>(lo), static_cast<int64_t>(hi)};
  }
  bool QualifiedCondition(const ExprPtr& expression) {
    if (!(As<Lt>(expression) || As<Le>(expression) || As<Gt>(expression) || As<Ge>(expression) ||
          As<Eq>(expression) || As<Ne>(expression)))
      return false;
    const auto comparison = As<BinaryExpr>(expression);
    return SameCoordinateType(comparison->left_, comparison->right_) && Range(comparison->left_) &&
           Range(comparison->right_);
  }
  void Unbind(const VarPtr& var) {
    analyzer_->const_int_bound.Unbind(var);
    analyzer_->rewrite_simplify.Update(var, nullptr);
  }
  void ForgetDefinitions(size_t first) {
    while (definitions_.size() > first) {
      Unbind(definitions_.back());
      unproved_definitions_.erase(definitions_.back().get());
      scalar_definitions_.erase(definitions_.back().get());
      definitions_.pop_back();
    }
  }
  void VisitBranch(const StmtPtr& body, const ExprPtr& condition) {
    const auto first = definitions_.size();
    Invalidate();
    if (condition) {
      auto scope = analyzer_->GetConstraintContext(condition);
      VisitStmt(body);
      ForgetDefinitions(first);
    } else {
      VisitStmt(body);
      ForgetDefinitions(first);
    }
    // ConstraintContext restores every analyzer, including substitutions.
    // Branch-local SSA definitions never become evidence at sibling/later cuts.
    Invalidate();
  }
  void RecordCall(const CallPtr& call) {
    if (auto global = As<GlobalVar>(call->op_)) {
      auto found = functions_.find(global->name_);
      if (found != functions_.end()) {
        std::optional<Bound> launch, count_range;
        ExprPtr launch_count;
        if (found->second->func_type_ == FunctionType::Spmd) {
          auto count = call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
          if (count) {
            auto bound = Range(count);
            if (bound && bound->second > 0) {
              // The launch contributes worker invocations only on its
              // nonempty path. This remains true before the launch pass
              // materializes the corresponding host guard.
              bound->first = std::max<int64_t>(1, bound->first);
              launch = Bound{0, bound->second - 1};
              count_range = bound;
              launch_count = count;
            }
          } else if (auto constant = As<ConstInt>(found->second->GetAttr<ExprPtr>(kAttrCoreNum, nullptr));
                     constant && constant->value_ > 0) {
            launch = Bound{0, constant->value_ - 1};
            count_range = Bound{constant->value_, constant->value_};
          }
        } else if (caller_->func_type_ == FunctionType::Spmd &&
                   (found->second->func_type_ == FunctionType::AIC ||
                    found->second->func_type_ == FunctionType::AIV)) {
          // Outlined workers receive the wrapper's logical task block index.
          // Dual-AIV dispatch shares that index between lanes; it does not
          // establish a subblock range or disjoint storage between the lanes.
          launch = block_index_;
          count_range = block_count_;
        }
        auto [launch_it, first_launch] = launches_.emplace(global->name_, launch);
        if (!first_launch) {
          if (!launch_it->second || !launch)
            launch_it->second = std::nullopt;
          else
            launch_it->second->second = std::max(launch_it->second->second, launch->second);
        }
        auto [count_it, first_count] = counts_.emplace(global->name_, count_range);
        if (!first_count) {
          if (!count_it->second || !count_range) {
            count_it->second = std::nullopt;
          } else {
            count_it->second->first = std::min(count_it->second->first, count_range->first);
            count_it->second->second = std::max(count_it->second->second, count_range->second);
          }
        }
        const auto& params = found->second->params_;
        auto& callee = evidence_[global->name_];
        for (size_t i = 0; i < params.size(); ++i) {
          if (!As<ScalarType>(params[i]->GetType())) continue;
          auto argument = transform_utils::CallerSuppliedArg(call, found->second, i);
          auto bound = argument ? Range(argument) : std::nullopt;
          if (launch_count && launch_count == argument) {
            bound = count_range;
          }
          const auto maximum = SyncCoordinateMaximum(params[i]);
          if (bound && (!maximum || bound->second > *maximum)) bound = std::nullopt;
          auto [it, inserted] = callee.emplace(i, bound);
          if (!inserted) {
            if (!it->second || !bound) {
              it->second = std::nullopt;
            } else {
              it->second->first = std::min(it->second->first, bound->first);
              it->second->second = std::max(it->second->second, bound->second);
            }
          }
        }
      }
    }
  }

 public:
  InvocationBounds(const std::map<std::string, FunctionPtr>& functions,
                   std::map<std::string, std::map<size_t, std::optional<Bound>>>& evidence,
                   std::map<std::string, std::optional<Bound>>& launches,
                   std::map<std::string, std::optional<Bound>>& counts, FunctionPtr caller,
                   SyncParameterWork& work)
      : functions_(functions),
        evidence_(evidence),
        launches_(launches),
        counts_(counts),
        caller_(std::move(caller)),
        work_(work) {
    BindHostShapeSymbols();
    // Every predecessor has been processed before this environment is built.
    // Missing/unknown formal bounds are not inferred from a sample invocation.
    if (auto incoming = evidence.find(caller_->name_); incoming != evidence.end()) {
      for (const auto& [index, bound] : incoming->second) {
        if (bound)
          analyzer_->const_int_bound.Update(caller_->params_.at(index), {bound->first, bound->second});
      }
    }
    if (auto count = counts.find(caller_->name_); count != counts.end()) {
      block_count_ = count->second;
    }
    if (auto launch = launches.find(caller_->name_); launch != launches.end()) {
      block_index_ = launch->second;
    }
  }
  void VisitExpr(const ExprPtr& expression) override {
    if (expression && visited_.insert(expression.get()).second) {
      ++work_.expression_visits;
      IRVisitor::VisitExpr(expression);
    }
  }
  void VisitStmt(const StmtPtr& statement) override {
    const auto first = definitions_.size();
    IRVisitor::VisitStmt(statement);
    if (As<ScopeStmt>(statement)) {
      ForgetDefinitions(first);
      Invalidate();
    }
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const CallPtr& call) override {
    RecordCall(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    RecordCall(SubmitToCallView(call));
    IRVisitor::VisitExpr_(call);
  }
  void VisitStmt_(const AssignStmtPtr& statement) override {
    VisitExpr(statement->value_);
    auto b = Range(statement->value_);
    definitions_.push_back(statement->var_);
    if (b && SameCoordinateType(statement->var_, statement->value_)) {
      auto definition = Definition(statement->value_);
      if (const auto cast = As<Cast>(definition); cast && SameCoordinateType(cast, cast->operand_))
        definition = Definition(cast->operand_);
      scalar_definitions_[statement->var_.get()] = definition;
      // This producer consumes only substitutions and constant bounds. Do not
      // populate transitive-comparison inventories which it never queries (and
      // whose per-definition removal scans the entire retained inventory).
      // Keep the runtime observation as an SSA variable. Inlining the opaque
      // query into a branch condition would discard the variable's scoped bound.
      if (!IsLaunchCapacity(statement->value_)) {
        analyzer_->rewrite_simplify.Update(statement->var_, analyzer_->Simplify(statement->value_));
      }
      // Preserve specialized bounds for calls such as get_block_idx which the
      // generic arithmetic analyzer intentionally treats as opaque.
      analyzer_->const_int_bound.Update(statement->var_, {b->first, b->second});
    } else {
      Unbind(statement->var_);
      unproved_definitions_.insert(statement->var_.get());
    }
    Invalidate();
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitExpr(branch->condition_);
    const bool qualified = QualifiedCondition(branch->condition_);
    VisitBranch(branch->then_body_, qualified ? branch->condition_ : nullptr);
    if (branch->else_body_) {
      ExprPtr complement =
          qualified ? std::make_shared<Not>(branch->condition_, DataType::BOOL, branch->condition_->span_)
                    : nullptr;
      VisitBranch(*branch->else_body_, complement);
    }
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    VisitExpr(loop->condition_);
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    VisitBranch(loop->body_, nullptr);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    // Calls in bounds and initial values execute outside the iteration domain.
    // Missing one such site could incorrectly retain a narrower callee range.
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    auto start = Range(loop->start_), stop = Range(loop->stop_), step = Range(loop->step_);
    const auto maximum = SyncCoordinateMaximum(loop->loop_var_);
    if (maximum && start && stop && step && step->first > 0 && stop->second > start->first &&
        __int128(stop->second) - 1 + step->second < arith::ConstIntBound::kPosInf &&
        __int128(stop->second) - 1 + step->second <= *maximum) {
      int64_t last = stop->second - 1;
      if (start->first == start->second && stop->first == stop->second && step->first == step->second)
        last = start->first + (stop->first - start->first - 1) / step->first * step->first;
      analyzer_->const_int_bound.Update(loop->loop_var_, {start->first, last});
      if (const auto tiled = QualifyTiledLoop(loop)) {
        tiled_loops_.emplace(loop->loop_var_.get(), *tiled);
      }
    } else {
      Unbind(loop->loop_var_);
      unproved_definitions_.insert(loop->loop_var_.get());
    }
    // Inclusive hull is deliberately conservative for nonunit strides. Calls
    // in empty/unknown loops do not manufacture vacuous range certificates.
    Invalidate();
    const auto first = definitions_.size();
    VisitStmt(loop->body_);
    ForgetDefinitions(first);
    Unbind(loop->loop_var_);
    tiled_loops_.erase(loop->loop_var_.get());
    unproved_definitions_.erase(loop->loop_var_.get());
    Invalidate();
  }
};
// Index edges once, including Submit's original call ABI. Kahn traversal joins
// all predecessors before inspecting a function; no whole-program fixpoint or
// visit expansion is needed. Cycles and their descendants are left unproved.
class InvocationCallGraph : public IRVisitor {
  std::set<const Expr*> visited_;
  void Record(const OpPtr& op) {
    if (auto global = As<GlobalVar>(op)) callees.insert(global->name_);
  }

 public:
  std::set<std::string> callees;
  size_t expression_visits = 0;
  void VisitExpr(const ExprPtr& expression) override {
    if (expression && visited_.insert(expression.get()).second) {
      ++expression_visits;
      IRVisitor::VisitExpr(expression);
    }
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const CallPtr& call) override {
    Record(call->op_);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    Record(call->op_);
    IRVisitor::VisitExpr_(call);
  }
};
}  // namespace
std::optional<int64_t> SyncCoordinateMaximum(const ExprPtr& e) {
  auto type = As<ScalarType>(e->GetType());
  if (!type || (type->dtype_ != DataType::INDEX && !type->dtype_.IsInt())) return std::nullopt;
  const auto bits = type->dtype_ == DataType::INDEX ? 64 : type->dtype_.GetBit();
  if (!bits || bits > 64) return std::nullopt;
  if (bits == 64) return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>((uint64_t{1} << (bits - type->dtype_.IsSignedInt())) - 1);
}
SyncInvocationEvidence CollectSyncInvocationEvidence(const ProgramPtr& program, SyncParameterWork* work) {
  SyncParameterWork local_work;
  if (!work) work = &local_work;
  std::map<std::string, FunctionPtr> functions;
  for (const auto& [global, function] : program->functions_) functions.emplace(global->name_, function);
  std::map<std::string, std::map<size_t, std::optional<std::pair<int64_t, int64_t>>>> evidence;
  std::map<std::string, std::optional<std::pair<int64_t, int64_t>>> launches, counts;
  std::map<std::string, std::set<std::string>> edges;
  std::map<std::string, size_t> incoming;
  for (const auto& [name, function] : functions) incoming.emplace(name, 0);
  for (const auto& [name, function] : functions) {
    InvocationCallGraph graph;
    graph.VisitFunction(function);
    work->call_graph_expression_visits += graph.expression_visits;
    ++work->call_graph_nodes;
    for (const auto& callee : graph.callees)
      if (functions.count(callee)) {
        edges[name].insert(callee);
        ++incoming.at(callee);
        ++work->call_graph_edges;
      }
  }
  if (work == &local_work && work->call_graph_edges == 0) return {};
  std::set<std::string> ready, processed;
  for (const auto& [name, count] : incoming)
    if (!count) ready.insert(name);
  while (!ready.empty()) {
    const auto name = *ready.begin();
    ready.erase(ready.begin());
    InvocationBounds collector(functions, evidence, launches, counts, functions.at(name), *work);
    collector.VisitFunction(functions.at(name));
    processed.insert(name);
    ++work->processed_functions;
    for (const auto& callee : edges[name])
      if (!--incoming.at(callee)) ready.insert(callee);
  }
  SyncInvocationEvidence result;
  result.callees = std::move(edges);
  for (const auto& [name, params] : evidence)
    for (const auto& [index, bound] : params)
      if (bound && processed.count(name)) result.parameters[name].emplace(index, *bound);
  for (const auto& [name, bound] : launches)
    if (bound && processed.count(name)) result.block_indices.emplace(name, *bound);
  for (const auto& [name, bound] : counts)
    if (bound && processed.count(name)) result.block_counts.emplace(name, *bound);
  return result;
}
// Qualify only already recovered exact geometry. This is one pass over static
// access descriptions: a noncontiguous rectangle is not expanded into rows and
// no enclosing interval is substituted for its footprint. Coordinates are
// relative to the canonical storage identity (including the Vec byte domain).
static void QualifyPhysicalIntervals(CompactSyncInput& input) {
  for (auto& segment : input.segments) {
    for (auto& site : segment.sites) {
      for (auto& access : site.accesses) {
        ++input.interval_access_visits;
        const auto& storage = input.storage.at(access.storage);
        access.physical_interval = PhysicalInterval(storage, access);
        if (access.physical_interval) ++input.qualified_intervals;
      }
    }
  }
}

static CompactSyncInput ImportQualifiedSync(
    const FunctionPtr& function, const std::string& target, const SyncParameterBounds& parameter_bounds,
    const std::optional<std::pair<int64_t, int64_t>>& block_bounds, bool recover_accesses,
    SyncExternalPolicy external_policy,
    std::shared_ptr<const SyncInvocationCoordinate> coordinate = nullptr) {
  CHECK(function && (function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV))
      << "Synchronization prototype requires one AIC or AIV leaf";
  if (recover_accesses) {
    for (const auto& parameter : function->params_)
      if (const auto tensor = As<TensorType>(parameter->GetType()); tensor && tensor->shape_.size() == 1)
        CHECK(false) << "Exact rank-one GM access and selector exports are unavailable; use an explicit "
                        "conservative request";
  }
  if (function->func_type_ == FunctionType::AIV) {
    // Reuse the production view/recipe alias verifier. Synchronization cannot
    // repair an illegal partial in-place instruction or a malformed descriptor.
    CHECK(function->ir_stage_ == FunctionIRStage::Buffer) << "Vec synchronization requires Buffer IR";
    auto program =
        std::make_shared<Program>(std::vector<FunctionPtr>{function}, function->name_, function->span_);
    std::vector<Diagnostic> diagnostics;
    CreateBufferIRPropertyVerifier()->Verify(program, diagnostics);
    for (const auto& diagnostic : diagnostics)
      CHECK(diagnostic.severity != DiagnosticSeverity::Error) << diagnostic.message;
  }
  Importer importer(recover_accesses, external_policy);
  if (importer.readonly_external) importer.result.unavailable_storage_spaces.insert(MemorySpace::DDR);
  if (importer.disjoint_readonly) {
    importer.RegisterExternalFormals(function);
  }
  importer.result.owner = function;
  importer.result.hardware_target = target;
  importer.result.parameter_bounds = parameter_bounds;
  if (block_bounds)
    CHECK(block_bounds->first >= 0 && block_bounds->second >= block_bounds->first &&
          block_bounds->second < std::numeric_limits<int64_t>::max())
        << "Invalid logical launch range";
  if (coordinate) {
    CHECK(coordinate->Owner() == function && coordinate->Parameters() == parameter_bounds && block_bounds &&
          block_bounds->first >= 0 && block_bounds->second < coordinate->Count())
        << "Foreign effective invocation coordinate or case contract";
  }
  importer.result.invocation_coordinate = std::move(coordinate);
  importer.result.block_bounds = block_bounds;
  for (const auto& [index, range] : parameter_bounds) {
    CHECK(index < function->params_.size() && As<ScalarType>(function->params_[index]->GetType()))
        << "Invocation range must identify an original scalar parameter";
    const auto maximum = SyncCoordinateMaximum(function->params_[index]);
    CHECK(maximum && range.first >= 0 && range.second >= range.first &&
          range.second < std::numeric_limits<int64_t>::max() && range.second <= *maximum)
        << "Invalid invocation coordinate range";
  }
  if (recover_accesses) {
    importer.prover = std::make_unique<SyncControlRangeProver>(function, parameter_bounds, block_bounds);
  }
  importer.resolver = std::make_unique<SyncFootprintResolver>(function, importer.prover.get(),
                                                              /*run_scalar_layout_check=*/false, target);
  // Discover scalar identities before tensor shapes, regardless of ABI order.
  for (size_t index = 0; index < function->params_.size(); ++index) {
    const auto& param = function->params_[index];
    if (As<ScalarType>(param->GetType())) {
      if (recover_accesses && !importer.DeferredCoordinates()) {
        CHECK(SyncCoordinateMaximum(param)) << "Unavailable invocation coordinate type";
        CHECK(parameter_bounds.count(index)) << "Invocation coordinate has no established caller range";
      }
      importer.parameters.emplace(param.get(), index);
    }
  }
  for (const auto& param : function->params_) {
    if (As<ScalarType>(param->GetType())) continue;
    if (importer.ProjectsExternal(param)) {
      const auto type = As<TensorType>(param->GetType());
      CHECK(type && (type->shape_.size() == 1 || type->shape_.size() == 2) &&
            type->dtype_.GetBit() % 8 == 0 && !type->tensor_view_)
          << "Projected external roots require the existing unviewed rank-one/two ND descriptor contract";
      for (const auto& dim : type->shape_) {
        if (As<ConstInt>(dim)) {
          CHECK(Constant(dim) > 0) << "Projected external extent must remain positive";
        } else {
          RequireIndex(dim);
          if (importer.disjoint_readonly) {
            CHECK(AsVarLike(dim)) << "Projected input extent must use a ready descriptor dimension";
          } else {
            importer.ReadyScalar(dim, SyncSegment{});
          }
        }
      }
      importer.objects[param.get()] = importer.result.storage.size();
      SyncStorage unavailable{param->name_hint_, MemorySpace::DDR, 0, 0, 0, 0, nullptr};
      unavailable.boundary_available = false;
      importer.result.storage.push_back(std::move(unavailable));
      continue;
    }
    if (!recover_accesses) continue;  // GM geometry is not token-order evidence.
    auto t = As<TensorType>(param->GetType());
    CHECK(t && t->shape_.size() == 2 && t->dtype_.GetBit() % 8 == 0 && !t->tensor_view_)
        << "Synchronization prototype requires unviewed ND tensor arguments";
    std::optional<size_t> row_parameter;
    int64_t rows;
    if (As<ConstInt>(t->shape_[0])) {
      rows = Constant(t->shape_[0]);
    } else {
      RequireIndex(t->shape_[0]);
      auto bound = importer.parameters.find(t->shape_[0].get());
      CHECK(bound != importer.parameters.end())
          << "Dynamic allocation extent has no established caller range";
      const auto extent = parameter_bounds.find(bound->second);
      CHECK(extent != parameter_bounds.end()) << "Dynamic allocation extent has no established caller range";
      row_parameter = bound->second;
      rows = extent->second.second;
    }
    auto cols = Constant(t->shape_[1]);
    importer.objects[param.get()] = importer.result.storage.size();
    importer.result.storage.push_back({param->name_hint_, MemorySpace::DDR, 0, rows, cols,
                                       Checked(__int128(rows) * cols * t->dtype_.GetByte()), nullptr});
    importer.result.storage.back().row_extent_parameter = row_parameter;
  }
  std::map<const ForStmt*, bool> nested;
  const auto shape = NestedShape(function->body_, nested);
  if (shape.second) {
    std::vector<SyncLoopFrame> frames;
    importer.result.region_tree = ImportStructuredBody(importer, function->body_, frames, nested);
    CanonicalizeMatrixSyncStorage(importer.result);
    if (recover_accesses) {
      std::set<const ForStmt*> phase_scopes;
      for (const auto& segment : importer.result.segments)
        for (const auto& site : segment.sites)
          for (const auto& access : site.accesses)
            if (access.slot_coordinate_frame && access.slot_coordinate_frame != segment.original_loop &&
                importer.result.storage[access.storage].slot_count /
                        std::gcd(importer.result.storage[access.storage].slot_count, access.slot_step) >
                    1)
              phase_scopes.insert(access.slot_coordinate_frame.get());
      CHECK(QualifyStartupScopes(*importer.result.region_tree, importer.result, phase_scopes).pending.empty())
          << "Initialization scope is outside the retained source tree";
      QualifyPhysicalIntervals(importer.result);
    }
    return std::move(importer.result);
  }
  for (const auto& stmt : Statements(function->body_)) {
    if (auto a = As<AssignStmt>(stmt)) {
      importer.Definition(a, SyncSegment{}, true);
      continue;
    }
    if (As<ReturnStmt>(stmt)) continue;
    if (auto loop = As<ForStmt>(stmt)) {
      SyncSegment s;
      RequireIndex(loop->loop_var_);
      RequireIndex(loop->start_);
      RequireIndex(loop->stop_);
      RequireIndex(loop->step_);
      CHECK(loop->kind_ == ForKind::Sequential && loop->iter_args_.empty() && loop->return_vars_.empty())
          << "Only sequential coordinate-only loops are qualified";
      const auto start = Constant(loop->start_), step = Constant(loop->step_), stop = Constant(loop->stop_);
      CHECK(step > 0) << "Nonpositive loop step is not qualified";
      const auto count = stop <= start ? 0 : 1 + (stop - start - 1) / step;
      Checked(__int128(start) + __int128(count) * step);
      s.original_loop = loop;
      s.start = start;
      s.step = step;
      s.count = count;
      importer.Body(loop->body_, s);
      importer.result.segments.push_back(std::move(s));
    } else {
      // Keep maximal adjacent finite statements in one numerical region.
      if (importer.result.segments.empty() || importer.result.segments.back().original_loop)
        importer.result.segments.emplace_back();
      importer.Body(stmt, importer.result.segments.back());
    }
  }
  // Each leaf has one permutation of every slot family. Different offsets
  // are permitted, but different strides do not justify a period-one graph.
  for (const auto& segment : importer.result.segments) {
    std::map<size_t, int64_t> strides;
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        if (importer.result.storage[access.storage].slot_count == 1) continue;
        auto [found, inserted] = strides.emplace(access.storage, access.slot_step);
        CHECK(inserted || found->second == access.slot_step)
            << "Rotating family requires one common stride within a region";
      }
  }
  CanonicalizeMatrixSyncStorage(importer.result);
  if (recover_accesses) QualifyPhysicalIntervals(importer.result);
  return std::move(importer.result);
}
CompactSyncInput ImportPlacedSync(const FunctionPtr& function, bool disjoint_gm, const std::string& target,
                                  const SyncParameterBounds& parameters,
                                  const std::optional<std::pair<int64_t, int64_t>>& blocks,
                                  std::shared_ptr<const SyncInvocationCoordinate> coordinate,
                                  SyncExternalPolicy external_policy) {
  CHECK(disjoint_gm)
      << "Disjoint GM argument contract must be explicitly supplied; may-alias import unavailable";
  INTERNAL_CHECK(external_policy == SyncExternalPolicy::Geometry ||
                 external_policy == SyncExternalPolicy::DisjointReadOnlyProjection);
  return ImportQualifiedSync(function, target, parameters, blocks, true, external_policy,
                             std::move(coordinate));
}
CompactSyncInput ImportSyncPayloads(const FunctionPtr& function, const std::string& target,
                                    const SyncParameterBounds& parameters,
                                    const std::optional<std::pair<int64_t, int64_t>>& blocks,
                                    SyncExternalPolicy external_policy,
                                    std::shared_ptr<const SyncInvocationCoordinate> coordinate) {
  return ImportQualifiedSync(function, target, parameters, blocks, false, external_policy,
                             std::move(coordinate));
}
CompactSyncInput ImportReadonlyExternalSync(const FunctionPtr& function, const std::string& target,
                                            const SyncParameterBounds& parameters,
                                            const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  return ImportQualifiedSync(function, target, parameters, blocks, true,
                             SyncExternalPolicy::ReadOnlyProjection);
}
}  // namespace pypto::ir::sync
