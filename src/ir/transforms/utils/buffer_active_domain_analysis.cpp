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

// Three bounded stages: index/forward value facts, backward observable uses,
// and one immutable rewrite. A domain is full, one common prefix, or unavailable.
// Backward requirements only grow unused -> prefix -> full (height three), so
// their indexed worklist visits each edge a constant number of times. There is
// no trip-count unfolding, global per-candidate scan or general domain solver.

#include "pypto/ir/transforms/utils/buffer_active_domain_analysis.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_acc_trim_analysis.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto::ir {
namespace {

enum class DomainKind { Unavailable, Full, Prefix, Unused };
struct Domain {
  DomainKind kind = DomainKind::Unavailable;
  ExprPtr extent;
  int axis = 0;
};

bool Same(const Domain& a, const Domain& b) {
  return a.kind == b.kind &&
         (a.kind != DomainKind::Prefix || (a.axis == b.axis && structural_equal(a.extent, b.extent)));
}

// Unknown intersection stays unknown; it is never an empty access/value answer.
Domain Meet(const Domain& a, const Domain& b) {
  if (a.kind == DomainKind::Unavailable || b.kind == DomainKind::Unavailable) return {};
  if (a.kind == DomainKind::Full) return b;
  if (b.kind == DomainKind::Full || Same(a, b)) return a;
  return {};
}

Domain Declared(const TypePtr& type) {
  const auto tile = As<TileType>(type);
  if (!tile || tile->GetMemorySpace() != MemorySpace::Vec ||
      (tile->shape_.size() != 1 && tile->shape_.size() != 2))
    return {};
  const auto& view = tile_view_semantics::GetEffectiveTileView(*tile);
  const auto& valid = view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  if (valid.size() != tile->shape_.size()) return {};
  Domain result{DomainKind::Full, nullptr, 0};
  for (size_t i = 0; i < tile->shape_.size(); ++i) {
    const auto capacity = As<ConstInt>(tile->shape_[i]);
    if (!capacity || capacity->value_ <= 0) return {};
    if (structural_equal(valid[i], tile->shape_[i])) continue;
    // A named bound gives constant-time lexical/invariance checks. More general
    // expressions retain the original IR until they have a shared proof owner.
    if (result.kind == DomainKind::Prefix || (!AsVarLike(valid[i]) && !As<ConstInt>(valid[i]))) return {};
    result = {DomainKind::Prefix, valid[i], tile->shape_.size() == 1 ? 1 : static_cast<int>(i)};
  }
  return result;
}

bool Contains(const Domain& defined, const Domain& required) {
  return required.kind == DomainKind::Unused ||
         (defined.kind != DomainKind::Unavailable &&
          (defined.kind == DomainKind::Full || Same(defined, required)));
}

bool IsElementwise(const CallPtr& call) {
  const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call);
  return recipe && recipe->types != backend::BufferElementwiseTypePolicy::DenseConcatVec;
}
bool IsCast(const CallPtr& call) {
  const auto recipe = backend::FindLogicalBufferConversionRecipe(call->op_->name_);
  return recipe && recipe->kind == backend::BufferConversionKind::Cast;
}
bool IsVectorReshape(const CallPtr& call) {
  return IsOp(call, "tile.reshape") && backend::SupportsLogicalContiguousVectorReshape(call);
}
bool IsRowAxisOp(const CallPtr& call) {
  const auto* axis = backend::FindLogicalBufferAxisRecipe(call->op_->name_);
  return axis && (axis->axis == 1 || !axis->reduction);
}
bool KnownData(const CallPtr& call) {
  return IsElementwise(call) || IsCast(call) || IsVectorReshape(call) || IsOp(call, "tile.full") ||
         IsOp(call, "tile.create") || IsOp(call, "tile.load") || IsOp(call, "tile.store") ||
         IsOp(call, "tile.set_validshape") || IsOp(call, "tile.ci") || IsOp(call, "tile.concat") ||
         IsRowAxisOp(call);
}

Domain AxisMap(Domain domain, const TypePtr& target) {
  if (domain.kind != DomainKind::Prefix) return domain;
  const auto tile = As<TileType>(target);
  if (!tile) return {};
  domain.axis = As<ConstInt>(tile->shape_[0])->value_ == 1 ? 1 : 0;
  return domain;
}

// Supported identities preserve the original logical axis in both directions.
// Only an actual orientation exchange maps the prefix to the target's axis.
Domain ReshapeDomain(Domain domain, const CallPtr& call, const TypePtr& target) {
  const auto source = As<TileType>(call->args_[0]->GetType());
  const auto destination = As<TileType>(call->GetType());
  if (tile_view_semantics::GetEffectiveTileView(*source).blayout ==
      tile_view_semantics::GetEffectiveTileView(*destination).blayout)
    return domain;
  return AxisMap(domain, target);
}

struct Node {
  VarPtr value;
  CallPtr call;
  std::vector<ExprPtr> dependencies;
  Domain defined;
  Domain required{DomainKind::Unused, nullptr, 0};
};

// The traversal preserves original lexical scope. A bound from a sibling branch
// is not made available by its earlier traversal clock.
class DomainIndex : public IRVisitor {
 public:
  DomainIndex(const FunctionPtr& function, std::unordered_map<const Var*, CallPtr> definitions,
              BufferActiveDomainWork* work)
      : definitions_(std::move(definitions)), work_(work) {
    scopes_.push_back({0, 0, 0});
    for (const auto& param : function->params_) Define(param);
    VisitStmt(function->body_);
    scopes_[0].end = ++clock_;
  }
  std::unordered_map<const Var*, Node> nodes;
  std::unordered_map<const IterArg*, Domain> carries;
  std::unordered_map<const Var*, std::vector<const Var*>> aliases;
  std::unordered_set<const Var*> unsafe_metadata;
  std::unordered_map<const Var*, size_t> storage_writers;
  std::unordered_set<const Var*> scratch_roots;
  std::set<std::string> names;
  std::vector<std::pair<ExprPtr, Domain>> observations;
  bool invalid_carry = false;

  // Domains and lexical definitions are recovered explicitly at statements.
  // Expression uses belong to MetadataUseIndex's bounded DAG walk; following
  // them here would revisit shared scalar yields once per expression path.
  void VisitExpr(const ExprPtr&) override {}
  void VisitStmt(const StmtPtr& statement) override {
    if (work_) ++work_->domain_statements;
    IRVisitor::VisitStmt(statement);
  }

  Domain Get(const ExprPtr& expression) const {
    const auto value = AsVarLike(expression);
    const auto found = value ? nodes.find(value.get()) : nodes.end();
    return found == nodes.end() ? Domain{} : found->second.defined;
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}

  void VisitStmt_(const AssignStmtPtr& statement) override {
    Define(statement->var_);
    const auto call = As<Call>(statement->value_);
    if (!As<TileType>(statement->var_->GetType())) {
      if (call) ObserveUnknown(call);
      return;
    }
    Node node{statement->var_, call, {}, {}, {DomainKind::Unused, nullptr, 0}};
    if (!call) {
      node.dependencies.push_back(statement->value_);
      node.defined = Get(statement->value_);
      Alias(statement->var_, statement->value_);
    } else {
      node.dependencies = call->args_;
      node.defined = Forward(call);
      if (!KnownData(call)) ObserveUnknown(call);
      if (IsVectorReshape(call) || IsOp(call, "tile.set_validshape")) {
        Alias(statement->var_, call->args_[0]);
      } else if (!IsOp(call, "tile.create")) {
        if (const auto memory = GetTypeMemRef(statement->var_->GetType())) {
          ++storage_writers[(*memory)->base_.get()];
        }
      }
      const auto axis = backend::FindLogicalBufferAxisRecipe(call->op_->name_);
      if (axis && axis->HasScratch() && call->args_.size() > 1) {
        if (const auto memory = GetTypeMemRef(call->args_[1]->GetType())) {
          scratch_roots.insert((*memory)->base_.get());
        }
      }
    }
    nodes.emplace(statement->var_.get(), std::move(node));
  }

  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (IsOp(call, "tile.store")) {
      observations.emplace_back(call->args_[0], Declared(call->args_[0]->GetType()));
    } else if (call) {
      ObserveUnknown(call);
    }
  }

  void VisitStmt_(const ReturnStmtPtr& statement) override {
    for (const auto& value : statement->value_) {
      if (As<TileType>(value->GetType())) {
        observations.emplace_back(value, Domain{DomainKind::Full, nullptr, 0});
        if (auto var = AsVarLike(value)) unsafe_metadata.insert(var.get());
      }
    }
  }

  void VisitStmt_(const IfStmtPtr& branch) override {
    Enter([&] { VisitStmt(branch->then_body_); });
    if (branch->else_body_) Enter([&] { VisitStmt(*branch->else_body_); });
    for (const auto& result : branch->return_vars_) {
      Define(result);
      if (As<TileType>(result->GetType())) nodes.emplace(result.get(), Node{result, nullptr, {}, {}, {}});
    }
  }

  void VisitStmt_(const ForStmtPtr& loop) override {
    // One round, with only direct invariant add recurrences admitted. An inner
    // carry that creates a new outer recurrence does not trigger another scan.
    const auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    const auto parent = current_scope_;
    const auto entry = ++clock_;
    std::vector<ExprPtr> backedges;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& carry = loop->iter_args_[i];
      const auto backedge = yield && i < yield->value_.size() ? yield->value_[i] : nullptr;
      backedges.push_back(backedge);
      const auto initial = Get(carry->initValue_);
      auto domain = initial;
      if (initial.kind == DomainKind::Prefix && Visible(initial.extent, parent, entry))
        carries.emplace(carry.get(), initial);
      if (backedge && As<TileType>(carry->GetType())) {
        const auto back_var = AsVarLike(backedge);
        const auto found = back_var ? definitions_.find(back_var.get()) : definitions_.end();
        // Definitions inside the loop have not been visited yet. A separate
        // call index, built once before this walk, supplies the direct yield.
        const auto call = found == definitions_.end() ? nullptr : found->second;
        const auto type = As<TileType>(carry->GetType());
        if (IsOp(call, "tile.add") && call->args_.size() == 2 && initial.kind == DomainKind::Full &&
            type->dtype_ == DataType::FP32) {
          const int other = call->args_[0] == carry ? 1 : call->args_[1] == carry ? 0 : -1;
          if (other >= 0) {
            const auto proposed = Declared(call->args_[other]->GetType());
            if (proposed.kind == DomainKind::Prefix && Visible(proposed.extent, parent, entry)) {
              domain = proposed;
              carries.emplace(carry.get(), domain);
            }
          }
        }
      }
      if (As<TileType>(carry->GetType()))
        nodes.emplace(carry.get(), Node{carry, nullptr, {carry->initValue_, backedge}, domain, {}});
      Alias(carry, carry->initValue_);
      if (backedge) Alias(carry, backedge);
    }
    Enter([&] {
      Define(loop->loop_var_);
      for (const auto& carry : loop->iter_args_) Define(carry);
      VisitStmt(loop->body_);
    });
    for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
      const auto& result = loop->return_vars_[i];
      Define(result);
      if (!As<TileType>(result->GetType())) continue;
      const auto& carry = loop->iter_args_[i];
      const auto backedge = backedges[i];
      const auto initial = Get(carry->initValue_);
      const auto actual = backedge ? Meet(initial, Get(backedge)) : Domain{};
      if (!Same(actual, Get(carry))) invalid_carry = true;
      nodes.emplace(result.get(), Node{result, nullptr, {carry->initValue_, backedge}, actual, {}});
      Alias(result, carry);
    }
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    // No loop-invariant prefix proof for arbitrary while carries in this slice.
    for (const auto& carry : loop->iter_args_)
      if (As<TileType>(carry->GetType())) invalid_carry = true;
    Enter([&] { VisitStmt(loop->body_); });
    for (const auto& result : loop->return_vars_) {
      Define(result);
      if (As<TileType>(result->GetType())) nodes.emplace(result.get(), Node{result, nullptr, {}, {}, {}});
    }
  }

 private:
  struct Scope {
    size_t begin, end, parent;
  };
  struct Definition {
    size_t clock, scope;
  };
  std::vector<Scope> scopes_;
  std::unordered_map<const Var*, Definition> defined_at_;
  std::unordered_map<const Var*, CallPtr> definitions_;
  BufferActiveDomainWork* work_;
  size_t current_scope_ = 0, clock_ = 0;

  void Define(const VarPtr& value) {
    names.insert(value->name_hint_);
    defined_at_.emplace(value.get(), Definition{++clock_, current_scope_});
    const auto memory = GetTypeMemRef(value->GetType());
    if (As<TileType>(value->GetType()) && memory) {
      const auto [first, inserted] = memory_roots_.emplace((*memory)->base_.get(), value);
      if (!inserted) Alias(value, first->second);
    }
  }
  template <typename Fn>
  void Enter(Fn&& action) {
    const auto parent = current_scope_;
    current_scope_ = scopes_.size();
    const auto scope = current_scope_;
    scopes_.push_back({++clock_, 0, parent});
    action();
    scopes_[scope].end = ++clock_;
    current_scope_ = parent;
  }
  bool Visible(const ExprPtr& extent, size_t scope, size_t clock) const {
    if (As<ConstInt>(extent)) return true;
    const auto var = AsVarLike(extent);
    const auto found = var ? defined_at_.find(var.get()) : defined_at_.end();
    if (found == defined_at_.end() || found->second.clock >= clock) return false;
    const auto owner = found->second.scope;
    // An active ancestor's end has not been assigned; completed sibling scopes
    // have a nonzero end before the present cut.
    return owner == scope || (scopes_[owner].begin <= scopes_[scope].begin && scopes_[owner].end == 0);
  }
  std::unordered_map<const Var*, VarPtr> memory_roots_;
  void Alias(const VarPtr& a, const ExprPtr& expression) {
    if (const auto b = AsVarLike(expression); b && As<TileType>(a->GetType())) {
      aliases[a.get()].push_back(b.get());
      aliases[b.get()].push_back(a.get());
    }
  }
  void ObserveUnknown(const CallPtr& call) {
    if (IsOp(call, "tile.store")) {
      observations.emplace_back(call->args_[0], Declared(call->args_[0]->GetType()));
      return;
    }
    for (const auto& arg : call->args_) {
      if (!As<TileType>(arg->GetType())) continue;
      observations.emplace_back(arg, Domain{DomainKind::Full, nullptr, 0});
      if (auto value = AsVarLike(arg)) unsafe_metadata.insert(value.get());
    }
  }
  Domain Forward(const CallPtr& call) const {
    if (IsOp(call, "tile.full") || IsOp(call, "tile.load") || IsOp(call, "tile.ci")) {
      return Declared(call->GetType());
    }
    if (IsOp(call, "tile.set_validshape")) return Meet(Get(call->args_[0]), Declared(call->GetType()));
    if (IsVectorReshape(call)) return ReshapeDomain(Get(call->args_[0]), call, call->GetType());
    if (IsOp(call, "tile.concat")) {
      // A column prefix of the second source starts after the first source's
      // entire width in the result. The elementwise meet does not model that
      // offset, and the native recipe accepts only fully defined rectangles.
      if (Get(call->args_[0]).kind != DomainKind::Full || Get(call->args_[1]).kind != DomainKind::Full) {
        return {};
      }
      return Declared(call->GetType());
    }
    if (const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call)) {
      Domain result{DomainKind::Full, nullptr, 0};
      for (const auto& input : recipe->inputs) {
        if (input.kind == backend::BufferElementwiseOperandKind::Buffer) {
          result = Meet(result, Get(call->args_[input.logical_index]));
        }
      }
      return Meet(result, Declared(call->GetType()));
    }
    if (IsCast(call)) {
      Domain result{DomainKind::Full, nullptr, 0};
      for (const auto& arg : call->args_)
        if (As<TileType>(arg->GetType())) result = Meet(result, Get(arg));
      return Meet(result, Declared(call->GetType()));
    }
    if (const auto axis = backend::FindLogicalBufferAxisRecipe(call->op_->name_);
        axis && (axis->axis == 1 || !axis->reduction)) {
      auto source = Get(call->args_[0]);
      if (source.kind == DomainKind::Prefix && source.axis != 0) return {};
      if (axis->reduction) return Meet(source, Declared(call->GetType()));
      if (axis->axis == 1)
        source = Meet(source, Get(call->args_[1]));
      else if (Get(call->args_[1]).kind != DomainKind::Full)
        return {};
      return Meet(source, Declared(call->GetType()));
    }
    return {};
  }
};

class CallIndex : public IRVisitor {
 public:
  explicit CallIndex(BufferActiveDomainWork* work) : work_(work) {}
  std::unordered_map<const Var*, CallPtr> definitions;
  // This index needs statement definitions, not expression use paths. In
  // particular, walking a shared condition DAG as a tree can do exponential
  // work before an otherwise immediate normalization refusal. Statements are
  // still visited at every original occurrence by the inherited traversal.
  void VisitExpr(const ExprPtr&) override {}
  void VisitStmt(const StmtPtr& statement) override {
    if (work_) ++work_->definition_statements;
    IRVisitor::VisitStmt(statement);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const AssignStmtPtr& statement) override {
    if (auto call = As<Call>(statement->value_)) definitions.emplace(statement->var_.get(), call);
  }

 private:
  BufferActiveDomainWork* work_;
};

// Observation is independent of expression position. A call nested in a guard,
// arithmetic expression, tuple or attribute can inspect the same descriptor as
// a top-level scalar assignment. Two memo states distinguish ordinary data uses
// from unqualified observation; sharing a Var between those contexts must not
// hide the latter. Statements remain occurrence-sensitive. IterArg references
// denote their original identities, not a recursive walk of their initializers.
class MetadataUseIndex : public IRVisitor {
 public:
  explicit MetadataUseIndex(BufferActiveDomainWork* work,
                            const std::unordered_set<const Call*>* qualified = nullptr)
      : work_(work), qualified_(qualified) {}
  std::unordered_set<const Var*> unsafe;
  std::vector<ExprPtr> scalar_expressions;
  std::vector<ExprPtr> aggregate_expressions;
  bool unbound_tile_observer = false;
  bool tile_bearing_aggregate = false;

  void VisitExpr(const ExprPtr& expression) override {
    if (!expression) return;
    const auto call = As<Call>(expression);
    const bool unqualified_call = qualified_ && call && !QualifiedTopLevel(call);
    auto& seen = observing_ || unqualified_call ? seen_observed_ : seen_data_;
    if (!seen.insert(expression.get()).second) return;
    if (work_) ++work_->metadata_expression_states;
    if (As<ScalarType>(expression->GetType())) scalar_expressions.push_back(expression);
    if (As<MakeTuple>(expression) || As<TupleGetItemExpr>(expression))
      aggregate_expressions.push_back(expression);
    if (observing_ && As<TileType>(expression->GetType()) && !AsVarLike(expression))
      unbound_tile_observer = true;
    ObserveType(expression->GetType());
    IRVisitor::VisitExpr(expression);
  }

  void VisitFunction(const FunctionPtr& function) override {
    for (const auto& parameter : function->params_) VisitExpr(parameter);
    for (const auto& type : function->return_types_) ObserveType(type);
    ObserveAttributes(function->attrs_);
    VisitStmt(function->body_);
  }

  void VisitStmt(const StmtPtr& statement) override {
    if (work_) ++work_->metadata_statements;
    if (const auto scope = As<ScopeStmt>(statement)) ObserveAttributes(scope->attrs_);
    if (const auto loop = As<ForStmt>(statement)) ObserveAttributes(loop->attrs_);
    IRVisitor::VisitStmt(statement);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& assignment) override {
    const auto previous = direct_call_;
    direct_call_ = As<Call>(assignment->value_).get();
    IRVisitor::VisitStmt_(assignment);
    direct_call_ = previous;
  }

  void VisitVarLike_(const VarPtr& value) override {
    if (observing_ && As<TileType>(value->GetType())) unsafe.insert(value.get());
  }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }
  void VisitExpr_(const CallPtr& call) override {
    const bool previous = observing_;
    observing_ |= qualified_ ? !QualifiedTopLevel(call) : !KnownData(call);
    for (const auto& argument : call->args_) VisitExpr(argument);
    // Reference-valued compiler metadata has no data-use restoration recipe.
    observing_ = true;
    for (const auto& [name, attribute] : call->attrs_)
      ForEachAttrExpr(attribute, [this](const ExprPtr& value) { VisitExpr(value); });
    if (qualified_) {
      for (const auto& [name, keyword] : call->kwargs_)
        ForEachAttrExpr(keyword, [this](const ExprPtr& value) { VisitExpr(value); });
    }
    observing_ = previous;
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    const bool previous = observing_;
    observing_ = true;
    IRVisitor::VisitExpr_(submit);
    observing_ = previous;
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    // Initializers execute at the declaration cut, including zero-trip loops.
    // Body references to an IterArg still denote only its SSA identity; they
    // must not recursively replay this expression or hide its observations.
    for (const auto& carry : loop->iter_args_) VisitExpr(carry->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& carry : loop->iter_args_) VisitExpr(carry->initValue_);
    IRVisitor::VisitStmt_(loop);
  }

 private:
  bool observing_ = false;
  std::unordered_set<const Expr*> seen_data_, seen_observed_;
  std::unordered_set<const Type*> seen_types_;
  std::unordered_map<const Type*, bool> contains_tile_;
  BufferActiveDomainWork* work_;
  const std::unordered_set<const Call*>* qualified_;
  const Call* direct_call_ = nullptr;

  bool QualifiedTopLevel(const CallPtr& call) const {
    return call.get() == direct_call_ && qualified_->count(call.get());
  }

  bool ContainsTile(const TypePtr& type) {
    const auto found = contains_tile_.find(type.get());
    if (found != contains_tile_.end()) return found->second;
    bool contains = As<TileType>(type) != nullptr;
    if (const auto tuple = As<TupleType>(type))
      for (const auto& element : tuple->types_) contains = ContainsTile(element) || contains;
    contains_tile_.emplace(type.get(), contains);
    return contains;
  }

  void ObserveAttributes(const std::vector<std::pair<std::string, std::any>>& attributes) {
    const bool previous = observing_;
    observing_ = true;
    for (const auto& [name, attribute] : attributes)
      ForEachAttrExpr(attribute, [this](const ExprPtr& value) { VisitExpr(value); });
    observing_ = previous;
  }

  void ObserveType(const TypePtr& type) {
    if (!seen_types_.insert(type.get()).second) return;
    if (work_) ++work_->metadata_types;
    // Aggregate def-use and projection normalization is outside this family.
    // Refuse Tile-bearing tuples explicitly instead of treating their SSA
    // handles as stable opaque leaves and hiding a changed contained identity.
    tile_bearing_aggregate |= As<TupleType>(type) && ContainsTile(type);
    const bool previous = observing_;
    observing_ = true;
    // Reuse the shared type-expression inventory without changing any field.
    // Types and expressions have separate identity caches, including shared
    // shape/stride/slot selectors. This also observes scalar calls in metadata.
    CloneTypeWithMemRefAndRemapExprs(type, GetTypeMemRef(type), [&](const ExprPtr& expression) {
      VisitExpr(expression);
      return expression;
    });
    observing_ = previous;
  }
};

// Translate a result's value requirement to one operand. Scratch and full row
// carriers retain their original contracts. This is also the only table used to
// construct matching native operand metadata during the subsequent rewrite.
Domain OperandDomain(const CallPtr& call, size_t argument, Domain output) {
  if (IsElementwise(call) || IsCast(call)) return output;
  if (IsVectorReshape(call)) return ReshapeDomain(output, call, call->args_[0]->GetType());
  if (const auto axis = backend::FindLogicalBufferAxisRecipe(call->op_->name_);
      axis && (axis->axis == 1 || !axis->reduction)) {
    if (axis->reduction && argument == 1) return {DomainKind::Unused, nullptr, 0};
    if (axis->axis == 0 && !axis->reduction && argument == 1) return {DomainKind::Full, nullptr, 0};
    return output;
  }
  if (IsOp(call, "tile.set_validshape")) return output;
  return {DomainKind::Full, nullptr, 0};
}

bool QualifyUses(DomainIndex& index) {
  std::deque<const Var*> pending;
  for (auto& [value, node] : index.nodes) node.required = {DomainKind::Unused, nullptr, 0};
  const auto require = [&](const ExprPtr& expression, const Domain& domain) {
    const auto value = AsVarLike(expression);
    auto found = value ? index.nodes.find(value.get()) : index.nodes.end();
    if (found == index.nodes.end() || domain.kind == DomainKind::Unused) return;
    auto& old = found->second.required;
    const auto merged =
        old.kind == DomainKind::Unused || Same(old, domain) ? domain : Domain{DomainKind::Full, nullptr, 0};
    if (!Same(old, merged)) {
      old = merged;
      pending.push_back(value.get());
    }
  };
  for (const auto& [value, domain] : index.observations) require(value, domain);
  while (!pending.empty()) {
    const auto value = pending.front();
    pending.pop_front();
    const auto& node = index.nodes.at(value);
    for (size_t arg = 0; arg < node.dependencies.size(); ++arg) {
      const auto& dependency = node.dependencies[arg];
      if (!dependency || !As<TileType>(dependency->GetType())) continue;
      if (node.call &&
          (IsOp(node.call, "tile.load") || IsOp(node.call, "tile.full") || IsOp(node.call, "tile.create")))
        continue;
      require(dependency, node.call ? OperandDomain(node.call, arg, node.required) : node.required);
    }
  }
  // Metadata aliases share native state. A known value read can restore its
  // original logical metadata; an unknown observer cannot. Propagate this
  // separate refusal fact through aliases, including the carry's initializer.
  pending.assign(index.unsafe_metadata.begin(), index.unsafe_metadata.end());
  while (!pending.empty()) {
    const auto value = pending.front();
    pending.pop_front();
    const auto found = index.aliases.find(value);
    if (found == index.aliases.end()) continue;
    for (const auto alias : found->second)
      if (index.unsafe_metadata.insert(alias).second) pending.push_back(alias);
  }
  bool changed = false;
  for (const auto& [value, node] : index.nodes) {
    const auto declared = Declared(node.value->GetType());
    if (node.defined.kind != DomainKind::Prefix || Same(node.defined, declared)) continue;
    const auto type = As<TileType>(node.value->GetType());
    // A named runtime extent promotes the shared root to the existing dynamic
    // metadata owner, which restores old full aliases at each data use. A
    // purely static restriction has no such restoration contract in lowering.
    if (!AsVarLike(node.defined.extent) || type->dtype_ != DataType::FP32 ||
        !Contains(node.defined, node.required) || index.unsafe_metadata.count(value))
      return false;
    changed = true;
  }
  for (const auto& [value, node] : index.nodes) {
    if (!node.call || node.defined.kind != DomainKind::Prefix) continue;
    for (size_t arg = 0; arg < node.dependencies.size(); ++arg) {
      const auto& source = node.dependencies[arg];
      if (!As<TileType>(source->GetType())) continue;
      const auto need = OperandDomain(node.call, arg, node.defined);
      if (need.kind == DomainKind::Prefix && !Same(Declared(source->GetType()), need)) {
        if (!AsVarLike(need.extent)) return false;
        changed = true;
      }
    }
  }
  // Also check the original full operand of every inserted restriction. Its
  // unchanged SSA identity may have a metadata observer outside the recurrence.
  if (changed) {
    for (const auto& [value, node] : index.nodes) {
      if (!Contains(node.defined, node.required)) return false;
      if (node.defined.kind != DomainKind::Prefix) continue;
      for (size_t arg = 0; arg < node.dependencies.size(); ++arg) {
        const auto source = AsVarLike(node.dependencies[arg]);
        if (!source || !As<TileType>(source->GetType())) continue;
        const auto need = node.call ? OperandDomain(node.call, arg, node.defined) : node.defined;
        if (need.kind == DomainKind::Prefix && index.Get(source).kind == DomainKind::Full &&
            index.unsafe_metadata.count(source.get()))
          return false;
      }
    }
  }
  return changed && !index.invalid_carry;
}

// Structural stability is a property of the original immutable expression DAG,
// not of a use's current bindings. The rewriter preserves scalar/Tensor bindings
// and changes only Tile declarations whose proved prefix differs. Scalar calls
// with unchanged operands/attrs can therefore remain the original Expr, even
// when they execute at several statement occurrences. This is not evaluation
// CSE. An expression depending on a changed Tile binding refuses the candidate.
class ScalarStability {
 public:
  explicit ScalarStability(std::unordered_set<const Var*> changed)
      : changed_tiles_(std::move(changed)), work_(nullptr) {}
  ScalarStability(const DomainIndex& index, BufferActiveDomainWork* work) : work_(work) {
    // Immutable superset of precisely the Tile identities that rewriting may
    // replace. Same-type definitions and already-matching carries retain their
    // original bindings, including every scalar/Tensor definition.
    for (const auto& [value, node] : index.nodes)
      if (!Same(node.defined, Declared(node.value->GetType()))) changed_tiles_.insert(value);
  }
  bool Check(const ExprPtr& expression) {
    const auto found = stable_.find(expression.get());
    if (found != stable_.end()) return found->second;
    if (work_) ++work_->scalar_stability_nodes;
    bool stable = false;
    if (const auto value = AsVarLike(expression)) {
      stable = As<ScalarType>(value->GetType()) || As<TensorType>(value->GetType()) ||
               (As<TileType>(value->GetType()) && !changed_tiles_.count(value.get()));
      // Aggregate bindings are not opaque stable leaves: they can contain a
      // Tile whose declaration changes. This bounded slice refuses their
      // projection rather than inspecting definitions again at each use.
    } else if (As<ConstInt>(expression) || As<ConstFloat>(expression) || As<ConstBool>(expression)) {
      stable = true;
    } else if (const auto binary = As<BinaryExpr>(expression)) {
      stable = Check(binary->left_) && Check(binary->right_);
    } else if (const auto unary = As<UnaryExpr>(expression)) {
      stable = Check(unary->operand_);
    } else if (const auto tuple = As<MakeTuple>(expression)) {
      stable = true;
      for (const auto& element : tuple->elements_) stable = Check(element) && stable;
    } else if (const auto projection = As<TupleGetItemExpr>(expression)) {
      stable = Check(projection->tuple_);
    } else if (const auto call = As<Call>(expression)) {
      stable = true;
      for (const auto& argument : call->args_) stable = Check(argument) && stable;
      for (const auto& [name, attribute] : call->attrs_)
        ForEachAttrExpr(attribute, [&](const ExprPtr& value) { stable = Check(value) && stable; });
    }
    stable_.emplace(expression.get(), stable);
    return stable;
  }

 private:
  std::unordered_set<const Var*> changed_tiles_;
  std::unordered_map<const Expr*, bool> stable_;
  BufferActiveDomainWork* work_;
};

class PrefixRewriter : public IRMutator {
 public:
  PrefixRewriter(const DomainIndex& index, BufferActiveDomainWork* work)
      : index_(index), names_(index.names), work_(work) {}

  bool Available() const { return available_; }

  FunctionPtr VisitFunction(const FunctionPtr& function) override {
    // Admission preserves the function ABI and metadata: only local Tile
    // declarations can change. Avoid the base function mutator's recursive
    // TupleType remapping of unchanged parameters/returns as well as Var uses.
    auto body = VisitStmt(function->body_);
    if (body == function->body_) return function;
    auto result = MutableCopy(function);
    result->body_ = std::move(body);
    return result;
  }

 protected:
  ExprPtr VisitExpr_(const VarPtr& expression) override {
    CountExpression();
    // Every non-Tile binding/type is preserved by this qualified rewrite.
    // The base mutator recursively walks TupleType as a tree, which would
    // expand a shared scalar-only tuple's type again at every use.
    return As<TileType>(expression->GetType()) ? IRMutator::VisitExpr_(expression) : expression;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& statement) override {
    CountStatement();
    return IRMutator::VisitStmt_(statement);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& statement) override {
    CountStatement();
    return IRMutator::VisitStmt_(statement);
  }
  StmtPtr VisitStmt_(const WhileStmtPtr& statement) override {
    CountStatement();
    return IRMutator::VisitStmt_(statement);
  }
  StmtPtr VisitStmt_(const YieldStmtPtr& statement) override {
    CountStatement();
    return IRMutator::VisitStmt_(statement);
  }
  StmtPtr VisitStmt_(const ReturnStmtPtr& statement) override {
    CountStatement();
    return IRMutator::VisitStmt_(statement);
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    CountStatement();
    return IRMutator::VisitStmt_(statement);
  }

  ExprPtr VisitExpr_(const IterArgPtr& expression) override {
    CountExpression();
    // Declaration initializers were qualified separately. A scalar reference
    // must not replay an arbitrarily deep initializer/carry chain at every use.
    return As<ScalarType>(expression->GetType()) ? expression : IRMutator::VisitExpr_(expression);
  }

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& expression) override {
    CountExpression();
    return expression;
  }

  ExprPtr VisitExpr_(const MakeTuplePtr& expression) override {
    CountExpression();
    // Aggregate expressions were checked before rewriting, including their
    // scalar calls and unchanged Tensor operands. Preserve their entire DAG.
    return expression;
  }

  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& expression) override {
    CountExpression();
    return expression;
  }

  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& expression) override {
    CountExpression();
    return expression;
  }

  ExprPtr VisitExpr_(const CallPtr& expression) override {
    CountExpression();
    return As<ScalarType>(expression->GetType()) ? expression : IRMutator::VisitExpr_(expression);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& original) override {
    CountStatement();
    const auto old_call = As<Call>(original->value_);
    const auto found = index_.nodes.find(original->var_.get());
    if (!old_call && AsVarLike(original->value_) && As<TileType>(original->var_->GetType())) {
      // A plain alias has the source's descriptor, including its proved active
      // prefix. Rebinding only its RHS leaves an overwide LHS and invalid IR;
      // bind the new declaration before later aliases and payload uses visit it.
      return RebindAssignment(original, VisitExpr(original->value_));
    }
    if (!old_call || !KnownData(old_call)) return IRMutator::VisitStmt_(original);
    auto rebuilt = As<AssignStmt>(IRMutator::VisitStmt_(original));
    auto call = As<Call>(rebuilt->value_);
    std::vector<StmtPtr> prologue;
    auto args = call->args_;
    if (found != index_.nodes.end() && found->second.defined.kind == DomainKind::Prefix) {
      for (size_t arg = 0; arg < args.size(); ++arg) {
        if (!As<TileType>(args[arg]->GetType())) continue;
        const auto expected = OperandDomain(old_call, arg, found->second.defined);
        if (expected.kind == DomainKind::Prefix && !Same(Declared(args[arg]->GetType()), expected))
          args[arg] = Restrict(args[arg], expected, prologue);
      }
    }
    bool moved = false;
    for (size_t arg = 0; arg < args.size(); ++arg) moved |= args[arg] != old_call->args_[arg];
    if (!moved) return rebuilt;
    auto value = OpRegistry::GetInstance().Create(call->op_->name_, args, call->kwargs_, call->span_);
    value = transform_utils::PreserveCallAttrs(call, value);
    prologue.push_back(RebindAssignment(original, value));
    return SeqStmts::Flatten(prologue, original->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& original) override {
    CountStatement();
    std::vector<StmtPtr> prologue;
    auto carries = original->iter_args_;
    auto results = original->return_vars_;
    for (size_t i = 0; i < carries.size(); ++i) {
      const auto decision = index_.carries.find(carries[i].get());
      if (decision == index_.carries.end()) continue;
      if (Same(Declared(carries[i]->GetType()), decision->second)) continue;
      auto initial = Restrict(VisitExpr(carries[i]->initValue_), decision->second, prologue);
      carries[i] =
          std::make_shared<IterArg>(carries[i]->name_hint_, initial->GetType(), initial, carries[i]->span_);
      Bind(original->iter_args_[i].get(), carries[i]);
      results[i] = std::make_shared<Var>(results[i]->name_hint_, initial->GetType(), results[i]->span_);
      Bind(original->return_vars_[i].get(), results[i]);
    }
    auto body = VisitStmt(original->body_);
    auto attributes = MutateScopeAttrs(original->attrs_).first;
    for (const auto& carry : original->iter_args_) var_remap_.erase(carry.get());
    auto loop = MutableCopy(As<ForStmt>(loop_repair::RebuildForStmt(original, carries, body, results)));
    loop->attrs_ = std::move(attributes);
    prologue.push_back(loop);
    return SeqStmts::Flatten(prologue, original->span_);
  }

 private:
  const DomainIndex& index_;
  std::set<std::string> names_;
  size_t next_name_ = 0;
  BufferActiveDomainWork* work_;
  bool available_ = true;

  void CountExpression() {
    if (work_) ++work_->rewrite_expression_dispatches;
  }
  void CountStatement() {
    // Base IRMutator children use StmtFunctor directly, bypassing its generic
    // VisitStmt dispatcher. Count the actual supported statement handlers.
    if (work_) ++work_->rewritten_statements;
  }

  void Bind(const Expr* original, const ExprPtr& replacement) { var_remap_[original] = replacement; }

  StmtPtr RebindAssignment(const AssignStmtPtr& original, const ExprPtr& value) {
    const auto type = WithCarriedMemRef(value->GetType(), original->var_->GetType());
    const auto old_memory = GetTypeMemRef(original->var_->GetType());
    const auto new_memory = GetTypeMemRef(type);
    if (old_memory && new_memory &&
        ((*old_memory)->base_ != (*new_memory)->base_ || !structural_equal(*old_memory, *new_memory))) {
      // Descriptor normalization cannot redirect an existing allocation. Keep
      // the entire original function when an alias cannot preserve its storage;
      // no partially rebound candidate may escape this immutable construction.
      available_ = false;
      return original;
    }
    if (value == original->value_ && structural_equal(type, original->var_->GetType())) return original;
    auto assignment = MutableCopy(original);
    assignment->value_ = value;
    if (!structural_equal(type, original->var_->GetType())) {
      INTERNAL_CHECK_SPAN(As<TileType>(type) && As<TileType>(original->var_->GetType()), original->span_)
          << "Internal error: prefix normalization must preserve scalar/Tensor binding types";
      assignment->var_ = std::make_shared<Var>(original->var_->name_hint_, type, original->var_->span_);
      Bind(original->var_.get(), assignment->var_);
    }
    // Same-type definitions retain their SSA identity. In particular, Tensor
    // stores keep scalar queries of that Tensor stable without use rewriting.
    return assignment;
  }

  std::string FreshName() {
    // One shared monotonically advancing namespace avoids quadratic probing
    // when many independent consumers require the same kind of restriction.
    std::string name;
    do {
      name = "active_prefix_" + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    return name;
  }

  ExprPtr Restrict(const ExprPtr& source, const Domain& domain, std::vector<StmtPtr>& prologue) {
    if (Same(Declared(source->GetType()), domain)) return source;
    const auto type = As<TileType>(source->GetType());
    auto valid = type->shape_;
    valid[domain.axis] = VisitExpr(domain.extent);
    auto call =
        OpRegistry::GetInstance().Create("tile.set_validshape", {source, valid[0], valid[1]}, source->span_);
    auto result = std::make_shared<Var>(FreshName(), WithCarriedMemRef(call->GetType(), source->GetType()),
                                        source->span_);
    prologue.push_back(std::make_shared<AssignStmt>(result, call, source->span_));
    return result;
  }
};

bool IsGuardedRowArithmetic(const CallPtr& call) {
  using backend::BufferElementwiseOperandKind;
  using backend::BufferElementwiseTypePolicy;
  using backend::BufferPrecisionKind;
  const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call);
  return recipe && recipe->default_enabled && recipe->inputs.size() == 2 &&
         recipe->precision == BufferPrecisionKind::None &&
         (recipe->types == BufferElementwiseTypePolicy::MatchingVec ||
          recipe->types == BufferElementwiseTypePolicy::DenseFP32) &&
         recipe->inputs[0].logical_index == 0 && recipe->inputs[1].logical_index == 1 &&
         std::all_of(recipe->inputs.begin(), recipe->inputs.end(),
                     [](const auto& input) { return input.kind == BufferElementwiseOperandKind::Buffer; });
}

// Follow a prefix only through equal-shaped lane operations. A guarded row
// extraction discharges it: consumers of that row receive exactly the same
// bytes and valid shape. The load remains partial; only selected read edges
// will use the separately allocated padded value.
class GuardedRowUses : public IRVisitor {
 public:
  struct Prefix {
    ExprPtr extent;
    bool refused = false;
    bool observed = false;
  };
  struct Value {
    Prefix* prefix;
    bool load;
  };

  std::unordered_map<const Var*, Value> values;
  std::unordered_set<const Call*> terminals;
  std::unordered_set<const Call*> extracts;
  std::unordered_set<const Call*> qualified;
  std::unordered_set<const Var*> padded_sources;
  std::unordered_map<const Var*, Prefix*> extract_results;
  std::set<std::string> names;

  void VisitExpr(const ExprPtr& value) override {
    // Arbitrary scalar or aggregate expression users are checked by the
    // bounded MetadataUseIndex, with these proven terminals as its exceptions.
    if (const auto var = AsVarLike(value)) Refuse(var);
  }

  void CheckMetadata(const MetadataUseIndex& metadata) {
    for (const auto* var : metadata.unsafe) {
      if (const auto value = values.find(var); value != values.end()) value->second.prefix->refused = true;
    }
  }

  bool Eligible(const Var* value) const {
    const auto found = values.find(value);
    return found != values.end() && padded_sources.count(value) && !found->second.prefix->refused &&
           found->second.prefix->observed;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    names.insert(stmt->var_->name_hint_);
    const auto call = As<Call>(stmt->value_);
    const auto type = As<TileType>(stmt->var_->GetType());
    if (!type) {
      if (!call && GetScalarDtypeIfBool(stmt->var_)) guards_.emplace(stmt->var_.get(), stmt->value_);
      if (call) CheckConsumers(call);
      return;
    }
    const auto memory = GetTypeMemRef(type).value_or(nullptr);
    const auto padded_load = LoadPrefix(call, stmt->var_, type, memory);
    if (padded_load) {
      auto [prefix, inserted] = prefixes_.try_emplace(padded_load.get(), Prefix{padded_load});
      AddValue(stmt->var_, memory, {&prefix->second, true});
      return;
    }
    if (const auto prefix = ArithmeticPrefix(call, type)) {
      AddValue(stmt->var_, memory, {prefix, false});
      for (const auto& arg : call->args_) RecordPad(arg);
      qualified.insert(call.get());
      return;
    }
    if (call && QualifyRowObserver(call, stmt->var_, type)) return;
    if (call)
      CheckConsumers(call);
    else
      VisitExpr(stmt->value_);
    MarkRoot(memory);
  }

  void VisitStmt_(const EvalStmtPtr& stmt) override {
    if (const auto call = As<Call>(stmt->expr_))
      CheckConsumers(call);
    else
      VisitExpr(stmt->expr_);
  }

  void VisitStmt_(const ForStmtPtr& loop) override {
    names.insert(loop->loop_var_->name_hint_);
    for (const auto& arg : loop->iter_args_) {
      names.insert(arg->name_hint_);
      VisitExpr(arg->initValue_);
      MarkRoot(GetTypeMemRef(arg->GetType()).value_or(nullptr));
    }
    const auto start = Literal(loop->start_);
    const auto stop = Literal(loop->stop_);
    const auto step = Literal(loop->step_);
    const bool bounded = start && stop && step && start->value_ < stop->value_ && step->value_ > 0 &&
                         __int128(stop->value_) - 1 + step->value_ <= std::numeric_limits<int64_t>::max() &&
                         GetScalarDtype(loop->loop_var_) == DataType::INDEX;
    if (bounded) analyzer_->Bind(loop->loop_var_, start->value_, stop->value_);
    VisitStmt(loop->body_);
    if (bounded) analyzer_->Unbind(loop->loop_var_);
    for (const auto& arg : loop->return_vars_) names.insert(arg->name_hint_);
  }

  void VisitStmt_(const IfStmtPtr& stmt) override {
    {
      BufferAccTrimGuardScope guard(stmt->condition_, true, analyzer_, guards_);
      VisitStmt(stmt->then_body_);
    }
    if (stmt->else_body_) {
      BufferAccTrimGuardScope guard(stmt->condition_, false, analyzer_, guards_);
      VisitStmt(*stmt->else_body_);
    }
    for (const auto& arg : stmt->return_vars_) names.insert(arg->name_hint_);
  }

 private:
  std::unordered_map<const Expr*, Prefix> prefixes_;
  std::unordered_map<const Var*, Prefix*> roots_;
  std::unordered_map<const Var*, ExprPtr> guards_;
  arith::AnalyzerPtr analyzer_ = std::make_shared<arith::Analyzer>();

  static ConstIntPtr Literal(const ExprPtr& value) {
    const auto constant = As<ConstInt>(value);
    return constant && arith::detail::ConstIntFitsDtype(constant->value_, constant->dtype()) ? constant
                                                                                             : nullptr;
  }

  static bool GetScalarDtypeIfBool(const VarPtr& var) {
    const auto scalar = As<ScalarType>(var->GetType());
    return scalar && scalar->dtype_ == DataType::BOOL;
  }

  static ExprPtr LoadPrefix(const CallPtr& call, const VarPtr& value, const TileTypePtr& tile,
                            const MemRefPtr& memory) {
    if (!IsOp(call, "tile.load") || tile->dtype_ != DataType::FP32 || tile->shape_.size() != 2 || !memory ||
        !memory->base_ || memory->slot_count_ != 1 || memory->slot_index_ || memory->is_pinned_) {
      return nullptr;
    }
    const auto domain = Declared(tile);
    if (domain.kind != DomainKind::Prefix || domain.axis != 0 || !AsVarLike(domain.extent)) return nullptr;
    const auto physical = backend::PhysicalLogicalVecDescriptor(tile);
    if (!backend::IsAlignedDenseVecBufferGeometry(physical)) return nullptr;
    const auto pad = OpRegistry::GetInstance().Create("tile.fillpad", {value},
                                                      {{"pad_value", PadValue::zero}}, call->span_);
    const auto* recipe = backend::FindLogicalBufferConversionRecipe("tile.fillpad");
    return recipe && backend::SupportsLogicalBufferConversion(*recipe, pad) ? domain.extent : nullptr;
  }

  void AddValue(const VarPtr& var, const MemRefPtr& memory, Value value) {
    values.emplace(var.get(), value);
    if (!memory || !memory->base_ || memory->is_pinned_ || memory->slot_count_ != 1 || memory->slot_index_) {
      value.prefix->refused = true;
      return;
    }
    const auto [root, added] = roots_.emplace(memory->base_.get(), value.prefix);
    if (!added) {
      value.prefix->refused = true;
      if (root->second) root->second->refused = true;
    }
  }

  void MarkRoot(const MemRefPtr& memory) {
    if (!memory || !memory->base_) return;
    const auto [root, added] = roots_.try_emplace(memory->base_.get(), nullptr);
    if (!added && root->second) root->second->refused = true;
  }

  const Value* Get(const ExprPtr& expr) const {
    const auto found = values.find(AsVarLike(expr).get());
    return found == values.end() ? nullptr : &found->second;
  }

  void RecordPad(const ExprPtr& expr) {
    if (const auto* value = Get(expr); value && value->load) padded_sources.insert(AsVarLike(expr).get());
  }

  void Refuse(const ExprPtr& expr) const {
    if (const auto* value = Get(expr)) value->prefix->refused = true;
    const auto memory = GetTypeMemRef(expr->GetType()).value_or(nullptr);
    if (!memory) return;
    const auto found = roots_.find(memory->base_.get());
    if (found != roots_.end() && found->second) found->second->refused = true;
  }

  void CheckConsumers(const CallPtr& call) const {
    const auto& registry = OpRegistry::GetInstance();
    const auto* entry = !As<GlobalVar>(call->op_) && registry.IsRegistered(call->op_->name_)
                            ? &registry.GetEntry(call->op_->name_)
                            : nullptr;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto& arg = call->args_[i];
      if (As<TileType>(arg->GetType()) &&
          (Get(arg) || !entry || entry->GetIRStage() != OpIRStage::Functional ||
           ArgEffectWrites(entry->GetArgEffect(i, call->kwargs_)))) {
        Refuse(arg);
      }
    }
  }

  Prefix* ArithmeticPrefix(const CallPtr& call, const TileTypePtr& result) const {
    if (!IsGuardedRowArithmetic(call) || call->args_.size() != 2 || result->dtype_ != DataType::FP32 ||
        Declared(result).kind != DomainKind::Full)
      return nullptr;
    Prefix* common = nullptr;
    for (const auto& arg : call->args_) {
      const auto type = As<TileType>(arg->GetType());
      const auto* value = Get(arg);
      if (!type || type->dtype_ != result->dtype_ ||
          !tile_view_semantics::ShapeExprListsEquivalent(type->shape_, result->shape_) ||
          (!value && Declared(type).kind != DomainKind::Full))
        return nullptr;
      if (!value) continue;
      if (common && common != value->prefix) return nullptr;
      common = value->prefix;
    }
    return common;
  }

  bool QualifyRowObserver(const CallPtr& call, const VarPtr& result_var, const TileTypePtr& result) {
    const bool slice = IsOp(call, "tile.slice") && call->args_.size() == 3;
    const bool extract = IsOp(call, "tile.extract") && call->args_.size() == 4;
    if (!slice && !extract) return false;
    const auto* value = Get(call->args_[0]);
    // Guarded slices are aliases of full logical arithmetic results. A guarded
    // extract from a load has independent storage and can use the padded copy.
    if (!value || (slice == value->load) || result->shape_.size() != 2) return false;
    const auto rows = Literal(result->shape_[0]);
    const auto source = As<TileType>(call->args_[0]->GetType());
    const auto declared = As<TileType>(call->GetType());
    if (!rows || rows->value_ != 1 || source->shape_.size() != 2 || source->dtype_ != result->dtype_ ||
        !declared || !structural_equal(CloneTypeWithoutMemRef(result), CloneTypeWithoutMemRef(declared)) ||
        (slice && Declared(result).kind != DomainKind::Full))
      return false;
    const auto offsets = slice ? As<MakeTuple>(call->args_[2]) : nullptr;
    if (slice && (!offsets || offsets->elements_.size() != 2)) return false;
    const auto row = slice ? offsets->elements_[0] : call->args_[1];
    const auto column = Literal(slice ? offsets->elements_[1] : call->args_[2]);
    // This recipe admits precisely a full-width row. Existing admission still
    // validates its native view, alignment, and independent extract layout.
    if (!column || column->value_ != 0 || !structural_equal(source->shape_[1], result->shape_[1]) ||
        (!AsVarLike(row) && !Literal(row)) || GetScalarDtype(row) != GetScalarDtype(value->prefix->extent) ||
        !analyzer_->CanProveGreaterEqual(row, 0) ||
        !analyzer_->CanProve(MakeLt(row, value->prefix->extent))) {
      return false;
    }
    if (extract) {
      // An annotated result must agree with the original extract's clipped
      // validity, including the full-width columns. Only the proved row is
      // preserved when the source is replaced with the full padded view.
      const auto deduced =
          OpRegistry::GetInstance().Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
      if (!structural_equal(CloneTypeWithoutMemRef(declared), CloneTypeWithoutMemRef(deduced->GetType())))
        return false;
    }
    if (!terminals.insert(call.get()).second) {
      value->prefix->refused = true;
      return false;
    }
    if (extract) {
      extracts.insert(call.get());
      RecordPad(call->args_[0]);
      extract_results.emplace(result_var.get(), value->prefix);
      MarkRoot(GetTypeMemRef(result).value_or(nullptr));
    }
    qualified.insert(call.get());
    value->prefix->observed = true;
    return true;
  }
};

class PaddingRewriter : public IRMutator {
 public:
  explicit PaddingRewriter(const std::set<std::string>& names) : names_(names) {}

  FunctionPtr VisitFunction(const FunctionPtr& original) override {
    auto body = VisitStmt(original->body_);
    if (allocations_.empty()) return original;
    auto result = MutableCopy(original);
    result->body_ = InsertAllocsIntoBody(body, allocations_);
    return result;
  }

 protected:
  AssignStmtPtr MakeZeroPadding(const ExprPtr& source, const Span& span) {
    const auto bytes = backend::DenseBufferBytes(backend::PhysicalLogicalVecDescriptor(source->GetType()));
    INTERNAL_CHECK_SPAN(bytes, span) << "Internal error: qualified zero extension lost its dense capacity";
    const auto base = std::make_shared<Var>(Fresh("mem_zero_extension_"), GetPtrType(), span);
    const auto memory = std::make_shared<MemRef>(base, int64_t{0}, *bytes, span);
    const auto deduced =
        OpRegistry::GetInstance().Create("tile.fillpad", {source}, {{"pad_value", PadValue::zero}}, span);
    const auto call = std::make_shared<Call>(deduced->op_, deduced->args_, deduced->kwargs_,
                                             CloneTypeWithoutMemRef(deduced->GetType()), span);
    const auto padded =
        std::make_shared<Var>(Fresh("zero_extended_"), CloneTypeWithMemRef(call->GetType(), memory), span);
    allocations_.push_back(CreateAllocStatement(memory, MemorySpace::Vec));
    return std::make_shared<AssignStmt>(padded, call, span);
  }

 private:
  std::set<std::string> names_;
  size_t next_name_ = 0;
  std::vector<StmtPtr> allocations_;

  std::string Fresh(const char* prefix) {
    std::string name;
    do {
      name = prefix + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    return name;
  }
};

class GuardedRowRewriter : public PaddingRewriter {
 public:
  explicit GuardedRowRewriter(const GuardedRowUses& uses) : PaddingRewriter(uses.names), uses_(uses) {}

 protected:
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& expression) override { return expression; }
  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& expression) override { return expression; }
  ExprPtr VisitExpr_(const MakeTuplePtr& expression) override { return expression; }
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& expression) override { return expression; }
  ExprPtr VisitExpr_(const IterArgPtr& expression) override {
    return As<ScalarType>(expression->GetType()) ? expression : IRMutator::VisitExpr_(expression);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& original) override {
    const auto call = As<Call>(original->value_);
    auto rewritten = As<AssignStmt>(IRMutator::VisitStmt_(original));
    if (uses_.Eligible(original->var_.get())) {
      const auto pad = MakeZeroPadding(rewritten->var_, rewritten->span_);
      padded_.emplace(rewritten->var_.get(), pad->var_);
      return SeqStmts::Flatten({rewritten, pad}, rewritten->span_);
    }
    if (!IsGuardedRowArithmetic(call) && !uses_.extracts.count(call.get())) return rewritten;
    auto replacement = RewriteUse(As<Call>(rewritten->value_), uses_.extracts.count(call.get()));
    if (!replacement) return rewritten;
    const auto type = WithCarriedMemRef(replacement->GetType(), original->var_->GetType());
    auto result = MutableCopy(rewritten);
    result->value_ = replacement;
    if (!structural_equal(type, original->var_->GetType())) {
      result->var_ = std::make_shared<Var>(original->var_->name_hint_, type, original->var_->span_);
      var_remap_[original->var_.get()] = result->var_;
    }
    return result;
  }

 private:
  const GuardedRowUses& uses_;
  std::unordered_map<const Var*, VarPtr> padded_;

  CallPtr RewriteUse(const CallPtr& call, bool extract) const {
    auto args = call->args_;
    bool changed = false;
    for (auto& arg : args) {
      const auto pad = padded_.find(AsVarLike(arg).get());
      if (pad == padded_.end()) continue;
      arg = pad->second;
      changed = true;
    }
    if (!changed) return nullptr;
    if (extract) {
      const auto deduced =
          OpRegistry::GetInstance().Create(call->op_->name_, args, call->kwargs_, call->span_);
      return transform_utils::PreserveCallAttrs(call, deduced);
    }
    auto result = MutableCopy(call);
    result->args_ = std::move(args);
    return result;
  }
};

FunctionPtr MaterializeGuardedRowPrefixes(const FunctionPtr& original, const DomainIndex& index,
                                          const MetadataUseIndex& original_metadata) {
  bool has_prefix_load = false;
  bool has_row_use = false;
  for (const auto& [value, node] : index.nodes) {
    has_prefix_load |=
        IsOp(node.call, "tile.load") && node.defined.kind == DomainKind::Prefix && node.defined.axis == 0;
    has_row_use |= IsOp(node.call, "tile.slice") || IsOp(node.call, "tile.extract");
    if (has_prefix_load && has_row_use) break;
  }
  if (!has_prefix_load || !has_row_use) return original;
  GuardedRowUses uses;
  for (const auto& param : original->params_) uses.names.insert(param->name_hint_);
  uses.VisitStmt(original->body_);
  if (uses.terminals.empty()) return original;
  MetadataUseIndex metadata(nullptr, &uses.qualified);
  metadata.VisitFunction(original);
  if (metadata.unbound_tile_observer || metadata.tile_bearing_aggregate) return original;
  uses.CheckMetadata(metadata);
  std::unordered_set<const Var*> rebound_extracts;
  for (const auto& [var, prefix] : uses.extract_results) {
    if (original_metadata.unsafe.count(var)) prefix->refused = true;
    if (!prefix->refused) rebound_extracts.insert(var);
  }
  if (std::none_of(uses.padded_sources.begin(), uses.padded_sources.end(),
                   [&](const Var* value) { return uses.Eligible(value); }))
    return original;
  ScalarStability scalars(std::move(rebound_extracts));
  for (const auto& expression : metadata.scalar_expressions)
    if (!scalars.Check(expression)) return original;
  for (const auto& expression : metadata.aggregate_expressions)
    if (!scalars.Check(expression)) return original;
  return GuardedRowRewriter(uses).VisitFunction(original);
}

// A partial load has no defined value outside its prefix when no padding
// was requested. These producers preserve a zero extension, so it is legal
// to choose zero for those unspecified lanes at a full-result add. In
// particular, set_validshape is excluded: it can narrow a fully defined
// buffer, whose old tail values must remain available to a physical read.
class ZeroExtensionProof {
 public:
  explicit ZeroExtensionProof(const DomainIndex& index) : index_(index) {}

  bool Proves(const ExprPtr& source) {
    std::vector<const Var*> path;
    bool proved = false;
    for (auto value = AsVarLike(source); value;) {
      const auto cached = proofs_.find(value.get());
      if (cached != proofs_.end()) {
        proved = cached->second;
        break;
      }
      path.push_back(value.get());
      const auto found = index_.nodes.find(value.get());
      if (found == index_.nodes.end() || !FreshPrefix(found->second)) break;
      const auto& node = found->second;
      const auto& call = node.call;
      if (IsOp(call, "tile.load")) {
        proved = true;
        break;
      }
      if (call && (IsVectorReshape(call) || IsCast(call) || IsOp(call, "tile.row_sum") ||
                   (IsOp(call, "tile.mul") && call->args_.size() == 2 && call->args_[0] == call->args_[1]))) {
        value = AsVarLike(call->args_[0]);
      } else if (!call && node.dependencies.size() == 1) {
        value = AsVarLike(node.dependencies[0]);
      } else {
        break;
      }
    }
    for (const auto* value : path) proofs_.emplace(value, proved);
    return proved;
  }

 private:
  const DomainIndex& index_;
  std::unordered_map<const Var*, bool> proofs_;

  bool FreshPrefix(const Node& node) const {
    const auto tile = As<TileType>(node.value->GetType());
    const auto memory = GetTypeMemRef(tile);
    if (!memory || node.defined.kind != DomainKind::Prefix || !Same(node.defined, Declared(tile)) ||
        tile_view_semantics::GetEffectiveTileView(*tile).pad != PadValue::null ||
        index_.unsafe_metadata.count(node.value.get()))
      return false;
    const auto* root = (*memory)->base_.get();
    const auto writers = index_.storage_writers.find(root);
    return writers != index_.storage_writers.end() && writers->second == 1 &&
           !index_.scratch_roots.count(root);
  }
};

class FullAddPaddingRewriter : public PaddingRewriter {
 public:
  FullAddPaddingRewriter(const std::set<std::string>& names, const std::unordered_set<const Var*>& additions)
      : PaddingRewriter(names), additions_(additions) {}

 protected:
  ExprPtr VisitExpr(const ExprPtr& expression) override { return expression; }

  StmtPtr VisitStmt_(const AssignStmtPtr& statement) override {
    if (!additions_.count(statement->var_.get())) return statement;
    const auto call = As<Call>(statement->value_);
    const auto padding = MakeZeroPadding(call->args_[1], statement->span_);
    const auto rewritten = MutableCopy(call);
    rewritten->args_[1] = padding->var_;
    const auto sum = MutableCopy(statement);
    sum->value_ = rewritten;
    return SeqStmts::Flatten({padding, sum}, statement->span_);
  }

 private:
  const std::unordered_set<const Var*>& additions_;
};

FunctionPtr MaterializeFullAddPadding(const FunctionPtr& original, const DomainIndex& index) {
  // An unknown Tile observer has no active-byte or value contract. Only
  // ordinary dataflow may observe the deterministic extension.
  if (!index.unsafe_metadata.empty()) return original;
  ZeroExtensionProof proof(index);
  std::unordered_set<const Var*> additions;
  for (const auto& [value, node] : index.nodes) {
    const auto& call = node.call;
    if (!IsOp(call, "tile.add") || call->args_.size() != 2 ||
        Declared(call->GetType()).kind != DomainKind::Full ||
        Declared(call->args_[0]->GetType()).kind != DomainKind::Full)
      continue;
    const auto& source = call->args_[1];
    const auto physical = backend::PhysicalLogicalVecDescriptor(source->GetType());
    if (!backend::IsAlignedDenseVecBufferGeometry(physical) || physical->dtype_ != DataType::FP32 ||
        !proof.Proves(source))
      continue;
    const auto pad = OpRegistry::GetInstance().Create("tile.fillpad", {source},
                                                      {{"pad_value", PadValue::zero}}, call->span_);
    const auto* fill = backend::FindLogicalBufferConversionRecipe("tile.fillpad");
    if (!backend::SupportsLogicalBufferConversion(*fill, pad)) continue;
    // Check the completed native add before allocating its new input. Keep
    // both the original operand and return declarations on every other use.
    const auto candidate = MutableCopy(call);
    candidate->args_[1] = pad;
    const auto* add = backend::FindLogicalBufferElementwiseRecipe(candidate);
    if (backend::SupportsLogicalBufferElementwise(*add, candidate)) additions.insert(value);
  }
  return additions.empty() ? original
                           : FullAddPaddingRewriter(index.names, additions).VisitFunction(original);
}

}  // namespace

BufferActiveDomainCandidate NormalizeBufferActiveDomains(const FunctionPtr& function,
                                                         BufferActiveDomainWork* work) {
  CHECK(function) << "Active-domain normalization requires a function";
  if (work) *work = {};
  CallIndex calls(work);
  calls.VisitStmt(function->body_);
  MetadataUseIndex metadata(work);
  metadata.VisitFunction(function);
  if (metadata.unbound_tile_observer || metadata.tile_bearing_aggregate) return {function, function};
  DomainIndex index(function, std::move(calls.definitions), work);
  index.unsafe_metadata.insert(metadata.unsafe.begin(), metadata.unsafe.end());
  if (!QualifyUses(index)) {
    auto candidate = MaterializeGuardedRowPrefixes(function, index, metadata);
    if (candidate == function) candidate = MaterializeFullAddPadding(function, index);
    return {function, std::move(candidate)};
  }
  ScalarStability scalars(index, work);
  for (const auto& expression : metadata.scalar_expressions)
    if (!scalars.Check(expression)) return {function, function};
  for (const auto& expression : metadata.aggregate_expressions)
    if (!scalars.Check(expression)) return {function, function};
  PrefixRewriter rewriter(index, work);
  auto candidate = rewriter.VisitFunction(function);
  if (!rewriter.Available()) return {function, function};
  return {function, std::move(candidate)};
}

}  // namespace pypto::ir
