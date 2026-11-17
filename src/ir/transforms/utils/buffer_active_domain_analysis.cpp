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

#include <any>
#include <cstddef>
#include <deque>
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
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
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
  if (!tile || tile->GetMemorySpace() != MemorySpace::Vec || tile->shape_.size() != 2) return {};
  const auto& view = tile_view_semantics::GetEffectiveTileView(*tile);
  const auto& valid = view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  if (valid.size() != 2) return {};
  Domain result{DomainKind::Full, nullptr, 0};
  for (int axis = 0; axis < 2; ++axis) {
    const auto capacity = As<ConstInt>(tile->shape_[axis]);
    if (!capacity || capacity->value_ <= 0) return {};
    if (structural_equal(valid[axis], tile->shape_[axis])) continue;
    // A named bound gives constant-time lexical/invariance checks. More general
    // expressions retain the original IR until they have a shared proof owner.
    if (result.kind == DomainKind::Prefix || (!AsVarLike(valid[axis]) && !As<ConstInt>(valid[axis])))
      return {};
    result = {DomainKind::Prefix, valid[axis], axis};
  }
  return result;
}

bool Contains(const Domain& defined, const Domain& required) {
  return required.kind == DomainKind::Unused ||
         (defined.kind != DomainKind::Unavailable &&
          (defined.kind == DomainKind::Full || Same(defined, required)));
}

bool IsElementwise(const CallPtr& call) {
  return backend::FindLogicalBufferElementwiseRecipe(call->op_->name_) != nullptr;
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
         IsOp(call, "tile.set_validshape") || IsRowAxisOp(call);
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
      if (IsVectorReshape(call) || IsOp(call, "tile.set_validshape")) Alias(statement->var_, call->args_[0]);
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
    if (IsOp(call, "tile.full") || IsOp(call, "tile.load")) return Declared(call->GetType());
    if (IsOp(call, "tile.set_validshape")) return Meet(Get(call->args_[0]), Declared(call->GetType()));
    if (IsVectorReshape(call)) return ReshapeDomain(Get(call->args_[0]), call, call->GetType());
    if (IsElementwise(call) || IsCast(call)) {
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
  explicit MetadataUseIndex(BufferActiveDomainWork* work) : work_(work) {}
  std::unordered_set<const Var*> unsafe;
  std::vector<ExprPtr> scalar_expressions;
  std::vector<ExprPtr> aggregate_expressions;
  bool unbound_tile_observer = false;
  bool tile_bearing_aggregate = false;

  void VisitExpr(const ExprPtr& expression) override {
    if (!expression) return;
    auto& seen = observing_ ? seen_observed_ : seen_data_;
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
  void VisitVarLike_(const VarPtr& value) override {
    if (observing_ && As<TileType>(value->GetType())) unsafe.insert(value.get());
  }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }
  void VisitExpr_(const CallPtr& call) override {
    const bool previous = observing_;
    observing_ |= !KnownData(call);
    for (const auto& argument : call->args_) VisitExpr(argument);
    // Reference-valued compiler metadata has no data-use restoration recipe.
    observing_ = true;
    for (const auto& [name, attribute] : call->attrs_)
      ForEachAttrExpr(attribute, [this](const ExprPtr& value) { VisitExpr(value); });
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
  if (!QualifyUses(index)) return {function, function};
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
