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

// A body-only immutable candidate reuses ordinary placed Tile allocations,
// flattened transfer windows and scalar SSA. Naming inspects distinct expression
// and type nodes; rewriting visits statement occurrences without walking their
// unchanged expressions, metadata, guards or carry initializers. Native memory
// and synchronization qualification remain owned by the shared contracts.

#include "pypto/ir/transforms/utils/buffer_scalar_staging.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_scalar_recipes.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/type_metadata.h"

namespace pypto::ir {
namespace {
class Names : public IRVisitor {
 public:
  std::set<std::string> names;
  void VisitFunction(const FunctionPtr& function) override {
    for (const auto& type : function->return_types_) TypeNames(type);
    IRVisitor::VisitFunction(function);
  }
  void VisitExpr(const ExprPtr& expression) override {
    if (!expression || !seen_.insert(expression.get()).second) return;
    TypeNames(expression->GetType());
    type_metadata::ForEachStorageMetadataExpr(
        expression, [this](const ExprPtr& value, type_metadata::StorageMetadataField) { VisitExpr(value); });
    IRVisitor::VisitExpr(expression);
  }

 protected:
  void VisitVarLike_(const VarPtr& variable) override { names.insert(variable->name_hint_); }

 private:
  std::unordered_set<const Expr*> seen_;
  std::unordered_set<const Type*> types_;
  void TypeNames(const TypePtr& type) {
    if (!type || !types_.insert(type.get()).second) return;
    type_metadata::ForEachTypeMetadataChild(
        type, [this](const TypePtr& child) { TypeNames(child); },
        [this](const ExprPtr& value) { VisitExpr(value); },
        [this](const ExprPtr& value) { VisitExpr(value); });
  }
};

class Stage : public IRMutator {
 public:
  explicit Stage(std::set<std::string> names) : names_(std::move(names)) {}
  std::vector<StmtPtr> allocations;
  ExprPtr VisitExpr(const ExprPtr& expression) override { return expression; }
  StmtPtr VisitStmt(const StmtPtr& statement) override {
    if (!As<AssignStmt>(statement) && !As<EvalStmt>(statement) && !As<SeqStmts>(statement) &&
        !As<ForStmt>(statement) && !As<WhileStmt>(statement) && !As<IfStmt>(statement))
      return statement;
    return IRMutator::VisitStmt(statement);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& statement) override {
    std::vector<StmtPtr> children;
    bool changed = false;
    for (const auto& original : statement->stmts_) {
      auto next = VisitStmt(original);
      changed |= next != original;
      children.push_back(std::move(next));
    }
    return changed ? SeqStmts::Flatten(children, statement->span_) : statement;
  }
  StmtPtr VisitStmt_(const ForStmtPtr& statement) override {
    const auto body = VisitStmt(statement->body_);
    if (body == statement->body_) return statement;
    const auto result = std::make_shared<ForStmt>(*statement);
    result->body_ = body;
    return result;
  }
  StmtPtr VisitStmt_(const WhileStmtPtr& statement) override {
    const auto body = VisitStmt(statement->body_);
    if (body == statement->body_) return statement;
    const auto result = std::make_shared<WhileStmt>(*statement);
    result->body_ = body;
    return result;
  }
  StmtPtr VisitStmt_(const IfStmtPtr& statement) override {
    const auto then_body = VisitStmt(statement->then_body_);
    auto else_body = statement->else_body_;
    if (else_body) else_body = VisitStmt(*else_body);
    if (then_body == statement->then_body_ && else_body == statement->else_body_) return statement;
    const auto result = std::make_shared<IfStmt>(*statement);
    result->then_body_ = then_body;
    result->else_body_ = std::move(else_body);
    return result;
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& assignment) override {
    const auto call = As<Call>(assignment->value_);
    if (IsOp(call, "tensor.write")) return Write(call, assignment->var_, assignment);
    if (!IsOp(call, "tensor.read")) {
      return assignment;
    }
    const auto source = ScalarGMOperand(call, true);
    if (!source) {
      return assignment;
    }
    const auto scalar = As<ScalarType>(assignment->var_->GetType());
    const auto read_type = As<ScalarType>(call->GetType());
    if (!scalar || !read_type || scalar->dtype_ != source->dtype_ || read_type->dtype_ != source->dtype_) {
      return assignment;
    }
    const auto span = assignment->span_;
    const auto ci = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INDEX, span); };
    const auto tuple = [&](std::vector<ExprPtr> values) { return std::make_shared<MakeTuple>(values, span); };
    const auto one = ci(1), eight = ci(8);
    const auto valid_source = backend::MakeBufferScalarGMExtent(*source, span);
    const auto capacity = source->shape_.size() == 1 ? tuple({eight}) : tuple({one, eight});
    const auto [carrier, memory] = NewCarrier(one, eight, source->dtype_, span);
    const auto type = carrier->GetType();
    // Flattened Tile loads already preserve source-rank window arguments with
    // an independently typed two-dimensional native destination.
    const auto load = std::make_shared<Call>(
        OpRegistry::GetInstance().GetOp("tile.load"),
        std::vector<ExprPtr>{call->args_[0], call->args_[1], capacity, valid_source},
        std::vector<std::pair<std::string, std::any>>{{"target_memory", MemorySpace::Vec}}, type, span);
    if (!backend::SupportsLogicalBufferScalarTransfer(load)) return assignment;
    allocations.push_back(CreateAllocStatement(memory, MemorySpace::Vec));
    const auto read =
        OpRegistry::GetInstance().Create("tile.read", {carrier, tuple({ci(0), ci(0)})}, {}, span);
    return SeqStmts::Flatten({std::make_shared<AssignStmt>(carrier, load, span),
                              std::make_shared<AssignStmt>(assignment->var_, read, span)},
                             span);
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    return IsOp(call, "tensor.write") ? Write(call, nullptr, statement) : statement;
  }

 private:
  TensorTypePtr ScalarGMOperand(const CallPtr& call, bool read) {
    if (!call || call->args_.size() != (read ? 2 : 3) || !AsVarLike(call->args_[0]) ||
        !call->kwargs_.empty() || !call->attrs_.empty()) {
      return nullptr;
    }
    const auto tensor = As<TensorType>(call->args_[0]->GetType());
    const auto indices = As<MakeTuple>(call->args_[1]);
    if (!backend::IsBufferScalarGMSource(tensor) || !indices ||
        indices->elements_.size() != tensor->shape_.size()) {
      return nullptr;
    }
    // Keep effects at their original statement cuts. A read index can be
    // ordinary scalar arithmetic. Writes retain their normalized index form.
    for (const auto& index : indices->elements_) {
      if (read ? !PureScalarValue(index) : !(AsVarLike(index) || As<ConstInt>(index))) {
        return nullptr;
      }
    }
    return tensor;
  }

  bool PureScalarValue(const ExprPtr& expression) {
    if (!expression) return false;
    std::vector<std::pair<ExprPtr, bool>> pending{{expression, false}};
    while (!pending.empty()) {
      auto [value, finish] = pending.back();
      pending.pop_back();
      if (!value) continue;
      if (finish) {
        const auto child_is_pure = [this](const ExprPtr& child) {
          return child && pure_values_.at(child.get());
        };
        if (const auto binary = As<BinaryExpr>(value)) {
          pure_values_[value.get()] = child_is_pure(binary->left_) && child_is_pure(binary->right_);
        } else if (const auto unary = As<UnaryExpr>(value)) {
          pure_values_[value.get()] = child_is_pure(unary->operand_);
        }
        continue;
      }
      if (!pure_values_.emplace(value.get(), false).second || !As<ScalarType>(value->GetType())) continue;
      if (AsVarLike(value) || As<ConstInt>(value) || As<ConstFloat>(value)) {
        // A carry is its current SSA value, never an invitation to inspect or
        // replay its initializer. Lexical readiness remains a separate check.
        pure_values_[value.get()] = true;
      } else if (const auto binary = As<BinaryExpr>(value)) {
        pending.emplace_back(value, true);
        pending.emplace_back(binary->right_, false);
        pending.emplace_back(binary->left_, false);
      } else if (const auto unary = As<UnaryExpr>(value)) {
        // Cast is a UnaryExpr. Preserve every original narrowing and operation;
        // this is syntactic purity, not a range proof or algebraic rewrite.
        pending.emplace_back(value, true);
        pending.emplace_back(unary->operand_, false);
      }
      // Calls, submits, aggregates and all unknown forms remain false, even
      // when hidden beneath arithmetic. Cache both answers for shared DAGs.
    }
    return pure_values_.at(expression.get());
  }

  static bool SameTensorAlias(const TypePtr& candidate, const TypePtr& destination) {
    const auto tensor = As<TensorType>(candidate);
    const auto target = As<TensorType>(destination);
    if (!tensor || !target ||
        !structural_equal(CloneTypeWithoutMemRef(tensor), CloneTypeWithoutMemRef(target)))
      return false;
    const auto memory = tensor->memref_.value_or(nullptr);
    const auto target_memory = target->memref_.value_or(nullptr);
    // InitMemRef binds variables, while a registry-created Call result may
    // retain its unplaced logical type. Only absent placement is completed by
    // the registered destination alias; an explicit foreign window is refused.
    return !memory || (target_memory && memory->base_ == target_memory->base_ &&
                       structural_equal(memory, target_memory));
  }

  std::pair<VarPtr, MemRefPtr> NewCarrier(const ExprPtr& one, const ExprPtr& eight, DataType dtype,
                                          const Span& span) {
    const auto base = std::make_shared<Var>(Fresh("gm_scalar_base_"), GetPtrType(), span);
    const auto memory = std::make_shared<MemRef>(base, 0, backend::kBufferScalarCarrierBytes, span);
    TileView view;
    view.valid_shape = {one, one};
    const auto type =
        std::make_shared<TileType>(std::vector<ExprPtr>{one, eight}, dtype, memory, view, MemorySpace::Vec);
    return {std::make_shared<Var>(Fresh("gm_scalar_carrier_"), type, span), memory};
  }

  StmtPtr Write(const CallPtr& call, const VarPtr& result, const StmtPtr& original) {
    const auto target = ScalarGMOperand(call, false);
    if (!target || !call->args_[2]) {
      return original;
    }
    const auto scalar = As<ScalarType>(call->args_[2]->GetType());
    const auto alias = ResultAliasedArgIndex(call);
    if (!scalar || scalar->dtype_ != target->dtype_ || !alias || *alias != 0 ||
        !SameTensorAlias(call->GetType(), call->args_[0]->GetType()) ||
        (result && !SameTensorAlias(result->GetType(), call->args_[0]->GetType())))
      return original;
    if (!PureScalarValue(call->args_[2])) {
      return original;
    }
    const auto span = original->span_;
    const auto ci = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INDEX, span); };
    const auto tuple = [&](std::vector<ExprPtr> values) { return std::make_shared<MakeTuple>(values, span); };
    const auto one = ci(1), eight = ci(8);
    const auto [carrier, memory] = NewCarrier(one, eight, target->dtype_, span);
    const auto type = carrier->GetType();
    const auto store = std::make_shared<Call>(OpRegistry::GetInstance().GetOp("tile.store"),
                                              std::vector<ExprPtr>{carrier, call->args_[1], call->args_[0]},
                                              std::vector<std::pair<std::string, std::any>>{},
                                              call->args_[0]->GetType(), span);
    if (!backend::SupportsLogicalBufferScalarTransfer(store)) return original;
    // The descriptor allocates no values. SetValue initializes exactly the one
    // valid element before MTE3 reads it; inactive carrier lanes never escape.
    const auto create = std::make_shared<Call>(
        OpRegistry::GetInstance().GetOp("tile.create"), std::vector<ExprPtr>{tuple({one, eight})},
        std::vector<std::pair<std::string, std::any>>{{"dtype", target->dtype_},
                                                      {"target_memory", MemorySpace::Vec}},
        type, span);
    const auto write = OpRegistry::GetInstance().Create(
        "tile.write", {carrier, tuple({ci(0), ci(0)}), call->args_[2]}, {}, span);
    allocations.push_back(CreateAllocStatement(memory, MemorySpace::Vec));
    // Store-result lowering already preserves the original destination alias.
    // Both statement forms retain the same executed cut, value and index.
    const StmtPtr transfer = result ? StmtPtr(std::make_shared<AssignStmt>(result, store, span))
                                    : StmtPtr(std::make_shared<EvalStmt>(store, span));
    return SeqStmts::Flatten({std::make_shared<AssignStmt>(carrier, create, span),
                              std::make_shared<EvalStmt>(write, span), transfer},
                             span);
  }
  std::set<std::string> names_;
  std::unordered_map<const Expr*, bool> pure_values_;
  size_t next_name_ = 0;
  std::string Fresh(const char* prefix) {
    std::string result;
    do {
      result = std::string(prefix) + std::to_string(next_name_++);
    } while (!names_.insert(result).second);
    return result;
  }
};
}  // namespace

FunctionPtr StageBufferScalarAccesses(const FunctionPtr& original) {
  Names names;
  names.VisitFunction(original);
  Stage stage(std::move(names.names));
  const auto body = stage.VisitStmt(original->body_);
  if (stage.allocations.empty()) return original;
  const auto candidate = std::make_shared<Function>(*original);
  candidate->body_ = InsertAllocsIntoBody(body, stage.allocations);
  return candidate;
}
}  // namespace pypto::ir
