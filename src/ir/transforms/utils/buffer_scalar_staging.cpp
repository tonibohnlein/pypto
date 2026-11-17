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
#include "pypto/ir/transforms/utils/memref_utils.h"
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
    if (!As<AssignStmt>(statement) && !As<SeqStmts>(statement) && !As<ForStmt>(statement) &&
        !As<WhileStmt>(statement) && !As<IfStmt>(statement))
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
    if (!IsOp(call, "tensor.read") || call->args_.size() != 2 || !call->args_[0] || !call->args_[1] ||
        !call->kwargs_.empty() || !call->attrs_.empty() ||
        !backend::IsBufferScalarGMSource(As<TensorType>(call->args_[0]->GetType())))
      return assignment;
    const auto scalar = As<ScalarType>(assignment->var_->GetType());
    if (!scalar || scalar->dtype_ != DataType::INT32) return assignment;
    const auto span = assignment->span_;
    const auto ci = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INDEX, span); };
    const auto tuple = [&](std::vector<ExprPtr> values) { return std::make_shared<MakeTuple>(values, span); };
    const auto one = ci(1), eight = ci(8);
    const auto valid_source = tuple({one});
    const auto base = std::make_shared<Var>(Fresh("gm_scalar_base_"), GetPtrType(), span);
    const auto memory = std::make_shared<MemRef>(base, 0, backend::kBufferScalarCarrierBytes, span);
    TileView view;
    view.valid_shape = {one, one};
    const auto type = std::make_shared<TileType>(std::vector<ExprPtr>{one, eight}, DataType::INT32, memory,
                                                 view, MemorySpace::Vec);
    const auto carrier = std::make_shared<Var>(Fresh("gm_scalar_carrier_"), type, span);
    // Flattened Tile loads already preserve source-rank window arguments with
    // an independently typed two-dimensional native destination.
    const auto load = std::make_shared<Call>(
        OpRegistry::GetInstance().GetOp("tile.load"),
        std::vector<ExprPtr>{call->args_[0], call->args_[1], tuple({eight}), valid_source},
        std::vector<std::pair<std::string, std::any>>{{"target_memory", MemorySpace::Vec}}, type, span);
    if (!backend::SupportsLogicalBufferScalarTransfer(load)) return assignment;
    allocations.push_back(CreateAllocStatement(memory, MemorySpace::Vec));
    const auto read =
        OpRegistry::GetInstance().Create("tile.read", {carrier, tuple({ci(0), ci(0)})}, {}, span);
    return SeqStmts::Flatten({std::make_shared<AssignStmt>(carrier, load, span),
                              std::make_shared<AssignStmt>(assignment->var_, read, span)},
                             span);
  }

 private:
  std::set<std::string> names_;
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

FunctionPtr StageBufferScalarReads(const FunctionPtr& original) {
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
