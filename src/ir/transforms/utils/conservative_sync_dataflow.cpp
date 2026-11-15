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

#include "src/ir/transforms/utils/conservative_sync_dataflow.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto::ir::sync {
namespace {
// ParentStmtAnalysis owns the shared statement index. Expressions cannot carry
// admitted statements; skipping them avoids retraversing IterArg initializer
// DAGs merely to establish statement ownership.
class StatementParents : public ParentStmtAnalysis {
 public:
  void VisitExpr(const ExprPtr&) override {}
};

void Union(ConservativeAccessSet& into, const ConservativeAccessSet& from) {
  into.insert(from.begin(), from.end());
}

MemorySpace Space(const ExprPtr& operand) {
  if (As<TensorType>(operand->GetType())) return MemorySpace::DDR;
  if (const auto buffer = As<BufferType>(operand->GetType())) return buffer->memory_space_;
  CHECK_SPAN(false, operand->span_)
      << "Conservative synchronization: data operand has no qualified storage space";
  return MemorySpace::DDR;
}

struct End {
  std::optional<size_t> yield;
  bool returned = false;
  bool Terminal() const { return yield.has_value() || returned; }
};

class Builder {
 public:
  Builder(const FunctionPtr& function, std::string target,
          const std::vector<std::shared_ptr<const NativeSyncChild>>& retained)
      : function_(function), target_(std::move(target)) {
    scopes_.emplace_back();
    for (const auto& parameter : function->params_) {
      const auto& type = parameter->GetType();
      CHECK_SPAN(
          As<ScalarType>(type) || As<TensorType>(type) || As<BufferType>(type) || As<MultiBufferType>(type),
          parameter->span_)
          << "Conservative synchronization: parameter requires an invocation ownership/readiness contract";
      Bind(parameter);
    }
    for (const auto& child : retained) {
      CHECK(child) << "Conservative synchronization: missing retained child certificate";
      child->ValidateFor(function, target_);
      CHECK(child_index_.emplace(child->Scope().get(), children.size()).second)
          << "Conservative synchronization: duplicate retained child scope";
      children.push_back({child->Scope(), child, {}, {}, {}});
    }
  }

  void Build() {
    Read(function_->body_);
    for (auto& child : children) {
      const auto found = generated_.find(child.scope.get());
      CHECK(found != generated_.end()) << "Retained child is outside the original semantic input";
      child.effects = found->second;
    }
  }

  std::vector<ConservativeSyncSite> PrimitiveSites() {
    std::vector<ConservativeSyncSite> result;
    for (size_t index = 0; index < sites.size(); ++index)
      if (!inside_child_[index]) result.push_back(std::move(sites[index]));
    return result;
  }

  ConservativeAccessSet Flow(const StmtPtr& statement, ConservativeAccessSet incoming) {
    ++cost.dataflow_visits;
    if (const auto found = child_index_.find(statement.get()); found != child_index_.end()) {
      auto& entry = children[found->second];
      entry.incoming = incoming;
      std::set<std::pair<PipeType, PipeType>> services;
      for (const auto& [space, source, write] : incoming)
        for (const auto& [consumer_space, destination, consumer_write] : entry.effects)
          if (space == consumer_space && (write || consumer_write)) services.emplace(source, destination);
      entry.services.assign(services.begin(), services.end());
      cost.services += services.size();
      cost.retained_facts += incoming.size() + entry.effects.size();
      // Empty notification state does not complete any payload. All child
      // effects remain possibly pending, including across zero/skipped visits.
      Union(incoming, entry.effects);
      return incoming;
    }
    if (const auto seq = As<SeqStmts>(statement)) {
      for (const auto& child : seq->stmts_) incoming = Flow(child, std::move(incoming));
    } else if (const auto branch = As<IfStmt>(statement)) {
      auto then_exit = Flow(branch->then_body_, incoming);
      if (branch->else_body_)
        Union(then_exit, Flow(*branch->else_body_, incoming));
      else
        Union(then_exit, incoming);
      incoming = std::move(then_exit);
    } else if (const auto loop = As<ForStmt>(statement)) {
      Union(incoming, generated_.at(loop->body_.get()));
      incoming = Flow(loop->body_, std::move(incoming));
    } else if (const auto loop = As<WhileStmt>(statement)) {
      Union(incoming, generated_.at(loop->body_.get()));
      incoming = Flow(loop->body_, std::move(incoming));
    } else if (const auto found = site_index_.find(statement.get()); found != site_index_.end()) {
      auto& site = sites[found->second];
      site.incoming = incoming;
      std::set<PipeType> sources;
      for (const auto& [space, source, write] : incoming) {
        for (const auto& [consumer_space, destination, consumer_write] : site.effects) {
          (void)destination;
          if (space == consumer_space && (write || consumer_write)) sources.insert(source);
        }
      }
      site.source_pipes.assign(sources.begin(), sources.end());
      cost.services += sources.size();
      cost.retained_facts += incoming.size() + site.effects.size();
      Union(incoming, site.effects);
    }
    return incoming;
  }

  std::vector<ConservativeSyncSite> sites;
  std::vector<ConservativeSyncChildEntry> children;
  ConservativeSyncCost cost;

 private:
  void Bind(const VarPtr& variable) {
    CHECK_SPAN(ready_.insert(variable.get()).second, variable->span_)
        << "Conservative synchronization: a ready SSA definition is redefined";
    scopes_.back().push_back(variable.get());
  }
  void PopScope() {
    for (const auto* variable : scopes_.back()) ready_.erase(variable);
    scopes_.pop_back();
  }

  void Ready(const ExprPtr& expression) {
    std::unordered_set<const Expr*> visited;
    Ready(expression, visited);
  }
  void Ready(const ExprPtr& expression, std::unordered_set<const Expr*>& visited) {
    CHECK(expression) << "Conservative synchronization: missing expression";
    if (!visited.insert(expression.get()).second) return;
    ++cost.expression_visits;
    if (const auto variable = AsVarLike(expression)) {
      if (ready_.count(variable.get())) return;
      CHECK_SPAN(false, expression->span_)
          << "Conservative synchronization: scalar or descriptor use lacks a ready definition";
    } else if (As<ConstInt>(expression) || As<ConstFloat>(expression) || As<ConstBool>(expression)) {
      return;
    } else if (const auto binary = As<BinaryExpr>(expression)) {
      Ready(binary->left_, visited);
      Ready(binary->right_, visited);
      return;
    } else if (const auto unary = As<UnaryExpr>(expression)) {
      Ready(unary->operand_, visited);
      return;
    } else if (const auto tuple = As<MakeTuple>(expression)) {
      for (const auto& element : tuple->elements_) Ready(element, visited);
      return;
    } else if (const auto item = As<TupleGetItemExpr>(expression)) {
      Ready(item->tuple_, visited);
      return;
    }
    CHECK_SPAN(false, expression->span_)
        << "Conservative synchronization: control/metadata expression requires an explicit readiness "
           "and visibility contract (nested calls and scalar memory reads are unavailable)";
  }

  void Payload(const StmtPtr& cut, const CallPtr& call, const VarPtr& result) {
    CHECK_SPAN(call->attrs_.empty(), call->span_)
        << "Conservative synchronization: operation attributes require an execution contract";
    for (const auto& argument : call->args_) Ready(argument);
    if (auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
      const auto behavior = descriptor->Result().behavior;
      CHECK_SPAN(static_cast<bool>(result) == (behavior != BufferResultBehavior::None), call->span_)
          << "Conservative synchronization: descriptor result must have an original SSA binding";
      if (result) Bind(result);
      return;
    }
    auto operation = backend::QualifyBufferSyncOperation(call, function_->func_type_, target_);
    CHECK_SPAN(!result && operation.Result().behavior == BufferResultBehavior::None, call->span_)
        << "Conservative synchronization: payload-produced scalar/handle readiness is unavailable";
    ConservativeAccessSet effects;
    for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
      const auto effect = operation.ArgumentEffects()[index].data;
      if (effect == BufferAccess::None) continue;
      const auto space = Space(call->args_[index]);
      const auto pipe = operation.Implementation().Pipe();
      if (effect == BufferAccess::Read || effect == BufferAccess::ReadWrite)
        effects.emplace(space, pipe, false);
      if (effect == BufferAccess::Write || effect == BufferAccess::ReadWrite)
        effects.emplace(space, pipe, true);
    }
    for (const auto& effect : operation.Implementation().ImplicitEffects()) {
      const auto pipe = operation.Implementation().Pipe();
      if (effect.data == BufferAccess::Read || effect.data == BufferAccess::ReadWrite)
        effects.emplace(effect.space, pipe, false);
      if (effect.data == BufferAccess::Write || effect.data == BufferAccess::ReadWrite)
        effects.emplace(effect.space, pipe, true);
    }
    CHECK_SPAN(!effects.empty(), call->span_)
        << "Conservative synchronization: command-only payload requires a protocol contract";
    const auto inserted = site_index_.emplace(cut.get(), sites.size()).second;
    CHECK_SPAN(inserted, cut->span_)
        << "Conservative synchronization: shared statement identity has ambiguous insertion cuts";
    generated_[cut.get()] = effects;
    sites.push_back({cut, std::move(operation), std::move(effects), {}, {}});
    inside_child_.push_back(native_child_depth_ != 0);
  }

  End Read(const StmtPtr& statement) {
    ++cost.statements;
    generated_.try_emplace(statement.get());
    if (const auto seq = As<SeqStmts>(statement)) {
      End end;
      for (const auto& child : seq->stmts_) {
        CHECK_SPAN(!end.Terminal(), child->span_)
            << "Conservative synchronization: statements after a terminal cut are unavailable";
        end = Read(child);
        Union(generated_.at(statement.get()), generated_.at(child.get()));
      }
      return end;
    }
    if (const auto assignment = As<AssignStmt>(statement)) {
      if (const auto call = As<Call>(assignment->value_)) {
        Payload(statement, call, assignment->var_);
      } else {
        Ready(assignment->value_);
        Bind(assignment->var_);
      }
      return {};
    }
    if (const auto eval = As<EvalStmt>(statement)) {
      if (const auto call = As<Call>(eval->expr_))
        Payload(statement, call, nullptr);
      else
        Ready(eval->expr_);
      return {};
    }
    if (const auto branch = As<IfStmt>(statement)) {
      Ready(branch->condition_);
      ++control_depth_;
      scopes_.emplace_back();
      const auto then_end = Read(branch->then_body_);
      PopScope();
      End else_end;
      if (branch->else_body_) {
        scopes_.emplace_back();
        else_end = Read(*branch->else_body_);
        PopScope();
        Union(generated_.at(statement.get()), generated_.at(branch->else_body_->get()));
      }
      --control_depth_;
      CHECK_SPAN(!then_end.returned && !else_end.returned, statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      const auto count = branch->return_vars_.size();
      CHECK_SPAN(then_end.yield.value_or(0) == count && else_end.yield.value_or(0) == count, statement->span_)
          << "Conservative synchronization: branch result/yield arity mismatch";
      for (const auto& variable : branch->return_vars_) Bind(variable);
      Union(generated_.at(statement.get()), generated_.at(branch->then_body_.get()));
      return {};
    }
    if (const auto loop = As<ForStmt>(statement)) {
      CHECK_SPAN(loop->kind_ != ForKind::Parallel, statement->span_)
          << "Conservative synchronization: concurrent loop execution requires another domain contract";
      Ready(loop->start_);
      Ready(loop->stop_);
      Ready(loop->step_);
      ReadLoop(statement, loop->body_, loop->iter_args_, loop->return_vars_, loop->loop_var_, nullptr);
      return {};
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      ReadLoop(statement, loop->body_, loop->iter_args_, loop->return_vars_, nullptr, loop->condition_);
      return {};
    }
    if (const auto yield = As<YieldStmt>(statement)) {
      for (const auto& value : yield->value_) Ready(value);
      return {yield->value_.size(), false};
    }
    if (const auto ret = As<ReturnStmt>(statement)) {
      CHECK_SPAN(control_depth_ == 0, statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      for (const auto& value : ret->value_) Ready(value);
      return {std::nullopt, true};
    }
    CHECK_SPAN(false, statement->span_)
        << "Conservative synchronization: unsupported structured control or command statement: "
        << statement->TypeName();
    return {};
  }

  void ReadLoop(const StmtPtr& loop, const StmtPtr& body, const std::vector<IterArgPtr>& arguments,
                const std::vector<VarPtr>& results, const VarPtr& index, const ExprPtr& condition) {
    CHECK_SPAN(arguments.size() == results.size(), loop->span_)
        << "Conservative synchronization: loop result/initializer arity mismatch";
    for (const auto& argument : arguments) Ready(argument->initValue_);
    scopes_.emplace_back();
    if (index) Bind(index);
    for (const auto& argument : arguments) Bind(argument);
    if (condition) Ready(condition);
    const bool child = child_index_.count(loop.get()) != 0;
    CHECK(!child || native_child_depth_ == 0)
        << "Conservative synchronization: retained child scopes must not overlap";
    if (child) ++native_child_depth_;
    ++control_depth_;
    const auto end = Read(body);
    --control_depth_;
    if (child) --native_child_depth_;
    CHECK_SPAN(!end.returned && end.yield.value_or(0) == arguments.size(), body->span_)
        << "Conservative synchronization: loop backedge lacks matching ready yields";
    PopScope();
    for (const auto& result : results) Bind(result);
    Union(generated_.at(loop.get()), generated_.at(body.get()));
  }

  FunctionPtr function_;
  std::string target_;
  std::vector<std::vector<const Var*>> scopes_;
  std::unordered_set<const Var*> ready_;
  size_t control_depth_ = 0;
  size_t native_child_depth_ = 0;
  std::vector<bool> inside_child_;
  std::unordered_map<const Stmt*, size_t> child_index_;
  std::unordered_map<const Stmt*, ConservativeAccessSet> generated_;
  std::unordered_map<const Stmt*, size_t> site_index_;
};
}  // namespace

std::vector<StmtPtr> ConservativeSyncAnalysis::ControlPath(size_t site) const {
  CHECK(site < sites_.size()) << "Conservative synchronization: invalid diagnostic site index";
  std::vector<StmtPtr> path;
  for (auto parent = parents_->GetParent(sites_[site].cut); parent; parent = parents_->GetParent(parent)) {
    if (As<IfStmt>(parent) || As<ForStmt>(parent) || As<WhileStmt>(parent)) path.push_back(parent);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

ConservativeSyncAnalysis AnalyzeConservativeSync(
    const FunctionPtr& function, const std::string& target,
    const std::vector<std::shared_ptr<const NativeSyncChild>>& children) {
  CHECK(function && function->ir_stage_ == FunctionIRStage::Buffer)
      << "Conservative synchronization requires Buffer IR";
  CHECK(function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV)
      << "Conservative synchronization requires an owned local AIC/AIV domain";
  backend::RequireLocalSyncExecutionContract(target);
  Builder builder(function, target, children);
  builder.Build();
  auto exit = builder.Flow(function->body_, {});
  auto parents = std::make_shared<StatementParents>();
  parents->BuildMap(function);
  builder.cost.parent_links = builder.cost.statements ? builder.cost.statements - 1 : 0;
  return ConservativeSyncAnalysis(function, builder.PrimitiveSites(), std::move(exit), builder.cost,
                                  std::move(parents), std::move(builder.children));
}
}  // namespace pypto::ir::sync
