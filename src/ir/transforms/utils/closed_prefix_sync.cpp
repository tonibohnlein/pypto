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
#include "src/ir/transforms/utils/closed_prefix_sync.h"

#include <array>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"

namespace pypto::ir::sync {
namespace {
PipeType Pipe(const CallPtr& operation, const char* key) {
  return static_cast<PipeType>(operation->GetKwarg<int>(key));
}
CallPtr Record(const StmtPtr& statement) {
  const auto assign = As<AssignStmt>(statement);
  const auto call = assign ? As<Call>(assign->value_) : nullptr;
  return IsOp(call, "sync.record") ? call : nullptr;
}
CallPtr Wait(const StmtPtr& statement) {
  const auto eval = As<EvalStmt>(statement);
  const auto call = eval ? As<Call>(eval->expr_) : nullptr;
  return IsOp(call, "sync.wait") ? call : nullptr;
}
struct Transaction {
  std::array<StmtPtr, 4> cuts;
  PipeType source, destination;
};
struct LocalCompletion {
  std::array<StmtPtr, 2> cuts;
  PipeType pipe;
};

// The protocol grammar, not storage effects, establishes progress. Every
// transaction is closed on one common control path, and each enclosing
// sequence/branch/loop composes closed transactions. No historical flag/token
// can escape its visit. While conditions may not contain commands; the final
// false evaluation therefore creates no outstanding protocol state.
class Discover : public IRVisitor {
 public:
  std::vector<Transaction> transactions;
  std::vector<LocalCompletion> local;
  size_t logical_calls = 0;

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const SeqStmtsPtr& sequence) override {
    for (size_t i = 0; i < sequence->stmts_.size();) {
      const auto& statements = sequence->stmts_;
      if (i + 3 < statements.size() && Record(statements[i]) && Wait(statements[i + 1]) &&
          Record(statements[i + 2]) && Wait(statements[i + 3])) {
        const auto publish = As<AssignStmt>(statements[i]);
        const auto acknowledge = As<AssignStmt>(statements[i + 2]);
        const auto forward = Wait(statements[i + 1]);
        const auto returned = Wait(statements[i + 3]);
        const auto source = Pipe(Record(statements[i]), "pipe");
        const auto destination = Pipe(forward, "dst_pipe");
        if (source != destination && forward->args_[0] == publish->var_ &&
            returned->args_[0] == acknowledge->var_ &&
            Pipe(Record(statements[i + 2]), "pipe") == destination && Pipe(returned, "dst_pipe") == source) {
          transactions.push_back({{statements[i], statements[i + 1], statements[i + 2], statements[i + 3]},
                                  source,
                                  destination});
          i += 4;
          logical_calls += 4;
          continue;
        }
      }
      if (i + 1 < statements.size() && Record(statements[i]) && Wait(statements[i + 1])) {
        const auto publish = As<AssignStmt>(statements[i]);
        const auto wait = Wait(statements[i + 1]);
        const auto pipe = Pipe(Record(statements[i]), "pipe");
        if (wait->args_[0] == publish->var_ && Pipe(wait, "dst_pipe") == pipe) {
          local.push_back({{statements[i], statements[i + 1]}, pipe});
          i += 2;
          logical_calls += 2;
          continue;
        }
      }
      VisitStmt(statements[i++]);
    }
  }
  void VisitExpr_(const CallPtr& call) override {
    if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None) {
      ++logical_calls;
      IRVisitor::VisitExpr_(call);
    }
  }
};

// Reinspect every operation independently of loop counts and of the memory
// dependencies used to insert tokens. Unrecognized native effects are never
// treated as resource-free simply because a logical transaction is closed.
class NativeOperations : public IRVisitor {
 public:
  NativeOperations(FunctionType core, std::string target) : core_(core), target_(std::move(target)) {
    CHECK(core == FunctionType::AIC || core == FunctionType::AIV)
        << "Closed prefix protocols require one AIC/AIV invocation owner";
  }
  std::vector<CallPtr> payloads;
  void VisitExpr(const ExprPtr& expression) override {
    // Qualification is independent of visit/value context. Cache static DAG
    // nodes; repeated SSA references must not rewalk their expression trees.
    if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
  }
  void VisitStmt(const StmtPtr& statement) override {
    CHECK(As<SeqStmts>(statement) || As<AssignStmt>(statement) || As<EvalStmt>(statement) ||
          As<IfStmt>(statement) || As<ForStmt>(statement) || As<WhileStmt>(statement) ||
          As<YieldStmt>(statement) || As<ReturnStmt>(statement))
        << "Closed prefix protocols require local structured sequential control";
    IRVisitor::VisitStmt(statement);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const ForStmtPtr& loop) override {
    CHECK(loop->kind_ == ForKind::Sequential)
        << "Closed prefix reuse requires a sequential loop after scheduling";
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitExpr_(const SubmitPtr&) override {
    CHECK(false) << "Closed prefix protocols require an intra-leaf execution owner";
  }
  void VisitExpr_(const CallPtr& call) override {
    if (GetLogicalSyncKind(call->op_) == LogicalSyncKind::None) {
      CHECK(call->attrs_.empty())
          << "Closed prefix operation attributes require an explicit native execution contract";
      if (const auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
        CHECK(descriptor->SignalFree()) << "Closed prefix descriptor has unqualified event resources";
      } else {
        backend::QualifyBufferSyncOperation(call, core_, target_);
        payloads.push_back(call);
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  FunctionType core_;
  std::string target_;
  std::set<const Expr*> seen_;
};

// A role is the forward (false) or return (true) use of a directed namespace.
// Oppositely oriented services use distinct roles in the same namespace. Every
// reuse of one role therefore belongs to the SAME oriented transaction pair.
using Role = std::pair<SyncDirection, bool>;
enum class CommandKind { Publish, Acquire, Barrier, Erase };
struct AssignedCommand {
  CommandKind kind;
  SyncDirection direction;
  int id;
  const char* barrier = nullptr;
};

const char* LocalBarrier(PipeType pipe, FunctionType core) {
  // These are existing target operations, not inferred same-pipe payload order.
  // The contiguous record/wait pair denotes exactly the current pipe prefix.
  if (pipe == PipeType::S) return nullptr;
  const bool vector = core == FunctionType::AIV;
  switch (pipe) {
    case PipeType::MTE2:
      return "system.bar_mte2";
    case PipeType::MTE3:
      CHECK(vector) << "MTE3 completion is unavailable on this synchronization core";
      return "system.bar_mte3";
    case PipeType::V:
      CHECK(vector) << "Vector completion is unavailable on this synchronization core";
      return "system.bar_v";
    case PipeType::MTE1:
      CHECK(!vector) << "MTE1 completion is unavailable on this synchronization core";
      return "system.bar_mte1";
    case PipeType::M:
      CHECK(!vector) << "Cube completion is unavailable on this synchronization core";
      return "system.bar_m";
    case PipeType::FIX:
      CHECK(!vector) << "FIX completion is unavailable on this synchronization core";
      return "system.bar_fix";
    default:
      CHECK(false) << "Unavailable current-prefix completion pipe";
  }
  return nullptr;
}

// Factory-local immutable assignment: the exact original cut inventory and
// native reservations are fixed before constructing any output statement.
// Protocol proof: Fcons(v)->Apub(v)->Acons(v)->Fpub(v+1) through q then p
// command order. Also Fcons(v+1)->Apub(v+1), so BOTH roles are rearmed. A
// skipped branch/zero-trip loop is the identity; induction applies to arbitrary
// visit counts without decoding, ranking or unfolding occurrence coordinates.
class Assignment {
 public:
  Assignment(const FunctionPtr& owner, const Discover& discovered, const std::vector<CallPtr>& payloads,
             const std::string& target, const std::optional<std::vector<int>>& eligible) {
    const auto& native = backend::RequireLocalSyncExecutionContract(target);
    CHECK(native.set_completes_source_prefix && native.wait_consumes_before_advance)
        << "Closed prefix protocols require ordered consuming event commands";
    reservations_ = SyncNativeClaims(owner, payloads, target);
    std::map<SyncDirection, std::set<bool>> needs;
    for (const auto& transaction : discovered.transactions) {
      needs[{transaction.source, transaction.destination}].insert(false);
      needs[{transaction.destination, transaction.source}].insert(true);
    }
    std::map<Role, int> assignment;
    for (const auto& [direction, roles] : needs) {
      const auto ids = SyncEligibleIds(owner, target, direction, eligible, reservations_);
      CHECK(ids.size() >= roles.size())
          << "Closed prefix role reservation unavailable; this is not a fixed-plan capacity lower bound";
      size_t next = 0;
      for (const bool role : roles) assignment.emplace(Role{direction, role}, ids[next++]);
    }
    roles_ = assignment.size();
    for (const auto& transaction : discovered.transactions) {
      const SyncDirection forward{transaction.source, transaction.destination};
      const SyncDirection backward{transaction.destination, transaction.source};
      for (size_t i = 0; i < transaction.cuts.size(); ++i) {
        const auto direction = i < 2 ? forward : backward;
        const int id = assignment.at({direction, i >= 2});
        CHECK(commands_
                  .emplace(transaction.cuts[i].get(),
                           AssignedCommand{i % 2 == 0 ? CommandKind::Publish : CommandKind::Acquire,
                                           direction, id})
                  .second)
            << "Closed prefix protocol reuses an original static command cut";
      }
    }
    for (const auto& local : discovered.local) {
      const auto* barrier = LocalBarrier(local.pipe, owner->func_type_);
      CHECK(local.pipe != PipeType::S || native.scalar_prefix_serialized)
          << "Scalar current-prefix completion lacks native order evidence";
      CHECK(commands_.emplace(local.cuts[0].get(), AssignedCommand{CommandKind::Erase, {}, -1}).second)
          << "Local completion reuses an original static cut";
      CHECK(commands_
                .emplace(local.cuts[1].get(),
                         AssignedCommand{barrier ? CommandKind::Barrier : CommandKind::Erase,
                                         {local.pipe, local.pipe},
                                         -1,
                                         barrier})
                .second)
          << "Local completion reuses an original static cut";
    }
  }
  const std::map<const Stmt*, AssignedCommand>& Commands() const { return commands_; }
  const SyncNativeReservations& Reservations() const { return reservations_; }
  size_t Roles() const { return roles_; }

 private:
  std::map<const Stmt*, AssignedCommand> commands_;
  SyncNativeReservations reservations_;
  size_t roles_ = 0;
};

class Emit : public IRMutator {
 public:
  Emit(const Assignment& assignment, FunctionType core)
      : assignment_(assignment), core_(core == FunctionType::AIV ? "aiv" : "aic") {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    return RewriteSyncSequence(sequence, [this](const StmtPtr& statement) {
      const auto command = assignment_.Commands().find(statement.get());
      if (command == assignment_.Commands().end()) return VisitStmt(statement);
      const auto& value = command->second;
      if (value.kind == CommandKind::Erase)
        return StmtPtr(std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, statement->span_));
      if (value.kind == CommandKind::Barrier) {
        const auto operation =
            OpRegistry::GetInstance().Create(value.barrier, {}, {{"core_type", core_}}, statement->span_);
        return StmtPtr(std::make_shared<EvalStmt>(operation, statement->span_));
      }
      const auto operation = OpRegistry::GetInstance().Create(
          value.kind == CommandKind::Publish ? "system.set_flag" : "system.wait_flag", {},
          {{"src_pipe", static_cast<int>(value.direction.first)},
           {"dst_pipe", static_cast<int>(value.direction.second)},
           {"event_id", value.id},
           {"core_type", core_}},
          statement->span_);
      return StmtPtr(std::make_shared<EvalStmt>(operation, statement->span_));
    });
  }
  ExprPtr VisitExpr(const ExprPtr& expression) override { return expression; }

 private:
  const Assignment& assignment_;
  std::string core_;
};
}  // namespace

bool HasOnlyClosedPrefixSyntax(const FunctionPtr& function) {
  Discover discovered;
  discovered.VisitFunction(function);
  return (!discovered.transactions.empty() || !discovered.local.empty()) &&
         discovered.logical_calls == 4 * discovered.transactions.size() + 2 * discovered.local.size();
}

std::optional<PhysicalSyncRealization> TryLowerClosedPrefixSync(
    const FunctionPtr& function, const std::string& target,
    const std::optional<std::vector<int>>& eligible_ids) {
  Discover discovered;
  discovered.VisitFunction(function);
  if ((discovered.transactions.empty() && discovered.local.empty()) ||
      discovered.logical_calls != 4 * discovered.transactions.size() + 2 * discovered.local.size())
    return std::nullopt;
  CHECK(!function->requires_runtime_binding_ && !function->HasAttr(kAttrExternalSource) &&
        !function->HasAttr(kAttrBuiltinTemplateDir))
      << "Closed prefix protocols cannot replace an external invocation owner";
  // Check actual SSA uses too: counting instructions alone would miss fanout or
  // token aliases escaping the four recognized cuts. The ordinary verifier
  // checks ownership; the following visitor restricts uses to the two WAITs.
  std::set<const Var*> local_tokens;
  for (const auto& transaction : discovered.transactions)
    for (const size_t index : {size_t{0}, size_t{2}})
      local_tokens.insert(As<AssignStmt>(transaction.cuts[index])->var_.get());
  for (const auto& local : discovered.local) local_tokens.insert(As<AssignStmt>(local.cuts[0])->var_.get());
  class TokenUses : public IRVisitor {
   public:
    explicit TokenUses(const std::set<const Var*>& tokens) : tokens_(tokens) {}
    size_t uses = 0;

   protected:
    void VisitStmt_(const AssignStmtPtr& assignment) override { VisitExpr(assignment->value_); }
    void VisitVarLike_(const VarPtr& variable) override {
      if (tokens_.count(variable.get())) ++uses;
    }

   private:
    const std::set<const Var*>& tokens_;
  } token_uses(local_tokens);
  token_uses.VisitFunction(function);
  CHECK(token_uses.uses == 2 * discovered.transactions.size() + discovered.local.size())
      << "Closed prefix tokens require one within-visit consumer and no escapes";
  NativeOperations native(function->func_type_, target);
  native.VisitFunction(function);
  const Assignment assignment(function, discovered, native.payloads, target, eligible_ids);
  PhysicalSyncRealization result;
  result.function = Emit(assignment, function->func_type_).VisitFunction(function);
  result.set_sites = result.wait_sites = 2 * discovered.transactions.size();
  result.reserved_lanes = assignment.Roles();
  for (const auto& [cut, command] : assignment.Commands())
    result.barrier_sites += command.kind == CommandKind::Barrier;
  result.finite_minimum = false;
  for (const auto& [direction, ids] : assignment.Reservations())
    for (const int id : ids) result.native_reserved_ids.emplace_back(direction.first, direction.second, id);
  return result;
}
}  // namespace pypto::ir::sync
