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

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/retained_sync_child.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

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
  std::vector<ForStmtPtr> children;
  std::map<const ForStmt*, std::set<SyncDirection>> entries;
  std::set<const Var*> completed_tokens;

  void VisitExpr(const ExprPtr& expression) override {
    // This is static syntax discovery, not execution. Recognized services are
    // counted at their statement cuts above; shared expression DAGs need only
    // one inspection, even when this recognizer ultimately declines them.
    if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& assignment) override {
    const auto call = As<Call>(assignment->value_);
    const auto alias = As<Var>(assignment->value_);
    if (As<SyncTokenType>(assignment->var_->GetType()) &&
        (IsOp(call, "sync.completed") || (alias && completed_tokens.count(alias.get())))) {
      // Logical verification establishes dominance. These are pure aliases of
      // empty facts, never publications or evidence of payload completion.
      completed_tokens.insert(assignment->var_.get());
    }
    IRVisitor::VisitStmt_(assignment);
  }
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loop->HasAttr(kClosedSyncChildAttr)) {
      CHECK(loop->GetAttr<int>(kClosedSyncChildAttr) == 1) << "Unsupported closed-child routing version";
      children.push_back(loop);
      return;  // Actual child tokens are independently parsed below.
    }
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const SeqStmtsPtr& sequence) override {
    std::set<SyncDirection> preceding;
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
          preceding.emplace(source, destination);
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
      const auto child = As<ForStmt>(statements[i]);
      if (child && child->HasAttr(kClosedSyncChildAttr)) entries.emplace(child.get(), preceding);
      preceding.clear();
      VisitStmt(statements[i++]);
    }
  }
  void VisitExpr_(const CallPtr& call) override {
    const auto kind = GetLogicalSyncKind(call->op_);
    if (kind != LogicalSyncKind::None) {
      // Completed initializers carry no publication, but remain in the actual
      // child binding context until its token parser has interpreted them.
      if (kind != LogicalSyncKind::Completed) ++logical_calls;
      IRVisitor::VisitExpr_(call);
    }
  }

 private:
  std::set<const Expr*> seen_;
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

class Emit : public SyncStatementMutator {
 public:
  Emit(const Assignment& assignment, FunctionType core, const std::map<const ForStmt*, StmtPtr>& children,
       const std::set<const Var*>& completed_tokens)
      : assignment_(assignment),
        core_(core == FunctionType::AIV ? "aiv" : "aic"),
        children_(children),
        completed_tokens_(completed_tokens) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& assignment) override {
    if (completed_tokens_.count(assignment->var_.get()))
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, assignment->span_);
    return SyncStatementMutator::VisitStmt_(assignment);
  }
  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    const auto found = children_.find(loop.get());
    return found == children_.end() ? SyncStatementMutator::VisitStmt_(loop) : found->second;
  }
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

 private:
  const Assignment& assignment_;
  std::string core_;
  const std::map<const ForStmt*, StmtPtr>& children_;
  const std::set<const Var*>& completed_tokens_;
};
}  // namespace

bool QualifyLocalSyncInvocation(const FunctionPtr& function, const std::string& target) {
  backend::RequireLocalSyncExecutionContract(target);
  NativeOperations native(function->func_type_, target);
  native.VisitFunction(function);
  // Even a token-free plan may contain a native macro with hidden claims.
  // Resolve every selected variant instead of interpreting zero flags as an
  // independently established resource-empty implementation.
  (void)SyncNativeClaims(function, native.payloads, target);
  // Descriptor sentinels do not prove the selected native implementation's
  // runtime shape relations. Ownership must consume the same use-site proof as
  // Buffer code generation, including for token-free native-only programs.
  const BufferRuntimeValidAnalysis metadata(function);
  if (const auto& issue = metadata.issue()) CHECK_SPAN(false, issue->call->span_) << issue->message;
  return !native.payloads.empty();
}

bool HasOnlyClosedPrefixSyntax(const FunctionPtr& function) {
  Discover discovered;
  discovered.VisitFunction(function);
  return (!discovered.transactions.empty() || !discovered.local.empty() || !discovered.children.empty()) &&
         discovered.logical_calls == 4 * discovered.transactions.size() + 2 * discovered.local.size();
}

std::optional<PhysicalSyncRealization> TryLowerClosedPrefixSync(
    const FunctionPtr& function, const std::string& target,
    const std::optional<std::vector<int>>& eligible_ids) {
  Discover discovered;
  discovered.VisitFunction(function);
  if ((discovered.transactions.empty() && discovered.local.empty() && discovered.children.empty()) ||
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
    void VisitExpr(const ExprPtr& expression) override {
      if (!expression) return;
      const auto found = counts_.find(expression.get());
      if (found != counts_.end()) {
        uses += found->second;
        return;
      }
      // Summarize static expressions once, then charge that summary on EVERY
      // use. Unlike a seen-set, this still detects repeated command/alias uses
      // of one publication while avoiding repeated walks of scalar DAGs.
      const auto before = uses;
      IRVisitor::VisitExpr(expression);
      counts_.emplace(expression.get(), uses - before);
    }

   protected:
    void VisitStmt_(const AssignStmtPtr& assignment) override { VisitExpr(assignment->value_); }
    void VisitVarLike_(const VarPtr& variable) override {
      if (tokens_.count(variable.get())) ++uses;
    }

   private:
    const std::set<const Var*>& tokens_;
    std::map<const Expr*, size_t> counts_;
  } token_uses(local_tokens);
  token_uses.VisitFunction(function);
  CHECK(token_uses.uses == 2 * discovered.transactions.size() + discovered.local.size())
      << "Closed prefix tokens require one within-visit consumer and no escapes";
  // Token carries/signatures and aliases of publications need another parser.
  // Decline before emitting anything rather than leave residual token values.
  auto erased_tokens = local_tokens;
  erased_tokens.insert(discovered.completed_tokens.begin(), discovered.completed_tokens.end());
  class TokenShape : public IRVisitor {
   public:
    TokenShape(const std::set<const Var*>& erased, const std::vector<ForStmtPtr>& children)
        : erased_(erased), children_(children.begin(), children.end()) {}
    bool supported = true;
    void VisitExpr(const ExprPtr& expression) override {
      if (expression && seen_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
    }

   protected:
    void VisitVarLike_(const VarPtr& variable) override {
      if (As<SyncTokenType>(variable->GetType()) && !erased_.count(variable.get())) supported = false;
    }
    void VisitStmt_(const ForStmtPtr& loop) override {
      if (!children_.count(loop)) IRVisitor::VisitStmt_(loop);
    }

   private:
    const std::set<const Var*>& erased_;
    std::set<ForStmtPtr> children_;
    std::set<const Expr*> seen_;
  } shape(erased_tokens, discovered.children);
  shape.VisitFunction(function);
  if (!shape.supported) return std::nullopt;
  NativeOperations native(function->func_type_, target);
  native.VisitFunction(function);
  const auto reservations = SyncNativeClaims(function, native.payloads, target);
  std::map<SyncDirection, std::set<bool>> roles;
  for (const auto& transaction : discovered.transactions) {
    roles[{transaction.source, transaction.destination}].insert(false);
    roles[{transaction.destination, transaction.source}].insert(true);
  }
  std::set<int> parent_ids;
  for (const auto& [direction, uses] : roles) {
    const auto available = SyncEligibleIds(function, target, direction, eligible_ids, reservations);
    CHECK(available.size() >= uses.size())
        << "Closed parent role reservation unavailable; this is not a fixed-plan capacity lower bound";
    parent_ids.insert(available.begin(), available.begin() + uses.size());
  }
  std::map<const ForStmt*, StmtPtr> children;
  size_t child_sets = 0, child_waits = 0, child_barriers = 0, child_lanes = 0;
  const auto bindings = discovered.children.empty()
                            ? std::optional<SyncRegionBindings>{}
                            : std::optional<SyncRegionBindings>{std::in_place, function, target};
  for (const auto& child : discovered.children) {
    const auto external_policy = HasReadonlyExternalSyncEffects(child, function->func_type_, target)
                                     ? SyncExternalPolicy::ReadOnlyProjection
                                     : SyncExternalPolicy::Geometry;
    auto context = bindings->Bind(child, external_policy);
    CheckLogicalSync(context.binding_owner);
    NativeOperations child_native(function->func_type_, target);
    child_native.VisitFunction(context.binding_owner);
    std::set<PipeType> pipes;
    for (const auto& call : child_native.payloads)
      pipes.insert(
          backend::QualifyBufferSyncOperation(call, function->func_type_, target).Implementation().Pipe());
    // The actual program can use command-only pipes, including a recursively
    // recognized closed service. Payload pipes alone are not its footprint.
    // Logical verification above confines commands to statement value roots;
    // inspect those roots without repeatedly walking unrelated scalar DAGs.
    class CommandPipes : public IRVisitor {
     public:
      std::set<PipeType> participating;
      void VisitExpr(const ExprPtr& expression) override {
        const auto call = As<Call>(expression);
        if (!call) return;
        const auto kind = GetLogicalSyncKind(call->op_);
        if (kind == LogicalSyncKind::Record) participating.insert(Pipe(call, "pipe"));
        if (kind == LogicalSyncKind::Wait) participating.insert(Pipe(call, "dst_pipe"));
      }
    } command_pipes;
    command_pipes.VisitFunction(context.binding_owner);
    pipes.insert(command_pipes.participating.begin(), command_pipes.participating.end());
    const auto entry = discovered.entries.find(child.get());
    CHECK(entry != discovered.entries.end()) << "Closed child lacks an original entry protocol cut";
    std::optional<std::set<int>> intersection;
    for (const auto source : pipes) {
      for (const auto destination : pipes) {
        if (source == destination) continue;
        CHECK(entry->second.count({source, destination}))
            << "Closed child entry lacks a participating-pipe acknowledgment; cross-visit rearm unproved";
        const auto available =
            SyncEligibleIds(function, target, {source, destination}, eligible_ids, reservations);
        std::set<int> candidates;
        for (const int id : available)
          if (!parent_ids.count(id)) candidates.insert(id);
        if (!intersection) {
          intersection = candidates;
        } else {
          std::set<int> common;
          std::set_intersection(intersection->begin(), intersection->end(), candidates.begin(),
                                candidates.end(), std::inserter(common, common.end()));
          intersection = std::move(common);
        }
      }
    }
    // The descriptor/caller binding is not a proof of token closure. Remove
    // only the routing hint and lower the actual token program through the
    // existing parser, endpoint decoder and allocator with its private pool.
    auto unmarked = std::make_shared<ForStmt>(*child);
    unmarked->attrs_.erase(
        std::remove_if(unmarked->attrs_.begin(), unmarked->attrs_.end(),
                       [](const auto& attribute) { return attribute.first == kClosedSyncChildAttr; }),
        unmarked->attrs_.end());
    class Replace : public IRMutator {
     public:
      Replace(ForStmtPtr before, ForStmtPtr after) : before_(std::move(before)), after_(std::move(after)) {}

     protected:
      StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
        return loop == before_ ? after_ : IRMutator::VisitStmt_(loop);
      }

     private:
      ForStmtPtr before_, after_;
    } replace(child, unmarked);
    auto actual = replace.VisitFunction(context.binding_owner);
    std::optional<std::vector<int>> private_ids;
    if (intersection) private_ids = std::vector<int>(intersection->begin(), intersection->end());
    auto lowered = LowerLogicalSync(actual, target, private_ids, {}, std::nullopt, {}, external_policy);
    const auto original_body = As<SeqStmts>(actual->body_);
    const auto lowered_body = As<SeqStmts>(lowered.function->body_);
    INTERNAL_CHECK(original_body && lowered_body) << "Internal error: child binding body is not a sequence";
    std::set<const Var*> descriptor_variables;
    for (const auto& statement : original_body->stmts_)
      if (const auto assignment = As<AssignStmt>(statement))
        descriptor_variables.insert(assignment->var_.get());
    std::vector<StmtPtr> recipe;
    for (const auto& statement : lowered_body->stmts_) {
      if (As<ReturnStmt>(statement)) continue;
      const auto assignment = As<AssignStmt>(statement);
      if (assignment && descriptor_variables.count(assignment->var_.get())) continue;
      recipe.push_back(statement);
    }
    children.emplace(child.get(), std::make_shared<SeqStmts>(recipe, child->span_));
    child_sets += lowered.set_sites;
    child_waits += lowered.wait_sites;
    for (const auto& plan : lowered.physical_cases) {
      child_barriers += plan.barrier_sites;
      child_lanes += plan.reserved_lanes;
    }
  }
  const Assignment assignment(function, discovered, native.payloads, target, eligible_ids);
  PhysicalSyncRealization result;
  Emit emit(assignment, function->func_type_, children, discovered.completed_tokens);
  auto rewritten = MutableCopy(function);
  rewritten->body_ = emit.VisitStmt(function->body_);
  result.function = std::move(rewritten);
  result.set_sites = 2 * discovered.transactions.size() + child_sets;
  result.wait_sites = 2 * discovered.transactions.size() + child_waits;
  result.barrier_sites = child_barriers;
  result.reserved_lanes = assignment.Roles() + child_lanes;
  for (const auto& [cut, command] : assignment.Commands())
    result.barrier_sites += command.kind == CommandKind::Barrier;
  result.finite_minimum = false;
  for (const auto& [direction, ids] : assignment.Reservations())
    for (const int id : ids) result.native_reserved_ids.emplace_back(direction.first, direction.second, id);
  return result;
}
}  // namespace pypto::ir::sync
