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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace {
using Event = std::tuple<PipeType, PipeType, int>;
using Ready = std::unordered_set<const Var*>;
using Cuts = std::vector<StmtPtr>;

std::optional<Event> Flag(const StmtPtr& statement, bool publish) {
  const auto eval = As<EvalStmt>(statement);
  const auto call = eval ? As<Call>(eval->expr_) : nullptr;
  if (!call || !IsOp(call, publish ? "system.set_flag" : "system.wait_flag")) {
    return std::nullopt;
  }
  return Event{RequireSyncPipeKwarg(call, "src_pipe"), RequireSyncPipeKwarg(call, "dst_pipe"),
               call->GetKwarg<int>("event_id")};
}

bool AtomReady(const ExprPtr& expression, const Ready& ready) {
  const auto variable = AsVarLike(expression);
  return variable ? ready.count(variable.get()) != 0 : static_cast<bool>(As<ConstInt>(expression));
}

bool ComparisonReady(const ExprPtr& expression, const Ready& ready) {
  if (!(As<Lt>(expression) || As<Le>(expression) || As<Gt>(expression) || As<Ge>(expression) ||
        As<Eq>(expression) || As<Ne>(expression))) {
    return false;
  }
  const auto binary = As<BinaryExpr>(expression);
  return AtomReady(binary->left_, ready) && AtomReady(binary->right_, ready);
}

bool BoundReady(const ExprPtr& expression, const Ready& ready) {
  const auto sum = As<Add>(expression);
  return sum ? AtomReady(sum->left_, ready) && AtomReady(sum->right_, ready) : AtomReady(expression, ready);
}

struct Receiver {
  Event event;
  StmtPtr publication;
  StmtPtr consumer;
  ExprPtr steady_guard;
  Cuts prefix;  // All issued statements from branch entry to the original SET.
};

struct Suffix {
  StmtPtr anchor;
  std::vector<const Stmt*> payloads;
  Cuts corridor;
};

struct Move {
  EarlierPeerPrefix proof;
  StmtPtr publication;
  ExprPtr guard;
  std::vector<StmtPtr> anchors;
};

bool HasUniqueMoveCuts(const FunctionPtr& physical, const std::vector<Move>& moves) {
  class CountCuts final : public IRVisitor {
   public:
    void VisitStmt(const StmtPtr& statement) override {
      if (const auto found = counts.find(statement.get()); found != counts.end()) {
        ++found->second;
      }
      IRVisitor::VisitStmt(statement);
    }
    std::unordered_map<const Stmt*, size_t> counts;
  } counter;
  for (const auto& move : moves) {
    counter.counts.emplace(move.publication.get(), 0);
    for (const auto& anchor : move.anchors) {
      counter.counts.emplace(anchor.get(), 0);
    }
  }
  counter.VisitStmt(physical->body_);
  for (const auto& cut : counter.counts) {
    if (cut.second != 1) {
      return false;
    }
  }
  return true;
}

class Nominate {
 public:
  Nominate(const ImportedLocalSync& imported, const std::string& target, FunctionType core)
      : imported_(imported), target_(target), core_(core) {}

  std::vector<Move> Run(const FunctionPtr& owner) {
    for (const auto& parameter : owner->params_) {
      ready_.insert(parameter.get());
    }
    Visit(owner->body_);
    return std::move(moves_);
  }

 private:
  const ImportedLocalCut* Imported(const StmtPtr& statement) const {
    const auto call = transform_utils::GetCallFromStmt(statement);
    const auto found = imported_.calls.find(call.get());
    if (found == imported_.calls.end() || found->second.statement != statement.get()) {
      return nullptr;
    }
    return &found->second;
  }

  bool IsPeerWait(const StmtPtr& statement) const {
    const auto imported = Imported(statement);
    return imported && !imported->pipe && !imported->HasOpaqueProtocol() &&
           IsOp(transform_utils::GetCallFromStmt(statement), "system.sync_wait");
  }

  bool Metadata(const StmtPtr& statement) const {
    const auto call = transform_utils::GetCallFromStmt(statement);
    if (!call) {
      // All source scalar expressions on the imported route have passed its
      // readiness checks. A pure definition changes neither queue.
      return static_cast<bool>(As<AssignStmt>(statement));
    }
    return backend::QualifyBufferSyncNonPayload(call, target_).has_value();
  }

  // Every existing consume-before-republish path reaching the old SET must
  // already reach its source pipe here. In particular there is no native
  // handoff or WAIT into it, and nothing touches the same physical event.
  // This also preserves reuse paths for other IDs that traverse the moved
  // event: their consuming WAIT already reached the publishing pipe.
  bool Crossable(const StmtPtr& statement, const Event& moving) const {
    const auto [source, destination, id] = moving;
    (void)destination;
    (void)id;
    if (const auto wait = Flag(statement, false)) {
      return *wait != moving && std::get<1>(*wait) != source;
    }
    if (const auto set = Flag(statement, true)) {
      return *set != moving;
    }
    if (const auto imported = Imported(statement); imported && imported->pipe) {
      return *imported->pipe == source && !imported->HasOpaqueProtocol();
    }
    const auto call = transform_utils::GetCallFromStmt(statement);
    const auto barrier = RequireLocalSyncBarrier(source, core_);
    return (call && barrier && IsOp(call, barrier)) || IsPeerWait(statement) || Metadata(statement);
  }

  std::optional<Receiver> LeadingReceiver(const IfStmtPtr& branch) const {
    Cuts prefix;
    StmtPtr publication;
    ExprPtr steady_guard;
    std::optional<Event> event;
    bool author_wait = false;
    size_t publication_prefix_size = 0;
    const auto body = transform_utils::FlattenToStmts(branch->then_body_);
    for (size_t i = 0; i < body.size(); ++i) {
      const auto& statement = body[i];
      if (const auto imported = Imported(statement); imported && imported->pipe) {
        if (author_wait && event && !imported->HasOpaqueProtocol() &&
            *imported->pipe == std::get<1>(*event)) {
          prefix.resize(publication_prefix_size);
          return Receiver{*event, publication, statement, steady_guard, std::move(prefix)};
        }
        break;
      }

      if (const auto guarded = As<IfStmt>(statement);
          guarded && !guarded->else_body_ && guarded->return_vars_.empty()) {
        const auto guarded_body = transform_utils::FlattenToStmts(guarded->then_body_);
        if (guarded_body.size() != 2) {
          break;
        }
        const auto set = Flag(guarded_body.front(), true);
        if (!set || set != Flag(guarded_body.back(), false)) {
          break;
        }
        if (AsyncCrossPipe(*set)) {
          event = set;
          publication_prefix_size = prefix.size();
          publication = guarded_body.front();
          steady_guard = guarded->condition_;
        }
      } else if (const auto set = Flag(statement, true);
                 set && i + 1 < body.size() && set == Flag(body[i + 1], false) && AsyncCrossPipe(*set)) {
        event = set;
        publication_prefix_size = prefix.size();
        publication = statement;
        steady_guard = nullptr;
      } else if (IsPeerWait(statement)) {
        author_wait = true;
      } else if (!Metadata(statement) && !Flag(statement, false)) {
        break;
      }
      prefix.push_back(statement);
    }
    return std::nullopt;
  }

  static bool AsyncCrossPipe(const Event& event) {
    const auto [source, destination, id] = event;
    (void)id;
    return source != destination && source != PipeType::S && destination != PipeType::S;
  }

  std::optional<Suffix> TrailingSource(const Cuts& statements, const Event& moving) const {
    Suffix suffix;
    size_t begin = statements.size();
    if (begin && As<YieldStmt>(statements.back())) {
      --begin;
    }
    for (size_t i = begin; i > 0; --i) {
      const auto& statement = statements[i - 1];
      if (!Crossable(statement, moving)) {
        break;
      }
      suffix.corridor.push_back(statement);
      suffix.anchor = statement;
      const auto imported = Imported(statement);
      if (imported && imported->pipe) {
        suffix.payloads.push_back(statement.get());
      }
    }
    if (suffix.payloads.empty()) {
      return std::nullopt;
    }
    return suffix;
  }

  void Add(const Receiver& receiver, const ExprPtr& guard, const std::vector<Suffix>& suffixes,
           const Var* backedge_iv) {
    // Disjoint corridors compose using the already allocated event names.
    // A candidate contributes at most one SET in each of two exclusive arms.
    std::vector<const Stmt*> corridor{receiver.publication.get()};
    auto remember = [&](const Cuts& cuts) {
      for (const auto& cut : cuts) {
        corridor.push_back(cut.get());
      }
    };
    remember(receiver.prefix);
    Move move{{receiver.consumer.get(), std::get<0>(receiver.event), {}, backedge_iv},
              receiver.publication,
              guard,
              {}};
    for (const auto& suffix : suffixes) {
      remember(suffix.corridor);
      move.anchors.push_back(suffix.anchor);
      move.proof.displaced.insert(move.proof.displaced.end(), suffix.payloads.begin(), suffix.payloads.end());
    }
    for (const auto* cut : corridor) {
      if (claimed_.count(cut)) {
        return;
      }
    }
    claimed_.insert(corridor.begin(), corridor.end());
    moves_.push_back(std::move(move));
  }

  bool EntryTransparent(const Receiver& receiver) const {
    for (const auto& cut : receiver.prefix) {
      const auto imported = Imported(cut);
      if ((imported && imported->pipe) || !Crossable(cut, receiver.event)) {
        return false;
      }
    }
    return true;
  }

  void Sibling(const IfStmtPtr& from, const IfStmtPtr& to, const Cuts& between) {
    if (!from->else_body_ || !ComparisonReady(to->condition_, ready_)) {
      return;
    }
    const auto receiver = LeadingReceiver(to);
    if (!receiver || receiver->steady_guard || !EntryTransparent(*receiver)) {
      return;
    }
    const auto then_tail = TrailingSource(transform_utils::FlattenToStmts(from->then_body_), receiver->event);
    const auto else_tail =
        TrailingSource(transform_utils::FlattenToStmts(*from->else_body_), receiver->event);
    if (then_tail && else_tail) {
      Receiver joined = *receiver;
      joined.prefix.insert(joined.prefix.end(), between.begin(), between.end());
      Add(joined, to->condition_, {*then_tail, *else_tail}, nullptr);
    }
  }

  void Latch(const ForStmtPtr& loop, const Cuts& body) {
    const auto step = As<ConstInt>(loop->step_);
    const auto index_type = [](const ExprPtr& expression) {
      const auto scalar = As<ScalarType>(expression->GetType());
      return scalar && scalar->dtype_ == DataType::INDEX;
    };
    if (loop->kind_ != ForKind::Sequential || loop->hardware_loop_policy_ != HardwareLoopPolicy::Auto ||
        !loop->iter_args_.empty() || !step || step->value_ != 1 || !index_type(loop->step_) ||
        !index_type(loop->loop_var_) || !index_type(loop->start_) || !index_type(loop->stop_) ||
        !BoundReady(loop->stop_, ready_)) {
      return;
    }
    Cuts before;
    size_t i = 0;
    while (i < body.size() && Metadata(body[i])) {
      before.push_back(body[i++]);
    }
    if (i == body.size()) {
      return;
    }
    const auto head = As<IfStmt>(body[i]);
    const auto head_test = head ? As<Lt>(head->condition_) : nullptr;
    if (!head_test || head_test->left_ != loop->loop_var_ || !AtomReady(head_test->right_, ready_)) {
      return;
    }
    auto receiver = LeadingReceiver(head);
    if (!receiver || !receiver->steady_guard ||
        !structural_equal(receiver->steady_guard, MakeNe(loop->loop_var_, loop->start_, loop->span_))) {
      return;
    }
    receiver->prefix.insert(receiver->prefix.end(), before.begin(), before.end());
    if (!EntryTransparent(*receiver)) {
      return;
    }
    const auto tail = TrailingSource(body, receiver->event);
    if (!tail) {
      return;
    }
    // The original unit-stride INDEX loop reached this latch with iv < stop,
    // so iv + 1 is representable even if stop is the largest signed index.
    // Test the original loop bound before publishing; an empty or one-trip
    // loop cannot publish at this cut for a nonexistent steady visit.
    const auto next = MakeAdd(loop->loop_var_, loop->step_, loop->span_);
    const auto guard = MakeAnd(MakeLt(next, loop->stop_, loop->span_),
                               MakeLt(next, head_test->right_, head_test->span_), loop->span_);
    Add(*receiver, guard, {*tail}, loop->loop_var_.get());
  }

  void VisitSeq(const Cuts& body) {
    std::vector<const Var*> bindings;
    for (size_t i = 0; i < body.size(); ++i) {
      const auto& statement = body[i];
      if (const auto branch = As<IfStmt>(statement)) {
        Cuts between;
        size_t next = i + 1;
        while (next < body.size() && Metadata(body[next])) {
          between.push_back(body[next++]);
        }
        if (next < body.size()) {
          if (const auto sibling = As<IfStmt>(body[next])) {
            Sibling(branch, sibling, between);
          }
        }
        Visit(branch->then_body_);
        if (branch->else_body_) {
          Visit(*branch->else_body_);
        }
        for (const auto& result : branch->return_vars_) {
          Bind(result, bindings);
        }
      } else if (const auto loop = As<ForStmt>(statement)) {
        std::vector<const Var*> local;
        Bind(loop->loop_var_, local);
        for (const auto& item : loop->iter_args_) {
          Bind(item, local);
        }
        const auto nested = transform_utils::FlattenToStmts(loop->body_);
        Latch(loop, nested);
        VisitSeq(nested);
        for (const auto* variable : local) {
          ready_.erase(variable);
        }
        for (const auto& result : loop->return_vars_) {
          Bind(result, bindings);
        }
      } else if (const auto assignment = As<AssignStmt>(statement)) {
        Bind(assignment->var_, bindings);
      }
    }
    for (const auto* variable : bindings) {
      ready_.erase(variable);
    }
  }

  void Bind(const VarPtr& variable, std::vector<const Var*>& bindings) {
    if (ready_.insert(variable.get()).second) {
      bindings.push_back(variable.get());
    }
  }

  void Visit(const StmtPtr& body) { VisitSeq(transform_utils::FlattenToStmts(body)); }

  const ImportedLocalSync& imported_;
  const std::string& target_;
  FunctionType core_;
  Ready ready_;
  std::unordered_set<const Stmt*> claimed_;
  std::vector<Move> moves_;
};

class Rewrite final : public SyncStatementMutator {
 public:
  explicit Rewrite(const std::vector<Move>& moves) {
    for (const auto& move : moves) {
      removed_.insert(move.publication.get());
      for (const auto& anchor : move.anchors) {
        const auto& span = move.publication->span_;
        auto publication = std::make_shared<EvalStmt>(As<EvalStmt>(move.publication)->expr_, span);
        auto body = SeqStmts::Flatten({publication}, span);
        auto guarded =
            std::make_shared<IfStmt>(move.guard, std::move(body), std::nullopt, std::vector<VarPtr>{}, span);
        inserted_.emplace(anchor.get(), SeqStmts::Flatten({guarded, anchor}, span));
      }
    }
  }

  StmtPtr VisitStmt(const StmtPtr& statement) override {
    if (removed_.count(statement.get())) {
      return std::make_shared<SeqStmts>(Cuts{}, statement->span_);
    }
    if (const auto found = inserted_.find(statement.get()); found != inserted_.end()) {
      return found->second;
    }
    return SyncStatementMutator::VisitStmt(statement);
  }

 private:
  std::unordered_set<const Stmt*> removed_;
  std::unordered_map<const Stmt*, StmtPtr> inserted_;
};
}  // namespace

FunctionPtr MoveEarlierPeerPrefixes(const FunctionPtr& physical, const ConservativeSyncAnalysis& baseline,
                                    const ImportedLocalSync& imported, const std::string& target,
                                    const SyncParameterBounds& parameters) {
  const auto moves = Nominate(imported, target, physical->func_type_).Run(physical);
  if (moves.empty() || !HasUniqueMoveCuts(physical, moves)) {
    return physical;
  }
  std::vector<EarlierPeerPrefix> proofs;
  for (const auto& move : moves) {
    proofs.push_back(move.proof);
  }
  if (!CertifyEarlierPeerPrefixes(baseline, imported, target, parameters, proofs)) {
    return physical;
  }
  auto candidate = Rewrite(moves).VisitFunction(physical);
  // A new scalar guard can close the descriptor proof's lexical epoch. Keep
  // the old allocation when the issue cuts need earlier valid-shape evidence.
  if (BufferRuntimeValidAnalysis(candidate).issue()) {
    return physical;
  }
  return candidate;
}

}  // namespace pypto::ir::sync
