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

// Finite RFC token interpretation. Fixed-size pipe frontiers make propagation
// O(k*(N+E)); ordered maps and the shared lane sweep add O((N+F) log(N+F)).
// E counts distinct token-expression edges and F actual consuming handoffs.
// This is command semantics, not access-conflict analysis or demand reduction.
// Original operation/footprint import is reused for pipe/native qualification;
// its normalization costs are separate from these token-interpretation bounds.
#include "src/ir/transforms/utils/token_sync_lowering.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"

namespace pypto::ir::sync {
namespace {
using Row = std::array<size_t, kPipeCount>;
using Token = std::optional<size_t>;  // Index of a position-sensitive record; null is completed.

void Join(Row& target, const Row& source) {
  for (size_t pipe = 0; pipe < kPipeCount; ++pipe) target[pipe] = std::max(target[pipe], source[pipe]);
}
StmtPtr Empty(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }
PipeType Pipe(const CallPtr& call, const char* key) {
  for (const auto& [name, value] : call->kwargs_)
    if (name == key) return static_cast<PipeType>(AnyCast<int>(value, name));
  INTERNAL_CHECK(false) << "Internal error: verified synchronization pipe is missing";
  return PipeType::ALL;
}
// Discovery and admission use the canonical logical-op classification. Cache
// expression nodes because token joins and scalar metadata are shared DAGs.
class FindLogical : public IRVisitor {
 public:
  explicit FindLogical(bool check_lowerable = false) : check_lowerable_(check_lowerable) {}
  bool found = false;
  void VisitExpr(const ExprPtr& expression) override {
    if ((check_lowerable_ || !found) && expression && seen_.insert(expression.get()).second)
      IRVisitor::VisitExpr(expression);
  }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    const auto kind = GetLogicalSyncKind(call->op_);
    if (kind != LogicalSyncKind::None) found = true;
    CHECK(!check_lowerable_ ||
          (kind != LogicalSyncKind::MemoryOverlaps && kind != LogicalSyncKind::VFBarrier))
        << "LowerSync unsupported logical operation: " << call->op_->name_;
    IRVisitor::VisitExpr_(call);
  }

 private:
  bool check_lowerable_;
  std::set<const Expr*> seen_;
};

// Statement-only rewriting deliberately preserves original expression DAGs.
// The base mutator dispatches sequence children directly through StmtFunctor,
// bypassing a generic VisitStmt override. Own the sequence traversal explicitly
// so admission and replacement run on every occurrence, including nested Seq.
class FiniteStatements : public IRMutator {
 public:
  StmtPtr VisitStmt(const StmtPtr& op) override {
    CHECK(As<SeqStmts>(op) || As<AssignStmt>(op) || As<EvalStmt>(op) || As<ReturnStmt>(op))
        << "LowerSync structured token transport is unavailable";
    return IRMutator::VisitStmt(op);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return RewriteSyncSequence(op, [this](const StmtPtr& child) { return VisitStmt(child); });
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override { return op; }
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override { return op; }
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override { return op; }
};

// Erasure supplies only the payload/native-variant importer. It never escapes
// as executable output and never supplies dependency edges.
class StripTokens : public FiniteStatements {
 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    return As<SyncTokenType>(op->var_->GetType()) ? Empty(op->span_) : op;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    return IsOp(As<Call>(op->expr_), "sync.wait") ? Empty(op->span_) : op;
  }
};

struct Record {
  AssignStmtPtr cut;
  PipeType pipe;
  Row ancestors;
  size_t payload;  // Latest source-pipe payload number, zero if the prefix is empty.
};
struct Handoff {
  size_t record;
  EvalStmtPtr cut;
  PipeType destination;
  size_t wait_rank;
  int id = -1;
};

class FiniteLowering {
 public:
  FiniteLowering(const FunctionPtr& function, CompactSyncInput input, const std::string& target,
                 const std::optional<std::vector<int>>& eligible)
      : function_(function), input_(std::move(input)), target_(target), eligible_(eligible) {
    for (const auto& segment : input_.segments) {
      CHECK(!segment.original_loop && segment.nested_frames.empty() && segment.count == 1)
          << "LowerSync structured token transport is unavailable";
      for (const auto& site : segment.sites) {
        CHECK(!site.continuation) << "LowerSync structured token transport is unavailable";
        CHECK(pipes_.emplace(site.original.get(), site.pipe).second)
            << "LowerSync requires distinct original payload occurrence anchors";
      }
    }
  }
  PhysicalSyncRealization Run() {
    Read(function_->body_);
    auto reservations = SyncNativeClaims(input_, target_);
    for (const auto& [direction, ids] : reservations)
      for (int id : ids) result_.native_reserved_ids.emplace_back(direction.first, direction.second, id);
    std::map<SyncDirection, std::vector<size_t>> directions;
    for (size_t i = 0; i < handoffs_.size(); ++i)
      directions[{records_[handoffs_[i].record].pipe, handoffs_[i].destination}].push_back(i);
    for (auto& [direction, lanes] : directions) {
      std::sort(lanes.begin(), lanes.end(), [&](size_t a, size_t b) {
        return std::make_pair(handoffs_[a].record, a) < std::make_pair(handoffs_[b].record, b);
      });
      const auto colors = ColorOrderedSyncLanes(
          lanes.size(), direction,
          [&](size_t a, size_t b) { return handoffs_[lanes[a]].wait_rank < handoffs_[lanes[b]].wait_rank; },
          [&](size_t a, size_t b) {
            ++result_.reuse_queries;
            const auto& consumed = handoffs_[lanes[a]];
            const auto& publication = records_[handoffs_[lanes[b]].record];
            return publication.ancestors[consumed.destination] >= consumed.wait_rank;
          },
          // Minimum for this selected per-use physical handoff graph only;
          // no alternative fanout protocol or hardware timing claim is made.
          SyncEligibleIds(input_, target_, direction, eligible_, reservations), true);
      for (size_t i = 0; i < lanes.size(); ++i) handoffs_[lanes[i]].id = colors[i];
    }
    std::map<const Stmt*, std::vector<StmtPtr>> emitted;
    for (const auto& record : records_) emitted[record.cut.get()];
    for (const auto& handoff : handoffs_) {
      const auto& record = records_[handoff.record];
      emitted[record.cut.get()].push_back(Flag("system.set_flag", record.pipe, handoff, record.cut->span_));
      emitted[handoff.cut.get()].push_back(
          Flag("system.wait_flag", record.pipe, handoff, handoff.cut->span_));
      ++result_.set_sites;
      ++result_.wait_sites;
      ++result_.reserved_lanes;
    }
    for (const auto& [cut, statement] : barriers_) emitted[cut].push_back(statement);
    class Emit : public FiniteStatements {
     public:
      explicit Emit(const std::map<const Stmt*, std::vector<StmtPtr>>& replacements)
          : replacements_(replacements) {}
      StmtPtr VisitStmt(const StmtPtr& op) override {
        const auto found = replacements_.find(op.get());
        if (found != replacements_.end()) return SeqStmts::Flatten(found->second, op->span_);
        if (auto assign = As<AssignStmt>(op))
          if (As<SyncTokenType>(assign->var_->GetType())) return Empty(op->span_);
        if (auto eval = As<EvalStmt>(op))
          if (IsOp(As<Call>(eval->expr_), "sync.wait")) return Empty(op->span_);
        // Scalar/descriptor/payload expressions are unchanged. Only statement
        // sequences contain replacement cuts; do not rewalk shared expression
        // DAGs during emission.
        return FiniteStatements::VisitStmt(op);
      }

     private:
      const std::map<const Stmt*, std::vector<StmtPtr>>& replacements_;
    };
    result_.function = Emit(emitted).VisitFunction(function_);
    return std::move(result_);
  }

 private:
  FunctionPtr function_;
  CompactSyncInput input_;
  std::string target_;
  const std::optional<std::vector<int>>& eligible_;
  std::map<const Call*, PipeType> pipes_;
  std::map<const Var*, Token> variables_;
  std::map<const Expr*, Token> values_;
  std::set<const Stmt*> token_cuts_;
  std::array<Row, kPipeCount> starts_{}, completions_{}, gates_{};
  std::array<size_t, kPipeCount> payloads_{}, wait_ranks_{};
  std::vector<Record> records_;
  std::vector<Handoff> handoffs_;
  std::map<const Stmt*, StmtPtr> barriers_;
  PhysicalSyncRealization result_;
  bool returned_ = false;

  Token Value(const ExprPtr& expression) {
    const auto cached = values_.find(expression.get());
    if (cached != values_.end()) return cached->second;
    Token result;
    if (auto var = As<Var>(expression)) {
      const auto found = variables_.find(var.get());
      CHECK(found != variables_.end()) << "LowerSync token definition is unavailable";
      result = found->second;
    } else if (auto call = As<Call>(expression)) {
      if (IsOp(call, "sync.completed")) {
        result = std::nullopt;
      } else if (IsOp(call, "sync.join")) {
        const auto a = Value(call->args_[0]), b = Value(call->args_[1]);
        // In one straight-line execution, later records on the same pipe
        // include earlier payload completions and imported waits. Thus the
        // later prefix is exactly the conjunction, not a stronger substitute.
        result = !a ? b : (!b ? a : Token(std::max(*a, *b)));
      } else {
        CHECK(false) << "LowerSync unsupported token expression: " << call->op_->name_;
      }
    } else {
      CHECK(false) << "LowerSync token value transport is unavailable";
    }
    values_.emplace(expression.get(), result);
    return result;
  }
  void Read(const StmtPtr& statement) {
    if (auto seq = As<SeqStmts>(statement)) {
      for (const auto& child : seq->stmts_) Read(child);
      return;
    }
    CHECK(!returned_) << "LowerSync requires a terminal return; unreachable token cuts are unsupported";
    if (As<ReturnStmt>(statement)) {
      returned_ = true;
    } else if (auto assign = As<AssignStmt>(statement)) {
      if (!As<SyncTokenType>(assign->var_->GetType())) return;
      CHECK(token_cuts_.insert(statement.get()).second)
          << "LowerSync requires distinct token occurrence cuts";
      auto call = As<Call>(assign->value_);
      Token token;
      if (IsOp(call, "sync.record")) {
        const auto pipe = Pipe(call, "pipe");
        Row row = completions_[pipe];
        Join(row, gates_[pipe]);
        // A record of a genuinely empty prefix denotes the same satisfied
        // fact as completed. An imported wait makes that prefix nonempty even
        // when this pipe has issued no payload of its own.
        const bool imported = std::any_of(row.begin(), row.end(), [](size_t rank) { return rank != 0; });
        if (payloads_[pipe] || imported) {
          token = records_.size();
          records_.push_back({assign, pipe, row, payloads_[pipe]});
        }
      } else {
        token = Value(assign->value_);
      }
      variables_.emplace(assign->var_.get(), token);
    } else if (auto eval = As<EvalStmt>(statement)) {
      const auto call = As<Call>(eval->expr_);
      if (IsOp(call, "sync.wait")) {
        CHECK(token_cuts_.insert(statement.get()).second)
            << "LowerSync requires distinct token occurrence cuts";
        const auto token = Value(call->args_[0]);
        if (!token) return;  // No completion obligation, and no consuming event.
        const auto destination = Pipe(call, "dst_pipe");
        const auto& record = records_[*token];
        Row row = starts_[destination];
        Join(row, gates_[destination]);
        Join(row, record.ancestors);
        row[destination] = ++wait_ranks_[destination];
        gates_[destination] = row;
        if (record.pipe == destination) {
          // A hardware barrier completes the entire destination prefix. It
          // implements this token wait exactly only if no later payload was
          // issued on that pipe, except the natively serial scalar pipeline.
          CHECK(destination == PipeType::S || record.payload == payloads_[destination])
              << "LowerSync same-pipe barrier would strengthen the recorded prefix";
          if (destination == PipeType::S || !payloads_[destination]) return;
          CHECK(destination == PipeType::M || destination == PipeType::V)
              << "LowerSync same-pipe barrier mechanism is unavailable";
          const auto barrier = OpRegistry::GetInstance().Create(
              destination == PipeType::M ? "system.bar_m" : "system.bar_v", {}, {}, statement->span_);
          barriers_.emplace(statement.get(), std::make_shared<EvalStmt>(barrier, statement->span_));
          ++result_.barrier_sites;
        } else {
          handoffs_.push_back({*token, eval, destination, wait_ranks_[destination], -1});
        }
      } else {
        const auto found = pipes_.find(call.get());
        CHECK(found != pipes_.end()) << "LowerSync payload pipe/native contract is unavailable";
        const auto pipe = found->second;
        Row start = starts_[pipe];
        Join(start, gates_[pipe]);
        // This is the shared target's qualified scalar order, not a barrier.
        if (pipe == PipeType::S) Join(start, completions_[pipe]);
        starts_[pipe] = start;
        Join(completions_[pipe], start);
        ++payloads_[pipe];
      }
    }
  }
  StmtPtr Flag(const char* name, PipeType source, const Handoff& handoff, const Span& span) const {
    const auto call = OpRegistry::GetInstance().Create(
        name, {},
        {{"src_pipe", static_cast<int>(source)},
         {"dst_pipe", static_cast<int>(handoff.destination)},
         {"event_id", handoff.id},
         {"core_type", std::string(function_->func_type_ == FunctionType::AIV ? "aiv" : "aic")}},
        span);
    return std::make_shared<EvalStmt>(call, span);
  }
};
}  // namespace
bool HasLogicalSync(const FunctionPtr& function) {
  FindLogical logical;
  logical.VisitFunction(function);
  return logical.found;
}
PhysicalSyncRealization LowerLogicalSync(const FunctionPtr& function, const std::string& target,
                                         const std::optional<std::vector<int>>& eligible_ids,
                                         const SyncParameterBounds& parameters,
                                         std::optional<std::pair<int64_t, int64_t>> blocks) {
  CheckLogicalSync(function);
  if (!HasLogicalSync(function)) {
    PhysicalSyncRealization result;
    result.function = function;
    return result;
  }
  FindLogical(true).VisitFunction(function);
  class Structure : public IRVisitor {
   public:
    bool loop = false;

   protected:
    void VisitStmt_(const ForStmtPtr&) override { loop = true; }
    void VisitExpr(const ExprPtr&) override {}
  } structure;
  structure.VisitFunction(function);
  if (structure.loop) return LowerStructuredLogicalSync(function, target, eligible_ids, parameters, blocks);
  auto stripped = StripTokens().VisitFunction(function);
  // Only the payload pipe/native-variant inventory is consumed here. The
  // importer's GM partition is not used for conflicts, selectors or reuse;
  // token IR, including any manually supplied extra ordering, is authoritative.
  auto input = ImportPlacedSync(stripped, true, target, parameters, blocks);
  return FiniteLowering(function, std::move(input), target, eligible_ids).Run();
}
}  // namespace pypto::ir::sync
