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
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/native_sync_placement.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/retained_sync_child.h"
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
                 const std::optional<std::vector<int>>& eligible,
                 SyncNativeReservations extra_reservations = {})
      : function_(function),
        input_(std::move(input)),
        target_(target),
        eligible_(eligible),
        extra_reservations_(std::move(extra_reservations)) {
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
  PhysicalSyncRealization Run(bool dry_run_only = false) {
    scalar_native_order_ = backend::RequireLocalSyncExecutionContract(target_).scalar_prefix_serialized;
    Read(function_->body_);
    auto reservations = SyncNativeClaims(input_, target_);
    for (const auto& [direction, ids] : reservations)
      for (int id : ids) result_.native_reserved_ids.emplace_back(direction.first, direction.second, id);
    for (const auto& [direction, ids] : extra_reservations_)
      reservations[direction].insert(ids.begin(), ids.end());
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
    if (dry_run_only) {
      result_.function = function_;
      return std::move(result_);
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
  SyncNativeReservations extra_reservations_;
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
  bool scalar_native_order_ = false;

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
          CHECK((destination == PipeType::S && scalar_native_order_) ||
                record.payload == payloads_[destination])
              << "LowerSync same-pipe barrier would strengthen the recorded prefix";
          if ((destination == PipeType::S && scalar_native_order_) || !payloads_[destination]) return;
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
        if (pipe == PipeType::S && scalar_native_order_) Join(start, completions_[pipe]);
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
void CheckPhysicalSyncResult(const FunctionPtr& function) {
  // A successful physical result must be standalone executable IR. Looking
  // only for logical calls misses aliases, carries, tuple members and signatures
  // left behind after their token-producing definitions have been erased.
  class PhysicalResult : public IRVisitor {
   public:
    void CheckType(const TypePtr& type, const Span& span) {
      if (!type || !types_.insert(type.get()).second) return;
      INTERNAL_CHECK_SPAN(!As<SyncTokenType>(type), span)
          << "Internal error: physical synchronization left a logical token type";
      if (const auto tuple = As<TupleType>(type))
        for (const auto& member : tuple->types_) CheckType(member, span);
    }
    void VisitExpr(const ExprPtr& expression) override {
      if (!expression || !seen_.insert(expression.get()).second) return;
      CheckType(expression->GetType(), expression->span_);
      IRVisitor::VisitExpr(expression);
    }

   protected:
    void VisitExpr_(const CallPtr& call) override {
      INTERNAL_CHECK_SPAN(GetLogicalSyncKind(call->op_) == LogicalSyncKind::None, call->span_)
          << "Internal error: physical synchronization left a logical operation";
      IRVisitor::VisitExpr_(call);
    }

   private:
    std::set<const Expr*> seen_;
    std::set<const Type*> types_;
  } physical;
  for (const auto& type : function->return_types_) physical.CheckType(type, function->span_);
  physical.VisitFunction(function);
}
void CheckGeneratedLogicalSync(const FunctionPtr& function) {
  std::vector<Diagnostic> diagnostics;
  VerifyLogicalSync(function, diagnostics);
  if (!diagnostics.empty()) {
    INTERNAL_CHECK_SPAN(false, diagnostics.front().span)
        << "Internal error: synchronization insertion produced invalid logical IR: "
        << diagnostics.front().message;
  }
}
namespace {

// #3375 Pass 2 (`OptimizeSync`):
// 1. Forward CFG simplification over SyncTokenType values:
//    - eliminate waits on `sync.completed` / completed aliases,
//    - simplify `sync.join` with completed/identical operands,
//    - eliminate dominated duplicate `(token, dst)` waits,
//    - eliminate intra-bundle FIFO-subsumed waits on the same producer pipe.
// 2. Backward liveness DCE removing dead `sync.record`/`sync.completed`/`sync.join` bindings.
class ForwardSyncOptimizer : public IRMutator {
 public:
  StmtPtr Run(const StmtPtr& body) {
    waited_masks_.clear();
    completed_.clear();
    local_record_.clear();
    alias_of_.clear();
    record_seq_ = 0;
    scope_depth_ = 0;
    next_scope_id_ = 1;
    return VisitStmt(body);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> rewritten;
    rewritten.reserve(op->stmts_.size());
    for (const auto& child : op->stmts_) {
      auto next = VisitStmt(child);
      if (const auto nested = As<SeqStmts>(next)) {
        for (const auto& s : nested->stmts_) rewritten.push_back(s);
      } else {
        rewritten.push_back(std::move(next));
      }
    }
    std::vector<StmtPtr> pruned;
    pruned.reserve(rewritten.size());
    size_t i = 0;
    while (i < rewritten.size()) {
      auto wait_info = ParseWait(rewritten[i]);
      if (!wait_info) {
        pruned.push_back(std::move(rewritten[i++]));
        continue;
      }
      size_t j = i;
      struct BundleWait {
        StmtPtr stmt;
        const Var* tok;
        PipeType dst;
        bool drop = false;
      };
      std::vector<BundleWait> bundle;
      while (j < rewritten.size()) {
        auto info = ParseWait(rewritten[j]);
        if (!info) break;
        bundle.push_back({rewritten[j], info->first.get(), info->second, false});
        ++j;
      }
      // Single reverse pass per (dst_pipe, src_pipe, scope_id) drops earlier dominated waits in O(K).
      std::map<std::tuple<PipeType, PipeType, size_t>, size_t> max_seq_after;
      for (size_t k = bundle.size(); k-- > 0;) {
        const auto rec = local_record_.find(bundle[k].tok);
        if (rec == local_record_.end()) continue;
        const auto key = std::make_tuple(bundle[k].dst, rec->second.pipe, rec->second.scope_id);
        const auto [it, inserted] = max_seq_after.emplace(key, rec->second.seq);
        if (!inserted) {
          if (rec->second.seq < it->second) {
            bundle[k].drop = true;
          } else if (rec->second.seq > it->second) {
            it->second = rec->second.seq;
          }
        }
      }
      for (auto& item : bundle) {
        if (!item.drop) pruned.push_back(std::move(item.stmt));
      }
      i = j;
    }
    return std::make_shared<SeqStmts>(std::move(pruned), op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (!As<SyncTokenType>(op->var_->GetType())) return op;
    alias_of_.erase(op->var_.get());
    completed_.erase(op->var_.get());
    local_record_.erase(op->var_.get());
    waited_masks_.erase(op->var_.get());
    auto bind_alias = [&](const VarPtr& target, const VarPtr& rhs) -> StmtPtr {
      const VarPtr root = Canonical(rhs);
      alias_of_[target.get()] = root;
      if (completed_.count(root.get())) completed_.insert(target.get());
      if (const auto rec = local_record_.find(root.get()); rec != local_record_.end()) {
        local_record_[target.get()] = rec->second;
      }
      return std::make_shared<AssignStmt>(target, root, op->span_, op->leading_comments_);
    };
    if (const auto rhs_var = AsVarLike(op->value_)) {
      const VarPtr root = Canonical(rhs_var);
      alias_of_[op->var_.get()] = root;
      if (completed_.count(root.get())) completed_.insert(op->var_.get());
      if (const auto rec = local_record_.find(root.get()); rec != local_record_.end()) {
        local_record_[op->var_.get()] = rec->second;
      }
      return root.get() == rhs_var.get()
                 ? StmtPtr(op)
                 : std::make_shared<AssignStmt>(op->var_, root, op->span_, op->leading_comments_);
    }
    if (const auto call = As<Call>(op->value_)) {
      if (IsOp(call, "sync.completed")) {
        completed_.insert(op->var_.get());
        return op;
      }
      if (IsOp(call, "sync.record")) {
        local_record_[op->var_.get()] = {Pipe(call, "pipe"), scope_depth_, record_seq_++};
        return op;
      }
      if (IsOp(call, "sync.join") && call->args_.size() == 2) {
        auto is_completed_expr = [&](const ExprPtr& e) -> bool {
          if (const auto c = As<Call>(e); c && IsOp(c, "sync.completed")) return true;
          if (const auto v = AsVarLike(e)) return completed_.count(v.get()) != 0;
          return false;
        };
        std::function<ExprPtr(const ExprPtr&)> simplify_join = [&](const ExprPtr& e) -> ExprPtr {
          if (const auto v = AsVarLike(e)) return Canonical(v);
          const auto c = As<Call>(e);
          if (!c || !IsOp(c, "sync.join") || c->args_.size() != 2) return e;
          ExprPtr lhs = simplify_join(c->args_[0]);
          ExprPtr rhs = simplify_join(c->args_[1]);
          const bool lc = is_completed_expr(lhs);
          const bool rc = is_completed_expr(rhs);
          if (lc && !rc) return rhs;
          if (rc) return lhs;
          const auto va = AsVarLike(lhs);
          const auto vb = AsVarLike(rhs);
          if (va && vb && va.get() == vb.get()) return va;
          if (lhs.get() != c->args_[0].get() || rhs.get() != c->args_[1].get()) {
            return OpRegistry::GetInstance().Create(c->op_->name_, {std::move(lhs), std::move(rhs)},
                                                    c->kwargs_, c->span_);
          }
          return e;
        };
        const ExprPtr simplified_rhs = simplify_join(call);
        if (const auto v = AsVarLike(simplified_rhs)) {
          return bind_alias(op->var_, v);
        }
        if (is_completed_expr(simplified_rhs)) {
          completed_.insert(op->var_.get());
        }
        if (simplified_rhs.get() != op->value_.get()) {
          return std::make_shared<AssignStmt>(op->var_, simplified_rhs, op->span_, op->leading_comments_);
        }
      }
    }
    return op;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    const auto info = ParseWait(op);
    if (!info) return op;
    const VarPtr root = Canonical(info->first);
    const size_t pipe_idx = static_cast<size_t>(info->second);
    INTERNAL_CHECK(pipe_idx < 32) << "Pipe index out of bitmask range";
    const uint32_t pipe_bit = uint32_t{1} << pipe_idx;
    uint32_t& mask = waited_masks_[root.get()];
    if (completed_.count(root.get()) || (mask & pipe_bit) != 0) {
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    mask |= pipe_bit;
    if (root.get() != info->first.get()) {
      const auto old_call = As<Call>(op->expr_);
      auto new_call = OpRegistry::GetInstance().Create(old_call->op_->name_, {root}, old_call->kwargs_,
                                                       old_call->span_);
      return std::make_shared<EvalStmt>(std::move(new_call), op->span_, op->leading_comments_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    bool changed = false;
    std::vector<ExprPtr> values = op->value_;
    for (auto& value : values) {
      if (const auto tok = AsVarLike(value); tok && As<SyncTokenType>(tok->GetType())) {
        const VarPtr root = Canonical(tok);
        if (root.get() != tok.get()) {
          value = root;
          changed = true;
        }
      }
    }
    if (!changed) return op;
    return std::make_shared<YieldStmt>(std::move(values), op->span_, op->leading_comments_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    const auto saved_waited = waited_masks_;
    const auto saved_completed = completed_;
    const auto saved_aliases = alias_of_;
    scope_depth_ = next_scope_id_++;
    StmtPtr then_body = VisitStmt(op->then_body_);
    const auto then_waited = std::move(waited_masks_);
    waited_masks_ = saved_waited;
    completed_ = saved_completed;
    alias_of_ = saved_aliases;
    std::optional<StmtPtr> else_body;
    if (op->else_body_) {
      scope_depth_ = next_scope_id_++;
      else_body = VisitStmt(*op->else_body_);
    }
    const auto& else_waited = op->else_body_ ? waited_masks_ : saved_waited;
    std::unordered_map<const Var*, uint32_t> intersected;
    for (const auto& [tok, then_mask] : then_waited) {
      const auto eit = else_waited.find(tok);
      if (eit == else_waited.end()) continue;
      const uint32_t merged = then_mask & eit->second;
      if (merged != 0) intersected.emplace(tok, merged);
    }
    waited_masks_ = std::move(intersected);
    completed_ = saved_completed;
    alias_of_ = saved_aliases;
    scope_depth_ = next_scope_id_++;
    return std::make_shared<IfStmt>(op->condition_, then_body, else_body, op->return_vars_, op->span_,
                                    op->leading_comments_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const auto saved_waited = waited_masks_;
    const auto saved_completed = completed_;
    const auto saved_aliases = alias_of_;
    waited_masks_.erase(op->loop_var_.get());
    completed_.erase(op->loop_var_.get());
    alias_of_.erase(op->loop_var_.get());
    local_record_.erase(op->loop_var_.get());
    for (const auto& arg : op->iter_args_) {
      waited_masks_.erase(arg.get());
      completed_.erase(arg.get());
      alias_of_.erase(arg.get());
      local_record_.erase(arg.get());
    }
    const size_t outer_scope = next_scope_id_++;
    scope_depth_ = next_scope_id_++;
    StmtPtr body = VisitStmt(op->body_);
    waited_masks_ = saved_waited;
    completed_ = saved_completed;
    alias_of_ = saved_aliases;
    scope_depth_ = outer_scope;
    return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_, body,
                                     op->return_vars_, op->span_, op->kind_, op->attrs_, op->leading_comments_,
                                     op->hardware_loop_policy_);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    const auto saved_waited = waited_masks_;
    const auto saved_completed = completed_;
    const auto saved_aliases = alias_of_;
    for (const auto& arg : op->iter_args_) {
      waited_masks_.erase(arg.get());
      completed_.erase(arg.get());
      alias_of_.erase(arg.get());
      local_record_.erase(arg.get());
    }
    const size_t outer_scope = next_scope_id_++;
    scope_depth_ = next_scope_id_++;
    StmtPtr body = VisitStmt(op->body_);
    waited_masks_ = saved_waited;
    completed_ = saved_completed;
    alias_of_ = saved_aliases;
    scope_depth_ = outer_scope;
    return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, body, op->return_vars_, op->span_,
                                       op->leading_comments_);
  }

 private:
  struct RecordInfo {
    PipeType pipe;
    size_t scope_id;
    size_t seq;
  };
  VarPtr Canonical(VarPtr v) const {
    while (v) {
      const auto it = alias_of_.find(v.get());
      if (it == alias_of_.end() || it->second.get() == v.get()) break;
      v = it->second;
    }
    return v;
  }
  static std::optional<std::pair<VarPtr, PipeType>> ParseWait(const StmtPtr& stmt) {
    const auto eval = As<EvalStmt>(stmt);
    const auto call = eval ? As<Call>(eval->expr_) : nullptr;
    if (!call || !IsOp(call, "sync.wait") || call->args_.empty()) return std::nullopt;
    const auto tok = AsVarLike(call->args_[0]);
    if (!tok) return std::nullopt;
    return std::make_pair(tok, Pipe(call, "dst_pipe"));
  }
  std::unordered_map<const Var*, uint32_t> waited_masks_;
  std::unordered_set<const Var*> completed_;
  std::unordered_map<const Var*, RecordInfo> local_record_;
  std::unordered_map<const Var*, VarPtr> alias_of_;
  size_t record_seq_ = 0;
  size_t scope_depth_ = 0;
  size_t next_scope_id_ = 1;
};

class SyncLivenessCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> Solve(const StmtPtr& body) {
    VisitStmt(body);
    std::deque<const Var*> work(live_.begin(), live_.end());
    while (!work.empty()) {
      const Var* cur = work.front();
      work.pop_front();
      if (const auto found = deps_.find(cur); found != deps_.end()) {
        for (const Var* next : found->second) {
          if (live_.insert(next).second) work.push_back(next);
        }
      }
    }
    return live_;
  }

 protected:
  void VisitExpr_(const VarPtr& var) override {
    if (As<SyncTokenType>(var->GetType())) live_.insert(var.get());
  }

  void VisitExpr_(const IterArgPtr& arg) override {
    if (As<SyncTokenType>(arg->GetType())) live_.insert(arg.get());
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!As<SyncTokenType>(op->var_->GetType())) {
      IRVisitor::VisitStmt_(op);
      return;
    }
    auto& dst = deps_[op->var_.get()];
    std::function<void(const ExprPtr&)> collect_token_deps = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (const auto v = AsVarLike(expr)) {
        dst.push_back(v.get());
        return;
      }
      if (const auto call = As<Call>(expr)) {
        if (IsOp(call, "sync.completed") || IsOp(call, "sync.record")) return;
        if (IsOp(call, "sync.join")) {
          for (const auto& arg : call->args_) collect_token_deps(arg);
          return;
        }
      }
      VisitExpr(expr);
    };
    collect_token_deps(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    std::vector<std::vector<const Var*>> targets(op->return_vars_.size());
    for (size_t i = 0; i < op->return_vars_.size(); ++i) targets[i].push_back(op->return_vars_[i].get());
    active_yield_targets_.push_back(std::move(targets));
    VisitStmt(op->then_body_);
    if (op->else_body_) VisitStmt(*op->else_body_);
    active_yield_targets_.pop_back();
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    LinkLoop(op->iter_args_, op->return_vars_, op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    VisitExpr(op->condition_);
    LinkLoop(op->iter_args_, op->return_vars_, op->body_);
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    // Yielded SyncToken variables remain referenced inside YieldStmt unless the
    // enclosing IfStmt/loop return arity is also pruned; keep them alive.
    for (const auto& val : op->value_) VisitExpr(val);
    if (active_yield_targets_.empty() || active_yield_targets_.back().size() != op->value_.size()) {
      return;
    }
    const auto& targets = active_yield_targets_.back();
    for (size_t i = 0; i < targets.size(); ++i) {
      if (const auto v = AsVarLike(op->value_[i]); v && As<SyncTokenType>(v->GetType())) {
        for (const Var* target : targets[i]) deps_[target].push_back(v.get());
      }
    }
  }

 private:
  void LinkLoop(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars,
                const StmtPtr& body) {
    std::vector<std::vector<const Var*>> targets(iter_args.size());
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& arg = iter_args[i];
      VisitExpr(arg->initValue_);
      if (As<SyncTokenType>(arg->GetType())) {
        if (const auto init_v = AsVarLike(arg->initValue_)) deps_[arg.get()].push_back(init_v.get());
        targets[i].push_back(arg.get());
        if (i < return_vars.size()) {
          deps_[return_vars[i].get()].push_back(arg.get());
          if (const auto init_v = AsVarLike(arg->initValue_))
            deps_[return_vars[i].get()].push_back(init_v.get());
          targets[i].push_back(return_vars[i].get());
        }
      }
    }
    active_yield_targets_.push_back(std::move(targets));
    VisitStmt(body);
    active_yield_targets_.pop_back();
  }

  std::unordered_set<const Var*> live_;
  std::unordered_map<const Var*, std::vector<const Var*>> deps_;
  std::vector<std::vector<std::vector<const Var*>>> active_yield_targets_;
};

class DeadSyncTokenEraser : public IRMutator {
 public:
  explicit DeadSyncTokenEraser(std::unordered_set<const Var*> live) : live_(std::move(live)) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    for (const auto& child : op->stmts_) {
      auto next = VisitStmt(child);
      if (const auto nested = As<SeqStmts>(next)) {
        for (const auto& s : nested->stmts_) out.push_back(s);
      } else {
        out.push_back(std::move(next));
      }
    }
    return std::make_shared<SeqStmts>(std::move(out), op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (As<SyncTokenType>(op->var_->GetType()) && !live_.count(op->var_.get())) {
      if (const auto call = As<Call>(op->value_)) {
        const auto kind = GetLogicalSyncKind(call->op_);
        if (kind == LogicalSyncKind::Completed || kind == LogicalSyncKind::Record ||
            kind == LogicalSyncKind::Join) {
          return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
        }
      } else if (AsVarLike(op->value_)) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
      }
    }
    return op;
  }

 private:
  std::unordered_set<const Var*> live_;
};

}  // namespace

FunctionPtr OptimizeLogicalSync(const FunctionPtr& function) {
  CheckLogicalSync(function);
  if (!HasLogicalSync(function)) return function;
  StmtPtr simplified = ForwardSyncOptimizer().Run(function->body_);
  std::unordered_set<const Var*> live = SyncLivenessCollector().Solve(simplified);
  StmtPtr cleaned = DeadSyncTokenEraser(std::move(live)).VisitStmt(simplified);
  if (cleaned == function->body_) return function;
  auto result = std::make_shared<Function>(
      function->name_, function->params_, function->param_directions_, function->return_types_, cleaned,
      function->span_, function->func_type_, function->level_, function->role_, function->attrs_,
      function->requires_runtime_binding_, function->ir_stage_);
  CheckLogicalSync(result);
  return result;
}
FunctionPtr StripAllLogicalSync(const FunctionPtr& function) {
  class StripAll : public SyncStatementMutator {
   protected:
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      return As<SyncTokenType>(op->var_->GetType()) ? Empty(op->span_) : op;
    }
    StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
      return IsOp(As<Call>(op->expr_), "sync.wait") ? Empty(op->span_) : op;
    }
    StmtPtr VisitStmt_(const IfStmtPtr& op) override {
      std::vector<VarPtr> return_vars;
      return_vars.reserve(op->return_vars_.size());
      for (const auto& rv : op->return_vars_) {
        if (!As<SyncTokenType>(rv->GetType())) return_vars.push_back(rv);
      }
      auto then_body = VisitStmt(op->then_body_);
      auto else_body = op->else_body_ ? std::optional<StmtPtr>(VisitStmt(*op->else_body_)) : std::nullopt;
      auto is_empty = [](const StmtPtr& s) {
        const auto seq = As<SeqStmts>(s);
        return seq && seq->stmts_.empty();
      };
      if (return_vars.empty() && is_empty(then_body) && (!else_body || is_empty(*else_body))) {
        return Empty(op->span_);
      }
      if (return_vars.size() == op->return_vars_.size() && then_body == op->then_body_ &&
          else_body == op->else_body_) {
        return op;
      }
      return std::make_shared<IfStmt>(op->condition_, std::move(then_body), std::move(else_body),
                                      std::move(return_vars), op->span_, op->leading_comments_);
    }
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      std::vector<IterArgPtr> iter_args;
      std::vector<VarPtr> return_vars;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        if (As<SyncTokenType>(op->iter_args_[i]->GetType())) continue;
        iter_args.push_back(op->iter_args_[i]);
        if (i < op->return_vars_.size()) return_vars.push_back(op->return_vars_[i]);
      }
      auto body = VisitStmt(op->body_);
      auto attrs = op->attrs_;
      attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                                 [](const auto& attr) {
                                   return attr.first == kClosedSyncChildAttr ||
                                          attr.first == kClosedSyncChildExternalPolicyAttr ||
                                          attr.first == "pypto.closed_sync_child_first_visit";
                                 }),
                  attrs.end());
      if (iter_args.size() == op->iter_args_.size() && return_vars.size() == op->return_vars_.size() &&
          body == op->body_ && attrs.size() == op->attrs_.size()) {
        return op;
      }
      return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_,
                                       std::move(iter_args), std::move(body), std::move(return_vars),
                                       op->span_, op->kind_, std::move(attrs), op->leading_comments_,
                                       op->hardware_loop_policy_);
    }
    StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
      std::vector<IterArgPtr> iter_args;
      std::vector<VarPtr> return_vars;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        if (As<SyncTokenType>(op->iter_args_[i]->GetType())) continue;
        iter_args.push_back(op->iter_args_[i]);
        if (i < op->return_vars_.size()) return_vars.push_back(op->return_vars_[i]);
      }
      auto body = VisitStmt(op->body_);
      if (iter_args.size() == op->iter_args_.size() && return_vars.size() == op->return_vars_.size() &&
          body == op->body_) {
        return op;
      }
      return std::make_shared<WhileStmt>(op->condition_, std::move(iter_args), std::move(body),
                                         std::move(return_vars), op->span_, op->leading_comments_);
    }
    StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
      std::vector<ExprPtr> values;
      values.reserve(op->value_.size());
      for (const auto& v : op->value_) {
        if (!As<SyncTokenType>(v->GetType())) values.push_back(v);
      }
      if (values.size() == op->value_.size()) return op;
      return std::make_shared<YieldStmt>(std::move(values), op->span_, op->leading_comments_);
    }
  } strip;
  auto body = strip.VisitStmt(function->body_);
  auto attrs = function->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& attr) {
                               return attr.first == kAttrAutosyncAnalysisPolicy ||
                                      attr.first == kAttrAutosyncAccessPrecision ||
                                      attr.first == kAttrAutosyncRealizationOrder ||
                                      attr.first == "autosync_child_attempt_count" ||
                                      attr.first == "autosync_retained_child_count" ||
                                      attr.first == "autosync_child_refusals" ||
                                      attr.first == kAttrPyptoAutosync ||
                                      attr.first == kAttrAutosyncFallbackReason;
                             }),
              attrs.end());
  return std::make_shared<Function>(function->name_, function->params_, function->param_directions_,
                                    function->return_types_, std::move(body), function->span_,
                                    function->func_type_, function->level_, function->role_,
                                    std::move(attrs), function->requires_runtime_binding_,
                                    function->ir_stage_);
}

LoweredLogicalSync LowerLogicalSync(const FunctionPtr& function, const std::string& target,
                                    const std::optional<std::vector<int>>& eligible_ids,
                                    const SyncParameterBounds& parameters,
                                    std::optional<std::pair<int64_t, int64_t>> blocks,
                                    const SyncSchedulePolicy& policy, SyncExternalPolicy external_policy,
                                    std::optional<int64_t> launch_count,
                                    const SyncNativeReservations& extra_reservations, bool dry_run_only) {
  CheckLogicalSync(function);
  RequireBufferSyncScalarLayout(function);
  RequireBufferRuntimeValid(function);
  if (!HasLogicalSync(function)) {
    if (dry_run_only) return {function, 0, 0, {}};
    PhysicalSyncRealization result;
    result.function = function;
    CheckPhysicalSyncResult(result.function);
    FinishSyncScheduleReceipt(function, result, policy);
    return {function, 0, 0, {std::move(result)}};
  }
  FindLogical(true).VisitFunction(function);
  if (HasOnlyClosedPrefixSyntax(function)) {
    if (auto closed = TryLowerClosedPrefixSync(function, target, eligible_ids, extra_reservations,
                                               parameters, blocks)) {
      if (dry_run_only) return {closed->function, closed->set_sites, closed->wait_sites, {}};
      CheckPhysicalSyncResult(closed->function);
      FinishSyncScheduleReceipt(closed->function, *closed, policy);
      return {closed->function, closed->set_sites, closed->wait_sites, {std::move(*closed)}};
    }
  }
  class Structure : public IRVisitor {
   public:
    bool loop = false;

   protected:
    void VisitStmt_(const ForStmtPtr&) override { loop = true; }
    void VisitExpr(const ExprPtr&) override {}
  } structure;
  structure.VisitFunction(function);
  if (structure.loop) {
    auto result = LowerStructuredLogicalSync(function, target, eligible_ids, parameters, blocks, policy,
                                             external_policy, launch_count, extra_reservations, dry_run_only);
    if (!dry_run_only) CheckPhysicalSyncResult(result.function);
    return result;
  }
  auto stripped = StripTokens().VisitFunction(function);
  // No storage-demand importer or GM partition authorizes this lowering.
  // Actual token uses, including manually supplied ordering, are authoritative.
  auto input = ImportSyncPayloads(stripped, target, parameters, blocks, external_policy);
  auto result =
      FiniteLowering(function, std::move(input), target, eligible_ids, extra_reservations).Run(dry_run_only);
  if (!dry_run_only) {
    CheckPhysicalSyncResult(result.function);
    FinishSyncScheduleReceipt(result.function, result, policy);
  }
  return {result.function, result.set_sites, result.wait_sites, {std::move(result)}};
}
}  // namespace pypto::ir::sync
