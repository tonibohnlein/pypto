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
#include <functional>
#include <iterator>
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
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/retained_sync_child.h"
#include "src/ir/transforms/utils/sync_loop_trip_proof.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_transform_common.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {

const char* RequireLocalSyncBarrier(PipeType pipe, FunctionType core) {
  const bool vector = core == FunctionType::AIV;
  switch (pipe) {
    case PipeType::S:
      return nullptr;
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

namespace {
constexpr size_t kRelayPipeCount = 7;  // MTE1..FIX.
using RelayMask = uint32_t;
using RelayMatrix = std::array<RelayMask, kRelayPipeCount>;

RelayMask RelayPipeBit(PipeType pipe) {
  if (pipe == PipeType::ALL) return (RelayMask{1} << kRelayPipeCount) - 1;
  const int index = static_cast<int>(pipe);
  if (index < 0 || static_cast<size_t>(index) >= kRelayPipeCount) return 0;
  return RelayMask{1} << static_cast<size_t>(index);
}

RelayMatrix IdentityRelay() {
  RelayMatrix m{};
  for (size_t i = 0; i < kRelayPipeCount; ++i) m[i] = RelayMask{1} << i;
  return m;
}

RelayMatrix IntersectRelay(const RelayMatrix& a, const RelayMatrix& b) {
  RelayMatrix out{};
  for (size_t i = 0; i < kRelayPipeCount; ++i) out[i] = a[i] & b[i];
  return out;
}

RelayMask RelayReach(const RelayMatrix& matrix, RelayMask sources) {
  RelayMask reached = 0;
  for (size_t i = 0; i < kRelayPipeCount; ++i) {
    if ((sources & (RelayMask{1} << i)) != 0) {
      reached |= matrix[i];
    }
  }
  return reached;
}

RelayMatrix ComposeRelay(const RelayMatrix& first, const RelayMatrix& second) {
  RelayMatrix out{};
  for (size_t i = 0; i < kRelayPipeCount; ++i) {
    out[i] = first[i] | RelayReach(second, first[i]);
  }
  return out;
}

void ApplyCapturedRelayEdge(RelayMatrix& matrix, const RelayMatrix& captured, RelayMask source,
                            RelayMask destination) {
  for (size_t i = 0; i < kRelayPipeCount; ++i) {
    if ((captured[i] & source) != 0) {
      matrix[i] |= destination;
    }
  }
}

struct SeqWaitEdge {
  size_t rec_idx;
  size_t wait_idx;
  PipeType src;
  PipeType dst;
};

struct SeqRelays {
  // Cut i is before child i, and cut n is after the last child.
  std::vector<RelayMatrix> prefix;  // Reachability from cut 0 to cut i.
  std::vector<RelayMatrix> suffix;  // Reachability from cut i to cut n.
};

SeqRelays ComputeSequenceRelays(const std::vector<RelayMatrix>& children,
                                const std::vector<SeqWaitEdge>& waits, size_t& suffix_visits) {
  const size_t n = children.size();
  const auto identity = IdentityRelay();
  SeqRelays out{std::vector<RelayMatrix>(n + 1, identity), std::vector<RelayMatrix>(n + 1, identity)};
  std::vector<const SeqWaitEdge*> capture_at(n, nullptr);
  for (const auto& wait : waits) {
    // Discover accepts only one wait per record on this lowering path.
    capture_at[wait.rec_idx] = &wait;
  }

  // Each child connects adjacent cuts. Each capture connects the cut before
  // its record to the cut after its wait. Both kinds of edges go forward.
  for (size_t i = 0; i < n; ++i) {
    const auto flow = children[i] == identity ? out.prefix[i] : ComposeRelay(out.prefix[i], children[i]);
    for (size_t p = 0; p < kRelayPipeCount; ++p) {
      out.prefix[i + 1][p] |= flow[p];
    }
    if (const auto* wait = capture_at[i]) {
      ApplyCapturedRelayEdge(out.prefix[wait->wait_idx + 1], out.prefix[i], RelayPipeBit(wait->src),
                             RelayPipeBit(wait->dst));
    }
  }

  // Reverse the cut traversal to summarize every suffix in one pass. The
  // capture edge contributes only at its record cut, so a suffix starting
  // after the record cannot acquire facts from its later wait.
  for (size_t cut = n; cut > 0; --cut) {
    ++suffix_visits;
    const size_t i = cut - 1;
    out.suffix[i] = children[i] == identity ? out.suffix[cut] : ComposeRelay(children[i], out.suffix[cut]);
    if (const auto* wait = capture_at[i]) {
      ++suffix_visits;
      const auto source = RelayPipeBit(wait->src);
      const auto reached = RelayReach(out.suffix[wait->wait_idx + 1], RelayPipeBit(wait->dst));
      for (size_t p = 0; p < kRelayPipeCount; ++p) {
        if ((source & (RelayMask{1} << p)) != 0) {
          out.suffix[i][p] |= reached;
        }
      }
    }
  }
  return out;
}

RelayMatrix ComputeChildLoopRelayMatrix(const StmtPtr& root, const SyncControlRangeProver& prover) {
  struct TokenSource {
    RelayMask src_bit = 0;
    RelayMatrix captured{};
  };
  using TokenMap = std::unordered_map<const Var*, std::vector<TokenSource>>;
  TokenMap tokens;

  std::function<RelayMatrix(const StmtPtr&, RelayMatrix)> walk = [&](const StmtPtr& stmt,
                                                                     RelayMatrix cur) -> RelayMatrix {
    if (!stmt) return cur;
    if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) cur = walk(child, cur);
      return cur;
    }
    if (const auto branch = As<IfStmt>(stmt)) {
      const TokenMap saved = tokens;
      const RelayMatrix then_cur = walk(branch->then_body_, cur);
      tokens = saved;
      const RelayMatrix else_cur = branch->else_body_ ? walk(*branch->else_body_, cur) : cur;
      tokens = saved;
      return IntersectRelay(then_cur, else_cur);
    }
    if (const auto loop = As<ForStmt>(stmt)) {
      if (!prover.IsPositiveTripCompatible(loop.get())) return cur;
      const TokenMap saved = tokens;
      const RelayMatrix out = walk(loop->body_, cur);
      tokens = saved;
      return out;
    }
    if (As<WhileStmt>(stmt)) {
      return cur;
    }
    if (const auto assign = As<AssignStmt>(stmt)) {
      if (As<SyncTokenType>(assign->var_->GetType())) {
        if (const auto rhs = AsVarLike(assign->value_)) {
          const auto found = tokens.find(rhs.get());
          tokens[assign->var_.get()] = found == tokens.end() ? std::vector<TokenSource>{} : found->second;
        } else if (const auto call = As<Call>(assign->value_)) {
          if (IsOp(call, "sync.record")) {
            if (const auto p = TryGetSyncPipeKwarg(call, "pipe")) {
              tokens[assign->var_.get()] = {{RelayPipeBit(*p), cur}};
            }
          } else if (IsOp(call, "sync.completed")) {
            tokens[assign->var_.get()] = {};
          } else if (IsOp(call, "sync.join")) {
            std::vector<TokenSource> joined;
            for (const auto& arg : call->args_) {
              const auto v = AsVarLike(arg);
              const auto found = v ? tokens.find(v.get()) : tokens.end();
              if (found != tokens.end()) {
                joined.insert(joined.end(), found->second.begin(), found->second.end());
              }
            }
            tokens[assign->var_.get()] = std::move(joined);
          }
        }
      }
      return cur;
    }
    if (const auto eval = As<EvalStmt>(stmt)) {
      if (const auto call = As<Call>(eval->expr_); call && IsOp(call, "sync.wait") && !call->args_.empty()) {
        const auto v = AsVarLike(call->args_[0]);
        const auto found = v ? tokens.find(v.get()) : tokens.end();
        const auto dst = TryGetSyncPipeKwarg(call, "dst_pipe");
        if (found != tokens.end() && dst) {
          for (const auto& src : found->second) {
            ApplyCapturedRelayEdge(cur, src.captured, src.src_bit, RelayPipeBit(*dst));
          }
        }
      }
      return cur;
    }
    return cur;
  };

  return walk(root, IdentityRelay());
}

PipeType Pipe(const CallPtr& operation, const char* key) {
  return RequireSyncPipeKwarg(operation, key, "Internal error: missing pipeline attribute");
}

CallPtr SyncRecord(const StmtPtr& statement) {
  const auto assignment = As<AssignStmt>(statement);
  const auto call = assignment ? As<Call>(assignment->value_) : nullptr;
  return call && IsOp(call, "sync.record") ? call : nullptr;
}

CallPtr SyncWait(const StmtPtr& statement) {
  const auto evaluation = As<EvalStmt>(statement);
  const auto call = evaluation ? As<Call>(evaluation->expr_) : nullptr;
  return call && IsOp(call, "sync.wait") ? call : nullptr;
}

// Local same-pipe hardware barrier (`record(p); wait(p)`).
struct LocalCompletion {
  PipeType pipe;
  std::array<StmtPtr, 2> cuts;
  const SeqStmts* parent_seq = nullptr;
  size_t rec_idx = 0;
  size_t wait_idx = 0;
};

// #3375 1-way logical synchronization edge (`tok = record(src); ...; wait(tok, dst)`).
struct OneWayTransaction {
  PipeType source;
  PipeType destination;
  StmtPtr record_cut;
  StmtPtr wait_cut;
  std::vector<const Stmt*> loop_stack;
  uint32_t loop_scope_id = 0;
  bool needs_return_ack = false;
  RelayMask loop_pre_reach = 0;
  RelayMask loop_post_reach = 0;
  int forward_id = -1;
  int return_id = -1;
  uint64_t record_seq = 0;
  const SeqStmts* parent_seq = nullptr;
  size_t rec_idx = 0;
  size_t wait_idx = 0;
};

class Discover : public IRVisitor {
 public:
  explicit Discover(const SyncControlRangeProver* prover = nullptr) : prover_(prover) {}
  std::vector<LocalCompletion> local;
  std::vector<OneWayTransaction> one_way;
  std::vector<StmtPtr> erased_waits;
  std::vector<StmtPtr> erased_defs;
  std::vector<ForStmtPtr> children;
  std::map<const ForStmt*, RelayMatrix> entries;
  std::unordered_set<const ForStmt*> loop_nested_children;
  std::map<const Stmt*, uint32_t> loop_scope_ids;
  std::unordered_set<const ForStmt*> spmd_one_trip_loops;
  bool unsupported_syntax = false;
  bool has_closed_pair = false;
  bool has_for = false;
  bool has_while = false;
  bool has_if = false;
  bool has_payload = false;

  void VisitExpr(const ExprPtr&) override {}

  void VisitFunction(const FunctionPtr& function) override {
    IRVisitor::VisitFunction(function);
    for (const auto& [var, pending] : pending_records_) {
      (void)var;
      if (!pending.used) unsupported_syntax = true;
    }
  }

 protected:
  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (const auto call = As<Call>(eval->expr_)) {
      if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None) {
        unsupported_syntax = true;
      } else if (!IsSyncNonPayloadCall(call)) {
        has_payload = true;
      }
    }
    IRVisitor::VisitStmt_(eval);
  }
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (As<SyncTokenType>(assign->var_->GetType())) {
      unsupported_syntax = true;
    } else {
      ssa_defs_[assign->var_.get()] = assign->value_;
    }
    if (const auto call = As<Call>(assign->value_)) {
      if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None) {
        unsupported_syntax = true;
      } else if (!IsSyncNonPayloadCall(call)) {
        has_payload = true;
      }
    }
    IRVisitor::VisitStmt_(assign);
  }
  void VisitStmt_(const IfStmtPtr& branch) override {
    for (const auto& rv : branch->return_vars_) {
      if (ContainsSyncTokenType(rv->GetType())) unsupported_syntax = true;
    }
    has_if = true;
    control_stack_.push_back(branch.get());
    VisitExpr(branch->condition_);
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
    control_stack_.pop_back();
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loop->HasAttr(kClosedSyncChildAttr)) {
      CHECK(loop->GetAttr<int>(kClosedSyncChildAttr) == 1) << "Unsupported closed-child routing version";
      if (!loop_stack_.empty()) loop_nested_children.insert(loop.get());
      children.push_back(loop);
      return;  // Actual child tokens are independently parsed below.
    }
    for (const auto& arg : loop->iter_args_) {
      if (ContainsSyncTokenType(arg->GetType())) unsupported_syntax = true;
    }
    for (const auto& rv : loop->return_vars_) {
      if (ContainsSyncTokenType(rv->GetType())) unsupported_syntax = true;
    }
    has_for = true;
    if (IsSpmdAtMostOneTripFor(loop, ssa_defs_) ||
        (prover_ && !As<ConstInt>(loop->start_) && prover_->IsAtMostOneTrip(loop.get()))) {
      spmd_one_trip_loops.insert(loop.get());
      control_stack_.push_back(loop.get());
      IRVisitor::VisitStmt_(loop);
      control_stack_.pop_back();
      return;
    }
    loop_stack_.push_back(loop.get());
    control_stack_.push_back(loop.get());
    const uint32_t scope_id = ++next_loop_scope_id_;
    loop_scope_ids.emplace(loop.get(), scope_id);
    loop_scope_stack_.push_back(scope_id);
    IRVisitor::VisitStmt_(loop);
    loop_scope_stack_.pop_back();
    control_stack_.pop_back();
    loop_stack_.pop_back();
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& arg : loop->iter_args_) {
      if (ContainsSyncTokenType(arg->GetType())) unsupported_syntax = true;
    }
    for (const auto& rv : loop->return_vars_) {
      if (ContainsSyncTokenType(rv->GetType())) unsupported_syntax = true;
    }
    has_while = true;
    loop_stack_.push_back(loop.get());
    control_stack_.push_back(loop.get());
    const uint32_t scope_id = ++next_loop_scope_id_;
    loop_scope_ids.emplace(loop.get(), scope_id);
    loop_scope_stack_.push_back(scope_id);
    IRVisitor::VisitStmt_(loop);
    loop_scope_stack_.pop_back();
    control_stack_.pop_back();
    loop_stack_.pop_back();
  }

  void VisitStmt_(const SeqStmtsPtr& sequence) override {
    const auto& statements = sequence->stmts_;
    // An entry rearm orders work issued before the contiguous handoff starts.
    // A token recorded before an intervening payload or child cannot prove it.
    RelayMatrix preceding_relay = IdentityRelay();
    size_t relay_start = 0;
    const auto reset_relay = [&](size_t index) {
      relay_start = index + 1;
      preceding_relay = IdentityRelay();
    };
    for (size_t index = 0; index < statements.size(); ++index) {
      if (const auto child = As<ForStmt>(statements[index]); child && child->HasAttr(kClosedSyncChildAttr)) {
        entries.emplace(child.get(), preceding_relay);
      }
      if (const auto assign = As<AssignStmt>(statements[index]);
          assign && As<SyncTokenType>(assign->var_->GetType())) {
        if (const auto call = As<Call>(assign->value_)) {
          if (IsOp(call, "sync.completed")) {
            completed_tokens_.insert(assign->var_.get());
            erased_defs.push_back(statements[index]);
            reset_relay(index);
            continue;
          }
        } else if (const auto alias = As<Var>(assign->value_);
                   alias && completed_tokens_.count(alias.get())) {
          completed_tokens_.insert(assign->var_.get());
          erased_defs.push_back(statements[index]);
          reset_relay(index);
          continue;
        }
      }
      if (const auto w = SyncWait(statements[index])) {
        const auto token = As<Var>(w->args_.empty() ? ExprPtr{} : w->args_[0]);
        if (token && completed_tokens_.count(token.get())) {
          erased_waits.push_back(statements[index]);
          continue;
        }
      }

      // Record whether this function contains an explicit 4-command closed pair
      // so HasOnlyClosedPrefixSyntax keeps closed-prefix ownership without
      // bypassing general 1-way allocation.
      if (index + 3 < statements.size()) {
        const auto r0 = SyncRecord(statements[index]);
        const auto w0 = SyncWait(statements[index + 1]);
        const auto r1 = SyncRecord(statements[index + 2]);
        const auto w1 = SyncWait(statements[index + 3]);
        if (r0 && w0 && r1 && w1) {
          const auto src = Pipe(r0, "pipe");
          const auto dst = Pipe(w0, "dst_pipe");
          const auto t0 = As<AssignStmt>(statements[index])->var_;
          const auto t1 = As<AssignStmt>(statements[index + 2])->var_;
          if (src != dst && Pipe(r1, "pipe") == dst && Pipe(w1, "dst_pipe") == src && w0->args_.size() == 1 &&
              w0->args_[0] == t0 && w1->args_.size() == 1 && w1->args_[0] == t1) {
            has_closed_pair = true;
          }
        }
      }

      // Recognize local two-command same-pipe barrier patterns first.
      if (index + 1 < statements.size()) {
        const auto r0 = SyncRecord(statements[index]);
        const auto w0 = SyncWait(statements[index + 1]);
        if (r0 && w0) {
          const auto src = Pipe(r0, "pipe");
          const auto dst = Pipe(w0, "dst_pipe");
          const auto t0 = As<AssignStmt>(statements[index])->var_;
          if (src == dst && w0->args_.size() == 1 && w0->args_[0] == t0) {
            local.push_back(
                {src, {statements[index], statements[index + 1]}, sequence.get(), index, index + 1});
            index += 1;
            continue;
          }
        }
      }

      if (const auto rec = SyncRecord(statements[index])) {
        const auto var = As<AssignStmt>(statements[index])->var_;
        auto it = pending_records_.find(var.get());
        if (it != pending_records_.end() && !it->second.used) {
          unsupported_syntax = true;
        }
        pending_records_[var.get()] = {Pipe(rec, "pipe"),
                                       statements[index],
                                       control_stack_,
                                       sequence.get(),
                                       index,
                                       preceding_relay,
                                       false};
        continue;
      }
      if (const auto w = SyncWait(statements[index])) {
        const auto token = As<Var>(w->args_.empty() ? ExprPtr{} : w->args_[0]);
        if (!token) {
          unsupported_syntax = true;
          reset_relay(index);
          continue;
        }
        const auto rec_it = pending_records_.find(token.get());
        if (rec_it == pending_records_.end() || rec_it->second.used ||
            rec_it->second.control_stack != control_stack_) {
          unsupported_syntax = true;
          reset_relay(index);
          continue;
        }
        rec_it->second.used = true;
        const auto src = rec_it->second.pipe;
        const auto dst = Pipe(w, "dst_pipe");
        if (src == dst) {
          const bool same_seq = rec_it->second.seq == sequence.get();
          local.push_back({src,
                           {rec_it->second.stmt, statements[index]},
                           same_seq ? sequence.get() : nullptr,
                           same_seq ? rec_it->second.index_in_seq : 0,
                           same_seq ? index : 0});
        } else {
          const bool same_seq = rec_it->second.seq == sequence.get();
          one_way.push_back({src, dst, rec_it->second.stmt, statements[index], loop_stack_,
                             loop_scope_stack_.back(), false, 0, 0, -1, -1, 0,
                             same_seq ? sequence.get() : nullptr, same_seq ? rec_it->second.index_in_seq : 0,
                             same_seq ? index : 0});
          if (same_seq && rec_it->second.index_in_seq >= relay_start) {
            ApplyCapturedRelayEdge(preceding_relay, rec_it->second.captured_relay, RelayPipeBit(src),
                                   RelayPipeBit(dst));
          }
        }
        continue;
      }

      reset_relay(index);
      VisitStmt(statements[index]);
    }
  }

 private:
  struct PendingRecord {
    PipeType pipe;
    StmtPtr stmt;
    std::vector<const Stmt*> control_stack;
    const SeqStmts* seq = nullptr;
    size_t index_in_seq = 0;
    RelayMatrix captured_relay{};
    bool used = false;
  };
  const SyncControlRangeProver* prover_ = nullptr;
  std::set<const Var*> completed_tokens_;
  std::unordered_map<const Var*, PendingRecord> pending_records_;
  std::unordered_map<const Var*, ExprPtr> ssa_defs_;
  std::vector<const Stmt*> loop_stack_;
  std::vector<const Stmt*> control_stack_;
  std::vector<uint32_t> loop_scope_stack_{0};
  uint32_t next_loop_scope_id_ = 0;
};

class NativeOperations : public IRVisitor {
 public:
  NativeOperations(FunctionType core, std::string target, bool track_payload_pipes = false)
      : core_(core), target_(std::move(target)), track_payload_pipes_(track_payload_pipes) {
    CHECK(core == FunctionType::AIC || core == FunctionType::AIV)
        << "Closed prefix protocols require one AIC/AIV invocation owner";
  }
  std::vector<CallPtr> payloads;
  std::vector<CallPtr> all_native_payloads;
  std::unordered_map<const Stmt*, RelayMask> subtree_payload_pipes;

 protected:
  void VisitExpr(const ExprPtr& expression) override {
    if (visited_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
  }
  void VisitStmt(const StmtPtr& statement) override {
    if (!statement) return;
    if (in_child_scope_ == 0) {
      CHECK(As<SeqStmts>(statement) || As<AssignStmt>(statement) || As<EvalStmt>(statement) ||
            As<IfStmt>(statement) || As<ForStmt>(statement) || As<WhileStmt>(statement) ||
            As<YieldStmt>(statement) || As<ReturnStmt>(statement))
          << "Closed prefix protocols require local structured sequential control";
    }
    stmt_pipe_stack_.push_back(0);
    IRVisitor::VisitStmt(statement);
    const RelayMask mask = stmt_pipe_stack_.back();
    stmt_pipe_stack_.pop_back();
    if (mask != 0) {
      subtree_payload_pipes[statement.get()] = mask;
      if (!stmt_pipe_stack_.empty()) stmt_pipe_stack_.back() |= mask;
    }
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loop->HasAttr(kClosedSyncChildAttr)) {
      ++in_child_scope_;
      VisitStmt(loop->body_);
      --in_child_scope_;
      return;
    }
    if (in_child_scope_ == 0) {
      CHECK(loop->kind_ == ForKind::Sequential)
          << "Closed prefix reuse requires a sequential loop after scheduling";
      for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    }
    VisitStmt(loop->body_);
  }
  void VisitExpr_(const SubmitPtr&) override {
    CHECK(false) << "Closed prefix protocols require an intra-leaf execution owner";
  }
  void VisitExpr_(const CallPtr& call) override {
    if (GetLogicalSyncKind(call->op_) == LogicalSyncKind::None) {
      if (in_child_scope_ == 0) {
        CHECK(call->attrs_.empty())
            << "Closed prefix operation attributes require an explicit native execution contract";
        if (const auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
          CHECK(descriptor->SignalFree()) << "Closed prefix descriptor has unqualified event resources";
        } else {
          payloads.push_back(call);
          all_native_payloads.push_back(call);
          if (track_payload_pipes_ && !stmt_pipe_stack_.empty()) {
            stmt_pipe_stack_.back() |= RelayPipeBit(
                backend::QualifyBufferSyncOperation(call, core_, target_).Implementation().Pipe());
          }
        }
      } else if (!backend::QualifyBufferSyncNonPayload(call, target_)) {
        all_native_payloads.push_back(call);
        if (track_payload_pipes_ && !stmt_pipe_stack_.empty()) {
          stmt_pipe_stack_.back() |=
              RelayPipeBit(backend::QualifyBufferSyncOperation(call, core_, target_).Implementation().Pipe());
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  FunctionType core_;
  std::string target_;
  bool track_payload_pipes_ = false;
  size_t in_child_scope_ = 0;
  std::vector<RelayMask> stmt_pipe_stack_;
  std::set<const Expr*> visited_;
};

enum class CommandKind { Erase, SetFlag, WaitFlag, Barrier, SetWaitAck };
struct AssignedCommand {
  CommandKind kind;
  SyncDirection direction;
  int id = -1;
  SyncDirection ack_direction{};
  int ack_id = -1;
};

class Assignment {
 public:
  // The loop-closure decision determines which physical directions exist,
  // independently of their lane assignment. Retained children use the same
  // decision before the parent plans its entry protocols.
  static size_t PlanOneWayClosure(const FunctionPtr& owner, Discover& discovered,
                                  const SyncControlRangeProver& prover) {
    // Fall back to the 4-command protocol when a 1-way loop edge lacks an
    // unconditional reverse cycle.
    std::unordered_map<const SeqStmts*, std::vector<SeqWaitEdge>> seq_wait_edges;
    for (const auto& ow : discovered.one_way) {
      if (ow.parent_seq) {
        seq_wait_edges[ow.parent_seq].push_back({ow.rec_idx, ow.wait_idx, ow.source, ow.destination});
      }
    }

    // Precompute bottom-up relay summaries and cut tables in O(statements + waits)
    // so each transaction checks loop backedge closure in O(nesting_depth).
    struct AncestorLink {
      const Stmt* parent = nullptr;
      size_t index_in_seq = 0;
    };
    std::unordered_map<const Stmt*, SeqRelays> seq_relays;
    std::unordered_map<const Stmt*, AncestorLink> ancestors;
    size_t suffix_visits = 0;

    std::function<RelayMatrix(const StmtPtr&)> precompute = [&](const StmtPtr& s) -> RelayMatrix {
      if (!s) return IdentityRelay();
      if (const auto seq = As<SeqStmts>(s)) {
        const size_t n = seq->stmts_.size();
        std::vector<RelayMatrix> child_sums(n, IdentityRelay());
        for (size_t i = 0; i < n; ++i) {
          const auto& c = seq->stmts_[i];
          if (c) ancestors[c.get()] = {seq.get(), i};
          child_sums[i] = precompute(c);
        }
        const auto found = seq_wait_edges.find(seq.get());
        const std::vector<SeqWaitEdge> no_waits;
        SeqRelays sr = ComputeSequenceRelays(
            child_sums, found == seq_wait_edges.end() ? no_waits : found->second, suffix_visits);
        const RelayMatrix total = sr.prefix[n];
        seq_relays.emplace(seq.get(), std::move(sr));
        return total;
      }
      if (const auto branch = As<IfStmt>(s)) {
        if (branch->then_body_) ancestors[branch->then_body_.get()] = {branch.get(), 0};
        if (branch->else_body_) ancestors[(*branch->else_body_).get()] = {branch.get(), 0};
        const RelayMatrix then_sum = precompute(branch->then_body_);
        const RelayMatrix else_sum = branch->else_body_ ? precompute(*branch->else_body_) : IdentityRelay();
        return IntersectRelay(then_sum, else_sum);
      }
      if (const auto loop = As<ForStmt>(s)) {
        if (loop->HasAttr(kClosedSyncChildAttr)) {
          return prover.IsPositiveTripCompatible(loop.get())
                     ? ComputeChildLoopRelayMatrix(loop->body_, prover)
                     : IdentityRelay();
        }
        if (loop->body_) ancestors[loop->body_.get()] = {loop.get(), 0};
        const RelayMatrix body_relay = precompute(loop->body_);
        return prover.IsPositiveTrip(loop.get()) ? body_relay : IdentityRelay();
      }
      if (const auto loop = As<WhileStmt>(s)) {
        if (loop->body_) ancestors[loop->body_.get()] = {loop.get(), 0};
        precompute(loop->body_);
        return IdentityRelay();
      }
      return IdentityRelay();
    };
    const bool has_loop_one_way =
        std::any_of(discovered.one_way.begin(), discovered.one_way.end(),
                    [](const OneWayTransaction& ow) { return !ow.loop_stack.empty(); });
    if (has_loop_one_way) precompute(owner->body_);

    for (auto& ow : discovered.one_way) {
      if (ow.loop_stack.empty()) {
        ow.needs_return_ack = false;
        continue;
      }
      bool closed_all = true;
      RelayMask pre_reach_all = (RelayMask{1} << kRelayPipeCount) - 1;
      RelayMask post_reach_all = (RelayMask{1} << kRelayPipeCount) - 1;
      const Stmt* cur_rec = ow.record_cut.get();
      const Stmt* cur_wt = ow.wait_cut.get();
      RelayMatrix rec_prefix = IdentityRelay();
      RelayMatrix wt_suffix = IdentityRelay();
      for (size_t li = ow.loop_stack.size(); li-- > 0;) {
        const Stmt* target_loop = ow.loop_stack[li];
        auto advance = [&](const Stmt*& cur, RelayMatrix& mat, bool is_prefix) -> bool {
          while (cur != target_loop) {
            const auto ait = ancestors.find(cur);
            if (ait == ancestors.end()) return false;
            const AncestorLink& link = ait->second;
            if (const auto sit = seq_relays.find(link.parent); sit != seq_relays.end()) {
              mat = is_prefix ? ComposeRelay(sit->second.prefix[link.index_in_seq], mat)
                              : ComposeRelay(mat, sit->second.suffix[link.index_in_seq + 1]);
            }
            cur = link.parent;
          }
          return true;
        };
        if (!advance(cur_rec, rec_prefix, true) || !advance(cur_wt, wt_suffix, false)) {
          closed_all = false;
          pre_reach_all = 0;
          post_reach_all = 0;
          break;
        }
        RelayMask pre_mask = 0;
        const RelayMask src_bit = RelayPipeBit(ow.source);
        for (size_t p = 0; p < kRelayPipeCount; ++p) {
          if ((rec_prefix[p] & src_bit) != 0) pre_mask |= (RelayMask{1} << p);
        }
        pre_reach_all &= pre_mask;
        post_reach_all &= wt_suffix[static_cast<size_t>(ow.destination)];
        const RelayMatrix back = ComposeRelay(wt_suffix, rec_prefix);
        if ((back[static_cast<size_t>(ow.destination)] & RelayPipeBit(ow.source)) == 0) {
          closed_all = false;
          break;
        }
      }
      ow.needs_return_ack = !closed_all;
      ow.loop_pre_reach = closed_all ? pre_reach_all : 0;
      ow.loop_post_reach = closed_all ? post_reach_all : 0;
    }
    return suffix_visits;
  }

  Assignment(const FunctionPtr& owner, Discover& discovered, const NativeOperations& native_ops,
             const std::string& target, const std::optional<std::vector<int>>& eligible,
             const SyncNativeReservations& extra_reservations, const SyncControlRangeProver& prover) {
    const auto& native = backend::RequireLocalSyncExecutionContract(target);
    CHECK(native.set_completes_source_prefix && native.wait_consumes_before_advance)
        << "Closed prefix protocols require ordered consuming event commands";
    reservations_ = SyncNativeClaims(owner, native_ops.all_native_payloads, target);
    for (const auto& [direction, ids] : reservations_)
      for (int id : ids) native_claimed_.emplace_back(direction.first, direction.second, id);
    for (const auto& [direction, ids] : extra_reservations)
      reservations_[direction].insert(ids.begin(), ids.end());

    suffix_visits_ = PlanOneWayClosure(owner, discovered, prover);

    // Allocate directional IDs for all 1-way transactions (both 2-cmd and 4-cmd fallback),
    // tracking in-flight lifetimes so concurrent operations never share a flag ID and
    // reusing non-overlapping idle IDs when causally safe.
    struct ActiveId {
      int id;
      bool in_flight;
      RelayMask reach_from_dst;
      bool acyclic;
      RelayMask loop_pre_reach = 0;
      RelayMask loop_post_reach = 0;
      uint32_t loop_scope_id = 0;
      uint64_t wait_seq = 0;
      std::array<uint64_t, kRelayPipeCount> pipe_reach_seq{};
      bool is_ack = false;

      void AddReach(PipeType pipe, uint64_t seq) {
        const RelayMask bit = RelayPipeBit(pipe);
        if (bit == 0) return;
        const int idx = static_cast<int>(pipe);
        if ((reach_from_dst & bit) == 0) {
          reach_from_dst |= bit;
          if (idx >= 0 && static_cast<size_t>(idx) < kRelayPipeCount) {
            pipe_reach_seq[static_cast<size_t>(idx)] = seq;
          }
        }
      }
      bool ReachedAtOrBefore(PipeType pipe, uint64_t seq) const {
        const RelayMask bit = RelayPipeBit(pipe);
        if ((reach_from_dst & bit) == 0) return false;
        const int idx = static_cast<int>(pipe);
        return idx >= 0 && static_cast<size_t>(idx) < kRelayPipeCount &&
               pipe_reach_seq[static_cast<size_t>(idx)] <= seq;
      }
      void IntersectReach(const ActiveId& other) {
        reach_from_dst &= other.reach_from_dst;
        for (size_t p = 0; p < kRelayPipeCount; ++p) {
          if ((reach_from_dst & (RelayMask{1} << p)) != 0) {
            pipe_reach_seq[p] = std::max(pipe_reach_seq[p], other.pipe_reach_seq[p]);
          } else {
            pipe_reach_seq[p] = 0;
          }
        }
      }
    };
    uint64_t next_seq = 1;
    std::map<SyncDirection, std::vector<ActiveId>> active_by_dir;
    std::map<SyncDirection, std::vector<int>> one_way_pools;
    auto pool_for = [&](SyncDirection dir) -> const std::vector<int>& {
      auto it = one_way_pools.find(dir);
      if (it != one_way_pools.end()) return it->second;
      return one_way_pools.emplace(dir, SyncEligibleIds(owner, target, dir, eligible, reservations_))
          .first->second;
    };
    std::map<const Stmt*, OneWayTransaction*> wait_to_ow;
    std::map<const Stmt*, OneWayTransaction*> record_to_ow;
    for (auto& ow : discovered.one_way) {
      record_to_ow[ow.record_cut.get()] = &ow;
      wait_to_ow[ow.wait_cut.get()] = &ow;
    }

    auto merge_loop_exit = [&](const std::map<SyncDirection, std::vector<ActiveId>>& saved,
                               uint32_t outer_scope_id, uint32_t inner_scope_id, bool positive_trip) {
      for (auto& [dir, vec] : active_by_dir) {
        const auto sit = saved.find(dir);
        for (auto& entry : vec) {
          const ActiveId* matched_se = nullptr;
          if (sit != saved.end()) {
            for (const auto& se : sit->second) {
              if (se.id == entry.id) {
                matched_se = &se;
                break;
              }
            }
          }
          if (outer_scope_id != 0) {
            if (matched_se) {
              const bool touched =
                  entry.wait_seq != matched_se->wait_seq || entry.in_flight != matched_se->in_flight;
              if (touched) {
                if (positive_trip && !entry.in_flight && entry.acyclic) {
                  entry.loop_scope_id = outer_scope_id;
                  entry.loop_pre_reach = 0;
                  entry.loop_post_reach = 0;
                } else if (!positive_trip && !entry.in_flight && entry.acyclic && !matched_se->in_flight &&
                           matched_se->acyclic) {
                  entry.IntersectReach(*matched_se);
                  entry.acyclic = true;
                  entry.loop_scope_id = (matched_se->loop_scope_id == 0) ? 0 : outer_scope_id;
                  entry.loop_pre_reach = 0;
                  entry.loop_post_reach = 0;
                } else {
                  entry.reach_from_dst = 0;
                  entry.pipe_reach_seq.fill(0);
                  entry.acyclic = false;
                  entry.loop_pre_reach = 0;
                  entry.loop_post_reach = 0;
                  entry.loop_scope_id = outer_scope_id;
                }
              } else {
                if (!positive_trip) {
                  entry.IntersectReach(*matched_se);
                  entry.acyclic = entry.acyclic && matched_se->acyclic;
                  entry.loop_pre_reach &= matched_se->loop_pre_reach;
                  entry.loop_post_reach &= matched_se->loop_post_reach;
                }
                if (entry.loop_scope_id == inner_scope_id) {
                  entry.loop_scope_id = outer_scope_id;
                  entry.loop_pre_reach = 0;
                  entry.loop_post_reach = 0;
                }
              }
              entry.in_flight = entry.in_flight || matched_se->in_flight;
              entry.wait_seq = std::max(entry.wait_seq, matched_se->wait_seq);
            } else {
              if (!entry.in_flight && entry.acyclic) {
                entry.loop_scope_id = outer_scope_id;
                entry.loop_pre_reach = 0;
                entry.loop_post_reach = 0;
              } else {
                entry.reach_from_dst = 0;
                entry.pipe_reach_seq.fill(0);
                entry.acyclic = false;
                entry.loop_pre_reach = 0;
                entry.loop_post_reach = 0;
                entry.loop_scope_id = outer_scope_id;
              }
            }
          } else {
            if (!matched_se) {
              entry.in_flight = false;
              entry.AddReach(dir.second, entry.wait_seq);
              entry.acyclic = true;
              entry.loop_pre_reach = 0;
              entry.loop_post_reach = 0;
              entry.loop_scope_id = 0;
            } else {
              if (entry.wait_seq != matched_se->wait_seq) {
                entry.AddReach(dir.second, entry.wait_seq);
              }
              if (!positive_trip) {
                entry.IntersectReach(*matched_se);
              }
              entry.in_flight = entry.in_flight || matched_se->in_flight;
              entry.acyclic = positive_trip ? true : matched_se->acyclic;
              entry.loop_pre_reach = positive_trip ? 0 : matched_se->loop_pre_reach;
              entry.loop_post_reach = positive_trip ? 0 : matched_se->loop_post_reach;
              entry.loop_scope_id = 0;
              entry.wait_seq = std::max(entry.wait_seq, matched_se->wait_seq);
            }
          }
        }
      }
    };

    size_t loop_depth = 0;
    std::vector<uint32_t> walk_loop_scopes{0};
    std::vector<std::map<SyncDirection, std::set<int>>> branch_disjoint_ids_;
    auto is_branch_blocked = [&](SyncDirection dir, int id) -> bool {
      for (const auto& frame : branch_disjoint_ids_) {
        const auto fit = frame.find(dir);
        if (fit != frame.end() && fit->second.count(id)) return true;
      }
      return false;
    };

    std::function<void(const StmtPtr&)> allocate_walk = [&](const StmtPtr& stmt) {
      if (!stmt) return;
      if (const auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) allocate_walk(child);
        return;
      }
      if (const auto branch = As<IfStmt>(stmt)) {
        auto saved = active_by_dir;
        allocate_walk(branch->then_body_);
        auto then_active = active_by_dir;
        active_by_dir = saved;
        if (branch->else_body_) {
          if (loop_depth > 0) {
            std::map<SyncDirection, std::set<int>> then_touched;
            for (const auto& [dir, then_vec] : then_active) {
              const auto sit = saved.find(dir);
              for (const auto& te : then_vec) {
                const ActiveId* matched_se = nullptr;
                if (sit != saved.end()) {
                  for (const auto& se : sit->second) {
                    if (se.id == te.id) {
                      matched_se = &se;
                      break;
                    }
                  }
                }
                if (!matched_se || te.wait_seq != matched_se->wait_seq ||
                    te.in_flight != matched_se->in_flight) {
                  then_touched[dir].insert(te.id);
                }
              }
            }
            branch_disjoint_ids_.push_back(std::move(then_touched));
            allocate_walk(*branch->else_body_);
            branch_disjoint_ids_.pop_back();
          } else {
            allocate_walk(*branch->else_body_);
          }
        }
        for (auto& [dir, vec] : active_by_dir) {
          const auto tit = then_active.find(dir);
          if (tit == then_active.end()) {
            // Every entry in `vec` was freshly allocated in `else_body_` (none
            // existed in `saved[dir]` or `then_active[dir]`). On the `then` path
            // these IDs were never issued, so they retain their `else_body_` reachability.
            continue;
          }
          for (auto& entry : vec) {
            const ActiveId* matched_te = nullptr;
            for (const auto& te : tit->second) {
              if (te.id == entry.id) {
                matched_te = &te;
                break;
              }
            }
            if (matched_te) {
              entry.in_flight = entry.in_flight || matched_te->in_flight;
              entry.IntersectReach(*matched_te);
              entry.acyclic = entry.acyclic && matched_te->acyclic;
              const bool same_scope = entry.loop_scope_id == matched_te->loop_scope_id;
              entry.loop_pre_reach = same_scope ? (entry.loop_pre_reach & matched_te->loop_pre_reach) : 0;
              entry.loop_post_reach = same_scope ? (entry.loop_post_reach & matched_te->loop_post_reach) : 0;
              entry.wait_seq = std::max(entry.wait_seq, matched_te->wait_seq);
              if (matched_te->loop_scope_id != 0) entry.loop_scope_id = matched_te->loop_scope_id;
            }
            // If `!matched_te`, `entry` was freshly allocated in `else_body_` and
            // never touched on the `then` path; retain its `else_body_` reachability.
          }
          for (const auto& te : tit->second) {
            if (std::none_of(vec.begin(), vec.end(),
                             [&](const ActiveId& entry) { return entry.id == te.id; })) {
              // Freshly allocated in `then_body_` and untouched in `else_body_`.
              vec.push_back(te);
            }
          }
        }
        for (const auto& [dir, then_vec] : then_active) {
          if (active_by_dir.count(dir)) continue;
          auto& dst_vec = active_by_dir[dir];
          for (const auto& te : then_vec) {
            // Freshly allocated in `then_body_` and untouched in `else_body_`.
            dst_vec.push_back(te);
          }
        }
        return;
      }
      if (const auto loop = As<ForStmt>(stmt)) {
        if (loop->HasAttr(kClosedSyncChildAttr)) {
          if (prover.IsPositiveTripCompatible(loop.get())) {
            const RelayMatrix child_relay = ComputeChildLoopRelayMatrix(loop->body_, prover);
            const uint64_t seq_now = next_seq++;
            for (auto& [dir, vec] : active_by_dir) {
              for (auto& entry : vec) {
                if (entry.in_flight) continue;
                RelayMask extra = 0;
                for (size_t p = 0; p < kRelayPipeCount; ++p) {
                  if ((entry.reach_from_dst & (RelayMask{1} << p)) != 0) extra |= child_relay[p];
                }
                for (size_t q = 0; q < kRelayPipeCount; ++q) {
                  if ((extra & (RelayMask{1} << q)) != 0) entry.AddReach(static_cast<PipeType>(q), seq_now);
                }
              }
            }
          }
          return;
        }
        const bool positive_trip = prover.IsPositiveTrip(loop.get());
        const bool at_most_one = prover.IsAtMostOneTrip(loop.get());
        if (positive_trip && at_most_one) {
          allocate_walk(loop->body_);
          return;
        }
        if (discovered.spmd_one_trip_loops.count(loop.get()) || (!positive_trip && at_most_one)) {
          auto saved = active_by_dir;
          allocate_walk(loop->body_);
          for (auto& [dir, vec] : active_by_dir) {
            const auto sit = saved.find(dir);
            if (sit == saved.end()) {
              // Freshly allocated inside a <=1-trip loop; untouched on 0-trip path.
              continue;
            }
            for (auto& entry : vec) {
              const ActiveId* matched_se = nullptr;
              for (const auto& se : sit->second) {
                if (se.id == entry.id) {
                  matched_se = &se;
                  break;
                }
              }
              if (matched_se) {
                entry.in_flight = entry.in_flight || matched_se->in_flight;
                entry.IntersectReach(*matched_se);
                entry.acyclic = entry.acyclic && matched_se->acyclic;
                entry.loop_pre_reach &= matched_se->loop_pre_reach;
                entry.loop_post_reach &= matched_se->loop_post_reach;
                entry.wait_seq = std::max(entry.wait_seq, matched_se->wait_seq);
              }
              // If `!matched_se`, freshly allocated inside a <=1-trip loop; untouched on 0-trip path.
            }
          }
          return;
        }
        const uint32_t outer_scope_id = walk_loop_scopes.back();
        const uint32_t this_scope_id = discovered.loop_scope_ids.at(loop.get());
        auto saved = active_by_dir;
        ++loop_depth;
        walk_loop_scopes.push_back(this_scope_id);
        allocate_walk(loop->body_);
        walk_loop_scopes.pop_back();
        --loop_depth;
        merge_loop_exit(saved, outer_scope_id, this_scope_id, positive_trip);
        return;
      }
      if (const auto loop = As<WhileStmt>(stmt)) {
        const uint32_t outer_scope_id = walk_loop_scopes.back();
        const uint32_t this_scope_id = discovered.loop_scope_ids.at(loop.get());
        auto saved = active_by_dir;
        ++loop_depth;
        walk_loop_scopes.push_back(this_scope_id);
        allocate_walk(loop->body_);
        walk_loop_scopes.pop_back();
        --loop_depth;
        merge_loop_exit(saved, outer_scope_id, this_scope_id, /*positive_trip=*/false);
        return;
      }
      if (const auto rit = record_to_ow.find(stmt.get()); rit != record_to_ow.end()) {
        OneWayTransaction& ow = *rit->second;
        ow.record_seq = next_seq++;
        const SyncDirection dir{ow.source, ow.destination};
        auto& active = active_by_dir[dir];
        const bool may_reuse = ow.loop_stack.empty() || ow.needs_return_ack;
        int chosen = -1;
        for (auto& entry : active) {
          if (entry.in_flight || is_branch_blocked(dir, entry.id)) continue;
          if (ow.loop_scope_id != 0 && entry.is_ack) continue;
          if ((entry.reach_from_dst & RelayPipeBit(ow.source)) == 0) continue;
          if (entry.loop_scope_id != ow.loop_scope_id && entry.loop_scope_id != 0) continue;
          const bool acyclic_reuse = (may_reuse || entry.loop_scope_id == 0) && entry.acyclic;
          const bool loop_closed_reuse = ow.loop_scope_id != 0 && entry.loop_scope_id == ow.loop_scope_id &&
                                         (entry.loop_post_reach & ow.loop_pre_reach) != 0 &&
                                         (ow.loop_post_reach & entry.loop_pre_reach) != 0;
          if (!acyclic_reuse && !loop_closed_reuse) continue;
          chosen = entry.id;
          entry.in_flight = true;
          entry.is_ack = false;
          entry.reach_from_dst = 0;
          entry.pipe_reach_seq.fill(0);
          entry.loop_scope_id = ow.loop_scope_id;
          entry.acyclic = may_reuse;
          if (acyclic_reuse) {
            entry.loop_pre_reach = ow.loop_pre_reach;
            entry.loop_post_reach = ow.loop_post_reach;
          } else {
            entry.loop_pre_reach &= ow.loop_pre_reach;
            entry.loop_post_reach &= ow.loop_post_reach;
          }
          break;
        }
        if (chosen < 0) {
          for (int candidate : pool_for(dir)) {
            if (!is_branch_blocked(dir, candidate) &&
                std::none_of(active.begin(), active.end(),
                             [&](const ActiveId& e) { return e.id == candidate; })) {
              chosen = candidate;
              active.push_back({chosen,
                                true,
                                0,
                                may_reuse,
                                ow.loop_pre_reach,
                                ow.loop_post_reach,
                                ow.loop_scope_id,
                                0,
                                {},
                                false});
              if (reservations_[dir].insert(chosen).second) ++roles_count_;
              break;
            }
          }
        }
        CHECK(chosen >= 0)
            << "Closed prefix role reservation unavailable; this is not a fixed-plan capacity lower bound";
        ow.forward_id = chosen;
        return;
      }
      if (const auto wit = wait_to_ow.find(stmt.get()); wit != wait_to_ow.end()) {
        OneWayTransaction& ow = *wit->second;
        const SyncDirection fwd{ow.source, ow.destination};
        const uint64_t seq_now = next_seq++;
        for (auto& entry : active_by_dir[fwd]) {
          if (entry.id == ow.forward_id) {
            entry.in_flight = false;
            entry.wait_seq = seq_now;
          }
        }
        for (auto& [dir, vec] : active_by_dir) {
          for (auto& entry : vec) {
            if (dir == fwd && entry.id == ow.forward_id) {
              entry.AddReach(ow.destination, seq_now);
              if (ow.needs_return_ack) entry.AddReach(ow.source, seq_now);
            } else if (entry.wait_seq <= ow.record_seq && entry.ReachedAtOrBefore(ow.source, ow.record_seq)) {
              entry.AddReach(ow.destination, seq_now);
              if (ow.needs_return_ack) entry.AddReach(ow.source, seq_now);
            }
          }
        }
        if (ow.needs_return_ack) {
          const SyncDirection rev{ow.destination, ow.source};
          auto& rev_active = active_by_dir[rev];
          int ack_chosen = -1;
          for (auto& entry : rev_active) {
            if (!entry.in_flight && entry.acyclic && (ow.loop_scope_id == 0 || entry.is_ack) &&
                (entry.loop_scope_id == ow.loop_scope_id || entry.loop_scope_id == 0) &&
                (entry.reach_from_dst & RelayPipeBit(rev.first)) != 0 && !is_branch_blocked(rev, entry.id)) {
              ack_chosen = entry.id;
              entry.is_ack = true;
              entry.reach_from_dst = 0;
              entry.pipe_reach_seq.fill(0);
              entry.AddReach(rev.second, seq_now);
              entry.loop_scope_id = ow.loop_scope_id;
              entry.acyclic = true;
              entry.loop_pre_reach = 0;
              entry.loop_post_reach = 0;
              entry.wait_seq = seq_now;
              break;
            }
          }
          if (ack_chosen < 0) {
            for (int candidate : pool_for(rev)) {
              if (!is_branch_blocked(rev, candidate) &&
                  std::none_of(rev_active.begin(), rev_active.end(),
                               [&](const ActiveId& e) { return e.id == candidate; })) {
                ack_chosen = candidate;
                ActiveId fresh{ack_chosen, false, 0, true, 0, 0, ow.loop_scope_id, seq_now, {}, true};
                fresh.AddReach(rev.second, seq_now);
                rev_active.push_back(fresh);
                if (reservations_[rev].insert(ack_chosen).second) ++roles_count_;
                break;
              }
            }
          }
          CHECK(ack_chosen >= 0)
              << "Closed prefix role reservation unavailable; this is not a fixed-plan capacity lower bound";
          ow.return_id = ack_chosen;
          // `SetWaitAck` appends `set_flag(rev) ; wait_flag(rev)` at `ow.wait_cut`,
          // ordering `rev.first -> rev.second` after every prior idle completion.
          for (auto& [dir, vec] : active_by_dir) {
            for (auto& entry : vec) {
              if (dir == rev && entry.id == ack_chosen) continue;
              if (!entry.in_flight && (entry.reach_from_dst & RelayPipeBit(rev.first)) != 0) {
                entry.AddReach(rev.second, seq_now);
              }
            }
          }
        }
        return;
      }
    };
    allocate_walk(owner->body_);

    for (const auto& w : discovered.erased_waits) {
      commands_.emplace(w.get(), AssignedCommand{CommandKind::Erase, {}, -1});
    }
    for (const auto& d : discovered.erased_defs) {
      commands_.emplace(d.get(), AssignedCommand{CommandKind::Erase, {}, -1});
    }
    for (const auto& ow : discovered.one_way) {
      const SyncDirection fwd{ow.source, ow.destination};
      if (!ow.needs_return_ack) {
        CHECK(
            commands_.emplace(ow.record_cut.get(), AssignedCommand{CommandKind::SetFlag, fwd, ow.forward_id})
                .second)
            << "One-way prefix protocol reuses an original static command cut";
        CHECK(commands_.emplace(ow.wait_cut.get(), AssignedCommand{CommandKind::WaitFlag, fwd, ow.forward_id})
                  .second)
            << "One-way prefix protocol reuses an original static command cut";
        ++one_way_sites_;
      } else {
        const SyncDirection rev{ow.destination, ow.source};
        CHECK(
            commands_.emplace(ow.record_cut.get(), AssignedCommand{CommandKind::SetFlag, fwd, ow.forward_id})
                .second)
            << "Fallback prefix protocol reuses an original static command cut";
        CHECK(commands_
                  .emplace(ow.wait_cut.get(),
                           AssignedCommand{CommandKind::SetWaitAck, fwd, ow.forward_id, rev, ow.return_id})
                  .second)
            << "Fallback prefix protocol reuses an original static command cut";
        one_way_fallback_sites_ += 2;
      }
    }
    for (const auto& local : discovered.local) {
      const auto* barrier = RequireLocalSyncBarrier(local.pipe, owner->func_type_);
      CHECK(local.pipe != PipeType::S || native.scalar_prefix_serialized)
          << "Scalar current-prefix completion lacks native order evidence";
      if (local.pipe != PipeType::S || !native.scalar_prefix_serialized) {
        RelayMask intervening_pipes = 0;
        if (local.parent_seq) {
          for (size_t i = local.rec_idx + 1; i < local.wait_idx; ++i) {
            const auto pit = native_ops.subtree_payload_pipes.find(local.parent_seq->stmts_[i].get());
            if (pit != native_ops.subtree_payload_pipes.end()) intervening_pipes |= pit->second;
          }
        } else {
          enum class ScanState { Before, Inside, Done };
          ScanState state = ScanState::Before;
          std::function<void(const StmtPtr&)> scan = [&](const StmtPtr& s) {
            if (!s || state == ScanState::Done) return;
            if (s == local.cuts[0]) {
              state = ScanState::Inside;
              return;
            }
            if (s == local.cuts[1]) {
              state = ScanState::Done;
              return;
            }
            if (const auto seq = As<SeqStmts>(s)) {
              for (const auto& child : seq->stmts_) {
                scan(child);
                if (state == ScanState::Done) return;
              }
            } else if (state == ScanState::Inside) {
              const auto pit = native_ops.subtree_payload_pipes.find(s.get());
              if (pit != native_ops.subtree_payload_pipes.end()) intervening_pipes |= pit->second;
            }
          };
          scan(owner->body_);
        }
        CHECK((intervening_pipes & RelayPipeBit(local.pipe)) == 0)
            << "LowerSync same-pipe barrier would strengthen the recorded prefix";
      }
      CHECK(commands_.emplace(local.cuts[0].get(), AssignedCommand{CommandKind::Erase, {}, -1}).second)
          << "Local completion reuses an original static cut";
      CHECK(
          commands_
              .emplace(local.cuts[1].get(),
                       AssignedCommand{
                           barrier ? CommandKind::Barrier : CommandKind::Erase, {local.pipe, local.pipe}, -1})
              .second)
          << "Local completion reuses an original static cut";
      if (barrier) ++barriers_;
    }
  }

  const AssignedCommand* Find(const StmtPtr& statement) const {
    const auto found = commands_.find(statement.get());
    return found == commands_.end() ? nullptr : &found->second;
  }
  const SyncNativeReservations& Reservations() const { return reservations_; }
  const std::vector<std::tuple<PipeType, PipeType, int>>& NativeClaimed() const { return native_claimed_; }
  size_t Roles() const { return roles_count_; }
  size_t Barriers() const { return barriers_; }
  size_t OneWaySites() const { return one_way_sites_; }
  size_t OneWayFallbackSites() const { return one_way_fallback_sites_; }
  size_t SuffixVisits() const { return suffix_visits_; }

 private:
  std::map<const Stmt*, AssignedCommand> commands_;
  SyncNativeReservations reservations_;
  std::vector<std::tuple<PipeType, PipeType, int>> native_claimed_;
  size_t roles_count_ = 0;
  size_t barriers_ = 0;
  size_t one_way_sites_ = 0;
  size_t one_way_fallback_sites_ = 0;
  size_t suffix_visits_ = 0;
};

class Emit : public SyncStatementMutator {
 public:
  Emit(const Assignment& assignment, FunctionType core, const std::map<const ForStmt*, StmtPtr>& children)
      : assignment_(assignment), core_(core == FunctionType::AIV ? "aiv" : "aic"), children_(children) {}

  FunctionPtr Rewrite(const FunctionPtr& function) {
    const auto body = VisitStmt(function->body_);
    return std::make_shared<Function>(function->name_, function->params_, function->param_directions_,
                                      function->return_types_, body, function->span_, function->func_type_,
                                      function->level_, function->role_, function->attrs_,
                                      function->requires_runtime_binding_, function->ir_stage_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    std::vector<StmtPtr> rewritten;
    rewritten.reserve(sequence->stmts_.size());
    for (const auto& statement : sequence->stmts_) {
      auto next = VisitStmt(statement);
      if (const auto nested = As<SeqStmts>(next)) {
        for (const auto& item : nested->stmts_) rewritten.push_back(item);
      } else {
        rewritten.push_back(std::move(next));
      }
    }
    return std::make_shared<SeqStmts>(std::move(rewritten), sequence->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    const auto found = children_.find(loop.get());
    return found == children_.end() ? SyncStatementMutator::VisitStmt_(loop) : found->second;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& statement) override {
    if (const auto cmd = assignment_.Find(statement)) {
      if (cmd->kind == CommandKind::Erase) return Empty(statement->span_);
      return std::make_shared<EvalStmt>(
          Command(cmd->kind == CommandKind::SetFlag ? "system.set_flag" : "system.wait_flag", cmd->direction,
                  cmd->id, statement->span_),
          statement->span_);
    }
    return SyncStatementMutator::VisitStmt_(statement);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    if (const auto cmd = assignment_.Find(statement)) {
      if (cmd->kind == CommandKind::Erase) return Empty(statement->span_);
      if (cmd->kind == CommandKind::Barrier) {
        return std::make_shared<EvalStmt>(
            OpRegistry::GetInstance().Create(
                RequireLocalSyncBarrier(cmd->direction.first,
                                        core_ == "aiv" ? FunctionType::AIV : FunctionType::AIC),
                {}, {{"core_type", core_}}, statement->span_),
            statement->span_);
      }
      if (cmd->kind == CommandKind::SetWaitAck) {
        std::vector<StmtPtr> seq;
        seq.push_back(std::make_shared<EvalStmt>(
            Command("system.wait_flag", cmd->direction, cmd->id, statement->span_), statement->span_));
        seq.push_back(std::make_shared<EvalStmt>(
            Command("system.set_flag", cmd->ack_direction, cmd->ack_id, statement->span_), statement->span_));
        seq.push_back(std::make_shared<EvalStmt>(
            Command("system.wait_flag", cmd->ack_direction, cmd->ack_id, statement->span_),
            statement->span_));
        return std::make_shared<SeqStmts>(std::move(seq), statement->span_);
      }
      return std::make_shared<EvalStmt>(
          Command(cmd->kind == CommandKind::SetFlag ? "system.set_flag" : "system.wait_flag", cmd->direction,
                  cmd->id, statement->span_),
          statement->span_);
    }
    return SyncStatementMutator::VisitStmt_(statement);
  }

 private:
  StmtPtr Empty(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }
  CallPtr Command(const char* name, SyncDirection direction, int id, const Span& span) const {
    return OpRegistry::GetInstance().Create(name, {},
                                            {{"src_pipe", static_cast<int>(direction.first)},
                                             {"dst_pipe", static_cast<int>(direction.second)},
                                             {"event_id", id},
                                             {"core_type", core_}},
                                            span);
  }
  const Assignment& assignment_;
  std::string core_;
  const std::map<const ForStmt*, StmtPtr>& children_;
};

}  // namespace

bool QualifyLocalSyncInvocation(const FunctionPtr& function, const std::string& target) {
  if (!function || function->ir_stage_ != FunctionIRStage::Buffer ||
      (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV))
    return false;
  if (function->requires_runtime_binding_ || function->HasAttr(kAttrExternalSource) ||
      function->HasAttr(kAttrBuiltinTemplateDir))
    return false;
  backend::RequireLocalSyncExecutionContract(target);
  NativeOperations native(function->func_type_, target);
  native.VisitFunction(function);
  const BufferRuntimeValidAnalysis metadata(function);
  if (const auto& issue = metadata.issue()) CHECK_SPAN(false, issue->call->span_) << issue->message;
  return !native.payloads.empty();
}

namespace {
bool HasMatchingClosedPrefixUses(const FunctionPtr& function, const Discover& discovered) {
  if (discovered.unsupported_syntax) return false;
  if (discovered.local.empty() && discovered.one_way.empty() && discovered.children.empty()) return false;

  // Reject token fanout, carried iter_args, or non-trivial token containers (e.g. TupleType([SyncTokenType]))
  // so both HasOnlyClosedPrefixSyntax and TryLowerClosedPrefixSync agree on finite/exact shapes.
  class CombinedTokenCheck : public IRVisitor {
   public:
    size_t uses = 0;
    bool supported = true;
    explicit CombinedTokenCheck(const std::vector<ForStmtPtr>& children) {
      for (const auto& child : children) child_scopes_.insert(child.get());
    }
    void VisitStmt_(const ForStmtPtr& loop) override {
      if (child_scopes_.count(loop.get())) return;
      IRVisitor::VisitStmt_(loop);
    }
    void VisitStmt_(const AssignStmtPtr& statement) override {
      if (As<SyncTokenType>(statement->var_->GetType())) return;
      IRVisitor::VisitStmt_(statement);
    }
    void VisitStmt_(const EvalStmtPtr& statement) override {
      if (const auto call = As<Call>(statement->expr_); call && IsOp(call, "sync.wait")) {
        if (!call->args_.empty()) ++uses;
        return;
      }
      IRVisitor::VisitStmt_(statement);
    }
    void VisitExpr(const ExprPtr& expression) override {
      if (!expression || !visited_.insert(expression.get()).second) return;
      if (ContainsSyncTokenType(expression->GetType())) supported = false;
      IRVisitor::VisitExpr(expression);
    }
    void VisitExpr_(const CallPtr& call) override {
      if (IsOp(call, "sync.wait") && !call->args_.empty()) ++uses;
      IRVisitor::VisitExpr_(call);
    }

   private:
    std::set<const ForStmt*> child_scopes_;
    std::unordered_set<const Expr*> visited_;
  } check(discovered.children);
  check.VisitFunction(function);
  if (!check.supported) return false;
  const size_t expected_uses =
      discovered.local.size() + discovered.one_way.size() + discovered.erased_waits.size();
  return check.uses == expected_uses;
}

using ChildResourceDirections = std::map<const ForStmt*, std::set<SyncDirection>>;

// Read claims from the independently lowered child to cover both closed-prefix
// ACKs and structured event commands. Parent-reserved lanes are exclusions from
// the child's palette, so derive native claims solely from the child's payloads.
std::set<SyncDirection> CollectChildResourceDirections(const FunctionPtr& logical, const StmtPtr& physical,
                                                       const std::string& target) {
  class Publications : public IRVisitor {
   public:
    std::set<SyncDirection> directions;

   protected:
    void VisitStmt_(const EvalStmtPtr& statement) override {
      if (const auto call = As<Call>(statement->expr_); call && IsOp(call, "system.set_flag")) {
        directions.emplace(Pipe(call, "src_pipe"), Pipe(call, "dst_pipe"));
      }
    }
    void VisitExpr(const ExprPtr&) override {}
  } publications;
  publications.VisitStmt(physical);
  NativeOperations native(logical->func_type_, target);
  native.VisitFunction(logical);
  for (const auto& [dir, ids] : SyncNativeClaims(logical, native.all_native_payloads, target)) {
    (void)ids;
    if (dir.first != dir.second) {
      publications.directions.insert(dir);
    }
  }
  return publications.directions;
}

// Walk original control cuts with actual child directions. Both arms start
// with the entering claims; only a following cut sees the union of the arms.
class ChildEntryVerifier : public IRVisitor {
 public:
  ChildEntryVerifier(const Discover& discovered, const ChildResourceDirections& directions)
      : discovered_(discovered), directions_(directions) {}

 protected:
  void VisitExpr(const ExprPtr&) override {}
  void VisitStmt_(const IfStmtPtr& branch) override {
    const uint64_t before = prior_directions_;
    VisitStmt(branch->then_body_);
    const uint64_t after_then = prior_directions_;
    prior_directions_ = before;
    if (branch->else_body_) {
      VisitStmt(*branch->else_body_);
    }
    prior_directions_ |= after_then;
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    const auto child = directions_.find(loop.get());
    if (child == directions_.end()) {
      IRVisitor::VisitStmt_(loop);
      return;
    }
    const auto entry = discovered_.entries.find(loop.get());
    CHECK(entry != discovered_.entries.end()) << "Closed child lacks an original entry protocol cut";
    const bool loop_nested = discovered_.loop_nested_children.count(loop.get()) != 0;
    const bool first_visit = loop->HasAttr(kClosedSyncChildFirstVisitAttr);
    CHECK(!first_visit || !loop_nested)
        << "First-visit closed child cannot appear inside a multi-trip loop or after an earlier child";
    uint64_t mask = 0;
    for (const auto& [producer, consumer] : child->second) {
      const auto bit = SyncDirectionBit(producer, consumer);
      const bool prior_claim = (prior_directions_ & bit) != 0;
      CHECK(!first_visit || !prior_claim)
          << "First-visit closed child cannot appear inside a multi-trip loop or after an earlier child";
      if (!first_visit && (loop_nested || prior_directions_ == 0 || prior_claim)) {
        CHECK((entry->second[static_cast<size_t>(consumer)] & RelayPipeBit(producer)) != 0)
            << "Closed child entry lacks a participating-pipe acknowledgment; cross-visit rearm unproved";
      }
      mask |= bit;
    }
    prior_directions_ |= mask;
  }

 private:
  const Discover& discovered_;
  const ChildResourceDirections& directions_;
  uint64_t prior_directions_ = 0;
};
}  // namespace

bool HasOnlyClosedPrefixSyntax(const FunctionPtr& function) {
  Discover discovered;
  discovered.VisitFunction(function);
  if (!HasMatchingClosedPrefixUses(function, discovered)) return false;
  if (function->HasAttr(kAttrAutosyncAnalysisPolicy) &&
      function->GetAttr<std::string>(kAttrAutosyncAnalysisPolicy) == "conservative_prefix")
    return true;
  if (discovered.has_closed_pair || !discovered.children.empty()) return true;
  return discovered.has_while || !discovered.has_payload || (discovered.has_if && !discovered.has_for);
}

std::optional<std::set<SyncDirection>> PlanClosedPrefixEventDirections(
    const FunctionPtr& function, const SyncParameterBounds& parameters,
    const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  const SyncControlRangeProver prover(function, parameters, blocks);
  Discover discovered(&prover);
  discovered.VisitFunction(function);
  if (!HasMatchingClosedPrefixUses(function, discovered) || !discovered.children.empty()) {
    return std::nullopt;
  }
  Assignment::PlanOneWayClosure(function, discovered, prover);
  std::set<SyncDirection> directions;
  for (const auto& transaction : discovered.one_way) {
    directions.emplace(transaction.source, transaction.destination);
    if (transaction.needs_return_ack) {
      directions.emplace(transaction.destination, transaction.source);
    }
  }
  return directions;
}

bool HasClosedSyncChild(const FunctionPtr& function) {
  if (!function || !function->body_) return false;
  class Finder : public IRVisitor {
   public:
    bool found = false;
    void VisitExpr(const ExprPtr&) override {}
    void VisitStmt(const StmtPtr& s) override {
      if (!found && s) IRVisitor::VisitStmt(s);
    }
    void VisitStmt_(const ForStmtPtr& loop) override {
      if (loop->HasAttr(kClosedSyncChildAttr)) {
        found = true;
        return;
      }
      IRVisitor::VisitStmt_(loop);
    }
  } finder;
  finder.VisitFunction(function);
  return finder.found;
}

std::optional<PhysicalSyncRealization> TryLowerClosedPrefixSync(
    const FunctionPtr& function, const std::string& target,
    const std::optional<std::vector<int>>& eligible_ids, const SyncNativeReservations& extra_reservations,
    const SyncParameterBounds& parameters, const std::optional<std::pair<int64_t, int64_t>>& blocks) {
  const SyncControlRangeProver prover(function, parameters, blocks);
  Discover discovered(&prover);
  discovered.VisitFunction(function);
  if (!HasMatchingClosedPrefixUses(function, discovered)) return std::nullopt;
  CHECK(!function->requires_runtime_binding_ && !function->HasAttr(kAttrExternalSource) &&
        !function->HasAttr(kAttrBuiltinTemplateDir))
      << "Closed prefix protocols cannot replace an external invocation owner";

  NativeOperations native(function->func_type_, target, /*track_payload_pipes=*/true);
  native.VisitFunction(function);
  const Assignment assignment(function, discovered, native, target, eligible_ids, extra_reservations, prover);

  std::map<const ForStmt*, StmtPtr> children;
  ChildResourceDirections child_directions;
  size_t child_sets = 0, child_waits = 0, child_barriers = 0, child_lanes = 0, child_suffix_visits = 0;
  const auto bindings =
      discovered.children.empty()
          ? std::optional<SyncRegionBindings>{}
          : std::optional<SyncRegionBindings>{std::in_place, function, target, parameters, blocks};
  const auto* ctx = PassContext::Current();
  const bool disjoint_gm = ctx && ctx->GetAutoSyncDisjointGM();
  for (const auto& child : discovered.children) {
    SyncExternalPolicy external_policy = SyncExternalPolicy::Geometry;
    if (child->HasAttr(kClosedSyncChildExternalPolicyAttr)) {
      const auto policy_attr = child->GetAttr<std::string>(kClosedSyncChildExternalPolicyAttr);
      if (policy_attr == "readonly_projection") {
        CHECK(HasReadonlyExternalSyncEffects(child, function->func_type_, target, nullptr, false))
            << "Closed child readonly projection policy is no longer valid";
        external_policy = SyncExternalPolicy::ReadOnlyProjection;
      } else if (policy_attr == "geometry") {
        (void)HasReadonlyExternalSyncEffects(child, function->func_type_, target, nullptr, true);
        external_policy = SyncExternalPolicy::Geometry;
      } else {
        CHECK(false) << "Unsupported closed child external policy: " << policy_attr;
      }
    } else {
      external_policy =
          HasReadonlyExternalSyncEffects(child, function->func_type_, target, nullptr, disjoint_gm)
              ? SyncExternalPolicy::ReadOnlyProjection
              : SyncExternalPolicy::Geometry;
    }
    auto context = bindings->Bind(child, external_policy);
    CheckLogicalSync(context.binding_owner);
    const bool is_subblock = child->HasAttr(kClosedSyncChildSubblockAttr);
    auto unmarked = std::make_shared<ForStmt>(*child);
    unmarked->attrs_.erase(std::remove_if(unmarked->attrs_.begin(), unmarked->attrs_.end(),
                                          [](const auto& attribute) {
                                            return attribute.first == kClosedSyncChildAttr ||
                                                   attribute.first == kClosedSyncChildExternalPolicyAttr ||
                                                   attribute.first == kClosedSyncChildFirstVisitAttr ||
                                                   attribute.first == kClosedSyncChildSubblockAttr;
                                          }),
                           unmarked->attrs_.end());
    class Replace : public SyncStatementMutator {
     public:
      Replace(ForStmtPtr before, ForStmtPtr after) : before_(std::move(before)), after_(std::move(after)) {}

     protected:
      StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
        return loop == before_ ? after_ : SyncStatementMutator::VisitStmt_(loop);
      }

     private:
      ForStmtPtr before_, after_;
    } replace(child, unmarked);
    auto actual = MutableCopy(context.binding_owner);
    actual->body_ = replace.VisitStmt(context.binding_owner->body_);
    // Unified per-direction resource management: forward `eligible_ids` along
    // with per-direction parent reservations instead of a flat global subtraction.
    auto child_bounds = ProjectSyncParameterBounds(function, actual, parameters);
    child_bounds = ProjectEnclosingLoopBounds(context, bindings->Prover(), child, child_bounds);
    auto lowered = LowerLogicalSync(actual, target, eligible_ids, child_bounds, blocks, {}, external_policy,
                                    std::nullopt, assignment.Reservations());
    const auto original_body = As<SeqStmts>(actual->body_);
    const auto lowered_body = As<SeqStmts>(lowered.function->body_);
    INTERNAL_CHECK(original_body && lowered_body) << "Internal error: child binding body is not a sequence";
    child_directions.emplace(child.get(),
                             CollectChildResourceDirections(context.binding_owner, lowered_body, target));
    std::set<const Var*> descriptor_variables;
    for (const auto& statement : original_body->stmts_)
      if (const auto assignment_stmt = As<AssignStmt>(statement))
        descriptor_variables.insert(assignment_stmt->var_.get());
    std::vector<StmtPtr> recipe;
    for (const auto& statement : lowered_body->stmts_) {
      if (As<ReturnStmt>(statement)) continue;
      const auto assignment_stmt = As<AssignStmt>(statement);
      if (assignment_stmt && descriptor_variables.count(assignment_stmt->var_.get())) continue;
      if (is_subblock) {
        if (const auto lowered_loop = As<ForStmt>(statement);
            lowered_loop && lowered_loop->loop_var_ == child->loop_var_) {
          var_collectors::VarDefUseCollector uses;
          uses.VisitStmt(lowered_loop->body_);
          INTERNAL_CHECK(!uses.var_uses.count(child->loop_var_.get()))
              << "Internal error: lowered sub-block window retains synthetic loop coordinate uses";
          if (const auto inner_seq = As<SeqStmts>(lowered_loop->body_)) {
            for (size_t i = 0; i < inner_seq->stmts_.size(); ++i) {
              const auto& inner_stmt = inner_seq->stmts_[i];
              if (const auto yield = As<YieldStmt>(inner_stmt); yield && i + 1 == inner_seq->stmts_.size()) {
                INTERNAL_CHECK(yield->value_.empty())
                    << "Internal error: sub-block unwrap encountered non-empty loop yield";
                continue;
              }
              recipe.push_back(inner_stmt);
            }
          } else if (const auto yield = As<YieldStmt>(lowered_loop->body_)) {
            INTERNAL_CHECK(yield->value_.empty())
                << "Internal error: sub-block unwrap encountered non-empty loop yield";
          } else {
            recipe.push_back(lowered_loop->body_);
          }
          continue;
        }
      }
      recipe.push_back(statement);
    }
    children.emplace(child.get(), std::make_shared<SeqStmts>(recipe, child->span_));
    child_sets += lowered.set_sites;
    child_waits += lowered.wait_sites;
    for (const auto& plan : lowered.physical_cases) {
      child_barriers += plan.barrier_sites;
      child_lanes += plan.reserved_lanes;
      child_suffix_visits += plan.closed_prefix_suffix_visits;
    }
  }

  if (!child_directions.empty()) {
    ChildEntryVerifier(discovered, child_directions).VisitFunction(function);
  }
  Emit emit(assignment, function->func_type_, children);
  PhysicalSyncRealization result;
  result.function = emit.Rewrite(function);
  result.set_sites = assignment.OneWaySites() + assignment.OneWayFallbackSites() + child_sets;
  result.wait_sites = assignment.OneWaySites() + assignment.OneWayFallbackSites() + child_waits;
  result.barrier_sites = assignment.Barriers() + child_barriers;
  result.reserved_lanes = assignment.Roles() + child_lanes;
  result.closed_prefix_suffix_visits = assignment.SuffixVisits() + child_suffix_visits;
  result.native_reserved_ids = assignment.NativeClaimed();
  return result;
}
}  // namespace pypto::ir::sync
