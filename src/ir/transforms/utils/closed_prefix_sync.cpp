/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/retained_sync_child.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
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

RelayMatrix ComposeRelay(const RelayMatrix& first, const RelayMatrix& second) {
  RelayMatrix out{};
  for (size_t i = 0; i < kRelayPipeCount; ++i) {
    RelayMask r = first[i];
    for (size_t mid = 0; mid < kRelayPipeCount; ++mid) {
      if ((first[i] & (RelayMask{1} << mid)) != 0) r |= second[mid];
    }
    out[i] = r;
  }
  return out;
}

void ApplyRelayEdge(RelayMatrix& matrix, PipeType src, PipeType dst) {
  const RelayMask sm = RelayPipeBit(src);
  const RelayMask dm = RelayPipeBit(dst);
  if (sm == 0 || dm == 0) return;
  for (size_t i = 0; i < kRelayPipeCount; ++i) {
    if ((matrix[i] & sm) != 0) matrix[i] |= dm;
  }
}

PipeType Pipe(const CallPtr& operation, const char* key) {
  for (const auto& [name, value] : operation->kwargs_) {
    if (name == key) return static_cast<PipeType>(pypto::AnyCast<int>(value, name));
  }
  INTERNAL_CHECK(false) << "Internal error: missing pipeline attribute";
  return PipeType::S;
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

// Explicit four-command closed prefix transaction.
struct Transaction {
  PipeType source;
  PipeType destination;
  std::array<StmtPtr, 4> cuts;
};

// Local same-pipe hardware barrier (`record(p); wait(p)`).
struct LocalCompletion {
  PipeType pipe;
  std::array<StmtPtr, 2> cuts;
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
  int forward_id = -1;
  int return_id = -1;
  uint64_t record_seq = 0;
};

bool ContainsSyncTokenType(const TypePtr& type) {
  if (!type) return false;
  if (As<SyncTokenType>(type)) return true;
  if (const auto tup = As<TupleType>(type)) {
    for (const auto& elem : tup->types_) {
      if (ContainsSyncTokenType(elem)) return true;
    }
  }
  return false;
}

class Discover : public IRVisitor {
 public:
  std::vector<Transaction> transactions;
  std::vector<LocalCompletion> local;
  std::vector<OneWayTransaction> one_way;
  std::vector<StmtPtr> erased_waits;
  std::vector<StmtPtr> erased_defs;
  std::vector<ForStmtPtr> children;
  std::map<const ForStmt*, std::set<SyncDirection>> entries;
  bool unsupported_syntax = false;
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
      } else if (!IsNonPayloadCall(call)) {
        has_payload = true;
      }
    }
    IRVisitor::VisitStmt_(eval);
  }
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (As<SyncTokenType>(assign->var_->GetType())) {
      unsupported_syntax = true;
    }
    if (const auto call = As<Call>(assign->value_)) {
      if (GetLogicalSyncKind(call->op_) != LogicalSyncKind::None) {
        unsupported_syntax = true;
      } else if (!IsNonPayloadCall(call)) {
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
    IRVisitor::VisitStmt_(branch);
    control_stack_.pop_back();
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loop->HasAttr(kClosedSyncChildAttr)) {
      CHECK(loop->GetAttr<int>(kClosedSyncChildAttr) == 1) << "Unsupported closed-child routing version";
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
    loop_stack_.push_back(loop.get());
    control_stack_.push_back(loop.get());
    loop_scope_stack_.push_back(++next_loop_scope_id_);
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
    loop_scope_stack_.push_back(++next_loop_scope_id_);
    IRVisitor::VisitStmt_(loop);
    loop_scope_stack_.pop_back();
    control_stack_.pop_back();
    loop_stack_.pop_back();
  }

  void VisitStmt_(const SeqStmtsPtr& sequence) override {
    const auto& statements = sequence->stmts_;
    std::set<SyncDirection> preceding;
    for (size_t index = 0; index < statements.size(); ++index) {
      if (const auto child = As<ForStmt>(statements[index]); child && child->HasAttr(kClosedSyncChildAttr)) {
        entries.emplace(child.get(), preceding);
      }
      if (const auto assign = As<AssignStmt>(statements[index]);
          assign && As<SyncTokenType>(assign->var_->GetType())) {
        if (const auto call = As<Call>(assign->value_)) {
          if (IsOp(call, "sync.completed")) {
            completed_tokens_.insert(assign->var_.get());
            erased_defs.push_back(statements[index]);
            preceding.clear();
            continue;
          }
        } else if (const auto alias = As<Var>(assign->value_); alias && completed_tokens_.count(alias.get())) {
          completed_tokens_.insert(assign->var_.get());
          erased_defs.push_back(statements[index]);
          preceding.clear();
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

      // Recognize adjacent closed four-command or local two-command patterns first.
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
          if (src != dst && Pipe(r1, "pipe") == dst && Pipe(w1, "dst_pipe") == src &&
              w0->args_.size() == 1 && w0->args_[0] == t0 && w1->args_.size() == 1 &&
              w1->args_[0] == t1) {
            transactions.push_back({src, dst,
                                    {statements[index], statements[index + 1], statements[index + 2],
                                     statements[index + 3]}});
            preceding.insert({src, dst});
            preceding.insert({dst, src});
            index += 3;
            continue;
          }
        }
      }
      if (index + 1 < statements.size()) {
        const auto r0 = SyncRecord(statements[index]);
        const auto w0 = SyncWait(statements[index + 1]);
        if (r0 && w0) {
          const auto src = Pipe(r0, "pipe");
          const auto dst = Pipe(w0, "dst_pipe");
          const auto t0 = As<AssignStmt>(statements[index])->var_;
          if (src == dst && w0->args_.size() == 1 && w0->args_[0] == t0) {
            local.push_back({src, {statements[index], statements[index + 1]}});
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
        pending_records_[var.get()] = {Pipe(rec, "pipe"), statements[index], control_stack_, false};
        continue;
      }
      if (const auto w = SyncWait(statements[index])) {
        const auto token = As<Var>(w->args_.empty() ? ExprPtr{} : w->args_[0]);
        if (!token) {
          unsupported_syntax = true;
          preceding.clear();
          continue;
        }
        const auto rec_it = pending_records_.find(token.get());
        if (rec_it == pending_records_.end() || rec_it->second.used ||
            rec_it->second.control_stack != control_stack_) {
          unsupported_syntax = true;
          preceding.clear();
          continue;
        }
        rec_it->second.used = true;
        const auto src = rec_it->second.pipe;
        const auto dst = Pipe(w, "dst_pipe");
        if (src == dst) {
          local.push_back({src, {rec_it->second.stmt, statements[index]}});
        } else {
          one_way.push_back({src, dst, rec_it->second.stmt, statements[index], loop_stack_,
                             loop_scope_stack_.back()});
          preceding.insert({src, dst});
        }
        continue;
      }

      preceding.clear();
      VisitStmt(statements[index]);
    }
  }

 private:
  static bool IsNonPayloadCall(const CallPtr& call) {
    return IsOp(call, "buffer.get_block_idx") || IsOp(call, "buffer.get_block_num") ||
           IsOp(call, "buffer.get_subblock_idx") || IsOp(call, "buffer.alloc") ||
           IsOp(call, "buffer.alloc_multi") || IsOp(call, "buffer.subview") ||
           IsOp(call, "buffer.reshape") || IsOp(call, "buffer.slot") || IsOp(call, "buffer.set_validshape");
  }
  struct PendingRecord {
    PipeType pipe;
    StmtPtr stmt;
    std::vector<const Stmt*> control_stack;
    bool used = false;
  };
  std::set<const Var*> completed_tokens_;
  std::unordered_map<const Var*, PendingRecord> pending_records_;
  std::vector<const Stmt*> loop_stack_;
  std::vector<const Stmt*> control_stack_;
  std::vector<uint32_t> loop_scope_stack_{0};
  uint32_t next_loop_scope_id_ = 0;
};

class NativeOperations : public IRVisitor {
 public:
  NativeOperations(FunctionType core, std::string target) : core_(core), target_(std::move(target)) {
    CHECK(core == FunctionType::AIC || core == FunctionType::AIV)
        << "Closed prefix protocols require one AIC/AIV invocation owner";
  }
  std::vector<CallPtr> payloads;

 protected:
  void VisitExpr(const ExprPtr& expression) override {
    if (visited_.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
  }
  void VisitStmt(const StmtPtr& statement) override {
    CHECK(As<SeqStmts>(statement) || As<AssignStmt>(statement) || As<EvalStmt>(statement) ||
          As<IfStmt>(statement) || As<ForStmt>(statement) || As<WhileStmt>(statement) ||
          As<YieldStmt>(statement) || As<ReturnStmt>(statement))
        << "Closed prefix protocols require local structured sequential control";
    IRVisitor::VisitStmt(statement);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loop->HasAttr(kClosedSyncChildAttr)) return;
    CHECK(loop->kind_ == ForKind::Sequential)
        << "Closed prefix reuse requires a sequential loop after scheduling";
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    VisitStmt(loop->body_);
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
        payloads.push_back(call);
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  FunctionType core_;
  std::string target_;
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

const char* LocalBarrier(PipeType pipe, FunctionType core) {
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

class Assignment {
 public:
  Assignment(const FunctionPtr& owner, Discover& discovered, const std::vector<CallPtr>& payloads,
             const std::string& target, const std::optional<std::vector<int>>& eligible,
             const SyncNativeReservations& extra_reservations) {
    const auto& native = backend::RequireLocalSyncExecutionContract(target);
    CHECK(native.set_completes_source_prefix && native.wait_consumes_before_advance)
        << "Closed prefix protocols require ordered consuming event commands";
    reservations_ = SyncNativeClaims(owner, payloads, target);
    for (const auto& [direction, ids] : reservations_)
      for (int id : ids) native_claimed_.emplace_back(direction.first, direction.second, id);
    for (const auto& [direction, ids] : extra_reservations)
      reservations_[direction].insert(ids.begin(), ids.end());

    // Index statement ordering/edges inside loops so we only fall back to the
    // 4-command protocol when a 1-way loop edge lacks an unconditional reverse cycle.
    std::map<const Stmt*, std::vector<std::pair<PipeType, PipeType>>> cut_edges;
    for (const auto& t : discovered.transactions) {
      cut_edges[t.cuts[1].get()].emplace_back(t.source, t.destination);
      cut_edges[t.cuts[3].get()].emplace_back(t.destination, t.source);
    }
    for (const auto& ow : discovered.one_way) {
      cut_edges[ow.wait_cut.get()].emplace_back(ow.source, ow.destination);
    }

    // Precompute O(N) bottom-up relay summaries + prefix/suffix tables for SeqStmts so each
    // 1-way transaction checks loop backedge closure in O(nesting_depth).
    struct SeqRelays {
      std::vector<RelayMatrix> prefix;
      std::vector<RelayMatrix> suffix;
    };
    struct IfRelays {
      RelayMatrix then_sum;
      RelayMatrix else_sum;
    };
    struct AncestorLink {
      const Stmt* parent = nullptr;
      size_t index_in_seq = 0;
      bool is_if = false;
      bool in_then = false;
    };
    std::unordered_map<const Stmt*, SeqRelays> seq_relays;
    std::unordered_map<const Stmt*, IfRelays> if_relays;
    std::unordered_map<const Stmt*, AncestorLink> ancestors;

    std::function<RelayMatrix(const StmtPtr&)> precompute = [&](const StmtPtr& s) -> RelayMatrix {
      if (!s) return IdentityRelay();
      if (const auto seq = As<SeqStmts>(s)) {
        const size_t n = seq->stmts_.size();
        std::vector<RelayMatrix> child_sums(n, IdentityRelay());
        for (size_t i = 0; i < n; ++i) {
          const auto& c = seq->stmts_[i];
          if (c) ancestors[c.get()] = {seq.get(), i, false, false};
          child_sums[i] = precompute(c);
        }
        SeqRelays sr;
        sr.prefix.assign(n + 1, IdentityRelay());
        sr.suffix.assign(n, IdentityRelay());
        for (size_t i = 0; i < n; ++i) {
          sr.prefix[i + 1] = ComposeRelay(sr.prefix[i], child_sums[i]);
        }
        if (n > 0) {
          for (size_t i = n - 1; i-- > 0;) {
            sr.suffix[i] = ComposeRelay(child_sums[i + 1], sr.suffix[i + 1]);
          }
        }
        const RelayMatrix total = sr.prefix[n];
        seq_relays.emplace(seq.get(), std::move(sr));
        return total;
      }
      if (const auto branch = As<IfStmt>(s)) {
        if (branch->then_body_) ancestors[branch->then_body_.get()] = {branch.get(), 0, true, true};
        if (branch->else_body_) ancestors[(*branch->else_body_).get()] = {branch.get(), 0, true, false};
        const RelayMatrix then_sum = precompute(branch->then_body_);
        const RelayMatrix else_sum = branch->else_body_ ? precompute(*branch->else_body_) : IdentityRelay();
        if_relays.emplace(branch.get(), IfRelays{then_sum, else_sum});
        return IntersectRelay(then_sum, else_sum);
      }
      if (const auto loop = As<ForStmt>(s)) {
        if (loop->HasAttr(kClosedSyncChildAttr)) return IdentityRelay();
        if (loop->body_) ancestors[loop->body_.get()] = {loop.get(), 0, false, false};
        precompute(loop->body_);
        return IdentityRelay();
      }
      if (const auto loop = As<WhileStmt>(s)) {
        if (loop->body_) ancestors[loop->body_.get()] = {loop.get(), 0, false, false};
        precompute(loop->body_);
        return IdentityRelay();
      }
      RelayMatrix m = IdentityRelay();
      if (const auto found = cut_edges.find(s.get()); found != cut_edges.end()) {
        for (const auto& [a, b] : found->second) ApplyRelayEdge(m, a, b);
      }
      return m;
    };
    const bool has_loop_one_way = std::any_of(discovered.one_way.begin(), discovered.one_way.end(),
                                              [](const OneWayTransaction& ow) { return !ow.loop_stack.empty(); });
    if (has_loop_one_way) precompute(owner->body_);

    for (auto& ow : discovered.one_way) {
      if (ow.loop_stack.empty()) {
        ow.needs_return_ack = false;
        continue;
      }
      bool closed_all = true;
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
                              : ComposeRelay(mat, sit->second.suffix[link.index_in_seq]);
            } else if (link.is_if) {
              if (is_prefix) {
                const auto& arms = if_relays.at(link.parent);
                mat = IntersectRelay(mat, link.in_then ? arms.else_sum : arms.then_sum);
              } else {
                mat = IntersectRelay(mat, IdentityRelay());
              }
            }
            cur = link.parent;
          }
          return true;
        };
        if (!advance(cur_rec, rec_prefix, true) || !advance(cur_wt, wt_suffix, false)) {
          closed_all = false;
          break;
        }
        const RelayMatrix back = ComposeRelay(wt_suffix, rec_prefix);
        if ((back[static_cast<size_t>(ow.destination)] & RelayPipeBit(ow.source)) == 0) {
          closed_all = false;
          break;
        }
      }
      ow.needs_return_ack = !closed_all;
    }

    // Reserve constant-role pairs for explicit legacy ClosedTransactions.
    std::map<SyncDirection, std::set<bool>> roles;
    for (const auto& transaction : discovered.transactions) {
      roles[{transaction.source, transaction.destination}].insert(false);
      roles[{transaction.destination, transaction.source}].insert(true);
    }
    std::map<std::pair<SyncDirection, bool>, int> assignment;
    for (const auto& [direction, uses] : roles) {
      const auto ids = SyncEligibleIds(owner, target, direction, eligible, reservations_);
      CHECK(ids.size() >= uses.size())
          << "Closed prefix role reservation unavailable; this is not a fixed-plan capacity lower bound";
      size_t next = 0;
      for (const bool ack : uses) {
        const int chosen = ids[next++];
        assignment.emplace(std::make_pair(direction, ack), chosen);
        reservations_[direction].insert(chosen);
      }
      roles_count_ += uses.size();
    }

    // Allocate directional IDs for all 1-way transactions (both 2-cmd and 4-cmd fallback),
    // tracking in-flight lifetimes so concurrent operations never share a flag ID and
    // reusing non-overlapping idle IDs when causally safe.
    struct ActiveId {
      int id;
      bool in_flight;
      RelayMask reach_from_dst;
      bool acyclic;
      uint32_t loop_scope_id = 0;
      uint64_t wait_seq = 0;
    };
    uint64_t next_seq = 1;
    std::map<SyncDirection, std::vector<ActiveId>> active_by_dir;
    std::map<SyncDirection, std::vector<int>> one_way_pools;
    auto pool_for = [&](SyncDirection dir) -> const std::vector<int>& {
      auto it = one_way_pools.find(dir);
      if (it != one_way_pools.end()) return it->second;
      return one_way_pools.emplace(dir, SyncEligibleIds(owner, target, dir, eligible, reservations_)).first->second;
    };
    std::map<const Stmt*, OneWayTransaction*> wait_to_ow;
    std::map<const Stmt*, OneWayTransaction*> record_to_ow;
    for (auto& ow : discovered.one_way) {
      record_to_ow[ow.record_cut.get()] = &ow;
      wait_to_ow[ow.wait_cut.get()] = &ow;
    }

    auto merge_zero_trip_loop = [&](const std::map<SyncDirection, std::vector<ActiveId>>& saved) {
      for (auto& [dir, vec] : active_by_dir) {
        const auto sit = saved.find(dir);
        for (auto& entry : vec) {
          bool matched = false;
          if (sit != saved.end()) {
            for (const auto& se : sit->second) {
              if (se.id == entry.id) {
                entry.in_flight = entry.in_flight || se.in_flight;
                entry.reach_from_dst &= se.reach_from_dst;
                entry.acyclic = entry.acyclic && se.acyclic;
                entry.wait_seq = std::max(entry.wait_seq, se.wait_seq);
                matched = true;
                break;
              }
            }
          }
          if (!matched) {
            entry.reach_from_dst = 0;
            entry.acyclic = false;
          }
        }
      }
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
        if (branch->else_body_) allocate_walk(*branch->else_body_);
        for (auto& [dir, vec] : active_by_dir) {
          const auto tit = then_active.find(dir);
          if (tit == then_active.end()) continue;
          for (auto& entry : vec) {
            for (const auto& te : tit->second) {
              if (te.id == entry.id) {
                entry.in_flight = entry.in_flight || te.in_flight;
                entry.reach_from_dst &= te.reach_from_dst;
                entry.acyclic = entry.acyclic && te.acyclic;
                entry.wait_seq = std::max(entry.wait_seq, te.wait_seq);
              }
            }
          }
          for (const auto& te : tit->second) {
            if (std::none_of(vec.begin(), vec.end(), [&](const ActiveId& e) { return e.id == te.id; })) {
              ActiveId merged = te;
              merged.reach_from_dst = 0;
              vec.push_back(merged);
            }
          }
        }
        for (const auto& [dir, then_vec] : then_active) {
          if (active_by_dir.count(dir)) continue;
          auto& dst_vec = active_by_dir[dir];
          for (const auto& te : then_vec) {
            ActiveId merged = te;
            merged.reach_from_dst = 0;
            dst_vec.push_back(merged);
          }
        }
        return;
      }
      if (const auto loop = As<ForStmt>(stmt)) {
        if (!loop->HasAttr(kClosedSyncChildAttr)) {
          auto saved = active_by_dir;
          allocate_walk(loop->body_);
          merge_zero_trip_loop(saved);
        }
        return;
      }
      if (const auto loop = As<WhileStmt>(stmt)) {
        auto saved = active_by_dir;
        allocate_walk(loop->body_);
        merge_zero_trip_loop(saved);
        return;
      }
      if (const auto rit = record_to_ow.find(stmt.get()); rit != record_to_ow.end()) {
        OneWayTransaction& ow = *rit->second;
        ow.record_seq = next_seq++;
        const SyncDirection dir{ow.source, ow.destination};
        auto& active = active_by_dir[dir];
        const bool may_reuse = ow.loop_stack.empty() || ow.needs_return_ack;
        int chosen = -1;
        if (may_reuse) {
          for (auto& entry : active) {
            if (!entry.in_flight && entry.acyclic && entry.loop_scope_id == ow.loop_scope_id &&
                (entry.reach_from_dst & RelayPipeBit(ow.source)) != 0) {
              chosen = entry.id;
              entry.in_flight = true;
              entry.reach_from_dst = 0;
              break;
            }
          }
        }
        if (chosen < 0) {
          for (int candidate : pool_for(dir)) {
            if (std::none_of(active.begin(), active.end(),
                             [&](const ActiveId& e) { return e.id == candidate; })) {
              chosen = candidate;
              active.push_back({chosen, true, 0, may_reuse, ow.loop_scope_id, 0});
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
              entry.reach_from_dst |= RelayPipeBit(ow.destination);
              if (ow.needs_return_ack) entry.reach_from_dst |= RelayPipeBit(ow.source);
            } else if (entry.wait_seq <= ow.record_seq &&
                       (entry.reach_from_dst & RelayPipeBit(ow.source)) != 0) {
              entry.reach_from_dst |= RelayPipeBit(ow.destination);
              if (ow.needs_return_ack) entry.reach_from_dst |= RelayPipeBit(ow.source);
            }
          }
        }
        if (ow.needs_return_ack) {
          const SyncDirection rev{ow.destination, ow.source};
          auto& rev_active = active_by_dir[rev];
          int ack_chosen = -1;
          for (auto& entry : rev_active) {
            if (!entry.in_flight && entry.acyclic && entry.loop_scope_id == ow.loop_scope_id &&
                (entry.reach_from_dst & RelayPipeBit(rev.first)) != 0) {
              ack_chosen = entry.id;
              entry.reach_from_dst = RelayPipeBit(rev.second);
              entry.wait_seq = seq_now;
              break;
            }
          }
          if (ack_chosen < 0) {
            for (int candidate : pool_for(rev)) {
              if (std::none_of(rev_active.begin(), rev_active.end(),
                               [&](const ActiveId& e) { return e.id == candidate; })) {
                ack_chosen = candidate;
                rev_active.push_back({ack_chosen, false, RelayPipeBit(rev.second), true, ow.loop_scope_id, seq_now});
                if (reservations_[rev].insert(ack_chosen).second) ++roles_count_;
                break;
              }
            }
          }
          CHECK(ack_chosen >= 0)
              << "Closed prefix role reservation unavailable; this is not a fixed-plan capacity lower bound";
          ow.return_id = ack_chosen;
        }
        return;
      }
      if (const auto cit = cut_edges.find(stmt.get()); cit != cut_edges.end()) {
        for (const auto& [src_e, dst_e] : cit->second) {
          for (auto& [dir, vec] : active_by_dir) {
            for (auto& entry : vec) {
              if ((entry.reach_from_dst & RelayPipeBit(src_e)) != 0) {
                entry.reach_from_dst |= RelayPipeBit(dst_e);
              }
            }
          }
        }
      }
    };
    allocate_walk(owner->body_);

    for (const auto& w : discovered.erased_waits) {
      commands_.emplace(w.get(), AssignedCommand{CommandKind::Erase, {}, -1});
    }
    for (const auto& d : discovered.erased_defs) {
      commands_.emplace(d.get(), AssignedCommand{CommandKind::Erase, {}, -1});
    }
    for (const auto& transaction : discovered.transactions) {
      for (size_t i = 0; i < 4; ++i) {
        const bool set = i % 2 == 0;
        const auto direction = i < 2 ? std::make_pair(transaction.source, transaction.destination)
                                     : std::make_pair(transaction.destination, transaction.source);
        const int id = assignment.at({direction, i >= 2});
        CHECK(commands_
                  .emplace(transaction.cuts[i].get(),
                           AssignedCommand{set ? CommandKind::SetFlag : CommandKind::WaitFlag, direction, id})
                  .second)
            << "Closed prefix protocol reuses an original static command cut";
      }
    }
    for (const auto& ow : discovered.one_way) {
      const SyncDirection fwd{ow.source, ow.destination};
      if (!ow.needs_return_ack) {
        CHECK(commands_.emplace(ow.record_cut.get(), AssignedCommand{CommandKind::SetFlag, fwd, ow.forward_id})
                  .second)
            << "One-way prefix protocol reuses an original static command cut";
        CHECK(commands_.emplace(ow.wait_cut.get(), AssignedCommand{CommandKind::WaitFlag, fwd, ow.forward_id})
                  .second)
            << "One-way prefix protocol reuses an original static command cut";
        ++one_way_sites_;
      } else {
        const SyncDirection rev{ow.destination, ow.source};
        CHECK(commands_.emplace(ow.record_cut.get(), AssignedCommand{CommandKind::SetFlag, fwd, ow.forward_id})
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
      const auto* barrier = LocalBarrier(local.pipe, owner->func_type_);
      CHECK(local.pipe != PipeType::S || native.scalar_prefix_serialized)
          << "Scalar current-prefix completion lacks native order evidence";
      CHECK(commands_.emplace(local.cuts[0].get(), AssignedCommand{CommandKind::Erase, {}, -1}).second)
          << "Local completion reuses an original static cut";
      CHECK(commands_
                .emplace(local.cuts[1].get(),
                         AssignedCommand{barrier ? CommandKind::Barrier : CommandKind::Erase,
                                         {local.pipe, local.pipe},
                                         -1})
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

 private:
  std::map<const Stmt*, AssignedCommand> commands_;
  SyncNativeReservations reservations_;
  std::vector<std::tuple<PipeType, PipeType, int>> native_claimed_;
  size_t roles_count_ = 0;
  size_t barriers_ = 0;
  size_t one_way_sites_ = 0;
  size_t one_way_fallback_sites_ = 0;
};

class Emit : public SyncStatementMutator {
 public:
  Emit(const Assignment& assignment, FunctionType core, const std::map<const ForStmt*, StmtPtr>& children,
       const std::vector<Transaction>&, const std::vector<LocalCompletion>&, bool strip_tokens = true)
      : assignment_(assignment), core_(core == FunctionType::AIV ? "aiv" : "aic"), children_(children) {
    (void)strip_tokens;
  }

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
          Command(cmd->kind == CommandKind::SetFlag ? "system.set_flag" : "system.wait_flag",
                  cmd->direction, cmd->id, statement->span_),
          statement->span_);
    }
    return SyncStatementMutator::VisitStmt_(statement);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    if (const auto cmd = assignment_.Find(statement)) {
      if (cmd->kind == CommandKind::Erase) return Empty(statement->span_);
      if (cmd->kind == CommandKind::Barrier) {
        return std::make_shared<EvalStmt>(
            OpRegistry::GetInstance().Create(LocalBarrier(cmd->direction.first,
                                                          core_ == "aiv" ? FunctionType::AIV
                                                                         : FunctionType::AIC),
                                             {}, {{"core_type", core_}}, statement->span_),
            statement->span_);
      }
      if (cmd->kind == CommandKind::SetWaitAck) {
        std::vector<StmtPtr> seq;
        seq.push_back(std::make_shared<EvalStmt>(
            Command("system.wait_flag", cmd->direction, cmd->id, statement->span_), statement->span_));
        seq.push_back(std::make_shared<EvalStmt>(
            Command("system.set_flag", cmd->ack_direction, cmd->ack_id, statement->span_),
            statement->span_));
        seq.push_back(std::make_shared<EvalStmt>(
            Command("system.wait_flag", cmd->ack_direction, cmd->ack_id, statement->span_),
            statement->span_));
        return std::make_shared<SeqStmts>(std::move(seq), statement->span_);
      }
      return std::make_shared<EvalStmt>(
          Command(cmd->kind == CommandKind::SetFlag ? "system.set_flag" : "system.wait_flag",
                  cmd->direction, cmd->id, statement->span_),
          statement->span_);
    }
    return SyncStatementMutator::VisitStmt_(statement);
  }

 private:
  StmtPtr Empty(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }
  CallPtr Command(const char* name, SyncDirection direction, int id, const Span& span) const {
    return OpRegistry::GetInstance().Create(
        name, {},
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
  if (discovered.transactions.empty() && discovered.local.empty() && discovered.one_way.empty() &&
      discovered.children.empty())
    return false;

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
  const size_t expected_uses = 2 * discovered.transactions.size() + discovered.local.size() +
                               discovered.one_way.size() + discovered.erased_waits.size();
  return check.uses == expected_uses;
}
}  // namespace

bool HasOnlyClosedPrefixSyntax(const FunctionPtr& function) {
  Discover discovered;
  discovered.VisitFunction(function);
  if (!HasMatchingClosedPrefixUses(function, discovered)) return false;
  if (function->HasAttr(kAttrAutosyncAnalysisPolicy) &&
      function->GetAttr<std::string>(kAttrAutosyncAnalysisPolicy) == "conservative_prefix")
    return true;
  if (!discovered.transactions.empty() || !discovered.children.empty()) return true;
  return discovered.has_while || !discovered.has_payload || (discovered.has_if && !discovered.has_for);
}

std::optional<PhysicalSyncRealization> TryLowerClosedPrefixSync(
    const FunctionPtr& function, const std::string& target,
    const std::optional<std::vector<int>>& eligible_ids, const SyncNativeReservations& extra_reservations) {
  Discover discovered;
  discovered.VisitFunction(function);
  if (!HasMatchingClosedPrefixUses(function, discovered)) return std::nullopt;
  CHECK(!function->requires_runtime_binding_ && !function->HasAttr(kAttrExternalSource) &&
        !function->HasAttr(kAttrBuiltinTemplateDir))
      << "Closed prefix protocols cannot replace an external invocation owner";

  NativeOperations native(function->func_type_, target);
  native.VisitFunction(function);
  const Assignment assignment(function, discovered, native.payloads, target, eligible_ids,
                              extra_reservations);

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
    for (const auto source : pipes) {
      for (const auto destination : pipes) {
        if (source == destination) continue;
        CHECK(entry->second.count({source, destination}))
            << "Closed child entry lacks a participating-pipe acknowledgment; cross-visit rearm unproved";
      }
    }
    auto unmarked = std::make_shared<ForStmt>(*child);
    unmarked->attrs_.erase(
        std::remove_if(unmarked->attrs_.begin(), unmarked->attrs_.end(),
                       [](const auto& attribute) { return attribute.first == kClosedSyncChildAttr; }),
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
    auto lowered = LowerLogicalSync(actual, target, eligible_ids, {}, std::nullopt, {}, external_policy,
                                    std::nullopt, assignment.Reservations());
    const auto original_body = As<SeqStmts>(actual->body_);
    const auto lowered_body = As<SeqStmts>(lowered.function->body_);
    INTERNAL_CHECK(original_body && lowered_body) << "Internal error: child binding body is not a sequence";
    std::set<const Var*> descriptor_variables;
    for (const auto& statement : original_body->stmts_)
      if (const auto assignment_stmt = As<AssignStmt>(statement))
        descriptor_variables.insert(assignment_stmt->var_.get());
    std::vector<StmtPtr> recipe;
    for (const auto& statement : lowered_body->stmts_) {
      if (As<ReturnStmt>(statement)) continue;
      const auto assignment_stmt = As<AssignStmt>(statement);
      if (assignment_stmt && descriptor_variables.count(assignment_stmt->var_.get())) continue;
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

  Emit emit(assignment, function->func_type_, children, discovered.transactions, discovered.local);
  PhysicalSyncRealization result;
  result.function = emit.Rewrite(function);
  result.set_sites = 2 * discovered.transactions.size() + assignment.OneWaySites() +
                     assignment.OneWayFallbackSites() + child_sets;
  result.wait_sites = 2 * discovered.transactions.size() + assignment.OneWaySites() +
                      assignment.OneWayFallbackSites() + child_waits;
  result.barrier_sites = assignment.Barriers() + child_barriers;
  result.reserved_lanes = assignment.Roles() + child_lanes;
  result.native_reserved_ids = assignment.NativeClaimed();
  return result;
}
}  // namespace pypto::ir::sync
