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

// Specialize the already assigned and emitted commands inside a verified work
// item. Each original coordinate still executes the same ordered local commands
// and payloads. The proof is local to the child loop, so the work-item tape and
// the allocated events, FIFO ordinals and seam proofs carry over unchanged.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/peer_local_work_loop.h"
#include "src/ir/transforms/utils/physical_sync_schedule.h"
#include "src/ir/transforms/utils/sync_coordinate_eval.h"
#include "src/ir/transforms/utils/sync_loop_trip_proof.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace {

constexpr int64_t kMaxTrips = 16;
constexpr size_t kLoopGrowthAllowance = 16;
struct GrowthBudget {
  size_t payloads, nodes;
};

using VisitMask = uint32_t;
struct GuardTruth {
  VisitMask known = 0;
  VisitMask yes = 0;
};
using GuardFacts = std::unordered_map<const IfStmt*, GuardTruth>;
using GuardDecisions = std::unordered_map<const IfStmt*, bool>;

bool IsLocalCommand(const CallPtr& call) {
  return call && call->args_.empty() &&
         (IsOp(call, "system.set_flag") || IsOp(call, "system.wait_flag") ||
          core_affinity::IsPipeBarrier(call));
}

// Inventory allowed statement shapes and count operand occurrences as well as
// statements before considering any copies. A child with a descriptor
// allocation, nested loop, nonempty yield, or unusual control stays guarded.
class WorkItemShape : public IRVisitor {
  size_t remaining_;
  bool saw_call_ = false;
  std::unordered_set<const Var*> definitions_;

  bool Count() {
    if (remaining_ == 0) {
      valid = false;
      return false;
    }
    --remaining_;
    return true;
  }

  bool CheckCall(const CallPtr& call) {
    return !As<GlobalVar>(call->op_) && !core_affinity::IsNoDuplicateCall(call) &&
           !cross_core_pipe::IsPeerPipeSetup(call) && !IsOp(call, "tile.alloc") &&
           !IsOp(call, "tile.create") && !IsOp(call, "buffer.alloc") && !IsOp(call, "buffer.alloc_multi");
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!valid || !expr || !Count()) {
      return;
    }
    const auto call = As<Call>(expr);
    saw_call_ |= static_cast<bool>(call);
    if (call && !IsLocalCommand(call)) ++payloads;
    valid = !IsA<Submit>(expr) && (!call || CheckCall(call));
    if (valid) {
      IRVisitor::VisitExpr(expr);
    }
  }

 public:
  bool valid = true;
  GuardFacts guards;
  size_t payloads = 0;

  explicit WorkItemShape(size_t budget) : remaining_(budget) {}

  size_t Remaining() const { return remaining_; }

  bool Collect(const StmtPtr& stmt) {
    if (!valid || !Count()) {
      return false;
    }
    if (const auto seq = As<SeqStmts>(stmt)) {
      bool commands = true;
      for (const auto& child : seq->stmts_) {
        commands &= Collect(child);
      }
      return commands;
    }
    if (const auto branch = As<IfStmt>(stmt)) {
      valid = branch->return_vars_.empty();
      VisitExpr(branch->condition_);
      const bool then_commands = Collect(branch->then_body_);
      const bool else_commands = !branch->else_body_ || Collect(*branch->else_body_);
      if (then_commands && else_commands) {
        // One decision rewrites every occurrence of this node. A shared
        // branch reached under different enclosing predicates can have
        // different truth values on the two subsets of visits.
        valid = guards.try_emplace(branch.get()).second;
        return valid;
      }
      return false;
    }
    if (const auto eval = As<EvalStmt>(stmt)) {
      const auto call = As<Call>(eval->expr_);
      valid = call != nullptr;
      VisitExpr(eval->expr_);
      return IsLocalCommand(call);
    }
    if (const auto assign = As<AssignStmt>(stmt)) {
      valid = definitions_.insert(assign->var_.get()).second;
      VisitExpr(assign->var_);
      saw_call_ = false;
      VisitExpr(assign->value_);
      return valid && !saw_call_ && IsCoordinateScalar(assign->value_);
    }
    valid = false;
    return false;
  }
};

// Follow already emitted scalar SSA recipes, and use the existing constant
// folder only when every operand is known at this original visit. No symbolic
// identities are needed, and unknown/non-scalar expressions remain in place.
class VisitGuardFacts {
  const VisitMask visit_;
  GuardFacts& guards_;
  std::unordered_map<const Var*, ExprPtr> bindings_;
  std::unordered_map<const Expr*, ExprPtr> expressions_;

  ExprPtr Fold(const ExprPtr& expr) {
    return FoldCoordinate(expr, [&](const ExprPtr& operand) -> ExprPtr {
      if (const auto var = As<Var>(operand)) {
        const auto it = bindings_.find(var.get());
        return it == bindings_.end() ? nullptr : it->second;
      }
      return Evaluate(operand);
    });
  }

  ExprPtr Evaluate(const ExprPtr& expr) {
    if (const auto found = expressions_.find(expr.get()); found != expressions_.end()) {
      return found->second;
    }
    return expressions_.emplace(expr.get(), Fold(expr)).first->second;
  }

 public:
  VisitGuardFacts(VisitMask visit, GuardFacts& guards, const VarPtr& iv, const ExprPtr& value)
      : visit_(visit), guards_(guards), bindings_{{iv.get(), value}} {}

  void Collect(const StmtPtr& stmt) {
    if (const auto assign = As<AssignStmt>(stmt)) {
      bindings_.emplace(assign->var_.get(), Evaluate(assign->value_));
    } else if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) {
        Collect(child);
      }
    } else if (const auto branch = As<IfStmt>(stmt)) {
      const auto at = guards_.find(branch.get());
      if (at != guards_.end()) {
        if (const auto verdict = As<ConstBool>(Evaluate(branch->condition_))) {
          at->second.known |= visit_;
          at->second.yes |= verdict->value_ ? visit_ : 0;
        }
      }
      // Definitions are unique SSA binders. Arm-local definitions cannot
      // escape except via result variables, which this shape excludes.
      Collect(branch->then_body_);
      if (branch->else_body_) {
        Collect(*branch->else_body_);
      }
    }
  }
};

struct Projection {
  GuardDecisions decisions;
  int64_t savings = 0;

  void Collect(const StmtPtr& stmt, const GuardFacts& guards, VisitMask possible, VisitMask certain) {
    if (const auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) {
        Collect(child, guards, possible, certain);
      }
    } else if (const auto branch = As<IfStmt>(stmt)) {
      Branch(branch, guards, possible, certain);
    }
  }

  void Branch(const IfStmtPtr& branch, const GuardFacts& guards, VisitMask possible, VisitMask certain) {
    if (!possible) {
      return;
    }
    GuardTruth fact;
    if (const auto at = guards.find(branch.get()); at != guards.end()) {
      fact = at->second;
    }
    const auto no = fact.known & ~fact.yes;
    if ((possible & fact.yes) == possible || (possible & no) == possible) {
      decisions.emplace(branch.get(), (possible & fact.yes) == possible);
      savings += __builtin_popcount(certain);
    }
    Collect(branch->then_body_, guards, possible & ~no, certain & fact.yes);
    if (branch->else_body_) {
      Collect(*branch->else_body_, guards, possible & ~fact.yes, certain & no);
    }
  }
};

class FoldGuards : public SyncStatementMutator {
  const GuardDecisions& decisions_;

  StmtPtr VisitStmt_(const IfStmtPtr& branch) override {
    const auto found = decisions_.find(branch.get());
    if (found == decisions_.end()) {
      return SyncStatementMutator::VisitStmt_(branch);
    }
    if (found->second) {
      return VisitStmt(branch->then_body_);
    }
    if (branch->else_body_) {
      return VisitStmt(*branch->else_body_);
    }
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, branch->span_);
  }

 public:
  explicit FoldGuards(const GuardDecisions& decisions) : decisions_(decisions) {}
};

struct Candidate {
  std::array<Projection, 2> copies;
  std::optional<int64_t> split;
  int64_t saving = 0;
};

void Consider(Candidate& best, const StmtPtr& body, const GuardFacts& facts, VisitMask first,
              VisitMask second, const std::optional<int64_t>& split, int64_t loop_saving) {
  Candidate candidate;
  candidate.split = split;
  candidate.copies[0].Collect(body, facts, first, first);
  candidate.copies[1].Collect(body, facts, second, second);
  const auto removed = candidate.copies[0].savings + candidate.copies[1].savings;
  candidate.saving = removed + loop_saving;
  if (removed > 0 && candidate.saving > best.saving) {
    best = std::move(candidate);
  }
}

class SpecializedLoop {
  const ForStmtPtr& loop_;
  SyncTokenNames& names_;

  VarPtr Coordinate() const {
    return std::make_shared<Var>(names_.Next(loop_->loop_var_->name_hint_ + "_sync_visit_"),
                                 loop_->loop_var_->GetType(), loop_->span_);
  }

  StmtPtr CloneVisit(const StmtPtr& body, const Projection& projection, const VarPtr& coordinate) const {
    return DeepClone(FoldGuards(projection.decisions).VisitStmt(body), {{loop_->loop_var_.get(), coordinate}},
                     true, [this](const std::string& name) { return names_.Next(name + "_sync_visit_"); })
        .cloned_body;
  }

  ForStmtPtr Emit(const VarPtr& iv, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step,
                  const StmtPtr& body, bool first) const {
    return std::make_shared<ForStmt>(iv, start, stop, step, std::vector<IterArgPtr>{}, body,
                                     std::vector<VarPtr>{}, loop_->span_, ForKind::Sequential, loop_->attrs_,
                                     first ? loop_->leading_comments_ : std::vector<std::string>{});
  }

 public:
  SpecializedLoop(const ForStmtPtr& loop, SyncTokenNames& names) : loop_(loop), names_(names) {}

  StmtPtr Run(const LiteralIndexLoop& frame, const Candidate& candidate, const StmtPtr& body,
              const YieldStmtPtr& yield) const {
    const auto& span = loop_->span_;
    const auto append_yield = [&](std::vector<StmtPtr> statements) {
      if (yield) {
        statements.push_back(yield);
      }
      return SeqStmts::Flatten(std::move(statements), span);
    };
    const auto first_iv = Coordinate(), second_iv = Coordinate();
    const auto first = CloneVisit(body, candidate.copies[0], first_iv);
    const auto second = CloneVisit(body, candidate.copies[1], second_iv);
    if (candidate.split) {
      const auto boundary = frame.At(*candidate.split, span);
      return SeqStmts::Flatten(
          {Emit(first_iv, loop_->start_, boundary, loop_->step_, append_yield({first}), true),
           Emit(second_iv, boundary, loop_->stop_, loop_->step_, append_yield({second}), false)},
          span);
    }
    const auto offset = std::make_shared<AssignStmt>(
        second_iv, std::make_shared<Add>(first_iv, loop_->step_, DataType::INDEX, span), span);
    const auto step = std::make_shared<ConstInt>(2 * frame.step, DataType::INDEX, span);
    return Emit(first_iv, loop_->start_, loop_->stop_, step, append_yield({first, offset, second}), true);
  }
};

StmtPtr SpecializeLeaf(const ForStmtPtr& loop, SyncTokenNames& names, GrowthBudget& remaining) {
  const auto frame = LiteralIndexLoop::From(loop.get(), kMaxTrips);
  if (!frame || frame->count < 2 || !loop->iter_args_.empty() || !loop->return_vars_.empty() ||
      !loop->attrs_.empty() || loop->hardware_loop_policy_ != HardwareLoopPolicy::Auto ||
      remaining.nodes <= kLoopGrowthAllowance) {
    return loop;
  }
  // Both candidate layouts emit at most two complete copies. Charge the
  // full replacement for control and the additional copy for payload sites,
  // using the same per-leaf limits as ordinary local scheduling.
  const size_t body_limit = (remaining.nodes - kLoopGrowthAllowance) / 2;
  WorkItemShape shape(body_limit);
  auto stmts = transform_utils::FlattenToStmts(loop->body_);
  const auto yield = stmts.empty() ? nullptr : As<YieldStmt>(stmts.back());
  if (yield && !yield->value_.empty()) {
    return loop;
  }
  if (yield) {
    stmts.pop_back();
  }
  const auto body = std::make_shared<SeqStmts>(std::move(stmts), loop->body_->span_);
  shape.Collect(body);
  if (!shape.valid || shape.guards.empty() || shape.payloads > remaining.payloads) {
    return loop;
  }
  for (int64_t visit = 0; visit < frame->count; ++visit) {
    VisitGuardFacts(VisitMask{1} << visit, shape.guards, loop->loop_var_, frame->At(visit, loop->span_))
        .Collect(body);
  }
  Candidate best;
  const auto all = (VisitMask{1} << frame->count) - 1;
  for (int64_t split = 1; split < frame->count; ++split) {
    const auto first = (VisitMask{1} << split) - 1;
    // Splitting adds one loop initialization and one exit check.
    Consider(best, body, shape.guards, first, all ^ first, split, -2);
  }
  if (frame->count % 2 == 0 && frame->step <= std::numeric_limits<int64_t>::max() / 2) {
    constexpr VisitMask kEvenVisits = 0x55555555;
    const auto first = all & kEvenVisits;
    // Two residues share a header and add one scalar coordinate per pair.
    Consider(best, body, shape.guards, first, all ^ first, std::nullopt, frame->count / 2);
  }
  if (best.saving <= 0) {
    return loop;
  }
  remaining.nodes -= kLoopGrowthAllowance + 2 * (body_limit - shape.Remaining());
  remaining.payloads -= shape.payloads;
  return SpecializedLoop(loop, names).Run(*frame, best, body, yield);
}

StmtPtr SpecializeItem(const StmtPtr& body, SyncTokenNames& names, GrowthBudget& remaining) {
  bool changed = false;
  const auto specialize = [&](const StmtPtr& stmt) -> StmtPtr {
    if (const auto loop = As<ForStmt>(stmt)) {
      const auto result = SpecializeLeaf(loop, names, remaining);
      changed |= result != stmt;
      return result;
    }
    return stmt;
  };
  if (const auto seq = As<SeqStmts>(body)) {
    auto result = RewriteSyncSequence(seq, specialize);
    return changed ? result : body;
  }
  return specialize(body);
}

class WorkGuardSpecializer : public SyncStatementMutator {
  const std::set<const Var*>& work_ivs_;
  SyncTokenNames names_;
  GrowthBudget remaining_;

  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    if (!work_ivs_.count(loop->loop_var_.get())) {
      return loop;
    }
    auto body = SpecializeItem(loop->body_, names_, remaining_);
    if (body == loop->body_) {
      return loop;
    }
    auto copy = MutableCopy(loop);
    copy->body_ = std::move(body);
    return copy;
  }

 public:
  WorkGuardSpecializer(const FunctionPtr& owner, const std::set<const Var*>& work_ivs,
                       const SyncSchedulePolicy& policy)
      : work_ivs_(work_ivs),
        names_(owner),
        remaining_{policy.max_extra_payload_sites, policy.max_extra_control_nodes} {}
};

}  // namespace

FunctionPtr SpecializePeerWorkGuards(const FunctionPtr& owner, const std::set<const Var*>& work_ivs) {
  const auto policy = physical::LocalSyncSchedulePolicy();
  if (work_ivs.empty() || policy.no_clone) {
    return owner;
  }
  return WorkGuardSpecializer(owner, work_ivs, policy).VisitFunction(owner);
}

}  // namespace pypto::ir::sync
