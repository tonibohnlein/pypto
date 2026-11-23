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

#include "src/ir/transforms/utils/regional_sync_handoff.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/physical_sync_replay.h"
#include "src/ir/transforms/utils/sync_coordinate_eval.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"

namespace pypto::ir::sync {
namespace {

using replay::QueryQueues;
using replay::Ranks;
using replay::Step;
using replay::StepKind;
using Units = std::vector<std::vector<StmtPtr>>;

// Bound expansion before publishing anything. In addition to effectful cuts,
// charge control, descriptors and empty loop iterations to the same budget.
constexpr size_t kReplayWorkLimit = 32768;
constexpr size_t kReplayCandidateLimit = 32;
constexpr size_t kReplayDepthLimit = 128;
constexpr size_t kReplayVisitChoiceLimit = 5;

std::optional<Step> ReadCommand(const CallPtr& call, FunctionType type) {
  if (!call || !call->args_.empty()) return std::nullopt;
  if (IsOp(call, "system.set_flag") || IsOp(call, "system.wait_flag")) {
    const auto source = call->GetKwarg<int>("src_pipe", -1);
    const auto dest = call->GetKwarg<int>("dst_pipe", -1);
    const auto id = call->GetKwarg<int>("event_id", -1);
    if (source < 0 || dest < 0 || static_cast<size_t>(source) >= kPipeCount ||
        static_cast<size_t>(dest) >= kPipeCount || source == dest || id < 0 ||
        call->GetKwarg<std::string>("core_type", "") != (type == FunctionType::AIC ? "aic" : "aiv"))
      return std::nullopt;
    return Step{IsOp(call, "system.set_flag") ? StepKind::Set : StepKind::Wait,
                {static_cast<PipeType>(source), static_cast<PipeType>(dest)},
                id};
  }
  const auto local = type == FunctionType::AIC
                         ? std::vector<PipeType>{PipeType::MTE2, PipeType::MTE1, PipeType::M, PipeType::FIX}
                         : std::vector<PipeType>{PipeType::MTE2, PipeType::MTE3, PipeType::V};
  for (const auto pipe : local) {
    if (IsOp(call, RequireLocalSyncBarrier(pipe, type))) {
      if (call->GetKwarg<std::string>("core_type", "") != (type == FunctionType::AIC ? "aic" : "aiv"))
        return std::nullopt;
      return Step{StepKind::Barrier, {pipe, pipe}};
    }
  }
  return std::nullopt;
}

CallPtr EvalCall(const StmtPtr& stmt) {
  const auto eval = As<EvalStmt>(stmt);
  return eval ? As<Call>(eval->expr_) : nullptr;
}

Units ReadHandoff(const std::vector<StmtPtr>& handoff, FunctionType type) {
  Units units;
  for (size_t at = 0; at < handoff.size(); ++at) {
    const auto command = ReadCommand(EvalCall(handoff[at]), type);
    if (!command) return {};
    if (command->kind == StepKind::Barrier) {
      units.push_back({handoff[at]});
      continue;
    }
    if (command->kind != StepKind::Set || at + 1 == handoff.size()) return {};
    const auto wait = ReadCommand(EvalCall(handoff[at + 1]), type);
    if (!wait || wait->kind != StepKind::Wait || wait->direction != command->direction ||
        wait->id != command->id)
      return {};
    units.push_back({handoff[at], handoff[at + 1]});
    ++at;
  }
  return units;
}

struct TapeStep {
  Step step;
  uint32_t candidate = 0;
  uint32_t required_visits = 0;
  Ranks original_prefix{};
};

// Replay concrete control and loops proved to execute at most once. An unknown
// induction value is sufficient when the body's command sequence is still known:
// the backend proves native pipe ownership independently of payload operands.
class InvocationTape {
 public:
  InvocationTape(FunctionType core, const std::string& target, const std::map<const Stmt*, uint32_t>& bits,
                 const std::map<const ForStmt*, HandoffReplayTrip>& single_visit_loops)
      : core_(core), target_(target), bits_(bits), single_visit_loops_(single_visit_loops) {}

  bool Build(const SeqStmtsPtr& body) {
    if (!body || body->stmts_.empty()) return false;
    const auto terminal = As<ReturnStmt>(body->stmts_.back());
    if (!terminal || !std::all_of(terminal->value_.begin(), terminal->value_.end(),
                                  [&](const ExprPtr& value) { return PlainExpression(value); }))
      return false;
    for (size_t i = 0; i + 1 < body->stmts_.size(); ++i) {
      if (!Read(body->stmts_[i])) return false;
    }
    return true;
  }

  size_t PathCount() const { return size_t{1} << visit_choices_; }
  std::vector<TapeStep> steps;

 private:
  FunctionType core_;
  const std::string& target_;
  const std::map<const Stmt*, uint32_t>& bits_;
  const std::map<const ForStmt*, HandoffReplayTrip>& single_visit_loops_;
  std::map<const Var*, ExprPtr> coordinates_;
  size_t work_ = 0;
  size_t visit_choices_ = 0;
  uint32_t active_visits_ = 0;
  std::map<const Call*, std::optional<Step>> calls_;

  // Memoize subtree costs by identity so operand DAGs are inspected once.
  // Calls and submit expressions are only admitted as qualified leaf issues.
  class ExpressionCosts : public IRVisitor {
   public:
    size_t Measure(const ExprPtr& expr) { return MeasureShape(expr).nodes; }

   protected:
    void VisitExpr(const ExprPtr& expr) override {
      if (shape_.nodes >= invalid) return;
      const auto operand = MeasureShape(expr);
      shape_.depth = std::max(shape_.depth, operand.depth + 1);
      shape_.nodes =
          shape_.depth > kReplayDepthLimit ? invalid : std::min(invalid, shape_.nodes + operand.nodes);
    }
    void VisitExpr_(const CallPtr&) override { shape_.nodes = invalid; }
    void VisitExpr_(const SubmitPtr&) override { shape_.nodes = invalid; }

   private:
    struct Shape {
      size_t nodes, depth;
    };

    static constexpr size_t invalid = kReplayWorkLimit + 1;
    std::map<const Expr*, Shape> shapes_;
    Shape shape_{};
    size_t active_depth_ = 0;

    Shape MeasureShape(const ExprPtr& expr) {
      if (const auto found = shapes_.find(expr.get()); found != shapes_.end()) {
        return found->second;
      }
      if (active_depth_ == kReplayDepthLimit || shapes_.size() >= kReplayWorkLimit) return {invalid, 0};
      const auto outer = std::exchange(shape_, Shape{1, 1});
      ++active_depth_;
      IRVisitor::VisitExpr(expr);
      --active_depth_;
      const auto measured = std::exchange(shape_, outer);
      shapes_.emplace(expr.get(), measured);
      return measured;
    }
  } expression_costs_;

  bool Read(const StmtPtr& stmt, size_t depth = 0) {
    if (++work_ > kReplayWorkLimit || depth == kReplayDepthLimit) return false;
    if (const auto seq = As<SeqStmts>(stmt)) {
      return std::all_of(seq->stmts_.begin(), seq->stmts_.end(),
                         [&](const StmtPtr& child) { return Read(child, depth + 1); });
    }
    if (const auto loop = As<ForStmt>(stmt)) return ReadLoop(loop, depth + 1);
    if (const auto branch = As<IfStmt>(stmt)) {
      if (!branch->return_vars_.empty()) return false;
      if (const auto result = As<ConstBool>(Evaluate(branch->condition_))) {
        return result->value_ ? Read(branch->then_body_, depth + 1)
                              : (!branch->else_body_ || Read(*branch->else_body_, depth + 1));
      }
      return false;
    }
    if (const auto yield = As<YieldStmt>(stmt)) return yield->value_.empty();
    if (const auto assign = As<AssignStmt>(stmt)) {
      if (const auto call = As<Call>(assign->value_)) {
        coordinates_.erase(assign->var_.get());
        return ReadCall(call, stmt);
      }
      if (!PlainExpression(assign->value_)) return false;
      if (IsCoordinateScalar(assign->var_)) {
        auto value = Evaluate(assign->value_);
        if (value && GetScalarDtype(value) == GetScalarDtype(assign->var_))
          coordinates_.insert_or_assign(assign->var_.get(), std::move(value));
        else
          coordinates_.erase(assign->var_.get());
      }
      return true;
    }
    if (const auto eval = As<EvalStmt>(stmt)) {
      if (const auto call = As<Call>(eval->expr_)) return ReadCall(call, stmt);
      return PlainExpression(eval->expr_);
    }
    return false;
  }

  bool PlainExpression(const ExprPtr& expr) {
    const auto cost = expression_costs_.Measure(expr);
    if (cost > kReplayWorkLimit - work_) return false;
    work_ += cost;
    return true;
  }

  ExprPtr Evaluate(const ExprPtr& expr) {
    // Each evaluation uses the concrete bindings at this cut. The memo
    // does not outlive updates to induction variables or scalar SSA names.
    std::map<const Expr*, ExprPtr> memo;
    const auto visit = [&](const auto& self, const ExprPtr& value, size_t depth) -> ExprPtr {
      if (work_ == kReplayWorkLimit || depth == kReplayDepthLimit) return nullptr;
      ++work_;
      if (const auto found = memo.find(value.get()); found != memo.end()) {
        return found->second;
      }
      auto folded = FoldCoordinate(value, [&](const ExprPtr& operand) -> ExprPtr {
        if (const auto var = As<Var>(operand)) {
          const auto bound = coordinates_.find(var.get());
          return bound == coordinates_.end() ? nullptr : bound->second;
        }
        return self(self, operand, depth + 1);
      });
      memo.emplace(value.get(), folded);
      return folded;
    };
    return visit(visit, expr, 0);
  }

  bool ReadLoop(const ForStmtPtr& loop, size_t body_depth) {
    if (loop->kind_ != ForKind::Sequential || loop->hardware_loop_policy_ != HardwareLoopPolicy::Auto ||
        !loop->iter_args_.empty() || !loop->return_vars_.empty() ||
        GetScalarDtype(loop->loop_var_) != DataType::INDEX)
      return false;
    const auto start = As<ConstInt>(Evaluate(loop->start_));
    const auto stop = As<ConstInt>(Evaluate(loop->stop_));
    const auto step = As<ConstInt>(Evaluate(loop->step_));
    if (!start || !stop || !step) {
      return ReadSingleVisit(loop, body_depth, start);
    }
    if (step->value_ <= 0) {
      return false;
    }
    for (__int128 iv = start->value_; iv < stop->value_; iv += step->value_) {
      // Reject the overflowing final update as well as oversized tapes.
      if (++work_ > kReplayWorkLimit || iv + step->value_ > std::numeric_limits<int64_t>::max()) return false;
      coordinates_.insert_or_assign(loop->loop_var_.get(),
                                    arith::MakeConstInt(static_cast<int64_t>(iv), DataType::INDEX));
      if (!Read(loop->body_, body_depth)) return false;
    }
    coordinates_.erase(loop->loop_var_.get());
    return true;
  }

  bool ReadSingleVisit(const ForStmtPtr& loop, size_t body_depth, const ConstIntPtr& start) {
    const auto fact = single_visit_loops_.find(loop.get());
    if (fact == single_visit_loops_.end() || !PlainExpression(loop->start_) ||
        !PlainExpression(loop->stop_) || !PlainExpression(loop->step_)) {
      return false;
    }
    const auto enclosing_visits = active_visits_;
    if (fact->second == HandoffReplayTrip::ZeroOrOne) {
      if (visit_choices_ == kReplayVisitChoiceLimit) {
        return false;
      }
      active_visits_ |= uint32_t{1} << visit_choices_++;
    }
    if (start) {
      coordinates_.insert_or_assign(loop->loop_var_.get(), start);
    } else {
      coordinates_.erase(loop->loop_var_.get());
    }
    // A skipped visit contributes no commands. Bindings defined inside the
    // body cannot escape a loop without carries, so following cuts use the
    // same scalar environment regardless of the visit.
    const bool supported = Read(loop->body_, body_depth);
    coordinates_.erase(loop->loop_var_.get());
    active_visits_ = enclosing_visits;
    return supported;
  }

  bool ReadCall(const CallPtr& call, const StmtPtr& stmt) {
    auto found = calls_.find(call.get());
    if (found == calls_.end()) {
      if (!std::all_of(call->args_.begin(), call->args_.end(),
                       [&](const ExprPtr& arg) { return PlainExpression(arg); }))
        return false;
      auto qualified = ReadCommand(call, core_);
      if (!qualified && !backend::QualifyBufferSyncNonPayload(call, target_)) {
        const auto payload = backend::QualifyBufferSyncOperation(call, core_, target_);
        const auto& impl = payload.Implementation();
        const auto pipe = impl.Pipe();
        if (static_cast<size_t>(pipe) >= kPipeCount || !impl.Claims().empty() ||
            !impl.ImplicitEffects().empty())
          return false;
        qualified = Step{StepKind::Payload, {pipe, pipe}};
      }
      found = calls_.emplace(call.get(), qualified).first;
    }
    if (found->second) {
      const auto candidate = bits_.find(stmt.get());
      steps.push_back({*found->second, candidate == bits_.end() ? 0 : candidate->second, active_visits_, {}});
    }
    return true;
  }
};

bool CaptureBaseline(std::vector<TapeStep>& tape, bool scalar_serial) {
  QueryQueues queues(scalar_serial);
  for (auto& item : tape) {
    if (item.step.kind == StepKind::Payload)
      item.original_prefix = queues.IssuePrefix(item.step.direction.second);
    if (!queues.Apply(item.step)) return false;
  }
  return queues.AllConsumed();
}

std::optional<std::vector<std::vector<TapeStep>>> CapturePaths(const InvocationTape& tape,
                                                               bool scalar_serial) {
  const auto paths = tape.PathCount();
  if (tape.steps.size() > kReplayWorkLimit / paths) {
    return std::nullopt;
  }
  std::vector<std::vector<TapeStep>> baselines(paths);
  // Independent choices may include infeasible combinations. Every feasible
  // path is present, and skipped loops leave the entering queues untouched.
  for (size_t path = 0; path < paths; ++path) {
    auto& baseline = baselines[path];
    for (const auto& item : tape.steps) {
      if ((path & item.required_visits) == item.required_visits) {
        baseline.push_back(item);
      }
    }
    if (!CaptureBaseline(baseline, scalar_serial)) {
      return std::nullopt;
    }
  }
  return baselines;
}

bool PreservesOrder(const std::vector<TapeStep>& tape, uint32_t removed, bool scalar_serial) {
  QueryQueues queues(scalar_serial);
  for (const auto& item : tape) {
    if (removed & item.candidate) {
      queues.Skip();
      continue;
    }
    if (item.step.kind == StepKind::Payload &&
        item.original_prefix != queues.IssuePrefix(item.step.direction.second))
      return false;
    if (!queues.Apply(item.step)) return false;
  }
  return queues.AllConsumed();
}

class RemoveHandoffCommands : public SyncStatementMutator {
 public:
  RemoveHandoffCommands(const std::map<const Stmt*, uint32_t>& bits, uint32_t removed)
      : bits_(bits), removed_(removed) {}

  size_t RemovedSites() const { return removed_sites_; }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    std::vector<StmtPtr> preserved;
    preserved.reserve(sequence->stmts_.size());
    bool changed = false;
    for (const auto& stmt : sequence->stmts_) {
      const auto bit = bits_.find(stmt.get());
      if (bit != bits_.end() && (bit->second & removed_) != 0) {
        ++removed_sites_;
        changed = true;
        continue;
      }
      auto next = VisitStmt(stmt);
      changed |= next != stmt;
      preserved.push_back(std::move(next));
    }
    if (!changed) {
      return sequence;
    }
    return std::make_shared<SeqStmts>(std::move(preserved), sequence->span_);
  }

 private:
  const std::map<const Stmt*, uint32_t>& bits_;
  uint32_t removed_;
  size_t removed_sites_ = 0;
};

}  // namespace

void ReduceRegionalSyncHandoffs(PhysicalSyncRealization& result, const std::string& target,
                                const std::vector<std::vector<StmtPtr>>& handoffs,
                                const std::map<const ForStmt*, HandoffReplayTrip>& single_visit_loops) {
  if (handoffs.empty()) return;
  const auto body = As<SeqStmts>(result.function->body_);
  if (!body) return;
  const auto core = result.function->func_type_;
  const auto& execution = backend::RequireLocalSyncExecutionContract(target);
  if (!execution.set_completes_source_prefix || !execution.wait_consumes_before_advance) return;
  const auto serial = execution.scalar_prefix_serialized;

  Units candidates;
  for (const auto& handoff : handoffs) {
    auto units = ReadHandoff(handoff, core);
    if (units.size() > kReplayCandidateLimit - candidates.size()) return;
    for (auto& unit : units) candidates.push_back(std::move(unit));
  }
  if (candidates.empty()) return;

  std::map<const Stmt*, uint32_t> bits;
  for (size_t i = 0; i < candidates.size(); ++i) {
    for (const auto& stmt : candidates[i]) {
      if (!bits.emplace(stmt.get(), uint32_t{1} << i).second) return;
    }
  }
  InvocationTape tape(core, target, bits, single_visit_loops);
  try {
    if (!tape.Build(body)) {
      return;
    }
  } catch (const pypto::ValueError&) {
    return;
  }
  auto baselines = CapturePaths(tape, serial);
  if (!baselines) {
    return;
  }

  // Compare the original prefix at each preserved issue, using aligned tape
  // ranks. Bank reuse is independently checked at each SET, including the
  // later child’s first publication. The owner still emits the terminal
  // all-pipe drain after the replayed body and all banks must be consumed.
  const auto all = (uint64_t{1} << candidates.size()) - 1;
  // Try empty and single-unit cuts before a greedy pass. Some relay groups
  // become redundant only when several interdependent units vanish together.
  const auto accepts = [&](uint32_t mask) {
    size_t removed_pairs = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if ((mask & (uint32_t{1} << i)) && candidates[i].size() == 2) ++removed_pairs;
    }
    // Keep a complete literal flag ownership marker for the optional leaf
    // path. Deleting the last pair would send the original owner to PTOAS.
    return removed_pairs < result.set_sites && removed_pairs < result.wait_sites &&
           std::all_of(
               baselines->begin(), baselines->end(),
               [&](const auto& path) { return PreservesOrder(path, mask, serial); });
  };
  auto removed = static_cast<uint32_t>(all);
  if (!accepts(removed)) {
    for (size_t i = 0; i < candidates.size(); ++i) {
      removed = static_cast<uint32_t>(all) ^ (uint32_t{1} << i);
      if (accepts(removed)) break;
      removed = 0;
    }
  }
  if (!removed) {
    for (size_t i = candidates.size(); i-- > 0;) {
      const auto trial = removed | (uint32_t{1} << i);
      if (accepts(trial)) removed = trial;
    }
  }
  if (!removed) return;

  size_t removed_pairs = 0;
  size_t removed_barriers = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if ((removed & (uint32_t{1} << i)) == 0) continue;
    if (candidates[i].size() == 1) {
      ++removed_barriers;
    } else {
      ++removed_pairs;
    }
  }
  RemoveHandoffCommands rewrite(bits, removed);
  auto reduced_body = rewrite.VisitStmt(body);
  if (rewrite.RemovedSites() != 2 * removed_pairs + removed_barriers ||
      removed_barriers > result.barrier_sites) {
    return;
  }
  auto function = MutableCopy(result.function);
  function->body_ = std::move(reduced_body);
  result.function = std::move(function);
  result.set_sites -= removed_pairs;
  result.wait_sites -= removed_pairs;
  result.barrier_sites -= removed_barriers;
}

}  // namespace pypto::ir::sync
