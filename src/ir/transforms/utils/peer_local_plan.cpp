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

#include "src/ir/transforms/utils/peer_local_plan.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/peer_local_effects.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
namespace {

// The numerical analysis owns both the within-region dependence families and
// the finite crossing demands. Keep their original visit ordinals through the
// shared allocator and materializer, including the final visit of each loop.
FunctionPtr PlanLane(const PeerLocalLane& lane, const backend::LocalSyncExecutionContract& execution) {
  const auto& input = lane.input;
  const auto math = AnalyzeCompactSync(input);
  const auto crossings = math.ExportCrossingDemands();
  CHECK(crossings) << "Peer local region has no complete crossing demand export";
  std::vector<PhysicalSyncFamily> families;
  auto append = [&](PhysicalSyncFamily family) {
    const auto [source, target] = SyncFamilyPipes(input, family);
    if (source != PipeType::S || target != source || !execution.scalar_prefix_serialized)
      families.push_back(std::move(family));
  };
  for (size_t r = 0; r < input.segments.size(); ++r) {
    const auto& segment = math.State().segments.at(r);
    for (const auto& d : segment.demands) {
      if (segment.count > d.distance)
        append({r, d.source, r, d.target, 0, d.distance, segment.count - d.distance, {}});
    }
  }
  for (const auto& edge : *crossings) {
    const auto& [r, v, s] = edge.first;
    const auto& [t, w, u] = edge.second;
    append({r, s, t, u, v, w, 1, {}});
  }
  DirectSyncOrder order;
  order.precedes = [&](const Occurrence& a, const Occurrence& b) { return math.Precedes(a, true, b, false); };
  order.reference_before = [&](const Occurrence& a, const Occurrence& b) {
    return math.ReferenceBefore(a, b);
  };
  order.threshold = [&](size_t r, size_t a, size_t b) {
    return math.State().segments.at(r).Threshold(a, true, b, false);
  };
  order.allow_stronger_barriers = true;
  order.finite_minimum = input.segments.size() == 1;
  order.imported_native_claims = &lane.reservations;
  auto realized =
      RealizeDirectSync(input, std::move(families), order, input.hardware_target, std::nullopt, {});
  return realized.function;
}

// Each lane is planned in its original lexical scope. The endpoint has a
// single terminal subblock branch, so the arms can share exactly the prefix
// whose physical commands agree on both paths.
FunctionPtr JoinLanes(const FunctionPtr& then_plan, const FunctionPtr& else_plan) {
  auto left = transform_utils::FlattenToStmts(then_plan->body_);
  auto right = transform_utils::FlattenToStmts(else_plan->body_);
  CHECK(!left.empty() && left.size() == right.size())
      << "Peer local lane schedules disagree before the branch";
  const auto a = As<IfStmt>(left.back()), b = As<IfStmt>(right.back());
  CHECK(a && b && a->else_body_ && b->else_body_ && a->condition_ == b->condition_)
      << "Missing original peer subblock branch";
  for (size_t i = 0; i + 1 < left.size(); ++i) {
    if (left[i] == right[i]) continue;
    // Independently generated SSA definitions cannot be shared across arms.
    const auto lhs = As<EvalStmt>(left[i]), rhs = As<EvalStmt>(right[i]);
    CHECK(lhs && rhs && structural_equal(lhs, rhs)) << "Peer local lane schedules disagree before the branch";
  }
  auto branch = MutableCopy(a);
  branch->else_body_ = b->else_body_;
  left.back() = branch;
  auto result = MutableCopy(then_plan);
  result->body_ = SeqStmts::Flatten(std::move(left), then_plan->span_);
  return result;
}

class DrainReturns : public SyncStatementMutator {
 public:
  DrainReturns(FunctionType type, const char* operation)
      : core_(type == FunctionType::AIC ? "aic" : "aiv"), operation_(operation) {}

 private:
  const std::string core_;
  const char* operation_;
  StmtPtr VisitStmt_(const ReturnStmtPtr& stmt) override {
    const auto drain = OpRegistry::GetInstance().Create(operation_, {}, {{"core_type", core_}}, stmt->span_);
    return std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<EvalStmt>(drain, stmt->span_), stmt}, stmt->span_);
  }
};
}  // namespace

FunctionPtr RealizePeerLocalPlan(const PeerLocalEffects& effects) {
  CHECK(!effects.lanes.empty() && effects.lanes.size() <= 2)
      << "Peer local physical schedule lacks a qualified lane";
  const auto& input = effects.lanes.front().input;
  CHECK(input.owner) << "Peer local physical schedule lacks an endpoint owner";
  const auto& execution = backend::RequireLocalSyncExecutionContract(input.hardware_target);
  CHECK(execution.set_completes_source_prefix && execution.wait_consumes_before_advance &&
        execution.terminal_completion_op)
      << "Unavailable peer local event execution contract";
  auto plan = PlanLane(effects.lanes.front(), execution);
  if (effects.lanes.size() == 2) {
    const auto& other = effects.lanes.back();
    INTERNAL_CHECK(other.input.owner == input.owner && other.input.hardware_target == input.hardware_target);
    plan = JoinLanes(plan, PlanLane(other, execution));
  }
  auto result = MutableCopy(plan);
  result->body_ = DrainReturns(plan->func_type_, execution.terminal_completion_op).VisitStmt(plan->body_);
  CheckPhysicalSyncResult(result);
  return result;
}
}  // namespace pypto::ir::sync
