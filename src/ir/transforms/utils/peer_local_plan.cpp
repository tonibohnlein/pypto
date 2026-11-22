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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/local_completion_relay.h"
#include "src/ir/transforms/utils/peer_local_effects.h"
#include "src/ir/transforms/utils/peer_local_work_loop.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
namespace {

struct PeerPhase {
  size_t first;
  size_t region;
  CompactSyncResult math;
  std::set<SyncDirection> events;
  PeerWorkPairs seam;

  Occurrence Local(Occurrence occurrence) const {
    std::get<0>(occurrence) -= first;
    return occurrence;
  }
};

SyncDirection SeamPair(PipeType source, PipeType target) {
  return {std::min(source, target), std::max(source, target)};
}

// Finite regions still use the ordinary rotating/static demand provider.
// A runtime item is one invocation of that finite model. Keep original visit
// ordinals and call cuts, offsetting only the phase's segment indices.
PeerPhase AnalyzePeerPhase(const CompactSyncInput& input, PeerWorkRange range, size_t region,
                           bool scalar_serial, std::vector<PhysicalSyncFamily>& families) {
  CompactSyncInput finite;
  finite.owner = input.owner;
  finite.hardware_target = input.hardware_target;
  finite.storage = input.storage;
  finite.segments.assign(input.segments.begin() + range.first, input.segments.begin() + range.second);
  PeerPhase phase{range.first, region, AnalyzeCompactSync(std::move(finite)), {}, {}};
  const auto append = [&](PhysicalSyncFamily family) {
    const auto [source, target] = SyncFamilyPipes(input, family);
    if (source != PipeType::S || target != source || !scalar_serial) {
      families.push_back(std::move(family));
    }
    if (source != target) {
      phase.events.emplace(source, target);
    }
  };
  const auto& segments = phase.math.State().segments;
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& segment = segments[i];
    const auto r = i + range.first;
    for (const auto& demand : segment.demands) {
      if (segment.count > demand.distance) {
        append({r, demand.source, r, demand.target, 0, demand.distance, segment.count - demand.distance, {}});
      }
    }
  }
  const auto crossings = phase.math.ExportCrossingDemands();
  CHECK(crossings) << "Peer local region has no complete crossing demand export";
  for (const auto& edge : *crossings) {
    const auto& [r, v, s] = edge.first;
    const auto& [t, w, u] = edge.second;
    append({r + range.first, s, t + range.first, u, v, w, 1, {}});
  }
  return phase;
}

PeerWorkSeamIds ReserveWorkSeams(const PeerLocalLane& lane, const std::vector<PeerPhase>& phases,
                                 SyncNativeReservations& reserved) {
  PeerWorkSeamIds ids;
  const auto reserve = [&](SyncDirection direction) {
    if (ids.count(direction)) {
      return;
    }
    const auto eligible =
        SyncEligibleIds(lane.input, lane.input.hardware_target, direction, std::nullopt, lane.reservations);
    CHECK(!eligible.empty()) << "Runtime work-item seam has no eligible directional event";
    const auto id = eligible.back();
    ids.emplace(direction, id);
    reserved[direction].insert(id);
  };
  for (const auto& phase : phases) {
    for (const auto& [source, target] : phase.seam) {
      if (source != target) {
        reserve({source, target});
        reserve({target, source});
      }
    }
  }
  return ids;
}

void AnalyzePeerWorkSeams(const CompactSyncInput& input, const PeerLocalRegion& region, PeerPhase& before,
                          PeerPhase& body, const PeerPhase& after) {
  const auto& work = *region.work_loop;
  const PeerWorkRange prefix{region.first, work.first}, item{work.first, work.end},
      suffix{work.end, region.end};
  AddPeerWorkConflicts(input, work, prefix, item, before.seam);
  // A zero-trip path goes directly from prefix to suffix. Even on a nonempty
  // path a disjoint item may leave a prefix hazard live until that suffix.
  AddPeerWorkConflicts(input, work, prefix, suffix, before.seam);
  AddPeerWorkConflicts(input, work, item, item, body.seam);
  AddPeerWorkConflicts(input, work, item, suffix, body.seam);
  for (const auto& direction : before.events) {
    if (body.events.count(direction) || after.events.count(direction)) {
      before.seam.insert(SeamPair(direction.first, direction.second));
    }
  }
  // Within-item allocations are proved in the finite model. Its last WAIT
  // must also precede the first SET of the next item on that event bank.
  for (const auto& [source, target] : body.events) {
    body.seam.insert(SeamPair(source, target));
  }
}

FunctionPtr PlanLane(const PeerLocalLane& lane, const backend::LocalSyncExecutionContract& execution) {
  const auto& input = lane.input;
  std::vector<PhysicalSyncFamily> families;
  std::vector<PeerPhase> phases;
  std::vector<size_t> phase_by_segment(input.segments.size());
  std::vector<PeerWorkPlacement> placements;
  const auto add_phase = [&](PeerWorkRange range, size_t region) {
    std::fill(phase_by_segment.begin() + range.first, phase_by_segment.begin() + range.second, phases.size());
    phases.push_back(AnalyzePeerPhase(input, range, region, execution.scalar_prefix_serialized, families));
  };
  for (size_t i = 0; i < lane.regions.size(); ++i) {
    const auto& region = lane.regions[i];
    if (!region.work_loop) {
      add_phase({region.first, region.end}, i);
      continue;
    }
    const auto& work = *region.work_loop;
    const auto first = phases.size();
    add_phase({region.first, work.first}, i);
    add_phase({work.first, work.end}, i);
    add_phase({work.end, region.end}, i);
    AnalyzePeerWorkSeams(input, region, phases[first], phases[first + 1], phases[first + 2]);
    placements.push_back(
        {&work, ClosePeerWorkPairs(phases[first].seam), ClosePeerWorkPairs(phases[first + 1].seam)});
  }
  DirectSyncOrder order;
  order.precedes = [&](const Occurrence& a, const Occurrence& b) {
    const auto ar = std::get<0>(a), br = std::get<0>(b);
    const auto i = phase_by_segment[ar], j = phase_by_segment[br];
    const auto& phase = phases[i];
    if (i == j) {
      return phase.math.Precedes(phase.Local(a), true, phase.Local(b), false);
    }
    if (phase.region != phases[j].region) {
      // The authored hard MIX SYNCALL drains both directions. Any earlier
      // event WAIT has completed when a later region can issue its SET.
      return i < j;
    }
    // Reuse queries compare WAIT and SET cuts on the same directional bank.
    // The earlier phase's seam consumes waits on the destination queue
    // before a later phase can publish again on the source queue.
    const auto pair = SeamPair(input.segments[ar].sites[std::get<2>(a)].pipe,
                               input.segments[br].sites[std::get<2>(b)].pipe);
    return i < j && phase.seam.count(pair) != 0;
  };
  order.reference_before = [&](const Occurrence& a, const Occurrence& b) {
    const auto i = phase_by_segment[std::get<0>(a)], j = phase_by_segment[std::get<0>(b)];
    const auto& phase = phases[i];
    return i == j ? phase.math.ReferenceBefore(phase.Local(a), phase.Local(b)) : i < j;
  };
  order.threshold = [&](size_t r, size_t a, size_t b) {
    const auto& phase = phases[phase_by_segment[r]];
    return phase.math.State().segments.at(r - phase.first).Threshold(a, true, b, false);
  };
  order.allow_stronger_barriers = true;
  order.finite_minimum = input.segments.size() == 1;
  order.imported_native_claims = &lane.reservations;
  SyncNativeReservations seam_reservations;
  const auto ids = ReserveWorkSeams(lane, phases, seam_reservations);
  SyncSchedulePolicy policy;
  if (!placements.empty() || lane.regions.size() > 1) {
    // Every call, descriptor and FIFO operation stays at its authored cut.
    policy.mode = "guarded";
    policy.boundary_motion = false;
  }
  auto realized = AllocateDirectSync(input, std::move(families), order, input.hardware_target, std::nullopt,
                                     seam_reservations);
  if (!placements.empty()) {
    std::vector<std::vector<const PhysicalSyncFamily*>> by_region(lane.regions.size());
    for (const auto& family : realized.assignment->Families()) {
      const auto region = phases[phase_by_segment[family.source_region]].region;
      INTERNAL_CHECK(region == phases[phase_by_segment[family.target_region]].region)
          << "A finite work-item event cannot cross a MIX region";
      by_region[region].push_back(&family);
    }
    size_t placement = 0;
    for (size_t region = 0; region < lane.regions.size(); ++region) {
      if (lane.regions[region].work_loop) {
        SelectPeerWorkSeams(input, lane.regions[region], by_region[region], ids,
                            execution.scalar_prefix_serialized, placements[placement++]);
      }
    }
  }
  EmitAssignedSync(input, realized, policy);
  return placements.empty() ? realized.function : AddPeerWorkSeams(realized.function, placements, ids);
}

// The importer has ruled out earlier exits. Visit each path separately so
// an explicit return in one arm does not hide a sibling's fallthrough.
class CompletePeerPath : public SyncStatementMutator {
 public:
  CompletePeerPath(FunctionType type, const char* operation)
      : core_(type == FunctionType::AIC ? "aic" : "aiv"), operation_(operation) {}

  StmtPtr Run(const StmtPtr& body) {
    auto completed = VisitStmt(body);
    if (has_return_) {
      return completed;
    }
    return SeqStmts::Flatten({completed, Drain(body->span_)}, body->span_);
  }

 private:
  const std::string core_;
  const char* operation_;
  bool has_return_ = false;

  StmtPtr Drain(const Span& span) const {
    const auto drain = OpRegistry::GetInstance().Create(operation_, {}, {{"core_type", core_}}, span);
    return std::make_shared<EvalStmt>(drain, span);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& stmt) override {
    has_return_ = true;
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{Drain(stmt->span_), stmt}, stmt->span_);
  }
};

FunctionPtr CompleteTerminalLanes(const FunctionPtr& owner, std::vector<StmtPtr> statements,
                                  const StmtPtr& else_body, const char* completion_op) {
  const auto original = As<IfStmt>(statements.back());
  auto branch = MutableCopy(original);
  branch->then_body_ = CompletePeerPath(owner->func_type_, completion_op).Run(original->then_body_);
  branch->else_body_ = CompletePeerPath(owner->func_type_, completion_op).Run(else_body);
  statements.back() = branch;
  auto result = MutableCopy(owner);
  result->body_ = SeqStmts::Flatten(std::move(statements), owner->span_);
  return result;
}

// Each lane is planned in its original lexical scope. The endpoint has a
// single terminal subblock branch, so the arms can share exactly the prefix
// whose physical commands agree on both paths.
FunctionPtr JoinLanes(const FunctionPtr& then_plan, const FunctionPtr& else_plan, const char* completion_op) {
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
  return CompleteTerminalLanes(then_plan, std::move(left), *b->else_body_, completion_op);
}

FunctionPtr PlanStructuredEndpoint(const PeerLocalEffects& effects) {
  const auto& imported = *effects.structured;
  const auto& target = effects.lanes.front().input.hardware_target;
  SyncNativeReservations reserved;
  for (const auto& lane : effects.lanes) {
    for (const auto& [direction, ids] : lane.reservations) {
      reserved[direction].insert(ids.begin(), ids.end());
    }
  }
  using Plan = std::pair<PhysicalSyncRealization, ConservativeSyncAnalysis>;
  const auto realize = [&](bool precise_captures) -> std::optional<Plan> {
    auto analysis = AnalyzeConservativeSync(
        imported.owner, target, {}, {}, /*retain_incoming_diagnostics=*/false,
        /*automatic_children=*/false, /*enable_refresh_kill=*/false, effects.parameters, std::nullopt,
        /*enable_loop_prefixes=*/false, &imported, precise_captures);
    const auto logical = InsertConservativePrefixes(analysis);
    CheckGeneratedLogicalSync(logical);
    auto physical = TryLowerClosedPrefixSync(logical, target, std::nullopt, reserved, effects.parameters,
                                             std::nullopt, &imported);
    if (!physical) {
      return std::nullopt;
    }
    return Plan{std::move(*physical), std::move(analysis)};
  };
  // Separating record and wait may keep more event slots live. Use the same
  // allocator and its consumption proof, with the original adjacent-prefix
  // schedule available when the directional palette cannot hold both.
  std::optional<Plan> plan;
  try {
    plan = realize(true);
  } catch (const pypto::ValueError&) {
    // The unmodified plan remains available for a supported original endpoint.
  }
  const bool precise = plan.has_value();
  if (!plan) {
    plan = realize(false);
  }
  CHECK(plan) << "No closed-prefix schedule for the imported peer endpoint";
  const auto refine = [&](const FunctionPtr& physical, const std::vector<EarlierPeerPrefix>& moves) {
    const auto moved =
        precise ? MoveEarlierPeerPrefixes(physical, plan->second, imported, target, effects.parameters, moves)
                : physical;
    return RefineLocalCompletionPlan(moved, imported.calls, target, reserved);
  };
  const auto baseline = refine(plan->first.function, {});
  if (const auto release = TryEarlierReaderReleases(plan->first, plan->second, imported, target, reserved,
                                                    effects.parameters)) {
    const auto candidate = refine(release->function, release->moves);
    if (OnlyAdvancesReaderRelease(baseline, candidate)) {
      return candidate;
    }
  }
  return baseline;
}

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
  auto plan =
      effects.structured ? PlanStructuredEndpoint(effects) : PlanLane(effects.lanes.front(), execution);
  if (effects.structured && effects.structured->terminal_branch) {
    auto body = transform_utils::FlattenToStmts(plan->body_);
    const auto branch = body.empty() ? nullptr : As<IfStmt>(body.back());
    INTERNAL_CHECK(branch && branch->else_body_ &&
                   branch->condition_ == effects.structured->terminal_branch->condition_)
        << "Original terminal peer branches must retain their control cut";
    plan =
        CompleteTerminalLanes(plan, std::move(body), *branch->else_body_, execution.terminal_completion_op);
  } else if (effects.lanes.size() == 2) {
    const auto& other = effects.lanes.back();
    INTERNAL_CHECK(other.input.owner == input.owner && other.input.hardware_target == input.hardware_target);
    plan = JoinLanes(plan, PlanLane(other, execution), execution.terminal_completion_op);
  } else {
    auto completed = MutableCopy(plan);
    completed->body_ = CompletePeerPath(plan->func_type_, execution.terminal_completion_op).Run(plan->body_);
    plan = completed;
  }
  std::set<const Var*> work_ivs;
  for (const auto& lane : effects.lanes) {
    for (const auto& region : lane.regions) {
      if (region.work_loop) {
        work_ivs.insert(region.work_loop->original->loop_var_.get());
      }
    }
  }
  plan = SpecializePeerWorkGuards(plan, work_ivs);
  CheckPhysicalSyncResult(plan);
  return plan;
}
}  // namespace pypto::ir::sync
