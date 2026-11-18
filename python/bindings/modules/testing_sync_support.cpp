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

#include "python/bindings/modules/testing_sync_support.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "src/ir/transforms/utils/numerical_sync_hierarchy.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"
#include "src/ir/transforms/utils/rotating_sync_analysis.h"
#include "src/ir/transforms/utils/scoped_logical_sync.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_token_names.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::python::sync_testing {
using ir::As;
using ir::EvalStmt;
using ir::Function;
using ir::IfStmt;
using ir::IRVisitor;
using ir::ParamDirection;
using ir::SeqStmts;
using ir::TypePtr;
using ir::Var;
using ir::VarPtr;
using ir::sync::LowerLogicalSync;
using ir::sync::OptimizeLogicalSync;
using ir::sync::RealizeCompactSync;
using ir::sync::RealizeLogicalInvocation;
using ir::sync::SyncTokenNames;
using ir::sync::physical::AxisAP;
using ir::sync::physical::AxisConstraint;
using ir::sync::physical::AxisId;
using ir::sync::physical::CaseCommands;
using ir::sync::physical::ControlBranch;
using ir::sync::physical::ControlCommand;
using ir::sync::physical::ControlForest;
using ir::sync::physical::ControlPtr;
using ir::sync::physical::ControlSequence;
using ir::sync::physical::GuardDomainArena;
using ir::sync::physical::IntersectGuards;
using ir::sync::physical::Intervals;
using ir::sync::physical::MaterializeFrozenCommands;
using ir::sync::physical::RadixBox;
using ir::sync::physical::RadixBoxGuard;
using ir::sync::physical::ScheduleCommandCases;
using ir::sync::physical::ShareCommandGuards;
using ir::sync::physical::SubtractGuards;
using ir::sync::physical::UnionGuards;
PhysicalSyncRealization RealizeDemandPlan(const CompactSyncResult& analysis, const std::string& target,
                                          const std::optional<std::vector<int>>& eligible_ids,
                                          const SyncSchedulePolicy& policy) {
  CHECK(analysis.State().input.hardware_target == target)
      << "Synchronization hardware contract differs from lowering target";
  auto logical = RealizeCompactSync(analysis);
  auto lowered =
      LowerLogicalSync(OptimizeLogicalSync(logical.function), target, eligible_ids,
                       analysis.State().input.parameter_bounds, analysis.State().input.block_bounds, policy);
  CHECK(lowered.physical_cases.size() == 1)
      << "Single-region test realization needs one physical case; use the invocation adapter";
  auto result = std::move(lowered.physical_cases.front());
  result.function = std::move(lowered.function);
  return result;
}
DemandInvocation RealizeDemandInvocation(const SyncInvocationAnalysis& analysis, const std::string& target,
                                         const std::optional<std::vector<int>>& eligible_ids,
                                         const SyncSchedulePolicy& policy) {
  for (const auto& item : analysis.cases)
    CHECK(item.mathematics && item.mathematics->State().input.hardware_target == target)
        << "Synchronization hardware contract differs from lowering target";
  auto logical = RealizeLogicalInvocation(analysis);
  auto lowered =
      LowerLogicalSync(OptimizeLogicalSync(logical.function), target, eligible_ids, analysis.parameter_bounds,
                       analysis.block_bounds, policy, ir::sync::SyncExternalPolicy::Geometry,
                       analysis.coordinate ? std::make_optional(analysis.coordinate->Count()) : std::nullopt);
  return {std::move(lowered.function), std::move(lowered.physical_cases)};
}

namespace {
void CollectOwnerLoopAxes(const FunctionPtr& owner, const std::shared_ptr<GuardDomainArena>& arena,
                          std::map<const Var*, AxisId>& loop_vars) {
  std::set<const Var*> duplicate_vars;
  class Collector : public ir::IRVisitor {
    const std::shared_ptr<GuardDomainArena>& arena_;
    std::map<const Var*, AxisId>& loop_vars_;
    std::set<const Var*>& duplicate_vars_;

   public:
    Collector(const std::shared_ptr<GuardDomainArena>& arena, std::map<const Var*, AxisId>& loop_vars,
              std::set<const Var*>& duplicate_vars)
        : arena_(arena), loop_vars_(loop_vars), duplicate_vars_(duplicate_vars) {}
    void VisitStmt_(const ir::ForStmtPtr& loop) override {
      if (loop->kind_ == ir::ForKind::Sequential && loop->iter_args_.empty() && loop->return_vars_.empty() &&
          loop->attrs_.empty() && loop->hardware_loop_policy_ == ir::HardwareLoopPolicy::Auto) {
        const auto type = As<ir::ScalarType>(loop->loop_var_->GetType());
        const auto start = As<ir::ConstInt>(loop->start_), stop = As<ir::ConstInt>(loop->stop_),
                   step = As<ir::ConstInt>(loop->step_);
        if (type && type->dtype_ == DataType::INDEX && start && stop && step && step->value_ > 0 &&
            start->value_ < stop->value_) {
          const __int128 wide_count =
              (__int128(stop->value_) - start->value_ + step->value_ - 1) / step->value_;
          if (wide_count >= 1 && wide_count <= std::numeric_limits<int64_t>::max()) {
            const __int128 last = __int128(start->value_) + (wide_count - 1) * step->value_;
            if (last + step->value_ <= std::numeric_limits<int64_t>::max()) {
              const AxisId ax = arena_->RegisterLoopFrame(ir::sync::SyncLoopFrame{
                  loop, static_cast<int64_t>(wide_count), start->value_, step->value_, nullptr});
              if (duplicate_vars_.count(loop->loop_var_.get()) == 0) {
                if (!loop_vars_.emplace(loop->loop_var_.get(), ax).second) {
                  loop_vars_.erase(loop->loop_var_.get());
                  duplicate_vars_.insert(loop->loop_var_.get());
                }
              }
            }
          }
        }
      }
      IRVisitor::VisitStmt_(loop);
    }
  };
  Collector(arena, loop_vars, duplicate_vars).VisitStmt(owner->body_);
}

std::optional<AxisId> MatchNormalizedDigit(const ir::ExprPtr& expr, const GuardDomainArena& arena,
                                           const std::map<const Var*, AxisId>& loop_vars) {
  if (const auto var = As<ir::Var>(expr)) {
    const auto it = loop_vars.find(var.get());
    if (it == loop_vars.end()) return std::nullopt;
    const auto& info = arena.Axis(it->second);
    if (info.start == 0 && info.step == 1) return it->second;
    return std::nullopt;
  }
  if (const auto div = As<ir::FloorDiv>(expr)) {
    const auto step_c = As<ir::ConstInt>(div->right_);
    if (!step_c || step_c->value_ <= 0) return std::nullopt;
    if (const auto sub = As<ir::Sub>(div->left_)) {
      const auto var = As<ir::Var>(sub->left_);
      const auto start_c = As<ir::ConstInt>(sub->right_);
      if (!var || !start_c) return std::nullopt;
      const auto it = loop_vars.find(var.get());
      if (it == loop_vars.end()) return std::nullopt;
      const auto& info = arena.Axis(it->second);
      if (info.start == start_c->value_ && info.step == step_c->value_) return it->second;
    } else if (const auto var = As<ir::Var>(div->left_)) {
      const auto it = loop_vars.find(var.get());
      if (it == loop_vars.end()) return std::nullopt;
      const auto& info = arena.Axis(it->second);
      if (info.start == 0 && info.step == step_c->value_) return it->second;
    }
    return std::nullopt;
  }
  if (const auto sub = As<ir::Sub>(expr)) {
    const auto var = As<ir::Var>(sub->left_);
    const auto start_c = As<ir::ConstInt>(sub->right_);
    if (!var || !start_c) return std::nullopt;
    const auto it = loop_vars.find(var.get());
    if (it == loop_vars.end()) return std::nullopt;
    const auto& info = arena.Axis(it->second);
    if (info.start == start_c->value_ && info.step == 1) return it->second;
  }
  return std::nullopt;
}

std::optional<RadixBoxGuard> TryLiftExprToRadixGuard(const ExprPtr& expr,
                                                     const std::shared_ptr<GuardDomainArena>& arena,
                                                     const std::map<const Var*, AxisId>& loop_vars) {
  if (!arena || loop_vars.empty()) return std::nullopt;
  if (const auto b = As<ir::ConstBool>(expr)) {
    return b->value_ ? RadixBoxGuard::True() : RadixBoxGuard::False();
  }
  if (const auto n = As<ir::Not>(expr)) {
    const auto inner = TryLiftExprToRadixGuard(n->operand_, arena, loop_vars);
    if (!inner) return std::nullopt;
    return SubtractGuards(RadixBoxGuard::True(), *inner);
  }
  if (const auto a = As<ir::And>(expr)) {
    const auto l = TryLiftExprToRadixGuard(a->left_, arena, loop_vars);
    const auto r = TryLiftExprToRadixGuard(a->right_, arena, loop_vars);
    if (!l || !r) return std::nullopt;
    return IntersectGuards(*l, *r);
  }
  if (const auto o = As<ir::Or>(expr)) {
    const auto l = TryLiftExprToRadixGuard(o->left_, arena, loop_vars);
    const auto r = TryLiftExprToRadixGuard(o->right_, arena, loop_vars);
    if (!l || !r) return std::nullopt;
    return UnionGuards(*l, *r);
  }
  const auto binary = As<ir::BinaryExpr>(expr);
  if (!binary) return std::nullopt;
  if (const auto eq = As<ir::Eq>(expr)) {
    const auto mod = As<ir::FloorMod>(eq->left_);
    const auto rem = As<ir::ConstInt>(eq->right_);
    if (mod && rem) {
      const auto divisor = As<ir::ConstInt>(mod->right_);
      const auto ax =
          divisor && divisor->value_ > 0 ? MatchNormalizedDigit(mod->left_, *arena, loop_vars) : std::nullopt;
      if (ax) {
        const int64_t M = divisor->value_;
        const int64_t r = rem->value_;
        if (r < 0 || r >= M) return RadixBoxGuard::False();
        const int64_t cnt = arena->Axis(*ax).count;
        const auto ap = AxisAP::MakeCanonical(r, cnt, M, cnt);
        if (!ap) return RadixBoxGuard::False();
        RadixBox box;
        if (!ap->IsFull(cnt)) box.constraints.push_back(AxisConstraint{*ax, *ap});
        return RadixBoxGuard::FromBox(arena, std::move(box));
      }
    }
  }
  auto var = As<ir::Var>(binary->left_);
  auto cst = As<ir::ConstInt>(binary->right_);
  bool flipped = false;
  if (!var || !cst) {
    var = As<ir::Var>(binary->right_);
    cst = As<ir::ConstInt>(binary->left_);
    flipped = true;
  }
  if (!var || !cst) return std::nullopt;
  const auto it = loop_vars.find(var.get());
  if (it == loop_vars.end()) return std::nullopt;
  const AxisId ax = it->second;
  const auto& info = arena->Axis(ax);
  const int64_t start = info.start, step = info.step, cnt = info.count;
  const __int128 v = cst->value_;
  auto make_box = [&](std::optional<AxisAP> ap) -> RadixBoxGuard {
    if (!ap) return RadixBoxGuard::False();
    RadixBox box;
    if (!ap->IsFull(cnt)) box.constraints.push_back(AxisConstraint{ax, *ap});
    return RadixBoxGuard::FromBox(arena, std::move(box));
  };
  const bool is_eq = As<ir::Eq>(expr) != nullptr;
  const bool is_ne = As<ir::Ne>(expr) != nullptr;
  const bool is_lt = (!flipped && As<ir::Lt>(expr)) || (flipped && As<ir::Gt>(expr));
  const bool is_le = (!flipped && As<ir::Le>(expr)) || (flipped && As<ir::Ge>(expr));
  const bool is_gt = (!flipped && As<ir::Gt>(expr)) || (flipped && As<ir::Lt>(expr));
  const bool is_ge = (!flipped && As<ir::Ge>(expr)) || (flipped && As<ir::Le>(expr));
  if (!is_eq && !is_ne && !is_lt && !is_le && !is_gt && !is_ge) return std::nullopt;

  if (is_eq || is_ne) {
    const __int128 diff = v - start;
    if (diff < 0 || diff % step != 0) return is_eq ? RadixBoxGuard::False() : RadixBoxGuard::True();
    const __int128 d = diff / step;
    if (d >= cnt) return is_eq ? RadixBoxGuard::False() : RadixBoxGuard::True();
    const auto eq_box = make_box(AxisAP{static_cast<int64_t>(d), static_cast<int64_t>(d) + 1, 1});
    if (is_eq) return eq_box;
    return SubtractGuards(RadixBoxGuard::True(), eq_box);
  }
  if (is_lt) {
    if (v <= start) return RadixBoxGuard::False();
    const __int128 d_end = (v - start - 1) / step + 1;
    const int64_t hi = d_end > cnt ? cnt : static_cast<int64_t>(d_end);
    return make_box(AxisAP::MakeCanonical(0, hi, 1, cnt));
  }
  if (is_le) {
    if (v < start) return RadixBoxGuard::False();
    const __int128 d_end = (v - start) / step + 1;
    const int64_t hi = d_end > cnt ? cnt : static_cast<int64_t>(d_end);
    return make_box(AxisAP::MakeCanonical(0, hi, 1, cnt));
  }
  if (is_gt) {
    if (v < start) return RadixBoxGuard::True();
    const __int128 d_first = (v - start) / step + 1;
    if (d_first >= cnt) return RadixBoxGuard::False();
    return make_box(AxisAP::MakeCanonical(static_cast<int64_t>(d_first), cnt, 1, cnt));
  }
  // is_ge
  if (v <= start) return RadixBoxGuard::True();
  const __int128 d_first = (v - start + step - 1) / step;
  if (d_first >= cnt) return RadixBoxGuard::False();
  return make_box(AxisAP::MakeCanonical(static_cast<int64_t>(d_first), cnt, 1, cnt));
}

ControlPtr ImportCommandsImpl(const StmtPtr& stmt, PhysicalSyncRealization& work,
                              const std::shared_ptr<GuardDomainArena>& arena,
                              const std::map<const Var*, AxisId>& loop_vars) {
  if (auto sequence = As<SeqStmts>(stmt)) {
    ControlForest children;
    for (const auto& child : sequence->stmts_)
      children.push_back(ImportCommandsImpl(child, work, arena, loop_vars));
    return ControlSequence(std::move(children), stmt->span_, work);
  }
  if (auto branch = As<IfStmt>(stmt)) {
    CHECK(branch->return_vars_.empty()) << "Command schedule cannot move definitions";
    auto then_ctrl = ImportCommandsImpl(branch->then_body_, work, arena, loop_vars);
    auto else_ctrl =
        branch->else_body_
            ? std::optional<ControlPtr>(ImportCommandsImpl(*branch->else_body_, work, arena, loop_vars))
            : std::nullopt;
    // Validate predicate syntax & qualification first.
    auto opaque = ControlBranch(branch->condition_, then_ctrl, else_ctrl, stmt->span_, work);
    if (opaque->predicate_qualified) {
      if (const auto lifted = TryLiftExprToRadixGuard(branch->condition_, arena, loop_vars)) {
        return ControlBranch(*lifted, then_ctrl, else_ctrl, stmt->span_, work);
      }
    }
    return opaque;
  }
  auto command = As<EvalStmt>(stmt);
  CHECK(command) << "Command schedule requires command-only input";
  return ControlCommand(command->expr_, stmt->span_, work);
}
}  // namespace

ControlPtr ImportCommands(const StmtPtr& stmt, PhysicalSyncRealization& work) {
  return ImportCommandsImpl(stmt, work, nullptr, {});
}

namespace {
// Seed names from all supplied fragments. This owner is only a lexical naming
// scope, never evidence of readiness, matching, physical reuse or payload order.
FunctionPtr CommandOwner(const std::vector<StmtPtr>& commands) {
  return std::make_shared<Function>("test_commands", std::vector<VarPtr>{}, std::vector<ParamDirection>{},
                                    std::vector<TypePtr>{},
                                    std::make_shared<SeqStmts>(commands, Span::unknown()), Span::unknown());
}
}  // namespace
StmtPtr MaterializeValues(const FunctionPtr& owner, const StmtPtr& commands) {
  SyncTokenNames names(owner);
  PhysicalSyncRealization work;
  return std::make_shared<SeqStmts>(MaterializeFrozenCommands({ImportCommands(commands, work)}, names, work),
                                    commands->span_);
}
std::vector<StmtPtr> ShareGuards(const std::vector<StmtPtr>& commands, PhysicalSyncRealization& work) {
  ControlForest forest;
  for (const auto& command : commands) forest.push_back(ImportCommands(command, work));
  SyncTokenNames names(CommandOwner(commands));
  return MaterializeFrozenCommands(ShareCommandGuards(forest, work), names, work);
}
std::vector<StmtPtr> EmitCases(const ExprPtr& block, const Intervals& possible,
                               const std::vector<CommandCase>& cases, PhysicalSyncRealization& work,
                               const Span& span) {
  std::vector<CaseCommands> descriptors;
  std::vector<StmtPtr> bodies;
  for (const auto& item : cases) {
    descriptors.push_back({item.blocks, ImportCommands(item.body, work)});
    bodies.push_back(item.body);
  }
  // Include the dispatch coordinate when reserving generated names.
  bodies.push_back(std::make_shared<EvalStmt>(block, span));
  SyncTokenNames names(CommandOwner(bodies));
  return MaterializeFrozenCommands(ScheduleCommandCases(block, possible, descriptors, work, span, {}), names,
                                   work);
}
std::pair<std::vector<Intervals>, PhysicalSyncRealization> InspectGroups(
    const FunctionPtr& owner, const std::vector<CommandCase>& supplied) {
  CHECK(!supplied.empty()) << "Expected nonempty schedule cases";
  PhysicalSyncRealization work;
  std::vector<std::pair<int64_t, int64_t>> domains;
  auto cut = std::make_shared<physical::PhysicalCuts>();
  const auto blocks = std::make_pair(supplied.front().blocks.first, supplied.back().blocks.second);
  int64_t next = blocks.first;
  for (const auto& item : supplied) {
    CHECK(item.blocks.first == next && item.blocks.second >= next &&
          item.blocks.second < std::numeric_limits<int64_t>::max())
        << "Non-exhaustive cases";
    next = item.blocks.second + 1;
    domains.push_back(item.blocks);
    cut->before_cases.push_back({item.blocks, ImportCommands(item.body, work)});
  }
  const std::vector<physical::PreparedSchedulePhase> phases{
      {0, 1, 1, false, {physical::CutInventory{{nullptr, cut}}}}};
  const auto grouped =
      physical::GroupInvocationSchedules(phases, ir::sync::DominatingSyncBlock(owner), domains, work);
  std::vector<Intervals> result;
  for (const auto& group : grouped) result.push_back(group.intervals);
  return {std::move(result), std::move(work)};
}
PhysicalSyncRealization EmitBoundaries(const FunctionPtr& owner,
                                       const std::vector<std::tuple<CallPtr, StmtPtr, StmtPtr>>& supplied,
                                       bool enabled, bool validate_scope) {
  PhysicalSyncRealization work;
  auto arena = std::make_shared<GuardDomainArena>();
  std::map<const Var*, AxisId> loop_vars;
  CollectOwnerLoopAxes(owner, arena, loop_vars);
  physical::CutInventory inventory;
  for (const auto& [call, before, after] : supplied) {
    auto cut = std::make_shared<physical::PhysicalCuts>();
    cut->before = {ImportCommandsImpl(before, work, arena, loop_vars)};
    cut->after = {ImportCommandsImpl(after, work, arena, loop_vars)};
    CHECK(inventory.emplace(call.get(), std::move(cut)).second) << "Duplicate diagnostic command cut";
  }
  if (validate_scope) {
    physical::PopulateCutDomains(owner, inventory, work);
    for (const auto& [call, cut] : inventory) {
      CHECK(cut->predicate_scope) << "Diagnostic cut has no original owner scope";
      physical::RequireCommandScope(cut->before, *cut->predicate_scope, work);
      physical::RequireCommandScope(cut->after, *cut->predicate_scope, work);
    }
  }
  SyncTokenNames names(owner);
  ++work.schedule_name_scans;
  work.schedule_name_visits += names.visits;
  physical::PreparedLoops loops;
  auto plans = physical::PlanCommandBoundaries(owner, inventory, enabled, work, arena);
  work.function =
      physical::MaterializeSelectedSchedule(owner, nullptr, inventory, loops, plans, {}, work, names);
  return work;
}
namespace {
using ir::ForStmt;
using ir::ForStmtPtr;
using ir::PipeType;
using ir::sync::CompactSyncView;
using ir::sync::Crossing;
using ir::sync::DecodeNumericalOccurrence;
using ir::sync::EvaluateSyncCoordinate;
using ir::sync::kPipeCount;
using ir::sync::MergeNumericalBoundary;
using ir::sync::NumericalAdd;
using ir::sync::NumericalBoundary;
using ir::sync::NumericalCell;
using ir::sync::NumericalNode;
using ir::sync::NumericalOrdinal;
using ir::sync::NumericalReferenceBefore;
using ir::sync::Occurrence;
using ir::sync::PointBoundary;
using ir::sync::RotatingBoundary;
using ir::sync::SyncAccess;
using ir::sync::SyncCoordinateNode;
using ir::sync::SyncSegment;
using ir::sync::SyncTranslation;

NumericalOrdinal Length(int64_t count, size_t sites) { return NumericalOrdinal(count) * sites; }

std::optional<std::pair<int64_t, int64_t>> Visits(const SyncAccess& a, int64_t count, int64_t row,
                                                  int64_t col) {
  if (count <= 0) return std::nullopt;
  int64_t lo = a.after_first ? 1 : 0, hi = count - 1;
  auto restrict = [&](int64_t x, int64_t start, int64_t step, int64_t size) {
    if (step == 0) {
      if (x < start || x - start >= size) hi = -1;
      return;
    }
    if (x < start) {
      hi = -1;
      return;
    }
    hi = std::min(hi, (x - start) / step);
    if (x - start >= size) lo = std::max(lo, (x - start - size) / step + 1);
  };
  restrict(row, a.row, a.row_step, a.rows);
  restrict(col, a.column, a.column_step, a.columns);
  if (lo > hi) return std::nullopt;
  return std::make_pair(lo, hi);
}

std::optional<std::pair<int64_t, int64_t>> PointVisits(const SyncAccess& access, const SyncSegment& segment,
                                                       int64_t row, int64_t column) {
  if (!segment.count) return std::nullopt;
  if (access.coordinate_frame && !segment.nested_frames.empty()) {
    int64_t stride = 1;
    for (auto frame = segment.nested_frames.rbegin(); frame != segment.nested_frames.rend(); ++frame) {
      if (frame->loop == access.coordinate_frame) {
        auto visits = Visits(access, frame->count, row, column);
        if (!visits) return std::nullopt;
        return std::make_pair(visits->first * stride,
                              segment.count - 1 - (frame->count - 1 - visits->second) * stride);
      }
      INTERNAL_CHECK(!__builtin_mul_overflow(stride, frame->count, &stride))
          << "Internal error: access coordinate stride overflow";
    }
    INTERNAL_CHECK(false) << "Internal error: missing original access frame";
  }
  const auto scope = segment.reset_period ? segment.reset_period : segment.count;
  auto visits = Visits(access, scope, row, column);
  if (visits) visits->second += segment.count - scope;
  return visits;
}

std::optional<std::pair<int64_t, int64_t>> OriginalAccessPointVisits(const SyncAccess& access,
                                                                     const SyncSegment& segment, int64_t row,
                                                                     int64_t column) {
  CHECK(!access.external) << "External point selector requires its qualified coordinate provider";
  return PointVisits(access, segment, row, column);
}

std::optional<PointBoundary> RelativeBoundary(const CompactSyncView& view, size_t storage, int64_t row,
                                              int64_t column, std::optional<size_t> leaf = std::nullopt) {
  if (storage < view.State().input.storage.size() &&
      (!view.State().input.storage[storage].boundary_available ||
       view.State().input.unavailable_storage_spaces.count(view.State().input.storage[storage].space)))
    return std::nullopt;
  if (storage >= view.State().input.storage.size() || row < 0 || column < 0 ||
      row >= view.State().input.storage[storage].rows ||
      column >= view.State().input.storage[storage].columns)
    return std::nullopt;
  PointBoundary out;
  struct Range {
    size_t r, s;
    int64_t lo, hi;
    bool write;
    PipeType pipe;
  };
  std::vector<Range> ranges;
  for (size_t r = leaf.value_or(0); r < (leaf ? *leaf + 1 : view.State().input.segments.size()); ++r)
    for (size_t s = 0; s < view.State().input.segments[r].sites.size(); ++s) {
      const auto& site = view.State().input.segments[r].sites[s];
      for (const auto& a : site.accesses)
        if (a.storage == storage) {
          if (a.external) return std::nullopt;
          auto visits = [&](const SyncAccess& access) {
            return leaf ? Visits(access, view.State().segments[r].internal_scope_count, row, column)
                        : PointVisits(access, view.State().input.segments[r], row, column);
          };
          auto v = visits(a);
          if (!v) continue;
          bool rmw =
              !a.write && std::any_of(site.accesses.begin(), site.accesses.end(), [&](const SyncAccess& w) {
                return w.write && w.storage == storage && visits(w).has_value();
              });
          if (rmw) continue;
          ranges.push_back({r, s, v->first, v->second, a.write, site.pipe});
          if (a.write) {
            Occurrence first{r, v->first, s}, last{r, v->second, s};
            if (!out.first_writer || first < *out.first_writer) out.first_writer = first;
            if (!out.last_writer || last > *out.last_writer) out.last_writer = last;
          }
        }
    }
  for (const auto& r : ranges)
    if (!r.write) {
      Occurrence first{r.r, r.lo, r.s}, last{r.r, r.hi, r.s};
      if ((!out.first_writer || first < *out.first_writer) &&
          (!out.first_readers[r.pipe] || first < *out.first_readers[r.pipe]))
        out.first_readers[r.pipe] = first;
      if ((!out.last_writer || last > *out.last_writer) &&
          (!out.last_readers[r.pipe] || last > *out.last_readers[r.pipe]))
        out.last_readers[r.pipe] = last;
    }
  return out;
}

PointBoundary LeafPointBoundary(const CompactSyncView& view, size_t region, size_t storage, int64_t row,
                                int64_t column) {
  CHECK(region < view.State().segments.size()) << "Invalid compact leaf";
  CHECK(storage < view.State().input.storage.size() &&
        view.State().input.storage[storage].boundary_available &&
        !view.State().input.unavailable_storage_spaces.count(view.State().input.storage[storage].space))
      << "Leaf storage selector unavailable for this storage domain";
  for (const auto& site : view.State().input.segments[region].sites)
    for (const auto& access : site.accesses)
      CHECK(access.storage != storage ||
            (!access.external &&
             (!access.coordinate_frame ||
              access.coordinate_frame == view.State().input.segments[region].original_loop)))
          << "Leaf point selector requires its enclosing occurrence binding";
  auto result = RelativeBoundary(view, storage, row, column, region);
  CHECK(result) << "Leaf storage point outside the supplied domain";
  return *result;
}

NumericalBoundary LeafBoundary(const CompactSyncView& result, size_t leaf, const NumericalCell& cell) {
  if (result.State().hierarchy) {
    const auto provider = result.State().hierarchy->finite_providers.find(leaf);
    if (provider != result.State().hierarchy->finite_providers.end()) {
      NumericalBoundary out;
      for (size_t i = 0; i < provider->second.sites.size(); ++i) {
        const auto& site = provider->second.sites[i];
        bool reads = false, writes = false;
        for (const auto& access : site.accesses) {
          if (access.storage != cell.storage || access.slot_first != cell.slot || cell.row < access.row ||
              cell.row >= access.row + access.rows || cell.column < access.column ||
              cell.column >= access.column + access.columns)
            continue;
          (access.write ? writes : reads) = true;
        }
        if (writes) {
          if (!out.first_writer) out.first_writer = i;
          out.last_writer = i;
          out.last_readers.fill(std::nullopt);
        } else if (reads) {
          if (!out.first_writer && !out.first_readers[site.pipe]) out.first_readers[site.pipe] = i;
          out.last_readers[site.pipe] = i;
        }
      }
      return out;
    }
  }
  const auto selected = result.State().segments[leaf].rotating
                            ? RotatingBoundary(result.State().input, result.State().segments[leaf],
                                               cell.storage, cell.slot, cell.row, cell.column, leaf)
                            : LeafPointBoundary(result, leaf, cell.storage, cell.row, cell.column);
  const size_t sites = result.State().input.segments[leaf].sites.size();
  auto ordinal = [&](const std::optional<Occurrence>& occurrence) -> std::optional<NumericalOrdinal> {
    if (!occurrence) return std::nullopt;
    return Length(std::get<1>(*occurrence), sites) + std::get<2>(*occurrence);
  };
  NumericalBoundary out;
  out.first_writer = ordinal(selected.first_writer);
  out.last_writer = ordinal(selected.last_writer);
  for (size_t p = 0; p < kPipeCount; ++p) {
    out.first_readers[p] = ordinal(selected.first_readers[p]);
    out.last_readers[p] = ordinal(selected.last_readers[p]);
  }
  return out;
}

NumericalBoundary Point(const CompactSyncView& result, const std::shared_ptr<const NumericalNode>& node,
                        const NumericalCell& cell,
                        std::unordered_map<const NumericalNode*, NumericalBoundary>& cache) {
  const auto found = cache.find(node.get());
  if (found != cache.end()) return found->second;
  NumericalBoundary out;
  if (node->length) {
    if (!node->left)
      out = LeafBoundary(result, node->leaf, cell);
    else
      out = MergeNumericalBoundary(Point(result, node->left, cell, cache),
                                   Point(result, node->right, cell, cache), node->left->length);
  }
  cache.emplace(node.get(), out);
  return out;
}

std::optional<PointBoundary> QueryExternalNumericalBoundary(const CompactSyncView& result, size_t storage,
                                                            int64_t slot, int64_t row, int64_t column,
                                                            const std::map<size_t, int64_t>& parameters,
                                                            std::optional<int64_t> block_index) {
  const auto& domain = result.State().input.storage.at(storage);
  if (slot != 0 || row < 0 || column < 0 || row >= domain.rows || column >= domain.columns)
    return std::nullopt;
  if (!result.State().hierarchy->phase_owned_outputs.count(storage)) return std::nullopt;
  for (const auto& [parameter, value] : parameters) {
    const auto bound = result.State().input.parameter_bounds.find(parameter);
    if (bound == result.State().input.parameter_bounds.end() || value < bound->second.first ||
        value > bound->second.second)
      return std::nullopt;
  }
  if (block_index &&
      (!result.State().input.block_bounds || *block_index < result.State().input.block_bounds->first ||
       *block_index > result.State().input.block_bounds->second))
    return std::nullopt;
  const auto& directory = result.State().storage_access_index.at(storage);
  INTERNAL_CHECK(directory.accesses.size() == 1) << "Owned output lost its complete access inventory";
  const auto [r, s, a] = directory.accesses.front();
  const auto& segment = result.State().input.segments[r];
  const auto& access = segment.sites[s].accesses[a];
  const auto& external = *access.external;
  if (!segment.count) return PointBoundary{};
  if (!external.row_major_tiles && !external.swizzled_tiles) {
    INTERNAL_CHECK(segment.count == 1) << "Output ownership missing nontrivial inverse";
    std::map<const ForStmt*, int64_t> binding;
    std::set<const ForStmt*> needed_frames;
    for (const auto& frame : segment.nested_frames) needed_frames.insert(frame.loop.get());
    for (const auto& node : external.program->nodes) {
      if (node.kind != SyncCoordinateNode::Kind::Loop || !needed_frames.count(node.frame.get())) continue;
      auto start = node.dynamic_start ? EvaluateSyncCoordinate(*external.program, node.left, binding,
                                                               parameters, block_index)
                                      : std::optional<int64_t>(node.start);
      if (!start) return std::nullopt;
      binding[node.frame.get()] = *start;
    }
    const auto r0 = EvaluateSyncCoordinate(*external.program, external.row, binding, parameters, block_index);
    const auto c0 =
        EvaluateSyncCoordinate(*external.program, external.column, binding, parameters, block_index);
    if (!r0 || !c0) return std::nullopt;
    PointBoundary out;
    if (row >= *r0 && row - *r0 < access.rows && column >= *c0 && column - *c0 < access.columns)
      out.first_writer = out.last_writer = Occurrence{r, 0, s};
    return out;
  }
  ForStmtPtr tile_frame;
  size_t tile_coord_id = 0;
  std::optional<size_t> frame_start_root;
  int64_t tile = 0;
  if (external.row_major_tiles) {
    const auto& tiles = *external.row_major_tiles;
    const auto tile_row = row / tiles.row_scale, tile_column = column / tiles.column_scale;
    if (tile_row >= tiles.row_tiles || tile_column >= tiles.column_tiles) return PointBoundary{};
    const auto inverse = __int128(tile_row) * tiles.column_tiles + tile_column;
    if (inverse > std::numeric_limits<int64_t>::max()) return std::nullopt;
    tile = static_cast<int64_t>(inverse);
    tile_frame = tiles.frame;
    tile_coord_id = tiles.tile_coordinate;
    frame_start_root = tiles.frame_start_root;
  } else {
    const auto& tiles = *external.swizzled_tiles;
    const auto tile_row = row / tiles.row_scale, tile_column = column / tiles.column_scale;
    if (tile_row >= tiles.row_tiles || tile_column >= tiles.column_tiles) return PointBoundary{};
    __int128 inverse = 0;
    if (tiles.col_major_groups) {
      const auto block = tile_column / tiles.group_tiles;
      const auto local_c = tile_column % tiles.group_tiles;
      const auto n_col_b = std::min(tiles.group_tiles, tiles.column_tiles - block * tiles.group_tiles);
      const auto base_r = (tiles.snake && (block % 2 == 1)) ? (tiles.row_tiles - 1 - tile_row) : tile_row;
      inverse = __int128(block) * tiles.group_size + __int128(base_r) * n_col_b + local_c;
    } else {
      const auto block = tile_row / tiles.group_tiles;
      const auto local_r = tile_row % tiles.group_tiles;
      const auto n_row_b = std::min(tiles.group_tiles, tiles.row_tiles - block * tiles.group_tiles);
      const auto base_c =
          (tiles.snake && (block % 2 == 1)) ? (tiles.column_tiles - 1 - tile_column) : tile_column;
      inverse = __int128(block) * tiles.group_size + __int128(base_c) * n_row_b + local_r;
    }
    if (inverse > std::numeric_limits<int64_t>::max()) return std::nullopt;
    tile = static_cast<int64_t>(inverse);
    tile_frame = tiles.frame;
    tile_coord_id = tiles.tile_coordinate;
    frame_start_root = tiles.frame_start_root;
  }
  const auto& coordinate = external.program->nodes.at(tile_coord_id);
  std::optional<int64_t> start = coordinate.start;
  if (frame_start_root)
    start = EvaluateSyncCoordinate(*external.program, *frame_start_root, {}, parameters, block_index);
  if (!start) return std::nullopt;
  if (tile < *start || (tile - *start) % coordinate.step) return PointBoundary{};
  const auto visit = (tile - *start) / coordinate.step;
  int64_t ordinal = 0;
  for (const auto& frame : segment.nested_frames) {
    if (!frame.count) return PointBoundary{};
    const auto digit = frame.loop == tile_frame ? visit : 0;
    if (digit >= frame.count) return PointBoundary{};
    const auto next = __int128(ordinal) * frame.count + digit;
    if (next > std::numeric_limits<int64_t>::max()) return std::nullopt;
    ordinal = static_cast<int64_t>(next);
  }
  const std::map<const ForStmt*, int64_t> binding{{tile_frame.get(), tile}};
  const auto first_row =
      EvaluateSyncCoordinate(*external.program, external.row, binding, parameters, block_index);
  const auto first_column =
      EvaluateSyncCoordinate(*external.program, external.column, binding, parameters, block_index);
  if (!first_row || !first_column) return std::nullopt;
  INTERNAL_CHECK(row >= *first_row && row - *first_row < access.rows && column >= *first_column &&
                 column - *first_column < access.columns)
      << "Owned output inverse disagrees with its exact footprint";
  PointBoundary out;
  out.first_writer = out.last_writer = Occurrence{r, ordinal, s};
  return out;
}

PointBoundary QueryNumericalBoundary(const CompactSyncView& result, size_t storage, int64_t slot, int64_t row,
                                     int64_t column) {
  if (result.State().hierarchy->phase_readonly_domains.count(storage)) {
    PointBoundary out;
    CHECK(slot == 0) << "External selector has an invalid physical slot";
    for (const auto& [r, s, a] : result.State().storage_access_index.at(storage).accesses) {
      ++result.Diagnostics().point_selector_access_visits;
      const auto& segment = result.State().input.segments[r];
      const auto& site = segment.sites[s];
      const auto& access = site.accesses[a];
      INTERNAL_CHECK(!access.write) << "Read-only phase domain contains a write";
      const auto range = OriginalAccessPointVisits(access, segment, row, column);
      if (!range) continue;
      const Occurrence first{r, range->first, s}, last{r, range->second, s};
      auto& earliest = out.first_readers[site.pipe];
      auto& latest = out.last_readers[site.pipe];
      if (!earliest || NumericalReferenceBefore(result, first, *earliest)) earliest = first;
      if (!latest || NumericalReferenceBefore(result, *latest, last)) latest = last;
    }
    return out;
  }
  std::unordered_map<const NumericalNode*, NumericalBoundary> cache;
  const auto selected = Point(result, result.State().hierarchy->root, {storage, slot, row, column}, cache);
  auto decode = [&](const std::optional<NumericalOrdinal>& ordinal) -> std::optional<Occurrence> {
    return ordinal ? std::optional<Occurrence>(DecodeNumericalOccurrence(result, *ordinal)) : std::nullopt;
  };
  PointBoundary out;
  out.first_writer = decode(selected.first_writer);
  out.last_writer = decode(selected.last_writer);
  for (size_t p = 0; p < kPipeCount; ++p) {
    out.first_readers[p] = decode(selected.first_readers[p]);
    out.last_readers[p] = decode(selected.last_readers[p]);
  }
  return out;
}

void ExpandNodeDemands(const CompactSyncView& result, const std::shared_ptr<const NumericalNode>& node,
                       NumericalOrdinal offset, std::vector<Crossing>& out) {
  if (!node->length) return;
  if (!node->left) {
    if (const auto provider = result.State().hierarchy->finite_providers.find(node->leaf);
        provider != result.State().hierarchy->finite_providers.end()) {
      for (const auto& [a, b] : provider->second.order.demands)
        out.emplace_back(DecodeNumericalOccurrence(result, NumericalAdd(offset, a)),
                         DecodeNumericalOccurrence(result, NumericalAdd(offset, b)));
      return;
    }
    const auto& segment = result.State().segments[node->leaf];
    const auto sites = result.State().input.segments[node->leaf].sites.size();
    for (const auto& demand : segment.demands)
      for (int64_t i = 0; i < segment.internal_scope_count - demand.distance; ++i) {
        const auto a = NumericalAdd(offset, Length(i, sites) + demand.source);
        const auto b = NumericalAdd(offset, Length(i + demand.distance, sites) + demand.target);
        out.emplace_back(DecodeNumericalOccurrence(result, a), DecodeNumericalOccurrence(result, b));
      }
    return;
  }
  ExpandNodeDemands(result, node->left, offset, out);
  const auto right_offset = NumericalAdd(offset, node->left->length);
  ExpandNodeDemands(result, node->right, right_offset, out);
  for (const auto& crossing : node->crossings) {
    if (crossing.native) continue;
    const auto a = NumericalAdd(offset, node->left->chains[crossing.source.first][crossing.source.second]);
    const auto b =
        NumericalAdd(right_offset, node->right->chains[crossing.target.first][crossing.target.second]);
    out.emplace_back(DecodeNumericalOccurrence(result, a), DecodeNumericalOccurrence(result, b));
  }
}
}  // namespace

std::optional<PointBoundary> BoundaryAt(const CompactSyncView& view, size_t storage, int64_t row,
                                        int64_t column, const std::map<size_t, int64_t>& parameters,
                                        std::optional<int64_t> block_index) {
  if (storage >= view.State().input.storage.size() || view.State().input.storage[storage].slot_count != 1)
    return std::nullopt;
  return BoundarySlot(view, storage, 0, row, column, parameters, block_index);
}
std::optional<PointBoundary> BoundarySlot(const CompactSyncView& view, size_t storage, int64_t slot,
                                          int64_t row, int64_t column,
                                          const std::map<size_t, int64_t>& parameters,
                                          std::optional<int64_t> block_index) {
  if (storage >= view.State().input.storage.size() || slot < 0 ||
      slot >= view.State().input.storage[storage].slot_count)
    return std::nullopt;
  if (const auto& coordinate = view.State().input.invocation_coordinate) {
    if (!block_index) return std::nullopt;
    block_index = coordinate->Map(*block_index, parameters);
    if (!block_index || *block_index < view.State().input.block_bounds->first ||
        *block_index > view.State().input.block_bounds->second)
      return std::nullopt;
  }
  const auto& domain = view.State().input.storage[storage];
  if (!domain.boundary_available || view.State().input.unavailable_storage_spaces.count(domain.space))
    return std::nullopt;
  if (domain.row_extent_parameter) {
    const auto parameter = *domain.row_extent_parameter;
    auto value = parameters.find(parameter);
    const auto bounds = view.State().input.parameter_bounds.at(parameter);
    if (value == parameters.end() || value->second < bounds.first || value->second > bounds.second ||
        row >= value->second)
      return std::nullopt;
  }
  if (row < 0 || column < 0 || row >= domain.rows || column >= domain.columns) return std::nullopt;
  if (view.State().hierarchy && view.State().storage_access_index.at(storage).external_geometry)
    return QueryExternalNumericalBoundary(view, storage, slot, row, column, parameters, block_index);
  auto shift = [&](const SyncTranslation& translation) -> std::optional<int64_t> {
    int64_t value = 0;
    for (auto [index, coefficient] : translation) {
      auto bound = view.State().input.parameter_bounds.find(index);
      auto supplied = parameters.find(index);
      if (bound == view.State().input.parameter_bounds.end() || supplied == parameters.end() ||
          supplied->second < bound->second.first || supplied->second > bound->second.second)
        return std::nullopt;
      const auto sum = __int128(value) + __int128(coefficient) * supplied->second;
      if (sum > std::numeric_limits<int64_t>::max()) return std::nullopt;
      value = static_cast<int64_t>(sum);
    }
    return value;
  };
  auto r = shift(domain.row_parameters), c = shift(domain.column_parameters);
  if (!r || !c) return std::nullopt;
  if (domain.row_block || domain.column_block) {
    if (!block_index || !view.State().input.block_bounds ||
        *block_index < view.State().input.block_bounds->first ||
        *block_index > view.State().input.block_bounds->second)
      return std::nullopt;
    const auto coordinate = *block_index - view.State().input.block_bounds->first;
    const auto row_shift = __int128(*r) + __int128(domain.row_block) * coordinate;
    const auto column_shift = __int128(*c) + __int128(domain.column_block) * coordinate;
    if (row_shift > std::numeric_limits<int64_t>::max() || column_shift > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    r = static_cast<int64_t>(row_shift);
    c = static_cast<int64_t>(column_shift);
  }
  if (row < *r || column < *c) return PointBoundary{};
  if (view.State().hierarchy && view.State().input.region_tree) {
    const auto& index = view.State().storage_access_index.at(storage);
    if (index.enclosing_motion) {
      if (index.written) return std::nullopt;
      PointBoundary out;
      for (const auto& [leaf, site, access_index] : index.accesses) {
        ++view.Diagnostics().point_selector_access_visits;
        const auto& segment = view.State().input.segments[leaf];
        const auto& access = segment.sites[site].accesses[access_index];
        auto visits = PointVisits(access, segment, row - *r, column - *c);
        if (!visits) continue;
        const auto pipe = static_cast<size_t>(segment.sites[site].pipe);
        const Occurrence first{leaf, visits->first, site}, last{leaf, visits->second, site};
        if (!out.first_readers[pipe] || view.Precedes(first, false, *out.first_readers[pipe], false))
          out.first_readers[pipe] = first;
        if (!out.last_readers[pipe] || view.Precedes(*out.last_readers[pipe], false, last, false))
          out.last_readers[pipe] = last;
      }
      return out;
    }
  }
  if (view.State().hierarchy) return QueryNumericalBoundary(view, storage, slot, row - *r, column - *c);
  if (view.State().segments.size() == 1 && view.State().segments.front().rotating)
    return RotatingBoundary(view.State().input, view.State().segments.front(), storage, slot, row - *r,
                            column - *c);
  return RelativeBoundary(view, storage, row - *r, column - *c);
}
std::vector<Crossing> ExpandRetainedDemands(const CompactSyncView& view, size_t max_payloads) {
  NumericalOrdinal count = 0;
  for (const auto& segment : view.State().input.segments)
    count = NumericalAdd(count, Length(segment.count, segment.sites.size()));
  CHECK(count <= max_payloads) << "Retained-demand audit exceeds its requested payload bound";
  std::vector<Crossing> out;
  if (view.State().hierarchy) {
    ExpandNodeDemands(view, view.State().hierarchy->root, 0, out);
  } else {
    out = view.State().crossing_demands;
    for (size_t r = 0; r < view.State().segments.size(); ++r) {
      const auto& segment = view.State().segments[r];
      for (const auto& demand : segment.demands)
        for (int64_t i = 0; i < segment.count - demand.distance; ++i)
          out.push_back({{r, i, demand.source}, {r, i + demand.distance, demand.target}});
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}
std::shared_ptr<const CompactSyncResult> SelectInvocationCase(const SyncInvocationAnalysis& analysis,
                                                              int64_t block_index,
                                                              const std::map<size_t, int64_t>& parameters) {
  if (block_index < analysis.block_bounds.first || block_index > analysis.block_bounds.second) return nullptr;
  for (const auto& [parameter, bound] : analysis.parameter_bounds) {
    auto value = parameters.find(parameter);
    if (value == parameters.end() || value->second < bound.first || value->second > bound.second)
      return nullptr;
  }
  if (analysis.coordinate) {
    const auto mapped = analysis.coordinate->Map(block_index, parameters);
    if (!mapped) return nullptr;
    block_index = *mapped;
  }
  auto found =
      std::upper_bound(analysis.cases.begin(), analysis.cases.end(), block_index,
                       [](int64_t value, const auto& candidate) { return value < candidate.block_first; });
  if (found == analysis.cases.begin()) return nullptr;
  --found;
  if (block_index > found->block_last) return nullptr;
  return found->mathematics;
}
}  // namespace pypto::python::sync_testing

namespace pypto::ir::sync {
std::optional<int64_t> EvaluateSyncCoordinate(const SyncCoordinateProgram& program, size_t root,
                                              const std::map<const ForStmt*, int64_t>& original_values,
                                              const std::map<size_t, int64_t>& parameters,
                                              std::optional<int64_t> block) {
  using Kind = SyncCoordinateNode::Kind;
  if (root >= program.nodes.size()) return std::nullopt;
  std::set<size_t> needed;
  std::vector<size_t> pending{root};
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!needed.insert(id).second) continue;
    const auto& n = program.nodes[id];
    auto dependency = [&](size_t child) {
      if (child >= id) return false;
      pending.push_back(child);
      return true;
    };
    if (n.kind == Kind::Scale || n.kind == Kind::ModularShift || n.kind == Kind::Not ||
        (n.kind == Kind::Loop && n.dynamic_start)) {
      if (!dependency(n.left)) return std::nullopt;
    } else if (n.kind == Kind::Add || n.kind == Kind::Sub || n.kind == Kind::FloorDivide ||
               n.kind == Kind::Remainder || n.kind == Kind::Equal || n.kind == Kind::NotEqual ||
               n.kind == Kind::LessThan || n.kind == Kind::LessEqual || n.kind == Kind::GreaterThan ||
               n.kind == Kind::GreaterEqual || n.kind == Kind::And || n.kind == Kind::Or) {
      if (!dependency(n.left) || !dependency(n.right)) return std::nullopt;
    } else if (n.kind == Kind::Select) {
      if (!dependency(n.condition) || !dependency(n.left) || !dependency(n.right)) return std::nullopt;
    }
  }
  std::map<size_t, int64_t> values;
  for (auto id : needed) {
    const auto& n = program.nodes[id];
    __int128 value = 0;
    switch (n.kind) {
      case Kind::Constant:
        value = n.upper;
        break;
      case Kind::Block:
        if (!block) return std::nullopt;
        value = *block;
        break;
      case Kind::Parameter: {
        const auto found = parameters.find(n.parameter);
        if (found == parameters.end()) return std::nullopt;
        value = found->second;
        break;
      }
      case Kind::Loop: {
        const auto found = original_values.find(n.frame.get());
        if (found == original_values.end() || n.step <= 0) return std::nullopt;
        value = found->second;
        const auto start = n.dynamic_start ? values.at(n.left) : n.start;
        if (value < start || (value - start) % n.step) return std::nullopt;
        break;
      }
      case Kind::Scale:
        value = __int128(values.at(n.left)) * n.factor;
        break;
      case Kind::ModularShift:
        value = __int128(values.at(n.left)) - n.factor;
        break;
      case Kind::Add:
        value = __int128(values.at(n.left)) + values.at(n.right);
        break;
      case Kind::Sub:
        value = __int128(values.at(n.left)) - values.at(n.right);
        break;
      case Kind::FloorDivide:
      case Kind::Remainder:
        if (values.at(n.right) <= 0) return std::nullopt;
        value = n.kind == Kind::FloorDivide ? values.at(n.left) / values.at(n.right)
                                            : values.at(n.left) % values.at(n.right);
        break;
      case Kind::Equal:
        value = values.at(n.left) == values.at(n.right) ? 1 : 0;
        break;
      case Kind::NotEqual:
        value = values.at(n.left) != values.at(n.right) ? 1 : 0;
        break;
      case Kind::LessThan:
        value = values.at(n.left) < values.at(n.right) ? 1 : 0;
        break;
      case Kind::LessEqual:
        value = values.at(n.left) <= values.at(n.right) ? 1 : 0;
        break;
      case Kind::GreaterThan:
        value = values.at(n.left) > values.at(n.right) ? 1 : 0;
        break;
      case Kind::GreaterEqual:
        value = values.at(n.left) >= values.at(n.right) ? 1 : 0;
        break;
      case Kind::And:
        value = (values.at(n.left) != 0 && values.at(n.right) != 0) ? 1 : 0;
        break;
      case Kind::Or:
        value = (values.at(n.left) != 0 || values.at(n.right) != 0) ? 1 : 0;
        break;
      case Kind::Not:
        value = values.at(n.left) == 0 ? 1 : 0;
        break;
      case Kind::Select:
        value = values.at(n.condition) != 0 ? values.at(n.left) : values.at(n.right);
        break;
    }
    if (value < n.lower || value > n.upper || value > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    values.emplace(id, static_cast<int64_t>(value));
  }
  return values.at(root);
}
}  // namespace pypto::ir::sync
