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

#include "src/ir/transforms/utils/physical_sync_schedule.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_placement.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace physical {
// Only generated predicates enter this simplifier. Original payload/control
// expressions are never rewritten. Bounds come from the retained occurrence
// frame. Each region caches an analyzer containing only immutable original-loop
// bounds; it never learns branch-local assumptions or a previous visit's values.
class CommandConditions {
  arith::Analyzer analyzer_;

 public:
  ExprPtr Simplify(const ExprPtr& expression) {
    try {
      return analyzer_.Simplify(expression);
    } catch (const StructuralComparisonUnsupported&) {
      return expression;
    }
  }
  explicit CommandConditions(const SyncSegment& segment) {
    auto bind = [&](const ForStmtPtr& loop) {
      const auto start = As<ConstInt>(loop->start_), stop = As<ConstInt>(loop->stop_),
                 step = As<ConstInt>(loop->step_);
      if (start && stop && step && step->value_ > 0 && start->value_ < stop->value_)
        analyzer_.Bind(loop->loop_var_, start->value_, stop->value_);
    };
    if (!segment.nested_frames.empty()) {
      for (const auto& frame : segment.nested_frames) bind(frame.loop);
    } else if (segment.original_loop) {
      bind(segment.original_loop);
    }
  }
};
using Family = PhysicalSyncFamily;
using AbsentLoops = std::map<const ForStmt*, Intervals>;
void RetainAbsentLoops(const SyncRegion& region, std::pair<int64_t, int64_t> blocks, AbsentLoops& absent) {
  if (region.repetition && !region.repetition->count) {
    auto& ranges = absent[region.repetition->loop.get()];
    // Cases arrive in increasing order; retain a normalized union without
    // visiting any dynamic occurrence or the body of an absent loop.
    if (!ranges.empty() && ranges.back().second + 1 == blocks.first)
      ranges.back().second = blocks.second;
    else
      ranges.push_back(blocks);
    return;
  }
  for (const auto& child : region.children) RetainAbsentLoops(child, blocks, absent);
}
class CutDomains : public IRVisitor {
  const AbsentLoops& absent_;
  CutInventory& cuts_;
  Intervals domain_;
  PhysicalSyncRealization& work_;
  std::set<const Var*> uniform_variables_;
  CommandPredicateScope scope_;
  size_t conditional_depth_ = 0;
  void VisitStmt_(const ForStmtPtr& loop) override {
    auto saved = domain_;
    work_.cut_domain_interval_visits += domain_.size();
    if (auto found = absent_.find(loop.get()); found != absent_.end()) {
      work_.cut_domain_interval_visits += found->second.size();
      domain_ = SubtractIntervals(domain_, found->second);
    }
    const auto start = As<ConstInt>(loop->start_), stop = As<ConstInt>(loop->stop_),
               step = As<ConstInt>(loop->step_);
    const bool uniform = start && stop && step && step->value_ > 0;
    auto saved_variables = uniform_variables_;
    if (!uniform) uniform_variables_.clear();
    if (uniform) uniform_variables_.insert(loop->loop_var_.get());
    const bool inserted = scope_.ready.insert(loop->loop_var_.get()).second;
    VisitStmt(loop->body_);
    if (inserted) scope_.ready.erase(loop->loop_var_.get());
    uniform_variables_ = std::move(saved_variables);
    domain_ = std::move(saved);
  }
  void VisitStmt_(const AssignStmtPtr&) override {}
  void VisitStmt_(const IfStmtPtr& branch) override {
    ++conditional_depth_;
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
    --conditional_depth_;
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (auto found = cuts_.find(call.get()); found != cuts_.end()) {
      work_.cut_domain_interval_visits += domain_.size();
      found->second->domain = domain_;
      // A branch could correlate a loop digit with the launch coordinate.
      // Without that branch, literal-bound IV domains are identical in every
      // launch case that reaches this cut. Do not infer this from sample runs.
      auto& cut = *found->second;
      if (!cut.predicate_scope) {
        cut.predicate_scope = scope_;
      } else {
        // Repeated static anchors can only use coordinates ready at every use.
        auto& ready = cut.predicate_scope->ready;
        for (auto it = ready.begin(); it != ready.end();) {
          if (!scope_.ready.count(*it))
            it = ready.erase(it);
          else
            ++it;
        }
      }
      const auto qualified = conditional_depth_ ? std::set<const Var*>{} : uniform_variables_;
      if (!cut.uniform_checked) {
        cut.uniform_variables = qualified;
      } else {
        for (auto it = cut.uniform_variables.begin(); it != cut.uniform_variables.end();) {
          if (!qualified.count(*it))
            it = cut.uniform_variables.erase(it);
          else
            ++it;
        }
      }
      cut.uniform_checked = true;
    }
  }

 public:
  CutDomains(const FunctionPtr& owner, const AbsentLoops& absent, CutInventory& cuts,
             std::pair<int64_t, int64_t> blocks, PhysicalSyncRealization& work)
      : absent_(absent), cuts_(cuts), domain_{blocks}, work_(work), scope_{owner, nullptr, {}} {
    auto entry = std::make_shared<std::set<const Var*>>();
    for (const auto& parameter : owner->params_) {
      const auto type = As<ScalarType>(parameter->GetType());
      if (type && (type->dtype_ == DataType::INDEX || type->dtype_ == DataType::BOOL))
        entry->insert(parameter.get());
    }
    // This helper returns only the original leading launch lookup, which
    // dominates every payload. Other scalar assignments confer no evidence.
    if (auto block = As<Var>(DominatingSyncBlock(owner))) entry->insert(block.get());
    scope_.entry_ready = std::move(entry);
    for (auto& [anchor, cut] : cuts_) cut->domain = domain_;
  }
};
// A synchronous schedule owns its endpoint inventory and original function;
// input/allocation owners outlive compilation and are borrowed, never copied.
// Collection retains domains and ID-table references without constructing a
// scalar predicate, physical call or per-ID command alternative.
struct AssignedEndpoint {
  size_t case_index, family_index;
  bool publish;
  std::shared_ptr<const SyncEndpointDomain> domain;
};
struct AssignedCut {
  size_t case_index;
  CallPtr original, continuation;
  std::vector<AssignedEndpoint> before, after;
};
class ScheduleDomains {
  const CompactSyncInput& input_;
  std::map<size_t, std::shared_ptr<const SyncEndpointFrame>> endpoint_frames_;
  using EndpointKey = std::pair<size_t, std::vector<std::pair<int64_t, int64_t>>>;
  std::map<EndpointKey, std::shared_ptr<const SyncEndpointDomain>> endpoint_domains_;

 public:
  explicit ScheduleDomains(const CompactSyncInput& input) : input_(input) {}
  std::shared_ptr<const SyncEndpointDomain> Get(const Family& f, bool publish) {
    const auto r = publish ? f.source_region : f.target_region;
    auto retained = publish ? f.source_domain : f.target_domain;
    const auto& frames = input_.segments[r].nested_frames;
    if (retained) {
      INTERNAL_CHECK(retained->frame) << "Missing original endpoint frame";
      const auto& context = *retained->frame;
      INTERNAL_CHECK(context.owner == input_.owner && context.blocks == input_.block_bounds &&
                     context.region == r && context.loops.size() == frames.size() &&
                     retained->ranges.size() == frames.size())
          << "Foreign endpoint domain";
      for (size_t i = 0; i < frames.size(); ++i) {
        const auto [first, end] = retained->ranges[i];
        INTERNAL_CHECK(first >= 0 && first < end && end <= frames[i].count)
            << "Invalid retained endpoint interval";
        INTERNAL_CHECK(context.loops[i].loop == frames[i].loop && context.loops[i].count == frames[i].count &&
                       context.loops[i].start == frames[i].start && context.loops[i].step == frames[i].step &&
                       context.loops[i].start_expression == frames[i].start_expression)
            << "Incompatible original endpoint frame";
      }
      return retained;
    }
    // Mathematics-only clients use the same domain representation. Decode once
    // at this compatibility boundary; actual-token clients never take this path.
    auto base = publish ? f.source_first : f.target_first;
    std::vector<std::pair<int64_t, int64_t>> ranges(frames.size());
    for (size_t i = frames.size(); i-- > 0;) {
      INTERNAL_CHECK(frames[i].count > 0) << "Physical endpoint in absent frame";
      const auto first = base % frames[i].count;
      base /= frames[i].count;
      const auto count = i < f.axes.size() ? f.axes[i].count : 1;
      INTERNAL_CHECK(i >= f.axes.size() || f.axes[i].frame == frames[i].loop) << "Physical scope identity";
      INTERNAL_CHECK(count > 0 && count <= frames[i].count - first) << "Physical endpoint range";
      ranges[i] = {first, first + count};
    }
    INTERNAL_CHECK(!base && f.axes.size() <= frames.size()) << "Physical scope arity";
    EndpointKey key{r, std::move(ranges)};
    if (auto found = endpoint_domains_.find(key); found != endpoint_domains_.end()) {
      return found->second;
    }
    auto& frame = endpoint_frames_[r];
    if (!frame)
      frame = std::make_shared<SyncEndpointFrame>(
          SyncEndpointFrame{input_.owner, input_.block_bounds, r, frames});
    auto domain = std::make_shared<SyncEndpointDomain>(SyncEndpointDomain{frame, key.second});
    endpoint_domains_.emplace(std::move(key), domain);
    return domain;
  }
};
class AssignedRegionSchedule {
 public:
  const FunctionPtr owner;
  const ExprPtr block;  // One source-invariant dominating launch lookup.
  const std::vector<AssignedSyncCase> cases;
  std::vector<AssignedCut> cuts;
  AssignedRegionSchedule(FunctionPtr function, std::vector<AssignedSyncCase> supplied)
      : owner(std::move(function)), block(DominatingSyncBlock(owner)), cases(std::move(supplied)) {
    for (size_t c = 0; c < cases.size(); ++c) {
      CHECK(cases[c].input && cases[c].assignment && cases[c].work) << "Missing assigned schedule case";
      const auto& input = *cases[c].input;
      auto& work = *cases[c].work;
      CHECK(work.assignment == cases[c].assignment) << "Assigned schedule accounting belongs to another plan";
      cases[c].assignment->ValidateFor(input, input.hardware_target);
      const auto& families = cases[c].assignment->Families();
      INTERNAL_CHECK(input.owner == owner) << "Foreign assigned schedule owner";
      ScheduleDomains domains(input);
      std::map<SyncSiteKey, std::vector<size_t>> incoming, outgoing;
      for (size_t f = 0; f < families.size(); ++f) {
        const auto& family = families[f];
        incoming[{family.target_region, family.target_site}].push_back(f);
        if (!family.ids.empty()) outgoing[{family.source_region, family.source_site}].push_back(f);
      }
      for (size_t r = 0; r < input.segments.size(); ++r) {
        const auto& region = input.segments[r];
        if (!region.count) continue;
        for (size_t site = 0; site < region.sites.size(); ++site) {
          const SyncSiteKey key{r, site};
          const auto& payload = region.sites[site];
          AssignedCut cut{c, payload.original, payload.continuation, {}, {}};
          auto append = [&](size_t f, bool publish, std::vector<AssignedEndpoint>& target) {
            const auto& family = families[f];
            target.push_back({c, f, publish, input.region_tree ? domains.Get(family, publish) : nullptr});
            ++work.assigned_schedule_endpoints;
          };
          for (auto f : incoming[key])
            if (families[f].ids.empty()) append(f, false, cut.before);
          for (auto f : incoming[key])
            if (!families[f].ids.empty()) append(f, false, cut.before);
          for (auto f : outgoing[key]) append(f, true, cut.after);
          if (cut.before.empty() && cut.after.empty()) continue;
          cuts.push_back(std::move(cut));
        }
      }
    }
  }
  const Family& Record(const AssignedEndpoint& endpoint) const {
    return cases[endpoint.case_index].assignment->Families()[endpoint.family_index];
  }
};
const Call* SingletonCall(StmtPtr body) {
  if (auto seq = As<SeqStmts>(body)) {
    if (seq->stmts_.size() != 1) return nullptr;
    body = seq->stmts_.front();
  }
  auto eval = As<EvalStmt>(body);
  return eval ? As<Call>(eval->expr_).get() : nullptr;
}
struct SchedulePhase {
  ForStmtPtr loop;
  int64_t first, end, period = 1;
  // Interval specialization alone never supplies an ordinal congruence.
  std::optional<int64_t> residue;
};
// This scope proof admits no allocations or effectful definitions. Slot views
// select an existing placed allocation and stay at their original per-visit cut.
// DeepClone later refreshes their SSA names, not the captured allocation/root.
class RegularBody {
  const std::map<const Call*, const Call*>& admitted_;
  bool yielded_ = false;

 public:
  bool valid = true;
  std::vector<const Call*> payloads;
  explicit RegularBody(const std::map<const Call*, const Call*>& admitted) : admitted_(admitted) {}
  void Visit(const StmtPtr& body) {
    if (auto seq = As<SeqStmts>(body)) {
      for (const auto& s : seq->stmts_) Visit(s);
      return;
    }
    if (yielded_) {
      valid = false;
      return;
    }
    if (auto yield = As<YieldStmt>(body)) {
      valid &= yield->value_.empty();
      yielded_ = true;
      return;
    }
    if (auto assign = As<AssignStmt>(body)) {
      auto call = As<Call>(assign->value_);
      valid &= IsOp(call, "buffer.slot");
      return;
    }
    if (auto branch = As<IfStmt>(body)) {
      auto a = admitted_.find(SingletonCall(branch->then_body_));
      auto b = branch->else_body_ ? admitted_.find(SingletonCall(*branch->else_body_)) : admitted_.end();
      if (!branch->return_vars_.empty() || a == admitted_.end() || b == admitted_.end() ||
          a->second != b->second) {
        valid = false;
        return;
      }
      payloads.push_back(a->second);
      return;
    }
    auto eval = As<EvalStmt>(body);
    auto found = eval ? admitted_.find(As<Call>(eval->expr_).get()) : admitted_.end();
    if (found == admitted_.end()) {
      valid = false;
      return;
    }
    payloads.push_back(found->second);
  }
};

struct LoopUse {
  size_t cut_index;
  const AssignedEndpoint* endpoint;
  int64_t first, end;
  bool full_outer;
};
RegularSchedules CollectRegularLoops(const AssignedRegionSchedule& schedule, PhysicalSyncRealization& work) {
  std::map<const Call*, const Call*> admitted;
  std::map<const ForStmt*, std::vector<LoopUse>> uses;
  std::map<const ForStmt*, ForStmtPtr> loops;
  std::map<const Call*, std::vector<size_t>> anchor_cuts;
  for (size_t c = 0; c < schedule.cases.size(); ++c)
    for (const auto& segment : schedule.cases[c].input->segments) {
      for (const auto& site : segment.sites) {
        admitted[site.original.get()] = site.original.get();
        if (site.continuation) admitted[site.continuation.get()] = site.original.get();
      }
    }
  for (size_t i = 0; i < schedule.cuts.size(); ++i) anchor_cuts[schedule.cuts[i].original.get()].push_back(i);
  for (size_t index = 0; index < schedule.cuts.size(); ++index) {
    const auto& cut = schedule.cuts[index];
    for (const auto* side : {&cut.before, &cut.after})
      for (const auto& endpoint : *side) {
        const auto& input = *schedule.cases[endpoint.case_index].input;
        const auto& family = schedule.Record(endpoint);
        const auto& segment = input.segments[endpoint.publish ? family.source_region : family.target_region];
        ForStmtPtr loop;
        int64_t first = endpoint.publish ? family.source_first : family.target_first;
        int64_t end = 0;
        bool full_outer = true;
        if (endpoint.domain) {
          const auto& frames = endpoint.domain->frame->loops;
          if (frames.empty()) continue;
          loop = frames.back().loop;
          std::tie(first, end) = endpoint.domain->ranges.back();
          for (size_t i = 0; i + 1 < frames.size(); ++i)
            full_outer &=
                endpoint.domain->ranges[i] == std::make_pair(static_cast<int64_t>(0), frames[i].count);
        } else {
          if (!segment.nested_frames.empty()) continue;
          loop = segment.original_loop;
          const __int128 limit = __int128(first) + family.count;
          if (limit > std::numeric_limits<int64_t>::max()) continue;
          end = static_cast<int64_t>(limit);
        }
        if (!loop) continue;
        loops[loop.get()] = loop;
        uses[loop.get()].push_back({index, &endpoint, first, end, full_outer});
      }
  }
  RegularSchedules plans;
  for (const auto& [key, inventory] : uses) {
    const auto loop = loops.at(key);
    const auto type = As<ScalarType>(loop->loop_var_->GetType());
    const auto start = As<ConstInt>(loop->start_), stop = As<ConstInt>(loop->stop_),
               step = As<ConstInt>(loop->step_);
    if (!type || type->dtype_ != DataType::INDEX || loop->kind_ != ForKind::Sequential ||
        !loop->iter_args_.empty() || !loop->return_vars_.empty() || !loop->attrs_.empty() ||
        loop->hardware_loop_policy_ != HardwareLoopPolicy::Auto || !start || !stop || !step ||
        step->value_ <= 0 || start->value_ >= stop->value_)
      continue;
    const __int128 wide_count = (__int128(stop->value_) - start->value_ + step->value_ - 1) / step->value_;
    if (wide_count < 2 || wide_count > std::numeric_limits<int64_t>::max() ||
        wide_count * step->value_ > std::numeric_limits<int64_t>::max())
      continue;
    const auto count = static_cast<int64_t>(wide_count);
    if (__int128(start->value_) + wide_count * step->value_ > std::numeric_limits<int64_t>::max()) continue;
    RegularBody body(admitted);
    body.Visit(loop->body_);
    if (!body.valid || body.payloads.empty() ||
        std::set<const Call*>(body.payloads.begin(), body.payloads.end()).size() != body.payloads.size())
      continue;
    ++work.regular_schedule_candidates;
    int64_t period = 1;
    bool valid = true;
    std::set<int64_t> boundaries{0, count};
    for (const auto& use : inventory) {
      const auto& endpoint = *use.endpoint;
      const auto& family = schedule.Record(endpoint);
      const auto p = static_cast<int64_t>(family.ids.size());
      if (use.first < 0 || use.end > count || use.first >= use.end) {
        valid = false;
        break;
      }
      if (p > 1) {
        if (endpoint.domain || period == 0 || (period > 1 && period != p)) {
          period = 0;  // Only periodic expansion loses its optional certificate.
        } else {
          period = p;
        }
      }
      const auto& input = *schedule.cases[endpoint.case_index].input;
      const auto& segment = input.segments[endpoint.publish ? family.source_region : family.target_region];
      const auto& frame = endpoint.domain
                              ? endpoint.domain->frame->loops.back()
                              : SyncLoopFrame{loop, segment.count, segment.start, segment.step, {}};
      if (frame.count != count || frame.start != start->value_ || frame.step != step->value_ ||
          frame.start_expression) {
        valid = false;
        break;
      }
      boundaries.insert(use.first);
      boundaries.insert(use.end);
    }
    if (!valid) continue;
    if (__int128(period) * step->value_ > std::numeric_limits<int64_t>::max()) period = 0;
    RegularLoopSchedule plan{loop,
                             count,
                             start->value_,
                             step->value_,
                             period,
                             {},
                             std::vector<std::vector<size_t>>(schedule.cases.size()),
                             {},
                             body.payloads.size()};
    std::vector<int64_t> edges(boundaries.begin(), boundaries.end());
    // A schedule replaces the WHOLE body. Every command at every body cut
    // in every case must belong to this same loop/frame, even if another case
    // used a different regional representation for that original operation.
    std::map<size_t, size_t> covered;
    for (const auto& use : inventory) ++covered[use.cut_index];
    const std::set<const Call*> body_anchors(body.payloads.begin(), body.payloads.end());
    for (const auto* anchor : body_anchors)
      for (const auto i : anchor_cuts[anchor]) {
        const auto& cut = schedule.cuts[i];
        if (covered[i] != cut.before.size() + cut.after.size()) {
          valid = false;
          break;
        }
        plan.case_cuts[cut.case_index].push_back(i);
        plan.anchors.insert(cut.original.get());
        if (cut.continuation) plan.anchors.insert(cut.continuation.get());
      }
    if (!valid) continue;
    for (size_t i = 1; i < edges.size(); ++i) plan.phases.emplace_back(edges[i - 1], edges[i]);
    plans.emplace(key, std::move(plan));
  }
  return plans;
}

// Baseline schedule choice. It consumes retained endpoints only after the
// complete symbolic inventory has been collected for every invocation case.
class AssignedCommandLowering {
 public:
  AssignedCommandLowering(const AssignedRegionSchedule& schedule, size_t case_index,
                          PhysicalSyncRealization& work, const SchedulePhase* phase = nullptr)
      : input_(*schedule.cases[case_index].input),
        schedule_(schedule),
        case_(case_index),
        result_(work),
        phase_(phase) {}

  void AppendCuts(CutInventory& inventory, std::pair<int64_t, int64_t> blocks,
                  const std::vector<size_t>& selected) {
    for (auto index : selected) {
      const auto& cut = schedule_.cuts[index];
      auto commands = LowerCut(cut);
      auto& retained = inventory[cut.original.get()];
      if (!retained) retained = std::make_shared<PhysicalCuts>();
      if (cut.continuation) {
        auto& continuation = inventory[cut.continuation.get()];
        INTERNAL_CHECK(!continuation || continuation == retained)
            << "Internal error: incompatible original initialization cut";
        continuation = retained;
      }
      AppendCase(retained->before_cases, blocks, commands.before, cut.original->span_, result_);
      AppendCase(retained->after_cases, blocks, commands.after, cut.original->span_, result_);
    }
  }

 private:
  const CompactSyncInput& input_;
  const AssignedRegionSchedule& schedule_;
  size_t case_;
  PhysicalSyncRealization& result_;
  const SchedulePhase* phase_;
  std::map<const SyncEndpointDomain*, std::vector<ExprPtr>> endpoint_guards_;
  std::map<size_t, std::unique_ptr<CommandConditions>> conditions_;
  std::map<size_t, ExprPtr> ordinals_;
  CommandConditions& Conditions(size_t region) {
    auto& value = conditions_[region];
    if (!value) value = std::make_unique<CommandConditions>(input_.segments[region]);
    return *value;
  }

  ExprPtr Integer(int64_t n, const Span& span) {
    return std::make_shared<ConstInt>(n, DataType::INDEX, span);
  }
  ExprPtr Ordinal(size_t region, const Span& span) {
    auto& value = ordinals_[region];
    if (!value) value = BuildOrdinal(region, span);
    return value;
  }
  ExprPtr BuildOrdinal(size_t region, const Span& span) {
    const auto& s = input_.segments[region];
    if (!s.original_loop) return Integer(0, span);
    if (!s.nested_frames.empty()) {
      ExprPtr ordinal = Integer(0, span);
      for (const auto& frame : s.nested_frames) {
        ExprPtr digit = frame.loop->loop_var_;
        if (frame.start)
          digit = std::make_shared<Sub>(digit, Integer(frame.start, span), DataType::INDEX, span);
        if (frame.step != 1)
          digit = std::make_shared<FloorDiv>(digit, Integer(frame.step, span), DataType::INDEX, span);
        ordinal = std::make_shared<Add>(
            std::make_shared<Mul>(ordinal, Integer(frame.count, span), DataType::INDEX, span), digit,
            DataType::INDEX, span);
      }
      return ordinal;
    }
    ExprPtr ordinal = s.original_loop->loop_var_;
    if (s.start) ordinal = std::make_shared<Sub>(ordinal, Integer(s.start, span), DataType::INDEX, span);
    if (s.step != 1)
      ordinal = std::make_shared<FloorDiv>(ordinal, Integer(s.step, span), DataType::INDEX, span);
    return ordinal;
  }
  ExprPtr FrameDigit(const SyncLoopFrame& frame, const Span& span) {
    // This original loop has exactly one admitted visit regardless of its
    // invocation-dependent start. No normalized placeholder is read as an IV.
    if (frame.count == 1) return Integer(0, span);
    ExprPtr value = frame.loop->loop_var_;
    if (frame.start_expression || frame.start)
      value = std::make_shared<Sub>(
          value, frame.start_expression ? frame.start_expression : Integer(frame.start, span),
          DataType::INDEX, span);
    if (frame.step != 1)
      value = std::make_shared<FloorDiv>(value, Integer(frame.step, span), DataType::INDEX, span);
    return value;
  }
  // Rewrite only on the original positive-step lattice. In addition to each
  // threshold, prove the normalized subtraction itself cannot wrap: endpoints
  // can both fit int64 while their difference does not. A symbolic start keeps
  // the established quotient expression; no new range assumption is introduced.
  std::optional<int64_t> OriginalBoundary(const SyncLoopFrame& frame, int64_t digit) const {
    const auto start = As<ConstInt>(frame.loop->start_), step = As<ConstInt>(frame.loop->step_);
    if (!start || !step || frame.start_expression || frame.step <= 0 || frame.count <= 0 ||
        start->value_ != frame.start || step->value_ != frame.step || digit < 0 || digit >= frame.count)
      return std::nullopt;
    const auto extent = __int128(frame.count - 1) * frame.step;
    const auto last = __int128(frame.start) + extent;
    const auto bound = __int128(frame.start) + __int128(digit) * frame.step;
    if (extent > std::numeric_limits<int64_t>::max() || last > std::numeric_limits<int64_t>::max() ||
        last < std::numeric_limits<int64_t>::min() || bound > std::numeric_limits<int64_t>::max() ||
        bound < std::numeric_limits<int64_t>::min())
      return std::nullopt;
    return static_cast<int64_t>(bound);
  }
  const std::vector<ExprPtr>& ScopedGuards(const AssignedEndpoint& endpoint, const Span& span) {
    const auto& domain = endpoint.domain;
    if (auto found = endpoint_guards_.find(domain.get()); found != endpoint_guards_.end())
      return found->second;
    const auto& frames = domain->frame->loops;
    result_.endpoint_domain_entries += domain->ranges.size();
    std::vector<ExprPtr> guards;
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto [first, end] = domain->ranges[i];
      INTERNAL_CHECK(first >= 0 && end > first && end <= frames[i].count) << "Invalid endpoint interval";
      if (phase_ && frames[i].loop == phase_->loop && phase_->first >= first && phase_->end <= end) continue;
      if (!first && end == frames[i].count) continue;
      // Each requested threshold is within the admitted frame, including
      // nondivisible stop values. Never substitute stop-step for the last IV.
      const auto first_bound = OriginalBoundary(frames[i], first);
      const auto end_bound = end < frames[i].count ? OriginalBoundary(frames[i], end) : first_bound;
      const bool original = first_bound && end_bound;
      const auto value = original ? frames[i].loop->loop_var_ : FrameDigit(frames[i], span);
      auto boundary = [&](int64_t digit) {
        if (original) {
          ++result_.normalized_endpoint_bounds;
          return Integer(*OriginalBoundary(frames[i], digit), span);
        }
        return Integer(digit, span);
      };
      if (end - first == 1) {
        guards.push_back(std::make_shared<Eq>(value, boundary(first), DataType::BOOL, span));
      } else {
        if (first) guards.push_back(std::make_shared<Ge>(value, boundary(first), DataType::BOOL, span));
        if (end < frames[i].count)
          guards.push_back(std::make_shared<Lt>(value, boundary(end), DataType::BOOL, span));
      }
    }
    result_.endpoint_predicates_constructed += guards.size();
    for (auto& guard : guards) guard = Conditions(domain->frame->region).Simplify(guard);
    return endpoint_guards_.emplace(domain.get(), std::move(guards)).first->second;
  }
  ControlPtr Guard(ExprPtr condition, ControlPtr body) {
    ++result_.guard_sites;
    if (auto known = As<ConstBool>(condition)) {
      return known->value_ ? body : ControlSequence({}, body->span, result_);
    }
    return ControlBranch(condition, body, std::nullopt, body->span, result_);
  }
  ControlPtr LowerAssignedCommand(const AssignedEndpoint& endpoint, const Span& span) {
    INTERNAL_CHECK(endpoint.case_index == case_) << "Foreign assigned endpoint case";
    const auto& f = schedule_.Record(endpoint);
    const bool publish = endpoint.publish;
    const auto [src, dst] = SyncFamilyPipes(input_, f);
    const size_t r = publish ? f.source_region : f.target_region;
    const int64_t first = publish ? f.source_first : f.target_first;
    const bool scoped = input_.region_tree.has_value();
    // The initial scoped certificate deliberately proves period one. Keep
    // physical table coordinates separate from the full logical record tuple.
    INTERNAL_CHECK(!scoped || f.ids.size() <= 1) << "Internal error: unsupported scoped physical period";
    // Decide activity before constructing calls or predicates. Unknown outer
    // dimensions remain in the retained domain and are emitted normally.
    if (phase_) {
      int64_t lo = first, hi = scoped ? 0 : first + f.count;
      bool local = input_.segments[r].original_loop == phase_->loop;
      if (endpoint.domain) {
        local = !endpoint.domain->frame->loops.empty() &&
                endpoint.domain->frame->loops.back().loop == phase_->loop;
        if (local) std::tie(lo, hi) = endpoint.domain->ranges.back();
      }
      INTERNAL_CHECK(local) << "Assigned schedule phase belongs to another endpoint loop";
      if (phase_->end <= lo || phase_->first >= hi) return ControlSequence({}, span, result_);
    }
    const auto ordinal = scoped ? Integer(0, span) : Ordinal(r, span);
    ExprPtr index = ordinal;
    if (!scoped && first) index = std::make_shared<Sub>(index, Integer(first, span), DataType::INDEX, span);
    ControlForest commands;
    if (src == dst) {
      INTERNAL_CHECK(dst == PipeType::M || dst == PipeType::V)
          << "Only qualified M/V barriers may be emitted; scalar barriers are forbidden";
      auto call = OpRegistry::GetInstance().Create(dst == PipeType::M ? "system.bar_m" : "system.bar_v", {},
                                                   {}, span);
      commands.push_back(ControlCommand(call, span, result_));
      ++result_.barrier_sites;
    } else {
      ExprPtr residue;
      const bool literal_residue = phase_ && phase_->residue.has_value() && !scoped;
      if (f.ids.size() > 1 && !literal_residue)
        residue = std::make_shared<FloorMod>(index, Integer(f.ids.size(), span), DataType::INDEX, span);
      ControlPtr selection;
      // This is the complete table for FloorMod(index, positive table size),
      // whose range is [0, size). After excluding residues 0..size-2, the last
      // entry is exhaustive. This proof belongs to this table constructor,
      // not arbitrary conditional trees. Keep the entire selection beneath
      // the original presence guards, including lower bounds on index.
      for (size_t i = f.ids.size(); i-- > 0;) {
        if (literal_residue) {
          const auto period = static_cast<int64_t>(f.ids.size());
          const auto digit =
              static_cast<int64_t>((__int128(phase_->first) + *phase_->residue - first) % period);
          if (i != static_cast<size_t>((digit + period) % period)) continue;
        }
        auto call = OpRegistry::GetInstance().Create(
            publish ? "system.set_flag" : "system.wait_flag", {},
            {{"src_pipe", static_cast<int>(src)},
             {"dst_pipe", static_cast<int>(dst)},
             {"event_id", f.ids[i]},
             {"core_type", std::string(input_.owner->func_type_ == FunctionType::AIV ? "aiv" : "aic")}},
            span);
        ControlPtr command = ControlCommand(call, span, result_);
        if (residue && i + 1 < f.ids.size()) {
          ++result_.guard_sites;
          command = ControlBranch(
              Conditions(r).Simplify(std::make_shared<Eq>(residue, Integer(i, span), DataType::BOOL, span)),
              command, selection ? std::optional<ControlPtr>(selection) : std::nullopt, span, result_);
        }
        selection = command;
        if (publish)
          ++result_.set_sites;
        else
          ++result_.wait_sites;
      }
      if (selection) commands.push_back(selection);
    }
    ControlPtr body = ControlSequence(commands, span, result_);
    // The range guard surrounds residue arithmetic: no absent predecessor is
    // evaluated/consumed. Final-only exports never publish on earlier visits.
    if (scoped) {
      const auto& guards = ScopedGuards(endpoint, span);
      for (auto guard = guards.rbegin(); guard != guards.rend(); ++guard) body = Guard(*guard, body);
      return body;
    }
    if (first + f.count < input_.segments[r].count && (!phase_ || phase_->end > first + f.count))
      body = Guard(Conditions(r).Simplify(
                       std::make_shared<Lt>(ordinal, Integer(first + f.count, span), DataType::BOOL, span)),
                   body);
    if (first && (!phase_ || phase_->first < first))
      body = Guard(
          Conditions(r).Simplify(std::make_shared<Ge>(ordinal, Integer(first, span), DataType::BOOL, span)),
          body);
    return body;
  }
  PhysicalCuts LowerCut(const AssignedCut& cut) {
    INTERNAL_CHECK(cut.case_index == case_) << "Foreign assigned cut";
    PhysicalCuts cuts;
    const auto previous_guards = result_.guard_sites;
    const auto& span = cut.original->span_;
    // The shared owner already ordered barriers, WAITs and SETs. Consume its
    // immutable inventories directly; no second site directory is needed.
    for (const auto& endpoint : cut.before) cuts.before.push_back(LowerAssignedCommand(endpoint, span));
    for (const auto& endpoint : cut.after) cuts.after.push_back(LowerAssignedCommand(endpoint, span));
    const auto remaining = CaseCodeSize(cuts.before).guards + CaseCodeSize(cuts.after).guards;
    INTERNAL_CHECK(result_.guard_sites >= previous_guards + remaining) << "Unexpected new command guards";
    result_.simplified_guard_sites += result_.guard_sites - previous_guards - remaining;
    result_.guard_sites = previous_guards + remaining;
    cuts.before = ShareCommandGuards(cuts.before, result_);
    cuts.after = ShareCommandGuards(cuts.after, result_);
    return cuts;
  }
};
CutInventory LowerScheduleCuts(const AssignedRegionSchedule& schedule, const CutInventory& domains,
                               const std::vector<std::vector<size_t>>& selected,
                               std::pair<int64_t, int64_t> blocks, PhysicalSyncRealization& work,
                               const SchedulePhase* phase = nullptr) {
  CutInventory cuts;
  const auto block = schedule.cases.size() > 1 ? schedule.block : nullptr;
  for (size_t c = 0; c < schedule.cases.size(); ++c)
    AssignedCommandLowering(schedule, c, work, phase)
        .AppendCuts(cuts, schedule.cases[c].input->block_bounds.value_or(blocks), selected[c]);
  std::set<const PhysicalCuts*> seen;
  for (const auto& [anchor, cut] : cuts) {
    if (!seen.insert(cut.get()).second) continue;
    const auto& facts = *domains.at(anchor);
    INTERNAL_CHECK(facts.predicate_scope && facts.predicate_scope->owner == schedule.owner)
        << "Missing original-cut predicate qualification";
    cut->predicate_scope = facts.predicate_scope;
    for (const auto& item : cut->before_cases) RequireCommandScope({item.body}, *cut->predicate_scope, work);
    for (const auto& item : cut->after_cases) RequireCommandScope({item.body}, *cut->predicate_scope, work);
    cut->before = ScheduleCommandCases(block, facts.domain, cut->before_cases, work, schedule.owner->span_,
                                       facts.uniform_variables);
    cut->after = ScheduleCommandCases(block, facts.domain, cut->after_cases, work, schedule.owner->span_,
                                      facts.uniform_variables);
    RequireCommandScope(cut->before, *cut->predicate_scope, work);
    RequireCommandScope(cut->after, *cut->predicate_scope, work);
  }
  return cuts;
}

// A boundary schedule moves only commands at the first/last original payload
// cut of a known-nonempty sequential loop. No payload is copied or reordered.

// Reject definitions, arbitrary control, unknown operations and nested loops.
// Thus every captured value dominates the loop, and moving a boundary command
// crosses no original scalar computation or effect. An init/acc alternative is
// one original cut only when both arms already share that cut.
class BoundaryInput {
  const CutInventory& inventory_;
  PhysicalSyncRealization& work_;
  bool yielded_ = false;

 public:
  bool valid = true;
  std::vector<const PhysicalCuts*> cuts;
  BoundaryInput(const CutInventory& inventory, PhysicalSyncRealization& work)
      : inventory_(inventory), work_(work) {}
  void Visit(const StmtPtr& stmt) {
    ++work_.schedule_body_visits;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) Visit(child);
      return;
    }
    if (yielded_) {
      valid = false;
      return;
    }
    if (auto yield = As<YieldStmt>(stmt)) {
      valid &= yield->value_.empty();
      yielded_ = true;
      return;
    }
    const auto call = SingletonCall(stmt);
    auto found = inventory_.find(call);
    if (auto branch = As<IfStmt>(stmt)) {
      found = inventory_.find(SingletonCall(branch->then_body_));
      const auto other =
          branch->else_body_ ? inventory_.find(SingletonCall(*branch->else_body_)) : inventory_.end();
      if (!branch->return_vars_.empty() || found == inventory_.end() || other == inventory_.end() ||
          found->second != other->second) {
        valid = false;
        return;
      }
    }
    if (found == inventory_.end()) {
      valid = false;
      return;
    }
    cuts.push_back(found->second.get());
  }
};
class BoundarySpecializer {
  arith::Analyzer analyzer_;
  PhysicalSyncRealization& work_;

 public:
  BoundarySpecializer(const VarPtr& iv, int64_t first, int64_t end, PhysicalSyncRealization& work)
      : work_(work) {
    analyzer_.Bind(iv, first, end);
  }
  ControlForest Run(const ControlForest& forest) {
    ControlForest out;
    for (const auto& node : forest) {
      ++work_.schedule_descriptor_visits;
      if (node->command) {
        out.push_back(node);
        continue;
      }
      if (node->condition) {
        ExprPtr simplified = node->condition;
        try {
          simplified = analyzer_.Simplify(node->condition);
          if (!As<ConstBool>(simplified)) {
            if (analyzer_.CanProve(simplified))
              simplified = std::make_shared<ConstBool>(true, node->span);
            else if (analyzer_.CanProve(MakeNot(simplified)))
              simplified = std::make_shared<ConstBool>(false, node->span);
          }
        } catch (const StructuralComparisonUnsupported&) { /* Unproved remains unchanged. */
        }
        if (auto known = As<ConstBool>(simplified)) {
          auto selected = Run(known->value_ ? node->body : node->alternative);
          out.insert(out.end(), selected.begin(), selected.end());
          continue;
        }
      }
      auto body = Run(node->body), alternative = Run(node->alternative);
      if (body.empty() && alternative.empty()) continue;
      if (!node->condition) {
        out.insert(out.end(), body.begin(), body.end());
        continue;
      }
      out.push_back(
          RebuildControl(node, node->condition, nullptr, std::move(body), std::move(alternative), work_));
    }
    return out;
  }
};
class BindBoundary : public IRMutator {
  const VarPtr& iv_;
  int64_t value_;
  ExprPtr VisitExpr_(const VarPtr& var) override {
    if (var == iv_)
      return std::make_shared<ConstInt>(value_, As<ScalarType>(iv_->GetType())->dtype_, var->span_);
    return var;
  }

 public:
  BindBoundary(const VarPtr& iv, int64_t value) : iv_(iv), value_(value) {}
  ControlForest Run(const ControlForest& forest, PhysicalSyncRealization& work) {
    ControlForest out;
    for (const auto& node : forest) {
      out.push_back(RebuildControl(node, node->condition ? VisitExpr(node->condition) : nullptr,
                                   node->command ? VisitExpr(node->command) : nullptr, Run(node->body, work),
                                   Run(node->alternative, work), work));
    }
    return out;
  }
};
// Scan only a leading/trailing command sequence. A command active on the
// boundary is a movement fence unless it is itself extracted. Commands proved
// absent there are retained in place and do not fence other boundary commands.
ControlForest ExtractBoundary(ControlForest& original, BoundarySpecializer& boundary,
                              BoundarySpecializer& rest, bool reverse, const VarPtr& iv, int64_t value,
                              PhysicalSyncRealization& work) {
  ControlForest flat;
  for (const auto& node : original) FlattenControlSequence(node, flat, work);
  if (reverse) std::reverse(flat.begin(), flat.end());
  ControlForest retained, moved;
  std::vector<ControlForest> groups;
  bool fence = false;
  for (const auto& node : flat) {
    if (!fence) {
      auto active = boundary.Run({node});
      if (active.empty()) {
        retained.push_back(node);
        continue;
      }
      if (rest.Run({node}).empty()) {
        auto bound = BindBoundary(iv, value).Run(active, work);
        groups.push_back(std::move(bound));
        continue;
      }
      fence = true;
    }
    retained.push_back(node);
  }
  if (reverse) {
    std::reverse(retained.begin(), retained.end());
    std::reverse(groups.begin(), groups.end());
  }
  for (const auto& group : groups) moved.insert(moved.end(), group.begin(), group.end());
  original = std::move(retained);
  return moved;
}
BoundarySchedule PlanBoundarySchedule(const ForStmtPtr& loop, const CutInventory& inventory,
                                      PhysicalSyncRealization& work) {
  if (loop->kind_ != ForKind::Sequential || !loop->iter_args_.empty() || !loop->return_vars_.empty() ||
      !loop->attrs_.empty() || loop->hardware_loop_policy_ != HardwareLoopPolicy::Auto)
    return {};
  const auto type = As<ScalarType>(loop->loop_var_->GetType());
  if (!type || type->dtype_ != DataType::INDEX) return {};
  const auto start = As<ConstInt>(loop->start_), stop = As<ConstInt>(loop->stop_),
             step = As<ConstInt>(loop->step_);
  // A singleton needs no loop-control specialization; retain it unchanged.
  if (!start || !stop || !step || step->value_ <= 0 || start->value_ >= stop->value_) return {};
  const __int128 count = (__int128(stop->value_) - start->value_ + step->value_ - 1) / step->value_;
  if (count < 2 || count > std::numeric_limits<int64_t>::max()) return {};
  const __int128 last = __int128(start->value_) + (count - 1) * step->value_;
  if (last + step->value_ > std::numeric_limits<int64_t>::max()) return {};
  BoundaryInput input(inventory, work);
  input.Visit(loop->body_);
  if (!input.valid || input.cuts.empty()) return {};
  // Repeated anchors cannot be independently edited through this cut inventory.
  if (std::set<const PhysicalCuts*>(input.cuts.begin(), input.cuts.end()).size() != input.cuts.size())
    return {};
  ++work.schedule_candidates;
  BoundarySchedule plan;
  const auto* first = input.cuts.front();
  const auto* final = input.cuts.back();
  plan.cuts[first] = {first->before, first->after};
  plan.cuts[final] = {final->before, final->after};
  BoundarySpecializer at_first(loop->loop_var_, start->value_, start->value_ + 1, work);
  BoundarySpecializer after_first(loop->loop_var_, start->value_ + step->value_,
                                  static_cast<int64_t>(last) + 1, work);
  BoundarySpecializer at_last(loop->loop_var_, static_cast<int64_t>(last), static_cast<int64_t>(last) + 1,
                              work);
  BoundarySpecializer before_last(loop->loop_var_, start->value_, static_cast<int64_t>(last), work);
  plan.entry = ExtractBoundary(plan.cuts[first].first, at_first, after_first, false, loop->loop_var_,
                               start->value_, work);
  plan.exit = ExtractBoundary(plan.cuts[final].second, at_last, before_last, true, loop->loop_var_,
                              static_cast<int64_t>(last), work);
  if (plan.entry.empty() && plan.exit.empty()) return {};
  ControlForest old = first->before, candidate = plan.cuts[first].first;
  old.insert(old.end(), final->after.begin(), final->after.end());
  candidate.insert(candidate.end(), plan.cuts[final].second.begin(), plan.cuts[final].second.end());
  candidate.insert(candidate.end(), plan.entry.begin(), plan.entry.end());
  candidate.insert(candidate.end(), plan.exit.begin(), plan.exit.end());
  const CaseCodeSize before(old), after(candidate);
  work.schedule_original_nodes += before.statements + before.expressions;
  work.schedule_candidate_nodes += after.statements + after.expressions;
  // Payload and loop IR are unchanged. Commands are neither duplicated nor
  // dropped; require strictly fewer static decisions and no expression growth.
  if (after.commands != before.commands || after.guards >= before.guards ||
      after.expressions > before.expressions) {
    ++work.schedule_size_refusals;
    return {};
  }
  ++work.scheduled_loops;
  work.schedule_boundary_commands += CaseCodeSize(plan.entry).commands + CaseCodeSize(plan.exit).commands;
  return plan;
}

ControlPtr CaseLeaf(const std::vector<CaseCommands>& cases, std::pair<int64_t, int64_t> domain) {
  const auto found =
      std::lower_bound(cases.begin(), cases.end(), domain.first,
                       [](const CaseCommands& c, int64_t first) { return c.blocks.second < first; });
  if (found == cases.end() || found->blocks.first > domain.first) return nullptr;
  INTERNAL_CHECK(found->blocks.second >= domain.second) << "Partial invocation schedule case";
  return found->body;
}
ExprPtr InvocationGroupGuard(const ExprPtr& block, std::pair<int64_t, int64_t> domain,
                             const Intervals& intervals, const Span& span) {
  ExprPtr result;
  for (const auto& interval : intervals) {
    auto guard = SyncInvocationGuard(block, domain, interval, span);
    if (!guard) return std::make_shared<ConstBool>(true, span);
    result = result ? std::make_shared<Or>(result, guard, DataType::BOOL, span) : guard;
  }
  INTERNAL_CHECK(result) << "Empty invocation schedule group";
  return result;
}
// Hashing only indexes potential equality. Exact descriptor comparison below
// preserves guards, frames, IDs and ordered cuts even under hash collisions.
size_t ScheduleHash(const ControlPtr& node, std::map<const CommandControl*, size_t>& cache) {
  if (!node) return 0;
  if (auto found = cache.find(node.get()); found != cache.end()) {
    return found->second;
  }
  size_t result = 1;
  auto mix = [&](size_t h) { result ^= h + size_t(0x9e3779b9) + (result << 6) + (result >> 2); };
  if (node->condition) mix(structural_hash(node->condition, false));
  if (node->command) mix(structural_hash(node->command, false));
  mix(node->body.size());
  for (const auto& child : node->body) mix(ScheduleHash(child, cache));
  mix(node->alternative.size());
  for (const auto& child : node->alternative) mix(ScheduleHash(child, cache));
  cache[node.get()] = result;
  return result;
}
// Partition launch predicates over complete supplied cases without enumerating
// block values. The production cost model owns range-addition accounting; this
// helper only provides exact case intervals and rejects interior case splits.
class InvocationCasePartition {
 public:
  struct Decision {
    Intervals yes;
  };

 private:
  ExprPtr block_;
  std::vector<int64_t> splits_;
  PhysicalSyncRealization& work_;
  Intervals all_;
  static Intervals Intersect(const Intervals& a, const Intervals& b) {
    Intervals out;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
      const auto lo = std::max(a[i].first, b[j].first), hi = std::min(a[i].second, b[j].second);
      if (lo <= hi) out.emplace_back(lo, hi);
      if (a[i].second < b[j].second)
        ++i;
      else
        ++j;
    }
    return out;
  }
  std::optional<int64_t> Split(__int128 value) {
    if (value <= splits_.front()) return 0;
    if (value >= splits_.back()) return static_cast<int64_t>(splits_.size() - 1);
    auto found = std::lower_bound(splits_.begin(), splits_.end(), static_cast<int64_t>(value));
    if (*found != value) return {};
    return found - splits_.begin();
  }
  std::optional<Decision> Condition(const ExprPtr& expression) {
    ++work_.invocation_schedule_visits;
    if (auto known = As<ConstBool>(expression)) return Decision{known->value_ ? all_ : Intervals{}};
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return {};
    if (As<And>(expression) || As<Or>(expression)) {
      auto a = Condition(binary->left_), b = Condition(binary->right_);
      if (!a || !b) return {};
      auto yes = As<And>(expression) ? Intersect(a->yes, b->yes)
                                     : SubtractIntervals(all_, Intersect(SubtractIntervals(all_, a->yes),
                                                                         SubtractIntervals(all_, b->yes)));
      return Decision{std::move(yes)};
    }
    const auto bound = As<ConstInt>(binary->right_);
    if (binary->left_ != block_ || !bound) return {};
    const bool inclusive = As<Le>(expression) || As<Gt>(expression);
    if (!inclusive && !As<Lt>(expression) && !As<Ge>(expression) && !As<Eq>(expression)) return {};
    auto split = Split(__int128(bound->value_) + inclusive);
    if (!split) return {};
    Intervals yes;
    const auto size = static_cast<int64_t>(splits_.size() - 1);
    if (As<Eq>(expression)) {
      const auto end = Split(__int128(bound->value_) + 1);
      if (!end) return {};
      if (*split < *end) yes.emplace_back(*split, *end - 1);
    } else if (As<Lt>(expression) || As<Le>(expression)) {
      if (*split) yes.emplace_back(0, *split - 1);
    } else if (*split < size) {
      yes.emplace_back(*split, size - 1);
    }
    return Decision{std::move(yes)};
  }

 public:
  InvocationCasePartition(ExprPtr block, const AssignedRegionSchedule& assigned,
                          PhysicalSyncRealization& work)
      : block_(std::move(block)), work_(work), all_{{0, static_cast<int64_t>(assigned.cases.size() - 1)}} {
    for (const auto& c : assigned.cases) splits_.push_back(c.input->block_bounds->first);
    splits_.push_back(assigned.cases.back().input->block_bounds->second + 1);
  }
  std::optional<Decision> Partition(const ExprPtr& expression) { return Condition(expression); }
  static Intervals Intersection(const Intervals& a, const Intervals& b) { return Intersect(a, b); }
  Intervals CaseIndices(const Intervals& ranges) {
    Intervals result;
    for (const auto& [lo, hi] : ranges) {
      const auto first = Split(lo), end = Split(__int128(hi) + 1);
      if (!first || !end) return {};
      if (*first < *end) result.emplace_back(*first, *end - 1);
    }
    return result;
  }
};
std::vector<InvocationScheduleGroup> GroupInvocationSchedules(
    const std::vector<PreparedSchedulePhase>& phases, const ExprPtr& block,
    const std::vector<std::pair<int64_t, int64_t>>& domains, PhysicalSyncRealization& work) {
  if (domains.size() < 2) return {};
  if (!block) return {};
  std::vector<std::vector<ControlPtr>> signatures(domains.size());
  std::map<const CommandControl*, size_t> hashes;
  for (const auto& phase : phases)
    for (const auto& inventory : phase.visits) {
      std::set<const PhysicalCuts*> seen;
      for (const auto& [anchor, cut] : inventory) {
        if (!seen.insert(cut.get()).second) continue;
        for (size_t c = 0; c < domains.size(); ++c) {
          const auto domain = domains[c];
          const auto before_leaf = CaseLeaf(cut->before_cases, domain);
          const auto after_leaf = CaseLeaf(cut->after_cases, domain);
          signatures[c].push_back(before_leaf);
          signatures[c].push_back(after_leaf);
        }
      }
    }
  std::vector<InvocationScheduleGroup> groups;
  std::map<size_t, std::vector<size_t>> buckets;
  for (size_t c = 0; c < domains.size(); ++c) {
    size_t hash = 0;
    try {
      for (const auto& node : signatures[c])
        hash ^= ScheduleHash(node, hashes) + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
    } catch (const StructuralComparisonUnsupported&) {
      return {};
    }
    size_t group = groups.size();
    for (const auto candidate : buckets[hash]) {
      const auto& other = signatures[groups[candidate].representative];
      bool equal = other.size() == signatures[c].size();
      for (size_t i = 0; equal && i < other.size(); ++i) {
        ++work.invocation_schedule_comparisons;
        equal = other[i] == signatures[c][i] ||
                (other[i] && signatures[c][i] && SameControl(other[i], signatures[c][i]));
      }
      if (equal) {
        group = candidate;
        break;
      }
    }
    if (group == groups.size()) {
      buckets[hash].push_back(group);
      groups.push_back({c, {}});
    }
    auto& intervals = groups[group].intervals;
    const auto domain = domains[c];
    if (!intervals.empty() && intervals.back().second + 1 == domain.first)
      intervals.back().second = domain.second;
    else
      intervals.push_back(domain);  // Never fill a hole in a group's union.
  }
  return groups;
}

CutInventory SelectInvocationSchedule(const CutInventory& inventory, std::pair<int64_t, int64_t> domain) {
  CutInventory selected;
  std::map<const PhysicalCuts*, std::shared_ptr<PhysicalCuts>> owners;
  for (const auto& [anchor, cut] : inventory) {
    auto& replacement = owners[cut.get()];
    if (!replacement) {
      replacement = std::make_shared<PhysicalCuts>();
      if (auto leaf = CaseLeaf(cut->before_cases, domain)) replacement->before = {leaf};
      if (auto leaf = CaseLeaf(cut->after_cases, domain)) replacement->after = {leaf};
    }
    selected[anchor] = replacement;
  }
  return selected;
}

// Candidate-local construction does not inflate the accepted output counters.
// All command lowering consumes the original assigned owner and maps; this is
// accounting isolation, not another allocation or dependence analysis.
void AccumulateScheduleWork(PhysicalSyncRealization& result, const PhysicalSyncRealization& local) {
  for (auto member : {&PhysicalSyncRealization::set_sites,
                      &PhysicalSyncRealization::wait_sites,
                      &PhysicalSyncRealization::barrier_sites,
                      &PhysicalSyncRealization::guard_sites,
                      &PhysicalSyncRealization::shared_guard_sites,
                      &PhysicalSyncRealization::shared_composed_guard_sites,
                      &PhysicalSyncRealization::simplified_guard_sites,
                      &PhysicalSyncRealization::endpoint_domain_entries,
                      &PhysicalSyncRealization::endpoint_predicates_constructed,
                      &PhysicalSyncRealization::normalized_endpoint_bounds,
                      &PhysicalSyncRealization::control_descriptors,
                      &PhysicalSyncRealization::invocation_scheduled_loops,
                      &PhysicalSyncRealization::invocation_schedule_groups,
                      &PhysicalSyncRealization::invocation_schedule_visits,
                      &PhysicalSyncRealization::invocation_schedule_comparisons,
                      &PhysicalSyncRealization::invocation_schedule_model_saving,
                      &PhysicalSyncRealization::cut_case_comparisons,
                      &PhysicalSyncRealization::case_sequence_visits,
                      &PhysicalSyncRealization::case_sequence_comparisons,
                      &PhysicalSyncRealization::coalesced_cut_cases,
                      &PhysicalSyncRealization::factored_case_cuts,
                      &PhysicalSyncRealization::shared_disjoint_case_cuts}) {
    result.*member += local.*member;
  }
}

// The catalogue has independent C/B/P bits: case versioning, endpoint-boundary
// splitting and periodic expansion. A required unroll remainder is part of P.
constexpr std::array<const char*, 8> kScheduleModes{"guarded",           "case",     "boundary",
                                                    "case_boundary",     "periodic", "case_periodic",
                                                    "boundary_periodic", "combined"};
std::vector<PreparedSchedulePhase> PrepareLoopSchedule(const RegularLoopSchedule& plan, unsigned mode,
                                                       const AssignedRegionSchedule& assigned,
                                                       const CutInventory& domains,
                                                       std::pair<int64_t, int64_t> blocks,
                                                       PhysicalSyncRealization& work) {
  std::vector<PreparedSchedulePhase> phases;
  auto prepare = [&](int64_t first, int64_t end, int64_t period) {
    if (first >= end) return;
    PreparedSchedulePhase prepared{first, end, period, period > 1, {}};
    for (int64_t residue = 0; residue < period; ++residue) {
      SchedulePhase phase{plan.loop, first, end, period,
                          period > 1 ? std::make_optional(residue) : std::nullopt};
      prepared.visits.push_back(
          LowerScheduleCuts(assigned, domains, plan.case_cuts, blocks, work, (mode & 6) ? &phase : nullptr));
    }
    phases.push_back(std::move(prepared));
  };
  const auto intervals = (mode & 2) ? plan.phases : std::vector<std::pair<int64_t, int64_t>>{{0, plan.count}};
  // A single longest interval receives P expansion. Other B intervals retain
  // dynamic IDs; no global LCM or Cartesian residue search is constructed.
  const auto longest = std::max_element(intervals.begin(), intervals.end(), [](auto a, auto b) {
    return a.second - a.first < b.second - b.first;
  });
  for (const auto& interval : intervals) {
    const auto [first, end] = interval;
    if ((mode & 4) && interval == *longest) {
      const auto complete = first + (end - first) / plan.period * plan.period;
      prepare(first, complete, plan.period);
      prepare(complete, end, 1);
    } else {
      prepare(first, end, 1);
    }
  }
  return phases;
}

CutInventory CopyScheduleCuts(const CutInventory& inventory) {
  CutInventory result;
  std::map<const PhysicalCuts*, std::shared_ptr<PhysicalCuts>> copies;
  for (const auto& [anchor, cut] : inventory) {
    auto& copy = copies[cut.get()];
    if (!copy) copy = std::make_shared<PhysicalCuts>(*cut);
    result[anchor] = copy;
  }
  return result;
}

// Reprove boundary movement for this complete version and phase. The first
// and final residue use different original-coordinate lattices. Conservative
// interval proofs may include unused gaps, never exclude executed coordinates.
void PrepareBoundaryMotion(const RegularLoopSchedule& loop, PreparedSchedulePhase& phase,
                           PhysicalSyncRealization& work) {
  if ((phase.end - phase.first) / phase.period < 2) return;
  BoundaryInput first_input(phase.visits.front(), work), last_input(phase.visits.back(), work);
  first_input.Visit(loop.loop->body_);
  last_input.Visit(loop.loop->body_);
  if (!first_input.valid || !last_input.valid || first_input.cuts.empty() || last_input.cuts.empty()) return;
  auto coordinate = [&](int64_t ordinal) {
    return static_cast<int64_t>(__int128(loop.start) + __int128(ordinal) * loop.step);
  };
  const auto first = coordinate(phase.first), final = coordinate(phase.end - 1);
  const auto first_last = coordinate(phase.end - phase.period);
  const auto last_first = coordinate(phase.first + phase.period - 1);
  const auto* a = first_input.cuts.front();
  const auto* b = last_input.cuts.back();
  auto before = a->before, after = b->after;
  BoundarySpecializer at_first(loop.loop->loop_var_, first, first + 1, work);
  BoundarySpecializer rest_first(loop.loop->loop_var_, coordinate(phase.first + phase.period), first_last + 1,
                                 work);
  BoundarySpecializer at_last(loop.loop->loop_var_, final, final + 1, work);
  BoundarySpecializer rest_last(loop.loop->loop_var_, last_first, final, work);
  auto entry = ExtractBoundary(before, at_first, rest_first, false, loop.loop->loop_var_, first, work);
  auto exit = ExtractBoundary(after, at_last, rest_last, true, loop.loop->loop_var_, final, work);
  if (entry.empty() && exit.empty()) return;
  ControlForest old = a->before, candidate = before;
  old.insert(old.end(), b->after.begin(), b->after.end());
  for (const auto* nodes : {&after, &entry, &exit})
    candidate.insert(candidate.end(), nodes->begin(), nodes->end());
  const CaseCodeSize old_size(old), new_size(candidate);
  if (old_size.commands != new_size.commands || new_size.guards >= old_size.guards ||
      new_size.expressions > old_size.expressions)
    return;
  // The inventories were copied before this function; no shared candidate is mutated.
  for (auto& [anchor, cut] : phase.visits.front())
    if (cut.get() == a) cut->before = before;
  for (auto& [anchor, cut] : phase.visits.back())
    if (cut.get() == b) cut->after = after;
  phase.entry = std::move(entry);
  phase.exit = std::move(exit);
}

// Adjacent after/before forests denote the same executable gap when no original
// statement separates them. Join only those gaps, before scalar materialization.
// Definitions, unsupported control, and loop headers are fences. The only loop
// boundary commands admitted here have already been proved movable by planning.
// Each nonempty forest is joined at most once; payloads and their order are untouched.
class CommandSeams {
  CutInventory& cuts_;
  PhysicalSyncRealization& work_;
  BoundarySchedules* boundaries_;
  PreparedLoops* loops_;
  BoundarySchedule* boundary_ = nullptr;
  ControlForest* pending_ = nullptr;
  std::map<const PhysicalCuts*, size_t> uses_;
  std::map<const ForStmt*, size_t> loop_uses_;

  void Join(ControlForest& before, const std::string& selected_loop = {}) {
    if (!pending_ || pending_->empty() || before.empty() || pending_ == &before) return;
    ControlForest joined = *pending_;
    joined.insert(joined.end(), before.begin(), before.end());
    const auto previous_shared = work_.shared_composed_guard_sites;
    auto shared = ShareCommandGuards(joined, work_, false);
    const CaseCodeSize original(joined), replacement(shared);
    // Charge inspections even on refusal. Crossing a selected-candidate
    // boundary additionally preserves its charged raw descriptor bound. Ordinary
    // gap sharing keeps its existing policy: arm sequence wrappers can grow
    // even when fewer statements will be materialized.
    work_.schedule_inspected_descriptors += original.statements + replacement.statements;
    work_.schedule_inspected_expressions += original.expressions + replacement.expressions;
    const bool within_budget = selected_loop.empty() || (replacement.expressions <= original.expressions &&
                                                         replacement.statements <= original.statements);
    if (replacement.commands == original.commands && replacement.guards < original.guards && within_budget) {
      if (!selected_loop.empty())
        work_.schedule_parent_gaps.push_back(
            {selected_loop,
             original.commands,
             {original.guards, original.expressions, original.statements},
             {replacement.guards, replacement.expressions, replacement.statements}});
      *pending_ = std::move(shared);
      before.clear();
    } else {
      work_.shared_composed_guard_sites = previous_shared;
    }
  }

 public:
  CommandSeams(CutInventory& cuts, PhysicalSyncRealization& work, BoundarySchedules* boundaries = nullptr,
               PreparedLoops* loops = nullptr)
      : cuts_(cuts), work_(work), boundaries_(boundaries), loops_(loops) {}
  void Run(const StmtPtr& body) {
    class Uses : public IRVisitor {
      CommandSeams& owner_;
      void VisitStmt_(const ForStmtPtr& loop) override {
        ++owner_.loop_uses_[loop.get()];
        IRVisitor::VisitStmt_(loop);
      }
      void VisitStmt_(const IfStmtPtr& branch) override {
        const auto first = owner_.cuts_.find(SingletonCall(branch->then_body_));
        const auto second =
            branch->else_body_ ? owner_.cuts_.find(SingletonCall(*branch->else_body_)) : owner_.cuts_.end();
        if (branch->return_vars_.empty() && first != owner_.cuts_.end() && second != owner_.cuts_.end() &&
            first->second == second->second) {
          // Initialization/accumulation alternatives are one wrapped payload
          // cut in the materializer, not two independently placed occurrences.
          ++owner_.uses_[first->second.get()];
          return;
        }
        IRVisitor::VisitStmt_(branch);
      }
      void VisitExpr_(const CallPtr& call) override {
        ++owner_.work_.schedule_body_visits;
        if (auto found = owner_.cuts_.find(call.get()); found != owner_.cuts_.end())
          ++owner_.uses_[found->second.get()];
        IRVisitor::VisitExpr_(call);
      }

     public:
      explicit Uses(CommandSeams& owner) : owner_(owner) {}
    };
    Uses(*this).VisitStmt(body);
    Visit(body);
  }
  void Visit(const StmtPtr& stmt) {
    ++work_.schedule_body_visits;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) Visit(child);
      return;
    }
    if (auto loop = As<ForStmt>(stmt)) {
      if (loops_) {
        auto found = loops_->find(loop.get());
        if (found != loops_->end()) {
          auto& plan = found->second;
          // This is a parent-gap composition, not candidate re-selection. With
          // one unversioned phase, its proved entry precedes the loop header
          // directly. Dispatch/multiple phases remain fences. A shared AST loop
          // cannot lose its entry at only one of its static occurrences.
          if (loop_uses_[loop.get()] == 1 && plan.groups.empty() && plan.versions.size() == 1 &&
              plan.versions.front().size() == 1)
            Join(plan.versions.front().front().entry, plan.receipt_label);
          // Do not mutate candidate internals or export its exit for sharing.
          pending_ = nullptr;
          return;
        }
      }
      auto found = boundaries_ ? boundaries_->find(loop.get()) : BoundarySchedules::iterator{};
      auto* plan = boundaries_ && found != boundaries_->end() ? &found->second : nullptr;
      if (plan) Join(plan->entry);
      pending_ = nullptr;
      auto* previous = boundary_;
      boundary_ = plan;
      Visit(loop->body_);
      boundary_ = previous;
      pending_ = plan ? &plan->exit : nullptr;
      return;
    }
    auto found = cuts_.find(SingletonCall(stmt));
    const auto branch = As<IfStmt>(stmt);
    if (branch) {
      found = cuts_.find(SingletonCall(branch->then_body_));
      const auto other = branch->else_body_ ? cuts_.find(SingletonCall(*branch->else_body_)) : cuts_.end();
      if (!branch->return_vars_.empty() || found == cuts_.end() || other == cuts_.end() ||
          found->second != other->second)
        found = cuts_.end();
    }
    if (found != cuts_.end() && uses_[found->second.get()] == 1) {
      auto& cut = *found->second;
      auto changed = boundary_ ? boundary_->cuts.find(&cut) : decltype(boundary_->cuts)::iterator{};
      auto* override = boundary_ && changed != boundary_->cuts.end() ? &changed->second : nullptr;
      Join(override ? override->first : cut.before);
      pending_ = override ? &override->second : &cut.after;
      return;
    }
    pending_ = nullptr;
    if (branch) {
      Visit(branch->then_body_);
      pending_ = nullptr;
      if (branch->else_body_) Visit(*branch->else_body_);
      pending_ = nullptr;
    }
  }
};

class ScheduleTreeSize : public IRVisitor {
 public:
  size_t nodes = 0, payloads = 0, commands = 0, expressions = 0, statements = 0;
  void VisitExpr(const ExprPtr& expression) override {
    ++nodes;
    ++expressions;
    IRVisitor::VisitExpr(expression);
  }
  void VisitStmt(const StmtPtr& statement) override {
    ++nodes;
    ++statements;
    IRVisitor::VisitStmt(statement);
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (call && !IsOp(call, "system.set_flag") && !IsOp(call, "system.wait_flag") &&
        !IsOp(call, "system.bar_m") && !IsOp(call, "system.bar_v"))
      ++payloads;
    else if (call)
      ++commands;
    IRVisitor::VisitStmt_(statement);
  }
};

// Costs are accumulated by interval range additions over invocation cases.
// Unsupported mixed predicates make automatic profitability unproved.
struct ScheduleCost {
  std::array<__int128, 6> value{};
  bool valid = true;
  __int128 Total() const { return std::accumulate(value.begin(), value.end(), __int128(0)); }
};
struct ScheduleCostProfile {
  std::vector<ScheduleCost> difference;
  bool valid = true;
  explicit ScheduleCostProfile(size_t cases) : difference(cases + 1) {}
  void Add(const ScheduleCost& cost, const Intervals& active) {
    valid &= cost.valid;
    for (const auto& [lo, hi] : active)
      for (size_t i = 0; i < cost.value.size(); ++i) {
        difference[lo].value[i] += cost.value[i];
        difference[hi + 1].value[i] -= cost.value[i];
      }
  }
};
class ScheduleCostModel {
  struct Range {
    __int128 lo, hi, stride;
  };
  const RegularLoopSchedule& loop_;
  ExprPtr block_;
  PhysicalSyncRealization& work_;
  std::unique_ptr<InvocationCasePartition> partition_;
  size_t cases_;
  int64_t first_ = 0, last_ = 0, stride_ = 1;
  std::pair<int64_t, int64_t> blocks_{};
  static __int128 Gcd(__int128 a, __int128 b) {
    while (b) {
      const auto next = a % b;
      a = b;
      b = next;
    }
    return a;
  }
  static __int128 Mod(__int128 value, __int128 divisor) {
    const auto r = value % divisor;
    return r < 0 ? r + divisor : r;
  }
  static std::optional<Range> Limit(Range value) {
    if (value.lo < std::numeric_limits<int64_t>::min() || value.hi > std::numeric_limits<int64_t>::max() ||
        value.stride > std::numeric_limits<int64_t>::max())
      return {};
    return value;
  }
  std::optional<Range> Evaluate(const ExprPtr& expression) {
    ++work_.schedule_inspected_expressions;
    if (auto v = As<ConstInt>(expression)) return Range{v->value_, v->value_, 0};
    if (auto v = As<ConstBool>(expression)) return Range{v->value_, v->value_, 0};
    if (expression == loop_.loop->loop_var_) {
      return Range{__int128(loop_.start) + __int128(first_) * loop_.step,
                   __int128(loop_.start) + __int128(last_) * loop_.step, __int128(stride_) * loop_.step};
    }
    if (partition_ && expression == block_)
      return Range{blocks_.first, blocks_.second, blocks_.first == blocks_.second ? 0 : 1};
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return {};
    const auto a = Evaluate(binary->left_), b = Evaluate(binary->right_);
    if (!a || !b) return {};
    const auto step = Gcd(a->stride, b->stride);
    if (As<ir::Add>(expression)) return Limit({a->lo + b->lo, a->hi + b->hi, step});
    if (As<Sub>(expression)) return Limit({a->lo - b->hi, a->hi - b->lo, step});
    if (As<Mul>(expression)) {
      if (b->lo != b->hi || b->lo < 0) return {};
      return Limit({a->lo * b->lo, a->hi * b->lo, a->stride * b->lo});
    }
    if (As<FloorMod>(expression) || As<FloorDiv>(expression)) {
      if (b->lo != b->hi || b->lo <= 0) return {};
      const auto divisor = b->lo;
      if (As<FloorMod>(expression)) {
        if (a->stride % divisor == 0) {
          const auto r = Mod(a->lo, divisor);
          return Range{r, r, 0};
        }
        return Range{0, divisor - 1, 1};
      }
      return Range{(a->lo - Mod(a->lo, divisor)) / divisor, (a->hi - Mod(a->hi, divisor)) / divisor,
                   a->stride % divisor == 0 ? a->stride / divisor : 1};
    }
    std::optional<bool> truth;
    if (As<Lt>(expression)) {
      if (a->hi < b->lo)
        truth = true;
      else if (a->lo >= b->hi)
        truth = false;
    }
    if (As<Le>(expression)) {
      if (a->hi <= b->lo)
        truth = true;
      else if (a->lo > b->hi)
        truth = false;
    }
    if (As<Gt>(expression)) {
      if (a->lo > b->hi)
        truth = true;
      else if (a->hi <= b->lo)
        truth = false;
    }
    if (As<Ge>(expression)) {
      if (a->lo >= b->hi)
        truth = true;
      else if (a->hi < b->lo)
        truth = false;
    }
    if (As<Eq>(expression)) {
      if (a->lo == a->hi && b->lo == b->hi)
        truth = a->lo == b->lo;
      else if (a->hi < b->lo || b->hi < a->lo)
        truth = false;
    }
    if (As<And>(expression)) {
      if (a->hi == 0 || b->hi == 0)
        truth = false;
      else if (a->lo != 0 && b->lo != 0)
        truth = true;
    }
    if (As<Or>(expression)) {
      if (a->lo != 0 || b->lo != 0)
        truth = true;
      else if (a->hi == 0 && b->hi == 0)
        truth = false;
    }
    return truth ? Range{*truth, *truth, 0} : std::optional<Range>{};
  }
  void Forest(const FrozenCommandForest& forest, ScheduleCostProfile& profile, __int128 weight,
              const Intervals& active) {
    if (active.empty()) return;
    for (const auto& command : forest.commands) Visit(command, profile, weight, active);
  }
  void Visit(const FrozenCommand& command, ScheduleCostProfile& profile, __int128 weight,
             const Intervals& active) {
    if (active.empty()) return;
    ++work_.schedule_inspected_descriptors;
    const auto& node = command.original;
    if (!node->condition) {
      if (command.body) Forest(*command.body, profile, weight, active);
      return;
    }
    Intervals yes;
    const auto condition = Evaluate(node->condition);
    if (condition && condition->lo == condition->hi) {
      if (condition->lo) yes = active;
    } else {
      const auto before = work_.invocation_schedule_visits;
      const auto partition = partition_ ? partition_->Partition(node->condition) : std::nullopt;
      work_.schedule_inspected_descriptors += work_.invocation_schedule_visits - before;
      if (!partition) {
        profile.valid = false;
        return;
      }
      yes = InvocationCasePartition::Intersection(active, partition->yes);
    }
    ScheduleCost local;
    for (size_t i = 0; i < command.scalar_cost.size(); ++i) local.value[i] = weight * command.scalar_cost[i];
    profile.Add(local, active);
    Forest(*command.body, profile, weight, yes);
    Forest(*command.alternative, profile, weight, SubtractIntervals(active, yes));
  }

 public:
  ScheduleCostModel(const RegularLoopSchedule& loop, const AssignedRegionSchedule& assigned,
                    std::pair<int64_t, int64_t> blocks, PhysicalSyncRealization& work)
      : loop_(loop), block_(assigned.block), work_(work), cases_(assigned.cases.size()), blocks_(blocks) {
    if (block_ && std::all_of(assigned.cases.begin(), assigned.cases.end(),
                              [](const auto& c) { return c.input->block_bounds.has_value(); }))
      partition_ = std::make_unique<InvocationCasePartition>(block_, assigned, work);
  }
  Intervals AllCases() const { return {{0, static_cast<int64_t>(cases_ - 1)}}; }
  Intervals CaseIndices(const Intervals& ranges) {
    return partition_ ? partition_->CaseIndices(ranges) : Intervals{};
  }
  void Add(const ControlForest& forest, int64_t first, int64_t last, int64_t stride, const Intervals& active,
           ScheduleCostProfile& profile) {
    if (first > last) return;
    first_ = first;
    last_ = last;
    stride_ = stride;
    const __int128 weight = (last - first) / stride + 1;
    Forest(*FreezeCommandForest(forest, work_), profile, weight, active);
  }
};

struct CandidateSchedule {
  PreparedLoopSchedule plan;
  PhysicalSyncRealization work;
  SyncScheduleCandidateReceipt receipt;
  std::vector<ScheduleCost> costs;
};

size_t SkeletonCopies(const RegularLoopSchedule& loop, unsigned mode) {
  if (!(mode & 6)) return 1;
  const auto intervals = (mode & 2) ? loop.phases : std::vector<std::pair<int64_t, int64_t>>{{0, loop.count}};
  int64_t longest = 0;
  for (const auto& [first, end] : intervals) longest = std::max(longest, end - first);
  return (mode & 4) ? intervals.size() - 1 + loop.period + (longest % loop.period != 0) : intervals.size();
}

CandidateSchedule PrepareCandidate(const RegularLoopSchedule& loop, unsigned mode,
                                   const AssignedRegionSchedule& assigned, const CutInventory& domains,
                                   std::pair<int64_t, int64_t> blocks, const SyncSchedulePolicy& policy) {
  CandidateSchedule candidate;
  candidate.plan.loop = loop;
  candidate.plan.identity = !(mode & 6);
  candidate.plan.mode = mode;
  auto phases = PrepareLoopSchedule(loop, mode, assigned, domains, blocks, candidate.work);
  if (mode & 1) {
    std::vector<std::pair<int64_t, int64_t>> domains;
    for (const auto& item : assigned.cases) domains.push_back(*item.input->block_bounds);
    candidate.plan.groups = GroupInvocationSchedules(phases, assigned.block, domains, candidate.work);
  }
  const auto versions = std::max<size_t>(1, candidate.plan.groups.size());
  for (size_t v = 0; v < versions; ++v) {
    auto selected = phases;
    for (auto& phase : selected) {
      for (auto& cuts : phase.visits) {
        cuts = (mode & 1) && !candidate.plan.groups.empty()
                   ? SelectInvocationSchedule(
                         cuts, *assigned.cases[candidate.plan.groups[v].representative].input->block_bounds)
                   : CopyScheduleCuts(cuts);
      }
      if (policy.boundary_motion) PrepareBoundaryMotion(loop, phase, candidate.work);
      for (auto& cuts : phase.visits) CommandSeams(cuts, candidate.work).Run(loop.loop->body_);
    }
    candidate.plan.versions.push_back(std::move(selected));
  }
  return candidate;
}

// This conservative expanded bound includes every payload operand occurrence,
// not just descriptors or operation sites. Three times the command tree covers
// sequence wrappers and first-use SSA bindings retained by the frozen scalar plan.
bool BoundCandidate(CandidateSchedule& candidate, const ScheduleTreeSize& body,
                    const AssignedRegionSchedule& assigned, std::pair<int64_t, int64_t> blocks) {
  __int128 nodes = 0, payloads = 0, commands = 0;
  auto add = [&](const ControlForest& forest) {
    const CaseCodeSize size(forest);
    candidate.work.schedule_inspected_descriptors += size.statements;
    candidate.work.schedule_inspected_expressions += size.expressions;
    nodes += 3 * (__int128(size.statements) + size.expressions) + 4;
    commands += size.commands;
  };
  for (const auto& version : candidate.plan.versions) {
    for (const auto& phase : version) {
      nodes += 32;  // Loop header, coordinate frame and enclosing sequence.
      add(phase.entry);
      add(phase.exit);
      for (const auto& inventory : phase.visits) {
        nodes += body.nodes + 8;  // Fresh offset binding and residue sequence.
        payloads += body.payloads;
        std::set<const PhysicalCuts*> seen;
        for (const auto& [anchor, cut] : inventory) {
          if (!seen.insert(cut.get()).second) continue;
          add(cut->before);
          add(cut->after);
        }
      }
    }
  }
  for (size_t i = 0; i + 1 < candidate.plan.groups.size(); ++i) {
    ScheduleTreeSize size;
    size.VisitExpr(InvocationGroupGuard(assigned.block, blocks, candidate.plan.groups[i].intervals,
                                        assigned.owner->span_));
    nodes += size.nodes + 4;
    candidate.work.schedule_inspected_expressions += size.expressions;
  }
  if (nodes > std::numeric_limits<size_t>::max() || payloads > std::numeric_limits<size_t>::max() ||
      commands > std::numeric_limits<size_t>::max())
    return false;
  candidate.receipt.control_node_bound = static_cast<size_t>(nodes);
  candidate.receipt.payload_sites = static_cast<size_t>(payloads);
  candidate.receipt.command_sites = static_cast<size_t>(commands);
  return true;
}

void MeasureCandidate(CandidateSchedule& candidate, const AssignedRegionSchedule& assigned,
                      std::pair<int64_t, int64_t> blocks, ScheduleCostModel& model) {
  const auto& loop = candidate.plan.loop;
  ScheduleCostProfile profile(assigned.cases.size());
  ControlForest dispatch;
  for (size_t i = candidate.plan.groups.size(); i > 1; --i) {
    const auto condition = InvocationGroupGuard(
        assigned.block, blocks, candidate.plan.groups[i - 2].intervals, assigned.owner->span_);
    const auto empty = ControlSequence({}, assigned.owner->span_, candidate.work);
    const auto alternative = ControlSequence(std::move(dispatch), assigned.owner->span_, candidate.work);
    dispatch = {ControlBranch(condition, empty, alternative, assigned.owner->span_, candidate.work)};
  }
  model.Add(dispatch, 0, 0, 1, model.AllCases(), profile);
  for (size_t version = 0; version < candidate.plan.versions.size(); ++version) {
    const auto active = candidate.plan.groups.empty()
                            ? model.AllCases()
                            : model.CaseIndices(candidate.plan.groups[version].intervals);
    if (active.empty()) {
      profile.valid = false;
      continue;
    }
    for (const auto& phase : candidate.plan.versions[version]) {
      ScheduleCost cost;
      const auto visits = (phase.end - phase.first) / phase.period;
      cost.value[4] += __int128(2) * visits + 2;
      cost.value[0] += __int128(phase.period - 1) * visits;
      cost.value[5] += __int128(phase.period - 1) * visits;
      profile.Add(cost, active);
      model.Add(phase.entry, phase.first, phase.first, 1, active, profile);
      model.Add(phase.exit, phase.end - 1, phase.end - 1, 1, active, profile);
      std::set<int64_t> boundaries{phase.first, phase.end};
      for (const auto& [first, end] : loop.phases) {
        if (phase.first < first && first < phase.end) boundaries.insert(first);
        if (phase.first < end && end < phase.end) boundaries.insert(end);
      }
      if (visits > 1) {
        boundaries.insert(phase.first + phase.period);
        boundaries.insert(phase.end - phase.period);
      }
      const std::vector<int64_t> edges(boundaries.begin(), boundaries.end());
      const auto stride = std::max<int64_t>(1, loop.period);
      for (size_t i = 1; i < edges.size(); ++i) {
        for (int64_t residue = 0; residue < stride; ++residue) {
          const auto first = edges[i - 1] + (residue - edges[i - 1] % stride + stride) % stride;
          if (first >= edges[i]) continue;
          const auto last = first + (edges[i] - first - 1) / stride * stride;
          const auto visit = (first - phase.first) % phase.period;
          std::set<const PhysicalCuts*> seen;
          for (const auto& [anchor, cut] : phase.visits[visit]) {
            if (!seen.insert(cut.get()).second) continue;
            model.Add(cut->before, first, last, stride, active, profile);
            model.Add(cut->after, first, last, stride, active, profile);
          }
        }
      }
    }
  }
  ScheduleCost total;
  total.valid = profile.valid;
  for (size_t c = 0; c < assigned.cases.size(); ++c) {
    std::array<int64_t, 6> row{};
    for (size_t i = 0; i < row.size(); ++i) {
      total.value[i] += profile.difference[c].value[i];
      if (total.value[i] > std::numeric_limits<int64_t>::max() || total.value[i] < 0)
        total.valid = false;
      else
        row[i] = static_cast<int64_t>(total.value[i]);
    }
    candidate.receipt.model_cost.push_back(row);
    candidate.costs.push_back(total);
  }
  candidate.receipt.model_exact = std::all_of(candidate.costs.begin(), candidate.costs.end(),
                                              [](const auto& cost) { return cost.valid; });
}

bool Comparable(const CandidateSchedule& a, const CandidateSchedule& b) {
  if (a.costs.size() != b.costs.size()) return false;
  for (size_t i = 0; i < a.costs.size(); ++i) {
    if (!a.costs[i].valid || !b.costs[i].valid) return false;
  }
  return true;
}
bool Improves(const CandidateSchedule& candidate, const CandidateSchedule& baseline) {
  if (!Comparable(candidate, baseline)) return false;
  bool better = false;
  for (size_t i = 0; i < candidate.costs.size(); ++i) {
    if (candidate.costs[i].Total() > baseline.costs[i].Total()) return false;
    better |= candidate.costs[i].Total() < baseline.costs[i].Total();
  }
  return better;
}

class ScheduleLoopOrder : public IRVisitor {
 public:
  std::vector<const ForStmt*> loops;
  void VisitStmt_(const ForStmtPtr& loop) override {
    loops.push_back(loop.get());
    IRVisitor::VisitStmt_(loop);
  }
};
unsigned TransformCount(unsigned mode) { return (mode & 1) + ((mode >> 1) & 1) + ((mode >> 2) & 1); }
PreparedLoops PlanLoopSchedules(const AssignedRegionSchedule& assigned, const CutInventory& domains,
                                const RegularSchedules& inventory, std::pair<int64_t, int64_t> blocks,
                                PhysicalSyncRealization& work, const SyncSchedulePolicy& policy) {
  const auto requested = std::find(kScheduleModes.begin(), kScheduleModes.end(), policy.mode);
  CHECK(policy.mode == "auto" || requested != kScheduleModes.end())
      << "Unknown synchronization schedule mode";
  PreparedLoops selected;
  size_t payload_debit = 0, control_debit = 0;
  ScheduleLoopOrder order;
  order.VisitStmt(assigned.owner->body_);
  for (size_t ordinal = 0; ordinal < order.loops.size(); ++ordinal) {
    const auto* key = order.loops[ordinal];
    const auto found = inventory.find(key);
    if (found == inventory.end()) continue;
    const auto& loop = found->second;
    const auto label = loop.loop->loop_var_->name_hint_ + "#" + std::to_string(ordinal);
    ScheduleTreeSize original, body;
    original.VisitStmt(loop.loop);
    body.VisitStmt(loop.loop->body_);
    work.schedule_inspected_descriptors += original.statements + body.statements;
    work.schedule_inspected_expressions += original.expressions + body.expressions;
    ScheduleCostModel model(loop, assigned, blocks, work);
    std::vector<CandidateSchedule> candidates;
    candidates.reserve(kScheduleModes.size());
    candidates.push_back(PrepareCandidate(loop, 0, assigned, domains, blocks, policy));
    auto& baseline = candidates.front();
    baseline.receipt = {label, kScheduleModes[0], "eligible", ""};
    INTERNAL_CHECK(BoundCandidate(baseline, body, assigned, blocks)) << "Guarded schedule size overflow";
    MeasureCandidate(baseline, assigned, blocks, model);
    baseline.receipt.model_comparable = Comparable(baseline, baseline);
    size_t chosen = 0;
    int64_t longest_boundary = 0;
    for (const auto& [first, end] : loop.phases) longest_boundary = std::max(longest_boundary, end - first);
    for (unsigned mode = 1; mode < kScheduleModes.size(); ++mode) {
      if (policy.mode != "auto" && policy.mode != kScheduleModes[mode]) continue;
      CandidateSchedule candidate;
      candidate.receipt = {label, kScheduleModes[mode], "ineligible", ""};
      if ((mode & 1) && (assigned.cases.size() < 2 || !assigned.block))
        candidate.receipt.reason = "case versioning needs multiple cases and a dominating launch binding";
      else if ((mode & 2) && loop.phases.size() < 2)
        candidate.receipt.reason = "no internal endpoint boundary";
      else if ((mode & 4) && (loop.period <= 1 || ((mode & 2) ? longest_boundary : loop.count) < loop.period))
        candidate.receipt.reason = "no supported common nonunit period with a complete group";
      if (!candidate.receipt.reason.empty()) {
        candidates.push_back(std::move(candidate));
        continue;
      }
      const __int128 minimum_payloads =
          __int128(SkeletonCopies(loop, mode)) * ((mode & 1) ? 2 : 1) * body.payloads;
      const auto payload_limit = __int128(original.payloads) +
                                 (policy.no_clone ? 0 : policy.max_extra_payload_sites - payload_debit);
      if (minimum_payloads > payload_limit) {
        candidate.receipt.status = "over_budget";
        candidate.receipt.reason = "payload-site lower bound exceeds remaining budget";
        candidates.push_back(std::move(candidate));
        continue;
      }
      candidate = PrepareCandidate(loop, mode, assigned, domains, blocks, policy);
      candidate.receipt = {label, kScheduleModes[mode], "eligible", ""};
      if ((mode & 1) && candidate.plan.groups.size() < 2) {
        candidate.receipt.status = "ineligible";
        candidate.receipt.reason = "fewer than two distinct complete-case schedules";
      } else if (!BoundCandidate(candidate, body, assigned, blocks) ||
                 candidate.receipt.payload_sites > payload_limit ||
                 __int128(candidate.receipt.control_node_bound) >
                     __int128(policy.max_extra_control_nodes) - control_debit) {
        candidate.receipt.status = "over_budget";
        candidate.receipt.reason = "expanded emitted-site bound exceeds remaining budget";
      } else {
        MeasureCandidate(candidate, assigned, blocks, model);
        candidate.receipt.model_comparable = Comparable(candidate, candidates.front());
        const bool profitable = Improves(candidate, candidates.front());
        if (policy.mode != "auto")
          chosen = candidates.size();
        else if (profitable &&
                 (chosen == 0 || Improves(candidate, candidates[chosen]) ||
                  (Comparable(candidate, candidates[chosen]) &&
                   std::equal(candidate.costs.begin(), candidate.costs.end(),
                              candidates[chosen].costs.begin(),
                              [](const auto& a, const auto& b) { return a.Total() == b.Total(); }) &&
                   std::make_tuple(candidate.receipt.payload_sites, candidate.receipt.control_node_bound,
                                   TransformCount(mode)) <
                       std::make_tuple(candidates[chosen].receipt.payload_sites,
                                       candidates[chosen].receipt.control_node_bound,
                                       TransformCount(candidates[chosen].plan.mode)))))
          chosen = candidates.size();
        if (!profitable) {
          candidate.receipt.status = candidate.receipt.model_comparable ? "not_profitable" : "cost_unproved";
          candidate.receipt.reason = candidate.receipt.model_comparable ? "no known per-case improvement"
                                                                        : "mixed control cost is unproved";
        }
      }
      candidates.push_back(std::move(candidate));
    }
    auto& accepted = candidates[chosen];
    accepted.receipt.selected = true;
    accepted.receipt.status = policy.mode == "auto" || chosen == 0 ? "selected" : "selected_forced";
    if (policy.mode != "auto" && accepted.receipt.mode != policy.mode) {
      accepted.receipt.status = "fallback";
      accepted.receipt.reason = "guarded fallback; requested mode was not admissible";
    }
    if (chosen) {
      payload_debit += accepted.receipt.payload_sites > original.payloads
                           ? accepted.receipt.payload_sites - original.payloads
                           : 0;
      // Zero is a proved baseline lower bound even if source wrappers flatten.
      // Debiting the full replacement upper bound is stricter than extra growth.
      control_debit += accepted.receipt.control_node_bound;
    }
    for (const auto& candidate : candidates) {
      work.schedule_inspected_descriptors +=
          candidate.work.schedule_inspected_descriptors + candidate.work.control_descriptors +
          candidate.work.schedule_descriptor_visits + candidate.work.schedule_body_visits +
          candidate.work.case_sequence_visits + candidate.work.case_sequence_comparisons +
          candidate.work.cut_case_comparisons + candidate.work.invocation_schedule_visits +
          candidate.work.invocation_schedule_comparisons;
      work.schedule_inspected_expressions +=
          candidate.work.schedule_inspected_expressions + candidate.work.control_expression_visits;
      work.schedule_choices.push_back(candidate.receipt);
    }
    AccumulateScheduleWork(work, accepted.work);
    if (accepted.plan.mode & 6) ++work.regular_schedule_loops;
    if (accepted.plan.mode & 1) {
      ++work.invocation_scheduled_loops;
      work.invocation_schedule_groups += accepted.plan.groups.size();
    }
    if (chosen && accepted.receipt.model_comparable) {
      __int128 saving = std::numeric_limits<int64_t>::max();
      for (size_t c = 0; c < accepted.costs.size(); ++c)
        saving = std::min(saving, candidates.front().costs[c].Total() - accepted.costs[c].Total());
      if (saving > 0 && saving <= std::numeric_limits<size_t>::max() - work.regular_schedule_model_saving)
        work.regular_schedule_model_saving += static_cast<size_t>(saving);
    }
    accepted.plan.receipt_label = label;
    selected.emplace(key, std::move(accepted.plan));
  }
  return selected;
}

// Boundary motion is also selected before materialization. Unsupported outer
// control retains the same conservative rule; no emitter discovers a rewrite.
class BoundaryScheduleCollector : public IRVisitor {
  const CutInventory& cuts_;
  const PreparedLoops& loops_;
  PhysicalSyncRealization& work_;
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loops_.count(loop.get())) return;
    auto plan = PlanBoundarySchedule(loop, cuts_, work_);
    if (!plan.entry.empty() || !plan.exit.empty()) plans.emplace(loop.get(), std::move(plan));
    IRVisitor::VisitStmt_(loop);
  }

 public:
  BoundarySchedules plans;
  BoundaryScheduleCollector(const CutInventory& cuts, const PreparedLoops& loops,
                            PhysicalSyncRealization& work)
      : cuts_(cuts), loops_(loops), work_(work) {}
};

FunctionPtr CompileAssignedSyncSchedule(const FunctionPtr& owner, const std::vector<AssignedSyncCase>& cases,
                                        std::pair<int64_t, int64_t> blocks,
                                        const SyncSchedulePolicy& policy) {
  AssignedRegionSchedule schedule(owner, cases);
  auto& work = *schedule.cases.front().work;
  CutInventory domains;
  AbsentLoops absent;
  for (const auto& c : schedule.cases)
    if (c.input->region_tree)
      RetainAbsentLoops(*c.input->region_tree, c.input->block_bounds.value_or(blocks), absent);
  for (const auto& cut : schedule.cuts) {
    auto& facts = domains[cut.original.get()];
    if (!facts) facts = std::make_shared<PhysicalCuts>();
    if (cut.continuation) domains[cut.continuation.get()] = facts;
  }
  CutDomains(schedule.owner, absent, domains, blocks, work).VisitStmt(schedule.owner->body_);
  const auto inventory = CollectRegularLoops(schedule, work);
  auto plans = PlanLoopSchedules(schedule, domains, inventory, blocks, work, policy);
  std::set<const Call*> specialized;
  for (const auto& [loop, plan] : plans)
    specialized.insert(plan.loop.anchors.begin(), plan.loop.anchors.end());
  std::vector<std::vector<size_t>> selected(schedule.cases.size());
  for (size_t i = 0; i < schedule.cuts.size(); ++i)
    if (!specialized.count(schedule.cuts[i].original.get()))
      selected[schedule.cuts[i].case_index].push_back(i);
  auto cuts = LowerScheduleCuts(schedule, domains, selected, blocks, work);
  SyncTokenNames names(schedule.owner);
  ++work.schedule_name_scans;
  work.schedule_name_visits += names.visits;
  BoundaryScheduleCollector boundary(cuts, plans, work);
  if (policy.boundary_motion) boundary.VisitStmt(schedule.owner->body_);
  CommandSeams(cuts, work, &boundary.plans, &plans).Run(schedule.owner->body_);
  auto function = MaterializeSelectedSchedule(schedule.owner, schedule.block, cuts, plans, boundary.plans,
                                              blocks, work, names);
  FinishSyncScheduleReceipt(function, work, policy);
  return function;
}

}  // namespace physical

using namespace physical;
void FinishSyncScheduleReceipt(const FunctionPtr& function, PhysicalSyncRealization& work,
                               const SyncSchedulePolicy& policy) {
  CHECK(policy.mode == "auto" ||
        std::find(kScheduleModes.begin(), kScheduleModes.end(), policy.mode) != kScheduleModes.end())
      << "Unknown synchronization schedule mode";
  if (work.schedule_choices.empty()) {
    if (policy.mode != "auto" && policy.mode != "guarded")
      work.schedule_choices.push_back(
          {"<function>", policy.mode, "ineligible", "no qualified repeated loop"});
    SyncScheduleCandidateReceipt identity{
        "<function>", "guarded", policy.mode == "auto" || policy.mode == "guarded" ? "selected" : "fallback",
        "no qualified repeated loop"};
    identity.selected = true;
    work.schedule_choices.push_back(std::move(identity));
  }
  ScheduleTreeSize measured;
  measured.VisitFunction(function);
  work.schedule_emitted_payload_sites = measured.payloads;
  work.schedule_emitted_control_nodes = measured.nodes;
  for (auto& receipt : work.schedule_choices) {
    if (receipt.loop != "<function>" || !receipt.selected) continue;
    receipt.payload_sites = measured.payloads;
    receipt.control_node_bound = measured.nodes;
    receipt.command_sites = measured.commands;
  }
}
std::pair<std::vector<std::vector<std::pair<int64_t, int64_t>>>, PhysicalSyncRealization>
InspectCommandGroups(const FunctionPtr& owner, const std::vector<CaseCommands>& supplied,
                     PhysicalSyncRealization work) {
  CHECK(!supplied.empty()) << "Expected nonempty schedule cases";
  std::vector<std::pair<int64_t, int64_t>> domains;
  auto cut = std::make_shared<PhysicalCuts>();
  const auto blocks = std::make_pair(supplied.front().blocks.first, supplied.back().blocks.second);
  int64_t next = blocks.first;
  for (size_t c = 0; c < supplied.size(); ++c) {
    CHECK(supplied[c].blocks.first == next && supplied[c].blocks.second >= next &&
          supplied[c].blocks.second < std::numeric_limits<int64_t>::max())
        << "Non-exhaustive cases";
    next = supplied[c].blocks.second + 1;
    domains.push_back(supplied[c].blocks);
    cut->before_cases.push_back({supplied[c].blocks, supplied[c].body});
  }
  const std::vector<PreparedSchedulePhase> phases{{0, 1, 1, false, {CutInventory{{nullptr, cut}}}}};
  const auto grouped = GroupInvocationSchedules(phases, DominatingSyncBlock(owner), domains, work);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> result;
  for (const auto& group : grouped) result.push_back(group.intervals);
  return {std::move(result), std::move(work)};
}
PhysicalSyncRealization EmitCommandBoundaries(
    const FunctionPtr& owner, const std::vector<std::tuple<CallPtr, ControlForest, ControlForest>>& supplied,
    bool enabled, bool validate_scope, PhysicalSyncRealization work) {
  CutInventory inventory;
  for (const auto& [call, before, after] : supplied) {
    auto cut = std::make_shared<PhysicalCuts>();
    cut->before = before;
    cut->after = after;
    CHECK(inventory.emplace(call.get(), std::move(cut)).second) << "Duplicate diagnostic command cut";
  }
  if (validate_scope) {
    const AbsentLoops absent;
    CutDomains(owner, absent, inventory, {0, 0}, work).VisitStmt(owner->body_);
    for (const auto& [call, cut] : inventory) {
      CHECK(cut->predicate_scope) << "Diagnostic cut has no original owner scope";
      RequireCommandScope(cut->before, *cut->predicate_scope, work);
      RequireCommandScope(cut->after, *cut->predicate_scope, work);
    }
  }
  SyncTokenNames names(owner);
  ++work.schedule_name_scans;
  work.schedule_name_visits += names.visits;
  PreparedLoops loops;
  BoundaryScheduleCollector boundary(inventory, loops, work);
  if (enabled) {
    boundary.VisitStmt(owner->body_);
    CommandSeams(inventory, work, &boundary.plans, &loops).Run(owner->body_);
  }
  work.function =
      MaterializeSelectedSchedule(owner, nullptr, inventory, loops, boundary.plans, {}, work, names);
  return work;
}

}  // namespace pypto::ir::sync
